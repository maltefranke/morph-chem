"""Convert LoQI's processed ChEMBL3D release to Morph's LMDB format.

Source: https://github.com/isayevlab/LoQI (Release 2, Figshare 31441570).
The supplied train/val/test partitions are retained. Atom labels in ``x``
are decoded using LoQI's FULL_ATOM_ENCODER; auxiliary stereo edges 5--8
are excluded from chemical topology. Hydrogens, charges and coordinates
in Angstroms are retained, with no GEOM-specific size or element filter.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from functools import partial
from pathlib import Path
from urllib.request import urlopen

import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data.separate import separate
from tqdm import tqdm

from morph.dataset.core.conversion import mol_to_data
from morph.dataset.core.lmdb import (
    HYDROGEN_MODES,
    MolecularLMDBDataset,
)
from morph.dataset.core.tokenized_decode import TokenizedMolecularDatasetMixin
from morph.utils.rdkit_utils import (
    IDX_BOND_MAP,
    sanitize_mol_correctly,
)

# Pinned to the processed release linked by LoQI, version 1. Only graph
# splits are needed; checkpoints and LoQI's statistics are not downloaded.
RELEASE_FILES = {
    "train": (62280814, 6761964617, "391ddd92375ed908527ccfe2e12b83a0"),
    "val": (62280805, 845871535, "dc2bd573245520a2bcbc709c252b72cb"),
    "test": (62280802, 845962054, "7eacc9247a19b6c0865609d4b0e28364"),
}
ATOM_NUMBERS = torch.tensor(
    [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83, 34],
    dtype=torch.long,
)
PHYSICAL_EDGE_LABELS = tuple(label for label in IDX_BOND_MAP if label != 0)
STEREO_EDGE_LABELS = (5, 6, 7, 8)  # LoQI auxiliary stereo constraints.
GRAPH_FIELDS = ("x", "pos", "charges", "edge_index", "edge_attr")


def _download_split(split: str, destination: Path) -> None:
    """Download and verify a split before publishing it in the raw cache."""
    file_id, size, checksum = RELEASE_FILES[split]
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://ndownloader.figshare.com/files/{file_id}"
    digest = hashlib.md5()  # Release-provided integrity checksum.
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        temporary = Path(staging) / destination.name
        with (
            urlopen(url, timeout=120) as response,
            temporary.open("wb") as output,
            tqdm(total=size, unit="B", unit_scale=True, desc=destination.name) as bar,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                bar.update(len(chunk))
        if temporary.stat().st_size != size or digest.hexdigest() != checksum:
            raise ValueError(f"Size/checksum mismatch downloading {destination.name}")
        os.replace(temporary, destination)


def _load_split(path: Path) -> tuple[Data, dict | None, int]:
    """Load collated PyG tensors without the per-record InMemoryDataset cache.

    Tensor storage is memory mapped. PyTorch still unpickles the source's
    Python metadata (including RDKit molecules); discard it immediately.
    Only trusted LoQI release files should be passed to this loader.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, tuple) or len(payload) not in (2, 3):
        raise ValueError(f"Expected a collated PyG dataset in {path}")
    source, slices = payload[:2]
    fields = {key: source[key] for key in GRAPH_FIELDS}
    data = Data(**fields)
    if slices is None:
        return data, None, 1
    slices = {key: slices[key] for key in GRAPH_FIELDS}
    length = len(slices["x"]) - 1
    if any(len(value) != length + 1 for value in slices.values()):
        raise ValueError(f"Inconsistent graph slices in {path}")
    return data, slices, length


def convert_record(
    record: Data,
    *,
    atom_numbers: torch.Tensor = ATOM_NUMBERS,
    stereo_edge_labels: tuple[int, ...] = STEREO_EDGE_LABELS,
    hydrogen_mode: str = "explicit",
    max_atoms: int | None = None,
    allowed_atom_symbols: tuple[str, ...] | None = None,
) -> Data | None:
    """Decode a LoQI graph, retaining only valid physical molecular topology.

    Malformed encodings raise an error; invalid chemistry/non-finite geometry
    is skipped. Rebuilding from the physical graph avoids dependence on the
    optional pickled RDKit molecule and its conformer/stereochemistry state.
    """
    if hydrogen_mode not in HYDROGEN_MODES:
        raise ValueError(f"hydrogen_mode must be one of {HYDROGEN_MODES}")
    atom_types = record.x.long()
    n_atoms = atom_types.numel()
    if atom_types.shape != (n_atoms,) or n_atoms == 0:
        raise ValueError("Expected a nonempty one-dimensional LoQI atom label array")
    if (atom_types < 0).any() or (atom_types >= len(atom_numbers)).any():
        raise ValueError("Unknown LoQI atom label")
    if record.pos.shape != (n_atoms, 3) or record.charges.shape != (n_atoms,):
        raise ValueError("Coordinates/charges do not match the atom count")
    if not torch.isfinite(record.pos).all():
        return None
    edges = record.edge_index.long()
    labels = record.edge_attr.long()
    if edges.shape != (2, labels.numel()) or labels.ndim != 1:
        raise ValueError("Expected edge_index [2, E] and edge_attr [E]")
    physical = torch.isin(labels, labels.new_tensor(PHYSICAL_EDGE_LABELS))
    stereo = torch.isin(labels, labels.new_tensor(stereo_edge_labels))
    if not (physical | stereo).all():
        raise ValueError("Unknown LoQI edge label")
    if (edges < 0).any() or (edges >= n_atoms).any():
        raise ValueError("Edge index outside the molecule")

    edges, labels = edges[:, physical], labels[physical]
    topology = dict(zip(map(tuple, edges.t().tolist()), labels.tolist(), strict=True))
    if len(topology) != labels.numel() or any(
        i == j or topology.get((j, i)) != label for (i, j), label in topology.items()
    ):
        raise ValueError(
            "Physical bonds must be unique, symmetric and without self edges"
        )

    z = atom_numbers[atom_types]
    charges = record.charges.long().clone()
    # A slice shares storage with the entire split. Clone even when dtype is
    # unchanged, or pickle would serialize that whole storage per molecule.
    pos = record.pos.float().clone()
    editable = Chem.RWMol()
    for atomic_number, charge in zip(z.tolist(), charges.tolist(), strict=True):
        atom = Chem.Atom(atomic_number)
        atom.SetFormalCharge(charge)
        atom.SetNoImplicit(True)
        editable.AddAtom(atom)
    for (i, j), label in topology.items():
        if i < j:
            editable.AddBond(i, j, IDX_BOND_MAP[label])
    mol = editable.GetMol()
    mol = sanitize_mol_correctly(mol)
    if mol is None:
        return None
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
        conformer = Chem.Conformer(n_atoms)
        conformer.SetPositions(pos.double().numpy())
        mol.AddConformer(conformer)
        Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    except (ValueError, RuntimeError):
        return None
    return mol_to_data(
        mol,
        hydrogen_mode=hydrogen_mode,
        max_atoms=max_atoms,
        allowed_atom_symbols=allowed_atom_symbols,
    )


def _iter_records(data: Data, slices: dict | None, length: int):
    for index in range(length):
        if slices is None:
            yield data.clone()
        else:
            # Cloning also keeps multiprocessing from sharing the entire split
            # tensor storage when a single graph is sent to a worker.
            yield separate(Data, data, index, slices, decrement=False).clone()


class ChEMBL3D(MolecularLMDBDataset):
    """Read local LoQI release splits or download missing files into raw/."""

    def __init__(self, root, source_dir: str | None = None, **options):
        self.source_dir = source_dir
        super().__init__(root, **options)

    def prepare(self) -> None:
        """Locate or download the requested release split, then convert it."""
        self.download()
        self.process()

    def _source_path(self) -> Path | None:
        base = Path(self.source_dir) if self.source_dir is not None else Path(self.root)
        for directory in (
            base,
            base / "raw",
            base / "processed",
            base / "chembl3d_stereo" / "processed",
        ):
            path = directory / f"{self.split}_h.pt"
            if path.is_file():
                return path
        return None

    def download(self) -> None:
        """Reuse supplied files, otherwise download the requested release split."""
        if self._source_path() is not None:
            return
        if self.source_dir is not None:
            raise FileNotFoundError(
                f"{self.split}_h.pt not found under source_dir={self.source_dir!r}"
            )
        _download_split(self.split, Path(self.raw_dir) / f"{self.split}_h.pt")

    def process(self) -> None:
        """Stream the selected split into a new LMDB; never modify source files."""
        source = self._source_path()
        if source is None:
            raise FileNotFoundError(f"Missing source for {self.split}")
        if os.path.exists(self.lmdb_path):
            raise FileExistsError(self.lmdb_path)
        print(f"Loading {source} (tensor mmap; Python metadata is loaded into RAM)")
        data, slices, length = _load_split(source)
        self._write_records(
            _iter_records(data, slices, length),
            partial(convert_record, **self.processing_options),
            total=length,
        )


class FlowMatchingChEMBL3DDataset(TokenizedMolecularDatasetMixin, ChEMBL3D):
    """Decode ChEMBL3D records into tokenized flow-matching targets."""
