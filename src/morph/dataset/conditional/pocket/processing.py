"""CrossDocked source download, pocket parsing, and DiffDec ligand processing.

Adapted from morph at f81996ace95ee45c2e1d23332252d38137b3a918. In particular,
do not run the GEOM validity screen: it discards otherwise usable CrossDocked
ligands. Heavy-atom mode removes *all* hydrogens, including isotopic H.
"""

from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from morph.utils.rdkit_utils import BOND_IDX_MAP

RAW_TREE = "crossdocked_pocket10"
RAW_ARCHIVE = f"{RAW_TREE}.tar.gz"
RAW_SPLIT = "split_by_name.pt"
CROSSDOCKED_GDRIVE_IDS = {
    RAW_ARCHIVE: "10KGuj15mxOJ2FBsduun2Lggzx0yPreEU",
    RAW_SPLIT: "1mycOKpphVBQjxEbpn1AwdpQs8tNVbxKY",
}
BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def _download_raw(name: str, destination: Path) -> None:
    try:
        import gdown  # noqa: PLC0415 (optional download dependency)
    except ImportError as exc:
        raise ImportError(
            "CrossDocked download requires `uv sync --extra crossdocked`; "
            "alternatively provide a local --source-dir."
        ) from exc
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        temporary = Path(staging) / name
        result = gdown.download(
            id=CROSSDOCKED_GDRIVE_IDS[name], output=str(temporary), quiet=False
        )
        if result is None or not temporary.is_file() or temporary.stat().st_size == 0:
            raise OSError(f"Failed to download {name}")
        os.replace(temporary, destination)


def ensure_crossdocked_raw(
    raw_dir: str | Path, source_dir: str | Path | None = None
) -> tuple[Path, Path]:
    """Find the pocket tree and official split, downloading only by default.

    A source directory may contain the tree (or archive) and split directly, or
    use morph's older layout with the tree beside a ``raw/`` directory. An
    explicit source, including MORPH_CROSSDOCKED_RAW, is never downloaded into.
    Archive extraction always goes into this dataset's writable raw cache.
    """
    source_dir = source_dir or os.environ.get("MORPH_CROSSDOCKED_RAW")
    raw = Path(raw_dir)
    source = Path(source_dir) if source_dir is not None else raw
    directories = [source, source / "raw"]
    if source.name in (RAW_TREE, "raw"):
        directories.append(source.parent)

    def locate(name):
        return next((d / name for d in directories if (d / name).exists()), None)

    tree, archive, split = locate(RAW_TREE), locate(RAW_ARCHIVE), locate(RAW_SPLIT)
    if split is None or (tree is None and archive is None):
        if source_dir is not None:
            raise FileNotFoundError(
                f"Expected {RAW_TREE}/ (or {RAW_ARCHIVE}) and {RAW_SPLIT} "
                f"under source_dir={source}"
            )
        raw.mkdir(parents=True, exist_ok=True)
        for name in CROSSDOCKED_GDRIVE_IDS:
            if name == RAW_ARCHIVE and tree is not None:
                continue
            if not (raw / name).is_file():
                _download_raw(name, raw / name)
        archive, split = raw / RAW_ARCHIVE, raw / RAW_SPLIT

    if tree is None:
        raw.mkdir(parents=True, exist_ok=True)
        tree = raw / RAW_TREE
        if not tree.is_dir():
            with tempfile.TemporaryDirectory(dir=raw) as staging:
                with tarfile.open(archive) as stream:
                    stream.extractall(staging, filter="data")
                extracted = Path(staging) / RAW_TREE
                if not extracted.is_dir():
                    raise ValueError(f"{archive} does not contain {RAW_TREE}/")
                os.replace(extracted, tree)
    if not tree.is_dir() or not split.is_file():
        raise FileNotFoundError(f"Incomplete CrossDocked sources under {source}")
    return tree, split


def parse_pocket(path: str | Path) -> dict | None:
    """Read heavy ATOM/HETATM records, retaining residue and backbone metadata.

    Elements come from PDB columns 77-78, as upstream. Chain, residue number,
    and insertion code jointly identify a residue. No hydrogenation, docking
    preparation, ligand-based cropping, or coordinate scaling happens here.
    """
    positions, elements, names, backbone, residues = [], [], [], [], []
    seen = {}
    with Path(path).open() as stream:
        for line in stream:
            if line[:6].strip() not in ("ATOM", "HETATM"):
                continue
            element = line[76:78].strip().capitalize()
            if not element or element in ("H", "D"):
                continue
            key = (line[21], line[22:27])
            if key not in seen:
                seen[key] = len(seen)
            positions.append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            )
            elements.append(element)
            names.append(line[17:20].strip())
            backbone.append(line[12:16].strip() in BACKBONE_ATOMS)
            residues.append(seen[key])
    if not positions:
        return None
    pos = np.asarray(positions, dtype=np.float32)
    if not np.isfinite(pos).all():
        raise ValueError(f"Non-finite pocket coordinates: {path}")
    return {
        "pos": pos,
        "elem": elements,
        "resname": names,
        "backbone": np.asarray(backbone, dtype=bool),
        "resid": np.asarray(residues, dtype=np.int64),
    }


def read_ligand_diffdec(
    path: str | Path, hydrogen_mode: str = "implicit"
) -> Chem.Mol | None:
    """Sanitize the source SDF, then remove all H or materialize explicit H."""
    if hydrogen_mode not in ("implicit", "explicit"):
        raise ValueError("hydrogen_mode must be implicit or explicit")
    try:
        mol = next(iter(Chem.SDMolSupplier(str(path), removeHs=True)))
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return (
            Chem.RemoveAllHs(mol)
            if hydrogen_mode == "implicit"
            else Chem.AddHs(mol, addCoords=True)
        )
    except (OSError, ValueError, RuntimeError, StopIteration):
        return None


def mol_to_data(mol: Chem.Mol) -> Data | None:
    """Tensorize sanitized chemistry without GEOM's radical/fragment filters."""
    try:
        mol = Chem.Mol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        positions = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
        if mol.GetNumAtoms() == 0 or not torch.isfinite(positions).all():
            return None
        pairs, labels = [], []
        for bond in mol.GetBonds():
            label = BOND_IDX_MAP.get(bond.GetBondType())
            if label is None:
                return None
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            pairs.extend(((i, j), (j, i)))
            labels.extend((label, label))
        edges = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t()
        order = (edges[0] * mol.GetNumAtoms() + edges[1]).argsort()
        return Data(
            z=torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()]),
            pos=positions,
            charges=torch.tensor([a.GetFormalCharge() for a in mol.GetAtoms()]),
            edge_index=edges[:, order].contiguous(),
            edge_attr=torch.tensor(labels, dtype=torch.long)[order],
            smiles=Chem.MolToSmiles(mol) or "",
        )
    except (ValueError, RuntimeError):
        return None


def process_ligand_diffdec(
    path: str | Path, hydrogen_mode: str = "implicit"
) -> Data | None:
    """Read and tensorize a ligand while preserving heavy-atom order and pose."""
    mol = read_ligand_diffdec(path, hydrogen_mode)
    return None if mol is None else mol_to_data(mol)
