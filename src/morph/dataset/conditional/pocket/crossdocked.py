"""Lazy paired ligand/pocket reads in a shared, ligand-centered coordinate frame."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import weakref
from pathlib import Path

import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data

from morph.dataset.core.atomic_data import make_chem_atomic
from morph.dataset.core.lmdb import RECORD_FIELDS, SPLITS, _int_key, open_read_env
from morph.dataset.core.representation import Capabilities
from morph.dataset.core.source import molecular_source_identity
from morph.dataset.core.tokenized_decode import TokenizedDecodeMixin, _lookup_tokens
from morph.dataset.core.vocab import Vocab

LOGGER = logging.getLogger(__name__)


class _Reader:
    """Share immutable LMDB mappings between datasets in the same process."""

    def __init__(self, path):
        self.pid = os.getpid()
        self.env = open_read_env(str(path))
        self.txn = self.env.begin(buffers=True)

    def __del__(self):
        self.close()

    def close(self):
        """Release the mapping, including handles inherited by forked workers."""
        if getattr(self, "txn", None) is not None:
            self.txn.abort()
            self.txn = None
        if getattr(self, "env", None) is not None:
            self.env.close()
            self.env = None


_READERS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


class CrossDockedPocketDataset(TokenizedDecodeMixin, Dataset):
    """CrossDocked ligands with independently accessible protein-pocket context.

    ``get(i)`` returns raw ligand fields for existing preprocessing/scaffold
    utilities. ``dataset[i]`` uses the supplied ligand vocabulary and
    distributions. ``get_pocket(i)`` requires an explicit ``pocket_vocab`` and
    uses the same coordinate scale. These two tokenized reads return ChemAtomicData.
    ``get_raw_pocket(i)`` retains the complete, untrimmed pocket and residue
    information; ``get_metadata(i)`` provides the original receptor translation.

    Missing caches are built automatically. The default processing matches
    morph's DiffDec heavy-atom recipe; no size or element filter is imposed.
    For fine-tuning, supply the checkpoint's vocabulary and distributions.
    Tokenized readers automatically exclude ligands with unsupported elements;
    ``vocab_filter_report`` records coverage and dropped counts. Raw caches stay
    unchanged. ``filter_by_vocab=False`` opts out for inspection only.
    ``allowed_atom_symbols`` aliases ``allowed_ligand_atom_symbols`` for shared
    dataset configurations; both filter ligand elements only.
    """

    CAPABILITIES = Capabilities(provides_charges=True, provides_topology=True)

    def __init__(
        self,
        root: str | Path,
        vocab=None,
        distributions=None,
        split: str = "train",
        source_dir: str | Path | None = None,
        hydrogen_mode: str = "implicit",
        max_atoms: int | None = None,
        allowed_ligand_atom_symbols=None,
        val_fraction: float = 0.02,
        seed: int = 42,
        max_pocket_atoms: int | None = 512,
        preprocess_workers: int = 0,
        preprocess_chunk_size: int = 32,
        *,
        pocket_vocab=None,
        filter_by_vocab: bool = True,
        allowed_atom_symbols=None,
    ):
        # Keep the build CLI importable via python -m without pre-importing it.
        from morph.dataset.conditional.pocket.build import build_lmdbs  # noqa: PLC0415

        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if max_pocket_atoms is not None and max_pocket_atoms < 1:
            raise ValueError("max_pocket_atoms must be positive or None")
        if allowed_atom_symbols is not None:
            if allowed_ligand_atom_symbols is not None:
                raise ValueError("Supply only one ligand element filter")
            allowed_ligand_atom_symbols = allowed_atom_symbols
        self.root, self.split = str(root), split
        self.processed_dir = str(Path(root) / "processed")
        self.lmdb_path = str(Path(self.processed_dir) / f"{split}_data.lmdb")
        self.vocab, self.distributions = vocab, distributions
        self.pocket_vocab = pocket_vocab
        self.max_pocket_atoms = max_pocket_atoms
        manifest = build_lmdbs(
            root,
            source_dir,
            hydrogen_mode=hydrogen_mode,
            max_atoms=max_atoms,
            allowed_ligand_atom_symbols=allowed_ligand_atom_symbols,
            val_fraction=val_fraction,
            seed=seed,
            preprocess_workers=preprocess_workers,
            preprocess_chunk_size=preprocess_chunk_size,
        )
        self._length = manifest["splits"][split]["count"]
        self._pocket_elements = sorted(manifest["splits"]["train"]["pocket_elements"])
        self._reader = None
        self._cache_index = self._cache_record = None
        self._indices = None
        self.vocab_filter_report = None
        if vocab is not None:
            self._build_token_maps(vocab)
            if filter_by_vocab:
                self._filter_ligands(manifest["splits"][split])

    def get_pocket_vocab(self) -> Vocab:
        """Derive protein tokens from training data, shared by all splits."""
        return Vocab(list(self._pocket_elements), ["<NO_BOND>"], ["0"])

    def source_identity(self) -> dict:
        """Identify the exact ligand records retained by this reader."""
        # Preserve the published v2 identity: preprocessing is already reflected
        # in the LMDB bytes; hydrogen_mode and processing_options remain null.
        return molecular_source_identity(
            self,
            storage=Path(self.lmdb_path) / "data.mdb",
            split=self.split,
            indices=self._indices,
        )

    def cache_namespace(self) -> str:
        """Group caches by source recipe and accepted ligand elements."""
        from morph.dataset.conditional.pocket.build import MANIFEST  # noqa: PLC0415

        # Preserve the directory names used by the published scaffold caches.
        manifest = Path(self.processed_dir, MANIFEST).read_bytes()
        atoms = sorted(self.vocab.atom_tokens) if self.vocab is not None else None
        return hashlib.sha256(manifest + b"\0" + json.dumps(atoms).encode()).hexdigest()

    def _filter_ligands(self, split_summary):
        """Check raw element coverage and retain only checkpoint-compatible ligands."""
        table = Chem.GetPeriodicTable()
        present = {
            table.GetElementSymbol(int(z))
            for z in split_summary["ligand_atomic_numbers"]
        }
        missing = sorted(present - set(self.vocab.atom_tokens))
        source_count = self._length
        # The inventory makes the common, fully compatible case constant-time.
        if missing:
            allowed = self._z_to_atom_idx >= 0
            retained = [
                i
                for i in range(source_count)
                if bool(allowed[self._read(i)["z"]].all())
            ]
            self.close()
            self._indices = torch.tensor(retained, dtype=torch.long)
            self._length = len(retained)
        self.vocab_filter_report = {
            "source_count": source_count,
            "retained": self._length,
            "dropped": source_count - self._length,
            "missing_elements": missing,
        }
        LOGGER.info(
            "CrossDocked %s checkpoint vocabulary: %s",
            self.split,
            self.vocab_filter_report,
        )

    def __len__(self):
        """Count retained ligand-pocket pairs in this split."""
        return self._length

    def _read(self, index):
        if index < 0 or index >= self._length:
            raise IndexError(index)
        if self._indices is not None:
            index = int(self._indices[index])
        if self._reader is None or self._reader.pid != os.getpid():
            # Drop inherited handles before opening a fresh mapping after fork.
            for key, reader in list(_READERS.items()):
                if key[0] != os.getpid():
                    reader.close()
                    del _READERS[key]
            self.close()
            key = (os.getpid(), str(Path(self.lmdb_path).resolve()))
            reader = _READERS.get(key)
            if reader is None:
                reader = _Reader(key[1])
                _READERS[key] = reader
            self._reader = reader
        if self._cache_index != index:
            raw = self._reader.txn.get(_int_key(int(index)))
            if raw is None:
                raise ValueError(
                    f"Missing CrossDocked record {index}: {self.lmdb_path}"
                )
            self._cache_record = pickle.loads(raw)
            self._cache_index = index
        return self._cache_record

    def close(self) -> None:
        """Release this dataset's reference to the process-local mapping."""
        self._reader = None
        self._cache_index = self._cache_record = None

    def __getstate__(self):
        """Exclude handles and cached records when spawning DataLoader workers."""
        return {
            **self.__dict__,
            "_reader": None,
            "_cache_index": None,
            "_cache_record": None,
        }

    def get(self, index) -> Data:
        """Read unnormalized ligand tensors, preserving source atom order."""
        record = self._read(index)
        return Data(**{key: record[key] for key in RECORD_FIELDS}).clone()

    def get_raw_pocket(self, index) -> dict:
        """Return untrimmed pocket fields in ligand-centered Angstroms, as copies."""
        record = self._read(index)
        return {
            key: record[f"poc_{key}"].copy()
            for key in ("pos", "elem", "resname", "backbone", "resid")
        }

    def get_metadata(self, index) -> dict:
        """Return provenance and translation back into the original receptor frame."""
        record = self._read(index)
        return {
            "lig_center": torch.as_tensor(record["lig_center"]).clone().float(),
            "pocket_fn": record["pocket_fn"],
            "ligand_fn": record["ligand_fn"],
            "smiles": record["smiles"],
            "source_index": record["source_index"],
        }

    def _require_tokens(self):
        if self.vocab is None or self.distributions is None:
            raise RuntimeError(
                "A vocab and distributions are required; use .get() or "
                ".get_raw_pocket() for raw data"
            )

    def _scale(self, positions):
        std = self.distributions.coordinate_std
        return positions / std if std is not None else positions.clone()

    def __getitem__(self, index):
        """Tokenize the ligand without changing its frame relative to the pocket."""
        self._require_tokens()
        data = self.get(index)
        return make_chem_atomic(
            x=self._scale(data.pos),
            a=_lookup_tokens(data.z, self._z_to_atom_idx, "atom"),
            c=_lookup_tokens(
                data.charges - self._charge_min, self._charge_to_idx, "charge"
            ),
            e=_lookup_tokens(data.edge_attr, self._bond_to_idx, "bond"),
            edge_index=data.edge_index,
            smiles=data.smiles,
        )

    def get_pocket(self, index):
        """Tokenize heavy pocket atoms, optionally retaining the nearest inner shell.

        Pocket charges are neutral placeholders and its edges are empty, as in
        morph. Cropping never re-centers the pocket. Unknown elements fail
        explicitly. The protein vocabulary is independent of the ligand output
        vocabulary; raw pockets remain available for other protein encoders.
        """
        if self.pocket_vocab is None or self.distributions is None:
            raise RuntimeError(
                "A pocket_vocab and distributions are required; use "
                ".get_raw_pocket() for raw data"
            )
        raw = self.get_raw_pocket(index)
        pos = torch.from_numpy(raw["pos"]).float()
        keep = torch.arange(len(pos))
        if self.max_pocket_atoms is not None and len(pos) > self.max_pocket_atoms:
            keep = (
                torch.argsort(pos.square().sum(dim=1), stable=True)[
                    : self.max_pocket_atoms
                ]
                .sort()
                .values
            )
        elem_to_idx = {e: i for i, e in enumerate(self.pocket_vocab.atom_tokens)}
        elements = [raw["elem"][i] for i in keep.tolist()]
        unknown = sorted(set(elements) - elem_to_idx.keys())
        if unknown:
            raise ValueError(
                f"Pocket elements absent from the pocket vocabulary: {unknown}"
            )
        if "0" not in self.pocket_vocab.charge_tokens:
            raise ValueError("Pocket context requires neutral charge token '0'")
        pocket = make_chem_atomic(
            x=self._scale(pos[keep]),
            a=torch.tensor([elem_to_idx[e] for e in elements], dtype=torch.long),
            c=torch.full(
                (len(keep),),
                self.pocket_vocab.charge_tokens.index("0"),
                dtype=torch.long,
            ),
            e=torch.empty(0, dtype=torch.long),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )
        pocket.add_node_property("backbone", torch.from_numpy(raw["backbone"])[keep])
        pocket.add_node_property("residue_index", torch.from_numpy(raw["resid"])[keep])
        return pocket

    def _field_values(self, key, suffix):
        path = Path(self.processed_dir) / f"{self.split}_{suffix}.txt"
        if path.is_file():
            values = path.read_text().splitlines()
            source_count = (
                self.vocab_filter_report["source_count"]
                if self.vocab_filter_report is not None
                else self._length
            )
            if len(values) == source_count:
                return (
                    values
                    if self._indices is None
                    else [values[i] for i in self._indices.tolist()]
                )
        return [str(self._read(i)[key]) for i in range(self._length)]

    def get_all_smiles(self) -> list[str]:
        """Return ordered ligand SMILES, keeping duplicates for frequency analysis."""
        return self._field_values("smiles", "smiles")

    def get_all_ligand_fns(self) -> list[str]:
        """Return ordered source ligand paths, matching retained dataset indices."""
        return self._field_values("ligand_fn", "ligand_fns")
