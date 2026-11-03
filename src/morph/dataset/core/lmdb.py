"""Molecular LMDB storage, streaming preprocessing, and process-local readers."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from contextlib import nullcontext
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING

import lmdb
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from morph.dataset.core.source import molecular_source_identity

if TYPE_CHECKING:
    from collections.abc import Iterable

    from morph.dataset.core.vocab import Distributions, Vocab

SPLITS = ("train", "val", "test")
HYDROGEN_MODES = ("explicit", "implicit")
RECORD_FIELDS = ("z", "pos", "charges", "edge_index", "edge_attr", "smiles")
LEN_KEY = b"__len__"  # Cannot collide with 8-byte integer record keys.
LMDB_MAP_SIZE = 128 * 1024**3  # Virtual address space; disk usage grows with data.


def _int_key(index: int) -> bytes:
    return index.to_bytes(8, "big")


def mol_to_bytes(data: Data) -> bytes:
    """Store raw fields without PyG object or collated tensor-storage overhead."""
    return pickle.dumps({key: getattr(data, key) for key in RECORD_FIELDS}, protocol=4)


def mol_from_bytes(raw: bytes) -> Data:
    """Decode one stored molecular graph."""
    return Data(**pickle.loads(raw))


def open_read_env(path: str) -> lmdb.Environment:
    """Open immutable data for concurrent random access by worker processes."""
    return lmdb.open(
        path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
        max_readers=512,
    )


def read_lmdb_length(path: str) -> int:
    """Read the record count, with support for caches predating the length key."""
    with open_read_env(path) as env, env.begin() as txn:
        length = txn.get(LEN_KEY)
        return int(length) if length is not None else txn.stat()["entries"]


def _convert_to_bytes(record, converter) -> bytes | None:
    data = converter(record)
    return None if data is None else mol_to_bytes(data)


def write_bytes_to_lmdb(
    bytes_iter: Iterable[bytes | None],
    lmdb_path: str,
    total: int | None = None,
    map_size: int = LMDB_MAP_SIZE,
    commit_every: int = 10_000,
    desc: str = "Writing LMDB",
) -> tuple[int, int]:
    """Stream serialized molecule bytes into a new LMDB environment.

    ``None`` entries in the iterator are counted as failures and skipped.
    """
    os.makedirs(os.path.dirname(lmdb_path) or ".", exist_ok=True)
    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        subdir=True,
        meminit=False,
        writemap=False,
        max_readers=512,
    )
    n_written = 0
    n_failed = 0
    try:
        txn = env.begin(write=True)
        for out in tqdm(bytes_iter, total=total, desc=desc):
            if out is None:
                n_failed += 1
                continue
            txn.put(_int_key(n_written), out)
            n_written += 1
            if n_written % commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.put(LEN_KEY, str(n_written).encode())
        txn.commit()
    finally:
        env.close()
    return n_written, n_failed


class MolecularLMDBDataset(Dataset):
    """Shared dataset options and storage; subclasses implement source loading."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        vocab: Vocab | None = None,
        distributions: Distributions | None = None,
        preprocess_workers: int | None = 10,
        preprocess_chunk_size: int = 128,
        hydrogen_mode: str = "explicit",
        max_atoms: int | None = None,
        allowed_atom_symbols: tuple[str, ...] | None = None,
    ):
        if split not in SPLITS or hydrogen_mode not in HYDROGEN_MODES:
            raise ValueError(f"Expected split in {SPLITS} and mode in {HYDROGEN_MODES}")
        if (preprocess_workers or 0) < 0 or preprocess_chunk_size < 1:
            raise ValueError("Workers must be non-negative and chunk size positive")
        if max_atoms is not None and max_atoms < 1:
            raise ValueError("max_atoms must be positive or None")
        self.root, self.split = os.fspath(root), split
        self.vocab, self.distributions = vocab, distributions
        self.preprocess_workers = preprocess_workers or 0
        self.preprocess_chunk_size = preprocess_chunk_size
        self.hydrogen_mode = hydrogen_mode
        filters = {
            "max_atoms": max_atoms,
            "allowed_atom_symbols": (
                list(allowed_atom_symbols) if allowed_atom_symbols is not None else None
            ),
        }
        self.processing_options = {**filters, "hydrogen_mode": hydrogen_mode}
        self.raw_dir = os.path.join(self.root, "raw")
        self.processed_dir = os.path.join(self.root, "processed")
        self.lmdb_path = os.path.join(self.processed_dir, f"{split}_data.lmdb")
        self._env = self._txn = self._pid = None

        processed = Path(self.processed_dir)
        manifest = processed / "preprocessing_summary.json"
        metadata = json.loads(manifest.read_text()) if manifest.is_file() else {}
        if any(processed.glob("*_data.lmdb")):
            # Older caches have only includes_hydrogens, or no metadata (explicit).
            cached_mode = metadata.get(
                "hydrogen_mode",
                "implicit"
                if metadata.get("includes_hydrogens") is False
                else "explicit",
            )
            if cached_mode != hydrogen_mode:
                raise ValueError(
                    f"Requested {hydrogen_mode} H but cache is {cached_mode}: {root}"
                )
            cached_filters = metadata.get("processing_options")
            if cached_filters is not None and cached_filters != filters:
                raise ValueError(f"Cache uses different preprocessing options: {root}")
        processed.mkdir(parents=True, exist_ok=True)
        if not Path(self.lmdb_path).exists():
            self.prepare()
            metadata.update(
                hydrogen_mode=hydrogen_mode,
                includes_hydrogens=hydrogen_mode == "explicit",
                processing_options=filters,
            )
            with tempfile.NamedTemporaryFile(
                mode="w", dir=processed, delete=False
            ) as stream:
                json.dump(metadata, stream, indent=2)
            os.replace(stream.name, manifest)
        self._length = read_lmdb_length(self.lmdb_path)
        if not (processed / f"{split}_smiles.txt").is_file():
            self._write_smiles_sidecar()

    def source_identity(self) -> dict:
        """Identify the ordered records and preprocessing recipe for this split."""
        return molecular_source_identity(
            self,
            storage=Path(self.lmdb_path) / "data.mdb",
            split=self.split,
            hydrogen_mode=self.hydrogen_mode,
            processing_options=self.processing_options,
        )

    def prepare(self) -> None:
        """Load and convert a missing source split in dataset subclasses."""
        raise FileNotFoundError(self.lmdb_path)

    def _write_records(self, records, converter, total: int | None = None) -> None:
        """Convert in source order and publish a split only after a complete write."""
        convert = partial(_convert_to_bytes, converter=converter)
        workers = self.preprocess_workers
        if total is not None:
            # imap assigns whole chunks; extra workers cannot help small splits.
            chunks = (
                total + self.preprocess_chunk_size - 1
            ) // self.preprocess_chunk_size
            workers = min(workers, chunks)
        pool_context = (
            get_context("spawn").Pool(workers) if workers > 1 else nullcontext()
        )
        with (
            tempfile.TemporaryDirectory(dir=self.processed_dir) as staging,
            pool_context as pool,
        ):
            path = str(Path(staging) / "data.lmdb")
            outputs = (
                pool.imap(convert, records, chunksize=self.preprocess_chunk_size)
                if pool is not None
                else map(convert, records)
            )
            written, failed = write_bytes_to_lmdb(outputs, path, total=total)
            if written == 0:
                raise ValueError(f"No valid molecules in {self.split}")
            os.rename(path, self.lmdb_path)
        print(f"{self.split}: wrote {written} molecules; skipped {failed}")

    def _ensure_txn(self):
        """Reuse one transaction per process; reopen handles inherited by fork."""
        if self._env is None or self._pid != os.getpid():
            self.close()
            self._env = open_read_env(self.lmdb_path)
            self._txn = self._env.begin(buffers=True)
            self._pid = os.getpid()
        return self._txn

    def close(self) -> None:
        """Release this process's transaction and memory mapping."""
        if self._txn is not None:
            self._txn.abort()
        if self._env is not None:
            self._env.close()
        self._env = self._txn = self._pid = None

    def __getstate__(self):
        """Omit live handles when sending a dataset to spawned DataLoader workers."""
        return {**self.__dict__, "_env": None, "_txn": None, "_pid": None}

    def _write_smiles_sidecar(self) -> None:
        """Persist sorted unique SMILES for novelty metrics."""
        with open_read_env(self.lmdb_path) as env, env.begin() as txn:
            smiles = {
                pickle.loads(txn.get(_int_key(i)))["smiles"]
                for i in range(self._length)
            }
        path = Path(self.processed_dir) / f"{self.split}_smiles.txt"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, delete=False
            ) as stream:
                temporary = stream.name
                stream.write("\n".join(sorted(smiles)))
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def get_all_smiles(self) -> list[str]:
        """Read the unique SMILES for this split."""
        return (
            (Path(self.processed_dir) / f"{self.split}_smiles.txt")
            .read_text()
            .splitlines()
        )

    def iter_bytes(self) -> Iterable[bytes]:
        """Read serialized records in source order."""
        txn = self._ensure_txn()
        for index in range(self._length):
            yield bytes(txn.get(_int_key(index)))

    def __len__(self):
        """Return the number of records in this split."""
        return self._length

    def __getitem__(self, index):
        """Decode a raw molecular record."""
        if index < 0 or index >= self._length:
            raise IndexError(index)
        return mol_from_bytes(bytes(self._ensure_txn().get(_int_key(index))))

    def get(self, index):
        """Read raw fields for preprocessing, without tokenized decoding."""
        return self.__getitem__(index)
