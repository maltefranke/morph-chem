"""Versioned scaffold caches, built atomically and read lazily by workers."""

from __future__ import annotations

import os
import pickle
import tempfile
from array import array
from pathlib import Path
from threading import Lock

import lmdb
import numpy as np
from tqdm import tqdm

from morph.dataset.conditional.scaffold.cutters import (
    SCAFFOLD_KINDS,
    _atom_indices_for_data,
)
from morph.dataset.conditional.scaffold.fingerprint import (
    metadata_key,
    scaffold_cache_metadata,
)
from morph.dataset.core.lmdb import LMDB_MAP_SIZE, open_read_env
from morph.utils.cache import file_lock as scaffold_cache_lock

_META_N = b"meta:n"
_META_NONEMPTY = b"meta:scaffold_nonempty_indices"
_META_IDENTITY = b"meta:scaffold_identity"


def _kind_suffix(scaffold_kind: str) -> str:
    if scaffold_kind not in SCAFFOLD_KINDS:
        raise ValueError(
            f"unknown scaffold_kind {scaffold_kind!r}; "
            f"expected one of {sorted(SCAFFOLD_KINDS)}"
        )
    # Existing caches without a suffix contain Murcko matches.
    return "" if scaffold_kind == "murcko" else f"_{scaffold_kind}"


def scaffold_lmdb_path(
    scaffold_groups_dir: str,
    stage: str,
    scaffold_kind: str = "rgroup",
    *,
    cache_key: str | None = None,
) -> str:
    """Name a cache generation; no key denotes an unverifiable legacy cache."""
    generation = f"_{cache_key}" if cache_key else ""
    return os.path.join(
        scaffold_groups_dir,
        f"scaffold_{stage}{_kind_suffix(scaffold_kind)}{generation}.lmdb",
    )


def scaffold_pt_path(
    scaffold_groups_dir: str, stage: str, scaffold_kind: str = "rgroup"
) -> str:
    """Name a legacy, unverified PyTorch cache; new caches use versioned LMDBs."""
    return os.path.join(
        scaffold_groups_dir,
        f"scaffold_groups_{stage}{_kind_suffix(scaffold_kind)}.pt",
    )


def _key(prefix: str, i: int) -> bytes:
    return f"{prefix}:{int(i)}".encode()


class _LazyField:
    """Subscriptable view of one LMDB key prefix."""

    __slots__ = ("_cache", "_default", "_prefix")

    def __init__(self, cache: ScaffoldLMDBCache, prefix: str, default):
        self._cache = cache
        self._prefix = prefix
        self._default = default

    def __getitem__(self, i):
        return self._cache._read(self._prefix, i, self._default)


_READ_ENVS: dict[str, lmdb.Environment] = {}
_READ_TXNS: dict[tuple[str, int], object] = {}
_READ_LOCK = Lock()


def _shared_read_env(path: str) -> lmdb.Environment:
    with _READ_LOCK:
        env = _READ_ENVS.get(path)
        if env is None:
            env = open_read_env(path)
            _READ_ENVS[path] = env
        return env


def _shared_read_txn(path: str):
    key = (path, os.getpid())
    txn = _READ_TXNS.get(key)
    if txn is None:
        txn = _shared_read_env(path).begin()
        _READ_TXNS[key] = txn
    return txn


class ScaffoldLMDBCache:
    """Lazy reader over a scaffold LMDB shared by dataloader workers."""

    def __init__(self, lmdb_path: str):
        self.lmdb_path = str(Path(lmdb_path).resolve())
        with _shared_read_env(self.lmdb_path).begin() as txn:
            raw_metadata = txn.get(_META_IDENTITY)
            self.metadata = pickle.loads(raw_metadata) if raw_metadata else None
            self.n = pickle.loads(txn.get(_META_N))
            self._filtered = np.asarray(
                pickle.loads(txn.get(_META_NONEMPTY)), dtype=np.int64
            )
        self.atom_indices = _LazyField(self, "ai", [])

    def _read(self, prefix: str, i, default):
        raw = _shared_read_txn(self.lmdb_path).get(_key(prefix, int(i)))
        return default if raw is None else pickle.loads(raw)

    def filtered_indices(self):
        """Indices of source molecules with cached scaffold annotations."""
        return self._filtered

    def smiles(self, index):
        """Reuse optional SMILES metadata without decoding conformer tensors."""
        return self._read("smi", index, None)


def _build_cache(dataset, path: str, metadata: dict) -> None:
    """Stream cuts to staging storage; publish only a complete, unchanged source."""
    with tempfile.TemporaryDirectory(dir=Path(path).parent) as staging:
        temporary = str(Path(staging) / "cache.lmdb")
        with lmdb.open(temporary, map_size=LMDB_MAP_SIZE) as env:
            nonempty = array("q")
            txn = env.begin(write=True)
            try:
                for index in tqdm(range(len(dataset)), desc="Scaffold cache"):
                    data = dataset.get(index)
                    cuts = _atom_indices_for_data(data, metadata["cutter"]["kind"])
                    txn.put(_key("ai", index), pickle.dumps(cuts))
                    smiles = getattr(data, "smiles", None)
                    if smiles is not None:
                        txn.put(_key("smi", index), pickle.dumps(smiles))
                    if cuts:
                        nonempty.append(index)
                    if (index + 1) % 10_000 == 0:
                        txn.commit()
                        txn = env.begin(write=True)
                txn.put(_META_N, pickle.dumps(len(dataset)))
                txn.put(_META_NONEMPTY, pickle.dumps(nonempty))
                txn.put(_META_IDENTITY, pickle.dumps(metadata))
                txn.commit()
                txn = None
            finally:
                if txn is not None:
                    txn.abort()
        if (
            scaffold_cache_metadata(
                dataset, metadata["stage"], metadata["cutter"]["kind"]
            )
            != metadata
        ):
            raise RuntimeError("Scaffold source changed while building its cache")
        os.rename(temporary, path)


def load_scaffold_cache(
    base_dataset,
    scaffold_groups_dir: str,
    stage: str,
    *,
    scaffold_kind: str = "rgroup",
):
    """Load or build matches tied to the source and cutter fingerprints.

    Old unversioned LMDB/``.pt`` caches cannot establish atom ordering and are
    left untouched. Their first use rebuilds a separate versioned LMDB. Immutable
    generation paths let existing workers finish reading an older source safely.
    """
    if base_dataset is None or scaffold_groups_dir is None:
        raise ValueError("base_dataset and scaffold_groups_dir must be provided")
    metadata = scaffold_cache_metadata(base_dataset, stage, scaffold_kind)
    path = scaffold_lmdb_path(
        scaffold_groups_dir, stage, scaffold_kind, cache_key=metadata_key(metadata)
    )
    if not os.path.isdir(path):
        with scaffold_cache_lock(path):
            if not os.path.isdir(path):
                _build_cache(base_dataset, path, metadata)
    cache = ScaffoldLMDBCache(path)
    if cache.metadata != metadata or cache.n != len(base_dataset):
        raise ValueError(f"Invalid scaffold cache metadata: {path}")
    return cache
