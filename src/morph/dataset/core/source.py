"""Portable molecular source identities and shared file-checksum caching."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np

from morph.utils.cache import atomic_write, cache_root, file_lock


@lru_cache(maxsize=32)
def _file_digest(path: Path, stat_key: tuple) -> str:
    # Paths/stat data only identify local memoized reads. They never enter the
    # portable scaffold identity. The lock lets other ranks reuse the first read.
    key = hashlib.sha256(str(path).encode()).hexdigest()
    cached = cache_root() / "source_checksums" / f"{key}.json"
    with file_lock(cached):
        if _stat_key(path) != stat_key:
            raise RuntimeError(f"Molecular source changed before hashing: {path}")
        try:
            record = json.loads(cached.read_text())
        except (FileNotFoundError, ValueError):
            record = None
        if isinstance(record, dict) and record.get("stat") == list(stat_key):
            digest = record.get("sha256")
            if (
                isinstance(digest, str)
                and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest)
            ):
                return digest
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if _stat_key(path) != stat_key:
            raise RuntimeError(f"Molecular source changed while hashing: {path}")
        atomic_write(cached, json.dumps({"stat": stat_key, "sha256": digest}).encode())
    return digest


def _stat_key(path: Path) -> tuple:
    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def molecular_source_identity(
    dataset,
    *,
    storage: str | Path | None = None,
    split: str | None = None,
    hydrogen_mode: str | None = None,
    processing_options: dict | None = None,
    indices=None,
) -> dict:
    """Identify ordered molecular records, optionally using immutable file bytes.

    Dataset adapters supply their storage file and any retained record indices.
    Without a storage file, hash raw ``get(i)`` fields in order. Local checksum
    memoization uses paths and stat data; the returned identity never does.
    """
    identity = {
        "class": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "records": len(dataset),
        "split": split,
        "hydrogen_mode": hydrogen_mode,
        "processing_options": deepcopy(processing_options),
    }
    if indices is not None:
        if hasattr(indices, "detach"):
            indices = indices.detach().cpu().numpy()
        identity["index_sha256"] = hashlib.sha256(
            np.asarray(indices, dtype=np.int64).tobytes()
        ).hexdigest()
    if storage is not None:
        storage = Path(storage).resolve()
        identity["sha256"] = _file_digest(storage, _stat_key(storage))
    else:
        digest = hashlib.sha256()
        for index in range(len(dataset)):
            data = dataset.get(index)
            for field in ("z", "pos", "charges", "edge_index", "edge_attr"):
                value = getattr(data, field, None)
                digest.update(field.encode())
                if value is None:
                    digest.update(b"none")
                else:
                    value = value.detach().cpu().contiguous()
                    digest.update(str((value.dtype, tuple(value.shape))).encode())
                    digest.update(value.numpy().tobytes())
        identity["sha256"] = digest.hexdigest()
    return identity
