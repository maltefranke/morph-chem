"""Source and cutter identities for caches containing positional atom indices."""

from __future__ import annotations

import hashlib
import json

from morph.dataset.conditional.scaffold import cutters
from morph.dataset.core.source import molecular_source_identity

# Bump when the stored fields or scaffold-cut semantics change.
CACHE_VERSION = 2


def dataset_fingerprint(dataset) -> dict:
    """Use the dataset's source identity, or hash raw records for custom datasets."""
    if hasattr(dataset, "source_identity"):
        return dataset.source_identity()
    return molecular_source_identity(dataset)


def scaffold_cache_metadata(dataset, stage: str, kind: str) -> dict:
    """Identify the source, split and explicitly versioned scaffold protocol."""
    if kind not in cutters.SCAFFOLD_KINDS:
        raise ValueError(f"unknown scaffold_kind {kind!r}")
    return {
        "version": CACHE_VERSION,
        "source": dataset_fingerprint(dataset),
        "stage": stage,
        "cutter": {"kind": kind},
    }


def metadata_key(metadata: dict) -> str:
    """Stable filename component for one complete cache generation."""
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
