"""Checkpoint compatibility across eager and ``torch.compile`` modules."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch


def canonical_checkpoint_key(key: str) -> str:
    """Remove implementation-detail ``_orig_mod`` path components."""
    return ".".join(part for part in key.split(".") if part != "_orig_mod")


def canonicalize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Return a compile-independent state dict and reject ambiguous keys."""
    canonical: OrderedDict[str, torch.Tensor] = OrderedDict()
    source_keys: dict[str, str] = {}
    for key, value in state_dict.items():
        canonical_key = canonical_checkpoint_key(key)
        if canonical_key in canonical and source_keys[canonical_key] != key:
            raise RuntimeError(
                "Checkpoint keys collide after removing _orig_mod: "
                f"{source_keys[canonical_key]!r} and {key!r}"
            )
        canonical[canonical_key] = value
        source_keys[canonical_key] = key
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        canonical._metadata = {
            canonical_checkpoint_key(key): value for key, value in metadata.items()
        }
    return canonical


def state_dict_for_module(
    state_dict: Mapping[str, torch.Tensor], module: torch.nn.Module
) -> OrderedDict[str, torch.Tensor]:
    """Map canonical checkpoint keys onto an eager or compiled module."""
    canonical = canonicalize_state_dict(state_dict)
    target_state = module.state_dict()
    target_keys = {canonical_checkpoint_key(key): key for key in target_state}
    adapted = OrderedDict(
        (target_keys.get(key, key), value) for key, value in canonical.items()
    )
    metadata = getattr(canonical, "_metadata", None)
    target_metadata = getattr(target_state, "_metadata", {})
    if metadata is not None:
        target_metadata_keys = {
            canonical_checkpoint_key(key): key for key in target_metadata
        }
        adapted._metadata = {
            target_metadata_keys.get(key, key): value for key, value in metadata.items()
        }
    return adapted
