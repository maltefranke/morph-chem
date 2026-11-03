"""Dataset-independent bundle metadata, checkpoint selection and local validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from morph.inference.registry import ModelSpec, bundle_path

ASSET_FILES = (
    "config.yaml",
    "metadata.json",
    "atom_tokens.txt",
    "edge_tokens.txt",
    "charge_tokens.txt",
    "distributions.pt",
)


@dataclass(frozen=True)
class ModelBundle:
    """Local paths and resolved identity for a checkpoint and its inference assets."""

    spec: ModelSpec
    epoch: int | None
    directory: Path
    checkpoint: Path
    revision: str | None

    @property
    def config_path(self) -> Path:
        """Portable inference configuration, not the historical training config."""
        return self.directory / "config.yaml"


def read_metadata(directory: Path, spec: ModelSpec) -> dict:
    """Read a bundle's identity and check any expectations from the catalog."""
    metadata = json.loads((directory / "metadata.json").read_text())
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object")
    dataset = metadata.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("metadata.json must identify the dataset")
    if spec.dataset is not None and dataset.casefold() != spec.dataset.casefold():
        raise ValueError(
            f"Expected {spec.dataset} assets; found {dataset!r} in {directory}"
        )
    variant = metadata.get("variant")
    if spec.variant is not None and variant != spec.variant:
        raise ValueError(
            f"Expected {spec.variant} assets; found {variant!r} in {directory}"
        )
    if variant not in {"explicit_h", "implicit_h"}:
        raise ValueError("metadata.variant must be 'explicit_h' or 'implicit_h'")
    for field in ("atom_tokens", "edge_tokens", "charge_tokens"):
        tokens = metadata.get(field)
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(not isinstance(token, str) or not token for token in tokens)
            or len(set(tokens)) != len(tokens)
        ):
            raise ValueError(
                f"metadata.{field} must be a nonempty ordered list of unique tokens"
            )
    has_h = "H" in metadata["atom_tokens"]
    if has_h != (variant == "explicit_h"):
        raise ValueError("Atom vocabulary does not match the hydrogen representation")
    declared_h = metadata.get("includes_explicit_hydrogens", has_h)
    if type(declared_h) is not bool or declared_h != has_h:
        raise ValueError("includes_explicit_hydrogens does not match the vocabulary")
    return metadata


def select_checkpoint(metadata: dict, epoch: int | None, filename: str | None):
    """Select a filename from metadata, with an explicit filename escape hatch."""
    if epoch is not None and filename is not None:
        raise ValueError("Choose either epoch or filename, not both")
    entries = metadata.get("checkpoints", [])
    if not isinstance(entries, list):
        raise ValueError("metadata.checkpoints must be a list")
    by_epoch = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(
                "Each checkpoint must contain filename and completed_epochs"
            )
        completed = entry.get("completed_epochs")
        if type(completed) is not int or completed < 1:
            raise ValueError(
                "Each checkpoint must have positive integer completed_epochs"
            )
        path = entry.get("filename")
        if not isinstance(path, str):
            raise ValueError("Each checkpoint must have a filename")
        path = bundle_path(path)
        if completed in by_epoch or path in by_epoch.values():
            raise ValueError("Checkpoint epochs and filenames must be unique")
        by_epoch[completed] = path
    if filename is not None:
        filename = bundle_path(filename)
        selected_epoch = next(
            (e for e, path in by_epoch.items() if path == filename), None
        )
        return selected_epoch, filename
    if by_epoch:
        epoch = max(by_epoch) if epoch is None else epoch
        if epoch not in by_epoch:
            raise ValueError(
                f"Bundle has completed epochs {sorted(by_epoch)}; requested {epoch}"
            )
        return epoch, by_epoch[epoch]
    if epoch is not None:
        # Support older bundles without a checkpoint manifest.
        return epoch, f"epoch_{epoch:03d}.ckpt"
    raise ValueError("Provide epoch or filename, or list checkpoints in metadata.json")


def local_bundle(
    spec: ModelSpec,
    directory: str | Path,
    *,
    epoch: int | None = None,
    filename: str | None = None,
    checkpoint: str | Path | None = None,
    revision: str | None = None,
) -> ModelBundle:
    """Resolve a direct bundle directory or export root, without dataset access."""
    directory = Path(directory).expanduser().resolve()
    nested = directory / spec.subfolder
    if (nested / "metadata.json").is_file() or not (
        directory / "metadata.json"
    ).is_file():
        directory = nested
    missing = [
        str(directory / name)
        for name in ASSET_FILES
        if not (directory / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Incomplete model bundle. Missing: " + ", ".join(missing)
        )
    metadata = read_metadata(directory, spec)
    for field in ("atom_tokens", "edge_tokens", "charge_tokens"):
        tokens = (directory / f"{field}.txt").read_text().splitlines()
        if tokens != metadata[field]:
            raise ValueError(f"{field} does not match the bundle metadata.")
    if checkpoint is None:
        epoch, filename = select_checkpoint(metadata, epoch, filename)
        checkpoint = directory / filename
    else:
        if filename is not None:
            raise ValueError("Choose either checkpoint or filename, not both")
        checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Incomplete model bundle. Missing checkpoint: {checkpoint}"
        )
    spec = replace(spec, dataset=metadata["dataset"], variant=metadata["variant"])
    return ModelBundle(spec, epoch, directory, checkpoint, revision)
