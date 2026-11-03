"""Model specifications and name resolution through a discoverable Hub catalog."""

from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from morph.inference.catalog import ModelCatalog


def bundle_path(value: str, *, allow_empty: bool = False) -> str:
    """Validate a literal relative path inside a model bundle or repository."""
    if not value and allow_empty:
        return ""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() == "."
        or any(char in value for char in "\\*?[]")
    ):
        raise ValueError(
            f"Expected a relative bundle path without glob patterns: {value!r}"
        )
    return path.as_posix()


@dataclass(frozen=True)
class ModelSpec:
    """Location and optional identity checks for any compatible model bundle.

    An empty ``epochs`` tuple defers selection to ``metadata.json``. Each catalog
    entry owns its repository, subfolder and revision; none are dataset defaults.
    ``repo_id=None`` describes local-only assets.
    """

    name: str
    repo_id: str | None = None
    subfolder: str = ""
    revision: str | None = None
    dataset: str | None = None
    variant: str | None = None
    epochs: tuple[int, ...] = ()

    def __post_init__(self):
        """Normalize the bundle path and validate optional published epochs."""
        object.__setattr__(
            self, "subfolder", bundle_path(self.subfolder, allow_empty=True)
        )
        if any(type(epoch) is not int or epoch < 1 for epoch in self.epochs):
            raise ValueError("Catalog epochs must be positive completed-epoch integers")
        if len(set(self.epochs)) != len(self.epochs):
            raise ValueError("Catalog epochs must be unique")

    def select_epoch(self, epoch: int | None = None) -> int | None:
        """Use the latest catalog epoch, or defer to the bundle metadata."""
        if epoch is not None and (type(epoch) is not int or epoch < 1):
            raise ValueError("epoch must be a positive completed-epoch integer")
        selected = max(self.epochs) if epoch is None and self.epochs else epoch
        if self.epochs and selected not in self.epochs:
            raise ValueError(
                f"{self.name} has epochs {', '.join(map(str, self.epochs))}; "
                f"requested {selected}. Epochs count completed training epochs."
            )
        return selected


def get_model(
    name: str | ModelSpec | None = None,
    *,
    subfolder: str | None = None,
    revision: str | None = None,
    catalog: "ModelCatalog | None" = None,
    cache_dir: "str | Path | None" = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> ModelSpec:
    """Resolve a catalog name, explicit specification, or HF ``owner/repo`` ID."""
    if isinstance(name, ModelSpec):
        spec = name
    elif isinstance(name, str) and name.count("/") == 1 and all(name.split("/")):
        spec = ModelSpec(name, repo_id=name)
    else:
        # Explicit specifications and repository IDs never need catalog access.
        from morph.inference.catalog import load_catalog  # noqa: PLC0415

        if catalog is None:
            catalog = load_catalog(
                cache_dir=cache_dir, local_files_only=local_files_only, token=token
            )
        selected = catalog.default_model if name is None else name
        if selected is None:
            raise ValueError("This catalog has no default model; choose a model name")
        try:
            spec = catalog.models[selected]
        except KeyError:
            raise ValueError(
                f"Unknown model {selected!r}. Available: {', '.join(catalog.models)}. "
                "You can also pass a Hugging Face owner/repo ID or ModelSpec."
            ) from None
    if subfolder is not None and subfolder != spec.subfolder:
        # A different bundle in the same repository has its own identity/epochs.
        spec = replace(spec, subfolder=subfolder, dataset=None, variant=None, epochs=())
    if revision is not None and revision != spec.revision:
        spec = replace(spec, revision=revision, epochs=())
    return spec
