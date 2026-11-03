"""Discover model shortcuts from a versioned catalog on Hugging Face."""

# Discovery needs only JSON and the Hub client, never the model/scientific stack.
# ruff: noqa: PLC0415

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from morph.inference.registry import ModelSpec

DEFAULT_CATALOG_REPO = "maltefranke/morph"
CATALOG_FILENAME = "catalog.json"


@dataclass(frozen=True)
class ModelCatalog:
    """Discovered model specifications and the catalog's own provenance."""

    models: dict[str, ModelSpec]
    default_model: str | None
    path: Path
    repo_id: str | None = None
    revision: str | None = None


def _read_catalog(path: Path, *, repo_id=None, revision=None) -> ModelCatalog:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int:
        raise ValueError("Catalog must contain an integer schema_version")
    if payload["schema_version"] != 1:
        raise ValueError(
            f"Unsupported catalog schema_version: {payload['schema_version']}"
        )
    if set(payload) - {"schema_version", "default_model", "models"}:
        raise ValueError(
            "Unknown catalog fields; expected schema_version, default_model, models"
        )
    entries = payload.get("models")
    if not isinstance(entries, list):
        raise ValueError("Catalog models must be a list")
    models = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each catalog model must be an object")
        if set(entry) - {
            "name",
            "repo_id",
            "subfolder",
            "revision",
            "dataset",
            "variant",
            "epochs",
        }:
            raise ValueError(f"Unknown fields in catalog model {entry.get('name')!r}")
        name = entry.get("name")
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", name
        ):
            raise ValueError(
                "Catalog model names must be nonempty aliases without slashes or spaces"
            )
        if name in models:
            raise ValueError(f"Duplicate catalog model name: {name}")
        model_repo = entry.get("repo_id")
        if (
            not isinstance(model_repo, str)
            or model_repo.count("/") != 1
            or not all(model_repo.split("/"))
            or any(char.isspace() for char in model_repo)
        ):
            raise ValueError(f"Catalog model {name} must have an owner/repo repo_id")
        model_revision = entry.get("revision")
        if not isinstance(model_revision, str) or not model_revision.strip():
            raise ValueError(f"Catalog model {name} must declare its release revision")
        for field in ("subfolder", "dataset", "variant"):
            value = entry.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"Catalog model {name}: {field} must be a string")
        if "subfolder" in entry and entry["subfolder"] is None:
            raise ValueError(f"Catalog model {name}: subfolder must be a string")
        if entry.get("variant") not in {None, "explicit_h", "implicit_h"}:
            raise ValueError(f"Catalog model {name}: unsupported hydrogen variant")
        epochs = entry.get("epochs", [])
        if not isinstance(epochs, list):
            raise ValueError(f"Catalog model {name}: epochs must be a list")
        models[name] = ModelSpec(**{**entry, "epochs": tuple(epochs)})
    default = payload.get("default_model")
    if default is not None and (not isinstance(default, str) or default not in models):
        raise ValueError("Catalog default_model must name an entry in models")
    return ModelCatalog(models, default, path, repo_id, revision)


def load_catalog(
    repo_id: str | None = None,
    *,
    revision: str | None = None,
    path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> ModelCatalog:
    """Read the current Hub catalog, its cached copy, or an explicit local file.

    Downloads only ``catalog.json``. The default location is controlled by
    ``MORPH_CATALOG_REPO`` and ``MORPH_CATALOG_REVISION`` (default: main).
    A catalog's revision is independent of the release revisions in its entries.
    ``path`` bypasses the Hub and is mutually exclusive with repo_id/revision.
    """
    if path is not None:
        if repo_id is not None or revision is not None:
            raise ValueError(
                "A local catalog path cannot be combined with repo_id or revision"
            )
        return _read_catalog(Path(path).expanduser().resolve())
    repo_id = repo_id or os.environ.get("MORPH_CATALOG_REPO", DEFAULT_CATALOG_REPO)
    revision = revision or os.environ.get("MORPH_CATALOG_REVISION", "main")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=CATALOG_FILENAME,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
    except LocalEntryNotFoundError as exc:
        raise FileNotFoundError(
            f"No cached catalog for {repo_id}@{revision}. Run `morph models list` "
            "online first with the same catalog source and cache directory, "
            "or supply a local catalog file."
        ) from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Cannot read {repo_id}/{CATALOG_FILENAME} at {revision}. "
            "Check that the catalog exists and use `hf auth login` or HF_TOKEN "
            "if the repository requires access."
        ) from exc
    path = Path(downloaded)
    # Hub cache path is snapshots/<resolved commit>/catalog.json. Do not resolve
    # the symlink: its target is a blob, which would lose revision provenance.
    return _read_catalog(path, repo_id=repo_id, revision=path.parent.name)
