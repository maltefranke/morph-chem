"""Download and load complete Hugging Face model bundles without a dataset."""

# Keep downloads and CLI discovery independent of the scientific stack.
# ruff: noqa: PLC0415

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from morph.inference.bundles import (
    ASSET_FILES,
    ModelBundle,
    local_bundle,
    read_metadata,
    select_checkpoint,
)
from morph.inference.registry import ModelSpec, bundle_path, get_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from omegaconf import DictConfig

    from morph.dataset.core.vocab import Distributions, Vocab
    from morph.inference.catalog import ModelCatalog
    from morph.inference.module import InferenceModule


def download_pretrained(
    name: str | ModelSpec | None = None,
    *,
    epoch: int | None = None,
    subfolder: str | None = None,
    revision: str | None = None,
    filename: str | None = None,
    catalog: ModelCatalog | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> ModelBundle:
    """Download a checkpoint and assets from a catalog entry or HF repository.

    Unregistered repositories use ``subfolder`` and their ``metadata.json`` to
    select checkpoints. ``filename`` overrides epoch selection. A branch or tag
    is resolved once so the metadata, config and weights share a commit.
    Authentication uses ``hf auth login`` or ``HF_TOKEN`` unless supplied here.
    """
    spec = get_model(
        name,
        subfolder=subfolder,
        revision=revision,
        catalog=catalog,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    if spec.repo_id is None:
        raise ValueError("Downloading requires a ModelSpec with repo_id")
    if filename is not None:
        filename = bundle_path(filename)
        if epoch is not None:
            raise ValueError("Choose either epoch or filename, not both")
    else:
        epoch = spec.select_epoch(epoch)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    prefix = f"{spec.subfolder}/" if spec.subfolder else ""
    download_args = dict(
        repo_id=spec.repo_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
        max_workers=4,
    )
    try:
        # HF snapshots end in the resolved commit hash, including offline reads.
        # Always resolve metadata in the cache, even when exporting to local_dir.
        snapshot = snapshot_download(
            **download_args,
            revision=spec.revision,
            allow_patterns=[f"{prefix}metadata.json"],
        )
        resolved_revision = Path(snapshot).name
        metadata = read_metadata(Path(snapshot) / spec.subfolder, spec)
        epoch, selected_file = select_checkpoint(metadata, epoch, filename)
        root = snapshot_download(
            **download_args,
            revision=resolved_revision,
            local_dir=local_dir,
            allow_patterns=[
                f"{prefix}{path}" for path in (*ASSET_FILES, selected_file)
            ],
        )
        return local_bundle(
            spec,
            Path(root) / spec.subfolder,
            epoch=epoch if filename is None else None,
            filename=selected_file if filename is not None else None,
            revision=resolved_revision,
        )
    except LocalEntryNotFoundError as exc:
        raise FileNotFoundError(
            f"{spec.name} is not available in the local cache. "
            "Run `morph models download` online with the same model, subfolder "
            "and checkpoint selection first."
        ) from exc
    except HfHubHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status in {401, 403, 404}:
            raise RuntimeError(
                f"Cannot access {spec.repo_id}. Run `hf auth login` with an account "
                "that can read the repository, or set HF_TOKEN. "
                f"Requested revision: {spec.revision or 'main'}."
            ) from exc
        raise


@dataclass
class PretrainedModel:
    """Loaded model, vocabulary, empirical statistics and sampling configuration."""

    module: InferenceModule
    vocab: Vocab
    distributions: Distributions
    config: DictConfig
    bundle: ModelBundle

    def generate(
        self,
        n_mols: int = 16,
        *,
        batch_size: int | None = None,
        seed: int = 42,
        return_traj: bool = False,
    ) -> tuple[list, list[bool], Any, float]:
        """Return (trajectories, validity_flags, vocab, coordinate_std).

        Every attempted molecule is retained. Coordinates are in Angstrom.
        Each trajectory contains its final frame unless ``return_traj=True``.
        Sampling uses EMA weights by default and never loads a dataset.
        This method retains all results; use ``generate_to`` to consume batches.
        """
        from morph.inference.generate import sample_module

        return sample_module(
            self.module,
            self.vocab,
            self.config,
            n_mols=n_mols,
            batch_size=batch_size,
            seed=seed,
            return_traj=return_traj,
        )

    def generate_to(
        self,
        on_batch: Callable[[list, list[bool]], None],
        n_mols: int = 16,
        *,
        batch_size: int | None = None,
        seed: int = 42,
        return_traj: bool = False,
    ) -> int:
        """Consume CPU batches in attempt order without retaining predictions.

        Calls ``on_batch(trajectories, validity_flags)`` for each batch, including
        failures, and returns the total attempt count. Coordinates are in Angstrom.
        Use ``self.vocab`` to decode the frames. The default batch size is at most
        32; consumers must release batches to keep memory bounded.
        """
        from morph.inference.generate import sample_module_batches

        return sample_module_batches(
            self.module,
            self.config,
            on_batch,
            n_mols=n_mols,
            batch_size=batch_size,
            seed=seed,
            return_traj=return_traj,
        )


def load_pretrained(
    name: str | ModelSpec | None = None,
    *,
    epoch: int | None = None,
    subfolder: str | None = None,
    revision: str | None = None,
    filename: str | None = None,
    catalog: ModelCatalog | None = None,
    device: str = "cpu",
    weights_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
    steps: int = 100,
    use_ema: bool = True,
) -> PretrainedModel:
    """Load a catalog name, HF owner/repo ID, ModelSpec, or local bundle.

    With only ``weights_dir``, identity and available epochs come from the bundle
    metadata; the default catalog model is not consulted. ``subfolder``,
    ``revision`` and ``filename`` support unregistered Hub releases.
    Download only the selected checkpoint unless ``weights_dir`` supplies a
    local bundle (no network). ``checkpoint`` can override the weights in a local
    bundle; its architecture and vocabulary must match. Construction uses the
    saved portable config, not today's training defaults. Only the selected EMA
    or current network is constructed and loaded, with strict key/shape checks.
    No remote Python code is imported. The returned module is eager, in
    evaluation mode, and on ``device``, with gradients disabled.
    """
    if name is None and weights_dir is not None:
        spec = ModelSpec(Path(weights_dir).expanduser().name, subfolder=subfolder or "")
    else:
        spec = get_model(
            name,
            subfolder=subfolder,
            revision=revision,
            catalog=catalog,
            cache_dir=cache_dir,
            local_files_only=local_files_only or weights_dir is not None,
            token=token,
        )
    if filename is not None:
        filename = bundle_path(filename)
        if epoch is not None or checkpoint is not None:
            raise ValueError("filename cannot be combined with epoch or checkpoint")
    else:
        epoch = spec.select_epoch(epoch)
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if steps < 1:
        raise ValueError("steps must be positive")
    if checkpoint is not None and weights_dir is None:
        raise ValueError(
            "checkpoint requires weights_dir with matching inference assets"
        )
    if weights_dir is None:
        bundle = download_pretrained(
            spec,
            epoch=epoch,
            filename=filename,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
    else:
        bundle = local_bundle(
            spec, weights_dir, epoch=epoch, filename=filename, checkpoint=checkpoint
        )

    import torch
    from omegaconf import OmegaConf

    from morph.inference.module import build_inference_module, load_inference_weights

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use device='cpu' or a GPU worker.")
    cfg = OmegaConf.load(bundle.config_path)
    dataset = cfg.data.get("dataset_name")
    if (
        dataset is not None
        and str(dataset).casefold() != bundle.spec.dataset.casefold()
    ):
        raise ValueError("Saved config dataset_name does not match bundle metadata")
    cfg.data.preprocessing.root = str(bundle.directory)
    cfg.data.preprocessing.train_dataset = None
    cfg.data.preprocessing.num_workers = 0
    for field in ("atom_tokens", "edge_tokens", "charge_tokens", "distributions"):
        suffix = "pt" if field == "distributions" else "txt"
        cfg.data.preprocessing[f"{field}_path"] = str(
            bundle.directory / f"{field}.{suffix}"
        )
    cfg.integrator.integrator.num_integration_steps = steps
    cfg.model.module.use_ema_for_eval = use_ema
    cfg.trainer.trainer.accelerator = "gpu" if device == "cuda" else "cpu"
    cfg.trainer.trainer.devices = 1
    cfg.trainer.trainer.strategy = "auto"
    cfg.logging.enabled = False
    module, vocab, distributions = build_inference_module(cfg)
    load_inference_weights(module.model, bundle.checkpoint, use_ema=use_ema)
    module.eval().to(device)
    return PretrainedModel(module, vocab, distributions, cfg, bundle)
