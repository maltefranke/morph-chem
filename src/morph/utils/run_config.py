"""Runtime settings, Hydra/Lightning configuration, callbacks, and run IDs."""

import faulthandler
import os
import sys
from datetime import timedelta

import hydra
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, StrategyRegistry
from rdkit import RDLogger


def configure_runtime(*, seed: int | None = None, diagnostics: bool = False) -> None:
    """Set arithmetic, chemistry logging, and optional reproducibility diagnostics."""
    torch.set_float32_matmul_precision("medium")
    RDLogger.DisableLog("rdApp.*")
    if seed is not None:
        pl.seed_everything(seed)
    if diagnostics and os.environ.get("MORPH_STALL_DUMP"):
        faulthandler.dump_traceback_later(
            int(os.environ["MORPH_STALL_DUMP"]), repeat=True, file=sys.stderr
        )
        faulthandler.enable(file=sys.stderr, all_threads=True)
    if diagnostics and os.environ.get("MORPH_FORCE_MATH_SDP"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


class ManualAccumulationWandbLogger(WandbLogger):
    """Report manual-optimization updates on W&B's standard step axis."""

    def __init__(self, *args, accumulation_steps: int = 1, **kwargs):
        if accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive")
        self._accumulation_steps = accumulation_steps
        super().__init__(*args, **kwargs)

    def log_metrics(self, metrics, step=None):
        """Log metrics against completed updates rather than microbatches."""
        # Lightning's logger counter advances per batch when its Trainer-side
        # accumulation is one. Manual optimization accumulates inside the
        # module, so convert that counter back to completed model updates.
        if step is not None:
            step //= self._accumulation_steps
        return super().log_metrics(metrics, step=step)


def validation_batches_per_rank(
    num_samples: int, batch_size: int, world_size: int
) -> int:
    """Return the per-rank batch limit for an exact global sample count."""
    if num_samples <= 0 or batch_size <= 0 or world_size <= 0:
        raise ValueError(
            "validation sample count, batch size and world size must be positive"
        )
    samples_per_distributed_batch = batch_size * world_size
    if num_samples % samples_per_distributed_batch:
        raise ValueError(
            f"num_val_samples ({num_samples}) must be divisible by val batch size "
            f"({batch_size}) * world size ({world_size})"
        )
    return num_samples // samples_per_distributed_batch


def build_trainer(
    *,
    validation_batch_size: int,
    num_val_samples: int | None = None,
    **trainer_kwargs,
) -> pl.Trainer:
    """Build a Trainer and size validation using its resolved process topology.

    Native Lightning launches set ``WORLD_SIZE`` only after constructing the
    parent Trainer. Use Lightning's device/node resolution so parent and child
    ranks get identical settings. Explicit validation limits and Lightning's
    development/overfit modes take precedence over the global sample budget.
    """
    strategy = trainer_kwargs.get("strategy", "auto")
    if isinstance(strategy, str) and strategy in StrategyRegistry:
        preset = StrategyRegistry[strategy]
        if preset["strategy"] is DDPStrategy:
            # Preserve the preset's launch method and unused-parameter policy.
            # Morph's buffers are constants shared by all ranks at construction.
            trainer_kwargs["strategy"] = DDPStrategy(
                **preset["init_params"],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                timeout=timedelta(
                    seconds=int(os.environ.get("MORPH_DDP_TIMEOUT", "1800"))
                ),
            )

    trainer = pl.Trainer(**trainer_kwargs)
    if (
        num_val_samples is not None
        and trainer_kwargs.get("limit_val_batches") is None
        and not trainer.fast_dev_run
        and not trainer.overfit_batches
    ):
        trainer.limit_val_batches = validation_batches_per_rank(
            int(num_val_samples), validation_batch_size, trainer.world_size
        )
        hydra.utils.log.info(
            f"Validation population: {int(num_val_samples)} global samples "
            f"({trainer.limit_val_batches} batches per rank)"
        )
    return trainer


def configure_manual_gradient_accumulation(module, trainer_kwargs: dict) -> None:
    """Move Lightning's accumulation setting onto a manual-optimization module.

    Lightning rejects ``Trainer(accumulate_grad_batches != 1)`` for manual
    optimization. The optimizer mixin implements the equivalent accumulation,
    while the Trainer must therefore retain its neutral value of one.
    """
    steps = int(trainer_kwargs.get("accumulate_grad_batches", 1))
    if steps <= 0:
        raise ValueError("accumulate_grad_batches must be positive")
    if module.automatic_optimization:
        return
    module.manual_gradient_accumulation_steps = steps
    trainer_kwargs["accumulate_grad_batches"] = 1


def gather_rank_outputs(outputs: list) -> list | None:
    """Gather picklable rank outputs on rank zero without replicating them."""
    if not (dist.is_available() and dist.is_initialized()):
        return outputs

    is_global_zero = dist.get_rank() == 0
    gathered = [None] * dist.get_world_size() if is_global_zero else None
    dist.gather_object(outputs, gathered, dst=0)
    if not is_global_zero:
        return None
    return [output for rank_outputs in gathered for output in rank_outputs]


def build_callbacks(cfg: DictConfig) -> list[Callback]:
    callbacks: list[Callback] = []

    if "early_stopping" in cfg.callbacks:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        es_cfg = cfg.callbacks.early_stopping
        # stopping_threshold (optional): halt the moment the monitored metric
        # crosses this value (>= for mode=max), independent of the plateau path.
        # Used to stop a run once validity reaches a target (e.g. 0.8).
        callbacks.append(
            EarlyStopping(
                monitor=cfg.callbacks.monitor_metric,
                mode=cfg.callbacks.monitor_metric_mode,
                patience=es_cfg.patience,
                verbose=es_cfg.verbose,
                stopping_threshold=es_cfg.get("stopping_threshold", None),
            )
        )

    if "model_checkpoints" in cfg.callbacks:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                monitor=cfg.callbacks.monitor_metric,
                mode=cfg.callbacks.monitor_metric_mode,
                every_n_epochs=cfg.callbacks.model_checkpoints.get("every_n_epochs", 1),
                save_top_k=cfg.callbacks.model_checkpoints.save_top_k,
                verbose=cfg.callbacks.model_checkpoints.verbose,
                save_last=cfg.callbacks.model_checkpoints.save_last,
            )
        )

    if "every_n_epochs_checkpoint" in cfg.callbacks:
        cb_cfg = cfg.callbacks.every_n_epochs_checkpoint
        hydra.utils.log.info(
            "Adding callback <ModelCheckpoint> for every "
            f"{cb_cfg.every_n_epochs} epochs"
        )
        # Keep best-by-validation separate from latest-for-resume. Lightning only saves
        # train-epoch checkpoints when save_on_train_epoch_end=True if validation is not
        # run every epoch.
        callbacks.append(
            ModelCheckpoint(
                dirpath="every_n_epochs",
                every_n_epochs=cb_cfg.every_n_epochs,
                save_top_k=cb_cfg.save_top_k,
                verbose=cb_cfg.verbose,
                save_last=cb_cfg.save_last,
                save_on_train_epoch_end=cb_cfg.get("save_on_train_epoch_end", None),
            )
        )

    # Task-specific callbacks are built generically: any ``callbacks.<name>`` that
    # carries a ``_target_`` is instantiated as-is. This module therefore imports
    # none of them, which matters because the per-task eval callbacks ship with
    # their own recipe (edit / multimode) and may not be installed at
    # all -- a hard import here would make this core module depend on them.
    # Defaults live on the callback classes, so the config need only override.
    for entry in (cfg.callbacks or {}).values():
        if not isinstance(entry, DictConfig) or "_target_" not in entry:
            continue  # the explicitly-handled callbacks above
        hydra.utils.log.info(f"Adding callback <{entry._target_.rsplit('.', 1)[-1]}>")
        callbacks.append(hydra.utils.instantiate(entry, _convert_="all"))

    return callbacks


def bootstrap_run_id() -> None:
    """Set the shared run IDs before Hydra composes ``hydra.run.dir``.

    A single process generates a W&B-style ID. Multi-process ``torchrun`` must
    receive the ID in its environment so every rank shares one output directory.
    """
    if os.environ.get("MORPH_RUN_ID"):
        os.environ.setdefault("WANDB_RUN_ID", os.environ["MORPH_RUN_ID"])
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        raise RuntimeError(
            "Multi-process runs require MORPH_RUN_ID in the environment (use the same "
            "value as WANDB_RUN_ID). Example:\n"
            "  export MORPH_RUN_ID=<shared-id>\n"
            '  export WANDB_RUN_ID="$MORPH_RUN_ID"'
        )

    rid = wandb.util.generate_id()
    os.environ["MORPH_RUN_ID"] = rid
    os.environ["WANDB_RUN_ID"] = rid
