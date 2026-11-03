"""Exercise native Lightning launch and partial accumulation on two CPU ranks."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from pytorch_lightning.plugins.environments import LightningEnvironment

from morph.model.lightning._optimizer import OptimizerMixin
from morph.utils.run_config import build_trainer


class _AccumulationHarness(OptimizerMixin, LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.offset = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("constant", torch.ones(()))
        self.automatic_optimization = False
        self.manual_gradient_accumulation_steps = 4
        self.grad_norm_log_interval = 0
        self.optimizer_config = OmegaConf.create({"gradient_clip_val": 1.0})
        self.ema_updates = 0
        self.validation_batches = 0

    def configure_optimizers(self):
        optimizers = [
            torch.optim.SGD([self.weight], lr=0.1),
            torch.optim.SGD([self.offset], lr=0.1),
        ]
        return optimizers, [
            torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
            for optimizer in optimizers
        ]

    def shared_step(self, batch, batch_idx):
        x, y = batch
        return ((self.weight * x + self.offset - y) ** 2).mean()

    def training_step(self, batch, batch_idx):
        return self._manual_training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.validation_batches += 1

    def _maybe_log_per_loss_grad_norms(self):
        pass

    def _update_ema(self):
        self.ema_updates += 1

    def on_train_start(self):
        ddp = self.trainer.strategy.model
        self.ddp_settings = {
            "broadcast_buffers": ddp.broadcast_buffers,
            "gradient_as_bucket_view": ddp.gradient_as_bucket_view,
        }


def _run_distributed_training(output_dir):
    trainer = build_trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp_find_unused_parameters_false",
        plugins=[LightningEnvironment()],
        validation_batch_size=1,
        num_val_samples=4,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    # Six distinct examples give three microbatches per rank: the final window
    # is shorter than the configured four, and its gradients require clipping.
    x = torch.arange(1, 7, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, x), batch_size=1
    )
    module = _AccumulationHarness()
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    (output_dir / f"rank-{trainer.global_rank}.json").write_text(
        json.dumps(
            {
                "world_size": trainer.world_size,
                "limit_val_batches": trainer.limit_val_batches,
                "validation_batches": module.validation_batches,
                "global_step": trainer.global_step,
                "ema_updates": module.ema_updates,
                "scheduler_steps": [s.last_epoch for s in module.lr_schedulers()],
                "weight": module.weight.item(),
                "offset": module.offset.item(),
                **module.ddp_settings,
            }
        )
    )


def test_native_two_process_launch_matches_combined_batch(tmp_path):
    env = os.environ.copy()
    for key in (
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MORPH_DDP_TIMEOUT="60")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # Independent combined-batch reference, including the same norm clipping.
    weight = torch.nn.Parameter(torch.zeros(()))
    offset = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([weight, offset], lr=0.1)
    x = torch.arange(1, 7, dtype=torch.float32)
    ((weight * x + offset - x) ** 2).mean().backward()
    torch.nn.utils.clip_grad_norm_([weight, offset], 1.0)
    optimizer.step()

    for rank in range(2):
        result = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert result.pop("weight") == pytest.approx(weight.item())
        assert result.pop("offset") == pytest.approx(offset.item())
        assert result == {
            "world_size": 2,
            "limit_val_batches": 2,
            "validation_batches": 2,
            "global_step": 1,
            "ema_updates": 1,
            "scheduler_steps": [1, 1],
            "broadcast_buffers": False,
            "gradient_as_bucket_view": True,
        }


if __name__ == "__main__":
    _run_distributed_training(Path(sys.argv[1]))
