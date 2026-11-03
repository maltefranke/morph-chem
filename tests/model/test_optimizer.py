"""Optimizer regression tests."""

import copy
import os
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule, Trainer

import morph.model.lightning._optimizer as optimizer_module
import morph.utils.run_config as run_config
from morph.model.layers.lora import LoRALinear
from morph.model.lightning._optimizer import CudaGraphMuon, gradient_accumulation_plan
from morph.model.lightning.base import LightningModuleRates
from morph.utils.run_config import (
    ManualAccumulationWandbLogger,
    configure_manual_gradient_accumulation,
    validation_batches_per_rank,
)

from ._helpers import (
    _native_optimizer_config,
)


@pytest.mark.parametrize(("world_size", "expected_batches"), [(1, 32), (2, 16), (4, 8)])
def test_validation_population_is_independent_of_world_size(
    world_size, expected_batches
):
    assert validation_batches_per_rank(4096, 128, world_size) == expected_batches


def test_validation_population_rejects_partial_distributed_batches():
    with pytest.raises(ValueError, match="must be divisible"):
        validation_batches_per_rank(4097, 128, 2)


def test_manual_accumulation_is_removed_from_lightning_trainer():
    module = SimpleNamespace(automatic_optimization=False)
    trainer_kwargs = {"accumulate_grad_batches": 2}

    configure_manual_gradient_accumulation(module, trainer_kwargs)

    assert module.manual_gradient_accumulation_steps == 2
    assert trainer_kwargs["accumulate_grad_batches"] == 1


def test_manual_accumulation_wandb_logger_uses_optimizer_updates(monkeypatch):
    logged = {}

    def record_metrics(self, metrics, step=None):
        del self
        logged.update(metrics=metrics, step=step)

    monkeypatch.setattr(run_config.WandbLogger, "log_metrics", record_metrics)
    logger = object.__new__(ManualAccumulationWandbLogger)
    logger._accumulation_steps = 2

    logger.log_metrics({"loss/train": 1.0}, step=144_000)

    assert logged == {"metrics": {"loss/train": 1.0}, "step": 72_000}


def test_gradient_accumulation_plan_flushes_a_short_final_window():
    assert gradient_accumulation_plan(0, 2, False) == (True, False, 2)
    assert gradient_accumulation_plan(1, 2, False) == (False, True, 2)
    assert gradient_accumulation_plan(2, 2, True) == (True, True, 1)


def test_automatic_optimizer_delegates_to_the_standard_training_step():
    class Harness(LightningModuleRates):
        def __init__(self):
            LightningModule.__init__(self)
            self.automatic_optimization = True
            self.model_ema = torch.nn.Linear(1, 1)
            self.logged_grad_norms = False

        def shared_step(self, batch, batch_idx):
            del batch, batch_idx
            return torch.tensor(2.0)

        def _maybe_log_per_loss_grad_norms(self):
            self.logged_grad_norms = True

        def log(self, *args, **kwargs):
            pass

    module = Harness()

    loss = module.training_step(None, 0)

    assert loss.item() == 2.0
    assert module.logged_grad_norms


@pytest.mark.parametrize("supports_sync_block", [False, True])
@pytest.mark.parametrize(
    ("accumulation_steps", "num_batches"), [(1, 3), (2, 2), (4, 3), (4, 5), (4, 7)]
)
def test_manual_two_optimizer_accumulation_matches_one_combined_batch(
    supports_sync_block, accumulation_steps, num_batches
):
    class Harness(LightningModuleRates):
        def __init__(self):
            LightningModule.__init__(self)
            self.model = torch.nn.Linear(1, 1, bias=False)
            self.offset = torch.nn.Parameter(torch.zeros(()))
            self.model_ema = torch.nn.Linear(1, 1, bias=False)
            self.manual_gradient_accumulation_steps = accumulation_steps
            self.automatic_optimization = False
            self._optimizers = (
                torch.optim.SGD(self.model.parameters(), lr=0.1),
                torch.optim.SGD([self.offset], lr=0.1),
            )
            self._trainer = SimpleNamespace(
                is_last_batch=False,
                strategy=(
                    SimpleNamespace(block_backward_sync=nullcontext)
                    if supports_sync_block
                    else SimpleNamespace()
                ),
            )

        def optimizers(self):
            return self._optimizers

        def lr_schedulers(self):
            return []

        def shared_step(self, batch, batch_idx):
            del batch_idx
            x, y = batch
            return ((self.model(x) + self.offset - y) ** 2).mean()

        def manual_backward(self, loss):
            loss.backward()

        def log(self, *args, **kwargs):
            pass

        def _maybe_log_per_loss_grad_norms(self):
            pass

        def _update_ema(self):
            pass

    accumulated = Harness()
    combined = Harness()
    for module in (accumulated, combined):
        module.model.weight.data.zero_()

    batches = [
        (torch.tensor([[float(i + 1)]]), torch.tensor([[float(2 * i + 1)]]))
        for i in range(num_batches)
    ]
    for start in range(0, num_batches, accumulation_steps):
        window = batches[start : start + accumulation_steps]
        for offset, batch in enumerate(window):
            batch_idx = start + offset
            accumulated._trainer.is_last_batch = batch_idx == num_batches - 1
            accumulated.training_step(batch, batch_idx)
            if offset < len(window) - 1:
                torch.testing.assert_close(
                    accumulated.model.weight, combined.model.weight
                )
                torch.testing.assert_close(accumulated.offset, combined.offset)

        for optimizer in combined._optimizers:
            optimizer.zero_grad(set_to_none=True)
        x = torch.cat([batch[0] for batch in window])
        y = torch.cat([batch[1] for batch in window])
        loss = ((combined.model(x) + combined.offset - y) ** 2).mean()
        loss.backward()
        for optimizer in combined._optimizers:
            optimizer.step()

        torch.testing.assert_close(accumulated.model.weight, combined.model.weight)
        torch.testing.assert_close(accumulated.offset, combined.offset)


def test_native_two_optimizer_update_advances_global_step_once():
    class Harness(LightningModuleRates):
        def __init__(self):
            LightningModule.__init__(self)
            self.model = torch.nn.Linear(1, 1, bias=False)
            self.offset = torch.nn.Parameter(torch.zeros(()))
            self.model_ema = torch.nn.Linear(1, 1, bias=False)
            self.manual_gradient_accumulation_steps = 1
            self.automatic_optimization = False
            self.grad_norm_log_interval = 0
            self.optimizer_config = OmegaConf.create({"gradient_clip_val": 0.0})

        def configure_optimizers(self):
            optimizers = [
                torch.optim.SGD(self.model.parameters(), lr=0.1),
                torch.optim.SGD([self.offset], lr=0.1),
            ]
            return optimizers, [
                torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
                for optimizer in optimizers
            ]

        def shared_step(self, batch, batch_idx):
            del batch_idx
            x, y = batch
            return ((self.model(x) + self.offset - y) ** 2).mean()

        def _maybe_log_per_loss_grad_norms(self):
            pass

        def _update_ema(self):
            pass

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor([[1.0], [2.0]]),
            torch.tensor([[2.0], [3.0]]),
        ),
        batch_size=1,
    )
    module = Harness()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        max_epochs=1,
    )

    trainer.fit(module, train_dataloaders=loader)

    assert trainer.global_step == 2


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("backbone.blocks.0.qkv.weight", True),
        ("xattn_encoder.biased_blocks.0.qkv.weight", True),
        ("xattn_encoder.pos_proj.0.weight", False),
        ("xattn_encoder.token_proj.0.weight", False),
        ("heads.atom_type_head.weight", False),
        ("backbone.pos_out.output.weight", True),
        ("backbone.pair_bias.atom_pair_embedding.weight", False),
        ("backbone.dit.pos_out.output.weight", True),
        ("backbone.dit.pair_bias.atom_pair_embedding.weight", False),
        ("backbone.blocks.0.qkv.lora_A", False),
        ("backbone.dit.blocks.0.mlp.out_proj.lora_B", False),
    ],
)
def test_optimizer_routing(name, expected):
    matrix = torch.nn.Parameter(torch.randn(4, 4))
    assert LightningModuleRates._use_muon(name, matrix) is expected
    assert LightningModuleRates._use_muon(f"_orig_mod.{name}", matrix) is expected


def test_external_muon_uses_one_combined_optimizer():
    module = LightningModuleRates.__new__(LightningModuleRates)
    torch.nn.Module.__init__(module)
    module.model = torch.nn.ModuleDict(
        {
            "embedding_backbone": torch.nn.Linear(4, 4),
            "backbone": torch.nn.Linear(4, 4),
        }
    )
    module.loss_weight_wrapper = SimpleNamespace(
        use_learnable=False, learnable_wrapper=None
    )
    module.optimizer_config = _native_optimizer_config()
    module.optimizer_config.implementation = "external"

    configured = module.configure_optimizers()

    optimizer = configured["optimizer"]
    assert type(optimizer).__name__ == "SingleDeviceMuonWithAuxAdam"
    assert module._optimizer_group_names == ["adam_aux", "muon"]


def test_lora_only_finetune_uses_one_adamw_optimizer():
    module = LightningModuleRates.__new__(LightningModuleRates)
    torch.nn.Module.__init__(module)
    adapter = LoRALinear.from_linear(torch.nn.Linear(4, 4), rank=2, alpha=2.0)
    module.model = torch.nn.ModuleDict({"backbone": adapter})
    module.loss_weight_wrapper = SimpleNamespace(
        use_learnable=False, learnable_wrapper=None
    )
    module.optimizer_config = _native_optimizer_config()

    optimizers, schedulers = module.configure_optimizers()

    assert len(optimizers) == 1
    assert type(optimizers[0]) is torch.optim.AdamW
    assert len(schedulers) == 1
    optimized = {
        id(parameter)
        for group in optimizers[0].param_groups
        for parameter in group["params"]
    }
    assert optimized == {
        id(parameter) for parameter in adapter.parameters() if parameter.requires_grad
    }


def test_ema_copies_first_5000_logical_updates_then_decays():
    module = LightningModuleRates.__new__(LightningModuleRates)
    torch.nn.Module.__init__(module)
    module.automatic_optimization = False
    module.model = torch.nn.Linear(2, 2, bias=False)
    module.model_ema = torch.nn.Linear(2, 2, bias=False)
    module.model.register_buffer("running", torch.tensor([2.0]))
    module.model_ema.register_buffer("running", torch.tensor([float("nan")]))
    module.ema_decay = 0.9
    module.ema_warmup_steps = 5000
    module.ema_decay_scheduler = None
    module.log = lambda *args, **kwargs: None

    with torch.no_grad():
        module.model.weight.fill_(2.0)
        module.model_ema.weight.zero_()
    module._optimizer_update_step = lambda: 5000
    module._update_ema()
    cached_tensors = module._ema_tensors
    torch.testing.assert_close(module.model_ema.weight, module.model.weight)
    torch.testing.assert_close(module.model_ema.running, module.model.running)

    with torch.no_grad():
        module.model.weight.fill_(4.0)
    module._optimizer_update_step = lambda: 5001
    module._update_ema()
    assert module._ema_tensors is cached_tensors
    torch.testing.assert_close(
        module.model_ema.weight, torch.full_like(module.model_ema.weight, 2.2)
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("nesterov", [True, False])
def test_muon_execution_preserves_updates_and_resume(device, nesterov):
    if device == "cuda" and not torch.cuda.is_available():
        assert not os.environ.get("MORPH_REQUIRE_CUDA_TESTS"), (
            "Allocated GPU is unavailable"
        )
        pytest.skip("CUDA required for graph replay")
    torch.manual_seed(93)
    reference = [
        torch.nn.Parameter(torch.randn(*s, device=device))
        for s in [(32, 64), (64, 32), (16, 32)]
    ]
    candidate = [torch.nn.Parameter(p.detach().clone()) for p in reference]
    kwargs = dict(lr=0.01, momentum=0.95, nesterov=nesterov, weight_decay=0.1)
    native = torch.optim.Muon(reference, **kwargs)
    optimized = CudaGraphMuon(candidate, **kwargs)
    for step in range(5):
        for i, (a, b) in enumerate(zip(reference, candidate, strict=True)):
            # Exercise a changing active set, zero gradients, and fresh storage.
            a.grad = None if step == 2 and i == 1 else torch.randn_like(a)
            if step == 0:
                a.grad.zero_()
            b.grad = None if a.grad is None else a.grad.clone()
        for optimizer in (native, optimized):
            optimizer.param_groups[0]["lr"] = 0.01 / (step + 1)
            optimizer.param_groups[0]["weight_decay"] = 0.0 if step % 2 else 0.1
        native.step()
        optimized.step()
        for a, b in zip(reference, candidate, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(
                native.state[a]["momentum_buffer"],
                optimized.state[b]["momentum_buffer"],
                rtol=0,
                atol=0,
            )
        if step == 3:
            # Native checkpoints load into optimized execution and vice versa.
            optimized.load_state_dict(copy.deepcopy(native.state_dict()))
            native.load_state_dict(copy.deepcopy(optimized.state_dict()))


@pytest.mark.parametrize("nesterov", [True, False])
def test_muon_capture_allows_concurrent_pinned_allocation(monkeypatch, nesterov):
    """A real loader's pin thread must not invalidate optimizer graph capture."""
    if not torch.cuda.is_available():
        assert not os.environ.get("MORPH_REQUIRE_CUDA_TESTS"), (
            "Allocated GPU is unavailable"
        )
        pytest.skip("CUDA required for capture/pinned-allocation race")
    device = torch.cuda.current_device()
    torch.manual_seed(93)
    reference = [torch.nn.Parameter(torch.randn(64, 128, device="cuda"))]
    candidate = [torch.nn.Parameter(p.detach().clone()) for p in reference]
    kwargs = dict(lr=0.01, momentum=0.95, nesterov=nesterov, weight_decay=0.1)
    native = torch.optim.Muon(reference, **kwargs)
    optimized = CudaGraphMuon(candidate, **kwargs)
    ready, start, done = (threading.Event() for _ in range(3))
    errors, pinned = [], []

    def pin_memory():
        try:
            torch.cuda.set_device(device)
            pinned.append(torch.empty(1, pin_memory=True))
            ready.set()
            if not start.wait(30):
                raise TimeoutError("Optimizer capture did not start")
            # Bypass cached small blocks and force new pinned allocations.
            pinned.extend(
                torch.empty(8 * 1024 * 1024 + i * 4096, pin_memory=True)
                for i in range(8)
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=pin_memory, daemon=True)
    worker.start()
    assert ready.wait(30)
    original = optimizer_module._zeropower_via_newtonschulz
    raced = False

    def orthogonalize(*args, **kwargs):
        nonlocal raced
        if torch.cuda.is_current_stream_capturing() and not raced:
            raced = True
            start.set()
            assert done.wait(30), "Pin-memory allocation did not complete"
        return original(*args, **kwargs)

    monkeypatch.setattr(optimizer_module, "_zeropower_via_newtonschulz", orthogonalize)
    try:
        for _ in range(3):
            for a, b in zip(reference, candidate, strict=True):
                a.grad = torch.randn_like(a)
                b.grad = a.grad.clone()
            native.step()
            optimized.step()
            torch.cuda.synchronize()
            for a, b in zip(reference, candidate, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(
                    native.state[a]["momentum_buffer"],
                    optimized.state[b]["momentum_buffer"],
                    rtol=0,
                    atol=0,
                )
        assert raced
        assert not errors, errors
    finally:
        start.set()
        worker.join(timeout=30)
    assert not worker.is_alive()
