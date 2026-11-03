"""Optimizer construction, routing, stepping, and gradient diagnostics."""

from __future__ import annotations

from contextlib import nullcontext

import hydra
import torch
import torch.distributed as dist
from lightning.pytorch.utilities import grad_norm
from omegaconf import OmegaConf
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

from morph._vendor.muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam


class CudaGraphMuon(torch.optim.Muon):
    """Replay native momentum/orthogonalization kernels with fixed input buffers.

    The model's variable atom counts do not change its parameter shapes. Copy
    each new gradient into a stable buffer, replay the native Muon operations,
    then apply the current scheduler learning rate outside the graph. The
    optimizer state dict remains identical to torch.optim.Muon. CPU and mixed
    device parameter sets use the native implementation directly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_graphs = {}

    def load_state_dict(self, state_dict):
        # Loading replaces momentum tensors; captured graphs own old addresses.
        self._update_graphs.clear()
        return super().load_state_dict(state_dict)

    @staticmethod
    def _capture(grads, buffers, group):
        static_grads = [grad.clone() for grad in grads]
        backup = [buffer.clone() for buffer in buffers]

        def compute_updates():
            updates = []
            for grad, buf in zip(static_grads, buffers, strict=True):
                buf.lerp_(grad, 1 - group["momentum"])
                update = grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                updates.append(
                    _zeropower_via_newtonschulz(
                        update,
                        group["ns_coefficients"],
                        group["ns_steps"],
                        group["eps"],
                    )
                )
            return updates

        stream = torch.cuda.Stream(device=grads[0].device)
        stream.wait_stream(torch.cuda.current_stream())
        # Warm library handles and allocations before stream capture. Restore
        # momentum afterwards so warm-up never counts as a training update.
        with torch.cuda.stream(stream):
            compute_updates()
            compute_updates()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        # The DataLoader pin-memory thread may call cudaHostAlloc concurrently.
        # Global capture rejects that unrelated allocation and can invalidate
        # this graph, surfacing as a cuBLAS internal error. All graph inputs
        # here are private, stable device buffers; only this thread records
        # optimizer work. Keep capture safety checks local to this thread.
        with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
            updates = compute_updates()
        torch._foreach_copy_(buffers, backup)
        return graph, static_grads, updates

    @torch.no_grad()
    def step(self, closure=None):
        devices = {p.device for group in self.param_groups for p in group["params"]}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            return super().step(closure)
        device = next(iter(devices))
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.cuda.device(device):
            for gi, group in enumerate(self.param_groups):
                params, grads, buffers = [], [], []
                self._init_group(group, params, grads, buffers)
                if not params:
                    continue
                # Gradient participation, layout and recurrence settings may
                # change, while LR and weight decay can change without capture.
                key = (
                    tuple(id(p) for p in params),
                    tuple(id(b) for b in buffers),
                    tuple((g.shape, g.stride(), g.dtype, g.device) for g in grads),
                    group["momentum"],
                    group["nesterov"],
                    tuple(group["ns_coefficients"]),
                    group["ns_steps"],
                    group["eps"],
                )
                cached = self._update_graphs.get(gi)
                if cached is None or cached[0] != key:
                    self._update_graphs.pop(gi, None)
                    cached = (key, *self._capture(grads, buffers, group))
                    self._update_graphs[gi] = cached
                _, graph, static_grads, updates = cached
                torch._foreach_copy_(static_grads, grads)
                graph.replay()
                for param, update in zip(params, updates, strict=True):
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                    param.add_(
                        update,
                        alpha=-_adjust_lr(
                            group["lr"], group["adjust_lr_fn"], param.shape
                        ),
                    )
        return loss


def gradient_accumulation_plan(
    batch_idx: int, steps: int, is_last_batch: bool
) -> tuple[bool, bool, int]:
    """Return ``(zero_first, step_now, window_size)`` for one microbatch.

    The window size is exact when stepping; otherwise it is the configured
    maximum. Normalize every backward pass by ``steps``, then correct the
    accumulated gradient if the final window turns out to be shorter.
    """
    if steps <= 0:
        raise ValueError("gradient accumulation steps must be positive")
    offset = batch_idx % steps
    step_now = offset + 1 == steps or is_last_batch
    window_size = offset + 1 if is_last_batch else steps
    return offset == 0, step_now, window_size


class OptimizerMixin:
    """Own all automatic and native Muon/AdamW optimization behavior."""

    _AUXILIARY_MATRIX_PREFIXES = (
        "embedding_backbone.",
        "heads.",
        "insertion_head.",
        "ins_edge_head.",
        # The context encoder's input feature projections play the same role as
        # ``embedding_backbone``. Its attention/MLP blocks remain on Muon.
        "xattn_encoder.pos_proj.",
        "xattn_encoder.token_proj.",
    )
    _AUXILIARY_MATRIX_PATHS = (
        ".pair_bias.atom_pair_embedding.",
        ".lora_A",
        ".lora_B",
    )

    def _setup_optimization(self) -> None:
        """Select Lightning's automatic or native multi-optimizer path."""
        implementation = self.optimizer_config.get("implementation", "torch")
        if implementation not in {"torch", "external"}:
            raise ValueError(
                "optimizer implementation must be 'torch' or 'external', got "
                f"{implementation!r}"
            )
        # A Hydra target and the legacy combined implementation each expose a
        # single optimizer, so Lightning can own their stepping natively. The
        # documented torch Muon + AdamW split requires manual optimization.
        self.automatic_optimization = (
            "_target_" in self.optimizer_config.optimizer
            or implementation == "external"
        )
        self.manual_gradient_accumulation_steps = 1

    def _optimizer_update_step(self) -> int:
        """Return completed model updates."""
        return self.global_step

    def _should_log_grad_norms(self) -> bool:
        every = self.grad_norm_log_interval
        return every > 0 and (self._optimizer_update_step() % every) == 0

    @classmethod
    def _use_muon(cls, name, parameter) -> bool:
        """Route hidden 2D matrices to Muon and auxiliary tensors to AdamW."""
        # ``torch.compile`` wraps the model and prefixes every parameter name.
        name = name.removeprefix("_orig_mod.")
        is_auxiliary = name.startswith(cls._AUXILIARY_MATRIX_PREFIXES) or any(
            path in name for path in cls._AUXILIARY_MATRIX_PATHS
        )
        return parameter.ndim == 2 and not is_auxiliary

    @staticmethod
    def _embedding_parameter_ids(model) -> set[int]:
        """Find lookup tables by module type rather than extension-specific names."""
        return {
            id(parameter)
            for module in model.modules()
            if isinstance(module, torch.nn.Embedding)
            for parameter in module.parameters(recurse=False)
        }

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        """Apply module-level clipping settings for automatic optimizers."""
        clip_val = self.optimizer_config.get("gradient_clip_val", gradient_clip_val)
        algorithm = self.optimizer_config.get(
            "gradient_clip_algorithm", gradient_clip_algorithm
        )
        self.clip_gradients(
            optimizer,
            gradient_clip_val=clip_val,
            gradient_clip_algorithm=algorithm,
        )

    def configure_optimizers(self):
        """Build one configured optimizer or native Muon plus auxiliary AdamW."""
        opt_cfg = self.optimizer_config.optimizer
        params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        learnable_loss_weights = (
            self.loss_weight_wrapper.use_learnable
            and self.loss_weight_wrapper.learnable_wrapper is not None
        )
        if learnable_loss_weights:
            params.extend(self.loss_weight_wrapper.learnable_wrapper.parameters())

        if "_target_" in opt_cfg:
            optimizer = hydra.utils.instantiate(opt_cfg, params=params)
            self._optimizer_group_names = ["all"]
            scheduler = self._make_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.optimizer_config.interval,
                    "monitor": self.optimizer_config.monitor,
                },
                "monitor": self.optimizer_config.monitor,
            }

        muon_params, adam_params = [], []
        embedding_parameter_ids = self._embedding_parameter_ids(self.model)
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            is_embedding = id(parameter) in embedding_parameter_ids
            use_muon = self._use_muon(name, parameter) and not is_embedding
            target = muon_params if use_muon else adam_params
            target.append(parameter)
        if learnable_loss_weights:
            adam_params.extend(self.loss_weight_wrapper.learnable_wrapper.parameters())

        implementation = self.optimizer_config.get("implementation", "torch")
        aux_cfg = self.optimizer_config.aux_optimizer
        if implementation == "external":
            named_groups = []
            if adam_params:
                named_groups.append(
                    (
                        "adam_aux",
                        {
                            "params": adam_params,
                            "lr": aux_cfg.lr,
                            "betas": tuple(aux_cfg.betas),
                            "eps": aux_cfg.get("eps", 1e-10),
                            "weight_decay": aux_cfg.weight_decay,
                            "use_muon": False,
                        },
                    )
                )
            if muon_params:
                named_groups.append(
                    (
                        "muon",
                        {
                            "params": muon_params,
                            "lr": opt_cfg.lr,
                            "momentum": opt_cfg.momentum,
                            "weight_decay": opt_cfg.weight_decay,
                            "use_muon": True,
                        },
                    )
                )
            if not named_groups:
                raise RuntimeError("The model has no trainable parameters")
            optimizer_cls = (
                MuonWithAuxAdam
                if dist.is_available() and dist.is_initialized()
                else SingleDeviceMuonWithAuxAdam
            )
            optimizer = optimizer_cls([group for _, group in named_groups])
            self._optimizer_group_names = [name for name, _ in named_groups]
            scheduler = self._make_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.optimizer_config.interval,
                    "monitor": self.optimizer_config.monitor,
                },
                "monitor": self.optimizer_config.monitor,
            }

        optimizers = []
        optimizer_names = []
        if muon_params:
            execution = opt_cfg.get("execution", "native")
            if execution not in {"native", "cuda_graph"}:
                raise ValueError("Muon execution must be native or cuda_graph")
            optimizer_cls = torch.optim.Muon if execution == "native" else CudaGraphMuon
            optimizers.append(
                optimizer_cls(
                    muon_params,
                    lr=opt_cfg.lr,
                    momentum=opt_cfg.momentum,
                    weight_decay=opt_cfg.weight_decay,
                    nesterov=opt_cfg.get("nesterov", True),
                    ns_steps=opt_cfg.get("ns_steps", 5),
                    eps=opt_cfg.get("eps", 1e-7),
                    adjust_lr_fn=opt_cfg.get("adjust_lr_fn", "original"),
                )
            )
            optimizer_names.append("muon")

        if adam_params:
            optimizers.append(
                torch.optim.AdamW(
                    adam_params,
                    lr=aux_cfg.lr,
                    betas=tuple(aux_cfg.betas),
                    eps=aux_cfg.get("eps", 1e-10),
                    weight_decay=aux_cfg.weight_decay,
                )
            )
            optimizer_names.append("adam_aux")
        if not optimizers:
            raise RuntimeError("The model has no trainable parameters")

        self._optimizer_group_names_by_id = {
            id(optimizer): [name]
            for optimizer, name in zip(optimizers, optimizer_names, strict=True)
        }
        return optimizers, [self._make_scheduler(optimizer) for optimizer in optimizers]

    def _make_scheduler(self, optimizer):
        scheduler_cfg = dict(
            OmegaConf.to_container(self.optimizer_config.scheduler, resolve=True)
        )
        scheduler_cfg["optimizer"] = optimizer
        return hydra.utils.instantiate(scheduler_cfg)

    def _manual_training_step(self, batch, batch_idx):
        """Accumulate gradients and step all native optimizers together."""
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = (optimizers,)
        accumulation_steps = self.manual_gradient_accumulation_steps
        zero_first, step_now, window_size = gradient_accumulation_plan(
            batch_idx,
            accumulation_steps,
            self.trainer.is_last_batch,
        )
        if zero_first:
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)

        loss = self.shared_step(batch, batch_idx)
        if step_now:
            self._maybe_log_per_loss_grad_norms()
        self.log("loss/train", loss.detach(), prog_bar=True)

        block_sync = getattr(self.trainer.strategy, "block_backward_sync", nullcontext)
        sync_context = nullcontext() if step_now else block_sync()
        with sync_context:
            self.manual_backward(loss / accumulation_steps)
        if not step_now:
            return loss

        # The final window can be shorter than requested. Correct all of its
        # gradients together, before optimizer hooks perform clipping/stepping.
        if window_size < accumulation_steps:
            with torch.no_grad():
                for parameter in self.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(accumulation_steps / window_size)

        # The two native optimizers form one model update. Step the first
        # through Lightning so precision hooks and the manual-loop progress
        # tracker run once, then step the remaining native optimizer(s) through
        # the same strategy/precision path without incrementing global_step.
        optimizers[0].step()
        for optimizer in optimizers[1:]:
            raw_optimizer = getattr(optimizer, "optimizer", None)
            if raw_optimizer is None:  # lightweight test harness / direct call
                optimizer.step()
            else:
                self.trainer.strategy.optimizer_step(
                    raw_optimizer,
                    lambda: None,
                )
        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, (list, tuple)):
            schedulers = (schedulers,)
        for scheduler in schedulers:
            scheduler.step()
        self._update_ema()
        return loss

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms and clip the shared manual gradient once."""
        if self.automatic_optimization:
            self._log_optimizer_grad_norms(optimizer)
            return

        # Both native optimizers form one logical update. Log both branches and
        # clip the full gradient vector once, immediately before the first step.
        if optimizer is not self.trainer.optimizers[0]:
            return
        self._log_optimizer_grad_norms(optimizer)
        if self._should_log_grad_norms():
            for other_optimizer in self.trainer.optimizers[1:]:
                self._log_param_group_grad_norms(other_optimizer)
        self._clip_all_gradients()

    def _log_optimizer_grad_norms(self, optimizer) -> None:
        if not self._should_log_grad_norms():
            return
        # Do not log per-parameter norms: their data-dependent key sets can
        # differ across DDP ranks and deadlock metric synchronization.
        norms = grad_norm(self, norm_type=2)
        total = norms.get("grad_2.0_norm_total")
        if total is not None:
            self.log("grad_2.0_norm_total", total, on_step=True, on_epoch=False)
        self._log_param_group_grad_norms(optimizer)

    def _log_param_group_grad_norms(self, optimizer):
        """Log one stable, unclipped L2 gradient norm per parameter group."""
        names_by_id = getattr(self, "_optimizer_group_names_by_id", {})
        group_names = names_by_id.get(
            id(optimizer), getattr(self, "_optimizer_group_names", None)
        )
        entries: dict[str, torch.Tensor] = {}
        for index, group in enumerate(optimizer.param_groups):
            name = (
                group_names[index]
                if group_names is not None and index < len(group_names)
                else f"group{index}"
            )
            sq_sum: torch.Tensor | None = None
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad_sq = parameter.grad.detach().norm(2) ** 2
                sq_sum = grad_sq if sq_sum is None else sq_sum + grad_sq
            entries[f"grad_2.0_norm_pg/{name}"] = (
                torch.zeros((), device=self.device) if sq_sum is None else sq_sum.sqrt()
            )
        if entries:
            self.log_dict(entries, on_step=True, on_epoch=False)

    def _clip_all_gradients(self) -> None:
        clip_val = float(self.optimizer_config.get("gradient_clip_val", 0.0))
        if clip_val <= 0:
            return
        parameters = (
            parameter for parameter in self.parameters() if parameter.grad is not None
        )
        algorithm = self.optimizer_config.get("gradient_clip_algorithm", "norm")
        if algorithm == "value":
            torch.nn.utils.clip_grad_value_(parameters, clip_val)
        else:
            torch.nn.utils.clip_grad_norm_(parameters, clip_val)

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        """Run an automatic optimizer step and update EMA exactly once."""
        optimizer.step(closure=optimizer_closure)
        self._update_ema()
