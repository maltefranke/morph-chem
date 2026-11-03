# Private compiler configuration is loaded only when compilation is enabled.
# ruff: noqa: PLC0415

"""Model setup: compile, EMA, forward.

Mixin split out of :class:`morph.model.lightning.base.LightningModuleRates`.
"""

import os

import torch

from morph.model.layers.attention import fix_efficient_attention_bias_metadata
from morph.utils.checkpoints import canonicalize_state_dict, state_dict_for_module


class SetupMixin:
    def compile(self):
        """Compile current and EMA models in place, unless MORPH_COMPILE=0.

        Both entry points use this setup. Compilation is lazy: training and
        inference acquire the appropriate graphs when first executed. In-place
        compilation keeps module identity, parameter references and state keys.
        """
        if self.is_compiled:
            return
        if os.environ.get("MORPH_COMPILE", "1") == "0":
            print("Model compilation disabled (MORPH_COMPILE=0).")
            return
        print("Compiling current and EMA models (lazy; MORPH_COMPILE=0 disables).")
        fix_efficient_attention_bias_metadata()
        # PyTorch 2.9's DDP graph partitioner can lose FakeTensor metadata on
        # recompilation after validation. Keep this workaround opt-in.
        import torch._dynamo.config as dynamo_config

        if os.environ.get("MORPH_COMPILE_DDP_OPTIMIZER") == "0":
            dynamo_config.optimize_ddp = False
            print("DDP compiler graph splitting disabled.")
        if os.environ.get("MORPH_COMPILE_FALLBACK") == "1":
            dynamo_config.suppress_errors = True
            print("Compiler errors will warn and fall back to eager execution.")
        # Per-loss grad-norm logging calls torch.autograd.grad(..., retain_graph=True)
        # before the main backward. torch.compile's donated-buffer optimization frees
        # activations after the first backward and rejects any subsequent one, so
        # disable it when grad-norm tracking is active.
        if self.log_grad_norms_every_n_steps > 0:
            import torch._functorch.config as _functorch_config

            _functorch_config.donated_buffer = False
        for model in (self.model, self.model_ema):
            if model is not None:
                model.compile(dynamic=True)
        self.is_compiled = True

    def on_save_checkpoint(self, checkpoint):
        """Save portable parameter names independent of ``torch.compile``."""
        state_dict = checkpoint.get("state_dict")
        if state_dict is not None:
            checkpoint["state_dict"] = canonicalize_state_dict(state_dict)

    def on_load_checkpoint(self, checkpoint):
        """Adapt eager or compiled checkpoint keys to the current module."""
        state_dict = checkpoint.get("state_dict")
        if state_dict is not None:
            checkpoint["state_dict"] = state_dict_for_module(state_dict, self)
        # Lightning logs a microbatch counter. Rebase it when a resumed manual
        # run changes microbatch size/accumulation, preserving the update axis.
        if not getattr(self, "automatic_optimization", True):
            loop_state = (
                checkpoint.get("loops", {})
                .get("fit_loop", {})
                .get("epoch_loop.state_dict")
            )
            if loop_state is not None and "global_step" in checkpoint:
                loop_state["_batches_that_stepped"] = (
                    checkpoint["global_step"] * self.manual_gradient_accumulation_steps
                )

    def _get_model(self):
        """Return model or model_ema for inference based on use_ema_for_eval."""
        return (
            self.model_ema
            if self.use_ema_for_eval and self.model_ema is not None
            else self.model
        )

    def _update_ema(self):
        """Update EMA parameters: ema = decay * ema + (1 - decay) * model."""
        # In automatic optimization this hook runs just before Lightning marks
        # the optimizer step complete; manual optimization has already marked
        # both optimizer branches complete. Convert both cases to the same
        # one-based count of logical model updates.
        update_step = self._optimizer_update_step()
        if self.automatic_optimization:
            update_step += 1

        ema_decay = 0.0
        if update_step > self.ema_warmup_steps:
            ema_decay = float(self.ema_decay)
        if ema_decay and self.ema_decay_scheduler is not None:
            ema_decay = self.ema_decay_scheduler.get_decay(
                current_epoch=self.current_epoch,
            )
        with torch.no_grad():
            ema_tensors = getattr(self, "_ema_tensors", None)
            if ema_tensors is None:
                ema_params = tuple(self.model_ema.parameters())
                model_params = tuple(self.model.parameters())
                ema_buffers = tuple(self.model_ema.buffers())
                model_buffers = tuple(self.model.buffers())
                if len(ema_params) != len(model_params):
                    raise RuntimeError("EMA and training model parameter sets differ")
                if len(ema_buffers) != len(model_buffers):
                    raise RuntimeError("EMA and training model buffer sets differ")
                ema_tensors = (
                    ema_params,
                    model_params,
                    ema_buffers,
                    model_buffers,
                )
                self._ema_tensors = ema_tensors
            ema_params, model_params, ema_buffers, model_buffers = ema_tensors
            # One grouped multi-tensor recurrence instead of two launches per
            # parameter. A representative CPU transformer mix was ~22% faster;
            # CUDA benefits more from eliminating the Python/kernel-launch loop.
            if ema_decay == 0.0:
                torch._foreach_copy_(ema_params, model_params)
            else:
                torch._foreach_lerp_(ema_params, model_params, 1.0 - ema_decay)
            # Buffers are state rather than trainable weights; keep them exact,
            # including integer counters that cannot be averaged.
            if ema_buffers:
                torch._foreach_copy_(ema_buffers, model_buffers)

        self.log(
            "ema/decay",
            ema_decay,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

    def forward(self, *args, **kwargs):
        """Forward through the trainable model."""
        return self.model(*args, **kwargs)
