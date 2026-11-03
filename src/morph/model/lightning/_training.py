"""Lightning training orchestration over the reusable molecular objective."""

from __future__ import annotations

from typing import TYPE_CHECKING

from morph.model.losses import FlowLossMixin

if TYPE_CHECKING:
    import torch


class TrainingMixin(FlowLossMixin):
    """Run the standalone Morph forward, loss, and training diagnostics."""

    def shared_step(self, batch, batch_idx):
        """Predict once, compute the molecular objective, and log its terms."""
        if self.vocab.atom_tokens is None:
            raise ValueError(
                "vocab.atom_tokens must be supplied when constructing "
                "the training module."
            )

        mols_t, _, _, t = batch
        # Resolve the Python graph-size bound outside a compiled forward.
        preds = self.model(mols_t, t.view(-1, 1), max_seqlen=mols_t.max_seqlen)
        loss = self.loss_from_predictions(preds, batch)
        self.log_dict(self.loss_accumulator.log_dict(), prog_bar=False)
        return loss

    def _automatic_training_step(self, batch, batch_idx):
        """Compute and log a loss for Lightning's automatic optimization."""
        loss = self.shared_step(batch, batch_idx)
        self._maybe_log_per_loss_grad_norms()
        self.log("loss/train", loss.detach(), prog_bar=True)
        return loss

    def _maybe_log_per_loss_grad_norms(self) -> None:
        """Log per-loss-term gradient norms at the configured cadence.

        Runs *before* Lightning calls ``loss.backward()``, leveraging
        ``retain_graph=True`` inside ``LossAccumulator.compute_grad_norms`` so
        the autograd graph survives for the actual backward pass. The
        ``.grad`` attributes on parameters are not touched, so this is safe to
        interleave with DDP's reducer (which only fires on
        ``loss.backward()``).
        """
        every = self.log_grad_norms_every_n_steps
        if every <= 0 or (self._optimizer_update_step() % every) != 0:
            return

        norms = self.loss_accumulator.compute_grad_norms(
            self.model.parameters(),
            granularity=self.grad_norms_granularity,
            use_weighted=False,
            apply_component_weights=False,
        )
        if not norms:
            return

        component_keys = {k for keys in self.LOSS_GROUPS.values() for k in keys}
        entries: dict[str, torch.Tensor] = {}
        for key, val in norms.items():
            prefix = "grad_norm" if key in component_keys else "grad_norm_group"
            entries[f"{prefix}/{key}"] = val.detach()

        self.log_dict(
            entries,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
