"""Scaffold-decoration fine-tuning LightningModule."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.distributed as dist

from morph.model.finetune.base import LightningModuleRatesFinetune
from morph.utils.metrics import calc_metrics_
from morph.utils.metrics.scaffold import scaffold_row, summarize_scaffold_rows

if TYPE_CHECKING:
    import torch


class LightningModuleRatesScaffoldFinetune(LightningModuleRatesFinetune):
    """Scaffold-aware fine-tuning.

    On top of pretrained loading from :class:`LightningModuleRatesFinetune`,
    implements the two scaffold hooks defined on the base
    :class:`LightningModuleRates`:

    * independent discrete and coordinate hooks return ``mols_t.keep_mask``
      when their corresponding exclusion flag is enabled.
    The inference-time scaffold force-fix (``_apply_inference_edit_masks``)
    lives on the shared parent :class:`LightningModuleRatesFinetune`; this class
    adds the training-time loss exclusion.

    Retained atom identities and internal bonds are fixed. Scaffold positions
    are supervised during training and may move at inference, matching the
    original scaffold denoising protocol.

    ``freeze_keep_charges`` independently controls formal charges: fixed input
    charges are excluded from the charge loss; predicted charges are supervised
    on retained atoms even when atom substitutions remain excluded.
    """

    def __init__(
        self,
        *args,
        exclude_scaffold_from_sub_loss: bool = True,
        exclude_scaffold_from_x_loss: bool = False,
        freeze_keep_positions: bool = False,
        freeze_keep_charges: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            freeze_keep_positions=freeze_keep_positions,
            freeze_keep_charges=freeze_keep_charges,
            **kwargs,
        )
        self._exclude_scaffold_from_sub_loss = exclude_scaffold_from_sub_loss
        self._exclude_scaffold_from_x_loss = exclude_scaffold_from_x_loss

    def _node_loss_exclusion_mask(self, mols_t) -> torch.Tensor | None:
        if not self._exclude_scaffold_from_sub_loss:
            return None
        return getattr(mols_t, "keep_mask", None)

    def _position_loss_exclusion_mask(self, mols_t) -> torch.Tensor | None:
        if not self._exclude_scaffold_from_x_loss:
            return None
        return getattr(mols_t, "keep_mask", None)

    def _charge_loss_exclusion_mask(self, mols_t) -> torch.Tensor | None:
        if not self._freeze_keep_charges:
            return None
        return getattr(mols_t, "keep_mask", None)

    def on_validation_epoch_start(self):
        """Start a fresh population of scaffold-conditioned attempts."""
        super().on_validation_epoch_start()
        self._scaffold_rows = []

    def _validation_context(self, batch):
        if not hasattr(batch[0], "scaffold_id"):
            return None
        # Sampling recenters its evolving state. Preserve the exact input frame
        # before integration, moving each batch to CPU just once.
        source, target = (graph.to("cpu") for graph in batch)
        source.x *= self.distributions.coordinate_std
        target.x *= self.distributions.coordinate_std
        return source.to_data_list(), target.to_data_list()

    def _update_task_metrics(self, context, mols, rdkit_mols):
        if context is None:
            return
        sources, targets = context
        self._scaffold_rows.extend(
            scaffold_row(
                source,
                target,
                generated,
                mol,
                self.vocab,
                allow_charged=self.allow_charged,
            )
            for source, target, generated, mol in zip(
                sources, targets, mols, rdkit_mols, strict=True
            )
        )

    def on_validation_epoch_end(self):
        """Pool attempts across ranks before computing per-scaffold metrics."""
        rows = self._scaffold_rows
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            gathered = [None] * dist.get_world_size() if self.global_rank == 0 else None
            dist.gather_object(rows, gathered, dst=0)
            rows = [row for shard in gathered for row in shard] if gathered else []
        metrics = summarize_scaffold_rows(rows) if self.global_rank == 0 else {}
        if distributed:
            payload = [metrics]
            dist.broadcast_object_list(payload, src=0)
            metrics = payload[0]
        if metrics:
            self.log_dict(
                {f"val/scaffold/{key}": value for key, value in metrics.items()},
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
        # Rows and RDKit molecules are gathered in the same rank/batch order.
        # Retain the full rows until the shared chemistry evaluation consumes
        # their input geometries. The scalar task summary deduplicates separately.
        self._scaffold_rows = rows
        super().on_validation_epoch_end()
        self._scaffold_rows = []

    def _compute_chemistry_metrics(self, rdkit_mols):
        rows = self._scaffold_rows
        contexts = [row.get("posebusters_reference") for row in rows] if rows else None
        metrics = calc_metrics_(rdkit_mols, self.metrics, posebusters_contexts=contexts)
        return {
            key.replace("posebusters/scaffold_", "scaffold/posebusters/", 1)
            if key.startswith("posebusters/scaffold_")
            else key: value
            for key, value in metrics.items()
        }
