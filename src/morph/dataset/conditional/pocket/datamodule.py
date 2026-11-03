"""Pocket conditioning and unconditional replay in a shared training batch."""

from __future__ import annotations

from functools import partial

import hydra
from omegaconf import OmegaConf

from morph.dataset.conditional.replay import ReplayLightningDataModule
from morph.dataset.core.flow_matching_wrapper import (
    FlowMatchingDatasetWrapper,
    eval_collate_fn,
    train_collate_fn,
)

# Keep the pocket encoder in the training graph for all-replay batches as well.
pocket_train_collate_fn = partial(train_collate_fn, context_names=("pocket",))
pocket_eval_collate_fn = partial(eval_collate_fn, context_names=("pocket",))


class PocketFlowMatchingWrapper(FlowMatchingDatasetWrapper):
    """Keep protein context in the interpolated ligand's translated/rotated frame."""

    def _attach_context(self, index, mol_t):
        pocket = self.base_dataset.get_pocket(index)
        shift = getattr(mol_t, "frame_shift", None)
        if shift is not None:
            pocket.x = pocket.x - shift
        mol_t.contexts["pocket"] = pocket


class PocketLightningDataModule(ReplayLightningDataModule):
    """Filtered pocket targets plus a configurable pretraining replay fraction."""

    wrapper_cls = PocketFlowMatchingWrapper
    train_collate_fn = staticmethod(pocket_train_collate_fn)
    eval_collate_fn = staticmethod(pocket_eval_collate_fn)

    def __init__(self, *args, pocket_vocab=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pocket_vocab = pocket_vocab

    def _wrap_dataset(self, dataset_cfg, stage):
        dataset_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
        base = hydra.utils.instantiate(
            dataset_cfg,
            vocab=self.vocab,
            distributions=self.distributions,
            split=stage,
            filter_by_vocab=True,
        )
        if self.pocket_vocab is None:
            self.pocket_vocab = base.get_pocket_vocab()
        base.pocket_vocab = self.pocket_vocab
        return self._wrap_pocket(base, stage)

    def _wrap_pocket(self, base, stage):
        """Attach task-specific wrappers after checkpoint filtering and token setup."""
        return self.wrapper_cls(base_dataset=base, **self._wrapper_kwargs(stage))

    def configure_model(self, cfg):
        """Save protein tokens in the run config for model construction and reuse."""
        cfg.data.pocket_vocab = OmegaConf.structured(self.pocket_vocab)

    def val_dataloader(self):
        """Keep the final batch: the task validation split may be small."""
        return self._eval_dataloaders(self.val_datasets, "val", drop_last=False)
