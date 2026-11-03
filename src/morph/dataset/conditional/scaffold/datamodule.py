"""Lightning datamodule for scaffold decoration."""

from __future__ import annotations

import hydra

from morph.dataset.conditional.scaffold.panel import ScaffoldEvaluationDataset
from morph.dataset.conditional.scaffold.wrappers import (
    FlowMatchingDatasetWrapperScaffold,
)
from morph.dataset.core.lightning_datamodule import LightningDataModule


class ScaffoldLightningDataModule(LightningDataModule):
    """Wrap base datasets using a selectable retained-subgraph convention."""

    wrapper_cls = FlowMatchingDatasetWrapperScaffold

    def __init__(
        self,
        *args,
        scaffold_groups_dir: str,
        scaffold_kind: str = "rgroup",
        enumerate_slicings: bool = False,
        eval_panel=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._scaffold_groups_dir = scaffold_groups_dir
        self._scaffold_kind = scaffold_kind
        self._enumerate_slicings = bool(enumerate_slicings)
        self._eval_panel = eval_panel

    def _wrap_dataset(self, dataset_cfg, stage: str):
        if stage == "train" or self._eval_panel is None:
            return super()._wrap_dataset(dataset_cfg, stage)
        dataset = hydra.utils.instantiate(
            dataset_cfg.copy(),
            vocab=self.vocab,
            distributions=self.distributions,
            split=stage,
        )
        return ScaffoldEvaluationDataset(
            dataset,
            scaffold_groups_dir=self._scaffold_groups_dir,
            scaffold_kind=self._scaffold_kind,
            **self._eval_panel,
        )

    def _wrapper_kwargs(self, stage: str) -> dict:
        kwargs = super()._wrapper_kwargs(stage)
        kwargs["scaffold_groups_dir"] = self._scaffold_groups_dir
        kwargs["scaffold_kind"] = self._scaffold_kind
        kwargs["enumerate_slicings"] = self._enumerate_slicings and stage != "train"
        return kwargs

    def val_dataloader(self):
        """Retain the final partial batch when evaluating every scaffold."""
        return self._eval_dataloaders(
            self.val_datasets,
            "val",
            drop_last=self._eval_panel is None and not self._enumerate_slicings,
        )
