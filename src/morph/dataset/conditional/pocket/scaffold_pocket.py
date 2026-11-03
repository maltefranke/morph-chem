"""Scaffold decoration in a protein pocket, with unconditional replay."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from morph.dataset.conditional.pocket.datamodule import (
    PocketFlowMatchingWrapper,
    PocketLightningDataModule,
    pocket_train_collate_fn,
)
from morph.dataset.conditional.scaffold.wrappers import (
    FlowMatchingDatasetWrapperScaffold,
)


def scaffold_cache_directory(dataset, directory):
    """Isolate scaffold indices by raw-cache identity and checkpoint element set.

    Two filtered readers can have equal lengths but retain different molecules.
    A length-only cache check would silently assign cuts to the wrong ligands.
    Token order does not affect cuts, only which elements the reader accepts.
    """
    return str(Path(directory) / dataset.cache_namespace())


class ScaffoldPocketWrapper(
    FlowMatchingDatasetWrapperScaffold, PocketFlowMatchingWrapper
):
    """Use scaffold sources and attach the matching pocket before augmentation.

    The scaffold wrapper resolves the molecule index, then the pocket hook
    applies the interpolation's translation. The shared augmentation rotates
    scaffold, decoration targets, and pocket together. Evaluation attaches the
    pocket to the scaffold in the original ligand-centered frame.
    """


class ScaffoldPocketEvaluationDataset(Dataset):
    """Repeat fixed scaffold/pocket pairs with IDs for conditional metrics.

    Sample from the enumerated cuts with a local seed, keeping distinct pocket
    contexts separate even when their scaffold topologies coincide. Small test
    splits use every available pair. IDs identify pairs and generation attempts,
    so DDP padding can be deduplicated without discarding independent repeats.
    """

    def __init__(self, dataset, *, n_scaffolds=512, samples_per_scaffold=8, seed=42):
        if n_scaffolds < 1 or samples_per_scaffold < 1 or seed < 0:
            raise ValueError("Evaluation counts must be positive and seed nonnegative")
        if dataset._slicing_offsets is None:
            raise ValueError("Pocket scaffold evaluation requires enumerated cuts")
        self.dataset = dataset
        self.base_dataset = dataset.base_dataset
        self.samples_per_scaffold = int(samples_per_scaffold)
        self.indices = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(int(seed))
        )[: int(n_scaffolds)].tolist()

    def __len__(self):
        """Count all independent generation attempts."""
        return len(self.indices) * self.samples_per_scaffold

    def __getitem__(self, index):
        """Attach the pair ID and a distinct attempt ID to an unchanged pose."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        pair_index = self.indices[index // self.samples_per_scaffold]
        source, target = self.dataset[pair_index]
        source.add_system_property("scaffold_id", torch.tensor([pair_index]))
        source.add_system_property("sample_id", torch.tensor([index]))
        return source, target


def scaffold_pocket_train_collate_fn(batch):
    """Retain keep masks in mixed batches; replay atoms are entirely editable."""
    samples = []
    for item in batch:
        for sample in item if isinstance(item, list) else [item]:
            state = sample[0]
            if getattr(state, "keep_mask", None) is None:
                state = state.clone()
                state.add_node_property(
                    "keep_mask", torch.zeros(state.num_nodes, dtype=torch.long)
                )
                sample = (state, *sample[1:])
            samples.append(sample)
    return pocket_train_collate_fn(samples)


class ScaffoldPocketDataModule(PocketLightningDataModule):
    """Filter ligands, select scaffold cuts, and mix in unconditional pretraining."""

    wrapper_cls = ScaffoldPocketWrapper
    train_collate_fn = staticmethod(scaffold_pocket_train_collate_fn)

    def __init__(
        self,
        *args,
        scaffold_groups_dir,
        scaffold_kind="libinvent",
        enumerate_slicings=True,
        eval_panel=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scaffold_groups_dir = scaffold_groups_dir
        self.scaffold_kind = scaffold_kind
        self.enumerate_slicings = enumerate_slicings
        self.eval_panel = eval_panel

    def _wrap_pocket(self, base, stage):
        # Scaffold options belong only to the fine-tuning wrapper. Replay keeps
        # the ordinary unconditional wrapper and its normal prior/noise recipe.
        use_panel = stage != "train" and self.eval_panel is not None
        wrapped = self.wrapper_cls(
            base_dataset=base,
            scaffold_groups_dir=scaffold_cache_directory(
                base, self.scaffold_groups_dir
            ),
            scaffold_kind=self.scaffold_kind,
            enumerate_slicings=(self.enumerate_slicings or use_panel)
            and stage != "train",
            **self._wrapper_kwargs(stage),
        )
        if use_panel:
            return ScaffoldPocketEvaluationDataset(wrapped, **self.eval_panel)
        return wrapped
