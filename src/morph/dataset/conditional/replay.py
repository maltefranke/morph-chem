"""Sample fine-tuning and unconditional replay independently of corpus sizes."""

from __future__ import annotations

import math

import hydra
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from morph.dataset.core.flow_matching_wrapper import (
    FlowMatchingDatasetWrapper,
    worker_init_fn,
)
from morph.dataset.core.lightning_datamodule import LightningDataModule


class ReplayDataset(Dataset):
    """Route sampler indices into the two already-wrapped training datasets."""

    def __init__(self, finetune, pretrain=None):
        self.finetune, self.pretrain = finetune, pretrain
        # Existing training metrics measure novelty against the downstream task.
        self.base_dataset = finetune.base_dataset

    def __len__(self):
        """Count accessible examples across the two sources."""
        return len(self.finetune) + (
            len(self.pretrain) if self.pretrain is not None else 0
        )

    def __getitem__(self, index):
        """Read a source-local example selected by ReplaySampler."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < len(self.finetune):
            return self.finetune[index]
        return self.pretrain[index - len(self.finetune)]


class ReplaySampler(Sampler):
    """Epoch-seeded, distributed sampling with replacement from two sources.

    ``finetune_fraction`` is the probability of selecting the fine-tuning
    source, then a molecule is sampled uniformly within that source. All ranks
    generate the same stream and take disjoint strided positions. Like a
    DistributedSampler, epoch length is rounded up to a multiple of world size.
    """

    def __init__(
        self,
        finetune_size,
        pretrain_size,
        finetune_fraction,
        epoch_size,
        seed=42,
        num_replicas=1,
        rank=0,
    ):
        if not 0 <= finetune_fraction <= 1:
            raise ValueError("finetune_fraction must be in [0, 1]")
        if epoch_size < 1 or num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError(
                "epoch_size and num_replicas must be positive, with a valid rank"
            )
        if finetune_fraction > 0 and finetune_size < 1:
            raise ValueError("No compatible fine-tuning molecules remain")
        if finetune_fraction < 1 and pretrain_size < 1:
            raise ValueError(
                "Unconditional replay requires a nonempty pretraining dataset"
            )
        self.finetune_size, self.pretrain_size = finetune_size, pretrain_size
        self.finetune_fraction = finetune_fraction
        self.num_samples = math.ceil(epoch_size / num_replicas)
        self.num_replicas, self.rank = num_replicas, rank
        self.seed, self.epoch = int(seed), 0

    def set_epoch(self, epoch):
        """Select the next reproducible stream (called by Lightning each epoch)."""
        self.epoch = int(epoch)

    def __len__(self):
        """Count draws for this rank in one epoch."""
        return self.num_samples

    def __iter__(self):
        """Generate this rank's portion of the shared epoch stream."""
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        size = self.num_samples * self.num_replicas
        fine = torch.rand(size, generator=generator) < self.finetune_fraction
        fine_indices = torch.randint(
            max(1, self.finetune_size), (size,), generator=generator
        )
        pre_indices = self.finetune_size + torch.randint(
            max(1, self.pretrain_size), (size,), generator=generator
        )
        indices = torch.where(fine, fine_indices, pre_indices)
        return iter(indices[self.rank :: self.num_replicas].tolist())


class ReplayLightningDataModule(LightningDataModule):
    """Mix conditional training items with ordinary unconditional flow matching."""

    def __init__(
        self,
        *args,
        pretrain_dataset=None,
        finetune_fraction=0.5,
        train_epoch_size=None,
        mix_seed=42,
        **kwargs,
    ):
        if not 0 <= finetune_fraction <= 1:
            raise ValueError("finetune_fraction must be in [0, 1]")
        if train_epoch_size is not None and train_epoch_size < 1:
            raise ValueError("train_epoch_size must be positive")
        if finetune_fraction < 1 and pretrain_dataset is None:
            raise ValueError("pretrain_dataset is required when finetune_fraction < 1")
        super().__init__(*args, **kwargs)
        self.pretrain_dataset_cfg = pretrain_dataset
        self.finetune_fraction = float(finetune_fraction)
        self.train_epoch_size, self.mix_seed = train_epoch_size, int(mix_seed)

    def setup(self, stage=None):
        """Build replay from pretraining TRAIN only; evaluation remains task-only."""
        super().setup(stage)
        if stage not in (None, "fit") or isinstance(self.train_dataset, ReplayDataset):
            return
        pretrain = None
        if self.finetune_fraction < 1:
            base = hydra.utils.instantiate(
                self.pretrain_dataset_cfg.copy(),
                vocab=self.vocab,
                distributions=self.distributions,
                split="train",
            )
            pretrain = FlowMatchingDatasetWrapper(
                base_dataset=base, **self._wrapper_kwargs("train")
            )
        self.train_dataset = ReplayDataset(self.train_dataset, pretrain)
        # Fail during setup, before workers or training are started.
        self._replay_sampler()

    def _replay_sampler(self):
        fine, pre = self.train_dataset.finetune, self.train_dataset.pretrain
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        return ReplaySampler(
            len(fine),
            len(pre) if pre is not None else 0,
            self.finetune_fraction,
            self.train_epoch_size
            or (len(fine) if self.finetune_fraction > 0 else len(pre)),
            seed=self.mix_seed,
            num_replicas=torch.distributed.get_world_size() if distributed else 1,
            rank=torch.distributed.get_rank() if distributed else 0,
        )

    def train_dataloader(self):
        """Use replay sampling with Lightning's sampler replacement disabled."""
        nw = self.num_workers.train
        return DataLoader(
            self.train_dataset,
            sampler=self._replay_sampler(),
            batch_size=self.batch_size.train // self.n_augmentations,
            num_workers=nw,
            persistent_workers=nw > 0,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.train_collate_fn,
            worker_init_fn=worker_init_fn,
            prefetch_factor=self.train_prefetch_factor if nw > 0 else None,
        )

    def _eval_sampler(self, dataset):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return DistributedSampler(dataset, shuffle=False)
        return None
