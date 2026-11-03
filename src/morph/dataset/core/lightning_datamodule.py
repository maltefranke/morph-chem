import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from morph.dataset.core.flow_matching_wrapper import (
    FlowMatchingDatasetWrapper,
    eval_collate_fn,
    train_collate_fn,
    worker_init_fn,
)
from morph.dataset.core.representation import Representation


class LightningDataModule(pl.LightningDataModule):
    """Build flow-matching datasets and their Lightning data loaders."""

    wrapper_cls = FlowMatchingDatasetWrapper
    train_collate_fn = staticmethod(train_collate_fn)
    eval_collate_fn = staticmethod(eval_collate_fn)

    def __init__(
        self,
        vocab: DictConfig,
        distributions: DictConfig,
        datasets: DictConfig,
        interpolator: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        representation: str | Representation = Representation.MOLECULE,
        n_atoms_strategy: str = "uniform",
        uniform_n_atoms_padding: int = 5,
        time_dist: DictConfig = None,
        rotate: bool = True,
        n_augmentations: int = 1,
        noise_distributions=None,
        train_prefetch_factor: int = 1,
    ):
        super().__init__()
        self.vocab = vocab
        self.distributions = distributions
        self.datasets = datasets
        self.num_workers = num_workers
        if train_prefetch_factor < 1:
            raise ValueError("train_prefetch_factor must be positive")
        self.train_prefetch_factor = train_prefetch_factor
        self.batch_size = batch_size
        self.representation = Representation(representation)
        self.n_atoms_strategy = n_atoms_strategy
        self.uniform_n_atoms_padding = int(uniform_n_atoms_padding)
        self.rotate = rotate
        self.n_augmentations = max(1, int(n_augmentations))

        if (
            self.n_augmentations > 1
            and self.batch_size.train % self.n_augmentations != 0
        ):
            raise ValueError(
                f"batch_size.train ({self.batch_size.train}) must be divisible by "
                f"n_augmentations ({self.n_augmentations})."
            )

        if time_dist is None:
            time_dist = DictConfig(
                {
                    "_target_": "torch.distributions.Uniform",
                    "low": 0.0,
                    "high": 1.0,
                }
            )
        self.time_dist = hydra.utils.instantiate(time_dist)

        self.train_dataset = None
        self.val_datasets = None
        self.test_datasets = None
        self.interpolator = hydra.utils.instantiate(
            interpolator,
            distributions=self.distributions,
            noise_distributions=noise_distributions,
        )

    def _wrapper_kwargs(self, stage: str) -> dict:
        is_train = stage == "train"
        return {
            "distributions": self.distributions,
            "interpolator": self.interpolator,
            "vocab": self.vocab,
            "representation": self.representation,
            "n_atoms_strategy": self.n_atoms_strategy,
            "uniform_n_atoms_padding": self.uniform_n_atoms_padding,
            "time_dist": self.time_dist,
            "stage": stage,
            "rotate": self.rotate if is_train else False,
            "n_augmentations": self.n_augmentations if is_train else 1,
        }

    def _wrap_dataset(self, dataset_cfg, stage: str):
        base_dataset = hydra.utils.instantiate(
            dataset_cfg.copy(),
            vocab=self.vocab,
            distributions=self.distributions,
            split=stage,
        )
        return self.wrapper_cls(
            base_dataset=base_dataset,
            **self._wrapper_kwargs(stage),
        )

    def setup(self, stage=None):
        """Construct only the datasets required by *stage*, once."""
        if self.vocab is None:
            raise ValueError("vocab is required before setup()")

        if stage in (None, "fit") and self.train_dataset is None:
            self.train_dataset = self._wrap_dataset(self.datasets.train, "train")
        if stage in (None, "fit", "validate") and self.val_datasets is None:
            self.val_datasets = [
                self._wrap_dataset(cfg, "val") for cfg in self.datasets.val
            ]
        if stage in (None, "test", "predict") and self.test_datasets is None:
            self.test_datasets = [
                self._wrap_dataset(cfg, "test") for cfg in self.datasets.test
            ]

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Move batch to *device* with ``non_blocking=True``.

        Lightning's default tensor traversal does not recurse into Morph or PyG
        graph containers. Handle both explicitly so their tensors benefit from
        the asynchronous transfer enabled by ``pin_memory=True``.
        """

        def _move(item):
            if hasattr(item, "to_inplace"):
                return item.to_inplace(device)
            if isinstance(item, torch.Tensor) or hasattr(item, "stores"):
                return item.to(device, non_blocking=True)
            return item

        if isinstance(batch, (tuple, list)):
            return type(batch)(_move(item) for item in batch)
        return _move(batch)

    def configure_model(self, cfg):
        """Publish any dataset-derived conditioning dimensions before model creation."""

    def train_dataloader(self):
        """Return the shuffled training loader."""
        nw = self.num_workers.train
        # Each __getitem__ produces n_augmentations training items, so we
        # request batch_size.train // n_augmentations unique molecules per
        # batch and the wrapper expands them to the configured effective size.
        per_load_batch_size = self.batch_size.train // self.n_augmentations
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=per_load_batch_size,
            num_workers=nw,
            persistent_workers=nw > 0,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.train_collate_fn,
            worker_init_fn=worker_init_fn,
            prefetch_factor=self.train_prefetch_factor if nw > 0 else None,
        )

    def _eval_dataloaders(self, datasets, stage: str, drop_last: bool):
        nw = self.num_workers[stage]
        return [
            DataLoader(
                dataset,
                shuffle=False,
                sampler=self._eval_sampler(dataset),
                batch_size=self.batch_size[stage],
                num_workers=nw,
                persistent_workers=nw > 0,
                pin_memory=True,
                drop_last=drop_last,
                collate_fn=self.eval_collate_fn,
                worker_init_fn=worker_init_fn,
                prefetch_factor=2 if nw > 0 else None,
            )
            for dataset in datasets
        ]

    def val_dataloader(self):
        """Return validation loaders."""
        return self._eval_dataloaders(self.val_datasets, "val", drop_last=True)

    def _eval_sampler(self, dataset):
        """Let Lightning shard evaluation unless a subclass overrides this hook."""
        return None

    def test_dataloader(self):
        """Return test loaders."""
        return self._eval_dataloaders(self.test_datasets, "test", drop_last=False)

    def predict_dataloader(self) -> list[DataLoader]:
        """Reuse test loaders for prediction."""
        return self.test_dataloader()
