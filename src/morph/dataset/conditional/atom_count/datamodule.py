"""Atom-count-conditioned training pairs and Lightning data loaders."""

import torch

from morph.dataset.core.flow_matching_wrapper import FlowMatchingDatasetWrapper
from morph.dataset.core.lightning_datamodule import LightningDataModule


def _attach_target_count(state, target):
    state.add_system_property(
        "target_n_atoms", torch.tensor([target.num_nodes], dtype=torch.long)
    )


class AtomCountFlowMatchingWrapper(FlowMatchingDatasetWrapper):
    """Carry the full target's count through interpolation and evaluation.

    The interpolator's aligned ``mol_1`` describes only currently present atoms;
    its size is not the final molecule size. Read the count before interpolation.
    Counts include hydrogens when the dataset uses explicit hydrogens.
    """

    def _interpolate_train_atomic(self, source_atomic, target, *args, **kwargs):
        result = super()._interpolate_train_atomic(
            source_atomic, target, *args, **kwargs
        )
        _attach_target_count(result[0], target)
        return result

    def __getitem__(self, index):
        """Return a training item or evaluation pair carrying the final count."""
        result = super().__getitem__(index)
        if self.stage != "train":
            source, target = result
            _attach_target_count(source, target)
        return result


class AtomCountLightningDataModule(LightningDataModule):
    """Flow matching with a per-molecule requested final atom count."""

    wrapper_cls = AtomCountFlowMatchingWrapper
