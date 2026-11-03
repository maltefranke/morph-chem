"""Carry cached target logP through interpolation, augmentation, and sampling."""

import numpy as np
import torch

from morph.dataset.core.flow_matching_wrapper import FlowMatchingDatasetWrapper
from morph.dataset.core.lightning_datamodule import LightningDataModule


def _attach_logp(state, target):
    value = getattr(target, "logp", None)
    if value is None or value.numel() != 1 or not torch.isfinite(value).all():
        raise ValueError("LogP conditioning requires one finite logp per target")
    state.add_system_property("target_logp", value.reshape(1).clone())


class LogPFlowMatchingWrapper(FlowMatchingDatasetWrapper):
    """Condition the prior on the full target's absolute RDKit Crippen logP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        arrays = getattr(self.base_dataset, "_property_arrays", None)
        if arrays is not None:
            values = arrays.get("logp")
            if (
                values is None
                or values.shape != (len(self.base_dataset),)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    "LogP labels must be finite and match the processed split order. "
                    "Run scripts/preprocess_properties.py before training."
                )

    def _interpolate_train_atomic(self, source_atomic, target, *args, **kwargs):
        result = super()._interpolate_train_atomic(
            source_atomic, target, *args, **kwargs
        )
        _attach_logp(result[0], target)
        return result

    def __getitem__(self, index):
        """Attach the requested logP to evaluation priors as well as training."""
        result = super().__getitem__(index)
        if self.stage != "train":
            source, target = result
            _attach_logp(source, target)
        return result


class LogPLightningDataModule(LightningDataModule):
    """Use the ordinary molecular pipeline with one graph-level logP request."""

    wrapper_cls = LogPFlowMatchingWrapper
