"""Prediction-only module: one network, with either EMA or current weights."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import TYPE_CHECKING

import hydra
import pytorch_lightning as pl
import torch

from morph.model.build import prepare_model_data
from morph.model.layers.attention import fix_efficient_attention_bias_metadata
from morph.model.lightning._sampling import SamplingMixin
from morph.utils.checkpoints import canonicalize_state_dict, state_dict_for_module

if TYPE_CHECKING:
    from pathlib import Path

    from omegaconf import DictConfig

    from morph.model.build import ModelData


class InferenceModule(SamplingMixin, pl.LightningModule):
    """Reuse the production sampler without constructing training or EMA copies."""

    def __init__(self, cfg: DictConfig, data: ModelData):
        super().__init__()
        self.model = hydra.utils.instantiate(cfg.model.module.model)
        self.vocab = data.vocab
        self.distributions = data.prior
        self.representation = data.representation
        self.allow_charged = data.allow_charged
        self.integrator = hydra.utils.instantiate(
            cfg.model.module.integrator, distributions=data.prior
        )
        self.weights = "ema" if cfg.model.module.use_ema_for_eval else "current"
        self.is_compiled = False

    def _get_model(self):
        return self.model

    def forward(self, *args, **kwargs):
        """Forward through the selected inference network."""
        return self.model(*args, **kwargs)

    def compile(self):
        """Compile the single network unless MORPH_COMPILE=0."""
        if not self.is_compiled and os.environ.get("MORPH_COMPILE", "1") != "0":
            fix_efficient_attention_bias_metadata()
            self.model.compile(dynamic=True)
            self.is_compiled = True


def build_inference_module(cfg):
    """Build one unconditional network from the saved config and bundle assets."""
    if cfg.model.module._target_ != "morph.model.lightning.base.LightningModuleRates":
        raise ValueError(
            "The pretrained inference loader supports unconditional models."
        )
    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)
    data = prepare_model_data(cfg, preprocessing.vocab, preprocessing.distributions)
    return InferenceModule(cfg, data), data.vocab, data.empirical


def load_inference_weights(
    model: torch.nn.Module, checkpoint_path: str | Path, *, use_ema: bool = True
) -> None:
    """Strictly load just one network from a memory-mapped Lightning checkpoint.

    The unselected network and optimizer tensor storage remain on disk. After
    copying the selected tensors into ``model``, local mappings are released.
    Missing EMA weights are an error, never a fallback to current weights.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    canonical = canonicalize_state_dict(checkpoint.get("state_dict", checkpoint))
    prefix = "model_ema." if use_ema else "model."
    selected = OrderedDict(
        (name.removeprefix(prefix), tensor)
        for name, tensor in canonical.items()
        if name.startswith(prefix)
    )
    if not selected:
        raise RuntimeError(
            f"Checkpoint has no {prefix[:-1]} weights: {checkpoint_path}"
        )
    metadata = getattr(canonical, "_metadata", None)
    if metadata is not None:
        selected._metadata = {
            ("" if name == prefix[:-1] else name.removeprefix(prefix)): value
            for name, value in metadata.items()
            if name == prefix[:-1] or name.startswith(prefix)
        }
    model.load_state_dict(state_dict_for_module(selected, model), strict=True)
    model.requires_grad_(False)
    model.eval()
