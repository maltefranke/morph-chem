"""Low-rank adapters for DiT linear projections."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Iterator


class LoRALinear(nn.Linear):
    """A linear layer with a trainable low-rank residual.

    Keeping ``weight`` and ``bias`` directly on the layer preserves the state
    dict layout of :class:`torch.nn.Linear`, so ordinary DiT checkpoints load
    without key rewriting. The residual starts at exactly zero.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}")
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.empty(out_features, self.rank, device=device, dtype=dtype)
        )
        self._merged = False
        self.reset_lora_parameters()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> LoRALinear:
        """Wrap ``linear`` without copying or renaming its base parameters."""
        adapter = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        adapter.weight = linear.weight
        adapter.bias = linear.bias
        if freeze_base:
            adapter.weight.requires_grad_(False)
            if adapter.bias is not None:
                adapter.bias.requires_grad_(False)
        return adapter

    def reset_lora_parameters(self) -> None:
        """Initialize the adapter so its initial residual is exactly zero."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def lora_delta(self) -> torch.Tensor:
        """Return the scaled low-rank weight update in base-weight layout."""
        return (self.lora_B @ self.lora_A) * self.scaling

    @torch.no_grad()
    def merge_lora_(self) -> LoRALinear:
        """Fold the adapter into the base weight for inference."""
        if not self._merged:
            self.weight.add_(self.lora_delta().to(dtype=self.weight.dtype))
            self._merged = True
        return self

    @torch.no_grad()
    def unmerge_lora_(self) -> LoRALinear:
        """Restore the separate base and adapter weights."""
        if self._merged:
            self.weight.sub_(self.lora_delta().to(dtype=self.weight.dtype))
            self._merged = False
        return self

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # Checkpoints always store the unmerged base weight plus the two adapter
        # matrices. This makes save/load independent of inference merge state.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self._merged:
            weight = self.weight - self.lora_delta().to(dtype=self.weight.dtype)
            destination[f"{prefix}weight"] = weight if keep_vars else weight.detach()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.unmerge_lora_()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the base projection and the unmerged low-rank residual."""
        output = F.linear(x, self.weight, self.bias)
        if self._merged:
            return output
        update = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return output + update * self.scaling


def iter_lora_layers(module: nn.Module) -> Iterator[LoRALinear]:
    """Yield all LoRA layers below ``module`` exactly once."""
    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield child


def set_lora_trainable(module: nn.Module, trainable: bool = True) -> int:
    """Set adapter gradients and return the number of adapter parameters."""
    count = 0
    for layer in iter_lora_layers(module):
        layer.lora_A.requires_grad_(trainable)
        layer.lora_B.requires_grad_(trainable)
        count += layer.lora_A.numel() + layer.lora_B.numel()
    return count


def merge_lora_(module: nn.Module) -> nn.Module:
    """Merge every low-rank adapter below ``module``."""
    for layer in iter_lora_layers(module):
        layer.merge_lora_()
    return module


def unmerge_lora_(module: nn.Module) -> nn.Module:
    """Unmerge every low-rank adapter below ``module``."""
    for layer in iter_lora_layers(module):
        layer.unmerge_lora_()
    return module
