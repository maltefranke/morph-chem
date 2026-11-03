"""Per-graph loss records, time/component weights, and batch aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LossTerm:
    """Per-graph loss values and the graphs included in their final mean.

    ``values`` has shape ``[B, ...]``; ``graph_mask`` is an optional boolean
    tensor of shape ``[B]``. Empty supervision uses zero values and a false
    mask, retaining any required autograd dependencies in ``values``.
    """

    values: torch.Tensor
    graph_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate the batch dimension and optional boolean graph mask."""
        if self.values.ndim == 0:
            raise ValueError("LossTerm requires per-graph values, not a scalar")
        if self.graph_mask is not None:
            if self.graph_mask.dtype != torch.bool:
                raise ValueError("LossTerm graph_mask must be boolean")
            if self.graph_mask.shape != self.values.shape[:1]:
                raise ValueError("LossTerm graph_mask must have one entry per graph")


class InverseSquaredTimeLossWeighting:
    """Clipped inverse-square weighting near the flow endpoint."""

    def __init__(self, clamp_max: float = 10.0):
        self.clamp_max = float(clamp_max)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Return clipped inverse-square weights for graph times."""
        w = 1.0 / ((1.0 - t + 1e-6) ** 2)
        return torch.clamp(w, max=self.clamp_max)


class LearnableWeightedLoss(nn.Module):
    """Weight component losses using learned Kendall-Gal log variances."""

    def __init__(self, initial_weights, component_keys):
        """Create Kendall-Gal log-variance weights from heuristic weights."""
        super().__init__()
        self.component_keys = component_keys
        s_values = [-math.log(initial_weights[key]) for key in component_keys]
        self.s = nn.Parameter(torch.tensor(s_values, dtype=torch.float32))

    def forward(self, losses_dict):
        """Apply learned uncertainty weights to the available losses."""
        weights = torch.exp(-self.s)
        total_loss = 0
        for i, key in enumerate(self.component_keys):
            if key in losses_dict:
                s_i = self.s[i]
                total_loss += weights[i] * losses_dict[key] + 0.5 * s_i
        return total_loss


class UnifiedWeightedLoss(nn.Module):
    """Apply either fixed component weights or learned uncertainty weights."""

    def __init__(self, manual_weights, component_keys, use_learnable=False):
        super().__init__()
        self.use_learnable = use_learnable
        self.component_keys = component_keys

        if use_learnable:
            eps = 1e-6
            initial_weights = {
                key: max(float(manual_weights[key]), eps)
                if manual_weights[key] > 0
                else eps
                for key in component_keys
            }
            self.learnable_wrapper = LearnableWeightedLoss(
                initial_weights, component_keys
            )
        else:
            self.register_buffer(
                "manual_weights",
                torch.tensor(
                    [manual_weights[key] for key in component_keys],
                    dtype=torch.float32,
                ),
            )
            self.learnable_wrapper = None

    def forward(self, losses_dict):
        """Return the weighted sum of available loss components."""
        if self.use_learnable and self.learnable_wrapper is not None:
            return self.learnable_wrapper(losses_dict)
        total_loss = 0
        for i, key in enumerate(self.component_keys):
            if key in losses_dict:
                total_loss += self.manual_weights[i] * losses_dict[key]
        return total_loss

    def get_weights(self):
        """Return current component weights as Python floats."""
        weights = self._weight_vector().cpu().tolist()
        return dict(zip(self.component_keys, weights, strict=True))

    def _weight_vector(self) -> torch.Tensor:
        """Return detached component weights on their owning device."""
        with torch.no_grad():
            if self.use_learnable and self.learnable_wrapper is not None:
                return torch.exp(-self.learnable_wrapper.s)
            return self.manual_weights

    def get_weight_tensors(self):
        """Return component-weight views on their owning device."""
        return dict(zip(self.component_keys, self._weight_vector(), strict=True))


class LossAccumulator:
    """Per-step helper that collects raw losses, applies time and component weights,
    and builds a uniform log dict separating pure losses from fully weighted losses.

    Args:
        weight_module: ``UnifiedWeightedLoss`` instance that owns the component weights.
        groups: Mapping from group name to list of component keys.
        time_weight_modules: Mapping from GROUP NAME to a time-weighting callable.
    """

    def __init__(
        self,
        weight_module,
        groups: dict[str, list[str]],
        time_weight_modules: dict[str, callable] | None = None,
    ):
        self._weight_module = weight_module
        self._groups = groups
        self._time_weight_modules = time_weight_modules or {}

        # Reverse mapping: component_key -> group_name for fast lookups
        self._key_to_group = {
            key: group for group, keys in groups.items() for key in keys
        }

        # Track pure and time-weighted losses separately
        self._raw_losses: dict[str, torch.Tensor] = {}
        self._tw_losses: dict[str, torch.Tensor] = {}

        self._stats: dict[str, torch.Tensor | float] = {}
        self._current_time_weights: dict[str, torch.Tensor | float] = {}

    # ── Population ───────────────────────────────────────────────────

    def begin_batch(self) -> None:
        """Discard diagnostics left by the previous batch."""
        self._stats.clear()

    def set_batch_losses(
        self,
        batch_losses: dict[str, LossTerm],
        t: torch.Tensor | None = None,
    ):
        """Apply per-graph time weights and masks, then reduce each named term."""
        self._raw_losses.clear()
        self._tw_losses.clear()
        self._current_time_weights.clear()

        # Precompute time-weight tensors
        group_tw_tensors = {}
        if t is not None:
            for group, tw_module in self._time_weight_modules.items():
                tw = tw_module(t)
                group_tw_tensors[group] = tw
                # Keep on-device (detach only); .item() would force a per-step
                # GPU->CPU sync. Lightning offloads it lazily at log time.
                self._current_time_weights[group] = tw.mean().detach()

        for key, term in batch_losses.items():
            loss_b = term.values
            if t is not None and loss_b.shape[0] != t.shape[0]:
                raise ValueError(f"Loss {key!r} must have one row per graph time")
            # 1. Apply time-weights to create a weighted copy
            if t is not None:
                group = self._key_to_group.get(key)
                if group is not None and group in group_tw_tensors:
                    tw = group_tw_tensors[group]
                    target_shape = [-1] + [1] * (loss_b.ndim - 1)
                    loss_tw_b = loss_b * tw.view(*target_shape)
                else:
                    loss_tw_b = loss_b
            else:
                loss_tw_b = loss_b

            # 2. Apply masking if provided
            mask = term.graph_mask
            if mask is not None:
                mask_f = mask.reshape(-1, *([1] * (loss_b.ndim - 1))).to(loss_b.dtype)
                per_row = loss_b.numel() // loss_b.shape[0]
                denom = (mask.sum() * per_row).clamp(min=1)
                self._raw_losses[key] = (loss_b * mask_f).sum() / denom
                self._tw_losses[key] = (loss_tw_b * mask_f).sum() / denom
            else:
                self._raw_losses[key] = loss_b.mean()
                self._tw_losses[key] = loss_tw_b.mean()

    def set_losses(self, losses: dict[str, torch.Tensor]):
        """Fallback for pre-reduced scalar losses."""
        self._raw_losses = losses
        self._tw_losses = losses.copy()
        self._current_time_weights.clear()

    def add_stat(self, key: str, value):
        """Record a diagnostic value without adding it to the objective."""
        self._stats[key] = value

    def add_stats(self, stats: dict):
        """Record diagnostic values without adding them to the objective."""
        self._stats.update(stats)

    # ── Gradient diagnostics ─────────────────────────────────────────

    def compute_grad_norms(
        self,
        parameters,
        *,
        granularity: str = "component",
        norm_type: float = 2.0,
        use_weighted: bool = False,
        apply_component_weights: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Per-loss-term gradient norms w.r.t. ``parameters``.

        For each loss term (or group), computes ``torch.autograd.grad`` against
        ``parameters`` with ``retain_graph=True`` so the graph remains intact
        for the subsequent ``loss.backward()`` call. The ``.grad`` attribute of
        each parameter is *not* mutated.

        Cost scales linearly with the number of terms reported (one extra
        backward each), so callers should rate-limit invocation.

        Args:
            parameters: Iterable of parameters to differentiate against.
            granularity: ``"component"`` for per-key norms, ``"group"`` for
                per-group sums of raw losses, or ``"both"``.
            norm_type: p-norm used to reduce gradient tensors (default L2).
            use_weighted: If ``True``, differentiate the time-weighted loss
                ``self._tw_losses[k]``; otherwise use the raw loss
                ``self._raw_losses[k]``.
            apply_component_weights: If ``True``, multiply each term by its
                current component weight (matches what ``total_loss`` actually
                backprops). Useful to gauge effective gradient contribution.

        Returns:
            Mapping ``{key_or_group: scalar grad-norm tensor}``. Keys for
            losses that don't depend on any parameter (or whose graph was
            detached) report ``0.0``.
        """
        if granularity not in {"component", "group", "both"}:
            raise ValueError(
                f"granularity must be one of component/group/both, got {granularity!r}"
            )

        params = [p for p in parameters if p.requires_grad]
        losses = self._tw_losses if use_weighted else self._raw_losses

        if apply_component_weights:
            current_weights = self._weight_module.get_weight_tensors()
            losses = {
                k: float(current_weights.get(k, 1.0)) * v for k, v in losses.items()
            }

        terms: dict[str, torch.Tensor] = {}
        if granularity in ("component", "both"):
            terms.update(losses)
        if granularity in ("group", "both"):
            for group, keys in self._groups.items():
                summands = [losses[k] for k in keys if k in losses]
                if not summands:
                    continue
                terms[f"__group__{group}"] = sum(summands)

        norms: dict[str, torch.Tensor] = {}
        if not params or not terms:
            return norms

        zero = next(iter(losses.values())).new_zeros(())
        for key, loss in terms.items():
            out_key = (
                key.removeprefix("__group__") if key.startswith("__group__") else key
            )
            if not (isinstance(loss, torch.Tensor) and loss.requires_grad):
                norms[out_key] = zero
                continue
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                allow_unused=True,
                create_graph=False,
            )
            present = [g.detach() for g in grads if g is not None]
            if not present:
                norms[out_key] = zero
                continue
            stacked = torch.stack([g.norm(norm_type) ** norm_type for g in present])
            norms[out_key] = stacked.sum() ** (1.0 / norm_type)

        return norms

    # ── Outputs ──────────────────────────────────────────────────────

    def total_loss(self) -> torch.Tensor:
        """Apply component weights to time-weighted losses for backpropagation."""
        return self._weight_module(self._tw_losses)

    def log_dict(self) -> dict[str, torch.Tensor | float]:
        """Builds a uniform log dict separating pure and fully weighted losses."""
        entries: dict[str, torch.Tensor | float] = {}

        current_weights = self._weight_module.get_weight_tensors()

        total_pure = 0.0
        total_fully_weighted = 0.0

        # Keep values on-device (detach only); Lightning offloads/reduces them
        # lazily. A per-step .cpu() here forces a blocking D2H sync per component.
        # 1. Log individual components
        for key in self._raw_losses:
            pure_val = self._raw_losses[key].detach()

            # Use current_weights here
            weight_k = current_weights.get(key, 1.0)
            fully_weighted_val = (weight_k * self._tw_losses[key]).detach()

            entries[f"loss/{key}"] = pure_val
            entries[f"loss_weighted/{key}"] = fully_weighted_val

            total_pure += pure_val
            total_fully_weighted += fully_weighted_val

        # 2. Log group sums
        for group, keys in self._groups.items():
            pure_terms = [
                self._raw_losses[k].detach() for k in keys if k in self._raw_losses
            ]
            weighted_terms = [
                (current_weights.get(k, 1.0) * self._tw_losses[k]).detach()
                for k in keys
                if k in self._tw_losses
            ]

            if pure_terms:
                entries[f"loss/{group}"] = sum(pure_terms)
            if weighted_terms:
                entries[f"loss_weighted/{group}"] = sum(weighted_terms)

        # 3. Log totals
        entries["loss/total"] = total_pure
        entries["loss_weighted/total"] = total_fully_weighted

        # 4. Log stats, weights, and time-weights
        for key, val in self._stats.items():
            entries[f"stats/{key}"] = val

        if getattr(self._weight_module, "use_learnable", False):
            for k, w in current_weights.items():
                entries[f"weight/{k}"] = w

        for group, tw_val in self._current_time_weights.items():
            entries[f"time_weight/{group}"] = tw_val

        return entries
