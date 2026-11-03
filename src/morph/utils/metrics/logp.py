"""Population-weighted logP errors, with failed molecules in the score coverage."""

import math

import torch
from rdkit.Chem import Crippen
from torchmetrics import Metric


class LogPMetrics(Metric):
    """Sum errors and counts before DDP reduction, including empty-scoring ranks."""

    full_state_update = False

    def __init__(self):
        super().__init__(sync_on_compute=False)
        for name in ("absolute_error", "squared_error", "scored", "attempts"):
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")
        # List states let TorchMetrics concatenate unequal per-rank populations;
        # tensor states are stacked before reduction and require equal sizes.
        self.add_state("pair_batches", default=[], dist_reduce_fx="cat")

    @property
    def pairs(self):
        """Return paired values with a stable (N, 2) shape before/after sync."""
        if isinstance(self.pair_batches, torch.Tensor):
            return self.pair_batches.reshape(-1, 2)
        if not self.pair_batches:
            return self.absolute_error.new_empty(0, 2)
        return torch.cat(self.pair_batches).reshape(-1, 2)

    def update(self, requested, molecules):
        """Score generated molecules; unscorable outputs reduce coverage."""
        requested = requested.detach().cpu().reshape(-1)
        if len(requested) != len(molecules):
            raise ValueError("Each generated molecule requires one logP request")
        pairs = []
        for target, molecule in zip(requested.tolist(), molecules, strict=True):
            if molecule is None or molecule.GetNumAtoms() == 0:
                continue
            try:
                actual = float(Crippen.MolLogP(molecule))
            except (RuntimeError, ValueError):
                continue
            if math.isfinite(actual):
                pairs.append((target, actual))
        pairs = self.absolute_error.new_tensor(pairs).reshape(-1, 2)
        error = pairs[:, 1] - pairs[:, 0]
        self.absolute_error += error.abs().sum()
        self.squared_error += error.square().sum()
        self.scored += len(pairs)
        self.attempts += len(molecules)
        self.pair_batches.append(pairs)

    def compute(self):
        """Return MAE/RMSE over scorable outputs and coverage over all attempts."""
        denominator = self.scored.clamp_min(1)
        missing = self.absolute_error.new_tensor(float("nan"))
        return {
            "logp_mae": torch.where(
                self.scored > 0, self.absolute_error / denominator, missing
            ),
            "logp_rmse": torch.where(
                self.scored > 0, (self.squared_error / denominator).sqrt(), missing
            ),
            "logp_scored_fraction": self.scored / self.attempts.clamp_min(1),
        }

    def build_parity_plot(self):
        """Plot synchronized requested/generated pairs; call inside sync_context."""
        from matplotlib import pyplot as plt  # noqa: PLC0415

        pairs = self.pairs.detach().cpu().numpy()
        figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
        if len(pairs):
            low, high = float(pairs.min()), float(pairs.max())
            padding = max((high - low) * 0.05, 0.1)
            limits = (low - padding, high + padding)
        else:
            limits = (-1.0, 1.0)
            axis.text(
                0.5,
                0.5,
                "No scorable molecules",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        axis.scatter(pairs[:, 0], pairs[:, 1], s=10, alpha=0.4, linewidths=0)
        axis.plot(limits, limits, "--", color="0.4", label="y = x", linewidth=1)
        axis.set(
            xlabel="Requested logP",
            ylabel="Generated logP",
            xlim=limits,
            ylim=limits,
            title=f"logP parity ({len(pairs)}/{int(self.attempts.item())} scorable)",
            aspect="equal",
        )
        axis.legend(loc="upper left")
        return figure
