"""KL(gen || target) distribution metrics, computed directly from the batch.

All four are marginal over every generated sample (no RDKit, no validity
conditioning) so they run in every representation that carries the field.
They share one attribute contract — ``gen_hist`` / ``target`` / ``labels`` —
so ``plotting.build_marginal_plots`` can render any of them generically. They
live in their own epoch-accumulated collection (see ``init_metrics``).
"""

import torch

from morph.dataset.core.atomic_data import ChemBatch
from morph.utils.metrics.tensor import BatchMetric


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p / p.sum().clamp(min=eps)
    q = q / q.sum().clamp(min=eps)
    return (p * (p.clamp(min=eps).log() - q.clamp(min=eps).log())).sum()


class _HistogramKL(BatchMetric):
    """Shared machinery: a fixed-length generated histogram vs a target one.

    Subclasses implement ``_indices(batch) -> LongTensor`` returning the token
    indices to bin for one update; everything else (binning, KL, plotting
    attributes) is common.
    """

    def __init__(
        self,
        target_distribution: torch.Tensor,
        labels: list[str] | None = None,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        target = target_distribution.detach().to(torch.float32)
        self.eps = eps
        self.K = int(target.numel())
        self.labels = list(labels) if labels is not None else None
        self.register_buffer("target", target / target.sum().clamp(min=eps))
        self.add_state(
            "gen_hist",
            default=torch.zeros(self.K, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def _indices(self, batch: ChemBatch) -> torch.Tensor:
        raise NotImplementedError

    def update(self, batch: ChemBatch) -> None:
        idx = self._indices(batch)
        if idx.numel() == 0:
            return
        idx = idx.clamp(0, self.K - 1)
        self.gen_hist += torch.bincount(idx, minlength=self.K).to(torch.float32)

    def compute(self) -> torch.Tensor:
        if self.gen_hist.sum() <= 0:
            return torch.tensor(0.0, device=self.gen_hist.device)
        return _kl(self.gen_hist, self.target, self.eps)


class AtomCountKL(_HistogramKL):
    """KL over per-molecule atom counts."""

    def _indices(self, batch: ChemBatch) -> torch.Tensor:
        return batch.num_nodes_per_graph


class AtomTypeKL(_HistogramKL):
    """KL over the atom-type histogram."""

    def _indices(self, batch: ChemBatch) -> torch.Tensor:
        return batch.a


class ChargeTypeKL(_HistogramKL):
    """KL over the formal-charge-token histogram."""

    def _indices(self, batch: ChemBatch) -> torch.Tensor:
        return batch.c


class EdgeTypeKL(_HistogramKL):
    """KL over upper-triangular pair edge types (incl. ``<NO_BOND>`` at index 0).

    Mirrors the training-side target built in preprocessing: for every molecule
    take the dense adjacency and read its strict upper triangle, so non-bonded
    pairs contribute ``<NO_BOND>``. PyG batches lay out each graph's nodes
    contiguously, so a graph's local indices are ``global - first_node``.
    """

    def _indices(self, batch: ChemBatch) -> torch.Tensor:
        rows, cols = batch.edge_index
        return batch.e[rows < cols]
