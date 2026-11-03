"""Molecular generation metrics; upstream attribution is in docs/development.md.

Package layout (split by dependency):
- ``tensor``      : geometric tensor metrics + bin edges/accumulators (no RDKit)
- ``distribution``: KL distribution metrics, batch-fed (no RDKit)
- ``chemistry``   : RDKit checks, valence-table checks, shared pool, PoseBusters
- ``plotting``    : GT-vs-generated marginal plots

Public surface re-exported here so ``from morph.utils.metrics import X``
keeps working unchanged.
"""

import torch
from torchmetrics import MetricCollection

from morph.utils.metrics.chemistry import (
    AtomStability,
    Diversity,
    MoleculeStability,
    Novelty,
    SanitizationValidity,
    Uniqueness,
    ValenceTableAtomStability,
    ValenceTableMoleculeStability,
    Validity,
    calc_atom_stabilities,
    calc_posebusters_metrics,
)
from morph.utils.metrics.distribution import (
    AtomCountKL,
    AtomTypeKL,
    ChargeTypeKL,
    EdgeTypeKL,
)
from morph.utils.metrics.plotting import build_marginal_plots
from morph.utils.metrics.tensor import build_batch_metrics

# Public surface consumed elsewhere via ``from morph.utils.metrics import X``.
__all__ = [
    "MetricCollection",
    "build_marginal_plots",
    "calc_atom_stabilities",
    "calc_metrics_",
    "init_metrics",
]


def init_metrics(
    train_smiles: list[str] | None = None,
    target_n_atoms_distribution: torch.Tensor | None = None,
    atom_type_distribution: torch.Tensor | None = None,
    edge_type_distribution: torch.Tensor | None = None,
    charge_type_distribution: torch.Tensor | None = None,
    atom_tokens: list[str] | None = None,
    edge_tokens: list[str] | None = None,
    charge_tokens: list[str] | None = None,
    allow_charged: bool = True,
    distributions=None,
    representation=None,
    train_smiles_are_canonical: bool = False,
):
    # Chemistry metrics consume the molecule population that validation has
    # already gathered onto rank zero. Their compute calls must therefore not
    # enter TorchMetrics collectives that the other ranks cannot join.
    chemistry_metric_kwargs = {"sync_on_compute": False}
    metrics = {
        "validity": Validity(allow_charged=allow_charged, **chemistry_metric_kwargs),
        "connected-sanitization-validity": SanitizationValidity(
            **chemistry_metric_kwargs
        ),
        "valence-table-atom-stability": ValenceTableAtomStability(
            **chemistry_metric_kwargs
        ),
        "valence-table-molecule-stability": ValenceTableMoleculeStability(
            **chemistry_metric_kwargs
        ),
        "uniqueness": Uniqueness(**chemistry_metric_kwargs),
        "diversity": Diversity(**chemistry_metric_kwargs),
        **(
            {
                "novelty": Novelty(
                    train_smiles,
                    already_canonical=train_smiles_are_canonical,
                    **chemistry_metric_kwargs,
                )
            }
            if train_smiles is not None
            else {}
        ),
    }

    # Distribution metrics: tensor-native KLs computed straight from the batch,
    # so they run in every representation. Kept in their own collection so they
    # accumulate across an entire validation epoch (rather than per batch) and
    # drive the ground-truth-vs-generated marginal plots. Edge/charge KLs are
    # only registered when the representation actually carries that field —
    # otherwise they are degenerate (all <NO_BOND> / neutral by projection) and
    # would log a meaningless 0.
    want_edges = representation is None or representation.requires_topology
    want_charges = representation is None or representation.requires_charges
    distribution_metrics: dict = {}
    tensor_metric_kwargs = {"sync_on_compute": False}
    if target_n_atoms_distribution is not None:
        distribution_metrics["atom_count_dist_kl"] = AtomCountKL(
            target_n_atoms_distribution, **tensor_metric_kwargs
        )
    if atom_type_distribution is not None and atom_tokens is not None:
        distribution_metrics["atom_type_dist_kl"] = AtomTypeKL(
            atom_type_distribution, labels=atom_tokens, **tensor_metric_kwargs
        )
    if edge_type_distribution is not None and edge_tokens is not None and want_edges:
        distribution_metrics["edge_type_dist_kl"] = EdgeTypeKL(
            edge_type_distribution, labels=edge_tokens, **tensor_metric_kwargs
        )
    if (
        charge_type_distribution is not None
        and charge_tokens is not None
        and want_charges
    ):
        distribution_metrics["charge_type_dist_kl"] = ChargeTypeKL(
            charge_type_distribution, labels=charge_tokens, **tensor_metric_kwargs
        )

    stability_metrics = {
        "rdkit-atom-stability": AtomStability(),
        "rdkit-molecule-stability": MoleculeStability(),
    }

    metrics = MetricCollection(metrics, compute_groups=False)
    stability_metrics = MetricCollection(stability_metrics, compute_groups=False)
    distribution_metrics = MetricCollection(distribution_metrics, compute_groups=False)

    # Geometric batch metrics — built from training-set target stats stored in
    # Distributions. Run in every representation. The collection may be empty if
    # Distributions lacks the geometric target stats.
    batch_metrics_dict: dict = {}
    if distributions is not None:
        batch_metrics_dict = build_batch_metrics(distributions)
    batch_metrics = MetricCollection(batch_metrics_dict, compute_groups=False)

    return metrics, stability_metrics, distribution_metrics, batch_metrics


def calc_metrics_(rdkit_mols, metrics, *, posebusters_contexts=None) -> dict:
    """Run an RDKit-mol metric collection over a list of mols and return the
    scalar results. Stability lives in its own collection with a different
    input contract (precomputed bools) — drive it separately. Atom-count KL
    runs tensor-side as ``AtomCountKL``.
    """
    # These molecules are gathered explicitly by validation. Compute each
    # metric without torchmetrics' implicit DDP synchronization: list-backed
    # states (uniqueness/diversity) cannot be reduced by torchmetrics, and a
    # rank-zero-only compute would otherwise deadlock on tensor-backed metrics.
    results = {}
    for name, metric in metrics.items():
        metric.reset()
        metric.update(rdkit_mols)
        value = metric.compute()
        results[name] = value.item() if isinstance(value, torch.Tensor) else value
    try:
        kwargs = (
            {"scaffold_contexts": posebusters_contexts}
            if posebusters_contexts is not None
            else {}
        )
        posebusters = calc_posebusters_metrics(rdkit_mols, **kwargs)
    except Exception as exc:
        # PoseBusters is diagnostic and must not discard the core chemistry
        # metrics or abort a long training run if one structural check fails.
        print(f"Error calculating PoseBusters metrics: {exc}")
    else:
        results.update(
            {f"posebusters/{name}": value for name, value in posebusters.items()}
        )
    return results
