"""Shared parameterization and optional trace records for sampling steps.

The integrator remains the source of transition semantics. Callbacks can inspect
the same deletion, insertion, edge, and schedule decisions without duplicating
the transition implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from morph.dataset.core.insertion_data import InsertedAtoms
    from morph.dataset.core.insertion_targets import (
        BondsBetweenInserted,
        BondsToExisting,
    )


@dataclass(frozen=True)
class ScheduleStepMasses:
    """Integrated categorical jump masses for one graph-level time step."""

    substitution: torch.Tensor
    edge_substitution: torch.Tensor
    insertion: torch.Tensor
    deletion: torch.Tensor


@dataclass(frozen=True)
class NodeEditProbabilities:
    """Mutually exclusive numerical-step edit probabilities."""

    any_edit: torch.Tensor
    deletion_given_edit: torch.Tensor


def node_edit_probabilities(
    substitution: torch.Tensor,
    deletion: torch.Tensor,
    eps: float = 1e-8,
) -> NodeEditProbabilities:
    """Combine mutually exclusive substitution and deletion probabilities."""
    raw_any = substitution + deletion
    return NodeEditProbabilities(
        any_edit=raw_any.clamp(0.0, 1.0),
        deletion_given_edit=(deletion / raw_any.clamp_min(eps)).clamp(0.0, 1.0),
    )


@dataclass(frozen=True)
class InsertionStepTrace:
    """Realized insertion edits in the pre-edit node index space."""

    spawn_node_idx: torch.Tensor
    clean_atoms: InsertedAtoms
    bonds_to_existing: BondsToExisting | None
    bonds_between_inserted: BondsBetweenInserted | None


@dataclass(frozen=True)
class IntegrationStepTrace:
    """Edits realized by one production integration step."""

    positions: torch.Tensor
    atom_type_targets: torch.Tensor
    charge_targets: torch.Tensor
    node_substitutions: torch.Tensor
    node_deletions: torch.Tensor
    edge_index: torch.Tensor
    edge_type_targets: torch.Tensor
    edge_substitutions: torch.Tensor
    insertions: torch.Tensor
    insertion: InsertionStepTrace | None


@dataclass(frozen=True)
class SamplingStepTrace:
    """Observable model outputs, edits, and states for one sampling step.

    Tensor references are valid during the callback. Consumers retaining a
    trace should detach and clone only the fields they need.
    """

    mol_t: object
    mol_t_next: object
    t: torch.Tensor
    dt: float | torch.Tensor
    preds: dict
    integration: IntegrationStepTrace
    is_final: bool


__all__ = [
    "InsertionStepTrace",
    "IntegrationStepTrace",
    "NodeEditProbabilities",
    "SamplingStepTrace",
    "ScheduleStepMasses",
    "node_edit_probabilities",
]
