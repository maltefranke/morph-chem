"""Representation mode for morph training runs.

Declares what subset of (x, a, c, edge_index, e) a run should train on:
- POINTCLOUD: x, a only (c and edges projected to dummies)
- CHARGED_POINTCLOUD: x, a, c (edges projected to dummies)
- GEOMETRIC_GRAPH: x, a, edge_index, e (charges projected to neutral dummies)
- MOLECULE: full chemistry (no projection)

A dataset's Capabilities declares what fields it provides. ``validate_representation``
checks the requested mode is achievable by the dataset.

The cache and vocab are **canonical** in all modes (one preprocessing per dataset).
The projection helpers transform canonical artifacts into the model-facing view:
- ``project_molecule_to_representation``: canonical ChemAtomicData → model-facing.
- ``project_distributions_to_representation``: canonical priors → projected priors
  (used by sample_prior_graph / interpolator / integrator so sampled tokens match
  projected targets).
- ``make_uniform_type_prior``: the shared uniform categorical source used by
  training and inference, followed by the representation projection above.
"""

from dataclasses import dataclass, replace
from enum import Enum

import torch
import torch.nn.functional as F

from morph.dataset.core.atomic_data import ChemAtomicData
from morph.dataset.core.vocab import Distributions, Vocab
from morph.utils.graph_math import build_fully_connected_edge_index


class Representation(str, Enum):  # noqa: UP042 -- preserve enum string semantics
    """Chemical channels exposed to the model after dataset projection."""

    POINTCLOUD = "pointcloud"
    CHARGED_POINTCLOUD = "charged_pointcloud"
    GEOMETRIC_GRAPH = "geometric_graph"
    MOLECULE = "molecule"

    @property
    def requires_charges(self) -> bool:
        """Whether charge prediction and supervision are enabled."""
        return self in {
            Representation.CHARGED_POINTCLOUD,
            Representation.MOLECULE,
        }

    @property
    def requires_topology(self) -> bool:
        """Whether runtime edges represent chemical bonds."""
        return self in {
            Representation.GEOMETRIC_GRAPH,
            Representation.MOLECULE,
        }


@dataclass(frozen=True)
class Capabilities:
    """Chemical channels available in a dataset's canonical records."""

    provides_charges: bool
    provides_topology: bool


def validate_representation(caps: Capabilities, mode: Representation) -> None:
    """Raise if the dataset cannot satisfy the requested representation."""
    if mode.requires_charges and not caps.provides_charges:
        raise ValueError(
            f"Representation {mode} requires charges but dataset does not provide them"
        )
    if mode.requires_topology and not caps.provides_topology:
        raise ValueError(
            f"Representation {mode} requires topology but dataset does not provide it"
        )


def neutral_charge_index(vocab: Vocab) -> int:
    """Index of the neutral ``"0"`` charge token in the canonical vocab; falls back
    to literal 0 when ``"0"`` isn't a discovered token (e.g. TMQM without charges).
    The fallback is just a constant the embedding absorbs into bias.
    """
    try:
        return vocab.charge_tokens.index("0")
    except (ValueError, AttributeError):
        return 0


def project_molecule_to_representation(
    mol: ChemAtomicData, vocab: Vocab, mode: Representation
) -> ChemAtomicData:
    """Project canonical ``ChemAtomicData`` onto the run's representation.

    Mutates and returns ``mol`` so any extra attributes (e.g. ``y``) are preserved.

    - non-topology modes: ``neighbor_list = fully_connected(N)``, ``edge_types =
      zeros(E)`` (token 0 of canonical edge vocab is ``<NO_BOND>`` by construction).
    - non-charge modes: ``charge_types = neutral_idx(N)`` (canonical index of ``"0"``).
    - molecule: unchanged.
    """
    n = mol.positions.shape[0]
    device = mol.positions.device

    if not mode.requires_topology:
        edge_index = build_fully_connected_edge_index(n, device=device)
        mol.neighbor_list = edge_index.t().contiguous()
        mol.edge_types = torch.zeros(
            mol.neighbor_list.shape[0], dtype=torch.long, device=device
        )

    if not mode.requires_charges:
        mol.charge_types = torch.full(
            (n,), neutral_charge_index(vocab), dtype=torch.long, device=device
        )

    return mol


def project_distributions_to_representation(
    distributions: Distributions, vocab: Vocab, mode: Representation
) -> Distributions:
    """Project canonical (or uniform) distributions onto the run's representation.

    Mirrors ``project_molecule_to_representation`` for the priors that
    ``sample_prior_graph``, the interpolator's internal Categoricals, and the
    integrator's insertion sampler draw from. Pins ``edge_type_distribution`` to
    one-hot at the canonical ``<NO_BOND>`` token (index 0) in non-topology modes,
    and ``charge_type_distribution`` to one-hot at the neutral charge index in
    non-charge modes. Other fields pass through untouched.
    """
    new = replace(distributions)
    if not mode.requires_topology:
        new.edge_type_distribution = F.one_hot(
            torch.tensor(0), len(vocab.edge_tokens)
        ).float()
    if not mode.requires_charges:
        new.charge_type_distribution = F.one_hot(
            torch.tensor(neutral_charge_index(vocab)), len(vocab.charge_tokens)
        ).float()
    return new


def make_uniform_type_prior(
    distributions: Distributions,
    vocab: Vocab,
    mode: Representation,
) -> Distributions:
    """Build the representation-aware uniform categorical flow source.

    Atom, edge, and charge tokens have equal probability before projection.
    Channels absent from the selected representation are then pinned to their
    inert token. Other statistics, including atom counts and coordinate scale,
    remain shared with the canonical training statistics.
    """
    prior = replace(
        distributions,
        atom_type_distribution=torch.full_like(
            distributions.atom_type_distribution,
            1.0 / distributions.atom_type_distribution.numel(),
        ),
        edge_type_distribution=torch.full_like(
            distributions.edge_type_distribution,
            1.0 / distributions.edge_type_distribution.numel(),
        ),
        charge_type_distribution=torch.full_like(
            distributions.charge_type_distribution,
            1.0 / distributions.charge_type_distribution.numel(),
        ),
    )
    return project_distributions_to_representation(prior, vocab, mode)
