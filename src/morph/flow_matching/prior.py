"""Prior-graph sampling for Morph's native geometric-graph container.

The random-draw order matches the original, so the two agree under the same seed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.distributions import Categorical

from morph.dataset.core.atomic_data import ChemAtomicData
from morph.utils.graph_math import complete_graph_layout

if TYPE_CHECKING:
    from morph.dataset.core.vocab import Distributions


# Cache the (fixed) prior categoricals per distributions object with
# validate_args=False: rebuilding + simplex-validating them per __getitem__ cost
# ~6s in the dataloader profile. Construction draws no RNG -> sample stream unchanged.
_PRIOR_CAT_CACHE: dict[int, tuple] = {}


def _prior_categoricals(distributions: Distributions) -> tuple:
    key = id(distributions)
    cats = _PRIOR_CAT_CACHE.get(key)
    if cats is None:
        cats = tuple(
            Categorical(probs=p, validate_args=False)
            for p in (
                distributions.atom_type_distribution,
                distributions.edge_type_distribution,
                distributions.charge_type_distribution,
                distributions.n_atoms_distribution,
            )
        )
        _PRIOR_CAT_CACHE[key] = cats
    return cats


def sample_prior_graph(
    distributions: Distributions,
    n_atoms: int | None = None,
) -> ChemAtomicData:
    """Sample a fully-connected prior graph from the training distributions.

    Atom / charge / edge types are drawn from their categorical priors, and
    coordinates from a centered standard normal.
    """
    p_atom_types, p_edge_types, p_charge_types, p_n_atoms = _prior_categoricals(
        distributions
    )

    num_atoms = n_atoms if n_atoms is not None else int(p_n_atoms.sample())

    atom_types = p_atom_types.sample((num_atoms,))
    charge_types = p_charge_types.sample((num_atoms,))

    coord = torch.randn(num_atoms, 3)
    coord = coord - coord.mean(dim=0)

    edge_index, _, _, triu_index = complete_graph_layout(num_atoms)
    num_triu_edges = (num_atoms**2 - num_atoms) // 2
    edge_types = p_edge_types.sample((num_triu_edges,))[triu_index]

    return ChemAtomicData(
        positions=coord,
        atom_types=atom_types,
        charge_types=charge_types,
        edge_types=edge_types,
        neighbor_list=edge_index.t().contiguous(),
    )
