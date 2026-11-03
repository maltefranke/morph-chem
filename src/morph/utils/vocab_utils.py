"""Token vocabulary helpers and loss-weight computation."""

from copy import deepcopy

import torch
from rdkit import Chem


def z_to_atom_types(z):
    """Convert the atomic numbers to atom symbols with rdkit."""

    table = Chem.GetPeriodicTable()
    return [table.GetElementSymbol(z_i) for z_i in z]


def token_to_index(token_list, token: str):
    return token_list.index(token)


def index_to_token(token_list, index: int):
    return token_list[index]


def compute_token_weights(
    token_list: list[str],
    distribution: torch.Tensor,
    special_token_names: list[str],
    weight_alpha: float = 0.4,
    type_loss_token_weights: str = "uniform",
) -> torch.Tensor:
    """
    Compute token weights with special handling for special tokens.

    This function can be used for both node tokens (atom types) and edge tokens.
    It computes inverse frequency weights, normalizes regular tokens, and assigns
    appropriate weights to special tokens.

    Args:
        token_list: List of all tokens (e.g., atom types or edge types)
        distribution: Distribution of token types from training data.
            MUST have the same length as token_list and be aligned by index.
        special_token_names: List of special token names to handle specially
            (e.g., ["<NO_BOND>"] for edges)
        weight_alpha: Alpha parameter for weight scaling (default: 0.4)
        type_loss_token_weights: "uniform" or "training" - if "uniform", returns
            uniform weights (all 1.0)

    Returns:
        Weights tensor with same shape as distribution
    """
    # Validate inputs
    if len(distribution) != len(token_list):
        raise ValueError(
            f"Distribution length ({len(distribution)}) must match "
            f"token_list length ({len(token_list)})"
        )

    # Get indices for special tokens (with validation)
    special_token_indices = set()
    for token_name in special_token_names:
        if token_name not in token_list:
            print(f"Warning: Special token '{token_name}' not in token_list, skipping")
            continue
        idx = token_to_index(token_list, token_name)
        special_token_indices.add(idx)

    # --- Handle uniform weights case early ---
    if type_loss_token_weights == "uniform":
        return torch.ones_like(distribution)

    # --- 1. Initial Weight Calculation (Inverse Frequency) ---
    epsilon = 1e-8
    weights = 1.0 / (distribution + epsilon)
    weights = weights**weight_alpha  # Apply alpha scaling

    # --- 2. Isolate & Normalize REGULAR Tokens ---
    all_indices = set(range(len(token_list)))
    regular_token_indices = list(all_indices - special_token_indices)

    if not regular_token_indices:
        return torch.ones_like(weights)

    # Get weights for only the regular tokens
    regular_weights = weights[regular_token_indices]

    # Normalize *only* the regular weights to have a mean of 1.0
    mean_regular_weight = regular_weights.mean()
    if mean_regular_weight > 0:
        regular_weights = regular_weights / mean_regular_weight

    # --- 3. Build Final Weights Tensor ---
    # Start with the full weights (including special tokens computed from distribution)
    final_weights = weights.clone()

    # Assign normalized regular weights to their correct positions
    final_weights[regular_token_indices] = regular_weights

    # Handle special tokens: use their actual inverse frequency weight,
    # normalized the same way as regular tokens
    for token_name in special_token_names:
        if token_name not in token_list:
            continue
        token_idx = token_to_index(token_list, token_name)
        special_weight = weights[token_idx]
        if mean_regular_weight > 0:
            special_weight = special_weight / mean_regular_weight
        final_weights[token_idx] = special_weight

    # set the minimum weight to 1.0
    # final_weights = final_weights / final_weights.min()

    return final_weights


def init_uniform_prior(distributions):
    """Return a new Distributions with uniform atom, edge, and charge type
        distributions.

    All other fields (n_atoms_distribution, coordinate_std) are copied from the
    original. The original object is not modified.
    """
    prior = deepcopy(distributions)
    for attr in (
        "atom_type_distribution",
        "edge_type_distribution",
        "charge_type_distribution",
    ):
        dist = torch.ones_like(getattr(prior, attr))
        setattr(prior, attr, dist / dist.sum())
    return prior


_PRIOR_MODES = ("uniform", "marginal", "absorbing")


def build_token_prior(
    distributions,
    *,
    edge_mode: str = "uniform",
    atom_mode: str = "uniform",
    charge_mode: str = "uniform",
):
    """Return a new Distributions with per-token prior modes.

    Generalises :func:`init_uniform_prior` (``all modes == "uniform"`` reproduces
    it exactly). The prior is the flow-matching *source*: it seeds
    ``sample_prior_graph`` and the interpolator, so changing it changes the whole
    transport path the denoiser is trained on.

    Modes per token channel:
      * ``uniform``   -- maximum entropy (the historical default; for edges this
                        is why the prior graph starts massively over-bonded).
      * ``marginal``  -- the data marginal already on ``distributions`` (DiGress-
                        style; the source is close to the data, so far fewer edits).
      * ``absorbing`` -- one-hot on token index 0 (``<NO_BOND>`` for edges): an
                        empty-graph / masked source with zero deletions to make.
    """
    prior = deepcopy(distributions)
    for attr, mode in (
        ("atom_type_distribution", atom_mode),
        ("edge_type_distribution", edge_mode),
        ("charge_type_distribution", charge_mode),
    ):
        cur = getattr(prior, attr)
        if mode == "uniform":
            dist = torch.ones_like(cur)
        elif mode == "marginal":
            dist = cur.clamp_min(0).clone()  # keep the data marginal already present
        elif mode == "absorbing":
            dist = torch.zeros_like(cur)
            dist[0] = 1.0
        else:
            raise ValueError(
                f"prior mode for {attr} must be one of {_PRIOR_MODES}, got {mode!r}"
            )
        setattr(prior, attr, dist / dist.sum())
    return prior
