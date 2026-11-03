"""Partial-optimal-transport alignment for flow matching.

Two layers in one module:
  * the data-structure-agnostic assignment maths (cost matrices, Hungarian
    solving and MCS constraints) in pure ``numpy`` / ``scipy``;
  * thin wrappers operating on
    :class:`~morph.dataset.core.atomic_data.ChemAtomicData`.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from morph.dataset.core.atomic_data import ChemAtomicData
from morph.utils.graph_math import rigid_alignment


def _pairwise_cost(x0, x1, squared=True):
    """Pairwise ground cost between two point sets.

    Squared Euclidean distance (W2 ground cost) by default; Euclidean distance
    (W1 ground cost) when ``squared=False``. The W2 cost is consistent with both the
    flow-matching straight-line interpolant and the Kabsch (RMSD) alignment, so
    the permutation and the rotation then descend the same energy.
    """
    diff = x0[:, np.newaxis] - x1[np.newaxis, :]
    if squared:
        return np.einsum("ijd,ijd->ij", diff, diff)
    return np.linalg.norm(diff, axis=2)


def distance_based_assignment(x0, x1, squared=True):
    """
    Assign targets from x1 to x0 using the Hungarian algorithm.

    Args:
        x0 (np.ndarray): Shape (N_x0, D)
        x1 (np.ndarray): Shape (N_x1, D)
        squared (bool): Use squared-Euclidean (W2) ground cost instead of
            Euclidean (W1).

    Returns:
        tuple: A tuple of two arrays:
        - row_ind (np.ndarray): Shape (N_x0,)
        - col_ind (np.ndarray): Shape (N_x1,)
    """
    # Calculate the cost matrix *only* for valid items
    # (N_valid, 1, D) - (1, M_valid, D) => (N_valid, M_valid, D)
    cost_matrix_b = _pairwise_cost(x0, x1, squared=squared)
    # cost_matrix_b shape is (N_x0, N_x1)

    # Perform the assignment
    # row_ind and col_ind are indices *into* x0 and x1
    row_ind, col_ind = linear_sum_assignment(cost_matrix_b)
    # K_b = len(row_ind)

    return row_ind, col_ind


def pre_align_positions(x0, x1, num_iterations=2, squared=True):
    """Pre-align positions by iterating assignment and Kabsch alignment.

    This is the equivariant-OT inner loop (Klein et al. 2023): alternating the
    assignment and the rigid (Kabsch) alignment jointly optimizes permutation
    *and* rotation. With ``squared=True`` both steps minimize the same energy
    (sum of matched squared distances), so the alternation is well-posed.
    """
    for _ in range(num_iterations):
        row_ind, col_ind = distance_based_assignment(x0, x1, squared=squared)
        x0_tensor = torch.from_numpy(x0)
        x1_tensor = torch.from_numpy(x1)
        R, t = rigid_alignment(x0_tensor[row_ind], x1_tensor[col_ind])
        x0 = (x0_tensor @ R.T + t).numpy()

    return x0, x1


def distance_and_class_based_assignment(
    x0,
    x1,
    a0,
    a1,
    c_move=1.0,
    c_sub=0.0,
    c_ins=6.125,
    c_del=6.125,
    squared=True,
):
    """
    Assign targets from x1 to x0 for a single graph pair using the Hungarian
    algorithm on an augmented (move / substitute / insert / delete) cost matrix.

    Special cases:
    1) Distance-based assignment
    c_move=const, c_sub=0, c_ins>>c_move

    2) Class-based assignment.
    Note: This will not be useful for our application!
    c_move=0, c_sub=const, c_ins>>c_sub

    3) Pure insertion-deletion assignment - no move.
    c_move>>c_ins, c_sub=const, c_ins=const

    Args:
        x0 (np.ndarray): Shape (N_x0, D)
        x1 (np.ndarray): Shape (N_x1, D)
        a0 (np.ndarray): Shape (N_x0, C)
        a1 (np.ndarray): Shape (N_x1, C)
        c_move (float): Distance cost weight
        c_sub (float): Class cost weight
        c_ins (float): Insertion cost weight
        c_del (float): Deletion cost weight
        squared (bool): If True, use squared-Euclidean (W2) move costs.

    Returns:
        tuple: A tuple of two arrays:
        - row_ind (np.ndarray): Shape (N_x0,)
        - col_ind (np.ndarray): Shape (N_x1,)
    """
    N = x0.shape[0]
    M = x1.shape[0]

    if N == 0 or M == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # 1. Calculate Real Costs. With ``squared`` the move term is squared-Euclidean
    #    (W2); c_ins/c_del are then in squared-distance units (so c_ins + c_del is
    #    the squared move-distance horizon beyond which del+ins beats a move).
    dist_matrix = _pairwise_cost(x0, x1, squared=squared)
    cost_real = c_move * dist_matrix
    if c_sub:
        class_diff = (a0[:, np.newaxis] != a1[np.newaxis, :]).astype(float)
        cost_real = cost_real + c_sub * class_diff

    # 2. Construct Augmented Cost Matrix (N+M, N+M)
    total_size = N + M
    aug_cost = np.zeros((total_size, total_size))

    # [Top-Left] Real matches (N x M)
    aug_cost[:N, :M] = cost_real

    # Real atoms may only use their own insertion/deletion dummy. The remaining
    # dummy-to-dummy block stays at the zero initialization above.
    aug_cost[:N, M:] = 1e8
    np.fill_diagonal(aug_cost[:N, M:], c_del)
    aug_cost[N:, :M] = 1e8
    np.fill_diagonal(aug_cost[N:, :M], c_ins)

    # 3. Solve
    row_ind, col_ind = linear_sum_assignment(aug_cost)

    return row_ind, col_ind


def mcs_constrained_assignment(
    x0,
    x1,
    a0,
    a1,
    mcs_pairs: list[tuple[int, int]],
    c_move=1.0,
    c_sub=0.0,
    c_ins=6.125,
    c_del=6.125,
    squared=True,
):
    """Run OT on non-MCS atoms and merge with fixed MCS pairs.

    This enforces backbone matches from ``mcs_pairs`` and computes all
    remaining assignments with the same partial OT routine used elsewhere.
    """
    N = x0.shape[0]
    M = x1.shape[0]

    # Keep only valid, one-to-one MCS pairs.
    valid_pairs = []
    used_src = set()
    used_tgt = set()
    for i_s, i_t in mcs_pairs:
        if not (0 <= i_s < N and 0 <= i_t < M):
            continue
        if i_s in used_src or i_t in used_tgt:
            continue
        valid_pairs.append((i_s, i_t))
        used_src.add(i_s)
        used_tgt.add(i_t)

    unmatched_src = [i for i in range(N) if i not in used_src]
    unmatched_tgt = [j for j in range(M) if j not in used_tgt]
    n_u = len(unmatched_src)
    m_u = len(unmatched_tgt)

    # Pair list in the augmented global index spaces:
    # source indices in [0, N+M), target indices in [0, M+N)
    global_pairs: list[tuple[int, int]] = []

    # 1) Force MCS real-real pairs.
    for i_s, i_t in valid_pairs:
        global_pairs.append((i_s, i_t))

    # 2) Pair corresponding MCS dummies with each other.
    #    source dummy for target i_t is N + i_t
    #    target dummy for source i_s is M + i_s
    for i_s, i_t in valid_pairs:
        global_pairs.append((N + i_t, M + i_s))

    # 3) Solve the reduced OT for non-MCS atoms only.
    if n_u > 0 and m_u > 0:
        row_u, col_u = distance_and_class_based_assignment(
            x0[np.array(unmatched_src)],
            x1[np.array(unmatched_tgt)],
            a0[np.array(unmatched_src)],
            a1[np.array(unmatched_tgt)],
            c_move=c_move,
            c_sub=c_sub,
            c_ins=c_ins,
            c_del=c_del,
            squared=squared,
        )

        for r_loc, c_loc in zip(row_u.tolist(), col_u.tolist(), strict=True):
            # Map reduced augmented indices to full augmented indices.
            if r_loc < n_u:
                r_glob = unmatched_src[r_loc]
            else:
                r_glob = N + unmatched_tgt[r_loc - n_u]

            if c_loc < m_u:
                c_glob = unmatched_tgt[c_loc]
            else:
                c_glob = M + unmatched_src[c_loc - m_u]

            global_pairs.append((r_glob, c_glob))
    elif n_u == 0 and m_u > 0:
        # Only insertions remain (all unmatched targets).
        global_pairs.extend((N + j, j) for j in unmatched_tgt)
    elif n_u > 0 and m_u == 0:
        # Only deletions remain (all unmatched sources).
        global_pairs.extend((i, M + i) for i in unmatched_src)

    if len(global_pairs) != N + M:
        raise ValueError(
            "MCS-constrained assignment did not produce a complete matching "
            f"(got {len(global_pairs)}, expected {N + M})."
        )

    row_ind = np.array([p[0] for p in global_pairs], dtype=int)
    col_ind = np.array([p[1] for p in global_pairs], dtype=int)

    # Safety checks: augmented assignment must be a permutation on both sides.
    if len(np.unique(row_ind)) != (N + M) or len(np.unique(col_ind)) != (N + M):
        raise ValueError("MCS-constrained assignment produced duplicate indices.")

    return row_ind, col_ind


# ----------------------------------------------------------------------
# ChemAtomicData-coupled wrappers
# ----------------------------------------------------------------------
def _align_graph(
    data: ChemAtomicData,
    permutation: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
) -> ChemAtomicData:
    """Align real nodes with zero-valued auxiliaries in one allocation."""
    num_nodes = data.num_nodes
    is_real = permutation < num_nodes
    source_idx = permutation[is_real]
    output_idx = is_real.nonzero(as_tuple=True)[0]

    def _align(value):
        if value is None:
            return None
        aligned = value.new_zeros((permutation.numel(), *value.shape[1:]))
        aligned[output_idx] = value[source_idx]
        return aligned

    node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=data.device)
    node_map[source_idx] = output_idx
    neighbor_list = None if data.neighbor_list is None else node_map[data.neighbor_list]

    extra = dict(data.extra)
    for key in data.node_extra_keys:
        if isinstance(extra.get(key), torch.Tensor):
            extra[key] = _align(extra[key])
    if neighbor_list is not None and neighbor_list.numel() > 0:
        for key in data.edge_extra_keys:
            if isinstance(extra.get(key), torch.Tensor):
                extra[key] = extra[key].clone()

    is_auxiliary = _align(data.is_auxiliary)
    if is_auxiliary is None:
        is_auxiliary = torch.zeros(
            (permutation.numel(), 1), dtype=torch.bool, device=data.device
        )
    is_auxiliary[~is_real] = True

    return ChemAtomicData(
        positions=_align(data.positions) if positions is None else positions,
        atom_types=_align(data.atom_types),
        charge_types=_align(data.charge_types),
        edge_types=(
            data.edge_types.clone()
            if data.edge_types is not None and neighbor_list is not None
            else data.edge_types
        ),
        neighbor_list=neighbor_list,
        is_auxiliary=is_auxiliary,
        extra=extra,
        node_extra_keys=data.node_extra_keys,
        edge_extra_keys=data.edge_extra_keys,
        system_extra_keys=data.system_extra_keys,
        smiles=data.smiles,
    )


def _align_and_filter(
    sample: ChemAtomicData,
    target: ChemAtomicData,
    row_ind,
    col_ind,
    optimal_transport: str,
) -> tuple[ChemAtomicData, ChemAtomicData]:
    """Shared tail of every alignment routine.

    Align each side under the augmented assignment and drop dummy-dummy pairs.
    """
    n, m = sample.num_nodes, target.num_nodes
    row_ind = torch.as_tensor(row_ind, dtype=torch.long, device=sample.device)
    col_ind = torch.as_tensor(col_ind, dtype=torch.long, device=target.device)

    sample_is_real = row_ind < n
    target_is_real = col_ind < m
    sample_is_aux = torch.ones(row_ind.numel(), dtype=torch.bool, device=sample.device)
    target_is_aux = torch.ones(col_ind.numel(), dtype=torch.bool, device=target.device)
    if sample.is_auxiliary is None:
        sample_is_aux[sample_is_real] = False
    else:
        sample_is_aux[sample_is_real] = sample.is_auxiliary[
            row_ind[sample_is_real]
        ].squeeze(-1)
    if target.is_auxiliary is None:
        target_is_aux[target_is_real] = False
    else:
        target_is_aux[target_is_real] = target.is_auxiliary[
            col_ind[target_is_real]
        ].squeeze(-1)

    valid = ~(sample_is_aux & target_is_aux)
    sample_positions = sample.positions.new_zeros(
        (row_ind.numel(), *sample.positions.shape[1:])
    )
    sample_positions[sample_is_real] = sample.positions[row_ind[sample_is_real]]

    if optimal_transport == "equivariant":
        match_mask = ~sample_is_aux & ~target_is_aux
        if match_mask.sum() > 0:
            target_matches = target.positions[col_ind[match_mask]]
            rot, trans = rigid_alignment(sample_positions[match_mask], target_matches)
            # Preserve the established arithmetic order: transform the padded
            # matrix before dropping dummy-dummy rows. GEMM shape can otherwise
            # change the final float32 bit at machine precision.
            sample_positions = sample_positions @ rot.T + trans

    sample = _align_graph(sample, row_ind[valid], positions=sample_positions[valid])
    target = _align_graph(target, col_ind[valid])
    return sample, target


def partial_optimal_transport_single(
    sample_data: ChemAtomicData,
    target_data: ChemAtomicData,
    c_move: float = 1.0,
    c_sub: float = 0.0,
    c_ins: float = 6.125,
    c_del: float = 6.125,
    optimal_transport: str = "equivariant",
    pre_align: bool = False,
    squared: bool = True,
    pre_align_iters: int = 2,
) -> tuple[ChemAtomicData, ChemAtomicData]:
    """Single-pair partial optimal transport alignment.

    ``squared`` selects the W2 (squared-Euclidean) ground cost. ``pre_align_iters``
    > 0 runs the equivariant-OT inner loop (alternating Hungarian + Kabsch) for
    that many iterations before the final class-aware assignment; it supersedes
    the legacy ``pre_align`` boolean (which maps to 3 iterations).
    """
    x0 = sample_data.positions.detach().cpu().numpy()
    x1 = target_data.positions.detach().cpu().numpy()
    a0 = sample_data.atom_types.detach().cpu().numpy()
    a1 = target_data.atom_types.detach().cpu().numpy()

    iters = pre_align_iters if pre_align_iters > 0 else (3 if pre_align else 0)
    if iters > 0:
        x0, x1 = pre_align_positions(x0, x1, num_iterations=iters, squared=squared)

    row_ind, col_ind = distance_and_class_based_assignment(
        x0, x1, a0, a1, c_move, c_sub, c_ins, c_del, squared=squared
    )
    return _align_and_filter(
        sample_data, target_data, row_ind, col_ind, optimal_transport
    )


def scaffold_based_assignment_single(
    sample_data: ChemAtomicData,
    target_data: ChemAtomicData,
    scaffold_pairs: list[tuple[int, int]],
    c_move: float = 1.0,
    c_sub: float = 0.0,
    c_ins: float = 6.125,
    c_del: float = 6.125,
    optimal_transport: str = "equivariant",
    squared: bool = True,
) -> tuple[ChemAtomicData, ChemAtomicData]:
    """Scaffold-aware partial OT alignment.

    Locks ``scaffold_pairs`` and runs MCS-constrained OT on the remaining atoms.
    """
    x0 = sample_data.positions.detach().cpu().numpy()
    x1 = target_data.positions.detach().cpu().numpy()
    a0 = sample_data.atom_types.detach().cpu().numpy()
    a1 = target_data.atom_types.detach().cpu().numpy()

    # Kabsch pre-alignment using the scaffold anchors.
    if len(scaffold_pairs) >= 3:
        s_idxs = [p[0] for p in scaffold_pairs]
        t_idxs = [p[1] for p in scaffold_pairs]
        rot, trans = rigid_alignment(
            torch.tensor(x0[s_idxs], dtype=torch.float32),
            torch.tensor(x1[t_idxs], dtype=torch.float32),
        )
        x0 = (torch.tensor(x0, dtype=torch.float32) @ rot.T + trans).numpy()
        sample_data = sample_data.clone()
        sample_data.positions = torch.tensor(
            x0,
            dtype=sample_data.positions.dtype,
            device=sample_data.positions.device,
        )

    row_ind, col_ind = mcs_constrained_assignment(
        x0,
        x1,
        a0,
        a1,
        scaffold_pairs,
        c_move,
        c_sub,
        c_ins,
        c_del,
        squared=squared,
    )
    return _align_and_filter(
        sample_data, target_data, row_ind, col_ind, optimal_transport
    )
