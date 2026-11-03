"""Flow-matching interpolation for Morph geometric graphs.

Rate targets are registered as node-, edge-, or graph-level extras on
``ChemAtomicData``. Ragged insertion supervision lives in the dedicated
``InsertionTargets`` dataclass.

The interpolation maths follows the original implementation, except that charge
is intentionally a clean prediction target rather than a dynamic state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from morph.dataset.core.atomic_data import ChemAtomicData
from morph.dataset.core.insertion_targets import (
    BondsBetweenInserted,
    BondsToExisting,
    InsertionTargets,
)
from morph.flow_matching.assignment import (
    partial_optimal_transport_single,
    scaffold_based_assignment_single,
)
from morph.flow_matching.scaffold_insertion import scaffold_insertion_candidates
from morph.flow_matching.schedules import (
    KappaSchedule,
    SmoothstepSchedule,
)
from morph.utils.graph_math import complete_graph_layout

if TYPE_CHECKING:
    from morph.dataset.core.vocab import Distributions


def existing_graph_layout(mask_exists, full_edge_index):
    """Return the node map, retained complete edges, and source-edge selection."""
    keep = mask_exists.nonzero(as_tuple=True)[0]
    node_map = torch.full(
        (mask_exists.numel(),), -1, dtype=torch.long, device=mask_exists.device
    )
    node_map[keep] = torch.arange(keep.numel(), device=mask_exists.device)
    remapped = node_map[full_edge_index]
    selected = (remapped[0] >= 0) & (remapped[1] >= 0)
    return node_map, remapped[:, selected].t().contiguous(), selected


def _nearest_insertion_per_node(spawn_node_idx, spawn_dist, num_nodes):
    """Invert the one-to-many ``insertion -> nearest node`` map into a one-to-one
    ``node -> nearest insertion`` assignment.

    For every existing node that is the nearest node of at least one insertion,
    return ``(node_idx, ins_idx)`` where ``ins_idx`` is that node's *single
    closest* assigned insertion. Each node id appears at most once.

    Args:
        spawn_node_idx: ``[N_ins]`` nearest existing node per inserted atom.
        spawn_dist: ``[N_ins]`` distance of each insertion to its nearest node.
        num_nodes: number of existing nodes (size of the scatter target).
    """
    if spawn_node_idx.numel() == 0:
        empty = spawn_node_idx.new_empty((0,))
        return empty, empty.clone()

    # Order insertions by ascending distance; the earliest occurrence of a node
    # id in that order is its nearest insertion. A stable sort makes ties (which
    # are measure-zero for continuous coordinates anyway) deterministic.
    order = torch.argsort(spawn_dist, stable=True)
    nodes_in_order = spawn_node_idx[order]
    ranks = torch.arange(nodes_in_order.numel(), device=spawn_node_idx.device)

    sentinel = nodes_in_order.numel()
    earliest = torch.full(
        (num_nodes,), sentinel, dtype=torch.long, device=spawn_node_idx.device
    )
    earliest.scatter_reduce_(0, nodes_in_order, ranks, reduce="amin", include_self=True)

    has_ins = earliest < sentinel
    nearest_node_idx = torch.nonzero(has_ins, as_tuple=False).squeeze(1)
    nearest_ins_idx = order[earliest[nearest_node_idx]]
    return nearest_node_idx, nearest_ins_idx


class Interpolator:
    """Flow-matching interpolator operating on :class:`ChemAtomicData`."""

    def __init__(
        self,
        distributions: Distributions,
        optimal_transport="equivariant",
        move_noise_scale=0.5,
        ins_noise_scale=0.65,
        ins_schedule: KappaSchedule | None = None,
        del_schedule: KappaSchedule | None = None,
        sub_schedule: KappaSchedule | None = None,
        sub_e_schedule: KappaSchedule | None = None,
        c_move=1.0,
        c_sub=0.0,
        c_ins=6.125,
        c_del=6.125,
        pre_align_iters: int = 2,
        atom_type_noise_prob: float = 0.0,
        noise_distributions: Distributions | None = None,
        jitter_scale: float = 0.1,
        insertion_assignment: str = "nearest",
        include_prior_context: bool = False,
    ):
        if insertion_assignment not in {"nearest", "substituent"}:
            raise ValueError("insertion_assignment must be 'nearest' or 'substituent'")
        self.insertion_assignment = insertion_assignment
        self.include_prior_context = include_prior_context
        self.optimal_transport = optimal_transport
        self.atom_type_noise_prob = float(atom_type_noise_prob)
        if not 0.0 <= self.atom_type_noise_prob <= 1.0:
            raise ValueError("atom_type_noise_prob must be between zero and one")
        # Corruption uses empirical frequencies, independently of the uniform
        # source prior. Standalone callers may use their supplied distributions.
        noise_distributions = noise_distributions or distributions
        self._noise_atom_probs = noise_distributions.atom_type_distribution
        self._noise_edge_probs = noise_distributions.edge_type_distribution
        self._noise_has_edges = bool(distributions.edge_type_distribution[1:].any())
        self.del_schedule = del_schedule or SmoothstepSchedule(shift=0.8)
        self.ins_schedule = ins_schedule or SmoothstepSchedule(shift=0.8)
        self.sub_schedule = sub_schedule or SmoothstepSchedule(shift=1.5)
        self.sub_e_schedule = sub_e_schedule or self.sub_schedule

        # OT matching. The ground cost is *always* the squared-Euclidean (W2)
        # distance, so the permutation, the Kabsch alignment and the FM path all
        # minimize the same energy. The move cost is counted in ANGSTROM: c_move
        # multiplies the squared move distance in A^2. Positions are stored
        # normalized (÷ coordinate_std), so we fold coordinate_std^2 into c_move
        # here; c_ins / c_del / c_sub are then plain A^2 costs, directly
        # comparable and set straight from the config. The GEOM defaults
        # (c_ins=c_del=6.125, c_sub=0) prefer deletion plus insertion over
        # moving a matched pair farther than 3.5 Angstroms.
        # pre_align_iters runs the equivariant-OT inner loop (Hungarian+Kabsch)
        # before the class-aware assignment.
        self.pre_align_iters = int(pre_align_iters)

        cstd = getattr(distributions, "coordinate_std", None)
        cstd = float(cstd) if cstd is not None else 1.0
        # Public noise scales are physical Angstroms while coordinates are stored
        # normalized by ``coordinate_std``.
        self.move_noise_scale = float(move_noise_scale) / cstd
        self.ins_noise_scale = float(ins_noise_scale) / cstd
        self.jitter_scale = float(jitter_scale) / cstd
        self.c_move = c_move * cstd * cstd  # squared: normalized -> A^2 move cost
        self.c_sub = c_sub
        self.c_ins = c_ins
        self.c_del = c_del

        self._cat_atom = torch.distributions.Categorical(
            probs=distributions.atom_type_distribution
        )
        self._cat_charge = torch.distributions.Categorical(
            probs=distributions.charge_type_distribution
        )
        self._cat_edge = torch.distributions.Categorical(
            probs=distributions.edge_type_distribution
        )

    # -- elementary interpolation (identical to the original) -----------------
    def interpolate_continuous(self, x0, x1, t):
        """Linear interpolation between continuous tensors at time ``t``."""
        return x0 * (1 - t) + x1 * t

    def interpolate_discrete(self, y0_idx, y1_idx, t):
        """Bernoulli interpolation of class indices: keep ``y0`` w.p. ``1 - t``."""
        keep_source = torch.rand(y0_idx.shape[0], device=t.device) > t
        return torch.where(keep_source, y0_idx, y1_idx)

    @staticmethod
    def _corrupt_categorical(values, eligible, probability, base_probs):
        """Redraw selected tokens from empirical alternatives, excluding self."""
        if probability <= 0.0 or values.numel() == 0 or base_probs.numel() <= 1:
            return values
        changed = eligible & (
            torch.rand(values.shape, device=values.device) < probability
        )
        current = values[changed].long()
        if current.numel() == 0:
            return values
        probs = base_probs.to(device=values.device, dtype=torch.float32)
        probs = probs.expand(current.shape[0], -1).clone()
        probs.scatter_(1, current.unsqueeze(-1), 0.0)
        # A degenerate empirical categorical can have no alternative mass.
        # Fall back to uniform alternatives in those rows only.
        no_alternative = probs.sum(dim=-1) <= 0.0
        probs[no_alternative] = 1.0
        probs.scatter_(1, current.unsqueeze(-1), 0.0)
        corrupted = values.clone()
        corrupted[changed] = torch.multinomial(probs, 1).squeeze(-1).to(values.dtype)
        return corrupted

    def _balanced_edge_noise_probability(self, num_atoms, num_pairs):
        """Match the expected number of changed pairs to changed atom tokens."""
        if num_pairs == 0 or not self._noise_has_edges:
            return 0.0
        return min(self.atom_type_noise_prob * num_atoms / num_pairs, 1.0)

    @staticmethod
    def _extract_triu_feats(edge_index, e, n, triu_rows):
        if e is None or e.numel() == 0:
            return triu_rows.new_zeros(triu_rows.shape)
        out = e.new_zeros((triu_rows.shape[0],))
        row, col = edge_index[0], edge_index[1]
        upper = row < col
        i, j = row[upper], col[upper]
        pos = (i * (2 * n - i - 1)) // 2 + (j - i - 1)
        out[pos] = e[upper]
        return out

    def interpolate_single(
        self,
        sample_data: ChemAtomicData,
        target_data: ChemAtomicData,
        t_scalar: float,
        scaffold_pairs: list[tuple[int, int]] | None = None,
        existing_noise: str = "move",
    ) -> tuple[ChemAtomicData, ChemAtomicData, InsertionTargets]:
        """Interpolate a training pair, including role-specific coordinate noise.

        Scaffold training uses ``existing_noise="jitter"``: matched atoms get
        small denoising jitter, while insertions get only their (1-t) bridge
        noise. This training interpolator is never used for inference inputs.

        With ``include_prior_context``, ``interp_state.contexts["prior"]`` holds
        the aligned source's real atoms before interpolation, in the same frame
        as the interpolated state. Its topology is independent of current atoms.
        """
        if existing_noise not in {"move", "jitter"}:
            raise ValueError("existing_noise must be 'move' or 'jitter'")
        if (
            self.insertion_assignment == "substituent"
            and scaffold_pairs
            and len(scaffold_pairs) != sample_data.num_nodes
        ):
            raise ValueError(
                "Substituent insertion assignment requires a bare scaffold"
            )
        device = sample_data.positions.device
        t_i = torch.tensor(t_scalar, device=device)

        # ===== 1. OT alignment =====
        if scaffold_pairs:
            sample, target = scaffold_based_assignment_single(
                sample_data,
                target_data,
                scaffold_pairs=scaffold_pairs,
                c_move=self.c_move,
                c_sub=self.c_sub,
                c_ins=self.c_ins,
                c_del=self.c_del,
                optimal_transport=self.optimal_transport,
                squared=True,
            )
        else:
            sample, target = partial_optimal_transport_single(
                sample_data,
                target_data,
                c_move=self.c_move,
                c_sub=self.c_sub,
                c_ins=self.c_ins,
                c_del=self.c_del,
                optimal_transport=self.optimal_transport,
                squared=True,
                pre_align_iters=self.pre_align_iters,
            )

        prior_context = (
            sample.filter_nodes(~sample.is_auxiliary.squeeze(-1))
            if self.include_prior_context
            else None
        )

        n = sample.num_nodes
        xs, as_, cs = sample.positions, sample.atom_types, sample.charge_types
        xt, at, ct = target.positions, target.atom_types, target.charge_types
        is_aux_s = sample.is_auxiliary
        is_aux_t = target.is_auxiliary
        s_nl, t_nl = sample.neighbor_list, target.neighbor_list
        s_edge_index = s_nl.t() if s_nl is not None else None
        s_e = sample.edge_types
        t_edge_index = t_nl.t() if t_nl is not None else None
        t_e = target.edge_types

        # ===== 2. Masks =====
        is_sub = ((~is_aux_s) & (~is_aux_t)).squeeze(-1)
        is_del = ((~is_aux_s) & (is_aux_t)).squeeze(-1)
        is_ins = ((is_aux_s) & (~is_aux_t)).squeeze(-1)

        t_kappa, inst_rate_sub = self.sub_schedule.kappa_and_rate(t_i)
        if self.sub_e_schedule is self.sub_schedule:
            t_kappa_e, inst_rate_sub_e = t_kappa, inst_rate_sub
        else:
            t_kappa_e, inst_rate_sub_e = self.sub_e_schedule.kappa_and_rate(t_i)

        tau_del = torch.rand(n, 1, device=device)
        t_del, inst_rate_del = self.del_schedule.kappa_and_rate(t_i)

        tau_ins = torch.rand(n, 1, device=device)
        t_ins, inst_rate_ins = self.ins_schedule.kappa_and_rate(t_i)

        mask_keep_sub = is_sub
        mask_keep_del = is_del & (t_del < tau_del).squeeze(-1)
        mask_keep_ins = is_ins & (t_ins > tau_ins).squeeze(-1)
        mask_future_ins = is_ins & (t_ins <= tau_ins).squeeze(-1)
        mask_exists = mask_keep_sub | mask_keep_del | mask_keep_ins

        # A. Deletions freeze at source
        if mask_keep_del.any():
            xt[mask_keep_del] = xs[mask_keep_del]
            at[mask_keep_del] = as_[mask_keep_del]
            ct[mask_keep_del] = cs[mask_keep_del]

        # B. Insertions jump to target with noise
        if mask_keep_ins.any():
            n_ins = int(mask_keep_ins.sum())
            xs[mask_keep_ins] = (
                xt[mask_keep_ins]
                + torch.randn(n_ins, xs.shape[-1], device=device) * self.ins_noise_scale
            )
            as_[mask_keep_ins] = self._cat_atom.sample((n_ins,)).to(device)
            cs[mask_keep_ins] = self._cat_charge.sample((n_ins,)).to(device)

        # C. Edge handling on the upper triangle over the N aligned nodes
        full_edge_index, triu_rows, triu_cols, full_to_triu = complete_graph_layout(
            n, device
        )

        e0_triu = self._extract_triu_feats(s_edge_index, s_e, n, triu_rows)
        e1_triu = self._extract_triu_feats(t_edge_index, t_e, n, triu_rows)

        edge_has_del = mask_keep_del[triu_rows] | mask_keep_del[triu_cols]
        edge_has_ins = mask_keep_ins[triu_rows] | mask_keep_ins[triu_cols]
        edge_mask_del = edge_has_del & ~edge_has_ins
        edge_mask_ins = edge_has_ins & ~edge_has_del

        if edge_mask_del.any():
            e1_triu[edge_mask_del] = e0_triu[edge_mask_del]
        if edge_mask_ins.any():
            n_edges_ins = int(edge_mask_ins.sum())
            rand_edges = self._cat_edge.sample((n_edges_ins,)).to(device)
            e0_triu[edge_mask_ins] = rand_edges.to(dtype=e0_triu.dtype)

        # D. Interpolate (positions, then discrete attrs -- order preserved)
        x_t = self.interpolate_continuous(xs, xt, t_i)
        # Restore the original scaffold path: retained/matched atoms receive
        # 0.1 A jitter; insertions have only the bridge noise already in xs.
        # Unconditional training retains its move-noise floor on matched atoms
        # and present insertions. Deletions and future insertions stay untouched.
        if existing_noise == "jitter":
            noise_scale, noise_mask = self.jitter_scale, is_sub
        else:
            noise_scale, noise_mask = self.move_noise_scale, is_sub | mask_keep_ins

        position_noise = torch.randn_like(x_t) * noise_scale
        position_noise[~noise_mask] = 0.0
        x_t = x_t + position_noise
        a_t = self.interpolate_discrete(as_, at, t_kappa)
        # Charge is a clean-state prediction target, not an interpolated state.
        # It is not embedded by the backbone; retaining the source value merely
        # keeps the atomic container schema complete until the final charge head
        # prediction is sampled.
        c_t = cs
        e_t = self.interpolate_discrete(e0_triu, e1_triu, t_kappa_e)

        if self.atom_type_noise_prob > 0.0:
            # Keep the clean endpoint and derive the usual edit labels below
            # from the corrupted input. Source-only deletion nodes have frozen
            # targets and masked type losses, so do not corrupt them.
            eligible_atoms = mask_keep_sub | mask_keep_ins
            if scaffold_pairs is not None:
                # Scaffold nodes occupy the leading aligned slots; their atom
                # types and internal bonds cannot change during generation.
                editable_atoms = eligible_atoms.clone()
                editable_atoms[: len(scaffold_pairs)] = False
            else:
                editable_atoms = eligible_atoms
            a_t = self._corrupt_categorical(
                a_t, editable_atoms, self.atom_type_noise_prob, self._noise_atom_probs
            )
            eligible_pairs = eligible_atoms[triu_rows] & eligible_atoms[triu_cols]
            eligible_pairs &= editable_atoms[triu_rows] | editable_atoms[triu_cols]
            edge_noise_prob = self._balanced_edge_noise_probability(
                int(editable_atoms.sum()), int(eligible_pairs.sum())
            )
            e_t = self._corrupt_categorical(
                e_t, eligible_pairs, edge_noise_prob, self._noise_edge_probs
            )

        # E. Both states share the cached complete topology.
        interp_e = e_t[full_to_triu]
        target_e = e1_triu[full_to_triu]
        target_full = ChemAtomicData(
            positions=xt,
            atom_types=at,
            charge_types=ct,
            neighbor_list=full_edge_index.t().contiguous(),
            edge_types=target_e,
        )

        node_map_exists, neighbor_list, edge_exists = existing_graph_layout(
            mask_exists, full_edge_index
        )

        interp_state = ChemAtomicData(
            positions=x_t[mask_exists],
            atom_types=a_t[mask_exists],
            charge_types=c_t[mask_exists],
            neighbor_list=neighbor_list,
            edge_types=interp_e[edge_exists],
            is_auxiliary=(is_aux_s | is_aux_t)[mask_exists],
        )
        target_state = ChemAtomicData(
            positions=xt[mask_exists],
            atom_types=at[mask_exists],
            charge_types=ct[mask_exists],
            neighbor_list=neighbor_list.clone(),
            edge_types=target_e[edge_exists],
        )

        # F. Spawn logic + insertion supervision (local indices)
        ins_targets = self._build_insertion_targets(
            interp_state,
            target_full,
            mask_future_ins.nonzero(as_tuple=True)[0],
            node_map_exists,
            n_scaffold=len(scaffold_pairs) if scaffold_pairs else 0,
        )

        # G. Rate targets
        ins_counts = torch.bincount(
            ins_targets.spawn_node_idx, minlength=interp_state.num_nodes
        ).to(dtype=interp_state.positions.dtype)

        node_del_rate_target = torch.zeros((n,), device=device)
        node_del_rate_target[mask_keep_del] = inst_rate_del

        interp_state.add_node_property("lambda_ins", ins_counts * inst_rate_ins)
        interp_state.add_node_property("n_ins", ins_counts)
        interp_state.add_node_property("lambda_del", node_del_rate_target[mask_exists])
        interp_state.add_node_property(
            "lambda_a_sub",
            (interp_state.atom_types != target_state.atom_types).int() * inst_rate_sub,
        )
        interp_state.add_edge_property(
            "lambda_e_sub",
            (interp_state.edge_types != target_state.edge_types).int()
            * inst_rate_sub_e,
        )
        interp_state.add_system_property(
            "n_ins_missing", mask_future_ins.sum().float().reshape(1)
        )
        interp_state.add_system_property(
            "n_del_missing", mask_keep_del.sum().float().reshape(1)
        )

        # H. Coordinate centering (ZCoM in the active frame)
        x_mean = interp_state.positions.mean(dim=0)
        # Fixed contexts bypass interpolation and must receive this translation.
        interp_state.frame_shift = x_mean
        interp_state.positions = interp_state.positions - x_mean
        if prior_context is not None:
            prior_context.positions = prior_context.positions - x_mean
            interp_state.contexts["prior"] = prior_context
        if ins_targets.num_nodes > 0:
            ins_targets.positions = ins_targets.positions - x_mean
        target_state.positions = target_state.positions - target_state.positions.mean(
            dim=0
        )

        # Per-sample topology edits remain on the lightweight native container;
        # the collate concatenates these tensors directly into a ChemBatch.
        return interp_state, target_state, ins_targets

    def _build_insertion_targets(
        self,
        interp_state,
        target_full,
        future_indices,
        orig_to_interp,
        *,
        n_scaffold=0,
    ) -> InsertionTargets:
        """Spawn-node assignment and insertion-bond supervision (local indices)."""
        num_interp = interp_state.num_nodes
        num_inserted = future_indices.numel()
        pos = target_full.positions[future_indices]
        atoms = target_full.atom_types[future_indices]
        charges = target_full.charge_types[future_indices]

        if num_inserted == 0 or num_interp == 0:
            return InsertionTargets.empty(pos, atoms, charges, target_full.device)

        # Use the nearest eligible existing atom. Site constraints apply only
        # to scaffold training; unconditional/replay pairs keep the usual rule.
        dist_to_interp = torch.cdist(pos, interp_state.positions)
        if self.insertion_assignment == "substituent" and n_scaffold:
            candidates = scaffold_insertion_candidates(
                target_full, n_scaffold, future_indices, orig_to_interp
            )
            dist_to_interp = dist_to_interp.masked_fill(~candidates, float("inf"))
        spawn_node_idx = dist_to_interp.argmin(dim=1)
        # One-to-one inverse: each spawn node's single nearest insertion (used by
        # the unimodal insertion head).
        spawn_dist = dist_to_interp.gather(1, spawn_node_idx.unsqueeze(1)).squeeze(1)
        nearest_node_idx, nearest_ins_idx = _nearest_insertion_per_node(
            spawn_node_idx, spawn_dist, num_interp
        )

        orig_to_ins = torch.full_like(orig_to_interp, -1)
        orig_to_ins[future_indices] = torch.arange(
            num_inserted, device=target_full.device
        )

        # Classify each (directed) target edge by its endpoints. Boolean indexing
        # naturally yields empty tensors when a class has no edges, so no special
        # casing is needed.
        src, dst = target_full.neighbor_list.t()
        src_is_inserted = orig_to_ins[src] >= 0
        dst_is_existing = orig_to_interp[dst] >= 0
        dst_is_inserted = orig_to_ins[dst] >= 0

        # inserted -> existing
        to_exist = src_is_inserted & dst_is_existing
        inserted_idx = orig_to_ins[src[to_exist]]
        bonds_to_existing = BondsToExisting(
            spawn_idx=spawn_node_idx[inserted_idx],
            existing_idx=orig_to_interp[dst[to_exist]],
            inserted_idx=inserted_idx,
            bond_type=target_full.edge_types[to_exist],
        )

        # inserted -> inserted, keeping the upper triangle (src_local < dst_local)
        between = src_is_inserted & dst_is_inserted
        src_idx = orig_to_ins[src[between]]
        dst_idx = orig_to_ins[dst[between]]
        upper = src_idx < dst_idx
        src_idx, dst_idx = src_idx[upper], dst_idx[upper]
        bonds_between_inserted = BondsBetweenInserted(
            spawn_src_idx=spawn_node_idx[src_idx],
            spawn_dst_idx=spawn_node_idx[dst_idx],
            src_idx=src_idx,
            dst_idx=dst_idx,
            bond_type=target_full.edge_types[between][upper],
        )

        return InsertionTargets(
            positions=pos,
            atom_types=atoms,
            charge_types=charges,
            spawn_node_idx=spawn_node_idx,
            bonds_to_existing=bonds_to_existing,
            bonds_between_inserted=bonds_between_inserted,
            nearest_node_idx=nearest_node_idx,
            nearest_ins_idx=nearest_ins_idx,
        )
