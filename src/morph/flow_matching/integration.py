"""Molecular flow integration with coordinate, substitution, deletion, and insertion
    steps.

States are native ``ChemBatch`` objects. Insertion heads consume flat tensors;
trace records expose the exact decisions made by the production integrator.
"""

from __future__ import annotations

import itertools
import math
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from scipy.stats import beta as scipy_beta

from morph.dataset.core.atomic_data import (
    ChemBatch,
    batch_from_flat,
    batch_unpack,
    filter_nodes,
)
from morph.dataset.core.insertion_data import (
    InsertedAtoms,
    count_new_atom_edges,
    insertion_pair_indices,
    join_with_atoms,
    join_with_predicted_edges,
    sample_edge_types,
)
from morph.dataset.core.insertion_targets import (
    BondsBetweenInserted,
    BondsToExisting,
)
from morph.flow_matching.sampling_step import (
    InsertionStepTrace,
    IntegrationStepTrace,
    ScheduleStepMasses,
    node_edit_probabilities,
)
from morph.flow_matching.schedules import (
    KappaSchedule,
    SmoothstepSchedule,
)
from morph.model.layers.gmm import sample_from_typed_gmm
from morph.utils.graph_math import batched_triu_edge_layout

if TYPE_CHECKING:
    from collections.abc import Callable

    from morph.dataset.core.vocab import Distributions, Vocab
    from morph.model.layers.heads import InsertionEdgeHead


@dataclass(frozen=True)
class IntegrationPredictions:
    """Clean endpoint and edit predictions, all aligned with the current batch."""

    mol_1_pred: ChemBatch
    do_sub_a_probs: torch.Tensor
    do_sub_e_probs: torch.Tensor
    do_del_probs: torch.Tensor
    num_ins_pred: torch.Tensor
    insertion_preds: dict[str, torch.Tensor]
    h_latent: torch.Tensor | None = None


@dataclass(frozen=True)
class _InsertionContext:
    """Original node space and its mapping into the post-deletion batch.

    Bond prediction uses the original positions/features but the substituted
    atom types. Counts and batch indices always come from ``mol_t``.
    """

    mol_t: ChemBatch
    atom_types: torch.Tensor
    spawn_indices: torch.Tensor
    time_next: torch.Tensor
    predictions: IntegrationPredictions
    edge_head: InsertionEdgeHead | None
    original_to_postdel: torch.Tensor | None


@dataclass(frozen=True)
class _PredictedBondsToExisting:
    bonds: BondsToExisting
    postdel_existing_idx: torch.Tensor


@dataclass(frozen=True)
class _EdgeSubstitution:
    edge_types: torch.Tensor
    edge_index: torch.Tensor
    targets: torch.Tensor
    selected: torch.Tensor


class RateIntegrator:
    """Rate integrator for flow matching on ``ChemBatch`` states."""

    def __init__(
        self,
        vocab: Vocab,
        distributions: Distributions,
        insertion_head_mode: str,
        insertion_dimension: int = 3,
        insertion_num_components: int = 1,
        num_integration_steps=100,
        time_strategy="log",
        time_beta_alpha: float = 2.0,
        time_beta_beta: float = 1.0,
        ins_noise_scale=0.65,
        renoise_insertions: bool = True,  # match unconditional training-time noising
        exact_schedule_integration: bool = True,
        del_schedule: KappaSchedule | None = None,
        ins_schedule: KappaSchedule | None = None,
        sub_schedule: KappaSchedule | None = None,
        sub_e_schedule: KappaSchedule | None = None,
        device="cuda" if torch.cuda.is_available() else "cpu",  # noqa: B008
    ):
        self.vocab = vocab
        self.distributions = distributions
        if insertion_head_mode not in {"unimodal", "gmm"}:
            raise ValueError(
                "insertion_head_mode must be 'unimodal' or 'gmm', got "
                f"{insertion_head_mode!r}"
            )
        if insertion_num_components < 1:
            raise ValueError("insertion_num_components must be positive")
        if insertion_head_mode == "unimodal" and insertion_num_components != 1:
            raise ValueError("the unimodal insertion head must have one component")
        self.insertion_head_mode = insertion_head_mode
        self.insertion_dimension = int(insertion_dimension)
        self.insertion_num_components = int(insertion_num_components)
        cstd = getattr(distributions, "coordinate_std", None)
        cstd = float(cstd) if cstd is not None else 1.0
        # Match the interpolator: configuration is in physical Angstroms while
        # integration operates in normalized coordinate units.
        self.ins_noise_scale = float(ins_noise_scale) / cstd
        self.renoise_insertions = renoise_insertions
        self.exact_schedule_integration = bool(exact_schedule_integration)
        self.time_strategy = time_strategy
        self.time_beta_alpha = time_beta_alpha
        self.time_beta_beta = time_beta_beta
        self.num_integration_steps = num_integration_steps
        self.device = device

        self.del_schedule = del_schedule or SmoothstepSchedule(shift=0.8)
        self.ins_schedule = ins_schedule or SmoothstepSchedule(shift=0.8)
        self.sub_schedule = sub_schedule or SmoothstepSchedule(shift=1.5)
        self.sub_e_schedule = sub_e_schedule or self.sub_schedule

        n_atoms_distribution = self.distributions.n_atoms_distribution
        self.max_atoms = len(n_atoms_distribution) + 10  # buffer for ins / del

        self._cat_atom = torch.distributions.Categorical(
            probs=distributions.atom_type_distribution.to(device)
        )
        self._cat_edge = torch.distributions.Categorical(
            probs=distributions.edge_type_distribution.to(device)
        )
        self._cat_device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )

    def schedule_step_masses(
        self, t: torch.Tensor, dt: float | torch.Tensor, *, is_final: bool = False
    ) -> ScheduleStepMasses:
        """Parameterise all categorical edit probabilities for a time interval.

        Keeping the time discretization here ensures every sampling path uses
        the same exact or Euler schedule convention. Exact mode converts the
        integrated schedule hazard into the conditional probability that a
        still-pending endpoint transition occurs during this interval.

        The sampler's final interval always completes the discrete bridge:
        its schedule masses are one, leaving the learned edit
        probabilities and masks in control. In particular, stopping the log
        grid at 0.999 must not leave predicted endpoint edits pending.
        ``is_final`` is step context supplied by the sampling loop, not a
        configuration switch; it avoids guessing from rounded time values.
        """
        if is_final:
            mass = torch.ones_like(t)
            return ScheduleStepMasses(mass, mass, mass, mass)

        def step(schedule: KappaSchedule) -> torch.Tensor:
            if self.exact_schedule_integration:
                return -torch.expm1(-schedule.integrated_rate(t, dt))
            return schedule.rate(t) * dt

        return ScheduleStepMasses(
            substitution=step(self.sub_schedule),
            edge_substitution=step(self.sub_e_schedule),
            insertion=step(self.ins_schedule),
            deletion=step(self.del_schedule),
        )

    def _distr_to_device(self, device: torch.device):
        device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        if getattr(self, "_cat_device", None) == device:
            return
        self._cat_atom = torch.distributions.Categorical(
            probs=self._cat_atom.probs.to(device)
        )
        self._cat_edge = torch.distributions.Categorical(
            probs=self._cat_edge.probs.to(device)
        )
        self._cat_device = device

    def get_time_steps(self, num_steps: int | None = None) -> list[float]:
        """Return time deltas for a grid with a positive number of steps.

        Linear and Beta grids end at 1; the log grid ends at 0.999. The sampler
        explicitly completes discrete edits on the final interval of every grid.
        """
        if num_steps is None:
            num_steps = self.num_integration_steps
        try:
            num_steps = operator.index(num_steps)
        except TypeError as exc:
            raise ValueError("num_steps must be a positive integer") from exc
        if num_steps < 1:
            raise ValueError("num_steps must be a positive integer")

        if self.time_strategy == "linear":
            time_points = torch.linspace(0, 1, num_steps + 1).tolist()
        elif self.time_strategy == "log":
            # ``num_steps + 1`` grid points produce exactly ``num_steps``
            # integration deltas. Stop at 0.999 so rates containing 1 / (1-t)
            # are never evaluated at their singular endpoint.
            time_points = (1 - torch.logspace(0, -3, num_steps + 1)).tolist()
        elif self.time_strategy == "beta":
            if any(
                not math.isfinite(shape) or shape <= 0
                for shape in (self.time_beta_alpha, self.time_beta_beta)
            ):
                raise ValueError("Beta time-grid shapes must be finite and positive")
            # Place points via the Beta(alpha, beta) inverse CDF of a uniform
            # grid, so the step density follows the same Beta law used to sample
            # training time (the ``beta_sched`` in the model config).
            u = torch.linspace(0, 1, num_steps + 1).numpy()
            ppf = scipy_beta.ppf(u, self.time_beta_alpha, self.time_beta_beta)
            time_points = torch.from_numpy(ppf).clamp(0.0, 1.0).tolist()
        else:
            raise ValueError(f"Invalid time strategy: {self.time_strategy}")

        return [t1 - t0 for t0, t1 in itertools.pairwise(time_points)]

    def sample_insertions(self, insertion_pred: dict, t: torch.Tensor) -> InsertedAtoms:
        """Sample one new atom per provided head output (works with batch ids)."""
        dimension = self.insertion_dimension

        if self.insertion_head_mode == "unimodal":
            # Deterministic mean position; the integrator's insertion renoising
            # supplies the spatial spread (mirrors the GMM path's renoising).
            mu = insertion_pred["mu"]
            sampled_a = torch.multinomial(
                torch.softmax(insertion_pred["a_logits"].float(), dim=-1), 1
            ).squeeze(1)
            sampled_c = torch.multinomial(
                torch.softmax(insertion_pred["c_logits"].float(), dim=-1), 1
            ).squeeze(1)
            return InsertedAtoms(
                x=mu.view(-1, dimension),
                a=sampled_a.view(-1),
                c=sampled_c.view(-1),
            )

        num_ins = 1
        num_components = self.insertion_num_components
        A = len(self.vocab.atom_tokens)
        C = len(self.vocab.charge_tokens)

        sampled_x, sampled_a, sampled_c = sample_from_typed_gmm(
            insertion_pred, num_ins, num_components, dimension, A, C
        )
        return InsertedAtoms(
            x=sampled_x.view(-1, dimension),
            a=sampled_a.view(-1),
            c=sampled_c.view(-1),
        )

    def _renoise_insertions(
        self, new_atoms: InsertedAtoms, t_ins: torch.Tensor
    ) -> InsertedAtoms:
        """Push freshly-inserted nodes to the t+dt marginal (matches training)."""
        if t_ins.numel() == 0:
            return new_atoms

        t_ins = t_ins.to(new_atoms.x.device).clamp(0.0, 1.0)

        noise_x = torch.randn_like(new_atoms.x)
        scale_x = ((1.0 - t_ins) * self.ins_noise_scale).unsqueeze(-1)
        new_atoms.x = new_atoms.x + scale_x * noise_x

        # Atom types follow the same categorical bridge as training. Charge is
        # only a clean prediction target, not an interpolated model state.
        kappa = self.sub_schedule.kappa_t(t_ins).clamp(0.0, 1.0).unsqueeze(-1)
        new_atoms.a = self._kappa_mix_sample(new_atoms.a, self._cat_atom, kappa)
        return new_atoms

    def _kappa_mix_sample(self, clean_idx, prior_cat, kappa):
        """Resample class indices as a kappa-mix of the clean one-hot and a prior.

        Pushes a freshly predicted (clean) categorical sample toward the t+dt
        marginal: mix the clean one-hot with ``prior_cat``'s probabilities,
        weighted per-row by ``kappa`` (shape ``[N, 1]``), renormalize in fp32 to
        stay on the simplex, and resample. Shared by the atom / charge / edge
        renoising paths.
        """
        num_classes = prior_cat.probs.shape[-1]
        prior = prior_cat.probs.unsqueeze(0).expand(clean_idx.shape[0], -1)
        one_hot = F.one_hot(clean_idx, num_classes=num_classes).to(prior.dtype)
        # Clamp so the mix is always a convex combination: a schedule that returns
        # kappa marginally above 1 (fp32 cancellation near t=1) would otherwise make
        # ``prior * (1 - kappa)`` negative, which aborts the process inside
        # ``torch.multinomial`` rather than raising.
        kappa = (
            kappa.clamp(0.0, 1.0)
            if torch.is_tensor(kappa)
            else min(max(kappa, 0.0), 1.0)
        )
        mix = prior * (1.0 - kappa) + one_hot * kappa
        mix = mix.float()
        mix = mix / mix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        # torch.multinomial is exactly what Categorical(probs=).sample() calls
        # under the hood (same RNG draw) without building the distribution object.
        return torch.multinomial(mix, 1).squeeze(1)

    def _renoise_edge_types(self, clean_idx, t_per_edge):
        """Push clean inserted-edge types to the t+dt marginal (no-op if disabled)."""
        if not self.renoise_insertions:
            return clean_idx
        kappa = self.sub_e_schedule.kappa_t(t_per_edge).unsqueeze(1)
        return self._kappa_mix_sample(clean_idx, self._cat_edge, kappa)

    def _select_spawning_nodes(self, mol_t, num_ins_pred, ins_step_node):
        """Sample insertion events and enforce the per-graph atom limit."""
        batch_id = mol_t.batch
        # ``num_ins_pred`` is the expected number of pending endpoint
        # insertions. The schedule mass is the per-endpoint transition
        # probability for this interval, not a hazard to exponentiate jointly
        # with the model prediction. At most one insertion per spawning node is
        # supported in a numerical step, matching the historical sampler.
        p_ins = (num_ins_pred * ins_step_node).clamp(0.0, 1.0)
        do_ins = torch.rand_like(p_ins) < p_ins

        spawn_orig_indices = torch.where(do_ins)[0]
        has_insertions = spawn_orig_indices.numel() > 0
        if has_insertions:
            n_atoms_per_graph = mol_t.num_nodes_per_graph
            n_ins_per_graph = torch.zeros_like(n_atoms_per_graph).scatter_add_(
                0, batch_id, do_ins.long()
            )
            overflow = (n_atoms_per_graph + n_ins_per_graph - self.max_atoms).clamp(
                min=0
            )
            overflow_graphs = torch.nonzero(overflow, as_tuple=False).flatten().tolist()
            for g in overflow_graphs:
                n_to_remove = int(overflow[g].item())
                removed_indices = torch.where((batch_id == g) & do_ins)[0]
                removed_indices = removed_indices[
                    torch.randperm(removed_indices.shape[0], device=do_ins.device)
                ]
                do_ins[removed_indices[:n_to_remove]] = False
            if overflow_graphs:
                spawn_orig_indices = torch.where(do_ins)[0]

        return do_ins, spawn_orig_indices

    @staticmethod
    def _sample_node_edits(predictions, sub_step_node, del_step_node):
        """Sample mutually exclusive substitution and deletion events."""
        # The sigmoid heads estimate mutually-exclusive endpoint-edit
        # posteriors. Multiply each posterior by its schedule transition mass,
        # then add the resulting numerical-step probabilities.
        sub_step_mass = predictions.do_sub_a_probs.view(-1) * sub_step_node
        del_step_mass = predictions.do_del_probs.view(-1) * del_step_node
        node_edit = node_edit_probabilities(sub_step_mass, del_step_mass)
        do_edit = torch.rand_like(node_edit.any_edit) < node_edit.any_edit
        is_deletion_type = (
            torch.rand_like(node_edit.deletion_given_edit)
            < node_edit.deletion_given_edit
        )
        do_del = do_edit & is_deletion_type
        do_sub_a = do_edit & (~is_deletion_type)
        return do_sub_a, do_del

    @staticmethod
    def _substitute_edges(mol_t, predictions, step_masses):
        """Sample each undirected bond once and restore directed edge order."""
        edge_index, batch_id = mol_t.edge_index, mol_t.batch
        e_t, e_1 = mol_t.e, predictions.mol_1_pred.e
        # ``mol_1_pred`` is built on ``mol_t.edge_index`` by the sampling loop,
        # so all edge tensors already share an order. Avoid sorting them twice.
        triu_positions, full_to_triu = batched_triu_edge_layout(
            edge_index, batch_id, mol_t.num_nodes_per_graph
        )
        edge_index_triu = edge_index[:, triu_positions]
        e_triu = e_t[triu_positions]
        do_sub_e_probs_triu = predictions.do_sub_e_probs[triu_positions]
        e_1_triu = e_1[triu_positions]

        q_sub_e = do_sub_e_probs_triu.view(-1)
        batch_id_edge = batch_id[edge_index_triu[0]]
        sub_e_step_triu = step_masses.edge_substitution[batch_id_edge]
        p_mod_e = (q_sub_e * sub_e_step_triu).view(-1)
        do_sub_e = torch.rand_like(p_mod_e) < p_mod_e
        e_triu = torch.where(do_sub_e, e_1_triu, e_triu)

        # Mirror the changed per-pair values into the existing directed order.
        # The layout mapping remains valid after insertion/deletion reorders the
        # edge storage, avoiding a sort without assuming lexicographic order.
        e_t = e_triu[full_to_triu]

        return _EdgeSubstitution(e_t, edge_index_triu, e_1_triu, do_sub_e)

    @staticmethod
    def _constrain_deletions(mol_t, do_del):
        """Restore enough deleted nodes to retain at least two per graph.

        Update ``do_del`` in place so traces describe the edits actually applied.
        """
        batch_id = mol_t.batch
        node_keep = ~do_del

        # Fail-safe: never delete a whole sample (keep >= 2 nodes per graph).
        n_kept_per_graph = torch.zeros_like(mol_t.num_nodes_per_graph).scatter_add_(
            0, batch_id, node_keep.long()
        )
        under = (2 - n_kept_per_graph).clamp(min=0)
        for g in torch.nonzero(under, as_tuple=False).flatten().tolist():
            n_to_restore = int(under[g].item())
            deleted_in_graph_idx = torch.where((batch_id == g) & do_del)[0]
            n_restore = min(n_to_restore, deleted_in_graph_idx.shape[0])
            if n_restore > 0:
                restored = deleted_in_graph_idx[:n_restore]
                node_keep[restored] = True
                do_del[restored] = False

        return node_keep

    def integrate_step_gnn(
        self,
        mol_t: ChemBatch,
        predictions: IntegrationPredictions,
        t: torch.Tensor,
        dt: float | torch.Tensor,
        *,
        eps: float = 1e-6,
        ins_edge_head: InsertionEdgeHead | None = None,
        force_charge_idx: int | None = None,
        return_step_trace: bool = False,
        position_update: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, float | torch.Tensor],
            torch.Tensor,
        ]
        | None = None,
        is_final: bool = False,
    ) -> ChemBatch | tuple[ChemBatch, IntegrationStepTrace]:
        """Integrate one step, optionally returning its realized edit trace.

        By default positions use the standard Euler update. ``position_update``
        optionally supplies an alternative coordinate kernel while retaining the
        same topology transition. The sampling loop supplies ``is_final`` to
        complete discrete transitions on its final iteration.
        """
        active_device = mol_t["positions"].device
        self._distr_to_device(active_device)

        x_t, a_t, _, _, edge_index, batch_id = batch_unpack(mol_t)
        x_1, a_1, c_1, _, _, _ = batch_unpack(predictions.mol_1_pred)
        num_graphs = t.shape[0]

        # A scalar keeps the historical fixed-grid behavior.  A (B,) tensor
        # allows each graph to advance by a different amount without changing
        # any of the node/edge kernels below.
        if torch.is_tensor(dt):
            dt_graph = dt.to(device=active_device, dtype=t.dtype).reshape(-1)
            if dt_graph.numel() != num_graphs:
                raise ValueError(
                    f"tensor dt must have one value per graph ({num_graphs}), "
                    f"got shape {tuple(dt.shape)}"
                )
            dt_position = dt_graph[batch_id].unsqueeze(-1)
        else:
            dt_graph = dt
            dt_position = dt

        conditioning_keep_mask = (
            mol_t["keep_mask"] if "keep_mask" in mol_t else None  # noqa: SIM401
        )
        move_rate = 1 / (1 - t).clamp(min=eps, max=1.0 - eps)
        move_rate_node = move_rate[batch_id]

        step_masses = self.schedule_step_masses(t, dt_graph, is_final=is_final)
        sub_step_node = step_masses.substitution[batch_id]
        ins_step_node = step_masses.insertion[batch_id]
        del_step_node = step_masses.deletion[batch_id]

        # 1. Positions (deterministic Euler update). ``move_noise_scale`` in the
        # interpolator is training-time denoising jitter, not an SDE diffusion
        # coefficient, so no new Gaussian movement noise is injected here.
        x_t_original = x_t
        if position_update is None:
            velocity = (x_1 - x_t) * move_rate_node.view(-1, 1)
            x_t = x_t + velocity * dt_position
        else:
            x_t = position_update(x_t, x_1, t[batch_id], dt_position)
            if x_t.shape != x_t_original.shape:
                raise ValueError(
                    "position_update must preserve the position tensor shape; "
                    f"expected {tuple(x_t_original.shape)}, got {tuple(x_t.shape)}"
                )

        do_ins, spawn_orig_indices = self._select_spawning_nodes(
            mol_t, predictions.num_ins_pred, ins_step_node
        )
        has_insertions = spawn_orig_indices.numel() > 0

        do_sub_a, do_del = self._sample_node_edits(
            predictions, sub_step_node, del_step_node
        )
        a_t = torch.where(do_sub_a, a_1, a_t)

        edge_edit = self._substitute_edges(mol_t, predictions, step_masses)

        mol = batch_from_flat(
            x_t,
            a_t,
            c_1,
            edge_edit.edge_types,
            edge_index,
            batch_id,
            num_graphs=num_graphs,
            node_extras=(
                {"keep_mask": conditioning_keep_mask}
                if conditioning_keep_mask is not None
                else None
            ),
            system_extras={key: mol_t[key] for key in mol_t.keys["system"]},
            _already_sorted=True,
            _node_counts=mol_t.num_nodes_per_graph,
            _edge_counts=mol_t.num_edges_per_graph,
        )

        node_keep = self._constrain_deletions(mol_t, do_del)

        num_original = mol.num_nodes
        has_deletions = bool(do_del.any())
        if has_deletions:
            mol = filter_nodes(mol, node_keep)

        insertion_trace = None
        if has_insertions:
            original_to_postdel = None
            if ins_edge_head is not None and predictions.h_latent is not None:
                if has_deletions:
                    original_to_postdel = torch.full(
                        (num_original,),
                        -1,
                        dtype=torch.long,
                        device=active_device,
                    )
                    kept_indices = torch.where(node_keep)[0]
                    original_to_postdel[kept_indices] = torch.arange(
                        kept_indices.numel(), device=active_device
                    )
                else:
                    original_to_postdel = torch.arange(
                        num_original, device=active_device
                    )
            mol, insertion_trace = self._apply_insertions(
                mol=mol,
                context=_InsertionContext(
                    mol_t=mol_t,
                    atom_types=a_t,
                    spawn_indices=spawn_orig_indices,
                    time_next=t + dt_graph,
                    predictions=predictions,
                    edge_head=ins_edge_head,
                    original_to_postdel=original_to_postdel,
                ),
                force_charge_idx=force_charge_idx,
                return_step_trace=return_step_trace,
            )

        mol.contexts.update(mol_t.contexts)
        if not return_step_trace:
            return mol
        return mol, IntegrationStepTrace(
            positions=x_t,
            atom_type_targets=a_1,
            charge_targets=c_1,
            node_substitutions=do_sub_a,
            node_deletions=do_del,
            edge_index=edge_edit.edge_index,
            edge_type_targets=edge_edit.targets,
            edge_substitutions=edge_edit.selected,
            insertions=do_ins,
            insertion=insertion_trace,
        )

    def _apply_insertions(
        self,
        mol: ChemBatch,
        context: _InsertionContext,
        *,
        force_charge_idx: int | None = None,
        return_step_trace: bool = False,
    ) -> tuple[ChemBatch, InsertionStepTrace | None]:
        """Sample inserted atoms and join them (predicted edges if a head is given)."""
        if self.insertion_head_mode == "unimodal":
            ins_head_keys = ("mu", "a_logits", "c_logits")
        else:
            ins_head_keys = ("mu", "sigma", "pi", "a_probs", "c_probs")
        active_insertion_pred = {
            key: context.predictions.insertion_preds[key][context.spawn_indices]
            for key in ins_head_keys
        }
        ins_batch_id = context.mol_t.batch[context.spawn_indices]
        t_ins = context.time_next[ins_batch_id]
        clean_atoms = self.sample_insertions(active_insertion_pred, t_ins)
        if force_charge_idx is not None:
            clean_atoms.c = torch.full_like(clean_atoms.c, force_charge_idx)
        clean_atoms.batch = ins_batch_id

        # Predict clean bonds from the clean insertion sample, but insert a copy at the
        # current noise level so the next model step sees the expected state.
        new_atoms = InsertedAtoms(
            clean_atoms.x.clone(),
            clean_atoms.a.clone(),
            clean_atoms.c.clone(),
            clean_atoms.batch,
        )
        if self.renoise_insertions:
            new_atoms = self._renoise_insertions(new_atoms, t_ins)

        edge_dist = self._cat_edge.probs

        new_atoms_batch = batch_from_flat(
            new_atoms.x,
            new_atoms.a,
            new_atoms.c,
            None,
            None,
            new_atoms.batch,
            num_graphs=context.mol_t.num_graphs,
            _already_sorted=True,
        )

        bonds_to_existing = None
        bonds_between_inserted = None

        if context.edge_head is not None and context.predictions.h_latent is not None:
            bonds_to_existing = self._predict_inserted_to_existing_edges(
                context, clean_atoms
            )
            bonds_between_inserted = self._predict_inserted_to_inserted_edges(
                context, clean_atoms
            )

        use_predicted_edges = bonds_to_existing is not None
        if use_predicted_edges:
            result = join_with_predicted_edges(
                mol,
                new_atoms_batch,
                e_ins=bonds_to_existing.bonds.bond_type,
                spawn_node_idx=bonds_to_existing.bonds.inserted_idx,
                target_node_idx=bonds_to_existing.postdel_existing_idx,
                e_ins_to_ins=(
                    bonds_between_inserted.bond_type
                    if bonds_between_inserted is not None
                    else None
                ),
                ins_to_ins_src_idx=(
                    bonds_between_inserted.src_idx
                    if bonds_between_inserted is not None
                    else None
                ),
                ins_to_ins_dst_idx=(
                    bonds_between_inserted.dst_idx
                    if bonds_between_inserted is not None
                    else None
                ),
                sample_fallback_edge_types=lambda n: sample_edge_types(edge_dist, n),
            )
        else:
            # Fall back to random edge sampling (outsourced).
            n_edges = count_new_atom_edges(mol, new_atoms_batch)
            result = join_with_atoms(
                mol, new_atoms_batch, sample_edge_types(edge_dist, n_edges)
            )

        if not return_step_trace:
            return result, None
        return result, InsertionStepTrace(
            spawn_node_idx=context.spawn_indices,
            clean_atoms=clean_atoms,
            bonds_to_existing=(
                bonds_to_existing.bonds if bonds_to_existing is not None else None
            ),
            bonds_between_inserted=(
                bonds_between_inserted if use_predicted_edges else None
            ),
        )

    def _predict_inserted_to_existing_edges(
        self,
        context: _InsertionContext,
        new_atoms: InsertedAtoms,
    ) -> _PredictedBondsToExisting | None:
        """Predict (inserted -> existing) bonds -> (new_atom_idx, target_idx, types)."""
        mol_t = context.mol_t
        orig_spawn_idx, orig_existing_idx, ins_edge_logits = (
            context.edge_head.predict_inserted_to_existing(
                h=context.predictions.h_latent,
                x=mol_t.x,
                node_atom_types=context.atom_types,
                batch=mol_t.batch,
                insertion_indices=context.spawn_indices,
                node_counts=mol_t.num_nodes_per_graph,
                ins=InsertedAtoms(new_atoms.x, new_atoms.a, new_atoms.c),
            )
        )

        if ins_edge_logits is None or ins_edge_logits.numel() == 0:
            return None

        # Promote logits before normalization to avoid bf16 probability loss.
        ins_edge_probs = F.softmax(ins_edge_logits.float(), dim=-1)
        e1_ins_idx = torch.multinomial(ins_edge_probs, 1).squeeze(1)
        spawn_batch_id = mol_t.batch[orig_spawn_idx]
        e_t_ins = self._renoise_edge_types(
            e1_ins_idx,
            context.time_next[spawn_batch_id],
        )

        orig_to_new_atom = torch.full(
            (mol_t.num_nodes,), -1, dtype=torch.long, device=mol_t.x.device
        )
        orig_to_new_atom[context.spawn_indices] = torch.arange(
            context.spawn_indices.numel(), device=mol_t.x.device
        )
        ins_edge_new_atom_idx = orig_to_new_atom[orig_spawn_idx]
        ins_edge_target_idx = context.original_to_postdel[orig_existing_idx]
        # The deletion fail-safe leaves every insertion's graph with at least
        # one existing atom, so at least one endpoint is always valid.
        valid_edges = torch.where(ins_edge_target_idx >= 0)[0]

        return _PredictedBondsToExisting(
            bonds=BondsToExisting(
                spawn_idx=orig_spawn_idx[valid_edges],
                existing_idx=orig_existing_idx[valid_edges],
                inserted_idx=ins_edge_new_atom_idx[valid_edges],
                bond_type=e_t_ins[valid_edges],
            ),
            postdel_existing_idx=ins_edge_target_idx[valid_edges],
        )

    def _predict_inserted_to_inserted_edges(
        self,
        context: _InsertionContext,
        new_atoms: InsertedAtoms,
    ) -> BondsBetweenInserted | None:
        """Predict (inserted -> inserted) bonds for intra-graph upper-tri pairs."""
        ii_src, ii_dst, ii_spawn_src, ii_spawn_dst = insertion_pair_indices(
            new_atoms.batch, context.spawn_indices
        )
        if ii_src.numel() == 0:
            return None

        new_feats = InsertedAtoms(new_atoms.x, new_atoms.a, new_atoms.c)
        ii_logits = context.edge_head.forward_inserted_to_inserted(
            h=context.predictions.h_latent,
            x=context.mol_t.x,
            spawn_src_idx=ii_spawn_src,
            spawn_dst_idx=ii_spawn_dst,
            src=new_feats[ii_src],
            dst=new_feats[ii_dst],
        )
        # ins->ins passes logits directly (avoids a bf16 softmax precision loss).
        e1_ii_idx = torch.multinomial(
            torch.softmax(ii_logits.float(), dim=-1), 1
        ).squeeze(1)
        ii_batch_id = context.mol_t.batch[ii_spawn_src]
        e_t_ins_ii = self._renoise_edge_types(
            e1_ii_idx,
            context.time_next[ii_batch_id],
        )
        return BondsBetweenInserted(
            spawn_src_idx=ii_spawn_src,
            spawn_dst_idx=ii_spawn_dst,
            src_idx=ii_src,
            dst_idx=ii_dst,
            bond_type=e_t_ins_ii,
        )
