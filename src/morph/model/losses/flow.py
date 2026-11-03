"""Molecular flow losses from existing predictions, independent of Lightning."""

import torch
import torch.nn.functional as F

from morph.dataset.core.insertion_data import InsertedAtoms
from morph.model.losses.functional import (
    class_loss,
    do_action_loss,
    rate_loss,
    typed_gmm_loss,
    unimodal_insertion_loss,
    zero_dependency,
)
from morph.utils.graph_math import unsorted_segment_mean_and_count
from morph.utils.loss_accumulation import LossTerm


def _zero_module_dependency(module) -> torch.Tensor:
    """Keep every parameter in DDP's autograd graph without full reductions."""
    return sum(zero_dependency(parameter) for parameter in module.parameters())


class FlowLossMixin:
    """Share Morph's molecular objective with trainers that own their forward pass.

    The host supplies ``model`` (with ``insertion_head``, ``ins_edge_head`` and
    the two inserted-edge prediction methods), ``device``, ``representation``,
    ``insertion_head_mode``, the atom/edge/charge token weights, and a
    ``LossAccumulator`` named ``loss_accumulator``. This mixin neither constructs
    modules nor owns logging, optimization, or a training lifecycle.
    """

    def _node_loss_exclusion_mask(self, mols_t) -> torch.Tensor | None:
        """Optional per-node bool mask: True = drop this node from
        discrete non-deletion and substitution losses.

        Default ``None`` keeps all nodes.
        """
        return None

    def _position_loss_exclusion_mask(self, mols_t) -> torch.Tensor | None:
        """Optional per-node bool mask: True = drop this node from x loss."""
        return None

    def _charge_loss_exclusion_mask(self, mols_t) -> torch.Tensor | None:
        """Default to the discrete mask; tasks can supervise charges separately."""
        return self._node_loss_exclusion_mask(mols_t)

    def loss_from_predictions(self, preds, batch) -> torch.Tensor:
        """Return the weighted flow loss without running the model backbone.

        ``batch`` is ``(mols_t, mols_1, insertion_targets, t)``. Predictions use
        the ordinary Morph head contract and must follow ``mols_t`` node and
        directed-edge order; clean targets must share that ordering. The
        insertion-edge heads still run on target inserted atoms and the supplied
        ``h_latent``. All gradients through the predictions remain connected.

        Populate ``loss_accumulator`` with this batch's losses and diagnostics;
        callers decide when and how to log them. Task-specific node and position
        exclusion hooks apply exactly as in standalone training.
        """
        mols_t, mols_1, ins_targets, t = batch
        self.loss_accumulator.begin_batch()

        # Mask for nodes scheduled for deletion, reused for x, c, and a losses.
        # Deletion nodes have their target type/charge frozen to the prior sample, so
        # supervising them would bias predictions towards the prior distribution.
        to_delete_mask = mols_t.lambda_del > 0.0
        position_mask = ~to_delete_mask
        non_del_mask = position_mask

        # Optional per-node exclusion (e.g. scaffold atoms in fine-tuning).
        # ``exclusion_mask`` is True at nodes that should NOT contribute to discrete
        # non-deletion / substitution losses; defaults to ``None``.
        exclusion_mask = self._node_loss_exclusion_mask(mols_t)
        if exclusion_mask is not None:
            exclusion_mask = exclusion_mask.bool()
            non_del_mask = non_del_mask & ~exclusion_mask

        position_exclusion_mask = self._position_loss_exclusion_mask(mols_t)
        if position_exclusion_mask is not None:
            position_mask = position_mask & ~position_exclusion_mask.bool()

        charge_mask = ~to_delete_mask
        charge_exclusion_mask = self._charge_loss_exclusion_mask(mols_t)
        if charge_exclusion_mask is not None:
            charge_mask = charge_mask & ~charge_exclusion_mask.bool()
        charge = self._charge_loss(preds["charge_head"], mols_1, mols_t, charge_mask)
        losses = self._substitution_losses(
            preds, mols_t, mols_1, non_del_mask, to_delete_mask, exclusion_mask
        )
        # All current atoms may spawn insertions, including retained and
        # to-be-deleted atoms. Only deletion excludes an existing bond endpoint;
        # retained endpoints remain eligible for new bonds.
        losses.update(
            self._insertion_deletion_losses(preds, ins_targets, mols_t, ~to_delete_mask)
        )
        losses["x"] = self._position_loss(
            preds["pos_head"], mols_1, mols_t, position_mask
        )
        losses["c"] = charge
        self.loss_accumulator.set_batch_losses(losses, t=t)

        # Keep on-device (detach only); Lightning offloads these at log time.
        # A per-step .item() here forces a blocking GPU->CPU sync every step.
        self.loss_accumulator.add_stats(
            {
                "n_ins": (mols_t.lambda_ins > 0.0).sum().detach().float(),
                "n_del": (mols_t.lambda_del > 0.0).sum().detach().float(),
            }
        )

        loss = self.loss_accumulator.total_loss()

        # Keep every ins_edge_head parameter in the autograd graph for DDP.
        loss = loss + _zero_module_dependency(self.model.ins_edge_head)

        return loss

    def _charge_loss(self, c_pred, mols_1, mols_t, non_del_mask):
        """Per-graph charge classification loss (skipped in pointcloud mode)."""
        # NOTE take loss on all non-deletion nodes
        if self.representation.requires_charges:
            c_loss, c_batch_mask = class_loss(
                c_pred,
                mols_1.c,
                self.charge_token_weights,
                non_del_mask,
                mols_t.batch,
                mols_t.num_graphs,
                reduction="none",
            )
        else:
            # Pointcloud mode: c is the neutral dummy charge. Skip the loss
            # entirely (no information content) but keep charge_head in the graph.
            c_loss = torch.zeros(
                mols_t.num_graphs, 1, device=self.device
            ) + zero_dependency(c_pred)
            c_batch_mask = torch.zeros(
                mols_t.num_graphs, dtype=torch.bool, device=self.device
            )
        return LossTerm(c_loss, c_batch_mask)

    def _substitution_losses(
        self, preds, mols_t, mols_1, non_del_mask, to_delete_mask, exclusion_mask
    ):
        """Atom-type and edge-type substitution losses (do-action + class)."""
        a_pred = preds["atom_type_head"]
        e_pred = preds["edge_type_head"]
        do_sub_a_head = preds["do_sub_a_head"]
        do_sub_e_head = preds["do_sub_e_head"]

        # Atom substitutions
        # NOTE take loss on all nodes
        do_sub_a_loss = do_action_loss(
            do_sub_a_head,
            mols_t.lambda_a_sub,
            mols_t.batch,
            mols_t.num_graphs,
            reduction="none",
        )
        # NOTE take loss on all non-deletion & to-be-substituted nodes
        # del nodes have lambda_a_sub = 0 anyway, this is just for clarity
        to_sub_mask = (non_del_mask) & (mols_t.lambda_a_sub > 0.0)
        sub_a_class_loss, sub_a_batch_mask = class_loss(
            a_pred,
            mols_1.a,
            self.atom_type_weights,
            to_sub_mask,
            mols_t.batch,
            mols_t.num_graphs,
            reduction="none",
        )
        # Edge substitutions
        if self.representation.requires_topology:
            # Interpolation creates the clean target with a clone of the current
            # state's topology, and collation preserves that ordering. Select the
            # shared upper triangle directly instead of sorting both edge sets.
            edge_index = mols_t.edge_index
            # Boolean gathers each perform a synchronizing nonzero on CUDA.
            # Derive the shared index once for all six edge tensors.
            upper = (edge_index[0] < edge_index[1]).nonzero(as_tuple=True)[0]
            edge_index_triu = edge_index[:, upper]
            e_batch_triu = mols_t.batch[edge_index_triu[0]]
            e_pred_triu = e_pred[upper]
            do_sub_e_head_triu = do_sub_e_head[upper]
            e_target_triu = mols_1.e[upper]
            do_sub_e_target_triu = mols_t.lambda_e_sub[upper]

            # Supervise edge classes only on substituted, non-deletion edges.
            # Edges where either endpoint is scheduled for deletion are excluded:
            # their targets are frozen to the source value, so supervising e_pred
            # would train it to predict prior edge types — the wrong signal.
            _src_idx, _dst_idx = edge_index_triu
            non_del_edge_mask = ~to_delete_mask[_src_idx] & ~to_delete_mask[_dst_idx]
            if exclusion_mask is not None:
                # Exclude edges where BOTH endpoints are excluded (matches inference
                # masking: scaffold-scaffold edges can never change).
                non_del_edge_mask = non_del_edge_mask & ~(
                    exclusion_mask[_src_idx] & exclusion_mask[_dst_idx]
                )

            do_sub_e_loss = do_action_loss(
                do_sub_e_head_triu,
                do_sub_e_target_triu,
                e_batch_triu,
                mols_t.num_graphs,
                reduction="none",
            )

            to_sub_edge_mask = (non_del_edge_mask) & (do_sub_e_target_triu > 0.0)
            sub_e_class_loss, sub_e_batch_mask = class_loss(
                e_pred_triu,
                e_target_triu,
                self.edge_token_weights,
                to_sub_edge_mask,
                e_batch_triu,
                mols_t.num_graphs,
                reduction="none",
            )
        else:
            # Non-topology modes: runtime edges carry the inert <NO_BOND> token
            # and lambda_e_sub is identically zero, so the supervised edge-loss
            # path is both informationless and a DDP risk (the class_loss
            # short-circuit would drop edge_type_head from the autograd graph).
            # Use explicit graph-preserving dummies on the head outputs instead.
            do_sub_e_loss = torch.zeros(
                mols_t.num_graphs, 1, device=self.device
            ) + zero_dependency(do_sub_e_head)
            sub_e_class_loss = torch.zeros(
                mols_t.num_graphs, 1, device=self.device
            ) + zero_dependency(e_pred)
            sub_e_batch_mask = torch.zeros(
                mols_t.num_graphs, dtype=torch.bool, device=self.device
            )

        return {
            "do_sub_a": LossTerm(do_sub_a_loss),
            "sub_a_class": LossTerm(sub_a_class_loss, sub_a_batch_mask),
            "do_sub_e": LossTerm(do_sub_e_loss),
            "sub_e_class": LossTerm(sub_e_class_loss, sub_e_batch_mask),
        }

    def _insertion_deletion_losses(
        self, preds, ins_targets, mols_t, existing_endpoint_mask
    ):
        """Combine deletion decisions, insertion counts, content, and bond losses."""
        do_del_loss = do_action_loss(
            preds["do_del_head"],
            mols_t.lambda_del,
            mols_t.batch,
            mols_t.num_graphs,
            reduction="none",
        )
        # Supervise insertion counts at every node, including zero-count nodes.
        ins_loss_rate = rate_loss(
            preds["ins_rate_head"],
            mols_t.n_ins,
            mols_t.batch,
            mols_t.num_graphs,
            reduction="none",
        )
        if ins_targets.spawn_node_idx.numel() > 0:
            state = self._insertion_content_loss(
                preds["insertion_head"], ins_targets, mols_t
            )
            edges = self._insertion_edge_losses(
                preds, ins_targets, mols_t, existing_endpoint_mask
            )
        else:
            empty = torch.zeros(mols_t.num_graphs, 1, device=self.device)
            mask = torch.zeros(mols_t.num_graphs, dtype=torch.bool, device=self.device)
            state = LossTerm(
                empty + _zero_module_dependency(self.model.insertion_head), mask
            )
            edges = {"ins_e": LossTerm(empty, mask), "ins_e_ii": LossTerm(empty, mask)}
        return {
            "do_del": LossTerm(do_del_loss),
            "ins_rate": LossTerm(ins_loss_rate),
            "ins_state": state,
            **edges,
        }

    def _insertion_content_loss(self, insertion_pred, ins_targets, mols_t):
        """Supervise nearest insertions or all mixture targets, then reduce by graph."""
        spawn_node_idx = ins_targets.spawn_node_idx
        if self.insertion_head_mode == "unimodal":
            # One-to-one supervision: each spawning node predicts only its
            # single nearest pending insertion (no mixture marginalization).
            node_idx = ins_targets.nearest_node_idx
            ins_idx = ins_targets.nearest_ins_idx
            uni_dict_pred = {
                "mu": insertion_pred["mu"][node_idx],
                "a_logits": insertion_pred["a_logits"][node_idx],
                "c_logits": insertion_pred["c_logits"][node_idx],
            }

            uni_loss, uni_components = unimodal_insertion_loss(
                uni_dict_pred,
                ins_targets.positions[ins_idx],
                ins_targets.atom_types[ins_idx],
                ins_targets.charge_types[ins_idx],
                self.atom_type_weights,
                self.charge_token_weights,
                reduction="none",
                include_c=self.representation.requires_charges,
                return_components=True,
            )

            ins_loss_state = uni_loss.view(-1)

            # Diagnostic: log the per-component means (over the supervised
            # nearest nodes) so we can see whether position or atom-type/
            # charge dominates the next-atom loss. These share the single
            # ins_state weight; they are stats, not separately-weighted terms.
            self.loss_accumulator.add_stat(
                "ins_pos", uni_components["pos"].mean().detach()
            )
            self.loss_accumulator.add_stat(
                "ins_type", uni_components["type"].mean().detach()
            )
            if "charge" in uni_components:
                self.loss_accumulator.add_stat(
                    "ins_charge", uni_components["charge"].mean().detach()
                )
            # Keep c_logits in the autograd graph for DDP when dropped.
            if not self.representation.requires_charges:
                ins_loss_state = ins_loss_state + zero_dependency(
                    uni_dict_pred["c_logits"]
                )

            # reduce the loss to per-graph level (key on the nearest nodes)
            ins_loss_state, supervised_count = unsorted_segment_mean_and_count(
                ins_loss_state.view(-1, 1),
                mols_t.batch[node_idx],
                mols_t.num_graphs,
            )
        else:
            # Each pending insertion is supervised at its assigned spawning node.
            gmm_pred = {
                "mu": insertion_pred["mu"][spawn_node_idx],
                "sigma": insertion_pred["sigma"][spawn_node_idx],
                "pi": insertion_pred["pi"][spawn_node_idx],
                "a_probs": insertion_pred["a_probs"][spawn_node_idx],
                "c_probs": insertion_pred["c_probs"][spawn_node_idx],
            }

            gmm_loss, _ = typed_gmm_loss(
                gmm_pred,
                ins_targets.positions,
                ins_targets.atom_types,
                ins_targets.charge_types,
                self.atom_type_weights,
                self.charge_token_weights,
                reduction="none",
                include_c=self.representation.requires_charges,
            )

            ins_loss_state = gmm_loss.view(-1)
            # Keep c_probs in the autograd graph for DDP when its term is dropped.
            if not self.representation.requires_charges:
                ins_loss_state = ins_loss_state + zero_dependency(gmm_pred["c_probs"])

            # reduce the loss to per-graph level
            ins_loss_state, supervised_count = unsorted_segment_mean_and_count(
                ins_loss_state.view(-1, 1),
                mols_t.batch[spawn_node_idx],
                mols_t.num_graphs,
            )

        return LossTerm(ins_loss_state, supervised_count > 0)

    def _insertion_edge_losses(
        self, preds, ins_targets, mols_t, existing_endpoint_mask
    ):
        """Supervise bonds from future atoms to current atoms and to each other."""
        ins_loss_e = torch.zeros(
            mols_t.num_graphs, 1, device=self.device, dtype=preds["ins_rate_head"].dtype
        )
        ins_loss_e_ii = torch.zeros_like(ins_loss_e)
        ins_e_batch_mask = torch.zeros(
            mols_t.num_graphs, dtype=torch.bool, device=self.device
        )
        ins_e_ii_batch_mask = torch.zeros_like(ins_e_batch_mask)
        # Compute insertion edge loss if available
        # Index spaces:
        # - spawn/existing indices address current-state nodes in mols_t
        # - local indices address inserted nodes in ins_targets (future nodes)
        # - ins_edge_types provides integer edge-class supervision targets
        spawn_idx = ins_targets.bonds_to_existing.spawn_idx
        existing_idx = ins_targets.bonds_to_existing.existing_idx

        # Bundle the inserted-atom features once; gather per-edge subsets
        # below by indexing instead of repeating x/a/c gathers.
        ins_feats = InsertedAtoms(
            ins_targets.positions, ins_targets.atom_types, ins_targets.charge_types
        )

        # Insertion-edge prediction is only meaningful when chemical bonds
        # are part of the representation. In non-topology modes, skip the
        # head call entirely and emit a parameter-sum dummy so DDP still
        # sees ins_edge_head as participating.
        if (
            self.representation.requires_topology
            and spawn_idx.numel() > 0
            and existing_idx.numel() > 0
        ):
            # The spawning atom supplies context and may later disappear. Only
            # the actual existing endpoint needs a valid clean bond target.
            ins_edge_mask = existing_endpoint_mask[existing_idx]
            # Get edge predictions using the insertion edge head. Empty
            # supervision masks are handled tensor-natively by class_loss;
            # branching on ``.any()`` here would synchronize the GPU.
            ins_edge_logits = self.model.predict_inserted_to_existing_edges(
                mols_t=mols_t,
                out_dict=preds,
                spawn_node_idx=spawn_idx,
                existing_node_idx=existing_idx,
                # Gather per-edge inserted attrs from canonical future nodes.
                # NOTE predict edge types for future nodes, not current nodes
                # NOTE then at inference we will adjust to current noise level
                ins=ins_feats[ins_targets.bonds_to_existing.inserted_idx],
            )
            if ins_edge_logits is not None and ins_edge_logits.numel() > 0:
                # Reuse the shared class loss helper (same masking +
                # per-graph averaging) to compute insertion edge loss.
                ins_loss_e, ins_e_batch_mask = class_loss(
                    ins_edge_logits,
                    ins_targets.bonds_to_existing.bond_type,
                    self.edge_token_weights,
                    ins_edge_mask,
                    mols_t.batch[spawn_idx],
                    mols_t.num_graphs,
                    reduction="none",
                )
        # Compute ins -> ins edge loss (gated on topology for the same reason).
        spawn_src_ii = ins_targets.bonds_between_inserted.spawn_src_idx
        spawn_dst_ii = ins_targets.bonds_between_inserted.spawn_dst_idx
        if self.representation.requires_topology and spawn_src_ii.numel() > 0:
            edges_ii_src = ins_targets.bonds_between_inserted.src_idx
            edges_ii_dst = ins_targets.bonds_between_inserted.dst_idx
            predictor = self.model.predict_inserted_to_inserted_edges
            ins_ii_logits = predictor(
                mols_t=mols_t,
                out_dict=preds,
                spawn_src_idx=spawn_src_ii,
                spawn_dst_idx=spawn_dst_ii,
                src=ins_feats[edges_ii_src],
                dst=ins_feats[edges_ii_dst],
            )
            if ins_ii_logits is not None and ins_ii_logits.numel() > 0:
                # Both future endpoints survive, so all pairs are supervised
                # regardless of either spawning atom's deletion/retention mask.
                ins_loss_e_ii, ins_e_ii_batch_mask = class_loss(
                    ins_ii_logits,
                    ins_targets.bonds_between_inserted.bond_type,
                    self.edge_token_weights,
                    torch.ones_like(spawn_src_ii, dtype=torch.bool),
                    mols_t.batch[spawn_src_ii],
                    mols_t.num_graphs,
                    reduction="none",
                )

        return {
            "ins_e": LossTerm(ins_loss_e, ins_e_batch_mask),
            "ins_e_ii": LossTerm(ins_loss_e_ii, ins_e_ii_batch_mask),
        }

    def _position_loss(self, x_pred, mols_1, mols_t, position_mask):
        """Per-graph coordinate MSE over the selected supervised nodes."""
        # The graph-preserving dependency below handles an empty selection.
        x_loss = F.mse_loss(
            x_pred[position_mask], mols_1.x[position_mask], reduction="none"
        )
        x_loss, supervised_count = unsorted_segment_mean_and_count(
            x_loss, mols_t.batch[position_mask], mols_t.num_graphs
        )
        x_batch_mask = supervised_count > 0

        # Unconditionally retain the position output in the graph. This is a
        # no-op for ordinary batches and covers the all-deleted edge case without
        # a blocking device-to-host ``x_batch_mask.any()`` branch.
        x_loss = x_loss + zero_dependency(x_pred)

        return LossTerm(x_loss, x_batch_mask)
