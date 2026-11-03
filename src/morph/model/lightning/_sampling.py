"""Inference: prediction and trajectory sampling.

Mixin split out of :class:`morph.model.lightning.base.LightningModuleRates`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from morph.dataset.core.atomic_data import (
    batch_from_flat,
    batch_translate,
    iter_geometry,
)
from morph.dataset.core.representation import neutral_charge_index
from morph.flow_matching.integration import IntegrationPredictions
from morph.flow_matching.sampling_step import SamplingStepTrace
from morph.utils import rdkit_utils as morphRD

if TYPE_CHECKING:
    from collections.abc import Callable


def _sample_logits(logits: torch.Tensor) -> torch.Tensor:
    """Sample categorical logits in fp32 without a distribution wrapper."""
    return torch.multinomial(torch.softmax(logits.float(), dim=-1), 1).squeeze(1)


@dataclass
class _OutputFrame:
    """Restore physical units and, for fixed scaffolds, the input coordinate frame."""

    coordinate_std: float
    translation: torch.Tensor | None

    @classmethod
    def from_batch(cls, mol, coordinate_std, *, restore_input_frame):
        mean = mol.remove_com()
        return cls(
            coordinate_std, mean.detach().clone() if restore_input_frame else None
        )

    def recenter(self, mol):
        mean = mol.remove_com()
        if self.translation is not None:
            self.translation.add_(mean.detach())

    def restore(self, mol):
        if self.translation is not None:
            batch_translate(mol, self.translation)
        for graph in iter_geometry(mol):
            graph.positions.mul_(self.coordinate_std)
        return mol

    def snapshot(self, mol):
        return self.restore(mol.clone()).to_inplace("cpu", non_blocking=False)


def _sampling_interval(step_sizes, t_start):
    """Select a schedule suffix, snapping its start to the next grid point."""
    if t_start <= 0.0:
        return 0.0, step_sizes
    cumulative = torch.cumsum(
        torch.tensor([0.0] + [float(s) for s in step_sizes]), dim=0
    )
    start = int((cumulative >= float(t_start)).nonzero()[0].item())
    return float(cumulative[start].item()), step_sizes[start:]


class SamplingMixin:
    def _apply_inference_charge_mask(self, mol_t, c_pred):
        """Optionally preserve task-constrained charges on existing atoms."""
        return c_pred

    def _apply_inference_edit_masks(
        self,
        mol_t,
        x1_pred: torch.Tensor,
        do_sub_a_probs: torch.Tensor,
        do_sub_e_probs: torch.Tensor,
        do_del_probs: torch.Tensor,
        num_ins_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optionally pin clean-state predictions before integration.

        Conditional subclasses override this to preserve their scaffold.
        """
        return x1_pred, do_sub_a_probs, do_sub_e_probs, do_del_probs, num_ins_pred

    def predict_step(self, batch, batch_idx):
        return_traj = bool(getattr(self, "predict_return_traj", False))

        gen_mols = self.sample(
            batch,
            batch_idx,
            return_traj=return_traj,
        )
        if not return_traj:
            # Transfer once per batch, then retain the established list-of-
            # frames interface with one final frame per attempted molecule.
            final_mols = gen_mols.to_inplace("cpu", non_blocking=False).to_data_list()
            gen_mols = [[mol] for mol in final_mols]

        # In non-topology modes, runtime edges labeled <NO_BOND> are not chemical
        # bonds. Skip the validity pipeline and return the raw samples.
        if not self.representation.requires_topology:
            return {
                "all_mols": list(gen_mols),
                "is_valid": [False] * len(gen_mols),
                "valid_mols": [],
                "invalid_mols": list(gen_mols),
                "invalid_mols_rdkit": [],
            }

        # do quick validity check of the generated molecules
        # take the last state of the trajectory and check validity
        last_mols_rdkit = [
            i[-1].to_rdkit_mol(
                self.vocab.atom_tokens, self.vocab.edge_tokens, self.vocab.charge_tokens
            )
            for i in gen_mols
        ]
        mol_is_valid = []
        for mol in last_mols_rdkit:
            if mol is None:
                mol_is_valid.append(False)
                continue
            try:
                mol_is_valid.append(
                    morphRD.mol_is_valid(mol, allow_charged=self.allow_charged)
                )
            except Exception as e:
                print(f"Error checking validity of molecule: {e}")
                mol_is_valid.append(False)

        valid_mols = [
            i for i, valid in zip(gen_mols, mol_is_valid, strict=True) if valid
        ]
        # Keep trajectories for invalid molecules so failures can be inspected.
        invalid_mols = [
            i for i, valid in zip(gen_mols, mol_is_valid, strict=True) if not valid
        ]
        invalid_mols_rdkit = [
            i
            for i, valid in zip(last_mols_rdkit, mol_is_valid, strict=True)
            if not valid
        ]

        # ``all_mols`` / ``is_valid`` keep the dataloader batch order. The
        # valid/invalid split re-orders molecules, so consumers that join
        # per-molecule outputs must read the batch-ordered keys.
        return {
            "all_mols": list(gen_mols),
            "is_valid": mol_is_valid,
            "valid_mols": valid_mols,
            "invalid_mols": invalid_mols,
            "invalid_mols_rdkit": invalid_mols_rdkit,
        }

    def _integrate_step(self, mol_t, predictions: IntegrationPredictions, **kwargs):
        """Run one integration step on native Morph graph batches."""
        return self.integrator.integrate_step_gnn(
            mol_t=mol_t, predictions=predictions, **kwargs
        )

    def _decode_predictions(self, mol_t, preds, *, requires_topology, force_charge_idx):
        """Sample clean endpoint tokens, then apply representation and task masks."""
        # Extract predictions
        x1_pred = preds["pos_head"]  # (N_total, D)

        a_pred = _sample_logits(preds["atom_type_head"])
        c_pred = _sample_logits(preds["charge_head"])

        # NOTE: predictions are for full adj matrix.
        # NOTE: Will take triu and resymmetrize in integration step
        e_pred = _sample_logits(preds["edge_type_head"])

        insertion_pred = preds["insertion_head"]

        # Extract scalar decision logits; integration consumes probabilities.
        do_sub_a_probs = torch.sigmoid(preds["do_sub_a_head"].view(-1))
        do_sub_e_probs = torch.sigmoid(preds["do_sub_e_head"].view(-1))
        do_del_probs = torch.sigmoid(preds["do_del_head"].view(-1))

        # Channels absent from the representation retain their inert tokens.
        if not requires_topology:
            e_pred.fill_(0)  # all <NO_BOND>
            do_sub_e_probs.zero_()  # never substitute an edge
        if force_charge_idx is not None:
            c_pred.fill_(force_charge_idx)
        else:
            c_pred = self._apply_inference_charge_mask(mol_t, c_pred)

        # Extract insertion rate prediction (number of insertions per node)
        num_ins_pred = preds["ins_rate_head"].view(-1)

        # Subclass hook: scaffold fine-tuning hard-zeros del / sub-a / sub-e
        # (scaffold-scaffold) probs at scaffold positions and optionally pins
        # their clean coordinate target. This must run before mol_1_pred is built.
        x1_pred, do_sub_a_probs, do_sub_e_probs, do_del_probs, num_ins_pred = (
            self._apply_inference_edit_masks(
                mol_t,
                x1_pred,
                do_sub_a_probs,
                do_sub_e_probs,
                do_del_probs,
                num_ins_pred,
            )
        )

        # Clean-target prediction over mol_t's current topology.
        mol_1_pred = batch_from_flat(
            positions=x1_pred,
            atom_types=a_pred,
            charge_types=c_pred,
            edge_types=e_pred,
            edge_index=mol_t.edge_index,
            batch_idx=mol_t.batch,
            num_graphs=mol_t.num_graphs,
            _already_sorted=True,
            _node_counts=mol_t.num_nodes_per_graph,
            _edge_counts=mol_t.num_edges_per_graph,
        )

        h_latent = preds["h_latent"]

        return IntegrationPredictions(
            mol_1_pred=mol_1_pred,
            do_sub_a_probs=do_sub_a_probs,
            do_sub_e_probs=do_sub_e_probs,
            do_del_probs=do_del_probs,
            num_ins_pred=num_ins_pred,
            insertion_preds=insertion_pred,
            h_latent=h_latent,
        )

    def sample(
        self,
        batch,
        batch_idx,
        return_traj: bool = False,
        t_start: float = 0.0,
        sampling_step_callback: Callable[[SamplingStepTrace], None] | None = None,
        position_update: Callable | None = None,
    ):
        """Inference step for flow matching.

        Args:
            batch: Tuple ``(mol_t, mol_1)`` from the dataloader.
            batch_idx: Batch index.
            return_traj: Return per-molecule trajectories instead of the final state.
            t_start: If > 0, integrate only ``[t_start, 1]`` (snapped to a
                schedule time point) starting from ``batch[0]`` as the state at
                ``t_start``. Used by the renoise self-correction ablation.
            sampling_step_callback: Optional callback receiving a
                ``SamplingStepTrace`` after each integration step. It is only
                called when supplied, so normal inference has no trace overhead.
            position_update: Optional coordinate update forwarded to the integrator.

        Returns:
            If ``return_traj`` is False: a ``ChemBatch`` with the final state.
            If ``return_traj`` is True: a per-molecule list of trajectory frames.

        Sampling stays centered internally. When kept positions are frozen, the
        accumulated translations are restored on output so kept coordinates and
        attached geometric context remain in the input frame.
        """
        model = self._get_model()
        compiled_model = getattr(
            model, "_compiled_call_impl", None
        ) is not None or hasattr(model, "_orig_mod")
        if self.vocab.atom_tokens is None:
            raise ValueError(
                "vocab.atom_tokens must be supplied when constructing "
                "the sampling module."
            )

        # Get batch size from the batch (assuming it's similar to training)
        mol_t, _mol_1 = batch

        batch_size = mol_t.batch_size

        output_frame = _OutputFrame.from_batch(
            mol_t,
            self.distributions.coordinate_std,
            restore_input_frame=bool(
                getattr(self, "_freeze_keep_positions", False)
                and getattr(mol_t, "keep_mask", None) is not None
            ),
        )

        t0, step_sizes = _sampling_interval(self.integrator.get_time_steps(), t_start)
        t = torch.full((batch_size,), t0, device=self.device)

        # Trajectory storage (only used when return_traj=True).
        # Frames live on CPU to avoid linear GPU growth over the integration loop.
        mol_traj = [output_frame.snapshot(mol_t)] if return_traj else None

        # Per-trajectory conditioning that is invariant across integration steps
        # (e.g. scaffold tokens encoded from the fixed start molecule) is
        # computed once here and threaded into every step,
        # instead of the model re-encoding it each step. Models without extra
        # conditioning don't define the hook -> empty dict -> unchanged behaviour.
        cond_kwargs: dict = {}
        _precompute = getattr(model, "precompute_conditioning", None)
        if _precompute is not None:
            cond_kwargs = _precompute(mol_t)

        requires_topology = self.representation.requires_topology
        requires_charges = self.representation.requires_charges
        ins_edge_head = (
            getattr(model, "ins_edge_head", None) if requires_topology else None
        )
        force_charge_idx = (
            None if requires_charges else neutral_charge_index(self.vocab)
        )

        integration_extensions = {}
        if sampling_step_callback is not None:
            integration_extensions["return_step_trace"] = True
        if position_update is not None:
            integration_extensions["position_update"] = position_update

        # Integration loop: integrate from t=0 to t=1
        for step_index, step_size in enumerate(step_sizes):
            is_final = step_index == len(step_sizes) - 1
            state_before = mol_t

            # Resolve the dynamic graph size before entering Dynamo, as on the
            # training path. Reading its CUDA scalar inside the model breaks
            # compilation after topology changes invalidate the cached size.
            if compiled_model:
                cond_kwargs["max_seqlen"] = mol_t.max_seqlen
            preds = model(mol_t, t.view(-1, 1), **cond_kwargs)

            predictions = self._decode_predictions(
                mol_t,
                preds,
                requires_topology=requires_topology,
                force_charge_idx=force_charge_idx,
            )

            # Integrate one step (edge prediction happens inside if head is provided)
            step_result = self._integrate_step(
                mol_t=mol_t,
                predictions=predictions,
                t=t,
                dt=step_size,
                is_final=is_final,
                ins_edge_head=ins_edge_head,
                force_charge_idx=force_charge_idx,
                **integration_extensions,
            )
            if sampling_step_callback is None:
                mol_t = step_result
                integration_trace = None
            else:
                mol_t, integration_trace = step_result

            # Release decoded predictions before the next forward pass.
            del predictions
            output_frame.recenter(mol_t)

            if sampling_step_callback is not None:
                sampling_step_callback(
                    SamplingStepTrace(
                        mol_t=state_before,
                        mol_t_next=mol_t,
                        t=t,
                        dt=step_size,
                        preds=preds,
                        integration=integration_trace,
                        is_final=is_final,
                    )
                )
            del preds, state_before, integration_trace, step_result

            # Update time forward
            # Number of graphs stays constant (batch_size)
            t = t + step_size

            # Save state to trajectory on CPU; non-traj path only keeps the final frame.
            if return_traj:
                mol_traj.append(output_frame.snapshot(mol_t))

        if return_traj:
            # rectify the trajectory such that we get a traj for each molecule
            traj_lists = [mol_traj_i.to_data_list() for mol_traj_i in mol_traj]

            return [list(frames) for frames in zip(*traj_lists, strict=True)]

        # The final integration state is local and can be scaled in place.
        return output_frame.restore(mol_t)
