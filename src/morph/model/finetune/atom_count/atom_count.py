"""Atom-count-conditioned DiT and fine-tuning Lightning module."""

import math

import torch
from torch import nn

from morph.model.backbones.dit import DiTBackboneWithHeads
from morph.model.finetune.base import LightningModuleRatesFinetune


def atom_count_delta(mols):
    """Return target minus current count, recomputed after every topology edit."""
    target = getattr(mols, "target_n_atoms", None)
    if target is None:
        raise ValueError(
            "Atom-count conditioning requires a target_n_atoms system property "
            "with one final count per graph."
        )
    if target.shape != (mols.num_graphs,):
        raise ValueError("target_n_atoms must have shape (num_graphs,)")
    return target - mols.num_nodes_per_graph


class AtomCountDiTBackboneWithHeads(DiTBackboneWithHeads):
    """Add a small delta embedding to the existing time/current-count condition.

    A zero-initialized final projection preserves pretrained predictions at
    initialization. The fixed scale normalizes the signed scalar; it does not
    clip or round counts. No extra attention blocks are needed.
    """

    def __init__(self, *args, delta_scale=64.0, delta_hidden_dim=128, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(delta_scale) or delta_scale <= 0:
            raise ValueError("delta_scale must be finite and positive")
        self.delta_scale = float(delta_scale)
        cond_dim = int(kwargs["backbone_model_args"]["d_cond"])
        self.delta_embedding = nn.Sequential(
            nn.Linear(1, delta_hidden_dim),
            nn.SiLU(),
            nn.Linear(delta_hidden_dim, cond_dim, bias=False),
        )
        nn.init.zeros_(self.delta_embedding[-1].weight)

    def embed_inputs(self, mols_t, t):
        """Recompute the delta from the current graph and add its embedding."""
        atoms, edges, cond = super().embed_inputs(mols_t, t)
        delta = atom_count_delta(mols_t).to(cond.dtype).unsqueeze(-1)
        return atoms, edges, cond + self.delta_embedding(delta / self.delta_scale)

    def freeze_pretrained(self):
        """Keep the new scalar encoder and LoRA trainable when freezing the base."""
        before = sum(p.numel() for p in self.parameters() if p.requires_grad)
        super().freeze_pretrained()
        self.delta_embedding.requires_grad_(True)
        return before - sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightningModuleRatesAtomCountFinetune(LightningModuleRatesFinetune):
    """Report count control over every generated graph, including invalid ones."""

    def _validation_context(self, batch):
        return batch[0].target_n_atoms.detach().cpu()

    def _update_task_metrics(self, context, mols, rdkit_mols):
        counts = torch.tensor([mol.num_nodes for mol in mols])
        errors = (counts - context).abs().float().to(self.device)
        self.log_dict(
            {
                "val/atom_count_mae": errors.mean(),
                "val/atom_count_exact": (errors == 0).float().mean(),
            },
            on_step=False,
            on_epoch=True,
            batch_size=len(mols),
            sync_dist=True,
        )
