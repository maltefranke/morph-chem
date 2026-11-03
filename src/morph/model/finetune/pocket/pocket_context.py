"""Encode fixed protein atoms while leaving replay examples unconditional."""

from __future__ import annotations

import hydra
import torch
from torch_geometric.utils import to_dense_batch

from morph.model.finetune.base import (
    ConditioningDiTBackboneWithHeads,
    DistanceAttnBias,
    LightweightContextEncoder,
    build_distance_attn_bias,
)


class PocketContextEncoder(LightweightContextEncoder):
    """Protein atom encoder supporting empty context rows and entire empty batches."""

    @torch.compiler.disable
    def forward(self, pocket):
        """Mask absent context without NaNs or unused parameters in replay batches."""
        tokens = self.token_proj(self.base_feats(pocket.a, pocket.c, pocket.x))
        dense, mask = to_dense_batch(
            tokens,
            pocket.batch,
            batch_size=pocket.num_graphs,
            max_num_nodes=max(1, pocket.max_seqlen),
        )
        # Keep a zero-gradient dependency even when the input has no atoms.
        dense = dense + tokens.sum() * 0.0
        safe_mask = mask.clone()
        safe_mask[~mask.any(dim=1), 0] = True
        dense = self.apply_self_attn(dense, safe_mask)
        return dense.masked_fill(~mask.unsqueeze(-1), 0.0), mask


class PocketConditionalDiTBackboneWithHeads(ConditioningDiTBackboneWithHeads):
    """Cross-attend to a pocket, with exact adapter bypass for empty context rows."""

    def __init__(
        self,
        *args,
        encoder_args,
        use_distance_bias=True,
        dist_bias_init_gamma=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encoder = hydra.utils.instantiate(encoder_args)
        self.dist_bias = (
            DistanceAttnBias(dist_bias_init_gamma) if use_distance_bias else None
        )

    @torch.compiler.disable
    def _encode_pocket(self, mols_t):
        pocket = mols_t.contexts["pocket"]
        tokens, mask = self.encoder(pocket)
        bias = None
        if self.dist_bias is not None:
            bias = build_distance_attn_bias(
                self.dist_bias,
                mols_t.x,
                mols_t.batch,
                int(mols_t.max_seqlen),
                pocket.x,
                pocket.batch,
                tokens.shape[1],
                tokens.shape[0],
            )
        return tokens, mask, bias

    def _build_conditioning(self, mols_t, cond, force_uncond):
        if (
            not force_uncond
            and "pocket" in mols_t.contexts
            and cond["cond_tokens"] is None
        ):
            tokens, mask, bias = self._encode_pocket(mols_t)
            cond["cond_tokens"] = tokens
            if cond["cond_token_mask"] is None:
                cond["cond_token_mask"] = mask
            if cond["cond_token_attn_bias"] is None:
                cond["cond_token_attn_bias"] = bias
        return cond
