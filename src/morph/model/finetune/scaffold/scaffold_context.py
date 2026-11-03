"""Scaffold conditioning for scaffold-decoration fine-tuning.

Two conditioning signals, **always on**, derived from ``mols_t.keep_mask``
(both recomputed each forward so they track the integrator's del/ins):

* ``node_cond`` — per-atom scaffold membership (0 = decoration, 1 = keep), via
  the shared :class:`~morph.model.finetune.base.KeepNodeCond`.
* ``edge_cond`` — per-edge scaffold class (dec-dec / scaffold-dec /
  scaffold-scaffold), via the shared
  :class:`~morph.model.finetune.base.KeepEdgeCond`.

Plus an **optional** cross-attention stream, toggled by the backbone's
``use_cross_attn`` flag:

* **Option 1** (``use_cross_attn: false``) — node + edge conditioning only.
* **Option 2** (``use_cross_attn: true``) — node + edge + cross-attention. The
  scaffold atoms are extracted from ``mols_t`` (induced subgraph via
  :func:`~morph.dataset.core.atomic_data.filter_nodes`) and encoded into context
  tokens by a :class:`~morph.model.finetune.base.LightweightContextEncoder`.

Scaffold decoration carries no graph-level property condition. The per-node,
per-edge, and distance-bias building blocks live in
:mod:`morph.model.finetune.base.context_encoder`.
"""

from __future__ import annotations

import hydra
import torch

from morph.dataset.core.atomic_data import ChemBatch, filter_nodes
from morph.model.finetune.base import (
    ConditioningDiTBackboneWithHeads,
    CondPorts,
    DistanceAttnBias,
    KeepEdgeCond,
    KeepNodeCond,
    LightweightContextEncoder,
    build_distance_attn_bias,
)


class ScaffoldConditionalDiTBackboneWithHeads(ConditioningDiTBackboneWithHeads):
    """:class:`ConditionalDiTBackboneWithHeads` with built-in scaffold context.

    On every forward, builds ``node_cond`` + ``edge_cond`` from
    ``mols_t.keep_mask`` (when the matching backbone port is enabled) and,
    when ``use_cross_attn`` is set, encodes the scaffold subgraph into
    ``cond_tokens`` via a shared ``LightweightContextEncoder``. Caller-supplied
    kwargs always win (the inference-time caching hook: pass ``cond_tokens``
    directly to skip the encoder).

    Extra Hydra-instantiable args on top of the parent:
      * ``xattn_encoder_args`` — the ``LightweightContextEncoder`` config;
        required iff the backbone has ``use_cross_attn: true``.
      * ``use_distance_bias`` / ``dist_bias_init_gamma`` — geometric
        correspondence prior (query atom ↔ co-located scaffold token).
    """

    def __init__(
        self,
        *args,
        xattn_encoder_args: dict | None = None,
        use_distance_bias: bool = True,
        dist_bias_init_gamma: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        bb = self.backbone
        self.node_cond_encoder = KeepNodeCond(bb.d_model) if bb.use_node_cond else None
        self.edge_cond_encoder = (
            KeepEdgeCond(bb.in_edge_nf) if bb.use_edge_cond else None
        )

        # Optional cross-attention (option 2): encode the scaffold subgraph with
        # a lightweight encoder shared by conditional fine-tuning tasks.
        if bb.use_cross_attn:
            if xattn_encoder_args is None:
                raise ValueError(
                    "backbone.use_cross_attn is true but xattn_encoder_args is None; "
                    "provide the LightweightContextEncoder config, or set "
                    "use_cross_attn=false for node+edge-only scaffold conditioning."
                )
            self.xattn_encoder: LightweightContextEncoder | None = (
                hydra.utils.instantiate(xattn_encoder_args)
            )
            self.dist_bias = (
                DistanceAttnBias(dist_bias_init_gamma) if use_distance_bias else None
            )
        else:
            self.xattn_encoder = None
            self.dist_bias = None

    def freeze_pretrained(self) -> int:
        """Freeze the clean lane; the scaffold conditioning modules stay
        trainable (they carry no pretrained body).
        """
        frozen = super().freeze_pretrained()
        if self.xattn_encoder is not None:
            frozen += self.xattn_encoder.freeze_pretrained_body()
        return frozen

    # filter_nodes + the encoder run data-dependent shape ops dynamo can't trace;
    # isolate them so the rest of the model still compiles.
    @torch.compiler.disable
    def _encode_scaffold_cross_attn(
        self, mols_t: ChemBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encode the scaffold subgraph of ``mols_t`` into cross-attn tokens."""
        scaffold = filter_nodes(mols_t, mols_t.keep_mask.bool())
        cond_tokens, cond_token_mask = self.xattn_encoder(scaffold)
        cond_token_attn_bias = None
        if self.dist_bias is not None:
            cond_token_attn_bias = build_distance_attn_bias(
                self.dist_bias,
                mols_t.x,
                mols_t.batch,
                int(mols_t.max_seqlen),
                scaffold.x,
                scaffold.batch,
                cond_tokens.shape[1],
                cond_tokens.shape[0],
            )
        return cond_tokens, cond_token_mask, cond_token_attn_bias

    def _build_conditioning(
        self, mols_t: ChemBatch, cond: CondPorts, force_uncond: bool
    ) -> CondPorts:
        if getattr(mols_t, "keep_mask", None) is not None:
            if cond["node_cond"] is None and self.node_cond_encoder is not None:
                cond["node_cond"] = self.node_cond_encoder(mols_t)
            if cond["edge_cond"] is None and self.edge_cond_encoder is not None:
                cond["edge_cond"] = self.edge_cond_encoder(mols_t)
            if (
                not force_uncond
                and self.xattn_encoder is not None
                and cond["cond_tokens"] is None
            ):
                cond["cond_tokens"], cond["cond_token_mask"], sc_bias = (
                    self._encode_scaffold_cross_attn(mols_t)
                )
                if cond["cond_token_attn_bias"] is None:
                    cond["cond_token_attn_bias"] = sc_bias
        return cond
