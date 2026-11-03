"""
Conditional DiT backbone with four **independent,
opt-in** conditioning ports (cross-attention / graph / node / edge). With every
port disabled it is functionally identical to the plain ``DiTBackbone``; each
enabled port adds a zero-init gated stream.

**Embed-outside contract.** The backbone performs *no* modality-specific
embedding. Every conditioning input arrives already at its port's fusion width
from an external encoder, and the backbone only does zero-init ``tanh``-gated
fusion (plus the intrinsic per-block attention K/V projection). A freshly loaded
unconditional DiT is therefore bit-identical to its baseline until the gates
train away from 0. Each port is skipped (free pass-through) when its kwarg is
``None``.

Callers are responsible for providing correctly scaled, pre-embedded conditioning
tokens/features and checking the resulting gated update scale at the chosen port.
The adapter's query/KV LayerNorms do not calibrate its residual update to the raw
DiT stream; uniformly rescaling context tokens can be canceled by KV LayerNorm.

* **Cross-attention conditioning** (``cond_tokens`` ``(B, K, ctx_dim)``): atoms
  attend to a set of ``K`` external context tokens via gated cross-attention
  (Flamingo-style). ``cross_attn_position="residual"`` (default) inserts it before
  each ``DiTBlock`` on the raw stream. Opt-in ``"post_norm1"`` instead conditions
  the block's normalized attention input before native AdaLN, preserving the raw
  residual skip. The tokens are *not* tied
  to any one modality or to a sequential structure — today they are per-atom
  encodings of a scaffold, but they can
  be any token set (e.g. language tokens). ``K`` is arbitrary and need not
  relate to the atom count. Logits accept an additive
  per-pair bias ``(B, K, N_padded)`` to seed prior coupling between specific
  tokens and atoms.

* **Graph conditioning** (``graph_cond`` ``(B, d_model)``): added to the
  projected time/current-count condition ``c`` through ``tanh(graph_gate)``
  right after ``self.dit.encode``. One embedding per graph, used by every
  DiT block and the decoder.

* **Node conditioning** (``node_cond`` ``(N, d_model)``): added to the per-atom
  hidden state ``h`` through ``tanh(node_gate)`` right after ``self.dit.encode``.
  1-1 aligned with the denoised atoms (ragged batch order), so no attention.

* **Edge conditioning** (``edge_cond`` ``(E, in_edge_nf)``): added to
  ``e_embed`` through ``tanh(edge_cond_gate)`` after the transformer blocks,
  before edge decoding. It does not enter the attention pair bias. Already
  1-1 aligned with edges, so no attention.

Each port's parameters (the ``cross_attn_blocks`` and graph/node/edge gates) are
only built when its ``use_*`` flag is set, so unused parameters never enter
the graph (the trainer runs ``ddp_find_unused_parameters_false``).

Composition (not inheritance): :class:`ConditionalDiTBackbone` *holds* a
:class:`morph.model.backbones.dit.DiTBackbone` instance as ``self.dit`` and,
when ``use_cross_attn`` is set, a parallel ``self.cross_attn_blocks`` ModuleList.
Its forward calls the DiT's :meth:`encode`/:meth:`decode` helpers and iterates
``self.dit.blocks``, applying cross-attention at the configured insertion point.
To load a pretrained DiT, use :meth:`ConditionalDiTBackbone.load_dit_state_dict`,
which forwards to ``self.dit.load_state_dict``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from morph.model.backbones import dit as _dit
from morph.model.backbones.dit import (
    DiTBackbone,
    DiTBackboneWithHeads,
    SwiGLU,
    _swiglu_width,
)
from morph.model.layers.attention import compute_dense_index, pad_dense, unpad_dense
from morph.utils.graph_math import batched_triu_edge_layout

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from morph.dataset.core.atomic_data import ChemBatch


# ---------------------------------------------------------------------------
#  Gated cross-attention block (Flamingo-style)
# ---------------------------------------------------------------------------


class GatedCrossAttention(nn.Module):
    """Gated cross-attention: atoms (queries) attend to conditioning tokens.

    Both the attention and feed-forward branches are wrapped in a tanh gate
    on a learnable scalar initialised to 0, so the layer is the identity at
    init (Flamingo, https://arxiv.org/abs/2204.14198).

    Attention logits accept an additive per-pair bias of shape ``(B, K, N)``
    (token-then-atom indexing); ``bias[b, k, n]`` is added to the logit of
    atom ``n`` attending to token ``k``.
    """

    def __init__(
        self,
        hidden_dim: int,
        ctx_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(ctx_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.kv_proj = nn.Linear(ctx_dim, 2 * hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.attn_gate = nn.Parameter(torch.zeros(1))

        mlp_hidden = _swiglu_width(hidden_dim, mlp_ratio, 64)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, mlp_hidden)
        self.ffn_gate = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.zeros_(self.attn_gate)
        nn.init.zeros_(self.ffn_gate)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor | None,
        attn_bias: torch.Tensor | None,
        query_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, d) atom features (queries).
            ctx: (B, K, d_ctx) conditioning tokens.
            ctx_mask: (B, K) bool, True for valid tokens. ``None`` attends to all.
            attn_bias: additive bias on attention logits, or ``None``. Either
                (B, K, N) shared across heads or (B, H, K, N) per head.
            query_mask: (B, N) bool, True for valid atoms. Used to zero out
                cross-attention output at padded query positions.

        Returns:
            (B, N, d) updated features.
        """
        B, N, _ = x.shape
        K = ctx.shape[1]
        H, Dh = self.num_heads, self.head_dim

        # A padded mixed-context batch can contain K=0 examples alongside
        # conditioned examples. Those rows must be an exact identity through
        # BOTH branches of this adapter, matching an all-K=0 batch where the
        # cross-attention port is skipped altogether.
        if K == 0:
            return x
        has_context = None
        safe_ctx_mask = ctx_mask
        if ctx_mask is not None:
            has_context = ctx_mask.any(dim=-1)
            if bool((~has_context).any()):
                # Avoid an all--inf SDPA row. Its arbitrary placeholder output is
                # removed below together with the independent FFN residual.
                safe_ctx_mask = ctx_mask.clone()
                safe_ctx_mask[~has_context, 0] = True

        q = self.q_proj(self.norm_q(x))
        kv = self.kv_proj(self.norm_kv(ctx))
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)
        k = k.view(B, K, H, Dh).transpose(1, 2)
        v = v.view(B, K, H, Dh).transpose(1, 2)

        attn_mask: torch.Tensor | None = None
        if attn_bias is not None or ctx_mask is not None:
            if attn_bias is not None:
                # -> (B, H_or_1, N, K). A 4-D bias is already per-head; a 3-D one
                # gets a singleton head axis and broadcasts inside SDPA.
                attn_mask = (
                    attn_bias.transpose(-1, -2)
                    if attn_bias.dim() == 4
                    else attn_bias.transpose(1, 2).unsqueeze(1)
                ).to(q.dtype)
            else:
                attn_mask = torch.zeros((B, 1, N, K), dtype=q.dtype, device=q.device)
            if safe_ctx_mask is not None:
                attn_mask = attn_mask.masked_fill(
                    ~safe_ctx_mask[:, None, None, :], float("-inf")
                )

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_proj(out)

        if query_mask is not None:
            out = out * query_mask.unsqueeze(-1).to(out.dtype)
        if has_context is not None:
            out = out * has_context[:, None, None].to(out.dtype)

        x = x + torch.tanh(self.attn_gate) * out
        ffn_out = self.ffn(self.norm_ffn(x))
        if has_context is not None:
            ffn_out = ffn_out * has_context[:, None, None].to(ffn_out.dtype)
        x = x + torch.tanh(self.ffn_gate) * ffn_out
        return x


# ---------------------------------------------------------------------------
#  Backbone (composes a DiTBackbone with optional gated conditioning ports)
# ---------------------------------------------------------------------------


class ConditionalDiTBackbone(nn.Module):
    """DiT backbone with four opt-in gated conditioning ports (§ module doc).

    Holds a :class:`DiTBackbone` (``self.dit``) plus, when ``use_cross_attn`` is
    set, a parallel ``ModuleList`` of :class:`GatedCrossAttention` modules
    (``self.cross_attn_blocks``). The forward delegates input projection / final
    layer / heads to ``self.dit.encode`` / ``self.dit.decode``, iterates
    ``self.dit.blocks``, and applies gated cross-attention when the port is enabled
    and ``cond_tokens`` is supplied. ``cross_attn_position`` selects the original
    ``"residual"`` placement or opt-in ``"post_norm1"`` placement before AdaLN.
    Keep this choice in the saved model config; it adds no state-dict keys.

    Loading a pretrained vanilla DiT is just::

        model.load_dit_state_dict(dit_backbone_sd)

    which forwards to ``self.dit.load_state_dict`` — the conditioning modules
    keep their zero-gated init so the augmented model behaves
    identically to the loaded DiT until the gates train away from 0.
    """

    def __init__(
        self,
        d_input: int,
        d_cond: int,
        ctx_dim: int | None = None,
        d_model: int = 512,
        out_node_nf: int = 256,
        out_edge_nf: int | None = None,
        num_layers: int = 8,
        nhead: int = 16,
        mlp_ratio: float = 4.0,
        in_edge_nf: int = 128,
        rbf_embedding_args: DictConfig | None = None,
        pair_bias_hidden_dim: int = 32,
        pair_bias_all_layers: bool | None = None,
        pair_bias_mode: str | None = None,
        ffn_hidden_dim: int | None = None,
        ffn_multiple_of: int = 64,
        qk_norm: bool = True,
        edge_pair_dim: int | None = None,
        coordinate_output_init_std: float = 0.0,
        center_coordinate_output: bool = True,
        num_atom_types: int | None = 10,
        use_cross_attn: bool = False,
        use_node_cond: bool = False,
        use_edge_cond: bool = False,
        cross_attn_mlp_ratio: float | None = None,
        lora: DictConfig | dict[str, Any] | None = None,
        use_graph_cond: bool = False,
        cross_attn_position: str = "residual",
    ):
        super().__init__()
        if use_cross_attn and ctx_dim is None:
            raise ValueError("ctx_dim is required when use_cross_attn=True")
        if cross_attn_position not in ("residual", "post_norm1"):
            raise ValueError(
                "cross_attn_position must be 'residual' or 'post_norm1', "
                f"got {cross_attn_position!r}"
            )
        self.ctx_dim = ctx_dim
        self.in_edge_nf = in_edge_nf
        self.d_model = d_model
        # Kept so wrappers building a per-head cross-attn bias know the head count.
        self.nhead = nhead
        self.num_layers = num_layers

        self.use_cross_attn = use_cross_attn
        self.cross_attn_position = cross_attn_position
        self.use_node_cond = use_node_cond
        self.use_edge_cond = use_edge_cond
        self.use_graph_cond = use_graph_cond

        self.dit = DiTBackbone(
            d_input=d_input,
            d_cond=d_cond,
            d_model=d_model,
            out_node_nf=out_node_nf,
            out_edge_nf=out_edge_nf,
            num_layers=num_layers,
            nhead=nhead,
            mlp_ratio=mlp_ratio,
            in_edge_nf=in_edge_nf,
            rbf_embedding_args=rbf_embedding_args,
            pair_bias_hidden_dim=pair_bias_hidden_dim,
            pair_bias_all_layers=pair_bias_all_layers,
            pair_bias_mode=pair_bias_mode,
            ffn_hidden_dim=ffn_hidden_dim,
            ffn_multiple_of=ffn_multiple_of,
            qk_norm=qk_norm,
            edge_pair_dim=edge_pair_dim,
            coordinate_output_init_std=coordinate_output_init_std,
            center_coordinate_output=center_coordinate_output,
            num_atom_types=num_atom_types,
            lora=lora,
        )

        # Cross-attention (sequence / cond_tokens) port. Built only when
        # enabled — the per-block blocks are the heaviest conditioning params,
        # and leaving them unused would trip ddp_find_unused_parameters_false.
        # Per-block cross-attention FFN dominates trainable-param count (~76% of
        # a block at mlp_ratio=4). Expose its width independently of the DiT
        # body's mlp_ratio (which must stay tied to the loaded pretrained
        # checkpoint) so adapter capacity can be tuned cheaply. ``None`` falls
        # back to ``mlp_ratio``.
        if use_cross_attn:
            x_attn_mlp_ratio = (
                mlp_ratio
                if cross_attn_mlp_ratio is None
                else float(cross_attn_mlp_ratio)
            )
            self.cross_attn_blocks = nn.ModuleList(
                [
                    GatedCrossAttention(
                        d_model, ctx_dim, nhead, mlp_ratio=x_attn_mlp_ratio
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.cross_attn_blocks = None

        # Embed-outside: conditioning arrives pre-projected, so the backbone
        # keeps only the zero-init fusion gate per port (built when enabled so
        # unused params never trip ddp_find_unused_parameters_false).
        if use_graph_cond:
            self.graph_gate = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("graph_gate", None)

        if use_node_cond:
            self.node_gate = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("node_gate", None)

        if use_edge_cond:
            self.edge_cond_gate = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("edge_cond_gate", None)

    @property
    def pos_out(self) -> nn.Module:
        """Expose the DiT coordinate head without registering a second copy."""
        return self.dit.pos_out

    def forward(
        self,
        a_embed: torch.Tensor,
        x: torch.Tensor,
        cond: torch.Tensor,
        edge_index: tuple[torch.Tensor, torch.Tensor],
        e_embed: torch.Tensor,
        batch: torch.Tensor,
        max_seqlen: int,
        cond_tokens: torch.Tensor | None = None,
        cond_token_mask: torch.Tensor | None = None,
        cond_token_attn_bias: torch.Tensor | None = None,
        node_cond: torch.Tensor | None = None,
        edge_cond: torch.Tensor | None = None,
        atom_types: torch.Tensor | None = None,
        node_counts: torch.Tensor | None = None,
        graph_cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as :meth:`DiTBackbone.forward` plus the cross-attention / graph / node /
        edge conditioning kwargs. All conditioning arrives pre-embedded
        at its port's fusion width (see the module docstring); each port is
        skipped when its kwarg is ``None``.

        Args:
            a_embed: (N, d_input) embedded atom features.
            x: (N, 3) normalized input positions.
            cond: (B, d_cond) intrinsic time/current-count features.
            edge_index: Two (E,) tensors of directed source/destination indices.
            e_embed: (E, in_edge_nf) embedded bond features.
            batch: (N,) graph indices for packed nodes.
            max_seqlen: Largest graph size, resolved before entering compilation.
            atom_types: Optional (N,) atom indices for pair-type attention bias.
            node_counts: Optional (B,) cached graph sizes.
            graph_cond: (B, d_model) per-graph conditioning added after the
                intrinsic condition projection through a zero-init gate.
                Requires ``use_graph_cond=True``.
            cond_tokens: (B, K, ctx_dim) external context tokens the atoms
                attend to. If ``None`` the gated cross-attention blocks are
                skipped (cheaper and equivalent to gate=0). The caller is
                responsible for their scale and the resulting adapter-update
                scale at the configured ``cross_attn_position``.
            cond_token_mask: (B, K) bool, True at valid tokens.
            cond_token_attn_bias: additive bias on cross-attention logits, either
                (B, K, N_padded) shared across heads or (B, nhead, K, N_padded)
                per head. ``N_padded`` is the max number of atoms across the
                current batch as produced by ``to_dense_batch``.
            node_cond: (N, d_model) per-atom conditioning aligned 1-to-1 with
                the denoised atoms (ragged batch order), added to ``h`` through a
                zero-init gate. Requires ``use_node_cond=True``.
            edge_cond: (E, in_edge_nf) per-edge conditioning features aligned
                1-to-1 with ``edge_index`` / ``e_embed``, added through a
                zero-init gate for edge decoding only, after pair-bias and
                transformer evaluation. Requires ``use_edge_cond=True``.
        See ``DiTBackbone.forward`` for the remaining arguments.
        """
        if node_counts is None:
            node_counts = torch.bincount(batch, minlength=cond.shape[0])
        h, cu_seqlens, c = self.dit.encode(a_embed, x, cond, batch, node_counts)
        if graph_cond is not None:
            if self.graph_gate is None:
                raise ValueError(
                    "graph_cond was provided but the backbone was constructed "
                    "with use_graph_cond=False. Set use_graph_cond=True to enable "
                    "per-graph conditioning."
                )
            if graph_cond.shape != c.shape:
                raise ValueError(
                    f"graph_cond shape {tuple(graph_cond.shape)} != condition "
                    f"shape {tuple(c.shape)} (B, d_model). graph_cond must arrive "
                    "pre-embedded and aligned 1-to-1 with the graphs."
                )
            # Keep the float32 gated residual under autocast: converting it to
            # BF16 can erase its small contribution while the gate is opening.
            c = c + torch.tanh(self.graph_gate) * graph_cond
        triu_edge_positions, full_to_triu = batched_triu_edge_layout(
            edge_index, batch, node_counts
        )

        if node_cond is not None:
            if self.node_gate is None:
                raise ValueError(
                    "node_cond was provided but the backbone was constructed "
                    "with use_node_cond=False. Set use_node_cond=True to enable "
                    "per-node conditioning."
                )
            if node_cond.shape != h.shape:
                raise ValueError(
                    f"node_cond shape {tuple(node_cond.shape)} != node feature "
                    f"shape {tuple(h.shape)} (N, d_model). node_cond must arrive "
                    "pre-embedded and aligned 1-to-1 with the atoms."
                )
            h = h + torch.tanh(self.node_gate) * node_cond.to(dtype=h.dtype)

        if cond_tokens is not None:
            if self.cross_attn_blocks is None:
                raise ValueError(
                    "cond_tokens was provided but the backbone was constructed "
                    "with use_cross_attn=False. Set use_cross_attn=True to enable "
                    "the cross-attention port."
                )
            if (
                cond_token_attn_bias is not None
                and cond_token_attn_bias.shape[-1] != max_seqlen
            ):
                raise ValueError(
                    "cond_token_attn_bias last dim "
                    f"{cond_token_attn_bias.shape[-1]} != padded num atoms "
                    f"{max_seqlen}. Bias must match the padded length produced "
                    "by to_dense_batch."
                )

        batch_size = cond.shape[0]
        do_cross_attn = self.cross_attn_blocks is not None and cond_tokens is not None
        # One cached flat<->dense scatter index for all attention layers (the
        # cross-attn query padding here + the self-attn padding inside each
        # DiTBlock). Reused instead of recomputing to_dense_batch per layer.
        dense_slot = (
            compute_dense_index(batch, batch_size, max_seqlen, node_counts)
            if _dit.USE_DENSE_INDEX
            else None
        )
        pair_bias, atom_mask, distance_embed = self.dit.pair_bias(
            x,
            edge_index,
            e_embed,
            batch,
            batch_size,
            max_seqlen,
            atom_types=atom_types,
            lengths=node_counts,
            dense_slot=dense_slot,
            triu_edge_positions=triu_edge_positions,
        )

        for i, block in enumerate(self.dit.blocks):
            normalized_attention_input = None
            if do_cross_attn:
                # Select the fusion units without changing the raw residual
                # skip in post_norm1 mode. Query/KV LayerNorm inside the adapter
                # does not set the scale of the update added at this seam.
                cross_attn_input = (
                    block.norm1(h) if self.cross_attn_position == "post_norm1" else h
                )
                if dense_slot is not None:
                    h_dense = pad_dense(
                        cross_attn_input, dense_slot, batch_size, max_seqlen
                    )
                else:
                    h_dense, _ = to_dense_batch(
                        cross_attn_input,
                        batch,
                        batch_size=batch_size,
                        max_num_nodes=max_seqlen,
                    )
                h_dense = self.cross_attn_blocks[i](
                    h_dense,
                    cond_tokens,
                    cond_token_mask,
                    cond_token_attn_bias,
                    query_mask=atom_mask,
                )
                conditioned_input = (
                    unpad_dense(h_dense, dense_slot)
                    if dense_slot is not None
                    else h_dense[atom_mask]
                )
                if self.cross_attn_position == "post_norm1":
                    normalized_attention_input = conditioned_input
                else:
                    h = conditioned_input

            h = block(
                h,
                c,
                batch,
                cu_seqlens,
                max_seqlen,
                pair_bias=self.dit.pair_bias_for_layer(pair_bias, i),
                atom_mask=atom_mask,
                dense_slot=dense_slot,
                normalized_attention_input=normalized_attention_input,
            )

        if edge_cond is not None:
            if self.edge_cond_gate is None:
                raise ValueError(
                    "edge_cond was provided but the backbone was constructed "
                    "with use_edge_cond=False. Set use_edge_cond=True to enable "
                    "per-edge conditioning."
                )
            if edge_cond.shape[0] != e_embed.shape[0]:
                raise ValueError(
                    f"edge_cond first dim {edge_cond.shape[0]} != number of "
                    f"edges {e_embed.shape[0]}; edge_cond must be aligned to "
                    "edge_index / e_embed."
                )
            if edge_cond.shape[-1] != self.in_edge_nf:
                raise ValueError(
                    f"edge_cond last dim {edge_cond.shape[-1]} != backbone "
                    f"in_edge_nf {self.in_edge_nf}. edge_cond must arrive "
                    "pre-embedded."
                )
            e_embed = e_embed + torch.tanh(self.edge_cond_gate) * edge_cond

        triu_edge_index = (
            edge_index[0][triu_edge_positions],
            edge_index[1][triu_edge_positions],
        )
        outputs = self.dit.decode(
            h,
            c,
            batch,
            x,
            triu_edge_index,
            e_embed[triu_edge_positions],
            distance_embed,
        )
        return *outputs, full_to_triu

    def load_dit_state_dict(
        self,
        dit_state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> Any:
        """Load a vanilla ``DiTBackbone`` state_dict into ``self.dit``.

        Cross-attention blocks are unaffected and keep their zero-gated init.
        """
        return self.dit.load_state_dict(dit_state_dict, strict=strict)


# ---------------------------------------------------------------------------
#  Top-level wrapper (extends DiTBackboneWithHeads)
# ---------------------------------------------------------------------------


class ConditionalDiTBackboneWithHeads(DiTBackboneWithHeads):
    """:class:`DiTBackboneWithHeads` with optional conditioning ports.

    Constructor is identical to the parent — set ``backbone_model_args._target_``
    to ``morph.model.backbones.conditional_dit.ConditionalDiTBackbone`` to swap
    the backbone in via Hydra. The forward simply forwards the new kwargs into
    ``self.backbone``.
    """

    def freeze_pretrained(self) -> int:
        """Freeze every submodule whose weights load from the unconditional
        pretrained checkpoint.

        On a freshly-loaded checkpoint this leaves trainable only the
        modules introduced by ``ConditionalDiTBackbone`` itself: the gated
        cross-attention blocks and, when present, the graph, node and edge
        conditioning streams. Subclasses can extend the freeze policy to their
        own auxiliary modules.
        """
        before = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Clean-lane modules loaded from the pretrained checkpoint:
        self.embedding_backbone.requires_grad_(False)
        self.backbone.dit.requires_grad_(False)
        # LoRA is part of the clean lane structurally, but it is the trainable
        # fine-tuning delta rather than a pretrained parameter.
        self.backbone.dit.set_lora_trainable()
        self.heads.requires_grad_(False)
        self.insertion_head.requires_grad_(False)
        self.ins_edge_head.requires_grad_(False)
        after = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return before - after

    def forward(
        self,
        mols_t: ChemBatch,
        t: torch.Tensor,
        cond_tokens: torch.Tensor | None = None,
        cond_token_mask: torch.Tensor | None = None,
        cond_token_attn_bias: torch.Tensor | None = None,
        node_cond: torch.Tensor | None = None,
        edge_cond: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        graph_cond: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Predict clean states and edits using optional external conditions."""
        x, a, _c, _e, edge_index, batch = mols_t.unpack()
        # Eager callers (sampling) leave max_seqlen None; the compiled training
        # path passes it precomputed to keep the graph host-read-free.
        if max_seqlen is None:
            max_seqlen = mols_t.max_seqlen

        a_embed, e_embed, cond = self.embedding_backbone(mols_t, t)

        edge_index_tuple = (edge_index[0], edge_index[1])

        h, x_out, e_out_triu, full_to_triu = self.backbone(
            a_embed,
            x,
            cond,
            edge_index_tuple,
            e_embed,
            batch,
            max_seqlen,
            cond_tokens=cond_tokens,
            cond_token_mask=cond_token_mask,
            cond_token_attn_bias=cond_token_attn_bias,
            node_cond=node_cond,
            edge_cond=edge_cond,
            graph_cond=graph_cond,
            atom_types=a,
            node_counts=mols_t.num_nodes_per_graph,
        )

        return self._assemble_outputs(
            h,
            x_out,
            e_out_triu,
            full_to_triu,
            x,
            batch,
            edge_index_tuple,
        )
