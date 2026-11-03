"""Lightweight context encoding for conditional fine-tuning.

The scaffold recipe conditions a frozen DiT on the scaffold atoms by encoding
them into cross-attention tokens with a per-atom featuriser:

    type embedding + charge embedding + 3D-position MLP  ->  concat  ->
    token MLP (-> ctx_dim)  ->  pack to dense ``(B, K, ctx_dim)``  ->
    optional self-attention so the tokens mix.

:class:`LightweightContextEncoder` factors that shared featurization out into
``atom_embed`` / ``charge_embed`` / ``pos_proj`` / ``token_proj`` / ``attn``.

This class directly encodes the scaffold subgraph for scaffold decoration.

The optional ``use_pair_bias`` flavour biases the self-attention with an
edge-based :class:`~morph.model.layers.attention.PairBias` (RBF distance +
edge-type of the context molecule's own bonds), so tokens mix with
geometry/topology awareness instead of plain token-space attention.

The per-node / per-edge scaffold-membership conditioning modules
(:class:`KeepNodeCond`, :class:`KeepEdgeCond`) also live here.

The output projection is intentionally *un-gated*: the clean lane's
cross-attention gates carry the zero-init identity guarantee, so a freshly
initialised encoder leaves an unconditional checkpoint unchanged at step 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from morph.model.layers.attention import BiasedSelfAttentionBlock, PairBias
from morph.model.layers.embedding import RBFEmbedding
from morph.utils.graph_math import batched_triu_edge_layout

if TYPE_CHECKING:
    from morph.dataset.core.atomic_data import ChemBatch


class LightweightContextEncoder(nn.Module):
    """Per-atom (type/charge/position) featuriser + optional self-attention.

    Carries no pretrained weights — all parameters train from scratch, so
    :meth:`freeze_pretrained_body` is a no-op (present so the wrappers'
    ``freeze_pretrained`` dispatch is uniform across encoder types).

    Args:
        num_atom_types: Atom-type vocabulary size.
        num_charges: Charge vocabulary size.
        atom_embed_dim / charge_embed_dim / pos_embed_dim: Per-feature widths.
        ctx_dim: Output token width. Must match ``ConditionalDiTBackbone.ctx_dim``.
        extra_feat_dim: Extra per-atom feature width concatenated *before* the
            token MLP (e.g. error embeddings). 0 for the plain encoders.
        num_attn_layers: Transformer self-attention layers over the tokens of
            each graph. 0 → mixing-free per-atom encoder (no ``attn`` submodule).
        nhead / mlp_ratio / dropout: Self-attention hyper-parameters.
        use_pair_bias: When True, the self-attention is biased by an
            edge-based :class:`~morph.model.layers.attention.PairBias` (RBF of
            pairwise distance + edge-type embedding of the context molecule's
            own bonds), so the tokens mix with geometry/topology awareness like
            the DiT backbone — instead of the plain token-space attention. The
            biased path reads ``mol.e`` / ``mol.edge_index`` in :meth:`encode`
            (the plain path needs only nodes). Requires ``num_attn_layers > 0``
            and a ``num_edge_types``.
        num_edge_types / edge_embed_dim / num_rbf / rbf_dmax /
            pair_bias_hidden_dim: pair-bias hyper-parameters (used only when
            ``use_pair_bias``).
    """

    def __init__(
        self,
        num_atom_types: int,
        num_charges: int,
        atom_embed_dim: int = 64,
        charge_embed_dim: int = 16,
        pos_embed_dim: int = 32,
        ctx_dim: int = 128,
        extra_feat_dim: int = 0,
        num_attn_layers: int = 0,
        nhead: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_pair_bias: bool = False,
        num_edge_types: int | None = None,
        edge_embed_dim: int = 32,
        num_rbf: int = 32,
        rbf_dmax: float = 5.0,
        pair_bias_hidden_dim: int = 32,
    ):
        super().__init__()
        self.ctx_dim = ctx_dim

        self.atom_embed = nn.Embedding(num_atom_types, atom_embed_dim)
        self.charge_embed = nn.Embedding(num_charges, charge_embed_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(3, pos_embed_dim),
            nn.GELU(),
            nn.Linear(pos_embed_dim, pos_embed_dim),
        )

        feat_dim = atom_embed_dim + charge_embed_dim + pos_embed_dim + extra_feat_dim
        self.token_proj = nn.Sequential(
            nn.Linear(feat_dim, ctx_dim),
            nn.GELU(),
            nn.Linear(ctx_dim, ctx_dim),
        )

        # Two self-attention flavours. ``use_pair_bias`` swaps the plain
        # token-space TransformerEncoder for a custom biased stack that respects
        # the context molecule's bonds/geometry. They are mutually exclusive;
        # only the active one holds parameters.
        self.attn: nn.TransformerEncoder | None = None
        self.biased_blocks: nn.ModuleList | None = None
        self.edge_embed: nn.Embedding | None = None
        self.pair_bias: PairBias | None = None

        if use_pair_bias:
            if num_attn_layers <= 0:
                raise ValueError("use_pair_bias requires num_attn_layers > 0")
            if num_edge_types is None:
                raise ValueError("use_pair_bias requires num_edge_types")
            self.edge_embed = nn.Embedding(num_edge_types, edge_embed_dim)
            self.pair_bias = PairBias(
                rbf_embedding=RBFEmbedding(
                    num_rbf=num_rbf,
                    rbf_dmax=rbf_dmax,
                    out_dim=num_rbf,
                    trainable=False,
                ),
                rbf_out_dim=num_rbf,
                edge_nf=edge_embed_dim,
                num_heads=nhead,
                hidden_dim=pair_bias_hidden_dim,
            )
            self.biased_blocks = nn.ModuleList(
                BiasedSelfAttentionBlock(ctx_dim, nhead, mlp_ratio, dropout)
                for _ in range(num_attn_layers)
            )
        elif num_attn_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=ctx_dim,
                nhead=nhead,
                dim_feedforward=int(ctx_dim * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.attn = nn.TransformerEncoder(
                layer, num_layers=num_attn_layers, norm=nn.LayerNorm(ctx_dim)
            )

    # ------------------------------------------------------------------ #
    # Shared building blocks used by subclass forwards.
    # ------------------------------------------------------------------ #
    def base_feats(
        self, a: torch.Tensor, c: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Concatenated per-atom ``[type | charge | position]`` features."""
        a_embed = self.atom_embed(a)
        c_embed = self.charge_embed(c)
        x_embed = self.pos_proj(x.to(dtype=a_embed.dtype))
        return torch.cat([a_embed, c_embed, x_embed], dim=-1)

    def apply_self_attn(
        self, cond_tokens: torch.Tensor, cond_token_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mix tokens within each graph; zero padded rows (no NaN leakage)."""
        if self.attn is None:
            return cond_tokens
        out = self.attn(cond_tokens, src_key_padding_mask=~cond_token_mask)
        return out.masked_fill(~cond_token_mask.unsqueeze(-1), 0.0)

    def encode(
        self, mol: ChemBatch, extra_feat: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full pipeline: featurise every atom → tokens → dense → self-attn.

        Args:
            mol: a :class:`ChemBatch` exposing ``.a``, ``.c``, ``.x``, ``.batch``
                (and, when ``use_pair_bias``, ``.e`` / ``.edge_index``).
            extra_feat: optional ``(N, extra_feat_dim)`` per-atom features to
                concatenate before the token MLP.

        Returns ``(cond_tokens, cond_token_mask)`` of shapes
        ``(B, K_max, ctx_dim)`` and ``(B, K_max)``.
        """
        batch = mol.batch
        batch_size = mol.num_graphs
        feats = self.base_feats(mol.a, mol.c, mol.x)
        if extra_feat is not None:
            feats = torch.cat([feats, extra_feat.to(dtype=feats.dtype)], dim=-1)
        tokens = self.token_proj(feats)
        cond_tokens, cond_token_mask = to_dense_batch(
            tokens, batch, batch_size=batch_size
        )
        if self.pair_bias is not None:
            cond_tokens = self._apply_biased_self_attn(
                cond_tokens, cond_token_mask, mol, batch, batch_size
            )
        else:
            cond_tokens = self.apply_self_attn(cond_tokens, cond_token_mask)
        return cond_tokens, cond_token_mask

    def _apply_biased_self_attn(
        self,
        cond_tokens: torch.Tensor,
        cond_token_mask: torch.Tensor,
        mol: ChemBatch,
        batch: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Self-attention biased by the context molecule's own edges/geometry."""
        K = cond_tokens.shape[1]
        edge_index = mol.edge_index
        e_embed = self.edge_embed(mol.e)
        triu_edge_positions, _ = batched_triu_edge_layout(
            edge_index, batch, mol.num_nodes_per_graph
        )
        # (B, H, K, K), already -inf at padded pairs. K matches the token pad
        # length because both dense-pack ``batch`` at the same max node count.
        bias, atom_mask, _ = self.pair_bias(
            mol.x,
            (edge_index[0], edge_index[1]),
            e_embed,
            batch,
            batch_size,
            K,
            lengths=mol.num_nodes_per_graph,
            triu_edge_positions=triu_edge_positions,
        )
        # A fully padded query row is all -inf → NaN softmax. Zero those rows'
        # bias (their outputs are masked to 0 below anyway).
        bias = bias.masked_fill((~atom_mask)[:, None, :, None], 0.0)
        for block in self.biased_blocks:
            cond_tokens = block(cond_tokens, bias)
        return cond_tokens.masked_fill(~cond_token_mask.unsqueeze(-1), 0.0)

    @torch.compiler.disable
    def forward(self, mol: ChemBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode every atom of ``mol`` into dense cross-attention tokens
        ``(cond_tokens (B,K,ctx_dim), cond_token_mask (B,K))``.

        The scaffold wrapper instantiates this class directly. Compiler-disabled:
        ``to_dense_batch`` /
        `PairBias` derive shapes dynamo can't trace. Subclasses that need extra
        per-atom inputs (error tokens) override ``forward`` with their own
        signature; the plain per-atom / per-structure encoders use this as-is.
        """
        return self.encode(mol)

    def freeze_pretrained_body(self) -> int:
        """No-op: this encoder has no pretrained-loaded body. Returns 0."""
        return 0


# Categorical edge types by scaffold membership of the two endpoints.
_EDGE_DEC_DEC = 0  # both endpoints in decoration
_EDGE_SC_DEC = 1  # exactly one endpoint in the scaffold
_EDGE_SC_SC = 2  # both endpoints in the scaffold
_N_EDGE_SCAFFOLD_TYPES = 3


class KeepEdgeCond(nn.Module):
    """Per-edge scaffold-class → pre-embedded ``edge_cond``.

    Classifies each edge as decoration-decoration / scaffold-decoration /
    scaffold-scaffold and looks it up straight at ``in_edge_nf`` (embed-outside:
    the backbone adds it through a zero-init ``edge_cond_gate``). Recomputed each
    forward from the live ``edge_index`` + ``keep_mask`` so it tracks the
    integrator's deletions/insertions. Returns ``None`` when the batch carries no
    ``keep_mask``.
    """

    def __init__(self, in_edge_nf: int):
        super().__init__()
        self.embed = nn.Embedding(_N_EDGE_SCAFFOLD_TYPES, in_edge_nf)

    def forward(self, mols_t: ChemBatch) -> torch.Tensor | None:
        """Embed retained-atom edge classes; return None without a keep mask."""
        sc_mask = getattr(mols_t, "keep_mask", None)
        if sc_mask is None:
            return None
        sc_mask = sc_mask.bool()
        edge_index = mols_t.edge_index
        edge_types = (
            sc_mask[edge_index[0]].long() + sc_mask[edge_index[1]].long()
        ).clamp(max=_EDGE_SC_SC)
        out = self.embed(edge_types)
        # Type 0 (editable-editable) carries NO signal — only kept-touching edges
        # condition. An all-zero keep_mask therefore yields
        # zero edge_cond instead of a constant leak through the shared gate.
        return out * (edge_types > 0).unsqueeze(-1).to(out.dtype)


class KeepNodeCond(nn.Module):
    """Per-node scaffold membership (0 = decoration, 1 = keep) → ``node_cond``.

    Recomputed each forward from the live ``keep_mask``. Returns ``None`` when
    the batch carries no ``keep_mask``.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(2, d_model)

    def forward(self, mols_t: ChemBatch) -> torch.Tensor | None:
        """Embed kept atoms while leaving editable atoms unconditioned."""
        sc_mask = getattr(mols_t, "keep_mask", None)
        if sc_mask is None:
            return None
        keep = sc_mask.long()
        out = self.embed(keep)
        # Only KEPT atoms (keep_mask=1) condition; editable atoms (0) carry no
        # signal, so an all-zero keep_mask yields zero
        # node_cond instead of a constant leak through the shared node_gate.
        return out * keep.unsqueeze(-1).to(out.dtype)
