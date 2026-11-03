# Private compiler patches are imported only when compilation is requested.
# ruff: noqa: PLC0415

"""Attention for the DiT / context backbones — one home for the attention path.

Centralises packed variable-length self-attention, its packed<->dense plumbing,
the dense context self-attention block, and the additive attention biases:

* :class:`BiasedSelfAttentionBlock` — dense pre-norm self-attention + FFN.
* :class:`PairBias`         — per-head self-attention bias (RBF distance + edge type
                              + optional atom-pair type),
                              used by the DiT backbone / context encoders.
* :class:`DistanceAttnBias` — fixed ``-softplus(gamma)·d²`` cross-attention bias (shared
                              across heads).
* :func:`build_distance_attn_bias` — dense-batch masking/plumbing for that bias.

Backbone self-attention runs in packed ``(total_N, ...)`` layout. Unbiased blocks use
FlashAttention's true variable-length kernel when the optional dependency is
installed and explicitly enabled. Packed biased attention defaults to PyTorch SDPA;
``MORPH_ATTENTION_BACKEND=flex`` opts into Triton FlexAttention on CUDA.
The dense context block uses PyTorch SDPA directly.
"""

from __future__ import annotations

import os
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from morph.model.layers.kernels import pack_normalized_qkv

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func  # type: ignore

    _HAS_FLASH_ATTN = True
except ImportError:  # pragma: no cover
    flash_attn_varlen_qkvpacked_func = None
    _HAS_FLASH_ATTN = False

USE_FLASH_ATTN = os.environ.get("MORPH_FLASH_ATTN", "0") == "1"
BIASED_ATTENTION_BACKEND = os.environ.get("MORPH_ATTENTION_BACKEND", "sdpa")
if BIASED_ATTENTION_BACKEND not in {"sdpa", "flex"}:
    raise ValueError("MORPH_ATTENTION_BACKEND must be 'sdpa' or 'flex'")
_FLEX_ATTENTION = None
if BIASED_ATTENTION_BACKEND == "flex":
    from torch.nn.attention.flex_attention import flex_attention

    _FLEX_ATTENTION = torch.compile(flex_attention, dynamic=True)


@cache
def fix_efficient_attention_bias_metadata():
    """Keep symbolic bias-gradient alignment correct across multiples of 16.

    PyTorch 2.9's efficient-attention backward meta kernel chooses a Python
    branch for aligned lengths. A dynamic compiled graph can retain the wrong
    logical size or stride when lengths cross 32, 48, etc. Specify the logical
    shape separately from storage strides computed with ceiling division. The
    CUDA kernel and numerical attention operations remain PyTorch's native ones.
    """
    if not torch.__version__.startswith("2.9."):
        return
    from torch._meta_registrations import meta__scaled_dot_product_efficient_backward

    def backward_meta(
        grad_out,
        query,
        key,
        value,
        attn_bias,
        out,
        logsumexp,
        philox_seed,
        philox_offset,
        dropout_p,
        grad_input_mask,
        is_causal=False,
        scale=None,
    ):
        grad_q, grad_k, grad_v, _ = meta__scaled_dot_product_efficient_backward(
            grad_out,
            query,
            key,
            value,
            attn_bias,
            out,
            logsumexp,
            philox_seed,
            philox_offset,
            dropout_p,
            [*grad_input_mask[:3], False],
            is_causal,
            scale,
        )
        grad_bias = None
        if attn_bias is not None and grad_input_mask[3]:
            length = attn_bias.shape[-1]
            aligned_length = ((length + 15) // 16) * 16
            # A padded allocation followed by [:length] can itself specialize
            # the logical size to the aligned extent when first traced at an
            # aligned length. Specify logical sizes and storage strides directly.
            padded_sizes = (*attn_bias.shape[:-1], aligned_length)
            strides = []
            stride = 1
            for size in reversed(padded_sizes):
                strides.append(stride)
                stride *= torch.sym_max(size, 1)
            grad_bias = torch.empty_strided(
                attn_bias.shape,
                tuple(reversed(strides)),
                dtype=attn_bias.dtype,
                device=attn_bias.device,
            )
        return grad_q, grad_k, grad_v, grad_bias

    torch.library.register_fake(
        torch.ops.aten._scaled_dot_product_efficient_attention_backward.default,
        backward_meta,
        allow_override=True,
    )


@cache
def _flash_attention_device_supported(device: torch.device) -> bool:
    """Whether flash-attn supports the CUDA architecture behind ``device``."""
    if device.type != "cuda":
        return False
    # flash-attn v2 requires Ampere (compute capability 8.x) or newer.  Merely
    # importing the extension is insufficient on mixed-generation clusters.
    return torch.cuda.get_device_capability(device)[0] >= 8


# ---------------------------------------------------------------------------
#  Packed <-> dense index helpers (shared with conditional_dit via re-export)
# ---------------------------------------------------------------------------
def compute_local_index(
    batch_idx: torch.Tensor,
    batch_size: int,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Intra-graph position of each packed node (``batch_idx`` graph-contiguous)."""
    if lengths is None:
        lengths = torch.bincount(batch_idx, minlength=batch_size)
    offsets = torch.cumsum(lengths, 0) - lengths
    return (
        torch.arange(batch_idx.shape[0], device=batch_idx.device) - offsets[batch_idx]
    )


def compute_dense_index(
    batch_idx: torch.Tensor,
    batch_size: int,
    max_seqlen: int,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flat-node -> dense-slot map ``b * max_seqlen + intra_graph_pos``."""
    return batch_idx * max_seqlen + compute_local_index(batch_idx, batch_size, lengths)


def pad_dense(x: torch.Tensor, dense_slot, batch_size, max_seqlen) -> torch.Tensor:
    """Scatter packed values into ``(batch_size, max_seqlen, *rest)``."""
    flat = x.new_zeros((batch_size * max_seqlen, *x.shape[1:]))
    flat[dense_slot] = x
    return flat.view(batch_size, max_seqlen, *x.shape[1:])


def unpad_dense(x_dense: torch.Tensor, dense_slot) -> torch.Tensor:
    """Gather dense ``(B, max_seqlen, *rest)`` back to packed ``(total_N, *rest)``."""
    return x_dense.reshape(-1, *x_dense.shape[2:])[dense_slot]


class _UniqueGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, slots):
        ctx.save_for_backward(slots)
        ctx.input_shape = values.shape
        return values[slots]

    @staticmethod
    def backward(ctx, grad):
        (slots,) = ctx.saved_tensors
        result = grad.new_zeros(ctx.input_shape)
        result[slots] = grad
        return result, None


def unpad_dense_unique(values, slots):
    """Gather unique dense slots with a direct-copy backward (profiling opt-in).

    Slots must come from compute_dense_index: each real atom owns one slot.
    Repeated slots require the general unpad_dense implementation instead.
    """
    flat = values.reshape(-1, *values.shape[2:])
    return _UniqueGather.apply(flat, slots)


# ---------------------------------------------------------------------------
#  Packed self-attention
# ---------------------------------------------------------------------------
def flash_attention_supported(reference: torch.Tensor) -> bool:
    """Whether the optional FlashAttention varlen kernel can run on ``reference``."""
    return (
        _HAS_FLASH_ATTN
        and reference.is_cuda
        and reference.dtype in (torch.float16, torch.bfloat16)
        and _flash_attention_device_supported(reference.device)
    )


def packed_self_attention(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    batch_idx: torch.Tensor,
    *,
    pair_bias: torch.Tensor | None = None,
    atom_mask: torch.Tensor | None = None,
    dense_slot: torch.Tensor | None = None,
    q_norm: nn.Module | None = None,
    k_norm: nn.Module | None = None,
) -> torch.Tensor:
    """Variable-length self-attention on packed sequences.

    Args:
        qkv: (total_N, 3, H, D) packed q/k/v.
        cu_seqlens: (B+1,) int32 cumulative sequence lengths.
        max_seqlen: largest sequence length in the batch (Python int).
        batch_idx: (total_N,) batch assignment, used by the SDPA path.
        pair_bias: optional dense ``(B, H, N, N)`` additive attention bias.
        atom_mask: optional cached ``(B, N)`` valid-atom mask.
        dense_slot: cached ``compute_dense_index`` scatter map (avoids repadding).
        q_norm: Optional query normalization, fused with CUDA inference packing.
        k_norm: Optional key normalization, fused with CUDA inference packing.

    Returns:
        (total_N, H, D)
    """
    use_flash = pair_bias is None and USE_FLASH_ATTN and flash_attention_supported(qkv)
    qkv_padded = None
    if dense_slot is not None and atom_mask is not None and not use_flash:
        qkv_padded = pack_normalized_qkv(qkv, cu_seqlens, max_seqlen, q_norm, k_norm)
    if qkv_padded is None and q_norm is not None and k_norm is not None:
        q, k, v = qkv.unbind(dim=1)
        qkv = torch.stack((q_norm(q).to(q.dtype), k_norm(k).to(k.dtype), v), dim=1)

    if use_flash:
        return flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen, dropout_p=0.0, causal=False
        )

    # Dense SDPA handles the biased block and is the portable fallback.
    batch_size = cu_seqlens.numel() - 1
    if dense_slot is not None:
        if qkv_padded is None:
            qkv_padded = pad_dense(qkv, dense_slot, batch_size, max_seqlen)
        if atom_mask is None:
            flat_mask = torch.zeros(
                batch_size * max_seqlen, dtype=torch.bool, device=qkv.device
            )
            flat_mask[dense_slot] = True
            atom_mask = flat_mask.view(batch_size, max_seqlen)

        def _unpad(out):
            return unpad_dense(out, dense_slot)
    else:
        qkv_padded, inferred_mask = to_dense_batch(
            qkv, batch_idx, max_num_nodes=max_seqlen
        )
        if atom_mask is None:
            atom_mask = inferred_mask

        def _unpad(out):
            return out[inferred_mask]

    q, k, v = qkv_padded.unbind(dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    assert atom_mask is not None
    attn_mask = pair_bias if pair_bias is not None else atom_mask[:, None, None, :]
    if pair_bias is not None and BIASED_ATTENTION_BACKEND == "flex" and q.is_cuda:
        # Torch 2.9 accumulates score-mod buffer gradients in FP32. Keep this
        # differentiable cast so PairBias receives a gradient of its own dtype.
        bias = attn_mask.float()

        def score_mod(score, b, h, qi, ki):
            return score + bias[b, h, qi, ki]

        out = _FLEX_ATTENTION(q, k, v, score_mod=score_mod)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out = out.transpose(1, 2)
    return _unpad(out)


# ---------------------------------------------------------------------------
#  Dense context self-attention
# ---------------------------------------------------------------------------
class BiasedSelfAttentionBlock(nn.Module):
    """Pre-norm self-attention + FFN over context tokens with an additive
    per-head pair bias ``(B, H, K, K)`` (``-inf`` at masked pairs). Mirrors the
    DiT block's biased SDPA attention but without adaLN conditioning — the
    context encoder has no time signal.
    """

    def __init__(self, dim: int, nhead: int, mlp_ratio: float, dropout: float):
        super().__init__()
        assert dim % nhead == 0, "ctx_dim must be divisible by nhead"
        self.nhead = nhead
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        """Mix dense tokens using the supplied per-head attention bias."""
        B, K, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, K, 3, self.nhead, D // self.nhead)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, K, d)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        o = o.transpose(1, 2).reshape(B, K, D)
        x = x + self.drop(self.proj(o))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
#  Self-attention pair bias (distance + edge type + optional atom-pair type)
# ---------------------------------------------------------------------------
class PairBias(nn.Module):
    """Per-head additive attention bias from distance, edge, and atom-pair type.

    Logically the bias is::

        bias[b, h, i, j] = down_proj( GELU(
            distance_proj(rbf(||x_i - x_j||))
            + edge_proj(edge_features[i, j])
            + edge_present[i, j] * atom_pair_embedding({atom_type_i, atom_type_j})
        ) )[h]

    where ``edge_features[i, j] = e_embed`` at edges in ``edge_index`` and
    zero elsewhere. Distance and hidden features are computed only for real
    atom pairs. The small final per-head bias is scattered into the one dense
    tensor needed by the biased SDPA block.

    The optional atom-pair table uses an unordered triangular index, so its
    contribution is exactly symmetric under swapping atoms ``i`` and ``j``.
    It is zero-initialized so enabling it preserves the pre-existing PairBias
    at initialization while still receiving gradients through ``down_proj``.
    Blocks gate the full attention residual via adaLN-Zero.
    Padded pairs are ``-inf``-masked. ``down_proj`` is left at its Xavier init
    (inherited from the parent backbone's ``_basic_init``).
    """

    def __init__(
        self,
        rbf_embedding: nn.Module,
        rbf_out_dim: int,
        edge_nf: int,
        num_heads: int,
        hidden_dim: int,
        num_atom_types: int | None = 10,
    ):
        super().__init__()
        if num_atom_types is not None and num_atom_types <= 0:
            raise ValueError("num_atom_types must be positive when provided")
        self.num_heads = num_heads
        self.num_atom_types = num_atom_types
        self.rbf_embedding = rbf_embedding
        self.dist_to_hidden = nn.Linear(rbf_out_dim, hidden_dim, bias=True)
        self.edge_to_hidden = nn.Linear(edge_nf, hidden_dim, bias=False)
        self.atom_pair_embedding: nn.Embedding | None = None
        if num_atom_types is not None:
            num_unordered_pairs = num_atom_types * (num_atom_types + 1) // 2
            self.atom_pair_embedding = nn.Embedding(num_unordered_pairs, hidden_dim)
            nn.init.zeros_(self.atom_pair_embedding.weight)
        self.act = nn.GELU(approximate="tanh")
        self.down_proj = nn.Linear(hidden_dim, num_heads, bias=False)

    def _project_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Read out one shared geometry bias per attention head."""
        return self.down_proj(self.act(hidden))

    def _atom_pair_indices(
        self, atom_types_i: torch.Tensor, atom_types_j: torch.Tensor
    ) -> torch.Tensor:
        """Return unordered triangular indices for paired atom types."""
        assert self.num_atom_types is not None
        lower = torch.minimum(atom_types_i, atom_types_j)
        upper = torch.maximum(atom_types_i, atom_types_j)
        return lower * self.num_atom_types - lower * (lower - 1) // 2 + upper - lower

    def _scatter_bias(
        self,
        edge_bias,
        diagonal_bias,
        batch,
        local,
        rows,
        cols,
        batch_size,
        max_seqlen,
        symmetric,
    ):
        bias = edge_bias.new_full(
            (batch_size, self.num_heads, max_seqlen, max_seqlen), -torch.inf
        )
        bias[batch, :, local, local] = diagonal_bias
        bias[batch[rows], :, local[rows], local[cols]] = edge_bias
        if symmetric:
            bias[batch[cols], :, local[cols], local[rows]] = edge_bias
        return bias

    def _edge_hidden(
        self,
        e_embed: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        atom_types: torch.Tensor | None,
    ) -> torch.Tensor:
        """Project edge features with the optional unordered atom-pair term."""
        edge_hidden = self.edge_to_hidden(e_embed)
        if self.atom_pair_embedding is not None:
            if atom_types is None:
                raise ValueError(
                    "atom_types is required when atom-pair PairBias is enabled"
                )
            pair_indices = self._atom_pair_indices(atom_types[rows], atom_types[cols])
            edge_hidden = edge_hidden + self.atom_pair_embedding(pair_indices)
        return edge_hidden

    def forward(
        self,
        x: torch.Tensor,
        edge_index: tuple[torch.Tensor, torch.Tensor],
        e_embed: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
        max_seqlen: int,
        atom_types: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        dense_slot: torch.Tensor | None = None,
        triu_edge_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            bias: (B, num_heads, max_seqlen, max_seqlen) additive attention
                bias, already ``-inf`` at padded pairs so softmax masks them.
            atom_mask: (B, max_seqlen) True at valid atoms (cached for the
                attention path so it doesn't repad).
            edge_distance_embed: RBF embeddings aligned with ``edge_index`` or,
                when ``triu_edge_positions`` is supplied, those representative
                undirected edges, for reuse by the edge decoder.
        """
        if lengths is None:
            lengths = torch.bincount(batch, minlength=batch_size)
        if dense_slot is None:
            dense_slot = compute_dense_index(batch, batch_size, max_seqlen, lengths)
        local = dense_slot - batch * max_seqlen

        # Runtime molecular graphs are complete and directed without self-edges,
        # so edge_index already enumerates every off-diagonal atom pair. Use it
        # directly instead of rebuilding sum(n_i**2) pair indices and mapping the
        # edges back into that layout. Only the shared zero-distance diagonal is
        # missing.
        full_rows, full_cols = edge_index
        if triu_edge_positions is None:
            rows, cols = full_rows, full_cols
            selected_e_embed = e_embed
        else:
            rows = full_rows[triu_edge_positions]
            cols = full_cols[triu_edge_positions]
            selected_e_embed = e_embed[triu_edge_positions]
        edge_distances = (x[rows] - x[cols]).norm(dim=-1)
        distance_embed = self.rbf_embedding(torch.cat((edge_distances, x.new_zeros(1))))
        edge_distance_embed = distance_embed[:-1]
        diagonal_distance_embed = distance_embed[-1:]

        edge_hidden = self.dist_to_hidden(edge_distance_embed)
        edge_hidden = edge_hidden + self._edge_hidden(
            selected_e_embed, rows, cols, atom_types
        ).to(dtype=edge_hidden.dtype)
        edge_bias = self._project_hidden(edge_hidden)
        diagonal_bias = self._project_hidden(
            self.dist_to_hidden(diagonal_distance_embed)
        )

        bias = self._scatter_bias(
            edge_bias,
            diagonal_bias,
            batch,
            local,
            rows,
            cols,
            batch_size,
            max_seqlen,
            triu_edge_positions is not None,
        )
        atom_mask = torch.arange(max_seqlen, device=x.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        return (
            bias,
            atom_mask,
            edge_distance_embed,
        )


# ---------------------------------------------------------------------------
#  Cross-attention distance biases
# ---------------------------------------------------------------------------
class DistanceAttnBias(nn.Module):
    """Additive cross-attention bias from query↔key 3D distances.

    Gives the gated cross-attention a geometric **correspondence prior**: a query
    atom attends most strongly to the context atom that sits at the *same
    location*, falling off with distance. Returns a ``(B, K, N)`` bias matching
    ``cond_token_attn_bias`` (``bias[b, k, n]`` is added to the logit of query
    atom ``n`` attending to context token ``k``). Monotone in distance (max at
    d=0), i.e. "attend to the co-located atom".

    Well-posed only when queries and keys share a 3D frame. The overall
    cross-attn contribution is still governed by the per-block ``attn_gate``
    (~0 at init), so this only shapes WHERE attention lands.

    The single ``(B, K, N)`` map is broadcast over every head, so all heads share
    the same learned length scale.
    """

    def __init__(self, init_gamma: float = 1.0):
        super().__init__()
        # softplus(raw_gamma) is the inverse length-scale² of the falloff.
        g = max(float(init_gamma), 1e-4)
        raw = torch.log(torch.expm1(torch.tensor(g))) if g < 20 else torch.tensor(g)
        self.raw_gamma = nn.Parameter(raw)

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor) -> torch.Tensor:
        """Return negative squared distances scaled by a positive coefficient."""
        # x_q: (B, N, 3) query positions; x_k: (B, K, 3) key positions.
        d2 = torch.cdist(x_k.float(), x_q.float()).pow(2)  # (B, K, N)
        return -F.softplus(self.raw_gamma) * d2


def build_distance_attn_bias(
    dist_bias: DistanceAttnBias,
    x_q: torch.Tensor,
    batch_q: torch.Tensor,
    n_pad_q: int,
    x_k: torch.Tensor,
    batch_k: torch.Tensor,
    k_pad: int,
    batch_size: int,
) -> torch.Tensor:
    """Dense ``(B, k_pad, n_pad_q)`` geometric cross-attn bias (query atom ↔
    co-located context token). Shared plumbing for conditional
    wrappers, which differ only in which batch supplies the key tokens.
    """
    xq, _ = to_dense_batch(x_q, batch_q, batch_size=batch_size, max_num_nodes=n_pad_q)
    xk, _ = to_dense_batch(x_k, batch_k, batch_size=batch_size, max_num_nodes=k_pad)
    return dist_bias(xq, xk)
