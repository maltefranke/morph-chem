# Triton imports are deferred so CPU-only installations can import the model.
# ruff: noqa: PLC0415

"""Fused CUDA operations shared by the model layers.

Keep Triton definitions, registration and eligibility checks here; ordinary
attention and backbone code use these helpers with native PyTorch fallbacks.
Triton is imported only when the corresponding CUDA implementation is loaded.
"""

from __future__ import annotations

import os
from functools import cache

import torch
import torch.nn as nn


@cache
def _load_fused_qkv_packing():
    """Normalize Q/K and write the padded attention layout in one CUDA launch."""
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton

    @triton.jit
    def pack_kernel(
        QKV,
        QW,
        KW,
        OFFSETS,
        OUT,
        N,
        HEADS: tl.constexpr,
        D: tl.constexpr,
        EPS: tl.constexpr,
        S0: tl.constexpr,
        S1: tl.constexpr,
        S2: tl.constexpr,
        S3: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        graph = tl.program_id(0)
        row = tl.program_id(1) * 8 + tl.arange(0, 8)
        head = tl.program_id(2)
        start = tl.load(OFFSETS + graph)
        end = tl.load(OFFSETS + graph + 1)
        d = tl.arange(0, BLOCK)
        valid = (start + row[:, None] < end) & (d[None, :] < D)
        offset = (start + row[:, None]) * S0 + head * S2 + d[None, :] * S3
        q = tl.load(QKV + offset, valid, other=0).to(tl.float32)
        k = tl.load(QKV + offset + S1, valid, other=0).to(tl.float32)
        v = tl.load(QKV + offset + 2 * S1, valid, other=0)
        q = q * tl.rsqrt(tl.sum(q * q, 1) / D + EPS)[:, None]
        k = k * tl.rsqrt(tl.sum(k * k, 1) / D + EPS)[:, None]
        q = q * tl.load(QW + d, d < D, other=0).to(tl.float32)[None, :]
        k = k * tl.load(KW + d, d < D, other=0).to(tl.float32)[None, :]
        out = ((graph * N + row[:, None]) * 3 * HEADS + head) * D + d[None, :]
        store = (row[:, None] < N) & (d[None, :] < D)
        tl.store(OUT + out, q, store)
        tl.store(OUT + out + HEADS * D, k, store)
        tl.store(OUT + out + 2 * HEADS * D, v, store)

    @triton_op("morph::pack_normalized_qkv", mutates_args={})
    def pack(
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        offsets: torch.Tensor,
        max_seqlen: int,
        eps: float,
    ) -> torch.Tensor:
        batch_size = offsets.numel() - 1
        heads, dim = qkv.shape[2:]
        out = qkv.new_empty((batch_size, max_seqlen, 3, heads, dim))
        wrap_triton(pack_kernel)[(batch_size, triton.cdiv(max_seqlen, 8), heads)](
            qkv,
            q_weight,
            k_weight,
            offsets,
            out,
            max_seqlen,
            heads,
            dim,
            eps,
            *qkv.stride(),
            triton.next_power_of_2(dim),
            num_warps=4,
            enable_fp_fusion=False,
        )
        return out

    return pack


_pack_normalized_qkv = _load_fused_qkv_packing() if torch.cuda.is_available() else None


def pack_normalized_qkv(qkv, offsets, max_seqlen, q_norm, k_norm):
    """Return fused padded QKV, or ``None`` when native packing is required.

    The caller must supply graph-contiguous packed QKV and the corresponding
    cumulative lengths. CPU, autograd, and unsupported normalization layouts
    use the ordinary PyTorch path in attention.py.
    """
    supported = (
        _pack_normalized_qkv is not None
        and qkv.is_cuda
        and not torch.is_grad_enabled()
        and qkv.dtype in (torch.float32, torch.bfloat16)
        and isinstance(q_norm, nn.RMSNorm)
        and isinstance(k_norm, nn.RMSNorm)
        and q_norm.weight is not None
        and k_norm.weight is not None
        and q_norm.weight.dtype == torch.float32
        and k_norm.weight.dtype == torch.float32
        and q_norm.normalized_shape == k_norm.normalized_shape == (qkv.shape[-1],)
        and q_norm.weight.is_contiguous()
        and k_norm.weight.is_contiguous()
        and q_norm.eps is not None
        and q_norm.eps == k_norm.eps
    )
    if not supported:
        return None
    return _pack_normalized_qkv(
        qkv, q_norm.weight, k_norm.weight, offsets, max_seqlen, q_norm.eps
    )


def graph_broadcast(values, batch_idx, offsets=None):
    return values[batch_idx]


@cache
def _load_fused_graph_broadcast():
    """Load the first-order CUDA graph-condition reduction once.

    Keep Triton optional for native indexing and CPU imports. The reduction
    uses FP32 accumulation with a different order from native indexing.
    """
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton

    @triton.jit
    def _sum_graph_rows(
        grad,
        offsets,
        output,
        width: tl.constexpr,
        row_stride: tl.constexpr,
        col_stride: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        graph = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        start = tl.load(offsets + graph)
        end = tl.load(offsets + graph + 1)
        rows = start + tl.arange(0, BLOCK_N)
        accum = tl.full((BLOCK_N, BLOCK_D), 0, tl.float32)
        # Fixed-size tiles cover arbitrary graph lengths without atom bucketing or
        # specializing the compiled model for each maximum atom count.
        for tile_start in range(start, end, BLOCK_N):
            rows = tile_start + tl.arange(0, BLOCK_N)
            values = tl.load(
                grad + rows[:, None] * row_stride + cols[None, :] * col_stride,
                (rows[:, None] < end) & (cols[None, :] < width),
                other=0,
            )
            accum += values.to(tl.float32)
        total = tl.sum(accum, axis=0)
        tl.store(output + graph * width + cols, total, cols < width)

    @triton_op("morph::sum_graph_rows", mutates_args={})
    def sum_graph_rows(grad: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            (offsets.numel() - 1, grad.shape[1]), device=grad.device, dtype=grad.dtype
        )
        wrap_triton(_sum_graph_rows)[(out.shape[0], triton.cdiv(grad.shape[1], 64))](
            grad,
            offsets,
            out,
            grad.shape[1],
            grad.stride(0),
            grad.stride(1),
            32,
            64,
        )
        return out

    class TritonBroadcast(torch.autograd.Function):
        @staticmethod
        def forward(ctx, values, batch_idx, offsets):
            ctx.save_for_backward(offsets)
            return values[batch_idx]

        @staticmethod
        def backward(ctx, grad):
            (offsets,) = ctx.saved_tensors
            return sum_graph_rows(grad, offsets), None, None

    def graph_broadcast(values, batch_idx, offsets=None):
        if (
            offsets is None
            or not values.is_cuda
            or not torch.is_grad_enabled()
            or values.dtype not in (torch.float32, torch.bfloat16)
        ):
            return values[batch_idx]
        return TritonBroadcast.apply(values, batch_idx, offsets)

    return graph_broadcast


if os.environ.get("MORPH_FUSED_BROADCAST", "1") == "1" and torch.cuda.is_available():
    graph_broadcast = _load_fused_graph_broadcast()


@cache
def _load_edge_normalization():
    """Build symmetric pair features and normalize without materializing a cat."""
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton

    @triton.jit
    def kernel(
        P,
        SRC,
        DST,
        EDGE,
        RBF,
        W,
        B,
        OUT,
        E,
        PS0: tl.constexpr,
        PS1: tl.constexpr,
        SS: tl.constexpr,
        DS: tl.constexpr,
        ES0: tl.constexpr,
        ES1: tl.constexpr,
        RS0: tl.constexpr,
        RS1: tl.constexpr,
        PD: tl.constexpr,
        ED: tl.constexpr,
        RD: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0) * 4 + tl.arange(0, 4)
        col = tl.arange(0, BLOCK)
        src = tl.load(SRC + row * SS, row < E, other=0)
        dst = tl.load(DST + row * DS, row < E, other=0)
        pair_col = col % PD
        pair_mask = (row[:, None] < E) & (col[None, :] < 2 * PD)
        a = tl.load(
            P + src[:, None] * PS0 + pair_col[None, :] * PS1, pair_mask, other=0
        )
        b = tl.load(
            P + dst[:, None] * PS0 + pair_col[None, :] * PS1, pair_mask, other=0
        )
        pair = tl.where(
            col[None, :] < PD,
            a.to(tl.float32) + b.to(tl.float32),
            tl.abs(a.to(tl.float32) - b.to(tl.float32)),
        )
        # Match the native pair arithmetic's intermediate BF16 rounding.
        pair = (
            (pair.to(a.dtype).to(tl.float32) * 0.7071067811865476)
            .to(a.dtype)
            .to(tl.float32)
        )
        edge = tl.load(
            EDGE + row[:, None] * ES0 + (col[None, :] - 2 * PD) * ES1,
            (row[:, None] < E)
            & (col[None, :] >= 2 * PD)
            & (col[None, :] < 2 * PD + ED),
            other=0,
        ).to(tl.float32)
        rbf = tl.load(
            RBF + row[:, None] * RS0 + (col[None, :] - 2 * PD - ED) * RS1,
            (row[:, None] < E)
            & (col[None, :] >= 2 * PD + ED)
            & (col[None, :] < 2 * PD + ED + RD),
            other=0,
        ).to(tl.float32)
        width = 2 * PD + ED + RD
        value = tl.where(
            col[None, :] < 2 * PD, pair, tl.where(col[None, :] < 2 * PD + ED, edge, rbf)
        )
        value = tl.where(col[None, :] < width, value, 0.0)
        mean = tl.sum(value, 1) / width
        centered = tl.where(col[None, :] < width, value - mean[:, None], 0.0)
        var = tl.sum(centered * centered, 1) / width
        norm = centered * tl.rsqrt(var + EPS)[:, None]
        weight = tl.load(W + col, col < width, other=0).to(tl.float32)
        bias = tl.load(B + col, col < width, other=0).to(tl.float32)
        tl.store(
            OUT + row[:, None] * width + col[None, :],
            norm * weight + bias,
            (row[:, None] < E) & (col[None, :] < width),
        )

    @triton_op("morph::edge_normalization", mutates_args={})
    def normalize(
        p: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge: torch.Tensor,
        rbf: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        width = 2 * p.shape[1] + edge.shape[1] + rbf.shape[1]
        # CUDA autocast runs LayerNorm in FP32; the following linear casts it.
        out = torch.empty((src.numel(), width), dtype=torch.float32, device=p.device)
        wrap_triton(kernel)[(triton.cdiv(src.numel(), 4),)](
            p,
            src,
            dst,
            edge,
            rbf,
            w,
            b,
            out,
            src.numel(),
            *p.stride(),
            src.stride(0),
            dst.stride(0),
            *edge.stride(),
            *rbf.stride(),
            p.shape[1],
            edge.shape[1],
            rbf.shape[1],
            eps,
            triton.next_power_of_2(width),
            num_warps=4,
            enable_fp_fusion=False,
        )
        return out

    return normalize


_edge_normalization = _load_edge_normalization() if torch.cuda.is_available() else None


def normalized_edge_features(p, edge_index, edge, distance, norm):
    """Return fused CUDA inference features, or ``None`` for the native path."""
    if (
        _edge_normalization is None
        or not p.is_cuda
        or torch.is_grad_enabled()
        or any(
            x.dtype not in (torch.float32, torch.bfloat16) for x in (p, edge, distance)
        )
        or not isinstance(norm, nn.LayerNorm)
        or norm.weight is None
        or norm.bias is None
        or norm.weight.dtype != torch.float32
        or norm.bias.dtype != torch.float32
        or not norm.weight.is_contiguous()
        or not norm.bias.is_contiguous()
        or norm.normalized_shape
        != (2 * p.shape[1] + edge.shape[1] + distance.shape[1],)
        or (p.dtype != torch.float32 and not torch.is_autocast_enabled("cuda"))
    ):
        return None
    return _edge_normalization(
        p, *edge_index, edge, distance, norm.weight, norm.bias, norm.eps
    )
