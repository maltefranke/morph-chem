# Optional CUDA/compiler imports stay behind test skip gates.
# ruff: noqa: PLC0415

"""Kernels regression tests."""

import math
import os

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_fused_edge_features_match_native_normalization(dtype, compiled):
    if not torch.cuda.is_available():
        assert not os.environ.get("MORPH_REQUIRE_CUDA_TESTS"), (
            "Allocated GPU is unavailable"
        )
        pytest.skip("CUDA required")
    from morph.model.layers.kernels import normalized_edge_features

    torch.manual_seed(53)
    norm = torch.nn.LayerNorm(37, device="cuda")
    with torch.no_grad():
        norm.weight.normal_()
        norm.bias.normal_()
    normalize = (
        torch.compile(normalized_edge_features, fullgraph=True, dynamic=True)
        if compiled
        else normalized_edge_features
    )
    # Strided inputs and changing edge counts exercise the kernel's addressing.
    for count in (17, 193, 0):
        p = torch.randn(43, 24, device="cuda", dtype=dtype)[:, ::2]
        edge = torch.randn(count, 10, device="cuda", dtype=dtype)[:, ::2]
        distance = torch.randn(count, 16, device="cuda", dtype=dtype)[:, ::2]
        rows, cols = torch.randint(43, (2, count), device="cuda")
        assert normalized_edge_features(p, (rows, cols), edge, distance, norm) is None
        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda", enabled=dtype == torch.bfloat16, dtype=torch.bfloat16
            ),
        ):
            expected = norm(
                torch.cat(
                    (
                        math.sqrt(0.5) * (p[rows] + p[cols]),
                        math.sqrt(0.5) * (p[rows] - p[cols]).abs(),
                        edge,
                        distance,
                    ),
                    dim=-1,
                )
            )
            actual = normalize(p, (rows, cols), edge, distance, norm)
            assert actual is not None
            assert actual.dtype == expected.dtype
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            swapped = normalize(p, (cols, rows), edge, distance, norm)
            torch.testing.assert_close(actual, swapped, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_fused_broadcast_variable_lengths_and_gradients(dtype):
    if not torch.cuda.is_available():
        assert not os.environ.get("MORPH_REQUIRE_CUDA_TESTS"), (
            "Allocated GPU is unavailable"
        )
        pytest.skip("CUDA required")
    from morph.model.layers.kernels import _load_fused_graph_broadcast

    compiled = torch.compile(_load_fused_graph_broadcast(), dynamic=True)
    torch.manual_seed(91)
    for lengths in ([0, 1, 2, 31, 32, 33, 70], [0, 3, 40, 0]):
        lengths = torch.tensor(lengths, device="cuda")
        offsets = torch.nn.functional.pad(lengths.cumsum(0), (1, 0)).int()
        batch = torch.arange(len(lengths), device="cuda").repeat_interleave(lengths)
        values = torch.randn(
            len(lengths), 67, device="cuda", dtype=dtype, requires_grad=True
        )
        probe = torch.randn(batch.numel(), 67, device="cuda", dtype=dtype)
        expected = values[batch]
        actual = compiled(values, batch, offsets)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        (reference,) = torch.autograd.grad(
            (expected.float() * probe.float()).sum(), (values,)
        )
        (result,) = torch.autograd.grad(
            (actual.float() * probe.float()).sum(), (values,)
        )
        assert torch.isfinite(result).all()
        error = (
            result.float() - reference.float()
        ).norm() / reference.float().norm().clamp_min(1e-8)
        assert error < (0.03 if dtype == torch.bfloat16 else 1e-6)
        assert torch.count_nonzero(result[lengths == 0]) == 0
