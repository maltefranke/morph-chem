# Optional CUDA/compiler imports stay behind test skip gates.
# ruff: noqa: PLC0415

import copy
import os

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from morph.model.layers import attention
from morph.model.layers.attention import (
    PairBias,
    fix_efficient_attention_bias_metadata,
    packed_self_attention,
    pad_dense,
    unpad_dense,
    unpad_dense_unique,
)
from morph.model.layers.embedding import RBFEmbedding


def _complete_ragged_edges(
    sequence_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_chunks = []
    start = 0
    for length in sequence_lengths.tolist():
        local = torch.arange(length)
        rows, cols = torch.where(local[:, None] != local[None, :])
        edge_chunks.append(torch.stack((rows + start, cols + start)))
        start += length
    edge_index = torch.cat(edge_chunks, dim=1)
    return edge_index[0], edge_index[1]


def _pair_bias() -> PairBias:
    return PairBias(
        rbf_embedding=RBFEmbedding(
            num_rbf=8,
            rbf_dmax=4.0,
            out_dim=8,
            dropout=0.0,
            trainable=False,
        ),
        rbf_out_dim=8,
        edge_nf=5,
        num_heads=4,
        hidden_dim=8,
        num_atom_types=6,
    )


def _dense_pair_bias_reference(
    module: PairBias,
    x: torch.Tensor,
    edge_index: tuple[torch.Tensor, torch.Tensor],
    e_embed: torch.Tensor,
    batch: torch.Tensor,
    batch_size: int,
    max_seqlen: int,
    atom_types: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Previous padded implementation, retained only as a test oracle."""
    x_dense, atom_mask = to_dense_batch(
        x, batch, batch_size=batch_size, max_num_nodes=max_seqlen
    )
    distances = (x_dense.unsqueeze(2) - x_dense.unsqueeze(1)).norm(dim=-1)
    rbf = module.rbf_embedding(distances.reshape(-1)).reshape(*distances.shape, -1)
    hidden = module.dist_to_hidden(rbf)

    rows, cols = edge_index
    lengths = torch.bincount(batch, minlength=batch_size)
    starts = lengths.cumsum(0) - lengths
    edge_graph = batch[rows]
    local_rows = rows - starts[edge_graph]
    local_cols = cols - starts[edge_graph]
    edge_hidden = module._edge_hidden(e_embed, rows, cols, atom_types).to(hidden.dtype)
    hidden[edge_graph, local_rows, local_cols] = (
        hidden[edge_graph, local_rows, local_cols] + edge_hidden
    )

    bias = module.down_proj(module.act(hidden)).permute(0, 3, 1, 2).contiguous()
    padding = ~(atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1))
    bias.masked_fill_(padding.unsqueeze(1), float("-inf"))
    return bias, atom_mask


def _select_real_pairs(
    dense_bias: torch.Tensor, sequence_lengths: torch.Tensor
) -> torch.Tensor:
    return torch.cat(
        [
            dense_bias[index, :, :length, :length]
            .permute(1, 2, 0)
            .reshape(-1, dense_bias.shape[1])
            for index, length in enumerate(sequence_lengths.tolist())
        ]
    )


def test_packed_pair_bias_matches_dense_outputs_and_all_gradients():
    torch.manual_seed(0)
    lengths = torch.tensor([2, 4, 1])
    batch = torch.arange(lengths.numel()).repeat_interleave(lengths)
    edge_index = _complete_ragged_edges(lengths)
    atom_types = torch.tensor([0, 1, 2, 3, 4, 5, 1])

    dense_module = _pair_bias()
    packed_module = copy.deepcopy(dense_module)
    x_dense = torch.randn(lengths.sum(), 3, requires_grad=True)
    e_dense = torch.randn(edge_index[0].numel(), 5, requires_grad=True)
    x_packed = x_dense.detach().clone().requires_grad_()
    e_packed = e_dense.detach().clone().requires_grad_()
    rbf_input_sizes = []
    packed_module.rbf_embedding.register_forward_pre_hook(
        lambda _module, args: rbf_input_sizes.append(args[0].numel())
    )

    dense, dense_mask = _dense_pair_bias_reference(
        dense_module,
        x_dense,
        edge_index,
        e_dense,
        batch,
        batch_size=lengths.numel(),
        max_seqlen=int(lengths.max()),
        atom_types=atom_types,
    )
    packed, packed_mask, edge_distance_embed = packed_module(
        x_packed,
        edge_index,
        e_packed,
        batch,
        batch_size=lengths.numel(),
        max_seqlen=int(lengths.max()),
        atom_types=atom_types,
    )
    torch.testing.assert_close(packed, dense, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(packed_mask, dense_mask)
    assert rbf_input_sizes == [edge_index[0].numel() + 1]
    assert rbf_input_sizes[0] < lengths.numel() * int(lengths.max()) ** 2
    rows, cols = edge_index
    expected_edge_embed = packed_module.rbf_embedding(
        (x_packed[rows] - x_packed[cols]).norm(dim=-1)
    )
    torch.testing.assert_close(edge_distance_embed, expected_edge_embed)

    dense_real = _select_real_pairs(dense, lengths)
    packed_real = _select_real_pairs(packed, lengths)
    output_gradient = torch.randn_like(packed_real)
    (dense_real * output_gradient).sum().backward()
    (packed_real * output_gradient).sum().backward()
    torch.testing.assert_close(x_packed.grad, x_dense.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(e_packed.grad, e_dense.grad, rtol=1e-4, atol=1e-5)
    for (dense_name, dense_parameter), (packed_name, packed_parameter) in zip(
        dense_module.named_parameters(), packed_module.named_parameters(), strict=True
    ):
        assert dense_name == packed_name
        assert dense_parameter.grad is not None, dense_name
        assert packed_parameter.grad is not None, packed_name
        torch.testing.assert_close(
            packed_parameter.grad,
            dense_parameter.grad,
            rtol=1e-4,
            atol=1e-5,
            msg=lambda message, name=dense_name: f"{name}: {message}",
        )


def test_packed_self_attention_sdpa_matches_dense_reference():
    torch.manual_seed(1)
    lengths = torch.tensor([2, 4, 1])
    batch = torch.arange(lengths.numel()).repeat_interleave(lengths)
    cu_seqlens = F.pad(lengths.cumsum(0), (1, 0)).int()
    max_seqlen = int(lengths.max())
    qkv = torch.randn(int(lengths.sum()), 3, 4, 8, requires_grad=True)
    qkv_reference = qkv.detach().clone().requires_grad_()
    qkv_dense, atom_mask = to_dense_batch(
        qkv_reference, batch, max_num_nodes=max_seqlen
    )
    pair_bias = torch.randn(lengths.numel(), 4, max_seqlen, max_seqlen)
    pair_mask = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1)
    pair_bias = pair_bias.masked_fill(~pair_mask.unsqueeze(1), float("-inf"))

    actual = packed_self_attention(
        qkv,
        cu_seqlens,
        max_seqlen,
        batch,
        pair_bias=pair_bias,
        atom_mask=atom_mask,
    )
    query, key, value = qkv_dense.unbind(dim=2)
    expected_dense = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=pair_bias,
    ).transpose(1, 2)
    expected = expected_dense[atom_mask]
    torch.testing.assert_close(actual, expected)

    gradient = torch.randn_like(actual)
    (actual * gradient).sum().backward()
    (expected * gradient).sum().backward()
    torch.testing.assert_close(qkv.grad, qkv_reference.grad)


def test_flash_attention_dispatches_only_for_unbiased_attention(monkeypatch):
    calls = []

    def fake_flash(qkv, cu_seqlens, max_seqlen, dropout_p, causal):
        calls.append((cu_seqlens, max_seqlen, dropout_p, causal))
        return qkv[:, 2]

    monkeypatch.setattr(attention, "flash_attention_supported", lambda _: True)
    monkeypatch.setattr(attention, "flash_attn_varlen_qkvpacked_func", fake_flash)
    monkeypatch.setattr(attention, "USE_FLASH_ATTN", True)
    qkv = torch.randn(3, 3, 2, 8)
    lengths = torch.tensor([3])
    batch = torch.zeros(3, dtype=torch.long)
    cu_seqlens = F.pad(lengths.cumsum(0), (1, 0)).int()

    output = packed_self_attention(qkv, cu_seqlens, 3, batch)
    torch.testing.assert_close(output, qkv[:, 2])
    assert len(calls) == 1

    packed_self_attention(
        qkv,
        cu_seqlens,
        3,
        batch,
        pair_bias=torch.zeros(1, 2, 3, 3),
        atom_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    assert len(calls) == 1


def test_flash_attention_is_guarded_on_cpu():
    assert not attention.flash_attention_supported(torch.zeros(1))


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not attention._HAS_FLASH_ATTN
    or torch.cuda.get_device_capability()[0] < 8,
    reason="requires an Ampere+ GPU and the optional flash-attn dependency",
)
def test_flash_attention_matches_sdpa_forward_and_backward(monkeypatch):
    torch.manual_seed(2)
    monkeypatch.setattr(attention, "USE_FLASH_ATTN", True)
    device = torch.device("cuda")
    lengths = torch.tensor([3, 5, 2], device=device)
    batch = torch.arange(lengths.numel(), device=device).repeat_interleave(lengths)
    cu_seqlens = F.pad(lengths.cumsum(0), (1, 0)).int()
    max_seqlen = int(lengths.max())
    qkv = torch.randn(
        int(lengths.sum()),
        3,
        4,
        32,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    qkv_reference = qkv.detach().clone().requires_grad_()

    actual = packed_self_attention(qkv, cu_seqlens, max_seqlen, batch)
    qkv_dense, atom_mask = to_dense_batch(
        qkv_reference, batch, max_num_nodes=max_seqlen
    )
    query, key, value = qkv_dense.unbind(dim=2)
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=atom_mask[:, None, None, :],
    ).transpose(1, 2)[atom_mask]
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    gradient = torch.randn_like(actual)
    (actual * gradient).sum().backward()
    (expected * gradient).sum().backward()
    torch.testing.assert_close(qkv.grad, qkv_reference.grad, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("compile_model", [False, True])
def test_unique_mapping_outputs_and_gradients(device, compile_model):
    if device == "cuda" and not torch.cuda.is_available():
        assert not os.environ.get("MORPH_REQUIRE_CUDA_TESTS"), (
            "Allocated GPU is unavailable"
        )
        pytest.skip("CUDA required")
    if device == "cpu" and compile_model:
        pytest.skip("Compilation exercised on the training GPU")

    def candidate(x, slots, length):
        padded = pad_dense(x, slots, 3, length)
        # Transpose creates non-contiguous dense features, as attention does.
        return unpad_dense_unique(padded.transpose(2, 3), slots)

    if compile_model:
        candidate = torch.compile(candidate, dynamic=True)
    for length in (3, 5, 4):
        slots = torch.tensor([0, 1, length, 2 * length + 1], device=device)
        x = torch.randn(4, 2, 3, device=device, requires_grad=True)
        expected = unpad_dense(pad_dense(x, slots, 3, length).transpose(2, 3), slots)
        actual = candidate(x, slots, length)
        probe = torch.randn_like(expected)
        (expected_grad,) = torch.autograd.grad((expected * probe).sum(), (x,))
        (actual_grad,) = torch.autograd.grad((actual * probe).sum(), (x,))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
        # Exercise unpadding's backward independently, including padding zeros.
        dense = torch.randn(3, length, 2, 3, device=device, requires_grad=True)
        output = unpad_dense_unique(dense, slots)
        (grad,) = torch.autograd.grad(output.sum(), (dense,))
        reference = torch.zeros_like(dense).reshape(-1, 2, 3)
        reference[slots] = 1
        torch.testing.assert_close(grad, reference.reshape_as(dense), atol=0, rtol=0)


@pytest.mark.parametrize("lengths", [(31, 32, 33, 48, 16), (32, 34, 48, 33, 64, 65)])
def test_compiled_bias_gradients_across_alignment_boundaries(lengths):
    if not torch.cuda.is_available():
        assert not os.environ.get("MORPH_REQUIRE_CUDA_TESTS"), (
            "Allocated GPU is unavailable"
        )
        pytest.skip("CUDA attention backward required")
    fix_efficient_attention_bias_metadata()

    def attention(q, k, v, bias):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)

    torch._dynamo.reset()
    compiled = torch.compile(attention, dynamic=True)
    torch.manual_seed(23)
    for length in lengths:
        inputs = [
            torch.randn(
                2,
                2,
                length,
                32,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for _ in range(3)
        ]
        inputs.append(
            torch.randn(
                2,
                2,
                length,
                length,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
        )
        expected = attention(*inputs)
        expected_grads = torch.autograd.grad(expected.float().square().sum(), inputs)
        actual = compiled(*inputs)
        actual_grads = torch.autograd.grad(actual.float().square().sum(), inputs)
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.005)
        for actual_grad, expected_grad in zip(
            actual_grads, expected_grads, strict=True
        ):
            assert torch.isfinite(actual_grad).all()
            error = (actual_grad.float() - expected_grad.float()).norm()
            assert error / expected_grad.float().norm().clamp_min(1e-8) < 0.03


@pytest.mark.parametrize("graphs", [2, 1])
def test_compiled_dit_bias_shape_from_aligned_first_batch(graphs):
    if not torch.cuda.is_available():
        assert not os.environ.get("MORPH_REQUIRE_CUDA_TESTS"), (
            "Allocated GPU is unavailable"
        )
        pytest.skip("CUDA required")
    from omegaconf import OmegaConf

    from morph.model.backbones.dit import DiTBackbone

    fix_efficient_attention_bias_metadata()
    torch._dynamo.reset()
    torch.manual_seed(123)
    model = DiTBackbone(
        d_input=16,
        d_cond=16,
        d_model=128,
        out_node_nf=16,
        out_edge_nf=16,
        num_layers=2,
        nhead=4,
        in_edge_nf=8,
        pair_bias_hidden_dim=8,
        edge_pair_dim=8,
        # Bias-shape regression without the atom-pair term (no atom_types input).
        num_atom_types=None,
        rbf_embedding_args=OmegaConf.create(
            dict(
                _target_="morph.model.layers.embedding.RBFEmbedding",
                num_rbf=8,
                rbf_dmax=4.0,
                out_dim=8,
                dropout=0.0,
                trainable=False,
            )
        ),
    ).cuda()
    # Exercise learned geometry rather than the zero-gated initialization.
    for block in model.blocks:
        block.adaLN_modulation[-1].bias.data.fill_(0.2)
    compiled = torch.compile(model, dynamic=True)
    gradient_parameters = [model.pair_bias.down_proj.weight]
    gradient_parameters.extend(block.geometry_scale for block in model.blocks)
    for size in (32, 34, 48, 33, 64, 35):
        edges = torch.where(~torch.eye(size, dtype=torch.bool, device="cuda"))
        edges = tuple(
            torch.cat([indices + size * g for g in range(graphs)]) for indices in edges
        )
        args = (
            torch.randn(graphs * size, 16, device="cuda"),
            torch.randn(graphs * size, 3, device="cuda"),
            torch.randn(graphs, 16, device="cuda"),
            edges,
            torch.randn(graphs * size * (size - 1), 8, device="cuda"),
            torch.arange(graphs, device="cuda").repeat_interleave(size),
            size,
        )
        kwargs = {"node_counts": torch.full((graphs,), size, device="cuda")}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = model(*args, **kwargs)
        expected_grads = torch.autograd.grad(
            sum(t.float().square().mean() for t in expected[:3]),
            gradient_parameters,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = compiled(*args, **kwargs)
        actual_grads = torch.autograd.grad(
            sum(t.float().square().mean() for t in actual[:3]),
            gradient_parameters,
        )
        # BF16 eager and fused projections differ at individual near-zero
        # outputs. Bound whole-tensor error, as in the real-step profiler.
        for a, b in zip(actual[:3], expected[:3], strict=True):
            assert torch.isfinite(a).all()
            assert (a.float() - b.float()).norm() / b.float().norm().clamp_min(
                1e-8
            ) < 0.02
        # The zero-initialized coordinate head must still remove graph COM.
        torch.testing.assert_close(actual[1], expected[1], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            actual[1].reshape(graphs, size, 3).mean(1),
            torch.zeros(graphs, 3, device="cuda"),
            atol=1e-6,
            rtol=0,
        )
        # In FP32 these gradients agree to ~1e-6; BF16 rounding alone reaches ~5%
        # for the small geometry-scale gradients.
        for actual_grad, expected_grad in zip(
            actual_grads, expected_grads, strict=True
        ):
            assert torch.isfinite(actual_grad).all()
            assert actual_grad.norm() > 0
            assert (actual_grad - expected_grad).norm() / expected_grad.norm() < 0.1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_normalized_packing_matches_native_attention(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from morph.model.layers.attention import compute_dense_index

    counts = torch.tensor([3, 0, 7, 1], device="cuda")
    offsets = F.pad(counts.cumsum(0), (1, 0)).int()
    batch = torch.repeat_interleave(torch.arange(4, device="cuda"), counts)
    slots = compute_dense_index(batch, 4, 7, counts)
    mask = torch.arange(7, device="cuda")[None, :] < counts[:, None]
    qkv = torch.randn(11, 3, 2, 24, device="cuda", dtype=dtype)
    q_norm = torch.nn.RMSNorm(24, eps=1e-6, device="cuda")
    k_norm = torch.nn.RMSNorm(24, eps=1e-6, device="cuda")
    with torch.no_grad():
        q_norm.weight.normal_()
        k_norm.weight.normal_()

    def fused(qkv):
        return packed_self_attention(
            qkv,
            offsets,
            7,
            batch,
            atom_mask=mask,
            dense_slot=slots,
            q_norm=q_norm,
            k_norm=k_norm,
        )

    with torch.inference_mode():
        q, k, v = qkv.unbind(1)
        normalized = torch.stack((q_norm(q).to(dtype), k_norm(k).to(dtype), v), 1)
        expected = packed_self_attention(
            normalized, offsets, 7, batch, atom_mask=mask, dense_slot=slots
        )
        actual = torch.compile(fused, fullgraph=True, dynamic=True)(qkv)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
