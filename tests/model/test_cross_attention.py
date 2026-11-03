"""Conditioning placement, masking, gradients, and checkpoint compatibility."""

import copy

import hydra
import pytest
import torch
from omegaconf import OmegaConf

from morph.model.backbones import dit
from morph.model.backbones.conditional_dit import ConditionalDiTBackbone
from morph.utils.graph_math import batched_triu_edge_layout

from ._helpers import _complete_edges, _rbf_args


def _config(**kwargs):
    return OmegaConf.create(
        dict(
            _target_=("morph.model.backbones.conditional_dit.ConditionalDiTBackbone"),
            _recursive_=False,
            d_input=7,
            d_cond=6,
            ctx_dim=9,
            d_model=32,
            out_node_nf=16,
            out_edge_nf=10,
            num_layers=2,
            nhead=4,
            in_edge_nf=5,
            rbf_embedding_args=_rbf_args(),
            pair_bias_hidden_dim=8,
            ffn_multiple_of=8,
            edge_pair_dim=12,
            num_atom_types=6,
            use_cross_attn=True,
            **kwargs,
        )
    )


def _inputs():
    edges = torch.cat((_complete_edges(3), _complete_edges(2) + 3), dim=1)
    args = (
        torch.randn(5, 7),
        torch.randn(5, 3),
        torch.randn(2, 6),
        (edges[0], edges[1]),
        torch.randn(8, 5),
        torch.tensor([0, 0, 0, 1, 1]),
        3,
    )
    return args, dict(atom_types=torch.tensor([0, 1, 2, 3, 1]))


def _open_native(model):
    # Exercise a trained clean lane, rather than the zero-initialized readout
    # and adaLN gates hiding changes to the adapter's insertion point.
    with torch.no_grad():
        for block in model.dit.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=0.05)
            block.adaLN_modulation[-1].bias.fill_(0.2)
        model.dit.final_layer.linear.weight.normal_(std=0.1)
        model.dit.pos_out.output.weight.normal_(std=0.1)


def _open_adapters(model):
    with torch.no_grad():
        for adapter in model.cross_attn_blocks:
            adapter.attn_gate.fill_(0.3)
            adapter.ffn_gate.fill_(0.2)


@pytest.mark.parametrize("position", ["residual", "post_norm1"])
@pytest.mark.parametrize("dense_index", [False, True])
def test_zero_gates_preserve_pretrained_predictions(position, dense_index, monkeypatch):
    monkeypatch.setattr(dit, "USE_DENSE_INDEX", dense_index)
    torch.manual_seed(73)
    model = hydra.utils.instantiate(_config(cross_attn_position=position)).eval()
    _open_native(model)
    args, kwargs = _inputs()
    baseline = model.dit(*args, **kwargs)
    # Strictly loading a plain pretrained backbone leaves no new clean-lane keys.
    model.load_dit_state_dict(copy.deepcopy(model.dit.state_dict()), strict=True)
    with_context = model(
        *args,
        **kwargs,
        cond_tokens=torch.randn(2, 3, 9),
        cond_token_mask=torch.tensor([[True, True, False], [False] * 3]),
    )
    torch.testing.assert_close(with_context, baseline, rtol=0, atol=0)


@pytest.mark.parametrize("position", ["residual", "post_norm1"])
@pytest.mark.parametrize("dense_index", [False, True])
@pytest.mark.parametrize("bias_heads", [None, 1, 4])
def test_open_adapters_preserve_empty_rows_and_ignore_padding(
    position, dense_index, bias_heads, monkeypatch
):
    monkeypatch.setattr(dit, "USE_DENSE_INDEX", dense_index)
    torch.manual_seed(74)
    model = hydra.utils.instantiate(_config(cross_attn_position=position)).eval()
    _open_native(model)
    _open_adapters(model)
    args, kwargs = _inputs()
    baseline = model(*args, **kwargs)
    tokens = torch.randn(2, 3, 9)
    mask = torch.tensor([[True, True, False], [False] * 3])
    bias = None
    if bias_heads is not None:
        bias = torch.randn(2, bias_heads, 3, 3)
        if bias_heads == 1:
            bias = bias[:, 0]
    conditioned = model(
        *args,
        **kwargs,
        cond_tokens=tokens,
        cond_token_mask=mask,
        cond_token_attn_bias=bias,
    )
    assert not torch.allclose(conditioned[0][:3], baseline[0][:3])
    torch.testing.assert_close(conditioned[0][3:], baseline[0][3:], rtol=0, atol=0)
    torch.testing.assert_close(conditioned[1][3:], baseline[1][3:], rtol=0, atol=0)
    triu, _ = batched_triu_edge_layout(args[3], args[5], torch.tensor([3, 2]))
    empty_edges = args[5][args[3][0][triu]] == 1
    torch.testing.assert_close(
        conditioned[2][empty_edges], baseline[2][empty_edges], rtol=0, atol=0
    )
    padded = tokens.clone()
    padded[~mask] = 1000 * torch.randn_like(padded[~mask])
    # Keep the placeholder token for an empty row finite; its result is masked.
    torch.testing.assert_close(
        model(
            *args,
            **kwargs,
            cond_tokens=padded,
            cond_token_mask=mask,
            cond_token_attn_bias=bias,
        ),
        conditioned,
        rtol=0,
        atol=0,
    )
    for empty_tokens, empty_mask in (
        (tokens[:, :0], mask[:, :0]),
        (tokens, torch.zeros_like(mask)),
    ):
        torch.testing.assert_close(
            model(
                *args, **kwargs, cond_tokens=empty_tokens, cond_token_mask=empty_mask
            ),
            baseline,
            rtol=0,
            atol=0,
        )


def test_post_norm1_changes_attention_input_and_keeps_raw_skip():
    torch.manual_seed(75)
    model = hydra.utils.instantiate(_config(cross_attn_position="post_norm1")).eval()
    _open_adapters(model)
    args, kwargs = _inputs()
    # Native attention and FFN gates stay zero, so even open adapters must leave
    # the raw residual untouched. Attention inputs must still change.
    qkv_inputs, adapter_inputs, block_skips, norm_calls = [], [], [], []
    handles = []
    for block, adapter in zip(model.dit.blocks, model.cross_attn_blocks, strict=True):
        handles.extend(
            [
                block.qkv.register_forward_pre_hook(
                    lambda _module, inputs: qkv_inputs.append(
                        inputs[0].detach().clone()
                    )
                ),
                block.norm1.register_forward_hook(
                    lambda _module, _inputs, _output: norm_calls.append(True)
                ),
                adapter.register_forward_pre_hook(
                    lambda _module, inputs: adapter_inputs.append(
                        inputs[0].detach().clone()
                    )
                ),
                block.register_forward_hook(
                    lambda _module, inputs, output: block_skips.append(
                        (inputs[0].detach().clone(), output.detach().clone())
                    )
                ),
            ]
        )
    try:
        baseline = model(*args, **kwargs)
        baseline_qkv = qkv_inputs[:]
        qkv_inputs.clear()
        norm_calls.clear()
        conditioned = model(*args, **kwargs, cond_tokens=torch.randn(2, 3, 9))
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(conditioned, baseline, rtol=0, atol=0)
    assert len(norm_calls) == model.num_layers
    for raw, updated in block_skips:
        torch.testing.assert_close(updated, raw, rtol=0, atol=0)
    for original, conditioned_input in zip(baseline_qkv, qkv_inputs, strict=True):
        assert not torch.allclose(conditioned_input, original)
    for normalized in adapter_inputs:
        valid = normalized[torch.tensor([[True] * 3, [True, True, False]])]
        torch.testing.assert_close(valid.mean(-1), torch.zeros(5), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            valid.square().mean(-1), torch.ones(5), atol=1e-4, rtol=0
        )


def test_post_norm1_trains_gates_and_context_with_a_frozen_generator():
    torch.manual_seed(76)
    model = hydra.utils.instantiate(_config(cross_attn_position="post_norm1"))
    _open_native(model)
    model.dit.requires_grad_(False)
    args, kwargs = _inputs()
    tokens = torch.randn(2, 3, 9, requires_grad=True)
    mask = torch.tensor([[True, True, False], [False] * 3])
    optimizer = torch.optim.AdamW(model.cross_attn_blocks.parameters(), lr=1e-3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        tokens.grad = None
        outputs = model(*args, **kwargs, cond_tokens=tokens, cond_token_mask=mask)
        sum(output.square().mean() for output in outputs[:3]).backward()
        for adapter in model.cross_attn_blocks:
            assert torch.isfinite(adapter.attn_gate.grad).all()
            assert adapter.attn_gate.grad.abs().sum() > 0
            if step:
                for projection in (adapter.q_proj, adapter.kv_proj, adapter.out_proj):
                    assert torch.isfinite(projection.weight.grad).all()
                    assert projection.weight.grad.abs().sum() > 0
        assert all(parameter.grad is None for parameter in model.dit.parameters())
        assert torch.isfinite(tokens.grad).all()
        assert tokens.grad[~mask].abs().sum() == 0
        if step:
            assert tokens.grad[mask].abs().sum() > 0
        optimizer.step()


@pytest.mark.parametrize("position", [None, "residual", "post_norm1"])
def test_checkpoint_config_and_strict_loading_preserve_placement(position, tmp_path):
    torch.manual_seed(77)
    config = _config(**({} if position is None else {"cross_attn_position": position}))
    model = hydra.utils.instantiate(config).eval()
    _open_native(model)
    _open_adapters(model)
    args, kwargs = _inputs()
    kwargs["cond_tokens"] = torch.randn(2, 3, 9)
    expected = model(*args, **kwargs)
    checkpoint = tmp_path / "conditional.pt"
    torch.save(
        dict(config=OmegaConf.to_container(config), state_dict=model.state_dict()),
        checkpoint,
    )
    saved = torch.load(checkpoint, weights_only=True)
    restored = hydra.utils.instantiate(OmegaConf.create(saved["config"])).eval()
    restored.load_state_dict(saved["state_dict"], strict=True)
    assert restored.cross_attn_position == (position or "residual")
    torch.testing.assert_close(restored(*args, **kwargs), expected, rtol=0, atol=0)
    # The new placement adds no parameters, buffers, or random initialization.
    torch.manual_seed(78)
    legacy = hydra.utils.instantiate(_config())
    rng = torch.random.get_rng_state()
    torch.manual_seed(78)
    candidate = hydra.utils.instantiate(config)
    assert torch.equal(torch.random.get_rng_state(), rng)
    torch.testing.assert_close(
        candidate.state_dict(), legacy.state_dict(), rtol=0, atol=0
    )


def test_invalid_cross_attention_position_is_rejected():
    with pytest.raises(ValueError, match="cross_attn_position"):
        ConditionalDiTBackbone(7, 6, cross_attn_position="unknown")


def test_post_norm1_compilation_preserves_outputs_gradients_and_checkpoint_keys():
    torch.manual_seed(79)
    model = hydra.utils.instantiate(_config(cross_attn_position="post_norm1"))
    _open_native(model)
    _open_adapters(model)
    model.dit.requires_grad_(False)
    candidate = copy.deepcopy(model)
    keys = set(candidate.state_dict())
    candidate.compile(backend="eager", dynamic=True)
    args, kwargs = _inputs()
    tokens = torch.randn(2, 3, 9, requires_grad=True)
    compiled_tokens = tokens.detach().clone().requires_grad_()
    expected = model(*args, **kwargs, cond_tokens=tokens)
    actual = candidate(*args, **kwargs, cond_tokens=compiled_tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradients = []
    for network, outputs, context in (
        (model, expected, tokens),
        (candidate, actual, compiled_tokens),
    ):
        gradients.append(
            torch.autograd.grad(
                sum(output.square().mean() for output in outputs[:3]),
                (context, network.cross_attn_blocks[0].kv_proj.weight),
            )
        )
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    assert set(candidate.state_dict()) == keys
