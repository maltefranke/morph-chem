"""Dit regression tests."""

import copy
from types import SimpleNamespace

import hydra
import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.utils import scatter

import morph.model.backbones.dit as dit
from morph.model.backbones.conditional_dit import ConditionalDiTBackbone
from morph.model.backbones.dit import (
    DiTBlock,
    ResidualCoordinateReadout,
    SymmetricEdgeDecoder,
)
from morph.model.layers.attention import PairBias
from morph.model.lightning.base import LightningModuleRates
from morph.utils.graph_math import batched_triu_edge_layout

from ._helpers import (
    _backbone,
    _complete_edges,
    _layer_bias_backbone,
    _rbf_args,
)


def test_swiglu_matches_four_x_gelu_weight_budget():
    block = DiTBlock(768, 12, mlp_ratio=4.0)
    assert block.mlp.hidden_features == 2048
    assert 3 * 768 * block.mlp.hidden_features == 2 * 768 * (4 * 768)


@pytest.mark.parametrize(
    ("pair_bias_all_layers", "expected"),
    [
        (None, [True, True, True, True]),
        (True, [True, True, True, True]),
        (False, [True, False, False, False]),
    ],
)
def test_pair_bias_default_and_legacy_depth(
    monkeypatch, pair_bias_all_layers, expected
):
    model = _backbone(num_layers=4, pair_bias_all_layers=pair_bias_all_layers)
    calls = []
    real_attention = dit.packed_self_attention

    def attention_spy(*args, **kwargs):
        calls.append(kwargs["pair_bias"] is not None)
        return real_attention(*args, **kwargs)

    monkeypatch.setattr(dit, "packed_self_attention", attention_spy)
    edge_index = _complete_edges(3)
    model(
        torch.randn(3, 7),
        torch.randn(3, 3),
        torch.randn(1, 6),
        (edge_index[0], edge_index[1]),
        torch.randn(edge_index.shape[1], 5),
        torch.zeros(3, dtype=torch.long),
        3,
        atom_types=torch.tensor([0, 1, 2]),
    )
    assert calls == expected


def test_attention_reuses_pair_mask_and_dense_index(monkeypatch):
    model = _backbone(num_layers=3)
    calls = []

    def attention_spy(qkv, *args, **kwargs):
        calls.append(
            (
                kwargs["pair_bias"],
                kwargs["atom_mask"],
                kwargs["dense_slot"],
            )
        )
        return qkv[:, 2]

    monkeypatch.setattr(dit, "packed_self_attention", attention_spy)
    edge_index = _complete_edges(3)
    model(
        torch.randn(3, 7),
        torch.randn(3, 3),
        torch.randn(1, 6),
        (edge_index[0], edge_index[1]),
        torch.randn(edge_index.shape[1], 5),
        torch.zeros(3, dtype=torch.long),
        3,
        atom_types=torch.tensor([0, 1, 2]),
    )
    assert len(calls) == 3
    assert calls[0][0] is not None
    for call in calls:
        torch.testing.assert_close(call[0], calls[0][0])
    assert all(call[1] is calls[0][1] for call in calls)
    assert all(call[2] is calls[0][2] for call in calls)


@pytest.mark.parametrize("use_graph_cond", [False, True])
def test_disabled_conditional_ports_match_the_base_dit(use_graph_cond):
    base = _backbone(num_layers=2).eval()
    conditional = ConditionalDiTBackbone(
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
        use_graph_cond=use_graph_cond,
    ).eval()
    conditional.dit.load_state_dict(base.state_dict())
    edge_index = _complete_edges(3)
    inputs = (
        torch.randn(3, 7),
        torch.randn(3, 3),
        torch.randn(1, 6),
        (edge_index[0], edge_index[1]),
        torch.randn(edge_index.shape[1], 5),
        torch.zeros(3, dtype=torch.long),
        3,
    )
    atom_types = torch.tensor([0, 1, 2])
    expected = base(*inputs, atom_types=atom_types)
    actual = conditional(*inputs, atom_types=atom_types)
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    graph_cond = torch.randn(1, 32)
    if use_graph_cond:
        actual = conditional(*inputs, atom_types=atom_types, graph_cond=graph_cond)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for shape in ((2, 32), (1, 6), (32,)):
            with pytest.raises(ValueError, match="graph_cond shape"):
                conditional(*inputs, graph_cond=torch.randn(shape))
    else:
        assert "graph_gate" not in conditional.state_dict()
        with pytest.raises(ValueError, match="use_graph_cond=False"):
            conditional(*inputs, graph_cond=graph_cond)


def test_graph_conditioning_composes_with_node_and_edge_ports():
    model = ConditionalDiTBackbone(
        d_input=7,
        d_cond=6,
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
        use_graph_cond=True,
        use_node_cond=True,
        use_edge_cond=True,
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "adaLN_modulation" in name and name.endswith("weight"):
                parameter.normal_(std=0.02)
        model.graph_gate.fill_(0.2)
        model.node_gate.fill_(0.3)
        model.edge_cond_gate.fill_(0.4)
    edges = _complete_edges(3)
    inputs = (
        torch.randn(3, 7),
        torch.randn(3, 3),
        torch.randn(1, 6),
        (edges[0], edges[1]),
        torch.randn(6, 5),
        torch.zeros(3, dtype=torch.long),
        3,
    )
    graph_cond = torch.randn(1, 32, requires_grad=True)
    node_cond = torch.randn(3, 32, requires_grad=True)
    edge_cond = torch.randn(6, 5, requires_grad=True)
    outputs = model(
        *inputs,
        graph_cond=graph_cond,
        node_cond=node_cond,
        edge_cond=edge_cond,
        atom_types=torch.tensor([0, 1, 2]),
    )
    sum(output.square().sum() for output in outputs[:3]).backward()
    for tensor in (
        graph_cond,
        node_cond,
        edge_cond,
        model.graph_gate,
        model.node_gate,
        model.edge_cond_gate,
    ):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def test_symmetric_edge_decoder_is_symmetric_in_train_mode():
    decoder = SymmetricEdgeDecoder(16, 5, 8, 16, pair_dim=12).train()
    h = torch.randn(2, 16)
    edge_embed = torch.randn(1, 5).expand(2, -1).clone()
    distance_embed = torch.randn(1, 8).expand(2, -1).clone()
    out = decoder(
        h,
        (torch.tensor([0, 1]), torch.tensor([1, 0])),
        edge_embed,
        distance_embed,
    )
    # Both directions receive bitwise-identical pair features, but CPU BLAS
    # kernels (e.g. on AMD EPYC) may round identical GEMM rows differently.
    torch.testing.assert_close(out[0], out[1], rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("fused_features", [False, True])
def test_edge_decoder_loads_legacy_weights_without_changing_outputs_or_gradients(
    monkeypatch, fused_features
):
    decoder = SymmetricEdgeDecoder(16, 5, 8, 16, pair_dim=12)
    legacy = torch.nn.ModuleDict(
        {
            "node_proj": copy.deepcopy(decoder.node_proj),
            "output": torch.nn.Sequential(
                torch.nn.LayerNorm(37),
                torch.nn.Linear(37, 16),
                torch.nn.LayerNorm(16),
                torch.nn.GELU(),
            ),
        }
    )
    # Exercise loading under a parent prefix, as in training and EMA checkpoints.
    parent = torch.nn.ModuleDict({"decoder": decoder})
    parent.load_state_dict(
        torch.nn.ModuleDict({"decoder": legacy}).state_dict(), strict=True
    )

    def pair_features(p, edges, edge, distance):
        left, right = p[edges[0]], p[edges[1]]
        return torch.cat(
            ((left + right) * 2**-0.5, (left - right).abs() * 2**-0.5, edge, distance),
            dim=-1,
        )

    if fused_features:
        # Exercise the already-normalized feature path on CPU. CUDA kernel
        # numerics are covered separately in test_kernels.py.
        monkeypatch.setattr(
            dit,
            "normalized_edge_features",
            lambda p, edges, edge, distance, norm: norm(
                pair_features(p, edges, edge, distance)
            ),
        )
    h = torch.randn(3, 16, requires_grad=True)
    edges = _complete_edges(3)
    edge = torch.randn(6, 5, requires_grad=True)
    distance = torch.randn(6, 8, requires_grad=True)
    expected = legacy["output"](
        pair_features(legacy["node_proj"](h), edges, edge, distance)
    )
    actual = decoder(h, (edges[0], edges[1]), edge, distance)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probe = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(
        (expected * probe).sum(), (h, edge, distance, *legacy.parameters())
    )
    actual_grads = torch.autograd.grad(
        (actual * probe).sum(), (h, edge, distance, *decoder.parameters())
    )
    torch.testing.assert_close(actual_grads, expected_grads, rtol=0, atol=0)


def test_undirected_edge_path_matches_full_bidirectional_evaluation():
    edge_index = _complete_edges(3)
    batch = torch.zeros(3, dtype=torch.long)
    triu_positions, full_to_triu = batched_triu_edge_layout(
        edge_index, batch, torch.tensor([3])
    )
    h = torch.randn(3, 16)
    triu_e = torch.randn(3, 5)
    triu_distance = torch.randn(3, 8)
    e_embed = triu_e[full_to_triu]
    distance_embed = triu_distance[full_to_triu]
    decoder = SymmetricEdgeDecoder(16, 5, 8, 16, pair_dim=12).eval()

    full = decoder(
        h,
        (edge_index[0], edge_index[1]),
        e_embed,
        distance_embed,
    )
    triu = decoder(
        h,
        (
            edge_index[0][triu_positions],
            edge_index[1][triu_positions],
        ),
        e_embed[triu_positions],
        distance_embed[triu_positions],
    )

    torch.testing.assert_close(triu[full_to_triu], full)


def test_triu_layout_restores_symmetric_attributes_after_edge_reordering():
    edge_index = _complete_edges(4)
    batch = torch.zeros(4, dtype=torch.long)
    node_counts = torch.tensor([4])
    _, canonical_map = batched_triu_edge_layout(edge_index, batch, node_counts)
    edge_types = torch.arange(6)[canonical_map]

    permutation = torch.tensor([7, 1, 10, 3, 5, 0, 8, 11, 2, 6, 4, 9])
    edge_index = edge_index[:, permutation]
    edge_types = edge_types[permutation]
    triu_positions, full_to_triu = batched_triu_edge_layout(
        edge_index, batch, node_counts
    )

    torch.testing.assert_close(edge_types[triu_positions][full_to_triu], edge_types)


def test_pair_bias_matches_undirected_evaluation():
    edge_index = _complete_edges(3)
    batch = torch.zeros(3, dtype=torch.long)
    node_counts = torch.tensor([3])
    triu_positions, full_to_triu = batched_triu_edge_layout(
        edge_index, batch, node_counts
    )
    triu_e = torch.randn(3, 5)
    e_embed = triu_e[full_to_triu]
    pair_bias = PairBias(
        rbf_embedding=hydra.utils.instantiate(_rbf_args()),
        rbf_out_dim=8,
        edge_nf=5,
        num_heads=4,
        hidden_dim=8,
        num_atom_types=6,
    ).eval()
    x = torch.randn(3, 3)
    atom_types = torch.tensor([0, 1, 2])

    full_bias, full_mask, full_distance = pair_bias(
        x,
        (edge_index[0], edge_index[1]),
        e_embed,
        batch,
        1,
        3,
        atom_types=atom_types,
        lengths=node_counts,
    )
    triu_bias, triu_mask, triu_distance = pair_bias(
        x,
        (edge_index[0], edge_index[1]),
        e_embed,
        batch,
        1,
        3,
        atom_types=atom_types,
        lengths=node_counts,
        triu_edge_positions=triu_positions,
    )

    torch.testing.assert_close(triu_bias, full_bias)
    torch.testing.assert_close(triu_mask, full_mask)
    torch.testing.assert_close(triu_distance, full_distance[triu_positions])


def test_backbone_forward_backward_shapes():
    torch.manual_seed(0)
    model = _backbone()
    edges_a = _complete_edges(3)
    edges_b = _complete_edges(2) + 3
    edge_index = torch.cat([edges_a, edges_b], dim=1)
    h, x_out, e_out_triu, full_to_triu = model(
        torch.randn(5, 7),
        torch.randn(5, 3),
        torch.randn(2, 6),
        (edge_index[0], edge_index[1]),
        torch.randn(edge_index.shape[1], 5),
        torch.tensor([0, 0, 0, 1, 1]),
        3,
        atom_types=torch.tensor([0, 1, 2, 3, 4]),
    )
    assert h.shape == (5, 16)
    assert x_out.shape == (5, 3)
    assert e_out_triu.shape == (edge_index.shape[1] // 2, 10)
    assert full_to_triu.shape == (edge_index.shape[1],)
    (h.square().mean() + x_out.square().mean() + e_out_triu.square().mean()).backward()
    assert model.final_layer.linear.weight.grad is not None
    assert model.edge_output.output[1].weight.grad is not None


def test_backbone_reuses_pair_bias_distance_embedding_for_edge_decode():
    model = _backbone(num_layers=1)
    rbf_input_sizes = []
    model.rbf_embedding.register_forward_pre_hook(
        lambda _module, args: rbf_input_sizes.append(args[0].numel())
    )
    edge_index = _complete_edges(3)

    model(
        torch.randn(3, 7),
        torch.randn(3, 3),
        torch.randn(1, 6),
        (edge_index[0], edge_index[1]),
        torch.randn(edge_index.shape[1], 5),
        torch.zeros(3, dtype=torch.long),
        3,
        atom_types=torch.tensor([0, 1, 2]),
    )

    assert rbf_input_sizes == [edge_index.shape[1] // 2 + 1]


def test_backbone_coordinate_head_starts_as_centered_residual():
    model = _backbone(num_layers=2).eval()
    edge_index = _complete_edges(3)
    positions = torch.randn(3, 3)
    _, x_out, _, _ = model(
        torch.randn(3, 7),
        positions,
        torch.randn(1, 6),
        (edge_index[0], edge_index[1]),
        torch.randn(edge_index.shape[1], 5),
        torch.zeros(3, dtype=torch.long),
        3,
        atom_types=torch.tensor([0, 1, 2]),
    )
    torch.testing.assert_close(x_out, positions - positions.mean(dim=0))


def test_atom_pair_bias_is_symmetric_zero_init_and_learnable():
    pair_bias = PairBias(
        rbf_embedding=hydra.utils.instantiate(_rbf_args()),
        rbf_out_dim=8,
        edge_nf=5,
        num_heads=4,
        hidden_dim=8,
        num_atom_types=6,
    )
    assert pair_bias.atom_pair_embedding is not None
    assert torch.count_nonzero(pair_bias.atom_pair_embedding.weight) == 0
    left = torch.tensor([1, 4, 2])
    right = torch.tensor([4, 2, 1])
    forward = pair_bias._atom_pair_indices(left, right)
    reverse = pair_bias._atom_pair_indices(right, left)
    torch.testing.assert_close(forward, reverse)
    pair_bias.atom_pair_embedding(forward).sum().backward()
    assert pair_bias.atom_pair_embedding.weight.grad is not None


def test_residual_coordinate_readout_is_graph_centered():
    readout = ResidualCoordinateReadout(16, 32)
    h = torch.randn(5, 16)
    x = torch.randn(5, 3)
    batch = torch.tensor([0, 0, 0, 1, 1])
    out = readout(h, x, batch, num_graphs=2)
    expected = x - scatter(x, batch, dim=0, dim_size=2, reduce="mean")[batch]
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(
        scatter(out, batch, dim=0, dim_size=2, reduce="mean"),
        torch.zeros(2, 3),
        rtol=0.0,
        atol=1e-6,
    )


def test_coordinate_readout_routes_only_matrices_to_muon():
    matrix = torch.nn.Parameter(torch.randn(4, 4))
    vector = torch.nn.Parameter(torch.randn(4))
    assert LightningModuleRates._use_muon("backbone.pos_out.hidden.weight", matrix)
    assert LightningModuleRates._use_muon("backbone.pos_out.output.weight", matrix)
    assert not LightningModuleRates._use_muon("backbone.pos_out.norm.weight", vector)
    assert not LightningModuleRates._use_muon("backbone.pos_out.output.bias", vector)


def test_torch_muon_and_adamw_follow_official_split():
    module = LightningModuleRates.__new__(LightningModuleRates)
    torch.nn.Module.__init__(module)
    module.model = torch.nn.ModuleDict(
        {
            "embedding_backbone": torch.nn.Linear(4, 4),
            "backbone": torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.GELU(),
                torch.nn.Linear(4, 4),
            ),
            "heads": torch.nn.Linear(4, 3),
            "node_cond_encoder": torch.nn.Embedding(2, 4),
            "xattn_encoder": torch.nn.ModuleDict(
                {
                    "atom_embed": torch.nn.Embedding(4, 4),
                    "blocks": torch.nn.Sequential(torch.nn.Linear(4, 4)),
                }
            ),
        }
    )
    module.loss_weight_wrapper = SimpleNamespace(
        use_learnable=False, learnable_wrapper=None
    )
    module.optimizer_config = OmegaConf.create(
        {
            "optimizer": {
                "lr": 0.01,
                "momentum": 0.95,
                "weight_decay": 0.0,
                "nesterov": True,
                "ns_steps": 5,
                "eps": 1e-7,
                "adjust_lr_fn": "original",
            },
            "aux_optimizer": {
                "lr": 3e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
            "scheduler": {
                "_target_": "morph.utils.lr_schedulers.MultiGroupCosineWarmupLR",
                "warmup_steps": 2,
                "total_steps": 10,
                "min_lr_fraction": 0.05,
                "last_epoch": -1,
            },
            "interval": "step",
            "monitor": None,
        }
    )

    optimizers, schedulers = module.configure_optimizers()
    muon, adamw = optimizers
    muon_params = {id(p) for group in muon.param_groups for p in group["params"]}
    adamw_params = {id(p) for group in adamw.param_groups for p in group["params"]}
    model_params = {id(p) for p in module.model.parameters()}

    assert type(muon) is torch.optim.Muon
    assert type(adamw) is torch.optim.AdamW
    assert len(schedulers) == 2
    assert not muon_params & adamw_params
    assert muon_params | adamw_params == model_params
    assert all(p.ndim == 2 for group in muon.param_groups for p in group["params"])
    assert id(module.model["node_cond_encoder"].weight) in adamw_params
    assert id(module.model["xattn_encoder"]["atom_embed"].weight) in adamw_params
    assert id(module.model["xattn_encoder"]["blocks"][0].weight) in muon_params
    assert not LightningModuleRates._use_muon(
        "backbone.kernel", torch.nn.Parameter(torch.randn(2, 2, 2))
    )


@pytest.mark.parametrize("mode", [None, "shared", "scale"])
def test_layer_bias_initialization_gradients_and_checkpoint(mode):
    torch.manual_seed(13)
    reference = _layer_bias_backbone("shared").double()
    state = torch.random.get_rng_state()
    torch.manual_seed(13)
    candidate = _layer_bias_backbone(mode).double()
    assert torch.equal(state, torch.random.get_rng_state())
    for name, param in reference.named_parameters():
        other = dict(candidate.named_parameters())[name]
        torch.testing.assert_close(param, other, atol=0, rtol=0)
    # Open the zero-initialized attention gates to test geometry gradients.
    for model in (reference, candidate):
        for block in model.blocks:
            block.adaLN_modulation[-1].bias.data.fill_(0.2)
    edges = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 4], [1, 2, 0, 2, 0, 1, 4, 3]])
    inputs = (
        torch.randn(5, 7, dtype=torch.double),
        torch.randn(5, 3, dtype=torch.double),
        torch.randn(2, 6, dtype=torch.double),
        edges,
        torch.randn(8, 5, dtype=torch.double),
        torch.tensor([0, 0, 0, 1, 1]),
        3,
    )
    kwargs = dict(atom_types=torch.tensor([0, 1, 2, 3, 1]))
    expected = reference(*inputs, **kwargs)
    actual = candidate(*inputs, **kwargs)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)
    sum(t.square().sum() for t in expected[:3]).backward()
    sum(t.square().sum() for t in actual[:3]).backward()
    grad = candidate.pair_bias.down_proj.weight.grad
    assert torch.isfinite(grad).all()
    assert grad.norm() > 0
    torch.testing.assert_close(grad, reference.pair_bias.down_proj.weight.grad)
    if mode in (None, "scale"):
        for block in candidate.blocks:
            assert torch.isfinite(block.geometry_scale.grad).all()
            assert block.geometry_scale.grad.norm() > 0
    resumed = _layer_bias_backbone(mode).double()
    resumed.load_state_dict(copy.deepcopy(candidate.state_dict()), strict=True)
    for a, b in zip(actual, resumed(*inputs, **kwargs), strict=True):
        torch.testing.assert_close(a, b)


def test_zero_and_negative_scales_keep_padded_attention_gradients_finite():
    # Learned scales can approach or cross zero. Padded -inf entries must
    # still mask attention without introducing NaNs in the scale gradients.
    torch.manual_seed(17)
    model = _layer_bias_backbone().double()
    with torch.no_grad():
        for block in model.blocks:
            block.adaLN_modulation[-1].bias.fill_(0.2)
            block.geometry_scale.copy_(torch.tensor([0.0, -0.5, 0.001, 1.4]))
    edges = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 4], [1, 2, 0, 2, 0, 1, 4, 3]])
    outputs = model(
        torch.randn(5, 7, dtype=torch.double),
        torch.randn(5, 3, dtype=torch.double),
        torch.randn(2, 6, dtype=torch.double),
        edges,
        torch.randn(8, 5, dtype=torch.double),
        torch.tensor([0, 0, 0, 1, 1]),
        4,
        atom_types=torch.tensor([0, 1, 2, 3, 1]),
    )
    assert all(torch.isfinite(value).all() for value in outputs[:3])
    sum(value.square().sum() for value in outputs[:3]).backward()
    for block in model.blocks:
        grad = block.geometry_scale.grad
        assert torch.isfinite(grad).all()
        assert (grad.abs() > 0).all()
