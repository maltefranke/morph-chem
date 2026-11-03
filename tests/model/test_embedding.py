import copy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from morph.model.backbones.dit import DiTEmbedding
from morph.model.layers.embedding import Embedding


@pytest.mark.parametrize("dropout,training", [(0.0, True), (0.25, True), (0.25, False)])
@pytest.mark.parametrize("vocab", [5, 65])
def test_category_projection_preserves_outputs_gradients_and_checkpoint(
    dropout, training, vocab
):
    torch.manual_seed(42)
    layer = Embedding(vocab, 16, out_dim=24, dropout=dropout).double().train(training)
    reference = copy.deepcopy(layer)
    indices = torch.tensor([[0, 0, 0, 1], [0, 2, 1, 4]], dtype=torch.int32)
    probe = torch.randn(2, 4, 24, dtype=torch.float64)
    torch.manual_seed(12)
    expected = reference.projection(reference.emb(indices))
    (expected * probe).sum().backward()
    torch.manual_seed(12)
    actual = layer(indices)
    (actual * probe).sum().backward()
    torch.testing.assert_close(actual, expected)
    for actual_param, expected_param in zip(
        layer.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_param.grad, expected_param.grad)
    reference.load_state_dict(layer.state_dict(), strict=True)


@pytest.mark.parametrize("training", [False, True])
def test_category_inference_preserves_dropout_and_rng(training):
    layer = Embedding(5, 16, out_dim=24, dropout=0.25).double().train(training)
    indices = torch.tensor([[0, 0, 1], [2, 4, 3]], dtype=torch.int32)
    with torch.no_grad():
        torch.manual_seed(12)
        expected = layer.projection(layer.emb(indices))
        rng = torch.get_rng_state()
        torch.manual_seed(12)
        actual = layer(indices)
        torch.testing.assert_close(actual, expected)
        assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_category_inference_matches_dense_lookup(dtype):
    layer = Embedding(5, 64, out_dim=128).cuda().eval()

    def dense_lookup(indices):
        table = layer.projection(layer.emb.weight)
        return torch.nn.functional.one_hot(indices.long(), 5).to(table.dtype) @ table

    reference = torch.compile(dense_lookup, fullgraph=True, dynamic=True)
    candidate = torch.compile(layer, fullgraph=True, dynamic=True)
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32),
    ):
        for length in (1, 31, 1025, 0):
            indices = torch.randint(5, (length,), device="cuda", dtype=torch.int32)
            expected, actual = reference(indices), candidate(indices)
            assert actual.dtype == expected.dtype
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(
                actual,
                expected,
                atol=0.03 if dtype == torch.bfloat16 else 2e-6,
                rtol=0.01 if dtype == torch.bfloat16 else 2e-5,
            )
            error = (actual.float() - expected.float()).norm() / (
                expected.float().norm().clamp_min(1e-8)
            )
            assert error < (0.01 if dtype == torch.bfloat16 else 2e-5)


def _incident_embedding(enabled=True):
    token = {
        "_target_": "morph.model.layers.embedding.Embedding",
        "num_embeddings": 4,
        "embedding_dim": 8,
        "out_dim": 12,
    }
    return DiTEmbedding(
        OmegaConf.create(token),
        OmegaConf.create(token),
        OmegaConf.create(
            {
                "_target_": "morph.model.layers.embedding.TimeEmbedding",
                "embedding_dim": 8,
                "out_dim": 12,
            }
        ),
        OmegaConf.create(
            {
                "_target_": "morph.model.layers.embedding.ExtrapolatableCountEmbedding",
                "embedding_dim": 8,
                "out_dim": 12,
            }
        ),
        incident_bond_embedding=enabled,
    ).double()


def _incident_graph():
    # Two graphs: a three-atom chain and an isolated atom. NO_BOND pairs
    # deliberately remain in the runtime topology, as in training/sampling.
    return SimpleNamespace(
        a=torch.tensor([0, 1, 2, 3]),
        e=torch.tensor([1, 0, 1, 2, 0, 2]),
        edge_index=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
        num_nodes_per_graph=torch.tensor([3, 1]),
    )


def test_incident_bond_embedding_starts_at_baseline_and_learns():
    reference = _incident_embedding(False)
    candidate = _incident_embedding()
    missing, unexpected = candidate.load_state_dict(
        reference.state_dict(), strict=False
    )
    assert missing == ["incident_bond_projection.weight"]
    assert not unexpected
    graph, t = _incident_graph(), torch.tensor([[0.5], [0.9]], dtype=torch.float64)
    expected = reference(graph, t)
    actual = candidate(graph, t)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    actual[0].sum().backward()
    grad = candidate.incident_bond_projection.weight.grad
    edge_vectors = candidate.edge_type_embedding(torch.tensor([1, 2])).detach()
    expected_grad = (2 * edge_vectors.sum(dim=0)).expand(12, -1)
    torch.testing.assert_close(grad, expected_grad)
    assert grad.abs().max() > 0


def test_incident_bond_embedding_counts_once_and_is_permutation_equivariant():
    candidate = _incident_embedding()
    with torch.no_grad():
        candidate.incident_bond_projection.weight.copy_(
            torch.arange(144, dtype=torch.float64).reshape(12, 12) / 144
        )
    graph, t = _incident_graph(), torch.tensor([[0.5], [0.9]], dtype=torch.float64)
    actual = candidate(graph, t)[0] - candidate.atom_type_embedding(graph.a)
    single, double = candidate.edge_type_embedding(torch.tensor([1, 2]))
    pooled = torch.stack([single, single + double, double, torch.zeros(12)])
    expected = candidate.incident_bond_projection(pooled)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert candidate.edge_type_embedding.emb.weight.grad.abs().max() > 0

    order = torch.tensor([2, 0, 1, 3])
    inverse = torch.argsort(order)
    permuted = SimpleNamespace(
        a=graph.a[order],
        e=graph.e.flip(0),
        edge_index=inverse[graph.edge_index.flip(1)],
        num_nodes_per_graph=graph.num_nodes_per_graph,
    )
    torch.testing.assert_close(candidate(permuted, t)[0], candidate(graph, t)[0][order])


def test_incident_bond_embedding_compiles_and_handles_no_bonds():
    candidate = _incident_embedding()
    with torch.no_grad():
        candidate.incident_bond_projection.weight.copy_(torch.eye(12))
    graph, t = _incident_graph(), torch.tensor([[0.5], [0.9]], dtype=torch.float64)
    compiled = torch.compile(candidate, backend="eager", fullgraph=True, dynamic=True)
    for left, right in zip(compiled(graph, t), candidate(graph, t), strict=True):
        torch.testing.assert_close(left, right)
    graph.e.zero_()
    torch.testing.assert_close(
        compiled(graph, t)[0], candidate.atom_type_embedding(graph.a)
    )
