"""Regression tests for Morph's native graph containers."""

from types import SimpleNamespace

import pytest
import torch
from rdkit import Chem

from morph.dataset.core.atomic_data import (
    ChemAtomicData,
    ChemBatch,
    batch_from_flat,
    filter_nodes,
)
from morph.dataset.core.lightning_datamodule import LightningDataModule


def _path_graph(num_nodes: int, offset: float = 0.0) -> ChemAtomicData:
    src = torch.arange(num_nodes - 1)
    dst = torch.arange(1, num_nodes)
    edge_index = torch.stack([src, dst])
    graph = ChemAtomicData(
        positions=torch.arange(num_nodes * 3, dtype=torch.float32).view(-1, 3) + offset,
        atom_types=torch.arange(num_nodes),
        charge_types=torch.zeros(num_nodes, dtype=torch.long),
        edge_types=torch.ones(num_nodes - 1, dtype=torch.long),
        neighbor_list=edge_index.t().contiguous(),
        smiles=f"graph-{num_nodes}",
    )
    graph.add_node_property("keep_mask", torch.ones(num_nodes, dtype=torch.long))
    graph.add_edge_property("lambda_e_sub", torch.arange(num_nodes - 1).float())
    graph.add_system_property("y", torch.tensor([[float(num_nodes)]]))
    return graph


def test_graph_batch_roundtrip_preserves_levels_and_local_edges():
    graphs = [_path_graph(3), _path_graph(2, offset=20.0)]
    batch = ChemBatch.from_data_list(graphs)

    assert batch.keys["node"] >= {
        "positions",
        "atom_types",
        "charge_types",
        "keep_mask",
    }
    assert batch.keys["edge"] >= {"edge_index", "edge_types", "lambda_e_sub"}
    assert batch.keys["system"] == {"y"}
    assert batch.batch.tolist() == [0, 0, 0, 1, 1]
    assert batch.edge_index.tolist() == [[0, 1, 3], [1, 2, 4]]

    restored = batch.to_data_list()
    assert [graph.num_nodes for graph in restored] == [3, 2]
    assert [graph.edge_index.tolist() for graph in restored] == [
        [[0, 1], [1, 2]],
        [[0], [1]],
    ]
    assert restored[1].smiles == "graph-2"
    torch.testing.assert_close(restored[0].y, torch.tensor([[3.0]]))
    torch.testing.assert_close(restored[1].lambda_e_sub, torch.tensor([0.0]))


def test_batch_from_flat_groups_nodes_and_remaps_edges():
    batch = batch_from_flat(
        positions=torch.arange(4, dtype=torch.float32).unsqueeze(1).expand(-1, 3),
        atom_types=torch.tensor([10, 20, 11, 21]),
        charge_types=torch.zeros(4, dtype=torch.long),
        edge_types=torch.tensor([1, 2, 3, 4]),
        edge_index=torch.tensor([[0, 2, 1, 3], [2, 0, 3, 1]]),
        batch_idx=torch.tensor([1, 0, 1, 0]),
        num_graphs=2,
    )

    assert batch.atom_types.tolist() == [20, 21, 10, 11]
    assert batch.batch.tolist() == [0, 0, 1, 1]
    assert batch.edge_index.tolist() == [[0, 1, 2, 3], [1, 0, 3, 2]]
    assert batch.edge_types.tolist() == [3, 4, 1, 2]


def test_batch_from_flat_keeps_already_sorted_storage():
    positions = torch.arange(4, dtype=torch.float32).unsqueeze(1).expand(-1, 3)
    atom_types = torch.tensor([10, 11, 20, 21])
    edge_types = torch.tensor([1, 2])
    edge_index = torch.tensor([[0, 2], [1, 3]])
    batch_idx = torch.tensor([0, 0, 1, 1])

    batch = batch_from_flat(
        positions=positions,
        atom_types=atom_types,
        charge_types=None,
        edge_types=edge_types,
        edge_index=edge_index,
        batch_idx=batch_idx,
        num_graphs=2,
        _already_sorted=True,
    )

    assert batch.positions is positions
    assert batch.atom_types is atom_types
    assert batch.edge_types is edge_types
    assert batch.edge_index is edge_index
    assert batch.batch is batch_idx


def test_batched_filter_preserves_empty_graphs_and_registered_properties():
    batch = ChemBatch.from_data_list([_path_graph(2), _path_graph(3)])
    filtered = filter_nodes(batch, torch.tensor([False, False, True, False, True]))

    assert filtered.num_graphs == 2
    assert filtered.num_nodes_per_graph.tolist() == [0, 2]
    assert filtered.num_edges_per_graph.tolist() == [0, 0]
    assert filtered.keep_mask.tolist() == [1, 1]
    assert filtered.y.tolist() == [[2.0], [3.0]]
    assert [graph.num_nodes for graph in filtered.to_data_list()] == [0, 2]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rebatching_sparse_edges_preserves_empty_graph_counts(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    batch = batch_from_flat(
        positions=torch.arange(18, dtype=torch.float64, device=device).reshape(6, 3),
        atom_types=torch.arange(6, device=device),
        charge_types=None,
        edge_types=torch.tensor([1, 2, 3, 4, 5], device=device),
        edge_index=torch.tensor([[0, 2, 1, 4, 3], [2, 0, 4, 1, 3]], device=device),
        batch_idx=torch.tensor([3, 0, 3, 1, 0, 3], device=device),
        num_graphs=5,
    )
    assert batch.num_nodes_per_graph.tolist() == [2, 1, 0, 3, 0]
    assert batch.num_edges_per_graph.tolist() == [2, 1, 0, 2, 0]
    assert batch.a.tolist() == [1, 4, 3, 0, 2, 5]
    assert batch.e.tolist() == [3, 4, 5, 1, 2]
    selected = filter_nodes(
        batch, torch.tensor([False, True, False, True, True, False], device=device)
    )
    assert selected.num_nodes_per_graph.tolist() == [1, 0, 0, 2, 0]
    assert selected.num_edges_per_graph.tolist() == [0, 0, 0, 2, 0]
    torch.testing.assert_close(
        selected.remove_com(),
        torch.tensor(
            [[12, 13, 14], [0, 0, 0], [0, 0, 0], [3, 4, 5], [0, 0, 0]],
            dtype=torch.float64,
            device=device,
        ),
        rtol=0,
        atol=0,
    )
    empty = filter_nodes(batch, torch.zeros(6, dtype=torch.bool, device=device))
    assert empty.num_nodes_per_graph.tolist() == [0] * 5
    assert empty.num_edges_per_graph.tolist() == [0] * 5


def test_batch_clone_is_independent_and_has_no_atomistic_placeholders():
    batch = ChemBatch.from_data_list([_path_graph(2)])
    clone = batch.clone()
    clone.positions.add_(10)

    assert not torch.equal(batch.positions, clone.positions)
    assert "atomic_numbers" not in batch
    assert "atomic_masses" not in batch


def test_rdkit_conversion_forwards_sanitize(monkeypatch):
    seen = {}

    def convert(*_args, sanitize=True, implicit_hydrogens=False):
        seen["sanitize"] = sanitize
        return object()

    monkeypatch.setattr("morph.dataset.core.containers.tensors_to_rdkit_mol", convert)

    result = _path_graph(2).to_rdkit_mol(
        atom_tokens=["C", "N"],
        edge_tokens=["<NO_BOND>", "1"],
        charge_tokens=["0"],
        sanitize=False,
    )

    assert result is not None
    assert seen["sanitize"] is False


@pytest.mark.parametrize("no_bond", [0, 2])
def test_rdkit_conversion_keeps_bond_order_with_reordered_vocabulary(
    monkeypatch, no_bond
):

    tokens = ["1", "2", "<NO_BOND>"]
    tokens[no_bond], tokens[2] = tokens[2], tokens[no_bond]
    graph = _path_graph(3)
    graph.neighbor_list = torch.tensor([[0, 1], [0, 2], [1, 2]])
    graph.edge_types = torch.tensor([tokens.index("2"), no_bond, tokens.index("1")])
    seen = {}

    def convert(
        atoms, positions, charges, bonds, edges, *, sanitize, implicit_hydrogens
    ):
        seen.update(atoms=atoms, bonds=bonds, edges=edges, sanitize=sanitize)
        return object()

    monkeypatch.setattr("morph.dataset.core.containers.tensors_to_rdkit_mol", convert)
    graph.to_rdkit_mol(["C", "C", "O"], tokens, ["0"], sanitize=False)
    assert seen == {
        "atoms": ["C", "C", "O"],
        "bonds": [Chem.BondType.DOUBLE, Chem.BondType.SINGLE],
        "edges": [[0, 1], [1, 2]],
        "sanitize": False,
    }


def test_rdkit_conversion_preserves_strided_float32_coordinates():
    graph = _path_graph(2)
    graph.positions = torch.arange(12, dtype=torch.float32).view(2, 6)[:, ::2] / 7
    graph.positions = graph.positions.T.contiguous().T
    assert not graph.positions.is_contiguous()
    mol = graph.to_rdkit_mol(["C", "N"], ["<NO_BOND>", "1"], ["0"], sanitize=False)
    torch.testing.assert_close(
        torch.from_numpy(mol.GetConformer().GetPositions()),
        graph.positions.double(),
        rtol=0,
        atol=0,
    )


def test_remove_com_translates_coframed_context_with_each_graph():
    ligand = ChemBatch.from_data_list(
        [_path_graph(2, offset=10.0), _path_graph(3, offset=30.0)]
    )
    pocket = ChemBatch.from_data_list(
        [_path_graph(3, offset=20.0), _path_graph(2, offset=40.0)]
    )
    ligand.contexts["pocket"] = pocket
    pocket_before = pocket.positions.clone()

    removed = ligand.remove_com()

    torch.testing.assert_close(
        ligand.positions.new_zeros((ligand.num_graphs, 3)),
        torch.stack([ligand.positions[ligand.batch == i].mean(0) for i in range(2)]),
    )
    torch.testing.assert_close(
        pocket.positions,
        pocket_before - removed[pocket.batch],
    )


def test_datamodule_setup_is_stage_specific_and_idempotent():
    datamodule = LightningDataModule.__new__(LightningDataModule)
    datamodule.vocab = object()
    datamodule.datasets = SimpleNamespace(
        train="train-config", val=["val-config"], test=["test-config"]
    )
    datamodule.train_dataset = None
    datamodule.val_datasets = None
    datamodule.test_datasets = None
    calls = []

    def wrap(config, stage):
        calls.append((config, stage))
        return f"{stage}-dataset"

    datamodule._wrap_dataset = wrap

    datamodule.setup("fit")
    datamodule.setup("fit")
    assert calls == [("train-config", "train"), ("val-config", "val")]
    assert datamodule.test_datasets is None

    datamodule.setup("predict")
    datamodule.setup("predict")
    assert calls[-1] == ("test-config", "test")
    assert len(calls) == 3
