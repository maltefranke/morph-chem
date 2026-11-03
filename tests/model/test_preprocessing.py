import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from morph.dataset.core.preprocessing import Preprocessing
from morph.dataset.core.vocab import Vocab
from morph.dataset.unconditional.qm9 import FlowMatchingQM9Dataset, RevisedQM9
from morph.utils.metrics.tensor import (
    DIST_N_BINS,
    RG_N_BINS,
    accumulate_pairwise_distance_hist,
    accumulate_rog_hist,
    dist_edges,
    rg_edges,
)
from morph.utils.vocab_utils import z_to_atom_types


class _RawDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def get(self, index):
        return self.items[index]


def test_preprocessing_artifacts_accept_paths_in_current_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    preprocessing = object.__new__(Preprocessing)
    preprocessing.distributions_path = "distributions.pt"
    tokens = ["C", "H"]
    distributions = {"atom_type_distribution": torch.tensor([0.25, 0.75])}

    preprocessing._save_tokens("atom_tokens.txt", tokens)
    preprocessing._save_distributions(distributions)

    assert preprocessing._load_tokens("atom_tokens.txt") == tokens
    torch.testing.assert_close(preprocessing._load_distributions(), distributions)
    assert {path.name for path in tmp_path.iterdir()} == {
        "atom_tokens.txt",
        "distributions.pt",
    }


def test_qm9_processed_cache_loads_without_raw_download(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    molecule = Data(z=torch.tensor([6]), pos=torch.zeros(1, 3))
    for split in ("train", "val", "test"):
        RevisedQM9.save([molecule], str(processed / f"{split}.pt"))

    def unexpected_download(self):
        pytest.fail("A complete processed QM9 cache must load without raw data")

    monkeypatch.setattr(RevisedQM9, "download", unexpected_download)
    dataset = FlowMatchingQM9Dataset(
        root=str(tmp_path), vocab=None, distributions=None, split="test"
    )
    assert len(dataset) == 1
    assert dataset.get(0).z.tolist() == [6]
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize("force_reload", [False, True])
def test_qm9_downloads_raw_inputs_when_processing_is_needed(
    tmp_path, monkeypatch, force_reload
):
    processed = tmp_path / "processed"
    processed.mkdir()
    molecule = Data(z=torch.tensor([6]), pos=torch.zeros(1, 3))
    splits = ("train", "val", "test") if force_reload else ("train", "val")
    for split in splits:
        RevisedQM9.save([molecule], str(processed / f"{split}.pt"))

    def requested_download(self):
        raise RuntimeError("Raw inputs requested")

    monkeypatch.setattr(RevisedQM9, "download", requested_download)
    with pytest.raises(RuntimeError, match="Raw inputs requested"):
        RevisedQM9(root=str(tmp_path), force_reload=force_reload)


def _legacy_statistics(dataset, vocab):
    n_atoms = []
    atoms = []
    edges = []
    charges = []
    coords = []
    pair_hist = torch.zeros(len(vocab.atom_tokens), len(vocab.atom_tokens), DIST_N_BINS)
    rog_hist = torch.zeros(RG_N_BINS)

    for data in dataset.items:
        n = data.num_nodes
        atom_idx = torch.tensor(
            [vocab.atom_tokens.index(t) for t in z_to_atom_types(data.z.tolist())]
        )
        atoms.append(atom_idx)
        dense = to_dense_adj(
            data.edge_index, edge_attr=data.edge_attr, max_num_nodes=n
        )[0]
        triu = torch.triu_indices(n, n, offset=1)
        edges.append(dense[triu[0], triu[1]].long())
        coord = data.pos - data.pos.mean(dim=0)
        coords.append(coord)
        accumulate_pairwise_distance_hist(pair_hist, coord, atom_idx, dist_edges())
        accumulate_rog_hist(rog_hist, coord, rg_edges())
        charges.append(
            torch.tensor(
                [vocab.charge_tokens.index(str(c)) for c in data.charges.tolist()]
            )
        )
        n_atoms.append(n)

    def _distribution(values, minlength):
        counts = torch.cat(values).bincount(minlength=minlength).float()
        return counts / counts.sum()

    n_atom_counts = torch.tensor(n_atoms).bincount().float()
    return {
        "n_atoms_distribution": n_atom_counts / n_atom_counts.sum(),
        "atom_type_distribution": _distribution(atoms, len(vocab.atom_tokens)),
        "edge_type_distribution": _distribution(edges, len(vocab.edge_tokens)),
        "charge_type_distribution": _distribution(charges, len(vocab.charge_tokens)),
        "coordinate_std": torch.cat(coords).std().item(),
        "pairwise_distance_histogram": pair_hist,
        "radius_of_gyration_histogram": rog_hist,
    }


@pytest.mark.parametrize("workers", [0, 2])
def test_streaming_statistics_match_legacy_concatenation_and_dense_adjacency(workers):
    items = [
        Data(
            z=torch.tensor([6, 1, 8]),
            pos=torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.4, 0.2]]),
            charges=torch.tensor([0, 0, -1]),
            edge_index=torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]]),
            edge_attr=torch.tensor([1, 1, 2, 2]),
        ),
        Data(
            z=torch.tensor([7, 6, 1, 1]),
            pos=torch.tensor(
                [[-0.2, 0.1, 0.3], [1.0, 0.2, -0.1], [1.5, 1.0, 0.0], [1.4, -0.8, 0.2]]
            ),
            charges=torch.tensor([1, 0, 0, 0]),
            edge_index=torch.tensor([[0, 1, 1, 2, 1, 3], [1, 0, 2, 1, 3, 1]]),
            edge_attr=torch.tensor([2, 2, 1, 1, 1, 1]),
        ),
    ]
    dataset = _RawDataset(items)
    vocab = Vocab(
        atom_tokens=["H", "C", "N", "O"],
        edge_tokens=["<NO_BOND>", "1", "2"],
        charge_tokens=["-1", "0", "1"],
    )
    preprocessing = object.__new__(Preprocessing)
    preprocessing.train_dataset = dataset
    preprocessing.vocab = vocab
    preprocessing.num_workers = workers

    actual = preprocessing._compute_distributions()
    expected = _legacy_statistics(dataset, vocab)

    for key, expected_value in expected.items():
        if isinstance(expected_value, torch.Tensor):
            torch.testing.assert_close(actual[key], expected_value)
        else:
            assert actual[key] == expected_value


@pytest.mark.parametrize("workers", [0, 2])
def test_parallel_preprocessing_without_topology_or_charges(workers):
    dataset = _RawDataset(
        [
            Data(
                z=torch.tensor([6, 1]),
                pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            ),
            Data(z=torch.tensor([8]), pos=torch.tensor([[2.0, 0.0, 0.0]])),
            Data(
                z=torch.tensor([6, 8, 1]),
                pos=torch.tensor([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]]),
            ),
        ]
    )
    preprocessing = object.__new__(Preprocessing)
    preprocessing.train_dataset = dataset
    preprocessing.vocab = None
    preprocessing.num_workers = workers
    atoms, edges, charges = preprocessing._compute_tokens_from_data()
    assert atoms == ["C", "H", "O"]
    assert edges == ["<NO_BOND>"]
    assert charges == ["0"]
    preprocessing.vocab = Vocab(
        atom_tokens=atoms, edge_tokens=edges, charge_tokens=charges
    )
    actual = preprocessing._compute_distributions()
    preprocessing.num_workers = 0
    expected = preprocessing._compute_distributions()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
    assert actual["edge_type_distribution"].tolist() == [1.0]
    assert actual["charge_type_distribution"].tolist() == [1.0]
    assert actual["pairwise_distance_histogram"].sum() == 4
    assert actual["radius_of_gyration_histogram"].sum() == 3


def test_parallel_statistics_reject_empty_dataset():
    preprocessing = object.__new__(Preprocessing)
    preprocessing.train_dataset = _RawDataset([])
    preprocessing.vocab = Vocab(
        atom_tokens=["H"], edge_tokens=["<NO_BOND>"], charge_tokens=["0"]
    )
    preprocessing.num_workers = 2
    with pytest.raises(ValueError, match="empty dataset"):
        preprocessing._compute_distributions()


def test_statistics_count_known_absent_bonds_alongside_present_bonds():
    dataset = _RawDataset(
        [
            Data(
                z=torch.tensor([6, 6]),
                pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                charges=torch.zeros(2, dtype=torch.long),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long),
            ),
            Data(
                z=torch.tensor([6, 6]),
                pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                charges=torch.zeros(2, dtype=torch.long),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                edge_attr=torch.ones(2, dtype=torch.long),
            ),
        ]
    )
    preprocessing = object.__new__(Preprocessing)
    preprocessing.train_dataset = dataset
    preprocessing.vocab = Vocab(
        atom_tokens=["C"], edge_tokens=["<NO_BOND>", "1"], charge_tokens=["0"]
    )
    preprocessing.num_workers = 0
    stats = preprocessing._compute_distributions()
    torch.testing.assert_close(
        stats["edge_type_distribution"], torch.tensor([0.5, 0.5])
    )
