from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from rdkit import Chem
from torchmetrics import MetricCollection

import morph.utils.metrics as metric_module
from morph.dataset.core.atomic_data import batch_from_flat
from morph.model.lightning._validation import ValidationMixin
from morph.utils.metrics import init_metrics
from morph.utils.metrics.chemistry import (
    AtomStability,
    MoleculeStability,
    Novelty,
    Uniqueness,
    _posebusters_config,
    calc_posebusters_metrics,
)
from morph.utils.metrics.distribution import AtomCountKL, EdgeTypeKL
from morph.utils.metrics.tensor import (
    DIST_N_BINS,
    RG_N_BINS,
    MinDistanceViolation,
    PairwiseDistanceL1,
    RangeOverflow,
    RoGL1,
    accumulate_pairwise_distance_hist,
    accumulate_rog_hist,
    compute_batch_geometry,
    dist_edges,
    rg_edges,
)
from morph.utils.rdkit_utils import mol_is_valid


def _complete_edges(num_nodes: int, offset: int = 0) -> torch.Tensor:
    rows, cols = torch.where(~torch.eye(num_nodes, dtype=torch.bool))
    return torch.stack((rows, cols)) + offset


def test_gathered_chemistry_metrics_do_not_sync_on_compute():
    metrics, _, _, _ = init_metrics()

    assert metrics
    assert all(not metric.sync_on_compute for metric in metrics.values())


def test_explicitly_synced_tensor_metrics_do_not_sync_twice():
    distributions = SimpleNamespace(
        pairwise_distance_histogram=torch.ones(2, 2, DIST_N_BINS),
        radius_of_gyration_histogram=torch.ones(RG_N_BINS),
    )
    _, _, distribution_metrics, batch_metrics = init_metrics(
        target_n_atoms_distribution=torch.ones(8),
        atom_type_distribution=torch.ones(2),
        atom_tokens=["C", "N"],
        distributions=distributions,
    )

    assert all(not metric.sync_on_compute for metric in distribution_metrics.values())
    assert all(not metric.sync_on_compute for metric in batch_metrics.values())


def test_chemistry_evaluation_includes_posebusters_metrics(monkeypatch):
    monkeypatch.setattr(
        metric_module,
        "calc_posebusters_metrics",
        lambda mols: {
            "validity": len(mols) / 2,
            "all_pass_over_generated": len(mols) / 2,
        },
    )

    results = metric_module.calc_metrics_([None, None], {})

    assert results["posebusters/all_pass_over_generated"] == 1.0
    assert results["posebusters/validity"] == 1.0


def test_posebusters_aggregate_requires_every_check_over_all_generations(
    monkeypatch,
):
    class FakePoseBusters:
        def __init__(self, **kwargs):
            del kwargs

        def bust(self, mols, *_args):
            assert len(mols) == 3
            return pd.DataFrame(
                {
                    "check_a": [True, True, None],
                    "check_b": [True, False, True],
                }
            )

    monkeypatch.setitem(
        __import__("sys").modules,
        "posebusters",
        SimpleNamespace(PoseBusters=FakePoseBusters),
    )
    monkeypatch.setattr("morph.utils.metrics.chemistry._posebusters_config", dict)

    results = calc_posebusters_metrics(
        [object(), object(), object(), None], max_workers=1
    )

    assert results["n_all_pass"] == 1.0
    assert results["all_pass_over_busted"] == pytest.approx(1 / 3)
    assert results["all_pass_over_generated"] == 0.25
    assert results["validity"] == 0.25


def test_posebusters_defaults_to_allocation_bounded_workers(monkeypatch):
    seen = {}

    class FakePoseBusters:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def bust(self, _mols, *_args):
            return pd.DataFrame({"check": [True]})

    monkeypatch.setitem(
        __import__("sys").modules,
        "posebusters",
        SimpleNamespace(PoseBusters=FakePoseBusters),
    )
    monkeypatch.setattr("morph.utils.metrics.chemistry._posebusters_config", dict)

    calc_posebusters_metrics([object()])

    assert 1 <= seen["max_workers"] <= 8


def test_posebusters_aggregate_counts_unbuildable_molecules_as_failures():
    results = calc_posebusters_metrics([None, None])

    assert results["validity"] == 0.0
    assert results["all_pass_over_generated"] == 0.0


def test_posebusters_config_excludes_energy_and_outputless_modules(
    monkeypatch, tmp_path
):
    package = tmp_path / "posebusters"
    config_dir = package / "config"
    config_dir.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (config_dir / "mol.yml").write_text(
        """modules:
  - function: energy_ratio
    chosen_binary_test_output: [energy_ok]
  - function: distance_geometry
    chosen_binary_test_output: [geometry_ok]
  - function: flatness
    chosen_binary_test_output: [flatness_ok]
  - function: unused_expensive_module
loading: {}
""",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "posebusters",
        SimpleNamespace(__file__=str(package / "__init__.py")),
    )

    config = _posebusters_config()

    assert [module["function"] for module in config["modules"]] == ["distance_geometry"]


def test_validity_check_does_not_mutate_the_molecule():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    no_implicit = [atom.GetNoImplicit() for atom in mol.GetAtoms()]

    assert mol_is_valid(mol)
    assert [atom.GetNoImplicit() for atom in mol.GetAtoms()] == no_implicit


def test_uniqueness_and_novelty_share_canonical_smiles(monkeypatch):
    calls = 0

    def canonicalize(mol, canonical=True):
        nonlocal calls
        assert canonical
        calls += 1
        return Chem.MolToSmiles(mol)

    monkeypatch.setattr(
        "morph.utils.metrics.chemistry.morphRD.smiles_from_mol", canonicalize
    )
    mols = [Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CO")]
    uniqueness = Uniqueness()
    novelty = Novelty(["CC"], already_canonical=True)

    uniqueness.update(mols)
    novelty.update(mols)

    assert calls == len(mols)


def test_vectorized_batch_geometry_matches_per_molecule_metrics():
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ]
    )
    atom_types = torch.tensor([0, 1, 2, 1, 2])
    edge_index = torch.cat((_complete_edges(3), _complete_edges(2, 3)), dim=1)
    edge_types = torch.arange(edge_index.shape[1]) % 3
    batch = batch_from_flat(
        positions,
        atom_types,
        torch.zeros(5, dtype=torch.long),
        edge_types,
        edge_index,
        torch.tensor([0, 0, 0, 1, 1]),
        num_graphs=2,
        _already_sorted=True,
    )
    geometry = compute_batch_geometry(batch)

    expected_pair_hist = torch.zeros(3, 3, DIST_N_BINS)
    expected_rog_hist = torch.zeros(RG_N_BINS)
    for node_slice in (slice(0, 3), slice(3, 5)):
        accumulate_pairwise_distance_hist(
            expected_pair_hist,
            positions[node_slice],
            atom_types[node_slice],
            dist_edges(),
        )
        accumulate_rog_hist(expected_rog_hist, positions[node_slice], rg_edges())

    pair_metric = PairwiseDistanceL1(expected_pair_hist)
    rog_metric = RoGL1(expected_rog_hist)
    min_metric = MinDistanceViolation(threshold=0.5)
    range_metric = RangeOverflow()
    for metric in (pair_metric, rog_metric, min_metric, range_metric):
        metric.update(geometry)

    torch.testing.assert_close(pair_metric.gen_hist, expected_pair_hist)
    torch.testing.assert_close(rog_metric.gen_hist, expected_rog_hist)
    assert min_metric.violations == 1
    assert min_metric.total == 2
    assert range_metric.dist_over == 1
    assert range_metric.dist_total == 4
    assert range_metric.rog_over == 1
    assert range_metric.rog_total == 2

    atom_count_metric = AtomCountKL(torch.ones(8))
    edge_type_metric = EdgeTypeKL(torch.ones(3))
    torch.testing.assert_close(atom_count_metric._indices(batch), torch.tensor([3, 2]))
    upper = edge_index[0] < edge_index[1]
    torch.testing.assert_close(edge_type_metric._indices(batch), edge_types[upper])


def test_validation_rng_is_salted_by_distributed_rank(monkeypatch):
    monkeypatch.setenv("PL_GLOBAL_SEED", "42")
    training_rng_state = torch.random.get_rng_state()
    module = SimpleNamespace(
        global_rank=3,
        model_ema=torch.nn.Identity(),
        distribution_metrics=None,
        batch_metrics=None,
        stability_metrics=None,
        metrics=None,
    )

    ValidationMixin.on_validation_epoch_start(module)
    actual = torch.rand(4)
    torch.manual_seed(45)
    expected = torch.rand(4)

    torch.testing.assert_close(actual, expected)
    ValidationMixin._restore_training_rng(module)
    torch.testing.assert_close(torch.random.get_rng_state(), training_rng_state)


def test_validation_stability_includes_unsanitizable_generated_graphs():
    good = Chem.MolFromSmiles("CC")
    bad = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)

    class Graph:
        def __init__(self, raw, sanitized):
            self.raw, self.sanitized = raw, sanitized
            self.num_nodes = raw.GetNumAtoms()

        def to_rdkit_mol(self, *args, sanitize=True):
            return self.sanitized if sanitize else self.raw

    graphs = [Graph(good, good), Graph(bad, None)]
    batch = SimpleNamespace(to_data_list=lambda: graphs)
    batch.to_inplace = lambda *args, **kwargs: batch
    stability = MetricCollection(
        {"atom": AtomStability(), "molecule": MoleculeStability()}, compute_groups=False
    )
    module = SimpleNamespace(
        _validation_context=lambda batch: None,
        sample=lambda *args, **kwargs: batch,
        batch_metrics=None,
        distribution_metrics=None,
        representation=SimpleNamespace(requires_topology=True),
        vocab=SimpleNamespace(atom_tokens=[], edge_tokens=[], charge_tokens=[]),
        _update_task_metrics=lambda *args: None,
        _validation_rdkit_mols=[],
        stability_metrics=stability,
    )
    ValidationMixin.validation_step(module, batch, 0)
    result = stability.compute()
    assert float(result["atom"]) == 7 / 8
    assert float(result["molecule"]) == 0.5
    assert module._validation_rdkit_mols == [good, None]
