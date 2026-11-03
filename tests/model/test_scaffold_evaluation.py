"""Scaffold panel sampling and conditional evaluation contracts."""

import json
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, local
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from rdkit import Chem

from morph.dataset.conditional.scaffold import panel as panel_module
from morph.dataset.conditional.scaffold.cutters import scaffold_indices
from morph.dataset.conditional.scaffold.datamodule import ScaffoldLightningDataModule
from morph.dataset.conditional.scaffold.panel import (
    ScaffoldEvaluationDataset,
    build_panel,
    panel_metadata,
    panel_path,
    scaffold_smiles,
    write_panel,
)
from morph.dataset.core.atomic_data import ChemAtomicData, chembatch_from_graphs
from morph.dataset.core.conversion import mol_to_data
from morph.dataset.core.flow_matching_wrapper import eval_collate_fn
from morph.dataset.core.vocab import Vocab
from morph.dataset.unconditional.qm9 import FlowMatchingQM9Dataset, RevisedQM9
from morph.model.finetune.scaffold.scaffold import LightningModuleRatesScaffoldFinetune
from morph.model.lightning._validation import ValidationMixin
from morph.utils.metrics.scaffold import scaffold_row, summarize_scaffold_rows

VOCAB = Vocab(
    atom_tokens=["H", "C", "N", "O"],
    edge_tokens=["<NO_BOND>", "1", "2", "3", "4"],
    charge_tokens=["-1", "0", "1"],
)


class Molecules:
    """Conformers with distinguishable coordinates and permuted atom order."""

    split = "val"
    hydrogen_mode = "explicit"

    def __init__(self):
        self.mols = []
        for i, smi in enumerate(
            [
                "Cc1ccccc1",
                "Cc1ccccc1",
                "Cc1ccccc1",
                "CCc1ccccc1",
                "Cc1ccncc1",
                "CC1CCCCC1",
                "CC1CCOC1",
            ]
        ):
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
            conformer = Chem.Conformer(mol.GetNumAtoms())
            for j in range(mol.GetNumAtoms()):
                conformer.SetAtomPosition(j, (i * 10 + j, j % 2, j % 3))
            mol.AddConformer(conformer)
            self.mols.append(mol)

    def __len__(self):
        """Return the conformer count."""
        return len(self.mols)

    def get(self, index):
        """Return stored raw fields."""
        return mol_to_data(Chem.Mol(self.mols[index]))

    def __getitem__(self, index):
        """Return tokenized coordinates."""
        return ChemAtomicData.from_rdkit(self.mols[index], VOCAB)


def _panel(tmp_path, n=4, seed=42, repeats=8):
    dataset = Molecules()
    cache = SimpleNamespace(filtered_indices=lambda: range(len(dataset)))
    panel = build_panel(dataset, cache, "murcko", n_scaffolds=n, seed=seed)
    path = panel_path(tmp_path, "val", "murcko", n, seed)
    write_panel(panel, path)
    wrapped = ScaffoldEvaluationDataset(
        dataset,
        scaffold_groups_dir=tmp_path,
        scaffold_kind="murcko",
        directory=tmp_path,
        n_scaffolds=n,
        seed=seed,
        samples_per_scaffold=repeats,
    )
    return dataset, cache, panel, wrapped


def test_qm9_panel_is_reused_after_downloading_to_another_directory(
    tmp_path, monkeypatch
):
    root = tmp_path / "original"
    processed = root / "processed"
    processed.mkdir(parents=True)
    molecules = Molecules()
    records = [molecules.get(i) for i in range(len(molecules))]
    for split in ("train", "val", "test"):
        RevisedQM9.save(records, str(processed / f"{split}.pt"))
    dataset = FlowMatchingQM9Dataset(
        str(root), vocab=None, distributions=None, split="val"
    )
    options = dict(scaffold_kind="murcko", n_scaffolds=4)
    original = ScaffoldEvaluationDataset(
        dataset, scaffold_groups_dir=processed / "scaffolds", **options
    )
    download = tmp_path / "download"
    shutil.copytree(root, download, copy_function=shutil.copyfile)
    downloaded = FlowMatchingQM9Dataset(
        str(download), vocab=None, distributions=None, split="val"
    )
    monkeypatch.setattr(
        panel_module, "build_panel", lambda *_args, **_kw: pytest.fail("rebuilt panel")
    )
    reused = ScaffoldEvaluationDataset(
        downloaded, scaffold_groups_dir=download / "processed/scaffolds", **options
    )
    assert reused.entries == original.entries
    assert reused.path.name.startswith("val_")


def test_panel_deduplicates_scaffolds_and_preserves_stored_atom_order(tmp_path):
    state = random.getstate()
    dataset, cache, panel, wrapped = _panel(tmp_path)
    assert random.getstate() == state
    assert panel == build_panel(dataset, cache, "murcko", n_scaffolds=4, seed=42)
    assert panel["candidate_molecules"] == 5
    assert panel["candidate_scaffolds"] == 4
    assert len(wrapped) == 32
    for i, entry in enumerate(panel["entries"]):
        assert (
            scaffold_smiles(dataset.mols[entry["base_idx"]], entry["atom_indices"])
            == entry["scaffold_smiles"]
        )
        source, target = wrapped[8 * i]
        torch.testing.assert_close(source.x, target.x[entry["atom_indices"]])
        for j in range(8):
            repeat, _ = wrapped[8 * i + j]
            torch.testing.assert_close(source.x, repeat.x)
            assert repeat.scaffold_id.item() == i
            assert repeat.sample_id.item() == 8 * i + j
    sources, _ = eval_collate_fn([wrapped[i] for i in (0, 7, 8)])
    assert sources.scaffold_id.tolist() == [0, 0, 1]
    assert sources.sample_id.tolist() == [0, 7, 8]


def test_panel_rejects_stale_or_undersized_cache_and_can_force_rebuild(tmp_path):
    dataset, cache, panel, _ = _panel(tmp_path)
    with pytest.raises(ValueError, match="Requested 5 scaffolds"):
        build_panel(dataset, cache, "murcko", n_scaffolds=5)
    path = panel_path(tmp_path, "val", "murcko", 4)
    write_panel({**panel, "metadata": {**panel["metadata"], "records": 999}}, path)
    kwargs = dict(
        scaffold_groups_dir=tmp_path,
        scaffold_kind="murcko",
        directory=tmp_path,
        n_scaffolds=4,
    )
    with pytest.raises(ValueError, match=r"preprocess_scaffold\.py --force"):
        ScaffoldEvaluationDataset(dataset, **kwargs)
    wrapped = ScaffoldEvaluationDataset(dataset, **kwargs, force=True)
    assert json.loads(path.read_text()) == panel
    assert len(wrapped) == 32


def test_first_instantiation_prepares_missing_caches_and_second_reuses_them(
    tmp_path, monkeypatch
):
    dataset = Molecules()
    kwargs = dict(scaffold_groups_dir=tmp_path, scaffold_kind="murcko", n_scaffolds=4)
    wrapped = ScaffoldEvaluationDataset(dataset, **kwargs)
    assert wrapped.path.is_file()
    assert len(list(tmp_path.glob("scaffold_val_*.lmdb"))) == 1
    assert len(wrapped) == 32
    original = wrapped.path.read_bytes()

    def unexpected_build(*args, **kwargs):
        pytest.fail("Existing panels must be reused without recomputing cuts")

    monkeypatch.setattr(panel_module, "build_panel", unexpected_build)
    monkeypatch.setattr(panel_module, "load_scaffold_cache", unexpected_build)
    reused = ScaffoldEvaluationDataset(dataset, **kwargs)
    assert reused.entries == wrapped.entries
    assert reused.path.read_bytes() == original


def test_concurrent_first_instantiation_builds_panel_once(tmp_path, monkeypatch):
    barrier, state = Barrier(2), local()
    read_panel, build = panel_module._read_panel, panel_module.build_panel
    builds = []

    def concurrent_read(path, metadata):
        result = read_panel(path, metadata)
        if not getattr(state, "started", False):
            state.started = True
            barrier.wait(timeout=10)
        return result

    def counted_build(*args, **kwargs):
        builds.append(1)
        return build(*args, **kwargs)

    def instantiate(_):
        return ScaffoldEvaluationDataset(
            Molecules(),
            scaffold_groups_dir=tmp_path,
            scaffold_kind="murcko",
            n_scaffolds=4,
        )

    monkeypatch.setattr(panel_module, "_read_panel", concurrent_read)
    monkeypatch.setattr(panel_module, "build_panel", counted_build)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(instantiate, range(2))
    assert builds == [1]
    assert first.entries == second.entries


def test_failed_preparation_publishes_no_panel_and_can_retry(tmp_path, monkeypatch):
    kwargs = dict(scaffold_groups_dir=tmp_path, scaffold_kind="murcko", n_scaffolds=4)
    build = panel_module.build_panel

    def interrupted_build(*args, **kwargs):
        raise RuntimeError("interrupted preprocessing")

    monkeypatch.setattr(panel_module, "build_panel", interrupted_build)
    with pytest.raises(RuntimeError, match="interrupted preprocessing"):
        ScaffoldEvaluationDataset(Molecules(), **kwargs)
    assert not list(tmp_path.rglob("*.json"))
    monkeypatch.setattr(panel_module, "build_panel", build)
    assert len(ScaffoldEvaluationDataset(Molecules(), **kwargs)) == 32


def test_random_conformer_is_not_always_the_first(tmp_path):
    dataset = Molecules()
    dataset.mols = dataset.mols[:3]
    cache = SimpleNamespace(filtered_indices=lambda: range(len(dataset)))
    chosen = {
        build_panel(dataset, cache, "murcko", n_scaffolds=1, seed=seed)["entries"][0][
            "base_idx"
        ]
        for seed in range(12)
    }
    assert chosen == {0, 1, 2}


def test_scaffold_identity_handles_cut_aromatic_nitrogen():
    mol = Chem.AddHs(Chem.MolFromSmiles("C#CCCCc1nc2ccccc2c(=O)n1N"))
    cut = scaffold_indices(mol, "murcko")[0]
    smi = scaffold_smiles(mol, cut)
    assert Chem.MolFromSmiles(smi) is not None
    perm = list(reversed(range(mol.GetNumAtoms())))
    reordered = Chem.RenumberAtoms(mol, perm)
    assert scaffold_smiles(reordered, [perm.index(i) for i in cut]) == smi


def _attempt(wrapped, index=0):
    source, target = wrapped[index]
    generated = target.clone()
    mask = torch.zeros(target.num_nodes, dtype=torch.long)
    mask[wrapped.entries[index // wrapped.samples_per_scaffold]["atom_indices"]] = 1
    generated.add_node_property("keep_mask", mask)
    mol = generated.to_rdkit_mol(
        VOCAB.atom_tokens, VOCAB.edge_tokens, VOCAB.charge_tokens
    )
    return source, target, generated, mol


def test_scores_detect_growth_recovery_graph_changes_and_coordinate_drift(tmp_path):
    *_, wrapped = _panel(tmp_path)
    source, target, generated, mol = _attempt(wrapped)
    row = scaffold_row(source, target, generated, mol, VOCAB)
    assert row["validity"]
    assert row["retention"]
    assert row["recovered"]
    assert row["rmsd"] == 0
    assert row["added_heavy_atoms"] > 0
    generated.x += torch.tensor([1.0, 0, 0])
    assert scaffold_row(source, target, generated, mol, VOCAB)["rmsd"] == pytest.approx(
        1
    )
    first_kept = generated.keep_mask.nonzero()[0].item()
    generated.a[first_kept] = VOCAB.atom_tokens.index("O")
    assert not scaffold_row(source, target, generated, mol, VOCAB)["retention"]
    failed = scaffold_row(source, target, generated, None, VOCAB)
    assert not failed["validity"]
    assert failed["smiles"] is None
    unchanged = scaffold_row(source, target, source, None, VOCAB)
    assert unchanged["added_atoms"] == unchanged["added_heavy_atoms"] == 0


def test_aligned_rmsd_measures_deformation_without_input_frame_drift(tmp_path):
    *_, wrapped = _panel(tmp_path)
    source, target, generated, mol = _attempt(wrapped)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    generated.x = generated.x @ rotation + 10
    row = scaffold_row(source, target, generated, mol, VOCAB)
    assert row["rmsd"] > 1
    assert row["rmsd_aligned"] == pytest.approx(0, abs=1e-8)
    reference = row["posebusters_reference"]
    assert reference["positions"] == source.x.tolist()
    assert reference["kept_indices"] == generated.keep_mask.nonzero().flatten().tolist()
    generated.x[reference["kept_indices"][0], 0] += 0.5
    assert scaffold_row(source, target, generated, mol, VOCAB)["rmsd_aligned"] > 0.01


def test_scaffold_chemistry_routes_adjusted_metrics_and_supplies_ordered_context(
    monkeypatch,
):
    contexts = [{"kept_indices": [0], "positions": [[1, 2, 3]]}, None]
    module = SimpleNamespace(
        _scaffold_rows=[{"posebusters_reference": ctx} for ctx in contexts], metrics={}
    )
    mols = [object(), None]

    def compute(actual, metrics, *, posebusters_contexts):
        assert actual is mols
        assert posebusters_contexts == contexts
        return {"posebusters/validity": 0.0, "posebusters/scaffold_validity": 0.5}

    monkeypatch.setattr("morph.model.finetune.scaffold.scaffold.calc_metrics_", compute)
    result = LightningModuleRatesScaffoldFinetune._compute_chemistry_metrics(
        module, mols
    )
    assert result == {"posebusters/validity": 0.0, "scaffold/posebusters/validity": 0.5}


def test_grouped_scores_include_failures_and_gather_repeats_before_diversity(tmp_path):
    *_, wrapped = _panel(tmp_path)
    rows = []
    for index in range(16):
        source, target, generated, mol = _attempt(wrapped, index)
        # An entirely failed scaffold must remain in the macro denominator.
        if index >= 8:
            mol = None
        rows.append(scaffold_row(source, target, generated, mol, VOCAB))
    # Two rank-strided populations + a padded duplicate attempt.
    metrics = summarize_scaffold_rows(rows[::2] + rows[1::2] + rows[:1])
    assert metrics["n_samples"] == 16
    assert metrics["n_scaffolds"] == 2
    assert metrics["validity"] == metrics["recovery"] == 0.5
    assert (
        metrics["min_samples_per_scaffold"] == metrics["max_samples_per_scaffold"] == 8
    )
    assert metrics["uniqueness"] == metrics["unique_yield"] == 1 / 16
    assert metrics["diversity"] == 0
    rows[1]["smiles"] = "CCO"
    assert summarize_scaffold_rows(rows)["diversity"] > 0


def test_context_is_copied_before_sampling_and_restored_to_physical_units(tmp_path):
    *_, wrapped = _panel(tmp_path)
    batch = eval_collate_fn([wrapped[0], wrapped[1]])
    batch[0].x /= 2
    batch[1].x /= 2
    module = SimpleNamespace(distributions=SimpleNamespace(coordinate_std=2.0))
    sources, _ = LightningModuleRatesScaffoldFinetune._validation_context(module, batch)
    torch.testing.assert_close(sources[0].x, wrapped[0][0].x)
    batch[0].x += 100
    torch.testing.assert_close(sources[0].x, wrapped[0][0].x)


def test_validation_scores_the_existing_generation_and_preserves_input_context(
    tmp_path,
):
    *_, wrapped = _panel(tmp_path)
    batch = eval_collate_fn([wrapped[i] for i in range(8)])
    generated = chembatch_from_graphs([_attempt(wrapped, i)[2] for i in range(8)])

    class Validator(ValidationMixin):
        _validation_context = LightningModuleRatesScaffoldFinetune._validation_context
        _update_task_metrics = LightningModuleRatesScaffoldFinetune._update_task_metrics

        def sample(self, batch, batch_idx, return_traj=False):
            """Return the reference graphs and simulate in-place recentering."""
            self.calls += 1
            batch[0].x += 100  # Integration recenters the original input in place.
            return generated

    module = Validator()
    module.distributions = SimpleNamespace(coordinate_std=1.0)
    module.representation = SimpleNamespace(requires_topology=True)
    module.vocab = VOCAB
    module.allow_charged = False
    module.batch_metrics = module.distribution_metrics = module.stability_metrics = None
    module._scaffold_rows = []
    module._validation_rdkit_mols = []
    module.calls = 0
    module.validation_step(batch, 0)
    assert module.calls == 1
    metrics = summarize_scaffold_rows(module._scaffold_rows)
    assert metrics["n_samples"] == 8
    assert metrics["rmsd"] == 0
    assert metrics["valid_retention"] == 1


@pytest.mark.parametrize("rank", [0, 1])
def test_epoch_end_gathers_and_broadcasts_scaffold_scores(tmp_path, monkeypatch, rank):
    *_, wrapped = _panel(tmp_path)
    rows = [scaffold_row(*_attempt(wrapped, i), VOCAB) for i in range(8)]
    expected = summarize_scaffold_rows(rows)
    # Exercise the actual module hook without loading pretrained model weights.
    module = LightningModuleRatesScaffoldFinetune.__new__(
        LightningModuleRatesScaffoldFinetune
    )
    pl.LightningModule.__init__(module)
    module._trainer = SimpleNamespace(global_rank=rank)
    module._scaffold_rows = rows[rank::2]
    logged = {}
    monkeypatch.setattr(
        module, "log_dict", lambda values, **kwargs: logged.update(values)
    )
    chemistry_rows = []
    monkeypatch.setattr(
        ValidationMixin,
        "on_validation_epoch_end",
        lambda self: chemistry_rows.extend(self._scaffold_rows),
    )
    path = "morph.model.finetune.scaffold.scaffold.dist"
    monkeypatch.setattr(f"{path}.is_available", lambda: True)
    monkeypatch.setattr(f"{path}.is_initialized", lambda: True)
    monkeypatch.setattr(f"{path}.get_world_size", lambda: 2)

    def gather(local, result, dst):
        assert local == rows[rank::2]
        if rank == 0:
            result[:] = [rows[::2], rows[1::2]]
        else:
            assert result is None

    def broadcast(payload, src):
        if rank == 0:
            assert payload[0] == expected
        payload[0] = expected

    monkeypatch.setattr(f"{path}.gather_object", gather)
    monkeypatch.setattr(f"{path}.broadcast_object_list", broadcast)
    module.on_validation_epoch_end()
    assert logged == {f"val/scaffold/{key}": value for key, value in expected.items()}
    assert module._scaffold_rows == []
    assert chemistry_rows == (rows[::2] + rows[1::2] if rank == 0 else [])


def test_datamodule_uses_panel_only_for_eval_and_retains_partial_batch(
    tmp_path, monkeypatch
):
    dataset, _, panel, _ = _panel(tmp_path)
    panel_path(tmp_path, "val", "murcko", 4).unlink()
    module = ScaffoldLightningDataModule(
        vocab=VOCAB,
        distributions=None,
        interpolator=None,
        datasets=OmegaConf.create({}),
        num_workers=OmegaConf.create({"val": 0}),
        batch_size=OmegaConf.create({"train": 1, "val": 3}),
        scaffold_groups_dir=str(tmp_path),
        scaffold_kind="murcko",
        eval_panel=OmegaConf.create(
            {
                "directory": str(tmp_path),
                "n_scaffolds": 4,
                "seed": 42,
                "samples_per_scaffold": 8,
            }
        ),
    )
    monkeypatch.setattr("hydra.utils.instantiate", lambda *args, **kwargs: dataset)
    wrapped = module._wrap_dataset({}, "val")
    assert isinstance(wrapped, ScaffoldEvaluationDataset)
    assert panel_metadata(dataset, "murcko", 4, 42) == panel["metadata"]
    assert "eval_panel" not in module._wrapper_kwargs("train")
    module.val_datasets = [wrapped]
    loader = module.val_dataloader()[0]
    assert not loader.drop_last
    assert sum(batch[0].batch_size for batch in loader) == 32
