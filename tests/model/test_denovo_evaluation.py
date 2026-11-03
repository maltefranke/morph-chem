"""Regression coverage for the offline de novo SDF evaluation protocol."""

import json
import math
import os
import runpy
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from posebusters.modules import flatness
from rdkit import Chem
from rdkit.Chem import AllChem

from morph.utils import rdkit_utils as morphRD
from morph.utils.metrics import denovo
from morph.utils.metrics.posebusters_energy import EnergyRatioWorker

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_denovo.py"


def _conformer(smiles="CCO"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


def _write_sdf(path, mols):
    with Chem.SDWriter(str(path)) as writer:
        for mol in mols:
            writer.write(mol)


def test_load_retains_failures_and_both_hydrogen_representations(tmp_path):
    path = tmp_path / "generated.sdf"
    explicit = _conformer()
    _write_sdf(path, [explicit, Chem.RemoveHs(explicit)])
    with path.open("a") as handle:
        handle.write("broken\nrecord\n\nnot a counts line\n$$$$\n")
    mols = denovo.load_sdf(path)
    assert len(mols) == 3
    assert mols[-1] is None
    assert [m.GetNumAtoms() for m in mols[:2]] == [9, 3]
    assert all(morphRD.mol_is_valid(m) for m in mols[:2])
    stripped = denovo.load_sdf(path, remove_hs=True)
    assert [m.GetNumAtoms() for m in stripped[:2]] == [3, 3]
    report = denovo.compute_denovo_metrics(
        mols, run_energy=False, run_posebusters=False
    )
    assert report["results"]["validity"] == pytest.approx(2 / 3)
    assert report["results"]["uniqueness"] == 0.5
    assert report["counts"]["n_smiles"] == 2


def test_sdf_record_after_64k_boundary_is_not_dropped(tmp_path):
    mol = _conformer()
    mol.SetProp("_Name", "first")
    prefix = Chem.MolToMolBlock(mol) + "\n> <padding>\n"
    suffix = "\n\n$$$$\n"
    first = prefix + "x" * (65537 - len(prefix) - len(suffix)) + suffix
    mol.SetProp("_Name", "second")
    path = tmp_path / "boundary.sdf"
    path.write_text(first + Chem.MolToMolBlock(mol) + "\n$$$$\n")
    mols = denovo.load_sdf(path)
    assert [m.GetProp("_Name") for m in mols] == ["first", "second"]
    np.testing.assert_array_equal(
        mols[0].GetConformer().GetPositions(), mols[1].GetConformer().GetPositions()
    )
    assert mols[0].HasProp("padding")


def test_chemistry_uses_current_validity_and_correct_stability_direction():
    mols = [Chem.MolFromSmiles(s) for s in ("CCO", "[NH4+]", "[CH3]", "C.C")]
    mols.append(None)
    report = denovo.compute_denovo_metrics(
        mols, allow_charged=False, run_energy=False, run_posebusters=False
    )
    values = report["results"]
    assert values["validity"] == pytest.approx(1 / 5)
    assert values["connected-sanitization-validity"] == pytest.approx(3 / 5)
    assert math.isnan(values["rdkit-atom-stability"])  # Missing raw graph.
    assert report["counts"]["n_rdkit_stability_missing_graphs"] == 1
    assert values["rdkit-molecule-stability"] == pytest.approx(4 / 5)
    charged = denovo.compute_denovo_metrics(
        mols, run_energy=False, run_posebusters=False
    )
    assert charged["results"]["validity"] == pytest.approx(2 / 5)


def test_raw_graph_stability_includes_atoms_from_failed_sanitization():
    good = Chem.MolFromSmiles("CC")
    bad = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)
    before = bad.ToBinary()
    report = denovo.compute_denovo_metrics(
        [good, None],
        rdkit_stability_mols=[good, bad],
        run_energy=False,
        run_posebusters=False,
        run_diversity=False,
    )
    assert report["results"]["rdkit-atom-stability"] == 7 / 8
    assert report["results"]["rdkit-molecule-stability"] == 0.5
    assert report["results"]["validity"] == 0.5
    assert report["counts"]["n_rdkit_unstable_atoms"] == 1
    assert report["counts"]["n_rdkit_stability_atoms"] == 8
    assert report["counts"]["n_rdkit_stability_missing_graphs"] == 0
    assert bad.ToBinary() == before
    with pytest.raises(ValueError, match="every attempt"):
        denovo.compute_denovo_metrics([good, None], rdkit_stability_mols=[good])


def test_rdkit_stability_counts_multi_atom_kekulization_problems():
    mol = Chem.MolFromSmiles("c1cccc1", sanitize=False)
    assert denovo.calc_atom_stabilities(mol) == [False] * 5


def test_cli_raw_stability_survives_sanitization_failure(tmp_path):
    good = Chem.MolFromSmiles("CC")
    bad = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)
    raw, survivors, output = (
        tmp_path / name for name in ("raw.sdf", "ok.sdf", "report.json")
    )
    _write_sdf(raw, [good, bad])
    _write_sdf(survivors, [good])
    main = runpy.run_path(str(SCRIPT))["main"]
    common = [
        "--output-json",
        str(output),
        "--n-attempts",
        "2",
        "--no-energy",
        "--no-posebusters",
        "--no-diversity",
    ]
    # Both records in the main input: stability uses a separate raw load.
    main(["--sdf", str(raw), *common])
    report = json.loads(output.read_text())
    assert report["results"]["rdkit-atom-stability"] == 7 / 8
    assert report["counts"]["n_failed_load"] == 1
    # Exporter omitted the bad record: require its raw graph explicitly.
    main(["--sdf", str(survivors), *common])
    assert json.loads(output.read_text())["results"]["rdkit-atom-stability"] is None
    main(["--sdf", str(survivors), "--stability-sdf", str(raw), *common])
    report = json.loads(output.read_text())
    assert report["results"]["rdkit-atom-stability"] == 7 / 8
    assert report["results"]["rdkit-molecule-stability"] == 0.5
    assert report["counts"]["n_omitted_attempts"] == 1


def test_combined_metrics_intersect_before_deduplicating(monkeypatch):
    mols = [
        Chem.MolFromSmiles("CCO"),
        Chem.AddHs(Chem.MolFromSmiles("OCC")),
        Chem.MolFromSmiles("CCC"),
        Chem.MolFromSmiles("CCC"),
        Chem.MolFromSmiles("CCN"),
        Chem.MolFromSmiles("[NH4+]"),
        None,
        Chem.MolFromSmiles("C.C"),
    ]
    frame = pd.DataFrame({"check_a": [True, True, False, True, None, True, False]})
    monkeypatch.setattr(denovo, "_posebusters_chunk", lambda _mols, **_kwargs: frame)
    # Return one frame for the complete non-None population, so flags must be
    # realigned around the failed input at index 6.
    monkeypatch.setattr(denovo, "_chunks", lambda items, _workers: [items])
    report = denovo.compute_denovo_metrics(
        mols,
        allow_charged=False,
        reference_smiles={"CCO"},
        run_energy=False,
        run_diversity=False,
    )
    values = report["results"]
    assert values["novelty"] == pytest.approx(5 / 7)
    assert values["novelty-among-valid"] == pytest.approx(3 / 5)
    assert values["valid-unique"] == 3 / 8
    assert values["valid-novel"] == 3 / 8
    assert values["valid-unique-novel"] == 2 / 8
    assert values["valid-posebusters"] == 3 / 8
    assert values["valid-unique-novel-posebusters"] == 1 / 8
    rows = report["molecule_results"]
    assert rows[2]["canonical_smiles"] == rows[3]["canonical_smiles"]
    assert not rows[2]["posebusters_valid"]
    assert rows[3]["posebusters_valid"]
    assert not rows[4]["posebusters_valid"]  # A missing check fails.
    assert rows[6]["canonical_smiles"] is None
    assert not rows[6]["posebusters_valid"]
    assert not rows[7]["valid"]
    assert (
        report["counts"]["combined_numerators"]["valid-unique-novel-posebusters"] == 1
    )
    # A passing duplicate must count even if a failing pose occurred first.
    reversed_values, _, _ = denovo._combined_metrics(
        mols[::-1],
        allow_charged=False,
        reference_smiles={"CCO"},
        pb_passes=[row["posebusters_valid"] for row in rows][::-1],
    )
    for key, value in reversed_values.items():
        assert values[key] == value


@pytest.mark.parametrize("stereochemistry,novelty", [("ignore", 0), ("retain", 0.5)])
def test_novelty_reference_matches_hydrogens_and_stereo_policy(
    stereochemistry, novelty
):
    reference = denovo.canonicalize_reference_smiles(
        ["OCC", "[H]OC([H])([H])C([H])([H])[H]", "F[C@H](Cl)Br"],
        smiles_stereochemistry=stereochemistry,
    )
    assert len(reference) == 2
    assert "CCO" in reference
    report = denovo.compute_denovo_metrics(
        [Chem.MolFromSmiles("F[C@@H](Cl)Br"), Chem.AddHs(Chem.MolFromSmiles("CCO"))],
        reference_smiles=reference,
        smiles_stereochemistry=stereochemistry,
        run_energy=False,
        run_posebusters=False,
    )
    assert report["results"]["valid-unique-novel"] == novelty
    assert report["results"]["novelty"] == novelty
    assert "valid-posebusters" not in report["results"]
    assert all(row["posebusters_valid"] is None for row in report["molecule_results"])
    for invalid in ([], ["CCO", "invalid"], [""]):
        with pytest.raises(ValueError, match="reference"):
            denovo.canonicalize_reference_smiles(invalid)


def test_sdf_inferred_stereo_does_not_inflate_training_graph_novelty(tmp_path):
    # Training constructs graphs with coordinates but no assigned stereotags.
    graph = _conformer("CC(O)F")
    Chem.RemoveStereochemistry(graph)
    training_smiles = morphRD.smiles_from_mol(graph)
    assert "@" not in training_smiles
    sdf, reference, output = [
        tmp_path / name for name in ("generated.sdf", "train.smi", "report.json")
    ]
    _write_sdf(sdf, [graph])
    reference.write_text(training_smiles + "\n")
    loaded = denovo.load_sdf(sdf)[0]
    assert "@" in morphRD.smiles_from_mol(loaded)
    main = runpy.run_path(str(SCRIPT))["main"]
    args = [
        "--sdf",
        str(sdf),
        "--reference-smiles",
        str(reference),
        "--output-json",
        str(output),
        "--no-energy",
        "--no-posebusters",
        "--no-diversity",
    ]
    main(args)
    report = json.loads(output.read_text())
    assert report["results"]["novelty"] == 0
    assert report["molecule_results"][0]["canonical_smiles"] == training_smiles
    assert report["settings"]["smiles_stereochemistry"] == "ignore"
    main([*args, "--smiles-stereochemistry", "retain"])
    report = json.loads(output.read_text())
    assert report["results"]["novelty"] == 1
    assert report["settings"]["smiles_stereochemistry"] == "retain"


def test_identity_preserves_isotopes_and_input_stereo_and_coordinates():
    mols = [_conformer(s) for s in ("C[C@H](O)F", "C[C@@H](O)F", "[13CH3][C@H](O)F")]
    before_smiles = [Chem.MolToSmiles(m) for m in mols]
    before_coords = [m.GetConformer().GetPositions().copy() for m in mols]
    for policy, expected in (("ignore", 2 / 3), ("retain", 1)):
        report = denovo.compute_denovo_metrics(
            mols,
            run_energy=False,
            run_posebusters=False,
            run_diversity=False,
            smiles_stereochemistry=policy,
        )
        assert report["results"]["uniqueness"] == expected
        assert report["results"]["valid-unique"] == expected
        assert "13" in report["molecule_results"][2]["canonical_smiles"]
        assert [Chem.MolToSmiles(m) for m in mols] == before_smiles
        for mol, coords in zip(mols, before_coords, strict=True):
            np.testing.assert_array_equal(mol.GetConformer().GetPositions(), coords)
    with pytest.raises(ValueError, match="smiles_stereochemistry"):
        denovo.compute_denovo_metrics(mols, smiles_stereochemistry="invalid")
    with pytest.raises(ValueError, match="smiles_stereochemistry"):
        denovo.canonicalize_reference_smiles(["CCO"], smiles_stereochemistry="invalid")


def test_combined_metrics_without_reference_and_with_all_failures():
    report = denovo.compute_denovo_metrics(
        [None], run_energy=False, run_posebusters=False
    )
    assert report["results"]["valid-unique"] == 0
    assert "novelty" not in report["results"]
    assert report["molecule_results"][0]["novel"] is None
    report = denovo.compute_denovo_metrics(
        [None],
        reference_smiles={"CCO"},
        run_energy=False,
    )
    assert math.isnan(report["results"]["novelty"])
    assert math.isnan(report["results"]["novelty-among-valid"])
    assert report["results"]["valid-unique-novel-posebusters"] == 0
    with pytest.raises(ValueError, match="reference"):
        denovo.compute_denovo_metrics([None], reference_smiles=set())


def test_cli_novelty_records_reference_and_input_alignment(tmp_path):
    sdf, reference, output = [
        tmp_path / name for name in ("generated.sdf", "train.smi", "metrics.json")
    ]
    _write_sdf(sdf, [_conformer(), _conformer("CCC")])
    reference.write_text("OCC\nCCO\n")
    main = runpy.run_path(str(SCRIPT))["main"]
    args = [
        "--sdf",
        str(sdf),
        "--reference-smiles",
        str(reference),
        "--output-json",
        str(output),
        "--n-attempts",
        "3",
        "--workers",
        "1",
        "--no-energy",
        "--no-posebusters",
    ]
    main(args)
    report = json.loads(output.read_text())
    assert report["results"]["valid-unique-novel"] == pytest.approx(1 / 3)
    assert report["novelty_reference"]["n_entries"] == 2
    assert report["novelty_reference"]["n_unique_canonical"] == 1
    assert len(report["novelty_reference"]["sha256"]) == 64
    rows = report["molecule_results"]
    assert [row["sdf_record_index"] for row in rows] == [0, 1, None]
    assert rows[0]["source_path"] == str(sdf.resolve())
    assert rows[2]["source_path"] is None
    assert not rows[2]["valid"]
    reference.write_text("invalid\n")
    with pytest.raises(SystemExit):
        main(args)
    with pytest.raises(SystemExit):
        main(
            [
                "--sdf",
                str(sdf),
                "--reference-smiles",
                str(reference),
                "--output-json",
                str(reference),
            ]
        )


def test_energy_matches_original_morph_formulas_without_mutating_input():
    mol = _conformer()
    coords = mol.GetConformer().GetPositions().copy()
    values = denovo._energy_record(mol, add_hs=False, max_iters=1000)
    optimized = morphRD.optimise_mol(mol)
    energy = morphRD.calc_energy(mol)
    opt_energy = morphRD.calc_energy(optimized)
    assert values["energy"] == pytest.approx(energy)
    assert values["energy-per-atom"] == pytest.approx(energy / mol.GetNumAtoms())
    assert values["strain"] == pytest.approx(energy - opt_energy)
    assert values["strain-per-atom"] == pytest.approx(
        (energy - opt_energy) / mol.GetNumAtoms()
    )
    assert values["opt-rmsd"] == pytest.approx(morphRD.conf_distance(mol, optimized))
    assert values["optimization_status"] == 0
    np.testing.assert_array_equal(mol.GetConformer().GetPositions(), coords)


def test_energy_hydrogen_addition_only_changes_energy_copy():
    mol = Chem.RemoveHs(_conformer())
    coords = mol.GetConformer().GetPositions().copy()
    values = denovo._energy_record(mol, add_hs=True, max_iters=1000)
    added = Chem.AddHs(mol, addCoords=True)
    assert values["energy"] == pytest.approx(morphRD.calc_energy(added))
    assert values["energy-per-atom"] == pytest.approx(
        values["energy"] / added.GetNumAtoms()
    )
    assert mol.GetNumAtoms() == 3
    np.testing.assert_array_equal(mol.GetConformer().GetPositions(), coords)


def test_energy_counts_missing_poses_and_nonconvergence():
    mols = [_conformer("CCCC"), None, Chem.MolFromSmiles("CCO")]
    values, counts = denovo._energy_metrics(mols, workers=1, add_hs=False, max_iters=1)
    assert values["energy-validity"] == pytest.approx(1 / 3)
    assert values["opt-energy-validity"] == pytest.approx(1 / 3)
    assert counts["strain"] == 1
    assert counts["optimization-not-converged"] == 1
    assert counts["optimization-failed"] == 2


@pytest.mark.parametrize("bad_energy", [None, float("nan"), float("inf")])
def test_strain_never_pairs_a_failed_input_energy_with_an_optimized_energy(
    monkeypatch, bad_energy
):
    energies = iter([bad_energy, 1.0])
    monkeypatch.setattr(morphRD, "calc_energy", lambda _mol: next(energies))
    values, counts = denovo._energy_metrics(
        [_conformer()], workers=1, add_hs=False, max_iters=1000
    )
    assert math.isnan(values["strain"])
    assert counts["strain"] == 0
    assert values["energy-validity"] == 0
    assert values["opt-energy-validity"] == 1


def test_posebusters_aggregates_keep_all_attempts_and_nan_failures():
    frame = pd.DataFrame(
        {"check_a": [True, True, None], "check_b": [True, False, True]}
    )
    values = denovo._summarize_posebusters(frame, n_attempts=4, n_busted=3)
    assert values["check_a"] == 1.0  # Original per-check non-NaN mean.
    assert values["check_a__over_generated"] == 0.5
    assert values["check_a__n_nan"] == 1
    assert values["validity"] == 0.25
    assert values["all_pass_over_busted"] == pytest.approx(1 / 3)
    assert values["n_incomplete"] == 1
    with pytest.raises(ValueError, match="one row"):
        denovo._summarize_posebusters(frame, n_attempts=4, n_busted=4)


def test_real_posebusters_uses_full_original_mol_config():
    report = denovo.compute_denovo_metrics(
        [_conformer(), None], run_energy=False, run_diversity=False
    )
    values = report["results"]
    assert values["posebusters/n_busted"] == 1
    assert values["posebusters/n_generated"] == 2
    assert values["posebusters/n_checks"] > 8
    modules = report["settings"]["posebusters_config"]["modules"]
    assert {"energy_ratio", "flatness"} <= {m["function"] for m in modules}
    assert sum(m["function"] == "energy_ratio" for m in modules) == 1
    for module in modules:
        if module["function"] == "energy_ratio":
            assert module["parameters"]["num_threads"] == 1
            assert module["parameters"]["ensemble_number_conformations"] == 50
            assert module["parameters"]["threshold_energy_ratio"] == 100.0
    assert any("flatness" in name for name in values)
    assert any("energy" in name for name in values)


def _slow_energy_server(connection):
    connection.send("ready")
    while True:
        mol, _parameters = connection.recv()
        if mol.GetNumAtoms() == 1:
            time.sleep(30)
        connection.send({"results": {"energy_ratio_passes": True}})


def _chunk_names(mols):
    return [mol.GetProp("_Name") for mol in mols]


def test_parallel_chunks_preserve_molecule_ids():
    mol = _conformer()
    mol.SetProp("_Name", "seed_42_00001")
    assert denovo._map_chunks(_chunk_names, [[mol]], 2) == [["seed_42_00001"]]


def test_energy_timeout_kills_worker_and_next_molecule_still_runs():
    mol = Chem.MolFromSmiles("C")
    mol.SetProp("_Name", "slow")
    with EnergyRatioWorker(0.1, _target=_slow_energy_server) as worker:
        result = worker.check(mol)
        assert math.isnan(result["results"]["energy_ratio_passes"])
        assert worker.timed_out_names == ["slow"]
        assert worker.process is None
        assert worker.check(Chem.MolFromSmiles("CC"))["results"]["energy_ratio_passes"]
        child = worker.process
    assert worker.process is None
    assert child._closed


def test_chunk_cache_survives_failure_and_invalidates_changed_coordinates(
    tmp_path, monkeypatch
):
    first, second = _conformer("CCO"), _conformer("CCN")
    function = partial(denovo._energy_chunk, add_hs=False, max_iters=1)
    original = denovo._energy_record
    calls = []

    def interrupted(mol, **kwargs):
        calls.append(mol)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return original(mol, **kwargs)

    monkeypatch.setattr(denovo, "_energy_record", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        denovo._map_chunks(function, [[first], [second]], 1, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.pkl"))) == 1
    calls.clear()
    results = denovo._map_chunks(function, [[first], [second]], 1, cache_dir=tmp_path)
    assert len(calls) == 1
    assert calls[0] is second
    assert len(results) == 2
    assert results == denovo._map_chunks(
        function, [[first], [second]], 1, cache_dir=tmp_path
    )
    first.GetConformer().SetAtomPosition(0, (0, 0, 0))
    calls.clear()
    denovo._map_chunks(function, [[first], [second]], 1, cache_dir=tmp_path)
    assert len(calls) == 1
    assert calls[0] is first


def test_posebusters_flatness_coordinates_preserve_atom_order():
    mol = _conformer("c1ccccc1")
    indices = [4, 0, 2]
    conf = mol.GetConformer()
    expected = np.array(
        [
            (
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            )
            for i in indices
        ]
    )
    np.testing.assert_array_equal(
        denovo._flatness_coordinates(mol, iter(indices)), expected
    )


def test_posebusters_flatness_detects_distorted_aromatic_ring():
    original_reader = flatness._get_coords
    mol = _conformer("c1ccccc1")
    # Put the ring in a plane, then displace one atom out of that plane.
    conf = mol.GetConformer()
    for i in range(6):
        theta = i * math.pi / 3
        conf.SetAtomPosition(i, (1.4 * math.cos(theta), 1.4 * math.sin(theta), 0))
    flat = denovo._posebusters_chunk([mol])
    conf.SetAtomPosition(0, (1.4, 0, 1.0))
    distorted = denovo._posebusters_chunk([mol])
    assert bool(flat.iloc[0]["aromatic_ring_flatness"])
    assert not bool(distorted.iloc[0]["aromatic_ring_flatness"])
    assert flatness._get_coords is original_reader


def test_all_failures_have_zero_rates_and_undefined_averages():
    report = denovo.compute_denovo_metrics([None, None], run_posebusters=False)
    assert report["results"]["validity"] == 0
    assert report["results"]["rdkit-molecule-stability"] == 0
    assert report["results"]["energy-validity"] == 0
    assert math.isnan(report["results"]["uniqueness"])
    assert math.isnan(report["results"]["energy"])
    with pytest.raises(ValueError, match="No attempted"):
        denovo.compute_denovo_metrics([])


def test_cli_pools_files_accounts_for_omitted_attempts_and_writes_strict_json(tmp_path):
    first, second = tmp_path / "first.sdf", tmp_path / "second.sdf"
    _write_sdf(first, [_conformer()])
    _write_sdf(second, [_conformer()])
    output = tmp_path / "report.json"
    main = runpy.run_path(str(SCRIPT))["main"]
    main(
        [
            "--sdf",
            str(first),
            str(second),
            "--output-json",
            str(output),
            "--n-attempts",
            "4",
            "--workers",
            "1",
            "--no-energy",
            "--no-posebusters",
        ]
    )
    report = json.loads(output.read_text())
    assert report["counts"]["n_attempts"] == 4
    assert report["counts"]["n_sdf_records"] == 2
    assert report["counts"]["n_omitted_attempts"] == 2
    assert report["results"]["validity"] == 0.5
    assert report["results"]["uniqueness"] == 0.5
    assert report["results"]["diversity"] == 0
    assert len(report["inputs"]) == 2
    assert len(report["inputs"][0]["sha256"]) == 64
    assert report["settings"]["denominator_source"] == "explicit_n_attempts"
    with pytest.raises(SystemExit):
        main(
            [
                "--sdf",
                str(first),
                str(second),
                "--output-json",
                str(output),
                "--n-attempts",
                "1",
            ]
        )


def test_empty_sdf_with_known_failed_attempts_writes_null_not_nan(tmp_path):
    path, output = tmp_path / "empty.sdf", tmp_path / "report.json"
    path.touch()
    main = runpy.run_path(str(SCRIPT))["main"]
    main(
        [
            "--sdf",
            str(path),
            "--output-json",
            str(output),
            "--n-attempts",
            "3",
            "--workers",
            "1",
            "--no-posebusters",
        ]
    )
    report = json.loads(output.read_text())
    assert report["results"]["validity"] == 0
    assert report["results"]["uniqueness"] is None
    assert report["results"]["energy"] is None
    assert "NaN" not in output.read_text()
    with pytest.raises(SystemExit):
        main(["--sdf", str(path), "--output-json", str(output)])


def test_cli_full_evaluation_with_spawn_workers(tmp_path):
    path, output = tmp_path / "generated.sdf", tmp_path / "report.json"
    mol = Chem.RemoveHs(_conformer("c1ccccc1"))
    _write_sdf(path, [mol, mol])
    # Spawned descendants can retain inherited pipe handles after the CLI
    # exits. A file keeps diagnostics without waiting for those handles to close.
    log_path = tmp_path / "evaluation.log"
    with log_path.open("w") as log:
        process = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--sdf",
                str(path),
                "--output-json",
                str(output),
                "--n-attempts",
                "3",
                "--energy-hydrogens",
                "add",
                "--workers",
                "2",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
            env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
        )
    assert process.returncode == 0, log_path.read_text()
    report = json.loads(output.read_text())
    assert report["results"]["validity"] == pytest.approx(2 / 3)
    assert report["results"]["energy-validity"] == pytest.approx(2 / 3)
    assert report["counts"]["energy"]["strain"] == 2
    assert report["results"]["posebusters/n_busted"] == 2
    assert report["results"]["posebusters/n_generated"] == 3
    assert report["results"]["posebusters/n_checks"] > 8
    assert report["settings"]["energy_hydrogens"] == "add"
    assert report["settings"]["workers"] == 2
