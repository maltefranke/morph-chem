"""Docking denominators and failure handling must match the released DiffDec code."""

import copy
import subprocess
import sys

import pytest

from morph.utils.docking import aggregate_docking, bounded_dock


def _cases():
    records, results = [], []
    for condition, (reference, successes, score, group) in enumerate(
        [(-7, 80, -8, "a"), (-5, 50, -6, "a"), (None, 25, -9, "b")]
    ):
        results.append(
            dict(
                kind="reference",
                i=condition,
                affinity=reference,
                status="ok" if reference is not None else "failed",
                replacement=False,
            )
        )
        for index in range(100):
            records.append(
                dict(scaffold="C", true_molecule="CC", protein_filename=group)
            )
            results.append(
                dict(
                    kind="prediction",
                    i=condition * 100 + index,
                    affinity=score if index < successes else None,
                    status="ok" if index < successes else "failed",
                    replacement=index < 10,
                )
            )
    return records, results


def test_group_weighting_and_missing_reference_match_released_formulas():
    records, results = _cases()
    report = aggregate_docking(records, results)
    assert report["vina_score"] == pytest.approx(((-8 * 80 - 6 * 50) / 130 - 9) / 2)
    assert report["reference_vina_score"] == -6
    assert report["high_affinity"] == pytest.approx(0.325)
    assert report["high_affinity_denominator"] == 275
    assert report["docking_successes"] == 155
    assert report["docking_failures"] == 145
    assert report["reference_docking_successes"] == 2
    assert report["invalid_predictions_replaced"] == 30


def test_incomplete_duplicate_or_misaligned_results_cannot_look_complete():
    records, results = _cases()
    with pytest.raises(ValueError, match="Missing"):
        aggregate_docking(records, results[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_docking(records, [*results, results[0]])
    changed = copy.deepcopy(records)
    changed[1]["protein_filename"] = "wrong"
    with pytest.raises(ValueError, match="mixes"):
        aggregate_docking(changed, results)


def test_all_failed_docking_scores_are_undefined_and_high_affinity_zero():
    records, results = _cases()
    for result in results:
        result.update(status="failed", affinity=None)
    report = aggregate_docking(records, results)
    assert report["vina_score"] is None
    assert report["reference_vina_score"] is None
    assert report["high_affinity"] == 0
    assert report["high_affinity_denominator"] == len(records)


def test_whole_ligand_timeout_terminates_worker(monkeypatch, tmp_path):
    original = subprocess.Popen
    processes = []

    def sleeping_worker(command, **kwargs):
        process = original(
            [sys.executable, "-c", "import time; time.sleep(60)"], **kwargs
        )
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", sleeping_worker)
    task = dict(
        kind="prediction",
        i=0,
        embedding_seed=1,
        replacement=False,
        receptor={},
        timeout=0.1,
    )
    result = bounded_dock(task, tmp_path)
    assert result["status"] == "timeout"
    assert result["affinity"] is None
    assert processes[0].poll() is not None
    assert not list(tmp_path.iterdir())
