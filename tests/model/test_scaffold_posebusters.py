"""Inherited scaffold clashes must not hide newly generated geometry failures."""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from morph.utils.metrics.chemistry import _posebusters_config, calc_posebusters_metrics
from morph.utils.metrics.scaffold_posebusters import (
    inherited_clash_counts,
    scaffold_posebusters_metrics,
)


def _folded_chain():
    mol = Chem.MolFromSmiles("CCCCC")
    positions = np.array(
        [[0, 0, 0], [1.5, 0, 0], [1.5, 1.5, 0], [0, 1.5, 0], [0, 0, 0.1]], dtype=float
    )
    conformer = Chem.Conformer(5)
    for index, position in enumerate(positions):
        conformer.SetAtomPosition(index, position)
    mol.AddConformer(conformer)
    reference = {"kept_indices": list(range(5)), "positions": positions.tolist()}
    parameters = next(
        m["parameters"]
        for m in _posebusters_config()["modules"]
        if m["function"] == "distance_geometry"
    )
    return mol, reference, parameters


def test_inherited_clash_exemption_is_invariant_to_rigid_motion():
    mol, reference, parameters = _folded_chain()
    coords = mol.GetConformer().GetPositions()
    for i, (x, y, z) in enumerate(coords):
        mol.GetConformer().SetAtomPosition(i, (10 - y, x - 20, z + 5))
    count, inherited = inherited_clash_counts(mol, reference, parameters)
    assert count > 0
    assert inherited == count


@pytest.mark.parametrize("failure", ["worsened", "new", "interface"])
def test_new_worsened_and_interface_clashes_are_not_exempt(failure):
    mol, reference, parameters = _folded_chain()
    if failure == "worsened":
        mol.GetConformer().SetAtomPosition(4, (0, 0, 0.05))
    elif failure == "new":
        reference["positions"][4] = [0, 0, 10]
    else:
        reference["kept_indices"].pop()
        reference["positions"].pop()
    count, inherited = inherited_clash_counts(mol, reference, parameters)
    assert inherited < count


def test_adjustment_preserves_other_failures_missing_checks_and_denominator():
    mol, reference, parameters = _folded_chain()
    checks = pd.DataFrame(
        {
            "internal_steric_clash": [False, False, False, None, False],
            "bond_lengths": [True, False, None, True, True],
        }
    )
    contexts = [reference] * 4 + [None]
    scores = scaffold_posebusters_metrics(checks, [mol] * 5, contexts, 6, parameters)
    assert scores["scaffold_n_clash_exemptions"] == 3
    assert scores["scaffold_n_all_pass"] == 1
    assert scores["scaffold_validity"] == pytest.approx(1 / 6)


def test_real_posebusters_keeps_strict_score_and_counts_unbuildable_attempts():
    mol, reference, _ = _folded_chain()
    scores = calc_posebusters_metrics(
        [mol, None], max_workers=0, scaffold_contexts=[reference, None]
    )
    assert scores["validity"] == 0
    assert scores["scaffold_n_clash_exemptions"] == 1
    assert scores["scaffold_validity"] == 0.5
    assert scores["scaffold_n_all_pass"] == 1


def test_failed_clash_analysis_does_not_grant_an_exemption():
    mol, reference, parameters = _folded_chain()
    reference["positions"][0][0] = float("nan")
    scores = scaffold_posebusters_metrics(
        pd.DataFrame({"internal_steric_clash": [False]}),
        [mol],
        [reference],
        1,
        parameters,
    )
    assert scores["scaffold_validity"] == 0
    assert scores["scaffold_n_clash_analysis_failed"] == 1


def test_no_buildable_molecules_still_logs_scaffold_scores():
    scores = calc_posebusters_metrics([None], scaffold_contexts=[None])
    assert scores["scaffold_validity"] == 0
    with pytest.raises(ValueError, match="align"):
        calc_posebusters_metrics([None], scaffold_contexts=[])


def test_entirely_unavailable_check_is_a_failure(monkeypatch):
    class FakePoseBusters:
        def __init__(self, **kwargs):
            pass

        def bust(self, *_args):
            return pd.DataFrame({"one_check": [True], "missing_check": [None]})

    monkeypatch.setattr("posebusters.PoseBusters", FakePoseBusters)
    scores = calc_posebusters_metrics([object()], max_workers=0)
    assert scores["validity"] == 0
    assert scores["n_incomplete"] == 1
