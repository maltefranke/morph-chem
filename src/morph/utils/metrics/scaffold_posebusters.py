"""Scaffold-aware clash scoring without changing the strict PoseBusters report."""

import numpy as np


def inherited_clash_counts(mol, reference, parameters, *, tolerance=1e-4):
    """Count all clashes and the non-worsened pairs inherited from input geometry.

    Both distances use the generated molecule's topology/bounds. Only retained
    pairs with a verified atom mapping can be exempted; the hidden target is
    never consulted. ``tolerance`` is a numerical tolerance in Angstroms, not
    allowance for additional overlap. An unavailable check raises and the caller
    retains the strict failure.
    """
    from posebusters.modules.distance_geometry import check_geometry  # noqa: PLC0415

    indices = reference["kept_indices"]
    positions = np.asarray(reference["positions"], dtype=float)
    if (
        positions.shape != (len(indices), 3)
        or not np.isfinite(positions).all()
        or len(set(indices)) != len(indices)
        or any(i < 0 or i >= mol.GetNumAtoms() for i in indices)
    ):
        raise ValueError("Invalid scaffold coordinate mapping")
    original = dict(zip(indices, positions, strict=True))
    geometry = check_geometry(mol, **parameters)
    pairs = geometry["details"]["clash"]
    threshold = parameters.get("threshold_clash", 0.2)
    clashes = pairs.loc[pairs["bound_percent_error"] < -threshold]
    if not len(clashes) or len(clashes) != geometry["results"]["number_clashes"]:
        raise ValueError("Detailed clash check disagrees with the strict result")
    inherited = 0
    for pair in clashes.itertuples(index=False):
        i, j = pair.atom_pair
        if i not in original or j not in original:
            continue
        baseline_distance = float(np.linalg.norm(original[i] - original[j]))
        baseline_clashes = baseline_distance < (1 - threshold) * pair.lower_bound
        if baseline_clashes and pair.distance + tolerance >= baseline_distance:
            inherited += 1
    return len(clashes), inherited


def scaffold_posebusters_metrics(checks, mols, contexts, n_generated, parameters):
    """Exempt inherited clashes; require every other strict check to pass."""
    # Normalize booleans and missing entries before replacing a failed check;
    # pandas 3 rejects assigning integer 1 into a boolean column.
    adjusted = checks.astype(float)
    clash_column = "internal_steric_clash"
    exemptions = inherited_pairs = failed_analysis = 0
    if clash_column in adjusted:
        column_index = adjusted.columns.get_loc(clash_column)
        for index, (mol, reference) in enumerate(zip(mols, contexts, strict=True)):
            # Missing context/NaN checks never earn an exemption.
            if reference is None or adjusted.iat[index, column_index] != 0:
                continue
            try:
                count, inherited = inherited_clash_counts(mol, reference, parameters)
            except Exception:
                failed_analysis += 1
                continue
            inherited_pairs += inherited
            if count == inherited:
                adjusted.iat[index, column_index] = 1
                exemptions += 1
    passed = adjusted.fillna(0).astype(bool).all(axis=1)
    n_passed = int(passed.sum()) if len(adjusted.columns) else 0
    return {
        "scaffold_validity": n_passed / n_generated if n_generated else float("nan"),
        "scaffold_n_all_pass": float(n_passed),
        "scaffold_internal_steric_clash_over_generated": (
            float(adjusted[clash_column].fillna(0).sum()) / n_generated
            if n_generated and clash_column in adjusted
            else 0.0
        ),
        "scaffold_n_clash_exemptions": float(exemptions),
        "scaffold_n_inherited_clash_pairs": float(inherited_pairs),
        "scaffold_n_clash_analysis_failed": float(failed_analysis),
    }
