"""Non-docking evaluation functions used in DiffDec (Xie et al., JCIM 2024).

Ported from ``biomed-AI/DiffDec/compute_metrics.py`` at revision
``916ae14207b2783a90336bb8509374535c5791f9``:
https://github.com/biomed-AI/DiffDec/blob/916ae14207b2783a90336bb8509374535c5791f9/compute_metrics.py

These reproduce the released evaluator, including its fixed uniqueness
denominator, raw SMILES grouping, and omission of all-invalid groups from
uniqueness/similarity. They must not be substituted for Morph's chemistry
validity or pairwise diversity. Upstream copyright: 2024 Biomed-AI@NSCC-GZ.
See ``morph/_vendor/licenses/DIFFDEC_LICENSE`` for the upstream MIT license.

Two explicit boundary extensions: empty input raises ValueError; when *every*
attempt is invalid, uniqueness and similarity are NaN (upstream divides by
zero). Otherwise failures are propagated, never silently filtered out.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from rdkit import Chem, DataStructs

DIFFDEC_REVISION = "916ae14207b2783a90336bb8509374535c5791f9"
DIFFDEC_SAMPLES_PER_CONDITION = 100
_FIELDS = (
    "scaffold",
    "true_molecule",
    "pred_molecule",
    "pred_rgroup",
    "protein_filename",
)


def _parse(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol if mol is not None else Chem.MolFromSmiles(smiles, sanitize=False)


def diffdec_is_valid(pred_smiles: str, scaffold_smiles: str) -> bool:
    """Return validity as used in DiffDec, including unsanitized parsing.

    The prediction must be nonempty, differ textually from the scaffold, and
    contain it as a substructure. Connectivity, radicals, valence, coordinates,
    and stereochemical agreement are not additional requirements here.
    """
    if pred_smiles == "" or pred_smiles == scaffold_smiles:
        return False
    pred = _parse(pred_smiles)
    scaffold = _parse(scaffold_smiles)
    if scaffold is None:
        raise ValueError(f"Cannot parse reference scaffold: {scaffold_smiles!r}")
    if pred is None:
        return False
    return len(pred.GetSubstructMatch(scaffold)) == scaffold.GetNumAtoms()


def diffdec_similarity(pred_smiles: str, true_smiles: str) -> float:
    """Return reference-ligand similarity as used in DiffDec.

    This is Tanimoto similarity of RDKit path fingerprints (RDKFingerprint),
    with the released defaults, not Morgan fingerprints or pairwise diversity.
    Call only for predictions accepted by ``diffdec_is_valid``.
    """
    return DataStructs.FingerprintSimilarity(
        Chem.RDKFingerprint(_parse(pred_smiles)),
        Chem.RDKFingerprint(_parse(true_smiles)),
    )


def diffdec_recovery_smiles(smiles: str) -> str:
    """Return the hydrogen/stereo-stripped recovery key used in DiffDec.

    Retry the complete operation without sanitization if the first attempt
    fails, as in the released evaluator. Uniqueness does not use this key.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False))


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def compute_diffdec_metrics(records) -> dict[str, float]:
    """Compute the four non-docking metrics used in DiffDec, exactly as released.

    Records use keys ``scaffold``, ``true_molecule``, ``pred_molecule``, and
    ``protein_filename``; ``pred_rgroup`` is optional and unused by the scores.
    Every attempted generation must be present, including failures (empty
    prediction). Input records are not modified or canonicalized.

    * Validity: mean success fraction per (protein, raw scaffold SMILES), then
      mean across that protein's groups, then mean across proteins.
    * Uniqueness: distinct *raw* valid prediction SMILES / **100**, per group;
      then the same nested averaging, omitting groups/proteins with no valid
      predictions. Multiple benchmark instances sharing a group are merged but
      still divided by 100; this upstream convention is deliberately preserved.
    * Similarity: mean reference similarity over valid attempts **per protein**,
      then mean across proteins with valid outputs (no scaffold-level mean).
    * Recovery: fraction of distinct canonical, stereo/H-stripped scaffolds
      per protein with any valid prediction matching its reference after the
      same normalization, then mean across proteins, including failures.

    Validity, uniqueness and recovery are percentages; similarity is in [0, 1].
    Use 100 attempts per benchmark instance; the denominator is not configurable.
    """
    records = list(records)
    if not records:
        raise ValueError("DiffDec evaluation requires at least one attempt")
    valid_groups = defaultdict(list)
    unique_groups = defaultdict(set)
    similarities = defaultdict(list)
    seen_scaffolds = defaultdict(set)
    recovered_scaffolds = defaultdict(set)
    for row in records:
        protein, scaffold = row["protein_filename"], row["scaffold"]
        pred, true = row["pred_molecule"], row["true_molecule"]
        key = (protein, scaffold)
        valid = diffdec_is_valid(pred, scaffold)
        valid_groups[key].append(valid)
        scaffold_key = diffdec_recovery_smiles(scaffold)
        seen_scaffolds[protein].add(scaffold_key)
        if valid:
            unique_groups[key].add(pred)
            similarities[protein].append(diffdec_similarity(pred, true))
            if diffdec_recovery_smiles(pred) == diffdec_recovery_smiles(true):
                recovered_scaffolds[protein].add(scaffold_key)

    validity_by_protein = defaultdict(list)
    uniqueness_by_protein = defaultdict(list)
    for (protein, _), values in valid_groups.items():
        validity_by_protein[protein].append(_mean(values))
    for (protein, _), values in unique_groups.items():
        uniqueness_by_protein[protein].append(
            len(values) / DIFFDEC_SAMPLES_PER_CONDITION
        )
    return {
        "validity": 100 * _mean(map(_mean, validity_by_protein.values())),
        "uniqueness": 100 * _mean(map(_mean, uniqueness_by_protein.values())),
        "similarity": _mean(map(_mean, similarities.values())),
        "recovery": 100
        * _mean(
            len(recovered_scaffolds[protein]) / len(scaffolds)
            for protein, scaffolds in seen_scaffolds.items()
        ),
    }


def read_diffdec_records(path) -> list[dict[str, str]]:
    """Read the four/five-column metric .smi format used in DiffDec.

    Literal spaces delimit fields: adjacent spaces encode an empty prediction
    or R-group. Generic whitespace splitting would silently shift those fields.
    """
    records = []
    with Path(path).open() as handle:
        for lineno, line in enumerate(handle, 1):
            fields = line.strip().split(" ")
            if len(fields) == 4:
                fields.insert(3, "")
            if len(fields) != 5 or not all(fields[i] for i in (0, 1, 4)):
                raise ValueError(f"{path}:{lineno}: expected DiffDec metric fields")
            records.append(dict(zip(_FIELDS, fields, strict=True)))
    return records


def write_diffdec_records(records, path) -> None:
    """Write all attempts in the five-column metric .smi format used in DiffDec."""
    with Path(path).open("w") as handle:
        for row in records:
            fields = [row.get(key, "") for key in _FIELDS]
            if not all(fields[i] for i in (0, 1, 4)) or any(
                any(char.isspace() for char in value) for value in fields
            ):
                raise ValueError(
                    "DiffDec fields need nonempty references and no spaces"
                )
            handle.write(" ".join(fields) + "\n")
