#!/usr/bin/env python3
"""Evaluate generated de novo SDFs, as in Morph's eval_sdf_metrics.py.

Example:
    python scripts/evaluate_denovo.py --sdf generated.sdf \
        --n-attempts 10000 --output-json eval_outputs/denovo.json

See docs/evaluation.md for metric definitions and hydrogen/denominator policies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from morph.utils.metrics.denovo import (
    canonicalize_reference_smiles,
    compute_denovo_metrics,
    load_sdf,
)


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--stability-sdf",
        type=Path,
        nargs="+",
        help="Raw graphs for every attempt, including sanitization failures. "
        "Loaded without sanitizing or removing H for RDKit stability only.",
    )
    parser.add_argument(
        "--reference-smiles",
        type=Path,
        help="Training SMILES, one per line, for novelty and combined novel rates.",
    )
    parser.add_argument(
        "--smiles-stereochemistry",
        choices=["ignore", "retain"],
        default="ignore",
        help=(
            "Identity for uniqueness/novelty: ignore stereo to match Morph's "
            "generated graphs (default), or retain it to distinguish stereoisomers."
        ),
    )
    parser.add_argument(
        "--n-attempts",
        type=_positive_int,
        help="Total attempts across all files; omitted outputs count as failures.",
    )
    parser.add_argument("--allow-charged", action="store_true")
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Keep parseable but unsanitizable graphs for raw chemistry diagnostics.",
    )
    parser.add_argument(
        "--remove-hs",
        action="store_true",
        help="Remove explicit H atoms on load (default: preserve the SDF).",
    )
    affinity = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    parser.add_argument("--workers", type=_positive_int, default=min(4, affinity))
    parser.add_argument("--no-energy", action="store_true", help="Skip MMFF metrics.")
    parser.add_argument(
        "--energy-hydrogens",
        choices=["preserve", "add"],
        default="preserve",
        help="Use SDF atoms for MMFF (default), or add H coordinates on a copy.",
    )
    parser.add_argument("--max-iters", type=_positive_int, default=1000)
    parser.add_argument(
        "--posebusters-energy-timeout",
        type=_positive_int,
        default=180,
        help="Wall-time limit per PoseBusters energy check, in seconds.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Save and reuse completed energy/PoseBusters chunks.",
    )
    parser.add_argument(
        "--no-posebusters",
        action="store_true",
        help="Skip full PoseBusters mol checks, including its internal energy test.",
    )
    parser.add_argument(
        "--no-diversity",
        action="store_true",
        help="Skip the O(N squared) pairwise ECFP6 diversity computation.",
    )
    return parser


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv=None):
    """Score pooled SDF records and write a strict JSON report with provenance."""
    parser = _parser()
    args = parser.parse_args(argv)
    input_paths = [*args.sdf, *(args.stability_sdf or [])]
    if args.reference_smiles is not None:
        input_paths.append(args.reference_smiles)
    if args.output_json.resolve() in {path.resolve() for path in input_paths}:
        parser.error("--output-json must not overwrite an input file")
    reference_smiles, reference = None, None
    if args.reference_smiles is not None:
        raw = args.reference_smiles.read_bytes()
        lines = [line.strip() for line in raw.decode().splitlines() if line.strip()]
        try:
            reference_smiles = canonicalize_reference_smiles(
                lines, smiles_stereochemistry=args.smiles_stereochemistry
            )
        except ValueError as exc:
            parser.error(str(exc))
        reference = {
            "path": str(args.reference_smiles.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "n_entries": len(lines),
            "n_unique_canonical": len(reference_smiles),
        }
    mols, inputs, stability_mols = [], [], []
    for path in args.sdf:
        loaded = load_sdf(path, sanitize=not args.no_sanitize, remove_hs=args.remove_hs)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        inputs.append(
            {
                "path": str(path.resolve()),
                "sha256": digest,
                "n_records": len(loaded),
                "n_loaded": sum(mol is not None for mol in loaded),
            }
        )
        mols.extend(loaded)
        if args.stability_sdf is None:
            stability_mols.extend(load_sdf(path, sanitize=False))
        print(f"Loaded {len(loaded)} records from {path}", flush=True)

    n_records = len(mols)
    n_attempts = args.n_attempts if args.n_attempts is not None else n_records
    if n_attempts < n_records:
        parser.error("--n-attempts cannot be smaller than the number of SDF records")
    if n_attempts == 0:
        parser.error(
            "No SDF records; supply --n-attempts only for known failed attempts"
        )
    mols.extend([None] * (n_attempts - n_records))
    stability_inputs = []
    if args.stability_sdf is not None:
        for path in args.stability_sdf:
            loaded = load_sdf(path, sanitize=False)
            stability_mols.extend(loaded)
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            stability_inputs.append(
                {
                    "path": str(path.resolve()),
                    "sha256": digest,
                    "n_records": len(loaded),
                }
            )
        if len(stability_mols) != n_attempts:
            parser.error("--stability-sdf must contain one record for every attempt")
    else:
        stability_mols.extend([None] * (n_attempts - n_records))
    notes = []
    if any(mol is None for mol in stability_mols):
        notes.append(
            "RDKit atom stability is undefined: raw graphs are missing "
            "for some attempts. Supply --stability-sdf with every graph."
        )
    if args.n_attempts is None:
        notes.append(
            "Rates use SDF records only. Morph's SDF exporter skips failures; "
            "supply --n-attempts to include known omitted generation failures."
        )
    elif n_attempts > n_records:
        notes.append("Every attempt missing from the SDFs is counted as a failure.")
    if reference is None:
        notes.append(
            "Novelty and novel intersections omitted: "
            "no training SMILES reference supplied."
        )
    print("Evaluating chemistry, geometry and enabled energy checks...", flush=True)
    report = compute_denovo_metrics(
        mols,
        rdkit_stability_mols=stability_mols,
        allow_charged=args.allow_charged,
        run_energy=not args.no_energy,
        run_posebusters=not args.no_posebusters,
        run_diversity=not args.no_diversity,
        workers=args.workers,
        energy_hydrogens=args.energy_hydrogens,
        max_iters=args.max_iters,
        reference_smiles=reference_smiles,
        smiles_stereochemistry=args.smiles_stereochemistry,
        posebusters_energy_timeout=args.posebusters_energy_timeout,
        checkpoint_dir=args.checkpoint_dir,
    )
    report["inputs"] = inputs
    report["rdkit_stability_inputs"] = stability_inputs or inputs
    report["novelty_reference"] = reference
    index = 0
    for source in inputs:
        for record_index in range(source["n_records"]):
            report["molecule_results"][index].update(
                {"source_path": source["path"], "sdf_record_index": record_index}
            )
            index += 1
    for row in report["molecule_results"][n_records:]:
        row.update({"source_path": None, "sdf_record_index": None})
    report["notes"] = notes
    report["counts"].update(
        {
            "n_sdf_records": n_records,
            "n_loaded": sum(item["n_loaded"] for item in inputs),
            "n_failed_load": sum(
                item["n_records"] - item["n_loaded"] for item in inputs
            ),
            "n_omitted_attempts": n_attempts - n_records,
        }
    )
    report["settings"].update(
        {
            "sanitize_on_load": not args.no_sanitize,
            "remove_hs_on_load": args.remove_hs,
            "denominator_source": (
                "explicit_n_attempts" if args.n_attempts is not None else "sdf_records"
            ),
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n"
    )
    for name, value in sorted(report["results"].items()):
        print(f"{name:60s} {value:.6g}")
    for note in notes:
        print(note)
    print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
