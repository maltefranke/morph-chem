"""SDF-based de novo evaluation adapted from Morph's eval_sdf_metrics.py.

Retains the original chemistry, MMFF94 and stock PoseBusters ``mol`` coverage.
Chemistry checks use this repository's corrected stability flags and hydrogen
policy. Failed records stay in rate denominators; finite-only averages expose
their sample counts. No reference dataset, checkpoint or GPU is required.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from morph.utils import rdkit_utils as morphRD
from morph.utils.metrics.chemistry import (
    Diversity,
    SanitizationValidity,
    ValenceTableAtomStability,
    ValenceTableMoleculeStability,
    Validity,
    calc_atom_stabilities,
)
from morph.utils.metrics.posebusters_energy import EnergyRatioWorker

if TYPE_CHECKING:
    from collections.abc import Collection


def _validate_stereochemistry(stereochemistry: str) -> None:
    if stereochemistry not in {"ignore", "retain"}:
        raise ValueError("smiles_stereochemistry must be 'ignore' or 'retain'")


def _identity_smiles(mol, *, stereochemistry: str) -> str | None:
    if mol is None:
        return None
    identity = Chem.Mol(mol)
    if stereochemistry == "ignore":
        # SDF loading can infer stereo from coordinates that was never present
        # on training-time generated graphs. Strip it on an identity-only copy;
        # geometry checks must still see the original molecule and stereotags.
        # Unlike isomericSmiles=False, this preserves isotope identities.
        Chem.RemoveStereochemistry(identity)
    return morphRD.smiles_from_mol(identity, canonical=True)


def canonicalize_reference_smiles(
    smiles: Collection[str], *, smiles_stereochemistry: str = "ignore"
) -> set[str]:
    """Normalize a training reference in the generated molecules' SMILES space.

    Remove hydrogens and, by default, stereochemistry to match Morph's
    generated graphs. Use the same stereo policy for evaluation. Reject invalid
    entries instead of silently inflating novelty.
    """
    _validate_stereochemistry(smiles_stereochemistry)
    canonical = set()
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        smi = _identity_smiles(mol, stereochemistry=smiles_stereochemistry)
        if not smi:
            raise ValueError(
                f"Invalid reference SMILES at entry {index + 1}: {value!r}"
            )
        canonical.add(smi)
    if not canonical:
        raise ValueError("The novelty reference must contain at least one molecule")
    return canonical


def load_sdf(
    path: str | Path, *, sanitize: bool = True, remove_hs: bool = False
) -> list[Chem.Mol | None]:
    """Load SDF records in order, retaining RDKit failures as ``None``.

    Hydrogens and their implicit/explicit policy are preserved by default.
    ``remove_hs`` removes explicit H atoms after successful sanitization.
    Empty files yield no records; missing or unreadable files raise an error.
    """
    path = Path(path)
    if path.stat().st_size == 0:
        return []
    # list(SDMolSupplier(...)) asks for a length hint, which first indexes the
    # file. RDKit 2025.09.6 can misplace record offsets at 64 KiB boundaries,
    # prepend a newline, and reject a perfectly valid record's counts line.
    # The forward reader avoids that indexing pass and preserves properties.
    with path.open("rb") as handle:
        mols = list(
            Chem.ForwardSDMolSupplier(handle, sanitize=sanitize, removeHs=False)
        )
    for index, mol in enumerate(mols):
        if mol is None:
            continue
        try:
            # Raw loading still needs a property cache for total-valence checks.
            mol.UpdatePropertyCache(strict=False)
            if remove_hs:
                mols[index] = Chem.RemoveHs(mol)
        except Exception:
            mols[index] = None
    return mols


def _finite(value):
    return value is not None and math.isfinite(value)


def _energy_record(mol, *, add_hs, max_iters):
    values = dict.fromkeys(
        ["energy", "energy-per-atom", "strain", "strain-per-atom", "opt-rmsd"]
    )
    values.update({"opt-energy": None, "optimization_status": None})
    if mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
        return values
    try:
        original = Chem.Mol(mol)
        if add_hs:
            original = Chem.AddHs(original, addCoords=True)
        n_atoms = original.GetNumAtoms()
        energy = morphRD.calc_energy(original)
        if _finite(energy):
            values["energy"] = energy
            values["energy-per-atom"] = energy / n_atoms
        optimized = Chem.Mol(original)
        status = AllChem.MMFFOptimizeMolecule(
            optimized, mmffVariant="MMFF94", maxIters=max_iters
        )
        values["optimization_status"] = status
        if status < 0:
            return values
        opt_energy = morphRD.calc_energy(optimized)
        if _finite(opt_energy):
            values["opt-energy"] = opt_energy
            if _finite(energy):
                values["strain"] = energy - opt_energy
                values["strain-per-atom"] = (energy - opt_energy) / n_atoms
        rmsd = morphRD.conf_distance(original, optimized)
        if _finite(rmsd):
            values["opt-rmsd"] = float(rmsd)
    except Exception:
        # Unsupported chemistry/poses are counted, never removed from rates.
        # Retain an input energy even if subsequent optimization failed.
        pass
    return values


def _call_named_chunk(function, chunk, names):
    # RDKit's default multiprocessing pickle omits molecular properties,
    # including _Name. Preserve IDs explicitly for per-molecule diagnostics.
    for mol, name in zip(chunk, names, strict=True):
        if name is not None:
            mol.SetProp("_Name", name)
    return function(chunk)


def _map_chunks(function, chunks, workers, *, cache_dir=None):
    results, pending = [None] * len(chunks), []
    cache_paths = {}
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        source = (
            Path(__file__).read_bytes()
            + Path(__file__).with_name("posebusters_energy.py").read_bytes()
            + Path(morphRD.__file__).read_bytes()
        )
        for index, chunk in enumerate(chunks):
            payload = (
                function,
                rdBase.rdkitVersion,
                version("posebusters"),
                source,
                [m.ToBinary(Chem.PropertyPickleOptions.AllProps) for m in chunk],
            )
            digest = hashlib.sha256(pickle.dumps(payload)).hexdigest()
            path = cache_dir / f"{index:05d}-{digest}.pkl"
            cache_paths[index] = path
            if path.exists():
                results[index] = pickle.loads(path.read_bytes())
            else:
                pending.append(index)
    else:
        pending = list(range(len(chunks)))
    stage = getattr(function, "func", function).__name__
    completed = len(chunks) - len(pending)
    print(f"{stage}: {completed}/{len(chunks)} chunks restored", flush=True)

    def record(index, result):
        nonlocal completed
        results[index] = result
        if index in cache_paths:
            path = cache_paths[index]
            temporary = path.with_suffix(".pending")
            temporary.write_bytes(pickle.dumps(result))
            temporary.replace(path)
        completed += 1
        print(f"{stage}: {completed}/{len(chunks)} chunks complete", flush=True)

    if workers == 1:
        for index in pending:
            record(index, function(chunks[index]))
        return results
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = {
            executor.submit(
                _call_named_chunk,
                function,
                chunks[i],
                [m.GetProp("_Name") if m.HasProp("_Name") else None for m in chunks[i]],
            ): i
            for i in pending
        }
        for future in as_completed(futures):
            record(futures[future], future.result())
    return results


def _energy_chunk(mols, *, add_hs, max_iters):
    return [_energy_record(mol, add_hs=add_hs, max_iters=max_iters) for mol in mols]


def _chunks(mols, workers):
    size = max(1, math.ceil(len(mols) / (workers * 4)))
    return [mols[start : start + size] for start in range(0, len(mols), size)]


def _energy_metrics(mols, *, workers, add_hs, max_iters, cache_dir=None):
    chunks = _chunks([m for m in mols if m is not None], workers)
    function = partial(_energy_chunk, add_hs=add_hs, max_iters=max_iters)
    records = [
        r
        for batch in _map_chunks(function, chunks, workers, cache_dir=cache_dir)
        for r in batch
    ]
    results, counts = {}, {}
    for key in (
        "energy",
        "energy-per-atom",
        "strain",
        "strain-per-atom",
        "opt-rmsd",
        "opt-energy",
    ):
        values = [r[key] for r in records if _finite(r[key])]
        results[key] = math.fsum(values) / len(values) if values else float("nan")
        counts[key] = len(values)
    results["energy-validity"] = counts["energy"] / len(mols)
    results["opt-energy-validity"] = counts["opt-energy"] / len(mols)
    counts["optimization-converged"] = sum(
        r["optimization_status"] == 0 for r in records
    )
    counts["optimization-not-converged"] = sum(
        r["optimization_status"] == 1 for r in records
    )
    counts["optimization-failed"] = (
        len(mols)
        - counts["optimization-converged"]
        - counts["optimization-not-converged"]
    )
    return results, counts


def _flatness_coordinates(mol, indices):
    """Read coordinates without NumPy converting RDKit Point3D objects.

    PoseBusters 0.6.5's Point3D conversion can segfault with the installed
    RDKit/Python/NumPy stack. GetPositions returns the same coordinates as a
    numeric array, preserving atom order and the original flatness calculation.
    """
    return mol.GetConformer().GetPositions()[list(indices)]


def _posebusters_config():
    from posebusters import PoseBusters  # noqa: PLC0415

    config = PoseBusters(config="mol", max_workers=0).config
    # The stock config repeats the same energy calculation, first without any
    # selected output. Failed ensembles aren't cached by PoseBusters, causing
    # an expensive second attempt. Keep the identical scored entry once.
    config["modules"] = [
        module
        for module in config["modules"]
        if module["function"] != "energy_ratio"
        or module.get("chosen_binary_test_output")
    ]
    for module in config["modules"]:
        if module["function"] == "energy_ratio":
            # Parallelism is across molecules. RDKit's default num_threads=0
            # would let every worker launch an additional machine-wide pool.
            module.setdefault("parameters", {})["num_threads"] = 1
    return config


def _posebusters_chunk(mols, *, energy_timeout=180):
    from posebusters import PoseBusters  # noqa: PLC0415

    # Match the original offline evaluator, including flatness and energy.
    # Chunk outside PoseBusters to avoid repeatedly pickling the entire set.
    with (
        EnergyRatioWorker(energy_timeout) as energy,
        patch("posebusters.modules.flatness._get_coords", _flatness_coordinates),
        patch.dict(
            "posebusters.posebusters.module_dict", {"energy_ratio": energy.check}
        ),
    ):
        frame = PoseBusters(config=_posebusters_config(), max_workers=0).bust(
            mols, None, None
        )
        frame.attrs["energy_timeout_names"] = energy.timed_out_names
        return frame


def _posebusters_checks(frame):
    return frame.drop(columns=["file", "molecule"], errors="ignore").apply(
        pd.to_numeric, errors="coerce"
    )


def _posebusters_passes(frame):
    checks = _posebusters_checks(frame)
    if not len(checks.columns):
        return [False] * len(frame)
    return checks.eq(1).all(axis=1).tolist()


def _summarize_posebusters(frame, n_attempts, n_busted):
    if len(frame) != n_busted:
        raise ValueError("PoseBusters must return one row per supplied molecule")
    results = {"n_generated": n_attempts, "n_busted": n_busted}
    checks = _posebusters_checks(frame)
    for name in checks:
        column = checks[name]
        results[name] = float(column.mean())
        results[f"{name}__n_eval"] = int(column.notna().sum())
        results[f"{name}__n_nan"] = int(column.isna().sum())
        results[f"{name}__over_generated"] = float(column.fillna(0).sum()) / n_attempts
    if len(checks.columns):
        passed = checks.eq(1).all(axis=1)
        n_complete = int(checks.notna().all(axis=1).sum())
        n_passed = int(passed.sum())
    else:
        n_complete, n_passed = 0, 0
    results.update(
        {
            "n_checks": len(checks.columns),
            "n_complete": n_complete,
            "n_incomplete": n_busted - n_complete,
            "n_all_pass": n_passed,
            "all_pass_over_generated": n_passed / n_attempts,
            "all_pass_over_busted": (n_passed / n_busted if n_busted else float("nan")),
            "validity": n_passed / n_attempts,
        }
    )
    return results


def _combined_metrics(
    mols,
    *,
    allow_charged,
    reference_smiles,
    pb_passes,
    smiles_stereochemistry="ignore",
):
    """Intersect per-molecule predicates, then deduplicate each passing subset."""
    if pb_passes is not None and len(pb_passes) != len(mols):
        raise ValueError("PoseBusters flags must align with every attempted molecule")
    records = []
    for index, mol in enumerate(mols):
        smi = _identity_smiles(mol, stereochemistry=smiles_stereochemistry)
        records.append(
            {
                "input_index": index,
                "name": mol.GetProp("_Name")
                if mol is not None and mol.HasProp("_Name")
                else None,
                "canonical_smiles": smi,
                "valid": bool(morphRD.mol_is_valid(mol, allow_charged=allow_charged)),
                "novel": smi not in reference_smiles
                if reference_smiles is not None and smi is not None
                else None,
                "posebusters_valid": bool(pb_passes[index])
                if pb_passes is not None
                else None,
            }
        )
    valid = [row for row in records if row["valid"]]
    valid_smiles = [
        row["canonical_smiles"] for row in valid if row["canonical_smiles"] is not None
    ]
    convertible = [row for row in records if row["canonical_smiles"] is not None]
    all_smiles = [row["canonical_smiles"] for row in convertible]
    counts = {
        "n_smiles": len(all_smiles),
        "n_valid": len(valid),
        "n_valid_smiles": len(valid_smiles),
    }
    numerators = {"valid-unique": len(set(valid_smiles))}
    results = {
        "uniqueness": len(set(all_smiles)) / len(all_smiles)
        if all_smiles
        else float("nan")
    }
    if reference_smiles is not None:
        novel = [row for row in convertible if row["novel"]]
        valid_novel = [row for row in valid if row["novel"]]
        counts.update(
            {"n_novel_smiles": len(novel), "n_reference_smiles": len(reference_smiles)}
        )
        results["novelty"] = (
            len(novel) / len(convertible) if convertible else float("nan")
        )
        results["novelty-among-valid"] = (
            len(valid_novel) / len(valid_smiles) if valid_smiles else float("nan")
        )
        numerators["valid-novel"] = len(valid_novel)
        numerators["valid-unique-novel"] = len(
            {row["canonical_smiles"] for row in valid_novel}
        )
    if pb_passes is not None:
        valid_pb = [row for row in valid if row["posebusters_valid"]]
        numerators["valid-posebusters"] = len(valid_pb)
        if reference_smiles is not None:
            numerators["valid-unique-novel-posebusters"] = len(
                {row["canonical_smiles"] for row in valid_pb if row["novel"]}
            )
    results.update({name: count / len(mols) for name, count in numerators.items()})
    counts["combined_numerators"] = numerators
    return results, counts, records


def summarize_rdkit_stability(stabilities: list[list[bool] | None]) -> dict:
    """Summarize raw-graph flags without hiding attempts with unknown atoms.

    ``None`` means the raw graph could not be evaluated. In that case the
    overall atom denominator is unknown, so atom stability is undefined.
    Molecule stability counts such attempts as failures. This is a chemistry
    problem check, not a geometric, radical, or connectivity check.
    """
    if not stabilities:
        raise ValueError("No attempted molecules to evaluate")
    known = [flags for flags in stabilities if flags is not None]
    n_atoms = sum(map(len, known))
    n_stable = sum(sum(flags) for flags in known)
    missing = len(stabilities) - len(known)
    return {
        "results": {
            "rdkit-atom-stability": (
                n_stable / n_atoms if n_atoms and not missing else float("nan")
            ),
            "rdkit-molecule-stability": (
                sum(bool(flags) and all(flags) for flags in known) / len(stabilities)
            ),
        },
        "counts": {
            "n_rdkit_stability_atoms": n_atoms,
            "n_rdkit_stable_atoms": n_stable,
            "n_rdkit_unstable_atoms": n_atoms - n_stable,
            "n_rdkit_stability_molecules": sum(bool(flags) for flags in known),
            "n_rdkit_stability_missing_graphs": missing,
        },
    }


def compute_denovo_metrics(
    mols: list[Chem.Mol | None],
    *,
    rdkit_stability_mols: list[Chem.Mol | None] | None = None,
    allow_charged: bool = True,
    run_energy: bool = True,
    run_posebusters: bool = True,
    run_diversity: bool = True,
    workers: int = 1,
    energy_hydrogens: str = "preserve",
    max_iters: int = 1000,
    reference_smiles: Collection[str] | None = None,
    smiles_stereochemistry: str = "ignore",
    posebusters_energy_timeout: float = 180,
    checkpoint_dir: str | Path | None = None,
) -> dict:
    """Evaluate the de novo SDF metric stack from the original Morph repo.

    ``mols`` must include every attempted output, with ``None`` for failures.
    Rates are fractions; energies use kcal/mol and RMSD uses Angstrom. Undefined
    averages are NaN. Counts expose the populations used by averages.
    Supply unsanitized graphs for every attempt in ``rdkit_stability_mols``.
    Otherwise stability checks use copies of ``mols`` before other metrics.
    Missing graphs make atom stability undefined rather than survivor-biased.

    MMFF optimizes one copy per molecule, retaining finite nonconverged results
    as the original evaluator did, but recording convergence separately.
    ``energy_hydrogens='add'`` adds H coordinates on an energy-only copy; the
    default ``'preserve'`` uses exactly the SDF atoms, matching the original.
    All atoms in that copy enter per-atom normalization and alignment RMSD.

    PoseBusters uses the full installed ``mol`` config, unlike the cheaper
    training checks. Exceptions propagate instead of silently dropping checks.
    Diversity is the repository's ECFP6 pairwise Tanimoto distance (O(N squared)).
    Novelty requires a training reference normalized with
    ``canonicalize_reference_smiles`` with the same ``smiles_stereochemistry``.
    Identity ignores stereo by default, matching Morph's generated graphs;
    ``'retain'`` distinguishes stereoisomers. Neither policy changes geometry.
    Combined rates use every attempt as their
    denominator, deduplicating only after applying all requested predicates.
    For a molecular identity with several poses, any passing pose qualifies.
    """
    if not mols:
        raise ValueError("No attempted molecules to evaluate")
    stability_inputs = mols if rdkit_stability_mols is None else rdkit_stability_mols
    if len(stability_inputs) != len(mols):
        raise ValueError("rdkit_stability_mols must contain every attempt")
    stabilities = []
    for mol in stability_inputs:
        try:
            stabilities.append(calc_atom_stabilities(mol) if mol is not None else None)
        except Exception:
            stabilities.append(None)
    stability = summarize_rdkit_stability(stabilities)
    if workers < 1 or max_iters < 1:
        raise ValueError("workers and max_iters must be positive")
    if posebusters_energy_timeout <= 0:
        raise ValueError("posebusters_energy_timeout must be positive")
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if energy_hydrogens not in {"preserve", "add"}:
        raise ValueError("energy_hydrogens must be 'preserve' or 'add'")
    _validate_stereochemistry(smiles_stereochemistry)
    if reference_smiles is not None:
        reference_smiles = set(reference_smiles)
        if not reference_smiles:
            raise ValueError("The novelty reference must contain at least one molecule")

    metric_types = {
        "validity": partial(Validity, allow_charged=allow_charged),
        "connected-sanitization-validity": SanitizationValidity,
        "valence-table-atom-stability": ValenceTableAtomStability,
        "valence-table-molecule-stability": ValenceTableMoleculeStability,
    }
    if run_diversity:
        metric_types["diversity"] = Diversity
    results, counts = {}, {"n_attempts": len(mols)}
    for name, metric_type in metric_types.items():
        metric = metric_type(sync_on_compute=False)
        metric.update(mols)
        results[name] = float(metric.compute())
        if name == "diversity":
            counts["n_fingerprints"] = len(metric.fps)
        elif name == "valence-table-atom-stability":
            counts["n_valence_table_atoms"] = int(metric.total)

    results.update(stability["results"])
    counts.update(stability["counts"])

    if run_energy:
        energies, energy_counts = _energy_metrics(
            mols,
            workers=workers,
            add_hs=energy_hydrogens == "add",
            max_iters=max_iters,
            cache_dir=checkpoint_dir / "energy" if checkpoint_dir is not None else None,
        )
        results.update(energies)
        counts["energy"] = energy_counts

    pb_config = None
    pb_passes = None
    if run_posebusters:
        pb_config = _posebusters_config()
        non_none = [m for m in mols if m is not None]
        function = partial(
            _posebusters_chunk, energy_timeout=posebusters_energy_timeout
        )
        # Small work units prevent a few hard conformers from monopolizing
        # the last worker and provide frequent durable checkpoints.
        pb_chunks = [
            chunk[start : start + 32]
            for chunk in _chunks(non_none, workers)
            for start in range(0, len(chunk), 32)
        ]
        frames = _map_chunks(
            function,
            pb_chunks,
            workers,
            cache_dir=checkpoint_dir / "posebusters"
            if checkpoint_dir is not None
            else None,
        )
        counts["posebusters_energy_timeout_names"] = [
            name
            for frame in frames
            for name in frame.attrs.get("energy_timeout_names", [])
        ]
        counts["n_posebusters_energy_timeouts"] = len(
            counts["posebusters_energy_timeout_names"]
        )
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        results.update(
            {
                f"posebusters/{key}": value
                for key, value in _summarize_posebusters(
                    frame, len(mols), len(non_none)
                ).items()
            }
        )
        # Preserve correspondence through failed/omitted records. Pool mapping
        # and concatenation retain the original non-None molecule order.
        non_none_passes = iter(_posebusters_passes(frame))
        pb_passes = [
            next(non_none_passes) if mol is not None else False for mol in mols
        ]
    combined, combined_counts, records = _combined_metrics(
        mols,
        allow_charged=allow_charged,
        reference_smiles=reference_smiles,
        pb_passes=pb_passes,
        smiles_stereochemistry=smiles_stereochemistry,
    )
    results.update(combined)
    counts.update(combined_counts)
    return {
        "protocol": "morph_denovo_sdf",
        "results": results,
        "counts": counts,
        "molecule_results": records,
        "settings": {
            "rdkit_stability_input": (
                "separate_unsanitized_graphs"
                if rdkit_stability_mols is not None
                else "provided_molecules"
            ),
            "rdkit_stability_missing_graph_policy": "atom_rate_undefined",
            "allow_charged": allow_charged,
            "run_energy": run_energy,
            "energy_hydrogens": energy_hydrogens,
            "force_field": "MMFF94",
            "max_iters": max_iters,
            "run_posebusters": run_posebusters,
            "posebusters_energy_timeout_seconds": posebusters_energy_timeout,
            "checkpoint_dir": str(checkpoint_dir)
            if checkpoint_dir is not None
            else None,
            "posebusters_config": pb_config,
            "posebusters_flatness_coordinates": (
                "rdkit_getpositions" if run_posebusters else None
            ),
            "run_diversity": run_diversity,
            "novelty_reference_provided": reference_smiles is not None,
            "smiles_stereochemistry": smiles_stereochemistry,
            "smiles_identity": (
                "canonical, H-stripped, stereochemistry "
                + ("ignored" if smiles_stereochemistry == "ignore" else "retained")
            ),
            "combined_denominator": "all_attempts",
            "combined_uniqueness": (
                "distinct SMILES after filtering; any qualifying pose counts"
            ),
            "workers": workers,
        },
        "versions": {
            "rdkit": rdBase.rdkitVersion,
            "posebusters": version("posebusters") if run_posebusters else None,
        },
    }
