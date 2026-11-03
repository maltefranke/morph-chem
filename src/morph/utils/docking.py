"""Bounded, resumable QuickVina execution using the released DiffDec protocol.

Protocol source: biomed-AI/DiffDec revision
916ae14207b2783a90336bb8509374535c5791f9, vina_docking.py and vina_preprocess.py.
Upstream copyright: 2024 Biomed-AI@NSCC-GZ. The MIT notice is retained in
morph/_vendor/licenses/DIFFDEC_LICENSE.
Embedding seeds are explicit here for reproducibility;
the original script left them unspecified. Modern RDKit and reconstructed reference
strings also prevent a claim of bitwise reproduction of the published results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

SAMPLES = 100
PROTOCOL = {
    "box_angstrom": 30,
    "exhaustiveness": 16,
    "docking_seed": 1,
    "cpu_per_ligand": 1,
    "embedding": "RDKit default EmbedMolecule with an explicit per-attempt seed",
    "uff_max_calls": 10,
    "invalid_prediction": "replace with scaffold before docking",
    "pose_source": "re-embedded prediction SMILES, not generated coordinates",
}


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _command(command, cwd, timeout=120):
    with (Path(cwd) / "commands.log").open("a") as log:
        log.write(json.dumps(list(map(str, command))) + "\n")
        log.flush()
        subprocess.run(
            command, cwd=cwd, stdout=log, stderr=log, check=True, timeout=timeout
        )


def prepare_receptor(source, directory, tools):
    """Prepare an unmodified pocket PDB and its original atom bounding-box center."""
    import numpy as np  # noqa: PLC0415
    from Bio.PDB import PDBParser  # noqa: PLC0415

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(source).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    result_path = directory / "receptor.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        if cached["sha256"] != digest:
            raise ValueError(f"Receptor changed during evaluation: {source}")
        return cached
    shutil.copyfile(source, directory / "receptor.pdb")
    structure = PDBParser(QUIET=True).get_structure("", str(source))
    xyz = np.array([atom.get_coord() for atom in structure.get_atoms()])
    center = (xyz.max(0) + xyz.min(0)) / 2
    _command(
        [tools["prepare_receptor4.py"], "-r", "receptor.pdb", "-o", "receptor.pdbqt"],
        directory,
    )
    pdbqt = directory / "receptor.pdbqt"
    if not pdbqt.is_file() or not pdbqt.stat().st_size:
        raise RuntimeError(f"Receptor preparation produced no PDBQT for {source}")
    result = {
        "source": str(source),
        "sha256": digest,
        "pdbqt": str(pdbqt.resolve()),
        "center": center.tolist(),
    }
    _json(result_path, result)
    return result


def dock_ligand(task, directory):
    """Embed, UFF-minimize, dock, and read the first parseable SDF mode."""
    mol = Chem.MolFromSmiles(task["smiles"])
    if mol is None:
        mol = Chem.MolFromSmiles(task["smiles"], sanitize=False)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("No parseable ligand")
    mol = Chem.AddHs(mol, addCoords=True)
    if AllChem.EmbedMolecule(mol, randomSeed=task["embedding_seed"]) != 0:
        raise ValueError("RDKit embedding failed")
    uff_converged = False
    try:
        for _ in range(PROTOCOL["uff_max_calls"]):
            if AllChem.UFFOptimizeMolecule(mol) == 0:
                uff_converged = True
                break
    except RuntimeError:
        pass  # Released DiffDec still docks this conformation.
    with Chem.SDWriter(str(directory / "ligand.sdf")) as writer:
        writer.write(mol)
    tools = task["tools"]
    _command([tools["obabel"], "ligand.sdf", "-Oligand.pdbqt"], directory)
    command = [
        tools["qvina2"],
        "--receptor",
        task["receptor"]["pdbqt"],
        "--ligand",
        "ligand.pdbqt",
        "--out",
        "ligand_out.pdbqt",
        "--size_x",
        "30",
        "--size_y",
        "30",
        "--size_z",
        "30",
        "--exhaustiveness",
        "16",
        "--cpu",
        "1",
        "--seed",
        "1",
    ]
    for axis, coordinate in zip("xyz", task["receptor"]["center"], strict=True):
        command.extend([f"--center_{axis}", f"{coordinate:.4f}"])
    _command(command, directory, timeout=task["timeout"])
    _command([tools["obabel"], "ligand_out.pdbqt", "-Oligand_out.sdf", "-h"], directory)
    poses = Chem.SDMolSupplier(str(directory / "ligand_out.sdf"))
    for pose in poses:
        if pose is None:
            continue
        affinity = float(pose.GetProp("REMARK").splitlines()[0].split()[2])
        if not math.isfinite(affinity):
            raise ValueError("Nonfinite docking affinity")
        return {"status": "ok", "affinity": affinity, "uff_converged": uff_converged}
    raise ValueError("No parseable docking poses")


def bounded_dock(task, scratch=None):
    """Bound the whole ligand task and terminate its subprocess tree on timeout."""
    result = {key: task[key] for key in ("kind", "i", "embedding_seed", "replacement")}
    start = time.monotonic()
    if "error" in task["receptor"]:
        return {
            **result,
            "status": "receptor_failed",
            "affinity": None,
            "error": task["receptor"]["error"],
            "seconds": 0,
        }
    with tempfile.TemporaryDirectory(prefix="morph-dock-", dir=scratch) as temporary:
        directory = Path(temporary)
        _json(directory / "task.json", task)
        with (directory / "worker.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "morph.utils.docking",
                    "--worker",
                    str(directory),
                ],
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=task["timeout"])
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                result.update(
                    status="timeout", affinity=None, error="Ligand task timed out"
                )
            else:
                if returncode == 0 and (directory / "result.json").exists():
                    result.update(json.loads((directory / "result.json").read_text()))
                else:
                    result.update(
                        status="failed",
                        affinity=None,
                        error=f"Worker exit {returncode}",
                    )
            finally:
                # Also reap a native child if a worker-side timeout interrupted it.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
        if result["status"] != "ok":
            result["log_tail"] = "\n".join(
                p.read_text(errors="replace")[-6000:]
                for p in (directory / "worker.log", directory / "commands.log")
                if p.exists()
            )
    return {**result, "seconds": time.monotonic() - start}


def aggregate_docking(records, results):
    """Apply DiffDec's group weighting and missing-reference denominator behavior.

    Successful predictions with missing references are skipped by high-affinity;
    failed predictions still count in its denominator, as in the released script.
    Empty docking results are treated as failures instead of indexing an empty list.
    Every reference and prediction must have an explicit result (failure or success).
    """
    if not records or len(records) % SAMPLES:
        raise ValueError("Docking requires complete 100-attempt condition blocks")
    predictions, references = {}, {}
    for result in results:
        if result["kind"] not in ("prediction", "reference"):
            raise ValueError("Unknown docking result kind")
        target = predictions if result["kind"] == "prediction" else references
        if result["i"] in target:
            raise ValueError("Duplicate docking result")
        score = result.get("affinity")
        if result["status"] == "ok" and (score is None or not math.isfinite(score)):
            raise ValueError("Successful docking must have a finite affinity")
        target[result["i"]] = result
    if set(predictions) != set(range(len(records))) or set(references) != set(
        range(len(records) // SAMPLES)
    ):
        raise ValueError(
            "Missing or out-of-range docking results; refusing a partial summary"
        )
    scores, reference_scores = defaultdict(list), defaultdict(list)
    hits, totals = defaultdict(int), defaultdict(int)
    for i, row in enumerate(records):
        condition = i // SAMPLES
        first = records[condition * SAMPLES]
        if any(
            row[key] != first[key]
            for key in ("scaffold", "true_molecule", "protein_filename")
        ):
            raise ValueError("A 100-attempt block mixes benchmark conditions")
        key = row["true_molecule"] + row["protein_filename"]
        pred, ref = predictions[i], references[condition]
        if i % SAMPLES == 0 and ref["status"] == "ok":
            reference_scores[key].append(ref["affinity"])
        if pred["status"] == "ok":
            scores[key].append(pred["affinity"])
            if ref["status"] != "ok":
                continue
            hits[key] += pred["affinity"] <= ref["affinity"]
        totals[key] += 1

    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else None

    return {
        "vina_score": mean(mean(values) for values in scores.values()),
        "reference_vina_score": mean(
            mean(values) for values in reference_scores.values()
        ),
        "high_affinity": mean(hits[key] / count for key, count in totals.items()),
        "docking_attempts": len(predictions),
        "docking_successes": sum(r["status"] == "ok" for r in predictions.values()),
        "docking_failures": sum(r["status"] != "ok" for r in predictions.values()),
        "docking_timeouts": sum(r["status"] == "timeout" for r in predictions.values()),
        "reference_docking_attempts": len(references),
        "reference_docking_successes": sum(
            r["status"] == "ok" for r in references.values()
        ),
        "invalid_predictions_replaced": sum(
            r["replacement"] for r in predictions.values()
        ),
        "high_affinity_denominator": sum(totals.values()),
        "docking_scored_groups": len(scores),
    }


def run_shard(args):
    """Dock one deterministic shard, saving every success/failure as it completes."""
    from morph.utils.metrics.diffdec import (  # noqa: PLC0415
        diffdec_is_valid,
        read_diffdec_records,
    )

    records = read_diffdec_records(args.metrics_file)
    if len(records) % SAMPLES:
        raise ValueError("Expected 100 attempts per condition")
    tools = {
        name: shutil.which(name)
        for name in ("qvina2", "obabel", "prepare_receptor4.py")
    }
    if not all(tools.values()):
        raise RuntimeError(
            "Missing docking tools; run bash setup_env.sh --extra crossdocked"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_sha256": hashlib.sha256(args.metrics_file.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "receptor_root": str(args.receptor_root.resolve()),
        "seed": args.seed,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "timeout_seconds": args.timeout,
        "rdkit_version": rdBase.rdkitVersion,
        "protocol": PROTOCOL,
        "tools": tools,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Cannot resume docking with changed inputs/protocol")
    _json(manifest_path, manifest)
    result_path = output / "results.jsonl"
    done = set()
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            record = json.loads(line)
            key = (record["kind"], record["i"])
            if key in done:
                raise ValueError("Duplicate result in existing shard")
            done.add(key)
    tasks, receptors = [], {}
    for condition in range(args.shard_index, len(records) // SAMPLES, args.shard_count):
        block = records[condition * SAMPLES : (condition + 1) * SAMPLES]
        first = block[0]
        if any(
            any(
                row[k] != first[k]
                for k in ("scaffold", "true_molecule", "protein_filename")
            )
            for row in block
        ):
            raise ValueError("A 100-attempt block mixes benchmark conditions")
        filename = first["protein_filename"]
        source = Path(filename)
        if not source.is_absolute():
            source = args.receptor_root / filename.split("crossdocked_pocket10/")[-1]
        if not source.is_file():
            raise FileNotFoundError(source)
        if filename not in receptors:
            key = hashlib.sha256(filename.encode()).hexdigest()[:20]
            try:
                receptors[filename] = prepare_receptor(
                    source, output / "receptors" / key, tools
                )
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as exc:
                receptors[filename] = {"error": f"{type(exc).__name__}: {exc}"}
        for offset in range(-1, SAMPLES):
            kind = "reference" if offset == -1 else "prediction"
            i = condition if offset == -1 else condition * SAMPLES + offset
            if (kind, i) in done:
                continue
            row = first if offset == -1 else block[offset]
            replacement = kind == "prediction" and not diffdec_is_valid(
                row["pred_molecule"], row["scaffold"]
            )
            smiles = (
                row["true_molecule"]
                if kind == "reference"
                else row["scaffold"]
                if replacement
                else row["pred_molecule"]
            )
            seed_key = f"{args.seed}:{kind}:{i}"
            embedding_seed = (
                int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:4], "little")
                % 2147483647
            )
            tasks.append(
                {
                    "kind": kind,
                    "i": i,
                    "smiles": smiles,
                    "replacement": replacement,
                    "embedding_seed": embedding_seed,
                    "receptor": receptors[filename],
                    "tools": tools,
                    "timeout": args.timeout,
                }
            )
    with (
        result_path.open("a") as handle,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = [pool.submit(bounded_dock, task, args.scratch) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            handle.write(json.dumps(result, allow_nan=False) + "\n")
            handle.flush()
            if count % 25 == 0 or count == len(tasks):
                print(
                    f"Docked {len(done) + count} tasks ({count}/{len(tasks)} this run)",
                    flush=True,
                )
    _json(
        output / "complete.json",
        {"tasks": len(done) + len(tasks), "manifest": manifest},
    )


def main():
    """Run a docking shard or its isolated per-ligand worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--metrics-file", type=Path)
    parser.add_argument("--receptor-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()
    if args.worker:
        task = json.loads((args.worker / "task.json").read_text())
        try:
            result = dock_ligand(task, args.worker)
        except Exception as exc:
            result = {
                "status": "failed",
                "affinity": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        _json(args.worker / "result.json", result)
    else:
        if not all((args.metrics_file, args.receptor_root, args.output_dir)):
            parser.error(
                "--metrics-file, --receptor-root and --output-dir are required"
            )
        if (
            args.workers < 1
            or args.timeout < 1
            or not 0 <= args.shard_index < args.shard_count
        ):
            parser.error("Invalid worker, timeout, or shard settings")
        run_shard(args)


if __name__ == "__main__":
    main()
