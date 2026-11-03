#!/usr/bin/env python3
"""Run the non-docking scaffold-in-pocket evaluation used in DiffDec.

Score an existing DiffDec metric .smi file, or sample 100 attempts per condition
from a Morph checkpoint and its saved Hydra config. See docs/evaluation.md for
commands and the distinction between published DiffDec cuts and our own cuts.
Run ``python -m morph.utils.docking`` on the resulting metric file for docking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from rdkit import Chem, rdBase

from morph.utils.metrics.diffdec import (
    DIFFDEC_REVISION,
    DIFFDEC_SAMPLES_PER_CONDITION,
    compute_diffdec_metrics,
    read_diffdec_records,
    write_diffdec_records,
)


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def _file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def score_records(records, provenance):
    """Report the metrics used in DiffDec without silently changing denominators."""
    values = compute_diffdec_metrics(records)
    groups = Counter((r["protein_filename"], r["scaffold"]) for r in records)
    notes = []
    if any(size != DIFFDEC_SAMPLES_PER_CONDITION for size in groups.values()):
        notes.append(
            "Some protein/scaffold groups do not contain 100 attempts. DiffDec's "
            "uniqueness denominator remains 100, including merged duplicate "
            "benchmark conditions. Inspect group_sizes before comparing scores."
        )
    if any(not math.isfinite(value) for value in values.values()):
        notes.append(
            "No valid predictions: uniqueness and similarity are undefined "
            "(JSON null); the original DiffDec script divides by zero."
        )
    return {
        "protocol": "diffdec",
        "upstream_revision": DIFFDEC_REVISION,
        "rdkit_version": rdBase.rdkitVersion,
        "metrics": {
            key: value if math.isfinite(value) else None
            for key, value in values.items()
        },
        "units": {
            "validity": "percent",
            "uniqueness": "percent",
            "recovery": "percent",
            "similarity": "fraction",
        },
        "n_attempts": len(records),
        "n_proteins": len({r["protein_filename"] for r in records}),
        "n_scaffold_groups": len(groups),
        "group_sizes": [
            {"protein_filename": protein, "scaffold": scaffold, "n_attempts": size}
            for (protein, scaffold), size in groups.items()
        ],
        "notes": notes,
        "provenance": provenance,
        "docking": {
            "status": "not_run",
            "vina_score": None,
            "high_affinity": None,
        },
    }


def _build_task(args):
    # Keep checkpoint/training dependencies out of score-only execution.
    import hydra  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415

    from morph.inference.generate import (  # noqa: PLC0415
        _load_repo_env,
        load_checkpoint,
    )
    from morph.model.build import (  # noqa: PLC0415
        build_model_module,
        prepare_model_data,
    )
    from morph.utils.run_config import configure_runtime  # noqa: PLC0415

    _load_repo_env()
    configure_runtime(seed=args.seed)
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.set))
    OmegaConf.set_struct(cfg, False)
    if "defaults" in cfg:
        raise ValueError("--config must be a composed run/.hydra/config.yaml")
    if args.data_root:
        cfg.data.root = str(args.data_root.resolve())
    if args.assets_root:
        cfg.data.pretrained_assets_root = str(args.assets_root.resolve())
    cfg.model.module.pretrained_ckpt = None
    if args.steps is not None:
        cfg.model.module.integrator.num_integration_steps = args.steps
    if args.weights is not None:
        cfg.model.module.use_ema_for_eval = args.weights == "ema"
    expected = (
        "morph.dataset.conditional.pocket.scaffold_pocket.ScaffoldPocketDataModule"
    )
    if cfg.data.datamodule._target_ != expected:
        raise ValueError("--config must describe a scaffold-in-pocket model")
    pre = hydra.utils.instantiate(cfg.data.preprocessing)
    data = prepare_model_data(cfg, pre.vocab, pre.distributions)
    dataset_config = OmegaConf.create(
        OmegaConf.to_container(cfg.data.datamodule, resolve=True)
    )
    if args.diffdec_data:
        # Published atom indices replace cut enumeration entirely. In particular,
        # this path does not require LibINVENT or write a native scaffold cache.
        dataset_config._target_ = (
            "morph.dataset.conditional.pocket.datamodule.PocketLightningDataModule"
        )
        for key in (
            "scaffold_groups_dir",
            "scaffold_kind",
            "enumerate_slicings",
            "eval_panel",
        ):
            dataset_config.pop(key, None)
    else:
        dataset_config.eval_panel = None
        dataset_config.enumerate_slicings = True
    datamodule = hydra.utils.instantiate(
        dataset_config,
        _recursive_=False,
        vocab=data.vocab,
        distributions=data.prior,
        noise_distributions=data.empirical,
    )
    datamodule.setup("test" if args.split == "test" else "validate")
    datamodule.configure_model(cfg)
    datasets = (
        datamodule.test_datasets if args.split == "test" else datamodule.val_datasets
    )
    if len(datasets) != 1:
        raise ValueError("Evaluation requires exactly one task dataset for this split")
    module = build_model_module(cfg, data)
    load_checkpoint(module, args.checkpoint)
    module.requires_grad_(False).eval().to(args.device)
    return module, datasets[0], cfg


def _format_prediction(graph, vocab):
    """Keep the largest component, as in DiffDec's reformat.py, without filtering.

    Morph supplies predicted bonds directly; DiffDec reconstructs them from XYZ
    with Open Babel. Both feed SMILES to the same metrics. Return an empty SMILES
    for conversion failures so every attempt still contributes to validity.
    """
    try:
        mol = graph.to_rdkit_mol(
            vocab.atom_tokens, vocab.edge_tokens, vocab.charge_tokens, sanitize=False
        )
        if mol is None:
            return None, "", "empty_graph"
        mol = max(
            Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False),
            key=lambda fragment: fragment.GetNumAtoms(),
            default=mol,
        )
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
        except RuntimeError:
            smiles = Chem.MolToSmiles(mol, canonical=False)
        return mol, smiles, None
    except (ValueError, RuntimeError, IndexError) as exc:
        return None, "", f"{type(exc).__name__}: {exc}"


def sample_records(args):
    """Generate the 100 attempts per condition used in DiffDec, then score them."""
    import torch  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from morph.dataset.conditional.pocket.datamodule import (  # noqa: PLC0415
        pocket_eval_collate_fn,
    )
    from morph.dataset.conditional.pocket.evaluation import (  # noqa: PLC0415
        DiffDecPocketEvaluationDataset,
        cached_cut_conditions,
        published_cut_conditions,
    )
    from morph.dataset.core.atomic_data import iter_geometry  # noqa: PLC0415

    module, wrapped, cfg = _build_task(args)
    base = wrapped.base_dataset
    try:
        if args.diffdec_data:
            entries = torch.load(
                args.diffdec_data, map_location="cpu", weights_only=False
            )
            table = None
            if args.diffdec_table:
                with args.diffdec_table.open() as handle:
                    table = list(csv.DictReader(handle))
            conditions = published_cut_conditions(base, entries, table)
            condition_source = "published_diffdec"
        else:
            conditions = cached_cut_conditions(wrapped)
            condition_source = "morph_cached_cuts"
        total_conditions = len(conditions)
        if args.limit_conditions is not None:
            conditions = conditions[: args.limit_conditions]
        dataset = DiffDecPocketEvaluationDataset(base, conditions)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=pocket_eval_collate_fn,
        )
        _write_json(args.output_dir / "conditions.json", conditions)
        OmegaConf.save(cfg, args.output_dir / "config.yaml", resolve=True)
        provenance = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _file_sha256(args.checkpoint),
            "config": str(args.config.resolve()),
            "condition_source": condition_source,
            "condition_count": len(conditions),
            "available_condition_count": total_conditions,
            "split": args.split,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "samples_per_condition": DIFFDEC_SAMPLES_PER_CONDITION,
            "weights": "ema" if module.use_ema_for_eval else "current",
            "vocab_filter_report": base.vocab_filter_report,
            "bond_source": "morph_predicted_topology",
            "reference_source": "diffdec_table"
            if args.diffdec_table
            else "mapped_crossdocked",
            "sdf_coordinate_frame": "original_receptor_angstrom",
        }
        if args.diffdec_data:
            provenance["diffdec_data_sha256"] = _file_sha256(args.diffdec_data)
            if args.diffdec_table:
                provenance["diffdec_table_sha256"] = _file_sha256(args.diffdec_table)
        records = []
        with (
            torch.inference_mode(),
            Chem.SDWriter(str(args.output_dir / "generated.sdf")) as sdf,
            (args.output_dir / "attempts.jsonl").open("w") as attempts,
        ):
            sdf.SetKekulize(False)
            for batch_idx, batch in enumerate(loader):
                # The sampler recenters a movable scaffold and pocket together.
                # Retain the input pocket to restore the original receptor frame.
                input_pockets = batch[0].contexts["pocket"].to_data_list()
                batch = tuple(graph.to(args.device) for graph in batch)
                final = module.sample(batch, batch_idx, return_traj=False).cpu()
                frames = final.to_data_list()
                for graph, before in zip(frames, input_pockets, strict=True):
                    after = graph.contexts["pocket"]
                    sample_id = len(records)
                    condition = conditions[sample_id // DIFFDEC_SAMPLES_PER_CONDITION]
                    center = base.get_metadata(condition["base_idx"])["lig_center"]
                    shift = before.x * module.distributions.coordinate_std - after.x
                    translation = shift.mean(0) + center
                    for geometry in iter_geometry(graph):
                        geometry.x += translation
                    mol, smiles, error = _format_prediction(graph, module.vocab)
                    record = {
                        key: condition[key]
                        for key in ("scaffold", "true_molecule", "protein_filename")
                    }
                    record.update(pred_molecule=smiles, pred_rgroup="")
                    sdf_error = None
                    if mol is not None:
                        mol.SetProp("_Name", str(sample_id))
                        mol.SetProp("condition_uuid", condition["uuid"])
                        mol.SetProp("protein_filename", condition["protein_filename"])
                        try:
                            sdf.write(mol)
                        except (ValueError, RuntimeError) as exc:
                            sdf_error = str(exc)
                    records.append(record)
                    attempts.write(
                        json.dumps(
                            {
                                **record,
                                "sample_id": sample_id,
                                "condition_uuid": condition["uuid"],
                                "conversion_error": error,
                                "sdf_error": sdf_error,
                            }
                        )
                        + "\n"
                    )
                attempts.flush()
                print(f"Sampled {len(records)}/{len(dataset)} attempts", flush=True)
        if len(records) != len(dataset):
            raise RuntimeError("Sampler did not return every requested attempt")
        return records, provenance
    finally:
        base.close()


def main(argv=None):
    """Run the standalone non-docking evaluation used in DiffDec."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--metrics-file", type=Path, help="DiffDec metric .smi to score"
    )
    source.add_argument("--checkpoint", type=Path, help="Morph fine-tuned checkpoint")
    parser.add_argument("--config", type=Path, help="Saved, composed Hydra run config")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--assets-root", type=Path, help="Checkpoint preprocessing assets"
    )
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "current"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-conditions", type=int, help="Explicit subset for smoke tests"
    )
    parser.add_argument(
        "--diffdec-data", type=Path, help="Published test tensor .pt file"
    )
    parser.add_argument(
        "--diffdec-table", type=Path, help="Optional matching test CSV, if available"
    )
    parser.add_argument(
        "--cached-cuts",
        action="store_true",
        help="Explicitly use Morph's cuts instead of the published benchmark panel",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a saved config value (repeatable)",
    )
    args = parser.parse_args(argv)
    if args.checkpoint and not args.config:
        parser.error("--checkpoint requires --config")
    if args.diffdec_table and not args.diffdec_data:
        parser.error("--diffdec-table requires --diffdec-data")
    if args.metrics_file and (args.diffdec_data or args.config or args.cached_cuts):
        parser.error("--config/--diffdec-data/--cached-cuts apply only to sampling")
    if args.checkpoint and bool(args.diffdec_data) == args.cached_cuts:
        parser.error("Choose published --diffdec-data or --cached-cuts")
    if args.diffdec_data and args.split != "test":
        parser.error("Published DiffDec test conditions require --split test")
    if (
        args.batch_size < 1
        or args.seed < 0
        or any(n is not None and n < 1 for n in (args.steps, args.limit_conditions))
    ):
        parser.error("Counts must be positive and seed must be nonnegative")
    # Avoid mixing new attempts with old artifacts after an interrupted run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.metrics_file:
        records = read_diffdec_records(args.metrics_file)
        provenance = {
            "metrics_file": str(args.metrics_file.resolve()),
            "metrics_file_sha256": _file_sha256(args.metrics_file),
        }
    else:
        records, provenance = sample_records(args)
    metric_path = args.output_dir / "crossdock_test_metric.smi"
    write_diffdec_records(records, metric_path)
    report = score_records(records, provenance)
    _write_json(args.output_dir / "metrics.json", report)
    print(json.dumps(report["metrics"], indent=2, allow_nan=False))
    for note in report["notes"]:
        print(note)
    print(f"Results: {args.output_dir / 'metrics.json'}; docking: not run")


if __name__ == "__main__":
    main()
