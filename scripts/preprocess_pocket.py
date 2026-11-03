#!/usr/bin/env python3
"""Build CrossDocked caches and validate them for a pretrained ligand model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from morph.dataset.conditional.pocket import CrossDockedPocketDataset
from morph.dataset.conditional.pocket.build import build_lmdbs
from morph.dataset.conditional.scaffold.cutters import (
    SCAFFOLD_KINDS,
    libinvent_enumerator,
)
from morph.dataset.conditional.scaffold.lmdb import load_scaffold_cache
from morph.dataset.core.lmdb import SPLITS
from morph.dataset.core.preprocessing import Preprocessing
from morph.dataset.core.vocab import Vocab

ARTIFACTS = (
    "atom_tokens.txt",
    "edge_tokens.txt",
    "charge_tokens.txt",
    "distributions.pt",
)
SUMMARY = "finetune_preprocessing_summary.json"


def _write_summary(path, summary):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(summary, f, indent=2)
            f.write("\n")
            f.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def prepare(args):
    """Reuse checkpoint tokens/scale; leave raw records independent of the model."""
    if args.scaffold_kind:
        from morph.dataset.conditional.pocket.scaffold_pocket import (  # noqa: PLC0415
            scaffold_cache_directory,
        )

    assets = args.pretrained_assets.resolve()
    for name in ARTIFACTS:
        if not (assets / name).is_file():
            raise FileNotFoundError(f"Missing pretrained artifact: {assets / name}")
    pre = Preprocessing(root=str(assets))
    if args.scaffold_kind == "libinvent":
        libinvent_enumerator()  # Check the optional dependency before building data.
    if ("H" in pre.vocab.atom_tokens) != (args.hydrogen_mode == "explicit"):
        raise ValueError("hydrogen-mode must match the pretrained atom vocabulary")
    coordinate_std = float(pre.distributions.coordinate_std)
    if not math.isfinite(coordinate_std) or coordinate_std <= 0:
        raise ValueError("Pretrained coordinate_std must be finite and positive")
    fingerprints = {
        name: hashlib.sha256((assets / name).read_bytes()).hexdigest()
        for name in ARTIFACTS
    }
    options = dict(
        source_dir=args.source_dir,
        hydrogen_mode=args.hydrogen_mode,
        max_atoms=args.max_atoms,
        val_fraction=args.val_frac,
        seed=args.seed,
        preprocess_workers=args.workers,
        preprocess_chunk_size=args.chunk_size,
    )
    root = args.root.resolve()
    print(f"Preparing {root} with pretrained assets from {assets}", flush=True)
    manifest = build_lmdbs(root, **options)
    # Match the training datamodule: protein tokens come from TRAIN pockets,
    # independently of the fixed ligand output vocabulary.
    pocket_vocab = Vocab(
        sorted(manifest["splits"]["train"]["pocket_elements"]),
        ["<NO_BOND>"],
        ["0"],
    )
    summary = {
        "status": "validating",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "root": str(root),
        "source_dir": str(args.source_dir.resolve()) if args.source_dir else None,
        "processing_options": manifest["options"],
        "source_split_sha256": manifest["split_sha256"],
        "upstream_revision": manifest["upstream_revision"],
        "pretrained_assets": str(assets),
        "pretrained_artifact_sha256": fingerprints,
        "ligand_vocab": asdict(pre.vocab),
        "coordinate_std": coordinate_std,
        "pocket_vocab": asdict(pocket_vocab),
        "max_pocket_atoms": args.max_pocket_atoms,
        "scaffold_kind": args.scaffold_kind,
        "splits": {},
    }
    report_path = (
        root
        / "processed"
        / (
            f"scaffold_{args.scaffold_kind}_{SUMMARY}"
            if args.scaffold_kind
            else SUMMARY
        )
    )
    _write_summary(report_path, summary)
    try:
        for split in SPLITS:
            dataset = CrossDockedPocketDataset(
                root,
                split=split,
                vocab=pre.vocab,
                distributions=pre.distributions,
                pocket_vocab=pocket_vocab,
                max_pocket_atoms=args.max_pocket_atoms,
                **options,
            )
            try:
                report = dict(dataset.vocab_filter_report)
                report["raw_source_count"] = manifest["splits"][split]["source_count"]
                report["processing_status"] = manifest["splits"][split]["status"]
                report["validated"] = 0
                summary["splits"][split] = report
                print(f"[{split}] vocabulary coverage: {report}", flush=True)
                if split == "train" and not len(dataset):
                    raise ValueError(
                        "No training ligands remain after vocabulary filtering"
                    )
                for i in tqdm(
                    range(len(dataset)), desc=f"Validate {split}", mininterval=30
                ):
                    try:
                        dataset[i]
                        dataset.get_pocket(i)
                    except (ValueError, RuntimeError) as error:
                        ligand_fn = dataset.get_metadata(i)["ligand_fn"]
                        raise ValueError(f"{split} {ligand_fn}: {error}") from error
                    report["validated"] += 1
                if args.scaffold_kind:
                    directory = scaffold_cache_directory(
                        dataset, args.scaffold_groups_dir or root / "scaffold_groups"
                    )
                    print(
                        f"[{split}] preparing {args.scaffold_kind} scaffold cache",
                        flush=True,
                    )
                    cache = load_scaffold_cache(
                        dataset,
                        directory,
                        split,
                        scaffold_kind=args.scaffold_kind,
                    )
                    indices = cache.filtered_indices()
                    report["scaffolds"] = {
                        "directory": directory,
                        "molecules": len(indices),
                        "slicings": sum(
                            len(cache.atom_indices[int(i)]) for i in indices
                        ),
                        "dropped_without_scaffold": len(dataset) - len(indices),
                    }
                    if split == "train" and not len(indices):
                        raise ValueError("No training ligands have a usable scaffold")
            finally:
                dataset.close()
            _write_summary(report_path, summary)
    except Exception as error:
        summary.update(status="failed", error=str(error))
        _write_summary(report_path, summary)
        raise
    summary["status"] = "ready"
    _write_summary(report_path, summary)
    print(f"CrossDocked fine-tuning data ready. Report: {report_path}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/crossdocked"))
    parser.add_argument(
        "--source-dir", type=Path, default=os.environ.get("MORPH_CROSSDOCKED_RAW")
    )
    parser.add_argument(
        "--pretrained-assets",
        type=Path,
        default=os.environ.get("MORPH_PRETRAINED_ASSETS"),
        help="Directory containing the checkpoint's token files and distributions.pt",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-atoms", type=int)
    parser.add_argument("--max-pocket-atoms", type=int, default=512)
    parser.add_argument(
        "--scaffold-kind",
        choices=sorted(SCAFFOLD_KINDS),
        help="Also prepare scaffold-decoration caches (default: pocket-only)",
    )
    parser.add_argument(
        "--scaffold-groups-dir",
        type=Path,
        help="Scaffold cache parent (default: <root>/scaffold_groups)",
    )
    parser.add_argument(
        "--hydrogen-mode", choices=("implicit", "explicit"), default="implicit"
    )
    args = parser.parse_args(argv)
    if args.pretrained_assets is None:
        parser.error("provide --pretrained-assets or set MORPH_PRETRAINED_ASSETS")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("run preprocessing as one process, before torchrun")
    if args.workers < 0 or args.chunk_size < 1 or args.max_pocket_atoms < 1:
        parser.error(
            "workers must be non-negative; chunk and pocket sizes must be positive"
        )
    if not 0 <= args.val_frac < 1 or (
        args.max_atoms is not None and args.max_atoms < 1
    ):
        parser.error("val-frac must be in [0, 1); max-atoms must be positive")
    torch.set_num_threads(1)
    return prepare(args)


if __name__ == "__main__":
    main()
