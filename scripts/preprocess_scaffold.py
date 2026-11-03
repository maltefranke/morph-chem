#!/usr/bin/env python3
"""Prepare scaffold caches and diverse evaluation panels ahead of first use."""

import argparse
import os
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir

import morph.config  # noqa: F401
from morph.dataset.conditional.scaffold.lmdb import load_scaffold_cache
from morph.dataset.conditional.scaffold.panel import ScaffoldEvaluationDataset


def _prepare_split(cfg, split, force=False):
    dataset_cfg = cfg.data.datamodule.datasets[split]
    if split != "train":
        dataset_cfg = dataset_cfg[0]
    dataset = hydra.utils.instantiate(dataset_cfg, vocab=None, split=split)
    try:
        settings = cfg.data.get("eval_panel")
        if split != "train" and settings is not None:
            panel = ScaffoldEvaluationDataset(
                dataset,
                scaffold_groups_dir=cfg.data.scaffold_groups_dir,
                scaffold_kind=cfg.data.scaffold_kind,
                force=force,
                **settings,
            )
            print(f"[{split}] ready: {len(panel.entries)} scaffolds in {panel.path}")
        else:
            cache = load_scaffold_cache(
                dataset,
                cfg.data.scaffold_groups_dir,
                split,
                scaffold_kind=cfg.data.scaffold_kind,
            )
            print(f"[{split}] ready: {len(cache.filtered_indices())} scaffold records")
    finally:
        if hasattr(dataset, "close"):
            dataset.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides for the run")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        action="append",
        help="Prepare only this split; repeat to select several (default: all).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild existing evaluation panels."
    )
    args = parser.parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("run preprocessing as one process, before torchrun")
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PROJECT_ROOT", str(root))
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.1"):
        cfg = compose(config_name="default", overrides=args.overrides)
    if "scaffold_groups_dir" not in cfg.data:
        parser.error("select a scaffold config, e.g. defaults=geom/scaffold_decoration")
    splits = list(dict.fromkeys(args.split or ("train", "val", "test")))
    if any(
        len(cfg.data.datamodule.datasets[split]) != 1
        for split in splits
        if split != "train"
    ):
        parser.error("scaffold preprocessing expects one dataset per split")
    for split in splits:
        _prepare_split(cfg, split, args.force)


if __name__ == "__main__":
    main()
