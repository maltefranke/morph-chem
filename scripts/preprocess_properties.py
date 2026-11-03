#!/usr/bin/env python3
"""Prepare cached RDKit properties for a configured molecular dataset."""

import argparse
import os
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir

import morph.config  # noqa: F401
from morph.dataset.conditional.logp.preprocessing import build_property_array
from morph.dataset.core.properties import PROPERTY_FNS


def _prepare_split(cfg, split, names, force):
    dataset_cfg = cfg.data.datamodule.datasets[split]
    if split != "train":
        if len(dataset_cfg) != 1:
            raise ValueError("Select one molecular dataset per evaluation split")
        dataset_cfg = dataset_cfg[0]
    dataset = hydra.utils.instantiate(
        dataset_cfg,
        vocab=None,
        distributions=None,
        compute_properties=False,
        split=split,
    )
    try:
        for name in names:
            path = build_property_array(dataset, name, force=force)
            print(f"[{split}] ready: {path}", flush=True)
    finally:
        if hasattr(dataset, "close"):
            dataset.close()


def main(argv=None):
    """Compose ordinary dataset settings and prepare selected properties/splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra dataset overrides")
    parser.add_argument("--split", choices=("train", "val", "test"), action="append")
    parser.add_argument("--property", choices=tuple(PROPERTY_FNS), action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("run preprocessing as one process, before torchrun")
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PROJECT_ROOT", str(root))
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.1"):
        cfg = compose(config_name="default", overrides=args.overrides)
    names = list(dict.fromkeys(args.property or ["logp"]))
    for split in dict.fromkeys(args.split or ["train", "val", "test"]):
        _prepare_split(cfg, split, names, args.force)


if __name__ == "__main__":
    main()
