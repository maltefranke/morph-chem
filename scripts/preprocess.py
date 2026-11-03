#!/usr/bin/env python3
"""Download and preprocess QM9, GEOM-Drugs or ChEMBL3D for training."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir

import morph.config  # noqa: F401  (import registers OmegaConf resolvers)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DATASETS = tuple(
    sorted(
        path.parent.name
        for path in (CONFIG_DIR / "defaults").glob("*/*.yaml")
        if path.stem == path.parent.name
    )
)


def _prepare_dataset(
    name: str,
    workers: int | None,
    source_dir: Path | None = None,
    hydrogen_mode: str | None = None,
) -> None:
    overrides = [f"defaults={name}/{name}"]
    if workers is not None:
        overrides.append(f"data.preprocess_workers={workers}")
    cfg = compose(config_name="default", overrides=overrides)
    for key, value in (
        ("source_dir", str(source_dir.resolve()) if source_dir is not None else None),
        ("hydrogen_mode", hydrogen_mode),
    ):
        if value is None:
            continue
        if key not in cfg.data:
            raise ValueError(f"{name} does not support --{key.replace('_', '-')}")
        cfg.data[key] = value

    print(f"[{name}] preparing {cfg.data.preprocessing.root}")
    print(f"[{name}] downloading missing sources and preparing training statistics")
    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)

    # Force construction of every split in this single process. This matters for
    # GEOM, whose train/val/test LMDBs are produced independently; leaving val or
    # test until torchrun would let multiple ranks race to create the same LMDB.
    split_configs = [
        ("train", cfg.data.datamodule.datasets.train),
        *(("val", item) for item in cfg.data.datamodule.datasets.val),
        *(("test", item) for item in cfg.data.datamodule.datasets.test),
    ]
    for split, dataset_cfg in split_configs:
        print(f"[{name}] ensuring {split} split is ready")
        dataset = hydra.utils.instantiate(
            dataset_cfg,
            vocab=preprocessing.vocab,
            distributions=preprocessing.distributions,
            split=split,
        )
        if hasattr(dataset, "close"):
            dataset.close()
        del dataset
        gc.collect()
    print(f"[{name}] ready: {preprocessing.root}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        metavar="DATASET",
        help=f"Datasets to prepare: {', '.join(DATASETS)}. Omit to prepare all.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override the dataset's preprocessing worker count.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Optional local source directory (default: download into raw/).",
    )
    parser.add_argument(
        "--hydrogen-mode",
        choices=("explicit", "implicit"),
        help="Choose explicit H or heavy atoms with implicit H (GEOM/ChEMBL3D).",
    )
    args = parser.parse_args(argv)
    args.datasets = args.datasets or list(DATASETS)
    unknown = [name for name in args.datasets if name not in DATASETS]
    if unknown:
        parser.error(
            f"Unknown datasets: {', '.join(unknown)}; choose {', '.join(DATASETS)}"
        )
    if args.workers is not None and args.workers < 0:
        parser.error("--workers must be non-negative")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("run preprocessing as one process, before torchrun")

    os.environ.setdefault("PROJECT_ROOT", str(REPO_ROOT))
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        for dataset_name in args.datasets:
            _prepare_dataset(
                dataset_name, args.workers, args.source_dir, args.hydrogen_mode
            )


if __name__ == "__main__":
    main()
