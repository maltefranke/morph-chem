"""Build CrossDocked train/val/test LMDBs, optionally discovering ligand tokens.

Run ``python -m morph.dataset.conditional.pocket.build --help`` for the CLI.
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import os
import pickle
import random
import tempfile
from contextlib import nullcontext
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import torch
from rdkit import Chem

from morph.dataset.conditional.pocket.processing import (
    ensure_crossdocked_raw,
    parse_pocket,
    process_ligand_diffdec,
)
from morph.dataset.core.lmdb import RECORD_FIELDS, SPLITS, write_bytes_to_lmdb

UPSTREAM_REVISION = "f81996ace95ee45c2e1d23332252d38137b3a918"
MANIFEST = "crossdocked_summary.json"


def processing_options(
    hydrogen_mode, max_atoms, allowed_ligand_atom_symbols, val_fraction, seed
):
    """Validate and normalize all settings that affect cache contents."""
    if hydrogen_mode not in ("implicit", "explicit"):
        raise ValueError("hydrogen_mode must be implicit or explicit")
    if max_atoms is not None and max_atoms < 1:
        raise ValueError("max_atoms must be positive or None")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    return {
        "schema_version": 2,
        "processing": "diffdec",
        "hydrogen_mode": hydrogen_mode,
        "max_atoms": max_atoms,
        "allowed_ligand_atom_symbols": (
            sorted(set(allowed_ligand_atom_symbols))
            if allowed_ligand_atom_symbols is not None
            else None
        ),
        "val_fraction": val_fraction,
        "seed": seed,
    }


def read_manifest(root: str | Path, options: dict) -> dict | None:
    """Reuse only complete caches built with the requested processing settings."""
    processed = Path(root) / "processed"
    path = processed / MANIFEST
    if not path.is_file():
        if processed.is_dir() and any(processed.iterdir()):
            raise ValueError(
                f"Incomplete or incompatible CrossDocked cache: {processed}"
            )
        return None
    manifest = json.loads(path.read_text())
    if manifest["options"] != options:
        raise ValueError(
            f"Cache uses different CrossDocked preprocessing options: {root}; "
            "choose a separate root for another hydrogen mode, filter, or split"
        )
    if not all(
        (processed / f"{split}_data.lmdb" / "data.mdb").is_file() for split in SPLITS
    ):
        raise ValueError(f"Incomplete CrossDocked cache: {processed}")
    return manifest


def carve_val(train_entries, val_fraction: float, seed: int = 42):
    """Hold out whole pocket directories, matching morph's grouped validation split."""
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    by_directory = collections.defaultdict(list)
    for entry in train_entries:
        by_directory[entry[0].split("/")[0]].append(entry)
    directories = sorted(by_directory)
    random.Random(seed).shuffle(directories)
    target, count, val_directories = int(len(train_entries) * val_fraction), 0, set()
    for directory in directories:
        if count >= target:
            break
        val_directories.add(directory)
        count += len(by_directory[directory])
    train = [
        e
        for d in sorted(by_directory)
        if d not in val_directories
        for e in by_directory[d]
    ]
    val = [e for d in sorted(val_directories) for e in by_directory[d]]
    return train, val, val_directories


def _load_splits(path: Path, val_fraction: float, seed: int):
    # The official release contains Python tuples, not just tensors.
    split = torch.load(path, map_location="cpu", weights_only=False)
    for name in ("train", "test"):
        for entry in split[name]:
            if len(entry) != 2 or any(
                not isinstance(p, str) or Path(p).is_absolute() or ".." in Path(p).parts
                for p in entry
            ):
                raise ValueError(f"Invalid CrossDocked pair in {name}: {entry!r}")
    train_dirs = {e[0].split("/")[0] for e in split["train"]}
    test_dirs = {e[0].split("/")[0] for e in split["test"]}
    if train_dirs & test_dirs:
        raise ValueError("CrossDocked train/test pocket directories overlap")
    train, val, val_dirs = carve_val(split["train"], val_fraction, seed)
    if not train:
        raise ValueError("The grouped validation split leaves no training complexes")
    return {"train": train, "val": val, "test": split["test"]}, sorted(val_dirs)


def _process_one(indexed_entry, *, tree: Path, options: dict):
    source_index, (pocket_fn, ligand_fn) = indexed_entry
    data = process_ligand_diffdec(tree / ligand_fn, options["hydrogen_mode"])
    if data is None:
        return None, "lig_unreadable"
    if options["max_atoms"] is not None and data.num_nodes > options["max_atoms"]:
        return None, "lig_too_large"
    allowed = options["allowed_ligand_atom_symbols"]
    if allowed is not None:
        table = Chem.GetPeriodicTable()
        if any(table.GetElementSymbol(z) not in allowed for z in data.z.tolist()):
            return None, "lig_offvocab"
    try:
        pocket = parse_pocket(tree / pocket_fn)
    except (OSError, ValueError, IndexError):
        return None, "pocket_unreadable"
    if pocket is None:
        return None, "pocket_empty"
    # One translation for both sides. Store it before normalization so future
    # sampling/evaluation can recover the original, full-receptor frame.
    center = data.pos.mean(dim=0)
    record = {key: getattr(data, key) for key in RECORD_FIELDS}
    record.update(
        pos=data.pos - center,
        poc_pos=pocket["pos"] - center.numpy(),
        poc_elem=pocket["elem"],
        poc_resname=pocket["resname"],
        poc_backbone=pocket["backbone"],
        poc_resid=pocket["resid"],
        lig_center=center,
        pocket_fn=pocket_fn,
        ligand_fn=ligand_fn,
        source_index=source_index,
    )
    # Serialize in the worker, avoiding one shared-memory allocation/descriptor
    # per tensor when a full release is streamed through a multiprocessing pool.
    return pickle.dumps(record, protocol=4), "ok"


def _worker_init():
    torch.set_num_threads(1)


def _build_split(name, entries, directory, tree, options, workers, chunk_size):
    status = collections.Counter()
    rejected = []
    smiles, ligand_fns = [], []
    residues, ligand_elements, pocket_elements = (
        set(),
        collections.Counter(),
        collections.Counter(),
    )
    convert = partial(_process_one, tree=tree, options=options)
    pool_context = (
        get_context("spawn").Pool(workers, initializer=_worker_init)
        if workers > 1
        else nullcontext()
    )
    with pool_context as pool:
        results = (
            pool.imap(convert, enumerate(entries), chunksize=chunk_size)
            if pool is not None
            else map(convert, enumerate(entries))
        )

        def stream():
            for index, (blob, reason) in enumerate(results):
                status[reason] += 1
                if blob is None:
                    rejected.append(
                        {
                            "source_index": index,
                            "pair": entries[index],
                            "reason": reason,
                        }
                    )
                    yield None
                    continue
                record = pickle.loads(blob)
                smiles.append(record["smiles"])
                ligand_fns.append(record["ligand_fn"])
                residues.update(record["poc_resname"])
                ligand_elements.update(str(z) for z in record["z"].tolist())
                pocket_elements.update(record["poc_elem"])
                yield blob

        written, _ = write_bytes_to_lmdb(
            stream(), str(directory / f"{name}_data.lmdb"), total=len(entries)
        )
    if entries and not written:
        raise ValueError(f"No valid complexes in {name}: {dict(status)}")
    for suffix, values in (("smiles", smiles), ("ligand_fns", ligand_fns)):
        (directory / f"{name}_{suffix}.txt").write_text(
            "".join(f"{value}\n" for value in values)
        )
    return {
        "count": written,
        "source_count": len(entries),
        "status": dict(status),
        "rejected": rejected,
        "residues": sorted(residues),
        "ligand_atomic_numbers": dict(ligand_elements),
        "pocket_elements": dict(pocket_elements),
    }


def build_lmdbs(
    root: str | Path,
    source_dir: str | Path | None = None,
    *,
    hydrogen_mode: str = "implicit",
    max_atoms: int | None = None,
    allowed_ligand_atom_symbols=None,
    val_fraction: float = 0.02,
    seed: int = 42,
    preprocess_workers: int = 0,
    preprocess_chunk_size: int = 32,
) -> dict:
    """Build missing data once, publishing all splits together after success.

    The OS releases the build lock even if a process dies. Readers never see
    partially written LMDBs. Completed caches work without the raw sources.
    Raw chemistry is independent of model vocabularies. An explicit ligand
    element filter may be supplied; it never filters protein-pocket elements.
    """
    options = processing_options(
        hydrogen_mode, max_atoms, allowed_ligand_atom_symbols, val_fraction, seed
    )
    if preprocess_workers < 0 or preprocess_chunk_size < 1:
        raise ValueError("Workers must be non-negative and chunk size positive")
    root = Path(root)
    cached = read_manifest(root, options)
    if cached is not None:
        return cached
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".crossdocked.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = read_manifest(root, options)
        if cached is not None:
            return cached
        tree, split_path = ensure_crossdocked_raw(root / "raw", source_dir)
        splits, val_dirs = _load_splits(split_path, val_fraction, seed)
        with tempfile.TemporaryDirectory(dir=root) as staging:
            output = Path(staging) / "processed"
            output.mkdir()
            split_results = {
                name: _build_split(
                    name,
                    entries,
                    output,
                    tree,
                    options,
                    preprocess_workers,
                    preprocess_chunk_size,
                )
                for name, entries in splits.items()
            }
            manifest = {
                "options": options,
                "upstream_revision": UPSTREAM_REVISION,
                "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
                "val_directories": val_dirs,
                "splits": split_results,
            }
            (output / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
            os.replace(output, root / "processed")
        return manifest


def compute_vocab_distributions(root: str | Path, **dataset_options):
    """Discover tokens/statistics for training from scratch in processed/.

    Fine-tuning must instead reuse the pretrained model's vocabulary and
    coordinate normalization; discovering new tokens can change their indices.
    """
    from morph.dataset.conditional.pocket.crossdocked import (  # noqa: PLC0415
        CrossDockedPocketDataset,
    )
    from morph.dataset.core.preprocessing import Preprocessing  # noqa: PLC0415

    root = Path(root)
    dataset = CrossDockedPocketDataset(root, split="train", **dataset_options)
    try:
        return Preprocessing(
            root=str(root / "processed"),
            train_dataset=dataset,
            num_workers=dataset_options.get("preprocess_workers", 0),
        )
    finally:
        dataset.close()


def main(argv: list[str] | None = None) -> None:
    """Prepare pocket data without constructing a model or training datamodule."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/crossdocked"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hydrogen-mode", choices=("implicit", "explicit"), default="implicit"
    )
    parser.add_argument("--max-atoms", type=int)
    parser.add_argument(
        "--compute-statistics",
        action="store_true",
        help="discover ligand vocabulary/statistics for training from scratch; "
        "for fine-tuning, reuse the checkpoint's preprocessing artifacts",
    )
    args = parser.parse_args(argv)
    if (
        args.workers < 0
        or not 0 <= args.val_frac < 1
        or (args.max_atoms is not None and args.max_atoms < 1)
    ):
        parser.error(
            "workers must be non-negative, val-frac in [0, 1), and max-atoms positive"
        )
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("run preprocessing as one process, before torchrun")
    options = dict(
        source_dir=args.source_dir,
        preprocess_workers=args.workers,
        val_fraction=args.val_frac,
        seed=args.seed,
        hydrogen_mode=args.hydrogen_mode,
        max_atoms=args.max_atoms,
    )
    manifest = build_lmdbs(args.root, **options)
    if args.compute_statistics:
        compute_vocab_distributions(args.root, **options)
    counts = {name: value["count"] for name, value in manifest["splits"].items()}
    print(f"CrossDocked ready at {args.root / 'processed'}: {counts}")


if __name__ == "__main__":
    main()
