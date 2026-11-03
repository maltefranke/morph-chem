"""Precomputed diverse scaffold panels, independent of training-time sampling."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from torch.utils.data import Dataset
from tqdm import tqdm

from morph.dataset.conditional.scaffold.cutters import mol_from_data, scaffold_indices
from morph.dataset.conditional.scaffold.fingerprint import scaffold_cache_metadata
from morph.dataset.conditional.scaffold.lmdb import (
    load_scaffold_cache,
    scaffold_cache_lock,
)
from morph.dataset.conditional.scaffold.wrappers import evaluation_pair


def panel_path(directory, split, kind, n_scaffolds=512, seed=42):
    """Name panels by split, cutter, budget and selection seed."""
    return Path(directory) / f"{split}_{kind}_{n_scaffolds}_seed{seed}.json"


def panel_metadata(dataset, kind, n_scaffolds, seed):
    """Identify the record ordering and selection protocol for cache validation."""
    return {
        "version": 2,
        "split": dataset.split,
        "records": len(dataset),
        "scaffold_cache": scaffold_cache_metadata(dataset, dataset.split, kind),
        "kind": kind,
        "n_scaffolds": n_scaffolds,
        "seed": seed,
    }


def scaffold_smiles(mol, indices):
    """Canonical scaffold topology, ignoring H spelling and stereochemistry."""
    # Cutting a substituted aromatic N can require a new implicit H. Spell the
    # fragment with explicit bond orders so RDKit can infer that valence before
    # perceiving aromaticity again (e.g. N-substituted quinazolinones).
    mol = Chem.Mol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    fragment = Chem.MolFromSmiles(
        Chem.MolFragmentToSmiles(mol, atomsToUse=list(indices), isomericSmiles=False)
    )
    if fragment is None:
        raise ValueError("Scaffold cut does not produce a parseable SMILES")
    return Chem.MolToSmiles(Chem.RemoveHs(fragment), isomericSmiles=False)


def build_panel(dataset, cache, kind, *, n_scaffolds=512, seed=42):
    """Select diverse topologies and one uniform random conformer per molecule.

    Reservoir sampling removes conformer-count bias. Scaffold selection uses
    MaxMin Tanimoto distance on Morgan radius-3/2048-bit fingerprints. Repeated
    conformers never enter the diversity pool. All randomness is local and seeded.
    """
    if n_scaffolds < 1 or seed < 0:
        raise ValueError("n_scaffolds must be positive and seed non-negative")
    rng = random.Random(seed)
    canonical = {}
    representatives = {}
    counts = defaultdict(int)
    read_smiles = getattr(cache, "smiles", lambda _index: None)
    for index in tqdm(cache.filtered_indices(), desc=f"{dataset.split} molecules"):
        index = int(index)
        raw = read_smiles(index) or dataset.get(index).smiles
        if raw not in canonical:
            mol = Chem.MolFromSmiles(raw)
            if mol is None:
                raise ValueError(f"Invalid dataset SMILES at record {index}")
            canonical[raw] = Chem.MolToSmiles(Chem.RemoveHs(mol))
        smi = canonical[raw]
        counts[smi] += 1
        if rng.randrange(counts[smi]) == 0:
            representatives[smi] = index

    candidates = defaultdict(list)
    for smi in tqdm(sorted(representatives), desc="Scaffold topologies"):
        mol = Chem.MolFromSmiles(smi)
        if getattr(dataset, "hydrogen_mode", "explicit") == "explicit":
            mol = Chem.AddHs(mol)
        for cut in scaffold_indices(mol, kind):
            key = scaffold_smiles(mol, cut)
            if smi not in candidates[key]:
                candidates[key].append(smi)
    keys = sorted(candidates)
    if len(keys) < n_scaffolds:
        raise ValueError(
            f"Requested {n_scaffolds} scaffolds but {dataset.split} has {len(keys)}"
        )
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in keys]
    picks = MaxMinPicker().LazyBitVectorPick(
        fingerprints, len(keys), n_scaffolds, seed=seed
    )
    entries = []
    for pick in picks:
        key = keys[pick]
        smi = rng.choice(candidates[key])
        index = representatives[smi]
        # SMILES atom order differs from stored atom order. Resolve the selected
        # topology against the actual conformer, never reuse SMILES atom indices.
        mol = mol_from_data(dataset.get(index))
        cuts = [
            cut
            for cut in scaffold_indices(mol, kind)
            if scaffold_smiles(mol, cut) == key
        ]
        if not cuts:
            raise ValueError(f"Selected scaffold does not match record {index}")
        entries.append(
            {
                "base_idx": index,
                "atom_indices": list(rng.choice(cuts)),
                "molecule_smiles": smi,
                "scaffold_smiles": key,
            }
        )
    return {
        "metadata": panel_metadata(dataset, kind, n_scaffolds, seed),
        "candidate_molecules": len(representatives),
        "candidate_scaffolds": len(keys),
        "entries": entries,
    }


def write_panel(panel, path):
    """Atomically publish the small JSON panel after preprocessing completes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        temporary = out.name
        json.dump(panel, out, indent=2)
        out.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_panel(path, metadata):
    if path.is_file():
        panel = json.loads(path.read_text())
        if panel["metadata"] != metadata:
            raise ValueError(
                f"Stale scaffold evaluation panel: {path}. "
                "Rebuild with scripts/preprocess_scaffold.py --force."
            )
        entries = panel["entries"]
        if len(entries) != metadata["n_scaffolds"] or len(
            {row["scaffold_smiles"] for row in entries}
        ) != len(entries):
            raise ValueError(
                f"Invalid scaffold panel size or duplicate scaffolds: {path}"
            )
        return panel
    return None


def _load_or_build_panel(dataset, path, metadata, scaffold_groups_dir, *, force=False):
    if not force and (panel := _read_panel(path, metadata)) is not None:
        return panel
    with scaffold_cache_lock(path):
        if not force and (panel := _read_panel(path, metadata)) is not None:
            return panel
        print(f"Preparing scaffold evaluation panel: {path}", flush=True)
        cache = load_scaffold_cache(
            dataset,
            scaffold_groups_dir,
            dataset.split,
            scaffold_kind=metadata["kind"],
        )
        panel = build_panel(
            dataset,
            cache,
            metadata["kind"],
            n_scaffolds=metadata["n_scaffolds"],
            seed=metadata["seed"],
        )
        write_panel(panel, path)
        return panel


class ScaffoldEvaluationDataset(Dataset):
    """Prepare a missing panel, then repeat its geometries with stable scoring IDs."""

    def __init__(
        self,
        dataset,
        *,
        scaffold_groups_dir,
        scaffold_kind="rgroup",
        directory=None,
        n_scaffolds=512,
        seed=42,
        samples_per_scaffold=8,
        force=False,
    ):
        if samples_per_scaffold < 1 or n_scaffolds < 1 or seed < 0:
            raise ValueError("Scaffold counts must be positive and seed non-negative")
        if directory is None:
            directory = Path(scaffold_groups_dir) / "eval_panels"
        self.path = panel_path(
            directory, dataset.split, scaffold_kind, n_scaffolds, seed
        )
        metadata = panel_metadata(dataset, scaffold_kind, n_scaffolds, seed)
        panel = _load_or_build_panel(
            dataset, self.path, metadata, scaffold_groups_dir, force=force
        )
        self.entries = panel["entries"]
        self.base_dataset = dataset
        self.samples_per_scaffold = samples_per_scaffold

    def __len__(self):
        """Return the total generation budget, including repeats."""
        return len(self.entries) * self.samples_per_scaffold

    def __getitem__(self, index):
        """Return one fixed scaffold/target pair and its global attempt ID."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        scaffold_id = index // self.samples_per_scaffold
        entry = self.entries[scaffold_id]
        target = self.base_dataset[entry["base_idx"]]
        keep = torch.zeros(target.num_nodes, dtype=torch.bool)
        keep[entry["atom_indices"]] = True
        source, target = evaluation_pair(target.filter_nodes(keep), target)
        source.add_system_property("scaffold_id", torch.tensor([scaffold_id]))
        source.add_system_property("sample_id", torch.tensor([index]))
        return source, target
