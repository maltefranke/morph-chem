"""Fixed scaffold/pocket conditions for the evaluation protocol used in DiffDec."""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath

import torch
from rdkit import Chem
from torch.utils.data import Dataset

from morph.dataset.conditional.scaffold.wrappers import evaluation_pair
from morph.utils.metrics.diffdec import DIFFDEC_SAMPLES_PER_CONDITION


def graph_smiles(graph, vocab):
    """Serialize a reference graph; malformed references must stop evaluation."""
    mol = graph.to_rdkit_mol(vocab.atom_tokens, vocab.edge_tokens, vocab.charge_tokens)
    if mol is None:
        raise ValueError("Cannot decode evaluation reference")
    return Chem.MolToSmiles(mol)


def cached_cut_conditions(wrapped):
    """Enumerate Morph's cached cuts, explicitly a different benchmark panel."""
    if wrapped._slicing_offsets is None:
        raise ValueError("Evaluation requires fixed, enumerated scaffold cuts")
    conditions = []
    for base_idx in wrapped._filtered_indices:
        base_idx = int(base_idx)
        target = wrapped.base_dataset[base_idx]
        metadata = wrapped.base_dataset.get_metadata(base_idx)
        for indices in wrapped._scaffold_choices(base_idx):
            mask = torch.zeros(target.num_nodes, dtype=torch.bool)
            mask[list(indices)] = True
            conditions.append(
                {
                    "uuid": str(len(conditions)),
                    "base_idx": base_idx,
                    "atom_indices": list(indices),
                    "scaffold": graph_smiles(target.filter_nodes(mask), wrapped.vocab),
                    "true_molecule": graph_smiles(target, wrapped.vocab),
                    "protein_filename": metadata["pocket_fn"],
                }
            )
    return conditions


def _canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Malformed benchmark reference: {smiles!r}")
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def _protein_key(filename):
    parts = PurePosixPath(filename).parts
    if "crossdocked_pocket10" in parts:
        parts = parts[parts.index("crossdocked_pocket10") + 1 :]
    return "/".join(parts)


def published_cut_conditions(base, entries, table=None, *, tolerance=0.05):
    """Map the published DiffDec cuts to this dataset by element and absolute pose.

    ``entries`` is a released ``crossdock{single,multi}_test_full.pt`` list; optional
    ``table`` contains dictionaries from its matching ``*_test_table.csv``. Every UUID
    must map uniquely, including duplicate cuts/ligands: never deduplicate or
    silently drop benchmark cases. ``charges`` in DiffDec's file are atomic
    numbers, with zero for fake atoms; no model vocabulary is assumed here.

    Coordinates are matched within 0.05 Angstrom by default, far below a bond
    length. Native ligand bond labels and the checkpoint's pocket representation
    are retained. With a table, reference strings/grouping keys are kept verbatim.
    Without one, ligand SMILES come from ``entry['name']`` and scaffold SMILES
    and protein identities from the mapped CrossDocked record. The public Zenodo
    release supplies tensors but no CSV tables.
    This imports the conditions, not DiffDec's XYZ/Open Babel bond perception.
    """
    if tolerance <= 0:
        raise ValueError("Coordinate matching tolerance must be positive")
    by_uuid = {str(row["uuid"]): row for row in table} if table is not None else None
    if by_uuid is not None and len(by_uuid) != len(table):
        raise ValueError("Duplicate UUID in DiffDec reference table")
    uuids = {str(entry["uuid"]) for entry in entries}
    if by_uuid is not None and uuids != set(by_uuid):
        raise ValueError("DiffDec tensor file and reference table have different UUIDs")
    if len(entries) != len(uuids):
        raise ValueError("Duplicate UUID in DiffDec tensor file")
    by_smiles = defaultdict(list)
    for i in range(len(base)):
        metadata = base.get_metadata(i)
        by_smiles[_canonical(metadata["smiles"])].append(i)

    conditions = []
    for entry in entries:
        uuid = str(entry["uuid"])
        row = by_uuid[uuid] if by_uuid is not None else {"molecule": entry["name"]}
        if _canonical(entry["name"]) != _canonical(row["molecule"]):
            raise ValueError(f"DiffDec UUID {uuid}: tensor/table ligand mismatch")
        z = torch.as_tensor(entry["charges"]).flatten().long()
        positions = torch.as_tensor(entry["positions"]).double()
        scaffold = torch.as_tensor(entry["scaffold_only_mask"]).flatten().bool()
        ligand = scaffold | torch.as_tensor(entry["rgroup_mask"]).flatten().bool()
        ligand &= z > 0
        scaffold &= z > 0
        if not scaffold.any() or not (ligand & ~scaffold).any():
            raise ValueError(f"DiffDec UUID {uuid}: empty scaffold or decoration")
        matches = []
        for i in by_smiles[_canonical(row["molecule"])]:
            metadata = base.get_metadata(i)
            # Preserve exact protein identities when multiple poses coincide.
            if "protein_filename" in row and _protein_key(
                metadata["pocket_fn"]
            ) != _protein_key(row["protein_filename"]):
                continue
            raw = base.get(i)
            absolute = raw.pos.double() + metadata["lig_center"].double()
            distances = torch.cdist(positions[ligand], absolute)
            distances[z[ligand, None] != raw.z[None, :]] = float("inf")
            values, indices = distances.min(dim=1)
            if values.max() > tolerance or len(indices.unique()) != len(indices):
                continue
            # DiffDec pockets are subsets of the supplied pocket10 PDB. Validate
            # their actual coordinates too, even when no filename table exists.
            pocket_mask = torch.as_tensor(entry["pocket_mask"]).flatten().bool()
            if not pocket_mask.any():
                raise ValueError(f"DiffDec UUID {uuid}: empty protein pocket")
            pocket = base.get_raw_pocket(i)
            pocket_pos = (
                torch.as_tensor(pocket["pos"]).double()
                + metadata["lig_center"].double()
            )
            pocket_z = torch.tensor(
                [
                    Chem.GetPeriodicTable().GetAtomicNumber(str(element))
                    for element in pocket["elem"]
                ]
            )
            pocket_distances = torch.cdist(positions[pocket_mask], pocket_pos)
            pocket_distances[z[pocket_mask, None] != pocket_z[None, :]] = float("inf")
            if pocket_distances.min(dim=1).values.max() > tolerance:
                continue
            kept = indices[scaffold[ligand]].sort().values.tolist()
            matches.append((i, kept))
        if len(matches) != 1:
            raise ValueError(
                f"DiffDec UUID {uuid}: expected one compatible ligand/pocket pose, "
                f"found {len(matches)}; check split, vocabulary and preprocessing"
            )
        i, indices = matches[0]
        target = base[i]
        mask = torch.zeros(target.num_nodes, dtype=torch.bool)
        mask[indices] = True
        observed = graph_smiles(target.filter_nodes(mask), base.vocab)
        if "scaffold" in row and _canonical(observed) != _canonical(row["scaffold"]):
            raise ValueError(f"DiffDec UUID {uuid}: mapped scaffold topology differs")
        conditions.append(
            {
                "uuid": uuid,
                "base_idx": i,
                "atom_indices": indices,
                "scaffold": row.get("scaffold", observed),
                "true_molecule": row["molecule"],
                "protein_filename": row.get(
                    "protein_filename", base.get_metadata(i)["pocket_fn"]
                ),
            }
        )
    return conditions


class DiffDecPocketEvaluationDataset(Dataset):
    """Make exactly 100 independent attempts per condition, as used in DiffDec."""

    def __init__(self, base_dataset, conditions):
        if not conditions:
            raise ValueError("No scaffold/pocket evaluation conditions")
        self.base_dataset, self.conditions = base_dataset, conditions

    def __len__(self):
        """Count attempts without deduplicating repeated benchmark conditions."""
        return len(self.conditions) * DIFFDEC_SAMPLES_PER_CONDITION

    def __getitem__(self, index):
        """Return a fixed scaffold and its pocket in the original shared frame."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        condition = self.conditions[index // DIFFDEC_SAMPLES_PER_CONDITION]
        base_idx = condition["base_idx"]
        target = self.base_dataset[base_idx]
        mask = torch.zeros(target.num_nodes, dtype=torch.bool)
        mask[condition["atom_indices"]] = True
        source, target = evaluation_pair(target.filter_nodes(mask), target)
        source.contexts["pocket"] = self.base_dataset.get_pocket(base_idx)
        return source, target
