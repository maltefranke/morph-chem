"""Scaffold-conditioned scores, with one row per attempted generation."""

from collections import defaultdict
from statistics import mean

import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from morph.utils.rdkit_utils import mol_is_valid, smiles_from_mol


def _bonds(graph, no_bond):
    matrix = torch.full((graph.num_nodes, graph.num_nodes), no_bond)
    matrix[graph.edge_index[0], graph.edge_index[1]] = graph.e.long()
    return matrix


def _aligned_rmsd(generated, source):
    """Measure deformation with fixed atom correspondence and a proper rotation."""
    predicted = generated.double() - generated.double().mean(dim=0)
    reference = source.double() - source.double().mean(dim=0)
    if not torch.isfinite(predicted).all() or not torch.isfinite(reference).all():
        return None
    u, _, vh = torch.linalg.svd(predicted.T @ reference)
    correction = torch.eye(3, dtype=u.dtype, device=u.device)
    correction[-1, -1] = torch.linalg.det(u @ vh)
    residual = predicted @ (u @ correction @ vh) - reference
    return float(residual.square().sum(-1).mean().sqrt())


def scaffold_row(source, target, generated, mol, vocab, *, allow_charged=True):
    """Score one attempt; graph coordinates must already be in physical units.

    Retained atoms keep their relative order through integration. Compare their
    types, charges, bonds and *unaligned* coordinates to the exact input graph.
    Invalid RDKit conversions still contribute to all-attempt denominators.
    """
    keep_mask = getattr(generated, "keep_mask", None)
    kept = generated.filter_nodes(keep_mask.bool()) if keep_mask is not None else None
    same_size = kept is not None and kept.num_nodes == source.num_nodes
    no_bond = list(vocab.edge_tokens).index("<NO_BOND>")
    preserved = bool(
        same_size
        and torch.equal(source.a, kept.a)
        and torch.equal(source.c, kept.c)
        and torch.equal(_bonds(source, no_bond), _bonds(kept, no_bond))
    )
    rmsd = (
        float((kept.x - source.x).square().sum(-1).mean().sqrt()) if same_size else None
    )
    valid = mol_is_valid(mol, allow_charged=allow_charged)
    smiles = smiles_from_mol(mol) if valid else None
    target_mol = target.to_rdkit_mol(
        vocab.atom_tokens, vocab.edge_tokens, vocab.charge_tokens
    )
    target_smiles = smiles_from_mol(target_mol)
    heavy = torch.tensor([symbol != "H" for symbol in vocab.atom_tokens])
    return {
        "sample_id": int(source.sample_id.item()),
        "scaffold_id": int(source.scaffold_id.item()),
        "validity": valid,
        "retention": preserved,
        "valid_retention": valid and preserved,
        "rmsd": rmsd,
        "rmsd_aligned": _aligned_rmsd(kept.x, source.x) if same_size else None,
        "added_atoms": generated.num_nodes - source.num_nodes,
        "added_heavy_atoms": int(heavy[generated.a].sum() - heavy[source.a].sum()),
        "smiles": smiles if preserved else None,
        "recovered": bool(valid and preserved and smiles and smiles == target_smiles),
        # Same atom order as the decoded molecule; no hidden target geometry is
        # used to excuse an inherited clash. Keep plain lists for DDP/JSON.
        "posebusters_reference": {
            "kept_indices": keep_mask.bool().nonzero(as_tuple=True)[0].tolist(),
            "positions": source.x.tolist(),
        }
        if preserved and mol is not None and mol.GetNumAtoms() == generated.num_nodes
        else None,
    }


def summarize_scaffold_rows(rows):
    """Macro-average over scaffolds after gathering all ranks' attempt rows.

    Uniqueness/diversity use chemically valid, scaffold-preserving outputs.
    ``unique_yield`` divides by *all* attempts; ``uniqueness`` by eligible outputs.
    Empty groups score zero. Diversity is zero with fewer than two eligible
    outputs. Recovery is an any-hit rate per scaffold, at the evaluated budget.
    Repeated sample IDs from distributed-sampler padding are counted only once.
    """
    rows = list({row["sample_id"]: row for row in rows}.values())
    if not rows:
        return {}
    groups = defaultdict(list)
    for row in rows:
        groups[row["scaffold_id"]].append(row)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fingerprints = {}
    summaries = []
    for group in groups.values():
        smis = [row["smiles"] for row in group if row["smiles"] is not None]
        for smi in set(smis) - fingerprints.keys():
            fingerprints[smi] = generator.GetFingerprint(Chem.MolFromSmiles(smi))
        fps = [fingerprints[smi] for smi in smis]
        distance = sum(
            sum(
                1 - value
                for value in DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :])
            )
            for i, fp in enumerate(fps[:-1])
        )
        rmsds = [row["rmsd"] for row in group if row["rmsd"] is not None]
        aligned_rmsds = [
            row["rmsd_aligned"] for row in group if row.get("rmsd_aligned") is not None
        ]
        summary = {
            key: mean(row[key] for row in group)
            for key in (
                "validity",
                "retention",
                "valid_retention",
                "added_atoms",
                "added_heavy_atoms",
            )
        }
        summary.update(
            no_growth=mean(row["added_atoms"] <= 0 for row in group),
            no_heavy_atom_growth=mean(row["added_heavy_atoms"] <= 0 for row in group),
            rmsd=mean(rmsds) if rmsds else float("nan"),
            rmsd_aligned=mean(aligned_rmsds) if aligned_rmsds else float("nan"),
            uniqueness=len(set(smis)) / len(smis) if smis else 0.0,
            unique_yield=len(set(smis)) / len(group),
            diversity=2 * distance / (len(fps) * (len(fps) - 1))
            if len(fps) > 1
            else 0.0,
            recovery=float(any(row["recovered"] for row in group)),
        )
        summaries.append(summary)
    return {
        **{key: mean(row[key] for row in summaries) for key in summaries[0]},
        "n_scaffolds": len(groups),
        "n_samples": len(rows),
        "min_samples_per_scaffold": min(map(len, groups.values())),
        "max_samples_per_scaffold": max(map(len, groups.values())),
    }
