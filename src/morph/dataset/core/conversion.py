"""Convert sanitized RDKit conformers to the shared molecular graph fields."""

import torch
from rdkit import Chem
from torch_geometric.data import Data

from morph.utils.rdkit_utils import BOND_IDX_MAP, mol_is_valid, smiles_from_mol


def mol_to_data(
    mol: Chem.Mol,
    *,
    hydrogen_mode: str = "explicit",
    max_atoms: int | None = None,
    allowed_atom_symbols: tuple[str, ...] | None = None,
) -> Data | None:
    """Filter a sanitized conformer and encode its explicit or heavy-atom graph.

    Filters use the source atom count. RDKit removes H before extracting fields,
    preserving heavy-atom order, coordinates, charges, and stereochemistry.
    """
    if hydrogen_mode not in ("explicit", "implicit"):
        raise ValueError("hydrogen_mode must be explicit or implicit")
    if not mol_is_valid(mol, allow_charged=True):
        return None
    if max_atoms is not None and mol.GetNumAtoms() > max_atoms:
        return None
    if allowed_atom_symbols is not None and any(
        atom.GetSymbol() not in allowed_atom_symbols for atom in mol.GetAtoms()
    ):
        return None
    if hydrogen_mode == "implicit":
        if all(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
            raise ValueError("Cannot represent a molecule with no heavy atoms")
        mol = Chem.RemoveAllHs(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    pairs, labels = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pairs.extend(((i, j), (j, i)))
        labels.extend([BOND_IDX_MAP[bond.GetBondType()]] * 2)
    edges = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t()
    order = (edges[0] * mol.GetNumAtoms() + edges[1]).argsort()
    data = Data(
        z=torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()]),
        pos=torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float),
        charges=torch.tensor([atom.GetFormalCharge() for atom in mol.GetAtoms()]),
        edge_index=edges[:, order].contiguous(),
        edge_attr=torch.tensor(labels, dtype=torch.long)[order],
        smiles=smiles_from_mol(mol, canonical=True) or "",
    )

    return data if torch.isfinite(data.pos).all() else None
