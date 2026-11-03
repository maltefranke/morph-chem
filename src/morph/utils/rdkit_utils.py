"""RDKit molecule construction, conversion, and checks.

Includes adaptations from SemlaFlow's semlaflow/util/rdkit.py,
Copyright (c) 2024 Ross Irwin. The upstream MIT notice is retained in
morph/_vendor/licenses/SEMLAFLOW_LICENSE; see morph/_vendor/README.md.
"""

import threading

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import NumRadicalElectrons
from scipy.spatial.transform import Rotation

# Silence RDKit's noisy validity/kekulize warnings. These fire once per
# generated molecule during validation and, on a shared Lustre stdout,
# can block every Python thread for minutes -- including the TorchElastic
# rendezvous keep-alive thread, which then times out and kills the job.
RDLogger.DisableLog("rdApp.*")

ArrT = np.ndarray


class PeriodicTable:
    """Thread-safe singleton wrapper for RDKit periodic-table queries."""

    _instance = None

    def __new__(cls):
        """Return the shared periodic-table wrapper instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self):
        self._table = Chem.GetPeriodicTable()

        # Just to be certain that vocab objects are thread safe
        self._pt_lock = threading.Lock()

    def atomic_from_symbol(self, symbol: str) -> int:
        """Look up an atomic number under the periodic-table lock."""
        with self._pt_lock:
            symbol = symbol.upper() if len(symbol) == 1 else symbol
            atomic = self._table.GetAtomicNumber(symbol)

        return atomic

    def symbol_from_atomic(self, atomic_num: int) -> str:
        """Look up an element symbol under the periodic-table lock."""
        with self._pt_lock:
            token = self._table.GetElementSymbol(atomic_num)

        return token

    def valence(self, atom: str | int) -> int:
        """Look up the default valence for an element symbol or atomic number."""
        with self._pt_lock:
            valence = self._table.GetDefaultValence(atom)

        return valence


PT = PeriodicTable()

IDX_BOND_MAP = {
    0: "NO_BOND",
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}
BOND_IDX_MAP = {bond: idx for idx, bond in IDX_BOND_MAP.items()}

# IDX_CHARGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: -1, 5: -2, 6: -3}
# CHARGE_IDX_MAP = {charge: idx for idx, charge in IDX_CHARGE_MAP.items()}


# TODO merge these with check functions in other files


def tensors_to_rdkit_mol(
    atom_types: list[str],
    coords: np.ndarray,
    charge_types: list[int],
    edge_types: list[Chem.BondType],
    edge_index: list[tuple[int, int]],
    sanitize: bool = True,
    implicit_hydrogens: bool = False,
):
    mol = Chem.EditableMol(Chem.Mol())

    # Add atoms with formal charges
    for idx, atomic in enumerate(atom_types):
        atom = Chem.Atom(atomic)
        charge = charge_types[idx]
        atom.SetFormalCharge(charge)
        atom.SetNoImplicit(not implicit_hydrogens)
        mol.AddAtom(atom)

    mol = mol.GetMol()

    # Add 3D coords
    conf = Chem.Conformer(len(atom_types))

    conf.SetPositions(np.ascontiguousarray(coords, dtype=np.float64))

    mol.AddConformer(conf, assignId=True)

    mol = Chem.EditableMol(mol)
    for edge, edge_type in zip(edge_index, edge_types, strict=True):
        start, end = edge
        mol.AddBond(start, end, edge_type)

    try:
        mol = mol.GetMol()
        # DetermineBonds(mol)
        for atom in mol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)
    except Exception:
        return None

    if sanitize:
        return sanitize_mol_correctly(mol, implicit_hydrogens=implicit_hydrogens)

    return mol


def sanitize_mol_correctly(mol: Chem.Mol, *, implicit_hydrogens: bool | None = False):
    """Sanitize in place; None preserves each atom's existing hydrogen policy."""
    for a in mol.GetAtoms():
        if implicit_hydrogens is not None:
            a.SetNoImplicit(not implicit_hydrogens)
        if a.HasProp("_MolFileHCount"):
            a.ClearProp("_MolFileHCount")

    flags = Chem.SanitizeFlags.SANITIZE_ALL
    if all(a.GetNoImplicit() for a in mol.GetAtoms()):
        flags &= ~Chem.SanitizeFlags.SANITIZE_ADJUSTHS

    # Explicit-H graphs must not acquire extra H; implicit-H graphs need ADJUSTHS.
    err = Chem.SanitizeMol(
        mol,
        sanitizeOps=flags,
        catchErrors=True,
    )
    if err:  # nonzero bitmask means some step failed
        return None
    return mol


def mol_is_valid(
    mol: Chem.Mol, allow_radical: bool = False, allow_charged: bool = True
) -> bool:
    # Validity is a predicate: do not let sanitization mutate the molecule that
    # subsequent metrics, serialization, or dataset processing will consume.
    if mol is None or mol.GetNumAtoms() == 0:
        return False
    sanitized_mol = sanitize_mol_correctly(Chem.Mol(mol), implicit_hydrogens=None)
    return not (
        sanitized_mol is None
        or len(Chem.GetMolFrags(sanitized_mol)) > 1
        or (not allow_radical and NumRadicalElectrons(sanitized_mol) != 0)
        or (not allow_charged and Chem.GetFormalCharge(sanitized_mol) != 0)
    )


def calc_energy(mol: Chem.Mol, per_atom: bool = False, level: str = "MMFF94") -> float:
    """Calculate the energy for an RDKit molecule using the MMFF forcefield

    Evaluate the first conformer on a copy, leaving the original unchanged.

    Args:
        mol (Chem.Mol): RDKit molecule
        per_atom (bool): Whether to normalise by the number of atoms.
        level: Force-field variant; currently only "MMFF94" is supported.

    Returns:
        float: Energy of the molecule or None if the energy could not be calculated
    """

    mol_copy = Chem.Mol(mol)

    try:
        if level == "MMFF94":
            mmff_props = AllChem.MMFFGetMoleculeProperties(
                mol_copy, mmffVariant="MMFF94"
            )
            ff = AllChem.MMFFGetMoleculeForceField(mol_copy, mmff_props, confId=0)
            energy = ff.CalcEnergy()
            energy = energy / mol.GetNumAtoms() if per_atom else energy
    except Exception:
        energy = None

    return energy


def optimise_mol(
    mol: Chem.rdchem.Mol, level: str = "MMFF94", max_iters: int = 1000
) -> Chem.rdchem.Mol:
    """Optimise the conformation of an RDKit molecule

    Optimize the first conformer on a copy, leaving the original unchanged.

    Args:
        mol (Chem.Mol): RDKit molecule
        max_iters (int): Maximum iterations for conformer optimization.
        level: Force-field variant; currently only "MMFF94" is supported.

    Returns:
        Chem.Mol: Optimized copy, or None if the force-field call raises an error.
    """

    mol_copy = Chem.Mol(mol)
    try:
        if level == "MMFF94":
            AllChem.MMFFOptimizeMolecule(mol_copy, maxIters=max_iters)
    except Exception:
        return None

    return mol_copy


def conf_distance(
    mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol, fix_order: bool = True
) -> float:
    """Approximately align two molecules and then calculate RMSD between them

    Alignment and distance use the default conformer of each molecule.

    Args:
        mol1 (Chem.Mol): First molecule to align
        mol2 (Chem.Mol): Second molecule to align
        fix_order (bool): Whether to fix the atom order of the molecules

    Returns:
        float: RMSD between molecules after approximate alignment
    """

    assert len(mol1.GetAtoms()) == len(mol2.GetAtoms())

    if not fix_order:
        raise NotImplementedError

    coords1 = np.array(mol1.GetConformer().GetPositions())
    coords2 = np.array(mol2.GetConformer().GetPositions())

    # Firstly, centre both molecules
    coords1 = coords1 - (coords1.sum(axis=0) / coords1.shape[0])
    coords2 = coords2 - (coords2.sum(axis=0) / coords2.shape[0])

    if coords1.shape[0] == 1:
        # catch the case where the molecule is a single atom
        return 0.0

    else:
        # Find the best rotation alignment between the centred mols
        rotation, _ = Rotation.align_vectors(coords1, coords2)
        aligned_coords2 = rotation.apply(coords2)

        sqrd_dists = (coords1 - aligned_coords2) ** 2
        rmsd = np.sqrt(sqrd_dists.sum(axis=1).mean())
        return rmsd


def smiles_from_mol(
    mol: Chem.rdchem.Mol, canonical: bool = True, explicit_hs: bool = False
) -> str | None:
    """Create a SMILES string from a molecule

    Args:
        mol (Chem.Mol): RDKit molecule object
        canonical (bool): Whether to create a canonical SMILES, default True
        explicit_hs (bool): Add explicit hydrogen atoms before creating the SMILES.
            If False, remove hydrogen atoms. Explicit and implicit hydrogen
            representations can produce different strings.

    Returns:
        str: SMILES string which could be None if the SMILES generation failed
    """

    if mol is None:
        return None

    try:
        mol = Chem.AddHs(mol) if explicit_hs else Chem.RemoveHs(mol)
        smiles = Chem.MolToSmiles(mol, canonical=canonical)
    except Exception:
        smiles = None

    return smiles
