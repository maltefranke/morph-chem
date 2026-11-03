"""Select scaffold atom sets in the input molecule's original indexing.

Murcko keeps the ring/linker core; R-group cuts retain the largest BRICS
fragment; heavy-atom cuts leave hydrogens to generate; linker cuts keep
multiple fragments, including their original hydrogens. The cached atom sets
feed the same keep-mask conditioning and integration path for every kind.
"""

from __future__ import annotations

import json
from contextlib import suppress
from functools import lru_cache
from itertools import combinations

from rdkit import Chem
from rdkit.Chem import BRICS, RWMol
from rdkit.Chem.Scaffolds import MurckoScaffold

from morph.dataset.conditional.scaffold.resources import libinvent_resources
from morph.utils.rdkit_utils import IDX_BOND_MAP

MAX_SCAFFOLD_AUTOMORPHISMS = 16
MAX_CUTS = 4
MAX_RGROUP_ATOMS = 10
MIN_SCAFFOLD_FRACTION = 0.4
MAX_SLICINGS = 32


@lru_cache(maxsize=1)
def libinvent_enumerator():
    """Load cached upstream rules, downloading them on first LibINVENT use."""
    try:
        from reinvent_chemistry.library_design import FragmentReactions  # noqa: PLC0415
        from reinvent_chemistry.library_design.dtos import (  # noqa: PLC0415
            FilteringConditionDTO,
        )
        from reinvent_chemistry.library_design.fragment_reaction_slice_enumerator import (  # noqa: PLC0415, E501
            FragmentReactionSliceEnumerator,
        )
    except ImportError as error:
        raise ImportError(
            "The libinvent scaffold cutter requires the scaffold extra: "
            "uv sync --extra scaffold"
        ) from error
    reactions, filters = libinvent_resources()
    smirks = [
        line.strip()
        for line in reactions.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    conditions = json.loads(filters.read_text())
    return FragmentReactionSliceEnumerator(
        FragmentReactions().create_reactions_from_smirks(smirks),
        [FilteringConditionDTO(**c) for c in conditions["scaffold"]],
        [FilteringConditionDTO(**c) for c in conditions["decoration"]],
    )


@lru_cache(maxsize=100_000)
def _libinvent_scaffold_smiles(smiles, max_cuts):
    # Fail loudly for missing dependencies/resources instead of caching an
    # empty dataset. Enumeration failures remain molecule-local as in morph.
    enumerator = libinvent_enumerator()
    try:
        sliced = enumerator.enumerate(Chem.MolFromSmiles(smiles), cuts=max_cuts)
    except (ValueError, RuntimeError):
        return ()
    return tuple(sorted({sm.scaffold_smiles for sm in sliced}))


def libinvent_indices(mol, max_cuts=MAX_CUTS, max_slicings=MAX_SLICINGS):
    """Map LibINVENT reaction scaffolds into the input molecule's atom order."""
    heavy_mol, heavy_map = _heavy_submol(mol)
    if heavy_mol is None:
        return []
    cuts, seen = [], set()
    for smiles in _libinvent_scaffold_smiles(Chem.MolToSmiles(heavy_mol), max_cuts):
        scaffold = Chem.MolFromSmiles(smiles, sanitize=False)
        if scaffold is None:
            continue
        query = Chem.RWMol(scaffold)
        for atom in reversed(list(scaffold.GetAtoms())):
            if atom.GetAtomicNum() == 0:
                query.RemoveAtom(atom.GetIdx())
        query = query.GetMol()
        with suppress(ValueError, RuntimeError):
            Chem.SanitizeMol(query)
        if not query.GetNumAtoms():
            continue
        for match in heavy_mol.GetSubstructMatches(query, uniquify=True, maxMatches=8):
            key = tuple(sorted(heavy_map[i] for i in match))
            if key not in seen:
                seen.add(key)
                cuts.append(key)
                if len(cuts) >= max_slicings:
                    return cuts
    return cuts


def _heavy_submol(mol):
    """Heavy-atom copy of ``mol`` plus a new-index -> original-index map.

    Bond-cut rules operate on heavy atoms. Removing hydrogens from a copy
    preserves aromatic flags, formal charges and the hydrogen counts needed
    to sanitize heteroaromatic rings. The map restores the dataset's indexing.
    """
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    if not heavy:
        return None, None
    try:
        out = (
            Chem.Mol(mol)
            if len(heavy) == mol.GetNumAtoms()
            else Chem.RemoveAllHs(mol, sanitize=False)
        )
        Chem.SanitizeMol(out)
    except Exception:
        return None, None
    # RemoveAllHs keeps the relative order of the remaining atoms, so the map is
    # positional -- but guard it, since a mismatch would silently mis-index every
    # scaffold produced for this molecule.
    if out.GetNumAtoms() != len(heavy):
        return None, None
    return out, heavy


def _brics_cuttable(heavy_mol) -> set:
    """Bond indices BRICS considers synthetically accessible."""
    cuttable = set()
    for (i, j), _ in BRICS.FindBRICSBonds(heavy_mol):
        bond = heavy_mol.GetBondBetweenAtoms(int(i), int(j))
        # Ring bonds would open a cycle rather than detach a substituent.
        if bond is not None and not bond.IsInRing():
            cuttable.add(bond.GetIdx())
    return cuttable


def _slice_on_bonds(
    mol,
    cuttable_fn,
    max_cuts: int,
    max_rgroup_atoms: int,
    min_scaffold_fraction: float,
    max_slicings: int,
) -> list[tuple]:
    """Cut 1..``max_cuts`` of the candidate bonds; largest fragment is the core.

    A slicing is rejected when a severed piece exceeds ``max_rgroup_atoms``
    (that is a fragmentation, not a decoration) or the core falls below
    ``min_scaffold_fraction`` of the molecule.
    """
    heavy_mol, heavy_map = _heavy_submol(mol)
    if heavy_mol is None:
        return []
    n_heavy = heavy_mol.GetNumAtoms()

    cuttable = sorted(cuttable_fn(heavy_mol))
    if not cuttable:
        return []

    # Collect per cut-count, then round-robin when truncating. Returning the
    # first ``max_slicings`` found instead would be a pure 1-cut sample, since
    # enumeration walks k ascending -- multi-R slicings would never survive.
    by_cuts, seen = {}, set()
    for k in range(1, min(max_cuts, len(cuttable)) + 1):
        bucket = by_cuts.setdefault(k, [])
        for combo in combinations(cuttable, k):
            try:
                fragmented = Chem.FragmentOnBonds(
                    heavy_mol, list(combo), addDummies=False
                )
                pieces = Chem.GetMolFrags(fragmented)
            except Exception:
                continue
            if len(pieces) < 2:
                continue
            core = max(pieces, key=len)
            if any(len(p) > max_rgroup_atoms for p in pieces if p is not core):
                continue
            if len(core) < min_scaffold_fraction * n_heavy:
                continue
            key = tuple(sorted(core))
            if key in seen:
                continue
            seen.add(key)
            bucket.append(tuple(sorted(heavy_map[i] for i in key)))

    slicings = []
    for rank in range(max(len(b) for b in by_cuts.values()) if by_cuts else 0):
        for k in sorted(by_cuts):
            if rank < len(by_cuts[k]):
                slicings.append(by_cuts[k][rank])
                if len(slicings) >= max_slicings:
                    return slicings
    return slicings


def murcko_indices(mol) -> list[tuple]:
    """All Bemis-Murcko scaffold matches, as heavy-atom index tuples."""
    heavy_mol, heavy_map = _heavy_submol(mol)
    if heavy_mol is None:
        return []
    try:
        scaffold_smiles = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(heavy_mol))
    except Exception:
        return []
    if not scaffold_smiles or "." in scaffold_smiles:
        return []
    scaffold = Chem.MolFromSmiles(scaffold_smiles)
    if scaffold is None:
        return []
    matches = heavy_mol.GetSubstructMatches(
        scaffold, uniquify=False, maxMatches=MAX_SCAFFOLD_AUTOMORPHISMS
    )
    return [tuple(sorted(heavy_map[i] for i in match)) for match in matches]


def heavy_indices(mol, **_ignored) -> list[tuple]:
    """The whole heavy-atom skeleton as the scaffold -- hydrogen decoration.

    A single match, not an automorphism list: the atoms are taken by index, not
    matched through a substructure query, so there is nothing to enumerate.
    """
    heavy = tuple(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
    # A molecule with no hydrogens has nothing to decorate -- drop it, the same
    # way an empty Murcko core drops a molecule from the scaffold task.
    return [heavy] if heavy and len(heavy) < mol.GetNumAtoms() else []


def rgroup_indices(
    mol,
    max_cuts: int = MAX_CUTS,
    max_rgroup_atoms: int = MAX_RGROUP_ATOMS,
    min_scaffold_fraction: float = MIN_SCAFFOLD_FRACTION,
    max_slicings: int = MAX_SLICINGS,
) -> list[tuple]:
    """BRICS slicings retaining the largest fragment as the scaffold."""
    return _slice_on_bonds(
        mol,
        _brics_cuttable,
        max_cuts,
        max_rgroup_atoms,
        min_scaffold_fraction,
        max_slicings,
    )


def linker_indices(
    mol,
    min_frag: int = 3,
    min_linker: int = 1,
    max_linker: int = 12,
    max_group: int = 3,
    max_slicings: int = MAX_SLICINGS,
    **_ignored,
) -> list[tuple]:
    """BRICS linker cuts retaining at least two fragments and their hydrogens.

    The removed linker is a connected group of up to ``max_group`` BRICS
    fragments with ``min_linker`` to ``max_linker`` heavy atoms. Each retained
    component has at least ``min_frag`` heavy atoms. Only acyclic bonds are cut;
    rings can occur inside the linker. Returned indices address the original
    all-atom molecule, including hydrogens belonging to retained fragments.
    """
    heavy_mol, heavy_map = _heavy_submol(mol)
    if heavy_mol is None:
        return []
    # BRICS bonds in FULL-atom indices (BRICS runs on the heavy skeleton; the bond
    # atoms are heavy). We fragment on the FULL graph so each fragment carries its
    # own hydrogens (GEOM is all-atom) and the kept fragments stay intact.
    brics = []
    for (i, j), _ in BRICS.FindBRICSBonds(heavy_mol):
        b = heavy_mol.GetBondBetweenAtoms(int(i), int(j))
        if b is not None and not b.IsInRing():
            brics.append((heavy_map[int(i)], heavy_map[int(j)]))
    if len(brics) < 2:  # need ≥2 cuts to isolate a linker between fragments
        return []

    N = mol.GetNumAtoms()
    is_heavy = [mol.GetAtomWithIdx(a).GetAtomicNum() > 1 for a in range(N)]
    adj = [[] for _ in range(N)]
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[u].append(v)
        adj[v].append(u)
    cut = {frozenset(e) for e in brics}

    # BRICS fragments = components of the FULL graph with BRICS bonds removed.
    frag_of = [-1] * N
    frag_atoms: list[set] = []
    for s in range(N):
        if frag_of[s] != -1:
            continue
        comp, stack = set(), [s]
        frag_of[s] = len(frag_atoms)
        while stack:
            x = stack.pop()
            comp.add(x)
            for y in adj[x]:
                if frozenset((x, y)) in cut or frag_of[y] != -1:
                    continue
                frag_of[y] = len(frag_atoms)
                stack.append(y)
        frag_atoms.append(comp)
    nf = len(frag_atoms)
    if nf < 3:  # need a linker + ≥2 fragments
        return []

    def nheavy(atoms):
        return sum(1 for a in atoms if is_heavy[a])

    fadj = [set() for _ in range(nf)]
    for u, v in brics:
        if frag_of[u] != frag_of[v]:
            fadj[frag_of[u]].add(frag_of[v])
            fadj[frag_of[v]].add(frag_of[u])

    # Connected fragment subsets (size 1..max_group) — linker candidates.
    subsets: set = set()
    stack = [frozenset([f]) for f in range(nf)]
    while stack:
        s = stack.pop()
        if s in subsets:
            continue
        subsets.add(s)
        if len(s) >= max_group:
            continue
        for f in s:
            for nb in sorted(fadj[f]):
                if nb not in s:
                    stack.append(s | {nb})

    keep_sets: set = set()
    for S in sorted(subsets, key=lambda s: (len(s), tuple(sorted(s)))):
        latoms = set().union(*(frag_atoms[f] for f in S))
        if not (min_linker <= nheavy(latoms) <= max_linker):  # size = HEAVY atoms
            continue
        # Kept fragment components after removing S; each must be ≥ min_frag heavy.
        seen: set = set(S)
        keep_atoms: set = set()
        comps, ok = 0, True
        for f in range(nf):
            if f in seen:
                continue
            comps += 1
            catoms, cstack = set(), [f]
            seen.add(f)
            while cstack:
                x = cstack.pop()
                catoms |= frag_atoms[x]
                for y in fadj[x]:
                    if y not in seen:
                        seen.add(y)
                        cstack.append(y)
            if nheavy(catoms) < min_frag:
                ok = False
                break
            keep_atoms |= catoms
        if ok and comps >= 2:
            keep_sets.add(
                tuple(sorted(keep_atoms))
            )  # FULL-atom kept fragments (with H)
            if len(keep_sets) >= max_slicings:
                break
    return sorted(keep_sets)


SCAFFOLD_KINDS = {
    "murcko": murcko_indices,
    "libinvent": libinvent_indices,
    "rgroup": rgroup_indices,
    "heavy": heavy_indices,
    "linker": linker_indices,
}


def scaffold_indices(mol, kind: str = "rgroup", **params) -> list[tuple[int, ...]]:
    """Return distinct kept atom sets for the selected scaffold convention."""
    try:
        cutter = SCAFFOLD_KINDS[kind]
    except KeyError:
        raise ValueError(
            f"unknown scaffold_kind {kind!r}; expected one of {sorted(SCAFFOLD_KINDS)}"
        ) from None
    # Atom order inside a cut is irrelevant to filter_nodes. Deduplicate
    # automorphisms so evaluation counts each retained subgraph only once.
    return list(dict.fromkeys(tuple(sorted(cut)) for cut in cutter(mol, **params)))


def _atom_indices_for_data(
    data, scaffold_kind: str = "rgroup"
) -> list[tuple[int, ...]]:
    """Return the chosen cutter's scaffold atom sets for one molecule.

    The molecule is reconstructed from stored atom and bond tensors so the
    matches stay in the dataset's atom-index space.
    """
    mol = mol_from_data(data)
    return [] if mol is None else scaffold_indices(mol, scaffold_kind)


def mol_from_data(data):
    """Reconstruct a sanitized molecule in the stored atom-index space."""
    rwmol = RWMol()
    charges = getattr(data, "charges", None)
    for idx, z_val in enumerate(data.z.tolist()):
        atom = Chem.Atom(int(z_val))
        if charges is not None:
            atom.SetFormalCharge(int(charges[idx]))
        rwmol.AddAtom(atom)
    for k in range(data.edge_index.shape[1]):
        i, j = int(data.edge_index[0, k]), int(data.edge_index[1, k])
        if i < j:
            rwmol.AddBond(i, j, IDX_BOND_MAP[int(data.edge_attr[k])])
    try:
        mol = rwmol.GetMol()
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def compute_scaffold_atom_indices(
    dataset, scaffold_kind: str = "rgroup"
) -> list[list[tuple[int, ...]]]:
    """Return alternative scaffold atom sets for every dataset molecule."""
    return [
        _atom_indices_for_data(dataset.get(idx), scaffold_kind)
        for idx in range(len(dataset))
    ]
