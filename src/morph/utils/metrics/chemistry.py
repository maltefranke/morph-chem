"""RDKit-mol metrics + the helpers they need (process pool, explicit valence
tables, PoseBusters). Everything in this file pulls RDKit; the tensor /
distribution side stays RDKit-free.

Includes adaptations from SemlaFlow's semlaflow/util/metrics.py,
Copyright (c) 2024 Ross Irwin. The upstream MIT notice is retained in
morph/_vendor/licenses/SEMLAFLOW_LICENSE; see morph/_vendor/README.md.
"""

import faulthandler
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torchmetrics import Metric

from morph.utils import rdkit_utils as morphRD
from morph.utils.metrics.scaffold_posebusters import scaffold_posebusters_metrics

faulthandler.enable()
# Silence RDKit warnings (in addition to morph.utils.rdkit_utils) because
# this module may be imported first from some code paths.
RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class GenerativeMetric(Metric):
    """torchmetrics base for metrics consuming a list of RDKit mols."""

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """Accumulate statistics from a batch of RDKit molecules."""
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        """Reduce the accumulated molecular statistics to a metric value."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Shared process pool for CPU-heavy RDKit work (optimise_mol / calc_energy).
#
# Why this exists:
#   * MMFF optimisation + energy calc is Python/C serial per-molecule.
#   * Running it on the main thread inside validation_step starves the
#     TorchElastic rendezvous keep-alive thread and triggers
#     RendezvousTimeoutError on long val cycles.
#   * We use a "spawn" context so the children do not inherit the parent's
#     CUDA context (fork + CUDA = undefined behavior).
#   * The pool is lazy so simply importing this module does not pay the
#     spawn cost; it starts on the first validation step.
# ---------------------------------------------------------------------------

_RD_POOL: ProcessPoolExecutor | None = None
_RD_POOL_MAX_WORKERS = int(os.environ.get("MORPH_RD_POOL_WORKERS", "4"))
_PB_POOL: ProcessPoolExecutor | None = None
_PB_POOL_WORKERS = 0


def _get_rdkit_pool() -> ProcessPoolExecutor:
    global _RD_POOL  # noqa: PLW0603 -- one reusable process-local worker pool
    if _RD_POOL is None:
        affinity = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1
        )
        # Leave at least half the cores for DataLoader workers + rendezvous.
        n = max(1, min(_RD_POOL_MAX_WORKERS, affinity // 2))
        _RD_POOL = ProcessPoolExecutor(
            max_workers=n,
            mp_context=mp.get_context("spawn"),
        )
    return _RD_POOL


def _get_posebusters_pool(max_workers: int) -> ProcessPoolExecutor:
    """Return a persistent spawn-safe pool for chunked PoseBusters checks."""
    global _PB_POOL, _PB_POOL_WORKERS  # noqa: PLW0603
    if _PB_POOL is None or max_workers != _PB_POOL_WORKERS:
        if _PB_POOL is not None:
            _PB_POOL.shutdown(wait=True)
        _PB_POOL = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
        )
        _PB_POOL_WORKERS = max_workers
    return _PB_POOL


def _pool_map(fn, items: list) -> list:
    """Run ``fn`` over ``items`` in the shared RDKit pool. Falls back to a
    plain list comprehension if the pool cannot be constructed (e.g. spawn is
    unavailable) so metrics never break training.
    """
    if not items:
        return []
    try:
        pool = _get_rdkit_pool()
        n_workers = pool._max_workers  # type: ignore[attr-defined]
        chunk = max(1, len(items) // (n_workers * 4))
        return list(pool.map(fn, items, chunksize=chunk))
    except Exception:
        return [fn(x) for x in items]


def _rd_optimise_mol(mol):
    return morphRD.optimise_mol(mol)


# Validation runs MMFF optimisation up to 4x over an identical batch
# (opt-energy-validity + strain + strain-per-atom + opt-rmsd), each calling
# ``_pool_map(_rd_optimise_mol, ...)`` on the same ordered non-None mols.
# ``calc_metrics_`` feeds the *same* list object to every metric in the
# collection, so memoize one shared optimise pass per batch keyed on the list
# identity. A reference to ``mols`` is held while cached so its ``id`` cannot be
# recycled (size-1 cache); a different list object recomputes. Correctness does
# not depend on the cache hitting -- a miss simply recomputes. ``optimise_mol``
# copies its input and the consumers (energy, conf_distance) are read-only, so
# sharing the optimised mols across metrics is safe.
_OPT_CACHE: dict = {"mols": None, "optimised": None}
_SMILES_CACHE: dict = {"mols": None, "smiles": None}


def _shared_optimise(mols: list) -> list:
    """MMFF-optimise ``[m for m in mols if m is not None]`` once per batch,
    shared across every optimise-using metric. Returns the optimised mols
    aligned to that non-None subset (in original order; ``None`` where
    optimisation failed).
    """
    if _OPT_CACHE["mols"] is not mols:
        non_none = [m for m in mols if m is not None]
        _OPT_CACHE["mols"] = mols
        _OPT_CACHE["optimised"] = _pool_map(_rd_optimise_mol, non_none)
    return _OPT_CACHE["optimised"]


def _shared_canonical_smiles(mols: list) -> list[str | None]:
    """Canonicalise one generated population once for uniqueness and novelty."""
    if _SMILES_CACHE["mols"] is not mols:
        _SMILES_CACHE["mols"] = mols
        _SMILES_CACHE["smiles"] = [
            None if mol is None else morphRD.smiles_from_mol(mol, canonical=True)
            for mol in mols
        ]
    return _SMILES_CACHE["smiles"]


def _rd_calc_energy(mol):
    return morphRD.calc_energy(mol)


def _rd_calc_energy_per_atom(mol):
    return morphRD.calc_energy(mol, per_atom=True)


def _is_valid_float(num):
    return num is not None and math.isfinite(float(num))


def _canonical_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


# ---------------------------------------------------------------------------
# Stability helpers
# ---------------------------------------------------------------------------


def calc_atom_stabilities(mol):
    """Per-atom RDKit-sanity flags. ``True`` means *stable* (no problem
    detected), ``False`` means the atom is involved in a
    ``DetectChemistryProblems`` finding. Aligned so that downstream metrics
    named ``*_stability`` report a *higher = better* number.
    """
    # Diagnose the raw graph on a copy: sanitization would discard the very
    # valence/kekulization failures this metric is meant to count.
    mol = Chem.Mol(mol)
    stabilities = [True] * mol.GetNumAtoms()
    for p in Chem.DetectChemistryProblems(mol):
        if hasattr(p, "GetAtomIdx"):
            stabilities[p.GetAtomIdx()] = False
        elif hasattr(p, "GetAtomIndices"):
            for index in p.GetAtomIndices():
                stabilities[index] = False
        else:
            raise ValueError(f"Cannot assign RDKit problem to atoms: {p.Message()}")
    return stabilities


ALLOWED_VALENCES_BY_ELEMENT = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    # QM9 contains N+ in forms such as NH+ and NH2+.
    "N": {0: [2, 3], 1: [2, 3, 4], -1: 2},
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}


def _matches_allowed_valence(valence, allowed, charge):
    if isinstance(allowed, int):
        return allowed == valence
    if isinstance(allowed, list):
        return valence in allowed
    if isinstance(allowed, dict):
        sub = allowed.get(charge)
        if sub is None:
            return False
        return _matches_allowed_valence(valence, sub, charge)
    return False


def mol_passes_sanitization(
    mol: Chem.rdchem.Mol, with_hs: bool = True, connected: bool = True
) -> bool:
    """Check sanitisation and optional connectivity, allowing charges and radicals."""
    if mol is None:
        return False
    mol_copy = Chem.Mol(mol)
    if not with_hs:
        mol_copy = Chem.RemoveAllHs(mol_copy)
    try:
        AllChem.SanitizeMol(mol_copy)
    except Exception:
        return False
    return not (connected and len(AllChem.GetMolFrags(mol_copy)) != 1)


def calc_valence_table_stabilities(mol):
    """Check each atom's total valence against the element/formal-charge table.

    Total valence includes implicit hydrogens. Unknown elements or charge states
    absent from an element's charge-specific rules fail. This is independent of
    RDKit's chemistry-problem flags and imposes no separate radical check.
    """
    stabilities = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in ALLOWED_VALENCES_BY_ELEMENT:
            stabilities.append(False)
            continue
        stabilities.append(
            _matches_allowed_valence(
                atom.GetTotalValence(),
                ALLOWED_VALENCES_BY_ELEMENT[sym],
                atom.GetFormalCharge(),
            )
        )
    return stabilities


# ---------------------------------------------------------------------------
# PoseBusters
# ---------------------------------------------------------------------------


def _posebusters_config() -> dict:
    """The reliable stock ``mol`` checks without energy/strain or flatness.

    Upstream ``config/mol.yml`` includes an ``energy_ratio`` entry without
    ``chosen_binary_test_output``. It contributes no column when
    ``full_report=False`` but can still execute, so outputless modules are
    removed together with every energy-ratio entry.

    Energy relaxation dominates runtime and is deliberately outside this
    structural plausibility check. PoseBusters' RDKit flatness implementation
    can segfault on malformed generated structures; one such child terminates
    the process pool and discards every metric. Keep the robust cheap checks
    (sanity, connectivity, radicals, and distance geometry) in routine
    validation. Energy/strain and flatness can be run as separate, explicitly
    requested analyses on a prevalidated molecule set.
    """
    # PoseBusters stays out of the ordinary metric import path.
    import posebusters  # noqa: PLC0415
    from yaml import safe_load  # noqa: PLC0415

    cfg_path = Path(posebusters.__file__).parent / "config" / "mol.yml"
    cfg = safe_load(cfg_path.read_text(encoding="utf-8"))

    # We always call `bust(..., full_report=False)`, which returns only the
    # columns named in `chosen_binary_test_output`. A module without that key
    # therefore produces nothing while still running — drop it. Remove all
    # explicitly unsupported modules in the second filter.
    modules = [m for m in cfg.get("modules", []) if m.get("chosen_binary_test_output")]
    modules = [
        m for m in modules if m.get("function") not in {"energy_ratio", "flatness"}
    ]
    cfg["modules"] = modules
    return cfg


def _bust_posebusters_chunk(mols: list[Chem.rdchem.Mol]) -> pd.DataFrame:
    """Run PoseBusters serially on one worker-owned molecule chunk."""
    from posebusters import PoseBusters  # noqa: PLC0415

    return PoseBusters(config=_posebusters_config(), max_workers=0).bust(
        mols, None, None
    )


def calc_posebusters_metrics(
    rdkit_mols: list[Chem.rdchem.Mol | None],
    *,
    max_workers: int | None = None,
    scaffold_contexts: list[dict | None] | None = None,
) -> dict[str, float]:
    """PoseBusters plausibility checks, reported with explicit denominators.

    ``rdkit_mols`` is the FULL generated list, ``None`` where the molecule could
    not be built/sanitised. PoseBusters can only run on the non-``None`` subset,
    so every rate has one of two bases and we report both plus their counts:

    * ``*_over_busted``   — denominator is the molecules PoseBusters evaluated
      (the sanitisable subset). This is what the stock implementation reported,
      and what Morph's published per-check table is on.
    * ``*_over_generated`` — denominator is ``len(rdkit_mols)``, with
      unsanitisable molecules counted as failures. This is the convention
      TABASCO uses ("we save all outputs from the model, as opposed to only
      valid molecules").

    ``all_pass`` implements the headline definition — *a molecule is only
    PoseBusters-valid if it passes every test*. A check that could not be
    evaluated (NaN from an unavailable structural check, for example) counts
    as a FAILURE, never as a pass: ``bool(float("nan"))`` is ``True`` in Python,
    so a naive ``df.all(axis=1)`` would silently mark those molecules valid.
    ``n_incomplete`` and the per-check ``*__n_nan`` keys expose exactly how many
    molecules that affects, and ``all_pass_complete_only`` gives the rate over
    molecules where every check did evaluate.

    ``max_workers=None`` (the default) uses up to eight allocation-bound CPU
    workers. PoseBusters' own parallel path pickles its complete input dataframe
    once for every molecule, which is quadratic in the population size. Instead,
    each spawned worker receives one disjoint chunk and runs it in-process.
    Pass ``0`` or a negative value to force in-process execution.
    """
    n_generated = len(rdkit_mols)
    if scaffold_contexts is not None and len(scaffold_contexts) != n_generated:
        raise ValueError("Scaffold contexts must align with every generated attempt")
    valid_mols = [mol for mol in rdkit_mols if mol is not None]
    valid_contexts = (
        [
            ctx
            for mol, ctx in zip(rdkit_mols, scaffold_contexts, strict=True)
            if mol is not None
        ]
        if scaffold_contexts is not None
        else None
    )
    n_busted = len(valid_mols)
    if not valid_mols:
        strict_validity = 0.0 if n_generated else float("nan")
        results = {
            "validity": strict_validity,
            "n_generated": float(n_generated),
            "n_busted": 0.0,
            "n_all_pass": 0.0,
            "all_pass_over_generated": strict_validity,
        }
        if valid_contexts is not None:
            results.update(
                scaffold_posebusters_metrics(pd.DataFrame(), [], [], n_generated, {})
            )
        return results

    if max_workers is None:
        affinity = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1
        )
        max_workers = max(1, min(8, affinity))
    # Keep the heavy optional dependency and its CUDA probing out of ordinary
    # training startup. The local config always excludes energy/strain.
    from posebusters import PoseBusters  # noqa: PLC0415

    if max_workers > 0 and PoseBusters.__module__.startswith("posebusters."):
        chunk_size = math.ceil(n_busted / max_workers)
        chunks = [
            valid_mols[start : start + chunk_size]
            for start in range(0, n_busted, chunk_size)
        ]
        frames = list(
            _get_posebusters_pool(max_workers).map(_bust_posebusters_chunk, chunks)
        )
        df = pd.concat(frames, ignore_index=True)
    else:
        df = PoseBusters(config=_posebusters_config(), max_workers=max_workers).bust(
            valid_mols, None, None
        )

    results: dict[str, float] = {
        "n_generated": float(n_generated),
        "n_busted": float(n_busted),
    }

    # Per-check rates. `pd.to_numeric` maps True/False -> 1.0/0.0 and leaves
    # un-evaluated entries as NaN; `.mean()` skips NaN, so we record the
    # evaluated count alongside every rate rather than hiding the shrinkage.
    check_cols: list[str] = []
    for col in df.columns:
        if col in ("file", "molecule"):
            continue
        try:
            numeric = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            print(f"Warning: could not process PoseBusters column '{col}'.")
            continue
        check_cols.append(col)
        n_nan = int(numeric.isna().sum())
        n_eval = n_busted - n_nan
        results[col] = float(numeric.mean())
        results[f"{col}__n_eval"] = float(n_eval)
        results[f"{col}__n_nan"] = float(n_nan)
        # Same check, but every un-evaluated molecule counts as a failure and
        # the denominator is every generated molecule.
        results[f"{col}__over_generated"] = float(
            numeric.fillna(0.0).sum() / n_generated
        )

    checks = df[check_cols].apply(pd.to_numeric, errors="coerce")
    if valid_contexts is not None:
        geometry = next(
            m
            for m in _posebusters_config()["modules"]
            if m["function"] == "distance_geometry"
        )
        results.update(
            scaffold_posebusters_metrics(
                checks,
                valid_mols,
                valid_contexts,
                n_generated,
                geometry.get("parameters", {}),
            )
        )

    if not check_cols:
        results["n_all_pass"] = 0.0
        results["all_pass_over_generated"] = 0.0
        results["validity"] = 0.0
        return results

    evaluated = checks.notna().all(axis=1)
    passed = checks.fillna(0.0).astype(bool).all(axis=1)  # NaN -> failure
    n_all_pass = int(passed.sum())
    n_incomplete = int((~evaluated).sum())

    results["n_checks"] = float(len(check_cols))
    results["n_all_pass"] = float(n_all_pass)
    results["n_incomplete"] = float(n_incomplete)
    results["all_pass_over_busted"] = n_all_pass / n_busted
    results["all_pass_over_generated"] = n_all_pass / n_generated
    # Canonical aggregate: every enabled check must pass, and failed molecule
    # construction is a failure because the denominator is all generations.
    results["validity"] = results["all_pass_over_generated"]
    # Rate over molecules where every check actually evaluated, for comparison
    # against the strict number above.
    n_complete = int(evaluated.sum())
    results["n_complete"] = float(n_complete)
    results["all_pass_complete_only"] = (
        float((passed & evaluated).sum()) / n_complete if n_complete else float("nan")
    )
    return results


# ---------------------------------------------------------------------------
# RDKit-stability metrics (fed precomputed bools)
# ---------------------------------------------------------------------------


class AtomStability(Metric):
    """Fraction of atoms passing RDKit ``DetectChemistryProblems``. Fed
    precomputed per-atom bools by ``calc_atom_stabilities`` (True = stable).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("atom_stable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, stabilities: list[list[bool]]) -> None:
        """Accumulate per-atom stability flags across molecules."""
        flat = [b for mol in stabilities for b in mol]
        self.atom_stable += sum(flat)
        self.total += len(flat)

    def compute(self) -> torch.Tensor:
        """Return the fraction of evaluated atoms that are stable."""
        return self.atom_stable.float() / self.total


class MoleculeStability(Metric):
    """Fraction of molecules where every atom is stable."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("mol_stable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, stabilities: list[list[bool]]) -> None:
        """Count molecules whose atoms are all stable."""
        mol_stables = [len(mol) > 0 and all(mol) for mol in stabilities]
        self.mol_stable += sum(mol_stables)
        self.total += len(mol_stables)

    def compute(self) -> torch.Tensor:
        """Return the fraction of evaluated molecules that are stable."""
        return self.mol_stable.float() / self.total


# ---------------------------------------------------------------------------
# Sanitisation validity and explicit valence-table checks (RDKit-mol input)
# ---------------------------------------------------------------------------


class SanitizationValidity(GenerativeMetric):
    """Sanitisable, by default connected molecules divided by all attempts.

    Applies no additional charge or radical filter. ``Validity`` also rejects
    radicals and, unless configured otherwise, nonzero net formal charge.
    """

    def __init__(self, with_hs: bool = True, connected: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.with_hs = with_hs
        self.connected = connected
        self.add_state("valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """Count molecules passing sanitisation and the connectivity policy."""
        is_valid = [
            mol_passes_sanitization(mol, with_hs=self.with_hs, connected=self.connected)
            for mol in mols
        ]
        self.valid += sum(is_valid)
        self.total += len(mols)

    def compute(self) -> torch.Tensor:
        """Return valid molecules divided by all attempted molecules."""
        return self.valid.float() / self.total


class ValenceTableAtomStability(GenerativeMetric):
    """Atoms matching the element/charge valence table divided by evaluable atoms.

    Unbuildable molecules and failed valence evaluations are excluded.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("atom_stable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """Accumulate valence-based atom stability for evaluable molecules."""
        for mol in mols:
            if mol is None:
                continue
            try:
                stabilities = calc_valence_table_stabilities(mol)
            except Exception:
                continue
            self.atom_stable += sum(stabilities)
            self.total += len(stabilities)

    def compute(self) -> torch.Tensor:
        """Return stable atoms divided by all evaluable atoms."""
        return self.atom_stable.float() / self.total


class ValenceTableMoleculeStability(GenerativeMetric):
    """All atoms matching the element/charge valence table, over all attempts.

    Empty molecules, unbuildable molecules and failed evaluations count as failures.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("mol_stable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """Count fully stable molecules, retaining failures in the denominator."""
        for mol in mols:
            if mol is None:
                self.total += 1
                continue
            try:
                stabilities = calc_valence_table_stabilities(mol)
            except Exception:
                self.total += 1
                continue
            if len(stabilities) > 0 and all(stabilities):
                self.mol_stable += 1
            self.total += 1

    def compute(self) -> torch.Tensor:
        """Return stable molecules divided by all attempted molecules."""
        return self.mol_stable.float() / self.total


# ---------------------------------------------------------------------------
# Validity / uniqueness / novelty
# ---------------------------------------------------------------------------


class Validity(GenerativeMetric):
    """Fraction of attempted molecules passing Morph's chemistry checks."""

    def __init__(self, allow_charged: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.allow_charged = allow_charged
        self.add_state("valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """Count valid molecules, including failed decodes in the denominator."""
        is_valid = [
            morphRD.mol_is_valid(mol, allow_charged=self.allow_charged)
            for mol in mols
            if mol is not None
        ]
        self.valid += sum(is_valid)
        self.total += len(mols)

    def compute(self) -> torch.Tensor:
        """Return valid molecules divided by all attempted molecules."""
        return self.valid.float() / self.total


class Uniqueness(GenerativeMetric):
    """Uniqueness among molecules that can be converted to canonical SMILES.

    State is process-local. Distributed callers must gather molecules before
    updating; Morph validation does this on rank zero for the full population.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valid_smiles = []

    def reset(self):
        """Clear accumulated canonical SMILES and metric state."""
        super().reset()
        self.valid_smiles = []

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """Collect canonical SMILES for successfully converted molecules."""
        # Use the same canonicalizer as Novelty (RemoveHs + canonical MolToSmiles
        # with exception-safe None fallback) so the two metrics share a SMILES
        # space and comparable denominators.
        smiles = _shared_canonical_smiles(mols)
        self.valid_smiles.extend(smi for smi in smiles if smi is not None)

    def compute(self) -> torch.Tensor:
        """Return distinct canonical SMILES divided by valid SMILES count."""
        if not self.valid_smiles:
            return torch.tensor(float("nan"))
        return torch.tensor(len(set(self.valid_smiles))) / len(self.valid_smiles)


class Diversity(GenerativeMetric):
    """Mean pairwise Tanimoto *distance* over ECFP fingerprints. Higher = more diverse.

    Matches TABASCO's ``MolecularDiversity``, which calls ``datamol.pdist`` ->
    ``scipy.spatial.distance.pdist(metric="jaccard")``. Two details are needed
    for their number to be comparable: datamol's ECFP default is **radius 3**
    (ECFP6), not RDKit's usual radius 2, and the reported quantity is a
    *distance* (1 - similarity), so their 0.89 means mean similarity ~0.11.

    Population is every molecule accumulated since ``reset()``, not a per-batch
    average: pairs are formed across the whole set in ``compute()``.
    """

    def __init__(self, radius: int = 3, n_bits: int = 2048, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        self.n_bits = n_bits
        self.fps = []

    def reset(self):
        """Clear accumulated molecular fingerprints and metric state."""
        super().reset()
        self.fps = []

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """Collect Morgan fingerprints for successfully converted molecules."""
        for mol in mols:
            if mol is None:
                continue
            try:
                self.fps.append(
                    AllChem.GetMorganFingerprintAsBitVect(
                        mol, self.radius, nBits=self.n_bits
                    )
                )
            except Exception:
                continue

    def compute(self) -> torch.Tensor:
        """Return mean pairwise Tanimoto distance, or NaN for fewer than two."""
        if len(self.fps) < 2:
            return torch.tensor(float("nan"))
        # Sum over the upper triangle only; BulkTanimotoSimilarity is the C++
        # bulk path, so this stays O(n^2) comparisons without n^2 Python calls.
        total, n_pairs = 0.0, 0
        for i, fp in enumerate(self.fps[:-1]):
            sims = DataStructs.BulkTanimotoSimilarity(fp, self.fps[i + 1 :])
            total += len(sims) - sum(sims)  # distance = 1 - similarity
            n_pairs += len(sims)
        return torch.tensor(total / n_pairs)


class Novelty(GenerativeMetric):
    """Fraction of canonical generated SMILES absent from the training set."""

    def __init__(
        self,
        train_smiles: list[str],
        already_canonical: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if already_canonical:
            canonical = train_smiles
        else:
            affinity = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else os.cpu_count() or 1
            )
            n_workers = min(8, affinity)
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                canonical = list(executor.map(_canonical_smiles, train_smiles))

        self.smiles = {smi for smi in canonical if smi is not None}

        self.add_state("novel", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """Count novel SMILES among successfully canonicalized molecules."""
        smiles = _shared_canonical_smiles(mols)
        valid_smiles = [smi for smi in smiles if smi is not None]
        novel = [smi not in self.smiles for smi in valid_smiles]

        self.novel += sum(novel)
        self.total += len(novel)

    def compute(self) -> torch.Tensor:
        """Return novel molecules divided by canonicalized molecule count."""
        return self.novel.float() / self.total


# ---------------------------------------------------------------------------
# Energy / strain / RMSD
# ---------------------------------------------------------------------------


class EnergyValidity(GenerativeMetric):
    """Fraction of attempted molecules with a finite force-field energy."""

    def __init__(self, optimise=False, **kwargs):
        super().__init__(**kwargs)
        self.optimise = optimise
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """Count finite energies, optionally after geometry optimization."""
        num_mols = len(mols)
        non_none = [m for m in mols if m is not None]
        if self.optimise:
            non_none = [m for m in _shared_optimise(mols) if m is not None]
        energies = _pool_map(_rd_calc_energy, non_none)
        valid_energies = [e for e in energies if _is_valid_float(e)]
        self.n_valid += len(valid_energies)
        self.total += num_mols

    def compute(self) -> torch.Tensor:
        """Return finite-energy molecules divided by all attempted molecules."""
        return self.n_valid.float() / self.total


class AverageEnergy(GenerativeMetric):
    """Average energy for molecules for which energy can be calculated.

    Energy can't be calculated for invalid mols and pose optimisation isn't
    guaranteed to succeed; failures don't count towards the metric. Does not
    require sanitisation, but sanitising first is usually a good idea.
    """

    def __init__(self, optimise=False, per_atom=False, **kwargs):
        super().__init__(**kwargs)
        self.optimise = optimise
        self.per_atom = per_atom
        self.add_state("energy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_valid_energies", default=torch.tensor(0), dist_reduce_fx="sum"
        )

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """Accumulate finite molecular or per-atom force-field energies."""
        non_none = [m for m in mols if m is not None]
        if self.optimise:
            non_none = [m for m in _shared_optimise(mols) if m is not None]
        energy_fn = _rd_calc_energy_per_atom if self.per_atom else _rd_calc_energy
        energies = _pool_map(energy_fn, non_none)
        valid_energies = [e for e in energies if _is_valid_float(e)]
        self.energy += sum(valid_energies)
        self.n_valid_energies += len(valid_energies)

    def compute(self) -> torch.Tensor:
        """Return mean energy over successfully evaluated molecules."""
        return self.energy / self.n_valid_energies


# TODO: Add xTB as level of theory option and add forces as a metric


class AverageStrainEnergy(GenerativeMetric):
    """Energy difference between a molecule's pose and its optimised pose.

    Only counted when the molecule has an energy, pose optimisation succeeds,
    and the optimised-pose energy is calculable. Combine with
    ``EnergyValidity(optimise=True)`` to track the fraction of molecules this
    metric can be calculated on.
    """

    def __init__(self, per_atom=False, **kwargs):
        super().__init__(**kwargs)
        self.per_atom = per_atom
        self.add_state(
            "total_energy_diff", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """Accumulate energy drops between input and optimized geometries."""
        indexed = [(idx, m) for idx, m in enumerate(mols) if m is not None]
        if not indexed:
            return

        indices, _ = zip(*indexed, strict=True)
        optimised = _shared_optimise(mols)

        energy_fn = _rd_calc_energy_per_atom if self.per_atom else _rd_calc_energy

        opt_pairs = [
            (i, m) for i, m in zip(indices, optimised, strict=True) if m is not None
        ]
        if not opt_pairs:
            return
        opt_indices, opt_mols_list = zip(*opt_pairs, strict=True)
        opt_energies = _pool_map(energy_fn, list(opt_mols_list))

        pairs = [
            (i, e)
            for i, e in zip(opt_indices, opt_energies, strict=True)
            if e is not None
        ]
        if not pairs:
            return

        valid_indices, valid_energies = zip(*pairs, strict=True)
        original_energies = _pool_map(energy_fn, [mols[i] for i in valid_indices])

        energy_diffs = [
            orig - opt
            for orig, opt in zip(original_energies, valid_energies, strict=True)
        ]
        self.total_energy_diff += sum(energy_diffs)
        self.n_valid += len(energy_diffs)

    def compute(self) -> torch.Tensor:
        """Return mean energy drop for pairs with valid energies."""
        return self.total_energy_diff / self.n_valid


class AverageOptRmsd(GenerativeMetric):
    """Average RMSD between a molecule and its optimised pose.

    Only counted when the molecule is valid and pose optimisation succeeds.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_rmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """Accumulate alignment RMSD between input and optimized geometries."""
        indexed = [(idx, m) for idx, m in enumerate(mols) if m is not None]
        if not indexed:
            return

        indices, _ = zip(*indexed, strict=True)
        optimised = _shared_optimise(mols)

        pairs = [
            (i, m) for i, m in zip(indices, optimised, strict=True) if m is not None
        ]
        if not pairs:
            return

        valid_indices, opt_mols = zip(*pairs, strict=True)
        original_mols = [mols[i] for i in valid_indices]
        rmsds = [
            morphRD.conf_distance(m1, m2)
            for m1, m2 in zip(original_mols, opt_mols, strict=True)
        ]
        self.total_rmsd += sum(rmsds)
        self.n_valid += len(rmsds)

    def compute(self) -> torch.Tensor:
        """Return mean RMSD over successfully evaluated molecule pairs."""
        return self.total_rmsd / self.n_valid
