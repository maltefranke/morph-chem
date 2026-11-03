"""PoseCheck strain energy (Harris et al., 2023).

Upstream MIT notices: morph/_vendor/licenses/POSECHECK_LICENSE
(Copyright (c) 2023 Charlie Harris) and TABASCO_LICENSE in the same directory
(Copyright (c) 2025 Carlos Vonessen). See morph/_vendor/README.md for provenance.

This is the metric TABASCO's "Strain Energy" column reports, and it is *not*
what ``AverageStrainEnergy`` in ``chemistry.py`` computes. The two differ in
both terms of the difference:

                        this module (PoseCheck)      AverageStrainEnergy
    force field         UFF                          MMFF94
    minuend             pose after a POSITION-       raw pose, unrelaxed
                        CONSTRAINED relaxation
                        (maxDispl 0.1 A, k = 1e5)
    subtrahend          min over 50 freshly embedded global minimum of the
                        + UFF-relaxed conformers,    *given* pose only
                        plus the pose relaxed freely

Both differences push the same way: PoseCheck's constrained relaxation removes
local geometric noise so the number reflects *conformational* strain rather
than bad bond lengths, while its global reference is <= the nearest local
minimum. So the two metrics are not on a common scale and must not be compared.

There is a third difference that dwarfs both, in the AGGREGATION:

    TABASCO reports the MEDIAN. ``AverageStrainEnergy`` reports the MEAN.

TABASCO's ``src/tabasco/utils/metrics.py`` calls PoseCheck's
``calculate_strain_energy`` and aggregates with ``mode`` defaulting to
``"median"``. The per-molecule distribution has a brutal right tail -- on a
6-molecule smoke test the mean was 20.19 against a median of 1.09 -- so the two
statistics differ by more than an order of magnitude and the choice must be
stated with any published number. This module therefore defaults to the median,
matching the table it exists to be compared against.

``AverageStrainEnergy`` in ``chemistry.py`` follows the upstream MMFF94 metric
with mean aggregation; see ``docs/development.md`` for its attribution.

References:
  - https://github.com/cch1999/posecheck ``posecheck/utils/strain.py``
  - https://github.com/carlosinator/tabasco ``src/tabasco/utils/metrics.py``

Cost is ~0.9 s/molecule serial (51 UFF relaxations), ~0.15 s/molecule across the
shared pool. Evaluation-time only -- it is too expensive for a per-step training
objective without a cheaper surrogate.
"""

from __future__ import annotations

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from .chemistry import GenerativeMetric, _pool_map

# PoseCheck defaults, from posecheck/utils/strain.py.
NUM_CONFS = 50
MAX_DISPLACEMENT = 0.1
FORCE_CONSTANT = 1.0e5


def _uff_energy(mol: Chem.Mol, conf_id: int = 0) -> float | None:
    try:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        return float(ff.CalcEnergy())
    except Exception:
        return None


def _local_energy(mol: Chem.Mol) -> float | None:
    """UFF energy after relaxing the pose under per-atom position restraints.

    The restraints are what make this "strain of *this* conformer" rather than
    "how bad is this geometry": every atom may move at most ``MAX_DISPLACEMENT``
    from where the model put it, so clashes and stretched bonds relax out while
    the conformational hypothesis is preserved.
    """
    probe = Chem.Mol(mol)
    try:
        ff = AllChem.UFFGetMoleculeForceField(probe, confId=0)
        for i in range(probe.GetNumAtoms()):
            ff.UFFAddPositionConstraint(i, MAX_DISPLACEMENT, FORCE_CONSTANT)
        ff.Minimize()
        return float(ff.CalcEnergy())
    except Exception:
        return None


def _global_energy(
    mol: Chem.Mol, num_confs: int = NUM_CONFS, seed: int = 0xF00D
) -> float | None:
    """Lowest UFF energy over freshly embedded conformers plus the relaxed pose.

    Including the freely-relaxed pose (PoseCheck's ``relax_global_on_pose``)
    matters: ETKDG can miss the basin the model actually generated, and without
    it a molecule whose pose is *better* than anything embedded would score a
    spurious negative strain.
    """
    energies: list[float] = []

    freed = Chem.Mol(mol)
    try:
        AllChem.UFFOptimizeMolecule(freed, maxIters=1000)
        e = _uff_energy(freed)
        if e is not None:
            energies.append(e)
    except Exception:
        pass

    embedded = Chem.Mol(mol)
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        embedded.RemoveAllConformers()
        AllChem.EmbedMultipleConfs(embedded, numConfs=num_confs, params=params)
        AllChem.UFFOptimizeMoleculeConfs(embedded, maxIters=1000)
        energies += [
            e
            for c in embedded.GetConformers()
            if (e := _uff_energy(embedded, c.GetId())) is not None
        ]
    except Exception:
        pass

    return min(energies) if energies else None


def posecheck_strain_energy(mol: Chem.Mol | None) -> float | None:
    """``local - global`` in kcal/mol; ``None`` when either term fails."""
    if mol is None or mol.GetNumConformers() == 0:
        return None
    local = _local_energy(mol)
    if local is None:
        return None
    glob = _global_energy(mol)
    if glob is None:
        return None
    return local - glob


class PoseCheckStrainEnergy(GenerativeMetric):
    """Mean PoseCheck strain over molecules where it is calculable.

    Molecules that fail (unembeddable, no UFF parameters) are dropped rather
    than counted as zero, matching how ``AverageStrainEnergy`` handles failures.

    Defaults to the MEDIAN because that is what TABASCO's table reports. Pass
    ``median=False`` for the mean, but never compare the two: the tail makes
    them differ by ~20x. ``self.values`` holds the raw per-molecule numbers if
    another statistic is wanted without paying for a second pass.
    """

    def __init__(self, median: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.median = median
        self.values: list[float] = []

    def reset(self):
        """Clear all accumulated strain energies."""
        super().reset()
        self.values = []

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """Accumulate strain energies for the provided molecules."""
        vals = _pool_map(posecheck_strain_energy, [m for m in mols if m is not None])
        self.values += [v for v in vals if v is not None]

    def compute(self) -> torch.Tensor:
        """Return the configured aggregate over accumulated energies."""
        if not self.values:
            return torch.tensor(float("nan"))
        t = torch.tensor(self.values, dtype=torch.float64)
        return (t.median() if self.median else t.mean()).float()
