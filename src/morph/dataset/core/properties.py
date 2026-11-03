"""Precomputed per-molecule scalar properties (QED, logP, …) for GEOM/QM9.

Each property is precomputed once into a float32 ``.npy`` array positionally aligned
to the split's LMDB index and stored at

    <processed_dir>/properties/<property>/<split>.npy

so ``__getitem__`` reads a single array element. Property arrays are optional
and supplied separately; ``compute_properties`` defaults to false. Each array
must contain one value per processed molecule in the split's dataset order.
``PROPERTY_FNS`` provides RDKit functions for offline array construction.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from rdkit.Chem import Crippen
from rdkit.Chem.QED import qed

if TYPE_CHECKING:
    from collections.abc import Callable

    from rdkit import Chem

PROPERTIES_SUBDIR = "properties"


def _qed(rdmol) -> float:
    return float(qed(rdmol))


def _logp(rdmol) -> float:
    return float(Crippen.MolLogP(rdmol))


# property name -> callable(rdmol) -> float
PROPERTY_FNS: dict[str, Callable[[Chem.Mol], float]] = {
    "qed": _qed,
    "logp": _logp,
}


def property_path(processed_dir: str, property_name: str, split: str) -> str:
    """Path of the precomputed ``.npy`` array for one property/split."""
    return os.path.join(processed_dir, PROPERTIES_SUBDIR, property_name, f"{split}.npy")


def load_property_array(
    processed_dir: str, property_name: str, split: str
) -> np.ndarray:
    """Load a precomputed property array (raises with build instructions if absent)."""
    path = property_path(processed_dir, property_name, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing precomputed property '{property_name}' for split '{split}' "
            f"at {path}. Run scripts/preprocess_properties.py with the same "
            "dataset preset and data.root_path before training."
        )
    return np.load(path)
