"""Build ordered property arrays once, outside distributed training workers."""

import io
import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from tqdm import tqdm

from morph.dataset.core.properties import PROPERTY_FNS, property_path
from morph.dataset.core.source import molecular_source_identity
from morph.utils.cache import atomic_write, file_lock


def _ordered_smiles(dataset):
    if hasattr(dataset, "iter_bytes"):
        for raw in dataset.iter_bytes():
            yield pickle.loads(raw)["smiles"]
    else:
        for index in range(len(dataset)):
            yield dataset.get(index).smiles


def build_property_array(dataset, name="logp", *, force=False):
    """Save finite float32 labels and source metadata in processed split order.

    SMILES sidecars can contain sorted unique molecules and must not be used
    as an index-aligned label source. Read the actual ordered source records.
    The bounded cache stores scalar values, never live RDKit molecules.
    """
    if name not in PROPERTY_FNS:
        raise ValueError(f"Unknown molecular property: {name}")
    split = dataset.split
    source = (
        dataset.source_identity()
        if hasattr(dataset, "source_identity")
        else molecular_source_identity(dataset, split=split)
    )
    expected = {
        "version": 1,
        "property": name,
        "rdkit_version": rdBase.rdkitVersion,
        "source": source,
    }
    path = Path(property_path(dataset.processed_dir, name, split))
    manifest = path.with_suffix(".json")
    with file_lock(manifest):
        if not force and path.is_file() and manifest.is_file():
            metadata = json.loads(manifest.read_text())
            values = np.load(path, mmap_mode="r")
            if (
                metadata == expected
                and values.dtype == np.float32
                and values.shape == (len(dataset),)
                and np.isfinite(values).all()
            ):
                return path

        @lru_cache(maxsize=65536)
        def score(smiles):
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is None:
                raise ValueError(f"Cannot calculate {name} from SMILES {smiles!r}")
            value = PROPERTY_FNS[name](molecule)
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite {name} for SMILES {smiles!r}")
            return value

        values = np.fromiter(
            (
                score(smiles)
                for smiles in tqdm(
                    _ordered_smiles(dataset), total=len(dataset), desc=f"{split} {name}"
                )
            ),
            dtype=np.float32,
        )
        if values.shape != (len(dataset),) or not np.isfinite(values).all():
            raise ValueError("Property values must match the finite source records")
        buffer = io.BytesIO()
        np.save(buffer, values)
        atomic_write(path, buffer.getvalue())
        atomic_write(manifest, (json.dumps(expected, indent=2) + "\n").encode())
    return path
