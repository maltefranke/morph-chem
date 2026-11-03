# Training bundles

Scope: reproduce released training setups using frozen source/configuration.
Sources: [bundle downloads](../README.md#reproducibility) in
[maltefranke/morph](https://huggingface.co/maltefranke/morph).
For new runs with current code, use [training](training.md).

## Download and set up

Prerequisites: [uv](https://docs.astral.sh/uv/), Hub access (`hf auth login`), and a
compute allocation for training. Download a model's archive and checksum metadata:

```bash
hf download maltefranke/morph geom/explicit_h/training_bundle.tar.gz \
  geom/explicit_h/training_bundle.json --local-dir morph-training
cd morph-training
```

Run this Python snippet from `morph-training/` to verify before extraction:

```python
import hashlib
import json
from pathlib import Path

folder = Path("geom/explicit_h")
metadata = json.loads((folder / "training_bundle.json").read_text())
with (folder / metadata["archive"]).open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == metadata["sha256"]
```

```bash
tar -xzf geom/explicit_h/training_bundle.tar.gz
cd geom-explicit-h
uv venv --python 3.13.7 .venv
uv pip install --python .venv/bin/python -r pyproject.toml \
  -c reproduction/requirements-environment.txt
source .venv/bin/activate
export MORPH_DATASET_ROOT=/absolute/path/to/geom  # Contains processed/
export MORPH_OUTPUT_DIR="$PWD/outputs/repeat-01"
bash train.sh
```

| Hub folder | Extracted directory | Dataset directory |
| --- | --- | --- |
| `qm9/explicit_h` | `qm9-explicit-h` | `qm9` |
| `geom/explicit_h` | `geom-explicit-h` | `geom` |
| `chembl3d/explicit_h` | `chembl3d-explicit-h` | `chembl3d_max100` |
| `chembl3d/implicit_h` | `chembl3d-implicit-h` | `chembl3d_max100_implicit_h` |

The extracted directory is also recorded in `training_bundle.json`. Preserve the
archive and JSON with results; no version suffix is required in the download command.

## Bundle contents

Archives are about 0.2–0.5 MB, excluding datasets and weights.

| Path in archive | Purpose |
| --- | --- |
| `run.py`, `src/`, `configs/`, `pyproject.toml`, `uv.lock` where originally present | Frozen source, checked against saved SHA-256 manifest. |
| `reproduction/original_training_config.yaml`, `original_launcher.sbatch` | Historical settings/paths. QM9 config follows successful GPU preflight; launcher records preflight reuse and allocation limit. |
| `configs/reproduce.yaml`, `train.sh` | Portable fresh-training recipe. |
| `reproduction/provenance.json`, `dataset_reference.json`, `original_source_manifest.json`, `manifest.json` | Source/model releases, data references, file verification, and recipe changes. |
| `reproduction/requirements-environment.txt` | Shared environment captured **2026-10-08**, not historical per-job captures; reference Python 3.13.7. |
| `reproduction/environment.sh` | Explicit runtime flags. |

Each run records package/GPU/driver details, overrides, provenance, and data references
in `MORPH_OUTPUT_DIR`; Hydra saves effective config in `.hydra/`. Use a fresh output
directory and retain these records with checkpoints.

### Portable recipe settings

All recipes: two RTX 4090s, seed 42, BF16 mixed precision, SDPA, compiled networks,
4096 validation attempts. GEOM uses Muon CUDA graph execution; QM9/ChEMBL3D use native
Muon, disabled DDP compiler graph splitting, and compilation fallback.

| Model | Batch/GPU | Accumulation | Effective batch | Historical validation | Portable validation / saving | Default epoch limit |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| QM9 explicit H | 1024 | 2 | 4096 | Every epoch | Every 1 / 25 epochs | 3000 |
| GEOM explicit H | 320 | 1 | 640 | Every epoch | Every 1 / 5 epochs | 50 |
| ChEMBL3D explicit H | 320 | 1 | 640 | Every 5 epochs | Every 1 / 5 epochs | 400 |
| ChEMBL3D implicit H | 320 | 1 | 640 | Every epoch | Every 1 / 5 epochs | 300 |

Override the limit, e.g. `bash train.sh trainer.trainer.max_epochs=100` for GEOM
or 1000/1500 for published QM9 epochs. Portable configs preserve model, optimizer,
schedule, and batch settings; change data/output paths; disable W&B; remove historical
resume paths/timers; and apply the table's epoch/validation/save policy.
`provenance.json` lists differences; original configs remain available.

## Data needed for training

`MORPH_DATASET_ROOT` must contain `processed/`. Preserve split membership,
molecule/conformer and atom order, `*_smiles.txt`, ordered vocabularies, and
`distributions.pt`. GEOM/ChEMBL3D use `<split>_data.lmdb/data.mdb`; QM9 uses
`<split>.pt`. Training SMILES are required for novelty metrics.

| Dataset | Selection and split sizes (train / validation / test) |
| --- | --- |
| QM9 | Original explicit-H split: 100,000 / 20,000 / 12,947. |
| GEOM | At most 72 total atoms, including H. |
| ChEMBL3D | Both variants: cap 100 atoms **before removing H**; 1,474,008 / 184,295 / 184,242. |

Implicit-H ChEMBL3D repairs stale charges/bond representations from canonical source
SMILES and recomputes heavy-atom statistics; its cache cannot replace the explicit-H cache.

The launcher checks vocabulary/statistics SHA-256 and split file sizes. **File sizes do
not verify contents**; historical releases lack full dataset hashes. Datasets are not
inside the model repository. Rebuilding requires checking split membership, conformer
selection, and order, not just atom caps. Preprocess before distributed training and
retain the cache; compute full content hashes on a CPU worker for future comparisons.

## Fresh training versus resuming

| Released run | History |
| --- | --- |
| ChEMBL3D explicit H | Resumed after 5 completed epochs. |
| GEOM epochs 50/100 | Resumed from best checkpoint after 48 completed epochs. |
| ChEMBL3D implicit H; QM9 epochs 1000/1500 | Started fresh. |

Epoch-5/48 intermediate checkpoints are not released. Fresh seed-42 recipes reproduce
settings, not those resume histories. To continue an available full checkpoint, supply
its absolute path and the desired **total** completed epochs:

```bash
bash train.sh \
  "trainer.checkpoint_path='/absolute/path/to/epoch_050.ckpt'" \
  trainer.trainer.max_epochs=100
```

- `trainer.checkpoint_path` restores optimizer, scheduler, EMA, and loop state;
  inference weights alone do not resume training.
- Periodic names use zero-based Lightning epochs (49 = 50 completed). After-fit
  `final.ckpt` may have advanced the counter; inspect completed loop count.
- Portable recipes disable wall-clock timers. When resuming with `max_time`, Lightning
  restores elapsed timer state; include it in the new budget.
- Preserve source, data order, ranks, and microbatches for closest continuation.
  State restoration does not guarantee bitwise RNG/data-worker replay.
