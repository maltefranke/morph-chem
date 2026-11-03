# Pretrained models and inference

Scope: compatible unconditional bundles via CLI/Python, without dataset preparation.
Sources: [loader](../src/morph/inference/hub.py), [catalog](../src/morph/inference/catalog.py).
Related: [training bundles](reproducibility.md), [offline evaluation](evaluation.md).
The CLI and Python API work from any directory after package installation.
Commands invoking repository scripts run from the repository root.

## Select a source

| Source | Python | CLI |
| --- | --- | --- |
| Catalog alias | `load_pretrained("geom-explicit-h", epoch=100)` | `--model geom-explicit-h --epoch 100` |
| Hub repository | `load_pretrained("owner/repo", subfolder="bundle", revision="commit")` | `--model owner/repo --subfolder bundle --revision commit` |
| Local bundle | `load_pretrained(weights_dir="/path/to/bundle")` | `--weights-dir /path/to/bundle` |

- Bundles provide config, ordered vocabularies, distributions, scaling, validity policy,
  and checkpoint metadata. Never mix assets from different bundles/representations.
- Choose `epoch` / `--epoch` **or** `filename` / `--filename`. Filenames are relative to
  the bundle and may include subdirectories. Default: largest available completed epoch.
  An explicit filename absent from metadata gives `model.bundle.epoch = None`.
- Unregistered Hub IDs default to root subfolder and revision `main`. Branches/tags resolve
  once; all assets come from that commit, recorded in `bundle.revision`. Pin commits for reproduction.
- Local `weights_dir` can pair with `subfolder`. Omit the model name to avoid catalog lookup.
  A named alias with local weights still needs a cached/explicit catalog; it accepts the
  export root or variant subdirectory.
- `--checkpoint /path/to/custom.ckpt` selects an external file and requires `--weights-dir`;
  weights must match the bundle's architecture/vocabulary. `--filename` selects an internal file.
- Catalog revision overrides rediscover epochs; subfolder overrides also rediscover dataset
  and hydrogen mode. Without overrides, aliases retain pinned defaults.

## Discover available models

```bash
morph models list
```

| Alias | Hydrogen representation | Completed epochs | Default epoch |
| --- | --- | --- | --- |
| `qm9-explicit-h` | H graph vertices | 1000, 1500 | 1500 |
| `geom-explicit-h` | H graph vertices | 50, 100 | 100 |
| `chembl3d-explicit-h` | H graph vertices | 200, 300, 400 | 400 |
| `chembl3d-implicit-h` | Heavy atoms; RDKit infers H | 100, 200, 300 | 300 |

This table describes documented releases; `morph models list` is authoritative for
current discovery. Completed epoch 200 corresponds to Lightning index 199.
Omitting model and local weights uses the catalog's `default_model`
(currently `chembl3d-explicit-h`). Omitting epoch uses the latest published epoch.

Training filters: GEOM ≤72 atoms including H; both ChEMBL3D variants ≤100 atoms
**before removing H**. These filters do not cap generation. ChEMBL3D variants retain
matching molecules but separate vocabularies/statistics; implicit-H bonds and charges
remain explicit. GEOM epochs 50/100 share the resume history in [reproducibility](reproducibility.md).

### Catalog and offline discovery

Discovery fetches only [catalog.json](https://huggingface.co/maltefranke/morph/blob/main/catalog.json),
without Torch or weights. No built-in fallback list exists; offline aliases need a prior fetch.

```bash
morph models list --cache-dir /path/to/shared-cache
morph models list --cache-dir /path/to/shared-cache --offline
morph models list --catalog-repo your-org/morph-models
morph models list --catalog-file /path/to/catalog.json
```

Catalog and cache options apply to `models list`, `models download`, and `generate`:

| Option | Meaning |
| --- | --- |
| `--catalog-repo`, `--catalog-revision` | Catalog source; defaults `maltefranke/morph`, `main`. Environment defaults: `MORPH_CATALOG_REPO`, `MORPH_CATALOG_REVISION`; explicit flags win. |
| `--catalog-file` | Local catalog JSON; skips catalog network access. Remote catalogs use `catalog.json`. |
| `--revision` (download/generate only) | Model bundle revision, independent of catalog revision. |
| `--cache-dir`, `--offline` | Shared Hugging Face cache; offline applies to both discovery and bundles. Reuse the same cache across commands. |

`morph models list` prints the resolved catalog commit. Direct repository IDs, explicit `ModelSpec`,
and unnamed local bundles bypass the catalog.

```python
from morph.inference import load_catalog, load_pretrained

catalog = load_catalog()  # No model/weights loaded
print(catalog.models.keys(), catalog.default_model, catalog.revision)
model = load_pretrained("chembl3d-implicit-h", catalog=catalog, device="cuda")
local_catalog = load_catalog(path="/path/to/catalog.json")
cached_catalog = load_catalog(local_files_only=True)
```

## Download model weights

```bash
morph models download chembl3d-implicit-h --epoch 300
```

The published Morph models are public and require no login. For a private or gated
repository, use `hf auth login` or set `HF_TOKEN` to an account with access;
never commit tokens.
Downloads read metadata, then fetch one checkpoint (about 510 MB for ChEMBL3D) and
`config.yaml`, `metadata.json`, three token files, and `distributions.pt`.
Other epochs/archived runtimes are excluded. No Torch import or model construction is needed.

Files use the standard Hugging Face cache (`HF_HOME` or `--cache-dir`). On a
cluster, choose a cache visible to compute nodes. If they lack internet access,
download weights on a network-enabled node first, then generate with `--offline`
using the same cache. Run GPU generation inside a suitable allocation; see the
[Slurm guide](slurm.md).

```bash
# Export to assets/model_ckpts/chembl3d/implicit_h/:
morph models download chembl3d-implicit-h --epoch 300 --local-dir assets/model_ckpts
# Load the same release directly, without catalog resolution:
morph models download maltefranke/morph --subfolder chembl3d/implicit_h \
  --revision ab9bea92f499f03450ebc99459fce04461504a41 --epoch 300
```

Additional epochs share support files in the same destination. ChEMBL3D aliases pin
the commit above. Python equivalents are `download_pretrained(..., local_dir=...)`
and `load_pretrained(...)`; both accept `cache_dir` and `local_files_only=True`.

## Generate molecules

```bash
morph generate --model chembl3d-implicit-h --epoch 300 \
  --device cuda --batch 64 --batch-size 16 --output molecules.sdf
# Fully local, catalog-free bundle:
morph generate --weights-dir assets/model_ckpts/chembl3d/implicit_h \
  --epoch 300 --device cuda --batch 64 --batch-size 16
```

| Option / output | Contract |
| --- | --- |
| Weights | EMA by default; `--no-ema` uses current weights. One network is built, without training losses/metrics; checkpoint is memory-mapped on CPU. |
| Sampling | Defaults: `--steps 100`, `--seed 42`, eager execution. `--device auto` chooses CUDA or CPU; `--device cpu` works on CPU workers. |
| `--batch` | Total attempts, including failures. |
| `--batch-size` | Concurrent molecules; default at most 32. Priors are lazy; completed batches write immediately. |
| `--output` | Default `eval_outputs/denovo.sdf`; valid molecules only, coordinates in Å. Terminal metrics count all attempts. Only SMILES accumulate for exact uniqueness. |
| `--offline` | Use previously fetched catalog/assets in the same cache. Direct local bundles need neither authentication nor catalog. |
| `--save-trajs` | Write `<output>_trajs/batch-000000.pt`, etc. Each contains `trajectories`, `is_valid`, zero-based `start_index`. A nonempty directory is rejected; choose a new output name. |

Load trajectory chunks with `torch.load(path, map_location="cpu", weights_only=False)`.
Chunks replace the former single `<output>_trajs.pt` file and bound memory to one batch.
Missing assets/unavailable epochs raise errors. Missing online assets download automatically.

## Load from Python

```python
from morph.inference import load_pretrained
from morph.inference.generate import write_sdf

model = load_pretrained("geom-explicit-h", epoch=100, device="cuda")
trajectories, is_valid, vocab, coordinate_std = model.generate(
    n_mols=64, batch_size=16, seed=42, return_traj=False,
)
valid = [t for t, ok in zip(trajectories, is_valid, strict=True) if ok]
write_sdf(valid, vocab, "molecules.sdf")
```

| Attribute / result | Meaning |
| --- | --- |
| `model.module.model` | Sole selected network; prediction-only Lightning module, evaluation mode, gradients disabled. |
| `model.module.weights` | `"ema"` or `"current"`; set `use_ema=False` at load time for current weights. Reload to switch; no `model_ema` copy. |
| `model.vocab`, `model.distributions` | Ordered tokens and empirical training statistics. `model.module.distributions` holds the derived sampling prior. |
| `model.config`, `model.bundle.checkpoint` | Saved config with localized asset paths; checkpoint path. |
| `trajectories[i][-1]`, `is_valid[i]` | Final graph and validity for attempt `i`; failures remain in the list. Coordinates already include scaling: **do not multiply by `coordinate_std` again**. |

Python defaults: CPU, `steps=100`, `use_ema=True`. Loading strictly checks the selected
network including buffers; missing EMA never silently falls back to current weights.
Use training entry points for fine-tuning.

### Stream large runs

`generate()` retains all results. `generate_to(on_batch, ...)` passes CPU trajectories
and validity flags to a callback in attempt order, including failures, and returns the
total attempts. It does not retain predictions; consumers must release processed batches.

```python
from itertools import count
from pathlib import Path
from morph.inference.generate import write_sdf

directory = Path("generated_batches")
batch_number = count()

def save_batch(trajectories, is_valid):
    valid = [t for t, ok in zip(trajectories, is_valid, strict=True) if ok]
    write_sdf(valid, model.vocab, directory / f"batch-{next(batch_number):06d}.sdf")

attempted = model.generate_to(save_batch, n_mols=10_000, batch_size=16, seed=42)
```

Fix seed **and batch size** for comparisons. Lazy priors use a separate CPU random
stream; identical seeds do not reproduce the earlier eager sampler's draw order.

## Add GEOM, QM9 or another model

Compatible unconditional bundles need no dataset-specific loader. Place these files at
a repository root/subfolder or in a local directory:

```text
config.yaml
metadata.json
atom_tokens.txt
edge_tokens.txt
charge_tokens.txt
distributions.pt
weights/epoch_100.ckpt
weights/epoch_200.ckpt
```

### Bundle contract

- `config.yaml`: composed inference config, dataset construction disabled; preserve training
  architecture, embeddings, integrator, scaling, and `data.allow_charged` (QM9 false, GEOM true).
  The loader rewrites asset paths and disables training-dataset fallback.
- `data.dataset_name` matches metadata case-insensitively. Architecture matches installed Morph.
  Prediction-only loading supports unconditional `LightningModuleRates`; conditional/scaffold
  models use their existing entry points.
- Token files contain one token per line in training order; distributions match preprocessing.
- `variant` is `explicit_h` or `implicit_h`, consistent with `H` in vocabulary and optional
  `includes_explicit_hydrogens`. Checkpoints need not encode epoch in their filenames.
- Legacy metadata without `checkpoints` needs an explicit filename or `epoch_NNN.ckpt` convention.

Example metadata; replace vocabularies and epochs with actual training values:

```json
{
  "dataset": "qm9",
  "variant": "explicit_h",
  "atom_tokens": ["C", "F", "H", "N", "O"],
  "edge_tokens": ["<NO_BOND>", "1", "2", "3"],
  "charge_tokens": ["0"],
  "checkpoints": [
    {"completed_epochs": 100, "filename": "weights/epoch_100.ckpt"},
    {"completed_epochs": 200, "filename": "weights/epoch_200.ckpt"}
  ]
}
```

### Optional discovery alias

Direct Hub IDs work immediately after upload. A reusable Python `ModelSpec` can also
supply `name`, `repo_id`, `subfolder`, `revision`, `dataset`, and `variant` without
catalog edits. For discoverable aliases, publish the bundle first, then update the Hub
catalog while preserving existing entries:

```json
{
  "schema_version": 1,
  "default_model": "your-model",
  "models": [
    {
      "name": "your-model",
      "repo_id": "your-org/morph-models",
      "subfolder": "geom/implicit_h",
      "revision": "<bundle-commit>",
      "dataset": "geom",
      "variant": "implicit_h",
      "epochs": [100, 200]
    }
  ]
}
```

- Require unique alias names, `repo_id`, and `revision`. `subfolder` defaults to root;
  `dataset`/`variant` are optional identity checks.
- Optional `epochs` restricts available completed epochs, defaulting to their maximum.
  Without it, listing shows "from bundle" and metadata supplies epochs.
- Optional `default_model` allows callers to omit model selection.
- Catalog and bundle revisions are independent; entries can point to different repositories.

After replacing placeholders and validating the catalog:

```bash
morph models list --catalog-file catalog.json
hf upload your-org/morph-models catalog.json catalog.json \
  --commit-message "Add model to discovery catalog"
```

The shared catalog is `maltefranke/morph/catalog.json`. Parsing/downloading lives in
[`catalog.py`](../src/morph/inference/catalog.py); `registry.py` defines specifications
and resolves aliases, without a hardcoded model table.

## Configuration and reproducibility

Inference uses installed Morph plus saved `config.yaml`, not current training defaults
or executable Hub source. Original configs, full statistics, hashes, validation reports,
and frozen runtimes remain in the release; see [training bundles](reproducibility.md).

Advanced `morph.inference.generate` APIs `build_sampling_module(overrides, checkpoint)`
and `generate_denovo(checkpoint, overrides, ...)` compose local Hydra configs and obey
`MORPH_COMPILE`. The pretrained loader starts eager; `model.module.compile()` opts in.
See Hugging Face's [cache/download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download)
and [authentication guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
