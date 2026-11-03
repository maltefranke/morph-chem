# Development and regression checks

Scope: contributor checks and extension contracts. Commands run from the repository root.
Related: [model interfaces](../src/morph/model/README.md), [training](training.md).

## Environment and checks

`uv sync --locked` installs the editable package and development tools. For intentional
dependency updates, edit `pyproject.toml`, run `uv lock`, and review both files.
The tracked `.python-version` selects Python 3.13 by default; package metadata allows
Python 3.11–3.13. Use `uv sync --locked --python 3.12` to select another allowed
interpreter explicitly.

The package accepts `torch>=2.9.0a0,<2.10`, allowing 2.9 patch releases and the
NVIDIA container's custom prerelease. `uv.lock` retains the tested Torch 2.9.0
environment. The NVIDIA [Dockerfile](../Dockerfile) preserves vendor Torch and its
matching torchvision through separate constraints, using the container's Python 3.12.

For a container with Torch outside the declared range, uv supports explicit
[dependency overrides](https://docs.astral.sh/uv/pip/compile/#overriding-dependency-versions).
Write the installed Torch version to an override file inside the container:

```bash
python -c 'import torch; print("torch==" + torch.__version__)' > /tmp/morph-torch-override.txt
```

Add `--overrides /tmp/morph-torch-override.txt` to the Dockerfile's `uv pip install`
command, keeping its base-image constraints. Constraints alone cannot widen the
package's version range. An override changes dependency resolution; validate
inference and compiled training with the chosen Torch build before relying on it.

```bash
uv sync --locked
uv run --no-sync ruff format --check src run.py scripts/preprocess.py tests/model
uv run --no-sync ruff check src run.py scripts/preprocess.py tests/model
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MORPH_COMPILE=0 WANDB_MODE=disabled \
  uv run --no-sync pytest tests/model
```

- Tests use synthetic graphs and temporary datasets; no dataset downloads or W&B login.
- CUDA tests skip without a GPU. Set `MORPH_REQUIRE_CUDA_TESTS=1` inside a GPU allocation
  to detect lost CUDA access in attention/compiler checks.
- `tests/model/_helpers.py` holds shared fixtures. `test_model_build.py` covers strict
  checkpoint loading, dataset-free construction, and both insertion heads' forward/backward passes.

## Refactor invariants

- Preserve categorical draw order, graph indices, per-graph reductions, coordinate frames,
  and zero-gradient dependencies.
- Optimizer/EMA changes require logical-update and accumulation regressions; GPU kernel
  changes require CUDA checks as well as CPU tests.
- Keep numerical lambda targets in batches: positivity determines binary actions;
  magnitudes remain available for analysis.

## Adding datasets and geometric conditions

| Interface | Contract |
| --- | --- |
| `source_identity()` | JSON-serializable identity of ordered raw records. Use `morph.dataset.core.source.molecular_source_identity` with backing file, split, processing options, and retained indices. Changes to records, atom order, or coordinates change identity; paths/timestamps only memoize checksums locally. Without this method, raw `get(i)` fields are hashed. |
| Pocket constructor | Accept `vocab`, `distributions`, `split`, `filter_by_vocab`. |
| Pocket access | Provide `get_pocket(i)`, `get_pocket_vocab()`, and a writable `pocket_vocab`; the datamodule sets the training-derived vocabulary for every split before tokenization. |
| `cache_namespace()` | Stable directory component covering source recipe and accepted ligand elements; each cache still verifies full split identity. Preserve CrossDocked's published metadata/names. |
| Storage ownership | Adapters own storage; generic datamodules must not inspect CrossDocked summaries or private storage/index attributes. |

Attach independent geometric conditions through `contexts`:

```python
ligand.contexts["pocket"] = pocket
batch = ChemBatch.from_data_list([ligand])
pocket_batch = batch.contexts["pocket"]
```

- Context coordinates use the parent's frame and units. Shared augmentation, recentering,
  output scaling, cloning, device transfer, and batching handle contexts automatically.
- Topology edits retain contexts without editing their nodes. Context trees require distinct
  objects: no cycles or shared graph objects. Direct context batches match parent count/order.
- Missing conditions collate to empty rows, preserving graph IDs during replay. Pass
  `context_names=("pocket",)` to the collator when even unconditional batches need the encoder.
- Encoders read `batch.contexts[name]`; replace legacy `pocket_data` / `batch.pocket` access.
  New contexts need no additions to coordinate-transform field lists.

Set `data.interpolator.include_prior_context=true` to expose the target-aligned
source at `mol_t.contexts["prior"]` (or `batch.contexts["prior"]`) during training.
It contains all real source atoms and their original bonds, including atoms
deleted along the path, and excludes auxiliary insertion slots. It is captured
before interpolation/noise and shares the interpolated molecule's coordinate
frame and augmentations. The default is `false`; evaluation and inference callers
can attach their reference graph through the same `contexts` interface.

## Attribution

The [third-party inventory](../src/morph/_vendor/README.md) maps vendored modules,
integrated adaptations, and protocol references to upstream revisions and license files.
Standalone copied modules live in `morph/_vendor`; integrated metric and RDKit helpers
retain their existing module paths and identify the relevant notices in their docstrings.
The notices ship in both the source archive and wheel through `project.license-files`.

SemlaFlow attribution covers both `semlaflow/util/metrics.py` adaptations (including
the valence table, sanitization check, and mean MMFF94 strain) and
`semlaflow/util/rdkit.py` adaptations in `morph/utils/rdkit_utils.py`.
Runtime metric names describe their calculations; retain upstream attribution separately.
