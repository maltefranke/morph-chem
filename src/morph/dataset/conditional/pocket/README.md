# CrossDocked pocket data

This package supplies ligand targets and protein-pocket context, with a
fine-tuning pipeline that mixes pocket examples and unconditional pretraining
replay.

For the DiffDec scaffold-in-pocket evaluation functions and standalone sampling
script, see [evaluation instructions](../../../../../docs/evaluation.md).

## Prepare data

Use the pre-extracted CrossDocked pocket10 release and its official
`split_by_name.pt`. With no local source, the builder downloads both files from
the upstream Google Drive release:

```bash
bash setup_env.sh --extra crossdocked
uv run --no-sync python -m morph.dataset.conditional.pocket.build \
  --root data/crossdocked --workers 10
```

The CrossDocked setup also installs Open Babel, QuickVina 2, and AutoDockTools
for docking evaluation on Linux x86_64, with the legacy runtime isolated under
`.venv/crossdocked/`. For data preparation alone, `uv sync --locked --extra
crossdocked` installs the Python dependencies without the native docking tools.
See the [docking setup details](../../../../../docs/evaluation.md#crossdocked-docking-requirements-and-reproduction-limits).

Reuse previously downloaded source files without installing the CrossDocked
extra:

```bash
python -m morph.dataset.conditional.pocket.build \
  --root data/crossdocked \
  --source-dir /path/to/crossdocked \
  --workers 10
```

`--source-dir` accepts the parent of `crossdocked_pocket10/`, the extracted tree
itself, or the older layout with the tree beside `raw/split_by_name.pt`.
An archive can replace the extracted tree. `MORPH_CROSSDOCKED_RAW` provides the
same source override. Supplied sources are read-only; archive extraction goes
into the destination's `raw/` cache. With neither override, missing sources are
downloaded into that cache.

Run preprocessing once on a CPU worker before distributed training. Workers
default to zero (serial); `--workers 10` enables a spawn-based process pool.
By default this command builds only the paired raw LMDBs. Fine-tuning reuses
the pretrained checkpoint's vocabulary and coordinate normalization. For training
from scratch, `--compute-statistics` also discovers a vocabulary and statistics
from CrossDocked training ligands. The existing `scripts/preprocess.py` still
handles the unconditional datasets.

Outputs under `root/processed/`:

- `train_data.lmdb`, `val_data.lmdb`, `test_data.lmdb`: paired raw records.
- `{split}_smiles.txt`, `{split}_ligand_fns.txt`: ordered identifiers, retaining
  duplicates for novelty and ligand-frequency analysis.
- `crossdocked_summary.json`: processing settings, upstream revision, source
  split checksum, held-out pocket directories, counts, inventories, and rejected
  pairs with reasons and their original split indices.
- `atom_tokens.txt`, `edge_tokens.txt`, `charge_tokens.txt`, `distributions.pt`:
  only with `--compute-statistics`; artifacts learned from training ligands.

Completed caches can be reused without source files. All splits are published
together after a successful build; failed builds leave no partial processed
cache. Concurrent builders share an OS-managed lock. Changing hydrogen mode,
element/size filters, validation fraction, or seed requires a separate root.

## Processing conventions

The default is the updated DiffDec ligand path: read/sanitize the SDF, remove
**all** hydrogens (including isotopic H), then encode Kekulé bonds and formal
charges. Heavy-atom order and coordinates are preserved. GEOM's additional
radical/fragment validity screen is not applied, and there is no default ligand
size cutoff. `--max-atoms` enables an explicit cutoff. `--hydrogen-mode explicit`
materializes hydrogens with RDKit; use another root for that mode.

There is no default element whitelist and no dependency on GEOM's vocabulary.
Raw records retain chemical identities independently of any model's token IDs.
Python callers can opt into a ligand-only filter with
`allowed_ligand_atom_symbols=checkpoint_vocab.atom_tokens`. This filter never
restricts protein-pocket elements. Rejections are recorded in the manifest.
Caches from the earlier implementation with its implicit GEOM filter must be
rebuilt under a new root; they are rejected by the cache version check.

Pocket heavy atoms come from PDB `ATOM` and `HETATM` records and the element
column. Residue names, chain/residue/insertion-code grouping, and backbone flags
are retained. Hydrogens are removed; missing element labels are skipped. The
stored pocket is the full supplied pocket10 structure.

Both graphs are translated by the **same ligand centroid**, stored as
`lig_center` in Angstroms. Raw fields remain unscaled; tokenized reads divide
both sides by the supplied `distributions.coordinate_std` (the pretrained scale
when fine-tuning). Reading a pocket never
re-centers it. The encoder-facing pocket defaults to the 512 atoms nearest the
ligand centroid, preserving source order; `max_pocket_atoms=None` disables this.
Residue features remain available in the uncapped raw read.

Validation is carved from official training pairs by whole pocket directory,
using a deterministic split (fraction 0.02, seed 42). Official
test order is retained among accepted records. `source_index` identifies the
original position in that split even when earlier records were rejected.

## Reuse a ChEMBL3D no-H checkpoint

Prepare and validate every retained pair before fine-tuning:

```bash
python scripts/preprocess_pocket.py \
  --root data/crossdocked \
  --pretrained-assets data/chembl3d_max100_implicit_h/processed --workers 15
```

The script builds all three raw splits, applies the training reader's ligand
element filter, and validates tokenized ligands and protein context. It reuses
the checkpoint's statistics and discovers protein tokens from training pockets.
`processed/finetune_preprocessing_summary.json` records artifact checksums,
retained/dropped counts, unsupported elements, and validation status. Unsupported
bond/charge labels or protein tokens fail validation with the source ligand path.
The raw LMDB remains reusable; training applies the same vocabulary filter.
As with the builder, `--source-dir` reuses existing source files; otherwise
missing sources are downloaded. `MORPH_CROSSDOCKED_RAW` and
`MORPH_PRETRAINED_ASSETS` set the defaults of `--source-dir` and
`--pretrained-assets`.

Keep `hydrogen_mode="implicit"` and load the **exact artifacts used by the
checkpoint**. Do not rediscover CrossDocked token IDs or substitute its coordinate
standard deviation: even equal vocabulary sizes can encode different classes.
Ligand atom, bond, and charge token ordering all come from these artifacts.
Supplying `vocab` automatically checks the split's ligand element inventory.
Molecules containing any unsupported atom type are excluded from the reader's
view, including their paired pockets. `dataset.vocab_filter_report` and the setup
log report missing elements and retained/dropped counts. All accessors and
sidecars follow the same filtered indices; metadata retains the original source
index. The raw LMDB is unchanged and can be reused with another checkpoint.
Raw readers without a vocabulary retain all processed pairs. For inspection,
`filter_by_vocab=False` disables this view filter; the training datamodule always
enables it.

```python
import json
from pathlib import Path

from morph.dataset.conditional.pocket import CrossDockedPocketDataset
from morph.dataset.core.preprocessing import Preprocessing
from morph.dataset.core.vocab import Vocab

root = "data/crossdocked"
# Local checkpoint bundle, or the preprocessing directory that trained it.
pre = Preprocessing(root="/path/to/chembl3d-no-h-checkpoint-assets")

# The new protein encoder can have its own vocabulary, discovered from training
# pockets. This does not extend or reorder the pretrained ligand vocabulary.
summary = json.loads(Path(root, "processed", "crossdocked_summary.json").read_text())
pocket_vocab = Vocab(
    atom_tokens=sorted(summary["splits"]["train"]["pocket_elements"]),
    edge_tokens=["<NO_BOND>"],
    charge_tokens=["0"],  # Placeholder labels; protein charges are not inferred.
)
dataset = CrossDockedPocketDataset(
    root,
    split="test",
    hydrogen_mode="implicit",
    vocab=pre.vocab,
    distributions=pre.distributions,
    pocket_vocab=pocket_vocab,
)
ligand = dataset[0]  # ChemAtomicData; normalized coordinates
pocket = dataset.get_pocket(0)  # Same frame, empty bonds, neutral charge labels
raw_ligand = dataset.get(0)  # Raw ligand tensors for statistics/scaffolds
raw_pocket = dataset.get_raw_pocket(0)  # Full pocket and residue metadata
metadata = dataset.get_metadata(0)

# Recover source coordinates from a tokenized ligand:
original_positions = ligand.positions * pre.distributions.coordinate_std
original_positions += metadata["lig_center"]
dataset.close()
```

Raw readers need no vocabulary. Tokenized readers use the supplied ligand
vocabulary, including its actual bond/charge token ordering. If a model is
already loaded with `load_pretrained`, pass its `.vocab` and `.distributions`
directly. For training from scratch, use the CrossDocked artifacts produced by
`--compute-statistics` instead.

`get_pocket()` requires an explicit `pocket_vocab`; pass the ligand vocabulary
there too only if the protein encoder deliberately shares those categories.
Alternatively, use `get_raw_pocket()` for a residue-based or other protein
encoder. Metals in a pocket do not require new ligand output classes. Unsupported
ligand bond/charge labels or pocket elements raise errors during tokenization;
vocabularies are never silently expanded or remapped. Element filtering uses only
ligand atoms and applies consistently to train, validation, and test readers.

All returned tensors/metadata can be modified without corrupting the cached
record. Readers support repeated dataset instances and fork/spawn DataLoader
workers.

## Fine-tune with unconditional replay

The `crossdocked/finetune` preset loads the pretrained ligand vocabulary and
coordinate scale, filters the pocket dataset, adds a protein encoder and gated
cross-attention to the DiT, and trains on a configurable source mixture:

```bash
export MORPH_PRETRAINED_CKPT=/path/to/pretrained.ckpt
export MORPH_PRETRAINED_ASSETS=/path/to/pretrained/preprocessing/artifacts
python run.py defaults=crossdocked/finetune \
  data.pretrain_root=data/chembl3d_max100_implicit_h \
  data.pretrain_max_atoms=100 \
  data.finetune_fraction=0.3 \
  data.train_epoch_size=100000
```

The example draws 30% pocket fine-tuning samples and 70% unconditional ChEMBL3D
samples **in expectation**, independently per example, with replacement within
each source. Set `data.finetune_fraction=p/100`; 0 and 1 select pure replay and
pure fine-tuning. The percentage is independent of corpus sizes and is not a
fixed quota in every batch. All augmentations of a draw share its source.
`train_epoch_size` counts source draws globally before augmentation; by default
it is the retained fine-tuning size (or pretraining size at fraction 0). DDP
rounds this count up to divide among ranks; incomplete training batches are
dropped. Seeds and epoch numbers define reproducible source/index draws. Keep
`trainer.trainer.use_distributed_sampler=false`: the replay sampler already
partitions draws across ranks. Evaluation is sharded separately.

Replay uses only the **training split** of the pretraining corpus. Point
`pretrain_root`, `pretrain_max_atoms`, `pretrain_allowed_atom_symbols`, and
`hydrogen_mode` at the exact corpus variant used by the checkpoint. The preset
defaults to the unfiltered ChEMBL3D implicit-H variant; the example selects the
100-atom variant. Set `data.datamodule.pretrain_dataset._target_` and its options
to use another molecular source. Backbone/head architecture settings must match
the pretrained run; checkpoint loading retains the usual shape checks.

Pretraining examples carry empty protein-context rows and remain unconditional
even after the cross-attention gates train. Pocket examples carry the protein
through the ligand's interpolation translation, rotation augmentation, batching,
and sampling. Validation and test use only the filtered pocket task, with no
replay. The discovered TRAIN protein vocabulary is saved in the run config.
The pretrained backbone and output heads train by default so replay can constrain
their updates; freezing remains configurable in the fine-tuning module.

## Scaffold decoration inside a pocket

`crossdocked/scaffold_decoration` combines scaffold and pocket conditioning.
A retained ligand scaffold provides node and bond conditioning;
protein atoms provide cross-attention context. Ligand, scaffold, and pocket
share translations and rotation augmentations. Evaluation can enumerate every
distinct ligand/scaffold pair, with the matching pocket attached to each pair.

The default cutter is the **LibINVENT** reaction enumerator. The first
use downloads reaction SMIRKS and DiffDec filters from pinned upstream revisions,
verifies their SHA-256 hashes, and caches them for offline reuse under
`${XDG_CACHE_HOME:-~/.cache}/morph` (override with `MORPH_CACHE_DIR`). Reading
published scaffold caches does not fetch these rules. Install the optional
dependency and prepare new cuts on a network-enabled CPU node:

```bash
uv sync --locked --extra scaffold --extra crossdocked
python scripts/preprocess_pocket.py --root data/crossdocked \
  --pretrained-assets /path/to/pretrained/processed --scaffold-kind libinvent
```

The script reuses the pocket LMDBs and writes a separate
`processed/scaffold_libinvent_finetune_preprocessing_summary.json`. Scaffold
caches live under `data/crossdocked/scaffold_groups/`, keyed by the raw-data
manifest and allowed ligand elements. This prevents equal-length filtered
datasets from accidentally sharing cuts for different molecules. Molecules
with no accepted cut are excluded from the scaffold task. Reaction enumeration
is memoized across repeated ligand SMILES; scaffold-cache construction runs
serially after the parallel raw-data build.

```bash
export MORPH_PRETRAINED_CKPT=/path/to/chembl3d-no-h.ckpt
export MORPH_PRETRAINED_ASSETS=/path/to/its/preprocessing/artifacts
python run.py defaults=crossdocked/scaffold_decoration \
  data.pretrain_root=data/chembl3d_max100_implicit_h \
  data.pretrain_max_atoms=100 \
  data.finetune_fraction=0.3
```

This uses the same checkpoint vocabulary, normalization, and source-probability
replay as the de-novo pocket task. Unconditional replay carries empty pockets
and zero keep masks, including in batches containing only replay examples.
The scaffold's identities, internal bonds, and formal charges remain fixed.
Scaffold coordinates are movable and supervised, with 0.1 Å training jitter and
no inference jitter. The pretrained body is frozen; rank-8 LoRA, conditioning
modules, and complete prediction heads (including the coordinate readout) train.
Peak rates are Muon `1e-3` and auxiliary Adam `1e-4`.
The earlier rigid-scaffold experiment can be reproduced by setting
`model.module.freeze_keep_positions=true`,
`model.module.exclude_scaffold_from_x_loss=true`, and `data.jitter_scale=0.0`.
Choose `data.scaffold_kind=murcko`, `rgroup`, or `linker` for other cuts, and
preprocess with the matching `--scaffold-kind`.
Prediction through the task datamodule and `module.sample` uses the shared
scaffold constraints and restores frozen poses to their input coordinate frame.

### Scaffold evaluation

The default evaluation panel selects up to 512 fixed scaffold/pocket pairs with
seed 42 and draws eight generations per pair. Selection is over enumerated cuts;
identical scaffold topologies in different pocket contexts remain separate
conditions. Smaller splits use every available pair. The two-GPU preset evaluates
the complete panel rather than truncating its repeats. Training and replay
sampling are unchanged.

Stable pair and attempt IDs activate `val/scaffold/*`: retention, valid retention,
unaligned/aligned RMSD, growth, recovery, uniqueness, unique yield, and diversity.
These are averaged over evaluated pairs; they are not DiffDec's per-pocket
benchmark definitions. `val/scaffold/posebusters/*` additionally reports the
existing inherited-clash adjustment alongside the unchanged strict
`val/posebusters/*` scores. The adjustment exempts only verified, non-worsened
clashes between retained atoms; bond-length and bond-angle failures still count.

Routine PoseBusters here checks ligand geometry only, without protein clashes,
energy/strain, or flatness. Full-receptor docking evaluation remains separate.

## Receptors for docking

Training uses the cropped pocket10 structures, whereas benchmark docking needs
the corresponding **full receptors** from `test_set.zip`, prepared with the
selected protocol. This builder neither hydrogenates receptor PDBs nor produces
receptor PDBQT files. Metadata preserves the source paths and the coordinate
translation needed to place generated ligands back in the receptor frame. See
[DiffDec evaluation](../../../../../docs/evaluation.md#scaffold-decoration-in-a-pocket-diffdec-evaluation)
for docking and its reproduction limits.
