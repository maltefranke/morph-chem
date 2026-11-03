# Training and fine-tuning

Scope: current training presets, preprocessing, and conditional fine-tuning.
Commands run from the repository root in the installed environment.
Sources: [configs](../configs/), [run.py](../run.py).
Related: [frozen training recipes](reproducibility.md), [evaluation](evaluation.md),
[model interfaces](../src/morph/model/README.md), [Slurm](slurm.md).

## Training locally

Preprocess once on a CPU worker before launching distributed training:

```bash
python scripts/preprocess.py qm9
python scripts/preprocess.py geom
python scripts/preprocess.py chembl3d
# Multiple datasets; omit dataset arguments to prepare all three:
python scripts/preprocess.py qm9 chembl3d --workers 20

python run.py                       # GEOM-Drugs
python run.py defaults=qm9/qm9
python run.py defaults=chembl3d/chembl3d
```

Preprocessing downloads missing inputs to `data/<dataset>/raw/`, prepares all splits,
and writes training vocabularies/statistics to `processed/`. Completed artifacts are
reused. `--workers` overrides `data.preprocess_workers` (GEOM/ChEMBL3D default 10).
Distributed training fails early if shared preprocessing is missing; prepare it before
`torchrun` to avoid first-run races.

## Data options

GEOM/ChEMBL3D support implicit H with separate caches and statistics:

```bash
python scripts/preprocess.py chembl3d --hydrogen-mode implicit
python run.py defaults=chembl3d/implicit_h
```

Replace `chembl3d` with `geom` in both commands for GEOM.

The released ChEMBL3D models use a 100-atom filter before hydrogen removal.
Their portable presets select the matching `chembl3d_max100` and
`chembl3d_max100_implicit_h` caches:

```bash
python run.py defaults=chembl3d/max100 trainer.trainer.max_epochs=400
python run.py defaults=chembl3d/max100_implicit_h trainer.trainer.max_epochs=300
```

Download these prepared caches as described in the README.
The presets retain the released loader and checkpoint settings; set an epoch or
time limit when launching. Exact launch settings are in the
[training bundles](reproducibility.md).

| Setting | Contract |
| --- | --- |
| Implicit H | Preserve heavy-atom coordinates, charges, and bond orders; use `data/<dataset>_implicit_h/`. |
| `data.max_atoms` | Inclusive filter **before H removal**; `null` disables. |
| `data.allowed_atom_symbols` | Element filter before H removal; `null` accepts all. |
| `data.cache_name` | Use a separate name when changing filters. Presets choose defaults; archived recipes define published runs. |
| `--source-dir` | Optional existing source, not a download destination; must contain requested splits. |
| `CHEMBL3D_SOURCE_DIR` | Optional LoQI release containing `train_h.pt`, `val_h.pt`, `test_h.pt`, directly or under `processed/`. |
| `GEOM_SOURCE_DIR` | Optional prefiltered conformers; automatic preparation otherwise creates `raw_filtered/`. |

Example local-source preprocessing:

```bash
python scripts/preprocess.py geom --hydrogen-mode implicit \
  --source-dir /absolute/path/to/geom/raw_filtered
```

## Configuration and flow matching

`configs/default.yaml` selects `defaults=geom/geom`; presets combine dataset/embeddings
with `core` (model, smoothstep schedule, trainer, callbacks, W&B). Override groups inline,
e.g. `model=backbones/dit trainer.trainer.max_epochs=200`. Set `logging.enabled=false`
to avoid constructing a W&B logger.

| Default | Value / behavior |
| --- | --- |
| Architecture | Released explicit-H GEOM DiT: 8 layers, width 512; matching optimizer, losses, noise, schedules, embeddings, and independent head adapters. |
| Loader | 320 training molecules/GPU; validation/test batch 32; 20 training workers; prefetch 1. |
| Python defaults | GEOM parameters, including charged-molecule validity. Pass explicit alternatives; QM9 preset sets `allow_charged=false`. |
| Partial OT | Squared physical distances in Å; QM9/GEOM `c_move=1`, `c_sub=0`, `c_ins=c_del=6.125`. Move/substitute below 3.5 Å; delete+insert above 3.5 Å. |
| Integration | `integrator.integrator.exact_schedule_integration=true`: exact categorical hazard integral per step; coordinates still use Euler velocity updates. |

Original run paths, GPU counts, and durations are in [training bundles](reproducibility.md).
Choose resources through command-line overrides such as `trainer.trainer.devices=2`,
`data.datamodule.batch_size.train=64`, and `data.datamodule.num_workers.train=4`;
presets describe datasets and tasks rather than a particular GPU model.
See [Slurm](slurm.md) for a complete launch example.

### Training validation metrics

Stability measures chemistry rules, not dynamical stability or conformer geometry.

| Metric (under `val/`) | Definition / denominator |
| --- | --- |
| `validity` | Nonempty, sanitized, connected, radical-free; net charge allowed only with `data.allow_charged=true`. Denominator: all attempts. |
| `connected-sanitization-validity` | Sanitized and connected, without extra radical/charge restrictions; all attempts. |
| `valence-table-atom-stability` | Element/formal-charge valence table including implicit H; evaluable atoms. |
| `valence-table-molecule-stability` | Nonempty, all atoms pass table; all attempts, including failed builds/checks. |
| `rdkit-atom-stability` | Raw unsanitized graphs, atoms not flagged by `DetectChemistryProblems`; all generated atoms. Failed decoding/checks count known graph atoms as unstable. |
| `rdkit-molecule-stability` | Nonempty graphs with every atom stable; all attempts, including failed decoding/checks. |
| `posebusters/validity` | Every enabled cheap structural check passes; all attempts. Missing checks/unbuildable molecules fail; individual checks under `posebusters/*`. |

Connected-sanitization and valence-table keys replace `semla-validity`,
`semla-atom-stability`, and `semla-molecule-stability` without changing those calculations.
Historical logs/snapshots retain old names. Full offline PoseBusters adds flatness/energy
checks; see [evaluation populations](evaluation.md#metrics-and-populations).

## Cross-attention conditioning placement

Conditional scaffold and pocket recipes default to
`model.backbone.cross_attn_position=residual`, preserving the existing adapter
placement and checkpoint behavior. For a new fine-tuning run, opt in to applying
the adapter after each block's first LayerNorm and before native AdaLN:

```bash
python run.py defaults=geom/scaffold_decoration \
  model.module.pretrained_ckpt=/path/to/checkpoint.ckpt \
  model.backbone.cross_attn_position=post_norm1
```

The raw residual skip and native feed-forward computation are retained. Zero
adapter gates preserve the pretrained computation. The user is responsible for
providing correctly scaled conditioning tokens/features and checking the adapter
update scale; internal query/KV LayerNorm does not calibrate residual strength,
and can cancel uniform token rescaling. See the
[conditioning interface](../src/morph/model/README.md#cross-attention-placement-and-conditioning-scale)
for the exact insertion point and scaling contract.

Keep the selected placement in the saved model config when resuming. Old configs
that omit the option retain `residual`; no checkpoint parameter keys or shapes
change. Unconditional recipes are unaffected.

## Scaffold decoration

Presets: `defaults=qm9/scaffold_decoration` or `defaults=geom/scaffold_decoration`.

```bash
python run.py defaults=geom/scaffold_decoration \
  model.module.pretrained_ckpt=/path/to/checkpoint.ckpt \
  data.scaffold_kind=rgroup
```

| `data.scaffold_kind` | Retained atoms / limits |
| --- | --- |
| `murcko` | Bemis–Murcko rings and connecting linkers; symmetry-equivalent matches count once. |
| `rgroup` (default) | Largest BRICS fragment; ≤4 cuts, ≥40% heavy atoms retained, ≤10 heavy atoms per removed group, ≤32 cuts/molecule. |
| `libinvent` | LibINVENT reaction cuts with DiffDec filters. |
| `heavy` | All heavy atoms; generate H. |
| `linker` | Multiple fragments with original H; generate missing linker; ≤32 cuts/molecule. |

Training randomly selects a cached cut per visit; molecules without valid cuts are excluded.
BRICS/linker use RDKit only. New LibINVENT cuts require `uv sync --extra scaffold`.
First use downloads checksum-verified rules at pinned revisions:
[Lib-INVENT reactions](https://github.com/MolecularAI/Lib-INVENT-dataset/blob/540c1b015a091d3b79f57505168017baeefe10a5/tutorial/data/reaction.smirks)
and [DiffDec filters](https://github.com/biomed-AI/DiffDec/blob/916ae14207b2783a90336bb8509374535c5791f9/data/filter_conditions.json).
These match published caches; loading saved cuts needs neither the cutter dependency nor downloads.
Prepare resources before offline jobs:

```bash
python -c 'from morph.dataset.conditional.scaffold.resources import libinvent_resources; libinvent_resources()'
```

Cache default: `${XDG_CACHE_HOME:-~/.cache}/morph`; `MORPH_CACHE_DIR` selects a writable,
shared alternative. Later resource loads work offline.

### Scaffold caches

- First construction builds split/kind caches under `data.scaffold_groups_dir`, with file
  locks and atomic writes. Versioned `scaffold_<split>_<...>.lmdb` names include source/protocol
  identity and kind suffix (`_rgroup`, etc.; none for Murcko).
- Metadata covers source contents/count, processing options, cutter, and format version.
  Identical downloads reuse caches across paths, timestamps, and library versions.
- Source SHA-256 is verified once under a shared user-cache lock. Local file identity,
  size, modification/change times invalidate the checksum memo, not portable identity.
  New paths/downloads need a verification read (minutes for GEOM). Read-only dataset roots
  work; keep source files immutable while readers are active.
- Other datasets hash ordered raw molecular fields. Legacy unversioned LMDB/`.pt` caches
  lack verifiable identity: recompute into versioned LMDB, preserving old files.
- Published [morph-data](https://huggingface.co/datasets/maltefranke/morph-data) cuts already
  have portable metadata. Remove old cache generations only after their jobs finish.

### Coordinate, charge, and insertion policies

| Setting | Default / effect |
| --- | --- |
| Scaffold identity | Retained atoms and internal bonds fixed. |
| `model.module.freeze_keep_positions=false` | Coordinates may move. |
| `model.module.exclude_scaffold_from_x_loss=false` | Kept coordinates receive supervision. Set both position flags to `true` for frozen coordinates. |
| `data.jitter_scale=0.1` | Training-only independent Gaussian jitter, 0.1 Å standard deviation per axis; normalized by `coordinate_std`. No validation/inference jitter. |
| Inserted-atom noise | Usual `(1-t)` bridge, no extra persistent jitter floor. Unconditional move-noise policy unchanged. |
| `model.module.freeze_keep_charges=true` | Copy input charges at each step; exclude kept atoms from charge loss. Current cutters copy target charges. |
| `freeze_keep_charges=false` | Supervise kept charges even with `exclude_scaffold_from_sub_loss=true`; deletion targets remain excluded. Charge changes can lower exact retention without violating this policy. |
| Sampling steps | 100 by default; start from supplied scaffold coordinates. |

GEOM/in-pocket `data.insertion_assignment=substituent` restricts each future atom's
nearest parent to its target substituent branch or that branch's scaffold attachments.
Branches are target bond-graph components after removing the scaffold; explicit H and
all attachment atoms participate. Unattached components keep global nearest assignment.
Default `nearest` permits every existing atom. This changes insertion-rate/head/bond
supervision only: flow states and scaffold correspondences stay unchanged; inference
needs no target bonds/branch labels. Unconditional pairs are unchanged. This differs
from per-site OT between fully decorated molecules.

These defaults apply to new training. Old checkpoints retain old supervision/noise;
inference flags cannot teach an unsupervised coordinate/charge head. Older snapshots
excluded kept-charge supervision through the substitution mask but resampled those
charges at inference. Reproduce historical runs with saved source/configuration.

### Evaluation panels and checkpointing

Prepare caches/panels ahead of distributed training using matching root/kind:

```bash
python scripts/preprocess_scaffold.py defaults=geom/scaffold_decoration \
  data.scaffold_kind=murcko
```

- Default panel: **512 scaffolds × 8 generations = 4096 attempts**, independently selected
  from validation/test. Train, validation, and test preprocessing runs by default;
  repeat `--split train`, `--split val`, or `--split test` to restrict it.
- Group conformers by canonical molecule SMILES (stereochemistry retained); select scaffold
  topologies (stereochemistry ignored) by Morgan-fingerprint MaxMin. Cache one seeded random
  conformer per molecule; all eight attempts share geometry, fixed across epochs/model variants.
- Panels live under `data.eval_panel.directory`, default `${data.scaffold_groups_dir}/eval_panels`.
  Missing panels build automatically; incompatible metadata errors. `--force` rebuilds panels.
- Set `data.eval_panel.n_scaffolds`, `data.eval_panel.samples_per_scaffold`, and
  `data.eval_panel.seed`; update `trainer.trainer.num_val_samples` to their product.
  Fewer distinct scaffolds require a smaller panel; no duplicate padding.
- To use all cached cuts: `data.eval_panel=null data.enumerate_slicings=true`.
  Trainer evaluation limits still apply.
- Both scaffold presets validate each epoch; periodic and best-by-validation saving only
  on fifth epochs. Intermediate evaluations remain logged; monitor is pooled `val/validity`.
- Use `defaults=geom/scaffold_decoration` and set batch size, gradient accumulation,
  and worker count for your allocation. Effective global batch is the per-GPU batch
  times the GPU count times accumulation; profile throughput and peak memory when
  changing these settings.

### Scaffold metrics

Under `val/scaffold/`, collect all DDP ranks, deduplicate sampler-padding attempt IDs,
and average equally across scaffolds:

| Metric | Meaning |
| --- | --- |
| `validity`, `retention`, `valid_retention` | Chemistry validity, exact atom/charge/bond preservation, and intersection. |
| `rmsd`, `rmsd_aligned` | Kept-atom RMSD in Å, raw versus rotation/translation aligned; undefined if kept atoms are missing. Raw RMSD includes frame drift. |
| `added_atoms`, `added_heavy_atoms`, `no_growth`, `no_heavy_atom_growth` | Growth diagnostics. |
| `uniqueness`, `diversity` | Valid, preserving outputs only; Morgan Tanimoto-distance diversity. Failed scaffolds score zero; diversity zero with <2 eligible outputs. |
| `unique_yield` | Distinct valid, preserving outputs / all attempts. |
| `recovery` | Scaffolds with any canonical graph match to selected reference at the eight-sample budget. |
| Counts | Scaffolds, attempts, minimum/maximum attempts per scaffold. |

Invalid attempts remain in validity/yield denominators. Additional pooled
`val/scaffold/posebusters/validity` exempts an inherited clash only if both atoms are
kept, the **input scaffold** violates the generated graph's same bound, and generated
distance has not decreased by >0.0001 Å. Never consult hidden target coordinates.
New/worsened or scaffold–decoration clashes still fail; other checks stay strict.
Unbuildable molecules, unavailable checks, and failed detailed analysis receive no exemption.

Both strict `val/posebusters/validity` and scaffold-aware scores use all attempts.
The scaffold-aware namespace also records `n_all_pass`, `internal_steric_clash_over_generated`,
`n_clash_exemptions`, `n_inherited_clash_pairs`, `n_clash_analysis_failed`.
An exemption rescues the clash check, not necessarily overall validity.

Reevaluate with matching architecture/config and these overrides:
`trainer.do_train=false trainer.do_validate=true trainer.do_predict=false
trainer.checkpoint_path=/path/to/finetuned.ckpt`.

## LogP-conditioned generation

`defaults=geom/logp` and `defaults=qm9/logp` fine-tune the released explicit-H
models for generation at a requested **absolute RDKit Crippen logP**. The
pretrained backbone and output heads are fully trainable; LoRA is disabled.
A small MLP feeds the shared graph-conditioning port through its zero-initialized
gate, preserving unconditional predictions at initialization.

Prepare molecular data as above, then build property arrays before launching
distributed training. The builder reads the processed records in order,
memoizes scalar descriptors across conformers, and writes finite float32 labels
under `processed/properties/logp/<split>.npy`. Each array has source-identity and
RDKit-version metadata; rerunning preparation reuses matching caches and
rebuilds stale ones. Invalid SMILES fail preparation instead of receiving a
fabricated property value.

```bash
python scripts/preprocess_properties.py defaults=geom/logp
python scripts/preprocess_properties.py defaults=qm9/logp
# Optional: prepare only validation, or use a different processed-data root.
python scripts/preprocess_properties.py defaults=geom/logp \
  data.root_path=/path/to/data --split val

python run.py defaults=geom/logp \
  model.module.pretrained_ckpt=/path/to/geom/explicit_h/epoch_100.ckpt
python run.py defaults=qm9/logp \
  model.module.pretrained_ckpt=/path/to/qm9/explicit_h/epoch_1500.ckpt
```

Use the checkpoint's matching vocabulary, embeddings, and processed dataset.
The published checkpoints are available through `morph models download` or
`hf download maltefranke/morph` as described in the README. There are no required
personal filesystem paths in these presets; `MORPH_PRETRAINED_CKPT` can supply
the checkpoint instead of the command-line override.

During sampling, attach `target_logp` as a system property of the prior with one
float per graph; the request stays fixed while atoms are inserted or deleted.
Validation logs `val/logp_mae` and `val/logp_rmse` over scorable generated
molecules, plus `val/logp_scored_fraction` over **all** attempts. With no scorable
outputs, the errors are NaN and coverage is zero. Errors and counts are summed
across GPUs before reduction, so ranks with different valid-molecule counts
receive the correct weights. Chemistry and geometry metrics remain available.
At each validation epoch, W&B also receives `val/logp_parity`: requested logP on
the x axis, generated logP on the y axis, and a `y = x` reference line. The plot
gathers matched pairs from every GPU and reports scorable coverage in its title.

Defaults are 50 GEOM epochs and 500 QM9 epochs with validation during fitting.
Separate post-fit validation/prediction is disabled, and `final.ckpt` is saved
when fitting ends. See [Slurm](slurm.md#logp-fine-tuning) for GPU job submission.

## Conditional fine-tuning from a pretrained DiT

Scaffold/pocket models load pretrained body, embeddings, and heads into a clean DiT
inside `ConditionalDiTBackbone`; task encoders and zero-initialized `tanh` cross-attention
gates preserve baseline predictions initially. `freeze_pretrained_backbone=true`
freezes the clean lane; conditioning remains trainable.

| Task | Preset | Signal |
| --- | --- | --- |
| Scaffold | `qm9/scaffold_decoration`, `geom/scaffold_decoration` | Scaffold cross-attention + node/edge conditions. |
| Final count | `geom/atom_count` | Signed target-minus-current count. |

```bash
python run.py defaults=qm9/scaffold_decoration \
  model.module.pretrained_ckpt=/path/to/uncond_dit.ckpt \
  model.module.pretrained_prefer_ema=true

python run.py defaults=qm9/qm9 model=finetune/lora_finetune \
  model.module.pretrained_ckpt=/path/to/uncond_dit.ckpt \
  model.lora.rank=8 model.lora.alpha=8
```

LoRA loads then freezes base weights, adding residuals to `qkv`, `proj`, `mlp.gate_value`,
`mlp.out_proj`; restrict via `model.lora.targets`. Dropout defaults to zero. Adapters use
AdamW, are saved in ordinary checkpoints, and can be folded with
`model.backbone.merge_lora_()` without changing subsequently saved checkpoint meaning.
For conditional LoRA, set `model.lora.enabled=true`; adapters remain trainable with frozen base.

Recipes select encoder, freezing, and ports `model.backbone.use_cross_attn`,
`use_graph_cond`, `use_node_cond`, `use_edge_cond`. Base keys retarget automatically
from `backbone.*` to `backbone.dit.*`. See [port contracts](../src/morph/model/README.md).

### Final atom-count conditioning

`defaults=geom/atom_count` uses `delta = target_n_atoms - current_n_atoms`, including H:
15 → target 20 gives +5, then +4 after insertion. Training/validation use full clean
molecule count, including pending insertions. Source sizes use the uniform GEOM prior.
Carry the target as a graph property; recompute delta every model call after topology edits.

```bash
export MORPH_PRETRAINED_CKPT=/path/to/geom/explicit_h/epoch_100.ckpt
python run.py defaults=geom/atom_count \
  data.preprocessing.root=/path/to/geom \
  trainer.trainer.devices=2 +trainer.trainer.max_time.hours=23
```

Use matching processed data/vocabulary. All variants start from pretrained EMA with
fresh optimizer state and cosine warmup; validate 4096 attempts every epoch, save every
five epochs, and write `final.ckpt` after fitting.

| Model override | Conditioning / trainable parameters | Peak Muon / AdamW LR |
| --- | --- | --- |
| Default `finetune/atom_count` | MLP on `delta/64`, zero final projection into time/current-count conditioning; rank-32 LoRA (alpha 32), all heads including coordinates/insertion, and delta MLP train. Other pretrained parameters frozen. | 0.001 / 0.0001 |
| `finetune/atom_count_fourier` | 256 Fourier features, periods 512–4096, plus `delta/128`, projected to conditioning width; zero scalar gate after pretrained conditioning projection. Encoder/gate + same LoRA/heads train. | 0.001 / 0.0001 |
| `finetune/atom_count_fourier_full` | LoRA disabled; train backbone, heads, and learned current-count/delta projections. Fourier frequencies remain fixed buffers. Fresh fine-tune, not a LoRA resume. | 0.0003 / 0.00003 |

Fourier variants pass `graph_cond` with `use_graph_cond=true`; `ConditionalDiTBackbone`
owns the gate. Earlier Fourier keys map to `backbone.dit.*` / `backbone.graph_gate`.
Choose architecture before training: an MLP checkpoint cannot convert via a config override.
`val/atom_count_mae` and `val/atom_count_exact` include chemically invalid graphs;
conditioning learns count control without forcing exact size.

For custom sampling, attach one target per graph before batching:

```python
source.add_system_property("target_n_atoms", torch.tensor([20]))
```

Store the target, never a fixed delta that would become stale during integration.
