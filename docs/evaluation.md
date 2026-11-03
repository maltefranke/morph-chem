# Molecular evaluation

Scope: offline de novo scoring and DiffDec pocket benchmarks.
Commands run from the repository root; sampling requires a compute allocation.
Related: [generation](inference.md), [training validation](training.md#training-validation-metrics).

## Reporting contract

Publish **all metrics in one `summary.csv`**: individual seed values, mean, sample
standard deviation (`ddof=1`), and contributing seed count. Include novelty, combined
rates, energies, and individual PoseBusters checks; add metrics to this table instead
of separate subset CSVs. Link the complete CSV when reporting results. JSON and
per-molecule records supplement it for provenance.

## De novo generation from SDF files

[`scripts/evaluate_denovo.py`](../scripts/evaluate_denovo.py) scores existing SDFs on CPU;
it does not sample or load checkpoints. Implementation:
[`morph.utils.metrics.denovo`](../src/morph/utils/metrics/denovo.py).

```bash
uv run --no-sync python scripts/evaluate_denovo.py \
  --sdf /path/to/generated.sdf \
  --n-attempts 10000 \
  --reference-smiles /path/to/qm9/processed/train_smiles.txt \
  --workers 4 \
  --output-json eval_outputs/denovo.json
```

### Inputs and failure accounting

- `--n-attempts` is the actual total generations: omitted SDF records count as failures.
  Use only when omissions are known failures (e.g. Morph's exporter), not arbitrary subsets.
  Without it, rates cover SDF records only and the report states that limitation.
- Multiple `--sdf` files pool records without deduplication; attempts then refer to the pooled
  total. Malformed/unsanitizable records remain failures. Streaming reading avoids RDKit's
  indexed-reader offset bug at 64 KiB boundaries.
- RDKit stability uses an independent **unsanitized** graph copy, including sanitization
  failures. If export omitted graphs, supply `--stability-sdf raw_attempts.sdf`, one raw
  record per attempt. Otherwise missing atom counts make atom stability `null`, with
  missing-graph counts. Python: `rdkit_stability_mols`; decode saved graph chunks with
  `sanitize=False` to recover raw graphs.

### Metrics and populations

Fractions use the denominators below; energies are kcal/mol and RMSD is Å.

| Metric | Definition / denominator |
| --- | --- |
| `validity` | Connected, sanitizable, radical-free, zero net charge / all attempts. `--allow-charged` allows nonzero net charge. |
| `connected-sanitization-validity` | Connected and sanitizable / all attempts; no additional radical/charge restriction. |
| `valence-table-atom-stability` | Element/charge valence table including implicit H / evaluable atoms. |
| `valence-table-molecule-stability` | Nonempty, every atom passes table / all attempts. |
| `rdkit-atom-stability` | Raw-graph atoms not flagged for chemistry problems, including valence/multi-atom kekulization / all generated atoms. Missing graphs make rate undefined. |
| `rdkit-molecule-stability` | Nonempty raw graphs with all atoms stable / all attempts; missing graphs fail. |
| `uniqueness` | Distinct canonical H-stripped SMILES / convertible SMILES, duplicates in denominator. |
| `novelty` | Convertible SMILES absent from training reference / convertible SMILES, duplicates retained. |
| `novelty-among-valid` | Novel valid molecules / valid SMILES-convertible molecules, duplicates retained. |
| `valid-unique` | Distinct valid identities / all attempts. |
| `valid-novel` | Valid novel attempts / all attempts, duplicates retained. |
| `valid-unique-novel` | Distinct valid novel identities / all attempts. |
| `valid-posebusters` | Strictly valid attempts passing all enabled PoseBusters checks / all attempts. |
| `valid-unique-novel-posebusters` | Distinct novel identities with at least one strictly valid pose passing all PoseBusters checks / all attempts. |
| `diversity` | Mean pairwise ECFP6 Tanimoto distance (radius 3, 2048 bits), duplicates retained. |
| `energy-validity`, `opt-energy-validity` | Finite MMFF94 energy before/after optimization / all attempts. |
| `energy`, `opt-energy` | Mean finite MMFF94 energy before/after optimization. |
| `strain` | Mean input minus optimized energy, only pairs with both finite. |
| `energy-per-atom`, `strain-per-atom` | Per-molecule energy/strain divided by atom count, then averaged (kcal/mol/atom). |
| `opt-rmsd` | Atom-order-preserving, rotation/translation-aligned RMSD to optimized pose. |
| `posebusters/*` | Full installed `mol` configuration, including flatness/internal energy; fractions and counts. |

Population rules:

- Chemistry/energy inputs are not prefiltered by strict validity. SDF loading sanitizes
  by default; `--no-sanitize` retains parseable invalid graphs. Coordinates are not re-embedded.
- Missing conformers, unsupported force-field chemistry, and nonfinite energies are excluded
  from energy means and exposed by counts. Default `--max-iters 1000`; finite results at
  the iteration limit remain, with converged/nonconverged/failed counts separate.
- RDKit stability does not detect clashes, disconnectedness, or radicals. High atom stability
  can coexist with low molecular validity. Valence-table metrics use supplied molecules,
  not omitted raw graphs. Training validation counts known atoms in failed decodes as unstable.
- Combined rates intersect molecule-level flags, never multiply aggregate rates. Deduplicate
  after filtering; any qualifying pose can make a duplicate identity qualify.

### PoseBusters execution

- Per-check means use evaluated subsets; each also reports evaluated/missing counts and
  rate over all attempts. Joint `posebusters/validity` needs every enabled check; missing
  checks fail. Training uses a cheaper configuration without flatness/energy.
- Reports save actual configuration/package versions. RDKit numeric `GetPositions` replaces
  PoseBusters 0.6.5's crashing Point3D-to-NumPy conversion; flatness thresholds/math stay unchanged.
- Execution errors abort scoring. Energy checks use isolated subprocesses with
  `--posebusters-energy-timeout` default 180 seconds/molecule. Timeouts leave energy missing,
  fail joint validity, retain other checks/attempt denominators, and record names/counts.
- Energy thresholds and 50-conformer ensemble stay unchanged; omit the duplicate unreported
  energy module in PoseBusters 0.6.5 so failed ensembles are not computed twice.
- `--checkpoint-dir /path/to/checkpoints` saves completed energy/PoseBusters chunks for resume.
  Keys include molecules, parameters, implementation, package versions; logs report progress.
  Keep these locally generated pickle caches with their run; do not load untrusted caches.

### Hydrogen-free models and runtime

| Option | Effect |
| --- | --- |
| Default H policy | Preserve SDF representation for MMFF, matching original evaluator. Chemistry honors implicit H. |
| `--energy-hydrogens add` | Add H coordinates to an energy-only copy, leaving heavy atoms initially fixed. Added H enters per-atom normalization and optimization RMSD; other metrics/PoseBusters use loaded SDF molecules. Compare models using the same H policy. |
| `--remove-hs` | Remove H on load for other metrics; raw RDKit stability retains original graph atoms. Unnecessary for hydrogen-free SDFs. |
| `--no-energy` | Skip explicit MMFF metrics; PoseBusters still runs its own energy check. |
| `--no-posebusters` | Skip PoseBusters and its dependent intersections. |
| `--no-diversity` | Skip quadratic pairwise fingerprints. |
| `--workers` | Default at most four available CPUs; 1 runs in process. Each PoseBusters worker uses one RDKit thread to prevent oversubscription. |

### Molecular identity and reports

`--reference-smiles` expects one training SMILES per line from the checkpoint's training
split, not validation/test or all QM9. Blank lines are ignored; invalid entries/empty
references error. Without a reference, novelty-dependent metrics are omitted;
reference-distribution comparisons are unavailable.

Generated/reference identities use canonical H-stripped SMILES, ignoring stereochemistry
by default: SDF-inferred stereo would otherwise inflate novelty against stereo-free graphs.
`--smiles-stereochemistry retain` enables stereo-sensitive identity on both populations.
Stereo removal uses copies, preserves isotopes, and leaves energy/PoseBusters inputs untouched.

| JSON field | Contents |
| --- | --- |
| `results`, `counts` | Scores, denominator/averaging counts; `counts.combined_numerators` contains integer joint-rate numerators. Undefined means are `null`. |
| Provenance/settings | Input paths/SHA-256, reference hash/canonical size, package versions, identity/H policies, notes. |
| `molecule_results` | SMILES, validity/novelty/joint PoseBusters flags; pooled SDF order followed by omitted attempts, **not generation order**. Loaded rows include SDF path/index/name for attempt-ID mapping. Disabled/unavailable-reference flags are `null`; known failures fail enabled checks. |

Differences from the old evaluator: corrected inverted RDKit flags, implicit-H-aware
valence checks, H-normalized uniqueness, descriptive names replacing `semla-*`, and
failed attempts included in molecule stability. Added diagnostics include diversity,
optimized mean energy, convergence counts, and joint PoseBusters validity; original
MMFF energy/strain/aligned-RMSD formulas remain.

## Scaffold decoration in a pocket: DiffDec evaluation

[`scripts/evaluate_scaffold_pocket.py`](../scripts/evaluate_scaffold_pocket.py) provides
validity, uniqueness, similarity, and recovery using
[`morph.utils.metrics.diffdec`](../src/morph/utils/metrics/diffdec.py).
Source: [DiffDec `compute_metrics.py`, revision `916ae142`](https://github.com/biomed-AI/DiffDec/blob/916ae14207b2783a90336bb8509374535c5791f9/compute_metrics.py);
upstream MIT license retained beside implementation. Sampling marks docking `not_run`;
CPU docking is a separate stage using the saved metric file.

### CrossDocked docking requirements and reproduction limits

```bash
bash setup_env.sh --extra crossdocked
source .venv/bin/activate
qvina2 --version
obabel -V
prepare_receptor4.py -h
```

| Dependency/setup | Contract |
| --- | --- |
| Python extra | `crossdocked`: Open Babel `openbabel-wheel==3.1.1.23`, Biopython, EasyDict, dataset downloader. Base setup excludes these. |
| Native tools | QuickVina 2.1.0, MGLTools 1.5.7, pinned in `pyproject.toml` `[tool.morph.crossdocked]`. Pip/uv alone cannot install them. |
| Installation | Linux x86_64; checksum-verified micromamba installs under `.venv/crossdocked/`. Bundled Python 2 stays isolated. Wrappers `qvina2`/`prepare_receptor4.py` in `.venv/bin`; no conda activation. URLs/checksums/startup checks in `.venv/crossdocked/installed.json`. |
| Existing extra | After `uv sync --locked --extra crossdocked`, run `.venv/bin/python scripts/install_crossdocked_tools.py`. Re-running verifies installation; `setup_env.sh --inexact` preserves other installed extras. |

Run docking after sampling:

```bash
uv run --no-sync python -m morph.utils.docking \
  --metrics-file eval_outputs/pocket_diffdec/crossdock_test_metric.smi \
  --receptor-root /path/to/crossdocked_pocket10 \
  --output-dir eval_outputs/pocket_diffdec/docking \
  --workers 8 --seed 42 --timeout 900
```

- One JSONL record per prediction/reference, including failures. Same command resumes completed
  tasks; changed inputs/protocol are rejected. Shard with `--shard-count N --shard-index I`
  and separate output directories. `morph.utils.docking.aggregate_docking` requires all shards,
  rejecting missing/duplicate records and mixed condition blocks.
- The historical p10 epoch-100 run aggregated both panels, three seeds, six benchmark metrics,
  reference affinity, and failure counts into one `summary.csv`.

#### Docking protocol and numerical differences

| Aspect | Released DiffDec protocol / reproduction limit |
| --- | --- |
| Toolchain | QuickVina 2, Open Babel, AutoDockTools; Python `vina` alone is insufficient. Upstream main environment pins Python 3.10.5 / RDKit 2022.03.2, with separate `adt` environment and unpinned docking versions. Current Python 3.13/RDKit 2025 lacks upstream `rdkit.six`; audit compatibility or isolate the older environment. |
| Ligand pose | Rebuild from prediction SMILES, add H, embed, UFF-minimize. **Generated in-pocket coordinates are not scored.** |
| Search | 30×30×30 Å box centered at receptor bounding-box midpoint; exhaustiveness 16, one CPU, docking seed 1. |
| Embedding seed | Upstream unspecified; Morph records deterministic per-attempt seeds. This and newer RDKit prevent claims of exact numerical reproduction. |
| Invalid predictions | Released preprocessing substitutes scaffold before docking; preserve/report this convention for benchmark reproduction. |
| References | Re-dock through same procedure. High affinity: `prediction <= reference`; preserve upstream missing-reference handling and 100-attempt ordering. |
| Vina aggregation | Average successful first-mode affinities within reference-SMILES/protein groups, then nonempty groups equally. Preserve failures and bounded subprocesses; upstream early returns/empty results must not truncate runs silently. |

Sources: [docking script](https://github.com/biomed-AI/DiffDec/blob/916ae14207b2783a90336bb8509374535c5791f9/vina_docking.py),
[preprocessing](https://github.com/biomed-AI/DiffDec/blob/916ae14207b2783a90336bb8509374535c5791f9/vina_preprocess.py),
[environment](https://github.com/biomed-AI/DiffDec/blob/916ae14207b2783a90336bb8509374535c5791f9/environment.yaml).

#### Historical condition-mapping audit

Local audit **2026-10-15** verified Zenodo checksums and mapped all **43 single / 102 multi**
conditions (4300 / 10,200 attempts per seed). No missing receptor PDBs or vocabulary exclusions
among 100 native test ligands. Multi conditions map to 31 ligand records and 80 distinct
protein/scaffold groups; repeated conditions must remain. Pocket coordinates were checked.

Original reference CSV tables are absent from the pinned repository/Zenodo release;
exact raw grouping strings remain a limitation. This verified mapping only, without
docking, coordinate changes, or a model benchmark.

### Score existing predictions

```bash
uv run --no-sync python scripts/evaluate_scaffold_pocket.py \
  --metrics-file /path/to/crossdock_test_metric.smi \
  --output-dir eval_outputs/diffdec_scores
```

Metric file: one row per attempt, fields separated by **literal spaces**:

```text
<scaffold_smiles> <reference_smiles> <prediction_smiles> <rgroup_smiles> <protein_filename>
```

Empty prediction/R-group fields use adjacent spaces; four columns without R-group also
work. R-group does not affect these metrics. Preserve failed attempts, raw strings,
duplicates, and protein identities. Output directory must not already exist.

### Sample and evaluate a checkpoint on published conditions

Required: saved composed `.hydra/config.yaml`, matching preprocessing assets, processed
CrossDocked test split, and [published DiffDec tensor](https://zenodo.org/records/10527451).
Multi-R-group example:

```bash
uv run --no-sync python scripts/evaluate_scaffold_pocket.py \
  --checkpoint /path/to/epoch.ckpt \
  --config /path/to/run/.hydra/config.yaml \
  --assets-root /path/to/pretrained/processed \
  --data-root data/crossdocked \
  --diffdec-data /path/to/crossdockmulti_test_full.pt \
  --device cuda --batch-size 16 --weights ema \
  --output-dir eval_outputs/pocket_diffdec
```

- Generate **100 attempts per published instance**, including repeated scaffold/pocket cases.
  Every UUID must map uniquely to a compatible test ligand/protein. Mapping checks absolute
  scaffold coordinates, atomic numbers, and pocket coordinates; unmatched/ambiguous/incompatible
  cases fail. No substituted cuts, dropped cases, or LibINVENT dependency.
- Without reference CSVs: ligand SMILES from tensor `name`, scaffold from native graph, protein
  from CrossDocked metadata. Supply `--diffdec-table /path/to/crossdockmulti_test_table.csv`
  when available to preserve exact raw strings/grouping and check scaffold topology.
  Path-prefix normalization applies only to matching; scoring retains supplied strings.
- Saved conditioning/pocket representation and movable-coordinate policy apply. Steps and
  EMA/current selection default to run config; override with `--steps 100`, `--weights ema`
  or `current`, and repeatable `--set KEY=VALUE`. One device, requested split only, no replay.
- Smoke check: `--device cpu --steps 2 --limit-conditions 1` still generates 100 attempts.
  Reports mark subsets; this is not a full benchmark.

#### Our cached cuts

Replace `--diffdec-*` with `--cached-cuts` to evaluate all current cuts in the selected
split; `--split val` chooses validation. Independent of training's 512×8 panel.
Reports label this `morph_cached_cuts`, versus `published_diffdec`: same metric
formulas, different benchmark conditions.

### Exact scoring conventions

| Metric | Released definition | Unit |
| --- | --- | --- |
| Validity | Nonempty prediction, textually different from scaffold, contains scaffold substructure; parsing retries unsanitized. Average attempts per protein/raw-scaffold group, groups per protein, then proteins. | Percent |
| Uniqueness | Distinct raw valid prediction SMILES / **100** per protein/raw-scaffold group; average groups per protein, then proteins. Omit groups with no valid outputs. | Percent |
| Similarity | Prediction/reference RDKit `RDKFingerprint` Tanimoto; average valid attempts per protein, then proteins with valid attempts. No scaffold-group averaging. | Fraction |
| Recovery | Per protein, fraction of distinct canonical scaffolds with any valid reference-matching prediction after removing stereo/H; average proteins, including zero-recovery proteins. | Percent |

Keep uniqueness denominator 100 even when repeated instances merge into a group;
record/flag other group sizes without correcting/clipping scores. Uniqueness is not
normalized by valid count; similarity is not pairwise diversity (no diversity metric here).
Empty input errors. All-invalid input gives undefined uniqueness/similarity (`NaN` Python,
`null` JSON), validity/recovery zero; upstream divides by zero. Other errors propagate.

### Chemistry conversion and output artifacts

Identical metric files and RDKit versions reproduce upstream scoring. Sampling differs:
Morph uses predicted bonds/charges; DiffDec perceives XYZ bonds with Open Babel. Retain
largest connected component as in DiffDec `reformat.py`, without Morph's strict-validity
prefilter. Reports record conversion/reference choices; exact scoring does not imply
identical model inputs or bond perception.

| Artifact | Contents |
| --- | --- |
| `crossdock_test_metric.smi` | All attempts; upstream-compatible. |
| `metrics.json` | Scores/units/groups, docking status, versions/revision, hashes, vocabulary filtering, settings. |
| `attempts.jsonl` | All attempts with condition/sample IDs and conversion/SDF failures; authoritative attempt count. |
| `conditions.json` | UUIDs, ligand/retained-atom indices, references, proteins. |
| `generated.sdf` | Serializable molecules identified by attempt ID; original receptor frame in Å. Failed conversions may be absent. |
| `config.yaml` | Resolved sampling config. |

Regression coverage: upstream golden scores, empty fields, duplicates, stereo,
unsanitized validity, atom mapping, and 100-attempt CPU checkpoint sampling.
Training's stricter validity and Morgan diversity remain separate definitions.
