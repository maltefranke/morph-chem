# Third-party code and notices

Standalone copied modules live in this directory. Adaptations integrated with
Morph's metrics and RDKit helpers keep their existing module paths, with pointers
to the notices collected under [licenses/](licenses/).

These upstream MIT licenses apply to the respective upstream contributions.
Morph's original code is also MIT-licensed; see the repository's root `LICENSE`.
Model weights, datasets, downloaded resources, and installed dependencies have
their own licensing.

The upstream snapshots below were reviewed on 2026-11-02. Unless explicitly stated,
a reviewed revision identifies the source and license inspected for this inventory;
it is not a claim that the local adaptation was originally imported from that commit.

## Standalone vendored modules

### Muon

- Local file: [muon.py](muon.py).
- Upstream: [KellerJordan/Muon, muon.py](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py).
- License: MIT, Copyright (c) 2024 Keller Jordan.
- Notice: [MUON_LICENSE](licenses/MUON_LICENSE), copied verbatim from
  [upstream LICENSE](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/LICENSE).
- Local adaptations include configurable Newton–Schulz iterations, optimizer
  state handling, shape batching, and distributed execution. The original
  implementation's contributor comments and algorithm references are retained.

### GEOM-Drugs Revisited preprocessing

- Local file: [geom_drugs_preprocessing.py](geom_drugs_preprocessing.py).
- Upstream: [isayevlab/geom-drugs-3dgen-evaluation, scripts/geom_drugs_processing.py](https://github.com/isayevlab/geom-drugs-3dgen-evaluation/blob/bfc5bed26c0ccfae0a2edba043735d50e3433c31/scripts/geom_drugs_processing.py).
- License: MIT, Copyright (c) 2025 Ian Dunn.
- Notice: [GEOM_DRUGS_LICENSE](licenses/GEOM_DRUGS_LICENSE), copied verbatim from
  [upstream LICENSE](https://github.com/isayevlab/geom-drugs-3dgen-evaluation/blob/bfc5bed26c0ccfae0a2edba043735d50e3433c31/LICENSE).
- The local implementation matches this upstream file apart from its module
  docstring, which adds the paper citation and attribution.
- This is the processing code's license, not a license grant for the GEOM dataset.

## Integrated adaptations and protocol references

### SemlaFlow

- Local files: [chemistry.py](../utils/metrics/chemistry.py) and
  [rdkit_utils.py](../utils/rdkit_utils.py).
- Upstream: [semlaflow/util/metrics.py](https://github.com/rssrwn/semla-flow/blob/3f43103d3af138b86dbe9f29fe8085e83f9a6283/semlaflow/util/metrics.py)
  and [semlaflow/util/rdkit.py](https://github.com/rssrwn/semla-flow/blob/3f43103d3af138b86dbe9f29fe8085e83f9a6283/semlaflow/util/rdkit.py).
- License: MIT, Copyright (c) 2024 Ross Irwin.
- Notice: [SEMLAFLOW_LICENSE](licenses/SEMLAFLOW_LICENSE), copied verbatim from
  [upstream LICENSE](https://github.com/rssrwn/semla-flow/blob/3f43103d3af138b86dbe9f29fe8085e83f9a6283/LICENSE).
- Adaptations include molecular metrics, the valence table and sanitization
  checks, periodic-table helpers, and RDKit energy/optimization/RMSD helpers.
  Morph adds its own representations, metric semantics, batching, and caching.

### DiffDec

- Local files: [diffdec.py](../utils/metrics/diffdec.py) and
  [docking.py](../utils/docking.py).
- Recorded upstream revision: `916ae14207b2783a90336bb8509374535c5791f9`,
  [compute_metrics.py](https://github.com/biomed-AI/DiffDec/blob/916ae14207b2783a90336bb8509374535c5791f9/compute_metrics.py),
  [vina_docking.py](https://github.com/biomed-AI/DiffDec/blob/916ae14207b2783a90336bb8509374535c5791f9/vina_docking.py),
  and [vina_preprocess.py](https://github.com/biomed-AI/DiffDec/blob/916ae14207b2783a90336bb8509374535c5791f9/vina_preprocess.py).
- License: MIT, Copyright (c) 2024 Biomed-AI@NSCC-GZ.
- Notice: [DIFFDEC_LICENSE](licenses/DIFFDEC_LICENSE), verified against
  [upstream LICENSE](https://github.com/biomed-AI/DiffDec/blob/916ae14207b2783a90336bb8509374535c5791f9/LICENSE).
- The evaluator preserves the released metric definitions with documented empty
  input handling. The docking integration adds bounded/resumable execution and
  explicit seeds; it does not claim bitwise reproduction of upstream results.

### PoseCheck and TABASCO

- Local file: [posecheck_strain.py](../utils/metrics/posecheck_strain.py).
- PoseCheck source: [posecheck/utils/strain.py](https://github.com/cch1999/posecheck/blob/56799eb242dbeb0df94076a9a36774613e3010b1/posecheck/utils/strain.py).
- PoseCheck license: MIT, Copyright (c) 2023 Charlie Harris;
  [POSECHECK_LICENSE](licenses/POSECHECK_LICENSE), copied verbatim from
  [upstream LICENSE](https://github.com/cch1999/posecheck/blob/56799eb242dbeb0df94076a9a36774613e3010b1/LICENSE).
- TABASCO source: [src/tabasco/utils/metrics.py](https://github.com/carlosinator/tabasco/blob/097eaae106e90a88294a93d42a3f91f1798f383f/src/tabasco/utils/metrics.py).
- TABASCO license: MIT, Copyright (c) 2025 Carlos Vonessen;
  [TABASCO_LICENSE](licenses/TABASCO_LICENSE), copied verbatim from
  [upstream LICENSE](https://github.com/carlosinator/tabasco/blob/097eaae106e90a88294a93d42a3f91f1798f383f/LICENSE).
- Morph implements PoseCheck's UFF strain-energy protocol with local error
  handling, deterministic conformer sampling, and process-pool integration.
  TABASCO is the reference for the metric wrapper and default median aggregation;
  the local module is not a verbatim copy of either upstream module.

## Distribution and maintenance

The license texts are included verbatim through `project.license-files`.
Both wheel and source distribution contain Morph's root `LICENSE` and all six
upstream notices; installed wheels include them under their standard
`.dist-info/licenses/` directory.

When adding copied code, preserve upstream copyright and license text, record the
source URL and revision, and describe local changes. For code adapted into an
existing module, keep the attribution near the implementation and add it to this
inventory. Update the notices whenever the copied upstream material changes.
