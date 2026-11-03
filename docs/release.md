# Release metadata

Scope: publication metadata and release provenance.
Related: [bundle format](inference.md#add-geom-qm9-or-another-model),
[training bundles](reproducibility.md).

## Sources of truth

| Item | Location / rule |
| --- | --- |
| Package name, version, dependencies | [pyproject.toml](../pyproject.toml) |
| Paper title and authors | [CITATION.cff](../CITATION.cff); [README BibTeX](../README.md#citation) |
| Paper | [Franke et al., 2026, arXiv:2606.07239](https://arxiv.org/abs/2606.07239) |
| Code license | [MIT](../LICENSE). Retain the [third-party notices](../src/morph/_vendor/README.md). |
| Model discovery | Hub [catalog.json](https://huggingface.co/maltefranke/morph/blob/main/catalog.json); entries pin immutable bundle revisions. |

## Release requirements

- Associate each reported result with source commit, experiment config, preprocessing,
  seeds, checkpoint, and matching statistics/vocabularies.
- Keep paper and code versions separate in the README header; update its code version
  with `pyproject.toml` and its paper metadata with `CITATION.cff`.
- Each model folder includes inference assets and `training_bundle.tar.gz`;
  `training_bundle.json` records archive checksum and provenance.
- Publish compatible bundles before adding catalog aliases. Direct repository/subfolder/revision
  and local loading need no catalog entry or Python registry change.

## Documented model releases

| Model | Completed epochs |
| --- | --- |
| QM9 explicit H | 1000, 1500 |
| GEOM explicit H | 50, 100 |
| ChEMBL3D explicit H | 200, 300, 400 |
| ChEMBL3D implicit H | 100, 200, 300 |

ChEMBL3D entries pin `ab9bea92f499f03450ebc99459fce04461504a41` in
[maltefranke/morph](https://huggingface.co/maltefranke/morph), including configs,
ordered vocabularies, distributions, original checkpoints, frozen source, and validation reports.
Use `morph models list` for current discovery.
