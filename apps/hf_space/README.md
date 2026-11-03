---
title: Morph Playground
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
python_version: 3.12
app_file: app.py
pinned: false
models:
  - maltefranke/morph
short_description: 3D molecules with logP and atom-count targets
---

# Morph Playground

Generate molecules with the released **GEOM-Drugs and QM9 logP models**, the
**GEOM Fourier atom-count model**, and unconditional models for **GEOM-Drugs,
QM9 and ChEMBL3D** (explicit or implicit hydrogens).

**GEOM-Drugs unconditional (epoch 100) is the default.** Choose a model, set a
target when applicable, and generate up to eight molecules on a shared GPU.
Explore structures in 2D and 3D, then open **Trajectory** to play the actual
generation process. The player supports pause, restart, frame scrubbing,
playback speed, looping and a bond-display toggle. Atom counts and bonds update
with each sampled state, including insertions and deletions.

Download final structures as SDF, attempt-level properties as CSV, and all
trajectories as a ZIP containing JSON, multi-frame XYZ and generation metadata.
Trajectories contain the initial prior plus all 100 integration steps. Playback
uses 25 frames per second at 1× speed. Coordinates are in angstrom; saved times
follow the sampler's logarithmic grid, whose last boundary is 0.999.

All models use EMA weights and the evaluated 100-step sampler. LogP means
RDKit Crippen MolLogP. The initial atom count is independent of the target:
GEOM logP uses uniform 3–76, QM9 logP uses 3–34, and GEOM N-conditional uses
the broader 3–120 prior. N includes explicit hydrogen atoms, except for the
ChEMBL3D implicit-H model. Large N targets and extreme logP targets are
extrapolations; targets are not guaranteed and validity can decrease.

Every generation attempt is retained in the table. SDF includes sanitized
structures, including those failing the separate connectedness/radical checks.
Chemical validity is not a 3D geometry assessment; PoseBusters is not run here.
Full evaluation results and checkpoint descriptions are in the
[model card](https://huggingface.co/maltefranke/morph).

The Space uses **real CUDA inference in bfloat16 on Hugging Face ZeroGPU**.
All seven models have been checked on an NVIDIA RTX PRO 6000 Blackwell GPU,
including trajectory exports and a batch of eight molecules targeting N=120.
The first request can take longer while its model initializes. Hugging Face
login, queueing and daily GPU quotas may apply. No CPU fallback is used.
The interface does not require a separate GitHub release:
the inference runtime is included here and in the model bundles. Scaffold and
pocket-conditioned workflows are available in the model repository and require
their specialized inputs; this interface covers free molecule generation.

Checkpoints are downloaded from immutable revisions in `models.json`, verified
with SHA-256, and loaded strictly using EMA weights. `runtime/morph` contains
the inference source used for the released logP evaluations. The Space uses
PyTorch 2.9.1 and does not compile the networks.
The native PyTorch kernel paths are used instead of the optional Triton helpers.
Model dependencies and weights are loaded inside the allocated GPU worker:
importing the scientific stack at parent startup poisoned the CUDA state
inherited by ZeroGPU workers. Checkpoint files are downloaded in advance, and
loaded models are cached within each worker.
