# Runtime and performance

Scope: runtime switches, architecture compatibility, and historical benchmarks.
Related: [Slurm setup](slurm.md), [training](training.md), [inference](inference.md).

## Compilation

| Setting | Behavior |
| --- | --- |
| `MORPH_COMPILE=0` | Disable compilation in training and lower-level generation entry points; useful for smoke tests or GPUs without native BF16. |
| Default compilation | In-place `nn.Module.compile(dynamic=True)` on current and EMA networks; lazy compilation preserves parameters, EMA references, and checkpoint keys. EMA runs without gradients in evaluation mode. |
| Pretrained inference | `load_pretrained()` starts eager with one network. Opt in with `model.module.compile()`. |
| `MORPH_COMPILE_DDP_OPTIMIZER=0` | Disable DDP compiler graph splitting. |
| `MORPH_COMPILE_FALLBACK=1` | Warn and fall back to eager on compiler failure, preserving model/optimizer state. |
| `MORPH_MUON_EXECUTION=native` | Use native Muon execution (the default), independently of model compilation. CPU always uses native Muon; checkpoints are compatible with CUDA graph replay. |
| `MORPH_FUSED_BROADCAST=0` | Use native condition indexing, including for higher-order differentiation. Default CUDA training uses first-order Triton reductions over contiguous atom ranges: FP32/BF16 inputs, FP32 accumulation. CPU, inference, and other dtypes use native indexing. |
| `data.datamodule.num_workers.train=N` | Set data-loader workers per training process through a Hydra override. Budget workers across all GPU processes within the allocated CPU count. |

Numerical and memory considerations:

- Fused condition-gradient reduction changes summation order; training is not bitwise identical.
- Categorical embeddings project each category once per forward. Sampling gathers projections;
  training uses dense lookup and preserves independent per-token dropout. BF16 rounding can
  change trajectories. Sampling reuses known atom counts to avoid CUDA histograms.
- Pinned PyTorch 2.9 needs a compiled bias-gradient metadata correction across multiples-of-16
  sequence lengths. Tests cover outputs/gradients; native attention arithmetic is unchanged.
- Batches remain shuffled; no molecular bucketing. Training prefetch is one batch per worker.

## Optional FlashAttention acceleration

SDPA is the default for QM9/GEOM. Pair-biased DiT blocks do not use the optional
packed FlashAttention path; only custom components with unbiased attention can use it.

```bash
uv sync --locked --extra flash-attn
MORPH_FLASH_ATTN=1 python run.py
```

Building requires a CUDA toolkit/compiler compatible with Torch. The optional path
requires Ampere or newer; unsupported GPUs or missing packages fall back to SDPA.
Benchmark large/ragged workloads before enabling it. Compare molecules/second when
batch sizes differ, rather than milliseconds/step.

## Pair-bias architecture and compatibility

The geometry network computes pair/head bias from distance, bond features, and atom-pair
type once per forward. Each layer/head learns a multiplier initialized to one:
8 layers × 16 heads add 128 parameters. Multipliers can emphasize, suppress, or reverse
bias; they do not change geometry features. This architecture was selected on QM9;
its quality advantage on GEOM has not been measured.

Current YAML exposes no bias mode/depth switch. Scaling checkpoints load directly.
Legacy configs with `pair_bias_all_layers` but no mode retain shared bias. To rebuild
an old shared-bias model from current presets, match its original architecture:

```bash
python run.py +model.backbone.pair_bias_mode=shared \
  +model.backbone.pair_bias_all_layers=true
```

Use `false` for legacy first-layer-only checkpoints.

## ChEMBL3D attention comparison

**Historical experiment protocol; experiment launchers are not public recipes.**
`MORPH_ATTENTION_BACKEND=flex` selects differentiable, pair-biased CUDA FlexAttention
(FP32 bias capture for PyTorch 2.9 backward). Default `sdpa`, CPU behavior, parameters,
and checkpoint format are unchanged.

The three-day max100 run used explicit-H molecules capped at 100 total atoms and
recomputed statistics: source-count prior 3–105 versus unfiltered 3–146. This data
filter does not cap generated/intermediate states.

| Preflight stage | Rule |
| --- | --- |
| Attention comparison | Fresh SDPA/Flex/Flex/SDPA processes on one GPU; batch 80, accumulation 4; 64 warm-up + 128 measured microbatches; verify outputs, Q/K-normalization gradients, and learned-bias gradients. |
| Backend selection | Flex needs ≥5% throughput gain in both matched trials and <22 GiB reserved memory. Failed Flex retains checked SDPA; failed SDPA stops the job. |
| Integration check | Two-GPU smoke test: selected backend, generation, fifth-epoch saving. Native Muon after intermittent RTX 4090 optimizer-capture failures; model compilation enabled. |
| Run budget | 72 hours including preflight; reserve 45 minutes for shutdown. Validate each epoch; periodic/best checkpoints only every fifth epoch, plus `final.ckpt`. |

## ChEMBL3D recovery and batch sizing

On 2026-09-26, DDP compiler repartitioning failed after epoch 5 / step 11,515
(`tensor does not have a device`), followed by NCCL timeout. Recovery disabled DDP
compiler splitting, enabled eager fallback, and used a ten-minute DDP timeout.

A profiling run compared batch/accumulation 80/4, 160/2, 320/1 (effective batch 640
on two GPUs). Candidates needed finite losses/bias gradients and <20 GiB live
allocations in compiled and eager modes. Selection was checked by resuming the real
checkpoint for three short epochs on two GPUs before the long run continued.

Measured single-GPU result: compiled 320/1 reached **1714 molecules/s**, versus 951
for 80/4; peak live/reserved memory was 12.1/18.0 GiB. Eager 320/1 also passed
(17.5 GiB live). These are workload-specific observations, not throughput guarantees.
Checkpoint cadence remained every five epochs plus final.
