# Running on a Slurm cluster

Scope: environment preparation and single-node GPU jobs on Slurm.
Commands run from the repository root. Adapt resource requests, partitions,
accounts, and any environment modules to your cluster's policies.
Related: [training](training.md), [performance](performance.md), [inference](inference.md).

## Prepare the environment and data

Keep the checkout, Python environment, and processed data on storage visible to
the compute nodes. Install the environment before submitting training:

```bash
bash setup_env.sh
source .venv/bin/activate
source .env
```

By default, runs are written to `<repo>/outputs/runs`. To use shared scratch,
run `MORPH_SCRATCH=/path/to/shared/scratch/morph bash setup_env.sh` for the first setup.
Setup creates `.env` only if it is absent; edit it to change existing paths.
Sourcing `.env` shares `MORPH_RUN_ROOT` between Hydra outputs, checkpoints, and
W&B. Check your site's scratch retention policy and copy checkpoints you want
to retain to persistent storage.

Download a prepared dataset or run preprocessing once on an appropriate CPU
worker, as described in the [training quick start](../README.md#training-quick-start).
Prepare pretrained weights and other downloads before starting jobs if compute
nodes lack internet access. Put caches on shared storage and use the inference
CLI's `--offline` option for generation from cached models.

## Submit a GPU training job

Save the following as `train.slurm` in the repository root. This example requests
two GPUs on one node and runs one `torchrun` launcher, which starts one training
process per GPU:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=morph
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
# Load any site-required Python/CUDA modules here before activating the environment.
source .venv/bin/activate
source .env
export OMP_NUM_THREADS=1

torchrun --standalone --nnodes=1 --nproc_per_node=2 run.py \
  defaults=geom/geom trainer.trainer.accelerator=gpu \
  trainer.trainer.devices=2 \
  data.datamodule.batch_size.train=64 \
  data.datamodule.num_workers.train=4 \
  +trainer.trainer.max_time=00:23:30:00 \
  trainer.do_validate=false trainer.do_predict=false \
  logging.enabled=false "$@"
```

Create the log directory before submitting, then launch from the repository root:

```bash
mkdir -p slurm_logs
sbatch train.slurm
# If your site requires them, add its account and GPU partition:
# sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION train.slurm
```

- Match `--gpus-per-node`, `--nproc_per_node`, and `trainer.trainer.devices`.
  For one GPU, set all three to `1`; keep `--ntasks=1` for this launcher pattern.
  Sites using GRES syntax can replace `--gpus-per-node=2` with `--gres=gpu:2`
  (adding a GPU type if required). See the [Slurm options](https://slurm.schedmd.com/sbatch.html).
- CPU and memory requests apply to the whole job here. Data-loader workers and
  batch sizes are **per GPU**; four workers on each of two GPUs use eight workers
  in total. Tune these values for your allocation and GPU memory.
- Keep `max_time` below Slurm's wall-time limit to allow validation and checkpoint
  writes to finish. The example reserves 30 minutes and disables separate post-fit
  validation/prediction; validation during training remains enabled.
- The example disables W&B. Use `logging.enabled=true` with suitable credentials
  and network access. For offline W&B logging, enable it and also set
  `WANDB_MODE=offline` to log locally for later syncing.
- Slurm reads `#SBATCH` directives without shell variable expansion. Set changing
  paths on the `sbatch` command line, where the shell can expand them.
- Jobs read the checkout when they start. Keep the code stable while jobs are
  queued/running, and record the source revision and any local changes.

This example is for a single node. Multi-node runs need a shared rendezvous and
one launcher per node; `--standalone` is not a multi-node configuration.

## LogP fine-tuning

Prepare the dataset and its ordered property arrays using
[training](training.md#logp-conditioned-generation), then select a compatible
pretrained checkpoint. Reuse the job script above with a different preset:

```bash
export MORPH_PRETRAINED_CKPT=/path/to/geom/explicit_h/epoch_100.ckpt
sbatch train.slurm defaults=geom/logp

export MORPH_PRETRAINED_CKPT=/path/to/qm9/explicit_h/epoch_1500.ckpt
sbatch train.slurm defaults=qm9/logp
```

Slurm exports the submitting shell's environment by default. The script forwards
arguments to Hydra, so these overrides replace its GEOM training preset while
retaining the requested resources and time limit. The logP recipes train up to
50 GEOM epochs or 500 QM9 epochs and save `final.ckpt` when fitting ends.
