# Model package

The repository has one first-class generative architecture: the molecular DiT
in `backbones/dit.py`. Conditional applications wrap that backbone without
forking its clean denoising lane.

## Layout

| Area | Purpose |
| --- | --- |
| `backbones/dit.py` | Molecular DiT backbone and prediction-head assembly |
| `backbones/conditional_dit.py` | Optional global, node, edge, and cross-attention conditioning |
| `layers/attention.py` | Pair bias, geometric attention, and context-distance bias |
| `layers/heads.py` | Node, edge, insertion, deletion, and coordinate heads |
| `layers/lora.py` | Checkpoint-compatible low-rank linear adapters |
| `lightning/base.py` | The single Lightning module used by the DiT models |
| `lightning/_optimizer.py` | Muon/AdamW routing, stepping, clipping, and schedulers |
| `finetune/base/` | Shared checkpoint loading and conditional encoder components |
| `finetune/atom_count/` | Signed target-minus-current count conditioning |
| `finetune/logp/` | Absolute Crippen logP conditioning and validation metrics |
| `finetune/scaffold/` | Scaffold-decoration Lightning module and backbone wrapper |

Task packages re-export their public classes. Hydra targets therefore use paths
such as
`morph.model.finetune.scaffold.ScaffoldConditionalDiTBackboneWithHeads`, not
leaf-module or legacy flat paths.

## First-class DiT

`DiTBackboneWithHeads` embeds atoms, edges, diffusion time, and current graph
size, applies the DiT stack, and exposes the complete rate-model output
contract. Its pair representation is directed during featurization: each
directed edge receives its own source/target node features. Symmetry is imposed
only on prediction lanes whose mathematical targets are symmetric.

The default config is `configs/model/backbones/dit.yaml`. It uses the shared
Muon/AuxAdam optimizer recipe and the tuned multi-head loss weights.

`model.module.model.incident_bond_embedding=true` adds the sum of each atom's
existing incident edge embeddings, excluding `NO_BOND`, through a
zero-initialized linear projection into its node embedding. It preserves bond
counts, shares the edge embedding parameters with attention and the decoder,
and uses the existing prediction heads.

For GEOM, `data.interpolator.atom_type_noise_prob=0.01` adds training-only
categorical corruption after interpolation. Selected tokens are redrawn from
empirical alternatives, excluding their current value. The probability for
each undirected pair is scaled to match the expected number of atom changes;
clean targets, deletion-only nodes, and locked scaffold types/bonds are retained.
The default is zero. This augmentation does not change inference schedules.

Low-rank adaptation is opt-in through `model.lora`. The
`finetune/lora_finetune` model recipe loads a pretrained unconditional model,
freezes its ordinary parameters, and trains LoRA matrices on the attention and
SwiGLU block projections with AdamW. The adapter residual is exactly zero at
initialization, and its base `weight`/`bias` state-dict names are unchanged.

The insertion-content head is a separate, explicit Hydra choice. The default is
`insertion_head=unimodal`; use `insertion_head=gmm` for the mixture-density
alternative. Common state/edit heads live in `configs/heads/head.yaml`, while
the two insertion-head recipes live under `configs/insertion_head/`.

## Tensor contract

Let `B` be the number of molecules, `N` the total current atoms, `E` the
directed edges, `A`/`C`/`T` the atom/charge/bond vocabulary sizes, and `K` the
number of mixture components. Nodes and edges are contiguous by molecule.
Runtime graphs contain both directions of every non-self atom pair, including
the `<NO_BOND>` token. Output edges follow the input `edge_index` order.

`model(mols_t, t, max_seqlen=...)` receives:

| Input | Shape | Meaning |
| --- | --- | --- |
| `mols_t.positions` (`x`) | `[N, 3]` | Centered coordinates divided by training `coordinate_std` |
| `mols_t.atom_types` (`a`) | `[N]` | Integer atom-token indices |
| `mols_t.charge_types` (`c`) | `[N]` | Charge-token indices; retained in the container, not embedded |
| `mols_t.edge_types` (`e`) | `[E]` | Integer bond-token indices |
| `mols_t.edge_index` | `[2, E]` | Directed atom pairs; integer indices into current nodes |
| `mols_t.batch` | `[N]` | Molecule index for each node |
| `t` | `[B, 1]` | Interpolation time for each molecule |
| `max_seqlen` | Python integer | Largest current graph, resolved before a compiled forward pass |

The output dictionary contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `pos_head` | `[N, 3]` | Predicted clean coordinates in normalized units |
| `atom_type_head` | `[N, A]` | Clean atom-type logits |
| `charge_head` | `[N, C]` | Clean charge logits |
| `edge_type_head` | `[E, T]` | Clean bond-type logits, symmetric in endpoints |
| `do_sub_a_head` | `[N, 1]` | Logit for a pending atom substitution |
| `do_sub_e_head` | `[E, 1]` | Logit for a pending bond substitution, symmetric in endpoints |
| `do_del_head` | `[N, 1]` | Logit for a pending deletion |
| `ins_rate_head` | `[N, 1]` | Positive expected number of pending insertions per spawning node |
| `h_latent` | `[N, head_node_dim]` | Node features used by insertion-edge prediction |
| `insertion_head` | dictionary | Clean inserted-atom predictions, described below |

Unimodal insertion outputs are `mu: [N, 3]`, `a_logits: [N, A]`, and
`c_logits: [N, C]`. Each spawning node is supervised with its nearest assigned
pending insertion. The mean is the spawning atom's position plus a learned offset.

GMM insertion outputs are `mu: [N, K, 3]`, `sigma: [N, K]`, `pi: [N, K]`,
`a_probs: [N, K, A]`, and `c_probs: [N, K, C]`. `sigma` is an isotropic standard
deviation in normalized units; `pi`, `a_probs`, and `c_probs` are probabilities.
The mixture likelihood supervises every pending insertion assigned to a node.

Sampling applies sigmoid to edit logits and samples categorical logits. The
integrator multiplies edit posteriors and expected insertion counts by schedule
masses. At most one insertion per spawning node is applied in a numerical step.
Inserted bonds are predicted from clean inserted features before insertion noise
is applied. Final samples and saved trajectory frames are in Angstroms. Frozen
scaffold coordinates are restored to their input frame.

## Training targets and loss contract

Training receives `(mols_t, mols_1, insertion_targets, t)`. The first two states
share current-node and edge ordering; `mols_1` contains their clean targets.
`InsertionTargets` retains the future atoms and two structured bond groups.
Its spawn/existing indices address `mols_t`; inserted indices address the future
atoms. `collate_insertion_targets` offsets these spaces independently without
mutating individual samples.

Numerical `lambda_ins`, `lambda_del`, `lambda_a_sub`, and `lambda_e_sub` values
are retained on `mols_t`. Binary labels and supervision masks use their `> 0`
tests. `n_ins` is the pending insertion count used by Poisson NLL.

Training helpers return named `LossTerm(values, graph_mask)` entries. Values have
shape `[B, ...]`; masks, when present, have shape `[B]`. Time weights are applied
per molecule before averaging. Empty supervision produces zero-valued terms with
false masks and retains autograd dependencies required by distributed training.

| Loss | Supervision |
| --- | --- |
| `do_sub_a`, `do_sub_e`, `do_del` | Binary cross-entropy against positive lambda values |
| `sub_a_class`, `sub_e_class` | Cross-entropy on substitutions, excluding deletion targets and task-protected states |
| `ins_rate` | Poisson NLL on all current nodes, including zero insertion counts |
| `ins_state` | Unimodal position/type/charge loss or joint typed-GMM NLL |
| `ins_e`, `ins_e_ii` | Bond cross-entropy using target inserted-atom features |
| `x` | Coordinate MSE on supervised, non-deletion nodes |
| `c` | Charge cross-entropy on supervised, non-deletion nodes |

The representation controls charge and topology loss participation. Position
and charge time weighting is always active; `use_time_weights` additionally
enables schedule weighting for edit loss groups. Component weights come from
`configs/loss_weights/`.

## Conditional tasks

The conditional backbone composes the clean DiT and introduces zero-gated
conditioning ports. At initialization, a conditional model is therefore
identity-equivalent to the loaded unconditional checkpoint.

- Fourier atom-count conditioning embeds the signed target-minus-current count
  and passes it as `graph_cond`. The backbone owns the gate and adds the embedding
  after the intrinsic time/current-count projection. Existing Fourier checkpoint
  weights load into this layout automatically. The base `dit.py` stays unchanged.
- LogP conditioning embeds `target_logp`, one absolute Crippen logP per graph,
  through an MLP into the same graph port. The request survives topology edits
  and the zero gate preserves the pretrained generator at initialization.
  `finetune/logp` trains the complete backbone and output heads by default.
- Scaffold decoration derives node and edge conditions from `keep_mask`. It can
  also encode the scaffold subgraph as cross-attention context with a geometric
  distance bias. `freeze_keep_positions` controls whether kept coordinates are
  fixed (and returned in their input frame) or allowed to flow; discrete and
  coordinate loss exclusions remain independently configurable.

The fine-tuning base loads unconditional keys into the conditional clean lane,
resynchronizes EMA weights, and optionally freezes pretrained modules according
to each task wrapper's `freeze_pretrained()` policy.

`ConditionalDiTBackbone.forward` accepts these optional pre-embedded ports:

| Port | Shape | Meaning |
| --- | --- | --- |
| `cond_tokens` | `[B, L, ctx_dim]` | External context tokens |
| `cond_token_mask` | `[B, L]` | Boolean, true for valid context tokens |
| `cond_token_attn_bias` | `[B, L, N_max]` or `[B, H, L, N_max]` | Additive token-to-atom attention bias |
| `graph_cond` | `[B, d_model]` | Conditions aligned to graphs, added to the projected time/current-count condition |
| `node_cond` | `[N, d_model]` | Conditions aligned to current nodes |
| `edge_cond` | `[E, in_edge_nf]` | Conditions aligned to current directed edges, used only by the edge decoder; excluded from attention pair bias |

Here `L` is context length and `H` the number of attention heads. Each port
has its own zero-initialized tanh gate and corresponding `use_*` flag;
`use_graph_cond=true` enables the graph port. `ctx_dim` is required only for
cross-attention. `keep_mask` is a node-level boolean
mask carried through topology edits; newly inserted nodes are not retained atoms.

### Cross-attention placement and conditioning scale

`ConditionalDiTBackbone(cross_attn_position="residual")` preserves the existing
placement: the gated adapter updates the raw residual stream before each DiT
block. Omitting this option, including in older checkpoint configs, selects that
same behavior. Unconditional models and state-dict keys are unchanged.

Opt in with `cross_attn_position="post_norm1"` to apply the existing gated adapter
immediately after the block's first LayerNorm and before native AdaLN:

```python
z = norm1(h)
z = gated_cross_attention(z, cond_tokens)
h = h + native_attention_gate(c) * self_attention(AdaLN(z, c))
# The native feed-forward branch follows as before.
```

The raw `h` remains the residual skip. Both of the adapter's gated branches, when
enabled, operate on `z`. Zero adapter gates preserve the pretrained computation;
absent or fully masked contexts remain unconditional, including in mixed batches.
The native attention gate and AdaLN modulation still control this path's effect.

**The caller is responsible for providing correctly scaled, pre-embedded
conditioning tokens/features and validating the resulting gated update scale.**
Query/KV LayerNorm controls attention inputs; it does not match adapter updates
to the raw residual stream. Uniform token rescaling may be canceled by KV
LayerNorm. Inspect update magnitudes relative to the receiving features across
layers and timesteps. This placement option does not rescale the other ports.

Save `cross_attn_position` with the model configuration and retain it when
resuming. Both placements use the same parameter names and tensor shapes, but
switching the placement of a trained adapter changes its computation. Existing
pretrained models and checkpoints keep the legacy placement by default.

## Config entry points

The task-oriented Hydra groups mirror the source layout:

```text
configs/model/backbones/dit.yaml
configs/model/finetune/scaffold_finetune.yaml
configs/data/qm9/{qm9,scaffold_decoration}.yaml
configs/data/geom/{geom,scaffold_decoration}.yaml
configs/defaults/qm9/{qm9,scaffold_decoration}.yaml
configs/defaults/geom/{geom,scaffold_decoration}.yaml
```

Examples:

```bash
python run.py defaults=qm9/qm9
python run.py defaults=geom/geom
python run.py defaults=qm9/scaffold_decoration \
  model.module.pretrained_ckpt=/path/to/unconditional.ckpt
```

There are intentionally no compatibility aliases for the previous flat source
modules or Hydra group names.
