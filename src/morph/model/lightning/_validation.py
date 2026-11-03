"""Validation: per-batch and epoch-end metrics.

Mixin split out of :class:`morph.model.lightning.base.LightningModuleRates`.
"""

import os
from contextlib import ExitStack

import torch
import torch.distributed as dist

from morph.utils.metrics import (
    build_marginal_plots,
    calc_atom_stabilities,
    calc_metrics_,
)
from morph.utils.metrics.tensor import compute_batch_geometry


class ValidationMixin:
    def _validation_context(self, batch):
        """Optionally snapshot task-specific inputs before sampling."""
        return None

    def _update_task_metrics(self, context, mols, rdkit_mols):
        """Consume generated graphs and decoded molecules in conditional tasks."""

    def _compute_chemistry_metrics(self, rdkit_mols):
        """Evaluate the gathered population, optionally with task context."""
        return calc_metrics_(rdkit_mols, self.metrics)

    def on_validation_epoch_start(self):
        # ``seed_everything`` deliberately gives every DDP process the same base
        # seed. That is correct for model initialization, but an unconditional
        # validation dataset otherwise draws the same priors on every rank and
        # turns 4,096 requested samples into four copies of 1,024. Salt the
        # validation RNG by rank; keeping the seed fixed across epochs also makes
        # changes in the validation curve attributable to the model, not priors.
        self._training_rng_state = torch.random.get_rng_state()
        self._training_cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        base_seed = int(os.environ.get("PL_GLOBAL_SEED", "0"))
        torch.manual_seed(base_seed + self.global_rank)

        self.model_ema.eval()
        # Tensor-style collections accumulate raw counts/histograms across the
        # epoch so the reported metric is computed from the full epoch's atoms
        # (not a per-batch rate averaged with the wrong weighting).
        if self.distribution_metrics is not None:
            self.distribution_metrics.reset()
        if self.batch_metrics is not None:
            self.batch_metrics.reset()
        if self.stability_metrics is not None:
            self.stability_metrics.reset()
        if self.metrics is not None:
            self.metrics.reset()
        self._validation_rdkit_mols = []

    def validation_step(self, batch, batch_idx):
        context = self._validation_context(batch)
        batched_mols = self.sample(batch, batch_idx, return_traj=False)

        # Tensor / distribution metrics consume the batch directly and work in
        # every representation, so they always run — no early return. Only the
        # RDKit-mol chemistry metrics below require real topology.
        if self.batch_metrics is not None and len(self.batch_metrics) > 0:
            self.batch_metrics.update(compute_batch_geometry(batched_mols))
        if self.distribution_metrics is not None and len(self.distribution_metrics) > 0:
            self.distribution_metrics.update(batched_mols)

        if not self.representation.requires_topology:
            del batched_mols
            return

        # RDKit is CPU-only. Transfer each batch tensor once before splitting;
        # converting GPU graph views one molecule at a time would synchronize
        # every coordinate/type/edge ``.cpu()`` call separately.
        mols = batched_mols.to_inplace("cpu", non_blocking=False).to_data_list()
        del batched_mols

        # List of RDKit molecules or None
        rdkit_mols = []
        for mol in mols:
            try:
                rdkit_mols.append(
                    mol.to_rdkit_mol(
                        self.vocab.atom_tokens,
                        self.vocab.edge_tokens,
                        self.vocab.charge_tokens,
                    )
                )
            except Exception as e:
                print(f"Error converting molecule to RDKit: {e}")
                rdkit_mols.append(None)

        self._update_task_metrics(context, mols, rdkit_mols)
        self._validation_rdkit_mols.extend(rdkit_mols)

        # RDKit per-atom stability — fed precomputed bools (not RDKit mols).
        # Accumulate only here; compute/log/reset happens in
        # on_validation_epoch_end so the rate is weighted by atom count across
        # the full epoch, not molecule count per batch.
        if self.stability_metrics is not None and len(self.stability_metrics) > 0:
            stabilities = []
            for graph in mols:
                try:
                    mol = graph.to_rdkit_mol(
                        self.vocab.atom_tokens,
                        self.vocab.edge_tokens,
                        self.vocab.charge_tokens,
                        sanitize=False,
                    )
                    stabilities.append(calc_atom_stabilities(mol))
                except Exception as exc:
                    # The generated atom count is known even if decoding fails;
                    # do not drop these atoms or molecules from the denominator.
                    print(f"Raw graph stability check failed: {exc}")
                    stabilities.append([False] * graph.num_nodes)
            if stabilities:
                self.stability_metrics.update(stabilities)
        del mols

    def on_validation_epoch_end(self):
        # Python/list-backed chemistry metrics cannot use torchmetrics' DDP
        # state reduction. Gather the complete molecule population, compute it
        # once on rank zero, then broadcast the small scalar result dictionary
        # so every rank exposes identical callback metrics.
        if self.representation.requires_topology:
            local_mols = self._validation_rdkit_mols
            if dist.is_available() and dist.is_initialized():
                gathered = (
                    [None] * dist.get_world_size() if dist.get_rank() == 0 else None
                )
                dist.gather_object(local_mols, gathered, dst=0)
                rdkit_mols = (
                    [mol for rank_mols in gathered for mol in rank_mols]
                    if gathered is not None
                    else []
                )
            else:
                rdkit_mols = local_mols

            eval_metrics = {}
            if self.trainer.is_global_zero:
                try:
                    eval_metrics = self._compute_chemistry_metrics(rdkit_mols)
                except Exception as e:
                    print(f"Error calculating chemistry metrics: {e}")
            if dist.is_available() and dist.is_initialized():
                payload = [eval_metrics]
                dist.broadcast_object_list(payload, src=0)
                eval_metrics = payload[0]
            if eval_metrics:
                self.log_dict(
                    {f"val/{key}": value for key, value in eval_metrics.items()},
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=False,
                )
            self._validation_rdkit_mols = []

        # RDKit per-atom stability — accumulated across the full epoch in
        # validation_step; compute the atom-weighted rate once here.
        if self.stability_metrics is not None and len(self.stability_metrics) > 0:
            try:
                stab_results = self.stability_metrics.compute()
                self.log_dict(
                    {
                        f"val/{k}": (v.detach() if isinstance(v, torch.Tensor) else v)
                        for k, v in stab_results.items()
                    },
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=False,
                )
            except Exception as e:
                print(f"Error in stability-metric epoch-end logging: {e}")
            finally:
                self.stability_metrics.reset()

        # Tensor-based metrics run in every representation; logged here so the
        # values land in the same wandb step as everything else.
        if self.batch_metrics is not None and len(self.batch_metrics) > 0:
            try:
                # Sync once for both scalar computation and histogram plots.
                # Every rank must enter each collective context; only rank zero
                # reads the synchronized state to render figures.
                with ExitStack() as stack:
                    for m in self.batch_metrics.values():
                        stack.enter_context(m.sync_context())

                    batch_results = self.batch_metrics.compute()
                    self.log_dict(
                        {
                            f"val/batch/{key}": (
                                v.detach() if isinstance(v, torch.Tensor) else v
                            )
                            for key, v in batch_results.items()
                        },
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=False,
                    )

                    if self.trainer.is_global_zero and hasattr(
                        self.logger, "log_image"
                    ):
                        from morph.utils.metrics.plotting import (  # noqa: PLC0415
                            build_batch_marginal_plots,
                        )

                        figs = build_batch_marginal_plots(
                            self.batch_metrics,
                            atom_tokens=list(self.vocab.atom_tokens),
                        )
                        for name, fig in figs.items():
                            try:
                                self.logger.log_image(key=name, images=[fig])
                            except Exception as e:
                                print(f"Error logging {name}: {e}")
                        if figs:
                            import matplotlib.pyplot as plt  # noqa: PLC0415

                            for fig in figs.values():
                                plt.close(fig)
            except Exception as e:
                print(f"Error in batch-metric epoch-end logging: {e}")
            finally:
                self.batch_metrics.reset()

        # Distribution KLs are tensor-native and accumulate in every
        # representation; edge/charge KLs are only registered when meaningful
        # (see init_metrics), so an empty/absent collection just no-ops.
        if self.distribution_metrics is None or len(self.distribution_metrics) == 0:
            self._restore_training_rng()
            return

        try:
            # Sync once for both pooled KLs and their marginal plots.
            with ExitStack() as stack:
                for m in self.distribution_metrics.values():
                    stack.enter_context(m.sync_context())

                dist_results = self.distribution_metrics.compute()
                self.log_dict(
                    {
                        f"val/{key}": (v.detach() if isinstance(v, torch.Tensor) else v)
                        for key, v in dist_results.items()
                    },
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=False,
                )

                if self.trainer.is_global_zero and hasattr(self.logger, "log_image"):
                    figures = build_marginal_plots(self.distribution_metrics)
                    for name, fig in figures.items():
                        self.logger.log_image(key=f"val/marginals/{name}", images=[fig])
                    if figures:
                        import matplotlib.pyplot as plt  # noqa: PLC0415

                        for fig in figures.values():
                            plt.close(fig)
        except Exception as e:
            print(f"Error in validation-epoch-end distribution logging: {e}")
        finally:
            self.distribution_metrics.reset()
            self._restore_training_rng()

    def _restore_training_rng(self):
        """Keep deterministic validation sampling from changing training RNG."""
        torch.random.set_rng_state(self._training_rng_state)
        if self._training_cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(self._training_cuda_rng_states)
