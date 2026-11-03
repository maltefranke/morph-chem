"""Absolute-logP conditioning through the shared zero-gated graph port."""

from torch import nn

from morph.model.backbones.conditional_dit import ConditionalDiTBackboneWithHeads
from morph.model.finetune.base import LightningModuleRatesFinetune
from morph.utils.metrics.logp import LogPMetrics


class LogPDiTBackboneWithHeads(ConditionalDiTBackboneWithHeads):
    """Embed one absolute logP per graph without changing the clean DiT lane."""

    def __init__(self, *args, logp_hidden_dim=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.logp_embedding = nn.Sequential(
            nn.Linear(1, logp_hidden_dim),
            nn.SiLU(),
            nn.Linear(logp_hidden_dim, int(kwargs["backbone_model_args"]["d_model"])),
        )

    def forward(self, mols_t, t, max_seqlen=None):
        """Reuse the absolute request after every insertion and deletion."""
        requested = getattr(mols_t, "target_logp", None)
        if requested is None or requested.shape != (mols_t.num_graphs,):
            raise ValueError("target_logp must contain one scalar per graph")
        return super().forward(
            mols_t,
            t,
            max_seqlen=max_seqlen,
            graph_cond=self.logp_embedding(requested.float().unsqueeze(-1)),
        )

    def freeze_pretrained(self):
        """Allow optional adapter training while retaining the logP encoder."""
        before = sum(p.numel() for p in self.parameters() if p.requires_grad)
        super().freeze_pretrained()
        self.logp_embedding.requires_grad_(True)
        return before - sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightningModuleRatesLogPFinetune(LightningModuleRatesFinetune):
    """Track requested versus generated logP over the complete validation set."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logp_metrics = LogPMetrics()

    def on_validation_epoch_start(self):
        """Start independent property and chemistry accumulators."""
        super().on_validation_epoch_start()
        self.logp_metrics.reset()

    def _validation_context(self, batch):
        return batch[0].target_logp.detach().cpu()

    def _update_task_metrics(self, context, mols, rdkit_mols):
        self.logp_metrics.update(context, rdkit_mols)

    def on_validation_epoch_end(self):
        """Reduce paired properties once and log a parity figure on rank zero."""
        figure = None
        with self.logp_metrics.sync_context():
            if self.trainer.is_global_zero and hasattr(self.logger, "log_image"):
                figure = self.logp_metrics.build_parity_plot()
            # Metric.compute() may restore local state on return, so render
            # from the synchronized pairs before computing the scalar values.
            values = self.logp_metrics.compute()
        self.log_dict(
            {f"val/{name}": value for name, value in values.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        if figure is not None:
            from matplotlib import pyplot as plt  # noqa: PLC0415

            try:
                self.logger.log_image(key="val/logp_parity", images=[figure])
            finally:
                plt.close(figure)
        self.logp_metrics.reset()
        super().on_validation_epoch_end()
