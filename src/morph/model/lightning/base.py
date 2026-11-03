import copy

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from morph.dataset.core.representation import Representation
from morph.dataset.core.vocab import Distributions, Vocab
from morph.model.lightning._optimizer import OptimizerMixin
from morph.model.lightning._sampling import SamplingMixin
from morph.model.lightning._setup import SetupMixin
from morph.model.lightning._training import TrainingMixin
from morph.model.lightning._validation import ValidationMixin
from morph.utils.edge_alignment import EDGE_ALIGNER
from morph.utils.loss_accumulation import (
    InverseSquaredTimeLossWeighting,
    LossAccumulator,
    UnifiedWeightedLoss,
)
from morph.utils.lr_schedulers import EMADecayScheduler
from morph.utils.metrics import MetricCollection


class LightningModuleRates(
    SetupMixin,
    OptimizerMixin,
    TrainingMixin,
    ValidationMixin,
    SamplingMixin,
    pl.LightningModule,
):
    """Train endpoint predictions and edit posteriors for molecular flow matching.

    Deletion and substitution heads predict binary logits, supervised by the
    positivity of numerical lambda targets. The insertion-count head predicts
    the number of pending atoms per spawning node using Poisson NLL. A unimodal
    or mixture head predicts their clean positions, atom types, and charges.
    Insertion-edge training uses target inserted-atom features; sampling uses
    predicted features before applying insertion noise.

    Position and charge heads predict the clean endpoint. The integrator combines
    edit posteriors with schedule masses to advance the molecular state.
    See ``src/morph/model/README.md`` for tensor and loss contracts.
    """

    def training_step(self, batch, batch_idx):
        """Dispatch explicitly to automatic or manual optimizer stepping."""
        if self.automatic_optimization:
            return self._automatic_training_step(batch, batch_idx)
        return self._manual_training_step(batch, batch_idx)

    LOSS_GROUPS: dict[str, list[str]] = {
        "sub": ["do_sub_a", "sub_a_class", "do_sub_e", "sub_e_class"],
        "del": ["do_del"],
        "ins": ["ins_rate", "ins_state", "ins_e", "ins_e_ii"],
        "x": ["x"],
        "c": ["c"],
    }

    def __init__(
        self,
        model: DictConfig,
        integrator: DictConfig,
        loss_weights: DictConfig,
        optimizer_config: DictConfig,
        insertion_head_mode: str,
        vocab: Vocab,
        distributions: Distributions,
        loss_weight_distributions: Distributions,
        atom_type_weights: torch.Tensor,
        edge_token_weights: torch.Tensor,
        charge_token_weights: torch.Tensor,
        time_dist: DictConfig,
        metrics: MetricCollection,
        stability_metrics: MetricCollection,
        distribution_metrics: MetricCollection | None = None,
        batch_metrics: MetricCollection | None = None,
        ins_noise_scale: float = 0.65,
        use_learnable_loss_weights: bool = False,
        ema_decay: float = 0.999,
        ema_warmup_steps: int = 5000,
        ema_decay_scheduler: EMADecayScheduler | DictConfig | None = None,
        use_ema_for_eval: bool = True,
        use_time_weights: bool = False,
        allow_charged: bool = True,
        log_grad_norms_every_n_steps: int = 0,
        grad_norm_log_interval: int = 500,
        grad_norms_granularity: str = "component",
        representation: str | Representation = Representation.MOLECULE,
    ):
        super().__init__()

        self.model = hydra.utils.instantiate(model)

        # vocab and distributions
        self.vocab = vocab
        self.distributions = distributions
        self.loss_weight_distributions = loss_weight_distributions

        self.ins_noise_scale = ins_noise_scale

        # Whether charged species are considered chemically valid for the dataset
        # (e.g. QM9 contains only neutral molecules; GEOM contains charged ones).
        self.allow_charged = allow_charged

        # Controls mode-aware losses, validation metrics, and prediction cleanup.
        self.representation = Representation(representation)

        # setup ema scheduler
        self.ema_decay = float(ema_decay)
        self.ema_warmup_steps = max(0, int(ema_warmup_steps))
        if isinstance(ema_decay_scheduler, DictConfig):
            ema_decay_scheduler = hydra.utils.instantiate(ema_decay_scheduler)
        self.ema_decay_scheduler = ema_decay_scheduler
        self.use_ema_for_eval = use_ema_for_eval

        if insertion_head_mode not in {"unimodal", "gmm"}:
            raise ValueError(
                "insertion_head_mode must be 'unimodal' or 'gmm', got "
                f"{insertion_head_mode!r}"
            )
        self.insertion_head_mode = insertion_head_mode
        self.time_dist = hydra.utils.instantiate(time_dist)

        self.integrator = hydra.utils.instantiate(
            integrator, distributions=self.distributions
        )

        # weights for class losses
        self.register_buffer("atom_type_weights", atom_type_weights)
        self.register_buffer("edge_token_weights", edge_token_weights)
        self.register_buffer("charge_token_weights", charge_token_weights)

        # metrics tracking for validation
        self.metrics = metrics
        self.stability_metrics = stability_metrics
        # Distribution metrics accumulate over the entire validation epoch so we
        # can log a single pooled KL and render marginal plots at epoch end.
        self.distribution_metrics = distribution_metrics
        # Tensor-based metrics for non-topology modes. Updated in validation_step
        # and computed/logged in on_validation_epoch_end.
        self.batch_metrics = batch_metrics

        # Exponential moving average of model parameters for stable inference
        self.model_ema = copy.deepcopy(self.model)
        for p in self.model_ema.parameters():
            p.requires_grad = False
        self.model_ema.eval()

        # handling of edge utilities, especially for upper-triangular handling
        self.edge_aligner = EDGE_ALIGNER

        # set up loss weighting for individual loss components
        loss_weight_values = {k: float(v) for k, v in loss_weights.items()}
        self.loss_weight_wrapper = UnifiedWeightedLoss(
            manual_weights=loss_weight_values,
            component_keys=list(loss_weight_values.keys()),
            use_learnable=use_learnable_loss_weights,
        )

        # optionally weigh by interpolation time
        if use_time_weights:
            time_weights = {
                "x": InverseSquaredTimeLossWeighting(clamp_max=100.0),
                "c": lambda t: self.integrator.sub_schedule.rate(t).clamp(max=100.0),
                "ins": lambda t: self.integrator.ins_schedule.rate(t).clamp(max=100.0),
                "del": lambda t: self.integrator.del_schedule.rate(t).clamp(max=100.0),
                "sub": lambda t: self.integrator.sub_schedule.rate(t).clamp(max=100.0),
            }
        else:
            # always use time-weighting for x and c
            time_weights = {
                "x": InverseSquaredTimeLossWeighting(clamp_max=100.0),
                "c": lambda t: self.integrator.sub_schedule.rate(t).clamp(max=100.0),
            }

        # object to tie loss computation, weighting, and logging together
        self.loss_accumulator = LossAccumulator(
            self.loss_weight_wrapper, self.LOSS_GROUPS, time_weights
        )

        self.optimizer_config = optimizer_config
        self._setup_optimization()

        if hasattr(self.model, "initialize_output_priors"):
            # Dataset categorical marginals initialize the DiT prediction
            # heads; keep the EMA copy exactly synchronized with that model.
            self.model.initialize_output_priors(self.loss_weight_distributions)
            self.model_ema.load_state_dict(self.model.state_dict())

        # Per-loss gradient-norm logging. Each invocation runs one extra
        # backward per reported term, so this is rate-limited via a step
        # interval and is off by default.
        self.log_grad_norms_every_n_steps = int(log_grad_norms_every_n_steps)
        # Cadence for the cheap global + per-group grad-norm logging in
        # on_before_optimizer_step (two full per-parameter .norm() passes). Pure
        # diagnostics (gradient clipping is separate and always runs), so log them
        # every N steps rather than every step. <=0 disables. global_step is
        # DDP-synchronized, so every rank logs the same keys on the same steps.
        self.grad_norm_log_interval = int(grad_norm_log_interval)
        if grad_norms_granularity not in {"component", "group", "both"}:
            raise ValueError(
                "grad_norms_granularity must be one of component/group/both, "
                f"got {grad_norms_granularity!r}"
            )
        self.grad_norms_granularity = grad_norms_granularity

        self.is_compiled = False

        # lastly, save hyperparameters
        self.save_hyperparameters(
            ignore=[
                "ema_decay_scheduler",
                "metrics",
                "stability_metrics",
                "distribution_metrics",
                "batch_metrics",
            ]
        )
