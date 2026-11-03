"""Shared model construction for training and dataset-free inference."""

from copy import deepcopy
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig, OmegaConf

from morph.dataset.core.representation import Representation, make_uniform_type_prior
from morph.dataset.core.vocab import Distributions, Vocab, setup_token_weights
from morph.utils.metrics import init_metrics


@dataclass(frozen=True)
class ModelData:
    """Vocabulary, empirical statistics, and the representation-aware flow prior."""

    vocab: Vocab
    empirical: Distributions
    prior: Distributions
    representation: Representation
    allow_charged: bool


def prepare_model_data(
    cfg: DictConfig, vocab: Vocab, distributions: Distributions
) -> ModelData:
    """Resolve vocabulary config and separate empirical statistics from the prior."""
    OmegaConf.set_struct(cfg, False)
    cfg.data.vocab = vocab
    representation = Representation(cfg.representation)
    return ModelData(
        vocab=vocab,
        empirical=deepcopy(distributions),
        prior=make_uniform_type_prior(distributions, vocab, representation),
        representation=representation,
        allow_charged=bool(cfg.data.get("allow_charged", True)),
    )


def build_model_metrics(
    data: ModelData, *, train_smiles=None, train_smiles_are_canonical=False
) -> dict:
    """Build matching metric buffers for training and checkpoint loading.

    Dataset-free callers omit training SMILES; novelty has no persistent buffers.
    Distribution and geometry buffers must still match the training checkpoint.
    """
    distributions, vocab = data.empirical, data.vocab
    collections = init_metrics(
        train_smiles=train_smiles,
        target_n_atoms_distribution=distributions.n_atoms_distribution,
        atom_type_distribution=distributions.atom_type_distribution,
        edge_type_distribution=distributions.edge_type_distribution,
        charge_type_distribution=distributions.charge_type_distribution,
        atom_tokens=list(vocab.atom_tokens),
        edge_tokens=list(vocab.edge_tokens),
        charge_tokens=list(vocab.charge_tokens),
        allow_charged=data.allow_charged,
        distributions=distributions,
        representation=data.representation,
        train_smiles_are_canonical=train_smiles_are_canonical,
    )
    names = ("metrics", "stability_metrics", "distribution_metrics", "batch_metrics")
    return dict(zip(names, collections, strict=True))


def build_model_module(
    cfg: DictConfig, data: ModelData, *, metrics: dict | None = None
):
    """Construct the configured Lightning module from explicit data and metrics."""
    weighting = cfg.model.token_weighting
    atom_weights, edge_weights, charge_weights = setup_token_weights(
        vocab=data.vocab,
        distributions=data.empirical,
        weight_alpha=weighting.weight_alpha,
        type_loss_token_weights=weighting.type_loss_token_weights,
    )
    if metrics is None:
        metrics = build_model_metrics(data)
    return hydra.utils.instantiate(
        cfg.model.module,
        _recursive_=False,
        distributions=data.prior,
        loss_weight_distributions=data.empirical,
        atom_type_weights=atom_weights,
        edge_token_weights=edge_weights,
        charge_token_weights=charge_weights,
        allow_charged=data.allow_charged,
        representation=data.representation.value,
        **metrics,
    )
