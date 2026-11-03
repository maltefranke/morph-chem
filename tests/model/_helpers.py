"""Small deterministic fixtures shared by model regression tests."""

import torch
from omegaconf import OmegaConf

from morph.dataset.core.representation import Representation, make_uniform_type_prior
from morph.dataset.core.vocab import Distributions, Vocab
from morph.flow_matching.integration import RateIntegrator
from morph.flow_matching.interpolation import Interpolator
from morph.model.backbones.dit import (
    DiTBackbone,
)


def _rbf_args():
    return OmegaConf.create(
        {
            "_target_": "morph.model.layers.embedding.RBFEmbedding",
            "num_rbf": 8,
            "rbf_dmax": 4.0,
            "out_dim": 8,
            "dropout": 0.0,
            "trainable": False,
        }
    )


def _complete_edges(num_nodes):
    rows, cols = torch.where(~torch.eye(num_nodes, dtype=torch.bool))
    return torch.stack([rows, cols])


def _backbone(num_layers=4, *, pair_bias_all_layers=None):
    return DiTBackbone(
        d_input=7,
        d_cond=6,
        d_model=32,
        out_node_nf=16,
        out_edge_nf=10,
        num_layers=num_layers,
        nhead=4,
        in_edge_nf=5,
        rbf_embedding_args=_rbf_args(),
        pair_bias_hidden_dim=8,
        pair_bias_all_layers=pair_bias_all_layers,
        ffn_multiple_of=8,
        edge_pair_dim=12,
        num_atom_types=6,
    )


def _native_optimizer_config():
    return OmegaConf.create(
        {
            "implementation": "torch",
            "optimizer": {
                "lr": 0.01,
                "momentum": 0.95,
                "weight_decay": 0.0,
                "nesterov": True,
                "ns_steps": 5,
                "eps": 1e-7,
                "adjust_lr_fn": "original",
            },
            "aux_optimizer": {
                "lr": 3e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
            "scheduler": {
                "_target_": "morph.utils.lr_schedulers.MultiGroupCosineWarmupLR",
                "warmup_steps": 2,
                "total_steps": 10,
                "min_lr_fraction": 0.05,
                "last_epoch": -1,
            },
            "interval": "step",
            "monitor": None,
        }
    )


def _noise_consistency_setup():
    vocab = Vocab(
        atom_tokens=["C", "N", "O"],
        edge_tokens=["<NO_BOND>", "1"],
        charge_tokens=["-1", "0", "1"],
    )
    empirical = Distributions(
        atom_type_distribution=torch.tensor([0.8, 0.15, 0.05]),
        edge_type_distribution=torch.tensor([0.95, 0.05]),
        charge_type_distribution=torch.tensor([0.1, 0.8, 0.1]),
        n_atoms_distribution=torch.tensor([0.0, 0.0, 1.0]),
        coordinate_std=2.0,
    )
    prior = make_uniform_type_prior(empirical, vocab, Representation.MOLECULE)
    interpolator = Interpolator(
        distributions=prior,
        move_noise_scale=0.6,
        ins_noise_scale=0.8,
    )
    integrator = RateIntegrator(
        vocab=vocab,
        distributions=prior,
        insertion_head_mode="unimodal",
        ins_noise_scale=0.8,
        device="cpu",
    )
    return prior, interpolator, integrator


def _layer_bias_backbone(mode=None):
    return DiTBackbone(
        d_input=7,
        d_cond=6,
        d_model=32,
        out_node_nf=16,
        out_edge_nf=10,
        num_layers=3,
        nhead=4,
        in_edge_nf=5,
        pair_bias_hidden_dim=8,
        pair_bias_mode=mode,
        ffn_multiple_of=8,
        edge_pair_dim=12,
        num_atom_types=6,
        rbf_embedding_args=OmegaConf.create(
            dict(
                _target_="morph.model.layers.embedding.RBFEmbedding",
                num_rbf=8,
                rbf_dmax=4.0,
                out_dim=8,
                dropout=0.0,
                trainable=False,
            )
        ),
    )
