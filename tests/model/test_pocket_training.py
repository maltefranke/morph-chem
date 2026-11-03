"""Checkpoint filtering, replay sampling, geometric context, and training contracts."""

from pathlib import Path

import hydra
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from morph.dataset.conditional.pocket.datamodule import (
    PocketFlowMatchingWrapper,
    pocket_train_collate_fn,
)
from morph.dataset.conditional.replay import ReplaySampler
from morph.dataset.core.flow_matching_wrapper import FlowMatchingDatasetWrapper
from morph.dataset.core.vocab import Distributions
from morph.flow_matching.interpolation import Interpolator
from morph.model.build import build_model_module, prepare_model_data
from tests.model._helpers import _native_optimizer_config
from tests.model.test_crossdocked import _tokenized
from tests.model.test_model_build import _setup

pytest_plugins = ["tests.model.test_crossdocked"]


@pytest.mark.parametrize("fraction", [0.0, 0.25, 1.0])
def test_replay_probability_does_not_depend_on_dataset_sizes(fraction):
    sampler = ReplaySampler(3, 100_000, fraction, 20_000, seed=81)
    indices = torch.tensor(list(sampler))
    assert len(indices) == 20_000
    assert abs(float((indices < 3).float().mean()) - fraction) < 0.015
    assert 0 <= indices.min() <= indices.max() < 100_003
    assert indices.tolist() == list(sampler)
    sampler.set_epoch(1)
    assert indices.tolist() != list(sampler)


def test_distributed_replay_partitions_one_reproducible_epoch_stream():
    single = list(ReplaySampler(7, 21, 0.3, 64, seed=15))
    ranks = [
        ReplaySampler(7, 21, 0.3, 64, seed=15, num_replicas=2, rank=r) for r in range(2)
    ]
    assert list(ranks[0]) == single[::2]
    assert list(ranks[1]) == single[1::2]
    for sampler in ranks:
        sampler.set_epoch(3)
    assert list(ranks[0]) != single[::2]
    with pytest.raises(ValueError, match="compatible"):
        ReplaySampler(0, 20, 0.1, 64)
    with pytest.raises(ValueError, match="pretraining"):
        ReplaySampler(7, 0, 0.5, 64)
    assert len(list(ReplaySampler(0, 20, 0.0, 64))) == 64
    assert len(list(ReplaySampler(7, 0, 1.0, 64))) == 64


def _wrapper_options(dataset, **overrides):
    distributions = Distributions(
        torch.ones(3) / 3,
        torch.ones(3) / 3,
        torch.ones(3) / 3,
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        2.5,
    )
    options = dict(
        base_dataset=dataset,
        distributions=distributions,
        vocab=dataset.vocab,
        interpolator=Interpolator(
            distributions,
            optimal_transport="none",
            pre_align_iters=0,
            c_ins=1e6,
            c_del=1e6,
            move_noise_scale=0.1,
            ins_noise_scale=0.2,
        ),
        time_dist=torch.distributions.Uniform(0.2, 0.8),
        n_atoms_strategy="approx",
        stage="train",
        rotate=False,
    )
    options.update(overrides)
    return options


def test_pocket_translation_rotation_and_augmentation_share_the_ligand_frame(cache):
    dataset = _tokenized(cache, max_pocket_atoms=3)
    try:
        torch.manual_seed(25)
        plain = PocketFlowMatchingWrapper(**_wrapper_options(dataset))[0]
        expected = dataset.get_pocket(0).x - plain[0].frame_shift
        torch.testing.assert_close(plain[0].contexts["pocket"].x, expected)
        distances = torch.cdist(plain[0].x, expected)
        torch.manual_seed(25)
        np.random.seed(35)
        augmented = PocketFlowMatchingWrapper(
            **_wrapper_options(dataset, rotate=True, n_augmentations=3)
        )[0]
        for sample in augmented:
            torch.testing.assert_close(
                torch.cdist(sample[0].x, sample[0].contexts["pocket"].x),
                distances,
                atol=1e-5,
                rtol=1e-5,
            )
        assert (
            len({sample[0].contexts["pocket"].x.data_ptr() for sample in augmented})
            == 3
        )
        assert not torch.allclose(
            augmented[0][0].contexts["pocket"].x, augmented[1][0].contexts["pocket"].x
        )
        torch.testing.assert_close(
            dataset.get_pocket(0).x, expected + plain[0].frame_shift
        )
    finally:
        dataset.close()


@pytest.mark.parametrize("conditioned", [(True, False), (False, False), (True, True)])
def test_mixed_pocket_training_loads_pretrained_weights_and_backpropagates(
    cache, tmp_path, conditioned
):
    dataset = _tokenized(cache, max_pocket_atoms=3)
    options = _wrapper_options(dataset)
    fine = PocketFlowMatchingWrapper(**options)
    pretrain = FlowMatchingDatasetWrapper(**options)
    try:
        batch = pocket_train_collate_fn(
            [
                fine[i] if with_context else pretrain[i]
                for i, with_context in enumerate(conditioned)
            ]
        )
        assert batch[0].contexts["pocket"].num_nodes_per_graph.tolist() == [
            3 if c else 0 for c in conditioned
        ]
        assert batch[0].contexts["pocket"].num_graphs == 2
        cfg, _ = _setup()
        data = prepare_model_data(cfg, dataset.vocab, options["distributions"])
        pretrained = build_model_module(cfg, data).eval()
        checkpoint = tmp_path / "pretrained.ckpt"
        torch.save({"state_dict": pretrained.state_dict()}, checkpoint)
        cfg.model.module._target_ = (
            "morph.model.finetune.base.LightningModuleRatesFinetune"
        )
        cfg.model.module.pretrained_ckpt = str(checkpoint)
        cfg.model.module.pretrained_prefer_ema = False
        cfg.model.module.model._target_ = (
            "morph.model.finetune.pocket.PocketConditionalDiTBackboneWithHeads"
        )
        cfg.model.backbone._target_ = (
            "morph.model.backbones.conditional_dit.ConditionalDiTBackbone"
        )
        cfg.model.backbone.ctx_dim = 16
        cfg.model.backbone.use_cross_attn = True
        cfg.model.module.model.encoder_args = dict(
            _target_="morph.model.finetune.pocket.PocketContextEncoder",
            num_atom_types=3,
            num_charges=3,
            ctx_dim=16,
            num_attn_layers=1,
            nhead=4,
        )
        module = build_model_module(cfg, data).eval()
        module.log_dict = lambda *args, **kwargs: None
        with torch.no_grad():
            reference = pretrained.model(batch[0], batch[3].view(-1, 1))
            initial = module.model(batch[0], batch[3].view(-1, 1))
        for key in ("pos_head", "atom_type_head", "edge_type_head", "charge_head"):
            torch.testing.assert_close(initial[key], reference[key])
        # With trained, nonzero gates, unconditional rows must still bypass
        # both the attention and the adapter FFN residuals exactly.
        with torch.no_grad():
            for block in module.model.backbone.cross_attn_blocks:
                block.attn_gate.fill_(0.4)
                block.ffn_gate.fill_(0.3)
            mixed = module.model(batch[0], batch[3].view(-1, 1))
            unconditional = module.model(
                batch[0], batch[3].view(-1, 1), force_uncond=True
            )
            without_context = batch[0].clone()
            without_context.contexts.clear()
            reference_unconditional = module.model(
                without_context, batch[3].view(-1, 1)
            )
        # The explicit flag bypasses pocket encoding for every graph, including
        # conditioned rows, without needing state attached to the molecular batch.
        torch.testing.assert_close(
            unconditional["h_latent"], reference_unconditional["h_latent"]
        )
        rows = torch.tensor(conditioned)[batch[0].batch]
        torch.testing.assert_close(
            mixed["h_latent"][~rows], unconditional["h_latent"][~rows]
        )
        loss = module.shared_step(batch, 0)
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in module.model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
    finally:
        dataset.close()


def test_pocket_preset_builds_replay_loader_and_keeps_evaluation_separate(
    cache, monkeypatch, tmp_path
):
    monkeypatch.setenv("PROJECT_ROOT", str(cache.parent))
    monkeypatch.setenv("MORPH_PRETRAINED_ASSETS", str(cache / "artifacts"))
    monkeypatch.setenv("MORPH_PRETRAINED_CKPT", "/tmp/model.ckpt")
    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[2] / "configs"), version_base="1.1"
    ):
        cfg = compose(
            config_name="default",
            overrides=[
                "defaults=crossdocked/finetune",
                f"data.root={cache}",
                "data.val_fraction=0.25",
                "data.preprocess_workers=0",
                "data.finetune_fraction=0.25",
                "data.train_epoch_size=32",
                "data.datamodule.batch_size.train=8",
                "data.datamodule.num_workers.train=0",
            ],
        )
    dataset = _tokenized(cache)
    options = _wrapper_options(dataset)
    data = prepare_model_data(cfg, dataset.vocab, options["distributions"])
    # Use another raw reader as a small local stand-in for the ChEMBL3D source.
    cfg.data.datamodule.pretrain_dataset = OmegaConf.create(
        {
            "_target_": "morph.dataset.conditional.pocket.CrossDockedPocketDataset",
            "root": str(cache),
            "val_fraction": 0.25,
        }
    )
    dm = hydra.utils.instantiate(
        cfg.data.datamodule,
        _recursive_=False,
        vocab=data.vocab,
        distributions=data.prior,
    )
    try:
        assert cfg.data.preprocessing.train_dataset is None
        assert cfg.trainer.trainer.use_distributed_sampler is False
        dm.setup("fit")
        dm.configure_model(cfg)
        loader = dm.train_dataloader()
        assert len(loader) == 4
        assert loader.sampler.finetune_fraction == 0.25
        assert cfg.data.pocket_vocab.atom_tokens == ["C", "N", "O"]
        flags = []
        for batch in loader:
            flags.extend((batch[0].contexts["pocket"].num_nodes_per_graph > 0).tolist())
        assert any(flags)
        assert not all(flags)
        assert all(ds.stage == "val" for ds in dm.val_datasets)
        assert dm.train_dataset.pretrain.base_dataset.split == "train"
        dm.setup("fit")
        assert len(dm.train_dataloader()) == 4
        # Exercise the actual preset through Lightning's optimizer/loader path.
        # Checkpoint loading is covered independently above; use a tiny model here.
        cfg.model.module.pretrained_ckpt = None
        cfg.model.module.optimizer_config = _native_optimizer_config()
        for field, value in dict(
            d_model=32,
            num_layers=1,
            nhead=4,
            out_node_nf=16,
            out_edge_nf=16,
            ffn_multiple_of=8,
            ctx_dim=16,
        ).items():
            cfg.model.backbone[field] = value
        for field in (
            "head_node_adapter_dim",
            "head_edge_adapter_dim",
            "head_insertion_adapter_dim",
            "head_insertion_edge_dim",
        ):
            cfg.model[field] = 16
        cfg.model.pocket_encoder.nhead = 4
        cfg.model.pocket_encoder.num_attn_layers = 1
        module = build_model_module(cfg, data)
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=2,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            use_distributed_sampler=False,
            default_root_dir=tmp_path,
        )
        trainer.fit(module, datamodule=dm)
        assert trainer.global_step == 2
    finally:
        dataset.close()
