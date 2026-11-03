"""Config regression tests."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from morph.utils.run_config import build_trainer


@pytest.mark.parametrize(
    ("devices", "num_nodes", "child_rank", "expected_world_size"),
    [
        (1, 1, None, 1),
        (2, 1, None, 2),
        ("auto", 1, None, 1),
        (2, 2, None, 4),
        (2, 1, 1, 2),
    ],
)
def test_trainer_resolves_validation_topology_before_launch(
    monkeypatch, devices, num_nodes, child_rank, expected_world_size
):
    monkeypatch.setattr(rank_zero_only, "rank", 0)
    for key in ("WORLD_SIZE", "LOCAL_RANK", "NODE_RANK"):
        monkeypatch.delenv(key, raising=False)
    if child_rank is not None:
        monkeypatch.setenv("WORLD_SIZE", str(expected_world_size))
        monkeypatch.setenv("LOCAL_RANK", str(child_rank))
    trainer = build_trainer(
        accelerator="cpu",
        devices=devices,
        num_nodes=num_nodes,
        strategy="ddp_find_unused_parameters_false",
        plugins=[LightningEnvironment()],
        logger=False,
        enable_checkpointing=False,
        validation_batch_size=32,
        num_val_samples=4096,
    )
    assert trainer.world_size == expected_world_size
    assert trainer.limit_val_batches == 4096 // (32 * expected_world_size)
    assert trainer.strategy._ddp_kwargs == {
        "find_unused_parameters": False,
        "broadcast_buffers": False,
        "gradient_as_bucket_view": True,
    }


@pytest.mark.parametrize(
    ("overrides", "expected_limit"),
    [
        ({"limit_val_batches": 3}, 3),
        ({"limit_val_batches": 0}, 0),
        ({"limit_val_batches": 0.5}, 0.5),
        ({"fast_dev_run": True}, 1),
        ({"overfit_batches": 2}, 2),
    ],
)
def test_trainer_validation_overrides_take_precedence(overrides, expected_limit):
    trainer = build_trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        validation_batch_size=2,
        num_val_samples=7,  # Invalid unless the explicit/debug limit wins.
        **overrides,
    )
    assert trainer.limit_val_batches == expected_limit


def test_trainer_preserves_ddp_launch_preset_and_custom_strategy():
    options = {
        "accelerator": "cpu",
        "devices": 2,
        "logger": False,
        "enable_checkpointing": False,
        "validation_batch_size": 32,
    }
    trainer = build_trainer(strategy="ddp_spawn_find_unused_parameters_true", **options)
    assert trainer.strategy._start_method == "spawn"
    assert trainer.strategy._ddp_kwargs["find_unused_parameters"] is True

    custom = DDPStrategy(broadcast_buffers=True)
    trainer = build_trainer(strategy=custom, **options)
    assert trainer.strategy is custom
    assert trainer.strategy._ddp_kwargs["broadcast_buffers"] is True


def test_default_config_resolves_only_first_class_dit():
    config_dir = Path(__file__).parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        config = compose(config_name="default")

    assert config.model.module._target_.endswith("LightningModuleRates")
    assert config.model.module.model._target_.endswith("DiTBackboneWithHeads")
    assert config.model.backbone._target_.endswith("DiTBackbone")
    assert config.heads.heads._target_.endswith("MultiHeadModule")
    assert config.insertion_head.mode == "unimodal"
    assert config.insertion_head.num_components == 1
    assert config.insertion_head.model._target_.endswith("UnimodalInsertionHead")
    assert config.model.module.ema_warmup_steps == 5000
    assert config.model.lora.enabled is False
    assert config.data.interpolator.c_ins == 6.125
    assert config.data.interpolator.c_del == 6.125
    assert config.integrator.integrator.exact_schedule_integration is True
    assert config.trainer.trainer.num_val_samples == 4096
    assert config.logging.project == "morph-geom"
    assert "pair_bias_mode" not in config.model.backbone
    assert "pair_bias_all_layers" not in config.model.backbone
    assert "modern" not in OmegaConf.to_yaml(config.model)
    assert "V2" not in OmegaConf.to_yaml(config.model)


def test_gmm_insertion_head_is_an_explicit_config_choice():
    config_dir = Path(__file__).parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        config = compose(config_name="default", overrides=["insertion_head=gmm"])

    assert config.insertion_head.mode == "gmm"
    assert config.insertion_head.num_components == 10
    assert config.insertion_head.model._target_.endswith("GMMInsertionHead")


def test_lora_finetune_is_an_explicit_config_choice():
    config_dir = Path(__file__).parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        config = compose(
            config_name="default",
            overrides=[
                "model=finetune/lora_finetune",
                "model.module.pretrained_ckpt=/tmp/pretrained.ckpt",
            ],
        )

    assert config.model.lora.enabled is True
    assert config.model.module.freeze_pretrained_backbone is True
    assert config.model.module._target_.endswith("LightningModuleRatesFinetune")


def test_geom_uses_the_same_partial_ot_horizon():
    config_dir = Path(__file__).parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        config = compose(config_name="default", overrides=["defaults=geom/geom"])

    assert config.data.dataset_name == "geom"
    assert config.data.interpolator.c_move == 1.0
    assert config.data.interpolator.c_sub == 0.0
    assert config.data.interpolator.c_ins == 6.125
    assert config.data.interpolator.c_del == 6.125


def test_logging_can_be_disabled_without_wandb_configuration():
    config_dir = Path(__file__).parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        config = compose(config_name="default", overrides=["logging.enabled=false"])

    assert config.logging.enabled is False


@pytest.mark.parametrize("dataset", ["geom", "chembl3d"])
def test_hydrogen_mode_override_selects_separate_cache_and_all_splits(
    dataset, tmp_path, monkeypatch
):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    config_dir = Path(__file__).parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        explicit = compose(
            config_name="default", overrides=[f"defaults={dataset}/{dataset}"]
        )
        implicit = compose(
            config_name="default",
            overrides=[f"defaults={dataset}/{dataset}", "data.hydrogen_mode=implicit"],
        )
    assert explicit.data.preprocessing.root == str(tmp_path / "data" / dataset)
    assert implicit.data.preprocessing.root == str(
        tmp_path / "data" / f"{dataset}_implicit_h"
    )
    for cfg in (
        implicit.data.datamodule.datasets.train,
        *implicit.data.datamodule.datasets.val,
        *implicit.data.datamodule.datasets.test,
    ):
        assert cfg.hydrogen_mode == "implicit"
        assert cfg.root == implicit.data.preprocessing.root
