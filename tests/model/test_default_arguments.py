"""Direct Python construction agrees with the GEOM Hydra training recipe."""

import inspect
from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from morph.flow_matching.integration import RateIntegrator
from morph.flow_matching.interpolation import Interpolator

from ._helpers import _noise_consistency_setup


@pytest.fixture
def geom_config():
    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[2] / "configs"), version_base="1.1"
    ):
        cfg = compose(config_name="default", overrides=["defaults=geom/geom"])
    cfg.data.vocab = {
        "atom_tokens": ["Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S"],
        "edge_tokens": ["<NO_BOND>", "1", "2", "3"],
        "charge_tokens": ["-2", "-1", "0", "1", "2"],
    }
    return cfg


MODEL_COMPONENTS = [
    "embeddings.atom_type_embedding",
    "embeddings.edge_type_embedding",
    "embeddings.time_embedding",
    "embeddings.node_count_embedding",
    "model.backbone.rbf_embedding_args",
    "model.backbone",
    "heads.heads",
    "heads.ins_edge_head",
    "insertion_head.model",
]


@pytest.mark.parametrize(
    "path",
    [
        *MODEL_COMPONENTS,
        "model.module",
        "model.module.optimizer_config.scheduler",
        "data.interpolator",
        "data.datamodule",
        "data.preprocessing",
        "integrator.integrator",
        "schedule",
    ],
)
def test_scalar_argument_defaults_match_geom_recipe(geom_config, path):
    config = OmegaConf.select(geom_config, path)
    target = hydra.utils.get_class(config._target_)
    for name, parameter in inspect.signature(target).parameters.items():
        default = parameter.default
        # None can request runtime construction or infer a size from another arg.
        if default is inspect.Parameter.empty or default is None or name not in config:
            continue
        expected = config[name]
        if not OmegaConf.is_config(expected):
            assert default == expected, f"{path}.{name}"


@pytest.mark.parametrize("path", MODEL_COMPONENTS)
def test_default_components_accept_geom_checkpoint_shapes(geom_config, path):
    config = OmegaConf.select(geom_config, path)
    target = hydra.utils.get_class(config._target_)
    required = {
        name: config[name]
        for name, parameter in inspect.signature(target).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    # Check the full default architecture without allocating large weight tensors.
    with torch.device("meta"):
        direct = target(**required)
        configured = hydra.utils.instantiate(config, _recursive_=False)
    direct.load_state_dict(configured.state_dict(), strict=True)


def test_default_flow_matches_configured_noise_units_and_schedules(geom_config):
    prior, _, reference = _noise_consistency_setup()
    direct_interpolator = Interpolator(prior)
    configured_interpolator = hydra.utils.instantiate(
        geom_config.data.interpolator, distributions=prior
    )
    direct_integrator = RateIntegrator(reference.vocab, prior, "unimodal", device="cpu")
    configured_integrator = hydra.utils.instantiate(
        geom_config.integrator.integrator,
        vocab=reference.vocab,
        distributions=prior,
        device="cpu",
    )
    times = torch.tensor([0.0, 0.1, 0.4, 0.8, 0.999])
    for direct, configured in (
        (direct_interpolator, configured_interpolator),
        (direct_integrator, configured_integrator),
    ):
        assert direct.ins_noise_scale == configured.ins_noise_scale
        assert direct.ins_noise_scale * prior.coordinate_std == pytest.approx(
            geom_config.data.ins_noise_scale
        )
        for name in ("del_schedule", "ins_schedule", "sub_schedule", "sub_e_schedule"):
            actual, expected = getattr(direct, name), getattr(configured, name)
            torch.testing.assert_close(actual.kappa_t(times), expected.kappa_t(times))
            torch.testing.assert_close(actual.rate(times), expected.rate(times))
    assert (
        direct_interpolator.move_noise_scale == configured_interpolator.move_noise_scale
    )
    for name in ("c_move", "c_sub", "c_ins", "c_del"):
        assert getattr(direct_interpolator, name) == getattr(
            configured_interpolator, name
        )
    torch.testing.assert_close(
        direct_integrator.get_time_steps(), configured_integrator.get_time_steps()
    )
