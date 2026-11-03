"""Absolute logP labels, pretrained compatibility, gradients, and metrics."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Crippen

from morph.dataset.conditional.logp import LogPFlowMatchingWrapper
from morph.dataset.conditional.logp.preprocessing import build_property_array
from morph.dataset.core.flow_matching_wrapper import eval_collate_fn, train_collate_fn
from morph.flow_matching.interpolation import Interpolator
from morph.model.build import build_model_module
from morph.model.finetune.base import LightningModuleRatesFinetune
from morph.model.finetune.logp import LightningModuleRatesLogPFinetune
from morph.utils.metrics.logp import LogPMetrics

from .test_atom_count import _Targets
from .test_model_build import _setup


class _LogPTargets(_Targets):
    def __getitem__(self, index):
        target = super().__getitem__(index)
        target.add_system_property("logp", torch.tensor([float(index) - 1.5]))
        return target


def _wrapper(data, **kwargs):
    wrapper = LogPFlowMatchingWrapper(
        _LogPTargets(),
        data.prior,
        Interpolator(data.prior, optimal_transport="none", c_ins=1e6, c_del=1e6),
        data.vocab,
        **kwargs,
    )
    wrapper._get_n_atoms = lambda target: 3
    return wrapper


@pytest.mark.parametrize("stage", ["train", "val", "test"])
@pytest.mark.parametrize("augmentations", [1, 2])
def test_absolute_request_survives_interpolation_and_augmentation(stage, augmentations):
    _, data = _setup()
    wrapper = _wrapper(data, stage=stage, n_augmentations=augmentations)
    collate = train_collate_fn if stage == "train" else eval_collate_fn
    state = collate([wrapper[0], wrapper[1]])[0]
    expected = torch.tensor([-1.5, -0.5])
    if stage == "train":
        expected = expected.repeat_interleave(augmentations)
    torch.testing.assert_close(state.target_logp, expected)
    # Requests are absolute properties, independent of the current atom count.
    assert not torch.equal(state.target_logp, state.num_nodes_per_graph.float())
    torch.testing.assert_close(state.clone().target_logp, expected)


def test_pretrained_equivalence_and_unfrozen_gradients(tmp_path):
    cfg, data = _setup()
    baseline = build_model_module(cfg, data).eval()
    with torch.no_grad():
        for name, parameter in baseline.model.named_parameters():
            if "adaLN_modulation" in name and name.endswith("weight"):
                parameter.normal_(std=0.02)
    checkpoint = tmp_path / "unconditional.ckpt"
    torch.save({"state_dict": baseline.state_dict()}, checkpoint)
    cfg.model.module._target_ = (
        "morph.model.finetune.logp.LightningModuleRatesLogPFinetune"
    )
    cfg.model.module.model._target_ = (
        "morph.model.finetune.logp.LogPDiTBackboneWithHeads"
    )
    cfg.model.backbone._target_ = (
        "morph.model.backbones.conditional_dit.ConditionalDiTBackbone"
    )
    cfg.model.backbone.use_graph_cond = True
    cfg.model.module.pretrained_ckpt = str(checkpoint)
    cfg.model.module.pretrained_prefer_ema = False
    cfg.model.module.freeze_pretrained_backbone = False
    cfg.model.module.freeze_output_heads = False
    adapted = build_model_module(cfg, data).eval()
    batch = train_collate_fn([_wrapper(data, stage="train")[0]])
    state, _, _, times = batch
    torch.testing.assert_close(
        baseline.model(state, times[:, None]),
        adapted.model(state, times[:, None]),
    )
    assert all(p.requires_grad for p in adapted.model.parameters())
    assert all(not p.requires_grad for p in adapted.model_ema.parameters())
    adapted.log_dict = lambda *args, **kwargs: None
    loss = adapted.shared_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in adapted.model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert adapted.model.backbone.graph_gate.grad.abs().sum() > 0

    # Once the gate opens, the encoder and request affect the shared condition.
    adapted.model.backbone.graph_gate.data.fill_(0.5)
    conditions = []
    handle = adapted.model.backbone.dit.blocks[0].register_forward_pre_hook(
        lambda module, args: conditions.append(args[1].detach().clone())
    )
    changed = state.clone()
    changed.target_logp += 2
    adapted.model(state, times[:, None])
    adapted.model(changed, times[:, None])
    handle.remove()
    assert not torch.equal(conditions[0], conditions[1])
    adapted.model.zero_grad()
    adapted.shared_step(batch, 0).backward()
    assert adapted.model.logp_embedding[0].weight.grad.abs().sum() > 0


def test_metrics_weight_molecules_and_count_failed_outputs():
    metric = LogPMetrics()
    molecule = Chem.MolFromSmiles("CCO")
    value = Crippen.MolLogP(molecule)
    metric.update(torch.tensor([value + 1, 0]), [molecule, None])
    metric.update(torch.tensor([value + 3]), [molecule])
    result = metric.compute()
    torch.testing.assert_close(result["logp_mae"], torch.tensor(2.0))
    torch.testing.assert_close(result["logp_rmse"], torch.tensor(5.0).sqrt())
    torch.testing.assert_close(result["logp_scored_fraction"], torch.tensor(2 / 3))
    metric.reset()
    metric.update(torch.tensor([0.0]), [None])
    assert torch.isnan(metric.compute()["logp_mae"])
    assert metric.compute()["logp_scored_fraction"] == 0


def test_parity_pairs_match_requests_and_reset_with_the_epoch():
    metric = LogPMetrics()
    assert not metric.sync_on_compute
    molecule = Chem.MolFromSmiles("CCO")
    actual = Crippen.MolLogP(molecule)
    metric.update(torch.tensor([1.5, 9.0]), [molecule, None])
    metric.update(torch.tensor([-2.0]), [molecule])
    torch.testing.assert_close(
        metric.pairs, torch.tensor([[1.5, actual], [-2.0, actual]])
    )
    assert metric.pairs.shape[0] == metric.scored
    metric.reset()
    assert metric.pairs.shape == (0, 2)
    metric.update(torch.tensor([0.0]), [None])
    assert metric.pairs.shape == (0, 2)


@pytest.mark.parametrize("local_requests", [[], [1.5]])
@pytest.mark.parametrize("other_requests", [[], [-2.0, 3.0]])
def test_parity_sync_concatenates_unequal_populations(local_requests, other_requests):
    metric, other = LogPMetrics(), LogPMetrics()
    molecule = Chem.MolFromSmiles("CCO")
    for state, requests in [(metric, local_requests), (other, other_requests)]:
        state.update(
            torch.tensor([*requests, 99.0]), [*[molecule] * len(requests), None]
        )
    other_scalars = {
        id(getattr(metric, name)): getattr(other, name)
        for name in ("absolute_error", "squared_error", "scored", "attempts")
    }

    def gather(tensor, group=None):
        return [tensor, other.pairs if tensor.ndim == 2 else other_scalars[id(tensor)]]

    with metric.sync_context(dist_sync_fn=gather, distributed_available=lambda: True):
        actual = Crippen.MolLogP(molecule)
        expected = torch.tensor(
            [[value, actual] for value in [*local_requests, *other_requests]]
        ).reshape(-1, 2)
        torch.testing.assert_close(metric.pairs, expected)
        assert metric.scored == len(expected)
    assert metric.pairs.shape == (len(local_requests), 2)
    metric.update(torch.tensor([0.0]), [None])
    assert metric.pairs.shape == (len(local_requests), 2)


@pytest.mark.parametrize("scorable", [True, False])
def test_parity_plot_axes_points_and_identity_reference(scorable):
    metric = LogPMetrics()
    molecule = Chem.MolFromSmiles("CCO") if scorable else None
    metric.update(torch.tensor([1.5]), [molecule])
    with metric.sync_context():
        figure = metric.build_parity_plot()
    try:
        axis = figure.axes[0]
        assert axis.get_xlabel() == "Requested logP"
        assert axis.get_ylabel() == "Generated logP"
        assert axis.get_xlim() == axis.get_ylim()
        points = axis.collections[0].get_offsets()
        assert points.shape == (int(scorable), 2)
        if scorable:
            np.testing.assert_allclose(points[0], [1.5, Crippen.MolLogP(molecule)])
        reference = axis.lines[0]
        np.testing.assert_array_equal(reference.get_xdata(), reference.get_ydata())
        assert f"{int(scorable)}/1 scorable" in axis.get_title()
    finally:
        plt.close(figure)


@pytest.mark.parametrize("rank_zero", [True, False])
def test_parity_logged_only_on_rank_zero(tmp_path, monkeypatch, rank_zero):
    cfg, data = _setup()
    cfg.model.module._target_ = (
        "morph.model.finetune.logp.LightningModuleRatesLogPFinetune"
    )
    cfg.model.module.model._target_ = (
        "morph.model.finetune.logp.LogPDiTBackboneWithHeads"
    )
    cfg.model.backbone._target_ = (
        "morph.model.backbones.conditional_dit.ConditionalDiTBackbone"
    )
    cfg.model.backbone.use_graph_cond = True
    module = build_model_module(cfg, data)
    assert isinstance(module, LightningModuleRatesLogPFinetune)
    images, scalars, base_calls = [], {}, []

    def log_image(**kwargs):
        assert kwargs["key"] == "val/logp_parity"
        figure = kwargs["images"][0]
        assert plt.fignum_exists(figure.number)
        figure.savefig(tmp_path / "logp-parity.png")
        images.append(figure)

    module._trainer = SimpleNamespace(
        is_global_zero=rank_zero, logger=SimpleNamespace(log_image=log_image)
    )
    module.log_dict = lambda values, **kwargs: scalars.update(values)

    def base_epoch_end(self):
        base_calls.append(self)

    monkeypatch.setattr(
        LightningModuleRatesFinetune,
        "on_validation_epoch_end",
        base_epoch_end,
    )
    module.logp_metrics.update(torch.tensor([1.5]), [Chem.MolFromSmiles("CCO")])
    module.on_validation_epoch_end()
    assert len(images) == int(rank_zero)
    assert all(not plt.fignum_exists(figure.number) for figure in images)
    assert len(base_calls) == 1
    assert "val/logp_mae" in scalars
    assert module.logp_metrics.pairs.shape == (0, 2)
    if rank_zero:
        assert (tmp_path / "logp-parity.png").stat().st_size > 0


class _PropertySource:
    split = "train"

    def __init__(self, root, smiles):
        self.processed_dir, self.smiles = root, smiles

    def __len__(self):
        return len(self.smiles)

    def get(self, index):
        return SimpleNamespace(smiles=self.smiles[index])

    def source_identity(self):
        return {"records": len(self), "ordered_smiles": self.smiles}


def test_property_preparation_preserves_record_order_and_rebuilds_stale_cache(tmp_path):
    source = _PropertySource(tmp_path, ["CCO", "C", "CCO"])
    path = build_property_array(source)
    expected = [Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in source.smiles]
    np.testing.assert_allclose(np.load(path), expected, atol=1e-7)
    assert json.loads(path.with_suffix(".json").read_text())["source"] == (
        source.source_identity()
    )
    original_time = path.stat().st_mtime_ns
    assert build_property_array(source).stat().st_mtime_ns == original_time
    source.smiles = ["C", "CCO", "C"]
    build_property_array(source)
    expected = [Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in source.smiles]
    np.testing.assert_allclose(np.load(path), expected, atol=1e-7)
    previous = path.read_bytes()
    source.smiles[0] = "invalid smiles"
    with pytest.raises(ValueError, match="Cannot calculate"):
        build_property_array(source)
    assert path.read_bytes() == previous


@pytest.mark.parametrize("family,epochs", [("geom", 50), ("qm9", 500)])
def test_logp_presets(family, epochs, monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(Path(__file__).parents[2]))
    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[2] / "configs"), version_base="1.1"
    ):
        cfg = compose(config_name="default", overrides=[f"defaults={family}/logp"])
    assert cfg.model.module.freeze_pretrained_backbone is False
    assert cfg.model.module.freeze_output_heads is False
    assert not cfg.model.lora.enabled
    assert cfg.model.backbone.use_graph_cond
    assert cfg.trainer.trainer.max_epochs == epochs
    assert cfg.data.datamodule._target_.endswith("LogPLightningDataModule")
    for dataset in [
        cfg.data.datamodule.datasets.train,
        *cfg.data.datamodule.datasets.val,
        *cfg.data.datamodule.datasets.test,
    ]:
        assert dataset.compute_properties
        assert list(dataset.property_names) == ["logp"]
