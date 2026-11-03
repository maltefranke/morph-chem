"""Shared construction preserves training/inference checkpoint and loss contracts."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from morph.dataset.core.atomic_data import ChemAtomicData
from morph.dataset.core.flow_matching_wrapper import train_collate_fn
from morph.dataset.core.insertion_targets import InsertionTargets
from morph.dataset.core.vocab import Distributions, Vocab
from morph.flow_matching.interpolation import Interpolator
from morph.inference.module import InferenceModule, load_inference_weights
from morph.model.build import (
    build_model_metrics,
    build_model_module,
    prepare_model_data,
)
from morph.utils.metrics.tensor import DIST_N_BINS, RG_N_BINS


def _setup(representation="molecule", insertion_head="unimodal"):
    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[2] / "configs"), version_base="1.1"
    ):
        cfg = compose(
            config_name="default",
            overrides=[
                f"representation={representation}",
                f"insertion_head={insertion_head}",
                "model.backbone.d_model=32",
                "model.backbone.num_layers=1",
                "model.backbone.nhead=4",
                "model.backbone.out_node_nf=16",
                "model.backbone.out_edge_nf=16",
                "model.head_node_adapter_dim=16",
                "model.head_edge_adapter_dim=16",
                "model.head_insertion_adapter_dim=16",
                "model.head_insertion_edge_dim=16",
                "model.backbone.ffn_multiple_of=8",
            ],
        )
    vocab = Vocab(["C", "N"], ["<NO_BOND>", "1"], ["0", "1"])
    empirical = Distributions(
        torch.tensor([0.8, 0.2]),
        torch.tensor([0.9, 0.1]),
        torch.tensor([0.95, 0.05]),
        torch.tensor([0.0, 0.0, 0.0, 0.5, 0.5]),
        2.0,
        pairwise_distance_histogram=torch.ones(2, 2, DIST_N_BINS),
        radius_of_gyration_histogram=torch.ones(RG_N_BINS),
    )
    data = prepare_model_data(cfg, vocab, empirical)
    return cfg, data


@pytest.mark.parametrize("representation", ["molecule", "pointcloud"])
def test_training_and_dataset_free_builds_share_checkpoint_buffers(representation):
    cfg, data = _setup(representation)
    training = build_model_module(
        cfg, data, metrics=build_model_metrics(data, train_smiles=["C"])
    )
    inference = build_model_module(cfg, data)
    inference.load_state_dict(training.state_dict(), strict=True)
    assert training.state_dict().keys() == inference.state_dict().keys()
    assert any(name.startswith("batch_metrics.") for name in training.state_dict())
    torch.testing.assert_close(
        data.empirical.atom_type_distribution, torch.tensor([0.8, 0.2])
    )
    torch.testing.assert_close(
        data.prior.atom_type_distribution, torch.tensor([0.5, 0.5])
    )


@pytest.mark.parametrize("use_ema", [True, False])
def test_prediction_module_has_one_network_with_matching_weights(tmp_path, use_ema):
    cfg, data = _setup()
    training = build_model_module(cfg, data).eval()
    with torch.no_grad():
        next(training.model_ema.parameters()).add_(0.01)
    checkpoint = tmp_path / "training.ckpt"
    torch.save({"state_dict": training.state_dict()}, checkpoint)
    cfg.model.module.use_ema_for_eval = use_ema
    inference = InferenceModule(cfg, data).eval()
    load_inference_weights(inference.model, checkpoint, use_ema=use_ema)
    expected = training.model_ema if use_ema else training.model
    assert not hasattr(inference, "model_ema")
    assert not hasattr(inference, "loss_accumulator")
    assert not hasattr(inference, "metrics")
    assert inference.weights == ("ema" if use_ema else "current")
    assert sum(p.numel() for p in inference.parameters()) == sum(
        p.numel() for p in expected.parameters()
    )
    # Buffers and all heads must match too, not just the embedding weights.
    torch.testing.assert_close(inference.model.state_dict(), expected.state_dict())


@pytest.mark.parametrize("insertion_head", ["unimodal", "gmm"])
def test_structured_training_batch_runs_forward_and_backward(insertion_head):
    torch.manual_seed(53)
    cfg, data = _setup(insertion_head=insertion_head)
    module = build_model_module(cfg, data)
    logged = []
    module.log_dict = lambda values, **kwargs: logged.append(values)
    forwards = []
    handle = module.model.register_forward_hook(lambda *args: forwards.append(None))
    interpolator = Interpolator(
        data.prior,
        optimal_transport="none",
        pre_align_iters=0,
        move_noise_scale=0.0,
        c_ins=1e6,
        c_del=1e6,
    )
    samples = []
    for source_count, target_count, time in [(3, 5, 0.0), (4, 4, 0.5)]:
        source = ChemAtomicData(
            torch.randn(source_count, 3),
            torch.zeros(source_count, dtype=torch.long),
            torch.zeros(source_count, dtype=torch.long),
        )
        target = ChemAtomicData(
            torch.randn(target_count, 3),
            torch.ones(target_count, dtype=torch.long),
            torch.zeros(target_count, dtype=torch.long),
        )
        samples.append(
            (*interpolator.interpolate_single(source, target, time), torch.tensor(time))
        )
    batch = train_collate_fn(samples)
    assert isinstance(batch[2], InsertionTargets)
    assert batch[2].num_nodes == 2
    loss = module.shared_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in module.model.parameters()
        if parameter.requires_grad
    )
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in module.model.parameters()
        if parameter.grad is not None
    )
    standalone_gradients = {
        name: parameter.grad.clone()
        for name, parameter in module.model.named_parameters()
        if parameter.requires_grad
    }
    standalone_logs = module.loss_accumulator.log_dict()
    assert len(forwards) == len(logged) == 1

    # A fusion trainer owns its forward and may transform the predictions. A
    # zero residual preserves the reference values while proving gradients can
    # reach external parameters through the public objective.
    module.model.zero_grad(set_to_none=True)
    mols_t, _, _, times = batch
    predictions = module.model(mols_t, times.view(-1, 1))
    residual = torch.zeros_like(predictions["pos_head"], requires_grad=True)
    predictions["pos_head"] = predictions["pos_head"] + residual
    module.loss_accumulator.add_stat("previous_batch_only", 1.0)
    external_loss = module.loss_from_predictions(predictions, batch)
    assert len(forwards) == 2
    assert len(logged) == 1
    torch.testing.assert_close(external_loss, loss)
    torch.testing.assert_close(module.loss_accumulator.log_dict(), standalone_logs)
    external_loss.backward()
    assert torch.isfinite(residual.grad).all()
    assert torch.count_nonzero(residual.grad) > 0
    for name, parameter in module.model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, standalone_gradients[name])
    handle.remove()


@pytest.mark.parametrize("freeze_charges", [True, False])
@pytest.mark.parametrize("exclude_substitutions", [True, False])
def test_scaffold_charge_supervision_matches_inference_policy(
    freeze_charges, exclude_substitutions
):
    cfg, data = _setup()
    cfg.model.module._target_ = (
        "morph.model.finetune.scaffold.LightningModuleRatesScaffoldFinetune"
    )
    cfg.model.module.freeze_keep_charges = freeze_charges
    cfg.model.module.exclude_scaffold_from_sub_loss = exclude_substitutions
    module = build_model_module(cfg, data)
    assert not module._freeze_keep_positions
    assert not module._exclude_scaffold_from_x_loss
    interpolator = Interpolator(
        data.prior,
        optimal_transport="none",
        pre_align_iters=0,
        move_noise_scale=0.0,
        c_ins=1e6,
        c_del=1e6,
    )
    source = ChemAtomicData(
        torch.randn(4, 3),
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
    )
    target = source.clone()
    # The retained source charge may differ from the desired full-molecule charge.
    target.c = torch.ones(4, dtype=torch.long)
    mol_t, mol_1, insertion_targets = interpolator.interpolate_single(
        source, target, 0.5
    )
    mol_t.add_node_property("keep_mask", torch.tensor([True, True, False, False]))
    mol_t.lambda_del[-1] = 1.0
    batch = train_collate_fn([(mol_t, mol_1, insertion_targets, torch.tensor(0.5))])
    mol_t, _, _, times = batch
    predictions = module.model(mol_t, times.view(-1, 1))
    charge_logits = predictions["charge_head"]
    charge_logits.retain_grad()
    # The fresh coordinate readout predicts the already-clean input exactly;
    # introduce a known geometric error so a supervised loss has a gradient.
    coordinates = (
        predictions["pos_head"] + torch.tensor([0.2, -0.2, 0.1, -0.1])[:, None]
    )
    predictions["pos_head"] = coordinates
    coordinates.retain_grad()
    module.loss_from_predictions(predictions, batch).backward()

    # Scaffold coordinates are supervised even when their discrete losses are off.
    assert torch.all(coordinates.grad[:2].abs().sum(-1) > 0)

    gradients = charge_logits.grad.abs().sum(-1)
    if freeze_charges:
        assert torch.all(gradients[:2] == 0)
    else:
        assert torch.all(gradients[:2] > 0)
    assert gradients[2] > 0  # Decoration atoms always receive charge supervision.
    assert gradients[3] == 0  # Deletion targets never do.
