"""Dynamic count conditioning, checkpoint compatibility and trainability."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from morph.dataset.conditional.atom_count import AtomCountFlowMatchingWrapper
from morph.dataset.core.atomic_data import ChemAtomicData, chembatch_from_graphs
from morph.dataset.core.flow_matching_wrapper import eval_collate_fn, train_collate_fn
from morph.dataset.core.representation import Capabilities
from morph.flow_matching.integration import IntegrationPredictions
from morph.flow_matching.interpolation import Interpolator
from morph.flow_matching.prior import sample_prior_graph
from morph.inference.module import load_inference_weights
from morph.model.build import build_model_module
from morph.model.finetune.atom_count import atom_count_delta
from morph.model.finetune.base.finetune_base import load_pretrained_into_lightning_model

from ._helpers import _noise_consistency_setup
from .test_model_build import _setup


class _Targets:
    CAPABILITIES = Capabilities(provides_charges=True, provides_topology=True)

    def __getitem__(self, index):
        return ChemAtomicData(
            torch.randn(5 + index, 3),
            torch.ones(5 + index, dtype=torch.long),
            torch.zeros(5 + index, dtype=torch.long),
        )

    def __len__(self):
        return 2


def _wrapper(data, **kwargs):
    wrapper = AtomCountFlowMatchingWrapper(
        _Targets(),
        data.prior,
        Interpolator(data.prior, optimal_transport="none", c_ins=1e6, c_del=1e6),
        data.vocab,
        **kwargs,
    )
    wrapper._get_n_atoms = lambda target: 3
    wrapper.time_dist = torch.distributions.Uniform(0.0, 1e-6)
    return wrapper


def _conditioned_config(cfg, variant="mlp"):
    cfg.model.module._target_ = (
        "morph.model.finetune.atom_count.LightningModuleRatesAtomCountFinetune"
    )
    cfg.model.module.model._target_ = (
        "morph.model.finetune.atom_count."
        + ("Fourier" if variant == "fourier" else "")
        + "AtomCountDiTBackboneWithHeads"
    )
    cfg.model.lora.enabled = True
    cfg.model.lora.rank = 32
    cfg.model.lora.alpha = 32.0
    cfg.model.module.freeze_pretrained_backbone = True
    cfg.model.module.freeze_output_heads = False


@pytest.mark.parametrize("stage", ["train", "val", "test"])
@pytest.mark.parametrize("augmentations", [1, 2])
def test_count_uses_full_target_and_survives_augmentation_batching(
    stage, augmentations
):
    _, data = _setup()
    wrapper = _wrapper(data, stage=stage, n_augmentations=augmentations)
    items = [wrapper[i] for i in range(2)]
    collate = train_collate_fn if stage == "train" else eval_collate_fn
    batch = collate(items)[0]
    expected = torch.tensor([5, 6])
    if stage == "train":
        expected = expected.repeat_interleave(augmentations)
    torch.testing.assert_close(batch.target_n_atoms, expected)
    torch.testing.assert_close(atom_count_delta(batch), expected - 3)
    restored = chembatch_from_graphs(batch.clone().to_data_list())
    torch.testing.assert_close(restored.target_n_atoms, expected)


@pytest.mark.parametrize("variant", ["mlp", "fourier"])
def test_pretrained_outputs_and_complete_finetuning_gradients(tmp_path, variant):
    cfg, data = _setup()
    baseline = build_model_module(cfg, data).eval()
    # A fresh DiT has zero AdaLN weights and ignores every graph condition.
    # Model a trained checkpoint so the added condition has a gradient path.
    with torch.no_grad():
        for name, parameter in baseline.model.named_parameters():
            if "adaLN_modulation" in name and name.endswith("weight"):
                parameter.normal_(std=0.02)
    checkpoint = tmp_path / "pretrained.ckpt"
    torch.save({"state_dict": baseline.state_dict()}, checkpoint)
    _conditioned_config(cfg, variant)
    adapted = build_model_module(cfg, data).eval()
    missing, unexpected = load_pretrained_into_lightning_model(
        adapted.model, checkpoint, prefer_ema=False
    )
    assert not unexpected
    assert missing
    assert all(
        name.startswith("delta_embedding.")
        or name == "backbone.graph_gate"
        or name.endswith(("lora_A", "lora_B"))
        for name in missing
    )
    batch = train_collate_fn([_wrapper(data, stage="train")[0]])
    state, _, _, time = batch
    torch.testing.assert_close(
        baseline.model.embed_inputs(state, time[:, None]),
        adapted.model.embedding_backbone(state, time[:, None]),
        rtol=0,
        atol=0,
    )
    # Frozen and trainable projections may use slightly different CPU kernels.
    torch.testing.assert_close(
        baseline.model(state, time[:, None]),
        adapted.model(state, time[:, None]),
        rtol=1e-5,
        atol=1e-7,
    )
    assert all(
        not p.requires_grad for p in adapted.model.embedding_backbone.parameters()
    )
    assert all(p.requires_grad for p in adapted.model.delta_embedding.parameters())
    assert all(
        p.requires_grad
        for head in adapted.model.output_head_modules()
        for p in head.parameters()
    )
    assert all(not p.requires_grad for p in adapted.model_ema.parameters())
    adapted.log_dict = lambda *args, **kwargs: None
    loss = adapted.shared_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in adapted.model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    if variant == "fourier":
        assert adapted.model.backbone.graph_gate.requires_grad
        assert adapted.model.backbone.graph_gate.grad.abs().sum() > 0
        # The initially closed gate delays encoder learning until it opens.
        assert adapted.model.delta_embedding.projection[-1].weight.grad.abs().sum() == 0
    else:
        assert adapted.model.delta_embedding[-1].weight.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(
        [p for p in adapted.model.parameters() if p.requires_grad], lr=1e-3
    )
    optimizer.step()
    changed = state.clone()
    changed.target_n_atoms += 10
    if variant == "fourier":
        captured = []
        handle = adapted.model.backbone.dit.blocks[0].register_forward_pre_hook(
            lambda module, args: captured.append(args[1].detach().clone())
        )
        with torch.no_grad():
            adapted.model(state, time[:, None])
            adapted.model(changed, time[:, None])
            base_condition = adapted.model.backbone.dit.cond_proj(
                adapted.model.embedding_backbone(state, time[:, None])[2]
            )
            expected = base_condition + adapted.model.backbone.graph_gate.tanh() * (
                adapted.model.delta_embedding(atom_count_delta(state))
            )
        handle.remove()
        torch.testing.assert_close(captured[0], expected)
        assert not torch.equal(captured[0], captured[1])
        assert all(
            not p.requires_grad
            for p in adapted.model.backbone.dit.cond_proj.parameters()
        )
        optimizer.zero_grad()
        adapted.shared_step(batch, 0).backward()
        assert adapted.model.delta_embedding.projection[-1].weight.grad.abs().sum() > 0
    else:
        first = adapted.model.embed_inputs(state, time[:, None])[2]
        second = adapted.model.embed_inputs(changed, time[:, None])[2]
        assert not torch.allclose(first, second)


def test_fourier_gate_preserves_small_residual_under_mixed_precision():
    cfg, data = _setup()
    _conditioned_config(cfg, "fourier")
    model = build_model_module(cfg, data).model.eval()
    state = eval_collate_fn([_wrapper(data, stage="val")[0]])[0]
    time = torch.zeros(1, 1)
    captured = []
    handle = model.backbone.dit.blocks[0].register_forward_pre_hook(
        lambda module, args: captured.append(args[1].detach().clone())
    )
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        model(state, time)
        model.backbone.graph_gate.fill_(1e-7)
        model(state, time)
        expected_residual = model.backbone.graph_gate.tanh() * model.delta_embedding(
            atom_count_delta(state)
        )
    handle.remove()
    assert captured[1].dtype == torch.float32
    torch.testing.assert_close(captured[1], captured[0] + expected_residual)
    assert not torch.equal(captured[0], captured[1])


@pytest.mark.parametrize("force_uncond", [False, True])
def test_scaffold_wrapper_preserves_explicit_graph_condition(force_uncond):
    cfg, data = _setup()
    cfg.model.module.model._target_ = (
        "morph.model.finetune.scaffold.ScaffoldConditionalDiTBackboneWithHeads"
    )
    cfg.model.backbone._target_ = (
        "morph.model.backbones.conditional_dit.ConditionalDiTBackbone"
    )
    cfg.model.backbone.use_graph_cond = True
    cfg.model.backbone.use_node_cond = True
    cfg.model.backbone.use_edge_cond = True
    model = build_model_module(cfg, data).model.eval()
    with torch.no_grad():
        model.backbone.graph_gate.fill_(0.2)
        for name, parameter in model.backbone.dit.named_parameters():
            if "adaLN_modulation" in name and name.endswith("weight"):
                parameter.normal_(std=0.02)
    state = eval_collate_fn([_wrapper(data, stage="val")[0]])[0]
    state.keep_mask = torch.tensor([True, False, False])
    graph_cond = torch.randn(1, model.backbone.d_model, requires_grad=True)
    outputs = model(
        state, torch.zeros(1, 1), graph_cond=graph_cond, force_uncond=force_uncond
    )
    outputs["atom_type_head"].square().sum().backward()
    assert graph_cond.grad is not None
    assert torch.isfinite(graph_cond.grad).all()
    assert graph_cond.grad.abs().sum() > 0


@pytest.mark.parametrize("loader", ["module", "pretrained", "inference"])
def test_fourier_legacy_checkpoint_loads_graph_gate_and_clean_backbone(
    tmp_path, loader
):
    cfg, data = _setup()
    _conditioned_config(cfg, "fourier")
    original = build_model_module(cfg, data)
    with torch.no_grad():
        original.model.backbone.graph_gate.fill_(0.2)
        original.model_ema.backbone.graph_gate.fill_(-0.3)
    legacy = {}
    for name, value in original.state_dict().items():
        for prefix in ("model.", "model_ema."):
            name = name.replace(prefix + "backbone.dit.", prefix + "backbone.")
            if name == prefix + "backbone.graph_gate":
                name = prefix + "graph_gate"
        legacy[name] = value.clone()
    checkpoint = tmp_path / "legacy_fourier.ckpt"
    torch.save({"state_dict": legacy}, checkpoint)
    restored = build_model_module(cfg, data)
    if loader == "module":
        restored.load_state_dict(legacy, strict=True)
        torch.testing.assert_close(
            restored.state_dict(), original.state_dict(), rtol=0, atol=0
        )
    else:
        if loader == "pretrained":
            missing, unexpected = load_pretrained_into_lightning_model(
                restored.model, checkpoint, prefer_ema=True, strict=True
            )
            assert not missing
            assert not unexpected
        else:
            load_inference_weights(restored.model, checkpoint, use_ema=True)
        torch.testing.assert_close(
            restored.model.state_dict(), original.model_ema.state_dict(), rtol=0, atol=0
        )


@pytest.mark.parametrize("deleting", [True, False])
def test_delta_updates_after_real_insertion_or_deletion(monkeypatch, deleting):
    prior, _, integrator = _noise_consistency_setup()
    integrator.renoise_insertions = False
    monkeypatch.setattr(torch, "rand_like", lambda value: torch.full_like(value, 0.5))
    mol = sample_prior_graph(prior, n_atoms=4)
    mol.add_system_property("target_n_atoms", torch.tensor([5]))
    batch = chembatch_from_graphs([mol])
    torch.testing.assert_close(atom_count_delta(batch), torch.tensor([1]))
    zeros = torch.zeros(4)
    edit = torch.tensor([0.75, 0.0, 0.0, 0.0])
    result, _ = integrator.integrate_step_gnn(
        mol_t=batch,
        predictions=IntegrationPredictions(
            mol_1_pred=batch.clone(),
            do_sub_a_probs=zeros,
            do_sub_e_probs=torch.zeros(batch.e.shape[0]),
            do_del_probs=edit if deleting else zeros,
            num_ins_pred=zeros if deleting else edit,
            insertion_preds={
                "mu": torch.zeros(4, 3),
                "a_logits": torch.zeros(4, 3),
                "c_logits": torch.zeros(4, 3),
            },
        ),
        t=torch.tensor([0.9]),
        dt=0.1,
        is_final=True,
        return_step_trace=True,
    )
    torch.testing.assert_close(result.target_n_atoms, torch.tensor([5]))
    torch.testing.assert_close(
        atom_count_delta(result), torch.tensor([2 if deleting else 0])
    )


def test_signed_delta_and_missing_target():
    _, data = _setup()
    batch = eval_collate_fn([_wrapper(data, stage="val")[0] for _ in range(3)])[0]
    batch.target_n_atoms = torch.tensor([8, 3, 1])
    torch.testing.assert_close(atom_count_delta(batch), torch.tensor([5, 0, -2]))
    with pytest.raises(ValueError, match="target_n_atoms"):
        atom_count_delta(chembatch_from_graphs([_Targets()[0]]))


def test_count_metrics_include_invalid_molecules():
    cfg, data = _setup()
    _conditioned_config(cfg)
    module = build_model_module(cfg, data)
    logged = {}
    module.log_dict = lambda values, **kwargs: logged.update(values)
    module._update_task_metrics(
        torch.tensor([5, 3]), [_Targets()[0], _Targets()[1]], [None, None]
    )
    assert logged["val/atom_count_mae"] == 1.5
    assert logged["val/atom_count_exact"] == 0.5


@pytest.mark.parametrize("variant", ["mlp", "fourier", "fourier_full"])
def test_geom_atom_count_preset(variant):
    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[2] / "configs"), version_base="1.1"
    ):
        overrides = ["defaults=geom/atom_count"]
        if variant != "mlp":
            overrides.append(f"model=finetune/atom_count_{variant}")
        cfg = compose(config_name="default", overrides=overrides)
    full_finetune = variant == "fourier_full"
    assert cfg.model.lora.enabled == (not full_finetune)
    assert cfg.model.lora.rank == 32
    assert cfg.model.lora.alpha == 32
    assert cfg.model.module.freeze_pretrained_backbone == (not full_finetune)
    assert not cfg.model.module.freeze_output_heads
    assert cfg.data.datamodule._target_.endswith("AtomCountLightningDataModule")
    assert cfg.data.n_atoms_strategy == "uniform"
    assert cfg.trainer.trainer.check_val_every_n_epoch == 1
    assert cfg.callbacks.every_n_epochs_checkpoint.every_n_epochs == 5
    assert cfg.callbacks.every_n_epochs_checkpoint.save_top_k == -1
    assert cfg.callbacks.model_checkpoints.every_n_epochs == 5
    if variant != "mlp":
        assert cfg.model.backbone._target_.endswith(".ConditionalDiTBackbone")
        assert cfg.model.backbone.use_graph_cond
        assert cfg.model.module.model._target_.endswith(
            "FourierAtomCountDiTBackboneWithHeads"
        )
        assert cfg.model.module.model.delta_scale == 128
        assert cfg.model.module.model.delta_hidden_dim == 256
    if full_finetune:
        assert not cfg.model.lora.freeze_base
        assert cfg.model.module.optimizer_config.optimizer.lr == 3e-4
        assert cfg.model.module.optimizer_config.aux_optimizer.lr == 3e-5
