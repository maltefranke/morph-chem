"""Lora regression tests."""

import copy

import pytest
import torch
from omegaconf import OmegaConf

from morph.model.backbones.conditional_dit import ConditionalDiTBackboneWithHeads
from morph.model.backbones.dit import DiTBackboneWithHeads
from morph.model.finetune.base import LightningModuleRatesFinetune
from morph.model.layers.lora import LoRALinear
from morph.model.lightning.base import LightningModuleRates

from ._helpers import _backbone, _rbf_args


def test_lora_linear_preserves_initial_output_and_merged_checkpoint():
    base = torch.nn.Linear(7, 5)
    adapter = LoRALinear.from_linear(base, rank=3, alpha=6.0).eval()
    x = torch.randn(4, 7)

    torch.testing.assert_close(adapter(x), base(x), rtol=0.0, atol=0.0)
    with torch.no_grad():
        adapter.lora_B.normal_(std=0.1)
    expected = adapter(x)

    adapter.merge_lora_()
    torch.testing.assert_close(adapter(x), expected)
    state_dict = adapter.state_dict()

    restored = LoRALinear(7, 5, rank=3, alpha=6.0).eval()
    restored.load_state_dict(state_dict)
    torch.testing.assert_close(restored(x), expected)


def test_dit_lora_targets_only_block_projections_and_freezes_base():
    backbone = _backbone(num_layers=2)
    adapter_params = backbone.enable_lora(rank=2, alpha=4.0)

    assert backbone.has_lora
    assert isinstance(backbone.blocks[0].qkv, LoRALinear)
    assert isinstance(backbone.blocks[0].proj, LoRALinear)
    assert isinstance(backbone.blocks[0].mlp.gate_value, LoRALinear)
    assert isinstance(backbone.blocks[0].mlp.out_proj, LoRALinear)
    assert adapter_params == sum(
        parameter.numel()
        for name, parameter in backbone.named_parameters()
        if name.endswith(("lora_A", "lora_B"))
    )
    assert all(
        name.endswith(("lora_A", "lora_B"))
        for name, parameter in backbone.named_parameters()
        if parameter.requires_grad
    )
    assert "blocks.0.qkv.weight" in backbone.state_dict()


def test_dit_lora_loads_an_unmodified_backbone_state_dict():
    source = _backbone(num_layers=1).eval()
    adapted = _backbone(num_layers=1).eval()
    adapted.enable_lora(rank=2)

    missing, unexpected = adapted.load_state_dict(source.state_dict(), strict=False)

    assert not unexpected
    assert missing
    assert all(name.endswith(("lora_A", "lora_B")) for name in missing)
    x = torch.randn(5, 32)
    torch.testing.assert_close(adapted.blocks[0].qkv(x), source.blocks[0].qkv(x))


def _model_with_heads(conditional):
    backbone_args = {
        "_target_": "morph.model.backbones.dit.DiTBackbone",
        "_recursive_": False,
        "d_input": 7,
        "d_cond": 6,
        "d_model": 32,
        "out_node_nf": 16,
        "out_edge_nf": 10,
        "num_layers": 1,
        "nhead": 4,
        "in_edge_nf": 5,
        "rbf_embedding_args": _rbf_args(),
        "ffn_multiple_of": 8,
        "edge_pair_dim": 12,
        "num_atom_types": 6,
    }
    wrapper = DiTBackboneWithHeads
    if conditional:
        wrapper = ConditionalDiTBackboneWithHeads
        backbone_args.update(
            _target_="morph.model.backbones.conditional_dit.ConditionalDiTBackbone",
            ctx_dim=8,
            use_cross_attn=True,
            use_node_cond=True,
            use_edge_cond=True,
        )
    # Small stand-ins for the independent embedding/head MLPs; the real DiT
    # and conditional backbone exercise the nested coordinate-readout policy.
    linear_args = OmegaConf.create(
        {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 4}
    )
    return wrapper(
        atom_type_embedding_args=linear_args,
        edge_type_embedding_args=linear_args,
        time_embedding_args=linear_args,
        node_count_embedding_args=linear_args,
        backbone_model_args=OmegaConf.create(backbone_args),
        heads_args=linear_args,
        insertion_head_args=linear_args,
        ins_edge_head_args=linear_args,
    )


@pytest.mark.parametrize("conditional", [False, True])
def test_lora_can_unfreeze_all_output_heads_without_shared_features(conditional):
    model = _model_with_heads(conditional)
    dit = model.backbone.dit if conditional else model.backbone
    dit.enable_lora(rank=2)
    model.freeze_pretrained()
    before_trainable = {id(p) for p in model.parameters() if p.requires_grad}
    head_params = [
        p
        for head in (
            model.heads,
            model.insertion_head,
            model.ins_edge_head,
            dit.pos_out,
        )
        for p in head.parameters()
    ]
    assert all(not p.requires_grad for p in head_params)
    before_weights = {name: p.detach().clone() for name, p in model.named_parameters()}

    assert model.set_output_heads_trainable() == sum(p.numel() for p in head_params)
    assert {id(p) for p in model.parameters() if p.requires_grad} == (
        before_trainable | {id(p) for p in head_params}
    )
    assert all(not p.requires_grad for p in model.embedding_backbone.parameters())
    assert all(not p.requires_grad for p in dit.node_out.parameters())
    assert all(not p.requires_grad for p in dit.edge_output.parameters())
    assert model.set_output_heads_trainable() == 0
    changed = model.set_output_heads_trainable(False)
    assert changed == sum(p.numel() for p in head_params)
    assert {id(p) for p in model.parameters() if p.requires_grad} == before_trainable
    if conditional:
        assert model.backbone.pos_out is dit.pos_out
        assert not any(
            name.startswith("backbone.pos_out.") for name in model.state_dict()
        )
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before_weights[name], rtol=0, atol=0)


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("freeze_backbone", [False, True])
@pytest.mark.parametrize("freeze_heads", [None, False, True])
@pytest.mark.parametrize("load_pretrained", [False, True])
def test_finetune_head_freezing_is_independent_and_preserves_defaults(
    monkeypatch, tmp_path, conditional, freeze_backbone, freeze_heads, load_pretrained
):
    model = _model_with_heads(conditional)
    checkpoint = tmp_path / "pretrained.ckpt"
    torch.save(
        {"state_dict": {f"model.{n}": p for n, p in model.state_dict().items()}},
        checkpoint,
    )

    def init_model_only(self):
        # Exercise the real fine-tuning init/checkpoint load without unrelated
        # data, metric and integrator setup in the parent Lightning module.
        torch.nn.Module.__init__(self)
        self.model = model
        self.model_ema = copy.deepcopy(model).requires_grad_(False)

    monkeypatch.setattr(LightningModuleRates, "__init__", init_model_only)
    module = LightningModuleRatesFinetune(
        pretrained_ckpt=str(checkpoint) if load_pretrained else None,
        freeze_pretrained_backbone=freeze_backbone,
        freeze_output_heads=freeze_heads,
    )
    expected_frozen = freeze_backbone if freeze_heads is None else freeze_heads
    assert all(
        p.requires_grad == (not expected_frozen)
        for head in module.model.output_head_modules()
        for p in head.parameters()
    )
    assert all(
        p.requires_grad == (not freeze_backbone)
        for p in module.model.embedding_backbone.parameters()
    )
    dit = model.backbone.dit if conditional else model.backbone
    assert all(
        p.requires_grad == (not freeze_backbone) for p in dit.blocks.parameters()
    )
    assert all(not p.requires_grad for p in module.model_ema.parameters())


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("freeze_heads", [False, True])
def test_finetune_resume_without_pretrained_file_preserves_optimizer_state(
    monkeypatch, tmp_path, conditional, freeze_heads
):
    def init_model_only(self):
        torch.nn.Module.__init__(self)
        self.model = _model_with_heads(conditional)
        dit = self.model.backbone.dit if conditional else self.model.backbone
        dit.enable_lora(rank=2)
        self.model_ema = copy.deepcopy(self.model).requires_grad_(False)

    monkeypatch.setattr(LightningModuleRates, "__init__", init_model_only)
    source = _model_with_heads(conditional)
    pretrained = tmp_path / "pretrained.ckpt"
    torch.save(
        {"state_dict": {f"model.{n}": p for n, p in source.state_dict().items()}},
        pretrained,
    )
    original = LightningModuleRatesFinetune(
        pretrained_ckpt=str(pretrained),
        freeze_pretrained_backbone=True,
        freeze_output_heads=freeze_heads,
    )

    def make_optimizer(module):
        return torch.optim.AdamW(
            [p for p in module.model.parameters() if p.requires_grad], lr=0.001
        )

    def step(module, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(p.square().sum() for p in module.model.parameters())
        loss.backward()
        optimizer.step()

    optimizer = make_optimizer(original)
    step(original, optimizer)
    checkpoint = tmp_path / "finetuned.ckpt"
    torch.save(
        {
            "state_dict": original.state_dict(),
            "optimizer_states": [optimizer.state_dict()],
        },
        checkpoint,
    )
    pretrained.unlink()

    resumed = LightningModuleRatesFinetune(
        pretrained_ckpt=None,
        freeze_pretrained_backbone=True,
        freeze_output_heads=freeze_heads,
    )
    state = torch.load(checkpoint, weights_only=True)
    resumed.load_state_dict(state["state_dict"], strict=True)
    assert {n for n, p in resumed.named_parameters() if p.requires_grad} == {
        n for n, p in original.named_parameters() if p.requires_grad
    }
    assert all(not p.requires_grad for p in resumed.model_ema.parameters())
    resumed_optimizer = make_optimizer(resumed)
    resumed_optimizer.load_state_dict(state["optimizer_states"][0])
    assert resumed_optimizer.state

    step(original, optimizer)
    step(resumed, resumed_optimizer)
    for name, expected in original.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)
