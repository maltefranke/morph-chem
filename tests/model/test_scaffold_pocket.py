"""Scaffold-in-pocket geometry, replay, checkpoint loading, and preprocessing."""

import json
import random
from pathlib import Path

import hydra
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rdkit import Chem

from morph.dataset.conditional.pocket import CrossDockedPocketDataset, build, processing
from morph.dataset.conditional.pocket.datamodule import pocket_eval_collate_fn
from morph.dataset.conditional.pocket.scaffold_pocket import (
    ScaffoldPocketEvaluationDataset,
    ScaffoldPocketWrapper,
    scaffold_cache_directory,
    scaffold_pocket_train_collate_fn,
)
from morph.dataset.conditional.scaffold import cutters
from morph.dataset.conditional.scaffold.cutters import scaffold_indices
from morph.dataset.core.flow_matching_wrapper import FlowMatchingDatasetWrapper
from morph.dataset.core.vocab import Vocab
from morph.model.build import build_model_module, prepare_model_data
from tests.model._helpers import _native_optimizer_config
from tests.model.test_crossdocked import (
    _preprocess_main,
    _pretrained_assets,
    _tokenized,
    _write_pair,
)
from tests.model.test_model_build import _setup
from tests.model.test_pocket_training import _wrapper_options

EXAMPLE = "CCOc1ccc(CCNC(=O)c2ccccc2)cc1"


@pytest.fixture
def pocket_scaffolds(tmp_path):
    source = tmp_path / "source"
    tree = source / processing.RAW_TREE
    pairs = [_write_pair(tree, f"train_{i}", "lig", EXAMPLE) for i in range(4)]
    test = [_write_pair(tree, "test", "lig", EXAMPLE)]
    torch.save({"train": pairs, "test": test}, source / processing.RAW_SPLIT)
    root = tmp_path / "dataset"
    build.build_lmdbs(root, source, val_fraction=0.25)
    return root


def _scaffold_options(dataset, tmp_path, **overrides):
    options = _wrapper_options(dataset, **overrides)
    options["interpolator"].jitter_scale = 0.0
    options.update(
        scaffold_groups_dir=str(tmp_path / "scaffolds"), scaffold_kind="rgroup"
    )
    return options


def test_scaffold_and_pocket_share_translation_and_each_augmentation(
    pocket_scaffolds, tmp_path
):
    dataset = _tokenized(pocket_scaffolds)
    try:
        random.seed(7)
        torch.manual_seed(8)
        wrapper = ScaffoldPocketWrapper(**_scaffold_options(dataset, tmp_path))
        selected = random.Random(7).choice(wrapper._scaffold_choices(0))
        plain = wrapper[0]
        state = plain[0]
        expected = dataset.get_pocket(0).x - state.frame_shift
        torch.testing.assert_close(state.contexts["pocket"].x, expected)
        assert state.keep_mask.any()
        kept = state.keep_mask.bool()
        torch.testing.assert_close(
            state.x[kept] + state.frame_shift,
            dataset[0].x[torch.tensor(selected)],
            atol=1e-5,
            rtol=1e-5,
        )
        distances = torch.cdist(state.x, expected)
        random.seed(7)
        torch.manual_seed(8)
        np.random.seed(9)
        augmented = ScaffoldPocketWrapper(
            **_scaffold_options(dataset, tmp_path, rotate=True, n_augmentations=3)
        )[0]
        for sample in augmented:
            torch.testing.assert_close(
                torch.cdist(sample[0].x, sample[0].contexts["pocket"].x),
                distances,
                atol=1e-5,
                rtol=1e-5,
            )
        assert len({s[0].contexts["pocket"].x.data_ptr() for s in augmented}) == 3
        torch.testing.assert_close(
            dataset.get_pocket(0).x, expected + state.frame_shift
        )
    finally:
        dataset.close()


def test_enumerated_cuts_keep_the_correct_ligand_pocket(pocket_scaffolds, tmp_path):
    dataset = _tokenized(pocket_scaffolds)
    # Distinct context for each molecule exposes accidental indexing by slicing ID.
    get_pocket = dataset.get_pocket
    dataset.get_pocket = lambda i: _translated_pocket(get_pocket(i), i)
    try:
        wrapper = ScaffoldPocketWrapper(
            **_scaffold_options(
                dataset, tmp_path, stage="test", enumerate_slicings=True
            )
        )
        assert len(wrapper) > len(dataset)
        for i in range(len(wrapper)):
            molecule = int(
                np.searchsorted(wrapper._slicing_offsets, i, side="right") - 1
            )
            base_idx = int(wrapper._filtered_indices[molecule])
            source, target = wrapper[i]
            torch.testing.assert_close(
                source.contexts["pocket"].x, dataset.get_pocket(base_idx).x
            )
            torch.testing.assert_close(target.x, dataset[base_idx].x)
            assert source.keep_mask.bool().all()
        batch = pocket_eval_collate_fn([wrapper[0], wrapper[-1]])
        assert batch[0].contexts["pocket"].num_graphs == 2
    finally:
        dataset.close()


def _translated_pocket(pocket, index):
    pocket.x = pocket.x + index * 10.0
    return pocket


def test_pocket_evaluation_repeats_fixed_pairs_with_scoring_ids(
    pocket_scaffolds, tmp_path
):
    dataset = _tokenized(pocket_scaffolds)
    get_pocket = dataset.get_pocket
    dataset.get_pocket = lambda i: _translated_pocket(get_pocket(i), i)
    try:
        wrapped = ScaffoldPocketWrapper(
            **_scaffold_options(dataset, tmp_path, stage="val", enumerate_slicings=True)
        )
        rng_state = torch.random.get_rng_state()
        panel = ScaffoldPocketEvaluationDataset(
            wrapped, n_scaffolds=512, samples_per_scaffold=3
        )
        assert torch.equal(torch.random.get_rng_state(), rng_state)
        assert len(panel) == len(wrapped) * 3  # Small splits retain every pair.
        assert panel.indices == ScaffoldPocketEvaluationDataset(wrapped).indices
        assert len(set(panel.indices)) == len(wrapped)
        assert len(ScaffoldPocketEvaluationDataset(wrapped, n_scaffolds=2)) == 16
        for pair, source_index in enumerate(panel.indices):
            expected, target = wrapped[source_index]
            for repeat in range(3):
                source, actual_target = panel[3 * pair + repeat]
                assert source.scaffold_id.item() == source_index
                assert source.sample_id.item() == 3 * pair + repeat
                torch.testing.assert_close(source.x, expected.x)
                torch.testing.assert_close(
                    source.contexts["pocket"].x, expected.contexts["pocket"].x
                )
                torch.testing.assert_close(actual_target.x, target.x)
        batch = pocket_eval_collate_fn([panel[i] for i in (0, 2, 3)])
        assert batch[0].sample_id.tolist() == [0, 2, 3]
        assert batch[0].scaffold_id.tolist() == [
            panel.indices[0],
            panel.indices[0],
            panel.indices[1],
        ]
        assert batch[0].contexts["pocket"].num_graphs == 3
    finally:
        dataset.close()


def test_cache_identity_distinguishes_equal_length_filtered_views(tmp_path):
    source = tmp_path / "source"
    tree = source / processing.RAW_TREE
    pairs = [
        _write_pair(tree, "train", "silicon", "C[Si](C)(C)c1ccccc1"),
        _write_pair(tree, "train", "selenium", "C[Se]c1ccccc1"),
    ]
    torch.save({"train": pairs, "test": []}, source / processing.RAW_SPLIT)
    root = tmp_path / "dataset"
    datasets = [
        CrossDockedPocketDataset(
            root,
            source_dir=source,
            val_fraction=0,
            vocab=Vocab(atoms, ["<NO_BOND>", "1", "2"], ["0"]),
        )
        for atoms in (["C", "Si"], ["C", "Se"], ["Si", "C"])
    ]
    try:
        assert [len(d) for d in datasets] == [1, 1, 1]
        paths = [scaffold_cache_directory(d, tmp_path / "scaffolds") for d in datasets]
        assert paths[0] != paths[1]
        assert paths[0] == paths[2]
    finally:
        for dataset in datasets:
            dataset.close()


@pytest.mark.parametrize("conditioned", [(True, False), (False, False), (True, True)])
def test_scaffold_pocket_replay_and_checkpoint_gradients(
    pocket_scaffolds, tmp_path, conditioned
):
    dataset = _tokenized(pocket_scaffolds)
    options = _wrapper_options(dataset)
    fine = ScaffoldPocketWrapper(**_scaffold_options(dataset, tmp_path))
    pretrain = FlowMatchingDatasetWrapper(**options)
    try:
        batch = scaffold_pocket_train_collate_fn(
            [
                fine[i] if with_context else pretrain[i]
                for i, with_context in enumerate(conditioned)
            ]
        )
        mask = batch[0].keep_mask.bool()
        rows = torch.tensor(conditioned)[batch[0].batch]
        assert not mask[~rows].any()
        assert bool(mask.any()) == any(conditioned)
        cfg, _ = _setup()
        data = prepare_model_data(cfg, dataset.vocab, options["distributions"])
        pretrained = build_model_module(cfg, data).eval()
        checkpoint = tmp_path / "pretrained.ckpt"
        torch.save({"state_dict": pretrained.state_dict()}, checkpoint)
        cfg.model.module._target_ = (
            "morph.model.finetune.scaffold.LightningModuleRatesScaffoldFinetune"
        )
        cfg.model.module.pretrained_ckpt = str(checkpoint)
        cfg.model.module.pretrained_prefer_ema = False
        cfg.model.module.freeze_keep_positions = True
        cfg.model.module.exclude_scaffold_from_x_loss = True
        cfg.model.module.model._target_ = (
            "morph.model.finetune.pocket.ScaffoldPocketConditionalDiTBackboneWithHeads"
        )
        cfg.model.backbone._target_ = (
            "morph.model.backbones.conditional_dit.ConditionalDiTBackbone"
        )
        cfg.model.backbone.ctx_dim = 16
        cfg.model.backbone.use_cross_attn = True
        cfg.model.backbone.use_node_cond = True
        cfg.model.backbone.use_edge_cond = True
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
            module.model.backbone.node_gate.fill_(0.3)
            module.model.backbone.edge_cond_gate.fill_(0.3)
            for block in module.model.backbone.cross_attn_blocks:
                block.attn_gate.fill_(0.4)
                block.ffn_gate.fill_(0.3)
            mixed = module.model(batch[0], batch[3].view(-1, 1))
            torch.testing.assert_close(
                mixed["h_latent"][~rows], reference["h_latent"][~rows]
            )
        loss = module.shared_step(batch, 0)
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in module.model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
        if all(conditioned):
            evaluation = ScaffoldPocketWrapper(
                **_scaffold_options(dataset, tmp_path, stage="test")
            )
            eval_batch = pocket_eval_collate_fn([evaluation[0]])
            expected_scaffold = eval_batch[0].x.clone() * 2.5
            expected_pocket = eval_batch[0].contexts["pocket"].x.clone() * 2.5
            module.integrator.num_integration_steps = 2
            result = module.sample(eval_batch, batch_idx=0)
            torch.testing.assert_close(
                result.x[result.keep_mask.bool()], expected_scaffold
            )
            torch.testing.assert_close(result.contexts["pocket"].x, expected_pocket)
    finally:
        dataset.close()


@pytest.mark.parametrize("insertion_assignment", ["nearest", "substituent"])
def test_scaffold_preset_trains_with_replay(
    pocket_scaffolds, tmp_path, monkeypatch, insertion_assignment
):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MORPH_PRETRAINED_ASSETS", str(tmp_path / "assets"))
    monkeypatch.setenv("MORPH_PRETRAINED_CKPT", "/tmp/model.ckpt")
    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[2] / "configs"), version_base="1.1"
    ):
        cfg = compose(
            config_name="default",
            overrides=[
                "defaults=crossdocked/scaffold_decoration",
                f"data.root={pocket_scaffolds}",
                "data.val_fraction=0.25",
                "data.scaffold_kind=rgroup",
                f"data.insertion_assignment={insertion_assignment}",
                "data.preprocess_workers=0",
                "data.finetune_fraction=0.5",
                "data.train_epoch_size=8",
                "data.datamodule.batch_size.train=4",
                "data.datamodule.num_workers.train=0",
                "data.datamodule.num_workers.val=0",
                "data.datamodule.batch_size.val=4",
                "data.eval_panel.n_scaffolds=2",
                "data.eval_panel.samples_per_scaffold=2",
            ],
        )
    dataset = _tokenized(pocket_scaffolds)
    options = _wrapper_options(dataset)
    data = prepare_model_data(cfg, dataset.vocab, options["distributions"])
    cfg.data.datamodule.pretrain_dataset = OmegaConf.create(
        dict(
            _target_="morph.dataset.conditional.pocket.CrossDockedPocketDataset",
            root=str(pocket_scaffolds),
            val_fraction=0.25,
        )
    )
    dm = hydra.utils.instantiate(
        cfg.data.datamodule,
        _recursive_=False,
        vocab=data.vocab,
        distributions=data.prior,
    )
    try:
        dm.setup("fit")
        dm.configure_model(cfg)
        assert (
            dm.interpolator.jitter_scale * data.prior.coordinate_std
            == pytest.approx(0.1)
        )
        assert not cfg.model.module.freeze_keep_positions
        assert not cfg.model.module.exclude_scaffold_from_x_loss
        assert cfg.model.module.freeze_pretrained_backbone
        assert not cfg.model.module.freeze_output_heads
        assert cfg.model.lora.enabled
        assert cfg.model.module.optimizer_config.aux_optimizer.lr == pytest.approx(1e-4)
        assert len(dm.val_datasets[0]) > len(dm.val_datasets[0].base_dataset)
        batch = next(iter(dm.train_dataloader()))
        assert batch[0].keep_mask.any()
        assert (~batch[0].contexts["pocket"].num_nodes_per_graph.bool()).any()
        assert isinstance(dm.train_dataset.pretrain, FlowMatchingDatasetWrapper)
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
        module.integrator.num_integration_steps = 2
        contexts_seen = []

        def chemistry_metrics(mols, metrics, *, posebusters_contexts):
            # Exercise real sampling/task scoring while keeping expensive
            # PoseBusters chemistry in its dedicated metric regression tests.
            assert len(posebusters_contexts) == len(mols)
            contexts_seen.extend(posebusters_contexts)
            return {"posebusters/scaffold_n_clash_exemptions": 0.0}

        monkeypatch.setattr(
            "morph.model.finetune.scaffold.scaffold.calc_metrics_", chemistry_metrics
        )
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
        trainer.limit_val_batches = 1
        trainer.validate(module, datamodule=dm, verbose=False)
        scores = trainer.callback_metrics
        assert scores["val/scaffold/n_samples"] == 4
        assert scores["val/scaffold/n_scaffolds"] == 2
        assert scores["val/scaffold/retention"] == 1
        assert torch.isfinite(scores["val/scaffold/rmsd"])
        for key in ("recovery", "uniqueness", "diversity", "valid_retention"):
            assert f"val/scaffold/{key}" in scores
        assert "val/scaffold/posebusters/n_clash_exemptions" in scores
        assert len(contexts_seen) == 4
    finally:
        dataset.close()


def test_libinvent_cuts_and_preprocessing_match_training(
    pocket_scaffolds, tmp_path, monkeypatch
):
    pytest.importorskip("reinvent_chemistry")
    # Exercise the real enumerator offline with a small synthetic amide rule.
    reactions, filters = tmp_path / "reaction.smirks", tmp_path / "filters.json"
    reactions.write_text("[C:1](=[O:2])-[N:3]>>[C:1](=[O:2])[*].[*][N:3]\n")
    filters.write_text('{"scaffold": [], "decoration": []}')
    monkeypatch.setattr(cutters, "libinvent_resources", lambda: (reactions, filters))
    # Bypass the process memo so the synthetic enumerator cannot leak to other tests.
    monkeypatch.setattr(
        cutters, "libinvent_enumerator", cutters.libinvent_enumerator.__wrapped__
    )
    monkeypatch.setattr(
        cutters,
        "_libinvent_scaffold_smiles",
        cutters._libinvent_scaffold_smiles.__wrapped__,
    )
    mol = Chem.MolFromSmiles(EXAMPLE)
    cuts = scaffold_indices(mol, "libinvent")
    assert cuts
    assert len(cuts) == len(set(cuts))
    assert all(0 < len(cut) < mol.GetNumAtoms() for cut in cuts)
    assets = _pretrained_assets(tmp_path / "assets")
    summary = _preprocess_main()(
        [
            "--root",
            str(pocket_scaffolds),
            "--pretrained-assets",
            str(assets),
            "--val-frac",
            "0.25",
            "--scaffold-kind",
            "libinvent",
        ]
    )
    assert summary["status"] == "ready"
    for split, report in summary["splits"].items():
        dataset = _tokenized(pocket_scaffolds, split=split)
        try:
            directory = scaffold_cache_directory(
                dataset, pocket_scaffolds / "scaffold_groups"
            )
            wrapper = ScaffoldPocketWrapper(
                **dict(
                    _wrapper_options(dataset, stage=split),
                    scaffold_groups_dir=directory,
                    scaffold_kind="libinvent",
                    enumerate_slicings=True,
                )
            )
            expected = report["scaffolds"][
                "molecules" if split == "train" else "slicings"
            ]
            assert len(wrapper) == expected
            assert report["scaffolds"]["directory"] == directory
        finally:
            dataset.close()
    report = (
        pocket_scaffolds
        / "processed/scaffold_libinvent_finetune_preprocessing_summary.json"
    )
    assert json.loads(report.read_text())["status"] == "ready"
