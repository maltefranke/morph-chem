"""Inference regression tests."""

import json
import weakref
from types import SimpleNamespace

import httpx
import huggingface_hub
import pytest
import pytorch_lightning as pl
import torch
from click.testing import CliRunner
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from omegaconf import OmegaConf
from rdkit import Chem

import morph.flow_matching.prior as prior
import morph.inference.generate as generation
import morph.inference.hub as hub
import morph.inference.module as inference_module
import morph.utils.run_config as run_config
from morph.cli import cli
from morph.dataset.core.atomic_data import batch_from_flat
from morph.dataset.core.vocab import Vocab
from morph.inference.generate import load_checkpoint, make_predict_trainer
from morph.inference.hub import ASSET_FILES, download_pretrained, load_pretrained
from morph.inference.module import load_inference_weights
from morph.inference.registry import get_model
from morph.model.lightning._sampling import SamplingMixin
from morph.model.lightning._setup import SetupMixin
from morph.utils.run_config import (
    gather_rank_outputs,
)

pytestmark = pytest.mark.usefixtures("hub_catalog")
HF_REVISION = "ab9bea92f499f03450ebc99459fce04461504a41"


def test_inference_catalog_lists_published_epochs(hub_catalog):
    assert hub_catalog.models["chembl3d-explicit-h"].epochs == (200, 300, 400)
    assert hub_catalog.models["chembl3d-implicit-h"].epochs == (100, 200, 300)
    assert hub_catalog.default_model == "chembl3d-explicit-h"
    assert get_model().select_epoch() == 400
    assert get_model("chembl3d-implicit-h").select_epoch() == 300


@pytest.mark.parametrize("name,epoch", [("unknown", 200), ("chembl3d-explicit-h", 199)])
def test_unpublished_models_or_epochs_fail_before_weight_download(
    monkeypatch, name, epoch
):

    def unexpected_download(**_kwargs):
        pytest.fail("invalid selection must not use the network")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected_download)
    with pytest.raises(ValueError):
        download_pretrained(name, epoch=epoch)


def _fake_hub_bundle(
    tmp_path, variant="implicit_h", epoch=300, dataset="chembl3d", subfolder=None
):
    directory = tmp_path / (f"{dataset}/{variant}" if subfolder is None else subfolder)
    directory.mkdir(parents=True)
    tokens = {
        "atom_tokens": ["C", "H"] if variant == "explicit_h" else ["C"],
        "edge_tokens": ["<NO_BOND>", "1"],
        "charge_tokens": ["0"],
    }
    for name, values in tokens.items():
        (directory / f"{name}.txt").write_text("\n".join(values) + "\n")
    (directory / "metadata.json").write_text(
        json.dumps({"dataset": dataset, "variant": variant, **tokens})
    )
    (directory / "distributions.pt").write_bytes(b"placeholder")
    (directory / "config.yaml").write_text("{}\n")
    (directory / f"epoch_{epoch:03d}.ckpt").write_bytes(b"placeholder")
    return directory


def test_hub_download_selects_only_one_epoch_and_matching_assets(monkeypatch, tmp_path):

    directory = _fake_hub_bundle(tmp_path)
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    bundle = download_pretrained(
        "chembl3d-implicit-h", epoch=300, local_files_only=True
    )
    assert bundle.directory == directory
    assert bundle.checkpoint.name == "epoch_300.ckpt"
    assert calls[0]["repo_id"] == "maltefranke/morph"
    assert calls[0]["revision"] == HF_REVISION
    assert calls[0]["local_files_only"] is True
    assert calls[0]["allow_patterns"] == ["chembl3d/implicit_h/metadata.json"]
    assert calls[1]["revision"] == tmp_path.name
    assert set(calls[1]["allow_patterns"]) == {
        f"chembl3d/implicit_h/{name}" for name in (*ASSET_FILES, "epoch_300.ckpt")
    }


def test_partial_offline_snapshot_does_not_count_as_complete(monkeypatch, tmp_path):

    directory = _fake_hub_bundle(tmp_path, epoch=100)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **_kw: str(tmp_path)
    )
    with pytest.raises(FileNotFoundError, match=r"epoch_300\.ckpt"):
        download_pretrained("chembl3d-implicit-h", epoch=300, local_files_only=True)
    assert (directory / "epoch_100.ckpt").is_file()


@pytest.mark.parametrize("status", [401, 403, 404])
def test_hub_access_error_explains_authentication(monkeypatch, status):

    def inaccessible(**_kwargs):
        response = httpx.Response(
            status, request=httpx.Request("GET", "https://huggingface.co")
        )
        raise HfHubHTTPError("access denied", response=response)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", inaccessible)
    with pytest.raises(RuntimeError, match="hf auth login"):
        download_pretrained("chembl3d-implicit-h")


def test_missing_offline_cache_has_download_instruction(monkeypatch):

    def no_cache(**_kwargs):
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_cache)
    with pytest.raises(FileNotFoundError, match="morph models download"):
        download_pretrained("chembl3d-explicit-h", local_files_only=True)


def test_local_bundle_rejects_other_hydrogen_representation(tmp_path):

    directory = _fake_hub_bundle(tmp_path)
    with pytest.raises(ValueError, match="Expected explicit_h"):
        load_pretrained("chembl3d-explicit-h", epoch=300, weights_dir=directory)


def test_local_bundle_rejects_reordered_vocabulary(tmp_path):

    directory = _fake_hub_bundle(tmp_path)
    (directory / "edge_tokens.txt").write_text("1\n<NO_BOND>\n")
    with pytest.raises(ValueError, match="edge_tokens does not match"):
        load_pretrained("chembl3d-implicit-h", weights_dir=directory)


def test_local_loader_uses_saved_config_and_disables_dataset_fallback(
    monkeypatch, tmp_path
):

    directory = _fake_hub_bundle(tmp_path)
    cfg = OmegaConf.create(
        {
            "data": {
                "preprocessing": {
                    "root": "/old/cluster/path",
                    "train_dataset": {"_target_": "forbidden"},
                }
            },
            "integrator": {"integrator": {"num_integration_steps": 100}},
            "model": {"module": {"use_ema_for_eval": True}},
            "trainer": {"trainer": {}},
            "logging": {"enabled": True},
        }
    )
    OmegaConf.save(cfg, directory / "config.yaml")
    source = torch.nn.Linear(2, 2)
    torch.save(
        {"state_dict": {f"model.{k}": v for k, v in source.state_dict().items()}},
        directory / "epoch_300.ckpt",
    )

    def build(saved_cfg):
        preprocessing = saved_cfg.data.preprocessing
        assert preprocessing.train_dataset is None
        assert preprocessing.num_workers == 0
        assert preprocessing.distributions_path == str(directory / "distributions.pt")
        assert preprocessing.atom_tokens_path == str(directory / "atom_tokens.txt")
        module = torch.nn.Module()
        module.model = torch.nn.Linear(2, 2)
        return module, "vocab", "distributions"

    def unexpected_download(*_args, **_kwargs):
        pytest.fail("local bundle must not use the network")

    monkeypatch.setattr(inference_module, "build_inference_module", build)
    monkeypatch.setattr(hub, "download_pretrained", unexpected_download)
    model = hub.load_pretrained(
        "chembl3d-implicit-h", weights_dir=tmp_path, steps=12, use_ema=False
    )
    assert not model.module.training
    assert model.config.integrator.integrator.num_integration_steps == 12
    assert model.config.model.module.use_ema_for_eval is False
    torch.testing.assert_close(model.module.model.weight, source.weight)


@pytest.mark.parametrize("use_ema", [True, False])
@pytest.mark.parametrize("compiled", [True, False])
def test_inference_loads_only_selected_weights_with_mmap(
    monkeypatch, tmp_path, use_ema, compiled
):
    current = torch.nn.Linear(2, 2)
    ema = torch.nn.Linear(2, 2)
    with torch.no_grad():
        current.weight.fill_(1.0)
        current.bias.fill_(2.0)
        ema.weight.fill_(3.0)
        ema.bias.fill_(4.0)
    wrapper = "_orig_mod." if compiled else ""
    state = {
        f"{prefix}.{wrapper}{key}": value
        for prefix, net in [("model", current), ("model_ema", ema)]
        for key, value in net.state_dict().items()
    }
    state["training_metric.buffer"] = torch.ones(5)
    checkpoint = tmp_path / "both.ckpt"
    torch.save(
        {"state_dict": state, "optimizer_states": [{"buffer": torch.ones(20)}]},
        checkpoint,
    )
    original_load = torch.load
    calls = []

    def tracked_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    target = torch.nn.Linear(2, 2)
    load_inference_weights(target, checkpoint, use_ema=use_ema)
    expected = ema if use_ema else current
    torch.testing.assert_close(target.weight, expected.weight)
    torch.testing.assert_close(target.bias, expected.bias)
    assert calls[0]["mmap"] is True
    assert calls[0]["map_location"] == "cpu"
    assert not target.training
    assert all(not p.requires_grad for p in target.parameters())


def test_missing_ema_is_not_silently_replaced_by_current_weights(tmp_path):
    model = torch.nn.Linear(2, 2)
    checkpoint = tmp_path / "current-only.ckpt"
    torch.save(
        {"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}},
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="no model_ema weights"):
        load_inference_weights(model, checkpoint)


def test_selected_weights_are_strict_but_unused_branch_is_ignored(tmp_path):
    model = torch.nn.Linear(2, 2)
    checkpoint = tmp_path / "partial.ckpt"
    torch.save(
        {
            "state_dict": {
                "model_ema.weight": model.weight.detach(),
                "model_ema.bias": model.bias.detach(),
                "model.unrelated_key": torch.ones(7),
            }
        },
        checkpoint,
    )
    load_inference_weights(model, checkpoint, use_ema=True)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_inference_weights(model, checkpoint, use_ema=False)


def test_models_cli_lists_without_model_loading_and_reports_invalid_epoch():

    runner = CliRunner()
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0
    assert "chembl3d-explicit-h" in result.output
    assert "100, 200, 300" in result.output
    result = runner.invoke(
        cli, ["models", "download", "chembl3d-implicit-h", "--epoch", "199"]
    )
    assert result.exit_code != 0
    assert "completed training epochs" in result.output


def test_generate_cli_forwards_hub_and_sampling_options(monkeypatch, tmp_path):

    calls = {}

    def generate_to(on_batch, **kwargs):
        calls["sample"] = kwargs
        on_batch([["frame"]], [False])
        return 1

    def load(name, **kwargs):
        calls["load"] = (name, kwargs)
        return SimpleNamespace(
            bundle=SimpleNamespace(spec=get_model(name), checkpoint="weights.ckpt"),
            generate_to=generate_to,
            vocab="vocab",
        )

    monkeypatch.setattr(hub, "load_pretrained", load)
    monkeypatch.setattr(
        generation,
        "record_mol",
        lambda *_a: {"smiles": None, "n_atoms": None, "logp": None},
    )
    result = CliRunner().invoke(
        cli,
        [
            "generate",
            "--model",
            "chembl3d-implicit-h",
            "--epoch",
            "100",
            "--offline",
            "--device",
            "cpu",
            "--no-ema",
            "--steps",
            "12",
            "--batch",
            "1",
            "--batch-size",
            "1",
            "--output",
            str(tmp_path / "samples.sdf"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls["load"][1]["epoch"] == 100
    assert calls["load"][1]["local_files_only"] is True
    assert calls["load"][1]["use_ema"] is False
    assert calls["load"][1]["steps"] == 12
    assert calls["sample"]["n_mols"] == calls["sample"]["batch_size"] == 1
    assert "wrote 0/1 attempts" in result.output


def _sampling_config():
    return OmegaConf.create(
        {
            "data": {"n_atoms_strategy": "uniform", "uniform_n_atoms_padding": 0},
            "trainer": {
                "trainer": {
                    "accelerator": "cpu",
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                }
            },
        }
    )


def _sampling_distributions():
    return SimpleNamespace(
        atom_type_distribution=torch.ones(1),
        edge_type_distribution=torch.ones(1),
        charge_type_distribution=torch.ones(1),
        n_atoms_distribution=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        coordinate_std=1.5,
    )


@pytest.mark.parametrize("strategy", ["uniform", "empirical"])
def test_prior_loader_is_lazy_and_uses_an_isolated_random_stream(monkeypatch, strategy):
    sample = prior.sample_prior_graph
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs["n_atoms"])
        assert len(calls) <= generation.DEFAULT_BATCH_SIZE
        return sample(*args, **kwargs)

    monkeypatch.setattr(prior, "sample_prior_graph", counted)
    torch.manual_seed(42)
    loader = generation._prior_loader(
        _sampling_distributions(), 1_000_000, n_atoms_strategy=strategy
    )
    assert calls == []
    first = next(iter(loader))[0]
    assert len(calls) == generation.DEFAULT_BATCH_SIZE
    assert first.num_graphs == generation.DEFAULT_BATCH_SIZE
    calls.clear()
    torch.randn(113)  # Consumer/model randomness must not change the priors.
    repeated = next(iter(loader))[0]
    torch.testing.assert_close(first.x, repeated.x)
    torch.testing.assert_close(first.a, repeated.a)
    torch.testing.assert_close(first.e, repeated.e)


class _PredictionHarness(pl.LightningModule):
    def __init__(self, *, check_release=False, malformed=False):
        super().__init__()
        self.distributions = _sampling_distributions()
        self.check_release = check_release
        self.malformed = malformed
        self.previous = []
        self.attempted = 0

    def predict_step(self, batch, batch_idx):
        if self.check_release:
            assert all(reference() is None for reference in self.previous)
        size = batch[0].num_graphs
        trajectories = [
            [torch.randn(3) for _ in range(2 if self.predict_return_traj else 1)]
            for _ in range(size)
        ]
        self.previous = [weakref.ref(frame) for traj in trajectories for frame in traj]
        validity = [(self.attempted + i) % 2 == 0 for i in range(size)]
        self.attempted += size
        return {
            "all_mols": trajectories,
            "is_valid": [] if self.malformed else validity,
        }


@pytest.mark.parametrize("return_traj", [False, True])
def test_streaming_predictions_release_frames_and_match_collecting_api(return_traj):
    module = _PredictionHarness(check_release=True)
    batches, flags, values = [], [], []

    def consume(trajectories, is_valid):
        batches.append(len(trajectories))
        flags.extend(is_valid)
        values.extend([[frame.tolist() for frame in traj] for traj in trajectories])

    count = generation.sample_module_batches(
        module,
        _sampling_config(),
        consume,
        n_mols=13,
        batch_size=4,
        seed=27,
        return_traj=return_traj,
    )
    assert count == 13
    assert batches == [4, 4, 4, 1]
    assert all(reference() is None for reference in module.previous)
    assert module.trainer.predict_loop.predictions == []
    collected, valid, vocab, scale = generation.sample_module(
        _PredictionHarness(),
        "vocab",
        _sampling_config(),
        n_mols=13,
        batch_size=4,
        seed=27,
        return_traj=return_traj,
    )
    assert values == [[frame.tolist() for frame in traj] for traj in collected]
    assert flags == valid == [i % 2 == 0 for i in range(13)]
    assert vocab == "vocab"
    assert scale == 1.5


def test_streaming_rejects_misaligned_outputs_and_releases_callback():
    module = _PredictionHarness(malformed=True)
    with pytest.raises(RuntimeError, match="misaligned"):
        generation.sample_module_batches(
            module,
            _sampling_config(),
            lambda *_: pytest.fail("invalid batch delivered"),
            n_mols=2,
        )
    assert not any(
        type(cb).__name__ == "BatchConsumer" for cb in module.trainer.callbacks
    )


def test_streaming_propagates_consumer_errors_and_releases_callback():
    module = _PredictionHarness()

    def fail(*_args):
        raise RuntimeError("consumer failed")

    with pytest.raises(RuntimeError, match="consumer failed"):
        generation.sample_module_batches(module, _sampling_config(), fail, n_mols=2)
    assert not any(
        type(cb).__name__ == "BatchConsumer" for cb in module.trainer.callbacks
    )


def test_cli_streams_sdf_and_trajectory_batches_without_mixing_runs(
    monkeypatch, tmp_path
):
    output = tmp_path / "samples.sdf"

    def generate_to(on_batch, **kwargs):
        assert kwargs["return_traj"] is True
        on_batch([[torch.tensor(0)], [torch.tensor(1)]], [True, False])
        assert output.stat().st_size > 0
        on_batch([[torch.tensor(2)]], [True])
        return 3

    model = SimpleNamespace(
        bundle=SimpleNamespace(spec=get_model(), checkpoint="weights.ckpt"),
        vocab="vocab",
        generate_to=generate_to,
    )
    monkeypatch.setattr(hub, "load_pretrained", lambda *_a, **_kw: model)
    monkeypatch.setattr(generation, "final_mol", lambda *_: Chem.MolFromSmiles("CC"))
    monkeypatch.setattr(
        generation,
        "record_mol",
        lambda *_: {
            "smiles": "CC",
            "n_atoms": 2,
            "logp": 1.0,
        },
    )
    args = [
        "generate",
        "--batch",
        "3",
        "--batch-size",
        "2",
        "--save-trajs",
        "--output",
        str(output),
    ]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert "wrote 2/3 attempts" in result.output
    assert "valid: 2/3 | unique SMILES: 1" in result.output
    assert len(list(Chem.SDMolSupplier(str(output)))) == 2
    files = sorted((tmp_path / "samples_trajs").glob("*.pt"))
    batches = [torch.load(path, weights_only=True) for path in files]
    assert [b["start_index"] for b in batches] == [0, 2]
    assert [v for b in batches for v in b["is_valid"]] == [True, False, True]
    assert [t[0].item() for b in batches for t in b["trajectories"]] == [0, 1, 2]
    original = output.read_bytes()
    result = CliRunner().invoke(cli, args)
    assert result.exit_code != 0
    assert "not empty" in result.output
    assert output.read_bytes() == original


def test_inference_loads_compiled_checkpoint_into_eager_module(tmp_path):
    source = torch.nn.Linear(2, 2)
    module = torch.nn.Linear(2, 2)
    checkpoint = tmp_path / "compiled.ckpt"
    torch.save(
        {
            "state_dict": {
                "_orig_mod.weight": source.weight.detach().clone(),
                "_orig_mod.bias": source.bias.detach().clone(),
            }
        },
        checkpoint,
    )

    load_checkpoint(module, checkpoint)

    torch.testing.assert_close(module.weight, source.weight)
    torch.testing.assert_close(module.bias, source.bias)


def test_inference_loads_eager_checkpoint_into_compiled_module(tmp_path):
    source = torch.nn.Linear(2, 2)
    module = torch.compile(torch.nn.Linear(2, 2))
    checkpoint = tmp_path / "eager.ckpt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)

    load_checkpoint(module, checkpoint)

    torch.testing.assert_close(module._orig_mod.weight, source.weight)
    torch.testing.assert_close(module._orig_mod.bias, source.bias)


def test_lightning_checkpoint_hooks_make_compile_mode_portable():
    class Harness(SetupMixin, torch.nn.Module):
        def __init__(self, compiled):
            super().__init__()
            model = torch.nn.Linear(2, 2)
            self.model = torch.compile(model) if compiled else model

    source = Harness(compiled=True)
    checkpoint = {"state_dict": source.state_dict()}
    source.on_save_checkpoint(checkpoint)
    assert set(checkpoint["state_dict"]) == {"model.weight", "model.bias"}

    target = Harness(compiled=True)
    target.on_load_checkpoint(checkpoint)
    assert set(checkpoint["state_dict"]) == {
        "model._orig_mod.weight",
        "model._orig_mod.bias",
    }
    target.load_state_dict(checkpoint["state_dict"], strict=True)
    torch.testing.assert_close(
        target.model._orig_mod.weight, source.model._orig_mod.weight
    )


def test_predict_trainer_drops_morph_only_validation_setting(monkeypatch):
    monkeypatch.setattr(pl, "Trainer", lambda **kwargs: kwargs)
    cfg = OmegaConf.create(
        {
            "trainer": {
                "trainer": {
                    "num_val_samples": 4096,
                    "limit_val_batches": None,
                    "accelerator": "cpu",
                }
            }
        }
    )

    kwargs = make_predict_trainer(cfg)

    assert "num_val_samples" not in kwargs
    assert "limit_val_batches" not in kwargs


def test_rank_outputs_are_gathered_only_on_zero(monkeypatch):
    monkeypatch.setattr(run_config.dist, "is_available", lambda: True)
    monkeypatch.setattr(run_config.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(run_config.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(run_config.dist, "get_rank", lambda: 0)

    def gather_object(_local, gathered, dst):
        assert dst == 0
        gathered[:] = [["rank-0"], ["rank-1"]]

    monkeypatch.setattr(run_config.dist, "gather_object", gather_object)
    assert gather_rank_outputs(["local"]) == ["rank-0", "rank-1"]

    monkeypatch.setattr(run_config.dist, "get_rank", lambda: 1)

    def gather_nonzero(_local, gathered, dst):
        assert gathered is None
        assert dst == 0

    monkeypatch.setattr(run_config.dist, "gather_object", gather_nonzero)
    assert gather_rank_outputs(["local"]) is None


def test_final_only_prediction_preserves_attempts_validity_and_frame_interface():
    edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    mols = batch_from_flat(
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        ),
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        torch.tensor([1, 1, 0, 0]),
        edges,
        torch.tensor([0, 0, 1, 1]),
        num_graphs=2,
    )

    class Harness(SamplingMixin):
        vocab = Vocab(
            atom_tokens=["C"], edge_tokens=["<NO_BOND>", "1"], charge_tokens=["0"]
        )
        representation = SimpleNamespace(requires_topology=True)
        allow_charged = False
        requested = []

        def sample(self, batch, batch_idx, return_traj):
            self.requested.append(return_traj)
            if return_traj:
                return [[mol.clone(), mol] for mol in mols.clone().to_data_list()]
            return mols.clone()

    harness = Harness()
    final = harness.predict_step(None, 0)
    harness.predict_return_traj = True
    trajectory = harness.predict_step(None, 0)
    assert harness.requested == [False, True]
    assert len(final["all_mols"]) == len(trajectory["all_mols"]) == 2
    assert final["is_valid"] == trajectory["is_valid"]
    for final_frames, full_frames in zip(
        final["all_mols"], trajectory["all_mols"], strict=True
    ):
        assert len(final_frames) == 1
        assert len(full_frames) == 2
        torch.testing.assert_close(final_frames[0].x, full_frames[-1].x)
        torch.testing.assert_close(final_frames[0].e, full_frames[-1].e)
