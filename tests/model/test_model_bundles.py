"""Unregistered repositories and local bundles share dataset-independent loading."""

import json
from dataclasses import asdict
from pathlib import Path

import huggingface_hub
import pytest
import torch
from click.testing import CliRunner
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from morph.cli import cli
from morph.dataset.core.preprocessing import Preprocessing
from morph.dataset.core.vocab import Distributions, Vocab
from morph.inference import ModelSpec, download_pretrained, load_pretrained
from morph.inference.bundles import ASSET_FILES, local_bundle
from morph.inference.module import InferenceModule
from morph.inference.registry import get_model
from morph.model.build import prepare_model_data

pytestmark = pytest.mark.usefixtures("hub_catalog")
HF_REVISION = "ab9bea92f499f03450ebc99459fce04461504a41"


def _bundle(directory, dataset="geom", variant="implicit_h"):
    directory.mkdir(parents=True)
    metadata = {
        "dataset": dataset,
        "variant": variant,
        "atom_tokens": ["C", "H", "N"] if variant == "explicit_h" else ["C", "N"],
        "edge_tokens": ["<NO_BOND>", "1", "2"],
        "charge_tokens": ["0"] if dataset == "qm9" else ["-1", "0", "1"],
        # Deliberately unsorted, with filenames unrelated to the epoch number.
        "checkpoints": [
            {"completed_epochs": 75, "filename": "weights/final.ckpt"},
            {"completed_epochs": 25, "filename": "early.ckpt"},
        ],
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    for name in ASSET_FILES:
        if name.endswith("_tokens.txt"):
            (directory / name).write_text("\n".join(metadata[name[:-4]]) + "\n")
        elif name != "metadata.json":
            (directory / name).write_text("{}")
    for entry in metadata["checkpoints"]:
        path = directory / entry["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    return metadata


def _patch_snapshot(monkeypatch, snapshot, exported=None):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(exported if kwargs.get("local_dir") else snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    return calls


@pytest.mark.parametrize("subfolder", ["", "models/geom/small"])
def test_repository_metadata_selects_epoch_and_pins_all_assets(
    monkeypatch, tmp_path, subfolder
):
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    _bundle(snapshot / subfolder)
    calls = _patch_snapshot(monkeypatch, snapshot)
    result = download_pretrained(
        "someone/models", subfolder=subfolder, revision="release-v2"
    )
    prefix = f"{subfolder}/" if subfolder else ""
    assert result.epoch == 75
    assert result.checkpoint.name == "final.ckpt"
    assert result.spec.dataset == "geom"
    assert result.spec.variant == "implicit_h"
    assert result.revision == "a" * 40
    assert calls[0]["revision"] == "release-v2"
    assert calls[0]["allow_patterns"] == [f"{prefix}metadata.json"]
    assert calls[1]["revision"] == result.revision
    assert set(calls[1]["allow_patterns"]) == {
        f"{prefix}{name}" for name in (*ASSET_FILES, "weights/final.ckpt")
    }


def test_local_export_uses_resolved_commit_not_folder_name(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshots" / ("b" * 40)
    export = tmp_path / "my-models"
    _bundle(snapshot / "qm9", dataset="qm9", variant="explicit_h")
    _bundle(export / "qm9", dataset="qm9", variant="explicit_h")
    calls = _patch_snapshot(monkeypatch, snapshot, export)
    result = download_pretrained("someone/models", subfolder="qm9", local_dir=export)
    assert result.directory == export / "qm9"
    assert result.revision == "b" * 40
    assert "local_dir" not in calls[0]
    assert calls[1]["local_dir"] == export


def test_explicit_model_spec_needs_no_catalog_entry(monkeypatch, tmp_path):
    snapshot = tmp_path / ("c" * 40)
    _bundle(snapshot / "release", dataset="qm9", variant="explicit_h")
    calls = _patch_snapshot(monkeypatch, snapshot)
    spec = ModelSpec(
        "my-qm9",
        repo_id="someone/qm9",
        subfolder="release",
        revision="v1",
        dataset="QM9",
        variant="explicit_h",
        epochs=(25, 75),
    )
    result = download_pretrained(spec, epoch=25, local_files_only=True, token=False)
    assert result.checkpoint.name == "early.ckpt"
    assert all(call["local_files_only"] and call["token"] is False for call in calls)
    assert get_model(spec) is spec


def test_subfolder_and_revision_overrides_do_not_retain_old_epoch_catalog():
    spec = get_model("chembl3d-explicit-h", subfolder="geom/implicit_h", revision="v2")
    assert spec.subfolder == "geom/implicit_h"
    assert spec.dataset is None
    assert spec.variant is None
    assert spec.epochs == ()
    assert spec.revision == "v2"
    assert get_model("chembl3d-explicit-h").revision == HF_REVISION


@pytest.mark.parametrize("filename", ["../escape.ckpt", "/absolute.ckpt", "*.ckpt"])
def test_unsafe_checkpoint_paths_fail_before_network(monkeypatch, filename):
    def no_network(**_kwargs):
        pytest.fail("invalid input must not download anything")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_network)
    with pytest.raises(ValueError, match="relative bundle path"):
        download_pretrained("someone/models", filename=filename)


def test_metadata_path_and_unavailable_epoch_fail_before_weight_download(
    monkeypatch, tmp_path
):
    snapshot = tmp_path / ("d" * 40)
    metadata = _bundle(snapshot)
    calls = _patch_snapshot(monkeypatch, snapshot)
    with pytest.raises(ValueError, match="requested 50"):
        download_pretrained("someone/models", epoch=50)
    assert len(calls) == 1
    metadata["checkpoints"][0]["filename"] = "../../outside.ckpt"
    (snapshot / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="relative bundle path"):
        download_pretrained("someone/models")
    assert len(calls) == 2


def test_remote_filename_override_and_mutually_exclusive_selection(
    monkeypatch, tmp_path
):
    snapshot = tmp_path / ("e" * 40)
    _bundle(snapshot)
    (snapshot / "best.ckpt").write_bytes(b"placeholder")
    calls = _patch_snapshot(monkeypatch, snapshot)
    result = download_pretrained("someone/models", filename="best.ckpt")
    assert result.epoch is None
    assert result.checkpoint == snapshot / "best.ckpt"
    with pytest.raises(ValueError, match="either epoch or filename"):
        download_pretrained("someone/models", epoch=75, filename="best.ckpt")
    assert len(calls) == 2


@pytest.mark.parametrize("identity", [{"dataset": "qm9"}, {"variant": "explicit_h"}])
def test_registered_identity_is_validated_before_weights(
    monkeypatch, tmp_path, identity
):
    snapshot = tmp_path / ("f" * 40)
    _bundle(snapshot)
    calls = _patch_snapshot(monkeypatch, snapshot)
    spec = ModelSpec("expected", repo_id="someone/models", **identity)
    with pytest.raises(ValueError, match="Expected"):
        download_pretrained(spec)
    assert len(calls) == 1


def test_partial_generic_cache_cannot_select_another_checkpoint(monkeypatch, tmp_path):
    snapshot = tmp_path / ("1" * 40)
    _bundle(snapshot)
    (snapshot / "weights/final.ckpt").unlink()
    _patch_snapshot(monkeypatch, snapshot)
    with pytest.raises(FileNotFoundError, match=r"final\.ckpt"):
        download_pretrained("someone/models", local_files_only=True)


def test_local_subfolder_and_filename_selection(tmp_path):
    directory = tmp_path / "geom/implicit_h"
    _bundle(directory)
    spec = ModelSpec("local", subfolder="geom/implicit_h")
    result = local_bundle(spec, tmp_path, filename="early.ckpt")
    assert result.epoch == 25
    assert result.spec.dataset == "geom"
    assert result.directory == directory


def test_explicit_subfolder_takes_precedence_over_repository_root_bundle(tmp_path):
    root = tmp_path / "models"
    _bundle(root, dataset="qm9", variant="explicit_h")
    _bundle(root / "geom", dataset="geom")
    result = local_bundle(ModelSpec("local", subfolder="geom"), root)
    assert result.spec.dataset == "geom"
    assert result.directory == root / "geom"


def test_local_config_cannot_silently_change_dataset_identity(tmp_path):
    directory = tmp_path / "bundle"
    _bundle(directory, dataset="qm9", variant="explicit_h")
    (directory / "config.yaml").write_text("data:\n  dataset_name: geom\n")
    with pytest.raises(ValueError, match="dataset_name does not match"):
        load_pretrained(weights_dir=directory)


@pytest.mark.parametrize("operation", ["download", "generate"])
def test_cli_accepts_generic_source_options(monkeypatch, tmp_path, operation):
    snapshot = tmp_path / ("2" * 40)
    _bundle(snapshot / "geom")
    calls = _patch_snapshot(monkeypatch, snapshot)
    if operation == "download":
        result = CliRunner().invoke(
            cli,
            [
                "models",
                "download",
                "someone/models",
                "--subfolder",
                "geom",
                "--revision",
                "v2",
                "--filename",
                "early.ckpt",
                "--offline",
            ],
        )
        assert result.exit_code == 0, result.output
        assert str(snapshot / "geom") in result.output
        assert calls[1]["allow_patterns"][-1] == "geom/early.ckpt"
    else:
        # Stop before constructing Torch models, after verifying source resolution.
        def loaded(name, **kwargs):
            assert name == "someone/models"
            assert kwargs["subfolder"] == "geom"
            assert kwargs["revision"] == "v2"
            assert kwargs["filename"] == "early.ckpt"
            raise RuntimeError("source options received")

        monkeypatch.setattr("morph.inference.hub.load_pretrained", loaded)
        result = CliRunner().invoke(
            cli,
            [
                "generate",
                "--model",
                "someone/models",
                "--subfolder",
                "geom",
                "--revision",
                "v2",
                "--filename",
                "early.ckpt",
            ],
        )
        assert "source options received" in result.output


@pytest.mark.parametrize(
    "dataset,variant",
    [
        ("qm9", "explicit_h"),
        ("geom", "explicit_h"),
        ("geom", "implicit_h"),
    ],
)
@pytest.mark.parametrize("use_ema", [True, False])
def test_real_dataset_configs_load_local_bundles_without_training_data(
    monkeypatch, tmp_path, dataset, variant, use_ema
):
    directory = tmp_path / "portable-bundle"
    metadata = _bundle(directory, dataset=dataset, variant=variant)
    vocab = Vocab(
        *(metadata[field] for field in ("atom_tokens", "edge_tokens", "charge_tokens"))
    )
    distributions = Distributions(
        torch.ones(len(vocab.atom_tokens)) / len(vocab.atom_tokens),
        torch.ones(len(vocab.edge_tokens)) / len(vocab.edge_tokens),
        torch.ones(len(vocab.charge_tokens)) / len(vocab.charge_tokens),
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
        1.5,
    )
    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[2] / "configs"), version_base="1.1"
    ):
        cfg = compose(
            config_name="default",
            overrides=[
                f"defaults={dataset}/{dataset}",
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
                "data.uniform_n_atoms_padding=0",
            ],
        )
    cfg.data.root_path = "."
    if "hydrogen_mode" in cfg.data:
        cfg.data.hydrogen_mode = "implicit" if variant == "implicit_h" else "explicit"
    cfg.data.datamodule = None
    cfg.data.preprocessing.train_dataset = None
    OmegaConf.save(cfg, directory / "config.yaml")
    torch.save(asdict(distributions), directory / "distributions.pt")
    data = prepare_model_data(cfg, vocab, distributions)
    source = InferenceModule(cfg, data).eval()
    current = {
        name: tensor.detach().clone()
        for name, tensor in source.model.state_dict().items()
    }
    ema = {name: tensor.clone() for name, tensor in current.items()}
    first_parameter = next(iter(dict(source.model.named_parameters())))
    ema[first_parameter].add_(0.01)
    torch.save(
        {
            "state_dict": {
                **{f"model.{key}": value for key, value in current.items()},
                **{f"model_ema.{key}": value for key, value in ema.items()},
            }
        },
        directory / "weights/final.ckpt",
    )

    def no_dataset(*_args, **_kwargs):
        pytest.fail("portable loading must not access a training dataset or the Hub")

    monkeypatch.setattr(Preprocessing, "_ensure_train_dataset", no_dataset)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_dataset)
    # No name or catalog entry: dataset identity comes exclusively from the bundle.
    loaded = load_pretrained(weights_dir=directory, use_ema=use_ema, steps=2)
    assert loaded.bundle.epoch == 75
    assert loaded.bundle.spec.dataset == dataset
    assert loaded.vocab == vocab
    assert loaded.module.allow_charged is (dataset == "geom")
    assert not hasattr(loaded.module, "model_ema")
    assert loaded.distributions.coordinate_std == 1.5
    torch.testing.assert_close(
        loaded.module.model.state_dict(), ema if use_ema else current, rtol=0, atol=0
    )
    trajectories, valid, _, scale = loaded.generate(n_mols=1, batch_size=1)
    assert len(trajectories) == len(valid) == 1
    assert scale == 1.5
