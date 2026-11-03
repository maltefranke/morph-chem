"""Remote catalogs discover new model names without Python registry edits."""

import json
import subprocess
import sys

import httpx
import huggingface_hub
import pytest
from click.testing import CliRunner
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from morph.cli import cli
from morph.inference import (
    ModelSpec,
    download_pretrained,
    load_catalog,
    load_pretrained,
)
from morph.inference.catalog import DEFAULT_CATALOG_REPO
from morph.inference.registry import get_model


def _write_catalog(tmp_path, payload, revision="b" * 40):
    path = tmp_path / "snapshots" / revision / "catalog.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_discovery_downloads_only_catalog_with_auth_and_cache(
    monkeypatch, tmp_path, catalog_payload
):
    path = _write_catalog(tmp_path, catalog_payload)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    result = load_catalog(cache_dir=tmp_path, local_files_only=True, token=False)
    assert calls == [
        dict(
            repo_id=DEFAULT_CATALOG_REPO,
            filename="catalog.json",
            revision="main",
            cache_dir=tmp_path,
            local_files_only=True,
            token=False,
        )
    ]
    assert result.revision == "b" * 40
    assert result.models["chembl3d-implicit-h"].revision != result.revision
    assert result.path == path


def test_new_remote_model_is_discovered_and_resolves_without_code_changes(
    monkeypatch, tmp_path, catalog_payload
):
    first = _write_catalog(tmp_path, catalog_payload)
    updated = json.loads(json.dumps(catalog_payload))
    updated["models"].append(
        dict(
            name="qm9-v2",
            repo_id="new-owner/new-models",
            subfolder="qm9/explicit_h",
            revision="c" * 40,
            dataset="qm9",
            variant="explicit_h",
            epochs=[15, 45],
        )
    )
    updated["default_model"] = "qm9-v2"
    second = _write_catalog(tmp_path, updated, "d" * 40)
    current_path = [first]
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda **_kw: str(current_path[0])
    )
    assert "qm9-v2" not in load_catalog().models
    current_path[0] = second
    catalog = load_catalog()
    assert catalog.revision == "d" * 40
    spec = get_model(catalog=catalog)
    assert spec.name == "qm9-v2"
    assert spec.repo_id == "new-owner/new-models"
    assert spec.revision == "c" * 40
    assert spec.select_epoch() == 45
    assert get_model("qm9-v2") == spec
    listed = CliRunner().invoke(cli, ["models", "list"])
    assert listed.exit_code == 0, listed.output
    assert "qm9-v2" in listed.output
    assert "15, 45" in listed.output
    assert "Default model: qm9-v2" in listed.output


def test_alias_download_uses_the_catalog_entry_release(monkeypatch, tmp_path):
    payload = dict(
        schema_version=1,
        models=[
            dict(
                name="geom-test",
                repo_id="owner/geom",
                subfolder="v2/heavy",
                revision="e" * 40,
            )
        ],
        default_model="geom-test",
    )
    path = _write_catalog(tmp_path, payload)
    catalog = load_catalog(path=path)
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        raise LocalEntryNotFoundError("no weight cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    with pytest.raises(FileNotFoundError, match="local cache"):
        download_pretrained("geom-test", catalog=catalog, local_files_only=True)
    assert calls[0]["repo_id"] == "owner/geom"
    assert calls[0]["revision"] == "e" * 40
    assert calls[0]["allow_patterns"] == ["v2/heavy/metadata.json"]


def test_explicit_sources_and_local_catalog_do_not_contact_catalog_hub(
    monkeypatch, tmp_path, catalog_payload
):
    def no_network(**_kwargs):
        pytest.fail("explicit source must not query the Hub catalog")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_network)
    spec = ModelSpec("local-spec", repo_id="owner/repo")
    assert get_model(spec) is spec
    assert get_model("owner/repo").repo_id == "owner/repo"
    path = _write_catalog(tmp_path, catalog_payload)
    catalog = load_catalog(path=path)
    assert get_model("chembl3d-explicit-h", catalog=catalog).select_epoch() == 400
    listed = CliRunner().invoke(
        cli, ["models", "list", "--catalog-file", str(path), "--offline"]
    )
    assert listed.exit_code == 0, listed.output
    assert "chembl3d-implicit-h" in listed.output


def test_catalog_source_env_and_explicit_override(
    monkeypatch, tmp_path, catalog_payload
):
    monkeypatch.setenv("MORPH_CATALOG_REPO", "custom/models")
    monkeypatch.setenv("MORPH_CATALOG_REVISION", "stable")
    path = _write_catalog(tmp_path, catalog_payload)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    load_catalog()
    load_catalog("explicit/models", revision="v2")
    assert [(c["repo_id"], c["revision"]) for c in calls] == [
        ("custom/models", "stable"),
        ("explicit/models", "v2"),
    ]


def test_offline_missing_catalog_does_not_fall_back_to_hidden_models(monkeypatch):
    def missing(**_kwargs):
        raise LocalEntryNotFoundError("missing")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    with pytest.raises(FileNotFoundError, match="No cached catalog"):
        get_model("chembl3d-explicit-h", local_files_only=True)
    result = CliRunner().invoke(cli, ["models", "list", "--offline"])
    assert result.exit_code != 0
    assert "morph models list" in result.output


def test_named_local_weights_use_only_cached_catalog(
    monkeypatch, tmp_path, catalog_payload
):
    path = _write_catalog(tmp_path, catalog_payload)
    calls = []

    def cached(**kwargs):
        calls.append(kwargs)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", cached)
    with pytest.raises(FileNotFoundError, match="Incomplete model bundle"):
        load_pretrained("chembl3d-implicit-h", weights_dir=tmp_path, cache_dir=tmp_path)
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True
    assert calls[0]["cache_dir"] == tmp_path


@pytest.mark.parametrize("status", [401, 403, 404])
def test_catalog_errors_explain_source_and_authentication(monkeypatch, status):
    def inaccessible(**_kwargs):
        response = httpx.Response(
            status, request=httpx.Request("GET", "https://huggingface.co")
        )
        raise HfHubHTTPError("unavailable", response=response)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", inaccessible)
    with pytest.raises(RuntimeError, match="hf auth login"):
        load_catalog()


@pytest.mark.parametrize(
    "broken",
    [
        {"schema_version": 2, "models": []},
        {"schema_version": True, "models": []},
        {"schema_version": 1, "models": [], "default_model": "missing"},
        {"schema_version": 1, "models": {}},
        {"schema_version": 1, "models": [], "typo": True},
    ],
)
def test_invalid_catalog_schema_is_rejected(tmp_path, broken):
    path = _write_catalog(tmp_path, broken)
    with pytest.raises(ValueError):
        load_catalog(path=path)


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "owner/repo"},
        {"repo_id": None},
        {"revision": None},
        {"subfolder": "../elsewhere"},
        {"epochs": [True]},
        {"epochs": [5, 5]},
        {"epochs": "100"},
        {"variant": "unknown"},
        {"repo": "typo"},
    ],
)
def test_invalid_model_entries_fail_before_weight_download(
    tmp_path, catalog_payload, changes
):
    catalog_payload["models"][0].update(changes)
    path = _write_catalog(tmp_path, catalog_payload)
    with pytest.raises(ValueError):
        load_catalog(path=path)


def test_duplicate_aliases_are_rejected(tmp_path, catalog_payload):
    catalog_payload["models"].append(catalog_payload["models"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        load_catalog(path=_write_catalog(tmp_path, catalog_payload))


def test_empty_catalog_lists_no_models_and_requires_explicit_selection(tmp_path):
    path = _write_catalog(tmp_path, dict(schema_version=1, models=[]))
    catalog = load_catalog(path=path)
    with pytest.raises(ValueError, match="no default model"):
        get_model(catalog=catalog)
    result = CliRunner().invoke(cli, ["models", "list", "--catalog-file", str(path)])
    assert result.exit_code == 0, result.output
    assert "Default model: (none)" in result.output
    assert "chembl" not in result.output


@pytest.mark.parametrize("operation", ["list", "download", "generate"])
def test_cli_catalog_source_options_reach_discovery(
    monkeypatch, tmp_path, catalog_payload, operation
):
    path = _write_catalog(tmp_path, catalog_payload)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(path)

    def stop_weights(**_kwargs):
        raise LocalEntryNotFoundError("no weights cached")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", stop_weights)
    command = {
        "list": ["models", "list"],
        "download": ["models", "download", "chembl3d-implicit-h"],
        "generate": ["generate", "--model", "chembl3d-implicit-h"],
    }[operation]
    result = CliRunner().invoke(
        cli,
        [
            *command,
            "--catalog-repo",
            "custom/models",
            "--catalog-revision",
            "stable",
            "--cache-dir",
            str(tmp_path),
            "--offline",
        ],
    )
    assert result.exit_code == (0 if operation == "list" else 1), result.output
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "custom/models"
    assert calls[0]["revision"] == "stable"
    assert calls[0]["local_files_only"] is True
    assert calls[0]["cache_dir"] == str(tmp_path)


def test_discovery_does_not_import_torch_or_download_weights(tmp_path, catalog_payload):
    path = _write_catalog(tmp_path, catalog_payload)
    # Isolated interpreter: the rest of the model test suite imports Torch.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from morph.inference import load_catalog; import sys; "
            "load_catalog(path=sys.argv[1]); assert 'torch' not in sys.modules",
            str(path),
        ],
        check=True,
    )
