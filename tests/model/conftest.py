"""Shared Hydra resolvers and isolated Hugging Face catalog fixtures."""

import json

import huggingface_hub
import pytest

import morph.config  # noqa: F401
from morph.inference import load_catalog


@pytest.fixture(autouse=True)
def isolated_morph_cache(tmp_path, monkeypatch):
    """Keep resource downloads and checksum memos out of the user's cache."""
    monkeypatch.setenv("MORPH_CACHE_DIR", str(tmp_path / "morph-cache"))


@pytest.fixture
def catalog_payload():
    """A release catalog fixture; production discovery has no built-in entries."""
    return {
        "schema_version": 1,
        "default_model": "chembl3d-explicit-h",
        "models": [
            {
                "name": f"chembl3d-{variant.replace('_', '-')}",
                "repo_id": "maltefranke/morph",
                "subfolder": f"chembl3d/{variant}",
                "revision": "ab9bea92f499f03450ebc99459fce04461504a41",
                "dataset": "chembl3d",
                "variant": variant,
                "epochs": epochs,
            }
            for variant, epochs in [
                ("explicit_h", [200, 300, 400]),
                ("implicit_h", [100, 200, 300]),
            ]
        ],
    }


@pytest.fixture
def hub_catalog(monkeypatch, tmp_path, catalog_payload):
    """Serve a catalog from the Hub cache without using the network."""
    path = tmp_path / "catalog-cache" / "snapshots" / ("a" * 40) / "catalog.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(catalog_payload))

    def download(**kwargs):
        assert kwargs["filename"] == "catalog.json"
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return load_catalog(path=path)
