"""Preprocessing command selection must be checked before any downloads."""

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "arguments,expected",
    [([], ["chembl3d", "geom", "qm9"]), (["chembl3d"], ["chembl3d"])],
)
def test_preprocess_selects_requested_datasets(arguments, expected, monkeypatch):
    namespace = runpy.run_path(str(Path(__file__).parents[2] / "scripts/preprocess.py"))
    main = namespace["main"]
    calls = []
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setitem(
        main.__globals__, "_prepare_dataset", lambda name, *args: calls.append(name)
    )
    main(arguments)
    assert calls == expected


def test_unknown_dataset_fails_before_processing_valid_names(monkeypatch, capsys):
    namespace = runpy.run_path(str(Path(__file__).parents[2] / "scripts/preprocess.py"))

    def unexpected_processing(*args):
        pytest.fail("Invalid command must not download or process any dataset")

    main = namespace["main"]
    monkeypatch.setitem(main.__globals__, "_prepare_dataset", unexpected_processing)
    with pytest.raises(SystemExit) as error:
        main(["qm9", "unknown"])
    assert error.value.code == 2
    assert "Unknown datasets: unknown" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ([], ["train", "val", "test"]),
        (["--split", "val"], ["val"]),
        (["--split", "val", "--split", "test", "--split", "val"], ["val", "test"]),
    ],
)
def test_scaffold_preprocess_selects_splits_and_forwards_settings(
    arguments, expected, monkeypatch
):
    namespace = runpy.run_path(
        str(Path(__file__).parents[2] / "scripts/preprocess_scaffold.py")
    )
    main, calls = namespace["main"], []
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    def prepare(cfg, split, force):
        assert cfg.data.scaffold_kind == "murcko"
        assert cfg.data.eval_panel.n_scaffolds == 512
        assert force
        calls.append(split)

    monkeypatch.setitem(main.__globals__, "_prepare_split", prepare)
    main(
        [
            "defaults=geom/scaffold_decoration",
            "data.scaffold_kind=murcko",
            "--force",
            *arguments,
        ]
    )
    assert calls == expected


@pytest.mark.parametrize("arguments", [[], ["defaults=geom/geom"]])
def test_scaffold_preprocess_requires_scaffold_config(arguments, monkeypatch, capsys):
    namespace = runpy.run_path(
        str(Path(__file__).parents[2] / "scripts/preprocess_scaffold.py")
    )
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(SystemExit) as error:
        namespace["main"](arguments)
    assert error.value.code == 2
    assert "select a scaffold config" in capsys.readouterr().err
