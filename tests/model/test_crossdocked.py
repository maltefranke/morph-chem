"""CrossDocked chemistry, residue metadata, split integrity, and frame contracts."""

from __future__ import annotations

import io
import json
import pickle
import runpy
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader

from morph.dataset.conditional.pocket import CrossDockedPocketDataset, build, processing
from morph.dataset.core.vocab import Vocab


def _pretrained_assets(path, atoms=("O", "C", "N"), charges=("1", "0", "-1")):
    path.mkdir()
    for name, tokens in (
        ("atom", atoms),
        ("edge", ("<NO_BOND>", "2", "1")),
        ("charge", charges),
    ):
        (path / f"{name}_tokens.txt").write_text("\n".join(tokens) + "\n")
    torch.save(
        dict(
            atom_type_distribution=torch.ones(len(atoms)) / len(atoms),
            edge_type_distribution=torch.ones(3) / 3,
            charge_type_distribution=torch.ones(len(charges)) / len(charges),
            n_atoms_distribution=torch.ones(10) / 10,
            coordinate_std=2.5,
        ),
        path / "distributions.pt",
    )
    return path


def _preprocess_main():
    return runpy.run_path(
        str(Path(__file__).parents[2] / "scripts/preprocess_pocket.py")
    )["main"]


def test_preprocess_pocket_filters_and_validates_without_changing_assets(tmp_path):
    source = tmp_path / "source"
    tree = source / processing.RAW_TREE
    pairs = [
        _write_pair(tree, "train", "allowed", "CO"),
        _write_pair(tree, "train", "unsupported", "C[Si](C)(C)C"),
    ]
    pocket_path = tree / pairs[0][0]
    pocket_path.write_text(
        pocket_path.read_text()
        + _pdb_line(6, "ZN", "ZN", [10, 20, 32], residue="ZN", record="HETATM")
    )
    torch.save({"train": pairs, "test": []}, source / processing.RAW_SPLIT)
    assets = _pretrained_assets(tmp_path / "assets", atoms=("O", "C"))
    before = {p.name: p.read_bytes() for p in assets.iterdir()}
    root = tmp_path / "dataset"
    args = [
        "--root",
        str(root),
        "--source-dir",
        str(source),
        "--pretrained-assets",
        str(assets),
        "--val-frac",
        "0",
    ]
    summary = _preprocess_main()(args)
    assert summary["status"] == "ready"
    assert summary["ligand_vocab"]["atom_tokens"] == ["O", "C"]
    assert summary["coordinate_std"] == 2.5
    assert summary["pocket_vocab"]["atom_tokens"] == ["C", "N", "O", "Zn"]
    report = summary["splits"]["train"]
    assert (report["source_count"], report["retained"], report["dropped"]) == (2, 1, 1)
    assert report["missing_elements"] == ["Si"]
    assert report["validated"] == 1
    assert summary["splits"]["val"]["validated"] == 0
    assert {p.name: p.read_bytes() for p in assets.iterdir()} == before
    assert not (root / "processed" / "atom_tokens.txt").exists()
    # Reuse the completed raw cache even when the source is no longer supplied.
    args[2:4] = []
    assert _preprocess_main()(args)["splits"] == summary["splits"]


def test_preprocess_pocket_records_tokenization_failure_and_can_retry(
    tmp_path, sources
):
    assets = _pretrained_assets(tmp_path / "assets", charges=("1", "0"))
    root = tmp_path / "dataset"
    args = [
        "--root",
        str(root),
        "--source-dir",
        str(sources),
        "--pretrained-assets",
        str(assets),
        "--val-frac",
        "0.25",
    ]
    with pytest.raises(ValueError, match=r"train .*\.sdf: Source charge labels"):
        _preprocess_main()(args)
    report_path = root / "processed" / "finetune_preprocessing_summary.json"
    assert json.loads(report_path.read_text())["status"] == "failed"
    # A complete rerun must replace the failed status and validate every pair.
    (assets / "charge_tokens.txt").write_text("1\n0\n-1\n")
    summary = _preprocess_main()(args)
    assert summary["status"] == "ready"
    assert sum(s["validated"] for s in summary["splits"].values()) == 14
    assert json.loads(report_path.read_text()) == summary


@pytest.mark.parametrize("invalid", ["missing_artifacts", "hydrogen_mode"])
def test_preprocess_pocket_rejects_invalid_assets_before_building(tmp_path, invalid):
    root = tmp_path / "dataset"
    assets = _pretrained_assets(tmp_path / "assets")
    args = ["--root", str(root), "--pretrained-assets", str(assets)]
    if invalid == "missing_artifacts":
        (assets / "distributions.pt").unlink()
        error, match = FileNotFoundError, "Missing pretrained artifact"
    else:
        args += ["--hydrogen-mode", "explicit"]
        error, match = ValueError, "hydrogen-mode must match"
    with pytest.raises(error, match=match):
        _preprocess_main()(args)
    assert not root.exists()


def _pdb_line(
    serial,
    name,
    element,
    pos,
    residue="ALA",
    chain="A",
    number=1,
    icode=" ",
    record="ATOM",
):
    return (
        f"{record:<6}{serial:5d} {name:^4} {residue:>3} {chain}{number:4d}{icode}   "
        f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}{1.0:6.2f}{0.0:6.2f}"
        f"          {element:>2}  \n"
    )


def _write_ligand(path, smiles="C[NH2+]CC(=O)[O-]"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    conformer = mol.GetConformer()
    conformer.SetPositions(conformer.GetPositions() + np.array([10.0, 20.0, 30.0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)
    return mol


def _write_pair(tree, directory, name, smiles="C[NH2+]CC(=O)[O-]"):
    ligand = f"{directory}/{name}.sdf"
    pocket = f"{directory}/{name}_pocket10.pdb"
    _write_ligand(tree / ligand, smiles)
    (tree / pocket).write_text(
        _pdb_line(1, "CA", "C", [10, 20, 30])
        + _pdb_line(2, "N", "N", [11, 20, 30])
        + _pdb_line(3, "O", "O", [10, 22, 30], number=2)
        + _pdb_line(4, "CB", "C", [50, 20, 30], number=2)
        + _pdb_line(5, "H", "H", [10, 20, 31])
    )
    return pocket, ligand


@pytest.fixture
def sources(tmp_path):
    source = tmp_path / "source"
    tree = source / processing.RAW_TREE
    train = [
        _write_pair(tree, f"train_{i}", f"lig_{j}") for i in range(6) for j in range(2)
    ]
    test = [_write_pair(tree, "test", f"lig_{j}") for j in range(3)]
    # An invalid middle test pair must not shift the reported source identities.
    (tree / test[1][1]).write_text("invalid sdf\n")
    torch.save({"train": train, "test": test}, source / processing.RAW_SPLIT)
    return source


@pytest.fixture
def cache(tmp_path, sources):
    root = tmp_path / "dataset"
    build.build_lmdbs(root, sources, val_fraction=0.25)
    return root


def _dataset(root, **options):
    return CrossDockedPocketDataset(root, val_fraction=0.25, **options)


def _tokenized(root, **options):
    # Deliberately reorder token classes: stored bond orders are not token indices.
    vocab = Vocab(["O", "C", "N"], ["<NO_BOND>", "2", "1"], ["1", "0", "-1"])
    options.setdefault("pocket_vocab", vocab)
    return _dataset(
        root,
        vocab=vocab,
        distributions=SimpleNamespace(coordinate_std=2.5),
        **options,
    )


def _identity(item):
    return item


def test_pdb_residue_keys_backbone_heteroatoms_and_hydrogens(tmp_path):
    path = tmp_path / "pocket.pdb"
    path.write_text(
        _pdb_line(1, "CA", "C", [1, 2, 3])
        + _pdb_line(2, "CB", "C", [2, 2, 3])
        + _pdb_line(3, "O", "O", [3, 2, 3], icode="A")
        + _pdb_line(4, "N", "N", [4, 2, 3], chain="B")
        + _pdb_line(5, "ZN", "ZN", [5, 2, 3], residue="ZN", number=2, record="HETATM")
        + _pdb_line(6, "H", "H", [6, 2, 3])
        + _pdb_line(7, "D", "D", [7, 2, 3])
    )
    pocket = processing.parse_pocket(path)
    assert pocket["elem"] == ["C", "C", "O", "N", "Zn"]
    assert pocket["resid"].tolist() == [0, 0, 1, 2, 3]
    assert pocket["backbone"].tolist() == [True, False, True, True, False]
    assert pocket["resname"][-1] == "ZN"
    np.testing.assert_array_equal(pocket["pos"][0], [1, 2, 3])
    path.write_text(_pdb_line(1, "CA", "C", [float("nan"), 0, 0]))
    with pytest.raises(ValueError, match="Non-finite"):
        processing.parse_pocket(path)


@pytest.mark.parametrize("smiles", ["c1ccccc1", "C[NH3+]", "[2H]OC", "[CH3]", "CC.O"])
def test_diffdec_ligands_preserve_heavy_pose_charge_and_kekule_bonds(tmp_path, smiles):
    path = tmp_path / "ligand.sdf"
    _write_ligand(path, smiles)
    mol = Chem.MolFromMolFile(str(path), removeHs=False)
    expected = Chem.RemoveAllHs(mol)
    data = processing.process_ligand_diffdec(path)
    assert data is not None
    assert data.z.tolist() == [a.GetAtomicNum() for a in expected.GetAtoms()]
    assert 1 not in data.z.tolist()
    assert data.charges.tolist() == [a.GetFormalCharge() for a in expected.GetAtoms()]
    torch.testing.assert_close(
        data.pos, torch.tensor(expected.GetConformer().GetPositions()).float()
    )
    assert set(data.edge_attr.tolist()) <= {1, 2, 3}
    bonds = dict(
        zip(
            map(tuple, data.edge_index.t().tolist()),
            data.edge_attr.tolist(),
            strict=True,
        )
    )
    assert all(bonds[j, i] == label for (i, j), label in bonds.items())
    if smiles == "c1ccccc1":
        assert data.edge_attr.tolist().count(2) == 6


def test_explicit_mode_materializes_hydrogens_and_invalid_ligands_are_skipped(tmp_path):
    path = tmp_path / "ligand.sdf"
    _write_ligand(path, "CO")
    assert processing.process_ligand_diffdec(path, "implicit").num_nodes == 2
    assert processing.process_ligand_diffdec(path, "explicit").num_nodes == 6
    path.write_text("broken sdf\n")
    assert processing.process_ligand_diffdec(path) is None
    assert processing.process_ligand_diffdec(tmp_path / "missing.sdf") is None


@pytest.mark.parametrize("filter_ligands", [False, True])
def test_raw_elements_are_unrestricted_and_explicit_filters_only_apply_to_ligands(
    tmp_path, filter_ligands
):
    source = tmp_path / "source"
    tree = source / processing.RAW_TREE
    pairs = [
        _write_pair(tree, "train", f"lig_{i}", smiles)
        for i, smiles in enumerate(["CB(O)O", "C[Se]C", "C[Si](C)(C)C"])
    ]
    pocket_path = tree / pairs[0][0]
    pocket_path.write_text(
        pocket_path.read_text()
        + _pdb_line(6, "ZN", "ZN", [10, 20, 32], residue="ZN", record="HETATM")
    )
    torch.save({"train": pairs, "test": []}, source / processing.RAW_SPLIT)
    # An explicit filter can come from a checkpoint's vocabulary; protein atoms
    # outside that ligand vocabulary must survive unchanged.
    vocab = Vocab(["O", "B", "C"], ["<NO_BOND>", "1"], ["0"])
    root = tmp_path / "dataset"
    options = (
        {"allowed_ligand_atom_symbols": vocab.atom_tokens} if filter_ligands else {}
    )
    dataset = CrossDockedPocketDataset(root, source_dir=source, **options)
    try:
        assert len(dataset) == (1 if filter_ligands else 3)
        assert "Zn" in dataset.get_raw_pocket(0)["elem"]
        atoms = {z for i in range(len(dataset)) for z in dataset.get(i).z.tolist()}
        assert 1 not in atoms
        assert ({5, 6, 8} if filter_ligands else {5, 6, 8, 14, 34}) == atoms
        manifest = json.loads((root / "processed" / build.MANIFEST).read_text())
        status = manifest["splits"]["train"]["status"]
        assert status == ({"ok": 1, "lig_offvocab": 2} if filter_ligands else {"ok": 3})
    finally:
        dataset.close()


def test_protein_vocabulary_is_explicit_and_independent_of_checkpoint_tokens(cache):
    pocket_vocab = Vocab(["N", "C", "O"], ["<NO_BOND>"], ["0"])
    dataset = _tokenized(cache, pocket_vocab=pocket_vocab, max_pocket_atoms=None)
    try:
        ligand, pocket = dataset[0], dataset.get_pocket(0)
        assert pocket.a.tolist() == [1, 0, 2, 1]
        assert pocket.c.tolist() == [0, 0, 0, 0]
        assert ligand.a.tolist() == [1, 2, 1, 1, 0, 0]
        assert ligand.c.tolist() == [1, 0, 1, 1, 1, 2]
        torch.testing.assert_close(ligand.x * 2.5, dataset.get(0).pos)
        torch.testing.assert_close(
            pocket.x * 2.5, torch.from_numpy(dataset.get_raw_pocket(0)["pos"])
        )
        # Pocket-only encoders do not need a ligand output vocabulary.
        dataset.vocab = None
        torch.testing.assert_close(dataset.get_pocket(0).a, pocket.a)
        dataset.pocket_vocab = None
        with pytest.raises(RuntimeError, match="pocket_vocab"):
            dataset.get_pocket(0)
    finally:
        dataset.close()


@pytest.mark.parametrize("kind", ["atom", "bond", "charge"])
def test_checkpoint_tokens_are_never_extended_or_remapped(cache, kind):
    vocab = Vocab(["O", "C", "N"], ["<NO_BOND>", "2", "1"], ["1", "0", "-1"])
    {
        "atom": vocab.atom_tokens,
        "bond": vocab.edge_tokens,
        "charge": vocab.charge_tokens,
    }[kind].pop()
    dataset = _dataset(
        cache,
        vocab=vocab,
        distributions=SimpleNamespace(coordinate_std=2.5),
        filter_by_vocab=False,
    )
    try:
        with pytest.raises(ValueError, match=f"Source {kind} labels"):
            dataset[0]
    finally:
        dataset.close()


def test_grouped_split_and_ordered_metadata(cache, sources):
    manifest = json.loads((cache / "processed" / build.MANIFEST).read_text())
    train, val, test = (_dataset(cache, split=s) for s in ("train", "val", "test"))
    try:
        train_dirs = {p.split("/")[0] for p in train.get_all_ligand_fns()}
        val_dirs = {p.split("/")[0] for p in val.get_all_ligand_fns()}
        assert train_dirs.isdisjoint(val_dirs)
        assert len(train) + len(val) == 12
        assert len(train.get_all_smiles()) == len(train)
        assert len(set(train.get_all_smiles())) == 1
        assert test.get_all_ligand_fns() == ["test/lig_0.sdf", "test/lig_2.sdf"]
        assert [test.get_metadata(i)["source_index"] for i in range(2)] == [0, 2]
        assert manifest["splits"]["test"]["rejected"][0]["reason"] == "lig_unreadable"
        assert manifest["splits"]["test"]["status"] == {"ok": 2, "lig_unreadable": 1}
        expected = build.carve_val(
            torch.load(sources / processing.RAW_SPLIT, weights_only=False)["train"],
            0.25,
        )
        assert train.get_all_ligand_fns() == [pair[1] for pair in expected[0]]
        assert val.get_all_ligand_fns() == [pair[1] for pair in expected[1]]
    finally:
        for dataset in (train, val, test):
            dataset.close()


def test_checkpoint_filter_preserves_pair_and_source_indices(tmp_path, sources):
    tree = sources / processing.RAW_TREE
    _write_ligand(tree / "test/lig_0.sdf", "CBr")
    root = tmp_path / "filtered"
    raw = _dataset(root, source_dir=sources, split="test")
    filtered = _tokenized(root, split="test")
    try:
        assert len(raw) == 2
        assert len(filtered) == 1
        assert filtered.vocab_filter_report == {
            "source_count": 2,
            "retained": 1,
            "dropped": 1,
            "missing_elements": ["Br"],
        }
        assert filtered.get_all_ligand_fns() == ["test/lig_2.sdf"]
        assert filtered.get_all_smiles() == raw.get_all_smiles()[1:]
        assert filtered.get_metadata(0)["source_index"] == 2
        torch.testing.assert_close(filtered.get(0).pos, raw.get(1).pos)
        np.testing.assert_equal(filtered.get_raw_pocket(0), raw.get_raw_pocket(1))
        assert filtered[0].num_nodes == raw.get(1).num_nodes
        restored = pickle.loads(pickle.dumps(filtered))
        assert restored.get_metadata(0)["source_index"] == 2
        restored.close()
        # A broader checkpoint can reuse every original pair in the same cache.
        broader = _dataset(
            root,
            split="test",
            vocab=Vocab(
                ["Br", "O", "C", "N"], ["<NO_BOND>", "1", "2"], ["-1", "0", "1"]
            ),
            distributions=SimpleNamespace(coordinate_std=2.5),
        )
        assert len(broader) == 2
        assert broader.vocab_filter_report["dropped"] == 0
        broader.close()
    finally:
        raw.close()
        filtered.close()


def test_roundtrip_token_mapping_shared_frame_and_defensive_copies(cache, sources):
    dataset = _tokenized(cache, max_pocket_atoms=3)
    try:
        ligand, pocket = dataset[0], dataset.get_pocket(0)
        raw, raw_pocket, metadata = (
            dataset.get(0),
            dataset.get_raw_pocket(0),
            dataset.get_metadata(0),
        )
        sdf = sources / processing.RAW_TREE / metadata["ligand_fn"]
        original = processing.process_ligand_diffdec(sdf)
        torch.testing.assert_close(
            ligand.x * 2.5 + metadata["lig_center"], original.pos
        )
        torch.testing.assert_close(
            pocket.x * 2.5 + metadata["lig_center"],
            torch.tensor([[10.0, 20.0, 30.0], [11.0, 20.0, 30.0], [10.0, 22.0, 30.0]]),
        )
        assert len(raw_pocket["elem"]) == 4
        assert pocket.edge_index.shape == (2, 0)
        assert pocket.c.tolist() == [1] * 3
        assert pocket.backbone.tolist() == [True, True, True]
        assert pocket.residue_index.tolist() == [0, 0, 1]
        assert [dataset.vocab.edge_tokens[i] for i in ligand.e.tolist()] == [
            str(i) for i in raw.edge_attr.tolist()
        ]
        assert [dataset.vocab.charge_tokens[i] for i in ligand.c.tolist()] == [
            str(i) for i in raw.charges.tolist()
        ]
        raw.pos.fill_(999)
        raw.z.fill_(999)
        raw_pocket["pos"].fill(999)
        raw_pocket["elem"][0] = "bad"
        metadata["lig_center"].fill_(999)
        ligand.x.fill_(999)
        pocket.x.fill_(999)
        torch.testing.assert_close(
            dataset.get(0).pos + dataset.get_metadata(0)["lig_center"], original.pos
        )
        assert dataset.get_raw_pocket(0)["elem"][0] == "C"
        assert dataset.get(0).z.max() < 999
        assert dataset.get_pocket(0).x.max() < 999
    finally:
        dataset.close()


def test_preprocessing_uses_only_train_ligands_and_keeps_charge_vocabulary(
    tmp_path, sources
):
    tree = sources / processing.RAW_TREE
    # Bromine is present only in the official test split and must not enter
    # ligand token discovery or training statistics.
    _write_ligand(tree / "test/lig_0.sdf", "CBr")
    root = tmp_path / "processed"
    preprocessing = build.compute_vocab_distributions(
        root, source_dir=sources, val_fraction=0.25
    )
    assert preprocessing.vocab.atom_tokens == ["C", "N", "O"]
    assert set(preprocessing.vocab.charge_tokens) == {"-1", "0", "1"}
    raw = _dataset(root)
    try:
        coordinates = torch.cat([raw.get(i).pos for i in range(len(raw))])
        assert preprocessing.distributions.coordinate_std == pytest.approx(
            float(coordinates.std())
        )
        assert preprocessing.distributions.n_atoms_distribution.argmax().item() == 6
        assert (root / "processed" / "distributions.pt").is_file()
    finally:
        raw.close()


def test_cached_reads_do_not_need_sources_and_reject_changed_build_settings(
    cache, monkeypatch
):
    def unexpected_source(*args):
        pytest.fail("Completed cache must not revisit raw sources")

    monkeypatch.setattr(build, "ensure_crossdocked_raw", unexpected_source)
    raw = _dataset(cache)
    try:
        assert len(raw) > 0
        with pytest.raises(RuntimeError, match="vocab"):
            raw[0]
        for index in (-1, len(raw)):
            with pytest.raises(IndexError):
                raw.get(index)
        for options in ({"hydrogen_mode": "explicit"}, {"seed": 7}, {"max_atoms": 10}):
            with pytest.raises(ValueError, match="different CrossDocked"):
                _dataset(cache, **options)
    finally:
        raw.close()


@pytest.mark.parametrize("context", ["spawn", "fork"])
def test_multiple_readers_and_dataloader_workers(cache, context):
    first, second = _tokenized(cache), _tokenized(cache)
    try:
        expected = first[0]
        torch.testing.assert_close(second[0].x, expected.x)
        assert first._reader is second._reader
        restored = pickle.loads(pickle.dumps(first))
        assert restored._reader is None
        restored.close()
        loader = DataLoader(
            first,
            batch_size=None,
            num_workers=2,
            multiprocessing_context=context,
            collate_fn=_identity,
        )
        samples = list(loader)
        assert len(samples) == len(first)
        torch.testing.assert_close(samples[0].x, expected.x)
        first.close()
        torch.testing.assert_close(second[0].x, expected.x)
    finally:
        first.close()
        second.close()


def test_parallel_build_matches_serial(cache, sources, tmp_path):
    other = tmp_path / "parallel"
    build.build_lmdbs(
        other, sources, val_fraction=0.25, preprocess_workers=2, preprocess_chunk_size=1
    )
    for split in ("train", "val", "test"):
        serial, parallel = _dataset(cache, split=split), _dataset(other, split=split)
        try:
            assert serial.get_all_ligand_fns() == parallel.get_all_ligand_fns()
            for i in range(len(serial)):
                torch.testing.assert_close(serial.get(i).pos, parallel.get(i).pos)
                np.testing.assert_array_equal(
                    serial.get_raw_pocket(i)["pos"], parallel.get_raw_pocket(i)["pos"]
                )
        finally:
            serial.close()
            parallel.close()


def test_build_failure_does_not_publish_partial_cache(tmp_path, sources, monkeypatch):
    root = tmp_path / "failed"
    original = build._build_split

    def fail_after_train(name, *args):
        if name == "val":
            raise RuntimeError("interrupted build")
        return original(name, *args)

    monkeypatch.setattr(build, "_build_split", fail_after_train)
    with pytest.raises(RuntimeError, match="interrupted"):
        build.build_lmdbs(root, sources, val_fraction=0.25)
    assert not (root / "processed").exists()
    monkeypatch.setattr(build, "_build_split", original)
    assert (
        build.build_lmdbs(root, sources, val_fraction=0.25)["splits"]["train"]["count"]
        > 0
    )


def test_source_discovery_and_archive_extraction(tmp_path, sources):
    raw = tmp_path / "raw"
    tree, split = processing.ensure_crossdocked_raw(raw, sources / processing.RAW_TREE)
    assert tree == sources / processing.RAW_TREE
    assert split == sources / processing.RAW_SPLIT
    assert not raw.exists()
    with pytest.raises(FileNotFoundError, match="source_dir"):
        processing.ensure_crossdocked_raw(raw, tmp_path / "missing")
    archive_source = tmp_path / "archive"
    archive_source.mkdir()
    (archive_source / processing.RAW_SPLIT).write_bytes(split.read_bytes())
    with tarfile.open(archive_source / processing.RAW_ARCHIVE, "w:gz") as archive:
        archive.add(tree, arcname=processing.RAW_TREE)
    extracted, found_split = processing.ensure_crossdocked_raw(raw, archive_source)
    assert extracted == raw / processing.RAW_TREE
    assert found_split == archive_source / processing.RAW_SPLIT
    assert (extracted / "test/lig_0.sdf").is_file()


def test_archive_rejects_traversal(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / processing.RAW_SPLIT).touch()
    with tarfile.open(source / processing.RAW_ARCHIVE, "w:gz") as archive:
        member = tarfile.TarInfo("../../escaped")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(tarfile.FilterError):
        processing.ensure_crossdocked_raw(tmp_path / "raw", source)
    assert not (tmp_path / "escaped").exists()


def test_download_failure_does_not_publish_partial_file(tmp_path, monkeypatch):
    destination = tmp_path / processing.RAW_SPLIT

    def download(**kwargs):
        with open(kwargs["output"], "wb") as stream:
            stream.write(b"partial download")
        return None

    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=download))
    with pytest.raises(OSError, match="Failed to download"):
        processing._download_raw(processing.RAW_SPLIT, destination)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_default_download_fetches_missing_sources_once(tmp_path, sources, monkeypatch):
    monkeypatch.delenv("MORPH_CROSSDOCKED_RAW", raising=False)
    contents = io.BytesIO()
    with tarfile.open(fileobj=contents, mode="w:gz") as archive:
        archive.add(sources / processing.RAW_TREE, arcname=processing.RAW_TREE)
    calls = []

    def download(name, destination):
        calls.append(name)
        destination.write_bytes(
            contents.getvalue()
            if name == processing.RAW_ARCHIVE
            else (sources / processing.RAW_SPLIT).read_bytes()
        )

    monkeypatch.setattr(processing, "_download_raw", download)
    raw = tmp_path / "downloaded"
    first = processing.ensure_crossdocked_raw(raw)
    assert first == processing.ensure_crossdocked_raw(raw)
    assert calls == [processing.RAW_ARCHIVE, processing.RAW_SPLIT]


def test_split_leakage_and_unusable_training_split_are_rejected(tmp_path, sources):
    path = sources / processing.RAW_SPLIT
    splits = torch.load(path, weights_only=False)
    splits["test"] = splits["train"][:1]
    torch.save(splits, path)
    root = tmp_path / "leakage"
    with pytest.raises(ValueError, match="overlap"):
        build.build_lmdbs(root, sources)
    assert not (root / "processed").exists()
    splits["test"] = []
    splits["train"] = splits["train"][:2]
    torch.save(splits, path)
    with pytest.raises(ValueError, match="no training"):
        build.build_lmdbs(root, sources, val_fraction=0.5)


def test_unknown_pocket_tokens_and_no_trimming(cache):
    dataset = _tokenized(cache, max_pocket_atoms=None)
    try:
        assert dataset.get_pocket(0).num_nodes == 4
        dataset.pocket_vocab.atom_tokens = ["C", "O"]
        with pytest.raises(ValueError, match=r"Pocket elements.*N"):
            dataset.get_pocket(0)
    finally:
        dataset.close()


@pytest.mark.parametrize("compute_statistics", [False, True])
def test_cli_only_discovers_tokens_when_requested(
    tmp_path, sources, monkeypatch, compute_statistics
):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    root = tmp_path / "cli"
    args = ["--root", str(root), "--source-dir", str(sources), "--val-frac", "0.25"]
    if compute_statistics:
        args.append("--compute-statistics")
    build.main(args)
    if compute_statistics:
        assert (root / "processed" / "atom_tokens.txt").read_text().splitlines() == [
            "C",
            "N",
            "O",
        ]
    else:
        assert not (root / "processed" / "atom_tokens.txt").exists()
        assert not (root / "processed" / "distributions.pt").exists()
    assert all(
        (root / "processed" / f"{split}_data.lmdb").is_dir()
        for split in ("train", "val", "test")
    )


def test_cli_rejects_bad_options_before_download(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    for args in (["--workers", "-1"], ["--val-frac", "1"], ["--max-atoms", "0"]):
        with pytest.raises(SystemExit):
            build.main(args)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(SystemExit):
        build.main([])
