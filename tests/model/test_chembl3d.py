"""ChEMBL3D format translation and LMDB/preprocessing integration contracts."""

import hashlib
import io
import pickle
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset

from morph.dataset.core.lmdb import (
    mol_from_bytes,
    mol_to_bytes,
)
from morph.dataset.core.vocab import Vocab
from morph.dataset.unconditional import chembl3d
from morph.dataset.unconditional.chembl3d import (
    ATOM_NUMBERS,
    ChEMBL3D,
    FlowMatchingChEMBL3DDataset,
    convert_record,
)
from morph.utils.rdkit_utils import BOND_IDX_MAP, mol_is_valid, smiles_from_mol


def _loqi_record(smiles="C[NH3+]", stereo_edges=True):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    encoder = {z: i for i, z in enumerate(ATOM_NUMBERS.tolist())}
    pairs, labels = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pairs.extend(((i, j), (j, i)))
        labels.extend([BOND_IDX_MAP[bond.GetBondType()]] * 2)
    if stereo_edges:
        pairs.extend([(0, 1)] * 4)
        labels.extend([5, 6, 7, 8])
    return Data(
        x=torch.tensor(
            [encoder[a.GetAtomicNum()] for a in mol.GetAtoms()], dtype=torch.uint8
        ),
        pos=torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float),
        charges=torch.tensor(
            [a.GetFormalCharge() for a in mol.GetAtoms()], dtype=torch.int8
        ),
        edge_index=torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t(),
        edge_attr=torch.tensor(labels, dtype=torch.uint8),
        mol=mol,
        smiles=Chem.MolToSmiles(mol),
    )


def _save_split(directory, records, split="train", modern=False):
    directory.mkdir(parents=True, exist_ok=True)
    data, slices = InMemoryDataset.collate(records)
    payload = (data.to_dict(), slices, Data) if modern else (data, slices)
    torch.save(payload, directory / f"{split}_h.pt")


@pytest.mark.parametrize("smiles", ["C[NH3+]", "c1ccccc1", "B(O)O", "F[C@](Cl)(Br)I"])
def test_conversion_preserves_atoms_geometry_charge_and_physical_bonds(smiles):
    record = _loqi_record(smiles)
    converted = convert_record(record)
    assert converted is not None
    assert converted.z.tolist() == [a.GetAtomicNum() for a in record.mol.GetAtoms()]
    torch.testing.assert_close(converted.pos, record.pos)
    assert converted.charges.tolist() == record.charges.tolist()
    assert converted.edge_index.shape[1] == 2 * record.mol.GetNumBonds()
    assert set(converted.edge_attr.tolist()) <= {1, 2, 3}
    order = converted.edge_index[0] * converted.num_nodes + converted.edge_index[1]
    assert (order[1:] > order[:-1]).all()
    Chem.AssignStereochemistryFrom3D(record.mol, replaceExistingTags=True)
    assert converted.smiles == smiles_from_mol(record.mol)
    assert converted.pos.untyped_storage().nbytes() == converted.pos.numel() * 4


def test_conversion_keeps_molecules_above_geom_size_limit():
    record = _loqi_record("C" * 25, stereo_edges=False)
    assert record.num_nodes > 72
    assert convert_record(record).num_nodes == record.num_nodes


def test_conversion_rejects_bad_geometry_chemistry_and_unknown_encodings():
    record = _loqi_record()
    record.pos[0, 0] = float("nan")
    assert convert_record(record) is None
    assert convert_record(_loqi_record("[CH3]", stereo_edges=False)) is None
    record = _loqi_record()
    record.x[0] = len(ATOM_NUMBERS)
    with pytest.raises(ValueError, match="atom label"):
        convert_record(record)
    record = _loqi_record()
    record.edge_attr[0] = 9
    with pytest.raises(ValueError, match="edge label"):
        convert_record(record)
    record = _loqi_record()
    record.edge_attr[0] = 2
    with pytest.raises(ValueError, match="symmetric"):
        convert_record(record)


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("count", [1, 3])
def test_lmdb_roundtrip_uses_local_release_and_reuses_cache(
    tmp_path, monkeypatch, modern, count
):
    source = tmp_path / "source" / "chembl3d_stereo" / "processed"
    records = [_loqi_record()] * count
    _save_split(source, records, modern=modern)

    def unexpected_download(*args):
        pytest.fail("Local source must not trigger a download")

    monkeypatch.setattr(chembl3d, "_download_split", unexpected_download)
    root = tmp_path / "output"
    dataset = FlowMatchingChEMBL3DDataset(
        root, vocab=None, distributions=None, source_dir=tmp_path / "source"
    )
    assert len(dataset) == count
    raw = dataset.get(0)
    torch.testing.assert_close(raw.pos, records[0].pos)
    assert dataset.get_all_smiles() == ["C[NH3+]"]
    with pytest.raises(RuntimeError, match="vocab"):
        dataset[0]
    serialized = list(dataset.iter_bytes())
    assert len(serialized) == count
    torch.testing.assert_close(mol_from_bytes(serialized[0]).z, raw.z)
    state = pickle.loads(pickle.dumps(dataset)).__dict__
    assert state["_env"] is None
    assert state["_txn"] is None
    dataset._env.close()
    (source / "train_h.pt").unlink()
    cached = ChEMBL3D(root, source_dir=source)
    assert len(cached) == count


def test_serialization_does_not_include_collated_tensor_storage(tmp_path):
    _save_split(tmp_path, [_loqi_record()] * 256)
    data, slices, count = chembl3d._load_split(tmp_path / "train_h.pt")
    record = next(chembl3d._iter_records(data, slices, count))
    converted = convert_record(record)
    blob = mol_to_bytes(converted)
    assert len(blob) < 8000
    assert set(data.keys()) == set(chembl3d.GRAPH_FIELDS)


def test_conversion_failure_does_not_publish_partial_cache(tmp_path):
    good, bad = _loqi_record(), _loqi_record()
    bad.x[0] = 99
    source, root = tmp_path / "source", tmp_path / "output"
    _save_split(source, [good, bad])
    with pytest.raises(ValueError, match="atom label"):
        ChEMBL3D(root, source_dir=source)
    assert list((root / "processed").iterdir()) == []
    _save_split(source, [good])
    assert len(ChEMBL3D(root, source_dir=source)) == 1


def test_download_is_verified_and_atomic(tmp_path, monkeypatch):
    contents = b"a small release file"
    digest = hashlib.md5(contents).hexdigest()
    monkeypatch.setitem(chembl3d.RELEASE_FILES, "train", (123, len(contents), digest))
    monkeypatch.setattr(
        chembl3d, "urlopen", lambda *args, **kwargs: io.BytesIO(contents)
    )
    target = tmp_path / "raw" / "train_h.pt"
    chembl3d._download_split("train", target)
    assert target.read_bytes() == contents
    monkeypatch.setattr(
        chembl3d, "urlopen", lambda *args, **kwargs: io.BytesIO(b"partial")
    )
    with pytest.raises(ValueError, match="checksum"):
        chembl3d._download_split("train", target)
    assert target.read_bytes() == contents
    assert list(target.parent.iterdir()) == [target]


def test_download_fallback_and_missing_explicit_source(tmp_path, monkeypatch):
    calls = []

    def download(split, destination):
        calls.append(split)
        _save_split(destination.parent, [_loqi_record()], split=split)

    monkeypatch.setattr(chembl3d, "_download_split", download)
    assert len(ChEMBL3D(tmp_path / "auto", split="val")) == 1
    assert calls == ["val"]
    with pytest.raises(FileNotFoundError, match="source_dir"):
        ChEMBL3D(tmp_path / "explicit", source_dir=tmp_path / "missing")
    assert calls == ["val"]


@pytest.mark.parametrize("mode", ["explicit", "implicit"])
def test_preprocess_command_downloads_all_splits_and_reuses_cache(
    tmp_path, monkeypatch, mode
):
    """Exercise download -> raw files -> all splits/statistics through the CLI."""
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("CHEMBL3D_SOURCE_DIR", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    responses = {}
    for file_id, split in enumerate(("train", "val", "test"), start=1):
        buffer = io.BytesIO()
        torch.save(InMemoryDataset.collate([_loqi_record()] * 2), buffer)
        contents = buffer.getvalue()
        responses[f"https://ndownloader.figshare.com/files/{file_id}"] = contents
        monkeypatch.setitem(
            chembl3d.RELEASE_FILES,
            split,
            (file_id, len(contents), hashlib.md5(contents).hexdigest()),
        )
    requests = []

    def download(url, **kwargs):
        requests.append(url)
        return io.BytesIO(responses[url])

    monkeypatch.setattr(chembl3d, "urlopen", download)
    arguments = ["scripts/preprocess.py", "chembl3d", "--workers", "0"]
    if mode == "implicit":
        arguments += ["--hydrogen-mode", "implicit"]
    monkeypatch.setattr(sys, "argv", arguments)
    command = Path(__file__).parents[2] / "scripts/preprocess.py"
    runpy.run_path(str(command), run_name="__main__")

    suffix = "_implicit_h" if mode == "implicit" else ""
    root = tmp_path / "data" / f"chembl3d{suffix}"
    assert requests == list(responses)
    artifacts = [
        root / "processed" / name
        for name in (
            "atom_tokens.txt",
            "edge_tokens.txt",
            "charge_tokens.txt",
            "distributions.pt",
        )
    ]
    atoms = (root / "processed/atom_tokens.txt").read_text().splitlines()
    assert ("H" in atoms) == (mode == "explicit")
    for file_id, split in enumerate(("train", "val", "test"), start=1):
        raw = root / "raw" / f"{split}_h.pt"
        assert (
            raw.read_bytes()
            == responses[f"https://ndownloader.figshare.com/files/{file_id}"]
        )
        dataset = ChEMBL3D(root, split=split, hydrogen_mode=mode)
        assert len(dataset) == 2
        assert dataset.get_all_smiles() == ["C[NH3+]"]
        dataset.close()
        artifacts.extend([raw, root / "processed" / f"{split}_data.lmdb/data.mdb"])
    timestamps = {path: path.stat().st_mtime_ns for path in artifacts}

    def unexpected_download(*args, **kwargs):
        pytest.fail("A completed cache must work without network access")

    monkeypatch.setattr(chembl3d, "urlopen", unexpected_download)
    runpy.run_path(str(command), run_name="__main__")
    assert {path: path.stat().st_mtime_ns for path in artifacts} == timestamps


def test_parallel_conversion_and_spawn_dataloader(tmp_path):
    source = tmp_path / "source"
    _save_split(source, [_loqi_record("C"), _loqi_record("O")])
    dataset = ChEMBL3D(tmp_path / "output", source_dir=source, preprocess_workers=2)
    loader = DataLoader(
        dataset, batch_size=None, num_workers=2, multiprocessing_context="spawn"
    )
    assert [record.smiles for record in loader] == ["C", "O"]


@pytest.mark.parametrize("workers", [0, 2])
def test_config_preprocesses_training_statistics_and_decodes_all_splits(
    tmp_path, monkeypatch, workers
):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CHEMBL3D_SOURCE_DIR", str(tmp_path / "source"))
    for split, smiles in [("train", "C[NH3+]"), ("val", "C"), ("test", "N")]:
        _save_split(tmp_path / "source", [_loqi_record(smiles)] * 3, split=split)
    config_dir = Path(__file__).parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        config = compose(
            config_name="default",
            overrides=[
                "defaults=chembl3d/chembl3d",
                f"data.preprocess_workers={workers}",
            ],
        )
    assert config.data.preprocessing.root == str(tmp_path / "data" / "chembl3d")
    assert config.data.allow_charged is True
    assert config.data.vocab is None
    preprocessing = hydra.utils.instantiate(config.data.preprocessing)
    assert set(preprocessing.vocab.atom_tokens) == {"C", "H", "N"}
    assert preprocessing.vocab.edge_tokens == ["<NO_BOND>", "1"]
    assert preprocessing.vocab.charge_tokens == ["0", "1"]
    assert preprocessing.distributions.coordinate_std > 0
    for split, cfg in [
        ("train", config.data.datamodule.datasets.train),
        ("val", config.data.datamodule.datasets.val[0]),
        ("test", config.data.datamodule.datasets.test[0]),
    ]:
        dataset = hydra.utils.instantiate(
            cfg,
            split=split,
            vocab=preprocessing.vocab,
            distributions=preprocessing.distributions,
        )
        decoded = dataset[0]
        raw = dataset.get(0)
        assert decoded.a.shape[0] == raw.num_nodes
        torch.testing.assert_close(decoded.x.mean(0), torch.zeros(3), atol=1e-6, rtol=0)
        assert (decoded.a >= 0).all()
        dataset._env.close()


@pytest.mark.parametrize("mode", ["explicit", "implicit"])
def test_processing_applies_inclusive_cutoff_before_hydrogen_removal(tmp_path, mode):
    source = tmp_path / "source"
    _save_split(
        source, [_loqi_record(s, stereo_edges=False) for s in ("C", "CC", "CCC")]
    )
    dataset = ChEMBL3D(
        tmp_path / mode,
        source_dir=source,
        max_atoms=8,
        hydrogen_mode=mode,
    )
    assert len(dataset) == 2
    assert [dataset.get(i).num_nodes for i in range(2)] == (
        [5, 8] if mode == "explicit" else [1, 2]
    )
    assert dataset.get_all_smiles() == ["C", "CC"]
    dataset.close()


def test_conversion_saves_rdkit_normalized_charge_labels():
    # The raw neutral N(=O)=O representation is normalized by RDKit to nitro.
    record = _loqi_record("C[N+](=O)[O-]", stereo_edges=False)
    oxygen = next(
        a.GetIdx() for a in record.mol.GetAtoms() if a.GetFormalCharge() == -1
    )
    record.charges.zero_()
    record.edge_attr[(record.edge_index == oxygen).any(dim=0)] = 2
    result = convert_record(record)
    assert result is not None
    assert sorted(result.charges.tolist()) == sorted(
        a.GetFormalCharge() for a in record.mol.GetAtoms()
    )
    assert "[N+]" in result.smiles
    assert "[O-]" in result.smiles


@pytest.mark.parametrize(
    "edge_tokens", [["<NO_BOND>", "1", "3"], ["3", "<NO_BOND>", "1"]]
)
def test_decode_maps_sparse_and_reordered_bond_vocabularies(tmp_path, edge_tokens):
    source = tmp_path / "source"
    _save_split(source, [_loqi_record("C#N", stereo_edges=False)])
    vocab = Vocab(
        atom_tokens=["C", "H", "N"], edge_tokens=edge_tokens, charge_tokens=["0"]
    )
    dataset = FlowMatchingChEMBL3DDataset(
        tmp_path / "output",
        source_dir=source,
        vocab=vocab,
        distributions=SimpleNamespace(coordinate_std=1.0),
    )
    raw, graph = dataset.get(0), dataset[0]
    assert graph.e.tolist() == [
        edge_tokens.index(str(int(label))) for label in raw.edge_attr
    ]
    mol = graph.to_rdkit(vocab)
    assert mol_is_valid(mol)
    roundtrip = type(graph).from_rdkit(mol, vocab)
    assert mol_is_valid(roundtrip.to_rdkit(vocab))
    dataset.close()


def test_decode_rejects_unseen_negative_charge_instead_of_neutralizing_it(tmp_path):
    source = tmp_path / "source"
    _save_split(source, [_loqi_record("C[O-]", stereo_edges=False)])
    dataset = FlowMatchingChEMBL3DDataset(
        tmp_path / "output",
        source_dir=source,
        vocab=Vocab(
            atom_tokens=["C", "H", "O"],
            edge_tokens=["<NO_BOND>", "1"],
            charge_tokens=["0"],
        ),
        distributions=SimpleNamespace(coordinate_std=1.0),
    )
    with pytest.raises(ValueError, match="charge labels"):
        dataset[0]
    dataset.close()


def test_smiles_sidecar_publish_failure_preserves_previous_file(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _save_split(source, [_loqi_record("C", stereo_edges=False)])
    dataset = ChEMBL3D(tmp_path / "output", source_dir=source, preprocess_workers=0)
    sidecar = Path(dataset.processed_dir) / "train_smiles.txt"
    sidecar.write_text("previous complete contents")
    before = set(sidecar.parent.iterdir())

    def fail_replace(source, destination):
        raise OSError("Publication interrupted")

    monkeypatch.setattr("morph.dataset.core.lmdb.os.replace", fail_replace)
    with pytest.raises(OSError, match="Publication interrupted"):
        dataset._write_smiles_sidecar()
    assert sidecar.read_text() == "previous complete contents"
    assert set(sidecar.parent.iterdir()) == before
    dataset.close()
