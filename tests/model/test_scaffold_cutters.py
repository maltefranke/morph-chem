"""Scaffold cuts, cache isolation, and train/evaluation data paths."""

import hashlib
import io
import os
import pickle
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import lmdb
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rdkit import Chem

from morph.dataset.conditional.scaffold import fingerprint, resources
from morph.dataset.conditional.scaffold import lmdb as scaffold_storage
from morph.dataset.conditional.scaffold.cutters import (
    _atom_indices_for_data,
    scaffold_indices,
)
from morph.dataset.conditional.scaffold.datamodule import ScaffoldLightningDataModule
from morph.dataset.conditional.scaffold.lmdb import (
    load_scaffold_cache,
    scaffold_lmdb_path,
    scaffold_pt_path,
)
from morph.dataset.conditional.scaffold.wrappers import (
    FlowMatchingDatasetWrapperScaffold,
)
from morph.dataset.core import source as source_identity
from morph.dataset.core.atomic_data import ChemAtomicData
from morph.dataset.core.flow_matching_wrapper import eval_collate_fn, train_collate_fn
from morph.dataset.core.representation import Capabilities
from morph.dataset.core.vocab import Distributions, Vocab
from morph.flow_matching.interpolation import Interpolator
from morph.utils.rdkit_utils import BOND_IDX_MAP

KINDS = ("murcko", "rgroup", "heavy", "linker")
EXAMPLE = "CCOc1ccc(CCNC(=O)C)cc1"
VOCAB = Vocab(
    atom_tokens=["H", "C", "N", "O"],
    edge_tokens=["<NO_BOND>", "1", "2", "3", "4"],
    charge_tokens=["-1", "0", "1"],
)


def _molecule(smiles=EXAMPLE):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(i, (float(i), float(i % 3), float(i % 5)))
    mol.AddConformer(conformer)
    return mol


def _raw(mol):
    edges, types = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((i, j), (j, i)))
        types.extend([BOND_IDX_MAP[bond.GetBondType()]] * 2)
    return SimpleNamespace(
        z=torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()]),
        charges=torch.tensor([a.GetFormalCharge() for a in mol.GetAtoms()]),
        edge_index=torch.tensor(edges).reshape(-1, 2).T.contiguous(),
        edge_attr=torch.tensor(types),
    )


class _Dataset:
    CAPABILITIES = Capabilities(provides_charges=True, provides_topology=True)

    def __init__(self, *smiles):
        self.mols = [_molecule(s) for s in (smiles or (EXAMPLE,))]
        self.raw_reads = 0

    def source_identity(self):
        storage = (
            Path(self.lmdb_path) / "data.mdb" if hasattr(self, "lmdb_path") else None
        )
        return source_identity.molecular_source_identity(
            self,
            storage=storage,
            hydrogen_mode=getattr(self, "hydrogen_mode", None),
            processing_options=getattr(self, "processing_options", None),
            indices=getattr(self, "_indices", None),
        )

    def __len__(self):
        return len(self.mols)

    def get(self, index):
        self.raw_reads += 1
        return _raw(self.mols[index])

    def __getitem__(self, index):
        result = ChemAtomicData.from_rdkit(self.mols[index], VOCAB)
        result.add_node_property("original_index", torch.arange(result.num_nodes))
        return result


def _wrapper(tmp_path, kind, stage="test", enumerate_slicings=False, dataset=None):
    distributions = Distributions(
        atom_type_distribution=torch.ones(4) / 4,
        edge_type_distribution=torch.ones(5) / 5,
        charge_type_distribution=torch.ones(3) / 3,
        n_atoms_distribution=torch.ones(64) / 64,
        coordinate_std=1.0,
    )
    return FlowMatchingDatasetWrapperScaffold(
        base_dataset=dataset if dataset is not None else _Dataset(),
        distributions=distributions,
        interpolator=Interpolator(
            distributions=distributions, move_noise_scale=0, ins_noise_scale=0
        ),
        vocab=VOCAB,
        scaffold_groups_dir=str(tmp_path),
        scaffold_kind=kind,
        enumerate_slicings=enumerate_slicings,
        stage=stage,
        time_dist=torch.distributions.Uniform(0.4, 0.6),
    )


@pytest.mark.parametrize("kind", KINDS)
def test_cutters_return_distinct_reproducible_original_atom_sets(kind):
    mol = _molecule()
    cuts = scaffold_indices(mol, kind)
    assert cuts
    assert len(cuts) == len(set(cuts))
    assert cuts == scaffold_indices(mol, kind)
    assert all(cut == tuple(sorted(set(cut))) for cut in cuts)
    assert all(0 <= i < mol.GetNumAtoms() for cut in cuts for i in cut)
    assert all(0 < len(cut) < mol.GetNumAtoms() for cut in cuts)
    if kind != "linker":
        assert all(
            mol.GetAtomWithIdx(i).GetAtomicNum() > 1 for cut in cuts for i in cut
        )


@pytest.mark.parametrize("kind", KINDS)
def test_cutters_preserve_indices_when_hydrogens_are_interleaved(kind):
    mol = _molecule()
    order = torch.randperm(
        mol.GetNumAtoms(), generator=torch.Generator().manual_seed(27)
    ).tolist()
    permuted = Chem.RenumberAtoms(mol, order)
    expected = {frozenset(cut) for cut in scaffold_indices(mol, kind)}
    actual = {
        frozenset(order[i] for i in cut) for cut in scaffold_indices(permuted, kind)
    }
    assert actual == expected


def test_brics_provides_alternative_cores_and_respects_size_constraints():
    mol = _molecule()
    cuts = scaffold_indices(mol, "rgroup")
    assert scaffold_indices(mol) == cuts
    heavy_count = Chem.RemoveHs(mol).GetNumAtoms()
    assert len(cuts) > 1
    assert len({len(cut) for cut in cuts}) > 1
    for cut in cuts:
        assert len(cut) >= 0.4 * heavy_count
        core = set(cut)
        boundaries = [
            b
            for b in mol.GetBonds()
            if (b.GetBeginAtomIdx() in core) != (b.GetEndAtomIdx() in core)
            and b.GetBeginAtom().GetAtomicNum() > 1
            and b.GetEndAtom().GetAtomicNum() > 1
        ]
        assert 1 <= len(boundaries) <= 4
        assert all(not bond.IsInRing() for bond in boundaries)
    assert len(scaffold_indices(mol, "rgroup", max_slicings=2)) == 2


def test_murcko_and_heavy_have_the_expected_semantics():
    mol = _molecule("CCc1ccccc1")
    assert len(scaffold_indices(mol, "murcko")) == 1
    assert len(scaffold_indices(mol, "murcko")[0]) == 6
    assert scaffold_indices(_molecule("CCCC"), "murcko") == []
    assert scaffold_indices(Chem.RemoveHs(mol), "heavy") == []
    heavy = scaffold_indices(mol, "heavy")[0]
    assert all(
        mol.GetAtomWithIdx(i).GetAtomicNum() == 1
        for i in set(range(mol.GetNumAtoms())) - set(heavy)
    )


def test_linker_keeps_multiple_fragments_with_their_original_hydrogens():
    mol = _molecule()
    for cut in scaffold_indices(mol, "linker"):
        kept = set(cut)
        edited = Chem.RWMol(mol)
        for i in reversed(range(mol.GetNumAtoms())):
            if i not in kept:
                edited.RemoveAtom(i)
        pieces = Chem.GetMolFrags(edited.GetMol())
        assert len(pieces) >= 2
        assert all(
            sum(mol.GetAtomWithIdx(cut[i]).GetAtomicNum() > 1 for i in p) >= 3
            for p in pieces
        )
        for i in cut:
            atom = mol.GetAtomWithIdx(i)
            if atom.GetAtomicNum() > 1:
                assert all(
                    n.GetIdx() in kept
                    for n in atom.GetNeighbors()
                    if n.GetAtomicNum() == 1
                )


def test_dataset_reconstruction_retains_formal_charges_and_atom_indices():
    mol = _molecule("C[N+](C)(C)Cc1ccccc1")
    expected = scaffold_indices(mol, "rgroup")
    assert expected
    assert _atom_indices_for_data(_raw(mol), "rgroup") == expected


def test_unsupported_cutters_fail_before_cache_creation(tmp_path):
    kind = "typo"
    with pytest.raises(ValueError, match="unknown scaffold_kind"):
        scaffold_indices(_molecule(), kind)
    with pytest.raises(ValueError, match="unknown scaffold_kind"):
        _wrapper(tmp_path, kind)
    assert not list(tmp_path.iterdir())


def test_cache_kinds_are_isolated_and_matching_cuts_are_reused(tmp_path, monkeypatch):
    dataset = _Dataset()
    for kind in KINDS:
        first = load_scaffold_cache(dataset, tmp_path, "train", scaffold_kind=kind)
        with monkeypatch.context() as patch:
            patch.setattr(
                scaffold_storage, "_build_cache", lambda *_: pytest.fail("cache miss")
            )
            second = load_scaffold_cache(dataset, tmp_path, "train", scaffold_kind=kind)
        assert first.lmdb_path == second.lmdb_path
        assert first.atom_indices[0] == second.atom_indices[0]
    assert len(list(tmp_path.glob("*.lmdb"))) == len(KINDS)


@pytest.mark.parametrize("format", ["pt", "lmdb"])
def test_legacy_cache_is_rebuilt_without_overwriting_unverified_data(tmp_path, format):
    dataset = _Dataset()
    if format == "pt":
        path = scaffold_pt_path(tmp_path, "test", "rgroup")
        torch.save({"scaffold_atom_indices": [[(999,)]]}, path)
        original = Path(path).read_bytes()
    else:
        path = scaffold_lmdb_path(tmp_path, "test", "rgroup")
        with lmdb.open(path, map_size=1024 * 1024) as env, env.begin(write=True) as txn:
            for key, value in (
                (b"meta:n", 1),
                (b"meta:scaffold_nonempty_indices", [0]),
                (b"ai:0", [(999,)]),
            ):
                txn.put(key, pickle.dumps(value))
        original = (Path(path) / "data.mdb").read_bytes()
    cache = load_scaffold_cache(dataset, tmp_path, "test", scaffold_kind="rgroup")
    assert cache.atom_indices[0] == scaffold_indices(dataset.mols[0], "rgroup")
    assert cache.lmdb_path != path
    storage = Path(path) if format == "pt" else Path(path) / "data.mdb"
    assert storage.read_bytes() == original


@pytest.mark.parametrize(
    "change", ["source", "record_order", "atom_order", "options", "version"]
)
def test_same_length_source_changes_never_reuse_atom_indices(
    tmp_path, monkeypatch, change
):
    dataset = _Dataset(EXAMPLE, "CCc1ccccc1")
    first = load_scaffold_cache(dataset, tmp_path, "train", scaffold_kind="murcko")
    original = first.atom_indices[0]
    if change == "source":
        dataset.mols[0] = _molecule("CCCC")
    elif change == "record_order":
        dataset.mols.reverse()
    elif change == "atom_order":
        mol = dataset.mols[0]
        dataset.mols[0] = Chem.RenumberAtoms(
            mol, list(reversed(range(mol.GetNumAtoms())))
        )
    elif change == "options":
        dataset.hydrogen_mode = "implicit"
        dataset.processing_options = {"hydrogen_mode": "implicit"}
        dataset.mols = [Chem.RemoveHs(mol) for mol in dataset.mols]
    else:
        monkeypatch.setattr(fingerprint, "CACHE_VERSION", fingerprint.CACHE_VERSION + 1)
    second = load_scaffold_cache(dataset, tmp_path, "train", scaffold_kind="murcko")
    assert second.lmdb_path != first.lmdb_path
    for i, mol in enumerate(dataset.mols):
        assert second.atom_indices[i] == scaffold_indices(mol, "murcko")
    assert first.atom_indices[0] == original


def test_lmdb_source_identity_avoids_scans_and_detects_replacement(
    tmp_path, monkeypatch
):
    dataset = _Dataset()
    dataset.lmdb_path = str(tmp_path / "source.lmdb")
    storage = Path(dataset.lmdb_path) / "data.mdb"
    storage.parent.mkdir()
    storage.write_bytes(b"original")
    first = load_scaffold_cache(dataset, tmp_path / "cuts", "train")
    with monkeypatch.context() as patch:
        patch.setattr(dataset, "get", lambda _: pytest.fail("unchanged source scanned"))
        assert (
            load_scaffold_cache(dataset, tmp_path / "cuts", "train").lmdb_path
            == first.lmdb_path
        )
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replaced")
    replacement.replace(storage)
    dataset.mols = [_molecule("CCc1ccccc1")]
    second = load_scaffold_cache(dataset, tmp_path / "cuts", "train")
    assert second.lmdb_path != first.lmdb_path
    assert second.atom_indices[0] == scaffold_indices(dataset.mols[0])


def test_failed_cache_build_is_not_published(tmp_path, monkeypatch):
    dataset = _Dataset()
    with monkeypatch.context() as patch:

        def fail(*_args):
            raise RuntimeError("cut failed")

        patch.setattr(scaffold_storage, "_atom_indices_for_data", fail)
        with pytest.raises(RuntimeError, match="cut failed"):
            load_scaffold_cache(dataset, tmp_path, "train")
    assert not list(tmp_path.glob("*.lmdb"))
    assert not [path for path in tmp_path.iterdir() if path.is_dir()]
    assert load_scaffold_cache(dataset, tmp_path, "train").n == 1


def test_downloaded_source_and_cache_are_reused_at_a_different_path(
    tmp_path, monkeypatch
):
    dataset = _Dataset()
    source = tmp_path / "original" / "source.lmdb"
    source.mkdir(parents=True)
    (source / "data.mdb").write_bytes(b"unchanged molecular records")
    dataset.lmdb_path = str(source)
    original = load_scaffold_cache(dataset, source.parent / "cuts", "train")
    download = tmp_path / "download"
    shutil.copytree(source.parent, download, copy_function=shutil.copyfile)
    dataset.lmdb_path = str(download / "source.lmdb")
    monkeypatch.setattr(
        scaffold_storage, "_build_cache", lambda *_: pytest.fail("recomputed cuts")
    )
    downloaded = load_scaffold_cache(dataset, download / "cuts", "train")
    assert downloaded.metadata == original.metadata
    assert Path(downloaded.lmdb_path).name == Path(original.lmdb_path).name
    assert downloaded.atom_indices[0] == original.atom_indices[0]


def test_same_size_source_edit_invalidates_memoized_checksum(tmp_path):
    dataset = _Dataset()
    source = tmp_path / "source.lmdb"
    source.mkdir()
    dataset.lmdb_path = str(source)
    storage = source / "data.mdb"
    storage.write_bytes(b"original")
    before = fingerprint.dataset_fingerprint(dataset)
    stat = storage.stat()
    storage.write_bytes(b"modified")
    # Two tiny writes can share a filesystem timestamp tick.
    os.utime(storage, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert fingerprint.dataset_fingerprint(dataset) != before


def test_source_checksums_are_shared_between_processes(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"immutable molecular records")
    counter = tmp_path / "reads"
    script = """
import sys
import time
from pathlib import Path
from morph.dataset.core import source as f
source, counter = map(Path, sys.argv[1:])
original = f.hashlib.file_digest
def counted(*args):
    with counter.open('a') as stream:
        stream.write('read\\n')
    time.sleep(0.2)
    return original(*args)
f.hashlib.file_digest = counted
print(f._file_digest(source, f._stat_key(source)))
"""
    command = [sys.executable, "-c", script, str(source), str(counter)]
    options = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
    with (
        subprocess.Popen(command, **options) as first,
        subprocess.Popen(command, **options) as second,
    ):
        for process in (first, second):
            stdout, stderr = process.communicate(timeout=60)
            assert process.returncode == 0, stderr
            assert stdout.strip() == hashlib.sha256(source.read_bytes()).hexdigest()
    assert counter.read_text().splitlines() == ["read"]


def test_corrupt_checksum_memo_is_rebuilt(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"immutable molecular records")
    expected = source_identity._file_digest(source, source_identity._stat_key(source))
    memo = next((tmp_path / "morph-cache/source_checksums").glob("*.json"))
    memo.write_text("{partial")
    source_identity._file_digest.cache_clear()
    assert (
        source_identity._file_digest(source, source_identity._stat_key(source))
        == expected
    )


def test_source_change_during_hash_is_not_memoized(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"original")
    stat = source.stat()
    original = source_identity.hashlib.file_digest

    def changed(*args):
        digest = original(*args)
        source.write_bytes(b"modified")
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        return digest

    monkeypatch.setattr(source_identity.hashlib, "file_digest", changed)
    with pytest.raises(RuntimeError, match="changed while hashing"):
        source_identity._file_digest(source, source_identity._stat_key(source))
    assert not list((tmp_path / "morph-cache/source_checksums").glob("*.json"))


def test_libinvent_resources_download_once_and_reuse_offline(tmp_path, monkeypatch):
    payloads = {
        "https://example.test/reactions": b"rule",
        "https://example.test/filters": b"{}",
    }
    specs = [(url, hashlib.sha256(data).hexdigest()) for url, data in payloads.items()]
    monkeypatch.setattr(resources, "_REACTIONS", specs[0])
    monkeypatch.setattr(resources, "_FILTERS", specs[1])
    calls = []

    def download(url, *, timeout):
        calls.append(url)
        return io.BytesIO(payloads[url])

    monkeypatch.setattr(resources, "urlopen", download)
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda _: resources.libinvent_resources(), range(2)))
    assert paths[0] == paths[1]
    assert sorted(calls) == sorted(payloads)
    assert [p.read_bytes() for p in paths[0]] == list(payloads.values())
    monkeypatch.setattr(resources, "urlopen", lambda *_a, **_k: pytest.fail("offline"))
    assert resources.libinvent_resources() == paths[0]


def test_libinvent_bad_download_is_rejected_and_can_be_retried(tmp_path, monkeypatch):
    url, data = "https://example.test/reactions", b"verified rules"
    checksum = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(resources, "urlopen", lambda *_a, **_k: io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="SHA-256 verification"):
        resources._download_resource(url, checksum)
    assert not list((tmp_path / "morph-cache/libinvent").glob("*/reactions"))
    monkeypatch.setattr(resources, "urlopen", lambda *_a, **_k: io.BytesIO(data))
    path = resources._download_resource(url, checksum)
    assert path.read_bytes() == data
    path.write_bytes(b"corrupted cached rules")
    assert resources._download_resource(url, checksum).read_bytes() == data


def test_libinvent_download_error_explains_offline_setup(monkeypatch):
    def offline(*_args, **_kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(resources, "urlopen", offline)
    with pytest.raises(OSError, match="Run once with network access"):
        resources.libinvent_resources()


def test_published_libinvent_cache_does_not_fetch_resources(tmp_path, monkeypatch):
    dataset = _Dataset()
    monkeypatch.setattr(
        resources, "urlopen", lambda *_a, **_k: pytest.fail("unexpected download")
    )
    with monkeypatch.context() as patch:
        patch.setattr(scaffold_storage, "_atom_indices_for_data", lambda *_: [(0, 1)])
        original = load_scaffold_cache(
            dataset, tmp_path, "test", scaffold_kind="libinvent"
        )
    monkeypatch.setattr(
        scaffold_storage, "_build_cache", lambda *_: pytest.fail("unexpected rebuild")
    )
    cached = load_scaffold_cache(dataset, tmp_path, "test", scaffold_kind="libinvent")
    assert cached.lmdb_path == original.lmdb_path
    assert cached.atom_indices[0] == [(0, 1)]


def test_filtered_lmdb_views_with_equal_counts_have_separate_caches(tmp_path):
    class FilteredDataset(_Dataset):
        def __len__(self):
            return len(self._indices)

        def get(self, index):
            return super().get(int(self._indices[index]))

    dataset = FilteredDataset(EXAMPLE, "CCc1ccccc1", "CCCC")
    dataset._indices = torch.tensor([0, 1])
    storage = tmp_path / "source.lmdb" / "data.mdb"
    storage.parent.mkdir()
    storage.write_bytes(b"immutable source")
    dataset.lmdb_path = str(storage.parent)
    first = load_scaffold_cache(
        dataset, tmp_path / "cuts", "train", scaffold_kind="murcko"
    )
    dataset._indices = torch.tensor([2, 1])
    second = load_scaffold_cache(
        dataset, tmp_path / "cuts", "train", scaffold_kind="murcko"
    )
    assert first.lmdb_path != second.lmdb_path
    assert first.atom_indices[0]
    assert second.atom_indices[0] == []
    assert second.filtered_indices().tolist() == [1]


def test_source_changes_during_build_are_not_published(tmp_path, monkeypatch):
    dataset = _Dataset()
    dataset.processing_options = {"hydrogen_mode": "explicit"}
    cut = scaffold_storage._atom_indices_for_data

    def mutate_source(*args):
        dataset.processing_options["hydrogen_mode"] = "implicit"
        return cut(*args)

    monkeypatch.setattr(scaffold_storage, "_atom_indices_for_data", mutate_source)
    with pytest.raises(RuntimeError, match="source changed"):
        load_scaffold_cache(dataset, tmp_path, "train")
    assert not list(tmp_path.glob("*.lmdb"))


def test_concurrent_cache_builders_publish_once(tmp_path, monkeypatch):
    build = scaffold_storage._build_cache
    calls = []

    def counted(*args):
        calls.append(args[1])
        return build(*args)

    monkeypatch.setattr(scaffold_storage, "_build_cache", counted)
    with ThreadPoolExecutor(max_workers=2) as pool:
        caches = list(
            pool.map(
                lambda _: load_scaffold_cache(_Dataset(), tmp_path, "train"), range(2)
            )
        )
    assert len(calls) == 1
    assert caches[0].lmdb_path == caches[1].lmdb_path


@pytest.mark.parametrize("kind", KINDS)
def test_evaluation_enumerates_every_cut_and_collates_native_graphs(tmp_path, kind):
    dataset = _Dataset(EXAMPLE, "CCc1ccccc1")
    wrapper = _wrapper(tmp_path, kind, enumerate_slicings=True, dataset=dataset)
    expected = [
        (i, cut)
        for i, mol in enumerate(dataset.mols)
        for cut in scaffold_indices(mol, kind)
    ]
    assert len(wrapper) == len(expected)
    for index, (molecule_idx, cut) in enumerate(expected):
        source, target = wrapper[index]
        assert tuple(source.original_index.tolist()) == cut
        assert source.keep_mask.tolist() == [1] * len(cut)
        torch.testing.assert_close(source.x, dataset[molecule_idx].x[list(cut)])
        batch, full = eval_collate_fn([(source, target)])
        assert batch.keep_mask.sum() == len(cut)
        assert full.num_nodes == target.num_nodes
    assert torch.equal(
        wrapper[-1][0].original_index, wrapper[len(wrapper) - 1][0].original_index
    )
    with pytest.raises(IndexError):
        wrapper[len(wrapper)]


def test_evaluation_deduplicates_legacy_murcko_automorphisms(tmp_path):
    dataset = _Dataset("CCc1ccccc1")
    cut = scaffold_indices(dataset.mols[0], "murcko")[0]
    torch.save(
        {"scaffold_atom_indices": [[cut, cut[::-1]]]},
        scaffold_pt_path(str(tmp_path), "test", "murcko"),
    )
    wrapper = _wrapper(tmp_path, "murcko", enumerate_slicings=True, dataset=dataset)
    assert len(wrapper) == 1


def test_empty_scaffold_dataset_has_no_evaluation_instances(tmp_path):
    wrapper = _wrapper(
        tmp_path, "murcko", enumerate_slicings=True, dataset=_Dataset("CCCC")
    )
    assert len(wrapper) == 0
    with pytest.raises(IndexError):
        wrapper[0]


def test_enumerated_validation_keeps_the_final_partial_batch(tmp_path):
    wrapper = _wrapper(tmp_path, "rgroup", enumerate_slicings=True)
    module = ScaffoldLightningDataModule(
        vocab=VOCAB,
        distributions=None,
        datasets=OmegaConf.create({}),
        interpolator=None,
        num_workers=OmegaConf.create({"val": 0}),
        batch_size=OmegaConf.create({"train": 1, "val": 4}),
        scaffold_groups_dir=str(tmp_path),
        enumerate_slicings=True,
    )
    module.val_datasets = [wrapper]
    batches = list(module.val_dataloader()[0])
    assert sum(source.batch_size for source, _target in batches) == len(wrapper)
    assert batches[-1][0].batch_size == len(wrapper) % 4


@pytest.mark.parametrize("kind", KINDS)
def test_training_randomly_selects_cut_and_preserves_keep_mask_through_collation(
    tmp_path, monkeypatch, kind
):
    choices = []

    def choose_last(cuts):
        choices.append(cuts)
        return cuts[-1]

    monkeypatch.setattr(
        "morph.dataset.conditional.scaffold.wrappers.random.choice", choose_last
    )
    wrapper = _wrapper(tmp_path, kind, stage="train", enumerate_slicings=True)
    assert len(wrapper) == 1
    item = wrapper[0]
    assert len(choices) == 1
    cut = choices[0][-1]
    mol_t, _target, _insertions, _t = item
    assert mol_t.keep_mask.sum() == len(cut)
    batch, *_ = train_collate_fn([item])
    assert batch.keep_mask.sum() == len(cut)
    assert torch.isfinite(batch.x).all()
    assert torch.all(mol_t.keep_mask[: len(cut)] == 1)
    torch.testing.assert_close(
        mol_t.a[: len(cut)], wrapper.base_dataset[0].a[list(cut)]
    )


def test_scaffold_training_wrapper_uses_small_jitter_only_on_retained_atoms(
    tmp_path, monkeypatch
):
    wrapper = _wrapper(tmp_path, "murcko", stage="train")
    wrapper.time_dist = SimpleNamespace(sample=lambda _: torch.ones(1))
    monkeypatch.setattr(
        "morph.flow_matching.interpolation.torch.randn_like",
        lambda x: torch.arange(x.numel(), dtype=x.dtype).reshape_as(x),
    )
    mol_t, mol_1, _, _ = wrapper[0]
    expected = torch.arange(mol_t.x.numel(), dtype=mol_t.x.dtype).reshape_as(mol_t.x)
    expected *= 0.1
    expected[~mol_t.keep_mask.bool()] = 0
    expected -= expected.mean(dim=0)
    torch.testing.assert_close(mol_t.x - mol_1.x, expected, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("dataset", ["qm9", "geom"])
def test_scaffold_configs_route_kind_and_enumerate_only_evaluation(dataset):
    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[2] / "configs"), version_base="1.1"
    ):
        cfg = compose(
            config_name="default",
            overrides=[
                f"defaults={dataset}/scaffold_decoration",
                "data.enumerate_slicings=true",
            ],
        )
    assert cfg.data.datamodule.scaffold_kind == "rgroup"
    assert cfg.data.datamodule.enumerate_slicings is True
    assert cfg.data.interpolator.jitter_scale == 0.1
    assert cfg.model.module.freeze_keep_positions is False
    assert cfg.model.module.exclude_scaffold_from_x_loss is False
    assert cfg.model.module.exclude_scaffold_from_sub_loss is True
    assert cfg.model.module.freeze_keep_charges is True
    module = ScaffoldLightningDataModule(
        vocab=VOCAB,
        distributions=SimpleNamespace(),
        datasets=OmegaConf.create({}),
        interpolator=None,
        num_workers=OmegaConf.create({}),
        batch_size=OmegaConf.create({"train": 1}),
        scaffold_groups_dir="/unused",
        scaffold_kind=cfg.data.datamodule.scaffold_kind,
        enumerate_slicings=cfg.data.datamodule.enumerate_slicings,
    )
    assert module._wrapper_kwargs("train")["enumerate_slicings"] is False
    assert module._wrapper_kwargs("test")["enumerate_slicings"] is True
    assert module._wrapper_kwargs("val")["scaffold_kind"] == "rgroup"
