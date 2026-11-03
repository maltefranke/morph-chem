"""Dataset adapters and arbitrary geometric contexts use shared infrastructure."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from morph.dataset.conditional.pocket.datamodule import PocketLightningDataModule
from morph.dataset.conditional.pocket.scaffold_pocket import scaffold_cache_directory
from morph.dataset.conditional.scaffold.fingerprint import dataset_fingerprint
from morph.dataset.core.atomic_data import ChemAtomicData, ChemBatch, filter_nodes
from morph.dataset.core.flow_matching_wrapper import FlowMatchingDatasetWrapper
from morph.dataset.core.lmdb import (
    MolecularLMDBDataset,
    mol_to_bytes,
    write_bytes_to_lmdb,
)
from morph.dataset.core.source import molecular_source_identity
from morph.dataset.core.vocab import Vocab
from morph.dataset.unconditional.qm9 import FlowMatchingQM9Dataset, RevisedQM9
from morph.model.lightning._sampling import _OutputFrame
from tests.model.test_atomic_data import _path_graph
from tests.model.test_crossdocked import _tokenized
from tests.model.test_model_build import _setup
from tests.model.test_scaffold_cutters import _Dataset
from tests.model.test_scaffold_evaluation import Molecules

pytest_plugins = ["tests.model.test_crossdocked"]


class _AlternatePocketDataset(_Dataset):
    """An in-memory source with no CrossDocked files or storage attributes."""

    def __init__(
        self, *, vocab, distributions, split, filter_by_vocab, allowed_atom_symbols
    ):
        super().__init__()
        self.vocab, self.distributions, self.split = vocab, distributions, split
        assert filter_by_vocab
        assert allowed_atom_symbols == ["C", "N"]

    def get_pocket_vocab(self):
        return Vocab(["C", "N"], ["<NO_BOND>"], ["0"])

    def get_pocket(self, index):
        return _path_graph(2)

    def source_identity(self):
        return molecular_source_identity(self, split=self.split)

    def cache_namespace(self):
        return "alternate-source"


def test_pocket_datamodule_accepts_another_source_without_crossdocked_files(tmp_path):
    cfg, data = _setup()
    dm = PocketLightningDataModule(
        vocab=data.vocab,
        distributions=data.prior,
        datasets=OmegaConf.create({}),
        interpolator=cfg.data.datamodule.interpolator,
        batch_size=OmegaConf.create(dict(train=2, val=2, test=2)),
        num_workers=OmegaConf.create(dict(train=0, val=0, test=0)),
        finetune_fraction=1.0,
    )
    source_cfg = OmegaConf.create(
        dict(
            _target_=f"{__name__}._AlternatePocketDataset",
            allowed_atom_symbols=["C", "N"],
        )
    )
    for split in ("val", "test"):
        wrapper = dm._wrap_dataset(source_cfg, split)
        source, _ = wrapper[0]
        assert source.contexts["pocket"].num_nodes == 2
        dataset = wrapper.base_dataset
        assert dataset.pocket_vocab == dm.pocket_vocab
        assert dataset_fingerprint(dataset)["split"] == split
        assert scaffold_cache_directory(dataset, tmp_path) == str(
            tmp_path / "alternate-source"
        )
    assert not list(tmp_path.iterdir())


def test_crossdocked_identity_and_namespace_match_published_format(cache):
    dataset = _tokenized(cache)
    try:
        storage = Path(dataset.lmdb_path) / "data.mdb"
        expected = {
            "class": (
                "morph.dataset.conditional.pocket.crossdocked.CrossDockedPocketDataset"
            ),
            "records": len(dataset),
            "split": "train",
            "hydrogen_mode": None,
            "processing_options": None,
            "sha256": hashlib.sha256(storage.read_bytes()).hexdigest(),
        }
        if dataset._indices is not None:
            expected["index_sha256"] = hashlib.sha256(
                dataset._indices.numpy().tobytes()
            ).hexdigest()
        assert dataset_fingerprint(dataset) == expected
        manifest = (cache / "processed/crossdocked_summary.json").read_bytes()
        old_namespace = hashlib.sha256(
            manifest + b"\0" + json.dumps(sorted(dataset.vocab.atom_tokens)).encode()
        ).hexdigest()
        assert dataset.cache_namespace() == old_namespace
        assert dataset.get_pocket_vocab().atom_tokens == ["C", "N", "O"]
    finally:
        dataset.close()


@pytest.mark.parametrize("storage_kind", ["qm9", "lmdb"])
def test_molecular_adapters_preserve_the_v2_source_identity(tmp_path, storage_kind):
    processed = tmp_path / "processed"
    processed.mkdir()
    record = Molecules().get(0)
    if storage_kind == "qm9":
        for split in ("train", "val", "test"):
            RevisedQM9.save([record], str(processed / f"{split}.pt"))
        dataset = FlowMatchingQM9Dataset(
            str(tmp_path), vocab=None, distributions=None, split="val"
        )
        storage = processed / "val.pt"
        hydrogen_mode, options = None, None
    else:
        storage = processed / "val_data.lmdb/data.mdb"
        write_bytes_to_lmdb([mol_to_bytes(record)], str(storage.parent))
        dataset = MolecularLMDBDataset(str(tmp_path), split="val")
        hydrogen_mode = "explicit"
        options = {
            "hydrogen_mode": "explicit",
            "max_atoms": None,
            "allowed_atom_symbols": None,
        }
    try:
        assert dataset_fingerprint(dataset) == {
            "class": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
            "records": 1,
            "split": "val",
            "hydrogen_mode": hydrogen_mode,
            "processing_options": options,
            "sha256": hashlib.sha256(storage.read_bytes()).hexdigest(),
        }
    finally:
        if storage_kind == "lmdb":
            dataset.close()


def test_custom_contexts_keep_graph_ids_through_batching_and_topology_edits():
    first, replay, last = _path_graph(2), _path_graph(1), _path_graph(3)
    first.contexts["landmarks"] = _path_graph(3, 20)
    last.contexts["landmarks"] = _path_graph(1, 40)
    first.contexts["landmarks"].contexts["reference"] = _path_graph(1, 30)
    batch = ChemBatch.from_data_list([first, replay, last])
    context = batch.contexts["landmarks"]
    assert context.num_nodes_per_graph.tolist() == [3, 0, 1]
    assert context.keep_mask.tolist() == [1, 1, 1, 1]
    assert context.y.tolist() == [[3.0], [0.0], [1.0]]
    assert context.contexts["reference"].num_nodes_per_graph.tolist() == [1, 0, 0]
    mean = batch.remove_com()
    torch.testing.assert_close(context.x[:3], first.contexts["landmarks"].x - mean[0])
    torch.testing.assert_close(context.x[3:], last.contexts["landmarks"].x - mean[2])
    # Deleting all ligand atoms in a row must not delete its independent context.
    filtered = filter_nodes(
        batch, torch.tensor([False, False, True, True, False, True])
    )
    assert filtered.num_nodes_per_graph.tolist() == [0, 1, 2]
    assert filtered.contexts["landmarks"] is context
    restored = filtered.to_data_list()
    assert [g.contexts["landmarks"].num_nodes for g in restored] == [3, 0, 1]
    rebatch = ChemBatch.from_data_list(restored)
    torch.testing.assert_close(rebatch.contexts["landmarks"].x, context.x)
    assert (
        first.filter_nodes(torch.tensor([True, False])).contexts["landmarks"]
        is first.contexts["landmarks"]
    )


def test_custom_contexts_clone_rotate_transfer_and_restore_together():
    original = _path_graph(2, 5)
    original.contexts["landmarks"] = _path_graph(3, 20)
    original.contexts["landmarks"].contexts["reference"] = _path_graph(1, 30)
    wrapper = object.__new__(FlowMatchingDatasetWrapper)
    wrapper.rotate = True
    rotated, *_ = wrapper._augment(original, _path_graph(1), _path_graph(1), 0.5)
    torch.testing.assert_close(
        torch.cdist(rotated.x, rotated.contexts["landmarks"].x),
        torch.cdist(original.x, original.contexts["landmarks"].x),
    )
    assert (
        rotated.contexts["landmarks"].x.data_ptr()
        != original.contexts["landmarks"].x.data_ptr()
    )
    copied = ChemAtomicData.from_atomic(original)
    copied.contexts["landmarks"].x.add_(1)
    torch.testing.assert_close(original.contexts["landmarks"].x, _path_graph(3, 20).x)
    batch = ChemBatch.from_data_list([rotated])
    moved = batch.to("cpu", dtype=torch.float64)
    assert moved.contexts["landmarks"].x.dtype == torch.float64
    assert moved.contexts["landmarks"].a.dtype == torch.long
    assert moved.contexts["landmarks"].contexts["reference"].x.dtype == torch.float64
    frame = _OutputFrame.from_batch(moved, 2.5, restore_input_frame=True)
    frame.recenter(moved)
    restored = frame.snapshot(moved).to_data_list()[0]
    torch.testing.assert_close(restored.x, rotated.x.double() * 2.5)
    torch.testing.assert_close(
        restored.contexts["landmarks"].x, rotated.contexts["landmarks"].x.double() * 2.5
    )
    torch.testing.assert_close(
        restored.contexts["landmarks"].contexts["reference"].x,
        rotated.contexts["landmarks"].contexts["reference"].x.double() * 2.5,
    )
