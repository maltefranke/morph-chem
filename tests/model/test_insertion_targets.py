"""Insertion collation preserves both index spaces and input ownership."""

import torch

from morph.dataset.core.insertion_targets import (
    BondsBetweenInserted,
    BondsToExisting,
    InsertionTargets,
    collate_insertion_targets,
)
from morph.dataset.core.lightning_datamodule import LightningDataModule


def _targets():
    return InsertionTargets(
        positions=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        atom_types=torch.tensor([1, 0]),
        charge_types=torch.tensor([0, 0]),
        spawn_node_idx=torch.tensor([0, 2]),
        nearest_node_idx=torch.tensor([0, 2]),
        nearest_ins_idx=torch.tensor([0, 1]),
        bonds_to_existing=BondsToExisting(
            torch.tensor([0, 2]),
            torch.tensor([1, 0]),
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
        ),
        bonds_between_inserted=BondsBetweenInserted(
            torch.tensor([0]),
            torch.tensor([2]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([1]),
        ),
    )


def test_collation_offsets_both_spaces_across_an_empty_insertion_set():
    first, last = _targets(), _targets()
    empty = InsertionTargets.empty(
        torch.empty(0, 3),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        "cpu",
    )
    result = collate_insertion_targets([first, empty, last], [3, 4, 3])
    assert result.spawn_node_idx.tolist() == [0, 2, 7, 9]
    assert result.nearest_node_idx.tolist() == [0, 2, 7, 9]
    assert result.nearest_ins_idx.tolist() == [0, 1, 2, 3]
    assert result.bonds_to_existing.existing_idx.tolist() == [1, 0, 8, 7]
    assert result.bonds_to_existing.inserted_idx.tolist() == [0, 1, 2, 3]
    assert result.bonds_between_inserted.spawn_src_idx.tolist() == [0, 7]
    assert result.bonds_between_inserted.spawn_dst_idx.tolist() == [2, 9]
    assert result.bonds_between_inserted.src_idx.tolist() == [0, 2]
    assert result.bonds_between_inserted.dst_idx.tolist() == [1, 3]
    assert result.bonds_between_inserted.bond_type.tolist() == [1, 1]
    # A second collation must not apply offsets twice to the source samples.
    repeated = collate_insertion_targets([first, empty, last], [3, 4, 3])
    torch.testing.assert_close(repeated.spawn_node_idx, result.spawn_node_idx)
    assert last.spawn_node_idx.tolist() == [0, 2]
    result.positions.zero_()
    assert first.positions[0].tolist() == [1.0, 2.0, 3.0]


def test_nested_clone_and_lightning_transfer_preserve_integer_indices():
    original = _targets()
    copied = original.clone()
    copied.bonds_to_existing.spawn_idx.add_(10)
    assert original.bonds_to_existing.spawn_idx.tolist() == [0, 2]
    transferred = LightningDataModule.transfer_batch_to_device(
        None, (torch.tensor([0.5]), copied), "cpu", 0
    )[1]
    assert transferred is copied
    assert transferred.bonds_between_inserted.src_idx.dtype == torch.long
    assert transferred.positions.dtype == torch.float32
