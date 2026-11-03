"""Structured insertion supervision and its two independent node index spaces."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from itertools import accumulate

import torch


class _TensorRecord:
    """Tensor operations shared by the nested insertion records."""

    def clone(self):
        """Copy every tensor, including the nested bond records."""
        return replace(
            self,
            **{field.name: getattr(self, field.name).clone() for field in fields(self)},
        )

    def to_inplace(self, device, non_blocking: bool = True):
        """Move all tensors while preserving integer index dtypes."""
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, _TensorRecord):
                value.to_inplace(device, non_blocking=non_blocking)
            else:
                setattr(self, field.name, value.to(device, non_blocking=non_blocking))
        return self

    def pin_memory(self):
        """Pin every tensor for DataLoader's asynchronous device transfer."""
        for field in fields(self):
            setattr(self, field.name, getattr(self, field.name).pin_memory())
        return self


@dataclass
class BondsToExisting(_TensorRecord):
    """Target bonds connecting a newly inserted atom to an already-existing atom.

    One aligned entry per supervised bond. ``spawn_idx`` / ``existing_idx`` index
    the interpolated graph (``mol_t``); ``inserted_idx`` indexes the insertion
    atoms (:class:`InsertionTargets` nodes).
    """

    spawn_idx: torch.Tensor  # existing node that spawned the inserted endpoint
    existing_idx: torch.Tensor  # the existing endpoint
    inserted_idx: torch.Tensor  # the inserted endpoint (local)
    bond_type: torch.Tensor  # target bond token


@dataclass
class BondsBetweenInserted(_TensorRecord):
    """Target bonds between two newly inserted atoms (upper triangle, src < dst).

    ``spawn_*`` index the interpolated graph (``mol_t``); ``src_idx`` / ``dst_idx``
    index the insertion atoms.
    """

    spawn_src_idx: torch.Tensor  # existing node that spawned the src inserted atom
    spawn_dst_idx: torch.Tensor  # existing node that spawned the dst inserted atom
    src_idx: torch.Tensor  # src inserted atom (local)
    dst_idx: torch.Tensor  # dst inserted atom (local)
    bond_type: torch.Tensor


@dataclass
class InsertionTargets(_TensorRecord):
    """Supervision for the atoms a model must insert, for one graph or a batch.

    ``positions`` / ``atom_types`` / ``charge_types`` are the future-inserted
    atoms. ``spawn_node_idx`` maps each inserted atom to the existing node it is
    spawned from (in ``mol_t``) -- a one-to-many ``insertion -> nearest node``
    map used by the GMM head and the insertion-edge supervision. The two bond
    groups carry per-bond edge-type targets.

    ``nearest_node_idx`` / ``nearest_ins_idx`` are the *inverse*, one-to-one
    ``node -> nearest insertion`` assignment used by the unimodal insertion head:
    for each existing node that spawns at least one insertion, ``nearest_node_idx``
    holds the node (in ``mol_t``) and ``nearest_ins_idx`` the single nearest
    pending insertion assigned to it (indexing these inserted atoms). Each node
    appears at most once, so the unimodal head has exactly one target per node.
    """

    positions: torch.Tensor
    atom_types: torch.Tensor
    charge_types: torch.Tensor
    spawn_node_idx: torch.Tensor
    bonds_to_existing: BondsToExisting
    bonds_between_inserted: BondsBetweenInserted
    nearest_node_idx: torch.Tensor
    nearest_ins_idx: torch.Tensor

    @property
    def num_nodes(self) -> int:
        """Number of future-inserted atoms."""
        return int(self.positions.shape[0])

    @staticmethod
    def empty(positions, atom_types, charge_types, device) -> InsertionTargets:
        """An insertion target with inserted atoms but no supervised bonds."""
        e = torch.empty((0,), device=device, dtype=torch.long)
        return InsertionTargets(
            positions=positions,
            atom_types=atom_types,
            charge_types=charge_types,
            spawn_node_idx=e,
            bonds_to_existing=BondsToExisting(e, e, e, e),
            bonds_between_inserted=BondsBetweenInserted(e, e, e, e, e),
            nearest_node_idx=e,
            nearest_ins_idx=e,
        )


def collate_insertion_targets(
    targets: list[InsertionTargets], current_node_counts: list[int]
) -> InsertionTargets:
    """Concatenate supervision without modifying the input samples.

    Spawn/existing indices receive offsets in the current molecular batch;
    inserted indices receive offsets in the concatenated future atoms. Empty
    insertion sets still contribute their current graph's node count.
    """
    if not targets or len(targets) != len(current_node_counts):
        raise ValueError("Expected one insertion target per current graph")
    node_offsets = list(accumulate(current_node_counts, initial=0))[:-1]
    ins_offsets = list(accumulate((t.num_nodes for t in targets), initial=0))[:-1]

    def cat(records, name, offsets=None):
        values = [getattr(record, name) for record in records]
        if offsets is not None:
            values = [
                value + offset for value, offset in zip(values, offsets, strict=True)
            ]
        return torch.cat(values)

    existing = [t.bonds_to_existing for t in targets]
    inserted = [t.bonds_between_inserted for t in targets]
    return InsertionTargets(
        positions=cat(targets, "positions"),
        atom_types=cat(targets, "atom_types"),
        charge_types=cat(targets, "charge_types"),
        spawn_node_idx=cat(targets, "spawn_node_idx", node_offsets),
        nearest_node_idx=cat(targets, "nearest_node_idx", node_offsets),
        nearest_ins_idx=cat(targets, "nearest_ins_idx", ins_offsets),
        bonds_to_existing=BondsToExisting(
            spawn_idx=cat(existing, "spawn_idx", node_offsets),
            existing_idx=cat(existing, "existing_idx", node_offsets),
            inserted_idx=cat(existing, "inserted_idx", ins_offsets),
            bond_type=cat(existing, "bond_type"),
        ),
        bonds_between_inserted=BondsBetweenInserted(
            spawn_src_idx=cat(inserted, "spawn_src_idx", node_offsets),
            spawn_dst_idx=cat(inserted, "spawn_dst_idx", node_offsets),
            src_idx=cat(inserted, "src_idx", ins_offsets),
            dst_idx=cat(inserted, "dst_idx", ins_offsets),
            bond_type=cat(inserted, "bond_type"),
        ),
    )
