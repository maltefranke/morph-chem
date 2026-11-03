"""Retained and to-be-deleted atoms can spawn fully supervised insertions."""

from types import SimpleNamespace

import pytest
import torch

from morph.model.losses.flow import FlowLossMixin
from morph.utils.loss_accumulation import LossAccumulator, UnifiedWeightedLoss


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ins_edge_head = torch.nn.Linear(1, 1)
        self.existing_logits = torch.zeros(9, 2, requires_grad=True)
        self.inserted_logits = torch.zeros(3, 2, requires_grad=True)

    def predict_inserted_to_existing_edges(self, **kwargs):
        return self.existing_logits

    def predict_inserted_to_inserted_edges(self, **kwargs):
        return self.inserted_logits


class _Harness(FlowLossMixin):
    device = "cpu"
    representation = SimpleNamespace(requires_topology=True, requires_charges=True)
    insertion_head_mode = "unimodal"
    atom_type_weights = edge_token_weights = charge_token_weights = None

    def __init__(self):
        self.model = _Model()
        names = [
            "x",
            "c",
            "do_del",
            "do_sub_a",
            "sub_a_class",
            "do_sub_e",
            "sub_e_class",
            "ins_rate",
            "ins_state",
            "ins_e",
            "ins_e_ii",
        ]
        self.loss_accumulator = LossAccumulator(
            UnifiedWeightedLoss(dict.fromkeys(names, 1.0), names), {"all": names}
        )

    def _node_loss_exclusion_mask(self, mols):
        return mols.keep_mask


def _batch(kept, deleted):
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    keep = torch.tensor(kept, dtype=torch.bool)
    deletion = torch.tensor(deleted, dtype=torch.float)
    keep &= deletion == 0
    mol = SimpleNamespace(
        num_graphs=1,
        batch=torch.zeros(3, dtype=torch.long),
        edge_index=edge_index,
        keep_mask=keep,
        lambda_del=deletion,
        lambda_ins=torch.ones(3),
        lambda_a_sub=torch.ones(3),
        lambda_e_sub=torch.ones(6),
        n_ins=torch.ones(3),
    )
    target = SimpleNamespace(
        x=torch.ones(3, 3),
        a=torch.ones(3, dtype=torch.long),
        c=torch.ones(3, dtype=torch.long),
        e=torch.ones(6, dtype=torch.long),
    )
    spawn = torch.arange(3).repeat_interleave(3)
    existing = torch.arange(3).repeat(3)
    src, dst = torch.triu_indices(3, 3, offset=1)
    insertions = SimpleNamespace(
        positions=torch.ones(3, 3),
        atom_types=torch.ones(3, dtype=torch.long),
        charge_types=torch.ones(3, dtype=torch.long),
        spawn_node_idx=torch.arange(3),
        nearest_node_idx=torch.arange(3),
        nearest_ins_idx=torch.arange(3),
        bonds_to_existing=SimpleNamespace(
            spawn_idx=spawn,
            existing_idx=existing,
            inserted_idx=spawn,
            bond_type=torch.ones(9, dtype=torch.long),
        ),
        bonds_between_inserted=SimpleNamespace(
            spawn_src_idx=src,
            spawn_dst_idx=dst,
            src_idx=src,
            dst_idx=dst,
            bond_type=torch.ones(3, dtype=torch.long),
        ),
    )
    return mol, target, insertions, torch.tensor([0.3])


def _predictions():
    return {
        "pos_head": torch.zeros(3, 3, requires_grad=True),
        "charge_head": torch.zeros(3, 2, requires_grad=True),
        "atom_type_head": torch.zeros(3, 2, requires_grad=True),
        "edge_type_head": torch.zeros(6, 2, requires_grad=True),
        "do_sub_a_head": torch.zeros(3, requires_grad=True),
        "do_sub_e_head": torch.zeros(6, requires_grad=True),
        "do_del_head": torch.zeros(3, requires_grad=True),
        "ins_rate_head": torch.full((3,), 0.5, requires_grad=True),
        "insertion_head": {
            "mu": torch.zeros(3, 3, requires_grad=True),
            "a_logits": torch.zeros(3, 2, requires_grad=True),
            "c_logits": torch.zeros(3, 2, requires_grad=True),
        },
    }


@pytest.mark.parametrize("kept", [[False] * 3, [True, False, False], [True] * 3])
@pytest.mark.parametrize("deleted", [[0, 0, 0], [0, 0, 1], [1, 1, 1]])
def test_insertion_losses_allow_retained_and_to_be_deleted_spawning_nodes(
    kept, deleted
):
    """Only deletion of an existing bond endpoint removes insertion supervision."""
    torch.set_num_threads(1)
    harness, predictions = _Harness(), _predictions()
    batch = _batch(kept, deleted)
    harness.loss_from_predictions(predictions, batch).backward()
    mol, _, insertions, _ = batch
    eligible = ~mol.lambda_del.bool()
    edges = insertions.bonds_to_existing
    expected = eligible[edges.existing_idx]
    got = harness.model.existing_logits.grad.abs().sum(1) > 0
    assert torch.equal(got, expected)
    # Future atoms survive regardless of whether either spawning node does.
    assert (harness.model.inserted_logits.grad.abs().sum(1) > 0).all()

    # Counts and inserted atom content remain supervised at every spawning node.
    assert (predictions["ins_rate_head"].grad.abs() > 0).all()
    for output in predictions["insertion_head"].values():
        assert (output.grad.abs().sum(1) > 0).all()

    # Existing kept atoms/edges retain their separate class-loss exclusions.
    free = eligible & ~mol.keep_mask
    assert torch.equal(predictions["atom_type_head"].grad.abs().sum(1) > 0, free)
    assert torch.equal(predictions["charge_head"].grad.abs().sum(1) > 0, free)
    # Retained coordinates are still denoised by default.
    assert torch.equal(predictions["pos_head"].grad.abs().sum(1) > 0, eligible)
    src, dst = mol.edge_index
    expected = (
        (src < dst)
        & eligible[src]
        & eligible[dst]
        & ~(mol.keep_mask[src] & mol.keep_mask[dst])
    )
    assert torch.equal(predictions["edge_type_head"].grad.abs().sum(1) > 0, expected)
