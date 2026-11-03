"""Loss contracts retain per-graph weighting and zero-gradient dependencies."""

import pytest
import torch

from morph.dataset.core.atomic_data import batch_from_flat
from morph.dataset.core.insertion_targets import InsertionTargets
from morph.dataset.core.representation import Representation
from morph.model.lightning._training import TrainingMixin
from morph.utils.loss_accumulation import LossAccumulator, LossTerm, UnifiedWeightedLoss


def test_atom_action_loss_is_weighted_before_graph_averaging():
    trainer = TrainingMixin()
    trainer.device = "cpu"
    trainer.representation = Representation.POINTCLOUD
    trainer.atom_type_weights = torch.ones(1)
    mol = batch_from_flat(
        torch.zeros(2, 3),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty(2, 0, dtype=torch.long),
        torch.arange(2),
        num_graphs=2,
    )
    # Numerical lambdas remain rates; their positivity determines action labels.
    mol.lambda_a_sub = torch.tensor([2.0, 7.0])
    logits = torch.tensor([0.0, 2.0], requires_grad=True)
    predictions = {
        "do_sub_a_head": logits,
        "atom_type_head": torch.zeros(2, 1),
        "do_sub_e_head": torch.empty(0, 1),
        "edge_type_head": torch.empty(0, 1),
    }
    losses = trainer._substitution_losses(
        predictions,
        mol,
        mol,
        torch.ones(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        None,
    )
    accumulator = LossAccumulator(
        UnifiedWeightedLoss({"do_sub_a": 1.0}, ["do_sub_a"]),
        {"sub": ["do_sub_a"]},
        {"sub": lambda t: 1.0 + 2.0 * t},
    )
    accumulator.set_batch_losses(
        {"do_sub_a": losses["do_sub_a"]}, t=torch.tensor([0.0, 1.0])
    )
    loss = accumulator.total_loss()
    expected = (torch.nn.functional.softplus(-logits) * torch.tensor([1.0, 3.0])).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(
        logits.grad, (logits.detach().sigmoid() - 1) * torch.tensor([0.5, 1.5])
    )


def test_empty_insertion_losses_keep_per_graph_shapes_and_gradients():
    trainer = TrainingMixin()
    trainer.device = "cpu"
    trainer.model = torch.nn.Module()
    trainer.model.insertion_head = torch.nn.Linear(2, 3)
    mol = batch_from_flat(
        torch.zeros(2, 3),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty(2, 0, dtype=torch.long),
        torch.arange(2),
        num_graphs=2,
    )
    mol.n_ins = mol.lambda_del = torch.zeros(2)
    targets = InsertionTargets.empty(
        torch.empty(0, 3),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        "cpu",
    )
    losses = trainer._insertion_deletion_losses(
        {
            "ins_rate_head": torch.ones(2),
            "insertion_head": {},
            "do_del_head": torch.zeros(2),
        },
        targets,
        mol,
        torch.ones(2, dtype=torch.bool),
    )
    assert all(term.values.shape == (2, 1) for term in losses.values())
    assert not losses["ins_state"].graph_mask.any()
    losses["ins_state"].values.sum().backward()
    for parameter in trainer.model.insertion_head.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def test_pre_reduced_losses_are_rejected():
    with pytest.raises(ValueError, match="per-graph"):
        LossTerm(torch.tensor(1.0))


def test_loss_accumulator_uses_live_weight_device_and_clears_batch_stats():
    weights = UnifiedWeightedLoss({"x": 2.0}, ["x"])
    accumulator = LossAccumulator(weights, {"state": ["x"]})
    accumulator.add_stat("previous_batch", torch.tensor(3.0))

    accumulator.begin_batch()
    accumulator.add_stat("current_batch", torch.tensor(4.0))
    accumulator.set_batch_losses({"x": LossTerm(torch.tensor([1.0, 2.0]))})

    logged = accumulator.log_dict()
    assert "stats/previous_batch" not in logged
    assert logged["stats/current_batch"] == 4.0
    assert weights.get_weight_tensors()["x"].device == weights.manual_weights.device
