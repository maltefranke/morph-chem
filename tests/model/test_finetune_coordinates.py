"""Scaffold loss-mask and inference coordinate-frame regressions."""

from types import SimpleNamespace

import pytest
import torch

from morph.dataset.core.atomic_data import batch_from_flat
from morph.dataset.core.vocab import Distributions, Vocab
from morph.flow_matching.integration import IntegrationPredictions, RateIntegrator
from morph.model.finetune.base import LightningModuleRatesFinetune
from morph.model.finetune.scaffold import LightningModuleRatesScaffoldFinetune
from morph.model.lightning._sampling import SamplingMixin


class _MaskHarness:
    _apply_inference_edit_masks = (
        LightningModuleRatesFinetune._apply_inference_edit_masks
    )


def test_scaffold_loss_exclusions_are_independent():
    class Harness:
        _node_loss_exclusion_mask = (
            LightningModuleRatesScaffoldFinetune._node_loss_exclusion_mask
        )
        _position_loss_exclusion_mask = (
            LightningModuleRatesScaffoldFinetune._position_loss_exclusion_mask
        )

    mol = SimpleNamespace(keep_mask=torch.tensor([True, False]))
    harness = Harness()
    harness._exclude_scaffold_from_sub_loss = True
    harness._exclude_scaffold_from_x_loss = False
    assert harness._node_loss_exclusion_mask(mol).tolist() == [True, False]
    assert harness._position_loss_exclusion_mask(mol) is None

    harness._exclude_scaffold_from_sub_loss = False
    harness._exclude_scaffold_from_x_loss = True
    assert harness._node_loss_exclusion_mask(mol) is None
    assert harness._position_loss_exclusion_mask(mol).tolist() == [True, False]


def test_position_freeze_only_pins_kept_clean_predictions():
    harness = _MaskHarness()
    harness._freeze_keep_positions = True
    mol = SimpleNamespace(
        positions=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        keep_mask=torch.tensor([True, False]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
    )
    x1 = torch.zeros_like(mol.positions)
    node_probs = torch.ones(2)
    edge_probs = torch.ones(2)
    insertion_rates = torch.ones(2)

    pinned, sub_a, sub_e, delete, insertion = harness._apply_inference_edit_masks(
        mol, x1, node_probs, edge_probs, node_probs, insertion_rates
    )

    torch.testing.assert_close(pinned[0], mol.positions[0])
    torch.testing.assert_close(pinned[1], x1[1])
    assert sub_a.tolist() == [0.0, 1.0]
    assert delete.tolist() == [0.0, 1.0]
    assert sub_e.tolist() == [1.0, 1.0]
    assert insertion is insertion_rates

    harness._freeze_keep_positions = False
    flowing, *_ = harness._apply_inference_edit_masks(
        mol, x1, node_probs, edge_probs, node_probs, insertion_rates
    )
    assert flowing is x1


def test_integrator_carries_geometric_context_across_a_step():
    mol, clean = _sampling_batch()
    pocket = mol.contexts["pocket"]
    distributions = Distributions(
        atom_type_distribution=torch.ones(1),
        edge_type_distribution=torch.ones(1),
        charge_type_distribution=torch.ones(1),
        n_atoms_distribution=torch.ones(2),
        coordinate_std=1.0,
    )
    integrator = RateIntegrator(
        vocab=Vocab(atom_tokens=["C"], edge_tokens=["<NO_BOND>"], charge_tokens=["0"]),
        distributions=distributions,
        insertion_head_mode="unimodal",
        num_integration_steps=1,
        time_strategy="linear",
        renoise_insertions=False,
        device="cpu",
    )

    result = integrator.integrate_step_gnn(
        mol_t=mol,
        predictions=IntegrationPredictions(
            mol_1_pred=clean,
            do_sub_a_probs=torch.zeros(mol.num_nodes),
            do_sub_e_probs=torch.zeros(mol.num_edges),
            do_del_probs=torch.zeros(mol.num_nodes),
            num_ins_pred=torch.zeros(mol.num_nodes),
            insertion_preds={},
        ),
        t=torch.zeros(1),
        dt=0.1,
    )

    assert result.contexts["pocket"] is pocket


class _Model:
    ins_edge_head = None

    def __call__(self, mol, _t, **_kwargs):
        num_nodes = mol.num_nodes
        num_edges = mol.num_edges
        x1 = torch.full_like(mol.x, 4.0)
        return {
            "pos_head": x1,
            "atom_type_head": torch.zeros(num_nodes, 1),
            "charge_head": torch.zeros(num_nodes, 1),
            "edge_type_head": torch.zeros(num_edges, 1),
            "insertion_head": {},
            "ins_rate_head": torch.zeros(num_nodes, 1),
            "do_sub_a_head": torch.zeros(num_nodes, 1),
            "do_sub_e_head": torch.zeros(num_edges, 1),
            "do_del_head": torch.full((num_nodes, 1), -100.0),
            "h_latent": torch.zeros(num_nodes, 1),
        }


class _Integrator:
    @staticmethod
    def get_time_steps():
        return [0.5, 0.5]

    @staticmethod
    def integrate_step_gnn(*, mol_t, predictions, t, dt, **_kwargs):
        result = mol_t.clone()
        rate = 1.0 / (1.0 - t[result.batch])
        result.x = result.x + (predictions.mol_1_pred.x - result.x) * rate[:, None] * dt
        return result


class _SamplingHarness(SamplingMixin):
    _apply_inference_charge_mask = (
        LightningModuleRatesFinetune._apply_inference_charge_mask
    )
    _apply_inference_edit_masks = (
        LightningModuleRatesFinetune._apply_inference_edit_masks
    )
    device = torch.device("cpu")
    _freeze_keep_positions = True
    _freeze_keep_charges = True
    vocab = SimpleNamespace(atom_tokens=("C",), charge_tokens=(0,))
    representation = SimpleNamespace(requires_topology=True, requires_charges=True)
    distributions = SimpleNamespace(coordinate_std=2.0)
    integrator = _Integrator()

    @staticmethod
    def _get_model():
        return _Model()


def _sampling_batch():
    mol = batch_from_flat(
        positions=torch.tensor([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]]),
        atom_types=torch.zeros(2, dtype=torch.long),
        charge_types=torch.zeros(2, dtype=torch.long),
        edge_types=torch.zeros(2, dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        batch_idx=torch.zeros(2, dtype=torch.long),
        num_graphs=1,
        node_extras={"keep_mask": torch.tensor([True, False])},
        _already_sorted=True,
    )
    pocket = batch_from_flat(
        positions=torch.tensor([[9.0, 1.0, 0.0], [15.0, -1.0, 0.0]]),
        atom_types=torch.zeros(2, dtype=torch.long),
        charge_types=torch.zeros(2, dtype=torch.long),
        edge_types=torch.zeros(2, dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        batch_idx=torch.zeros(2, dtype=torch.long),
        num_graphs=1,
        _already_sorted=True,
    )
    mol.contexts["pocket"] = pocket
    return mol, mol.clone()


def test_sampling_restores_frozen_positions_to_input_frame():
    result = _SamplingHarness().sample(_sampling_batch(), batch_idx=0)

    # Sampling outputs data units; the kept input coordinate was 10 model units.
    torch.testing.assert_close(result.x[0], torch.tensor([20.0, 0.0, 0.0]))
    torch.testing.assert_close(
        result.contexts["pocket"].x,
        torch.tensor([[18.0, 2.0, 0.0], [30.0, -2.0, 0.0]]),
    )


def test_sampling_trajectory_restores_every_frame_to_input_frame():
    trajectory = _SamplingHarness().sample(
        _sampling_batch(), batch_idx=0, return_traj=True
    )

    for frame in trajectory[0]:
        torch.testing.assert_close(frame.x[0], torch.tensor([20.0, 0.0, 0.0]))


@pytest.mark.parametrize("identity_prediction", [True, False])
def test_movable_scaffold_inference_uses_model_updates_without_jitter(
    monkeypatch, identity_prediction
):
    class Model(_Model):
        def __call__(self, mol, t, **kwargs):
            predictions = super().__call__(mol, t, **kwargs)
            if identity_prediction:
                predictions["pos_head"] = mol.x.clone()
            return predictions

    def unexpected_noise(*args, **kwargs):
        raise AssertionError("Scaffold inference must not add coordinate jitter")

    monkeypatch.setattr(torch, "randn_like", unexpected_noise)
    harness = _SamplingHarness()
    harness._freeze_keep_positions = False
    harness._get_model = Model
    source, target = _sampling_batch()
    source.keep_mask[:] = True
    result = harness.sample((source, target), batch_idx=0)
    expected_distance = 4.0 if identity_prediction else 0.0
    assert float((result.x[1] - result.x[0]).norm()) == pytest.approx(expected_distance)


@pytest.mark.parametrize("freeze_charges", [True, False])
@pytest.mark.parametrize("with_scaffold", [True, False])
def test_sampling_charge_policy_applies_at_every_step(freeze_charges, with_scaffold):
    class ChargedModel(_Model):
        def __call__(self, mol, t, **kwargs):
            predictions = super().__call__(mol, t, **kwargs)
            # Force a charge different from the input, including at kept atoms.
            predictions["charge_head"] = torch.tensor([-torch.inf, 0.0]).repeat(
                mol.num_nodes, 1
            )
            return predictions

    harness = _SamplingHarness()
    harness._freeze_keep_charges = freeze_charges
    harness._get_model = ChargedModel
    harness.vocab = Vocab(
        atom_tokens=["C"], edge_tokens=["<NO_BOND>"], charge_tokens=["-1", "1"]
    )
    harness.integrator = RateIntegrator(
        vocab=harness.vocab,
        distributions=Distributions(
            atom_type_distribution=torch.ones(1),
            edge_type_distribution=torch.ones(1),
            charge_type_distribution=torch.ones(2),
            n_atoms_distribution=torch.ones(3),
            coordinate_std=2.0,
        ),
        insertion_head_mode="unimodal",
        num_integration_steps=2,
        time_strategy="linear",
        renoise_insertions=False,
        device="cpu",
    )
    batch = _sampling_batch()
    if not with_scaffold:
        batch = (batch_from_flat(*batch[0].unpack(), num_graphs=1), batch[1])
    charges_by_step = []
    result = harness.sample(
        batch,
        batch_idx=0,
        sampling_step_callback=lambda trace: charges_by_step.append(
            trace.mol_t_next.c.clone()
        ),
    )
    expected = torch.tensor([0 if freeze_charges and with_scaffold else 1, 1])
    assert len(charges_by_step) == 2
    for charges in [*charges_by_step, result.c]:
        torch.testing.assert_close(charges, expected)


@pytest.mark.parametrize("return_traj", [False, True])
@pytest.mark.parametrize("t_start", [0.0, 0.5])
def test_sampling_always_marks_the_final_iteration(return_traj, t_start):
    class Integrator(_Integrator):
        def __init__(self):
            self.final_steps = []

        @staticmethod
        def get_time_steps():
            # Use the real default grid, including its endpoint below one.
            integrator = object.__new__(RateIntegrator)
            integrator.time_strategy = "log"
            integrator.num_integration_steps = 100
            return integrator.get_time_steps()

        def integrate_step_gnn(self, *, is_final, **kwargs):
            self.final_steps.append(is_final)
            return super().integrate_step_gnn(**kwargs)

    harness = _SamplingHarness()
    harness.integrator = Integrator()
    result = harness.sample(
        _sampling_batch(), batch_idx=0, return_traj=return_traj, t_start=t_start
    )

    flags = harness.integrator.final_steps
    assert flags[-1] is True
    assert not any(flags[:-1])
    if t_start == 0.0:
        assert len(flags) == 100
    else:
        assert 0 < len(flags) < 100
    if return_traj:
        assert len(result[0]) == len(flags) + 1


def test_compiled_sampling_resolves_graph_size_outside_the_model():
    class Model(torch.nn.Module):
        def forward(self, mol, t, max_seqlen=None):
            if max_seqlen is None:
                raise RuntimeError("compiled sampling requires a resolved graph size")
            return _Model()(mol, t)

    compiled = Model()
    compiled.compile(backend="eager", fullgraph=True, dynamic=True)
    harness = _SamplingHarness()
    harness._get_model = lambda: compiled
    with torch.inference_mode():
        result = harness.sample(_sampling_batch(), batch_idx=0)
        expected = _SamplingHarness().sample(_sampling_batch(), batch_idx=0)
    torch.testing.assert_close(result.x, expected.x)
    torch.testing.assert_close(
        result.contexts["pocket"].x, expected.contexts["pocket"].x
    )
