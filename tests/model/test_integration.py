"""Integration regression tests."""

from types import SimpleNamespace

import pytest
import torch

from morph.dataset.core.atomic_data import batch_from_flat
from morph.dataset.core.flow_matching_wrapper import uniform_n_atoms_range
from morph.dataset.core.vocab import Distributions, Vocab
from morph.flow_matching.integration import IntegrationPredictions, RateIntegrator
from morph.flow_matching.sampling_step import node_edit_probabilities
from morph.flow_matching.schedules import LinearSchedule, SmoothstepSchedule

from ._helpers import _noise_consistency_setup


def test_log_integration_schedule_has_configured_number_of_steps():
    integrator = object.__new__(RateIntegrator)
    integrator.time_strategy = "log"
    integrator.num_integration_steps = 100

    step_sizes = integrator.get_time_steps()

    assert len(step_sizes) == 100
    assert sum(step_sizes) == pytest.approx(0.999)


@pytest.mark.parametrize("num_steps", [0, -1, 2.5])
def test_invalid_integration_step_count_fails_instead_of_returning_priors(num_steps):
    integrator = object.__new__(RateIntegrator)
    integrator.time_strategy = "linear"
    integrator.num_integration_steps = num_steps

    with pytest.raises(ValueError, match="positive integer"):
        integrator.get_time_steps()
    integrator.num_integration_steps = 100
    with pytest.raises(ValueError, match="positive integer"):
        integrator.get_time_steps(num_steps)


@pytest.mark.parametrize("shape", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("parameter", ["time_beta_alpha", "time_beta_beta"])
def test_beta_time_grid_rejects_shapes_that_produce_nan_steps(shape, parameter):
    integrator = object.__new__(RateIntegrator)
    integrator.time_strategy = "beta"
    integrator.num_integration_steps = 10
    integrator.time_beta_alpha = integrator.time_beta_beta = 1.0
    setattr(integrator, parameter, shape)

    with pytest.raises(ValueError, match="finite and positive"):
        integrator.get_time_steps()


def test_schedule_integrated_rate_matches_closed_form():
    schedule = LinearSchedule()
    t = torch.tensor([0.2, 0.6])
    dt = torch.tensor([0.3, 0.1])

    actual = schedule.integrated_rate(t, dt)
    expected = torch.log((1.0 - t) / (1.0 - t - dt))

    torch.testing.assert_close(actual, expected)


def test_integrator_schedule_step_masses_select_exact_or_euler():
    integrator = object.__new__(RateIntegrator)
    schedule = SmoothstepSchedule(shift=0.8)
    integrator.sub_schedule = schedule
    integrator.sub_e_schedule = schedule
    integrator.ins_schedule = schedule
    integrator.del_schedule = schedule
    t = torch.tensor([0.25])
    dt = 0.2

    integrator.exact_schedule_integration = True
    exact = integrator.schedule_step_masses(t, dt)
    integrator.exact_schedule_integration = False
    euler = integrator.schedule_step_masses(t, dt)

    kappa_t = schedule.kappa_t(t)
    kappa_next = schedule.kappa_t(t + dt)
    expected_exact = (kappa_next - kappa_t) / (1.0 - kappa_t)
    expected_euler = schedule.rate(t) * dt

    torch.testing.assert_close(exact.substitution, expected_exact)
    torch.testing.assert_close(euler.substitution, expected_euler)
    assert not torch.allclose(exact.substitution, euler.substitution)


@pytest.mark.parametrize("time_strategy", ["log", "linear", "beta"])
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("num_steps", [1, 100])
def test_final_interval_completes_edits_for_every_time_grid(
    time_strategy, exact, num_steps
):
    _prior, _interpolator, integrator = _noise_consistency_setup()
    integrator.time_strategy = time_strategy
    integrator.exact_schedule_integration = exact
    integrator.num_integration_steps = num_steps
    # Include the production GEOM schedules and distinct node/edge schedules.
    integrator.ins_schedule = integrator.del_schedule = SmoothstepSchedule(0.8)
    integrator.sub_e_schedule = SmoothstepSchedule(1.5)
    t = torch.zeros(1)
    steps = integrator.get_time_steps()

    assert len(steps) == num_steps
    for index, dt in enumerate(steps):
        masses = integrator.schedule_step_masses(
            t, dt, is_final=index == len(steps) - 1
        )
        for name, schedule in (
            ("substitution", integrator.sub_schedule),
            ("edge_substitution", integrator.sub_e_schedule),
            ("insertion", integrator.ins_schedule),
            ("deletion", integrator.del_schedule),
        ):
            if index == len(steps) - 1:
                expected = torch.ones_like(t)
            elif exact:
                expected = -torch.expm1(-schedule.integrated_rate(t, dt))
            else:
                expected = schedule.rate(t) * dt
            torch.testing.assert_close(getattr(masses, name), expected, rtol=0, atol=0)
        t = t + dt


def test_final_interval_completion_accepts_graph_specific_intervals():
    _prior, _interpolator, integrator = _noise_consistency_setup()
    t = torch.tensor([0.5, 0.998, 0.9989], dtype=torch.float64)
    dt = torch.tensor([0.499, 0.001, 0.0001], dtype=torch.float64)

    masses = integrator.schedule_step_masses(t, dt, is_final=True)

    for mass in vars(masses).values():
        torch.testing.assert_close(mass, torch.ones_like(t))


@pytest.mark.parametrize("terminal", [False, True])
def test_terminal_edits_preserve_posteriors_masks_and_coordinate_update(
    monkeypatch, terminal
):
    _prior, _interpolator, integrator = _noise_consistency_setup()
    integrator.renoise_insertions = False
    # A draw above ordinary late-step masses but below the learned posterior
    # distinguishes completion from both regular integration and forced edits.
    monkeypatch.setattr(torch, "rand_like", lambda value: torch.full_like(value, 0.5))
    src, dst = torch.where(~torch.eye(4, dtype=torch.bool))
    mol = batch_from_flat(
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
        torch.zeros(4, dtype=torch.long),
        torch.ones(4, dtype=torch.long),
        torch.zeros(len(src), dtype=torch.long),
        torch.stack([src, dst]),
        torch.zeros(4, dtype=torch.long),
        num_graphs=1,
    )
    context = mol.clone()
    context.x.add_(20)
    context_positions = context.x.clone()
    mol.contexts["landmarks"] = context
    clean = mol.clone()
    clean.a = torch.ones_like(mol.a)
    clean.e = torch.ones_like(mol.e)
    clean.x = torch.zeros_like(mol.x)
    t = torch.tensor([0.9989 if terminal else 0.9988])
    dt = 0.0001
    edge_posterior = ((src == 0) & (dst == 3)) | ((src == 3) & (dst == 0))
    uncertain_edge = ((src == 2) & (dst == 3)) | ((src == 3) & (dst == 2))

    result, trace = integrator.integrate_step_gnn(
        mol_t=mol,
        predictions=IntegrationPredictions(
            mol_1_pred=clean,
            do_sub_a_probs=torch.tensor([0.75, 0.0, 0.0, 0.0]),
            do_sub_e_probs=edge_posterior.float() * 0.75
            + uncertain_edge.float() * 0.25,
            do_del_probs=torch.tensor([0.0, 0.75, 0.0, 0.0]),
            num_ins_pred=torch.tensor([0.0, 0.0, 0.75, 0.0]),
            insertion_preds={
                "mu": torch.zeros(4, 3),
                "a_logits": torch.tensor([-torch.inf, 0.0, -torch.inf]).repeat(4, 1),
                "c_logits": torch.tensor([-torch.inf, 0.0, -torch.inf]).repeat(4, 1),
            },
        ),
        t=t,
        dt=dt,
        is_final=terminal,
        return_step_trace=True,
    )

    assert trace.node_substitutions.tolist() == [terminal, False, False, False]
    assert trace.node_deletions.tolist() == [False, terminal, False, False]
    assert trace.insertions.tolist() == [False, False, terminal, False]
    assert int(trace.edge_substitutions.sum()) == int(terminal)
    assert result.a.tolist() == ([1, 0, 0, 1] if terminal else [0, 0, 0, 0])
    assert result.contexts["landmarks"] is context
    torch.testing.assert_close(context.x, context_positions)
    torch.testing.assert_close(
        trace.positions, mol.x + (clean.x - mol.x) * dt / (1 - t)
    )


def test_uniform_n_atoms_padding_is_explicit():
    distributions = SimpleNamespace(
        n_atoms_distribution=torch.tensor([0.0, 0.0, 0.0] + [1.0] * 27)
    )

    assert uniform_n_atoms_range(distributions, padding=5) == (3, 34)
    assert uniform_n_atoms_range(distributions, padding=0) == (3, 29)
    with pytest.raises(ValueError, match="non-negative"):
        uniform_n_atoms_range(distributions, padding=-1)


def test_node_edit_parameterization_preserves_conditional_split_when_saturated():
    edit = node_edit_probabilities(
        substitution=torch.tensor([0.8, 0.0]),
        deletion=torch.tensor([0.4, 0.0]),
    )

    torch.testing.assert_close(edit.any_edit, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(edit.deletion_given_edit, torch.tensor([1.0 / 3.0, 0.0]))


def test_exact_schedule_probability_precedes_model_posterior():
    integrated_schedule_hazard = torch.tensor([2.0])
    endpoint_posterior = torch.tensor([0.25])

    schedule_probability = -torch.expm1(-integrated_schedule_hazard)
    actual = endpoint_posterior * schedule_probability
    expected = 0.25 * (1.0 - torch.exp(torch.tensor([-2.0])))

    torch.testing.assert_close(actual, expected)
    assert actual.item() != pytest.approx(
        (1.0 - torch.exp(-endpoint_posterior * integrated_schedule_hazard)).item()
    )


def test_exact_node_edit_probability_adds_endpoint_action_masses():
    edit = node_edit_probabilities(
        substitution=torch.tensor([0.3]),
        deletion=torch.tensor([0.2]),
    )

    torch.testing.assert_close(edit.any_edit, torch.tensor([0.5]))
    torch.testing.assert_close(edit.deletion_given_edit, torch.tensor([0.4]))


def test_integrator_trace_records_the_production_transition():
    vocab = Vocab(
        atom_tokens=["C", "N"],
        edge_tokens=["<NO_BOND>", "1"],
        charge_tokens=["0"],
    )
    distributions = Distributions(
        atom_type_distribution=torch.tensor([0.5, 0.5]),
        edge_type_distribution=torch.tensor([0.5, 0.5]),
        charge_type_distribution=torch.tensor([1.0]),
        n_atoms_distribution=torch.tensor([0.0, 0.0, 1.0]),
        coordinate_std=1.0,
    )
    integrator = RateIntegrator(
        vocab=vocab,
        distributions=distributions,
        insertion_head_mode="unimodal",
        device="cpu",
    )
    edge_index = torch.tensor([[0, 1], [1, 0]])
    batch_idx = torch.zeros(2, dtype=torch.long)
    mol_t = batch_from_flat(
        torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        torch.tensor([0, 1]),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        edge_index,
        batch_idx,
        num_graphs=1,
        system_extras={"condition": torch.tensor([[3.0]])},
    )
    mol_1 = batch_from_flat(
        torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        torch.tensor([1, 0]),
        torch.zeros(2, dtype=torch.long),
        torch.ones(2, dtype=torch.long),
        edge_index,
        batch_idx,
        num_graphs=1,
    )

    mol_next, trace = integrator.integrate_step_gnn(
        mol_t=mol_t,
        predictions=IntegrationPredictions(
            mol_1_pred=mol_1,
            do_sub_a_probs=torch.zeros(2),
            do_sub_e_probs=torch.zeros(2),
            do_del_probs=torch.zeros(2),
            num_ins_pred=torch.zeros(2),
            insertion_preds={
                "mu": torch.zeros(2, 3),
                "a_logits": torch.zeros(2, 2),
                "c_logits": torch.zeros(2, 1),
            },
        ),
        t=torch.tensor([0.2]),
        dt=0.1,
        return_step_trace=True,
        position_update=lambda x, _x1, _t, _dt: x + 0.25,
    )

    torch.testing.assert_close(mol_next.x, mol_t.x + 0.25)
    torch.testing.assert_close(trace.positions, mol_next.x)
    assert not trace.node_substitutions.any()
    assert not trace.node_deletions.any()
    assert not trace.edge_substitutions.any()
    assert not trace.insertions.any()
    assert trace.insertion is None
    torch.testing.assert_close(mol_next.condition, mol_t.condition)


def test_insertion_bond_prediction_uses_clean_atoms(monkeypatch):
    _prior, _interpolator, integrator = _noise_consistency_setup()
    edge_index = torch.tensor([[0, 1], [1, 0]])
    batch = torch.zeros(2, dtype=torch.long)
    mol = batch_from_flat(
        positions=torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atom_types=torch.tensor([0, 1]),
        charge_types=torch.ones(2, dtype=torch.long),
        edge_types=torch.zeros(2, dtype=torch.long),
        edge_index=edge_index,
        batch_idx=batch,
        num_graphs=1,
    )
    observed = {}

    def capture_clean_atoms(_context, new_atoms):
        observed["atoms"] = new_atoms
        return None

    monkeypatch.setattr(
        integrator, "_predict_inserted_to_existing_edges", capture_clean_atoms
    )
    monkeypatch.setattr(
        integrator,
        "_predict_inserted_to_inserted_edges",
        lambda _context, _new_atoms: None,
    )
    clean_position = torch.tensor([[3.0, 0.0, 0.0]])
    torch.manual_seed(4)
    result, trace = integrator.integrate_step_gnn(
        mol_t=mol,
        predictions=IntegrationPredictions(
            mol_1_pred=mol,
            do_sub_a_probs=torch.zeros(2),
            do_sub_e_probs=torch.zeros(2),
            do_del_probs=torch.zeros(2),
            num_ins_pred=torch.tensor([1.0, 0.0]),
            insertion_preds={
                "mu": torch.cat([clean_position, torch.zeros(1, 3)]),
                "a_logits": torch.zeros(2, 3),
                "c_logits": torch.zeros(2, 3),
            },
            h_latent=torch.zeros(2, 4),
        ),
        t=torch.tensor([0.0]),
        dt=0.0,
        ins_edge_head=object(),
        return_step_trace=True,
        is_final=True,
    )

    torch.testing.assert_close(observed["atoms"].x, clean_position)
    torch.testing.assert_close(trace.insertion.clean_atoms.x, clean_position)
    assert not torch.equal(result.x[-1:], clean_position)
