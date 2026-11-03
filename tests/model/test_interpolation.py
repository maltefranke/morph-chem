"""Interpolation regression tests."""

import copy
from types import SimpleNamespace

import pytest
import torch

import morph.flow_matching.interpolation as interpolation
from morph.dataset.core.atomic_data import ChemAtomicData
from morph.dataset.core.flow_matching_wrapper import (
    FlowMatchingDatasetWrapper,
    train_collate_fn,
)
from morph.dataset.core.insertion_data import InsertedAtoms
from morph.dataset.core.representation import Representation, make_uniform_type_prior
from morph.dataset.core.vocab import Vocab
from morph.flow_matching.interpolation import Interpolator
from morph.flow_matching.scaffold_insertion import scaffold_insertion_candidates
from morph.flow_matching.schedules import LinearSchedule

from ._helpers import (
    _complete_edges,
    _noise_consistency_setup,
)


@pytest.mark.parametrize("scaffold", [False, True])
@pytest.mark.parametrize("time", [0.0, 0.5, 1.0])
def test_prior_context_preserves_aligned_source_through_edits_and_augmentation(
    scaffold, time
):
    prior, _, _ = _noise_consistency_setup()
    interpolator = Interpolator(
        prior,
        pre_align_iters=0,
        c_sub=1000,
        c_ins=50,
        c_del=50,
        atom_type_noise_prob=1.0,
    )
    source = ChemAtomicData(
        positions=torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.0, 2, 0], [30.0, 30, 30]]),
        atom_types=torch.tensor([0, 1, 2, 0]),
        charge_types=torch.tensor([0, 1, 2, 0]),
        neighbor_list=_complete_edges(4).T,
        edge_types=torch.ones(12, dtype=torch.long),
    )
    rotation = torch.tensor([[0.0, -1, 0], [1.0, 0, 0], [0.0, 0, 1]])
    translation = torch.tensor([0.25, 0.5, -0.25])
    aligned_positions = source.x @ rotation + translation
    target = source.clone()
    target.x = aligned_positions.clone()
    # One source-only atom and one target-only atom exercise both edit paths.
    target.x[-1] = -30
    target.a[-1] = 1
    target.permute_nodes([2, 0, 3, 1])
    pairs = [(0, 1), (1, 3), (2, 0)] if scaffold else None

    torch.manual_seed(42)
    baseline = interpolator.interpolate_single(source, target, time, pairs)
    baseline_rng = torch.random.get_rng_state()
    assert "prior" not in baseline[0].contexts
    interpolator.include_prior_context = True
    torch.manual_seed(42)
    mol_t, mol_1, insertions = interpolator.interpolate_single(
        source, target, time, pairs
    )
    torch.testing.assert_close(torch.random.get_rng_state(), baseline_rng)
    for actual, expected in zip((mol_t, mol_1, insertions), baseline, strict=True):
        for name in ("positions", "atom_types", "charge_types"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
    torch.testing.assert_close(mol_t.extra, baseline[0].extra)

    context = mol_t.contexts["prior"]
    assert context.num_nodes == source.num_nodes
    assert not context.is_auxiliary.any()
    torch.testing.assert_close(context.x, aligned_positions - mol_t.frame_shift)
    for name in ("a", "c", "e", "edge_index"):
        torch.testing.assert_close(getattr(context, name), getattr(source, name))

    wrapper = object.__new__(FlowMatchingDatasetWrapper)
    wrapper.rotate = True
    wrapper.n_augmentations = 2
    augmented = wrapper._training_output(mol_t, mol_1, insertions, torch.tensor(time))
    batch, *_ = train_collate_fn([augmented])
    assert batch.contexts["prior"].num_nodes_per_graph.tolist() == [4, 4]
    expected_distances = torch.cdist(mol_t.x, context.x)
    for graph in batch.to_data_list():
        torch.testing.assert_close(
            torch.cdist(graph.x, graph.contexts["prior"].x),
            expected_distances,
            atol=1e-4,
            rtol=1e-5,
        )
    augmented[0][0].contexts["prior"].x.add_(100)
    torch.testing.assert_close(context.x, aligned_positions - mol_t.frame_shift)


def test_interpolator_and_integrator_share_the_uniform_type_prior():
    prior, interpolator, integrator = _noise_consistency_setup()

    for training_cat, inference_cat in (
        (interpolator._cat_atom, integrator._cat_atom),
        (interpolator._cat_edge, integrator._cat_edge),
    ):
        torch.testing.assert_close(training_cat.probs, inference_cat.probs)
        torch.testing.assert_close(
            inference_cat.probs,
            torch.full_like(inference_cat.probs, 1.0 / inference_cat.probs.numel()),
        )
    torch.testing.assert_close(
        prior.charge_type_distribution, torch.full((3,), 1.0 / 3.0)
    )


def test_categorical_noise_uses_empirical_alternatives_and_preserves_masks():
    values = torch.tensor([0, 0, 1, 2])
    eligible = torch.tensor([True, False, True, True])
    empirical = torch.tensor([0.0, 1.0, 0.0])
    noisy = Interpolator._corrupt_categorical(values, eligible, 1.0, empirical)
    assert noisy[0] == noisy[3] == 1
    assert noisy[1] == values[1]
    assert noisy[2] != values[2]  # Uniform alternative fallback for token 1.
    torch.testing.assert_close(values, torch.tensor([0, 0, 1, 2]))
    before = torch.random.get_rng_state()
    assert Interpolator._corrupt_categorical(values, eligible, 0.0, empirical) is values
    torch.testing.assert_close(torch.random.get_rng_state(), before)


def test_categorical_noise_balances_pairs_and_keeps_uniform_source_prior():
    prior, _, _ = _noise_consistency_setup()
    empirical = copy.deepcopy(prior)
    empirical.atom_type_distribution = torch.tensor([0.9, 0.05, 0.05])
    empirical.edge_type_distribution = torch.tensor([0.98, 0.02])
    noisy = Interpolator(
        prior, atom_type_noise_prob=0.01, noise_distributions=empirical
    )
    torch.testing.assert_close(noisy._cat_atom.probs, prior.atom_type_distribution)
    torch.testing.assert_close(
        noisy._noise_atom_probs, empirical.atom_type_distribution
    )
    torch.testing.assert_close(
        noisy._noise_edge_probs, empirical.edge_type_distribution
    )
    for n in (1, 30, 45, 70):
        pairs = n * (n - 1) // 2
        expected = 0.01 * n if pairs else 0.0
        actual = pairs * noisy._balanced_edge_noise_probability(n, pairs)
        assert actual == pytest.approx(expected)
    prior.edge_type_distribution = torch.tensor([1.0, 0.0])
    no_topology = Interpolator(
        prior, atom_type_noise_prob=1.0, noise_distributions=empirical
    )
    assert no_topology._balanced_edge_noise_probability(45, 990) == 0.0


def test_categorical_noise_retargets_existing_losses_and_excludes_deletions(
    monkeypatch,
):
    prior, _, _ = _noise_consistency_setup()
    common = dict(
        distributions=prior,
        move_noise_scale=0.0,
        ins_noise_scale=0.0,
        # This endpoint corruption check needs a nonzero rate at t=1;
        # GEOM's default smoothstep schedule has zero endpoint derivative.
        sub_schedule=LinearSchedule(),
        del_schedule=SimpleNamespace(
            kappa_and_rate=lambda t: (torch.zeros_like(t), torch.ones_like(t))
        ),
    )
    clean = Interpolator(**common)
    noisy = Interpolator(**common, atom_type_noise_prob=1.0)
    edges = _complete_edges(4)
    source = ChemAtomicData(
        positions=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        atom_types=torch.tensor([0, 1, 2, 0]),
        charge_types=torch.ones(4, dtype=torch.long),
        edge_types=torch.zeros(12, dtype=torch.long),
        neighbor_list=edges.T,
        is_auxiliary=torch.tensor([[False], [False], [False], [True]]),
    )
    target = source.clone()
    target.is_auxiliary = torch.tensor([[False], [False], [True], [False]])
    target.atom_types = torch.tensor([1, 2, 0, 1])
    monkeypatch.setattr(
        interpolation,
        "partial_optimal_transport_single",
        lambda source, target, **kwargs: (source.clone(), target.clone()),
    )
    torch.manual_seed(42)
    baseline, target_baseline, _ = clean.interpolate_single(source, target, 1.0)
    torch.manual_seed(42)
    actual, target_actual, _ = noisy.interpolate_single(source, target, 1.0)
    deleted = actual.lambda_del > 0
    assert deleted.tolist() == [False, False, True, False]
    torch.testing.assert_close(actual.a[deleted], baseline.a[deleted])
    assert (actual.a[~deleted] != target_actual.a[~deleted]).all()
    assert (actual.lambda_a_sub[~deleted] > 0).all()
    assert (actual.lambda_a_sub[deleted] == 0).all()
    protected_edges = deleted[actual.edge_index].any(dim=0)
    torch.testing.assert_close(actual.e[protected_edges], baseline.e[protected_edges])
    assert (actual.e[~protected_edges] != target_actual.e[~protected_edges]).all()
    assert (actual.lambda_e_sub[~protected_edges] > 0).all()
    for name in ("x", "a", "c", "e"):
        torch.testing.assert_close(
            getattr(target_actual, name), getattr(target_baseline, name)
        )
    torch.testing.assert_close(actual.x, baseline.x)
    torch.testing.assert_close(actual.c, baseline.c)
    dense = torch.zeros(4, 4, dtype=actual.e.dtype)
    dense[tuple(actual.edge_index)] = actual.e
    torch.testing.assert_close(dense, dense.T)


def test_categorical_noise_preserves_locked_scaffold(monkeypatch):
    prior, _, _ = _noise_consistency_setup()
    noisy = Interpolator(prior, atom_type_noise_prob=1.0, move_noise_scale=0.0)
    source = ChemAtomicData(
        positions=torch.randn(3, 3),
        atom_types=torch.tensor([0, 1, 2]),
        charge_types=torch.ones(3, dtype=torch.long),
        edge_types=torch.zeros(6, dtype=torch.long),
        neighbor_list=_complete_edges(3).T,
        is_auxiliary=torch.zeros(3, 1, dtype=torch.bool),
    )
    monkeypatch.setattr(
        interpolation,
        "scaffold_based_assignment_single",
        lambda source, target, **kwargs: (source.clone(), target.clone()),
    )
    actual, target, _ = noisy.interpolate_single(
        source, source, 1.0, scaffold_pairs=[(0, 0), (1, 1)]
    )
    torch.testing.assert_close(actual.a[:2], target.a[:2])
    assert actual.a[2] != target.a[2]
    protected = (actual.edge_index < 2).all(dim=0)
    torch.testing.assert_close(actual.e[protected], target.e[protected])


def test_insertion_renoising_matches_training_scale_and_keeps_charge_clean():
    _prior, interpolator, integrator = _noise_consistency_setup()
    torch.testing.assert_close(
        torch.tensor(integrator.ins_noise_scale),
        torch.tensor(interpolator.ins_noise_scale),
    )

    atoms = InsertedAtoms(
        x=torch.zeros(2, 3),
        a=torch.tensor([0, 2]),
        c=torch.tensor([0, 2]),
    )
    clean_charge = atoms.c.clone()
    t = torch.tensor([0.25, 0.75])

    torch.manual_seed(7)
    expected_noise = torch.randn_like(atoms.x)
    expected_x = expected_noise * ((1.0 - t) * interpolator.ins_noise_scale).unsqueeze(
        -1
    )
    torch.manual_seed(7)
    noised = integrator._renoise_insertions(atoms, t)

    torch.testing.assert_close(noised.x, expected_x)
    torch.testing.assert_close(noised.c, clean_charge)


def test_present_insertions_receive_base_coordinate_jitter(monkeypatch):
    _prior, interpolator, _integrator = _noise_consistency_setup()
    edge_index = torch.tensor([[0, 1], [1, 0]])
    sample = ChemAtomicData(
        positions=torch.zeros(2, 3),
        atom_types=torch.tensor([0, 0]),
        charge_types=torch.tensor([1, 1]),
        edge_types=torch.zeros(2, dtype=torch.long),
        neighbor_list=edge_index.t().contiguous(),
        is_auxiliary=torch.tensor([[False], [True]]),
    )
    target = ChemAtomicData(
        positions=torch.zeros(2, 3),
        atom_types=torch.tensor([0, 2]),
        charge_types=torch.tensor([1, 1]),
        edge_types=torch.zeros(2, dtype=torch.long),
        neighbor_list=edge_index.t().contiguous(),
        is_auxiliary=torch.zeros(2, 1, dtype=torch.bool),
    )
    monkeypatch.setattr(
        interpolation,
        "partial_optimal_transport_single",
        lambda source, clean, **_kwargs: (source, clean),
    )
    jitter = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    monkeypatch.setattr(
        interpolation.torch, "randn_like", lambda _value: jitter.clone()
    )

    mol_t, _mol_1, _insertions = interpolator.interpolate_single(
        sample, target, t_scalar=1.0
    )

    expected = jitter * interpolator.move_noise_scale
    expected = expected - expected.mean(dim=0)
    torch.testing.assert_close(mol_t.positions, expected)


def test_scaffold_jitter_is_physical_and_insertions_have_no_endpoint_floor(
    monkeypatch,
):
    prior, interpolator, _integrator = _noise_consistency_setup()
    source = ChemAtomicData(
        positions=torch.arange(15, dtype=torch.float32).reshape(5, 3),
        atom_types=torch.zeros(5, dtype=torch.long),
        charge_types=torch.ones(5, dtype=torch.long),
        edge_types=torch.zeros(20, dtype=torch.long),
        neighbor_list=_complete_edges(5).T,
        is_auxiliary=torch.tensor([[False], [False], [False], [True], [True]]),
    )
    target = source.clone()
    target.is_auxiliary = torch.zeros(5, 1, dtype=torch.bool)
    monkeypatch.setattr(
        interpolation,
        "scaffold_based_assignment_single",
        lambda source, target, **kwargs: (source.clone(), target.clone()),
    )
    jitter = torch.tensor(
        [
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 2.0],
            [0.0, 0.0, -2.0],
            [9.0, 9.0, 9.0],
            [-9.0, -9.0, -9.0],
        ]
    )
    monkeypatch.setattr(interpolation.torch, "randn_like", lambda _: jitter.clone())

    mol_t, mol_1, _ = interpolator.interpolate_single(
        source,
        target,
        1.0,
        scaffold_pairs=[(0, 0), (1, 1), (2, 2)],
        existing_noise="jitter",
    )

    physical_residual = (mol_t.x - mol_1.x) * prior.coordinate_std
    torch.testing.assert_close(physical_residual[:3], 0.1 * jitter[:3])
    torch.testing.assert_close(physical_residual[3:], torch.zeros(2, 3))
    torch.testing.assert_close(target.x, source.x)  # Clean endpoint is untouched.


def test_uniform_type_prior_is_representation_aware():
    prior, _interpolator, _integrator = _noise_consistency_setup()
    vocab = Vocab(
        atom_tokens=["C", "N", "O"],
        edge_tokens=["<NO_BOND>", "1"],
        charge_tokens=["-1", "0", "1"],
    )

    pointcloud = make_uniform_type_prior(prior, vocab, Representation.POINTCLOUD)
    torch.testing.assert_close(
        pointcloud.atom_type_distribution, torch.full((3,), 1.0 / 3.0)
    )
    torch.testing.assert_close(
        pointcloud.edge_type_distribution, torch.tensor([1.0, 0.0])
    )
    torch.testing.assert_close(
        pointcloud.charge_type_distribution, torch.tensor([0.0, 1.0, 0.0])
    )


def _branched_target():
    # A branch attached to scaffold atom 0 folds close to scaffold atom 1;
    # the other branch does the reverse. Geometry alone picks the wrong sites.
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.9, 0.0, 0.0],
            [3.05, 0.1, 0.0],
            [0.1, 0.0, 0.0],
        ]
    )
    edges = _complete_edges(6)
    bonds = torch.zeros(6, 6, dtype=torch.long)
    for a, b in [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (1, 5)]:
        bonds[a, b] = bonds[b, a] = 1
    return ChemAtomicData(
        positions=positions,
        atom_types=torch.zeros(6, dtype=torch.long),
        charge_types=torch.ones(6, dtype=torch.long),
        neighbor_list=edges.T,
        edge_types=bonds[edges[0], edges[1]],
    )


def test_substituent_candidates_follow_branches_and_multiple_attachments():
    target = _branched_target()
    # Keep branch atom 3 as an additional parent; use a nontrivial node map.
    mapping = torch.tensor([2, 0, 3, 1, -1, -1])
    candidates = scaffold_insertion_candidates(target, 3, torch.tensor([4, 5]), mapping)
    assert candidates.tolist() == [
        [False, True, True, False],
        [True, False, False, False],
    ]
    # A bridging branch may use either of its actual scaffold attachment atoms.
    is_new_bond = ((target.edge_index[0] == 3) & (target.edge_index[1] == 2)) | (
        (target.edge_index[0] == 2) & (target.edge_index[1] == 3)
    )
    target.edge_types[is_new_bond] = 1
    candidates = scaffold_insertion_candidates(target, 3, torch.tensor([4, 5]), mapping)
    assert candidates[0].tolist() == [False, True, True, True]


def test_unattached_components_keep_nearest_rule_and_sibling_branches_stay_separate():
    target = _branched_target()
    # Sever atom 5's attachment; it now has no site constraint.
    target.edge_types[(target.edge_index == 5).any(dim=0)] = 0
    mapping = torch.tensor([0, 1, 2, 3, -1, -1])
    candidates = scaffold_insertion_candidates(target, 3, torch.tensor([4, 5]), mapping)
    assert candidates[1].all()
    # Attach it at the SAME site as the other branch; it cannot use that branch.
    target.edge_types[((target.edge_index == torch.tensor([[0], [5]])).all(dim=0))] = 1
    target.edge_types[((target.edge_index == torch.tensor([[5], [0]])).all(dim=0))] = 1
    candidates = scaffold_insertion_candidates(target, 3, torch.tensor([4, 5]), mapping)
    assert candidates[1].tolist() == [True, False, False, False]


def test_site_assignment_changes_rate_unimodal_and_bond_parents_but_not_flow_state():
    prior, _, _ = _noise_consistency_setup()
    target = _branched_target()
    source = target.filter_nodes(torch.tensor([True, True, True, False, False, False]))
    pairs = [(i, i) for i in range(3)]
    outputs = []
    for mode in ("nearest", "substituent"):
        interpolator = Interpolator(
            prior,
            insertion_assignment=mode,
            jitter_scale=0,
            ins_noise_scale=0,
        )
        torch.manual_seed(42)
        outputs.append(
            interpolator.interpolate_single(
                source.clone(),
                target.clone(),
                0.0,
                scaffold_pairs=pairs,
                existing_noise="jitter",
            )
        )
    baseline, expected_target, old_ins = outputs[0]
    actual, actual_target, new_ins = outputs[1]
    for field in ("x", "a", "c", "e"):
        torch.testing.assert_close(getattr(actual, field), getattr(baseline, field))
        torch.testing.assert_close(
            getattr(actual_target, field), getattr(expected_target, field)
        )
    assert old_ins.spawn_node_idx.tolist() == [1, 1, 0]
    assert new_ins.spawn_node_idx.tolist() == [0, 0, 1]
    assert actual.n_ins.tolist() == [2, 1, 0]
    assert baseline.n_ins.tolist() == [1, 2, 0]
    assert actual.n_ins.sum() == baseline.n_ins.sum() == 3
    torch.testing.assert_close(actual.lambda_ins.sum(), baseline.lambda_ins.sum())
    assert new_ins.nearest_node_idx.tolist() == [0, 1]
    assert new_ins.nearest_ins_idx.tolist() == [0, 2]
    bonds = new_ins.bonds_to_existing
    torch.testing.assert_close(
        bonds.spawn_idx, new_ins.spawn_node_idx[bonds.inserted_idx]
    )
    bonds = new_ins.bonds_between_inserted
    torch.testing.assert_close(
        bonds.spawn_src_idx, new_ins.spawn_node_idx[bonds.src_idx]
    )
    torch.testing.assert_close(
        bonds.spawn_dst_idx, new_ins.spawn_node_idx[bonds.dst_idx]
    )


def test_substituent_mode_keeps_unconditional_interpolation_unchanged():
    prior, _, _ = _noise_consistency_setup()
    outputs = []
    for mode in ("nearest", "substituent"):
        interpolator = Interpolator(prior, insertion_assignment=mode)
        torch.manual_seed(42)
        outputs.append(
            interpolator.interpolate_single(_branched_target(), _branched_target(), 0.4)
        )
    for a, b in zip(outputs[0][:2], outputs[1][:2], strict=True):
        for field in ("x", "a", "c", "e", "lambda_ins", "n_ins"):
            if hasattr(a, field):
                torch.testing.assert_close(getattr(a, field), getattr(b, field))
    torch.testing.assert_close(
        outputs[0][2].spawn_node_idx, outputs[1][2].spawn_node_idx
    )


def test_insertion_assignment_rejects_unknown_policy():
    prior, _, _ = _noise_consistency_setup()
    with pytest.raises(ValueError, match="insertion_assignment"):
        Interpolator(prior, insertion_assignment="substitutent")
