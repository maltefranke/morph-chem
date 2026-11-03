import numpy as np
import torch

from morph.dataset.core.atomic_data import ChemAtomicData
from morph.flow_matching.assignment import (
    distance_and_class_based_assignment,
    partial_optimal_transport_single,
)


def _is_real_match(distance: float, target_type: int = 0) -> bool:
    rows, cols = distance_and_class_based_assignment(
        x0=np.array([[0.0, 0.0, 0.0]]),
        x1=np.array([[distance, 0.0, 0.0]]),
        a0=np.array([0]),
        a1=np.array([target_type]),
    )
    return bool(((rows == 0) & (cols == 0)).any())


def test_partial_ot_uses_delete_insert_beyond_3p5_angstrom():
    assert _is_real_match(3.49)
    assert not _is_real_match(3.51)


def test_partial_ot_cutoff_also_applies_to_substitutions():
    assert _is_real_match(3.49, target_type=1)
    assert not _is_real_match(3.51, target_type=1)


def test_partial_ot_preserves_inputs_and_graph_fields():
    source = ChemAtomicData(
        positions=torch.tensor([[0.0, 0, 0], [1.0, 0, 0]]),
        atom_types=torch.tensor([0, 1]),
        charge_types=torch.tensor([1, 1]),
        neighbor_list=torch.tensor([[0, 1], [1, 0]]),
        edge_types=torch.tensor([2, 2]),
    )
    target = ChemAtomicData(
        positions=torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]]),
        atom_types=torch.tensor([0, 1, 2]),
        charge_types=torch.tensor([1, 1, 1]),
        neighbor_list=torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0]]),
        edge_types=torch.tensor([2, 2, 1, 1]),
    )
    source_before, target_before = source.clone(), target.clone()

    aligned_source, aligned_target = partial_optimal_transport_single(
        source,
        target,
        c_sub=0,
        c_ins=5,
        c_del=5,
        optimal_transport="none",
        pre_align_iters=0,
    )

    assert torch.equal(source.positions, source_before.positions)
    assert torch.equal(target.positions, target_before.positions)
    assert aligned_source.num_nodes == aligned_target.num_nodes == 3
    assert aligned_source.is_auxiliary.squeeze(-1).tolist() == [False, False, True]
    assert not aligned_target.is_auxiliary.any()
    assert torch.equal(aligned_source.neighbor_list, source.neighbor_list)
    assert torch.equal(aligned_source.edge_types, source.edge_types)
