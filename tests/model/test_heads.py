"""Heads regression tests."""

import torch
from omegaconf import OmegaConf

from morph.dataset.core.insertion_data import InsertedAtoms, insertion_pair_indices
from morph.model.layers.heads import (
    GMMInsertionHead,
    InsertionEdgeHead,
    MultiHeadModule,
)

from ._helpers import _rbf_args


def test_edge_heads_expand_only_final_undirected_predictions():
    config = OmegaConf.create(
        {
            "edge_heads": {
                "edge_type_head": {
                    "input_dim": 10,
                    "output_dim": 5,
                    "adapter_dim": 8,
                },
                "do_sub_e_head": {
                    "input_dim": 10,
                    "output_dim": 1,
                    "adapter_dim": 8,
                },
            }
        }
    )
    heads = MultiHeadModule(config).eval()
    full_to_triu = torch.tensor([0, 1, 0, 2, 1, 2])
    edge_attr_triu = torch.randn(3, 10)

    full_outputs = heads(torch.empty(0, 10), None, edge_attr_triu[full_to_triu])
    triu_outputs = heads(
        torch.empty(0, 10),
        None,
        edge_attr_triu,
        full_to_triu=full_to_triu,
    )

    for name in full_outputs:
        torch.testing.assert_close(triu_outputs[name], full_outputs[name])


def test_inserted_pair_head_is_exactly_symmetric():
    head = InsertionEdgeHead(
        hidden_dim=12,
        readout_dim=10,
        n_edge_types=5,
        n_atom_types=6,
        n_charge_types=3,
        rbf_embedding_args=_rbf_args(),
    ).eval()
    torch.nn.init.normal_(head.inserted_pair_decoder[-1].weight, std=0.2)
    h = torch.randn(4, 12)
    x = torch.randn(4, 3)
    src = InsertedAtoms(torch.randn(1, 3), torch.tensor([2]), torch.tensor([0]))
    dst = InsertedAtoms(torch.randn(1, 3), torch.tensor([4]), torch.tensor([2]))
    forward = head.forward_inserted_to_inserted(
        h, x, torch.tensor([0]), torch.tensor([2]), src, dst
    )
    reverse = head.forward_inserted_to_inserted(
        h, x, torch.tensor([2]), torch.tensor([0]), dst, src
    )
    torch.testing.assert_close(forward, reverse, rtol=0.0, atol=0.0)


def test_inserted_pairs_carry_both_spawn_contexts():
    src, dst, spawn_src, spawn_dst = insertion_pair_indices(
        ins_batch=torch.tensor([0, 0, 0, 1]),
        spawn_orig_indices=torch.tensor([4, 7, 9, 12]),
    )
    assert src.tolist() == [0, 0, 1]
    assert dst.tolist() == [1, 2, 2]
    assert spawn_src.tolist() == [4, 4, 7]
    assert spawn_dst.tolist() == [7, 9, 9]


def test_inserted_pairs_handle_empty_graph_ids_without_python_loops():
    src, dst, spawn_src, spawn_dst = insertion_pair_indices(
        ins_batch=torch.tensor([0, 0, 2, 2, 2, 4]),
        spawn_orig_indices=torch.tensor([3, 5, 8, 9, 12, 17]),
    )
    assert src.tolist() == [0, 2, 2, 3]
    assert dst.tolist() == [1, 3, 4, 4]
    assert spawn_src.tolist() == [3, 8, 8, 9]
    assert spawn_dst.tolist() == [5, 9, 12, 12]


def test_insertion_edge_inference_reuses_spawn_features_without_changing_logits():
    head = InsertionEdgeHead(
        hidden_dim=12,
        readout_dim=10,
        n_edge_types=5,
        n_atom_types=6,
        n_charge_types=3,
        rbf_embedding_args=_rbf_args(),
    ).eval()
    h = torch.randn(6, 12)
    x = torch.randn(6, 3)
    atom_types = torch.tensor([0, 1, 2, 3, 4, 5])
    batch = torch.tensor([0, 0, 2, 2, 2, 4])
    node_counts = torch.tensor([2, 0, 3, 0, 1, 0])
    insertion_mask = torch.tensor([False, True, False, True, True, False])
    insertion_indices = torch.where(insertion_mask)[0]
    ins = InsertedAtoms(
        x=torch.randn(3, 3),
        a=torch.tensor([2, 4, 1]),
        c=torch.tensor([0, 2, 1]),
    )
    expected_spawn, expected_existing, local = head._insertion_edge_pairs(
        insertion_indices, batch, node_counts
    )
    assert expected_spawn.tolist() == [1, 1, 3, 3, 3, 4, 4, 4]
    assert expected_existing.tolist() == [0, 1, 2, 3, 4, 2, 3, 4]
    expected_logits = head(
        h,
        x,
        atom_types,
        expected_spawn,
        expected_existing,
        ins[local],
    )

    spawn, existing, logits = head.predict_inserted_to_existing(
        h, x, atom_types, batch, insertion_indices, node_counts, ins
    )

    torch.testing.assert_close(spawn, expected_spawn)
    torch.testing.assert_close(existing, expected_existing)
    torch.testing.assert_close(logits, expected_logits)


def test_empty_insertion_edge_logits_follow_model_dtype():
    head = InsertionEdgeHead(
        hidden_dim=12,
        readout_dim=10,
        n_edge_types=5,
        n_atom_types=6,
        n_charge_types=3,
        rbf_embedding_args=_rbf_args(),
    )
    logits = head(
        torch.empty(0, 12, dtype=torch.float64),
        torch.empty(0, 3, dtype=torch.float64),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        InsertedAtoms(
            torch.empty(0, 3),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        ),
    )
    assert logits.dtype == torch.float64


def test_gmm_scales_are_bounded_and_initialized():
    head = GMMInsertionHead(
        hidden_dim=16,
        readout_dim=8,
        num_components=5,
        N_a=6,
        N_c=3,
        sigma_min=0.05,
        sigma_max=2.0,
        sigma_init=0.5,
    )
    output = head(
        torch.randn(7, 16),
        torch.randn(7, 3),
        (torch.empty(0, dtype=torch.long),) * 2,
    )
    torch.testing.assert_close(output["sigma"], torch.full((7, 5), 0.5))
    assert (output["sigma"] > 0.05).all()
    assert (output["sigma"] < 2.0).all()
