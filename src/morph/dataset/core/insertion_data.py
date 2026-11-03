"""Atom insertion / deletion editing ops for molecule batches.

Split out of :mod:`morph.dataset.core.atomic_data`: the point-cloud
insertion primitives -- bundling freshly inserted atoms
(:class:`InsertedAtoms`), pairing them within a graph, sampling new edge
types, and joining them into an existing :class:`ChemBatch`
with either sampled (:func:`join_with_atoms`) or model-predicted
(:func:`join_with_predicted_edges`) bonds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from morph.dataset.core.atomic_data import ChemBatch, batch_from_flat
from morph.utils.edge_alignment import EDGE_ALIGNER

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class InsertedAtoms:
    """Flat features of freshly inserted atoms (the point-cloud analogue).

    Bundles the position / atom-type / charge-type tensors that were otherwise
    threaded around as loose ``ins_x`` / ``ins_a`` / ``ins_c`` triplets, with an
    optional per-atom ``batch`` assignment. Indexing gathers every field at once,
    so ``atoms[idx]`` replaces three (or more) parallel gathers and is how callers
    select an edge-aligned subset to feed the insertion-edge head.
    """

    x: torch.Tensor  # [N, 3] positions
    a: torch.Tensor  # [N] atom-type tokens
    c: torch.Tensor  # [N] charge tokens
    batch: torch.Tensor | None = None  # [N] graph id per inserted atom

    def __getitem__(self, idx) -> InsertedAtoms:
        """Select matching position, atom, charge, and batch entries."""
        return InsertedAtoms(
            self.x[idx],
            self.a[idx],
            self.c[idx],
            None if self.batch is None else self.batch[idx],
        )


def insertion_pair_indices(
    ins_batch: torch.Tensor,
    spawn_orig_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build inserted-atom pairs and both endpoints' spawn-node indices.

    ``ins_batch`` is the per-inserted-atom graph id and ``spawn_orig_indices`` the
    per-inserted-atom original spawn-node index (both length ``N_ins``, aligned).
    ``ins_batch`` must be graph-contiguous, as it is in a :class:`ChemBatch`.
    Pairs are emitted per graph in ascending graph order. Returns four empty
    long tensors when no graph holds at least two inserted atoms.
    """
    if ins_batch.numel() == 0:
        empty = ins_batch.new_zeros(0, dtype=torch.long)
        return empty, empty.clone(), empty.clone(), empty.clone()

    counts = torch.bincount(ins_batch)
    max_count = int(counts.max())
    if max_count < 2:
        empty = ins_batch.new_zeros(0, dtype=torch.long)
        return empty, empty.clone(), empty.clone(), empty.clone()

    # Build one local upper triangle, then retain the valid prefix for each
    # graph. This avoids one GPU launch and Python iteration per graph while
    # using O(num_graphs * max_insertions_per_graph**2) temporary storage rather
    # than an O(total_insertions**2) cross-graph matrix.
    local_src, local_dst = torch.triu_indices(
        max_count, max_count, offset=1, device=ins_batch.device
    )
    graph_idx, pair_idx = torch.where(local_dst.unsqueeze(0) < counts.unsqueeze(1))
    offsets = counts.cumsum(0) - counts
    src = offsets[graph_idx] + local_src[pair_idx]
    dst = offsets[graph_idx] + local_dst[pair_idx]
    return src, dst, spawn_orig_indices[src], spawn_orig_indices[dst]


def sample_edge_types(edge_dist: torch.Tensor, num_edges: int) -> torch.Tensor:
    """Draw ``num_edges`` edge-type token indices iid from a categorical prior.

    Helper for sampling new edge types *outside* the join, then passing the
    resulting tensor into :func:`join_with_atoms`.
    """
    return torch.multinomial(edge_dist, num_edges, replacement=True)


def _new_atom_edge_count(num_mol: int, num_atom: int) -> int:
    """Number of new upper-triangular edges when ``num_atom`` atoms join a mol."""
    return num_mol * num_atom + num_atom * (num_atom - 1) // 2


def count_new_atom_edges(mol_batch: ChemBatch, atoms_batch: ChemBatch) -> int:
    """Total new edges :func:`join_with_atoms` will create across the batch.

    The caller uses this to know how many edge types to sample (e.g. via
    :func:`sample_edge_types`) before calling :func:`join_with_atoms`.
    """
    mol_n = mol_batch.num_nodes_per_graph.to(torch.long)
    atom_n = atoms_batch.num_nodes_per_graph.to(torch.long)
    return int((mol_n * atom_n + atom_n * (atom_n - 1) // 2).sum())


def join_with_atoms(
    mol_batch: ChemBatch,
    atoms_batch: ChemBatch,
    new_edge_types: torch.Tensor,
) -> ChemBatch:
    """Join a batch of molecules with a batch of new atoms (per graph).

    Each new atom set is
    fully connected to its molecule. Edge sampling is *not* done here: the caller
    samples the new edge types beforehand (e.g. ``new_edge_types =
    sample_edge_types(edge_dist, count_new_atom_edges(mol_batch, atoms_batch))``)
    and passes the flat tensor in. It holds one type per new upper-triangular
    edge, concatenated graph by graph in (mol->atom, then atom->atom) order.

    The graph is assembled in the concatenated ``[existing, inserted]`` node space
    and re-batched via :func:`batch_from_flat` (which restores graph-contiguous
    ordering and per-graph offsets) -- the same flat-tensor path as
    :func:`join_with_predicted_edges`, so no per-graph ``ChemAtomicData`` objects
    are constructed. The small per-graph loop only assembles edge *indices* (cheap
    tensor ops) in the order ``new_edge_types`` is laid out.
    """
    device = mol_batch["positions"].device
    num_existing = mol_batch.num_nodes
    num_graphs = mol_batch.num_graphs
    mol_batch_idx = mol_batch.batch_idx.long()
    atoms_batch_idx = atoms_batch.batch_idx.long()

    positions = torch.cat([mol_batch["positions"], atoms_batch["positions"]])
    atom_types = torch.cat([mol_batch["atom_types"], atoms_batch["atom_types"]])
    charge_types = None
    if "charge_types" in mol_batch and "charge_types" in atoms_batch:
        charge_types = torch.cat(
            [mol_batch["charge_types"], atoms_batch["charge_types"]]
        )
    new_batch = torch.cat([mol_batch_idx, atoms_batch_idx])

    if mol_batch.edge_index.numel() > 0:
        existing_edge_index = mol_batch.edge_index
        existing_e = mol_batch["edge_types"]
    else:
        existing_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        existing_e = torch.empty((0,), dtype=torch.long, device=device)

    # New upper-tri edges per graph, in the combined node space ([mol nodes,
    # then inserted atoms]) in count order (mol->atom block,
    # then atom->atom) so ``new_edge_types`` is consumed graph by graph.
    triu_list: list[torch.Tensor] = []
    e_list: list[torch.Tensor] = []
    start = 0
    for g in range(num_graphs):
        mol_g = torch.where(mol_batch_idx == g)[0]
        atom_g = torch.where(atoms_batch_idx == g)[0] + num_existing
        num_mol = int(mol_g.numel())
        num_atom = int(atom_g.numel())
        count = _new_atom_edge_count(num_mol, num_atom)
        if count == 0:
            continue
        blocks: list[torch.Tensor] = []
        if num_mol > 0 and num_atom > 0:
            blocks.append(torch.cartesian_prod(mol_g, atom_g).t())  # mol -> atom
        if num_atom > 1:
            atom_atom = torch.cartesian_prod(atom_g, atom_g).t()
            atom_atom = atom_atom[:, atom_atom[0] < atom_atom[1]]
            if atom_atom.shape[1] > 0:
                blocks.append(atom_atom)
        triu_list.append(torch.cat(blocks, dim=1))
        e_list.append(new_edge_types[start : start + count].to(device, torch.long))
        start += count

    assert start == int(new_edge_types.shape[0]), (
        f"expected {start} edge types, got {new_edge_types.shape[0]}; "
        "sample count_new_atom_edges(mol_batch, atoms_batch) of them"
    )

    if triu_list:
        new_triu = torch.cat(triu_list, dim=1)
        new_e_triu = torch.cat(e_list, dim=0)
        new_edge_index, (new_e,) = EDGE_ALIGNER.symmetrize_edges(
            new_triu, [new_e_triu], num_nodes=positions.shape[0]
        )
        combined_edge_index = torch.cat([existing_edge_index, new_edge_index], dim=1)
        combined_e = torch.cat([existing_e, new_e], dim=0)
    else:
        combined_edge_index, combined_e = existing_edge_index, existing_e

    # keep_mask extends with zeros for the new atoms; system properties
    # graph-level properties are carried unchanged.
    node_extras: dict[str, torch.Tensor] = {}
    if "keep_mask" in mol_batch:
        keep_mask = mol_batch["keep_mask"]
        node_extras["keep_mask"] = torch.cat(
            [
                keep_mask,
                torch.zeros(
                    atoms_batch.num_nodes, dtype=keep_mask.dtype, device=device
                ),
            ]
        )
    system_extras = {key: mol_batch[key] for key in mol_batch.keys["system"]}

    return batch_from_flat(
        positions,
        atom_types,
        charge_types,
        combined_e,
        combined_edge_index,
        new_batch,
        num_graphs=num_graphs,
        _node_counts=mol_batch.num_nodes_per_graph + atoms_batch.num_nodes_per_graph,
        node_extras=node_extras,
        system_extras=system_extras,
    )


def join_with_predicted_edges(
    mol_batch: ChemBatch,
    new_atoms_batch: ChemBatch,
    e_ins: torch.Tensor,
    spawn_node_idx: torch.Tensor,
    target_node_idx: torch.Tensor,
    *,
    e_ins_to_ins: torch.Tensor | None = None,
    ins_to_ins_src_idx: torch.Tensor | None = None,
    ins_to_ins_dst_idx: torch.Tensor | None = None,
    sample_fallback_edge_types: Callable[[int], torch.Tensor] | None = None,
) -> ChemBatch:
    """Join molecules with newly inserted atoms using model-predicted edges.

    ``e_ins`` are
    the predicted (new-atom -> existing) bond types with ``spawn_node_idx``
    (local new-atom ids) and ``target_node_idx`` (existing-node ids in
    ``mol_batch``). New-atom -> new-atom bonds come from ``e_ins_to_ins`` /
    ``ins_to_ins_src_idx`` / ``ins_to_ins_dst_idx`` when given; otherwise every
    intra-graph pair is created and typed by ``sample_fallback_edge_types``
    (outsourced, like :func:`join_with_atoms`). The graph is assembled in the
    concatenated [existing, inserted] space and re-batched via
    :func:`batch_from_flat`, which restores graph-contiguous ordering.
    """
    device = mol_batch["positions"].device
    num_existing = mol_batch.num_nodes
    new_atoms_batch_idx = new_atoms_batch.batch_idx

    positions = torch.cat([mol_batch["positions"], new_atoms_batch["positions"]])
    atom_types = torch.cat([mol_batch["atom_types"], new_atoms_batch["atom_types"]])
    charge_types = None
    if "charge_types" in mol_batch and "charge_types" in new_atoms_batch:
        charge_types = torch.cat(
            [mol_batch["charge_types"], new_atoms_batch["charge_types"]]
        )
    new_batch = torch.cat([mol_batch.batch_idx, new_atoms_batch_idx])

    if mol_batch.edge_index.numel() > 0:
        existing_edge_index = mol_batch.edge_index
        existing_e = mol_batch["edge_types"]
    else:
        existing_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        existing_e = torch.empty((0,), dtype=torch.long, device=device)

    triu_list: list[torch.Tensor] = []
    e_list: list[torch.Tensor] = []

    # inserted -> existing (upper triangle: target precedes the new atom)
    if e_ins is not None and e_ins.numel() > 0:
        global_new_atom_idx = spawn_node_idx + num_existing
        triu_list.append(torch.stack([target_node_idx, global_new_atom_idx], dim=0))
        e_list.append(e_ins.to(device=device, dtype=torch.long))

    # inserted -> inserted
    use_predicted_ii = (
        e_ins_to_ins is not None
        and e_ins_to_ins.numel() > 0
        and ins_to_ins_src_idx is not None
        and ins_to_ins_dst_idx is not None
    )
    if use_predicted_ii:
        triu_list.append(
            torch.stack(
                [ins_to_ins_src_idx + num_existing, ins_to_ins_dst_idx + num_existing],
                dim=0,
            )
        )
        e_list.append(e_ins_to_ins.to(device=device, dtype=torch.long))
    else:
        local_idx = torch.arange(new_atoms_batch.num_nodes, device=device)
        src, dst, _, _ = insertion_pair_indices(new_atoms_batch_idx, local_idx)
        if src.numel() > 0:
            if sample_fallback_edge_types is None:
                raise ValueError(
                    "sample_fallback_edge_types is required when predicted "
                    "inserted->inserted edges are not provided"
                )
            triu_list.append(
                torch.stack([src + num_existing, dst + num_existing], dim=0)
            )
            e_list.append(
                sample_fallback_edge_types(src.numel()).to(
                    device=device, dtype=torch.long
                )
            )

    if triu_list:
        new_triu = torch.cat(triu_list, dim=1)
        new_e_triu = torch.cat(e_list, dim=0)
        new_edge_index, (new_e,) = EDGE_ALIGNER.symmetrize_edges(
            new_triu, [new_e_triu], num_nodes=positions.shape[0]
        )
        combined_edge_index = torch.cat([existing_edge_index, new_edge_index], dim=1)
        combined_e = torch.cat([existing_e, new_e], dim=0)
    else:
        combined_edge_index, combined_e = existing_edge_index, existing_e

    # Conditioning: keep_mask extends with zeros for new atoms.
    node_extras: dict[str, torch.Tensor] = {}
    if "keep_mask" in mol_batch:
        keep_mask = mol_batch["keep_mask"]
        node_extras["keep_mask"] = torch.cat(
            [
                keep_mask,
                torch.zeros(
                    new_atoms_batch.num_nodes,
                    dtype=keep_mask.dtype,
                    device=device,
                ),
            ]
        )
    system_extras = {key: mol_batch[key] for key in mol_batch.keys["system"]}
    return batch_from_flat(
        positions,
        atom_types,
        charge_types,
        combined_e,
        combined_edge_index,
        new_batch,
        num_graphs=mol_batch.num_graphs,
        _node_counts=mol_batch.num_nodes_per_graph
        + new_atoms_batch.num_nodes_per_graph,
        node_extras=node_extras,
        system_extras=system_extras,
    )
