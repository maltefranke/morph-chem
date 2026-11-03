"""Tensor and edge-index helpers: Kabsch alignment and segment aggregation."""

from functools import cache

import torch
from torch_geometric.utils import to_dense_adj


def edge_types_to_symmetric(edge_index, edge_types, num_atoms):
    """
    Convert edge types to a symmetric adjacency matrix.
    edge_index: Shape (2, E) - edge indices
    edge_types: Shape (E) - edge types
    num_atoms: int - number of atoms in the graph
    Returns:
        adj_matrix: Shape (N, N) - symmetric adjacency matrix
    """
    adj_matrix = to_dense_adj(edge_index, edge_attr=edge_types, max_num_nodes=num_atoms)
    adj_matrix = adj_matrix.squeeze()
    return adj_matrix


def rigid_alignment(x_0, x_1, pre_centered=False):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    Alignment of two point clouds using the Kabsch algorithm.
    """
    d = x_0.shape[1]
    assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape"

    # remove COM from data and record initial COM
    if pre_centered:
        x_0_mean = x_0.new_zeros((1, d))
        x_1_mean = x_1.new_zeros((1, d))
        x_0_c = x_0
        x_1_c = x_1
    else:
        x_0_mean = x_0.mean(dim=0, keepdim=True)
        x_1_mean = x_1.mean(dim=0, keepdim=True)
        x_0_c = x_0 - x_0_mean
        x_1_c = x_1 - x_1_mean

    # Covariance matrix
    H = x_0_c.T.mm(x_1_c)
    U, _, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = x_0.new_zeros((1, d)) if pre_centered else x_1_mean - R.mm(x_0_mean.T).T

    return R, t


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Sum ``data`` rows into ``num_segments`` buckets keyed by ``segment_ids``.

    Equivalent to a scatter-add over dim 0. Uses ``index_add_``, which takes a
    1-D index and broadcasts over the feature columns — avoiding the
    ``unsqueeze().expand()`` index materialization a ``scatter_add_`` needs.

    Args:
        data: Shape (N, K) - values to aggregate.
        segment_ids: Shape (N,) - segment index in [0, num_segments) per row.
        num_segments: int - number of output segments.

    Returns:
        Shape (num_segments, K); empty segments are 0.
    """
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    """Mean of ``data`` rows per segment; empty segments yield 0.

    Counts via ``bincount`` (one 1-D op of shape ``(num_segments,)``) instead of
    scattering a full ``ones_like(data)``.

    Args:
        data: Shape (N, K) - values to aggregate.
        segment_ids: Shape (N,) - segment index in [0, num_segments) per row.
        num_segments: int - number of output segments.

    Returns:
        Shape (num_segments, K); empty segments are 0.
    """
    return unsorted_segment_mean_and_count(data, segment_ids, num_segments)[0]


def unsorted_segment_mean_and_count(data, segment_ids, num_segments):
    """Return per-segment means and the un-clamped number of contributing rows."""
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    count = torch.bincount(segment_ids, minlength=num_segments)
    return result / count.clamp(min=1).unsqueeze(1), count


def build_fully_connected_edge_index(n, device=None):
    """Directed, no-self-loop edge_index of shape ``(2, N*(N-1))``.

    Canonical home for the fully-connected edge builder (consolidated from a
    second, divergent copy that used to live in ``dataset.core.representation``).
    Guards the ``n <= 1`` case explicitly; ``device=None`` yields a CPU tensor.
    """
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    rng = torch.arange(n, device=device)
    src, dst = torch.meshgrid(rng, rng, indexing="ij")
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


@cache
def _complete_graph_layout(num_nodes: int, device: torch.device):
    edge_index = build_fully_connected_edge_index(num_nodes, device=device)
    triu_rows, triu_cols = torch.triu_indices(
        num_nodes, num_nodes, offset=1, device=device
    )
    row, col = edge_index
    i, j = torch.minimum(row, col), torch.maximum(row, col)
    full_to_triu = (i * (2 * num_nodes - i - 1)) // 2 + (j - i - 1)
    return edge_index, triu_rows, triu_cols, full_to_triu


def complete_graph_layout(num_nodes: int, device=None):
    """Return cached complete edges, upper-triangle rows/cols, and their mapping."""
    device = torch.device("cpu") if device is None else torch.device(device)
    return _complete_graph_layout(num_nodes, device)


def batched_full_to_triu(
    edge_index: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    batch: torch.Tensor,
    node_counts: torch.Tensor,
) -> torch.Tensor:
    """Map complete directed edges to their batched upper triangles.

    Graph nodes must be contiguous and each graph's edges must be complete and
    directed. Edge storage order is arbitrary. The returned indices let
    ``edge_attr_triu[full_to_triu]`` restore full symmetric attributes without
    rebuilding or sorting ``edge_index``.
    """
    rows, cols = edge_index
    graph = batch[rows]
    node_starts = node_counts.cumsum(0) - node_counts
    local_rows = rows - node_starts[graph]
    local_cols = cols - node_starts[graph]
    lower = torch.minimum(local_rows, local_cols)
    upper = torch.maximum(local_rows, local_cols)
    widths = node_counts[graph]

    triu_counts = node_counts * (node_counts - 1) // 2
    triu_starts = triu_counts.cumsum(0) - triu_counts
    within_graph = lower * (2 * widths - lower - 1) // 2 + upper - lower - 1
    return triu_starts[graph] + within_graph


def batched_triu_edge_layout(
    edge_index: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    batch: torch.Tensor,
    node_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one representative edge per pair and the full-to-triu mapping.

    This is shape-stable under ``torch.compile``: it avoids boolean indexing,
    whose output length depends on tensor data, by selecting the first of each
    pair of symmetric directed edges with an integer scatter reduction.
    """
    full_to_triu = batched_full_to_triu(edge_index, batch, node_counts)
    num_edges = edge_index[0].numel()
    triu_edge_positions = full_to_triu.new_full((num_edges // 2,), num_edges)
    triu_edge_positions.scatter_reduce_(
        0,
        full_to_triu,
        torch.arange(num_edges, device=edge_index[0].device),
        reduce="amin",
        include_self=True,
    )
    return triu_edge_positions, full_to_triu
