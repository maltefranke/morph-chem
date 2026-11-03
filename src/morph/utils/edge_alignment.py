"""Upper-triangle and full-symmetric edge alignment.

Prefer the module-level ``EDGE_ALIGNER`` singleton in hot loops.
"""

import torch
from torch_geometric.utils import sort_edge_index


class EdgeAligner:
    """
    Helper class to handle edge indices and attributes.
    Can be used to align edge indices and attributes between two groups of edges.

    The class is stateless, so prefer the module-level ``EDGE_ALIGNER`` singleton
    over instantiating new objects in hot loops (e.g. per-batch integration or
    per-molecule join ops) to avoid unnecessary Python allocations.
    """

    def __init__(self):
        pass

    def get_canonical_upper_triangle_with_index(
        self, edge_index, edge_attr, include_diag=False
    ):
        """Returns (edge_index, edge_attr) for the upper triangle."""
        # 1. Sort first to ensure canonical order
        edge_index, edge_attr = sort_edge_index(edge_index, edge_attr)
        row, col = edge_index

        # 2. Filter
        mask = row <= col if include_diag else row < col

        # Return BOTH the filtered index and attributes
        return edge_index[:, mask], edge_attr[mask]

    def _process_edge_group(self, edge_index, attrs):
        """Sort one edge group once and return its upper-triangle attributes."""
        if not attrs:
            return None, []
        canonical_index, sorted_attrs = sort_edge_index(edge_index, attrs)
        row, col = canonical_index
        mask = row < col
        return canonical_index[:, mask], [attr[mask] for attr in sorted_attrs]

    def align_edges(self, source_group, target_group=None):
        """
        Flexible handler for aligning edge attributes.

        Args:
            source_group: Tuple of (edge_index, [attr1, attr2, ...])
            target_group: (Optional) Tuple of (edge_index, [attr1, attr2, ...])

        Returns:
            Flat tuple of all processed attributes from source then target.
        """
        src_index, src_attrs = source_group
        tgt_index, tgt_attrs = (None, None) if target_group is None else target_group

        # 1. Process Source Group
        src_triu_index, src_triu_attrs = self._process_edge_group(src_index, src_attrs)

        # 2. Process Target Group (if exists)
        tgt_triu_attrs = []
        if target_group:
            tgt_triu_index, tgt_triu_attrs = self._process_edge_group(
                tgt_index, tgt_attrs
            )

            # Return all attributes flattened
            out = {
                "edge_index": (src_triu_index, tgt_triu_index),
                "edge_attr": tuple(src_triu_attrs + tgt_triu_attrs),
            }

        else:
            out = {
                "edge_index": src_triu_index,
                "edge_attr": src_triu_attrs,
            }
        return out

    def symmetrize_edges(
        self,
        edge_index_triu: torch.Tensor,
        attrs: list[torch.Tensor],
        num_nodes: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """
        Reverse: Upper Triangle -> Full Symmetric
        Handles attributes efficiently by calculating the sort permutation once.

        ``num_nodes`` is the radix for the lexical sort key and only needs to
        exceed the largest node index. Callers that already know the node count
        (e.g. ``positions.shape[0]``) should pass it to skip a
        ``full_edge_index.max().item()`` device sync in the hot integration loop.
        """
        row, col = edge_index_triu

        # 1. Identify off-diagonal edges to mirror
        mask = row < col

        # 2. Create the mirrored index
        mirror_index = torch.stack([col[mask], row[mask]], dim=0)
        full_edge_index = torch.cat([edge_index_triu, mirror_index], dim=1)

        # 3. Compute Sort Permutation ONCE
        # Repeated sorts risk putting equal-key attributes out of sync.
        # if the sort isn't stable. We compute indices explicitly.
        # Simple Lexical Sort: row * max_cols + col
        # (Or use PyTorch's stable sort on rows then cols)
        if num_nodes is None:
            num_nodes = int(full_edge_index.max().item()) + 1
        perm = (full_edge_index[0] * num_nodes + full_edge_index[1]).argsort()

        # Apply sort to index
        full_edge_index_sorted = full_edge_index[:, perm]

        # 4. Process all attributes
        full_attrs_sorted = []
        for attr in attrs:
            # Mirror the attributes corresponding to the mask
            mirror_attr = attr[mask]
            full_attr = torch.cat([attr, mirror_attr], dim=0)

            # Apply the SAME permutation
            full_attrs_sorted.append(full_attr[perm])

        return full_edge_index_sorted, tuple(full_attrs_sorted)


# Stateless shared instance — reuse in hot paths instead of constructing new
# ``EdgeAligner()`` objects per batch / per molecule.


EDGE_ALIGNER = EdgeAligner()
