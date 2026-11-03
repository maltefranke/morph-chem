"""Flow-matching dataset wrapper for scaffold decoration."""

from __future__ import annotations

import random

import numpy as np
import torch

from morph.dataset.conditional.scaffold.lmdb import load_scaffold_cache
from morph.dataset.core.flow_matching_wrapper import FlowMatchingDatasetWrapper
from morph.utils.graph_math import build_fully_connected_edge_index


def rebuild_atomic_with_edges(data, edge_index, edge_types):
    """Type-preserving copy of ``data`` with fully replaced connectivity."""
    out = data.clone()
    out.edge_index = edge_index.long()
    out.edge_types = edge_types.long()
    return out


class FlowMatchingDatasetWrapperScaffold(FlowMatchingDatasetWrapper):
    """Build retained-subgraph → full-molecule flow-matching examples.

    ``keep_mask`` marks the scaffold atoms in the interpolated state. Scaffold
    atoms are placed first by the scaffold-aware interpolator.
    Training draws a random cut per visit. Evaluation can enumerate all
    distinct molecule/scaffold pairs with ``enumerate_slicings=True``.
    """

    def __init__(
        self,
        base_dataset,
        distributions,
        interpolator,
        vocab,
        representation=None,
        n_atoms_strategy="uniform",
        uniform_n_atoms_padding: int = 5,
        time_dist=None,
        stage="train",
        rotate: bool = False,
        n_augmentations: int = 1,
        scaffold_groups_dir: str | None = None,
        scaffold_kind: str = "rgroup",
        enumerate_slicings: bool = False,
    ):
        base_kwargs = dict(
            n_atoms_strategy=n_atoms_strategy,
            uniform_n_atoms_padding=uniform_n_atoms_padding,
            time_dist=time_dist,
            stage=stage,
            rotate=rotate,
            n_augmentations=n_augmentations,
        )
        if representation is not None:
            base_kwargs["representation"] = representation
        super().__init__(
            base_dataset,
            distributions,
            interpolator,
            vocab,
            **base_kwargs,
        )
        cache = load_scaffold_cache(
            base_dataset,
            scaffold_groups_dir,
            stage,
            scaffold_kind=scaffold_kind,
        )
        self._scaffold_atom_indices = cache.atom_indices
        self._filtered_indices = cache.filtered_indices()
        self._slicing_offsets = None
        if enumerate_slicings and stage != "train":
            # One offset per molecule avoids millions of Python (mol, cut)
            # pairs for large evaluation sets. Deduplicate legacy automorphisms.
            counts = [
                len(self._scaffold_choices(int(i))) for i in self._filtered_indices
            ]
            self._slicing_offsets = np.concatenate(
                (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
            )

    def __len__(self):
        """Count molecules, or distinct molecule/scaffold evaluation pairs."""
        if self._slicing_offsets is not None:
            return int(self._slicing_offsets[-1])
        return len(self._filtered_indices)

    def _scaffold_choices(self, base_idx: int) -> list[tuple[int, ...]]:
        return list(
            dict.fromkeys(
                tuple(sorted(cut)) for cut in self._scaffold_atom_indices[base_idx]
            )
        )

    def _select_source(self, base_idx: int, target, slicing_idx: int | None = None):
        choices = self._scaffold_choices(base_idx)
        scaffold_indices = (
            random.choice(choices) if slicing_idx is None else choices[slicing_idx]
        )
        mask = torch.zeros(target.num_nodes, dtype=torch.bool)
        mask[torch.as_tensor(scaffold_indices, dtype=torch.long)] = True
        source = target.filter_nodes(mask)
        kept = sorted(int(i) for i in scaffold_indices)
        pairs = [(i, target_idx) for i, target_idx in enumerate(kept)]
        return source, pairs

    def __getitem__(self, index):
        """Build an interpolated training item or a kept-subgraph evaluation pair."""
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        slicing_idx = None
        if self._slicing_offsets is not None:
            molecule_idx = int(
                np.searchsorted(self._slicing_offsets, index, side="right") - 1
            )
            slicing_idx = int(index - self._slicing_offsets[molecule_idx])
            index = molecule_idx
        base_idx = int(self._filtered_indices[index])
        target = self.base_dataset[base_idx]
        source, scaffold_pairs = self._select_source(base_idx, target, slicing_idx)

        if self.stage == "train":
            t = self.time_dist.sample((1,)).squeeze(0)
            t = torch.clamp(t, min=0.0, max=1 - 1e-8)
            mol_t, mol_1, ins_targets = self._interpolate_train_atomic(
                source,
                target,
                t,
                scaffold_pairs=scaffold_pairs,
                existing_noise="jitter",
            )
            keep_mask = torch.zeros(mol_t.num_nodes, dtype=torch.long)
            keep_mask[: len(scaffold_pairs)] = 1
            mol_t.add_node_property("keep_mask", keep_mask)

            if hasattr(target, "y") and target.y is not None:
                mol_t.add_system_property("y", target.y)
            # Use the resolved molecule index, including when evaluation
            # enumerates multiple scaffold cuts of the same ligand.
            self._attach_context(base_idx, mol_t)
            return self._training_output(mol_t, mol_1, ins_targets, t)

        source, target = evaluation_pair(source, target)
        self._attach_context(base_idx, source)
        return source, target


def evaluation_pair(source, target):
    """Complete the source graph and mark every input atom as retained."""
    keep_mask = torch.ones(source.num_nodes, dtype=torch.long)
    n_nodes = source.num_nodes
    edge_index = build_fully_connected_edge_index(n_nodes)
    edge_dtype = source.e.dtype if source.e is not None else torch.long
    dense = torch.zeros(n_nodes, n_nodes, dtype=edge_dtype)
    if source.edge_index is not None and source.e is not None:
        dense[source.edge_index[0], source.edge_index[1]] = source.e
    source = rebuild_atomic_with_edges(
        source, edge_index, dense[edge_index[0], edge_index[1]]
    )
    source.add_node_property("keep_mask", keep_mask)
    return source, target
