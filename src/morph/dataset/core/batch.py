"""Native flat-tensor batching for Morph geometric graphs."""

# The short alias properties and mapping protocol are documented on the class.
# ruff: noqa: D102, D105

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from morph.dataset.core.containers import ChemAtomicData, iter_geometry
from morph.utils.graph_math import unsorted_segment_sum

if TYPE_CHECKING:
    from collections.abc import Iterable


def batch_unpack(batch: ChemBatch):
    """Return ``(x, a, c, e, edge_index, batch_idx)`` from a batch."""
    return batch.unpack()


def batch_remove_com(
    batch: ChemBatch, x_mean: torch.Tensor | None = None
) -> torch.Tensor:
    """Remove the per-graph coordinate mean, including co-framed context."""
    positions = batch.positions
    if x_mean is None:
        x_mean = unsorted_segment_sum(positions, batch.batch, batch.num_graphs)
        x_mean = x_mean / batch.num_nodes_per_graph.clamp(min=1).unsqueeze(1)
    else:
        assert x_mean.shape == (batch.num_graphs, positions.shape[-1]), (
            "x_mean must have the same shape as the coordinates"
        )
    for graph in iter_geometry(batch):
        graph.positions = graph.positions - x_mean[graph.batch]
    return x_mean


def batch_translate(batch: ChemBatch, translation: torch.Tensor) -> None:
    """Translate a batch and its geometric context by one vector per graph."""
    if translation.shape != (batch.num_graphs, batch.positions.shape[-1]):
        raise ValueError(
            "translation must have shape "
            f"({batch.num_graphs}, {batch.positions.shape[-1]})"
        )
    for graph in iter_geometry(batch):
        graph.positions.add_(translation[graph.batch])


def _clone_value(value):
    return value.clone() if hasattr(value, "clone") else value


class ChemBatch:
    """A batch of Morph graphs stored as graph-contiguous flat tensors.

    ``data`` holds node-, edge-, and graph-level tensors, while ``keys`` records
    their level. Connectivity is natively PyG-shaped ``[2, E]`` and all index
    tensors are ``int64``, matching the model and scatter operations directly.
    ``contexts[name]`` stores a separate batch sharing this batch's graph IDs
    and coordinate frame, including empty rows for unconditioned examples.
    """

    _ALIASES = {
        "x": "positions",
        "a": "atom_types",
        "c": "charge_types",
        "e": "edge_types",
    }
    _INTERNAL_ATTRS = frozenset(
        {
            "_data",
            "_keys",
            "_batch_idx",
            "_node_counts",
            "_edge_counts",
            "_num_graphs",
            "_max_seqlen_cache",
            "contexts",
        }
    )

    def __init__(
        self,
        *,
        data: dict[str, torch.Tensor],
        keys: dict[str, set[str]],
        batch_idx: torch.Tensor,
        node_counts: torch.Tensor,
        edge_counts: torch.Tensor,
        num_graphs: int,
        max_seqlen: int | None = None,
        contexts: dict[str, ChemBatch] | None = None,
    ) -> None:
        object.__setattr__(self, "_data", data)
        object.__setattr__(
            self, "_keys", {level: set(keys.get(level, ())) for level in keys}
        )
        for level in ("node", "edge", "system"):
            self._keys.setdefault(level, set())
        object.__setattr__(self, "_batch_idx", batch_idx.long())
        object.__setattr__(self, "_node_counts", node_counts.long())
        object.__setattr__(self, "_edge_counts", edge_counts.long())
        object.__setattr__(self, "_num_graphs", int(num_graphs))
        object.__setattr__(self, "_max_seqlen_cache", max_seqlen)
        object.__setattr__(self, "contexts", dict(contexts or {}))

    def __getattr__(self, name: str):
        data = self.__dict__.get("_data")
        if data is not None and name in data:
            return data[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name == "edge_index":
            self._data["edge_index"] = value.long().contiguous()
        elif name == "neighbor_list":
            self._data["edge_index"] = value.t().long().contiguous()
        elif name == "batch":
            object.__setattr__(self, "_batch_idx", value.long())
        elif (key := self._ALIASES.get(name)) is not None:
            self._data[key] = value
        elif name in self.__dict__.get("_data", {}):
            self._data[name] = value
        else:
            object.__setattr__(self, name, value)

    def __contains__(self, key: str) -> bool:
        if key == "neighbor_list":
            key = "edge_index"
        return key in self._data

    def __getitem__(self, key: str):
        if key == "neighbor_list":
            return self.neighbor_list
        return self._data[key]

    def __setitem__(self, key: str, value) -> None:
        if key == "neighbor_list":
            self.neighbor_list = value
        else:
            self._data[key] = value

    def __len__(self) -> int:
        return self.num_graphs

    @property
    def keys(self) -> dict[str, set[str]]:
        return self._keys

    @property
    def device(self) -> torch.device:
        return self.positions.device

    @property
    def x(self) -> torch.Tensor:
        return self.positions

    @property
    def a(self) -> torch.Tensor:
        return self.atom_types

    @property
    def c(self) -> torch.Tensor | None:
        return self._data.get("charge_types")

    @property
    def e(self) -> torch.Tensor | None:
        return self._data.get("edge_types")

    @property
    def edge_index(self) -> torch.Tensor:
        return self._data["edge_index"]

    @property
    def neighbor_list(self) -> torch.Tensor:
        """Return connectivity in the single-graph ``[E, 2]`` layout."""
        return self.edge_index.t()

    @property
    def batch(self) -> torch.Tensor:
        return self._batch_idx

    @property
    def batch_idx(self) -> torch.Tensor:
        return self._batch_idx

    @property
    def batch_size(self) -> int:
        return self._num_graphs

    @property
    def num_graphs(self) -> int:
        return self._num_graphs

    @property
    def num_nodes(self) -> int:
        return self.positions.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def num_nodes_per_graph(self) -> torch.Tensor:
        return self._node_counts

    @property
    def num_edges_per_graph(self) -> torch.Tensor:
        return self._edge_counts

    @property
    def max_num_nodes(self) -> int:
        return self.max_seqlen

    @property
    def max_seqlen(self) -> int:
        cached = self._max_seqlen_cache
        if cached is None:
            cached = (
                int(self._node_counts.max().item())
                if self._node_counts.numel() > 0
                else 0
            )
            object.__setattr__(self, "_max_seqlen_cache", cached)
        return cached

    def unpack(self):
        """Return ``(x, a, c, e, edge_index, batch)``."""
        return self.x, self.a, self.c, self.e, self.edge_index, self.batch

    def remove_com(self, x_mean: torch.Tensor | None = None) -> torch.Tensor:
        return batch_remove_com(self, x_mean)

    def _extra_attrs(self) -> dict:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in self._INTERNAL_ATTRS
        }

    def clone(self) -> ChemBatch:
        """Deep-clone all tensor storage and tensor-like loose attributes."""
        out = type(self)(
            data={key: value.clone() for key, value in self._data.items()},
            keys={level: set(names) for level, names in self._keys.items()},
            batch_idx=self._batch_idx.clone(),
            node_counts=self._node_counts.clone(),
            edge_counts=self._edge_counts.clone(),
            num_graphs=self._num_graphs,
            max_seqlen=self._max_seqlen_cache,
            contexts={key: value.clone() for key, value in self.contexts.items()},
        )
        for key, value in self._extra_attrs().items():
            object.__setattr__(out, key, _clone_value(value))
        return out

    @staticmethod
    def _move_tensor(
        value: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype | None,
        non_blocking: bool,
    ) -> torch.Tensor:
        target_dtype = (
            dtype if dtype is not None and value.is_floating_point() else None
        )
        return value.to(device=device, dtype=target_dtype, non_blocking=non_blocking)

    def to(
        self,
        device,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> ChemBatch:
        """Return a moved copy, following ``torch.Tensor.to`` semantics."""
        out = self.clone()
        return out.to_inplace(device, dtype=dtype, non_blocking=non_blocking)

    def to_inplace(
        self,
        device,
        dtype: torch.dtype | None = None,
        non_blocking: bool = True,
    ) -> ChemBatch:
        """Move this single-use batch without first copying it."""
        device = torch.device(device)
        for key, value in self._data.items():
            self._data[key] = self._move_tensor(value, device, dtype, non_blocking)
        object.__setattr__(
            self,
            "_batch_idx",
            self._batch_idx.to(device=device, non_blocking=non_blocking),
        )
        object.__setattr__(
            self,
            "_node_counts",
            self._node_counts.to(device=device, non_blocking=non_blocking),
        )
        object.__setattr__(
            self,
            "_edge_counts",
            self._edge_counts.to(device=device, non_blocking=non_blocking),
        )
        for key, value in self._extra_attrs().items():
            if isinstance(value, ChemBatch):
                value.to_inplace(device, dtype=dtype, non_blocking=non_blocking)
            elif isinstance(value, torch.Tensor):
                object.__setattr__(
                    self,
                    key,
                    self._move_tensor(value, device, dtype, non_blocking),
                )
            elif hasattr(value, "to"):
                object.__setattr__(self, key, value.to(device))
        for context in self.contexts.values():
            context.to_inplace(device, dtype=dtype, non_blocking=non_blocking)
        return self

    def cpu(self) -> ChemBatch:
        """Return a CPU copy."""
        return self.to("cpu")

    def pin_memory(self) -> ChemBatch:
        """Pin CPU tensor storage in place for asynchronous CUDA transfer."""
        for key, value in self._data.items():
            self._data[key] = value.pin_memory()
        object.__setattr__(self, "_batch_idx", self._batch_idx.pin_memory())
        object.__setattr__(self, "_node_counts", self._node_counts.pin_memory())
        object.__setattr__(self, "_edge_counts", self._edge_counts.pin_memory())
        for key, value in self._extra_attrs().items():
            if isinstance(value, torch.Tensor) or hasattr(value, "pin_memory"):
                object.__setattr__(self, key, value.pin_memory())
        for context in self.contexts.values():
            context.pin_memory()
        return self

    def contiguous(self) -> ChemBatch:
        """Make all stored tensors contiguous in place."""
        for key, value in self._data.items():
            self._data[key] = value.contiguous()
        object.__setattr__(self, "_batch_idx", self._batch_idx.contiguous())
        for context in self.contexts.values():
            context.contiguous()
        return self

    def to_data_list(self) -> list[ChemAtomicData]:
        """Split this batch into graph-local ``ChemAtomicData`` objects."""
        node_counts = self._node_counts.tolist()
        edge_counts = self._edge_counts.tolist()
        node_offsets = [0]
        edge_offsets = [0]
        for count in node_counts:
            node_offsets.append(node_offsets[-1] + count)
        for count in edge_counts:
            edge_offsets.append(edge_offsets[-1] + count)

        graphs = []
        smiles = getattr(self, "smiles", None)
        contexts = {key: value.to_data_list() for key, value in self.contexts.items()}
        for graph_idx in range(self.num_graphs):
            node_slice = slice(node_offsets[graph_idx], node_offsets[graph_idx + 1])
            edge_slice = slice(edge_offsets[graph_idx], edge_offsets[graph_idx + 1])
            edge_index = self.edge_index[:, edge_slice] - node_offsets[graph_idx]

            extra = {}
            for key in self._keys["node"]:
                if key not in {
                    "positions",
                    "atom_types",
                    "charge_types",
                    "is_auxiliary",
                }:
                    extra[key] = self._data[key][node_slice]
            for key in self._keys["edge"]:
                if key not in {"edge_index", "edge_types"}:
                    extra[key] = self._data[key][edge_slice]
            for key in self._keys["system"]:
                extra[key] = self._data[key][graph_idx : graph_idx + 1]

            graph = ChemAtomicData(
                positions=self.positions[node_slice],
                atom_types=self.atom_types[node_slice],
                charge_types=(
                    None
                    if "charge_types" not in self
                    else self.charge_types[node_slice]
                ),
                edge_types=(
                    None if "edge_types" not in self else self.edge_types[edge_slice]
                ),
                neighbor_list=edge_index.t().contiguous(),
                is_auxiliary=(
                    None
                    if "is_auxiliary" not in self
                    else self.is_auxiliary[node_slice]
                ),
                extra=extra,
                node_extra_keys=self._keys["node"]
                - {"positions", "atom_types", "charge_types", "is_auxiliary"},
                edge_extra_keys=self._keys["edge"] - {"edge_index", "edge_types"},
                system_extra_keys=self._keys["system"],
                smiles=(
                    smiles[graph_idx] if isinstance(smiles, (list, tuple)) else smiles
                ),
                contexts={key: values[graph_idx] for key, values in contexts.items()},
            )
            graphs.append(graph)
        return graphs

    @classmethod
    def from_data_list(cls, graphs: Iterable[ChemAtomicData]) -> ChemBatch:
        """Batch an iterable of Morph graphs."""
        return chembatch_from_graphs(list(graphs), attach_smiles=True)


def filter_nodes(
    data: ChemAtomicData | ChemBatch, mask: torch.Tensor
) -> ChemAtomicData | ChemBatch:
    """Keep selected nodes, drop dangling edges, and preserve graph identities."""
    if isinstance(data, ChemAtomicData):
        return data.filter_nodes(mask)

    mask = mask.to(torch.bool).view(-1)
    batch_idx = data.batch
    keep = mask.nonzero(as_tuple=True)[0]
    mapping = torch.full((mask.numel(),), -1, dtype=torch.long, device=data.device)
    mapping[keep] = torch.arange(keep.numel(), device=data.device)

    edge_index = data.edge_index
    if edge_index.numel() > 0:
        remapped = mapping[edge_index]
        edge_keep = torch.where((remapped[0] >= 0) & (remapped[1] >= 0))[0]
        new_edge_index = remapped[:, edge_keep]
        edge_graph = batch_idx[edge_index[0, edge_keep]]
    else:
        edge_keep = torch.empty(0, dtype=torch.long, device=data.device)
        new_edge_index = edge_index
        edge_graph = torch.empty(0, dtype=torch.long, device=data.device)

    # Resolve each variable-length selection once. Repeating boolean indexing
    # for every feature would synchronize the CPU with CUDA each time.
    new_batch_idx = batch_idx[keep]
    node_counts = torch.bincount(new_batch_idx, minlength=data.num_graphs)
    edge_counts = torch.bincount(edge_graph, minlength=data.num_graphs)
    node_data = {key: data[key][keep] for key in data.keys["node"]}
    edge_data = {
        key: data[key][edge_keep] for key in data.keys["edge"] if key != "edge_index"
    }
    system_data = {key: data[key] for key in data.keys["system"]}
    result = _chem_batch_from_level_data(
        node_data=node_data,
        edge_data=edge_data,
        system_data=system_data,
        edge_index=new_edge_index,
        node_counts=node_counts,
        edge_counts=edge_counts,
        batch_idx=new_batch_idx,
        num_graphs=data.num_graphs,
    )
    result.contexts.update(data.contexts)
    return result


def cumulative_node_offsets(data_list: list[ChemAtomicData]) -> list[int]:
    """Return the exclusive prefix sum of per-graph node counts."""
    offsets = []
    total = 0
    for data in data_list:
        offsets.append(total)
        total += data.num_nodes
    return offsets


def batch_from_flat(
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    charge_types: torch.Tensor | None,
    edge_types: torch.Tensor | None,
    edge_index: torch.Tensor | None,
    batch_idx: torch.Tensor,
    *,
    num_graphs: int | None = None,
    node_extras: dict[str, torch.Tensor] | None = None,
    edge_extras: dict[str, torch.Tensor] | None = None,
    system_extras: dict[str, torch.Tensor] | None = None,
    _already_sorted: bool = False,
    _node_counts: torch.Tensor | None = None,
    _edge_counts: torch.Tensor | None = None,
) -> ChemBatch:
    """Build a graph-contiguous batch from flat tensors.

    Generic inputs are stably grouped by graph. Internal collates can set
    ``_already_sorted`` because they construct graph-contiguous tensors directly,
    avoiding redundant sorting in the dataloader path.
    """
    batch_idx = batch_idx.long()
    if num_graphs is None:
        num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
    node_extras = node_extras or {}
    edge_extras = edge_extras or {}
    system_extras = system_extras or {}
    device = positions.device

    if _already_sorted:
        sorted_batch = batch_idx
        node_data = {"positions": positions, "atom_types": atom_types}
        if charge_types is not None:
            node_data["charge_types"] = charge_types
        node_data.update(node_extras)
    else:
        order = torch.argsort(batch_idx, stable=True)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=device)
        sorted_batch = batch_idx[order]
        node_data = {
            "positions": positions[order],
            "atom_types": atom_types[order],
        }
        if charge_types is not None:
            node_data["charge_types"] = charge_types[order]
        node_data.update({key: value[order] for key, value in node_extras.items()})
    node_counts = (
        _node_counts
        if _node_counts is not None
        else torch.bincount(sorted_batch, minlength=num_graphs)
    )

    if edge_index is None:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = edge_index.long()
    edge_data = {}
    edge_counts = (
        _edge_counts
        if _edge_counts is not None
        else torch.zeros(num_graphs, dtype=torch.long, device=device)
    )
    if edge_index.numel() > 0:
        if not _already_sorted or _edge_counts is None:
            edge_graph = batch_idx[edge_index[0]]
        if _already_sorted:
            if edge_types is not None:
                edge_data["edge_types"] = edge_types
            edge_data.update(edge_extras)
        else:
            edge_order = torch.argsort(edge_graph, stable=True)
            edge_index = inverse[edge_index][:, edge_order].contiguous()
            if edge_types is not None:
                edge_data["edge_types"] = edge_types[edge_order]
            edge_data.update(
                {key: value[edge_order] for key, value in edge_extras.items()}
            )
        if _edge_counts is None:
            edge_counts = torch.bincount(edge_graph, minlength=num_graphs)
    elif edge_types is not None:
        edge_data["edge_types"] = edge_types
    return _chem_batch_from_level_data(
        node_data=node_data,
        edge_data=edge_data,
        system_data=dict(system_extras),
        edge_index=edge_index,
        node_counts=node_counts,
        edge_counts=edge_counts,
        batch_idx=sorted_batch,
        num_graphs=num_graphs,
    )


def _common_extra_keys(graphs: list[ChemAtomicData], attr: str) -> set[str]:
    if not graphs:
        return set()
    common = set(getattr(graphs[0], attr))
    for graph in graphs[1:]:
        common.intersection_update(getattr(graph, attr))
    return common


def chembatch_from_graphs(
    graphs: list[ChemAtomicData],
    *,
    node_extra_keys: tuple = (),
    edge_extra_keys: tuple = (),
    system_extra_keys: tuple = (),
    attach_smiles: bool = False,
    context_names: tuple[str, ...] = (),
) -> ChemBatch:
    """Batch graphs, their registered tensors, and independent context graphs.

    ``context_names`` also creates empty context batches when every example is
    unconditional, so task encoders can retain their zero-gradient dependencies.
    """
    graphs = list(graphs)
    if not graphs:
        raise ValueError("cannot batch an empty graph list")
    node_keys = set(node_extra_keys) | _common_extra_keys(graphs, "node_extra_keys")
    edge_keys = set(edge_extra_keys) | _common_extra_keys(graphs, "edge_extra_keys")
    system_keys = set(system_extra_keys) | _common_extra_keys(
        graphs, "system_extra_keys"
    )

    def _gather(keys: set[str]) -> dict[str, torch.Tensor]:
        gathered = {}
        for key in keys:
            values = [graph.extra.get(key) for graph in graphs]
            if all(isinstance(value, torch.Tensor) for value in values):
                gathered[key] = torch.cat(values)
        return gathered

    positions = torch.cat([graph.positions for graph in graphs])
    atom_types = torch.cat([graph.atom_types for graph in graphs])
    charge_types = (
        torch.cat([graph.charge_types for graph in graphs])
        if all(graph.charge_types is not None for graph in graphs)
        else None
    )
    edge_types = (
        torch.cat([graph.edge_types for graph in graphs])
        if all(graph.edge_types is not None for graph in graphs)
        else None
    )
    node_extras = _gather(node_keys)
    if all(graph.is_auxiliary is not None for graph in graphs):
        node_extras["is_auxiliary"] = torch.cat(
            [graph.is_auxiliary for graph in graphs]
        )

    device = positions.device
    edge_parts = []
    batch_parts = []
    offset = 0
    for graph_idx, graph in enumerate(graphs):
        if graph.edge_index is not None and graph.edge_index.numel() > 0:
            edge_parts.append(graph.edge_index + offset)
        batch_parts.append(
            torch.full((graph.num_nodes,), graph_idx, dtype=torch.long, device=device)
        )
        offset += graph.num_nodes
    edge_index = (
        torch.cat(edge_parts, dim=1)
        if edge_parts
        else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    batch = batch_from_flat(
        positions,
        atom_types,
        charge_types,
        edge_types,
        edge_index,
        torch.cat(batch_parts),
        num_graphs=len(graphs),
        node_extras=node_extras,
        edge_extras=_gather(edge_keys),
        system_extras=_gather(system_keys),
        _already_sorted=True,
    )
    object.__setattr__(batch, "_max_seqlen_cache", max(g.num_nodes for g in graphs))
    # A missing context contributes an empty graph, retaining molecule graph IDs
    # for mixed conditional/unconditional batches and recursive context batches.
    for name in sorted(
        set(context_names) | {key for graph in graphs for key in graph.contexts}
    ):
        prototype = next(
            (graph.contexts[name] for graph in graphs if name in graph.contexts),
            graphs[0],
        )
        empty = ChemAtomicData(
            positions=prototype.positions[:0],
            atom_types=prototype.atom_types[:0],
            charge_types=None if prototype.c is None else prototype.c[:0],
            edge_types=None if prototype.e is None else prototype.e[:0],
            neighbor_list=None
            if prototype.neighbor_list is None
            else prototype.neighbor_list[:0],
            is_auxiliary=None
            if prototype.is_auxiliary is None
            else prototype.is_auxiliary[:0],
            extra={
                key: torch.zeros_like(prototype.extra[key])
                if key in prototype.system_extra_keys
                else prototype.extra[key][:0]
                for key in prototype.node_extra_keys
                | prototype.edge_extra_keys
                | prototype.system_extra_keys
                if isinstance(prototype.extra.get(key), torch.Tensor)
            },
            node_extra_keys=prototype.node_extra_keys,
            edge_extra_keys=prototype.edge_extra_keys,
            system_extra_keys=prototype.system_extra_keys,
        )
        batch.contexts[name] = chembatch_from_graphs(
            [graph.contexts.get(name, empty) for graph in graphs],
            attach_smiles=attach_smiles,
        )
    if attach_smiles:
        smiles = [graph.smiles for graph in graphs]
        if any(value is not None for value in smiles):
            batch.smiles = smiles
    return batch


def _chem_batch_from_level_data(
    *,
    node_data: dict[str, torch.Tensor],
    edge_data: dict[str, torch.Tensor],
    system_data: dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    node_counts: torch.Tensor,
    edge_counts: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
) -> ChemBatch:
    data = {**node_data, **edge_data, **system_data, "edge_index": edge_index}
    return ChemBatch(
        data=data,
        keys={
            "node": set(node_data),
            "edge": set(edge_data) | {"edge_index"},
            "system": set(system_data),
        },
        batch_idx=batch_idx,
        node_counts=node_counts,
        edge_counts=edge_counts,
        num_graphs=num_graphs,
    )
