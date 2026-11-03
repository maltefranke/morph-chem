"""Native per-graph container and RDKit conversion helpers.

``ChemAtomicData`` is Morph's single graph representation. It deliberately
contains only the tensors used by the generative model and lightweight metadata
that tells the batcher whether an optional tensor belongs to nodes, edges, or
the whole graph. Construction and topology edits do not invoke schema
validation; dataset adapters are responsible for validating external input.
"""

# The short alias properties and mapping protocol are documented on the class.
# ruff: noqa: D102, D105

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.utils import to_dense_adj

from morph.utils.rdkit_utils import BOND_IDX_MAP, IDX_BOND_MAP, tensors_to_rdkit_mol
from morph.utils.vocab_utils import index_to_token

if TYPE_CHECKING:
    from collections.abc import Iterable

    from morph.dataset.core.vocab import Vocab


def rdkit_to_token_payload(mol, vocab: Vocab, sanitize: bool = False) -> dict:
    """Convert an RDKit molecule to Morph's token-index graph tensors."""
    if sanitize:
        mol = Chem.Mol(mol)
        Chem.SanitizeMol(mol)
    if mol.GetNumConformers() == 0:
        raise ValueError("from_rdkit requires a mol with a 3D conformer")

    atoms = list(mol.GetAtoms())
    atom_types = torch.tensor(
        [vocab.atom_tokens.index(a.GetSymbol()) for a in atoms], dtype=torch.long
    )
    charge_types = torch.tensor(
        [vocab.charge_tokens.index(str(a.GetFormalCharge())) for a in atoms],
        dtype=torch.long,
    )
    positions = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)

    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_type = vocab.edge_tokens.index(str(BOND_IDX_MAP[bond.GetBondType()]))
        rows += [i, j]
        cols += [j, i]
        edge_types += [edge_type, edge_type]
    return {
        "positions": positions,
        "atom_types": atom_types,
        "charge_types": charge_types,
        "edge_types": torch.tensor(edge_types, dtype=torch.long),
        "neighbor_list": torch.tensor([rows, cols], dtype=torch.long).t().contiguous(),
    }


def _remap_neighbor_list(
    mask: torch.Tensor,
    neighbor_list: torch.Tensor | None,
    num_nodes: int,
    node_index_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Compact node indices under a keep mask and drop dangling edges."""
    if neighbor_list is None or neighbor_list.numel() == 0:
        return neighbor_list, None, node_index_map
    device = neighbor_list.device
    if node_index_map is None:
        keep = mask.nonzero(as_tuple=True)[0]
        node_index_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        node_index_map[keep] = torch.arange(keep.numel(), device=device)
    remapped = node_index_map[neighbor_list]
    edge_keep = (remapped[:, 0] >= 0) & (remapped[:, 1] >= 0)
    return remapped[edge_keep], edge_keep, node_index_map


# Dataset/conditioning fields known by the shared collates. A graph can register
# arbitrary additional fields through add_{node,edge,system}_property as well.
_NODE_CONDITION_KEYS = ("keep_mask", "fragment_mask", "linker_mask", "anchors")
_SYSTEM_CONDITION_KEYS = ("y", "qed", "logp")
_STRING_CONDITION_KEYS = ("smiles",)
_CONDITION_KEYS = (
    *_NODE_CONDITION_KEYS,
    *_SYSTEM_CONDITION_KEYS,
    *_STRING_CONDITION_KEYS,
)


def iter_geometry(graph):
    """Yield a graph and its context graphs, all sharing one coordinate frame.

    Contexts form a tree: attach each graph once, without cycles or shared nodes.
    Batched contexts must retain the same graph count and ordering as their owner.
    """
    yield graph
    for context in getattr(graph, "contexts", {}).values():
        yield from iter_geometry(context)


class ChemAtomicData:
    """A token-index geometric graph used throughout Morph.

    Core tensors use descriptive storage names and expose the established Morph
    aliases ``x``/``a``/``c``/``e``/``edge_index``. Connectivity is stored as an
    ``[E, 2]`` neighbor list on single graphs because topology-edit operations
    index both endpoints together. Optional tensors live in ``extra`` and are
    explicitly registered at node, edge, or graph level for batching.
    ``contexts[name]`` holds a separate graph in the same coordinate frame.
    Context nodes are unaffected by molecule topology edits.
    """

    __slots__ = (
        "_edge_extra_keys",
        "_node_extra_keys",
        "_system_extra_keys",
        "atom_types",
        "charge_types",
        "contexts",
        "edge_types",
        "extra",
        "is_auxiliary",
        "neighbor_list",
        "positions",
        "smiles",
    )

    _CORE_SLOTS = frozenset(__slots__)
    _NODE_CORE_FIELDS = ("positions", "atom_types", "charge_types", "is_auxiliary")

    def __init__(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        charge_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
        neighbor_list: torch.Tensor | None = None,
        is_auxiliary: torch.Tensor | None = None,
        *,
        extra: dict | None = None,
        node_extra_keys: Iterable[str] = (),
        edge_extra_keys: Iterable[str] = (),
        system_extra_keys: Iterable[str] = (),
        smiles: str | None = None,
        contexts: dict[str, ChemAtomicData] | None = None,
    ) -> None:
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "atom_types", atom_types)
        object.__setattr__(self, "charge_types", charge_types)
        object.__setattr__(self, "edge_types", edge_types)
        object.__setattr__(self, "neighbor_list", neighbor_list)
        object.__setattr__(self, "is_auxiliary", is_auxiliary)
        object.__setattr__(self, "extra", dict(extra) if extra is not None else {})
        object.__setattr__(self, "smiles", smiles)
        object.__setattr__(self, "_node_extra_keys", set(node_extra_keys))
        object.__setattr__(self, "_edge_extra_keys", set(edge_extra_keys))
        object.__setattr__(self, "_system_extra_keys", set(system_extra_keys))
        object.__setattr__(self, "contexts", dict(contexts or {}))

    def __setattr__(self, name: str, value) -> None:
        aliases = {
            "x": "positions",
            "a": "atom_types",
            "c": "charge_types",
            "e": "edge_types",
        }
        if name == "edge_index":
            object.__setattr__(
                self,
                "neighbor_list",
                None if value is None else value.t().contiguous().long(),
            )
        elif (slot := aliases.get(name)) is not None:
            object.__setattr__(self, slot, value)
        elif name in self._CORE_SLOTS:
            object.__setattr__(self, name, value)
        else:
            self.extra[name] = value

    def __getattr__(self, name: str):
        if name.startswith("__") or name == "extra":
            raise AttributeError(name)
        try:
            return self.extra[name]
        except (KeyError, AttributeError):
            raise AttributeError(name) from None

    def __delattr__(self, name: str) -> None:
        if name in self._CORE_SLOTS:
            object.__delattr__(self, name)
            return
        try:
            del self.extra[name]
        except KeyError:
            raise AttributeError(name) from None
        self._node_extra_keys.discard(name)
        self._edge_extra_keys.discard(name)
        self._system_extra_keys.discard(name)

    def __contains__(self, key: str) -> bool:
        if key in self._CORE_SLOTS:
            return getattr(self, key, None) is not None
        return key in self.extra

    def __getitem__(self, key: str):
        return getattr(self, key)

    @property
    def num_nodes(self) -> int:
        return self.positions.shape[0]

    @property
    def num_edges(self) -> int:
        return 0 if self.neighbor_list is None else self.neighbor_list.shape[0]

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
        return self.charge_types

    @property
    def e(self) -> torch.Tensor | None:
        return self.edge_types

    @property
    def edge_index(self) -> torch.Tensor | None:
        return None if self.neighbor_list is None else self.neighbor_list.t()

    @property
    def node_extra_keys(self) -> frozenset[str]:
        return frozenset(self._node_extra_keys)

    @property
    def edge_extra_keys(self) -> frozenset[str]:
        return frozenset(self._edge_extra_keys)

    @property
    def system_extra_keys(self) -> frozenset[str]:
        return frozenset(self._system_extra_keys)

    def add_node_property(self, key: str, value, node_dim: int = 0) -> None:
        del node_dim  # Node tensors are consistently node-major in Morph.
        self.extra[key] = value
        self._node_extra_keys.add(key)
        self._edge_extra_keys.discard(key)
        self._system_extra_keys.discard(key)

    def add_edge_property(self, key: str, value) -> None:
        self.extra[key] = value
        self._edge_extra_keys.add(key)
        self._node_extra_keys.discard(key)
        self._system_extra_keys.discard(key)

    def add_system_property(self, key: str, value) -> None:
        self.extra[key] = value
        self._system_extra_keys.add(key)
        self._node_extra_keys.discard(key)
        self._edge_extra_keys.discard(key)

    def unpack(self):
        """Return ``(x, a, c, e, edge_index)``."""
        return self.x, self.a, self.c, self.e, self.edge_index

    def get_edge_types_triu(self, no_bond_idx: int = 0):
        """Return upper-triangular edge types and their ``[2, E]`` indices."""
        dense = to_dense_adj(
            self.edge_index, edge_attr=self.edge_types, max_num_nodes=self.num_nodes
        ).squeeze()
        if no_bond_idx != 0:
            present = (
                to_dense_adj(self.edge_index, max_num_nodes=self.num_nodes)
                .squeeze()
                .bool()
            )
            dense = dense.masked_fill(~present, no_bond_idx)
        edge_index_triu = torch.triu_indices(
            self.num_nodes, self.num_nodes, offset=1, device=self.device
        )
        return dense[edge_index_triu[0], edge_index_triu[1]], edge_index_triu

    def to_rdkit_mol(
        self,
        atom_tokens: list[str],
        edge_tokens: list[str],
        charge_tokens: list[str],
        sanitize: bool = True,
        *,
        implicit_hydrogens: bool | None = None,
    ):
        """Convert to RDKit; H-free vocabularies default to implicit hydrogens.

        This uses the vocabulary, never the atoms in an individual sample: a
        missing H in an explicit-H model must still count as a chemistry error.
        """
        if self.num_nodes == 0:
            return None

        # Hydra can supply ListConfig vocabularies. Resolve each small table
        # once instead of repeating config lookups for every atom and pair.
        atom_tokens = tuple(atom_tokens)
        edge_tokens = tuple(edge_tokens)
        charge_tokens = tuple(charge_tokens)

        atom_indices = self.atom_types.detach().cpu().numpy()
        positions = self.positions.detach().cpu().numpy()
        charge_indices = self.charge_types.detach().cpu().numpy()
        atom_names = [index_to_token(atom_tokens, int(index)) for index in atom_indices]
        charges = [
            int(index_to_token(charge_tokens, int(index))) for index in charge_indices
        ]

        bond_types = []
        bond_indices = []
        if self.num_nodes > 1:
            edge_types_triu, edge_index_triu = self.get_edge_types_triu(
                no_bond_idx=edge_tokens.index("<NO_BOND>")
            )
            indices = (
                edge_types_triu.detach().cpu().numpy().astype(np.int64, copy=False)
            )
            # Convert only chemical bonds to Python objects; dense graphs mostly
            # contain <NO_BOND> pairs. Preserve upper-triangle insertion order.
            is_bond = np.array([token != "<NO_BOND>" for token in edge_tokens])[indices]
            bond_indices = edge_index_triu.detach().cpu().numpy()[:, is_bond].T.tolist()
            bond_types = [
                IDX_BOND_MAP[int(edge_tokens[index])] for index in indices[is_bond]
            ]

        return tensors_to_rdkit_mol(
            atom_names,
            positions,
            charges,
            bond_types,
            bond_indices,
            sanitize=sanitize,
            implicit_hydrogens=(
                "H" not in atom_tokens
                if implicit_hydrogens is None
                else implicit_hydrogens
            ),
        )

    def to_rdkit(self, vocab: Vocab, sanitize: bool = True):
        """Convert this graph to an RDKit molecule using ``vocab``."""
        return self.to_rdkit_mol(
            vocab.atom_tokens,
            vocab.edge_tokens,
            vocab.charge_tokens,
            sanitize=sanitize,
        )

    @classmethod
    def from_rdkit(cls, mol, vocab: Vocab, sanitize: bool = False) -> ChemAtomicData:
        """Build a graph from an RDKit molecule."""
        return cls(**rdkit_to_token_payload(mol, vocab, sanitize))

    @classmethod
    def from_atomic(cls, data: ChemAtomicData, clone: bool = True) -> ChemAtomicData:
        """Copy ``data`` and ensure it carries an auxiliary-node mask."""

        def _copy(value):
            return value.clone() if clone and isinstance(value, torch.Tensor) else value

        is_auxiliary = data.is_auxiliary
        if is_auxiliary is None:
            is_auxiliary = torch.zeros(
                (data.num_nodes, 1), dtype=torch.bool, device=data.device
            )
        else:
            is_auxiliary = _copy(is_auxiliary)
        return cls(
            positions=_copy(data.positions),
            atom_types=_copy(data.atom_types),
            charge_types=_copy(data.charge_types),
            edge_types=_copy(data.edge_types),
            neighbor_list=_copy(data.neighbor_list),
            is_auxiliary=is_auxiliary,
            extra={key: _copy(value) for key, value in data.extra.items()},
            node_extra_keys=data.node_extra_keys,
            edge_extra_keys=data.edge_extra_keys,
            system_extra_keys=data.system_extra_keys,
            smiles=data.smiles,
            contexts={
                key: value.clone() if clone else value
                for key, value in data.contexts.items()
            },
        )

    def filter_nodes(
        self, mask: torch.Tensor, node_index_map: torch.Tensor | None = None
    ) -> ChemAtomicData:
        """Return a graph containing only nodes selected by ``mask``."""
        mask = mask.to(torch.bool).view(-1)
        neighbor_list, edge_keep, _ = _remap_neighbor_list(
            mask, self.neighbor_list, self.num_nodes, node_index_map
        )

        extra = dict(self.extra)
        for key in self._node_extra_keys:
            value = extra.get(key)
            if isinstance(value, torch.Tensor):
                extra[key] = value[mask]
        if edge_keep is not None:
            for key in self._edge_extra_keys:
                value = extra.get(key)
                if isinstance(value, torch.Tensor):
                    extra[key] = value[edge_keep]

        return type(self)(
            positions=self.positions[mask],
            atom_types=self.atom_types[mask],
            charge_types=(
                None if self.charge_types is None else self.charge_types[mask]
            ),
            edge_types=(
                self.edge_types
                if self.edge_types is None or edge_keep is None
                else self.edge_types[edge_keep]
            ),
            neighbor_list=neighbor_list,
            is_auxiliary=(
                None if self.is_auxiliary is None else self.is_auxiliary[mask]
            ),
            extra=extra,
            node_extra_keys=self._node_extra_keys,
            edge_extra_keys=self._edge_extra_keys,
            system_extra_keys=self._system_extra_keys,
            smiles=self.smiles,
            contexts=self.contexts,
        )

    def pad(self, num_auxiliary: int) -> ChemAtomicData:
        """Append zero-initialized auxiliary nodes in place."""
        if num_auxiliary <= 0:
            return self
        old_num_nodes = self.num_nodes
        if self.is_auxiliary is None:
            self.is_auxiliary = torch.zeros(
                (old_num_nodes, 1), dtype=torch.bool, device=self.device
            )
        for name in self._NODE_CORE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                padding = torch.zeros(
                    (num_auxiliary, *value.shape[1:]),
                    dtype=value.dtype,
                    device=value.device,
                )
                object.__setattr__(self, name, torch.cat([value, padding]))
        for key in self._node_extra_keys:
            value = self.extra.get(key)
            if isinstance(value, torch.Tensor):
                padding = torch.zeros(
                    (num_auxiliary, *value.shape[1:]),
                    dtype=value.dtype,
                    device=value.device,
                )
                self.extra[key] = torch.cat([value, padding])
        self.is_auxiliary[old_num_nodes:] = True
        return self

    def permute_nodes(self, indices: torch.Tensor | list[int]) -> ChemAtomicData:
        """Reorder nodes and remap connectivity in place."""
        perm = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        if self.neighbor_list is not None and self.neighbor_list.numel() > 0:
            mapping = torch.full(
                (self.num_nodes,), -1, dtype=torch.long, device=self.device
            )
            mapping[perm] = torch.arange(perm.numel(), device=self.device)
            self.neighbor_list = mapping[self.neighbor_list]
        for name in self._NODE_CORE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                object.__setattr__(self, name, value[perm])
        for key in self._node_extra_keys:
            value = self.extra.get(key)
            if isinstance(value, torch.Tensor):
                self.extra[key] = value[perm]
        return self

    def clone(self) -> ChemAtomicData:
        """Deep-clone tensors and context graphs, preserving other metadata."""

        def _copy(value):
            return value.clone() if isinstance(value, torch.Tensor) else value

        return type(self)(
            positions=_copy(self.positions),
            atom_types=_copy(self.atom_types),
            charge_types=_copy(self.charge_types),
            edge_types=_copy(self.edge_types),
            neighbor_list=_copy(self.neighbor_list),
            is_auxiliary=_copy(self.is_auxiliary),
            extra={key: _copy(value) for key, value in self.extra.items()},
            node_extra_keys=self._node_extra_keys,
            edge_extra_keys=self._edge_extra_keys,
            system_extra_keys=self._system_extra_keys,
            smiles=self.smiles,
            contexts={key: value.clone() for key, value in self.contexts.items()},
        )


def make_chem_atomic(
    x: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor | None = None,
    e: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    **conditions,
) -> ChemAtomicData:
    """Build a graph from Morph's canonical short-form tensor arguments."""
    graph = ChemAtomicData(
        positions=x,
        atom_types=a.long(),
        charge_types=None if c is None else c.long(),
        edge_types=None if e is None else e.long(),
        neighbor_list=(
            None if edge_index is None else edge_index.t().contiguous().long()
        ),
    )
    for key, value in conditions.items():
        if value is None:
            continue
        if key in _NODE_CONDITION_KEYS:
            graph.add_node_property(key, value)
        elif key in _SYSTEM_CONDITION_KEYS:
            graph.add_system_property(key, value)
        else:
            setattr(graph, key, value)
    return graph
