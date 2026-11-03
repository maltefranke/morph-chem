"""Prediction heads for the molecular DiT.

SwiGLU readout adapters feed compact task projections. Inserted--inserted bonds
use an undirected decoder that observes both endpoints and both spawn contexts.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from morph.model.layers.embedding import RBFEmbedding

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from morph.dataset.core.insertion_data import InsertedAtoms


def _small_output_init(linear: nn.Linear, std: float) -> None:
    if std < 0:
        raise ValueError(f"output init std must be non-negative, got {std}")
    if std == 0:
        nn.init.zeros_(linear.weight)
    else:
        nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError(
            "initial_probability must lie strictly between zero and one, "
            f"got {probability}"
        )
    return math.log(probability) - math.log1p(-probability)


def _inverse_softplus(rate: float) -> float:
    if rate <= 0:
        raise ValueError(f"initial_rate must be positive, got {rate}")
    # log(expm1(x)) is the inverse of softplus.  Avoid overflow for large x.
    return rate if rate > 20.0 else math.log(math.expm1(rate))


def _set_categorical_bias(linear: nn.Linear, probabilities: torch.Tensor) -> None:
    probabilities = torch.as_tensor(
        probabilities, device=linear.bias.device, dtype=linear.bias.dtype
    ).flatten()
    if probabilities.numel() != linear.out_features:
        raise ValueError(
            f"categorical prior has {probabilities.numel()} entries, expected "
            f"{linear.out_features}"
        )
    log_prior = probabilities.clamp_min(1e-8).log()
    # Softmax is invariant to a shared constant.  Centering avoids an unnecessary
    # large common bias for vocabularies with very small probabilities.
    linear.bias.data.copy_(log_prior - log_prior.mean())


class SwiGLUReadoutAdapter(nn.Module):
    """Pre-normalized, width-reducing SwiGLU feature adapter."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(
                f"adapter dimensions must be positive, got {input_dim}->{output_dim}"
            )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.RMSNorm(input_dim, eps=1e-6)
        self.gate_value = nn.Linear(input_dim, 2 * output_dim)
        nn.init.xavier_uniform_(self.gate_value.weight)
        nn.init.zeros_(self.gate_value.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features and apply a SiLU-gated readout adapter."""
        gate, value = self.gate_value(self.norm(features)).chunk(2, dim=-1)
        return F.silu(gate) * value


class MultiHeadModule(nn.Module):
    """Independent readout adapters with task-specific projections by default.

    Each head config retains ``input_dim`` and ``output_dim`` and additionally
    selects an adapter ``group``. With ``independent_adapters=False``, heads of
    the same level/group reuse one adapter.
    ``self.heads`` intentionally remains a ModuleDict of the final task projections:
    shared training code uses ``heads[name].parameters()`` for DDP-safe dummies.
    """

    _LEVEL_CONFIG_KEYS = {
        "node": "node_heads",
        "graph": "graph_heads",
        "edge": "edge_heads",
    }

    def __init__(
        self,
        heads_configs: DictConfig,
        output_init_std: float = 1e-3,
        independent_adapters: bool = True,
    ):
        super().__init__()
        self.output_init_std = float(output_init_std)
        self.independent_adapters = bool(independent_adapters)
        self.graph_aggregation = heads_configs.get("graph_aggregation", "mean")
        self.adapters = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.head_types: dict[str, str] = {}
        self.head_adapter_keys: dict[str, str] = {}
        self.has_graph_heads = False
        self.has_edge_heads = False

        adapter_specs: dict[str, tuple[int, int]] = {}
        for level, config_key in self._LEVEL_CONFIG_KEYS.items():
            level_configs = heads_configs.get(config_key)
            if level_configs is None:
                continue
            if level == "graph":
                self.has_graph_heads = True
            elif level == "edge":
                self.has_edge_heads = True

            for name, config in level_configs.items():
                input_dim = int(config.input_dim)
                output_dim = int(config.output_dim)
                adapter_dim = int(config.adapter_dim)
                group = (
                    name
                    if self.independent_adapters
                    else str(config.get("group", "default"))
                )
                adapter_key = f"{level}_{group}"
                spec = (input_dim, adapter_dim)
                if adapter_key in adapter_specs and adapter_specs[adapter_key] != spec:
                    raise ValueError(
                        f"heads sharing adapter {adapter_key!r} disagree on "
                        "dimensions: "
                        f"{adapter_specs[adapter_key]} versus {spec}"
                    )
                adapter_specs[adapter_key] = spec

                projection = nn.Linear(adapter_dim, output_dim)
                _small_output_init(projection, self.output_init_std)
                if "initial_probability" in config:
                    nn.init.constant_(
                        projection.bias, _logit(float(config.initial_probability))
                    )
                elif "initial_rate" in config:
                    nn.init.constant_(
                        projection.bias, _inverse_softplus(float(config.initial_rate))
                    )

                self.heads[name] = projection
                self.head_types[name] = level
                self.head_adapter_keys[name] = adapter_key

        for key, (input_dim, output_dim) in adapter_specs.items():
            self.adapters[key] = SwiGLUReadoutAdapter(input_dim, output_dim)
        self.rate_heads = tuple(name for name in self.heads if "rate" in name)
        self.edge_heads = tuple(
            name for name, level in self.head_types.items() if level == "edge"
        )

    def initialize_class_priors(
        self,
        *,
        atom: torch.Tensor | None = None,
        charge: torch.Tensor | None = None,
        edge: torch.Tensor | None = None,
    ) -> None:
        """Initialize categorical output biases from token probabilities."""
        priors = {
            "atom_type_head": atom,
            "charge_head": charge,
            "edge_type_head": edge,
        }
        for name, probabilities in priors.items():
            if probabilities is not None and name in self.heads:
                _set_categorical_bias(self.heads[name], probabilities)

    def _graph_features(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if batch is None:
            raise ValueError("batch is required when graph heads are configured")
        num_graphs = int(batch.max()) + 1 if batch.numel() else 0
        if self.graph_aggregation == "mean":
            return scatter(h, batch, dim=0, dim_size=num_graphs, reduce="mean")
        if self.graph_aggregation == "sum":
            return scatter(h, batch, dim=0, dim_size=num_graphs, reduce="sum")
        raise ValueError(f"Unknown graph aggregation: {self.graph_aggregation}")

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor | None,
        edge_attr: torch.Tensor | None = None,
        full_to_triu: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Apply configured node, graph, and edge prediction heads."""
        sources: dict[str, torch.Tensor] = {"node": h}
        if self.has_graph_heads:
            sources["graph"] = self._graph_features(h, batch)
        if self.has_edge_heads:
            if edge_attr is None:
                raise ValueError("edge_attr is required when edge heads are configured")
            sources["edge"] = edge_attr

        adapter_cache: dict[str, torch.Tensor] = {}
        outputs: dict[str, torch.Tensor] = {}
        for name, projection in self.heads.items():
            adapter_key = self.head_adapter_keys[name]
            if adapter_key not in adapter_cache:
                level = self.head_types[name]
                adapter_cache[adapter_key] = self.adapters[adapter_key](sources[level])
            outputs[name] = projection(adapter_cache[adapter_key])

        for name in self.rate_heads:
            outputs[name] = F.softplus(outputs[name])
        if full_to_triu is not None:
            for name in self.edge_heads:
                outputs[name] = outputs[name][full_to_triu]
        return outputs


class UnimodalInsertionHead(nn.Module):
    """Shared-adapter position/type head for the nearest pending insertion."""

    def __init__(
        self,
        hidden_dim: int,
        N_a: int,
        N_c: int,
        readout_dim: int,
        output_init_std: float = 1e-3,
        independent_adapters: bool = True,
    ):
        super().__init__()
        self.N_a = int(N_a)
        self.N_c = int(N_c)
        self.independent_adapters = bool(independent_adapters)
        if self.independent_adapters:
            self.adapters = nn.ModuleDict(
                {
                    name: SwiGLUReadoutAdapter(hidden_dim, readout_dim)
                    for name in ("atom", "charge", "offset")
                }
            )
        else:
            self.adapter = SwiGLUReadoutAdapter(hidden_dim, readout_dim)
        self.atom_projection = nn.Linear(readout_dim, self.N_a)
        self.charge_projection = nn.Linear(readout_dim, self.N_c)
        self.offset_projection = nn.Linear(readout_dim, 3)
        _small_output_init(self.atom_projection, output_init_std)
        _small_output_init(self.charge_projection, output_init_std)
        # A small nonzero output lets an independent offset adapter receive
        # gradients immediately while keeping the initial mean near the spawn.
        _small_output_init(self.offset_projection, output_init_std)

    def initialize_class_priors(
        self,
        atom: torch.Tensor | None = None,
        charge: torch.Tensor | None = None,
    ) -> None:
        """Initialize inserted-atom and charge biases from token priors."""
        if atom is not None:
            _set_categorical_bias(self.atom_projection, atom)
        if charge is not None:
            _set_categorical_bias(self.charge_projection, charge)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Predict one insertion location and categorical logits per node."""
        del edge_index
        if self.independent_adapters:
            atom_features = self.adapters["atom"](h)
            charge_features = self.adapters["charge"](h)
            offset_features = self.adapters["offset"](h)
        else:
            features = self.adapter(h)
            atom_features = charge_features = offset_features = features
        return {
            "mu": x + self.offset_projection(offset_features),
            "a_logits": self.atom_projection(atom_features),
            "c_logits": self.charge_projection(charge_features),
        }


class GMMInsertionHead(nn.Module):
    """Compact, numerically bounded mixture readout for insertion content.

    Cartesian component means are residual offsets from the spawning atom. A
    shared normalized adapter feeds independent projections for mixture weights, scales,
    categorical content and component offsets.  In particular, scales are
    bounded in log space: this prevents both variance collapse (exploding NLL)
    and an arbitrarily broad component that can cheaply absorb several targets.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_components: int,
        N_a: int,
        N_c: int,
        readout_dim: int | None = None,
        output_init_std: float = 1e-3,
        sigma_min: float = 0.05,
        sigma_max: float = 2.0,
        sigma_init: float = 0.5,
    ):
        super().__init__()
        if readout_dim is None:
            readout_dim = hidden_dim
        if not 0.0 < sigma_min < sigma_init < sigma_max:
            raise ValueError(
                "GMM scales must satisfy 0 < sigma_min < sigma_init < "
                f"sigma_max, got {sigma_min}, {sigma_init}, {sigma_max}"
            )

        self.num_components = int(num_components)
        self.N_a = int(N_a)
        self.N_c = int(N_c)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        self.adapter = SwiGLUReadoutAdapter(hidden_dim, readout_dim)
        self.mixture_projection = nn.Linear(readout_dim, self.num_components)
        self.scale_projection = nn.Linear(readout_dim, self.num_components)
        self.atom_projection = nn.Linear(readout_dim, self.num_components * self.N_a)
        self.charge_projection = nn.Linear(readout_dim, self.num_components * self.N_c)
        self.offset_projection = nn.Linear(readout_dim, self.num_components * 3)

        # Uniform component weights, exact categorical marginals and a fixed,
        # useful initial scale make the first NLL finite and predictable.  Small
        # independent offset weights break component symmetry without initially
        # displacing the means much.
        _small_output_init(self.mixture_projection, 0.0)
        _small_output_init(self.scale_projection, 0.0)
        _small_output_init(self.atom_projection, 0.0)
        _small_output_init(self.charge_projection, 0.0)
        _small_output_init(self.offset_projection, output_init_std)

        log_min = math.log(self.sigma_min)
        log_max = math.log(self.sigma_max)
        init_fraction = (math.log(sigma_init) - log_min) / (log_max - log_min)
        nn.init.constant_(self.scale_projection.bias, _logit(init_fraction))

    def initialize_class_priors(
        self,
        atom: torch.Tensor | None = None,
        charge: torch.Tensor | None = None,
    ) -> None:
        """Initialize every component with the empirical categorical marginals."""
        if atom is not None:
            bias = self.atom_projection.bias
            atom = torch.as_tensor(atom, device=bias.device, dtype=bias.dtype)
            if atom.numel() != self.N_a:
                raise ValueError(
                    f"atom prior has {atom.numel()} entries, expected {self.N_a}"
                )
            atom_logits = atom.clamp_min(1e-8).log()
            atom_logits = atom_logits - atom_logits.mean()
            bias.data.copy_(atom_logits.repeat(self.num_components))
        if charge is not None:
            bias = self.charge_projection.bias
            charge = torch.as_tensor(charge, device=bias.device, dtype=bias.dtype)
            if charge.numel() != self.N_c:
                raise ValueError(
                    f"charge prior has {charge.numel()} entries, expected {self.N_c}"
                )
            charge_logits = charge.clamp_min(1e-8).log()
            charge_logits = charge_logits - charge_logits.mean()
            bias.data.copy_(charge_logits.repeat(self.num_components))

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Predict typed Gaussian mixture parameters for each spawning node."""
        del edge_index
        n_nodes = h.shape[0]
        features = self.adapter(h)
        scale_fraction = torch.sigmoid(self.scale_projection(features))
        log_sigma = math.log(self.sigma_min) + scale_fraction * (
            math.log(self.sigma_max) - math.log(self.sigma_min)
        )
        return {
            "pi": F.softmax(self.mixture_projection(features), dim=-1),
            "mu": x.unsqueeze(1)
            + self.offset_projection(features).view(n_nodes, self.num_components, 3),
            "sigma": log_sigma.exp(),
            "a_probs": F.softmax(
                self.atom_projection(features).view(
                    n_nodes, self.num_components, self.N_a
                ),
                dim=-1,
            ),
            "c_probs": F.softmax(
                self.charge_projection(features).view(
                    n_nodes, self.num_components, self.N_c
                ),
                dim=-1,
            ),
        }


class InsertionEdgeHead(nn.Module):
    """Compact insertion-edge head with a symmetric inserted-pair branch.

    Inserted-to-existing bonds retain their meaningful role asymmetry.  For a bond
    between two simultaneously inserted atoms, both endpoints are encoded with the
    same function and combined through sum/absolute-difference features.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_edge_types: int,
        n_atom_types: int,
        n_charge_types: int,
        readout_dim: int,
        rbf_embedding_args: DictConfig | None = None,
        output_init_std: float = 1e-3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_edge_types = n_edge_types
        self.n_atom_types = n_atom_types
        self.n_charge_types = n_charge_types
        self.rbf_embedding = (
            RBFEmbedding()
            if rbf_embedding_args is None
            else hydra.utils.instantiate(rbf_embedding_args)
        )
        self._rbf_has_dropout = any(
            isinstance(module, nn.Dropout) and module.p > 0
            for module in self.rbf_embedding.modules()
        )

        dist_feat_dim = self.rbf_embedding.out_dim
        inserted_attr_dim = n_atom_types + n_charge_types
        existing_input_dim = (
            2 * hidden_dim + 2 * dist_feat_dim + n_atom_types + inserted_attr_dim
        )
        self.edge_mlp = self._make_decoder(
            existing_input_dim, readout_dim, n_edge_types, output_init_std
        )

        endpoint_input_dim = hidden_dim + dist_feat_dim + inserted_attr_dim
        self.inserted_endpoint = nn.Sequential(
            nn.LayerNorm(endpoint_input_dim),
            nn.Linear(endpoint_input_dim, readout_dim),
            nn.SiLU(),
        )
        pair_input_dim = 2 * readout_dim + dist_feat_dim
        self.inserted_pair_decoder = self._make_decoder(
            pair_input_dim, readout_dim, n_edge_types, output_init_std
        )
        nn.init.xavier_uniform_(self.inserted_endpoint[1].weight)
        nn.init.zeros_(self.inserted_endpoint[1].bias)

    @staticmethod
    def _make_decoder(
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        output_init_std: float,
    ) -> nn.Sequential:
        decoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.kaiming_normal_(decoder[1].weight, nonlinearity="relu")
        nn.init.zeros_(decoder[1].bias)
        _small_output_init(decoder[-1], output_init_std)
        return decoder

    def initialize_edge_prior(self, probabilities: torch.Tensor) -> None:
        """Initialize both insertion-bond decoders from bond probabilities."""
        _set_categorical_bias(self.edge_mlp[-1], probabilities)
        _set_categorical_bias(self.inserted_pair_decoder[-1], probabilities)

    def _inserted_endpoint_features(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        spawn_idx: torch.Tensor,
        inserted: InsertedAtoms,
    ) -> torch.Tensor:
        atom = F.one_hot(inserted.a.long(), num_classes=self.n_atom_types).to(
            device=h.device, dtype=h.dtype
        )
        charge = F.one_hot(inserted.c.long(), num_classes=self.n_charge_types).to(
            device=h.device, dtype=h.dtype
        )
        distance = (inserted.x.detach() - x.detach()[spawn_idx]).norm(dim=-1)
        distance_features = self.rbf_embedding(distance.clamp_min(1e-6))
        return self.inserted_endpoint(
            torch.cat([h[spawn_idx], distance_features, atom, charge], dim=-1)
        )

    def forward_inserted_to_inserted(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        spawn_src_idx: torch.Tensor,
        spawn_dst_idx: torch.Tensor,
        src: InsertedAtoms,
        dst: InsertedAtoms,
    ) -> torch.Tensor:
        """Predict bond logits between pairs of future atoms."""
        if spawn_src_idx.numel() == 0:
            return torch.empty((0, self.n_edge_types), device=h.device, dtype=h.dtype)
        if spawn_src_idx.shape != spawn_dst_idx.shape:
            raise ValueError(
                "spawn_src_idx and spawn_dst_idx must have identical shape, got "
                f"{tuple(spawn_src_idx.shape)} and {tuple(spawn_dst_idx.shape)}"
            )

        src_features = self._inserted_endpoint_features(h, x, spawn_src_idx, src)
        dst_features = self._inserted_endpoint_features(h, x, spawn_dst_idx, dst)
        scale = math.sqrt(0.5)
        pair = torch.cat(
            [
                scale * (src_features + dst_features),
                scale * (src_features - dst_features).abs(),
                self.rbf_embedding(
                    (src.x.detach() - dst.x.detach()).norm(dim=-1).clamp_min(1e-6)
                ),
            ],
            dim=-1,
        )
        return self.inserted_pair_decoder(pair)

    def _prepare_edge_features(
        self,
        h_spawn: torch.Tensor,
        h_existing: torch.Tensor,
        spawn_pos: torch.Tensor,
        existing_pos: torch.Tensor,
        existing_atom_idx: torch.Tensor,
        ins_pos: torch.Tensor,
        ins_a: torch.Tensor,
        ins_c: torch.Tensor,
        insertion_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build edge-MLP features for insertion edge prediction."""
        ins_atom_type = F.one_hot(ins_a.long(), num_classes=self.n_atom_types).to(
            dtype=h_spawn.dtype, device=h_spawn.device
        )
        ins_charge_type = F.one_hot(ins_c.long(), num_classes=self.n_charge_types).to(
            dtype=h_spawn.dtype, device=h_spawn.device
        )

        dist_to_spawn = torch.norm(ins_pos - spawn_pos, dim=-1).clamp(min=1e-6)
        if insertion_index is not None:
            # At inference the spawn-side features are constant for all candidate
            # existing endpoints. Embed them once per insertion, then expand.
            dist_spawn_features = self.rbf_embedding(dist_to_spawn)[insertion_index]
            h_spawn = h_spawn[insertion_index]
            ins_atom_type = ins_atom_type[insertion_index]
            ins_charge_type = ins_charge_type[insertion_index]
            ins_pos = ins_pos[insertion_index]

        dist_to_existing = torch.norm(ins_pos - existing_pos, dim=-1).clamp(min=1e-6)
        if insertion_index is None and not (self.training and self._rbf_has_dropout):
            distance_features = self.rbf_embedding(
                torch.cat([dist_to_spawn, dist_to_existing])
            )
            dist_spawn_features, dist_existing_features = distance_features.chunk(2)
        else:
            if insertion_index is None:
                dist_spawn_features = self.rbf_embedding(dist_to_spawn)
            dist_existing_features = self.rbf_embedding(dist_to_existing)

        existing_atom_type = F.one_hot(
            existing_atom_idx.long(), num_classes=self.n_atom_types
        ).to(dtype=h_spawn.dtype, device=h_spawn.device)

        return torch.cat(
            [
                h_spawn,
                h_existing,
                dist_spawn_features,
                existing_atom_type,
                dist_existing_features,
                ins_atom_type,
                ins_charge_type,
            ],
            dim=-1,
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        node_atom_types: torch.Tensor,
        spawn_node_idx: torch.Tensor,
        existing_node_idx: torch.Tensor,
        ins: InsertedAtoms,
    ) -> torch.Tensor:
        """Predict bond logits from future atoms to existing atoms."""
        if spawn_node_idx.numel() == 0:
            return torch.empty((0, self.n_edge_types), device=h.device, dtype=h.dtype)

        # Use fixed insertion attributes to decouple edge prediction from GMM sampling.
        x_detached = x.detach()
        edge_features = self._prepare_edge_features(
            h_spawn=h[spawn_node_idx],
            h_existing=h[existing_node_idx],
            spawn_pos=x_detached[spawn_node_idx],
            existing_pos=x_detached[existing_node_idx],
            existing_atom_idx=node_atom_types[existing_node_idx],
            ins_pos=ins.x.detach(),
            ins_a=ins.a,
            ins_c=ins.c,
        )

        edge_logits = self.edge_mlp(edge_features)

        return edge_logits

    @staticmethod
    def _insertion_edge_pairs(
        insertion_indices: torch.Tensor, batch: torch.Tensor, node_counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pair every insertion with every node sharing its graph.

        ``batch`` must be graph-contiguous, as every :class:`ChemBatch` is.
        Returns ``(spawn_idx, existing_idx, local)`` where ``spawn_idx`` /
        ``existing_idx`` are global node indices and ``local`` indexes the
        per-insertion tensors. Row-major over insertions then nodes, matching the
        original per-insertion loop order (downstream code relies on it).
        """
        insertion_graphs = batch[insertion_indices]
        pair_counts = node_counts[insertion_graphs]
        num_pairs = int(pair_counts.sum())
        local = torch.arange(
            insertion_indices.numel(), device=batch.device
        ).repeat_interleave(pair_counts, output_size=num_pairs)
        graph_offsets = node_counts.cumsum(0) - node_counts
        pair_offsets = pair_counts.cumsum(0) - pair_counts
        within_graph = (
            torch.arange(num_pairs, device=batch.device) - pair_offsets[local]
        )
        existing_idx = graph_offsets[insertion_graphs[local]] + within_graph
        return insertion_indices[local], existing_idx, local

    def predict_inserted_to_existing(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        node_atom_types: torch.Tensor,
        batch: torch.Tensor,
        insertion_indices: torch.Tensor,
        node_counts: torch.Tensor,
        ins: InsertedAtoms,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict edges for all insertions to all other nodes in the same graph.

        This is used during inference to get edge predictions.

        Args:
            h: [N, hidden_dim] - Node features
            x: [N, 3] - Node positions
            node_atom_types: [N] - Atom type index for each current-state node
            batch: [N] - Batch assignment
            insertion_indices: [N_ins] - Selected spawning node indices.
            node_counts: [B] - Current node counts before deletion.
            ins: Inserted-atom features aligned with ``insertion_indices``.

        Returns:
            spawn_idx: [E_ins] - Spawn node indices
            existing_idx: [E_ins] - Existing endpoint node indices
            edge_logits: [E_ins, n_edge_types] - Edge type logits
        """
        device = h.device

        if insertion_indices.numel() == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, self.n_edge_types, device=device),
            )

        # Pair each insertion with every node in its graph (including the spawn
        # node). ``local`` indexes the per-insertion inserted-atom tensors.
        spawn_idx, existing_idx, local = self._insertion_edge_pairs(
            insertion_indices, batch, node_counts
        )

        # Compute insertion-constant features once rather than once for every
        # candidate existing endpoint.
        x_detached = x.detach()
        edge_features = self._prepare_edge_features(
            h_spawn=h[insertion_indices],
            h_existing=h[existing_idx],
            spawn_pos=x_detached[insertion_indices],
            existing_pos=x_detached[existing_idx],
            existing_atom_idx=node_atom_types[existing_idx],
            ins_pos=ins.x.detach(),
            ins_a=ins.a,
            ins_c=ins.c,
            insertion_index=local,
        )
        edge_logits = self.edge_mlp(edge_features)

        return spawn_idx, existing_idx, edge_logits


class InsertionEdgePredictorMixin:
    """Prediction glue shared by DiT models with insertion-edge heads."""

    heads: MultiHeadModule
    insertion_head: nn.Module
    ins_edge_head: InsertionEdgeHead

    def _assemble_outputs(
        self,
        h: torch.Tensor,
        x_out: torch.Tensor,
        e_out_triu: torch.Tensor,
        full_to_triu: torch.Tensor,
        x: torch.Tensor,
        batch: torch.Tensor,
        edge_index_tuple: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, object]:
        out_dict = self.heads(h, batch, e_out_triu, full_to_triu=full_to_triu)
        out_dict["pos_head"] = x_out
        out_dict["insertion_head"] = self.insertion_head(h, x, edge_index_tuple)
        out_dict["h_latent"] = h
        return out_dict

    def predict_inserted_to_existing_edges(
        self,
        mols_t,
        out_dict: dict[str, object],
        spawn_node_idx: torch.Tensor,
        existing_node_idx: torch.Tensor,
        ins: InsertedAtoms,
    ) -> torch.Tensor:
        """Decode future-to-current bonds using the cached node latents."""
        return self.ins_edge_head(
            h=out_dict["h_latent"],
            x=mols_t.x,
            node_atom_types=mols_t.a,
            spawn_node_idx=spawn_node_idx,
            existing_node_idx=existing_node_idx,
            ins=ins,
        )

    def predict_inserted_to_inserted_edges(
        self,
        mols_t,
        out_dict: dict[str, object],
        spawn_src_idx: torch.Tensor,
        spawn_dst_idx: torch.Tensor,
        src: InsertedAtoms,
        dst: InsertedAtoms,
    ) -> torch.Tensor:
        """Decode future-to-future bonds using the cached node latents."""
        return self.ins_edge_head.forward_inserted_to_inserted(
            h=out_dict["h_latent"],
            x=mols_t.x,
            spawn_src_idx=spawn_src_idx,
            spawn_dst_idx=spawn_dst_idx,
            src=src,
            dst=dst,
        )
