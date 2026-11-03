"""The molecular Diffusion Transformer.

This is the repository's sole first-class generative backbone. It combines
adaLN-Zero conditioning, equal-budget SwiGLU feed-forwards, QK normalization,
pair-biased attention, exactly symmetric undirected-edge decoding,
and residual graph-centred coordinate prediction.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from morph.model.layers import lora as lora_layers
from morph.model.layers.attention import (
    PairBias,
    compute_dense_index,
    packed_self_attention,
)
from morph.model.layers.embedding import RBFEmbedding
from morph.model.layers.heads import InsertionEdgePredictorMixin
from morph.model.layers.kernels import graph_broadcast, normalized_edge_features
from morph.model.layers.lora import LoRALinear
from morph.utils.graph_math import batched_triu_edge_layout

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from morph.dataset.core.atomic_data import ChemBatch

USE_DENSE_INDEX = os.environ.get("MORPH_DENSE_INDEX", "1") == "1"


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    batch_idx: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply graph-level adaptive modulation to packed node features."""
    return x * (1 + graph_broadcast(scale, batch_idx, offsets)) + graph_broadcast(
        shift, batch_idx, offsets
    )


class FinalLayer(nn.Module):
    """Final adaLN-conditioned projection in packed layout."""

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim)
        )

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """Modulate normalized node features with graph-level conditioning."""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale, batch_idx))


def _swiglu_width(hidden_dim: int, mlp_ratio: float, multiple_of: int) -> int:
    """Match the parameter budget of a GELU FFN with the given ratio."""
    if mlp_ratio <= 0:
        raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
    if multiple_of <= 0:
        raise ValueError(f"multiple_of must be positive, got {multiple_of}")
    target = 2.0 * hidden_dim * mlp_ratio / 3.0
    return max(multiple_of, round(target / multiple_of) * multiple_of)


class SwiGLU(nn.Module):
    """Fused SwiGLU feed-forward network."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.gate_value = nn.Linear(in_features, 2 * hidden_features)
        self.out_proj = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a SiLU-gated feed-forward projection."""
        gate, value = self.gate_value(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(gate) * value)


class DiTBlock(nn.Module):
    """adaLN-Zero block with QK normalization and equal-budget SwiGLU."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        ffn_hidden_dim: int | None = None,
        ffn_multiple_of: int = 64,
        qk_norm: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        if ffn_hidden_dim is None:
            ffn_hidden_dim = _swiglu_width(hidden_dim, mlp_ratio, ffn_multiple_of)
        if ffn_hidden_dim <= 0:
            raise ValueError(f"ffn_hidden_dim must be positive, got {ffn_hidden_dim}")
        self.mlp = SwiGLU(hidden_dim, ffn_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        batch_idx: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        pair_bias: torch.Tensor | None = None,
        atom_mask: torch.Tensor | None = None,
        dense_slot: torch.Tensor | None = None,
        *,
        normalized_attention_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply conditioned packed attention and gated feed-forward updates.

        ``normalized_attention_input`` optionally supplies an already normalized
        and conditioned replacement for ``norm1(x)`` before native AdaLN. It
        changes only the attention branch input; ``x`` remains the raw residual
        skip. Omitting it retains the pretrained block's original computation.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        x_norm = (
            self.norm1(x)
            if normalized_attention_input is None
            else normalized_attention_input
        )
        x_mod = modulate(x_norm, shift_msa, scale_msa, batch_idx, cu_seqlens)
        qkv = self.qkv(x_mod).view(-1, 3, self.num_heads, self.head_dim)
        attn_out = packed_self_attention(
            qkv,
            cu_seqlens,
            max_seqlen,
            batch_idx,
            pair_bias=pair_bias,
            atom_mask=atom_mask,
            dense_slot=dense_slot,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        attn_out = self.proj(attn_out.reshape(-1, self.hidden_dim))

        x = x + graph_broadcast(gate_msa, batch_idx, cu_seqlens) * attn_out
        x = x + graph_broadcast(gate_mlp, batch_idx, cu_seqlens) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp, batch_idx, cu_seqlens)
        )
        return x


class SymmetricEdgeDecoder(nn.Module):
    """Decode undirected edges from a compact DeepSets-style pair feature."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        distance_dim: int,
        output_dim: int,
        pair_dim: int,
    ):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, pair_dim),
            nn.SiLU(),
        )
        input_dim = 2 * pair_dim + edge_dim + distance_dim
        self.output = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: tuple[torch.Tensor, torch.Tensor],
        edge_embed: torch.Tensor,
        distance_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Decode symmetric edge features from nodes, bonds, and distances."""
        rows, cols = edge_index
        p = self.node_proj(h)
        normalized = normalized_edge_features(
            p, edge_index, edge_embed, distance_embed, self.output[0]
        )
        if normalized is not None:
            return self.output[3](self.output[2](self.output[1](normalized)))
        p_i, p_j = p[rows], p[cols]
        scale = math.sqrt(0.5)
        pair_sum = scale * (p_i + p_j)
        pair_difference = scale * (p_i - p_j).abs()
        del p_i, p_j
        return self.output(
            torch.cat([pair_sum, pair_difference, edge_embed, distance_embed], dim=-1)
        )


class ResidualCoordinateReadout(nn.Module):
    """Predict residual clean coordinates and optionally project out graph COM."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_init_std: float = 0.0,
        center_output: bool = True,
    ):
        super().__init__()
        if output_init_std < 0:
            raise ValueError(
                f"output_init_std must be non-negative, got {output_init_std}"
            )
        self.center_output = center_output
        self.norm = nn.RMSNorm(input_dim, eps=1e-6)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 3)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        if output_init_std == 0:
            nn.init.zeros_(self.output.weight)
        else:
            nn.init.normal_(self.output.weight, mean=0.0, std=output_init_std)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        h: torch.Tensor,
        x_t: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Predict clean coordinates with optional graphwise centering."""
        delta = self.output(F.silu(self.hidden(self.norm(h))))
        x_1 = x_t + delta
        if not self.center_output:
            return x_1
        if num_graphs == 1:
            # Inductor 2.9 can miscompile scatter-mean with one output graph.
            return x_1 - x_1.mean(dim=0, keepdim=True)
        center = scatter(x_1, batch, dim=0, dim_size=num_graphs, reduce="mean")
        return x_1 - center[batch]


class DiTBackbone(nn.Module):
    """Scalar DiT trunk with learned geometry scaling in every layer/head."""

    LORA_TARGETS = ("qkv", "proj", "mlp.gate_value", "mlp.out_proj")

    def __init__(
        self,
        d_input: int,
        d_cond: int,
        d_model: int = 512,
        out_node_nf: int = 256,
        out_edge_nf: int | None = None,
        num_layers: int = 8,
        nhead: int = 16,
        mlp_ratio: float = 4.0,
        in_edge_nf: int = 128,
        rbf_embedding_args: DictConfig | None = None,
        pair_bias_hidden_dim: int = 32,
        pair_bias_all_layers: bool | None = None,
        pair_bias_mode: str | None = None,
        ffn_hidden_dim: int | None = None,
        ffn_multiple_of: int = 64,
        qk_norm: bool = True,
        edge_pair_dim: int | None = None,
        coordinate_output_init_std: float = 0.0,
        center_coordinate_output: bool = True,
        num_atom_types: int | None = 10,
        lora: DictConfig | dict[str, Any] | None = None,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.d_model = d_model
        self.out_node_nf = out_node_nf
        self.out_edge_nf = out_node_nf if out_edge_nf is None else out_edge_nf
        # These two arguments only remain for saved checkpoint configurations.
        # New presets omit both and always use scaling in every layer. Older
        # shared-bias configs supplied all_layers but predate the mode field.
        if pair_bias_mode is None:
            pair_bias_mode = "scale" if pair_bias_all_layers is None else "shared"
        self.pair_bias_all_layers = (
            True if pair_bias_all_layers is None else bool(pair_bias_all_layers)
        )
        if pair_bias_mode not in {"shared", "scale"}:
            raise ValueError(
                f"Pair-bias sweep mode {pair_bias_mode!r} has been retired; "
                "use its archived training source to load that checkpoint"
            )
        if pair_bias_mode != "shared" and not self.pair_bias_all_layers:
            raise ValueError(
                "Layer-specific pair bias requires pair_bias_all_layers=True"
            )
        self.pair_bias_mode = pair_bias_mode
        if self.out_edge_nf <= 0:
            raise ValueError(f"out_edge_nf must be positive, got {self.out_edge_nf}")
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(d_cond, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    d_model,
                    nhead,
                    mlp_ratio=mlp_ratio,
                    ffn_hidden_dim=ffn_hidden_dim,
                    ffn_multiple_of=ffn_multiple_of,
                    qk_norm=qk_norm,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer = FinalLayer(d_model, d_model)
        self.node_out = (
            nn.Identity()
            if out_node_nf == d_model
            else nn.Sequential(
                nn.Linear(d_model, out_node_nf),
                nn.SiLU(),
                nn.Linear(out_node_nf, out_node_nf),
            )
        )
        self.rbf_embedding = (
            RBFEmbedding()
            if rbf_embedding_args is None
            else hydra.utils.instantiate(rbf_embedding_args)
        )
        self._rbf_has_dropout = any(
            isinstance(module, nn.Dropout) and module.p > 0
            for module in self.rbf_embedding.modules()
        )
        rbf_out_dim = self.rbf_embedding.out_dim
        self.pair_bias = PairBias(
            rbf_embedding=self.rbf_embedding,
            rbf_out_dim=rbf_out_dim,
            edge_nf=in_edge_nf,
            num_heads=nhead,
            hidden_dim=pair_bias_hidden_dim,
            num_atom_types=num_atom_types,
        )
        if edge_pair_dim is None:
            edge_pair_dim = min(out_node_nf, 256)
        if edge_pair_dim <= 0:
            raise ValueError(f"edge_pair_dim must be positive, got {edge_pair_dim}")
        self.edge_output = SymmetricEdgeDecoder(
            node_dim=out_node_nf,
            edge_dim=in_edge_nf,
            distance_dim=rbf_out_dim,
            output_dim=self.out_edge_nf,
            pair_dim=edge_pair_dim,
        )

        self._initialize_weights()
        # FinalLayer produces a shared latent, not the task prediction itself.
        nn.init.xavier_uniform_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        # Build this after the generic initialization so its deliberately zero
        # task projection is preserved at initialization.
        self.pos_out = ResidualCoordinateReadout(
            input_dim=out_node_nf,
            hidden_dim=d_model,
            output_init_std=coordinate_output_init_std,
            center_output=center_coordinate_output,
        )
        # Keep parameter names and initialization compatible with the selected
        # scaling checkpoint and preserve the common model's RNG state.
        if pair_bias_mode == "scale":
            for block in self.blocks:
                block.geometry_scale = nn.Parameter(torch.ones(nhead))
        if lora is not None and bool(lora.get("enabled", False)):
            rank = int(lora.get("rank", 8))
            self.enable_lora(
                rank=rank,
                alpha=float(lora.get("alpha", rank)),
                dropout=float(lora.get("dropout", 0.0)),
                targets=lora.get("targets", self.LORA_TARGETS),
                freeze_base=bool(lora.get("freeze_base", True)),
            )

    def pair_bias_for_layer(self, pair_bias, index):
        """Select a layer's bias while retaining shared geometry computation."""
        if not self.pair_bias_all_layers and index > 0:
            return None
        block = self.blocks[index]
        if self.pair_bias_mode == "scale":
            # Multiplying a padded -inf by a learnable factor gives NaN factor
            # gradients; scale finite entries and restore padding afterwards.
            valid = torch.isfinite(pair_bias)
            scaled = pair_bias.masked_fill(~valid, 0) * block.geometry_scale.to(
                pair_bias.dtype
            ).view(1, -1, 1, 1)
            return scaled.masked_fill(~valid, -torch.inf)
        return pair_bias

    def enable_lora(
        self,
        *,
        rank: int = 8,
        alpha: float | None = None,
        dropout: float = 0.0,
        targets: tuple[str, ...] | list[str] = LORA_TARGETS,
        freeze_base: bool = True,
    ) -> int:
        """Add LoRA to selected block projections.

        Targets are paths relative to every :class:`DiTBlock`. With
        ``freeze_base=True``, only the low-rank matrices remain trainable inside
        this backbone. Returns their total parameter count.
        """
        if any(lora_layers.iter_lora_layers(self)):
            raise RuntimeError("LoRA is already enabled on this DiT backbone")
        if isinstance(targets, str):
            targets = (targets,)
        targets = tuple(targets)
        if not targets:
            raise ValueError("At least one LoRA target is required")
        if len(set(targets)) != len(targets):
            raise ValueError(f"LoRA targets must be unique, got {targets}")
        unknown = sorted(set(targets) - set(self.LORA_TARGETS))
        if unknown:
            raise ValueError(
                f"Unknown LoRA targets {unknown}; choose from {self.LORA_TARGETS}"
            )
        alpha = float(rank if alpha is None else alpha)
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}")
        if freeze_base:
            self.requires_grad_(False)

        for block in self.blocks:
            modules = {
                "qkv": (block, "qkv"),
                "proj": (block, "proj"),
                "mlp.gate_value": (block.mlp, "gate_value"),
                "mlp.out_proj": (block.mlp, "out_proj"),
            }
            for target in targets:
                parent, name = modules[target]
                linear = getattr(parent, name)
                setattr(
                    parent,
                    name,
                    LoRALinear.from_linear(
                        linear,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        freeze_base=freeze_base,
                    ),
                )
        return lora_layers.set_lora_trainable(self)

    @property
    def has_lora(self) -> bool:
        """Whether this backbone contains low-rank adapters."""
        return any(lora_layers.iter_lora_layers(self))

    def set_lora_trainable(self, trainable: bool = True) -> int:
        """Toggle gradients for every adapter and return its parameter count."""
        return lora_layers.set_lora_trainable(self, trainable)

    def merge_lora_(self) -> DiTBackbone:
        """Fold all adapters into their base projections for inference."""
        lora_layers.merge_lora_(self)
        return self

    def unmerge_lora_(self) -> DiTBackbone:
        """Restore the separate adapter projections used for training."""
        lora_layers.unmerge_lora_(self)
        return self

    def train(self, mode: bool = True) -> DiTBackbone:
        """Set training mode, unmerging adapters before any optimization."""
        # Never optimize a base weight that already contains its adapter delta.
        if mode:
            self.unmerge_lora_()
        return super().train(mode)

    def _initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

        # Keep PairBias non-trivial once adaLN-Zero opens the attention residual.
        nn.init.xavier_uniform_(self.pair_bias.down_proj.weight)

    def encode(
        self,
        a_embed: torch.Tensor,
        x: torch.Tensor,
        cond: torch.Tensor,
        batch: torch.Tensor,
        node_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project packed node inputs and graph conditioning."""
        batch_size = cond.shape[0]
        h = self.input_proj(a_embed) + self.pos_embedding(x)
        c = self.cond_proj(cond)
        if node_counts is None:
            node_counts = torch.bincount(batch, minlength=batch_size)
        cu_seqlens = F.pad(node_counts.cumsum(dim=0), (1, 0)).to(torch.int32)
        return h, cu_seqlens, c

    def decode(
        self,
        h: torch.Tensor,
        c: torch.Tensor,
        batch_idx: torch.Tensor,
        x: torch.Tensor,
        edge_index: tuple[torch.Tensor, torch.Tensor],
        e_embed: torch.Tensor,
        distance_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the node, position, and undirected-edge output projections."""
        h = self.node_out(self.final_layer(h, c, batch_idx))
        # Preserve independent dropout draws for custom stochastic RBF configs.
        # Shipped configs use dropout=0 and take the one-pass reuse path.
        if self.training and self._rbf_has_dropout:
            rows, cols = edge_index
            distance_embed = self.rbf_embedding((x[rows] - x[cols]).norm(dim=1))
        e_out = self.edge_output(h, edge_index, e_embed, distance_embed)
        return h, self.pos_out(h, x, batch_idx, c.shape[0]), e_out

    def forward(
        self,
        a_embed: torch.Tensor,
        x: torch.Tensor,
        cond: torch.Tensor,
        edge_index: tuple[torch.Tensor, torch.Tensor],
        e_embed: torch.Tensor,
        batch: torch.Tensor,
        max_seqlen: int,
        atom_types: torch.Tensor | None = None,
        node_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the packed DiT and return node, position, and triu edge features."""
        if node_counts is None:
            node_counts = torch.bincount(batch, minlength=cond.shape[0])
        h, cu_seqlens, c = self.encode(a_embed, x, cond, batch, node_counts)
        batch_size = cond.shape[0]
        triu_edge_positions, full_to_triu = batched_triu_edge_layout(
            edge_index, batch, node_counts
        )
        dense_slot = (
            compute_dense_index(batch, batch_size, max_seqlen, node_counts)
            if USE_DENSE_INDEX
            else None
        )
        pair_bias, atom_mask, distance_embed = self.pair_bias(
            x,
            edge_index,
            e_embed,
            batch,
            batch_size,
            max_seqlen,
            atom_types=atom_types,
            lengths=node_counts,
            dense_slot=dense_slot,
            triu_edge_positions=triu_edge_positions,
        )

        for idx, block in enumerate(self.blocks):
            h = block(
                h,
                c,
                batch,
                cu_seqlens,
                max_seqlen,
                pair_bias=self.pair_bias_for_layer(pair_bias, idx),
                atom_mask=atom_mask,
                dense_slot=dense_slot,
            )

        triu_edge_index = (
            edge_index[0][triu_edge_positions],
            edge_index[1][triu_edge_positions],
        )
        outputs = self.decode(
            h,
            c,
            batch,
            x,
            triu_edge_index,
            e_embed[triu_edge_positions],
            distance_embed,
        )
        return *outputs, full_to_triu


# ---------------------------------------------------------------------------
#  Embedding and complete model
# ---------------------------------------------------------------------------


class DiTEmbedding(nn.Module):
    """Embedding module for DiT.

    Produces atom embeddings, edge embeddings, and a *graph-level* conditioning
    vector (time + node count).

    Conditioning remains graph-level so ``DiTBackbone`` can feed it through
    adaLN without duplicating it across packed nodes.
    """

    def __init__(
        self,
        atom_type_embedding_args: DictConfig,
        edge_type_embedding_args: DictConfig,
        time_embedding_args: DictConfig,
        node_count_embedding_args: DictConfig,
        incident_bond_embedding: bool = False,
    ):
        super().__init__()
        self.atom_type_embedding = hydra.utils.instantiate(atom_type_embedding_args)
        self.edge_type_embedding = hydra.utils.instantiate(edge_type_embedding_args)
        self.time_embedding = hydra.utils.instantiate(time_embedding_args)
        self.node_count_embedding = hydra.utils.instantiate(node_count_embedding_args)
        self.incident_bond_projection = None
        if incident_bond_embedding:
            if int(edge_type_embedding_args.num_embeddings) < 2:
                raise ValueError("Incident-bond embeddings require bond tokens")
            atom_dim = (
                atom_type_embedding_args.get("out_dim")
                or atom_type_embedding_args.embedding_dim
            )
            edge_dim = (
                edge_type_embedding_args.get("out_dim")
                or edge_type_embedding_args.embedding_dim
            )
            self.incident_bond_projection = nn.Linear(
                int(edge_dim), int(atom_dim), bias=False
            )
            nn.init.zeros_(self.incident_bond_projection.weight)

    def forward(
        self,
        mols: ChemBatch,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            a_embed: (N, atom_dim) atom embeddings (per-node)
            e_embed: (E, edge_dim) edge embeddings (per-edge)
            cond: (B, cond_dim) graph-level conditioning vector
        """
        a, e = mols.a, mols.e

        a_embed = self.atom_type_embedding(a)
        e_embed = self.edge_type_embedding(e)
        if self.incident_bond_projection is not None:
            # Reuse the same edge embeddings supplied to attention and the
            # pair decoder. NO_BOND contributes nothing to the incident sum.
            incident = e_embed.new_zeros((a.shape[0], e_embed.shape[-1]))
            bonded_embeddings = e_embed * (e != 0).unsqueeze(-1)
            # Edges are directed and symmetric: summing each row counts every
            # neighbour once at its receiving node. Keep sums, not averages.
            incident.index_add_(0, mols.edge_index[0], bonded_embeddings)
            a_embed = a_embed + self.incident_bond_projection(incident)

        t_embed = self.time_embedding(t)
        n_embed = self.node_count_embedding(mols.num_nodes_per_graph)

        cond = torch.cat([t_embed, n_embed], dim=-1)

        return a_embed, e_embed, cond


# ---------------------------------------------------------------------------
#  Complete model
# ---------------------------------------------------------------------------


class DiTBackboneWithHeads(InsertionEdgePredictorMixin, nn.Module):
    """Full DiT Model: Embedding -> DiT Backbone -> Prediction Heads.

    This is the public model assembled by ``model=backbones/dit``.
    """

    def __init__(
        self,
        atom_type_embedding_args: DictConfig,
        edge_type_embedding_args: DictConfig,
        time_embedding_args: DictConfig,
        node_count_embedding_args: DictConfig,
        backbone_model_args: DictConfig,
        heads_args: DictConfig,
        insertion_head_args: DictConfig,
        ins_edge_head_args: DictConfig,
        incident_bond_embedding: bool = False,
    ):
        super().__init__()

        self.embedding_backbone = DiTEmbedding(
            atom_type_embedding_args=atom_type_embedding_args,
            edge_type_embedding_args=edge_type_embedding_args,
            time_embedding_args=time_embedding_args,
            node_count_embedding_args=node_count_embedding_args,
            incident_bond_embedding=incident_bond_embedding,
        )

        self.backbone = hydra.utils.instantiate(backbone_model_args)
        self.heads = hydra.utils.instantiate(heads_args)
        self.insertion_head = hydra.utils.instantiate(insertion_head_args)
        self.ins_edge_head = hydra.utils.instantiate(ins_edge_head_args)

    def freeze_pretrained(self) -> int:
        """Freeze the loaded model while leaving DiT LoRA matrices trainable."""
        before = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.requires_grad_(False)
        self.backbone.set_lora_trainable()
        after = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return before - after

    def output_head_modules(self) -> tuple[nn.Module, ...]:
        """Return prediction heads, including the backbone's coordinate readout."""
        return (
            self.heads,
            self.insertion_head,
            self.ins_edge_head,
            self.backbone.pos_out,
        )

    def set_output_heads_trainable(self, trainable: bool = True) -> int:
        """Set head trainability, returning the number of parameters changed."""
        before = sum(p.numel() for p in self.parameters() if p.requires_grad)
        for head in self.output_head_modules():
            head.requires_grad_(trainable)
        after = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return abs(after - before)

    def initialize_output_priors(self, distributions) -> None:
        """Initialize categorical outputs from training-set marginal priors."""
        self.heads.initialize_class_priors(
            atom=distributions.atom_type_distribution,
            charge=distributions.charge_type_distribution,
            edge=distributions.edge_type_distribution,
        )
        self.insertion_head.initialize_class_priors(
            atom=distributions.atom_type_distribution,
            charge=distributions.charge_type_distribution,
        )
        self.ins_edge_head.initialize_edge_prior(distributions.edge_type_distribution)

    def embed_inputs(self, mols_t: ChemBatch, t: torch.Tensor):
        """Embed molecular inputs and graph conditions; scalar tasks may extend this."""
        return self.embedding_backbone(mols_t, t)

    def forward(
        self,
        mols_t: ChemBatch,
        t: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> dict[str, Any]:
        """Forward pass through Embedding -> DiT Backbone -> Heads."""
        x, a, _c, _e, edge_index, batch = mols_t.unpack()
        # max_seqlen must be a precomputed Python int on the compiled (training)
        # path: computing it here reads segment_lengths -> .item()
        # (aten._local_scalar_dense), which fails torch.compile's backend and drops
        # the whole DiT to eager. Eager callers (sampling) leave it None.
        if max_seqlen is None:
            max_seqlen = mols_t.max_seqlen

        a_embed, e_embed, cond = self.embed_inputs(mols_t, t)

        edge_index_tuple = (edge_index[0], edge_index[1])

        h, x_out, e_out_triu, full_to_triu = self.backbone(
            a_embed,
            x,
            cond,
            edge_index_tuple,
            e_embed,
            batch,
            max_seqlen,
            atom_types=a,
            node_counts=mols_t.num_nodes_per_graph,
        )

        return self._assemble_outputs(
            h,
            x_out,
            e_out_triu,
            full_to_triu,
            x,
            batch,
            edge_index_tuple,
        )
