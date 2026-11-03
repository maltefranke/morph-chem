"""Count-embedding delta conditioning through the pretrained graph port."""

from copy import deepcopy

from morph.model.backbones.conditional_dit import ConditionalDiTBackboneWithHeads
from morph.model.finetune.atom_count.atom_count import atom_count_delta
from morph.model.layers.embedding import ExtrapolatableCountEmbedding


class FourierAtomCountDiTBackboneWithHeads(ConditionalDiTBackboneWithHeads):
    """Embed signed deltas with Fourier features and a zero-initialized gate.

    The count embedding accepts signed inputs without clipping. With scale
    128, its scalar feature is exactly the old symmetric [-128, 128] scalar
    encoder's normalization. ``delta_hidden_dim`` is the Fourier feature width.
    The gate preserves pretrained predictions at initialization; the encoder
    and gate remain trainable alongside LoRA when the base model is frozen.
    """

    def __init__(
        self,
        *args,
        delta_scale=128.0,
        delta_hidden_dim=256,
        delta_min_period=512.0,
        delta_max_period=4096.0,
        **kwargs,
    ):
        # Select the graph port here as well as in the preset so saved configs
        # from earlier Fourier runs still construct the current architecture.
        backbone_args = deepcopy(kwargs["backbone_model_args"])
        backbone_args["_target_"] = (
            "morph.model.backbones.conditional_dit.ConditionalDiTBackbone"
        )
        backbone_args["use_graph_cond"] = True
        kwargs["backbone_model_args"] = backbone_args
        super().__init__(*args, **kwargs)
        self.delta_embedding = ExtrapolatableCountEmbedding(
            embedding_dim=delta_hidden_dim,
            out_dim=int(kwargs["backbone_model_args"]["d_model"]),
            min_period=delta_min_period,
            max_period=delta_max_period,
            max_count=delta_scale,
        )

    def forward(self, mols_t, t, max_seqlen=None):
        """Recompute the signed delta and supply the graph-conditioning port."""
        return super().forward(
            mols_t,
            t,
            max_seqlen=max_seqlen,
            graph_cond=self.delta_embedding(atom_count_delta(mols_t)),
        )

    def _load_from_state_dict(self, state_dict, prefix, *args):
        """Load earlier Fourier weights into the shared conditional backbone."""
        legacy_gate = prefix + "graph_gate"
        if legacy_gate in state_dict:
            backbone_prefix = prefix + "backbone."
            legacy_backbone = not any(
                key.startswith(backbone_prefix + "dit.") for key in state_dict
            )
            for key in list(state_dict):
                if key == legacy_gate:
                    destination = backbone_prefix + "graph_gate"
                elif legacy_backbone and key.startswith(backbone_prefix):
                    destination = backbone_prefix + "dit." + key[len(backbone_prefix) :]
                else:
                    continue
                if destination in state_dict:
                    raise RuntimeError(
                        f"Conflicting Fourier checkpoint keys: {key}, {destination}"
                    )
                state_dict[destination] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args)

    def freeze_pretrained(self):
        """Keep the Fourier projection and graph gate trainable with LoRA."""
        before = sum(p.numel() for p in self.parameters() if p.requires_grad)
        super().freeze_pretrained()
        self.delta_embedding.requires_grad_(True)
        return before - sum(p.numel() for p in self.parameters() if p.requires_grad)
