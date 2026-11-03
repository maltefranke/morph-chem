"""Scaffold membership conditions nodes/edges; the protein supplies cross-attention."""

from morph.model.finetune.base import KeepEdgeCond, KeepNodeCond
from morph.model.finetune.pocket.pocket_context import (
    PocketConditionalDiTBackboneWithHeads,
)


class ScaffoldPocketConditionalDiTBackboneWithHeads(
    PocketConditionalDiTBackboneWithHeads
):
    """Compose scaffold constraints with protein context and empty replay rows.

    Coordinate pinning and topology/charge constraints use the shared scaffold
    Lightning module's inference masks. All conditioning gates start at zero,
    preserving the pretrained model's predictions before fine-tuning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bb = self.backbone
        self.node_cond_encoder = KeepNodeCond(bb.d_model) if bb.use_node_cond else None
        self.edge_cond_encoder = (
            KeepEdgeCond(bb.in_edge_nf) if bb.use_edge_cond else None
        )

    def _build_conditioning(self, mols_t, cond, force_uncond):
        if getattr(mols_t, "keep_mask", None) is not None:
            if cond["node_cond"] is None and self.node_cond_encoder is not None:
                cond["node_cond"] = self.node_cond_encoder(mols_t)
            if cond["edge_cond"] is None and self.edge_cond_encoder is not None:
                cond["edge_cond"] = self.edge_cond_encoder(mols_t)
        return super()._build_conditioning(mols_t, cond, force_uncond)
