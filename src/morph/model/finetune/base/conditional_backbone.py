"""Template-method base for task-specific conditioning backbones.

The scaffold wrapper extends
:class:`~morph.model.backbones.conditional_dit.ConditionalDiTBackboneWithHeads`
with the *same* ``forward``: take the standard conditioning ports, optionally
populate some of them from task-specific inputs, then delegate to the parent.
Only the "populate" step differs between tasks, so it is the single overridable
hook (:meth:`_build_conditioning`); the (long) signature and the delegation live
here once instead of being copy-pasted into every subclass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from morph.model.backbones.conditional_dit import ConditionalDiTBackboneWithHeads

if TYPE_CHECKING:
    import torch

    from morph.dataset.core.atomic_data import ChemBatch

# Conditioning inputs threaded from ``forward`` into ``super().forward``.
CondPorts = dict[str, "torch.Tensor | None"]


class ConditioningDiTBackboneWithHeads(ConditionalDiTBackboneWithHeads):
    """A :class:`ConditionalDiTBackboneWithHeads` with a task-specific
    conditioning hook run in front of the conditional forward.
    """

    def _build_conditioning(
        self,
        mols_t: ChemBatch,
        cond: CondPorts,
        force_uncond: bool,
    ) -> CondPorts:
        """Populate / adjust the conditioning ports for this task.

        ``cond`` holds the caller-supplied inputs (``cond_tokens``,
        ``cond_token_mask``, ``cond_token_attn_bias``, ``node_cond``,
        ``edge_cond``, ``graph_cond``). Caller-supplied (non-``None``) values
        must be preserved — the inference-time caching hook passes them
        directly. Returns the (possibly updated) dict. Default: no conditioning.
        """
        return cond

    def forward(
        self,
        mols_t: ChemBatch,
        t: torch.Tensor,
        cond_tokens: torch.Tensor | None = None,
        cond_token_mask: torch.Tensor | None = None,
        cond_token_attn_bias: torch.Tensor | None = None,
        node_cond: torch.Tensor | None = None,
        edge_cond: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        force_uncond: bool = False,
        graph_cond: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Encode task context and run the conditional DiT with its heads.

        ``force_uncond=True`` skips automatic cross-attention context encoding
        for this call. Explicit conditioning inputs and scaffold node/edge
        features still apply.
        """
        cond = self._build_conditioning(
            mols_t,
            {
                "cond_tokens": cond_tokens,
                "cond_token_mask": cond_token_mask,
                "cond_token_attn_bias": cond_token_attn_bias,
                "node_cond": node_cond,
                "edge_cond": edge_cond,
                "graph_cond": graph_cond,
            },
            force_uncond,
        )
        return super().forward(
            mols_t,
            t,
            max_seqlen=max_seqlen,
            **cond,
        )
