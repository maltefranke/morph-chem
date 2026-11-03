"""Shared fine-tuning LightningModule and pretrained-checkpoint loading.

:class:`LightningModuleRatesFinetune` loads an unconditional DiT checkpoint into
the clean lane at init and (optionally) freezes it, dispatching to the model's own
``freeze_pretrained``. The module-level helpers do the checkpoint surgery.

The unconditional DiT checkpoints store backbone weights at
``state_dict["model.backbone.<param>"]``. When fine-tuning with a
``ConditionalDiTBackboneWithHeads`` (which composes a ``DiTBackbone`` as
``self.backbone.dit``), the matching destination keys are ``backbone.dit.<param>``;
:func:`load_pretrained_into_lightning_model` rewrites keys accordingly and uses
``load_state_dict(strict=False)`` so the cross-attn blocks (zero-gated at init) and
any other new modules keep their initial values.

The task-specific subclass lives in the sibling ``scaffold`` package.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch

from morph.model.lightning.base import LightningModuleRates
from morph.utils.checkpoints import canonical_checkpoint_key

LOGGER = logging.getLogger(__name__)


def _extract_lightning_state_dict(
    ckpt: dict, prefer_ema: bool = False
) -> dict[str, torch.Tensor]:
    """Pull the model state dict out of a raw Lightning checkpoint.

    The Lightning module wraps the actual model under ``model.<...>`` and an
    EMA copy under ``model_ema.<...>``. We return just the keys under
    ``model.`` (or ``model_ema.`` if ``prefer_ema``), with that prefix
    stripped so they line up with ``self.model.load_state_dict`` callers.
    """
    sd = ckpt.get("state_dict", ckpt)
    src_prefix = "model_ema." if prefer_ema else "model."
    fallback = "model." if prefer_ema else "model_ema."

    matches = {k: v for k, v in sd.items() if k.startswith(src_prefix)}
    if not matches:
        matches = {k: v for k, v in sd.items() if k.startswith(fallback)}
        if not matches:
            raise ValueError(
                "No keys starting with 'model.' or 'model_ema.' in checkpoint."
            )
        src_prefix = fallback

    # Strip the outer wrapper and any torch.compile() ``_orig_mod`` component.
    stripped: dict[str, torch.Tensor] = {}
    for k, v in matches.items():
        name = canonical_checkpoint_key(k[len(src_prefix) :])
        stripped[name] = v
    LOGGER.info(
        "Extracted %d params from checkpoint under '%s'.", len(stripped), src_prefix
    )
    return stripped


def _retarget_backbone_keys_for_conditional(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rewrite ``backbone.<param>`` → ``backbone.dit.<param>``.

    Used when loading a vanilla DiT checkpoint into a ConditionalDiT
    backbone, which holds the DiT under ``self.backbone.dit``.
    """
    pattern = re.compile(r"^backbone\.(?!dit\.)(.+)")
    retargeted = {}
    for key, value in state_dict.items():
        match = pattern.match(key)
        if match:
            retargeted[f"backbone.dit.{match.group(1)}"] = value
        else:
            retargeted[key] = value
    return retargeted


def load_pretrained_into_lightning_model(
    model: torch.nn.Module,
    ckpt_path: str | Path,
    *,
    prefer_ema: bool = False,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load a pretrained checkpoint into a Lightning model's ``self.model``.

    Automatically retargets vanilla-DiT keys to the ConditionalDiT layout when
    necessary. Returns ``(missing, unexpected)`` from ``load_state_dict``.

    Args:
        model: the inner model (``LightningModuleRates.model``), NOT the
            outer Lightning module.
        ckpt_path: Path to the Lightning checkpoint (``.ckpt`` file).
        prefer_ema: If True, load from ``model_ema.*`` keys (typically what
            you want for fine-tuning since EMA weights are smoother).
        strict: Forwarded to ``load_state_dict``. Default ``False`` because
            new modules (cross-attn blocks, conditioning gates/encoders) won't
            exist in the source checkpoint.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    LOGGER.info(
        "Loading pretrained weights from %s (prefer_ema=%s)", ckpt_path, prefer_ema
    )
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _extract_lightning_state_dict(raw, prefer_ema=prefer_ema)

    # Retarget ONLY a vanilla-DiT source (keys ``backbone.<p>``) into the
    # ConditionalDiT layout (``backbone.dit.<p>``). An already-conditional source
    # (its own ``backbone.dit.*`` + ``backbone.cross_attn_blocks.*`` +
    # ``backbone.graph_gate``) must load as-is: retargeting would mis-move the
    # cross-attn adapter and gates under ``backbone.dit.*``, where the destination
    # has no such params, so they drop under ``strict=False`` and the trained
    # adapter silently resets to init. Guard on the SOURCE, not just the dest.
    dest_is_conditional = any(
        name.startswith("backbone.dit.") for name, _ in model.named_parameters()
    )
    source_is_conditional = any(k.startswith("backbone.dit.") for k in sd)
    if dest_is_conditional and not source_is_conditional:
        sd = _retarget_backbone_keys_for_conditional(sd)

    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        LOGGER.info(
            "Pretrained load: %d missing keys (expected for new modules like "
            "cross_attn_blocks). First few: %s",
            len(missing),
            missing[:5],
        )
    if unexpected:
        LOGGER.warning(
            "Pretrained load: %d unexpected keys (may indicate arch mismatch). "
            "First few: %s",
            len(unexpected),
            unexpected[:5],
        )
    return missing, unexpected


class LightningModuleRatesFinetune(LightningModuleRates):
    """Fine-tuning variant that loads pretrained DiT weights at init.

    The shared Lightning module delegates its Muon/AuxAdam behavior to the
    optimizer mixin configured by ``configs/model/backbones/dit.yaml``.
    ``freeze_output_heads`` defaults to ``freeze_pretrained_backbone``; an
    explicit boolean controls head freezing independently of the shared features.
    Freeze policies also apply without ``pretrained_ckpt``, including when
    Lightning restores a complete fine-tuning checkpoint for resume.
    """

    def __init__(
        self,
        *args,
        pretrained_ckpt: str | None = None,
        pretrained_prefer_ema: bool = True,
        freeze_pretrained_backbone: bool = False,
        freeze_output_heads: bool | None = None,
        freeze_keep_positions: bool = False,
        freeze_keep_charges: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._freeze_keep_positions = freeze_keep_positions
        self._freeze_keep_charges = freeze_keep_charges

        if pretrained_ckpt:
            load_pretrained_into_lightning_model(
                self.model,
                pretrained_ckpt,
                prefer_ema=pretrained_prefer_ema,
            )
            # Re-sync the EMA copy so it starts from the loaded weights,
            # not from the freshly instantiated (random) ones.
            self.model_ema.load_state_dict(self.model.state_dict())

        if freeze_pretrained_backbone:
            # Each BackboneWithHeads subclass owns its freeze policy via
            # ``freeze_pretrained()`` (no fragile name-prefix matching); a
            # subclass may extend it to freeze its own encoder body too.
            frozen = self.model.freeze_pretrained()
            self.model_ema.freeze_pretrained()
            LOGGER.info(
                "Froze %s params via %s.freeze_pretrained().",
                frozen,
                type(self.model).__name__,
            )
        # The EMA copy is never optimized, including any LoRA matrices
        # that the model-side freeze policy deliberately re-enabled.
        self.model_ema.requires_grad_(False)

        if freeze_output_heads is None:
            freeze_output_heads = freeze_pretrained_backbone
        changed = self.model.set_output_heads_trainable(not freeze_output_heads)
        LOGGER.info(
            "%s %s output-head params, including coordinate readout.",
            "Froze" if freeze_output_heads else "Unfroze",
            changed,
        )

    def _apply_inference_charge_mask(self, mol_t, c_pred):
        """Keep input formal charges when the task treats them as constraints."""
        keep_mask = getattr(mol_t, "keep_mask", None)
        if not self._freeze_keep_charges or keep_mask is None:
            return c_pred
        return torch.where(keep_mask.bool(), mol_t.c, c_pred)

    def _apply_inference_edit_masks(
        self,
        mol_t,
        x1_pred,
        do_sub_a_probs,
        do_sub_e_probs,
        do_del_probs,
        num_ins_pred,
    ):
        """Force-fix the kept atoms & their internal edges at inference.

        Zeroes the substitution / deletion probabilities for the kept-atom set
        (and sub-e for kept-kept edges) *before* the integrator, so kept-atom types
        and bonds cannot drift during sampling — true inpainting. The keep set is
        the ``keep_mask`` node-property port, which every edit task reuses as
        its generic keep mask (Bemis-Murcko scaffold, grown fragment, linker
        endpoints, …), not only literal scaffolds. ``num_ins_pred`` is left
        untouched so decorations can still grow from kept-atom seeds. With
        ``freeze_keep_positions``, the clean coordinate prediction is pinned to
        the current kept coordinates, giving them zero Euler velocity. A no-op
        when the batch carries no keep mask.
        """
        keep_mask = getattr(mol_t, "keep_mask", None)  # generic keep-set port
        if keep_mask is None:
            return x1_pred, do_sub_a_probs, do_sub_e_probs, do_del_probs, num_ins_pred
        keep_node = keep_mask.bool()
        do_sub_a_probs = do_sub_a_probs.clone()
        do_del_probs = do_del_probs.clone()
        do_sub_a_probs[keep_node] = 0.0
        do_del_probs[keep_node] = 0.0
        keep_edge = keep_node[mol_t.edge_index[0]] & keep_node[mol_t.edge_index[1]]
        do_sub_e_probs = do_sub_e_probs.clone()
        do_sub_e_probs[keep_edge] = 0.0
        if self._freeze_keep_positions:
            x1_pred = torch.where(keep_node[:, None], mol_t.positions, x1_pred)
        return x1_pred, do_sub_a_probs, do_sub_e_probs, do_del_probs, num_ins_pred
