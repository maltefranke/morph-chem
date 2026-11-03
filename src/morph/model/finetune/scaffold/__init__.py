"""Scaffold-decoration model and Lightning module."""

from morph.model.finetune.scaffold.scaffold import (
    LightningModuleRatesScaffoldFinetune,
)
from morph.model.finetune.scaffold.scaffold_context import (
    ScaffoldConditionalDiTBackboneWithHeads,
)

__all__ = [
    "LightningModuleRatesScaffoldFinetune",
    "ScaffoldConditionalDiTBackboneWithHeads",
]
