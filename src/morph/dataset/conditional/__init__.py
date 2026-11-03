"""Conditional / finetuning task wrappers.

Each subpackage wraps an unconditional source into a paired / conditioned
dataset for a downstream finetuning task:

* :mod:`morph.dataset.conditional.scaffold` -- bare-scaffold decoration.
* :mod:`morph.dataset.conditional.pocket` -- CrossDocked ligand/pocket pairs.
* :mod:`morph.dataset.conditional.atom_count` -- requested final atom counts.
"""
