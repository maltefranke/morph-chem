"""Molecular objectives shared by standalone and external trainers.

``FlowLossMixin.loss_from_predictions`` assembles the complete objective from
existing head outputs. ``functional`` contains the individual mathematical
losses; records, weighting, and aggregation live in
``morph.utils.loss_accumulation``.
"""

from morph.model.losses.flow import FlowLossMixin

__all__ = ["FlowLossMixin"]
