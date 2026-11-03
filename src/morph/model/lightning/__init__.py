"""PyTorch-Lightning training modules.

:mod:`base` holds :class:`LightningModuleRates`, assembled from per-concern
mixins (:mod:`_setup`, :mod:`_training`, :mod:`_validation`, :mod:`_sampling`,
:mod:`_optimizer`). The repository's models all use the DiT backbone.
Task-specific finetuning modules live in :mod:`morph.model.finetune`.
"""
