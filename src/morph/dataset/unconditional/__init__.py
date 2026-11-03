"""Unconditional dataset sources.

Each module ports a raw dataset (download / preprocess into LMDB or an
in-memory PyG dataset) and exposes a ``FlowMatching*`` subclass that
tokenises samples into :class:`~morph.dataset.core.atomic_data.ChemAtomicData`:
GEOM-Drugs (:mod:`geom`), revised QM9 (:mod:`qm9`), materials / MP-20
(:mod:`materials`), tmQM (:mod:`tmqm`) and Vaska's space (:mod:`vaskas`).
"""
