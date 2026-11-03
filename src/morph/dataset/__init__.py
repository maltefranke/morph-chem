"""morph dataset layer.

Organised into three groups so the dependency direction is legible
(``conditional`` -> ``unconditional`` -> ``core``, never upward):

* :mod:`morph.dataset.core` -- shared infrastructure every dataset builds
  on (the ``ChemAtomicData`` / ``ChemBatch`` containers, vocab,
  representation, the flow-matching dataset wrapper, the base Lightning
  datamodule, preprocessing and property caching).
* :mod:`morph.dataset.unconditional` -- concrete GEOM and QM9 sources,
  each exposing a ``FlowMatching*`` variant.
* :mod:`morph.dataset.conditional` -- scaffold-decoration fine-tuning wrappers.
"""
