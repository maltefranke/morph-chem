"""Shared OmegaConf resolver registration.

Importing this module registers Morph's custom resolvers. Registration is
idempotent so entry points and libraries can safely import it in any order.
"""

from omegaconf import OmegaConf


def register_resolvers() -> None:
    """Register the resolvers used by Morph's Hydra configurations."""
    resolvers = {
        "oc.eval": eval,
        "len": len,
        "if": lambda condition, true, false: true if condition else false,
        "eq": lambda left, right: left == right,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


register_resolvers()
