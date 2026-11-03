"""Task-specific fine-tuning, one subpackage per application.

The ``base`` package contains shared building blocks; ``scaffold`` pairs the
task's Lightning module with its conditional backbone. Public classes are
re-exported at each package boundary for concise Hydra targets.
"""
