"""learning package: color-image -> layered editable SVG node hierarchy.

The public API lives in :mod:`learning.learner` (importable as
``from learning import learner`` — the submodule is auto-imported on attribute
access). Supporting sub-packages (`classifiers`, `composites`, `shape_registry`,
plus the metrics/detectors/generators/reconstructors/renderers layers) are also
importable beneath `learning`.

The learner submodule is intentionally NOT imported eagerly here, to avoid a
double-import RuntimeWarning when the CLI runs ``python -m learning.learner``.
"""

__all__ = [
    "learner",
    "classifiers",
    "composites",
    "shape_registry",
]
