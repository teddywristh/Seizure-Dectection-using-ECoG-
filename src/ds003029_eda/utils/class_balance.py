from __future__ import annotations

import numpy as np


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute balanced binary class weights after excluding boundary label -1."""

    values = np.asarray(y)
    filtered = values[values != -1]
    if filtered.size == 0:
        raise ValueError("Cannot compute class weights from an empty label array.")

    classes, counts = np.unique(filtered, return_counts=True)
    total = int(np.sum(counts))
    n_classes = int(len(classes))
    return {int(label): float(total / (n_classes * count)) for label, count in zip(classes, counts)}