from __future__ import annotations

import numpy as np

from ds003029_eda.utils.class_balance import compute_class_weights


def test_compute_class_weights_ignores_boundary_label() -> None:
    weights = compute_class_weights(np.array([0, 0, 0, 1, 1, -1], dtype=int))

    assert set(weights) == {0, 1}
    assert weights[0] == 5.0 / (2.0 * 3.0)
    assert weights[1] == 5.0 / (2.0 * 2.0)