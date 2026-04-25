from __future__ import annotations

import numpy as np

from ds003029_eda.experiments.common import compute_binary_metrics


def test_compute_binary_metrics_reports_expected_confusion_values() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_score = np.array([0.1, 0.7, 0.8, 0.2], dtype=float)

    metrics = compute_binary_metrics(y_true, y_score, threshold=0.5)

    assert metrics["tp"] == 1.0
    assert metrics["tn"] == 1.0
    assert metrics["fp"] == 1.0
    assert metrics["fn"] == 1.0
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5