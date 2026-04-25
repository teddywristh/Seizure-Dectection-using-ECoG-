from __future__ import annotations

from ds003029_eda.labels.labeler import label_window


def test_boundary_margin_labeling_matches_plan() -> None:
    intervals = [(10.0, 20.0), (40.0, 45.0)]
    probes = [5.0, 10.2, 15.0, 19.8, 25.0, 39.8, 42.0]
    labels = [label_window(t_mid_s, intervals, margin_sec=0.5) for t_mid_s in probes]
    assert labels == [0, -1, 1, -1, 0, -1, 1]