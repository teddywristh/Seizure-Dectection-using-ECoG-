from __future__ import annotations

from ds003029_eda.features.windowing import WindowingConfig, compute_window_samples, count_windows, iter_window_slices


def test_window_count_formula() -> None:
    config = WindowingConfig(window_sec=2.0, step_sec=0.5)
    window_samples, step_samples = compute_window_samples(256.0, config)
    n_samples = 4096
    assert count_windows(n_samples, window_samples, step_samples) == ((n_samples - window_samples) // step_samples) + 1


def test_window_slices_are_deterministic() -> None:
    config = WindowingConfig(window_sec=2.0, step_sec=0.5)
    slices = list(iter_window_slices(4096, 256.0, config))
    assert slices[0].start_sample == 0
    assert slices[0].stop_sample == 512
    assert slices[1].start_sample == 128
    assert len(slices) == count_windows(4096, 512, 128)