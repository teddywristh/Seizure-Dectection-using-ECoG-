from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowingConfig:
    window_sec: float = 2.0
    step_sec: float = 0.5


@dataclass(frozen=True)
class WindowSlice:
    start_sample: int
    stop_sample: int
    start_s: float
    stop_s: float
    mid_s: float


def compute_window_samples(sfreq: float, config: WindowingConfig) -> tuple[int, int]:
    window_samples = int(round(config.window_sec * sfreq))
    step_samples = int(round(config.step_sec * sfreq))
    if window_samples <= 0:
        raise ValueError("window_sec must produce at least one sample.")
    if step_samples <= 0:
        raise ValueError("step_sec must produce at least one sample.")
    return window_samples, step_samples


def count_windows(n_samples: int, window_samples: int, step_samples: int) -> int:
    if n_samples < window_samples:
        return 0
    return ((n_samples - window_samples) // step_samples) + 1


def iter_window_slices(n_samples: int, sfreq: float, config: WindowingConfig):
    window_samples, step_samples = compute_window_samples(sfreq, config)
    total = count_windows(n_samples, window_samples, step_samples)
    for window_idx in range(total):
        start_sample = window_idx * step_samples
        stop_sample = start_sample + window_samples
        start_s = start_sample / sfreq
        stop_s = stop_sample / sfreq
        yield WindowSlice(
            start_sample=start_sample,
            stop_sample=stop_sample,
            start_s=start_s,
            stop_s=stop_s,
            mid_s=(start_s + stop_s) / 2.0,
        )