from __future__ import annotations

import numpy as np


FREQUENCY_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma_low": (30.0, 80.0),
    "gamma_high": (80.0, 120.0),
}

FREQUENCY_FEATURE_NAMES = (
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_low_power",
    "gamma_high_power",
    "spectral_entropy",
    "peak_frequency",
)

AGGREGATE_SUFFIXES = ("mean", "std", "max")


def compute_frequency_domain_features(window: np.ndarray, sfreq: float) -> dict[str, np.ndarray]:
    values = np.asarray(window, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected window with shape (channels, samples).")
    if values.shape[1] < 2:
        zeros = np.zeros(values.shape[0], dtype=float)
        return {name: zeros.copy() for name in FREQUENCY_FEATURE_NAMES}

    centered = values - np.nanmean(values, axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    freqs = np.fft.rfftfreq(centered.shape[1], d=1.0 / sfreq)

    usable_mask = (freqs >= 0.5) & (freqs <= 120.0)
    usable_spectrum = spectrum[:, usable_mask]
    usable_freqs = freqs[usable_mask]
    total_power = np.sum(usable_spectrum, axis=1)

    outputs: dict[str, np.ndarray] = {}
    for band_name, (low_hz, high_hz) in FREQUENCY_BANDS.items():
        band_mask = (usable_freqs >= low_hz) & (usable_freqs < high_hz)
        if not np.any(band_mask):
            outputs[f"{band_name}_power"] = np.zeros(values.shape[0], dtype=float)
            continue
        outputs[f"{band_name}_power"] = np.sum(usable_spectrum[:, band_mask], axis=1)

    normalized_power = np.divide(
        usable_spectrum,
        total_power[:, None],
        out=np.zeros_like(usable_spectrum),
        where=total_power[:, None] > 0,
    )
    entropy = -np.sum(normalized_power * np.log(normalized_power + 1e-12), axis=1)
    if usable_spectrum.shape[1] > 1:
        entropy = entropy / np.log(float(usable_spectrum.shape[1]))
    peak_indices = np.argmax(usable_spectrum, axis=1) if usable_spectrum.shape[1] else np.zeros(values.shape[0], dtype=int)

    outputs["spectral_entropy"] = entropy.astype(float)
    outputs["peak_frequency"] = usable_freqs[peak_indices].astype(float) if usable_freqs.size else np.zeros(values.shape[0], dtype=float)
    return outputs