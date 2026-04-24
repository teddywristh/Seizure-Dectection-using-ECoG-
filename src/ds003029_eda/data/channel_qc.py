from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChannelQcConfig:
    flat_thresh: float = 1e-7
    var_z_thresh: float = 5.0
    line_noise_ratio: float = 3.0
    line_noise_z_thresh: float = 3.0
    harmonic_tolerance_hz: float = 2.0


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    zscores = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if np.sum(finite) < 2:
        return np.nan_to_num(zscores, nan=0.0)

    valid = values[finite]
    median = float(np.nanmedian(valid))
    mad = float(np.nanmedian(np.abs(valid - median)))
    if mad > 0:
        zscores[finite] = 0.6744897501960817 * (valid - median) / mad
        return zscores

    std = float(np.nanstd(valid))
    if std > 0:
        zscores[finite] = (valid - float(np.nanmean(valid))) / std
    else:
        zscores[finite] = 0.0
    return zscores


def _estimate_line_noise_ratio(
    signal: np.ndarray,
    sfreq: float,
    line_freq: float | None,
    *,
    harmonic_tolerance_hz: float,
) -> float:
    if line_freq is None or line_freq <= 0:
        return float("nan")

    centered = np.asarray(signal, dtype=float) - float(np.nanmean(signal))
    if centered.size < 8:
        return float("nan")

    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / sfreq)
    nyquist = sfreq / 2.0
    ratios: list[float] = []

    harmonic = line_freq
    while harmonic < nyquist:
        center_idx = int(np.argmin(np.abs(freqs - harmonic)))
        lower = max(0, int(np.searchsorted(freqs, harmonic - harmonic_tolerance_hz, side="left")))
        upper = min(len(freqs), int(np.searchsorted(freqs, harmonic + harmonic_tolerance_hz, side="right")))
        if upper - lower <= 2:
            harmonic += line_freq
            continue

        neighborhood = spectrum[lower:upper]
        neighborhood_indices = np.arange(lower, upper)
        background = neighborhood[neighborhood_indices != center_idx]
        background = background[np.isfinite(background)]
        if background.size == 0:
            harmonic += line_freq
            continue

        baseline = float(np.nanmedian(background))
        baseline = baseline if baseline > 0 else 1e-12
        ratios.append(float(spectrum[center_idx] / baseline))
        harmonic += line_freq

    if not ratios:
        return float("nan")
    return float(np.nanmax(ratios))


def detect_bad_channels(
    raw,
    config: ChannelQcConfig | None = None,
    *,
    line_freq: float | None = None,
) -> tuple[list[str], pd.DataFrame]:
    config = config or ChannelQcConfig()
    data = raw.get_data()
    if data.size == 0:
        empty = pd.DataFrame(columns=["channel", "variance", "variance_zscore", "peak_to_peak", "line_noise_ratio", "is_flat", "is_high_variance", "is_line_noise", "is_bad"])
        return [], empty

    variance = np.nanvar(data, axis=1)
    variance_std = float(np.nanstd(variance))
    if variance_std > 0:
        variance_z = (variance - float(np.nanmean(variance))) / variance_std
    else:
        variance_z = np.zeros_like(variance)
    peak_to_peak = np.nanmax(data, axis=1) - np.nanmin(data, axis=1)
    line_ratios = np.array(
        [
            _estimate_line_noise_ratio(channel, float(raw.info["sfreq"]), line_freq, harmonic_tolerance_hz=config.harmonic_tolerance_hz)
            for channel in data
        ],
        dtype=float,
    )
    safe_line_ratios = np.where(np.isfinite(line_ratios) & (line_ratios > 0), line_ratios, np.nan)
    line_ratio_z = _robust_zscore(np.log10(safe_line_ratios))

    is_flat = peak_to_peak <= config.flat_thresh
    is_high_variance = variance_z >= config.var_z_thresh
    is_line_noise = (np.nan_to_num(line_ratios, nan=0.0) >= config.line_noise_ratio) & (
        np.nan_to_num(line_ratio_z, nan=0.0) >= config.line_noise_z_thresh
    )
    is_bad = is_flat | is_high_variance | is_line_noise

    qc_df = pd.DataFrame(
        {
            "channel": raw.ch_names,
            "variance": variance.astype(float),
            "variance_zscore": variance_z.astype(float),
            "peak_to_peak": peak_to_peak.astype(float),
            "line_noise_ratio": line_ratios.astype(float),
            "line_noise_zscore": line_ratio_z.astype(float),
            "is_flat": is_flat.astype(bool),
            "is_high_variance": is_high_variance.astype(bool),
            "is_line_noise": is_line_noise.astype(bool),
            "is_bad": is_bad.astype(bool),
        }
    )
    bads = qc_df.loc[qc_df["is_bad"], "channel"].astype(str).tolist()
    return bads, qc_df