from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .channel_qc import ChannelQcConfig, detect_bad_channels


@dataclass(frozen=True)
class PreprocessConfig:
    target_sfreq: float = 256.0
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 120.0
    notch_harmonics: int = 3
    filter_order: int = 4
    filter_type: str = "butter"
    reference_mode: str = "average"
    qc: ChannelQcConfig = field(default_factory=ChannelQcConfig)


def resolve_notch_frequencies(
    line_freq: float | None,
    sfreq: float,
    *,
    max_hz: float,
    n_harmonics: int,
) -> list[float]:
    if line_freq is None or line_freq <= 0:
        return []

    nyquist = sfreq / 2.0
    freqs: list[float] = []
    for harmonic_idx in range(1, n_harmonics + 1):
        harmonic = float(line_freq) * harmonic_idx
        if harmonic >= nyquist or harmonic > max_hz:
            continue
        freqs.append(harmonic)
    return freqs


def _apply_common_average_reference(raw) -> None:
    if not hasattr(raw, "_data"):
        raise ValueError("Raw object must be preloaded before average referencing.")

    bads = set(raw.info.get("bads", []))
    good_indices = [idx for idx, channel in enumerate(raw.ch_names) if channel not in bads]
    if len(good_indices) < 2:
        return

    reference = np.nanmean(raw._data[good_indices, :], axis=0, keepdims=True)
    raw._data = raw._data - reference


def _center_channels(raw) -> None:
    if not hasattr(raw, "_data"):
        raise ValueError("Raw object must be preloaded before centering.")

    channel_means = np.nanmean(raw._data, axis=1, keepdims=True)
    raw._data = raw._data - channel_means


def _validate_preprocessed_raw(raw, config: PreprocessConfig) -> None:
    if not np.isclose(float(raw.info["sfreq"]), config.target_sfreq, rtol=0.0, atol=1e-6):
        raise ValueError(f"Expected sfreq={config.target_sfreq}, got {raw.info['sfreq']}")

    data = raw.get_data()
    if not np.isfinite(data).all():
        raise ValueError("Preprocessed signal contains NaN or Inf values.")

    bads = set(raw.info.get("bads", []))
    n_good_channels = sum(channel not in bads for channel in raw.ch_names)
    if n_good_channels <= 0:
        raise ValueError("No good channels remain after preprocessing.")

    all_zero = [
        channel
        for channel, samples in zip(raw.ch_names, data)
        if channel not in bads and np.allclose(samples, 0.0)
    ]
    if all_zero:
        raise ValueError(f"Good channels became all-zero after preprocessing: {all_zero}")


def preprocess_raw(
    raw,
    *,
    config: PreprocessConfig | None = None,
    line_freq: float | None = None,
) -> tuple[object, pd.DataFrame, dict[str, object]]:
    config = config or PreprocessConfig()

    work_raw = raw.copy().load_data()
    initial_bads = sorted(set(work_raw.info.get("bads", [])))
    detected_bads, qc_df = detect_bad_channels(work_raw, config=config.qc, line_freq=line_freq)
    final_bads = sorted(set(initial_bads).union(set(detected_bads)))
    work_raw.info["bads"] = final_bads

    notch_freqs = resolve_notch_frequencies(
        line_freq,
        float(work_raw.info["sfreq"]),
        max_hz=config.bandpass_high_hz,
        n_harmonics=config.notch_harmonics,
    )
    if notch_freqs:
        work_raw.notch_filter(freqs=notch_freqs, verbose="ERROR")

    work_raw.filter(
        l_freq=config.bandpass_low_hz,
        h_freq=config.bandpass_high_hz,
        method="iir",
        iir_params={"order": config.filter_order, "ftype": config.filter_type},
        verbose="ERROR",
    )

    if not np.isclose(float(work_raw.info["sfreq"]), config.target_sfreq, rtol=0.0, atol=1e-6):
        work_raw.resample(config.target_sfreq, npad="auto")

    if config.reference_mode.lower() == "average":
        _apply_common_average_reference(work_raw)

    _center_channels(work_raw)

    _validate_preprocessed_raw(work_raw, config)

    qc_df = qc_df.copy()
    qc_df["preexisting_bad"] = qc_df["channel"].isin(initial_bads)
    qc_df["final_bad"] = qc_df["channel"].isin(final_bads)

    metadata = {
        "sfreq_in": float(raw.info["sfreq"]),
        "sfreq_out": float(work_raw.info["sfreq"]),
        "n_channels_in": int(len(raw.ch_names)),
        "n_channels_out": int(len(work_raw.ch_names)),
        "n_channels_good": int(len(work_raw.ch_names) - len(final_bads)),
        "initial_bads": initial_bads,
        "detected_bads": sorted(set(detected_bads)),
        "final_bads": final_bads,
        "line_freq": float(line_freq) if line_freq is not None else None,
        "notch_freqs": notch_freqs,
        "bandpass": [config.bandpass_low_hz, config.bandpass_high_hz],
        "reference_mode": config.reference_mode,
    }
    return work_raw, qc_df, metadata