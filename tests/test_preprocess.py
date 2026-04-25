from __future__ import annotations

import numpy as np
import pytest

from ds003029_eda.data.preprocess import PreprocessConfig, preprocess_raw


mne = pytest.importorskip("mne")


def _power_around(signal: np.ndarray, sfreq: float, center_hz: float, width_hz: float = 1.0) -> float:
    centered = signal - float(np.mean(signal))
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / sfreq)
    mask = np.abs(freqs - center_hz) <= width_hz
    return float(np.sum(spectrum[mask]))


def test_preprocess_attenuates_line_noise_and_resamples() -> None:
    sfreq = 1000.0
    duration_s = 10.0
    times = np.arange(int(sfreq * duration_s)) / sfreq
    drift = 3.0 * np.sin(2.0 * np.pi * 0.1 * times)
    line = 1.5 * np.sin(2.0 * np.pi * 60.0 * times)
    rhythm = 0.4 * np.sin(2.0 * np.pi * 10.0 * times)

    data = np.vstack(
        [
            drift + line + rhythm,
            0.5 * drift + 0.7 * line + 0.2 * rhythm,
        ]
    )
    raw = mne.io.RawArray(data, mne.create_info(["E1", "E2"], sfreq=sfreq, ch_types=["ecog", "ecog"]))

    preprocessed_raw, _, _ = preprocess_raw(
        raw,
        config=PreprocessConfig(target_sfreq=256.0, notch_harmonics=1),
        line_freq=60.0,
    )

    before_line_power = _power_around(data[0], sfreq, 60.0)
    after_data = preprocessed_raw.get_data()
    after_line_power = _power_around(after_data[0], float(preprocessed_raw.info["sfreq"]), 60.0)

    assert float(preprocessed_raw.info["sfreq"]) == 256.0
    assert after_line_power < before_line_power * 0.05
    assert abs(float(np.mean(after_data[0]))) < abs(float(np.mean(data[0])))


def test_preprocess_skips_average_reference_when_only_one_good_channel_remains() -> None:
    sfreq = 512.0
    times = np.arange(int(sfreq * 4.0)) / sfreq
    signal = np.sin(2.0 * np.pi * 8.0 * times) + 0.2 * np.sin(2.0 * np.pi * 40.0 * times)
    raw = mne.io.RawArray(signal[np.newaxis, :], mne.create_info(["E1"], sfreq=sfreq, ch_types=["ecog"]))

    preprocessed_raw, _, meta = preprocess_raw(
        raw,
        config=PreprocessConfig(target_sfreq=256.0, notch_harmonics=0),
        line_freq=None,
    )

    assert meta["n_channels_good"] == 1
    assert not np.allclose(preprocessed_raw.get_data()[0], 0.0)