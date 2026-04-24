from __future__ import annotations

import numpy as np
import pytest

from ds003029_eda.data.channel_qc import ChannelQcConfig, detect_bad_channels


mne = pytest.importorskip("mne")


def test_detect_bad_channels_flags_line_noise_outliers_not_uniform_mains_contamination() -> None:
    sfreq = 512.0
    times = np.arange(int(sfreq * 8.0)) / sfreq
    base = 0.4 * np.sin(2.0 * np.pi * 8.0 * times)
    common_line = 0.6 * np.sin(2.0 * np.pi * 60.0 * times)
    extreme_line = 3.0 * np.sin(2.0 * np.pi * 60.0 * times)

    data = np.vstack(
        [
            base + common_line,
            base + common_line,
            base + extreme_line,
        ]
    )
    raw = mne.io.RawArray(data, mne.create_info(["E1", "E2", "E3"], sfreq=sfreq, ch_types=["ecog", "ecog", "ecog"]))

    bads, qc_df = detect_bad_channels(
        raw,
        config=ChannelQcConfig(line_noise_ratio=3.0, line_noise_z_thresh=1.0),
        line_freq=60.0,
    )

    assert bads == ["E3"]
    assert qc_df.loc[qc_df["channel"] == "E3", "is_line_noise"].item()
    assert not qc_df.loc[qc_df["channel"] == "E1", "is_line_noise"].item()
    assert not qc_df.loc[qc_df["channel"] == "E2", "is_line_noise"].item()