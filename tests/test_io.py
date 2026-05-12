from __future__ import annotations

import pandas as pd
import pytest

from ds003029_eda.data.io import _select_signal_channels


mne = pytest.importorskip("mne")


def test_select_signal_channels_filters_declared_bads_to_selected_signals() -> None:
    raw = mne.io.RawArray(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.5, 0.5, 0.5]],
        mne.create_info(["E1", "E2", "EKG1"], sfreq=256.0, ch_types=["ecog", "ecog", "ecg"]),
    )
    channels_df = pd.DataFrame(
        {
            "name": ["E1", "E2", "EKG1"],
            "type": ["ECOG", "ECOG", "ECG"],
            "status": ["bad", "good", "bad"],
        }
    )

    signal_names, declared_bads = _select_signal_channels(raw, channels_df)

    assert signal_names == ["E1", "E2"]
    assert declared_bads == ["E1"]