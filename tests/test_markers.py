from __future__ import annotations

import pandas as pd

from ds003029_eda.markers import is_offset_trial_type, is_onset_trial_type, pair_intervals, parse_events_df


def test_eeg_sz_end_is_only_offset() -> None:
    assert is_offset_trial_type("eeg sz end")
    assert not is_onset_trial_type("eeg sz end")


def test_pair_intervals_uses_first_offset_after_onset() -> None:
    events_df = pd.DataFrame(
        {
            "onset": [5.0, 10.0, 20.0, 30.0],
            "trial_type": ["sz onset", "offset", "sz onset", "eeg sz end"],
        }
    )
    parsed = parse_events_df(events_df)
    assert pair_intervals(parsed.onset_events, parsed.offset_events) == [(5.0, 10.0), (20.0, 30.0)]