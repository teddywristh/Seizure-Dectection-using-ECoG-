from __future__ import annotations

import numpy as np

from ds003029_eda.pipelines.run_raw_windows import _normalize_output_dtype, _pad_raw_window


def test_pad_raw_window_preserves_signal_and_builds_mask() -> None:
    window = np.arange(12, dtype=float).reshape(3, 4)
    padded, mask = _pad_raw_window(window, max_channels=5, n_samples=6, output_dtype=_normalize_output_dtype("float16"))

    assert padded.shape == (5, 6)
    assert padded.dtype == np.float16
    assert mask.tolist() == [True, True, True, False, False]
    assert np.allclose(padded[:3, :4], window)
    assert np.allclose(padded[3:], 0.0)