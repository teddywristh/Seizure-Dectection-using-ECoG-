from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_torch_module():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "FoldDataset requires PyTorch. Install torch before using ds003029_eda.datasets.FoldDataset."
        ) from exc
    return torch


class FoldDataset:
    """Map-style dataset for fold train/test_dataset.npz artifacts.

    This wrapper is intentionally light: it exposes the tensors already produced
    by the v2 pipeline and can be passed directly to torch.utils.data.DataLoader.
    """

    VALID_MODES = {"agg", "channel", "both"}

    def __init__(self, npz_path: str | Path, mode: str = "agg") -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Expected one of {sorted(self.VALID_MODES)}.")

        payload = np.load(Path(npz_path), allow_pickle=False)
        torch = _load_torch_module()

        self.x_agg = torch.from_numpy(payload["x_agg"].astype(np.float32))
        self.x_channel = torch.from_numpy(payload["x_channel"].astype(np.float32))
        self.mask = torch.from_numpy(payload["x_channel_mask"].astype(bool))
        self.y = torch.from_numpy(payload["y"].astype(np.int64))
        self.mode = mode

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        if self.mode == "agg":
            return self.x_agg[idx], self.y[idx]
        if self.mode == "channel":
            return (self.x_channel[idx], self.mask[idx]), self.y[idx]
        return (self.x_agg[idx], self.x_channel[idx], self.mask[idx]), self.y[idx]