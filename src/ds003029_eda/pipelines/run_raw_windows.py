from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.io import load_mne
from ..paths import WorkspacePaths, get_paths


@dataclass(frozen=True)
class RawWindowExportConfig:
    artifact_subdir: str = "data_processing_v2"
    output_subdir: str = "raw_folds"
    overwrite: bool = False
    window_sec: float = 2.0
    output_dtype: str = "float16"


def _load_preprocessed_cache_map(artifact_root: Path) -> dict[str, Path]:
    summary_path = artifact_root / "preprocess_run_summary.csv"
    summary_df = pd.read_csv(summary_path)
    return {
        str(row.base): Path(str(row.cache_path))
        for row in summary_df.itertuples(index=False)
        if bool(row.preprocess_ok)
    }


def _normalize_output_dtype(output_dtype: str) -> np.dtype:
    normalized = str(output_dtype).strip().lower()
    if normalized == "float16":
        return np.dtype(np.float16)
    if normalized == "float32":
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported raw export dtype '{output_dtype}'. Expected float16 or float32.")


def _pad_raw_window(
    window: np.ndarray,
    max_channels: int,
    n_samples: int,
    *,
    output_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    padded = np.zeros((max_channels, n_samples), dtype=output_dtype)
    mask = np.zeros((max_channels,), dtype=bool)
    n_channels = min(max_channels, window.shape[0])
    n_time = min(n_samples, window.shape[1])
    padded[:n_channels, :n_time] = window[:n_channels, :n_time].astype(output_dtype, copy=False)
    mask[:n_channels] = True
    return padded, mask


def _extract_split_raw_arrays(
    *,
    index_df: pd.DataFrame,
    cache_map: dict[str, Path],
    max_channels: int,
    n_samples: int,
    paths: WorkspacePaths,
    output_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mne = load_mne(paths)

    n_rows = int(len(index_df))
    x_raw = np.zeros((n_rows, max_channels, n_samples), dtype=output_dtype)
    x_raw_mask = np.zeros((n_rows, max_channels), dtype=bool)
    y = index_df["y"].to_numpy(dtype=np.int8, copy=True) if "y" in index_df.columns else np.zeros((n_rows,), dtype=np.int8)

    for base, group in index_df.groupby("base", sort=False):
        cache_path = cache_map.get(str(base))
        if cache_path is None or not cache_path.exists():
            raise FileNotFoundError(f"Missing preprocessed cache for base {base}: {cache_path}")

        raw = mne.io.read_raw_fif(cache_path, preload=True, verbose="ERROR")
        bads = set(raw.info.get("bads", []))
        good_indices = [idx for idx, channel in enumerate(raw.ch_names) if channel not in bads]
        data = raw.get_data(picks=good_indices)
        sfreq = float(raw.info["sfreq"])

        for row_idx, row in group.iterrows():
            start_sample = int(round(float(row.t_start_s) * sfreq))
            stop_sample = int(round(float(row.t_stop_s) * sfreq))
            window = data[:, start_sample:stop_sample]
            padded, mask = _pad_raw_window(
                window,
                max_channels=max_channels,
                n_samples=n_samples,
                output_dtype=output_dtype,
            )
            x_raw[int(row_idx)] = padded
            x_raw_mask[int(row_idx)] = mask

    return x_raw, x_raw_mask, y


def _load_existing_manifest(export_dir: Path) -> dict[str, object] | None:
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def run_raw_window_export(
    *,
    paths: WorkspacePaths | None = None,
    config: RawWindowExportConfig | None = None,
) -> tuple[pd.DataFrame, Path]:
    paths = paths or get_paths()
    config = config or RawWindowExportConfig()
    artifact_root = paths.outputs_dir / config.artifact_subdir
    folds_root = artifact_root / "folds"
    output_root = artifact_root / config.output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    output_dtype = _normalize_output_dtype(config.output_dtype)

    cache_map = _load_preprocessed_cache_map(artifact_root)
    manifest_rows: list[dict[str, object]] = []

    for fold_dir in sorted(path for path in folds_root.iterdir() if path.is_dir() and path.name.startswith("fold_")):
        fold_manifest = json.loads((fold_dir / "manifest.json").read_text(encoding="utf-8"))
        max_channels = int(fold_manifest["global_max_channels"])
        target_samples = int(round(config.window_sec * 256.0))
        export_dir = output_root / fold_dir.name
        export_dir.mkdir(parents=True, exist_ok=True)
        train_output_path = export_dir / "raw_train_dataset.npz"
        test_output_path = export_dir / "raw_test_dataset.npz"

        if not config.overwrite and train_output_path.exists() and test_output_path.exists():
            existing_manifest = _load_existing_manifest(export_dir)
            manifest_rows.append(
                {
                    "fold_id": fold_dir.name,
                    "max_channels": max_channels,
                    "n_samples": target_samples,
                    "output_dtype": str((existing_manifest or {}).get("output_dtype", output_dtype.name)),
                    "resumed": True,
                    "output_dir": export_dir.as_posix(),
                }
            )
            continue

        for split, output_path in (("train", train_output_path), ("test", test_output_path)):
            index_df = pd.read_csv(fold_dir / f"{split}_index.csv")
            x_raw, x_raw_mask, y = _extract_split_raw_arrays(
                index_df=index_df,
                cache_map=cache_map,
                max_channels=max_channels,
                n_samples=target_samples,
                paths=paths,
                output_dtype=output_dtype,
            )
            np.savez_compressed(
                output_path,
                x_raw=x_raw,
                x_raw_mask=x_raw_mask,
                y=y,
            )

        manifest_rows.append(
            {
                "fold_id": fold_dir.name,
                "max_channels": max_channels,
                "n_samples": target_samples,
                "output_dtype": output_dtype.name,
                "resumed": False,
                "output_dir": export_dir.as_posix(),
            }
        )

        (export_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "fold_id": fold_dir.name,
                    "max_channels": max_channels,
                    "n_samples": target_samples,
                    "output_dtype": output_dtype.name,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output_root / "raw_fold_manifest.csv", index=False)
    return manifest_df, output_root