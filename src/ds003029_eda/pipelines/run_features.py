from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.io import intervals_for_base, load_mne, resolve_artifact_path
from ..data.normalize import fit_array_scaler, save_scalers_json, transform_array
from ..features.freq_domain import FREQUENCY_FEATURE_NAMES, compute_frequency_domain_features
from ..features.time_domain import AGGREGATE_SUFFIXES, TIME_DOMAIN_FEATURE_NAMES, compute_time_domain_features
from ..features.windowing import WindowingConfig, iter_window_slices
from ..labels.labeler import LabelingConfig, build_window_label_frame
from ..paths import WorkspacePaths, get_paths
from ..splits.patient_cv import build_loso_folds, build_subject_table, validate_folds


CHANNEL_FEATURE_NAMES = tuple(list(TIME_DOMAIN_FEATURE_NAMES) + list(FREQUENCY_FEATURE_NAMES))
AGGREGATE_FEATURE_NAMES = tuple(
    [f"agg_mean_{feature_name}" for feature_name in CHANNEL_FEATURE_NAMES]
    + [f"agg_std_{feature_name}" for feature_name in CHANNEL_FEATURE_NAMES]
    + [f"agg_max_{feature_name}" for feature_name in CHANNEL_FEATURE_NAMES]
)


def _portable_artifact_path(path: Path, artifact_root: Path) -> str:
    try:
        return path.relative_to(artifact_root).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass(frozen=True)
class FeaturePipelineConfig:
    artifact_subdir: str = "data_processing_v2"
    overwrite: bool = False
    windowing: WindowingConfig = field(default_factory=WindowingConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    fill_missing_after_scaling: float = 0.0


def _artifact_root(paths: WorkspacePaths, artifact_subdir: str) -> Path:
    return paths.outputs_dir / artifact_subdir


def _safe_stem(base: str) -> str:
    return Path(str(base)).name


def _md5_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_final_bad_channel_map(artifact_root: Path) -> dict[str, list[str]]:
    qc_path = artifact_root / "bad_channels.csv"
    if not qc_path.exists():
        return {}

    qc_df = pd.read_csv(qc_path)
    required_columns = {"base", "channel", "final_bad"}
    if qc_df.empty or not required_columns.issubset(qc_df.columns):
        return {}

    final_bad_df = qc_df[qc_df["final_bad"].astype(bool)].copy()
    if final_bad_df.empty:
        return {}

    return {
        str(base): sorted(set(group["channel"].astype(str)))
        for base, group in final_bad_df.groupby("base", sort=False)
    }


def _aggregate_channel_matrix(channel_feature_matrix: np.ndarray) -> np.ndarray:
    means = np.nanmean(channel_feature_matrix, axis=0)
    stds = np.nanstd(channel_feature_matrix, axis=0)
    maxs = np.nanmax(channel_feature_matrix, axis=0)
    return np.concatenate([means, stds, maxs]).astype(np.float32)


def _extract_run_features(raw, intervals: list[tuple[float, float]], config: FeaturePipelineConfig):
    bads = set(raw.info.get("bads", []))
    good_indices = [idx for idx, channel in enumerate(raw.ch_names) if channel not in bads]
    if not good_indices:
        raise ValueError("Run has no good channels after preprocessing.")

    data = raw.get_data(picks=good_indices)
    channel_names = [raw.ch_names[idx] for idx in good_indices]
    sfreq = float(raw.info["sfreq"])
    slices = list(iter_window_slices(data.shape[1], sfreq, config.windowing))
    if not slices:
        raise ValueError("Run is shorter than the configured window length.")

    starts = np.array([window_slice.start_s for window_slice in slices], dtype=np.float32)
    stops = np.array([window_slice.stop_s for window_slice in slices], dtype=np.float32)
    labels_df = build_window_label_frame(
        starts,
        stops,
        intervals,
        margin_sec=config.labeling.boundary_margin_sec,
    )

    n_windows = len(slices)
    n_channels = len(channel_names)
    x_channel = np.zeros((n_windows, n_channels, len(CHANNEL_FEATURE_NAMES)), dtype=np.float32)
    x_agg = np.zeros((n_windows, len(AGGREGATE_FEATURE_NAMES)), dtype=np.float32)

    for window_idx, window_slice in enumerate(slices):
        window = data[:, window_slice.start_sample : window_slice.stop_sample]
        feature_map = {}
        feature_map.update(compute_time_domain_features(window))
        feature_map.update(compute_frequency_domain_features(window, sfreq))
        channel_matrix = np.stack([feature_map[feature_name] for feature_name in CHANNEL_FEATURE_NAMES], axis=-1).astype(np.float32)
        x_channel[window_idx] = channel_matrix
        x_agg[window_idx] = _aggregate_channel_matrix(channel_matrix)

    return labels_df, channel_names, x_channel, x_agg


def _save_run_tensor_artifacts(
    run_dir: Path,
    *,
    stem: str,
    labels_df: pd.DataFrame,
    channel_names: list[str],
    x_channel: np.ndarray,
    x_agg: np.ndarray,
    base: str,
    subject: str,
) -> tuple[Path, Path]:
    tensor_path = run_dir / f"{stem}_window_tensor.npz"
    index_path = run_dir / f"{stem}_window_index.csv"
    np.savez_compressed(
        tensor_path,
        x_channel=x_channel.astype(np.float32),
        x_agg=x_agg.astype(np.float32),
        y=labels_df["y"].to_numpy(dtype=np.int8),
        t_start_s=labels_df["t_start_s"].to_numpy(dtype=np.float32),
        t_stop_s=labels_df["t_stop_s"].to_numpy(dtype=np.float32),
        t_mid_s=labels_df["t_mid_s"].to_numpy(dtype=np.float32),
        channel_names=np.asarray(channel_names),
        channel_feature_names=np.asarray(CHANNEL_FEATURE_NAMES),
        aggregate_feature_names=np.asarray(AGGREGATE_FEATURE_NAMES),
    )

    index_df = labels_df.copy()
    index_df.insert(0, "window_id", np.arange(len(index_df), dtype=int))
    index_df.insert(0, "subject", str(subject))
    index_df.insert(0, "base", str(base))
    index_df.to_csv(index_path, index=False)
    return tensor_path, index_path


def _pad_channel_tensor(x_channel: np.ndarray, max_channels: int) -> tuple[np.ndarray, np.ndarray]:
    n_windows, n_channels, n_features = x_channel.shape
    padded = np.zeros((n_windows, max_channels, n_features), dtype=np.float32)
    mask = np.zeros((n_windows, max_channels), dtype=bool)
    padded[:, :n_channels, :] = x_channel
    mask[:, :n_channels] = True
    return padded, mask


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    valid_mask = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[valid_mask]
    labels = labels[valid_mask]
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=float)
    sum_ranks_pos = float(np.sum(ranks[positives]))
    return (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)


def _build_feature_reports(
    run_inventory: pd.DataFrame,
    reports_dir: Path,
    *,
    artifact_root: Path,
    paths: WorkspacePaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_distribution = (
        run_inventory.groupby("subject", as_index=False)
        .agg(
            n_runs=("base", "nunique"),
            n_windows=("n_windows", "sum"),
            n_ictal=("n_ictal", "sum"),
            n_interictal=("n_interictal", "sum"),
            n_dropped_margin=("n_dropped_margin", "sum"),
        )
        .sort_values("subject", kind="mergesort")
        .reset_index(drop=True)
    )

    agg_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    for _, row in run_inventory.iterrows():
        tensor_path = resolve_artifact_path(
            row["tensor_path"],
            workspace=paths.workspace,
            outputs_dir=paths.outputs_dir,
            artifact_root=artifact_root,
        )
        payload = np.load(tensor_path, allow_pickle=True)
        y = payload["y"].astype(np.int8)
        keep = y != -1
        agg_blocks.append(payload["x_agg"][keep].astype(np.float32))
        y_blocks.append(y[keep].astype(np.int8))

    all_agg = np.concatenate(agg_blocks, axis=0) if agg_blocks else np.zeros((0, len(AGGREGATE_FEATURE_NAMES)), dtype=np.float32)
    all_y = np.concatenate(y_blocks, axis=0) if y_blocks else np.zeros((0,), dtype=np.int8)

    feature_stat_rows: list[dict[str, object]] = []
    for label_value, label_name in [(0, "interictal"), (1, "ictal")]:
        label_mask = all_y == label_value
        subset = all_agg[label_mask]
        if subset.size == 0:
            continue
        for feature_idx, feature_name in enumerate(AGGREGATE_FEATURE_NAMES):
            column = subset[:, feature_idx]
            feature_stat_rows.append(
                {
                    "feature": feature_name,
                    "class_name": label_name,
                    "n_rows": int(column.size),
                    "mean": float(np.nanmean(column)),
                    "std": float(np.nanstd(column)),
                    "nan_rate": float(np.mean(~np.isfinite(column))),
                }
            )

    separability_rows: list[dict[str, object]] = []
    for feature_idx, feature_name in enumerate(AGGREGATE_FEATURE_NAMES):
        auc = _binary_auc(all_agg[:, feature_idx], all_y)
        separability_rows.append(
            {
                "feature": feature_name,
                "auc": float(auc),
                "median_class_gap": float(
                    np.nanmedian(all_agg[all_y == 1, feature_idx]) - np.nanmedian(all_agg[all_y == 0, feature_idx])
                )
                if np.any(all_y == 1) and np.any(all_y == 0)
                else float("nan"),
            }
        )

    feature_stats_df = pd.DataFrame(feature_stat_rows).sort_values(["feature", "class_name"], kind="mergesort")
    separability_df = pd.DataFrame(separability_rows).sort_values("auc", ascending=False, kind="mergesort")

    label_distribution.to_csv(reports_dir / "label_distribution.csv", index=False)
    feature_stats_df.to_csv(reports_dir / "feature_stats.csv", index=False)
    separability_df.to_csv(reports_dir / "class_separability.csv", index=False)
    return label_distribution, feature_stats_df, separability_df


def run_feature_pipeline(
    *,
    paths: WorkspacePaths | None = None,
    config: FeaturePipelineConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    paths = paths or get_paths()
    config = config or FeaturePipelineConfig()

    artifact_root = _artifact_root(paths, config.artifact_subdir)
    preprocess_summary_path = artifact_root / "preprocess_run_summary.csv"
    if not preprocess_summary_path.exists():
        raise FileNotFoundError(
            f"Missing preprocess summary at {preprocess_summary_path}. Run preprocessing first."
        )

    preprocessed_summary = pd.read_csv(preprocess_summary_path)
    preprocessed_summary = preprocessed_summary[preprocessed_summary["preprocess_ok"].astype(bool)].copy()
    if preprocessed_summary.empty:
        raise RuntimeError("No successfully preprocessed runs were found.")
    final_bad_channel_map = _load_final_bad_channel_map(artifact_root)

    intervals_df = pd.read_csv(paths.outputs_dir / "ds003029_seizure_intervals_by_run.csv")
    reports_dir = artifact_root / "reports"
    run_dir = artifact_root / "features" / "runs"
    folds_dir = artifact_root / "folds"
    reports_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(parents=True, exist_ok=True)

    mne = load_mne(paths)
    run_rows: list[dict[str, object]] = []
    for _, row in preprocessed_summary.iterrows():
        base = str(row.get("base", ""))
        subject = str(row.get("subject", ""))
        stem = _safe_stem(base)
        tensor_path = run_dir / f"{stem}_window_tensor.npz"
        index_path = run_dir / f"{stem}_window_index.csv"

        try:
            raw = mne.io.read_raw_fif(str(row["cache_path"]), preload=True, verbose="ERROR")
            final_bads = final_bad_channel_map.get(base)
            if final_bads is not None:
                raw.info["bads"] = [channel for channel in final_bads if channel in raw.ch_names]
            labels_df, channel_names, x_channel, x_agg = _extract_run_features(
                raw,
                intervals_for_base(intervals_df, base),
                config,
            )
            if config.overwrite or not tensor_path.exists() or not index_path.exists():
                _save_run_tensor_artifacts(
                    run_dir,
                    stem=stem,
                    labels_df=labels_df,
                    channel_names=channel_names,
                    x_channel=x_channel,
                    x_agg=x_agg,
                    base=base,
                    subject=subject,
                )

            y = labels_df["y"].to_numpy(dtype=np.int8)
            run_rows.append(
                {
                    "subject": subject,
                    "base": base,
                    "tensor_path": _portable_artifact_path(tensor_path, artifact_root),
                    "index_path": _portable_artifact_path(index_path, artifact_root),
                    "n_channels_used": int(len(channel_names)),
                    "n_windows": int(len(labels_df)),
                    "n_ictal": int(np.sum(y == 1)),
                    "n_interictal": int(np.sum(y == 0)),
                    "n_dropped_margin": int(np.sum(y == -1)),
                    "feature_ok": True,
                    "error": "",
                }
            )
        except Exception as exc:
            run_rows.append(
                {
                    "subject": subject,
                    "base": base,
                    "tensor_path": _portable_artifact_path(tensor_path, artifact_root),
                    "index_path": _portable_artifact_path(index_path, artifact_root),
                    "n_channels_used": float("nan"),
                    "n_windows": 0,
                    "n_ictal": 0,
                    "n_interictal": 0,
                    "n_dropped_margin": 0,
                    "feature_ok": False,
                    "error": str(exc),
                }
            )

    run_inventory = pd.DataFrame(run_rows).sort_values(["subject", "base"], kind="mergesort").reset_index(drop=True)
    run_inventory.to_csv(artifact_root / "run_feature_inventory.csv", index=False)

    valid_runs = run_inventory[run_inventory["feature_ok"].astype(bool)].copy()
    if valid_runs.empty:
        raise RuntimeError("Feature extraction failed for every preprocessed run.")

    label_distribution, feature_stats_df, separability_df = _build_feature_reports(
        valid_runs,
        reports_dir,
        artifact_root=artifact_root,
        paths=paths,
    )

    subject_table = build_subject_table(valid_runs.assign(n_intervals=1))
    folds = build_loso_folds(subject_table["subject"].astype(str).tolist())
    validate_folds(folds)

    global_max_channels = int(valid_runs["n_channels_used"].max())
    fold_manifest_rows: list[dict[str, object]] = []

    for fold in folds:
        fold_dir = folds_dir / fold.fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_rows = valid_runs[valid_runs["subject"].astype(str).isin(fold.train_subjects)].copy()
        test_rows = valid_runs[valid_runs["subject"].astype(str).isin(fold.test_subjects)].copy()

        def collect_dataset(rows: pd.DataFrame):
            channel_blocks: list[np.ndarray] = []
            agg_blocks: list[np.ndarray] = []
            mask_blocks: list[np.ndarray] = []
            y_blocks: list[np.ndarray] = []
            meta_frames: list[pd.DataFrame] = []

            for _, row in rows.iterrows():
                tensor_path = resolve_artifact_path(
                    row["tensor_path"],
                    workspace=paths.workspace,
                    outputs_dir=paths.outputs_dir,
                    artifact_root=artifact_root,
                )
                index_path = resolve_artifact_path(
                    row["index_path"],
                    workspace=paths.workspace,
                    outputs_dir=paths.outputs_dir,
                    artifact_root=artifact_root,
                )
                payload = np.load(tensor_path, allow_pickle=True)
                index_df = pd.read_csv(index_path)
                y = payload["y"].astype(np.int8)
                keep = y != -1
                x_channel = payload["x_channel"][keep].astype(np.float32)
                x_agg = payload["x_agg"][keep].astype(np.float32)
                y = y[keep]
                padded, mask = _pad_channel_tensor(x_channel, global_max_channels)

                channel_blocks.append(padded)
                agg_blocks.append(x_agg)
                mask_blocks.append(mask)
                y_blocks.append(y)
                meta_frames.append(index_df.loc[keep].reset_index(drop=True))

            x_channel = np.concatenate(channel_blocks, axis=0) if channel_blocks else np.zeros((0, global_max_channels, len(CHANNEL_FEATURE_NAMES)), dtype=np.float32)
            x_agg = np.concatenate(agg_blocks, axis=0) if agg_blocks else np.zeros((0, len(AGGREGATE_FEATURE_NAMES)), dtype=np.float32)
            mask = np.concatenate(mask_blocks, axis=0) if mask_blocks else np.zeros((0, global_max_channels), dtype=bool)
            y = np.concatenate(y_blocks, axis=0) if y_blocks else np.zeros((0,), dtype=np.int8)
            meta = pd.concat(meta_frames, ignore_index=True) if meta_frames else pd.DataFrame(columns=["base", "subject", "window_id", "t_start_s", "t_stop_s", "t_mid_s", "y"])
            return x_channel, x_agg, mask, y, meta

        train_x_channel, train_x_agg, train_mask, train_y, train_meta = collect_dataset(train_rows)
        test_x_channel, test_x_agg, test_mask, test_y, test_meta = collect_dataset(test_rows)
        if train_x_channel.size == 0 or test_x_channel.size == 0:
            raise RuntimeError(f"Fold {fold.fold_id} has an empty train or test set after dropping boundary windows.")

        scalers = {
            "channel_features": fit_array_scaler(
                train_x_channel,
                feature_names=CHANNEL_FEATURE_NAMES,
                valid_mask=train_mask,
            ),
            "aggregate_features": fit_array_scaler(
                train_x_agg,
                feature_names=AGGREGATE_FEATURE_NAMES,
            ),
        }
        train_x_channel_scaled = transform_array(
            train_x_channel,
            scalers["channel_features"],
            fill_missing_with=config.fill_missing_after_scaling,
        )
        test_x_channel_scaled = transform_array(
            test_x_channel,
            scalers["channel_features"],
            fill_missing_with=config.fill_missing_after_scaling,
        )
        train_x_channel_scaled[~train_mask] = config.fill_missing_after_scaling
        test_x_channel_scaled[~test_mask] = config.fill_missing_after_scaling

        train_x_agg_scaled = transform_array(
            train_x_agg,
            scalers["aggregate_features"],
            fill_missing_with=config.fill_missing_after_scaling,
        )
        test_x_agg_scaled = transform_array(
            test_x_agg,
            scalers["aggregate_features"],
            fill_missing_with=config.fill_missing_after_scaling,
        )

        np.savez_compressed(
            fold_dir / "train_dataset.npz",
            x_channel=train_x_channel_scaled.astype(np.float32),
            x_channel_mask=train_mask,
            x_agg=train_x_agg_scaled.astype(np.float32),
            y=train_y.astype(np.int8),
        )
        np.savez_compressed(
            fold_dir / "test_dataset.npz",
            x_channel=test_x_channel_scaled.astype(np.float32),
            x_channel_mask=test_mask,
            x_agg=test_x_agg_scaled.astype(np.float32),
            y=test_y.astype(np.int8),
        )
        train_meta.to_csv(fold_dir / "train_index.csv", index=False)
        test_meta.to_csv(fold_dir / "test_index.csv", index=False)
        save_scalers_json(fold_dir / "scalers.json", scalers)

        manifest = {
            "fold_id": fold.fold_id,
            "train_subjects": list(fold.train_subjects),
            "test_subjects": list(fold.test_subjects),
            "global_max_channels": global_max_channels,
            "channel_feature_names": list(CHANNEL_FEATURE_NAMES),
            "aggregate_feature_names": list(AGGREGATE_FEATURE_NAMES),
            "n_train_windows": int(train_y.size),
            "n_test_windows": int(test_y.size),
            "n_train_positive": int(np.sum(train_y == 1)),
            "n_test_positive": int(np.sum(test_y == 1)),
        }
        (fold_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        fold_manifest_rows.append(manifest)

    fold_manifest_df = pd.DataFrame(fold_manifest_rows).sort_values("fold_id", kind="mergesort")
    fold_manifest_df.to_json(artifact_root / "fold_manifest.json", orient="records", indent=2)

    overall_manifest = {
        "artifact_root": str(artifact_root),
        "config": asdict(config),
        "input_hashes": {
            "preprocess_run_summary": _md5_file(preprocess_summary_path),
            "seizure_intervals": _md5_file(paths.outputs_dir / "ds003029_seizure_intervals_by_run.csv"),
        },
        "feature_names": {
            "channel": list(CHANNEL_FEATURE_NAMES),
            "aggregate": list(AGGREGATE_FEATURE_NAMES),
            "aggregate_suffixes": list(AGGREGATE_SUFFIXES),
        },
        "outputs": {
            "run_feature_inventory": str(artifact_root / "run_feature_inventory.csv"),
            "label_distribution": str(reports_dir / "label_distribution.csv"),
            "feature_stats": str(reports_dir / "feature_stats.csv"),
            "class_separability": str(reports_dir / "class_separability.csv"),
            "fold_manifest": str(artifact_root / "fold_manifest.json"),
        },
    }
    (artifact_root / "manifest_features.json").write_text(json.dumps(overall_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return valid_runs, fold_manifest_df, artifact_root