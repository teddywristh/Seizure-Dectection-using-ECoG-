from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ds003029_eda.data.io import resolve_artifact_path
    from ds003029_eda.paths import WorkspacePaths, get_paths
else:
    from ..data.io import resolve_artifact_path
    from ..paths import WorkspacePaths, get_paths


COMBINED_OUTPUT_NAME = "ds003029_sarima_v2_input.csv"
MANIFEST_OUTPUT_NAME = "sarima_prep_manifest.json"
TARGET_FEATURE_ALIASES = {
    "agg_mean_rms": "rms",
    "agg_mean_gamma_high_power": "rms",
    "agg_mean_hjorth_activity": "rms",
}


@dataclass(frozen=True)
class SarimaPrepConfig:
    artifact_subdir: str = "data_processing_v2"
    output_subdir: str = "sarima"
    aggregate_feature_name: str = "agg_mean_rms"
    exogenous_feature_names: tuple[str, ...] = ()
    overwrite: bool = False


def _is_truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _series_id_from_base(base_value: object) -> str:
    normalized = str(base_value).replace("\\", "/")
    return PurePosixPath(normalized).name


def _load_successful_inventory(artifact_root: Path) -> pd.DataFrame:
    inventory_path = artifact_root / "run_feature_inventory.csv"
    inventory = pd.read_csv(inventory_path)
    required = {"subject", "base", "tensor_path", "index_path"}
    missing = sorted(required.difference(inventory.columns))
    if missing:
        raise KeyError(f"run_feature_inventory.csv is missing required columns: {missing}")

    if "feature_ok" in inventory.columns:
        inventory = inventory[inventory["feature_ok"].map(_is_truthy)].copy()

    if inventory.empty:
        raise RuntimeError(f"No successful feature runs found in {inventory_path}")
    return inventory.reset_index(drop=True)


def _load_rms_series(tensor_path: Path, aggregate_feature_name: str) -> np.ndarray:
    payload = np.load(tensor_path, allow_pickle=False)
    if "x_agg" not in payload or "aggregate_feature_names" not in payload:
        raise KeyError(f"Tensor artifact is missing x_agg or aggregate_feature_names: {tensor_path}")

    feature_names = [str(name) for name in payload["aggregate_feature_names"].tolist()]
    if aggregate_feature_name not in feature_names:
        raise KeyError(
            f"Aggregate feature '{aggregate_feature_name}' not found in {tensor_path}. "
            f"Available features: {feature_names}"
        )

    feature_idx = feature_names.index(aggregate_feature_name)
    x_agg = np.asarray(payload["x_agg"], dtype=float)
    if x_agg.ndim != 2:
        raise ValueError(f"Expected x_agg to be 2D, got shape {x_agg.shape} from {tensor_path}")
    return x_agg[:, feature_idx]


def _load_feature_matrix(
    tensor_path: Path,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    if not feature_names:
        return np.zeros((0, 0), dtype=float), []

    payload = np.load(tensor_path, allow_pickle=False)
    if "x_agg" not in payload or "aggregate_feature_names" not in payload:
        raise KeyError(f"Tensor artifact is missing x_agg or aggregate_feature_names: {tensor_path}")

    available_names = [str(name) for name in payload["aggregate_feature_names"].tolist()]
    missing = sorted(set(feature_names).difference(available_names))
    if missing:
        raise KeyError(
            f"Aggregate feature(s) {missing} not found in {tensor_path}. Available features: {available_names}"
        )

    x_agg = np.asarray(payload["x_agg"], dtype=float)
    feature_indices = [available_names.index(name) for name in feature_names]
    return x_agg[:, feature_indices], list(feature_names)


def build_sarima_run_frame(
    index_df: pd.DataFrame,
    rms_series: np.ndarray,
    *,
    aggregate_feature_name: str,
    exogenous_matrix: np.ndarray | None = None,
    tensor_exogenous_feature_names: list[str] | None = None,
    exogenous_feature_names: list[str] | None = None,
) -> pd.DataFrame:
    required = {"base", "subject", "window_id", "t_mid_s", "y"}
    missing = sorted(required.difference(index_df.columns))
    if missing:
        raise KeyError(f"Window index is missing required columns: {missing}")
    if len(index_df) != len(rms_series):
        raise ValueError(
            f"Index/tensor row mismatch: {len(index_df)} index rows vs {len(rms_series)} signal rows"
        )

    frame = index_df.copy()
    frame["series_id"] = frame["base"].astype(str).map(_series_id_from_base)
    frame["rms"] = np.asarray(rms_series, dtype=float)
    frame["t_mid_s"] = pd.to_numeric(frame["t_mid_s"], errors="coerce")
    if "t_start_s" in frame.columns:
        frame["t_start_s"] = pd.to_numeric(frame["t_start_s"], errors="coerce")
    if "t_stop_s" in frame.columns:
        frame["t_stop_s"] = pd.to_numeric(frame["t_stop_s"], errors="coerce")
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce").astype("Int64")
    frame["source_feature"] = aggregate_feature_name
    frame["target_feature"] = aggregate_feature_name

    alias_name = TARGET_FEATURE_ALIASES.get(aggregate_feature_name, "rms")
    if alias_name != "rms":
        raise ValueError(f"Unsupported target alias for feature {aggregate_feature_name}: {alias_name}")

    if exogenous_matrix is not None and tensor_exogenous_feature_names:
        if len(exogenous_matrix) != len(frame):
            raise ValueError(
                f"Exogenous/index row mismatch: {len(exogenous_matrix)} exogenous rows vs {len(frame)} index rows"
            )
        for feature_idx, feature_name in enumerate(tensor_exogenous_feature_names):
            frame[feature_name] = exogenous_matrix[:, feature_idx]

    if "y_lagged" in (exogenous_feature_names or []):
        binary_y = frame["y"].fillna(0).replace(-1, 0).astype(int)
        frame["y_lagged"] = binary_y.shift(1).fillna(0).astype(float)

    keep_cols = [
        "series_id",
        "subject",
        "base",
        "window_id",
        "t_start_s",
        "t_stop_s",
        "t_mid_s",
        "rms",
        "y",
        "source_feature",
        "target_feature",
    ]
    keep_cols.extend(exogenous_feature_names or [])
    available_cols = [col for col in keep_cols if col in frame.columns]
    return frame[available_cols].sort_values("t_mid_s").reset_index(drop=True)


def run_sarima_prep(
    *,
    paths: WorkspacePaths | None = None,
    config: SarimaPrepConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    paths = paths or get_paths()
    config = config or SarimaPrepConfig()

    artifact_root = paths.outputs_dir / config.artifact_subdir
    inventory = _load_successful_inventory(artifact_root)

    output_root = artifact_root / config.output_subdir
    runs_root = output_root / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    if config.overwrite:
        for stale_file in runs_root.glob("*_sarima_input.csv"):
            stale_file.unlink()

    run_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    for row in inventory.itertuples(index=False):
        tensor_path = resolve_artifact_path(
            row.tensor_path,
            workspace=paths.workspace,
            outputs_dir=paths.outputs_dir,
            artifact_root=artifact_root,
        )
        index_path = resolve_artifact_path(
            row.index_path,
            workspace=paths.workspace,
            outputs_dir=paths.outputs_dir,
            artifact_root=artifact_root,
        )
        index_df = pd.read_csv(index_path)
        rms_series = _load_rms_series(tensor_path, config.aggregate_feature_name)
        exogenous_matrix, exogenous_feature_names = _load_feature_matrix(
            tensor_path,
            tuple(name for name in config.exogenous_feature_names if name != "y_lagged"),
        )
        final_exogenous_feature_names = list(exogenous_feature_names)
        if "y_lagged" in config.exogenous_feature_names:
            final_exogenous_feature_names.append("y_lagged")
        run_frame = build_sarima_run_frame(
            index_df,
            rms_series,
            aggregate_feature_name=config.aggregate_feature_name,
            exogenous_matrix=exogenous_matrix if exogenous_feature_names else None,
            tensor_exogenous_feature_names=exogenous_feature_names,
            exogenous_feature_names=final_exogenous_feature_names,
        )
        series_id = str(run_frame["series_id"].iloc[0])
        run_output_path = runs_root / f"{series_id}_sarima_input.csv"
        run_frame.to_csv(run_output_path, index=False)
        run_frames.append(run_frame)
        manifest_rows.append(
            {
                "subject": str(row.subject),
                "series_id": series_id,
                "base": str(row.base),
                "n_windows": int(len(run_frame)),
                "n_positive": int((run_frame["y"] == 1).sum()),
                "n_boundary": int((run_frame["y"] == -1).sum()),
                "target_feature": config.aggregate_feature_name,
                "exogenous_features": final_exogenous_feature_names,
                "csv_path": run_output_path.as_posix(),
            }
        )

    combined_df = (
        pd.concat(run_frames, ignore_index=True)
        .sort_values(["subject", "series_id", "t_mid_s"])
        .reset_index(drop=True)
    )
    manifest_df = pd.DataFrame(manifest_rows)

    combined_output_path = output_root / COMBINED_OUTPUT_NAME
    combined_df.to_csv(combined_output_path, index=False)

    manifest_payload = {
        "config": asdict(config),
        "artifact_root": artifact_root.as_posix(),
        "output_root": output_root.as_posix(),
        "combined_csv": combined_output_path.as_posix(),
        "n_series": int(len(manifest_df)),
        "n_rows": int(len(combined_df)),
        "target_feature": config.aggregate_feature_name,
        "exogenous_features": list(config.exogenous_feature_names),
        "series": manifest_rows,
    }
    (output_root / MANIFEST_OUTPUT_NAME).write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return combined_df, manifest_df, output_root