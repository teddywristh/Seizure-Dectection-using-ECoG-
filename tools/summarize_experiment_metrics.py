from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _workspace_cli import load_family_report_config


AGGREGATE_COLUMNS = ("metric", "mean", "std", "min", "max")
ML_DL_TABLE_COLUMNS = (
    "model",
    "roc_auc_mean",
    "average_precision_mean",
    "f1_mean",
    "precision_mean",
    "sensitivity_mean",
    "specificity_mean",
    "accuracy_mean",
)
TIMESERIES_TABLE_COLUMNS = (
    "model",
    "exogenous_cols",
    "n_series",
    "ok_series",
    "rmse_test_mean",
    "mae_test_mean",
    "smape_test_pct_mean",
    "r2_test_mean",
    "anomaly_count_test_mean",
)
DEFAULT_REPORT_CONFIG = "configs/reports/family_reports.json"
FAMILY_CHOICES = ("all", "ml", "dl", "timeseries")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize timeseries, ML, and DL experiment metrics into CSV and Markdown reports."
    )
    parser.add_argument(
        "--workspace-root",
        default=None,
        help="Workspace root to search from. If omitted, the script searches upward from the current directory.",
    )
    parser.add_argument(
        "--experiments-root",
        default=None,
        help="Optional direct path to eda_outputs/experiments.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional direct path for generated summaries. Defaults to <experiments-root>/summary.",
    )
    parser.add_argument("--family", choices=FAMILY_CHOICES, default="all")
    parser.add_argument("--report-config", default=DEFAULT_REPORT_CONFIG)
    return parser


def _resolve_experiments_root(*, workspace_root: str | None, experiments_root: str | None) -> Path:
    if experiments_root is not None:
        resolved = Path(experiments_root).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Experiments root does not exist: {resolved}")
        return resolved

    search_start = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
    candidates: list[tuple[int, int, Path]] = []
    for distance, candidate in enumerate([search_start, *search_start.parents]):
        experiments_dir = candidate / "eda_outputs" / "experiments"
        if not experiments_dir.exists():
            continue
        family_dir_count = sum(1 for child in experiments_dir.iterdir() if child.is_dir())
        candidates.append((family_dir_count, -distance, experiments_dir))

    if not candidates:
        raise FileNotFoundError(
            "Could not find eda_outputs/experiments from the provided start path. "
            "Pass --workspace-root or --experiments-root explicitly."
        )

    return max(candidates)[2]


def _read_aggregate_metrics(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing = [column for column in AGGREGATE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    for column in ("mean", "std", "min", "max"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _flatten_aggregate_metrics(*, model_name: str, metrics_frame: pd.DataFrame) -> dict[str, object]:
    flattened: dict[str, object] = {"model": model_name}
    for row in metrics_frame.itertuples(index=False):
        metric_name = str(row.metric)
        flattened[f"{metric_name}_mean"] = row.mean
        flattened[f"{metric_name}_std"] = row.std
        flattened[f"{metric_name}_min"] = row.min
        flattened[f"{metric_name}_max"] = row.max
    return flattened


def _summarize_family_from_aggregates(*, family_name: str, family_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not family_dir.exists():
        return pd.DataFrame(), pd.DataFrame(columns=["family", "model", "metric", "mean", "std", "min", "max"])

    summary_rows: list[dict[str, object]] = []
    long_rows: list[pd.DataFrame] = []

    for experiment_dir in sorted(path for path in family_dir.iterdir() if path.is_dir()):
        aggregate_path = experiment_dir / "aggregate_metrics.csv"
        if not aggregate_path.exists():
            continue
        metrics_frame = _read_aggregate_metrics(aggregate_path)
        summary_rows.append(_flatten_aggregate_metrics(model_name=experiment_dir.name, metrics_frame=metrics_frame))
        long_rows.append(metrics_frame.assign(family=family_name, model=experiment_dir.name))

    summary_frame = pd.DataFrame(summary_rows).sort_values("model", kind="mergesort") if summary_rows else pd.DataFrame()
    long_frame = (
        pd.concat(long_rows, ignore_index=True)[["family", "model", "metric", "mean", "std", "min", "max"]]
        if long_rows
        else pd.DataFrame(columns=["family", "model", "metric", "mean", "std", "min", "max"])
    )
    return summary_frame, long_frame


def _mode_text(values: pd.Series) -> str:
    cleaned = values.dropna().astype(str)
    if cleaned.empty:
        return ""
    modes = cleaned.mode(dropna=True)
    if not modes.empty:
        return str(modes.iloc[0])
    return str(cleaned.iloc[0])


def _summarize_timeseries_family(timeseries_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not timeseries_dir.exists():
        return pd.DataFrame(), pd.DataFrame(columns=["family", "model", "metric", "mean", "std", "min", "max"])

    summary_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []

    for experiment_dir in sorted(path for path in timeseries_dir.iterdir() if path.is_dir()):
        series_metrics_path = experiment_dir / "series_metrics.csv"
        if not series_metrics_path.exists():
            continue

        metrics_frame = pd.read_csv(series_metrics_path)
        ok_mask = metrics_frame["status"].fillna("").eq("ok") if "status" in metrics_frame.columns else pd.Series(True, index=metrics_frame.index)
        ok_frame = metrics_frame.loc[ok_mask].copy()
        if ok_frame.empty:
            ok_frame = metrics_frame.copy()

        summary_row: dict[str, object] = {
            "model": experiment_dir.name,
            "n_series": int(len(metrics_frame)),
            "ok_series": int(len(ok_frame)),
            "target_col": _mode_text(ok_frame["target_col"]) if "target_col" in ok_frame.columns else "",
            "exogenous_cols": _mode_text(ok_frame["exogenous_cols"]) if "exogenous_cols" in ok_frame.columns else "",
            "selected_order_mode": _mode_text(ok_frame["selected_order"]) if "selected_order" in ok_frame.columns else "",
            "selected_seasonal_order_mode": _mode_text(ok_frame["selected_seasonal_order"]) if "selected_seasonal_order" in ok_frame.columns else "",
        }

        for column in ok_frame.columns:
            numeric_values = pd.to_numeric(ok_frame[column], errors="coerce").dropna()
            if numeric_values.empty:
                continue
            summary_row[f"{column}_mean"] = float(numeric_values.mean())
            summary_row[f"{column}_std"] = float(numeric_values.std(ddof=0))
            summary_row[f"{column}_min"] = float(numeric_values.min())
            summary_row[f"{column}_max"] = float(numeric_values.max())
            long_rows.append(
                {
                    "family": "timeseries",
                    "model": experiment_dir.name,
                    "metric": column,
                    "mean": float(numeric_values.mean()),
                    "std": float(numeric_values.std(ddof=0)),
                    "min": float(numeric_values.min()),
                    "max": float(numeric_values.max()),
                }
            )

        summary_rows.append(summary_row)

    summary_frame = pd.DataFrame(summary_rows).sort_values("model", kind="mergesort") if summary_rows else pd.DataFrame()
    long_frame = pd.DataFrame(long_rows).sort_values(["model", "metric"], kind="mergesort") if long_rows else pd.DataFrame(
        columns=["family", "model", "metric", "mean", "std", "min", "max"]
    )
    return summary_frame, long_frame


def _format_markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    numeric_value = float(value)
    if numeric_value.is_integer() and abs(numeric_value) < 100000:
        return str(int(numeric_value))
    magnitude = abs(numeric_value)
    if magnitude >= 1000 or (magnitude > 0 and magnitude < 1e-3):
        return f"{numeric_value:.4e}"
    return f"{numeric_value:.4f}"


def _markdown_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    available_columns = [column for column in columns if column in frame.columns]
    if not available_columns:
        return ["No completed runs found."]

    header = "| " + " | ".join(column.replace("_", " ") for column in available_columns) + " |"
    separator = "|" + "|".join("---" for _ in available_columns) + "|"
    lines = [header, separator]

    for _, row in frame.loc[:, available_columns].iterrows():
        cells = [_format_markdown_value(row[column]) for column in available_columns]
        lines.append("| " + " | ".join(cells) + " |")

    return lines


def _pick_extreme(frame: pd.DataFrame, *, metric: str, ascending: bool) -> tuple[str, float] | None:
    if metric not in frame.columns or frame.empty:
        return None
    candidate_frame = frame.loc[:, ["model", metric]].dropna().copy()
    if candidate_frame.empty:
        return None
    candidate_frame = candidate_frame.sort_values(metric, ascending=ascending, kind="mergesort")
    best_row = candidate_frame.iloc[0]
    return str(best_row["model"]), float(best_row[metric])


def _build_summary_markdown(
    *,
    experiments_root: Path,
    selected_families: list[str],
    ml_summary: pd.DataFrame,
    dl_summary: pd.DataFrame,
    timeseries_summary: pd.DataFrame,
    generated_csv_files: list[str],
) -> str:
    family_labels = {
        "ml": "ML",
        "dl": "DL",
        "timeseries": "Timeseries",
    }
    lines: list[str] = [
        "# Metrics Summary",
        "",
        f"_Auto-generated from {experiments_root.as_posix()}_",
        "",
        "---",
        "",
        "## Available outputs",
        "",
    ]

    summary_lookup = {
        "ml": ml_summary,
        "dl": dl_summary,
        "timeseries": timeseries_summary,
    }
    for family in selected_families:
        family_summary = summary_lookup[family]
        lines.append(
            f"- {family_labels[family]}: {', '.join(family_summary['model'].tolist()) if not family_summary.empty else 'none'}"
        )

    if "ml" in selected_families:
        lines.extend(["", "## Machine learning", ""])
        lines.extend(
            _markdown_table(
                ml_summary.sort_values("roc_auc_mean", ascending=False, kind="mergesort")
                if "roc_auc_mean" in ml_summary.columns
                else ml_summary,
                ML_DL_TABLE_COLUMNS,
            )
        )

        ml_best_roc = _pick_extreme(ml_summary, metric="roc_auc_mean", ascending=False)
        ml_best_f1 = _pick_extreme(ml_summary, metric="f1_mean", ascending=False)
        if ml_best_roc or ml_best_f1:
            lines.extend(["", "Quick read:"])
            if ml_best_roc:
                lines.append(f"- Best ROC AUC mean: `{ml_best_roc[0]}` ({_format_markdown_value(ml_best_roc[1])})")
            if ml_best_f1:
                lines.append(f"- Best F1 mean: `{ml_best_f1[0]}` ({_format_markdown_value(ml_best_f1[1])})")

    if "dl" in selected_families:
        lines.extend(["", "## Deep learning", ""])
        lines.extend(
            _markdown_table(
                dl_summary.sort_values("roc_auc_mean", ascending=False, kind="mergesort")
                if "roc_auc_mean" in dl_summary.columns
                else dl_summary,
                ML_DL_TABLE_COLUMNS,
            )
        )

        dl_best_roc = _pick_extreme(dl_summary, metric="roc_auc_mean", ascending=False)
        dl_best_f1 = _pick_extreme(dl_summary, metric="f1_mean", ascending=False)
        if dl_best_roc or dl_best_f1:
            lines.extend(["", "Quick read:"])
            if dl_best_roc:
                lines.append(f"- Best ROC AUC mean: `{dl_best_roc[0]}` ({_format_markdown_value(dl_best_roc[1])})")
            if dl_best_f1:
                lines.append(f"- Best F1 mean: `{dl_best_f1[0]}` ({_format_markdown_value(dl_best_f1[1])})")

    if "timeseries" in selected_families:
        lines.extend(["", "## Timeseries", ""])
        lines.extend(
            _markdown_table(
                timeseries_summary.sort_values("rmse_test_mean", ascending=True, kind="mergesort")
                if "rmse_test_mean" in timeseries_summary.columns
                else timeseries_summary,
                TIMESERIES_TABLE_COLUMNS,
            )
        )

        ts_best_rmse = _pick_extreme(timeseries_summary, metric="rmse_test_mean", ascending=True)
        ts_best_smape = _pick_extreme(timeseries_summary, metric="smape_test_pct_mean", ascending=True)
        if ts_best_rmse or ts_best_smape:
            lines.extend(["", "Quick read:"])
            if ts_best_rmse:
                lines.append(f"- Lowest RMSE mean: `{ts_best_rmse[0]}` ({_format_markdown_value(ts_best_rmse[1])})")
            if ts_best_smape:
                lines.append(f"- Lowest sMAPE mean: `{ts_best_smape[0]}` ({_format_markdown_value(ts_best_smape[1])})")

    lines.extend(["", "## Detailed files", ""])
    for file_name in generated_csv_files:
        lines.append(f"- `{file_name}`")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    experiments_root = _resolve_experiments_root(
        workspace_root=args.workspace_root,
        experiments_root=args.experiments_root,
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else experiments_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    _resolved_report_config, family_order, family_specs, all_spec = load_family_report_config(args.report_config)
    selected_families = list(family_order) if args.family == "all" else [args.family]

    family_summaries: dict[str, pd.DataFrame] = {
        "ml": pd.DataFrame(),
        "dl": pd.DataFrame(),
        "timeseries": pd.DataFrame(),
    }
    family_longs: dict[str, pd.DataFrame] = {
        family: pd.DataFrame(columns=["family", "model", "metric", "mean", "std", "min", "max"])
        for family in ("ml", "dl", "timeseries")
    }

    for family in selected_families:
        if family == "timeseries":
            summary_frame, long_frame = _summarize_timeseries_family(experiments_root / family)
        else:
            summary_frame, long_frame = _summarize_family_from_aggregates(
                family_name=family,
                family_dir=experiments_root / family,
            )
        family_summaries[family] = summary_frame
        family_longs[family] = long_frame

    long_frames = [family_longs[family] for family in selected_families if not family_longs[family].empty]
    all_metrics_long = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame(
        columns=["family", "model", "metric", "mean", "std", "min", "max"]
    )
    if not all_metrics_long.empty:
        all_metrics_long = all_metrics_long.sort_values(["family", "model", "metric"], kind="mergesort")

    generated_csv_files: list[str] = []
    for family in selected_families:
        summary_file = str(family_specs[family]["summary_file"])
        family_summaries[family].to_csv(output_dir / summary_file, index=False)
        generated_csv_files.append(summary_file)

    long_file_name = (
        str(all_spec["summary_long_file"])
        if args.family == "all"
        else str(family_specs[args.family]["long_file"])
    )
    all_metrics_long.to_csv(output_dir / long_file_name, index=False)
    generated_csv_files.append(long_file_name)

    markdown_file_name = (
        str(all_spec["summary_markdown"])
        if args.family == "all"
        else str(family_specs[args.family]["summary_markdown"])
    )

    markdown_summary = _build_summary_markdown(
        experiments_root=experiments_root,
        selected_families=selected_families,
        ml_summary=family_summaries["ml"],
        dl_summary=family_summaries["dl"],
        timeseries_summary=family_summaries["timeseries"],
        generated_csv_files=generated_csv_files,
    )
    (output_dir / markdown_file_name).write_text(markdown_summary, encoding="utf-8")

    print(f"Experiments root: {experiments_root.as_posix()}")
    print(f"Family scope: {', '.join(selected_families)}")
    print(f"Summary output: {output_dir.as_posix()}")
    for family in selected_families:
        print(f"{family} presets summarized: {len(family_summaries[family])}")
    print("Generated files:")
    for file_name in [*generated_csv_files, markdown_file_name]:
        print(f"- {(output_dir / file_name).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())