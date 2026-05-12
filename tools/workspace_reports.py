from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _workspace_cli import load_family_report_config, resolve_workspace_paths, run_python_script


DEFAULT_REPORT_CONFIG = "configs/reports/family_reports.json"
FAMILY_CHOICES = ("all", "ml", "dl", "timeseries")
PAPER_MODEL_NAME_OVERRIDES = {
    "catboost": "CatBoost",
    "ce_tss_transformer": "CE-TSS-Transformer",
    "lightgbm_dart": "LightGBM-DART",
    "sarimax_gamma": "SARIMAX-Gamma",
    "xgboost_optuna": "XGBoost",
    "inresformer": "InResFormer",
    "stacking": "Stacking",
    "dbconformer": "DBConformer",
}
PAPER_FAMILY_LABELS = {
    "ML": "ML",
    "DL": "DL",
    "TIMESERIES": "TS",
}
PAPER_LINE_STYLES = {
    "ML": "-",
    "DL": "--",
    "TIMESERIES": ":",
}
CURVE_GRID = np.linspace(0.0, 1.0, 201)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--family", choices=FAMILY_CHOICES, default="all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task 4: summarize experiment outputs, verify artifacts, and draw metric plots "
            "without rerunning the models."
        )
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    summarize_parser = subparsers.add_parser("summarize", help="Regenerate CSV and Markdown metric summaries.")
    _add_common_arguments(summarize_parser)
    summarize_parser.add_argument("--output-dir", default=None)

    verify_parser = subparsers.add_parser("verify", help="Validate experiment outputs already present on disk.")
    _add_common_arguments(verify_parser)
    verify_parser.add_argument("--output-dir", default=None)
    verify_parser.add_argument("--fail-on-error", dest="fail_on_error", action="store_true")
    verify_parser.add_argument("--no-fail-on-error", dest="fail_on_error", action="store_false")
    verify_parser.set_defaults(fail_on_error=True)

    plot_parser = subparsers.add_parser("plot", help="Create simple PNG metric dashboards from summary CSV files.")
    _add_common_arguments(plot_parser)
    plot_parser.add_argument("--summary-dir", default=None)
    plot_parser.add_argument("--plot-dir", default=None)

    leaderboard_parser = subparsers.add_parser(
        "cross_family_leaderboard",
        help="Build a ROC/PRAUC leaderboard across timeseries, ML, and DL experiments.",
    )
    leaderboard_parser.add_argument("--workspace-root", default=None)
    leaderboard_parser.add_argument("--output-file", default=None)

    paper_parser = subparsers.add_parser(
        "paper_figures",
        help="Create LOSO class-balance, ROC, and PR figures for the top-ranked cross-family models.",
    )
    paper_parser.add_argument("--workspace-root", default=None)
    paper_parser.add_argument("--plot-dir", default=None)
    paper_parser.add_argument("--leaderboard-file", default=None)
    paper_parser.add_argument("--fold-metrics-file", default=None)
    paper_parser.add_argument("--top-k", type=int, default=8)
    return parser


def _resolve_report_config(config_path: str | None) -> tuple[Path, list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    return load_family_report_config(config_path or DEFAULT_REPORT_CONFIG)


def _summary_dir(*, workspace_root: str | None, summary_dir: str | None) -> Path:
    if summary_dir is not None:
        return Path(summary_dir).resolve()
    paths = resolve_workspace_paths(workspace_root)
    return paths.outputs_dir / "experiments" / "summary"


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency at runtime
        raise ImportError("matplotlib is required for workspace_reports.py plot. Install the project requirements first.") from exc
    return plt


def _selected_families(requested_family: str, family_order: list[str]) -> list[str]:
    return list(family_order) if requested_family == "all" else [requested_family]


def _load_sklearn_metrics():
    try:
        from sklearn.metrics import precision_recall_curve, roc_curve
    except ImportError as exc:  # pragma: no cover - optional dependency at runtime
        raise ImportError(
            "scikit-learn is required for workspace_reports.py paper_figures. Install the project requirements first."
        ) from exc
    return precision_recall_curve, roc_curve


def _pretty_model_name(preset: str) -> str:
    if preset in PAPER_MODEL_NAME_OVERRIDES:
        return PAPER_MODEL_NAME_OVERRIDES[preset]
    return preset.replace("_", "-").title()


def _fold_sort_key(fold_id: str) -> tuple[int, str]:
    prefix = str(fold_id).split("_")
    if len(prefix) > 1 and prefix[1].isdigit():
        return int(prefix[1]), str(fold_id)
    return 10**9, str(fold_id)


def _timeseries_residual_to_score(residuals: pd.Series) -> np.ndarray:
    abs_residuals = np.abs(pd.to_numeric(residuals, errors="coerce").to_numpy(dtype=float))
    if abs_residuals.size == 0:
        return abs_residuals

    mean_value = float(abs_residuals.mean())
    std_value = float(abs_residuals.std(ddof=0))
    if std_value < 1e-12:
        z_score = np.zeros_like(abs_residuals, dtype=float)
    else:
        z_score = (abs_residuals - mean_value) / (std_value + 1e-12)
    z_score = np.clip(z_score, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z_score))


def _load_paper_leaderboard(*, workspace_root: str | None, leaderboard_file: str | None) -> pd.DataFrame:
    if leaderboard_file is not None:
        leaderboard_path = Path(leaderboard_file).resolve()
    else:
        leaderboard_path = build_cross_family_leaderboard(workspace_root=workspace_root, output_file=None)

    leaderboard = pd.read_csv(leaderboard_path)
    if leaderboard.empty:
        raise RuntimeError("Cross-family leaderboard is empty; cannot build paper figures.")
    return leaderboard


def _load_loso_fold_counts(fold_metrics_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(fold_metrics_path)
    required_columns = {"fold_id", "n_positive", "n_negative"}
    missing = required_columns.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"Missing required LOSO class-count columns in {fold_metrics_path.as_posix()}: {missing_text}")

    plot_frame = frame.loc[:, ["fold_id", "n_positive", "n_negative"]].copy()
    plot_frame["fold_id"] = plot_frame["fold_id"].astype(str)
    plot_frame["n_positive"] = pd.to_numeric(plot_frame["n_positive"], errors="coerce")
    plot_frame["n_negative"] = pd.to_numeric(plot_frame["n_negative"], errors="coerce")
    plot_frame = plot_frame.dropna(subset=["n_positive", "n_negative"])
    plot_frame = plot_frame.sort_values("fold_id", key=lambda values: values.map(_fold_sort_key), kind="mergesort")
    plot_frame["subject"] = plot_frame["fold_id"].str.rsplit("_", n=1).str[-1]
    plot_frame["display_label"] = plot_frame["subject"] + " (" + plot_frame["fold_id"].str.split("_").str[1] + ")"
    plot_frame["total_windows"] = plot_frame["n_positive"] + plot_frame["n_negative"]
    plot_frame["ictal_fraction"] = np.where(
        plot_frame["total_windows"] > 0,
        plot_frame["n_positive"] / plot_frame["total_windows"],
        np.nan,
    )
    return plot_frame.reset_index(drop=True)


def _load_curve_source(*, experiments_root: Path, family: str, preset: str) -> tuple[pd.DataFrame, str]:
    family_upper = str(family).upper()
    if family_upper in {"ML", "DL"}:
        prediction_path = experiments_root / family_upper.lower() / preset / "all_predictions.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing prediction file for {family_upper}/{preset}: {prediction_path.as_posix()}")
        frame = pd.read_csv(prediction_path)
        frame["y_true"] = pd.to_numeric(frame.get("y_true", frame.get("y")), errors="coerce")
        frame["y_score"] = pd.to_numeric(frame["y_score"], errors="coerce")
        if "split" in frame.columns:
            frame = frame.loc[frame["split"].astype(str) == "test"].copy()
        group_col = "fold_id" if "fold_id" in frame.columns else "subject"
        return frame, group_col

    if family_upper == "TIMESERIES":
        prediction_path = experiments_root / "timeseries" / preset / "sarima_results" / "sarima_all_predictions.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing SARIMA prediction file for {preset}: {prediction_path.as_posix()}")
        frame = pd.read_csv(prediction_path)
        if "split" in frame.columns:
            frame = frame.loc[frame["split"].astype(str) == "test"].copy()
        frame["y_true"] = pd.to_numeric(frame["y"], errors="coerce")
        frame["y_score"] = _timeseries_residual_to_score(frame["sarima_residual"])
        return frame, "series_id"

    raise ValueError(f"Unsupported family for paper curves: {family}")


def _prepare_curve_groups(frame: pd.DataFrame, *, group_col: str) -> pd.DataFrame:
    required_columns = {group_col, "y_true", "y_score"}
    missing = required_columns.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"Missing required curve columns: {missing_text}")

    plot_frame = frame.loc[:, [group_col, "y_true", "y_score"]].copy()
    plot_frame["y_true"] = pd.to_numeric(plot_frame["y_true"], errors="coerce")
    plot_frame["y_score"] = pd.to_numeric(plot_frame["y_score"], errors="coerce")
    plot_frame = plot_frame.dropna(subset=[group_col, "y_true", "y_score"])
    plot_frame = plot_frame.loc[plot_frame["y_true"].isin((0, 1))].copy()

    valid_groups: list[pd.DataFrame] = []
    for _, group_frame in plot_frame.groupby(group_col, sort=True):
        if group_frame["y_true"].nunique() < 2:
            continue
        valid_groups.append(group_frame)

    if not valid_groups:
        raise RuntimeError("No valid groups with both classes were available to compute averaged curves.")

    return pd.concat(valid_groups, ignore_index=True)


def _interpolate_curve(x_values: np.ndarray, y_values: np.ndarray, *, grid: np.ndarray) -> np.ndarray:
    curve_frame = pd.DataFrame({"x": x_values, "y": y_values}).dropna().sort_values("x", kind="mergesort")
    if curve_frame.empty:
        return np.full_like(grid, np.nan, dtype=float)
    curve_frame = curve_frame.groupby("x", as_index=False)["y"].max()
    return np.interp(
        grid,
        curve_frame["x"].to_numpy(dtype=float),
        curve_frame["y"].to_numpy(dtype=float),
        left=float(curve_frame["y"].iloc[0]),
        right=float(curve_frame["y"].iloc[-1]),
    )


def _mean_roc_curve(frame: pd.DataFrame, *, group_col: str) -> tuple[np.ndarray, np.ndarray]:
    _, roc_curve = _load_sklearn_metrics()
    interpolated_curves: list[np.ndarray] = []
    for _, group_frame in frame.groupby(group_col, sort=True):
        fpr, tpr, _ = roc_curve(group_frame["y_true"].to_numpy(dtype=int), group_frame["y_score"].to_numpy(dtype=float))
        interpolated = _interpolate_curve(fpr, tpr, grid=CURVE_GRID)
        interpolated[0] = 0.0
        interpolated[-1] = 1.0
        interpolated_curves.append(interpolated)
    return CURVE_GRID, np.nanmean(np.vstack(interpolated_curves), axis=0)


def _mean_pr_curve(frame: pd.DataFrame, *, group_col: str) -> tuple[np.ndarray, np.ndarray]:
    precision_recall_curve, _ = _load_sklearn_metrics()
    interpolated_curves: list[np.ndarray] = []
    for _, group_frame in frame.groupby(group_col, sort=True):
        precision, recall, _ = precision_recall_curve(
            group_frame["y_true"].to_numpy(dtype=int),
            group_frame["y_score"].to_numpy(dtype=float),
        )
        interpolated_curves.append(_interpolate_curve(recall, precision, grid=CURVE_GRID))
    mean_precision = np.nanmean(np.vstack(interpolated_curves), axis=0)
    mean_precision = np.clip(mean_precision, 0.0, 1.0)
    return CURVE_GRID, mean_precision


def _render_loso_class_balance(*, plot_frame: pd.DataFrame, output_path: Path) -> Path:
    plt = _load_pyplot()
    fig_height = max(5.5, 0.7 * len(plot_frame) + 1.5)
    fig, axis = plt.subplots(figsize=(12, fig_height))

    labels = plot_frame["display_label"].tolist()
    positions = np.arange(len(plot_frame))
    axis.barh(positions, plot_frame["n_negative"], color="#4C78A8", label="Interictal")
    axis.barh(
        positions,
        plot_frame["n_positive"],
        left=plot_frame["n_negative"],
        color="#E45756",
        label="Ictal",
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel("Window count")
    axis.set_ylabel("Test fold")
    axis.set_title("Class Balance per LOSO Fold")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")

    max_total = float(plot_frame["total_windows"].max()) if not plot_frame.empty else 0.0
    text_offset = max(max_total * 0.01, 8.0)
    for position, row in enumerate(plot_frame.itertuples(index=False)):
        if not np.isfinite(row.total_windows):
            continue
        axis.text(
            float(row.total_windows) + text_offset,
            position,
            f"Ictal {row.ictal_fraction:.1%}",
            va="center",
            ha="left",
            fontsize=9,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_top_curves(
    *,
    leaderboard: pd.DataFrame,
    experiments_root: Path,
    output_path: Path,
    curve_kind: str,
) -> Path:
    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=(11, 8))
    color_cycle = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, max(len(leaderboard), 1)))

    for color, row in zip(color_cycle, leaderboard.itertuples(index=False)):
        curve_source, group_col = _load_curve_source(
            experiments_root=experiments_root,
            family=str(row.family),
            preset=str(row.preset),
        )
        prepared = _prepare_curve_groups(curve_source, group_col=group_col)
        if curve_kind == "roc":
            x_values, y_values = _mean_roc_curve(prepared, group_col=group_col)
            metric_name = "roc_auc"
            title = "ROC Curves for Top 8 Cross-Family Models"
            x_label = "False Positive Rate"
            y_label = "True Positive Rate"
        else:
            x_values, y_values = _mean_pr_curve(prepared, group_col=group_col)
            metric_name = "average_precision"
            title = "Precision-Recall Curves for Top 8 Cross-Family Models"
            x_label = "Recall"
            y_label = "Precision"

        family_label = PAPER_FAMILY_LABELS.get(str(row.family).upper(), str(row.family).upper())
        line_style = PAPER_LINE_STYLES.get(str(row.family).upper(), "-")
        metric_value = getattr(row, metric_name)
        label = f"{_pretty_model_name(str(row.preset))} [{family_label}] ({metric_value:.3f})"
        axis.plot(x_values, y_values, color=color, linewidth=2.0, linestyle=line_style, label=label)

    if curve_kind == "roc":
        axis.plot([0.0, 1.0], [0.0, 1.0], color="#666666", linewidth=1.0, linestyle="--", alpha=0.7)

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_paper_figures(
    *,
    workspace_root: str | None,
    plot_dir: str | None,
    leaderboard_file: str | None,
    fold_metrics_file: str | None,
    top_k: int,
) -> list[Path]:
    if top_k <= 0:
        raise ValueError("--top-k must be a positive integer.")

    paths = resolve_workspace_paths(workspace_root)
    experiments_root = paths.outputs_dir / "experiments"
    output_dir = Path(plot_dir).resolve() if plot_dir is not None else experiments_root / "summary" / "plots"

    leaderboard = _load_paper_leaderboard(workspace_root=workspace_root, leaderboard_file=leaderboard_file)
    top_models = leaderboard.head(top_k).copy()

    fold_metrics_path = (
        Path(fold_metrics_file).resolve()
        if fold_metrics_file is not None
        else experiments_root / "ml" / "catboost" / "fold_metrics.csv"
    )
    if not fold_metrics_path.exists():
        raise FileNotFoundError(f"Missing LOSO fold metrics file: {fold_metrics_path.as_posix()}")

    loso_counts = _load_loso_fold_counts(fold_metrics_path)
    created_paths = [
        _render_loso_class_balance(plot_frame=loso_counts, output_path=output_dir / "loso_class_balance.png"),
        _render_top_curves(
            leaderboard=top_models,
            experiments_root=experiments_root,
            output_path=output_dir / "top8_roc_curves.png",
            curve_kind="roc",
        ),
        _render_top_curves(
            leaderboard=top_models,
            experiments_root=experiments_root,
            output_path=output_dir / "top8_pr_curves.png",
            curve_kind="pr",
        ),
    ]
    return created_paths


def _read_metric_means_from_aggregate(aggregate_path: Path) -> dict[str, float]:
    frame = pd.read_csv(aggregate_path)
    if "metric" not in frame.columns or "mean" not in frame.columns:
        return {}

    metrics: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        metric_name = str(getattr(row, "metric", ""))
        mean_value = pd.to_numeric(pd.Series([getattr(row, "mean", float("nan"))]), errors="coerce").iloc[0]
        if pd.notna(mean_value):
            metrics[metric_name] = float(mean_value)
    return metrics


def _mean_numeric_column(frame: pd.DataFrame, column_name: str) -> float:
    if column_name not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column_name], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _collect_classifier_family_rows(*, family_name: str, family_dir: Path) -> list[dict[str, Any]]:
    if not family_dir.exists():
        return []

    rows: list[dict[str, Any]] = []
    for preset_dir in sorted(path for path in family_dir.iterdir() if path.is_dir()):
        aggregate_path = preset_dir / "aggregate_metrics.csv"
        if not aggregate_path.exists():
            continue

        metric_means = _read_metric_means_from_aggregate(aggregate_path)
        rows.append(
            {
                "family": family_name.upper(),
                "preset": preset_dir.name,
                "roc_auc": metric_means.get("roc_auc", float("nan")),
                "average_precision": metric_means.get("average_precision", float("nan")),
                "f1": metric_means.get("f1", float("nan")),
                "precision": metric_means.get("precision", float("nan")),
                "sensitivity": metric_means.get("sensitivity", float("nan")),
                "specificity": metric_means.get("specificity", float("nan")),
                "accuracy": metric_means.get("accuracy", float("nan")),
                "n_runs": float("nan"),
                "note": "native_classifier",
                "source_file": aggregate_path.as_posix(),
            }
        )
    return rows


def _collect_timeseries_bridge_rows(timeseries_dir: Path) -> list[dict[str, Any]]:
    if not timeseries_dir.exists():
        return []

    rows: list[dict[str, Any]] = []
    for preset_dir in sorted(path for path in timeseries_dir.iterdir() if path.is_dir()):
        clf_path = preset_dir / "sarima_classification_metrics.csv"
        if not clf_path.exists():
            continue

        clf_frame = pd.read_csv(clf_path)
        if clf_frame.empty:
            continue

        n_runs = int(clf_frame["run_id"].nunique()) if "run_id" in clf_frame.columns else int(len(clf_frame))
        rows.append(
            {
                "family": "TIMESERIES",
                "preset": preset_dir.name,
                "roc_auc": _mean_numeric_column(clf_frame, "roc_auc"),
                "average_precision": _mean_numeric_column(clf_frame, "average_precision"),
                "f1": _mean_numeric_column(clf_frame, "f1"),
                "precision": _mean_numeric_column(clf_frame, "precision"),
                "sensitivity": _mean_numeric_column(clf_frame, "sensitivity"),
                "specificity": _mean_numeric_column(clf_frame, "specificity"),
                "accuracy": _mean_numeric_column(clf_frame, "accuracy"),
                "n_runs": float(n_runs),
                "note": "derived_from_residuals",
                "source_file": clf_path.as_posix(),
            }
        )
    return rows


def build_cross_family_leaderboard(*, workspace_root: str | None, output_file: str | None) -> Path:
    paths = resolve_workspace_paths(workspace_root)
    experiments_root = paths.outputs_dir / "experiments"

    rows: list[dict[str, Any]] = []
    rows.extend(_collect_classifier_family_rows(family_name="ml", family_dir=experiments_root / "ml"))
    rows.extend(_collect_classifier_family_rows(family_name="dl", family_dir=experiments_root / "dl"))
    rows.extend(_collect_timeseries_bridge_rows(experiments_root / "timeseries"))

    if not rows:
        raise RuntimeError("No experiment metrics were found to build cross-family leaderboard.")

    leaderboard = pd.DataFrame(rows)
    sort_columns = [column for column in ("roc_auc", "average_precision", "f1") if column in leaderboard.columns]
    if sort_columns:
        leaderboard = leaderboard.sort_values(sort_columns, ascending=[False] * len(sort_columns), kind="mergesort")

    output_path = Path(output_file).resolve() if output_file is not None else experiments_root / "summary" / "cross_family_leaderboard.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(output_path, index=False)
    return output_path


def _render_family_plot(*, family: str, spec: dict[str, Any], summary_dir: Path, plot_dir: Path) -> Path | None:
    summary_path = summary_dir / str(spec["summary_file"])
    if not summary_path.exists():
        return None

    frame = pd.read_csv(summary_path)
    if frame.empty:
        return None

    metrics = [str(metric) for metric in spec.get("plot_metrics", []) if str(metric) in frame.columns]
    if not metrics:
        return None

    ascending_metrics = {str(metric) for metric in spec.get("plot_ascending", [])}
    color = str(spec.get("plot_color", "#1f77b4"))

    plt = _load_pyplot()
    n_rows = math.ceil(len(metrics) / 2)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, max(4, 4 * n_rows)), squeeze=False)
    axes_flat = list(axes.flat)

    for axis, metric_name in zip(axes_flat, metrics):
        metric_frame = frame.loc[:, ["model", metric_name]].dropna().copy()
        if metric_frame.empty:
            axis.set_visible(False)
            continue
        metric_frame = metric_frame.sort_values(
            metric_name,
            ascending=metric_name in ascending_metrics,
            kind="mergesort",
        )
        axis.barh(metric_frame["model"], metric_frame[metric_name], color=color)
        axis.set_title(metric_name.replace("_", " "))
        axis.grid(axis="x", alpha=0.25)

    for axis in axes_flat[len(metrics) :]:
        axis.set_visible(False)

    fig.suptitle(f"{family.upper()} metric overview", fontsize=14)
    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / str(spec.get("plot_file", f"{family}_metrics_overview.png"))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = build_parser().parse_args()

    if args.task == "cross_family_leaderboard":
        output_path = build_cross_family_leaderboard(
            workspace_root=args.workspace_root,
            output_file=args.output_file,
        )
        print(f"Cross-family leaderboard: {output_path.as_posix()}")
        return 0

    if args.task == "paper_figures":
        created_paths = build_paper_figures(
            workspace_root=args.workspace_root,
            plot_dir=args.plot_dir,
            leaderboard_file=args.leaderboard_file,
            fold_metrics_file=args.fold_metrics_file,
            top_k=args.top_k,
        )
        for output_path in created_paths:
            print(f"Plot: {output_path.as_posix()}")
        return 0

    resolved_config, family_order, family_specs, _ = _resolve_report_config(args.config)

    if args.task == "summarize":
        paths = resolve_workspace_paths(args.workspace_root)
        command = [
            "--workspace-root",
            paths.workspace.as_posix(),
            "--family",
            args.family,
            "--report-config",
            resolved_config.as_posix(),
        ]
        if args.output_dir is not None:
            command.extend(["--output-dir", str(Path(args.output_dir).resolve())])
        run_python_script("summarize_experiment_metrics.py", *command)
        return 0

    if args.task == "verify":
        paths = resolve_workspace_paths(args.workspace_root)
        command = [
            "--workspace-root",
            paths.workspace.as_posix(),
            "--family",
            args.family,
            "--report-config",
            resolved_config.as_posix(),
        ]
        if args.output_dir is not None:
            command.extend(["--output-dir", str(Path(args.output_dir).resolve())])
        command.append("--fail-on-error" if args.fail_on_error else "--no-fail-on-error")
        run_python_script("verify_experiment_outputs.py", *command)
        return 0

    summary_dir = _summary_dir(workspace_root=args.workspace_root, summary_dir=args.summary_dir)
    if not summary_dir.exists():
        raise FileNotFoundError(
            f"Missing summary directory: {summary_dir.as_posix()}. Run `python tools/workspace_reports.py summarize` first."
        )

    plot_dir = Path(args.plot_dir).resolve() if args.plot_dir is not None else summary_dir / "plots"
    families = _selected_families(args.family, family_order)
    created_paths: list[Path] = []
    for family in families:
        output_path = _render_family_plot(
            family=family,
            spec=family_specs[family],
            summary_dir=summary_dir,
            plot_dir=plot_dir,
        )
        if output_path is not None:
            created_paths.append(output_path)

    if not created_paths:
        raise RuntimeError(
            "No plots were created. Confirm the summary CSV files exist and contain completed experiment rows."
        )

    print(f"Report config: {resolved_config.as_posix()}")
    print(f"Summary dir: {summary_dir.as_posix()}")
    for output_path in created_paths:
        print(f"Plot: {output_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())