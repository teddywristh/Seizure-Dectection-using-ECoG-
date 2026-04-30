from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from _workspace_cli import load_family_report_config, resolve_workspace_paths, run_python_script


DEFAULT_REPORT_CONFIG = "configs/reports/family_reports.json"
FAMILY_CHOICES = ("all", "ml", "dl", "timeseries")


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