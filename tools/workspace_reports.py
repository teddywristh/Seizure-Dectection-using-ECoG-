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