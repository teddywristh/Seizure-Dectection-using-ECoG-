from __future__ import annotations

import argparse

import pandas as pd

from _workspace_cli import resolve_workspace_paths, run_python_script


CONTENT_MANIFEST_NAME = "ds003029_content_run_manifest.csv"
READY_MANIFEST_NAME = "ds003029_model_ready_run_manifest.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task 1: refresh metadata tables and export the current run manifests for raw-content runs. "
            "By default this validates the current 16-run content subset."
        )
    )
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--expected-content-runs", type=int, default=16)
    parser.add_argument("--expected-ready-runs", type=int, default=16)
    parser.add_argument(
        "--allow-any-count",
        action="store_true",
        help="Do not fail when the current content-ready run counts differ from the default 16-run expectation.",
    )
    return parser


def _export_manifests(paths) -> tuple[int, int]:
    run_summary_path = paths.outputs_dir / "ds003029_run_summary.csv"
    seizure_intervals_path = paths.outputs_dir / "ds003029_seizure_intervals_by_run.csv"

    run_summary = pd.read_csv(run_summary_path)
    seizure_intervals = pd.read_csv(seizure_intervals_path)

    if "eeg_content_present" not in run_summary.columns:
        raise KeyError(f"Missing 'eeg_content_present' column in {run_summary_path.as_posix()}.")

    content_runs = run_summary[run_summary["eeg_content_present"].fillna(False).astype(bool)].copy()
    content_runs = content_runs.sort_values(["subject", "session", "run", "base"], kind="mergesort").reset_index(drop=True)

    ready_bases = set(seizure_intervals["base"].astype(str).tolist())
    ready_runs = content_runs[content_runs["base"].astype(str).isin(ready_bases)].copy()
    ready_runs["n_intervals"] = ready_runs["base"].map(seizure_intervals.groupby("base").size()).fillna(0).astype(int)
    ready_runs = ready_runs.sort_values(["subject", "session", "run", "base"], kind="mergesort").reset_index(drop=True)

    content_runs.to_csv(paths.outputs_dir / CONTENT_MANIFEST_NAME, index=False)
    ready_runs.to_csv(paths.outputs_dir / READY_MANIFEST_NAME, index=False)
    return len(content_runs), len(ready_runs)


def main() -> int:
    args = build_parser().parse_args()
    paths = resolve_workspace_paths(args.workspace_root)

    run_python_script("build_run_summary_ds003029.py", "--workspace-root", paths.workspace.as_posix())
    run_python_script("build_marker_qc_ds003029.py", "--workspace-root", paths.workspace.as_posix())

    n_content_runs, n_ready_runs = _export_manifests(paths)

    print(f"Workspace root: {paths.workspace.as_posix()}")
    print(f"Content manifest: {(paths.outputs_dir / CONTENT_MANIFEST_NAME).as_posix()}")
    print(f"Model-ready manifest: {(paths.outputs_dir / READY_MANIFEST_NAME).as_posix()}")
    print(f"Runs with real raw content: {n_content_runs}")
    print(f"Runs ready for downstream processing: {n_ready_runs}")

    if args.allow_any_count:
        return 0

    if n_content_runs != args.expected_content_runs:
        print(
            "Unexpected number of raw-content runs: "
            f"expected {args.expected_content_runs}, found {n_content_runs}."
        )
        return 1

    if n_ready_runs != args.expected_ready_runs:
        print(
            "Unexpected number of model-ready runs: "
            f"expected {args.expected_ready_runs}, found {n_ready_runs}."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())