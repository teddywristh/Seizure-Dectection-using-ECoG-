from __future__ import annotations

import argparse

from _workspace_cli import resolve_workspace_paths, run_python_script


def _add_processing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--artifact-subdir", default="data_processing_v2")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target-sfreq", type=float, default=256.0)
    parser.add_argument("--bandpass-low", type=float, default=0.5)
    parser.add_argument("--bandpass-high", type=float, default=120.0)
    parser.add_argument("--notch-harmonics", type=int, default=3)
    parser.add_argument("--reference-mode", default="average")
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument("--label-margin-sec", type=float, default=0.5)
    parser.add_argument("--fill-missing-after-scaling", type=float, default=0.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task 2: process the current content-ready runs for one modeling direction at a time. "
            "This keeps preprocessing separate from experiment execution."
        )
    )
    subparsers = parser.add_subparsers(dest="direction", required=True)

    timeseries_parser = subparsers.add_parser("timeseries", help="Preprocess, build features, then export SARIMA-ready inputs.")
    _add_processing_arguments(timeseries_parser)
    timeseries_parser.add_argument("--sarima-output-subdir", default="sarima")
    timeseries_parser.add_argument("--aggregate-feature-name", default="agg_mean_rms")
    timeseries_parser.add_argument("--exogenous-feature-name", nargs="*", default=())

    ml_parser = subparsers.add_parser("ml", help="Preprocess and build LOSO fold features for ML models.")
    _add_processing_arguments(ml_parser)

    dl_parser = subparsers.add_parser("dl", help="Preprocess, build fold features, and optionally export raw fold tensors for DL models.")
    _add_processing_arguments(dl_parser)
    dl_parser.add_argument("--skip-raw-export", action="store_true")
    dl_parser.add_argument("--raw-output-subdir", default="raw_folds")
    dl_parser.add_argument("--raw-window-sec", type=float, default=2.0)
    dl_parser.add_argument("--raw-output-dtype", choices=["float16", "float32"], default="float16")
    return parser


def _preprocess_args(args: argparse.Namespace, workspace_root: str) -> list[str]:
    command = [
        "preprocess",
        "--workspace-root",
        workspace_root,
        "--artifact-subdir",
        args.artifact_subdir,
        "--target-sfreq",
        str(args.target_sfreq),
        "--bandpass-low",
        str(args.bandpass_low),
        "--bandpass-high",
        str(args.bandpass_high),
        "--notch-harmonics",
        str(args.notch_harmonics),
        "--reference-mode",
        args.reference_mode,
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def _feature_args(args: argparse.Namespace, workspace_root: str) -> list[str]:
    command = [
        "features",
        "--workspace-root",
        workspace_root,
        "--artifact-subdir",
        args.artifact_subdir,
        "--window-sec",
        str(args.window_sec),
        "--step-sec",
        str(args.step_sec),
        "--label-margin-sec",
        str(args.label_margin_sec),
        "--fill-missing-after-scaling",
        str(args.fill_missing_after_scaling),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def main() -> int:
    args = build_parser().parse_args()
    paths = resolve_workspace_paths(args.workspace_root)
    workspace_root = paths.workspace.as_posix()

    run_python_script("run_data_processing_v2.py", *_preprocess_args(args, workspace_root))
    run_python_script("run_data_processing_v2.py", *_feature_args(args, workspace_root))

    if args.direction == "timeseries":
        sarima_command = [
            "sarima-prep",
            "--workspace-root",
            workspace_root,
            "--artifact-subdir",
            args.artifact_subdir,
            "--output-subdir",
            args.sarima_output_subdir,
            "--aggregate-feature-name",
            args.aggregate_feature_name,
        ]
        if args.overwrite:
            sarima_command.append("--overwrite")
        if args.exogenous_feature_name:
            sarima_command.append("--exogenous-feature-names")
            sarima_command.extend(args.exogenous_feature_name)
        run_python_script("run_data_processing_v2.py", *sarima_command)

    if args.direction == "dl" and not args.skip_raw_export:
        raw_command = [
            "--workspace-root",
            workspace_root,
            "--artifact-subdir",
            args.artifact_subdir,
            "--output-subdir",
            args.raw_output_subdir,
            "--window-sec",
            str(args.raw_window_sec),
            "--output-dtype",
            args.raw_output_dtype,
        ]
        if args.overwrite:
            raw_command.append("--overwrite")
        run_python_script("build_raw_folds.py", *raw_command)

    print(f"Workspace root: {workspace_root}")
    print(f"Direction: {args.direction}")
    print(f"Artifacts: {(paths.outputs_dir / args.artifact_subdir).as_posix()}")
    if args.direction == "timeseries":
        print(f"SARIMA inputs: {(paths.outputs_dir / args.artifact_subdir / args.sarima_output_subdir).as_posix()}")
    if args.direction == "dl" and not args.skip_raw_export:
        print(f"Raw fold tensors: {(paths.outputs_dir / args.artifact_subdir / args.raw_output_subdir).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())