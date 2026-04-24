from __future__ import annotations

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from ds003029_eda.data.preprocess import PreprocessConfig
from ds003029_eda.features.windowing import WindowingConfig
from ds003029_eda.labels.labeler import LabelingConfig
from ds003029_eda.paths import get_paths
from ds003029_eda.pipelines.run_features import FeaturePipelineConfig, run_feature_pipeline
from ds003029_eda.pipelines.run_preprocess import PreprocessPipelineConfig, run_preprocess_pipeline
from ds003029_eda.pipelines.run_sarima_prep import SarimaPrepConfig, run_sarima_prep


def _has_real_eeg_data(workspace_root: Path) -> bool:
    eeg_root = workspace_root / "EEG" / "ds003029"
    if not eeg_root.exists():
        return False
    try:
        next(eeg_root.rglob("*.eeg"))
    except StopIteration:
        return False
    return True


def _resolve_workspace_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()

    for candidate in [repo_root.parent, repo_root]:
        if _has_real_eeg_data(candidate):
            return candidate.resolve()

    for candidate in [repo_root.parent, repo_root]:
        if (candidate / "EEG" / "ds003029").exists():
            return candidate.resolve()

    return repo_root.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ds003029 data processing v2 for preprocessing and feature generation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--workspace-root", default=None)
        command_parser.add_argument("--artifact-subdir", default="data_processing_v2")
        command_parser.add_argument("--overwrite", action="store_true")

    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess BrainVision runs and export QC artifacts.")
    add_common_arguments(preprocess_parser)
    preprocess_parser.add_argument("--target-sfreq", type=float, default=256.0)
    preprocess_parser.add_argument("--bandpass-low", type=float, default=0.5)
    preprocess_parser.add_argument("--bandpass-high", type=float, default=120.0)
    preprocess_parser.add_argument("--notch-harmonics", type=int, default=3)
    preprocess_parser.add_argument("--reference-mode", default="average")

    features_parser = subparsers.add_parser("features", help="Build per-run feature tensors and LOSO folds from preprocessed runs.")
    add_common_arguments(features_parser)
    features_parser.add_argument("--window-sec", type=float, default=2.0)
    features_parser.add_argument("--step-sec", type=float, default=0.5)
    features_parser.add_argument("--label-margin-sec", type=float, default=0.5)
    features_parser.add_argument("--fill-missing-after-scaling", type=float, default=0.0)

    sarima_parser = subparsers.add_parser(
        "sarima-prep",
        help="Export v2 run tensors into SARIMA-ready per-run and combined CSV files.",
    )
    add_common_arguments(sarima_parser)
    sarima_parser.add_argument("--output-subdir", default="sarima")
    sarima_parser.add_argument("--aggregate-feature-name", default="agg_mean_rms")

    full_parser = subparsers.add_parser("full", help="Run preprocessing then feature generation end to end.")
    add_common_arguments(full_parser)
    full_parser.add_argument("--target-sfreq", type=float, default=256.0)
    full_parser.add_argument("--bandpass-low", type=float, default=0.5)
    full_parser.add_argument("--bandpass-high", type=float, default=120.0)
    full_parser.add_argument("--notch-harmonics", type=int, default=3)
    full_parser.add_argument("--reference-mode", default="average")
    full_parser.add_argument("--window-sec", type=float, default=2.0)
    full_parser.add_argument("--step-sec", type=float, default=0.5)
    full_parser.add_argument("--label-margin-sec", type=float, default=0.5)
    full_parser.add_argument("--fill-missing-after-scaling", type=float, default=0.0)
    return parser


def _build_preprocess_config(args: argparse.Namespace) -> PreprocessPipelineConfig:
    return PreprocessPipelineConfig(
        artifact_subdir=args.artifact_subdir,
        overwrite=args.overwrite,
        preprocess=PreprocessConfig(
            target_sfreq=args.target_sfreq,
            bandpass_low_hz=args.bandpass_low,
            bandpass_high_hz=args.bandpass_high,
            notch_harmonics=args.notch_harmonics,
            reference_mode=args.reference_mode,
        ),
    )


def _build_feature_config(args: argparse.Namespace) -> FeaturePipelineConfig:
    return FeaturePipelineConfig(
        artifact_subdir=args.artifact_subdir,
        overwrite=args.overwrite,
        windowing=WindowingConfig(window_sec=args.window_sec, step_sec=args.step_sec),
        labeling=LabelingConfig(boundary_margin_sec=args.label_margin_sec),
        fill_missing_after_scaling=args.fill_missing_after_scaling,
    )


def _build_sarima_prep_config(args: argparse.Namespace) -> SarimaPrepConfig:
    return SarimaPrepConfig(
        artifact_subdir=args.artifact_subdir,
        output_subdir=args.output_subdir,
        aggregate_feature_name=args.aggregate_feature_name,
        overwrite=args.overwrite,
    )


def main() -> int:
    args = build_parser().parse_args()
    workspace_root = _resolve_workspace_root(args.workspace_root)
    paths = get_paths(workspace_root)

    if args.command == "preprocess":
        summary_df, _, artifact_root = run_preprocess_pipeline(paths=paths, config=_build_preprocess_config(args))
        print(f"Workspace root: {workspace_root.as_posix()}")
        print(f"Preprocessing completed: {summary_df['preprocess_ok'].sum()} / {len(summary_df)} runs")
        print(f"Artifact root: {artifact_root.as_posix()}")
        return 0

    if args.command == "features":
        run_inventory, fold_manifest_df, artifact_root = run_feature_pipeline(paths=paths, config=_build_feature_config(args))
        print(f"Workspace root: {workspace_root.as_posix()}")
        print(f"Feature extraction completed: {run_inventory['feature_ok'].sum()} / {len(run_inventory)} runs")
        print(f"Generated folds: {len(fold_manifest_df)}")
        print(f"Artifact root: {artifact_root.as_posix()}")
        return 0

    if args.command == "sarima-prep":
        combined_df, manifest_df, output_root = run_sarima_prep(paths=paths, config=_build_sarima_prep_config(args))
        print(f"Workspace root: {workspace_root.as_posix()}")
        print(f"SARIMA prep completed: {len(manifest_df)} series, {len(combined_df)} rows")
        print(f"Output root: {output_root.as_posix()}")
        return 0

    if args.command == "full":
        run_preprocess_pipeline(paths=paths, config=_build_preprocess_config(args))
        run_inventory, fold_manifest_df, artifact_root = run_feature_pipeline(paths=paths, config=_build_feature_config(args))
        print(f"Workspace root: {workspace_root.as_posix()}")
        print(f"End-to-end processing completed: {run_inventory['feature_ok'].sum()} runs, {len(fold_manifest_df)} folds")
        print(f"Artifact root: {artifact_root.as_posix()}")
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())