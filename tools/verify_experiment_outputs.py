from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from _workspace_cli import load_family_report_config

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from ds003029_eda.paths import get_paths, resolve_workspace_root


AGGREGATE_COLUMNS = ("metric", "mean", "std", "min", "max")
TIMESERIES_KEY_METRICS = ("rmse_test", "mae_test", "smape_test_pct", "r2_test")
CLASSIFICATION_KEY_METRICS = ("roc_auc", "average_precision", "f1", "accuracy")
DEFAULT_REPORT_CONFIG = "configs/reports/family_reports.json"
FAMILY_CHOICES = ("all", "ml", "dl", "timeseries")


@dataclass
class ValidationIssue:
    severity: str
    family: str
    preset: str
    experiment_name: str
    check_name: str
    message: str


class ExperimentValidator:
    def __init__(self, expected_folds: set[str]) -> None:
        self.expected_folds = set(expected_folds)
        self.issues: list[ValidationIssue] = []

    def _report_issue(
        self,
        *,
        severity: str,
        family: str,
        preset: str,
        experiment_name: str,
        check_name: str,
        message: str,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                severity=severity,
                family=family,
                preset=preset,
                experiment_name=experiment_name,
                check_name=check_name,
                message=message,
            )
        )

    def _report_error(
        self,
        *,
        family: str,
        preset: str,
        experiment_name: str,
        check_name: str,
        message: str,
    ) -> None:
        self._report_issue(
            severity="error",
            family=family,
            preset=preset,
            experiment_name=experiment_name,
            check_name=check_name,
            message=message,
        )

    def _report_warning(
        self,
        *,
        family: str,
        preset: str,
        experiment_name: str,
        check_name: str,
        message: str,
    ) -> None:
        self._report_issue(
            severity="warning",
            family=family,
            preset=preset,
            experiment_name=experiment_name,
            check_name=check_name,
            message=message,
        )

    def _result_base(self, *, family: str, preset: str, experiment_name: str) -> dict[str, Any]:
        return {
            "family": family,
            "preset": preset,
            "experiment_name": experiment_name,
            "status": "ok",
            "n_errors": 0,
            "n_warnings": 0,
        }

    def _finalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        key = (
            result["family"],
            result["preset"],
            result["experiment_name"],
        )
        n_errors = 0
        n_warnings = 0
        for issue in self.issues:
            issue_key = (issue.family, issue.preset, issue.experiment_name)
            if issue_key != key:
                continue
            if issue.severity == "error":
                n_errors += 1
            else:
                n_warnings += 1

        result["n_errors"] = n_errors
        result["n_warnings"] = n_warnings
        if n_errors > 0:
            result["status"] = "error"
        elif n_warnings > 0:
            result["status"] = "warning"
        else:
            result["status"] = "ok"
        return result

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _validate_aggregate_metrics_file(
        self,
        *,
        path: Path,
        family: str,
        preset: str,
        experiment_name: str,
    ) -> pd.DataFrame:
        if not path.exists():
            self._report_error(
                family=family,
                preset=preset,
                experiment_name=experiment_name,
                check_name="aggregate_metrics_exists",
                message=f"Missing required file: {path.as_posix()}",
            )
            return pd.DataFrame()

        frame = pd.read_csv(path)
        missing_columns = [column for column in AGGREGATE_COLUMNS if column not in frame.columns]
        if missing_columns:
            self._report_error(
                family=family,
                preset=preset,
                experiment_name=experiment_name,
                check_name="aggregate_metrics_columns",
                message=f"aggregate_metrics.csv is missing columns: {missing_columns}",
            )
            return frame

        if frame.empty:
            self._report_error(
                family=family,
                preset=preset,
                experiment_name=experiment_name,
                check_name="aggregate_metrics_non_empty",
                message="aggregate_metrics.csv is empty.",
            )
            return frame

        for numeric_column in ("mean", "std", "min", "max"):
            frame[numeric_column] = pd.to_numeric(frame[numeric_column], errors="coerce")
            if frame[numeric_column].isna().all():
                self._report_warning(
                    family=family,
                    preset=preset,
                    experiment_name=experiment_name,
                    check_name=f"aggregate_metrics_{numeric_column}_nan",
                    message=f"Column '{numeric_column}' is all NaN in aggregate_metrics.csv.",
                )
        return frame

    def _validate_predictions_directory(
        self,
        *,
        predictions_dir: Path,
        family: str,
        preset: str,
        experiment_name: str,
    ) -> set[str]:
        if not predictions_dir.exists():
            self._report_error(
                family=family,
                preset=preset,
                experiment_name=experiment_name,
                check_name="predictions_directory_exists",
                message=f"Missing predictions directory: {predictions_dir.as_posix()}",
            )
            return set()

        fold_files = sorted(predictions_dir.glob("fold_*_predictions.csv"))
        observed_folds = {
            file_path.name[: -len("_predictions.csv")]
            if file_path.name.endswith("_predictions.csv")
            else file_path.stem
            for file_path in fold_files
        }

        missing_folds = sorted(self.expected_folds - observed_folds)
        extra_folds = sorted(observed_folds - self.expected_folds)
        if missing_folds:
            self._report_error(
                family=family,
                preset=preset,
                experiment_name=experiment_name,
                check_name="prediction_fold_files_missing",
                message=f"Missing prediction files for folds: {missing_folds}",
            )
        if extra_folds:
            self._report_warning(
                family=family,
                preset=preset,
                experiment_name=experiment_name,
                check_name="prediction_fold_files_extra",
                message=f"Unexpected prediction files for folds: {extra_folds}",
            )
        return observed_folds

    def _validate_checkpoints_directory(
        self,
        *,
        checkpoints_dir: Path,
        extension: str,
        family: str,
        preset: str,
        experiment_name: str,
    ) -> None:
        if not checkpoints_dir.exists():
            self._report_error(
                family=family,
                preset=preset,
                experiment_name=experiment_name,
                check_name="checkpoints_directory_exists",
                message=f"Missing checkpoints directory: {checkpoints_dir.as_posix()}",
            )
            return

        files = sorted(checkpoints_dir.glob(f"*{extension}"))
        if not files:
            self._report_error(
                family=family,
                preset=preset,
                experiment_name=experiment_name,
                check_name="checkpoint_files_exist",
                message=f"No checkpoint files (*{extension}) found in {checkpoints_dir.as_posix()}.",
            )
            return

        missing_checkpoint_folds: list[str] = []
        for fold_id in sorted(self.expected_folds):
            found_for_fold = any(path.name.startswith(f"{fold_id}_") for path in files)
            if not found_for_fold:
                missing_checkpoint_folds.append(fold_id)

        if missing_checkpoint_folds:
            self._report_error(
                family=family,
                preset=preset,
                experiment_name=experiment_name,
                check_name="checkpoint_fold_files_missing",
                message=f"Missing checkpoints for folds: {missing_checkpoint_folds}",
            )

    def _validate_fold_metrics(
        self,
        *,
        fold_metrics_path: Path,
        family: str,
        preset: str,
        experiment_name: str,
    ) -> pd.DataFrame:
        if not fold_metrics_path.exists():
            self._report_error(
                family=family,
                preset=preset,
                experiment_name=experiment_name,
                check_name="fold_metrics_exists",
                message=f"Missing required file: {fold_metrics_path.as_posix()}",
            )
            return pd.DataFrame()

        frame = pd.read_csv(fold_metrics_path)
        if frame.empty:
            self._report_error(
                family=family,
                preset=preset,
                experiment_name=experiment_name,
                check_name="fold_metrics_non_empty",
                message="fold_metrics.csv is empty.",
            )
            return frame

        if "fold_id" not in frame.columns:
            self._report_error(
                family=family,
                preset=preset,
                experiment_name=experiment_name,
                check_name="fold_metrics_fold_id_column",
                message="fold_metrics.csv is missing required column 'fold_id'.",
            )
            return frame

        observed_folds = set(frame["fold_id"].astype(str).tolist())
        missing_folds = sorted(self.expected_folds - observed_folds)
        extra_folds = sorted(observed_folds - self.expected_folds)
        if missing_folds:
            self._report_error(
                family=family,
                preset=preset,
                experiment_name=experiment_name,
                check_name="fold_metrics_missing_folds",
                message=f"fold_metrics.csv is missing folds: {missing_folds}",
            )
        if extra_folds:
            self._report_warning(
                family=family,
                preset=preset,
                experiment_name=experiment_name,
                check_name="fold_metrics_extra_folds",
                message=f"fold_metrics.csv contains unexpected folds: {extra_folds}",
            )
        return frame

    def validate_timeseries_experiment(
        self,
        *,
        experiment_dir: Path,
        family: str,
        preset: str,
        experiment_name: str,
    ) -> dict[str, Any]:
        result = self._result_base(family=family, preset=preset, experiment_name=experiment_name)

        for required_name in ("run_config.json", "series_metrics.csv"):
            required_path = experiment_dir / required_name
            if not required_path.exists():
                self._report_error(
                    family=family,
                    preset=preset,
                    experiment_name=experiment_name,
                    check_name="required_file_exists",
                    message=f"Missing required file: {required_path.as_posix()}",
                )

        for required_dir_name in ("predictions", "checkpoints", "reports", "sarima_results"):
            required_dir = experiment_dir / required_dir_name
            if not required_dir.exists():
                self._report_warning(
                    family=family,
                    preset=preset,
                    experiment_name=experiment_name,
                    check_name="required_directory_exists",
                    message=f"Missing expected directory: {required_dir.as_posix()}",
                )

        series_path = experiment_dir / "series_metrics.csv"
        if series_path.exists():
            frame = pd.read_csv(series_path)
            result["n_series"] = int(len(frame))
            if frame.empty:
                self._report_error(
                    family=family,
                    preset=preset,
                    experiment_name=experiment_name,
                    check_name="series_metrics_non_empty",
                    message="series_metrics.csv is empty.",
                )
            else:
                if "status" in frame.columns:
                    ok_mask = frame["status"].fillna("").eq("ok")
                    ok_count = int(ok_mask.sum())
                    result["ok_series"] = ok_count
                    if ok_count == 0:
                        self._report_error(
                            family=family,
                            preset=preset,
                            experiment_name=experiment_name,
                            check_name="series_status_ok",
                            message="No successful series (status == 'ok') in series_metrics.csv.",
                        )
                    elif ok_count < len(frame):
                        self._report_warning(
                            family=family,
                            preset=preset,
                            experiment_name=experiment_name,
                            check_name="series_status_partial_ok",
                            message=f"Only {ok_count}/{len(frame)} series have status == 'ok'.",
                        )
                    eval_frame = frame.loc[ok_mask].copy() if ok_count > 0 else frame
                else:
                    self._report_warning(
                        family=family,
                        preset=preset,
                        experiment_name=experiment_name,
                        check_name="series_status_column_missing",
                        message="series_metrics.csv has no 'status' column.",
                    )
                    eval_frame = frame

                for metric_name in TIMESERIES_KEY_METRICS:
                    if metric_name not in eval_frame.columns:
                        continue
                    numeric_metric = pd.to_numeric(eval_frame[metric_name], errors="coerce").dropna()
                    if numeric_metric.empty:
                        self._report_warning(
                            family=family,
                            preset=preset,
                            experiment_name=experiment_name,
                            check_name=f"timeseries_metric_{metric_name}_nan",
                            message=f"Metric '{metric_name}' has no numeric values.",
                        )
                        continue
                    result[f"{metric_name}_mean"] = float(numeric_metric.mean())

        run_config_path = experiment_dir / "run_config.json"
        if run_config_path.exists() and series_path.exists():
            run_config = self._read_json(run_config_path)
            if "n_series" in run_config and "n_series" in result:
                if int(run_config["n_series"]) != int(result["n_series"]):
                    self._report_warning(
                        family=family,
                        preset=preset,
                        experiment_name=experiment_name,
                        check_name="run_config_series_count_mismatch",
                        message=(
                            "run_config.json n_series does not match series_metrics.csv "
                            f"({run_config['n_series']} vs {result['n_series']})."
                        ),
                    )

        return self._finalize_result(result)

    def validate_classification_experiment(
        self,
        *,
        experiment_dir: Path,
        family: str,
        preset: str,
        experiment_name: str,
    ) -> dict[str, Any]:
        result = self._result_base(family=family, preset=preset, experiment_name=experiment_name)

        required_files = [
            experiment_dir / "run_config.json",
            experiment_dir / "fold_metrics.csv",
            experiment_dir / "aggregate_metrics.csv",
            experiment_dir / "all_predictions.csv",
        ]
        if family == "dl":
            required_files.append(experiment_dir / "source_metadata.json")

        for file_path in required_files:
            if not file_path.exists():
                self._report_error(
                    family=family,
                    preset=preset,
                    experiment_name=experiment_name,
                    check_name="required_file_exists",
                    message=f"Missing required file: {file_path.as_posix()}",
                )

        for dir_name in ("checkpoints", "predictions", "reports"):
            dir_path = experiment_dir / dir_name
            if not dir_path.exists():
                self._report_error(
                    family=family,
                    preset=preset,
                    experiment_name=experiment_name,
                    check_name="required_directory_exists",
                    message=f"Missing required directory: {dir_path.as_posix()}",
                )

        fold_metrics_path = experiment_dir / "fold_metrics.csv"
        fold_metrics = self._validate_fold_metrics(
            fold_metrics_path=fold_metrics_path,
            family=family,
            preset=preset,
            experiment_name=experiment_name,
        )
        if not fold_metrics.empty and "fold_id" in fold_metrics.columns:
            result["n_folds"] = int(fold_metrics["fold_id"].nunique())
            for metric_name in CLASSIFICATION_KEY_METRICS:
                if metric_name not in fold_metrics.columns:
                    continue
                metric_values = pd.to_numeric(fold_metrics[metric_name], errors="coerce").dropna()
                if metric_values.empty:
                    self._report_warning(
                        family=family,
                        preset=preset,
                        experiment_name=experiment_name,
                        check_name=f"fold_metric_{metric_name}_nan",
                        message=f"Metric '{metric_name}' has no numeric values in fold_metrics.csv.",
                    )
                    continue
                result[f"{metric_name}_mean"] = float(metric_values.mean())

        aggregate_metrics = self._validate_aggregate_metrics_file(
            path=experiment_dir / "aggregate_metrics.csv",
            family=family,
            preset=preset,
            experiment_name=experiment_name,
        )
        if not aggregate_metrics.empty and "metric" in aggregate_metrics.columns:
            available_metrics = set(aggregate_metrics["metric"].astype(str).tolist())
            for key_metric in CLASSIFICATION_KEY_METRICS:
                if key_metric not in available_metrics:
                    self._report_warning(
                        family=family,
                        preset=preset,
                        experiment_name=experiment_name,
                        check_name="aggregate_metrics_missing_key_metric",
                        message=f"aggregate_metrics.csv does not include metric '{key_metric}'.",
                    )

        predictions_path = experiment_dir / "all_predictions.csv"
        if predictions_path.exists():
            all_predictions = pd.read_csv(predictions_path)
            result["n_predictions"] = int(len(all_predictions))
            if all_predictions.empty:
                self._report_error(
                    family=family,
                    preset=preset,
                    experiment_name=experiment_name,
                    check_name="all_predictions_non_empty",
                    message="all_predictions.csv is empty.",
                )
            else:
                required_prediction_columns = {"fold_id", "y_true", "y_score", "y_pred"}
                missing_prediction_columns = sorted(required_prediction_columns - set(all_predictions.columns))
                if missing_prediction_columns:
                    self._report_error(
                        family=family,
                        preset=preset,
                        experiment_name=experiment_name,
                        check_name="all_predictions_required_columns",
                        message=f"all_predictions.csv is missing columns: {missing_prediction_columns}",
                    )

        observed_prediction_folds = self._validate_predictions_directory(
            predictions_dir=experiment_dir / "predictions",
            family=family,
            preset=preset,
            experiment_name=experiment_name,
        )

        extension = ".joblib" if family == "ml" else ".pt"
        self._validate_checkpoints_directory(
            checkpoints_dir=experiment_dir / "checkpoints",
            extension=extension,
            family=family,
            preset=preset,
            experiment_name=experiment_name,
        )

        if fold_metrics_path.exists() and not fold_metrics.empty and "fold_id" in fold_metrics.columns:
            fold_metrics_folds = set(fold_metrics["fold_id"].astype(str).tolist())
            mismatch = sorted(fold_metrics_folds - observed_prediction_folds)
            if mismatch:
                self._report_warning(
                    family=family,
                    preset=preset,
                    experiment_name=experiment_name,
                    check_name="fold_metrics_prediction_fold_mismatch",
                    message=f"fold_metrics.csv contains folds without matching prediction files: {mismatch}",
                )

        return self._finalize_result(result)


def _load_expected_experiment_names(config_path: Path) -> dict[str, str]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    defaults = payload.get("defaults", {})
    presets = payload.get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError(f"Invalid preset config format in {config_path.as_posix()}: expected object under 'presets'.")

    expected: dict[str, str] = {}
    for preset_name, preset_payload in sorted(presets.items()):
        merged = dict(defaults)
        merged.update(preset_payload)
        experiment_name = str(merged.get("experiment_name", preset_name))
        expected[str(preset_name)] = experiment_name
    return expected


def _load_expected_folds(manifest_path: Path) -> set[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    fold_ids = {
        str(item["fold_id"])
        for item in payload
        if isinstance(item, dict) and "fold_id" in item
    }
    if not fold_ids:
        raise ValueError(f"No fold_id values found in {manifest_path.as_posix()}.")
    return fold_ids


def _render_markdown_report(
    *,
    experiments_root: Path,
    results_df: pd.DataFrame,
    issues_df: pd.DataFrame,
) -> str:
    lines: list[str] = [
        "# Experiment Output Verification",
        "",
        f"- Experiments root: {experiments_root.as_posix()}",
        f"- Total experiments checked: {len(results_df)}",
        f"- Errors: {int((issues_df['severity'] == 'error').sum()) if not issues_df.empty else 0}",
        f"- Warnings: {int((issues_df['severity'] == 'warning').sum()) if not issues_df.empty else 0}",
        "",
        "## Status by family",
        "",
    ]

    if results_df.empty:
        lines.append("No experiments were validated.")
    else:
        family_summary = (
            results_df.groupby(["family", "status"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["family", "status"], kind="mergesort")
        )
        lines.append("| family | status | count |")
        lines.append("|---|---|---|")
        for row in family_summary.itertuples(index=False):
            lines.append(f"| {row.family} | {row.status} | {row.count} |")

    lines.extend(["", "## Key metrics", ""])
    if not results_df.empty:
        ml_rows = results_df.loc[results_df["family"].eq("ml")]
        if "roc_auc_mean" in ml_rows.columns:
            ml_rows = ml_rows.dropna(subset=["roc_auc_mean"], how="any")
        else:
            ml_rows = pd.DataFrame()
        if not ml_rows.empty:
            best_ml = ml_rows.sort_values("roc_auc_mean", ascending=False, kind="mergesort").iloc[0]
            lines.append(f"- Best ML ROC AUC mean: {best_ml['experiment_name']} ({best_ml['roc_auc_mean']:.4f})")

        dl_rows = results_df.loc[results_df["family"].eq("dl")]
        if "roc_auc_mean" in dl_rows.columns:
            dl_rows = dl_rows.dropna(subset=["roc_auc_mean"], how="any")
        else:
            dl_rows = pd.DataFrame()
        if not dl_rows.empty:
            best_dl = dl_rows.sort_values("roc_auc_mean", ascending=False, kind="mergesort").iloc[0]
            lines.append(f"- Best DL ROC AUC mean: {best_dl['experiment_name']} ({best_dl['roc_auc_mean']:.4f})")

        ts_rows = results_df.loc[results_df["family"].eq("timeseries")]
        if "rmse_test_mean" in ts_rows.columns:
            ts_rows = ts_rows.dropna(subset=["rmse_test_mean"], how="any")
        else:
            ts_rows = pd.DataFrame()
        if not ts_rows.empty:
            best_ts = ts_rows.sort_values("rmse_test_mean", ascending=True, kind="mergesort").iloc[0]
            lines.append(f"- Best timeseries RMSE mean (lower is better): {best_ts['experiment_name']} ({best_ts['rmse_test_mean']:.6f})")

    lines.extend(["", "## Issues", ""])
    if issues_df.empty:
        lines.append("No issues detected.")
    else:
        lines.append("| severity | family | experiment | check | message |")
        lines.append("|---|---|---|---|---|")
        for row in issues_df.sort_values(["severity", "family", "experiment_name", "check_name"], kind="mergesort").itertuples(index=False):
            message = str(row.message).replace("|", "/")
            lines.append(f"| {row.severity} | {row.family} | {row.experiment_name} | {row.check_name} | {message} |")

    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify outputs of timeseries, ML, and DL experiments.")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--experiments-root", default=None)
    parser.add_argument("--family", choices=FAMILY_CHOICES, default="all")
    parser.add_argument("--report-config", default=DEFAULT_REPORT_CONFIG)
    parser.add_argument("--artifact-subdir", default="data_processing_v2")
    parser.add_argument(
        "--timeseries-config",
        default="configs/experiments/timeseries_models.json",
    )
    parser.add_argument(
        "--ml-config",
        default="configs/experiments/ml_models.json",
    )
    parser.add_argument(
        "--dl-config",
        default="configs/experiments/dl_models.json",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fail-on-error", dest="fail_on_error", action="store_true")
    parser.add_argument("--no-fail-on-error", dest="fail_on_error", action="store_false")
    parser.set_defaults(fail_on_error=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    workspace_root = resolve_workspace_root(args.workspace_root)
    paths = get_paths(workspace_root)
    experiments_root = (
        Path(args.experiments_root).resolve()
        if args.experiments_root is not None
        else (paths.outputs_dir / "experiments")
    )

    fold_manifest_path = paths.outputs_dir / args.artifact_subdir / "fold_manifest.json"
    if not fold_manifest_path.exists():
        raise FileNotFoundError(f"Missing fold manifest: {fold_manifest_path.as_posix()}")

    expected_folds = _load_expected_folds(fold_manifest_path)
    validator = ExperimentValidator(expected_folds=expected_folds)

    _resolved_report_config, family_order, family_specs, all_spec = load_family_report_config(args.report_config)
    selected_families = list(family_order) if args.family == "all" else [args.family]

    config_paths = {
        "timeseries": (repo_root / args.timeseries_config).resolve(),
        "ml": (repo_root / args.ml_config).resolve(),
        "dl": (repo_root / args.dl_config).resolve(),
    }
    expected_by_family: dict[str, dict[str, str]] = {}
    for family in selected_families:
        config_path = config_paths[family]
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file for {family}: {config_path.as_posix()}")
        expected_by_family[family] = _load_expected_experiment_names(config_path)

    results: list[dict[str, Any]] = []

    for family in selected_families:
        family_dir = experiments_root / family
        expected_map = expected_by_family[family]
        expected_experiment_names = set(expected_map.values())

        if not family_dir.exists():
            for preset_name, experiment_name in expected_map.items():
                validator._report_error(
                    family=family,
                    preset=preset_name,
                    experiment_name=experiment_name,
                    check_name="family_directory_exists",
                    message=f"Missing family output directory: {family_dir.as_posix()}",
                )
                results.append(
                    validator._finalize_result(
                        validator._result_base(
                            family=family,
                            preset=preset_name,
                            experiment_name=experiment_name,
                        )
                    )
                )
            continue

        observed_experiment_names = {
            path.name
            for path in family_dir.iterdir()
            if path.is_dir()
        }

        missing_experiments = sorted(expected_experiment_names - observed_experiment_names)
        extra_experiments = sorted(observed_experiment_names - expected_experiment_names)

        reverse_lookup = {experiment_name: preset for preset, experiment_name in expected_map.items()}

        for missing_name in missing_experiments:
            preset_name = reverse_lookup.get(missing_name, "unknown")
            validator._report_error(
                family=family,
                preset=preset_name,
                experiment_name=missing_name,
                check_name="missing_expected_experiment",
                message=f"Expected experiment output is missing: {missing_name}",
            )
            results.append(
                validator._finalize_result(
                    validator._result_base(
                        family=family,
                        preset=preset_name,
                        experiment_name=missing_name,
                    )
                )
            )

        for extra_name in extra_experiments:
            validator._report_warning(
                family=family,
                preset="unexpected",
                experiment_name=extra_name,
                check_name="unexpected_experiment_directory",
                message=f"Unexpected experiment directory present: {extra_name}",
            )

        for preset_name, experiment_name in expected_map.items():
            experiment_dir = family_dir / experiment_name
            if not experiment_dir.exists():
                continue
            if family == "timeseries":
                results.append(
                    validator.validate_timeseries_experiment(
                        experiment_dir=experiment_dir,
                        family=family,
                        preset=preset_name,
                        experiment_name=experiment_name,
                    )
                )
            else:
                results.append(
                    validator.validate_classification_experiment(
                        experiment_dir=experiment_dir,
                        family=family,
                        preset=preset_name,
                        experiment_name=experiment_name,
                    )
                )

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(["family", "experiment_name"], kind="mergesort")

    issues_df = pd.DataFrame(
        [
            {
                "severity": issue.severity,
                "family": issue.family,
                "preset": issue.preset,
                "experiment_name": issue.experiment_name,
                "check_name": issue.check_name,
                "message": issue.message,
            }
            for issue in validator.issues
        ]
    )
    if not issues_df.empty:
        issues_df = issues_df.sort_values(["severity", "family", "experiment_name", "check_name"], kind="mergesort")

    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else (experiments_root / "verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.family == "all":
        results_path = output_dir / str(all_spec["verification_results_file"])
        issues_path = output_dir / str(all_spec["verification_issues_file"])
        report_path = output_dir / str(all_spec["verification_summary_file"])
    else:
        results_path = output_dir / str(family_specs[args.family]["verification_results_file"])
        issues_path = output_dir / str(family_specs[args.family]["verification_issues_file"])
        report_path = output_dir / str(family_specs[args.family]["verification_summary_file"])

    results_df.to_csv(results_path, index=False)
    issues_df.to_csv(issues_path, index=False)
    report_path.write_text(
        _render_markdown_report(
            experiments_root=experiments_root,
            results_df=results_df,
            issues_df=issues_df,
        ),
        encoding="utf-8",
    )

    error_count = int((issues_df["severity"] == "error").sum()) if not issues_df.empty else 0
    warning_count = int((issues_df["severity"] == "warning").sum()) if not issues_df.empty else 0

    print(f"Workspace root: {workspace_root.as_posix()}")
    print(f"Experiments root: {experiments_root.as_posix()}")
    print(f"Family scope: {', '.join(selected_families)}")
    print(f"Expected folds: {len(expected_folds)}")
    print(f"Results saved: {results_path.as_posix()}")
    print(f"Issues saved: {issues_path.as_posix()}")
    print(f"Summary report: {report_path.as_posix()}")
    print(f"Errors: {error_count}")
    print(f"Warnings: {warning_count}")

    if args.fail_on_error and error_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
