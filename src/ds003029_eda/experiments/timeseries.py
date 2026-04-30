from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from ..paths import WorkspacePaths, get_paths
from ..pipelines.run_sarima_prep import COMBINED_OUTPUT_NAME, SarimaPrepConfig, run_sarima_prep
from ..sarima_training import run_sarima_training
from .common import ensure_experiment_directories, write_json
from .sarima_clf_bridge import run_sarima_classification_eval


@dataclass(frozen=True)
class TimeSeriesExperimentConfig:
    experiment_name: str
    target_feature: str = "agg_mean_rms"
    exogenous_features: tuple[str, ...] = ()
    artifact_subdir: str = "data_processing_v2"
    output_subdir: str = "experiments/timeseries"
    sarima_input_subdir: str = "sarima"
    min_total_obs: int = 36
    min_test_obs: int = 12
    test_fraction: float = 0.2
    changepoint_penalty: float = 10.0
    residual_z_threshold: float = 2.5
    classification_score_method: str = "zscore"
    sarima_prep_overwrite: bool = False
    training_output_name: str = "sarima_results"
    metadata: dict[str, str] = field(default_factory=dict)


def run_timeseries_experiment(
    config: TimeSeriesExperimentConfig,
    *,
    paths: WorkspacePaths | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = paths or get_paths()
    directories = ensure_experiment_directories(
        paths,
        output_subdir=config.output_subdir,
        experiment_name=config.experiment_name,
    )

    prep_output_subdir = f"{config.artifact_subdir}/{config.sarima_input_subdir}/{config.experiment_name}"
    _, prep_manifest_df, prep_output_root = run_sarima_prep(
        paths=paths,
        config=SarimaPrepConfig(
            artifact_subdir=config.artifact_subdir,
            output_subdir=f"{config.sarima_input_subdir}/{config.experiment_name}",
            aggregate_feature_name=config.target_feature,
            exogenous_feature_names=config.exogenous_features,
            overwrite=config.sarima_prep_overwrite,
        ),
    )
    feature_path = prep_output_root / COMBINED_OUTPUT_NAME

    _, metrics_df = run_sarima_training(
        feature_path=feature_path,
        output_subdir=f"{config.output_subdir}/{config.experiment_name}/{config.training_output_name}",
        paths=paths,
        min_total_obs=config.min_total_obs,
        min_test_obs=config.min_test_obs,
        test_fraction=config.test_fraction,
        target_feature_name=config.target_feature,
        exogenous_cols=config.exogenous_features,
        changepoint_penalty=config.changepoint_penalty,
        residual_z_threshold=config.residual_z_threshold,
    )
    expected_series_ids = set(
        metrics_df.loc[metrics_df["status"].fillna("").eq("ok"), "series_id"].dropna().astype(str).tolist()
    )
    classification_df = run_sarima_classification_eval(
        experiment_dir=directories.root,
        training_output_name=config.training_output_name,
        score_method=config.classification_score_method,
        residual_col="sarima_residual",
        label_col="y",
        split_col="split",
        eval_split="test",
        expected_series_ids=expected_series_ids or None,
    )

    write_json(
        directories.root / "run_config.json",
        {
            **asdict(config),
            "workspace_root": paths.workspace.as_posix(),
            "prep_output_subdir": prep_output_subdir,
            "combined_feature_csv": feature_path.as_posix(),
            "n_series": int(len(prep_manifest_df)),
            "sarima_classification_metrics_csv": (directories.root / "sarima_classification_metrics.csv").as_posix(),
            "n_classification_rows": int(len(classification_df)),
        },
    )
    metrics_df.to_csv(directories.root / "series_metrics.csv", index=False)
    return prep_manifest_df, metrics_df