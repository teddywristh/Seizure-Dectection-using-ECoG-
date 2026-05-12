from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ..paths import WorkspacePaths, get_paths
from ..utils import compute_class_weights
from .common import (
    aggregate_metrics_frame,
    compute_binary_metrics,
    ensure_experiment_directories,
    merge_prediction_frame,
    seed_everything,
    write_json,
)

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - fallback for older sklearn
    StratifiedGroupKFold = None


@dataclass(frozen=True)
class MLExperimentConfig:
    model_name: str
    experiment_name: str
    artifact_subdir: str = "data_processing_v2"
    output_subdir: str = "experiments/ml"
    threshold: float = 0.5
    random_state: int = 42
    n_jobs: int = 1
    optuna_trials: int = 20
    cv_splits: int = 3
    svm_rfe_min_features: int = 12
    shap_sample_size: int = 256
    hyperparameters: dict[str, Any] = field(default_factory=dict)


def _load_fold_arrays(fold_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, list[str]]:
    train_payload = np.load(fold_dir / "train_dataset.npz", allow_pickle=False)
    test_payload = np.load(fold_dir / "test_dataset.npz", allow_pickle=False)
    train_index = pd.read_csv(fold_dir / "train_index.csv")
    test_index = pd.read_csv(fold_dir / "test_index.csv")
    manifest = pd.read_json(fold_dir / "manifest.json", typ="series")

    x_train = train_payload["x_agg"].astype(np.float32)
    y_train = train_payload["y"].astype(int)
    x_test = test_payload["x_agg"].astype(np.float32)
    y_test = test_payload["y"].astype(int)
    feature_names = [str(name) for name in manifest["aggregate_feature_names"]]
    return x_train, y_train, x_test, y_test, train_index, test_index, feature_names


def _build_group_cv(y: np.ndarray, groups: np.ndarray, n_splits: int, random_state: int):
    if StratifiedGroupKFold is not None and np.unique(groups).size >= n_splits:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _predict_scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision))
    raise TypeError(f"Model {type(model).__name__} does not expose predict_proba or decision_function.")


def _fit_lightgbm_dart(class_weights: dict[int, float], config: MLExperimentConfig):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("lightgbm is required for the lightgbm_dart experiment.") from exc

    params = {
        "boosting_type": "dart",
        "objective": "binary",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "class_weight": class_weights,
        "random_state": config.random_state,
        "n_jobs": config.n_jobs,
    }
    params.update(config.hyperparameters)
    return LGBMClassifier(**params), {"params": params}


def _fit_catboost(class_weights: dict[int, float], config: MLExperimentConfig):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("catboost is required for the catboost experiment.") from exc

    ordered_weights = [class_weights.get(0, 1.0), class_weights.get(1, 1.0)]
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "class_weights": ordered_weights,
        "random_seed": config.random_state,
        "verbose": False,
    }
    params.update(config.hyperparameters)
    return CatBoostClassifier(**params), {"params": params}


def _fit_stacking(class_weights: dict[int, float], config: MLExperimentConfig):
    lightgbm_model, lightgbm_meta = _fit_lightgbm_dart(class_weights, config)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=class_weights,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    lr = LogisticRegression(
        max_iter=1000,
        class_weight=class_weights,
        solver="lbfgs",
        random_state=config.random_state,
    )
    model = StackingClassifier(
        estimators=[("rf", rf), ("lgbm", lightgbm_model), ("lr", clone(lr))],
        final_estimator=clone(lr),
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=config.n_jobs,
    )
    return model, {"params": {"rf": rf.get_params(), "lgbm": lightgbm_meta["params"], "meta": lr.get_params()}}


def _fit_svm_rbf_rfe(class_weights: dict[int, float], config: MLExperimentConfig, x_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray):
    groups = np.asarray(groups).reshape(-1)
    selector_estimator = LogisticRegression(
        max_iter=1000,
        class_weight=class_weights,
        solver="liblinear",
        random_state=config.random_state,
    )
    cv = _build_group_cv(y_train, groups, config.cv_splits, config.random_state)
    selector = RFECV(
        estimator=selector_estimator,
        step=0.2,
        min_features_to_select=config.svm_rfe_min_features,
        cv=cv,
        scoring="roc_auc",
        n_jobs=config.n_jobs,
    )
    svm = SVC(
        kernel="rbf",
        probability=True,
        class_weight=class_weights,
        C=float(config.hyperparameters.get("C", 2.0)),
        gamma=config.hyperparameters.get("gamma", "scale"),
        random_state=config.random_state,
    )
    model = Pipeline([("selector", selector), ("svm", svm)])
    model.fit(x_train, y_train, selector__groups=groups)
    fitted_selector = model.named_steps["selector"]
    selector_results = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in fitted_selector.cv_results_.items()
    }
    return model, {"selected_features": int(fitted_selector.n_features_), "grid_scores": selector_results}


def _fit_xgboost_optuna(
    class_weights: dict[int, float],
    config: MLExperimentConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
):
    try:
        import optuna
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("xgboost and optuna are required for the xgboost_optuna experiment.") from exc

    scale_pos_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
    splitter = _build_group_cv(y_train, groups, config.cv_splits, config.random_state)

    def objective(trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": config.random_state,
            "n_jobs": config.n_jobs,
        }
        fold_scores: list[float] = []
        if StratifiedGroupKFold is not None and hasattr(splitter, "split"):
            split_iter = splitter.split(x_train, y_train, groups)
        else:
            split_iter = splitter.split(x_train, y_train)
        for train_idx, valid_idx in split_iter:
            model = XGBClassifier(**params)
            model.fit(x_train[train_idx], y_train[train_idx], verbose=False)
            valid_score = _predict_scores(model, x_train[valid_idx])
            fold_scores.append(compute_binary_metrics(y_train[valid_idx], valid_score)["roc_auc"])
        return float(np.nanmean(fold_scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.optuna_trials, show_progress_bar=False)

    best_params = {
        **study.best_params,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": config.random_state,
        "n_jobs": config.n_jobs,
    }
    model = XGBClassifier(**best_params)
    model.fit(x_train, y_train, verbose=False)
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    return model, {"best_params": best_params, "trials_df": trials_df}


def _build_model(
    config: MLExperimentConfig,
    *,
    class_weights: dict[int, float],
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
):
    if config.model_name == "xgboost_optuna":
        return _fit_xgboost_optuna(class_weights, config, x_train, y_train, groups)
    if config.model_name == "lightgbm_dart":
        model, meta = _fit_lightgbm_dart(class_weights, config)
        model.fit(x_train, y_train)
        return model, meta
    if config.model_name == "catboost":
        model, meta = _fit_catboost(class_weights, config)
        model.fit(x_train, y_train)
        return model, meta
    if config.model_name == "stacking":
        model, meta = _fit_stacking(class_weights, config)
        model.fit(x_train, y_train)
        return model, meta
    if config.model_name == "svm_rbf_rfe":
        return _fit_svm_rbf_rfe(class_weights, config, x_train, y_train, groups)
    raise ValueError(f"Unsupported ML model: {config.model_name}")


def _extract_feature_table(model, feature_names: list[str]) -> pd.DataFrame | None:
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": np.asarray(model.feature_importances_, dtype=float),
            }
        ).sort_values("importance", ascending=False, kind="mergesort")

    if hasattr(model, "coef_"):
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": np.abs(np.asarray(model.coef_).reshape(-1)),
            }
        ).sort_values("importance", ascending=False, kind="mergesort")

    if isinstance(model, Pipeline) and "selector" in model.named_steps:
        selector = model.named_steps["selector"]
        if hasattr(selector, "support_"):
            return pd.DataFrame(
                {
                    "feature": feature_names,
                    "selected": selector.support_.astype(bool),
                    "ranking": selector.ranking_.astype(int),
                }
            ).sort_values(["selected", "ranking", "feature"], ascending=[False, True, True], kind="mergesort")
    return None


def _compute_shap_table(model, x_train: np.ndarray, feature_names: list[str], config: MLExperimentConfig) -> pd.DataFrame | None:
    try:
        import shap
    except ImportError:  # pragma: no cover - optional dependency
        return None

    sample = x_train[: min(config.shap_sample_size, len(x_train))]
    shap_feature_names = list(feature_names)
    shap_model = model
    if isinstance(model, Pipeline) and "selector" in model.named_steps:
        selector = model.named_steps["selector"]
        sample = selector.transform(sample)
        shap_feature_names = [name for name, keep in zip(feature_names, selector.support_) if keep]
        shap_model = model.named_steps["svm"]

    try:
        explainer = shap.Explainer(shap_model, sample)
        explanation = explainer(sample)
    except Exception:
        return None

    values = np.abs(np.asarray(explanation.values, dtype=float))
    if values.ndim == 3:
        values = values[..., 0]
    mean_abs = values.mean(axis=0)
    return pd.DataFrame(
        {
            "feature": shap_feature_names,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False, kind="mergesort")


def run_ml_experiment(
    config: MLExperimentConfig,
    *,
    paths: WorkspacePaths | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = paths or get_paths()
    seed_everything(config.random_state)
    artifact_root = paths.outputs_dir / config.artifact_subdir
    folds_root = artifact_root / "folds"
    if not folds_root.exists():
        raise FileNotFoundError(f"Missing folds directory: {folds_root}")

    directories = ensure_experiment_directories(
        paths,
        output_subdir=config.output_subdir,
        experiment_name=config.experiment_name,
    )

    per_fold_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    run_config = asdict(config)
    write_json(directories.root / "run_config.json", run_config)

    for fold_dir in sorted(path for path in folds_root.iterdir() if path.is_dir() and path.name.startswith("fold_")):
        x_train, y_train, x_test, y_test, train_index, test_index, feature_names = _load_fold_arrays(fold_dir)
        groups = train_index["base"].astype(str).to_numpy() if "base" in train_index.columns else np.arange(len(train_index))
        class_weights = compute_class_weights(y_train)

        model, fit_meta = _build_model(
            config,
            class_weights=class_weights,
            x_train=x_train,
            y_train=y_train,
            groups=groups,
        )

        y_score = _predict_scores(model, x_test)
        fold_metrics = compute_binary_metrics(y_test, y_score, threshold=config.threshold)
        fold_metrics.update(
            {
                "fold_id": fold_dir.name,
                "model_name": config.model_name,
                "experiment_name": config.experiment_name,
            }
        )
        per_fold_metrics.append(fold_metrics)

        prediction_df = merge_prediction_frame(
            test_index,
            y_true=y_test,
            y_score=y_score,
            threshold=config.threshold,
            split="test",
        )
        prediction_df.insert(0, "fold_id", fold_dir.name)
        prediction_df.to_csv(directories.predictions_dir / f"{fold_dir.name}_predictions.csv", index=False)
        all_predictions.append(prediction_df)

        joblib.dump(model, directories.checkpoints_dir / f"{fold_dir.name}_{config.model_name}.joblib")
        write_json(
            directories.reports_dir / f"{fold_dir.name}_fit_meta.json",
            {
                "class_weights": class_weights,
                "fit_meta": {
                    key: value
                    for key, value in fit_meta.items()
                    if key != "trials_df"
                },
            },
        )

        if "trials_df" in fit_meta:
            fit_meta["trials_df"].to_csv(directories.reports_dir / f"{fold_dir.name}_optuna_trials.csv", index=False)

        feature_table = _extract_feature_table(model, feature_names)
        if feature_table is not None:
            feature_table.to_csv(directories.reports_dir / f"{fold_dir.name}_feature_table.csv", index=False)

        shap_table = _compute_shap_table(model, x_train, feature_names, config)
        if shap_table is not None:
            shap_table.to_csv(directories.reports_dir / f"{fold_dir.name}_shap_importance.csv", index=False)

    metrics_df = pd.DataFrame(per_fold_metrics).sort_values("fold_id", kind="mergesort")
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    metrics_df.to_csv(directories.root / "fold_metrics.csv", index=False)
    predictions_df.to_csv(directories.root / "all_predictions.csv", index=False)

    aggregate_df = aggregate_metrics_frame(metrics_df)
    aggregate_df.to_csv(directories.root / "aggregate_metrics.csv", index=False)
    return metrics_df, aggregate_df