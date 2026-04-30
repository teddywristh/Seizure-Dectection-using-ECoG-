# SARIMA Training

This document describes how SARIMA training fits into the current repo after the data-processing v2 bridge was added.

## 1. Recommended data source

The recommended SARIMA input is now the v2 bridge output:

- `eda_outputs/data_processing_v2/sarima/ds003029_sarima_v2_input.csv`

This CSV is generated from the v2 run tensors and is directly compatible with `tools/train_sarima.py` and `src/ds003029_eda/sarima_training.py`.

## 2. What the bridge exports

For each run, the bridge:

- reads `features/runs/*_window_tensor.npz`
- selects aggregate feature `agg_mean_rms`
- renames it to `rms`
- keeps `t_mid_s` as the time coordinate
- writes one `series_id` per run

The exported schema is:

- `series_id`
- `subject`
- `base`
- `window_id`
- `t_start_s`
- `t_stop_s`
- `t_mid_s`
- `rms`
- `y`
- `source_feature`

Boundary windows are retained in the SARIMA export because the chronological cadence matters for time-series modeling.

## 3. Build commands

### Build or refresh v2 artifacts

```powershell
python tools/run_data_processing_v2.py preprocess --workspace-root /path/to/Seizure-Dectection-using-ECoG- --artifact-subdir data_processing_v2 --overwrite
python tools/run_data_processing_v2.py features --workspace-root /path/to/Seizure-Dectection-using-ECoG- --artifact-subdir data_processing_v2 --overwrite
```

### Export SARIMA-ready CSV

```powershell
python tools/run_data_processing_v2.py sarima-prep --workspace-root /path/to/Seizure-Dectection-using-ECoG- --artifact-subdir data_processing_v2 --overwrite
```

### Train SARIMA from the v2 bridge output

```powershell
python tools/train_sarima.py --features data_processing_v2/sarima/ds003029_sarima_v2_input.csv --output-subdir sarima_v2_from_data_processing_v2
```

## 4. Trainer expectations

`sarima_training.py` expects:

- a target column named `rms` or `rms_mean`
- a time column named `t_mid_s`, `t_start`, or `t_end`
- a grouping column such as `series_id`, `run_id`, `base`, or `run`

The bridge intentionally writes `series_id`, `t_mid_s`, and `rms`, so no SARIMA code changes are needed downstream.

## 5. Outputs from training

Training writes under `eda_outputs/<output_subdir>/`:

- `sarima_metrics.csv`
- `sarima_all_predictions.csv`
- `predictions/*_sarima_predictions.csv`
- `summaries/*_sarima_summary.txt`
- `sarima_run_config.json`

When SARIMA runs through the experiment wrapper (`tools/run_timeseries_experiments.py`), a post-processing bridge also writes:

- `eda_outputs/experiments/timeseries/<experiment_name>/sarima_classification_metrics.csv`

This file converts test-split residuals into anomaly scores, then computes binary metrics (ROC-AUC, average precision, F1, precision, sensitivity, specificity, accuracy) using the shared `compute_binary_metrics()` utility.

## 6. Notes

- The trainer supports both SARIMA and SARIMAX. It runs as pure SARIMA when no exogenous columns are passed, and as SARIMAX when exogenous columns are configured.
- Seasonal periods are interpreted in window steps and then converted to seconds using each run's median cadence.
- If you want to model another v2 aggregate feature, change `--aggregate-feature-name` in `sarima-prep` and rerun the export.

## 7. Interpretation caveats for SARIMA-derived classification metrics

- SARIMA ROC-AUC and PR-AUC are derived from residual anomaly scores, not from a classifier trained directly for seizure labels.
- The split policy is not symmetric with ML and DL: SARIMA uses chronological in-run splits, while ML and DL use LOSO subject holdout.
- Thresholds are selected by per-run F1 sweep on the same test segment used for evaluation, which can inflate thresholded metrics.
