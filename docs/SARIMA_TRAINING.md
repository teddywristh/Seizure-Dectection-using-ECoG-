# SARIMA training on branch `bim`

## Repo structure
- `EEG/ds003029/`: raw BIDS iEEG/ECoG dataset.
- `src/ds003029_eda/`: reusable logic for path discovery, marker parsing, interval pairing, feature extraction, and SARIMA training.
- `tools/`: CLI scripts to build features and run training.
- `notebooks/`: exploratory notebooks. The active modeling path is now SARIMA-only.
- `eda_outputs/`: generated summaries, features, metrics, predictions, and model summaries.

## Current data pipeline
1. Metadata inventory: build `eda_outputs/ds003029_run_summary.csv`.
2. Marker QC and seizure intervals: build `eda_outputs/ds003029_seizure_intervals_by_run.csv`.
3. Window features: `src/ds003029_eda/window_features_multirun.py` reads BrainVision runs with MNE, cuts fixed windows, and exports run-level time series such as `rms`, `ptp`, `line_length`, and label `y`.
4. SARIMA training: `src/ds003029_eda/sarima_training.py` loads one feature file, groups by run, and fits one pure SARIMA model per run on the target series `rms`.

## What changed
- Removed the old ARIMA/ARIMAX direction from the main training path.
- The SARIMA trainer no longer uses `exog` columns or lag regressors.
- Model selection is now a compact grid search over `(p, d, q)` and seasonal `(P, D, Q, s)` only.
- Evaluation is now chronological: fit on the train segment and forecast the held-out test segment.
- Outputs are written under `eda_outputs/<output_subdir>/`:
  - `sarima_metrics.csv`
  - `sarima_all_predictions.csv`
  - `predictions/*_sarima_predictions.csv`
  - `summaries/*_sarima_summary.txt`
  - `sarima_run_config.json`

## Main commands
Build full multirun features, then train SARIMA:

```powershell
python tools/train_sarima.py --build-multirun-features
```

Train SARIMA from an existing feature file:

```powershell
python tools/train_sarima.py --features ds003029_window_features_multirun_full.csv
```

Run the trainer module directly:

```powershell
python src/ds003029_eda/sarima_training.py --features ds003029_window_features_multirun_full.csv
```

## Notes
- This is SARIMA, not SARIMAX. No exogenous regressors are passed into the model.
- Seasonal periods are interpreted in window steps, then mapped to seconds using the median cadence of each run.
- The trainer stores both train fitted values and held-out test forecasts for later evaluation or seizure-analysis work.
