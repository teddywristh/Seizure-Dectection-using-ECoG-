# Experiment Checklist - ds003029 Data Processing v2

## 0. Working directory
- [x] Open PowerShell.
- [x] Change directory to repo root:

```powershell
cd c:\Users\LENOVO\Downloads\eeg\Seizure-Dectection-using-ECoG-
```

Validation result (2026-04-21):
- Commands were executed from `c:\Users\LENOVO\Downloads\eeg\Seizure-Dectection-using-ECoG-`.
- Conda shell setup had to be repaired before the run. Fresh PowerShell sessions now resolve `conda`; the validation run used the `vitaldb_aki` environment Python directly for reliability.

## 1. Refresh metadata prerequisites
- [x] Rebuild run summary:

```powershell
conda run -n vitaldb_aki python tools/build_run_summary_ds003029.py --workspace-root c:/Users/LENOVO/Downloads/eeg
```

- [x] Rebuild marker QC and seizure intervals:

```powershell
conda run -n vitaldb_aki python tools/build_marker_qc_ds003029.py --workspace-root c:/Users/LENOVO/Downloads/eeg
```

- [x] Sanity check that these files exist:
  - `eda_outputs/ds003029_run_summary.csv`
  - `eda_outputs/ds003029_marker_qc_by_run.csv`
  - `eda_outputs/ds003029_seizure_intervals_by_run.csv`

Validation result (2026-04-21):
- `build_run_summary_ds003029.py` completed with `106` runs and `847` event-vocabulary entries.
- `build_marker_qc_ds003029.py` completed with `106` runs and `78` paired seizure intervals.
- Verified the three metadata CSV outputs under `c:/Users/LENOVO/Downloads/eeg/eda_outputs/`.

## 2. Optional test gate
- [x] Run the core test suite before the full experiment:

```powershell
conda run -n vitaldb_aki python -m pytest tests/test_markers.py tests/test_labeler.py tests/test_windowing.py tests/test_no_leakage.py tests/test_preprocess.py
```

Validation result (2026-04-21):
- Final core suite status: `9 passed, 14 warnings in 2.40s`.
- The initial run exposed preprocessing and QC regressions; after fixes, the rerun passed cleanly.

## 3. Preprocess signals
- [x] Run preprocessing and QC:

```powershell
conda run -n vitaldb_aki python tools/run_data_processing_v2.py preprocess --workspace-root c:/Users/LENOVO/Downloads/eeg --artifact-subdir data_processing_v2 --target-sfreq 256 --bandpass-low 0.5 --bandpass-high 120 --notch-harmonics 3 --reference-mode average
```

- [x] Inspect outputs:
  - `eda_outputs/data_processing_v2/preprocess_run_summary.csv`
  - `eda_outputs/data_processing_v2/bad_channels.csv`
  - `eda_outputs/data_processing_v2/reports/qc_report.html`

- [x] Confirm preprocessing health:
  - `preprocess_ok` should be `True` for expected runs.
  - `sfreq_out` should be `256.0`.
  - `n_channels_good` should be non-zero for all successful runs.

Validation result (2026-04-21):
- Preprocessing completed for `16 / 16` runs.
- Verified `preprocess_ok=True` for all `16` runs.
- Verified `sfreq_out=256.0` for all `16` runs.
- Verified `n_channels_good > 0` for all successful runs; observed range was `77` to `117` good channels.
- Verified `preprocess_run_summary.csv`, `bad_channels.csv`, and `reports/qc_report.html` under `c:/Users/LENOVO/Downloads/eeg/eda_outputs/data_processing_v2/`.

## 4. Extract features and build LOSO folds
- [x] Run feature extraction:

```powershell
conda run -n vitaldb_aki python tools/run_data_processing_v2.py features --workspace-root c:/Users/LENOVO/Downloads/eeg --artifact-subdir data_processing_v2 --window-sec 2.0 --step-sec 0.5 --label-margin-sec 0.5 --fill-missing-after-scaling 0.0
```

- [x] Inspect outputs:
  - `eda_outputs/data_processing_v2/run_feature_inventory.csv`
  - `eda_outputs/data_processing_v2/reports/label_distribution.csv`
  - `eda_outputs/data_processing_v2/reports/feature_stats.csv`
  - `eda_outputs/data_processing_v2/reports/class_separability.csv`
  - `eda_outputs/data_processing_v2/fold_manifest.json`

- [x] Verify fold artifacts exist under `eda_outputs/data_processing_v2/folds/`:
  - `train_dataset.npz`
  - `test_dataset.npz`
  - `train_index.csv`
  - `test_index.csv`
  - `scalers.json`
  - `manifest.json`

Validation result (2026-04-21):
- Feature extraction completed for `16 / 16` runs.
- Verified `run_feature_inventory.csv`, `label_distribution.csv`, `feature_stats.csv`, `class_separability.csv`, and `fold_manifest.json` under `c:/Users/LENOVO/Downloads/eeg/eda_outputs/data_processing_v2/`.
- Generated `8` current LOSO folds: `fold_01_jh101`, `fold_02_jh102`, `fold_03_jh103`, `fold_04_pt01`, `fold_05_pt13`, `fold_06_pt3`, `fold_07_pt7`, and `fold_08_ummc001`.
- Verified every current fold contains `train_dataset.npz`, `test_dataset.npz`, `train_index.csv`, `test_index.csv`, `scalers.json`, and `manifest.json`.
- Removed stale fold directories from earlier partial runs so `folds/` now matches `fold_manifest.json` exactly.

## 5. Go or no-go checks
- [x] `label_distribution.csv` shows both ictal and interictal windows for most subjects.
- [x] `n_dropped_margin / n_windows` is not unexpectedly large.
- [x] `class_separability.csv` has at least several aggregate features with `auc > 0.60`.
- [x] Fold manifests show no subject overlap.

Validation result (2026-04-21):
- All `8 / 8` subjects in `label_distribution.csv` have both ictal and interictal windows.
- `n_dropped_margin / n_windows = 64 / 7646 = 0.00837` (about `0.84%`).
- `40` aggregate features have `auc > 0.60`.
- Top AUC features in the final run were `agg_max_beta_power=0.824`, `agg_std_beta_power=0.819`, `agg_std_line_length=0.801`, `agg_max_line_length=0.799`, and `agg_std_gamma_low_power=0.789`.
- Verified `0` train/test subject-overlap violations across `8` folds.

## 6. One-command rerun
- [x] If you need a clean rerun after verifying prerequisites, use:

```powershell
conda run -n vitaldb_aki python tools/run_data_processing_v2.py full --workspace-root c:/Users/LENOVO/Downloads/eeg --artifact-subdir data_processing_v2 --target-sfreq 256 --bandpass-low 0.5 --bandpass-high 120 --notch-harmonics 3 --reference-mode average --window-sec 2.0 --step-sec 0.5 --label-margin-sec 0.5 --fill-missing-after-scaling 0.0 --overwrite
```

Validation result (2026-04-21):
- The end-to-end `full --overwrite` command completed successfully.
- Final CLI summary: `End-to-end processing completed: 16 runs, 8 folds`.

## 7. Hand-off to modeling
- [ ] Use `train_dataset.npz` and `test_dataset.npz` per fold for ML or DL experiments.
- [ ] Keep `train_index.csv` and `test_index.csv` for mapping predictions back to subject, run, and time.
- [ ] Archive `manifest_preprocess.json`, `manifest_features.json`, and `fold_manifest.json` with every experiment run.

Validation result (2026-04-21):
- Verified the current `8` folds contain `train_dataset.npz` and `test_dataset.npz` ready for downstream ML or DL experiments.
- Verified `train_index.csv` and `test_index.csv` exist for every current fold and retain subject/run/time mapping columns.
- Verified `manifest_preprocess.json`, `manifest_features.json`, and `fold_manifest.json` are present under `c:/Users/LENOVO/Downloads/eeg/eda_outputs/data_processing_v2/`.