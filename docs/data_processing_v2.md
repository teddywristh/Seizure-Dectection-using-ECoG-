# Data Processing v2 - ds003029 Seizure Detection

## 1. Scope and decisions
- Task: window-level binary seizure detection.
- Label states: `1=ictal`, `0=interictal`, `-1=boundary-drop`.
- Target sampling rate: `256 Hz`.
- Window config: `2.0 s` window with `0.5 s` step.
- Split strategy: leave-one-subject-out cross-validation.
- Channel policy: keep every good channel after QC; no fixed truncation to 16 channels.
- Normalization: fit only on train subjects for each fold.

## 2. Pipeline flow

```mermaid
flowchart LR
    A[BrainVision BIDS] --> B[run_summary]
    B --> C[marker_qc and seizure_intervals]
    C --> D[preprocess and QC]
    D --> E[window slicing]
    E --> F[labeling with boundary margin]
    F --> G[channel and aggregate features]
    G --> H[LOSO fold builder]
    H --> I[train-only scalers]
    I --> J[fold datasets and reports]
```

## 3. Code map
- `src/ds003029_eda/data/io.py`: inventory loading, BrainVision loading, channel selection.
- `src/ds003029_eda/data/channel_qc.py`: flat, variance-z, and line-noise QC.
- `src/ds003029_eda/data/preprocess.py`: notch, bandpass, resample, average reference.
- `src/ds003029_eda/data/normalize.py`: train-only array scalers for tensors and aggregate features.
- `src/ds003029_eda/labels/labeler.py`: midpoint-based labeling with boundary-drop margin.
- `src/ds003029_eda/features/windowing.py`: deterministic sliding windows.
- `src/ds003029_eda/features/time_domain.py`: RMS, line length, Hjorth, ZCR, kurtosis, skewness.
- `src/ds003029_eda/features/freq_domain.py`: band powers, spectral entropy, peak frequency.
- `src/ds003029_eda/splits/patient_cv.py`: LOSO folds and leakage checks.
- `src/ds003029_eda/pipelines/run_preprocess.py`: `.fif` cache and QC report generation.
- `src/ds003029_eda/pipelines/run_features.py`: run tensors, fold datasets, and analysis reports.
- `tools/run_data_processing_v2.py`: CLI entrypoint.

## 4. Output layout

All v2 outputs are written under `<workspace_root>/eda_outputs/data_processing_v2/`.

- `preprocess_run_summary.csv`: per-run preprocessing status and signal metadata.
- `bad_channels.csv`: channel-level QC flags.
- `reports/qc_report.html`: HTML summary of preprocessing QC.
- `features/runs/*_window_tensor.npz`: per-run tensors with channel and aggregate features.
- `features/runs/*_window_index.csv`: per-window metadata and labels.
- `run_feature_inventory.csv`: run-level feature extraction inventory.
- `reports/label_distribution.csv`: per-subject class counts.
- `reports/feature_stats.csv`: per-class summary statistics for aggregate features.
- `reports/class_separability.csv`: univariate AUC scores for aggregate features.
- `folds/<fold_id>/train_dataset.npz`: normalized train tensor bundle.
- `folds/<fold_id>/test_dataset.npz`: normalized test tensor bundle.
- `folds/<fold_id>/train_index.csv`: train window metadata.
- `folds/<fold_id>/test_index.csv`: test window metadata.
- `folds/<fold_id>/scalers.json`: fold-specific train-only scalers.
- `fold_manifest.json`: fold-level subject assignments and counts.

## 5. Invariants
- Preprocessed signal must have `sfreq == 256`.
- Good channels must contain no `NaN`, `Inf`, or all-zero series.
- Window labels use midpoint logic only.
- Boundary windows are excluded from fold datasets.
- `train_subjects ∩ test_subjects == ∅` for every fold.
- Every fold scaler is fitted only on the train partition.

## 6. Known limits
- `class_separability.csv` is computed on aggregate features, not the full per-channel tensor.
- Bad channels are preserved in cached `.fif` for traceability but excluded from feature extraction.
- The pipeline assumes `ds003029_run_summary.csv` and `ds003029_seizure_intervals_by_run.csv` already exist or are generated first.

## 7. Recommended execution order
1. Build metadata inventory.
2. Build marker QC and paired intervals.
3. Run preprocessing.
4. Run feature extraction and fold building.
5. Inspect `qc_report.html`, `label_distribution.csv`, and `class_separability.csv` before switching to modeling.

## 8. Workspace root note
- The new CLI tools accept `--workspace-root`.
- If you run commands from the nested repo but your real dataset lives one level above it, point `--workspace-root` to that outer workspace.
- In the current workspace, the correct root for real `.eeg` content is `c:/Users/LENOVO/Downloads/eeg`.