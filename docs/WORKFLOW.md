# Workflow - ds003029 on branch `bim`

## Big picture
Pipeline hien tai co 4 lop:
- Metadata-only: quet BIDS sidecars de lap inventory run, channels, events.
- Marker QC: chuan hoa onset/offset va ghep seizure intervals.
- Signal-level features: doc BrainVision bang MNE, cat window, trich `rms`, `ptp`, `line_length`, va label `y`.
- Modeling: fit SARIMA thuan theo tung run tren chuoi `rms`.

## Main execution order
1. `notebooks/01_metadata_run_summary_ds003029.ipynb`
   - Tao `ds003029_run_summary.csv` va `ds003029_event_vocab.csv`
2. `notebooks/02_marker_qc_intervals_ds003029.ipynb`
   - Tao `ds003029_marker_qc_by_run.csv` va `ds003029_seizure_intervals_by_run.csv`
3. `notebooks/03_signal_eda_windows_features_ds003029.ipynb`
   - Tao feature windows de train model
4. `notebooks/04_sarima_training_ds003029.ipynb`
   - Train SARIMA thuan tren tung run voi chronological train/test split

## Reusable code
- `src/ds003029_eda/paths.py`: resolve workspace, dataset, output paths
- `src/ds003029_eda/run_summary.py`: build metadata inventory
- `src/ds003029_eda/markers.py`: parse onset/offset markers
- `src/ds003029_eda/marker_qc.py`: build QC tables and seizure intervals
- `src/ds003029_eda/window_features_multirun.py`: build multirun window features
- `src/ds003029_eda/sarima_training.py`: train pure SARIMA without exogenous regressors

## Scripts
- `tools/analyze_event_markers_ds003029.py`
- `tools/build_window_features_multirun_full.py`
- `tools/train_sarima.py`
- `tools/validate_labels_ds003029.py`

## Notes
- Nhanh `bim` khong con giu workflow ARIMAX cu lam duong train chinh nua.
- Neu can chay tu command line, xem `docs/SARIMA_TRAINING.md`.
