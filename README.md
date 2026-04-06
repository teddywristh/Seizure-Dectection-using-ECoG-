# ds003029 - EEG/iEEG EDA and SARIMA training

Muc tieu workspace tren nhanh `bim`: giu pipeline BIDS -> marker QC -> window features -> train SARIMA thuan cho du lieu ECoG/iEEG.

## Quick start
1. Metadata inventory: run `notebooks/01_metadata_run_summary_ds003029.ipynb`
   - Outputs: `eda_outputs/ds003029_run_summary.csv`, `eda_outputs/ds003029_event_vocab.csv`
2. Marker QC and seizure intervals: run `notebooks/02_marker_qc_intervals_ds003029.ipynb`
   - Outputs: `eda_outputs/ds003029_marker_qc_by_run.csv`, `eda_outputs/ds003029_seizure_intervals_by_run.csv`
3. Signal EDA and window features: run `notebooks/03_signal_eda_windows_features_ds003029.ipynb`
   - Requires local BrainVision content (`*.vhdr`, `*.vmrk`, `*.eeg`)
4. SARIMA training:
   - Notebook path: `notebooks/04_sarima_training_ds003029.ipynb`
   - CLI path: `python tools/train_sarima.py --build-multirun-features`

## Main docs
- `docs/WORKFLOW.md`
- `docs/DATA_ACCESS.md`
- `docs/SARIMA_TRAINING.md`

## Code organization
- Shared logic lives in `src/ds003029_eda/`
- Command-line scripts live in `tools/`
- Generated artifacts live in `eda_outputs/`

## Modeling note
- Pipeline modeling chinh tren nhanh `bim` la SARIMA thuần.
- Danh gia duoc thuc hien bang chia chuoi theo thoi gian train/test, khong dung `exog`, khong dung SARIMAX.
