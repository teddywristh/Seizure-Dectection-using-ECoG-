# ds003029 — EEG/iEEG EDA (refactored)

Mục tiêu workspace: EDA tối giản nhưng đầy đủ luồng **raw BIDS → metadata EDA → marker QC/intervals → signal EDA (subset) → windowing/features demo**.

## Quick start (the intended pipeline)
1) **Step 01 — metadata inventory**: open and run [notebooks/01_metadata_run_summary_ds003029.ipynb](notebooks/01_metadata_run_summary_ds003029.ipynb)
   - Outputs: `eda_outputs/ds003029_run_summary.csv`, `eda_outputs/ds003029_event_vocab.csv`

2) **Step 02 — marker QC + seizure intervals**: open and run [notebooks/02_marker_qc_intervals_ds003029.ipynb](notebooks/02_marker_qc_intervals_ds003029.ipynb)
   - Outputs: `eda_outputs/ds003029_marker_qc_by_run.csv`, `eda_outputs/ds003029_seizure_intervals_by_run.csv`, onset/offset vocabs

3) **Step 03 — signal EDA + windowing/features demo**: open and run [notebooks/03_signal_eda_windows_features_ds003029.ipynb](notebooks/03_signal_eda_windows_features_ds003029.ipynb)
   - Requires: MNE + at least one run’s `*.vhdr/*.vmrk/*.eeg` content present locally
   - Outputs: `eda_outputs/ds003029_window_features_demo.csv`, `eda_outputs/ds003029_windowing_demo_info.csv`

## Docs
- Workflow overview: [docs/WORKFLOW.md](docs/WORKFLOW.md)
- Data access (git-annex minimal): [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md)

## Code organization
- Shared logic lives in [src/ds003029_eda](src/ds003029_eda) (thin notebooks; reusable code).
- Utility scripts live in [tools](tools).
