# ds003029 EDA Workflow (Current Workspace)

This document summarizes the *current* EDA pipeline implemented in this workspace for the OpenNeuro dataset **ds003029** (Epilepsy-iEEG-Multicenter-Dataset) and explains how the existing notebooks/Markdown files fit together.

The overall goal is **seizure detection** (primarily **ictal vs non-ictal**), using iEEG/ECoG/SEEG recordings stored in **BrainVision** format.

---

## 0) Big picture: two-layer EDA strategy

Your EDA is organized into two complementary layers:

1) **Metadata-only EDA (full dataset coverage)**
- Uses BIDS sidecars (`participants.tsv`, `*_channels.tsv`, `*_events.tsv`, `*_ieeg.json`, `*_scans.tsv`).
- Does **not** require downloading large `*.eeg` binaries.
- Produces dataset-wide summaries and *CSV artifacts* in `eda_outputs/`.

2) **Signal-level EDA (subset, intentionally small)**
- Requires actual BrainVision signal content (`*.vhdr + *.vmrk + *.eeg`) via git-annex.
- Loads a small number of runs to validate:
  - signal quality,
  - label alignment (onset/offset vs observed dynamics),
  - feature behavior and outlier patterns.

This “full metadata + small signal subset” approach is the core of your current workflow.

---

## 1) Inputs and prerequisites

### 1.1 Dataset location
The notebooks assume the dataset is available at:
- `EEG/ds003029`

### 1.2 Data format assumptions
- Dataset follows **iEEG-BIDS** structure.
- Each run typically has:
  - Signal: `*_ieeg.vhdr`, `*_ieeg.vmrk`, `*_ieeg.eeg`
  - Sidecars: `*_channels.tsv`, `*_events.tsv`, `*_ieeg.json`

### 1.3 DataLad / git-annex (important)
In this workspace, ds003029 behaves like a **git-annex dataset**.
- Files can “exist” but still be **placeholders** (no real content).
- Signal-level notebooks require that the `*.eeg` **content is present locally**.

Supporting docs:
- `config_datalad_and_annex.md` (Windows setup for DataLad/git-annex)
- `get_data_tutorial_ds003029.md` (minimum `git annex get` steps for one run)

---

## 2) Step-by-step EDA workflow (as implemented)

### Step A — Dataset orientation (context + pitfalls)
**Purpose:** understand ds003029 structure, modalities, label/marker realities.

Main reference:
- `ds003029_ecog_ieeg_overview_vi.md`

Key outcome:
- A clear understanding that marker naming is not fully standardized across centers.

---

### Step B — Participant-level EDA (metadata)
**Purpose:** explore participant demographics/metadata, missingness, and potential confounds.

Notebook:
- `eda_ds003029.ipynb` → **Part 1**

Typical outputs:
- Missingness plots
- Basic categorical/numeric distributions

Why it matters:
- Multi-center datasets often have strong site effects.
- This informs how you later split data (recommended: by subject, not by window).

---

### Step C — Run inventory + label extraction (metadata)
**Purpose:** build a dataset-wide run inventory without reading `*.eeg`.

Notebook:
- `eda_ds003029.ipynb` → **Part 2**

What it does:
- Scans all subjects/sessions for BIDS sidecars.
- Builds `run_summary` by joining per-run metadata from:
  - `*_channels.tsv` (channel count, good/bad counts, channel types)
  - `*_events.tsv` (trial_type markers)
  - `*_ieeg.json` (sampling frequency, line frequency)
- Uses regex heuristics over `events.tsv:trial_type` to detect:
  - seizure onset
  - seizure offset
  - and derives duration where possible.
- Detects whether signal content is available locally:
  - preferred: `git annex find -i here` (content presence)
  - fallback: file size heuristic for placeholders

Artifacts produced:
- `eda_outputs/ds003029_run_summary.csv`
- `eda_outputs/ds003029_event_vocab.csv`

Why it matters:
- This step defines the *operational label source* used by later notebooks.
- If the regex/mapping is wrong, window labels will be wrong in downstream work.

---

### Step D — Marker QA, pairing, and multi-seizure checks (metadata)
**Purpose:** validate seizure marker consistency and produce more robust interval tables.

Code/tools:
- `tools/analyze_event_markers_ds003029.py`
- `tools/validate_labels_ds003029.py`

Key outputs (CSV artifacts):
- `eda_outputs/ds003029_marker_qc_by_run.csv`
  - counts of onset/offset markers, pairing flags, and QC fields
- `eda_outputs/ds003029_seizure_intervals_by_run.csv`
  - per-run interval list: onset_s/offset_s plus matched trial_type strings
- `eda_outputs/ds003029_trial_type_onset_vocab.csv`
- `eda_outputs/ds003029_trial_type_offset_vocab.csv`
- `eda_outputs/ds003029_trial_type_vocab.csv`

Why it matters:
- ds003029 may contain multiple seizures per run (or unpaired markers).
- An interval table is safer than “first onset + first offset” if you later extend labeling.

---

### Step E — Signal-level EDA on a small subset (BrainVision)
**Purpose:** confirm that signal properties and label boundaries look sensible on real waveforms.

Prerequisite:
- Download at least one run’s BrainVision triple:
  - `*.vhdr`, `*.vmrk`, `*.eeg`
  - see `get_data_tutorial_ds003029.md`

Notebooks:
- `eda_ds003029.ipynb` → **Part 3** (quick “visualize one case”)
- `eda_signal_characteristics_ds003029.ipynb` (structured signal EDA)

Signal EDA themes (as implemented):
- **Scale & artifacts**: RMS/PTP per channel, “hot” channels
- **Spectral content**: PSD (Welch), line-noise evidence (60 Hz + harmonics), bandpower
- **Temporal dependence / complexity**: Hjorth parameters, entropy-style summaries
- **Non-stationarity / transitions**: feature trajectories with ictal overlays

Artifacts produced (demo feature dataset):
- `eda_outputs/ds003029_window_features_demo.csv`
- `eda_outputs/ds003029_windowing_demo_info.csv`

Why it matters:
- Confirms whether “ictal intervals” correspond to visible energy/complexity changes.
- Surfaces typical modeling pitfalls: line noise, outliers, bad channels, amplitude confounds.

---

### Step F — Rigorous “readiness” report (bounded, explicit limitations)
**Purpose:** write an EDA report that answers: “Is this dataset ready for seizure modeling, and what preprocessing is required?”

Notebook:
- `eda_rigorous_ds003029_eda_report.ipynb`

Characteristics:
- Enforces *EDA-only* constraints (no training).
- Treats conclusions as conditional on current downloaded subset.

Companion narrative report:
- `signal_eda_report_ds003029_vi.md`

---

### Step G — Visualization hub (figures for reporting)
**Purpose:** centralize plots for both CSV artifacts (metadata QA) and signal case studies.

Notebook:
- `eda_visualizations_ds003029.ipynb`

What it uses:
- CSV artifacts in `eda_outputs/`
- Optional signal reads from `EEG/ds003029/` when MNE + content are available

What it produces (conceptually):
- Coverage/QA plots (runs, content availability, pairing)
- Vocab plots for onset/offset trial_type strings
- Signal plots (waveform, PSD, spectrogram)
- Feature trajectories with seizure overlays
- Outlier scoring and window-level QC visuals

---

### Step H — “Trends / seasonality / issues” (run-level timeline proxy)
**Purpose:** satisfy a generic time-series EDA checklist using ds003029 as an example.

Notebook:
- `eda_trends_seasonality_issues_ds003029.ipynb`

Method:
- Builds a time series from BIDS `scans.tsv` using `acq_time`.
- The series is **run-count over time** (hourly/daily resampling).

Important interpretation note:
- This is a *recording schedule timeline*, not a physiological continuous measurement.
- Dates appear anonymized/date-shifted; interpret month/year seasonality with caution.

Companion review:
- `eda_trends_seasonality_issues_ds003029_review.md`

---

## 3) Where outputs live (current convention)

All generated artifacts are stored in:
- `eda_outputs/`

Key artifacts you currently rely on:
- `ds003029_run_summary.csv` (core metadata table; downstream notebooks depend on it)
- `ds003029_marker_qc_by_run.csv` (QC counts and pairing status)
- `ds003029_seizure_intervals_by_run.csv` (interval list for overlays and future labeling)
- `ds003029_window_features_demo.csv` + `ds003029_windowing_demo_info.csv` (demo windowed feature dataset)

---

## 4) Recommended execution order (practical)

1) (Once) Set up DataLad/git-annex: `config_datalad_and_annex.md`
2) Run metadata EDA + generate artifacts: `eda_ds003029.ipynb` (Parts 1–2)
3) Run marker QA scripts if needed (or confirm outputs already exist): `tools/*.py`
4) Download 1–N runs for signal EDA: `get_data_tutorial_ds003029.md`
5) Run signal-level EDA: `eda_signal_characteristics_ds003029.ipynb` (and optionally `eda_rigorous_ds003029_eda_report.ipynb`)
6) Use `eda_visualizations_ds003029.ipynb` to generate figures for reporting
7) (Optional) Run the checklist/time-series proxy notebook: `eda_trends_seasonality_issues_ds003029.ipynb`

---

## 5) Modeling/EDA cautions already implied by the current workflow

- **Avoid leakage**: do not split train/test by window if windows come from the same run/subject.
- **Subset bias**: conclusions drawn from a small number of downloaded runs may not generalize.
- **Label strategy**: window labels depend on onset/offset extraction quality and marker mapping.
- **Line noise**: PSD evidence suggests notch/mitigation is likely required before spectral features.

---

## 6) Glossary (terms used in this workflow)

- **Run**: one recording file instance (one BrainVision triple in BIDS).
- **Interval**: a seizure segment, typically [onset_s, offset_s].
- **Window**: a fixed-duration slice used for feature extraction and labeling (e.g., 2 s).
- **Ictal**: inside seizure interval.
- **Non-ictal**: outside seizure interval.
