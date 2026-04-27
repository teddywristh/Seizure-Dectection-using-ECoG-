# ds003029 - EEG/iEEG EDA and SARIMA training

Muc tieu workspace tren nhanh `bim`: giu pipeline BIDS -> marker QC -> window features -> train SARIMA thuan cho du lieu ECoG/iEEG.

## Canonical task scripts

Activate the conda environment first, then use the task-oriented scripts below with `python ...`.

1. Task 1 - refresh content manifests for the current 16-run subset

```bash
python tools/workspace_content.py --workspace-root /path/to/workspace
```

Outputs:
- `eda_outputs/ds003029_run_summary.csv`
- `eda_outputs/ds003029_marker_qc_by_run.csv`
- `eda_outputs/ds003029_seizure_intervals_by_run.csv`
- `eda_outputs/ds003029_content_run_manifest.csv`
- `eda_outputs/ds003029_model_ready_run_manifest.csv`

2. Task 2 - process the 16 runs for one modeling direction

```bash
python tools/workspace_data.py timeseries --workspace-root /path/to/workspace
python tools/workspace_data.py ml --workspace-root /path/to/workspace
python tools/workspace_data.py dl --workspace-root /path/to/workspace
```

3. Task 3 - train one preset at a time

```bash
python tools/workspace_experiment.py timeseries --list-presets
python tools/workspace_experiment.py timeseries --preset sarima_rms --workspace-root /path/to/workspace

python tools/workspace_experiment.py ml --list-presets
python tools/workspace_experiment.py ml --preset xgboost_optuna --workspace-root /path/to/workspace

python tools/workspace_experiment.py dl --list-presets
python tools/workspace_experiment.py dl --preset eegnet --workspace-root /path/to/workspace

python tools/workspace_experiment.py all --list-presets
python tools/workspace_experiment.py all --workspace-root /path/to/workspace --force-retrain
```

Notes:
- `--list-presets` only lists available presets. It does not train anything.
- `python tools/workspace_experiment.py ... --preset ...` runs exactly one preset, not all presets in that family.
- `python tools/workspace_experiment.py all ...` runs all presets grouped by `timeseries`, then `ml`, then `dl`, using `configs/experiments/all_models.json`.
- If a completed output for that preset already exists, the wrapper reuses it and does not retrain.
- If only partial outputs or checkpoints exist, the wrapper stops and asks for `--force-retrain`; automatic resume-from-checkpoint is not implemented.

4. Task 4 - summarize, plot, and verify after training

```bash
python tools/workspace_reports.py summarize --family all --workspace-root /path/to/workspace
python tools/workspace_reports.py summarize --family timeseries --workspace-root /path/to/workspace
python tools/workspace_reports.py plot --family ml --workspace-root /path/to/workspace
python tools/workspace_reports.py verify --family dl --workspace-root /path/to/workspace
```

Notes:
- `--family all` remains the default and generates the cross-family reports.
- `--family timeseries|ml|dl` generates only that scope.
- Report family/file naming is controlled by `configs/reports/family_reports.json`.

## Legacy orchestrator

`tools/run_workspace_pipeline.py` is still available for full-workspace reruns, but it is no longer the recommended day-to-day entrypoint.

## Main docs
- `docs/WORKFLOW.md`
- `docs/DATA_ACCESS.md`
- `docs/SARIMA_TRAINING.md`
- `docs/MODELING_DATA_QUICK_REFERENCE.md`

## Code organization
- Shared logic lives in `src/ds003029_eda/`
- Command-line scripts live in `tools/`
- Generated artifacts live in `eda_outputs/`

Recommended task entrypoints:
- `tools/workspace_content.py`
- `tools/workspace_data.py`
- `tools/workspace_experiment.py`
- `tools/workspace_reports.py`

## Modeling note
- Pipeline modeling chinh tren nhanh `bim` la SARIMA thuần.
- Danh gia duoc thuc hien bang chia chuoi theo thoi gian train/test, khong dung `exog`, khong dung SARIMAX.
