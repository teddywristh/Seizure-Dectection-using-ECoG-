# Experiment Runbook

This repository now exposes reproducible experiment runners for all three modeling directions.

## Install Dependencies

```bash
conda activate ecog
conda install pytorch pytorch-cuda -c pytorch -c nvidia
pip install -r requirements.txt
```

If `ecog` already has a working CUDA-enabled PyTorch build, skip the `conda install` line above.

If the `conda install` step fails with `errno 28` because `C:\Users\LENOVO\anaconda3\pkgs` runs out of space, use the lower-cache fallback instead:

```bash
conda clean -a -y
python -m pip uninstall -y torch
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1+cu124
pip install -r requirements.txt
```

Quick CUDA check:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

If you do not want to activate the environment in your current shell, prepend commands with `conda run -n ecog`.

On Linux, check `python --version` before running scripts. Some systems still map `python` to Python 2.
If that happens, use `python3` (or your conda env python) for all commands in this runbook.

## Recommended staged workflow

The canonical run path is now split into four task groups. This keeps the workspace structure unchanged, but avoids the old "run everything at once" entrypoint.

## Task 1 - content and metadata

Refresh run summary, marker QC, seizure intervals, and export the current 16-run manifests.

```bash
python tools/workspace_content.py --workspace-root /path/to/workspace
```

Outputs:
- `eda_outputs/ds003029_run_summary.csv`
- `eda_outputs/ds003029_marker_qc_by_run.csv`
- `eda_outputs/ds003029_seizure_intervals_by_run.csv`
- `eda_outputs/ds003029_content_run_manifest.csv`
- `eda_outputs/ds003029_model_ready_run_manifest.csv`

## Task 2 - process data by modeling direction

Timeseries processing:

```bash
python tools/workspace_data.py timeseries --workspace-root /path/to/workspace
```

ML processing:

```bash
python tools/workspace_data.py ml --workspace-root /path/to/workspace
```

DL processing:

```bash
python tools/workspace_data.py dl --workspace-root /path/to/workspace
```

What these wrappers do:
- `timeseries`: preprocess -> features -> SARIMA-ready export
- `ml`: preprocess -> features
- `dl`: preprocess -> features -> raw fold export by default

## Task 3 - run one preset at a time

List canonical presets:

```bash
python tools/workspace_experiment.py timeseries --list-presets
python tools/workspace_experiment.py ml --list-presets
python tools/workspace_experiment.py dl --list-presets
python tools/workspace_experiment.py all --list-presets
```

Behavior of the experiment wrapper:
- `--list-presets` only lists presets. It does not launch training.
- `--preset <name>` launches one preset only.
- `all` launches every preset in grouped order `timeseries -> ml -> dl` using `configs/experiments/all_models.json`.
- If a complete output already exists for that preset, the wrapper reuses the existing outputs and does not retrain.
- If only partial outputs or checkpoints exist, the wrapper does not auto-resume. Use `--force-retrain` to retrain from scratch.

Run one timeseries preset:

```bash
python tools/workspace_experiment.py timeseries --preset sarima_rms --workspace-root /path/to/workspace
python tools/workspace_experiment.py timeseries --preset sarimax_rms_std --workspace-root /path/to/workspace
python tools/workspace_experiment.py timeseries --preset sarimax_gamma --workspace-root /path/to/workspace
python tools/workspace_experiment.py timeseries --preset sarimax_hjorth --workspace-root /path/to/workspace
```

Run one ML preset:

```bash
python tools/workspace_experiment.py ml --preset xgboost_optuna --workspace-root /path/to/workspace
python tools/workspace_experiment.py ml --preset lightgbm_dart --workspace-root /path/to/workspace
python tools/workspace_experiment.py ml --preset catboost --workspace-root /path/to/workspace
python tools/workspace_experiment.py ml --preset stacking --workspace-root /path/to/workspace
python tools/workspace_experiment.py ml --preset svm_rbf_rfe --workspace-root /path/to/workspace
```

Run one DL preset:

```bash
python tools/workspace_experiment.py dl --preset eegnet --workspace-root /path/to/workspace --device cuda
python tools/workspace_experiment.py dl --preset eegwavenet --workspace-root /path/to/workspace --device cuda
python tools/workspace_experiment.py dl --preset cnn_bilstm --workspace-root /path/to/workspace --device cuda
python tools/workspace_experiment.py dl --preset bendr --workspace-root /path/to/workspace --device cuda
python tools/workspace_experiment.py dl --preset reve --workspace-root /path/to/workspace --device cuda
python tools/workspace_experiment.py dl --preset biseizurere --workspace-root /path/to/workspace --device cuda
python tools/workspace_experiment.py dl --preset inresformer --workspace-root /path/to/workspace --device cuda
python tools/workspace_experiment.py dl --preset gat_bilstm --workspace-root /path/to/workspace --device cuda
python tools/workspace_experiment.py dl --preset ce_tss_transformer --workspace-root /path/to/workspace --device cuda
python tools/workspace_experiment.py dl --preset dbconformer --workspace-root /path/to/workspace --device cuda
python tools/workspace_experiment.py dl --preset graphs4mer --workspace-root /path/to/workspace --device cuda
python tools/workspace_experiment.py dl --preset dcrnn --workspace-root /path/to/workspace --device cuda
```

If you want to force DL streaming from cached `.fif` instead of using `raw_folds`, add `--raw-loading-strategy on_demand`.

## Task 4 - post-training reports

Regenerate summaries:

```bash
python tools/workspace_reports.py summarize --family all --workspace-root /path/to/workspace
python tools/workspace_reports.py summarize --family ml --workspace-root /path/to/workspace
python tools/workspace_reports.py cross_family_leaderboard --workspace-root /path/to/workspace
```

Create metric plots:

```bash
python tools/workspace_reports.py plot --family all --workspace-root /path/to/workspace
python tools/workspace_reports.py plot --family timeseries --workspace-root /path/to/workspace
```

Verify all current outputs:

```bash
python tools/workspace_reports.py verify --family all --workspace-root /path/to/workspace
python tools/workspace_reports.py verify --family dl --workspace-root /path/to/workspace
```

Verification artifacts:
- `eda_outputs/experiments/verification/verification_results.csv`
- `eda_outputs/experiments/verification/verification_issues.csv`
- `eda_outputs/experiments/verification/verification_summary.md`
- `eda_outputs/experiments/verification/verification_summary_timeseries.md`
- `eda_outputs/experiments/verification/verification_summary_ml.md`
- `eda_outputs/experiments/verification/verification_summary_dl.md`

## Legacy bulk runner

`tools/run_workspace_pipeline.py` is still available for bulk reruns, but it is now legacy rather than the recommended daily workflow.

## Artifact Conventions

- ML checkpoints: `eda_outputs/experiments/ml/<experiment_name>/checkpoints/*.joblib`
- DL checkpoints: `eda_outputs/experiments/dl/<experiment_name>/checkpoints/*.pt`
- Timeseries metrics: `eda_outputs/experiments/timeseries/<experiment_name>/series_metrics.csv`
- Timeseries derived classification metrics: `eda_outputs/experiments/timeseries/<experiment_name>/sarima_classification_metrics.csv`
- Per-fold predictions: `predictions/*.csv`
- Aggregates: `aggregate_metrics.csv`, SARIMA metric exports, and `eda_outputs/experiments/summary/cross_family_leaderboard.csv`

## Notes

- `biseizurere` is exposed as a repository-local proxy architecture because the source list explicitly notes the absence of a public canonical implementation.
- `bendr` and `reve` are implemented as trainable in-repo adapters rather than upstream pretrained checkpoint loaders.
- Detailed source-fidelity notes for every model family are tracked in [docs/MODEL_IMPLEMENTATION_NOTES.md](docs/MODEL_IMPLEMENTATION_NOTES.md).