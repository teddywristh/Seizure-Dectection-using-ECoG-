# Workflow - staged workspace execution

## Principle

Keep the current workspace layout, but run it in four task groups instead of one monolithic full-pipeline command.

After activating the conda environment, the canonical commands are all plain `python tools/...` calls.

## Task 1 - content and metadata

Refresh run metadata, rebuild seizure intervals, and export the current raw-content manifests.

```bash
python tools/workspace_content.py --workspace-root /path/to/workspace
```

What this produces:
- `eda_outputs/ds003029_run_summary.csv`
- `eda_outputs/ds003029_marker_qc_by_run.csv`
- `eda_outputs/ds003029_seizure_intervals_by_run.csv`
- `eda_outputs/ds003029_content_run_manifest.csv`
- `eda_outputs/ds003029_model_ready_run_manifest.csv`

Default behavior:
- validates the current `16` raw-content runs
- validates the current `16` model-ready runs

## Task 2 - data processing by direction

Run preprocessing separately for each modeling direction.

Timeseries:

```bash
python tools/workspace_data.py timeseries --workspace-root /path/to/workspace
```

ML:

```bash
python tools/workspace_data.py ml --workspace-root /path/to/workspace
```

DL:

```bash
python tools/workspace_data.py dl --workspace-root /path/to/workspace
```

What each direction does:
- `timeseries`: `preprocess` -> `features` -> `sarima-prep`
- `ml`: `preprocess` -> `features`
- `dl`: `preprocess` -> `features` -> `raw_folds` export by default

## Task 3 - run one experiment preset at a time

List presets first:

```bash
python tools/workspace_experiment.py timeseries --list-presets
python tools/workspace_experiment.py ml --list-presets
python tools/workspace_experiment.py dl --list-presets
python tools/workspace_experiment.py all --list-presets
```

Run a single preset:

```bash
python tools/workspace_experiment.py timeseries --preset sarima_rms --workspace-root /path/to/workspace
python tools/workspace_experiment.py ml --preset xgboost_optuna --workspace-root /path/to/workspace
python tools/workspace_experiment.py dl --preset eegnet --workspace-root /path/to/workspace
python tools/workspace_experiment.py all --workspace-root /path/to/workspace --force-retrain
```

This intentionally does not default to "run all presets".

Run behavior:
- `--list-presets` only prints preset names.
- `--preset <name>` runs one preset only.
- `all` runs every preset from the grouped config `configs/experiments/all_models.json` in family order `timeseries -> ml -> dl`.
- If the preset already has a complete output directory, the wrapper reuses those results and exits without retraining.
- If the preset directory is only partially populated, the wrapper stops. Use `--force-retrain` to retrain from scratch in that output directory.

## Task 4 - post-training outputs

Summaries:

```bash
python tools/workspace_reports.py summarize --family all --workspace-root /path/to/workspace
python tools/workspace_reports.py summarize --family timeseries --workspace-root /path/to/workspace
python tools/workspace_reports.py cross_family_leaderboard --workspace-root /path/to/workspace
```

Metric plots:

```bash
python tools/workspace_reports.py plot --family all --workspace-root /path/to/workspace
python tools/workspace_reports.py plot --family ml --workspace-root /path/to/workspace
```

Verification:

```bash
python tools/workspace_reports.py verify --family all --workspace-root /path/to/workspace
python tools/workspace_reports.py verify --family dl --workspace-root /path/to/workspace
```

Main outputs:
- modeling outputs: `eda_outputs/experiments/{timeseries,ml,dl}/<experiment_name>/`
- summaries: `eda_outputs/experiments/summary/` (includes `cross_family_leaderboard.csv`)
- plots: `eda_outputs/experiments/summary/plots/`
- verification reports: `eda_outputs/experiments/verification/`

## Legacy script

`tools/run_workspace_pipeline.py` remains available for bulk reruns, but it is now a legacy convenience script rather than the recommended workflow.
