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

## 1. Timeseries

The timeseries runner rebuilds a SARIMA-ready CSV from v2 tensors, then trains SARIMA/SARIMAX with changepoint and anomaly outputs.

```bash
python tools/run_timeseries_experiments.py \
  --workspace-root C:/Users/LENOVO/Downloads/eeg \
  --config configs/experiments/timeseries_models.json \
  --preset sarima_rms

python tools/run_timeseries_experiments.py \
  --workspace-root C:/Users/LENOVO/Downloads/eeg \
  --config configs/experiments/timeseries_models.json \
  --preset sarimax_rms_std

python tools/run_timeseries_experiments.py \
  --workspace-root C:/Users/LENOVO/Downloads/eeg \
  --config configs/experiments/timeseries_models.json \
  --preset sarimax_gamma

python tools/run_timeseries_experiments.py \
  --workspace-root C:/Users/LENOVO/Downloads/eeg \
  --config configs/experiments/timeseries_models.json \
  --preset sarimax_hjorth
```

Outputs are written under `eda_outputs/experiments/timeseries/<experiment_name>/`.

## 2. Machine Learning

The ML runner consumes `x_agg` from LOSO folds, saves fold-level checkpoints, predictions, SHAP reports when available, and aggregate metrics.

```bash
python tools/run_ml_experiments.py \
  --workspace-root C:/Users/LENOVO/Downloads/eeg \
  --config configs/experiments/ml_models.json \
  --preset xgboost_optuna

python tools/run_ml_experiments.py \
  --workspace-root C:/Users/LENOVO/Downloads/eeg \
  --config configs/experiments/ml_models.json \
  --preset lightgbm_dart

python tools/run_ml_experiments.py \
  --workspace-root C:/Users/LENOVO/Downloads/eeg \
  --config configs/experiments/ml_models.json \
  --preset catboost

python tools/run_ml_experiments.py \
  --workspace-root C:/Users/LENOVO/Downloads/eeg \
  --config configs/experiments/ml_models.json \
  --preset stacking

python tools/run_ml_experiments.py \
  --workspace-root C:/Users/LENOVO/Downloads/eeg \
  --config configs/experiments/ml_models.json \
  --preset svm_rbf_rfe
```

Outputs are written under `eda_outputs/experiments/ml/<experiment_name>/`.

## 3. Raw Tensor Export For DL

Raw-signal models can either use pre-exported `(N, C, T)` tensors or stream windows directly from cached `*_preproc_raw.fif` files.
If disk is tight, skip this step and let the DL runner fall back to on-demand raw loading automatically.

```bash
python tools/build_raw_folds.py \
  --workspace-root C:/Users/LENOVO/Downloads/eeg \
  --artifact-subdir data_processing_v2 \
  --output-subdir raw_folds \
  --output-dtype float16
```

Outputs are written under `eda_outputs/data_processing_v2/raw_folds/<fold_id>/`.
The exporter is resume-safe by default: rerunning it will skip folds that already have both train/test raw `.npz` files unless you pass `--overwrite`.

## 4. Deep Learning

The DL runner saves a best checkpoint per fold, epoch history, fold predictions, and aggregate metrics.
By default the DL presets auto-select CUDA when available, use mixed precision for faster GPU training, and fall back to on-demand raw loading when `raw_folds` is absent.

### Raw-signal models

```bash
python tools/run_dl_experiments.py --workspace-root C:/Users/LENOVO/Downloads/eeg --config configs/experiments/dl_models.json --preset eegnet --device cuda
python tools/run_dl_experiments.py --workspace-root C:/Users/LENOVO/Downloads/eeg --config configs/experiments/dl_models.json --preset eegwavenet --device cuda
python tools/run_dl_experiments.py --workspace-root C:/Users/LENOVO/Downloads/eeg --config configs/experiments/dl_models.json --preset cnn_bilstm --device cuda
python tools/run_dl_experiments.py --workspace-root C:/Users/LENOVO/Downloads/eeg --config configs/experiments/dl_models.json --preset bendr --device cuda
python tools/run_dl_experiments.py --workspace-root C:/Users/LENOVO/Downloads/eeg --config configs/experiments/dl_models.json --preset reve --device cuda
python tools/run_dl_experiments.py --workspace-root C:/Users/LENOVO/Downloads/eeg --config configs/experiments/dl_models.json --preset biseizurere --device cuda
```

If you want to force streaming from cached `.fif` instead of using `raw_folds`, add `--raw-loading-strategy on_demand`.

### Feature-channel models

```bash
python tools/run_dl_experiments.py --workspace-root C:/Users/LENOVO/Downloads/eeg --config configs/experiments/dl_models.json --preset inresformer --device cuda
python tools/run_dl_experiments.py --workspace-root C:/Users/LENOVO/Downloads/eeg --config configs/experiments/dl_models.json --preset gat_bilstm --device cuda
python tools/run_dl_experiments.py --workspace-root C:/Users/LENOVO/Downloads/eeg --config configs/experiments/dl_models.json --preset ce_tss_transformer --device cuda
python tools/run_dl_experiments.py --workspace-root C:/Users/LENOVO/Downloads/eeg --config configs/experiments/dl_models.json --preset dbconformer --device cuda
python tools/run_dl_experiments.py --workspace-root C:/Users/LENOVO/Downloads/eeg --config configs/experiments/dl_models.json --preset graphs4mer --device cuda
python tools/run_dl_experiments.py --workspace-root C:/Users/LENOVO/Downloads/eeg --config configs/experiments/dl_models.json --preset dcrnn --device cuda
```

Outputs are written under `eda_outputs/experiments/dl/<experiment_name>/`.

## Artifact Conventions

- ML checkpoints: `eda_outputs/experiments/ml/<experiment_name>/checkpoints/*.joblib`
- DL checkpoints: `eda_outputs/experiments/dl/<experiment_name>/checkpoints/*.pt`
- Timeseries metrics: `eda_outputs/experiments/timeseries/<experiment_name>/series_metrics.csv`
- Per-fold predictions: `predictions/*.csv`
- Aggregates: `aggregate_metrics.csv` or SARIMA metric exports

## Notes

- `biseizurere` is exposed as a repository-local proxy architecture because the source list explicitly notes the absence of a public canonical implementation.
- `bendr` and `reve` are implemented as trainable in-repo adapters rather than upstream pretrained checkpoint loaders.
- Detailed source-fidelity notes for every model family are tracked in [docs/MODEL_IMPLEMENTATION_NOTES.md](docs/MODEL_IMPLEMENTATION_NOTES.md).