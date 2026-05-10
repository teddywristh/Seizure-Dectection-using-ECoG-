# Final Report Preparation Notes

This file is a report-prep evidence bank, not final report prose.

Intended use:
- Copy facts, tables, and claims from here into the final report.
- Rewrite into scientific prose later.
- Keep claims tied to evidence files listed here.
- Add formal citations during writing; this file only notes what should be cited.

Primary repo-local evidence files:
- `task.md`
- `docs/data_processing_v2.md`
- `docs/MODEL_IMPLEMENTATION_NOTES.md`
- `eda_outputs/ds003029_model_ready_run_manifest.csv`
- `eda_outputs/data_processing_v2/fold_manifest.json`
- `eda_outputs/data_processing_v2/sarima/sarima_prep_manifest.json`
- `eda_outputs/experiments/summary/cross_family_leaderboard.csv`
- `eda_outputs/experiments/summary/ml_metrics_summary.csv`
- `eda_outputs/experiments/summary/dl_metrics_summary.csv`
- `eda_outputs/experiments/summary/timeseries_metrics_summary.csv`
- `eda_outputs/experiments/verification/verification_summary.md`

Session note:
- Final experiment refresh in this work session used conda env `drug-tox-env`.
- Repo runbook examples still mention env `ecog`; for the final report, describe the actual environment used for the rerun if you want strict reproducibility.

## 1. Report Framing Notes

### 1.1 Recommended problem framing

Core problem:
- Window-level seizure detection on intracranial EEG / ECoG from OpenNeuro `ds003029`.
- Binary target at the window level: ictal vs interictal.
- Comparison across three modeling directions implemented in one repo and one processed cohort:
  - Timeseries / SARIMA-SARIMAX
  - Classical machine learning on engineered aggregate features
  - Deep learning on raw windows or channel-feature tensors

What the report can honestly claim:
- A unified benchmarking pipeline was built for three model families on the same model-ready cohort.
- Leakage-safe LOSO subject evaluation exists for ML and DL.
- A residual-to-classification bridge was added so timeseries outputs can be compared using ROC-AUC / PR-AUC style metrics.
- On this cohort and this preprocessing pipeline, feature-engineered boosting models outperform the tested DL models and strongly outperform residual-derived timeseries classifiers on both ROC-AUC and PR-AUC.

What the report should not overclaim:
- This is not a novel neural architecture paper unless you explicitly present the contribution as benchmarking / pipeline design rather than a new model.
- Timeseries classification results are not native classifiers; they are derived from forecast residuals.
- Timeseries split policy is not perfectly apples-to-apples with ML/DL because it is chronological within run rather than LOSO by subject.

### 1.2 Candidate title directions

Option A, benchmark framing:
- Cross-family Benchmarking for Window-level Seizure Detection on OpenNeuro ds003029 ECoG

Option B, practical-performance framing:
- Feature-engineered Boosting versus Deep Learning and SARIMAX for Window-level ECoG Seizure Detection

Option C, pipeline framing:
- A Unified Processing and Evaluation Pipeline for Timeseries, Machine Learning, and Deep Learning Seizure Detection on ds003029

Option D, result-led framing:
- Aggregate Feature Boosting Outperforms Tested Deep and Residual-derived Timeseries Baselines for Window-level ECoG Seizure Detection

### 1.3 Contribution logic notes

Possible contribution chain:
- Existing seizure-detection studies often compare methods within one family only.
- Cross-family comparison is hard because data preparation, leakage control, and tensor formats differ.
- This repo standardizes preprocessing, window labeling, feature extraction, fold construction, and reporting.
- A SARIMA residual bridge extends timeseries forecasts into comparable anomaly scores and binary metrics.
- Evidence shows that on this small, heterogeneous, subject-held-out cohort, engineered aggregate features with boosting remain the strongest practical baseline.

Possible contribution statement template for the final report:
- Existing seizure-detection workflows often evaluate either classical feature models, deep models, or timeseries models in isolation, making cross-family conclusions unreliable. To address this gap, we built a unified ds003029 pipeline that standardizes preprocessing, window labeling, subject-held-out evaluation, and summary reporting across timeseries, ML, and DL approaches. On the final model-ready cohort, boosted tree models on aggregate engineered features achieved the strongest overall ROC-AUC and PR-AUC, while residual-derived SARIMAX provided a weaker but still interpretable anomaly-based baseline.

Alternative contribution statement if you want CatBoost as the practical proposed method:
- We propose a leakage-safe aggregate-feature boosting workflow for window-level seizure detection and compare it against residual-derived timeseries baselines and multiple deep architectures under one standardized ds003029 processing pipeline. The resulting evidence shows that CatBoost delivers the most reliable overall discrimination on the current cohort, exceeding the best tested DL and timeseries alternatives in both ROC-AUC and PR-AUC.

### 1.4 Candidate research questions

Good 3-5 research questions aligned with current evidence:
- RQ1. Under a unified preprocessing pipeline, which model family gives the strongest seizure-detection performance on ds003029: timeseries, classical ML, or deep learning?
- RQ2. Do aggregate engineered features outperform the tested deep architectures on this small subject-held-out cohort?
- RQ3. Does adding exogenous aggregate features to SARIMAX improve residual-derived seizure discrimination compared with univariate SARIMA on RMS alone?
- RQ4. Within deep learning, do channel-feature models outperform raw-signal models under the current data volume and training setup?
- RQ5. Which metrics tell the clearest cross-family story: ROC-AUC, PR-AUC, F1, or forecast error?

## 2. Section-by-section Notes for `task.md`

### 2.1 Introduction notes

Background notes:
- Seizure detection from intracranial recordings is clinically important for monitoring, localization support, and possible future decision support.
- Intracranial EEG / ECoG has strong temporal structure, channel heterogeneity, and subject-specific variation.
- Window-level detection is operationally useful because it converts long recordings into a sequence of local decisions.

Challenge notes:
- Small, heterogeneous cohort after strict model-readiness filtering.
- Strong subject-level variation in ictal prevalence.
- Different model families require different input representations.
- Deep models need more data and stable validation to outperform strong feature-based baselines.
- Classical forecast metrics do not directly measure seizure-discrimination quality.

Motivation notes:
- Need one clean pipeline and one fair reporting layer before making family-level comparisons.
- Need to quantify whether heavy DL models are actually better than simpler engineered-feature approaches on this dataset size.
- Need a principled way to include timeseries methods in the comparison instead of leaving them in a separate forecast-only track.

### 2.2 Related work notes

Use repo-local source index:
- `model _sources.md`

Method families to cite in final report:
- Timeseries:
  - SARIMA / SARIMAX via statsmodels reference and related classical forecasting literature.
  - PELT changepoint detection via `ruptures`.
- Classical ML:
  - XGBoost + Optuna.
  - LightGBM + DART.
  - CatBoost.
  - StackingClassifier.
  - SVM-RBF with feature selection / SHAP workflows.
- DL raw-signal:
  - EEGNet.
  - EEGWaveNet.
  - CNN-BiLSTM.
  - BENDR.
  - REVE.
  - BISeizureRe note: no public canonical implementation available; repo uses a proxy.
- DL feature/channel:
  - InResFormer.
  - GAT-BiLSTM.
  - CE-TSS-Transformer.
  - DBConformer.
  - GraphS4mer.
  - DCRNN.

Important writing note:
- `model _sources.md` is a useful citation collection, but check every final bibliography item manually before submission.
- Some entries are docs or repo links rather than canonical papers.
- Keep the related-work section honest about implementation fidelity: most DL models are repo adapters, not byte-for-byte ports.

### 2.3 Methodology notes

Must include all of the following in the final report:
- Dataset source and final cohort definition.
- Exact label policy.
- Preprocessing defaults.
- Windowing and feature engineering.
- Fold construction and leakage prevention.
- Family-specific inputs and split policies.
- Model preset inventory and hyperparameters.
- Metrics.
- Reproducibility details.

### 2.4 Results and evidence notes

Must include:
- Per-family ranking tables.
- Cross-family ranking by ROC-AUC and PR-AUC.
- At least one ablation subsection.
- Honest explanation of timeseries comparison caveats.
- Discussion of why ML wins on this cohort.

### 2.5 Discussion / limitations / conclusion notes

Must include:
- Why feature-engineered boosting likely works well here.
- Why raw-signal DL underperforms here.
- Why timeseries residual scoring has limited PR-AUC.
- Small cohort and subject heterogeneity.
- Non-identical split policies across families.
- Adapter-fidelity caveat for DL source implementations.

## 3. Dataset and Cohort Notes

### 3.1 Dataset source and task

Source:
- OpenNeuro `ds003029`

Current task definition:
- Binary seizure detection at the window level.
- Label states before fold export:
  - `1 = ictal`
  - `0 = interictal`
  - `-1 = boundary`

Core preprocessing decisions:
- Target sampling rate after preprocessing: `256 Hz`
- Window length: `2.0 s`
- Step size: `0.5 s`
- Boundary exclusion margin: `0.5 s`
- Split policy for ML/DL: leave-one-subject-out cross-validation
- Normalization: fit on train subjects only in each fold
- Channel policy: keep surviving good channels; pad only to fold-global `max_channels`

### 3.2 Final model-ready cohort

Model-ready subset used for experiments:
- `8` subjects
- `16` ictal ECoG runs
- `7646` total window rows in SARIMA-export view, including boundary windows
- `64` boundary windows total (`4` per run)
- `7582` evaluable windows after dropping boundaries
- `3102` ictal windows
- `4480` interictal windows
- Positive rate across evaluable windows: about `40.91%`
- Negative rate across evaluable windows: about `59.09%`

Subject-level cohort summary:

| subject | runs | total windows incl boundary | evaluable windows | ictal windows | boundary windows |
|---|---:|---:|---:|---:|---:|
| jh101 | 3 | 1235 | 1223 | 496 | 12 |
| jh102 | 2 | 1350 | 1342 | 889 | 8 |
| jh103 | 2 | 870 | 862 | 504 | 8 |
| pt01 | 2 | 1242 | 1234 | 344 | 8 |
| pt13 | 2 | 576 | 568 | 33 | 8 |
| pt3 | 1 | 573 | 569 | 232 | 4 |
| pt7 | 2 | 1010 | 1002 | 185 | 8 |
| ummc001 | 2 | 790 | 782 | 419 | 8 |

Important heterogeneity note:
- Held-out positive counts vary from `33` windows in `pt13` to `889` windows in `jh102`.
- This likely contributes to variance in fold-level AP, F1, and threshold stability.

### 3.3 Run-level inventory used in the final experiments

Source file:
- `eda_outputs/data_processing_v2/sarima/sarima_prep_manifest.json`

| subject | series_id | n_windows | n_positive | n_boundary |
|---|---|---:|---:|---:|
| jh101 | sub-jh101_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg | 281 | 40 | 4 |
| jh101 | sub-jh101_ses-presurgery_task-ictal_acq-ecog_run-02_ieeg | 673 | 416 | 4 |
| jh101 | sub-jh101_ses-presurgery_task-ictal_acq-ecog_run-03_ieeg | 281 | 40 | 4 |
| jh102 | sub-jh102_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg | 769 | 556 | 4 |
| jh102 | sub-jh102_ses-presurgery_task-ictal_acq-ecog_run-02_ieeg | 581 | 333 | 4 |
| jh103 | sub-jh103_ses-presurgery_task-ictal_acq-ecog_run-02_ieeg | 461 | 264 | 4 |
| jh103 | sub-jh103_ses-presurgery_task-ictal_acq-ecog_run-03_ieeg | 409 | 240 | 4 |
| pt01 | sub-pt01_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg | 535 | 170 | 4 |
| pt01 | sub-pt01_ses-presurgery_task-ictal_acq-ecog_run-02_ieeg | 707 | 174 | 4 |
| pt13 | sub-pt13_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg | 283 | 16 | 4 |
| pt13 | sub-pt13_ses-presurgery_task-ictal_acq-ecog_run-02_ieeg | 293 | 17 | 4 |
| pt3 | sub-pt3_ses-presurgery_task-ictal_acq-ecog_run-02_ieeg | 573 | 232 | 4 |
| pt7 | sub-pt7_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg | 455 | 60 | 4 |
| pt7 | sub-pt7_ses-presurgery_task-ictal_acq-ecog_run-02_ieeg | 555 | 125 | 4 |
| ummc001 | sub-ummc001_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg | 389 | 201 | 4 |
| ummc001 | sub-ummc001_ses-presurgery_task-ictal_acq-ecog_run-02_ieeg | 401 | 218 | 4 |

### 3.4 Subject-held-out fold inventory

Source file:
- `eda_outputs/data_processing_v2/fold_manifest.json`

| fold_id | test subject | train windows | test windows | train positive | test positive |
|---|---|---:|---:|---:|---:|
| fold_01_jh101 | jh101 | 6359 | 1223 | 2606 | 496 |
| fold_02_jh102 | jh102 | 6240 | 1342 | 2213 | 889 |
| fold_03_jh103 | jh103 | 6720 | 862 | 2598 | 504 |
| fold_04_pt01 | pt01 | 6348 | 1234 | 2758 | 344 |
| fold_05_pt13 | pt13 | 7014 | 568 | 3069 | 33 |
| fold_06_pt3 | pt3 | 7013 | 569 | 2870 | 232 |
| fold_07_pt7 | pt7 | 6580 | 1002 | 2917 | 185 |
| fold_08_ummc001 | ummc001 | 6800 | 782 | 2683 | 419 |

Fold-design notes:
- LOSO folds are subject-disjoint.
- `global_max_channels = 117` in all final folds.
- Fold bundles contain `x_agg`, `x_channel`, `x_channel_mask`, and binary `y`.
- Boundary windows are already removed before fold export.

### 3.5 Cohort-readiness narrative notes

Good wording for final report:
- The original BIDS tree contains more runs and subjects than the final modeling subset.
- The final experimental cohort was restricted to runs with accessible EEG content plus usable seizure interval annotations needed for window labeling and downstream training.
- The model-ready manifest should be cited as the operational cohort definition used for every experiment family.

## 4. Data Preparation and Processing Notes

### 4.1 End-to-end flow

Pipeline sequence from `docs/data_processing_v2.md`:
- BrainVision files and metadata discovery
- Run summary and seizure interval extraction
- Preprocessing and bad-channel QC
- Cached preprocessed FIF output
- Deterministic sliding-window slicing
- Midpoint-based labeling with boundary exclusion
- Channel-level time and frequency features
- Aggregate mean/std/max features
- LOSO fold builder
- Train-only scalers
- Fold tensor export for ML and DL
- SARIMA prep bridge for run-level scalar series export

### 4.2 Preprocessing defaults

From `docs/data_processing_v2.md`:
- `target_sfreq = 256.0`
- `bandpass_low = 0.5 Hz`
- `bandpass_high = 120.0 Hz`
- `notch_harmonics = 3`, implying notch at `50`, `100`, `150 Hz`
- `reference_mode = average`

Channel QC and signal prep code map:
- `src/ds003029_eda/data/io.py`
- `src/ds003029_eda/data/channel_qc.py`
- `src/ds003029_eda/data/preprocess.py`

### 4.3 Feature engineering

Per-channel feature set (`16` features):
- rms
- line_length
- hjorth_activity
- hjorth_mobility
- hjorth_complexity
- zero_crossing_rate
- kurtosis
- skewness
- delta_power
- theta_power
- alpha_power
- beta_power
- gamma_low_power
- gamma_high_power
- spectral_entropy
- peak_frequency

Aggregate feature set (`48` features):
- `agg_mean_*` for all `16` channel features
- `agg_std_*` for all `16` channel features
- `agg_max_*` for all `16` channel features

Key implication:
- ML models operate directly on the `48` aggregate features.
- Channel-feature DL models operate on `(N, max_channels, 16)` plus mask.
- Timeseries bridge exports one scalar target from the aggregate space.

### 4.4 Artifact layout by approach

Timeseries / SARIMA artifacts:
- Input CSV: `eda_outputs/data_processing_v2/sarima/ds003029_sarima_v2_input.csv`
- Per-run CSVs: `eda_outputs/data_processing_v2/sarima/runs/*_sarima_input.csv`
- Manifest: `eda_outputs/data_processing_v2/sarima/sarima_prep_manifest.json`

ML / feature-based DL artifacts:
- `eda_outputs/data_processing_v2/folds/<fold_id>/train_dataset.npz`
- `eda_outputs/data_processing_v2/folds/<fold_id>/test_dataset.npz`
- `eda_outputs/data_processing_v2/folds/<fold_id>/train_index.csv`
- `eda_outputs/data_processing_v2/folds/<fold_id>/test_index.csv`
- `eda_outputs/data_processing_v2/folds/<fold_id>/scalers.json`

Raw-signal DL artifacts:
- Post-QC FIF cache under `preprocessed/*_preproc_raw.fif`
- Optional exported raw fold tensors under `data_processing_v2/raw_folds`
- On-demand raw loading fallback is also supported

### 4.5 Leakage-prevention notes

Important methods wording:
- ML and DL normalization is fit on train subjects only.
- LOSO subject holdout prevents subject identity leakage between train and test.
- Boundary windows are removed before fold export for classification families.
- Timeseries retains boundary windows to preserve chronological cadence.

## 5. Model Families, Presets, and Exact Config Notes

### 5.1 Shared metric computation

Shared binary metrics code:
- `src/ds003029_eda/experiments/common.py`

Binary metrics returned:
- `roc_auc`
- `average_precision`
- `f1`
- `precision`
- `sensitivity`
- `specificity`
- `accuracy`
- `tp`, `tn`, `fp`, `fn`

Common thresholding:
- ML and DL use a fixed threshold of `0.5`.
- Timeseries bridge selects threshold by per-run F1 sweep.

### 5.2 Timeseries family

Default config from `configs/experiments/timeseries_models.json`:
- `min_total_obs = 36`
- `min_test_obs = 12`
- `test_fraction = 0.2`
- `changepoint_penalty = 10.0`
- `residual_z_threshold = 2.5`
- `sarima_prep_overwrite = false`
- `training_output_name = sarima_results`

Residual-classification bridge details from `src/ds003029_eda/experiments/sarima_clf_bridge.py`:
- Default threshold sweep: `0.10` to `0.90` in `0.05` increments (`17` values total)
- Score method used in current experiment wrapper: `zscore`
- Residual score is based on absolute residual magnitude
- `zscore` mode:
  - compute absolute residuals
  - z-score them within the evaluation frame
  - clip to `[-50, 50]`
  - map through sigmoid to `[0, 1]`
- Best threshold chosen by maximum F1 on the same test segment
- Ties in F1 are broken toward threshold closest to `0.5`
- Repetitive single-class sklearn warnings are intentionally suppressed in the bridge
- Bridge defensively filters stale prediction files by `expected_series_ids` and deduplicates by `series_id`, keeping the newest file

Timeseries presets:

| preset | target_feature | exogenous_features |
|---|---|---|
| sarima_rms | agg_mean_rms | none |
| sarimax_rms_std | agg_mean_rms | agg_std_rms, y_lagged |
| sarimax_gamma | agg_mean_gamma_high_power | agg_std_gamma_high_power, y_lagged |
| sarimax_hjorth | agg_mean_hjorth_activity | agg_std_hjorth_activity, y_lagged |

Important apples-to-oranges caveat:
- Timeseries train/test split is chronological within each run.
- ML and DL train/test split is LOSO by subject.
- Keep this caveat explicit in the report whenever cross-family results are shown.

### 5.3 Machine learning family

Default config from `configs/experiments/ml_models.json`:
- `threshold = 0.5`
- `random_state = 42`
- `n_jobs = 1`
- `optuna_trials = 20`
- `cv_splits = 3`
- `svm_rfe_min_features = 12`
- `shap_sample_size = 256`

Implementation notes from `src/ds003029_eda/experiments/ml.py`:
- Group-aware CV helper prefers `StratifiedGroupKFold` when available and enough unique groups exist; otherwise falls back to `StratifiedKFold`.
- Prediction scoring uses `predict_proba` when available, otherwise a sigmoid-transformed `decision_function`.
- Per-fold reports can include feature importance tables and SHAP importance tables when supported.

ML presets and actual modeling details:

| preset | main details |
|---|---|
| xgboost_optuna | XGBoost with Optuna tuning; search over `n_estimators 200-600`, `max_depth 3-8`, `learning_rate 0.01-0.2 log`, `subsample 0.6-1.0`, `colsample_bytree 0.6-1.0`, `min_child_weight 1-10`, `reg_alpha 1e-6 to 1 log`, `reg_lambda 1e-4 to 10 log`; objective `binary:logistic`; eval metric `auc`; tree method `hist`; `scale_pos_weight` derived from class ratio |
| lightgbm_dart | LightGBM classifier, `boosting_type=dart`; base params include `n_estimators=400`, `learning_rate=0.05`, `num_leaves=63`, `subsample=0.85`, `colsample_bytree=0.8`; preset overrides to `n_estimators=500`, `learning_rate=0.05`, `num_leaves=63` |
| catboost | CatBoost classifier with `loss_function=Logloss`, `eval_metric=AUC`, base params `iterations=500`, `depth=6`, `learning_rate=0.05`, `l2_leaf_reg=3.0`; preset overrides to `iterations=600`, `depth=6`, `learning_rate=0.05` |
| stacking | Sklearn `StackingClassifier`; base estimators `RandomForestClassifier`, LightGBM-DART, and LogisticRegression; final estimator LogisticRegression; uses `predict_proba` stacking with `passthrough=True` |
| svm_rbf_rfe | Pipeline of `RFECV(LogisticRegression)` feature selector plus `SVC(kernel='rbf', probability=True)`; RFECV uses `step=0.2`, minimum selected features `12`, scoring `roc_auc`; preset hyperparams `C=2.0`, `gamma=scale` |

Natural subgrouping for ablation:
- Boosting models:
  - xgboost_optuna
  - lightgbm_dart
  - catboost
- Non-boosting models:
  - stacking
  - svm_rbf_rfe

### 5.4 Deep learning family

Default config from `configs/experiments/dl_models.json`:
- `batch_size = 64`
- `num_workers = 2`
- `epochs = 20`
- `learning_rate = 0.001`
- `weight_decay = 0.0001`
- `threshold = 0.5`
- `random_state = 42`
- `early_stopping_patience = 5`
- `device = auto`
- `mixed_precision = auto`
- `pin_memory = true`
- `non_blocking_transfers = true`
- `persistent_workers = true`
- `allow_tf32 = true`
- `cudnn_benchmark = true`
- `raw_loading_strategy = auto`

DL training logic from `src/ds003029_eda/experiments/dl.py`:
- Validation split is carved from the training fold only, not from the held-out test fold.
- Validation split uses `StratifiedShuffleSplit` when possible.
- Default `val_fraction = 0.15`.
- Validation split has safety guards:
  - at least `4` validation samples if possible
  - capped at `40%` of the training set
- Loss is `BCEWithLogitsLoss` with `pos_weight = n_neg / n_pos` computed from the full training fold.
- Optimizer is `AdamW`.
- Scheduler is `CosineAnnealingLR`.
- Early stopping monitors validation ROC-AUC, with validation AP as fallback if ROC-AUC is not finite.

Observed best-DL run config note from `eda_outputs/experiments/dl/ce_tss_transformer/run_config.json`:
- CUDA available: `true`
- Resolved device: `cuda`
- GPU: `NVIDIA GeForce RTX 3090`
- AMP dtype: `bfloat16`
- Validation fraction: `0.15`

DL presets by input type:

Raw-signal models:

| preset | input_mode | notes |
|---|---|---|
| eegnet | raw | preset `dropout=0.25` |
| eegwavenet | raw | default preset |
| cnn_bilstm | raw | default preset |
| bendr | raw | repo adapter, no upstream pretrained checkpoint bundled |
| reve | raw | transformer-style repo adapter |
| biseizurere_proxy | raw | exposed through preset key `biseizurere`; repo uses a proxy / surrogate implementation |

Channel-feature models:

| preset | input_mode | notes |
|---|---|---|
| inresformer | channel | inception-style front-end plus transformer encoder |
| gat_bilstm | channel | graph-attention plus recurrent adapter |
| ce_tss_transformer | channel | channel / temporal / spectral style transformer adapter |
| dbconformer | channel | dual-branch conformer-style adapter |
| graphs4mer | channel | graph sequence model adapter |
| dcrnn | channel | classifier adaptation of graph recurrent mixing |

Implementation fidelity note from `docs/MODEL_IMPLEMENTATION_NOTES.md`:
- ML and timeseries models are direct library-backed implementations.
- Most DL models are paper-inspired or repo-inspired adapters for current tensor formats.
- `BISeizureRe` is explicitly a surrogate because no public canonical implementation was available.

## 6. Reproducibility Notes

### 6.1 Core package requirements

From `requirements.txt`:
- numpy >= 1.26
- pandas >= 2.2
- scipy >= 1.12
- matplotlib >= 3.8
- scikit-learn >= 1.5
- joblib >= 1.4
- statsmodels >= 0.14
- ruptures >= 1.1.9
- mne >= 1.8
- xgboost >= 2.1
- lightgbm >= 4.5
- catboost >= 1.2.7
- optuna >= 4.0
- shap >= 0.46
- pytest >= 8.3

PyTorch note from requirements:
- CUDA-enabled PyTorch is intentionally not pinned in `requirements.txt`; install separately in the working conda environment.

### 6.2 Canonical staged commands

Useful commands to cite or reproduce in the appendix:

Refresh metadata manifests:

```bash
python tools/workspace_content.py --workspace-root /path/to/Seizure-Dectection-using-ECoG-
```

Run direction-specific data processing:

```bash
python tools/workspace_data.py timeseries --workspace-root /path/to/Seizure-Dectection-using-ECoG-
python tools/workspace_data.py ml --workspace-root /path/to/Seizure-Dectection-using-ECoG-
python tools/workspace_data.py dl --workspace-root /path/to/Seizure-Dectection-using-ECoG-
```

Run one preset:

```bash
python tools/workspace_experiment.py timeseries --preset sarimax_gamma --workspace-root /path/to/Seizure-Dectection-using-ECoG-
python tools/workspace_experiment.py ml --preset catboost --workspace-root /path/to/Seizure-Dectection-using-ECoG-
python tools/workspace_experiment.py dl --preset ce_tss_transformer --workspace-root /path/to/Seizure-Dectection-using-ECoG- --device cuda
```

Rebuild summaries and verification:

```bash
python tools/workspace_reports.py summarize --family all --workspace-root /path/to/Seizure-Dectection-using-ECoG-
python tools/workspace_reports.py cross_family_leaderboard --workspace-root /path/to/Seizure-Dectection-using-ECoG-
python tools/workspace_reports.py verify --family all --workspace-root /path/to/Seizure-Dectection-using-ECoG-
```

### 6.3 Verification status after final refresh

From `eda_outputs/experiments/verification/verification_summary.md`:
- Total experiments checked: `21`
- Errors: `0`
- Warnings: `0`

Family counts:
- DL: `12`
- ML: `5`
- Timeseries: `4`

## 7. Results and Evidence Notes

### 7.1 One-paragraph result story to later rewrite

Condensed result story:
- Classical ML is strongest overall, with CatBoost ranking first on both ROC-AUC and PR-AUC.
- The best deep model, CE-TSS-Transformer, is competitive in ROC-AUC but still behind CatBoost and below at least one additional boosting baseline.
- Timeseries models can be made comparable through residual-derived anomaly scoring, but even the best SARIMAX variant has very low PR-AUC compared with ML and DL.
- Within DL, channel-feature models clearly outperform raw-signal models under the current cohort size and training setup.
- Within timeseries, exogenous feature choice matters a lot; `sarimax_gamma` is the only timeseries preset that becomes competitively useful on ROC-AUC.

### 7.2 Cross-family leaderboard by ROC-AUC

Source file:
- `eda_outputs/experiments/summary/cross_family_leaderboard.csv`

Top 10 by ROC-AUC:

| rank | family | preset | roc_auc | average_precision | note |
|---|---|---|---:|---:|---|
| 1 | ML | catboost | 0.879939 | 0.872341 | native_classifier |
| 2 | DL | ce_tss_transformer | 0.859054 | 0.775192 | native_classifier |
| 3 | ML | lightgbm_dart | 0.848153 | 0.813661 | native_classifier |
| 4 | TIMESERIES | sarimax_gamma | 0.840058 | 0.180413 | derived_from_residuals |
| 5 | ML | xgboost_optuna | 0.839385 | 0.770200 | native_classifier |
| 6 | DL | inresformer | 0.824945 | 0.717238 | native_classifier |
| 7 | ML | stacking | 0.824755 | 0.781874 | native_classifier |
| 8 | DL | dbconformer | 0.808755 | 0.706051 | native_classifier |
| 9 | DL | dcrnn | 0.808199 | 0.727424 | native_classifier |
| 10 | DL | gat_bilstm | 0.800859 | 0.743200 | native_classifier |

Key ranking note:
- Best-of-family ROC order is `ML > DL > TIMESERIES`.

### 7.3 Cross-family leaderboard by PR-AUC

Top 5 by average precision:

| rank | family | preset | average_precision | roc_auc |
|---|---|---|---:|---:|
| 1 | ML | catboost | 0.872341 | 0.879939 |
| 2 | ML | lightgbm_dart | 0.813661 | 0.848153 |
| 3 | ML | stacking | 0.781874 | 0.824755 |
| 4 | DL | ce_tss_transformer | 0.775192 | 0.859054 |
| 5 | ML | xgboost_optuna | 0.770200 | 0.839385 |

Key ranking note:
- Timeseries does not appear near the top of PR-AUC because residual anomaly scores generate weak precision-recall discrimination relative to native classifiers.

### 7.4 Full ML result table

Source file:
- `eda_outputs/experiments/summary/ml_metrics_summary.csv`

| model | roc_auc_mean | average_precision_mean | f1_mean | precision_mean | sensitivity_mean | specificity_mean | accuracy_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| catboost | 0.879939 | 0.872341 | 0.653065 | 0.730848 | 0.724765 | 0.831538 | 0.761983 |
| lightgbm_dart | 0.848153 | 0.813661 | 0.632013 | 0.716562 | 0.718239 | 0.808558 | 0.739343 |
| stacking | 0.824755 | 0.781874 | 0.633201 | 0.769165 | 0.634296 | 0.824165 | 0.748784 |
| svm_rbf_rfe | 0.698627 | 0.639571 | 0.525497 | 0.556919 | 0.635925 | 0.742879 | 0.673792 |
| xgboost_optuna | 0.839385 | 0.770200 | 0.617331 | 0.702779 | 0.701308 | 0.807545 | 0.733080 |

Interpretation notes:
- CatBoost is best on ROC-AUC, PR-AUC, and F1.
- LightGBM-DART is second-best on both ROC-AUC and PR-AUC.
- Stacking has strong PR-AUC and the highest precision among ML models shown here, but lower sensitivity than CatBoost.
- SVM-RBF-RFE is the weakest ML baseline.

### 7.5 Full DL result table

Source file:
- `eda_outputs/experiments/summary/dl_metrics_summary.csv`

| model | roc_auc_mean | average_precision_mean | f1_mean | precision_mean | sensitivity_mean | specificity_mean | accuracy_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| ce_tss_transformer | 0.859054 | 0.775192 | 0.621795 | 0.653369 | 0.697631 | 0.795713 | 0.741562 |
| inresformer | 0.824945 | 0.717238 | 0.608445 | 0.660183 | 0.683802 | 0.842974 | 0.757224 |
| dbconformer | 0.808755 | 0.706051 | 0.588937 | 0.658354 | 0.619616 | 0.859642 | 0.759304 |
| dcrnn | 0.808199 | 0.727424 | 0.527305 | 0.600494 | 0.637866 | 0.738190 | 0.681638 |
| gat_bilstm | 0.800859 | 0.743200 | 0.547321 | 0.611517 | 0.656531 | 0.778575 | 0.701345 |
| graphs4mer | 0.779103 | 0.694914 | 0.536921 | 0.617983 | 0.642716 | 0.782402 | 0.692439 |
| eegnet | 0.778003 | 0.734094 | 0.464557 | 0.638680 | 0.537187 | 0.810726 | 0.679844 |
| cnn_bilstm | 0.709411 | 0.575763 | 0.169996 | 0.448586 | 0.247593 | 0.832996 | 0.544525 |
| eegwavenet | 0.628458 | 0.556199 | 0.295890 | 0.287883 | 0.450297 | 0.591853 | 0.518762 |
| bendr | 0.568553 | 0.485173 | 0.188488 | 0.297573 | 0.201610 | 0.860492 | 0.623323 |
| biseizurere_proxy | 0.512461 | 0.395516 | 0.144931 | 0.103427 | 0.375000 | 0.625000 | 0.442138 |
| reve | 0.488406 | 0.382118 | 0.404178 | 0.304175 | 0.750000 | 0.250000 | 0.468634 |

Interpretation notes:
- CE-TSS-Transformer is clearly the best DL model on ROC-AUC, PR-AUC, and F1.
- InResFormer is the second-best overall DL baseline and has strong specificity.
- DBConformer has the highest accuracy among DL models listed here, but lower ROC-AUC / PR-AUC than the top two.
- Feature-channel models dominate the upper half of the DL ranking.
- Raw-signal models are concentrated in the lower half, except EEGNet, which is the strongest raw-signal baseline.
- REVE appears degenerate in operating point behavior: high sensitivity (`0.75`) with very low specificity (`0.25`).

### 7.6 Timeseries classification table

Source file:
- `eda_outputs/experiments/summary/cross_family_leaderboard.csv`

Current comparable classification results for timeseries are residual-derived.

| preset | roc_auc | average_precision | f1 | precision | sensitivity | specificity | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| sarimax_gamma | 0.840058 | 0.180413 | 0.195397 | 0.205611 | 0.765256 | 0.681039 | 0.672713 |
| sarimax_rms_std | 0.397624 | 0.054114 | 0.099277 | 0.065812 | 0.877131 | 0.500437 | 0.550152 |
| sarimax_hjorth | 0.353833 | 0.072525 | 0.110076 | 0.106294 | 0.679205 | 0.581149 | 0.604250 |
| sarima_rms | 0.080449 | 0.036598 | 0.082790 | 0.054544 | 0.855417 | 0.418381 | 0.470080 |

Interpretation notes:
- `sarimax_gamma` is the only timeseries preset that becomes competitive on ROC-AUC.
- Even for `sarimax_gamma`, PR-AUC remains very low (`0.1804`) relative to ML and DL.
- The timeseries bridge tends to produce high sensitivity but poor precision, consistent with anomaly-style scoring.

### 7.7 Timeseries forecast table

Source file:
- `eda_outputs/experiments/summary/timeseries_metrics_summary.csv`

Forecast-fit metrics remain useful for the timeseries section, but they should not be confused with seizure classification quality.

| preset | exogenous_cols | rmse_test_mean | mae_test_mean | smape_test_pct_mean | r2_test_mean | anomaly_count_test_mean |
|---|---|---:|---:|---:|---:|---:|
| sarimax_hjorth | agg_std_hjorth_activity, y_lagged | 3.6159e-08 | 2.5584e-08 | 74.7165 | -1.5487 | 50.3750 |
| sarimax_gamma | agg_std_gamma_high_power, y_lagged | 4.2973e-06 | 2.3494e-06 | 90.1179 | -24.9513 | 40.8750 |
| sarimax_rms_std | agg_std_rms, y_lagged | 5.1907e-05 | 4.1712e-05 | 34.5963 | -2.0994 | 41.1875 |
| sarima_rms | none | 6.9802e-05 | 5.5457e-05 | 47.6519 | -2.8175 | 57.7500 |

Important narrative point:
- Best forecast fit does not equal best seizure-discrimination quality.
- `sarimax_hjorth` is best by RMSE, but `sarimax_gamma` is far better by ROC-AUC.
- This mismatch is a major discussion point and supports using both forecast metrics and classification metrics in the timeseries section.

### 7.8 Verification and completeness notes

Use these claims safely:
- All `21` experiment outputs verified successfully.
- No verification errors and no verification warnings remained after the final refresh.
- Timeseries bridge duplication bug was fixed by filtering expected `series_id` values and deduplicating stale prediction CSVs.

## 8. Ablation and Comparative Notes

### 8.1 Timeseries ablation against the SARIMA baseline

Baseline for comparison:
- `sarima_rms`

Residual-classification deltas relative to baseline:

| preset | delta_roc_auc_vs_sarima_rms | delta_average_precision_vs_sarima_rms |
|---|---:|---:|
| sarimax_gamma | +0.759610 | +0.143815 |
| sarimax_rms_std | +0.317176 | +0.017515 |
| sarimax_hjorth | +0.273384 | +0.035927 |

Interpretation notes:
- Exogenous feature choice matters strongly for timeseries utility.
- Gamma-derived target / exogenous pairing is the only timeseries variant that meaningfully improves ROC-AUC into a competitive range.
- Simple RMS augmentation and Hjorth augmentation improve over the baseline, but not enough to become strong classifiers.

### 8.2 ML subgroup ablation: boosting vs non-boosting

Group means from current result synthesis:

| ML subgroup | mean ROC-AUC | mean average precision |
|---|---:|---:|
| boosting models | 0.855826 | 0.818734 |
| non-boosting models | 0.761691 | 0.710723 |

Interpretation notes:
- Boosting-based methods dominate the ML family on this problem setup.
- This supports a practical conclusion that aggregate engineered features pair especially well with gradient-boosted tree ensembles on the current cohort size.

### 8.3 DL subgroup ablation: channel-feature vs raw-signal

Group means from current result synthesis:

| DL subgroup | mean ROC-AUC | mean average precision |
|---|---:|---:|
| channel-feature models | 0.813486 | 0.727336 |
| raw-signal models | 0.614215 | 0.521477 |

Interpretation notes:
- Channel-feature DL models are clearly stronger than raw-signal DL models on the current experimental setup.
- This likely reflects a combination of small cohort size, easier optimization on structured engineered features, and stronger inductive bias from channel-level summary features.

### 8.4 Best-of-family comparison

Best model in each family:

| family | best preset | ROC-AUC | average precision |
|---|---|---:|---:|
| ML | catboost | 0.879939 | 0.872341 |
| DL | ce_tss_transformer | 0.859054 | 0.775192 |
| TIMESERIES | sarimax_gamma | 0.840058 | 0.180413 |

Interpretation notes:
- ROC-AUC best-of-family order: ML > DL > Timeseries.
- PR-AUC best-of-family order: ML > DL >>> Timeseries.
- The giant PR-AUC gap between DL and timeseries is central evidence that residual anomaly scores are a weak proxy for window-level seizure classification in this setup.

## 9. Discussion and Limitation Notes

### 9.1 Why ML may be winning here

Possible reasons to discuss:
- Small effective cohort after readiness filtering.
- Strong handcrafted aggregate features already summarize useful seizure structure.
- Boosting handles nonlinear interactions and heterogeneous feature distributions well.
- LOSO subject shift may punish high-capacity DL models more than tree ensembles.

### 9.2 Why raw-signal DL may be lagging

Possible reasons to discuss:
- Limited number of subjects for subject-generalized raw-signal learning.
- Strong variation in seizure prevalence across held-out subjects.
- Raw windows may need more data, augmentation, or pretraining than currently available.
- Current repo uses architecture adapters rather than original upstream training ecosystems for many DL models.

### 9.3 Why timeseries PR-AUC is weak

Likely reasons:
- Forecast residual magnitude is only an indirect proxy for seizure state.
- Threshold is chosen per-run on the same test segment used for evaluation.
- Chronological in-run split does not match the subject-held-out discrimination setting used by ML/DL.
- Forecast fit on scalar aggregate trajectories does not guarantee class separation.

### 9.4 Explicit limitations to keep in the final report

Methodological limitations:
- Cross-family comparison is useful but not perfectly apples-to-apples because split policy differs for timeseries.
- Timeseries metrics in ROC-AUC / PR-AUC are derived from residual anomaly scores, not native predicted seizure probabilities.
- No statistical significance testing has been run on pairwise model differences yet.
- Hyperparameter search depth is modest for some families, for example `20` Optuna trials for XGBoost.

Data limitations:
- Final cohort is only `8` subjects and `16` runs.
- Subject-level prevalence is highly imbalanced across folds.
- Final cohort is a filtered model-ready subset, not the full raw BIDS collection.

Implementation limitations:
- Most DL models are architecture-faithful adapters rather than exact upstream ports.
- `BISeizureRe` is only a surrogate implementation.
- The current repo does not yet provide a stronger, dedicated raw-window extraction and pretraining workflow beyond the present setup.

### 9.5 Honest conclusion notes

Safe conclusion candidates:
- On the final model-ready ds003029 cohort, engineered aggregate features with boosting provide the strongest overall seizure-detection performance.
- Channel-feature deep models are viable and sometimes competitive, but they do not surpass CatBoost in the current setup.
- Residual-derived SARIMAX adds interpretability and cross-family breadth, but it is not yet competitive on PR-AUC and should be framed as a secondary comparison track rather than the strongest detector.

## 10. Figure and Table Checklist for the Final Report

### 10.1 Tables that should appear

Table candidates:
- Dataset / cohort summary table
  - Use subject-level cohort table from Section 3.2.
- LOSO fold table
  - Use Section 3.4.
- Model inventory table
  - One compact table by family or three small tables from Section 5.
- Main results table by family
  - Use Sections 7.4, 7.5, 7.6.
- Cross-family top-model comparison table
  - Use Section 8.4.
- Ablation table
  - Use Section 8.

### 10.2 Figures that should appear

Recommended figures:
- End-to-end pipeline figure
  - Adapt the flow in `docs/data_processing_v2.md`.
- Class balance by held-out fold
  - Build from `fold_manifest.json`.
- Cross-family bar chart for best ROC-AUC and PR-AUC by family
  - Build from `cross_family_leaderboard.csv`.
- Per-family ranking bar plots
  - ML, DL, timeseries separately.
- ROC and PR curves for top few models
  - CatBoost, CE-TSS-Transformer, sarimax_gamma.
- Example timeseries run figure
  - actual vs predicted scalar series plus residual / anomaly score for one representative run.
- Ablation chart
  - boosting vs non-boosting, channel-feature vs raw, SARIMA baseline vs SARIMAX variants.

### 10.3 Source files for plots

Useful plotting sources:
- `eda_outputs/experiments/ml/<preset>/all_predictions.csv`
- `eda_outputs/experiments/dl/<preset>/all_predictions.csv`
- `eda_outputs/experiments/timeseries/<preset>/sarima_classification_metrics.csv`
- `eda_outputs/experiments/timeseries/<preset>/sarima_results/predictions/*_sarima_predictions.csv`
- `eda_outputs/experiments/summary/cross_family_leaderboard.csv`

## 11. Open Decisions Before Writing Final Prose

Decisions to settle:
- Decide whether the paper framing is benchmark-focused or CatBoost-as-proposed-method focused.
- Decide whether timeseries should be positioned as a full comparison family or as an auxiliary anomaly-based baseline due to split differences.
- Decide whether raw-signal DL models should be described as a core result block or mainly as an ablation showing the benefit of engineered channel features.
- Decide how many DL models to discuss in depth; likely top 3-4 is enough in prose, while full tables can stay in appendix.
- Decide whether to add significance testing or confidence intervals before final submission.

## 12. Fast Assembly Plan for the Final Report

Practical writing order:
- Write Introduction from Sections 1 and 2.
- Write Dataset and preprocessing from Sections 3 and 4.
- Write model methodology from Section 5.
- Drop in full results tables from Section 7.
- Write ablation subsection from Section 8.
- Write limitations and conclusion from Section 9.
- Add citations by mapping method families to `model _sources.md` and checking canonical papers manually.

## 13. One-page Summary Notes for Fast Recall

If only a few facts are remembered, keep these:
- Final cohort: `8` subjects, `16` runs, `7582` evaluable windows.
- Best ML: `catboost`, ROC-AUC `0.879939`, PR-AUC `0.872341`.
- Best DL: `ce_tss_transformer`, ROC-AUC `0.859054`, PR-AUC `0.775192`.
- Best timeseries derived classifier: `sarimax_gamma`, ROC-AUC `0.840058`, PR-AUC `0.180413`.
- Best forecast RMSE: `sarimax_hjorth`, but best forecast fit did not produce best classification utility.
- ML boosting mean beats ML non-boosting mean.
- DL channel-feature mean beats DL raw-signal mean by a large margin.
- Verification after final refresh: `21` experiments checked, `0` errors, `0` warnings.
