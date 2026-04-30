# Model Implementation Notes

This file records how the repository-level experiment code maps to the model sources listed in `model _sources.md`.

## Timeseries

### SARIMA / SARIMAX
- Source reference: statsmodels SARIMAX plus SARIMA literature.
- Repository implementation: [src/ds003029_eda/sarima_training.py](src/ds003029_eda/sarima_training.py)
- Residual classification bridge: [src/ds003029_eda/experiments/sarima_clf_bridge.py](src/ds003029_eda/experiments/sarima_clf_bridge.py)
- Fidelity: direct statsmodels-based implementation with repository-specific experiment wiring.
- Notes: extended to support alternate scalar targets, optional exogenous regressors, changepoint detection with `ruptures`, residual anomaly flags, and residual-derived binary metrics.

### PELT changepoint detection
- Source reference: `ruptures` PELT user guide and repo.
- Repository implementation: [src/ds003029_eda/sarima_training.py](src/ds003029_eda/sarima_training.py)
- Fidelity: direct `ruptures.Pelt(model="rbf")` usage.

## Machine Learning

### XGBoost + Optuna
- Source reference: XGBoost repo/docs and Optuna examples.
- Repository implementation: [src/ds003029_eda/experiments/ml.py](src/ds003029_eda/experiments/ml.py)
- Fidelity: direct library-backed implementation with Optuna hyperparameter search.

### LightGBM + DART
- Source reference: LightGBM docs and DART paper.
- Repository implementation: [src/ds003029_eda/experiments/ml.py](src/ds003029_eda/experiments/ml.py)
- Fidelity: direct LightGBM classifier with `boosting_type="dart"`.

### CatBoost
- Source reference: CatBoost repo/docs.
- Repository implementation: [src/ds003029_eda/experiments/ml.py](src/ds003029_eda/experiments/ml.py)
- Fidelity: direct CatBoost classifier.

### Stacking ensemble
- Source reference: sklearn `StackingClassifier`.
- Repository implementation: [src/ds003029_eda/experiments/ml.py](src/ds003029_eda/experiments/ml.py)
- Fidelity: direct sklearn stacking implementation with RF, LightGBM, and LR meta-learner.

### Feature selection + SVM-RBF
- Source reference: sklearn SVC and SHAP workflows.
- Repository implementation: [src/ds003029_eda/experiments/ml.py](src/ds003029_eda/experiments/ml.py)
- Fidelity: direct sklearn SVM-RBF with RFECV-based feature selection; SHAP is emitted when the dependency is installed and the explainer is compatible.

## Deep Learning

### Raw-signal models

#### EEGNet
- Source reference: Lawhern et al. 2018 and `vlawhern/arl-eegmodels`.
- Repository implementation: [src/ds003029_eda/models/dl.py](src/ds003029_eda/models/dl.py)
- Fidelity: paper-architecture adapter for binary raw-window classification.

#### EEGWaveNet
- Source reference: `IoBT-VISTEC/EEGWaveNet`.
- Repository implementation: [src/ds003029_eda/models/dl.py](src/ds003029_eda/models/dl.py)
- Fidelity: repo-inspired adapter using multi-scale temporal convolutions for seizure windows.

#### CNN-BiLSTM
- Source reference: Mallick and Baths 2024 and related 1D CNN-BiLSTM seizure papers.
- Repository implementation: [src/ds003029_eda/models/dl.py](src/ds003029_eda/models/dl.py)
- Fidelity: paper-architecture adapter.

#### BENDR
- Source reference: `SPOClab-ca/BENDR`.
- Repository implementation: [src/ds003029_eda/models/dl.py](src/ds003029_eda/models/dl.py)
- Fidelity: classification adapter that mirrors the encoder plus contextualizer structure, but does not ship upstream pretrained checkpoints.

#### REVE
- Source reference: REVE paper/repo/project page.
- Repository implementation: [src/ds003029_eda/models/dl.py](src/ds003029_eda/models/dl.py)
- Fidelity: transformer-style classification adapter; upstream pretrained backbone integration is not bundled.

#### BISeizureRe
- Source reference: no public canonical code or paper was available in the provided source list.
- Repository implementation: [src/ds003029_eda/models/dl.py](src/ds003029_eda/models/dl.py)
- Fidelity: surrogate adapter only.
- Notes: this is the one model family in the list that cannot honestly be marked as a source-faithful implementation, because no public original implementation was available to mirror.

### Feature/channel models

#### InResformer
- Source reference: Hu et al. 2024.
- Repository implementation: [src/ds003029_eda/models/dl.py](src/ds003029_eda/models/dl.py)
- Fidelity: paper-architecture adapter with inception-style multi-kernel front-end plus transformer encoder.

#### GAT + BiLSTM
- Source reference: spatial-temporal GAT plus BiLSTM seizure papers.
- Repository implementation: [src/ds003029_eda/models/dl.py](src/ds003029_eda/models/dl.py)
- Fidelity: paper-architecture adapter.

#### CE-TSS-Transformer
- Source reference: MICCAI 2024 CE-TSS-Transformer.
- Repository implementation: [src/ds003029_eda/models/dl.py](src/ds003029_eda/models/dl.py)
- Fidelity: paper-architecture adapter with channel, temporal, and spectral branches.

#### DBConformer
- Source reference: DBConformer paper/repo.
- Repository implementation: [src/ds003029_eda/models/dl.py](src/ds003029_eda/models/dl.py)
- Fidelity: paper-architecture adapter with dual-branch conformer-style encoding.

#### GraphS4mer
- Source reference: GraphS4mer paper/repo.
- Repository implementation: [src/ds003029_eda/models/dl.py](src/ds003029_eda/models/dl.py)
- Fidelity: paper-architecture adapter.

#### DCRNN
- Source reference: DCRNN paper/repo.
- Repository implementation: [src/ds003029_eda/models/dl.py](src/ds003029_eda/models/dl.py)
- Fidelity: classifier adaptation of the recurrent graph-mixing idea.

## Practical boundary

- ML and timeseries models are direct library-backed implementations.
- Most DL models are architecture-faithful adapters for this repo's tensor formats, not byte-for-byte ports of the upstream training codebases.
- `BISeizureRe` remains a surrogate because the upstream public implementation is unavailable.