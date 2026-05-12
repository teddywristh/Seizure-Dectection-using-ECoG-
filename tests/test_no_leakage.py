from __future__ import annotations

import pandas as pd

from ds003029_eda.splits.patient_cv import build_loso_folds, build_subject_table, validate_folds


def test_loso_folds_have_no_subject_overlap() -> None:
    runs_df = pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01", "sub-02", "sub-03"],
            "base": ["run-01", "run-02", "run-03", "run-04"],
            "n_intervals": [1, 1, 2, 1],
        }
    )
    subject_table = build_subject_table(runs_df)
    folds = build_loso_folds(subject_table["subject"].tolist())
    validate_folds(folds)
    assert len(folds) == 3
    for fold in folds:
        assert not (set(fold.train_subjects) & set(fold.test_subjects))


def test_loso_fold_order_is_deterministic() -> None:
    folds_first = build_loso_folds(["sub-02", "sub-01", "sub-03", "sub-01"])
    folds_second = build_loso_folds(["sub-03", "sub-02", "sub-01"])
    assert [fold.fold_id for fold in folds_first] == [fold.fold_id for fold in folds_second]