from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class PatientFold:
    fold_id: str
    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]


def build_subject_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame(columns=["subject", "n_runs", "n_intervals"])

    subject_df = (
        runs_df.groupby("subject", as_index=False)
        .agg(
            n_runs=("base", "nunique"),
            n_intervals=("n_intervals", "sum"),
        )
        .sort_values(["subject"], kind="mergesort")
        .reset_index(drop=True)
    )
    return subject_df


def build_loso_folds(subjects: Sequence[str]) -> list[PatientFold]:
    unique_subjects = tuple(sorted({str(subject) for subject in subjects if str(subject)}))
    if len(unique_subjects) < 2:
        raise ValueError("LOSO requires at least two subjects.")

    folds: list[PatientFold] = []
    for fold_idx, test_subject in enumerate(unique_subjects, start=1):
        train_subjects = tuple(subject for subject in unique_subjects if subject != test_subject)
        folds.append(
            PatientFold(
                fold_id=f"fold_{fold_idx:02d}_{test_subject}",
                train_subjects=train_subjects,
                test_subjects=(test_subject,),
            )
        )
    return folds


def validate_folds(folds: Sequence[PatientFold]) -> None:
    for fold in folds:
        train_set = set(fold.train_subjects)
        test_set = set(fold.test_subjects)
        if train_set & test_set:
            raise ValueError(f"Fold {fold.fold_id} leaks subjects across train and test: {train_set & test_set}")