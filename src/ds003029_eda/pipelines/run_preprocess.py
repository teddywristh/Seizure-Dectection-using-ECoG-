from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from ..data.io import line_frequency_from_row, load_raw_run, read_run_inventory
from ..data.preprocess import PreprocessConfig, preprocess_raw
from ..paths import WorkspacePaths, get_paths


@dataclass(frozen=True)
class PreprocessPipelineConfig:
    artifact_subdir: str = "data_processing_v2"
    overwrite: bool = False
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


def _artifact_root(paths: WorkspacePaths, artifact_subdir: str) -> Path:
    return paths.outputs_dir / artifact_subdir


def _safe_stem(base: str) -> str:
    return Path(str(base)).name


def _md5_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.md5(path.read_bytes()).hexdigest()


def _build_qc_html(summary_df: pd.DataFrame, qc_df: pd.DataFrame) -> str:
    by_subject = (
        summary_df[summary_df["preprocess_ok"].astype(bool)]
        .groupby("subject", as_index=False)
        .agg(
            n_runs=("base", "nunique"),
            n_channels_good_mean=("n_channels_good", "mean"),
            n_bad_channels_mean=("n_final_bads", "mean"),
        )
        .sort_values("subject", kind="mergesort")
    )

    bad_counts = (
        qc_df[qc_df["final_bad"].astype(bool)]
        .groupby(["subject", "base"], as_index=False)
        .agg(n_flagged_channels=("channel", "nunique"))
        .sort_values(["subject", "base"], kind="mergesort")
    )

    return "\n".join(
        [
            "<html><head><meta charset='utf-8'><title>qc_report</title></head><body>",
            "<h1>ds003029 Data Processing v2 QC Report</h1>",
            "<h2>Subject Summary</h2>",
            by_subject.to_html(index=False, float_format=lambda v: f"{v:.3f}"),
            "<h2>Run Summary</h2>",
            summary_df.to_html(index=False, float_format=lambda v: f"{v:.3f}"),
            "<h2>Flagged Channels</h2>",
            bad_counts.to_html(index=False),
            "</body></html>",
        ]
    )


def run_preprocess_pipeline(
    *,
    paths: WorkspacePaths | None = None,
    config: PreprocessPipelineConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    paths = paths or get_paths()
    config = config or PreprocessPipelineConfig()

    runs_df, _ = read_run_inventory(paths)
    if runs_df.empty:
        raise RuntimeError(
            "No runs matched preprocessing inventory. Confirm ds003029_run_summary.csv and "
            "ds003029_seizure_intervals_by_run.csv exist and that real .eeg content is available under the selected workspace root."
        )

    artifact_root = _artifact_root(paths, config.artifact_subdir)
    preprocessed_dir = artifact_root / "preprocessed"
    reports_dir = artifact_root / "reports"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    qc_tables: list[pd.DataFrame] = []

    for _, row in runs_df.iterrows():
        base = str(row.get("base", ""))
        stem = _safe_stem(base)
        cache_path = preprocessed_dir / f"{stem}_preproc_raw.fif"

        try:
            raw, load_meta = load_raw_run(row, paths=paths, preload=True)
            line_freq = line_frequency_from_row(row, load_meta.get("ieeg_json", {}))
            preprocessed_raw, qc_df, preprocess_meta = preprocess_raw(
                raw,
                config=config.preprocess,
                line_freq=line_freq,
            )

            if config.overwrite or not cache_path.exists():
                preprocessed_raw.save(cache_path, overwrite=True, verbose="ERROR")

            qc_df = qc_df.copy()
            qc_df.insert(0, "base", base)
            qc_df.insert(1, "subject", str(row.get("subject", "")))
            qc_tables.append(qc_df)

            summary_rows.append(
                {
                    "subject": str(row.get("subject", "")),
                    "session": str(row.get("session", "")),
                    "run": str(row.get("run", "")),
                    "base": base,
                    "vhdr": str(row.get("vhdr", "")),
                    "cache_path": str(cache_path),
                    "line_freq": preprocess_meta["line_freq"],
                    "n_intervals": int(row.get("n_intervals", 0)),
                    "duration_s": float(preprocessed_raw.times[-1]) if len(preprocessed_raw.times) else 0.0,
                    "sfreq_in": preprocess_meta["sfreq_in"],
                    "sfreq_out": preprocess_meta["sfreq_out"],
                    "n_channels_in": preprocess_meta["n_channels_in"],
                    "n_channels_out": preprocess_meta["n_channels_out"],
                    "n_channels_good": preprocess_meta["n_channels_good"],
                    "n_initial_bads": int(len(preprocess_meta["initial_bads"])),
                    "n_detected_bads": int(len(preprocess_meta["detected_bads"])),
                    "n_final_bads": int(len(preprocess_meta["final_bads"])),
                    "notch_freqs": json.dumps(preprocess_meta["notch_freqs"]),
                    "bandpass": json.dumps(preprocess_meta["bandpass"]),
                    "reference_mode": str(preprocess_meta["reference_mode"]),
                    "preprocess_ok": True,
                    "error": "",
                }
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "subject": str(row.get("subject", "")),
                    "session": str(row.get("session", "")),
                    "run": str(row.get("run", "")),
                    "base": base,
                    "vhdr": str(row.get("vhdr", "")),
                    "cache_path": str(cache_path),
                    "line_freq": row.get("line_freq", None),
                    "n_intervals": int(row.get("n_intervals", 0)),
                    "duration_s": float("nan"),
                    "sfreq_in": row.get("sfreq", float("nan")),
                    "sfreq_out": float("nan"),
                    "n_channels_in": row.get("n_channels", float("nan")),
                    "n_channels_out": float("nan"),
                    "n_channels_good": float("nan"),
                    "n_initial_bads": float("nan"),
                    "n_detected_bads": float("nan"),
                    "n_final_bads": float("nan"),
                    "notch_freqs": "[]",
                    "bandpass": "[]",
                    "reference_mode": config.preprocess.reference_mode,
                    "preprocess_ok": False,
                    "error": str(exc),
                }
            )

    if not summary_rows:
        raise RuntimeError("Preprocessing pipeline did not produce any run-level records.")

    summary_df = pd.DataFrame(summary_rows).sort_values(["subject", "base"], kind="mergesort").reset_index(drop=True)
    qc_df = pd.concat(qc_tables, ignore_index=True) if qc_tables else pd.DataFrame()

    summary_path = artifact_root / "preprocess_run_summary.csv"
    qc_path = artifact_root / "bad_channels.csv"
    manifest_path = artifact_root / "manifest_preprocess.json"
    html_path = reports_dir / "qc_report.html"

    summary_df.to_csv(summary_path, index=False)
    qc_df.to_csv(qc_path, index=False)
    html_path.write_text(_build_qc_html(summary_df, qc_df), encoding="utf-8")

    payload = {
        "artifact_root": str(artifact_root),
        "config": asdict(config),
        "input_hashes": {
            "run_summary": _md5_file(paths.outputs_dir / "ds003029_run_summary.csv"),
            "seizure_intervals": _md5_file(paths.outputs_dir / "ds003029_seizure_intervals_by_run.csv"),
        },
        "outputs": {
            "preprocess_run_summary": str(summary_path),
            "bad_channels": str(qc_path),
            "qc_report_html": str(html_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return summary_df, qc_df, artifact_root