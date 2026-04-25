from __future__ import annotations

import argparse
from pathlib import Path
import sys

ws = Path.cwd().resolve()
src_dir = (ws / "src" if (ws / "src").exists() else ws.parent / "src").resolve()
sys.path.insert(0, str(src_dir))

from ds003029_eda.paths import get_paths
from ds003029_eda.window_features_multirun import WindowingConfig, build_multirun_window_features


def _console_safe(text: object) -> str:
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build full-duration window features for every run with EEG content and seizure intervals."
    )
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--max-channels", type=int, default=16)
    parser.add_argument(
        "--output-name",
        default="ds003029_window_features_multirun_full.csv",
    )
    parser.add_argument(
        "--output-info-name",
        default="ds003029_windowing_multirun_full_info.csv",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = get_paths()
    features_df, info_df = build_multirun_window_features(
        paths=paths,
        config=WindowingConfig(
            window_sec=args.window_sec,
            step_sec=args.step_sec,
            max_channels=args.max_channels,
        ),
        output_name=args.output_name,
        output_info_name=args.output_info_name,
    )

    print("=" * 72)
    print("MULTIRUN FULL-DURATION FEATURE EXTRACTION COMPLETED")
    print("=" * 72)
    print(f"Runs processed: {info_df['base'].nunique()}")
    print(f"Total windows: {len(features_df)}")
    print(f"Positive-window rate: {features_df['y'].mean():.4f}")
    print()
    printable_df = info_df[
        ["base", "duration_s", "n_channels_used", "n_windows", "n_positive_windows"]
    ].copy()
    printable_df["base"] = printable_df["base"].map(_console_safe)
    print(_console_safe(printable_df.to_string(index=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
