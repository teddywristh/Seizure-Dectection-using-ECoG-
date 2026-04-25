from __future__ import annotations

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from ds003029_eda.paths import get_paths, resolve_workspace_root
from ds003029_eda.pipelines.run_raw_windows import RawWindowExportConfig, run_raw_window_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export raw fold tensors from cached preprocessed FIF files.")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--artifact-subdir", default="data_processing_v2")
    parser.add_argument("--output-subdir", default="raw_folds")
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--output-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = get_paths(resolve_workspace_root(args.workspace_root))
    manifest_df, output_root = run_raw_window_export(
        paths=paths,
        config=RawWindowExportConfig(
            artifact_subdir=args.artifact_subdir,
            output_subdir=args.output_subdir,
            overwrite=args.overwrite,
            window_sec=args.window_sec,
            output_dtype=args.output_dtype,
        ),
    )
    print(f"Workspace root: {paths.workspace.as_posix()}")
    print(f"Exported folds: {len(manifest_df)}")
    print(f"Output root: {output_root.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())