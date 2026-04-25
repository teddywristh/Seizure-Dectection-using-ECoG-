from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ds003029_eda.paths import WorkspacePaths
from ds003029_eda.pipelines.run_sarima_prep import COMBINED_OUTPUT_NAME, SarimaPrepConfig, run_sarima_prep


def test_run_sarima_prep_exports_v2_series(tmp_path) -> None:
    artifact_root = tmp_path / "eda_outputs" / "data_processing_v2"
    runs_root = artifact_root / "features" / "runs"
    runs_root.mkdir(parents=True)

    tensor_path = runs_root / "sub-demo_run-01_window_tensor.npz"
    np.savez(
        tensor_path,
        x_agg=np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=float),
        aggregate_feature_names=np.array(["agg_mean_rms", "agg_mean_line_length"], dtype="U32"),
    )

    index_path = runs_root / "sub-demo_run-01_window_index.csv"
    pd.DataFrame(
        {
            "base": [str(tmp_path / "EEG" / "demo" / "sub-demo_run-01_ieeg")] * 3,
            "subject": ["demo"] * 3,
            "window_id": [0, 1, 2],
            "t_start_s": [0.0, 0.5, 1.0],
            "t_stop_s": [2.0, 2.5, 3.0],
            "t_mid_s": [1.0, 1.5, 2.0],
            "y": [0, -1, 1],
        }
    ).to_csv(index_path, index=False)

    inventory_path = artifact_root / "run_feature_inventory.csv"
    pd.DataFrame(
        [
            {
                "subject": "demo",
                "base": str(tmp_path / "EEG" / "demo" / "sub-demo_run-01_ieeg"),
                "tensor_path": str(tensor_path),
                "index_path": str(index_path),
                "feature_ok": True,
            }
        ]
    ).to_csv(inventory_path, index=False)

    paths = WorkspacePaths(
        workspace=tmp_path,
        dataset_root=tmp_path / "EEG" / "ds003029",
        outputs_dir=tmp_path / "eda_outputs",
    )
    combined_df, manifest_df, output_root = run_sarima_prep(
        paths=paths,
        config=SarimaPrepConfig(),
    )

    assert len(combined_df) == 3
    assert combined_df["series_id"].nunique() == 1
    assert combined_df["rms"].tolist() == [1.0, 2.0, 3.0]
    assert combined_df["y"].tolist() == [0, -1, 1]
    assert manifest_df.loc[0, "n_boundary"] == 1
    assert (output_root / COMBINED_OUTPUT_NAME).exists()

    manifest_payload = json.loads((output_root / "sarima_prep_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["n_series"] == 1
    assert manifest_payload["series"][0]["n_positive"] == 1


def test_run_sarima_prep_supports_alternate_target_and_exog(tmp_path) -> None:
    artifact_root = tmp_path / "eda_outputs" / "data_processing_v2"
    runs_root = artifact_root / "features" / "runs"
    runs_root.mkdir(parents=True)

    tensor_path = runs_root / "sub-demo_run-02_window_tensor.npz"
    np.savez(
        tensor_path,
        x_agg=np.array(
            [
                [1.0, 3.0, 10.0],
                [2.0, 4.0, 20.0],
                [3.0, 5.0, 30.0],
            ],
            dtype=float,
        ),
        aggregate_feature_names=np.array(
            ["agg_mean_rms", "agg_mean_gamma_high_power", "agg_std_rms"],
            dtype="U32",
        ),
    )

    index_path = runs_root / "sub-demo_run-02_window_index.csv"
    pd.DataFrame(
        {
            "base": [str(tmp_path / "EEG" / "demo" / "sub-demo_run-02_ieeg")] * 3,
            "subject": ["demo"] * 3,
            "window_id": [0, 1, 2],
            "t_start_s": [0.0, 0.5, 1.0],
            "t_stop_s": [2.0, 2.5, 3.0],
            "t_mid_s": [1.0, 1.5, 2.0],
            "y": [0, 1, 1],
        }
    ).to_csv(index_path, index=False)

    pd.DataFrame(
        [
            {
                "subject": "demo",
                "base": str(tmp_path / "EEG" / "demo" / "sub-demo_run-02_ieeg"),
                "tensor_path": str(tensor_path),
                "index_path": str(index_path),
                "feature_ok": True,
            }
        ]
    ).to_csv(artifact_root / "run_feature_inventory.csv", index=False)

    paths = WorkspacePaths(
        workspace=tmp_path,
        dataset_root=tmp_path / "EEG" / "ds003029",
        outputs_dir=tmp_path / "eda_outputs",
    )
    combined_df, manifest_df, _ = run_sarima_prep(
        paths=paths,
        config=SarimaPrepConfig(
            aggregate_feature_name="agg_mean_gamma_high_power",
            exogenous_feature_names=("agg_std_rms", "y_lagged"),
        ),
    )

    assert combined_df["rms"].tolist() == [3.0, 4.0, 5.0]
    assert combined_df["agg_std_rms"].tolist() == [10.0, 20.0, 30.0]
    assert combined_df["y_lagged"].tolist() == [0.0, 0.0, 1.0]
    assert combined_df["target_feature"].iloc[0] == "agg_mean_gamma_high_power"
    assert manifest_df.loc[0, "target_feature"] == "agg_mean_gamma_high_power"