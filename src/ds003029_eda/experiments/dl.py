from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..datasets import FoldDataset, OnDemandRawFoldDataset, RawFoldDataset
from ..models import MODEL_INPUT_MODES, MODEL_SOURCE_METADATA, build_dl_model, torch_is_available
from ..paths import WorkspacePaths, get_paths
from .common import (
    aggregate_metrics_frame,
    compute_binary_metrics,
    ensure_experiment_directories,
    merge_prediction_frame,
    seed_everything,
    write_json,
)


@dataclass(frozen=True)
class DLExperimentConfig:
    model_name: str
    experiment_name: str
    input_mode: str
    artifact_subdir: str = "data_processing_v2"
    raw_artifact_subdir: str = "data_processing_v2/raw_folds"
    output_subdir: str = "experiments/dl"
    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    threshold: float = 0.5
    random_state: int = 42
    early_stopping_patience: int = 5
    device: str = "auto"
    mixed_precision: str = "auto"
    pin_memory: bool = True
    non_blocking_transfers: bool = True
    persistent_workers: bool = True
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    raw_loading_strategy: str = "auto"
    model_kwargs: dict[str, Any] = field(default_factory=dict)


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("torch is required for deep-learning experiments.") from exc
    return torch, nn, DataLoader


def _resolve_device(torch, requested: str):
    requested = str(requested).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False in the active environment.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device '{requested}'. Expected auto, cuda, or cpu.")


def _resolve_amp_dtype(torch, device, mode: str):
    mode = str(mode).strip().lower()
    if device.type != "cuda" or mode == "none":
        return None
    if mode == "auto":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported mixed precision mode '{mode}'. Expected auto, none, fp16, or bf16.")


def _autocast_context(torch, *, device, amp_dtype):
    if device.type != "cuda" or amp_dtype is None:
        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    return torch.cuda.amp.autocast(dtype=amp_dtype)


def _build_grad_scaler(torch, *, device, amp_dtype):
    if device.type != "cuda" or amp_dtype != torch.float16:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def _configure_cuda_backend(torch, *, device, allow_tf32: bool, cudnn_benchmark: bool) -> None:
    if device.type != "cuda":
        return
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    if hasattr(torch.backends, "cudnn"):
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if hasattr(torch, "set_float32_matmul_precision") and allow_tf32:
        torch.set_float32_matmul_precision("high")


def _load_feature_dataset(fold_dir: Path, input_mode: str):
    mode = "channel" if input_mode == "channel" else "both"
    train_dataset = FoldDataset(fold_dir / "train_dataset.npz", mode=mode)
    test_dataset = FoldDataset(fold_dir / "test_dataset.npz", mode=mode)
    train_index = pd.read_csv(fold_dir / "train_index.csv")
    test_index = pd.read_csv(fold_dir / "test_index.csv")
    return train_dataset, test_dataset, train_index, test_index


def _load_raw_dataset(raw_fold_dir: Path):
    train_dataset = RawFoldDataset(raw_fold_dir / "raw_train_dataset.npz")
    test_dataset = RawFoldDataset(raw_fold_dir / "raw_test_dataset.npz")
    return train_dataset, test_dataset


def _load_on_demand_raw_dataset(*, fold_dir: Path, artifact_root: Path, paths: WorkspacePaths):
    from ..datasets.raw_fold_dataset import infer_n_samples_from_index, load_fold_manifest

    manifest = load_fold_manifest(fold_dir / "manifest.json")
    max_channels = int(manifest["global_max_channels"])
    train_index_path = fold_dir / "train_index.csv"
    test_index_path = fold_dir / "test_index.csv"
    train_index = pd.read_csv(train_index_path)
    test_index = pd.read_csv(test_index_path)
    n_samples = infer_n_samples_from_index(train_index)
    train_dataset = OnDemandRawFoldDataset(
        train_index_path,
        artifact_root=artifact_root,
        max_channels=max_channels,
        n_samples=n_samples,
        paths=paths,
    )
    test_dataset = OnDemandRawFoldDataset(
        test_index_path,
        artifact_root=artifact_root,
        max_channels=max_channels,
        n_samples=n_samples,
        paths=paths,
    )
    return train_dataset, test_dataset, train_index, test_index


def _resolve_raw_loading_strategy(*, strategy: str, raw_fold_dir: Path) -> str:
    strategy = str(strategy).strip().lower()
    if strategy not in {"auto", "export", "on_demand"}:
        raise ValueError(f"Unsupported raw loading strategy '{strategy}'. Expected auto, export, or on_demand.")
    if strategy == "auto":
        return "export" if raw_fold_dir.exists() else "on_demand"
    if strategy == "export" and not raw_fold_dir.exists():
        raise FileNotFoundError(f"Missing raw fold tensors for {raw_fold_dir.name}. Run tools/build_raw_folds.py first or use raw_loading_strategy='on_demand'.")
    return strategy


def _build_input_spec(dataset, input_mode: str) -> dict[str, int]:
    if hasattr(dataset, "max_channels") and hasattr(dataset, "n_samples") and input_mode == "raw":
        return {"max_channels": int(dataset.max_channels), "n_samples": int(dataset.n_samples)}
    sample = dataset[0][0]
    if input_mode == "raw":
        x_raw, _ = sample
        return {"max_channels": int(x_raw.shape[0]), "n_samples": int(x_raw.shape[1])}

    x_channel, _ = sample
    return {"max_channels": int(x_channel.shape[0]), "n_features": int(x_channel.shape[1])}


def _prepare_batch(batch, *, input_mode: str, device, torch, non_blocking: bool):
    if input_mode == "raw":
        (x_raw, mask), y = batch
        features = {
            "x_raw": x_raw.to(device, non_blocking=non_blocking),
            "mask": mask.to(device, non_blocking=non_blocking),
        }
    elif input_mode == "channel":
        (x_channel, mask), y = batch
        features = {
            "x_channel": x_channel.to(device, non_blocking=non_blocking),
            "mask": mask.to(device, non_blocking=non_blocking),
        }
    else:
        (x_agg, x_channel, mask), y = batch
        features = {
            "x_agg": x_agg.to(device, non_blocking=non_blocking),
            "x_channel": x_channel.to(device, non_blocking=non_blocking),
            "mask": mask.to(device, non_blocking=non_blocking),
        }
    return features, y.float().to(device, non_blocking=non_blocking)


def _collect_scores(model, data_loader, *, input_mode: str, device, torch, criterion, amp_dtype, non_blocking: bool):
    model.eval()
    losses: list[float] = []
    y_true_blocks: list[np.ndarray] = []
    y_score_blocks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in data_loader:
            features, y = _prepare_batch(
                batch,
                input_mode=input_mode,
                device=device,
                torch=torch,
                non_blocking=non_blocking,
            )
            with _autocast_context(torch, device=device, amp_dtype=amp_dtype):
                logits = model(**features)
                loss = criterion(logits, y)
            losses.append(float(loss.item()))
            y_true_blocks.append(y.cpu().numpy().astype(int))
            y_score_blocks.append(torch.sigmoid(logits.float()).cpu().numpy())
    y_true = np.concatenate(y_true_blocks, axis=0) if y_true_blocks else np.zeros((0,), dtype=int)
    y_score = np.concatenate(y_score_blocks, axis=0) if y_score_blocks else np.zeros((0,), dtype=float)
    return float(np.mean(losses)) if losses else float("nan"), y_true, y_score


def run_dl_experiment(
    config: DLExperimentConfig,
    *,
    paths: WorkspacePaths | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not torch_is_available():
        raise ImportError("torch is required for deep-learning experiments.")

    torch, nn, DataLoader = _require_torch()
    paths = paths or get_paths()
    seed_everything(config.random_state)

    if MODEL_INPUT_MODES.get(config.model_name) != config.input_mode:
        raise ValueError(
            f"Model {config.model_name} expects input_mode='{MODEL_INPUT_MODES.get(config.model_name)}', got '{config.input_mode}'."
        )

    artifact_root = paths.outputs_dir / config.artifact_subdir
    folds_root = artifact_root / "folds"
    raw_root = paths.outputs_dir / config.raw_artifact_subdir
    directories = ensure_experiment_directories(
        paths,
        output_subdir=config.output_subdir,
        experiment_name=config.experiment_name,
    )

    device = _resolve_device(torch, config.device)
    amp_dtype = _resolve_amp_dtype(torch, device, config.mixed_precision)
    use_pin_memory = bool(config.pin_memory and device.type == "cuda")
    non_blocking = bool(config.non_blocking_transfers and device.type == "cuda")
    _configure_cuda_backend(
        torch,
        device=device,
        allow_tf32=config.allow_tf32,
        cudnn_benchmark=config.cudnn_benchmark,
    )
    write_json(
        directories.root / "run_config.json",
        {
            **asdict(config),
            "resolved_device": str(device),
            "amp_dtype": None if amp_dtype is None else str(amp_dtype).replace("torch.", ""),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "resolved_raw_loading_strategy": config.raw_loading_strategy,
        },
    )
    per_fold_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []

    for fold_dir in sorted(path for path in folds_root.iterdir() if path.is_dir() and path.name.startswith("fold_")):
        if config.input_mode == "raw":
            raw_fold_dir = raw_root / fold_dir.name
            resolved_raw_loading_strategy = _resolve_raw_loading_strategy(
                strategy=config.raw_loading_strategy,
                raw_fold_dir=raw_fold_dir,
            )
            if resolved_raw_loading_strategy == "export":
                train_dataset, test_dataset = _load_raw_dataset(raw_fold_dir)
                train_index = pd.read_csv(fold_dir / "train_index.csv")
                test_index = pd.read_csv(fold_dir / "test_index.csv")
            else:
                train_dataset, test_dataset, train_index, test_index = _load_on_demand_raw_dataset(
                    fold_dir=fold_dir,
                    artifact_root=artifact_root,
                    paths=paths,
                )
        else:
            train_dataset, test_dataset, train_index, test_index = _load_feature_dataset(fold_dir, config.input_mode)
            resolved_raw_loading_strategy = "not_applicable"

        input_spec = _build_input_spec(train_dataset, config.input_mode)
        model = build_dl_model(
            config.model_name,
            input_mode=config.input_mode,
            max_channels=input_spec["max_channels"],
            n_features=input_spec.get("n_features"),
            n_samples=input_spec.get("n_samples"),
            model_kwargs=config.model_kwargs,
        ).to(device)

        loader_kwargs: dict[str, Any] = {
            "batch_size": config.batch_size,
            "num_workers": 0 if resolved_raw_loading_strategy == "on_demand" else config.num_workers,
            "pin_memory": use_pin_memory,
        }
        if loader_kwargs["num_workers"] > 0:
            loader_kwargs["persistent_workers"] = bool(config.persistent_workers)

        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

        y_train = np.asarray(train_dataset.y.cpu().numpy(), dtype=int)
        n_pos = max(1, int(np.sum(y_train == 1)))
        n_neg = max(1, int(np.sum(y_train == 0)))
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
        scaler = _build_grad_scaler(torch, device=device, amp_dtype=amp_dtype)

        best_metric = float("-inf")
        best_epoch = 0
        best_state = None
        history_rows: list[dict[str, float]] = []
        patience_counter = 0

        for epoch in range(1, config.epochs + 1):
            model.train()
            train_losses: list[float] = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                features, y = _prepare_batch(
                    batch,
                    input_mode=config.input_mode,
                    device=device,
                    torch=torch,
                    non_blocking=non_blocking,
                )
                with _autocast_context(torch, device=device, amp_dtype=amp_dtype):
                    logits = model(**features)
                    loss = criterion(logits, y)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                train_losses.append(float(loss.item()))
            scheduler.step()

            test_loss, y_true, y_score = _collect_scores(
                model,
                test_loader,
                input_mode=config.input_mode,
                device=device,
                torch=torch,
                criterion=criterion,
                amp_dtype=amp_dtype,
                non_blocking=non_blocking,
            )
            metrics = compute_binary_metrics(y_true, y_score, threshold=config.threshold)
            history_rows.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                    "test_loss": test_loss,
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                    "f1": metrics["f1"],
                    "sensitivity": metrics["sensitivity"],
                    "specificity": metrics["specificity"],
                }
            )

            current_metric = metrics["roc_auc"] if np.isfinite(metrics["roc_auc"]) else metrics["average_precision"]
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.early_stopping_patience:
                break

        if best_state is None:
            raise RuntimeError(f"Training did not produce a valid checkpoint for fold {fold_dir.name}.")

        model.load_state_dict(best_state)
        test_loss, y_true, y_score = _collect_scores(
            model,
            test_loader,
            input_mode=config.input_mode,
            device=device,
            torch=torch,
            criterion=criterion,
            amp_dtype=amp_dtype,
            non_blocking=non_blocking,
        )
        metrics = compute_binary_metrics(y_true, y_score, threshold=config.threshold)
        metrics.update(
            {
                "fold_id": fold_dir.name,
                "model_name": config.model_name,
                "experiment_name": config.experiment_name,
                "device": str(device),
                "raw_loading_strategy": resolved_raw_loading_strategy,
                "best_epoch": float(best_epoch),
                "test_loss": test_loss,
            }
        )
        per_fold_metrics.append(metrics)

        torch.save(
            {
                "state_dict": best_state,
                "config": asdict(config),
                "source_metadata": MODEL_SOURCE_METADATA.get(config.model_name, {}),
                "resolved_device": str(device),
                "amp_dtype": None if amp_dtype is None else str(amp_dtype).replace("torch.", ""),
                "raw_loading_strategy": resolved_raw_loading_strategy,
                "input_spec": input_spec,
                "fold_id": fold_dir.name,
            },
            directories.checkpoints_dir / f"{fold_dir.name}_{config.model_name}.pt",
        )

        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(directories.reports_dir / f"{fold_dir.name}_history.csv", index=False)

        prediction_df = merge_prediction_frame(
            test_index,
            y_true=y_true,
            y_score=y_score,
            threshold=config.threshold,
            split="test",
        )
        prediction_df.insert(0, "fold_id", fold_dir.name)
        prediction_df.to_csv(directories.predictions_dir / f"{fold_dir.name}_predictions.csv", index=False)
        all_predictions.append(prediction_df)

    metrics_df = pd.DataFrame(per_fold_metrics).sort_values("fold_id", kind="mergesort")
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    aggregate_df = aggregate_metrics_frame(metrics_df)
    metrics_df.to_csv(directories.root / "fold_metrics.csv", index=False)
    predictions_df.to_csv(directories.root / "all_predictions.csv", index=False)
    aggregate_df.to_csv(directories.root / "aggregate_metrics.csv", index=False)
    write_json(
        directories.root / "source_metadata.json",
        {
            "model_name": config.model_name,
            "input_mode": config.input_mode,
            "resolved_device": str(device),
            "amp_dtype": None if amp_dtype is None else str(amp_dtype).replace("torch.", ""),
            "metadata": MODEL_SOURCE_METADATA.get(config.model_name, {}),
        },
    )
    return metrics_df, aggregate_df