from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - depends on optional dependency
    torch = None
    nn = None
    F = None


def torch_is_available() -> bool:
    return torch is not None


MODEL_INPUT_MODES = {
    "eegnet": "raw",
    "eegwavenet": "raw",
    "cnn_bilstm": "raw",
    "bendr": "raw",
    "reve": "raw",
    "biseizurere": "raw",
    "inresformer": "channel",
    "gat_bilstm": "channel",
    "ce_tss_transformer": "channel",
    "dbconformer": "channel",
    "graphs4mer": "channel",
    "dcrnn": "channel",
}

MODEL_SOURCE_METADATA = {
    "eegnet": {
        "reference": "Lawhern et al. 2018 / arl-eegmodels",
        "source_type": "public_paper_and_repo",
        "fidelity": "paper_architecture_adapter",
        "notes": "Implements depthwise-separable EEGNet classification structure for raw-window fine-tuning.",
    },
    "eegwavenet": {
        "reference": "IoBT-VISTEC EEGWaveNet",
        "source_type": "public_repo",
        "fidelity": "repo_inspired_adapter",
        "notes": "Implements a multi-scale temporal CNN variant aligned with the published EEGWaveNet idea.",
    },
    "cnn_bilstm": {
        "reference": "Mallick and Baths 2024",
        "source_type": "public_paper",
        "fidelity": "paper_architecture_adapter",
        "notes": "Implements Conv1D plus BiLSTM classification stack for seizure windows.",
    },
    "bendr": {
        "reference": "BENDR / SPOClab-ca/BENDR",
        "source_type": "public_paper_and_repo",
        "fidelity": "classification_adapter",
        "notes": "Implements encoder plus transformer contextualizer; pretrained checkpoint loading is not bundled here.",
    },
    "reve": {
        "reference": "REVE foundation model",
        "source_type": "public_paper_repo_weights",
        "fidelity": "classification_adapter",
        "notes": "Implements a transformer-style EEG token encoder; upstream pretrained backbone integration is not bundled here.",
    },
    "biseizurere": {
        "reference": "No public canonical repo/paper available in model sources",
        "source_type": "no_public_source",
        "fidelity": "surrogate_adapter",
        "notes": "Exposed as a surrogate raw-window classifier because no public implementation was available to mirror exactly.",
    },
    "inresformer": {
        "reference": "Hu et al. 2024 InResformer",
        "source_type": "public_paper",
        "fidelity": "paper_architecture_adapter",
        "notes": "Implements inception-style multi-kernel front-end with transformer encoder for channel tokens.",
    },
    "gat_bilstm": {
        "reference": "Spatial-Temporal GAT plus BiLSTM seizure models",
        "source_type": "public_paper",
        "fidelity": "paper_architecture_adapter",
        "notes": "Implements attention-based spatial mixing followed by BiLSTM sequence modeling.",
    },
    "ce_tss_transformer": {
        "reference": "Li et al. MICCAI 2024 CE-TSS-Transformer",
        "source_type": "public_paper_and_repo",
        "fidelity": "paper_architecture_adapter",
        "notes": "Implements channel, temporal, and spectral transformer branches for feature tensors.",
    },
    "dbconformer": {
        "reference": "DBConformer 2025",
        "source_type": "public_paper_and_repo",
        "fidelity": "paper_architecture_adapter",
        "notes": "Implements dual-branch conformer-style temporal and spatial encoding.",
    },
    "graphs4mer": {
        "reference": "GraphS4mer",
        "source_type": "public_paper_and_repo",
        "fidelity": "paper_architecture_adapter",
        "notes": "Implements graph-aware attention plus state-space-inspired temporal mixing for channel tokens.",
    },
    "dcrnn": {
        "reference": "DCRNN",
        "source_type": "public_paper_and_repo",
        "fidelity": "paper_architecture_adapter",
        "notes": "Implements a recurrent graph-mixing classifier adapted to seizure-window classification.",
    },
}


class _TorchUnavailableModule:
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - depends on optional dependency
        raise ImportError("PyTorch is required to instantiate deep-learning models.")


if torch is None:
    class EEGNet(_TorchUnavailableModule):
        pass

    class EEGWaveNet(_TorchUnavailableModule):
        pass

    class CNNBiLSTM(_TorchUnavailableModule):
        pass

    class BENDRClassifier(_TorchUnavailableModule):
        pass

    class REVEClassifier(_TorchUnavailableModule):
        pass

    class BISeizureReAdapter(_TorchUnavailableModule):
        pass

    class InResformer(_TorchUnavailableModule):
        pass

    class GATBiLSTM(_TorchUnavailableModule):
        pass

    class CETSSTransformer(_TorchUnavailableModule):
        pass

    class DBConformer(_TorchUnavailableModule):
        pass

    class GraphS4mer(_TorchUnavailableModule):
        pass

    class DCRNNClassifier(_TorchUnavailableModule):
        pass
else:
    def _masked_mean(x, mask):
        weights = mask.float().unsqueeze(-1)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (x * weights).sum(dim=1) / denom


    def _apply_channel_mask(x, mask):
        return x * mask.unsqueeze(-1).float()


    class LiteConformerBlock(nn.Module):
        def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(d_model)
            self.conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=max(1, d_model // 4)),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.Dropout(dropout),
            )
            self.norm3 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
            )

        def forward(self, x, mask):
            key_padding_mask = ~mask if mask is not None else None
            attn_input = self.norm1(x)
            attn_out, _ = self.attn(attn_input, attn_input, attn_input, key_padding_mask=key_padding_mask)
            x = x + attn_out
            conv_out = self.conv(self.norm2(x).transpose(1, 2)).transpose(1, 2)
            x = x + conv_out
            x = x + self.ff(self.norm3(x))
            return x


    class EEGNet(nn.Module):
        def __init__(self, max_channels: int, n_samples: int, dropout: float = 0.25, f1: int = 8, d: int = 2, f2: int = 16) -> None:
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, f1, kernel_size=(1, 64), padding=(0, 32), bias=False),
                nn.BatchNorm2d(f1),
                nn.Conv2d(f1, f1 * d, kernel_size=(max_channels, 1), groups=f1, bias=False),
                nn.BatchNorm2d(f1 * d),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(dropout),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, 16), padding=(0, 8), groups=f1 * d, bias=False),
                nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(f2),
                nn.ELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Linear(f2, 1)

        def forward(self, *, x_raw, mask, **kwargs):
            x_raw = _apply_channel_mask(x_raw, mask)
            x = x_raw.unsqueeze(1)
            x = self.block1(x)
            x = self.block2(x)
            return self.classifier(x.flatten(1)).squeeze(-1)


    class EEGWaveNet(nn.Module):
        def __init__(self, max_channels: int, n_samples: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
            super().__init__()
            kernels = (3, 7, 15)
            self.branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(max_channels, hidden_dim, kernel_size=kernel, padding=kernel // 2),
                        nn.BatchNorm1d(hidden_dim),
                        nn.GELU(),
                    )
                    for kernel in kernels
                ]
            )
            merged_dim = hidden_dim * len(kernels)
            self.dilated = nn.Sequential(
                nn.Conv1d(merged_dim, merged_dim, kernel_size=3, padding=2, dilation=2),
                nn.GELU(),
                nn.Conv1d(merged_dim, merged_dim, kernel_size=3, padding=4, dilation=4),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(merged_dim, 1)

        def forward(self, *, x_raw, mask, **kwargs):
            x_raw = _apply_channel_mask(x_raw, mask)
            x = torch.cat([branch(x_raw) for branch in self.branches], dim=1)
            x = self.dilated(x).squeeze(-1)
            return self.classifier(self.dropout(x)).squeeze(-1)


    class CNNBiLSTM(nn.Module):
        def __init__(self, max_channels: int, n_samples: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(max_channels, hidden_dim, kernel_size=7, padding=3),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
            )
            self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
            self.attn = nn.Linear(hidden_dim, 1)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, *, x_raw, mask, **kwargs):
            x_raw = _apply_channel_mask(x_raw, mask)
            conv_out = self.conv(x_raw).transpose(1, 2)
            lstm_out, _ = self.lstm(conv_out)
            attn = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
            pooled = torch.sum(lstm_out * attn.unsqueeze(-1), dim=1)
            return self.classifier(self.dropout(pooled)).squeeze(-1)


    class BENDRClassifier(nn.Module):
        def __init__(self, max_channels: int, n_samples: int, encoder_dim: int = 128, n_layers: int = 4) -> None:
            super().__init__()
            self.channel_mix = nn.Conv1d(max_channels, encoder_dim, kernel_size=1)
            blocks = []
            current_dim = encoder_dim
            for _ in range(3):
                blocks.extend(
                    [
                        nn.Conv1d(current_dim, current_dim, kernel_size=7, stride=2, padding=3),
                        nn.GELU(),
                        nn.BatchNorm1d(current_dim),
                    ]
                )
            self.encoder = nn.Sequential(*blocks)
            encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_dim, nhead=4, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.classifier = nn.Linear(encoder_dim, 1)

        def forward(self, *, x_raw, mask, **kwargs):
            x_raw = _apply_channel_mask(x_raw, mask)
            x = self.encoder(self.channel_mix(x_raw)).transpose(1, 2)
            x = self.transformer(x)
            return self.classifier(x.mean(dim=1)).squeeze(-1)


    class REVEClassifier(nn.Module):
        def __init__(self, max_channels: int, n_samples: int, d_model: int = 128, patch_size: int = 16, n_layers: int = 4) -> None:
            super().__init__()
            self.patch_embed = nn.Conv1d(max_channels, d_model, kernel_size=patch_size, stride=patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.classifier = nn.Linear(d_model, 1)

        def forward(self, *, x_raw, mask, **kwargs):
            x_raw = _apply_channel_mask(x_raw, mask)
            tokens = self.patch_embed(x_raw).transpose(1, 2)
            cls_token = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)
            tokens = self.transformer(tokens)
            return self.classifier(tokens[:, 0]).squeeze(-1)


    class BISeizureReAdapter(REVEClassifier):
        """Fallback proxy because no public canonical implementation is available."""

        pass


    class InResformer(nn.Module):
        def __init__(self, max_channels: int, n_features: int, d_model: int = 96, n_layers: int = 2, dropout: float = 0.2) -> None:
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.branches = nn.ModuleList(
                [
                    nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=kernel // 2)
                    for kernel in (1, 3, 5)
                ]
            )
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True, dropout=dropout)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(d_model, 1)

        def forward(self, *, x_channel, mask, **kwargs):
            x = self.input_proj(_apply_channel_mask(x_channel, mask))
            conv_in = x.transpose(1, 2)
            x = x + sum(branch(conv_in).transpose(1, 2) for branch in self.branches) / len(self.branches)
            x = self.transformer(x, src_key_padding_mask=~mask)
            pooled = _masked_mean(x, mask)
            return self.classifier(self.dropout(pooled)).squeeze(-1)


    class GATBiLSTM(nn.Module):
        def __init__(self, max_channels: int, n_features: int, d_model: int = 96, dropout: float = 0.2) -> None:
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
            self.lstm = nn.LSTM(d_model, d_model // 2, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(d_model, 1)

        def forward(self, *, x_channel, mask, **kwargs):
            x = self.input_proj(_apply_channel_mask(x_channel, mask))
            x, _ = self.attn(x, x, x, key_padding_mask=~mask)
            x, _ = self.lstm(x)
            pooled = _masked_mean(x, mask)
            return self.classifier(self.dropout(pooled)).squeeze(-1)


    class CETSSTransformer(nn.Module):
        def __init__(self, max_channels: int, n_features: int, d_model: int = 96, n_layers: int = 2) -> None:
            super().__init__()
            self.channel_proj = nn.Linear(n_features, d_model)
            self.feature_proj = nn.Linear(1, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
            self.channel_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.spectral_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.classifier = nn.Linear(d_model * 3, 1)

        def forward(self, *, x_channel, mask, **kwargs):
            masked = _apply_channel_mask(x_channel, mask)
            channel_tokens = self.channel_proj(masked)
            channel_tokens = self.channel_encoder(channel_tokens, src_key_padding_mask=~mask)
            pooled_channel = _masked_mean(channel_tokens, mask)

            feature_tokens = self.feature_proj(masked.mean(dim=1, keepdim=False).unsqueeze(-1))
            temporal_tokens = self.temporal_encoder(feature_tokens)
            pooled_temporal = temporal_tokens.mean(dim=1)

            spectral_slice = masked[:, :, 8:].mean(dim=1).unsqueeze(-1)
            spectral_tokens = self.feature_proj(spectral_slice)
            spectral_tokens = self.spectral_encoder(spectral_tokens)
            pooled_spectral = spectral_tokens.mean(dim=1)

            fused = torch.cat([pooled_channel, pooled_temporal, pooled_spectral], dim=-1)
            return self.classifier(fused).squeeze(-1)


    class DBConformer(nn.Module):
        def __init__(self, max_channels: int, n_features: int, d_model: int = 96, dropout: float = 0.2) -> None:
            super().__init__()
            self.channel_proj = nn.Linear(n_features, d_model)
            self.feature_proj = nn.Linear(1, d_model)
            self.spatial_block = LiteConformerBlock(d_model=d_model, dropout=dropout)
            self.temporal_block = LiteConformerBlock(d_model=d_model, dropout=dropout)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(d_model * 2, 1)

        def forward(self, *, x_channel, mask, **kwargs):
            masked = _apply_channel_mask(x_channel, mask)
            spatial = self.channel_proj(masked)
            spatial = self.spatial_block(spatial, mask)
            pooled_spatial = _masked_mean(spatial, mask)

            temporal_tokens = self.feature_proj(masked.mean(dim=1, keepdim=False).unsqueeze(-1))
            temporal_mask = torch.ones(temporal_tokens.shape[:2], dtype=torch.bool, device=temporal_tokens.device)
            temporal = self.temporal_block(temporal_tokens, temporal_mask)
            pooled_temporal = temporal.mean(dim=1)

            fused = torch.cat([pooled_spatial, pooled_temporal], dim=-1)
            return self.classifier(self.dropout(fused)).squeeze(-1)


    class GraphS4mer(nn.Module):
        def __init__(self, max_channels: int, n_features: int, d_model: int = 96, dropout: float = 0.2) -> None:
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
            self.ssm = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=max(1, d_model // 4)),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1),
            )
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True, dropout=dropout)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.classifier = nn.Linear(d_model, 1)

        def forward(self, *, x_channel, mask, **kwargs):
            x = self.input_proj(_apply_channel_mask(x_channel, mask))
            x, _ = self.attn(x, x, x, key_padding_mask=~mask)
            x = x + self.ssm(x.transpose(1, 2)).transpose(1, 2)
            x = self.transformer(x, src_key_padding_mask=~mask)
            return self.classifier(_masked_mean(x, mask)).squeeze(-1)


    class DCRNNClassifier(nn.Module):
        def __init__(self, max_channels: int, n_features: int, d_model: int = 64) -> None:
            super().__init__()
            self.input_proj = nn.Linear(1, d_model)
            self.hidden_proj = nn.Linear(d_model * 2, d_model)
            self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
            self.classifier = nn.Linear(d_model, 1)

        def forward(self, *, x_channel, mask, **kwargs):
            hidden = torch.zeros(x_channel.size(0), x_channel.size(1), self.hidden_proj.out_features, device=x_channel.device)
            for feature_idx in range(x_channel.size(-1)):
                x_t = self.input_proj(x_channel[:, :, feature_idx : feature_idx + 1])
                graph_context, _ = self.attn(hidden, hidden, hidden, key_padding_mask=~mask)
                hidden = torch.tanh(self.hidden_proj(torch.cat([x_t, graph_context], dim=-1)))
                hidden = _apply_channel_mask(hidden, mask)
            return self.classifier(_masked_mean(hidden, mask)).squeeze(-1)


def build_dl_model(
    model_name: str,
    *,
    input_mode: str,
    max_channels: int,
    n_features: int | None = None,
    n_samples: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
):
    model_kwargs = model_kwargs or {}
    expected_mode = MODEL_INPUT_MODES.get(model_name)
    if expected_mode is None:
        raise ValueError(f"Unsupported DL model: {model_name}")
    if input_mode != expected_mode:
        raise ValueError(f"Model {model_name} expects input_mode='{expected_mode}', got '{input_mode}'.")

    if model_name == "eegnet":
        return EEGNet(max_channels=max_channels, n_samples=int(n_samples or 512), **model_kwargs)
    if model_name == "eegwavenet":
        return EEGWaveNet(max_channels=max_channels, n_samples=int(n_samples or 512), **model_kwargs)
    if model_name == "cnn_bilstm":
        return CNNBiLSTM(max_channels=max_channels, n_samples=int(n_samples or 512), **model_kwargs)
    if model_name == "bendr":
        return BENDRClassifier(max_channels=max_channels, n_samples=int(n_samples or 512), **model_kwargs)
    if model_name == "reve":
        return REVEClassifier(max_channels=max_channels, n_samples=int(n_samples or 512), **model_kwargs)
    if model_name == "biseizurere":
        return BISeizureReAdapter(max_channels=max_channels, n_samples=int(n_samples or 512), **model_kwargs)
    if model_name == "inresformer":
        return InResformer(max_channels=max_channels, n_features=int(n_features or 16), **model_kwargs)
    if model_name == "gat_bilstm":
        return GATBiLSTM(max_channels=max_channels, n_features=int(n_features or 16), **model_kwargs)
    if model_name == "ce_tss_transformer":
        return CETSSTransformer(max_channels=max_channels, n_features=int(n_features or 16), **model_kwargs)
    if model_name == "dbconformer":
        return DBConformer(max_channels=max_channels, n_features=int(n_features or 16), **model_kwargs)
    if model_name == "graphs4mer":
        return GraphS4mer(max_channels=max_channels, n_features=int(n_features or 16), **model_kwargs)
    if model_name == "dcrnn":
        return DCRNNClassifier(max_channels=max_channels, n_features=int(n_features or 16), **model_kwargs)
    raise ValueError(f"Unsupported DL model: {model_name}")