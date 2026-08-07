"""Causal multi-asset Temporal Fusion Transformer building blocks.

This module is intentionally model-only: it does not load a checkpoint or make a live trading
decision.  A caller must supply only features available at each decision bar and use the returned
quantiles as price-move forecasts, never account-PnL targets.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    """TFT-style gated residual transform with an optional context vector."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float, context_dim: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim * 2)
        self.skip = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc1(x)
        if self.context is not None:
            if context is None:
                raise ValueError("GRN requires context")
            h = h + self.context(context)
        h = self.fc2(self.dropout(F.elu(h)))
        value, gate = self.gate(h).chunk(2, dim=-1)
        return self.norm(self.skip(x) + value * torch.sigmoid(gate))


class VariableSelectionNetwork(nn.Module):
    """Select a sparse, time-varying mixture of scalar numeric features."""

    def __init__(self, n_features: int, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.weight_grn = GatedResidualNetwork(n_features, hidden_dim, n_features, dropout)
        self.dropout = nn.Dropout(dropout)
        # This is the per-variable GRN family from TFT, stored as feature-indexed tensors rather
        # than a Python ModuleList.  The old implementation launched one tiny GPU graph per
        # feature and made full-panel training impractically slow.
        self.fc1_weight = nn.Parameter(torch.empty(n_features, hidden_dim))
        self.fc1_bias = nn.Parameter(torch.zeros(n_features, hidden_dim))
        self.fc2_weight = nn.Parameter(torch.empty(n_features, hidden_dim, d_model))
        self.fc2_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.gate_weight = nn.Parameter(torch.empty(n_features, d_model, d_model * 2))
        self.gate_bias = nn.Parameter(torch.zeros(n_features, d_model * 2))
        self.skip_weight = nn.Parameter(torch.empty(n_features, d_model))
        self.skip_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.norm_weight = nn.Parameter(torch.ones(n_features, d_model))
        self.norm_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in (self.fc1_weight, self.fc2_weight, self.gate_weight, self.skip_weight):
            nn.init.xavier_uniform_(weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.n_features:
            raise ValueError(f"VSN expects [batch,time,{self.n_features}], got {tuple(x.shape)}")
        weights = torch.softmax(self.weight_grn(x), dim=-1)
        h = F.elu(x.unsqueeze(-1) * self.fc1_weight + self.fc1_bias)
        h = torch.einsum("btfh,fhd->btfd", self.dropout(h), self.fc2_weight) + self.fc2_bias
        value, gate = (
            torch.einsum("btfd,fdk->btfk", h, self.gate_weight) + self.gate_bias
        ).chunk(2, dim=-1)
        transformed = x.unsqueeze(-1) * self.skip_weight + self.skip_bias + value * torch.sigmoid(gate)
        mean = transformed.mean(dim=-1, keepdim=True)
        variance = transformed.var(dim=-1, unbiased=False, keepdim=True)
        transformed = (transformed - mean) / torch.sqrt(variance + 1e-5)
        transformed = transformed * self.norm_weight + self.norm_bias
        return (transformed * weights.unsqueeze(-1)).sum(dim=-2), weights


@dataclass(frozen=True)
class TFTForecast:
    quantiles: torch.Tensor
    entry_logits: torch.Tensor
    regime_logits: torch.Tensor
    exit_logits: torch.Tensor
    asset_variable_weights: torch.Tensor
    global_variable_weights: torch.Tensor | None
    target_asset_attention: torch.Tensor


class MultiAssetTFT(nn.Module):
    """TFT for a target asset plus a contemporaneous multi-asset market panel.

    ``asset_history`` is ``[batch, time, asset, feature]`` and must end at the decision bar.
    ``asset_ids`` is ``[batch, asset]``; ``target_asset_index`` identifies BTC (or another asset)
    inside that panel.  The output quantiles are ordered log price returns at ``horizon_bars``.
    """

    def __init__(
        self, *, n_asset_features: int, n_assets: int, quantile_count: int,
        d_model: int = 64, n_heads: int = 4, dropout: float = 0.1,
        n_global_features: int = 0,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_asset_features = int(n_asset_features)
        self.n_global_features = int(n_global_features)
        self.asset_vsn = VariableSelectionNetwork(n_asset_features, d_model, d_model * 2, dropout)
        self.global_vsn = (
            VariableSelectionNetwork(n_global_features, d_model, d_model * 2, dropout)
            if n_global_features else None
        )
        self.asset_embedding = nn.Embedding(n_assets, d_model)
        self.temporal_lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.temporal_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_asset_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.temporal_grn = GatedResidualNetwork(d_model, d_model * 2, d_model, dropout)
        self.fusion_grn = GatedResidualNetwork(d_model * 2, d_model * 2, d_model, dropout)
        self.quantile_base = nn.Linear(d_model, 1)
        self.quantile_steps = nn.Linear(d_model, quantile_count - 1)
        # All labels are supervised targets only; none may be accepted as model inputs.
        self.entry_head = nn.Linear(d_model, 3)  # cash / long / short triple-barrier entry
        self.regime_head = nn.Linear(d_model, 3)  # cash / long-wave / short-wave zigzag regime
        self.exit_head = nn.Linear(d_model, 3)  # hold / exit-to-cash / flip-direction within K bars

    def forward(
        self, asset_history: torch.Tensor, asset_ids: torch.Tensor, target_asset_index: torch.Tensor,
        global_history: torch.Tensor | None = None,
    ) -> TFTForecast:
        if asset_history.ndim != 4:
            raise ValueError("asset_history must be [batch,time,asset,feature]")
        batch, steps, assets, features = asset_history.shape
        if features != self.n_asset_features or asset_ids.shape != (batch, assets):
            raise ValueError("asset feature or asset_ids contract mismatch")
        if target_asset_index.shape != (batch,) or (target_asset_index < 0).any() or (target_asset_index >= assets).any():
            raise ValueError("target_asset_index must select one panel asset per batch row")
        if (global_history is None) != (self.global_vsn is None):
            raise ValueError("global_history must be supplied exactly when n_global_features is nonzero")
        if global_history is not None and global_history.shape != (batch, steps, self.n_global_features):
            raise ValueError("global_history contract mismatch")

        flat = asset_history.permute(0, 2, 1, 3).reshape(batch * assets, steps, features)
        selected, asset_weights = self.asset_vsn(flat)
        embedded_ids = self.asset_embedding(asset_ids).reshape(batch * assets, 1, -1)
        lstm, _ = self.temporal_lstm(selected + embedded_ids)
        causal_mask = torch.triu(torch.ones(steps, steps, device=lstm.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.temporal_attention(lstm, lstm, lstm, attn_mask=causal_mask, need_weights=False)
        temporal = self.temporal_grn(attended[:, -1] + lstm[:, -1]).reshape(batch, assets, -1)
        cross, cross_weights = self.cross_asset_attention(
            temporal, temporal, temporal, need_weights=True, average_attn_weights=False,
        )
        rows = torch.arange(batch, device=asset_history.device)
        target_temporal = temporal[rows, target_asset_index]
        target_cross = cross[rows, target_asset_index]
        fused = self.fusion_grn(torch.cat([target_temporal, target_cross], dim=-1))
        global_weights = None
        if self.global_vsn is not None:
            global_selected, global_weights = self.global_vsn(global_history)
            fused = self.fusion_grn(torch.cat([fused, global_selected[:, -1]], dim=-1))
        base = self.quantile_base(fused)
        increments = F.softplus(self.quantile_steps(fused))
        quantiles = torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)
        target_attention = cross_weights[rows, :, target_asset_index].mean(dim=1)
        # MultiheadAttention applies dropout to attention probabilities during training.  The
        # returned diagnostic must remain a probability distribution regardless of mode.
        target_attention = target_attention / target_attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return TFTForecast(
            quantiles=quantiles,
            entry_logits=self.entry_head(fused),
            regime_logits=self.regime_head(fused),
            exit_logits=self.exit_head(fused),
            asset_variable_weights=asset_weights.reshape(batch, assets, steps, features).permute(0, 2, 1, 3),
            global_variable_weights=global_weights,
            target_asset_attention=target_attention,
        )
