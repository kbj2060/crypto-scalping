"""Train a duration-free inventory-aware Mixture-of-Experts ETH scalp policy.

The policy emits SHORT/CASH/LONG every completed minute. It has no fixed or
maximum holding period, TP/SL, or cooldown. A train-only dynamic-programming
teacher supplies transaction-cost-aware action advantages for every possible
current position. A causal neural Mixture-of-Experts distills those advantages
from price and microstructure windows, then a tune-only action-gap margin controls
switching. Locked validation/development rows are diagnostic only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_eval_eth_micro_scalp_dynamic_20260718 import (  # noqa: E402
    CACHE_DIR,
    _json_default,
    _require_names,
    load_frozen_cache,
    purged_interval_mask,
    replay_positions,
)


MODEL_ID = "eth_micro_scalp_inventory_moe_v1_20260718"
ARTIFACT_DIR = ROOT / f"data/ensemble/{MODEL_ID}"
MODEL_PATH = ARTIFACT_DIR / "model.pt"
REPORT_PATH = ROOT / f"data/ensemble/reports/{MODEL_ID}.json"
VALIDATION_LEDGER_PATH = ARTIFACT_DIR / "validation_diagnostic_ledger.csv"
DEVELOPMENT_LEDGER_PATH = ARTIFACT_DIR / "development_diagnostic_ledger.csv"

ACTIONS = np.asarray((-1, 0, 1), dtype=np.int8)

BASE_FEATURES = (
    "bar_open_close_logret", "bar_range_pct", "log_volume", "log_quote_volume",
    "log_trade_count", "bar_taker_buy_ratio", "whale_retail_ratio", "whale_conviction",
    "smart_money_flow", "squeeze_power", "oi_change_rate", "net_taker_ratio",
    "taker_acceleration", "trade_intensity", "big_trade_ratio", "log_return",
    "volatility_z", "rsi", "macd_hist", "bb_width_z", "hma_slope", "wick_ratio",
    "garman_klass_vol", "realized_vol_ratio", "amihud_illiquidity_z", "chop_index",
    "cvp_volume_imbalance", "mean_reversion_z", "breakout_strength", "funding_z_score",
    "long_squeeze_risk", "short_squeeze_risk", "ofi_acceleration", "kalman_velocity",
    "realized_skewness", "cvd_slope_12", "cvd_slope_48", "compression_score",
    "vwap_dist_24", "upper_wick_z", "lower_wick_z", "liquidity_vacuum", "execution_quality",
)

MICRO_FEATURES = (
    "micro_obi", "micro_taker_buy_ratio", "micro_nif_whale", "micro_nif_retail",
    "micro_oi_delta_pct", "micro_funding_rate", "micro_recent_trade_count_5m",
    "micro_recent_trade_notional_5m", "micro_recent_whale_count_5m", "micro_data_stale",
    "micro_depth_connected", "micro_trade_connected", "micro_poll_connected",
    "micro_depth_age_sec", "micro_trade_age_sec", "micro_poll_age_sec",
    "micro_valid_taker_flow", "micro_valid_nif", "micro_warmup_30m_ready",
    "micro_available", "micro_age_min", "book_spread_bps", "book_available", "book_age_min",
)

HEALTH_FEATURES = (
    "micro_available", "micro_data_stale", "micro_depth_connected",
    "micro_warmup_30m_ready", "micro_age_min",
)


@dataclass(frozen=True)
class Config:
    seed: int = 18
    window: int = 60
    forecast_horizon_min: int = 5
    fee_per_notional_change: float = 0.00045
    teacher_gamma: float = 0.9995
    teacher_inventory_vol_weight: float = 0.01
    teacher_advantage_clip_bp: float = 50.0
    fit_start: str = "2026-05-03 00:00:00"
    tune_start: str = "2026-06-11 00:00:00"
    validation_start: str = "2026-06-21 00:00:00"
    development_start: str = "2026-07-01 00:00:00"
    development_end: str = "2026-07-12 09:01:00"
    base_channels: int = 48
    micro_channels: int = 32
    latent_dim: int = 64
    experts: int = 3
    dropout: float = 0.10
    batch_size: int = 512
    epochs: int = 8
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    q_loss_weight: float = 1.0
    expert_q_loss_weight: float = 0.10
    action_loss_weight: float = 0.5
    auxiliary_loss_weight: float = 0.15
    gate_balance_weight: float = 0.01
    min_tune_switches: int = 30


@dataclass(frozen=True)
class QPolicy:
    enabled: bool
    switch_margin_bp: float
    min_expert_agreement: int = 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_robust_scaler(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample = np.asarray(values[mask], dtype=np.float64)
    center = np.nanmedian(sample, axis=0)
    mad = np.nanmedian(np.abs(sample - center), axis=0) * 1.4826
    std = np.nanstd(sample, axis=0)
    scale = np.where((mad > 1e-8) & np.isfinite(mad), mad, std)
    center = np.where(np.isfinite(center), center, 0.0)
    scale = np.where((scale > 1e-8) & np.isfinite(scale), scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def apply_scaler(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scaled = (np.asarray(values, dtype=np.float32) - center) / scale
    return np.clip(np.nan_to_num(scaled, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0).astype(np.float32)


def build_sequence_frame(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any], config: Config
) -> dict[str, Any]:
    timestamp_all = pd.to_datetime(np.asarray(arrays["timestamp_ns"], dtype=np.int64))
    buffer_start = pd.Timestamp(config.fit_start) - pd.Timedelta(minutes=config.window)
    span = (timestamp_all >= buffer_start) & (timestamp_all < pd.Timestamp(config.development_end))
    indices = np.flatnonzero(span)
    base_names = list(metadata["base_feature_names"])
    micro_names = list(metadata["micro_feature_names"])
    base_idx = _require_names(base_names, BASE_FEATURES, "inventory-moe base")
    micro_idx = _require_names(micro_names, MICRO_FEATURES, "inventory-moe micro")
    health_idx = _require_names(micro_names, HEALTH_FEATURES, "inventory-moe health")

    base = np.asarray(arrays["base"][indices][:, base_idx], dtype=np.float32)
    micro_all = np.asarray(arrays["micro"][indices], dtype=np.float32)
    micro = micro_all[:, micro_idx]
    health = {name: micro_all[:, idx] for name, idx in zip(HEALTH_FEATURES, health_idx)}
    available = (
        np.isfinite(health["micro_available"]) & (health["micro_available"] > 0.5)
        & np.isfinite(health["micro_data_stale"]) & (health["micro_data_stale"] < 0.5)
        & np.isfinite(health["micro_depth_connected"]) & (health["micro_depth_connected"] > 0.5)
        & np.isfinite(health["micro_warmup_30m_ready"]) & (health["micro_warmup_30m_ready"] > 0.5)
        & np.isfinite(health["micro_age_min"]) & (health["micro_age_min"] >= 0.0)
        & (health["micro_age_min"] <= 2.0)
    )
    target = np.asarray(arrays["targets"][indices], dtype=np.float32)
    next_return = np.asarray(arrays["next_return"][indices], dtype=np.float64)
    timestamps = timestamp_all[indices]
    return {
        "base_raw": base,
        "micro_raw": micro,
        "target_raw": target,
        "next_return": next_return,
        "timestamps": timestamps,
        "available": available,
        "base_names": list(BASE_FEATURES),
        "micro_names": list(MICRO_FEATURES),
    }


def build_cost_aware_teacher(
    next_return: np.ndarray,
    available: np.ndarray,
    volatility: np.ndarray,
    fee_per_notional_change: float,
    gamma: float,
    inventory_vol_weight: float,
    advantage_clip_bp: float,
) -> tuple[np.ndarray, np.ndarray]:
    returns = np.nan_to_num(np.asarray(next_return, dtype=np.float64), nan=0.0)
    usable = np.asarray(available, dtype=bool)
    vol_bp = np.clip(np.nan_to_num(np.asarray(volatility, dtype=np.float64), nan=0.0) * 10_000.0, 0.0, 50.0)
    inventory_penalty = inventory_vol_weight * vol_bp / 10_000.0
    n = len(returns)
    q_values = np.full((n, 3, 3), -1e6, dtype=np.float64)
    terminal = -fee_per_notional_change * np.abs(ACTIONS.astype(np.float64))
    value_next = terminal.copy()
    for idx in range(n - 1, -1, -1):
        value_here = np.empty(3, dtype=np.float64)
        for previous_idx, previous in enumerate(ACTIONS):
            allowed = (1,) if not usable[idx] else (0, 1, 2)
            for action_idx in allowed:
                action = ACTIONS[action_idx]
                reward = (
                    float(action) * returns[idx]
                    - fee_per_notional_change * abs(float(action - previous))
                    - inventory_penalty[idx] * abs(float(action))
                )
                q_values[idx, previous_idx, action_idx] = reward + gamma * value_next[action_idx]
            value_here[previous_idx] = np.max(q_values[idx, previous_idx])
        value_next = value_here
    best = np.max(q_values, axis=2, keepdims=True)
    advantage_bp = np.clip((q_values - best) * 10_000.0, -advantage_clip_bp, 0.0).astype(np.float32)
    action = np.argmax(q_values, axis=2).astype(np.int64)
    return advantage_bp, action


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(F.pad(x, (self.padding, 0)))
        return x + self.dropout(F.gelu(self.norm(y)))


class CausalBranchEncoder(nn.Module):
    def __init__(self, n_features: int, channels: int, dilations: tuple[int, ...], dropout: float):
        super().__init__()
        self.projection = nn.Conv1d(n_features, channels, kernel_size=1)
        self.blocks = nn.Sequential(*(CausalResidualBlock(channels, dilation, dropout) for dilation in dilations))
        self.attention = nn.Sequential(nn.Linear(channels, channels // 2), nn.Tanh(), nn.Linear(channels // 2, 1))
        self.output = nn.Sequential(nn.Linear(channels * 2, channels), nn.LayerNorm(channels), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.blocks(self.projection(x.transpose(1, 2))).transpose(1, 2)
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return self.output(torch.cat([sequence[:, -1], pooled], dim=-1))


class InventoryMoEQPolicy(nn.Module):
    def __init__(self, n_base: int, n_micro: int, n_aux: int, config: Config):
        super().__init__()
        self.base_encoder = CausalBranchEncoder(n_base, config.base_channels, (1, 2, 4, 8), config.dropout)
        self.micro_encoder = CausalBranchEncoder(n_micro, config.micro_channels, (1, 2, 4), config.dropout)
        fused_dim = config.base_channels + config.micro_channels
        self.regime_gate = nn.Sequential(nn.Linear(fused_dim, 48), nn.GELU(), nn.Linear(48, config.experts))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fused_dim, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(config.dropout),
                    nn.Linear(96, config.latent_dim), nn.GELU(),
                )
                for _ in range(config.experts)
            ]
        )
        self.position_embedding = nn.Embedding(3, 12)
        self.q_head = nn.Sequential(
            nn.Linear(config.latent_dim + 12, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 3)
        )
        self.auxiliary_head = nn.Sequential(
            nn.Linear(config.latent_dim, 64), nn.GELU(), nn.Linear(64, n_aux)
        )

    def forward(
        self, base: torch.Tensor, micro: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = torch.cat([self.base_encoder(base), self.micro_encoder(micro)], dim=-1)
        gate = torch.softmax(self.regime_gate(fused), dim=-1)
        expert_values = torch.stack([expert(fused) for expert in self.experts], dim=1)
        latent = torch.sum(expert_values * gate.unsqueeze(-1), dim=1)
        position_ids = torch.arange(3, device=latent.device)
        position = self.position_embedding(position_ids).unsqueeze(0).expand(len(latent), -1, -1)
        expert_state = torch.cat(
            [
                expert_values.unsqueeze(2).expand(-1, -1, 3, -1),
                position.unsqueeze(1).expand(-1, len(self.experts), -1, -1),
            ],
            dim=-1,
        )
        expert_q = self.q_head(expert_state)
        q_values = torch.sum(expert_q * gate[:, :, None, None], dim=1)
        auxiliary = self.auxiliary_head(latent)
        return q_values, auxiliary, gate, expert_q


class PolicyDataset(Dataset):
    def __init__(
        self,
        base: np.ndarray,
        micro: np.ndarray,
        q_target: np.ndarray,
        action_target: np.ndarray,
        auxiliary: np.ndarray,
        end_indices: np.ndarray,
        window: int,
    ):
        self.base_windows = sliding_window_view(base, window_shape=window, axis=0)
        self.micro_windows = sliding_window_view(micro, window_shape=window, axis=0)
        self.q_target = q_target
        self.action_target = action_target
        self.auxiliary = auxiliary
        self.end_indices = np.asarray(end_indices, dtype=np.int64)
        self.window = window

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        end = int(self.end_indices[item])
        start = end - self.window + 1
        base = np.ascontiguousarray(self.base_windows[start].T)
        micro = np.ascontiguousarray(self.micro_windows[start].T)
        return (
            torch.from_numpy(base),
            torch.from_numpy(micro),
            torch.from_numpy(np.asarray(self.q_target[end], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.action_target[end], dtype=np.int64)),
            torch.from_numpy(np.asarray(self.auxiliary[end], dtype=np.float32)),
        )


def train_model(
    model: InventoryMoEQPolicy,
    base: np.ndarray,
    micro: np.ndarray,
    q_target: np.ndarray,
    action_target: np.ndarray,
    auxiliary: np.ndarray,
    train_indices: np.ndarray,
    config: Config,
    device: torch.device,
) -> list[dict[str, float]]:
    dataset = PolicyDataset(base, micro, q_target, action_target, auxiliary, train_indices, config.window)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda",
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(config.epochs):
        totals = {
            "loss": 0.0, "q": 0.0, "expert_q": 0.0, "action": 0.0,
            "aux": 0.0, "gate": 0.0, "batches": 0.0,
        }
        for xb, xm, yq, ya, yu in loader:
            xb, xm, yq, ya, yu = (tensor.to(device, non_blocking=True) for tensor in (xb, xm, yq, ya, yu))
            optimizer.zero_grad(set_to_none=True)
            predicted_q, predicted_aux, gate, expert_q = model(xb, xm)
            q_loss = F.smooth_l1_loss(predicted_q, yq)
            expert_q_loss = F.smooth_l1_loss(expert_q, yq[:, None].expand_as(expert_q))
            action_loss = F.cross_entropy(predicted_q.reshape(-1, 3), ya.reshape(-1))
            auxiliary_loss = F.smooth_l1_loss(predicted_aux, yu)
            mean_gate = gate.mean(dim=0)
            gate_balance = torch.sum(mean_gate * torch.log(mean_gate * config.experts + 1e-8))
            loss = (
                config.q_loss_weight * q_loss
                + config.expert_q_loss_weight * expert_q_loss
                + config.action_loss_weight * action_loss
                + config.auxiliary_loss_weight * auxiliary_loss
                + config.gate_balance_weight * gate_balance
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            for key, value in (
                ("loss", loss), ("q", q_loss), ("expert_q", expert_q_loss), ("action", action_loss),
                ("aux", auxiliary_loss), ("gate", gate_balance),
            ):
                totals[key] += float(value.detach())
            totals["batches"] += 1.0
        row = {
            key: totals[key] / max(totals["batches"], 1.0)
            for key in ("loss", "q", "expert_q", "action", "aux", "gate")
        }
        row["epoch"] = float(epoch + 1)
        history.append(row)
        print(
            f"epoch={epoch + 1} loss={row['loss']:.4f} q={row['q']:.4f} "
            f"expert_q={row['expert_q']:.4f} action={row['action']:.4f} aux={row['aux']:.4f}",
            flush=True,
        )
    return history


def valid_window_end_indices(mask: np.ndarray, timestamps: pd.DatetimeIndex, window: int) -> np.ndarray:
    minute = 60_000_000_000
    timestamp_ns = timestamps.astype("int64").to_numpy()
    continuous = np.ones(len(timestamps), dtype=np.int32)
    for idx in range(1, len(timestamps)):
        continuous[idx] = continuous[idx - 1] + 1 if timestamp_ns[idx] - timestamp_ns[idx - 1] == minute else 1
    return np.flatnonzero(np.asarray(mask, dtype=bool) & (continuous >= window))


@torch.no_grad()
def infer_q_tables(
    model: InventoryMoEQPolicy,
    base: np.ndarray,
    micro: np.ndarray,
    end_indices: np.ndarray,
    window: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_windows = sliding_window_view(base, window_shape=window, axis=0)
    micro_windows = sliding_window_view(micro, window_shape=window, axis=0)
    q_rows: list[np.ndarray] = []
    gate_rows: list[np.ndarray] = []
    expert_q_rows: list[np.ndarray] = []
    model.eval()
    for offset in range(0, len(end_indices), batch_size):
        batch_indices = end_indices[offset : offset + batch_size]
        xb = np.stack([base_windows[int(end) - window + 1].T for end in batch_indices]).astype(np.float32)
        xm = np.stack([micro_windows[int(end) - window + 1].T for end in batch_indices]).astype(np.float32)
        predicted_q, _, gate, expert_q = model(torch.from_numpy(xb).to(device), torch.from_numpy(xm).to(device))
        q_rows.append(predicted_q.cpu().numpy())
        gate_rows.append(gate.cpu().numpy())
        expert_q_rows.append(expert_q.cpu().numpy())
    return np.concatenate(q_rows), np.concatenate(gate_rows), np.concatenate(expert_q_rows)


def decide_q_positions(
    q_tables: np.ndarray,
    available: np.ndarray,
    policy: QPolicy,
    expert_q_tables: np.ndarray | None = None,
) -> np.ndarray:
    q_values = np.asarray(q_tables, dtype=np.float64)
    usable = np.asarray(available, dtype=bool)
    position = np.zeros(len(q_values), dtype=np.int8)
    if not policy.enabled:
        return position
    previous_idx = 1
    for idx in range(len(q_values)):
        if not usable[idx] or not np.isfinite(q_values[idx]).all():
            action_idx = 1
        else:
            state_q = q_values[idx, previous_idx]
            action_idx = int(np.argmax(state_q))
            improvement = float(state_q[action_idx] - state_q[previous_idx])
            if action_idx != previous_idx and improvement < policy.switch_margin_bp:
                action_idx = previous_idx
            if action_idx != previous_idx and policy.min_expert_agreement > 1:
                if expert_q_tables is None:
                    raise ValueError("expert Q tables are required for consensus switching")
                votes = np.argmax(expert_q_tables[idx, :, previous_idx], axis=1)
                agreement = int(np.sum(votes == action_idx))
                if agreement < policy.min_expert_agreement:
                    action_idx = previous_idx
        position[idx] = ACTIONS[action_idx]
        previous_idx = action_idx
    return position


def replay_q_policy(
    q_tables: np.ndarray,
    available: np.ndarray,
    next_return: np.ndarray,
    timestamps: pd.DatetimeIndex,
    policy: QPolicy,
    fee: float,
    expert_q_tables: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    position = decide_q_positions(q_tables, available, policy, expert_q_tables)
    metrics, ledger = replay_positions(position, next_return, timestamps, fee)
    ledger["available"] = np.asarray(available, dtype=bool)
    return metrics, ledger


def select_q_policy(
    fit_q: np.ndarray,
    tune_q: np.ndarray,
    tune_expert_q: np.ndarray,
    tune_available: np.ndarray,
    tune_returns: np.ndarray,
    tune_timestamps: pd.DatetimeIndex,
    config: Config,
) -> tuple[QPolicy, list[dict[str, Any]]]:
    gap = np.sort(fit_q, axis=2)[:, :, -1] - np.sort(fit_q, axis=2)[:, :, -2]
    finite_gap = gap[np.isfinite(gap)]
    quantiles = np.quantile(finite_gap, (0.25, 0.50, 0.75, 0.90, 0.95)) if len(finite_gap) else np.zeros(5)
    margins = sorted({round(float(value), 6) for value in np.r_[0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, quantiles] if value >= 0.0})
    candidates: list[dict[str, Any]] = []
    expert_count = int(tune_expert_q.shape[1])
    if expert_count <= 0:
        raise RuntimeError("consensus policy requires at least one expert")
    for agreement in range(1, expert_count + 1):
        for margin in margins:
            policy = QPolicy(True, margin, agreement)
            metrics, _ = replay_q_policy(
                tune_q, tune_available, tune_returns, tune_timestamps, policy,
                config.fee_per_notional_change, tune_expert_q,
            )
            net = metrics["compounded_return_pct"] / 100.0
            drawdown = metrics["max_drawdown_pct"] / 100.0
            eligible = metrics["entries_or_reversals"] >= config.min_tune_switches and net > 0.0
            score = net - 0.25 * drawdown if eligible else float("-inf")
            candidates.append(
                {"policy": asdict(policy), "eligible": bool(eligible), "selection_score": score, "metrics": metrics}
            )
    candidates.sort(key=lambda row: row["selection_score"], reverse=True)
    if not candidates or not np.isfinite(candidates[0]["selection_score"]) or candidates[0]["selection_score"] <= 0.0:
        return QPolicy(False, 0.0, expert_count), candidates
    return QPolicy(**candidates[0]["policy"]), candidates


def cost_stress(
    positions: np.ndarray, returns: np.ndarray, timestamps: pd.DatetimeIndex
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for fee in (0.00020, 0.000325, 0.00045, 0.00055, 0.00090):
        metrics, _ = replay_positions(positions, returns, timestamps, fee)
        result[f"{fee * 10_000:.2f}bp_per_notional_change"] = metrics
    return result


def run(config: Config) -> dict[str, Any]:
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays, metadata = load_frozen_cache()
    frame = build_sequence_frame(arrays, metadata, config)
    timestamps = frame["timestamps"]
    masks = {
        "fit": purged_interval_mask(timestamps, config.fit_start, config.tune_start, config.forecast_horizon_min),
        "tune": purged_interval_mask(timestamps, config.tune_start, config.validation_start, config.forecast_horizon_min),
        "validation": purged_interval_mask(
            timestamps, config.validation_start, config.development_start, config.forecast_horizon_min
        ),
        "development": purged_interval_mask(
            timestamps, config.development_start, config.development_end, config.forecast_horizon_min
        ),
    }
    fit_mask = masks["fit"]
    base_center, base_scale = fit_robust_scaler(frame["base_raw"], fit_mask)
    micro_center, micro_scale = fit_robust_scaler(frame["micro_raw"], fit_mask & frame["available"])
    aux_valid = fit_mask & np.isfinite(frame["target_raw"]).all(axis=1)
    aux_center, aux_scale = fit_robust_scaler(frame["target_raw"], aux_valid)
    base = apply_scaler(frame["base_raw"], base_center, base_scale)
    micro = apply_scaler(frame["micro_raw"], micro_center, micro_scale)
    auxiliary = apply_scaler(frame["target_raw"], aux_center, aux_scale)

    volatility_idx = frame["base_names"].index("garman_klass_vol")
    fit_indices = np.flatnonzero(fit_mask)
    teacher_q_local, teacher_action_local = build_cost_aware_teacher(
        frame["next_return"][fit_indices], frame["available"][fit_indices],
        frame["base_raw"][fit_indices, volatility_idx], config.fee_per_notional_change,
        config.teacher_gamma, config.teacher_inventory_vol_weight, config.teacher_advantage_clip_bp,
    )
    teacher_q = np.full((len(timestamps), 3, 3), np.nan, dtype=np.float32)
    teacher_action = np.zeros((len(timestamps), 3), dtype=np.int64)
    teacher_q[fit_indices] = teacher_q_local
    teacher_action[fit_indices] = teacher_action_local
    train_indices = valid_window_end_indices(fit_mask & np.isfinite(frame["target_raw"]).all(axis=1), timestamps, config.window)

    model = InventoryMoEQPolicy(base.shape[1], micro.shape[1], auxiliary.shape[1], config).to(device)
    history = train_model(
        model, base, micro, teacher_q, teacher_action, auxiliary, train_indices, config, device
    )

    split_indices = {
        name: valid_window_end_indices(mask, timestamps, config.window) for name, mask in masks.items()
    }
    inference: dict[str, dict[str, Any]] = {}
    for name, indices in split_indices.items():
        q_values, gates, expert_q = infer_q_tables(
            model, base, micro, indices, config.window, config.batch_size, device
        )
        inference[name] = {"indices": indices, "q": q_values, "gates": gates, "expert_q": expert_q}

    fit_data = inference["fit"]
    tune_data = inference["tune"]
    tune_idx = tune_data["indices"]
    policy, candidates = select_q_policy(
        fit_data["q"], tune_data["q"], tune_data["expert_q"], frame["available"][tune_idx],
        frame["next_return"][tune_idx], timestamps[tune_idx], config,
    )

    results: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    stresses: dict[str, Any] = {}
    for name in ("tune", "validation", "development"):
        data = inference[name]
        indices = data["indices"]
        metrics, ledger = replay_q_policy(
            data["q"], frame["available"][indices], frame["next_return"][indices],
            timestamps[indices], policy, config.fee_per_notional_change, data["expert_q"],
        )
        positions = ledger["position"].to_numpy(dtype=np.int8)
        results[name] = metrics
        ledgers[name] = ledger
        stresses[name] = cost_stress(positions, frame["next_return"][indices], timestamps[indices])

    active_and_positive = (
        policy.enabled
        and results["validation"]["compounded_return_pct"] > 0.0
        and results["development"]["compounded_return_pct"] > 0.0
    )
    execution_policy = policy if active_and_positive else QPolicy(False, 0.0, config.experts)
    if not policy.enabled:
        promotion_reason = "No active inventory-aware Q policy survived tune after modeled cost."
    elif not active_and_positive:
        promotion_reason = "The tune-selected Q policy failed locked validation/development; artifact execution is fail-safe CASH."
    else:
        promotion_reason = "Historical intervals are consumed development data; post-freeze fresh-forward evidence is still required."

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_id": MODEL_ID,
        "model_state": model.state_dict(),
        "config": asdict(config),
        "base_feature_names": frame["base_names"],
        "micro_feature_names": frame["micro_names"],
        "scalers": {
            "base_center": base_center, "base_scale": base_scale,
            "micro_center": micro_center, "micro_scale": micro_scale,
            "aux_center": aux_center, "aux_scale": aux_scale,
        },
        "policy": asdict(execution_policy),
        "selected_research_policy": asdict(policy),
        "activation_allowed": active_and_positive,
        "cache_contract_sha256": metadata["source_signature"]["contract_sha256"],
        "fixed_holding_period_used": False,
    }
    torch.save(checkpoint, MODEL_PATH)
    ledgers["validation"].to_csv(VALIDATION_LEDGER_PATH, index=False)
    ledgers["development"].to_csv(DEVELOPMENT_LEDGER_PATH, index=False)

    teacher_distribution = {
        str(int(ACTIONS[previous])): {
            str(int(ACTIONS[action])): int(np.sum(teacher_action_local[:, previous] == action))
            for action in range(3)
        }
        for previous in range(3)
    }
    gate_means = {
        name: inference[name]["gates"].mean(axis=0).tolist() for name in inference
    }
    report = {
        "model_id": MODEL_ID,
        "status": "research_shadow_candidate" if active_and_positive else "research_no_viable_active_policy",
        "model_family": "dual causal encoders + three-expert regime MoE + inventory-conditioned Q head",
        "device": str(device),
        "holding_contract": {
            "fixed_holding_period_used": False,
            "max_holding_period_used": False,
            "fixed_tp_sl_used": False,
            "cooldown_used": False,
            "decision_frequency": "every completed 1-minute bar",
            "exit_rule": "the inventory-conditioned action value selects CASH or the opposite side",
        },
        "teacher_contract": {
            "type": "train-only backward dynamic programming over SHORT/CASH/LONG inventory states",
            "future_path_used_as_training_target_only": True,
            "fee_per_notional_change": config.fee_per_notional_change,
            "gamma": config.teacher_gamma,
            "causal_volatility_inventory_penalty_weight": config.teacher_inventory_vol_weight,
            "action_distribution_by_previous_position": teacher_distribution,
        },
        "config": asdict(config),
        "feature_contract": {
            "base_features": frame["base_names"],
            "micro_features": frame["micro_names"],
            "btc_features_used": False,
            "rule_outputs_used": False,
            "trade_ledgers_used": False,
            "cache_contract_sha256": metadata["source_signature"]["contract_sha256"],
        },
        "data": {
            "cache_dir": str(CACHE_DIR),
            "splits": {
                "fit": [config.fit_start, config.tune_start],
                "tune": [config.tune_start, config.validation_start],
                "validation": [config.validation_start, config.development_start],
                "development": [config.development_start, config.development_end],
            },
            "purge_minutes": config.forecast_horizon_min,
            "window_minutes": config.window,
            "split_window_counts": {name: int(len(indices)) for name, indices in split_indices.items()},
        },
        "training_history": history,
        "regime_gate_mean_weights": gate_means,
        "selected_research_policy": asdict(policy),
        "artifact_execution_policy": asdict(execution_policy),
        "activation_allowed": active_and_positive,
        "tune": results["tune"],
        "tune_cost_stress": stresses["tune"],
        "validation": results["validation"],
        "validation_cost_stress": stresses["validation"],
        "development": results["development"],
        "development_cost_stress": stresses["development"],
        "top_tune_candidates": candidates[:10],
        "artifacts": {
            "model": str(MODEL_PATH),
            "validation_diagnostic_ledger": str(VALIDATION_LEDGER_PATH),
            "development_diagnostic_ledger": str(DEVELOPMENT_LEDGER_PATH),
        },
        "integrity": {
            "script_sha256": _sha256(Path(__file__)),
            "metadata_sha256": _sha256(CACHE_DIR / "metadata.json"),
        },
        "compliance": {
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "future_path_used_only_for_fit_teacher_target": True,
            "fixed_holding_period_used": False,
            "outer_results_used_for_policy_selection": False,
        },
        "promotion": {
            "promotion_pass": False,
            "live_candidate": False,
            "reason": promotion_reason,
            "next_untouched_start": "after this 2026-07-18 model freeze",
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=_json_default))
    print(json.dumps({
        "selected_policy": asdict(policy),
        "activation_allowed": active_and_positive,
        "tune": results["tune"],
        "validation": results["validation"],
        "development": results["development"],
    }, indent=2, default=_json_default))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    epochs = args.epochs if args.epochs is not None else (1 if args.smoke else Config.epochs)
    config = Config(epochs=epochs, batch_size=1024 if args.smoke else Config.batch_size)
    run(config)
