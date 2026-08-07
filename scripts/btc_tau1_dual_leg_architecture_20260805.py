"""Tau1 BTC dual-leg model contract; training/evaluation are intentionally separate.

Leg A consumes 48 causal 5-minute rows. Leg B consumes 192 completed hourly rows.
Both use 99 causalfix fields, six DVOL fields, six on-chain fields, and a separate
24-input Regime3 encoder. Targets are joined only after feature construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.causal_futures_backtest import purged_decision_mask  # noqa: E402

PANEL = ROOT / "data/splits/year_oos/btc_unified_raw_panel_20260804.parquet"
CAUSALFIX = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
LEG_A_LABELS = ROOT / "tmp/btc_leg_a_tactical_labels_20260805/labels.parquet"
LEG_B_LABELS = ROOT / "tmp/btc_tau1_continuation_labels_20260805/labels.parquet"

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
CHECKPOINT_END = pd.Timestamp("2025-11-01", tz="UTC")
CALIBRATION_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
LEG_A_SEQUENCE, LEG_B_SEQUENCE = 48, 192
LEG_A_HORIZON_BARS, LEG_B_HORIZON_BARS = 96, 3456

RAW_EXCLUDE = {
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value", "close_btc", "volume_btc",
    "quote_volume_btc", "mtf1h_ts_t_value",
}
DVOL_COLS = ["dvol_btc", "dvol_eth", "dvol_btc_eth_spread", "dvol_btc_pctrank_720h", "dvol_btc_roc_24h", "dvol_btc_roc_168h"]
ONCHAIN_COLS = ["mvrv", "mvrv_pctrank_90d", "net_exchange_flow_pct_supply", "sply_ex_roc_7d", "active_addr_roc_7d", "active_addr_pctrank_90d"]
REGIME24_COLS = [
    "state7_trend_score", "state7_trend_efficiency_48", "state7_directional_return_48", "state7_volatility_state",
    "state7_sign_flip_rate_24", "state7_range_compression", "state7_flow_alignment", "state12_log_return",
    "state12_garman_klass_vol", "state12_net_taker_ratio", "state12_oi_change_rate", "state12_chop_index",
    "volatility_z", "rsi", "macd_hist", "bb_width_z", "hma_slope", "wick_ratio", "mtf_trend_1h",
    "mtf_trend_4h", "breakout_strength", "mean_reversion_z", "ofi_acceleration", "taker_acceleration",
]


def load_feature_frame() -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = pd.read_parquet(PANEL).sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    causalfix_columns = pd.read_parquet(CAUSALFIX, columns=None).columns.tolist()
    causalfix99 = [column for column in causalfix_columns if column not in RAW_EXCLUDE]
    if len(causalfix99) != 99:
        raise RuntimeError(f"Expected 99 causalfix fields, got {len(causalfix99)}")
    if any(column not in frame for column in REGIME24_COLS + DVOL_COLS + ONCHAIN_COLS):
        raise RuntimeError("Tau1 feature source is missing a required field")
    market111 = causalfix99 + DVOL_COLS + ONCHAIN_COLS
    if len(market111) != 111:
        raise RuntimeError("Tau1 111-field market contract mismatch")
    # DVOL/on-chain/state7-12 columns carry scattered NaN (source availability gaps,
    # not confined to history warm-up -- confirmed up to the last row of the panel).
    # Forward-fill only (causal: never uses a future value) so a mid-window gap
    # doesn't propagate NaN through the sequence models below.
    frame[market111] = frame[market111].ffill()
    frame[REGIME24_COLS] = frame[REGIME24_COLS].ffill()
    return frame, market111, REGIME24_COLS


def hourly_completed_features(frame: pd.DataFrame, market111: list[str], regime24: list[str]) -> pd.DataFrame:
    """Timestamp each hourly row at its close; it is therefore available at that decision time."""
    indexed = frame.set_index("timestamp")
    market = indexed[market111].resample("1h", label="left", closed="left").last()
    regime = indexed[regime24].resample("1h", label="left", closed="left").last()
    regime.columns = [f"regime_input_{column}" for column in regime.columns]
    hourly = pd.concat([market, regime], axis=1).dropna().reset_index()
    hourly["timestamp"] = hourly["timestamp"] + pd.Timedelta(hours=1)
    return hourly


def join_targets(feature_timestamps: pd.Series, leg: str) -> tuple[np.ndarray, pd.DataFrame]:
    path = LEG_A_LABELS if leg == "A" else LEG_B_LABELS
    labels = pd.read_parquet(path).copy()
    labels["decision_timestamp"] = pd.to_datetime(labels["decision_timestamp"], utc=True)
    joined = pd.DataFrame({"timestamp": pd.to_datetime(feature_timestamps, utc=True)}).merge(
        labels[["decision_timestamp", "label"]], left_on="timestamp", right_on="decision_timestamp", how="left", validate="one_to_one"
    )
    target = joined["label"].fillna(-1).to_numpy(np.int64)
    return target, joined


def purged_splits(timestamps: pd.DatetimeIndex, horizon_bars: int) -> dict[str, np.ndarray]:
    return {
        "train": purged_decision_mask(timestamps, start=timestamps[0], end=TRAIN_END, horizon_bars=horizon_bars),
        "checkpoint": purged_decision_mask(timestamps, start=TRAIN_END, end=CHECKPOINT_END, horizon_bars=horizon_bars),
        "calibration": purged_decision_mask(timestamps, start=CHECKPOINT_END, end=CALIBRATION_END, horizon_bars=horizon_bars),
        "oos": purged_decision_mask(timestamps, start=CALIBRATION_END, end=OOS_END, horizon_bars=horizon_bars),
    }


class CausalBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(width, width, kernel_size=3, dilation=dilation)
        self.norm, self.dropout = nn.LayerNorm(width), nn.Dropout(0.10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(F.pad(x, (self.padding, 0)))
        z = self.norm(z.transpose(1, 2)).transpose(1, 2)
        return F.gelu(x + self.dropout(z))


class RegimeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(24, 16), nn.GELU(), nn.Linear(16, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LegANet(nn.Module):
    """48x111 -> 48-wide causal TCN; current Regime24 -> 4; 52 -> 32 -> 3."""
    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Linear(111, 48)
        self.tcn = nn.Sequential(CausalBlock(48, 1), CausalBlock(48, 2), CausalBlock(48, 4))
        self.regime = RegimeEncoder()
        self.head = nn.Sequential(nn.Linear(52, 32), nn.GELU(), nn.Linear(32, 3))

    def forward(self, market: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        z = self.tcn(self.project(market).transpose(1, 2))[:, :, -1]
        return self.head(torch.cat([z, self.regime(regime[:, -1])], dim=1))


class LegBNet(nn.Module):
    """192x111 -> 40-wide one-layer GRU; current Regime24 -> 4; 44 -> 24 -> 3."""
    def __init__(self) -> None:
        super().__init__()
        self.project, self.gru, self.regime = nn.Linear(111, 40), nn.GRU(40, 40, batch_first=True), RegimeEncoder()
        self.head = nn.Sequential(nn.Linear(44, 24), nn.GELU(), nn.Linear(24, 3))

    def forward(self, market: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(self.project(market))
        return self.head(torch.cat([hidden[-1], self.regime(regime[:, -1])], dim=1))
