"""Train and evaluate DeepScalp-PnL v1.

This is a causal, end-to-end ETH 1m position policy.  It does not predict the
project's triple-barrier labels and it does not apply entry thresholds, fixed
TP/SL, cooldowns, or fixed holding periods.  At every completed minute the
network emits a target side (SHORT/CASH/LONG) and margin fraction.  Training
maximizes cost-adjusted log equity directly while penalizing CVaR and drawdown.

Data integrity:
  * data/live/microstructure.duckdb is always opened read-only.
  * ETH/USDT order-book snapshots only (USDC-margined rows excluded -- too noisy,
    see project memory).
  * windows use only rows through the decision timestamp.  Future OHLC is used
    only as a training target / replay outcome, never as an input.
  * saved trade ledgers, saved exits, triple-barrier labels, kelly_mult and
    signal_bias are not read.

The current price-feature artifact ends on 2026-07-12.  Consequently the July
readout is named development_oos, not untouched/promotion OOS.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
BASE_CSV = ROOT / "data/training_features_1m.csv"
MICRO_DB = ROOT / "data/live/microstructure.duckdb"
BASE_MODEL_ID = "deepscalp_pnl_v1_20260717"
MODEL_ID = os.environ.get("DEEPSCALP_MODEL_ID", BASE_MODEL_ID)
ARTIFACT_DIR = ROOT / f"data/ensemble/{MODEL_ID}"
CACHE_DIR = ROOT / f"data/ensemble/{BASE_MODEL_ID}/cache"
REPORT_PATH = ROOT / f"data/ensemble/reports/{MODEL_ID}.json"
CHECKPOINT_PATH = ARTIFACT_DIR / "model.pt"
LEDGER_PATH = ARTIFACT_DIR / "development_oos_policy_ledger.csv"
SCALER_PATH = ARTIFACT_DIR / "scalers.json"

SIDE_NAMES = ("SHORT", "CASH", "LONG")
SIDE_VALUES = (-1.0, 0.0, 1.0)
AUX_NAMES = ("ret_1m_bp", "ret_2m_bp", "ret_3m_bp", "ret_5m_bp", "mfe_5m_bp", "mae_5m_bp", "rv_5m_bp")

# Stable, mostly dimensionless market inputs.  Raw OHLC is used to derive
# stationary bar features below, not passed as absolute price levels.
BASE_FEATURES = (
    "whale_retail_ratio", "whale_conviction", "smart_money_flow", "squeeze_power",
    "oi_change_rate", "net_taker_ratio", "taker_acceleration", "trade_intensity",
    "big_trade_ratio", "log_return", "volatility_z", "rsi", "macd_hist",
    "bb_width_z", "hma_slope", "wick_ratio", "garman_klass_vol",
    "realized_vol_ratio", "amihud_illiquidity_z", "chop_index", "hour_sin", "hour_cos", "minute_sin",
    "minute_cos", "cvp_poc_dist", "cvp_volume_imbalance", "mean_reversion_z",
    "breakout_strength", "funding_z_score", "long_squeeze_risk",
    "short_squeeze_risk", "ofi_acceleration", "kalman_velocity",
    "realized_skewness", "cvd_slope_12", "cvd_slope_48", "compression_score",
    "vwap_dist_24", "funding_oi_divergence", "upper_wick_z",
    "lower_wick_z", "liquidity_vacuum", "execution_quality",
)

# Deliberately excludes derived trading recommendations/scores such as EAI,
# kelly_mult, signal_bias, shadow regime tags and all legacy model predictions.
MICRO_COLUMNS = (
    "obi", "taker_buy_ratio", "nif_whale", "nif_retail", "oi_delta_pct",
    "funding_rate", "recent_trade_count_5m", "recent_trade_notional_5m",
    "recent_whale_count_5m", "data_stale", "depth_connected", "trade_connected",
    "poll_connected", "depth_age_sec", "trade_age_sec", "poll_age_sec",
    "valid_taker_flow", "valid_nif", "warmup_30m_ready",
)

BOOK_COLUMNS = (
    "spread_bps", "microprice_edge_bps", "imbalance_1", "imbalance_5",
    "imbalance_10", "imbalance_20", "bid_notional_1", "ask_notional_1",
    "bid_notional_5", "ask_notional_5", "bid_notional_10", "ask_notional_10",
    "bid_notional_20", "ask_notional_20",
)


@dataclass(frozen=True)
class Config:
    seed: int = 17
    window: int = 120
    block: int = 64
    burn_in: int = 8
    leverage: float = 3.0
    max_margin_fraction: float = 0.30
    fee_per_notional: float = 0.00045
    base_channels: int = 64
    base_hidden: int = 96
    micro_channels: int = 32
    dropout: float = 0.10
    pretrain_batch_size: int = 768
    policy_batch_size: int = 8
    pretrain_epochs: int = 4
    policy_epochs: int = 16
    pretrain_samples: int = 240_000
    pretrain_lr: float = 3e-4
    policy_lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    profit_scale: float = 10_000.0
    cvar_weight: float = 0.01
    drawdown_weight: float = 0.02
    auxiliary_weight: float = 0.05
    pretrain_end: str = "2026-03-31 23:59:00"
    pretrain_val_end: str = "2026-04-30 23:59:00"
    micro_train_start: str = "2026-05-03 00:00:00"
    micro_train_end: str = "2026-06-20 23:59:00"
    validation_end: str = "2026-06-30 23:59:00"
    development_oos_end: str = "2026-07-12 09:00:00"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _source_signature() -> dict[str, Any]:
    result: dict[str, Any] = {
        "cache_schema_version": 4,
        "base_csv": {"path": str(BASE_CSV), "size": BASE_CSV.stat().st_size, "mtime_ns": BASE_CSV.stat().st_mtime_ns},
        # The live DB can grow while a frozen price artifact is being trained.  Its
        # file size is recorded for audit but is not part of the feature-contract
        # hash because rows after the base artifact's maximum timestamp are never
        # queried and cannot change this cache.
        "micro_db": {"path": str(MICRO_DB), "observed_size": MICRO_DB.stat().st_size},
        "base_features": list(BASE_FEATURES),
        "micro_columns": list(MICRO_COLUMNS),
        "book_columns": list(BOOK_COLUMNS),
    }
    contract_payload = copy.deepcopy(result)
    contract_payload["micro_db"].pop("observed_size")
    payload = json.dumps(contract_payload, sort_keys=True).encode()
    result["contract_sha256"] = hashlib.sha256(payload).hexdigest()
    return result


def _quote_path(path: Path) -> str:
    return str(path).replace("'", "''")


def read_base_frame() -> pd.DataFrame:
    raw_cols = ("timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote")
    selected = list(dict.fromkeys(raw_cols + BASE_FEATURES))
    select_sql = ", ".join(f'"{col}"' for col in selected)
    csv_path = _quote_path(BASE_CSV)
    con = duckdb.connect()
    try:
        frame = con.execute(
            f"""
            SELECT {select_sql}
            FROM read_csv_auto('{csv_path}', header=true, sample_size=20000)
            ORDER BY timestamp
            """
        ).fetchdf()
    finally:
        con.close()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=False)
    if frame["timestamp"].duplicated().any():
        duplicates = int(frame["timestamp"].duplicated(keep=False).sum())
        raise RuntimeError(f"base feature contract violation: {duplicates} duplicate timestamps")
    if not frame["timestamp"].is_monotonic_increasing:
        raise RuntimeError("base feature timestamps are not increasing")
    return frame


def read_micro_frames(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read microstructure and USDT-only book snapshots without writing DuckDB."""
    con = duckdb.connect(str(MICRO_DB), read_only=True)
    try:
        micro_select = ", ".join(f'"{col}"' for col in MICRO_COLUMNS)
        micro = con.execute(
            f"""
            SELECT
                timezone('UTC', ts) AS timestamp,
                timezone('UTC', ts) AS micro_source_ts,
                {micro_select}
            FROM microstructure_1m
            WHERE timezone('UTC', ts) BETWEEN ? AND ?
            ORDER BY timestamp
            """,
            [start.to_pydatetime(), end.to_pydatetime()],
        ).fetchdf()
        book_select = ", ".join(f'"{col}"' for col in BOOK_COLUMNS)
        book = con.execute(
            f"""
            SELECT
                timezone('UTC', recorded_at_kst) AS timestamp,
                timezone('UTC', recorded_at_kst) AS book_source_ts,
                {book_select}
            FROM orderbook_decision_snapshots
            WHERE symbol = 'ETH/USDT:USDT'
              AND timezone('UTC', recorded_at_kst) BETWEEN ? AND ?
            ORDER BY timestamp
            """,
            [start.to_pydatetime(), end.to_pydatetime()],
        ).fetchdf()
    finally:
        con.close()
    for frame in (micro, book):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    micro["micro_source_ts"] = pd.to_datetime(micro["micro_source_ts"])
    book["book_source_ts"] = pd.to_datetime(book["book_source_ts"])
    return micro, book


def make_base_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    eps = 1e-12
    derived = pd.DataFrame(index=frame.index)
    derived["bar_open_close_logret"] = np.log(np.maximum(frame["close"], eps) / np.maximum(frame["open"], eps))
    derived["bar_range_pct"] = (frame["high"] - frame["low"]) / np.maximum(frame["close"], eps)
    derived["log_volume"] = np.log1p(np.maximum(frame["volume"], 0.0))
    derived["log_quote_volume"] = np.log1p(np.maximum(frame["quote_volume"], 0.0))
    derived["log_trade_count"] = np.log1p(np.maximum(frame["trades"], 0.0))
    derived["bar_taker_buy_ratio"] = frame["taker_buy_quote"] / np.maximum(frame["quote_volume"], eps)
    names = list(derived.columns) + list(BASE_FEATURES)
    values = np.column_stack([derived.to_numpy(dtype=np.float32), frame[list(BASE_FEATURES)].to_numpy(dtype=np.float32)])
    values[~np.isfinite(values)] = np.nan
    return values.astype(np.float32), names


def make_targets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    close = frame["close"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    n = len(frame)
    targets = np.full((n, len(AUX_NAMES)), np.nan, dtype=np.float32)
    for j, horizon in enumerate((1, 2, 3, 5)):
        targets[:-horizon, j] = (np.log(close[horizon:] / close[:-horizon]) * 10_000.0).astype(np.float32)
    future_highs = np.column_stack([high[h : n - 5 + h] for h in range(1, 6)])
    future_lows = np.column_stack([low[h : n - 5 + h] for h in range(1, 6)])
    targets[:-5, 4] = ((future_highs.max(axis=1) / close[:-5] - 1.0) * 10_000.0).astype(np.float32)
    targets[:-5, 5] = ((future_lows.min(axis=1) / close[:-5] - 1.0) * 10_000.0).astype(np.float32)
    one_min = np.full(n, np.nan, dtype=np.float64)
    one_min[:-1] = close[1:] / close[:-1] - 1.0
    future_sq = np.column_stack([np.r_[one_min[h - 1 :], np.full(h - 1, np.nan)][:n] ** 2 for h in range(1, 6)])
    targets[:, 6] = (np.sqrt(np.nansum(future_sq, axis=1)) * 10_000.0).astype(np.float32)
    next_return = one_min.astype(np.float32)
    targets[-5:] = np.nan
    return targets, next_return


def make_micro_features(base_ts: pd.Series, micro: pd.DataFrame, book: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    start = max(pd.Timestamp(base_ts.min()), pd.Timestamp("2026-05-03"))
    aligned = pd.DataFrame({"timestamp": base_ts})
    micro_renamed = micro.rename(columns={c: f"micro_{c}" for c in MICRO_COLUMNS})
    aligned = pd.merge_asof(
        aligned.sort_values("timestamp"), micro_renamed.sort_values("timestamp"), on="timestamp",
        direction="backward", tolerance=pd.Timedelta("2min"),
    )
    book_renamed = book.rename(columns={c: f"book_{c}" for c in BOOK_COLUMNS})
    aligned = pd.merge_asof(
        aligned.sort_values("timestamp"), book_renamed.sort_values("timestamp"), on="timestamp",
        direction="backward", tolerance=pd.Timedelta("6min"),
    )
    aligned["micro_available"] = aligned["micro_source_ts"].notna().astype(float)
    aligned["book_available"] = aligned["book_source_ts"].notna().astype(float)
    aligned["micro_age_min"] = (aligned["timestamp"] - aligned["micro_source_ts"]).dt.total_seconds() / 60.0
    aligned["book_age_min"] = (aligned["timestamp"] - aligned["book_source_ts"]).dt.total_seconds() / 60.0

    for col in ("recent_trade_count_5m", "recent_trade_notional_5m", "recent_whale_count_5m"):
        name = f"micro_{col}"
        aligned[name] = np.log1p(np.maximum(pd.to_numeric(aligned[name], errors="coerce"), 0.0))
    for col in BOOK_COLUMNS:
        if "notional" in col:
            name = f"book_{col}"
            aligned[name] = np.log1p(np.maximum(pd.to_numeric(aligned[name], errors="coerce"), 0.0))
    feature_names = [f"micro_{c}" for c in MICRO_COLUMNS] + [f"book_{c}" for c in BOOK_COLUMNS] + [
        "micro_available", "book_available", "micro_age_min", "book_age_min",
    ]
    values = aligned[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    values[~np.isfinite(values)] = np.nan
    coverage_mask = aligned["timestamp"] >= start
    stats = {
        "micro_rows_read": int(len(micro)),
        "book_rows_read": int(len(book)),
        "micro_coverage_after_start": float(aligned.loc[coverage_mask, "micro_available"].mean()),
        "book_coverage_after_start": float(aligned.loc[coverage_mask, "book_available"].mean()),
        "usdc_usdt_mixed_without_symbol_feature": False,
    }
    return values, feature_names, stats


def _cache_paths() -> dict[str, Path]:
    return {
        "base": CACHE_DIR / "base.npy",
        "micro": CACHE_DIR / "micro.npy",
        "targets": CACHE_DIR / "targets.npy",
        "next_return": CACHE_DIR / "next_return.npy",
        "timestamp_ns": CACHE_DIR / "timestamp_ns.npy",
        "metadata": CACHE_DIR / "metadata.json",
    }


def build_or_load_cache(rebuild: bool = False) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    paths = _cache_paths()
    signature = _source_signature()
    if not rebuild and all(path.exists() for path in paths.values()):
        metadata = json.loads(paths["metadata"].read_text())
        if metadata.get("source_signature", {}).get("contract_sha256") == signature["contract_sha256"]:
            arrays = {key: np.load(path, mmap_mode="r") for key, path in paths.items() if key != "metadata"}
            return arrays, metadata

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Reading selected base features from CSV...", flush=True)
    frame = read_base_frame()
    print(f"  base rows={len(frame):,} {frame['timestamp'].min()} -> {frame['timestamp'].max()}", flush=True)
    base, base_names = make_base_features(frame)
    targets, next_return = make_targets(frame)
    micro_start = pd.Timestamp("2026-05-03")
    micro, book = read_micro_frames(micro_start, frame["timestamp"].max())
    micro_values, micro_names, coverage = make_micro_features(frame["timestamp"], micro, book)
    arrays_to_save = {
        "base": base,
        "micro": micro_values,
        "targets": targets,
        "next_return": next_return,
        "timestamp_ns": frame["timestamp"].astype("datetime64[ns]").astype("int64").to_numpy(),
    }
    for key, values in arrays_to_save.items():
        np.save(paths[key], values)
    metadata = {
        "source_signature": signature,
        "base_feature_names": base_names,
        "micro_feature_names": micro_names,
        "aux_target_names": list(AUX_NAMES),
        "n_rows": len(frame),
        "timestamp_min": str(frame["timestamp"].min()),
        "timestamp_max": str(frame["timestamp"].max()),
        "coverage": coverage,
        "duckdb_open_mode": "read_only",
        "excluded_rule_inputs": [
            "scalp_action", "scalp_tp_move", "scalp_sl_move", "kelly_mult", "signal_bias",
            "eai", "shadow_regime_tag", "saved trade ledgers", "saved exit timestamps",
            "all BTC-derived features (source 5m candles are stamped at bar open and are not decision-time safe without a +5m availability shift)",
        ],
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, default=_json_default))
    arrays = {key: np.load(path, mmap_mode="r") for key, path in paths.items() if key != "metadata"}
    return arrays, metadata


def fit_robust_scaler(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample = np.asarray(values[mask], dtype=np.float64)
    center = np.nanmedian(sample, axis=0)
    mad = np.nanmedian(np.abs(sample - center), axis=0) * 1.4826
    std = np.nanstd(sample, axis=0)
    scale = np.where((mad > 1e-8) & np.isfinite(mad), mad, std)
    scale = np.where((scale > 1e-8) & np.isfinite(scale), scale, 1.0)
    center = np.where(np.isfinite(center), center, 0.0)
    return center.astype(np.float32), scale.astype(np.float32)


def apply_scaler(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    result = (np.asarray(values, dtype=np.float32) - center) / scale
    result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(result, -10.0, 10.0).astype(np.float32)


def continuous_run_lengths(timestamp_ns: np.ndarray) -> np.ndarray:
    timestamp_ns = np.asarray(timestamp_ns, dtype=np.int64)
    run = np.ones(len(timestamp_ns), dtype=np.int32)
    if len(run) == 0:
        return run
    consecutive = np.diff(timestamp_ns) == 60_000_000_000
    for idx in range(1, len(run)):
        run[idx] = run[idx - 1] + 1 if consecutive[idx - 1] else 1
    return run


def causal_window_end_indices(timestamp_ns: np.ndarray, targets: np.ndarray, next_return: np.ndarray, window: int) -> np.ndarray:
    run = continuous_run_lengths(timestamp_ns)
    valid = (run >= window) & np.isfinite(targets).all(axis=1) & np.isfinite(next_return)
    return np.flatnonzero(valid)


def make_block_starts(
    timestamp_ns: np.ndarray,
    valid_end: np.ndarray,
    split_start: pd.Timestamp,
    split_end: pd.Timestamp,
    block: int,
    stride: int,
) -> np.ndarray:
    eligible = np.zeros(len(timestamp_ns), dtype=bool)
    eligible[valid_end] = True
    start_ns, end_ns = split_start.value, split_end.value
    eligible &= (timestamp_ns >= start_ns) & (timestamp_ns <= end_ns)
    indices = np.flatnonzero(eligible)
    if len(indices) == 0:
        return np.empty(0, dtype=np.int64)
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    groups = np.split(indices, breaks)
    starts: list[int] = []
    for group in groups:
        if len(group) < block:
            continue
        starts.extend(range(int(group[0]), int(group[-1]) - block + 2, stride))
    return np.asarray(starts, dtype=np.int64)


class WindowDataset(Dataset):
    def __init__(self, base: np.ndarray, targets: np.ndarray, end_indices: np.ndarray, window: int):
        self.windows = sliding_window_view(base, window_shape=window, axis=0)
        self.targets = targets
        self.ends = np.asarray(end_indices, dtype=np.int64)
        self.window = window

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = int(self.ends[item])
        window_idx = end - self.window + 1
        x = np.ascontiguousarray(self.windows[window_idx].T)
        return torch.from_numpy(x), torch.from_numpy(np.asarray(self.targets[end], dtype=np.float32))


class PolicyBlockDataset(Dataset):
    def __init__(
        self,
        base: np.ndarray,
        micro: np.ndarray,
        targets: np.ndarray,
        next_return: np.ndarray,
        starts: np.ndarray,
        window: int,
        block: int,
    ):
        self.base_windows = sliding_window_view(base, window_shape=window, axis=0)
        self.micro_windows = sliding_window_view(micro, window_shape=window, axis=0)
        self.targets = targets
        self.next_return = next_return
        self.starts = np.asarray(starts, dtype=np.int64)
        self.window = window
        self.block = block

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(self.starts[item])
        first_window = start - self.window + 1
        last_window = first_window + self.block
        xb = np.ascontiguousarray(self.base_windows[first_window:last_window].transpose(0, 2, 1))
        xm = np.ascontiguousarray(self.micro_windows[first_window:last_window].transpose(0, 2, 1))
        stop = start + self.block
        ya = np.array(self.targets[start:stop], dtype=np.float32, copy=True, order="C")
        yr = np.array(self.next_return[start:stop], dtype=np.float32, copy=True, order="C")
        return torch.from_numpy(xb), torch.from_numpy(xm), torch.from_numpy(ya), torch.from_numpy(yr)


class ResidualCausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(F.pad(x, (self.left_padding, 0)))
        y = self.dropout(F.gelu(self.norm(y)))
        return x + y


class BaseEncoder(nn.Module):
    def __init__(self, n_features: int, channels: int, hidden: int, dropout: float):
        super().__init__()
        self.projection = nn.Conv1d(n_features, channels, kernel_size=1)
        self.tcn = nn.Sequential(*(ResidualCausalBlock(channels, dilation, dropout) for dilation in (1, 2, 4, 8)))
        self.gru = nn.GRU(channels, hidden, num_layers=1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(self.projection(x.transpose(1, 2))).transpose(1, 2)
        output, _ = self.gru(x)
        return output[:, -1]


class MicroEncoder(nn.Module):
    def __init__(self, n_features: int, channels: int, dropout: float):
        super().__init__()
        self.projection = nn.Conv1d(n_features, channels, kernel_size=1)
        self.tcn = nn.Sequential(*(ResidualCausalBlock(channels, dilation, dropout) for dilation in (1, 2, 4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.tcn(self.projection(x.transpose(1, 2)))
        return encoded[:, :, -1]


class BasePretrainer(nn.Module):
    def __init__(self, encoder: BaseEncoder, hidden: int, n_targets: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_targets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class DeepScalpPolicy(nn.Module):
    def __init__(self, n_base: int, n_micro: int, config: Config):
        super().__init__()
        self.config = config
        self.base_encoder = BaseEncoder(n_base, config.base_channels, config.base_hidden, config.dropout)
        self.micro_encoder = MicroEncoder(n_micro, config.micro_channels, config.dropout)
        market_dim = config.base_hidden + config.micro_channels
        self.state_encoder = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(market_dim + 16, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(config.dropout),
        )
        self.side_head = nn.Linear(128, 3)
        self.margin_head = nn.Linear(128, 1)
        self.auxiliary_head = nn.Sequential(nn.LayerNorm(market_dim), nn.Linear(market_dim, 64), nn.GELU(), nn.Linear(64, len(AUX_NAMES)))
        with torch.no_grad():
            self.side_head.bias.zero_()
            self.margin_head.bias.fill_(-0.5)

    def encode_market(self, base: torch.Tensor, micro: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.base_encoder(base), self.micro_encoder(micro)], dim=-1)

    def policy_step(
        self,
        market: torch.Tensor,
        state: torch.Tensor,
        *,
        action_mode: str,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = self.fusion(torch.cat([market, self.state_encoder(state)], dim=-1))
        logits = self.side_head(fused)
        if action_mode == "gumbel":
            side_one_hot = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        elif action_mode == "soft":
            side_one_hot = torch.softmax(logits / temperature, dim=-1)
        elif action_mode == "argmax":
            side_one_hot = F.one_hot(logits.argmax(dim=-1), num_classes=3).to(logits.dtype)
        else:
            raise ValueError(f"unsupported action_mode={action_mode!r}")
        side_values = logits.new_tensor(SIDE_VALUES)
        side = (side_one_hot * side_values).sum(dim=-1)
        margin = self.config.max_margin_fraction * torch.sigmoid(self.margin_head(fused).squeeze(-1))
        signed_notional = side * margin * self.config.leverage
        return signed_notional, side, margin, logits


def account_return(
    signed_notional: torch.Tensor,
    previous_notional: torch.Tensor,
    price_return: torch.Tensor,
    fee_per_notional: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gross = signed_notional * price_return
    turnover = torch.abs(signed_notional - previous_notional)
    net = gross - fee_per_notional * turnover
    return net, gross, turnover


def unroll_policy(
    model: DeepScalpPolicy,
    market: torch.Tensor,
    price_return: torch.Tensor,
    *,
    action_mode: str,
    temperature: float,
    fee_per_notional: float,
) -> dict[str, torch.Tensor]:
    batch, steps, _ = market.shape
    previous_notional = market.new_zeros(batch)
    previous_side = market.new_zeros(batch)
    holding = market.new_zeros(batch)
    unrealized = market.new_zeros(batch)
    previous_net = market.new_zeros(batch)
    net_list, gross_list, turnover_list, notional_list, side_list, margin_list = [], [], [], [], [], []
    for step in range(steps):
        state = torch.stack(
            [
                previous_notional.detach() / (model.config.leverage * model.config.max_margin_fraction),
                torch.clamp(holding.detach() / 120.0, 0.0, 4.0),
                torch.clamp(unrealized.detach() * 100.0, -5.0, 5.0),
                torch.clamp(previous_net.detach() * 100.0, -5.0, 5.0),
            ],
            dim=-1,
        )
        notional, side, margin, _ = model.policy_step(
            market[:, step], state, action_mode=action_mode, temperature=temperature,
        )
        net, gross, turnover = account_return(notional, previous_notional, price_return[:, step], fee_per_notional)
        net_list.append(net)
        gross_list.append(gross)
        turnover_list.append(turnover)
        notional_list.append(notional)
        side_list.append(side)
        margin_list.append(margin)
        state_side = torch.sign(side.detach())
        same_side = (state_side == previous_side) & (state_side != 0)
        holding = torch.where(same_side, holding + 1.0, torch.where(side.detach() != 0, torch.ones_like(holding), torch.zeros_like(holding)))
        unrealized = torch.where(
            same_side,
            unrealized + side.detach() * price_return[:, step].detach(),
            torch.zeros_like(unrealized),
        )
        previous_side = state_side
        previous_net = net.detach()
        previous_notional = notional
    return {
        "net": torch.stack(net_list, dim=1),
        "gross": torch.stack(gross_list, dim=1),
        "turnover": torch.stack(turnover_list, dim=1),
        "notional": torch.stack(notional_list, dim=1),
        "side": torch.stack(side_list, dim=1),
        "margin": torch.stack(margin_list, dim=1),
    }


def policy_loss(
    outputs: dict[str, torch.Tensor],
    auxiliary_prediction: torch.Tensor,
    auxiliary_target: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    net = outputs["net"][:, config.burn_in :]
    safe_net = torch.clamp(net, min=-0.95, max=0.95)
    negative_log_growth = -torch.log1p(safe_net).mean() * config.profit_scale
    flattened = net.flatten()
    tail_count = max(1, int(math.ceil(0.05 * flattened.numel())))
    cvar_loss = -torch.topk(flattened, tail_count, largest=False).values.mean() * config.profit_scale
    log_equity = torch.cumsum(torch.log1p(safe_net), dim=1)
    equity = torch.exp(log_equity)
    peak = torch.cummax(equity, dim=1).values
    soft_drawdown = (1.0 - equity / torch.clamp(peak, min=1e-8)).amax(dim=1).mean()
    aux_loss = F.smooth_l1_loss(auxiliary_prediction[:, config.burn_in :], auxiliary_target[:, config.burn_in :])
    loss = (
        negative_log_growth
        + config.cvar_weight * cvar_loss
        + config.drawdown_weight * soft_drawdown
        + config.auxiliary_weight * aux_loss
    )
    metrics = {
        "loss": float(loss.detach()),
        "negative_log_growth": float(negative_log_growth.detach()),
        "cvar_loss": float(cvar_loss.detach()),
        "soft_drawdown": float(soft_drawdown.detach()),
        "auxiliary_loss": float(aux_loss.detach()),
        "mean_net_bp": float(net.detach().mean() * 10_000.0),
        "mean_turnover": float(outputs["turnover"][:, config.burn_in :].detach().mean()),
    }
    return loss, metrics


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def train_base_pretrainer(
    base: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    target_center: np.ndarray,
    target_scale: np.ndarray,
    config: Config,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    rng = np.random.default_rng(config.seed)
    if len(train_indices) > config.pretrain_samples:
        train_indices = np.sort(rng.choice(train_indices, config.pretrain_samples, replace=False))
    if len(val_indices) > 40_000:
        val_indices = np.sort(rng.choice(val_indices, 40_000, replace=False))
    normalized_targets = apply_scaler(targets, target_center, target_scale)
    train_loader = DataLoader(
        WindowDataset(base, normalized_targets, train_indices, config.window),
        batch_size=config.pretrain_batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        WindowDataset(base, normalized_targets, val_indices, config.window),
        batch_size=config.pretrain_batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda",
    )
    encoder = BaseEncoder(base.shape[1], config.base_channels, config.base_hidden, config.dropout)
    model = BasePretrainer(encoder, config.base_hidden, targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.pretrain_lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, config.pretrain_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = F.smooth_l1_loss(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach()))
        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                prediction = model(xb)
                val_sum += float(F.smooth_l1_loss(prediction, yb, reduction="sum"))
                val_n += yb.numel()
        val_loss = val_sum / max(val_n, 1)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": val_loss}
        history.append(row)
        print(f"[pretrain {epoch:02d}] train={row['train_loss']:.5f} val={val_loss:.5f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.encoder.state_dict().items()}
    if best_state is None:
        raise RuntimeError("base pretraining produced no checkpoint")
    return best_state, history


def encode_end_indices(
    model: DeepScalpPolicy,
    base: np.ndarray,
    micro: np.ndarray,
    end_indices: np.ndarray,
    window: int,
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    base_windows = sliding_window_view(base, window_shape=window, axis=0)
    micro_windows = sliding_window_view(micro, window_shape=window, axis=0)
    embeddings: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(end_indices), batch_size):
            ends = end_indices[offset : offset + batch_size]
            window_indices = ends - window + 1
            xb = np.ascontiguousarray(base_windows[window_indices].transpose(0, 2, 1))
            xm = np.ascontiguousarray(micro_windows[window_indices].transpose(0, 2, 1))
            embeddings.append(model.encode_market(torch.from_numpy(xb).to(device), torch.from_numpy(xm).to(device)).cpu())
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty((0, model.config.base_hidden + model.config.micro_channels))


def replay_from_embeddings(
    model: DeepScalpPolicy,
    embeddings: torch.Tensor,
    price_return: np.ndarray,
    timestamp_ns: np.ndarray,
    fee_per_notional: float,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    previous_notional = torch.zeros(1, device=device)
    previous_side = torch.zeros(1, device=device)
    holding = torch.zeros(1, device=device)
    unrealized = torch.zeros(1, device=device)
    previous_net = torch.zeros(1, device=device)
    records = []
    with torch.no_grad():
        for idx in range(len(embeddings)):
            state = torch.stack(
                [
                    previous_notional / (model.config.leverage * model.config.max_margin_fraction),
                    torch.clamp(holding / 120.0, 0.0, 4.0),
                    torch.clamp(unrealized * 100.0, -5.0, 5.0),
                    torch.clamp(previous_net * 100.0, -5.0, 5.0),
                ],
                dim=-1,
            )
            notional, side, margin, logits = model.policy_step(
                embeddings[idx : idx + 1].to(device), state, action_mode="argmax",
            )
            step_return = torch.tensor([price_return[idx]], dtype=torch.float32, device=device)
            net, gross, turnover = account_return(notional, previous_notional, step_return, fee_per_notional)
            side_idx = int(logits.argmax(dim=-1).item())
            records.append(
                {
                    "timestamp": pd.Timestamp(int(timestamp_ns[idx])),
                    "side": SIDE_NAMES[side_idx],
                    "margin_fraction": float(margin.item()),
                    "signed_notional": float(notional.item()),
                    "price_return": float(price_return[idx]),
                    "gross_account_return": float(gross.item()),
                    "turnover": float(turnover.item()),
                    "net_account_return": float(net.item()),
                }
            )
            same_side = (side == previous_side) & (side != 0)
            holding = torch.where(same_side, holding + 1.0, torch.where(side != 0, torch.ones_like(holding), torch.zeros_like(holding)))
            unrealized = torch.where(same_side, unrealized + side * step_return, torch.zeros_like(unrealized))
            previous_side, previous_notional, previous_net = side, notional, net
    ledger = pd.DataFrame(records)
    metrics = summarize_ledger(ledger)
    return metrics, ledger


def summarize_ledger(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {}
    net = ledger["net_account_return"].to_numpy(dtype=float)
    gross = ledger["gross_account_return"].to_numpy(dtype=float)
    equity = np.cumprod(1.0 + np.clip(net, -0.95, None))
    peak = np.maximum.accumulate(equity)
    drawdown = 1.0 - equity / peak
    side = ledger["side"].to_numpy()
    previous_side = np.concatenate([np.asarray(["CASH"], dtype=side.dtype), side[:-1]])
    entries = (side != "CASH") & (side != previous_side)
    exits = (previous_side != "CASH") & (side != previous_side)
    active = side != "CASH"
    daily = ledger.assign(date=pd.to_datetime(ledger["timestamp"]).dt.date).groupby("date")["net_account_return"].apply(
        lambda x: float(np.prod(1.0 + x.to_numpy()) - 1.0)
    )
    rng = np.random.default_rng(20260717)
    if len(daily):
        boot = np.asarray([rng.choice(daily.to_numpy(), size=len(daily), replace=True).mean() for _ in range(5000)])
        daily_ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    else:
        daily_ci = [None, None]
    return {
        "bars": int(len(ledger)),
        "days": int(len(daily)),
        "compounded_return_pct": float((equity[-1] - 1.0) * 100.0),
        "additive_net_return_pct": float(net.sum() * 100.0),
        "additive_gross_return_pct": float(gross.sum() * 100.0),
        "max_drawdown_pct": float(drawdown.max() * 100.0),
        "entries_or_reversals": int(entries.sum()),
        "exits_or_reversals": int(exits.sum()),
        "exposure_fraction": float(active.mean()),
        "average_margin_when_active": float(ledger.loc[active, "margin_fraction"].mean()) if active.any() else 0.0,
        "turnover": float(ledger["turnover"].sum()),
        "mean_daily_return_pct": float(daily.mean() * 100.0),
        "positive_day_fraction": float((daily > 0).mean()),
        "mean_daily_return_95pct_bootstrap_ci": daily_ci,
        "side_fraction": {name: float((side == name).mean()) for name in SIDE_NAMES},
    }


def cost_stress_from_ledger(ledger: pd.DataFrame, costs: Iterable[float]) -> dict[str, dict[str, Any]]:
    result = {}
    for cost in costs:
        stressed = ledger.copy()
        stressed["net_account_return"] = stressed["gross_account_return"] - float(cost) * stressed["turnover"]
        result[f"{cost * 10_000:.2f}bp_per_notional_change"] = summarize_ledger(stressed)
    return result


def validation_score(metrics: dict[str, Any]) -> float:
    return float(metrics.get("compounded_return_pct", -1e9) - 0.5 * metrics.get("max_drawdown_pct", 1e9))


def train_policy(
    model: DeepScalpPolicy,
    base: np.ndarray,
    micro: np.ndarray,
    targets: np.ndarray,
    next_return: np.ndarray,
    timestamp_ns: np.ndarray,
    train_starts: np.ndarray,
    validation_indices: np.ndarray,
    normalized_targets: np.ndarray,
    config: Config,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    dataset = PolicyBlockDataset(base, micro, normalized_targets, next_return, train_starts, config.window, config.block)
    loader = DataLoader(
        dataset, batch_size=config.policy_batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda", drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.policy_lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    patience = 0
    soft_epochs = min(4, max(2, config.policy_epochs // 3))
    for epoch in range(1, config.policy_epochs + 1):
        model.train()
        epoch_rows = []
        temperature = max(0.35, 1.0 - 0.05 * (epoch - 1))
        action_mode = "soft" if epoch <= soft_epochs else "gumbel"
        training_fee = config.fee_per_notional * min(1.0, epoch / max(soft_epochs, 1))
        for xb, xm, ya, yr in loader:
            batch, steps, window, _ = xb.shape
            xb = xb.reshape(batch * steps, window, -1).to(device, non_blocking=True)
            xm = xm.reshape(batch * steps, window, -1).to(device, non_blocking=True)
            ya = ya.to(device, non_blocking=True)
            yr = yr.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                market = model.encode_market(xb, xm).reshape(batch, steps, -1)
                auxiliary = model.auxiliary_head(market)
                outputs = unroll_policy(
                    model, market, yr, action_mode=action_mode, temperature=temperature,
                    fee_per_notional=training_fee,
                )
                loss, batch_metrics = policy_loss(outputs, auxiliary, ya, config)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_rows.append(batch_metrics)
        train_metrics = _mean_metrics(epoch_rows)
        embeddings = encode_end_indices(model, base, micro, validation_indices, config.window, device)
        val_metrics, _ = replay_from_embeddings(
            model, embeddings, next_return[validation_indices], timestamp_ns[validation_indices],
            config.fee_per_notional, device,
        )
        score = validation_score(val_metrics)
        row = {
            "epoch": epoch,
            "temperature": temperature,
            "action_mode": action_mode,
            "training_fee_per_notional": training_fee,
            "train": train_metrics,
            "validation": val_metrics,
            "validation_score": score,
        }
        history.append(row)
        print(
            f"[policy {epoch:02d}] loss={train_metrics.get('loss', float('nan')):.4f} "
            f"train_net={train_metrics.get('mean_net_bp', float('nan')):.3f}bp "
            f"val_return={val_metrics.get('compounded_return_pct', float('nan')):.3f}% "
            f"val_mdd={val_metrics.get('max_drawdown_pct', float('nan')):.3f}%",
            flush=True,
        )
        if score > best_score + 1e-6:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                print(f"Early stopping policy at epoch {epoch}", flush=True)
                break
    if best_state is None:
        raise RuntimeError("policy training produced no checkpoint")
    return best_state, history


def _select_indices(valid_end: np.ndarray, timestamp_ns: np.ndarray, start: str, end: str) -> np.ndarray:
    start_ns, end_ns = pd.Timestamp(start).value, pd.Timestamp(end).value
    return valid_end[(timestamp_ns[valid_end] >= start_ns) & (timestamp_ns[valid_end] <= end_ns)]


def _save_checkpoint(
    model: DeepScalpPolicy,
    config: Config,
    metadata: dict[str, Any],
    scalers: dict[str, np.ndarray],
    histories: dict[str, Any],
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": MODEL_ID,
            "model_state": model.state_dict(),
            "config": asdict(config),
            "base_feature_names": metadata["base_feature_names"],
            "micro_feature_names": metadata["micro_feature_names"],
            "aux_target_names": metadata["aux_target_names"],
            "scalers": {key: value.tolist() for key, value in scalers.items()},
            "histories": histories,
            "feature_contract_sha256": metadata["source_signature"]["contract_sha256"],
        },
        CHECKPOINT_PATH,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = Config(
        pretrain_epochs=args.pretrain_epochs,
        policy_epochs=args.policy_epochs,
        pretrain_samples=args.pretrain_samples,
        fee_per_notional=args.fee_per_notional,
    )
    if args.smoke:
        config = Config(
            pretrain_epochs=1, policy_epochs=1, pretrain_samples=8_000, block=32, burn_in=4,
            fee_per_notional=args.fee_per_notional,
        )
    seed_everything(config.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)
    arrays, metadata = build_or_load_cache(args.rebuild_cache)
    timestamp_ns = np.asarray(arrays["timestamp_ns"])
    raw_base = np.asarray(arrays["base"])
    raw_micro = np.asarray(arrays["micro"])
    raw_targets = np.asarray(arrays["targets"])
    next_return = np.asarray(arrays["next_return"])
    ts = pd.to_datetime(timestamp_ns)

    pretrain_mask = ts <= pd.Timestamp(config.pretrain_end)
    policy_train_mask = (ts >= pd.Timestamp(config.micro_train_start)) & (ts <= pd.Timestamp(config.micro_train_end))
    base_center, base_scale = fit_robust_scaler(raw_base, pretrain_mask)
    micro_center, micro_scale = fit_robust_scaler(raw_micro, policy_train_mask)
    target_center, target_scale = fit_robust_scaler(raw_targets, pretrain_mask & np.isfinite(raw_targets).all(axis=1))
    base = apply_scaler(raw_base, base_center, base_scale)
    micro = apply_scaler(raw_micro, micro_center, micro_scale)
    normalized_targets = apply_scaler(raw_targets, target_center, target_scale)
    valid_end = causal_window_end_indices(timestamp_ns, raw_targets, next_return, config.window)

    pretrain_indices = _select_indices(valid_end, timestamp_ns, str(ts.min()), config.pretrain_end)
    pretrain_val_indices = _select_indices(
        valid_end, timestamp_ns, str(pd.Timestamp(config.pretrain_end) + pd.Timedelta(minutes=1)), config.pretrain_val_end,
    )
    validation_indices = _select_indices(
        valid_end, timestamp_ns, str(pd.Timestamp(config.micro_train_end) + pd.Timedelta(minutes=1)), config.validation_end,
    )
    development_oos_indices = _select_indices(
        valid_end, timestamp_ns, str(pd.Timestamp(config.validation_end) + pd.Timedelta(minutes=1)), config.development_oos_end,
    )
    train_starts = make_block_starts(
        timestamp_ns, valid_end, pd.Timestamp(config.micro_train_start), pd.Timestamp(config.micro_train_end),
        config.block, max(1, config.block // 2),
    )
    if args.smoke:
        train_starts = train_starts[np.linspace(0, len(train_starts) - 1, min(64, len(train_starts)), dtype=int)]
        validation_indices = validation_indices[-min(512, len(validation_indices)) :]
        development_oos_indices = development_oos_indices[-min(512, len(development_oos_indices)) :]
    counts = {
        "pretrain_windows": len(pretrain_indices),
        "pretrain_validation_windows": len(pretrain_val_indices),
        "policy_train_blocks": len(train_starts),
        "validation_bars": len(validation_indices),
        "development_oos_bars": len(development_oos_indices),
    }
    if min(counts.values()) <= 0:
        raise RuntimeError(f"empty split after causal-window checks: {counts}")
    print(f"Causal split counts: {counts}", flush=True)

    scalers = {
        "base_center": base_center, "base_scale": base_scale,
        "micro_center": micro_center, "micro_scale": micro_scale,
        "target_center": target_center, "target_scale": target_scale,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SCALER_PATH.write_text(json.dumps({key: value.tolist() for key, value in scalers.items()}, indent=2))

    policy = DeepScalpPolicy(base.shape[1], micro.shape[1], config).to(device)
    if args.evaluate_only:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        expected = metadata["source_signature"]["contract_sha256"]
        if checkpoint["feature_contract_sha256"] != expected:
            raise RuntimeError("feature contract mismatch; rebuild/retrain instead of applying a compatibility fallback")
        policy.load_state_dict(checkpoint["model_state"])
        histories = checkpoint.get("histories", {})
    else:
        base_state, pretrain_history = train_base_pretrainer(
            base, raw_targets, pretrain_indices, pretrain_val_indices, target_center, target_scale, config, device,
        )
        policy.base_encoder.load_state_dict(base_state)
        best_policy_state, policy_history = train_policy(
            policy, base, micro, raw_targets, next_return, timestamp_ns, train_starts,
            validation_indices, normalized_targets, config, device,
        )
        policy.load_state_dict(best_policy_state)
        histories = {"base_pretraining": pretrain_history, "policy_training": policy_history}
        _save_checkpoint(policy, config, metadata, scalers, histories)

    validation_embeddings = encode_end_indices(policy, base, micro, validation_indices, config.window, device)
    validation_metrics, validation_ledger = replay_from_embeddings(
        policy, validation_embeddings, next_return[validation_indices], timestamp_ns[validation_indices],
        config.fee_per_notional, device,
    )
    oos_embeddings = encode_end_indices(policy, base, micro, development_oos_indices, config.window, device)
    development_metrics, development_ledger = replay_from_embeddings(
        policy, oos_embeddings, next_return[development_oos_indices], timestamp_ns[development_oos_indices],
        config.fee_per_notional, device,
    )
    development_ledger.to_csv(LEDGER_PATH, index=False)
    validation_stress = cost_stress_from_ledger(validation_ledger, (0.00020, 0.000325, 0.00045, 0.00055))
    development_stress = cost_stress_from_ledger(development_ledger, (0.00020, 0.000325, 0.00045, 0.00055))
    if validation_metrics.get("entries_or_reversals", 0) == 0:
        promotion_reason = "No profitable active policy survived validation; the selected network is the degenerate all-CASH policy."
    elif validation_metrics.get("compounded_return_pct", 0.0) <= 0.0:
        promotion_reason = "The selected active policy has non-positive validation return after modeled execution cost."
    else:
        promotion_reason = (
            "July data was already used during model-family research and the available microstructure history "
            "is below the four-month continuous promotion requirement."
        )

    report = {
        "model_id": MODEL_ID,
        "model_family": "causal TCN-GRU discrete position policy with learned margin fraction",
        "objective": "cost-adjusted log-equity maximization with CVaR/drawdown penalties and auxiliary distribution targets",
        "execution_assumption": {
            "fee_per_notional_change": config.fee_per_notional,
            "low_cost_proxy_without_fill_model": config.fee_per_notional < 0.00045,
            "actual_order_fill_events_available": False,
        },
        "config": asdict(config),
        "device": str(device),
        "data": {**metadata, "split_counts": counts},
        "validation": validation_metrics,
        "validation_cost_stress": validation_stress,
        "development_oos": development_metrics,
        "development_oos_cost_stress": development_stress,
        "histories": histories,
        "legacy_reference_only_not_same_accounting": {
            "hgb_maker_triple_barrier_additive_total_pnl_pct": 3.7390646402123644,
            "gru_maker_triple_barrier_additive_total_pnl_pct": 3.601387173683871,
            "warning": "legacy models use overlapping triple-barrier trades and additive trade PnL; compare directionally, not as portfolio-equivalent returns",
        },
        "artifacts": {
            "checkpoint": str(CHECKPOINT_PATH),
            "scalers": str(SCALER_PATH),
            "development_oos_diagnostic_ledger": str(LEDGER_PATH),
        },
        "compliance": {
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "triple_barrier_labels_used": False,
            "fixed_entry_threshold_used": False,
            "fixed_tp_sl_used": False,
            "fixed_holding_period_used": False,
            "duckdb_opened_read_only": True,
            "usdc_usdt_orderbook_mixed": False,
            "leverage": config.leverage,
            "notional_formula": "signed_notional = side * margin_fraction * leverage",
        },
        "promotion": {
            "promotion_pass": False,
            "reason": promotion_reason,
            "next_untouched_start": "after model freeze at 2026-07-17",
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=_json_default))
    print(f"Saved report: {REPORT_PATH}", flush=True)
    print(json.dumps({"validation": validation_metrics, "development_oos": development_metrics}, indent=2), flush=True)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=Config.pretrain_epochs)
    parser.add_argument("--policy-epochs", type=int, default=Config.policy_epochs)
    parser.add_argument("--pretrain-samples", type=int, default=Config.pretrain_samples)
    parser.add_argument("--fee-per-notional", type=float, default=Config.fee_per_notional)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
