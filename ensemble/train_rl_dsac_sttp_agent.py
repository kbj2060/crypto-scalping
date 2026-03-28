"""
DSAC + ST/TP training entrypoint.

This file keeps the original DSAC implementation untouched and adds an
environment wrapper that injects stop-loss / take-profit exits into the
training and validation loop.

ST/TP policy:
- Volatility-scaled thresholds from EWMA std of log returns.
- TP = max(tp_floor, tp_mult * vol)
- SL = max(sl_floor, sl_mult * vol)
- If both TP and SL are touched on the same bar, SL wins.
- Max holding time closes the trade at market close.
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR, _SCRIPT_DIR, os.path.join(_ROOT_DIR, "ensemble"), os.path.join(_ROOT_DIR, "strategies")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from ensemble.train_rl_agent import MultiTimeframeFeatures, OnlineHMMDetector, REGIME_COLS  # noqa: E402
from ensemble.train_rl_dsac_agent import (  # noqa: E402
    DSACAgent,
    DSACRouter,
    DSAC_STATE_DIM,
    DSACCompactTradingEnv,
)


ANNUAL_FACTOR_5M = math.sqrt(365 * 24 * 12)


@dataclass
class STTPConfig:
    tp_mult: float = 2.5
    sl_mult: float = 1.2
    max_hold: int = 72
    tp_floor: float = 0.0030
    sl_floor: float = 0.0020
    vol_span: int = 48


@dataclass
class STTPProfile:
    """Regime-conditioned ST/TP profile calibrated from history.

    Arrays are ordered as REGIME_COLS:
      [regime_chop, regime_whipsaw, regime_bull, regime_bear, regime_normal]
    """

    tp_long: np.ndarray
    sl_long: np.ndarray
    tp_short: np.ndarray
    sl_short: np.ndarray
    hold_scale: np.ndarray


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x


def _load_merged_training_frame(csv_path: str, feature_csv: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"training csv not found: {csv_path}")
    if not os.path.exists(feature_csv):
        raise FileNotFoundError(f"feature csv not found: {feature_csv}")

    base_df = pd.read_csv(csv_path)
    feat_df = pd.read_csv(feature_csv, usecols=["timestamp", "open", "high", "low"])

    if "timestamp" not in base_df.columns:
        raise ValueError(f"timestamp column missing in training csv: {csv_path}")
    if "timestamp" not in feat_df.columns:
        raise ValueError(f"timestamp column missing in feature csv: {feature_csv}")

    base_df["timestamp"] = pd.to_datetime(base_df["timestamp"], errors="coerce")
    feat_df["timestamp"] = pd.to_datetime(feat_df["timestamp"], errors="coerce")
    base_df = base_df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    feat_df = feat_df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")

    merged = base_df.merge(feat_df, on="timestamp", how="inner", suffixes=("", "_feat"))
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    for col in ("open", "high", "low"):
        feat_col = f"{col}_feat"
        if feat_col in merged.columns:
            merged[col] = pd.to_numeric(merged[feat_col], errors="coerce")
            merged = merged.drop(columns=[feat_col])
        elif col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    if "close" in merged.columns:
        merged["close"] = pd.to_numeric(merged["close"], errors="coerce")

    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=["timestamp", "close", "open", "high", "low"]).reset_index(drop=True)

    merged["open"] = np.maximum(merged["open"].to_numpy(dtype=np.float64), 1e-8)
    merged["high"] = np.maximum(merged["high"].to_numpy(dtype=np.float64), merged["low"].to_numpy(dtype=np.float64))
    merged["low"] = np.minimum(merged["high"].to_numpy(dtype=np.float64), merged["low"].to_numpy(dtype=np.float64))
    merged["close"] = np.maximum(merged["close"].to_numpy(dtype=np.float64), 1e-8)

    logger.info(
        "[MERGE] base_rows=%d feat_rows=%d merged_rows=%d",
        len(base_df),
        len(feat_df),
        len(merged),
    )
    return merged


def _vol_proxy(close_np: np.ndarray, span: int = 48) -> np.ndarray:
    lr = np.zeros_like(close_np, dtype=np.float64)
    if len(close_np) > 1:
        c = np.maximum(close_np, 1e-8)
        lr[1:] = np.diff(np.log(c))
    v = pd.Series(lr).ewm(span=max(4, int(span)), adjust=False).std(bias=False).fillna(0.0).to_numpy(dtype=np.float64)
    return np.maximum(v, 1e-8)


def _default_sttp_profile() -> STTPProfile:
    # Conservative regime priors. These will be overwritten by calibration.
    return STTPProfile(
        tp_long=np.array([0.0040, 0.0040, 0.0060, 0.0055, 0.0050], dtype=np.float32),
        sl_long=np.array([0.0030, 0.0030, 0.0045, 0.0040, 0.0035], dtype=np.float32),
        tp_short=np.array([0.0040, 0.0040, 0.0060, 0.0055, 0.0050], dtype=np.float32),
        sl_short=np.array([0.0030, 0.0030, 0.0045, 0.0040, 0.0035], dtype=np.float32),
        hold_scale=np.array([0.70, 0.70, 1.25, 1.10, 1.00], dtype=np.float32),
    )


def _regime_index_from_row(row: np.ndarray, n_pred: int, n_conf: int, n_elite: int, n_alpha: int, n_regime: int) -> int:
    o = n_pred + n_conf + n_elite + n_alpha
    regime_vec = row[o:o + n_regime]
    if regime_vec.size == 0:
        return 4
    return int(np.argmax(regime_vec))


def _calibrate_sttp_profile(
    df: pd.DataFrame,
    max_hold: int,
    tp_quantile: float = 0.70,
    sl_quantile: float = 0.80,
    max_samples: int = 25000,
) -> STTPProfile:
    """Calibrate regime-specific MAE/MFE quantiles from training history.

    For each bar, we look ahead up to `max_hold` bars and estimate:
      - long TP: favorable upside excursion quantile
      - long SL: adverse downside excursion quantile
      - short TP/SL are mirrored counterparts

    This is intentionally sampled to keep startup cost bounded.
    """
    if "timestamp" not in df.columns:
        return _default_sttp_profile()

    req_cols = {"close", "high", "low"} | set(REGIME_COLS)
    if not req_cols.issubset(set(df.columns)):
        return _default_sttp_profile()

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=np.float64)
    reg = df[REGIME_COLS].fillna(0.0).to_numpy(dtype=np.float64)
    reg_idx = np.argmax(np.maximum(reg, 0.0), axis=1).astype(np.int64)

    n = len(df)
    horizon = max(8, int(max_hold))
    valid_end = max(0, n - horizon - 1)
    if valid_end <= 0:
        return _default_sttp_profile()

    sample_step = max(1, valid_end // max(1, min(max_samples, valid_end)))
    sample_idx = np.arange(0, valid_end, sample_step, dtype=np.int64)

    long_tp_bins = [[] for _ in range(len(REGIME_COLS))]
    long_sl_bins = [[] for _ in range(len(REGIME_COLS))]
    short_tp_bins = [[] for _ in range(len(REGIME_COLS))]
    short_sl_bins = [[] for _ in range(len(REGIME_COLS))]

    for i in sample_idx:
        c0 = float(close[i])
        if not np.isfinite(c0) or c0 <= 0:
            continue
        fut_h = high[i + 1:i + horizon + 1]
        fut_l = low[i + 1:i + horizon + 1]
        if fut_h.size == 0 or fut_l.size == 0:
            continue
        fut_high = float(np.nanmax(fut_h))
        fut_low = float(np.nanmin(fut_l))
        if not np.isfinite(fut_high) or not np.isfinite(fut_low):
            continue

        up_move = max(0.0, (fut_high - c0) / c0)
        dn_move = max(0.0, (c0 - fut_low) / c0)
        ridx = int(np.clip(reg_idx[i], 0, len(REGIME_COLS) - 1))

        # Long uses upside as TP and downside as SL.
        long_tp_bins[ridx].append(up_move)
        long_sl_bins[ridx].append(dn_move)
        # Short is mirrored.
        short_tp_bins[ridx].append(dn_move)
        short_sl_bins[ridx].append(up_move)

    # Candidate 1: regime-conditioned multiplier adjustments.
    tp_scale = np.array([0.90, 0.90, 1.15, 1.05, 1.00], dtype=np.float32)
    sl_scale = np.array([0.85, 0.85, 1.00, 0.95, 1.00], dtype=np.float32)
    hold_scale = np.array([0.65, 0.70, 1.25, 1.10, 1.00], dtype=np.float32)

    def _q(arr: list[float], fallback: float) -> float:
        if not arr:
            return float(fallback)
        x = np.asarray(arr, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float(fallback)
        return float(np.quantile(x, tp_quantile))

    def _q_sl(arr: list[float], fallback: float) -> float:
        if not arr:
            return float(fallback)
        x = np.asarray(arr, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float(fallback)
        return float(np.quantile(x, sl_quantile))

    base_tp = np.array([0.0040, 0.0040, 0.0060, 0.0055, 0.0050], dtype=np.float32)
    base_sl = np.array([0.0030, 0.0030, 0.0045, 0.0040, 0.0035], dtype=np.float32)

    tp_long = np.zeros(len(REGIME_COLS), dtype=np.float32)
    sl_long = np.zeros(len(REGIME_COLS), dtype=np.float32)
    tp_short = np.zeros(len(REGIME_COLS), dtype=np.float32)
    sl_short = np.zeros(len(REGIME_COLS), dtype=np.float32)

    for r in range(len(REGIME_COLS)):
        tp_long[r] = max(base_tp[r], _q(long_tp_bins[r], base_tp[r])) * tp_scale[r]
        sl_long[r] = max(base_sl[r], _q_sl(long_sl_bins[r], base_sl[r])) * sl_scale[r]
        tp_short[r] = max(base_tp[r], _q(short_tp_bins[r], base_tp[r])) * tp_scale[r]
        sl_short[r] = max(base_sl[r], _q_sl(short_sl_bins[r], base_sl[r])) * sl_scale[r]

    return STTPProfile(
        tp_long=tp_long,
        sl_long=sl_long,
        tp_short=tp_short,
        sl_short=sl_short,
        hold_scale=hold_scale,
    )


class DSACSTTPCompactTradingEnv(DSACCompactTradingEnv):
    """DSAC compact env with intrabar ST/TP exits."""

    def __init__(
        self,
        df,
        initial_balance: float = 10000.0,
        fee: float = 0.0005,
        slip: float = 0.0002,
        phase: str = "train",
        hmm_detector=None,
        mtf_features=None,
        sttp_cfg: STTPConfig | None = None,
        sttp_profile: STTPProfile | None = None,
        enable_sttp: bool = True,
    ):
        self.sttp_cfg = sttp_cfg or STTPConfig()
        self.sttp_profile = sttp_profile or _default_sttp_profile()
        self.enable_sttp = bool(enable_sttp)
        self._sttp_exit_reason = None
        self._sttp_vol_np = None
        self._high_np = None
        self._low_np = None
        super().__init__(
            df=df,
            initial_balance=initial_balance,
            fee=fee,
            slip=slip,
            phase=phase,
            hmm_detector=hmm_detector,
            mtf_features=mtf_features,
        )
        self._high_np = (
            self.df["high"].values.astype(np.float32)
            if "high" in self.df.columns
            else self._close_np.copy()
        )
        self._low_np = (
            self.df["low"].values.astype(np.float32)
            if "low" in self.df.columns
            else self._close_np.copy()
        )
        self._sttp_vol_np = _vol_proxy(self._close_np.astype(np.float64), span=self.sttp_cfg.vol_span)

    def reset(self, start_idx=None):
        state = super().reset(start_idx=start_idx)
        self._sttp_exit_reason = None
        return state

    def _sttp_thresholds(self, idx: int, side: str) -> tuple[float, float]:
        # Direction- and regime-aware thresholds derived from empirical MAE/MFE quantiles.
        ridx = self._regime_index_at(idx)
        vv = max(float(self._sttp_vol_np[min(max(idx, 0), len(self._sttp_vol_np) - 1)]), 1e-8)
        if side == "SHORT":
            base_tp = float(self.sttp_profile.tp_short[ridx])
            base_sl = float(self.sttp_profile.sl_short[ridx])
        else:
            base_tp = float(self.sttp_profile.tp_long[ridx])
            base_sl = float(self.sttp_profile.sl_long[ridx])
        tp_pct = max(float(self.sttp_cfg.tp_floor), base_tp * (1.0 + 0.10 * vv))
        sl_pct = max(float(self.sttp_cfg.sl_floor), base_sl * (1.0 + 0.05 * vv))
        return tp_pct, sl_pct

    def _regime_index_at(self, idx: int) -> int:
        if idx < 0 or idx >= len(self._feat_np):
            return 4
        row = self._feat_np[idx]
        return _regime_index_from_row(row, self._n_pred, self._n_conf, self._n_elite, self._n_alpha, self._n_regime)

    def _build_spread_proxy(self) -> np.ndarray:
        """Safe spread proxy.

        Parent implementation uses fillna(ndarray) on some pandas versions,
        which raises TypeError. This override keeps the same intent without
        relying on ndarray fillna semantics.
        """
        spread_cols = [
            "current_spread",
            "bid_ask_spread",
            "spread",
            "orderbook_spread",
            "rel_spread",
            "ask_bid_spread",
        ]
        for c in spread_cols:
            if c in self.df.columns:
                s = pd.to_numeric(self.df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).abs()
                return np.clip(s.to_numpy(dtype=np.float32), 0.0, 0.05).astype(np.float32)

        if "high" in self.df.columns and "low" in self.df.columns:
            high = pd.to_numeric(self.df["high"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            low = pd.to_numeric(self.df["low"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            close = pd.Series(self._close_np.astype(np.float64))
            high = high.fillna(close)
            low = low.fillna(close)
            close_np = np.maximum(self._close_np.astype(np.float64), 1e-8)
            out = np.abs((high.to_numpy(dtype=np.float64) - low.to_numpy(dtype=np.float64)) / close_np)
            return np.clip(out, 0.0, 0.05).astype(np.float32)

        proxy = np.abs(self._close_np.astype(np.float64))
        if len(proxy) > 1:
            lr = np.zeros_like(proxy, dtype=np.float64)
            lr[1:] = np.diff(np.log(np.maximum(proxy, 1e-8)))
            proxy = np.abs(lr) * 0.25 + 2e-4
        return np.clip(proxy, 0.0, 0.02).astype(np.float32)

    def _apply_sttp_exit(self) -> bool:
        if (not self.enable_sttp) or self.pos is None or self.entry_price <= 0 or self.current_leverage <= 0:
            return False

        idx = min(max(self.current_step, 0), len(self._close_np) - 1)
        high = float(self._high_np[idx])
        low = float(self._low_np[idx])
        cp = float(self._close_np[idx])
        ridx = self._regime_index_at(idx)
        tp_pct, sl_pct = self._sttp_thresholds(idx, self.pos)
        hold_limit = int(max(1, round(float(self.sttp_cfg.max_hold) * float(self.sttp_profile.hold_scale[ridx]))))

        hit_tp = False
        hit_sl = False
        tp_level = sl_level = 0.0

        if self.pos == "LONG":
            tp_level = self.entry_price * (1.0 + tp_pct)
            sl_level = self.entry_price * (1.0 - sl_pct)
            hit_tp = high >= tp_level
            hit_sl = low <= sl_level
        else:
            tp_level = self.entry_price * (1.0 - tp_pct)
            sl_level = self.entry_price * (1.0 + sl_pct)
            hit_tp = low <= tp_level
            hit_sl = high >= sl_level

        exit_reason = None
        exit_price = cp
        if hit_tp and hit_sl:
            exit_reason = "sl"
            exit_price = sl_level
        elif hit_sl:
            exit_reason = "sl"
            exit_price = sl_level
        elif hit_tp:
            exit_reason = "tp"
            exit_price = tp_level
        elif self.hold_count >= hold_limit:
            exit_reason = "timeout"
            exit_price = cp

        if exit_reason is None:
            return False

        base_balance = self.balance
        if self.pos == "LONG":
            realized_pnl = (exit_price * (1.0 - self.slip) - self.entry_price) / self.entry_price
        else:
            realized_pnl = (self.entry_price - exit_price * (1.0 + self.slip)) / self.entry_price
        realized_pnl *= self.current_leverage

        self.balance = base_balance * (1.0 + realized_pnl)
        self.balance -= base_balance * self.fee * self.current_leverage
        self.total_trades += 1
        if realized_pnl > 0:
            self.win_trades += 1

        self._just_closed = True
        self._last_realized_pnl = float(realized_pnl)
        self._was_force_closed = False
        self._sttp_exit_reason = exit_reason
        self.pos = None
        self.current_leverage = 0.0
        self.hold_count = 0
        self.unrealized_pnl = 0.0
        self.peak_pnl = 0.0
        self.max_drawdown = 0.0
        return True

    def step(self, action: float):
        action = float(np.clip(action, -1.0, 1.0))
        current_price = self._close_np[self.current_step]
        fill_step = min(self.current_step + 1, len(self._open_np) - 1)
        fill_price = float(self._open_np[fill_step])
        decision_step = self.current_step

        if self.pos is not None:
            prev_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            prev_portfolio_value = self.balance

        # ST/TP is applied before new decision at the bar close.
        self._just_closed = False
        self._last_realized_pnl = 0.0
        self._was_force_closed = False
        self._sttp_exit_reason = None
        sttp_closed = self._apply_sttp_exit()

        abs_action = abs(action)
        leverage_rate = abs_action
        force_close = False
        if self.pos is not None and self.unrealized_pnl <= -0.025:
            force_close = True

        is_entering_long = False
        is_entering_short = False
        is_closing = False
        is_adjusting = False

        if sttp_closed:
            # Do not re-open on the same bar after an ST/TP exit.
            pass
        elif force_close:
            is_closing = True
        elif self.pos is None:
            if action > 0.15:
                is_entering_long = True
            elif action < -0.15:
                is_entering_short = True
        else:
            if abs_action < 0.05:
                is_closing = True
            elif (self.pos == "LONG" and action < -0.15):
                is_closing = True
            elif (self.pos == "SHORT" and action > 0.15):
                is_closing = True
            else:
                is_adjusting = True

        self._was_force_closed = force_close

        if is_entering_long:
            self.pos = "LONG"
            self.entry_price = fill_price * (1.0 + self.slip)
            self.entry_idx = fill_step
            self.current_leverage = leverage_rate
            self.balance -= self.balance * self.fee * self.current_leverage
        elif is_entering_short:
            self.pos = "SHORT"
            self.entry_price = fill_price * (1.0 - self.slip)
            self.entry_idx = fill_step
            self.current_leverage = leverage_rate
            self.balance -= self.balance * self.fee * self.current_leverage
        elif is_adjusting and self.pos is not None:
            old_lev = self.current_leverage
            new_lev = leverage_rate
            lev_delta = abs(new_lev - old_lev)
            if lev_delta > 0.05:
                self.balance -= self.balance * self.fee * lev_delta
                self.current_leverage = new_lev
        elif is_closing and self.pos is not None:
            base_balance = self.balance
            if self.pos == "LONG":
                realized_pnl = (fill_price * (1.0 - self.slip) - self.entry_price) / self.entry_price
            else:
                realized_pnl = (self.entry_price - fill_price * (1.0 + self.slip)) / self.entry_price
            realized_pnl *= self.current_leverage
            self.balance = base_balance * (1.0 + realized_pnl)
            self.balance -= base_balance * self.fee * self.current_leverage
            self.total_trades += 1
            if realized_pnl > 0:
                self.win_trades += 1
            self._just_closed = True
            self._last_realized_pnl = realized_pnl
            self.pos = None
            self.current_leverage = 0.0
            self.hold_count = 0
            self.unrealized_pnl = 0.0
            self.peak_pnl = 0.0
            self.max_drawdown = 0.0

        self.current_step += 1
        done = self.current_step >= self.end_step
        next_price = self._close_np[min(self.current_step, len(self._close_np) - 1)]

        if self.pos is not None:
            self.hold_count = self.current_step - self.entry_idx
            if self.pos == "LONG":
                raw_pnl = (next_price * (1.0 - self.slip) - self.entry_price) / self.entry_price
            else:
                raw_pnl = (self.entry_price - next_price * (1.0 + self.slip)) / self.entry_price
            self.unrealized_pnl = raw_pnl * self.current_leverage
            self.max_drawdown = min(self.max_drawdown, self.unrealized_pnl)
            self.peak_pnl = max(self.peak_pnl, self.unrealized_pnl)

        if self.pos is not None:
            current_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            current_portfolio_value = self.balance

        step_delta = (current_portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8) * 50.0
        r1_pnl = float(np.tanh(step_delta))

        r2_drawdown = 0.0
        if self.pos is not None and self.unrealized_pnl < -0.01:
            dd_ratio = abs(self.unrealized_pnl) / 0.025
            r2_drawdown = -0.1 * (dd_ratio ** 2)

        r3_quality = 0.0
        if self._just_closed:
            if self._sttp_exit_reason == "sl":
                r3_quality = -0.20
            elif self._sttp_exit_reason == "tp":
                r3_quality = 0.20 + 0.15 * min(max(self._last_realized_pnl, 0.0) / 0.01, 1.0)
            elif self._sttp_exit_reason == "timeout":
                r3_quality = -0.03
            elif self._was_force_closed:
                r3_quality = -0.30
            elif self._last_realized_pnl > 0:
                r3_quality = 0.15 * min(self._last_realized_pnl / 0.01, 1.0)
            else:
                r3_quality = -0.05

        r4_time_decay = 0.0
        if self.pos is not None and self.hold_count > 12:
            r4_time_decay = -0.003 * (self.hold_count - 12) / 72.0

        r5_idle = 0.0
        if self.pos is None:
            regime_step = min(max(decision_step, 0), len(self._feat_np) - 1)
            regime_raw = self._feat_np[regime_step]
            o = self._n_pred + self._n_conf + self._n_elite + self._n_alpha
            regime_vec = regime_raw[o:o + self._n_regime]
            regime_idx = int(np.argmax(regime_vec))
            if regime_idx in (2, 3):
                r5_idle = -0.003
            elif regime_idx in (0, 1):
                r5_idle = -0.0003
            else:
                r5_idle = -0.001

        r6_trade_cost = 0.0
        if is_entering_long or is_entering_short:
            r6_trade_cost = -0.01 * leverage_rate

        raw_reward = r1_pnl + r2_drawdown + r3_quality + r4_time_decay + r5_idle + r6_trade_cost
        reward = float(np.tanh(raw_reward))

        if done and self.pos is not None:
            base_balance = self.balance
            ep_end_price = self._close_np[min(self.current_step, len(self._close_np) - 1)]
            if self.pos == "LONG":
                ep_realized = (ep_end_price * (1.0 - self.slip) - self.entry_price) / self.entry_price
            else:
                ep_realized = (self.entry_price - ep_end_price * (1.0 + self.slip)) / self.entry_price
            ep_realized *= self.current_leverage
            self.balance = base_balance * (1.0 + ep_realized)
            self.balance -= base_balance * self.fee * self.current_leverage
            self.total_trades += 1
            if ep_realized > 0:
                self.win_trades += 1
            terminal_r = float(np.tanh(ep_realized * 50.0))
            if ep_realized > 0:
                terminal_r += 0.15 * min(ep_realized / 0.01, 1.0)
            else:
                terminal_r -= 0.05
            reward = float(np.tanh(raw_reward + terminal_r))
            self.pos = None

        info = {
            "pnl_pct": (self.balance / self.initial_balance - 1) * 100,
            "wr": self.win_trades / max(1, self.total_trades),
            "sttp_exit_reason": self._sttp_exit_reason,
            "sttp_enabled": self.enable_sttp,
        }
        return self._get_stacked_state(self._build_state(self.current_step)), reward, done, info


SACRouter = DSACRouter


def train(
    csv_path: str = "data/rl_training_data_full.csv",
    feature_csv: str = "/home/llewyn/crypto-scalping/data/training_features_5m.csv",
    train_ratio: float = 0.8,
    episodes: int = 1000,
    fresh_start: bool = False,
    use_lr_scheduler: bool = True,
    lr_factor: float = 0.5,
    lr_patience: int = 3,
    lr_min: float = 1e-5,
    early_stop_patience: int = 12,
    val_interval: int = 10,
    sttp_cfg: STTPConfig | None = None,
    sttp_tp_quantile: float = 0.70,
    sttp_sl_quantile: float = 0.80,
    sttp_max_samples: int = 25000,
):
    if not os.path.exists(csv_path):
        logger.error("데이터가 없습니다. --mode generate_csv 실행 요망")
        return

    sttp_cfg = sttp_cfg or STTPConfig()
    df = _load_merged_training_frame(csv_path, feature_csv)
    logger.info("[DATA] csv_path=%s | feature_csv=%s | rows=%d", csv_path, feature_csv, len(df))
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.notna().any():
            years = sorted(pd.Series(ts.dt.year.dropna().unique()).astype(int).tolist())
            logger.info("[DATA] ts_range=%s -> %s | years=%s", ts.min(), ts.max(), years)
    split_idx = int(len(df) * float(train_ratio))
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val = df.iloc[split_idx:].reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    logger.info("DSAC compact state dim: %d", DSAC_STATE_DIM)
    logger.info(
        "ST/TP cfg | tp_mult=%.2f sl_mult=%.2f max_hold=%d tp_floor=%.4f sl_floor=%.4f vol_span=%d",
        sttp_cfg.tp_mult,
        sttp_cfg.sl_mult,
        sttp_cfg.max_hold,
        sttp_cfg.tp_floor,
        sttp_cfg.sl_floor,
        sttp_cfg.vol_span,
    )

    logger.info("[HMM] OnlineHMMDetector 초기 학습 시작...")
    hmm_detector = OnlineHMMDetector()
    hmm_detector.fit(df_train, n_iter=30)
    logger.info("[HMM] 초기 학습 완료.")

    logger.info("[MTF] 멀티타임프레임 피처 선계산 중...")
    mtf_train = MultiTimeframeFeatures(df_train["close"].values.astype(np.float32))
    mtf_val = MultiTimeframeFeatures(df_val["close"].values.astype(np.float32))
    logger.info("[MTF] 선계산 완료.")

    logger.info("[STTP] MAE/MFE regime calibration 시작...")
    sttp_profile = _calibrate_sttp_profile(
        df_train,
        max_hold=int(sttp_cfg.max_hold),
        tp_quantile=float(sttp_tp_quantile),
        sl_quantile=float(sttp_sl_quantile),
        max_samples=int(sttp_max_samples),
    )
    logger.info(
        "[STTP] profile tp_long=%s sl_long=%s hold_scale=%s",
        np.round(sttp_profile.tp_long, 6).tolist(),
        np.round(sttp_profile.sl_long, 6).tolist(),
        np.round(sttp_profile.hold_scale, 3).tolist(),
    )

    train_hmm = copy.deepcopy(hmm_detector)
    env = DSACSTTPCompactTradingEnv(
        df_train,
        phase="train",
        hmm_detector=train_hmm,
        mtf_features=mtf_train,
        sttp_cfg=sttp_cfg,
        sttp_profile=sttp_profile,
        enable_sttp=True,
    )
    agent = DSACAgent(DSAC_STATE_DIM, hidden_dim=256, n_quantiles=32, cvar_frac=0.25, device=device)

    nep = int(episodes)
    batch = 256
    update_freq = 4
    min_buffer = 4096
    warmup_steps = 10000
    global_step = 0

    best_val_score = -float("inf")
    best_val_pnl = -float("inf")
    bad_val_count = 0

    actor_scheduler = None
    critic_scheduler = None
    if use_lr_scheduler:
        actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            agent.actor_optimizer,
            mode="max",
            factor=float(lr_factor),
            patience=max(1, int(lr_patience)),
            min_lr=float(lr_min),
            threshold=1e-3,
            threshold_mode="rel",
        )
        critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            agent.critic_optimizer,
            mode="max",
            factor=float(lr_factor),
            patience=max(1, int(lr_patience)),
            min_lr=float(lr_min),
            threshold=1e-3,
            threshold_mode="rel",
        )
    logger.info(
        "[TRAIN CFG] val_interval=%d | lr_sched=%s (factor=%.3f patience=%d min_lr=%.1e) | early_stop_patience=%d",
        int(val_interval),
        "ON" if use_lr_scheduler else "OFF",
        float(lr_factor),
        int(lr_patience),
        float(lr_min),
        int(early_stop_patience),
    )

    os.makedirs("data/ensemble/ckpt", exist_ok=True)
    ckpt_path = "data/ensemble/ckpt/dsac_sttp_checkpoint.pth"
    best_path = "data/ensemble/ckpt/best_dsac_sttp_agents.pth"

    start_ep = 1
    if (not fresh_start) and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            agent.actor.load_state_dict(ckpt["actor"])
            agent.critic.load_state_dict(ckpt["critic"])
            agent.critic_target.load_state_dict(ckpt["critic_target"])
            agent.log_alpha.data.copy_(ckpt["log_alpha"])
            agent.actor_optimizer.load_state_dict(ckpt["actor_opt"])
            agent.critic_optimizer.load_state_dict(ckpt["critic_opt"])
            agent.alpha_optimizer.load_state_dict(ckpt["alpha_opt"])
            global_step = int(ckpt.get("global_step", 0))
            best_val_pnl = float(ckpt.get("best_val_pnl", -float("inf")))
            best_val_score = float(ckpt.get("best_val_score", -float("inf")))
            bad_val_count = int(ckpt.get("bad_val_count", 0))
            start_ep = int(ckpt.get("epoch", 0)) + 1
            logger.info("♻️ [복원] ep=%d | global_step=%d | best_pnl=%.2f%%", start_ep - 1, global_step, best_val_pnl)
            if use_lr_scheduler and actor_scheduler is not None and critic_scheduler is not None:
                try:
                    if "actor_sched" in ckpt:
                        actor_scheduler.load_state_dict(ckpt["actor_sched"])
                    if "critic_sched" in ckpt:
                        critic_scheduler.load_state_dict(ckpt["critic_sched"])
                except Exception as e:
                    logger.warning("⚠️ LR scheduler 상태 복원 실패: %s", e)

            if len(agent.memory) < min_buffer:
                refill_steps = max(warmup_steps, min_buffer)
                logger.info("    [WARMUP 재실행] 버퍼 비어있음 -> %d 스텝 랜덤 탐험으로 리필", refill_steps)
                warmup_env = DSACSTTPCompactTradingEnv(
                    df_train,
                    phase="train",
                    hmm_detector=copy.deepcopy(hmm_detector),
                    mtf_features=mtf_train,
                    sttp_cfg=sttp_cfg,
                    sttp_profile=sttp_profile,
                    enable_sttp=True,
                )
                ws = warmup_env.reset()
                for _ in range(refill_steps):
                    wa = np.random.uniform(-1.0, 1.0)
                    wns, wr, wd, _ = warmup_env.step(wa)
                    agent.memory.push(ws, wa, wr, wns, wd)
                    ws = wns
                    if wd:
                        ws = warmup_env.reset()
                logger.info("    [WARMUP 완료] 버퍼: %d", len(agent.memory))
        except Exception as e:
            logger.warning("⚠️ 체크포인트 복원 실패(아키텍처 변경 가능): %s", e)
    elif fresh_start:
        logger.info("🧹 [FRESH START] 체크포인트 복원을 건너뜁니다.")

    def _save_checkpoint(ep: int):
        actor_sched_state = actor_scheduler.state_dict() if actor_scheduler is not None else None
        critic_sched_state = critic_scheduler.state_dict() if critic_scheduler is not None else None
        torch.save(
            {
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "critic_target": agent.critic_target.state_dict(),
                "log_alpha": agent.log_alpha.data,
                "actor_opt": agent.actor_optimizer.state_dict(),
                "critic_opt": agent.critic_optimizer.state_dict(),
                "alpha_opt": agent.alpha_optimizer.state_dict(),
                "global_step": global_step,
                "best_val_pnl": best_val_pnl,
                "best_val_score": best_val_score,
                "bad_val_count": bad_val_count,
                "epoch": ep,
                "state_dim": DSAC_STATE_DIM,
                "sttp_cfg": {
                    "tp_mult": sttp_cfg.tp_mult,
                    "sl_mult": sttp_cfg.sl_mult,
                    "max_hold": sttp_cfg.max_hold,
                    "tp_floor": sttp_cfg.tp_floor,
                    "sl_floor": sttp_cfg.sl_floor,
                    "vol_span": sttp_cfg.vol_span,
                },
                "sttp_profile": {
                    "tp_long": sttp_profile.tp_long.tolist(),
                    "sl_long": sttp_profile.sl_long.tolist(),
                    "tp_short": sttp_profile.tp_short.tolist(),
                    "sl_short": sttp_profile.sl_short.tolist(),
                    "hold_scale": sttp_profile.hold_scale.tolist(),
                },
                "actor_sched": actor_sched_state,
                "critic_sched": critic_sched_state,
            },
            ckpt_path,
        )

    ep = start_ep
    try:
        for ep in range(start_ep, nep + 1):
            state = env.reset()
            ep_reward = 0.0
            done = False
            last_stats = {}

            while not done:
                global_step += 1

                if global_step < warmup_steps:
                    action = np.random.uniform(-1.0, 1.0)
                else:
                    action = agent.act(state, deterministic=False)

                next_state, reward, done, _ = env.step(action)
                agent.memory.push(state, action, reward, next_state, done)
                ep_reward += reward
                state = next_state

                if global_step % update_freq == 0 and len(agent.memory) >= min_buffer:
                    last_stats = agent.update(batch)

            pnl = (env.balance / env.initial_balance - 1.0) * 100.0
            _cvar = float(last_stats.get("cvar_q", 0.0))
            logger.info(
                "Ep %04d | PnL:%6.1f%% Tr:%4d WR:%4.0f%% Rew:%7.3f | buf:%6d | α:%.4f | CVaR_Q:%+.4f",
                ep,
                pnl,
                env.total_trades,
                env.win_rate * 100,
                ep_reward,
                len(agent.memory),
                agent.alpha,
                _cvar,
            )

            if ep % max(1, int(val_interval)) == 0:
                val_hmm = copy.deepcopy(hmm_detector)
                val_env = DSACSTTPCompactTradingEnv(
                    df_val,
                    phase="val",
                    hmm_detector=val_hmm,
                    mtf_features=mtf_val,
                    sttp_cfg=sttp_cfg,
                    sttp_profile=sttp_profile,
                    enable_sttp=True,
                )

                val_state = val_env.reset()
                val_done = False
                agent.actor.eval()
                while not val_done:
                    with torch.no_grad():
                        val_action = agent.act(val_state, deterministic=True)
                    val_state, _, val_done, _ = val_env.step(val_action)
                agent.actor.train()

                val_pnl = (val_env.balance / val_env.initial_balance - 1.0) * 100.0
                val_wr = val_env.win_rate
                if val_env.total_trades == 0:
                    val_trade_score = -5.0
                elif val_pnl > 0:
                    val_trade_score = min(val_env.total_trades / 30.0, 1.0) * 5.0
                else:
                    val_trade_score = -min(val_env.total_trades / 30.0, 1.0) * 10.0
                val_score = val_pnl * 5.0 + val_wr * 20.0 + val_trade_score

                logger.info(
                    "    [VAL] PnL:%6.2f%% | Tr:%4d | WR:%.0f%% | Score:%.2f",
                    val_pnl,
                    val_env.total_trades,
                    val_wr * 100,
                    val_score,
                )

                improved = val_score > best_val_score
                if improved:
                    best_val_score, best_val_pnl = val_score, val_pnl
                    bad_val_count = 0
                    torch.save(
                        {
                            "actor": agent.actor.state_dict(),
                            "critic": agent.critic.state_dict(),
                            "best_pnl": best_val_pnl,
                            "best_score": best_val_score,
                            "epoch": ep,
                            "state_dim": DSAC_STATE_DIM,
                            "sttp_cfg": {
                                "tp_mult": sttp_cfg.tp_mult,
                                "sl_mult": sttp_cfg.sl_mult,
                                "max_hold": sttp_cfg.max_hold,
                                "tp_floor": sttp_cfg.tp_floor,
                                "sl_floor": sttp_cfg.sl_floor,
                                "vol_span": sttp_cfg.vol_span,
                            },
                            "sttp_profile": {
                                "tp_long": sttp_profile.tp_long.tolist(),
                                "sl_long": sttp_profile.sl_long.tolist(),
                                "tp_short": sttp_profile.tp_short.tolist(),
                                "sl_short": sttp_profile.sl_short.tolist(),
                                "hold_scale": sttp_profile.hold_scale.tolist(),
                            },
                            "meta": {
                                "algo": "DSAC_STTP",
                                "n_quantiles": agent.n_quantiles,
                                "cvar_frac": agent.cvar_frac,
                            },
                        },
                        best_path,
                    )
                    logger.info("    🎉 [NEW BEST] 저장 완료 (PnL:%.2f%%)", best_val_pnl)
                else:
                    bad_val_count += 1

                if use_lr_scheduler and actor_scheduler is not None and critic_scheduler is not None:
                    prev_actor_lr = float(agent.actor_optimizer.param_groups[0]["lr"])
                    prev_critic_lr = float(agent.critic_optimizer.param_groups[0]["lr"])
                    actor_scheduler.step(val_score)
                    critic_scheduler.step(val_score)
                    new_actor_lr = float(agent.actor_optimizer.param_groups[0]["lr"])
                    new_critic_lr = float(agent.critic_optimizer.param_groups[0]["lr"])
                    if (new_actor_lr < prev_actor_lr) or (new_critic_lr < prev_critic_lr):
                        logger.info(
                            "    📉 [LR DROP] actor %.3e -> %.3e | critic %.3e -> %.3e",
                            prev_actor_lr,
                            new_actor_lr,
                            prev_critic_lr,
                            new_critic_lr,
                        )
                    else:
                        logger.info(
                            "    [LR] actor %.3e | critic %.3e | bad_val=%d",
                            new_actor_lr,
                            new_critic_lr,
                            bad_val_count,
                        )

                if ep % 50 == 0:
                    hmm_detector.update_online(n_iter=5)
                    train_hmm.A = hmm_detector.A.copy()
                    train_hmm.mu = hmm_detector.mu.copy()
                    train_hmm.sigma = hmm_detector.sigma.copy()
                    train_hmm.pi = hmm_detector.pi.copy()
                    train_hmm._obs_mean = hmm_detector._obs_mean.copy()
                    train_hmm._obs_std = hmm_detector._obs_std.copy()
                    logger.info("    [HMM] 온라인 업데이트 완료")

                _save_checkpoint(ep)
                if int(early_stop_patience) > 0 and bad_val_count >= int(early_stop_patience):
                    logger.info(
                        "⏹️ [EARLY STOP] bad_val_count=%d >= patience=%d | best_score=%.2f | best_pnl=%.2f%%",
                        bad_val_count,
                        int(early_stop_patience),
                        best_val_score,
                        best_val_pnl,
                    )
                    break

            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logger.info("⚠️ 학습 중단.")
        _save_checkpoint(ep)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DSAC agent with built-in ST/TP exits")
    p.add_argument("--csv-path", default="data/rl_training_data_full.csv")
    p.add_argument("--feature-csv", default="/home/llewyn/crypto-scalping/data/training_features_5m.csv")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--fresh-start", action="store_true", help="Ignore checkpoint and start from scratch")
    p.add_argument("--val-interval", type=int, default=10, help="Run validation every N episodes")
    p.add_argument("--no-lr-scheduler", action="store_true", help="Disable ReduceLROnPlateau schedulers")
    p.add_argument("--lr-factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    p.add_argument("--lr-patience", type=int, default=3, help="ReduceLROnPlateau patience (validation rounds)")
    p.add_argument("--lr-min", type=float, default=1e-5, help="Minimum learning rate for scheduler")
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=12,
        help="Stop if validation does not improve for N validation rounds (<=0 disables)",
    )
    p.add_argument("--sttp-tp-mult", type=float, default=2.5)
    p.add_argument("--sttp-sl-mult", type=float, default=1.2)
    p.add_argument("--sttp-max-hold", type=int, default=72)
    p.add_argument("--sttp-tp-floor", type=float, default=0.0030)
    p.add_argument("--sttp-sl-floor", type=float, default=0.0020)
    p.add_argument("--sttp-vol-span", type=int, default=48)
    p.add_argument("--sttp-tp-quantile", type=float, default=0.70)
    p.add_argument("--sttp-sl-quantile", type=float, default=0.80)
    p.add_argument("--sttp-max-samples", type=int, default=25000)
    p.add_argument("--startup-check-only", action="store_true", help="Validate imports/arguments and exit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.startup_check_only:
        logger.info("startup check ok: train_rl_dsac_sttp_agent")
        raise SystemExit(0)

    cfg = STTPConfig(
        tp_mult=float(args.sttp_tp_mult),
        sl_mult=float(args.sttp_sl_mult),
        max_hold=int(args.sttp_max_hold),
        tp_floor=float(args.sttp_tp_floor),
        sl_floor=float(args.sttp_sl_floor),
        vol_span=int(args.sttp_vol_span),
    )
    train(
        csv_path=args.csv_path,
        feature_csv=args.feature_csv,
        train_ratio=float(args.train_ratio),
        episodes=int(args.episodes),
        fresh_start=bool(args.fresh_start),
        use_lr_scheduler=not bool(args.no_lr_scheduler),
        lr_factor=float(args.lr_factor),
        lr_patience=int(args.lr_patience),
        lr_min=float(args.lr_min),
        early_stop_patience=int(args.early_stop_patience),
        val_interval=int(args.val_interval),
        sttp_cfg=cfg,
        sttp_tp_quantile=float(args.sttp_tp_quantile),
        sttp_sl_quantile=float(args.sttp_sl_quantile),
        sttp_max_samples=int(args.sttp_max_samples),
    )
