"""
DSAC Short Specialist 에이전트
================================================================
숏 방향 전용 DSAC. action ∈ [0, 1]:
  0.0        = 관망 / 청산
  0.0 ~ 0.15 = 데드존 (현상 유지)
  0.15 ~ 1.0 = 숏 진입/유지 (크기 = action)

원본 both-side DSAC에서 발생하던 롱 편향 문제를 구조적으로 분리.
Long specialist(train_rl_dsac_long.py)와 쌍으로 사용.

State: 28D compact (원본 30D에서 entry offset 전체 제거)
  Block A: Market Prediction Meta (17) — entry offset 제거, hour_cos/garch_vol_z 추가
  Block B: Immediate Tick Context (6)
  Block C: Agent Private State (5)
"""

import copy
import gc
import logging
import os
import random
import argparse
import sys
import warnings
from collections import deque
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR, _SCRIPT_DIR, os.path.join(_ROOT_DIR, "ensemble"), os.path.join(_ROOT_DIR, "strategies")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from features.schema import prune_to_feature_keep
from features.registry import find_missing_columns, get_m7_columns, M7_PROB_ALIASES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _resolve_runtime_device(requested: str) -> str:
    req = (requested or "auto").strip().lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req == "cpu":
        return "cpu"
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Check NVIDIA driver/runtime visibility."
            )
        return "cuda"
    raise ValueError(f"invalid device: {requested} (expected: auto/cpu/cuda)")


def _validate_m7_training_columns(df: pd.DataFrame, tag: str = "SHORT") -> None:
    required = get_m7_columns("rl_core", include_entry_price=False)
    missing = find_missing_columns(df.columns, required, aliases=M7_PROB_ALIASES)
    if not missing:
        return
    missing_txt = ", ".join(sorted(missing))
    raise ValueError(
        f"[{tag}] missing required M7 columns: {missing_txt}. "
        "Run scripts/augment_rl_training_with_model7.py before RL training."
    )

from ensemble.train_rl_agent import (  # noqa: E402
    MultiTimeframeFeatures,
    OnlineHMMDetector,
    REGIME_COLS,
    STATE_CONF,
    STATE_PRED,
)
from ensemble.rl_continuous_common import (  # noqa: E402
    ReplayBuffer,
    SACTradingEnv as _BaseSACTradingEnv,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SIDE = "short"
STATE_DIM = 28  # 원본 30D - entry_offset(2) - current_position(1) + hour_cos(1) + garch_vol_z(1)

_POS_THRESH = 0.15
_CLOSE_THRESH = 0.10  # specialist는 청산 임계를 약간 넓혀서 확실한 신호만 유지

LOG_STD_MIN = -20
LOG_STD_MAX = 2

_FALLBACK_SIGNAL_GAIN = float(os.getenv("DSAC_FALLBACK_SIGNAL_GAIN", "8.0"))


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions (원본과 동일)
# ─────────────────────────────────────────────────────────────────────────────
def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def _norm_tanh(x: float, scale: float) -> float:
    s = max(float(scale), 1e-8)
    return float(np.tanh(float(x) / s))


def _pick_first(features: dict[str, Any], keys: list[str], default: float = 0.0) -> float:
    for k in keys:
        if k in features:
            return _safe_float(features.get(k), default)
    return float(default)


def _normalize_prob3(dn: float, fl: float, up: float) -> tuple[float, float, float]:
    p = np.array([max(dn, 0.0), max(fl, 0.0), max(up, 0.0)], dtype=np.float64)
    s = float(p.sum())
    if s <= 1e-12:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    p = p / s
    return float(p[0]), float(p[1]), float(p[2])


def _prob_entropy_norm(dn: float, fl: float, up: float) -> float:
    p = np.array([dn, fl, up], dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / max(float(p.sum()), 1e-12)
    ent = float(-np.sum(p * np.log(p)))
    return float(np.clip(ent / np.log(3.0), 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Short Specialist Environment
# ─────────────────────────────────────────────────────────────────────────────
class ShortSpecialistEnv(_BaseSACTradingEnv):
    """숏 전용 환경.

    action ∈ [0, 1]:
      action > POS_THRESH  → 숏 진입/유지
      action < CLOSE_THRESH → 청산/관망
      그 사이 → 데드존 (현상 유지)

    롱 진입은 절대 불가. reward도 숏 거래에 최적화.
    """

    def __init__(
        self,
        df,
        initial_balance=10000.0,
        fee=0.0005,
        slip=0.0002,
        phase="train",
        hmm_detector=None,
        mtf_features=None,
        **kwargs,
    ):
        self._compact_ready = False
        self._n_rows = len(df)
        super().__init__(
            df=df,
            initial_balance=initial_balance,
            fee=fee,
            slip=slip,
            phase=phase,
            hmm_detector=hmm_detector,
            mtf_features=mtf_features,
            side_mode="both",
        )
        self._n_rows = len(self.df)

        # ── 사전 계산 배열 ──
        close = np.maximum(self._close_np.astype(np.float64), 1e-8)
        log_close = np.log(close)
        self._logret_np = np.zeros(self._n_rows, dtype=np.float32)
        if self._n_rows > 1:
            self._logret_np[1:] = np.diff(log_close).astype(np.float32)

        self._ret3_np = np.zeros(self._n_rows, dtype=np.float32)
        if self._n_rows > 3:
            self._ret3_np[3:] = (log_close[3:] - log_close[:-3]).astype(np.float32)

        lr_s = pd.Series(self._logret_np, dtype="float64")
        self._micro_vol5_np = lr_s.rolling(5, min_periods=1).std(ddof=0).fillna(0.0).to_numpy(dtype=np.float32)
        self._micro_vol10_np = lr_s.rolling(10, min_periods=1).std(ddof=0).fillna(0.0).to_numpy(dtype=np.float32)

        self._spread_np = self._build_spread_proxy()
        sp_s = pd.Series(self._spread_np, dtype="float64")
        sp_mean = sp_s.rolling(32, min_periods=4).mean()
        sp_std = sp_s.rolling(32, min_periods=4).std(ddof=0)
        global_mean = float(np.nanmean(self._spread_np)) if np.isfinite(np.nanmean(self._spread_np)) else 1e-4
        global_std = float(np.nanstd(self._spread_np)) if np.isfinite(np.nanstd(self._spread_np)) else 1e-5
        global_mean = max(global_mean, 1e-8)
        global_std = max(global_std, 1e-6)
        self._spread_mean_np = sp_mean.fillna(global_mean).to_numpy(dtype=np.float32)
        self._spread_std_np = sp_std.fillna(global_std).replace(0.0, global_std).to_numpy(dtype=np.float32)

        self._garch_vol_z_np = self._col_or_default("garch_vol_z", 0.0)
        self._mtf_trend_1h_np = self._col_or_default("mtf_trend_1h", 0.0)
        self._mtf_trend_4h_np = self._col_or_default("mtf_trend_4h", 0.0)
        self._smart_money_flow_np = self._col_or_default("smart_money_flow", 0.0)
        self._taker_acceleration_np = self._col_or_default("taker_acceleration", 0.0)
        self._rogers_satchell_vol_np = self._col_or_default("rogers_satchell_vol", 0.0)
        self._amihud_illiquidity_z_np = self._col_or_default("amihud_illiquidity_z", 0.0)
        self._jump_z_np = self._col_or_default("jump_z", 0.0)
        self._evt_excess_z_np = self._col_or_default("evt_excess_z", 0.0)
        self._jump_flag_np = self._col_or_default("jump_flag", 0.0)
        self._evt_tail_flag_np = self._col_or_default("evt_tail_flag", 0.0)

        # M7 columns
        self._m7_prob_up_np = self._col_or_none("m7_trend_xgb_up")
        self._m7_prob_dn_np = self._col_or_none("m7_trend_xgb_dn")
        self._m7_prob_fl_np = self._col_or_none("m7_trend_xgb_fl")
        self._m7_quality_np = self._col_or_none("m7_quality_pred")
        self._m7_hold_np = self._col_or_none("m7_hold_pred")
        self._m7_q10_np = self._col_or_none("m7_q10")
        self._m7_q50_np = self._col_or_none("m7_q50")
        self._m7_q90_np = self._col_or_none("m7_q90")
        self._m7_qwidth_np = self._col_or_none("m7_qwidth")
        self._m7_gmm_cluster_np = self._col_or_none("m7_gmm_cluster")
        self._m7_gmm_conf_np = self._col_or_none("m7_gmm_conf")
        self._m7_gmm_vol_rank_np = self._col_or_none("m7_gmm_vol_rank")
        self._m7_iso_score_np = self._col_or_none("m7_iso_score")
        self._m7_iso_anom_np = self._col_or_none("m7_iso_anom")
        self._m7_vae_error_np = self._col_or_none("m7_vae_error")
        self._m7_vae_anom_np = self._col_or_none("m7_vae_anom")
        self._m7_entry_short_offset_np = self._col_or_none("m7_entry_short_offset")
        self._m7_tp_offset_np = self._col_or_none("m7_tp_offset")
        self._m7_sl_offset_np = self._col_or_none("m7_sl_offset")

        self._pred_slice = slice(0, self._n_pred)
        self._conf_slice = slice(self._n_pred, self._n_pred + self._n_conf)
        self._regime_slice = slice(
            self._n_pred + self._n_conf + self._n_elite + self._n_alpha,
            self._n_pred + self._n_conf + self._n_elite + self._n_alpha + self._n_regime,
        )
        self._compact_ready = True
        self.reset()

    # ── 헬퍼 ──
    def _col_or_none(self, col: str) -> np.ndarray | None:
        if col not in self.df.columns:
            return None
        arr = pd.to_numeric(self.df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return arr.to_numpy(dtype=np.float32)

    def _col_or_default(self, col: str, default: float = 0.0) -> np.ndarray:
        arr = self._col_or_none(col)
        if arr is None:
            return np.full(self._n_rows, float(default), dtype=np.float32)
        return arr

    def _build_spread_proxy(self) -> np.ndarray:
        spread_cols = ["current_spread", "bid_ask_spread", "spread", "orderbook_spread", "rel_spread", "ask_bid_spread"]
        for c in spread_cols:
            if c in self.df.columns:
                s = pd.to_numeric(self.df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).abs()
                return np.clip(s.to_numpy(dtype=np.float32), 0.0, 0.05)
        if "high" in self.df.columns and "low" in self.df.columns:
            close_s = pd.Series(self._close_np, index=self.df.index)
            high = pd.to_numeric(self.df["high"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(close_s)
            low = pd.to_numeric(self.df["low"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(close_s)
            c = np.maximum(self._close_np.astype(np.float64), 1e-8)
            return np.clip(np.abs((high.to_numpy(np.float64) - low.to_numpy(np.float64)) / c), 0.0, 0.05).astype(np.float32)
        proxy = np.abs(self._logret_np.astype(np.float64)) * 0.25 + 2e-4
        return np.clip(proxy, 0.0, 0.02).astype(np.float32)

    def _arr_at(self, arr: np.ndarray | None, idx: int, default: float = 0.0) -> float:
        if arr is None:
            return float(default)
        if idx < 0 or idx >= len(arr):
            return float(default)
        return _safe_float(arr[idx], default)

    def _fallback_signal_score(self, idx: int) -> float:
        row = self._feat_np[idx]
        preds = row[self._pred_slice]
        confs = row[self._conf_slice]
        if len(preds) == 0:
            return 0.0
        signal = np.nan_to_num(preds * confs, nan=0.0)
        w = np.linspace(1.0, 1.3, len(signal), dtype=np.float32)
        return float(np.dot(signal, w) / max(np.sum(w), 1e-8))

    def _fallback_regime_info(self, idx: int) -> tuple[float, float]:
        row = self._feat_np[idx]
        regime_raw = np.nan_to_num(row[self._regime_slice], nan=0.0)
        if regime_raw.size == 0:
            return 0.0, 0.5
        r_sum = float(np.sum(np.maximum(regime_raw, 0.0)))
        if r_sum <= 1e-12:
            return 0.0, 0.5
        reg = regime_raw / r_sum
        return float(np.argmax(reg)), float(np.max(reg))

    def _get_stacked_state(self, raw_state):
        return np.asarray(raw_state, dtype=np.float32)

    # ── 28D Compact State (숏 전용) ──
    def _build_state(self, idx):
        if not getattr(self, "_compact_ready", False):
            return np.zeros(STATE_DIM, dtype=np.float32)
        if idx < 0 or idx >= self._n_rows:
            return np.zeros(STATE_DIM, dtype=np.float32)

        # Block A: Market Prediction Meta (17) — MTF 추세 추가
        signal_score = self._fallback_signal_score(idx)
        z = float(np.clip(signal_score * _FALLBACK_SIGNAL_GAIN, -12.0, 12.0))

        dn = self._arr_at(self._m7_prob_dn_np, idx, np.nan)
        fl = self._arr_at(self._m7_prob_fl_np, idx, np.nan)
        up = self._arr_at(self._m7_prob_up_np, idx, np.nan)
        if not np.isfinite(dn) or not np.isfinite(fl) or not np.isfinite(up):
            up = _sigmoid(z)
            dn = _sigmoid(-z)
            fl = float(np.exp(-abs(z) * 0.75))
        dn, fl, up = _normalize_prob3(dn, fl, up)
        trend_entropy = _prob_entropy_norm(dn, fl, up)

        quality_raw = self._arr_at(self._m7_quality_np, idx, np.nan)
        if not np.isfinite(quality_raw):
            quality_raw = self._arr_at(self._m7_q50_np, idx, signal_score * 0.0015)
        quality_norm = _norm_tanh(quality_raw, 0.003)

        hold_raw = self._arr_at(self._m7_hold_np, idx, np.nan)
        if not np.isfinite(hold_raw):
            hold_raw = 12.0
        hold_norm = float(np.clip(hold_raw / 48.0, 0.0, 1.0))

        q10 = self._arr_at(self._m7_q10_np, idx, np.nan)
        q50 = self._arr_at(self._m7_q50_np, idx, np.nan)
        q90 = self._arr_at(self._m7_q90_np, idx, np.nan)
        qwidth = self._arr_at(self._m7_qwidth_np, idx, np.nan) if hasattr(self, '_m7_qwidth_np') and self._m7_qwidth_np is not None else np.nan
        if not np.isfinite(q50):
            q50 = quality_raw if np.isfinite(quality_raw) else signal_score * 0.0015
        if not np.isfinite(qwidth):
            if np.isfinite(q10) and np.isfinite(q90):
                qwidth = max(float(q90 - q10), 1e-6)
            else:
                qwidth = max(abs(float(self._garch_vol_z_np[idx])) * 0.002, 5e-4)
        if not np.isfinite(q10):
            q10 = q50 - 0.5 * qwidth
        if not np.isfinite(q90):
            q90 = q50 + 0.5 * qwidth
        qwidth = max(float(qwidth), 1e-6)
        q_mid_norm = _norm_tanh(q50, 0.003)
        q_uncertainty_norm = _norm_tanh(qwidth, 0.010)
        q_skew = float(np.clip(((q90 - q50) - (q50 - q10)) / max(abs(q90 - q10), 1e-6), -1.0, 1.0))

        reg_idx, reg_conf = self._fallback_regime_info(idx)
        gmm_cluster = self._arr_at(self._m7_gmm_cluster_np, idx, reg_idx)
        gmm_conf = self._arr_at(self._m7_gmm_conf_np, idx, reg_conf)
        vol_rank = self._arr_at(
            self._m7_gmm_vol_rank_np, idx,
            np.clip((abs(_safe_float(self._garch_vol_z_np[idx])) - 0.2) / 2.5, 0.0, 1.0),
        )
        gmm_cluster_norm = float(np.clip(gmm_cluster / 4.0, -1.0, 1.0))
        gmm_conf = float(np.clip(gmm_conf, 0.0, 1.0))
        vol_rank = float(np.clip(vol_rank, 0.0, 1.0))

        rs_vol = max(0.0, _safe_float(self._rogers_satchell_vol_np[idx]))
        rs_vol_norm = _norm_tanh(rs_vol, 0.0100)
        amihud_norm = float(np.tanh(_safe_float(self._amihud_illiquidity_z_np[idx]) / 3.0))

        iso_score = self._arr_at(self._m7_iso_score_np, idx, 0.0)
        iso_anom = self._arr_at(self._m7_iso_anom_np, idx, 0.0) >= 0.5
        vae_anom = self._arr_at(self._m7_vae_anom_np, idx, 0.0) >= 0.5
        vae_ratio = 1.25 if vae_anom else 0.0
        shock = (
            abs(_safe_float(self._jump_z_np[idx]))
            + 0.6 * abs(_safe_float(self._evt_excess_z_np[idx]))
            + 0.4 * abs(_safe_float(self._garch_vol_z_np[idx]))
            + 0.8 * (_safe_float(self._jump_flag_np[idx]) > 0.5)
            + 0.8 * (_safe_float(self._evt_tail_flag_np[idx]) > 0.5)
        )
        anomaly_raw = (
            0.55 * max(iso_score, 0.0)
            + 0.40 * max(vae_ratio - 1.0, 0.0)
            + 0.20 * shock
            + 0.15 * max(amihud_norm, 0.0)
            + 0.10 * max(rs_vol_norm, 0.0)
        )
        if iso_anom or vae_anom:
            anomaly_raw += 0.60
        anomaly_score = float(np.clip(np.tanh(anomaly_raw), 0.0, 1.0))

        tp_offset = self._arr_at(self._m7_tp_offset_np, idx, max(q90, 0.0))
        sl_offset = self._arr_at(self._m7_sl_offset_np, idx, max(-q10, 0.0))
        tp_offset_norm = _norm_tanh(tp_offset, 0.0100)
        sl_offset_norm = _norm_tanh(sl_offset, 0.0100)
        mtf_1h_norm = _norm_tanh(_safe_float(self._mtf_trend_1h_np[idx]), 0.0100)
        mtf_4h_norm = _norm_tanh(_safe_float(self._mtf_trend_4h_np[idx]), 0.0200)

        # Block B: Immediate Tick Context (6)
        spread = max(0.0, _safe_float(self._spread_np[idx]))
        spread_mean = max(1e-8, _safe_float(self._spread_mean_np[idx], spread))
        spread_std = max(1e-6, _safe_float(self._spread_std_np[idx], 1e-6))
        spread_z = (spread - spread_mean) / spread_std
        spread_norm = _norm_tanh(spread, 0.0015)
        micro5 = max(0.0, _safe_float(self._micro_vol5_np[idx]))
        micro5_norm = _norm_tanh(micro5, 0.0030)
        smart_flow_norm = _norm_tanh(_safe_float(self._smart_money_flow_np[idx]), 0.0500)
        taker_accel_norm = _norm_tanh(_safe_float(self._taker_acceleration_np[idx]), 0.0500)

        # Block C: Agent Private State (5) — 포지션은 항상 None 또는 SHORT
        in_position = 1.0 if self.pos == "SHORT" else 0.0
        margin_usage = float(np.clip(self.current_leverage if self.pos is not None else 0.0, 0.0, 1.0))
        unrealized_norm = _norm_tanh(self.unrealized_pnl, 0.02)
        time_in_trade_norm = float(np.clip(self.hold_count / 96.0, 0.0, 1.0))
        drawdown_norm = float(np.clip(self.max_drawdown / 0.05, -1.0, 1.0))

        state = np.array(
            [
                # Block A (17)
                up, dn, fl, trend_entropy,
                quality_norm, hold_norm,
                q_mid_norm, q_uncertainty_norm, q_skew,
                gmm_cluster_norm, gmm_conf, vol_rank,
                anomaly_score,
                tp_offset_norm, sl_offset_norm,
                mtf_1h_norm, mtf_4h_norm,
                # Block B (6)
                spread_norm, rs_vol_norm,
                micro5_norm, amihud_norm,
                smart_flow_norm, taker_accel_norm,
                # Block C (5)
                in_position, margin_usage,
                unrealized_norm, time_in_trade_norm,
                drawdown_norm,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

    # ── step() 오버라이드: action [0,1] → 숏만 ──
    def step(self, action: float):
        """action ∈ [0, 1]: 0=관망/청산, 1=풀 숏."""
        action = float(np.clip(action, 0.0, 1.0))

        current_price = self._close_np[self.current_step]
        fill_step = min(self.current_step + 1, len(self._open_np) - 1)
        fill_price = float(self._open_np[fill_step])
        decision_step = self.current_step

        if self.pos is not None:
            prev_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            prev_portfolio_value = self.balance

        leverage_rate = action
        force_close = False
        if self.pos is not None and self.unrealized_pnl <= -0.025:
            force_close = True

        is_entering = False
        is_closing = False
        is_adjusting = False

        if force_close:
            is_closing = True
        elif self.pos is None:
            # 관망 중 → 숏 진입 여부
            if action > _POS_THRESH:
                is_entering = True
        else:
            # 숏 보유 중
            if action < _CLOSE_THRESH:
                is_closing = True
            else:
                is_adjusting = True

        # ── 거래 실행 ──
        self._just_closed = False
        self._last_realized_pnl = 0.0
        self._was_force_closed = force_close

        if is_entering:
            self.pos = "SHORT"
            self.entry_price = fill_price * (1 - self.slip)
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
            realized_pnl = (self.entry_price - fill_price * (1 + self.slip)) / self.entry_price
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

        # ── 스텝 전진 ──
        self.current_step += 1
        done = self.current_step >= self.end_step
        next_price = self._close_np[min(self.current_step, len(self._close_np) - 1)]

        if self.pos is not None:
            self.hold_count = self.current_step - self.entry_idx
            raw_pnl = (self.entry_price - next_price * (1 + self.slip)) / self.entry_price
            self.unrealized_pnl = raw_pnl * self.current_leverage
            self.max_drawdown = min(self.max_drawdown, self.unrealized_pnl)
            self.peak_pnl = max(self.peak_pnl, self.unrealized_pnl)

        # ── Reward ──
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
            if self._was_force_closed:
                r3_quality = -0.30
            elif self._last_realized_pnl > 0:
                # 대칭화: 승리 보상 +0.10
                r3_quality = 0.10 * min(self._last_realized_pnl / 0.01, 1.0)
            else:
                # 대칭화: 패배 페널티 -0.10
                r3_quality = -0.10

        r4_time_decay = 0.0
        if self.pos is not None and self.hold_count > 12:
            r4_time_decay = -0.003 * (self.hold_count - 12) / 72.0

        # r5_idle: 숏 specialist는 관망 페널티를 크게 줄임
        r5_idle = 0.0
        if self.pos is None:
            regime_step = min(max(decision_step, 0), len(self._feat_np) - 1)
            regime_raw = self._feat_np[regime_step]
            o = self._n_pred + self._n_conf + self._n_elite + self._n_alpha
            regime_vec = regime_raw[o : o + self._n_regime]
            regime_idx = int(np.argmax(regime_vec))
            # bull(0)/chop(2)/whipsaw(3) 일 때 관망은 페널티 없음
            if regime_idx == 1:  # bear
                r5_idle = -0.0005  # bear에서만 아주 약한 관망 페널티
            else:
                r5_idle = 0.0  # 나머지 레짐에서 관망은 자유

        r6_trade_cost = 0.0
        if is_entering:
            r6_trade_cost = -0.01 * leverage_rate

        raw_reward = r1_pnl + r2_drawdown + r3_quality + r4_time_decay + r5_idle + r6_trade_cost
        reward = float(np.tanh(raw_reward))

        # 에피소드 종료 시 강제 청산
        if done and self.pos is not None:
            base_balance = self.balance
            ep_fill_step = min(self.current_step, len(self._open_np) - 1)
            ep_end_price = float(self._open_np[ep_fill_step])
            ep_realized = (self.entry_price - ep_end_price * (1 + self.slip)) / self.entry_price
            ep_realized *= self.current_leverage
            self.balance = base_balance * (1.0 + ep_realized)
            self.balance -= base_balance * self.fee * self.current_leverage
            self.total_trades += 1
            if ep_realized > 0:
                self.win_trades += 1
            terminal_r = float(np.tanh(ep_realized * 50.0))
            if ep_realized > 0:
                terminal_r += 0.10 * min(ep_realized / 0.01, 1.0)
            else:
                terminal_r -= 0.10
            reward = float(np.tanh(raw_reward + terminal_r))
            self.pos = None

        info = {
            "pnl_pct": (self.balance / self.initial_balance - 1) * 100,
            "wr": self.win_trades / max(1, self.total_trades),
        }
        return self._get_stacked_state(self._build_state(self.current_step)), reward, done, info


# ─────────────────────────────────────────────────────────────────────────────
# Quantile Huber Loss
# ─────────────────────────────────────────────────────────────────────────────
def _quantile_huber_loss(pred_q, target_q, taus, kappa=1.0):
    td = target_q.unsqueeze(1) - pred_q.unsqueeze(2)
    abs_td = td.abs()
    huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa))
    tau = taus.view(1, -1, 1)
    weight = (tau - (td.detach() < 0).float()).abs()
    return (weight * huber / kappa).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Networks — action [0,1] → sigmoid 출력
# ─────────────────────────────────────────────────────────────────────────────
class CompactFeatureExtractor(nn.Module):
    def __init__(self, state_dim=STATE_DIM, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        )

    def forward(self, state):
        return self.net(state)


class SigmoidActor(nn.Module):
    """state → action ∈ [0, 1] (sigmoid squashed Gaussian)."""

    def __init__(self, state_dim=STATE_DIM, hidden_dim=256):
        super().__init__()
        self.feat = CompactFeatureExtractor(state_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        feat = self.feat(state)
        mu = self.mu_head(feat)
        log_std = self.log_std_head(feat).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def forward_logits(self, state):
        feat = self.feat(state)
        mu = self.mu_head(feat)
        log_std = self.log_std_head(feat).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return feat, mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        action = torch.sigmoid(x_t)  # [0, 1]
        log_prob = dist.log_prob(x_t) - torch.log(action * (1 - action) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state):
        mu, _ = self.forward(state)
        return torch.sigmoid(mu)


class DistributionalTwinCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, hidden_dim=256, n_quantiles=32):
        super().__init__()
        self.n_quantiles = int(n_quantiles)
        self.feat1 = CompactFeatureExtractor(state_dim, hidden_dim)
        self.q1 = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.n_quantiles))
        self.feat2 = CompactFeatureExtractor(state_dim, hidden_dim)
        self.q2 = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.n_quantiles))

    def forward(self, state, action):
        f1 = self.feat1(state)
        f2 = self.feat2(state)
        return self.q1(torch.cat([f1, action], dim=1)), self.q2(torch.cat([f2, action], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# DSAC Short Agent
# ─────────────────────────────────────────────────────────────────────────────
class DSACShortAgent:
    def __init__(
        self, state_dim=STATE_DIM, hidden_dim=256,
        lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
        gamma=0.99, tau=0.005, n_quantiles=32, cvar_frac=0.25, device="cuda",
    ):
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.n_quantiles = int(n_quantiles)
        self.cvar_frac = float(cvar_frac)

        self.actor = SigmoidActor(state_dim, hidden_dim).to(device)
        self.critic = DistributionalTwinCritic(state_dim, hidden_dim, self.n_quantiles).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_target.eval()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.target_entropy = -0.5
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.taus = torch.linspace(
            0.5 / self.n_quantiles, 1.0 - 0.5 / self.n_quantiles,
            self.n_quantiles, device=device, dtype=torch.float32,
        )
        self.memory = ReplayBuffer(capacity=500000)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    def act(self, state: np.ndarray, deterministic: bool = False) -> float:
        state_ts = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(state_ts)
            else:
                action, _ = self.actor.sample(state_ts)
        return float(action.cpu().item())

    def _target_quantiles(self, ns, r, d):
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(ns)
            tq1, tq2 = self.critic_target(ns, next_action)
            tq1_m = tq1.mean(dim=1, keepdim=True)
            tq2_m = tq2.mean(dim=1, keepdim=True)
            chosen_tq = torch.where(tq1_m <= tq2_m, tq1, tq2)
            entropy_term = self.log_alpha.exp().detach() * next_log_prob
            return r + self.gamma * (1.0 - d) * (chosen_tq - entropy_term)

    def _cvar_min(self, q1, q2):
        k = max(1, int(self.n_quantiles * self.cvar_frac))
        q1_s, _ = torch.sort(q1, dim=1)
        q2_s, _ = torch.sort(q2, dim=1)
        c1 = q1_s[:, :k].mean(dim=1, keepdim=True)
        c2 = q2_s[:, :k].mean(dim=1, keepdim=True)
        return torch.min(c1, c2)

    def update(self, batch_size=256) -> dict:
        if len(self.memory) < batch_size:
            return {}

        s, a, r, ns, d = self.memory.sample(batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        target_q = self._target_quantiles(ns, r, d)
        q1, q2 = self.critic(s, a)
        critic_loss = _quantile_huber_loss(q1, target_q, self.taus) + _quantile_huber_loss(q2, target_q, self.taus)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        new_action, log_prob = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, new_action)
        q_cvar = self._cvar_min(q1_new, q2_new)
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * log_prob - q_cvar).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.log_alpha.exp().item()),
            "mean_q": float(torch.min(q1_new.mean(dim=1), q2_new.mean(dim=1)).mean().item()),
            "cvar_q": float(q_cvar.mean().item()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Router (라이브 추론)
# ─────────────────────────────────────────────────────────────────────────────
class DSACShortRouter:
    """숏 specialist 라이브 라우터.

    decide() → (action_int, leverage, info)
      action_int: 0=관망, 2=SHORT
    """

    def __init__(self, actor, device="cuda", hmm_detector=None, mtf_features=None):
        self.actor = actor
        self.device = device
        self.hmm = hmm_detector
        self.mtf = mtf_features
        self._prev_close: float | None = None
        self._ret_hist: deque[float] = deque(maxlen=64)
        self._spread_hist: deque[float] = deque(maxlen=64)

    def _fallback_signal_score(self, features: dict[str, Any]) -> float:
        sig = []
        for p, c in zip(STATE_PRED, STATE_CONF):
            sig.append(_safe_float(features.get(p, 0.0), 0.0) * _safe_float(features.get(c, 0.5), 0.5))
        if not sig:
            return 0.0
        arr = np.asarray(sig, dtype=np.float32)
        w = np.linspace(1.0, 1.3, arr.size, dtype=np.float32)
        return float(np.dot(arr, w) / max(float(np.sum(w)), 1e-8))

    def _fallback_regime(self, features: dict[str, Any]) -> tuple[float, float]:
        reg = np.asarray([_safe_float(features.get(c, 0.0), 0.0) for c in REGIME_COLS], dtype=np.float32)
        reg = np.maximum(reg, 0.0)
        s = float(reg.sum())
        if s <= 1e-12:
            return 0.0, 0.5
        reg = reg / s
        return float(np.argmax(reg)), float(np.max(reg))

    def _build_compact_state(self, features: dict[str, Any], pos: dict[str, Any]) -> np.ndarray:
        """28D 숏 전용 state 구성 (라이브)."""
        signal_score = self._fallback_signal_score(features)
        z = float(np.clip(signal_score * _FALLBACK_SIGNAL_GAIN, -12.0, 12.0))

        dn = _pick_first(features, ["m7_trend_xgb_dn", "trend_dn_prob"], np.nan)
        fl = _pick_first(features, ["m7_trend_xgb_fl"], np.nan)
        up = _pick_first(features, ["m7_trend_xgb_up", "trend_up_prob"], np.nan)
        if not np.isfinite(dn) or not np.isfinite(fl) or not np.isfinite(up):
            up = _sigmoid(z); dn = _sigmoid(-z); fl = float(np.exp(-abs(z) * 0.75))
        dn, fl, up = _normalize_prob3(dn, fl, up)
        trend_entropy = _prob_entropy_norm(dn, fl, up)

        quality_raw = _pick_first(features, ["m7_quality_pred", "expected_quality"], np.nan)
        if not np.isfinite(quality_raw):
            quality_raw = _pick_first(features, ["m7_q50"], signal_score * 0.0015)
        quality_norm = _norm_tanh(quality_raw, 0.003)

        hold_raw = _pick_first(features, ["m7_hold_pred", "expected_hold_time"], np.nan)
        if not np.isfinite(hold_raw):
            hold_raw = 12.0
        hold_norm = float(np.clip(hold_raw / 48.0, 0.0, 1.0))

        q10 = _pick_first(features, ["m7_q10"], np.nan)
        q50 = _pick_first(features, ["m7_q50"], np.nan)
        q90 = _pick_first(features, ["m7_q90"], np.nan)
        qwidth = _pick_first(features, ["m7_qwidth", "quantile_uncertainty"], np.nan)
        if not np.isfinite(q50):
            q50 = quality_raw if np.isfinite(quality_raw) else signal_score * 0.0015
        if not np.isfinite(qwidth):
            if np.isfinite(q10) and np.isfinite(q90):
                qwidth = max(float(q90 - q10), 1e-6)
            else:
                qwidth = max(abs(_safe_float(features.get("garch_vol_z", 0.0))) * 0.002, 5e-4)
        if not np.isfinite(q10):
            q10 = q50 - 0.5 * qwidth
        if not np.isfinite(q90):
            q90 = q50 + 0.5 * qwidth
        qwidth = max(float(qwidth), 1e-6)
        q_mid_norm = _norm_tanh(q50, 0.003)
        q_uncertainty_norm = _norm_tanh(qwidth, 0.010)
        q_skew = float(np.clip(((q90 - q50) - (q50 - q10)) / max(abs(q90 - q10), 1e-6), -1.0, 1.0))

        reg_idx, reg_conf = self._fallback_regime(features)
        gmm_cluster = _pick_first(features, ["m7_gmm_cluster", "gmm_cluster_id"], reg_idx)
        gmm_conf = _pick_first(features, ["m7_gmm_conf"], reg_conf)
        vol_rank = _pick_first(features, ["m7_gmm_vol_rank"],
                               np.clip((abs(_safe_float(features.get("garch_vol_z", 0.0))) - 0.2) / 2.5, 0.0, 1.0))
        gmm_cluster_norm = float(np.clip(gmm_cluster / 4.0, -1.0, 1.0))
        gmm_conf = float(np.clip(gmm_conf, 0.0, 1.0))
        vol_rank = float(np.clip(vol_rank, 0.0, 1.0))

        iso_score = _pick_first(features, ["m7_iso_score"], 0.0)
        iso_anom = _pick_first(features, ["m7_iso_anom"], 0.0) >= 0.5
        vae_anom = _pick_first(features, ["m7_vae_anom"], 0.0) >= 0.5
        vae_ratio = 1.25 if vae_anom else 0.0
        shock = (abs(_safe_float(features.get("jump_z", 0.0)))
                 + 0.6 * abs(_safe_float(features.get("evt_excess_z", 0.0)))
                 + 0.4 * abs(_safe_float(features.get("garch_vol_z", 0.0)))
                 + 0.8 * (_safe_float(features.get("jump_flag", 0.0)) > 0.5)
                 + 0.8 * (_safe_float(features.get("evt_tail_flag", 0.0)) > 0.5))
        anomaly_raw = 0.55 * max(iso_score, 0.0) + 0.40 * max(vae_ratio - 1.0, 0.0) + 0.20 * shock
        if iso_anom or vae_anom:
            anomaly_raw += 0.60
        anomaly_score = float(np.clip(np.tanh(anomaly_raw), 0.0, 1.0))

        tp_offset = _safe_float(features.get("m7_tp_offset", max(q90, 0.0)), max(q90, 0.0))
        sl_offset = _safe_float(features.get("m7_sl_offset", max(-q10, 0.0)), max(-q10, 0.0))
        tp_offset_norm = _norm_tanh(tp_offset, 0.0100)
        sl_offset_norm = _norm_tanh(sl_offset, 0.0100)
        mtf_1h_norm = _norm_tanh(_safe_float(features.get("mtf_trend_1h", 0.0), 0.0), 0.0100)
        mtf_4h_norm = _norm_tanh(_safe_float(features.get("mtf_trend_4h", 0.0), 0.0), 0.0200)
        rs_vol_norm = _norm_tanh(max(0.0, _safe_float(features.get("rogers_satchell_vol", 0.0), 0.0)), 0.0100)
        amihud_norm = float(np.tanh(_safe_float(features.get("amihud_illiquidity_z", 0.0), 0.0) / 3.0))
        smart_flow_norm = _norm_tanh(_safe_float(features.get("smart_money_flow", 0.0), 0.0), 0.0500)
        taker_accel_norm = _norm_tanh(_safe_float(features.get("taker_acceleration", 0.0), 0.0), 0.0500)

        close = max(_safe_float(features.get("close", 0.0), 0.0), 0.0)
        logret = _safe_float(features.get("log_return", np.nan), np.nan)
        if not np.isfinite(logret):
            if self._prev_close is not None and self._prev_close > 0.0 and close > 0.0:
                logret = float(np.log(close / self._prev_close))
            else:
                logret = 0.0
        self._prev_close = close if close > 0.0 else self._prev_close
        self._ret_hist.append(float(logret))

        spread = _pick_first(features, ["current_spread", "bid_ask_spread", "spread", "orderbook_spread", "rel_spread", "ask_bid_spread"], np.nan)
        if not np.isfinite(spread):
            spread = abs(float(logret)) * 0.25 + 2e-4
        spread = float(np.clip(abs(spread), 0.0, 0.05))
        self._spread_hist.append(spread)

        ret_arr = np.asarray(self._ret_hist, dtype=np.float64)
        sp_arr = np.asarray(self._spread_hist, dtype=np.float64)
        micro5 = float(np.std(ret_arr[-5:])) if ret_arr.size > 0 else 0.0
        micro10 = float(np.std(ret_arr[-10:])) if ret_arr.size > 0 else 0.0
        sp_mean = float(np.mean(sp_arr[-32:])) if sp_arr.size > 0 else spread
        sp_std = max(float(np.std(sp_arr[-32:])) if sp_arr.size > 1 else 1e-6, 1e-6)
        spread_z = (spread - sp_mean) / sp_std

        spread_norm = _norm_tanh(spread, 0.0015)
        micro5_norm = _norm_tanh(micro5, 0.0030)

        pos_type = pos.get("type") if isinstance(pos, dict) else None
        in_position = 1.0 if pos_type == "SHORT" else 0.0
        margin_usage = _safe_float(pos.get("margin_usage", np.nan), np.nan) if isinstance(pos, dict) else np.nan
        if not np.isfinite(margin_usage):
            margin_usage = 1.0 if in_position > 0.5 else 0.0
        margin_usage = float(np.clip(margin_usage, 0.0, 1.0))
        unr = _safe_float(pos.get("unrealized", 0.0), 0.0) if isinstance(pos, dict) else 0.0
        unrealized_norm = _norm_tanh(unr, 0.02)

        hold_count_proxy = 0.0
        if isinstance(pos, dict):
            hold_count_proxy = max(0.0, _safe_float(pos.get("hold_count", np.nan), np.nan))
            if not np.isfinite(hold_count_proxy):
                hold_count_proxy = float(np.clip(_safe_float(pos.get("hold_norm", 0.0), 0.0), 0.0, 1.0)) * 96.0
        time_in_trade_norm = float(np.clip(hold_count_proxy / 96.0, 0.0, 1.0))
        mdd = _safe_float(pos.get("mdd", 0.0), 0.0) if isinstance(pos, dict) else 0.0
        drawdown_norm = float(np.clip(mdd / 0.05, -1.0, 1.0))

        state = np.array([
            up, dn, fl, trend_entropy,
            quality_norm, hold_norm, q_mid_norm, q_uncertainty_norm, q_skew,
            gmm_cluster_norm, gmm_conf, vol_rank, anomaly_score,
            tp_offset_norm, sl_offset_norm, mtf_1h_norm, mtf_4h_norm,
            spread_norm, rs_vol_norm, micro5_norm, amihud_norm, smart_flow_norm, taker_accel_norm,
            in_position, margin_usage, unrealized_norm, time_in_trade_norm, drawdown_norm,
        ], dtype=np.float32)
        return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

    def decide(self, features, pos):
        state = self._build_compact_state(features or {}, pos or {})
        state_ts = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            feat, mu, log_std = self.actor.forward_logits(state_ts)
            action = torch.sigmoid(mu)
        action_val = float(action.cpu().item())
        logit_val = float(mu.cpu().item())
        std_val = float(log_std.exp().cpu().item())
        feat_norm = float(torch.norm(feat, p=2).cpu().item())

        cur_pos = pos.get("type") if isinstance(pos, dict) else None
        if cur_pos == "SHORT":
            if action_val < _CLOSE_THRESH:
                action_int, leverage = 0, 0.0
            else:
                action_int, leverage = 2, action_val
        elif cur_pos is None:
            if action_val > _POS_THRESH:
                action_int, leverage = 2, action_val
            else:
                action_int, leverage = 0, 0.0
        else:
            # 롱 보유 중이면 이 라우터는 관여 안 함
            action_int, leverage = 0, 0.0

        return action_int, leverage, {
            "agent": "DSAC_SHORT",
            "raw_action": round(action_val, 4),
            "logit": logit_val,
            "mu": logit_val,
            "std": std_val,
            "feat_norm": feat_norm,
            "kelly": float(leverage),
            "long_edge": 0.0,
            "short_edge": action_val,
            "score": max(action_val, max(logit_val, 0.0)),
            "state_dim": STATE_DIM,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(
    csv_path: str = "data/rl_training_data_full.csv",
    train_ratio: float = 0.8,
    episodes: int = 1000,
    fresh_start: bool = False,
    use_lr_scheduler: bool = True,
    lr_factor: float = 0.5,
    lr_patience: int = 3,
    lr_min: float = 1e-5,
    early_stop_patience: int = 12,
    val_interval: int = 10,
    cvar_frac: float = 0.25,
    device: str = "auto",
):
    if not os.path.exists(csv_path):
        logger.error("데이터가 없습니다. --mode generate_csv 실행 요망")
        return

    df = pd.read_csv(csv_path)
    before_cols = len(df.columns)
    df = prune_to_feature_keep(df, include_entry_price=False, extra_keep=["timestamp"])
    if len(df.columns) != before_cols:
        logger.info("[SHORT] feature prune: %d -> %d cols (active M7+RL only)", before_cols, len(df.columns))
    _validate_m7_training_columns(df, tag="SHORT")
    logger.info("[SHORT] csv_path=%s | rows=%d", csv_path, len(df))
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.notna().any():
            years = sorted(pd.Series(ts.dt.year.dropna().unique()).astype(int).tolist())
            logger.info("[SHORT] ts_range=%s -> %s | years=%s", ts.min(), ts.max(), years)
    split_idx = int(len(df) * float(train_ratio))
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val = df.iloc[split_idx:].reset_index(drop=True)

    device = _resolve_runtime_device(device)
    logger.info("[SHORT] Device: %s | state_dim: %d", device, STATE_DIM)

    hmm_detector = OnlineHMMDetector()
    hmm_detector.fit(df_train, n_iter=30)
    mtf_train = MultiTimeframeFeatures(df_train["close"].values.astype(np.float32))
    mtf_val = MultiTimeframeFeatures(df_val["close"].values.astype(np.float32))

    train_hmm = copy.deepcopy(hmm_detector)
    env = ShortSpecialistEnv(df_train, phase="train", hmm_detector=train_hmm, mtf_features=mtf_train)
    agent = DSACShortAgent(STATE_DIM, hidden_dim=256, n_quantiles=32, cvar_frac=float(cvar_frac), device=device)

    nep = int(episodes)
    batch = 256
    update_freq = 4
    min_buffer = 4096
    warmup_steps = 10000
    global_step = 0
    best_val_score = -float("inf")
    best_val_pnl = -float("inf")
    bad_val_count = 0

    actor_scheduler = critic_scheduler = None
    if use_lr_scheduler:
        actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            agent.actor_optimizer, mode="max", factor=float(lr_factor),
            patience=max(1, int(lr_patience)), min_lr=float(lr_min), threshold=1e-3, threshold_mode="rel",
        )
        critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            agent.critic_optimizer, mode="max", factor=float(lr_factor),
            patience=max(1, int(lr_patience)), min_lr=float(lr_min), threshold=1e-3, threshold_mode="rel",
        )
    logger.info(
        "[SHORT][TRAIN CFG] val_interval=%d | lr_sched=%s (factor=%.3f patience=%d min_lr=%.1e) | early_stop_patience=%d | cvar_frac=%.2f | fallback_gain=%.2f",
        int(val_interval),
        "ON" if use_lr_scheduler else "OFF",
        float(lr_factor),
        int(lr_patience),
        float(lr_min),
        int(early_stop_patience),
        float(cvar_frac),
        float(_FALLBACK_SIGNAL_GAIN),
    )

    os.makedirs("data/ensemble/ckpt", exist_ok=True)
    ckpt_path = "data/ensemble/ckpt/dsac_short_checkpoint.pth"
    best_path = "data/ensemble/ckpt/best_dsac_short_agents.pth"

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
            logger.info(
                "[SHORT] ♻️ 복원 ep=%d | global_step=%d | best_pnl=%.2f%% | best_score=%.2f | bad_val=%d",
                start_ep - 1,
                global_step,
                best_val_pnl,
                best_val_score,
                bad_val_count,
            )
            if use_lr_scheduler:
                try:
                    if "actor_sched" in ckpt and actor_scheduler:
                        actor_scheduler.load_state_dict(ckpt["actor_sched"])
                    if "critic_sched" in ckpt and critic_scheduler:
                        critic_scheduler.load_state_dict(ckpt["critic_sched"])
                except Exception as e:
                    logger.warning("[SHORT] LR scheduler 복원 실패: %s", e)
            if len(agent.memory) < min_buffer:
                refill_steps = max(warmup_steps, min_buffer)
                logger.info("[SHORT] WARMUP 재실행 -> %d 스텝", refill_steps)
                warmup_env = ShortSpecialistEnv(df_train, phase="train", hmm_detector=copy.deepcopy(hmm_detector), mtf_features=mtf_train)
                ws = warmup_env.reset()
                for _ in range(refill_steps):
                    wa = random.random()  # [0, 1]
                    wns, wr, wd, _ = warmup_env.step(wa)
                    agent.memory.push(ws, wa, wr, wns, wd)
                    ws = wns
                    if wd:
                        ws = warmup_env.reset()
                logger.info("[SHORT] WARMUP 완료 | 버퍼: %d", len(agent.memory))
        except Exception as e:
            logger.warning("[SHORT] 체크포인트 복원 실패: %s", e)
    elif fresh_start:
        logger.info("[SHORT] 🧹 FRESH START")

    def _save_checkpoint(ep: int):
        torch.save({
            "actor": agent.actor.state_dict(), "critic": agent.critic.state_dict(),
            "critic_target": agent.critic_target.state_dict(), "log_alpha": agent.log_alpha.data,
            "actor_opt": agent.actor_optimizer.state_dict(), "critic_opt": agent.critic_optimizer.state_dict(),
            "alpha_opt": agent.alpha_optimizer.state_dict(), "global_step": global_step,
            "best_val_pnl": best_val_pnl, "best_val_score": best_val_score,
            "bad_val_count": bad_val_count, "epoch": ep, "state_dim": STATE_DIM,
            "actor_sched": actor_scheduler.state_dict() if actor_scheduler else None,
            "critic_sched": critic_scheduler.state_dict() if critic_scheduler else None,
            "meta": {
                "algo": "DSAC_SHORT",
                "side": SIDE,
                "state_dim": STATE_DIM,
                "n_quantiles": agent.n_quantiles,
                "cvar_frac": agent.cvar_frac,
                "pos_thresh": _POS_THRESH,
                "close_thresh": _CLOSE_THRESH,
                "fallback_signal_gain": _FALLBACK_SIGNAL_GAIN,
                "val_interval": int(val_interval),
                "early_stop_patience": int(early_stop_patience),
                "lr_factor": float(lr_factor),
                "lr_patience": int(lr_patience),
                "lr_min": float(lr_min),
                "use_lr_scheduler": bool(use_lr_scheduler),
            },
        }, ckpt_path)

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
                    action = random.random()  # [0, 1]
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
                "[SHORT] Ep %04d | PnL:%6.1f%% Tr:%4d WR:%4.0f%% Rew:%7.3f | buf:%6d | α:%.4f | CVaR:%+.4f",
                ep, pnl, env.total_trades, env.win_rate * 100, ep_reward, len(agent.memory), agent.alpha, _cvar,
            )

            if ep % max(1, int(val_interval)) == 0:
                agent.actor.eval()
                val_hmm = copy.deepcopy(hmm_detector)
                val_env = ShortSpecialistEnv(df_val, phase="val", hmm_detector=val_hmm, mtf_features=mtf_val)
                val_state = val_env.reset()
                val_done = False
                val_peak_eq = float(val_env.initial_balance)
                val_mdd_pct = 0.0
                val_entries = 0
                val_force_close_short = 0
                val_hold_sum_short = 0
                val_hold_n_short = 0

                while not val_done:
                    prev_pos = val_env.pos
                    with torch.no_grad():
                        val_action = agent.act(val_state, deterministic=True)
                    val_state, _, val_done, val_info = val_env.step(val_action)
                    if prev_pos is None and val_env.pos == "SHORT":
                        val_entries += 1
                    if bool(val_info.get("force_closed", False)) and str(val_info.get("closed_side", "") or "") == "SHORT":
                        val_force_close_short += 1
                    closed_hold_count = int(val_info.get("closed_hold_count", 0) or 0)
                    closed_side = str(val_info.get("closed_side", "") or "")
                    if closed_hold_count > 0 and closed_side == "SHORT":
                        val_hold_sum_short += closed_hold_count
                        val_hold_n_short += 1
                    cur_eq = val_env.balance * (1.0 + val_env.unrealized_pnl if val_env.pos is not None else 1.0)
                    val_peak_eq = max(val_peak_eq, cur_eq)
                    val_mdd_pct = min(val_mdd_pct, (cur_eq / max(val_peak_eq, 1e-8) - 1.0) * 100.0)
                agent.actor.train()

                val_pnl = (val_env.balance / val_env.initial_balance - 1.0) * 100.0
                val_wr = val_env.win_rate
                avg_hold_short = float(val_hold_sum_short / max(val_hold_n_short, 1)) if val_hold_n_short > 0 else 0.0
                val_trade_score = -5.0 if val_env.total_trades == 0 else (
                    min(val_env.total_trades / 30.0, 1.0) * 5.0 if val_pnl > 0
                    else -min(val_env.total_trades / 30.0, 1.0) * 10.0
                )
                val_score = val_pnl * 3.0 + val_wr * 60.0 + val_trade_score + val_mdd_pct * 2.0

                logger.info(
                    "[SHORT]   [VAL] PnL:%6.2f%% | Tr:%4d | WR:%.0f%% | MDD:%.2f%% | Entries:%4d | FCS:%3d | AvgHold:%4.1f | Score:%.2f",
                    val_pnl, val_env.total_trades, val_wr * 100, val_mdd_pct, val_entries, val_force_close_short, avg_hold_short, val_score,
                )

                improved = val_score > best_val_score
                if improved:
                    best_val_score, best_val_pnl = val_score, val_pnl
                    bad_val_count = 0
                    torch.save({
                        "actor": agent.actor.state_dict(), "critic": agent.critic.state_dict(),
                        "best_pnl": best_val_pnl, "best_score": best_val_score, "epoch": ep,
                        "state_dim": STATE_DIM,
                        "meta": {
                            "algo": "DSAC_SHORT",
                            "side": SIDE,
                            "state_dim": STATE_DIM,
                            "n_quantiles": agent.n_quantiles,
                            "cvar_frac": agent.cvar_frac,
                            "pos_thresh": _POS_THRESH,
                            "close_thresh": _CLOSE_THRESH,
                            "fallback_signal_gain": _FALLBACK_SIGNAL_GAIN,
                            "val_interval": int(val_interval),
                            "early_stop_patience": int(early_stop_patience),
                            "lr_factor": float(lr_factor),
                            "lr_patience": int(lr_patience),
                            "lr_min": float(lr_min),
                            "use_lr_scheduler": bool(use_lr_scheduler),
                        },
                    }, best_path)
                    logger.info("[SHORT]   🎉 NEW BEST (PnL:%.2f%%)", best_val_pnl)
                else:
                    bad_val_count += 1

                if use_lr_scheduler and actor_scheduler and critic_scheduler:
                    prev_actor_lr = float(agent.actor_optimizer.param_groups[0]["lr"])
                    prev_critic_lr = float(agent.critic_optimizer.param_groups[0]["lr"])
                    actor_scheduler.step(val_score)
                    critic_scheduler.step(val_score)
                    new_actor_lr = float(agent.actor_optimizer.param_groups[0]["lr"])
                    new_critic_lr = float(agent.critic_optimizer.param_groups[0]["lr"])
                    if (new_actor_lr < prev_actor_lr) or (new_critic_lr < prev_critic_lr):
                        logger.info(
                            "[SHORT]   📉 [LR DROP] actor %.3e -> %.3e | critic %.3e -> %.3e",
                            prev_actor_lr,
                            new_actor_lr,
                            prev_critic_lr,
                            new_critic_lr,
                        )
                    else:
                        logger.info(
                            "[SHORT]   [LR] actor %.3e | critic %.3e | bad_val=%d",
                            new_actor_lr,
                            new_critic_lr,
                            bad_val_count,
                        )

                if ep % 50 == 0:
                    hmm_detector.update_online(n_iter=5)
                    for attr in ("A", "mu", "sigma", "pi", "_obs_mean", "_obs_std"):
                        setattr(train_hmm, attr, getattr(hmm_detector, attr).copy())

                _save_checkpoint(ep)
                if int(early_stop_patience) > 0 and bad_val_count >= int(early_stop_patience):
                    logger.info(
                        "[SHORT] ⏹️ EARLY STOP bad_val_count=%d >= patience=%d | best_score=%.2f | best_pnl=%.2f%%",
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
        logger.info("[SHORT] ⚠️ 학습 중단.")
        _save_checkpoint(ep)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DSAC Short Specialist")
    p.add_argument("--csv-path", default="data/rl_training_data_full.csv")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--fresh-start", action="store_true")
    p.add_argument("--val-interval", type=int, default=10)
    p.add_argument("--no-lr-scheduler", action="store_true")
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--early-stop-patience", type=int, default=12)
    p.add_argument("--startup-check-only", action="store_true")
    p.add_argument("--cvar-frac", type=float, default=0.25)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.startup_check_only:
        logger.info("startup check ok: train_rl_dsac_short")
        raise SystemExit(0)
    train(
        csv_path=args.csv_path, train_ratio=args.train_ratio, episodes=args.episodes,
        fresh_start=args.fresh_start, use_lr_scheduler=not args.no_lr_scheduler,
        lr_factor=args.lr_factor, lr_patience=args.lr_patience, lr_min=args.lr_min,
        early_stop_patience=args.early_stop_patience, val_interval=args.val_interval,
        cvar_frac=args.cvar_frac, device=args.device,
    )
