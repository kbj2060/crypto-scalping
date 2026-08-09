"""
DSAC 코인 트레이딩 에이전트 (Distributional Soft Actor-Critic)
================================================================
기존 unified DSAC를 보존한 채, 레버리지 친화적인 상태 분리 버전을 실험한다.

Compact State (25D)
  Block A: M7 + Regime + MTF 문맥 (13)
  Block B: execution context (2)
  Block C: position/risk state (6)
  Block D: AI meta insights (4)
"""

import copy
import gc
import logging
import os
import random
import argparse
import sys
import warnings
import json
import hashlib
from datetime import datetime
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



from ensemble.train_rl_agent import (  # noqa: E402
    MultiTimeframeFeatures,
    OnlineHMMDetector,
    REGIME_COLS,
    STATE_CONF,
    STATE_PRED,
)
from ensemble.rl_continuous_common import ReplayBuffer  # noqa: E402
from ensemble.rl_continuous_dsac_baseline import SACTradingEnv as _BaseSACTradingEnv  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# DSAC Compact+AI State Spec (25D leverage-aware unified state)
# ─────────────────────────────────────────────────────────────────────────────
DSAC_STATE_DIM = 25
DSAC_ACTION_DIM = 1
DSAC_STATE_SCHEMA = "baseline29_plus_ai4_meta_state_leverage_unlev25_v1"

AI_FEATS = [
    "patchtst_regime_sim",
    "ai_vol_regime_pct",
    "tide_vol_zscore",
    "ai_flow_flip_prob",
]

# action threshold는 기존 SAC/DSAC와 동일
_POS_THRESH = 0.12
_NTR_ENTRY_THRESH = 0.06
_CLOSE_THRESH = 0.03

LOG_STD_MIN = -20
LOG_STD_MAX = 2


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


def _soft_gate_enabled(default: bool = False) -> bool:
    v = os.getenv("RL_UNIFIED_SOFT_GATE_ENABLE")
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _futures_exec_enabled(default: bool = False) -> bool:
    v = os.getenv("RL_UNIFIED_FUTURES_EXEC_ENABLE")
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_leverage_usage(current_leverage: float, margin_usage: float, max_leverage: float | None = None) -> float:
    cur = max(float(current_leverage), 0.0)
    margin = float(np.clip(float(margin_usage), 0.0, 1.0))
    max_lev = max(1.0, float(_max_futures_leverage() if max_leverage is None else max_leverage))
    effective = margin * max(cur, 1.0)
    return float(np.clip(effective / max_lev, 0.0, 1.0))


def _unlevered_pnl_from_exposure(pnl_value: float, notional_exposure: float) -> float:
    exp = max(float(notional_exposure), 1e-8)
    if not np.isfinite(exp) or exp <= 1e-8:
        return 0.0
    return float(float(pnl_value) / exp)


def _max_futures_leverage(default: float = 5.0) -> float:
    try:
        return max(1.0, float(os.getenv("RL_MAX_LEVERAGE", str(default))))
    except Exception:
        return float(default)


def _leverage_levels(default: tuple[float, ...] = (1.0, 3.0, 5.0)) -> tuple[float, ...]:
    raw = os.getenv("RL_LEVERAGE_LEVELS", ",".join(str(x) for x in default))
    levels = []
    for tok in str(raw).split(","):
        try:
            val = float(tok.strip())
        except Exception:
            continue
        if np.isfinite(val) and val >= 1.0:
            levels.append(val)
    if not levels:
        levels = list(default)
    levels = sorted(set(max(1.0, float(v)) for v in levels))
    max_lev = _max_futures_leverage()
    return tuple(float(v) for v in levels if v <= max_lev + 1e-9) or (1.0,)


def _snap_leverage_bucket(target_leverage: float, lev_cap: float | None = None) -> float:
    allowed = [float(v) for v in _leverage_levels() if lev_cap is None or float(v) <= float(lev_cap) + 1e-9]
    if not allowed:
        return 1.0
    target = max(1.0, float(target_leverage))
    return float(min(allowed, key=lambda v: abs(v - target)))


def _min_trade_exposure(default: float = 0.10) -> float:
    try:
        return float(np.clip(float(os.getenv("RL_MIN_TRADE_EXPOSURE", str(default))), 0.0, _max_futures_leverage()))
    except Exception:
        return float(default)


def _risk_max_margin(default: float = 0.35) -> float:
    try:
        return float(np.clip(float(os.getenv("RL_RISK_MAX_MARGIN", str(default))), 0.01, 1.0))
    except Exception:
        return float(default)


def _net_edge_min(default: float = 0.0015) -> float:
    try:
        return max(0.0, float(os.getenv("RL_NET_EDGE_MIN", str(default))))
    except Exception:
        return float(default)


def _action_to_futures_leverage(action_abs: float, pos_thresh: float = _POS_THRESH, max_leverage: float | None = None) -> float:
    max_lev = _max_futures_leverage() if max_leverage is None else max(1.0, float(max_leverage))
    den = max(1.0 - float(pos_thresh), 1e-6)
    intensity = float(np.clip((float(action_abs) - float(pos_thresh)) / den, 0.0, 1.0))
    return float(1.0 + intensity * (max_lev - 1.0))


def _risk_overlay_from_features(features: dict[str, Any] | None, conviction: float) -> tuple[float, float, float]:
    max_lev = _max_futures_leverage()
    features = features or {}
    regime = np.asarray([_safe_float(features.get(c, 0.0), 0.0) for c in REGIME_COLS], dtype=np.float32)
    regime = np.nan_to_num(regime, nan=0.0)
    regime_idx = int(np.argmax(regime)) if regime.size and float(np.max(regime)) > 0.0 else 4

    if regime_idx == 0:
        lev_cap = min(max_lev, float(os.getenv("RL_RISK_CHOP_LEV_CAP", "1.4")))
    elif regime_idx == 1:
        lev_cap = min(max_lev, float(os.getenv("RL_RISK_WHIPSAW_LEV_CAP", "1.2")))
    elif regime_idx in (2, 3):
        lev_cap = min(max_lev, float(os.getenv("RL_RISK_TREND_LEV_CAP", str(max_lev))))
    else:
        lev_cap = min(max_lev, float(os.getenv("RL_RISK_NORMAL_LEV_CAP", "2.5")))
    lev_cap = max(1.0, lev_cap)

    vol_pct = float(np.clip(_safe_float(features.get("ai_vol_regime_pct", 0.5), 0.5), 0.0, 1.0))
    tide_z = _safe_float(features.get("tide_vol_zscore", 0.0), 0.0)
    adverse = float(np.clip((np.tanh(tide_z / 3.0) + 1.0) * 0.5, 0.0, 1.0))
    flow_flip = float(np.clip(_safe_float(features.get("ai_flow_flip_prob", 0.5), 0.5), 0.0, 1.0))
    q50 = _safe_float(features.get("m7_q50", 0.0), 0.0)
    qwidth = max(0.0, _safe_float(features.get("m7_qwidth", np.nan), np.nan))
    if not np.isfinite(qwidth) or qwidth <= 1e-12:
        qwidth = max(abs(_safe_float(features.get("garch_vol_z", 0.0), 0.0)) * 0.002, 5e-4)
    uncertainty = float(np.clip(qwidth / 0.015, 0.0, 1.0))
    # Opportunity gate only: M7 direction can invert out-of-sample, so q50 is
    # used as magnitude/edge filter while the policy remains responsible for
    # long-vs-short direction.
    edge_buffer = _net_edge_min() * (1.0 + 0.35 * vol_pct + 0.35 * flow_flip + 0.20 * uncertainty)
    if abs(q50) <= edge_buffer:
        return 0.0, 1.0, 0.0

    reward_edge = float(np.clip(abs(q50) / max(qwidth, 1e-4), 0.0, 1.0))
    risk_damp = (
        (1.0 - 0.55 * vol_pct)
        * (1.0 - 0.45 * adverse)
        * (1.0 - 0.35 * flow_flip)
        * (1.0 - 0.25 * uncertainty)
    )
    risk_damp = float(np.clip(risk_damp, 0.10, 1.0))
    opportunity = float(np.clip(0.55 + 0.45 * reward_edge, 0.55, 1.0))
    effective = float(np.clip(conviction * risk_damp * opportunity, 0.0, 1.0))

    leverage = float(1.0 + (lev_cap - 1.0) * effective)
    leverage = _snap_leverage_bucket(leverage, lev_cap=lev_cap)
    exposure_cap = float(max(_min_trade_exposure(), lev_cap * _risk_max_margin()))
    exposure_cap = min(exposure_cap, lev_cap)
    exposure = float(_min_trade_exposure() + (exposure_cap - _min_trade_exposure()) * effective)
    if effective <= 1e-8:
        exposure = 0.0
    margin_fraction = float(np.clip(exposure / max(leverage, 1e-8), 0.0, _risk_max_margin()))
    exposure = float(margin_fraction * leverage)
    return margin_fraction, leverage, exposure


def _decode_futures_action(action: Any, features: dict[str, Any] | None = None) -> tuple[float, float, float, float]:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.size <= 1:
        direction = float(np.clip(arr[0] if arr.size else 0.0, -1.0, 1.0))
        den = max(1.0 - float(_POS_THRESH), 1e-6)
        conviction = float(np.clip((abs(direction) - float(_POS_THRESH)) / den, 0.0, 1.0))
    else:
        direction = float(np.clip(arr[0], -1.0, 1.0))
        dir_intensity = float(
            np.clip((abs(direction) - float(_POS_THRESH)) / max(1.0 - float(_POS_THRESH), 1e-6), 0.0, 1.0)
        )
        raw_conviction = float(np.clip((arr[1] + 1.0) * 0.5, 0.0, 1.0))
        conviction = dir_intensity * raw_conviction
    if abs(direction) <= _POS_THRESH:
        conviction = 0.0
    overlay_features = dict(features or {})
    overlay_features["_policy_direction"] = direction
    margin_fraction, leverage, exposure = _risk_overlay_from_features(overlay_features, conviction)
    return direction, margin_fraction, leverage, exposure


def _futures_risk_features(
    *,
    pos_sign: float,
    unrealized_pnl: float,
    max_drawdown: float,
    exposure_usage: float,
    current_leverage: float,
    max_leverage: float,
    hold_vs_expected: float,
    q50: float,
    tp_offset: float,
    sl_offset: float,
    ai_adverse_raw: float,
    ai_reward_raw: float,
    ai_vol_pct: float,
    ai_flow_flip: float,
) -> tuple[float, float, float, float, float, float]:
    in_pos = abs(float(pos_sign)) > 0.0
    exposure = max(float(exposure_usage), 0.0)
    leverage = max(float(current_leverage), 0.0)
    max_lev = max(float(max_leverage), 1.0)
    adverse_ref = max(abs(float(sl_offset)), abs(float(ai_adverse_raw)), 5e-4)
    reward_ref = max(abs(float(tp_offset)), abs(float(ai_reward_raw)) * adverse_ref, adverse_ref, 5e-4)

    risk_adjusted_exposure = float(np.clip(exposure * np.clip(float(ai_adverse_raw) / 0.010, 0.0, 3.0), 0.0, 1.0))
    stop_pressure = 0.0
    take_profit_pressure = 0.0
    liquidation_buffer = 1.0
    expected_edge_vs_risk = _norm_tanh(abs(float(q50)), adverse_ref)

    if in_pos:
        stop_pressure = float(np.clip(max(0.0, -float(unrealized_pnl)) / adverse_ref, 0.0, 3.0) / 3.0)
        take_profit_pressure = float(np.clip(max(0.0, float(unrealized_pnl)) / reward_ref, 0.0, 3.0) / 3.0)
        side_edge = float(pos_sign) * float(q50)
        expected_edge_vs_risk = _norm_tanh(side_edge, adverse_ref)
        notional = max(exposure * max_lev, 1e-6)
        raw_loss = max(0.0, -float(unrealized_pnl) / notional)
        raw_dd = max(0.0, -float(max_drawdown) / notional)
        liq_move = max(0.90 / max(leverage, 1.0), 1e-4)
        liquidation_buffer = float(np.clip((liq_move - max(raw_loss, raw_dd)) / liq_move, 0.0, 1.0))

    hold_risk = max(0.0, float(hold_vs_expected)) * float(np.clip(ai_vol_pct, 0.0, 1.0))
    scale_down_pressure = float(
        np.clip(
            0.35 * stop_pressure
            + 0.20 * risk_adjusted_exposure
            + 0.20 * float(np.clip(ai_flow_flip, 0.0, 1.0))
            + 0.15 * hold_risk
            + 0.10 * (1.0 - liquidation_buffer),
            0.0,
            1.0,
        )
    )
    return (
        float(liquidation_buffer),
        float(stop_pressure),
        float(take_profit_pressure),
        float(risk_adjusted_exposure),
        float(expected_edge_vs_risk),
        float(scale_down_pressure),
    )


def _pick_first(features: dict[str, Any], keys: list[str], default: float = 0.0) -> float:
    for k in keys:
        if k in features:
            return _safe_float(features.get(k), default)
    return float(default)


def _normalize_prob2(dn: float, up: float) -> tuple[float, float]:
    p = np.array([max(dn, 0.0), max(up, 0.0)], dtype=np.float64)
    s = float(p.sum())
    if s <= 1e-12:
        return (0.5, 0.5)
    p = p / s
    return float(p[0]), float(p[1])


def _prob_entropy_norm2(dn: float, up: float) -> float:
    p = np.array([dn, up], dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / max(float(p.sum()), 1e-12)
    ent = float(-np.sum(p * np.log(p)))
    return float(np.clip(ent / np.log(2.0), 0.0, 1.0))


def _hmm_cache_key(csv_path: str, train_ratio: float, n_iter: int) -> str:
    try:
        st = os.stat(csv_path)
        sig = f"{os.path.abspath(csv_path)}|{st.st_mtime_ns}|{st.st_size}|{float(train_ratio):.6f}|{int(n_iter)}"
    except Exception:
        sig = f"{os.path.abspath(csv_path)}|na|na|{float(train_ratio):.6f}|{int(n_iter)}"
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]


def _save_hmm_cache(detector: Any, cache_path: str, cache_key: str) -> None:
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        payload = {
            "cache_key": cache_key,
            "A": detector.A,
            "pi": detector.pi,
            "mu": detector.mu,
            "sigma": detector.sigma,
            "_obs_mean": detector._obs_mean,
            "_obs_std": detector._obs_std,
            "_alpha": detector._alpha,
        }
        np.savez_compressed(cache_path, **payload)
    except Exception as e:
        logger.warning("[HMM CACHE] save 실패: %s", e)


def _load_hmm_cache(cache_path: str, cache_key: str) -> OnlineHMMDetector | None:
    if not cache_path or (not os.path.exists(cache_path)):
        return None
    try:
        z = np.load(cache_path, allow_pickle=False)
        key = str(z["cache_key"].item()) if "cache_key" in z.files else ""
        if key != cache_key:
            return None
        d = OnlineHMMDetector()
        d.A = z["A"].astype(np.float64)
        d.pi = z["pi"].astype(np.float64)
        d.mu = z["mu"].astype(np.float64)
        d.sigma = z["sigma"].astype(np.float64)
        d._obs_mean = z["_obs_mean"].astype(np.float64)
        d._obs_std = z["_obs_std"].astype(np.float64)
        d._alpha = z["_alpha"].astype(np.float64)
        d._fitted = True
        return d
    except Exception as e:
        logger.warning("[HMM CACHE] load 실패: %s", e)
        return None


_FALLBACK_SIGNAL_GAIN = float(os.getenv("DSAC_FALLBACK_SIGNAL_GAIN", "8.0"))
_M7_DIR_SCALE = float(os.getenv("DSAC_M7_DIR_SCALE", "0.55"))
_M7_DIR_DROPOUT = float(os.getenv("DSAC_M7_DIR_DROPOUT", "0.15"))
_M7_DIR_NOISE = float(os.getenv("DSAC_M7_DIR_NOISE", "0.03"))


class DSACCompactTradingEnv(_BaseSACTradingEnv):
    """DSAC 전용 compact state 환경.

    거래/보상 로직은 부모 SACTradingEnv를 그대로 사용하고,
    상태 구성(_build_state)만 30차원 compact 벡터로 교체한다.
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
        side_mode="both",
        reward_beta=None,
        specialist_pos_thresh=None,
        specialist_close_thresh=None,
        specialist_min_opportunity_move=None,
        specialist_min_breakout=None,
        specialist_idle_penalty=None,
        dd_penalty_coeff=None,
        kelly_align_bonus=None,
        kelly_chop_loss_penalty=None,
        adverse_hold_enable=None,
        terminal_reward_scale: float = 0.0,
        terminal_quality_win: float = 0.0,
        terminal_quality_loss: float = 0.0,
    ):
        # 부모 __init__ 내부 reset() 호출 전에 guard 활성화
        self._compact_ready = False
        self._n_rows = len(df)
        if specialist_pos_thresh is None:
            specialist_pos_thresh = _POS_THRESH
        if specialist_close_thresh is None:
            specialist_close_thresh = _CLOSE_THRESH
        super().__init__(
            df=df,
            initial_balance=initial_balance,
            fee=fee,
            slip=slip,
            phase=phase,
            hmm_detector=hmm_detector,
            mtf_features=mtf_features,
            side_mode=side_mode,
            reward_beta=reward_beta,
            specialist_pos_thresh=specialist_pos_thresh,
            specialist_close_thresh=specialist_close_thresh,
            specialist_min_opportunity_move=specialist_min_opportunity_move,
            specialist_min_breakout=specialist_min_breakout,
            specialist_idle_penalty=specialist_idle_penalty,
            dd_penalty_coeff=dd_penalty_coeff,
            kelly_align_bonus=kelly_align_bonus,
            kelly_chop_loss_penalty=kelly_chop_loss_penalty,
            adverse_hold_enable=adverse_hold_enable,
            terminal_reward_scale=terminal_reward_scale,
            terminal_quality_win=terminal_quality_win,
            terminal_quality_loss=terminal_quality_loss,
        )
        self._n_rows = len(self.df)
        self.max_leverage = _max_futures_leverage()
        self.leverage_levels = _leverage_levels()
        self.current_notional_exposure = 0.0

        close = np.maximum(self._close_np.astype(np.float64), 1e-8)
        log_close = np.log(close)
        self._logret_np = np.zeros(self._n_rows, dtype=np.float32)
        if self._n_rows > 1:
            self._logret_np[1:] = np.diff(log_close).astype(np.float32)

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
        self._jump_z_np = self._col_or_default("jump_z", 0.0)
        self._evt_excess_z_np = self._col_or_default("evt_excess_z", 0.0)
        self._jump_flag_np = self._col_or_default("jump_flag", 0.0)
        self._evt_tail_flag_np = self._col_or_default("evt_tail_flag", 0.0)
        self._mtf_trend_1h_np = self._col_or_default("mtf_trend_1h", 0.0)
        self._mtf_trend_4h_np = self._col_or_default("mtf_trend_4h", 0.0)
        self._smart_money_flow_np = self._col_or_default("smart_money_flow", 0.0)
        self._taker_acceleration_np = self._col_or_default("taker_acceleration", 0.0)
        self._rogers_satchell_vol_np = self._col_or_default("rogers_satchell_vol", 0.0)
        self._amihud_illiquidity_z_np = self._col_or_default("amihud_illiquidity_z", 0.0)

        # M7 columns (없으면 fallback 사용)
        self._m7_prob_up_np = self._col_first_or_none("m7_trend_xgb_up", "m7_prob_up", "trend_up_prob")
        self._m7_prob_dn_np = self._col_first_or_none("m7_trend_xgb_dn", "m7_prob_dn", "trend_dn_prob")
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
        self._m7_entry_long_offset_np = self._col_or_none("m7_entry_long_offset")
        self._m7_entry_short_offset_np = self._col_or_none("m7_entry_short_offset")
        self._m7_tp_offset_np = self._col_or_none("m7_tp_offset")
        self._m7_sl_offset_np = self._col_or_none("m7_sl_offset")

        # AI 모델 피처 로드
        self._ai_feats_dict = {}
        missing_ai = []
        for col in AI_FEATS:
            if col not in self.df.columns:
                missing_ai.append(col)
                continue
            arr = pd.to_numeric(self.df[col], errors="coerce")
            if arr.isna().mean() > 0.5:
                raise ValueError(f"AI feature {col} has >50% NaN values. Data is corrupted.")
            self._ai_feats_dict[col] = arr.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        if missing_ai:
            raise ValueError(f"Missing required AI features: {missing_ai}")

        self._pred_slice = slice(0, self._n_pred)
        self._conf_slice = slice(self._n_pred, self._n_pred + self._n_conf)
        self._regime_slice = slice(
            self._n_pred + self._n_conf + self._n_elite + self._n_alpha,
            self._n_pred + self._n_conf + self._n_elite + self._n_alpha + self._n_regime,
        )
        self._compact_ready = True
        # 부모 init 시 생성된 placeholder state를 실제 unified state로 갱신
        self.reset()

    def reset(self, start_idx=None):
        st = super().reset(start_idx=start_idx)
        self.current_notional_exposure = 0.0
        return st

    def _col_or_none(self, col: str) -> np.ndarray | None:
        if col not in self.df.columns:
            return None
        arr = pd.to_numeric(self.df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return arr.to_numpy(dtype=np.float32)

    def _col_first_or_none(self, *cols: str) -> np.ndarray | None:
        for col in cols:
            arr = self._col_or_none(col)
            if arr is not None:
                return arr
        return None

    def _col_or_default(self, col: str, default: float = 0.0) -> np.ndarray:
        arr = self._col_or_none(col)
        if arr is None:
            return np.full(self._n_rows, float(default), dtype=np.float32)
        return arr

    def _build_spread_proxy(self) -> np.ndarray:
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
                out = s.to_numpy(dtype=np.float32)
                return np.clip(out, 0.0, 0.05).astype(np.float32)

        if "high" in self.df.columns and "low" in self.df.columns:
            close_series = pd.Series(self._close_np, index=self.df.index)
            high = pd.to_numeric(self.df["high"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(close_series)
            low = pd.to_numeric(self.df["low"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(close_series)
            close = np.maximum(self._close_np.astype(np.float64), 1e-8)
            out = np.abs((high.to_numpy(dtype=np.float64) - low.to_numpy(dtype=np.float64)) / close)
            return np.clip(out, 0.0, 0.05).astype(np.float32)

        proxy = np.abs(self._logret_np.astype(np.float64)) * 0.25 + 2e-4
        return np.clip(proxy, 0.0, 0.02).astype(np.float32)

    def _overlay_feature_dict(self, idx: int) -> dict[str, float]:
        i = min(max(int(idx), 0), self._n_rows - 1)
        out = {c: float(self._feat_np[i][self._regime_slice][j]) for j, c in enumerate(REGIME_COLS)}
        out["ai_vol_regime_pct"] = float(self._arr_at(self._ai_feats_dict.get("ai_vol_regime_pct"), i, 0.5))
        out["tide_vol_zscore"] = float(self._arr_at(self._ai_feats_dict.get("tide_vol_zscore"), i, 0.0))
        out["ai_flow_flip_prob"] = float(self._arr_at(self._ai_feats_dict.get("ai_flow_flip_prob"), i, 0.5))
        out["m7_q50"] = float(self._arr_at(self._m7_q50_np, i, 0.0))
        out["m7_qwidth"] = float(self._arr_at(self._m7_qwidth_np, i, np.nan))
        out["garch_vol_z"] = float(self._arr_at(self._garch_vol_z_np, i, 0.0))
        return out

    def step(self, action: float):
        if not _futures_exec_enabled(default=False):
            return super().step(action)
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        raw_direction = float(np.clip(arr[0] if arr.size else 0.0, -1.0, 1.0))
        fill_step = min(self.current_step + 1, len(self._open_np) - 1)
        fill_price = float(self._open_np[fill_step])
        decision_step = self.current_step
        overlay_features = self._overlay_feature_dict(decision_step)
        direction, target_margin, target_leverage, target_exposure = _decode_futures_action(raw_direction, overlay_features)
        target_exposure = float(np.clip(target_exposure, 0.0, self.max_leverage))

        if self.pos is not None:
            prev_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            prev_portfolio_value = self.balance

        force_close = bool(
            self.force_close_enable
            and self.pos is not None
            and self.unrealized_pnl <= self.force_close_th
        )

        is_entering_long = False
        is_entering_short = False
        is_closing = False
        is_adjusting = False

        if force_close:
            is_closing = True
        elif self.pos is None:
            if direction > self.pos_thresh and target_exposure > 1e-8:
                is_entering_long = True
            elif direction < -self.pos_thresh and target_exposure > 1e-8:
                is_entering_short = True
        else:
            if abs(direction) < self.close_thresh or target_exposure <= 1e-8:
                is_closing = True
            elif self.pos == "LONG" and direction < -self.pos_thresh:
                is_closing = True
            elif self.pos == "SHORT" and direction > self.pos_thresh:
                is_closing = True
            else:
                is_adjusting = True

        self._just_closed = False
        self._last_realized_pnl = 0.0
        self._was_force_closed = force_close
        self._last_closed_side = ""
        self._last_closed_hold_count = 0

        if is_entering_long:
            self.pos = "LONG"
            self.entry_price = fill_price * (1.0 + self.slip)
            self.entry_idx = fill_step
            self.current_leverage = float(target_leverage)
            self.current_margin_fraction = float(np.clip(target_margin, 0.0, 1.0))
            self.current_notional_exposure = float(np.clip(target_exposure, 0.0, self.max_leverage))
            self.balance -= self.balance * self.fee * self.current_notional_exposure
        elif is_entering_short:
            self.pos = "SHORT"
            self.entry_price = fill_price * (1.0 - self.slip)
            self.entry_idx = fill_step
            self.current_leverage = float(target_leverage)
            self.current_margin_fraction = float(np.clip(target_margin, 0.0, 1.0))
            self.current_notional_exposure = float(np.clip(target_exposure, 0.0, self.max_leverage))
            self.balance -= self.balance * self.fee * self.current_notional_exposure
        elif is_adjusting and self.pos is not None:
            old_exp = float(np.clip(getattr(self, "current_notional_exposure", 0.0), 0.0, self.max_leverage))
            new_exp = float(np.clip(target_exposure, 0.0, self.max_leverage))
            exp_delta = abs(new_exp - old_exp)
            if exp_delta > 0.025 or abs(float(target_leverage) - float(self.current_leverage)) > 1e-9:
                self.balance -= self.balance * self.fee * exp_delta
                self.current_leverage = float(target_leverage)
                self.current_margin_fraction = float(np.clip(target_margin, 0.0, 1.0))
                self.current_notional_exposure = new_exp
        elif is_closing and self.pos is not None:
            closed_side = str(self.pos)
            closed_hold_count = int(self.hold_count)
            base_balance = self.balance
            if self.pos == "LONG":
                realized_pnl = (fill_price * (1.0 - self.slip) - self.entry_price) / self.entry_price
            else:
                realized_pnl = (self.entry_price - fill_price * (1.0 + self.slip)) / self.entry_price
            realized_pnl *= float(np.clip(getattr(self, "current_notional_exposure", 0.0), 0.0, self.max_leverage))
            self.balance = base_balance * (1.0 + realized_pnl)
            self.balance -= base_balance * self.fee * float(
                np.clip(getattr(self, "current_notional_exposure", 0.0), 0.0, self.max_leverage)
            )
            self.total_trades += 1
            if realized_pnl > 0:
                self.win_trades += 1
            self._just_closed = True
            self._last_realized_pnl = realized_pnl
            self._last_closed_side = closed_side
            self._last_closed_hold_count = closed_hold_count
            self.pos = None
            self.current_leverage = 0.0
            self.current_margin_fraction = 0.0
            self.current_notional_exposure = 0.0
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
            exposure = float(np.clip(getattr(self, "current_notional_exposure", 0.0), 0.0, self.max_leverage))
            self.unrealized_pnl = raw_pnl * exposure
            self.max_drawdown = min(self.max_drawdown, self.unrealized_pnl)
            self.peak_pnl = max(self.peak_pnl, self.unrealized_pnl)

        if self.pos is not None:
            current_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            current_portfolio_value = self.balance

        step_delta = (current_portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8) * 50.0
        r1_pnl = float(np.tanh(step_delta))

        regime_step = min(max(decision_step, 0), len(self._feat_np) - 1)
        regime_raw = self._feat_np[regime_step]
        o = self._n_pred + self._n_conf + self._n_elite + self._n_alpha
        regime_vec = regime_raw[o : o + self._n_regime]
        regime_idx = int(np.argmax(regime_vec))

        r2_drawdown = 0.0
        if self.pos is not None and self.unrealized_pnl < -abs(self.dd_soft_start):
            dd_excess = abs(self.unrealized_pnl) - abs(self.dd_soft_start)
            dd_den = max(abs(self.dd_hard_scale) - abs(self.dd_soft_start), 1e-6)
            dd_ratio = np.clip(dd_excess / dd_den, 0.0, 3.0)
            r2_drawdown = -self.dd_penalty_coeff * float(dd_ratio ** 2)

        r3_quality = 0.0
        if self._just_closed:
            if self._was_force_closed:
                r3_quality = -0.30
            elif self._last_realized_pnl > 0:
                r3_quality = 0.15 * min(self._last_realized_pnl / 0.01, 1.0)
            else:
                r3_quality = -0.08 if self.side_mode == "both" else -0.05

        r4_time_decay = 0.0
        if self.pos is not None and self.hold_count > self.hold_plateau_start:
            if abs(float(self.unrealized_pnl)) < self.hold_plateau_pnl_abs:
                r4_time_decay = -self.hold_plateau_penalty * float(
                    np.clip((self.hold_count - self.hold_plateau_start) / 96.0, 0.0, 1.0)
                )

        r7_adverse_hold = 0.0
        if (
            self.pos is not None
            and self.unrealized_pnl < -abs(self.adverse_hold_pnl_th)
            and self.hold_count > self.adverse_hold_start
        ):
            r7_adverse_hold = -self.adverse_hold_penalty * float(
                np.clip(abs(self.unrealized_pnl) / 0.02, 0.0, 1.0)
            )

        r5_idle = 0.0
        if self.pos is None:
            if self.specialist_idle_penalty is not None:
                r5_idle = float(self.specialist_idle_penalty)
            else:
                if regime_idx in (2, 3):
                    r5_idle = -0.003
                elif regime_idx in (0, 1):
                    r5_idle = -0.0003
                else:
                    r5_idle = -0.001

        r6_trade_cost = 0.0

        r8_kelly_regime = 0.0
        if self.pos is not None:
            step_ret = (current_portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8)
            exposure_scale = float(np.clip(getattr(self, "current_notional_exposure", 0.0), 0.0, 1.5))
            is_aligned = (self.pos == "LONG" and regime_idx == 2) or (self.pos == "SHORT" and regime_idx == 3)
            if step_ret > 0.0 and is_aligned:
                r8_kelly_regime += self.kelly_align_bonus * exposure_scale * float(np.clip(step_ret / 0.002, 0.0, 1.0))
            if step_ret < 0.0 and regime_idx in (0, 1):
                extra = max(self.kelly_chop_loss_penalty - 1.0, 0.0)
                r8_kelly_regime -= extra * abs(r1_pnl) * exposure_scale

        raw_reward = (
            r1_pnl + r2_drawdown + r3_quality + r4_time_decay + r5_idle + r6_trade_cost + r7_adverse_hold + r8_kelly_regime
        )
        reward = float(np.clip(raw_reward, -2.0, 2.0))

        if done and self.pos is not None:
            base_balance = self.balance
            ep_fill_step = min(self.current_step, len(self._open_np) - 1)
            ep_end_price = float(self._open_np[ep_fill_step])
            if self.pos == "LONG":
                ep_realized = (ep_end_price * (1.0 - self.slip) - self.entry_price) / self.entry_price
            else:
                ep_realized = (self.entry_price - ep_end_price * (1.0 + self.slip)) / self.entry_price
            ep_realized *= float(np.clip(getattr(self, "current_notional_exposure", 0.0), 0.0, self.max_leverage))
            self.balance = base_balance * (1.0 + ep_realized)
            self.balance -= base_balance * self.fee * float(
                np.clip(getattr(self, "current_notional_exposure", 0.0), 0.0, self.max_leverage)
            )
            self.total_trades += 1
            if ep_realized > 0:
                self.win_trades += 1
            terminal_r = float(np.tanh(ep_realized * 50.0)) * self.terminal_reward_scale
            if ep_realized > 0:
                terminal_r += self.terminal_quality_win * min(ep_realized / 0.01, 1.0)
            else:
                terminal_r -= self.terminal_quality_loss
            reward = float(np.clip(raw_reward + terminal_r, -2.0, 2.0))
            self.pos = None
            self.current_leverage = 0.0
            self.current_margin_fraction = 0.0
            self.current_notional_exposure = 0.0

        info = {
            "pnl_pct": (self.balance / self.initial_balance - 1) * 100,
            "wr": self.win_trades / max(1, self.total_trades),
            "force_closed": bool(self._just_closed and self._was_force_closed),
            "closed_side": self._last_closed_side,
            "closed_hold_count": int(self._last_closed_hold_count),
            "regime_bucket": self.regime_bucket(decision_step),
            "leverage": float(self.current_leverage),
            "margin_fraction": float(self.current_margin_fraction),
            "notional_exposure": float(getattr(self, "current_notional_exposure", 0.0)),
        }
        return self._get_stacked_state(self._build_state(self.current_step)), reward, done, info

    def _get_stacked_state(self, raw_state):
        # DSAC는 frame stack 없이 compact 단일 상태 사용
        return np.asarray(raw_state, dtype=np.float32)

    def _fallback_signal_score(self, idx: int) -> float:
        row = self._feat_np[idx]
        preds = row[self._pred_slice]
        confs = row[self._conf_slice]
        if len(preds) == 0:
            return 0.0
        signal = np.nan_to_num(preds * confs, nan=0.0)
        w = np.linspace(1.0, 1.3, len(signal), dtype=np.float32)
        return float(np.dot(signal, w) / max(float(np.sum(w)), 1e-8))

    def _fallback_regime_info(self, idx: int) -> tuple[float, float]:
        row = self._feat_np[idx]
        regime_raw = np.nan_to_num(row[self._regime_slice], nan=0.0)
        if regime_raw.size == 0:
            return 0.0, 0.5
        r_sum = float(np.sum(np.maximum(regime_raw, 0.0)))
        if r_sum <= 1e-12:
            return 0.0, 0.5
        reg = regime_raw / r_sum
        reg_idx = float(np.argmax(reg))
        reg_conf = float(np.max(reg))
        return reg_idx, reg_conf

    def _arr_at(self, arr: np.ndarray | None, idx: int, default: float = 0.0) -> float:
        if arr is None:
            return float(default)
        if idx < 0 or idx >= len(arr):
            return float(default)
        return _safe_float(arr[idx], default)

    def _build_state(self, idx):
        if not getattr(self, "_compact_ready", False):
            return np.zeros(DSAC_STATE_DIM, dtype=np.float32)
        if idx < 0 or idx >= self._n_rows:
            return np.zeros(DSAC_STATE_DIM, dtype=np.float32)

        # ── Block A: Market Prediction Meta ───────────────────────────────
        signal_score = self._fallback_signal_score(idx)
        z = float(np.clip(signal_score * _FALLBACK_SIGNAL_GAIN, -12.0, 12.0))

        dn = self._arr_at(self._m7_prob_dn_np, idx, np.nan)
        up = self._arr_at(self._m7_prob_up_np, idx, np.nan)
        if not np.isfinite(dn) or not np.isfinite(up):
            up = _sigmoid(z)
            dn = _sigmoid(-z)
        dn, up = _normalize_prob2(dn, up)
        if self.phase == "train":
            if np.random.rand() < _M7_DIR_DROPOUT:
                dn, up = (0.5, 0.5)
            else:
                dn = float(np.clip(dn + np.random.normal(0.0, _M7_DIR_NOISE), 0.0, 1.0))
                up = float(np.clip(up + np.random.normal(0.0, _M7_DIR_NOISE), 0.0, 1.0))
                dn, up = _normalize_prob2(dn, up)
        trend_entropy = _prob_entropy_norm2(dn, up)

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
        qwidth = self._arr_at(self._m7_qwidth_np, idx, np.nan)
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
            self._m7_gmm_vol_rank_np,
            idx,
            np.clip((abs(_safe_float(self._garch_vol_z_np[idx])) - 0.2) / 2.5, 0.0, 1.0),
        )
        gmm_cluster_norm = float(np.clip(gmm_cluster / 4.0, -1.0, 1.0))
        gmm_conf = float(np.clip(gmm_conf, 0.0, 1.0))
        vol_rank = float(np.clip(vol_rank, 0.0, 1.0))

        iso_score = self._arr_at(self._m7_iso_score_np, idx, 0.0)
        iso_anom = self._arr_at(self._m7_iso_anom_np, idx, 0.0) >= 0.5
        vae_err = self._arr_at(self._m7_vae_error_np, idx, 0.0)
        vae_anom = self._arr_at(self._m7_vae_anom_np, idx, 0.0) >= 0.5
        vae_ratio = 1.25 if vae_anom else 0.0
        shock = (
            abs(_safe_float(self._jump_z_np[idx]))
            + 0.6 * abs(_safe_float(self._evt_excess_z_np[idx]))
            + 0.4 * abs(_safe_float(self._garch_vol_z_np[idx]))
            + 0.8 * (_safe_float(self._jump_flag_np[idx]) > 0.5)
            + 0.8 * (_safe_float(self._evt_tail_flag_np[idx]) > 0.5)
        )
        anomaly_raw = 0.55 * max(iso_score, 0.0) + 0.40 * max(vae_ratio - 1.0, 0.0) + 0.20 * shock
        if iso_anom or vae_anom:
            anomaly_raw += 0.60
        anomaly_score = float(np.clip(np.tanh(anomaly_raw), 0.0, 1.0))
        tp_offset = self._arr_at(self._m7_tp_offset_np, idx, max(q90, 0.0))
        sl_offset = self._arr_at(self._m7_sl_offset_np, idx, max(-q10, 0.0))
        tp_offset_norm = _norm_tanh(tp_offset, 0.0100)
        sl_offset_norm = _norm_tanh(sl_offset, 0.0100)
        mtf_1h_norm = _norm_tanh(_safe_float(self._mtf_trend_1h_np[idx]), 0.0100)
        mtf_4h_norm = _norm_tanh(_safe_float(self._mtf_trend_4h_np[idx]), 0.0200)

        # ── Block B: Immediate Tick Context ───────────────────────────────
        spread = max(0.0, _safe_float(self._spread_np[idx]))
        spread_mean = max(1e-8, _safe_float(self._spread_mean_np[idx], spread))
        spread_std = max(1e-6, _safe_float(self._spread_std_np[idx], 1e-6))
        spread_norm = _norm_tanh(spread, 0.0015)
        micro5 = max(0.0, _safe_float(self._micro_vol5_np[idx]))
        micro5_norm = _norm_tanh(micro5, 0.0030)
        rs_vol_norm = _norm_tanh(max(0.0, _safe_float(self._rogers_satchell_vol_np[idx])), 0.0100)
        amihud_norm = float(np.tanh(_safe_float(self._amihud_illiquidity_z_np[idx]) / 3.0))
        smart_flow_norm = _norm_tanh(_safe_float(self._smart_money_flow_np[idx]), 0.0500)
        taker_accel_norm = _norm_tanh(_safe_float(self._taker_acceleration_np[idx]), 0.0500)

        # ── Block C: Agent Private State (+ unlevered PnL / separate risk) ──
        pos_sign = 1.0 if self.pos == "LONG" else (-1.0 if self.pos == "SHORT" else 0.0)
        max_lev = max(1.0, _safe_float(getattr(self, "max_leverage", _max_futures_leverage()), _max_futures_leverage()))
        current_lev = max(0.0, _safe_float(self.current_leverage if self.pos is not None else 0.0, 0.0))
        current_exp = max(0.0, _safe_float(getattr(self, "current_notional_exposure", 0.0), 0.0))
        margin_usage = float(np.clip(_safe_float(getattr(self, "current_margin_fraction", 0.0), 0.0), 0.0, 1.0))
        effective_risk = float(np.clip(current_exp / max_lev, 0.0, 1.0))
        current_position = float(pos_sign * margin_usage)
        unrealized_unlev = _unlevered_pnl_from_exposure(self.unrealized_pnl, current_exp) if self.pos is not None else 0.0
        unrealized_norm = _norm_tanh(unrealized_unlev, 0.02)
        time_in_trade_norm = float(np.clip(self.hold_count / 96.0, 0.0, 1.0))
        hold_vs_expected = 0.0
        if self.pos is not None:
            hold_vs_expected = float(np.tanh(((self.hold_count / max(hold_raw, 1.0)) - 1.0) * 1.25))
        drawdown_unlev = _unlevered_pnl_from_exposure(self.max_drawdown, current_exp) if self.pos is not None else 0.0
        drawdown_norm = float(np.clip(drawdown_unlev / 0.05, -1.0, 1.0))

        # ── Block D: AI Meta Insights (4) ─────────────────────────────────
        ai_p_reg = float(np.clip(self._arr_at(self._ai_feats_dict["patchtst_regime_sim"], idx, 1.0), 0.0, 1.0))
        ai_vol_pct = float(np.clip(self._arr_at(self._ai_feats_dict["ai_vol_regime_pct"], idx, 0.5), 0.0, 1.0))
        ai_tide_z = _norm_tanh(self._arr_at(self._ai_feats_dict["tide_vol_zscore"], idx, 0.0), 3.0)
        ai_flow_flip = float(np.clip(self._arr_at(self._ai_feats_dict["ai_flow_flip_prob"], idx, 0.5), 0.0, 1.0))
        state = np.array(
            [
                # Block A (13)
                up * _M7_DIR_SCALE,
                dn * _M7_DIR_SCALE,
                float(np.clip(up - dn, -1.0, 1.0)) * _M7_DIR_SCALE,
                trend_entropy * _M7_DIR_SCALE,
                quality_norm,
                hold_norm,
                q_uncertainty_norm,
                q_skew,
                gmm_conf,
                vol_rank,
                tp_offset_norm,
                sl_offset_norm,
                mtf_1h_norm,
                # Block B (2)
                spread_norm,
                micro5_norm,
                # Block C (6)
                current_position,
                unrealized_norm,
                hold_vs_expected,
                margin_usage,
                effective_risk,
                drawdown_norm,
                # Block D (4)
                ai_p_reg,
                ai_vol_pct,
                ai_tide_z,
                ai_flow_flip,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

    def regime_bucket(self, idx: int | None = None) -> str:
        """현재 시점 레짐을 bull/bear/chop/whipsaw/normal 5버킷으로 매핑."""
        if idx is None:
            idx = int(self.current_step)
        idx = min(max(int(idx), 0), self._n_rows - 1)
        row = self._feat_np[idx]
        regime_raw = np.nan_to_num(row[self._regime_slice], nan=0.0)
        if regime_raw.size == 0:
            return "normal"
        reg_idx = int(np.argmax(regime_raw))
        # REGIME_COLS: chop, whipsaw, bull, bear, normal
        if reg_idx == 0:
            return "chop"
        if reg_idx == 1:
            return "whipsaw"
        if reg_idx == 2:
            return "bull"
        if reg_idx == 3:
            return "bear"
        return "normal"


class RegimeBalancedReplay:
    """레짐×시간 버킷 균형 샘플링 + 최근 버퍼 혼합 리플레이."""

    def __init__(self, capacity: int = 500000, recent_mix_ratio: float = 0.30, recent_window: int = 100000):
        cap = max(3000, int(capacity))
        per_cap = max(600, cap // 5)
        recent_cap = max(1000, int(recent_window))
        self._global = ReplayBuffer(capacity=cap)
        self._recent = ReplayBuffer(capacity=recent_cap)
        self._by_regime = {
            "bull": ReplayBuffer(capacity=per_cap),
            "bear": ReplayBuffer(capacity=per_cap),
            "chop": ReplayBuffer(capacity=per_cap),
            "whipsaw": ReplayBuffer(capacity=per_cap),
            "normal": ReplayBuffer(capacity=per_cap),
        }
        per_rt_cap = max(300, cap // 15)
        self._by_rt = {
            (r, t): ReplayBuffer(capacity=per_rt_cap)
            for r in ("bull", "bear", "chop", "whipsaw", "normal")
            for t in ("early", "mid", "late")
        }
        self.recent_mix_ratio = float(np.clip(recent_mix_ratio, 0.0, 0.9))

    def _time_bucket(self, progress: float) -> str:
        x = float(np.clip(progress, 0.0, 1.0))
        if x < 1.0 / 3.0:
            return "early"
        if x < 2.0 / 3.0:
            return "mid"
        return "late"

    def push(self, state, action, reward, next_state, done, regime: str = "normal", progress: float = 0.5):
        key = regime if regime in self._by_regime else "normal"
        tkey = self._time_bucket(progress)
        self._global.push(state, action, reward, next_state, done)
        self._recent.push(state, action, reward, next_state, done)
        self._by_regime[key].push(state, action, reward, next_state, done)
        self._by_rt[(key, tkey)].push(state, action, reward, next_state, done)

    def __len__(self):
        return len(self._global)

    def _sample_balanced(self, bs: int):
        chunks = []
        total_taken = 0
        regime_weights = {
            "normal": 0.35,
            "chop": 0.20,
            "whipsaw": 0.15,
            "bull": 0.15,
            "bear": 0.15,
        }
        quotas = {}
        allocated = 0
        for r in ("normal", "chop", "whipsaw", "bull", "bear"):
            rq = int(bs * regime_weights[r])
            base = rq // 3
            rem = rq - base * 3
            for i, t in enumerate(("early", "mid", "late")):
                q = base + (1 if i < rem else 0)
                quotas[(r, t)] = q
                allocated += q
        # rounding 보정
        keys = [(r, t) for r in ("normal", "chop", "whipsaw", "bull", "bear") for t in ("early", "mid", "late")]
        idx = 0
        while allocated < bs:
            k = keys[idx % len(keys)]
            quotas[k] += 1
            allocated += 1
            idx += 1
        for k in keys:
            q = quotas[k]
            rb = self._by_rt[k]
            if q > 0 and len(rb) >= q:
                chunks.append(rb.sample(q))
                total_taken += q
        # RT bucket이 비면 regime bucket으로 보강
        if total_taken < bs:
            left = bs - total_taken
            rq = {
                "normal": int(left * 0.35),
                "chop": int(left * 0.20),
                "whipsaw": int(left * 0.15),
                "bull": int(left * 0.15),
                "bear": 0,
            }
            rq["bear"] = max(0, left - sum(rq.values()))
            for r in ("normal", "chop", "whipsaw", "bull", "bear"):
                q = rq[r]
                rb = self._by_regime[r]
                if q > 0 and len(rb) >= q:
                    chunks.append(rb.sample(q))
                    total_taken += q
        if total_taken < bs:
            chunks.append(self._global.sample(bs - total_taken))
        return chunks

    def sample(self, batch_size: int):
        bs = int(batch_size)
        if bs <= 0:
            raise ValueError("batch_size must be positive")

        n_recent = int(bs * self.recent_mix_ratio)
        n_balanced = bs - n_recent

        chunks = []
        if n_balanced > 0:
            chunks.extend(self._sample_balanced(n_balanced))
        if n_recent > 0:
            if len(self._recent) >= n_recent:
                chunks.append(self._recent.sample(n_recent))
            else:
                chunks.append(self._global.sample(n_recent))

        s = np.concatenate([c[0] for c in chunks], axis=0)
        a = np.concatenate([c[1] for c in chunks], axis=0)
        r = np.concatenate([c[2] for c in chunks], axis=0)
        ns = np.concatenate([c[3] for c in chunks], axis=0)
        d = np.concatenate([c[4] for c in chunks], axis=0)

        idx = np.random.permutation(len(s))
        return s[idx], a[idx], r[idx], ns[idx], d[idx]


def _quantile_huber_loss(
    pred_q: torch.Tensor,
    target_q: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """IQN 계열 quantile Huber loss.

    Args:
        pred_q:   [B, N]
        target_q: [B, N]
        taus:     [N] in (0,1)
    """
    td = target_q.unsqueeze(1) - pred_q.unsqueeze(2)  # [B, N, N]
    abs_td = td.abs()
    huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa))
    tau = taus.view(1, -1, 1)
    weight = (tau - (td.detach() < 0).float()).abs()
    loss = weight * huber / kappa
    if sample_weight is not None:
        w = torch.clamp(sample_weight, min=1e-4).view(-1, 1, 1)
        loss = loss * w
    return loss.mean()


class CompactFeatureExtractor(nn.Module):
    """DSAC compact state 전용 MLP 인코더."""

    def __init__(self, state_dim=DSAC_STATE_DIM, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class GaussianActor(nn.Module):
    """공유 백본 + 레짐 헤드 3개를 사용하는 tanh-Gaussian actor.

    - Shared trunk: CompactFeatureExtractor
    - Heads: bull / bear / chop / whipsaw
    - Gating: state 기반 softmax mixture (hard 분기보다 안정적)
    """

    def __init__(self, state_dim=DSAC_STATE_DIM, hidden_dim=256, action_dim=DSAC_ACTION_DIM):
        super().__init__()
        self.feat = CompactFeatureExtractor(state_dim, hidden_dim)
        self.action_dim = int(action_dim)
        self.n_heads = 4
        self.gate_head = nn.Linear(hidden_dim, self.n_heads)
        self.mu_heads = nn.ModuleList([nn.Linear(hidden_dim, self.action_dim) for _ in range(self.n_heads)])
        self.log_std_heads = nn.ModuleList([nn.Linear(hidden_dim, self.action_dim) for _ in range(self.n_heads)])

    def _mix_heads(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_logits = self.gate_head(feat)  # [B,4]
        gate = torch.softmax(gate_logits, dim=-1)

        mu_stack = torch.stack([h(feat) for h in self.mu_heads], dim=1)  # [B,4,A]
        log_std_stack = torch.stack(
            [h(feat).clamp(LOG_STD_MIN, LOG_STD_MAX) for h in self.log_std_heads],
            dim=1,
        )  # [B,4,A]

        gate3 = gate.unsqueeze(-1)
        mu = (gate3 * mu_stack).sum(dim=1)
        log_std = (gate3 * log_std_stack).sum(dim=1).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std, gate

    def forward(self, state):
        feat = self.feat(state)
        mu, log_std, _ = self._mix_heads(feat)
        return mu, log_std

    def forward_with_gate(self, state):
        feat = self.feat(state)
        return self._mix_heads(feat)

    def sample(self, state):
        mu, log_std, _ = self.forward_with_gate(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state):
        mu, _ = self.forward(state)
        return torch.tanh(mu)

    def deterministic_with_gate(self, state):
        mu, _, gate = self.forward_with_gate(state)
        return torch.tanh(mu), gate


class DistributionalTwinCritic(nn.Module):
    """각 Critic이 N개 quantile을 출력하는 Twin Critic."""

    def __init__(self, state_dim=DSAC_STATE_DIM, hidden_dim=256, n_quantiles=32, action_dim=DSAC_ACTION_DIM):
        super().__init__()
        self.n_quantiles = int(n_quantiles)
        self.action_dim = int(action_dim)

        self.feat1 = CompactFeatureExtractor(state_dim, hidden_dim)
        self.q1 = nn.Sequential(
            nn.Linear(hidden_dim + self.action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_quantiles),
        )

        self.feat2 = CompactFeatureExtractor(state_dim, hidden_dim)
        self.q2 = nn.Sequential(
            nn.Linear(hidden_dim + self.action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_quantiles),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = self.feat1(state)
        f2 = self.feat2(state)
        x1 = torch.cat([f1, action], dim=1)
        x2 = torch.cat([f2, action], dim=1)
        return self.q1(x1), self.q2(x2)  # [B, N], [B, N]


class DSACAgent:
    """Distributional Soft Actor-Critic (risk-averse via CVaR)."""

    def __init__(
        self,
        state_dim=DSAC_STATE_DIM,
        hidden_dim=256,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        n_quantiles=32,
        cvar_frac=0.40,
        pessimism_min_weight=0.65,
        adaptive_pessimism=False,
        pessimism_disagree_scale=0.15,
        pessimism_weight_min=0.55,
        pessimism_weight_max=0.75,
        dynamic_entropy=True,
        entropy_min=-0.80,
        entropy_max=-0.45,
        entropy_std_low=0.18,
        entropy_std_high=0.35,
        entropy_step=0.05,
        critic_var_weight=False,
        critic_var_scale=1.0,
        critic_var_w_min=0.25,
        primacy_soft_reset=False,
        primacy_window=80,
        primacy_imbalance_th=0.60,
        primacy_entropy_low=0.45,
        primacy_reset_cooldown=120,
        direction_reg_lambda=0.08,
        side_balance_lambda=0.12,
        cql_reg=False,
        cql_alpha=0.02,
        redo_enable=False,
        redo_interval=500,
        redo_tau=5e-3,
        redo_ratio=0.10,
        alpha_min=5e-3,
        alpha_init=0.03,
        anti_flat_lambda=0.08,
        anti_flat_min_abs=0.18,
        anti_flat_anneal_updates=120000,
        device="cuda",
    ):
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.n_quantiles = int(n_quantiles)
        self.cvar_frac = float(cvar_frac)
        self.pessimism_min_weight = float(np.clip(pessimism_min_weight, 0.5, 1.0))
        self.adaptive_pessimism = bool(adaptive_pessimism)
        self.pessimism_disagree_scale = float(max(pessimism_disagree_scale, 0.0))
        self.pessimism_weight_min = float(np.clip(pessimism_weight_min, 0.5, 1.0))
        self.pessimism_weight_max = float(np.clip(pessimism_weight_max, self.pessimism_weight_min, 1.0))
        self.dynamic_entropy = bool(dynamic_entropy)
        self.entropy_min = float(entropy_min)
        self.entropy_max = float(entropy_max)
        self.entropy_std_low = float(entropy_std_low)
        self.entropy_std_high = float(entropy_std_high)
        self.entropy_step = float(max(entropy_step, 1e-4))
        self.critic_var_weight = bool(critic_var_weight)
        self.critic_var_scale = float(max(critic_var_scale, 0.0))
        self.critic_var_w_min = float(np.clip(critic_var_w_min, 1e-3, 1.0))
        self.primacy_soft_reset = bool(primacy_soft_reset)
        self.primacy_window = int(max(20, primacy_window))
        self.primacy_imbalance_th = float(np.clip(primacy_imbalance_th, 0.5, 0.99))
        self.primacy_entropy_low = float(max(primacy_entropy_low, 0.0))
        self.primacy_reset_cooldown = int(max(50, primacy_reset_cooldown))
        self.direction_reg_lambda = float(max(direction_reg_lambda, 0.0))
        self.side_balance_lambda = float(max(side_balance_lambda, 0.0))
        self.cql_reg = bool(cql_reg)
        self.cql_alpha = float(max(cql_alpha, 0.0))
        self.redo_enable = bool(redo_enable)
        self.redo_interval = int(max(50, redo_interval))
        self.redo_tau = float(max(redo_tau, 1e-8))
        self.redo_ratio = float(np.clip(redo_ratio, 0.0, 0.5))
        self.alpha_min = float(max(alpha_min, 1e-8))
        self.alpha_init = float(max(alpha_init, self.alpha_min))
        self.anti_flat_lambda = float(max(anti_flat_lambda, 0.0))
        self.anti_flat_min_abs = float(np.clip(anti_flat_min_abs, 0.0, 1.0))
        self.anti_flat_anneal_updates = int(max(0, anti_flat_anneal_updates))
        self._updates = 0
        self._last_soft_reset_update = -10**9
        self._soft_reset_count = 0
        self._redo_count = 0
        self._no_trade_hist: deque[float] = deque(maxlen=self.primacy_window)
        self._imb_hist: deque[float] = deque(maxlen=self.primacy_window)
        self._ent_hist: deque[float] = deque(maxlen=self.primacy_window)
        self.action_dim = int(DSAC_ACTION_DIM)
        self.actor = GaussianActor(state_dim, hidden_dim, self.action_dim).to(device)
        self.critic = DistributionalTwinCritic(state_dim, hidden_dim, self.n_quantiles, self.action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_target.eval()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.target_entropy = -float(self.action_dim)
        self.log_alpha = torch.tensor([np.log(self.alpha_init)], dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        self.taus = torch.linspace(
            0.5 / self.n_quantiles,
            1.0 - 0.5 / self.n_quantiles,
            self.n_quantiles,
            device=device,
            dtype=torch.float32,
        )

        self.memory = RegimeBalancedReplay(capacity=500000, recent_mix_ratio=0.30, recent_window=100000)

    @property
    def alpha(self) -> float:
        return float(torch.clamp(self.log_alpha.exp(), min=self.alpha_min).item())

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_ts = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(state_ts)
            else:
                action, _ = self.actor.sample(state_ts)
        return action.cpu().numpy().reshape(-1).astype(np.float32)

    def _pessimism_weight(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        base = torch.full((q1.shape[0], 1), self.pessimism_min_weight, device=q1.device, dtype=q1.dtype)
        if not self.adaptive_pessimism:
            return base
        disagree = (q1 - q2).abs().mean(dim=1, keepdim=True)
        w = base + self.pessimism_disagree_scale * disagree
        return torch.clamp(w, self.pessimism_weight_min, self.pessimism_weight_max)

    def _target_quantiles(self, ns: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(ns)  # [B,1], [B,1]
            tq1, tq2 = self.critic_target(ns, next_action)  # [B,N], [B,N]

            # Controlled pessimism: min과 max를 혼합
            w = self._pessimism_weight(tq1, tq2)
            tq_min = torch.minimum(tq1, tq2)
            tq_max = torch.maximum(tq1, tq2)
            chosen_tq = w * tq_min + (1.0 - w) * tq_max

            alpha = torch.clamp(self.log_alpha.exp().detach(), min=self.alpha_min)
            entropy_term = alpha * next_log_prob  # [B,1]
            # DSAC-T style expected-value substitution:
            # mean path and centered quantile path are separated for variance stability.
            tq_mean = chosen_tq.mean(dim=1, keepdim=True)
            tq_centered = chosen_tq - tq_mean
            target_mean = r + self.gamma * (1.0 - d) * (tq_mean - entropy_term)
            target_q = target_mean + self.gamma * (1.0 - d) * tq_centered
            return target_q

    def _cvar_min(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        k = max(1, int(self.n_quantiles * self.cvar_frac))
        q1_s, _ = torch.sort(q1, dim=1)
        q2_s, _ = torch.sort(q2, dim=1)
        c1 = q1_s[:, :k].mean(dim=1, keepdim=True)
        c2 = q2_s[:, :k].mean(dim=1, keepdim=True)
        c_min = torch.minimum(c1, c2)
        c_max = torch.maximum(c1, c2)
        w = self._pessimism_weight(q1, q2)
        return w * c_min + (1.0 - w) * c_max

    def update(self, batch_size=256) -> dict:
        if len(self.memory) < batch_size:
            return {}

        s, a, r, ns, d = self.memory.sample(batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        if a.ndim == 1:
            a = a.unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        target_q = self._target_quantiles(ns, r, d)  # [B,N]
        target_std = target_q.std(dim=1, keepdim=True, unbiased=False)
        if self.critic_var_weight:
            sample_weight = torch.clamp(
                1.0 / (1.0 + self.critic_var_scale * target_std),
                min=self.critic_var_w_min,
                max=1.0,
            )
        else:
            sample_weight = None

        q1, q2 = self.critic(s, a)  # [B,N], [B,N]
        critic_loss = _quantile_huber_loss(q1, target_q, self.taus, sample_weight=sample_weight) + _quantile_huber_loss(
            q2, target_q, self.taus, sample_weight=sample_weight
        )
        cql_pen = torch.tensor(0.0, device=self.device)
        if self.cql_reg and self.cql_alpha > 0.0:
            rand_action = torch.empty_like(a).uniform_(-1.0, 1.0)
            q1_rand, q2_rand = self.critic(s, rand_action)
            q_data_m = 0.5 * (q1.mean(dim=1, keepdim=True) + q2.mean(dim=1, keepdim=True))
            q_rand_m = 0.5 * (q1_rand.mean(dim=1, keepdim=True) + q2_rand.mean(dim=1, keepdim=True))
            cql_pen = torch.nn.functional.softplus(q_rand_m - q_data_m).mean()
            critic_loss = critic_loss + self.cql_alpha * cql_pen

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        new_action, log_prob = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, new_action)
        q_cvar = self._cvar_min(q1_new, q2_new)  # [B,1]
        alpha = torch.clamp(self.log_alpha.exp().detach(), min=self.alpha_min)
        dir_action = new_action[:, :1]
        bias_reg = dir_action.mean().pow(2)
        anti_flat_lambda_eff = self.anti_flat_lambda
        if self.anti_flat_anneal_updates > 0:
            anti_flat_lambda_eff *= max(0.0, 1.0 - float(self._updates) / float(self.anti_flat_anneal_updates))
        det_action_batch = self.actor.deterministic(s)
        action_abs_mean = dir_action.abs().mean()
        det_dir_batch = det_action_batch[:, :1]
        det_action_abs_mean = det_dir_batch.abs().mean()
        anti_flat_pen = torch.relu(torch.tensor(self.anti_flat_min_abs, device=self.device) - det_action_abs_mean)
        side_balance_pen = torch.tanh(4.0 * dir_action).mean().abs()
        actor_loss = (
            (alpha * log_prob - q_cvar).mean()
            + self.direction_reg_lambda * bias_reg
            + anti_flat_lambda_eff * anti_flat_pen
            + self.side_balance_lambda * side_balance_pen
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        alpha_for_loss = torch.clamp(self.log_alpha.exp(), min=self.alpha_min)
        alpha_loss = -(alpha_for_loss * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.data.clamp_(min=float(np.log(self.alpha_min)))

        if self.dynamic_entropy:
            action_std = float(new_action.detach().std().item())
            policy_entropy = float((-log_prob.detach()).mean().item())
            act_np = dir_action.detach().squeeze(-1).cpu().numpy()
            n_pos = float(np.sum(act_np > 0.05))
            n_neg = float(np.sum(act_np < -0.05))
            sign_imbalance = abs(n_pos - n_neg) / max(n_pos + n_neg, 1.0)
            det_action = self.actor.deterministic(s)
            det_np = det_action[:, :1].detach().squeeze(-1).cpu().numpy()
            n_entry = float(np.sum(np.abs(det_np) > _NTR_ENTRY_THRESH))
            entry_rate = n_entry / max(float(det_np.shape[0]), 1.0)
            no_trade_rate = 1.0 - entry_rate

            if (sign_imbalance > self.primacy_imbalance_th and policy_entropy < self.primacy_entropy_low) or (
                action_std < self.entropy_std_low
            ):
                self.target_entropy = min(self.entropy_max, self.target_entropy + 2.0 * self.entropy_step)
            elif action_std > self.entropy_std_high and policy_entropy > (self.primacy_entropy_low + 0.15):
                self.target_entropy = max(self.entropy_min, self.target_entropy - self.entropy_step)

            self._imb_hist.append(float(sign_imbalance))
            self._ent_hist.append(float(policy_entropy))
            self._no_trade_hist.append(float(no_trade_rate))

            if self.primacy_soft_reset and len(self._imb_hist) >= self.primacy_window:
                imb_mean = float(np.mean(self._imb_hist))
                ent_mean = float(np.mean(self._ent_hist))
                nt_mean = float(np.mean(self._no_trade_hist))
                cooldown_ok = (self._updates - self._last_soft_reset_update) >= self.primacy_reset_cooldown
                collapse_by_bias = imb_mean > self.primacy_imbalance_th and ent_mean < self.primacy_entropy_low
                collapse_by_no_trade = nt_mean > 0.90 and ent_mean < (self.primacy_entropy_low + 0.05)
                if cooldown_ok and (collapse_by_bias or collapse_by_no_trade):
                    self._soft_reset_critic_heads()
                    self._last_soft_reset_update = self._updates
                    self._soft_reset_count += 1
                    self._imb_hist.clear()
                    self._ent_hist.clear()
                    self._no_trade_hist.clear()

        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
        self._updates += 1
        if self.redo_enable and (self._updates % self.redo_interval == 0):
            self._redo_rejuvenate()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(torch.clamp(self.log_alpha.exp(), min=self.alpha_min).item()),
            "mean_q": float(torch.min(q1_new.mean(dim=1), q2_new.mean(dim=1)).mean().item()),
            "cvar_q": float(q_cvar.mean().item()),
            "target_entropy": float(self.target_entropy),
            "target_std": float(target_std.mean().item()),
            "var_w": float(sample_weight.mean().item()) if sample_weight is not None else 1.0,
            "sign_imb": float(np.mean(self._imb_hist)) if self._imb_hist else 0.0,
            "policy_entropy": float(np.mean(self._ent_hist)) if self._ent_hist else 0.0,
            "no_trade_rate": float(np.mean(self._no_trade_hist)) if self._no_trade_hist else 0.0,
            "soft_reset_count": int(self._soft_reset_count),
            "bias_reg": float(bias_reg.detach().item()),
            "cql_pen": float(cql_pen.detach().item()),
            "redo_count": int(self._redo_count),
            "q_disagree": float((q1_new - q2_new).abs().mean().item()),
            "anti_flat_pen": float(anti_flat_pen.detach().item()),
            "anti_flat_lambda_eff": float(anti_flat_lambda_eff),
            "action_abs_mean": float(action_abs_mean.detach().item()),
            "det_action_abs_mean": float(det_action_abs_mean.detach().item()),
            "side_balance_pen": float(side_balance_pen.detach().item()),
        }

    def _soft_reset_critic_heads(self):
        """Primacy 완화용: critic head만 소폭 리셋해 모드 고착을 완화."""
        for head in (self.critic.q1[-1], self.critic.q2[-1]):
            if isinstance(head, nn.Linear):
                nn.init.xavier_uniform_(head.weight)
                nn.init.zeros_(head.bias)
        # NOTE: keep target critic untouched here; let soft-update follow gradually.

    def _redo_rejuvenate_mlp(self, feat: CompactFeatureExtractor) -> int:
        net = getattr(feat, "net", None)
        if net is None or len(net) < 4:
            return 0
        lin1 = net[0]
        lin2 = net[3]
        if not isinstance(lin1, nn.Linear) or not isinstance(lin2, nn.Linear):
            return 0
        with torch.no_grad():
            out_norm = lin2.weight.data.norm(dim=0)
            mean_norm = float(out_norm.mean().item())
            th = self.redo_tau * mean_norm
            dormant_idx = torch.nonzero(out_norm < th, as_tuple=False).squeeze(-1)
            if dormant_idx.numel() == 0:
                return 0
            max_reset = max(1, int(lin1.out_features * self.redo_ratio))
            if dormant_idx.numel() > max_reset:
                vals = out_norm[dormant_idx]
                pick = torch.argsort(vals)[:max_reset]
                dormant_idx = dormant_idx[pick]
            for j in dormant_idx.tolist():
                nn.init.kaiming_uniform_(lin1.weight.data[j : j + 1], a=np.sqrt(5))
                if lin1.bias is not None:
                    lin1.bias.data[j].zero_()
                nn.init.normal_(lin2.weight.data[:, j : j + 1], mean=0.0, std=0.02)
            return int(dormant_idx.numel())

    def _redo_rejuvenate(self):
        n = 0
        n += self._redo_rejuvenate_mlp(self.actor.feat)
        n += self._redo_rejuvenate_mlp(self.critic.feat1)
        n += self._redo_rejuvenate_mlp(self.critic.feat2)
        if n > 0:
            self._redo_count += n
            self.critic_target.load_state_dict(self.critic.state_dict())


class DSACRouter:
    """라이브 추론 라우터.

    입력(features, pos)으로 compact 35D 상태를 구성해 actor를 실행한다.
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
            pv = _safe_float(features.get(p, 0.0), 0.0)
            cv = _safe_float(features.get(c, 0.5), 0.5)
            sig.append(pv * cv)
        if not sig:
            return 0.0
        arr = np.asarray(sig, dtype=np.float32)
        w = np.linspace(1.0, 1.3, len(arr), dtype=np.float32)
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
        signal_score = self._fallback_signal_score(features)
        z = float(np.clip(signal_score * _FALLBACK_SIGNAL_GAIN, -12.0, 12.0))

        dn = _pick_first(features, ["m7_trend_xgb_dn", "m7_prob_dn", "trend_dn_prob"], np.nan)
        up = _pick_first(features, ["m7_trend_xgb_up", "m7_prob_up", "trend_up_prob"], np.nan)
        if not np.isfinite(dn) or not np.isfinite(up):
            up = _sigmoid(z)
            dn = _sigmoid(-z)
        dn, up = _normalize_prob2(dn, up)
        trend_entropy = _prob_entropy_norm2(dn, up)

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
        vol_rank = _pick_first(
            features,
            ["m7_gmm_vol_rank"],
            np.clip((abs(_safe_float(features.get("garch_vol_z", 0.0))) - 0.2) / 2.5, 0.0, 1.0),
        )
        gmm_cluster_norm = float(np.clip(gmm_cluster / 4.0, -1.0, 1.0))
        gmm_conf = float(np.clip(gmm_conf, 0.0, 1.0))
        vol_rank = float(np.clip(vol_rank, 0.0, 1.0))

        iso_score = _pick_first(features, ["m7_iso_score"], 0.0)
        iso_anom = _pick_first(features, ["m7_iso_anom"], 0.0) >= 0.5
        vae_err = _pick_first(features, ["m7_vae_error"], 0.0)
        vae_anom = _pick_first(features, ["m7_vae_anom"], 0.0) >= 0.5
        vae_ratio = 1.25 if vae_anom else 0.0
        shock = (
            abs(_safe_float(features.get("jump_z", 0.0)))
            + 0.6 * abs(_safe_float(features.get("evt_excess_z", 0.0)))
            + 0.4 * abs(_safe_float(features.get("garch_vol_z", 0.0)))
            + 0.8 * (_safe_float(features.get("jump_flag", 0.0)) > 0.5)
            + 0.8 * (_safe_float(features.get("evt_tail_flag", 0.0)) > 0.5)
        )
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

        close = max(_safe_float(features.get("close", 0.0), 0.0), 0.0)
        logret = _safe_float(features.get("log_return", np.nan), np.nan)
        if not np.isfinite(logret):
            if self._prev_close is not None and self._prev_close > 0.0 and close > 0.0:
                logret = float(np.log(close / self._prev_close))
            else:
                logret = 0.0
        self._prev_close = close if close > 0.0 else self._prev_close
        self._ret_hist.append(float(logret))

        spread = _pick_first(
            features,
            ["current_spread", "bid_ask_spread", "spread", "orderbook_spread", "rel_spread", "ask_bid_spread"],
            np.nan,
        )
        if not np.isfinite(spread):
            spread = abs(float(logret)) * 0.25 + 2e-4
        spread = float(np.clip(abs(spread), 0.0, 0.05))
        self._spread_hist.append(spread)

        ret_arr = np.asarray(self._ret_hist, dtype=np.float64)
        sp_arr = np.asarray(self._spread_hist, dtype=np.float64)

        micro5 = float(np.std(ret_arr[-5:])) if ret_arr.size > 0 else 0.0

        spread_norm = _norm_tanh(spread, 0.0015)
        micro5_norm = _norm_tanh(micro5, 0.0030)
        rs_vol_norm = _norm_tanh(max(0.0, _safe_float(features.get("rogers_satchell_vol", 0.0), 0.0)), 0.0100)
        amihud_norm = float(np.tanh(_safe_float(features.get("amihud_illiquidity_z", 0.0), 0.0) / 3.0))
        smart_flow_norm = _norm_tanh(_safe_float(features.get("smart_money_flow", 0.0), 0.0), 0.0500)
        taker_accel_norm = _norm_tanh(_safe_float(features.get("taker_acceleration", 0.0), 0.0), 0.0500)

        pos_type = pos.get("type") if isinstance(pos, dict) else None
        pos_sign = 1.0 if pos_type == "LONG" else (-1.0 if pos_type == "SHORT" else 0.0)
        hold_count_proxy = 0.0
        if isinstance(pos, dict):
            hold_count_proxy = max(0.0, _safe_float(pos.get("hold_count", np.nan), np.nan))
            if not np.isfinite(hold_count_proxy):
                hold_norm_legacy = float(np.clip(_safe_float(pos.get("hold_norm", 0.0), 0.0), 0.0, 1.0))
                hold_count_proxy = hold_norm_legacy * 96.0
        time_in_trade_norm = float(np.clip(hold_count_proxy / 96.0, 0.0, 1.0))
        hold_vs_expected = 0.0
        if pos_sign != 0.0:
            hold_vs_expected = float(np.tanh(((hold_count_proxy / max(hold_raw, 1.0)) - 1.0) * 1.25))

        unr = _safe_float(pos.get("unrealized", 0.0), 0.0) if isinstance(pos, dict) else 0.0
        mdd = _safe_float(pos.get("mdd", 0.0), 0.0) if isinstance(pos, dict) else 0.0

        max_lev = _max_futures_leverage()
        current_lev = np.nan
        margin_usage = np.nan
        if isinstance(pos, dict):
            current_lev = _pick_first(pos, ["leverage", "current_leverage", "lev"], np.nan)
            margin_usage = _pick_first(pos, ["margin_fraction", "margin_usage", "position_fraction", "size_fraction"], np.nan)
            if not np.isfinite(current_lev):
                margin_raw = _safe_float(pos.get("margin_usage", np.nan), np.nan)
                if np.isfinite(margin_raw):
                    current_lev = margin_raw * max_lev if margin_raw <= 1.0 else margin_raw
        if not np.isfinite(current_lev):
            current_lev = 1.0 if pos_sign != 0.0 else 0.0
        if not np.isfinite(margin_usage):
            margin_usage = 1.0 if pos_sign != 0.0 else 0.0
        margin_usage = float(np.clip(margin_usage, 0.0, 1.0))
        current_exp = float(np.clip(margin_usage * max(current_lev, 1.0), 0.0, max_lev)) if pos_sign != 0.0 else 0.0
        effective_risk = float(np.clip(current_exp / max_lev, 0.0, 1.0))
        current_position = float(pos_sign * margin_usage)
        unrealized_unlev = _unlevered_pnl_from_exposure(unr, current_exp) if pos_sign != 0.0 else 0.0
        unrealized_norm = _norm_tanh(unrealized_unlev, 0.02)
        drawdown_unlev = _unlevered_pnl_from_exposure(mdd, current_exp) if pos_sign != 0.0 else 0.0
        drawdown_norm = float(np.clip(drawdown_unlev / 0.05, -1.0, 1.0))

        # ── Block D: AI Meta Insights (4) ─────────────────────────────────
        ai_p_reg = float(np.clip(_safe_float(features.get("patchtst_regime_sim", 1.0)), 0.0, 1.0))
        ai_vol_pct = float(np.clip(_safe_float(features.get("ai_vol_regime_pct", 0.5)), 0.0, 1.0))
        ai_tide_z = _norm_tanh(_safe_float(features.get("tide_vol_zscore", 0.0)), 3.0)
        ai_flow_flip = float(np.clip(_safe_float(features.get("ai_flow_flip_prob", 0.5)), 0.0, 1.0))
        state = np.array(
            [
                # Block A (13)
                up * _M7_DIR_SCALE,
                dn * _M7_DIR_SCALE,
                float(np.clip(up - dn, -1.0, 1.0)) * _M7_DIR_SCALE,
                trend_entropy * _M7_DIR_SCALE,
                quality_norm,
                hold_norm,
                q_uncertainty_norm,
                q_skew,
                gmm_conf,
                vol_rank,
                tp_offset_norm,
                sl_offset_norm,
                mtf_1h_norm,
                # Block B (2)
                spread_norm,
                micro5_norm,
                # Block C (6)
                current_position,
                unrealized_norm,
                hold_vs_expected,
                margin_usage,
                effective_risk,
                drawdown_norm,
                # Block D (4)
                ai_p_reg,
                ai_vol_pct,
                ai_tide_z,
                ai_flow_flip,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

    def _state_tensor(self, features, pos):
        vec = self._build_compact_state(features or {}, pos or {})
        return torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)

    def decide(self, features, pos):
        """라이브 추론: features + pos -> (action_int, leverage, info)."""
        state = self._state_tensor(features, pos)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor.deterministic(state)
        action_val = float(action.cpu().item())
        abs_action = abs(action_val)
        if not _futures_exec_enabled(default=False):
            cur_pos = pos.get("type") if isinstance(pos, dict) else None
            if cur_pos is not None:
                if abs_action < _CLOSE_THRESH:
                    action_int, leverage = 0, 0.0
                elif cur_pos == "LONG" and action_val < -_POS_THRESH:
                    action_int, leverage = 0, 0.0
                elif cur_pos == "SHORT" and action_val > _POS_THRESH:
                    action_int, leverage = 0, 0.0
                else:
                    action_int = 1 if cur_pos == "LONG" else 2
                    leverage = abs_action
            else:
                if action_val > _POS_THRESH:
                    action_int, leverage = 1, abs_action
                elif action_val < -_POS_THRESH:
                    action_int, leverage = 2, abs_action
                else:
                    action_int, leverage = 0, 0.0
            info = {
                "agent": "DSAC",
                "raw_action": round(action_val, 4),
                "kelly": float(leverage),
                "leverage": float(leverage),
                "long_edge": max(action_val, 0.0),
                "short_edge": max(-action_val, 0.0),
                "score": float(abs_action),
                "state_dim": DSAC_STATE_DIM,
                "futures_exec_enabled": False,
            }
            return action_int, leverage, info

        direction, margin_fraction, leverage, notional_exposure = _decode_futures_action(action_val, features or {})

        cur_pos = pos.get("type") if isinstance(pos, dict) else None
        if cur_pos is not None:
            if abs(direction) < _CLOSE_THRESH or notional_exposure <= 1e-8:
                action_int, leverage = 0, 0.0
                margin_fraction = 0.0
                notional_exposure = 0.0
            elif cur_pos == "LONG" and direction < -_POS_THRESH:
                action_int, leverage = 0, 0.0
                margin_fraction = 0.0
                notional_exposure = 0.0
            elif cur_pos == "SHORT" and direction > _POS_THRESH:
                action_int, leverage = 0, 0.0
                margin_fraction = 0.0
                notional_exposure = 0.0
            else:
                action_int = 1 if cur_pos == "LONG" else 2
        else:
            if direction > _POS_THRESH and notional_exposure > 1e-8:
                action_int = 1
            elif direction < -_POS_THRESH and notional_exposure > 1e-8:
                action_int = 2
            else:
                action_int, leverage = 0, 0.0
                margin_fraction = 0.0
                notional_exposure = 0.0

        info = {
            "agent": "DSAC",
            "raw_action": round(action_val, 4),
            "kelly": float(leverage),  # backward-compatible key
            "leverage": float(leverage),
            "margin_fraction": float(margin_fraction),
            "notional_exposure": float(notional_exposure),
            "long_edge": max(action_val, 0.0),
            "short_edge": max(-action_val, 0.0),
            "score": float(abs_action),
            "state_dim": DSAC_STATE_DIM,
            "futures_exec_enabled": True,
        }
        return action_int, leverage, info


# 호환용 alias (trading_bot에서 SACRouter 이름으로 import 가능)
SACRouter = DSACRouter


def train(
    csv_path: str = "data/rl_training_2025_unified.csv",
    train_ratio: float = 0.8,
    episodes: int = 1000,
    fresh_start: bool = False,
    use_lr_scheduler: bool = True,
    lr_factor: float = 0.5,
    lr_patience: int = 5,
    lr_min: float = 3e-5,
    early_stop_patience: int = 12,
    val_interval: int = 10,
    cvar_frac: float = 0.40,
    gamma: float = 0.99,
    pessimism_min_weight: float = 0.65,
    adaptive_pessimism: bool = False,
    pessimism_disagree_scale: float = 0.15,
    pessimism_weight_min: float = 0.55,
    pessimism_weight_max: float = 0.75,
    dynamic_entropy: bool = True,
    entropy_min: float = -0.80,
    entropy_max: float = -0.45,
    entropy_std_low: float = 0.18,
    entropy_std_high: float = 0.35,
    entropy_step: float = 0.05,
    critic_var_weight: bool = False,
    critic_var_scale: float = 1.0,
    critic_var_w_min: float = 0.25,
    primacy_soft_reset: bool = False,
    primacy_window: int = 80,
    primacy_imbalance_th: float = 0.60,
    primacy_entropy_low: float = 0.45,
    primacy_reset_cooldown: int = 120,
    direction_reg_lambda: float = 0.08,
    side_balance_lambda: float = 0.12,
    cql_reg: bool = False,
    cql_alpha: float = 0.02,
    redo_enable: bool = False,
    redo_interval: int = 500,
    redo_tau: float = 5e-3,
    redo_ratio: float = 0.10,
    alpha_min: float = 5e-3,
    alpha_init: float = 0.03,
    anti_flat_lambda: float = 0.08,
    anti_flat_min_abs: float = 0.18,
    anti_flat_anneal_updates: int = 120000,
    soft_gate_warmup_epochs: int = 20,
    soft_gate_ramp_epochs: int = 80,
    min_val_trades_for_best: int = 80,
    val_side_bias_penalty: float = 80.0,
    config_json_path: str = "data/ensemble/ckpt/dsac_unified_leverage_train_config.json",
    checkpoint_path: str = "data/ensemble/ckpt/dsac_unified_leverage_checkpoint.pth",
    best_path: str = "data/ensemble/ckpt/best_dsac_unified_leverage.pth",
    hmm_cache_path: str = "data/ensemble/ckpt/hmm_init_cache_dsac_unified_leverage.npz",
    hmm_force_refit: bool = False,
    device: str = "auto",
):
    if not os.path.exists(csv_path):
        logger.error("데이터가 없습니다: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    before_cols = len(df.columns)
    df = prune_to_feature_keep(df, include_entry_price=False, extra_keep=["timestamp", "ai_ready"] + AI_FEATS)
    if len(df.columns) != before_cols:
        logger.info("[DATA] feature prune: %d -> %d cols (active RL only)", before_cols, len(df.columns))
    logger.info("[DATA] csv_path=%s | rows=%d", csv_path, len(df))
    if "ai_ready" in df.columns:
        ready = pd.to_numeric(df["ai_ready"], errors="coerce").fillna(0.0)
        keep = ready >= 0.5
        dropped = int((~keep).sum())
        if dropped > 0:
            df = df.loc[keep].reset_index(drop=True)
            logger.info("[DATA] dropped warmup rows by ai_ready: %d", dropped)
        if len(df) < 200:
            raise RuntimeError(f"too few rows after ai_ready filter: {len(df)}")
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.notna().any():
            years = sorted(pd.Series(ts.dt.year.dropna().unique()).astype(int).tolist())
            logger.info("[DATA] ts_range=%s -> %s | years=%s", ts.min(), ts.max(), years)
    split_idx = int(len(df) * float(train_ratio))
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val = df.iloc[split_idx:].reset_index(drop=True)

    device = _resolve_runtime_device(device)
    logger.info("Device: %s", device)
    logger.info("DSAC compact state dim: %d", DSAC_STATE_DIM)

    hmm_fit_iter = 30
    hmm_key = _hmm_cache_key(csv_path=csv_path, train_ratio=train_ratio, n_iter=hmm_fit_iter)
    hmm_detector = None if bool(hmm_force_refit) else _load_hmm_cache(hmm_cache_path, hmm_key)
    if hmm_detector is not None:
        logger.info("[HMM] 캐시 로드 완료: %s", hmm_cache_path)
    else:
        logger.info("[HMM] OnlineHMMDetector 초기 학습 시작...")
        hmm_detector = OnlineHMMDetector()
        hmm_detector.fit(df_train, n_iter=hmm_fit_iter)
        _save_hmm_cache(hmm_detector, hmm_cache_path, hmm_key)
        logger.info("[HMM] 초기 학습 완료. cache=%s", hmm_cache_path)

    logger.info("[MTF] 멀티타임프레임 피처 선계산 중...")
    mtf_train = MultiTimeframeFeatures(df_train["close"].values.astype(np.float32))
    mtf_val = MultiTimeframeFeatures(df_val["close"].values.astype(np.float32))
    logger.info("[MTF] 선계산 완료.")

    # Standalone DSAC only: decouple reward from specialist defaults.
    dsac_dd_penalty_coeff = float(os.getenv("DSAC_DD_PENALTY_COEFF", "0.03"))
    dsac_kelly_align_bonus = float(os.getenv("DSAC_KELLY_ALIGN_BONUS", "0.0"))
    dsac_kelly_chop_loss_penalty = float(os.getenv("DSAC_KELLY_CHOP_LOSS_PENALTY", "1.30"))
    dsac_adverse_hold_enable = str(os.getenv("DSAC_ADVERSE_HOLD_ENABLE", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    logger.info(
        "[REWARD-DSAC] dd_coeff=%.3f kelly_align=%.3f chop_loss=%.3f adverse_hold=%s",
        dsac_dd_penalty_coeff,
        dsac_kelly_align_bonus,
        dsac_kelly_chop_loss_penalty,
        "ON" if dsac_adverse_hold_enable else "OFF",
    )

    train_hmm = copy.deepcopy(hmm_detector)
    env = DSACCompactTradingEnv(
        df_train,
        phase="train",
        hmm_detector=train_hmm,
        mtf_features=mtf_train,
        specialist_pos_thresh=float(_POS_THRESH),
        specialist_close_thresh=float(_CLOSE_THRESH),
        dd_penalty_coeff=dsac_dd_penalty_coeff,
        kelly_align_bonus=dsac_kelly_align_bonus,
        kelly_chop_loss_penalty=dsac_kelly_chop_loss_penalty,
        adverse_hold_enable=dsac_adverse_hold_enable,
        terminal_reward_scale=0.0,
        terminal_quality_win=0.0,
        terminal_quality_loss=0.0,
    )
    agent = DSACAgent(
        DSAC_STATE_DIM,
        hidden_dim=256,
        gamma=float(gamma),
        n_quantiles=32,
        cvar_frac=float(cvar_frac),
        pessimism_min_weight=float(pessimism_min_weight),
        adaptive_pessimism=bool(adaptive_pessimism),
        pessimism_disagree_scale=float(pessimism_disagree_scale),
        pessimism_weight_min=float(pessimism_weight_min),
        pessimism_weight_max=float(pessimism_weight_max),
        dynamic_entropy=bool(dynamic_entropy),
        entropy_min=float(entropy_min),
        entropy_max=float(entropy_max),
        entropy_std_low=float(entropy_std_low),
        entropy_std_high=float(entropy_std_high),
        entropy_step=float(entropy_step),
        critic_var_weight=bool(critic_var_weight),
        critic_var_scale=float(critic_var_scale),
        critic_var_w_min=float(critic_var_w_min),
        primacy_soft_reset=bool(primacy_soft_reset),
        primacy_window=int(primacy_window),
        primacy_imbalance_th=float(primacy_imbalance_th),
        primacy_entropy_low=float(primacy_entropy_low),
        primacy_reset_cooldown=int(primacy_reset_cooldown),
        direction_reg_lambda=float(direction_reg_lambda),
        side_balance_lambda=float(side_balance_lambda),
        cql_reg=bool(cql_reg),
        cql_alpha=float(cql_alpha),
        redo_enable=bool(redo_enable),
        redo_interval=int(redo_interval),
        redo_tau=float(redo_tau),
        redo_ratio=float(redo_ratio),
        alpha_min=float(alpha_min),
        alpha_init=float(alpha_init),
        anti_flat_lambda=float(anti_flat_lambda),
        anti_flat_min_abs=float(anti_flat_min_abs),
        anti_flat_anneal_updates=int(anti_flat_anneal_updates),
        device=device,
    )

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
        "[TRAIN CFG] val_interval=%d | lr_sched=%s (factor=%.3f patience=%d min_lr=%.1e) | early_stop_patience=%d | gamma=%.4f | cvar_frac=%.2f | pess_w=%.2f adap=%s(dscale=%.2f range=%.2f~%.2f) | dyn_ent=%s (min=%.2f max=%.2f low=%.2f high=%.2f step=%.3f) | var_weight=%s(scale=%.2f min=%.2f) | primacy_reset=%s(win=%d imb=%.2f ent=%.2f cd=%d) | cql=%s(alpha=%.3f) | redo=%s(int=%d tau=%.1e ratio=%.2f) | th(pos=%.2f close=%.2f) | alpha(min=%.1e init=%.3f) | anti_flat=%.3f(min_abs=%.2f anneal=%d) | soft_gate=(warmup=%d ramp=%d) | min_val_tr=%d | dir_reg=%.3f | side_reg=%.3f | val_side_pen=%.1f | terminal_shock=OFF | replay=(balanced+recent) | fallback_gain=%.2f",
        int(val_interval),
        "ON" if use_lr_scheduler else "OFF",
        float(lr_factor),
        int(lr_patience),
        float(lr_min),
        int(early_stop_patience),
        float(gamma),
        float(cvar_frac),
        float(pessimism_min_weight),
        "ON" if adaptive_pessimism else "OFF",
        float(pessimism_disagree_scale),
        float(pessimism_weight_min),
        float(pessimism_weight_max),
        "ON" if dynamic_entropy else "OFF",
        float(entropy_min),
        float(entropy_max),
        float(entropy_std_low),
        float(entropy_std_high),
        float(entropy_step),
        "ON" if critic_var_weight else "OFF",
        float(critic_var_scale),
        float(critic_var_w_min),
        "ON" if primacy_soft_reset else "OFF",
        int(primacy_window),
        float(primacy_imbalance_th),
        float(primacy_entropy_low),
        int(primacy_reset_cooldown),
        "ON" if cql_reg else "OFF",
        float(cql_alpha),
        "ON" if redo_enable else "OFF",
        int(redo_interval),
        float(redo_tau),
        float(redo_ratio),
        float(_POS_THRESH),
        float(_CLOSE_THRESH),
        float(alpha_min),
        float(alpha_init),
        float(anti_flat_lambda),
        float(anti_flat_min_abs),
        int(anti_flat_anneal_updates),
        int(soft_gate_warmup_epochs),
        int(soft_gate_ramp_epochs),
        int(min_val_trades_for_best),
        float(direction_reg_lambda),
        float(side_balance_lambda),
        float(val_side_bias_penalty),
        float(_FALLBACK_SIGNAL_GAIN),
    )

    os.makedirs("data/ensemble/ckpt", exist_ok=True)
    ckpt_path = str(checkpoint_path)
    best_path = str(best_path)

    run_cfg = {
        "saved_at": datetime.now().isoformat(),
        "csv_path": str(csv_path),
        "train_ratio": float(train_ratio),
        "episodes": int(episodes),
        "fresh_start": bool(fresh_start),
        "use_lr_scheduler": bool(use_lr_scheduler),
        "lr_factor": float(lr_factor),
        "lr_patience": int(lr_patience),
        "lr_min": float(lr_min),
        "early_stop_patience": int(early_stop_patience),
        "val_interval": int(val_interval),
        "cvar_frac": float(cvar_frac),
        "gamma": float(gamma),
        "pessimism_min_weight": float(pessimism_min_weight),
        "adaptive_pessimism": bool(adaptive_pessimism),
        "pessimism_disagree_scale": float(pessimism_disagree_scale),
        "pessimism_weight_min": float(pessimism_weight_min),
        "pessimism_weight_max": float(pessimism_weight_max),
        "dynamic_entropy": bool(dynamic_entropy),
        "entropy_min": float(entropy_min),
        "entropy_max": float(entropy_max),
        "entropy_std_low": float(entropy_std_low),
        "entropy_std_high": float(entropy_std_high),
        "entropy_step": float(entropy_step),
        "critic_var_weight": bool(critic_var_weight),
        "critic_var_scale": float(critic_var_scale),
        "critic_var_w_min": float(critic_var_w_min),
        "primacy_soft_reset": bool(primacy_soft_reset),
        "primacy_window": int(primacy_window),
        "primacy_imbalance_th": float(primacy_imbalance_th),
        "primacy_entropy_low": float(primacy_entropy_low),
        "primacy_reset_cooldown": int(primacy_reset_cooldown),
        "direction_reg_lambda": float(direction_reg_lambda),
        "side_balance_lambda": float(side_balance_lambda),
        "cql_reg": bool(cql_reg),
        "cql_alpha": float(cql_alpha),
        "redo_enable": bool(redo_enable),
        "redo_interval": int(redo_interval),
        "redo_tau": float(redo_tau),
        "redo_ratio": float(redo_ratio),
        "alpha_min": float(alpha_min),
        "alpha_init": float(alpha_init),
        "specialist_pos_thresh": float(_POS_THRESH),
        "specialist_close_thresh": float(_CLOSE_THRESH),
        "anti_flat_lambda": float(anti_flat_lambda),
        "anti_flat_min_abs": float(anti_flat_min_abs),
        "anti_flat_anneal_updates": int(anti_flat_anneal_updates),
        "soft_gate_warmup_epochs": int(soft_gate_warmup_epochs),
        "soft_gate_ramp_epochs": int(soft_gate_ramp_epochs),
        "min_val_trades_for_best": int(min_val_trades_for_best),
        "val_side_bias_penalty": float(val_side_bias_penalty),
        "hmm_cache_path": str(hmm_cache_path),
        "hmm_force_refit": bool(hmm_force_refit),
        "m7_dir_scale": float(_M7_DIR_SCALE),
        "m7_dir_dropout": float(_M7_DIR_DROPOUT),
        "m7_dir_noise": float(_M7_DIR_NOISE),
        "fallback_signal_gain": float(_FALLBACK_SIGNAL_GAIN),
        "max_futures_leverage": float(_max_futures_leverage()),
        "leverage_levels": [float(x) for x in _leverage_levels()],
        "min_trade_exposure": float(_min_trade_exposure()),
        "risk_max_margin": float(_risk_max_margin()),
        "net_edge_min": float(_net_edge_min()),
        "soft_gate_enabled": bool(_soft_gate_enabled(default=False)),
        "futures_exec_enabled": bool(_futures_exec_enabled(default=False)),
        "early_exit_window": int(os.getenv("RL_EARLY_EXIT_WINDOW", "3")),
        "early_exit_penalty": float(os.getenv("RL_EARLY_EXIT_PENALTY", "0.035")),
        "state_dim": int(DSAC_STATE_DIM),
        "action_dim": int(DSAC_ACTION_DIM),
        "state_schema": DSAC_STATE_SCHEMA,
        "ai_feats": list(AI_FEATS),
    }
    try:
        cfg_dir = os.path.dirname(config_json_path)
        if cfg_dir:
            os.makedirs(cfg_dir, exist_ok=True)
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(run_cfg, f, ensure_ascii=False, indent=2)
        logger.info("[CFG SAVE] %s", config_json_path)
    except Exception as e:
        logger.warning("⚠️ config json 저장 실패: %s", e)

    start_ep = 1
    if (not fresh_start) and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            ckpt_state_dim = int(ckpt.get("state_dim", -1))
            if ckpt_state_dim != int(DSAC_STATE_DIM):
                raise ValueError(f"checkpoint state_dim={ckpt_state_dim} != current state_dim={DSAC_STATE_DIM}")
            ckpt_action_dim = int(ckpt.get("action_dim", -1))
            if ckpt_action_dim != int(DSAC_ACTION_DIM):
                raise ValueError(f"checkpoint action_dim={ckpt_action_dim} != current action_dim={DSAC_ACTION_DIM}")
            ckpt_state_schema = str(ckpt.get("state_schema", ""))
            if ckpt_state_schema != DSAC_STATE_SCHEMA:
                raise ValueError(
                    f"checkpoint state_schema={ckpt_state_schema or 'missing'} != current state_schema={DSAC_STATE_SCHEMA}"
                )
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
                warmup_env = DSACCompactTradingEnv(
                    df_train,
                    phase="train",
                    hmm_detector=copy.deepcopy(hmm_detector),
                    mtf_features=mtf_train,
                    specialist_pos_thresh=float(_POS_THRESH),
                    specialist_close_thresh=float(_CLOSE_THRESH),
                    dd_penalty_coeff=dsac_dd_penalty_coeff,
                    kelly_align_bonus=dsac_kelly_align_bonus,
                    kelly_chop_loss_penalty=dsac_kelly_chop_loss_penalty,
                    adverse_hold_enable=dsac_adverse_hold_enable,
                    terminal_reward_scale=0.0,
                    terminal_quality_win=0.0,
                    terminal_quality_loss=0.0,
                )
                ws = warmup_env.reset()
                for _ in range(refill_steps):
                    wa = np.random.uniform(-1.0, 1.0, size=DSAC_ACTION_DIM).astype(np.float32)
                    w_regime = warmup_env.regime_bucket()
                    w_prog = float(warmup_env.current_step / max(1, warmup_env.end_step))
                    wns, wr, wd, _ = warmup_env.step(wa)
                    agent.memory.push(ws, wa, wr, wns, wd, regime=w_regime, progress=w_prog)
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
                "action_dim": DSAC_ACTION_DIM,
                "state_schema": DSAC_STATE_SCHEMA,
                "actor_sched": actor_sched_state,
                "critic_sched": critic_sched_state,
            },
            ckpt_path,
        )

    def _apply_normal_soft_gate(action, state_vec: np.ndarray, regime: str, gate_scale: float = 1.0):
        """저신뢰 레짐(normal/chop/whipsaw) 과매매를 줄이는 소프트 게이트."""
        a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        scalar = a.size <= 1
        if a.size == 0:
            a = np.zeros(DSAC_ACTION_DIM, dtype=np.float32)
        if not _soft_gate_enabled(default=False):
            return float(a[0]) if scalar else a
        gs = float(np.clip(gate_scale, 0.0, 1.0))
        if gs <= 1e-9:
            return float(a[0]) if scalar else a

        def _out(v: np.ndarray):
            return float(v[0]) if scalar else v.astype(np.float32)

        def _mix(mult: float):
            # gate_scale=0이면 원래 action, 1이면 full gate
            out = a.copy()
            m = 1.0 - gs * (1.0 - float(mult))
            out[0] *= m
            if out.size > 1:
                out[1:] = -1.0 + (out[1:] + 1.0) * m
            return _out(out)

        try:
            trend_entropy = float(state_vec[3])  # scaled by _M7_DIR_SCALE
            q_uncertainty = float(state_vec[7])  # 0~1
        except Exception:
            return _out(a)
        if regime == "whipsaw":
            if trend_entropy > 0.24:
                return _mix(0.80)
            return _out(a)
        if regime == "chop":
            if trend_entropy > 0.28 and q_uncertainty > 0.35:
                return _mix(0.40)
            return _mix(0.60)
        if regime != "normal":
            return _out(a)
        # trend entropy 높고 quantile 불확실성이 높으면 action 크기 축소
        if trend_entropy > 0.30 and q_uncertainty > 0.40:
            return _mix(0.45)
        if trend_entropy > 0.24 and q_uncertainty > 0.32:
            return _mix(0.65)
        return _out(a)

    def _soft_gate_scale(ep_now: int) -> float:
        warm = int(max(0, soft_gate_warmup_epochs))
        ramp = int(max(1, soft_gate_ramp_epochs))
        if ep_now <= warm:
            return 0.0
        if ep_now >= warm + ramp:
            return 1.0
        return float((ep_now - warm) / ramp)

    def _eval_policy(eval_df: pd.DataFrame) -> dict[str, float]:
        if len(eval_df) < 32:
            return {
                "pnl": 0.0,
                "wr": 0.0,
                "mdd": 0.0,
                "tr": 0,
                "long_entries": 0,
                "short_entries": 0,
                "fcl": 0,
                "fcs": 0,
                "avg_hold_long": 0.0,
                "avg_hold_short": 0.0,
                "side_balance": 0.0,
                "score": -5.0,
            }

        eval_hmm = copy.deepcopy(hmm_detector)
        eval_mtf = MultiTimeframeFeatures(eval_df["close"].values.astype(np.float32))
        e = DSACCompactTradingEnv(
            eval_df.reset_index(drop=True),
            phase="val",
            hmm_detector=eval_hmm,
            mtf_features=eval_mtf,
            specialist_pos_thresh=float(_POS_THRESH),
            specialist_close_thresh=float(_CLOSE_THRESH),
            dd_penalty_coeff=dsac_dd_penalty_coeff,
            kelly_align_bonus=dsac_kelly_align_bonus,
            kelly_chop_loss_penalty=dsac_kelly_chop_loss_penalty,
            adverse_hold_enable=dsac_adverse_hold_enable,
            terminal_reward_scale=0.0,
            terminal_quality_win=0.0,
            terminal_quality_loss=0.0,
        )
        st = e.reset()
        done = False
        peak_eq = float(e.initial_balance)
        mdd_pct = 0.0
        le = 0
        se = 0
        fcl = 0
        fcs = 0
        hs_l = 0
        hs_s = 0
        hn_l = 0
        hn_s = 0
        while not done:
            prev_pos = e.pos
            with torch.no_grad():
                a = agent.act(st, deterministic=True)
            # 평가 시에는 과매매 억제를 위해 gate 하한을 둔다.
            eval_gate_scale = max(0.50, _soft_gate_scale(ep))
            a = _apply_normal_soft_gate(a, st, e.regime_bucket(), gate_scale=eval_gate_scale)
            st, _, done, info = e.step(a)
            if prev_pos is None and e.pos == "LONG":
                le += 1
            elif prev_pos is None and e.pos == "SHORT":
                se += 1
            if bool(info.get("force_closed", False)):
                closed_side = str(info.get("closed_side", "") or "")
                if closed_side == "LONG":
                    fcl += 1
                elif closed_side == "SHORT":
                    fcs += 1
            ch = int(info.get("closed_hold_count", 0) or 0)
            cs = str(info.get("closed_side", "") or "")
            if ch > 0 and cs == "LONG":
                hs_l += ch
                hn_l += 1
            elif ch > 0 and cs == "SHORT":
                hs_s += ch
                hn_s += 1
            cur_eq = e.balance * (1.0 + e.unrealized_pnl if e.pos is not None else 1.0)
            peak_eq = max(peak_eq, cur_eq)
            mdd_pct = min(mdd_pct, (cur_eq / max(peak_eq, 1e-8) - 1.0) * 100.0)

        pnl = (e.balance / e.initial_balance - 1.0) * 100.0
        wr = e.win_rate
        if e.total_trades == 0:
            trade_score = -5.0
        elif pnl > 0:
            trade_score = min(e.total_trades / 30.0, 1.0) * 5.0
        else:
            trade_score = -min(e.total_trades / 30.0, 1.0) * 10.0
        side_total_entries = int(le + se)
        side_balance = float(min(le, se) / side_total_entries) if side_total_entries > 0 else 0.0
        side_imbalance = float(abs(le - se) / max(side_total_entries, 1))
        side_bias_pen = float(val_side_bias_penalty) * side_imbalance
        score = pnl * 3.0 + wr * 60.0 + trade_score + mdd_pct * 2.0 - side_bias_pen
        return {
            "pnl": float(pnl),
            "wr": float(wr),
            "mdd": float(mdd_pct),
            "tr": int(e.total_trades),
            "long_entries": int(le),
            "short_entries": int(se),
            "fcl": int(fcl),
            "fcs": int(fcs),
            "avg_hold_long": float(hs_l / max(hn_l, 1)) if hn_l > 0 else 0.0,
            "avg_hold_short": float(hs_s / max(hn_s, 1)) if hn_s > 0 else 0.0,
            "side_balance": float(side_balance),
            "side_pen": float(side_bias_pen),
            "score": float(score),
        }

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
                    action = np.random.uniform(-1.0, 1.0, size=DSAC_ACTION_DIM).astype(np.float32)
                else:
                    action = agent.act(state, deterministic=False)

                regime_bucket = env.regime_bucket()
                action = _apply_normal_soft_gate(action, state, regime_bucket, gate_scale=_soft_gate_scale(ep))
                prog = float(env.current_step / max(1, env.end_step))
                next_state, reward, done, _ = env.step(action)
                agent.memory.push(state, action, reward, next_state, done, regime=regime_bucket, progress=prog)
                ep_reward += reward
                state = next_state

                if global_step % update_freq == 0 and len(agent.memory) >= min_buffer:
                    last_stats = agent.update(batch)

            pnl = (env.balance / env.initial_balance - 1.0) * 100.0
            _cvar = float(last_stats.get("cvar_q", 0.0))
            _vstd = float(last_stats.get("target_std", 0.0))
            _vw = float(last_stats.get("var_w", 1.0))
            _imb = float(last_stats.get("sign_imb", 0.0))
            _pent = float(last_stats.get("policy_entropy", 0.0))
            _ntr = float(last_stats.get("no_trade_rate", 0.0))
            _sreset = int(last_stats.get("soft_reset_count", 0))
            _breg = float(last_stats.get("bias_reg", 0.0))
            _cql = float(last_stats.get("cql_pen", 0.0))
            _redo = int(last_stats.get("redo_count", 0))
            _qdis = float(last_stats.get("q_disagree", 0.0))
            _afp = float(last_stats.get("anti_flat_pen", 0.0))
            _afa = float(last_stats.get("action_abs_mean", 0.0))
            _dafa = float(last_stats.get("det_action_abs_mean", 0.0))
            _sbal = float(last_stats.get("side_balance_pen", 0.0))
            ep_steps = max(int(env.current_step - env.start_step), 1)
            avg_reward = float(ep_reward / ep_steps)
            logger.info(
                "Ep %04d | \033[32mPnL:%6.1f%%\033[0m Tr:%4d WR:%4.0f%% AvgRew:%+.4f | buf:%6d | α:%.4f | Htgt:%+.3f | CVaR_Q:%+.4f | Tstd:%.3f Vw:%.3f Imb:%.3f Ent:%.3f NTR:%.3f Breg:%.4f Sbal:%.4f AFp:%.4f Aabs:%.3f Dabs:%.3f CQL:%.4f Qdis:%.4f Rst:%d Redo:%d",
                ep,
                pnl,
                env.total_trades,
                env.win_rate * 100,
                avg_reward,
                len(agent.memory),
                agent.alpha,
                float(agent.target_entropy),
                _cvar,
                _vstd,
                _vw,
                _imb,
                _pent,
                _ntr,
                _breg,
                _sbal,
                _afp,
                _afa,
                _dafa,
                _cql,
                _qdis,
                _sreset,
                _redo,
            )

            if ep % max(1, int(val_interval)) == 0:
                agent.actor.eval()
                overall = _eval_policy(df_val)
                reg_scores = []
                reg_logs = []
                reg_cols = [c for c in REGIME_COLS if c in df_val.columns]
                if reg_cols:
                    reg_idx = np.argmax(df_val[reg_cols].to_numpy(dtype=np.float64), axis=1)
                    reg_weight = {"normal": 0.40, "bear": 0.15, "bull": 0.15, "chop": 0.15, "whipsaw": 0.15}
                    w_sum = 0.0
                    w_score = 0.0
                    for i, rc in enumerate(reg_cols):
                        rname = rc.replace("regime_", "")
                        sub = df_val.iloc[reg_idx == i].copy()
                        if len(sub) < 64:
                            continue
                        rs = _eval_policy(sub)
                        reg_scores.append(float(rs["score"]))
                        w = float(reg_weight.get(rname, 0.0))
                        w_score += w * float(rs["score"])
                        w_sum += w
                        reg_logs.append((rname, len(sub), rs["score"], rs["pnl"], rs["tr"]))
                regime_score = float(w_score / w_sum) if w_sum > 0 else (float(np.mean(reg_scores)) if reg_scores else float(overall["score"]))
                val_score = 0.5 * float(overall["score"]) + 0.5 * regime_score
                agent.actor.train()

                logger.info(
                    "    [VAL] \033[32mPnL:%6.2f%%\033[0m | Tr:%4d | WR:%.0f%% | MDD:%.2f%% | L:%4d S:%4d | SideBal:%.3f | SidePen:%.2f | FCL:%3d FCS:%3d | AvgHoldL:%4.1f AvgHoldS:%4.1f | Score:%.2f",
                    float(overall["pnl"]),
                    int(overall["tr"]),
                    float(overall["wr"]) * 100.0,
                    float(overall["mdd"]),
                    int(overall["long_entries"]),
                    int(overall["short_entries"]),
                    float(overall["side_balance"]),
                    float(overall["side_pen"]),
                    int(overall["fcl"]),
                    int(overall["fcs"]),
                    float(overall["avg_hold_long"]),
                    float(overall["avg_hold_short"]),
                    val_score,
                )
                if reg_logs:
                    reg_msg = " | ".join(
                        [f"{rn}:n={n} score={sc:.1f} pnl={pnl:.1f}% tr={tr}" for rn, n, sc, pnl, tr in reg_logs]
                    )
                    logger.info("    [VAL REGIME] %s", reg_msg)

                min_tr_ok = int(overall["tr"]) >= int(min_val_trades_for_best)
                if not min_tr_ok:
                    logger.info(
                        "    [VAL CHECKPOINT SKIP] trades=%d < min_val_trades_for_best=%d",
                        int(overall["tr"]),
                        int(min_val_trades_for_best),
                    )
                # 체크포인트 선택은 PnL 최우선으로 둔다.
                improved = min_tr_ok and (float(overall["pnl"]) > float(best_val_pnl))
                if improved:
                    best_val_score, best_val_pnl = val_score, float(overall["pnl"])
                    bad_val_count = 0
                    torch.save(
                        {
                            "actor": agent.actor.state_dict(),
                            "critic": agent.critic.state_dict(),
                            "best_pnl": best_val_pnl,
                            "best_score": best_val_score,
                            "epoch": ep,
                            "state_dim": DSAC_STATE_DIM,
                            "action_dim": DSAC_ACTION_DIM,
                            "state_schema": DSAC_STATE_SCHEMA,
                            "meta": {
                                "algo": "DSAC",
                                "n_quantiles": agent.n_quantiles,
                                "cvar_frac": agent.cvar_frac,
                                "gamma": float(agent.gamma),
                                "pessimism_min_weight": float(agent.pessimism_min_weight),
                                "adaptive_pessimism": bool(agent.adaptive_pessimism),
                                "pessimism_disagree_scale": float(agent.pessimism_disagree_scale),
                                "pessimism_weight_min": float(agent.pessimism_weight_min),
                                "pessimism_weight_max": float(agent.pessimism_weight_max),
                                "dynamic_entropy": bool(agent.dynamic_entropy),
                                "entropy_min": float(agent.entropy_min),
                                "entropy_max": float(agent.entropy_max),
                                "entropy_std_low": float(agent.entropy_std_low),
                                "entropy_std_high": float(agent.entropy_std_high),
                                "entropy_step": float(agent.entropy_step),
                                "critic_var_weight": bool(agent.critic_var_weight),
                                "critic_var_scale": float(agent.critic_var_scale),
                                "critic_var_w_min": float(agent.critic_var_w_min),
                                "primacy_soft_reset": bool(agent.primacy_soft_reset),
                                "primacy_window": int(agent.primacy_window),
                                "primacy_imbalance_th": float(agent.primacy_imbalance_th),
                                "primacy_entropy_low": float(agent.primacy_entropy_low),
                                "primacy_reset_cooldown": int(agent.primacy_reset_cooldown),
                                "direction_reg_lambda": float(agent.direction_reg_lambda),
                                "side_balance_lambda": float(agent.side_balance_lambda),
                                "cql_reg": bool(agent.cql_reg),
                                "cql_alpha": float(agent.cql_alpha),
                                "redo_enable": bool(agent.redo_enable),
                                "redo_interval": int(agent.redo_interval),
                                "redo_tau": float(agent.redo_tau),
                                "redo_ratio": float(agent.redo_ratio),
                                "anti_flat_lambda": float(agent.anti_flat_lambda),
                                "anti_flat_min_abs": float(agent.anti_flat_min_abs),
                                "anti_flat_anneal_updates": int(agent.anti_flat_anneal_updates),
                                "soft_gate_warmup_epochs": int(soft_gate_warmup_epochs),
                                "soft_gate_ramp_epochs": int(soft_gate_ramp_epochs),
                                "min_val_trades_for_best": int(min_val_trades_for_best),
                                "val_side_bias_penalty": float(val_side_bias_penalty),
                                "replay_recent_mix_ratio": float(agent.memory.recent_mix_ratio),
                            },
                        },
                        best_path,
                    )
                    logger.info("    🎉 [NEW BEST] 저장 완료 (\033[32mPnL:%.2f%%\033[0m | score:%.2f)", best_val_pnl, best_val_score)
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
    p = argparse.ArgumentParser(description="Train DSAC agent")
    p.add_argument("--csv-path", default="data/rl_training_2025_unified.csv")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--fresh-start", action="store_true", help="Ignore checkpoint and start from scratch")
    p.add_argument("--val-interval", type=int, default=10, help="Run validation every N episodes")
    p.add_argument("--no-lr-scheduler", action="store_true", help="Disable ReduceLROnPlateau schedulers")
    p.add_argument("--lr-factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    p.add_argument("--lr-patience", type=int, default=5, help="ReduceLROnPlateau patience (validation rounds)")
    p.add_argument("--lr-min", type=float, default=3e-5, help="Minimum learning rate for scheduler")
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=12,
        help="Stop if validation does not improve for N validation rounds (<=0 disables)",
    )
    p.add_argument(
        "--startup-check-only",
        action="store_true",
        help="Validate imports/arguments and exit without training",
    )
    p.add_argument("--cvar-frac", type=float, default=0.40, help="CVaR fraction for actor update")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--pessimism-min-weight", type=float, default=0.65)
    p.add_argument("--adaptive-pessimism", action="store_true", default=False)
    p.add_argument("--no-adaptive-pessimism", dest="adaptive_pessimism", action="store_false")
    p.add_argument("--pessimism-disagree-scale", type=float, default=0.15)
    p.add_argument("--pessimism-weight-min", type=float, default=0.55)
    p.add_argument("--pessimism-weight-max", type=float, default=0.75)
    p.add_argument("--dynamic-entropy", action="store_true", default=True)
    p.add_argument("--no-dynamic-entropy", dest="dynamic_entropy", action="store_false")
    p.add_argument("--entropy-min", type=float, default=-0.80)
    p.add_argument("--entropy-max", type=float, default=-0.45)
    p.add_argument("--entropy-std-low", type=float, default=0.18)
    p.add_argument("--entropy-std-high", type=float, default=0.35)
    p.add_argument("--entropy-step", type=float, default=0.05)
    p.add_argument("--no-critic-var-weight", action="store_true", default=True)
    p.add_argument("--critic-var-weight", dest="no_critic_var_weight", action="store_false")
    p.add_argument("--critic-var-scale", type=float, default=1.0)
    p.add_argument("--critic-var-w-min", type=float, default=0.25)
    p.add_argument("--no-primacy-soft-reset", action="store_true", default=True)
    p.add_argument("--primacy-soft-reset", dest="no_primacy_soft_reset", action="store_false")
    p.add_argument("--primacy-window", type=int, default=80)
    p.add_argument("--primacy-imbalance-th", type=float, default=0.60)
    p.add_argument("--primacy-entropy-low", type=float, default=0.45)
    p.add_argument("--primacy-reset-cooldown", type=int, default=120)
    p.add_argument("--direction-reg-lambda", type=float, default=0.08)
    p.add_argument("--side-balance-lambda", type=float, default=0.12)
    p.add_argument("--val-side-bias-penalty", type=float, default=80.0)
    p.add_argument("--cql-reg", action="store_true", default=False)
    p.add_argument("--no-cql-reg", dest="cql_reg", action="store_false")
    p.add_argument("--cql-alpha", type=float, default=0.02)
    p.add_argument("--redo-enable", action="store_true", default=False)
    p.add_argument("--no-redo-enable", dest="redo_enable", action="store_false")
    p.add_argument("--redo-interval", type=int, default=500)
    p.add_argument("--redo-tau", type=float, default=5e-3)
    p.add_argument("--redo-ratio", type=float, default=0.10)
    p.add_argument("--alpha-min", type=float, default=5e-3)
    p.add_argument("--alpha-init", type=float, default=0.03)
    p.add_argument("--anti-flat-lambda", type=float, default=0.08)
    p.add_argument("--anti-flat-min-abs", type=float, default=0.18)
    p.add_argument("--anti-flat-anneal-updates", type=int, default=120000)
    p.add_argument("--soft-gate-warmup-epochs", type=int, default=20)
    p.add_argument("--soft-gate-ramp-epochs", type=int, default=80)
    p.add_argument("--min-val-trades-for-best", type=int, default=80)
    p.add_argument("--hmm-cache-path", default="data/ensemble/ckpt/hmm_init_cache_dsac_unified_leverage.npz")
    p.add_argument("--hmm-force-refit", action="store_true", default=False)
    p.add_argument("--config-json-path", default="data/ensemble/ckpt/dsac_unified_leverage_train_config.json")
    p.add_argument("--checkpoint-path", default="data/ensemble/ckpt/dsac_unified_leverage_checkpoint.pth")
    p.add_argument("--best-path", default="data/ensemble/ckpt/best_dsac_unified_leverage.pth")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.startup_check_only:
        logger.info("startup check ok: train_rl_dsac_unified_2025")
        raise SystemExit(0)
    train(
        csv_path=args.csv_path,
        train_ratio=args.train_ratio,
        episodes=args.episodes,
        fresh_start=args.fresh_start,
        use_lr_scheduler=not args.no_lr_scheduler,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_min=args.lr_min,
        early_stop_patience=args.early_stop_patience,
        val_interval=args.val_interval,
        cvar_frac=args.cvar_frac,
        gamma=args.gamma,
        pessimism_min_weight=args.pessimism_min_weight,
        adaptive_pessimism=args.adaptive_pessimism,
        pessimism_disagree_scale=args.pessimism_disagree_scale,
        pessimism_weight_min=args.pessimism_weight_min,
        pessimism_weight_max=args.pessimism_weight_max,
        dynamic_entropy=args.dynamic_entropy,
        entropy_min=args.entropy_min,
        entropy_max=args.entropy_max,
        entropy_std_low=args.entropy_std_low,
        entropy_std_high=args.entropy_std_high,
        entropy_step=args.entropy_step,
        critic_var_weight=not args.no_critic_var_weight,
        critic_var_scale=args.critic_var_scale,
        critic_var_w_min=args.critic_var_w_min,
        primacy_soft_reset=not args.no_primacy_soft_reset,
        primacy_window=args.primacy_window,
        primacy_imbalance_th=args.primacy_imbalance_th,
        primacy_entropy_low=args.primacy_entropy_low,
        primacy_reset_cooldown=args.primacy_reset_cooldown,
        direction_reg_lambda=args.direction_reg_lambda,
        side_balance_lambda=args.side_balance_lambda,
        cql_reg=args.cql_reg,
        cql_alpha=args.cql_alpha,
        redo_enable=args.redo_enable,
        redo_interval=args.redo_interval,
        redo_tau=args.redo_tau,
        redo_ratio=args.redo_ratio,
        alpha_min=args.alpha_min,
        alpha_init=args.alpha_init,
        anti_flat_lambda=args.anti_flat_lambda,
        anti_flat_min_abs=args.anti_flat_min_abs,
        anti_flat_anneal_updates=args.anti_flat_anneal_updates,
        soft_gate_warmup_epochs=args.soft_gate_warmup_epochs,
        soft_gate_ramp_epochs=args.soft_gate_ramp_epochs,
        min_val_trades_for_best=args.min_val_trades_for_best,
        hmm_cache_path=args.hmm_cache_path,
        hmm_force_refit=args.hmm_force_refit,
        val_side_bias_penalty=args.val_side_bias_penalty,
        config_json_path=args.config_json_path,
        checkpoint_path=args.checkpoint_path,
        best_path=args.best_path,
        device=args.device,
    )
