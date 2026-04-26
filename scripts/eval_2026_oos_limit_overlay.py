#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR, os.path.join(_ROOT_DIR, "ensemble")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.train_rl_dsac_agent import DSAC_STATE_DIM, DSACRouter, GaussianActor

ANNUAL_FACTOR_5M = math.sqrt(365 * 24 * 12)
RL_CSV = "data/rl_training_data_full.csv"
FEAT_CSV = "data/training_features_5m.csv"
META_2026_CSV = "data/splits/year_oos/rl_meta_2026.csv"
CKPT = "data/ensemble/ckpt/best_dsac_agents.pth"
OUT_JSON = "data/ensemble/reports/eval_2026_limit_overlay.json"
TAKER_FEE = 0.0005
MAKER_FEE = 0.0002
SLIP = 0.0002


@dataclass
class OverlayConfig:
    name: str
    wait_enter_th: float
    wait_release_th: float
    max_wait_bars: int
    fallback_raw_th: float
    fallback_conf_th: float
    offset_scale: float
    adverse_mult: float


@dataclass
class HybridConfig:
    name: str
    trend_take_th: float
    pullback_wait_th: float
    release_th: float
    max_wait_bars: int
    maker_offset_mult: float
    shallow_offset_mult: float
    fallback_raw_th: float
    fallback_conf_th: float


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x


def _sharpe(eq_curve: list[float]) -> float:
    eq = np.array(eq_curve, dtype=np.float64)
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    if len(rets) < 3 or np.std(rets) < 1e-12:
        return 0.0
    return float(np.mean(rets) / np.std(rets) * ANNUAL_FACTOR_5M)


def _mdd(eq_curve: list[float]) -> float:
    eq = np.array(eq_curve, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(dd)) * 100.0


def _load_2026_df() -> pd.DataFrame:
    rl = pd.read_csv(RL_CSV)
    rl["timestamp"] = pd.to_datetime(rl["timestamp"], errors="coerce")
    df26 = rl.loc[rl["timestamp"].dt.year == 2026].copy().reset_index(drop=True)
    need_ohlc = [c for c in ("open", "high", "low") if c not in df26.columns]
    if need_ohlc:
        feat = pd.read_csv(FEAT_CSV, usecols=["timestamp", "open", "high", "low"])
        feat["timestamp"] = pd.to_datetime(feat["timestamp"], errors="coerce")
        merged = df26.merge(feat, on="timestamp", how="left", suffixes=("", "_feat"))
        for c in ("open", "high", "low"):
            feat_c = f"{c}_feat"
            if c not in merged.columns and feat_c in merged.columns:
                merged[c] = merged[feat_c]
        df26 = merged
    for c in ("close", "open", "high", "low"):
        df26[c] = pd.to_numeric(df26[c], errors="coerce")
    df26 = df26.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["timestamp", "close", "open", "high", "low"]
    ).reset_index(drop=True)
    if os.path.exists(META_2026_CSV):
        meta = pd.read_csv(META_2026_CSV, usecols=[
            "timestamp",
            "meta_long_logit",
            "meta_long_raw",
            "meta_long_std",
            "meta_primary_raw",
            "meta_primary_std",
            "meta_short_logit",
            "meta_short_raw",
            "meta_short_std",
        ])
        meta["timestamp"] = pd.to_datetime(meta["timestamp"], errors="coerce")
        df26 = df26.merge(meta, on="timestamp", how="left")
    df26["ret_1"] = df26["close"].pct_change().fillna(0.0)
    df26["ret_3"] = df26["close"].pct_change(3).fillna(0.0)
    return df26


def _unr(side: str | None, entry_price: float, current_price: float, lev: float) -> float:
    if side is None or entry_price <= 0.0 or lev <= 0.0:
        return 0.0
    raw = (
        (current_price * (1.0 - SLIP) - entry_price) / entry_price
        if side == "LONG"
        else (entry_price - current_price * (1.0 + SLIP)) / entry_price
    )
    return float(raw * lev)


def _real(side: str, entry_price: float, exit_price: float, lev: float) -> float:
    raw = (
        (exit_price * (1.0 - SLIP) - entry_price) / entry_price
        if side == "LONG"
        else (entry_price - exit_price * (1.0 + SLIP)) / entry_price
    )
    return float(raw * lev)


def _entry_plan(row: pd.Series, side: str, cfg: OverlayConfig) -> dict:
    close = _safe_float(row.get("close", 0.0), 0.0)
    conf = np.clip(_safe_float(row.get("m7_confidence", 0.5), 0.5), 0.0, 1.0)
    quality = np.clip(_safe_float(row.get("m7_quality_pred", 0.005), 0.005), 0.0, 0.05)
    if side == "LONG":
        offset = abs(_safe_float(row.get("m7_entry_long_offset", -0.0016), -0.0016))
        reco = _safe_float(row.get("m7_entry_long_price", close * (1.0 - offset)), 0.0)
        fallback = close * (1.0 - offset * cfg.offset_scale)
        limit_price = 0.7 * reco + 0.3 * fallback if reco > 0 else fallback
    else:
        offset = abs(_safe_float(row.get("m7_entry_short_offset", 0.0016), 0.0016))
        reco = _safe_float(row.get("m7_entry_short_price", close * (1.0 + offset)), 0.0)
        fallback = close * (1.0 + offset * cfg.offset_scale)
        limit_price = 0.7 * reco + 0.3 * fallback if reco > 0 else fallback
    ttl = 1 if conf > 0.85 and quality > 0.006 else 2
    return {"limit_price": float(limit_price), "ttl": ttl}


def _wait_score(row: pd.Series, side: str, cfg: OverlayConfig) -> float:
    sign = 1.0 if side == "LONG" else -1.0
    ret1 = sign * _safe_float(row.get("ret_1", 0.0), 0.0)
    ret3 = sign * _safe_float(row.get("ret_3", 0.0), 0.0)
    smf = sign * _safe_float(row.get("smart_money_flow", 0.0), 0.0)
    ofi = sign * _safe_float(row.get("ofi_acceleration", 0.0), 0.0)
    cvp = sign * _safe_float(row.get("cvp_volume_imbalance", 0.0), 0.0)
    qwidth = _safe_float(row.get("m7_qwidth", 0.01), 0.01)
    conf = _safe_float(row.get("m7_confidence", 0.5), 0.5)
    adverse = (
        0.34 * np.tanh(-ret1 / 0.0020)
        + 0.22 * np.tanh(-ret3 / 0.0040)
        + 0.14 * np.tanh(-smf / 0.0025)
        + 0.12 * np.tanh(-ofi / 0.14)
        + 0.10 * np.tanh(-cvp / 0.55)
        + 0.08 * np.tanh(qwidth / 0.012)
        - 0.10 * conf
    )
    return float(adverse * cfg.adverse_mult)


def _hybrid_state(row: pd.Series, side: str) -> dict[str, float]:
    sign = 1.0 if side == "LONG" else -1.0
    trend_xgb = sign * (
        _safe_float(row.get("m7_trend_xgb_up", 0.0), 0.0)
        - _safe_float(row.get("m7_trend_xgb_dn", 0.0), 0.0)
    )
    quant_edge = sign * (
        _safe_float(row.get("m7_quant_up", 0.0), 0.0)
        - _safe_float(row.get("m7_quant_dn", 0.0), 0.0)
    )
    meta_edge = sign * _safe_float(
        row.get("meta_primary_raw", row.get("meta_long_raw", 0.0)),
        0.0,
    )
    flow = sign * (
        0.35 * _safe_float(row.get("smart_money_flow", 0.0), 0.0)
        + 0.25 * _safe_float(row.get("ofi_acceleration", 0.0), 0.0)
        + 0.20 * _safe_float(row.get("cvp_volume_imbalance", 0.0), 0.0)
        + 0.20 * _safe_float(row.get("sig_whale", 0.0), 0.0)
    )
    crowd = sign * (
        0.55 * _safe_float(row.get("whale_conviction", 0.0), 0.0)
        + 0.45 * _safe_float(row.get("funding_price_divergence", 0.0), 0.0)
    )
    conf = np.clip(_safe_float(row.get("m7_confidence", 0.5), 0.5), 0.0, 1.0)
    quality = np.clip(_safe_float(row.get("m7_quality_pred", 0.0), 0.0), -0.05, 0.05)
    qwidth = max(_safe_float(row.get("m7_qwidth", 0.01), 0.01), 1e-4)
    tail = np.clip(_safe_float(row.get("m7_tail_risk", 0.0), 0.0), 0.0, 1.0)
    regime_push = sign * (
        _safe_float(row.get("regime_bull", 0.0), 0.0)
        - _safe_float(row.get("regime_bear", 0.0), 0.0)
    )
    ret1 = sign * _safe_float(row.get("ret_1", 0.0), 0.0)
    ret3 = sign * _safe_float(row.get("ret_3", 0.0), 0.0)
    continuation = (
        0.24 * np.tanh(trend_xgb / 0.30)
        + 0.18 * np.tanh(quant_edge / 0.25)
        + 0.12 * np.tanh(meta_edge / 0.20)
        + 0.14 * np.tanh(flow / 0.20)
        + 0.08 * np.tanh(crowd / 0.15)
        + 0.08 * np.tanh(ret1 / 0.0020)
        + 0.06 * np.tanh(ret3 / 0.0040)
        + 0.06 * conf
        + 0.05 * regime_push
        + 0.04 * np.tanh(quality / 0.006)
        - 0.05 * np.tanh(qwidth / 0.012)
        - 0.10 * tail
    )
    pullback = (
        0.28 * np.tanh(-ret1 / 0.0020)
        + 0.18 * np.tanh(-ret3 / 0.0040)
        + 0.16 * np.tanh(-flow / 0.20)
        + 0.10 * np.tanh(qwidth / 0.012)
        - 0.10 * conf
        - 0.08 * np.tanh(quality / 0.006)
        - 0.06 * regime_push
    )
    return {
        "continuation": float(continuation),
        "pullback": float(pullback),
        "conf": float(conf),
        "quality": float(quality),
        "qwidth": float(qwidth),
    }


def simulate_market(df26: pd.DataFrame, actor: GaussianActor, device: str) -> dict:
    router = DSACRouter(actor, device=device)
    numeric_cols = [c for c in df26.columns if c != "timestamp"]
    values = df26[numeric_cols].to_numpy(dtype=np.float64)
    open_np = df26["open"].to_numpy(dtype=np.float64)
    close_np = df26["close"].to_numpy(dtype=np.float64)

    balance = 1.0
    pos: str | None = None
    entry_price = 0.0
    cur_lev = 0.0
    hold_count = 0
    trades = wins = 0
    eq_curve = [1.0]
    n = len(df26)

    for i in range(n - 1):
        cp = float(close_np[i])
        next_open = float(open_np[i + 1])
        next_close = float(close_np[i + 1])
        if pos is not None:
            hold_count += 1
        pos_dict = {
            "type": pos,
            "entry_price": float(entry_price),
            "unrealized": float(_unr(pos, entry_price, cp, cur_lev)),
            "mdd": 0.0,
            "hold_norm": float(min(hold_count / 96.0, 1.0)),
            "margin_usage": float(cur_lev if pos else 0.0),
            "hold_count": float(hold_count),
        }
        features = {k: float(v) for k, v in zip(numeric_cols, values[i])}
        action_int, lev, _ = router.decide(features, pos_dict)
        lev = float(np.clip(lev, 0.0, 1.0))

        if pos is None:
            if action_int == 1 and lev > 0.0:
                pos = "LONG"
                entry_price = next_open * (1.0 + SLIP)
                cur_lev = lev
                hold_count = 0
                balance -= balance * TAKER_FEE * cur_lev
            elif action_int == 2 and lev > 0.0:
                pos = "SHORT"
                entry_price = next_open * (1.0 - SLIP)
                cur_lev = lev
                hold_count = 0
                balance -= balance * TAKER_FEE * cur_lev
        else:
            should_close = (
                action_int == 0
                or (action_int == 1 and pos == "SHORT")
                or (action_int == 2 and pos == "LONG")
            )
            if should_close:
                realized = _real(pos, entry_price, next_open, cur_lev)
                balance = balance * (1.0 + realized) - balance * TAKER_FEE * cur_lev
                trades += 1
                if realized > 0.0:
                    wins += 1
                pos = None
                entry_price = 0.0
                cur_lev = 0.0
                hold_count = 0
            else:
                delta = abs(lev - cur_lev)
                if delta > 0.05:
                    balance -= balance * TAKER_FEE * delta
                    cur_lev = lev
        eq = balance * (1.0 + _unr(pos, entry_price, next_close, cur_lev)) if pos else balance
        eq_curve.append(max(float(eq), 1e-8))

    if pos and entry_price > 0.0:
        realized = _real(pos, entry_price, float(close_np[-1]), cur_lev)
        balance = balance * (1.0 + realized) - balance * TAKER_FEE * cur_lev
        trades += 1
        if realized > 0.0:
            wins += 1

    return {
        "mode": "market_next_open",
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "wr_pct": round((wins / trades * 100.0) if trades > 0 else 0.0, 2),
        "trades": int(trades),
        "sharpe": round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
    }


def simulate_limit_overlay(df26: pd.DataFrame, actor: GaussianActor, device: str, cfg: OverlayConfig) -> dict:
    router = DSACRouter(actor, device=device)
    numeric_cols = [c for c in df26.columns if c != "timestamp"]
    values = df26[numeric_cols].to_numpy(dtype=np.float64)
    open_np = df26["open"].to_numpy(dtype=np.float64)
    high_np = df26["high"].to_numpy(dtype=np.float64)
    low_np = df26["low"].to_numpy(dtype=np.float64)
    close_np = df26["close"].to_numpy(dtype=np.float64)

    balance = 1.0
    pos: str | None = None
    entry_price = 0.0
    cur_lev = 0.0
    hold_count = 0
    trades = wins = 0
    eq_curve = [1.0]
    maker_entries = taker_fallback_entries = missed_entries = 0
    wait_releases = wait_cancels = 0
    pending: dict | None = None
    waiting: dict | None = None
    n = len(df26)

    for i in range(n - 1):
        cp = float(close_np[i])
        next_open = float(open_np[i + 1])
        next_high = float(high_np[i + 1])
        next_low = float(low_np[i + 1])
        next_close = float(close_np[i + 1])

        if pos is not None:
            hold_count += 1

        if pending is not None:
            fill = (
                (pending["side"] == "LONG" and next_low <= pending["price"])
                or (pending["side"] == "SHORT" and next_high >= pending["price"])
            )
            if fill:
                pos = pending["side"]
                entry_price = float(pending["price"])
                cur_lev = float(pending["lev"])
                hold_count = 0
                balance -= balance * MAKER_FEE * cur_lev
                maker_entries += 1
                pending = None
            elif i + 1 > pending["expire_idx"]:
                if pending["fallback"]:
                    pos = pending["side"]
                    entry_price = (
                        next_open * (1.0 + SLIP)
                        if pos == "LONG"
                        else next_open * (1.0 - SLIP)
                    )
                    cur_lev = float(pending["lev"])
                    hold_count = 0
                    balance -= balance * TAKER_FEE * cur_lev
                    taker_fallback_entries += 1
                else:
                    missed_entries += 1
                pending = None

        if waiting is not None:
            side = str(waiting["side"])
            score = _wait_score(df26.iloc[i], side, cfg)
            favorable = (
                _safe_float(df26.iloc[i].get("ret_1", 0.0), 0.0) > 0.0
                if side == "LONG"
                else _safe_float(df26.iloc[i].get("ret_1", 0.0), 0.0) < 0.0
            )
            if i > waiting["expire_idx"] or score > cfg.wait_enter_th + 0.10:
                wait_cancels += 1
                waiting = None
            elif score <= cfg.wait_release_th or favorable:
                plan = _entry_plan(df26.iloc[i], side, cfg)
                pending = {
                    "side": side,
                    "price": float(plan["limit_price"]),
                    "expire_idx": i + int(plan["ttl"]),
                    "lev": float(waiting["lev"]),
                    "fallback": bool(
                        waiting["raw_abs"] >= cfg.fallback_raw_th
                        and waiting["conf"] >= cfg.fallback_conf_th
                    ),
                }
                wait_releases += 1
                waiting = None

        pos_dict = {
            "type": pos,
            "entry_price": float(entry_price),
            "unrealized": float(_unr(pos, entry_price, cp, cur_lev)),
            "mdd": 0.0,
            "hold_norm": float(min(hold_count / 96.0, 1.0)),
            "margin_usage": float(cur_lev if pos else 0.0),
            "hold_count": float(hold_count),
        }
        features = {k: float(v) for k, v in zip(numeric_cols, values[i])}
        action_int, lev, info = router.decide(features, pos_dict)
        lev = float(np.clip(lev, 0.0, 1.0))
        raw_abs = abs(_safe_float(info.get("raw_action", 0.0), 0.0))
        conf = np.clip(_safe_float(df26.iloc[i].get("m7_confidence", 0.5), 0.5), 0.0, 1.0)

        if pos is None and pending is None and waiting is None:
            if action_int == 1 and lev > 0.0:
                score = _wait_score(df26.iloc[i], "LONG", cfg)
                if score > cfg.wait_enter_th:
                    waiting = {
                        "side": "LONG",
                        "expire_idx": i + cfg.max_wait_bars,
                        "lev": lev,
                        "raw_abs": raw_abs,
                        "conf": conf,
                    }
                else:
                    plan = _entry_plan(df26.iloc[i], "LONG", cfg)
                    pending = {
                        "side": "LONG",
                        "price": float(plan["limit_price"]),
                        "expire_idx": i + int(plan["ttl"]),
                        "lev": lev,
                        "fallback": bool(raw_abs >= cfg.fallback_raw_th and conf >= cfg.fallback_conf_th),
                    }
            elif action_int == 2 and lev > 0.0:
                score = _wait_score(df26.iloc[i], "SHORT", cfg)
                if score > cfg.wait_enter_th:
                    waiting = {
                        "side": "SHORT",
                        "expire_idx": i + cfg.max_wait_bars,
                        "lev": lev,
                        "raw_abs": raw_abs,
                        "conf": conf,
                    }
                else:
                    plan = _entry_plan(df26.iloc[i], "SHORT", cfg)
                    pending = {
                        "side": "SHORT",
                        "price": float(plan["limit_price"]),
                        "expire_idx": i + int(plan["ttl"]),
                        "lev": lev,
                        "fallback": bool(raw_abs >= cfg.fallback_raw_th and conf >= cfg.fallback_conf_th),
                    }
        elif pos is not None:
            should_close = (
                action_int == 0
                or (action_int == 1 and pos == "SHORT")
                or (action_int == 2 and pos == "LONG")
            )
            if should_close:
                realized = _real(pos, entry_price, next_open, cur_lev)
                balance = balance * (1.0 + realized) - balance * TAKER_FEE * cur_lev
                trades += 1
                if realized > 0.0:
                    wins += 1
                pos = None
                entry_price = 0.0
                cur_lev = 0.0
                hold_count = 0
            else:
                delta = abs(lev - cur_lev)
                if delta > 0.05:
                    balance -= balance * TAKER_FEE * delta
                    cur_lev = lev

        eq = balance * (1.0 + _unr(pos, entry_price, next_close, cur_lev)) if pos else balance
        eq_curve.append(max(float(eq), 1e-8))

    if pos and entry_price > 0.0:
        realized = _real(pos, entry_price, float(close_np[-1]), cur_lev)
        balance = balance * (1.0 + realized) - balance * TAKER_FEE * cur_lev
        trades += 1
        if realized > 0.0:
            wins += 1

    return {
        "mode": "limit_overlay_next_open_exit",
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "wr_pct": round((wins / trades * 100.0) if trades > 0 else 0.0, 2),
        "trades": int(trades),
        "sharpe": round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
        "maker_entry_ratio": round(
            maker_entries / max(maker_entries + taker_fallback_entries, 1),
            4,
        ),
        "missed_entries": int(missed_entries),
        "wait_releases": int(wait_releases),
        "wait_cancels": int(wait_cancels),
    }


def simulate_hybrid_overlay(df26: pd.DataFrame, actor: GaussianActor, device: str, cfg: HybridConfig) -> dict:
    router = DSACRouter(actor, device=device)
    numeric_cols = [c for c in df26.columns if c != "timestamp"]
    values = df26[numeric_cols].to_numpy(dtype=np.float64)
    open_np = df26["open"].to_numpy(dtype=np.float64)
    high_np = df26["high"].to_numpy(dtype=np.float64)
    low_np = df26["low"].to_numpy(dtype=np.float64)
    close_np = df26["close"].to_numpy(dtype=np.float64)

    balance = 1.0
    pos: str | None = None
    entry_price = 0.0
    cur_lev = 0.0
    hold_count = 0
    trades = wins = 0
    eq_curve = [1.0]
    maker_entries = taker_entries = missed_entries = 0
    wait_releases = wait_cancels = trend_taker_entries = 0
    pending: dict | None = None
    waiting: dict | None = None
    n = len(df26)

    for i in range(n - 1):
        row = df26.iloc[i]
        cp = float(close_np[i])
        next_open = float(open_np[i + 1])
        next_high = float(high_np[i + 1])
        next_low = float(low_np[i + 1])
        next_close = float(close_np[i + 1])

        if pos is not None:
            hold_count += 1

        if pending is not None:
            fill = (
                (pending["side"] == "LONG" and next_low <= pending["price"])
                or (pending["side"] == "SHORT" and next_high >= pending["price"])
            )
            if fill:
                pos = pending["side"]
                entry_price = float(pending["price"])
                cur_lev = float(pending["lev"])
                hold_count = 0
                balance -= balance * MAKER_FEE * cur_lev
                maker_entries += 1
                pending = None
            elif i + 1 > pending["expire_idx"]:
                if pending["fallback"]:
                    pos = pending["side"]
                    entry_price = (
                        next_open * (1.0 + SLIP)
                        if pos == "LONG"
                        else next_open * (1.0 - SLIP)
                    )
                    cur_lev = float(pending["lev"])
                    hold_count = 0
                    balance -= balance * TAKER_FEE * cur_lev
                    taker_entries += 1
                else:
                    missed_entries += 1
                pending = None

        if waiting is not None:
            state = _hybrid_state(row, str(waiting["side"]))
            if i > waiting["expire_idx"] or state["pullback"] > cfg.pullback_wait_th + 0.12:
                wait_cancels += 1
                waiting = None
            elif state["pullback"] <= cfg.release_th or state["continuation"] >= cfg.trend_take_th:
                side = str(waiting["side"])
                close = _safe_float(row.get("close", 0.0), 0.0)
                if side == "LONG":
                    offset = abs(_safe_float(row.get("m7_entry_long_offset", -0.0016), -0.0016))
                    reco = _safe_float(row.get("m7_entry_long_price", close * (1.0 - offset)), 0.0)
                    fallback = close * (1.0 - offset * cfg.maker_offset_mult)
                    price = 0.6 * reco + 0.4 * fallback if reco > 0 else fallback
                else:
                    offset = abs(_safe_float(row.get("m7_entry_short_offset", 0.0016), 0.0016))
                    reco = _safe_float(row.get("m7_entry_short_price", close * (1.0 + offset)), 0.0)
                    fallback = close * (1.0 + offset * cfg.maker_offset_mult)
                    price = 0.6 * reco + 0.4 * fallback if reco > 0 else fallback
                pending = {
                    "side": side,
                    "price": float(price),
                    "expire_idx": i + 1,
                    "lev": float(waiting["lev"]),
                    "fallback": bool(
                        waiting["raw_abs"] >= cfg.fallback_raw_th
                        and waiting["conf"] >= cfg.fallback_conf_th
                    ),
                }
                wait_releases += 1
                waiting = None

        pos_dict = {
            "type": pos,
            "entry_price": float(entry_price),
            "unrealized": float(_unr(pos, entry_price, cp, cur_lev)),
            "mdd": 0.0,
            "hold_norm": float(min(hold_count / 96.0, 1.0)),
            "margin_usage": float(cur_lev if pos else 0.0),
            "hold_count": float(hold_count),
        }
        features = {k: float(v) for k, v in zip(numeric_cols, values[i])}
        action_int, lev, info = router.decide(features, pos_dict)
        lev = float(np.clip(lev, 0.0, 1.0))
        raw_abs = abs(_safe_float(info.get("raw_action", 0.0), 0.0))

        if pos is None and pending is None and waiting is None:
            if action_int in (1, 2) and lev > 0.0:
                side = "LONG" if action_int == 1 else "SHORT"
                state = _hybrid_state(row, side)
                if state["continuation"] >= cfg.trend_take_th:
                    pos = side
                    entry_price = next_open * (1.0 + SLIP) if side == "LONG" else next_open * (1.0 - SLIP)
                    cur_lev = lev
                    hold_count = 0
                    balance -= balance * TAKER_FEE * cur_lev
                    taker_entries += 1
                    trend_taker_entries += 1
                elif state["pullback"] >= cfg.pullback_wait_th:
                    waiting = {
                        "side": side,
                        "expire_idx": i + cfg.max_wait_bars,
                        "lev": lev,
                        "raw_abs": raw_abs,
                        "conf": state["conf"],
                    }
                else:
                    close = _safe_float(row.get("close", 0.0), 0.0)
                    if side == "LONG":
                        offset = abs(_safe_float(row.get("m7_entry_long_offset", -0.0016), -0.0016))
                        shallow = close * (1.0 - offset * cfg.shallow_offset_mult)
                    else:
                        offset = abs(_safe_float(row.get("m7_entry_short_offset", 0.0016), 0.0016))
                        shallow = close * (1.0 + offset * cfg.shallow_offset_mult)
                    pending = {
                        "side": side,
                        "price": float(shallow),
                        "expire_idx": i + 1,
                        "lev": lev,
                        "fallback": bool(raw_abs >= cfg.fallback_raw_th and state["conf"] >= cfg.fallback_conf_th),
                    }
        elif pos is not None:
            should_close = (
                action_int == 0
                or (action_int == 1 and pos == "SHORT")
                or (action_int == 2 and pos == "LONG")
            )
            if should_close:
                realized = _real(pos, entry_price, next_open, cur_lev)
                balance = balance * (1.0 + realized) - balance * TAKER_FEE * cur_lev
                trades += 1
                if realized > 0.0:
                    wins += 1
                pos = None
                entry_price = 0.0
                cur_lev = 0.0
                hold_count = 0
            else:
                delta = abs(lev - cur_lev)
                if delta > 0.05:
                    balance -= balance * TAKER_FEE * delta
                    cur_lev = lev

        eq = balance * (1.0 + _unr(pos, entry_price, next_close, cur_lev)) if pos else balance
        eq_curve.append(max(float(eq), 1e-8))

    if pos and entry_price > 0.0:
        realized = _real(pos, entry_price, float(close_np[-1]), cur_lev)
        balance = balance * (1.0 + realized) - balance * TAKER_FEE * cur_lev
        trades += 1
        if realized > 0.0:
            wins += 1

    return {
        "mode": "hybrid_regime_aware_overlay",
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "wr_pct": round((wins / trades * 100.0) if trades > 0 else 0.0, 2),
        "trades": int(trades),
        "sharpe": round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
        "maker_entry_ratio": round(maker_entries / max(maker_entries + taker_entries, 1), 4),
        "trend_taker_entries": int(trend_taker_entries),
        "missed_entries": int(missed_entries),
        "wait_releases": int(wait_releases),
        "wait_cancels": int(wait_cancels),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    actor = GaussianActor(state_dim=int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM)).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    df26 = _load_2026_df()
    configs = [
        OverlayConfig("balanced", 0.12, 0.04, 2, 0.18, 0.70, 1.0, 1.0),
        OverlayConfig("conservative", 0.18, 0.06, 3, 0.24, 0.75, 0.9, 1.15),
        OverlayConfig("reactive", 0.10, 0.02, 1, 0.15, 0.65, 1.1, 0.9),
    ]
    hybrid_cfgs = [
        HybridConfig("hybrid_trend_first", 0.22, 0.18, 0.05, 2, 0.95, 0.45, 0.17, 0.68),
        HybridConfig("hybrid_balanced_flow", 0.18, 0.16, 0.04, 2, 1.05, 0.55, 0.16, 0.66),
        HybridConfig("hybrid_aggressive_capture", 0.15, 0.20, 0.03, 1, 1.10, 0.65, 0.14, 0.62),
        HybridConfig("report_phase2_regime_anchor", 0.20, 0.17, 0.04, 2, 1.15, 0.52, 0.16, 0.64),
    ]

    market = simulate_market(df26, copy.deepcopy(actor), device)
    results = []
    for cfg in configs:
        overlay = simulate_limit_overlay(df26, copy.deepcopy(actor), device, cfg)
        overlay["delta_vs_market_pct"] = round(overlay["pnl_pct"] - market["pnl_pct"], 4)
        results.append({"config": asdict(cfg), "overlay": overlay})
        print(cfg.name, market, overlay)
    for cfg in hybrid_cfgs:
        overlay = simulate_hybrid_overlay(df26, copy.deepcopy(actor), device, cfg)
        overlay["delta_vs_market_pct"] = round(overlay["pnl_pct"] - market["pnl_pct"], 4)
        results.append({"config": asdict(cfg), "overlay": overlay})
        print(cfg.name, market, overlay)

    best = max(results, key=lambda x: x["overlay"]["delta_vs_market_pct"])
    report = {
        "checkpoint": CKPT,
        "checkpoint_best_val_pnl": float(ckpt.get("best_pnl", 0.0)),
        "checkpoint_epoch": int(ckpt.get("epoch", 0)),
        "data_period": f"{df26['timestamp'].min()} -> {df26['timestamp'].max()}",
        "data_rows": int(len(df26)),
        "market_baseline": market,
        "overlays": results,
        "best_overlay": best,
        "note": "Canonical comparison on the same next-open closed-loop engine as eval_2026_oos.py.",
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"SAVED {OUT_JSON}")


if __name__ == "__main__":
    main()
