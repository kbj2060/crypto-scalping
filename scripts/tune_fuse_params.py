#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.seven_model_ensemble import SevenModelEnsemble
from ensemble.train_rl_agent import (
    REGIME_COLS,
    STATE_ALPHA,
    STATE_CONF,
    STATE_ELITE,
    STATE_PRED,
    STATE_SYNTH,
)
from ensemble.train_rl_dsac_agent import (
    DSAC_STATE_DIM,
    GaussianActor,
    SACRouter as DSACRouter,
)

ANNUAL_FACTOR_5M = math.sqrt(365 * 24 * 12)


@dataclass
class SliceMetrics:
    score: float
    total_return: float
    mdd: float
    sharpe: float
    win_rate: float
    trades: int


def _resolve_dsac_ckpt(custom_path: str | None = None) -> str:
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    candidates.extend(
        [
            str(ROOT / "data/ensemble/ckpt/best_dsac_agents.pth"),
            str(ROOT / "data/ensemble/ckpt/dsac_checkpoint.pth"),
        ]
    )
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"DSAC 체크포인트 없음: {candidates}")


def _load_merged_frame(
    feature_csv: str,
    rl_csv: str,
    max_rows: int,
    step: int,
) -> pd.DataFrame:
    feature_path = ROOT / feature_csv
    rl_path = ROOT / rl_csv
    if not feature_path.exists():
        raise FileNotFoundError(f"feature csv not found: {feature_path}")
    if not rl_path.exists():
        raise FileNotFoundError(f"rl csv not found: {rl_path}")

    print(f"[LOAD] {feature_path}")
    fdf = pd.read_csv(feature_path)
    print(f"[LOAD] {rl_path}")
    rdf = pd.read_csv(rl_path)

    fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], errors="coerce")
    rdf["timestamp"] = pd.to_datetime(rdf["timestamp"], errors="coerce")
    fdf = fdf.dropna(subset=["timestamp"]).sort_values("timestamp")
    rdf = rdf.dropna(subset=["timestamp"]).sort_values("timestamp")

    rl_cols = list(
        dict.fromkeys(
            ["timestamp", "close", "garch_vol_z", "log_return"]
            + STATE_PRED
            + STATE_CONF
            + STATE_ELITE
            + STATE_ALPHA
            + STATE_SYNTH
            + REGIME_COLS
        )
    )
    rl_cols = [c for c in rl_cols if c in rdf.columns]

    merged = fdf.merge(rdf[rl_cols], on="timestamp", how="inner", suffixes=("", "_rl"))
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # prefer RL columns for DSAC input where available
    for col in list(dict.fromkeys(["close", "garch_vol_z", "log_return"] + STATE_PRED + STATE_CONF + STATE_ELITE + STATE_ALPHA + STATE_SYNTH + REGIME_COLS)):
        rl_col = f"{col}_rl"
        if rl_col in merged.columns:
            merged[col] = pd.to_numeric(merged[rl_col], errors="coerce")
        elif col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
        else:
            merged[col] = 0.0

    merged = merged.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if max_rows > 0 and len(merged) > max_rows:
        merged = merged.iloc[-max_rows:].reset_index(drop=True)
    if step > 1:
        merged = merged.iloc[::step].reset_index(drop=True)

    print(f"[DATA] merged rows={len(merged):,} cols={len(merged.columns)}")
    return merged


def _compute_m7_signals(df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    hub = SevenModelEnsemble()
    avail = sum(
        [
            int(hub.trend_xgb.available),
            int(hub.multi_target.available),
            int(hub.quantile.available),
            int(hub.gmm.available),
            int(hub.hdbscan.available),
            int(hub.isolation.available),
            int(hub.vae.available),
        ]
    )
    print(f"[M7] available={avail}/7")

    outs: list[pd.DataFrame] = []
    for start in range(0, len(df), max(1, chunk_size)):
        end = min(start + chunk_size, len(df))
        part = df.iloc[start:end]
        out = hub.predict_batch(part)
        out = out.reset_index(drop=True)
        outs.append(out)
        if (start // chunk_size) % 4 == 0:
            print(f"[M7] {end:,}/{len(df):,}")

    m7 = pd.concat(outs, ignore_index=True)
    for col in m7.columns:
        m7[col] = pd.to_numeric(m7[col], errors="coerce").fillna(0.0)
    return m7


def _compute_dsac_stream(df: pd.DataFrame, ckpt_path: str, device: str) -> pd.DataFrame:
    print(f"[DSAC] load actor from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "actor" not in ckpt:
        raise KeyError(f"actor not found in checkpoint: {ckpt_path}")

    actor = GaussianActor(state_dim=DSAC_STATE_DIM).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    router = DSACRouter(actor, device=device)

    n = len(df)
    dsac_action = np.zeros(n, dtype=np.int64)
    dsac_kelly = np.zeros(n, dtype=np.float64)
    dsac_score = np.zeros(n, dtype=np.float64)
    dsac_raw = np.zeros(n, dtype=np.float64)

    pos: str | None = None
    entry_price = 0.0
    hold_count = 0
    cur_eq = 1.0
    peak_eq = 1.0

    for i in range(n):
        row = df.iloc[i]
        close = float(row.get("close", 0.0))

        unr = 0.0
        if pos is not None and entry_price > 0 and close > 0:
            if pos == "LONG":
                unr = (close - entry_price) / entry_price
            else:
                unr = (entry_price - close) / entry_price
            cur_eq = 1.0 + unr
            peak_eq = max(peak_eq, cur_eq)
        else:
            cur_eq = 1.0
            peak_eq = 1.0

        pos_dict = {
            "type": pos,
            "entry_price": entry_price,
            "unrealized": float(np.tanh(unr / 0.02)),
            "mdd": float(np.clip(min((cur_eq / max(peak_eq, 1e-8)) - 1.0, 0.0) / 0.05, -1.0, 1.0)),
            "hold_norm": min(hold_count / 144.0, 1.0),
        }

        features: dict[str, float] = {}
        for col in STATE_PRED + STATE_CONF + STATE_ELITE + STATE_ALPHA + STATE_SYNTH + REGIME_COLS + ["close"]:
            features[col] = float(row.get(col, 0.0))

        action, lev, info = router.decide(features, pos_dict)
        dsac_action[i] = int(action)
        dsac_kelly[i] = float(lev)
        dsac_score[i] = float(info.get("score", abs(info.get("raw_action", 0.0))))
        dsac_raw[i] = float(info.get("raw_action", 0.0))

        # lightweight DSAC-local position update for next state context
        if action == 1 and pos is None:
            pos, entry_price, hold_count = "LONG", close, 0
            cur_eq = peak_eq = 1.0
        elif action == 2 and pos is None:
            pos, entry_price, hold_count = "SHORT", close, 0
            cur_eq = peak_eq = 1.0
        elif action == 0 and pos is not None:
            pos, entry_price, hold_count = None, 0.0, 0
            cur_eq = peak_eq = 1.0
        elif pos is not None:
            hold_count += 1

        if i % 25000 == 0 and i > 0:
            print(f"[DSAC] {i:,}/{n:,}")

    return pd.DataFrame(
        {
            "dsac_action": dsac_action,
            "dsac_kelly": dsac_kelly,
            "dsac_score": dsac_score,
            "dsac_raw_action": dsac_raw,
        }
    )


def _trend_signal_from_row(row: dict[str, float]) -> dict[str, float]:
    p_dn = float(np.clip(row.get("m7_trend_xgb_dn", 0.0), 0.0, 1.0))
    p_fl = float(np.clip(row.get("m7_trend_xgb_fl", 0.0), 0.0, 1.0))
    p_up = float(np.clip(row.get("m7_trend_xgb_up", 0.0), 0.0, 1.0))
    s = p_dn + p_fl + p_up
    if s <= 1e-12:
        p_dn = p_fl = p_up = 1.0 / 3.0
    else:
        p_dn, p_fl, p_up = p_dn / s, p_fl / s, p_up / s

    t_dir = int(np.argmax([p_dn, p_fl, p_up]))
    m7_action = int(np.clip(round(row.get("m7_action", 0.0)), -1, 1))
    if m7_action > 0:
        t_dir = 2
    elif m7_action < 0:
        t_dir = 0

    m7_conf = float(np.clip(row.get("m7_confidence", 0.0), 0.0, 1.0))
    m7_gate_block = 1 if row.get("m7_gate_block", 0.0) >= 0.5 else 0
    strength = float(np.clip(m7_conf, 0.0, 1.0))
    rev_prob = float(np.clip((1.0 - m7_conf) * 0.70 + (0.30 if m7_gate_block else 0.0), 0.0, 1.0))

    return {
        "trend_dir": float(t_dir),
        "strength": strength,
        "rev_prob": rev_prob,
        "prob_dn": p_dn,
        "prob_flat": p_fl,
        "prob_up": p_up,
        "m7_confidence": m7_conf,
        "m7_action": float(m7_action),
        "m7_size": float(np.clip(row.get("m7_size", 0.0), 0.0, 1.0)),
        "m7_gate_block": float(m7_gate_block),
        "m7_quality_pred": float(row.get("m7_quality_pred", 0.0)),
        "m7_target_hold": float(max(0.0, row.get("m7_target_hold", 0.0))),
        "m7_gmm_vol_rank": float(np.clip(row.get("m7_gmm_vol_rank", 0.5), 0.0, 1.0)),
        "m7_iso_anom": 1.0 if row.get("m7_iso_anom", 0.0) >= 0.5 else 0.0,
        "m7_vae_anom": 1.0 if row.get("m7_vae_anom", 0.0) >= 0.5 else 0.0,
        "m7_expected_ret": float(row.get("m7_expected_ret", 0.0)),
        "m7_tail_risk": float(row.get("m7_tail_risk", 0.0)),
        "m7_composite_score": float(row.get("m7_composite_score", 0.0)),
    }


def _fuse_decision(
    dsac_action: int,
    dsac_raw_action: float,
    dsac_kelly: float,
    dsac_score_in: float,
    trend: dict[str, float],
    regime: dict[str, float],
    garch_vol_z: float,
    cur_side: int,
    hold_count: int,
    p: dict[str, float],
) -> tuple[int, float, str, float, float, float, float, bool, bool]:
    # DSAC
    dsac_side = 1 if dsac_action == 1 else (-1 if dsac_action == 2 else 0)
    base_kelly = float(np.clip(dsac_kelly, 0.0, 1.0))
    dsac_score = float(np.clip(max(abs(dsac_raw_action), abs(dsac_score_in), base_kelly), 0.0, 1.0))

    # M7
    t_dir = int(trend.get("trend_dir", 1.0))
    t_str = float(np.clip(trend.get("strength", 0.0), 0.0, 1.0))
    t_rev = float(np.clip(trend.get("rev_prob", 0.0), 0.0, 1.0))
    m7_action_hint = int(np.clip(round(trend.get("m7_action", 0.0)), -1, 1))
    m7_side = 1 if t_dir == 2 else (-1 if t_dir == 0 else 0)
    if m7_action_hint != 0:
        m7_side = m7_action_hint

    m7_conf = float(np.clip(trend.get("m7_confidence", t_str), 0.0, 1.0))
    m7_size = float(np.clip(trend.get("m7_size", 0.0), 0.0, 1.0))
    m7_quality = float(trend.get("m7_quality_pred", 0.0))
    m7_target_hold = int(max(0, round(float(trend.get("m7_target_hold", 0.0)))))
    m7_vol_rank = float(np.clip(trend.get("m7_gmm_vol_rank", 0.5), 0.0, 1.0))
    m7_iso_anom = bool(float(trend.get("m7_iso_anom", 0.0)) >= 0.5)
    m7_vae_anom = bool(float(trend.get("m7_vae_anom", 0.0)) >= 0.5)
    m7_gate_block = bool(float(trend.get("m7_gate_block", 0.0)) >= 0.5)
    m7_expected_ret = float(trend.get("m7_expected_ret", 0.0))
    m7_tail_risk = abs(float(trend.get("m7_tail_risk", 0.0)))
    m7_composite = float(np.clip(trend.get("m7_composite_score", 0.0), -1.0, 1.0))

    hard_anomaly = m7_gate_block or (m7_iso_anom and m7_vae_anom)
    soft_anomaly = (m7_iso_anom ^ m7_vae_anom) and not hard_anomaly

    m7_score = float(np.clip(m7_conf * (0.60 + 0.40 * m7_size), 0.0, 1.0))
    if m7_side == 0:
        m7_score *= 0.65

    dsac_signed = float(dsac_side * dsac_score)
    m7_signed = float(m7_side * m7_score)
    fused_signed = p["dsac_weight"] * dsac_signed + p["m7_weight"] * m7_signed
    fused_abs = abs(fused_signed)
    fused_side = 1 if fused_signed > p["flip_min_margin"] else (-1 if fused_signed < -p["flip_min_margin"] else 0)

    quality_factor = float(np.clip(1.0 + np.tanh(m7_quality * 12.0) * 0.20, 0.75, 1.25))
    ret_factor = float(np.clip(0.90 + np.tanh(abs(m7_expected_ret) * 250.0) * 0.20, 0.90, 1.10))
    tail_factor = float(np.clip(1.0 - min(m7_tail_risk * 180.0, 0.45), 0.55, 1.00))

    vol_factor = 1.0
    if m7_vol_rank >= 0.85:
        vol_factor *= 0.55
    elif m7_vol_rank >= 0.70:
        vol_factor *= 0.75
    elif m7_vol_rank <= 0.20:
        vol_factor *= 1.08
    if garch_vol_z >= 2.0:
        vol_factor *= 0.75
    elif garch_vol_z >= 1.2:
        vol_factor *= 0.88

    regime_factor = float(
        np.clip(
            1.0
            - 0.25 * float(regime.get("regime_chop", 0.0))
            - 0.20 * float(regime.get("regime_whipsaw", 0.0))
            + (0.10 if fused_side > 0 else 0.0) * float(regime.get("regime_bull", 0.0))
            + (0.10 if fused_side < 0 else 0.0) * float(regime.get("regime_bear", 0.0)),
            0.60,
            1.15,
        )
    )

    rev_factor = 0.60 if t_rev >= p["rev_reduce_prob"] else 1.0
    chop_factor = 0.80 if (t_dir == 1 and t_str >= p["chop_strength"]) else 1.0
    anom_factor = 0.0 if hard_anomaly else (0.50 if soft_anomaly else 1.0)

    agree_factor = 1.0
    if dsac_side != 0 and m7_side != 0:
        if dsac_side == m7_side:
            agree_factor *= 1.0 + 0.18 * min(dsac_score, m7_score)
        else:
            agree_factor *= max(0.45, 1.0 - 0.35 * m7_conf)
    elif dsac_side == 0 and m7_side != 0:
        agree_factor *= 0.90
    elif dsac_side != 0 and m7_side == 0:
        agree_factor *= 0.95

    kelly_pre = float(np.clip(0.60 * base_kelly + 0.40 * m7_size, 0.0, 1.0))
    unified_kelly = float(
        np.clip(
            kelly_pre
            * quality_factor
            * ret_factor
            * tail_factor
            * vol_factor
            * regime_factor
            * rev_factor
            * chop_factor
            * anom_factor
            * agree_factor,
            0.0,
            1.0,
        )
    )

    final_side = dsac_side
    source = "DSAC_BASE"

    if cur_side == 0:
        if hard_anomaly:
            final_side = 0
            source = "FUSE_BLOCK_ANOMALY"
        elif fused_abs >= p["min_enter_score"] and fused_side != 0:
            final_side = fused_side
            source = "FUSE_ENTER"
        elif dsac_side != 0 and dsac_score >= (p["min_enter_score"] + 0.12) and m7_score < 0.20:
            final_side = dsac_side
            source = "DSAC_HIGH_CONV_ENTER"
        elif m7_side != 0 and m7_score >= (p["min_enter_score"] + 0.15) and dsac_side == 0:
            final_side = m7_side
            source = "M7_SOLO_ENTER"
        else:
            final_side = 0
            source = "FUSE_HOLD_LOW_SCORE"

        if final_side != 0 and unified_kelly < p["min_live_kelly"]:
            final_side = 0
            source = "FUSE_HOLD_LOW_KELLY"
    else:
        final_side = cur_side
        source = "IN_POS_HOLD"

        if dsac_side == 0:
            final_side = 0
            source = "DSAC_EXIT"
        elif dsac_side != cur_side:
            final_side = 0
            source = "DSAC_REVERSE_EXIT"

        if final_side != 0 and m7_side != 0 and m7_side != cur_side and m7_conf >= p["veto_strength"]:
            final_side = 0
            source = "M7_OPPOSE_EXIT"

        if final_side != 0 and hard_anomaly and (p["anomaly_force_exit"] or m7_conf >= 0.75 or m7_side != cur_side):
            final_side = 0
            source = "M7_ANOMALY_EXIT"

        if final_side != 0 and m7_target_hold > 0 and hold_count >= m7_target_hold and m7_composite <= p["hold_exit_margin"]:
            final_side = 0
            source = "M7_TARGET_HOLD_EXIT"

        if final_side != 0 and t_rev >= max(p["rev_reduce_prob"], 0.85) and m7_conf >= 0.65:
            unified_kelly = float(np.clip(unified_kelly * 0.55, 0.0, 1.0))
            source = "M7_REV_REDUCE"

    if final_side == 0:
        unified_kelly = 0.0

    return final_side, unified_kelly, source, dsac_score, m7_score, fused_abs, fused_signed, hard_anomaly, soft_anomaly


def _simulate(arr: dict[str, np.ndarray], start: int, end: int, params: dict[str, float], fee_rate: float) -> SliceMetrics:
    if end - start <= 5:
        return SliceMetrics(-1e9, 0.0, 0.0, 0.0, 0.0, 0)

    close = arr["close"]
    n = len(close)
    start = max(1, start)
    end = min(end, n)

    equity = 1.0
    eq_curve = [equity]

    pos_side = 0  # -1 short, 0 flat, +1 long
    pos_lev = 0.0
    hold_count = 0
    entry_price = 0.0
    open_trade_pnl = 0.0

    trades = 0
    wins = 0

    for i in range(start, end):
        prev_close = float(close[i - 1])
        cur_close = float(close[i])
        if prev_close <= 0 or cur_close <= 0:
            eq_curve.append(equity)
            continue

        bar_ret = (cur_close / prev_close) - 1.0

        # previous exposure PnL first
        if pos_side != 0 and pos_lev > 0:
            pnl = float(pos_side) * float(pos_lev) * float(bar_ret)
            equity *= max(1e-8, 1.0 + pnl)
            open_trade_pnl += pnl

        row = {k: float(v[i]) for k, v in arr.items() if k != "timestamp"}
        trend = _trend_signal_from_row(row)
        regime = {
            "regime_chop": float(arr["regime_chop"][i]),
            "regime_whipsaw": float(arr["regime_whipsaw"][i]),
            "regime_bull": float(arr["regime_bull"][i]),
            "regime_bear": float(arr["regime_bear"][i]),
        }

        final_side, unified_kelly, _source, *_ = _fuse_decision(
            dsac_action=int(arr["dsac_action"][i]),
            dsac_raw_action=float(arr["dsac_raw_action"][i]),
            dsac_kelly=float(arr["dsac_kelly"][i]),
            dsac_score_in=float(arr["dsac_score"][i]),
            trend=trend,
            regime=regime,
            garch_vol_z=float(arr["garch_vol_z"][i]),
            cur_side=pos_side,
            hold_count=hold_count,
            p=params,
        )

        prev_side = pos_side
        prev_lev = pos_lev

        if final_side == 0 and prev_side != 0:
            # exit
            fee = fee_rate * max(prev_lev, 0.05)
            equity *= max(1e-8, 1.0 - fee)
            trades += 1
            if open_trade_pnl > 0:
                wins += 1
            open_trade_pnl = 0.0
            pos_side = 0
            pos_lev = 0.0
            hold_count = 0
            entry_price = 0.0
        elif final_side != 0 and prev_side == 0:
            # entry
            pos_side = final_side
            pos_lev = float(unified_kelly)
            hold_count = 0
            entry_price = cur_close
            fee = fee_rate * max(pos_lev, 0.05)
            equity *= max(1e-8, 1.0 - fee)
            open_trade_pnl = 0.0
        elif final_side != 0 and prev_side != 0:
            if final_side != prev_side:
                # close + open
                fee = fee_rate * (max(prev_lev, 0.05) + max(float(unified_kelly), 0.05))
                equity *= max(1e-8, 1.0 - fee)
                trades += 1
                if open_trade_pnl > 0:
                    wins += 1
                open_trade_pnl = 0.0
                pos_side = final_side
                pos_lev = float(unified_kelly)
                hold_count = 0
                entry_price = cur_close
            else:
                pos_side = prev_side
                pos_lev = float(unified_kelly)
                hold_count += 1
                entry_price = entry_price if entry_price > 0 else cur_close

        equity = max(equity, 1e-8)
        eq_curve.append(equity)

    if pos_side != 0:
        # terminal close for fair trade stats
        fee = fee_rate * max(pos_lev, 0.05)
        equity *= max(1e-8, 1.0 - fee)
        trades += 1
        if open_trade_pnl > 0:
            wins += 1
        eq_curve[-1] = equity

    eq = np.asarray(eq_curve, dtype=np.float64)
    total_return = float(eq[-1] - 1.0)
    run_max = np.maximum.accumulate(eq)
    drawdown = eq / np.maximum(run_max, 1e-12) - 1.0
    mdd = float(np.min(drawdown)) if len(drawdown) else 0.0

    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    if len(rets) < 3 or np.std(rets) < 1e-12:
        sharpe = 0.0
    else:
        sharpe = float(np.mean(rets) / np.std(rets) * ANNUAL_FACTOR_5M)

    win_rate = float(wins / trades) if trades > 0 else 0.0

    score = (
        total_return * 100.0
        + sharpe * 1.5
        + win_rate * 10.0
        - abs(min(mdd, 0.0)) * 130.0
        - trades * 0.01
    )
    return SliceMetrics(score, total_return, mdd, sharpe, win_rate, trades)


def _sample_params(rng: random.Random) -> dict[str, float]:
    dsac_weight = rng.uniform(0.35, 0.75)
    m7_weight = 1.0 - dsac_weight
    return {
        "dsac_weight": dsac_weight,
        "m7_weight": m7_weight,
        "min_enter_score": rng.uniform(0.12, 0.40),
        "flip_min_margin": rng.uniform(0.04, 0.20),
        "min_live_kelly": rng.uniform(0.02, 0.20),
        "hold_exit_margin": rng.uniform(-0.05, 0.12),
        "anomaly_force_exit": 1.0 if rng.random() < 0.35 else 0.0,
        "veto_strength": rng.uniform(0.55, 0.90),
        "chop_strength": rng.uniform(0.20, 0.60),
        "rev_reduce_prob": rng.uniform(0.55, 0.90),
    }


def _optimize_slice(
    arr: dict[str, np.ndarray],
    start: int,
    end: int,
    n_trials: int,
    fee_rate: float,
    seed: int,
) -> tuple[dict[str, float], SliceMetrics]:
    rng = random.Random(seed)
    best_p: dict[str, float] | None = None
    best_m: SliceMetrics | None = None

    for t in range(1, n_trials + 1):
        p = _sample_params(rng)
        m = _simulate(arr, start, end, p, fee_rate)
        if best_m is None or m.score > best_m.score:
            best_p, best_m = p, m
        if t % max(1, n_trials // 5) == 0:
            print(f"    [trial {t:>4}/{n_trials}] best_score={best_m.score:.3f} ret={best_m.total_return*100:.2f}% mdd={best_m.mdd*100:.2f}%")

    assert best_p is not None and best_m is not None
    return best_p, best_m


def _build_signal_cache(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    cache_path = ROOT / args.signals_cache
    if cache_path.exists() and not args.rebuild_signals:
        print(f"[CACHE] load {cache_path}")
        return pd.read_pickle(cache_path)

    m7 = _compute_m7_signals(df, chunk_size=args.m7_chunk)
    ckpt_path = _resolve_dsac_ckpt(args.dsac_ckpt)
    dsac = _compute_dsac_stream(df, ckpt_path=ckpt_path, device=args.device)

    out = pd.concat([df.reset_index(drop=True), m7.reset_index(drop=True), dsac.reset_index(drop=True)], axis=1)

    os.makedirs(cache_path.parent, exist_ok=True)
    out.to_pickle(cache_path)
    print(f"[CACHE] saved {cache_path}")
    return out


def _to_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    need_cols = [
        "close",
        "garch_vol_z",
        "regime_chop",
        "regime_whipsaw",
        "regime_bull",
        "regime_bear",
        "dsac_action",
        "dsac_kelly",
        "dsac_score",
        "dsac_raw_action",
        "m7_trend_xgb_dn",
        "m7_trend_xgb_fl",
        "m7_trend_xgb_up",
        "m7_confidence",
        "m7_action",
        "m7_size",
        "m7_gate_block",
        "m7_quality_pred",
        "m7_target_hold",
        "m7_gmm_vol_rank",
        "m7_iso_anom",
        "m7_vae_anom",
        "m7_expected_ret",
        "m7_tail_risk",
        "m7_composite_score",
    ]

    arr: dict[str, np.ndarray] = {}
    for c in need_cols:
        if c not in df.columns:
            arr[c] = np.zeros(len(df), dtype=np.float64)
        else:
            arr[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    arr["dsac_action"] = arr["dsac_action"].astype(np.int64)
    return arr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward tuning for DSAC+M7 fusion parameters")
    p.add_argument("--feature-csv", default="data/training_features_5m.csv")
    p.add_argument("--rl-csv", default="data/rl_training_data_full.csv")
    p.add_argument("--signals-cache", default="data/ensemble/fuse_tuning_signals.pkl")
    p.add_argument("--rebuild-signals", action="store_true")
    p.add_argument("--dsac-ckpt", default="")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--max-rows", type=int, default=60000)
    p.add_argument("--step", type=int, default=2, help="Use every Nth row for faster tuning")
    p.add_argument("--m7-chunk", type=int, default=15000)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--n-trials", type=int, default=120)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="data/ensemble/fuse_walkforward_best.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    df = _load_merged_frame(
        feature_csv=args.feature_csv,
        rl_csv=args.rl_csv,
        max_rows=args.max_rows,
        step=max(1, args.step),
    )

    replay = _build_signal_cache(df, args)
    arr = _to_arrays(replay)

    n = len(replay)
    if n < 3000:
        raise RuntimeError(f"rows too small for tuning: {n}")

    fold_size = n // (args.n_folds + 1)
    if fold_size < 500:
        raise RuntimeError(f"fold_size too small: {fold_size}")

    fold_results: list[dict[str, Any]] = []

    for fold in range(args.n_folds):
        train_start = 0
        train_end = fold_size * (fold + 1)
        val_start = train_end
        val_end = min(val_start + fold_size, n)
        if val_end - val_start < 300:
            break

        print(f"\n[FOLD {fold+1}] train=({train_start}:{train_end}) val=({val_start}:{val_end})")
        best_p, train_m = _optimize_slice(
            arr=arr,
            start=train_start,
            end=train_end,
            n_trials=args.n_trials,
            fee_rate=args.fee_rate,
            seed=args.seed + fold * 997,
        )
        val_m = _simulate(arr, val_start, val_end, best_p, args.fee_rate)
        print(
            f"  [BEST@TRAIN] score={train_m.score:.3f} ret={train_m.total_return*100:.2f}% mdd={train_m.mdd*100:.2f}% sharpe={train_m.sharpe:.2f}"
        )
        print(
            f"  [VAL]        score={val_m.score:.3f} ret={val_m.total_return*100:.2f}% mdd={val_m.mdd*100:.2f}% sharpe={val_m.sharpe:.2f} trades={val_m.trades}"
        )
        fold_results.append(
            {
                "fold": fold + 1,
                "train_range": [train_start, train_end],
                "val_range": [val_start, val_end],
                "best_params": best_p,
                "train_metrics": train_m.__dict__,
                "val_metrics": val_m.__dict__,
            }
        )

    if not fold_results:
        raise RuntimeError("no valid folds generated")

    best_fold = max(fold_results, key=lambda x: x["val_metrics"]["score"])
    best_params = dict(best_fold["best_params"])

    full_m = _simulate(arr, 1, n, best_params, args.fee_rate)
    print("\n[FINAL ON FULL]")
    print(
        f"score={full_m.score:.3f} ret={full_m.total_return*100:.2f}% mdd={full_m.mdd*100:.2f}% sharpe={full_m.sharpe:.2f} win={full_m.win_rate*100:.1f}% trades={full_m.trades}"
    )

    payload = {
        "rows": int(n),
        "feature_csv": args.feature_csv,
        "rl_csv": args.rl_csv,
        "signals_cache": args.signals_cache,
        "max_rows": args.max_rows,
        "step": args.step,
        "n_folds": args.n_folds,
        "n_trials": args.n_trials,
        "fee_rate": args.fee_rate,
        "seed": args.seed,
        "best_fold": best_fold["fold"],
        "best_params": best_params,
        "full_metrics": full_m.__dict__,
        "folds": fold_results,
        "env_export": {
            "FUSE_DSAC_WEIGHT": round(float(best_params["dsac_weight"]), 6),
            "FUSE_M7_WEIGHT": round(float(best_params["m7_weight"]), 6),
            "FUSE_MIN_ENTER_SCORE": round(float(best_params["min_enter_score"]), 6),
            "FUSE_FLIP_MIN_MARGIN": round(float(best_params["flip_min_margin"]), 6),
            "FUSE_MIN_LIVE_KELLY": round(float(best_params["min_live_kelly"]), 6),
            "FUSE_HOLD_EXIT_MARGIN": round(float(best_params["hold_exit_margin"]), 6),
            "FUSE_ANOMALY_FORCE_EXIT": "1" if float(best_params["anomaly_force_exit"]) >= 0.5 else "0",
            "TREND_VETO_STRENGTH": round(float(best_params["veto_strength"]), 6),
            "TREND_CHOP_STRENGTH": round(float(best_params["chop_strength"]), 6),
            "TREND_REV_REDUCE_PROB": round(float(best_params["rev_reduce_prob"]), 6),
        },
    }

    out_path = ROOT / args.output
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    print(f"\n[SAVED] {out_path}")
    print("[EXPORT]")
    for k, v in payload["env_export"].items():
        print(f"export {k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
