#!/usr/bin/env python3
"""Test the two Omega6 layers that were NOT used in the frozen v2 winner
(scripts/replay_omega6_v2_oos_freeze_20260704.py::FROZEN used fixed sizing, no L4/L6):

  L4 risk sizing sidecar (tmp/causal_regen_20260516/omega6_risk_sidecar_20260703/risk_sidecar.pkl,
  a HistGradientBoostingRegressor trained train-only on 2025-01-02..09-30) -- dynamic
  margin_fraction/leverage per bar instead of the frozen config's fixed margin=0.30/leverage=2.0.

  L6 event-risk governor (macro veto window + shock haircut) -- pure calendar/feature function,
  ported from trading_bot_modules/omega6_live.py, using the tape's own jump_flag/evt_tail_flag/
  jump_z columns (no lookahead: same columns used causally at inference time).

Same entry logic as the frozen winner (persistence_bars=3, quality_threshold=0.58, ATR
tp=15x/sl=5x, cooldown=12) -- only the sizing/governor layer changes. Validation-only
(2025-10-01..12-31); OOS is not touched here. If a variant beats the frozen winner while still
passing all pre-registered gates, it gets exactly one OOS look afterward, same protocol as
before -- never touched speculatively.
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402

SIDECAR_PATH = ROOT / "tmp/causal_regen_20260516/omega6_risk_sidecar_20260703/risk_sidecar.pkl"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega6_v2_l4l6_20260704"

# L6 constants, copied from trading_bot_modules/omega6_live.py (pure calendar function)
L6_MACRO_PRE_MINUTES = 30
L6_MACRO_POST_MINUTES = 120
L6_SHOCK_NOTIONAL_SCALE = 0.50
L6_SHOCK_JUMP_Z_THRESHOLD = 3.0
L6_SHOCK_RET_1H_THRESHOLD = 0.030
L6_SHOCK_RET_4H_THRESHOLD = 0.040
L6_FOMC_DECISION_DATES = {
    2025: ("2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10"),
    2026: ("2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"),
}

FROZEN_KW = dict(
    persistence_bars=3,
    quality_threshold=0.58,
    tp_mode="atr_scaled",
    tp_atr_mult=15.0,
    sl_atr_mult=5.0,
    cooldown_bars=12,
    fixed_margin=0.30,
    fixed_leverage=2.0,
)

L4_BASELINE_NOTIONAL = 0.45
L4_BASELINE_LEVERAGE = 2.0
L5_BASE_TP_PRICE_MOVE = 0.026
L5_BASE_SL_PRICE_MOVE = 0.014


def _weekday_on_or_after(year: int, month: int, day: int) -> pd.Timestamp:
    out = pd.Timestamp(year=year, month=month, day=day)
    while out.weekday() >= 5:
        out += pd.Timedelta(days=1)
    return out


def _nth_weekday(year: int, month: int, n: int) -> pd.Timestamp:
    out = pd.Timestamp(year=year, month=month, day=1)
    count = 0
    while True:
        if out.weekday() < 5:
            count += 1
            if count == int(n):
                return out
        out += pd.Timedelta(days=1)


def _first_friday(year: int, month: int) -> pd.Timestamp:
    out = pd.Timestamp(year=year, month=month, day=1)
    while out.weekday() != 4:
        out += pd.Timedelta(days=1)
    return out


def _et_to_utc_naive(day: pd.Timestamp, hour: int, minute: int) -> pd.Timestamp:
    ny = ZoneInfo("America/New_York")
    dt = datetime(int(day.year), int(day.month), int(day.day), int(hour), int(minute), tzinfo=ny)
    return pd.Timestamp(dt.astimezone(ZoneInfo("UTC")).replace(tzinfo=None))


def _macro_events_for_year(year: int) -> list[pd.Timestamp]:
    events = []
    for month in range(1, 13):
        events.append(_et_to_utc_naive(_first_friday(year, month), 8, 30))
        events.append(_et_to_utc_naive(_nth_weekday(year, month, 1), 10, 0))
        events.append(_et_to_utc_naive(_nth_weekday(year, month, 3), 10, 0))
        events.append(_et_to_utc_naive(_weekday_on_or_after(year, month, 23), 9, 45))
    for raw in L6_FOMC_DECISION_DATES.get(int(year), ()):
        events.append(_et_to_utc_naive(pd.Timestamp(raw), 14, 0))
    return events


def build_macro_veto_mask(timestamps: pd.Series) -> np.ndarray:
    years = sorted({t.year for t in timestamps} | {t.year - 1 for t in timestamps} | {t.year + 1 for t in timestamps})
    all_events: list[pd.Timestamp] = []
    for y in years:
        all_events.extend(_macro_events_for_year(y))
    events_arr = pd.Series(all_events).sort_values().to_numpy()
    veto = np.zeros(len(timestamps), dtype=bool)
    ts_arr = timestamps.to_numpy()
    for i, ts in enumerate(ts_arr):
        ts = pd.Timestamp(ts)
        for ev in events_arr:
            ev = pd.Timestamp(ev)
            if ev - pd.Timedelta(minutes=L6_MACRO_PRE_MINUTES) <= ts <= ev + pd.Timedelta(minutes=L6_MACRO_POST_MINUTES):
                veto[i] = True
                break
    return veto


def build_shock_haircut_mask(sub: pd.DataFrame) -> np.ndarray:
    close = sub["close"].to_numpy(dtype=np.float64)
    n = len(close)
    ret_1h = np.zeros(n)
    ret_4h = np.zeros(n)
    ret_1h[12:] = close[12:] / close[:-12] - 1.0
    ret_4h[48:] = close[48:] / close[:-48] - 1.0
    jump_flag = sub["jump_flag"].to_numpy(dtype=np.float64) if "jump_flag" in sub else np.zeros(n)
    evt_tail_flag = sub["evt_tail_flag"].to_numpy(dtype=np.float64) if "evt_tail_flag" in sub else np.zeros(n)
    jump_z = sub["jump_z"].to_numpy(dtype=np.float64) if "jump_z" in sub else np.zeros(n)
    return (
        (jump_flag > 0.0)
        | (evt_tail_flag > 0.0)
        | (np.abs(jump_z) >= L6_SHOCK_JUMP_Z_THRESHOLD)
        | (np.abs(ret_1h) >= L6_SHOCK_RET_1H_THRESHOLD)
        | (np.abs(ret_4h) >= L6_SHOCK_RET_4H_THRESHOLD)
    )


def load_sidecar() -> dict[str, Any]:
    with SIDECAR_PATH.open("rb") as f:
        return pickle.load(f)


def build_l4_features(sub: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Reconstruct the sidecar's expected feature_columns row-by-row from the tape's cached
    raw softmax probs, using whichever component (primary/fallback) actually fired -- matching
    trading_bot_modules/omega6_live.py::_risk_features() field-for-field."""
    n = len(sub)
    primary_side = sub["primary_side"].to_numpy(dtype=np.int64)
    use_primary = primary_side != 0
    rows = []
    for i in range(n):
        prefix = "primary" if use_primary[i] else "fallback"
        row = sub.iloc[i]
        p_dir = np.array([row[f"{prefix}_dir_p_cash"], row[f"{prefix}_dir_p_long"], row[f"{prefix}_dir_p_short"]], dtype=np.float64)
        p_qual = np.array([row[f"{prefix}_quality_p_cash"], row[f"{prefix}_quality_p_long"], row[f"{prefix}_quality_p_short"]], dtype=np.float64)
        dir_action = int(np.argmax(p_dir))
        qual_for_action = float(p_qual[dir_action] if dir_action > 0 else p_qual[0])
        final_action = dir_action if (dir_action != 0 and qual_for_action >= threshold) else 0
        side = 1 if final_action == 1 else (-1 if final_action == 2 else 0)
        expert = row[f"{prefix}_expert"]
        dec_notional = L4_BASELINE_NOTIONAL if side else 0.0
        dec_leverage = L4_BASELINE_LEVERAGE if side else 1.0
        dec_tp = L5_BASE_TP_PRICE_MOVE if side else 0.0
        dec_sl = L5_BASE_SL_PRICE_MOVE if side else 0.0
        rows.append(
            {
                "parent_router_confidence": float(row[f"{prefix}_route_confidence"]),
                "parent_router_margin": float(row[f"{prefix}_route_margin"]),
                "parent_dir_p_cash": float(p_dir[0]),
                "parent_dir_p_long": float(p_dir[1]),
                "parent_dir_p_short": float(p_dir[2]),
                "parent_dir_confidence": float(np.max(p_dir)),
                "parent_dir_side_edge": float(abs(p_dir[1] - p_dir[2])),
                "parent_dir_trade_prob": float(p_dir[1] + p_dir[2]),
                "parent_dir_action": int(dir_action),
                "parent_quality_p_cash": float(p_qual[0]),
                "parent_quality_p_long": float(p_qual[1]),
                "parent_quality_p_short": float(p_qual[2]),
                "parent_quality_for_action": float(qual_for_action),
                "parent_quality_threshold": float(threshold),
                "parent_final_action": int(final_action),
                "parent_router_expert_bear": 1.0 if expert == "bear" else 0.0,
                "parent_router_expert_bull": 1.0 if expert == "bull" else 0.0,
                "parent_router_expert_chop": 1.0 if expert == "chop" else 0.0,
                "decision_action": int(final_action),
                "decision_side": int(side),
                "decision_quality_score": float(qual_for_action if side else 0.0),
                "decision_confidence": float(np.max(p_dir)),
                "decision_notional_exposure": float(dec_notional),
                "decision_leverage": float(dec_leverage),
                "decision_position_fraction": float(dec_notional),
                "decision_take_profit": float(dec_tp),
                "decision_stop_loss": float(dec_sl),
                "decision_rr": float(dec_tp) / max(float(dec_sl), 1e-8),
                "atr_pct_runtime": float(row["atr_pct"]),
            }
        )
    return pd.DataFrame(rows)


def sidecar_size(sidecar: dict[str, Any], features_row: pd.DataFrame, side: int) -> tuple[float, float]:
    if side == 0:
        return 0.0, 0.0
    model = sidecar["model"][side]
    cols = sidecar["feature_columns"]
    score = float(model.predict(features_row[cols])[0])
    mapping = dict(sidecar["selected_mapping"])
    q50 = float(sidecar["train_score_q50"])
    iqr = max(float(sidecar["train_score_iqr"]), 1e-8)
    z_margin = float(np.clip((score - q50) / iqr, -8.0, 8.0))
    unit_margin = 1.0 / (1.0 + np.exp(-float(mapping["temp"]) * z_margin))
    scale = float(mapping["min_scale"]) + (float(mapping["max_scale"]) - float(mapping["min_scale"])) * unit_margin
    margin = float(np.clip(scale, float(mapping["floor"]), float(mapping["cap"])))
    margin *= float(mapping.get("long_scale", 1.0)) if side > 0 else float(mapping.get("short_scale", 1.0))
    margin = float(np.clip(margin, float(mapping["floor"]), float(mapping["cap"])))
    z_lev = z_margin
    unit_lev = 1.0 / (1.0 + np.exp(-float(mapping["leverage_temp"]) * z_lev))
    leverage = float(mapping["leverage_min"]) + (float(mapping["leverage_max"]) - float(mapping["leverage_min"])) * unit_lev
    leverage *= float(mapping.get("long_leverage_scale", 1.0)) if side > 0 else float(mapping.get("short_leverage_scale", 1.0))
    leverage = float(np.clip(leverage, float(mapping["leverage_floor"]), float(mapping["leverage_cap"])))
    return margin, leverage


def run(
    tape: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    use_l4_sidecar: bool,
    use_l6_governor: bool,
    fee_mult: float,
    sidecar: dict[str, Any] | None,
    tp_atr_mult: float,
    sl_atr_mult: float,
    cooldown_bars: int,
    persistence_bars: int,
    quality_threshold: float,
    fixed_margin: float,
    fixed_leverage: float,
) -> dict[str, Any]:
    sub = tape[(tape["timestamp"] >= start) & (tape["timestamp"] <= end)].reset_index(drop=True)
    n = len(sub)
    close = sub["close"].to_numpy(dtype=np.float64)
    open_ = sub["open"].to_numpy(dtype=np.float64)

    primary_side_arr = sub["primary_side"].to_numpy(dtype=np.int64)
    fallback_side_arr = sub["fallback_side"].to_numpy(dtype=np.int64)
    eff_side = np.where(primary_side_arr != 0, primary_side_arr, fallback_side_arr)
    persistence_ok = eff_side != 0
    for k in range(1, persistence_bars):
        shifted = np.roll(eff_side, k)
        shifted[:k] = 0
        persistence_ok &= shifted == eff_side

    atr_pct_arr = sub["atr_pct"].to_numpy(dtype=np.float64)
    macro_veto = build_macro_veto_mask(sub["timestamp"]) if use_l6_governor else np.zeros(n, dtype=bool)
    shock_haircut = build_shock_haircut_mask(sub) if use_l6_governor else np.zeros(n, dtype=bool)

    l4_features = build_l4_features(sub, quality_threshold) if use_l4_sidecar else None

    FEE = 0.00020 * fee_mult
    SLIP = 0.00050 * fee_mult

    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    hold_start = 0
    notional = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    max_hold = 288
    trades: list[dict[str, Any]] = []
    cooldown_until = -1
    i = 0
    while i < n - 1:
        if pos == 0:
            if i < cooldown_until or not persistence_ok[i] or eff_side[i] == 0:
                i += 1
                continue
            if macro_veto[i]:
                i += 1
                continue
            side = int(eff_side[i])
            if use_l4_sidecar:
                margin, leverage = sidecar_size(sidecar, l4_features.iloc[[i]], side)
                if margin <= 0.0 or leverage <= 0.0:
                    i += 1
                    continue
            else:
                margin, leverage = fixed_margin, fixed_leverage
            if shock_haircut[i]:
                margin *= L6_SHOCK_NOTIONAL_SCALE
            atr = max(atr_pct_arr[i], 1e-6)
            tp, sl = tp_atr_mult * atr, sl_atr_mult * atr
            entry_price = float(open_[min(i + 1, n - 1)]) * (1.0 + SLIP if side > 0 else 1.0 - SLIP)
            pos = side
            notional = margin * leverage
            take_profit, stop_loss = tp, sl
            hold_start = i
            entry_equity = cash
            cash -= cash * FEE * notional
            i += 1
            continue
        px = close[i]
        raw = (px * (1.0 - SLIP) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + SLIP)) / max(entry_price, 1e-12)
        unreal = raw * notional
        eq = cash * (1.0 + unreal)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        hold_bars = i - hold_start
        reason = ""
        if take_profit > 0.0 and unreal >= take_profit:
            reason = "take_profit"
        elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
            reason = "stop_loss"
        elif hold_bars >= max_hold:
            reason = "time_stop"
        if reason:
            exit_price = close[i] * (1.0 - SLIP if pos > 0 else 1.0 + SLIP)
            raw_exit = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
            before = cash
            cash = cash * (1.0 + raw_exit * notional)
            cash -= before * FEE * notional
            trades.append({"entry_i": hold_start, "exit_i": i, "side": pos, "reason": reason, "win": bool(cash > entry_equity), "month": str(sub.iloc[hold_start]["timestamp"])[:7], "notional": notional})
            pos = 0
            cooldown_until = i + cooldown_bars
        i += 1
    if pos != 0:
        exit_price = close[n - 1]
        raw_exit = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * FEE * notional
        trades.append({"entry_i": hold_start, "exit_i": n - 1, "side": pos, "reason": "forced_end", "win": cash > entry_equity, "month": str(sub.iloc[hold_start]["timestamp"])[:7], "notional": notional})

    wins = sum(1 for t in trades if t["win"])
    by_month: dict[str, int] = {}
    for t in trades:
        by_month.setdefault(t["month"], 0)
        by_month[t["month"]] += 1
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": len(trades),
        "wr": float(wins / len(trades)) if trades else 0.0,
        "avg_notional": float(np.mean([t["notional"] for t in trades])) if trades else 0.0,
        "max_notional": float(np.max([t["notional"] for t in trades])) if trades else 0.0,
        "trades_by_month": by_month,
    }


def main() -> int:
    tape_raw = v2.load_tape()
    tape = v2.apply_quality_threshold(tape_raw, FROZEN_KW["quality_threshold"])
    sidecar = load_sidecar()

    scenarios = [
        ("baseline_fixed_sizing", dict(use_l4_sidecar=False, use_l6_governor=False)),
        ("l4_sidecar_dynamic_sizing", dict(use_l4_sidecar=True, use_l6_governor=False)),
        ("l6_governor_only", dict(use_l4_sidecar=False, use_l6_governor=True)),
        ("l4_and_l6", dict(use_l4_sidecar=True, use_l6_governor=True)),
    ]

    results = {}
    for name, kw in scenarios:
        out = {}
        for tag, mult in (("cost1", 1.0), ("cost3", 3.0)):
            out[tag] = run(
                tape,
                start=v2.VAL_START,
                end=v2.VAL_END,
                fee_mult=mult,
                sidecar=sidecar,
                tp_atr_mult=FROZEN_KW["tp_atr_mult"],
                sl_atr_mult=FROZEN_KW["sl_atr_mult"],
                cooldown_bars=FROZEN_KW["cooldown_bars"],
                persistence_bars=FROZEN_KW["persistence_bars"],
                quality_threshold=FROZEN_KW["quality_threshold"],
                fixed_margin=FROZEN_KW["fixed_margin"],
                fixed_leverage=FROZEN_KW["fixed_leverage"],
                **kw,
            )
        results[name] = out
        print(
            f"{name}: cost1 pnl={out['cost1']['pnl']:.2f}% mdd={out['cost1']['mdd']:.2f}% "
            f"trades={out['cost1']['trades']} avg_notional={out['cost1']['avg_notional']:.3f} "
            f"max_notional={out['cost1']['max_notional']:.3f} | "
            f"cost3 pnl={out['cost3']['pnl']:.2f}% mdd={out['cost3']['mdd']:.2f}%",
            flush=True,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "l4l6_scenarios.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved: {OUT_DIR / 'l4l6_scenarios.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
