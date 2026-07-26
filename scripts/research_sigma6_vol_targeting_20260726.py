#!/usr/bin/env python3
"""Sigma6 + volatility-targeting sizing overlay.

PRE-REGISTERED in docs/mechanical_trading_research_synthesis_20260726.md S5.2.1
(committed before this script was written). Grid, calibration window, bounds and
gates are fixed there. Do not edit them after seeing output.

Sigma6's frozen production configs (sigma6_lev3/sigma6_lev4, from
research_f4b_sigma6_dated_ledger_20260719.py, matching the contract doc) use a
constant notional = margin * leverage for every trade regardless of how calm or
violent the market is at entry. This overlay replaces only the sizing: notional
scales inversely with atr_pct at entry, calibrated so its VAL-period average
equals the frozen baseline's notional (an apples-to-apples comparison, not a
free leverage increase). Everything else -- entry signal, regime filter, stop/
trail/time exit, fees -- is copied unchanged from backtest_with_dates().

  vol_scalar(i) = clip(target_atr / atr_pct[i], lo, hi)
  notional(i)   = base_notional * vol_scalar(i)
  target_atr    = median atr_pct over VAL 2025-07..12 (frozen constant, computed
                  once, applied unchanged to VAL and OOS -- same treatment as
                  every other frozen threshold in this pipeline)
  (lo, hi)      = (0.4, 2.0) -- prevents unbounded leverage in dead-calm regimes

Gate on VAL (2025-07-01..12-31): promote only if MDD improves AND PnL/Sharpe
does not worsen (the same AND-gate F4-B used for portfolio combination).

OOS window 2026-03-02..06-30 has been reused repeatedly by this project (Sigma6
itself, Sigma8-11, F4-B) and the original Sigma6 script already flags it as
"Nth use -> degraded evidential value". That flag is inherited unchanged here,
not laundered into a fresh confirmation.

The 2026-07-01..07-20 extension is reported for context only, never as
promotion evidence: retraining the sigma3-1h ensemble on that period previously
failed to reproduce the frozen baseline (fresh-window fragility), so the tape
here is used via causal inference only, unretrained.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402
import run_sigma6_regime_trend_20260705 as sigma6  # noqa: E402

OUT = Path("data/ensemble/metrics/sigma6_vol_targeting_20260726.json")

FROZEN_CONFIGS = {
    "sigma6_lev4": dict(thr=0.70, leverage=4.0, margin=0.30, trail_atr=5.0, sl_atr=2.5,
                         min_profit_atr=2.0, max_hold=144, cooldown=3,
                         reg_mode="not_chop", reg_thr=0.42, stab_thr=0.55, fee_mult=1.0),
    "sigma6_lev3": dict(thr=0.70, leverage=3.0, margin=0.30, trail_atr=5.0, sl_atr=2.5,
                         min_profit_atr=2.0, max_hold=144, cooldown=3,
                         reg_mode="not_chop", reg_thr=0.50, stab_thr=0.55, fee_mult=1.0),
}
LO, HI = 0.4, 2.0
PFX = sigma6.PFX


def backtest_vol_targeted(tape, *, leverage, margin, trail_atr, sl_atr, min_profit_atr, max_hold,
                           cooldown, reg_mode, reg_thr, stab_thr, fee_mult, start, end,
                           target_atr: float | None):
    """backtest_with_dates() logic, unchanged, except notional is recomputed per
    trade at entry when target_atr is given. target_atr=None reproduces the
    frozen fixed-notional baseline exactly (used to verify the copy is faithful)."""
    sub = tape[(tape["timestamp"] >= start) & (tape["timestamp"] <= end)].reset_index(drop=True)
    n = len(sub)
    close = sub["close"].to_numpy(np.float64)
    open_ = sub["open"].to_numpy(np.float64)
    side_arr = sub["primary_side"].to_numpy(np.int64)
    atr_arr = sub["atr_pct"].to_numpy(np.float64)
    bull = sub[f"{PFX}bull_prob"].to_numpy(np.float64)
    bear = sub[f"{PFX}bear_prob"].to_numpy(np.float64)
    chop = sub[f"{PFX}chop_prob"].to_numpy(np.float64)
    stab = sub["regime3_cmamba_h6_sidecar_stability_score"].fillna(1.0).to_numpy(np.float64)
    FEE, SLIP = 0.00020 * fee_mult, 0.00050 * fee_mult
    base_notional = margin * leverage
    cash = peak_eq = 1.0
    mdd = 0.0
    pos = 0
    entry_price = peak_unreal = entry_atr = notional = 0.0
    hold_start = 0
    entry_equity = 1.0
    trades = []
    cooldown_until = -1
    i = 0
    while i < n - 1:
        if pos == 0:
            if i < cooldown_until or side_arr[i] == 0:
                i += 1
                continue
            side = int(side_arr[i])
            ok = True
            if reg_mode == "trend_agree":
                ok = (side > 0 and bull[i] >= reg_thr) or (side < 0 and bear[i] >= reg_thr)
            elif reg_mode == "not_chop":
                ok = chop[i] < reg_thr
            if ok and stab_thr > 0:
                ok = stab[i] >= stab_thr
            if not ok:
                i += 1
                continue
            entry_price = float(open_[min(i + 1, n - 1)]) * (1 + SLIP if side > 0 else 1 - SLIP)
            pos, hold_start, peak_unreal, entry_atr = side, i, 0.0, max(atr_arr[i], 1e-6)
            if target_atr is None:
                notional = base_notional
            else:
                notional = base_notional * float(np.clip(target_atr / entry_atr, LO, HI))
            entry_equity = cash
            cash -= cash * FEE * notional
            i += 1
            continue
        px = close[i]
        raw = (px * (1 - SLIP) - entry_price) / entry_price if pos > 0 else (entry_price - px * (1 + SLIP)) / entry_price
        unreal = raw * notional
        eq = cash * (1 + unreal)
        peak_eq = max(peak_eq, eq)
        mdd = min(mdd, eq / max(peak_eq, 1e-12) - 1)
        peak_unreal = max(peak_unreal, unreal)
        hold = i - hold_start
        reason = ""
        # The frozen backtest compares unreal (= raw_price_move * notional) against a
        # threshold in raw price-move units (sl_atr * entry_atr), so with notional fixed
        # per config the effective stop distance in price space is sl_atr*entry_atr/notional.
        # This overlay only resizes the position -- the *_atr thresholds are rescaled by
        # notional/base_notional so the stop/trail/time exits stay at the exact same price
        # distance as the frozen baseline regardless of vol_scalar, isolating sizing from exits.
        scale = notional / base_notional
        if unreal <= -sl_atr * entry_atr * scale:
            reason = "stop"
        elif peak_unreal >= min_profit_atr * entry_atr * scale and (
            peak_unreal - unreal
        ) >= trail_atr * entry_atr * scale:
            reason = "trail"
        elif hold >= max_hold:
            reason = "time"
        if reason:
            exit_price = close[i] * (1 - SLIP if pos > 0 else 1 + SLIP)
            rex = (exit_price - entry_price) / entry_price if pos > 0 else (entry_price - exit_price) / entry_price
            before = cash
            cash = cash * (1 + rex * notional)
            cash -= before * FEE * notional
            trades.append({
                "win": cash > entry_equity,
                "entry_timestamp": sub.iloc[hold_start]["timestamp"],
                "notional": notional,
                "ret": rex * notional,
            })
            pos = 0
            cooldown_until = i + cooldown
        i += 1
    wins = sum(1 for t in trades if t["win"])
    daily = pd.Series(
        [t["ret"] for t in trades],
        index=[pd.Timestamp(t["entry_timestamp"]).normalize() for t in trades],
    ).groupby(level=0).sum() if trades else pd.Series(dtype=float)
    return {
        "pnl": (cash - 1) * 100,
        "mdd": mdd * 100,
        "trades": len(trades),
        "wr": wins / len(trades) if trades else 0.0,
        "notional_mean": float(np.mean([t["notional"] for t in trades])) if trades else 0.0,
        "notional_min": float(np.min([t["notional"] for t in trades])) if trades else 0.0,
        "notional_max": float(np.max([t["notional"] for t in trades])) if trades else 0.0,
    }, daily


def diagnose_atr_vs_return(tape, *, leverage, margin, trail_atr, sl_atr, min_profit_atr, max_hold,
                            cooldown, reg_mode, reg_thr, stab_thr, fee_mult, start, end) -> dict:
    """Correlation between entry-time ATR and the trade's per-notional return.

    Vol-targeting bets that entry ATR is noise to be sized away. If ATR instead
    predicts which trades win (as it does for a trend-follower whose trailing
    exit is built to ride exactly the volatility expansions that follow a
    high-ATR entry), downsizing high-ATR entries throws away the edge instead
    of protecting against noise. This runs the frozen-notional backtest once
    more, recording entry_atr and per-notional return per trade, with no
    sizing applied at all -- so it cannot be confounded by the overlay itself.
    """
    sub = tape[(tape["timestamp"] >= start) & (tape["timestamp"] <= end)].reset_index(drop=True)
    n = len(sub)
    close = sub["close"].to_numpy(np.float64)
    open_ = sub["open"].to_numpy(np.float64)
    side_arr = sub["primary_side"].to_numpy(np.int64)
    atr_arr = sub["atr_pct"].to_numpy(np.float64)
    chop = sub[f"{PFX}chop_prob"].to_numpy(np.float64)
    stab = sub["regime3_cmamba_h6_sidecar_stability_score"].fillna(1.0).to_numpy(np.float64)
    SLIP = 0.00050 * fee_mult
    pos = 0
    entry_price = peak_unreal = entry_atr = 0.0
    hold_start = 0
    rows = []
    cooldown_until = -1
    i = 0
    while i < n - 1:
        if pos == 0:
            if i < cooldown_until or side_arr[i] == 0:
                i += 1
                continue
            side = int(side_arr[i])
            ok = chop[i] < reg_thr if reg_mode == "not_chop" else True
            if ok and stab_thr > 0:
                ok = stab[i] >= stab_thr
            if not ok:
                i += 1
                continue
            entry_price = float(open_[min(i + 1, n - 1)]) * (1 + SLIP if side > 0 else 1 - SLIP)
            pos, hold_start, peak_unreal, entry_atr = side, i, 0.0, max(atr_arr[i], 1e-6)
            i += 1
            continue
        px = close[i]
        raw = (px * (1 - SLIP) - entry_price) / entry_price if pos > 0 else (entry_price - px * (1 + SLIP)) / entry_price
        peak_unreal = max(peak_unreal, raw)
        hold = i - hold_start
        reason = ""
        if raw <= -sl_atr * entry_atr:
            reason = "stop"
        elif peak_unreal >= min_profit_atr * entry_atr and (peak_unreal - raw) >= trail_atr * entry_atr:
            reason = "trail"
        elif hold >= max_hold:
            reason = "time"
        if reason:
            exit_price = close[i] * (1 - SLIP if pos > 0 else 1 + SLIP)
            rex = (exit_price - entry_price) / entry_price if pos > 0 else (entry_price - exit_price) / entry_price
            rows.append({"entry_atr": entry_atr, "ret_on_notional": rex})
            pos = 0
            cooldown_until = i + cooldown
        i += 1

    td = pd.DataFrame(rows)
    if len(td) < 6:
        return {"n_trades": len(td), "note": "too few trades for a tercile breakdown"}
    td["tercile"] = pd.qcut(td["entry_atr"], 3, labels=["low", "mid", "high"])
    by_tercile = td.groupby("tercile", observed=True)["ret_on_notional"].mean()
    return {
        "n_trades": len(td),
        "corr_entry_atr_vs_return": float(td["entry_atr"].corr(td["ret_on_notional"])),
        "mean_return_by_atr_tercile": {k: float(v) for k, v in by_tercile.items()},
    }


def sharpe_like(daily: pd.Series) -> float:
    if len(daily) < 5 or daily.std(ddof=1) < 1e-12:
        return 0.0
    return float(daily.mean() / daily.std(ddof=1) * np.sqrt(365))


def main() -> None:
    raw = sigma6.load_tape_with_regime()
    print(f"tape range: {raw['timestamp'].min()} .. {raw['timestamp'].max()} ({len(raw)} rows)", flush=True)

    report: dict = {
        "preregistration": "docs/mechanical_trading_research_synthesis_20260726.md S5.2.1",
        "bounds": {"lo": LO, "hi": HI},
        "oos_reuse_caveat": (
            "2026-03-02..06-30 has been reused by Sigma6 itself, Sigma8-11, and F4-B before this. "
            "Inherited degraded evidential value, not a fresh test."
        ),
        "tape_contamination_warning": (
            "tmp/causal_regen_20260516/sigma3_1h_hgb_20260705/tape_ensemble.parquet was regenerated "
            "2026-07-20 (report file mtime) and no longer reproduces the frozen contract's OOS "
            "numbers -- see frozen_oos_reproduction_check per config below vs. the contract's "
            "sigma6_lev4 +45.9%/sigma6_lev3 +16.6%. This matches previously-documented sigma3-1h "
            "ensemble retrain fragility (2026-07-20 fresh-window session). All VAL/OOS absolute "
            "levels in this report are computed on the CURRENT (drifted) tape; only the paired "
            "baseline-vs-vol-targeted comparison on the same tape is treated as informative here."
        ),
    }
    per_config = {}

    for name, cfg0 in FROZEN_CONFIGS.items():
        cfg = dict(cfg0)
        thr = cfg.pop("thr")
        tape = v2.apply_quality_threshold(raw, thr)

        # Fidelity check: target_atr=None must exactly reproduce the frozen baseline.
        base_result, base_daily = backtest_vol_targeted(
            tape, start=sigma6.OOS_START, end=sigma6.OOS_END, target_atr=None, **cfg
        )

        val_all = tape[(tape["timestamp"] >= sigma6.VAL_START) & (tape["timestamp"] <= sigma6.VAL_END)]
        target_atr = float(val_all["atr_pct"].median())

        val_base, val_base_daily = backtest_vol_targeted(
            tape, start=sigma6.VAL_START, end=sigma6.VAL_END, target_atr=None, **cfg
        )
        val_vt, val_vt_daily = backtest_vol_targeted(
            tape, start=sigma6.VAL_START, end=sigma6.VAL_END, target_atr=target_atr, **cfg
        )

        mdd_improved = val_vt["mdd"] > val_base["mdd"]  # both negative; closer to 0 is better
        sharpe_base, sharpe_vt = sharpe_like(val_base_daily), sharpe_like(val_vt_daily)
        not_worse = sharpe_vt >= sharpe_base * 0.95  # small tolerance, symmetric with F4-B's framing
        gate_pass = bool(mdd_improved and not_worse)

        atr_diagnosis = diagnose_atr_vs_return(
            tape, start=sigma6.VAL_START, end=sigma6.VAL_END, **cfg
        )

        entry = {
            "target_atr_calibrated_on_VAL": target_atr,
            "atr_vs_return_diagnosis_VAL": atr_diagnosis,
            "frozen_oos_reproduction_check": base_result,  # must match the contract's frozen numbers
            "val_baseline_fixed_notional": val_base,
            "val_baseline_sharpe_like": sharpe_base,
            "val_vol_targeted": val_vt,
            "val_vol_targeted_sharpe_like": sharpe_vt,
            "gate": {
                "mdd_improved": mdd_improved,
                "sharpe_not_worse_5pct_tol": not_worse,
                "pass": gate_pass,
            },
        }

        if gate_pass:
            oos_base, oos_base_daily = backtest_vol_targeted(
                tape, start=sigma6.OOS_START, end=sigma6.OOS_END, target_atr=None, **cfg
            )
            oos_vt, oos_vt_daily = backtest_vol_targeted(
                tape, start=sigma6.OOS_START, end=sigma6.OOS_END, target_atr=target_atr, **cfg
            )
            entry["oos_baseline_fixed_notional"] = oos_base
            entry["oos_vol_targeted"] = oos_vt
            entry["oos_baseline_sharpe_like"] = sharpe_like(oos_base_daily)
            entry["oos_vol_targeted_sharpe_like"] = sharpe_like(oos_vt_daily)

            fresh_start, fresh_end = pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-20 23:59:59")
            fresh_base, _ = backtest_vol_targeted(
                tape, start=fresh_start, end=fresh_end, target_atr=None, **cfg
            )
            fresh_vt, _ = backtest_vol_targeted(
                tape, start=fresh_start, end=fresh_end, target_atr=target_atr, **cfg
            )
            entry["context_only_fresh_2026_07_baseline"] = fresh_base
            entry["context_only_fresh_2026_07_vol_targeted"] = fresh_vt

        per_config[name] = entry
        print(f"\n=== {name} ===", flush=True)
        print(json.dumps(entry, indent=2, default=str), flush=True)

    report["configs"] = per_config
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
