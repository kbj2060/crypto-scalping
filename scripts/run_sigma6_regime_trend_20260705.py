#!/usr/bin/env python3
"""Sigma6: Sigma5 'let winners run' trend-follower + REGIME FILTER borrowed from the Omega stack.

Omega technique reused: the Regime3 'current' HMM nowcast classifier
(regime3_current_sensitive_hmm_wide24, bull/bear/chop probabilities per bar, trained on 2024 and
applied causally). Sigma5 hit +118% on trending H2-2025 but bled to -2.5% on the choppier fresh
window because a low-win-rate trailing trend-follower loses in range/chop. The regime filter
directly targets that: only enter LONG when the current regime is BULL, only SHORT when BEAR, and
SIT OUT chop entirely -- so the strategy stops trading in exactly the conditions where it bleeds.

Signal = Sigma3-1h ensemble (unchanged). Regime probs merged causally (merge_asof backward) from
the 5m wide24 sidecar onto the 1h tape timestamps. Optional cmamba stability gate. cost1 primary.

Validation 2025-07..12; one-shot 2026-03-02..06-30 (Nth use -> degraded evidential value, flagged).
"""

from __future__ import annotations

import itertools
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

TAPE = ROOT / "tmp/causal_regen_20260516/sigma3_1h_hgb_20260705/tape_ensemble.parquet"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma6_regime_trend_20260705"
REG_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
CM_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_h6_sidecar_20260601"
VAL_START, VAL_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-03-02"), pd.Timestamp("2026-06-30 23:59:59")
PFX = "regime3_current_sensitive_wide24_"


def load_tape_with_regime() -> pd.DataFrame:
    t = pd.read_parquet(TAPE)
    t["timestamp"] = pd.to_datetime(t["timestamp"]).astype("datetime64[ns]")
    t = t.sort_values("timestamp").reset_index(drop=True)
    reg = pd.concat([
        pd.read_csv(REG_DIR / "training_features_2025_regime3_current_sensitive_hmm_wide24.csv", parse_dates=["timestamp"]),
        pd.read_csv(REG_DIR / "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv", parse_dates=["timestamp"]),
    ], ignore_index=True).sort_values("timestamp")
    reg["timestamp"] = reg["timestamp"].astype("datetime64[ns]")
    keep = ["timestamp", f"{PFX}bull_prob", f"{PFX}bear_prob", f"{PFX}chop_prob"]
    t = pd.merge_asof(t, reg[keep], on="timestamp", direction="backward")
    cm = pd.concat([
        pd.read_csv(CM_DIR / "training_features_2025_regime3_cryptomamba_h6_sidecar_20260601.csv", parse_dates=["timestamp"]),
        pd.read_csv(CM_DIR / "training_features_2026_rebuilt_regime3_cryptomamba_h6_sidecar_20260601.csv", parse_dates=["timestamp"]),
    ], ignore_index=True).sort_values("timestamp")
    cm["timestamp"] = cm["timestamp"].astype("datetime64[ns]")
    t = pd.merge_asof(t, cm[["timestamp", "regime3_cmamba_h6_sidecar_stability_score"]], on="timestamp", direction="backward")
    return t.sort_values("i").reset_index(drop=True)


def backtest(tape, *, leverage, margin, trail_atr, sl_atr, min_profit_atr, max_hold, cooldown,
             reg_mode, reg_thr, stab_thr, fee_mult, start, end):
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
    notional = margin * leverage
    cash = peak_eq = 1.0
    mdd = 0.0
    pos = 0
    entry_price = peak_unreal = entry_atr = 0.0
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
            # regime filter
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
        if unreal <= -sl_atr * entry_atr:
            reason = "stop"
        elif peak_unreal >= min_profit_atr * entry_atr and (peak_unreal - unreal) >= trail_atr * entry_atr:
            reason = "trail"
        elif hold >= max_hold:
            reason = "time"
        if reason:
            exit_price = close[i] * (1 - SLIP if pos > 0 else 1 + SLIP)
            rex = (exit_price - entry_price) / entry_price if pos > 0 else (entry_price - exit_price) / entry_price
            before = cash
            cash = cash * (1 + rex * notional)
            cash -= before * FEE * notional
            trades.append({"win": cash > entry_equity, "month": str(sub.iloc[hold_start]["timestamp"])[:7], "reason": reason, "ret": rex * notional,
                           "entry_timestamp": sub.iloc[hold_start]["timestamp"], "exit_timestamp": sub.iloc[i]["timestamp"]})
            pos = 0
            cooldown_until = i + cooldown
        i += 1
    wins = sum(1 for t in trades if t["win"])
    bym = {}
    for t in trades:
        bym.setdefault(t["month"], 0.0)
        bym[t["month"]] += t["ret"]
    return {"pnl": (cash - 1) * 100, "mdd": mdd * 100, "trades": len(trades),
            "wr": wins / len(trades) if trades else 0.0, "by_month": bym, "trade_list": trades}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_tape_with_regime()
    tapes = {thr: v2.apply_quality_threshold(raw, thr) for thr in (0.60, 0.70)}
    # fixed the strong Sigma5 barrier region; sweep regime filter
    base = dict(margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144, cooldown=3)
    grid = list(itertools.product(
        [0.60, 0.70],           # signal threshold
        [3.0, 4.0],             # leverage
        [1.5, 2.5],             # sl_atr
        ["trend_agree", "not_chop", "none"],   # regime mode
        [0.34, 0.42, 0.50],     # reg_thr
        [0.0, 0.55],            # stability gate
    ))
    rows = []
    for thr, lev, sl, mode, rthr, stab in grid:
        if mode == "none" and (rthr != 0.34 or stab != 0.0):
            continue  # dedupe the no-filter baseline
        r1 = backtest(tapes[thr], leverage=lev, sl_atr=sl, reg_mode=mode, reg_thr=rthr, stab_thr=stab, fee_mult=1.0, start=VAL_START, end=VAL_END, **base)
        rows.append({"thr": thr, "lev": lev, "sl": sl, "mode": mode, "rthr": rthr, "stab": stab,
                     "c1": round(r1["pnl"], 1), "c1mdd": round(r1["mdd"], 1), "tr": r1["trades"],
                     "wr": round(r1["wr"], 3), "mo": len(r1["by_month"]),
                     "minmo": round(min(r1["by_month"].values()) * 100, 1) if r1["by_month"] else 0.0})
    df = pd.DataFrame(rows).sort_values("c1", ascending=False)
    df.to_csv(OUT_DIR / "val_regime_frontier.csv", index=False)
    print("=== VAL 2025-07..12, Sigma6 (trend-follower + regime filter), top 22 by cost1 ===", flush=True)
    print(df.head(22).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
