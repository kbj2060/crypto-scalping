#!/usr/bin/env python3
"""Sigma11: LEARNED SIZING, the 3rd of the 4 architecture directions proposed for "is Sigma6
architecturally optimal?" (multi-asset = Sigma9, failed; regime experts = Sigma10, unstable/failed;
learned sizing = this; order-book = blocked, insufficient history).

Sigma6 uses a FLAT leverage (3x or 4x) for every trade. This tests per-trade leverage that scales
with (a) the model's own quality/confidence score (primary_quality_score, already computed and
causally available at entry) and (b) inverse ATR (vol targeting: less leverage in high-vol regimes,
more in low-vol). Leverage is frozen at entry (no dynamic re-sizing mid-trade, matching how a real
order would be sized) and clipped to [lev_min, lev_max]. Same Sigma6 generalist signal + not_chop
regime filter + trailing-stop execution -- only the sizing rule changes.

Compared against fixed-leverage baselines at a MATCHED average realized leverage (not just raw PnL)
so the comparison isn't confounded by "more average leverage = more return".
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
import run_sigma6_regime_trend_20260705 as s6  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma11_dynamic_leverage_20260706"
PFX = s6.PFX


def backtest_dynamic(tape, *, margin, trail_atr, sl_atr, min_profit_atr, max_hold, cooldown,
                      reg_mode, reg_thr, stab_thr, fee_mult, start, end,
                      lev_base, lev_min, lev_max, use_quality, use_vol_target, quality_ref, atr_ref):
    sub = tape[(tape["timestamp"] >= start) & (tape["timestamp"] <= end)].reset_index(drop=True)
    n = len(sub)
    close = sub["close"].to_numpy(np.float64)
    open_ = sub["open"].to_numpy(np.float64)
    side_arr = sub["primary_side"].to_numpy(np.int64)
    atr_arr = sub["atr_pct"].to_numpy(np.float64)
    qual_arr = sub["primary_quality_score"].to_numpy(np.float64) if "primary_quality_score" in sub.columns else np.full(n, quality_ref)
    bull = sub[f"{PFX}bull_prob"].to_numpy(np.float64)
    bear = sub[f"{PFX}bear_prob"].to_numpy(np.float64)
    chop = sub[f"{PFX}chop_prob"].to_numpy(np.float64)
    stab = sub["regime3_cmamba_h6_sidecar_stability_score"].fillna(1.0).to_numpy(np.float64)
    FEE, SLIP = 0.00020 * fee_mult, 0.00050 * fee_mult
    cash = peak_eq = 1.0
    mdd = 0.0
    pos = 0
    entry_price = peak_unreal = entry_atr = 0.0
    hold_start = 0
    entry_equity = 1.0
    notional = 0.0
    lev_used = []
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
            lev = lev_base
            if use_quality:
                lev *= max(qual_arr[i], 1e-3) / quality_ref
            if use_vol_target:
                lev *= atr_ref / max(atr_arr[i], 1e-6)
            lev = float(np.clip(lev, lev_min, lev_max))
            lev_used.append(lev)
            notional = margin * lev
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
            trades.append({"win": cash > entry_equity, "month": str(sub.iloc[hold_start]["timestamp"])[:7], "ret": rex * notional})
            pos = 0
            cooldown_until = i + cooldown
        i += 1
    wins = sum(1 for t in trades if t["win"])
    bym = {}
    for t in trades:
        bym.setdefault(t["month"], 0.0)
        bym[t["month"]] += t["ret"]
    return {"pnl": (cash - 1) * 100, "mdd": mdd * 100, "trades": len(trades),
            "wr": wins / len(trades) if trades else 0.0, "avg_lev": float(np.mean(lev_used)) if lev_used else 0.0,
            "by_month": bym}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = s6.load_tape_with_regime()
    tapes = {thr: v2.apply_quality_threshold(raw, thr) for thr in (0.70,)}
    tape70 = tapes[0.70]
    base = dict(margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144, cooldown=3,
                reg_mode="not_chop", stab_thr=0.55, sl_atr=2.5, fee_mult=1.0, start=s6.VAL_START, end=s6.VAL_END)
    atr_ref = float(tape70.loc[(tape70["timestamp"] >= s6.VAL_START) & (tape70["timestamp"] <= s6.VAL_END), "atr_pct"].median())
    print(f"VAL median atr_pct (vol-target reference): {atr_ref:.5f}", flush=True)

    print("\n=== Fixed-leverage baselines (Sigma6) ===", flush=True)
    for lev, rthr in ((3.0, 0.50), (4.0, 0.42)):
        r = s6.backtest(tape70, leverage=lev, reg_thr=rthr, **base)
        print(f"  flat lev={lev}: c1={r['pnl']:.1f}% mdd={r['mdd']:.1f}% tr={r['trades']} wr={r['wr']:.3f}", flush=True)

    print("\n=== Dynamic leverage sweep (lev_base, mode, reg_thr) ===", flush=True)
    rows = []
    for lev_base, rthr in ((3.0, 0.50), (4.0, 0.42)):
        for use_q, use_v, name in ((True, False, "quality-only"), (False, True, "vol-target-only"), (True, True, "quality+vol")):
            r = backtest_dynamic(tape70, lev_base=lev_base, lev_min=1.5, lev_max=lev_base * 2.0,
                                  use_quality=use_q, use_vol_target=use_v, quality_ref=0.55, atr_ref=atr_ref,
                                  reg_thr=rthr, **base)
            rows.append({"lev_base": lev_base, "mode": name, "c1": round(r["pnl"], 1), "mdd": round(r["mdd"], 1),
                         "tr": r["trades"], "wr": round(r["wr"], 3), "avg_lev": round(r["avg_lev"], 2)})
            print(f"  lev_base={lev_base} {name}: c1={r['pnl']:.1f}% mdd={r['mdd']:.1f}% tr={r['trades']} "
                  f"wr={r['wr']:.3f} avg_realized_lev={r['avg_lev']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "dynamic_leverage_val_results.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
