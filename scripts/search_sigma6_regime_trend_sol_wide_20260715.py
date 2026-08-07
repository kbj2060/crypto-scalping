"""Wider, SOL-specific VAL grid search for the Sigma6 trend-follower + regime filter, replacing
the narrow grid in run_sigma6_regime_trend_sol_20260715.py (which mostly reused ETH's exact grid
points and found only the same weak/OOS-failing config as the Calmar-best). Staged search
(execution params first, then leverage+regime filter) to stay tractable, purely on VAL
(2025-07..12) -- no OOS window touched. Goal: establish whether ANY config in this family reaches
a genuinely strong return/MDD (Calmar-like) profile on SOL before spending further OOS looks.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_sigma6_regime_trend_sol_20260715 as sol6  # noqa: E402
import replay_omega6_v2_variants_20260704 as v2  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma6_regime_trend_sol_20260715"
VAL_START, VAL_END = sol6.VAL_START, sol6.VAL_END


def run(tapes, *, thresholds, leverages, sl_atrs, trail_atrs, min_profit_atrs, max_holds, cooldowns, reg_modes, reg_thrs, min_trades=15):
    rows = []
    n_total = 0
    for thr, lev, sl, trail, minp, maxh, cd, mode, rthr in itertools.product(
        thresholds, leverages, sl_atrs, trail_atrs, min_profit_atrs, max_holds, cooldowns, reg_modes, reg_thrs
    ):
        if mode == "none" and rthr != reg_thrs[0]:
            continue
        n_total += 1
        r = sol6.backtest(
            tapes[thr], leverage=lev, margin=0.30, trail_atr=trail, sl_atr=sl,
            min_profit_atr=minp, max_hold=maxh, cooldown=cd,
            reg_mode=mode, reg_thr=rthr, stab_thr=0.0, fee_mult=1.0,
            start=VAL_START, end=VAL_END,
        )
        if r["trades"] < min_trades:
            continue
        calmar = r["pnl"] / max(abs(r["mdd"]), 1.0)
        rows.append({
            "thr": thr, "lev": lev, "sl": sl, "trail": trail, "minp": minp, "maxh": maxh, "cd": cd,
            "mode": mode, "rthr": rthr, "pnl": round(r["pnl"], 1), "mdd": round(r["mdd"], 1),
            "trades": r["trades"], "wr": round(r["wr"], 3), "months": len(r["by_month"]),
            "minmo": round(min(r["by_month"].values()) * 100, 1) if r["by_month"] else 0.0,
            "calmar": round(calmar, 3),
        })
    print(f"  evaluated {n_total} combos, {len(rows)} kept (trades>={min_trades})", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    raw = sol6.load_tape_with_regime()
    thresholds_all = [0.55, 0.60, 0.65, 0.70, 0.75]
    tapes = {thr: v2.apply_quality_threshold(raw, thr) for thr in thresholds_all}

    print("=== stage 1: execution params (regime fixed: not_chop, rthr=0.34, lev=3.0) ===", flush=True)
    df1 = run(
        tapes, thresholds=thresholds_all, leverages=[3.0],
        sl_atrs=[1.0, 1.5, 2.0, 2.5, 3.0], trail_atrs=[3.0, 4.0, 5.0, 6.0],
        min_profit_atrs=[1.0, 1.5, 2.0, 2.5], max_holds=[72, 144, 216], cooldowns=[0, 3, 6],
        reg_modes=["not_chop"], reg_thrs=[0.34],
    )
    df1 = df1.sort_values("calmar", ascending=False)
    df1.to_csv(OUT_DIR / "val_wide_stage1_execution.csv", index=False)
    print(df1.head(15).to_string(index=False), flush=True)
    best1 = df1.iloc[0]
    print(f"stage1 best: sl={best1.sl} trail={best1.trail} minp={best1.minp} maxh={best1.maxh} cd={best1.cd} thr={best1.thr}", flush=True)

    print("=== stage 2: leverage + regime filter (execution params fixed from stage 1) ===", flush=True)
    df2 = run(
        tapes, thresholds=[float(best1.thr)], leverages=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        sl_atrs=[float(best1.sl)], trail_atrs=[float(best1.trail)], min_profit_atrs=[float(best1.minp)],
        max_holds=[int(best1.maxh)], cooldowns=[int(best1.cd)],
        reg_modes=["not_chop", "trend_agree", "none"], reg_thrs=[0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50],
    )
    df2 = df2.sort_values("calmar", ascending=False)
    df2.to_csv(OUT_DIR / "val_wide_stage2_leverage_regime.csv", index=False)
    print(df2.head(20).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
