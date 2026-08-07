"""Follow-up to search_sigma6_regime_trend_sol_wide_20260715.py: that search found a strong
Calmar config (thr=0.75, lev=1.5, PnL+21.6%/MDD-5.0%) but only 15 trades over 5-6 months --
too thin to trust. VAL window cannot be extended (train cutoff is 2025-06-30, and extending past
2025-12-31 would consume the reserved OOS window prematurely), so the lever here is a LOWER
quality threshold to admit more trades within the same VAL window, with a hard trades>=30 floor
enforced during the search itself (not just filtered after the fact), staged the same way.
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
MIN_TRADES = 30


def run(tapes, *, thresholds, leverages, sl_atrs, trail_atrs, min_profit_atrs, max_holds, cooldowns, reg_modes, reg_thrs):
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
        if r["trades"] < MIN_TRADES:
            continue
        calmar = r["pnl"] / max(abs(r["mdd"]), 1.0)
        rows.append({
            "thr": thr, "lev": lev, "sl": sl, "trail": trail, "minp": minp, "maxh": maxh, "cd": cd,
            "mode": mode, "rthr": rthr, "pnl": round(r["pnl"], 1), "mdd": round(r["mdd"], 1),
            "trades": r["trades"], "wr": round(r["wr"], 3), "months": len(r["by_month"]),
            "minmo": round(min(r["by_month"].values()) * 100, 1) if r["by_month"] else 0.0,
            "calmar": round(calmar, 3),
        })
    print(f"  evaluated {n_total} combos, {len(rows)} kept (trades>={MIN_TRADES})", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    raw = sol6.load_tape_with_regime()
    thresholds_all = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    tapes = {thr: v2.apply_quality_threshold(raw, thr) for thr in thresholds_all}

    print("=== stage A: threshold + execution params (regime fixed: not_chop, rthr=0.34, lev=3.0) ===", flush=True)
    dfA = run(
        tapes, thresholds=thresholds_all, leverages=[3.0],
        sl_atrs=[1.5, 2.0, 2.5], trail_atrs=[4.0, 5.0, 6.0],
        min_profit_atrs=[1.5, 2.0, 2.5], max_holds=[72, 144], cooldowns=[3, 6],
        reg_modes=["not_chop"], reg_thrs=[0.34],
    )
    dfA = dfA.sort_values("calmar", ascending=False)
    dfA.to_csv(OUT_DIR / "val_stable_stageA_execution.csv", index=False)
    print(dfA.head(15).to_string(index=False), flush=True)
    if len(dfA) == 0:
        print("no combo reached trades>=30 in stage A; widening thresholds further needed", flush=True)
        return 0
    bestA = dfA.iloc[0]
    print(f"stageA best: thr={bestA.thr} sl={bestA.sl} trail={bestA.trail} minp={bestA.minp} maxh={bestA.maxh} cd={bestA.cd}", flush=True)

    print("=== stage B: leverage + regime filter (execution+threshold fixed from stage A) ===", flush=True)
    dfB = run(
        tapes, thresholds=[float(bestA.thr)], leverages=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        sl_atrs=[float(bestA.sl)], trail_atrs=[float(bestA.trail)], min_profit_atrs=[float(bestA.minp)],
        max_holds=[int(bestA.maxh)], cooldowns=[int(bestA.cd)],
        reg_modes=["not_chop", "trend_agree", "none"], reg_thrs=[0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50],
    )
    dfB = dfB.sort_values("calmar", ascending=False)
    dfB.to_csv(OUT_DIR / "val_stable_stageB_leverage_regime.csv", index=False)
    print(dfB.head(20).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
