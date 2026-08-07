#!/usr/bin/env python3
"""Fresh-forward re-test of Sigma6 (1h trend-scan + regime filter) on TODAY's data, using this
project's CANONICAL VAL/OOS split (VAL 2025-10-01..12-31 to avoid the 2025-09 TRAIN-leak window
noted in project-eth-omega461-exit-logic-experiments-20260721.md; OOS 2026-01-01..03-31), not the
original contract's own VAL/OOS windows (2025-07..12 / 2026-03-02..06-30).

Rationale: the original Sigma6 frozen numbers do not reproduce on retrain even on their own
unchanged window (docs/model_contracts/sigma6_fresh_window_attempt_20260720.md) -- traced to
upstream 2024 feature drift, not a code bug, and consistent with this project's recurring
"source data retroactively revised, old baseline unrecoverable" pattern (see
project-omega461-baseline-drift-bisection-20260730.md). Chasing bit-exact reproduction of the old
number is therefore not useful; this script treats today's retrain as a NEW candidate and requires
it to clear the VAL-then-OOS funnel fresh, per the project's Fresh-Forward Validation/OOS rule
(CLAUDE.md) -- no saved ledger/trade-return data used as input, pure bar-by-bar causal replay.

VAL-first funnel: sweep grid on VAL only, select winners that beat a no-regime-filter baseline on
BOTH pnl and mdd, THEN touch OOS exactly once for those pre-registered points.
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

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402
from run_sigma6_regime_trend_20260705 import load_tape_with_regime, backtest  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260731/sigma6_freshforward_canonical"
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31 23:59:59")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_tape_with_regime()
    tapes = {thr: v2.apply_quality_threshold(raw, thr) for thr in (0.60, 0.70)}
    base = dict(margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144, cooldown=3)

    grid = list(itertools.product(
        [0.60, 0.70],
        [3.0, 4.0],
        [1.5, 2.5],
        ["trend_agree", "not_chop", "none"],
        [0.34, 0.42, 0.50],
        [0.0, 0.55],
    ))
    rows = []
    baseline = None
    for thr, lev, sl, mode, rthr, stab in grid:
        if mode == "none" and (rthr != 0.34 or stab != 0.0):
            continue
        r = backtest(tapes[thr], leverage=lev, sl_atr=sl, reg_mode=mode, reg_thr=rthr, stab_thr=stab,
                     fee_mult=1.0, start=VAL_START, end=VAL_END, **base)
        row = {"thr": thr, "lev": lev, "sl": sl, "mode": mode, "rthr": rthr, "stab": stab,
               "val_pnl": round(r["pnl"], 2), "val_mdd": round(r["mdd"], 2), "val_trades": r["trades"],
               "val_wr": round(r["wr"], 3)}
        rows.append(row)
        if mode == "none" and thr == 0.60 and lev == 3.0 and sl == 1.5 and stab == 0.0:
            baseline = row
    df = pd.DataFrame(rows).sort_values("val_pnl", ascending=False)
    df.to_csv(OUT_DIR / "val_grid.csv", index=False)
    print(f"=== VAL {VAL_START.date()}..{VAL_END.date()}, baseline(no filter, thr0.6/lev3/sl1.5): "
          f"pnl={baseline['val_pnl']}% mdd={baseline['val_mdd']}% trades={baseline['val_trades']} ===")
    print(df.head(15).to_string(index=False))

    winners = df[(df["val_pnl"] > baseline["val_pnl"]) & (df["val_mdd"] > baseline["val_mdd"])]
    print(f"\n{len(winners)} VAL winners (beat baseline on BOTH pnl and mdd):")
    print(winners.to_string(index=False))

    if winners.empty:
        print("\nNo VAL winners -> OOS not touched (VAL-first funnel).")
        return 0

    oos_rows = []
    for _, w in winners.iterrows():
        r = backtest(tapes[w["thr"]], leverage=w["lev"], sl_atr=w["sl"], reg_mode=w["mode"],
                     reg_thr=w["rthr"], stab_thr=w["stab"], fee_mult=1.0,
                     start=OOS_START, end=OOS_END, **base)
        oos_rows.append({**w.to_dict(), "oos_pnl": round(r["pnl"], 2), "oos_mdd": round(r["mdd"], 2),
                          "oos_trades": r["trades"], "oos_wr": round(r["wr"], 3)})
    # also score the baseline itself on OOS for reference
    rb = backtest(tapes[0.60], leverage=3.0, sl_atr=1.5, reg_mode="none", reg_thr=0.34, stab_thr=0.0,
                  fee_mult=1.0, start=OOS_START, end=OOS_END, **base)
    print(f"\nOOS baseline: pnl={rb['pnl']:.2f}% mdd={rb['mdd']:.2f}% trades={rb['trades']}")

    odf = pd.DataFrame(oos_rows)
    odf.to_csv(OUT_DIR / "oos_confirm.csv", index=False)
    print(f"\n=== OOS {OOS_START.date()}..{OOS_END.date()} confirmation of {len(winners)} pre-registered VAL winners ===")
    print(odf.to_string(index=False))

    cleared = odf[(odf["oos_pnl"] > rb["pnl"]) & (odf["oos_mdd"] > rb["mdd"])]
    print(f"\n{len(cleared)}/{len(odf)} cleared OOS on BOTH pnl and mdd vs OOS baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
