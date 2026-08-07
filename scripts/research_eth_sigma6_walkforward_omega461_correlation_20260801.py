#!/usr/bin/env python3
"""Two follow-ups to the walk-forward-confirmed Sigma6 regime-filter candidate
(thr=0.60/lev=3.0/sl=1.5/not_chop/rthr=0.50/stab=0.0 -- beat baseline in 4/5 leave-one-window-out
folds, same config selected every fold, see
project-sigma6-regime-filter-leave-one-window-out-CANDIDATE-20260801.md):

1. Standalone continuous backtest across the full available tape range (no window slicing) --
   what does this candidate look like as one continuous track record, not 5 separate windows?
2. Correlation/overlap vs live Omega4.6.1, reusing the same daily-allocated-return + block-bootstrap
   methodology already validated in research_eth_sigma3_1h_omega461_correlation_20260731.py.
   DIAGNOSTIC ONLY per CLAUDE.md -- not a promotion decision.

Data-availability note: the saved Omega4.6.1 greedy-router ledgers only cover 2025-10-01 onward
(greedy_router_ledger_VAL.csv: 2025-10-01..12-31; greedy_router_ledger_extended.csv:
2026-01-01..06-25) -- there is no Omega4.6.1 baseline to correlate against for 2025-07..09, so that
part is excluded from the correlation check (still included in the standalone backtest).
"""

from __future__ import annotations

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
from run_sigma6_regime_trend_20260705 import load_tape_with_regime, backtest  # noqa: E402
from research_eth_sigma3_1h_omega461_joint_portfolio_20260731 import (  # noqa: E402
    omega_trades_from_ledger, OMEGA_VAL_LEDGER, OMEGA_OOS_LEDGER,
)
from research_eth_sigma3_1h_omega461_correlation_20260731 import (  # noqa: E402
    daily_allocated_returns, occupancy_mask, block_bootstrap_corr_ci,
)

OUT_DIR = ROOT / "tmp/research_20260801/sigma6_walkforward_omega461_correlation"
BASE_KW = dict(margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144, cooldown=3, fee_mult=1.0)
WINNER = dict(thr=0.60, lev=3.0, sl=1.5, mode="not_chop", rthr=0.50, stab=0.0)

FULL_START, FULL_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2026-06-30 23:59:59")
# VAL ledger covers 2025-10-01..12-31, extended ledger covers 2026-01-01..06-25 -- correlation
# windows must stay inside those ranges since that's all the saved Omega4.6.1 trade data covers.
CORR_WINDOWS = [
    ("VAL_2025Q4", pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59"), OMEGA_VAL_LEDGER),
    ("OOS_2026H1", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-25 23:59:59"), OMEGA_OOS_LEDGER),
]


def sig6_trades(tapes: dict, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    r = backtest(tapes[WINNER["thr"]], leverage=WINNER["lev"], sl_atr=WINNER["sl"], reg_mode=WINNER["mode"],
                 reg_thr=WINNER["rthr"], stab_thr=WINNER["stab"], start=start, end=end, **BASE_KW)
    return [{"entry_timestamp": t["entry_timestamp"], "exit_timestamp": t["exit_timestamp"], "trade_return": t["ret"]}
            for t in r["trade_list"]]


def analyze(label: str, start: pd.Timestamp, end: pd.Timestamp, s6_trades: list[dict], om_trades: list[dict]) -> dict:
    day_index = pd.date_range(start.floor("D"), end.floor("D"), freq="D")
    s6_daily = daily_allocated_returns(s6_trades, day_index, "trade_return", None)
    om_daily = daily_allocated_returns(om_trades, day_index, "trade_return", None)

    corr = s6_daily.corr(om_daily)
    n_nonzero_both = int(((s6_daily != 0) & (om_daily != 0)).sum())
    ci = block_bootstrap_corr_ci(s6_daily, om_daily)

    s6_occ = occupancy_mask(s6_trades, day_index)
    om_occ = occupancy_mask(om_trades, day_index)
    both_occ = s6_occ & om_occ
    overlap_days = int(both_occ.sum())

    same_sign = float("nan")
    if overlap_days > 0:
        s6_sub, om_sub = s6_daily[both_occ], om_daily[both_occ]
        valid = (s6_sub != 0) & (om_sub != 0)
        if valid.sum() > 0:
            same_sign = float((np.sign(s6_sub[valid]) == np.sign(om_sub[valid])).mean())

    print(f"===== {label} {start.date()}..{end.date()} =====")
    print(f"  calendar days: {len(day_index)}   sigma6-filtered trades: {len(s6_trades)}   omega461 trades: {len(om_trades)}")
    print(f"  sigma6 days-in-position: {int(s6_occ.sum())}/{len(day_index)} ({100*s6_occ.mean():.1f}%)")
    print(f"  omega461 days-in-position: {int(om_occ.sum())}/{len(day_index)} ({100*om_occ.mean():.1f}%)")
    print(f"  BOTH-in-position days: {overlap_days} ({100*overlap_days/len(day_index):.1f}% of window)")
    print(f"  daily-allocated-return Pearson correlation: {corr:.3f}  (n_days_both_nonzero={n_nonzero_both})")
    print(f"  block-bootstrap (14d blocks, n=5000) 90% CI: [{ci['p05']:.3f}, {ci['p95']:.3f}]  median={ci['median']:.3f}  P(corr>0)={ci['prob_positive']:.2f}")
    print(f"  sign agreement on both-occupied days: {same_sign if same_sign == same_sign else 'n/a'}")
    print()
    return {"label": label, "corr": corr, "ci_p05": ci["p05"], "ci_p95": ci["p95"], "overlap_days": overlap_days}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_tape_with_regime()
    tapes = {thr: v2.apply_quality_threshold(raw, thr) for thr in (0.60, 0.70)}

    # 1) Standalone continuous backtest, full available range.
    r = backtest(tapes[WINNER["thr"]], leverage=WINNER["lev"], sl_atr=WINNER["sl"], reg_mode=WINNER["mode"],
                 reg_thr=WINNER["rthr"], stab_thr=WINNER["stab"], start=FULL_START, end=FULL_END, **BASE_KW)
    r_base = backtest(tapes[0.60], leverage=3.0, sl_atr=1.5, reg_mode="none", reg_thr=0.34, stab_thr=0.0,
                      start=FULL_START, end=FULL_END, **BASE_KW)
    print(f"=== Standalone continuous backtest {FULL_START.date()}..{FULL_END.date()} ===")
    print(f"  candidate (not_chop rthr0.50): pnl={r['pnl']:.2f}%  mdd={r['mdd']:.2f}%  trades={r['trades']}  wr={r['wr']:.3f}")
    print(f"  baseline  (no filter):         pnl={r_base['pnl']:.2f}%  mdd={r_base['mdd']:.2f}%  trades={r_base['trades']}  wr={r_base['wr']:.3f}")
    print()

    # 2) Correlation vs Omega4.6.1, per sub-window where a ledger exists, plus pooled.
    summary = []
    all_s6, all_om = [], []
    for label, start, end, ledger in CORR_WINDOWS:
        s6 = sig6_trades(tapes, start, end)
        om = omega_trades_from_ledger(ledger, start, end)
        summary.append(analyze(label, start, end, s6, om))
        all_s6.extend(s6)
        all_om.extend(om)

    pooled_start, pooled_end = CORR_WINDOWS[0][1], CORR_WINDOWS[-1][2]
    summary.append(analyze("POOLED (2025-10-01..2026-06-25)", pooled_start, pooled_end, all_s6, all_om))

    pd.DataFrame(summary).to_csv(OUT_DIR / "correlation_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
