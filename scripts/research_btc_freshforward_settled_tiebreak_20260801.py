#!/usr/bin/env python3
"""Fresh-forward-only BTC CryptoMamba regime_tiebreak check, restricted to bars that are BOTH
(a) genuinely new relative to Leg A/h48qual's original ledger cutoff (2026-06-25, exclusive) and
(b) "settled" per the data-finality buffer policy established in
scripts/extend_regime3_wide24_sol_btc_20260721.py (rows within 48h of the current extension's own
tail are provisional and excluded).

Leg A: the freshforward-extended ledger from
tmp/causal_regen_20260516/btc_final_scale_map_20260708_freshforward_ext/oos_ledger.csv (produced by
scripts/apply_final_scale_map_btc_freshforward_ext_20260801.py, rerun today against the just-
regenerated regime3 wide24 sidecar).
Leg B: UNCHANGED frozen ledger, tmp/causal_regen_20260516/btc_v2_regime_trendscan_hgb_20260714/.
Tiebreak: UNCHANGED CryptoMamba future-regime mechanism from
research_btc_tau1_cryptomamba_tiebreak_20260801.py (rule_weights/leg_side_series/weighted_pnl reused
verbatim from research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801.py).

Window: FRESHFORWARD_SETTLED_2026-06-25_to_<settled_cutoff>, where settled_cutoff is printed by
today's extend_regime3_wide24_sol_btc_20260721.py run (2026-07-19 11:45:00). A trade is only
"fully in-window" if BOTH its entry and exit timestamps fall inside [2026-06-25 00:00, settled_cutoff]
-- an open/incomplete trade whose exit lands in the provisional zone is excluded, not truncated,
since its ledger trade_return already bakes in provisional-zone feature values.

DIAGNOSTIC framing per CLAUDE.md Fresh-Forward Rule: this uses saved ledgers (both legs), so it is
NOT a bar-by-bar fresh-forward walk in the strict sense the policy defines for promotion purposes --
it inherits the same "diagnostic only" caveat every prior Tau1/CryptoMamba-tiebreak script in this
project has carried. What IS new here relative to prior runs is that the underlying bars themselves
(2026-06-25 onward) were never seen by Leg A's original frozen training/ledger-construction process,
and are now (for the settled portion) verified stable/reproducible bar feature values under the
finality-buffer discipline -- i.e. this is the first time this project has been able to trust ANY
BTC bars past 2026-06-25 enough to look at them at all.
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

from research_eth_sigma3_1h_omega461_joint_portfolio_20260731 import (  # noqa: E402
    omega_trades_from_ledger, build_leg_equity_path, summarize_equity,
)
from research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801 import (  # noqa: E402
    load_5m_prices_btc, leg_side_series, rule_weights, weighted_pnl,
)
from research_btc_tau1_cryptomamba_tiebreak_20260801 import load_cryptomamba_regime  # noqa: E402

LEG_A_FRESHFORWARD_LEDGER = ROOT / "tmp/causal_regen_20260516/btc_final_scale_map_20260708_freshforward_ext/oos_ledger.csv"
LEG_B_LEDGERS = [
    ROOT / "tmp/causal_regen_20260516/btc_v2_regime_trendscan_hgb_20260714/validation_ledger.csv",
    ROOT / "tmp/causal_regen_20260516/btc_v2_regime_trendscan_hgb_20260714/oos_ledger.csv",
]

WINDOW_START = pd.Timestamp("2026-06-25 00:00:00")
SETTLED_CUTOFF = pd.Timestamp("2026-07-19 11:45:00")  # printed by today's extend_regime3_wide24 run
CMAMBA_MAX = pd.Timestamp("2026-07-12 16:50:00")  # actual max timestamp found in the cmamba regime csv


def load_all_trades(paths, start, end) -> list[dict]:
    trades = []
    for p in paths:
        trades.extend(omega_trades_from_ledger(p, start, end))
    return trades


def fully_in_window(trades: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    return [t for t in trades if t["entry_timestamp"] >= start and t["exit_timestamp"] <= end]


def main() -> int:
    prices = load_5m_prices_btc()
    regime = load_cryptomamba_regime()
    cmamba_max_actual = regime["timestamp"].max()
    print(f"cmamba regime data actually extends to: {cmamba_max_actual} "
          f"(dir name says 20260721 but data is stale past this)")

    label = f"FRESHFORWARD_SETTLED_2026-06-25_to_{SETTLED_CUTOFF.date()}"
    start, end = WINDOW_START, SETTLED_CUTOFF

    trades_a_raw = omega_trades_from_ledger(LEG_A_FRESHFORWARD_LEDGER, start, end)
    trades_b_raw = load_all_trades(LEG_B_LEDGERS, start, end)
    trades_a = fully_in_window(trades_a_raw, start, end)
    trades_b = fully_in_window(trades_b_raw, start, end)

    print(f"\n=== {label} ({start.date()} .. {end}) ===")
    print(f"Leg A trades with entry in-window: {len(trades_a_raw)}; fully in-window (entry AND exit settled): {len(trades_a)}")
    for t in trades_a_raw:
        flag = "OK" if t in trades_a else "EXCLUDED(exit past settled cutoff)"
        print(f"  leg_a  entry={t['entry_timestamp']}  exit={t['exit_timestamp']}  ret={t['trade_return']:+.4f}  {flag}")
    print(f"Leg B trades with entry in-window: {len(trades_b_raw)}; fully in-window: {len(trades_b)}")
    for t in trades_b_raw:
        flag = "OK" if t in trades_b else "EXCLUDED(exit past settled cutoff)"
        print(f"  leg_b  entry={t['entry_timestamp']}  exit={t['exit_timestamp']}  ret={t['trade_return']:+.4f}  {flag}")

    if len(trades_a) < 5 or len(trades_b) < 5:
        print(f"\n*** THIN-SAMPLE WARNING: only {len(trades_a)} Leg-A / {len(trades_b)} Leg-B trades are "
              f"fully within the genuinely-new + settled window. Any pnl/mdd numbers below are NOT "
              f"sufficient to draw a promotion or architecture conclusion -- reporting them for the "
              f"record only. ***\n")

    eq_a = build_leg_equity_path(trades_a, prices, start, end, use_ledger_trade_return=True)
    eq_b = build_leg_equity_path(trades_b, prices, start, end, use_ledger_trade_return=True)
    eq_a_1h = eq_a.resample("1h").last().ffill()
    eq_b_1h = eq_b.resample("1h").last().ffill()
    ts = pd.Series(eq_a_1h.index)
    side_a = leg_side_series(trades_a, ts)
    side_b = leg_side_series(trades_b, ts)
    active_a, active_b = side_a != 0, side_b != 0
    conflict = active_a & active_b & (side_a != side_b)

    reg = regime[(regime["timestamp"] >= start) & (regime["timestamp"] <= end)]
    reg_frame = pd.merge_asof(pd.DataFrame({"timestamp": ts}), reg, on="timestamp", direction="backward")
    bull = reg_frame["bull_prob"].fillna(0.5).to_numpy()
    bear = reg_frame["bear_prob"].fillna(0.5).to_numpy()
    n_stale_regime_bars = int((ts > CMAMBA_MAX).sum())

    delta_a = eq_a_1h.diff().fillna(0.0).to_numpy()
    delta_b = eq_b_1h.diff().fillna(0.0).to_numpy()

    w_a, w_b = rule_weights(conflict, side_a, side_b, bull, bear)
    tiebreak = weighted_pnl(delta_a, delta_b, w_a, w_b)
    eq_ab_baseline = 1.0 + (eq_a - 1.0) + (eq_b - 1.0)
    sab_base = summarize_equity(eq_ab_baseline)
    sa, sb = summarize_equity(eq_a), summarize_equity(eq_b)

    n_conflict = int(conflict.sum())
    print(f"\nn_conflict_bars={n_conflict}, n_bars_with_stale(>{CMAMBA_MAX})_regime_carried_forward={n_stale_regime_bars}/{len(ts)}")
    print(f"Leg A alone      : pnl={sa['pnl_pct']:+.2f}% mdd={sa['mdd_pct']:.2f}% trades={len(trades_a)}")
    print(f"Leg B alone      : pnl={sb['pnl_pct']:+.2f}% mdd={sb['mdd_pct']:.2f}% trades={len(trades_b)}")
    print(f"Fixed 1x-1x      : pnl={sab_base['pnl_pct']:+.2f}% mdd={sab_base['mdd_pct']:.2f}%")
    print(f"CryptoMamba tiebreak: pnl={tiebreak['pnl_pct']:+.2f}% mdd={tiebreak['mdd_pct']:.2f}%")

    out_dir = ROOT / "tmp/research_20260801/btc_freshforward_settled_tiebreak"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "window": label, "start": str(start), "end": str(end),
        "leg_a_trades": len(trades_a), "leg_b_trades": len(trades_b),
        "leg_a_pnl": sa["pnl_pct"], "leg_a_mdd": sa["mdd_pct"],
        "leg_b_pnl": sb["pnl_pct"], "leg_b_mdd": sb["mdd_pct"],
        "baseline_pnl": sab_base["pnl_pct"], "baseline_mdd": sab_base["mdd_pct"],
        "cmamba_tiebreak_pnl": tiebreak["pnl_pct"], "cmamba_tiebreak_mdd": tiebreak["mdd_pct"],
        "n_conflict_bars": n_conflict, "n_stale_regime_bars": n_stale_regime_bars,
        "thin_sample": len(trades_a) < 5 or len(trades_b) < 5,
    }]).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nWrote {out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
