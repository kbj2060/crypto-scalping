#!/usr/bin/env python3
"""BTC Tau1-style two-leg portfolio check: does adding a BTC h48qual (Leg A) sleeve to a BTC
Sigma9-1h trend-scan + regime-gate (Leg B) sleeve, on independent margin, combined via the SAME
regime_tiebreak rule validated for ETH's Tau1 (project-tau1-name-spec-20260801.md), beat either
leg alone -- the first time this exact two-independent-signal-on-one-asset combination has been
tried for BTC. Prior BTC combination attempts were either a single-model regime GATE
(btc_v2_regime_trendscan_hgb_20260714, itself OOS-negative when run alone) or a CROSS-ASSET blend
(sigma9_btc_eth_2asset_20260706_contract.md, ETH+BTC 50/50, diluted ETH returns) -- never two
independent BTC signal families combined the way Tau1 combines Omega4.6.1+Sigma6-filtered on ETH.

Leg A -- BTC h48qual Omega4.6.1-style candidate (status: research_positive_signal_not_live_wired,
see docs/model_contracts/btc_omega4_6_1_full_stack_20260708_contract.md), NO-duration-gate ledger:
tmp/causal_regen_20260516/btc_final_scale_map_20260708/{validation_ledger,oos_ledger}.csv.
Leg B -- BTC Sigma9 1h trend-scan + 2024-fit BTC HMM regime gate (status:
research_negative_result_not_adopted when run standalone), ledgers + regime probs:
tmp/causal_regen_20260516/btc_v2_regime_trendscan_hgb_20260714/{validation_ledger,oos_ledger,
validation_oos_predictions}.csv (regime3_current_sensitive_wide24_{bull,bear}_prob -- BTC's OWN
HMM, not ETH's).

Windows: VAL=2025-10-01..2025-12-31 (Leg A's ledger already IS this exact window; Leg B's ledger
is trimmed to it here). OOS=2026-01-01..2026-06-25 (capped at Leg A's ledger's own last trade
date, so both legs get their full already-computed extended-OOS window rather than an arbitrarily
truncated one).

Equity model, additive-dollar-PnL independent-sleeve combination, and the exact regime_tiebreak
rule are UNCHANGED reuse of research_eth_sigma3_1h_omega461_joint_portfolio_20260731.py
(build_leg_equity_path, summarize_equity, omega_trades_from_ledger -- generic despite the ETH-
specific name) and eval_sigma6_omega_rule_and_meta_allocation_20260801.py's
rule_weights('regime_tiebreak') logic, both already audited for ETH. Both BTC legs use
use_ledger_trade_return=True (the ledger's own trade_return anchors the compounding, real 5m BTC
prices only supply the intrabar shape) -- this is the SAME safer path this project already used
for the Omega4.6.1 leg on ETH, not a new assumption, and sidesteps needing to verify whether each
BTC ledger's entry_price/exit_price columns are fee/slippage-consistent with trade_return.

Sanity checks against each leg's own already-published report numbers (per this project's
"reproduce before trusting" discipline) run BEFORE the combination is trusted:
- Leg A VAL Q4 (no duration gate): report says pnl=+7.45% mdd=-11.93% trades=16
- Leg B VAL 2025-H2 continuous (validation_ledger.csv IS this exact 44-trade window, matching
  report's "validation_full_2025_h2" row): report says pnl=+23.01% mdd=-7.47% trades=44
- Leg A OOS extended (2026-01-01..06-25): report says pnl=+22.69% mdd=-15.88% trades=30
Leg B has NO exact per-half anchor for VAL_2025Q4 alone or the 06-25-truncated OOS window: its
report's "second_*"/"oos_frozen_q1_2026" rows come from a SEPARATE backtest re-run starting flat at
each sub-window's boundary (see train_eval_btc_v2_regime_trendscan_20260714.py's _select_policy),
not from slicing the one continuous ledger used here by entry_timestamp -- the two are different,
both-legitimate methodologies (this script's slice-by-entry_timestamp approach matches the ETH
joint-portfolio precedent script exactly), so those two report rows are not used as anchors.

DIAGNOSTIC ONLY per CLAUDE.md Fresh-Forward Rule -- both windows have already been inspected
repeatedly in this project's history for EACH LEG SEPARATELY (not a genuinely blind Fresh-Forward
test), and neither leg is currently live-wired. This script answers "does the Tau1 combination
mechanism transfer to BTC candidates", not "should BTC be promoted".
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

OUT_DIR = ROOT / "tmp/research_20260801/btc_h48qual_sigma9regime_tau1_joint_portfolio"
BTC_PRICE_CSV = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"

LEG_A_DIR = ROOT / "tmp/causal_regen_20260516/btc_final_scale_map_20260708"
LEG_B_DIR = ROOT / "tmp/causal_regen_20260516/btc_v2_regime_trendscan_hgb_20260714"
PFX = "regime3_current_sensitive_wide24_"

VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-25 23:59:59")

WINDOWS = [
    ("VAL_2025Q4", VAL_START, VAL_END, LEG_A_DIR / "validation_ledger.csv", LEG_B_DIR / "validation_ledger.csv"),
    ("OOS_2026extended", OOS_START, OOS_END, LEG_A_DIR / "oos_ledger.csv", LEG_B_DIR / "oos_ledger.csv"),
]

# (pnl_pct, mdd_pct, trades) from each leg's own already-published report.json; None = no exact
# anchor available for that window (window boundaries don't line up with the published report).
REFERENCE = {
    "VAL_2025Q4": {"leg_a": (7.45, -11.93, 16), "leg_b": None},
    "OOS_2026extended": {"leg_a": (22.69, -15.88, 30), "leg_b": None},
}
# Separate one-off check: Leg B's ledger covers the full VAL 2025-H2, matching the report's
# "validation_full_2025_h2" row exactly -- verified once at import time below, not per-window.
LEG_B_H2_REFERENCE = (23.01, -7.47, 44)


def load_5m_prices_btc() -> pd.DataFrame:
    df = pd.read_csv(BTC_PRICE_CSV, usecols=["timestamp", "open", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").set_index("timestamp")


def load_regime_probs() -> pd.DataFrame:
    df = pd.read_csv(LEG_B_DIR / "validation_oos_predictions.csv",
                      usecols=["timestamp", f"{PFX}bull_prob", f"{PFX}bear_prob"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
    return df.sort_values("timestamp")


def leg_side_series(trades: list[dict], ts: pd.Series) -> np.ndarray:
    side = np.zeros(len(ts), dtype=np.float64)
    for tr in trades:
        mask = ((ts >= tr["entry_timestamp"]) & (ts <= tr["exit_timestamp"])).to_numpy()
        side[mask] = tr["side"]
    return side


def sanity_check(label: str, leg: str, eq: pd.Series, n_trades: int) -> None:
    ref = REFERENCE[label][leg]
    got = summarize_equity(eq)
    if ref is None:
        print(f"[sanity check {label}/{leg}] no exact reference window available -- "
              f"reproduced pnl={got['pnl_pct']:+.2f}% mdd={got['mdd_pct']:.2f}% trades={n_trades} (unverified)")
        return
    ref_pnl, ref_mdd, ref_trades = ref
    print(f"[sanity check {label}/{leg}] report pnl={ref_pnl:+.2f}% mdd={ref_mdd:.2f}% trades={ref_trades} | "
          f"reproduced pnl={got['pnl_pct']:+.2f}% mdd={got['mdd_pct']:.2f}% trades={n_trades}")
    assert n_trades == ref_trades, f"{label}/{leg} trade count mismatch: {n_trades} vs {ref_trades}"
    assert abs(got["pnl_pct"] - ref_pnl) < 1.5, f"{label}/{leg} pnl mismatch: {got['pnl_pct']} vs {ref_pnl}"


def rule_weights(conflict: np.ndarray, side_a: np.ndarray, side_b: np.ndarray,
                  bull: np.ndarray, bear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(conflict)
    w_a, w_b = np.ones(n), np.ones(n)
    regime_side = np.where(bull >= bear, 1, -1)
    w_a[conflict] = np.where(side_a[conflict] == regime_side[conflict], 1.0, 0.0)
    w_b[conflict] = np.where(side_b[conflict] == regime_side[conflict], 1.0, 0.0)
    return w_a, w_b


def weighted_pnl(delta_a: np.ndarray, delta_b: np.ndarray, w_a: np.ndarray, w_b: np.ndarray) -> dict:
    equity, peak, mdd = 1.0, 1.0, 0.0
    for i in range(len(delta_a)):
        equity += w_a[i] * delta_a[i] + w_b[i] * delta_b[i]
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1)
    return {"pnl_pct": (equity - 1) * 100, "mdd_pct": mdd * 100}


def run_window(label, start, end, leg_a_path, leg_b_path, prices, regime):
    trades_a = omega_trades_from_ledger(leg_a_path, start, end)
    trades_b = omega_trades_from_ledger(leg_b_path, start, end)

    eq_a = build_leg_equity_path(trades_a, prices, start, end, use_ledger_trade_return=True)
    eq_b = build_leg_equity_path(trades_b, prices, start, end, use_ledger_trade_return=True)
    sanity_check(label, "leg_a", eq_a, len(trades_a))
    sanity_check(label, "leg_b", eq_b, len(trades_b))

    eq_ab_baseline = 1.0 + (eq_a - 1.0) + (eq_b - 1.0)
    sab_base = summarize_equity(eq_ab_baseline)

    eq_a_1h = eq_a.resample("1h").last().ffill()
    eq_b_1h = eq_b.resample("1h").last().ffill()
    ts = pd.Series(eq_a_1h.index)
    side_a = leg_side_series(trades_a, ts)
    side_b = leg_side_series(trades_b, ts)
    active_a, active_b = side_a != 0, side_b != 0
    conflict = active_a & active_b & (side_a != side_b)

    # FIX 2026-08-02 (project-btc-run-window-merge-point-fixed-20260802.md): shift regime timestamp
    # +1h before merge_asof so the matched row is guaranteed at least 1h stale relative to the delta
    # window it gates (closes the same-bar look-ahead in the shared run_window() merge pattern).
    reg = regime[(regime["timestamp"] >= start) & (regime["timestamp"] <= end)].copy()
    reg["timestamp"] = reg["timestamp"] + pd.Timedelta(hours=1)
    reg_frame = pd.merge_asof(pd.DataFrame({"timestamp": ts}), reg, on="timestamp", direction="backward")
    bull = reg_frame[f"{PFX}bull_prob"].fillna(0.5).to_numpy()
    bear = reg_frame[f"{PFX}bear_prob"].fillna(0.5).to_numpy()

    delta_a = eq_a_1h.diff().fillna(0.0).to_numpy()
    delta_b = eq_b_1h.diff().fillna(0.0).to_numpy()

    w_a, w_b = rule_weights(conflict, side_a, side_b, bull, bear)
    tiebreak = weighted_pnl(delta_a, delta_b, w_a, w_b)

    n_conflict = int(conflict.sum())
    sa, sb = summarize_equity(eq_a), summarize_equity(eq_b)
    print(f"\n=== {label} {start.date()}..{end.date()} ===")
    print(f"Leg A (BTC h48qual) alone       : pnl={sa['pnl_pct']:+.2f}% mdd={sa['mdd_pct']:.2f}% trades={len(trades_a)}")
    print(f"Leg B (BTC sigma9+regime) alone : pnl={sb['pnl_pct']:+.2f}% mdd={sb['mdd_pct']:.2f}% trades={len(trades_b)}")
    print(f"Combined baseline (1x-1x fixed) : pnl={sab_base['pnl_pct']:+.2f}% mdd={sab_base['mdd_pct']:.2f}%")
    print(f"Combined regime_tiebreak (n_conflict_bars={n_conflict}): "
          f"pnl={tiebreak['pnl_pct']:+.2f}% mdd={tiebreak['mdd_pct']:.2f}%")
    return {
        "label": label, "leg_a_pnl": sa["pnl_pct"], "leg_a_mdd": sa["mdd_pct"],
        "leg_b_pnl": sb["pnl_pct"], "leg_b_mdd": sb["mdd_pct"],
        "baseline_pnl": sab_base["pnl_pct"], "baseline_mdd": sab_base["mdd_pct"],
        "tiebreak_pnl": tiebreak["pnl_pct"], "tiebreak_mdd": tiebreak["mdd_pct"],
        "n_conflict_bars": n_conflict,
    }


def check_leg_b_h2_reference(prices: pd.DataFrame) -> None:
    h2_start, h2_end = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31 23:59:59")
    trades = omega_trades_from_ledger(LEG_B_DIR / "validation_ledger.csv", h2_start, h2_end)
    eq = build_leg_equity_path(trades, prices, h2_start, h2_end, use_ledger_trade_return=True)
    got = summarize_equity(eq)
    ref_pnl, ref_mdd, ref_trades = LEG_B_H2_REFERENCE
    print(f"[sanity check leg_b VAL_2025_H2] report pnl={ref_pnl:+.2f}% mdd={ref_mdd:.2f}% trades={ref_trades} | "
          f"reproduced pnl={got['pnl_pct']:+.2f}% mdd={got['mdd_pct']:.2f}% trades={len(trades)}")
    assert len(trades) == ref_trades, f"leg_b H2 trade count mismatch: {len(trades)} vs {ref_trades}"
    assert abs(got["pnl_pct"] - ref_pnl) < 1.0, f"leg_b H2 pnl mismatch: {got['pnl_pct']} vs {ref_pnl}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_5m_prices_btc()
    regime = load_regime_probs()
    check_leg_b_h2_reference(prices)
    rows = [run_window(label, s, e, la, lb, prices, regime) for label, s, e, la, lb in WINDOWS]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "joint_portfolio_summary.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'joint_portfolio_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
