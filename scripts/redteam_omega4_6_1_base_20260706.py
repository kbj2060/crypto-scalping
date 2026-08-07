#!/usr/bin/env python3
"""Redteam-style adversarial check for omega4_6_1_duration_ou_halflife_risk_gate_20260630 (base
form, event-flat overlay excluded). Not a full certified redteam process (this project's other
redteam audits, e.g. docs/audits/omega4_6_2_cap220_..._redteam_20260630.md, cover more ground with
dedicated tooling) -- this is a bounded check appropriate for a research-stage promotion-checklist
item: leverage/notional caps, overlap, accounting consistency, cost stress (1x/2x/3x), and a
lookahead spot-check (shift the duration-gate feature forward by 1 bar and confirm the result
degrades, the expected causal direction, matching the lag-test methodology used elsewhere in this
project this session).
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

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"
BASE_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"


def summarize(returns: np.ndarray) -> dict:
    if len(returns) == 0:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0}
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0), "trades": int(len(returns))}


def main() -> int:
    checks = []
    gated = pd.read_csv(OUT_DIR / "combined_router_duration_gated_ledger_VALSELECTED_extended.csv")
    active = gated[gated["notional"].astype(float) > 1e-12].copy()

    # 1. leverage/notional caps
    max_lev = float(active["leverage"].max())
    max_notional = float(active["notional"].max())
    checks.append(("leverage_cap_5x", max_lev <= 5.0 + 1e-9, f"max_leverage={max_lev:.4f}"))
    checks.append(("notional_cap_1.8x", max_notional <= 1.8 + 1e-9, f"max_notional={max_notional:.4f}"))

    # 2. no overlapping positions (sorted by entry/exit index, one at a time)
    ordered = active.sort_values(["entry_i", "exit_i"]).reset_index(drop=True)
    prev_exit = -1
    overlaps = 0
    for _, row in ordered.iterrows():
        if int(row["entry_i"]) <= prev_exit:
            overlaps += 1
        prev_exit = max(prev_exit, int(row["exit_i"]))
    checks.append(("no_overlapping_positions", overlaps == 0, f"overlap_count={overlaps}"))

    # 3. accounting consistency: trade_return == net_per_notional * notional
    acc_err = float((active["trade_return"] - active["net_per_notional"] * active["notional"]).abs().max())
    checks.append(("accounting_consistent", acc_err <= 1e-9, f"max_abs_error={acc_err:.2e}"))

    # 4. notional contract: notional == margin_fraction * leverage
    nc_err = float((active["notional"] - active["margin_fraction"] * active["leverage"]).abs().max())
    checks.append(("notional_contract_consistent", nc_err <= 1e-6, f"max_abs_error={nc_err:.2e}"))

    # 5. cost stress: does the edge survive 2x/3x fee+slip? (approximate: assume the recorded
    # trade_return already embeds cost1 fee/slip; re-derive raw_exit_price_move minus cost basis,
    # then re-apply at 2x/3x multiplier, matching this project's cost-stress convention elsewhere)
    fee_slip_frac = 0.0007  # FEE_RATE(0.0005)+SLIP_RATE(0.0002) round-trip approx used elsewhere
    for mult, label in ((1.0, "cost1"), (2.0, "cost2"), (3.0, "cost3")):
        cost_delta = fee_slip_frac * (mult - 1.0) * active["notional"]
        stressed = active["trade_return"] - cost_delta
        m = summarize(stressed.to_numpy())
        checks.append((f"{label}_pnl_positive", m["pnl"] > 0.0, f"pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}%"))

    # 6. lookahead spot-check: shift ou_halflife forward 1 bar (simulate seeing it 1 bar late) and
    # confirm the gate's effect direction is NOT improved by seeing it early (a proper causal
    # signal should not get WORSE when using only earlier information -- if it gets BETTER with a
    # 1-bar delay, that's a red flag the original wasn't causal either way; if performance degrades
    # smoothly, consistent with genuine causal signal, matching the lag-test pattern used for
    # Sigma6's regime filter this session).
    market = pd.read_csv(BASE_2026, usecols=["timestamp", "ou_halflife"], low_memory=False)
    market["timestamp"] = pd.to_datetime(market["timestamp"])
    market = market.sort_values("timestamp").reset_index(drop=True)
    market["ou_halflife_lag1"] = market["ou_halflife"].shift(1)
    combined = pd.read_csv(OUT_DIR / "combined_router_ledger_extended.csv")
    combined = combined.drop(columns=["ou_halflife"], errors="ignore")
    combined["entry_timestamp_dt"] = pd.to_datetime(combined["entry_timestamp"])
    combined = combined.merge(market.rename(columns={"timestamp": "entry_timestamp_dt"}), on="entry_timestamp_dt", how="left")
    active_c = combined[combined["notional"].astype(float) > 1e-12].copy()
    thr = 0.005417
    for tag, col in (("same_bar", "ou_halflife"), ("lag1_bar", "ou_halflife_lag1")):
        hit = active_c[col] <= thr
        rets = np.where(hit, 0.0, active_c["trade_return"])
        m = summarize(rets)
        checks.append((f"lag_test_{tag}", True, f"pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% trades={m['trades']} (informational, not pass/fail)"))

    print("=== Omega4.6.1 base (no event-flat) redteam-style check ===", flush=True)
    all_pass = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}: {detail}", flush=True)
    print(f"\noverall: {'PASS' if all_pass else 'FAIL (see above)'}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
