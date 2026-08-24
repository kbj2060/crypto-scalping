#!/usr/bin/env python3
"""RESEARCH ONLY -- diagnostic, no retraining. User proposal: drop zig075 and go single-component
(h48qual only), retraining everything, hypothesizing better performance. Before committing to a
full retrain (GPU time + this repo's promotion gates), this is the cheap, no-retrain question:
using the EXISTING, currently-deployed bundles for both components (h48qual=NEW_H48QUAL_BUNDLE
liveATR, zig075=original), does the combined G0 portfolio (h48qual+zig075, single-slot-shared,
PRIORITY=h48qual-first, L4.5 duration gate) actually outperform EITHER component running alone
under the SAME replay methodology (replay_omega4_6_1_greedy_router_20260706.greedy_replay +
mfe_width._duration_gated), across all 6 standard windows? If a component contributes little or
negatively to the combined result, dropping it is well-motivated even before any retrain; if both
contribute substantially and the combination beats each alone, removing either is likely to hurt
regardless of retraining.

real_g0 (actual model direction) only -- this is a structural/architecture question, not a
direction-quality question already extensively tested elsewhere this session.
fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_
used=false, future_rows_used_for_entry=false. No live/shadow files touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_single_vs_dual_component_contribution_20260817"
WINDOW_KEYS = ("val", "oos_q1", "oos_q2", "2025q1", "2025q2", "2025q3")


def log(msg: str) -> None:
    print(msg, flush=True)


def _metrics(ledger: pd.DataFrame, frame: pd.DataFrame) -> dict:
    wg = mfe_width._duration_gated(ledger, frame, greedy.DURATION_THRESHOLD)
    return {"pnl": round(wg["pnl"], 2), "mdd": round(wg["mdd"], 2), "trades": wg["trades"]}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = portfolio.DEVICE
    fee, slip = omega._load_fee_slip()

    windows = dict(gate.load_all_windows())
    h48qual_cfg = portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE)
    zig075_cfg = portfolio._component_cfg("zig075")

    rows = []
    for window_key in WINDOW_KEYS:
        w = windows[window_key]
        split = gate.WINDOW_DEFS[window_key]["split"]
        q_tags = {"h48qual": h48qual_cfg["q_tag"], "zig075": zig075_cfg["q_tag"]}
        aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, OUT_DIR)
        prep = portfolio._prepare_component_val if w["oof"] else greedy.prepare_component
        h48qual_comp = prep(aligned_frame, aligned_paths["h48qual"], h48qual_cfg, device)
        zig075_comp = prep(aligned_frame, aligned_paths["zig075"], zig075_cfg, device)

        _diag_both, ledger_both = greedy.greedy_replay(aligned_frame, {"h48qual": h48qual_comp, "zig075": zig075_comp}, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        _diag_h, ledger_h = greedy.greedy_replay(aligned_frame, {"h48qual": h48qual_comp}, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        _diag_z, ledger_z = greedy.greedy_replay(aligned_frame, {"zig075": zig075_comp}, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)

        m_both = _metrics(ledger_both, aligned_frame)
        m_h = _metrics(ledger_h, aligned_frame)
        m_z = _metrics(ledger_z, aligned_frame)
        row = {"window": window_key, "both_pnl": m_both["pnl"], "both_mdd": m_both["mdd"], "both_trades": m_both["trades"],
               "h48qual_only_pnl": m_h["pnl"], "h48qual_only_mdd": m_h["mdd"], "h48qual_only_trades": m_h["trades"],
               "zig075_only_pnl": m_z["pnl"], "zig075_only_mdd": m_z["mdd"], "zig075_only_trades": m_z["trades"]}
        rows.append(row)
        log(f"=== {window_key} ===")
        log(f"  BOTH (current G0):  pnl={m_both['pnl']:+7.2f}%  mdd={m_both['mdd']:+7.2f}%  trades={m_both['trades']}")
        log(f"  h48qual ONLY:       pnl={m_h['pnl']:+7.2f}%  mdd={m_h['mdd']:+7.2f}%  trades={m_h['trades']}")
        log(f"  zig075  ONLY:       pnl={m_z['pnl']:+7.2f}%  mdd={m_z['mdd']:+7.2f}%  trades={m_z['trades']}")

    log("\n=== SUMMARY ===")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / "single_vs_dual_summary.csv", index=False)
    log(f"\nwrote {OUT_DIR / 'single_vs_dual_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
