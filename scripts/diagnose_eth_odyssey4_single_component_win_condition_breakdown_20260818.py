#!/usr/bin/env python3
"""RESEARCH ONLY -- diagnostic, no retraining. Follow-up to
research_eth_odyssey4_single_vs_dual_component_contribution_20260817.py, which found h48qual and
zig075 alternate as the stronger standalone component window-by-window (h48qual wins 2025Q1 and
OOS-Q2, zig075 wins 2025Q2/Q3/VAL/OOS-Q1). This script asks WHERE within each window each
component's win rate is concentrated -- by exit reason, side, hold duration, router regime
(bull/bear/chop expert), trend state, session, and quality/confidence at entry -- using the exact
same greedy_replay + duration-gate methodology as the parent script, just with the per-trade
ledger enriched (via entry_signal_i, which greedy_replay already returns) with the market/model
state at entry instead of only the aggregate PnL/MDD/trades summary.

real_g0 (actual model direction) only. fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=
false, saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false. Diagnostic only
-- not a promotion or model-selection artifact (see CLAUDE.md Omega Artifact Integrity Promotion
Gate); this only characterizes win-rate structure of the already-existing G0 components.
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

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_single_component_win_condition_breakdown_20260818"
WINDOW_KEYS = ("2025q1", "2025q2", "2025q3", "val", "oos_q1", "oos_q2")
DURATION_THRESHOLD = greedy.DURATION_THRESHOLD
STATE_COLS = (
    "ou_halflife", "mtf_trend_1h", "mtf_trend_4h", "regime_trending",
    "session_europe", "session_us", "atr_pct_rank_288", "chop_index", "hurst_48",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def _enrich_ledger(ledger: pd.DataFrame, frame: pd.DataFrame, comp: dict, window: str, comp_name: str) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    df = ledger.copy()
    df["hold_bars"] = (df["exit_i"] - df["entry_i"]).clip(lower=0)
    entry_idx = df["entry_signal_i"].to_numpy()
    dec = comp["dec"]
    df["quality_score"] = dec["quality_score"].to_numpy()[entry_idx]
    df["confidence"] = dec["confidence"].to_numpy()[entry_idx]
    df["router_expert"] = dec["router_expert"].to_numpy()[entry_idx]
    for col in STATE_COLS:
        if col in frame.columns:
            df[col] = frame[col].to_numpy()[entry_idx]
    df["gated_chop"] = df["ou_halflife"] <= DURATION_THRESHOLD if "ou_halflife" in df.columns else False
    df["window"] = window
    df["component"] = comp_name
    return df


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = portfolio.DEVICE
    fee, slip = omega._load_fee_slip()

    windows = dict(gate.load_all_windows())
    h48qual_cfg = portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE)
    zig075_cfg = portfolio._component_cfg("zig075")

    all_ledgers = []
    for window_key in WINDOW_KEYS:
        w = windows[window_key]
        split = gate.WINDOW_DEFS[window_key]["split"]
        q_tags = {"h48qual": h48qual_cfg["q_tag"], "zig075": zig075_cfg["q_tag"]}
        aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, OUT_DIR)
        prep = portfolio._prepare_component_val if w["oof"] else greedy.prepare_component
        h48qual_comp = prep(aligned_frame, aligned_paths["h48qual"], h48qual_cfg, device)
        zig075_comp = prep(aligned_frame, aligned_paths["zig075"], zig075_cfg, device)

        _diag_h, ledger_h = greedy.greedy_replay(aligned_frame, {"h48qual": h48qual_comp}, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        _diag_z, ledger_z = greedy.greedy_replay(aligned_frame, {"zig075": zig075_comp}, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)

        enriched_h = _enrich_ledger(ledger_h, aligned_frame, h48qual_comp, window_key, "h48qual")
        enriched_z = _enrich_ledger(ledger_z, aligned_frame, zig075_comp, window_key, "zig075")
        all_ledgers.append(enriched_h)
        all_ledgers.append(enriched_z)
        log(f"{window_key}: h48qual trades={len(enriched_h)} zig075 trades={len(enriched_z)}")

    full = pd.concat(all_ledgers, ignore_index=True)
    out_path = OUT_DIR / "enriched_trade_ledger.csv"
    full.to_csv(out_path, index=False)
    log(f"\nwrote {out_path} rows={len(full)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
