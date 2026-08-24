#!/usr/bin/env python3
"""RESEARCH ONLY -- diagnostic (not a promotion candidate). User observation: Odyssey4's live/shadow
TP/SL is a nominally ATR-scaled formula (`tp=clip(max(min_tp,atr_pct*tp_mult),0,max_tp)`,
`sl=clip(max(min_sl,atr_pct*sl_mult),0,max_sl)`, trading_bot_modules/odyssey_live_adapter.py) but
with min_tp=0.075/min_sl=0.040 it is EFFECTIVELY FIXED: on real ETH 5m data (both 2025/2026 base
CSVs), atr_pct(192bar)*tp_mult(12) exceeds the 0.075 floor in only ~1.8-2.5% of bars -- the floor
wins ~97.5-98.5% of the time (computed this session, not reproduced here). User's diagnosis: 7.5%/4%
is "too big"; proposed direction: shrink the range so the exit_head (already root-caused this
session as PASSIVE due to its own label design, not feature visibility --
[[eth_odyssey4_exit_head_passivity_root_cause_20260817]]) gets more opportunity to actually decide
exits instead of the hard TP/SL barrier deciding almost everything.

Two-step plan agreed with the user: (1) THIS SCRIPT -- cheap, no retraining -- shrink ONLY the
min_tp/min_sl floor (tp_mult/sl_mult/atr_window/max_tp/max_sl UNCHANGED, exit_head UNCHANGED) and
measure how much more the exit_head actually gets to act (reason-breakdown share) and what that
does to PnL/MDD, purely diagnostic before any label-redesign/retrain decision. (2) a later,
separate, much larger step (exit label redesign + retrain) only if this step shows the floor is
truly the bottleneck.

Floor scale levels are chosen as a fixed geometric ladder (1x=current, 0.5x, 0.25x, 0.125x,
0.0625x) preserving the CURRENT min_tp:min_sl ratio (0.075:0.040) -- not swept against any PnL
outcome, purely a "how does behavior change as the floor shrinks" characterization curve.
tp_mult/sl_mult/atr_window/max_tp/max_sl and every trained model (direction/quality/exit heads)
are held fixed throughout -- only the floor clip constants change.

Runs the PLAIN G0 baseline (replay_omega4_6_1_greedy_router_20260706.greedy_replay -- no h48qual
regime-exit-guard, no zig075 entry-veto -- both are orthogonal mechanisms already characterized
elsewhere this session) on real_g0 (actual model direction, not direction-randomized) across the
same 6 pre-registered windows used throughout this line. fresh_forward_bar_by_bar=true,
trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false,
future_rows_used_for_entry=false. No live/shadow files touched.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
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
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import research_eth_odyssey4_random_direction_exit_reason_distribution_20260817 as reasons_mod  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_tpsl_floor_shrink_exit_head_engagement_20260817"
DEVICE = portfolio.DEVICE

CURRENT_MIN_TP, CURRENT_MIN_SL = 0.075, 0.040
FLOOR_SCALES = [1.0, 0.5, 0.25, 0.125, 0.0625]

WINDOW_KEYS = ("val", "oos_q1", "oos_q2", "2025q1", "2025q2", "2025q3")


def log(msg: str) -> None:
    print(msg, flush=True)


def _floor_cfg(name: str, scale: float) -> dict[str, Any]:
    # BUG FIX 2026-08-17: h48qual's actually-deployed component uses NEW_H48QUAL_BUNDLE
    # (liveATR-relabeled exit head, gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR) -- omitting
    # bundle_override here silently fell back to the pre-liveATR ORIGINAL bundle (essentially
    # 0% exit_head engagement by construction), invalidating every h48qual exit_head-engagement
    # number this script previously produced. zig075 has no override in the live config either
    # way, so it is unaffected.
    bundle_override = portfolio.NEW_H48QUAL_BUNDLE if name == "h48qual" else None
    cfg = portfolio._component_cfg(name, bundle_override=bundle_override)  # adds exit_threshold=0.95
    cfg["min_tp"] = CURRENT_MIN_TP * scale
    cfg["min_sl"] = CURRENT_MIN_SL * scale
    return cfg


def _reason_shares(reason_breakdown: dict[str, Any]) -> dict[str, float]:
    kept = reason_breakdown.get("kept", {})
    total = sum(kept.values())
    if total == 0:
        return {}
    return {k: round(v / total, 4) for k, v in sorted(kept.items(), key=lambda kv: -kv[1])}


def run_window_floor(window_key: str, aligned_frame: pd.DataFrame, aligned_paths: dict, scale: float,
                      device, fee: float, slip: float, oof: bool) -> dict[str, Any]:
    prep = portfolio._prepare_component_val if oof else greedy.prepare_component
    h48qual = prep(aligned_frame, aligned_paths["h48qual"], _floor_cfg("h48qual", scale), device)
    zig075 = prep(aligned_frame, aligned_paths["zig075"], _floor_cfg("zig075", scale), device)
    components = {"h48qual": h48qual, "zig075": zig075}

    diag, ledger = greedy.greedy_replay(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
    with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
    reason_bd = reasons_mod._reason_breakdown(ledger, aligned_frame, greedy.DURATION_THRESHOLD)

    return {
        "window": window_key, "floor_scale": scale,
        "min_tp": round(CURRENT_MIN_TP * scale, 5), "min_sl": round(CURRENT_MIN_SL * scale, 5),
        "pnl": with_gate["pnl"], "mdd": with_gate["mdd"], "trades": with_gate["trades"],
        "n_trades_raw": reason_bd["n_trades"], "reason_shares": _reason_shares(reason_bd),
    }


def _floor_bound_frac(frame: pd.DataFrame, atr_window: int, mult: float, min_val: float) -> float:
    atr_pct = atr_eval._atr_pct(frame, atr_window)
    return float((atr_pct * mult <= min_val).mean())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()

    log("=== stage=load_windows ===")
    windows = dict(gate.load_all_windows())

    rows: list[dict[str, Any]] = []
    for window_key in WINDOW_KEYS:
        w = windows[window_key]
        oof = bool(w["oof"])
        split = gate.WINDOW_DEFS[window_key]["split"]
        q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in sweep.COMPONENTS}
        aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, OUT_DIR)

        bound_1x = _floor_bound_frac(aligned_frame, 192, 12.0, CURRENT_MIN_TP)
        log(f"=== window={window_key} (n_bars={len(aligned_frame)}, oof={oof}) === "
            f"floor-bound frac @1x(current 7.5%): {bound_1x * 100:.2f}%")

        for scale in FLOOR_SCALES:
            r = run_window_floor(window_key, aligned_frame, aligned_paths, scale, device, fee, slip, oof)
            bound_frac = _floor_bound_frac(aligned_frame, 192, 12.0, CURRENT_MIN_TP * scale)
            r["floor_bound_frac"] = round(bound_frac, 4)
            rows.append(r)
            log(f"  scale={scale:<7} min_tp={r['min_tp']*100:5.3f}% min_sl={r['min_sl']*100:5.3f}% "
                f"floor_bound={bound_frac*100:5.2f}%  pnl={r['pnl']:+7.2f}%  mdd={r['mdd']:+7.2f}%  "
                f"trades={r['trades']:3d}  reasons={r['reason_shares']}")

    log("\n\n=== FINAL: TP/SL floor shrink sweep (exit_head UNCHANGED, diagnostic only) ===")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / "floor_shrink_sweep_summary.csv", index=False)
    log(f"\nwrote {OUT_DIR / 'floor_shrink_sweep_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
