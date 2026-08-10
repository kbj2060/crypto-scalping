"""Sweep exit_threshold for both JM-regime3 components (h48qual, zig075) on the FULL Omega4.6.1
greedy router. NOT a VAL-selected tune -- there is no separate VAL window available at the router
level (retest.load_frame_current only has 2026 data; the component-level VAL/OOS split lives inside
each parent's own sidecar training, a different, earlier stage). This sweeps a small grid on the
SAME Jan-Feb window used throughout this session's router tests and reports EVERY cell, not just the
best, specifically to avoid presenting a cherry-picked number as a genuine tune.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import (  # noqa: E402
    DURATION_THRESHOLD, greedy_replay, prepare_component,
)

OUT_DIR = ROOT / "tmp/eth_greedy_router_exit_threshold_sweep_20260809"
START, END = "2026-01-01", "2026-02-28"
GRID = [0.80, 0.85, 0.90, 0.95, 0.99]

JM_H48QUAL_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_regime_jmlam4_20260809"
JM_H48QUAL_SIDECAR = ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_regime_jmlam4_q055_tuned_20260809/risk_sidecar.pkl"
JM_ZIG075_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zig075_regime_jmlam4_20260809"
JM_ZIG075_SIDECAR = ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_zig075_regime_jmlam4_q040_20260809/risk_sidecar.pkl"
JM_REGIME3_2026 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2026_maskedname.csv"


def curve_metrics(returns: np.ndarray) -> dict:
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return {"pnl": round(float((curve[-1] - 1.0) * 100.0), 4),
            "mdd": round(float(dd.min() * 100.0), 4),
            "trades": int(len(returns)),
            "wr": round(float((returns > 0).mean()), 4) if len(returns) else 0.0}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = retest.DEVICE

    retest.WIDE24_2026 = JM_REGIME3_2026
    retest.COMPONENTS["h48qual"]["bundle"] = JM_H48QUAL_DIR / "true_3head_tabm_bundle.pt"
    retest.COMPONENTS["h48qual"]["sidecar_pkl"] = JM_H48QUAL_SIDECAR
    retest.COMPONENTS["zig075"]["bundle"] = JM_ZIG075_DIR / "true_3head_tabm_bundle.pt"
    retest.COMPONENTS["zig075"]["sidecar_pkl"] = JM_ZIG075_SIDECAR

    frame = retest.load_frame_current(START, END)
    fee, slip = omega._load_fee_slip()

    h48qual_pred = pd.read_csv(JM_H48QUAL_DIR / "oos_predictions_q055.csv")
    h48qual_pred["timestamp"] = pd.to_datetime(h48qual_pred["timestamp"])
    zig075_pred = pd.read_csv(JM_ZIG075_DIR / "oos_predictions_q040.csv")
    zig075_pred["timestamp"] = pd.to_datetime(zig075_pred["timestamp"])

    common_ts = set(frame["timestamp"]) & set(h48qual_pred["timestamp"]) & set(zig075_pred["timestamp"])
    frame = frame[frame["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
    h48qual_pred = h48qual_pred[h48qual_pred["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
    zig075_pred = zig075_pred[zig075_pred["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
    print(f"rows={len(frame)}", flush=True)

    h48_tmp = OUT_DIR / "_aligned_h48qual.csv"
    zig_tmp = OUT_DIR / "_aligned_zig075.csv"
    h48qual_pred.to_csv(h48_tmp, index=False)
    zig075_pred.to_csv(zig_tmp, index=False)

    print("stage=prepare_component (once)", flush=True)
    comp_h48 = prepare_component(frame, h48_tmp, retest.COMPONENTS["h48qual"], device)
    comp_zig = prepare_component(frame, zig_tmp, retest.COMPONENTS["zig075"], device)

    market = frame[["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp_dt"})
    results = []
    for th_h48 in GRID:
        for th_zig in GRID:
            comp_h48["exit_threshold"] = th_h48
            comp_zig["exit_threshold"] = th_zig
            components = {"h48qual": comp_h48, "zig075": comp_zig}
            _, ledger = greedy_replay(frame, components, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
            returns = ledger["trade_return"].to_numpy(dtype=float)
            no_gate = curve_metrics(returns)
            led = ledger.copy()
            led["entry_timestamp_dt"] = pd.to_datetime(led["entry_timestamp"])
            led = led.merge(market, on="entry_timestamp_dt", how="left")
            hit = led["ou_halflife"] <= DURATION_THRESHOLD
            gated = curve_metrics(led.loc[~hit, "trade_return"].to_numpy(dtype=float))
            row = {"exit_threshold_h48qual": th_h48, "exit_threshold_zig075": th_zig,
                   "no_gate": no_gate, "with_gate": gated,
                   "source_component_counts": ledger["source_component"].value_counts().to_dict()}
            results.append(row)
            print(json.dumps(row), flush=True)

    (OUT_DIR / "sweep_result.json").write_text(json.dumps(results, indent=2))
    best = max(results, key=lambda r: r["no_gate"]["pnl"])
    print("\nBEST no_gate pnl cell (for reference only, NOT VAL-selected):")
    print(json.dumps(best, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
