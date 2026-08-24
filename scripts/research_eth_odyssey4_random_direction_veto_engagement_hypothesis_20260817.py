#!/usr/bin/env python3
"""RESEARCH ONLY -- isolates whether zig075's SHORT/sustained-uptrend entry-veto ENGAGEMENT RATE
(not raw price always_long-vs-short spread) explains why 2 of the 6 windows tested in the random-
direction ablation line showed a non-significant real_g0-vs-random PnL gap (OOS-Q1 t=0.33,
OOS-Q2 t=-0.74) despite having the LARGEST spreads (123.7pp, 75.9pp), while 4 other windows
(including some with much smaller spread) were strongly significant.

Motivating clue (already locked in the Odyssey4 contract, not re-derived here): the SHORT/uptrend
veto's engagement count for the REAL model is VAL=12 bars, OOS-Q1=0 bars, OOS-Q2=0 bars
(docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md G0 table). The veto
was completely INERT for the real model in both non-significant windows. Hypothesis: when
direction is randomized, the SAME external, direction-independent detector mask can intersect
very differently with a random arm's (uncorrelated) SHORT picks than it did with the real model's
own SHORT picks -- so a window where the real model happened to never trigger the veto could still
see random arms triggering it often (or vice versa), making "direction quality" and "veto exposure"
different things in exactly the windows where the raw price spread is largest. This script is
SCOPED to the SHORT/uptrend veto only, because that is the only veto active in the tested
Odyssey4 baseline (real_g0 in this whole line) -- the mirror LONG/downtrend veto (execution log #5)
is a separate, not-yet-promoted candidate and is NOT part of real_g0 here.

Method:
  1. For each of the 6 windows (3 downtrend + 3 ranging, same set as the large-N reverification),
     compute the pure price/detector background rate: what fraction of ALL bars have the
     sustained-uptrend detector mask active (guard._detector_mask_for_frame, unchanged from every
     prior script in this line -- zero new free parameters).
  2. Re-run real_g0 and N=15 random seeds per window (a fresh, independently-drawn seed batch --
     not the same 5 or 30 used before, to avoid any seed-specific artifact), this time capturing
     `diag["veto_bars"]` (flat-state bars where a SHORT signal was suppressed) alongside PnL for
     each run.
  3. Compare, per window: real_g0's veto_bars vs the random arms' veto_bars distribution, and
     correlate each random seed's veto_bars with that seed's own PnL (within-window).

Reuses (imports, never duplicates) research_eth_odyssey4_random_direction_risk_management_
ablation_20260817 (build_ablation_components, DEVICE/fee/slip, side selectors),
research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 (build_detector,
_detector_mask_for_frame, prepare_regime_aware_components), research_eth_omega461_zig075_short_
entry_veto_sustained_uptrend_20260814 (greedy_replay_entry_veto), eth_omega461_multiwindow_
confirmation_gate_20260814 (load_all_windows), research_eth_omega461_exit_sweep_20260721
(load_frame/BASE_2025/2026/WIDE24_2025/2026).

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false. No live files touched.
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

import research_eth_odyssey4_random_direction_risk_management_ablation_20260817 as abl  # noqa: E402
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402
import research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 as veto_mod  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_random_direction_veto_engagement_hypothesis_20260817"
N_SEEDS = 15

RANGING_CANDIDATES = [
    {"key": "ranging_2025_05_12_to_07_07", "start": "2025-05-12", "end": "2025-07-07",
     "oof": True, "split": "train", "base_csv": sweep.BASE_2025, "wide24_csv": sweep.WIDE24_2025},
    {"key": "ranging_2025_03_10_to_05_05", "start": "2025-03-10", "end": "2025-05-05",
     "oof": True, "split": "train", "base_csv": sweep.BASE_2025, "wide24_csv": sweep.WIDE24_2025},
    {"key": "ranging_2026_02_09_to_04_06", "start": "2026-02-09", "end": "2026-04-06",
     "oof": False, "split": "oos", "base_csv": sweep.BASE_2026, "wide24_csv": sweep.WIDE24_2026},
]
DOWNTREND_WINDOW_KEYS = ("val", "oos_q1", "oos_q2")

# From the large-N (N=30) reverification -- reused here only as annotation, not recomputed.
N30_RESULT = {
    "ranging_2025_05_12_to_07_07": {"spread_pp": 4.52, "t_stat": -3.36},
    "ranging_2025_03_10_to_05_05": {"spread_pp": 7.56, "t_stat": -5.05},
    "ranging_2026_02_09_to_04_06": {"spread_pp": 20.06, "t_stat": 4.72},
    "val": {"spread_pp": 72.94, "t_stat": 2.61},
    "oos_q2": {"spread_pp": 75.85, "t_stat": -0.74},
    "oos_q1": {"spread_pp": 123.74, "t_stat": 0.33},
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_custom_window(cand: dict) -> dict[str, Any]:
    frame = sweep.load_frame(cand["start"], cand["end"], base_csv=cand["base_csv"], wide24_csv=cand["wide24_csv"])
    frame, n_dropped = gate._drop_route_nan(frame)
    gate.WINDOW_DEFS[cand["key"]] = {"split": cand["split"], "base_csv": cand["base_csv"], "wide24_csv": cand["wide24_csv"]}
    return {"frame": frame, "oof": cand["oof"], "tier": "ranging_retest", "route_nan_dropped": n_dropped}


def run_arm_with_veto(arm_label: str, window_key: str, windows: dict, score_by_base: dict, threshold: float,
                       device, fee: float, slip: float, *, side_selector) -> dict[str, Any]:
    """Same shape as abl.run_arm but also returns diag['veto_bars']."""
    aligned_frame, components = abl.build_ablation_components(
        window_key, windows, score_by_base, threshold, OUT_DIR, device, side_selector=side_selector,
    )
    diag, ledger = veto_mod.greedy_replay_entry_veto(
        aligned_frame, components, fee=fee, slip=slip, cost_mult=abl.sweep.COST_MULT, device=device,
    )
    with_gate = mfe_width._duration_gated(ledger, aligned_frame, abl.greedy.DURATION_THRESHOLD)
    return {"arm": arm_label, "pnl": with_gate["pnl"], "mdd": with_gate["mdd"], "trades": with_gate["trades"],
            "veto_bars": diag.get("veto_bars", 0)}


def run_real_g0_with_veto(window_key: str, windows: dict, score_by_base: dict, threshold: float,
                           device, fee: float, slip: float) -> dict[str, Any]:
    aligned_frame, components, _prep_diag = guard.prepare_regime_aware_components(
        window_key, windows, score_by_base, threshold, OUT_DIR, device,
    )
    mask, _ = guard._detector_mask_for_frame(aligned_frame, window_key, score_by_base, threshold)
    components["zig075"]["short_entry_veto_mask"] = mask
    diag, ledger = veto_mod.greedy_replay_entry_veto(
        aligned_frame, components, fee=fee, slip=slip, cost_mult=abl.sweep.COST_MULT, device=device,
    )
    with_gate = mfe_width._duration_gated(ledger, aligned_frame, abl.greedy.DURATION_THRESHOLD)
    return {"arm": "real_g0", "pnl": with_gate["pnl"], "mdd": with_gate["mdd"], "trades": with_gate["trades"],
            "veto_bars": diag.get("veto_bars", 0)}


def run_window(window_key: str, windows: dict, score_by_base: dict, threshold: float, device, fee: float, slip: float,
               seeds: list[int]) -> dict[str, Any]:
    log(f"=== window={window_key} ===")

    # cheap alignment-only probe (no model inference) just to get the row-aligned frame for the
    # detector mask -- avoids running a full build_ablation_components() pass only to read .frame.
    w = windows[window_key]
    split = gate.WINDOW_DEFS[window_key]["split"]
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR}
    aligned_frame_probe, _ = gate.align_frame_and_predictions(w["frame"], q_tags, split, OUT_DIR)
    mask, n_nan = guard._detector_mask_for_frame(aligned_frame_probe, window_key, score_by_base, threshold)
    detector_active_frac = float(mask.mean())
    log(f"  detector background rate: {detector_active_frac:.4f} ({int(mask.sum())}/{len(mask)} bars, uptrend-active)")

    real = run_real_g0_with_veto(window_key, windows, score_by_base, threshold, device, fee, slip)
    log(f"  real_g0: pnl={real['pnl']:+.2f}%  veto_bars={real['veto_bars']}")

    random_results = []
    for i, seed in enumerate(seeds):
        r = run_arm_with_veto(f"random_seed{seed}", window_key, windows, score_by_base, threshold, device, fee, slip,
                               side_selector=lambda n, _seed=seed: abl._side_selector_random(n, _seed))
        random_results.append(r)
        if (i + 1) % 5 == 0:
            log(f"  ...random seeds done: {i + 1}/{len(seeds)}")

    veto_bars_arr = np.array([r["veto_bars"] for r in random_results])
    pnl_arr = np.array([r["pnl"] for r in random_results])
    corr = float(np.corrcoef(veto_bars_arr, pnl_arr)[0, 1]) if veto_bars_arr.std() > 0 and pnl_arr.std() > 0 else float("nan")

    return {
        "window": window_key, "detector_active_frac": detector_active_frac,
        "real_g0_veto_bars": real["veto_bars"], "real_g0_pnl": real["pnl"],
        "random_veto_bars_mean": float(veto_bars_arr.mean()), "random_veto_bars_std": float(veto_bars_arr.std()),
        "random_veto_bars_min": int(veto_bars_arr.min()), "random_veto_bars_max": int(veto_bars_arr.max()),
        "veto_bars_vs_pnl_corr_within_window": corr,
        "n30_spread_pp": N30_RESULT[window_key]["spread_pp"], "n30_t_stat": N30_RESULT[window_key]["t_stat"],
        "random_details": random_results,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = abl.DEVICE
    fee, slip = abl.omega._load_fee_slip()

    log("=== stage=load_windows ===")
    windows = dict(gate.load_all_windows())
    for cand in RANGING_CANDIDATES:
        windows[cand["key"]] = load_custom_window(cand)

    log("=== stage=detector_build ===")
    score_by_base, _robustness, threshold = guard.build_detector()

    # FRESH seed batch, independent of the N=5/N=30 batches used earlier (different entropy value).
    seed_sequence = np.random.SeedSequence(202608172)
    seeds = [int(s) for s in seed_sequence.generate_state(N_SEEDS)]
    log(f"N_SEEDS={N_SEEDS} fresh independently-spawned seeds: {seeds}")

    all_keys = list(DOWNTREND_WINDOW_KEYS) + [c["key"] for c in RANGING_CANDIDATES]
    results = [run_window(wkey, windows, score_by_base, threshold, device, fee, slip, seeds) for wkey in all_keys]

    log("\n\n=== FINAL: veto engagement vs N=30 significance ===")
    rows = []
    for r in results:
        rows.append({
            "window": r["window"], "n30_spread_pp": r["n30_spread_pp"], "n30_t_stat": r["n30_t_stat"],
            "detector_active_frac": round(r["detector_active_frac"], 4),
            "real_g0_veto_bars": r["real_g0_veto_bars"],
            "random_veto_bars_mean": round(r["random_veto_bars_mean"], 1),
            "random_veto_bars_min": r["random_veto_bars_min"], "random_veto_bars_max": r["random_veto_bars_max"],
            "veto_bars_vs_pnl_corr": round(r["veto_bars_vs_pnl_corr_within_window"], 3) if not np.isnan(r["veto_bars_vs_pnl_corr_within_window"]) else None,
        })
    df = pd.DataFrame(rows).sort_values("n30_spread_pp")
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / "veto_engagement_summary.csv", index=False)

    detail_rows = []
    for r in results:
        for rr in r["random_details"]:
            detail_rows.append({"window": r["window"], **rr})
    pd.DataFrame(detail_rows).to_csv(OUT_DIR / "per_seed_veto_detail.csv", index=False)

    log(f"\nwrote {OUT_DIR / 'veto_engagement_summary.csv'} and {OUT_DIR / 'per_seed_veto_detail.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
