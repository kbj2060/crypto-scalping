#!/usr/bin/env python3
"""RESEARCH ONLY -- large-N re-verification of every window tested so far in the random-direction
risk-management ablation line (research_eth_odyssey4_random_direction_risk_management_ablation_
20260817.py + both ranging retests). All prior real_g0/random-vs-model gap estimates used N=5
seeds; the ranging retests in particular gave a MIXED result across 3 windows (2/3 showed real_g0
worse than random, 1/3 showed the opposite) that N=5 cannot distinguish from noise. This script
reruns ALL 6 windows tested so far (val, oos_q1, oos_q2, and the 3 ranging candidates) with
N_SEEDS=30 (6x the original N=5) to tighten the standard error on each window's real_g0-vs-random
gap and give the regime-dependence question a fair statistical hearing.

Reuses (imports, never duplicates) every helper already built in this line:
research_eth_odyssey4_random_direction_risk_management_ablation_20260817 (build_ablation_components,
side selectors, DEVICE/fee/slip, G0 reference), research_eth_omega461_regime_aware_exit_head_
uptrend_guard_20260814 (build_detector, prepare_regime_aware_components for the real_g0 arm),
research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 (greedy_replay_entry_veto),
eth_omega461_multiwindow_confirmation_gate_20260814 (load_all_windows, WINDOW_DEFS),
research_eth_omega461_exit_sweep_20260721 (load_frame, BASE_2025/2026, WIDE24_2025/2026).
Only new code: the 3-ranging-window CANDIDATES table (same definitions used in the two ranging
retest scripts, consolidated here) and the run/aggregate loop.

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
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_random_direction_large_n_reverification_20260817"
N_SEEDS_LARGE = 30

# The 3 ranging-type windows tested in the two prior retests, consolidated in one place.
RANGING_CANDIDATES = [
    {"key": "ranging_2025_05_12_to_07_07", "start": "2025-05-12", "end": "2025-07-07",
     "oof": True, "split": "train", "base_csv": sweep.BASE_2025, "wide24_csv": sweep.WIDE24_2025},
    {"key": "ranging_2025_03_10_to_05_05", "start": "2025-03-10", "end": "2025-05-05",
     "oof": True, "split": "train", "base_csv": sweep.BASE_2025, "wide24_csv": sweep.WIDE24_2025},
    {"key": "ranging_2026_02_09_to_04_06", "start": "2026-02-09", "end": "2026-04-06",
     "oof": False, "split": "oos", "base_csv": sweep.BASE_2026, "wide24_csv": sweep.WIDE24_2026},
]
DOWNTREND_WINDOW_KEYS = ("val", "oos_q1", "oos_q2")


def log(msg: str) -> None:
    print(msg, flush=True)


def load_custom_window(cand: dict) -> dict[str, Any]:
    frame = sweep.load_frame(cand["start"], cand["end"], base_csv=cand["base_csv"], wide24_csv=cand["wide24_csv"])
    frame, n_dropped = gate._drop_route_nan(frame)
    gate.WINDOW_DEFS[cand["key"]] = {"split": cand["split"], "base_csv": cand["base_csv"], "wide24_csv": cand["wide24_csv"]}
    return {"frame": frame, "oof": cand["oof"], "tier": "ranging_retest", "route_nan_dropped": n_dropped}


def run_window(window_key: str, windows: dict, score_by_base: dict, threshold: float, device, fee: float, slip: float,
               seeds: list[int]) -> dict[str, Any]:
    log(f"=== window={window_key} ===")

    aligned_frame, real_components, _prep_diag = guard.prepare_regime_aware_components(
        window_key, windows, score_by_base, threshold, OUT_DIR, device,
    )
    mask, _ = guard._detector_mask_for_frame(aligned_frame, window_key, score_by_base, threshold)
    real_components["zig075"]["short_entry_veto_mask"] = mask
    _diag, real_ledger = veto_mod.greedy_replay_entry_veto(
        aligned_frame, real_components, fee=fee, slip=slip, cost_mult=abl.sweep.COST_MULT, device=device,
    )
    import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width
    real_g0 = mfe_width._duration_gated(real_ledger, aligned_frame, abl.greedy.DURATION_THRESHOLD)
    log(f"  real_g0: pnl={real_g0['pnl']:+.2f}% mdd={real_g0['mdd']:.2f}% trades={real_g0['trades']}")

    always_long = abl.run_arm("always_long", window_key, windows, score_by_base, threshold, OUT_DIR, device, fee, slip,
                               side_selector=lambda n: abl._side_selector_constant(n, 1))["with_gate"]
    always_short = abl.run_arm("always_short", window_key, windows, score_by_base, threshold, OUT_DIR, device, fee, slip,
                                side_selector=lambda n: abl._side_selector_constant(n, -1))["with_gate"]
    log(f"  always_long: pnl={always_long['pnl']:+.2f}%  always_short: pnl={always_short['pnl']:+.2f}%  "
        f"spread={always_short['pnl']-always_long['pnl']:+.2f}pp")

    random_pnls: list[float] = []
    random_mdds: list[float] = []
    for i, seed in enumerate(seeds):
        r = abl.run_arm(f"random_seed{seed}", window_key, windows, score_by_base, threshold, OUT_DIR, device, fee, slip,
                         side_selector=lambda n, _seed=seed: abl._side_selector_random(n, _seed))["with_gate"]
        random_pnls.append(r["pnl"])
        random_mdds.append(r["mdd"])
        if (i + 1) % 10 == 0:
            log(f"  ...random seeds done: {i + 1}/{len(seeds)}")

    random_pnls_arr = np.array(random_pnls)
    mean, std = float(random_pnls_arr.mean()), float(random_pnls_arr.std(ddof=1))
    se = std / np.sqrt(len(random_pnls_arr))
    gap = real_g0["pnl"] - mean
    t_stat = gap / se if se > 0 else float("nan")

    return {
        "window": window_key,
        "real_g0_pnl": real_g0["pnl"], "real_g0_mdd": real_g0["mdd"], "real_g0_trades": real_g0["trades"],
        "always_long_pnl": always_long["pnl"], "always_short_pnl": always_short["pnl"],
        "spread_pp": always_short["pnl"] - always_long["pnl"],
        "n_seeds": len(seeds), "random_mean": mean, "random_std": std, "random_se": se,
        "gap_pp": gap, "t_stat": t_stat,
        "random_pnls": random_pnls, "random_mdds": random_mdds,
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

    seed_sequence = np.random.SeedSequence(20260817)
    seeds = [int(s) for s in seed_sequence.generate_state(N_SEEDS_LARGE)]
    log(f"N_SEEDS_LARGE={N_SEEDS_LARGE} independently-spawned seeds: {seeds}")

    all_keys = list(DOWNTREND_WINDOW_KEYS) + [c["key"] for c in RANGING_CANDIDATES]
    results = []
    for wkey in all_keys:
        results.append(run_window(wkey, windows, score_by_base, threshold, device, fee, slip, seeds))

    log("\n\n=== FINAL SUMMARY (N=%d seeds per window) ===" % N_SEEDS_LARGE)
    summary_rows = []
    for r in results:
        summary_rows.append({
            "window": r["window"], "spread_pp": round(r["spread_pp"], 2),
            "real_g0_pnl": round(r["real_g0_pnl"], 2),
            "random_mean": round(r["random_mean"], 2), "random_std": round(r["random_std"], 2),
            "gap_pp": round(r["gap_pp"], 2), "t_stat": round(r["t_stat"], 2),
        })
    sdf = pd.DataFrame(summary_rows).sort_values("spread_pp")
    print(sdf.to_string(index=False))
    sdf.to_csv(OUT_DIR / "final_summary.csv", index=False)

    # full per-seed pnl for downstream re-analysis
    long_rows = []
    for r in results:
        for seed, pnl, mdd in zip(seeds, r["random_pnls"], r["random_mdds"]):
            long_rows.append({"window": r["window"], "seed": seed, "pnl": pnl, "mdd": mdd})
    pd.DataFrame(long_rows).to_csv(OUT_DIR / "per_seed_pnl.csv", index=False)

    log(f"\nwrote {OUT_DIR / 'final_summary.csv'} and {OUT_DIR / 'per_seed_pnl.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
