#!/usr/bin/env python3
"""RESEARCH ONLY -- clean, paired with/without-veto comparison, fixing the confound in the
earlier correlational veto-engagement analysis (research_eth_odyssey4_random_direction_veto_
engagement_hypothesis_20260817.py). That analysis correlated a random seed's veto_bars count with
its own PnL ACROSS seeds -- confounded, because a seed that happens to draw more SHORT signals
overall will mechanically generate both more veto_bars AND (in a downtrend window) better PnL,
with no causal link to the veto itself.

This script instead holds the EXACT SAME direction draw fixed (same seed -> same per-bar
LONG/SHORT/gated-CASH sequence for both h48qual and zig075) and runs the portfolio replay TWICE:
once with zig075's SHORT/sustained-uptrend veto mask attached (WITH), once with it removed
(WITHOUT) -- everything else (entries attempted, TP/SL, sizing, h48qual regime-exit guard,
priority routing) is byte-identical between the two runs. The only expensive step (TabM inference
+ risk-sidecar scoring via build_ablation_components / prepare_regime_aware_components) runs ONCE
per seed; only the cheap bar-by-bar greedy_replay_entry_veto loop runs twice. This isolates the
veto's own causal PnL contribution as a WITHIN-seed paired difference (with - without), letting a
paired t-test (which cancels seed-level noise) replace the earlier confounded cross-seed
correlation.

Also runs this same WITH/WITHOUT pair for real_g0 (the model's own real direction) in all 6
windows -- 3 of these (VAL/OOS-Q1/OOS-Q2) are already known from the locked Odyssey4 G0 contract
table (with_gate PnL identical in all 3 -- OOS-Q1/Q2 trivially since the veto never fired, VAL
non-trivially since 12 bars fired but the final with_gate number didn't move), reproduced here
only as a consistency check; the 3 ranging windows have never had this comparison run before.

Reuses (imports, never duplicates) every helper already built in this line: research_eth_odyssey4_
random_direction_risk_management_ablation_20260817 (build_ablation_components, DEVICE/fee/slip,
side selectors), research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 (build_
detector, _detector_mask_for_frame, prepare_regime_aware_components), research_eth_omega461_
zig075_short_entry_veto_sustained_uptrend_20260814 (greedy_replay_entry_veto -- called twice per
draw, once with the mask key present, once absent), eth_omega461_multiwindow_confirmation_gate_
20260814 (load_all_windows), research_eth_omega461_exit_sweep_20260721 (load_frame/BASE_2025/2026/
WIDE24_2025/2026), research_eth_omega461_live_sltp_mfe_width_20260813 (_duration_gated).

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

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_random_direction_veto_paired_onoff_20260817"
N_SEEDS = 20

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


def paired_with_without(aligned_frame: pd.DataFrame, components: dict, mask: np.ndarray, *, fee: float, slip: float,
                         device) -> tuple[dict, dict]:
    """Runs greedy_replay_entry_veto twice on the SAME components: once with zig075's
    short_entry_veto_mask attached (WITH), once with that key removed (WITHOUT). Only this cheap
    bar-by-bar loop runs twice -- no re-inference."""
    components["zig075"]["short_entry_veto_mask"] = mask
    diag_with, ledger_with = veto_mod.greedy_replay_entry_veto(
        aligned_frame, components, fee=fee, slip=slip, cost_mult=abl.sweep.COST_MULT, device=device,
    )
    with_gate_with = mfe_width._duration_gated(ledger_with, aligned_frame, abl.greedy.DURATION_THRESHOLD)

    components["zig075"].pop("short_entry_veto_mask", None)
    diag_without, ledger_without = veto_mod.greedy_replay_entry_veto(
        aligned_frame, components, fee=fee, slip=slip, cost_mult=abl.sweep.COST_MULT, device=device,
    )
    with_gate_without = mfe_width._duration_gated(ledger_without, aligned_frame, abl.greedy.DURATION_THRESHOLD)

    return (
        {"pnl": with_gate_with["pnl"], "mdd": with_gate_with["mdd"], "trades": with_gate_with["trades"],
         "veto_bars": diag_with.get("veto_bars", 0)},
        {"pnl": with_gate_without["pnl"], "mdd": with_gate_without["mdd"], "trades": with_gate_without["trades"]},
    )


def run_window(window_key: str, windows: dict, score_by_base: dict, threshold: float, device, fee: float, slip: float,
               seeds: list[int]) -> dict[str, Any]:
    log(f"=== window={window_key} ===")

    # real_g0 paired
    aligned_frame, real_components, _prep_diag = guard.prepare_regime_aware_components(
        window_key, windows, score_by_base, threshold, OUT_DIR, device,
    )
    mask, _ = guard._detector_mask_for_frame(aligned_frame, window_key, score_by_base, threshold)
    real_with, real_without = paired_with_without(aligned_frame, real_components, mask, fee=fee, slip=slip, device=device)
    real_delta = real_with["pnl"] - real_without["pnl"]
    log(f"  real_g0: WITH={real_with['pnl']:+.2f}%  WITHOUT={real_without['pnl']:+.2f}%  "
        f"delta={real_delta:+.2f}pp  veto_bars={real_with['veto_bars']}")

    # random seeds, paired
    deltas = []
    veto_bars_list = []
    for i, seed in enumerate(seeds):
        aligned_frame_r, components_r = abl.build_ablation_components(
            window_key, windows, score_by_base, threshold, OUT_DIR, device,
            side_selector=lambda n, _seed=seed: abl._side_selector_random(n, _seed),
        )
        # build_ablation_components already attaches the mask; reuse the same mask array.
        with_r, without_r = paired_with_without(aligned_frame_r, components_r, mask, fee=fee, slip=slip, device=device)
        delta = with_r["pnl"] - without_r["pnl"]
        deltas.append(delta)
        veto_bars_list.append(with_r["veto_bars"])
        if (i + 1) % 5 == 0:
            log(f"  ...random seeds done: {i + 1}/{len(seeds)}")

    deltas_arr = np.array(deltas)
    mean_delta = float(deltas_arr.mean())
    std_delta = float(deltas_arr.std(ddof=1))
    se_delta = std_delta / np.sqrt(len(deltas_arr))
    t_stat = mean_delta / se_delta if se_delta > 0 else float("nan")
    n_positive = int((deltas_arr > 0).sum())
    n_zero = int((deltas_arr == 0).sum())
    n_negative = int((deltas_arr < 0).sum())

    log(f"  random paired delta: mean={mean_delta:+.2f}pp std={std_delta:.2f}pp t={t_stat:+.2f}  "
        f"(+{n_positive}/0:{n_zero}/-{n_negative} of {len(seeds)})")

    return {
        "window": window_key, "real_g0_delta_pp": real_delta, "real_g0_veto_bars": real_with["veto_bars"],
        "random_mean_delta_pp": mean_delta, "random_std_delta_pp": std_delta, "random_t_stat": t_stat,
        "n_positive": n_positive, "n_zero": n_zero, "n_negative": n_negative,
        "deltas": deltas, "veto_bars_list": veto_bars_list,
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

    # fresh seed batch, independent of every prior batch in this line.
    seed_sequence = np.random.SeedSequence(2026081799)
    seeds = [int(s) for s in seed_sequence.generate_state(N_SEEDS)]
    log(f"N_SEEDS={N_SEEDS} fresh independently-spawned seeds: {seeds}")

    all_keys = list(DOWNTREND_WINDOW_KEYS) + [c["key"] for c in RANGING_CANDIDATES]
    results = [run_window(wkey, windows, score_by_base, threshold, device, fee, slip, seeds) for wkey in all_keys]

    log("\n\n=== FINAL: paired WITH-vs-WITHOUT veto PnL delta ===")
    rows = []
    for r in results:
        rows.append({
            "window": r["window"],
            "real_g0_delta_pp": round(r["real_g0_delta_pp"], 3), "real_g0_veto_bars": r["real_g0_veto_bars"],
            "random_mean_delta_pp": round(r["random_mean_delta_pp"], 3),
            "random_std_delta_pp": round(r["random_std_delta_pp"], 3),
            "random_t_stat": round(r["random_t_stat"], 2),
            "n_pos": r["n_positive"], "n_zero": r["n_zero"], "n_neg": r["n_negative"],
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / "paired_onoff_summary.csv", index=False)

    detail_rows = []
    for r in results:
        for seed, delta, vb in zip(seeds, r["deltas"], r["veto_bars_list"]):
            detail_rows.append({"window": r["window"], "seed": seed, "delta_pp": delta, "veto_bars": vb})
    pd.DataFrame(detail_rows).to_csv(OUT_DIR / "per_seed_delta_detail.csv", index=False)

    log(f"\nwrote {OUT_DIR / 'paired_onoff_summary.csv'} and {OUT_DIR / 'per_seed_delta_detail.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
