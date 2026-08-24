#!/usr/bin/env python3
"""RESEARCH ONLY -- tests whether h48qual's regime-aware EXIT guard (Odyssey3, which Ilias's
contract currently plans to inherit UNCHANGED into its L9 layer -- docs/model_contracts/
ilias_eth_human_direction_risk_management_contract_20260817.md Layer Contracts table) has the
SAME ranging-regime misfire found in zig075's SHORT ENTRY veto
([[eth_zig075_short_veto_paired_onoff_causal_test_20260817]]), and whether the same v3 fix
(research_eth_odyssey4_zig075_veto_v3_mean_confirm_fix_20260817.py's breakout-confirmed mask,
`close_t >= rolling_mean(close, WEEK_BARS)_t` AND-combined with the unmodified momentum
condition -- zero new tunable parameters, imported not reimplemented) helps there too.

=== Why this is a DIFFERENT mechanism from the zig075 veto test, not a re-run of it ===
The h48qual exit guard does NOT block entries. While h48qual holds an OPEN position (either
side), it switches which of h48qual's own two already-trained exit-head decision paths governs
the exit check that bar: mask ACTIVE -> h48qual's ORIGINAL frozen exit head (slow, static
exit_threshold=0.95); mask INACTIVE -> the current live-ATR-relabeled exit head (fast turnover,
the shadow-deployed default). It was designed to prevent the fast/liveATR exit head from
over-churning h48qual's SHORT positions during 2025-Q3's genuine sustained uptrend (Q3 no_gate PnL
-9.73% -> -46.26% when liveATR ran unguarded throughout). Hypothesis under test here: if the SAME
underlying detector misfires in genuine chop (as already found for the zig075 veto), the guard
would inappropriately fall back to the SLOW/original exit policy during that chop, holding
positions too long when the fast liveATR policy (designed for choppy/reversal-prone conditions)
would have been the better default -- a symmetric failure mode to the entry-veto's.

Isolation: zig075's own SHORT entry veto is REMOVED in every run here (no short_entry_veto_mask
attached) so any PnL difference is attributable ONLY to the h48qual exit-guard switch, not
conflated with the already-separately-tested zig075 veto. This exactly matches how the guard
module's own original "G0a" reference isolated Odyssey3's h48qual guard contribution before
Odyssey4 layered the zig075 veto on top (docs/model_contracts/odyssey3_eth_regime_guard_baseline_
contract_20260814.md).

Three arms per draw: NONE (h48qual guard never engages, pure liveATR throughout -- the original
"asymmetric_tabm_liveatr" baseline before Odyssey3's guard existed), V1 (original momentum-only
detector, Odyssey3's locked baseline), V3 (momentum AND close>=rolling-week-mean, same fix
candidate already validated for the zig075 veto). Includes a Q3 check (the guard's OWN original
raison d'etre, separate from zig075's) alongside the ranging windows (the misfire hypothesis
target) and VAL/OOS-Q1/OOS-Q2 (judged-tier consistency).

Reuses (imports, never duplicates) every helper in this line, including the v3 mask builder
itself (research_eth_odyssey4_zig075_veto_v3_mean_confirm_fix_20260817._rolling_close_high_
confirmed / build_v2_mask_for_frame -- not reimplemented).

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
import research_eth_odyssey4_zig075_veto_v3_mean_confirm_fix_20260817 as v3fix  # noqa: E402
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402
import research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 as veto_mod  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_h48qual_exit_guard_ranging_misfire_test_20260817"
N_SEEDS = 20

RANGING_CANDIDATES = v3fix.RANGING_CANDIDATES  # identical 3 windows, reused not redefined
DOWNTREND_WINDOW_KEYS = v3fix.DOWNTREND_WINDOW_KEYS
CHECK_WINDOW_KEYS = list(DOWNTREND_WINDOW_KEYS) + [c["key"] for c in RANGING_CANDIDATES]


def log(msg: str) -> None:
    print(msg, flush=True)


def paired_guard_v1_v3(aligned_frame: pd.DataFrame, components: dict, mask_v1: np.ndarray, mask_v3: np.ndarray,
                        *, fee: float, slip: float, device) -> tuple[dict, dict, dict]:
    """Isolates the h48qual exit-guard's own effect: zig075's entry veto is always OFF here.
    Three passes on the SAME components: guard mask absent (NONE), v1 (original), v3 (fix)."""
    components["zig075"].pop("short_entry_veto_mask", None)  # isolation: no zig075 veto in this test

    components["h48qual"].pop("sustained_uptrend_mask", None)
    diag_none, ledger_none = veto_mod.greedy_replay_entry_veto(
        aligned_frame, components, fee=fee, slip=slip, cost_mult=abl.sweep.COST_MULT, device=device)
    wg_none = mfe_width._duration_gated(ledger_none, aligned_frame, abl.greedy.DURATION_THRESHOLD)

    components["h48qual"]["sustained_uptrend_mask"] = mask_v1
    diag_v1, ledger_v1 = veto_mod.greedy_replay_entry_veto(
        aligned_frame, components, fee=fee, slip=slip, cost_mult=abl.sweep.COST_MULT, device=device)
    wg_v1 = mfe_width._duration_gated(ledger_v1, aligned_frame, abl.greedy.DURATION_THRESHOLD)
    guard_active_v1 = diag_v1.get("h48qual_guard_active_bars", 0)

    components["h48qual"]["sustained_uptrend_mask"] = mask_v3
    diag_v3, ledger_v3 = veto_mod.greedy_replay_entry_veto(
        aligned_frame, components, fee=fee, slip=slip, cost_mult=abl.sweep.COST_MULT, device=device)
    wg_v3 = mfe_width._duration_gated(ledger_v3, aligned_frame, abl.greedy.DURATION_THRESHOLD)
    guard_active_v3 = diag_v3.get("h48qual_guard_active_bars", 0)

    return (
        {"pnl": wg_none["pnl"], "mdd": wg_none["mdd"], "trades": wg_none["trades"]},
        {"pnl": wg_v1["pnl"], "mdd": wg_v1["mdd"], "trades": wg_v1["trades"], "guard_active_bars": guard_active_v1},
        {"pnl": wg_v3["pnl"], "mdd": wg_v3["mdd"], "trades": wg_v3["trades"], "guard_active_bars": guard_active_v3},
    )


def run_window(window_key: str, windows: dict, score_by_base: dict, breakout_by_base: dict, threshold: float,
               device, fee: float, slip: float, seeds: list[int]) -> dict[str, Any]:
    log(f"=== window={window_key} ===")

    aligned_frame, real_components, _prep_diag = guard.prepare_regime_aware_components(
        window_key, windows, score_by_base, threshold, OUT_DIR, device)
    mask_v1, mask_v3 = v3fix.build_v2_mask_for_frame(aligned_frame, window_key, score_by_base, breakout_by_base, threshold)

    none_r, v1_r, v3_r = paired_guard_v1_v3(aligned_frame, real_components, mask_v1, mask_v3, fee=fee, slip=slip, device=device)
    log(f"  real_g0: NONE(no guard)={none_r['pnl']:+.2f}%  V1(orig guard)={v1_r['pnl']:+.2f}%(active_bars={v1_r['guard_active_bars']})  "
        f"V3(fix)={v3_r['pnl']:+.2f}%(active_bars={v3_r['guard_active_bars']})")

    seed_rows = []
    for i, seed in enumerate(seeds):
        aligned_frame_r, components_r = abl.build_ablation_components(
            window_key, windows, score_by_base, threshold, OUT_DIR, device,
            side_selector=lambda n, _seed=seed: abl._side_selector_random(n, _seed))
        none_s, v1_s, v3_s = paired_guard_v1_v3(aligned_frame_r, components_r, mask_v1, mask_v3, fee=fee, slip=slip, device=device)
        seed_rows.append({"seed": seed, "none_pnl": none_s["pnl"], "v1_pnl": v1_s["pnl"], "v3_pnl": v3_s["pnl"],
                           "v1_delta": v1_s["pnl"] - none_s["pnl"], "v3_delta": v3_s["pnl"] - none_s["pnl"]})
        if (i + 1) % 5 == 0:
            log(f"  ...random seeds done: {i + 1}/{len(seeds)}")

    sdf = pd.DataFrame(seed_rows)
    v1_delta = sdf["v1_delta"].to_numpy()
    v3_delta = sdf["v3_delta"].to_numpy()

    def _t(arr):
        if arr.std(ddof=1) == 0:
            return float("nan")
        return float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))))

    log(f"  random v1(orig) delta: mean={v1_delta.mean():+.2f}pp std={v1_delta.std(ddof=1):.2f}pp t={_t(v1_delta):+.2f}")
    log(f"  random v3(fix)  delta: mean={v3_delta.mean():+.2f}pp std={v3_delta.std(ddof=1):.2f}pp t={_t(v3_delta):+.2f}")

    return {
        "window": window_key,
        "real_g0_none_pnl": none_r["pnl"], "real_g0_v1_pnl": v1_r["pnl"], "real_g0_v3_pnl": v3_r["pnl"],
        "real_g0_v1_guard_active_bars": v1_r["guard_active_bars"], "real_g0_v3_guard_active_bars": v3_r["guard_active_bars"],
        "random_v1_mean_delta": float(v1_delta.mean()), "random_v1_std_delta": float(v1_delta.std(ddof=1)), "random_v1_t": _t(v1_delta),
        "random_v3_mean_delta": float(v3_delta.mean()), "random_v3_std_delta": float(v3_delta.std(ddof=1)), "random_v3_t": _t(v3_delta),
        "seed_rows": seed_rows,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = abl.DEVICE
    fee, slip = abl.omega._load_fee_slip()

    log("=== stage=load_windows ===")
    windows = dict(gate.load_all_windows())
    for cand in RANGING_CANDIDATES:
        windows[cand["key"]] = v3fix.load_custom_window(cand)

    log("=== stage=detector_build (v1 unmodified momentum-only; v3 = momentum AND close>=rolling_week_mean, reused from the zig075 fix) ===")
    score_by_base, _robustness, threshold = guard.build_detector()
    breakout_by_base = {sweep.BASE_2025: v3fix._rolling_close_high_confirmed(sweep.BASE_2025),
                         sweep.BASE_2026: v3fix._rolling_close_high_confirmed(sweep.BASE_2026)}

    seed_sequence = np.random.SeedSequence(2026081799)  # SAME seeds as the zig075 v3 fix test
    seeds = [int(s) for s in seed_sequence.generate_state(N_SEEDS)]
    log(f"N_SEEDS={N_SEEDS} seeds (identical to the zig075 veto v1/v3 test): {seeds}")

    log("\n=== stage=q3_own_raison_detre_check (the h48qual guard's OWN original benefit, separate from zig075's) ===")
    q3_frame, q3_components, _ = guard.prepare_regime_aware_components("2025q3", windows, score_by_base, threshold, OUT_DIR, device)
    q3_mask_v1, q3_mask_v3 = v3fix.build_v2_mask_for_frame(q3_frame, "2025q3", score_by_base, breakout_by_base, threshold)
    q3_none, q3_v1, q3_v3 = paired_guard_v1_v3(q3_frame, q3_components, q3_mask_v1, q3_mask_v3, fee=fee, slip=slip, device=device)
    log(f"  Q3 real_g0 (h48qual guard, zig075 veto OFF throughout): NONE(no guard)={q3_none['pnl']:+.2f}%  "
        f"V1(orig)={q3_v1['pnl']:+.2f}%(active_bars={q3_v1['guard_active_bars']})  V3(fix)={q3_v3['pnl']:+.2f}%(active_bars={q3_v3['guard_active_bars']})")

    results = [run_window(wkey, windows, score_by_base, breakout_by_base, threshold, device, fee, slip, seeds) for wkey in CHECK_WINDOW_KEYS]

    log("\n\n=== FINAL: h48qual exit-guard v1(original) vs v3(fix), zig075 veto isolated OFF ===")
    rows = []
    for r in results:
        rows.append({
            "window": r["window"],
            "real_g0_none_pnl": round(r["real_g0_none_pnl"], 2), "real_g0_v1_pnl": round(r["real_g0_v1_pnl"], 2), "real_g0_v3_pnl": round(r["real_g0_v3_pnl"], 2),
            "real_g0_v1_guard_active_bars": r["real_g0_v1_guard_active_bars"], "real_g0_v3_guard_active_bars": r["real_g0_v3_guard_active_bars"],
            "random_v1_mean_delta": round(r["random_v1_mean_delta"], 2), "random_v1_t": round(r["random_v1_t"], 2),
            "random_v3_mean_delta": round(r["random_v3_mean_delta"], 2), "random_v3_t": round(r["random_v3_t"], 2),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / "h48qual_guard_v1_vs_v3_summary.csv", index=False)

    q3_row = {"window": "2025q3 (context, guard's own raison d'etre)",
              "real_g0_none_pnl": round(q3_none["pnl"], 2), "real_g0_v1_pnl": round(q3_v1["pnl"], 2), "real_g0_v3_pnl": round(q3_v3["pnl"], 2),
              "real_g0_v1_guard_active_bars": q3_v1["guard_active_bars"], "real_g0_v3_guard_active_bars": q3_v3["guard_active_bars"]}
    pd.DataFrame([q3_row]).to_csv(OUT_DIR / "q3_context_check.csv", index=False)
    print("\nQ3 context check:", q3_row)

    detail_rows = []
    for r in results:
        for sr in r["seed_rows"]:
            detail_rows.append({"window": r["window"], **sr})
    pd.DataFrame(detail_rows).to_csv(OUT_DIR / "per_seed_detail.csv", index=False)

    log(f"\nwrote outputs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
