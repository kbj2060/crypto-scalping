#!/usr/bin/env python3
"""RESEARCH ONLY -- follow-up to research_eth_odyssey4_random_direction_risk_management_ablation_
20260817.py (imported here, not duplicated). User question: "so this model's risk management IS
decent then? show the exit-reason distribution (SL/TP vs exit_head etc.) in that test."

Adds two things the original ablation script didn't capture:
  1. A `real_g0` arm -- the UNMODIFIED live-matching pipeline (guard.prepare_regime_aware_components
     + veto_mod.greedy_replay_entry_veto, no direction override at all) as the actual reference for
     "this model's" own exit-reason mix, not just its aggregate PnL/MDD from the contract table.
  2. Per-arm exit-reason counts (take_profit / stop_loss / exit_head) from
     greedy_replay_entry_veto's own `diag["reason_counts"]`, split into duration-gate KEPT vs
     SKIPPED (L4.5 OU-halflife gate) trades to match the with_gate view used throughout this line.

Reuses (imports, never edits) research_eth_odyssey4_random_direction_risk_management_ablation_
20260817 for build_ablation_components / side selectors / seeds / G0 reference table.
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
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402
import research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 as veto_mod  # noqa: E402

OUT_DIR = abl.OUT_DIR


def log(msg: str) -> None:
    print(msg, flush=True)


def _reason_breakdown(ledger: pd.DataFrame, frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Per-reason trade counts, split by the L4.5 duration gate (kept = counted in with_gate PnL,
    skipped = that trade's return is zeroed in the with_gate view -- see mfe_width._duration_gated).
    """
    if len(ledger) == 0:
        return {"n_trades": 0, "kept": {}, "skipped": {}}
    active = ledger.copy()
    active["entry_timestamp_dt"] = pd.to_datetime(active["entry_timestamp"])
    market = frame[["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp_dt"})
    active = active.merge(market, on="entry_timestamp_dt", how="left")
    hit = active["ou_halflife"] <= threshold
    kept_counts = active.loc[~hit, "reason"].value_counts().to_dict()
    skipped_counts = active.loc[hit, "reason"].value_counts().to_dict()
    return {"n_trades": int(len(active)), "kept": kept_counts, "skipped": skipped_counts}


def run_real_g0_arm(window_name: str, windows: dict, score_by_base: dict, threshold: float, out_dir: Path,
                     device, fee: float, slip: float) -> dict[str, Any]:
    """The actual unmodified pipeline (no direction override) -- reproduces the live G0 numbers and,
    unlike the contract table, also yields this run's own ledger for the reason breakdown."""
    aligned_frame, components, _prep_diag = guard.prepare_regime_aware_components(
        window_name, windows, score_by_base, threshold, out_dir, device,
    )
    mask, _n_nan = guard._detector_mask_for_frame(aligned_frame, window_name, score_by_base, threshold)
    components["zig075"]["short_entry_veto_mask"] = mask
    diag, ledger = veto_mod.greedy_replay_entry_veto(
        aligned_frame, components, fee=fee, slip=slip, cost_mult=abl.sweep.COST_MULT, device=device,
    )
    with_gate = mfe_width._duration_gated(ledger, aligned_frame, abl.greedy.DURATION_THRESHOLD)
    reasons = _reason_breakdown(ledger, aligned_frame, abl.greedy.DURATION_THRESHOLD)
    return {"arm": "real_g0", "window": window_name, "with_gate": with_gate,
            "reason_counts_raw": diag.get("reason_counts"), "reason_breakdown": reasons}


def run_arm_with_reasons(arm_label: str, window_name: str, windows: dict, score_by_base: dict,
                          threshold: float, out_dir: Path, device, fee: float, slip: float, *,
                          side_selector) -> dict[str, Any]:
    aligned_frame, components = abl.build_ablation_components(
        window_name, windows, score_by_base, threshold, out_dir, device, side_selector=side_selector,
    )
    diag, ledger = veto_mod.greedy_replay_entry_veto(
        aligned_frame, components, fee=fee, slip=slip, cost_mult=abl.sweep.COST_MULT, device=device,
    )
    with_gate = mfe_width._duration_gated(ledger, aligned_frame, abl.greedy.DURATION_THRESHOLD)
    reasons = _reason_breakdown(ledger, aligned_frame, abl.greedy.DURATION_THRESHOLD)
    return {"arm": arm_label, "window": window_name, "with_gate": with_gate,
            "reason_counts_raw": diag.get("reason_counts"), "reason_breakdown": reasons}


def main() -> int:
    device = abl.DEVICE
    fee, slip = abl.omega._load_fee_slip()

    log("=== stage=load_windows ===")
    windows = abl.gate.load_all_windows()
    score_by_base, _robustness, threshold = guard.build_detector()

    seed_sequence = np.random.SeedSequence(20260817)
    seeds = [int(s) for s in seed_sequence.generate_state(abl.N_SEEDS)]

    results: list[dict[str, Any]] = []
    for window_name in abl.JUDGED_WINDOWS:
        log(f"=== window={window_name} ===")
        log("  arm=real_g0")
        results.append(run_real_g0_arm(window_name, windows, score_by_base, threshold, OUT_DIR, device, fee, slip))
        log("  arm=always_long")
        results.append(run_arm_with_reasons(
            "always_long", window_name, windows, score_by_base, threshold, OUT_DIR, device, fee, slip,
            side_selector=lambda n: abl._side_selector_constant(n, 1),
        ))
        log("  arm=always_short")
        results.append(run_arm_with_reasons(
            "always_short", window_name, windows, score_by_base, threshold, OUT_DIR, device, fee, slip,
            side_selector=lambda n: abl._side_selector_constant(n, -1),
        ))
        for seed in seeds:
            log(f"  arm=random seed={seed}")
            results.append(run_arm_with_reasons(
                f"random_seed{seed}", window_name, windows, score_by_base, threshold, OUT_DIR, device, fee, slip,
                side_selector=lambda n, _seed=seed: abl._side_selector_random(n, _seed),
            ))

    log("\n=== exit-reason distribution (with_gate / KEPT trades only) ===")
    rows = []
    for r in results:
        kept = r["reason_breakdown"]["kept"]
        total = sum(kept.values())
        row = {"arm": r["arm"], "window": r["window"], "n_kept_trades": total}
        for reason in ("take_profit", "stop_loss", "exit_head", "trailing_stop"):
            n = kept.get(reason, 0)
            row[f"{reason}_n"] = n
            row[f"{reason}_pct"] = round(100.0 * n / total, 1) if total else 0.0
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "exit_reason_distribution.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    log(f"\nwrote {OUT_DIR / 'exit_reason_distribution.csv'}")

    log("\n=== aggregated by arm-type (real_g0 / always_long / always_short / random-pooled) across all 3 windows ===")
    df["arm_type"] = df["arm"].apply(lambda a: "random" if a.startswith("random_seed") else a)
    agg = df.groupby("arm_type")[["n_kept_trades", "take_profit_n", "stop_loss_n", "exit_head_n"]].sum()
    agg["take_profit_pct"] = (100 * agg["take_profit_n"] / agg["n_kept_trades"]).round(1)
    agg["stop_loss_pct"] = (100 * agg["stop_loss_n"] / agg["n_kept_trades"]).round(1)
    agg["exit_head_pct"] = (100 * agg["exit_head_n"] / agg["n_kept_trades"]).round(1)
    print(agg.to_string())
    agg.to_csv(OUT_DIR / "exit_reason_distribution_by_arm_type.csv")
    log(f"wrote {OUT_DIR / 'exit_reason_distribution_by_arm_type.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
