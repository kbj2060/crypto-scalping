#!/usr/bin/env python3
"""RESEARCH ONLY -- diagnostic (not a promotion candidate). Decomposes the gap found in
docs/experiments/eth_odyssey4_exit_head_tpsl_feature_barrier_mismatch_20260817.md: h48qual's
exit_head fires in 82.5% of trades (52/63) when evaluated ALONE
(research_eth_omega461_exit_head_h48cons_relabel_20260813._evaluate_val, same VAL window, same
currently-deployed NEW_H48QUAL_BUNDLE) but 0% when replayed inside the FULL PORTFOLIO
(h48qual+zig075 single-slot-shared, L4.5 duration gate applied -- the floor-shrink sweep's own
methodology). Two candidate causes, tested independently via a 2x2 design (both use the SAME
replay function, replay_omega4_6_1_greedy_router_20260706.greedy_replay, so the replay-function
mechanics are held constant -- only presence of zig075 and application of the L4.5 duration gate
vary):

  Arm A: h48qual ALONE,        no duration gate  -- closest analogue to the component-only eval;
         if this reproduces ~82.5%, the two replay FUNCTIONS are behaviourally equivalent and the
         gap is caused entirely by (zig075 presence) and/or (duration gate), tested next.
  Arm B: h48qual ALONE,        WITH duration gate -- isolates the L4.5 gate's own effect.
  Arm C: h48qual + zig075,     no duration gate  -- isolates zig075 slot-contention's own effect
         (h48qual's own trades only, filtered from the shared ledger by source_component).
  Arm D: h48qual + zig075,     WITH duration gate -- reproduces the floor-shrink sweep's original
         0% finding, as a sanity-check anchor.

All four arms use the SAME VAL window, SAME currently-deployed (still-buggy) NEW_H48QUAL_BUNDLE,
SAME current TP/SL floor (min_tp=0.075/min_sl=0.040) -- nothing about the model or floor is
touched here, only replay STRUCTURE (component set, gate on/off).

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_
used=false, future_rows_used_for_entry=false. No live/shadow files touched.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
import research_eth_odyssey4_random_direction_exit_reason_distribution_20260817 as reasons_mod  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_exit_head_component_vs_portfolio_gap_diagnosis_20260817"
WINDOW_KEY = "val"


def log(msg: str) -> None:
    print(msg, flush=True)


def _h48qual_reason_counts_raw(ledger: pd.DataFrame) -> dict[str, int]:
    h = ledger[ledger["source_component"] == "h48qual"] if "source_component" in ledger.columns else ledger
    return h["reason"].value_counts().to_dict() if len(h) else {}


def _h48qual_reason_counts_kept(ledger: pd.DataFrame, frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    h = ledger[ledger["source_component"] == "h48qual"] if "source_component" in ledger.columns else ledger
    return reasons_mod._reason_breakdown(h.reset_index(drop=True), frame, threshold)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = portfolio.DEVICE
    fee, slip = omega._load_fee_slip()

    log(f"=== stage=load_window window={WINDOW_KEY} ===")
    windows = dict(gate.load_all_windows())
    w = windows[WINDOW_KEY]
    split = gate.WINDOW_DEFS[WINDOW_KEY]["split"]
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in ("h48qual", "zig075")}
    aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, OUT_DIR)
    prep = portfolio._prepare_component_val if w["oof"] else greedy.prepare_component

    h48qual_cfg = portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE)
    zig075_cfg = portfolio._component_cfg("zig075")
    h48qual_comp = prep(aligned_frame, aligned_paths["h48qual"], h48qual_cfg, device)
    zig075_comp = prep(aligned_frame, aligned_paths["zig075"], zig075_cfg, device)

    def run(label: str, components: dict) -> dict:
        _diag, ledger = greedy.greedy_replay(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        raw_counts = _h48qual_reason_counts_raw(ledger)
        kept_bd = _h48qual_reason_counts_kept(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        n_h48qual_raw = int(sum(raw_counts.values()))
        n_h48qual_kept = int(sum(kept_bd["kept"].values())) if kept_bd["kept"] else 0
        exit_head_share_raw = round(raw_counts.get("exit_head", 0) / n_h48qual_raw, 4) if n_h48qual_raw else None
        exit_head_share_kept = round(kept_bd["kept"].get("exit_head", 0) / n_h48qual_kept, 4) if n_h48qual_kept else None
        log(f"  [{label}] h48qual trades: raw={n_h48qual_raw} (reasons={raw_counts})  "
            f"kept(post-gate)={n_h48qual_kept} (reasons={kept_bd['kept']})  "
            f"exit_head_share: raw={exit_head_share_raw} kept={exit_head_share_kept}")
        return {"label": label, "n_raw": n_h48qual_raw, "reasons_raw": raw_counts,
                "n_kept": n_h48qual_kept, "reasons_kept": kept_bd["kept"],
                "exit_head_share_raw": exit_head_share_raw, "exit_head_share_kept": exit_head_share_kept}

    log("\n=== Arm A: h48qual ALONE, no gate (raw) ===")
    arm_a = run("A_alone_raw", {"h48qual": h48qual_comp})
    log("\n=== Arm B: h48qual ALONE, WITH gate (kept) === (same replay as A, gate applied in post-processing)")
    # Arm B reuses Arm A's ledger via the same run() call -- kept/raw both computed together above.

    log("\n=== Arm C/D: h48qual + zig075 (full portfolio) ===")
    arm_cd = run("CD_portfolio", {"h48qual": h48qual_comp, "zig075": zig075_comp})

    log("\n\n=== SUMMARY ===")
    summary = {
        "arm_A_alone_no_gate": {"n": arm_a["n_raw"], "reasons": arm_a["reasons_raw"], "exit_head_share": arm_a["exit_head_share_raw"]},
        "arm_B_alone_with_gate": {"n": arm_a["n_kept"], "reasons": arm_a["reasons_kept"], "exit_head_share": arm_a["exit_head_share_kept"]},
        "arm_C_portfolio_no_gate": {"n": arm_cd["n_raw"], "reasons": arm_cd["reasons_raw"], "exit_head_share": arm_cd["exit_head_share_raw"]},
        "arm_D_portfolio_with_gate": {"n": arm_cd["n_kept"], "reasons": arm_cd["reasons_kept"], "exit_head_share": arm_cd["exit_head_share_kept"]},
        "reference_component_only_eval": {"n": 63, "reasons": {"exit_head": 52}, "exit_head_share": round(52 / 63, 4)},
    }
    for k, v in summary.items():
        log(f"  {k}: n={v['n']} exit_head_share={v['exit_head_share']} reasons={v['reasons']}")

    import json
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"\nwrote {OUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
