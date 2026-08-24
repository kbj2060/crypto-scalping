#!/usr/bin/env python3
"""RESEARCH ONLY -- Step B re-verification across all 3 regime experts, using a checkpoint-
selection criterion that isn't corrupted the way the original cheap_gate's was.

Discovery that motivated this (docs/experiments/eth_candidate_faithful_tabm_batchensemble_20260816
.md's "정정" section): the original Step B cheap_gate selected checkpoints via combined val_loss
(direction + 0.8*quality + 1.15*exit) with patience=8 early stopping. For architectures WITHOUT
embeddings (baseline_R_only, full_R_S_B) this criterion tracks the true direction_balanced_accuracy
peak almost perfectly (gap 0.0000-0.0008, confirmed on bull/seed=260816 full curves). But for
architectures WITH the piecewise-linear embedding, the gap is large (+0.038 to +0.053 across 4
capacity configs, same seed/expert) -- the combined-loss-selected checkpoint is NOT the true
direction accuracy peak, because quality/exit loss dynamics diverge from direction loss dynamics
once the embedding is added. This means the original Step B verdict ("3/3 experts decisively
worse") may have been measuring a checkpoint-selection artifact rather than a true architecture
effect -- the `quarter`-capacity config's TRUE peak (0.5717, bull only, tested so far) is nearly
identical to baseline's peak (0.5740), not decisively worse.

This script re-runs baseline_R_only AND full_R_S_B_embed (quarter capacity: hidden=96, d_embed=4,
n_bins=8 -- 109,836 params, fewer than baseline's 118,552) across ALL 3 regime experts
(bull/bear/chop, not just bull), fixed 40-epoch budget, NO early stopping, full curve logged --
so the TRUE direction_balanced_accuracy peak is directly observable and comparable for both
architectures on every expert, not just the one already checked. Single seed=260816 still (same
as every other diagnostic in this candidate) -- N>=5 seed reproduction would be the next step if
this shows a real, non-trivial signal.

Reuses curve_diag's _fit_one_curve (baseline_R_only) and stepb_capacity_sweep's
_fit_one_curve_embed (embedded architecture) unmodified via import.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_candidate_faithful_tabm_batchensemble_stepb_capacity_sweep_20260816 as capsweep  # noqa: E402
import research_eth_candidate_faithful_tabm_batchensemble_curve_diag_20260816 as curve_diag  # noqa: E402

gate = capsweep.gate
base = capsweep.base
hard = capsweep.hard
stepb = capsweep.stepb
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_stepb_verify_20260816"
SEED = 260816
EPOCHS = 40
QUARTER_CFG = {"hidden": 96, "d_embed": 4, "n_bins": 8}


def log(msg: str) -> None:
    print(f"[faithful_tabm_stepb_verify] {msg}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = base._device("cpu")
    base._seed_everything(SEED)
    log(f"=== stage=prepare_frames (light, shared) experts=bull/bear/chop epochs_fixed={EPOCHS} embed_config=quarter{QUARTER_CFG} ===")
    frames = gate._prepare_frames_light()
    fee, slip = base.omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    x_train = base._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    hold_offsets = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384]
    x_exit_raw, y_exit, frame_exit, exit_diag = gate.exit_head._build_exit_dataset_independent(
        frames["train_df"], frames["s_train_label"], frames["train_fixed"],
        fee=fee, slip=slip, cost_mult=3.0, exit_edge_min=0.0020, hold_offsets=hold_offsets, max_samples=0,
    )
    x_exit = base._exit_input_from_position_rows(x_exit_raw, base_cols)

    import dataclasses
    quarter_cfg = dataclasses.replace(stepb.CFG, hidden=QUARTER_CFG["hidden"])

    report: dict[str, Any] = {"design": "Step B re-verification, corrected checkpoint selection (full curve, true peak), all 3 experts.", "seed": SEED, "epochs_fixed": EPOCHS, "embed_config": QUARTER_CFG, "results": {}}
    t0 = time.time()
    for expert in hard.EXPERT_NAMES:
        expert_idx = list(hard.EXPERT_NAMES).index(expert)
        log(f"=== stage=train expert={expert} arch=baseline_R_only ===")
        curve_base = curve_diag._fit_one_curve(base.ThreeHeadTabM, x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device)
        best_base = max(curve_base, key=lambda r: r["direction_balanced_accuracy_val"])
        log(f"  baseline_R_only {expert}: true peak val_bacc={best_base['direction_balanced_accuracy_val']:.4f} @epoch{best_base['epoch']} (elapsed={time.time()-t0:.0f}s)")

        log(f"=== stage=train expert={expert} arch=full_R_S_B_embed[quarter] ===")
        curve_embed, n_params = capsweep._fit_one_curve_embed(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device, cfg=quarter_cfg, d_embed=QUARTER_CFG["d_embed"], n_bins=QUARTER_CFG["n_bins"])
        best_embed = max(curve_embed, key=lambda r: r["direction_balanced_accuracy_val"])
        log(f"  full_R_S_B_embed[quarter] {expert}: n_params={n_params} true peak val_bacc={best_embed['direction_balanced_accuracy_val']:.4f} @epoch{best_embed['epoch']} (elapsed={time.time()-t0:.0f}s)")

        delta = best_embed["direction_balanced_accuracy_val"] - best_base["direction_balanced_accuracy_val"]
        log(f"  {expert} DELTA(embed_true_peak - baseline_true_peak) = {delta:+.4f}")
        report["results"][expert] = {
            "baseline_R_only": {"curve": curve_base, "true_peak": best_base},
            "full_R_S_B_embed_quarter": {"curve": curve_embed, "n_params": n_params, "true_peak": best_embed},
            "delta_true_peak_bacc": delta,
        }

    log("=== stage=summary ===")
    for expert in hard.EXPERT_NAMES:
        d = report["results"][expert]["delta_true_peak_bacc"]
        log(f"  {expert:6s} delta_true_peak_bacc(embed-baseline) = {d:+.4f}")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log(f"stage=done total_seconds={time.time()-t0:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
