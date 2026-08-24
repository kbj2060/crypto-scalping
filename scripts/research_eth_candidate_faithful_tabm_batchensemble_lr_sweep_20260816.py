#!/usr/bin/env python3
"""RESEARCH ONLY -- LR sweep follow-up to the epoch-1-is-always-best mystery
(docs/experiments/eth_candidate_faithful_tabm_batchensemble_20260816.md). The full per-epoch curve
diagnostic (research_eth_candidate_faithful_tabm_batchensemble_curve_diag_20260816.py, at the
default lr=2e-3) showed a smooth, monotonic memorization signature -- train_loss/train_bacc keep
improving epoch over epoch while val_loss/val_bacc smoothly degrade from epoch ~2 onward, not an
oscillating/spiky curve. A literature research agent found this matches Arpit et al.'s "generalize
first, then memorize" pattern (arXiv:1706.05394), and separately confirmed lr=2e-3 with no schedule
is TabM's own paper default, not an obvious bug. User asked to test a MUCH smaller LR anyway, as a
direct empirical check: if a smaller LR slows the memorization-phase onset (giving the model more
useful epochs before noise-fitting dominates), the val curve should peak later and/or higher than
the lr=2e-3 baseline; if it doesn't change the qualitative shape, that's further evidence the
ceiling here is set by label information content, not optimization dynamics.

Tests architecture=baseline_R_only only (isolates the LR variable cleanly, consistent with this
whole candidate's one-variable-at-a-time ablation discipline), expert=bull, seed=260816 (matching
the existing curve diagnostic for direct comparability), fixed 40-epoch budget, no early stopping,
full per-epoch curve logged for each LR value.

Reuses research_eth_candidate_faithful_tabm_batchensemble_curve_diag_20260816.py's _fit_one_curve
and _prepare_frames_light unmodified via import -- only the LR sweep loop (via dataclasses.replace
on the imported module's CFG) is new.
"""
from __future__ import annotations

import dataclasses
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

import research_eth_candidate_faithful_tabm_batchensemble_curve_diag_20260816 as curve_diag  # noqa: E402

gate = curve_diag.gate
base = curve_diag.base
hard = curve_diag.hard
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_lr_sweep_20260816"
SEED = curve_diag.SEED
EXPERT = curve_diag.EXPERT
EPOCHS = curve_diag.EPOCHS
LR_VALUES = [2.0e-3, 2.0e-4, 2.0e-5, 2.0e-6]  # 2e-3 = existing default/control, then 10x/100x/1000x smaller


def log(msg: str) -> None:
    print(f"[faithful_tabm_lr_sweep] {msg}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = base._device("cpu")
    base._seed_everything(SEED)
    log(f"=== stage=prepare_frames (light, shared) expert={EXPERT} epochs_fixed={EPOCHS} lr_values={LR_VALUES} ===")
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
    expert_idx = list(hard.EXPERT_NAMES).index(EXPERT)

    original_cfg = curve_diag.CFG
    report: dict[str, Any] = {"design": "LR sweep on the epoch-1-best mystery, architecture=baseline_R_only, fixed epochs, no early stopping.", "seed": SEED, "expert": EXPERT, "epochs_fixed": EPOCHS, "lr_values": LR_VALUES, "curves": {}}
    t_start = time.time()
    try:
        for lr in LR_VALUES:
            lr_key = f"lr_{lr:.0e}"
            log(f"=== stage=train lr={lr:.1e} ===")
            curve_diag.CFG = dataclasses.replace(original_cfg, lr=float(lr))
            curve = curve_diag._fit_one_curve(base.ThreeHeadTabM, x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device)
            report["curves"][lr_key] = curve
            best = min(curve, key=lambda r: r["val_loss"])
            best_bacc_row = max(curve, key=lambda r: r["direction_balanced_accuracy_val"])
            log(f"  lr={lr:.1e}: best val_loss at epoch {best['epoch']} ({best['val_loss']:.4f}); "
                f"best val_bacc at epoch {best_bacc_row['epoch']} ({best_bacc_row['direction_balanced_accuracy_val']:.4f}); "
                f"final(epoch{EPOCHS}) val_loss={curve[-1]['val_loss']:.4f} val_bacc={curve[-1]['direction_balanced_accuracy_val']:.4f} "
                f"(elapsed={time.time()-t_start:.0f}s)")
    finally:
        curve_diag.CFG = original_cfg

    log("=== stage=summary ===")
    for lr in LR_VALUES:
        lr_key = f"lr_{lr:.0e}"
        curve = report["curves"][lr_key]
        best_bacc_row = max(curve, key=lambda r: r["direction_balanced_accuracy_val"])
        log(f"  lr={lr:.1e}: peak val_bacc={best_bacc_row['direction_balanced_accuracy_val']:.4f} at epoch {best_bacc_row['epoch']}/{EPOCHS}")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log(f"stage=done total_seconds={time.time()-t_start:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
