#!/usr/bin/env python3
"""RESEARCH ONLY -- quick diagnostic follow-up to the faithful-TabM N>=5 seed result
(docs/experiments/eth_candidate_faithful_tabm_batchensemble_20260816.md). Every one of the 66 runs
across the single-seed cheap_gate AND the N>=5 seed reproduction stopped at exactly epoch 9 with
patience=8 -- meaning the best validation checkpoint was reached at epoch 1 for EVERY run, both
architectures, every expert, every seed. User asked whether "all layers were actually retrained" --
answer is yes (both architectures are fully re-initialized and trained from scratch every fit, no
warm start), but this epoch=1-always pattern raises a fair follow-up question: is the fixed
patience=8/lr=2e-3 schedule cutting off full_R_S_B's extra ~6.5% parameters before they get a chance
to actually use their capacity, i.e. is the N-seed negative result an artifact of an unfair training
budget rather than a genuine architecture effect?

This script tests that directly: same 1 seed, same 3 experts, same 2 architectures, but with
patience relaxed 8->24 and epoch budget raised 28->60 (CFG is a frozen dataclass -- patched via
dataclasses.replace on the imported module's global, not by editing the base script). If full_R_S_B
still can't find a better optimum with 3x more patience and 2x more epoch budget, that rules out
"insufficient training" as the explanation for the earlier negative result.

Reuses research_eth_candidate_faithful_tabm_batchensemble_nseed_20260816.py's already-loaded
_prepare_frames_light()/_fit_one()/ThreeHeadTabMFull unmodified via import.
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

import research_eth_candidate_faithful_tabm_batchensemble_nseed_20260816 as nseed  # noqa: E402

gate = nseed.gate
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_patience_diag_20260816"
SEED = 707258  # reuse one of the N-seed run's seeds
EPOCHS = 60
PATIENCE = 24


def log(msg: str) -> None:
    print(f"[faithful_tabm_patience_diag] {msg}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = gate.base._device("cpu")
    gate.base._seed_everything(gate.SEED)
    log("=== stage=prepare_frames (light, reused) ===")
    frames = gate._prepare_frames_light()
    fee, slip = gate.base.omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    x_train = gate.base._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    hold_offsets = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384]
    x_exit_raw, y_exit, frame_exit, exit_diag = gate.exit_head._build_exit_dataset_independent(
        frames["train_df"], frames["s_train_label"], frames["train_fixed"],
        fee=fee, slip=slip, cost_mult=3.0, exit_edge_min=0.0020, hold_offsets=hold_offsets, max_samples=0,
    )
    x_exit = gate.base._exit_input_from_position_rows(x_exit_raw, base_cols)
    log(f"  n_train={len(x_train)} n_exit={len(x_exit)} n_features={x_train.shape[1]} seed={SEED} patience={PATIENCE} epochs_budget={EPOCHS}")

    # Patch the module-global CFG (frozen dataclass) that _fit_one closes over, so patience/epoch
    # dynamics change without touching train_eval_omega1_2_tabm_3head_20260603.py at all.
    original_cfg = gate.CFG
    patched_cfg = dataclasses.replace(original_cfg, patience=PATIENCE)
    gate.CFG = patched_cfg

    report: dict[str, Any] = {"design": "patience/epoch-budget relaxation diagnostic for faithful-TabM Step A.", "seed": SEED, "patience": PATIENCE, "epochs_budget": EPOCHS, "architectures": {}}
    try:
        for arch_name, model_cls in (("baseline_R_only", gate.base.ThreeHeadTabM), ("full_R_S_B", gate.ThreeHeadTabMFull)):
            log(f"=== stage=train architecture={arch_name} ===")
            expert_results = []
            for idx, expert in enumerate(gate.hard.EXPERT_NAMES):
                res = gate._fit_one(model_cls, x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=idx, seed=SEED, epochs=EPOCHS, device=device)
                expert_results.append(res)
                log(f"  {arch_name} {expert}: epochs_ran={res['epochs_ran']}/{EPOCHS} best_val_loss={res['best_validation_loss']:.4f} "
                    f"dir_bacc={res['best_components']['direction_balanced_accuracy']:.4f} ({res['train_seconds']}s)")
            report["architectures"][arch_name] = expert_results
    finally:
        gate.CFG = original_cfg

    log("=== stage=summary ===")
    for expert in gate.hard.EXPERT_NAMES:
        b = next(r for r in report["architectures"]["baseline_R_only"] if r["expert"] == expert)
        f = next(r for r in report["architectures"]["full_R_S_B"] if r["expert"] == expert)
        log(f"  {expert:6s}: baseline epochs={b['epochs_ran']} val_loss={b['best_validation_loss']:.4f} bacc={b['best_components']['direction_balanced_accuracy']:.4f}  "
            f"|  full epochs={f['epochs_ran']} val_loss={f['best_validation_loss']:.4f} bacc={f['best_components']['direction_balanced_accuracy']:.4f}  "
            f"|  delta_bacc={f['best_components']['direction_balanced_accuracy']-b['best_components']['direction_balanced_accuracy']:+.4f}")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=gate.base._json_default) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log("stage=done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
