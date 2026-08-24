#!/usr/bin/env python3
"""RESEARCH ONLY -- N>=5 seed reproduction of the "OLD recipe" (baseline_R_only + AdamW + plain CE
+ flat lr=2e-3, no schedule, patience=8 on combined multi-task val_loss) -- this is a faithful
replay of the exact procedure this whole candidate's very first cheap_gate used (research_eth_
candidate_faithful_tabm_batchensemble_cheap_gate_20260816.py's _fit_one), now run across the SAME
5 seeds x 3 experts as the "NEW recipe" (baseline_R_only + AdaBelief + GCE + cosine 2e-4->2e-6 +
Prechelt UP_4 on class-balanced CE, see research_eth_candidate_faithful_tabm_batchensemble_
baseline_adabelief_gce_nseed_20260816.py) for a direct, paired, N>=5-seed-confirmed "before vs
after" comparison.

Context: user asked to retrain "Odyssey4" with the new recipe and compare. The TRUE live 102-
feature pipeline (base._prepare_frames(), not the _prepare_frames_light() bypass used throughout
this candidate) was checked and confirmed BROKEN on both dev and server -- same FileNotFoundError
for the deleted LSTM/chronos context feature CSV documented in [[eth_omega4_quality_threshold_
alpha67_pipeline_irreproducible_20260815]] (a repo-wide, permanent data gap, not specific to this
candidate or this session). So this script uses the same _prepare_frames_light() data surface as
every other diagnostic in this candidate -- the best currently-available proxy, with the same
185-vs-102-feature caveat already documented throughout.

Note: this uses gate._fit_one directly (patience=8 on combined val_loss) -- for baseline_R_only
specifically, that criterion was already verified to have ~0 gap to the true accuracy peak
(0.0000-0.0008, see the Step B checkpoint-selection-bug investigation), so this OLD-recipe number
is NOT distorted by the criterion bug that affected embedding architectures -- a fair "as it was
actually run all along" baseline.
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

import research_eth_candidate_faithful_tabm_batchensemble_baseline_grid_prechelt_20260816 as bgrid  # noqa: E402

gate = bgrid.gate
base = bgrid.base
hard = bgrid.hard
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_baseline_oldrecipe_nseed_20260816"
SEEDS = [144285, 270781, 588791, 618465, 780732]  # identical to the new-recipe companion script
EPOCHS = 28  # matches this candidate's very first cheap_gate epoch budget


def log(msg: str) -> None:
    print(f"[faithful_tabm_baseline_oldrecipe_nseed] {msg}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = base._device("cpu")
    base._seed_everything(260816)
    log(f"=== stage=prepare_frames (light, shared across all seeds/experts) seeds={SEEDS} recipe=OLD(AdamW+CE+flat_lr2e-3+patience8) ===")
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

    report: dict[str, Any] = {"design": "N>=5 seed reproduction of the OLD recipe (baseline_R_only+AdamW+CE+flat lr=2e-3+patience=8), all 3 experts -- paired against the NEW-recipe companion script.", "seeds": SEEDS, "results": {}}
    t0 = time.time()
    for expert in hard.EXPERT_NAMES:
        expert_idx = list(hard.EXPERT_NAMES).index(expert)
        per_seed = []
        for seed in SEEDS:
            res = gate._fit_one(base.ThreeHeadTabM, x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=seed, epochs=EPOCHS, device=device)
            bacc = res["best_components"]["direction_balanced_accuracy"]
            per_seed.append({"seed": seed, "selected_bacc": bacc, "epochs_ran": res["epochs_ran"]})
            log(f"  {expert} seed={seed}: selected_bacc={bacc:.4f} epochs_ran={res['epochs_ran']} (elapsed={time.time()-t0:.0f}s)")
        arr = np.asarray([r["selected_bacc"] for r in per_seed], dtype=np.float64)
        report["results"][expert] = {"per_seed": per_seed, "mean_selected_bacc": float(arr.mean()), "std_selected_bacc": float(arr.std(ddof=1))}
        log(f"  {expert} SUMMARY: mean_selected_bacc={arr.mean():.4f} std={arr.std(ddof=1):.4f}")

    log("=== stage=summary ===")
    for expert in hard.EXPERT_NAMES:
        d = report["results"][expert]
        log(f"  {expert:6s} mean={d['mean_selected_bacc']:.4f} std={d['std_selected_bacc']:.4f}")
    log("  reference (NEW recipe, baseline_R_only+AdaBelief+GCE+cosine+Prechelt): bull=0.5534 bear=0.5570 chop=0.5617")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log(f"stage=done total_seconds={time.time()-t0:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
