#!/usr/bin/env python3
"""RESEARCH ONLY -- N>=5 seed reproduction of this candidate's overall-best combination found so
far: `baseline_R_only + AdaBelief + GCE`, cosine LR schedule (2e-4->2e-6, T_max=60), Prechelt UP_4
strip-based stopping on a plain class-balanced-CE selection criterion (decoupled from the GCE
training loss). Single-seed result (bull expert only): selected_bacc=0.5749, gap to true peak only
0.0014 -- the best and most reliable result in this whole investigation, beating both the plain
baseline's own earlier unscheduled result (0.5740) and every full_R_S_B_embed[quarter] combination
tested. Per this repo's seed-diversity policy, single-seed "winners" must be reproduced across N>=5
genuinely random seeds (not a fixed-increment cluster) before any adoption claim.

Tests all 3 regime experts (bull/bear/chop) x 5 random seeds ([144285, 270781, 588791, 618465,
780732], drawn via secrets.randbelow) = 15 fits. Reuses baseline_grid_prechelt's
_fit_grid_cell_prechelt unmodified via import -- only the seed x expert loop is new.
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
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_baseline_adabelief_gce_nseed_20260816"
SEEDS = [144285, 270781, 588791, 618465, 780732]  # secrets.randbelow draws, not a fixed-increment cluster
OPT_FACTORY = bgrid.OPTIMIZERS["AdaBelief"]
USE_GCE = True


def log(msg: str) -> None:
    print(f"[faithful_tabm_baseline_adabelief_gce_nseed] {msg}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = base._device("cpu")
    base._seed_everything(bgrid.SEED)
    log(f"=== stage=prepare_frames (light, shared across all seeds/experts) seeds={SEEDS} combo=AdaBelief+GCE ===")
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

    report: dict[str, Any] = {"design": "N>=5 seed reproduction of baseline_R_only+AdaBelief+GCE (cosine+Prechelt), all 3 experts.", "seeds": SEEDS, "results": {}}
    t0 = time.time()
    for expert in hard.EXPERT_NAMES:
        expert_idx = list(hard.EXPERT_NAMES).index(expert)
        per_seed = []
        for seed in SEEDS:
            result = bgrid._fit_grid_cell_prechelt(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=seed, device=device, opt_factory=OPT_FACTORY, use_gce=USE_GCE)
            per_seed.append({"seed": seed, "selected_bacc": result["selected_bacc"], "true_peak_bacc": result["true_peak_bacc"], "selected_epoch": result["selected_epoch"], "epochs_ran": result["epochs_ran"]})
            log(f"  {expert} seed={seed}: selected_bacc={result['selected_bacc']:.4f} @epoch{result['selected_epoch']} "
                f"true_peak={result['true_peak_bacc']:.4f} epochs_ran={result['epochs_ran']} (elapsed={time.time()-t0:.0f}s)")
        arr = np.asarray([r["selected_bacc"] for r in per_seed], dtype=np.float64)
        report["results"][expert] = {"per_seed": per_seed, "mean_selected_bacc": float(arr.mean()), "std_selected_bacc": float(arr.std(ddof=1))}
        log(f"  {expert} SUMMARY: mean_selected_bacc={arr.mean():.4f} std={arr.std(ddof=1):.4f}")

    log("=== stage=summary ===")
    for expert in hard.EXPERT_NAMES:
        d = report["results"][expert]
        log(f"  {expert:6s} mean={d['mean_selected_bacc']:.4f} std={d['std_selected_bacc']:.4f}")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log(f"stage=done total_seconds={time.time()-t0:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
