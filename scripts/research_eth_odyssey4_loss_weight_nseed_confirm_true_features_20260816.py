#!/usr/bin/env python3
"""RESEARCH ONLY -- C2 N>=5 seed re-confirmation on the TRUE live 102-feature pipeline
(Odyssey4 layer/parameter improvement proposal 20260816).

A parallel session recovered the true, ordered 102-base-column feature contract that the live
h48qual/zig075 bundle actually consumes (scripts/eth_odyssey4_true_feature_pipeline_20260816.py,
docs/experiments/eth_odyssey4_true_feature_pipeline_recovery_20260816.md) -- superseding the
185-feature `_prepare_frames_light()` proxy every earlier B/A/C1/C2/C3 script in this session used
(that proxy chain is a legitimate, separately-documented workaround for a genuinely dead vsnlstm/
chronos CSV cache, but it is NOT the live feature set).

C2's original result (research_eth_odyssey4_loss_weight_nseed_confirm_20260816.py) was flagged as
the most actionable thing to re-check: mean_delta=+0.0037, std=0.0038 -- std almost exactly equals
the mean, i.e. right at this repo's noise/signal boundary. This script reruns that EXACT same
comparison (baseline qw=0.80/ew=1.15 vs Optuna-found best qw=0.451/ew=0.598, N=5 genuinely random
seeds, expert=bull, plain CE per A1's reverted-GCE finding) with prepare_frames_true() swapped in
for gate._prepare_frames_light() -- everything else (fit_one, CFG, split logic, exit-dataset
build) is byte-for-byte the same code, imported unmodified.

fresh_forward_bar_by_bar=n/a (classifier training, internal 85/15 val split direction_balanced_
accuracy comparison only -- no backtest/portfolio ledger touched).
"""
from __future__ import annotations

import json
import secrets
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816 as gate  # noqa: E402
import research_eth_odyssey4_loss_weight_optuna_search_20260816 as opt  # noqa: E402
import eth_odyssey4_true_feature_pipeline_20260816 as truepipe  # noqa: E402

canon = gate.base
hard = gate.hard

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_loss_weight_nseed_confirm_true_features_20260816"
EXPERT = "bull"
EPOCHS = 28
N_SEEDS = 5
SEEDS = sorted(secrets.randbelow(900_000_000) + 100_000_000 for _ in range(N_SEEDS))
BASELINE_CFG = canon.CFG  # quality_loss_weight=0.80, exit_loss_weight=1.15
BEST_CFG = replace(canon.CFG, quality_loss_weight=0.45108572002184927, exit_loss_weight=0.5978814568312127)


def log(msg: str) -> None:
    print(f"[loss_weight_nseed_confirm_true] {msg}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = canon._device("cpu")
    canon._seed_everything(260816)
    log(f"=== stage=prepare_frames (TRUE 102-feature live pipeline) expert={EXPERT} seeds={SEEDS} ===")
    frames = truepipe.prepare_frames_true(disable_tp_sl=False)
    fee, slip = canon.omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    x_train = canon._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    hold_offsets = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384]
    x_exit_raw, y_exit, frame_exit, exit_diag = gate.exit_head._build_exit_dataset_independent(
        frames["train_df"], frames["s_train_label"], frames["train_fixed"],
        fee=fee, slip=slip, cost_mult=3.0, exit_edge_min=0.0020, hold_offsets=hold_offsets, max_samples=60000,
    )
    x_exit = canon._exit_input_from_position_rows(x_exit_raw, base_cols)
    expert_idx = list(hard.EXPERT_NAMES).index(EXPERT)
    log(f"  n_train={len(x_train)} n_exit={len(x_exit)} n_features={x_train.shape[1]} (should be 115 = 102 base + 13 pos)")

    report: dict[str, Any] = {
        "design": "C2 N>=5 seed re-confirmation on TRUE 102-feature live pipeline -- baseline (qw=0.80/ew=1.15) vs Optuna best (qw=0.451/ew=0.598).",
        "feature_pipeline": "eth_odyssey4_true_feature_pipeline_20260816.prepare_frames_true (102 base + 13 pos = 115 features, matches live bundle contract)",
        "seeds": SEEDS,
        "expert": EXPERT,
        "epochs_budget": EPOCHS,
        "baseline_params": {"quality_loss_weight": BASELINE_CFG.quality_loss_weight, "exit_loss_weight": BASELINE_CFG.exit_loss_weight},
        "best_params": {"quality_loss_weight": BEST_CFG.quality_loss_weight, "exit_loss_weight": BEST_CFG.exit_loss_weight},
        "results": {"baseline": [], "best": []},
    }
    t_start = time.time()
    for seed in SEEDS:
        res_base = opt.fit_one(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=seed, epochs=EPOCHS, device=device, cfg=BASELINE_CFG)
        report["results"]["baseline"].append(res_base)
        log(f"  baseline seed={seed}: dir_bacc={res_base['best_components']['direction_balanced_accuracy']:.4f} val_loss={res_base['best_validation_loss']:.4f} ({res_base['train_seconds']}s, elapsed={time.time()-t_start:.0f}s)")

        res_best = opt.fit_one(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=seed, epochs=EPOCHS, device=device, cfg=BEST_CFG)
        report["results"]["best"].append(res_best)
        log(f"  best     seed={seed}: dir_bacc={res_best['best_components']['direction_balanced_accuracy']:.4f} val_loss={res_best['best_validation_loss']:.4f} ({res_best['train_seconds']}s, elapsed={time.time()-t_start:.0f}s)")

    b_runs = report["results"]["baseline"]
    g_runs = report["results"]["best"]
    deltas = [g["best_components"]["direction_balanced_accuracy"] - b["best_components"]["direction_balanced_accuracy"] for b, g in zip(b_runs, g_runs)]
    arr = np.asarray(deltas, dtype=np.float64)
    n_pos = int((arr > 0).sum())
    n_neg = int((arr < 0).sum())
    summary = {
        "baseline_mean_dir_bacc": float(np.mean([b["best_components"]["direction_balanced_accuracy"] for b in b_runs])),
        "best_mean_dir_bacc": float(np.mean([g["best_components"]["direction_balanced_accuracy"] for g in g_runs])),
        "dir_bacc_deltas_best_minus_baseline": deltas,
        "dir_bacc_mean_delta": float(arr.mean()),
        "dir_bacc_std_delta": float(arr.std(ddof=1)),
        "n_seeds_improved": n_pos,
        "n_seeds_worsened": n_neg,
        "sign_consistent": bool(n_pos == 0 or n_neg == 0),
    }
    report["summary"] = summary
    log(f"=== SUMMARY: baseline_mean={summary['baseline_mean_dir_bacc']:.4f} best_mean={summary['best_mean_dir_bacc']:.4f} mean_delta={summary['dir_bacc_mean_delta']:+.4f} std={summary['dir_bacc_std_delta']:.4f} improved={n_pos}/{len(SEEDS)} sign_consistent={summary['sign_consistent']} ===")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
