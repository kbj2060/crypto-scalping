#!/usr/bin/env python3
"""RESEARCH ONLY -- N>=5 seed reproduction for the faithful-TabM candidate's Step A cheap_gate
(docs/model_contracts/eth_candidate_faithful_tabm_batchensemble_contract_20260816.md,
docs/experiments/eth_candidate_faithful_tabm_batchensemble_20260816.md). The single-seed cheap_gate
came back with signs that disagreed across the 3 regime experts (bull improved on
direction_balanced_accuracy but worsened on every loss component; bear was the mirror image; chop
was flat) -- exactly this repo's established "single-seed winner is often noise" pattern
([[tabm_hp_low_signal_pattern]] in memory). Per the Seed-Diversity Ensemble Promotion Gate
(CLAUDE.md) and the contract's own Red Team Gate ("N>=5 시드로 재현, 진짜 무작위 시드"), this
script reruns the SAME baseline_R_only vs full_R_S_B comparison across 5 genuinely random seeds
(drawn via `secrets.randbelow`, NOT a fixed-increment cluster like the Sigma3-1h incident that
motivated that policy) to see whether a consistent sign emerges once seed noise is averaged out.

Reuses everything from research_eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816.py
(the ThreeHeadTabMFull class, _fit_one, _prepare_frames_light) unmodified via import -- only the
seed loop and aggregation are new. Frame/exit-dataset prep runs ONCE (expensive, ~266s) and is
reused across all 30 fits (5 seeds x 2 architectures x 3 experts).

Single-seed caveats from the cheap_gate script still apply unchanged: _prepare_frames_light's
feature_cols is 185 (not the live 102), and this is an architecture-only ablation on this script's
own base zigzag_action-label pipeline, not a bit-for-bit live-bundle reproduction.

fresh_forward_bar_by_bar=n/a (classifier training, no backtest/portfolio ledger touched).
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

import research_eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816 as gate  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_nseed_20260816"
SEEDS = [211581, 262041, 393534, 646498, 707258]  # secrets.randbelow draws, not a fixed-increment cluster
EPOCHS = gate.EPOCHS


def log(msg: str) -> None:
    print(f"[faithful_tabm_nseed] {msg}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = gate.base._device("cpu")
    gate.base._seed_everything(gate.SEED)
    log(f"=== stage=prepare_frames (light, shared across all {len(SEEDS)} seeds) ===")
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
    log(f"  n_train={len(x_train)} n_exit={len(x_exit)} n_features={x_train.shape[1]} seeds={SEEDS}")

    report: dict[str, Any] = {
        "design": "faithful-TabM Step A N>=5 seed reproduction -- BatchEnsemble R+S+B completion.",
        "seeds": SEEDS,
        "epochs_budget": EPOCHS,
        "results": {"baseline_R_only": {}, "full_R_S_B": {}},
    }

    t_start = time.time()
    for arch_name, model_cls in (("baseline_R_only", gate.base.ThreeHeadTabM), ("full_R_S_B", gate.ThreeHeadTabMFull)):
        for expert_idx, expert in enumerate(gate.hard.EXPERT_NAMES):
            per_seed = []
            for seed in SEEDS:
                res = gate._fit_one(
                    model_cls, x_train, y_train, train_raw, x_exit, y_exit, frame_exit,
                    expert_idx=expert_idx, seed=seed, epochs=EPOCHS, device=device,
                )
                per_seed.append(res)
                log(f"  {arch_name} {expert} seed={seed}: val_loss={res['best_validation_loss']:.4f} "
                    f"dir_bacc={res['best_components']['direction_balanced_accuracy']:.4f} "
                    f"({res['train_seconds']}s, elapsed={time.time()-t_start:.0f}s)")
            report["results"][arch_name][expert] = per_seed

    log("=== stage=aggregate ===")
    summary: dict[str, Any] = {}
    for expert in gate.hard.EXPERT_NAMES:
        b_runs = report["results"]["baseline_R_only"][expert]
        f_runs = report["results"]["full_R_S_B"][expert]
        metrics = {
            "val_loss": [f["best_validation_loss"] - b["best_validation_loss"] for b, f in zip(b_runs, f_runs)],
            "dir_bacc": [f["best_components"]["direction_balanced_accuracy"] - b["best_components"]["direction_balanced_accuracy"] for b, f in zip(b_runs, f_runs)],
            "quality_val_loss": [f["best_components"]["quality_val_loss"] - b["best_components"]["quality_val_loss"] for b, f in zip(b_runs, f_runs)],
            "exit_val_loss": [f["best_components"]["exit_val_loss"] - b["best_components"]["exit_val_loss"] for b, f in zip(b_runs, f_runs)],
        }
        expert_summary = {}
        for name, deltas in metrics.items():
            arr = np.asarray(deltas, dtype=np.float64)
            n_pos = int((arr > 0).sum())
            n_neg = int((arr < 0).sum())
            expert_summary[name] = {
                "deltas": deltas,
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)),
                "n_improved": n_neg if name.endswith("loss") else n_pos,  # lower=better for *_loss, higher=better for bacc
                "n_worsened": n_pos if name.endswith("loss") else n_neg,
                "sign_consistent": bool(n_pos == 0 or n_neg == 0),
            }
        summary[expert] = expert_summary
        log(f"  {expert:6s} val_loss  mean_delta={expert_summary['val_loss']['mean']:+.4f} std={expert_summary['val_loss']['std']:.4f} improved={expert_summary['val_loss']['n_improved']}/5")
        log(f"  {expert:6s} dir_bacc  mean_delta={expert_summary['dir_bacc']['mean']:+.4f} std={expert_summary['dir_bacc']['std']:.4f} improved={expert_summary['dir_bacc']['n_improved']}/5")

    report["summary"] = summary
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=gate.base._json_default) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log(f"stage=done total_seconds={time.time()-t_start:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
