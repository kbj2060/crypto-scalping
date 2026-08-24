#!/usr/bin/env python3
"""RESEARCH ONLY -- N>=5 genuinely random seed confirmation for the NCL diversity-loss candidate,
following the single-seed cheap_gate (research_eth_odyssey4_ncl_diversity_loss_cheap_gate_20260817.py
+ its lambda-gap-fill follow-up). That sweep (lambda in {0,1,2,3,5,7,10,100,1000}, single seed) found
a smooth, monotonic dose-response, not a hard cliff: lambda<=3 barely moves diversity or accuracy;
lambda=5 is the first point where both move together; lambda=7 is the sharp transition (accuracy
-0.04 to -0.08, correlation 0.99->0.45-0.56); lambda>=10 is fully broken on both axes. No lambda in
the tested range improved accuracy while meaningfully increasing diversity.

This script confirms the two decision-relevant boundary points across N=5 genuinely random seeds
(secrets.randbelow, Seed-Diversity Ensemble Promotion Gate):
  - lambda=2: representative "small lambda, no measurable effect" region -- is the tiny +0.0016
    single-seed accuracy uptick at lambda in {1,2} real or noise?
  - lambda=7: representative "transition" region where diversity first meaningfully increases -- is
    the accuracy cost at that point a robust, seed-consistent tradeoff?
  - lambda=0 is the paired control (same seed/init as lambda=2/7 within each seed, only the loss
    differs -- this is a clean paired design, not an independent baseline draw).

Reuses _fit_with_ncl from the cheap_gate script unmodified.
"""
from __future__ import annotations

import json
import secrets
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

import research_eth_odyssey4_ncl_diversity_loss_cheap_gate_20260817 as ncl  # noqa: E402

canon = ncl.canon
exit_head = ncl.exit_head
hard = ncl.hard
truepipe = ncl.truepipe

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_ncl_diversity_loss_nseed_confirm_20260817"
EXPERT = "bull"
EPOCHS = 28
N_SEEDS = 5
SEEDS = sorted(secrets.randbelow(900_000_000) + 100_000_000 for _ in range(N_SEEDS))
LAMBDAS = [0.0, 2.0, 7.0]


def log(msg: str) -> None:
    print(f"[ncl_nseed] {msg}", flush=True)


def _delta_summary(arm_runs: list[dict[str, Any]], ctrl_runs: list[dict[str, Any]], *, key_path: list[str]) -> dict[str, Any]:
    def get(run: dict[str, Any]) -> float:
        v: Any = run
        for k in key_path:
            v = v[k]
        return float(v)

    deltas = [get(a) - get(c) for a, c in zip(arm_runs, ctrl_runs)]
    arr = np.asarray(deltas, dtype=np.float64)
    n_pos = int((arr > 0).sum())
    n_neg = int((arr < 0).sum())
    return {
        "deltas": deltas,
        "mean_delta": float(arr.mean()),
        "std_delta": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n_improved": n_pos,
        "n_worsened": n_neg,
        "sign_consistent": bool(n_pos == 0 or n_neg == 0),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = canon._device("cpu")
    canon._seed_everything(260816)
    log(f"=== stage=prepare_frames (true 115-feature pipeline) expert={EXPERT} seeds={SEEDS} lambdas={LAMBDAS} ===")
    frames = truepipe.prepare_frames_true(disable_tp_sl=False)
    fee, slip = canon.omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    x_train = canon._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    hold_offsets = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384]
    x_exit_raw, y_exit, frame_exit, exit_diag = exit_head._build_exit_dataset_independent(
        frames["train_df"], frames["s_train_label"], frames["train_fixed"],
        fee=fee, slip=slip, cost_mult=3.0, exit_edge_min=0.0020, hold_offsets=hold_offsets, max_samples=60000,
    )
    x_exit = ncl.canon._exit_input_from_position_rows(x_exit_raw, base_cols)
    expert_idx = list(hard.EXPERT_NAMES).index(EXPERT)
    log(f"  n_train={len(x_train)} n_exit={len(x_exit)} n_features={x_train.shape[1]}")

    report: dict[str, Any] = {
        "design": "NCL diversity-loss N>=5 seed confirmation at the two cheap_gate boundary points (lambda=2 no-effect region, lambda=7 transition region), paired against lambda=0 (same seed/init per seed).",
        "seed_source": "secrets.randbelow (genuinely random, not fixed-increment) -- Seed-Diversity Ensemble Promotion Gate",
        "seeds": SEEDS,
        "expert": EXPERT,
        "epochs_budget": EPOCHS,
        "lambdas_tested": LAMBDAS,
        "runs": {f"lambda_{lam}": [] for lam in LAMBDAS},
    }
    t_start = time.time()
    for seed in SEEDS:
        for lam in LAMBDAS:
            res = ncl._fit_with_ncl(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=seed, epochs=EPOCHS, device=device, lam=lam)
            report["runs"][f"lambda_{lam}"].append(res)
            cs = res["selected_collapse"]
            cp = res["true_peak_collapse"]
            log(
                f"  seed={seed} lambda={lam}: selected(ep{res['selected_epoch']})dir_bacc={res['selected_dir_bacc']:.4f} corr={cs['mean_pairwise_corr_of_top_confidence']:.4f} | "
                f"true_peak(ep{res['true_peak_epoch']})dir_bacc={res['true_peak_dir_bacc']:.4f} corr={cp['mean_pairwise_corr_of_top_confidence']:.4f} "
                f"({res['train_seconds']}s, elapsed={time.time()-t_start:.0f}s)"
            )

    ctrl_runs = report["runs"]["lambda_0.0"]
    summary: dict[str, Any] = {}
    for lam in LAMBDAS:
        if lam == 0.0:
            continue
        arm_runs = report["runs"][f"lambda_{lam}"]
        sel_bacc = _delta_summary(arm_runs, ctrl_runs, key_path=["selected_dir_bacc"])
        peak_bacc = _delta_summary(arm_runs, ctrl_runs, key_path=["true_peak_dir_bacc"])
        sel_corr = _delta_summary(arm_runs, ctrl_runs, key_path=["selected_collapse", "mean_pairwise_corr_of_top_confidence"])
        peak_corr = _delta_summary(arm_runs, ctrl_runs, key_path=["true_peak_collapse", "mean_pairwise_corr_of_top_confidence"])
        summary[f"lambda_{lam}_vs_0"] = {
            "selected_dir_bacc": sel_bacc, "true_peak_dir_bacc": peak_bacc,
            "selected_corr": sel_corr, "true_peak_corr": peak_corr,
        }
        log(
            f"=== VERDICT lambda_{lam}_vs_0: selected_bacc_delta={sel_bacc['mean_delta']:+.4f}(std={sel_bacc['std_delta']:.4f},{sel_bacc['n_improved']}/{N_SEEDS}) "
            f"true_peak_bacc_delta={peak_bacc['mean_delta']:+.4f}(std={peak_bacc['std_delta']:.4f},{peak_bacc['n_improved']}/{N_SEEDS}) "
            f"true_peak_corr_delta={peak_corr['mean_delta']:+.4f}(std={peak_corr['std_delta']:.4f}) ==="
        )
    report["summary"] = summary

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"report written to {out_path} (total elapsed {time.time()-t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
