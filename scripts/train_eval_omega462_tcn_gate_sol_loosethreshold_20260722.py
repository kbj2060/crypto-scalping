#!/usr/bin/env python3
"""Loosened-quality-threshold variant of the SOL TCN sequence-entry-gate sample-size fix.

Context: the live-parent gate retrain (train_eval_omega462_tcn_gate_sol_liveparent_retrain_
20260722.py) has only 74 train-window / 32 calibration candidate rows (TRAIN 2025-01-01..
2025-09-01, live adaptive_squeeze parent @ quality_threshold=0.70), and a 5-seed stability
check (stability_sweep_omega462_tcn_gate_sol_liveparent_20260722.py) found VAL PnL swinging
-52pp to +96pp across seeds. Extending the calendar window back into 2024 is off-limits
(SOL's regime3 HMM / CryptoMamba sidecars are fit on 2024 data; 2024-timestamp candidates
would leak in-sample fitting artifacts).

This script tests the alternative fix proposed: generate MORE candidates within the SAME
safe window by LOWERING the parent's entry quality-threshold from 0.70 to 0.50 (selected via
scripts/loosethreshold_candidate_count_sweep_sol_20260722.py: total candidates plateau at
~208-237 across the full 0.40-0.70 threshold range -- NOT the hoped-for 2-4x increase --
with 0.50 giving the highest train-window count, 113 vs 106 at 0.70, and reasonable
parent-alone quality). A NEW risk sidecar was retrained at this threshold (same
architecture/flags as the live one: risk-feature-mode=parent_outputs, side-split-model,
dynamic-leverage, exit-threshold=0.95 matching the live contract) since the live sidecar was
fit only on q0.70-admitted trades and would be extrapolating for newly-admitted
0.50-0.70-quality candidates. Sidecar: tmp/causal_regen_20260516/
sol_omega4_2_trade_risk_sidecar_20260707_adaptive_squeeze_q050_20260722_loosethreshold/
risk_sidecar.pkl.

Reuses verbatim (no modification):
  - scripts.train_eval_omega462_tcn_gate_sol_20260722 (`base`): slice_bundle,
    replay_with_gate, TRAIN/VAL/OOS window constants, json_default.
  - scripts.train_eval_omega462_tcn_gate_sol_liveparent_20260722 (`lp`): prepare_frame_live
    (monkeypatched QUALITY_THRESHOLD=0.50 and SIDECAR_PATH_LIVE -> the new q0.50 sidecar
    before calling it -- prepare_frame_live() resolves both as module globals at call time),
    FRESH_START/FRESH_END. All other live config (final scale map, duration gate threshold,
    leverage/notional caps, exit threshold, TP/SL ATR contract) is UNCHANGED from the live
    q0.70 config -- not re-tuned for the lower threshold, a known limitation noted in the
    report.
  - scripts.train_eval_omega462_tcn_gate_sol_liveparent_retrain_20260722 (`retrain`):
    label_rows_for_lookback, fit_gate (seed-parameterized), eval_split, CONFIGS,
    GATE_TRAIN_END, BATCH_SIZE, MIN_VAL_TRADES.

Procedure (same discipline as the live-parent scripts):
  1. VAL-only 6-config hyperparameter grid search (identical CONFIGS to the q0.70 retrain,
     not reused/copied from its winner -- re-selected fresh on this candidate distribution).
  2. Freeze ONE winner on VAL only. Touch OOS exactly once. Fresh 2026-04-01..07-21 check
     with the frozen config, once.
  3. Baseline B (parent-alone @ q0.50, no gate) on VAL/OOS/fresh -- computed here directly
     (not reused from the count-sweep CSV) so it comes from the exact same bundle/harness
     instance as the gated numbers.
  4. 5-seed stability sweep with the frozen config, to compare variance against the q0.70
     baseline (original_q070_stability_reference below, from
     tmp/causal_regen_20260516/omega462_tcn_gate_sol_liveparent_stability_20260722/report.json).

Fresh-forward contract: fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false (inherited from
replay_with_gate, unmodified).

Read-only w.r.t. all existing dated artifact dirs under tmp/causal_regen_20260516/ except
this script's own output dir
tmp/causal_regen_20260516/omega462_tcn_gate_sol_loosethreshold_20260722/ (the new risk
sidecar was written to its own new dir by the sidecar training script's own out-suffix
mechanism -- sol_omega4_2_trade_risk_sidecar_20260707_adaptive_squeeze_q050_20260722_
loosethreshold/ -- also a new dir, not a write into any existing one).

No live wiring, no .env / trading_bot_modules changes. Research only.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import scripts.train_eval_omega462_tcn_gate_sol_20260722 as base  # noqa: E402
import scripts.train_eval_omega462_tcn_gate_sol_liveparent_20260722 as lp  # noqa: E402
import scripts.train_eval_omega462_tcn_gate_sol_liveparent_retrain_20260722 as retrain  # noqa: E402

OUT_DIR = base.ROOT / "tmp/causal_regen_20260516/omega462_tcn_gate_sol_loosethreshold_20260722"

QUALITY_THRESHOLD_LOW = 0.50
SIDECAR_PATH_LOW = (
    base.ROOT
    / "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_adaptive_squeeze_q050_20260722_loosethreshold"
    / "risk_sidecar.pkl"
)

CONFIGS = retrain.CONFIGS
SEED = 260722
BATCH_SIZE = retrain.BATCH_SIZE
MIN_VAL_TRADES = retrain.MIN_VAL_TRADES
GATE_TRAIN_END = retrain.GATE_TRAIN_END

STABILITY_SEEDS = [1, 2, 3, 4, 5]

ORIGINAL_Q070_STABILITY_REFERENCE = {
    "source": "tmp/causal_regen_20260516/omega462_tcn_gate_sol_liveparent_stability_20260722/report.json",
    "train_rows": 74,
    "calib_rows": 32,
    "seeds": [1, 2, 3, 4, 5],
    "val_pnl_delta_range": [-51.964356, 25.101147],  # vs parent-alone, computed from seed_sweep_results.csv
    "val_pnl_delta_range_incl_original_seed260722": [-51.964356, 96.291054],  # includes the original single-seed 260722 run
    "fresh_gated_pnl_pct_summary": {"mean": -2.157267770713247, "std": 5.219249986601127, "min": -10.243208956304406, "max": 3.1770826269523322},
    "n_seeds_fresh_flip_neg_to_pos": 2,
    "n_seeds_total": 5,
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=base.json_default) + "\n", encoding="utf-8")


def run_one_seed(bundle, label_rows, fee, slip, device, lookback, epochs, lr, seed, gate_train_end, tag) -> dict[str, Any]:
    name = f"sol_tcn_seq_gate_loosethreshold_q050_{tag}_L{lookback}_ep{epochs}_lr{lr:g}_seed{seed}_20260722"
    artifact, fit_info = retrain.fit_gate(bundle, label_rows, lookback, epochs, lr, gate_train_end, name=name, seed=seed)
    val_parent, val_gated, _, _ = retrain.eval_split(bundle, base.VAL_START, base.VAL_END, artifact, fee, slip, device, lookback)
    oos_parent, oos_gated, _, _ = retrain.eval_split(bundle, base.OOS_START, base.OOS_END, artifact, fee, slip, device, lookback)
    ff_parent, ff_gated, _, _ = retrain.eval_split(bundle, lp.FRESH_START, lp.FRESH_END, artifact, fee, slip, device, lookback)
    return {
        "tag": tag, "seed": seed, "lookback": lookback, "epochs": epochs, "lr": lr, "gate_train_end": gate_train_end,
        "train_rows": fit_info["train_rows"], "calib_rows": fit_info["calib_rows"], "threshold": fit_info["threshold"],
        "val_parent_pnl_pct": val_parent["pnl_pct"], "val_parent_mdd_pct": val_parent["mdd_pct"], "val_parent_trades": val_parent["trades"],
        "val_gated_pnl_pct": val_gated["pnl_pct"], "val_gated_mdd_pct": val_gated["mdd_pct"], "val_gated_trades": val_gated["trades"], "val_gated_wr": val_gated["wr"],
        "val_pnl_delta": val_gated["pnl_pct"] - val_parent["pnl_pct"],
        "oos_parent_pnl_pct": oos_parent["pnl_pct"], "oos_parent_mdd_pct": oos_parent["mdd_pct"], "oos_parent_trades": oos_parent["trades"],
        "oos_gated_pnl_pct": oos_gated["pnl_pct"], "oos_gated_mdd_pct": oos_gated["mdd_pct"], "oos_gated_trades": oos_gated["trades"], "oos_gated_wr": oos_gated["wr"],
        "oos_pnl_delta": oos_gated["pnl_pct"] - oos_parent["pnl_pct"],
        "fresh_parent_pnl_pct": ff_parent["pnl_pct"], "fresh_parent_mdd_pct": ff_parent["mdd_pct"], "fresh_parent_trades": ff_parent["trades"],
        "fresh_gated_pnl_pct": ff_gated["pnl_pct"], "fresh_gated_mdd_pct": ff_gated["mdd_pct"], "fresh_gated_trades": ff_gated["trades"], "fresh_gated_wr": ff_gated["wr"],
        "fresh_pnl_delta": ff_gated["pnl_pct"] - ff_parent["pnl_pct"],
        "fresh_flip_neg_to_pos": bool(ff_parent["pnl_pct"] < 0 and ff_gated["pnl_pct"] > 0),
    }


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    t0 = time.time()

    if not SIDECAR_PATH_LOW.exists():
        raise RuntimeError(f"q0.50 risk sidecar not found: {SIDECAR_PATH_LOW}")

    print(f"stage=monkeypatch_lp threshold={QUALITY_THRESHOLD_LOW} sidecar={SIDECAR_PATH_LOW}", flush=True)
    lp.QUALITY_THRESHOLD = QUALITY_THRESHOLD_LOW
    lp.SIDECAR_PATH_LIVE = SIDECAR_PATH_LOW

    print("stage=prepare_frame_live_q050", flush=True)
    bundle = lp.prepare_frame_live(device)
    fee, slip = bundle["fee_slip"]
    print(f"stage=prepare_frame_live_done elapsed={time.time() - t0:.1f}s", flush=True)

    # --- Baseline B: parent-alone @ q0.50, no gate, on VAL/OOS/fresh (candidate for "does the
    # loosened threshold alone help or hurt") ---
    baseline_b = {}
    for split, start, end in (("validation", base.VAL_START, base.VAL_END), ("oos", base.OOS_START, base.OOS_END), ("fresh", lp.FRESH_START, lp.FRESH_END)):
        sl = base.slice_bundle(bundle, start, end)
        m, _, _ = base.replay_with_gate(
            frame=sl["frame"], base_x=sl["base_x"], dec=sl["dec_atr"], loaded=bundle["loaded"],
            margin=sl["margin"], leverage=sl["leverage"], static_tape=sl["static_tape"],
            fee=fee, slip=slip, device=device, gate_artifact=None, collect_labels=False,
        )
        baseline_b[split] = {"pnl_pct": m["pnl_pct"], "mdd_pct": m["mdd_pct"], "trades": m["trades"], "wr": m["wr"]}
    print(f"stage=baseline_b_done {json.dumps(baseline_b, default=base.json_default)}", flush=True)

    label_cache: dict[int, list[dict[str, Any]]] = {}
    grid_results = []
    print(f"stage=grid_search n_configs={len(CONFIGS)}", flush=True)
    for combo_idx, cfg in enumerate(CONFIGS):
        lookback, epochs, lr = cfg["lookback"], cfg["epochs"], cfg["lr"]
        if lookback not in label_cache:
            print(f"stage=collect_labels lookback={lookback}", flush=True)
            label_cache[lookback] = retrain.label_rows_for_lookback(bundle, lookback, fee, slip, device)
        label_rows = label_cache[lookback]
        row = run_one_seed(bundle, label_rows, fee, slip, device, lookback, epochs, lr, SEED, GATE_TRAIN_END, tag="grid_search")
        row["combo_idx"] = combo_idx
        grid_results.append(row)
        print(json.dumps(row, default=base.json_default), flush=True)

    grid_df = pd.DataFrame(grid_results)
    grid_df.to_csv(OUT_DIR / "val_grid_results.csv", index=False)

    eligible = grid_df[grid_df["val_gated_trades"] >= MIN_VAL_TRADES]
    if eligible.empty:
        eligible = grid_df
    best_row = eligible.sort_values("val_gated_pnl_pct", ascending=False).iloc[0]
    frozen_lookback, frozen_epochs, frozen_lr = int(best_row["lookback"]), int(best_row["epochs"]), float(best_row["lr"])
    print(f"stage=freeze_winner lookback={frozen_lookback} epochs={frozen_epochs} lr={frozen_lr}", flush=True)

    frozen_result = run_one_seed(bundle, label_cache[frozen_lookback], fee, slip, device, frozen_lookback, frozen_epochs, frozen_lr, SEED, GATE_TRAIN_END, tag="frozen_winner")
    print(f"stage=frozen_winner_result {json.dumps(frozen_result, default=base.json_default)}", flush=True)

    # --- 5-seed stability sweep with the frozen config ---
    print(f"stage=stability_sweep seeds={STABILITY_SEEDS}", flush=True)
    seed_results = []
    for seed in STABILITY_SEEDS:
        row = run_one_seed(bundle, label_cache[frozen_lookback], fee, slip, device, frozen_lookback, frozen_epochs, frozen_lr, seed, GATE_TRAIN_END, tag="stability_sweep")
        seed_results.append(row)
        print(json.dumps(row, default=base.json_default), flush=True)
    seed_df = pd.DataFrame(seed_results)
    seed_df.to_csv(OUT_DIR / "seed_sweep_results.csv", index=False)

    def summarize(col: str) -> dict[str, float]:
        s = seed_df[col]
        return {"mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()), "max": float(s.max())}

    seed_summary = {
        "val_pnl_delta": summarize("val_pnl_delta"),
        "oos_pnl_delta": summarize("oos_pnl_delta"),
        "fresh_pnl_delta": summarize("fresh_pnl_delta"),
        "val_gated_pnl_pct": summarize("val_gated_pnl_pct"),
        "oos_gated_pnl_pct": summarize("oos_gated_pnl_pct"),
        "fresh_gated_pnl_pct": summarize("fresh_gated_pnl_pct"),
        "n_seeds_fresh_flip_neg_to_pos": int(seed_df["fresh_flip_neg_to_pos"].sum()),
        "n_seeds_total": len(seed_df),
    }

    report = {
        "schema_version": "omega462.tcn_sequence_entry_gate.sol.loosethreshold.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": "omega462_tcn_sequence_entry_gate_sol_loosethreshold_q050_20260722",
        "purpose": (
            "Test whether lowering the live adaptive_squeeze SOL parent's entry quality-threshold "
            "from 0.70 to 0.50 (generating more candidates within the SAME safe 2025-2026 window, "
            "instead of extending back into 2024 which is off-limits due to HMM/Mamba in-sample "
            "leakage) fixes the TCN gate's seed-instability problem seen with the q0.70 sample size "
            "(74 train / 32 calib rows)."
        ),
        "quality_threshold": QUALITY_THRESHOLD_LOW,
        "sidecar_path": str(SIDECAR_PATH_LOW),
        "candidate_count_sweep_reference": str(base.ROOT / "tmp/causal_regen_20260516/omega462_tcn_gate_sol_loosethreshold_20260722/candidate_count_sweep.csv"),
        "baseline_a_live_q070_reference": {
            "note": "current live system: parent@q0.70 alone, no gate",
            "val_pnl_pct": 54.94270223709024, "oos_pnl_pct": 9.954283719743895, "fresh_pnl_pct": -14.530310505335908,
            "source": "tmp/causal_regen_20260516/omega462_tcn_gate_sol_loosethreshold_20260722/candidate_count_sweep.csv (threshold=0.70 row) == this script's own harness, matches liveparent_20260722.py's own numbers",
        },
        "baseline_b_parent_alone_q050": baseline_b,
        "grid_search": {"configs": CONFIGS, "seed": SEED, "gate_train_end": GATE_TRAIN_END, "results_csv": str(OUT_DIR / "val_grid_results.csv")},
        "frozen_winner": {"lookback": frozen_lookback, "epochs": frozen_epochs, "lr": frozen_lr, "seed": SEED, "result": frozen_result},
        "stability_sweep": {
            "seeds": STABILITY_SEEDS, "frozen_config": {"lookback": frozen_lookback, "epochs": frozen_epochs, "lr": frozen_lr},
            "results_csv": str(OUT_DIR / "seed_sweep_results.csv"), "summary": seed_summary,
        },
        "original_q070_stability_reference": ORIGINAL_Q070_STABILITY_REFERENCE,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "validation_window_canonical": [base.VAL_START, base.VAL_END],
        "oos_window_canonical": [base.OOS_START, base.OOS_END],
        "fresh_forward_window": [lp.FRESH_START, lp.FRESH_END],
        "total_elapsed_s": time.time() - t0,
        "artifacts": {"out_dir": str(OUT_DIR), "report": str(OUT_DIR / "loosethreshold_report.json")},
    }
    write_json(OUT_DIR / "loosethreshold_report.json", report)
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({"baseline_b": report["baseline_b_parent_alone_q050"], "frozen_winner": report["frozen_winner"], "stability_summary": report["stability_sweep"]["summary"]}, ensure_ascii=False, indent=2, default=base.json_default), flush=True)
