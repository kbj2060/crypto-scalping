#!/usr/bin/env python3
"""Retrain the SOL TCN sequence-entry-gate FROM SCRATCH on the LIVE adaptive_squeeze
parent's own counterfactual candidate stream (not the fresh-retrain parent's).

Context: scripts/train_eval_omega462_tcn_gate_sol_liveparent_20260722.py reused a gate
FROZEN and trained against the fresh-retrain parent's candidates
(scripts/train_eval_omega462_tcn_gate_sol_20260722.py), then swapped in the live
adaptive_squeeze parent at inference time only. That parent-swap showed a clear
out-of-distribution overfit signature: VAL/OOS gains ballooned (VAL +54.94%->+159.66%,
OOS +9.95%->+64.12%) while the untouched fresh window 2026-04-01..07-21 got WORSE
(-14.53%->-15.41%). This script tests whether training the gate on the CORRECT
candidate distribution (the live parent's own) fixes that.

Reuses verbatim (no modification):
  - scripts/train_eval_omega462_tcn_gate_sol_20260722.py (`base`): slice_bundle,
    replay_with_gate, compound_metrics, counterfactual_label (via replay_with_gate's
    collect_labels path), TRAIN_START/TRAIN_END/GATE_TRAIN_END/VAL_START/VAL_END/
    OOS_START/OOS_END constants.
  - scripts/train_eval_omega462_tcn_gate_sol_liveparent_20260722.py (`lp`):
    prepare_frame_live (live adaptive_squeeze parent bundle: TabM bundle + risk sidecar +
    final scale map + duration gate, feature source with funding-divisor fix, extended
    through 2026-07-21), FRESH_START/FRESH_END.
  - train_tcn / SequenceGateArtifact / save_artifact / select_threshold (imported inside
    `base`, asset-agnostic TCN architecture and training routine).

Discipline: train/calibration split strictly within TRAIN 2025-01-01..2025-09-01 (labels
built causally, no lookahead). VAL-only hyperparameter check (6 configs) around the
known-good starting point (lookback=144, epochs=24, lr=0.0016) from the fresh-retrain
gate tuning run. Freeze ONE winner on VAL only, then touch OOS (2026-01-01..03-31)
exactly once. Fresh 2026-04-01..07-21 check with the frozen config is the critical
validation: does the correctly-distributed gate generalize, unlike the mismatched-gate
result that got worse there.

Fresh-forward contract: causal bar-by-bar walk-forward, fresh_forward_bar_by_bar=true,
trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false,
future_rows_used_for_entry=false.

Read-only w.r.t. all existing dated artifact dirs under tmp/causal_regen_20260516/ except
this script's own new output dir
tmp/causal_regen_20260516/omega462_tcn_gate_sol_liveparent_retrain_20260722/.
No live wiring, no .env / trading_bot_modules changes. Research only.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import scripts.train_eval_omega462_tcn_gate_sol_20260722 as base  # noqa: E402
import scripts.train_eval_omega462_tcn_gate_sol_liveparent_20260722 as lp  # noqa: E402
from train_eval_omega462_live_native_sequence_entry_gate_20260703 import (  # noqa: E402
    SequenceGateArtifact,
    save_artifact,
    train_tcn,
)

import json  # noqa: E402

OUT_DIR = base.ROOT / "tmp/causal_regen_20260516/omega462_tcn_gate_sol_liveparent_retrain_20260722"

# --- small VAL-only hyperparameter check (6 configs around the known-good starting
# point lookback=144, epochs=24, lr=0.0016 from the fresh-retrain gate tuning run) ---
CONFIGS = [
    {"lookback": 144, "epochs": 24, "lr": 1.6e-3},  # known-good starting point
    {"lookback": 144, "epochs": 16, "lr": 1.6e-3},
    {"lookback": 144, "epochs": 32, "lr": 1.6e-3},
    {"lookback": 144, "epochs": 24, "lr": 8.0e-4},
    {"lookback": 144, "epochs": 24, "lr": 2.4e-3},
    {"lookback": 96, "epochs": 24, "lr": 1.6e-3},
]
SEED = 260722
BATCH_SIZE = 128
MIN_VAL_TRADES = 15
GATE_TRAIN_END = "2025-06-15 00:00:00"  # unchanged: VAL trade counts (46 live vs 44
# fresh-retrain) are close enough that the 06-15 split point from the original tuning
# run is kept rather than re-swept.


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=base.json_default) + "\n", encoding="utf-8")


def label_rows_for_lookback(bundle: dict[str, Any], lookback: int, fee: float, slip: float, device: torch.device) -> list[dict[str, Any]]:
    base.LOOKBACK = lookback
    train_slice = base.slice_bundle(bundle, base.TRAIN_START, base.TRAIN_END)
    _, _, label_rows = base.replay_with_gate(
        frame=train_slice["frame"],
        base_x=train_slice["base_x"],
        dec=train_slice["dec_atr"],
        loaded=bundle["loaded"],
        margin=train_slice["margin"],
        leverage=train_slice["leverage"],
        static_tape=train_slice["static_tape"],
        fee=fee,
        slip=slip,
        device=device,
        gate_artifact=None,
        collect_labels=True,
    )
    if not label_rows:
        raise RuntimeError(f"no live-parent counterfactual entry labels for lookback={lookback}")
    return label_rows


def eval_split(bundle: dict[str, Any], start: str, end: str, artifact: Any, fee: float, slip: float, device: torch.device, lookback: int) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    base.LOOKBACK = lookback
    sl = base.slice_bundle(bundle, start, end)
    parent_metrics, parent_ledger, _ = base.replay_with_gate(
        frame=sl["frame"], base_x=sl["base_x"], dec=sl["dec_atr"], loaded=bundle["loaded"],
        margin=sl["margin"], leverage=sl["leverage"], static_tape=sl["static_tape"],
        fee=fee, slip=slip, device=device, gate_artifact=None, collect_labels=False,
    )
    gated_metrics, gated_ledger, _ = base.replay_with_gate(
        frame=sl["frame"], base_x=sl["base_x"], dec=sl["dec_atr"], loaded=bundle["loaded"],
        margin=sl["margin"], leverage=sl["leverage"], static_tape=sl["static_tape"],
        fee=fee, slip=slip, device=device, gate_artifact=artifact, collect_labels=False,
    )
    return parent_metrics, gated_metrics, parent_ledger, gated_ledger


def fit_gate(bundle: dict[str, Any], label_rows: list[dict[str, Any]], lookback: int, epochs: int, lr: float, gate_train_end: str, name: str, seed: int = SEED) -> tuple[Any, dict[str, Any]]:
    train_rows = [r for r in label_rows if r["timestamp"] < gate_train_end]
    calib_rows = [r for r in label_rows if r["timestamp"] >= gate_train_end]
    if not train_rows or not calib_rows:
        raise RuntimeError(f"empty chronological gate split: train={len(train_rows)} calib={len(calib_rows)}")
    train_seq = np.stack([r["seq"] for r in train_rows]).astype(np.float32)
    train_y = np.asarray([r["trade_return"] for r in train_rows], dtype=np.float32)
    calib_seq = np.stack([r["seq"] for r in calib_rows]).astype(np.float32)
    calib_labels_df = pd.DataFrame({"trade_return": [r["trade_return"] for r in calib_rows]})
    model, norm, train_report = train_tcn(
        train_seq=train_seq, train_y=train_y, calib_seq=calib_seq, calib_labels=calib_labels_df,
        epochs=epochs, batch_size=BATCH_SIZE, lr=lr, seed=seed, device=torch.device("cpu"),
    )
    threshold = float(train_report["threshold"]["selected"]["threshold"])
    artifact = SequenceGateArtifact(
        name=name, lookback=lookback, sample_mode="flat", feature_cols=bundle["feature_names"],
        mean=norm["mean"], std=norm["std"], threshold=threshold, threshold_payload=train_report["threshold"],
        model=model, train_report=train_report, path="",
    )
    return artifact, {"train_rows": len(train_rows), "calib_rows": len(calib_rows), "threshold": threshold}


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    t0 = time.time()

    print("stage=prepare_frame_live (extended through fresh-forward window)", flush=True)
    bundle = lp.prepare_frame_live(device)
    fee, slip = bundle["fee_slip"]
    print(f"stage=prepare_frame_live_done elapsed={time.time() - t0:.1f}s", flush=True)

    label_cache: dict[int, list[dict[str, Any]]] = {}
    grid_results: list[dict[str, Any]] = []

    print(f"stage=grid_search n_configs={len(CONFIGS)}", flush=True)
    for combo_idx, cfg in enumerate(CONFIGS):
        t_combo = time.time()
        lookback, epochs, lr = cfg["lookback"], cfg["epochs"], cfg["lr"]
        if lookback not in label_cache:
            print(f"stage=collect_labels_liveparent lookback={lookback}", flush=True)
            label_cache[lookback] = label_rows_for_lookback(bundle, lookback, fee, slip, device)
        label_rows = label_cache[lookback]

        artifact, fit_info = fit_gate(
            bundle, label_rows, lookback, epochs, lr, GATE_TRAIN_END,
            name=f"sol_tcn_seq_gate_liveparent_L{lookback}_ep{epochs}_lr{lr:g}_20260722",
        )
        parent_metrics, gated_metrics, _, _ = eval_split(bundle, base.VAL_START, base.VAL_END, artifact, fee, slip, device, lookback)

        row = {
            "combo_idx": combo_idx,
            "lookback": lookback,
            "epochs": epochs,
            "lr": lr,
            "seed": SEED,
            "gate_train_end": GATE_TRAIN_END,
            "train_rows": fit_info["train_rows"],
            "calib_rows": fit_info["calib_rows"],
            "threshold": fit_info["threshold"],
            "val_parent_pnl_pct": parent_metrics["pnl_pct"],
            "val_parent_mdd_pct": parent_metrics["mdd_pct"],
            "val_parent_trades": parent_metrics["trades"],
            "val_gated_pnl_pct": gated_metrics["pnl_pct"],
            "val_gated_mdd_pct": gated_metrics["mdd_pct"],
            "val_gated_trades": gated_metrics["trades"],
            "val_gated_wr": gated_metrics["wr"],
            "val_pnl_delta": gated_metrics["pnl_pct"] - parent_metrics["pnl_pct"],
            "val_mdd_delta": gated_metrics["mdd_pct"] - parent_metrics["mdd_pct"],
            "elapsed_s": time.time() - t_combo,
        }
        grid_results.append(row)
        print(json.dumps(row, default=base.json_default), flush=True)

    grid_df = pd.DataFrame(grid_results)
    grid_df.to_csv(OUT_DIR / "val_grid_results.csv", index=False)

    eligible = grid_df[grid_df["val_gated_trades"] >= MIN_VAL_TRADES]
    if eligible.empty:
        eligible = grid_df
    best_row = eligible.sort_values("val_gated_pnl_pct", ascending=False).iloc[0]
    frozen_lookback = int(best_row["lookback"])
    frozen_epochs = int(best_row["epochs"])
    frozen_lr = float(best_row["lr"])
    print(f"stage=freeze_winner lookback={frozen_lookback} epochs={frozen_epochs} lr={frozen_lr} gate_train_end={GATE_TRAIN_END}", flush=True)

    # --- retrain frozen winner (fresh weight init, same recipe) and touch OOS exactly once ---
    label_rows_final = label_cache[frozen_lookback]
    frozen_artifact, fit_info = fit_gate(
        bundle, label_rows_final, frozen_lookback, frozen_epochs, frozen_lr, GATE_TRAIN_END,
        name=f"sol_tcn_seq_gate_liveparent_FROZEN_L{frozen_lookback}_ep{frozen_epochs}_lr{frozen_lr:g}_20260722",
    )
    frozen_artifact.path = save_artifact(frozen_artifact, OUT_DIR)

    val_parent, val_gated, val_parent_ledger, val_gated_ledger = eval_split(bundle, base.VAL_START, base.VAL_END, frozen_artifact, fee, slip, device, frozen_lookback)
    print("stage=touch_oos_once", flush=True)
    oos_parent, oos_gated, oos_parent_ledger, oos_gated_ledger = eval_split(bundle, base.OOS_START, base.OOS_END, frozen_artifact, fee, slip, device, frozen_lookback)
    print("stage=fresh_forward_check", flush=True)
    ff_parent, ff_gated, ff_parent_ledger, ff_gated_ledger = eval_split(bundle, lp.FRESH_START, lp.FRESH_END, frozen_artifact, fee, slip, device, frozen_lookback)

    for split, p_ledger, g_ledger in (
        ("validation", val_parent_ledger, val_gated_ledger),
        ("oos_canonical", oos_parent_ledger, oos_gated_ledger),
        ("fresh_forward", ff_parent_ledger, ff_gated_ledger),
    ):
        p_ledger.to_csv(OUT_DIR / f"{split}_liveparent_alone_ledger.csv", index=False)
        g_ledger.to_csv(OUT_DIR / f"{split}_liveparent_plus_retrained_tcn_gate_ledger.csv", index=False)

    mismatched_gate_comparison = {
        "note": "from tmp/causal_regen_20260516/omega462_tcn_gate_sol_liveparent_20260722/report.json "
        "(frozen fresh-retrain-parent-trained gate applied to live-parent candidates -- OOD mismatch)",
        "val_gated_pnl_pct": 159.66,
        "oos_gated_pnl_pct": 64.12,
        "fresh_gated_pnl_pct": -15.41,
        "val_parent_alone_pnl_pct": 54.94,
        "oos_parent_alone_pnl_pct": 9.95,
        "fresh_parent_alone_pnl_pct": -14.53,
    }

    report = {
        "schema_version": "omega462.tcn_sequence_entry_gate.sol.liveparent_retrain.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": "omega462_tcn_sequence_entry_gate_sol_liveparent_retrain_20260722",
        "purpose": (
            "Retrain the TCN sequence-entry-gate from scratch on the LIVE adaptive_squeeze SOL "
            "parent's own counterfactual candidate stream (train labels collected via this parent's "
            "own replay_with_gate(collect_labels=True) on TRAIN 2025-01-01..2025-09-01), to test "
            "whether the VAL/OOS overfit + fresh-window inversion seen when reusing a "
            "fresh-retrain-parent-trained gate against live-parent candidates was a "
            "gate/candidate-distribution mismatch (fixable by retraining) or reflects the live "
            "parent's own fresh-window weakness."
        ),
        "live_parent_config": lp_config_summary(),
        "grid_search": {
            "configs": CONFIGS,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "gate_train_end": GATE_TRAIN_END,
            "min_val_trades_floor": MIN_VAL_TRADES,
            "results_csv": str(OUT_DIR / "val_grid_results.csv"),
        },
        "frozen_winner": {
            "lookback": frozen_lookback,
            "epochs": frozen_epochs,
            "lr": frozen_lr,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "gate_train_end": GATE_TRAIN_END,
            "train_rows": fit_info["train_rows"],
            "calib_rows": fit_info["calib_rows"],
            "threshold": fit_info["threshold"],
            "artifact_path": frozen_artifact.path,
        },
        "results": {
            "validation": {"start": base.VAL_START, "end_exclusive": base.VAL_END, "parent_alone": val_parent, "parent_plus_tcn_gate": val_gated},
            "oos_canonical": {"start": base.OOS_START, "end_exclusive": base.OOS_END, "parent_alone": oos_parent, "parent_plus_tcn_gate": oos_gated,
                               "note": "touched exactly once, after VAL-only model selection was frozen"},
            "fresh_forward": {"start": lp.FRESH_START, "end_exclusive": lp.FRESH_END, "parent_alone": ff_parent, "parent_plus_tcn_gate": ff_gated,
                               "note": "critical validation: does the correctly-distributed gate generalize here, unlike the mismatched-gate result that got worse in this exact window"},
        },
        "mismatched_gate_comparison": mismatched_gate_comparison,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "validation_window_canonical": [base.VAL_START, base.VAL_END],
        "oos_window_canonical": [base.OOS_START, base.OOS_END],
        "fresh_forward_window": [lp.FRESH_START, lp.FRESH_END],
        "total_elapsed_s": time.time() - t0,
        "artifacts": {"out_dir": str(OUT_DIR), "report": str(OUT_DIR / "report.json")},
    }
    write_json(OUT_DIR / "report.json", report)
    return report


def lp_config_summary() -> dict[str, Any]:
    return {
        "bundle_path": str(lp.BUNDLE_PATH_LIVE),
        "sidecar_path": str(lp.SIDECAR_PATH_LIVE),
        "features_path": str(lp.FEATURES_PATH_LIVE),
        "quality_threshold": lp.QUALITY_THRESHOLD,
        "duration_gate_threshold": lp.DURATION_THRESHOLD,
        "final_scale_map": {"long_scale": lp.LONG_SCALE, "short_scale": lp.SHORT_SCALE},
        "exit_threshold": lp.EXIT_THRESHOLD,
        "leverage_cap": lp.LEVERAGE_CAP,
        "notional_cap": lp.NOTIONAL_CAP,
        "cost_mult": lp.COST_MULT,
    }


if __name__ == "__main__":
    report = run()
    print(json.dumps(report["results"], ensure_ascii=False, indent=2, default=base.json_default), flush=True)
