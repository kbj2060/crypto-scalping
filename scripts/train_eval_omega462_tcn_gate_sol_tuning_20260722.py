#!/usr/bin/env python3
"""VAL-only hyperparameter tuning for the SOL Omega4.6.2 TCN sequence-entry-gate.

Reuses every heavy building block (prepare_frame / slice_bundle / replay_with_gate /
train_tcn / select_threshold / SequenceEntryTCN) from
scripts/train_eval_omega462_tcn_gate_sol_20260722.py (the already-audited baseline
script) without modifying that file. `LOOKBACK` is a module-level global inside that
script that replay_with_gate() reads at call time, so we monkey-patch
`base.LOOKBACK` per grid cell before calling into it -- no code duplication of the
900-line bar-by-bar replay loop.

Discipline: every grid cell is scored on VAL (2025-09-01..2025-12-31) ONLY. The
single frozen winner is then evaluated on OOS (2026-01-01..2026-03-31) exactly once,
plus one additional fresh-forward check (2026-04-01..2026-07-21). Neither OOS window
is used for any model-selection decision.

Outputs -> tmp/causal_regen_20260516/omega462_tcn_gate_sol_tuning_20260722/
(does not touch tmp/causal_regen_20260516/omega462_tcn_gate_sol_20260722/, the
original untuned run's output dir).
"""
from __future__ import annotations

import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import scripts.train_eval_omega462_tcn_gate_sol_20260722 as base
from train_eval_omega462_live_native_sequence_entry_gate_20260703 import (
    SequenceGateArtifact,
    save_artifact,
    select_threshold,
    train_tcn,
)

OUT_DIR = base.ROOT / "tmp/causal_regen_20260516/omega462_tcn_gate_sol_tuning_20260722"
FRESH_FORWARD_START = "2026-04-01 00:00:00"
FRESH_FORWARD_END = "2026-07-21 12:00:00"  # data ends 2026-07-21 11:45

# Extend the working frame beyond the baseline's FRAME_END (2026-04-01) so the
# fresh-forward window (through 2026-07-21) is available without re-running TabM
# inference twice.
FRAME_END_EXTENDED = "2026-07-21 12:00:00"

# --- grid ---------------------------------------------------------------
LOOKBACKS = [24, 48, 96, 144]
EPOCHS_GRID = [8, 16, 24]
LR_GRID = [4.0e-4, 8.0e-4, 1.6e-3]
SEED = 260722  # baseline seed; held fixed in the primary grid
BATCH_SIZE = 128
MIN_VAL_TRADES = 15  # sanity floor so a lucky 3-trade config can't "win"


def json_default(obj: Any) -> Any:
    return base.json_default(obj)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


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
        raise RuntimeError(f"no counterfactual entry labels for lookback={lookback}")
    return label_rows


def eval_split(bundle: dict[str, Any], start: str, end: str, artifact: Any, fee: float, slip: float, device: torch.device, lookback: int) -> tuple[dict[str, Any], dict[str, Any]]:
    base.LOOKBACK = lookback
    sl = base.slice_bundle(bundle, start, end)
    parent_metrics, _, _ = base.replay_with_gate(
        frame=sl["frame"], base_x=sl["base_x"], dec=sl["dec_atr"], loaded=bundle["loaded"],
        margin=sl["margin"], leverage=sl["leverage"], static_tape=sl["static_tape"],
        fee=fee, slip=slip, device=device, gate_artifact=None, collect_labels=False,
    )
    gated_metrics, gated_ledger, _ = base.replay_with_gate(
        frame=sl["frame"], base_x=sl["base_x"], dec=sl["dec_atr"], loaded=bundle["loaded"],
        margin=sl["margin"], leverage=sl["leverage"], static_tape=sl["static_tape"],
        fee=fee, slip=slip, device=device, gate_artifact=artifact, collect_labels=False,
    )
    return parent_metrics, gated_metrics


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    t0 = time.time()

    base.FRAME_END = FRAME_END_EXTENDED
    print("stage=prepare_frame (extended through fresh-forward window)", flush=True)
    bundle = base.prepare_frame(device)
    fee, slip = bundle["fee_slip"]
    print(f"stage=prepare_frame_done elapsed={time.time() - t0:.1f}s", flush=True)

    grid_results: list[dict[str, Any]] = []
    label_cache: dict[int, list[dict[str, Any]]] = {}

    combos = list(itertools.product(LOOKBACKS, EPOCHS_GRID, LR_GRID))
    print(f"stage=grid_search n_combos={len(combos)}", flush=True)

    for combo_idx, (lookback, epochs, lr) in enumerate(combos):
        t_combo = time.time()
        if lookback not in label_cache:
            print(f"stage=collect_labels lookback={lookback}", flush=True)
            label_cache[lookback] = label_rows_for_lookback(bundle, lookback, fee, slip, device)
        label_rows = label_cache[lookback]

        train_rows = [r for r in label_rows if r["timestamp"] < base.GATE_TRAIN_END]
        calib_rows = [r for r in label_rows if r["timestamp"] >= base.GATE_TRAIN_END]
        if not train_rows or not calib_rows:
            print(f"skip lookback={lookback} epochs={epochs} lr={lr}: empty split train={len(train_rows)} calib={len(calib_rows)}", flush=True)
            continue

        train_seq = np.stack([r["seq"] for r in train_rows]).astype(np.float32)
        train_y = np.asarray([r["trade_return"] for r in train_rows], dtype=np.float32)
        calib_seq = np.stack([r["seq"] for r in calib_rows]).astype(np.float32)
        calib_labels_df = pd.DataFrame({"trade_return": [r["trade_return"] for r in calib_rows]})

        model, norm, train_report = train_tcn(
            train_seq=train_seq, train_y=train_y, calib_seq=calib_seq, calib_labels=calib_labels_df,
            epochs=epochs, batch_size=BATCH_SIZE, lr=lr, seed=SEED, device=device,
        )
        threshold = float(train_report["threshold"]["selected"]["threshold"])
        artifact = SequenceGateArtifact(
            name=f"sol_tcn_seq_gate_L{lookback}_ep{epochs}_lr{lr:g}_20260722",
            lookback=lookback, sample_mode="flat", feature_cols=bundle["feature_names"],
            mean=norm["mean"], std=norm["std"], threshold=threshold, threshold_payload=train_report["threshold"],
            model=model, train_report=train_report, path="",
        )

        parent_metrics, gated_metrics = eval_split(bundle, base.VAL_START, base.VAL_END, artifact, fee, slip, device, lookback)

        row = {
            "combo_idx": combo_idx,
            "lookback": lookback,
            "epochs": epochs,
            "lr": lr,
            "seed": SEED,
            "gate_train_end": base.GATE_TRAIN_END,
            "train_rows": len(train_rows),
            "calib_rows": len(calib_rows),
            "threshold": threshold,
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
        print(json.dumps(row, default=json_default), flush=True)

    grid_df = pd.DataFrame(grid_results)
    grid_df.to_csv(OUT_DIR / "val_grid_results.csv", index=False)

    # --- secondary cheap sweep: GATE_TRAIN_END split point, on best lookback from primary grid ---
    eligible = grid_df[grid_df["val_gated_trades"] >= MIN_VAL_TRADES]
    if eligible.empty:
        eligible = grid_df
    best_row = eligible.sort_values("val_gated_pnl_pct", ascending=False).iloc[0]
    best_lookback = int(best_row["lookback"])

    split_candidates = ["2025-05-15 00:00:00", "2025-06-15 00:00:00", "2025-07-01 00:00:00"]
    split_results: list[dict[str, Any]] = []
    label_rows_best_lb = label_cache[best_lookback]
    for split_point in split_candidates:
        train_rows = [r for r in label_rows_best_lb if r["timestamp"] < split_point]
        calib_rows = [r for r in label_rows_best_lb if r["timestamp"] >= split_point]
        if not train_rows or not calib_rows:
            continue
        train_seq = np.stack([r["seq"] for r in train_rows]).astype(np.float32)
        train_y = np.asarray([r["trade_return"] for r in train_rows], dtype=np.float32)
        calib_seq = np.stack([r["seq"] for r in calib_rows]).astype(np.float32)
        calib_labels_df = pd.DataFrame({"trade_return": [r["trade_return"] for r in calib_rows]})
        model, norm, train_report = train_tcn(
            train_seq=train_seq, train_y=train_y, calib_seq=calib_seq, calib_labels=calib_labels_df,
            epochs=int(best_row["epochs"]), batch_size=BATCH_SIZE, lr=float(best_row["lr"]), seed=SEED, device=device,
        )
        threshold = float(train_report["threshold"]["selected"]["threshold"])
        artifact = SequenceGateArtifact(
            name=f"sol_tcn_seq_gate_L{best_lookback}_split{split_point[:10]}_20260722",
            lookback=best_lookback, sample_mode="flat", feature_cols=bundle["feature_names"],
            mean=norm["mean"], std=norm["std"], threshold=threshold, threshold_payload=train_report["threshold"],
            model=model, train_report=train_report, path="",
        )
        parent_metrics, gated_metrics = eval_split(bundle, base.VAL_START, base.VAL_END, artifact, fee, slip, device, best_lookback)
        row = {
            "split_point": split_point,
            "train_rows": len(train_rows),
            "calib_rows": len(calib_rows),
            "val_gated_pnl_pct": gated_metrics["pnl_pct"],
            "val_gated_mdd_pct": gated_metrics["mdd_pct"],
            "val_gated_trades": gated_metrics["trades"],
            "val_gated_wr": gated_metrics["wr"],
        }
        split_results.append(row)
        print(json.dumps(row, default=json_default), flush=True)

    split_df = pd.DataFrame(split_results)
    split_df.to_csv(OUT_DIR / "val_split_point_sweep.csv", index=False)

    # winner selection: best of primary grid vs best split-point variant (still VAL only)
    best_overall = dict(best_row)
    best_split_row = None
    if not split_df.empty:
        elig_split = split_df[split_df["val_gated_trades"] >= MIN_VAL_TRADES]
        if elig_split.empty:
            elig_split = split_df
        best_split_row = elig_split.sort_values("val_gated_pnl_pct", ascending=False).iloc[0]
        if float(best_split_row["val_gated_pnl_pct"]) > float(best_overall["val_gated_pnl_pct"]):
            best_overall = {
                "lookback": best_lookback,
                "epochs": int(best_row["epochs"]),
                "lr": float(best_row["lr"]),
                "seed": SEED,
                "gate_train_end": best_split_row["split_point"],
                "val_gated_pnl_pct": float(best_split_row["val_gated_pnl_pct"]),
                "val_gated_mdd_pct": float(best_split_row["val_gated_mdd_pct"]),
                "val_gated_trades": int(best_split_row["val_gated_trades"]),
            }
        else:
            best_overall["gate_train_end"] = base.GATE_TRAIN_END

    frozen_lookback = int(best_overall["lookback"])
    frozen_epochs = int(best_overall["epochs"])
    frozen_lr = float(best_overall["lr"])
    frozen_gate_train_end = str(best_overall["gate_train_end"])

    print(f"stage=freeze_winner lookback={frozen_lookback} epochs={frozen_epochs} lr={frozen_lr} gate_train_end={frozen_gate_train_end}", flush=True)

    # --- retrain frozen winner and touch OOS exactly once ---
    label_rows_final = label_cache[frozen_lookback]
    train_rows = [r for r in label_rows_final if r["timestamp"] < frozen_gate_train_end]
    calib_rows = [r for r in label_rows_final if r["timestamp"] >= frozen_gate_train_end]
    train_seq = np.stack([r["seq"] for r in train_rows]).astype(np.float32)
    train_y = np.asarray([r["trade_return"] for r in train_rows], dtype=np.float32)
    calib_seq = np.stack([r["seq"] for r in calib_rows]).astype(np.float32)
    calib_labels_df = pd.DataFrame({"trade_return": [r["trade_return"] for r in calib_rows]})
    model, norm, train_report = train_tcn(
        train_seq=train_seq, train_y=train_y, calib_seq=calib_seq, calib_labels=calib_labels_df,
        epochs=frozen_epochs, batch_size=BATCH_SIZE, lr=frozen_lr, seed=SEED, device=device,
    )
    threshold = float(train_report["threshold"]["selected"]["threshold"])
    frozen_artifact = SequenceGateArtifact(
        name=f"sol_tcn_seq_gate_FROZEN_L{frozen_lookback}_ep{frozen_epochs}_lr{frozen_lr:g}_20260722",
        lookback=frozen_lookback, sample_mode="flat", feature_cols=bundle["feature_names"],
        mean=norm["mean"], std=norm["std"], threshold=threshold, threshold_payload=train_report["threshold"],
        model=model, train_report=train_report, path="",
    )
    frozen_artifact.path = save_artifact(frozen_artifact, OUT_DIR)

    val_parent, val_gated = eval_split(bundle, base.VAL_START, base.VAL_END, frozen_artifact, fee, slip, device, frozen_lookback)
    print("stage=touch_oos_once", flush=True)
    oos_parent, oos_gated = eval_split(bundle, base.OOS_START, base.OOS_END, frozen_artifact, fee, slip, device, frozen_lookback)
    print("stage=fresh_forward_extra_datapoint", flush=True)
    ff_parent, ff_gated = eval_split(bundle, FRESH_FORWARD_START, FRESH_FORWARD_END, frozen_artifact, fee, slip, device, frozen_lookback)

    report = {
        "schema_version": "omega462.tcn_sequence_entry_gate.sol.tuning.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": "omega462_tcn_sequence_entry_gate_sol_tuning_20260722",
        "baseline_untuned": {
            "note": "from tmp/causal_regen_20260516/omega462_tcn_gate_sol_20260722/report.json",
            "val_gated_pnl_pct": 45.28272463357141,
            "val_gated_mdd_pct": -9.339987491212609,
            "val_gated_trades": 41,
            "oos_gated_pnl_pct": 18.19958875155223,
            "oos_gated_mdd_pct": -10.801762630175437,
            "oos_gated_trades": 28,
        },
        "grid_search": {
            "lookbacks": LOOKBACKS,
            "epochs_grid": EPOCHS_GRID,
            "lr_grid": LR_GRID,
            "seed": SEED,
            "n_combos": len(combos),
            "min_val_trades_floor": MIN_VAL_TRADES,
            "results_csv": str(OUT_DIR / "val_grid_results.csv"),
        },
        "split_point_sweep": {
            "candidates": split_candidates,
            "best_lookback_used": best_lookback,
            "results_csv": str(OUT_DIR / "val_split_point_sweep.csv"),
        },
        "frozen_winner": {
            "lookback": frozen_lookback,
            "epochs": frozen_epochs,
            "lr": frozen_lr,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "gate_train_end": frozen_gate_train_end,
            "train_rows": len(train_rows),
            "calib_rows": len(calib_rows),
            "threshold": threshold,
            "artifact_path": frozen_artifact.path,
        },
        "results": {
            "validation": {"start": base.VAL_START, "end_exclusive": base.VAL_END, "parent_alone": val_parent, "parent_plus_tcn_gate": val_gated},
            "oos_canonical": {"start": base.OOS_START, "end_exclusive": base.OOS_END, "parent_alone": oos_parent, "parent_plus_tcn_gate": oos_gated,
                               "note": "touched exactly once, after VAL-only model selection was frozen"},
            "fresh_forward_extra": {"start": FRESH_FORWARD_START, "end_exclusive": FRESH_FORWARD_END, "parent_alone": ff_parent, "parent_plus_tcn_gate": ff_gated,
                                     "note": "additional out-of-sample data point using the same frozen config; not used for selection; canonical OOS above remains primary per repo convention"},
        },
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "total_elapsed_s": time.time() - t0,
    }
    write_json(OUT_DIR / "report.json", report)
    print(json.dumps(report["results"], default=json_default, indent=2), flush=True)


if __name__ == "__main__":
    run()
