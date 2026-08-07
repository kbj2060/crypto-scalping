#!/usr/bin/env python3
"""Candidate-count sweep for the LIVE SOL adaptive_squeeze parent at lower entry
quality-threshold values, to pick a threshold for the TCN sequence-entry-gate
sample-size fix (see docs task: loosen threshold instead of extending calendar window,
because SOL's regime3 HMM / CryptoMamba sidecars are fit on 2024 data and any
2024-timestamp candidates would leak in-sample fitting artifacts).

Reuses lp.prepare_frame_live() verbatim per threshold (monkeypatches lp.QUALITY_THRESHOLD
before calling it -- the function resolves QUALITY_THRESHOLD as a module-global at call
time, so this is equivalent to running the live-parent bundle inference at that threshold).
Counts parent-alone "trades" (== candidates the TCN gate would see) in TRAIN
(2025-01-01..2025-09-01), VAL (2025-09-01..2025-12-31), OOS (2026-01-01..2026-03-31), and
FRESH (2026-04-01..2026-07-21 12:00) via base.slice_bundle + base.replay_with_gate
(gate_artifact=None), identical harness to the gate-training scripts.

Read-only w.r.t. all existing tmp/causal_regen_20260516/ dirs except this script's own
output dir tmp/causal_regen_20260516/omega462_tcn_gate_sol_loosethreshold_20260722/.
No live wiring, research only.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

import scripts.train_eval_omega462_tcn_gate_sol_20260722 as base  # noqa: E402
import scripts.train_eval_omega462_tcn_gate_sol_liveparent_20260722 as lp  # noqa: E402

OUT_DIR = base.ROOT / "tmp/causal_regen_20260516/omega462_tcn_gate_sol_loosethreshold_20260722"

TRAIN_START, TRAIN_END = base.TRAIN_START, base.TRAIN_END
THRESHOLDS = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=base.json_default) + "\n", encoding="utf-8")


def count_split(bundle, start, end, fee, slip, device) -> dict[str, Any]:
    sl = base.slice_bundle(bundle, start, end)
    metrics, _, _ = base.replay_with_gate(
        frame=sl["frame"], base_x=sl["base_x"], dec=sl["dec_atr"], loaded=bundle["loaded"],
        margin=sl["margin"], leverage=sl["leverage"], static_tape=sl["static_tape"],
        fee=fee, slip=slip, device=device, gate_artifact=None, collect_labels=False,
    )
    return {"trades": metrics["trades"], "pnl_pct": metrics["pnl_pct"], "mdd_pct": metrics["mdd_pct"], "wr": metrics["wr"]}


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    t0 = time.time()
    rows = []
    for thr in THRESHOLDS:
        t_s = time.time()
        lp.QUALITY_THRESHOLD = thr
        print(f"stage=prepare_frame_live threshold={thr}", flush=True)
        bundle = lp.prepare_frame_live(device)
        fee, slip = bundle["fee_slip"]
        row = {"threshold": thr}
        for split, start, end in (
            ("train", TRAIN_START, TRAIN_END),
            ("validation", base.VAL_START, base.VAL_END),
            ("oos", base.OOS_START, base.OOS_END),
            ("fresh", lp.FRESH_START, lp.FRESH_END),
        ):
            r = count_split(bundle, start, end, fee, slip, device)
            row[f"{split}_trades"] = r["trades"]
            row[f"{split}_pnl_pct"] = r["pnl_pct"]
            row[f"{split}_mdd_pct"] = r["mdd_pct"]
            row[f"{split}_wr"] = r["wr"]
        row["total_trades"] = row["train_trades"] + row["validation_trades"] + row["oos_trades"] + row["fresh_trades"]
        row["elapsed_s"] = time.time() - t_s
        rows.append(row)
        print(json.dumps(row, default=base.json_default), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "candidate_count_sweep.csv", index=False)
    report = {
        "schema_version": "omega462.tcn_sequence_entry_gate.sol.loosethreshold_candidate_count.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Count parent-alone candidate trades per split at multiple SOL adaptive_squeeze "
            "quality thresholds, to pick a threshold that yields ~2-4x the current q0.70 "
            "candidate count (train~74/calib~32 within TRAIN window before the internal "
            "gate_train_end split, total across train/val/oos/fresh ~150) without diluting "
            "quality too much."
        ),
        "thresholds": THRESHOLDS,
        "results_csv": str(OUT_DIR / "candidate_count_sweep.csv"),
        "windows": {
            "train": [TRAIN_START, TRAIN_END],
            "validation": [base.VAL_START, base.VAL_END],
            "oos": [base.OOS_START, base.OOS_END],
            "fresh": [lp.FRESH_START, lp.FRESH_END],
        },
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "total_elapsed_s": time.time() - t0,
    }
    write_json(OUT_DIR / "candidate_count_sweep_report.json", report)
    return report


if __name__ == "__main__":
    run()
