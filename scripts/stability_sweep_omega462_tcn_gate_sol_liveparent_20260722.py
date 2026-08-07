#!/usr/bin/env python3
"""Stability sweep for the frozen SOL TCN sequence-entry-gate config selected in
scripts/train_eval_omega462_tcn_gate_sol_liveparent_retrain_20260722.py
(lookback=144, epochs=24, lr=0.0024, gate_train_end=2025-06-15, seed=260722).

That single-seed / single-split-point result looked strong (fresh window flipped from
parent-alone -14.53% to parent+gate +6.77%), but this project has repeated history of
single-config results turning out to be flukes or leakage artifacts. This script:

1. Seed variance: retrains the SAME frozen config with 5 different seeds on the live-parent
   candidate stream, applying "select on VAL, touch OOS/fresh once" per seed (no cherry-picking
   across seeds -- every seed's OOS/fresh number is reported, not just the best).
2. Split-point variance: with a representative seed, retrains at gate_train_end in
   {2025-05-15, 2025-06-15 (baseline), 2025-07-01}.

Reuses (imports, does not duplicate):
  - scripts.train_eval_omega462_tcn_gate_sol_liveparent_retrain_20260722 (`retrain`):
    label_rows_for_lookback, fit_gate (now seed-parameterized), eval_split, GATE_TRAIN_END,
    BATCH_SIZE constant.
  - scripts.train_eval_omega462_tcn_gate_sol_liveparent_20260722 (`lp`): prepare_frame_live,
    FRESH_START/FRESH_END.
  - scripts.train_eval_omega462_tcn_gate_sol_20260722 (`base`): VAL_START/VAL_END/
    OOS_START/OOS_END, json_default.

Frozen config under test: lookback=144, epochs=24, lr=0.0024, batch_size=128.

Fresh-forward contract: fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false (inherited from
replay_with_gate, unmodified).

Read-only w.r.t. all existing artifact dirs under tmp/causal_regen_20260516/ except this
script's own new output dir
tmp/causal_regen_20260516/omega462_tcn_gate_sol_liveparent_stability_20260722/.
No live wiring. Research only.
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
import scripts.train_eval_omega462_tcn_gate_sol_liveparent_retrain_20260722 as retrain  # noqa: E402

OUT_DIR = base.ROOT / "tmp/causal_regen_20260516/omega462_tcn_gate_sol_liveparent_stability_20260722"

LOOKBACK = 144
EPOCHS = 24
LR = 2.4e-3
BASELINE_GATE_TRAIN_END = "2025-06-15 00:00:00"

SEEDS = [1, 2, 3, 4, 5]
SPLIT_POINTS = ["2025-05-15 00:00:00", "2025-06-15 00:00:00", "2025-07-01 00:00:00"]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=base.json_default) + "\n", encoding="utf-8")


def run_one(bundle, label_rows, fee, slip, device, seed, gate_train_end, tag) -> dict[str, Any]:
    name = f"sol_tcn_seq_gate_liveparent_stability_{tag}_L{LOOKBACK}_ep{EPOCHS}_lr{LR:g}_seed{seed}_20260722"
    artifact, fit_info = retrain.fit_gate(
        bundle, label_rows, LOOKBACK, EPOCHS, LR, gate_train_end, name=name, seed=seed,
    )
    val_parent, val_gated, _, _ = retrain.eval_split(bundle, base.VAL_START, base.VAL_END, artifact, fee, slip, device, LOOKBACK)
    oos_parent, oos_gated, _, _ = retrain.eval_split(bundle, base.OOS_START, base.OOS_END, artifact, fee, slip, device, LOOKBACK)
    ff_parent, ff_gated, _, _ = retrain.eval_split(bundle, lp.FRESH_START, lp.FRESH_END, artifact, fee, slip, device, LOOKBACK)
    return {
        "tag": tag,
        "seed": seed,
        "gate_train_end": gate_train_end,
        "train_rows": fit_info["train_rows"],
        "calib_rows": fit_info["calib_rows"],
        "threshold": fit_info["threshold"],
        "val_parent_pnl_pct": val_parent["pnl_pct"], "val_parent_mdd_pct": val_parent["mdd_pct"], "val_parent_trades": val_parent["trades"],
        "val_gated_pnl_pct": val_gated["pnl_pct"], "val_gated_mdd_pct": val_gated["mdd_pct"], "val_gated_trades": val_gated["trades"], "val_gated_wr": val_gated["wr"],
        "oos_parent_pnl_pct": oos_parent["pnl_pct"], "oos_parent_mdd_pct": oos_parent["mdd_pct"], "oos_parent_trades": oos_parent["trades"],
        "oos_gated_pnl_pct": oos_gated["pnl_pct"], "oos_gated_mdd_pct": oos_gated["mdd_pct"], "oos_gated_trades": oos_gated["trades"], "oos_gated_wr": oos_gated["wr"],
        "fresh_parent_pnl_pct": ff_parent["pnl_pct"], "fresh_parent_mdd_pct": ff_parent["mdd_pct"], "fresh_parent_trades": ff_parent["trades"],
        "fresh_gated_pnl_pct": ff_gated["pnl_pct"], "fresh_gated_mdd_pct": ff_gated["mdd_pct"], "fresh_gated_trades": ff_gated["trades"], "fresh_gated_wr": ff_gated["wr"],
        "fresh_flip_neg_to_pos": bool(ff_parent["pnl_pct"] < 0 and ff_gated["pnl_pct"] > 0),
    }


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    t0 = time.time()

    print("stage=prepare_frame_live", flush=True)
    bundle = lp.prepare_frame_live(device)
    fee, slip = bundle["fee_slip"]
    print(f"stage=prepare_frame_live_done elapsed={time.time() - t0:.1f}s", flush=True)

    print("stage=collect_labels_liveparent lookback=144 (cached across all seeds/splits)", flush=True)
    label_rows = retrain.label_rows_for_lookback(bundle, LOOKBACK, fee, slip, device)
    print(f"stage=collect_labels_done n_rows={len(label_rows)}", flush=True)

    seed_results = []
    for seed in SEEDS:
        t_s = time.time()
        row = run_one(bundle, label_rows, fee, slip, device, seed, BASELINE_GATE_TRAIN_END, tag="seed_sweep")
        row["elapsed_s"] = time.time() - t_s
        seed_results.append(row)
        print(json.dumps(row, default=base.json_default), flush=True)

    seed_df = pd.DataFrame(seed_results)
    seed_df.to_csv(OUT_DIR / "seed_sweep_results.csv", index=False)

    # pick the median seed by VAL gated PnL as the representative seed for the split-point sweep
    seed_df_sorted = seed_df.sort_values("val_gated_pnl_pct").reset_index(drop=True)
    median_idx = len(seed_df_sorted) // 2
    representative_seed = int(seed_df_sorted.loc[median_idx, "seed"])
    print(f"stage=representative_seed_selected seed={representative_seed}", flush=True)

    split_results = []
    for gate_train_end in SPLIT_POINTS:
        t_s = time.time()
        row = run_one(bundle, label_rows, fee, slip, device, representative_seed, gate_train_end, tag="split_sweep")
        row["elapsed_s"] = time.time() - t_s
        split_results.append(row)
        print(json.dumps(row, default=base.json_default), flush=True)

    split_df = pd.DataFrame(split_results)
    split_df.to_csv(OUT_DIR / "split_point_sweep_results.csv", index=False)

    def summarize(col: str, df: pd.DataFrame) -> dict[str, float]:
        s = df[col]
        return {"mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()), "max": float(s.max())}

    seed_summary = {
        "val_gated_pnl_pct": summarize("val_gated_pnl_pct", seed_df),
        "val_gated_mdd_pct": summarize("val_gated_mdd_pct", seed_df),
        "oos_gated_pnl_pct": summarize("oos_gated_pnl_pct", seed_df),
        "oos_gated_mdd_pct": summarize("oos_gated_mdd_pct", seed_df),
        "fresh_gated_pnl_pct": summarize("fresh_gated_pnl_pct", seed_df),
        "fresh_gated_mdd_pct": summarize("fresh_gated_mdd_pct", seed_df),
        "n_seeds_fresh_flip_neg_to_pos": int(seed_df["fresh_flip_neg_to_pos"].sum()),
        "n_seeds_total": len(seed_df),
    }

    report = {
        "schema_version": "omega462.tcn_sequence_entry_gate.sol.liveparent_stability.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": "omega462_tcn_sequence_entry_gate_sol_liveparent_stability_20260722",
        "purpose": (
            "Stability sweep (seed variance + gate_train_end split-point variance) for the frozen "
            "config (lookback=144, epochs=24, lr=0.0024, gate_train_end=2025-06-15) selected in "
            "train_eval_omega462_tcn_gate_sol_liveparent_retrain_20260722.py on a single seed=260722. "
            "Tests whether the fresh-window flip (parent-alone -14.53% -> parent+gate +6.77%) is "
            "robust or a single-seed / single-split-point artifact."
        ),
        "frozen_config_under_test": {"lookback": LOOKBACK, "epochs": EPOCHS, "lr": LR, "batch_size": retrain.BATCH_SIZE},
        "original_single_seed_result": {
            "seed": 260722, "gate_train_end": BASELINE_GATE_TRAIN_END,
            "val_gated_pnl_pct": 151.2337556431519, "oos_gated_pnl_pct": 17.22747664631532,
            "fresh_parent_pnl_pct": -14.530310505335908, "fresh_gated_pnl_pct": 6.769682020268886,
            "source": "tmp/causal_regen_20260516/omega462_tcn_gate_sol_liveparent_retrain_20260722/report.json",
        },
        "seed_sweep": {
            "seeds": SEEDS, "gate_train_end": BASELINE_GATE_TRAIN_END,
            "results_csv": str(OUT_DIR / "seed_sweep_results.csv"),
            "summary": seed_summary,
        },
        "representative_seed_for_split_sweep": representative_seed,
        "split_point_sweep": {
            "gate_train_end_points": SPLIT_POINTS, "seed": representative_seed,
            "results_csv": str(OUT_DIR / "split_point_sweep_results.csv"),
        },
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


if __name__ == "__main__":
    report = run()
    print(json.dumps(report["seed_sweep"]["summary"], ensure_ascii=False, indent=2, default=base.json_default), flush=True)
