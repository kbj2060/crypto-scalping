#!/usr/bin/env python3
"""One-time OOS replay for the validation-selected SOL architecture-v2."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707 as sol  # noqa: E402
import train_eval_sol_architecture_v2_pipeline_20260719 as pipeline  # noqa: E402


def main() -> int:
    out_dir = pipeline.OUT_DIR
    with (out_dir / "pipeline.pkl").open("rb") as handle:
        artifact = pickle.load(handle)
    payload = torch.load(artifact["entry_artifact"], map_location="cpu", weights_only=False)
    frames = sol._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=pipeline.LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    oos = frames["oos_raw"].copy()
    prior = pd.concat([frames["train_raw"], frames["val_raw"]], ignore_index=True)
    combined = pd.concat([prior, oos], ignore_index=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    direction = pipeline._predict_tcn(payload, combined, len(prior), device)
    base_cols = list(artifact["base_cols"])
    quality = np.column_stack([artifact["quality_models"][side].predict_proba(oos[base_cols])[:, 1] for side in ("long", "short")])
    ret_pred = np.column_stack([artifact["outcome_models"][f"{side}_return"].predict(oos[base_cols]) for side in ("long", "short")])
    adverse_pred = np.column_stack([artifact["outcome_models"][f"{side}_adverse"].predict(oos[base_cols]) for side in ("long", "short")])
    cfg = pipeline.ReplayCfg(**artifact["selected_config"])
    metrics, ledger = pipeline._replay(
        oos,
        direction,
        quality,
        ret_pred,
        adverse_pred,
        cfg,
        exit_model=artifact["exit_model"] if cfg.exit_threshold is not None else None,
        exit_columns=artifact["exit_columns"],
        base_cols=base_cols,
    )
    ledger.to_csv(out_dir / "oos_selected_ledger.csv", index=False)
    report = {
        "selection_frozen_from": str(out_dir / "report.json"),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "oos_range": [str(oos.timestamp.iloc[0]), str(oos.timestamp.iloc[-1])],
        "selected_config": artifact["selected_config"],
        "oos": metrics,
    }
    (out_dir / "oos_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
