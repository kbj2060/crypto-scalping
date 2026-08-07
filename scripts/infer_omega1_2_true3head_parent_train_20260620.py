#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega1_2_true_3head_parent_train_inference_20260620"
PARENT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SPLIT_TS = pd.Timestamp("2025-10-01")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = PARENT_DIR / "true_3head_tabm_bundle.pt"
    if not bundle_path.exists():
        raise RuntimeError(f"missing parent bundle: {bundle_path}")
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    base_cols = list(bundle["base_cols"])
    models = dict(bundle["models"])

    frames = threehead._prepare_frames(disable_tp_sl=False)
    train_raw = frames["train_raw"].copy().reset_index(drop=True)
    train_raw["timestamp"] = pd.to_datetime(train_raw["timestamp"])
    train_raw = train_raw[train_raw["timestamp"] < SPLIT_TS].reset_index(drop=True)

    x = threehead._base_input(train_raw, base_cols)
    preds = {expert: threehead._predict_payload(models[expert], x, device=torch.device("cpu")) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(train_raw)
    direction = threehead._routed(preds, route, "direction", 3)
    quality = threehead._routed(preds, route, "quality", 3)
    out = threehead._prediction_output(
        train_raw,
        direction,
        quality,
        threshold=0.45,
        prefix="omega1_regime3_expertdq_train",
    )
    out_path = OUT_DIR / "train_predictions_2025_jan_sep_true3head_in_sample.csv"
    out.to_csv(out_path, index=False)
    report = {
        "model_id": MODEL_ID,
        "source_parent_dir": str(PARENT_DIR),
        "source_bundle": str(bundle_path),
        "prediction_scope": "2025-01-01 <= timestamp < 2025-10-01",
        "prediction_type": "in_sample_parent_inference_not_oof",
        "rows": int(len(out)),
        "columns": list(out.columns),
        "quality_threshold_for_final_action_column": 0.45,
        "artifacts": {
            "train_parent_predictions": str(out_path.relative_to(ROOT)),
        },
        "redteam_notes": [
            "This file is not OOF. It is generated from the already-trained parent bundle on its training period.",
            "Use only for research feature ablation; promotion-grade training needs walk-forward/OOF parent predictions.",
        ],
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
