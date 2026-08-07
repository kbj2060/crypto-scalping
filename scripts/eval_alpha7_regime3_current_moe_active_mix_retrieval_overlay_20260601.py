#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combo_metrics, _json_default  # noqa: E402
from scripts.eval_alpha7_regime3_current_moe_retrieval_overlay_20260601 import (  # noqa: E402
    RETRIEVAL_2025,
    RETRIEVAL_2026,
    _apply_overlay,
    _overlay_retrieval,
)
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import _load_frames_with_risk  # noqa: E402
from scripts.train_alpha7_regime3_expert_moe_20260601 import _flatten, _score  # noqa: E402


MODEL_ID = "alpha7_regime3_current_moe_active_mix_retrieval_overlay_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_retrieval_overlay_20260601"
BASE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_expert_source_mix_20260601"
VAL_DEC = BASE_DIR / "validation_decisions.csv"
OOS_DEC = BASE_DIR / "oos_2026_decisions.csv"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, risk_overlay = _load_frames_with_risk()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    val_df, val_pos, val_ret = _overlay_retrieval(val_df, RETRIEVAL_2025, tag="validation")
    eval_df, oos_pos, oos_ret = _overlay_retrieval(eval_df, RETRIEVAL_2026, tag="oos_2026")
    val_base = pd.read_csv(VAL_DEC).iloc[val_pos].reset_index(drop=True)
    oos_base = pd.read_csv(OOS_DEC).iloc[oos_pos].reset_index(drop=True)
    baseline_val = _combo_metrics(val_df, val_base)
    baseline_oos = _combo_metrics(eval_df, oos_base)
    rows: list[dict[str, Any]] = [{
        "candidate": "active_mix_baseline",
        "mode": None,
        "edge_thr": None,
        "trade_min": None,
        "edge_mean_min": None,
        "sim_min": None,
        "consensus_min": None,
        "score": float(_score(baseline_val)),
        "validation": baseline_val,
        "oos": baseline_oos,
        "validation_policy_counts": {str(k): int(v) for k, v in val_base["router_expert"].value_counts().to_dict().items()},
        "oos_policy_counts": {str(k): int(v) for k, v in oos_base["router_expert"].value_counts().to_dict().items()},
    }]
    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for mode in ["resize", "veto"]:
        for edge_thr in [0.10, 0.20]:
            for edge_mean_min in [0.0, 0.0025]:
                val_dec = _apply_overlay(val_base, val_df, mode=mode, edge_thr=edge_thr, trade_min=0.65, edge_mean_min=edge_mean_min, sim_min=0.08, consensus_min=0.35)
                oos_dec = _apply_overlay(oos_base, eval_df, mode=mode, edge_thr=edge_thr, trade_min=0.65, edge_mean_min=edge_mean_min, sim_min=0.08, consensus_min=0.35)
                val_costs = _combo_metrics(val_df, val_dec)
                oos_costs = _combo_metrics(eval_df, oos_dec)
                key = f"{mode}_e{edge_thr:.2f}_m{edge_mean_min:.4f}"
                payload[key] = (val_dec, oos_dec)
                rows.append({
                    "candidate": key,
                    "mode": mode,
                    "edge_thr": float(edge_thr),
                    "trade_min": 0.65,
                    "edge_mean_min": float(edge_mean_min),
                    "sim_min": 0.08,
                    "consensus_min": 0.35,
                    "score": float(_score(val_costs)),
                    "validation": val_costs,
                    "oos": oos_costs,
                    "validation_triggered": int(val_dec["retrieval_trigger"].sum()),
                    "oos_triggered": int(oos_dec["retrieval_trigger"].sum()),
                    "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
                    "oos_policy_counts": {str(k): int(v) for k, v in oos_dec["router_expert"].value_counts().to_dict().items()},
                })
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = rows[0]
    if selected["candidate"] == "active_mix_baseline":
        selected_val_dec = val_base.copy()
        selected_oos_dec = oos_base.copy()
    else:
        selected_val_dec, selected_oos_dec = payload[str(selected["candidate"])]
    selected_val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos_dec.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame([
        {
            "candidate": r["candidate"],
            "mode": r["mode"],
            "edge_thr": r["edge_thr"],
            "trade_min": r["trade_min"],
            "edge_mean_min": r["edge_mean_min"],
            "sim_min": r["sim_min"],
            "consensus_min": r["consensus_min"],
            "score": r["score"],
            **_flatten("val", r["validation"]),
            **_flatten("oos", r["oos"]),
            "validation_triggered": r.get("validation_triggered"),
            "oos_triggered": r.get("oos_triggered"),
            "validation_policy_counts": json.dumps(r["validation_policy_counts"], ensure_ascii=False),
            "oos_policy_counts": json.dumps(r["oos_policy_counts"], ensure_ascii=False),
        }
        for r in rows
    ]).to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "RAFT-style retrieval confirmation/veto overlay on top of active Regime3 current-context expert-source mix. The bull/bear/chop expert frame is unchanged.",
        "overlay": {"risk": risk_overlay, "validation_retrieval": val_ret, "oos_retrieval": oos_ret},
        "selected": {
            "candidate": selected["candidate"],
            "mode": selected["mode"],
            "edge_thr": selected["edge_thr"],
            "trade_min": selected["trade_min"],
            "edge_mean_min": selected["edge_mean_min"],
            "sim_min": selected["sim_min"],
            "consensus_min": selected["consensus_min"],
            "validation": selected["validation"],
            "oos": selected["oos"],
            "validation_policy_counts": {str(k): int(v) for k, v in selected_val_dec["router_expert"].value_counts().to_dict().items()},
            "oos_policy_counts": {str(k): int(v) for k, v in selected_oos_dec["router_expert"].value_counts().to_dict().items()},
        },
        "top_grid": rows[:12],
        "artifacts": {
            "report": str(OUT_DIR / "report.json"),
            "ranking": str(OUT_DIR / "ranking.csv"),
            "validation_decisions": str(OUT_DIR / "validation_decisions.csv"),
            "oos_decisions": str(OUT_DIR / "oos_2026_decisions.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": report["selected"]}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
