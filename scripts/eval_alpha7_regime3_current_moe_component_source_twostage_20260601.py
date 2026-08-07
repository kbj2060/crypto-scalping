#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combo_metrics, _json_default  # noqa: E402
from scripts.eval_alpha7_regime3_current_moe_active_component_source_mix_20260601 import (  # noqa: E402
    OUT_DIR as COMPONENT_OUT_DIR,
    ROUTER_NAME,
    _component_decisions,
    _load_model,
    _predict_combo,
    _route_decision,
)
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import _load_frames_with_risk  # noqa: E402
from scripts.train_alpha7_regime3_expert_moe_20260601 import BASE_CLEAN_DIR, _flatten, _route_conf, _route_id, _score  # noqa: E402


MODEL_ID = "alpha7_regime3_current_moe_component_source_twostage_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_component_source_twostage_20260601"
SELECTION_END_TS = pd.Timestamp("2025-12-01")


def _select_columns(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return _flatten(prefix, metrics)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_with_risk()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    val_select_mask = val_df["timestamp"] < SELECTION_END_TS
    val_confirm_mask = ~val_select_mask
    if not bool(val_select_mask.any()) or not bool(val_confirm_mask.any()):
        raise RuntimeError("two-stage validation split is empty")

    primary_base = joblib.load(BASE_CLEAN_DIR / "primary_no_tp/parent.pkl")
    fallback_base = joblib.load(BASE_CLEAN_DIR / "fallback_v2_tp/parent.pkl")
    baseline_val_dec = _predict_combo(primary_base, fallback_base, val_df)
    baseline_oos_dec = _predict_combo(primary_base, fallback_base, eval_df)
    val_route = _route_id(val_df, ROUTER_NAME)
    oos_route = _route_id(eval_df, ROUTER_NAME)
    val_conf = _route_conf(val_df, ROUTER_NAME)
    oos_conf = _route_conf(eval_df, ROUTER_NAME)

    cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source in ["practical", "risk"]:
        for expert in ["bull", "bear", "chop"]:
            for component in ["primary_no_tp", "fallback_v2_tp"]:
                cache[(source, expert, component)] = _load_model(source, expert, component)

    rows: list[dict[str, Any]] = []
    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for bear_primary in ["practical", "risk"]:
        for bear_fallback in ["practical", "risk"]:
            for chop_primary in ["practical", "risk"]:
                for chop_fallback in ["practical", "risk"]:
                    source_map = {
                        "bull": ("practical", "practical"),
                        "bear": (bear_primary, bear_fallback),
                        "chop": (chop_primary, chop_fallback),
                    }
                    val_expert_dec = _component_decisions(source_map, val_df, cache)
                    oos_expert_dec = _component_decisions(source_map, eval_df, cache)
                    val_dec = _route_decision(val_expert_dec, baseline_val_dec, val_route, val_conf)
                    oos_dec = _route_decision(oos_expert_dec, baseline_oos_dec, oos_route, oos_conf)
                    val_select_metrics = _combo_metrics(
                        val_df.loc[val_select_mask].reset_index(drop=True),
                        val_dec.loc[val_select_mask].reset_index(drop=True),
                    )
                    val_confirm_metrics = _combo_metrics(
                        val_df.loc[val_confirm_mask].reset_index(drop=True),
                        val_dec.loc[val_confirm_mask].reset_index(drop=True),
                    )
                    val_full_metrics = _combo_metrics(val_df, val_dec)
                    oos_metrics = _combo_metrics(eval_df, oos_dec)
                    key = f"bearP{bear_primary}_F{bear_fallback}__chopP{chop_primary}_F{chop_fallback}"
                    payload[key] = (val_dec, oos_dec)
                    rows.append({
                        "candidate": key,
                        "bear_primary": bear_primary,
                        "bear_fallback": bear_fallback,
                        "chop_primary": chop_primary,
                        "chop_fallback": chop_fallback,
                        "selection_score": float(_score(val_select_metrics)),
                        "confirm_score": float(_score(val_confirm_metrics)),
                        "full_score": float(_score(val_full_metrics)),
                        "val_select": val_select_metrics,
                        "val_confirm": val_confirm_metrics,
                        "validation": val_full_metrics,
                        "oos": oos_metrics,
                        "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
                        "oos_policy_counts": {str(k): int(v) for k, v in oos_dec["router_expert"].value_counts().to_dict().items()},
                    })

    rows.sort(key=lambda r: float(r["selection_score"]), reverse=True)
    selected = rows[0]
    selected_val_dec, selected_oos_dec = payload[str(selected["candidate"])]
    selected_val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos_dec.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame([
        {
            "candidate": r["candidate"],
            "bear_primary": r["bear_primary"],
            "bear_fallback": r["bear_fallback"],
            "chop_primary": r["chop_primary"],
            "chop_fallback": r["chop_fallback"],
            "selection_score": r["selection_score"],
            "confirm_score": r["confirm_score"],
            "full_score": r["full_score"],
            **_select_columns("sel", r["val_select"]),
            **_select_columns("confirm", r["val_confirm"]),
            **_select_columns("val", r["validation"]),
            **_select_columns("oos", r["oos"]),
            "validation_policy_counts": json.dumps(r["validation_policy_counts"], ensure_ascii=False),
            "oos_policy_counts": json.dumps(r["oos_policy_counts"], ensure_ascii=False),
        }
        for r in rows
    ]).to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Two-stage validation selection for component-level source mix. Selection uses 2025-10/11; 2025-12 is confirmation; 2026 remains fixed OOS evaluation.",
        "base_component_run": str(COMPONENT_OUT_DIR),
        "selection_split": {
            "select": [str(val_df.loc[val_select_mask, "timestamp"].iloc[0]), str(val_df.loc[val_select_mask, "timestamp"].iloc[-1])],
            "confirm": [str(val_df.loc[val_confirm_mask, "timestamp"].iloc[0]), str(val_df.loc[val_confirm_mask, "timestamp"].iloc[-1])],
        },
        "overlay": overlay,
        "selected": {
            "candidate": selected["candidate"],
            "bear_primary": selected["bear_primary"],
            "bear_fallback": selected["bear_fallback"],
            "chop_primary": selected["chop_primary"],
            "chop_fallback": selected["chop_fallback"],
            "selection_score": selected["selection_score"],
            "confirm_score": selected["confirm_score"],
            "full_score": selected["full_score"],
            "val_select": selected["val_select"],
            "val_confirm": selected["val_confirm"],
            "validation": selected["validation"],
            "oos": selected["oos"],
            "validation_policy_counts": selected["validation_policy_counts"],
            "oos_policy_counts": selected["oos_policy_counts"],
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
