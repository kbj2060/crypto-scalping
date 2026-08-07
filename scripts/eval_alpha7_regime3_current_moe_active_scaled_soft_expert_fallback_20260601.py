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

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _combine_primary_fallback,
    _combo_metrics,
    _json_default,
    _predict_scaled,
)
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import _load_frames_with_risk  # noqa: E402
from scripts.train_alpha7_regime3_expert_moe_20260601 import (  # noqa: E402
    BASE_CLEAN_DIR,
    OUT_DIR as PRACTICAL_EXPERT_DIR,
    _active,
    _flatten,
    _route_conf,
    _route_id,
    _score,
    _side_constrained,
)


MODEL_ID = "alpha7_regime3_current_moe_active_scaled_soft_expert_fallback_20260601"
ROUTER_NAME = "regime3_current_context"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_scaled_soft_expert_fallback_20260601"
VARIANT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_feature_variants_20260601"
SOURCE_BY_EXPERT = {
    "bull": PRACTICAL_EXPERT_DIR / ROUTER_NAME / "bull",
    "bear": VARIANT_DIR / "base_plus_current_risk/bear",
    "chop": PRACTICAL_EXPERT_DIR / ROUTER_NAME / "chop",
}
EXPERT_SCALE = {"bull": 0.85, "bear": 1.15, "chop": 1.25}
BASE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601"
VAL_DEC = BASE_DIR / "validation_decisions.csv"
OOS_DEC = BASE_DIR / "oos_2026_decisions.csv"


def _predict_combo(primary: dict[str, Any], fallback: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    return _combine_primary_fallback(_predict_scaled(primary, df, None), _predict_scaled(fallback, df, None)).reset_index(drop=True)


def _load_pair(expert: str) -> dict[str, Any]:
    root = SOURCE_BY_EXPERT[expert]
    p = root / "primary_no_tp/parent.pkl"
    f = root / "fallback_v2_tp/parent.pkl"
    if not p.exists() or not f.exists():
        raise FileNotFoundError(f"missing {expert} artifacts: {p}, {f}")
    return {"primary": joblib.load(p), "fallback": joblib.load(f)}


def _scale_expert_dec(dec: pd.DataFrame, expert: str) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    scale = float(EXPERT_SCALE[expert])
    out.loc[active, "notional_exposure"] = pd.to_numeric(out.loc[active, "notional_exposure"], errors="raise") * scale
    out.loc[active, "position_fraction"] = pd.to_numeric(out.loc[active, "position_fraction"], errors="raise") * scale
    return out


def _apply_soft_fallback(
    active_dec: pd.DataFrame,
    expert_dec: dict[str, pd.DataFrame],
    route: np.ndarray,
    conf: np.ndarray,
    *,
    floor: float,
    scale: float,
) -> pd.DataFrame:
    out = active_dec.copy().reset_index(drop=True)
    decision_cols = [c for c in active_dec.columns if c in expert_dec["bull"].columns]
    current_lowconf = out["router_expert"].astype(str).eq("lowconf_baseline")
    take = current_lowconf & (conf >= float(floor)) & (conf < 0.80)
    for idx, expert in enumerate(["bull", "bear", "chop"]):
        mask = take & (route == idx)
        out.loc[mask, decision_cols] = expert_dec[expert].loc[mask, decision_cols].to_numpy()
        out.loc[mask, "router_expert"] = f"soft_{expert}"
        active = mask & _active(out)
        out.loc[active, "notional_exposure"] = pd.to_numeric(out.loc[active, "notional_exposure"], errors="raise") * float(scale)
        out.loc[active, "position_fraction"] = pd.to_numeric(out.loc[active, "position_fraction"], errors="raise") * float(scale)
    out["soft_expert_floor"] = float(floor)
    out["soft_expert_scale"] = float(scale)
    out["soft_expert_trigger"] = take
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_with_risk()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    val_active = pd.read_csv(VAL_DEC).reset_index(drop=True)
    oos_active = pd.read_csv(OOS_DEC).reset_index(drop=True)
    if len(val_df) != len(val_active) or len(eval_df) != len(oos_active):
        raise RuntimeError(f"frame/decision mismatch: val {len(val_df)} {len(val_active)} oos {len(eval_df)} {len(oos_active)}")

    # Baseline parent is loaded only to make feature-contract failures obvious
    # before expert predictions are created.
    _ = joblib.load(BASE_CLEAN_DIR / "primary_no_tp/parent.pkl")
    models = {expert: _load_pair(expert) for expert in SOURCE_BY_EXPERT}
    val_expert_dec = {
        expert: _scale_expert_dec(_side_constrained(_predict_combo(models[expert]["primary"], models[expert]["fallback"], val_df), expert=expert), expert)
        for expert in SOURCE_BY_EXPERT
    }
    oos_expert_dec = {
        expert: _scale_expert_dec(_side_constrained(_predict_combo(models[expert]["primary"], models[expert]["fallback"], eval_df), expert=expert), expert)
        for expert in SOURCE_BY_EXPERT
    }
    val_route = _route_id(val_df, ROUTER_NAME)
    oos_route = _route_id(eval_df, ROUTER_NAME)
    val_conf = _route_conf(val_df, ROUTER_NAME)
    oos_conf = _route_conf(eval_df, ROUTER_NAME)

    rows: list[dict[str, Any]] = []
    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for floor in [0.65, 0.70, 0.75]:
        for scale in [0.25, 0.50, 0.70]:
            val_dec = _apply_soft_fallback(val_active, val_expert_dec, val_route, val_conf, floor=floor, scale=scale)
            oos_dec = _apply_soft_fallback(oos_active, oos_expert_dec, oos_route, oos_conf, floor=floor, scale=scale)
            val_costs = _combo_metrics(val_df, val_dec)
            oos_costs = _combo_metrics(eval_df, oos_dec)
            key = f"floor{floor:.2f}_scale{scale:.2f}"
            payload[key] = (val_dec, oos_dec)
            rows.append({
                "candidate": key,
                "floor": float(floor),
                "scale": float(scale),
                "score": float(_score(val_costs)),
                "validation": val_costs,
                "oos": oos_costs,
                "validation_triggered": int(val_dec["soft_expert_trigger"].sum()),
                "oos_triggered": int(oos_dec["soft_expert_trigger"].sum()),
                "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
                "oos_policy_counts": {str(k): int(v) for k, v in oos_dec["router_expert"].value_counts().to_dict().items()},
            })
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = rows[0]
    selected_val_dec, selected_oos_dec = payload[str(selected["candidate"])]
    selected_val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos_dec.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame([
        {
            "candidate": r["candidate"],
            "floor": r["floor"],
            "scale": r["scale"],
            "score": r["score"],
            **_flatten("val", r["validation"]),
            **_flatten("oos", r["oos"]),
            "validation_triggered": r["validation_triggered"],
            "oos_triggered": r["oos_triggered"],
            "validation_policy_counts": json.dumps(r["validation_policy_counts"], ensure_ascii=False),
            "oos_policy_counts": json.dumps(r["oos_policy_counts"], ensure_ascii=False),
        }
        for r in rows
    ]).to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Soft expert fallback for low-confidence rows on top of active scaled current-Regime3 MoE. Rows with route confidence in [floor, 0.80) may use the routed expert at reduced scale; lower confidence remains baseline.",
        "overlay": overlay,
        "selected": {
            "candidate": selected["candidate"],
            "floor": selected["floor"],
            "scale": selected["scale"],
            "validation": selected["validation"],
            "oos": selected["oos"],
            "validation_triggered": selected["validation_triggered"],
            "oos_triggered": selected["oos_triggered"],
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
