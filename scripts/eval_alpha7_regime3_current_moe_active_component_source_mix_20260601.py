#!/usr/bin/env python3
from __future__ import annotations

import itertools
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
    _flatten,
    _route_conf,
    _route_id,
    _score,
    _side_constrained,
)


MODEL_ID = "alpha7_regime3_current_moe_active_component_source_mix_20260601"
ROUTER_NAME = "regime3_current_context"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_component_source_mix_20260601"
VARIANT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_feature_variants_20260601"
SOURCES = {
    "practical": PRACTICAL_EXPERT_DIR / ROUTER_NAME,
    "risk": VARIANT_DIR / "base_plus_current_risk",
}
EXPERT_SCALE = {"bull": 0.85, "bear": 1.15, "chop": 1.25}


def _predict_combo(primary: dict[str, Any], fallback: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    return _combine_primary_fallback(_predict_scaled(primary, df, None), _predict_scaled(fallback, df, None)).reset_index(drop=True)


def _load_model(source: str, expert: str, component: str) -> dict[str, Any]:
    path = SOURCES[source] / expert / component / "parent.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def _scale_expert_dec(dec: pd.DataFrame, expert: str) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    action = pd.to_numeric(out["action"], errors="raise").to_numpy(dtype=np.int64)
    side = pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.int64)
    active = (action != 0) & (side != 0)
    scale = float(EXPERT_SCALE[expert])
    out.loc[active, "notional_exposure"] = pd.to_numeric(out.loc[active, "notional_exposure"], errors="raise") * scale
    out.loc[active, "position_fraction"] = pd.to_numeric(out.loc[active, "position_fraction"], errors="raise") * scale
    return out


def _component_decisions(
    source_map: dict[str, tuple[str, str]],
    df: pd.DataFrame,
    cache: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for expert, (primary_src, fallback_src) in source_map.items():
        primary = cache[(primary_src, expert, "primary_no_tp")]
        fallback = cache[(fallback_src, expert, "fallback_v2_tp")]
        dec = _side_constrained(_predict_combo(primary, fallback, df), expert=expert)
        out[expert] = _scale_expert_dec(dec, expert)
    return out


def _route_decision(
    expert_dec: dict[str, pd.DataFrame],
    base_dec: pd.DataFrame,
    route: np.ndarray,
    conf: np.ndarray,
) -> pd.DataFrame:
    out = base_dec.copy().reset_index(drop=True)
    decision_cols = list(base_dec.columns)
    selected = route.copy()
    selected[conf < 0.80] = 3
    for idx, expert in enumerate(["bull", "bear", "chop"]):
        mask = selected == idx
        out.loc[mask, decision_cols] = expert_dec[expert].loc[mask, decision_cols].to_numpy()
    out["router_expert"] = np.where(selected == 0, "bull", np.where(selected == 1, "bear", np.where(selected == 2, "chop_expert", "lowconf_baseline")))
    out["router_confidence"] = conf
    out["router_min_conf"] = 0.80
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_with_risk()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    primary_base = joblib.load(BASE_CLEAN_DIR / "primary_no_tp/parent.pkl")
    fallback_base = joblib.load(BASE_CLEAN_DIR / "fallback_v2_tp/parent.pkl")
    baseline_val_dec = _predict_combo(primary_base, fallback_base, val_df)
    baseline_oos_dec = _predict_combo(primary_base, fallback_base, eval_df)
    val_route = _route_id(val_df, ROUTER_NAME)
    oos_route = _route_id(eval_df, ROUTER_NAME)
    val_conf = _route_conf(val_df, ROUTER_NAME)
    oos_conf = _route_conf(eval_df, ROUTER_NAME)

    cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source in SOURCES:
        for expert in ["bull", "bear", "chop"]:
            for component in ["primary_no_tp", "fallback_v2_tp"]:
                cache[(source, expert, component)] = _load_model(source, expert, component)

    rows: list[dict[str, Any]] = []
    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    # Bull remains practical/practical because previous source-mix sweeps showed
    # risk/current bull variants degrading validation. Decompose bear/chop only.
    for bear_primary, bear_fallback, chop_primary, chop_fallback in itertools.product(
        ["practical", "risk"],
        ["practical", "risk"],
        ["practical", "risk"],
        ["practical", "risk"],
    ):
        source_map = {
            "bull": ("practical", "practical"),
            "bear": (bear_primary, bear_fallback),
            "chop": (chop_primary, chop_fallback),
        }
        val_expert_dec = _component_decisions(source_map, val_df, cache)
        oos_expert_dec = _component_decisions(source_map, eval_df, cache)
        val_dec = _route_decision(val_expert_dec, baseline_val_dec, val_route, val_conf)
        oos_dec = _route_decision(oos_expert_dec, baseline_oos_dec, oos_route, oos_conf)
        val_costs = _combo_metrics(val_df, val_dec)
        oos_costs = _combo_metrics(eval_df, oos_dec)
        key = f"bearP{bear_primary}_F{bear_fallback}__chopP{chop_primary}_F{chop_fallback}"
        payload[key] = (val_dec, oos_dec)
        rows.append({
            "candidate": key,
            "bear_primary": bear_primary,
            "bear_fallback": bear_fallback,
            "chop_primary": chop_primary,
            "chop_fallback": chop_fallback,
            "score": float(_score(val_costs)),
            "validation": val_costs,
            "oos": oos_costs,
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
            "bear_primary": r["bear_primary"],
            "bear_fallback": r["bear_fallback"],
            "chop_primary": r["chop_primary"],
            "chop_fallback": r["chop_fallback"],
            "score": r["score"],
            **_flatten("val", r["validation"]),
            **_flatten("oos", r["oos"]),
            "validation_policy_counts": json.dumps(r["validation_policy_counts"], ensure_ascii=False),
            "oos_policy_counts": json.dumps(r["oos_policy_counts"], ensure_ascii=False),
        }
        for r in rows
    ]).to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Component-level primary/fallback source mix inside fixed current-Regime3 MoE. Bull stays practical; bear/chop primary and fallback can independently use practical or current+risk source. Expert scales remain bull=0.85, bear=1.15, chop=1.25.",
        "overlay": overlay,
        "selected": {
            "candidate": selected["candidate"],
            "bear_primary": selected["bear_primary"],
            "bear_fallback": selected["bear_fallback"],
            "chop_primary": selected["chop_primary"],
            "chop_fallback": selected["chop_fallback"],
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
