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
from scripts.train_alpha7_regime3_expert_moe_20260601 import (  # noqa: E402
    BASE_CLEAN_DIR,
    EXPERT_NAMES,
    OUT_DIR as TRAINED_DIR,
    _active,
    _flatten,
    _load_router_frames,
    _route_conf,
    _route_id,
    _score,
    _side_constrained,
)


MODEL_ID = "alpha7_regime3_current_moe_per_expert_conf_20260601"
ROUTER_NAME = "regime3_current_context"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_per_expert_conf_20260601"


def _predict_combo(primary: dict[str, Any], fallback: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    return _combine_primary_fallback(_predict_scaled(primary, df, None), _predict_scaled(fallback, df, None)).reset_index(drop=True)


def _load_expert_models() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for expert in EXPERT_NAMES:
        p = TRAINED_DIR / ROUTER_NAME / expert / "primary_no_tp/parent.pkl"
        f = TRAINED_DIR / ROUTER_NAME / expert / "fallback_v2_tp/parent.pkl"
        if not p.exists() or not f.exists():
            raise FileNotFoundError(f"missing trained expert artifacts for {expert}: {p}, {f}")
        out[expert] = {"primary": joblib.load(p), "fallback": joblib.load(f)}
    return out


def _build_expert_decisions(experts: dict[str, dict[str, Any]], val_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    val: dict[str, pd.DataFrame] = {}
    oos: dict[str, pd.DataFrame] = {}
    for expert, models in experts.items():
        val[expert] = _side_constrained(_predict_combo(models["primary"], models["fallback"], val_df), expert=expert)
        oos[expert] = _side_constrained(_predict_combo(models["primary"], models["fallback"], eval_df), expert=expert)
    return val, oos


def _route_decision(
    expert_dec: dict[str, pd.DataFrame],
    base_dec: pd.DataFrame,
    route: np.ndarray,
    conf: np.ndarray,
    *,
    bull_thr: float,
    bear_thr: float,
    chop_thr: float,
) -> pd.DataFrame:
    out = base_dec.copy().reset_index(drop=True)
    decision_cols = list(base_dec.columns)
    out.loc[:, decision_cols] = base_dec.loc[:, decision_cols].to_numpy()
    thresholds = np.array([bull_thr, bear_thr, chop_thr], dtype=np.float64)
    selected = route.copy()
    selected[conf < thresholds[np.clip(route, 0, 2)]] = 3
    for idx, expert in enumerate(EXPERT_NAMES):
        mask = selected == idx
        out.loc[mask, decision_cols] = expert_dec[expert].loc[mask, decision_cols].to_numpy()
    out["router_expert"] = np.where(selected == 0, "bull", np.where(selected == 1, "bear", np.where(selected == 2, "chop_expert", "lowconf_baseline")))
    out["router_confidence"] = conf
    out["router_bull_thr"] = float(bull_thr)
    out["router_bear_thr"] = float(bear_thr)
    out["router_chop_thr"] = float(chop_thr)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_router_frames(ROUTER_NAME)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    primary_base = joblib.load(BASE_CLEAN_DIR / "primary_no_tp/parent.pkl")
    fallback_base = joblib.load(BASE_CLEAN_DIR / "fallback_v2_tp/parent.pkl")
    baseline_val_dec = _predict_combo(primary_base, fallback_base, val_df)
    baseline_oos_dec = _predict_combo(primary_base, fallback_base, eval_df)
    experts = _load_expert_models()
    val_expert_dec, oos_expert_dec = _build_expert_decisions(experts, val_df, eval_df)
    val_route = _route_id(val_df, ROUTER_NAME)
    oos_route = _route_id(eval_df, ROUTER_NAME)
    val_conf = _route_conf(val_df, ROUTER_NAME)
    oos_conf = _route_conf(eval_df, ROUTER_NAME)
    rows: list[dict[str, Any]] = []
    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for bull_thr in [0.75, 0.80, 0.85]:
        for bear_thr in [0.75, 0.80, 0.85]:
            for chop_thr in [0.75, 0.80, 0.85]:
                val_dec = _route_decision(val_expert_dec, baseline_val_dec, val_route, val_conf, bull_thr=bull_thr, bear_thr=bear_thr, chop_thr=chop_thr)
                oos_dec = _route_decision(oos_expert_dec, baseline_oos_dec, oos_route, oos_conf, bull_thr=bull_thr, bear_thr=bear_thr, chop_thr=chop_thr)
                val_costs = _combo_metrics(val_df, val_dec)
                oos_costs = _combo_metrics(eval_df, oos_dec)
                key = f"bull{bull_thr:.2f}_bear{bear_thr:.2f}_chop{chop_thr:.2f}"
                payload[key] = (val_dec, oos_dec)
                rows.append({
                    "candidate": key,
                    "bull_thr": float(bull_thr),
                    "bear_thr": float(bear_thr),
                    "chop_thr": float(chop_thr),
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
            "bull_thr": r["bull_thr"],
            "bear_thr": r["bear_thr"],
            "chop_thr": r["chop_thr"],
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
        "design": "Per-regime confidence thresholds for current-Regime3 MoE. Bull/bear/chop experts remain separate and side-constrained; low-confidence rows fall back to the practical parent baseline.",
        "overlay": overlay,
        "selected": {
            "candidate": selected["candidate"],
            "bull_thr": selected["bull_thr"],
            "bear_thr": selected["bear_thr"],
            "chop_thr": selected["chop_thr"],
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
