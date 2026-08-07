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


MODEL_ID = "alpha7_regime3_current_practical_moe_20260601"
ROUTER_NAME = "regime3_current_context"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_practical_moe_20260601"


def _predict_combo(primary: dict[str, Any], fallback: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    return _combine_primary_fallback(_predict_scaled(primary, df, None), _predict_scaled(fallback, df, None)).reset_index(drop=True)


def _cash_decision(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    out.loc[active, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[active, "leverage"] = 1.0
    return out


def _defensive_decision(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    out.loc[active, "notional_exposure"] = pd.to_numeric(out.loc[active, "notional_exposure"], errors="raise") * 0.50
    out.loc[active, "position_fraction"] = pd.to_numeric(out.loc[active, "position_fraction"], errors="raise") * 0.50
    out.loc[active, "take_profit"] = pd.to_numeric(out.loc[active, "take_profit"], errors="raise") * 0.75
    out.loc[active, "stop_loss"] = pd.to_numeric(out.loc[active, "stop_loss"], errors="raise") * 0.75
    hold = pd.to_numeric(out.loc[active, "max_hold_bars"], errors="raise").to_numpy(dtype=np.float64)
    out.loc[active, "max_hold_bars"] = np.maximum(1, np.ceil(hold * 0.50)).astype(int)
    return out


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
        val_dec = _predict_combo(models["primary"], models["fallback"], val_df)
        oos_dec = _predict_combo(models["primary"], models["fallback"], eval_df)
        val[expert] = _side_constrained(val_dec, expert=expert)
        oos[expert] = _side_constrained(oos_dec, expert=expert)
    return val, oos


def _route_decision(
    expert_dec: dict[str, pd.DataFrame],
    base_dec: pd.DataFrame,
    route: np.ndarray,
    conf: np.ndarray,
    *,
    min_conf: float,
    chop_mode: str,
    low_conf_mode: str,
) -> pd.DataFrame:
    out = base_dec.copy().reset_index(drop=True)
    cash = _cash_decision(base_dec)
    defensive_chop = _defensive_decision(expert_dec["chop"])
    decision_cols = list(base_dec.columns)
    fallback = base_dec if low_conf_mode == "baseline" else cash
    out.loc[:, decision_cols] = fallback.loc[:, decision_cols].to_numpy()
    selected = route.copy()
    selected[conf < float(min_conf)] = 3
    maps = {
        "bull": expert_dec["bull"],
        "bear": expert_dec["bear"],
        "chop": expert_dec["chop"],
    }
    if chop_mode == "baseline":
        maps["chop"] = base_dec
    elif chop_mode == "cash":
        maps["chop"] = cash
    elif chop_mode == "defensive":
        maps["chop"] = defensive_chop
    elif chop_mode != "expert":
        raise ValueError(f"unknown chop_mode={chop_mode}")
    for idx, expert in enumerate(EXPERT_NAMES):
        mask = selected == idx
        out.loc[mask, decision_cols] = maps[expert].loc[mask, decision_cols].to_numpy()
    names = np.where(selected == 0, "bull", np.where(selected == 1, "bear", np.where(selected == 2, f"chop_{chop_mode}", f"lowconf_{low_conf_mode}")))
    out["router_expert"] = names
    out["router_confidence"] = conf
    out["router_min_conf"] = float(min_conf)
    out["router_chop_mode"] = chop_mode
    out["router_low_conf_mode"] = low_conf_mode
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_router_frames(ROUTER_NAME)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    primary_base = joblib.load(BASE_CLEAN_DIR / "primary_no_tp/parent.pkl")
    fallback_base = joblib.load(BASE_CLEAN_DIR / "fallback_v2_tp/parent.pkl")
    baseline_val_dec = _predict_combo(primary_base, fallback_base, val_df)
    baseline_oos_dec = _predict_combo(primary_base, fallback_base, eval_df)
    baseline_val = _combo_metrics(val_df, baseline_val_dec)
    baseline_oos = _combo_metrics(eval_df, baseline_oos_dec)
    experts = _load_expert_models()
    val_expert_dec, oos_expert_dec = _build_expert_decisions(experts, val_df, eval_df)
    val_route = _route_id(val_df, ROUTER_NAME)
    oos_route = _route_id(eval_df, ROUTER_NAME)
    val_conf = _route_conf(val_df, ROUTER_NAME)
    oos_conf = _route_conf(eval_df, ROUTER_NAME)
    rows: list[dict[str, Any]] = [{
        "candidate": "baseline",
        "min_conf": None,
        "chop_mode": None,
        "low_conf_mode": None,
        "score": float(_score(baseline_val)),
        "validation": baseline_val,
        "oos": baseline_oos,
        "validation_policy_counts": {"baseline": int(_active(baseline_val_dec).sum()), "cash": int((~_active(baseline_val_dec)).sum())},
        "oos_policy_counts": {"baseline": int(_active(baseline_oos_dec).sum()), "cash": int((~_active(baseline_oos_dec)).sum())},
    }]
    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for min_conf in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        for chop_mode in ["expert", "defensive", "baseline", "cash"]:
            for low_conf_mode in ["baseline", "cash"]:
                val_dec = _route_decision(val_expert_dec, baseline_val_dec, val_route, val_conf, min_conf=min_conf, chop_mode=chop_mode, low_conf_mode=low_conf_mode)
                oos_dec = _route_decision(oos_expert_dec, baseline_oos_dec, oos_route, oos_conf, min_conf=min_conf, chop_mode=chop_mode, low_conf_mode=low_conf_mode)
                val_costs = _combo_metrics(val_df, val_dec)
                oos_costs = _combo_metrics(eval_df, oos_dec)
                key = f"conf{min_conf:.2f}_chop{chop_mode}_low{low_conf_mode}"
                payload[key] = (val_dec, oos_dec)
                rows.append({
                    "candidate": key,
                    "min_conf": float(min_conf),
                    "chop_mode": chop_mode,
                    "low_conf_mode": low_conf_mode,
                    "score": float(_score(val_costs)),
                    "validation": val_costs,
                    "oos": oos_costs,
                    "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
                    "oos_policy_counts": {str(k): int(v) for k, v in oos_dec["router_expert"].value_counts().to_dict().items()},
                })
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = rows[0]
    if selected["candidate"] == "baseline":
        selected_val_dec = baseline_val_dec.copy()
        selected_oos_dec = baseline_oos_dec.copy()
        selected_val_dec["router_expert"] = np.where(_active(selected_val_dec), "baseline", "cash")
        selected_oos_dec["router_expert"] = np.where(_active(selected_oos_dec), "baseline", "cash")
    else:
        selected_val_dec, selected_oos_dec = payload[str(selected["candidate"])]
    selected_val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos_dec.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame([
        {
            "candidate": r["candidate"],
            "min_conf": r["min_conf"],
            "chop_mode": r["chop_mode"],
            "low_conf_mode": r["low_conf_mode"],
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
        "design": "Practical Regime3 current-context MoE candidate. Bull/bear experts are side-constrained; grid tests chop handling and low-confidence fallback without retraining experts.",
        "overlay": overlay,
        "selected": {
            "candidate": selected["candidate"],
            "min_conf": selected["min_conf"],
            "chop_mode": selected["chop_mode"],
            "low_conf_mode": selected["low_conf_mode"],
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
