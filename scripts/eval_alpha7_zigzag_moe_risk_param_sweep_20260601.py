#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combo_metrics, _json_default
from scripts.eval_alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601 import _apply_scale
from scripts.retrain_alpha7_active_max_feature_contract_moe_20260601 import ROUTER_NAME, _load_frames_max
from scripts.retrain_alpha7_active_max_feature_zigzag_moe_20260601 import _feature_frame
from scripts.train_alpha7_regime3_expert_moe_20260601 import EXPERT_NAMES, _active, _flatten, _route_conf, _route_id, _score


MODEL_ID = "alpha7_active_max_feature_zigzag_moe_risk_redesign_20260601"
BASE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_active_max_feature_zigzag_moe_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_active_max_feature_zigzag_moe_risk_redesign_20260601"

ACTION_CASH = 0
ACTION_LONG = 1
ACTION_SHORT = 2


RISK_TEMPLATES: dict[str, dict[str, float | int]] = {
    "micro_rr18": {"notional": 0.25, "leverage": 2.0, "take_profit": 0.012, "stop_loss": 0.007, "max_hold": 24, "cooldown": 3},
    "scalp_rr20": {"notional": 0.35, "leverage": 2.0, "take_profit": 0.018, "stop_loss": 0.009, "max_hold": 48, "cooldown": 6},
    "balanced_rr19": {"notional": 0.45, "leverage": 2.0, "take_profit": 0.026, "stop_loss": 0.014, "max_hold": 72, "cooldown": 6},
    "mid_rr20": {"notional": 0.55, "leverage": 2.0, "take_profit": 0.034, "stop_loss": 0.017, "max_hold": 96, "cooldown": 8},
    "wide_rr22": {"notional": 0.45, "leverage": 2.0, "take_profit": 0.055, "stop_loss": 0.025, "max_hold": 144, "cooldown": 12},
    "trend_rr25": {"notional": 0.40, "leverage": 1.5, "take_profit": 0.080, "stop_loss": 0.032, "max_hold": 288, "cooldown": 18},
    "safe_rr20": {"notional": 0.25, "leverage": 1.5, "take_profit": 0.030, "stop_loss": 0.015, "max_hold": 96, "cooldown": 12},
}

PRIMARY_CONF_GRID = [0.55, 0.65, 0.70]
FALLBACK_CONF_GRID = [0.50]
MIN_EDGE_GRID = [0.04, 0.08, 0.12]
ROUTE_MIN_CONF_GRID = [0.65, 0.80]
BULL_SCALE_GRID = [0.75, 0.90]
BEAR_SCALE_GRID = [0.90, 1.05]
CHOP_SCALE_GRID = [0.75, 0.90]


def _load_model(path: Path) -> CatBoostClassifier:
    if not path.exists():
        raise FileNotFoundError(path)
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def _models() -> dict[str, CatBoostClassifier]:
    out: dict[str, CatBoostClassifier] = {
        "baseline_primary": _load_model(BASE_DIR / "baseline_max/primary_zigzag/zigzag_action.cbm"),
        "baseline_fallback": _load_model(BASE_DIR / "baseline_max/fallback_zigzag/zigzag_action.cbm"),
    }
    for expert in EXPERT_NAMES:
        out[f"{expert}_primary"] = _load_model(BASE_DIR / f"{expert}_max/primary_zigzag/zigzag_action.cbm")
        out[f"{expert}_fallback"] = _load_model(BASE_DIR / f"{expert}_max/fallback_zigzag/zigzag_action.cbm")
    return out


def _load_feature_cols() -> list[str]:
    report = json.loads((BASE_DIR / "report.json").read_text(encoding="utf-8"))
    cols = list(report["feature_contract"]["feature_cols"])
    if any("zigzag" in c.lower() for c in cols):
        raise RuntimeError("zigzag feature leaked into feature contract")
    return cols


def _probas(models: dict[str, CatBoostClassifier], df: pd.DataFrame, feature_cols: list[str]) -> dict[str, np.ndarray]:
    x = _feature_frame(df, feature_cols)
    out: dict[str, np.ndarray] = {}
    for name, model in models.items():
        arr = np.asarray(model.predict_proba(x), dtype=np.float64)
        if arr.shape[1] != 3:
            raise RuntimeError(f"{name}: expected 3 classes, got {arr.shape}")
        out[name] = arr
    return out


def _dec_from_proba(proba: np.ndarray, *, min_conf: float, min_edge: float, template: dict[str, float | int]) -> pd.DataFrame:
    active_prob = np.maximum(proba[:, ACTION_LONG], proba[:, ACTION_SHORT])
    cash_prob = proba[:, ACTION_CASH]
    side = np.where(proba[:, ACTION_LONG] >= proba[:, ACTION_SHORT], 1, -1).astype(np.int64)
    conf = np.max(proba, axis=1)
    edge = active_prob - cash_prob
    active = (active_prob >= float(min_conf)) & (edge >= float(min_edge))
    action = np.where(active, np.where(side > 0, ACTION_LONG, ACTION_SHORT), ACTION_CASH).astype(np.int64)
    side = np.where(active, side, 0).astype(np.int64)
    notional = float(template["notional"])
    return pd.DataFrame(
        {
            "action": action,
            "side": side,
            "notional_exposure": np.where(active, notional, 0.0),
            "leverage": np.where(active, float(template["leverage"]), 1.0),
            "position_fraction": np.where(active, notional, 0.0),
            "take_profit": np.where(active, float(template["take_profit"]), 0.0),
            "stop_loss": np.where(active, float(template["stop_loss"]), 0.0),
            "max_hold_bars": np.where(active, int(template["max_hold"]), 0).astype(np.int64),
            "cooldown_bars": np.where(active, int(template["cooldown"]), 0).astype(np.int64),
            "quality_score": active_prob.astype(np.float64),
            "confidence": conf.astype(np.float64),
        }
    )


def _combine(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    out = primary.copy().reset_index(drop=True)
    mask = (~_active(out)) & _active(fallback)
    for col in fallback.columns:
        out.loc[mask, col] = fallback.loc[mask, col].to_numpy()
    return out


def _pair_dec(probas: dict[str, np.ndarray], prefix: str, *, primary_conf: float, fallback_conf: float, min_edge: float, template: dict[str, float | int]) -> pd.DataFrame:
    p = _dec_from_proba(probas[f"{prefix}_primary"], min_conf=primary_conf, min_edge=min_edge, template=template)
    f = _dec_from_proba(probas[f"{prefix}_fallback"], min_conf=fallback_conf, min_edge=min_edge, template=template)
    return _combine(p, f)


def _side_constrained(dec: pd.DataFrame, *, expert: str) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    side = pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.int64)
    if expert == "bull":
        block = active & (side < 0)
    elif expert == "bear":
        block = active & (side > 0)
    else:
        block = np.zeros(len(out), dtype=bool)
    out.loc[block, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[block, "leverage"] = 1.0
    return out


def _route(experts: dict[str, pd.DataFrame], baseline: pd.DataFrame, route_id: np.ndarray, route_conf: np.ndarray, *, min_conf: float) -> pd.DataFrame:
    out = baseline.copy().reset_index(drop=True)
    cols = list(baseline.columns)
    selected = route_id.copy()
    selected[route_conf < float(min_conf)] = 3
    for idx, expert in enumerate(EXPERT_NAMES):
        mask = selected == idx
        out.loc[mask, cols] = experts[expert].loc[mask, cols].to_numpy()
    out["router_expert"] = np.where(selected == 0, "bull", np.where(selected == 1, "bear", np.where(selected == 2, "chop_expert", "lowconf_baseline")))
    out["router_confidence"] = route_conf
    out["router_min_conf"] = float(min_conf)
    return out


def _candidate_decisions(
    probas: dict[str, np.ndarray],
    route_id: np.ndarray,
    route_conf: np.ndarray,
    *,
    template: dict[str, float | int],
    primary_conf: float,
    fallback_conf: float,
    min_edge: float,
    route_min_conf: float,
) -> pd.DataFrame:
    base = _pair_dec(probas, "baseline", primary_conf=primary_conf, fallback_conf=fallback_conf, min_edge=min_edge, template=template)
    experts = {
        expert: _side_constrained(
            _pair_dec(probas, expert, primary_conf=primary_conf, fallback_conf=fallback_conf, min_edge=min_edge, template=template),
            expert=expert,
        )
        for expert in EXPERT_NAMES
    }
    return _route(experts, base, route_id, route_conf, min_conf=route_min_conf)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_max()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    feature_cols = _load_feature_cols()
    models = _models()
    val_probas = _probas(models, val_df, feature_cols)
    oos_probas = _probas(models, eval_df, feature_cols)
    val_route_id = _route_id(val_df, ROUTER_NAME)
    oos_route_id = _route_id(eval_df, ROUTER_NAME)
    val_route_conf = _route_conf(val_df, ROUTER_NAME)
    oos_route_conf = _route_conf(eval_df, ROUTER_NAME)

    rows: list[dict[str, Any]] = []
    total = (
        len(RISK_TEMPLATES)
        * len(PRIMARY_CONF_GRID)
        * len(FALLBACK_CONF_GRID)
        * len(MIN_EDGE_GRID)
        * len(ROUTE_MIN_CONF_GRID)
        * len(BULL_SCALE_GRID)
        * len(BEAR_SCALE_GRID)
        * len(CHOP_SCALE_GRID)
    )
    seen = 0
    for template_name, template in RISK_TEMPLATES.items():
        for primary_conf, fallback_conf, min_edge, route_min_conf in itertools.product(
            PRIMARY_CONF_GRID,
            FALLBACK_CONF_GRID,
            MIN_EDGE_GRID,
            ROUTE_MIN_CONF_GRID,
        ):
            routed_val = _candidate_decisions(
                val_probas,
                val_route_id,
                val_route_conf,
                template=template,
                primary_conf=primary_conf,
                fallback_conf=fallback_conf,
                min_edge=min_edge,
                route_min_conf=route_min_conf,
            )
            for bull, bear, chop in itertools.product(BULL_SCALE_GRID, BEAR_SCALE_GRID, CHOP_SCALE_GRID):
                val_dec = _apply_scale(routed_val, bull=bull, bear=bear, chop=chop)
                costs = _combo_metrics(val_df, val_dec)
                seen += 1
                row = {
                    "candidate": f"{template_name}_pc{primary_conf:.2f}_fc{fallback_conf:.2f}_edge{min_edge:.2f}_rc{route_min_conf:.2f}_b{bull:.2f}_r{bear:.2f}_c{chop:.2f}",
                    "template": template_name,
                    "primary_conf": float(primary_conf),
                    "fallback_conf": float(fallback_conf),
                    "min_edge": float(min_edge),
                    "route_min_conf": float(route_min_conf),
                    "bull_scale": float(bull),
                    "bear_scale": float(bear),
                    "chop_scale": float(chop),
                    "score": float(_score(costs)),
                    "validation": costs,
                    "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
                }
                rows.append(row)
                if seen % 100 == 0:
                    print(json.dumps({"progress": seen, "total": total, "best_score_so_far": max(float(r["score"]) for r in rows)}, ensure_ascii=False), flush=True)
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    top = rows[:30]

    selected_payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for row in top[:10]:
        template = RISK_TEMPLATES[str(row["template"])]
        val_base = _candidate_decisions(
            val_probas,
            val_route_id,
            val_route_conf,
            template=template,
            primary_conf=float(row["primary_conf"]),
            fallback_conf=float(row["fallback_conf"]),
            min_edge=float(row["min_edge"]),
            route_min_conf=float(row["route_min_conf"]),
        )
        oos_base = _candidate_decisions(
            oos_probas,
            oos_route_id,
            oos_route_conf,
            template=template,
            primary_conf=float(row["primary_conf"]),
            fallback_conf=float(row["fallback_conf"]),
            min_edge=float(row["min_edge"]),
            route_min_conf=float(row["route_min_conf"]),
        )
        selected_payload[str(row["candidate"])] = (
            _apply_scale(val_base, bull=float(row["bull_scale"]), bear=float(row["bear_scale"]), chop=float(row["chop_scale"])),
            _apply_scale(oos_base, bull=float(row["bull_scale"]), bear=float(row["bear_scale"]), chop=float(row["chop_scale"])),
        )

    selected = top[0]
    selected_val, selected_oos = selected_payload[str(selected["candidate"])]
    oos_costs = _combo_metrics(eval_df, selected_oos)
    selected_val.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": r["candidate"],
                "template": r["template"],
                "primary_conf": r["primary_conf"],
                "fallback_conf": r["fallback_conf"],
                "min_edge": r["min_edge"],
                "route_min_conf": r["route_min_conf"],
                "bull_scale": r["bull_scale"],
                "bear_scale": r["bear_scale"],
                "chop_scale": r["chop_scale"],
                "score": r["score"],
                **_flatten("val", r["validation"]),
                "validation_policy_counts": json.dumps(r["validation_policy_counts"], ensure_ascii=False),
            }
            for r in rows
        ]
    ).to_csv(OUT_DIR / "ranking_validation_only.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Risk/execution parameter redesign for the Omega1 ZigZag-only max-feature current-Regime3 MoE. No supervised target other than zigzag_action is used; this sweep only changes confidence, route, notional, TP/SL, max-hold, cooldown, and expert scales.",
        "feature_count": len(feature_cols),
        "overlay": overlay,
        "risk_templates": RISK_TEMPLATES,
        "sweep_grid": {
            "primary_conf": PRIMARY_CONF_GRID,
            "fallback_conf": FALLBACK_CONF_GRID,
            "min_edge": MIN_EDGE_GRID,
            "route_min_conf": ROUTE_MIN_CONF_GRID,
            "bull_scale": BULL_SCALE_GRID,
            "bear_scale": BEAR_SCALE_GRID,
            "chop_scale": CHOP_SCALE_GRID,
        },
        "selected": {
            **{k: selected[k] for k in ["candidate", "template", "primary_conf", "fallback_conf", "min_edge", "route_min_conf", "bull_scale", "bear_scale", "chop_scale"]},
            "validation": selected["validation"],
            "oos": oos_costs,
            "validation_policy_counts": {str(k): int(v) for k, v in selected_val["router_expert"].value_counts().to_dict().items()},
            "oos_policy_counts": {str(k): int(v) for k, v in selected_oos["router_expert"].value_counts().to_dict().items()},
        },
        "top_grid": top,
        "artifacts": {
            "report": str(OUT_DIR / "report.json"),
            "ranking_validation_only": str(OUT_DIR / "ranking_validation_only.csv"),
            "validation_decisions": str(OUT_DIR / "validation_decisions.csv"),
            "oos_decisions": str(OUT_DIR / "oos_2026_decisions.csv"),
            "source_zigzag_moe_report": str(BASE_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": report["selected"]}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
