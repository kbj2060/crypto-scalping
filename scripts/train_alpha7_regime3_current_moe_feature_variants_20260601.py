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
    _read,
)
from scripts.retrain_alpha7_1_01965_tp_sl_decontam_20260528 import (  # noqa: E402
    _assert_feature_cols,
    _load_or_train,
)
from scripts.train_alpha7_regime3_expert_moe_20260601 import (  # noqa: E402
    BASE_CLEAN_DIR,
    EVAL_CSV,
    EXPERT_NAMES,
    ROUTERS,
    TRAIN_CSV,
    _active,
    _flatten,
    _load_router_frames,
    _route_conf,
    _route_id,
    _score,
    _side_constrained,
)


MODEL_ID = "alpha7_regime3_current_moe_feature_variants_20260601"
ROUTER_NAME = "regime3_current_context"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_feature_variants_20260601"
RISK_2025 = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530/training_features_2025_regime3_stability_risk_h6.csv"
RISK_2026 = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530/training_features_2026_rebuilt_regime3_stability_risk_h6.csv"
RISK_COLS = [
    "regime3_stability_h6_score",
    "regime3_transition_h6_risk_prob",
    "regime3_transition_h6_risk_pred",
    "regime3_churn_h6_risk_score",
]


def _read_overlay(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _edge_name(mask: pd.Series) -> str | None:
    idx = np.flatnonzero(mask.to_numpy())
    if len(idx) == 0:
        return None
    if np.array_equal(idx, np.arange(len(idx))):
        return "head"
    if np.array_equal(idx, np.arange(len(mask) - len(idx), len(mask))):
        return "tail"
    return None


def _overlay_required(base: pd.DataFrame, source: Path, cols: list[str], *, tag: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    src = _read_overlay(source)
    missing = [c for c in cols if c not in src.columns]
    if missing:
        raise RuntimeError(f"{tag}: missing required columns: {missing}")
    out = base.copy()
    missing_ts = out.loc[~out["timestamp"].isin(set(src["timestamp"])), "timestamp"]
    dropped: list[dict[str, Any]] = []
    if len(missing_ts) > 0:
        head = out["timestamp"].head(len(missing_ts)).reset_index(drop=True)
        tail = out["timestamp"].tail(len(missing_ts)).reset_index(drop=True)
        miss = missing_ts.reset_index(drop=True)
        if miss.equals(head):
            edge = "head"
        elif miss.equals(tail):
            edge = "tail"
        else:
            raise RuntimeError(f"{tag}: non-edge missing timestamps: {missing_ts.head(20).tolist()}")
        dropped.append({"edge": edge, "rows": int(len(missing_ts)), "first": str(missing_ts.iloc[0]), "last": str(missing_ts.iloc[-1]), "path": str(source)})
        out = out.loc[out["timestamp"].isin(set(src["timestamp"]))].reset_index(drop=True)
    before = len(out)
    out = out.merge(src[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    if len(out) != before:
        raise RuntimeError(f"{tag}: row count changed after overlay")
    nan_mask = out[cols].isna().any(axis=1)
    edge = _edge_name(nan_mask)
    if edge is None and bool(nan_mask.any()):
        raise RuntimeError(f"{tag}: non-edge NaN rows: {out.loc[nan_mask, 'timestamp'].head(20).tolist()}")
    if edge is not None:
        bad = out.loc[nan_mask, "timestamp"]
        dropped.append({"edge": edge, "rows": int(len(bad)), "first": str(bad.iloc[0]), "last": str(bad.iloc[-1]), "path": str(source), "reason": "edge_nan"})
        out = out.loc[~nan_mask].reset_index(drop=True)
    return out, {"path": str(source), "cols": cols, "dropped_edge_rows": dropped}


def _load_frames_with_risk() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train, eval_df, router_report = _load_router_frames(ROUTER_NAME)
    train, train_risk = _overlay_required(train, RISK_2025, RISK_COLS, tag="train_risk")
    eval_df, eval_risk = _overlay_required(eval_df, RISK_2026, RISK_COLS, tag="eval_risk")
    return train, eval_df, {"router": router_report, "train_risk": train_risk, "eval_risk": eval_risk}


def _predict_combo(primary: dict[str, Any], fallback: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    return _combine_primary_fallback(_predict_scaled(primary, df, None), _predict_scaled(fallback, df, None)).reset_index(drop=True)


def _cash_decision(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    out.loc[active, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[active, "leverage"] = 1.0
    return out


def _route_decision(
    expert_dec: dict[str, pd.DataFrame],
    base_dec: pd.DataFrame,
    route: np.ndarray,
    conf: np.ndarray,
    *,
    min_conf: float,
    low_conf_mode: str,
) -> pd.DataFrame:
    out = base_dec.copy().reset_index(drop=True)
    cash = _cash_decision(base_dec)
    decision_cols = list(base_dec.columns)
    fallback = base_dec if low_conf_mode == "baseline" else cash
    out.loc[:, decision_cols] = fallback.loc[:, decision_cols].to_numpy()
    selected = route.copy()
    selected[conf < float(min_conf)] = 3
    for idx, expert in enumerate(EXPERT_NAMES):
        mask = selected == idx
        out.loc[mask, decision_cols] = expert_dec[expert].loc[mask, decision_cols].to_numpy()
    out["router_expert"] = np.where(selected == 0, "bull", np.where(selected == 1, "bear", np.where(selected == 2, "chop", f"lowconf_{low_conf_mode}")))
    out["router_confidence"] = conf
    return out


def _train_variant(
    variant: str,
    train_all: pd.DataFrame,
    eval_df: pd.DataFrame,
    primary_cols: list[str],
    fallback_cols: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    route = _route_id(train_all, ROUTER_NAME)
    experts: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(EXPERT_NAMES):
        mask = route == idx
        expert_train = train_all.loc[mask].reset_index(drop=True)
        primary, _, primary_summary = _load_or_train(
            train_all=expert_train,
            eval_df=eval_df,
            feature_cols=primary_cols,
            seed=6060300 + idx * 10,
            out_dir=OUT_DIR / variant / expert / "primary_no_tp",
        )
        fallback, _, fallback_summary = _load_or_train(
            train_all=expert_train,
            eval_df=eval_df,
            feature_cols=fallback_cols,
            seed=6060301 + idx * 10,
            out_dir=OUT_DIR / variant / expert / "fallback_v2_tp",
        )
        experts[expert] = {"primary": primary, "fallback": fallback}
        summaries[expert] = {"rows": int(mask.sum()), "primary": primary_summary, "fallback": fallback_summary}
    return experts, summaries


def _eval_variant(
    variant: str,
    experts: dict[str, dict[str, Any]],
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    baseline_val_dec: pd.DataFrame,
    baseline_oos_dec: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    val_route = _route_id(val_df, ROUTER_NAME)
    oos_route = _route_id(eval_df, ROUTER_NAME)
    val_conf = _route_conf(val_df, ROUTER_NAME)
    oos_conf = _route_conf(eval_df, ROUTER_NAME)
    val_dec_map: dict[str, pd.DataFrame] = {}
    oos_dec_map: dict[str, pd.DataFrame] = {}
    for expert, models in experts.items():
        val_dec_map[expert] = _side_constrained(_predict_combo(models["primary"], models["fallback"], val_df), expert=expert)
        oos_dec_map[expert] = _side_constrained(_predict_combo(models["primary"], models["fallback"], eval_df), expert=expert)
    rows: list[dict[str, Any]] = []
    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for min_conf in [0.75, 0.80, 0.85, 0.90]:
        for low_conf_mode in ["baseline", "cash"]:
            val_dec = _route_decision(val_dec_map, baseline_val_dec, val_route, val_conf, min_conf=min_conf, low_conf_mode=low_conf_mode)
            oos_dec = _route_decision(oos_dec_map, baseline_oos_dec, oos_route, oos_conf, min_conf=min_conf, low_conf_mode=low_conf_mode)
            val_costs = _combo_metrics(val_df, val_dec)
            oos_costs = _combo_metrics(eval_df, oos_dec)
            key = f"{variant}_conf{min_conf:.2f}_low{low_conf_mode}"
            payload[key] = (val_dec, oos_dec)
            rows.append({
                "candidate": key,
                "variant": variant,
                "min_conf": float(min_conf),
                "low_conf_mode": low_conf_mode,
                "score": float(_score(val_costs)),
                "validation": val_costs,
                "oos": oos_costs,
                "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
                "oos_policy_counts": {str(k): int(v) for k, v in oos_dec["router_expert"].value_counts().to_dict().items()},
            })
    return rows, payload


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_with_risk()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    primary_base = joblib.load(BASE_CLEAN_DIR / "primary_no_tp/parent.pkl")
    fallback_base = joblib.load(BASE_CLEAN_DIR / "fallback_v2_tp/parent.pkl")
    primary_base_cols = list(primary_base["feature_cols"])
    fallback_base_cols = list(fallback_base["feature_cols"])
    current_cols = [*ROUTERS[ROUTER_NAME]["cols"], *ROUTERS[ROUTER_NAME]["extra_cols"]]
    variants = {
        "base_plus_current": current_cols,
        "base_plus_current_risk": [*current_cols, *RISK_COLS],
    }
    baseline_val_dec = _predict_combo(primary_base, fallback_base, val_df)
    baseline_oos_dec = _predict_combo(primary_base, fallback_base, eval_df)
    baseline_val = _combo_metrics(val_df, baseline_val_dec)
    baseline_oos = _combo_metrics(eval_df, baseline_oos_dec)
    rows: list[dict[str, Any]] = [{
        "candidate": "baseline",
        "variant": "baseline",
        "min_conf": None,
        "low_conf_mode": None,
        "score": float(_score(baseline_val)),
        "validation": baseline_val,
        "oos": baseline_oos,
        "validation_policy_counts": {"baseline": int(_active(baseline_val_dec).sum()), "cash": int((~_active(baseline_val_dec)).sum())},
        "oos_policy_counts": {"baseline": int(_active(baseline_oos_dec).sum()), "cash": int((~_active(baseline_oos_dec)).sum())},
    }]
    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    summaries: dict[str, Any] = {}
    for variant, extra in variants.items():
        primary_cols = list(dict.fromkeys([*primary_base_cols, *extra]))
        fallback_cols = list(dict.fromkeys([*fallback_base_cols, *extra]))
        _assert_feature_cols(train_all, primary_cols, name=f"{variant}_primary_train")
        _assert_feature_cols(eval_df, primary_cols, name=f"{variant}_primary_eval")
        _assert_feature_cols(train_all, fallback_cols, name=f"{variant}_fallback_train")
        _assert_feature_cols(eval_df, fallback_cols, name=f"{variant}_fallback_eval")
        experts, variant_summary = _train_variant(variant, train_all, eval_df, primary_cols, fallback_cols)
        summaries[variant] = variant_summary
        variant_rows, variant_payload = _eval_variant(variant, experts, val_df, eval_df, baseline_val_dec, baseline_oos_dec)
        rows.extend(variant_rows)
        payload.update(variant_payload)
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
            "variant": r["variant"],
            "min_conf": r["min_conf"],
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
        "design": "Regime3 current-context MoE feature-variant test. Experts remain bull/bear/chop and side-constrained; only expert input contracts change.",
        "overlay": overlay,
        "summaries": summaries,
        "selected": {
            "candidate": selected["candidate"],
            "variant": selected["variant"],
            "min_conf": selected["min_conf"],
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
