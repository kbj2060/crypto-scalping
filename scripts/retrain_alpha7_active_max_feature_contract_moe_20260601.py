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

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combine_primary_fallback, _combo_metrics, _json_default, _predict_scaled
from scripts.eval_alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601 import _apply_scale
from scripts.retrain_alpha7_1_01965_tp_sl_decontam_20260528 import _assert_feature_cols, _load_or_train
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import RISK_COLS, _overlay_required
from scripts.train_alpha7_regime3_expert_moe_20260601 import (
    EVAL_CSV,
    EXPERT_NAMES,
    ROUTERS,
    TRAIN_CSV,
    _flatten,
    _read,
    _route_conf,
    _route_id,
    _score,
    _side_constrained,
)


MODEL_ID = "alpha7_active_max_feature_contract_moe_20260601"
ROUTER_NAME = "regime3_current_context"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_active_max_feature_contract_moe_20260601"
CMAMBA_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_h6_sidecar_20260601"
CMAMBA_2025 = CMAMBA_DIR / "training_features_2025_regime3_cryptomamba_h6_sidecar_20260601.csv"
CMAMBA_2026 = CMAMBA_DIR / "training_features_2026_rebuilt_regime3_cryptomamba_h6_sidecar_20260601.csv"
RISK_2025 = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530/training_features_2025_regime3_stability_risk_h6.csv"
RISK_2026 = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530/training_features_2026_rebuilt_regime3_stability_risk_h6.csv"

DENY_PREFIXES = ("teacher_", "a5dir_", "clean_regime4_", "regime4_pred_", "regime3_pred_")
DENY_TOKENS = ("target", "future", "pnl", "wave3", "zigzag", "tp_sl_action_score")
SAFE_EXACT = {"realized_vol_ratio", "realized_skewness", "m7_hdb_label"}
NON_FEATURE_COLS = {"timestamp"}


def _read_overlay(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "timestamp" not in df.columns:
        raise RuntimeError(f"{path} missing timestamp")
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _load_frames_max() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    current = ROUTERS[ROUTER_NAME]
    current_cols = [*current["cols"], *current["extra_cols"]]
    train = _read(TRAIN_CSV)
    eval_df = _read(EVAL_CSV)
    train, current_train = _overlay_required(train, current["train"], current_cols, tag="train_regime3_current")
    eval_df, current_eval = _overlay_required(eval_df, current["eval"], current_cols, tag="eval_regime3_current")
    cmamba_train_df = _read_overlay(CMAMBA_2025)
    cmamba_eval_df = _read_overlay(CMAMBA_2026)
    cmamba_cols = [c for c in cmamba_train_df.columns if c != "timestamp"]
    train, cmamba_train = _overlay_required(train, CMAMBA_2025, cmamba_cols, tag="train_regime3_cmamba_sidecar")
    eval_df, cmamba_eval = _overlay_required(eval_df, CMAMBA_2026, cmamba_cols, tag="eval_regime3_cmamba_sidecar")
    train, risk_train = _overlay_required(train, RISK_2025, RISK_COLS, tag="train_regime3_risk_h6")
    eval_df, risk_eval = _overlay_required(eval_df, RISK_2026, RISK_COLS, tag="eval_regime3_risk_h6")
    return train, eval_df, {
        "current": {"train": current_train, "eval": current_eval},
        "cmamba_sidecar": {"train": cmamba_train, "eval": cmamba_eval},
        "risk_h6": {"train": risk_train, "eval": risk_eval},
    }


def _is_allowed_feature(col: str, train: pd.DataFrame, eval_df: pd.DataFrame) -> bool:
    if col in NON_FEATURE_COLS:
        return False
    if col not in train.columns or col not in eval_df.columns:
        return False
    if col not in SAFE_EXACT:
        low = col.lower()
        if col.startswith(DENY_PREFIXES) or any(tok in low for tok in DENY_TOKENS):
            return False
    return pd.api.types.is_numeric_dtype(train[col]) and pd.api.types.is_numeric_dtype(eval_df[col])


def _max_feature_cols(train: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
    cols = [c for c in train.columns if _is_allowed_feature(c, train, eval_df)]
    cols = ["side_hint", *[c for c in cols if c != "side_hint"]]
    if len(cols) != len(set(cols)):
        raise RuntimeError("duplicate max feature columns")
    if len(cols) < 120:
        raise RuntimeError(f"unexpectedly small max feature set: {len(cols)}")
    return cols


def _assert_no_forbidden(cols: list[str]) -> None:
    bad: list[str] = []
    for col in cols:
        if col in SAFE_EXACT:
            continue
        low = col.lower()
        if col.startswith(DENY_PREFIXES) or any(tok in low for tok in DENY_TOKENS):
            bad.append(col)
    if bad:
        raise RuntimeError(f"forbidden max feature columns: {bad[:40]}")


def _predict_combo(primary: dict[str, Any], fallback: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    return _combine_primary_fallback(_predict_scaled(primary, df, None), _predict_scaled(fallback, df, None)).reset_index(drop=True)


def _train_pair(name: str, train_all: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str], seed: int) -> dict[str, Any]:
    _assert_feature_cols(train_all, feature_cols, name=f"{name}_train")
    _assert_feature_cols(eval_df, feature_cols, name=f"{name}_eval")
    _assert_no_forbidden(feature_cols)
    primary, _, primary_summary = _load_or_train(
        train_all=train_all,
        eval_df=eval_df,
        feature_cols=feature_cols,
        seed=seed,
        out_dir=OUT_DIR / name / "primary_max",
    )
    fallback, _, fallback_summary = _load_or_train(
        train_all=train_all,
        eval_df=eval_df,
        feature_cols=feature_cols,
        seed=seed + 1,
        out_dir=OUT_DIR / name / "fallback_max",
    )
    return {"primary": primary, "fallback": fallback, "summary": {"primary": primary_summary, "fallback": fallback_summary}}


def _route_decision(expert_dec: dict[str, pd.DataFrame], base_dec: pd.DataFrame, route: np.ndarray, conf: np.ndarray, *, min_conf: float) -> pd.DataFrame:
    out = base_dec.copy().reset_index(drop=True)
    decision_cols = list(base_dec.columns)
    selected = route.copy()
    selected[conf < float(min_conf)] = 3
    for idx, expert in enumerate(EXPERT_NAMES):
        mask = selected == idx
        out.loc[mask, decision_cols] = expert_dec[expert].loc[mask, decision_cols].to_numpy()
    out["router_expert"] = np.where(selected == 0, "bull", np.where(selected == 1, "bear", np.where(selected == 2, "chop_expert", "lowconf_baseline")))
    out["router_confidence"] = conf
    out["router_min_conf"] = float(min_conf)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_max()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    feature_cols = _max_feature_cols(train_all, eval_df)
    base = _train_pair("baseline_max", train_all, eval_df, feature_cols, seed=6063100)
    route_train = _route_id(train_all, ROUTER_NAME)
    experts: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(EXPERT_NAMES):
        mask = route_train == idx
        expert_train = train_all.loc[mask].reset_index(drop=True)
        pair = _train_pair(f"{expert}_max", expert_train, eval_df, feature_cols, seed=6063200 + idx * 10)
        experts[expert] = pair
        summaries[expert] = {"rows": int(mask.sum()), **pair["summary"]}
    baseline_val_dec = _predict_combo(base["primary"], base["fallback"], val_df)
    baseline_oos_dec = _predict_combo(base["primary"], base["fallback"], eval_df)
    val_route = _route_id(val_df, ROUTER_NAME)
    oos_route = _route_id(eval_df, ROUTER_NAME)
    val_conf = _route_conf(val_df, ROUTER_NAME)
    oos_conf = _route_conf(eval_df, ROUTER_NAME)
    val_expert_dec: dict[str, pd.DataFrame] = {}
    oos_expert_dec: dict[str, pd.DataFrame] = {}
    for expert, pair in experts.items():
        val_expert_dec[expert] = _side_constrained(_predict_combo(pair["primary"], pair["fallback"], val_df), expert=expert)
        oos_expert_dec[expert] = _side_constrained(_predict_combo(pair["primary"], pair["fallback"], eval_df), expert=expert)

    routed_val = _route_decision(val_expert_dec, baseline_val_dec, val_route, val_conf, min_conf=0.80)
    rows: list[dict[str, Any]] = []
    val_payload: dict[str, pd.DataFrame] = {}
    for bull, bear, chop in itertools.product([0.70, 0.80, 0.85], [1.15, 1.30, 1.45], [1.10, 1.25]):
        val_dec = _apply_scale(routed_val, bull=bull, bear=bear, chop=chop)
        val_costs = _combo_metrics(val_df, val_dec)
        key = f"bull{bull:.2f}_bear{bear:.2f}_chop{chop:.2f}"
        rows.append({
            "candidate": key,
            "bull_scale": float(bull),
            "bear_scale": float(bear),
            "chop_scale": float(chop),
            "score": float(_score(val_costs)),
            "validation": val_costs,
            "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
        })
        val_payload[key] = val_dec
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = rows[0]
    selected_val_dec = val_payload[str(selected["candidate"])]
    routed_oos = _route_decision(oos_expert_dec, baseline_oos_dec, oos_route, oos_conf, min_conf=0.80)
    selected_oos_dec = _apply_scale(routed_oos, bull=float(selected["bull_scale"]), bear=float(selected["bear_scale"]), chop=float(selected["chop_scale"]))
    oos_costs = _combo_metrics(eval_df, selected_oos_dec)
    selected_val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos_dec.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame([
        {
            "candidate": r["candidate"],
            "bull_scale": r["bull_scale"],
            "bear_scale": r["bear_scale"],
            "chop_scale": r["chop_scale"],
            "score": r["score"],
            **_flatten("val", r["validation"]),
            "validation_policy_counts": json.dumps(r["validation_policy_counts"], ensure_ascii=False),
        }
        for r in rows
    ]).to_csv(OUT_DIR / "ranking_validation_only.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Max allowed feature Alpha7 current-Regime3 MoE. Uses all common numeric train/eval features after strict deny-list filtering, plus renamed Regime3 CryptoMamba h6 sidecar, current context, and h6 risk sidecar.",
        "feature_contract": {
            "feature_count": len(feature_cols),
            "feature_cols": feature_cols,
            "deny_prefixes": DENY_PREFIXES,
            "deny_tokens": DENY_TOKENS,
            "safe_exact": sorted(SAFE_EXACT),
        },
        "overlay": overlay,
        "selection_guard": "Grid ranking contains validation metrics only; selected OOS is evaluated after validation selection.",
        "summaries": {"baseline": base["summary"], "experts": summaries},
        "selected": {
            "candidate": selected["candidate"],
            "bull_scale": selected["bull_scale"],
            "bear_scale": selected["bear_scale"],
            "chop_scale": selected["chop_scale"],
            "validation": selected["validation"],
            "oos": oos_costs,
            "validation_policy_counts": {str(k): int(v) for k, v in selected_val_dec["router_expert"].value_counts().to_dict().items()},
            "oos_policy_counts": {str(k): int(v) for k, v in selected_oos_dec["router_expert"].value_counts().to_dict().items()},
        },
        "artifacts": {
            "report": str(OUT_DIR / "report.json"),
            "ranking_validation_only": str(OUT_DIR / "ranking_validation_only.csv"),
            "validation_decisions": str(OUT_DIR / "validation_decisions.csv"),
            "oos_decisions": str(OUT_DIR / "oos_2026_decisions.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": report["selected"], "feature_count": len(feature_cols)}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
