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
    TP_COL,
    _combine_primary_fallback,
    _combo_metrics,
    _json_default,
    _predict_scaled,
)
from scripts.eval_alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601 import _apply_scale  # noqa: E402
from scripts.retrain_alpha7_1_01965_tp_sl_decontam_20260528 import _assert_feature_cols, _load_or_train  # noqa: E402
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import RISK_COLS, _load_frames_with_risk  # noqa: E402
from scripts.train_alpha7_regime3_expert_moe_20260601 import (  # noqa: E402
    BASE_CLEAN_DIR,
    EXPERT_NAMES,
    ROUTERS,
    _active,
    _flatten,
    _route_conf,
    _route_id,
    _score,
    _side_constrained,
)


MODEL_ID = "alpha7_active_clean_contract_moe_20260601"
ROUTER_NAME = "regime3_current_context"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_active_clean_contract_moe_20260601"

DROP_PREFIXES = (
    "clean_regime4_",
    "regime4_pred_",
)
DROP_EXACT = {
    TP_COL,
}
DROP_TOKENS = (
    "tp_sl_action_score",
)


def _clean_cols(cols: list[str]) -> list[str]:
    out: list[str] = []
    dropped: list[str] = []
    for col in cols:
        if col in DROP_EXACT or col.startswith(DROP_PREFIXES) or any(tok in col for tok in DROP_TOKENS):
            dropped.append(col)
            continue
        out.append(col)
    deduped = list(dict.fromkeys(out))
    if len(deduped) != len(out):
        raise RuntimeError("feature contract contains duplicate columns after clean filtering")
    return deduped


def _dropped_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if c not in _clean_cols(cols)]


def _assert_clean_contract(cols: list[str], *, name: str) -> None:
    bad = [c for c in cols if c in DROP_EXACT or c.startswith(DROP_PREFIXES) or any(tok in c for tok in DROP_TOKENS)]
    if bad:
        raise RuntimeError(f"{name} contains dropped feature columns: {bad[:40]}")


def _predict_combo(primary: dict[str, Any], fallback: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    return _combine_primary_fallback(_predict_scaled(primary, df, None), _predict_scaled(fallback, df, None)).reset_index(drop=True)


def _train_pair(
    *,
    name: str,
    train_all: pd.DataFrame,
    eval_df: pd.DataFrame,
    primary_cols: list[str],
    fallback_cols: list[str],
    seed: int,
) -> dict[str, Any]:
    _assert_feature_cols(train_all, primary_cols, name=f"{name}_primary_train")
    _assert_feature_cols(eval_df, primary_cols, name=f"{name}_primary_eval")
    _assert_feature_cols(train_all, fallback_cols, name=f"{name}_fallback_train")
    _assert_feature_cols(eval_df, fallback_cols, name=f"{name}_fallback_eval")
    _assert_clean_contract(primary_cols, name=f"{name}_primary")
    _assert_clean_contract(fallback_cols, name=f"{name}_fallback")
    primary, _, primary_summary = _load_or_train(
        train_all=train_all,
        eval_df=eval_df,
        feature_cols=primary_cols,
        seed=seed,
        out_dir=OUT_DIR / name / "primary_no_tp",
    )
    fallback, _, fallback_summary = _load_or_train(
        train_all=train_all,
        eval_df=eval_df,
        feature_cols=fallback_cols,
        seed=seed + 1,
        out_dir=OUT_DIR / name / "fallback_clean",
    )
    return {"primary": primary, "fallback": fallback, "summary": {"primary": primary_summary, "fallback": fallback_summary}}


def _route_decision(
    expert_dec: dict[str, pd.DataFrame],
    base_dec: pd.DataFrame,
    route: np.ndarray,
    conf: np.ndarray,
    *,
    min_conf: float,
) -> pd.DataFrame:
    out = base_dec.copy().reset_index(drop=True)
    decision_cols = list(base_dec.columns)
    selected = route.copy()
    selected[conf < float(min_conf)] = 3
    for idx, expert in enumerate(EXPERT_NAMES):
        mask = selected == idx
        out.loc[mask, decision_cols] = expert_dec[expert].loc[mask, decision_cols].to_numpy()
    out["router_expert"] = np.where(
        selected == 0,
        "bull",
        np.where(selected == 1, "bear", np.where(selected == 2, "chop_expert", "lowconf_baseline")),
    )
    out["router_confidence"] = conf
    out["router_min_conf"] = float(min_conf)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_with_risk()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    primary_base = joblib.load(BASE_CLEAN_DIR / "primary_no_tp/parent.pkl")
    fallback_base = joblib.load(BASE_CLEAN_DIR / "fallback_v2_tp/parent.pkl")
    raw_primary_cols = list(primary_base["feature_cols"])
    raw_fallback_cols = list(fallback_base["feature_cols"])
    clean_primary_cols = _clean_cols(raw_primary_cols)
    clean_fallback_cols = _clean_cols(raw_fallback_cols)
    current_cols = [*ROUTERS[ROUTER_NAME]["cols"], *ROUTERS[ROUTER_NAME]["extra_cols"]]
    risk_extra_cols = [*current_cols, *RISK_COLS]

    base = _train_pair(
        name="baseline_clean",
        train_all=train_all,
        eval_df=eval_df,
        primary_cols=clean_primary_cols,
        fallback_cols=clean_fallback_cols,
        seed=6061100,
    )
    route_train = _route_id(train_all, ROUTER_NAME)
    expert_models: dict[str, dict[str, Any]] = {}
    expert_summaries: dict[str, Any] = {}
    for idx, expert in enumerate(EXPERT_NAMES):
        source = "risk" if expert == "bear" else "practical"
        extra = risk_extra_cols if source == "risk" else []
        mask = route_train == idx
        expert_train = train_all.loc[mask].reset_index(drop=True)
        pair = _train_pair(
            name=f"{expert}_{source}_clean",
            train_all=expert_train,
            eval_df=eval_df,
            primary_cols=list(dict.fromkeys([*clean_primary_cols, *extra])),
            fallback_cols=list(dict.fromkeys([*clean_fallback_cols, *extra])),
            seed=6061200 + idx * 10,
        )
        expert_models[expert] = pair
        expert_summaries[expert] = {"source": source, "rows": int(mask.sum()), **pair["summary"]}

    baseline_val_dec = _predict_combo(base["primary"], base["fallback"], val_df)
    baseline_oos_dec = _predict_combo(base["primary"], base["fallback"], eval_df)
    val_route = _route_id(val_df, ROUTER_NAME)
    oos_route = _route_id(eval_df, ROUTER_NAME)
    val_conf = _route_conf(val_df, ROUTER_NAME)
    oos_conf = _route_conf(eval_df, ROUTER_NAME)
    val_expert_dec: dict[str, pd.DataFrame] = {}
    oos_expert_dec: dict[str, pd.DataFrame] = {}
    for expert, pair in expert_models.items():
        val_expert_dec[expert] = _side_constrained(_predict_combo(pair["primary"], pair["fallback"], val_df), expert=expert)
        oos_expert_dec[expert] = _side_constrained(_predict_combo(pair["primary"], pair["fallback"], eval_df), expert=expert)

    # Selection guard: only validation metrics are computed during grid selection.
    val_payload: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for bull, bear, chop in itertools.product([0.70, 0.80, 0.85], [1.15, 1.30, 1.45], [1.10, 1.25]):
        routed_val = _route_decision(val_expert_dec, baseline_val_dec, val_route, val_conf, min_conf=0.80)
        val_dec = _apply_scale(routed_val, bull=bull, bear=bear, chop=chop)
        val_costs = _combo_metrics(val_df, val_dec)
        key = f"bull{bull:.2f}_bear{bear:.2f}_chop{chop:.2f}"
        val_payload[key] = val_dec
        rows.append({
            "candidate": key,
            "bull_scale": float(bull),
            "bear_scale": float(bear),
            "chop_scale": float(chop),
            "score": float(_score(val_costs)),
            "validation": val_costs,
            "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
        })
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = rows[0]
    selected_val_dec = val_payload[str(selected["candidate"])]
    routed_oos = _route_decision(oos_expert_dec, baseline_oos_dec, oos_route, oos_conf, min_conf=0.80)
    selected_oos_dec = _apply_scale(
        routed_oos,
        bull=float(selected["bull_scale"]),
        bear=float(selected["bear_scale"]),
        chop=float(selected["chop_scale"]),
    )
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
        "design": "Strict clean-contract rebuild of the Alpha7 Regime3 current MoE active path. Baseline and selected bull/bear/chop experts are retrained after dropping Regime4-family and tp_sl_action_score features. OOS is evaluated only after validation scale selection.",
        "drop_contract": {
            "drop_prefixes": DROP_PREFIXES,
            "drop_exact": sorted(DROP_EXACT),
            "drop_tokens": DROP_TOKENS,
            "primary_dropped": _dropped_cols(raw_primary_cols),
            "fallback_dropped": _dropped_cols(raw_fallback_cols),
            "primary_feature_count_before": len(raw_primary_cols),
            "primary_feature_count_after": len(clean_primary_cols),
            "fallback_feature_count_before": len(raw_fallback_cols),
            "fallback_feature_count_after": len(clean_fallback_cols),
        },
        "overlay": overlay,
        "selection_guard": "Grid ranking contains validation metrics only; selected OOS is evaluated after validation selection.",
        "summaries": {"baseline": base["summary"], "experts": expert_summaries},
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
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": report["selected"], "drop_contract": report["drop_contract"]}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
