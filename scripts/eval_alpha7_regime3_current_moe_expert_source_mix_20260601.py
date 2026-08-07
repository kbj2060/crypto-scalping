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
    EXPERT_NAMES,
    OUT_DIR as PRACTICAL_EXPERT_DIR,
    _active,
    _flatten,
    _route_conf,
    _route_id,
    _score,
    _side_constrained,
)


MODEL_ID = "alpha7_regime3_current_moe_expert_source_mix_20260601"
ROUTER_NAME = "regime3_current_context"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_expert_source_mix_20260601"
VARIANT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_feature_variants_20260601"
SOURCES = {
    "practical": PRACTICAL_EXPERT_DIR / ROUTER_NAME,
    "current": VARIANT_DIR / "base_plus_current",
    "risk": VARIANT_DIR / "base_plus_current_risk",
}


def _predict_combo(primary: dict[str, Any], fallback: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    return _combine_primary_fallback(_predict_scaled(primary, df, None), _predict_scaled(fallback, df, None)).reset_index(drop=True)


def _load_pair(source: str, expert: str) -> dict[str, Any]:
    root = SOURCES[source] / expert
    p = root / "primary_no_tp/parent.pkl"
    f = root / "fallback_v2_tp/parent.pkl"
    if not p.exists() or not f.exists():
        raise FileNotFoundError(f"missing {source}/{expert} artifacts: {p}, {f}")
    return {"primary": joblib.load(p), "fallback": joblib.load(f)}


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
) -> pd.DataFrame:
    out = base_dec.copy().reset_index(drop=True)
    decision_cols = list(base_dec.columns)
    out.loc[:, decision_cols] = base_dec.loc[:, decision_cols].to_numpy()
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
    model_cache: dict[tuple[str, str], dict[str, Any]] = {}
    dec_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    for source in SOURCES:
        for expert in EXPERT_NAMES:
            model_cache[(source, expert)] = _load_pair(source, expert)
            dec_cache[("val", source, expert)] = _side_constrained(_predict_combo(model_cache[(source, expert)]["primary"], model_cache[(source, expert)]["fallback"], val_df), expert=expert)
            dec_cache[("oos", source, expert)] = _side_constrained(_predict_combo(model_cache[(source, expert)]["primary"], model_cache[(source, expert)]["fallback"], eval_df), expert=expert)
    rows: list[dict[str, Any]] = []
    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for bull_src, bear_src, chop_src in itertools.product(SOURCES, repeat=3):
        val_expert_dec = {
            "bull": dec_cache[("val", bull_src, "bull")],
            "bear": dec_cache[("val", bear_src, "bear")],
            "chop": dec_cache[("val", chop_src, "chop")],
        }
        oos_expert_dec = {
            "bull": dec_cache[("oos", bull_src, "bull")],
            "bear": dec_cache[("oos", bear_src, "bear")],
            "chop": dec_cache[("oos", chop_src, "chop")],
        }
        for min_conf in [0.80, 0.85]:
            val_dec = _route_decision(val_expert_dec, baseline_val_dec, val_route, val_conf, min_conf=min_conf)
            oos_dec = _route_decision(oos_expert_dec, baseline_oos_dec, oos_route, oos_conf, min_conf=min_conf)
            val_costs = _combo_metrics(val_df, val_dec)
            oos_costs = _combo_metrics(eval_df, oos_dec)
            key = f"bull_{bull_src}__bear_{bear_src}__chop_{chop_src}__conf{min_conf:.2f}"
            payload[key] = (val_dec, oos_dec)
            rows.append({
                "candidate": key,
                "bull_source": bull_src,
                "bear_source": bear_src,
                "chop_source": chop_src,
                "min_conf": float(min_conf),
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
            "bull_source": r["bull_source"],
            "bear_source": r["bear_source"],
            "chop_source": r["chop_source"],
            "min_conf": r["min_conf"],
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
        "design": "Expert source mix inside fixed Regime3 current-context MoE. Each bull/bear/chop expert can come from practical, current-feature, or current+risk-feature training; low confidence remains parent baseline.",
        "overlay": overlay,
        "sources": {k: str(v) for k, v in SOURCES.items()},
        "selected": {
            "candidate": selected["candidate"],
            "bull_source": selected["bull_source"],
            "bear_source": selected["bear_source"],
            "chop_source": selected["chop_source"],
            "min_conf": selected["min_conf"],
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
