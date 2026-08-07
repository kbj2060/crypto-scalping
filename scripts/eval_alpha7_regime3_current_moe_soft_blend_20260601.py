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

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combine_primary_fallback, _combo_metrics, _json_default, _predict_scaled  # noqa: E402
from scripts.eval_alpha7_regime3_current_moe_expert_source_mix_20260601 import SOURCES  # noqa: E402
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import _load_frames_with_risk  # noqa: E402
from scripts.train_alpha7_regime3_expert_moe_20260601 import BASE_CLEAN_DIR, EXPERT_NAMES, ROUTERS, _active, _flatten, _score, _side_constrained  # noqa: E402


MODEL_ID = "alpha7_regime3_current_moe_soft_blend_20260601"
ROUTER_NAME = "regime3_current_context"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_soft_blend_20260601"


def _predict_combo(primary: dict[str, Any], fallback: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    return _combine_primary_fallback(_predict_scaled(primary, df, None), _predict_scaled(fallback, df, None)).reset_index(drop=True)


def _load_pair(source: str, expert: str) -> dict[str, Any]:
    root = SOURCES[source] / expert
    p = root / "primary_no_tp/parent.pkl"
    f = root / "fallback_v2_tp/parent.pkl"
    if not p.exists() or not f.exists():
        raise FileNotFoundError(f"missing {source}/{expert} artifacts: {p}, {f}")
    return {"primary": joblib.load(p), "fallback": joblib.load(f)}


def _router_weights(frame: pd.DataFrame, *, power: float) -> tuple[np.ndarray, np.ndarray]:
    cols = ROUTERS[ROUTER_NAME]["cols"]
    raw = frame[cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise RuntimeError("non-finite router probabilities")
    raw = np.clip(raw, 0.0, None)
    raw = np.power(raw, float(power))
    denom = np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
    weights = raw / denom
    conf = weights.max(axis=1)
    return weights, conf


def _soft_blend(expert_dec: dict[str, pd.DataFrame], baseline_dec: pd.DataFrame, weights: np.ndarray, *, min_conf: float, side_threshold: float) -> pd.DataFrame:
    n = len(baseline_dec)
    out = baseline_dec.copy().reset_index(drop=True)
    sides = np.column_stack([
        pd.to_numeric(expert_dec[name]["side"], errors="raise").to_numpy(dtype=np.float64)
        for name in EXPERT_NAMES
    ])
    side_score = np.sum(weights * sides, axis=1)
    selected_side = np.where(side_score > float(side_threshold), 1, np.where(side_score < -float(side_threshold), -1, 0)).astype(np.int64)
    low_conf = weights.max(axis=1) < float(min_conf)
    selected_side[low_conf] = pd.to_numeric(baseline_dec.loc[low_conf, "side"], errors="raise").to_numpy(dtype=np.int64)

    for i in range(n):
        if low_conf[i]:
            continue
        side = int(selected_side[i])
        if side == 0:
            out.loc[i, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
            out.loc[i, "leverage"] = 1.0
            continue
        eligible = np.asarray([
            int(expert_dec[name].at[i, "side"]) == side and int(expert_dec[name].at[i, "action"]) != 0
            for name in EXPERT_NAMES
        ], dtype=bool)
        if not bool(eligible.any()):
            out.loc[i, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
            out.loc[i, "leverage"] = 1.0
            continue
        w = weights[i].copy()
        w[~eligible] = 0.0
        w = w / max(float(w.sum()), 1e-12)
        out.at[i, "side"] = side
        out.at[i, "action"] = 1 if side > 0 else 2
        for col in ["notional_exposure", "leverage", "position_fraction", "take_profit", "stop_loss", "quality_score", "confidence"]:
            vals = np.asarray([float(expert_dec[name].at[i, col]) for name in EXPERT_NAMES], dtype=np.float64)
            out.at[i, col] = float(np.dot(w, vals))
        for col in ["max_hold_bars", "cooldown_bars"]:
            vals = np.asarray([float(expert_dec[name].at[i, col]) for name in EXPERT_NAMES], dtype=np.float64)
            out.at[i, col] = int(round(float(np.dot(w, vals))))

    out["router_expert"] = np.where(low_conf, "lowconf_baseline", "soft_blend")
    out["router_confidence"] = weights.max(axis=1)
    out["router_min_conf"] = float(min_conf)
    out["soft_side_score"] = side_score
    out["soft_side_threshold"] = float(side_threshold)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_with_risk()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    primary_base = joblib.load(BASE_CLEAN_DIR / "primary_no_tp/parent.pkl")
    fallback_base = joblib.load(BASE_CLEAN_DIR / "fallback_v2_tp/parent.pkl")
    baseline_val_dec = _predict_combo(primary_base, fallback_base, val_df)
    baseline_oos_dec = _predict_combo(primary_base, fallback_base, eval_df)

    source_mix = {"bull": "practical", "bear": "risk", "chop": "practical"}
    val_expert_dec: dict[str, pd.DataFrame] = {}
    oos_expert_dec: dict[str, pd.DataFrame] = {}
    for expert, source in source_mix.items():
        models = _load_pair(source, expert)
        val_expert_dec[expert] = _side_constrained(_predict_combo(models["primary"], models["fallback"], val_df), expert=expert)
        oos_expert_dec[expert] = _side_constrained(_predict_combo(models["primary"], models["fallback"], eval_df), expert=expert)

    rows: list[dict[str, Any]] = []
    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    candidates = [
        (1.0, 0.00, 0.15),
        (1.0, 0.65, 0.15),
        (1.5, 0.00, 0.15),
        (1.5, 0.65, 0.15),
        (2.0, 0.00, 0.25),
        (2.0, 0.65, 0.25),
        (3.0, 0.75, 0.25),
        (3.0, 0.75, 0.35),
    ]
    for power, min_conf, side_th in candidates:
        val_w, _ = _router_weights(val_df, power=power)
        val_dec = _soft_blend(val_expert_dec, baseline_val_dec, val_w, min_conf=min_conf, side_threshold=side_th)
        val_costs = _combo_metrics(val_df, val_dec)
        key = f"p{power:.1f}_conf{min_conf:.2f}_side{side_th:.2f}"
        payload[key] = (val_dec, pd.DataFrame())
        rows.append({
            "candidate": key,
            "power": float(power),
            "min_conf": float(min_conf),
            "side_threshold": float(side_th),
            "score": float(_score(val_costs)),
            "validation": val_costs,
            "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
        })
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = rows[0]
    selected_val_dec, _ = payload[str(selected["candidate"])]
    oos_w, _ = _router_weights(eval_df, power=float(selected["power"]))
    selected_oos_dec = _soft_blend(
        oos_expert_dec,
        baseline_oos_dec,
        oos_w,
        min_conf=float(selected["min_conf"]),
        side_threshold=float(selected["side_threshold"]),
    )
    selected["oos"] = _combo_metrics(eval_df, selected_oos_dec)
    selected["oos_policy_counts"] = {str(k): int(v) for k, v in selected_oos_dec["router_expert"].value_counts().to_dict().items()}
    selected_val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos_dec.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame([
        {
            "candidate": r["candidate"],
            "power": r["power"],
            "min_conf": r["min_conf"],
            "side_threshold": r["side_threshold"],
            "score": r["score"],
            **_flatten("val", r["validation"]),
            "validation_policy_counts": json.dumps(r["validation_policy_counts"], ensure_ascii=False),
        }
        for r in rows
    ]).to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Decision-level Soft MoE adaptation of the current Regime3 bull/bear/chop HGB experts. Uses router probabilities to blend same-side expert decisions; no retraining and no OOS selection.",
        "source_mix": source_mix,
        "overlay": overlay,
        "selected": selected,
        "top_grid": rows[:12],
        "artifacts": {
            "report": str(OUT_DIR / "report.json"),
            "ranking": str(OUT_DIR / "ranking.csv"),
            "validation_decisions": str(OUT_DIR / "validation_decisions.csv"),
            "oos_decisions": str(OUT_DIR / "oos_2026_decisions.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
