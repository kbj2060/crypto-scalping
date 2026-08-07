#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combo_metrics, _json_default  # noqa: E402
from scripts.analyze_alpha7_regime3_current_moe_trade_ledger_wr_20260601 import _backtest_decisions  # noqa: E402
from scripts.eval_alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601 import _apply_scale  # noqa: E402
from scripts.eval_alpha7_regime3_current_moe_expert_source_mix_20260601 import _load_pair, _predict_combo, _route_decision  # noqa: E402
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import _load_frames_with_risk  # noqa: E402
from scripts.train_alpha7_regime3_expert_moe_20260601 import BASE_CLEAN_DIR, EXPERT_NAMES, _active, _flatten, _route_conf, _route_id, _score, _side_constrained  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402


MODEL_ID = "alpha7_regime3_current_moe_two_stage_entry_gate_20260601"
ROUTER_NAME = "regime3_current_context"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_two_stage_entry_gate_20260601"


def _active_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    primary_base = joblib.load(BASE_CLEAN_DIR / "primary_no_tp/parent.pkl")
    fallback_base = joblib.load(BASE_CLEAN_DIR / "fallback_v2_tp/parent.pkl")
    base_dec = _predict_combo(primary_base, fallback_base, frame)
    route = _route_id(frame, ROUTER_NAME)
    conf = _route_conf(frame, ROUTER_NAME)
    source_mix = {"bull": "practical", "bear": "risk", "chop": "practical"}
    expert_dec: dict[str, pd.DataFrame] = {}
    for expert, source in source_mix.items():
        models = _load_pair(source, expert)
        expert_dec[expert] = _side_constrained(_predict_combo(models["primary"], models["fallback"], frame), expert=expert)
    dec = _route_decision(expert_dec, base_dec, route, conf, min_conf=0.80)
    dec = _apply_scale(dec, bull=0.85, bear=1.15, chop=1.25)
    return dec


def _gate_features(frame: pd.DataFrame, dec: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "quality_score",
        "confidence",
        "router_confidence",
        "notional_exposure",
        "leverage",
        "take_profit",
        "stop_loss",
        "max_hold_bars",
    ]
    x = dec.reindex(columns=cols).copy()
    x["side"] = pd.to_numeric(dec["side"], errors="raise")
    x["is_bull"] = dec["router_expert"].astype(str).eq("bull").astype(float)
    x["is_bear"] = dec["router_expert"].astype(str).eq("bear").astype(float)
    x["is_chop"] = dec["router_expert"].astype(str).eq("chop_expert").astype(float)
    x["is_lowconf"] = dec["router_expert"].astype(str).eq("lowconf_baseline").astype(float)
    for c in [
        "regime3_current_sensitive_wide24_bull_prob",
        "regime3_current_sensitive_wide24_bear_prob",
        "regime3_current_sensitive_wide24_chop_prob",
        "regime3_current_sensitive_wide24_entropy",
        "regime3_current_sensitive_wide24_margin",
    ]:
        if c in frame.columns:
            x[c] = pd.to_numeric(frame[c], errors="raise")
    return x.replace([np.inf, -np.inf], np.nan)


def _train_labels(frame: pd.DataFrame, dec: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    fee = float(joblib.load(v31.DEFAULT_PARENT)["config"]["fee"]) * 3.0
    slip = float(joblib.load(v31.DEFAULT_PARENT)["config"]["slip"]) * 3.0
    bt = _backtest_decisions(frame, dec, fee=fee, slip=slip)
    ledger = pd.DataFrame(bt.get("trade_records", []))
    if ledger.empty:
        raise RuntimeError("entry gate training ledger is empty")
    ledger["entry_signal_timestamp"] = pd.to_datetime(ledger["entry_signal_timestamp"])
    ts_to_y = {
        ts: int(float(pnl) > 0.0)
        for ts, pnl in zip(ledger["entry_signal_timestamp"], ledger["realized_net_pct"], strict=True)
    }
    active = _active(dec)
    ts = pd.to_datetime(frame["timestamp"])
    mask = active & ts.isin(set(ts_to_y))
    if int(mask.sum()) < 40:
        raise RuntimeError(f"too few executed entry rows for gate training: {int(mask.sum())}")
    x = _gate_features(frame.loc[mask].reset_index(drop=True), dec.loc[mask].reset_index(drop=True))
    y = np.asarray([ts_to_y[t] for t in ts.loc[mask]], dtype=np.int64)
    return x, y, {"ledger_trades": int(len(ledger)), "train_rows": int(len(y)), "positive_rate": float(y.mean())}


def _apply_gate(dec: pd.DataFrame, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    veto = active & (proba < float(threshold))
    for col, value in {
        "action": 0,
        "side": 0,
        "notional_exposure": 0.0,
        "position_fraction": 0.0,
        "leverage": 1.0,
        "take_profit": 0.0,
        "stop_loss": 0.0,
        "max_hold_bars": 0,
        "cooldown_bars": 0,
    }.items():
        if col in out.columns:
            out.loc[veto, col] = value
    out["entry_gate_prob"] = proba
    out["entry_gate_threshold"] = float(threshold)
    out["entry_gate_veto"] = veto.astype(int)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_with_risk()
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    train_dec = _active_decisions(train_df)
    val_dec_base = _active_decisions(val_df)
    oos_dec_base = _active_decisions(eval_df)
    x_train, y_train, label_meta = _train_labels(train_df, train_dec)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.20,
            early_stopping=False,
            random_state=6060101,
            class_weight="balanced",
        ),
    )
    model.fit(x_train, y_train)
    x_val = _gate_features(val_df, val_dec_base)
    x_oos = _gate_features(eval_df, oos_dec_base)
    val_prob = model.predict_proba(x_val)[:, 1]
    oos_prob = model.predict_proba(x_oos)[:, 1]
    rows: list[dict[str, Any]] = []
    payload: dict[str, pd.DataFrame] = {}
    active_val = _combo_metrics(val_df, val_dec_base)
    for threshold in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        val_dec = _apply_gate(val_dec_base, val_prob, threshold)
        val_costs = _combo_metrics(val_df, val_dec)
        key = f"gate{threshold:.2f}"
        rows.append({
            "candidate": key,
            "threshold": float(threshold),
            "score": float(_score(val_costs)),
            "validation": val_costs,
            "validation_veto_rows": int(val_dec["entry_gate_veto"].sum()),
            "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
        })
        payload[key] = val_dec
    feasible = [
        r for r in rows
        if float(r["validation"]["cost1"]["pnl"]) > 0.0
        and float(r["validation"]["cost2"]["pnl"]) > 0.0
        and float(r["validation"]["cost3"]["pnl"]) >= float(active_val["cost3"]["pnl"]) * 0.75
        and int(r["validation"]["cost3"]["trades"]) >= 70
    ]
    pool = feasible or rows
    pool.sort(key=lambda r: (float(r["validation"]["cost3"]["wr"]), float(r["validation"]["cost3"]["pnl"]), float(r["score"])), reverse=True)
    selected = pool[0]
    selected_val_dec = payload[str(selected["candidate"])]
    selected_oos_dec = _apply_gate(oos_dec_base, oos_prob, float(selected["threshold"]))
    selected["oos"] = _combo_metrics(eval_df, selected_oos_dec)
    selected["oos_veto_rows"] = int(selected_oos_dec["entry_gate_veto"].sum())
    selected["oos_policy_counts"] = {str(k): int(v) for k, v in selected_oos_dec["router_expert"].value_counts().to_dict().items()}
    selected_val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos_dec.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame([
        {
            "candidate": r["candidate"],
            "threshold": r["threshold"],
            "score": r["score"],
            "validation_veto_rows": r["validation_veto_rows"],
            **_flatten("val", r["validation"]),
            "validation_policy_counts": json.dumps(r["validation_policy_counts"], ensure_ascii=False),
        }
        for r in rows
    ]).to_csv(OUT_DIR / "ranking.csv", index=False)
    joblib.dump(model, OUT_DIR / "entry_gate_hgb.pkl")
    report = {
        "model_id": MODEL_ID,
        "design": "Two-stage entry gate adaptation. Stage 1 is an HGB binary gate trained only on pre-validation active executed entries from the active MoE. Stage 2 remains the existing bull/bear/chop HGB MoE planner.",
        "label_meta": label_meta,
        "selection_rule": "Validation selects highest Cost3 WR among candidates preserving at least 75% of active validation Cost3 PnL, positive Cost1/2, and >=70 Cost3 trades. OOS is evaluated once after selection.",
        "active_validation": active_val,
        "overlay": overlay,
        "selected": selected,
        "top_grid": sorted(rows, key=lambda r: float(r["score"]), reverse=True),
        "artifacts": {
            "report": str(OUT_DIR / "report.json"),
            "ranking": str(OUT_DIR / "ranking.csv"),
            "model": str(OUT_DIR / "entry_gate_hgb.pkl"),
            "validation_decisions": str(OUT_DIR / "validation_decisions.csv"),
            "oos_decisions": str(OUT_DIR / "oos_2026_decisions.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
