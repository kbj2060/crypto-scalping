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
from sklearn.metrics import balanced_accuracy_score

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
from scripts.retrain_alpha7_clean_parent_plus_omega1_layer2_20260601 import (  # noqa: E402
    LAYER2_FEATURES_JSON,
    OVERLAY_SOURCES_2025,
    OVERLAY_SOURCES_2026,
    _load_feature_list,
    _overlay_exact,
)
from scripts.train_alpha7_parent_layer2_meta_gate_20260601 import (  # noqa: E402
    PARENT_DECISION_COLS,
    _active,
    _counterfactual_net_pnl,
    _meta_frame,
)


MODEL_ID = "alpha7_cryptomamba_expert_router_mvp_20260601"
BASE_CLEAN_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_submodel_01965_cleanfunding_v1_20260529"
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
CRYPTOMAMBA_2025 = ROOT / "data/ensemble/supervised/omega1_dir3_cryptomamba_20260531/training_features_2025_omega1_dir3_cryptomamba_20260531.csv"
CRYPTOMAMBA_2026 = ROOT / "data/ensemble/supervised/omega1_dir3_cryptomamba_20260531/training_features_2026_rebuilt_omega1_dir3_cryptomamba_20260531.csv"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_cryptomamba_expert_router_mvp_20260601"

CRYPTOMAMBA_COLS = [
    "dir3_cryptomamba_h6_fl_prob",
    "dir3_cryptomamba_h6_up_prob",
    "dir3_cryptomamba_h6_dn_prob",
    "dir3_cryptomamba_h6_confidence",
    "dir3_cryptomamba_h6_side_edge",
    "dir3_cryptomamba_h6_trade_prob",
]


def _read_source(path: Path) -> pd.DataFrame:
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


def _overlay_required_features(base: pd.DataFrame, source: Path, cols: list[str], *, tag: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    src = _read_source(source)
    missing = [c for c in cols if c not in src.columns]
    if missing:
        raise RuntimeError(f"{tag}: missing required CryptoMamba columns: {missing}")
    missing_ts = base.loc[~base["timestamp"].isin(set(src["timestamp"])), "timestamp"]
    out = base.copy()
    dropped: list[dict[str, Any]] = []
    if len(missing_ts) > 0:
        head_ts = out["timestamp"].head(len(missing_ts)).reset_index(drop=True)
        tail_ts = out["timestamp"].tail(len(missing_ts)).reset_index(drop=True)
        missing_norm = missing_ts.reset_index(drop=True)
        if missing_norm.equals(head_ts):
            edge = "head"
        elif missing_norm.equals(tail_ts):
            edge = "tail"
        else:
            raise RuntimeError(f"{tag}: {source} missing non-edge timestamps: {missing_ts.head(20).tolist()}")
        dropped.append({"path": str(source), "edge": edge, "rows": int(len(missing_ts)), "first": str(missing_ts.iloc[0]), "last": str(missing_ts.iloc[-1])})
        out = out.loc[out["timestamp"].isin(set(src["timestamp"]))].reset_index(drop=True)
    before = len(out)
    out = out.merge(src[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    if len(out) != before:
        raise RuntimeError(f"{tag}: row count changed after CryptoMamba overlay: {before} -> {len(out)}")
    nan_mask = out[cols].isna().any(axis=1)
    edge = _edge_name(nan_mask)
    if edge is None and bool(nan_mask.any()):
        bad_ts = out.loc[nan_mask, "timestamp"].head(20).tolist()
        raise RuntimeError(f"{tag}: CryptoMamba produced non-edge NaN rows: {bad_ts}")
    if edge is not None:
        bad = out.loc[nan_mask, "timestamp"]
        dropped.append({"path": str(source), "edge": edge, "rows": int(len(bad)), "first": str(bad.iloc[0]), "last": str(bad.iloc[-1]), "reason": "sequence_warmup_nan"})
        out = out.loc[~nan_mask].reset_index(drop=True)
    return out, {"path": str(source), "added": cols, "dropped_edge_rows": dropped}


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_base = _read(TRAIN_CSV)
    eval_base = _read(EVAL_CSV)
    layer2 = _load_feature_list(LAYER2_FEATURES_JSON)
    train, train_overlay = _overlay_exact(train_base, OVERLAY_SOURCES_2025, layer2, tag="train_2025")
    eval_df, eval_overlay = _overlay_exact(eval_base, OVERLAY_SOURCES_2026, layer2, tag="eval_2026")
    train, train_crypto = _overlay_required_features(train, CRYPTOMAMBA_2025, CRYPTOMAMBA_COLS, tag="train_2025")
    eval_df, eval_crypto = _overlay_required_features(eval_df, CRYPTOMAMBA_2026, CRYPTOMAMBA_COLS, tag="eval_2026")
    layer2_cols = [c for c in layer2 if c in train.columns and c in eval_df.columns]
    core_prefixes = ("ai_", "tide_", "chronos_", "regime3_", "m7_zigzag_", "dir3_")
    core = [c for c in layer2_cols if c.startswith(core_prefixes)]
    router_cols = list(dict.fromkeys([*core, *CRYPTOMAMBA_COLS]))
    return train, eval_df, {
        "train_overlay": train_overlay,
        "eval_overlay": eval_overlay,
        "train_cryptomamba": train_crypto,
        "eval_cryptomamba": eval_crypto,
        "router_cols": router_cols,
    }


def _defensive_decision(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    out.loc[active, "notional_exposure"] = pd.to_numeric(out.loc[active, "notional_exposure"], errors="raise") * 0.50
    out.loc[active, "position_fraction"] = pd.to_numeric(out.loc[active, "position_fraction"], errors="raise") * 0.50
    out.loc[active, "take_profit"] = pd.to_numeric(out.loc[active, "take_profit"], errors="raise") * 0.75
    out.loc[active, "stop_loss"] = pd.to_numeric(out.loc[active, "stop_loss"], errors="raise") * 0.75
    hold = pd.to_numeric(out.loc[active, "max_hold_bars"], errors="raise").to_numpy(dtype=np.float64)
    out.loc[active, "max_hold_bars"] = np.maximum(1, np.ceil(hold * 0.50)).astype(int)
    out.loc[active, "expert_policy"] = "defensive_resize"
    out.loc[~active, "expert_policy"] = "cash"
    return out


def _cash_decision(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    out.loc[active, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[active, "leverage"] = 1.0
    out["expert_policy"] = "cash"
    return out


def _apply_router(dec: pd.DataFrame, cls: np.ndarray, prob: np.ndarray, *, min_conf: float) -> pd.DataFrame:
    base = dec.copy().reset_index(drop=True)
    defensive = _defensive_decision(base)
    cash = _cash_decision(base)
    out = base.copy()
    decision_cols = list(base.columns)
    active = _active(base)
    chosen = np.asarray(cls, dtype=np.int64).copy()
    low_conf = np.asarray(prob, dtype=np.float64) < float(min_conf)
    chosen[low_conf] = 0
    use_def = active & (chosen == 1)
    use_cash = active & (chosen == 2)
    out.loc[use_def, decision_cols] = defensive.loc[use_def, decision_cols].to_numpy()
    out.loc[use_cash, decision_cols] = cash.loc[use_cash, decision_cols].to_numpy()
    out["expert_policy"] = "cash"
    out.loc[active & (chosen == 0), "expert_policy"] = "trend_base"
    out.loc[use_def, "expert_policy"] = "defensive_resize"
    out["router_class"] = chosen
    out["router_confidence"] = prob
    return out


def _score(costs: dict[str, Any]) -> float:
    c3 = costs["cost3"]
    if int(c3["trades"]) < 20:
        return -1e9
    return float(c3["pnl"] / max(abs(float(c3["mdd"])), 1e-9))


def _grid_row(name: str, min_conf: float | None, costs: dict[str, Any], policy_counts: dict[str, int]) -> dict[str, Any]:
    return {
        "candidate": name,
        "min_conf": None if min_conf is None else float(min_conf),
        "score": float(_score(costs)),
        "policy_counts": {str(k): int(v) for k, v in policy_counts.items()},
        "costs": costs,
    }


def _flatten_costs(prefix: str, costs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cost_name in ["cost1", "cost2", "cost3"]:
        c = costs[cost_name]
        out[f"{prefix}_{cost_name}_pnl"] = float(c["pnl"])
        out[f"{prefix}_{cost_name}_mdd"] = float(c["mdd"])
        out[f"{prefix}_{cost_name}_trades"] = int(c["trades"])
        out[f"{prefix}_{cost_name}_wr"] = float(c["wr"])
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay_report = _load_frames()
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    primary = joblib.load(BASE_CLEAN_DIR / "primary_no_tp/parent.pkl")
    fallback = joblib.load(BASE_CLEAN_DIR / "fallback_v2_tp/parent.pkl")
    train_dec = _combine_primary_fallback(_predict_scaled(primary, train_df, None), _predict_scaled(fallback, train_df, None)).reset_index(drop=True)
    val_dec = _combine_primary_fallback(_predict_scaled(primary, val_df, None), _predict_scaled(fallback, val_df, None)).reset_index(drop=True)
    oos_dec = _combine_primary_fallback(_predict_scaled(primary, eval_df, None), _predict_scaled(fallback, eval_df, None)).reset_index(drop=True)

    fee = float(primary["config"]["fee"]) * 3.0
    slip = float(primary["config"]["slip"]) * 3.0
    train_def = _defensive_decision(train_dec)
    pnl_base = _counterfactual_net_pnl(train_df, train_dec, fee=fee, slip=slip)
    pnl_def = _counterfactual_net_pnl(train_df, train_def, fee=fee, slip=slip)
    active = _active(train_dec) & np.isfinite(pnl_base) & np.isfinite(pnl_def)
    scores = np.vstack([pnl_base[active], pnl_def[active], np.zeros(int(active.sum()), dtype=np.float64)]).T
    y = np.argmax(scores, axis=1).astype(np.int64)

    router_cols = overlay_report["router_cols"]
    x_all = _meta_frame(train_df, train_dec, router_cols).loc[active].reset_index(drop=True)
    if len(np.unique(y)) < 2:
        raise RuntimeError(f"router label collapsed: {dict(zip(*np.unique(y, return_counts=True)))}")
    cut = int(len(x_all) * 0.80)
    clf = HistGradientBoostingClassifier(
        max_iter=450,
        max_leaf_nodes=31,
        learning_rate=0.035,
        l2_regularization=0.20,
        min_samples_leaf=35,
        class_weight="balanced",
        random_state=6060122,
    )
    clf.fit(x_all.iloc[:cut], y[:cut])
    holdout_pred = clf.predict(x_all.iloc[cut:])
    fit_metrics = {
        "rows": int(len(x_all)),
        "train_rows": int(cut),
        "holdout_rows": int(len(x_all) - cut),
        "label_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "holdout_bacc": float(balanced_accuracy_score(y[cut:], holdout_pred)) if len(np.unique(y[cut:])) > 1 else None,
    }
    feature_cols = list(x_all.columns)
    joblib.dump({"model_id": MODEL_ID, "model": clf, "feature_cols": feature_cols, "router_cols": router_cols}, OUT_DIR / "expert_router_hgb.joblib")

    x_val = _meta_frame(val_df, val_dec, router_cols)
    x_oos = _meta_frame(eval_df, oos_dec, router_cols)
    val_prob = clf.predict_proba(x_val)
    oos_prob = clf.predict_proba(x_oos)
    val_cls = np.argmax(val_prob, axis=1)
    oos_cls = np.argmax(oos_prob, axis=1)
    val_conf = np.max(val_prob, axis=1)
    oos_conf = np.max(oos_prob, axis=1)

    baseline_val = _combo_metrics(val_df, val_dec)
    baseline_oos = _combo_metrics(eval_df, oos_dec)
    grid: list[dict[str, Any]] = [
        _grid_row("clean_parent_baseline", None, baseline_val, {"trend_base": int(_active(val_dec).sum()), "cash": int((~_active(val_dec)).sum())})
    ]
    for min_conf in [0.0, 0.34, 0.40, 0.50, 0.60, 0.70, 0.80]:
        routed = _apply_router(val_dec, val_cls, val_conf, min_conf=min_conf)
        costs = _combo_metrics(val_df, routed)
        policy_counts = routed["expert_policy"].value_counts(dropna=False).to_dict()
        grid.append(_grid_row("cryptomamba_expert_router", min_conf, costs, policy_counts))
    grid.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = grid[0]
    if selected["min_conf"] is None:
        selected_val_dec = val_dec.copy().reset_index(drop=True)
        selected_oos_dec = oos_dec.copy().reset_index(drop=True)
        selected_val_dec["expert_policy"] = np.where(_active(selected_val_dec), "trend_base", "cash")
        selected_oos_dec["expert_policy"] = np.where(_active(selected_oos_dec), "trend_base", "cash")
        selected_val_dec["router_class"] = 0
        selected_oos_dec["router_class"] = 0
        selected_val_dec["router_confidence"] = 1.0
        selected_oos_dec["router_confidence"] = 1.0
    else:
        selected_val_dec = _apply_router(val_dec, val_cls, val_conf, min_conf=float(selected["min_conf"]))
        selected_oos_dec = _apply_router(oos_dec, oos_cls, oos_conf, min_conf=float(selected["min_conf"]))
    selected_val = _combo_metrics(val_df, selected_val_dec)
    selected_oos = _combo_metrics(eval_df, selected_oos_dec)

    selected_val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos_dec.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame([
        {
            "candidate": r["candidate"],
            "min_conf": r["min_conf"],
            "score": r["score"],
            **_flatten_costs("val", r["costs"]),
            "policy_counts": json.dumps(r["policy_counts"], ensure_ascii=False),
        }
        for r in grid
    ]).to_csv(OUT_DIR / "grid.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "design": "MVP expert router: frozen clean Alpha7 parent/fallback decisions are routed into trend_base, defensive_resize, or cash. Router features include parent decision state, Omega1 Layer2 core, and exact-joined dir3 CryptoMamba h6 outputs.",
        "assumptions": [
            "This is the fast MVP of the requested CryptoMamba-router/xLSTM-expert/DSAC-risk idea.",
            "It does not train new xLSTM experts or a new DSAC selector yet; it first tests whether a router can improve existing expert allocation.",
            "2026 is evaluation only; confidence threshold is selected on 2025 validation.",
        ],
        "expert_contract": {"0": "trend_base", "1": "defensive_resize", "2": "cash"},
        "fit_metrics": fit_metrics,
        "overlay_report": overlay_report,
        "feature_count": len(feature_cols),
        "router_cols": router_cols,
        "baseline": {"validation": baseline_val, "oos": baseline_oos},
        "selected": {
            "candidate": selected["candidate"],
            "min_conf": selected["min_conf"],
            "validation": selected_val,
            "oos": selected_oos,
            "validation_policy_counts": {str(k): int(v) for k, v in selected_val_dec["expert_policy"].value_counts(dropna=False).to_dict().items()},
            "oos_policy_counts": {str(k): int(v) for k, v in selected_oos_dec["expert_policy"].value_counts(dropna=False).to_dict().items()},
        },
        "top_grid": grid[:10],
        "artifacts": {
            "model": str(OUT_DIR / "expert_router_hgb.joblib"),
            "report": str(OUT_DIR / "report.json"),
            "grid": str(OUT_DIR / "grid.csv"),
            "validation_decisions": str(OUT_DIR / "validation_decisions.csv"),
            "oos_decisions": str(OUT_DIR / "oos_2026_decisions.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": report["selected"]}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
