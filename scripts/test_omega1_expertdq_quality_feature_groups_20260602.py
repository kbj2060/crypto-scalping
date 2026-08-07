#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combo_metrics, _json_default
from scripts.eval_alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601 import _apply_scale
from scripts.retrain_alpha7_active_max_feature_contract_moe_20260601 import _load_frames_max
from scripts.train_omega1_direction_head_raw_context_groups_20260602 import GROUP_CANDIDATES
import scripts.train_omega1_regime3_expert_direction_head_volpca_20260602 as hard
import scripts.train_omega1_regime3_routed_expert_direction_quality_20260602 as dq


MODEL_ID = "omega1_expertdq_quality_feature_groups_20260602"
BASE_VARIANT = "soft_floor_0p10"
EXPERTDQ_DIR = ROOT / "tmp/causal_regen_20260516/omega1_regime3_routed_expert_direction_quality_20260602"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"

ACTION_CASH = 0
ACTION_LONG = 1
ACTION_SHORT = 2

RISK_TEMPLATE = {
    "notional": 0.45,
    "leverage": 2.0,
    "take_profit": 0.026,
    "stop_loss": 0.014,
    "max_hold": 72,
    "cooldown": 6,
}
EXPERT_SCALES = {"bull": 0.75, "bear": 0.90, "chop": 0.90}

DIR_AUX_COLS = [
    "direction_p_cash",
    "direction_p_long",
    "direction_p_short",
    "direction_confidence",
    "direction_side_edge",
    "direction_trade_prob",
    "direction_action",
]
ROUTER_AUX_COLS = [
    "router_confidence",
    "router_margin",
]
FORBIDDEN_EXACT = {"tp_sl_action_score", "m7_hdb_label", "m7_target_hold"}
FORBIDDEN_PREFIXES = ("clean_regime4_", "regime4_pred_", "regime3_pred_", "teacher_", "a5dir_")
FORBIDDEN_TOKENS = ("label", "target", "future", "pnl", "action_score")


def _blocked(col: str) -> bool:
    lower = col.lower()
    return col in FORBIDDEN_EXACT or any(col.startswith(p) for p in FORBIDDEN_PREFIXES) or any(t in lower for t in FORBIDDEN_TOKENS)


def _dedupe(cols: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for col in cols:
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out


def _assert_feature_cols(frame: pd.DataFrame, cols: list[str], name: str) -> None:
    bad = [c for c in cols if _blocked(c)]
    if bad:
        raise RuntimeError(f"{name}: forbidden columns selected: {bad}")
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{name}: missing columns: {missing}")
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise RuntimeError(f"{name}: non-numeric columns: {non_numeric}")
    arr = frame[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        counts = {c: int((~np.isfinite(frame[c].to_numpy(dtype=np.float64))).sum()) for c in cols}
        counts = {k: v for k, v in counts.items() if v}
        raise RuntimeError(f"{name}: non-finite values: {counts}")


def _read_preds(variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = EXPERTDQ_DIR / variant
    oof = base / f"training_features_2025_{variant}_omega1_regime3_expertdq_oof_20260602.csv"
    oos = base / f"training_features_2026_rebuilt_{variant}_omega1_regime3_expertdq_20260602.csv"
    if not oof.exists() or not oos.exists():
        raise FileNotFoundError(f"missing base prediction CSVs for {variant}")
    return pd.read_csv(oof), pd.read_csv(oos)


def _rename_pred_cols(pred: pd.DataFrame, *, oof: bool) -> pd.DataFrame:
    prefix = "omega1_regime3_expertdq_oof_" if oof else "omega1_regime3_expertdq_"
    mapping = {
        f"{prefix}dir_p_cash": "direction_p_cash",
        f"{prefix}dir_p_long": "direction_p_long",
        f"{prefix}dir_p_short": "direction_p_short",
        f"{prefix}dir_confidence": "direction_confidence",
        f"{prefix}dir_side_edge": "direction_side_edge",
        f"{prefix}dir_trade_prob": "direction_trade_prob",
        f"{prefix}dir_action": "direction_action",
        f"{prefix}router_confidence": "router_confidence",
        f"{prefix}router_margin": "router_margin",
        f"{prefix}router_expert": "router_expert",
    }
    missing = [c for c in mapping if c not in pred.columns]
    if missing:
        raise RuntimeError(f"missing prediction cols: {missing}")
    out = pred[["timestamp", *mapping]].rename(columns=mapping)
    return out


def _align(frame: pd.DataFrame, pred: pd.DataFrame, name: str, *, require_all_pred_rows: bool = True) -> pd.DataFrame:
    left = frame.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], errors="raise")
    right = pred.copy()
    right["timestamp"] = pd.to_datetime(right["timestamp"], errors="raise")
    if right["timestamp"].duplicated().any():
        raise RuntimeError(f"{name}: duplicate prediction timestamps")
    merged = left.merge(right, on="timestamp", how="inner", validate="one_to_one").reset_index(drop=True)
    if require_all_pred_rows and len(merged) != len(right):
        raise RuntimeError(f"{name}: timestamp alignment lost rows: merged={len(merged)} pred={len(right)}")
    if len(merged) == 0:
        raise RuntimeError(f"{name}: empty timestamp intersection")
    return merged


def _quality_feature_variants(frame: pd.DataFrame) -> dict[str, list[str]]:
    base_quality = _dedupe([*hard.volpca.BASE_COLS, *[f"pca_volatility_{i:02d}" for i in range(1, 7)], *DIR_AUX_COLS, *ROUTER_AUX_COLS])
    groups = {k: [c for c in v if c in frame.columns and not _blocked(c)] for k, v in GROUP_CANDIDATES.items()}
    route_cols = [*hard.ROUTE_COLS, *hard.ROUTE_EXTRA_COLS]
    regime_sidecars = [
        c
        for c in [
            *route_cols,
            "regime3_stability_h6_score",
            "regime3_transition_h6_risk_prob",
            "regime3_transition_h6_risk_pred",
            "regime3_churn_h6_risk_score",
            "regime3_cmamba_h6_sidecar_bull_prob",
            "regime3_cmamba_h6_sidecar_bear_prob",
            "regime3_cmamba_h6_sidecar_chop_prob",
            "regime3_cmamba_h6_sidecar_confidence",
            "regime3_cmamba_h6_sidecar_transition_prob",
            "regime3_cmamba_h6_sidecar_stability_score",
        ]
        if c in frame.columns and not _blocked(c)
    ]
    variants = {
        "baseline_contract": base_quality,
        "minimal_dir_router": [*DIR_AUX_COLS, *ROUTER_AUX_COLS],
        "dir_router_regime_risk": [*DIR_AUX_COLS, *ROUTER_AUX_COLS, *regime_sidecars],
        "dir_router_volume_flow": [*DIR_AUX_COLS, *ROUTER_AUX_COLS, *groups.get("volume_flow", [])],
        "dir_router_liquidity": [*DIR_AUX_COLS, *ROUTER_AUX_COLS, *groups.get("liquidity_execution_spread_proxy", [])],
        "dir_router_funding": [*DIR_AUX_COLS, *ROUTER_AUX_COLS, *groups.get("funding_context", [])],
        "dir_router_session": [*DIR_AUX_COLS, *ROUTER_AUX_COLS, *groups.get("session_context", [])],
        "dir_router_raw_volatility": [*DIR_AUX_COLS, *ROUTER_AUX_COLS, *groups.get("volatility_context", [])],
        "dir_router_all_context": [*DIR_AUX_COLS, *ROUTER_AUX_COLS, *[c for cols in groups.values() for c in cols]],
        "baseline_plus_regime_risk": [*base_quality, *regime_sidecars],
        "baseline_plus_all_context": [*base_quality, *[c for cols in groups.values() for c in cols]],
    }
    return {k: _dedupe(v) for k, v in variants.items()}


def _attach_vol_pca(frame: pd.DataFrame, transformer: Any) -> pd.DataFrame:
    pca_frame = hard._features_with_transform(frame, transformer)
    pca_cols = [c for c in pca_frame.columns if c.startswith("pca_volatility_")]
    if not pca_cols:
        raise RuntimeError("VolPCA transform produced no pca_volatility_* columns")
    out = frame.reset_index(drop=True).copy()
    for col in pca_cols:
        out[col] = pca_frame[col].to_numpy(dtype=np.float64)
    return out


def _fit_quality_models(
    train: pd.DataFrame,
    feature_cols: list[str],
    *,
    floor: float,
    seed: int,
) -> dict[str, CatBoostClassifier]:
    y = train["zigzag_action"].to_numpy(dtype=np.int64)
    probs = train[hard.ROUTE_COLS].to_numpy(dtype=np.float64)
    models: dict[str, CatBoostClassifier] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        sample_weight = float(floor) + probs[:, idx]
        weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64) * sample_weight
        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            iterations=600,
            depth=5,
            learning_rate=0.035,
            l2_leaf_reg=6.0,
            random_seed=seed + idx,
            od_type="Iter",
            od_wait=50,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(train[feature_cols], y, sample_weight=weights)
        models[expert] = model
    return models


def _predict_quality(models: dict[str, CatBoostClassifier], frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    expert_pred = {expert: dq.base._proba3(model, frame[feature_cols]) for expert, model in models.items()}
    return dq._routed_proba(expert_pred, hard._route_id(frame))


def _apply_quality(direction_proba: np.ndarray, quality_proba: np.ndarray, threshold: float) -> np.ndarray:
    action = np.argmax(direction_proba, axis=1).astype(np.int64)
    q = quality_proba[np.arange(len(action)), action]
    final = action.copy()
    final[(action != ACTION_CASH) & (q < float(threshold))] = ACTION_CASH
    return final


def _metrics(y: np.ndarray, direction_proba: np.ndarray, quality_proba: np.ndarray, threshold: float) -> dict[str, Any]:
    final = _apply_quality(direction_proba, quality_proba, threshold)
    proba = direction_proba.copy()
    veto = final == ACTION_CASH
    proba[veto] = 0.0
    proba[veto, 0] = 1.0
    return dq.base._metrics(y, proba)


def _select_threshold(train: pd.DataFrame, quality_proba: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    y = train["zigzag_action"].to_numpy(dtype=np.int64)
    direction_proba = train[["direction_p_cash", "direction_p_long", "direction_p_short"]].to_numpy(dtype=np.float64)
    direction_metrics = dq.base._metrics(y, direction_proba)
    min_trades = max(1, int(direction_metrics["proxy_trades"] * 0.70))
    rows = []
    for threshold in dq.QUALITY_THRESHOLDS:
        m = _metrics(y, direction_proba, quality_proba, float(threshold))
        rows.append({"threshold": float(threshold), "metrics": m, "min_trades": int(min_trades)})
    eligible = [r for r in rows if int(r["metrics"]["proxy_trades"]) >= min_trades] or rows
    eligible.sort(
        key=lambda r: (
            float(r["metrics"]["balanced_accuracy"]),
            float(r["metrics"]["proxy_wr"] or 0.0),
            int(r["metrics"]["proxy_trades"]),
        ),
        reverse=True,
    )
    return float(eligible[0]["threshold"]), rows


def _to_decisions(frame: pd.DataFrame, quality_proba: np.ndarray, threshold: float) -> pd.DataFrame:
    direction_proba = frame[["direction_p_cash", "direction_p_long", "direction_p_short"]].to_numpy(dtype=np.float64)
    action = _apply_quality(direction_proba, quality_proba, threshold)
    active = action != ACTION_CASH
    side = np.where(action == ACTION_LONG, 1, np.where(action == ACTION_SHORT, -1, 0)).astype(np.int64)
    q = quality_proba[np.arange(len(action)), np.argmax(direction_proba, axis=1).astype(np.int64)]
    dec = pd.DataFrame(
        {
            "action": action,
            "side": side,
            "notional_exposure": np.where(active, float(RISK_TEMPLATE["notional"]), 0.0),
            "leverage": np.where(active, float(RISK_TEMPLATE["leverage"]), 1.0),
            "position_fraction": np.where(active, float(RISK_TEMPLATE["notional"]), 0.0),
            "take_profit": np.where(active, float(RISK_TEMPLATE["take_profit"]), 0.0),
            "stop_loss": np.where(active, float(RISK_TEMPLATE["stop_loss"]), 0.0),
            "max_hold_bars": np.where(active, int(RISK_TEMPLATE["max_hold"]), 0).astype(np.int64),
            "cooldown_bars": np.where(active, int(RISK_TEMPLATE["cooldown"]), 0).astype(np.int64),
            "quality_score": q.astype(np.float64),
            "confidence": frame["direction_confidence"].to_numpy(dtype=np.float64),
            "router_expert": frame["router_expert"].astype(str).replace({"chop": "chop_expert"}).to_numpy(),
        }
    )
    return _apply_scale(dec, **EXPERT_SCALES)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, _overlay = _load_frames_max()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    full_train = hard._build_frame(2025)
    full_oos = hard._build_frame(2026)
    vol_transformer = hard.volpca.VolPca(6).fit(full_train)
    oof_pred_raw, oos_pred_raw = _read_preds(BASE_VARIANT)
    oof_pred = _rename_pred_cols(oof_pred_raw, oof=True)
    oos_pred = _rename_pred_cols(oos_pred_raw, oof=False)
    train = _attach_vol_pca(_align(full_train, oof_pred, "train_oof"), vol_transformer)
    oos = _attach_vol_pca(_align(full_oos, _rename_pred_cols(oos_pred_raw, oof=False), "oos"), vol_transformer)
    val_df = val_df.copy()
    eval_df = eval_df.copy()
    val_df["timestamp"] = pd.to_datetime(val_df["timestamp"], errors="raise")
    eval_df["timestamp"] = pd.to_datetime(eval_df["timestamp"], errors="raise")
    val_ts = set(pd.to_datetime(val_df["timestamp"], errors="raise"))
    eval_ts = set(pd.to_datetime(eval_df["timestamp"], errors="raise"))
    val_feature_frame = full_train.loc[pd.to_datetime(full_train["timestamp"], errors="raise").isin(val_ts)].reset_index(drop=True)
    oos_feature_frame = full_oos.loc[pd.to_datetime(full_oos["timestamp"], errors="raise").isin(eval_ts)].reset_index(drop=True)
    val_eval = _attach_vol_pca(_align(val_feature_frame, oof_pred, "validation_replay", require_all_pred_rows=False), vol_transformer)
    oos_eval = _attach_vol_pca(_align(oos_feature_frame, oos_pred, "oos_replay", require_all_pred_rows=False), vol_transformer)
    val_frame_common = val_df.merge(val_eval[["timestamp"]], on="timestamp", how="inner", validate="one_to_one").reset_index(drop=True)
    oos_frame_common = eval_df.merge(oos_eval[["timestamp"]], on="timestamp", how="inner", validate="one_to_one").reset_index(drop=True)
    if len(val_frame_common) != len(val_eval) or len(oos_frame_common) != len(oos_eval):
        raise RuntimeError("market frame / prediction frame timestamp mismatch")

    variants = _quality_feature_variants(train)
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for i, (name, cols) in enumerate(variants.items(), start=1):
        _assert_feature_cols(train, cols, f"{name} train")
        _assert_feature_cols(oos, cols, f"{name} oos")
        _assert_feature_cols(oos_eval, cols, f"{name} oos_replay")
        models = _fit_quality_models(train, cols, floor=0.10, seed=20260620 + i * 10)
        q_train = _predict_quality(models, train, cols)
        q_val = _predict_quality(models, val_eval, cols)
        q_oos = _predict_quality(models, oos, cols)
        q_oos_eval = _predict_quality(models, oos_eval, cols)
        threshold, threshold_grid = _select_threshold(train, q_train)
        y_oos = oos["zigzag_action"].to_numpy(dtype=np.int64)
        direction_oos = oos[["direction_p_cash", "direction_p_long", "direction_p_short"]].to_numpy(dtype=np.float64)
        filtered_oos_metrics = _metrics(y_oos, direction_oos, q_oos, threshold)
        dec_val = _to_decisions(val_eval, q_val, threshold)
        dec_oos = _to_decisions(oos_eval, q_oos_eval, threshold)
        val_costs = _combo_metrics(val_frame_common, dec_val)
        oos_costs = _combo_metrics(oos_frame_common, dec_oos)
        report = {
            "variant": name,
            "feature_count": int(len(cols)),
            "feature_cols": cols,
            "selected_quality_threshold": float(threshold),
            "threshold_grid": threshold_grid,
            "filtered_oos_metrics": filtered_oos_metrics,
            "validation": val_costs,
            "oos": oos_costs,
        }
        reports[name] = report
        row = {
            "variant": name,
            "feature_count": int(len(cols)),
            "selected_quality_threshold": float(threshold),
            "filtered_oos_bacc": filtered_oos_metrics["balanced_accuracy"],
            "filtered_oos_auc": filtered_oos_metrics["ovr_auc"],
            "filtered_oos_proxy_wr": filtered_oos_metrics["proxy_wr"],
            "filtered_oos_proxy_trades": filtered_oos_metrics["proxy_trades"],
        }
        for split, costs in [("val", val_costs), ("oos", oos_costs)]:
            for mult in (1, 2, 3):
                c = costs[f"cost{mult}"]
                row[f"{split}_cost{mult}_pnl"] = float(c["pnl"])
                row[f"{split}_cost{mult}_mdd"] = float(c["mdd"])
                row[f"{split}_cost{mult}_trades"] = int(c["trades"])
                row[f"{split}_cost{mult}_wr"] = float(c["wr"])
        rows.append(row)

    rows.sort(key=lambda r: (float(r["oos_cost3_pnl"]), float(r["oos_cost3_wr"])), reverse=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "ranking.csv", index=False)
    out = {
        "model_id": MODEL_ID,
        "base_variant": BASE_VARIANT,
        "design": "Direction Head is fixed to the existing Regime3 ExpertDQ soft_floor_0p10 outputs. Only Quality CatBoost Head inputs are retrained by Omega1 feature-contract groups, then replayed through the same fixed Omega1 risk template.",
        "risk_template": RISK_TEMPLATE,
        "expert_scales": EXPERT_SCALES,
        "selected_by_oos_cost3_pnl": rows[0]["variant"],
        "ranking": rows,
        "variants": reports,
        "artifacts": {
            "ranking": str(OUT_DIR / "ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"selected": rows[0], "ranking": rows}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
