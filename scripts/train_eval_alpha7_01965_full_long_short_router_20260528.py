#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    FullyLearnedGovernorConfig,
    build_training_set,
    predict_policy_frame,
    train_policy,
)
from scripts import runtime_retest_alpha7_1_01965_decontam_20260528 as decontam  # noqa: E402
from scripts import sweep_decontam_deep_alpha_controls_20260528 as sweep  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    TP_COL,
    _combine_primary_fallback,
    _close,
    _json_default,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.retrain_alpha7_1_01965_tp_sl_decontam_20260528 import (  # noqa: E402
    DERIVABLE_FEATURES,
    EVAL_CSV,
    FORBIDDEN_PREFIXES,
    REQUIRED_PREFIX,
    TRAIN_CSV,
)

MODEL_ID = "alpha7_01965_full_long_short_router_20260528"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
CANDIDATE_DIR = ROOT / "data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528"
PRIMARY_PARENT = CANDIDATE_DIR / "primary_parent.pkl"
FALLBACK_PARENT = CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl"


def _forbidden_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if c.startswith(FORBIDDEN_PREFIXES)]


def _assert_clean_frame(df: pd.DataFrame, *, name: str) -> None:
    bad = _forbidden_cols(list(df.columns))
    if bad:
        raise RuntimeError(f"{name} contains forbidden legacy regime columns: {bad[:20]}")
    if TP_COL not in df.columns:
        raise RuntimeError(f"{name} missing required {TP_COL}")
    if not any(c.startswith(REQUIRED_PREFIX) for c in df.columns):
        raise RuntimeError(f"{name} missing required {REQUIRED_PREFIX} columns")


def _assert_feature_cols(df: pd.DataFrame, cols: list[str], *, name: str) -> None:
    bad = _forbidden_cols(cols)
    if bad:
        raise RuntimeError(f"{name} feature contract contains forbidden columns: {bad[:20]}")
    missing = [c for c in cols if c not in df.columns and c not in DERIVABLE_FEATURES]
    if missing:
        raise RuntimeError(f"{name} missing feature columns: {missing[:30]}")


def _side_y(y: dict[str, np.ndarray], action_keep: int) -> dict[str, np.ndarray]:
    out = {k: np.asarray(v).copy() for k, v in y.items()}
    action = np.asarray(out["action"], dtype=np.int64)
    keep = action == int(action_keep)
    out["action"][~keep] = ACTION_CASH
    for key in ("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown"):
        out[key][~keep] = 0
    out["quality"][~keep] = 0.0
    return out


def _train_side_parent(
    *,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    action_keep: int,
    seed: int,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "parent.pkl"
    summary_path = out_dir / "summary.json"
    if model_path.exists() and summary_path.exists():
        return joblib.load(model_path), json.loads(summary_path.read_text(encoding="utf-8"))

    ref = joblib.load(PRIMARY_PARENT)
    cfg = FullyLearnedGovernorConfig(**dict(ref["config"]))
    x, y, meta = build_training_set(
        train_df,
        cfg=cfg,
        stride_bars=6,
        batch_size=512,
        feature_cols=feature_cols,
    )
    y_side = _side_y(y, action_keep)
    parent = train_policy(x, y_side, cfg=cfg, random_state=int(seed), feature_cols=feature_cols)
    joblib.dump(parent, model_path)
    summary = {
        "model_id": MODEL_ID,
        "specialist": "LONG" if action_keep == ACTION_LONG else "SHORT",
        "feature_count": len(feature_cols),
        "train_meta": meta,
        "label_distribution": parent.get("label_distribution", {}),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return parent, summary


def _active(dec: pd.DataFrame) -> pd.Series:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(int)
    return (action != ACTION_CASH) & (side != 0)


def _sanitize_side(dec: pd.DataFrame, side_keep: int) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    side = pd.to_numeric(out["side"], errors="coerce").fillna(0).astype(int)
    bad = side != int(side_keep)
    out.loc[bad, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[bad, "leverage"] = 1.0
    return out


def _predict_parent_pair(
    *,
    primary: dict[str, Any],
    fallback: dict[str, Any],
    df: pd.DataFrame,
    side_keep: int,
) -> pd.DataFrame:
    p = predict_policy_frame(primary, df, close=_close(df), strict=False).reset_index(drop=True)
    f = predict_policy_frame(fallback, df, close=_close(df), strict=False).reset_index(drop=True)
    combo = _combine_primary_fallback(_sanitize_side(p, side_keep), _sanitize_side(f, side_keep))
    return _sanitize_side(combo, side_keep)


def _router_features(
    df: pd.DataFrame,
    *,
    long_dec: pd.DataFrame,
    short_dec: pd.DataFrame,
    q: np.ndarray,
) -> pd.DataFrame:
    long_active = _active(long_dec).to_numpy(dtype=np.float64)
    short_active = _active(short_dec).to_numpy(dtype=np.float64)
    out = pd.DataFrame(index=df.index)
    out["long_active"] = long_active
    out["short_active"] = short_active
    for prefix, dec in (("long", long_dec), ("short", short_dec)):
        out[f"{prefix}_confidence"] = pd.to_numeric(dec["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        out[f"{prefix}_quality"] = pd.to_numeric(dec["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        out[f"{prefix}_notional"] = pd.to_numeric(dec["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        out[f"{prefix}_tp"] = pd.to_numeric(dec["take_profit"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        out[f"{prefix}_sl"] = pd.to_numeric(dec["stop_loss"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["deep_q_long"] = np.asarray(q[:, 0], dtype=np.float64)
    out["deep_q_short"] = np.asarray(q[:, 1], dtype=np.float64)
    out["deep_q_margin"] = out["deep_q_long"] - out["deep_q_short"]
    for col in [
        "tp_sl_action_score",
        "clean_regime4_state24_sticky090_v2_bull_prob",
        "clean_regime4_state24_sticky090_v2_bear_prob",
        "clean_regime4_state24_sticky090_v2_chop_prob",
        "clean_regime4_state24_sticky090_v2_whipsaw_prob",
        "regime4_pred_bull_prob",
        "regime4_pred_bear_prob",
        "regime4_pred_chop_prob",
        "regime4_pred_whipsaw_prob",
        "net_taker_ratio",
        "taker_acceleration",
        "ofi_acceleration",
        "ai_flow_pressure",
        "garch_vol_z",
        "volatility_z",
    ]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _router_label_frame(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ref = joblib.load(PRIMARY_PARENT)
    cfg = FullyLearnedGovernorConfig(**dict(ref["config"]))
    _, y, _ = build_training_set(df, cfg=cfg, stride_bars=6, batch_size=512, feature_cols=feature_cols)
    h = int(cfg.max_train_horizon_bars)
    idx = np.arange(0, max(0, len(df) - h - 1), 6, dtype=np.int64)
    labels = np.asarray(y["action"], dtype=np.int64)
    return idx, labels


def _side_fill_price(px: float, side: int, *, entry: bool, slip: float) -> float:
    if entry:
        return float(px) * (1.0 + float(slip) * float(side))
    return float(px) * (1.0 - float(slip) * float(side))


def _counterfactual_stack_return(
    *,
    df: pd.DataFrame,
    close: np.ndarray,
    q: np.ndarray,
    dec: pd.DataFrame,
    i: int,
    side_keep: int,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    variant: sweep.Variant,
    cost_mult: int = 3,
) -> float:
    if i + 2 >= len(df):
        return 0.0
    fee = float(stack["fee"]) * float(cost_mult)
    slip = float(stack["slip"]) * float(cost_mult)
    overlay = sweep.precision._overlay(stack["overlay"], cfg)
    side = int(side_keep)
    dec_row = dec.iloc[int(i)]
    active = int(dec_row.action) != ACTION_CASH and int(dec_row.side) == side
    if active:
        notional = float(dec_row.notional_exposure)
        take_profit = float(dec_row.take_profit)
        stop_loss = float(dec_row.stop_loss)
        max_hold = int(dec_row.max_hold_bars)
    else:
        if i < 60:
            return 0.0
        ql = float(q[int(i), 0])
        qs = float(q[int(i), 1])
        deep_side = 1 if ql > qs else -1
        if deep_side != side:
            return 0.0
        edge = max(ql, qs)
        margin = abs(ql - qs)
        if edge < float(overlay.edge_th) or margin < float(overlay.margin_th):
            return 0.0
        notional = float(overlay.notional)
        take_profit = float(overlay.base_tp)
        stop_loss = float(overlay.base_sl)
        max_hold = int(overlay.base_hold)
    if notional <= 0.0 or max_hold <= 0:
        return 0.0

    entry_i = int(i + 1)
    entry_px = _side_fill_price(float(close[entry_i]), side, entry=True, slip=slip)
    last_i = int(min(len(close) - 1, entry_i + max_hold))
    exit_i = last_i
    for j in range(entry_i + 1, last_i + 1):
        mark_px = _side_fill_price(float(close[j]), side, entry=False, slip=slip)
        raw = (mark_px - entry_px) / max(entry_px, 1e-12) if side > 0 else (entry_px - mark_px) / max(entry_px, 1e-12)
        pnl = raw * notional
        if take_profit > 0.0 and pnl >= float(take_profit):
            exit_i = j
            break
        if stop_loss > 0.0 and pnl <= -abs(float(stop_loss)):
            exit_i = j
            break
    exit_px = _side_fill_price(float(close[exit_i]), side, entry=False, slip=slip)
    raw = (exit_px - entry_px) / max(entry_px, 1e-12) if side > 0 else (entry_px - exit_px) / max(entry_px, 1e-12)
    return float(raw * notional - 2.0 * fee * notional)


def _stack_pnl_router_labels(
    *,
    train_df: pd.DataFrame,
    long_dec: pd.DataFrame,
    short_dec: pd.DataFrame,
    q: np.ndarray,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    idx, inherited = _router_label_frame(train_df, feature_cols)
    close = _close(train_df)
    variant = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)
    labels = np.full(len(idx), ACTION_CASH, dtype=np.int64)
    long_ret = np.zeros(len(idx), dtype=np.float64)
    short_ret = np.zeros(len(idx), dtype=np.float64)
    min_edge = 0.0015
    for out_i, src_i in enumerate(idx):
        lr = _counterfactual_stack_return(
            df=train_df,
            close=close,
            q=q,
            dec=long_dec,
            i=int(src_i),
            side_keep=1,
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
        sr = _counterfactual_stack_return(
            df=train_df,
            close=close,
            q=q,
            dec=short_dec,
            i=int(src_i),
            side_keep=-1,
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
        long_ret[out_i] = lr
        short_ret[out_i] = sr
        if max(lr, sr) <= min_edge:
            labels[out_i] = ACTION_CASH
        elif lr >= sr:
            labels[out_i] = ACTION_LONG
        else:
            labels[out_i] = ACTION_SHORT
    audit = {
        "min_edge": float(min_edge),
        "inherited_label_distribution": pd.Series(inherited).value_counts().sort_index().to_dict(),
        "stack_pnl_label_distribution": pd.Series(labels).value_counts().sort_index().to_dict(),
        "long_ret_mean": float(np.mean(long_ret)),
        "short_ret_mean": float(np.mean(short_ret)),
        "long_ret_p95": float(np.quantile(long_ret, 0.95)),
        "short_ret_p95": float(np.quantile(short_ret, 0.95)),
    }
    return idx, labels, audit


def _stack_pnl_router_targets(
    *,
    train_df: pd.DataFrame,
    long_dec: pd.DataFrame,
    short_dec: pd.DataFrame,
    q: np.ndarray,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    idx, inherited = _router_label_frame(train_df, feature_cols)
    close = _close(train_df)
    variant = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)
    long_ret = np.zeros(len(idx), dtype=np.float64)
    short_ret = np.zeros(len(idx), dtype=np.float64)
    labels = np.full(len(idx), ACTION_CASH, dtype=np.int64)
    min_edge = 0.0015
    for out_i, src_i in enumerate(idx):
        long_ret[out_i] = _counterfactual_stack_return(
            df=train_df,
            close=close,
            q=q,
            dec=long_dec,
            i=int(src_i),
            side_keep=1,
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
        short_ret[out_i] = _counterfactual_stack_return(
            df=train_df,
            close=close,
            q=q,
            dec=short_dec,
            i=int(src_i),
            side_keep=-1,
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
        if max(long_ret[out_i], short_ret[out_i]) <= min_edge:
            labels[out_i] = ACTION_CASH
        elif long_ret[out_i] >= short_ret[out_i]:
            labels[out_i] = ACTION_LONG
        else:
            labels[out_i] = ACTION_SHORT
    audit = {
        "min_edge": float(min_edge),
        "inherited_label_distribution": pd.Series(inherited).value_counts().sort_index().to_dict(),
        "stack_pnl_label_distribution": pd.Series(labels).value_counts().sort_index().to_dict(),
        "long_ret_mean": float(np.mean(long_ret)),
        "short_ret_mean": float(np.mean(short_ret)),
        "long_ret_p95": float(np.quantile(long_ret, 0.95)),
        "short_ret_p95": float(np.quantile(short_ret, 0.95)),
    }
    return idx, long_ret, short_ret, audit


def _train_router(
    *,
    train_df: pd.DataFrame,
    long_dec: pd.DataFrame,
    short_dec: pd.DataFrame,
    q: np.ndarray,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    feature_cols: list[str],
    out_dir: Path,
) -> tuple[Any, list[str], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "router.pkl"
    summary_path = out_dir / "summary.json"
    if model_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(summary.get("router_weight_version", "")) == "stack_pnl_ternary_v1":
            payload = joblib.load(model_path)
            return payload["router"], list(payload["feature_cols"]), summary
    idx, labels, label_audit = _stack_pnl_router_labels(
        train_df=train_df,
        long_dec=long_dec,
        short_dec=short_dec,
        q=q,
        stack=stack,
        cfg=cfg,
        feature_cols=feature_cols,
    )
    x_all = _router_features(train_df, long_dec=long_dec, short_dec=short_dec, q=q)
    x = x_all.iloc[idx].reset_index(drop=True)
    labels_fit = labels
    counts = pd.Series(labels_fit).value_counts().to_dict()
    weights = np.asarray([1.0 / max(float(counts.get(int(v), 1)), 1.0) for v in labels_fit], dtype=np.float64)
    weights = weights / max(float(np.mean(weights)), 1e-12)
    router = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.12,
            early_stopping=False,
            random_state=5289601,
        ),
    )
    router.fit(x, labels_fit, histgradientboostingclassifier__sample_weight=weights)
    payload = {"router": router, "feature_cols": list(x.columns)}
    joblib.dump(payload, model_path)
    summary = {
        "model_id": MODEL_ID,
        "router_weight_version": "stack_pnl_ternary_v1",
        "rows": int(len(x)),
        "label_audit": label_audit,
        "fit_label_distribution": pd.Series(labels_fit).value_counts().sort_index().to_dict(),
        "feature_cols": list(x.columns),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return router, list(x.columns), summary


def _train_return_router(
    *,
    train_df: pd.DataFrame,
    long_dec: pd.DataFrame,
    short_dec: pd.DataFrame,
    q: np.ndarray,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    feature_cols: list[str],
    out_dir: Path,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "return_router.pkl"
    summary_path = out_dir / "return_summary.json"
    if model_path.exists() and summary_path.exists():
        payload = joblib.load(model_path)
        return payload, list(payload["feature_cols"]), json.loads(summary_path.read_text(encoding="utf-8"))
    idx, long_ret, short_ret, audit = _stack_pnl_router_targets(
        train_df=train_df,
        long_dec=long_dec,
        short_dec=short_dec,
        q=q,
        stack=stack,
        cfg=cfg,
        feature_cols=feature_cols,
    )
    x_all = _router_features(train_df, long_dec=long_dec, short_dec=short_dec, q=q)
    x = x_all.iloc[idx].reset_index(drop=True)
    long_w = 1.0 + np.clip(np.abs(long_ret) * 80.0, 0.0, 8.0)
    short_w = 1.0 + np.clip(np.abs(short_ret) * 80.0, 0.0, 8.0)
    long_model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=260,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.12,
            early_stopping=False,
            random_state=5289701,
        ),
    )
    short_model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=260,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.12,
            early_stopping=False,
            random_state=5289702,
        ),
    )
    long_model.fit(x, long_ret, histgradientboostingregressor__sample_weight=long_w)
    short_model.fit(x, short_ret, histgradientboostingregressor__sample_weight=short_w)
    payload = {"long_model": long_model, "short_model": short_model, "feature_cols": list(x.columns)}
    joblib.dump(payload, model_path)
    summary = {
        "model_id": MODEL_ID,
        "router_weight_version": "stack_pnl_return_regression_v1",
        "rows": int(len(x)),
        "target_audit": audit,
        "long_target_mean": float(np.mean(long_ret)),
        "short_target_mean": float(np.mean(short_ret)),
        "feature_cols": list(x.columns),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return payload, list(x.columns), summary


def _route_decisions(
    *,
    router: Any,
    router_cols: list[str],
    df: pd.DataFrame,
    long_dec: pd.DataFrame,
    short_dec: pd.DataFrame,
    q: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    x = _router_features(df, long_dec=long_dec, short_dec=short_dec, q=q).reindex(columns=router_cols).fillna(0.0)
    pred = np.asarray(router.predict(x), dtype=np.int64)
    out = long_dec.copy().reset_index(drop=True)
    out.loc[:, :] = 0
    out["leverage"] = 1.0
    use_long = pred == ACTION_LONG
    use_short = pred == ACTION_SHORT
    for col in long_dec.columns:
        out.loc[use_long, col] = long_dec.loc[use_long, col].to_numpy()
        out.loc[use_short, col] = short_dec.loc[use_short, col].to_numpy()
    return out, pred


def _route_return_decisions(
    *,
    payload: dict[str, Any],
    router_cols: list[str],
    df: pd.DataFrame,
    long_dec: pd.DataFrame,
    short_dec: pd.DataFrame,
    q: np.ndarray,
    threshold: float,
    margin: float,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    x = _router_features(df, long_dec=long_dec, short_dec=short_dec, q=q).reindex(columns=router_cols).fillna(0.0)
    pred_long = np.asarray(payload["long_model"].predict(x), dtype=np.float64)
    pred_short = np.asarray(payload["short_model"].predict(x), dtype=np.float64)
    pred = np.full(len(df), ACTION_CASH, dtype=np.int64)
    use_long = (pred_long >= float(threshold)) & ((pred_long - pred_short) >= float(margin))
    use_short = (pred_short >= float(threshold)) & ((pred_short - pred_long) > float(margin))
    pred[use_long] = ACTION_LONG
    pred[use_short] = ACTION_SHORT
    out = long_dec.copy().reset_index(drop=True)
    out.loc[:, :] = 0
    out["leverage"] = 1.0
    for col in long_dec.columns:
        out.loc[use_long, col] = long_dec.loc[use_long, col].to_numpy()
        out.loc[use_short, col] = short_dec.loc[use_short, col].to_numpy()
    pred_frame = pd.DataFrame({"pred_long_ret": pred_long, "pred_short_ret": pred_short, "route": pred})
    return out, pred, pred_frame


def _sl_ratio(res: dict[str, Any]) -> float:
    exits = dict(res.get("exits", {}))
    return float(sum(int(v) for k, v in exits.items() if "stop_loss" in str(k)) / max(int(res.get("trades", 0)), 1))


def _score(res: dict[str, Any]) -> float:
    if int(res.get("trades", 0)) < 20:
        return -1e9 + float(res.get("pnl", 0.0))
    return float(res["pnl"]) + 2.0 * float(res["mdd"]) + 40.0 * float(res["wr"]) - 0.03 * float(res["trades"])


def _row(name: str, split: str, res: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": name,
        "split": split,
        "pnl": float(res["pnl"]),
        "mdd": float(res["mdd"]),
        "wr": float(res["wr"]),
        "trades": int(res["trades"]),
        "deep_entries": int(res.get("deep_entries", 0)),
        "long_entries": int(res.get("long_entries", 0)),
        "short_entries": int(res.get("short_entries", 0)),
        "sl_ratio": float(_sl_ratio(res)),
        "score": float(_score(res)),
        "exits": json.dumps(res.get("exits", {}), ensure_ascii=False, sort_keys=True),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decontam._assert_clean_frame(decontam.TRAIN_CSV, name="train")
    decontam._assert_clean_frame(decontam.EVAL_CSV, name="eval")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "primary_parent.pkl", name="primary")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl", name="fallback")
    decontam._patch_runtime_sources()

    train_all = _read(TRAIN_CSV)
    eval_csv = _read(EVAL_CSV)
    _assert_clean_frame(train_all, name="train_csv")
    _assert_clean_frame(eval_csv, name="eval_csv")
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)

    cfg = sweep.precision._cfg_from_results()
    stack = sweep.precision._load_stack()
    val_df, eval_df = sweep.precision._load_frames()
    sources = sweep.precision._decision_sources(val_df, eval_df, stack["parent"])
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    train_q = v27._predict_all(stack["deep_model"], train_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])

    live_primary = joblib.load(PRIMARY_PARENT)
    live_fallback = joblib.load(FALLBACK_PARENT)
    primary_cols = list(live_primary["feature_cols"])
    fallback_cols = list(live_fallback["feature_cols"])
    for name, cols in (("primary", primary_cols), ("fallback", fallback_cols)):
        _assert_feature_cols(train_df, cols, name=f"{name}_train")
        _assert_feature_cols(val_df, cols, name=f"{name}_val")
        _assert_feature_cols(eval_df, cols, name=f"{name}_eval")

    long_primary, long_primary_summary = _train_side_parent(
        train_df=train_df,
        feature_cols=primary_cols,
        action_keep=ACTION_LONG,
        seed=5289801,
        out_dir=OUT_DIR / "long_primary",
    )
    long_fallback, long_fallback_summary = _train_side_parent(
        train_df=train_df,
        feature_cols=fallback_cols,
        action_keep=ACTION_LONG,
        seed=5289802,
        out_dir=OUT_DIR / "long_fallback",
    )
    short_primary, short_primary_summary = _train_side_parent(
        train_df=train_df,
        feature_cols=primary_cols,
        action_keep=ACTION_SHORT,
        seed=5289901,
        out_dir=OUT_DIR / "short_primary",
    )
    short_fallback, short_fallback_summary = _train_side_parent(
        train_df=train_df,
        feature_cols=fallback_cols,
        action_keep=ACTION_SHORT,
        seed=5289902,
        out_dir=OUT_DIR / "short_fallback",
    )

    train_long_dec = _predict_parent_pair(primary=long_primary, fallback=long_fallback, df=train_df, side_keep=1)
    train_short_dec = _predict_parent_pair(primary=short_primary, fallback=short_fallback, df=train_df, side_keep=-1)
    val_long_dec = _predict_parent_pair(primary=long_primary, fallback=long_fallback, df=val_df, side_keep=1)
    val_short_dec = _predict_parent_pair(primary=short_primary, fallback=short_fallback, df=val_df, side_keep=-1)
    eval_long_dec = _predict_parent_pair(primary=long_primary, fallback=long_fallback, df=eval_df, side_keep=1)
    eval_short_dec = _predict_parent_pair(primary=short_primary, fallback=short_fallback, df=eval_df, side_keep=-1)

    router, router_cols, router_summary = _train_router(
        train_df=train_df,
        long_dec=train_long_dec,
        short_dec=train_short_dec,
        q=train_q,
        stack=stack,
        cfg=cfg,
        feature_cols=primary_cols,
        out_dir=OUT_DIR / "router",
    )
    return_router, return_router_cols, return_router_summary = _train_return_router(
        train_df=train_df,
        long_dec=train_long_dec,
        short_dec=train_short_dec,
        q=train_q,
        stack=stack,
        cfg=cfg,
        feature_cols=primary_cols,
        out_dir=OUT_DIR / "router",
    )
    val_router_dec, val_router_pred = _route_decisions(
        router=router,
        router_cols=router_cols,
        df=val_df,
        long_dec=val_long_dec,
        short_dec=val_short_dec,
        q=val_q,
    )
    eval_router_dec, eval_router_pred = _route_decisions(
        router=router,
        router_cols=router_cols,
        df=eval_df,
        long_dec=eval_long_dec,
        short_dec=eval_short_dec,
        q=eval_q,
    )

    base_variant = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)
    baseline_val = sweep._backtest_variant(
        df=val_df,
        q=val_q,
        dec=sources[str(cfg["source"])][0],
        stack=stack,
        cfg=cfg,
        variant=base_variant,
        cost_mult=3,
    )
    baseline_oos = sweep._backtest_variant(
        df=eval_df,
        q=eval_q,
        dec=sources[str(cfg["source"])][1],
        stack=stack,
        cfg=cfg,
        variant=base_variant,
        cost_mult=3,
    )

    def router_gate(pred: np.ndarray):
        def _gate(i: int, side: int, ql: float, qs: float, row: pd.Series) -> tuple[bool, str]:
            want = int(pred[int(i)])
            if want == ACTION_LONG and side > 0:
                return True, "router_long"
            if want == ACTION_SHORT and side < 0:
                return True, "router_short"
            return False, "router_side_veto"

        return _gate

    router_val = sweep._backtest_variant(
        df=val_df,
        q=val_q,
        dec=val_router_dec,
        stack=stack,
        cfg=cfg,
        variant=base_variant,
        cost_mult=3,
        deep_gate=router_gate(val_router_pred),
    )
    router_oos = sweep._backtest_variant(
        df=eval_df,
        q=eval_q,
        dec=eval_router_dec,
        stack=stack,
        cfg=cfg,
        variant=base_variant,
        cost_mult=3,
        record=True,
        deep_gate=router_gate(eval_router_pred),
    )
    records = list(router_oos.pop("trade_records", []))
    ledger_path = OUT_DIR / "router_oos_cost3_ledger.csv"
    pd.DataFrame(records).to_csv(ledger_path, index=False)

    return_rows: list[dict[str, Any]] = []
    best_return: dict[str, Any] | None = None
    best_return_records: list[dict[str, Any]] = []
    best_return_val_score = -1e18
    for threshold in (-0.002, 0.0, 0.0015, 0.003, 0.006):
        for margin in (0.0, 0.0015, 0.003):
            val_ret_dec, val_ret_pred, _ = _route_return_decisions(
                payload=return_router,
                router_cols=return_router_cols,
                df=val_df,
                long_dec=val_long_dec,
                short_dec=val_short_dec,
                q=val_q,
                threshold=threshold,
                margin=margin,
            )
            eval_ret_dec, eval_ret_pred, _ = _route_return_decisions(
                payload=return_router,
                router_cols=return_router_cols,
                df=eval_df,
                long_dec=eval_long_dec,
                short_dec=eval_short_dec,
                q=eval_q,
                threshold=threshold,
                margin=margin,
            )
            val_ret = sweep._backtest_variant(
                df=val_df,
                q=val_q,
                dec=val_ret_dec,
                stack=stack,
                cfg=cfg,
                variant=base_variant,
                cost_mult=3,
                deep_gate=router_gate(val_ret_pred),
            )
            oos_ret = sweep._backtest_variant(
                df=eval_df,
                q=eval_q,
                dec=eval_ret_dec,
                stack=stack,
                cfg=cfg,
                variant=base_variant,
                cost_mult=3,
                record=True,
                deep_gate=router_gate(eval_ret_pred),
            )
            oos_records = list(oos_ret.pop("trade_records", []))
            item = {
                "threshold": float(threshold),
                "margin": float(margin),
                "val": _row("long_short_return_router", "val", val_ret),
                "oos": _row("long_short_return_router", "oos", oos_ret),
                "val_route_distribution": pd.Series(val_ret_pred).value_counts().sort_index().to_dict(),
                "oos_route_distribution": pd.Series(eval_ret_pred).value_counts().sort_index().to_dict(),
            }
            return_rows.append(item)
            val_score = float(item["val"]["score"])
            if val_score > best_return_val_score:
                best_return_val_score = val_score
                best_return = item
                best_return_records = oos_records
    return_ledger_path = OUT_DIR / "return_router_oos_cost3_ledger.csv"
    pd.DataFrame(best_return_records).to_csv(return_ledger_path, index=False)

    rows = [
        _row("deep_stop_cd18_baseline", "val", baseline_val),
        _row("deep_stop_cd18_baseline", "oos", baseline_oos),
        _row("long_short_specialist_router", "val", router_val),
        _row("long_short_specialist_router", "oos", router_oos),
    ]
    if best_return is not None:
        rows.extend([best_return["val"], best_return["oos"]])
    grid_path = OUT_DIR / "grid.csv"
    pd.DataFrame(rows).to_csv(grid_path, index=False)
    summary = {
        "model_id": MODEL_ID,
        "scope": "Duplicate the full deep_stop_cd18 stack into LONG-only and SHORT-only specialists, then route between them with a learned middle router. Active/live artifacts are unchanged.",
        "artifacts": {
            "long_primary": str(OUT_DIR / "long_primary" / "parent.pkl"),
            "long_fallback": str(OUT_DIR / "long_fallback" / "parent.pkl"),
            "short_primary": str(OUT_DIR / "short_primary" / "parent.pkl"),
            "short_fallback": str(OUT_DIR / "short_fallback" / "parent.pkl"),
            "router": str(OUT_DIR / "router" / "router.pkl"),
            "return_router": str(OUT_DIR / "router" / "return_router.pkl"),
            "grid": str(grid_path),
            "oos_ledger": str(ledger_path),
            "return_oos_ledger": str(return_ledger_path),
        },
        "summaries": {
            "long_primary": long_primary_summary,
            "long_fallback": long_fallback_summary,
            "short_primary": short_primary_summary,
            "short_fallback": short_fallback_summary,
            "router": router_summary,
            "return_router": return_router_summary,
        },
        "router_prediction_distribution": {
            "val": pd.Series(val_router_pred).value_counts().sort_index().to_dict(),
            "oos": pd.Series(eval_router_pred).value_counts().sort_index().to_dict(),
        },
        "return_router_candidates": return_rows,
        "best_return_router_by_validation_score": best_return,
        "rows": rows,
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "grid": str(grid_path), "oos_ledger": str(ledger_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
