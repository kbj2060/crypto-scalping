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

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, ACTION_SHORT  # noqa: E402
from scripts import precision_retest_01965_alpha7_combo_20260527 as precision  # noqa: E402
from scripts import runtime_retest_alpha7_1_01965_decontam_20260528 as decontam  # noqa: E402
from scripts import sweep_decontam_deep_alpha_controls_20260528 as sweep  # noqa: E402
from scripts import train_eval_alpha7_01965_full_long_short_router_20260528 as ls  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _close,
    _combine_primary_fallback,
    _json_default,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.retrain_alpha7_1_01965_tp_sl_decontam_20260528 import TRAIN_CSV  # noqa: E402

MODEL_ID = "alpha7_01965_router_overlay_refinements_20260528"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASE_LS_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_01965_full_long_short_router_20260528"
ALPHA7_DIR = ROOT / "data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528"


def _clf(seed: int, max_iter: int = 220) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=int(max_iter),
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.12,
            early_stopping=False,
            random_state=int(seed),
        ),
    )


def _side_from_deep(q: np.ndarray, i: int, overlay: Any) -> int:
    if int(i) < 60:
        return 0
    ql, qs = float(q[int(i), 0]), float(q[int(i), 1])
    side = 1 if ql > qs else -1
    if max(ql, qs) >= float(overlay.edge_th) and abs(ql - qs) >= float(overlay.margin_th):
        return side
    return 0


def _baseline_opportunity_side(base_dec: pd.DataFrame, q: np.ndarray, i: int, overlay: Any) -> int:
    row = base_dec.iloc[int(i)]
    if int(row.action) != ACTION_CASH and int(row.side) != 0:
        return int(row.side)
    return _side_from_deep(q, int(i), overlay)


def _base_features(df: pd.DataFrame, base_dec: pd.DataFrame, long_dec: pd.DataFrame, short_dec: pd.DataFrame, q: np.ndarray) -> pd.DataFrame:
    out = ls._router_features(df, long_dec=long_dec, short_dec=short_dec, q=q)
    active = ls._active(base_dec).to_numpy(dtype=np.float64)
    side = pd.to_numeric(base_dec["side"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["base_active"] = active
    out["base_side"] = side
    for col in ["confidence", "quality_score", "notional_exposure", "take_profit", "stop_loss"]:
        out[f"base_{col}"] = pd.to_numeric(base_dec[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _signed_side_features(features: pd.DataFrame, side: int) -> pd.DataFrame:
    x = features.copy()
    sign = float(side)
    for col in list(x.columns):
        if col.startswith("deep_q_") or col.endswith("_quality") or col.endswith("_confidence"):
            continue
    x["selected_side"] = sign
    x["side_deep_q"] = x["deep_q_long"] if side > 0 else x["deep_q_short"]
    x["side_deep_other_q"] = x["deep_q_short"] if side > 0 else x["deep_q_long"]
    x["side_deep_margin"] = (x["deep_q_long"] - x["deep_q_short"]) * sign
    x["side_base_match"] = (x["base_side"] == sign).astype(float)
    return x


def _train_side_veto(
    *,
    train_df: pd.DataFrame,
    base_dec: pd.DataFrame,
    long_dec: pd.DataFrame,
    short_dec: pd.DataFrame,
    q: np.ndarray,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    out_dir: Path,
) -> tuple[Any, list[str], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "side_veto.pkl"
    summary_path = out_dir / "side_veto_summary.json"
    if path.exists() and summary_path.exists():
        payload = joblib.load(path)
        return payload["model"], list(payload["feature_cols"]), json.loads(summary_path.read_text(encoding="utf-8"))
    overlay = precision._overlay(stack["overlay"], cfg)
    close = _close(train_df)
    feats = _base_features(train_df, base_dec, long_dec, short_dec, q)
    rows = []
    labels = []
    rets = []
    for i in range(0, max(0, len(train_df) - 2), 6):
        side = _baseline_opportunity_side(base_dec, q, i, overlay)
        if side == 0:
            continue
        dec = base_dec if int(base_dec.iloc[i].side) == side else (long_dec if side > 0 else short_dec)
        ret = ls._counterfactual_stack_return(
            df=train_df,
            close=close,
            q=q,
            dec=dec,
            i=int(i),
            side_keep=int(side),
            stack=stack,
            cfg=cfg,
            variant=sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18),
        )
        rows.append(_signed_side_features(feats.iloc[[i]], side).iloc[0])
        labels.append(int(ret > 0.0015))
        rets.append(float(ret))
    x = pd.DataFrame(rows).reset_index(drop=True)
    y = np.asarray(labels, dtype=np.int64)
    weights = 1.0 + np.clip(np.abs(np.asarray(rets)) * 100.0, 0.0, 6.0)
    model = _clf(5290101)
    model.fit(x, y, histgradientboostingclassifier__sample_weight=weights)
    payload = {"model": model, "feature_cols": list(x.columns)}
    joblib.dump(payload, path)
    summary = {
        "rows": int(len(x)),
        "label_distribution": pd.Series(y).value_counts().sort_index().to_dict(),
        "ret_mean": float(np.mean(rets)) if rets else 0.0,
        "ret_p95": float(np.quantile(rets, 0.95)) if rets else 0.0,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return model, list(x.columns), summary


def _proba_pos(model: Any, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)
    if 1 not in classes:
        return np.zeros(len(x), dtype=np.float64)
    return proba[:, int(np.flatnonzero(classes == 1)[0])]


def _apply_side_veto(
    *,
    df: pd.DataFrame,
    base_dec: pd.DataFrame,
    long_dec: pd.DataFrame,
    short_dec: pd.DataFrame,
    q: np.ndarray,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    model: Any,
    feature_cols: list[str],
    threshold: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    overlay = precision._overlay(stack["overlay"], cfg)
    feats = _base_features(df, base_dec, long_dec, short_dec, q)
    out = base_dec.copy().reset_index(drop=True)
    keep = np.ones(len(df), dtype=bool)
    side_cache = np.zeros(len(df), dtype=np.int64)
    for side in (1, -1):
        idx = []
        for i in range(len(df)):
            s = _baseline_opportunity_side(base_dec, q, i, overlay)
            if s == side:
                idx.append(i)
                side_cache[i] = side
        if not idx:
            continue
        x = _signed_side_features(feats.iloc[idx], side).reindex(columns=feature_cols).fillna(0.0)
        p = _proba_pos(model, x)
        bad = np.asarray(idx, dtype=np.int64)[p < float(threshold)]
        keep[bad] = False
    bad_parent = (~keep) & ls._active(out).to_numpy(dtype=bool)
    out.loc[bad_parent, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[bad_parent, "leverage"] = 1.0
    return out, keep


def _gate_from_keep(keep: np.ndarray):
    def _gate(i: int, side: int, ql: float, qs: float, row: pd.Series) -> tuple[bool, str]:
        if int(i) >= len(keep):
            return False, "side_veto_oob"
        return (bool(keep[int(i)]), "side_veto")

    return _gate


def _window_feature_rows(features: pd.DataFrame, window: int) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    starts = []
    for start in range(0, len(features), int(window)):
        end = min(len(features), start + int(window))
        if end - start < max(24, int(window) // 2):
            continue
        chunk = features.iloc[start:end]
        row: dict[str, float] = {}
        for col in features.columns:
            vals = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0)
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=0))
        rows.append(row)
        starts.append(start)
    return pd.DataFrame(rows).fillna(0.0), np.asarray(starts, dtype=np.int64)


def _train_portfolio_window_router(
    *,
    train_df: pd.DataFrame,
    long_dec: pd.DataFrame,
    short_dec: pd.DataFrame,
    q: np.ndarray,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    out_dir: Path,
    window: int = 96,
) -> tuple[Any, list[str], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "portfolio_window_router.pkl"
    summary_path = out_dir / "portfolio_window_router_summary.json"
    if path.exists() and summary_path.exists():
        payload = joblib.load(path)
        return payload["model"], list(payload["feature_cols"]), json.loads(summary_path.read_text(encoding="utf-8"))
    features = ls._router_features(train_df, long_dec=long_dec, short_dec=short_dec, q=q)
    x_win, starts = _window_feature_rows(features, window)
    close = _close(train_df)
    variant = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)
    labels = []
    sums = []
    for start in starts:
        end = min(len(train_df) - 2, int(start) + int(window))
        long_sum = short_sum = 0.0
        for i in range(int(start), end, 6):
            long_sum += ls._counterfactual_stack_return(df=train_df, close=close, q=q, dec=long_dec, i=i, side_keep=1, stack=stack, cfg=cfg, variant=variant)
            short_sum += ls._counterfactual_stack_return(df=train_df, close=close, q=q, dec=short_dec, i=i, side_keep=-1, stack=stack, cfg=cfg, variant=variant)
        if max(long_sum, short_sum) <= 0.01:
            lab = ACTION_CASH
        elif long_sum >= short_sum:
            lab = ACTION_LONG
        else:
            lab = ACTION_SHORT
        labels.append(lab)
        sums.append((float(long_sum), float(short_sum)))
    y = np.asarray(labels, dtype=np.int64)
    weights = np.asarray([1.0 / max(float(pd.Series(y).value_counts().to_dict().get(int(v), 1)), 1.0) for v in y])
    weights = weights / max(float(weights.mean()), 1e-12)
    model = _clf(5290201, max_iter=160)
    model.fit(x_win, y, histgradientboostingclassifier__sample_weight=weights)
    payload = {"model": model, "feature_cols": list(x_win.columns), "window": int(window)}
    joblib.dump(payload, path)
    summary = {
        "rows": int(len(x_win)),
        "window": int(window),
        "label_distribution": pd.Series(y).value_counts().sort_index().to_dict(),
        "sum_mean": np.asarray(sums).mean(axis=0).tolist() if sums else [0.0, 0.0],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return model, list(x_win.columns), summary


def _route_by_window(
    *,
    model: Any,
    feature_cols: list[str],
    df: pd.DataFrame,
    long_dec: pd.DataFrame,
    short_dec: pd.DataFrame,
    q: np.ndarray,
    window: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    features = ls._router_features(df, long_dec=long_dec, short_dec=short_dec, q=q)
    x_win, starts = _window_feature_rows(features, window)
    if len(x_win) == 0:
        pred = np.zeros(len(df), dtype=np.int64)
    else:
        y_win = np.asarray(model.predict(x_win.reindex(columns=feature_cols).fillna(0.0)), dtype=np.int64)
        pred = np.full(len(df), ACTION_CASH, dtype=np.int64)
        for start, lab in zip(starts, y_win):
            pred[int(start) : min(len(df), int(start) + int(window))] = int(lab)
    out = long_dec.copy().reset_index(drop=True)
    out.loc[:, :] = 0
    out["leverage"] = 1.0
    use_long = pred == ACTION_LONG
    use_short = pred == ACTION_SHORT
    for col in long_dec.columns:
        out.loc[use_long, col] = long_dec.loc[use_long, col].to_numpy()
        out.loc[use_short, col] = short_dec.loc[use_short, col].to_numpy()
    return out, pred


def _train_opportunity_router(
    *,
    train_df: pd.DataFrame,
    long_dec: pd.DataFrame,
    short_dec: pd.DataFrame,
    q: np.ndarray,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    out_dir: Path,
) -> tuple[Any, list[str], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "opportunity_router.pkl"
    summary_path = out_dir / "opportunity_router_summary.json"
    if path.exists() and summary_path.exists():
        payload = joblib.load(path)
        return payload["model"], list(payload["feature_cols"]), json.loads(summary_path.read_text(encoding="utf-8"))
    features = ls._router_features(train_df, long_dec=long_dec, short_dec=short_dec, q=q)
    close = _close(train_df)
    variant = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)
    overlay = precision._overlay(stack["overlay"], cfg)
    rows = []
    labels = []
    rets = []
    for i in range(0, max(0, len(train_df) - 2), 6):
        if not bool(ls._active(long_dec).iloc[i] or ls._active(short_dec).iloc[i] or _side_from_deep(q, i, overlay) != 0):
            continue
        lr = ls._counterfactual_stack_return(df=train_df, close=close, q=q, dec=long_dec, i=i, side_keep=1, stack=stack, cfg=cfg, variant=variant)
        sr = ls._counterfactual_stack_return(df=train_df, close=close, q=q, dec=short_dec, i=i, side_keep=-1, stack=stack, cfg=cfg, variant=variant)
        lab = ACTION_CASH if max(lr, sr) <= 0.0015 else (ACTION_LONG if lr >= sr else ACTION_SHORT)
        rows.append(features.iloc[i])
        labels.append(lab)
        rets.append(max(lr, sr))
    x = pd.DataFrame(rows).reset_index(drop=True).fillna(0.0)
    y = np.asarray(labels, dtype=np.int64)
    counts = pd.Series(y).value_counts().to_dict()
    weights = np.asarray([(1.0 / max(float(counts.get(int(v), 1)), 1.0)) * (1.0 + min(abs(float(r)) * 80.0, 6.0)) for v, r in zip(y, rets)])
    weights = weights / max(float(weights.mean()), 1e-12)
    model = _clf(5290301)
    model.fit(x, y, histgradientboostingclassifier__sample_weight=weights)
    payload = {"model": model, "feature_cols": list(x.columns)}
    joblib.dump(payload, path)
    summary = {
        "rows": int(len(x)),
        "label_distribution": pd.Series(y).value_counts().sort_index().to_dict(),
        "ret_mean": float(np.mean(rets)) if rets else 0.0,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return model, list(x.columns), summary


def _route_opportunity(
    *,
    model: Any,
    feature_cols: list[str],
    df: pd.DataFrame,
    long_dec: pd.DataFrame,
    short_dec: pd.DataFrame,
    q: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    x = ls._router_features(df, long_dec=long_dec, short_dec=short_dec, q=q).reindex(columns=feature_cols).fillna(0.0)
    pred = np.asarray(model.predict(x), dtype=np.int64)
    out = long_dec.copy().reset_index(drop=True)
    out.loc[:, :] = 0
    out["leverage"] = 1.0
    use_long = pred == ACTION_LONG
    use_short = pred == ACTION_SHORT
    for col in long_dec.columns:
        out.loc[use_long, col] = long_dec.loc[use_long, col].to_numpy()
        out.loc[use_short, col] = short_dec.loc[use_short, col].to_numpy()
    return out, pred


def _short_route_by_probability(
    *,
    model: Any,
    feature_cols: list[str],
    df: pd.DataFrame,
    long_dec: pd.DataFrame,
    short_dec: pd.DataFrame,
    q: np.ndarray,
    threshold: float,
) -> np.ndarray:
    x = ls._router_features(df, long_dec=long_dec, short_dec=short_dec, q=q).reindex(columns=feature_cols).fillna(0.0)
    proba = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)
    pred = np.full(len(df), ACTION_CASH, dtype=np.int64)
    if ACTION_SHORT not in classes:
        return pred
    p_short = proba[:, int(np.flatnonzero(classes == ACTION_SHORT)[0])]
    pred[p_short >= float(threshold)] = ACTION_SHORT
    return pred


def _short_override_decisions(base_dec: pd.DataFrame, short_dec: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    out = base_dec.copy().reset_index(drop=True)
    use_short = pred == ACTION_SHORT
    for col in short_dec.columns:
        out.loc[use_short, col] = short_dec.loc[use_short, col].to_numpy()
    return out


def _alpha7_combo_decision(df: pd.DataFrame) -> pd.DataFrame:
    primary = joblib.load(ALPHA7_DIR / "primary_parent.pkl")
    fallback = joblib.load(ALPHA7_DIR / "fallback_alpha43_no_legacy_parent.pkl")
    p_rt = _load_best_scale_runtime(ALPHA7_DIR / "primary_summary.json")
    f_rt = _load_best_scale_runtime(ALPHA7_DIR / "fallback_alpha43_no_legacy_summary.json")
    p = _predict_scaled(primary, df, p_rt)
    f = _predict_scaled(fallback, df, f_rt)
    return _combine_primary_fallback(p, f)


def _router_gate(pred: np.ndarray):
    def _gate(i: int, side: int, ql: float, qs: float, row: pd.Series) -> tuple[bool, str]:
        if int(i) >= len(pred):
            return False, "router_oob"
        want = int(pred[int(i)])
        if want == ACTION_LONG and side > 0:
            return True, "router_long"
        if want == ACTION_SHORT and side < 0:
            return True, "router_short"
        return False, "router_side_veto"

    return _gate


def _row(name: str, split: str, res: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "variant": name,
        "split": split,
        "pnl": float(res["pnl"]),
        "mdd": float(res["mdd"]),
        "wr": float(res["wr"]),
        "trades": int(res["trades"]),
        "deep_entries": int(res.get("deep_entries", 0)),
        "long_entries": int(res.get("long_entries", 0)),
        "short_entries": int(res.get("short_entries", 0)),
        "sl_ratio": float(sweep._sl_ratio(res)),
        "score": float(sweep._score(res)),
        "exits": json.dumps(res.get("exits", {}), ensure_ascii=False, sort_keys=True),
    }
    if extra:
        row.update(extra)
    return row


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decontam._assert_clean_frame(decontam.TRAIN_CSV, name="train")
    decontam._assert_clean_frame(decontam.EVAL_CSV, name="eval")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "primary_parent.pkl", name="primary")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl", name="fallback")
    decontam._patch_runtime_sources()

    cfg = precision._cfg_from_results()
    stack = precision._load_stack()
    val_df, eval_df = precision._load_frames()
    sources = precision._decision_sources(val_df, eval_df, stack["parent"])
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    base_val_dec = sources[str(cfg["source"])][0]
    base_eval_dec = sources[str(cfg["source"])][1]

    train_all = _read(TRAIN_CSV)
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    train_q = v27._predict_all(stack["deep_model"], train_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])

    primary_cols = list(joblib.load(ls.PRIMARY_PARENT)["feature_cols"])
    fallback_cols = list(joblib.load(ls.FALLBACK_PARENT)["feature_cols"])
    long_primary, _ = ls._train_side_parent(train_df=train_df, feature_cols=primary_cols, action_keep=ACTION_LONG, seed=5289801, out_dir=BASE_LS_DIR / "long_primary")
    long_fallback, _ = ls._train_side_parent(train_df=train_df, feature_cols=fallback_cols, action_keep=ACTION_LONG, seed=5289802, out_dir=BASE_LS_DIR / "long_fallback")
    short_primary, _ = ls._train_side_parent(train_df=train_df, feature_cols=primary_cols, action_keep=ACTION_SHORT, seed=5289901, out_dir=BASE_LS_DIR / "short_primary")
    short_fallback, _ = ls._train_side_parent(train_df=train_df, feature_cols=fallback_cols, action_keep=ACTION_SHORT, seed=5289902, out_dir=BASE_LS_DIR / "short_fallback")

    train_long_dec = ls._predict_parent_pair(primary=long_primary, fallback=long_fallback, df=train_df, side_keep=1)
    train_short_dec = ls._predict_parent_pair(primary=short_primary, fallback=short_fallback, df=train_df, side_keep=-1)
    val_long_dec = ls._predict_parent_pair(primary=long_primary, fallback=long_fallback, df=val_df, side_keep=1)
    val_short_dec = ls._predict_parent_pair(primary=short_primary, fallback=short_fallback, df=val_df, side_keep=-1)
    eval_long_dec = ls._predict_parent_pair(primary=long_primary, fallback=long_fallback, df=eval_df, side_keep=1)
    eval_short_dec = ls._predict_parent_pair(primary=short_primary, fallback=short_fallback, df=eval_df, side_keep=-1)
    variant = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)

    rows: list[dict[str, Any]] = []
    baseline_val = sweep._backtest_variant(df=val_df, q=val_q, dec=base_val_dec, stack=stack, cfg=cfg, variant=variant, cost_mult=3)
    baseline_oos = sweep._backtest_variant(df=eval_df, q=eval_q, dec=base_eval_dec, stack=stack, cfg=cfg, variant=variant, cost_mult=3)
    rows += [_row("deep_stop_cd18_baseline", "val", baseline_val), _row("deep_stop_cd18_baseline", "oos", baseline_oos)]

    train_base_dec = _alpha7_combo_decision(train_df)
    side_model, side_cols, side_summary = _train_side_veto(train_df=train_df, base_dec=train_base_dec, long_dec=train_long_dec, short_dec=train_short_dec, q=train_q, stack=stack, cfg=cfg, out_dir=OUT_DIR / "side_veto")
    best_side = None
    for th in (0.40, 0.50, 0.60, 0.70):
        val_dec, val_keep = _apply_side_veto(df=val_df, base_dec=base_val_dec, long_dec=val_long_dec, short_dec=val_short_dec, q=val_q, stack=stack, cfg=cfg, model=side_model, feature_cols=side_cols, threshold=th)
        oos_dec, oos_keep = _apply_side_veto(df=eval_df, base_dec=base_eval_dec, long_dec=eval_long_dec, short_dec=eval_short_dec, q=eval_q, stack=stack, cfg=cfg, model=side_model, feature_cols=side_cols, threshold=th)
        val = sweep._backtest_variant(df=val_df, q=val_q, dec=val_dec, stack=stack, cfg=cfg, variant=variant, cost_mult=3, deep_gate=_gate_from_keep(val_keep))
        oos = sweep._backtest_variant(df=eval_df, q=eval_q, dec=oos_dec, stack=stack, cfg=cfg, variant=variant, cost_mult=3, deep_gate=_gate_from_keep(oos_keep))
        item = {"threshold": th, "val": _row("baseline_side_veto", "val", val), "oos": _row("baseline_side_veto", "oos", oos)}
        if best_side is None or item["val"]["score"] > best_side["val"]["score"]:
            best_side = item

    port_model, port_cols, port_summary = _train_portfolio_window_router(train_df=train_df, long_dec=train_long_dec, short_dec=train_short_dec, q=train_q, stack=stack, cfg=cfg, out_dir=OUT_DIR / "portfolio_router")
    val_port_dec, val_port_pred = _route_by_window(model=port_model, feature_cols=port_cols, df=val_df, long_dec=val_long_dec, short_dec=val_short_dec, q=val_q, window=int(port_summary["window"]))
    oos_port_dec, oos_port_pred = _route_by_window(model=port_model, feature_cols=port_cols, df=eval_df, long_dec=eval_long_dec, short_dec=eval_short_dec, q=eval_q, window=int(port_summary["window"]))
    port_val = sweep._backtest_variant(df=val_df, q=val_q, dec=val_port_dec, stack=stack, cfg=cfg, variant=variant, cost_mult=3, deep_gate=_router_gate(val_port_pred))
    port_oos = sweep._backtest_variant(df=eval_df, q=eval_q, dec=oos_port_dec, stack=stack, cfg=cfg, variant=variant, cost_mult=3, deep_gate=_router_gate(oos_port_pred))

    opp_model, opp_cols, opp_summary = _train_opportunity_router(train_df=train_df, long_dec=train_long_dec, short_dec=train_short_dec, q=train_q, stack=stack, cfg=cfg, out_dir=OUT_DIR / "opportunity_router")
    val_opp_dec, val_opp_pred = _route_opportunity(model=opp_model, feature_cols=opp_cols, df=val_df, long_dec=val_long_dec, short_dec=val_short_dec, q=val_q)
    oos_opp_dec, oos_opp_pred = _route_opportunity(model=opp_model, feature_cols=opp_cols, df=eval_df, long_dec=eval_long_dec, short_dec=eval_short_dec, q=eval_q)
    opp_val = sweep._backtest_variant(df=val_df, q=val_q, dec=val_opp_dec, stack=stack, cfg=cfg, variant=variant, cost_mult=3, deep_gate=_router_gate(val_opp_pred))
    opp_oos = sweep._backtest_variant(df=eval_df, q=eval_q, dec=oos_opp_dec, stack=stack, cfg=cfg, variant=variant, cost_mult=3, deep_gate=_router_gate(oos_opp_pred))

    best_short = None
    short_records: list[dict[str, Any]] = []
    for th in (0.35, 0.45, 0.55, 0.65, 0.75):
        val_short_pred = _short_route_by_probability(
            model=opp_model,
            feature_cols=opp_cols,
            df=val_df,
            long_dec=val_long_dec,
            short_dec=val_short_dec,
            q=val_q,
            threshold=th,
        )
        oos_short_pred = _short_route_by_probability(
            model=opp_model,
            feature_cols=opp_cols,
            df=eval_df,
            long_dec=eval_long_dec,
            short_dec=eval_short_dec,
            q=eval_q,
            threshold=th,
        )
        short_val_dec = _short_override_decisions(base_val_dec, val_short_dec, val_short_pred)
        short_oos_dec = _short_override_decisions(base_eval_dec, eval_short_dec, oos_short_pred)
        short_val = sweep._backtest_variant(df=val_df, q=val_q, dec=short_val_dec, stack=stack, cfg=cfg, variant=variant, cost_mult=3, deep_gate=None)
        short_oos = sweep._backtest_variant(df=eval_df, q=eval_q, dec=short_oos_dec, stack=stack, cfg=cfg, variant=variant, cost_mult=3, record=True, deep_gate=None)
        records = list(short_oos.pop("trade_records", []))
        item = {
            "threshold": float(th),
            "val": _row("short_override_overlay", "val", short_val),
            "oos": _row("short_override_overlay", "oos", short_oos),
            "val_route_distribution": pd.Series(val_short_pred).value_counts().sort_index().to_dict(),
            "oos_route_distribution": pd.Series(oos_short_pred).value_counts().sort_index().to_dict(),
        }
        if best_short is None or item["val"]["score"] > best_short["val"]["score"]:
            best_short = item
            short_records = records
    short_ledger = OUT_DIR / "short_override_oos_cost3_ledger.csv"
    pd.DataFrame(short_records).to_csv(short_ledger, index=False)

    if best_side is not None:
        rows += [best_side["val"], best_side["oos"]]
    rows += [
        _row("portfolio_window_router", "val", port_val, {"route_dist": json.dumps(pd.Series(val_port_pred).value_counts().sort_index().to_dict(), sort_keys=True)}),
        _row("portfolio_window_router", "oos", port_oos, {"route_dist": json.dumps(pd.Series(oos_port_pred).value_counts().sort_index().to_dict(), sort_keys=True)}),
        _row("trade_opportunity_router", "val", opp_val, {"route_dist": json.dumps(pd.Series(val_opp_pred).value_counts().sort_index().to_dict(), sort_keys=True)}),
        _row("trade_opportunity_router", "oos", opp_oos, {"route_dist": json.dumps(pd.Series(oos_opp_pred).value_counts().sort_index().to_dict(), sort_keys=True)}),
    ]
    if best_short is not None:
        rows += [best_short["val"], best_short["oos"]]
    grid_path = OUT_DIR / "grid.csv"
    pd.DataFrame(rows).to_csv(grid_path, index=False)
    summary = {
        "model_id": MODEL_ID,
        "scope": "Refinement experiments after full LONG/SHORT stack router failed: portfolio window router, baseline side veto, trade opportunity router, short override overlay.",
        "grid": str(grid_path),
        "short_override_oos_ledger": str(short_ledger),
        "summaries": {
            "side_veto": side_summary,
            "portfolio_window_router": port_summary,
            "opportunity_router": opp_summary,
        },
        "best_side_veto_by_validation": best_side,
        "best_short_override_by_validation": best_short,
        "rows": rows,
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "grid": str(grid_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
