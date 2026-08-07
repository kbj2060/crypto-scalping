#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_1_multihorizon_tb_hgb_20260620"
RUN_ID = "omega1_2_1_multihorizon_tb_hgb_no_maxhold_20260620"
SOURCE_ID = "omega1_2_1_multihorizon_tb_daytrade_20260620"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / RUN_ID
SOURCE_DIR = ROOT / "tmp/causal_regen_20260516" / SOURCE_ID

TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
LABEL_2025 = SOURCE_DIR / "train_2025_multihorizon_tb_labels.csv"
PARENT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"
TRAIN_PARENT_PATH = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_parent_train_inference_20260620/train_predictions_2025_jan_sep_true3head_in_sample.csv"

SPLIT_TS = pd.Timestamp("2025-10-01")
LEVERAGE = 2.0
FEE_PER_SIDE = 0.0001 * 3.0
LABEL_MIN_UTILITY_GRID = [0.00100]
TRADE_THRESHOLD_GRID = [0.35, 0.45, 0.55, 0.65]
SIDE_EDGE_MIN_GRID = [0.00, 0.04, 0.08]
PARENT_MIN_QUALITY_GRID = [0.0, 0.65]
MARGIN_FRACTION_GRID = [0.025]
USE_PREDICTED_HORIZON_MAX_HOLD = False


def _normalize_parent_predictions(p: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": p["timestamp"]})
    expert = p[f"{prefix}router_expert"].astype(str).replace({"bull": 0, "bear": 1, "chop": 2, "chop_expert": 2})
    out["parent_router_id"] = pd.to_numeric(expert, errors="coerce").fillna(2).astype(int)
    out["parent_router_confidence"] = pd.to_numeric(p[f"{prefix}router_confidence"], errors="coerce").fillna(0.0)
    out["parent_router_margin"] = pd.to_numeric(p[f"{prefix}router_margin"], errors="coerce").fillna(0.0)
    out["parent_dir_p_cash"] = pd.to_numeric(p[f"{prefix}dir_p_cash"], errors="coerce").fillna(0.0)
    out["parent_dir_p_long"] = pd.to_numeric(p[f"{prefix}dir_p_long"], errors="coerce").fillna(0.0)
    out["parent_dir_p_short"] = pd.to_numeric(p[f"{prefix}dir_p_short"], errors="coerce").fillna(0.0)
    out["parent_dir_confidence"] = pd.to_numeric(p[f"{prefix}dir_confidence"], errors="coerce").fillna(0.0)
    out["parent_dir_side_edge"] = pd.to_numeric(p[f"{prefix}dir_side_edge"], errors="coerce").fillna(0.0)
    out["parent_dir_trade_prob"] = pd.to_numeric(p[f"{prefix}dir_trade_prob"], errors="coerce").fillna(0.0)
    out["parent_dir_action"] = pd.to_numeric(p[f"{prefix}dir_action"], errors="coerce").fillna(0).astype(int)
    out["parent_quality_p_cash"] = pd.to_numeric(p[f"{prefix}quality_p_cash"], errors="coerce").fillna(0.0)
    out["parent_quality_p_long"] = pd.to_numeric(p[f"{prefix}quality_p_long"], errors="coerce").fillna(0.0)
    out["parent_quality_p_short"] = pd.to_numeric(p[f"{prefix}quality_p_short"], errors="coerce").fillna(0.0)
    out["parent_quality"] = pd.to_numeric(p[f"{prefix}quality_for_action"], errors="coerce").fillna(0.0)
    out["parent_final_action"] = pd.to_numeric(p[f"{prefix}final_action"], errors="coerce").fillna(0).astype(int)
    return out


def _parent_predictions(split: str) -> pd.DataFrame:
    if split == "train":
        path = TRAIN_PARENT_PATH
        prefix = "omega1_regime3_expertdq_train_"
    elif split == "validation":
        path = PARENT_DIR / "validation_predictions_2025_true3head.csv"
        prefix = "omega1_regime3_expertdq_oof_"
    elif split == "oos":
        path = PARENT_DIR / "oos_predictions_2026_true3head.csv"
        prefix = "omega1_regime3_expertdq_"
    else:
        raise RuntimeError(split)
    p = pd.read_csv(path, parse_dates=["timestamp"])
    return _normalize_parent_predictions(p, prefix)


def _feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr_pct_12"] = tr.rolling(12, min_periods=1).mean() / close
    df["atr_pct_48"] = tr.rolling(48, min_periods=1).mean() / close
    df["atr_pct_96"] = tr.rolling(96, min_periods=1).mean() / close
    df["ret_3"] = close.pct_change(3)
    df["ret_12"] = close.pct_change(12)
    df["ret_48"] = close.pct_change(48)
    df["ret_96"] = close.pct_change(96)
    df["range_pct"] = (high - low) / close
    df["body_pct"] = (close - open_).abs() / close
    df["upper_wick_pct"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / close
    df["lower_wick_pct"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / close
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dow"] = df["timestamp"].dt.dayofweek
    return df


def _feature_cols(train: pd.DataFrame, oos: pd.DataFrame) -> list[str]:
    banned = {
        "timestamp",
        "label_side",
        "label_side_id",
        "label_horizon",
        "label_tp_price_move",
        "label_sl_price_move",
        "label_utility",
        "label_net_return",
        "label_hold_bars",
        "label_reason",
        "label_mfe",
        "label_mae",
        "split",
    }
    forbidden_prefixes = (
        "clean_regime4_",
        "regime4_pred_",
        "teacher_",
        "dp_",
        "trend_",
        "mfe_",
        "mae_",
        "quantile_",
        "survival_",
        "hazard_",
        "bandit_",
    )
    forbidden_exact = {"tp_sl_action_score"}
    common = [
        c
        for c in train.columns
        if c in oos.columns
        and c not in banned
        and c not in forbidden_exact
        and not any(c.startswith(prefix) for prefix in forbidden_prefixes)
    ]
    cols: list[str] = []
    for c in common:
        if pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(oos[c]):
            cols.append(c)
    if not cols:
        raise RuntimeError("no allowed numeric feature columns")
    return cols


def _class_from_side(side: pd.Series) -> pd.Series:
    return side.map({0: 0, 1: 1, -1: 2}).fillna(0).astype(int)


def _side_from_class(cls: int) -> int:
    if int(cls) == 1:
        return 1
    if int(cls) == 2:
        return -1
    return 0


def _fit_models(train: pd.DataFrame, cols: list[str], *, label_min_utility: float) -> dict[str, Any]:
    y_side_raw = pd.to_numeric(train["label_side_id"], errors="coerce").fillna(0).astype(int)
    util = pd.to_numeric(train["label_utility"], errors="coerce").fillna(0.0)
    y_side_raw = y_side_raw.where(util >= float(label_min_utility), 0)
    y_action = _class_from_side(y_side_raw)
    sample_weight = np.where(y_action.to_numpy() == 0, 0.65, 1.0 + np.clip(util.to_numpy(), 0.0, 0.01) * 80.0)
    x = train[cols].replace([np.inf, -np.inf], np.nan)
    action = HistGradientBoostingClassifier(max_iter=35, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=0.05, early_stopping=False, random_state=260620)
    action.fit(x, y_action, sample_weight=sample_weight)
    active = y_action != 0
    if int(active.sum()) < 100:
        raise RuntimeError("too few active labels")
    horizon = HistGradientBoostingClassifier(max_iter=30, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=0.05, early_stopping=False, random_state=260621)
    horizon.fit(x.loc[active], pd.to_numeric(train.loc[active, "label_horizon"], errors="raise").astype(int), sample_weight=sample_weight[active.to_numpy()])
    tp = HistGradientBoostingRegressor(max_iter=30, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=0.05, early_stopping=False, random_state=260622)
    sl = HistGradientBoostingRegressor(max_iter=30, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=0.05, early_stopping=False, random_state=260623)
    tp.fit(x.loc[active], pd.to_numeric(train.loc[active, "label_tp_price_move"], errors="raise"), sample_weight=sample_weight[active.to_numpy()])
    sl.fit(x.loc[active], pd.to_numeric(train.loc[active, "label_sl_price_move"], errors="raise"), sample_weight=sample_weight[active.to_numpy()])
    return {"action": action, "horizon": horizon, "tp": tp, "sl": sl, "label_min_utility": float(label_min_utility)}


def _signals(frame: pd.DataFrame, cols: list[str], models: dict[str, Any], cfg: dict[str, Any], parents: pd.DataFrame | None) -> pd.DataFrame:
    x = frame[cols].replace([np.inf, -np.inf], np.nan)
    proba = models["action"].predict_proba(x)
    classes = list(models["action"].classes_)
    pmap = {int(c): proba[:, i] for i, c in enumerate(classes)}
    p_cash = pmap.get(0, np.zeros(len(frame)))
    p_long = pmap.get(1, np.zeros(len(frame)))
    p_short = pmap.get(2, np.zeros(len(frame)))
    pred_cls = np.where(p_long >= p_short, 1, 2)
    pred_trade_p = np.maximum(p_long, p_short)
    side_edge = np.abs(p_long - p_short)
    out = pd.DataFrame({"timestamp": frame["timestamp"], "pred_cls": pred_cls, "pred_trade_p": pred_trade_p, "pred_side_edge": side_edge, "p_cash": p_cash, "p_long": p_long, "p_short": p_short})
    active = (out["pred_trade_p"] >= float(cfg["trade_threshold"])) & (out["pred_side_edge"] >= float(cfg["side_edge_min"]))
    out["side"] = [(_side_from_class(c) if a else 0) for c, a in zip(out["pred_cls"], active)]
    out["horizon"] = 0
    out["tp_price_move"] = 0.0
    out["sl_price_move"] = 0.0
    if active.any():
        xa = x.loc[active]
        out.loc[active, "horizon"] = models["horizon"].predict(xa).astype(int)
        out.loc[active, "tp_price_move"] = np.clip(models["tp"].predict(xa), 0.006, 0.050)
        out.loc[active, "sl_price_move"] = np.clip(models["sl"].predict(xa), 0.004, 0.035)
    if parents is not None and float(cfg["parent_min_quality"]) > 0.0:
        out = out.merge(parents, on="timestamp", how="left")
        pside = out["parent_dir_action"].map({1: 1, 2: -1}).fillna(0).astype(int)
        pq = pd.to_numeric(out["parent_quality"], errors="coerce").fillna(0.0)
        disagree = (pq >= float(cfg["parent_min_quality"])) & (pside != 0) & (pside != out["side"])
        out.loc[disagree, ["side", "horizon", "tp_price_move", "sl_price_move"]] = [0, 0, 0.0, 0.0]
    out["margin_fraction"] = float(cfg["margin_fraction"])
    out["notional"] = out["margin_fraction"] * LEVERAGE
    return out


def _run(frame: pd.DataFrame, signals: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    sigs = signals.set_index("timestamp")
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    rows: list[dict[str, Any]] = []
    i = 0
    trade_id = 1
    while i < len(frame) - 2:
        s = sigs.loc[frame.iloc[i]["timestamp"]]
        side = int(s["side"])
        if side == 0:
            i += 1
            continue
        entry_i = i + 1
        entry = float(frame.iloc[entry_i]["open"])
        if entry <= 0.0:
            i += 1
            continue
        horizon = int(s["horizon"])
        if USE_PREDICTED_HORIZON_MAX_HOLD and horizon > 0:
            end_i = min(len(frame) - 1, entry_i + horizon)
        else:
            end_i = len(frame) - 1
        exit_i = end_i
        exit_px = float(frame.iloc[end_i]["close"])
        reason = "forced_end"
        mfe = 0.0
        mae = 0.0
        tp = float(s["tp_price_move"])
        sl = float(s["sl_price_move"])
        notional = float(s["notional"])
        for j in range(entry_i, end_i + 1):
            high = float(frame.iloc[j]["high"])
            low = float(frame.iloc[j]["low"])
            if side > 0:
                hi_raw = (high - entry) / entry
                lo_raw = (low - entry) / entry
            else:
                hi_raw = (entry - low) / entry
                lo_raw = (entry - high) / entry
            mfe = max(mfe, hi_raw)
            mae = min(mae, lo_raw)
            hit_tp = hi_raw >= tp
            hit_sl = lo_raw <= -abs(sl)
            if hit_tp or hit_sl:
                exit_i = j
                if hit_sl and hit_tp:
                    reason = "stop_loss"
                    exit_px = entry * (1.0 - side * sl)
                elif hit_tp:
                    reason = "take_profit"
                    exit_px = entry * (1.0 + side * tp)
                else:
                    reason = "stop_loss"
                    exit_px = entry * (1.0 - side * sl)
                break
        cash -= cash * FEE_PER_SIDE * notional
        raw = (exit_px - entry) / entry if side > 0 else (entry - exit_px) / entry
        pnl_frac = raw * notional - FEE_PER_SIDE * notional
        before = cash
        cash *= 1.0 + pnl_frac
        peak = max(peak, cash)
        mdd = min(mdd, cash / peak - 1.0)
        rows.append(
            {
                "trade_id": trade_id,
                "side": "LONG" if side > 0 else "SHORT",
                "entry_time": frame.iloc[entry_i]["timestamp"],
                "exit_time": frame.iloc[exit_i]["timestamp"],
                "entry_price": entry,
                "exit_price": exit_px,
                "horizon_bars": horizon,
                "max_hold_enabled": bool(USE_PREDICTED_HORIZON_MAX_HOLD),
                "hold_bars": exit_i - entry_i,
                "notional_exposure": notional,
                "margin_fraction": float(s["margin_fraction"]),
                "execution_leverage": LEVERAGE,
                "tp_price_move": tp,
                "sl_price_move": sl,
                "take_profit": tp * notional,
                "stop_loss": sl * notional,
                "gross_raw_ret": raw,
                "net_trade_return_pct": pnl_frac * 100.0,
                "cash_before": before,
                "cash_after": cash,
                "mfe_pct": mfe * 100.0,
                "mae_pct": mae * 100.0,
                "exit_reason": reason,
                "pred_trade_p": float(s["pred_trade_p"]),
                "pred_side_edge": float(s["pred_side_edge"]),
            }
        )
        trade_id += 1
        i = exit_i + 1
    ledger = pd.DataFrame(rows)
    wins = int((ledger["net_trade_return_pct"] > 0.0).sum()) if len(ledger) else 0
    days = max(1.0, len(frame) * 5.0 / 1440.0)
    holds = ledger["hold_bars"].astype(int).tolist() if len(ledger) else []
    metrics = {
        "pnl_pct": (cash - 1.0) * 100.0,
        "mdd_pct": mdd * 100.0,
        "wr": wins / len(ledger) if len(ledger) else 0.0,
        "trades": int(len(ledger)),
        "trades_per_day": float(len(ledger) / days),
        "avg_hold_bars": float(sum(holds) / len(holds)) if holds else 0.0,
        "median_hold_bars": float(median(holds)) if holds else 0.0,
        "max_hold_bars": int(max(holds)) if holds else 0,
        "exit_reasons": ledger["exit_reason"].astype(str).value_counts().to_dict() if len(ledger) else {},
        "horizon_counts": ledger["horizon_bars"].astype(str).value_counts().to_dict() if len(ledger) else {},
    }
    return metrics, ledger


def _score(m: dict[str, Any]) -> float:
    pnl = float(m["pnl_pct"])
    tpd = float(m["trades_per_day"])
    mdd = abs(float(m["mdd_pct"]))
    return pnl + min(tpd, 3.0) * 8.0 - max(0.0, mdd - 15.0) * 4.0 - max(0.0, 1.0 - tpd) * 40.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_2025 = pd.read_csv(TRAIN_CSV)
    raw_oos = pd.read_csv(EVAL_CSV)
    labels = pd.read_csv(LABEL_2025)
    raw_2025["timestamp"] = pd.to_datetime(raw_2025["timestamp"])
    raw_oos["timestamp"] = pd.to_datetime(raw_oos["timestamp"])
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    labels = labels.drop(columns=["close"], errors="ignore")
    train_parent = _parent_predictions("train")
    val_parent = _parent_predictions("validation")
    oos_parent = _parent_predictions("oos")
    f2025_base = _feature_frame(raw_2025).merge(labels, on="timestamp", how="inner")
    f2025_base = pd.concat(
        [
            f2025_base[f2025_base["timestamp"] < SPLIT_TS].merge(train_parent, on="timestamp", how="left", validate="one_to_one"),
            f2025_base[f2025_base["timestamp"] >= SPLIT_TS].merge(val_parent, on="timestamp", how="left", validate="one_to_one"),
        ],
        ignore_index=True,
    ).sort_values("timestamp").reset_index(drop=True)
    parent_cols = [c for c in train_parent.columns if c != "timestamp"]
    dropped_parent_na_2025 = int(f2025_base[parent_cols].isna().any(axis=1).sum())
    if dropped_parent_na_2025:
        f2025_base = f2025_base.loc[~f2025_base[parent_cols].isna().any(axis=1)].reset_index(drop=True)
    f2025 = f2025_base
    foos = _feature_frame(raw_oos)
    oos_label_times = pd.read_csv(SOURCE_DIR / "oos_2026_multihorizon_tb_labels.csv", usecols=["timestamp"], parse_dates=["timestamp"])
    foos = foos.merge(oos_label_times, on="timestamp", how="inner").merge(oos_parent, on="timestamp", how="left", validate="one_to_one")
    dropped_parent_na_oos = int(foos[parent_cols].isna().any(axis=1).sum())
    if dropped_parent_na_oos:
        foos = foos.loc[~foos[parent_cols].isna().any(axis=1)].reset_index(drop=True)
    train = f2025[f2025["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val = f2025[f2025["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    cols = _feature_cols(train, foos)
    val_parents = val_parent
    oos_parents = oos_parent
    ranking: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    for label_min_utility in LABEL_MIN_UTILITY_GRID:
        print(json.dumps({"stage": "fit_start", "label_min_utility": label_min_utility}), flush=True)
        models = _fit_models(train, cols, label_min_utility=float(label_min_utility))
        print(json.dumps({"stage": "fit_done", "label_min_utility": label_min_utility, "feature_count": len(cols)}), flush=True)
        train_pred = models["action"].predict(train[cols].replace([np.inf, -np.inf], np.nan))
        train_label = _class_from_side(pd.to_numeric(train["label_side_id"], errors="coerce").fillna(0).astype(int).where(pd.to_numeric(train["label_utility"], errors="coerce").fillna(0.0) >= label_min_utility, 0))
        train_acc = float(accuracy_score(train_label, train_pred))
        for trade_threshold in TRADE_THRESHOLD_GRID:
            for side_edge_min in SIDE_EDGE_MIN_GRID:
                for parent_min_quality in PARENT_MIN_QUALITY_GRID:
                    for margin_fraction in MARGIN_FRACTION_GRID:
                        cfg = {
                            "label_min_utility": float(label_min_utility),
                            "trade_threshold": float(trade_threshold),
                            "side_edge_min": float(side_edge_min),
                            "parent_min_quality": float(parent_min_quality),
                            "margin_fraction": float(margin_fraction),
                        }
                        sig = _signals(val, cols, models, cfg, val_parents)
                        val_m, _ = _run(val, sig)
                        row = {**cfg, **{f"val_{k}": v for k, v in val_m.items() if not isinstance(v, dict)}, "train_action_acc": train_acc, "score": _score(val_m)}
                        ranking.append(row)
                        if best is None or row["score"] > best[0]:
                            best = (float(row["score"]), cfg, models, val_m)
    if best is None:
        raise RuntimeError("no HGB candidate")
    ranking_df = pd.DataFrame(ranking).sort_values(["score", "val_pnl_pct"], ascending=False)
    ranking_df.to_csv(OUT_DIR / "hgb_validation_ranking.csv", index=False)
    ranking_for_selection = pd.DataFrame(ranking)
    eligible = ranking_for_selection[(ranking_for_selection["val_pnl_pct"] > 0.0) & (ranking_for_selection["val_trades_per_day"] >= 1.0)].copy()
    if len(eligible) == 0:
        eligible = ranking_for_selection[ranking_for_selection["val_pnl_pct"] > 0.0].copy()
    if len(eligible) == 0:
        eligible = ranking_for_selection.copy()
    chosen_row = eligible.sort_values(["val_pnl_pct", "val_trades_per_day"], ascending=False).iloc[0].to_dict()
    cfg = {
        "label_min_utility": float(chosen_row["label_min_utility"]),
        "trade_threshold": float(chosen_row["trade_threshold"]),
        "side_edge_min": float(chosen_row["side_edge_min"]),
        "parent_min_quality": float(chosen_row["parent_min_quality"]),
        "margin_fraction": float(chosen_row["margin_fraction"]),
    }
    models = _fit_models(train, cols, label_min_utility=float(cfg["label_min_utility"]))
    val_sig = _signals(val, cols, models, cfg, val_parents)
    oos_sig = _signals(foos, cols, models, cfg, oos_parents)
    val_m, val_ledger = _run(val, val_sig)
    oos_m, oos_ledger = _run(foos, oos_sig)
    val_ledger.to_csv(OUT_DIR / "selected_hgb_validation_ledger.csv", index=False)
    oos_ledger.to_csv(OUT_DIR / "selected_hgb_oos_ledger.csv", index=False)
    joblib.dump({"models": models, "feature_cols": cols, "selected_cfg": cfg}, OUT_DIR / "hgb_multihorizon_tb_bundle.joblib")
    report = {
        "model_id": MODEL_ID,
        "run_id": RUN_ID,
        "status": "hgb_eval_complete",
        "design": "sklearn HistGradientBoosting multi-head approximation: action classifier, horizon classifier, TP/SL regressors trained on 2025 Jan-Sep labels. Validation selects thresholds; OOS is untouched for selection.",
        "selected": cfg,
        "validation": val_m,
        "oos": oos_m,
        "feature_count": len(cols),
        "parent_feature_contract": {
            "train_parent_prediction_type": "in_sample_parent_inference_not_oof",
            "dropped_2025_rows_without_parent": dropped_parent_na_2025,
            "dropped_oos_rows_without_parent": dropped_parent_na_oos,
        },
        "risk_contract": {
            "notional": "margin_fraction * execution_leverage",
            "execution_leverage": LEVERAGE,
            "take_profit": "tp_price_move * notional",
            "stop_loss": "sl_price_move * notional",
            "predicted_horizon_max_hold_enabled": bool(USE_PREDICTED_HORIZON_MAX_HOLD),
        },
        "artifacts": {
            "ranking": str((OUT_DIR / "hgb_validation_ranking.csv").relative_to(ROOT)),
            "bundle": str((OUT_DIR / "hgb_multihorizon_tb_bundle.joblib").relative_to(ROOT)),
            "validation_ledger": str((OUT_DIR / "selected_hgb_validation_ledger.csv").relative_to(ROOT)),
            "oos_ledger": str((OUT_DIR / "selected_hgb_oos_ledger.csv").relative_to(ROOT)),
        },
        "redteam_notes": [
            "Research evaluation only; no promotion PASS.",
            "2025 Jan-Sep parent features are in-sample parent inference, not OOF; use only for research ablation.",
            "Validation-only selection is respected.",
            "Fresh forward OOS and runtime-native replay are required before promotion.",
        ],
    }
    (OUT_DIR / "hgb_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
