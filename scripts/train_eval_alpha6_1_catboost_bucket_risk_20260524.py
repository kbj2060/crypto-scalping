#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_eval_alpha6_1_catboost_parent_baseline_20260521 import (  # noqa: E402
    DEFAULT_LABEL_DIR,
    DEFAULT_RAW_2025,
    DEFAULT_RAW_2026,
    DEFAULT_SPEC_DIR,
    CatSpec,
    _balanced_weights,
    _binary_proba,
    _build_projection,
    _cat_specs,
    _compose_policy,
    _fit_cat,
    _read_spec,
    _sanitize_feature_cols,
)
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _days, _fill_price, _json_default, _read  # noqa: E402


MODEL_ID = "alpha6_1_catboost_bucket_risk_20260524"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha6_1_catboost_bucket_risk_20260524"


@dataclass(frozen=True)
class RiskTemplate:
    name: str
    notional: float
    leverage: float
    tp_atr_mult: float
    sl_atr_mult: float


class ConstantBinaryModel:
    def __init__(self, p: float) -> None:
        self.p = float(np.clip(p, 0.0, 1.0))

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.full(len(x), self.p, dtype=np.float64)
        return np.vstack([1.0 - p, p]).T


RISK_TEMPLATES = [
    RiskTemplate("n010_l1_tp15_sl10", 0.10, 1.0, 1.5, 1.0),
    RiskTemplate("n015_l1_tp20_sl12", 0.15, 1.0, 2.0, 1.2),
    RiskTemplate("n025_l1_tp20_sl12", 0.25, 1.0, 2.0, 1.2),
    RiskTemplate("n025_l2_tp20_sl12", 0.25, 2.0, 2.0, 1.2),
    RiskTemplate("n035_l2_tp25_sl15", 0.35, 2.0, 2.5, 1.5),
    RiskTemplate("n015_l3_tp20_sl12", 0.15, 3.0, 2.0, 1.2),
    RiskTemplate("n020_l3_tp25_sl15", 0.20, 3.0, 2.5, 1.5),
    RiskTemplate("n025_l3_tp30_sl18", 0.25, 3.0, 3.0, 1.8),
]


def _cat_spec_by_name(name: str) -> CatSpec:
    specs = {s.name: s for s in _cat_specs()}
    if name not in specs:
        raise ValueError(f"unknown cat spec {name}; choices={sorted(specs)}")
    return specs[name]


def _template_values() -> dict[str, list[float]]:
    return {
        "notional": sorted({float(t.notional) for t in RISK_TEMPLATES}),
        "leverage": sorted({float(t.leverage) for t in RISK_TEMPLATES}),
        "tp": sorted({float(t.tp_atr_mult) for t in RISK_TEMPLATES}),
        "sl": sorted({float(t.sl_atr_mult) for t in RISK_TEMPLATES}),
    }


def _risk_features(x: pd.DataFrame, p_entry: np.ndarray, p_long: np.ndarray, actions: np.ndarray) -> pd.DataFrame:
    p_entry = np.asarray(p_entry, dtype=np.float64)
    p_long = np.asarray(p_long, dtype=np.float64)
    extra = pd.DataFrame(
        {
            "cb_p_entry": p_entry,
            "cb_p_long": p_long,
            "cb_p_short": 1.0 - p_long,
            "cb_side_margin": np.abs(p_long - (1.0 - p_long)),
            "cb_is_long": (np.asarray(actions) == 1).astype(np.float64),
            "cb_is_short": (np.asarray(actions) == 2).astype(np.float64),
        }
    )
    return pd.concat([x.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)


def _tp_sl(frame: pd.DataFrame, tp_mult: float, sl_mult: float) -> tuple[np.ndarray, np.ndarray]:
    atr = pd.to_numeric(frame.get("atr14_pct", 0.003), errors="coerce").fillna(0.003).to_numpy(dtype=np.float64)
    fallback_tp = pd.to_numeric(frame.get("label_tp_pct", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    fallback_sl = pd.to_numeric(frame.get("label_sl_pct", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    tp = np.clip(np.maximum(atr * float(tp_mult), fallback_tp * 0.5), 5e-4, 0.05)
    sl = np.clip(np.maximum(atr * float(sl_mult), fallback_sl * 0.5), 5e-4, 0.05)
    return tp, sl


def _path_pnl(
    frame: pd.DataFrame,
    idx: int,
    side_cls: int,
    *,
    notional: float,
    leverage: float,
    tp_pct: np.ndarray,
    sl_pct: np.ndarray,
    fee: float,
    slip: float,
    max_hold: int,
) -> tuple[float, float]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    side = 1 if int(side_cls) == 1 else -1
    exposure = float(notional) * float(leverage)
    entry_i = min(int(idx) + 1, len(frame) - 1)
    entry = max(float(close[entry_i]), 1e-12)
    end = min(entry_i + int(max_hold), len(frame) - 1)
    raw = 0.0
    mae = 0.0
    for j in range(entry_i + 1, end + 1):
        if side > 0:
            fav = float(high[j] / entry - 1.0)
            adv = float(low[j] / entry - 1.0)
        else:
            fav = float(entry / max(low[j], 1e-12) - 1.0)
            adv = float(entry / max(high[j], 1e-12) - 1.0)
        mae = max(mae, max(0.0, -adv * exposure))
        if adv <= -float(sl_pct[idx]):
            raw = -float(sl_pct[idx])
            break
        if fav >= float(tp_pct[idx]):
            raw = float(tp_pct[idx])
            break
    else:
        px = max(float(close[end]), 1e-12)
        raw = (px - entry) / entry if side > 0 else (entry - px) / entry
    pnl = raw * exposure - 2.0 * (float(fee) + float(slip)) * exposure
    return float(pnl), float(mae)


def _best_template_labels(
    frame: pd.DataFrame,
    actions: np.ndarray,
    candidate_idx: np.ndarray,
    *,
    fee: float,
    slip: float,
    max_hold: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    values = _template_values()
    value_to_id = {k: {v: i for i, v in enumerate(vs)} for k, vs in values.items()}
    labels = {k: np.zeros(len(candidate_idx), dtype=np.int64) for k in ("notional", "leverage", "tp", "sl")}
    quality = np.zeros(len(candidate_idx), dtype=np.int64)
    best_names: list[str] = []
    positive = 0
    tp_sl_cache = {(t.tp_atr_mult, t.sl_atr_mult): _tp_sl(frame, t.tp_atr_mult, t.sl_atr_mult) for t in RISK_TEMPLATES}
    for out_i, idx in enumerate(candidate_idx):
        best_score = -1e9
        best_tpl = RISK_TEMPLATES[0]
        for tpl in RISK_TEMPLATES:
            tp, sl = tp_sl_cache[(tpl.tp_atr_mult, tpl.sl_atr_mult)]
            pnl, mae = _path_pnl(
                frame,
                int(idx),
                int(actions[int(idx)]),
                notional=tpl.notional,
                leverage=tpl.leverage,
                tp_pct=tp,
                sl_pct=sl,
                fee=float(fee),
                slip=float(slip),
                max_hold=int(max_hold),
            )
            score = pnl - 0.45 * mae - 0.00015 * (tpl.notional * tpl.leverage)
            if score > best_score:
                best_score = score
                best_tpl = tpl
        quality[out_i] = int(best_score > 0.0)
        positive += int(quality[out_i])
        best_names.append(best_tpl.name)
        labels["notional"][out_i] = value_to_id["notional"][float(best_tpl.notional)]
        labels["leverage"][out_i] = value_to_id["leverage"][float(best_tpl.leverage)]
        labels["tp"][out_i] = value_to_id["tp"][float(best_tpl.tp_atr_mult)]
        labels["sl"][out_i] = value_to_id["sl"][float(best_tpl.sl_atr_mult)]
    diag = {
        "candidates": int(len(candidate_idx)),
        "positive_best_score_rate": float(positive / max(len(candidate_idx), 1)),
        "template_counts": {k: int(v) for k, v in pd.Series(best_names).value_counts().to_dict().items()},
        "bucket_values": values,
    }
    return labels, quality, diag


def _template_score_matrix(
    frame: pd.DataFrame,
    actions: np.ndarray,
    candidate_idx: np.ndarray,
    *,
    fee: float,
    slip: float,
    max_hold: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    scores = np.zeros((len(candidate_idx), len(RISK_TEMPLATES)), dtype=np.float64)
    tp_sl_cache = {(t.tp_atr_mult, t.sl_atr_mult): _tp_sl(frame, t.tp_atr_mult, t.sl_atr_mult) for t in RISK_TEMPLATES}
    for out_i, idx in enumerate(candidate_idx):
        for tpl_i, tpl in enumerate(RISK_TEMPLATES):
            tp, sl = tp_sl_cache[(tpl.tp_atr_mult, tpl.sl_atr_mult)]
            pnl, mae = _path_pnl(
                frame,
                int(idx),
                int(actions[int(idx)]),
                notional=tpl.notional,
                leverage=tpl.leverage,
                tp_pct=tp,
                sl_pct=sl,
                fee=float(fee),
                slip=float(slip),
                max_hold=int(max_hold),
            )
            scores[out_i, tpl_i] = pnl - 0.45 * mae - 0.00015 * (tpl.notional * tpl.leverage)
    best_idx = np.argmax(scores, axis=1) if len(scores) else np.zeros(0, dtype=np.int64)
    diag = {
        "candidates": int(len(candidate_idx)),
        "template_positive_rate": {
            tpl.name: float(np.mean(scores[:, i] > 0.0)) if len(scores) else 0.0
            for i, tpl in enumerate(RISK_TEMPLATES)
        },
        "best_template_counts": {
            RISK_TEMPLATES[int(k)].name: int(v)
            for k, v in pd.Series(best_idx).value_counts().to_dict().items()
        },
    }
    return scores, diag


def _fit_bucket_head(x: pd.DataFrame, y: np.ndarray, spec: CatSpec, seed: int, *, task_type: str, devices: str) -> CatBoostClassifier:
    classes = np.unique(y)
    loss = "MultiClass" if len(classes) > 2 else "Logloss"
    model = CatBoostClassifier(
        loss_function=loss,
        eval_metric=loss,
        iterations=int(spec.iterations),
        depth=int(spec.depth),
        learning_rate=float(spec.learning_rate),
        l2_leaf_reg=float(spec.l2_leaf_reg),
        random_strength=float(spec.random_strength),
        bagging_temperature=float(spec.bagging_temperature),
        task_type=str(task_type),
        devices=str(devices),
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x, y, sample_weight=_balanced_weights(y))
    return model


def _fit_template_experts(
    x: pd.DataFrame,
    score_matrix: np.ndarray,
    spec: CatSpec,
    *,
    seed: int,
    task_type: str,
    devices: str,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for i, tpl in enumerate(RISK_TEMPLATES):
        y = (score_matrix[:, i] > 0.0).astype(np.int64)
        if len(np.unique(y)) < 2:
            models[tpl.name] = ConstantBinaryModel(float(np.mean(y)))
            continue
        models[tpl.name] = _fit_bucket_head(
            x,
            y,
            spec,
            int(seed + i * 23),
            task_type=task_type,
            devices=devices,
        )
    return models


def _predict_template_experts(models: dict[str, Any], x: pd.DataFrame, *, exposure_penalty: float) -> tuple[np.ndarray, np.ndarray]:
    score_cols: list[np.ndarray] = []
    for tpl in RISK_TEMPLATES:
        raw = np.asarray(models[tpl.name].predict_proba(x), dtype=np.float64)
        p = raw[:, 1] if raw.ndim == 2 and raw.shape[1] > 1 else raw.reshape(-1)
        adjusted = p - float(exposure_penalty) * float(tpl.notional) * float(tpl.leverage)
        score_cols.append(adjusted)
    scores = np.vstack(score_cols).T if score_cols else np.zeros((len(x), 0), dtype=np.float64)
    best_idx = np.argmax(scores, axis=1) if scores.shape[1] else np.zeros(len(x), dtype=np.int64)
    best_score = scores[np.arange(len(x)), best_idx] if scores.shape[1] else np.zeros(len(x), dtype=np.float64)
    return best_idx.astype(np.int64), best_score.astype(np.float64)


def _arrays_from_template_ids(template_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(template_ids, dtype=np.int64)
    return (
        np.asarray([RISK_TEMPLATES[int(i)].notional for i in ids], dtype=np.float64),
        np.asarray([RISK_TEMPLATES[int(i)].leverage for i in ids], dtype=np.float64),
        np.asarray([RISK_TEMPLATES[int(i)].tp_atr_mult for i in ids], dtype=np.float64),
        np.asarray([RISK_TEMPLATES[int(i)].sl_atr_mult for i in ids], dtype=np.float64),
    )


def _predict_bucket(model: CatBoostClassifier, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(x), dtype=np.int64).reshape(-1)


def _dynamic_bucket_backtest(
    frame: pd.DataFrame,
    actions: np.ndarray,
    notional: np.ndarray,
    leverage: np.ndarray,
    tp_mult: np.ndarray,
    sl_mult: np.ndarray,
    *,
    fee: float,
    slip: float,
    max_hold: int,
) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    tp_cache = {}
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_i = -1
    entry_equity = 1.0
    cur_notional = 0.0
    cur_leverage = 1.0
    cur_tp = 0.0
    cur_sl = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = exposure_sum = 0.0
    exits: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}

    def exposure() -> float:
        return float(cur_notional) * float(cur_leverage)

    def equity(i: int) -> float:
        if side == 0:
            return cash
        raw = (close[i] - entry) / max(entry, 1e-12) if side > 0 else (entry - close[i]) / max(entry, 1e-12)
        return cash * (1.0 + raw * exposure())

    def exit_pos(i: int, reason: str, fill_px: float | None = None) -> None:
        nonlocal cash, side, entry, entry_i, cur_notional, cur_leverage, cur_tp, cur_sl, trades, wins
        if fill_px is None:
            fill_px = _fill_price(frame, min(i + 1, len(frame) - 1), side, float(slip), entry=False)
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure())
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        side = 0
        entry = 0.0
        entry_i = -1
        cur_notional = 0.0
        cur_leverage = 1.0
        cur_tp = 0.0
        cur_sl = 0.0

    for i in range(len(frame) - 2):
        desired = int(actions[i])
        if side != 0:
            hold = i - entry_i
            if side > 0:
                tp_hit = high[i] >= entry * (1.0 + cur_tp)
                sl_hit = low[i] <= entry * (1.0 - cur_sl)
                if tp_hit and sl_hit:
                    exit_pos(i, "ambiguous_bar_sl_first", entry * (1.0 - cur_sl) * (1.0 - float(slip)))
                elif tp_hit:
                    exit_pos(i, "tp", entry * (1.0 + cur_tp) * (1.0 - float(slip)))
                elif sl_hit:
                    exit_pos(i, "sl", entry * (1.0 - cur_sl) * (1.0 - float(slip)))
            else:
                tp_hit = low[i] <= entry * (1.0 - cur_tp)
                sl_hit = high[i] >= entry * (1.0 + cur_sl)
                if tp_hit and sl_hit:
                    exit_pos(i, "ambiguous_bar_sl_first", entry * (1.0 + cur_sl) * (1.0 + float(slip)))
                elif tp_hit:
                    exit_pos(i, "tp", entry * (1.0 - cur_tp) * (1.0 + float(slip)))
                elif sl_hit:
                    exit_pos(i, "sl", entry * (1.0 + cur_sl) * (1.0 + float(slip)))
            if side != 0 and hold >= int(max_hold):
                exit_pos(i, "max_hold")
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side == 0 and desired != 0:
            cur_notional = float(notional[i])
            cur_leverage = float(leverage[i])
            key = (float(tp_mult[i]), float(sl_mult[i]))
            if key not in tp_cache:
                tp_cache[key] = _tp_sl(frame, key[0], key[1])
            tp_vec, sl_vec = tp_cache[key]
            cur_tp = float(tp_vec[i])
            cur_sl = float(sl_vec[i])
            side = 1 if desired == 1 else -1
            entry_i = int(i)
            entry = _fill_price(frame, min(i + 1, len(frame) - 1), side, float(slip), entry=True)
            entry_equity = cash
            cash -= cash * float(fee) * exposure()
            long_entries += int(side > 0)
            short_entries += int(side < 0)
            notional_sum += cur_notional
            leverage_sum += cur_leverage
            exposure_sum += exposure()
            bucket_key = f"n{cur_notional:.2f}_l{cur_leverage:.1f}_tp{float(tp_mult[i]):.1f}_sl{float(sl_mult[i]):.1f}"
            bucket_counts[bucket_key] = bucket_counts.get(bucket_key, 0) + 1
    if side != 0:
        exit_pos(len(frame) - 2, "end")
    n = max(long_entries + short_entries, 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "calmar": float(((cash - 1.0) * 100.0) / max(abs(mdd * 100.0), 1e-12)),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(frame)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n),
        "avg_leverage": float(leverage_sum / n),
        "avg_exposure": float(exposure_sum / n),
        "exits": exits,
        "bucket_counts": bucket_counts,
    }


def _eval_dynamic(frame: pd.DataFrame, actions: np.ndarray, notional: np.ndarray, leverage: np.ndarray, tp: np.ndarray, sl: np.ndarray, *, fee: float, slip: float, max_hold: int) -> dict[str, Any]:
    return {
        f"cost{m}": _dynamic_bucket_backtest(frame, actions, notional, leverage, tp, sl, fee=float(fee) * m, slip=float(slip) * m, max_hold=int(max_hold))
        for m in (1, 2, 3)
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alpha6.1 CatBoost parent with CatBoost bucket heads for notional/leverage/TP/SL.")
    p.add_argument("--variant", default="stable48_global_pca32")
    p.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    p.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    p.add_argument("--train-file", default="alpha5_24_entry_rebalanced_train.parquet")
    p.add_argument("--val-file", default="alpha5_24_entry_rebalanced_val.parquet")
    p.add_argument("--oos-file", default="alpha5_24_entry_rebalanced_oos.parquet")
    p.add_argument("--raw-2025-csv", type=Path, default=DEFAULT_RAW_2025)
    p.add_argument("--raw-2026-csv", type=Path, default=DEFAULT_RAW_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--entry-spec", default="regularized")
    p.add_argument("--direction-spec", default="regularized")
    p.add_argument("--risk-spec", default="regularized")
    p.add_argument("--task-type", default="CPU")
    p.add_argument("--devices", default="0")
    p.add_argument("--entry-threshold", type=float, default=0.80)
    p.add_argument("--side-threshold", type=float, default=0.80)
    p.add_argument("--margin-threshold", type=float, default=0.05)
    p.add_argument("--guardrail", default="block_whipsaw_chop")
    p.add_argument("--baseline-tp-atr-mult", type=float, default=2.5)
    p.add_argument("--baseline-sl-atr-mult", type=float, default=1.0)
    p.add_argument("--risk-mode", choices=["independent", "governor"], default="governor")
    p.add_argument("--quality-thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--quality-min-trades", type=int, default=20)
    p.add_argument("--governor-score-thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    p.add_argument("--governor-exposure-penalty", type=float, default=0.04)
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=62161)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit = _verify_state24_sticky090_inputs(_read(args.raw_2025_csv), _read(args.raw_2026_csv), args.manifest, args.clean4_report)
    train_df = pd.read_parquet(args.label_dir / str(args.train_file))
    val_df = pd.read_parquet(args.label_dir / str(args.val_file))
    oos_df = pd.read_parquet(args.label_dir / str(args.oos_file))
    spec = _read_spec(args.spec_dir, str(args.variant))
    feature_cols, leak_audit = _sanitize_feature_cols(train_df, list(spec.get("features", [])))
    x_train_all, (x_val_all, x_oos_all), projection_meta, projection = _build_projection(
        train_df,
        [val_df, oos_df],
        feature_cols,
        enable_pca=bool(spec.get("extra_pca_enable", False)),
        pca_components=int(spec.get("extra_pca_components", 0) or 0),
    )
    entry_spec = _cat_spec_by_name(str(args.entry_spec))
    direction_spec = _cat_spec_by_name(str(args.direction_spec))
    risk_spec = _cat_spec_by_name(str(args.risk_spec))
    entry_mask = pd.to_numeric(train_df["entry_train_keep"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    dir_mask = pd.to_numeric(train_df["direction_train_keep"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    entry_model = _fit_cat(
        x_train_all.loc[entry_mask].reset_index(drop=True),
        pd.to_numeric(train_df.loc[entry_mask, "entry_label"], errors="coerce").fillna(0).to_numpy(np.int64),
        np.clip(pd.to_numeric(train_df.loc[entry_mask, "entry_sample_weight"], errors="coerce").fillna(0).to_numpy(np.float64), 1e-4, None)
        * _balanced_weights(pd.to_numeric(train_df.loc[entry_mask, "entry_label"], errors="coerce").fillna(0).to_numpy(np.int64)),
        entry_spec,
        int(args.seed + 11),
        task_type=str(args.task_type),
        devices=str(args.devices),
    )
    direction_model = _fit_cat(
        x_train_all.loc[dir_mask].reset_index(drop=True),
        (pd.to_numeric(train_df.loc[dir_mask, "direction_label"], errors="coerce").fillna(0).to_numpy(np.int64) == 1).astype(np.int64),
        np.clip(pd.to_numeric(train_df.loc[dir_mask, "direction_sample_weight"], errors="coerce").fillna(0).to_numpy(np.float64), 1e-4, None)
        * _balanced_weights((pd.to_numeric(train_df.loc[dir_mask, "direction_label"], errors="coerce").fillna(0).to_numpy(np.int64) == 1).astype(np.int64)),
        direction_spec,
        int(args.seed + 29),
        task_type=str(args.task_type),
        devices=str(args.devices),
    )
    p_entry_train = _binary_proba(entry_model, x_train_all)
    p_long_train = _binary_proba(direction_model, x_train_all)
    p_entry_val = _binary_proba(entry_model, x_val_all)
    p_long_val = _binary_proba(direction_model, x_val_all)
    p_entry_oos = _binary_proba(entry_model, x_oos_all)
    p_long_oos = _binary_proba(direction_model, x_oos_all)

    train_actions, _, _, _ = _compose_policy(
        train_df,
        p_entry_train,
        p_long_train,
        entry_threshold=float(args.entry_threshold),
        side_threshold=float(args.side_threshold),
        margin_threshold=float(args.margin_threshold),
        tp_atr_mult=float(args.baseline_tp_atr_mult),
        sl_atr_mult=float(args.baseline_sl_atr_mult),
        guardrail=str(args.guardrail),
    )
    val_actions, _, _, _ = _compose_policy(
        val_df,
        p_entry_val,
        p_long_val,
        entry_threshold=float(args.entry_threshold),
        side_threshold=float(args.side_threshold),
        margin_threshold=float(args.margin_threshold),
        tp_atr_mult=float(args.baseline_tp_atr_mult),
        sl_atr_mult=float(args.baseline_sl_atr_mult),
        guardrail=str(args.guardrail),
    )
    oos_actions, _, _, _ = _compose_policy(
        oos_df,
        p_entry_oos,
        p_long_oos,
        entry_threshold=float(args.entry_threshold),
        side_threshold=float(args.side_threshold),
        margin_threshold=float(args.margin_threshold),
        tp_atr_mult=float(args.baseline_tp_atr_mult),
        sl_atr_mult=float(args.baseline_sl_atr_mult),
        guardrail=str(args.guardrail),
    )

    train_candidates = np.flatnonzero(train_actions != 0)
    train_candidates = train_candidates[train_candidates < len(train_df) - int(args.max_hold_bars) - 2]
    risk_x_train_all = _risk_features(x_train_all, p_entry_train, p_long_train, train_actions)
    risk_x_val = _risk_features(x_val_all, p_entry_val, p_long_val, val_actions)
    risk_x_oos = _risk_features(x_oos_all, p_entry_oos, p_long_oos, oos_actions)
    values = _template_values()
    risk_models: dict[str, Any]
    model_extra: dict[str, Any]
    bucket_diag: dict[str, Any]
    selector_summary: dict[str, Any]

    if str(args.risk_mode) == "governor":
        score_matrix, bucket_diag = _template_score_matrix(
            train_df,
            train_actions,
            train_candidates,
            fee=float(args.fee),
            slip=float(args.slip),
            max_hold=int(args.max_hold_bars),
        )
        risk_models = _fit_template_experts(
            risk_x_train_all.iloc[train_candidates].reset_index(drop=True),
            score_matrix,
            risk_spec,
            seed=int(args.seed + 401),
            task_type=str(args.task_type),
            devices=str(args.devices),
        )
        val_tpl_id, val_score = _predict_template_experts(risk_models, risk_x_val, exposure_penalty=float(args.governor_exposure_penalty))
        oos_tpl_id, oos_score = _predict_template_experts(risk_models, risk_x_oos, exposure_penalty=float(args.governor_exposure_penalty))
        val_notional, val_leverage, val_tp, val_sl = _arrays_from_template_ids(val_tpl_id)
        oos_notional, oos_leverage, oos_tp, oos_sl = _arrays_from_template_ids(oos_tpl_id)
        best_threshold: dict[str, Any] | None = None
        for threshold in [float(x.strip()) for x in str(args.governor_score_thresholds).split(",") if x.strip()]:
            filtered = val_actions.copy()
            filtered[(filtered != 0) & (val_score < threshold)] = 0
            ev = _eval_dynamic(val_df, filtered, val_notional, val_leverage, val_tp, val_sl, fee=float(args.fee), slip=float(args.slip), max_hold=int(args.max_hold_bars))
            score = float(ev["cost1"]["pnl"]) + 0.35 * float(ev["cost2"]["pnl"]) + 0.10 * float(ev["cost3"]["pnl"]) - 0.20 * abs(float(ev["cost1"]["mdd"]))
            if int(ev["cost1"]["trades"]) < int(args.quality_min_trades):
                score -= 1000.0
            cand = {"threshold": threshold, "score": float(score), "validation": ev, "kept_signals": int(np.sum(filtered != 0))}
            if best_threshold is None or float(cand["score"]) > float(best_threshold["score"]):
                best_threshold = cand
        assert best_threshold is not None
        val_filtered_actions = val_actions.copy()
        val_filtered_actions[(val_filtered_actions != 0) & (val_score < float(best_threshold["threshold"]))] = 0
        oos_filtered_actions = oos_actions.copy()
        oos_filtered_actions[(oos_filtered_actions != 0) & (oos_score < float(best_threshold["threshold"]))] = 0
        val_eval = _eval_dynamic(val_df, val_filtered_actions, val_notional, val_leverage, val_tp, val_sl, fee=float(args.fee), slip=float(args.slip), max_hold=int(args.max_hold_bars))
        oos_eval = _eval_dynamic(oos_df, oos_filtered_actions, oos_notional, oos_leverage, oos_tp, oos_sl, fee=float(args.fee), slip=float(args.slip), max_hold=int(args.max_hold_bars))
        selector_summary = {
            "mode": "template_expert_governor",
            "target": "one binary CatBoost expert per complete risk template; governor selects max adjusted score",
            "exposure_penalty": float(args.governor_exposure_penalty),
            "selected_threshold_on_validation": {k: v for k, v in best_threshold.items() if k != "validation"},
            "validation_at_selected_threshold": best_threshold["validation"],
            "selected_template_counts": {
                "validation": {RISK_TEMPLATES[int(k)].name: int(v) for k, v in pd.Series(val_tpl_id[val_actions != 0]).value_counts().to_dict().items()},
                "oos": {RISK_TEMPLATES[int(k)].name: int(v) for k, v in pd.Series(oos_tpl_id[oos_actions != 0]).value_counts().to_dict().items()},
            },
        }
        model_extra = {"template_expert_models": risk_models}
    else:
        bucket_labels, quality_label, bucket_diag = _best_template_labels(
            train_df,
            train_actions,
            train_candidates,
            fee=float(args.fee),
            slip=float(args.slip),
            max_hold=int(args.max_hold_bars),
        )
        risk_models = {
            name: _fit_bucket_head(
                risk_x_train_all.iloc[train_candidates].reset_index(drop=True),
                y,
                risk_spec,
                int(args.seed + 101 + i * 17),
                task_type=str(args.task_type),
                devices=str(args.devices),
            )
            for i, (name, y) in enumerate(bucket_labels.items())
        }
        quality_model = _fit_bucket_head(
            risk_x_train_all.iloc[train_candidates].reset_index(drop=True),
            quality_label,
            risk_spec,
            int(args.seed + 191),
            task_type=str(args.task_type),
            devices=str(args.devices),
        )

        def predict_risk(x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            n_id = _predict_bucket(risk_models["notional"], x)
            l_id = _predict_bucket(risk_models["leverage"], x)
            tp_id = _predict_bucket(risk_models["tp"], x)
            sl_id = _predict_bucket(risk_models["sl"], x)
            return (
                np.asarray([values["notional"][int(i)] for i in n_id], dtype=np.float64),
                np.asarray([values["leverage"][int(i)] for i in l_id], dtype=np.float64),
                np.asarray([values["tp"][int(i)] for i in tp_id], dtype=np.float64),
                np.asarray([values["sl"][int(i)] for i in sl_id], dtype=np.float64),
            )

        val_notional, val_leverage, val_tp, val_sl = predict_risk(risk_x_val)
        oos_notional, oos_leverage, oos_tp, oos_sl = predict_risk(risk_x_oos)
        val_quality = np.asarray(quality_model.predict_proba(risk_x_val), dtype=np.float64)
        oos_quality = np.asarray(quality_model.predict_proba(risk_x_oos), dtype=np.float64)
        val_quality = val_quality[:, 1] if val_quality.ndim == 2 and val_quality.shape[1] > 1 else val_quality.reshape(-1)
        oos_quality = oos_quality[:, 1] if oos_quality.ndim == 2 and oos_quality.shape[1] > 1 else oos_quality.reshape(-1)
        best_threshold = None
        for threshold in [float(x.strip()) for x in str(args.quality_thresholds).split(",") if x.strip()]:
            filtered = val_actions.copy()
            filtered[(filtered != 0) & (val_quality < threshold)] = 0
            ev = _eval_dynamic(val_df, filtered, val_notional, val_leverage, val_tp, val_sl, fee=float(args.fee), slip=float(args.slip), max_hold=int(args.max_hold_bars))
            score = float(ev["cost1"]["pnl"]) + 0.35 * float(ev["cost2"]["pnl"]) + 0.10 * float(ev["cost3"]["pnl"]) - 0.20 * abs(float(ev["cost1"]["mdd"]))
            if int(ev["cost1"]["trades"]) < int(args.quality_min_trades):
                score -= 1000.0
            cand = {"threshold": threshold, "score": float(score), "validation": ev, "kept_signals": int(np.sum(filtered != 0))}
            if best_threshold is None or float(cand["score"]) > float(best_threshold["score"]):
                best_threshold = cand
        assert best_threshold is not None
        val_filtered_actions = val_actions.copy()
        val_filtered_actions[(val_filtered_actions != 0) & (val_quality < float(best_threshold["threshold"]))] = 0
        oos_filtered_actions = oos_actions.copy()
        oos_filtered_actions[(oos_filtered_actions != 0) & (oos_quality < float(best_threshold["threshold"]))] = 0
        val_eval = _eval_dynamic(val_df, val_filtered_actions, val_notional, val_leverage, val_tp, val_sl, fee=float(args.fee), slip=float(args.slip), max_hold=int(args.max_hold_bars))
        oos_eval = _eval_dynamic(oos_df, oos_filtered_actions, oos_notional, oos_leverage, oos_tp, oos_sl, fee=float(args.fee), slip=float(args.slip), max_hold=int(args.max_hold_bars))
        selector_summary = {
            "mode": "independent_bucket_heads",
            "target": "separate CatBoost heads predict notional/leverage/TP/SL plus quality filter",
            "train_positive_rate": float(np.mean(quality_label)),
            "selected_threshold_on_validation": {k: v for k, v in best_threshold.items() if k != "validation"},
            "validation_at_selected_threshold": best_threshold["validation"],
        }
        model_extra = {"risk_models": risk_models, "quality_model": quality_model}

    prefix = f"{args.variant}_{entry_spec.name}_{direction_spec.name}_{risk_spec.name}"
    joblib.dump(
        {
            "entry_model": entry_model,
            "direction_model": direction_model,
            **model_extra,
            "bucket_values": values,
            "risk_templates": [tpl.__dict__ for tpl in RISK_TEMPLATES],
            "projection": projection,
            "projection_meta": projection_meta,
            "feature_cols": feature_cols,
            "policy": {
                "entry_threshold": float(args.entry_threshold),
                "side_threshold": float(args.side_threshold),
                "margin_threshold": float(args.margin_threshold),
                "guardrail": str(args.guardrail),
            },
        },
        args.out_dir / f"{prefix}_model_bundle.joblib",
    )
    summary = {
        "model_id": MODEL_ID,
        "variant": str(args.variant),
        "risk_mode": str(args.risk_mode),
        "entry_spec": entry_spec.__dict__,
        "direction_spec": direction_spec.__dict__,
        "risk_spec": risk_spec.__dict__,
        "policy": {
            "entry_threshold": float(args.entry_threshold),
            "side_threshold": float(args.side_threshold),
            "margin_threshold": float(args.margin_threshold),
            "guardrail": str(args.guardrail),
            "max_hold_bars": int(args.max_hold_bars),
        },
        "train_candidates": int(len(train_candidates)),
        "val_candidates_before_quality": int(np.sum(val_actions != 0)),
        "oos_candidates_before_quality": int(np.sum(oos_actions != 0)),
        "val_candidates": int(np.sum(val_filtered_actions != 0)),
        "oos_candidates": int(np.sum(oos_filtered_actions != 0)),
        "bucket_label_diag": bucket_diag,
        "risk_selector": selector_summary,
        "validation": val_eval,
        "oos": oos_eval,
        "audit": {
            "preprocess_inputs": audit,
            "leak_audit": leak_audit,
            "risk_label_split": "risk bucket labels are generated from train candidates only; validation/OOS are never used to fit bucket heads",
            "risk_outputs": "governor mode trains one CatBoost expert per complete notional/leverage/TP/SL template; independent mode trains separate bucket heads",
            "accounting": "PnL, fee, slippage, and MDD use effective exposure = notional * leverage",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
