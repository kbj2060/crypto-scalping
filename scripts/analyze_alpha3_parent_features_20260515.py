#!/usr/bin/env python3
from __future__ import annotations

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

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    FullyLearnedGovernorConfig,
    build_training_set,
    prepare_features,
)
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402


MODEL_ID = "alpha3_parent_feature_analysis_20260515"
PARENT_MODEL = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl"
TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_parent_feature_analysis_20260515.json"
RANKING_OUT = ROOT / "data/ensemble/reports/alpha3_parent_feature_analysis_20260515_ranking.csv"
REDUNDANT_OUT = ROOT / "data/ensemble/reports/alpha3_parent_feature_analysis_20260515_redundant_pairs.csv"
SETS_OUT = ROOT / "data/ensemble/reports/alpha3_parent_feature_analysis_20260515_feature_sets.json"


HEAD_WEIGHTS = {
    "action": 3.0,
    "quality": 2.0,
    "notional": 0.85,
    "leverage": 0.65,
    "take_profit": 0.75,
    "stop_loss": 0.75,
    "max_hold": 0.35,
    "cooldown": 0.20,
}


def _group_feature(name: str) -> str:
    if name == "side_hint":
        return "control"
    if name.startswith("clean_regime_2024_unsup_v4_"):
        return "clean_regime"
    if name.startswith("m7_"):
        return "m7"
    if name.startswith("ai_") or name.startswith("patchtst") or name.startswith("pred_") or name.startswith("conf_"):
        return "ai"
    if name.startswith("tide_") or name.startswith("timesnet_") or name.startswith("dlinear_"):
        return "ai"
    if name in {"net_taker_ratio", "taker_acceleration", "ofi_acceleration", "trade_intensity", "big_trade_ratio", "whale_retail_ratio", "smart_money_flow", "whale_conviction"}:
        return "micro_flow"
    if name in {"volatility_z", "garch_vol_z", "rogers_satchell_vol", "amihud_illiquidity_z", "liquidity_vacuum", "execution_quality"}:
        return "vol_liquidity"
    if name.startswith("mom_") or name.startswith("abs_mom_") or name in {"log_return", "mtf_trend_1h", "mtf_trend_4h", "rsi", "squeeze_power", "breakout_strength"}:
        return "price_momentum"
    if name.startswith("funding") or name in {"long_squeeze_risk", "crowding_pressure"}:
        return "derivatives"
    if name.startswith("jump") or name.startswith("evt_"):
        return "tail_event"
    return "other"


def _target_side(action: np.ndarray) -> np.ndarray:
    return np.where(action == ACTION_LONG, 1.0, np.where(action == ACTION_SHORT, -1.0, 0.0)).astype(np.float64)


def _classifier_score(model: Any, x: pd.DataFrame, y: np.ndarray) -> float:
    if len(y) == 0 or np.unique(y).size < 2:
        return 0.0
    pred = model.predict(x)
    return float(np.mean(pred == y))


def _regressor_score(model: Any, x: pd.DataFrame, y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    pred = np.asarray(model.predict(x), dtype=np.float64)
    return float(-np.mean(np.abs(pred - y)))


def _permutation_importance(
    *,
    model: Any,
    x: pd.DataFrame,
    y: np.ndarray,
    kind: str,
    cols: list[str],
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    base_score = _classifier_score(model, x, y) if kind == "classifier" else _regressor_score(model, x, y)
    out: dict[str, float] = {}
    if len(x) == 0:
        return {c: 0.0 for c in cols}
    for col in cols:
        xx = x.copy()
        vals = xx[col].to_numpy(copy=True)
        rng.shuffle(vals)
        xx[col] = vals
        score = _classifier_score(model, xx, y) if kind == "classifier" else _regressor_score(model, xx, y)
        out[col] = float(max(0.0, base_score - score))
    return out


def _minmax(values: dict[str, float]) -> dict[str, float]:
    arr = np.asarray(list(values.values()), dtype=np.float64)
    if arr.size == 0 or float(np.nanmax(arr) - np.nanmin(arr)) <= 1e-12:
        return {k: 0.0 for k in values}
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    return {k: float((v - lo) / (hi - lo)) for k, v in values.items()}


def _source_missing(train: pd.DataFrame, eval_df: pd.DataFrame, col: str) -> tuple[bool, bool]:
    if col == "side_hint" or col.startswith("mom_") or col.startswith("abs_mom_"):
        return False, False
    return col not in train.columns, col not in eval_df.columns


def _redundant_pairs(x: pd.DataFrame, cols: list[str], ranking: pd.DataFrame, threshold: float = 0.97) -> pd.DataFrame:
    sample = x[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = sample.corr(method="spearman").abs()
    score = dict(zip(ranking["feature"], ranking["aggregate_score"]))
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            c = float(corr.loc[a, b])
            if c >= threshold:
                drop = b if score.get(a, 0.0) >= score.get(b, 0.0) else a
                keep = a if drop == b else b
                rows.append({"feature_a": a, "feature_b": b, "abs_spearman": c, "suggest_keep": keep, "suggest_drop": drop})
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False) if rows else pd.DataFrame(columns=["feature_a", "feature_b", "abs_spearman", "suggest_keep", "suggest_drop"])


def _feature_sets(ranking: pd.DataFrame, redundant: pd.DataFrame) -> dict[str, Any]:
    ordered = ranking["feature"].tolist()
    mandatory = ["side_hint"]
    source_missing = set(ranking.loc[ranking["source_missing_any"], "feature"].tolist())
    low_signal = set(ranking.loc[ranking["low_variance_or_zero"], "feature"].tolist())
    removable = source_missing | low_signal
    base_ordered = [f for f in ordered if f not in removable or f in mandatory]
    drops_corr = set(redundant["suggest_drop"].tolist()) if len(redundant) else set()
    pruned_corr = [f for f in base_ordered if f not in drops_corr or f in mandatory]

    def topn(n: int) -> list[str]:
        out = []
        for f in base_ordered:
            if f not in out:
                out.append(f)
            if len(out) >= n:
                break
        for f in mandatory:
            if f not in out:
                out.insert(0, f)
        return out

    return {
        "top32_raw_parent": topn(32),
        "top48_raw_parent": topn(48),
        "top64_raw_parent": topn(64),
        "corr_pruned_raw_parent": pruned_corr,
        "drop_reasons": {
            "source_missing_or_zero_fill": sorted(source_missing),
            "low_variance_or_zero": sorted(low_signal),
            "high_corr_suggest_drop": sorted(drops_corr),
        },
        "chronos_kairos_pls_next_candidates": [
            {
                "name": "top32_raw_plus_chronos_kairos_pls_4x4",
                "raw_features": topn(32),
                "macro_encoder": "Chronos-2 cached embeddings -> train-only PLSRegression n_components=4",
                "micro_encoder": "Kairos_23m cached embeddings -> train-only PLSRegression n_components=4",
            },
            {
                "name": "top48_raw_plus_chronos_kairos_pls_8x8",
                "raw_features": topn(48),
                "macro_encoder": "Chronos-2 cached embeddings -> train-only PLSRegression n_components=8",
                "micro_encoder": "Kairos_23m cached embeddings -> train-only PLSRegression n_components=8",
            },
        ],
    }


def main() -> int:
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{MODEL_ID}] loading parent/data", flush=True)
    parent = joblib.load(PARENT_MODEL)
    feature_cols = list(parent["feature_cols"])
    cfg = FullyLearnedGovernorConfig(**dict(parent["config"]))
    train_all = _read(TRAIN_CSV)
    eval_df = _read(EVAL_CSV)
    split_ts = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)

    print(f"[{MODEL_ID}] preparing feature frames", flush=True)
    x_train_full = prepare_features(train_all, side_hint=0, close=_close(train_all), feature_cols=feature_cols)
    x_val_full = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    x_eval_full = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)

    print(f"[{MODEL_ID}] building 2025Q4 labels for permutation analysis", flush=True)
    x_val_lab, y_val, label_meta = build_training_set(val_df, cfg=cfg, stride_bars=12, batch_size=512, feature_cols=feature_cols)
    action = np.asarray(y_val["action"], dtype=np.int64)
    trade_mask = action != ACTION_CASH
    x_trade = x_val_lab.loc[trade_mask].copy()
    if "side_hint" in x_trade.columns:
        x_trade["side_hint"] = _target_side(action[trade_mask])

    head_importance: dict[str, dict[str, float]] = {}
    print(f"[{MODEL_ID}] action/quality permutation", flush=True)
    head_importance["action"] = _permutation_importance(
        model=parent["action_model"], x=x_val_lab, y=action, kind="classifier", cols=feature_cols, seed=101
    )
    head_importance["quality"] = _permutation_importance(
        model=parent["quality_model"], x=x_val_lab, y=np.asarray(y_val["quality"], dtype=np.float64), kind="regressor", cols=feature_cols, seed=102
    )
    for offset, head in enumerate(("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown"), start=1):
        model_key = f"{head}_model"
        if model_key not in parent or not len(x_trade):
            head_importance[head] = {c: 0.0 for c in feature_cols}
            continue
        print(f"[{MODEL_ID}] {head} permutation", flush=True)
        head_importance[head] = _permutation_importance(
            model=parent[model_key],
            x=x_trade,
            y=np.asarray(y_val[head], dtype=np.int64)[trade_mask],
            kind="classifier",
            cols=feature_cols,
            seed=200 + offset,
        )

    normalized = {h: _minmax(v) for h, v in head_importance.items()}
    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        raw = {f"{head}_importance": float(head_importance[head].get(col, 0.0)) for head in HEAD_WEIGHTS}
        norm = {f"{head}_importance_norm": float(normalized[head].get(col, 0.0)) for head in HEAD_WEIGHTS}
        aggregate = sum(float(HEAD_WEIGHTS[head]) * float(normalized[head].get(col, 0.0)) for head in HEAD_WEIGHTS)
        tr_missing, ev_missing = _source_missing(train_all, eval_df, col)
        tr_series = x_train_full[col].replace([np.inf, -np.inf], np.nan)
        ev_series = x_eval_full[col].replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "feature": col,
                "group": _group_feature(col),
                "aggregate_score": float(aggregate),
                "train_source_missing": bool(tr_missing),
                "eval_source_missing": bool(ev_missing),
                "source_missing_any": bool(tr_missing or ev_missing),
                "train_nan_rate_prepared": float(tr_series.isna().mean()),
                "eval_nan_rate_prepared": float(ev_series.isna().mean()),
                "train_zero_rate_prepared": float((tr_series.fillna(0.0) == 0.0).mean()),
                "eval_zero_rate_prepared": float((ev_series.fillna(0.0) == 0.0).mean()),
                "train_std_prepared": float(tr_series.fillna(0.0).std()),
                "low_variance_or_zero": bool(float(tr_series.fillna(0.0).std()) <= 1e-12),
                **raw,
                **norm,
            }
        )
    ranking = pd.DataFrame(rows).sort_values("aggregate_score", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(RANKING_OUT, index=False)

    print(f"[{MODEL_ID}] correlation pruning", flush=True)
    corr_sample = x_train_full.iloc[:: max(1, len(x_train_full) // 8000)].reset_index(drop=True)
    redundant = _redundant_pairs(corr_sample, feature_cols, ranking, threshold=0.97)
    redundant.to_csv(REDUNDANT_OUT, index=False)
    feature_sets = _feature_sets(ranking, redundant)
    SETS_OUT.write_text(json.dumps(feature_sets, indent=2, ensure_ascii=False), encoding="utf-8")

    group_summary = (
        ranking.groupby("group", as_index=False)
        .agg(feature_count=("feature", "count"), score_sum=("aggregate_score", "sum"), score_mean=("aggregate_score", "mean"))
        .sort_values("score_sum", ascending=False)
        .to_dict(orient="records")
    )
    report = {
        "model_id": MODEL_ID,
        "scope": "Alpha3 parent feature analysis for hf_v13_clean_regime_margin110_20260511. This does not modify live Alpha3 artifacts.",
        "parent_model": str(PARENT_MODEL),
        "train_csv": str(TRAIN_CSV),
        "eval_csv": str(EVAL_CSV),
        "feature_count": len(feature_cols),
        "label_meta": label_meta,
        "top20_features": ranking.head(20).to_dict(orient="records"),
        "group_summary": group_summary,
        "source_missing_features": ranking.loc[ranking["source_missing_any"], ["feature", "group", "aggregate_score", "train_source_missing", "eval_source_missing"]].to_dict(orient="records"),
        "low_variance_features": ranking.loc[ranking["low_variance_or_zero"], ["feature", "group", "aggregate_score"]].to_dict(orient="records"),
        "redundant_pair_count_abs_spearman_ge_0_97": int(len(redundant)),
        "feature_sets": feature_sets,
        "artifacts": {
            "report": str(REPORT_OUT),
            "ranking_csv": str(RANKING_OUT),
            "redundant_pairs_csv": str(REDUNDANT_OUT),
            "feature_sets_json": str(SETS_OUT),
        },
        "next_experiment_plan": [
            "Train Alpha3 parent replacements using top32/top48/corr_pruned raw features while preserving original V21.2 runner feature frame.",
            "Then test Chronos/Kairos cached embeddings as train-only PLS factors added to top raw features, not as a full raw replacement.",
            "Backtest all variants with canonical Alpha3 corrected next_open_limit_touch0_fee20 execution contract.",
        ],
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "ranking": str(RANKING_OUT), "feature_sets": str(SETS_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
