#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.alpha6_catboost_5head_policy_20260522 import _days, _fill_price  # noqa: E402
from scripts.research_alpha_model_synergy_oos_20260525 import _parent_for_features  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    _compact_costs,
    _metrics,
    _score,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


BASELINE = get_live_baseline()
LIVE_DIR = BASELINE.live_dir
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
PRIMARY_PARENT = BASELINE.primary_parent
PRIMARY_SUMMARY = BASELINE.primary_summary
COMBO_SUMMARY = BASELINE.combo_summary
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_meta_fallback_cash_router_20260526"
OLD_CLEAN_PREFIX = "clean_regime_2024_unsup_v4_"
MARKET_CONTEXT_COLS = (
    "obi",
    "taker_buy_ratio",
    "nif_whale",
    "eai",
    "oi_delta_pct",
    "funding_rate",
    "atr14_pct",
    "tp_sl_action_score",
    "ai_dir_edge",
    "teacher_long_edge",
    "teacher_short_edge",
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    parent: Path
    summary: Path
    family: str


def _candidate_specs() -> list[CandidateSpec]:
    base = ROOT / "tmp/causal_regen_20260516"
    return [
        CandidateSpec(
            "alpha43_no_legacy",
            base / "alpha4_3_legacy_regime_block_ablation_alpha43basis_20260517/no_legacy/parent.pkl",
            base / "alpha4_3_legacy_regime_block_ablation_alpha43basis_20260517/no_legacy/no_legacy_summary.json",
            "alpha4.3_no_regime",
        ),
        CandidateSpec(
            "alpha43_sticky_alpha61_derived",
            base / "alpha4_3_sticky_alpha61_derived_retrain_20260525/sticky_alpha61_derived/parent.pkl",
            base / "alpha4_3_sticky_alpha61_derived_retrain_20260525/sticky_alpha61_derived/sticky_alpha61_derived_summary.json",
            "alpha4.3_sticky_alpha61",
        ),
        CandidateSpec(
            "alpha5_2_factor_bridge",
            base / "alpha5_2_regime4_factor_bridge_no_teacher_no_deep_20260517/parent.pkl",
            base / "alpha5_2_regime4_factor_bridge_no_teacher_no_deep_20260517/alpha5_2_regime4_factor_bridge_no_teacher_no_deep_summary.json",
            "alpha5_factor_bridge",
        ),
        CandidateSpec(
            "alpha5_regime4_tp_sl",
            base / "alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517/parent.pkl",
            base / "alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517/alpha5_regime4_tp18_sl10_no_teacher_no_deep_summary.json",
            "alpha5_regime4",
        ),
    ]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _load_best_scale_runtime(summary_path: Path) -> alpha2.Alpha2Runtime | None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target = summary.get("best_by_selection")
    experiments = summary.get("experiments", [])
    if isinstance(target, dict):
        rt = target.get("selected_parent_scale_runtime")
        if rt:
            return alpha2.Alpha2Runtime(
                name=str(rt["name"]),
                confidence=float(rt["confidence"]),
                parent_notional_scale=float(rt["parent_notional_scale"]),
                max_notional=float(rt["max_notional"]),
            )
    for exp in experiments:
        if target is not None and not isinstance(target, dict) and exp.get("name") != target:
            continue
        rt = exp.get("selected_parent_scale_runtime")
        if rt:
            return alpha2.Alpha2Runtime(
                name=str(rt["name"]),
                confidence=float(rt["confidence"]),
                parent_notional_scale=float(rt["parent_notional_scale"]),
                max_notional=float(rt["max_notional"]),
            )
    return None


def _predict_scaled(parent: dict[str, Any], df: pd.DataFrame, rt: alpha2.Alpha2Runtime | None) -> pd.DataFrame:
    dec = predict_policy_frame(parent, df, close=_close(df)).reset_index(drop=True)
    if rt is not None:
        dec = alpha2._scale_parent_notional(dec, rt).reset_index(drop=True)
    return dec


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return (action != ACTION_CASH) & (side != 0)


def _copy_rows(target: pd.DataFrame, source: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    out = target.copy()
    for col in source.columns:
        out.loc[mask, col] = source.loc[mask, col].to_numpy()
    return out


def _combine_primary_fallback(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    primary = primary.reset_index(drop=True)
    fallback = fallback.reset_index(drop=True)
    return _copy_rows(primary, fallback, ~_active(primary) & _active(fallback))


def _empty_dec_like(template: pd.DataFrame) -> pd.DataFrame:
    dec = template.copy().reset_index(drop=True)
    dec["action"] = 0
    dec["side"] = 0
    dec["notional_exposure"] = 0.0
    dec["leverage"] = 1.0
    dec["position_fraction"] = 0.0
    dec["take_profit"] = 0.0
    dec["stop_loss"] = 0.0
    dec["max_hold_bars"] = 0
    dec["cooldown_bars"] = 0
    dec["quality_score"] = 0.0
    dec["confidence"] = 0.0
    return dec


def _trade_reward(frame: pd.DataFrame, dec: pd.DataFrame, i: int, *, fee: float, slip: float) -> float:
    row = dec.iloc[int(i)]
    action = int(pd.to_numeric(row["action"], errors="coerce"))
    side = int(pd.to_numeric(row["side"], errors="coerce"))
    if action == 0 or side == 0 or i + 1 >= len(frame):
        return 0.0
    notional = float(np.clip(pd.to_numeric(row["notional_exposure"], errors="coerce"), 0.01, 2.75))
    tp = float(max(pd.to_numeric(row["take_profit"], errors="coerce"), 1e-4))
    sl = float(max(pd.to_numeric(row["stop_loss"], errors="coerce"), 1e-4))
    max_hold = int(np.clip(pd.to_numeric(row["max_hold_bars"], errors="coerce"), 1, 96))
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)

    entry_i = min(i + 1, len(frame) - 1)
    entry = _fill_price(frame, entry_i, side, slip, entry=True)
    cash = 1.0
    cash -= cash * fee * notional
    end_i = min(entry_i + max_hold, len(frame) - 1)
    raw: float | None = None
    for j in range(entry_i + 1, end_i + 1):
        if side > 0:
            adverse = float(low[j] / max(entry, 1e-12) - 1.0)
            favorable = float(high[j] / max(entry, 1e-12) - 1.0)
        else:
            adverse = float(entry / max(high[j], 1e-12) - 1.0)
            favorable = float(entry / max(low[j], 1e-12) - 1.0)
        if adverse <= -sl:
            raw = -sl
            break
        if favorable >= tp:
            raw = tp
            break
    if raw is None:
        exit_px = _fill_price(frame, end_i, side, slip, entry=False)
        raw = (exit_px - entry) / max(entry, 1e-12) if side > 0 else (entry - exit_px) / max(entry, 1e-12)
    before = cash
    cash = cash * (1.0 + raw * notional)
    cash -= before * fee * notional
    return float(cash - 1.0)


def _label_cash_region(
    frame: pd.DataFrame,
    primary_dec: pd.DataFrame,
    candidate_decs: list[pd.DataFrame],
    *,
    min_edge: float,
    gap_min: float,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    primary_cash = ~_active(primary_dec)
    n = len(frame)
    y_action = np.zeros(n, dtype=np.int64)
    y_quality = np.zeros(n, dtype=np.float64)
    per_candidate_reward = np.zeros((n, len(candidate_decs)), dtype=np.float64)
    for i in range(max(0, n - 98)):
        if not primary_cash[i]:
            continue
        rewards: list[float] = []
        for j, dec in enumerate(candidate_decs):
            row = dec.iloc[i]
            if int(row["action"]) == 0 or int(row["side"]) == 0 or float(row["confidence"]) < float(min_confidence):
                rewards.append(0.0)
                continue
            r = _trade_reward(frame, dec, i, fee=0.0004, slip=0.00015)
            rewards.append(r)
            per_candidate_reward[i, j] = r
        if not rewards:
            continue
        best_idx = int(np.argmax(rewards))
        best = float(rewards[best_idx])
        second = float(sorted(rewards, reverse=True)[1]) if len(rewards) > 1 else 0.0
        if best > float(min_edge) and (best - second) > float(gap_min):
            y_action[i] = best_idx + 1
            y_quality[i] = best
    meta = {
        "primary_cash_rows": int(primary_cash.sum()),
        "labeled_trade_rows": int(np.sum(y_action != 0)),
        "labeled_trade_ratio": float(np.mean((y_action != 0)[primary_cash])) if np.any(primary_cash) else 0.0,
        "label_distribution": pd.Series(y_action[primary_cash]).value_counts().sort_index().to_dict(),
        "mean_best_reward": float(np.mean(y_quality[primary_cash])) if np.any(primary_cash) else 0.0,
    }
    return y_action, y_quality, meta


def _candidate_feature_block(name: str, dec: pd.DataFrame) -> pd.DataFrame:
    active = _active(dec).astype(np.float64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(np.int64)
    quality = pd.to_numeric(dec["quality_score"], errors="coerce").fillna(0.0)
    conf = pd.to_numeric(dec["confidence"], errors="coerce").fillna(0.0)
    horizon = pd.to_numeric(dec["max_hold_bars"], errors="coerce").fillna(0.0) / 96.0
    out = pd.DataFrame(
        {
            f"{name}_active": active,
            f"{name}_long": (active.astype(bool) & (side > 0)).astype(np.float64),
            f"{name}_short": (active.astype(bool) & (side < 0)).astype(np.float64),
            f"{name}_quality": quality,
            f"{name}_confidence": conf,
            f"{name}_notional": pd.to_numeric(dec["notional_exposure"], errors="coerce").fillna(0.0),
            f"{name}_tp": pd.to_numeric(dec["take_profit"], errors="coerce").fillna(0.0),
            f"{name}_sl": pd.to_numeric(dec["stop_loss"], errors="coerce").fillna(0.0),
            f"{name}_horizon_frac": horizon,
            f"{name}_edge": quality * np.where(side > 0, 1.0, np.where(side < 0, -1.0, 0.0)),
        }
    )
    out[f"{name}_quality_roll5_mean"] = quality.rolling(5, min_periods=1).mean()
    out[f"{name}_quality_roll5_std"] = quality.rolling(5, min_periods=1).std().fillna(0.0)
    return out


def _entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-12, 1.0)
    return -(probs * np.log(probs)).sum(axis=1)


def _build_meta_features(
    frame: pd.DataFrame,
    primary_dec: pd.DataFrame,
    candidate_specs: list[CandidateSpec],
    candidate_decs: list[pd.DataFrame],
) -> pd.DataFrame:
    blocks = []
    for spec, dec in zip(candidate_specs, candidate_decs):
        blocks.append(_candidate_feature_block(spec.name, dec))
    out = pd.concat(blocks, axis=1)
    active_cols = [f"{spec.name}_active" for spec in candidate_specs]
    long_cols = [f"{spec.name}_long" for spec in candidate_specs]
    short_cols = [f"{spec.name}_short" for spec in candidate_specs]
    q_cols = [f"{spec.name}_quality" for spec in candidate_specs]
    c_cols = [f"{spec.name}_confidence" for spec in candidate_specs]
    h_cols = [f"{spec.name}_horizon_frac" for spec in candidate_specs]
    edge_cols = [f"{spec.name}_edge" for spec in candidate_specs]
    q = out[q_cols].to_numpy(dtype=np.float64)
    conf = out[c_cols].to_numpy(dtype=np.float64)
    active = out[active_cols].to_numpy(dtype=np.float64)
    out["active_count"] = active.sum(axis=1)
    out["long_count"] = out[long_cols].sum(axis=1)
    out["short_count"] = out[short_cols].sum(axis=1)
    out["quality_top"] = q.max(axis=1)
    out["quality_mean"] = q.mean(axis=1)
    out["quality_std"] = q.std(axis=1)
    out["confidence_top"] = conf.max(axis=1)
    out["confidence_mean"] = conf.mean(axis=1)
    out["horizon_mean"] = out[h_cols].mean(axis=1)
    out["horizon_std"] = out[h_cols].std(axis=1)
    out["weighted_long"] = (out[long_cols].to_numpy(dtype=np.float64) * q).sum(axis=1)
    out["weighted_short"] = (out[short_cols].to_numpy(dtype=np.float64) * q).sum(axis=1)
    out["edge_mean"] = out[edge_cols].mean(axis=1)
    out["edge_std"] = out[edge_cols].std(axis=1)
    out["disagreement_entropy"] = _entropy(
        np.column_stack(
            [
                np.clip(1.0 - (out["long_count"] + out["short_count"]) / max(len(candidate_specs), 1), 0.0, 1.0),
                out["long_count"] / max(len(candidate_specs), 1),
                out["short_count"] / max(len(candidate_specs), 1),
            ]
        )
    )
    q_sort = np.sort(q, axis=1)
    out["quality_gap_top2"] = q_sort[:, -1] - q_sort[:, -2] if q.shape[1] >= 2 else q_sort[:, -1]
    out["primary_quality"] = pd.to_numeric(primary_dec["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["primary_confidence"] = pd.to_numeric(primary_dec["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["primary_active"] = _active(primary_dec).astype(np.float64)
    for col in MARKET_CONTEXT_COLS:
        if col in frame.columns:
            out[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    regime_cols = [c for c in frame.columns if c.startswith("clean_regime4_2024_unsup_v1_") or c.startswith("regime4_pred_")]
    for col in regime_cols:
        out[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def _fit_meta_models(
    x_train: pd.DataFrame,
    y_action: np.ndarray,
    y_quality: np.ndarray,
    *,
    seed: int,
) -> tuple[Any, Any]:
    action_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=int(np.max(y_action)) + 1,
        class_weight="balanced",
        n_estimators=220,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=120,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=4.0,
        reg_lambda=16.0,
        random_state=seed,
        verbosity=-1,
    )
    quality_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=180,
        learning_rate=0.03,
        max_depth=2,
        num_leaves=5,
        min_child_samples=160,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=6.0,
        reg_lambda=20.0,
        random_state=seed + 17,
        verbosity=-1,
    )
    action_model.fit(x_train, y_action)
    quality_model.fit(x_train, y_quality)
    return action_model, quality_model


def _predict_action_proba(model: Any, x: pd.DataFrame, class_count: int) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x), dtype=np.float64)
    out = np.zeros((len(x), class_count), dtype=np.float64)
    for j, cls in enumerate(np.asarray(model.classes_, dtype=np.int64)):
        if 0 <= int(cls) < class_count:
            out[:, int(cls)] = proba[:, j]
    return out


def _build_meta_fallback_decisions(
    template: pd.DataFrame,
    primary_dec: pd.DataFrame,
    candidate_decs: list[pd.DataFrame],
    pred_class: np.ndarray,
    pred_quality: np.ndarray,
    pred_proba: np.ndarray,
    *,
    prob_min: float,
    quality_min: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = _empty_dec_like(template)
    counts = {"cash": 0}
    primary_cash = ~_active(primary_dec)
    for i in range(len(out)):
        if not primary_cash[i]:
            counts["cash"] += 1
            continue
        cls = int(pred_class[i])
        if cls <= 0 or cls > len(candidate_decs):
            counts["cash"] += 1
            continue
        cls_prob = float(pred_proba[i, cls])
        if cls_prob < float(prob_min) or float(pred_quality[i]) < float(quality_min):
            counts["cash"] += 1
            continue
        chosen = candidate_decs[cls - 1]
        if not _active(chosen.iloc[[i]]).item():
            counts["cash"] += 1
            continue
        for col in out.columns:
            out.iat[i, out.columns.get_loc(col)] = chosen.iat[i, chosen.columns.get_loc(col)]
        name = f"candidate_{cls}"
        counts[name] = counts.get(name, 0) + 1
    return out, counts


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a primary-cash-only meta fallback router for Alpha7.")
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--eval-csv", type=Path, default=EVAL_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=52626)
    ap.add_argument("--label-min-edge", type=float, default=0.00035)
    ap.add_argument("--label-gap-min", type=float, default=0.00010)
    ap.add_argument("--label-min-confidence", type=float, default=0.56)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    cutoff = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < cutoff].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= cutoff].reset_index(drop=True)

    primary_parent = joblib.load(PRIMARY_PARENT)
    primary_rt = _load_best_scale_runtime(PRIMARY_SUMMARY)
    primary_train = _predict_scaled(primary_parent, train_df, primary_rt)
    primary_val = _predict_scaled(primary_parent, val_df, primary_rt)
    primary_eval = _predict_scaled(primary_parent, eval_df, primary_rt)

    candidate_specs = []
    train_candidate_decs = []
    val_candidate_decs = []
    eval_candidate_decs = []
    for spec in _candidate_specs():
        if not spec.parent.exists():
            continue
        parent = joblib.load(spec.parent)
        feature_cols = list(parent.get("feature_cols", []))
        if any(str(c).startswith(OLD_CLEAN_PREFIX) for c in feature_cols):
            continue
        rt = _load_best_scale_runtime(spec.summary)
        candidate_specs.append(spec)
        train_candidate_decs.append(_predict_scaled(parent, train_df, rt))
        val_candidate_decs.append(_predict_scaled(parent, val_df, rt))
        eval_candidate_decs.append(_predict_scaled(parent, eval_df, rt))
    if not candidate_specs:
        raise RuntimeError("no valid fallback candidates")

    x_train = _build_meta_features(train_df, primary_train, candidate_specs, train_candidate_decs)
    x_val = _build_meta_features(val_df, primary_val, candidate_specs, val_candidate_decs)
    x_eval = _build_meta_features(eval_df, primary_eval, candidate_specs, eval_candidate_decs)

    y_action_train, y_quality_train, label_meta_train = _label_cash_region(
        train_df,
        primary_train,
        train_candidate_decs,
        min_edge=float(args.label_min_edge),
        gap_min=float(args.label_gap_min),
        min_confidence=float(args.label_min_confidence),
    )
    y_action_val, y_quality_val, label_meta_val = _label_cash_region(
        val_df,
        primary_val,
        val_candidate_decs,
        min_edge=float(args.label_min_edge),
        gap_min=float(args.label_gap_min),
        min_confidence=float(args.label_min_confidence),
    )
    train_cash_mask = ~_active(primary_train)
    if int((y_action_train[train_cash_mask] != 0).sum()) < 100:
        raise RuntimeError("too few labeled trade rows for meta-fallback training")

    action_model, quality_model = _fit_meta_models(
        x_train.loc[train_cash_mask].reset_index(drop=True),
        y_action_train[train_cash_mask],
        y_quality_train[train_cash_mask],
        seed=int(args.seed),
    )

    class_count = len(candidate_specs) + 1
    val_proba = _predict_action_proba(action_model, x_val, class_count)
    eval_proba = _predict_action_proba(action_model, x_eval, class_count)
    val_class = np.argmax(val_proba, axis=1).astype(np.int64)
    eval_class = np.argmax(eval_proba, axis=1).astype(np.int64)
    val_quality_pred = np.asarray(quality_model.predict(x_val), dtype=np.float64)
    eval_quality_pred = np.asarray(quality_model.predict(x_eval), dtype=np.float64)

    ref_parent = _parent_for_features(list(joblib.load(v31.DEFAULT_PARENT)["feature_cols"]))
    fee = float(joblib.load(v31.DEFAULT_PARENT)["config"]["fee"])
    slip = float(joblib.load(v31.DEFAULT_PARENT)["config"]["slip"])
    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    baseline_combo = json.loads(COMBO_SUMMARY.read_text(encoding="utf-8"))
    current_fallback_eval = eval_candidate_decs[0]
    baseline_metrics = _compact_costs(
        _metrics(
            eval_df,
            parent_for_features=ref_parent,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=_combine_primary_fallback(primary_eval, current_fallback_eval),
            fee=fee,
            slip=slip,
        )
    )

    active_quality = val_quality_pred[(~_active(primary_val)) & (val_class != 0)]
    if len(active_quality) == 0:
        raise RuntimeError("meta model produced no active validation candidates")
    quality_grid = sorted(set(float(x) for x in np.quantile(active_quality, [0.25, 0.40, 0.55, 0.70, 0.85])))
    prob_grid = [0.35, 0.45, 0.55, 0.65]
    grid_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for prob_min in prob_grid:
        for quality_min in quality_grid:
            val_fb_dec, val_counts = _build_meta_fallback_decisions(
                val_candidate_decs[0],
                primary_val,
                val_candidate_decs,
                val_class,
                val_quality_pred,
                val_proba,
                prob_min=prob_min,
                quality_min=quality_min,
            )
            eval_fb_dec, eval_counts = _build_meta_fallback_decisions(
                eval_candidate_decs[0],
                primary_eval,
                eval_candidate_decs,
                eval_class,
                eval_quality_pred,
                eval_proba,
                prob_min=prob_min,
                quality_min=quality_min,
            )
            val_metrics = _compact_costs(
                _metrics(
                    val_df,
                    parent_for_features=ref_parent,
                    runner=noop_runner,
                    runner_cfg=noop_cfg,
                    dec=_combine_primary_fallback(primary_val, val_fb_dec),
                    fee=fee,
                    slip=slip,
                )
            )
            eval_metrics = _compact_costs(
                _metrics(
                    eval_df,
                    parent_for_features=ref_parent,
                    runner=noop_runner,
                    runner_cfg=noop_cfg,
                    dec=_combine_primary_fallback(primary_eval, eval_fb_dec),
                    fee=fee,
                    slip=slip,
                )
            )
            row = {
                "prob_min": float(prob_min),
                "quality_min": float(quality_min),
                "selection_score": float(_score(val_metrics)),
                "val_cost3_pnl": float(val_metrics["cost3"]["pnl"]),
                "val_cost3_mdd": float(val_metrics["cost3"]["mdd"]),
                "val_cost3_trades": int(val_metrics["cost3"]["trades"]),
                "oos_cost3_pnl": float(eval_metrics["cost3"]["pnl"]),
                "oos_cost3_mdd": float(eval_metrics["cost3"]["mdd"]),
                "oos_cost3_trades": int(eval_metrics["cost3"]["trades"]),
                "oos_cost3_wr": float(eval_metrics["cost3"]["wr"]),
                "delta_vs_current_fallback": float(eval_metrics["cost3"]["pnl"]) - float(baseline_metrics["cost3"]["pnl"]),
                "val_counts": val_counts,
                "eval_counts": eval_counts,
            }
            grid_rows.append(row)
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row
    assert best is not None

    best_val_fb_dec, best_val_counts = _build_meta_fallback_decisions(
        val_candidate_decs[0],
        primary_val,
        val_candidate_decs,
        val_class,
        val_quality_pred,
        val_proba,
        prob_min=float(best["prob_min"]),
        quality_min=float(best["quality_min"]),
    )
    best_eval_fb_dec, best_eval_counts = _build_meta_fallback_decisions(
        eval_candidate_decs[0],
        primary_eval,
        eval_candidate_decs,
        eval_class,
        eval_quality_pred,
        eval_proba,
        prob_min=float(best["prob_min"]),
        quality_min=float(best["quality_min"]),
    )
    best_val_metrics = _compact_costs(
        _metrics(
            val_df,
            parent_for_features=ref_parent,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=_combine_primary_fallback(primary_val, best_val_fb_dec),
            fee=fee,
            slip=slip,
        )
    )
    best_eval_metrics = _compact_costs(
        _metrics(
            eval_df,
            parent_for_features=ref_parent,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=_combine_primary_fallback(primary_eval, best_eval_fb_dec),
            fee=fee,
            slip=slip,
        )
    )

    artifact = {
        "feature_cols": list(x_train.columns),
        "candidate_names": [spec.name for spec in candidate_specs],
        "action_model": action_model,
        "quality_model": quality_model,
        "prob_min": float(best["prob_min"]),
        "quality_min": float(best["quality_min"]),
        "label_min_edge": float(args.label_min_edge),
        "label_gap_min": float(args.label_gap_min),
        "label_min_confidence": float(args.label_min_confidence),
    }
    artifact_path = args.out_dir / "meta_fallback_router.joblib"
    joblib.dump(artifact, artifact_path)
    grid_path = args.out_dir / "grid.csv"
    pd.DataFrame(grid_rows).sort_values(["selection_score", "oos_cost3_pnl"], ascending=[False, False]).to_csv(grid_path, index=False)

    report = {
        "model_id": "alpha7_meta_fallback_cash_router_20260526",
        "design": "Primary Alpha7 stays fixed. On rows where the primary is CASH, a shallow meta router sees multiple fallback candidate outputs plus market/regime context and chooses CASH or one fallback candidate. Execution and accounting remain the existing Alpha7 noop runner contract.",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "candidate_names": [spec.name for spec in candidate_specs],
        "baseline": {
            "combo_selected_metrics": baseline_combo.get("selected_metrics"),
            "current_fallback_combo_metrics": baseline_metrics,
        },
        "labeling": {
            "train": label_meta_train,
            "validation": label_meta_val,
            "label_min_edge": float(args.label_min_edge),
            "label_gap_min": float(args.label_gap_min),
            "label_min_confidence": float(args.label_min_confidence),
        },
        "feature_contract": {
            "feature_count": int(len(x_train.columns)),
            "feature_cols": list(x_train.columns),
        },
        "primary_runtime": asdict(primary_rt) if primary_rt is not None else None,
        "best_by_selection": {
            **best,
            "val_metrics": best_val_metrics,
            "oos_metrics": best_eval_metrics,
            "best_val_counts": best_val_counts,
            "best_eval_counts": best_eval_counts,
        },
        "artifacts": {
            "meta_router": str(artifact_path),
            "grid": str(grid_path),
        },
        "audit": {
            "selection_uses_2026": False,
            "selection_window": "2025-10-01..2025-12-31",
            "oos_window": "2026 fixed OOS",
            "old_clean_regime_candidates_filtered": True,
        },
    }
    report_path = args.out_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "best_prob_min": best["prob_min"],
                "best_quality_min": best["quality_min"],
                "oos_cost3_pnl": best_eval_metrics["cost3"]["pnl"],
                "oos_cost3_mdd": best_eval_metrics["cost3"]["mdd"],
                "oos_cost3_trades": best_eval_metrics["cost3"]["trades"],
                "delta_vs_current_fallback": float(best_eval_metrics["cost3"]["pnl"]) - float(baseline_metrics["cost3"]["pnl"]),
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
