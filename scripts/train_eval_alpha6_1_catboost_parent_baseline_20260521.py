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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import prepare_features  # noqa: E402
from scripts.train_eval_alpha5_13_hgb_single_20260518 import _direction_metrics  # noqa: E402
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days, _fill_price, _json_default, _read  # noqa: E402


MODEL_ID = "alpha6_1_catboost_parent_baseline_20260521"
DEFAULT_SPEC_DIR = ROOT / "tmp/causal_regen_20260516/dsac_feature_variant_specs_regime_fixed_20260521"
DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_24_entry_rebalanced_labels_20260519"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha6_1_catboost_parent_baseline_20260521"
DEFAULT_VARIANTS = "stable48_global_pca32,current_tail111"

FORBIDDEN_FEATURE_COLS = {
    "label_action",
    "label_train_keep",
    "entry_label",
    "direction_label",
    "entry_train_keep",
    "direction_train_keep",
    "entry_sample_weight",
    "direction_sample_weight",
    "label_tp_pct",
    "label_sl_pct",
    "label_fixed_tp05_action",
    "meta_tp_first",
    "meta_tp_ge_005",
}
FORBIDDEN_SUBSTRINGS = (
    "future",
    "label_",
    "target",
)


@dataclass(frozen=True)
class CatSpec:
    name: str
    iterations: int
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    random_strength: float
    bagging_temperature: float


def _cat_specs() -> list[CatSpec]:
    return [
        CatSpec("base", 400, 6, 0.040, 3.0, 1.0, 0.0),
        CatSpec("regularized", 320, 5, 0.035, 6.0, 2.0, 0.5),
        CatSpec("deeper", 520, 8, 0.028, 3.0, 0.5, 0.0),
    ]


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    out = np.ones(len(y), dtype=np.float64)
    classes, counts = np.unique(y, return_counts=True)
    total = max(float(len(y)), 1.0)
    for cls, count in zip(classes, counts):
        out[y == int(cls)] = total / (float(len(classes)) * max(float(count), 1.0))
    return out


def _x(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return prepare_features(frame, side_hint=0, close=_close(frame), feature_cols=cols)


def _read_spec(spec_dir: Path, variant: str) -> dict[str, Any]:
    path = spec_dir / f"{variant}.json"
    if not path.exists():
        raise FileNotFoundError(f"feature spec not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_spec_path"] = str(path)
    return payload


def _sanitize_feature_cols(frame: pd.DataFrame, raw_cols: list[str]) -> tuple[list[str], dict[str, Any]]:
    usable: list[str] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for col in raw_cols:
        c = str(col).strip()
        if not c or c in seen:
            continue
        seen.add(c)
        if c not in frame.columns:
            rejected.append({"feature": c, "reason": "missing_from_labels_frame"})
            continue
        lowered = c.lower()
        if c in FORBIDDEN_FEATURE_COLS:
            rejected.append({"feature": c, "reason": "explicit_forbidden"})
            continue
        if any(token in lowered for token in FORBIDDEN_SUBSTRINGS):
            rejected.append({"feature": c, "reason": "forbidden_pattern"})
            continue
        usable.append(c)
    audit = {
        "requested_feature_count": int(len(raw_cols)),
        "usable_feature_count": int(len(usable)),
        "rejected_count": int(len(rejected)),
        "rejected_features": rejected,
    }
    return usable, audit


def _build_projection(
    train_frame: pd.DataFrame,
    eval_frames: list[pd.DataFrame],
    feature_cols: list[str],
    *,
    enable_pca: bool,
    pca_components: int,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, Any], Pipeline | None]:
    x_train = _x(train_frame, feature_cols)
    x_eval = [_x(frame, feature_cols) for frame in eval_frames]
    if not enable_pca or pca_components <= 0:
        meta = {
            "base_feature_count": int(len(feature_cols)),
            "final_feature_count": int(x_train.shape[1]),
            "pca_enable": False,
            "pca_components": 0,
            "pca_output_cols": [],
        }
        return x_train, x_eval, meta, None

    n_train = int(len(x_train))
    n_base = int(x_train.shape[1])
    n_comp = int(max(1, min(pca_components, n_train, n_base)))
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp, svd_solver="full", random_state=0)),
        ]
    )
    pipe.fit(x_train)
    pca_cols = [f"global_pca_{i+1:02d}" for i in range(n_comp)]
    pca_train = pd.DataFrame(pipe.transform(x_train), columns=pca_cols, index=x_train.index)
    out_train = pd.concat([x_train.reset_index(drop=True), pca_train.reset_index(drop=True)], axis=1)
    out_eval: list[pd.DataFrame] = []
    for frame in x_eval:
        pca_eval = pd.DataFrame(pipe.transform(frame), columns=pca_cols, index=frame.index)
        out_eval.append(pd.concat([frame.reset_index(drop=True), pca_eval.reset_index(drop=True)], axis=1))
    meta = {
        "base_feature_count": int(len(feature_cols)),
        "final_feature_count": int(out_train.shape[1]),
        "pca_enable": True,
        "pca_components": int(n_comp),
        "pca_output_cols": pca_cols,
        "explained_variance_ratio_sum": float(np.sum(pipe.named_steps["pca"].explained_variance_ratio_)),
    }
    return out_train, out_eval, meta, pipe


def _fit_cat(
    x: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    spec: CatSpec,
    seed: int,
    *,
    task_type: str,
    devices: str,
) -> Any:
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
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
    model.fit(x, y, sample_weight=w)
    return model


def _binary_proba(model: Any, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    if raw.ndim == 2 and raw.shape[1] >= 2:
        return raw[:, 1]
    return raw.reshape(-1)


def _atr_barrier_backtest(
    frame: pd.DataFrame,
    actions: np.ndarray,
    tp_pct: np.ndarray,
    sl_pct: np.ndarray,
    *,
    fee: float,
    slip: float,
    unit_exposure: float,
    max_hold_bars: int,
) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    tp_pct = np.asarray(tp_pct, dtype=np.float64).reshape(-1)
    sl_pct = np.asarray(sl_pct, dtype=np.float64).reshape(-1)
    if len(tp_pct) != len(frame) or len(sl_pct) != len(frame):
        raise ValueError("tp/sl vector length must match frame length")

    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_equity = 1.0
    hold = 0
    tp = 0.0
    sl = 0.0
    trades = wins = long_entries = short_entries = 0
    exits: dict[str, int] = {}
    action_counts = {"flat": 0, "long": 0, "short": 0}
    exposure = float(unit_exposure)

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, new_side: int) -> None:
        nonlocal side, entry, entry_equity, cash, hold, tp, sl, long_entries, short_entries
        fill_i = min(i + 1, len(frame) - 1)
        side = int(new_side)
        entry = _fill_price(frame, fill_i, side, float(slip), entry=True)
        entry_equity = cash
        cash -= cash * float(fee) * exposure
        hold = 0
        tp = float(max(tp_pct[i], 1e-4))
        sl = float(max(sl_pct[i], 1e-4))
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_pos(i: int, reason: str, fill_px: float | None = None) -> None:
        nonlocal side, entry, cash, hold, tp, sl, trades, wins
        if fill_px is None:
            fill_i = min(i + 1, len(frame) - 1)
            fill_px = _fill_price(frame, fill_i, side, float(slip), entry=False)
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * float(fee) * exposure
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        side = 0
        entry = 0.0
        hold = 0
        tp = 0.0
        sl = 0.0

    for i in range(len(frame) - 2):
        desired = int(actions[i])
        action_counts["flat" if desired == 0 else "long" if desired == 1 else "short"] += 1
        if side != 0:
            hold += 1
            if side > 0:
                tp_hit = high[i] >= entry * (1.0 + tp)
                sl_hit = low[i] <= entry * (1.0 - sl)
                if tp_hit and sl_hit:
                    exit_pos(i, "ambiguous_bar_sl_first", entry * (1.0 - sl) * (1.0 - float(slip)))
                elif tp_hit:
                    exit_pos(i, "tp", entry * (1.0 + tp) * (1.0 - float(slip)))
                elif sl_hit:
                    exit_pos(i, "sl", entry * (1.0 - sl) * (1.0 - float(slip)))
            else:
                tp_hit = low[i] <= entry * (1.0 - tp)
                sl_hit = high[i] >= entry * (1.0 + sl)
                if tp_hit and sl_hit:
                    exit_pos(i, "ambiguous_bar_sl_first", entry * (1.0 + sl) * (1.0 + float(slip)))
                elif tp_hit:
                    exit_pos(i, "tp", entry * (1.0 - tp) * (1.0 + float(slip)))
                elif sl_hit:
                    exit_pos(i, "sl", entry * (1.0 + sl) * (1.0 + float(slip)))
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side != 0 and int(max_hold_bars) > 0 and hold >= int(max_hold_bars):
            exit_pos(i, "max_hold")
        elif side == 0 and desired != 0:
            enter(i, 1 if desired == 1 else -1)
    if side != 0:
        exit_pos(len(frame) - 2, "end")
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(frame)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(trades * exposure / max(len(frame), 1)),
        "action_counts": action_counts,
        "exits": exits,
    }


def _score_eval(
    frame: pd.DataFrame,
    actions: np.ndarray,
    labels: np.ndarray,
    tp_pct: np.ndarray,
    sl_pct: np.ndarray,
    *,
    fee: float,
    slip: float,
    exposure: float,
    max_hold: int,
) -> dict[str, Any]:
    bt = {
        f"cost{m}": _atr_barrier_backtest(
            frame,
            actions,
            tp_pct,
            sl_pct,
            fee=float(fee) * float(m),
            slip=float(slip) * float(m),
            unit_exposure=float(exposure),
            max_hold_bars=int(max_hold),
        )
        for m in (1, 2, 3)
    }
    dm = _direction_metrics(actions, labels)
    c1, c2, c3 = bt["cost1"], bt["cost2"], bt["cost3"]
    if int(c1["trades"]) < 20:
        score = -1e6 + float(c1["pnl"])
    else:
        score = (
            18.0 * float(dm["balanced_trade_precision"])
            + 10.0 * float(dm["trade_precision"])
            + float(c1["pnl"])
            + 0.35 * float(c2["pnl"])
            + 0.10 * float(c3["pnl"])
            - 0.22 * abs(float(c1["mdd"]))
            - max(0.0, 0.10 - float(dm["coverage"])) * 12.0
            - max(0.0, float(c1["trades_per_day"]) - 2.5) * 2.5
        )
    return {"backtest": bt, "direction": dm, "score": float(score)}


def _compose_policy(
    frame: pd.DataFrame,
    p_entry: np.ndarray,
    p_long: np.ndarray,
    *,
    entry_threshold: float,
    side_threshold: float,
    margin_threshold: float,
    tp_atr_mult: float,
    sl_atr_mult: float,
    guardrail: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    p_entry = np.clip(np.asarray(p_entry, dtype=np.float64).reshape(-1), 0.0, 1.0)
    p_long = np.clip(np.asarray(p_long, dtype=np.float64).reshape(-1), 0.0, 1.0)
    p_short = 1.0 - p_long
    margin = np.abs(p_long - p_short)
    best_side = np.maximum(p_long, p_short)
    actions = np.where(p_long >= p_short, 1, 2).astype(np.int64)
    actions = np.where(p_entry < float(entry_threshold), 0, actions)
    actions = np.where(best_side < float(side_threshold), 0, actions)
    actions = np.where(margin < float(margin_threshold), 0, actions)

    regime = frame["regime4_state"].astype(str).str.lower().to_numpy()
    if guardrail == "block_whipsaw":
        actions = np.where(regime == "whipsaw", 0, actions)
    elif guardrail == "block_whipsaw_chop":
        actions = np.where(np.isin(regime, ["whipsaw", "chop"]), 0, actions)
    elif guardrail == "trend_only":
        actions = np.where(np.isin(regime, ["bull", "bear"]), actions, 0)

    atr14 = pd.to_numeric(frame.get("atr14_pct", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    fallback_tp = pd.to_numeric(frame.get("label_tp_pct", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    fallback_sl = pd.to_numeric(frame.get("label_sl_pct", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    base_tp = np.maximum(atr14 * float(tp_atr_mult), fallback_tp * 0.5)
    base_sl = np.maximum(atr14 * float(sl_atr_mult), fallback_sl * 0.5)
    tp_pct = np.clip(base_tp, 5e-4, 0.05)
    sl_pct = np.clip(base_sl, 5e-4, 0.05)
    return actions, tp_pct, sl_pct, {
        "p_entry": p_entry,
        "p_long": p_long,
        "p_short": p_short,
        "margin": margin,
        "best_side": best_side,
        "atr14_pct": atr14,
    }


def _grid(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train alpha6 CatBoost parent baselines on DSAC feature variants.")
    p.add_argument("--variants", default=DEFAULT_VARIANTS)
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
    p.add_argument("--task-type", default="CPU")
    p.add_argument("--devices", default="0")
    p.add_argument("--entry-thresholds", default="0.45,0.50,0.55,0.60,0.65")
    p.add_argument("--side-thresholds", default="0.50,0.55,0.60,0.65")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08")
    p.add_argument("--tp-atr-mults", default="1.5,2.0,2.5,3.0")
    p.add_argument("--sl-atr-mults", default="1.0,1.2,1.5,1.8")
    p.add_argument("--guardrails", default="none,block_whipsaw,block_whipsaw_chop,trend_only")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=62101)
    p.add_argument("--startup-check-only", action="store_true")
    return p.parse_args()


def _save_model_bundle(
    out_dir: Path,
    prefix: str,
    entry_model: Any,
    direction_model: Any,
    projection: Pipeline | None,
    feature_cols: list[str],
    projection_meta: dict[str, Any],
) -> dict[str, str]:
    entry_path = out_dir / f"{prefix}_entry_model.joblib"
    direction_path = out_dir / f"{prefix}_direction_model.joblib"
    meta_path = out_dir / f"{prefix}_meta.json"
    pca_path = out_dir / f"{prefix}_pca.joblib"
    joblib.dump(entry_model, entry_path)
    joblib.dump(direction_model, direction_path)
    if projection is not None:
        joblib.dump(projection, pca_path)
    meta_path.write_text(
        json.dumps(
            {
                "feature_cols": feature_cols,
                "projection_meta": projection_meta,
                "entry_model_path": str(entry_path),
                "direction_model_path": str(direction_path),
                "pca_path": str(pca_path) if projection is not None else None,
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return {
        "entry_model": str(entry_path),
        "direction_model": str(direction_path),
        "meta": str(meta_path),
        "pca": str(pca_path) if projection is not None else "",
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read(args.raw_2025_csv)
    raw_2026 = _read(args.raw_2026_csv)
    audit = _verify_state24_sticky090_inputs(raw_2025, raw_2026, args.manifest, args.clean4_report)

    train_df = pd.read_parquet(args.label_dir / str(args.train_file))
    val_df = pd.read_parquet(args.label_dir / str(args.val_file))
    oos_df = pd.read_parquet(args.label_dir / str(args.oos_file))
    variants = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    if not variants:
        raise ValueError("no variants selected")

    if args.startup_check_only:
        print(
            json.dumps(
                {
                    "status": "startup_check_ok",
                    "model_id": MODEL_ID,
                    "variants": variants,
                    "label_dir": str(args.label_dir),
                    "spec_dir": str(args.spec_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    y_val = pd.to_numeric(val_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    y_oos = pd.to_numeric(oos_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    rows: list[dict[str, Any]] = []
    best_by_variant: dict[str, dict[str, Any]] = {}
    summary_variants: list[dict[str, Any]] = []

    print(
        json.dumps(
            {
                "stage": "alpha6_start",
                "model_id": MODEL_ID,
                "variants": variants,
                "rows": {
                    "train": int(len(train_df)),
                    "validation": int(len(val_df)),
                    "oos": int(len(oos_df)),
                },
                "audit_expected_model_found": audit.get("expected_model_found_in_manifest"),
                "rule": "label_action is excluded from fit features and fit targets",
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )

    spec_cache = {name: _read_spec(args.spec_dir, name) for name in variants}
    cat_specs = _cat_specs()
    policy_counter = 0
    for variant_idx, variant in enumerate(variants, start=1):
        spec = spec_cache[variant]
        feature_cols, leak_audit = _sanitize_feature_cols(train_df, list(spec.get("features", [])))
        if not feature_cols:
            raise ValueError(f"no usable features remain for variant={variant}")
        x_train_all, (x_val_all, x_oos_all), projection_meta, projection = _build_projection(
            train_df,
            [val_df, oos_df],
            feature_cols,
            enable_pca=bool(spec.get("extra_pca_enable", False)),
            pca_components=int(spec.get("extra_pca_components", 0) or 0),
        )
        entry_mask = pd.to_numeric(train_df["entry_train_keep"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
        dir_mask = pd.to_numeric(train_df["direction_train_keep"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
        train_entry = train_df.loc[entry_mask].reset_index(drop=True)
        train_dir = train_df.loc[dir_mask].reset_index(drop=True)
        x_train_entry = x_train_all.loc[entry_mask].reset_index(drop=True)
        x_train_dir = x_train_all.loc[dir_mask].reset_index(drop=True)
        y_entry = pd.to_numeric(train_entry["entry_label"], errors="coerce").fillna(0).to_numpy(np.int64)
        y_dir = (pd.to_numeric(train_dir["direction_label"], errors="coerce").fillna(0).to_numpy(np.int64) == 1).astype(np.int64)
        w_entry = np.clip(pd.to_numeric(train_entry["entry_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64), 1e-4, None)
        w_dir = np.clip(pd.to_numeric(train_dir["direction_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64), 1e-4, None)
        w_entry *= _balanced_weights(y_entry)
        w_dir *= _balanced_weights(y_dir)
        variant_best: dict[str, Any] | None = None

        for spec_i, entry_spec in enumerate(cat_specs, start=1):
            for dir_spec in cat_specs:
                policy_counter += 1
                bundle_name = f"{variant}_{entry_spec.name}_{dir_spec.name}"
                print(
                    json.dumps(
                        {
                            "stage": "fit_variant",
                            "variant": variant,
                            "variant_index": variant_idx,
                            "variant_total": len(variants),
                            "bundle": bundle_name,
                            "entry_spec": entry_spec.name,
                            "direction_spec": dir_spec.name,
                            "feature_count": int(projection_meta["final_feature_count"]),
                            "pca_enable": bool(projection_meta["pca_enable"]),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                entry_model = _fit_cat(
                    x_train_entry,
                    y_entry,
                    w_entry,
                    entry_spec,
                    int(args.seed + variant_idx * 1000 + spec_i * 100 + 11),
                    task_type=str(args.task_type),
                    devices=str(args.devices),
                )
                direction_model = _fit_cat(
                    x_train_dir,
                    y_dir,
                    w_dir,
                    dir_spec,
                    int(args.seed + variant_idx * 1000 + spec_i * 100 + 29),
                    task_type=str(args.task_type),
                    devices=str(args.devices),
                )
                p_entry_val = _binary_proba(entry_model, x_val_all)
                p_long_val = _binary_proba(direction_model, x_val_all)
                best_val: dict[str, Any] | None = None
                for entry_th in _grid(args.entry_thresholds):
                    for side_th in _grid(args.side_thresholds):
                        for margin_th in _grid(args.margin_thresholds):
                            for tp_mult in _grid(args.tp_atr_mults):
                                for sl_mult in _grid(args.sl_atr_mults):
                                    for guardrail in [x.strip() for x in str(args.guardrails).split(",") if x.strip()]:
                                        val_actions, val_tp, val_sl, _ = _compose_policy(
                                            val_df,
                                            p_entry_val,
                                            p_long_val,
                                            entry_threshold=entry_th,
                                            side_threshold=side_th,
                                            margin_threshold=margin_th,
                                            tp_atr_mult=tp_mult,
                                            sl_atr_mult=sl_mult,
                                            guardrail=guardrail,
                                        )
                                        val_eval = _score_eval(
                                            val_df,
                                            val_actions,
                                            y_val,
                                            val_tp,
                                            val_sl,
                                            fee=float(args.fee),
                                            slip=float(args.slip),
                                            exposure=float(args.unit_exposure),
                                            max_hold=int(args.max_hold_bars),
                                        )
                                        candidate = {
                                            "variant": variant,
                                            "entry_spec": entry_spec.name,
                                            "direction_spec": dir_spec.name,
                                            "entry_threshold": float(entry_th),
                                            "side_threshold": float(side_th),
                                            "margin_threshold": float(margin_th),
                                            "tp_atr_mult": float(tp_mult),
                                            "sl_atr_mult": float(sl_mult),
                                            "guardrail": guardrail,
                                            "validation": val_eval,
                                        }
                                        if best_val is None or float(candidate["validation"]["score"]) > float(best_val["validation"]["score"]):
                                            best_val = candidate
                assert best_val is not None
                p_entry_oos = _binary_proba(entry_model, x_oos_all)
                p_long_oos = _binary_proba(direction_model, x_oos_all)
                oos_actions, oos_tp, oos_sl, _ = _compose_policy(
                    oos_df,
                    p_entry_oos,
                    p_long_oos,
                    entry_threshold=float(best_val["entry_threshold"]),
                    side_threshold=float(best_val["side_threshold"]),
                    margin_threshold=float(best_val["margin_threshold"]),
                    tp_atr_mult=float(best_val["tp_atr_mult"]),
                    sl_atr_mult=float(best_val["sl_atr_mult"]),
                    guardrail=str(best_val["guardrail"]),
                )
                best_val["oos"] = _score_eval(
                    oos_df,
                    oos_actions,
                    y_oos,
                    oos_tp,
                    oos_sl,
                    fee=float(args.fee),
                    slip=float(args.slip),
                    exposure=float(args.unit_exposure),
                    max_hold=int(args.max_hold_bars),
                )
                best_val["feature_spec"] = {
                    "name": variant,
                    "requested_feature_count": int(spec.get("feature_count", len(spec.get("features", [])))),
                    "usable_feature_count": int(len(feature_cols)),
                    "projection": projection_meta,
                    "spec_path": spec["_spec_path"],
                }
                best_val["leak_audit"] = leak_audit
                best_val["split_contract"] = {
                    "train_file": str(args.label_dir / str(args.train_file)),
                    "val_file": str(args.label_dir / str(args.val_file)),
                    "oos_file": str(args.label_dir / str(args.oos_file)),
                    "selection_uses_label_action_only_for_metrics": True,
                }
                best_val["artifact_paths"] = _save_model_bundle(
                    args.out_dir,
                    bundle_name,
                    entry_model,
                    direction_model,
                    projection,
                    feature_cols,
                    projection_meta,
                )
                rows.append(best_val)
                if variant_best is None or float(best_val["validation"]["score"]) > float(variant_best["validation"]["score"]):
                    variant_best = best_val
        assert variant_best is not None
        best_by_variant[variant] = variant_best
        summary_variants.append(
            {
                "variant": variant,
                "validation_score": float(variant_best["validation"]["score"]),
                "val_cost1_pnl": float(variant_best["validation"]["backtest"]["cost1"]["pnl"]),
                "val_cost1_trades": int(variant_best["validation"]["backtest"]["cost1"]["trades"]),
                "oos_cost1_pnl": float(variant_best["oos"]["backtest"]["cost1"]["pnl"]),
                "oos_cost1_trades": int(variant_best["oos"]["backtest"]["cost1"]["trades"]),
                "guardrail": str(variant_best["guardrail"]),
                "tp_atr_mult": float(variant_best["tp_atr_mult"]),
                "sl_atr_mult": float(variant_best["sl_atr_mult"]),
                "entry_spec": str(variant_best["entry_spec"]),
                "direction_spec": str(variant_best["direction_spec"]),
            }
        )

    grid_rows = []
    for row in rows:
        grid_rows.append(
            {
                "variant": row["variant"],
                "entry_spec": row["entry_spec"],
                "direction_spec": row["direction_spec"],
                "entry_threshold": row["entry_threshold"],
                "side_threshold": row["side_threshold"],
                "margin_threshold": row["margin_threshold"],
                "tp_atr_mult": row["tp_atr_mult"],
                "sl_atr_mult": row["sl_atr_mult"],
                "guardrail": row["guardrail"],
                "val_score": row["validation"]["score"],
                "val_cost1_pnl": row["validation"]["backtest"]["cost1"]["pnl"],
                "val_cost1_mdd": row["validation"]["backtest"]["cost1"]["mdd"],
                "val_cost1_trades": row["validation"]["backtest"]["cost1"]["trades"],
                "val_cost2_pnl": row["validation"]["backtest"]["cost2"]["pnl"],
                "val_cost3_pnl": row["validation"]["backtest"]["cost3"]["pnl"],
                "oos_score": row["oos"]["score"],
                "oos_cost1_pnl": row["oos"]["backtest"]["cost1"]["pnl"],
                "oos_cost1_mdd": row["oos"]["backtest"]["cost1"]["mdd"],
                "oos_cost1_trades": row["oos"]["backtest"]["cost1"]["trades"],
                "usable_feature_count": row["feature_spec"]["usable_feature_count"],
                "final_feature_count": row["feature_spec"]["projection"]["final_feature_count"],
            }
        )
    pd.DataFrame(grid_rows).sort_values(["variant", "val_score"], ascending=[True, False]).to_csv(
        args.out_dir / "alpha6_parent_baseline_grid.csv",
        index=False,
    )
    summary = {
        "model_id": MODEL_ID,
        "selection_rule": "alpha5_refined_score_with_label_action_metrics_only",
        "training_rule": "fit uses entry_label, direction_label, entry_sample_weight, direction_sample_weight; label_action excluded",
        "backtest_contract": "next_open_entry_intrabar_tp_sl_sl_first_collision_cost1_2_3",
        "split_contract": {
            "train_file": str(args.label_dir / str(args.train_file)),
            "val_file": str(args.label_dir / str(args.val_file)),
            "oos_file": str(args.label_dir / str(args.oos_file)),
        },
        "variants": summary_variants,
        "best_by_variant": best_by_variant,
        "audit": {
            "preprocess_inputs": audit,
            "forbidden_feature_cols": sorted(FORBIDDEN_FEATURE_COLS),
            "forbidden_substrings": list(FORBIDDEN_SUBSTRINGS),
        },
    }
    (args.out_dir / "alpha6_parent_baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": "alpha6_done",
                "summary_path": str(args.out_dir / "alpha6_parent_baseline_summary.json"),
                "grid_path": str(args.out_dir / "alpha6_parent_baseline_grid.csv"),
                "variants": summary_variants,
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
