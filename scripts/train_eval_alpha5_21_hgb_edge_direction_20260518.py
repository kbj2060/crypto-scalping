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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import prepare_features  # noqa: E402
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_alpha5_8_hgb_action_feature_contract_compare_20260518 import _alpha4_mapped_features  # noqa: E402
from scripts.train_eval_alpha5_13_hgb_single_20260518 import _backtest_barrier, _direction_metrics  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.tune_alpha5_9_hgb_action_master_20260518 import HGBSpec, _fit_hgb, _hgb_specs  # noqa: E402


MODEL_ID = "alpha5_21_hgb_edge_direction_20260518"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_18_hgb_soft_labels_20260518"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_21_hgb_edge_direction_20260518"


@dataclass(frozen=True)
class RegressorSpec:
    name: str
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float


def _x(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return prepare_features(frame, side_hint=0, close=_close(frame), feature_cols=cols)


def _feature_cols(train_raw: pd.DataFrame, eval_raw: pd.DataFrame, available: set[str]) -> list[str]:
    cols = _alpha4_mapped_features(train_raw, eval_raw, include_future=False)
    return [c for c in cols if c in available]


def _binary_proba(model: Any, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return raw[:, classes.index(1)]
    return raw[:, -1]


def _fit_hgbr(x: pd.DataFrame, y: np.ndarray, w: np.ndarray, spec: RegressorSpec, seed: int) -> Any:
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=int(spec.max_iter),
            learning_rate=float(spec.learning_rate),
            max_leaf_nodes=int(spec.max_leaf_nodes),
            min_samples_leaf=int(spec.min_samples_leaf),
            l2_regularization=float(spec.l2_regularization),
            early_stopping=False,
            random_state=int(seed),
        ),
    )
    model.fit(x, y, histgradientboostingregressor__sample_weight=w)
    return model


def _trade_target(frame: pd.DataFrame, strict_level: str) -> np.ndarray:
    action = pd.to_numeric(frame["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    tp = pd.to_numeric(frame["meta_tp_first"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    prof = pd.to_numeric(frame["meta_is_profitable"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    sel = pd.to_numeric(frame["regime_trade_selected"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    cons = pd.to_numeric(frame["label_consensus"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    edge = pd.to_numeric(frame["meta_edge_gap"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    if strict_level == "strict2":
        mask = (action != 0) & tp & prof & sel & (cons >= 0.75)
    elif strict_level == "strict3":
        mask = (action != 0) & tp & prof & sel & (cons >= 0.75) & (edge >= 1.5)
    elif strict_level == "strict4":
        mask = (action != 0) & tp & prof & sel & (cons >= 1.0) & (edge >= 2.0)
    else:
        raise ValueError(strict_level)
    return mask.astype(np.int64)


def _edge_targets(frame: pd.DataFrame, scale: float) -> tuple[np.ndarray, np.ndarray]:
    long_score = pd.to_numeric(frame["meta_long_score"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    short_score = pd.to_numeric(frame["meta_short_score"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    long_edge = np.maximum(long_score - short_score, 0.0)
    short_edge = np.maximum(short_score - long_score, 0.0)
    return np.tanh(np.clip(long_edge, 0.0, 12.0) / float(scale)), np.tanh(np.clip(short_edge, 0.0, 12.0) / float(scale))


def _trade_weights(frame: pd.DataFrame, y: np.ndarray, cash_dampen: float) -> np.ndarray:
    base = pd.to_numeric(frame["label_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    cons = pd.to_numeric(frame["label_consensus"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    out = np.clip(base, 1e-4, None) * (0.75 + 0.45 * np.clip(cons, 0.0, 1.0))
    out[y == 0] *= float(cash_dampen)
    # balanced binary
    cls, cnt = np.unique(y, return_counts=True)
    total = max(float(len(y)), 1.0)
    for c, n in zip(cls, cnt):
        out[y == int(c)] *= total / (2.0 * max(float(n), 1.0))
    return out


def _edge_weights(frame: pd.DataFrame, target: np.ndarray, boost: float) -> np.ndarray:
    base = pd.to_numeric(frame["label_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    tp = pd.to_numeric(frame["meta_tp_first"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    prof = pd.to_numeric(frame["meta_is_profitable"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    mag = np.abs(np.asarray(target, dtype=np.float64))
    out = np.clip(base, 1e-4, None) * (0.70 + 0.90 * np.clip(mag, 0.0, 1.0))
    out *= 1.0 + float(boost) * (target > 0).astype(np.float64)
    out *= 1.0 + 0.15 * tp + 0.10 * prof
    return out


def _reg_specs() -> dict[str, RegressorSpec]:
    return {
        "regularized": RegressorSpec("regularized", 240, 0.040, 25, 90, 0.16),
        "deeper": RegressorSpec("deeper", 420, 0.035, 45, 35, 0.05),
    }


def _archs() -> list[dict[str, Any]]:
    cls = {s.name: s for s in _hgb_specs()}
    reg = _reg_specs()
    return [
        {"name": "edge_v1", "trade": cls["regularized"], "long": reg["deeper"], "short": reg["deeper"], "strict": "strict3", "cash_dampen": 0.85, "edge_scale": 3.0, "edge_boost": 0.25},
        {"name": "edge_v2", "trade": cls["deeper"], "long": reg["deeper"], "short": reg["regularized"], "strict": "strict3", "cash_dampen": 0.80, "edge_scale": 3.5, "edge_boost": 0.30},
        {"name": "edge_v3", "trade": cls["regularized"], "long": reg["regularized"], "short": reg["regularized"], "strict": "strict4", "cash_dampen": 0.75, "edge_scale": 2.5, "edge_boost": 0.20},
    ]


def _compose_actions(
    p_trade: np.ndarray,
    long_edge: np.ndarray,
    short_edge: np.ndarray,
    *,
    trade_threshold: float,
    edge_threshold: float,
    margin_threshold: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    best = np.maximum(long_edge, short_edge)
    margin = np.abs(long_edge - short_edge)
    actions = np.where(long_edge >= short_edge, 1, 2).astype(np.int64)
    actions = np.where(p_trade < float(trade_threshold), 0, actions)
    actions = np.where(best < float(edge_threshold), 0, actions)
    actions = np.where(margin < float(margin_threshold), 0, actions)
    return actions, {
        "trade_score": p_trade,
        "long_edge": long_edge,
        "short_edge": short_edge,
        "best_edge": best,
        "edge_margin": margin,
    }


def _eval(frame: pd.DataFrame, actions: np.ndarray, labels: np.ndarray, *, fee: float, slip: float, exposure: float, max_hold: int) -> dict[str, Any]:
    bt = {
        f"cost{m}": _backtest_barrier(
            frame,
            actions,
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


def _grid(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Train HGB action selector with strict trade gate and edge-only direction heads.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--raw-2025-csv", type=Path, default=DEFAULT_RAW_2025)
    p.add_argument("--raw-2026-csv", type=Path, default=DEFAULT_RAW_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--trade-thresholds", default="0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    p.add_argument("--edge-thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35")
    p.add_argument("--margin-thresholds", default="0.03,0.05,0.08,0.12,0.16,0.20")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=52101)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read(args.raw_2025_csv)
    raw_2026 = _read(args.raw_2026_csv)
    audit = _verify_state24_sticky090_inputs(raw_2025, raw_2026, args.manifest, args.clean4_report)

    train_df = pd.read_parquet(args.data_dir / "alpha5_18_hgb_soft_labels_train.parquet")
    val_df = pd.read_parquet(args.data_dir / "alpha5_18_hgb_soft_labels_val.parquet")
    oos_df = pd.read_parquet(args.data_dir / "alpha5_18_hgb_soft_labels_oos.parquet")

    cols = _feature_cols(raw_2025, raw_2026, set(train_df.columns))
    x_train_all = _x(train_df, cols)
    x_val = _x(val_df, cols)
    x_oos = _x(oos_df, cols)
    y_val = pd.to_numeric(val_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    y_oos = pd.to_numeric(oos_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)

    print(json.dumps({
        "stage": "start",
        "model_id": MODEL_ID,
        "rows": {"train": int(len(train_df)), "validation": int(len(val_df)), "oos": int(len(oos_df))},
        "feature_count": int(len(cols)),
        "audit_expected_model_found": audit.get("expected_model_found_in_manifest"),
    }, ensure_ascii=False, default=_json_default), flush=True)

    rows: list[dict[str, Any]] = []
    for i, arch in enumerate(_archs(), start=1):
        spec_dump = {
            "name": arch["name"],
            "trade": arch["trade"].name,
            "long": arch["long"].name,
            "short": arch["short"].name,
            "strict": arch["strict"],
            "cash_dampen": arch["cash_dampen"],
            "edge_scale": arch["edge_scale"],
            "edge_boost": arch["edge_boost"],
        }
        y_trade = _trade_target(train_df, arch["strict"])
        trade_idx = np.flatnonzero(y_trade == 1)
        x_trade = x_train_all.iloc[trade_idx].reset_index(drop=True)
        trade_frame = train_df.iloc[trade_idx].reset_index(drop=True)
        y_long, y_short = _edge_targets(trade_frame, arch["edge_scale"])

        print(json.dumps({
            "stage": "fit",
            "done": i,
            "total": len(_archs()),
            "architecture": arch["name"],
            "strict": arch["strict"],
            "trade_ratio": float(np.mean(y_trade)),
            "trade_rows": int(len(trade_idx)),
            "long_edge_mean": float(np.mean(y_long)) if len(y_long) else 0.0,
            "short_edge_mean": float(np.mean(y_short)) if len(y_short) else 0.0,
        }, ensure_ascii=False), flush=True)

        w_trade = _trade_weights(train_df, y_trade, arch["cash_dampen"])
        w_long = _edge_weights(trade_frame, y_long, arch["edge_boost"])
        w_short = _edge_weights(trade_frame, y_short, arch["edge_boost"])

        trade_model = _fit_hgb(x_train_all, y_trade, w_trade, arch["trade"], int(args.seed + i * 100 + 1))
        long_model = _fit_hgbr(x_trade, y_long, w_long, arch["long"], int(args.seed + i * 100 + 2))
        short_model = _fit_hgbr(x_trade, y_short, w_short, arch["short"], int(args.seed + i * 100 + 3))

        p_trade_val = _binary_proba(trade_model, x_val)
        p_long_val = np.asarray(long_model.predict(x_val), dtype=np.float64)
        p_short_val = np.asarray(short_model.predict(x_val), dtype=np.float64)

        best_val: dict[str, Any] | None = None
        for trade_threshold in _grid(args.trade_thresholds):
            for edge_threshold in _grid(args.edge_thresholds):
                for margin_threshold in _grid(args.margin_thresholds):
                    val_actions, val_diag = _compose_actions(
                        p_trade_val,
                        p_long_val,
                        p_short_val,
                        trade_threshold=trade_threshold,
                        edge_threshold=edge_threshold,
                        margin_threshold=margin_threshold,
                    )
                    val_eval = _eval(
                        val_df,
                        val_actions,
                        y_val,
                        fee=float(args.fee),
                        slip=float(args.slip),
                        exposure=float(args.unit_exposure),
                        max_hold=int(args.max_hold_bars),
                    )
                    cand = {
                        "trade_threshold": float(trade_threshold),
                        "edge_threshold": float(edge_threshold),
                        "margin_threshold": float(margin_threshold),
                        "diag": {
                            "trade_score_mean": float(np.mean(val_diag["trade_score"])),
                            "long_edge_mean": float(np.mean(val_diag["long_edge"])),
                            "short_edge_mean": float(np.mean(val_diag["short_edge"])),
                            "best_edge_mean": float(np.mean(val_diag["best_edge"])),
                            "edge_margin_mean": float(np.mean(val_diag["edge_margin"])),
                        },
                        **val_eval,
                    }
                    if best_val is None or float(cand["score"]) > float(best_val["score"]):
                        best_val = cand
        assert best_val is not None

        p_trade_oos = _binary_proba(trade_model, x_oos)
        p_long_oos = np.asarray(long_model.predict(x_oos), dtype=np.float64)
        p_short_oos = np.asarray(short_model.predict(x_oos), dtype=np.float64)
        oos_actions, oos_diag = _compose_actions(
            p_trade_oos,
            p_long_oos,
            p_short_oos,
            trade_threshold=float(best_val["trade_threshold"]),
            edge_threshold=float(best_val["edge_threshold"]),
            margin_threshold=float(best_val["margin_threshold"]),
        )
        oos_eval = _eval(
            oos_df,
            oos_actions,
            y_oos,
            fee=float(args.fee),
            slip=float(args.slip),
            exposure=float(args.unit_exposure),
            max_hold=int(args.max_hold_bars),
        )
        oos_eval["diag"] = {
            "trade_score_mean": float(np.mean(oos_diag["trade_score"])),
            "long_edge_mean": float(np.mean(oos_diag["long_edge"])),
            "short_edge_mean": float(np.mean(oos_diag["short_edge"])),
            "best_edge_mean": float(np.mean(oos_diag["best_edge"])),
            "edge_margin_mean": float(np.mean(oos_diag["edge_margin"])),
        }

        artifact = args.out_dir / f"{arch['name']}_alpha5_21_hgb_edge_direction.joblib"
        joblib.dump({
            "model_id": MODEL_ID,
            "feature_cols": cols,
            "models": {"trade": trade_model, "long": long_model, "short": short_model},
            "specs": spec_dump,
            "decision": {
                "trade_threshold": float(best_val["trade_threshold"]),
                "edge_threshold": float(best_val["edge_threshold"]),
                "margin_threshold": float(best_val["margin_threshold"]),
            },
        }, artifact)

        row = {
            "architecture": arch["name"],
            "specs": spec_dump,
            "feature_count": int(len(cols)),
            "validation": best_val,
            "oos": oos_eval,
            "artifact": str(artifact),
        }
        rows.append(row)
        print(json.dumps({
            "stage": "candidate",
            "architecture": arch["name"],
            "specs": spec_dump,
            "val_score": best_val["score"],
            "val_cost1": best_val["backtest"]["cost1"],
            "val_direction": best_val["direction"],
            "oos_score": oos_eval["score"],
            "oos_cost1": oos_eval["backtest"]["cost1"],
            "oos_direction": oos_eval["direction"],
        }, ensure_ascii=False, default=_json_default), flush=True)

    best = max(rows, key=lambda r: float(r["validation"]["score"]))
    summary = {
        "model_id": MODEL_ID,
        "design": "Stage1 on full train, Stage2 edge-only regressors on strict trade subset.",
        "experiments": rows,
        "best": best,
        "top10": sorted(rows, key=lambda r: float(r["validation"]["score"]), reverse=True)[:10],
    }
    summary_path = args.out_dir / "alpha5_21_hgb_edge_direction_summary.json"
    grid_path = args.out_dir / "alpha5_21_hgb_edge_direction_grid.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame([
        {
            "architecture": r["architecture"],
            "strict": r["specs"]["strict"],
            "trade_hgb": r["specs"]["trade"].name if isinstance(r["specs"]["trade"], HGBSpec) else r["specs"]["trade"],
            "long_hgb": r["specs"]["long"].name if isinstance(r["specs"]["long"], RegressorSpec) else r["specs"]["long"],
            "short_hgb": r["specs"]["short"].name if isinstance(r["specs"]["short"], RegressorSpec) else r["specs"]["short"],
            "cash_dampen": r["specs"]["cash_dampen"],
            "edge_scale": r["specs"]["edge_scale"],
            "edge_boost": r["specs"]["edge_boost"],
            "feature_count": r["feature_count"],
            "val_score": r["validation"]["score"],
            "val_trade_threshold": r["validation"]["trade_threshold"],
            "val_edge_threshold": r["validation"]["edge_threshold"],
            "val_margin_threshold": r["validation"]["margin_threshold"],
            "val_trade_precision": r["validation"]["direction"]["trade_precision"],
            "val_balanced_trade_precision": r["validation"]["direction"]["balanced_trade_precision"],
            "val_coverage": r["validation"]["direction"]["coverage"],
            "val_cost1_pnl": r["validation"]["backtest"]["cost1"]["pnl"],
            "val_cost1_mdd": r["validation"]["backtest"]["cost1"]["mdd"],
            "val_cost1_trades": r["validation"]["backtest"]["cost1"]["trades"],
            "oos_score": r["oos"]["score"],
            "oos_trade_precision": r["oos"]["direction"]["trade_precision"],
            "oos_balanced_trade_precision": r["oos"]["direction"]["balanced_trade_precision"],
            "oos_coverage": r["oos"]["direction"]["coverage"],
            "oos_cost1_pnl": r["oos"]["backtest"]["cost1"]["pnl"],
            "oos_cost1_mdd": r["oos"]["backtest"]["cost1"]["mdd"],
            "oos_cost1_trades": r["oos"]["backtest"]["cost1"]["trades"],
            "artifact": r["artifact"],
        }
        for r in rows
    ]).sort_values("val_score", ascending=False).to_csv(grid_path, index=False)
    print(json.dumps({
        "stage": "complete",
        "summary": str(summary_path),
        "grid": str(grid_path),
        "best": {
            "architecture": best["architecture"],
            "val_score": best["validation"]["score"],
            "oos_score": best["oos"]["score"],
            "oos_cost1": best["oos"]["backtest"]["cost1"],
            "oos_direction": best["oos"]["direction"],
        },
    }, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
