#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from ensemble.fully_learned_governor_policy import prepare_features  # noqa: E402
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_alpha5_5_lgbm_supervised_parent_20260518 import _decide_actions, _predict_proba_3  # noqa: E402
from scripts.train_eval_alpha5_8_hgb_action_feature_contract_compare_20260518 import _alpha4_mapped_features  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _fill_price, _json_default, _read, _days  # noqa: E402
from scripts.tune_alpha5_9_hgb_action_master_20260518 import _fit_hgb, _hgb_specs  # noqa: E402


MODEL_ID = "alpha5_13_hgb_single_20260518"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_high_quality_training_data_20260518"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_single_20260518"


def _x(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return prepare_features(frame, side_hint=0, close=_close(frame), feature_cols=cols)


def _feature_cols(train_raw: pd.DataFrame, eval_raw: pd.DataFrame, track: str, available: set[str]) -> list[str]:
    include_future = track == "regime4_core_future"
    cols = _alpha4_mapped_features(train_raw, eval_raw, include_future=include_future)
    out = [c for c in cols if c in available]
    return out


def _direction_metrics(actions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    trade = actions != 0
    n_trade = int(np.sum(trade))
    out: dict[str, Any] = {"coverage": float(np.mean(trade)), "trades_pred": n_trade}
    if n_trade == 0:
        out.update({"trade_precision": 0.0, "balanced_trade_precision": 0.0, "long_precision": 0.0, "short_precision": 0.0})
        return out
    out["trade_precision"] = float(np.mean(actions[trade] == labels[trade]))
    parts = []
    for cls, name in ((1, "long"), (2, "short")):
        m = trade & (actions == cls)
        if np.any(m):
            p = float(np.mean(labels[m] == cls))
            out[f"{name}_precision"] = p
            out[f"{name}_pred"] = int(np.sum(m))
            parts.append(p)
        else:
            out[f"{name}_precision"] = 0.0
            out[f"{name}_pred"] = 0
    out["balanced_trade_precision"] = float(np.mean(parts)) if parts else 0.0
    return out


def _backtest_barrier(
    frame: pd.DataFrame,
    actions: np.ndarray,
    *,
    fee: float,
    slip: float,
    unit_exposure: float,
    max_hold_bars: int,
) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    tp_pct = pd.to_numeric(frame["label_tp_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    sl_pct = pd.to_numeric(frame["label_sl_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

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


def _eval_candidate(frame: pd.DataFrame, actions: np.ndarray, *, fee: float, slip: float, exposure: float, max_hold: int, labels: np.ndarray) -> dict[str, Any]:
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
    if int(c1["trades"]) < 15:
        score = -1e6 + float(c1["pnl"])
    else:
        score = (
            float(c1["pnl"])
            + 0.45 * float(c2["pnl"])
            + 0.20 * float(c3["pnl"])
            + 12.0 * float(dm["balanced_trade_precision"])
            + 8.0 * float(dm["trade_precision"])
            - 0.25 * abs(float(c1["mdd"]))
            - max(0.0, 0.15 - float(dm["coverage"])) * 10.0
            - max(0.0, float(c1["trades_per_day"]) - 5.0) * 1.5
        )
    return {"backtest": bt, "direction": dm, "score": float(score)}


def _grid(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Train and backtest a single HGB parent on Alpha5.13 high-quality labels.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--raw-2025-csv", type=Path, default=DEFAULT_RAW_2025)
    p.add_argument("--raw-2026-csv", type=Path, default=DEFAULT_RAW_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--tracks", default="regime4_core,regime4_core_future")
    p.add_argument("--prob-thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.93,0.95")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08,0.12,0.16,0.20")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=51301)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read(args.raw_2025_csv)
    raw_2026 = _read(args.raw_2026_csv)
    audit = _verify_state24_sticky090_inputs(raw_2025, raw_2026, args.manifest, args.clean4_report)

    train_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_train.parquet")
    val_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_val.parquet")
    oos_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_oos.parquet")

    train_fit = train_df[train_df["label_train_keep"] == 1].reset_index(drop=True)
    tracks = [x.strip() for x in str(args.tracks).split(",") if x.strip()]
    hgb_specs = _hgb_specs()
    total = len(tracks) * len(hgb_specs)
    done = 0
    rows: list[dict[str, Any]] = []

    print(json.dumps({
        "stage": "start",
        "model_id": MODEL_ID,
        "tracks": tracks,
        "rows": {
            "train_all": int(len(train_df)),
            "train_fit": int(len(train_fit)),
            "validation": int(len(val_df)),
            "oos": int(len(oos_df)),
        },
        "audit_expected_model_found": audit.get("expected_model_found_in_manifest"),
    }, ensure_ascii=False, default=_json_default), flush=True)

    for track_i, track in enumerate(tracks):
        cols = _feature_cols(raw_2025, raw_2026, track, set(train_df.columns))
        if not cols:
            raise ValueError(f"no usable features for track={track}")
        x_train = _x(train_fit, cols)
        x_val = _x(val_df, cols)
        x_oos = _x(oos_df, cols)
        y_train = pd.to_numeric(train_fit["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
        y_val = pd.to_numeric(val_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
        y_oos = pd.to_numeric(oos_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
        w_train = pd.to_numeric(train_fit["label_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)

        print(json.dumps({
            "stage": "features_ready",
            "track": track,
            "feature_count": len(cols),
            "future_pred_count": int(sum(str(c).startswith("regime4_pred_") for c in cols)),
            "clean4_count": int(sum(str(c).startswith("clean_regime4_2024_unsup_v1_") for c in cols)),
        }, ensure_ascii=False), flush=True)

        for spec_i, spec in enumerate(hgb_specs):
            done += 1
            print(json.dumps({"stage": "fit", "done": done, "total": total, "track": track, "hgb": spec.name}, ensure_ascii=False), flush=True)
            model = _fit_hgb(x_train, y_train, w_train, spec, int(args.seed + track_i * 100 + spec_i))
            val_proba = _predict_proba_3(model, x_val)
            oos_proba = _predict_proba_3(model, x_oos)

            best_val: dict[str, Any] | None = None
            for prob in _grid(args.prob_thresholds):
                for margin in _grid(args.margin_thresholds):
                    val_actions = _decide_actions(val_proba, prob, margin)
                    val_eval = _eval_candidate(
                        val_df,
                        val_actions,
                        fee=float(args.fee),
                        slip=float(args.slip),
                        exposure=float(args.unit_exposure),
                        max_hold=int(args.max_hold_bars),
                        labels=y_val,
                    )
                    row = {"prob": prob, "margin": margin, "actions": val_actions, **val_eval}
                    if best_val is None or float(row["score"]) > float(best_val["score"]):
                        best_val = row
            assert best_val is not None

            oos_actions = _decide_actions(oos_proba, float(best_val["prob"]), float(best_val["margin"]))
            oos_eval = _eval_candidate(
                oos_df,
                oos_actions,
                fee=float(args.fee),
                slip=float(args.slip),
                exposure=float(args.unit_exposure),
                max_hold=int(args.max_hold_bars),
                labels=y_oos,
            )
            artifact = args.out_dir / f"{track}_{spec.name}_alpha5_13_hgb_parent.joblib"
            joblib.dump({
                "model_id": MODEL_ID,
                "model": model,
                "feature_cols": cols,
                "track": track,
                "hgb": {
                    "name": spec.name,
                    "max_iter": int(spec.max_iter),
                    "learning_rate": float(spec.learning_rate),
                    "max_leaf_nodes": int(spec.max_leaf_nodes),
                    "min_samples_leaf": int(spec.min_samples_leaf),
                    "l2_regularization": float(spec.l2_regularization),
                },
                "decision": {"prob": float(best_val["prob"]), "margin": float(best_val["margin"])},
            }, artifact)
            row = {
                "track": track,
                "hgb": {
                    "name": spec.name,
                    "max_iter": int(spec.max_iter),
                    "learning_rate": float(spec.learning_rate),
                    "max_leaf_nodes": int(spec.max_leaf_nodes),
                    "min_samples_leaf": int(spec.min_samples_leaf),
                    "l2_regularization": float(spec.l2_regularization),
                },
                "feature_count": len(cols),
                "validation": {k: v for k, v in best_val.items() if k != "actions"},
                "oos": oos_eval,
                "artifact": str(artifact),
            }
            rows.append(row)
            print(json.dumps({
                "stage": "candidate",
                "track": track,
                "hgb": spec.name,
                "feature_count": len(cols),
                "prob": best_val["prob"],
                "margin": best_val["margin"],
                "val_score": best_val["score"],
                "val_dir": best_val["direction"],
                "val_cost1": best_val["backtest"]["cost1"],
                "oos_score": oos_eval["score"],
                "oos_dir": oos_eval["direction"],
                "oos_cost1": oos_eval["backtest"]["cost1"],
            }, ensure_ascii=False, default=_json_default), flush=True)

    best = max(rows, key=lambda r: float(r["validation"]["score"]))
    summary = {
        "model_id": MODEL_ID,
        "design": "Standalone HGB parent on Alpha5.13 high-quality labels with ATR barrier backtest.",
        "experiments": rows,
        "best": best,
        "top10": sorted(rows, key=lambda r: float(r["validation"]["score"]), reverse=True)[:10],
    }
    summary_path = args.out_dir / "alpha5_13_hgb_single_summary.json"
    grid_path = args.out_dir / "alpha5_13_hgb_single_grid.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame([
        {
            "track": r["track"],
            "hgb_name": r["hgb"]["name"],
            "feature_count": r["feature_count"],
            "val_score": r["validation"]["score"],
            "val_prob": r["validation"]["prob"],
            "val_margin": r["validation"]["margin"],
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
            "oos_cost2_pnl": r["oos"]["backtest"]["cost2"]["pnl"],
            "oos_cost3_pnl": r["oos"]["backtest"]["cost3"]["pnl"],
            "artifact": r["artifact"],
        }
        for r in rows
    ]).sort_values("val_score", ascending=False).to_csv(grid_path, index=False)
    print(json.dumps({
        "stage": "complete",
        "summary": str(summary_path),
        "grid": str(grid_path),
        "best": {
            "track": best["track"],
            "hgb": best["hgb"]["name"],
            "feature_count": best["feature_count"],
            "val_score": best["validation"]["score"],
            "oos_score": best["oos"]["score"],
            "oos_cost1": best["oos"]["backtest"]["cost1"],
            "oos_direction": best["oos"]["direction"],
        },
    }, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
