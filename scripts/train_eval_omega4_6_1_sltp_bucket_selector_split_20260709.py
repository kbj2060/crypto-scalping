#!/usr/bin/env python3
"""Supervised SLTP bucket selector for the three Omega4.6.1 coin models.

Splits:
- 2024: train labels from counterfactual bucket outcomes
- 2025-01..08: calibration / auxiliary validation
- 2025-09..12: final validation
- 2026-01..06: OOS

The selector does not change entry ownership. It only replaces the entry-time
TP/SL price-move barrier on candidates that the frozen model already emits.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import replay_portfolio_rl_gate_2action_native_20260708 as native  # noqa: E402
import train_portfolio_supervised_ranker_native_split_20260709 as split_world  # noqa: E402

MODEL_ID = "omega4_6_1_sltp_bucket_selector_split_20260709"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
ASSETS = ("eth", "sol", "btc")


@dataclass(frozen=True)
class Bucket:
    bucket_id: int
    name: str
    take_profit: float
    stop_loss: float


BUCKETS = (
    Bucket(0, "keep_baseline", np.nan, np.nan),
    Bucket(1, "tight", 0.025, 0.012),
    Bucket(2, "normal", 0.050, 0.025),
    Bucket(3, "wide", 0.075, 0.040),
    Bucket(4, "runner", 0.105, 0.045),
)

FEATURE_COLS = [
    "asset_eth",
    "asset_sol",
    "asset_btc",
    "component_h48qual",
    "component_zig075",
    "side_long",
    "side_short",
    "notional",
    "margin_fraction",
    "leverage",
    "base_take_profit",
    "base_stop_loss",
    "ou_halflife",
    "bar_range_pct",
    "ret_1",
    "ret_3",
    "ret_12",
    "ret_vol_12",
    "hour_sin",
    "hour_cos",
    "month_norm",
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _compound_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    for ret in ledger["trade_return"].to_numpy(dtype=np.float64):
        cash *= 1.0 + float(ret)
        wins += int(ret > 0.0)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(ledger)),
        "wr": float(wins / len(ledger)),
    }


def _num(frame: pd.DataFrame, col: str, i: int, default: float = 0.0) -> float:
    if col not in frame.columns:
        return float(default)
    val = frame[col].iloc[int(i)]
    if pd.isna(val):
        return float(default)
    return float(val)


def _entry_features(world: dict[str, Any], c: native.Candidate) -> dict[str, float]:
    frame = world[c.asset]["frame"]
    i = int(c.local_i)
    ts = pd.Timestamp(c.timestamp)
    return {
        "asset_eth": float(c.asset == "eth"),
        "asset_sol": float(c.asset == "sol"),
        "asset_btc": float(c.asset == "btc"),
        "component_h48qual": float(c.component == "h48qual"),
        "component_zig075": float(c.component == "zig075"),
        "side_long": float(c.side > 0),
        "side_short": float(c.side < 0),
        "notional": float(c.notional),
        "margin_fraction": float(c.margin),
        "leverage": float(c.leverage),
        "base_take_profit": float(c.take_profit),
        "base_stop_loss": float(c.stop_loss),
        "ou_halflife": _num(frame, "ou_halflife", i),
        "bar_range_pct": _num(frame, "bar_range_pct", i),
        "ret_1": _num(frame, "ret_1", i),
        "ret_3": _num(frame, "ret_3", i),
        "ret_12": _num(frame, "ret_12", i),
        "ret_vol_12": _num(frame, "ret_vol_12", i),
        "hour_sin": float(np.sin(2.0 * np.pi * ts.hour / 24.0)),
        "hour_cos": float(np.cos(2.0 * np.pi * ts.hour / 24.0)),
        "month_norm": float((ts.month - 6.5) / 6.0),
    }


def _bucket_candidate(c: native.Candidate, bucket: Bucket) -> native.Candidate:
    if bucket.name == "keep_baseline":
        return c
    return replace(c, take_profit=float(bucket.take_profit), stop_loss=float(bucket.stop_loss))


def _utility(row: dict[str, Any]) -> float:
    ret = float(row["trade_return"])
    mae = abs(min(float(row.get("mae_price_move", 0.0) or 0.0), 0.0))
    hold = max(int(row["exit_i"]) - int(row["entry_i"]), 0)
    stop_penalty = 0.004 if row.get("reason") == "stop_loss" else 0.0
    return float(ret - 0.25 * mae - 0.000005 * hold - stop_penalty)


def _iter_asset_candidates(world: dict[str, Any], asset: str):
    frame = world[asset]["frame"]
    for ts in frame["timestamp"]:
        ts = pd.Timestamp(ts)
        c = native._candidate_for_asset(world, asset, ts)
        if c is None:
            continue
        yield c


def _simulate_bucket_path_fast(world: dict[str, Any], c: native.Candidate, bucket: Bucket, *, max_hold_bars: int = 288) -> dict[str, Any]:
    aw = world[c.asset]
    arrays = aw["arrays"]
    fee, slip = aw["fee_slip"]
    fee_eff = float(fee) * native.COST_MULT
    slip_eff = float(slip) * native.COST_MULT
    entry_i = min(int(c.local_i) + 1, len(aw["frame"]) - 1)
    take_profit = float(c.take_profit) if bucket.name == "keep_baseline" else float(bucket.take_profit)
    stop_loss = float(c.stop_loss) if bucket.name == "keep_baseline" else float(bucket.stop_loss)
    entry_px = arrays["open"][entry_i] * (1.0 + slip_eff if c.side > 0 else 1.0 - slip_eff)
    cash_after_entry = 1.0 - fee_eff * float(c.notional)
    mfe = 0.0
    mae = 0.0
    exit_i = min(entry_i + int(max_hold_bars), len(aw["frame"]) - 1)
    reason = "time_exit"
    move = 0.0
    for j in range(entry_i, exit_i + 1):
        if c.side > 0:
            high_move = (arrays["high"][j] * (1.0 - slip_eff) - entry_px) / max(entry_px, 1e-12)
            low_move = (arrays["low"][j] * (1.0 - slip_eff) - entry_px) / max(entry_px, 1e-12)
            close_move = (arrays["close"][j] * (1.0 - slip_eff) - entry_px) / max(entry_px, 1e-12)
        else:
            high_move = (entry_px - arrays["low"][j] * (1.0 + slip_eff)) / max(entry_px, 1e-12)
            low_move = (entry_px - arrays["high"][j] * (1.0 + slip_eff)) / max(entry_px, 1e-12)
            close_move = (entry_px - arrays["close"][j] * (1.0 + slip_eff)) / max(entry_px, 1e-12)
        mfe = max(mfe, high_move)
        mae = min(mae, low_move)
        if low_move <= -abs(stop_loss):
            move = -abs(stop_loss)
            exit_i = int(j)
            reason = "stop_loss"
            break
        if high_move >= take_profit:
            move = take_profit
            exit_i = int(j)
            reason = "take_profit"
            break
        move = close_move
    cash = cash_after_entry * (1.0 + move * float(c.notional))
    cash -= cash_after_entry * fee_eff * float(c.notional)
    return {
        "reason": reason,
        "trade_return": float(cash - 1.0),
        "mae_price_move": float(mae),
        "mfe_price_move": float(mfe),
        "entry_i": int(entry_i),
        "exit_i": int(exit_i),
    }


def _build_label_dataset(world: dict[str, Any], max_events: int | None, switch_margin: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = {asset: 0 for asset in ASSETS}
    per_asset_limit = None if max_events is None else max(1, int(max_events) // len(ASSETS))
    for asset in ASSETS:
        for c in _iter_asset_candidates(world, asset):
            if per_asset_limit is not None and counts[asset] >= per_asset_limit:
                break
            bucket_rows: list[dict[str, Any]] = []
            utilities: list[float] = []
            for bucket in BUCKETS:
                row = _simulate_bucket_path_fast(world, c, bucket)
                bucket_rows.append(row)
                utilities.append(_utility(row))
            baseline_utility = float(utilities[0])
            alt_best = int(np.argmax(utilities[1:]) + 1)
            best = alt_best if float(utilities[alt_best]) > baseline_utility + float(switch_margin) else 0
            best_bucket = BUCKETS[best]
            feat = _entry_features(world, c)
            out = {
                "timestamp": c.timestamp,
                "asset": c.asset,
                "component": c.component,
                "best_bucket": int(best_bucket.bucket_id),
                "best_bucket_name": best_bucket.name,
                "best_utility": float(utilities[best]),
                "base_utility": baseline_utility,
                "oracle_best_bucket": int(np.argmax(utilities)),
                "oracle_best_bucket_name": BUCKETS[int(np.argmax(utilities))].name,
                "switch_margin": float(switch_margin),
            }
            for bucket, row, util in zip(BUCKETS, bucket_rows, utilities):
                out[f"utility_{bucket.name}"] = float(util)
                out[f"return_{bucket.name}"] = float(row["trade_return"])
                out[f"reason_{bucket.name}"] = str(row["reason"])
            rows.append({**out, **feat})
            counts[asset] += 1
            if len(rows) % 100 == 0:
                print(f"stage=labels rows={len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("empty SLTP bucket label dataset")
    diag = {
        "rows": int(len(df)),
        "asset_counts": counts,
        "bucket_counts": {str(k): int(v) for k, v in df["best_bucket"].value_counts().sort_index().items()},
    }
    return df, diag


def _train_model(train_df: pd.DataFrame) -> tuple[HistGradientBoostingClassifier, dict[str, Any]]:
    x = train_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = train_df["best_bucket"].astype(int).to_numpy()
    model = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.045,
        max_leaf_nodes=7,
        min_samples_leaf=12,
        l2_regularization=1.5,
        random_state=60709,
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    model.fit(x, y, sample_weight=sample_weight)
    pred = model.predict(x)
    diag = {
        "train_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "train_pred_counts": {str(k): int(v) for k, v in pd.Series(pred).value_counts().sort_index().items()},
    }
    return model, diag


def _predict_bucket(model: HistGradientBoostingClassifier, world: dict[str, Any], c: native.Candidate) -> Bucket:
    feat = pd.DataFrame([_entry_features(world, c)], columns=FEATURE_COLS)
    pred = int(model.predict(feat.to_numpy(dtype=np.float64))[0])
    pred = int(np.clip(pred, 0, len(BUCKETS) - 1))
    return BUCKETS[pred]


def _replay_asset(world: dict[str, Any], asset: str, device: torch.device, model: HistGradientBoostingClassifier | None) -> tuple[dict[str, Any], pd.DataFrame]:
    position: native.Position | None = None
    cash = 1.0
    rows: list[dict[str, Any]] = []
    frame = world[asset]["frame"]
    bucket_counts: dict[str, int] = {}
    for ts in frame["timestamp"]:
        ts = pd.Timestamp(ts)
        if position is not None:
            position, cash, closed, _mark = native._try_close(world, position, ts, cash, device)
            if closed is not None:
                rows.append(closed)
            continue
        c = native._candidate_for_asset(world, asset, ts)
        if c is None:
            continue
        bucket_name = "baseline"
        if model is not None:
            bucket = _predict_bucket(model, world, c)
            bucket_name = bucket.name
            c = _bucket_candidate(c, bucket)
        bucket_counts[bucket_name] = bucket_counts.get(bucket_name, 0) + 1
        position, cash = native._open_position(world, c, cash)
        setattr(position, "sltp_bucket", bucket_name)
    if position is not None:
        cash, closed = native._force_close(world, position, cash)
        rows.append(closed)
    ledger = pd.DataFrame(rows)
    if not ledger.empty and model is not None:
        # Bucket counts are entry counts; closed rows may be fewer only at forced end edge cases.
        ledger["sltp_bucket_model"] = True
    metrics = _compound_metrics(ledger)
    metrics["bucket_counts"] = bucket_counts
    metrics["reason_counts"] = ledger["reason"].value_counts().to_dict() if not ledger.empty else {}
    return metrics, ledger


def _replay_split(world: dict[str, Any], device: torch.device, model: HistGradientBoostingClassifier | None) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    metrics: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    all_ledgers: list[pd.DataFrame] = []
    for asset in ASSETS:
        m, ledger = _replay_asset(world, asset, device, model)
        metrics[asset] = m
        ledgers[asset] = ledger
        if not ledger.empty:
            all_ledgers.append(ledger)
    combined = pd.concat(all_ledgers, ignore_index=True) if all_ledgers else pd.DataFrame()
    metrics["combined_independent"] = {
        **_compound_metrics(combined),
        "reason_counts": combined["reason"].value_counts().to_dict() if not combined.empty else {},
    }
    ledgers["combined_independent"] = combined
    return metrics, ledgers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--max-train-events", type=int, default=1200)
    ap.add_argument("--switch-margin", type=float, default=0.003)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    split_specs = {
        "train_2024": ("train_2024", "2024-01-01", "2024-12-31 23:59:59"),
        "calibration_2025_01_08": ("validation_2025", "2025-01-01", "2025-08-31 23:59:59"),
        "final_validation_2025_09_12": ("validation_2025", "2025-09-01", "2025-12-31 23:59:59"),
        "oos_2026": ("oos", "2026-01-01", "2026-06-30 23:59:59"),
    }
    worlds: dict[str, dict[str, Any]] = {}
    for name, (split, start, end) in split_specs.items():
        print(f"stage=build_world name={name}", flush=True)
        worlds[name] = split_world._build_world(split, start, end, device)
        print(f"stage=world_ready name={name}", flush=True)

    print("stage=build_train_labels", flush=True)
    train_df, label_diag = _build_label_dataset(worlds["train_2024"], args.max_train_events, args.switch_margin)
    train_df.to_csv(OUT_DIR / "train_2024_sltp_bucket_labels.csv", index=False)
    model, fit_diag = _train_model(train_df)
    with open(OUT_DIR / "sltp_bucket_hgb.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS, "buckets": [b.__dict__ for b in BUCKETS]}, f)

    results: dict[str, Any] = {}
    for split_name, world in worlds.items():
        print(f"stage=replay split={split_name} baseline", flush=True)
        base_metrics, base_ledgers = _replay_split(world, device, model=None)
        print(f"stage=replay split={split_name} bucket_model", flush=True)
        bucket_metrics, bucket_ledgers = _replay_split(world, device, model=model)
        results[split_name] = {"baseline": base_metrics, "bucket_model": bucket_metrics}
        for key, ledger in base_ledgers.items():
            ledger.to_csv(OUT_DIR / f"{split_name}_{key}_baseline_ledger.csv", index=False)
        for key, ledger in bucket_ledgers.items():
            ledger.to_csv(OUT_DIR / f"{split_name}_{key}_bucket_model_ledger.csv", index=False)

    report = {
        "method": "omega4_6_1_supervised_sltp_bucket_selector",
        "model_id": MODEL_ID,
        "split_contract": {
            "train": "2024-01-01..2024-12-31",
            "calibration_aux_validation": "2025-01-01..2025-08-31",
            "final_validation": "2025-09-01..2025-12-31",
            "oos": "2026-01-01..2026-06-30",
        },
        "buckets": [b.__dict__ for b in BUCKETS],
        "feature_cols": FEATURE_COLS,
        "label_utility": "fast_OHLC_TP_SL_oracle: trade_return - 0.25*abs(MAE) - 0.000005*hold_bars - 0.004*stop_loss_hit",
        "switch_margin": float(args.switch_margin),
        "label_diag": label_diag,
        "fit_diag": fit_diag,
        "results": results,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "oos_used_for_training_or_selection": False,
        "promotion_grade": False,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "results": results}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
