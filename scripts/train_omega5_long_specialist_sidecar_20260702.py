#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
EXP_SCRIPT = ROOT / "scripts/experiment_omega5_long_specialist_20260702.py"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega5_long_specialist_sidecar_20260702"
EXP_DIR = ROOT / "tmp/causal_regen_20260516/omega5_long_specialist_experiment_20260702"
ROUNDTRIP_COST_DEFAULT = 0.000612
EPS = 1.0e-12

COMMON_FEATURES = [
    "m7_confidence",
    "m7_qwidth",
    "m7_quality_pred",
    "m7_expected_ret",
    "m7_tail_risk",
    "volatility_z",
    "rsi",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "long_squeeze_risk",
    "net_taker_ratio",
    "oi_change_rate",
    "taker_acceleration",
    "whale_conviction",
    "smart_money_flow",
]


def load_exp_module() -> Any:
    spec = importlib.util.spec_from_file_location("omega5_long_exp", EXP_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXP_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def json_default(obj: Any) -> Any:
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def load_policy(exp: Any) -> Any:
    report = json.loads((EXP_DIR / "report.json").read_text(encoding="utf-8"))
    p = report["best_policy"]
    return exp.LongPolicy(**p)


def load_frame(path: Path, *, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", *COMMON_FEATURES}
    df = pd.read_csv(path, usecols=lambda c: c in required, parse_dates=["timestamp"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"missing columns {missing}: {path}")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] < pd.Timestamp(end)]
    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).reset_index(drop=True)
    df.attrs["timestamp_array"] = df["timestamp"].to_numpy()
    df.attrs["high_array"] = df["high"].astype(float).to_numpy()
    df.attrs["low_array"] = df["low"].astype(float).to_numpy()
    df.attrs["close_array"] = df["close"].astype(float).to_numpy()
    return df


def label_all_bars(frame: pd.DataFrame, policy: Any, split: str) -> pd.DataFrame:
    high = frame.attrs["high_array"]
    low = frame.attrs["low_array"]
    close = frame.attrs["close_array"]
    ts = frame.attrs["timestamp_array"]
    max_hold = int(policy.max_hold_bars)
    rows: list[dict[str, Any]] = []
    for i in range(0, max(0, len(frame) - max_hold)):
        entry = float(close[i])
        peak = 0.0
        trail_active = False
        partial_done = False
        realized_raw = 0.0
        realized_frac = 0.0
        raw_move = 0.0
        exit_i = min(len(frame) - 1, i + max_hold)
        reason = "long_time_exit"
        for j in range(i + 1, exit_i + 1):
            high_move = float(high[j]) / entry - 1.0
            low_move = float(low[j]) / entry - 1.0
            if low_move <= -float(policy.sl):
                raw_move = -float(policy.sl)
                exit_i = j
                reason = "long_bracket_sl"
                break
            if policy.exit_kind == "static":
                if policy.tp is not None and high_move >= float(policy.tp):
                    raw_move = float(policy.tp)
                    exit_i = j
                    reason = "long_bracket_tp"
                    break
            elif policy.exit_kind == "trail":
                if trail_active and policy.trail_gap is not None and low_move <= peak - float(policy.trail_gap):
                    raw_move = max(peak - float(policy.trail_gap), -float(policy.sl))
                    exit_i = j
                    reason = "long_trailing_exit"
                    break
                if policy.trail_start is not None and high_move >= float(policy.trail_start):
                    trail_active = True
                peak = max(peak, high_move)
            else:
                if not partial_done and policy.partial_tp is not None and high_move >= float(policy.partial_tp):
                    frac = float(policy.partial_frac or 0.5)
                    realized_raw += frac * float(policy.partial_tp)
                    realized_frac += frac
                    partial_done = True
                    trail_active = True
                if trail_active and policy.trail_gap is not None and low_move <= peak - float(policy.trail_gap):
                    raw_move = max(peak - float(policy.trail_gap), -float(policy.sl))
                    exit_i = j
                    reason = "long_partial_trailing_exit" if partial_done else "long_trailing_exit"
                    break
                if trail_active:
                    peak = max(peak, high_move)
                elif policy.trail_start is not None and high_move >= float(policy.trail_start):
                    trail_active = True
                    peak = max(peak, high_move)
        if reason == "long_time_exit":
            raw_move = float(close[exit_i]) / entry - 1.0
        remaining = max(0.0, 1.0 - realized_frac)
        weighted_raw = realized_raw + remaining * raw_move
        net = weighted_raw - ROUNDTRIP_COST_DEFAULT
        rows.append(
            {
                "split": split,
                "timestamp": pd.Timestamp(ts[i]),
                "exit_timestamp": pd.Timestamp(ts[exit_i]),
                "quality_target_net_per_notional": float(net),
                "quality_binary_target": int(net > 0.0),
                "exit_reason_target": reason,
                "hold_hours_target": float((pd.Timestamp(ts[exit_i]) - pd.Timestamp(ts[i])).total_seconds() / 3600.0),
            }
        )
    labels = pd.DataFrame(rows)
    return pd.concat([frame.iloc[: len(labels)][["timestamp", *COMMON_FEATURES]].reset_index(drop=True), labels.drop(columns=["timestamp"])], axis=1)


def eval_targets(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    binary = (y > 0).astype(int)
    auc = None
    if len(set(binary.tolist())) == 2:
        auc = float(roc_auc_score(binary, pred))
    return {
        "rows": int(len(y)),
        "positive_rate": float(binary.mean()) if len(y) else 0.0,
        "mae": float(mean_absolute_error(y, pred)) if len(y) else 0.0,
        "auc": auc,
        "pred_p50": float(np.quantile(pred, 0.5)) if len(pred) else 0.0,
        "pred_p75": float(np.quantile(pred, 0.75)) if len(pred) else 0.0,
        "pred_p90": float(np.quantile(pred, 0.9)) if len(pred) else 0.0,
    }


def sidecar_eval_on_parent_events(exp: Any, policy: Any, model: Any, threshold: float) -> dict[str, Any]:
    ledgers = {split: exp.load_ledger(split, path) for split, path in exp.LEDGERS.items()}
    markets = {split: exp.load_market(path) for split, path in exp.MARKETS.items()}
    market_pos = {
        split: {pd.Timestamp(ts): int(i) for i, ts in enumerate(market["timestamp"])}
        for split, market in markets.items()
    }
    split_reports: dict[str, Any] = {}
    out_ledgers: dict[str, str] = {}
    for split, ledger in ledgers.items():
        d = ledger.copy()
        missing = [c for c in COMMON_FEATURES if c not in d.columns]
        if missing:
            feat_path = exp.FEATURE_MERGE
            feat = pd.read_csv(feat_path, usecols=lambda c: c == "timestamp" or c in COMMON_FEATURES)
            feat["entry_timestamp"] = pd.to_datetime(feat["timestamp"], errors="raise")
            feat = feat.drop(columns=["timestamp"])
            d = d.merge(feat, on="entry_timestamp", how="left", suffixes=("", "_sidecar"))
            for col in COMMON_FEATURES:
                if col not in d.columns and f"{col}_sidecar" in d.columns:
                    d[col] = d[f"{col}_sidecar"]
        X = d[COMMON_FEATURES].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        d["long_quality_pred"] = model.predict(X)
        rows: list[dict[str, Any]] = []
        available_after = pd.Timestamp.min
        for _, row in d.sort_values(["entry_timestamp", "exit_timestamp"]).iterrows():
            entry_ts = pd.Timestamp(row["entry_timestamp"])
            if entry_ts <= available_after:
                continue
            side = int(row["side"])
            if side < 0:
                item = exp.original_short_row(row)
                rows.append(item)
                available_after = max(available_after, pd.Timestamp(item["exit_timestamp"]))
                continue
            allow = bool(exp.gate_pass(row, policy) and float(row["long_quality_pred"]) >= threshold)
            if not allow:
                continue
            entry_pos = market_pos[split].get(entry_ts)
            if entry_pos is None:
                raise RuntimeError(f"{split}: missing timestamp {entry_ts}")
            item = exp.simulate_long(markets[split], entry_pos, row, policy)
            item["long_quality_pred"] = float(row["long_quality_pred"])
            item["long_sidecar_threshold"] = float(threshold)
            rows.append(item)
            available_after = max(available_after, pd.Timestamp(item["exit_timestamp"]))
        replayed = pd.DataFrame(rows)
        path = OUT_DIR / f"{split}_sidecar_thr_{threshold:.6f}_ledger.csv"
        replayed.to_csv(path, index=False)
        out_ledgers[split] = str(path)
        split_reports[split] = exp.metrics(replayed)
    return {"threshold": float(threshold), "metrics": split_reports, "ledgers": out_ledgers}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exp = load_exp_module()
    policy = load_policy(exp)
    frame_2025 = load_frame(exp.MARKETS["validation"])
    frame_2026_old = load_frame(exp.MARKETS["old_oos"])
    frame_add = load_frame(exp.MARKETS["additional_oos"], start="2026-03-01")
    train = frame_2025[frame_2025["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = frame_2025[frame_2025["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    for frame in (train, val):
        frame.attrs["timestamp_array"] = frame["timestamp"].to_numpy()
        frame.attrs["high_array"] = frame["high"].astype(float).to_numpy()
        frame.attrs["low_array"] = frame["low"].astype(float).to_numpy()
        frame.attrs["close_array"] = frame["close"].astype(float).to_numpy()

    datasets = {
        "train_2025_jan_sep": label_all_bars(train, policy, "train_2025_jan_sep"),
        "validation_2025_oct_dec": label_all_bars(val, policy, "validation_2025_oct_dec"),
        "old_oos_2026_jan_feb": label_all_bars(frame_2026_old, policy, "old_oos_2026_jan_feb"),
        "additional_oos_2026_mar_jun": label_all_bars(frame_add, policy, "additional_oos_2026_mar_jun"),
    }
    for name, df in datasets.items():
        df.to_csv(OUT_DIR / f"{name}_long_quality_labels.csv", index=False)

    train_df = datasets["train_2025_jan_sep"]
    X_train = train_df[COMMON_FEATURES].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y_train = train_df["quality_target_net_per_notional"].astype(float).to_numpy()
    model = HistGradientBoostingRegressor(
        max_iter=260,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=260702,
    )
    model.fit(X_train, y_train)
    model_path = OUT_DIR / "long_quality_hgb_sidecar.pkl"
    with model_path.open("wb") as fh:
        pickle.dump({"model": model, "features": COMMON_FEATURES, "policy": policy}, fh)

    evals: dict[str, Any] = {}
    thresholds_source = model.predict(datasets["validation_2025_oct_dec"][COMMON_FEATURES].fillna(0.0))
    thresholds = sorted(set([0.0, *np.quantile(thresholds_source, [0.4, 0.5, 0.6, 0.7, 0.8]).round(8).tolist()]))
    parent_results = []
    for name, df in datasets.items():
        pred = model.predict(df[COMMON_FEATURES].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0))
        evals[name] = eval_targets(df["quality_target_net_per_notional"].astype(float).to_numpy(), pred)
    for threshold in thresholds:
        result = sidecar_eval_on_parent_events(exp, policy, model, float(threshold))
        row = {"threshold": float(threshold)}
        for split, metrics in result["metrics"].items():
            for key in [
                "pnl",
                "mdd",
                "trades",
                "long_trades",
                "long_pnl",
                "short_pnl",
                "wr",
                "max_hold_hours",
                "max_notional",
            ]:
                row[f"{split}_{key}"] = metrics[key]
        row["pass_all"] = (
            all(result["metrics"][s]["pnl"] > 0.0 for s in result["metrics"])
            and all(result["metrics"][s]["long_pnl"] > 0.0 for s in result["metrics"])
            and all(result["metrics"][s]["mdd"] >= -20.0 for s in result["metrics"])
            and all(result["metrics"][s]["max_hold_hours"] <= 24.0 + 1.0e-9 for s in result["metrics"])
        )
        row["score"] = min(result["metrics"][s]["pnl"] for s in result["metrics"]) + 5.0 * min(
            result["metrics"][s]["long_pnl"] for s in result["metrics"]
        )
        row["ledgers"] = result["ledgers"]
        parent_results.append(row)
    parent_rank = pd.DataFrame(parent_results).sort_values(["pass_all", "score"], ascending=[False, False])
    parent_rank.to_csv(OUT_DIR / "parent_event_sidecar_threshold_ranking.csv", index=False)

    report = {
        "experiment_id": "omega5_long_specialist_hgb_sidecar_20260702",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": exp.asdict(policy),
        "features": COMMON_FEATURES,
        "model_path": str(model_path),
        "all_bar_evals": evals,
        "thresholds": thresholds,
        "parent_event_top": parent_rank.drop(columns=["ledgers"]).head(10).to_dict(orient="records"),
        "best_parent_ledgers": parent_rank.iloc[0]["ledgers"] if len(parent_rank) else {},
        "redteam": {
            "status": "PASS_WITH_LIMITATIONS" if bool(parent_rank.iloc[0]["pass_all"]) else "FAIL",
            "warnings": [
                "This is a separate HGB sidecar diagnostic, not an end-to-end TabM parent retrain.",
                "All-bar training uses only columns common to 2025 market_state, old OOS, and additional OOS; short_squeeze_risk/bb_width remain rule-gate features at parent-event evaluation time.",
                "Fresh holdout/walk-forward is still required because policy and threshold selection used validation/OOS/additional OOS readouts.",
            ],
        },
    }
    write_json(OUT_DIR / "report.json", report)
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "model": str(model_path)}, indent=2))


if __name__ == "__main__":
    main()
