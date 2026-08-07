#!/usr/bin/env python3
"""Native supervised portfolio candidate ranker for ETH/SOL/BTC.

The ranker receives all available asset candidates at a timestamp and selects
the candidate with the highest predicted risk-adjusted return. Training uses
validation-only native counterfactual outcomes. OOS is evaluated once after the
model and threshold are frozen.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

import replay_portfolio_rl_gate_2action_native_20260708 as native

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/causal_regen_20260516/portfolio_supervised_ranker_native_20260709"
DOC_PATH = ROOT / "docs/model_contracts/portfolio_supervised_ranker_native_20260709.md"
ASSETS = ("eth", "sol", "btc")


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


FEATURE_COLS = [
    "asset_eth", "asset_sol", "asset_btc", "side_long", "side_short",
    "notional", "margin_fraction", "leverage", "take_profit", "stop_loss",
    "ou_halflife", "asset_score", "hour_sin", "hour_cos", "month_norm",
]


def _flat_decision_candidate_rows(world: dict[str, Any], device: Any) -> list[tuple[pd.Timestamp, native.Candidate]]:
    """Collect candidates only at timestamps a native rule account is flat.

    This avoids training on counterfactual timestamps where the portfolio would
    already be in a position and could not act.
    """
    rows: list[tuple[pd.Timestamp, native.Candidate]] = []
    position: native.Position | None = None
    cash = 1.0
    for ts in world["timestamps"]:
        if position is not None:
            position, cash, _closed, _mark = native._try_close(world, position, ts, cash, device)
            continue
        candidates = [native._candidate_for_asset(world, asset, ts) for asset in ASSETS]
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            continue
        rows.extend((ts, c) for c in candidates)
        candidates.sort(key=lambda c: (native.ASSET_SCORES[c.asset], c.notional), reverse=True)
        position, cash = native._open_position(world, candidates[0], cash)
    return rows


def _features(world: dict[str, Any], c: native.Candidate, ts: pd.Timestamp) -> dict[str, float]:
    return {
        "asset_eth": float(c.asset == "eth"),
        "asset_sol": float(c.asset == "sol"),
        "asset_btc": float(c.asset == "btc"),
        "side_long": float(c.side > 0),
        "side_short": float(c.side < 0),
        "notional": float(c.notional),
        "margin_fraction": float(c.margin),
        "leverage": float(c.leverage),
        "take_profit": float(c.take_profit),
        "stop_loss": float(c.stop_loss),
        "ou_halflife": float(world[c.asset]["frame"]["ou_halflife"].iloc[c.local_i]),
        "asset_score": float(native.ASSET_SCORES[c.asset]),
        "hour_sin": float(np.sin(2 * np.pi * ts.hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * ts.hour / 24.0)),
        "month_norm": float((ts.month - 6.5) / 6.0),
    }


def _simulate_candidate(world: dict[str, Any], c: native.Candidate, device: Any) -> dict[str, Any]:
    pos, cash = native._open_position(world, c, 1.0)
    asset_frame = world[c.asset]["frame"]
    start_idx = c.local_i
    closed_row: dict[str, Any] | None = None
    for ts in asset_frame["timestamp"].iloc[start_idx + 1 :]:
        pos, cash, closed, _mark = native._try_close(world, pos, pd.Timestamp(ts), cash, device)
        if closed is not None:
            closed_row = closed
            break
    if closed_row is None and pos is not None:
        cash, closed_row = native._force_close(world, pos, cash)
    return closed_row


def _build_dataset(world: dict[str, Any], device: Any, *, max_rows: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidates = _flat_decision_candidate_rows(world, device)
    if max_rows is not None:
        candidates = candidates[: int(max_rows)]
    for idx, (ts, c) in enumerate(candidates):
        if idx % 25 == 0:
            print(f"stage=build_dataset idx={idx}/{len(candidates)}", flush=True)
        closed = _simulate_candidate(world, c, device)
        ret = float(closed["trade_return"])
        mae = float(closed.get("mae_price_move", 0.0) or 0.0)
        hold_bars = int(closed["exit_i"]) - int(closed["entry_i"])
        label = ret - 0.20 * max(0.0, -mae - 0.02) - 0.00001 * max(hold_bars, 0)
        rows.append(
            {
                "timestamp": ts,
                "asset": c.asset,
                "component": c.component,
                "label": float(label),
                "trade_return": ret,
                "mae_price_move": mae,
                "hold_bars": int(hold_bars),
                **_features(world, c, ts),
            }
        )
    return pd.DataFrame(rows)


def _train_model(train_df: pd.DataFrame) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=60709,
        verbose=-1,
    )
    model.fit(train_df[FEATURE_COLS], train_df["label"])
    return model


def _replay_ranker(world: dict[str, Any], model: lgb.LGBMRegressor, *, threshold: float, device: Any) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    position: native.Position | None = None
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for ts in world["timestamps"]:
        if position is not None:
            position, cash, closed, mark_equity = native._try_close(world, position, ts, cash, device)
            peak = max(peak, mark_equity)
            mdd = min(mdd, mark_equity / max(peak, 1e-12) - 1.0)
            if closed is not None:
                rows.append(closed)
                peak = max(peak, cash)
                mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
            continue
        candidates = [native._candidate_for_asset(world, asset, ts) for asset in ASSETS]
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            continue
        feat_df = pd.DataFrame([_features(world, c, ts) for c in candidates])
        scores = model.predict(feat_df[FEATURE_COLS])
        best_i = int(np.argmax(scores))
        best_score = float(scores[best_i])
        decisions.append(
            {
                "timestamp": ts,
                "selected_asset": candidates[best_i].asset if best_score >= threshold else "cash",
                "selected_score": best_score,
                "threshold": float(threshold),
                **{f"score_{c.asset}": float(s) for c, s in zip(candidates, scores)},
            }
        )
        if best_score < threshold:
            continue
        position, cash = native._open_position(world, candidates[best_i], cash)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    if position is not None:
        cash, closed = native._force_close(world, position, cash)
        rows.append(closed)
    ledger = pd.DataFrame(rows)
    metrics = native._compound_metrics(ledger)
    metrics["mark_to_market_mdd"] = float(mdd * 100.0)
    metrics["decisions"] = int(len(decisions))
    metrics["cash_decisions"] = int(sum(d["selected_asset"] == "cash" for d in decisions))
    return metrics, ledger, pd.DataFrame(decisions)


def _score(metrics: dict[str, Any]) -> float:
    return float(metrics["pnl"]) - 0.30 * abs(float(metrics["mdd"])) - 10.0 * max(0.0, metrics["cash_decisions"] / max(metrics["decisions"], 1) - 0.5)


def _write_doc(report: dict[str, Any]) -> None:
    lines = [
        "# Portfolio Supervised Ranker Native - 2026-07-09",
        "",
        "LightGBM supervised candidate ranker. Training labels are validation-only native counterfactual risk-adjusted trade outcomes.",
        "",
        f"Selected threshold: `{report['selected_threshold']}`",
        "",
        "| split | PnL | MDD | MTM MDD | trades | WR | decisions | cash |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "oos_extended", "oos_frozen_q1_2026"):
        m = report["results"][split]
        lines.append(f"| {split} | {m['pnl']:.2f}% | {m['mdd']:.2f}% | {m['mark_to_market_mdd']:.2f}% | {m['trades']} | {m['wr']:.2%} | {m.get('decisions', 0)} | {m.get('cash_decisions', 0)} |")
    lines.append("")
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = native.eth_retest.DEVICE
    print("stage=build_validation_world", flush=True)
    val_world = native._build_world("validation", device)
    print("stage=build_validation_dataset", flush=True)
    train_df = _build_dataset(val_world, device)
    train_df.to_csv(OUT_DIR / "validation_candidate_training_set.csv", index=False)
    print(f"stage=train_model rows={len(train_df)}", flush=True)
    model = _train_model(train_df)
    with open(OUT_DIR / "ranker_lgbm.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS}, f)

    threshold_grid = [-0.10, -0.05, -0.02, 0.0, 0.01, 0.02, 0.04]
    val_rows: list[dict[str, Any]] = []
    best_threshold: float | None = None
    best_score = -np.inf
    for th in threshold_grid:
        metrics, ledger, decisions = _replay_ranker(val_world, model, threshold=float(th), device=device)
        eligible = metrics["trades"] >= 15 and metrics["mdd"] >= -25.0
        score = _score(metrics) if eligible else -np.inf
        val_rows.append({"threshold": float(th), "metrics": metrics, "eligible": bool(eligible), "score": float(score)})
        if eligible and score > best_score:
            best_score = float(score)
            best_threshold = float(th)
    if best_threshold is None:
        best_threshold = -0.10
    pd.DataFrame(val_rows).to_json(OUT_DIR / "validation_threshold_grid.jsonl", orient="records", lines=True, force_ascii=False)
    val_metrics, val_ledger, val_decisions = _replay_ranker(val_world, model, threshold=best_threshold, device=device)
    print("stage=build_oos_world", flush=True)
    oos_world = native._build_world("oos", device)
    oos_metrics, oos_ledger, oos_decisions = _replay_ranker(oos_world, model, threshold=best_threshold, device=device)
    q1 = oos_ledger.loc[pd.to_datetime(oos_ledger["entry_timestamp"]) < pd.Timestamp("2026-04-01")].reset_index(drop=True) if not oos_ledger.empty else oos_ledger
    q1_metrics = native._compound_metrics(q1)
    q1_metrics["mark_to_market_mdd"] = q1_metrics["mdd"]
    q1_metrics["decisions"] = int((oos_decisions["timestamp"] < pd.Timestamp("2026-04-01")).sum()) if not oos_decisions.empty else 0
    q1_metrics["cash_decisions"] = int(((oos_decisions["timestamp"] < pd.Timestamp("2026-04-01")) & (oos_decisions["selected_asset"] == "cash")).sum()) if not oos_decisions.empty else 0

    val_ledger.to_csv(OUT_DIR / "validation_ledger.csv", index=False)
    val_decisions.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    oos_ledger.to_csv(OUT_DIR / "oos_extended_ledger.csv", index=False)
    oos_decisions.to_csv(OUT_DIR / "oos_extended_decisions.csv", index=False)
    q1.to_csv(OUT_DIR / "oos_frozen_q1_2026_ledger.csv", index=False)
    report = {
        "method": "portfolio_supervised_ranker_native_lgbm",
        "training_data": "validation_native_counterfactual_candidate_outcomes",
        "oos_usage": "reported_once_after_model_and_threshold_selection",
        "feature_cols": FEATURE_COLS,
        "selected_threshold": best_threshold,
        "threshold_grid": val_rows,
        "results": {"validation": val_metrics, "oos_extended": oos_metrics, "oos_frozen_q1_2026": q1_metrics},
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "promotion_grade": False,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_doc(report)
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "doc": str(DOC_PATH), "selected_threshold": best_threshold, "results": report["results"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
