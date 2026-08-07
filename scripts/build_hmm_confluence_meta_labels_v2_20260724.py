#!/usr/bin/env python3
"""Build and select causal ETH HMM trend-pullback meta-labels, version 2.

Selection is restricted to 2025 train/validation. The selected policy is frozen
before 2026 OOS/fresh candidates are generated. OOS is never used by the search.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_hmm_confluence_meta_labels_20260724 as v1  # noqa: E402


MODEL_ID = "eth_hmm_confluence_meta_labels_v2_20260724"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
MIN_TRAIN_LABELS = 180
MIN_VALIDATION_LABELS = 100
ROUND_TRIP_COST = 2.0 * (v1.FEE_RATE + v1.SLIPPAGE_RATE)


@dataclass(frozen=True)
class V2Params:
    persistence_window: int
    persistence_count: int
    mean_probability_min: float
    pullback_lookback: int
    stop_atr_buffer: float
    reward_r: float
    transition_entry_max: float
    structural_clearance_r: float
    horizon_bars: int
    transition_exit_threshold: float | None = None


def parameter_grid() -> list[V2Params]:
    persistence = [(6, 3, 0.55), (6, 4, 0.60), (8, 5, 0.60)]
    return [
        V2Params(window, count, probability, lookback, stop_buffer, reward_r, transition_max, clearance, horizon)
        for (window, count, probability), lookback, stop_buffer, reward_r, transition_max, clearance, horizon in itertools.product(
            persistence,
            (6, 12),
            (0.15, 0.30),
            (1.25, 1.50, 1.75),
            (0.45, 0.60),
            (0.0, 0.75),
            (72, 96),
        )
    ]


def _rolling_true(values: pd.Series, window: int, minimum: int) -> np.ndarray:
    return values.astype(float).rolling(window, min_periods=window).sum().ge(minimum).to_numpy(bool)


def _nearest_clearance(
    frame: pd.DataFrame,
    index: int,
    side: int,
    decision_price: float,
    risk: float,
) -> float:
    if side > 0:
        levels = (
            frame["context_vpvr_vah"].iloc[index],
            frame["context_swing_high24"].iloc[index],
            frame["context_swing_high48"].iloc[index],
        )
    else:
        levels = (
            frame["context_vpvr_val"].iloc[index],
            frame["context_swing_low24"].iloc[index],
            frame["context_swing_low48"].iloc[index],
        )
    favorable = [
        side * (float(level) - decision_price) / risk
        for level in levels
        if np.isfinite(level) and side * (float(level) - decision_price) > 0.0
    ]
    return float(min(favorable)) if favorable else float("inf")


def build_candidates(frame: pd.DataFrame, params: V2Params) -> pd.DataFrame:
    close = frame["close"].to_numpy(float)
    open_price = frame["open"].to_numpy(float)
    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    atr = frame["context_atr192"].to_numpy(float)
    vwma_fast = frame["context_vwma100"].to_numpy(float)
    vwma_slow = frame["context_vwma288"].to_numpy(float)
    poc = frame["context_vpvr_poc"].to_numpy(float)
    route = frame["context_regime_route"].astype(str)
    bull_probability = frame[f"{v1.HMM_PREFIX}bull_prob"].astype(float)
    bear_probability = frame[f"{v1.HMM_PREFIX}bear_prob"].astype(float)

    window = params.persistence_window
    bull_persistent = _rolling_true(route.eq("bull"), window, params.persistence_count)
    bear_persistent = _rolling_true(route.eq("bear"), window, params.persistence_count)
    bull_mean_ok = bull_probability.rolling(window, min_periods=window).mean().ge(params.mean_probability_min).to_numpy(bool)
    bear_mean_ok = bear_probability.rolling(window, min_periods=window).mean().ge(params.mean_probability_min).to_numpy(bool)
    slow_slope = pd.Series(vwma_slow).diff(12).to_numpy(float)

    long_anchor = np.fmax(vwma_fast, poc)
    short_anchor = np.fmin(vwma_fast, poc)
    long_touch = (low <= long_anchor + 0.25 * atr) & (low >= long_anchor - 0.75 * atr)
    short_touch = (high >= short_anchor - 0.25 * atr) & (high <= short_anchor + 0.75 * atr)
    long_pullback = pd.Series(long_touch).rolling(params.pullback_lookback, min_periods=params.pullback_lookback).max().fillna(0).to_numpy(bool)
    short_pullback = pd.Series(short_touch).rolling(params.pullback_lookback, min_periods=params.pullback_lookback).max().fillna(0).to_numpy(bool)

    candle_range = np.maximum(high - low, 1.0e-12)
    long_reclaim = (close > long_anchor) & (close > open_price) & ((close - low) / candle_range >= 0.60)
    short_reclaim = (close < short_anchor) & (close < open_price) & ((high - close) / candle_range >= 0.60)
    transition_ok = frame["regime3_transition_h6_risk_prob"].to_numpy(float) <= params.transition_entry_max
    valid = (
        frame["context_vpvr_valid"].to_numpy(bool)
        & np.isfinite(atr)
        & (atr > 0.0)
        & np.isfinite(long_anchor)
        & np.isfinite(short_anchor)
        & np.isfinite(slow_slope)
    )
    long_setup = valid & bull_persistent & bull_mean_ok & (slow_slope > 0.0) & long_pullback & long_reclaim & transition_ok
    short_setup = valid & bear_persistent & bear_mean_ok & (slow_slope < 0.0) & short_pullback & short_reclaim & transition_ok
    long_setup &= ~pd.Series(long_setup).shift(1, fill_value=False).to_numpy(bool)
    short_setup &= ~pd.Series(short_setup).shift(1, fill_value=False).to_numpy(bool)

    rolling_low = pd.Series(low).rolling(params.pullback_lookback, min_periods=params.pullback_lookback).min().to_numpy(float)
    rolling_high = pd.Series(high).rolling(params.pullback_lookback, min_periods=params.pullback_lookback).max().to_numpy(float)
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(long_setup | short_setup):
        if long_setup[index] and short_setup[index]:
            continue
        side = 1 if long_setup[index] else -1
        decision_price = float(close[index])
        stop = (
            float(rolling_low[index] - params.stop_atr_buffer * atr[index])
            if side > 0
            else float(rolling_high[index] + params.stop_atr_buffer * atr[index])
        )
        risk = side * (decision_price - stop)
        if not np.isfinite(risk) or risk <= 0.0:
            continue
        risk_return = risk / decision_price
        if risk_return < 4.0 * ROUND_TRIP_COST:
            continue
        clearance_r = _nearest_clearance(frame, index, side, decision_price, risk)
        if clearance_r < params.structural_clearance_r:
            continue
        target = decision_price + side * params.reward_r * risk
        rows.append(
            {
                "decision_index": int(index),
                "decision_timestamp": frame["timestamp"].iloc[index],
                "setup_family": "sequential_trend_pullback",
                "candidate_side": side,
                "candidate_side_name": "LONG" if side > 0 else "SHORT",
                "horizon_bars": params.horizon_bars,
                "planned_target_price": target,
                "planned_stop_price": stop,
                "planned_tp_price_move": params.reward_r * risk_return,
                "planned_sl_price_move": risk_return,
                "planned_rr": params.reward_r,
                "context_regime_route": "bull" if side > 0 else "bear",
                "context_regime_confidence": float(frame[f"{v1.HMM_PREFIX}confidence"].iloc[index]),
                "context_regime_margin": float(frame[f"{v1.HMM_PREFIX}margin"].iloc[index]),
                "context_regime_entropy": float(frame[f"{v1.HMM_PREFIX}entropy"].iloc[index]),
                "context_regime_mean_probability": float(
                    bull_probability.iloc[index - window + 1 : index + 1].mean()
                    if side > 0
                    else bear_probability.iloc[index - window + 1 : index + 1].mean()
                ),
                "context_regime_persistence": int(
                    route.iloc[index - window + 1 : index + 1].eq("bull" if side > 0 else "bear").sum()
                ),
                "context_transition_risk": float(frame["regime3_transition_h6_risk_prob"].iloc[index]),
                "context_churn_risk": float(frame["regime3_churn_h6_risk_score"].iloc[index]),
                "context_sample_weight": float(frame["context_regime_sample_weight"].iloc[index]),
                "context_rsi": float(frame["rsi"].iloc[index]),
                "context_vwma100": float(vwma_fast[index]),
                "context_vwma288": float(vwma_slow[index]),
                "context_vwma288_slope12": float(slow_slope[index]),
                "context_vpvr_poc": float(poc[index]),
                "context_vpvr_vah": float(frame["context_vpvr_vah"].iloc[index]),
                "context_vpvr_val": float(frame["context_vpvr_val"].iloc[index]),
                "context_atr192": float(atr[index]),
                "context_structural_clearance_r": clearance_r,
                "context_volume_confirm": float(frame["sig_volume_confirm"].iloc[index]),
                "context_oi_change_rate": float(frame["oi_change_rate"].iloc[index]),
                "context_volume_imbalance": float(frame["cvp_volume_imbalance"].iloc[index]),
                "context_funding_z": float(frame["funding_z_score"].iloc[index]),
            }
        )
    return pd.DataFrame(rows)


def _invalid(split: str, reason: str) -> dict[str, Any]:
    return {"split": split, "label_valid": 0, "label_invalid_reason": reason}


def simulate_candidate(
    frame: pd.DataFrame,
    row: pd.Series,
    tape: v1.FundingTape,
    *,
    transition_exit_threshold: float | None,
) -> dict[str, Any]:
    index = int(row["decision_index"])
    side = int(row["candidate_side"])
    entry_index = index + 1
    horizon = int(row["horizon_bars"])
    split, split_end = v1.split_contract(pd.Timestamp(row["decision_timestamp"]))
    if entry_index + horizon >= len(frame):
        return _invalid(split, "right_censored")
    timeout_index = entry_index + horizon
    if frame["timestamp"].iloc[timeout_index] > split_end:
        return _invalid(split, "split_boundary_censored")

    open_price = frame["open"].to_numpy(float)
    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    transition_risk = frame["regime3_transition_h6_risk_prob"].to_numpy(float)
    target = float(row["planned_target_price"])
    stop = float(row["planned_stop_price"])
    entry_fill = float(open_price[entry_index] * (1.0 + side * v1.SLIPPAGE_RATE))
    if side * (target / entry_fill - 1.0) <= 0.0 or -side * (stop / entry_fill - 1.0) <= 0.0:
        return _invalid(split, "entry_gap_invalid")

    outcome = "TIMEOUT"
    exit_index = timeout_index
    exit_level = float(open_price[timeout_index])
    exit_at_open = True
    pending_regime_exit = False
    mfe = 0.0
    mae = 0.0
    for bar in range(entry_index, timeout_index):
        open_stop = open_price[bar] <= stop if side > 0 else open_price[bar] >= stop
        open_target = open_price[bar] >= target if side > 0 else open_price[bar] <= target
        if open_stop:
            outcome, exit_index, exit_level, exit_at_open = "SL", bar, float(open_price[bar]), True
            break
        if open_target:
            outcome, exit_index, exit_level, exit_at_open = "TP", bar, float(open_price[bar]), True
            break
        if pending_regime_exit:
            outcome, exit_index, exit_level, exit_at_open = "REGIME_EXIT", bar, float(open_price[bar]), True
            break

        favorable = high[bar] / entry_fill - 1.0 if side > 0 else 1.0 - low[bar] / entry_fill
        adverse = low[bar] / entry_fill - 1.0 if side > 0 else 1.0 - high[bar] / entry_fill
        mfe = max(mfe, float(favorable))
        mae = min(mae, float(adverse))
        hit_stop = low[bar] <= stop if side > 0 else high[bar] >= stop
        hit_target = high[bar] >= target if side > 0 else low[bar] <= target
        if hit_stop and hit_target:
            outcome, exit_index, exit_level, exit_at_open = "AMBIGUOUS", bar, stop, False
            break
        if hit_stop:
            outcome, exit_index, exit_level, exit_at_open = "SL", bar, stop, False
            break
        if hit_target:
            outcome, exit_index, exit_level, exit_at_open = "TP", bar, target, False
            break
        if transition_exit_threshold is not None and transition_risk[bar] > transition_exit_threshold:
            pending_regime_exit = True

    exit_fill = float(exit_level * (1.0 - side * v1.SLIPPAGE_RATE))
    gross = float(side * (exit_fill / entry_fill - 1.0))
    entry_ns = int(pd.Timestamp(frame["timestamp"].iloc[entry_index]).value)
    exit_timestamp = frame["timestamp"].iloc[exit_index] if exit_at_open else frame["timestamp"].iloc[exit_index] + pd.Timedelta(minutes=v1.BAR_MINUTES)
    exit_ns = int(pd.Timestamp(exit_timestamp).value)
    funding = v1.funding_return(tape, entry_ns, exit_ns, entry_fill, side)
    net = gross - v1.FEE_RATE - v1.FEE_RATE * exit_fill / entry_fill + funding
    initial_risk_return = abs(entry_fill - stop) / entry_fill
    net_r = net / max(initial_risk_return, 1.0e-12)
    mfe_r = mfe / max(initial_risk_return, 1.0e-12)
    mae_r = mae / max(initial_risk_return, 1.0e-12)
    if outcome == "TP" and net_r >= 1.0:
        label_class = "positive"
    elif outcome in {"TIMEOUT", "REGIME_EXIT"} and net_r > -0.25:
        label_class = "neutral"
    else:
        label_class = "negative"
    return {
        "split": split,
        "entry_index": entry_index,
        "entry_timestamp": frame["timestamp"].iloc[entry_index],
        "entry_fill_price": entry_fill,
        "event_end_index": exit_index,
        "event_end_timestamp": exit_timestamp,
        "exit_fill_price": exit_fill,
        "label_valid": int(outcome != "AMBIGUOUS"),
        "label_invalid_reason": "same_bar_tp_sl" if outcome == "AMBIGUOUS" else "",
        "label_outcome": outcome,
        "label_success": int(outcome == "TP" and net > 0.0),
        "label_class": label_class,
        "label_net_return_per_notional": net,
        "label_gross_return_per_notional": gross,
        "label_funding_return_per_notional": funding,
        "label_net_r": net_r,
        "label_mfe_price_move": mfe,
        "label_mae_price_move": mae,
        "label_mfe_r": mfe_r,
        "label_mae_r": mae_r,
        "label_path_quality": mfe_r + 0.5 * mae_r,
        "label_bars_to_exit": max(exit_index - entry_index + (not exit_at_open), 0),
    }


def label_candidates(
    frame: pd.DataFrame,
    candidates: pd.DataFrame,
    tape: v1.FundingTape,
    *,
    transition_exit_threshold: float | None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    outcomes = [
        simulate_candidate(frame, row, tape, transition_exit_threshold=transition_exit_threshold)
        for _, row in candidates.iterrows()
    ]
    return pd.concat([candidates.reset_index(drop=True), pd.DataFrame(outcomes)], axis=1)


def _split_metrics(labels: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = labels.loc[(labels["split"] == split) & (labels["label_valid"] == 1)]
    trades = v1.replay_non_overlapping(selected, cooldown_bars=6)
    returns = trades["label_net_return_per_notional"].to_numpy(float) if len(trades) else np.empty(0)
    return {
        "labels": int(len(selected)),
        "success_rate": float(selected["label_success"].mean()) if len(selected) else 0.0,
        "mean_net_return": float(selected["label_net_return_per_notional"].mean()) if len(selected) else -9.0,
        "policy_trades": int(len(trades)),
        "policy_compounded_return": float(np.prod(1.0 + returns) - 1.0) if len(returns) else -9.0,
    }


def search_parameters(frame: pd.DataFrame, tape: v1.FundingTape) -> tuple[V2Params, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    for params in parameter_grid():
        candidates = build_candidates(frame, params)
        labels = label_candidates(frame, candidates, tape, transition_exit_threshold=None)
        train = _split_metrics(labels, "train")
        validation = _split_metrics(labels, "validation")
        eligible = (
            train["labels"] >= MIN_TRAIN_LABELS
            and validation["labels"] >= MIN_VALIDATION_LABELS
            and train["mean_net_return"] > 0.0
            and validation["mean_net_return"] > 0.0
            and train["policy_compounded_return"] > 0.0
            and validation["policy_compounded_return"] > 0.0
        )
        robust_mean = min(train["mean_net_return"], validation["mean_net_return"])
        results.append(
            {
                "params": asdict(params),
                "train": train,
                "validation": validation,
                "eligible": eligible,
                "selection_score": robust_mean if eligible else -9.0,
            }
        )
    eligible_results = [row for row in results if row["eligible"]]
    if not eligible_results:
        raise RuntimeError("no V2 parameter set passed the train/validation profitability contract")
    winner = max(
        eligible_results,
        key=lambda row: (
            row["selection_score"],
            min(row["train"]["labels"], row["validation"]["labels"]),
        ),
    )
    selected = V2Params(**winner["params"])

    base_candidates = build_candidates(frame, selected)
    exit_results: list[dict[str, Any]] = []
    for threshold in (None, 0.65, 0.75):
        labels = label_candidates(frame, base_candidates, tape, transition_exit_threshold=threshold)
        train = _split_metrics(labels, "train")
        validation = _split_metrics(labels, "validation")
        eligible = (
            train["mean_net_return"] > 0.0
            and validation["mean_net_return"] > 0.0
            and train["policy_compounded_return"] > 0.0
            and validation["policy_compounded_return"] > 0.0
        )
        exit_results.append(
            {
                "transition_exit_threshold": threshold,
                "train": train,
                "validation": validation,
                "eligible": eligible,
                "selection_score": min(train["mean_net_return"], validation["mean_net_return"]) if eligible else -9.0,
            }
        )
    exit_winner = max(exit_results, key=lambda row: row["selection_score"])
    selected = V2Params(**{**asdict(selected), "transition_exit_threshold": exit_winner["transition_exit_threshold"]})
    results.append({"stage": "transition_exit_selection", "candidates": exit_results, "selected": asdict(selected)})
    return selected, results


def plot_chart(context: pd.DataFrame, trades: pd.DataFrame, path: Path) -> dict[str, Any]:
    start, end = v1.choose_chart_window(trades)
    view = context.loc[context["timestamp"].between(start, end)]
    shown = trades.loc[pd.to_datetime(trades["decision_timestamp"]).between(start, end)].copy()
    fig, (axis, equity_axis) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    axis.plot(view["timestamp"], view["close"], color="#334155", linewidth=1.0, label="ETH close")
    axis.plot(view["timestamp"], view["context_vwma100"], color="#2563eb", linewidth=0.9, label="VWMA100")
    for _, trade in shown.iterrows():
        won = float(trade["label_net_return_per_notional"]) > 0.0
        color = "#16a34a" if won else "#dc2626"
        marker = "^" if int(trade["candidate_side"]) > 0 else "v"
        axis.scatter(pd.Timestamp(trade["entry_timestamp"]), trade["entry_fill_price"], marker=marker, s=65, color=color, zorder=5)
        axis.scatter(pd.Timestamp(trade["event_end_timestamp"]), trade["exit_fill_price"], marker="x", s=50, color=color, zorder=5)
        axis.plot(
            [pd.Timestamp(trade["entry_timestamp"]), pd.Timestamp(trade["event_end_timestamp"])],
            [trade["entry_fill_price"], trade["exit_fill_price"]],
            color=color,
            linewidth=1.0,
            alpha=0.65,
        )
    if len(shown):
        shown = shown.sort_values("event_end_timestamp")
        equity = (1.0 + shown["label_net_return_per_notional"].astype(float)).cumprod()
        equity_axis.step(pd.to_datetime(shown["event_end_timestamp"]), equity, where="post", color="#0f766e", linewidth=1.5)
    axis.set_ylabel("ETHUSDT price")
    equity_axis.set_ylabel("Equity")
    equity_axis.set_xlabel("UTC")
    axis.set_title(f"V2 sequential HMM pullback trades: {start.date()} to {end.date()}")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.15)
    equity_axis.grid(alpha=0.15)
    equity_axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"start": str(start), "end": str(end), "trades": int(len(shown))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hmm_manifest = v1.validate_hmm_artifact(v1.HMM_ARTIFACT)
    frame_2025 = v1.load_frame(v1.MARKET_2025, v1.HMM_2025, v1.RISK_ARTIFACT)
    thresholds = v1.derive_route_thresholds(frame_2025)
    context_2025 = v1.append_causal_context(frame_2025, thresholds)
    tape_2025, funding_hashes_2025 = v1.load_funding_tape(context_2025)
    print(json.dumps({"stage": "train_validation_search", "grid_size": len(parameter_grid())}), flush=True)
    selected, search_results = search_parameters(context_2025, tape_2025)
    print(json.dumps({"stage": "policy_locked_before_oos", "selected": asdict(selected)}), flush=True)

    candidates_2025 = build_candidates(context_2025, selected)
    labels_2025 = label_candidates(
        context_2025,
        candidates_2025,
        tape_2025,
        transition_exit_threshold=selected.transition_exit_threshold,
    )
    labels_2025["source_year"] = 2025

    frame_2026 = v1.load_frame(v1.MARKET_2026, v1.HMM_2026, v1.RISK_ARTIFACT)
    context_2026 = v1.append_causal_context(frame_2026, thresholds)
    tape_2026, funding_hashes_2026 = v1.load_funding_tape(context_2026)
    candidates_2026 = build_candidates(context_2026, selected)
    labels_2026 = label_candidates(
        context_2026,
        candidates_2026,
        tape_2026,
        transition_exit_threshold=selected.transition_exit_threshold,
    )
    labels_2026["source_year"] = 2026

    labels = pd.concat([labels_2025, labels_2026], ignore_index=True).sort_values("decision_timestamp").reset_index(drop=True)
    context = pd.concat([context_2025, context_2026], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    trades = v1.replay_non_overlapping(labels)
    artifacts = v1._write_split_artifacts(labels, trades, args.out_dir)
    chart_trades = trades.loc[trades["split"] == "oos"]
    chart_path = args.out_dir / "oos_trade_chart.png"
    chart = plot_chart(context, chart_trades, chart_path)

    search_path = args.out_dir / "train_validation_search.json"
    search_path.write_text(json.dumps(search_results, indent=2, default=v1._json_default), encoding="utf-8")
    report = {
        "model_id": MODEL_ID,
        "status": "research_labels_generated_policy_locked_before_oos",
        "asset": "ETHUSDT",
        "bar_minutes": v1.BAR_MINUTES,
        "hmm": hmm_manifest,
        "route_thresholds": asdict(thresholds),
        "selected_params": asdict(selected),
        "selection": {
            "train_end": "2025-08-31 23:55:00",
            "validation": ["2025-09-01", "2025-12-31 23:55:00"],
            "oos_used_for_selection": False,
            "search_results": str(search_path),
            "minimum_train_labels": MIN_TRAIN_LABELS,
            "minimum_validation_labels": MIN_VALIDATION_LABELS,
        },
        "label_contract": {
            "entry": "next_bar_open_with_adverse_slippage",
            "outcomes": ["TP", "SL", "TIMEOUT", "REGIME_EXIT", "AMBIGUOUS"],
            "same_bar_tp_sl": "AMBIGUOUS_and_label_valid_0",
            "multi_labels": ["label_class", "label_net_r", "label_path_quality"],
            "label_class_positive": "TP_and_net_r_gte_1",
            "label_class_neutral": "TIMEOUT_or_REGIME_EXIT_and_net_r_gt_minus_0.25",
        },
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "stored_trade_ledger_is_diagnostic_only": True,
        "funding_hashes": {**funding_hashes_2025, **funding_hashes_2026},
        "artifacts": artifacts,
        "chart": {"path": str(chart_path), **chart},
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=v1._json_default), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "selected": asdict(selected),
                "summaries": {key: value["summary"] for key, value in artifacts.items()},
            },
            default=v1._json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
