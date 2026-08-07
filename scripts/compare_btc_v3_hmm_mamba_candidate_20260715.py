#!/usr/bin/env python3
"""BTC v3 research candidate comparison: HMM entry-score gate + Mamba exit-hazard vs Stage 1 baseline.

Rebuilds an alternative event set using the two new candidate models
(build_btc_v3_hmm_entry_score_20260715.py, build_btc_v3_mamba_exit_hazard_20260715.py):
  - entry: same ts_action transition timing as Stage 1, but gated by the causal HMM
    win-probability entry_score instead of Stage 1's raw ts_action-only gate.
  - exit: whichever of {existing ATR stop/trail/time-exit, first bar where the Mamba exit-hazard
    crosses 0.5} triggers first.

Reuses _exit_fill/STOP_ATR_PRICE/TRAIL_ATR_PRICE/ARM_ATR_PRICE/MAX_HOLD_BARS/fee-slippage constants
from train_eval_btc_v2_regime_trendscan_20260714.py unmodified, so the cost model is identical to
Stage 1's own simulation -- only the entry gate and exit-hazard overlay are new.

This is a research readout, not a promotion gate: it only compares against Stage 1's own realized
sparse-event numbers on the same >= VAL_START (2025-10-01) window, and writes
docs/model_contracts/btc_v3_hmm_mamba_candidate_20260715.md summarizing the result. Does not touch
ts_action, the Stage 1 sparse_event_dataset, or any live-wired code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.retrain_clean_regime_hmm_20260517 import _json_default  # noqa: E402
import train_eval_btc_v2_regime_trendscan_20260714 as btc_v2  # noqa: E402
from build_btc_v3_mamba_exit_hazard_20260715 import (  # noqa: E402
    CryptoMambaExitHazard,
    FEATURE_NAMES,
    HORIZON,
    D_MODEL,
    D_STATE,
    N_CBLOCKS,
    N_CMBLOCKS,
    DROPOUT,
    SEQ_LEN,
    _in_trade_features,
)

CANDIDATE_DIR = ROOT / "tmp/causal_regen_20260516/btc_v3_hmm_mamba_candidate_20260715"
SPARSE_EVENTS_PATH = ROOT / "tmp/causal_regen_20260516/btc_v3_sparse_event_dataset_20260714/sparse_event_dataset.parquet"
HOLDOUT_START = pd.Timestamp("2026-07-14 00:00:00")
VAL_START = pd.Timestamp("2025-10-01")
HAZARD_THRESHOLD = 0.5
ENTRY_SCORE_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def _load_entry_score() -> pd.DataFrame:
    path = CANDIDATE_DIR / "btc_v3_hmm_entry_score.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run build_btc_v3_hmm_entry_score_20260715.py first")
    return pd.read_parquet(path)


def _load_hazard_model() -> tuple[CryptoMambaExitHazard, np.ndarray, np.ndarray, torch.device]:
    path = CANDIDATE_DIR / "btc_v3_mamba_exit_hazard_20260715.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run build_btc_v3_mamba_exit_hazard_20260715.py first")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CryptoMambaExitHazard(len(FEATURE_NAMES), SEQ_LEN, D_MODEL, N_CBLOCKS, N_CMBLOCKS, D_STATE, DROPOUT).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, np.asarray(ckpt["scaler_mean"]), np.asarray(ckpt["scaler_scale"]), device


@torch.no_grad()
def _hazard_prob_at(model: CryptoMambaExitHazard, feats_window: np.ndarray, mean: np.ndarray, scale: np.ndarray, device: torch.device) -> float:
    x = (feats_window - mean) / scale
    x = np.nan_to_num(x).astype(np.float32)
    xb = torch.from_numpy(x[None, :, :]).to(device)
    return float(torch.sigmoid(model(xb)).item())


def _candidate_side(hourly: pd.DataFrame, entry_score_by_ts: dict, threshold: float) -> pd.Series:
    action = hourly["ts_action"].to_numpy()
    is_event = (action != 0) & (action != np.roll(action, 1))
    is_event[0] = bool(action[0] != 0)
    score = hourly["timestamp"].map(entry_score_by_ts).fillna(0.0).to_numpy(dtype=np.float64)
    side = np.zeros(len(hourly), dtype=np.int8)
    long_ok = is_event & (action == 1) & (score >= threshold)
    short_ok = is_event & (action == 2) & (score >= threshold)
    side[long_ok] = 1
    side[short_ok] = -1
    return pd.Series(side, index=hourly.index)


def _replay_with_hazard(
    frame: pd.DataFrame, side_by_hourly_ts: dict, model: CryptoMambaExitHazard, mean: np.ndarray, scale: np.ndarray, device: torch.device
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    atr = pd.to_numeric(frame["parent_atr_pct"], errors="raise").to_numpy(dtype=np.float64)
    n = len(frame)
    cash = 1.0
    position = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_fill_i = -1
    entry_equity = 1.0
    peak_move = 0.0
    cooldown_until = -1
    rows: list[dict[str, Any]] = []
    hourly_ts_to_side = side_by_hourly_ts
    frame_avail_ts = frame["available_timestamp"].to_numpy()
    is_new_signal = frame["is_new_parent_signal"].to_numpy()

    for row_i in range(n - 1):
        if position != 0:
            close = float(arrays["close"][row_i])
            move = (
                (close * (1.0 - btc_v2.SLIP_RATE) - entry_price) / entry_price
                if position > 0
                else (entry_price - close * (1.0 + btc_v2.SLIP_RATE)) / entry_price
            )
            peak_move = max(peak_move, move)
            hold_bars = row_i - entry_fill_i
            reason = ""
            if move <= -btc_v2.STOP_ATR_PRICE * entry_atr:
                reason = "stop_loss"
            elif peak_move >= btc_v2.ARM_ATR_PRICE * entry_atr and peak_move - move >= btc_v2.TRAIL_ATR_PRICE * entry_atr:
                reason = "trailing_exit"
            elif hold_bars >= btc_v2.MAX_HOLD_BARS:
                reason = "time_exit"
            elif row_i - max(entry_fill_i, 0) >= SEQ_LEN - 1:
                window_start = row_i - SEQ_LEN + 1
                feats = _in_trade_features(arrays, entry_price, entry_atr, position, entry_fill_i, n)
                hazard = _hazard_prob_at(model, feats[window_start : row_i + 1], mean, scale, device)
                if hazard >= HAZARD_THRESHOLD:
                    reason = "hazard_exit"
            if reason:
                fill_i, exit_price, exit_fee, route = btc_v2._exit_fill(arrays, row_i, position)
                raw_return = (exit_price - entry_price) / entry_price if position > 0 else (entry_price - exit_price) / entry_price
                before = cash
                cash = cash * (1.0 + raw_return * btc_v2.NOTIONAL)
                cash -= before * exit_fee * btc_v2.NOTIONAL
                trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
                rows.append({
                    "entry_fill_i": entry_fill_i, "exit_fill_i": fill_i, "side": position, "reason": reason,
                    "route": route, "hold_bars": hold_bars, "trade_return": trade_return, "win": int(trade_return > 0.0),
                })
                position = 0
                cooldown_until = row_i + btc_v2.COOLDOWN_BARS
                continue

        if position != 0 or row_i < cooldown_until or not bool(is_new_signal[row_i]):
            continue
        avail_ts = pd.Timestamp(frame_avail_ts[row_i])
        source_hourly_ts = avail_ts - pd.Timedelta(hours=1)
        side = int(hourly_ts_to_side.get(source_hourly_ts, 0))
        if side == 0:
            continue
        fill_i = row_i + 1
        entry_price_candidate = float(arrays["open"][fill_i])
        touched = bool(arrays["low"][fill_i] <= entry_price_candidate) if side > 0 else bool(arrays["high"][fill_i] >= entry_price_candidate)
        if not touched:
            continue
        position = side
        entry_price = entry_price_candidate
        entry_equity = cash
        entry_fill_i = fill_i
        entry_atr = max(float(atr[row_i]), 1.0e-6)
        peak_move = 0.0
        cash -= cash * btc_v2.FEE_RATE * btc_v2.MAKER_FEE_MULT * btc_v2.NOTIONAL

    ledger = pd.DataFrame(rows)
    wins = int(ledger["win"].sum()) if len(ledger) else 0
    metrics = {
        "pnl": float((cash - 1.0) * 100.0),
        "trades": int(len(ledger)),
        "win_rate": float(wins / len(ledger)) if len(ledger) else 0.0,
        "mean_trade_return_pct": float(ledger["trade_return"].mean() * 100) if len(ledger) else None,
        "exit_reasons": ledger["reason"].value_counts().to_dict() if len(ledger) else {},
    }
    return metrics, ledger


def _baseline_from_stage1(events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    subset = events.loc[events["event_hour_timestamp"].between(start, end)]
    return {
        "n_events": int(len(subset)),
        "win_rate": float(subset["win"].mean()) if len(subset) else None,
        "mean_trade_return_pct": float(subset["trade_return"].mean() * 100) if len(subset) else None,
    }


def run(history_end: pd.Timestamp) -> dict[str, Any]:
    if history_end >= HOLDOUT_START:
        raise RuntimeError(f"history_end={history_end} >= HOLDOUT_START -- refusing per btc_v3_holdout_policy_20260714.md")

    print("stage=load_candidate_artifacts", flush=True)
    entry_score_df = _load_entry_score()
    entry_score_by_ts = dict(zip(entry_score_df["timestamp"], entry_score_df["btc_v3_hmm_entry_score"]))
    model, mean, scale, device = _load_hazard_model()

    print("stage=load_hourly_and_5m", flush=True)
    hourly, _ = btc_v2._read_hourly()
    hourly = hourly.loc[hourly["timestamp"] <= history_end].reset_index(drop=True)
    signal = pd.DataFrame({
        "source_timestamp": hourly["timestamp"],
        "available_timestamp": hourly["timestamp"] + pd.Timedelta(hours=1),
        "parent_atr_pct": pd.to_numeric(hourly["atr_pct"], errors="raise").to_numpy(dtype=np.float64),
    })
    five_minute = btc_v2._read_five_minute()
    five_minute = five_minute.loc[five_minute["timestamp"] <= history_end].reset_index(drop=True)
    merged = pd.merge_asof(
        five_minute.sort_values("timestamp"), signal.sort_values("available_timestamp"),
        left_on="timestamp", right_on="available_timestamp", direction="backward", allow_exact_matches=True,
    )
    merged["is_new_parent_signal"] = merged["available_timestamp"].ne(merged["available_timestamp"].shift(1))

    print("stage=grid_search_entry_threshold_on_train_val", flush=True)
    grid_rows = []
    for threshold in ENTRY_SCORE_GRID:
        side_by_ts = dict(zip(hourly["timestamp"], _candidate_side(hourly, entry_score_by_ts, threshold)))
        train_val_frame = merged.loc[merged["timestamp"].lt(VAL_START)].reset_index(drop=True)
        metrics, _ = _replay_with_hazard(train_val_frame, side_by_ts, model, mean, scale, device)
        grid_rows.append({"threshold": threshold, **metrics})
        print(f"  threshold={threshold:.2f} pnl={metrics['pnl']:.2f}% trades={metrics['trades']} wr={metrics['win_rate']:.1%}", flush=True)
    grid = pd.DataFrame(grid_rows)
    eligible = grid.loc[grid["trades"] >= 5]
    selected_threshold = float((eligible if len(eligible) else grid).sort_values("pnl", ascending=False).iloc[0]["threshold"])

    print(f"stage=eval_selected_threshold={selected_threshold}", flush=True)
    side_by_ts = dict(zip(hourly["timestamp"], _candidate_side(hourly, entry_score_by_ts, selected_threshold)))
    eval_frame = merged.loc[merged["timestamp"].ge(VAL_START) & merged["timestamp"].le(history_end)].reset_index(drop=True)
    candidate_metrics, candidate_ledger = _replay_with_hazard(eval_frame, side_by_ts, model, mean, scale, device)

    stage1_events = pd.read_parquet(SPARSE_EVENTS_PATH)
    baseline = _baseline_from_stage1(stage1_events, VAL_START, history_end)

    out_dir = CANDIDATE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_ledger.to_csv(out_dir / "candidate_ledger.csv", index=False)
    grid.to_csv(out_dir / "entry_threshold_grid.csv", index=False)

    report = {
        "model_id": "btc_v3_hmm_mamba_candidate_20260715",
        "status": "research_candidate_not_live",
        "supersedes": "none (parallel to stage1_sparse_events)",
        "history_end": str(history_end),
        "holdout_start": str(HOLDOUT_START),
        "eval_window": {"start": str(VAL_START), "end": str(history_end)},
        "entry_threshold_selection": {"grid": grid_rows, "selected_threshold": selected_threshold, "selection_rule": "max eval-window pnl on train/val period among thresholds with >=5 trades"},
        "hazard_threshold": HAZARD_THRESHOLD,
        "candidate": candidate_metrics,
        "stage1_baseline": baseline,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }
    report_path = out_dir / "btc_v3_hmm_mamba_candidate_comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default))
    print(f"stage=done report={report_path}", flush=True)
    print(json.dumps({"candidate": candidate_metrics, "stage1_baseline": baseline}, indent=2, default=_json_default), flush=True)
    return report


def _write_doc(report: dict[str, Any]) -> None:
    doc_path = ROOT / "docs/model_contracts/btc_v3_hmm_mamba_candidate_20260715.md"
    candidate = report["candidate"]
    baseline = report["stage1_baseline"]
    text = f"""# BTC v3 research candidate: HMM entry-score + Mamba exit-hazard (2026-07-15)

status: research_candidate_not_live
supersedes: none (parallel to docs/model_contracts/btc_v3_stage1_sparse_events_20260714.md)

## Design

- Entry: same ts_action transition timing as Stage 1, gated by a causal Gaussian-HMM
  win-probability nowcast (`scripts/build_btc_v3_hmm_entry_score_20260715.py`, reusing
  `GaussianStateModel` from `scripts/retrain_clean_regime_hmm_20260517.py` unmodified) instead of
  Stage 1's raw ts_action-only gate.
- Exit: whichever of {{existing ATR stop/trail/time-exit contract, Mamba exit-hazard >= {report["hazard_threshold"]}}}
  triggers first (`scripts/build_btc_v3_mamba_exit_hazard_20260715.py`, reusing the CBlock/CMBlock
  architecture from `CryptoMambaRegimePred` with a single-logit head).
- Entry-score threshold selected on train/validation only: {report["entry_threshold_selection"]["selection_rule"]}
  -> selected threshold = {report["entry_threshold_selection"]["selected_threshold"]}.

## Result (eval window {report["eval_window"]["start"]}..{report["eval_window"]["end"]})

| | Stage 1 baseline (ts_action + ATR-only) | This candidate (HMM entry + hazard exit) |
|---|---|---|
| n_events / trades | {baseline["n_events"]} | {candidate["trades"]} |
| win_rate | {baseline["win_rate"]} | {candidate["win_rate"]} |
| mean_trade_return_pct | {baseline["mean_trade_return_pct"]} | {candidate["mean_trade_return_pct"]} |

Exit reason breakdown (candidate): {json.dumps(candidate["exit_reasons"])}

## Compliance

- fresh_forward_bar_by_bar: true
- trade_ledgers_used_as_input (live scoring path): false
- saved_parent_exit_timestamps_used: false
- future_rows_used_for_entry: false
- All training/threshold-selection decisions use only data before docs/model_contracts/btc_v3_holdout_policy_20260714.md's HOLDOUT_START (2026-07-14 00:00:00 UTC).

## Judgment call

This is a research readout only. See the numbers above -- pursue further (e.g. promote into a real
Stage 3 candidate) or discard based on whether the candidate's win_rate/mean_trade_return_pct
meaningfully beats the Stage 1 baseline on this held-out eval window.
"""
    doc_path.write_text(text, encoding="utf-8")
    print(f"stage=doc_written path={doc_path}", flush=True)


def main() -> int:
    history_end = pd.Timestamp("2026-07-12 23:59:59")
    report = run(history_end)
    _write_doc(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
