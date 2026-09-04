#!/usr/bin/env python3
"""Regime-aware entry-gate + regime-based stop-loss for the 2 evidence-signal candidates confirmed
robust across all 3 regime sources today (orthogonal_combo bottom, short_term_return_z top -- see
memory eth_evidence_signal_regime_model_comparison_20260827), plus 3 secondary candidates at one
representative config each (liquidity_sweep/smt_divergence/volume_wick_climax, all bottom).

Extends backtest_eth_evidence_signal_chop_gated_costgate_20260827.py (untouched, kept as the
comparison baseline) two ways:

  1. ENTRY GATE: GBM3's un-debounced argmax chop gate -> GBM2's trend/chop, additionally requiring
     the confirmed state to have held for k_entry consecutive bars (via train_eth_regime_gbm2_
     trend_chop_20260827._debounce, reused unchanged) before a new evidence-signal trigger is
     allowed to open a position. Rationale (this session, discretionary-strategy discussion):
     entering a fade needs PATIENCE -- a chop reading that's about to flip to trend is exactly the
     worst moment to fade, so a longer k_entry should reduce bad entries right before breakouts.

  2. EXIT: an early "regime_stop" exit added on top of the existing TP/SL/max-hold ladder -- if
     GBM2's RAW (undebounced) trend_prob for the current bar crosses theta_exit WHILE the position
     is underwater, exit immediately at that bar's close rather than waiting for the fixed ATR-based
     SL or the 48-bar timeout. Rationale: once already in a losing fade, speed matters more than
     patience -- this is the opposite requirement from entry, hence two different debounce/threshold
     knobs on the SAME underlying trend_prob signal, not two different models.

Implementation note: core.causal_futures_backtest.{_resolve_trade,simulate_single_position} are NOT
modified -- other scripts depend on their exact current behavior. This script vendors a local
regime_stop-aware variant of both functions instead (small, ~60 lines total), verified against the
original with theta_exit=None (must reproduce a plain SL/TP/timeout run bit-for-bit).

Primary metric is NOT beats_benchmark (already known structurally unbeatable in strong-trend windows
per today's cost-gate run) but loss reduction and MAE (max adverse excursion during the hold) versus
the theta_exit=None baseline at the same k_entry -- does the regime stop actually cut losses on the
trades that would otherwise have gotten run over by a breakout?

Same 6 pre-registered windows / TP=1.6xATR / SL=1.0xATR / 48-bar horizon / 3x leverage / 30% margin /
10bp roundtrip cost as every sibling script in this lineage -- see backtest_eth_evidence_signal_
chop_gated_costgate_20260827.py for the shared convention this reuses unchanged.

CAVEAT (same as every regime-conditional script in this lineage, restated because it now applies to
a DIFFERENT model): all 6 windows (2025q1..oos_q2, ending 2026-06-30) sit inside GBM2's own TRAIN
range (2024-01-01~2026-06-30) -- its trend/chop calls here are in-sample. Symmetric with the
already-in-sample GBM3 gate this replaces, so this does not introduce a new asymmetry, but it means
neither this script's numbers, nor the original's, are a clean OOS test of the regime classifier
itself.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false,
future_rows_used_for_entry=false. No training here (GBM2 already trained); read-only inference only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.causal_futures_backtest import purged_decision_mask  # noqa: E402
from eval_omega4_1_atr_safety_sltp_20260622 import _atr_pct  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_eth_funding_crossasset_combo_signal_20260825 import load_funding_z  # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402
from train_eth_regime_gbm2_trend_chop_20260827 import _debounce  # noqa: E402

OUT_DIR = ROOT / "tmp/eth_evidence_signal_regime_entry_exit_20260827"
BTC_PATH = ROOT / "data" / "btc_5m_1year.csv"
GBM2_MODEL_PATH = ROOT / "tmp" / "eth_regime_gbm2_trend_chop_20260827" / "model.joblib"

TP_ATR_MULT = 1.6
SL_ATR_MULT = 1.0
HORIZON_BARS = 48
LEVERAGE = 3.0
MARGIN_FRACTION = 0.30
ROUNDTRIP_COST_RATE = 0.001
ATR_N = 14

PRIMARY_CANDIDATES = [("orthogonal_combo", "bottom"), ("short_term_return_z", "top")]
SECONDARY_CANDIDATES = [("liquidity_sweep", "bottom"), ("smt_divergence", "bottom"), ("volume_wick_climax", "bottom")]
K_ENTRY_GRID = [1, 3, 6, 12]
THETA_EXIT_GRID = [None, 0.5, 0.6, 0.7]
SECONDARY_K_ENTRY = 3
SECONDARY_THETA_EXIT_GRID = [None, 0.6]


def log(msg: str) -> None:
    print(f"[regime_entry_exit] {msg}", flush=True)


# --- vendored, regime_stop-aware variants of core.causal_futures_backtest -- the shared module is
# NOT modified; other scripts depend on its exact current behavior. ---

def _resolve_trade_regime_stop(
    *, side: int, entry: float, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    tp_move: float, sl_move: float, trend_prob: np.ndarray, theta_exit: float | None,
    regime_persist_bars: int = 1,
) -> tuple[float, str, int, float]:
    """Same SL(intrabar)->TP(intrabar)->timeout(close) ladder as core.causal_futures_backtest.
    _resolve_trade, plus an optional regime_stop check (close-based) after SL/TP both miss for the
    bar. Also tracks mae -- the worst (most negative) unrealized move seen before exit, for
    reporting; does not affect the exit decision itself.

    regime_persist_bars (added after diagnosing the first pass): a single-bar trend_prob spike
    fired on genuine SL-bound trades AND on temporary bounces that went on to hit TP anyway (see
    memory eth_regime_entry_exit_backtest_diagnosis_20260827 -- 36/47 interventions helped, 11/47
    were false alarms costing ~3x more per trade than a correct catch saved). Requiring the
    underwater+trend_prob>=theta condition to hold for this many CONSECUTIVE bars before firing is
    meant to filter the bounce-back false alarms while still beating the natural SL, which took
    ~11 bars on average in the diagnosed sample -- 2-3 bars of confirmation is cheap by comparison."""
    mae = 0.0
    streak = 0
    if side > 0:
        tp_level, sl_level = entry * (1.0 + tp_move), entry * (1.0 - sl_move)
        for offset, (bar_high, bar_low, bar_close) in enumerate(zip(high, low, close)):
            mae = min(mae, bar_low / entry - 1.0)
            if bar_low <= sl_level:
                return -sl_move, "sl", offset, mae
            if bar_high >= tp_level:
                return tp_move, "tp", offset, mae
            if theta_exit is not None and trend_prob[offset] >= theta_exit and bar_close < entry:
                streak += 1
                if streak >= regime_persist_bars:
                    return (bar_close / entry - 1.0), "regime_stop", offset, mae
            else:
                streak = 0
        return float(close[-1] / entry - 1.0), "timeout", len(close) - 1, mae

    tp_level, sl_level = entry * (1.0 - tp_move), entry * (1.0 + sl_move)
    for offset, (bar_high, bar_low, bar_close) in enumerate(zip(high, low, close)):
        mae = min(mae, 1.0 - bar_high / entry)
        if bar_high >= sl_level:
            return -sl_move, "sl", offset, mae
        if bar_low <= tp_level:
            return tp_move, "tp", offset, mae
        if theta_exit is not None and trend_prob[offset] >= theta_exit and bar_close > entry:
            streak += 1
            if streak >= regime_persist_bars:
                return (1.0 - bar_close / entry), "regime_stop", offset, mae
        else:
            streak = 0
    return float(1.0 - close[-1] / entry), "timeout", len(close) - 1, mae


def simulate_single_position_regime_stop(
    *, timestamps, open_px, high, low, close, trend_prob, decision_indices, scores,
    tp_moves, sl_moves, upper_threshold, lower_threshold, horizon_bars,
    margin_fraction, leverage, roundtrip_cost_rate, theta_exit, regime_persist_bars=1,
):
    ts = pd.DatetimeIndex(timestamps)
    open_values, high_values, low_values, close_values = (np.asarray(a, dtype=np.float64) for a in (open_px, high, low, close))
    trend_prob_values = np.asarray(trend_prob, dtype=np.float64)
    idxs = np.asarray(decision_indices, dtype=np.int64)
    score_values, tp_values, sl_values = (np.asarray(a, dtype=np.float64) for a in (scores, tp_moves, sl_moves))
    notional = float(margin_fraction * leverage)
    account_cost = float(roundtrip_cost_rate * notional)
    equity = np.ones(len(ts), dtype=np.float64)
    cash = 1.0
    filled_through = -1
    occupied_through = -1
    rows: list[dict] = []

    for decision_i, score, tp_move, sl_move in zip(idxs, score_values, tp_values, sl_values):
        if not np.isfinite(score) or not np.isfinite(tp_move) or not np.isfinite(sl_move):
            continue
        side = 1 if score >= upper_threshold else -1 if score <= lower_threshold else 0
        if side == 0:
            continue
        entry_i = int(decision_i) + 1
        if entry_i >= len(ts) or entry_i <= occupied_through:
            continue
        final_i = min(entry_i + horizon_bars - 1, len(ts) - 1)
        if final_i < entry_i:
            continue
        if filled_through + 1 < entry_i:
            equity[filled_through + 1 : entry_i] = cash

        entry = float(open_values[entry_i])
        price_move, reason, exit_offset, mae = _resolve_trade_regime_stop(
            side=side, entry=entry,
            high=high_values[entry_i : final_i + 1], low=low_values[entry_i : final_i + 1],
            close=close_values[entry_i : final_i + 1], tp_move=float(tp_move), sl_move=float(sl_move),
            trend_prob=trend_prob_values[entry_i : final_i + 1], theta_exit=theta_exit,
            regime_persist_bars=regime_persist_bars,
        )
        exit_i = entry_i + exit_offset
        for bar_i in range(entry_i, exit_i + 1):
            unrealized = close_values[bar_i] / entry - 1.0 if side > 0 else 1.0 - close_values[bar_i] / entry
            equity[bar_i] = cash * (1.0 + unrealized * notional - account_cost)
        trade_return = float(price_move * notional - account_cost)
        cash *= 1.0 + trade_return
        equity[exit_i] = cash
        filled_through = exit_i
        occupied_through = exit_i
        rows.append({"side": side, "reason": reason, "bars_held": int(exit_offset + 1),
                      "price_move": float(price_move), "mae": float(mae), "trade_return": trade_return})

    if filled_through + 1 < len(equity):
        equity[filled_through + 1 :] = cash
    return equity, pd.DataFrame(rows)


def _gbm2_trend_prob(raw: pd.DataFrame) -> np.ndarray:
    feats = _with_raw_state12(raw)
    payload = joblib.load(GBM2_MODEL_PATH)
    cols = payload["feature_cols"]
    med = pd.Series(payload["feature_medians"])
    for c in cols:
        if c not in feats.columns:
            feats[c] = med.get(c, 0.0)
    x = feats[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    proba = payload["model"].predict_proba(x)
    return proba[:, list(payload["classes"]).index("trend")]


def _compute_frame(base_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(base_csv, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    btc = pd.read_csv(BTC_PATH, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    funding = load_funding_z()
    base_cols = ["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]
    sig = compute_signals(raw[base_cols].copy(), btc_df=btc, funding_df=funding)

    sig["trend_prob"] = _gbm2_trend_prob(raw)
    sig["is_trend_raw"] = (sig["trend_prob"].to_numpy() >= 0.5).astype(int)
    sig["atr_pct"] = pd.Series(_atr_pct(raw, ATR_N), index=raw.index)
    return sig


def run_window(frame: pd.DataFrame, bcol: str | None, tcol: str | None, k_entry: int, theta_exit: float | None,
                *, start, end, regime_persist_bars: int = 1) -> dict[str, Any]:
    ts = frame["timestamp"]
    eligible = purged_decision_mask(ts, start=pd.Timestamp(start), end=pd.Timestamp(end), horizon_bars=HORIZON_BARS)

    bottom = frame[bcol].fillna(False).to_numpy() if bcol else np.zeros(len(frame), dtype=bool)
    top = frame[tcol].fillna(False).to_numpy() if tcol else np.zeros(len(frame), dtype=bool)
    confirmed = _debounce(frame["is_trend_raw"].to_numpy(), k_entry)
    chop_confirmed = confirmed == 0
    bottom = bottom & chop_confirmed
    top = top & chop_confirmed
    score = bottom.astype(np.float64) - top.astype(np.float64)

    has_score = frame["atr_pct"].notna().to_numpy()
    mask = eligible & has_score
    decision_indices = np.flatnonzero(mask)
    tp_moves = (TP_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices]
    sl_moves = (SL_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices]

    equity, ledger = simulate_single_position_regime_stop(
        timestamps=ts, open_px=frame["open"].to_numpy(), high=frame["high"].to_numpy(),
        low=frame["low"].to_numpy(), close=frame["close"].to_numpy(), trend_prob=frame["trend_prob"].to_numpy(),
        decision_indices=decision_indices, scores=score[decision_indices], tp_moves=tp_moves, sl_moves=sl_moves,
        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
        theta_exit=theta_exit, regime_persist_bars=regime_persist_bars,
    )
    n_trades = int(len(ledger))
    total_return = float(equity[-1] - 1.0) if len(equity) else float("nan")
    wr = float((ledger["price_move"] > 0).mean()) if n_trades else float("nan")
    n_regime_stop = int((ledger["reason"] == "regime_stop").sum()) if n_trades else 0
    n_sl = int((ledger["reason"] == "sl").sum()) if n_trades else 0
    mean_mae = float(ledger["mae"].mean()) if n_trades else float("nan")
    worst_mae = float(ledger["mae"].min()) if n_trades else float("nan")

    win_mask = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
    win_idx = np.flatnonzero(win_mask.to_numpy())
    if len(win_idx):
        p0, p1 = float(frame["close"].iloc[win_idx[0]]), float(frame["close"].iloc[win_idx[-1]])
        always_long, always_short = p1 / p0 - 1.0, p0 / p1 - 1.0
    else:
        always_long, always_short = float("nan"), float("nan")

    return {
        "n_trades": n_trades, "wr": wr, "total_return": total_return, "n_regime_stop": n_regime_stop,
        "n_sl": n_sl, "mean_mae": mean_mae, "worst_mae": worst_mae,
        "always_long_return": always_long, "always_short_return": always_short,
        "beats_benchmark": bool(total_return > max(always_long, always_short))
        if np.isfinite(always_long) and np.isfinite(always_short) else None,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("Building 2025/2026 frames (live evidence signals + GBM2 trend_prob + ATR)...")
    frames = {"2025": _compute_frame(gate.sweep.BASE_2025), "2026": _compute_frame(gate.sweep.BASE_2026)}

    # sanity: theta_exit=None, k_entry=1 must reproduce a plain (no-gate-persistence, no-regime-stop)
    # backtest -- i.e. chop_confirmed at k_entry=1 is just the raw is_trend_raw==0 mask.
    sanity = run_window(frames["2025"], "bottom_orthogonal_combo", None, k_entry=1, theta_exit=None,
                         start=gate.WINDOW_DEFS["2025q1"]["start"], end=gate.WINDOW_DEFS["2025q1"]["end"])
    log(f"sanity check (k_entry=1, theta_exit=None, orthogonal_combo 2025q1): {sanity}")

    report: dict[str, Any] = {"config": {"tp_atr_mult": TP_ATR_MULT, "sl_atr_mult": SL_ATR_MULT,
                                          "horizon_bars": HORIZON_BARS, "leverage": LEVERAGE,
                                          "margin_fraction": MARGIN_FRACTION,
                                          "roundtrip_cost_rate": ROUNDTRIP_COST_RATE},
                               "results": {}}

    def run_grid(name, side, k_entry_grid, theta_grid):
        bcol = f"bottom_{name}" if side == "bottom" else None
        tcol = f"top_{name}" if side == "top" else None
        for k_entry in k_entry_grid:
            for theta_exit in theta_grid:
                key = f"{name}:{side}:k{k_entry}:theta{theta_exit}"
                windows_out = {}
                for wname, wd in gate.WINDOW_DEFS.items():
                    frame = frames["2025"] if wd["base_csv"] == gate.sweep.BASE_2025 else frames["2026"]
                    windows_out[wname] = run_window(frame, bcol, tcol, k_entry, theta_exit, start=wd["start"], end=wd["end"])
                report["results"][key] = windows_out
                total_regime_stops = sum(w["n_regime_stop"] for w in windows_out.values())
                total_return_sum = sum(w["total_return"] for w in windows_out.values())
                mean_worst_mae = np.nanmean([w["worst_mae"] for w in windows_out.values()])
                log(f"{key}: sum(total_return)={total_return_sum*100:.1f}%  regime_stops={total_regime_stops}  mean(worst_mae)={mean_worst_mae*100:.2f}%")

    for name, side in PRIMARY_CANDIDATES:
        log(f"\n=== PRIMARY: {name} ({side}) ===")
        run_grid(name, side, K_ENTRY_GRID, THETA_EXIT_GRID)

    for name, side in SECONDARY_CANDIDATES:
        log(f"\n=== SECONDARY: {name} ({side}) ===")
        run_grid(name, side, [SECONDARY_K_ENTRY], SECONDARY_THETA_EXIT_GRID)

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
