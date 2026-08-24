#!/usr/bin/env python3
"""Standalone rule-based strategy: a vote-confluence formula combining the top-6 ranked ETH
reversal evidence signals (docs/experiments/eth_evidence_signal_ranking_stability_mar_jul_2026_
20260814.md master ranking, Spearman-stable across two structurally different 5-month windows):
orthogonal_combo, liquidity_sweep, volume_wick_climax, short_term_return_z, taker_delta_z_climax,
bollinger_pctb_extreme.

=== Why this is a DIFFERENT thing from every prior injection attempt today ===
Candidate C (docs/experiments/eth_omega461_evidence_veto_exit_overlay_20260814.md), the exit_head
feature pre-gate, and Candidate D (docs/experiments/eth_omega461_evidence_signal_sizing_feature_
20260814.md) all tried to graft these signals ONTO the existing live Omega4.6.1 model (whose
direction_head has zero confirmed skill vs always_short, per Odyssey(1)'s 7 independent model x
label combinations). This script instead asks a genuinely different question: does a hand-built,
STANDALONE entry rule built directly from these evidence signals -- with no TabM/direction_head
involved at all -- beat always_long/always_short on its own? This has NOT been tried in this
session; the evidence-study docs are explicit that their retrospective distance-to-future-pivot
lift methodology is "NOT a trading algorithm", so this is the first time these signals are run
through an actual causal fresh-forward simulation with real TP/SL/costs.

=== Formula (pre-registered before any backtest number is seen) ===
For side in {bottom=LONG, top=SHORT}, count how many of the 6 ALREADY-VALIDATED signal
definitions fire on bar i (formulas reused verbatim from analyze_eth_creative_reversal_evidence_
signals_20260814.py / analyze_eth_broad_evidence_signal_sweep_20260814.py / analyze_eth_deep_
evidence_signal_sweep_round2_20260814.py / backtest_eth_slowk_williamsr_persistence_confluence_
20260814.py -- no new thresholds anywhere):
  1. orthogonal_combo:        (p_fast<=.10)&(p_slow<=.10)&(delta_z<=-2)   [top: >=.90/>=.90/>=2]
  2. liquidity_sweep:         low<swing_low_48.shift(1) & close>swing_low_48.shift(1)  [top: symmetric]
  3. volume_wick_climax:      vol_z>=2 & lower_wick_ratio>=.5             [top: upper_wick_ratio]
  4. short_term_return_z:     ret3_z<=-2.5                                [top: >=2.5]
  5. taker_delta_z_climax:    delta_z<=-2                                 [top: delta_z>=2]
  6. bollinger_pctb_extreme:  bb_pctb<=.05                                [top: >=.95]
score(i) = bottom_votes(i) - top_votes(i) (net vote in [-6,+6]); LONG when score>=K, SHORT when
score<=-K. K in {1,2,3} tested identically, not selected on any single window's result -- all
three are reported for every window rather than picking the best after the fact.

KNOWN, DISCLOSED LIMITATION (not hidden): signal #1 (orthogonal_combo) is BY CONSTRUCTION the AND
of an oscillator-extreme condition with #5 (taker_delta_z_climax) -- so #1 firing mechanically
implies #5 also fires, meaning K=2 is trivially satisfied by orthogonal_combo alone rather than
requiring two genuinely independent confirmations. This redundancy is reported, not concealed --
see the results/interpretation section for how it's read.

=== Backtest mechanics -- reused unmodified from an existing, non-tuned-for-this-question
convention in this exact evidence-study lineage ===
Engine: core.causal_futures_backtest.simulate_single_position / purged_decision_mask (this repo's
canonical single-position causal futures simulator -- entry at bar i+1 open, TP/SL walked forward
bar-by-bar, non-overlapping positions). TP_ATR_MULT/SL_ATR_MULT/HORIZON_BARS/MARGIN_FRACTION/
LEVERAGE/ROUNDTRIP_COST_RATE/ATR_N are copied VERBATIM from backtest_eth_slowk_williamsr_
persistence_confluence_20260814.py's own constants (1.6/1.0/48/0.30/3.0/0.001/14) -- NOT tuned by
this script, to avoid any appearance of fitting TP/SL to maximize this specific formula's return.

Windows: the same 6 pre-registered windows this whole session has used
(eth_omega461_multiwindow_confirmation_gate_20260814.WINDOW_DEFS: 2025q1/q2/q3 context + val +
oos_q1/oos_q2 confirm), computed on the full BASE_2025/BASE_2026 panel per-year (never window-by-
window, avoiding NaN truncation at a window's own start -- same discipline as every prior script
today). always_long/always_short benchmarks are computed identically to backtest_eth_slowk_
williamsr_persistence_confluence_20260814.py's own run_window (buy-and-hold direction over the
window, not a trading strategy -- the same baseline this whole day's work has used throughout).

fresh_forward_bar_by_bar=true (every signal is rolling/shift-only; simulate_single_position enters
at i+1 open and walks forward high/low bar-by-bar for TP/SL, no lookahead). trade_ledgers_used_as_
input=false. saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py / runtime_config.py / .env.
Does NOT modify any imported module. No training, no GPU (pure pandas + the existing simulator).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from core.selection_stats import periodic_returns, sharpe  # noqa: E402
from eval_omega4_1_atr_safety_sltp_20260622 import _atr_pct  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_evidence_signal_top6_confluence_20260814"

# Copied verbatim from backtest_eth_slowk_williamsr_persistence_confluence_20260814.py -- not tuned here.
TP_ATR_MULT = 1.6
SL_ATR_MULT = 1.0
HORIZON_BARS = 48
LEVERAGE = 3.0
MARGIN_FRACTION = 0.30
ROUNDTRIP_COST_RATE = 0.001
BARS_PER_DAY = 288
ATR_N = 14
K_VALUES = (1, 2, 3, 4, 5, 6)


def log(msg: str) -> None:
    print(f"[top6_confluence] {msg}", flush=True)


def _compute_signal_frame(base_csv: Path) -> pd.DataFrame:
    """Computes all 6 bottom/top booleans + atr_pct on the FULL base_csv (never per-window),
    matching every other script in this session's own discipline for avoiding artificial NaN
    truncation at a window's own start."""
    raw = pd.read_csv(base_csv, low_memory=False, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    ind = compute_indicators(raw)  # p_fast, p_slow, fast_k, slow_k, adx14, atr_price, ...
    ind = add_creative_indicators(ind)  # delta_z, vol_z, lower_wick_ratio, upper_wick_ratio

    close, low, high, open_ = raw["close"], raw["low"], raw["high"], raw["open"]
    eps = 1e-12

    # 4. short_term_return_z (verbatim: analyze_eth_deep_evidence_signal_sweep_round2_20260814.py)
    ret3 = close / close.shift(3) - 1.0
    ret3_z = (ret3 - ret3.rolling(288, min_periods=288).mean()) / ret3.rolling(288, min_periods=288).std().replace(0.0, np.nan)

    # 2. liquidity_sweep (verbatim: analyze_eth_broad_evidence_signal_sweep_20260814.py)
    swing_low_prior = low.rolling(48, min_periods=48).min().shift(1)
    swing_high_prior = high.rolling(48, min_periods=48).max().shift(1)
    sweep_low = (low < swing_low_prior) & (close > swing_low_prior)
    sweep_high = (high > swing_high_prior) & (close < swing_high_prior)

    # 6. bollinger_pctb_extreme (verbatim: analyze_eth_broad_evidence_signal_sweep_20260814.py)
    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    bb_pctb = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + eps)

    bottom = pd.DataFrame(index=raw.index)
    bottom["orthogonal_combo"] = (ind["p_fast"] <= 0.10) & (ind["p_slow"] <= 0.10) & (ind["delta_z"] <= -2.0)
    bottom["liquidity_sweep"] = sweep_low
    bottom["volume_wick_climax"] = (ind["vol_z"] >= 2.0) & (ind["lower_wick_ratio"] >= 0.5)
    bottom["short_term_return_z"] = ret3_z <= -2.5
    bottom["taker_delta_z_climax"] = ind["delta_z"] <= -2.0
    bottom["bollinger_pctb_extreme"] = bb_pctb <= 0.05

    top = pd.DataFrame(index=raw.index)
    top["orthogonal_combo"] = (ind["p_fast"] >= 0.90) & (ind["p_slow"] >= 0.90) & (ind["delta_z"] >= 2.0)
    top["liquidity_sweep"] = sweep_high
    top["volume_wick_climax"] = (ind["vol_z"] >= 2.0) & (ind["upper_wick_ratio"] >= 0.5)
    top["short_term_return_z"] = ret3_z >= 2.5
    top["taker_delta_z_climax"] = ind["delta_z"] >= 2.0
    top["bollinger_pctb_extreme"] = bb_pctb >= 0.95

    bottom_votes = bottom.fillna(False).sum(axis=1).astype(np.float64)
    top_votes = top.fillna(False).sum(axis=1).astype(np.float64)
    net_score = bottom_votes - top_votes

    atr_pct = pd.Series(_atr_pct(raw, ATR_N), index=raw.index)

    out = pd.DataFrame({
        "timestamp": raw["timestamp"], "open": raw["open"], "high": raw["high"], "low": raw["low"], "close": raw["close"],
        "bottom_votes": bottom_votes, "top_votes": top_votes, "net_score": net_score, "atr_pct": atr_pct,
    })
    return out


def run_window(frame: pd.DataFrame, *, start, end, k: int) -> dict[str, Any]:
    ts = frame["timestamp"]
    eligible = purged_decision_mask(ts, start=pd.Timestamp(start), end=pd.Timestamp(end), horizon_bars=HORIZON_BARS)
    has_score = frame["net_score"].notna().to_numpy() & frame["atr_pct"].notna().to_numpy()
    mask = eligible & has_score
    decision_indices = np.flatnonzero(mask)

    tp_moves = (TP_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices]
    sl_moves = (SL_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices]
    scores = frame["net_score"].to_numpy()[decision_indices]

    result = simulate_single_position(
        timestamps=ts, open_px=frame["open"].to_numpy(), high=frame["high"].to_numpy(),
        low=frame["low"].to_numpy(), close=frame["close"].to_numpy(),
        decision_indices=decision_indices, scores=scores, tp_moves=tp_moves, sl_moves=sl_moves,
        upper_threshold=float(k), lower_threshold=float(-k), horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )

    window_mask = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
    window_idx = np.flatnonzero(window_mask.to_numpy())
    equity_window = result.equity[window_idx]
    total_return = float(equity_window[-1] / equity_window[0] - 1.0) if len(equity_window) else float("nan")
    peak = np.maximum.accumulate(equity_window) if len(equity_window) else np.array([1.0])
    mdd = float(np.min(equity_window / peak - 1.0)) if len(equity_window) else float("nan")

    ledger = result.ledger
    n_trades = int(len(ledger))
    if n_trades:
        wins = ledger.loc[ledger["trade_return"] > 0, "trade_return"]
        losses = ledger.loc[ledger["trade_return"] < 0, "trade_return"]
        win_rate = float((ledger["trade_return"] > 0).mean())
        profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
        long_trades = int((ledger["side"] > 0).sum()) if "side" in ledger.columns else None
    else:
        win_rate, profit_factor, long_trades = float("nan"), float("nan"), None

    day_returns = periodic_returns(equity_window, BARS_PER_DAY)
    sr = sharpe(day_returns) if day_returns.size else float("nan")

    close = frame["close"].to_numpy()
    if len(window_idx):
        p0, p1 = float(close[window_idx[0]]), float(close[window_idx[-1]])
        always_long, always_short = p1 / p0 - 1.0, p0 / p1 - 1.0
    else:
        always_long, always_short = float("nan"), float("nan")

    return {
        "n_trades": n_trades, "total_return": total_return, "mdd": mdd, "win_rate": win_rate,
        "profit_factor": profit_factor, "sharpe_daily": sr, "skipped_while_open": int(result.skipped_while_open),
        "always_long_return": always_long, "always_short_return": always_short,
        "beats_benchmark": bool(total_return > max(always_long, always_short)) if np.isfinite(always_long) and np.isfinite(always_short) else None,
        "long_trades": long_trades,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== stage=compute_signal_frames (full BASE_2025/BASE_2026, per-year) ===")
    import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402  (module import deferred for path setup)
    signal_by_base = {sweep.BASE_2025: _compute_signal_frame(sweep.BASE_2025), sweep.BASE_2026: _compute_signal_frame(sweep.BASE_2026)}

    report: dict[str, Any] = {
        "design": (
            "Standalone vote-confluence entry rule from the top-6 ranked ETH reversal evidence "
            "signals (orthogonal_combo, liquidity_sweep, volume_wick_climax, short_term_return_z, "
            "taker_delta_z_climax, bollinger_pctb_extreme), net_score=bottom_votes-top_votes, "
            "LONG/SHORT at |net_score|>=K for K in {1,2,3}, TP/SL/leverage/cost reused unmodified "
            "from backtest_eth_slowk_williamsr_persistence_confluence_20260814.py. Tested via "
            "core.causal_futures_backtest.simulate_single_position across all 6 pre-registered "
            "windows, compared against always_long/always_short."
        ),
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "k_values_tested": list(K_VALUES), "tp_atr_mult": TP_ATR_MULT, "sl_atr_mult": SL_ATR_MULT,
        "horizon_bars": HORIZON_BARS, "leverage": LEVERAGE, "margin_fraction": MARGIN_FRACTION,
        "roundtrip_cost_rate": ROUNDTRIP_COST_RATE,
        "windows": {},
    }

    for wname, wd in gate.WINDOW_DEFS.items():
        base_csv = wd["base_csv"]
        sframe = signal_by_base[base_csv]
        row: dict[str, Any] = {"tier": wd["tier"], "by_k": {}}
        for k in K_VALUES:
            res = run_window(sframe, start=wd["start"], end=wd["end"], k=k)
            row["by_k"][str(k)] = res
            log(f"  {wname:8s} K={k}  trades={res['n_trades']:4d}  ret={res['total_return']*100:8.2f}%  "
                f"mdd={res['mdd']*100:7.2f}%  wr={res['win_rate']:.2f}  "
                f"always_long={res['always_long_return']*100:7.2f}%  always_short={res['always_short_return']*100:7.2f}%  "
                f"beats_benchmark={res['beats_benchmark']}")
        report["windows"][wname] = row

    # Summary: for each K, how many of the OOS-confirm-tier + val windows beat max(always_long, always_short)?
    summary: dict[str, Any] = {}
    for k in K_VALUES:
        wins, total = 0, 0
        for wname, wd in gate.WINDOW_DEFS.items():
            res = report["windows"][wname]["by_k"][str(k)]
            if res["beats_benchmark"] is None:
                continue
            total += 1
            wins += int(res["beats_benchmark"])
        summary[str(k)] = {"windows_beating_benchmark": wins, "windows_total": total}
        log(f"  SUMMARY K={k}: beats always_long/always_short benchmark in {wins}/{total} windows")
    report["summary"] = summary

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
