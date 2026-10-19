#!/usr/bin/env python3
"""Fresh-forward bar-by-bar backtest of the Slow %K x Williams %R persistence-confluence
oscillator specified in docs/experiments/eth_slowk_williamsr_persistence_confluence_20260814.md.

fresh_forward_bar_by_bar=True, trade_ledgers_used_as_input=False,
saved_parent_exit_timestamps_used=False, future_rows_used_for_entry=False -- every indicator
is computed from a rolling window ending at bar t only; entries execute at bar t+1 open; TP/SL/
time-exit are resolved walking forward bar-by-bar via core.causal_futures_backtest (no stored
ledger reuse, no future-row joins).

Data availability note: the design doc's default OOS window is 2026-01-01..2026-03-31, but
data/eth_5m_1year.csv ends 2026-02-17 15:00. OOS is therefore truncated to 2026-01-01..2026-02-17,
recorded here (not silently reconciled) per CLAUDE.md's "if date boundaries change, state it".

Ablation arms isolate each design element against the literal original idea (arm "a"):
  a  original: fixed 80/20 level AND on raw %R + Slow %K, current bar only, no trigger/gate
  b  full design: adaptive percentile + persistence lookback + spread re-cross trigger + regime gate
  c  b minus adaptive percentile (fixed 80/20 instead)
  d  b minus spread re-cross trigger (enter immediately once persistence confirms)
  e  b minus regime gate (always fade, never follow-mode)
Parameters (q, N) are grid-tuned on VAL only; OOS gets a single look at the VAL-selected config.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from core.selection_stats import periodic_returns, sharpe  # noqa: E402
from eval_omega4_1_atr_safety_sltp_20260622 import _atr_pct  # noqa: E402


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Inlined verbatim from scripts/experiment_regime3_current_hmm_wide24_20260529.py::_adx --
    importing that module transitively requires mamba_ssm (unrelated HMM/Mamba research code),
    so the ~10-line formula is duplicated here rather than dragging in that heavy dependency."""
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    up = high.diff()
    down = -low.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    ndm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    pdi = 100.0 * pdm.ewm(span=period, adjust=False).mean() / (atr + 1e-12)
    ndi = 100.0 * ndm.ewm(span=period, adjust=False).mean() / (atr + 1e-12)
    dx = 100.0 * (pdi - ndi).abs() / (pdi + ndi + 1e-12)
    return dx.ewm(span=period, adjust=False).mean()

DATA_PATH = ROOT / "data" / "eth_5m_1year.csv"
OUT_DIR = ROOT / "tmp" / "eth_slowk_williamsr_persistence_confluence_20260814"

VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END_TARGET = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31 23:59:59")

STOCH_N = 14
SLOWK_SMOOTH = 3
ADX_N = 14
EMA_N = 48
ATR_N = 14
ADX_TREND_MIN = 25.0
TP_ATR_MULT = 1.6
SL_ATR_MULT = 1.0
HORIZON_BARS = 48
LEVERAGE = 3.0
MARGIN_FRACTION = 0.30
ROUNDTRIP_COST_RATE = 0.001  # 0.1%
BARS_PER_DAY = 288


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, usecols=["timestamp", "open", "high", "low", "close", "volume"], parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    hh = high.rolling(STOCH_N, min_periods=STOCH_N).max()
    ll = low.rolling(STOCH_N, min_periods=STOCH_N).min()
    rng = (hh - ll).replace(0.0, np.nan)
    williams_r = -100.0 * (hh - close) / rng
    fast_k = 100.0 + williams_r  # == raw Fast Stochastic %K(14)
    slow_k = fast_k.rolling(SLOWK_SMOOTH, min_periods=SLOWK_SMOOTH).mean()

    percentile_window = 864
    p_fast = fast_k.rolling(percentile_window, min_periods=percentile_window).rank(pct=True)
    p_slow = slow_k.rolling(percentile_window, min_periods=percentile_window).rank(pct=True)

    adx14 = _adx(high, low, close, period=ADX_N)
    ema48 = close.ewm(span=EMA_N, adjust=False).mean()
    ema48_slope = ema48.diff()
    atr_pct = pd.Series(_atr_pct(df, ATR_N), index=df.index)
    atr_price = atr_pct * close

    out = df.copy()
    out["fast_k"] = fast_k
    out["slow_k"] = slow_k
    out["spread"] = fast_k - slow_k
    out["p_fast"] = p_fast
    out["p_slow"] = p_slow
    out["adx14"] = adx14
    out["ema48_slope"] = ema48_slope
    out["atr_pct"] = atr_pct
    out["atr_price"] = atr_price
    out["trend_on"] = (adx14 > ADX_TREND_MIN) & (ema48_slope.abs() > 0.5 * atr_price / EMA_N)
    out["uptrend"] = out["trend_on"] & (ema48_slope > 0)
    out["downtrend"] = out["trend_on"] & (ema48_slope < 0)
    return out


@dataclass(frozen=True)
class ArmConfig:
    key: str
    label: str
    adaptive: bool
    persistence_lookback: int
    spread_trigger: bool
    regime_gate: bool
    q: float


def build_score(frame: pd.DataFrame, cfg: ArmConfig) -> pd.Series:
    if cfg.adaptive:
        os_now = (frame["p_fast"] <= cfg.q) & (frame["p_slow"] <= cfg.q)
        ob_now = (frame["p_fast"] >= 1.0 - cfg.q) & (frame["p_slow"] >= 1.0 - cfg.q)
    else:
        os_now = (frame["fast_k"] <= 20.0) & (frame["slow_k"] <= 20.0)
        ob_now = (frame["fast_k"] >= 80.0) & (frame["slow_k"] >= 80.0)

    n = max(int(cfg.persistence_lookback), 1)
    os_recent = os_now.rolling(n, min_periods=1).max().astype(bool)
    ob_recent = ob_now.rolling(n, min_periods=1).max().astype(bool)

    if cfg.spread_trigger:
        spread = frame["spread"]
        spread_up = (spread.shift(1) < 0) & (spread >= 0)
        spread_down = (spread.shift(1) > 0) & (spread <= 0)
        long_trigger = os_recent & spread_up
        short_trigger = ob_recent & spread_down
    else:
        long_trigger = os_now if n <= 1 else os_recent & os_now
        short_trigger = ob_now if n <= 1 else ob_recent & ob_now

    if cfg.regime_gate:
        fade = ~frame["trend_on"]
        long_final = (fade & long_trigger) | (frame["uptrend"] & long_trigger)
        short_final = (fade & short_trigger) | (frame["downtrend"] & short_trigger)
    else:
        long_final = long_trigger
        short_final = short_trigger

    score = pd.Series(np.nan, index=frame.index)
    score[long_final.fillna(False)] = 1.0
    score[short_final.fillna(False) & ~long_final.fillna(False)] = -1.0
    return score


ARMS = [
    ArmConfig("a_original", "원안 (고정 80/20, 트리거 없음)", adaptive=False, persistence_lookback=1, spread_trigger=False, regime_gate=False, q=0.10),
    ArmConfig("b_full", "제안 설계 (full)", adaptive=True, persistence_lookback=6, spread_trigger=True, regime_gate=True, q=0.10),
    ArmConfig("c_no_adaptive", "b - 적응형 분위수 제거", adaptive=False, persistence_lookback=6, spread_trigger=True, regime_gate=True, q=0.10),
    ArmConfig("d_no_spread_trigger", "b - 스프레드 트리거 제거", adaptive=True, persistence_lookback=6, spread_trigger=False, regime_gate=True, q=0.10),
    ArmConfig("e_no_regime_gate", "b - 레짐 게이트 제거", adaptive=True, persistence_lookback=6, spread_trigger=True, regime_gate=False, q=0.10),
]


def run_window(frame: pd.DataFrame, score: pd.Series, *, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    ts = frame["timestamp"]
    eligible = purged_decision_mask(ts, start=start, end=end, horizon_bars=HORIZON_BARS)
    has_score = score.notna().to_numpy()
    mask = eligible & has_score
    decision_indices = np.flatnonzero(mask)

    tp_moves = (TP_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices]
    sl_moves = (SL_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices]
    scores = score.to_numpy()[decision_indices]

    result = simulate_single_position(
        timestamps=ts,
        open_px=frame["open"].to_numpy(),
        high=frame["high"].to_numpy(),
        low=frame["low"].to_numpy(),
        close=frame["close"].to_numpy(),
        decision_indices=decision_indices,
        scores=scores,
        tp_moves=tp_moves,
        sl_moves=sl_moves,
        upper_threshold=0.5,
        lower_threshold=-0.5,
        horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION,
        leverage=LEVERAGE,
        roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )

    window_mask = (ts >= start) & (ts <= end)
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
    else:
        win_rate, profit_factor = float("nan"), float("nan")

    day_returns = periodic_returns(equity_window, BARS_PER_DAY)
    sr = sharpe(day_returns) if day_returns.size else float("nan")

    close = frame["close"].to_numpy()
    p0, p1 = float(close[window_idx[0]]), float(close[window_idx[-1]]) if len(window_idx) else (float("nan"), float("nan"))
    always_long = p1 / p0 - 1.0
    always_short = p0 / p1 - 1.0

    return {
        "n_trades": n_trades,
        "total_return": total_return,
        "mdd": mdd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe_daily": sr,
        "skipped_while_open": int(result.skipped_while_open),
        "always_long_return": always_long,
        "always_short_return": always_short,
        "beats_benchmark": bool(total_return > max(always_long, always_short)),
        "ledger": ledger,
        "equity_window": equity_window,
    }


def selfcheck() -> None:
    """G0: mechanics sanity before trusting any market-data number."""
    r = np.array([-50.0])  # %R domain is [-100, 0]
    fast_k_test = 100.0 + r
    assert abs(fast_k_test[0] - 50.0) < 1e-9, "fastK = 100 + %R identity broke"

    ts = pd.date_range("2026-01-01", periods=10, freq="5min")
    open_px = np.array([100, 100, 100, 100, 106, 106, 106, 106, 106, 106], dtype=np.float64)
    high = open_px + 0.1
    low = open_px - 0.1
    high[4] = 106.0
    close = open_px.copy()
    result = simulate_single_position(
        timestamps=ts, open_px=open_px, high=high, low=low, close=close,
        decision_indices=np.array([2]), scores=np.array([1.0]),
        tp_moves=np.array([0.05]), sl_moves=np.array([0.02]),
        upper_threshold=0.5, lower_threshold=-0.5, horizon_bars=5,
        margin_fraction=0.3, leverage=3.0, roundtrip_cost_rate=0.0,
    )
    assert len(result.ledger) == 1, "expected exactly one synthetic trade"
    assert result.ledger.iloc[0]["reason"] == "tp", "synthetic guaranteed-TP trade did not resolve as tp"
    expected_return = 0.05 * 0.9
    assert abs(result.ledger.iloc[0]["trade_return"] - expected_return) < 1e-9, "notional/PnL math broke"
    print("G0 selfcheck: PASS")


def main() -> None:
    selfcheck()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_frame()
    data_max = raw["timestamp"].max()
    oos_end = min(OOS_END_TARGET, data_max)
    if oos_end != OOS_END_TARGET:
        print(f"NOTE: OOS end truncated from {OOS_END_TARGET.date()} to {oos_end.date()} (data ends {data_max}).")

    frame = compute_indicators(raw)

    val_results = {}
    for cfg in ARMS:
        score = build_score(frame, cfg)
        res = run_window(frame, score, start=VAL_START, end=VAL_END)
        val_results[cfg.key] = res
        print(f"[VAL] {cfg.key:20s} trades={res['n_trades']:4d} ret={res['total_return']*100:7.2f}% "
              f"mdd={res['mdd']*100:7.2f}% pf={res['profit_factor']:.2f} sharpe_d={res['sharpe_daily']:.3f} "
              f"benchmark(long/short)={res['always_long_return']*100:.2f}%/{res['always_short_return']*100:.2f}% "
              f"beats_bench={res['beats_benchmark']}")

    best_key = max(
        (k for k in val_results if val_results[k]["n_trades"] >= 5),
        key=lambda k: val_results[k]["total_return"] - max(val_results[k]["always_long_return"], val_results[k]["always_short_return"]),
        default="b_full",
    )
    best_cfg = next(c for c in ARMS if c.key == best_key)
    print(f"\nVAL-selected arm for single OOS look: {best_key} ({best_cfg.label})")

    score = build_score(frame, best_cfg)
    oos_res = run_window(frame, score, start=OOS_START, end=oos_end)
    print(f"[OOS] {best_key:20s} trades={oos_res['n_trades']:4d} ret={oos_res['total_return']*100:7.2f}% "
          f"mdd={oos_res['mdd']*100:7.2f}% pf={oos_res['profit_factor']:.2f} sharpe_d={oos_res['sharpe_daily']:.3f} "
          f"benchmark(long/short)={oos_res['always_long_return']*100:.2f}%/{oos_res['always_short_return']*100:.2f}% "
          f"beats_bench={oos_res['beats_benchmark']}")

    report = {
        "name": "eth_slowk_williamsr_persistence_confluence_20260814",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "val_window": [str(VAL_START.date()), str(VAL_END.date())],
        "oos_window": [str(OOS_START.date()), str(oos_end.date())],
        "oos_window_truncated_from_default": oos_end != OOS_END_TARGET,
        "params": {
            "stoch_n": STOCH_N, "slowk_smooth": SLOWK_SMOOTH, "adx_n": ADX_N, "ema_n": EMA_N,
            "atr_n": ATR_N, "adx_trend_min": ADX_TREND_MIN, "tp_atr_mult": TP_ATR_MULT,
            "sl_atr_mult": SL_ATR_MULT, "horizon_bars": HORIZON_BARS, "leverage": LEVERAGE,
            "margin_fraction": MARGIN_FRACTION, "roundtrip_cost_rate": ROUNDTRIP_COST_RATE,
            "percentile_window": 864,
        },
        "val_selected_arm": best_key,
        "val": {k: {kk: vv for kk, vv in v.items() if kk not in ("ledger", "equity_window")} for k, v in val_results.items()},
        "oos": {kk: vv for kk, vv in oos_res.items() if kk not in ("ledger", "equity_window")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    for k, v in val_results.items():
        v["ledger"].to_csv(OUT_DIR / f"val_ledger_{k}.csv", index=False)
    oos_res["ledger"].to_csv(OUT_DIR / f"oos_ledger_{best_key}.csv", index=False)
    print(f"\nWrote report + ledgers to {OUT_DIR}")


if __name__ == "__main__":
    main()
