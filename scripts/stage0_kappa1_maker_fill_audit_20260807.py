"""Kappa1 Stage 0 — maker fill-rate audit in BTC event-gate windows vs baseline.

Design doc: docs/btc_kappa1_invariant_composite_policy_design_20260807.md (risk R1).
Kill gate: event-window fill rate < 30% within 15 minutes -> Kappa1 dies here.

Procedure (conservative throughout, 1m OHLC only, no BBO):
  - Decision: a gate-fired 5m bar t (gate inputs causal per the 2026-08-04 prototype).
    Decision information is complete at t+5min.
  - Direction: contrarian to the mean net_taker_ratio of the last five 1m bars at or
    before decision close (net selling -> BUY limit, net buying -> SELL limit).
  - Order: post-only limit at the decision-time 1m close. If the next 1m open already
    crosses the limit (order would execute as taker / be rejected post-only), count as
    NO FILL. Filled only if a later 1m bar trades strictly THROUGH the limit
    (buy: low < limit; sell: high > limit) within the cancel window.
  - Cancel windows: 5 / 15 / 30 minutes.
  - Baseline: up to 20,000 random non-event 5m bars, identical procedure.
  - Diagnostic markout (not a gate): signed close-to-close move 15m and 60m after fill.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GATE_SERIES = ROOT / "tmp/research_20260804/btc_event_gate_prototype/gate_series.csv"
M1_PATH = ROOT / "data/training_features_1m_causal_btc.csv"
OUT = ROOT / "docs/experiments/btc_kappa1_stage0_maker_fill_audit_20260807.json"

CANCEL_WINDOWS_MIN = [5, 15, 30]
BASELINE_N = 20000
RNG_SEED = 914237
FILL_KILL_THRESHOLD = 0.30  # event-window fill rate at 15m below this -> kill


def load_1m() -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "net_taker_ratio"]
    frame = pd.read_csv(M1_PATH, usecols=cols)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def simulate(events: pd.DatetimeIndex, m1: pd.DataFrame, limit_mode: str = "stale_close") -> dict:
    ts = m1["timestamp"].to_numpy()
    open_ = m1["open"].to_numpy()
    high = m1["high"].to_numpy()
    low = m1["low"].to_numpy()
    close = m1["close"].to_numpy()
    flow = m1["net_taker_ratio"].to_numpy()
    n = len(m1)
    max_window = max(CANCEL_WINDOWS_MIN)

    attempted = 0
    crossed_at_open = 0
    fills = {k: 0 for k in CANCEL_WINDOWS_MIN}
    markouts_15, markouts_60 = [], []

    for event_ts in events:
        decision_close_time = event_ts + pd.Timedelta(minutes=4)
        pos = np.searchsorted(ts, np.datetime64(decision_close_time), side="right") - 1
        if pos < 5 or pos + max_window + 60 >= n:
            continue
        if ts[pos] != np.datetime64(decision_close_time):
            continue  # missing decision bar -> skip event entirely
        recent_flow = np.nanmean(flow[pos - 4:pos + 1])
        if not np.isfinite(recent_flow):
            continue
        side = 1 if recent_flow < 0 else -1  # contrarian: sell pressure -> buy
        first = pos + 1
        attempted += 1
        if limit_mode == "stale_close":
            # v1 (pre-registered): rest at decision close; cross-at-open = no fill
            limit = close[pos]
            if (side == 1 and open_[first] <= limit) or (side == -1 and open_[first] >= limit):
                crossed_at_open += 1
                continue  # post-only would cross -> conservative no-fill
        else:
            # v2 (amended post-hoc, still conservative): order arrives at the start of
            # the next 1m bar and rests AT the current touch (that bar's open); fill
            # still requires strict trade-through, no queue-position credit.
            limit = open_[first]
        fill_offset = None
        for j in range(first, min(first + max_window, n)):
            through = low[j] < limit if side == 1 else high[j] > limit
            if through:
                fill_offset = j - first + 1  # minutes elapsed
                break
        if fill_offset is None:
            continue
        for k in CANCEL_WINDOWS_MIN:
            if fill_offset <= k:
                fills[k] += 1
        fill_idx = first + fill_offset - 1
        if fill_idx + 60 < n:
            markouts_15.append(side * (close[fill_idx + 15] / limit - 1))
            markouts_60.append(side * (close[fill_idx + 60] / limit - 1))

    result = {
        "events_attempted": attempted,
        "crossed_at_open_noFill": crossed_at_open,
        "fill_rate": {f"{k}m": (fills[k] / attempted if attempted else float("nan"))
                      for k in CANCEL_WINDOWS_MIN},
    }
    for name, arr in (("markout_15m", markouts_15), ("markout_60m", markouts_60)):
        if arr:
            a = np.array(arr)
            result[name] = {"mean_bps": float(a.mean() * 1e4),
                            "t_stat": float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))),
                            "n": int(len(a))}
        else:
            result[name] = None
    return result


def main() -> None:
    gate = pd.read_csv(GATE_SERIES)
    gate["timestamp"] = pd.to_datetime(gate["timestamp"])
    m1 = load_1m()
    m1_start, m1_end = m1["timestamp"].iloc[0], m1["timestamp"].iloc[-1]
    usable = gate[(gate["timestamp"] >= m1_start) &
                  (gate["timestamp"] <= m1_end - pd.Timedelta(minutes=90))]
    event_ts = pd.DatetimeIndex(usable.loc[usable["gate_fired"], "timestamp"])

    rng = np.random.default_rng(RNG_SEED)
    non_event = usable.loc[~usable["gate_fired"], "timestamp"]
    baseline_ts = pd.DatetimeIndex(
        non_event.sample(n=min(BASELINE_N, len(non_event)), random_state=RNG_SEED))

    print(f"event bars usable: {len(event_ts)}, baseline sample: {len(baseline_ts)}")
    report = {
        "design": "docs/btc_kappa1_invariant_composite_policy_design_20260807.md",
        "gate_series": str(GATE_SERIES.relative_to(ROOT)),
        "m1_coverage": [str(m1_start), str(m1_end)],
        "kill_threshold_fill_15m": FILL_KILL_THRESHOLD,
        "rules": {},
    }
    for mode, label in (("stale_close", "v1_preregistered"), ("at_touch", "v2_amended_posthoc")):
        event_result = simulate(event_ts, m1, limit_mode=mode)
        baseline_result = simulate(baseline_ts, m1, limit_mode=mode)
        fill_15 = event_result["fill_rate"]["15m"]
        verdict = "PASS" if fill_15 >= FILL_KILL_THRESHOLD else "KILL"
        report["rules"][label] = {
            "fill_rule": ("post-only resting at decision 1m close; cross-at-open = no fill; strict trade-through"
                          if mode == "stale_close" else
                          "post-only resting at next 1m open (the touch at arrival); strict trade-through; no queue credit"),
            "event_windows": event_result, "baseline_random": baseline_result,
            "verdict": verdict,
        }
        print(f"{label}: event 15m fill rate = {fill_15:.1%} -> {verdict}")
    report["note"] = ("v1 conflated post-only placement rejection (75% cross-at-open) with resting-order "
                      "non-fill; v2 re-prices to the touch at arrival, the behaviour of a real post-only "
                      "engine, and was adopted AFTER seeing v1 -- recorded as a post-hoc amendment.")
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
