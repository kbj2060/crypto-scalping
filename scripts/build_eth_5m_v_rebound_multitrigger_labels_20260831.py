#!/usr/bin/env python3
"""Multi-trigger V자반등 label (candidate pool widened from liquidity_sweep-only to 7 triggers OR'd
together). LABEL CONSTRUCTION ONLY -- no Tier0 feature building, no TabPFN training, no economic
gate. Per project convention (feedback_visual_verification_chart_gate_explain_before_proceed),
this must be followed by a 20-example visual verification the user reviews BEFORE any further
step (feature building / TabPFN retrain) proceeds.

Background (see memory eth_v_rebound_sweep_gated_recall_gap_20260831 for the full 5-stage
diagnostic that led here): the live V자반등 signal (scripts/live_eth_sweep_v_rebound_signal_
20260829.py) gates candidate detection 100% on liquidity_sweep. A 90-day recall-gap audit found
sweep alone catches only ~25.5% of an estimated true population of qualifying V-shaped reversals;
adding this dashboard's other 5 evidence-signal triggers (each independently ~12-21% standalone
hit-rate, on par with sweep's 16.4%) raises coverage to ~50.8%; adding a 7th, precondition-free
"local extreme" trigger closes the rest. This script builds the UNION of all 7 as the new
candidate pool.

OUTCOME/label formula is REUSED VERBATIM, NOT REDESIGNED -- this is the exact same v7b formula
that scored sweep-triggered candidates (docs/experiments/eth_liquidity_sweep_v_rebound_feature_
plan_20260829.md lines 1065-1075, and scripts/research_eth_v_rebound_sweep_gate_recall_check_
90d_20260831.py::realized_outcome, imported here unchanged):
  V자반등(1): within 30min (6 bars), best CLOSE reaches >=1.5x pre-event ATR(14) from the event
    extreme, AND over the full 60min (12 bars) window, giveback_ratio <= 0.20 (peak = best
    HIGH/LOW anywhere in the 60min window, not close-only).
  지지/횡보(0): 30min best-close move never reaches 1.0x pre-event ATR.
  else: excluded/ambiguous (not labeled either way, same as v7b).
Only the CANDIDATE POOL (which bars even get scored) changes -- trigger and label are fully
decoupled axes, matching how sweep+v7b already worked; widening the trigger set does not touch
this formula.

The 9 triggers (OR'd, downside=candidate for an upward rebound / upside=mirror):
  1. liquidity_sweep       -- wick pierces causal 48-bar swing level, close reclaims (existing)
  2. taker_delta_z_climax  -- net taker buy/sell volume z-score beyond +-2.0 (existing signal)
  3. short_term_return_z   -- 3-bar (15m) return z-score beyond +-2.5 (existing signal)
  4. orthogonal_combo      -- oscillator-extreme + orderflow-climax combo (existing signal)
  5. smt_divergence        -- ETH breaks 48-bar swing, BTC does not confirm (existing signal;
     needs BTC klines, silently produces no smt_divergence fires past BTC's local data coverage)
  6. fib_extension_exhaustion -- price in the 27.2-61.8% fib-extension zone past a swing (existing)
  7. local_extreme         -- no precondition: bar is the highest/lowest in a +-30min (+-6 bar)
     window. 1-6 reuse compute_signals() (scripts/live_evidence_signal_dashboard_20260823.py)
     verbatim, verified byte-for-byte against each signal's original source script.
  8. demarker_extreme      -- DeMarker(14) >=0.90 (top) / <=0.10 (bottom). Homer CANDIDATE POOL
     signal, NOT yet deployed/TabPFN-confirmed for its own task (2026-08-31 user decision: add
     its raw trigger to this label now anyway, since the trigger threshold itself -- unlike the
     signal's own outcome-K calibration, which is still being tuned -- is fixed and reused
     unchanged from research_eth_kalman_demarker_gridscreen_20260831.py's own screen_signal()
     call, i.e. compute_demarker() imported verbatim from research_eth_demarker_evidence_signal_
     lift_check_20260831.py).
  9. kalman_deviation_meanrev -- (close-kalman_level)/kalman_level, 288-bar z-score, >=2.0 (top)
     / <=-2.0 (bottom). Same Homer candidate-pool caveat and same-day inclusion decision as #8;
     kalman_level_and_velocity()/rolling_zscore() imported verbatim from research_eth_candidate_
     pool_raw_lift_check_20260831.py.

Data: full local history, binance_data/klines/{ETHUSDT,BTCUSDT}/*-5m-api.csv (no live-API gap
fill -- matches v1-v4's own build_eth_5m_liquidity_sweep_v_rebound_labels_20260829.py convention
of a frozen static snapshot, not a live-updated one). ETH covers through ~2026-08-28, BTC through
~2026-08-20 (staler) -- smt_divergence simply cannot fire in the ETH-only trailing ~8 days; all
other triggers are unaffected there.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker  # noqa: E402
from research_eth_candidate_pool_raw_lift_check_20260831 import (  # noqa: E402
    kalman_level_and_velocity, rolling_zscore,
)

RECALL_SCRIPT = ROOT / "scripts/research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py"
_spec = importlib.util.spec_from_file_location("recall_check_90d_20260831c", RECALL_SCRIPT)
_recall = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_recall)
realized_outcome = _recall.realized_outcome
load_impl = _recall.load_impl

ETH_LOCAL_CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_LOCAL_CSV = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
OUT_DIR = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831"

NAMED_SIGNALS = ["taker_delta_z_climax", "short_term_return_z", "orthogonal_combo",
                 "smt_divergence", "fib_extension_exhaustion"]
LOCAL_EXTREME_W = 6  # +-30min, same window used throughout the recall-gap diagnostic


def load_local(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def main() -> None:
    t0 = time.time()
    impl = load_impl()
    eth = load_local(ETH_LOCAL_CSV)
    btc = load_local(BTC_LOCAL_CSV)
    print(f"ETH {len(eth)}bars {eth['timestamp'].iloc[0]} ~ {eth['timestamp'].iloc[-1]}")
    print(f"BTC {len(btc)}bars {btc['timestamp'].iloc[0]} ~ {btc['timestamp'].iloc[-1]} "
          f"(smt_divergence only fires within BTC coverage)")

    causal = impl.add_causal_columns(eth[["timestamp", "open", "high", "low", "close"]].copy())
    sig = compute_signals(eth, btc_df=btc, funding_df=None).reset_index(drop=True)
    sig["atr"] = causal["atr"].to_numpy()
    n = len(sig)

    low, high, close = sig["low"].to_numpy(), sig["high"].to_numpy(), sig["close"].to_numpy()
    level_low, level_high = causal["sweep_level_low"].to_numpy(), causal["sweep_level_high"].to_numpy()
    is_down_sweep = np.where(np.isnan(level_low), False, (low < level_low) & (close > level_low))
    is_up_sweep = np.where(np.isnan(level_high), False, (high > level_high) & (close < level_high))

    # trigger membership per bar/direction, kept separate (not just OR'd) so each candidate row
    # can record WHICH of the 7 fired -- needed for the stratified visual-verification sample.
    down_triggers = {"liquidity_sweep": is_down_sweep}
    up_triggers = {"liquidity_sweep": is_up_sweep}
    for name in NAMED_SIGNALS:
        down_triggers[name] = sig[f"bottom_{name}"].to_numpy()
        up_triggers[name] = sig[f"top_{name}"].to_numpy()

    # Homer candidate-pool triggers (demarker_extreme, kalman_deviation_meanrev) -- formulas/
    # thresholds reused verbatim from research_eth_kalman_demarker_gridscreen_20260831.py's own
    # screen_signal() calls, not re-derived.
    dem = compute_demarker(sig["high"], sig["low"])
    down_triggers["demarker_extreme"] = (dem <= 0.10).fillna(False).to_numpy()
    up_triggers["demarker_extreme"] = (dem >= 0.90).fillna(False).to_numpy()

    kalman_levels, _ = kalman_level_and_velocity(close)
    kalman_dev = pd.Series((close - kalman_levels) / kalman_levels, index=sig.index)
    kalman_dev_z = rolling_zscore(kalman_dev)
    down_triggers["kalman_deviation_meanrev"] = (kalman_dev_z <= -2.0).fillna(False).to_numpy()
    up_triggers["kalman_deviation_meanrev"] = (kalman_dev_z >= 2.0).fillna(False).to_numpy()

    W = LOCAL_EXTREME_W
    local_low = np.zeros(n, dtype=bool)
    local_high = np.zeros(n, dtype=bool)
    for i in range(W, n - W):
        seg_lo = low[i - W:i + W + 1]
        seg_hi = high[i - W:i + W + 1]
        if low[i] == seg_lo.min():
            local_low[i] = True
        if high[i] == seg_hi.max():
            local_high[i] = True
    down_triggers["local_extreme"] = local_low
    up_triggers["local_extreme"] = local_high

    def build_side(triggers: dict, is_down: bool) -> list[dict]:
        any_fire = np.zeros(n, dtype=bool)
        for arr in triggers.values():
            any_fire |= arr
        rows = []
        for i in np.flatnonzero(any_fire):
            o = realized_outcome(sig, int(i), is_down)
            if o is None or o["partial_window"]:
                continue
            fired = sorted(name for name, arr in triggers.items() if arr[i])
            rows.append({
                "idx": int(i),
                "timestamp": sig["timestamp"].iloc[i].isoformat(),
                "direction": "downside" if is_down else "upside",
                "triggers": ",".join(fired),
                "n_triggers": len(fired),
                **o,
            })
        return rows

    print("스캔 중...")
    rows = build_side(down_triggers, True) + build_side(up_triggers, False)
    labels = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label_path = OUT_DIR / "eth_5m_v_rebound_multitrigger_labels.csv"
    labels.to_csv(label_path, index=False)

    outcome_counts = labels["outcome"].value_counts().to_dict()
    by_trigger_v_rebound = {}
    for name in list(down_triggers.keys()):
        mask = labels["triggers"].str.contains(name)
        sub = labels[mask]
        by_trigger_v_rebound[name] = {
            "n_candidates": int(len(sub)),
            "n_v_rebound": int((sub["outcome"] == "V자반등").sum()),
            "rate": None if not len(sub) else round(float((sub["outcome"] == "V자반등").mean()), 4),
        }

    report = {
        "label_contract": {"V자반등": "outcome==V자반등", "지지/횡보": "outcome==지지/횡보", "제외": "outcome==애매(제외권)"},
        "outcome_formula": "v7b, reused verbatim from research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py::realized_outcome",
        "triggers": list(down_triggers.keys()),
        "eth_source": str(ETH_LOCAL_CSV.relative_to(ROOT)),
        "btc_source": str(BTC_LOCAL_CSV.relative_to(ROOT)),
        "eth_period": {"start": str(eth["timestamp"].min()), "end": str(eth["timestamp"].max())},
        "btc_period": {"start": str(btc["timestamp"].min()), "end": str(btc["timestamp"].max())},
        "total_candidates": int(len(labels)),
        "outcome_counts": outcome_counts,
        "outcome_rate": {k: round(v / len(labels), 4) for k, v in outcome_counts.items()} if len(labels) else {},
        "by_trigger": by_trigger_v_rebound,
        "future_features_used_for_labels": True,
        "output_labels": str(label_path),
        "runtime_sec": round(time.time() - t0, 1),
        "NEXT_STEP_GATE": "label construction only -- per feedback_visual_verification_chart_gate_explain_before_proceed, "
                           "a 20-example visual verification chart must be shown and explained, and the user must "
                           "explicitly approve, before any feature-building/TabPFN-training step proceeds.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n총 후보: {len(labels)}건")
    for k, v in outcome_counts.items():
        print(f"  {k}: {v}건 ({v/len(labels)*100:.1f}%)")
    print("\n트리거별 (중복포함, 한 후보가 여러 트리거에 걸릴 수 있음):")
    for name, s in by_trigger_v_rebound.items():
        print(f"  {name:22s}: 후보 {s['n_candidates']:6d}건, V자반등 {s['n_v_rebound']:5d}건 ({s['rate']*100:.1f}%)")
    print(f"\n산출물: {label_path}, {OUT_DIR}/report.json")
    print(f"실행시간: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
