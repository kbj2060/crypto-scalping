#!/usr/bin/env python3
"""Diagnostic requested by the user after the cost-gate FAIL (win rate ~49% despite 61.6%
classification accuracy): per the external literature review's recommendation (Grinold's Law /
Transfer Coefficient), split realized trades into "intrabar-stopped but the classifier's call was
actually correct" vs "genuinely wrong calls that also lost" -- if the former dominates, the
problem is the execution/translation layer (intrabar SL vs the label's sustained-close
confirmation), not the classifier itself. Also checks the second diagnosed issue directly: does
the "continuation" (0-class) side perform worse than the "rebound" (1-class) side, since the
0-class isn't really "opposite direction" (this repo's own reference_direction_quality_exit_label_
methodology finding: "no-signal" should be abstain, not an opposite-direction bet).

Reruns the exact same cost-gate backtest (same TP/SL/horizon/leverage/cost, same TabPFN fit) as
backtest_eth_sweep_v_rebound_tabpfn_costgate_20260829.py, but this time keeps and analyzes the
per-trade ledger instead of only aggregate metrics. No new hypotheses tested here -- purely
descriptive of what already happened in that backtest.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from tabpfn import TabPFNClassifier  # noqa: E402

TIER0_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv"
RSI_SOURCES = [ROOT / f"data/splits/year_oos/training_features_{y}.csv" for y in ("2024", "2025", "2026_rebuilt")]
SWEEP_IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_tabpfn_costgate_20260829"

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile",
]
FEATURES = TIER0 + ["rsi"]
SEED = 20260829
TP_ATR_MULT, SL_ATR_MULT, HORIZON_BARS = 1.5, 1.0, 6
LEVERAGE, MARGIN_FRACTION, ROUNDTRIP_COST_RATE = 3.0, 0.30, 0.001


def log(msg: str) -> None:
    print(f"[loss_diagnostic] {msg}", flush=True)


def load_sweep_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_diag_20260829", SWEEP_IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    sweep_impl = load_sweep_impl()
    raw = sweep_impl.load_5m(SOURCE)

    tier0 = pd.read_csv(TIER0_CSV)
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)
    frames = []
    for p in RSI_SOURCES:
        f = pd.read_csv(p, usecols=["timestamp", "rsi"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        frames.append(f)
    rsi = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
    df = tier0.merge(rsi, on="timestamp", how="left").dropna(subset=FEATURES + ["label"]).reset_index(drop=True)

    train = df.loc[df["timestamp"] < VAL_START]
    evalset = df.loc[(df["timestamp"] >= VAL_START) & (df["timestamp"] <= OOS_END)].copy()

    clf = TabPFNClassifier(device="cuda", random_state=SEED)
    clf.fit(train[FEATURES], train["label"].to_numpy())
    evalset["proba_rebound"] = clf.predict_proba(evalset[FEATURES])[:, 1]
    evalset["call_rebound"] = evalset["proba_rebound"] >= 0.5
    evalset["model_correct"] = evalset["call_rebound"] == (evalset["label"] == 1)

    is_down = evalset["is_downside"] == 1
    long_signal = (is_down & evalset["call_rebound"]) | (~is_down & ~evalset["call_rebound"])
    evalset["score"] = np.where(long_signal, 1.0, -1.0)
    evalset["bet_side"] = np.where(evalset["call_rebound"], "rebound_bet", "continuation_bet")
    price_at_event = raw["close"].iloc[evalset["candidate_index"]].to_numpy()
    evalset["tp_move"] = TP_ATR_MULT * evalset["atr"] / price_at_event
    evalset["sl_move"] = SL_ATR_MULT * evalset["atr"] / price_at_event

    ts = raw["timestamp"]
    all_ledgers = []
    for wname, (start, end) in {"val": (VAL_START, VAL_END), "oos": (OOS_START, OOS_END)}.items():
        eligible = purged_decision_mask(ts, start=start, end=end, horizon_bars=HORIZON_BARS)
        eligible_idx = set(np.flatnonzero(eligible).tolist())
        sub = evalset[evalset["candidate_index"].isin(eligible_idx)].sort_values("candidate_index")
        result = simulate_single_position(
            timestamps=ts, open_px=raw["open"].to_numpy(), high=raw["high"].to_numpy(),
            low=raw["low"].to_numpy(), close=raw["close"].to_numpy(),
            decision_indices=sub["candidate_index"].to_numpy(), scores=sub["score"].to_numpy(),
            tp_moves=sub["tp_move"].to_numpy(), sl_moves=sub["sl_move"].to_numpy(),
            upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=HORIZON_BARS,
            margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
        )
        ledger = result.ledger.copy()
        ledger["window"] = wname
        ledger = ledger.merge(
            evalset[["timestamp", "label", "model_correct", "bet_side", "is_downside"]],
            left_on="decision_timestamp", right_on="timestamp", how="left",
        )
        all_ledgers.append(ledger)

    ledger = pd.concat(all_ledgers, ignore_index=True)
    log(f"total trades: {len(ledger)}  (join check: {ledger['label'].isna().sum()} unmatched -- should be 0)")

    report = {}

    # --- diagnostic 1: SL-triggered trades, split by whether the classifier call was correct ---
    sl_trades = ledger[ledger["reason"] == "sl"]
    n_sl = len(sl_trades)
    n_sl_model_correct = int(sl_trades["model_correct"].sum())
    n_sl_model_wrong = n_sl - n_sl_model_correct
    report["diagnostic_1_sl_trades_by_model_correctness"] = {
        "n_sl_trades_total": int(n_sl),
        "n_sl_but_model_call_was_correct": n_sl_model_correct,
        "pct_of_sl_trades_where_model_was_actually_right": round(n_sl_model_correct / n_sl, 4) if n_sl else None,
        "n_sl_where_model_was_also_wrong": n_sl_model_wrong,
        "interpretation": (
            "high pct = translation failure (intrabar SL cut a correctly-called trade before the "
            "label's sustained-close outcome could play out) -- fixable via exit redesign. "
            "low pct = losses mostly reflect genuinely wrong model calls -- exit redesign won't help much."
        ),
    }

    # --- diagnostic 2: does the 'continuation_bet' side (0-class -> opposite-direction bet) ---
    #     underperform the 'rebound_bet' side (1-class -> same-direction bet)?
    by_side = {}
    for side_name, sub in ledger.groupby("bet_side"):
        wins = int((sub["price_move"] > 0).sum())
        n = len(sub)
        by_side[side_name] = {
            "n_trades": n, "win_rate": round(wins / n, 4) if n else None,
            "total_return_sum_of_trade_returns": round(float(sub["trade_return"].sum()), 4),
            "mean_trade_return": round(float(sub["trade_return"].mean()), 6) if n else None,
        }
    report["diagnostic_2_rebound_bet_vs_continuation_bet"] = by_side

    # --- diagnostic 3: overall reason breakdown + model accuracy on traded subset (sanity check) ---
    report["diagnostic_3_context"] = {
        "reason_counts": ledger["reason"].value_counts().to_dict(),
        "model_accuracy_on_traded_events": round(float(ledger["model_correct"].mean()), 4),
        "overall_win_rate": round(float((ledger["price_move"] > 0).mean()), 4),
    }

    # --- diagnostic 4: if we had only traded the rebound_bet side (abstain on continuation), ---
    #     what would win rate / return have looked like? (descriptive only, not a new backtest --
    #     just filtering the same ledger, no re-simulation of freed-up capital/non-overlap slots)
    rebound_only = ledger[ledger["bet_side"] == "rebound_bet"]
    report["diagnostic_4_rebound_only_subset_naive_readback"] = {
        "n_trades": int(len(rebound_only)),
        "win_rate": round(float((rebound_only["price_move"] > 0).mean()), 4) if len(rebound_only) else None,
        "sum_trade_returns": round(float(rebound_only["trade_return"].sum()), 4) if len(rebound_only) else None,
        "caveat": "naive readback of the SAME ledger, not a re-simulation -- abstaining on "
                  "continuation_bet would free up occupied-slot time for more rebound_bet trades "
                  "to fire (non-overlapping engine), so a real re-backtest would likely show MORE "
                  "rebound_bet trades than this subset, not just fewer total trades.",
    }

    print(json.dumps(report, indent=2, default=str))
    (OUT_DIR / "loss_diagnostic_report.json").write_text(json.dumps(report, indent=2, default=str))
    ledger.to_csv(OUT_DIR / "loss_diagnostic_ledger.csv", index=False)
    log(f"wrote {OUT_DIR / 'loss_diagnostic_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
