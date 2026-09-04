#!/usr/bin/env python3
"""User asked directly: does the V_REBOUND TabPFN signal have an economic EDGE (not just
statistical/classification edge), enough to promote it from "model indicator" to "evidence
signal" tier -- the user's own stated bar for that promotion is "직접 매매에 도움". This repo's
established pattern (top-6 confluence, dalton_rule2_balance_edge, funding_oscillator_combo, liq_
direction+sweep combo -- see eth_liquidation_direction_signal detail text) is that real
statistical/informational edge FAILS an actual cost-gated trading test far more often than not;
this has never been tested for V_REBOUND, only classification AUC/accuracy (a different question).

Uses the SAME causal walk-forward engine as every other cost-gate script in this repo
(core.causal_futures_backtest.simulate_single_position/purged_decision_mask) -- not a new/
bespoke backtest mechanism. TP/SL are NOT hand-tuned to flatter this signal: TP=1.5xATR is the
label's OWN already-fixed V_REBOUND_ATR_MULT (pre-registered by the label definition itself, not
chosen for this test), SL=1.0xATR and LEVERAGE=3.0/MARGIN_FRACTION=0.30 are reused verbatim from
this repo's other cost-gate scripts (backtest_eth_dalton_rule2_balance_edge_costgate_20260815.py),
HORIZON_BARS=6 matches the label's own 30-minute forecast window (using the generic 48-bar/4h
default here would test a claim the model was never validated on).

Only VAL (2025-09-01..12-31) and OOS (2026-01-01..03-31) are used -- both already reused many
times this session for feature/model decisions, so reusing them again for a DIFFERENT question
(economic viability, not feature selection) is disclosed, not hidden. The 2026-04-01+ RESERVED
holdout is deliberately NOT touched here (already declared spent in
eth_liquidity_sweep_v_rebound_feature_plan_20260829.md) -- if this passes, a genuinely fresh
economic confirmation would need new data accumulated after 2026-08-29, not this window.

score construction: EVERY sweep is traded (no confidence threshold beyond the model's own 0.5
decision boundary, matching exactly what the validated AUC/accuracy numbers measure) -- LONG if
(downside sweep AND call=rebound) or (upside sweep AND call=continuation), SHORT otherwise.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

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
SEED = 20260829  # single seed -- this is an economic gate, not a seed-stability re-check;
                   # TabPFN's own seed std (~0.0005-0.0008 AUC) is negligible for this purpose

TP_ATR_MULT = 1.5   # == V_REBOUND_ATR_MULT, the label's own pre-registered magnitude, not tuned here
SL_ATR_MULT = 1.0   # reused verbatim from backtest_eth_dalton_rule2_balance_edge_costgate_20260815.py
HORIZON_BARS = 6    # == LOOKAHEAD_BARS, the label's own 30-minute window
LEVERAGE = 3.0
MARGIN_FRACTION = 0.30
ROUNDTRIP_COST_RATE = 0.001  # 10bp, this repo's standard
COST_SWEEP_BP = [0, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50]


def log(msg: str) -> None:
    print(f"[v_rebound_costgate] {msg}", flush=True)


def load_sweep_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_costgate_20260829", SWEEP_IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_window(raw: pd.DataFrame, scored: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp,
                roundtrip_cost: float) -> dict[str, Any]:
    ts = raw["timestamp"]
    eligible = purged_decision_mask(ts, start=start, end=end, horizon_bars=HORIZON_BARS)
    eligible_idx = set(np.flatnonzero(eligible).tolist())
    sub = scored[scored["candidate_index"].isin(eligible_idx)].sort_values("candidate_index")
    decision_indices = sub["candidate_index"].to_numpy()

    result = simulate_single_position(
        timestamps=ts, open_px=raw["open"].to_numpy(), high=raw["high"].to_numpy(),
        low=raw["low"].to_numpy(), close=raw["close"].to_numpy(),
        decision_indices=decision_indices, scores=sub["score"].to_numpy(),
        tp_moves=sub["tp_move"].to_numpy(), sl_moves=sub["sl_move"].to_numpy(),
        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=roundtrip_cost,
    )
    ledger = result.ledger
    total_return = float(result.equity[-1] - 1.0) if len(result.equity) else float("nan")
    n_trades = int(len(ledger))
    wr = float((ledger["price_move"] * ledger["side"] > 0).mean()) if n_trades else float("nan")

    win_mask = (ts >= start) & (ts <= end)
    win_idx = np.flatnonzero(win_mask.to_numpy())
    if len(win_idx):
        p0, p1 = float(raw["close"].iloc[win_idx[0]]), float(raw["close"].iloc[win_idx[-1]])
        always_long, always_short = p1 / p0 - 1.0, p0 / p1 - 1.0
    else:
        always_long, always_short = float("nan"), float("nan")

    return {
        "n_trades": n_trades, "wr": wr, "total_return": total_return,
        "always_long_return": always_long, "always_short_return": always_short,
        "beats_benchmark": bool(total_return > max(always_long, always_short))
        if np.isfinite(always_long) and np.isfinite(always_short) else None,
    }


def find_breakeven_bp(raw: pd.DataFrame, scored: pd.DataFrame, *, start, end) -> float | None:
    lo, hi = 0.0, 0.02
    r_lo = run_window(raw, scored, start=start, end=end, roundtrip_cost=lo)["total_return"]
    r_hi = run_window(raw, scored, start=start, end=end, roundtrip_cost=hi)["total_return"]
    if not np.isfinite(r_lo) or r_lo <= 0:
        return 0.0
    if r_hi > 0:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        r_mid = run_window(raw, scored, start=start, end=end, roundtrip_cost=mid)["total_return"]
        if r_mid > 0:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0 * 10000.0)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
    log(f"train n={len(train)}, eval(VAL+OOS) n={len(evalset)}")

    clf = TabPFNClassifier(device="cuda", random_state=SEED)
    clf.fit(train[FEATURES], train["label"].to_numpy())
    evalset["proba_rebound"] = clf.predict_proba(evalset[FEATURES])[:, 1]
    evalset["call_rebound"] = evalset["proba_rebound"] >= 0.5

    # score: LONG(+1) if (downside sweep AND rebound-call) or (upside sweep AND continuation-call)
    is_down = evalset["is_downside"] == 1
    long_signal = (is_down & evalset["call_rebound"]) | (~is_down & ~evalset["call_rebound"])
    evalset["score"] = np.where(long_signal, 1.0, -1.0)
    evalset["tp_move"] = TP_ATR_MULT * evalset["atr"] / raw["close"].iloc[evalset["candidate_index"]].to_numpy()
    evalset["sl_move"] = SL_ATR_MULT * evalset["atr"] / raw["close"].iloc[evalset["candidate_index"]].to_numpy()

    report: dict[str, Any] = {
        "signal": "eth_sweep_v_rebound TabPFN (Tier0+rsi), score=direction implied by call vs sweep side",
        "tp_atr_mult": TP_ATR_MULT, "sl_atr_mult": SL_ATR_MULT, "horizon_bars": HORIZON_BARS,
        "leverage": LEVERAGE, "margin_fraction": MARGIN_FRACTION,
        "roundtrip_cost_rate_standard": ROUNDTRIP_COST_RATE,
        "note": "VAL+OOS only -- 2026-04-01+ reserved holdout deliberately excluded (already spent)",
        "windows": {},
    }
    log(f"{'window':<6} {'n_trades':>8} {'wr':>7} {'return':>10} {'long':>9} {'short':>9} {'beats_bm':>9}  breakeven_bp")
    for wname, (start, end) in {"val": (VAL_START, VAL_END), "oos": (OOS_START, OOS_END)}.items():
        res_std = run_window(raw, evalset, start=start, end=end, roundtrip_cost=ROUNDTRIP_COST_RATE)
        be = find_breakeven_bp(raw, evalset, start=start, end=end)
        be_str = f"{be:.1f}bp" if be is not None else ">200bp"
        cost_curve = {bp: run_window(raw, evalset, start=start, end=end, roundtrip_cost=bp / 10000.0)["total_return"]
                      for bp in COST_SWEEP_BP}
        report["windows"][wname] = {**res_std, "breakeven_bp": be, "cost_curve_bp_to_return": cost_curve}
        wr_pct = res_std["wr"] * 100 if np.isfinite(res_std["wr"]) else float("nan")
        log(f"{wname:<6} {res_std['n_trades']:>8d} {wr_pct:>6.1f}% {res_std['total_return']*100:>9.2f}% "
            f"{res_std['always_long_return']*100:>8.2f}% {res_std['always_short_return']*100:>8.2f}%  "
            f"{str(res_std['beats_benchmark']):>9}  {be_str}")

    wins = sum(1 for w in report["windows"].values() if w["beats_benchmark"])
    log(f"SUMMARY: beats always_long/always_short in {wins}/2 windows at standard 10bp roundtrip cost")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
