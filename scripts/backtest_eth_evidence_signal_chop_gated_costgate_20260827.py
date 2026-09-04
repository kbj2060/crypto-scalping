#!/usr/bin/env python3
"""Chop-gated cost-gate backtest for the 5 evidence-signal candidates whose 1h lift-vs-zigzag-pivot
improved specifically during chop regime (research_eth_evidence_signal_regime_chop_conditional_
20260827.py, 2026-08-27): orthogonal_combo/liquidity_sweep/smt_divergence/volume_wick_climax
(bottom, LONG-only test) and short_term_return_z (top, SHORT-only test). Tests whether gating each
signal to fire ONLY when regime=chop (via the live GBM3 classifier, scripts/live_regime_gbm3_
signal_20260826.py) turns a higher diagnostic lift number into an actual fresh-forward TP:SL edge.

Same engine/TP:SL/6-window convention as every sibling evidence-signal cost-gate script in this
lineage (backtest_eth_funding_oscillator_combo_costgate_20260825.py, backtest_eth_dalton_rule2_
balance_edge_costgate_20260815.py) -- not invented here: core.causal_futures_backtest.
simulate_single_position/purged_decision_mask, TP=1.6xATR, SL=1.0xATR, horizon=48bar, leverage=3x,
margin=30%, roundtrip cost=0.1%, same 6 pre-registered windows via eth_omega461_multiwindow_
confirmation_gate_20260814.WINDOW_DEFS.

Reports BOTH the chop-gated and ungated (= today's unconditional live formula) variant of each
signal side by side, so a pass/fail can be attributed to the chop gate itself rather than just
re-measuring the already-known-failed unconditional signal. Evidence-signal formulas are the exact
live ones (live_evidence_signal_dashboard_20260823.compute_signals(), including the 2026-08-27
orthogonal_combo/funding_oscillator_combo merge), not re-derived.

Regime: _with_raw_state12() applied directly to the SAME base_csv each WINDOW_DEFS entry already
uses (gate.sweep.BASE_2025/BASE_2026 = data/splits/year_oos/training_features_{2025,2026_rebuilt}.csv)
-- identical inputs/outputs to research_eth_evidence_signal_regime_chop_conditional_20260827.py's
regime computation, just computed per-base_csv here to match this engine's per-file convention.
CAVEAT (same as the diagnostic script): 2025q1..oos_q2 fall inside the regime model's own TRAIN
range (2024-01-01~2026-06-30) -- its chop/non-chop split there is in-sample, not a clean OOS split
of the classifier itself. This does not retroactively validate/invalidate the classifier -- it only
means "which bars get gated in" should be read as best-available, not adversarially clean.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false,
future_rows_used_for_entry=false. No training, no GPU (the pre-trained regime classifier is only
applied here, read-only inference). Does not modify any imported module or live file.
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

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from eval_omega4_1_atr_safety_sltp_20260622 import _atr_pct  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_eth_funding_crossasset_combo_signal_20260825 import load_funding_z  # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402

OUT_DIR = ROOT / "tmp/eth_evidence_signal_chop_gated_costgate_20260827"
BTC_PATH = ROOT / "data" / "btc_5m_1year.csv"
REGIME_MODEL_PATH = ROOT / "tmp" / "eth_regime_gbm3_independent_20260826" / "model.joblib"

TP_ATR_MULT = 1.6
SL_ATR_MULT = 1.0
HORIZON_BARS = 48
LEVERAGE = 3.0
MARGIN_FRACTION = 0.30
ROUNDTRIP_COST_RATE = 0.001
ATR_N = 14

CANDIDATES = [
    ("orthogonal_combo", "bottom"),
    ("liquidity_sweep", "bottom"),
    ("smt_divergence", "bottom"),
    ("volume_wick_climax", "bottom"),
    ("short_term_return_z", "top"),
]


def log(msg: str) -> None:
    print(f"[chop_gated_costgate] {msg}", flush=True)


def _regime_labels(raw: pd.DataFrame) -> pd.Series:
    feats = _with_raw_state12(raw)
    payload = joblib.load(REGIME_MODEL_PATH)
    cols = payload["feature_cols"]
    med = pd.Series(payload["feature_medians"])
    for c in cols:
        if c not in feats.columns:
            feats[c] = med.get(c, 0.0)
    x = feats[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    proba = payload["model"].predict_proba(x)
    classes = list(payload["classes"])
    prob_df = pd.DataFrame(proba, columns=classes)
    return prob_df.idxmax(axis=1)


def _compute_frame(base_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(base_csv, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    btc = pd.read_csv(BTC_PATH, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    funding = load_funding_z()
    base_cols = ["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]
    sig = compute_signals(raw[base_cols].copy(), btc_df=btc, funding_df=funding)

    sig["regime_label"] = _regime_labels(raw).to_numpy()
    sig["atr_pct"] = pd.Series(_atr_pct(raw, ATR_N), index=raw.index)
    return sig


def run_window(frame: pd.DataFrame, bcol: str | None, tcol: str | None, chop_gate: bool,
                *, start, end, roundtrip_cost: float) -> dict[str, Any]:
    ts = frame["timestamp"]
    eligible = purged_decision_mask(ts, start=pd.Timestamp(start), end=pd.Timestamp(end), horizon_bars=HORIZON_BARS)

    bottom = frame[bcol].fillna(False).to_numpy() if bcol else np.zeros(len(frame), dtype=bool)
    top = frame[tcol].fillna(False).to_numpy() if tcol else np.zeros(len(frame), dtype=bool)
    if chop_gate:
        chop = (frame["regime_label"] == "chop").to_numpy()
        bottom = bottom & chop
        top = top & chop
    score = bottom.astype(np.float64) - top.astype(np.float64)

    has_score = frame["atr_pct"].notna().to_numpy()
    mask = eligible & has_score
    decision_indices = np.flatnonzero(mask)

    tp_moves = (TP_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices]
    sl_moves = (SL_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices]

    result = simulate_single_position(
        timestamps=ts, open_px=frame["open"].to_numpy(), high=frame["high"].to_numpy(),
        low=frame["low"].to_numpy(), close=frame["close"].to_numpy(),
        decision_indices=decision_indices, scores=score[decision_indices],
        tp_moves=tp_moves, sl_moves=sl_moves,
        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=roundtrip_cost,
    )
    ledger = result.ledger
    total_return = float(result.equity[-1] - 1.0) if len(result.equity) else float("nan")
    n_trades = int(len(ledger))
    wr = float((ledger["price_move"] * ledger["side"] > 0).mean()) if n_trades else float("nan")

    win_mask = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
    win_idx = np.flatnonzero(win_mask.to_numpy())
    if len(win_idx):
        p0, p1 = float(frame["close"].iloc[win_idx[0]]), float(frame["close"].iloc[win_idx[-1]])
        always_long, always_short = p1 / p0 - 1.0, p0 / p1 - 1.0
    else:
        always_long, always_short = float("nan"), float("nan")

    return {
        "n_trades": n_trades, "wr": wr, "total_return": total_return,
        "always_long_return": always_long, "always_short_return": always_short,
        "beats_benchmark": bool(total_return > max(always_long, always_short))
        if np.isfinite(always_long) and np.isfinite(always_short) else None,
    }


def find_breakeven_bp(frame: pd.DataFrame, bcol: str | None, tcol: str | None, chop_gate: bool, *, start, end) -> float | None:
    lo, hi = 0.0, 0.02
    r_lo = run_window(frame, bcol, tcol, chop_gate, start=start, end=end, roundtrip_cost=lo)["total_return"]
    r_hi = run_window(frame, bcol, tcol, chop_gate, start=start, end=end, roundtrip_cost=hi)["total_return"]
    if not np.isfinite(r_lo) or r_lo <= 0:
        return 0.0
    if r_hi > 0:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        r_mid = run_window(frame, bcol, tcol, chop_gate, start=start, end=end, roundtrip_cost=mid)["total_return"]
        if r_mid > 0:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0 * 10000.0)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("Building 2025/2026 frames (live evidence signals + chop regime + ATR)...")
    frames = {"2025": _compute_frame(gate.sweep.BASE_2025), "2026": _compute_frame(gate.sweep.BASE_2026)}
    for yr, fr in frames.items():
        counts = fr["regime_label"].value_counts().to_dict()
        log(f"  {yr}: {len(fr)} rows, regime counts {counts}")

    report: dict[str, Any] = {"config": {"tp_atr_mult": TP_ATR_MULT, "sl_atr_mult": SL_ATR_MULT,
                                          "horizon_bars": HORIZON_BARS, "leverage": LEVERAGE,
                                          "margin_fraction": MARGIN_FRACTION,
                                          "roundtrip_cost_rate": ROUNDTRIP_COST_RATE},
                               "results": {}}

    for name, side in CANDIDATES:
        bcol = f"bottom_{name}" if side == "bottom" else None
        tcol = f"top_{name}" if side == "top" else None
        for chop_gate in (False, True):
            variant = "chop_gated" if chop_gate else "ungated"
            key = f"{name}:{side}:{variant}"
            log(f"\n--- {key} ---")
            log(f"{'window':<8} {'n_trades':>8} {'wr':>7} {'return':>10} {'a_long':>9} {'a_short':>9} {'beats_bm':>9}  breakeven_bp")
            windows_out = {}
            for wname, wd in gate.WINDOW_DEFS.items():
                frame = frames["2025"] if wd["base_csv"] == gate.sweep.BASE_2025 else frames["2026"]
                res_std = run_window(frame, bcol, tcol, chop_gate, start=wd["start"], end=wd["end"], roundtrip_cost=ROUNDTRIP_COST_RATE)
                be = find_breakeven_bp(frame, bcol, tcol, chop_gate, start=wd["start"], end=wd["end"])
                be_str = f"{be:.1f}bp" if be is not None else ">200bp"
                windows_out[wname] = {**res_std, "breakeven_bp": be}
                log(f"{wname:<8} {res_std['n_trades']:>8d} "
                    f"{res_std['wr']*100 if np.isfinite(res_std['wr']) else float('nan'):>6.1f}% "
                    f"{res_std['total_return']*100:>9.2f}% {res_std['always_long_return']*100:>8.2f}% "
                    f"{res_std['always_short_return']*100:>8.2f}%  {str(res_std['beats_benchmark']):>9}  {be_str}")
            report["results"][key] = windows_out
            wins = sum(1 for w in windows_out.values() if w["beats_benchmark"])
            log(f"SUMMARY {key}: beats always_long/always_short in {wins}/{len(windows_out)} windows")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
