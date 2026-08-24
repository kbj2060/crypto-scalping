#!/usr/bin/env python3
"""0단계 진단 -- meta-label MI/R2-style pre-gate for the idea raised by the user: can the top-6
evidence signals (docs/experiments/eth_evidence_signal_ranking_stability_mar_jul_2026_20260814.md)
be used to build META-LABEL data layered on top of the EXISTING zigzag_action direction label
(direction_head's real production target), rather than replacing zigzag_action or h48_conservative
outright (both rejected as directions in the preceding turn's assessment -- zigzag_action is
already the ground truth these signals were validated AGAINST, and h48_conservative already
measures realized outcome directly, so the signals can't improve either as a replacement).

=== What a meta-label is, concretely, and what this script tests ===
De Prado's meta-labeling framing: given a PRIMARY model's directional bet, train a SECONDARY
classifier to decide whether to act on it -- but meta-labeling only adds value if the primary
model's bet correlates with the secondary features in a way the primary model doesn't already
capture. Before spending a real training run, this script runs the same "0단계 진단" (cheap
rank-correlation pre-gate) this sub-project always runs before committing to a label design
(matches eth_h48qual_quality_for_action_rank_correlation_20260811.md's own methodology): for every
REAL zigzag_action-active bar (the actual bars direction_head is trained to predict LONG/SHORT
on), does the evidence-signal "agreement" with that bar's own implied direction (bottom_votes if
zigzag_action=LONG, top_votes if SHORT) correlate with the REALIZED trade outcome you'd get by
actually taking that trade?

Simulation reuses TODAY's own already-vetted engine unmodified (core.causal_futures_backtest.
simulate_single_position, same TP/SL/leverage/cost constants as backtest_eth_evidence_signal_
top6_confluence_20260814.py -- not retuned here) with side FIXED to zigzag_action's own real
label (not the evidence-signal confluence -- the point is to test whether evidence-signal
agreement predicts the QUALITY of an already-decided zigzag_action bet, the literal meta-label
question, not to re-test the confluence formula as its own entry rule, already done and rejected).

=== Data coverage note (stated explicitly, not silently reconciled) ===
zigzag_action_labels_2026.csv only covers 2026-01-01..2026-02-28 16:00 (matches the OOS coverage
limit already documented throughout this session) -- OOS-Q2 (2026-04-01..06-30) has NO zigzag
label coverage and is EXCLUDED from this diagnostic, not silently dropped. 5 of the 6
pre-registered windows (2025q1/q2/q3, val, oos_q1) are covered and tested.

fresh_forward_bar_by_bar=true (entries at decision_i+1 open, TP/SL walked forward bar-by-bar, no
lookahead). trade_ledgers_used_as_input=false (ledger is a fresh simulation output here, not a
stored ledger reused as input). saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false. zigzag_action itself is the established production training
target already used for direction_head -- using it here as a research diagnostic label is not a
new leak, consistent with its existing role in this project.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py / runtime_config.py / .env.
No training, no GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
from backtest_eth_evidence_signal_top6_confluence_20260814 import (  # noqa: E402
    HORIZON_BARS, LEVERAGE, MARGIN_FRACTION, ROUNDTRIP_COST_RATE, SL_ATR_MULT, TP_ATR_MULT,
    _compute_signal_frame,
)

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_zigzag_metalabel_evidence_signal_rank_correlation_20260814"
ZIGZAG_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
TESTED_WINDOWS = ("2025q1", "2025q2", "2025q3", "val", "oos_q1")  # oos_q2 excluded, see module docstring


def log(msg: str) -> None:
    print(f"[metalabel_diag] {msg}", flush=True)


def _load_zigzag_labels() -> pd.DataFrame:
    frames = []
    for year_csv in ("zigzag_action_labels_2025.csv", "zigzag_action_labels_2026.csv"):
        df = pd.read_csv(ZIGZAG_LABEL_DIR / year_csv, usecols=["timestamp", "zigzag_action", "zigzag_transition_buffer"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return out


def run_window(frame: pd.DataFrame, *, start, end) -> dict[str, Any]:
    """frame must already have: timestamp, open, high, low, close, atr_pct, side (int, from
    zigzag_action, only non-zero on eligible decision bars), agreement_score (evidence-signal
    agreement with that bar's own side)."""
    ts = frame["timestamp"]
    eligible = purged_decision_mask(ts, start=pd.Timestamp(start), end=pd.Timestamp(end), horizon_bars=HORIZON_BARS)
    has_decision = (frame["side"] != 0).to_numpy() & frame["atr_pct"].notna().to_numpy()
    mask = eligible & has_decision
    decision_indices = np.flatnonzero(mask)

    tp_moves = (TP_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices]
    sl_moves = (SL_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices]
    scores = frame["side"].to_numpy(dtype=np.float64)[decision_indices]  # already +-1, exactly reproduces zigzag_action's own side

    result = simulate_single_position(
        timestamps=ts, open_px=frame["open"].to_numpy(), high=frame["high"].to_numpy(),
        low=frame["low"].to_numpy(), close=frame["close"].to_numpy(),
        decision_indices=decision_indices, scores=scores, tp_moves=tp_moves, sl_moves=sl_moves,
        upper_threshold=0.5, lower_threshold=-0.5, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )
    ledger = result.ledger
    if ledger.empty:
        return {"n_trades": 0}

    # join agreement_score at the DECISION bar (not entry bar) back onto each realized trade
    dec_ts_to_agreement = pd.Series(frame["agreement_score"].to_numpy(), index=frame["timestamp"])
    ledger = ledger.copy()
    ledger["agreement_score"] = dec_ts_to_agreement.reindex(ledger["decision_timestamp"]).to_numpy()

    n = len(ledger)
    n_agree = int((ledger["agreement_score"] >= 1).sum())
    if n < 20 or n_agree < 2 or n_agree == n:
        corr = {"n": n, "n_agreement_ge1": n_agree, "skipped": "too_few_trades_or_no_variation"}
    else:
        rho, p = stats.spearmanr(ledger["agreement_score"], ledger["trade_return"])
        corr = {"n": n, "n_agreement_ge1": n_agree, "spearman_rho": float(rho), "p_value": float(p)}
    return {
        "n_trades": n, "win_rate": float((ledger["trade_return"] > 0).mean()),
        "mean_return_agreement_ge1": float(ledger.loc[ledger["agreement_score"] >= 1, "trade_return"].mean()) if n_agree > 0 else float("nan"),
        "mean_return_agreement_eq0": float(ledger.loc[ledger["agreement_score"] == 0, "trade_return"].mean()) if (n - n_agree) > 0 else float("nan"),
        "rank_correlation": corr,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== stage=load_zigzag_labels_and_signal_frames ===")
    zz25 = _load_zigzag_labels()
    signal_by_base = {sweep.BASE_2025: _compute_signal_frame(sweep.BASE_2025), sweep.BASE_2026: _compute_signal_frame(sweep.BASE_2026)}

    report: dict[str, Any] = {
        "design": (
            "Meta-label 0-th-stage rank-correlation pre-gate: for every REAL zigzag_action-active "
            "bar (transition_buffer==0), simulate the trade zigzag_action itself implies (side "
            "fixed to the real label, same TP/SL/cost convention as today's standalone confluence "
            "backtest), then correlate evidence-signal agreement with that bar's own side "
            "(bottom_votes if LONG, top_votes if SHORT) against the realized trade_return. "
            "Answers: does evidence-signal confluence, AS A META-FEATURE on top of the existing "
            "direction label, predict which zigzag_action bets are worth taking?"
        ),
        "fresh_forward_bar_by_bar": True,
        "oracle_disclosure": (
            "The TP/SL walk-forward simulation itself is causal (entry at decision_i+1 open, "
            "bar-by-bar TP/SL), but the INPUT side (zigzag_action) is an ORACLE label -- it "
            "requires forward price confirmation to construct, exactly like this project's prior "
            "h48_conservative oracle-gate experiments. This diagnostic tests whether the "
            "meta-label MECHANISM works given a perfect primary signal; it does NOT test whether "
            "it would help direction_head's real (imperfect) predictions. Not promotion or model-"
            "selection evidence on its own -- see docs/experiments/eth_zigzag_metalabel_evidence_"
            "signal_rank_correlation_20260814.md."
        ),
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "excluded_window": {"name": "oos_q2", "reason": "zigzag_action_labels_2026.csv coverage ends 2026-02-28 16:00, before oos_q2 (2026-04-01..06-30) starts"},
        "windows": {},
    }

    results_by_window: dict[str, dict[str, Any]] = {}
    for wname in TESTED_WINDOWS:
        wd = gate.WINDOW_DEFS[wname]
        base_csv = wd["base_csv"]
        sframe = signal_by_base[base_csv]
        merged = sframe.merge(zz25, on="timestamp", how="left")
        merged["zigzag_action"] = merged["zigzag_action"].fillna(0).astype(int)
        merged["zigzag_transition_buffer"] = merged["zigzag_transition_buffer"].fillna(1).astype(int)  # missing -> treat as buffer, excluded
        eligible_side = np.where(
            (merged["zigzag_transition_buffer"] == 0) & (merged["zigzag_action"] == 1), 1,
            np.where((merged["zigzag_transition_buffer"] == 0) & (merged["zigzag_action"] == 2), -1, 0),
        )
        merged["side"] = eligible_side
        merged["agreement_score"] = np.where(merged["side"] > 0, merged["bottom_votes"], np.where(merged["side"] < 0, merged["top_votes"], np.nan))

        res = run_window(merged, start=wd["start"], end=wd["end"])
        results_by_window[wname] = res
        report["windows"][wname] = res
        if res.get("n_trades", 0) == 0:
            log(f"  {wname:8s} n_trades=0, skipped")
            continue
        rc = res["rank_correlation"]
        log(f"  {wname:8s} n_trades={res['n_trades']:5d}  win_rate={res['win_rate']:.3f}  "
            f"mean_ret(agree>=1)={res['mean_return_agreement_ge1']:+.4f}  mean_ret(agree=0)={res['mean_return_agreement_eq0']:+.4f}  "
            f"spearman_rho={rc.get('spearman_rho', float('nan')):+.4f}  p={rc.get('p_value', float('nan')):.4f}  n_agree>=1={rc.get('n_agreement_ge1', 0)}")

    signed_rhos = [r["rank_correlation"].get("spearman_rho") for r in results_by_window.values() if isinstance(r.get("rank_correlation"), dict) and "spearman_rho" in r["rank_correlation"]]
    n_positive = sum(1 for r in signed_rhos if r > 0)
    n_negative = sum(1 for r in signed_rhos if r < 0)
    report["summary"] = {
        "windows_with_correlation_computed": len(signed_rhos),
        "windows_positive_rho": n_positive,  # positive = agreement predicts BETTER outcome, the useful direction for a meta-label
        "windows_negative_rho": n_negative,
        "note": "positive spearman rho is the useful direction (higher evidence-signal agreement with zigzag_action's own side predicts a better realized trade_return). Sign consistency across the 5 available windows matters more than any single window's p-value given small per-window trade counts.",
    }
    log(f"  SUMMARY: {n_positive} positive / {n_negative} negative / {len(signed_rhos)} windows computed")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
