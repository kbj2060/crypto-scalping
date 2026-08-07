#!/usr/bin/env python3
"""DIAGNOSTIC CONTROL (not a promotion attempt): does the BTC "CryptoMamba future-regime tiebreak"
(scripts/research_btc_tau1_cryptomamba_tiebreak_20260801.py) actually depend on CryptoMamba's
regime-prediction quality, or would ANY reasonably-flip-rate-matched directional flag driving the
exact same regime_tiebreak combination mechanics produce a similar-looking edge purely because the
combination rule tends to help when applied to almost any not-too-noisy conflict-resolution signal?

This project has repeatedly found that regime-style tiebreaks can look like genuine regime-adaptive
skill while actually being a noise artifact that happens, in hindsight, to align with whichever leg
was favorable on the one realized price path tested (see the current-HMM version of this same
tiebreak, CLOSED as a noise artifact -- 34-35% bar-to-bar flip rate,
project-btc-tau1-style-leg-combination-first-attempt-20260801.md). CryptoMamba's flip rate (25.3%)
is lower but not zero, so that failure mode is not yet ruled out for CryptoMamba specifically -- only
for the much dumber "always favor whichever leg is SHORT" explanation (already tested and rejected).

Control design: replace CryptoMamba's bull_prob/bear_prob with a purely causal, purely mechanical
trailing-momentum flip-flop (sign of trailing N-hour log return -> hard bull_prob=1/bear_prob=0 or
the reverse, no smoothing, no ML) computed from BTC 5m close prices resampled to 1h. Feed it through
the IDENTICAL, UNMODIFIED regime_tiebreak mechanics (rule_weights/leg_side_series/weighted_pnl/
build_leg_equity_path, imported unchanged from the two existing scripts -- nothing about the
combination math is re-derived here) on the SAME 6 windows already used for the CryptoMamba result.

Methodology safeguard against p-hacking this control itself: the lookback N in {6, 12, 24} hours is
selected for "primary" reporting SOLELY by which one's bar-to-bar flip rate is closest to
CryptoMamba's already-published 25.3%, decided and printed BEFORE any performance number is computed
-- not by which N happens to perform best.

Leg A = BTC h48qual, Leg B = BTC Sigma9 trend-scan+regime, both loaded from the same frozen ledgers
used by research_btc_tau1_cryptomamba_tiebreak_20260801.py
(tmp/causal_regen_20260516/btc_final_scale_map_20260708/*). This is a control test on the
already-used windows, not a fresh-data test -- Fresh-Forward status is inherited/unchanged from the
CryptoMamba script it mirrors. Does not touch trading_bot.py or any live wiring.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from research_eth_sigma3_1h_omega461_joint_portfolio_20260731 import (  # noqa: E402
    omega_trades_from_ledger, build_leg_equity_path, summarize_equity,
)
from research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801 import (  # noqa: E402
    LEG_A_DIR, LEG_B_DIR, load_5m_prices_btc, leg_side_series, rule_weights, weighted_pnl,
)
from research_btc_tau1_cryptomamba_tiebreak_20260801 import (  # noqa: E402
    VAL_OOS_WINDOWS, ROLLING_WINDOWS, load_all_trades,
)

OUT_DIR = ROOT / "tmp/research_20260801/btc_cryptomamba_dumbmomentum_control"
LEG_A_LEDGERS = [LEG_A_DIR / "validation_ledger.csv", LEG_A_DIR / "oos_ledger.csv"]
LEG_B_LEDGERS = [LEG_B_DIR / "validation_ledger.csv", LEG_B_DIR / "oos_ledger.csv"]

CRYPTOMAMBA_FLIP_RATE = 25.3  # %, already published in research_btc_tau1_cryptomamba_tiebreak_20260801.py
CANDIDATE_N_HOURS = [6, 12, 24]

ALL_WINDOWS = VAL_OOS_WINDOWS + ROLLING_WINDOWS
COMBINED_SPAN_START = pd.Timestamp(min(s for _, s, _ in ALL_WINDOWS))
COMBINED_SPAN_END = pd.Timestamp(max(e for _, _, e in ALL_WINDOWS)) + pd.Timedelta("23h59min59s")

# Known CryptoMamba tiebreak results, read verbatim from the already-saved report CSVs under
# tmp/research_20260801/btc_tau1_cryptomamba_tiebreak/ (NOT retyped from memory/docstrings).
CMAMBA_REPORT_DIR = ROOT / "tmp/research_20260801/btc_tau1_cryptomamba_tiebreak"


def load_cmamba_known_results() -> pd.DataFrame:
    val_oos = pd.read_csv(CMAMBA_REPORT_DIR / "val_oos_summary.csv")
    rolling = pd.read_csv(CMAMBA_REPORT_DIR / "rolling_window_summary.csv")
    return pd.concat([val_oos, rolling], ignore_index=True)


def build_dumb_momentum_regime(prices_1h_close: pd.Series, n_hours: int) -> pd.DataFrame:
    """Trailing N-hour log return = log(close_t / close_{t-N}). Sign>0 -> bull_prob=1.0/bear_prob=0.0,
    else reverse.

    FIX 2026-08-01 (see project-btc-samebar-lookahead-bug-found-and-fixed-20260801.md): this used to
    label the output row for bar t with timestamp t, which looked causal but was NOT once combined
    downstream. `close_1h`/`eq_a_1h`/`eq_b_1h` are all built via pandas `.resample("1h").last()`,
    which left-labels bins -- label t actually holds data through ~t+55min. delta_a[t]/delta_b[t]
    (the equity change this signal gates) is therefore ALSO derived from prices through t+55min --
    the SAME window as this function's own close_t. Since callers merge this via
    `merge_asof(..., direction="backward")` (which admits timestamp <= t, equality included), the
    unshifted version let the regime "decision" for bar t use price information from bar t's own
    close -- a same-bar look-ahead. A before/after test (shift vs no-shift) collapsed a spurious
    41/41 (100%) episode win rate and a +509% backtest result down to 19/41 (46%) and -19.64% --
    confirming this was a real bug, not just a "misleading but correct" framing issue.

    Fix: the returned timestamp is now shifted forward by +1h from the bar the signal was actually
    computed from, so `merge_asof(direction='backward')` matching ts[i]=t only ever admits regime
    rows whose underlying close price predates t by at least one full hour -- guaranteed no overlap
    with delta_a[t]/delta_b[t]'s own window. Any caller that previously merge_asof'd this output
    directly is now automatically causal; no caller-side change needed."""
    log_ret = np.log(prices_1h_close / prices_1h_close.shift(n_hours))
    bull_prob = np.where(log_ret > 0, 1.0, 0.0)
    bear_prob = 1.0 - bull_prob
    # first n_hours bars have NaN return -> neutral 0.5/0.5 (matches reg_frame.fillna(0.5) downstream
    # convention used for CryptoMamba's own pre-coverage bars)
    nan_mask = log_ret.isna().to_numpy()
    bull_prob = np.where(nan_mask, 0.5, bull_prob)
    bear_prob = np.where(nan_mask, 0.5, bear_prob)
    shifted_timestamp = prices_1h_close.index + pd.Timedelta(hours=1)
    return pd.DataFrame({"timestamp": shifted_timestamp, "bull_prob": bull_prob, "bear_prob": bear_prob})


def flip_rate_over_span(regime: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    seg = regime[(regime["timestamp"] >= start) & (regime["timestamp"] <= end)].copy()
    side = np.where(seg["bull_prob"].to_numpy() >= seg["bear_prob"].to_numpy(), 1, -1)
    if len(side) < 2:
        return float("nan")
    flips = (side[1:] != side[:-1]).sum()
    return 100.0 * flips / (len(side) - 1)


def run_window(label, start_s, end_s, prices, regime) -> dict:
    start, end = pd.Timestamp(start_s), pd.Timestamp(end_s) + pd.Timedelta("23h59min59s")
    trades_a = load_all_trades(LEG_A_LEDGERS, start, end)
    trades_b = load_all_trades(LEG_B_LEDGERS, start, end)

    eq_a = build_leg_equity_path(trades_a, prices, start, end, use_ledger_trade_return=True)
    eq_b = build_leg_equity_path(trades_b, prices, start, end, use_ledger_trade_return=True)
    eq_a_1h = eq_a.resample("1h").last().ffill()
    eq_b_1h = eq_b.resample("1h").last().ffill()
    ts = pd.Series(eq_a_1h.index)
    side_a = leg_side_series(trades_a, ts)
    side_b = leg_side_series(trades_b, ts)
    active_a, active_b = side_a != 0, side_b != 0
    conflict = active_a & active_b & (side_a != side_b)

    reg = regime[(regime["timestamp"] >= start) & (regime["timestamp"] <= end)]
    reg_frame = pd.merge_asof(pd.DataFrame({"timestamp": ts}), reg, on="timestamp", direction="backward")
    bull = reg_frame["bull_prob"].fillna(0.5).to_numpy()
    bear = reg_frame["bear_prob"].fillna(0.5).to_numpy()

    delta_a = eq_a_1h.diff().fillna(0.0).to_numpy()
    delta_b = eq_b_1h.diff().fillna(0.0).to_numpy()

    w_a, w_b = rule_weights(conflict, side_a, side_b, bull, bear)
    tiebreak = weighted_pnl(delta_a, delta_b, w_a, w_b)
    eq_ab_baseline = 1.0 + (eq_a - 1.0) + (eq_b - 1.0)
    sab_base = summarize_equity(eq_ab_baseline)
    sa, sb = summarize_equity(eq_a), summarize_equity(eq_b)

    n_conflict = int(conflict.sum())
    beats_leg_a = tiebreak["pnl_pct"] > sa["pnl_pct"] and tiebreak["mdd_pct"] > sa["mdd_pct"]
    print(f"\n=== {label} {start_s}..{end_s} ===")
    print(f"Leg A alone: pnl={sa['pnl_pct']:+.2f}% mdd={sa['mdd_pct']:.2f}%   "
          f"Leg B alone: pnl={sb['pnl_pct']:+.2f}% mdd={sb['mdd_pct']:.2f}%")
    print(f"Fixed 1x-1x: pnl={sab_base['pnl_pct']:+.2f}% mdd={sab_base['mdd_pct']:.2f}%   "
          f"Dumb-momentum tiebreak (n_conflict={n_conflict}): pnl={tiebreak['pnl_pct']:+.2f}% "
          f"mdd={tiebreak['mdd_pct']:.2f}%  beats_leg_a_both_axes={beats_leg_a}")
    return {
        "window": label, "start": start_s, "end": end_s,
        "leg_a_pnl": sa["pnl_pct"], "leg_a_mdd": sa["mdd_pct"],
        "leg_b_pnl": sb["pnl_pct"], "leg_b_mdd": sb["mdd_pct"],
        "baseline_pnl": sab_base["pnl_pct"], "baseline_mdd": sab_base["mdd_pct"],
        "dumbmom_tiebreak_pnl": tiebreak["pnl_pct"], "dumbmom_tiebreak_mdd": tiebreak["mdd_pct"],
        "n_conflict_bars": n_conflict, "beats_leg_a_both_axes": beats_leg_a,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices_5m = load_5m_prices_btc()
    # causal 1h close series (last close observed within each hour, same convention eq_a_1h uses).
    # build_dumb_momentum_regime() applies its own +1h output shift (2026-08-01 fix) on top of this,
    # already validated via before/after test (41/41->19/41 win rate) -- left as-is intentionally.
    close_1h = prices_5m["close"].resample("1h").last().ffill()

    print("########## STEP 1: pre-commit N selection by flip rate (BEFORE any performance number) ##########")
    regimes = {}
    flip_rates = {}
    for n in CANDIDATE_N_HOURS:
        reg = build_dumb_momentum_regime(close_1h, n)
        regimes[n] = reg
        fr = flip_rate_over_span(reg, COMBINED_SPAN_START, COMBINED_SPAN_END)
        flip_rates[n] = fr
        print(f"N={n}h  flip_rate={fr:.2f}%  (CryptoMamba published flip_rate={CRYPTOMAMBA_FLIP_RATE:.2f}%, "
              f"diff={abs(fr - CRYPTOMAMBA_FLIP_RATE):.2f}pp)  span={COMBINED_SPAN_START.date()}..{COMBINED_SPAN_END.date()}")

    primary_n = min(CANDIDATE_N_HOURS, key=lambda n: abs(flip_rates[n] - CRYPTOMAMBA_FLIP_RATE))
    print(f"\n>>> PRIMARY N selected by closest-flip-rate rule (pre-committed, before performance run): "
          f"N={primary_n}h (flip_rate={flip_rates[primary_n]:.2f}%)")

    pd.DataFrame([{"n_hours": n, "flip_rate_pct": flip_rates[n],
                    "cryptomamba_flip_rate_pct": CRYPTOMAMBA_FLIP_RATE,
                    "abs_diff_pp": abs(flip_rates[n] - CRYPTOMAMBA_FLIP_RATE),
                    "is_primary": n == primary_n}
                   for n in CANDIDATE_N_HOURS]).to_csv(OUT_DIR / "flip_rate_selection.csv", index=False)

    print("\n########## STEP 2: run combination on all 6 windows for EACH candidate N ##########")
    all_results = {}
    for n in CANDIDATE_N_HOURS:
        print(f"\n---------------- N={n}h ----------------")
        rows = [run_window(label, s, e, prices_5m, regimes[n]) for label, s, e in ALL_WINDOWS]
        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / f"dumbmomentum_n{n}h_summary.csv", index=False)
        all_results[n] = df
        n_wins = int(df["beats_leg_a_both_axes"].sum())
        print(f"=== N={n}h SUMMARY: beats Leg A alone (pnl AND mdd) in {n_wins}/{len(df)} windows ===")

    print("\n########## STEP 3: side-by-side comparison, PRIMARY N vs CryptoMamba (known results) ##########")
    cmamba = load_cmamba_known_results()
    primary_df = all_results[primary_n]
    merged = cmamba.merge(primary_df[["window", "dumbmom_tiebreak_pnl", "dumbmom_tiebreak_mdd", "beats_leg_a_both_axes"]],
                           on="window", suffixes=("", "_dumbmom"))
    merged = merged.rename(columns={"cmamba_tiebreak_pnl": "cmamba_pnl", "cmamba_tiebreak_mdd": "cmamba_mdd",
                                     "beats_leg_a_both_axes": "cmamba_beats_leg_a",
                                     "beats_leg_a_both_axes_dumbmom": "dumbmom_beats_leg_a"})
    cols = ["window", "leg_a_pnl", "leg_a_mdd", "cmamba_pnl", "cmamba_mdd", "cmamba_beats_leg_a",
            "dumbmom_tiebreak_pnl", "dumbmom_tiebreak_mdd", "dumbmom_beats_leg_a"]
    print(merged[cols].to_string(index=False))
    merged.to_csv(OUT_DIR / f"comparison_primary_n{primary_n}h_vs_cryptomamba.csv", index=False)

    cmamba_wins = int(merged["cmamba_beats_leg_a"].sum())
    dumbmom_wins = int(merged["dumbmom_beats_leg_a"].sum())
    print(f"\n=== FINAL: CryptoMamba beats Leg A (both axes) in {cmamba_wins}/6 windows; "
          f"dumb-momentum (N={primary_n}h) beats Leg A (both axes) in {dumbmom_wins}/6 windows ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
