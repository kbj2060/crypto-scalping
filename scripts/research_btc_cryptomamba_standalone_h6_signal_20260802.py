#!/usr/bin/env python3
"""Test BTC's CryptoMamba FUTURE-regime model (regime3_cmamba_h6_future_{bull,bear}_prob,
data/ensemble/supervised/regime3_cryptomamba_pred_btc_h6_nocurrent_20260721/) as its OWN standalone
trading signal at its native forecast horizon (h6 = 6 x 5m bars = 30 minutes), instead of as a 1h
regime_tiebreak gate on other legs.

Motivation: every attempt to use this (and other) signals as a regime_tiebreak GATE on top of
h48qual/Sigma9 legs failed under genuine LOWO/rolling-window validation this session (see
project-btc-run-window-merge-point-fixed-20260802.md and
project-btc-h48qual-lgbm-legb-lowo-failed-stale-results-20260802.md). CryptoMamba's own signal is
confirmed causal (inference window x[t-59:t+1], OOS AUC 0.8365 for the 3-way regime label per
docs/model_contracts/sol_btc_regime_models_retrain_tuning_20260721.md) and was never tested directly
as a trading rule at its own horizon -- only indirectly, gating a different strategy's equity.

Rule: score = bull_prob - bear_prob (both in [0,1], score in [-1,1]). Enter long when score >
+threshold, short when score < -threshold, else stay flat. Hold EXACTLY 6 bars (30 min, matching the
label's own forecast horizon) then exit -- no overlapping re-entry until the position closes. This
directly tests "is the h6 forecast profitable if you trade it for exactly h6", not an arbitrarily
chosen holding period.

Cost: round-trip = 2 * (fee + slippage) using ensemble/microstructure_wnc_sleeve.py's own defaults
(fee=0.0005, slippage=0.0002) = 0.28% one round trip. Also test 2x/3x cost tiers for robustness
(repo convention, see cost1/cost2/cost3 usage elsewhere).

Validation discipline: predictions are valid 2024-01-01 04:55 .. 2026-07-12 16:50 (2.5y). Split into
K=8 non-overlapping contiguous folds. For each held-out fold, select threshold using ONLY the other 7
folds (majority beats-flat on both pnl and mdd at cost tier 1x), then evaluate the selected threshold
on the held-out fold it never influenced -- same LOWO discipline as
research_btc_h48qual_direction_quality_lgbm_lowo_20260801.py, applied here to a standalone signal
instead of a combination weight.

DIAGNOSTIC per CLAUDE.md Fresh-Forward Rule: predictions/prices are pre-computed causal artifacts
replayed vectorized, not a bar-by-bar live walk-forward. Does not touch trading_bot.py or any live
wiring.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CMAMBA_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_btc_h6_nocurrent_20260721"
BTC_PRICE_CSV = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
OUT_DIR = ROOT / "tmp/research_20260802/btc_cryptomamba_standalone_h6_signal"

HOLD_BARS = 6  # matches the model's own h6 forecast horizon (30 min at 5m bars)
FEE, SLIP = 0.0005, 0.0002
ROUND_TRIP_COST_1X = 2 * (FEE + SLIP)
COST_MULTIPLIERS = {"cost1": 1.0, "cost2": 2.0, "cost3": 3.0}
CANDIDATE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
K_FOLDS = 8
MIN_TRADES_FOR_SELECTION = 5


def load_cmamba_predictions() -> pd.DataFrame:
    cols = ["timestamp", "regime3_cmamba_h6_future_bull_prob", "regime3_cmamba_h6_future_bear_prob"]
    dfs = [pd.read_csv(CMAMBA_DIR / f"btc_features_{y}_regime3_cryptomamba_pred_btc_h6_nocurrent_20260721.csv",
                        usecols=cols)
           for y in (2024, 2025, 2026)]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.rename(columns={
        "regime3_cmamba_h6_future_bull_prob": "bull_prob",
        "regime3_cmamba_h6_future_bear_prob": "bear_prob",
    }).dropna(subset=["bull_prob", "bear_prob"])
    return df.sort_values("timestamp").reset_index(drop=True)


def load_5m_prices_btc() -> pd.Series:
    df = pd.read_csv(BTC_PRICE_CSV, usecols=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").set_index("timestamp")["close"]


def build_folds(start: pd.Timestamp, end: pd.Timestamp, k: int) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    total_days = (end.normalize() - start.normalize()).days + 1
    base, extra = divmod(total_days, k)
    folds, cursor = [], start
    for i in range(k):
        n_days = base + (1 if i < extra else 0)
        fold_end = cursor + pd.Timedelta(days=n_days) - pd.Timedelta(seconds=1)
        if i == k - 1:
            fold_end = end
        folds.append((f"F{i + 1}", cursor, fold_end))
        cursor = fold_end + pd.Timedelta(seconds=1)
    return folds


def simulate(merged: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, threshold: float, cost_mult: float) -> dict:
    """Non-overlapping trades: enter when |score| > threshold and flat, hold HOLD_BARS, exit."""
    win = merged[(merged["timestamp"] >= start) & (merged["timestamp"] <= end)].reset_index(drop=True)
    n = len(win)
    score = (win["bull_prob"] - win["bear_prob"]).to_numpy()
    close = win["close"].to_numpy()
    cost = ROUND_TRIP_COST_1X * cost_mult

    equity, peak, mdd = 1.0, 1.0, 0.0
    trades = []
    i = 0
    while i < n - HOLD_BARS:
        s = score[i]
        if s > threshold:
            side = 1.0
        elif s < -threshold:
            side = -1.0
        else:
            i += 1
            continue
        entry_px, exit_px = close[i], close[i + HOLD_BARS]
        ret = side * (exit_px / entry_px - 1.0) - cost
        equity *= (1.0 + ret)
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1.0)
        trades.append(ret)
        i += HOLD_BARS  # non-overlapping: next entry only after this one closes

    pnl_pct = (equity - 1.0) * 100
    mdd_pct = mdd * 100
    wr = float(np.mean([t > 0 for t in trades])) * 100 if trades else float("nan")
    return {"pnl_pct": pnl_pct, "mdd_pct": mdd_pct, "trades": len(trades), "win_rate_pct": wr}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds = load_cmamba_predictions()
    prices = load_5m_prices_btc()
    merged = pd.merge_asof(preds, prices.rename("close"), left_on="timestamp", right_index=True, direction="backward")
    merged = merged.dropna(subset=["close"]).reset_index(drop=True)

    full_start, full_end = merged["timestamp"].iloc[0], merged["timestamp"].iloc[-1]
    print(f"Prediction+price coverage: {full_start} .. {full_end} ({len(merged)} bars)")

    folds = build_folds(full_start, full_end, K_FOLDS)
    print("\n########## Fold definitions ##########")
    for label, s, e in folds:
        print(f"{label}: {s.date()} .. {e.date()}")

    print("\n########## Grid: threshold x fold x cost tier ##########")
    rows = []
    for thr in CANDIDATE_THRESHOLDS:
        for label, s, e in folds:
            for cost_label, mult in COST_MULTIPLIERS.items():
                r = simulate(merged, s, e, thr, mult)
                r.update({"threshold": thr, "fold": label, "cost_tier": cost_label})
                rows.append(r)
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(OUT_DIR / "grid_all_thresholds_all_folds.csv", index=False)

    def get(label: str, thr: float, cost_tier: str = "cost1") -> dict:
        rows_ = grid_df[(grid_df["fold"] == label) & (grid_df["threshold"] == thr) & (grid_df["cost_tier"] == cost_tier)]
        return rows_.iloc[0].to_dict()

    print("\n########## LOWO selection @ cost1 (select on other folds, test on held-out) ##########")
    fold_labels = [label for label, _, _ in folds]
    lowo_rows = []
    for held_idx, (held_label, held_start, held_end) in enumerate(folds):
        selection_labels = [lbl for i, lbl in enumerate(fold_labels) if i != held_idx]
        majority_needed = math.floor(len(selection_labels) / 2) + 1

        candidates = []
        for thr in CANDIDATE_THRESHOLDS:
            wins, margins = 0, []
            for lbl in selection_labels:
                r = get(lbl, thr)
                beats = r["pnl_pct"] > 0 and r["mdd_pct"] > -20.0 and r["trades"] >= MIN_TRADES_FOR_SELECTION
                wins += int(beats)
                margins.append(r["pnl_pct"])
            if wins >= majority_needed:
                candidates.append((thr, wins, sum(margins) / len(margins)))

        if not candidates:
            held = get(held_label, CANDIDATE_THRESHOLDS[0])
            lowo_rows.append({"held_out": held_label, "selected_threshold": None,
                               "held_pnl_pct": None, "held_mdd_pct": None, "held_trades": held["trades"],
                               "held_out_profitable": False})
            print(f"{held_label}: NO threshold reached majority ({majority_needed}/{len(selection_labels)}) on selection folds")
            continue

        candidates.sort(key=lambda t: (-t[1], -t[2]))
        best_thr, sel_wins, sel_margin = candidates[0]
        held = get(held_label, best_thr)
        profitable = held["pnl_pct"] > 0 and held["mdd_pct"] > -20.0
        lowo_rows.append({"held_out": held_label, "selected_threshold": best_thr,
                           "selection_wins": f"{sel_wins}/{len(selection_labels)}",
                           "held_pnl_pct": held["pnl_pct"], "held_mdd_pct": held["mdd_pct"],
                           "held_trades": held["trades"], "held_win_rate_pct": held["win_rate_pct"],
                           "held_out_profitable": profitable})
        print(f"{held_label}: selected threshold={best_thr} (wins {sel_wins}/{len(selection_labels)}, "
              f"margin={sel_margin:+.2f}pp) | held-out: pnl={held['pnl_pct']:+.2f}% mdd={held['mdd_pct']:.2f}% "
              f"trades={held['trades']} wr={held['win_rate_pct']:.1f}%  profitable={profitable}")

    lowo_df = pd.DataFrame(lowo_rows)
    lowo_df.to_csv(OUT_DIR / "leave_one_window_out_results.csv", index=False)

    resolved = lowo_df[lowo_df["selected_threshold"].notna()]
    n_pass = int(resolved["held_out_profitable"].sum()) if len(resolved) else 0
    print(f"\n########## SUMMARY: {n_pass}/{len(folds)} held-out folds profitable at LOWO-selected threshold ##########")
    thr_set = set(resolved["selected_threshold"]) if len(resolved) else set()
    if len(thr_set) <= 1 and thr_set:
        print(f"SAME threshold selected in every resolved fold: {thr_set} -- not noise-driven cherry-picking.")
    elif thr_set:
        print(f"VARYING thresholds selected across folds: {sorted(thr_set)} -- cherry-picking risk.")

    print(f"\nWrote {OUT_DIR / 'leave_one_window_out_results.csv'} and {OUT_DIR / 'grid_all_thresholds_all_folds.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
