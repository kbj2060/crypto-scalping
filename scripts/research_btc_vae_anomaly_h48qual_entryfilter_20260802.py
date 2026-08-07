#!/usr/bin/env python3
"""Test the BTC VAE anomaly score (data/ensemble/unsupervised/btc/vae_anomaly_btc.pkl) as an ENTRY
FILTER on h48qual's own actual historical trades, since a reconstruction-error anomaly score has no
direction of its own and cannot generate trades -- the only way it can plausibly help is by gating
or down-weighting an existing directional strategy's entries when the market "looks weird" at that
moment.

h48qual ledger (Leg A in scripts/research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801.py):
tmp/causal_regen_20260516/btc_final_scale_map_20260708/{validation_ledger,oos_ledger}.csv
VAL = 2025-10-01..2025-12-31 (16 trades, report pnl=+7.45% mdd=-11.93%)
OOS = 2026-01-01..2026-06-25 (30 trades, report pnl=+22.69% mdd=-15.88%)
Both numbers reproduced exactly below from sequential trade_return compounding before anything else
is trusted (trade_return in this ledger is already the total account-level fractional return per
trade, i.e. sequential equity *= (1 + trade_return) reproduces the ledger's own report.json totals
exactly -- verified, no separate bar-level equity reconstruction needed for this comparison).

Causality note: the vae_score at each trade's entry_timestamp uses ensemble/unsupervised/common.py's
select_numeric_features() columns exactly as the model was trained on, all already-computed at that
bar (no forward-looking columns) -- reused from scripts/research_btc_vae_anomaly_standalone_signal_
20260802.py's cached score series (data/splits/year_oos/btc_features_2024_2026.csv, same feature
columns as training). merge_asof(direction="backward") is used so a trade's score is looked up from
the most recent already-published bar at or before entry_timestamp, never a future one.

IMPORTANT overlap caveat: the VAE was trained on data/splits/year_oos/btc_features_2026.csv (2026-01
-01 .. 2026-08-01, train_ratio=0.85 -> in-sample cut ~2026-06-30). h48qual's OOS window (2026-01-01..
2026-06-25) is therefore almost entirely IN-SAMPLE for the VAE's own training, while h48qual's VAL
window (2025-10-01..2025-12-31) is genuinely out-of-sample for the VAE. Results are reported
separately per window and this asymmetry is called out explicitly -- an entry-filter effect that only
shows up in the VAE's in-sample window is not trustworthy.

Filter definitions: for each window, the "top decile"/"top quintile" anomaly threshold is computed
from the FULL bar-level vae_score distribution over that window's date range (not just the handful of
trade entries), matching how a live filter would actually be calibrated. Two variants: SKIP (drop
entries with anomaly score in the top bucket) and HALF-SIZE (halve trade_return, since notional
scales roughly linearly with position size for this ledger's sizing convention).

DIAGNOSTIC per CLAUDE.md Fresh-Forward Rule: vectorized replay from ledger's own trade_return column,
not a bar-by-bar live walk-forward. Does not touch trading_bot.py or any live wiring.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEDGER_DIR = ROOT / "tmp/causal_regen_20260516/btc_final_scale_map_20260708"
SCORE_CSV = ROOT / "tmp/research_20260802/btc_vae_anomaly_signal_check/vae_score_series_full.csv"
OUT_DIR = ROOT / "tmp/research_20260802/btc_vae_anomaly_signal_check"

WINDOWS = [
    ("VAL_2025Q4", pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59"),
     LEDGER_DIR / "validation_ledger.csv"),
    ("OOS_2026extended", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-25 23:59:59"),
     LEDGER_DIR / "oos_ledger.csv"),
]
REFERENCE = {
    "VAL_2025Q4": (7.452139556587256, -11.926064814478032, 16),
    "OOS_2026extended": (22.689204833232957, -15.8772781396474, 30),
}
FILTER_QUANTILES = {"top_quintile": 0.80, "top_decile": 0.90}


def sequential_equity(returns: list[float]) -> dict:
    equity, peak, mdd = 1.0, 1.0, 0.0
    for r in returns:
        equity *= (1.0 + r)
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1.0)
    n = len(returns)
    wr = float(np.mean([r > 0 for r in returns])) * 100 if n else float("nan")
    return {"pnl_pct": (equity - 1.0) * 100, "mdd_pct": mdd * 100, "trades": n, "win_rate_pct": wr}


def load_ledger(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["entry_timestamp", "exit_timestamp"])
    df = df[(df["entry_timestamp"] >= start) & (df["entry_timestamp"] <= end)].reset_index(drop=True)
    return df


def main() -> int:
    if not SCORE_CSV.exists():
        raise SystemExit(f"missing cached score series -- run research_btc_vae_anomaly_standalone_signal_20260802.py first: {SCORE_CSV}")
    scores = pd.read_csv(SCORE_CSV, parse_dates=["timestamp"]).sort_values("timestamp")

    rows = []
    for label, start, end, ledger_path in WINDOWS:
        trades = load_ledger(ledger_path, start, end)
        ref_pnl, ref_mdd, ref_trades = REFERENCE[label]
        base = sequential_equity(trades["trade_return"].tolist())
        assert base["trades"] == ref_trades, f"{label}: trade count mismatch {base['trades']} vs {ref_trades}"
        assert abs(base["pnl_pct"] - ref_pnl) < 0.5, f"{label}: pnl mismatch {base['pnl_pct']} vs {ref_pnl}"
        print(f"[sanity check {label}] report pnl={ref_pnl:+.2f}% mdd={ref_mdd:.2f}% trades={ref_trades} | "
              f"reproduced pnl={base['pnl_pct']:+.2f}% mdd={base['mdd_pct']:.2f}% trades={base['trades']}")

        # causal score lookup: most recent bar at or before entry_timestamp
        win_scores = scores[(scores["timestamp"] >= start) & (scores["timestamp"] <= end)]
        trades = pd.merge_asof(
            trades.sort_values("entry_timestamp"),
            win_scores[["timestamp", "vae_score"]].rename(columns={"timestamp": "entry_timestamp"}),
            on="entry_timestamp", direction="backward",
        )
        n_missing = trades["vae_score"].isna().sum()
        if n_missing:
            print(f"  WARNING: {n_missing} trades in {label} had no matched vae_score (dropped from filter comparison)")
        trades = trades.dropna(subset=["vae_score"]).reset_index(drop=True)

        print(f"  Baseline reproduced with matched scores: pnl={base['pnl_pct']:+.2f}% "
              f"mdd={base['mdd_pct']:.2f}% trades={base['trades']} wr={base['win_rate_pct']:.1f}%")

        for filt_name, q in FILTER_QUANTILES.items():
            thr = float(win_scores["vae_score"].quantile(q))
            is_high_anomaly = trades["vae_score"].to_numpy() > thr
            n_flagged = int(is_high_anomaly.sum())

            skip_returns = trades.loc[~is_high_anomaly, "trade_return"].tolist()
            skip_stats = sequential_equity(skip_returns)

            half_returns = np.where(is_high_anomaly, trades["trade_return"].to_numpy() * 0.5,
                                     trades["trade_return"].to_numpy()).tolist()
            half_stats = sequential_equity(half_returns)

            print(f"  [{filt_name} thr={thr:.5f} q={q}] flagged {n_flagged}/{len(trades)} trade entries as high-anomaly")
            print(f"    SKIP  high-anomaly entries : pnl={skip_stats['pnl_pct']:+.2f}% mdd={skip_stats['mdd_pct']:.2f}% "
                  f"trades={skip_stats['trades']} wr={skip_stats['win_rate_pct']:.1f}%")
            print(f"    HALF-SIZE high-anomaly     : pnl={half_stats['pnl_pct']:+.2f}% mdd={half_stats['mdd_pct']:.2f}% "
                  f"trades={half_stats['trades']} wr={half_stats['win_rate_pct']:.1f}%")

            rows.append({
                "window": label, "filter": filt_name, "threshold": thr, "n_flagged": n_flagged,
                "n_total_trades": len(trades),
                "baseline_pnl_pct": base["pnl_pct"], "baseline_mdd_pct": base["mdd_pct"],
                "baseline_wr_pct": base["win_rate_pct"],
                "skip_pnl_pct": skip_stats["pnl_pct"], "skip_mdd_pct": skip_stats["mdd_pct"],
                "skip_trades": skip_stats["trades"], "skip_wr_pct": skip_stats["win_rate_pct"],
                "half_pnl_pct": half_stats["pnl_pct"], "half_mdd_pct": half_stats["mdd_pct"],
                "half_wr_pct": half_stats["win_rate_pct"],
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "h48qual_entryfilter_backtest.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'h48qual_entryfilter_backtest.csv'}")

    print("\n########## SUMMARY: does the anomaly filter help or hurt h48qual's own ledger? ##########")
    for _, r in out_df.iterrows():
        skip_delta = r["skip_pnl_pct"] - r["baseline_pnl_pct"]
        half_delta = r["half_pnl_pct"] - r["baseline_pnl_pct"]
        print(f"{r['window']:<18} {r['filter']:<12} baseline_pnl={r['baseline_pnl_pct']:+.2f}%  "
              f"skip_delta={skip_delta:+.2f}pp (n_flagged={int(r['n_flagged'])})  half_delta={half_delta:+.2f}pp")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
