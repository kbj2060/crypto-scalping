#!/usr/bin/env python3
"""Test the BTC Isolation Forest anomaly score (data/ensemble/unsupervised/btc/isolation_forest_btc.pkl)
as an ENTRY FILTER on h48qual's own actual historical trades, since the anomaly score has no direction
of its own and cannot generate trades -- the only way it can plausibly help is by gating or
down-weighting an existing directional strategy's entries when the market "looks weird" at that moment.

h48qual ledger (same as scripts/research_btc_vae_anomaly_h48qual_entryfilter_20260802.py):
tmp/causal_regen_20260516/btc_final_scale_map_20260708/{validation_ledger,oos_ledger}.csv
VAL = 2025-10-01..2025-12-31 (16 trades, report pnl=+7.45% mdd=-11.93%)
OOS = 2026-01-01..2026-06-25 (30 trades, report pnl=+22.69% mdd=-15.88%)
Both numbers reproduced exactly below from sequential trade_return compounding (verified in the VAE
sibling script, same ledger, same method).

Causality note: the if_score at each trade's entry_timestamp is looked up via merge_asof(direction=
"backward") from the cached causal score series (scripts/research_btc_isolation_forest_standalone_
signal_20260802.py's if_score_series_full.csv, computed on data/splits/year_oos/btc_features_2024_2026
.csv with the same feature columns/mean/std/model as training) -- so it always uses the most recently
published bar at or before entry, never a future one.

Filter direction and threshold are chosen from the standalone analysis ONLY, never from this ledger
test (avoids the exact lookahead-bug-class mistake this project has repeatedly found and closed).
Standalone result: if_score correlates POSITIVELY with forward realized vol in pooled/2024/2026-train-
insample/2026-val-heldout groups (r=+0.05 to +0.18, all t>+9), i.e. higher anomaly -> higher forward
vol -- consistent sign in the model's own held-out split and in the most recent (2026) data, though 2025
alone flips negative (see standalone_summary.json, NOT the VAE failure pattern of held-out-vs-rest
flipping -- here held-out AGREES with pooled/2024/train). Direction chosen: SKIP/HALF-SIZE trades whose
entry-time if_score is in the top quintile/decile (high anomaly = higher expected forward vol = worse
risk-adjusted entry for a directional strategy), matching the VAE/GMM sibling scripts' filter convention.

IMPORTANT overlap caveat: the Isolation Forest was trained on data/splits/year_oos/btc_features_2026.csv
(2026-01-01..2026-08-01, train_ratio=0.8 -> in-sample cut ~2026-06-20). h48qual's OOS window (2026-01-01
..2026-06-25) is therefore almost entirely IN-SAMPLE for the Isolation Forest's own training, while
h48qual's VAL window (2025-10-01..2025-12-31) is genuinely out-of-sample for it. Results are reported
separately per window and this asymmetry is called out explicitly.

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
SCORE_CSV = ROOT / "tmp/research_20260802/btc_isolation_forest_signal_check/if_score_series_full.csv"
OUT_DIR = ROOT / "tmp/research_20260802/btc_isolation_forest_signal_check"

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
        raise SystemExit(f"missing cached score series -- run research_btc_isolation_forest_standalone_signal_20260802.py first: {SCORE_CSV}")
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

        win_scores = scores[(scores["timestamp"] >= start) & (scores["timestamp"] <= end)]
        trades = pd.merge_asof(
            trades.sort_values("entry_timestamp"),
            win_scores[["timestamp", "if_score"]].rename(columns={"timestamp": "entry_timestamp"}),
            on="entry_timestamp", direction="backward",
        )
        n_missing = trades["if_score"].isna().sum()
        if n_missing:
            print(f"  WARNING: {n_missing} trades in {label} had no matched if_score (dropped from filter comparison)")
        trades = trades.dropna(subset=["if_score"]).reset_index(drop=True)

        print(f"  Baseline reproduced with matched scores: pnl={base['pnl_pct']:+.2f}% "
              f"mdd={base['mdd_pct']:.2f}% trades={base['trades']} wr={base['win_rate_pct']:.1f}%")

        for filt_name, q in FILTER_QUANTILES.items():
            thr = float(win_scores["if_score"].quantile(q))
            is_high_anomaly = trades["if_score"].to_numpy() > thr
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
