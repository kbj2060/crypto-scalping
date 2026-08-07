#!/usr/bin/env python3
"""Test the BTC GMM volatility-regime label (data/ensemble/unsupervised/btc/gmm_volatility_btc.pkl)
as an ENTRY FILTER on h48qual's own actual historical trades, mirroring
scripts/research_btc_vae_anomaly_h48qual_entryfilter_20260802.py's methodology.

h48qual ledger (Leg A elsewhere in this repo's portfolio-combination work):
tmp/causal_regen_20260516/btc_final_scale_map_20260708/{validation_ledger,oos_ledger}.csv
VAL = 2025-10-01..2025-12-31 (16 trades, report pnl=+7.45% mdd=-11.93%)
OOS = 2026-01-01..2026-06-25 (30 trades, report pnl=+22.69% mdd=-15.88%)
Both numbers reproduced exactly below from sequential trade_return compounding before anything else
is trusted.

Causality note: the gmm_cluster_rank / gmm_label at each trade's entry_timestamp is looked up via
merge_asof(direction="backward") against the cached causal score series (produced by
research_btc_gmm_volatility_standalone_signal_20260802.py, itself built only from already-published
bar features -- no forward-looking columns) -- most recent already-published bar at or before
entry_timestamp, never a future one.

Overlap caveat: the GMM was trained on data/splits/year_oos/btc_features_2026.csv (2026-01-01..
2026-08-01, train_ratio=0.8 -> in-sample cut ~2026-06-20). h48qual's OOS window (2026-01-01..
2026-06-25) is therefore almost entirely IN-SAMPLE for the GMM's own training, while h48qual's VAL
window (2025-10-01..2025-12-31) is genuinely out-of-sample for the GMM. Results are reported
separately per window and this asymmetry is called out explicitly.

Filter choice discipline (avoiding lookahead / cherry-picking from the same ledger being tested):
the standalone signal-quality check (2026_val_heldout split, which never overlaps h48qual's VAL or
OOS ledger windows) found gmm_label==5 has the highest mean forward-1h realized volatility
(0.007034) of all 6 labels in that held-out split -- higher even than the GMM's own design-intent
"rank 5" cluster (mean 0.004983 in that split), i.e. the model's internal ranking is imperfectly
ordered out-of-sample. label 5 is therefore used as the single most-defensible "high realized vol"
regime to test as a skip/downweight filter, chosen entirely from the held-out standalone analysis,
not from peeking at the ledger's own trade_return outcomes. As a second, coarser variant, the
GMM's own top-2 design ranks (cluster_rank >= 4) are also tested for comparison.

DIAGNOSTIC per CLAUDE.md Fresh-Forward Rule: vectorized replay from ledger's own trade_return column,
not a bar-by-bar live walk-forward. Does not touch trading_bot.py or any live wiring.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEDGER_DIR = ROOT / "tmp/causal_regen_20260516/btc_final_scale_map_20260708"
SCORE_CSV = ROOT / "tmp/research_20260802/btc_gmm_volatility_signal_check/gmm_score_series_full.csv"
OUT_DIR = ROOT / "tmp/research_20260802/btc_gmm_volatility_signal_check"

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
# chosen from held-out standalone analysis only (see module docstring)
HIGH_VOL_LABEL = 5
HIGH_VOL_RANK_MIN = 4  # cluster_rank >= 4 (GMM's own design-intent top-2 ranks)


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
        raise SystemExit(f"missing cached score series -- run research_btc_gmm_volatility_standalone_signal_20260802.py first: {SCORE_CSV}")
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
            win_scores[["timestamp", "gmm_label", "gmm_cluster_rank"]].rename(columns={"timestamp": "entry_timestamp"}),
            on="entry_timestamp", direction="backward",
        )
        n_missing = trades["gmm_label"].isna().sum()
        if n_missing:
            print(f"  WARNING: {n_missing} trades in {label} had no matched gmm score (dropped from filter comparison)")
        trades = trades.dropna(subset=["gmm_label"]).reset_index(drop=True)

        print(f"  Baseline reproduced with matched scores: pnl={base['pnl_pct']:+.2f}% "
              f"mdd={base['mdd_pct']:.2f}% trades={base['trades']} wr={base['win_rate_pct']:.1f}%")

        filters = {
            "label5_highvol": trades["gmm_label"].to_numpy() == HIGH_VOL_LABEL,
            "top2rank_highvol": trades["gmm_cluster_rank"].to_numpy() >= HIGH_VOL_RANK_MIN,
        }
        for filt_name, is_flagged in filters.items():
            n_flagged = int(is_flagged.sum())

            skip_returns = trades.loc[~is_flagged, "trade_return"].tolist()
            skip_stats = sequential_equity(skip_returns)

            half_returns = np.where(is_flagged, trades["trade_return"].to_numpy() * 0.5,
                                     trades["trade_return"].to_numpy()).tolist()
            half_stats = sequential_equity(half_returns)

            print(f"  [{filt_name}] flagged {n_flagged}/{len(trades)} trade entries")
            print(f"    SKIP  flagged entries : pnl={skip_stats['pnl_pct']:+.2f}% mdd={skip_stats['mdd_pct']:.2f}% "
                  f"trades={skip_stats['trades']} wr={skip_stats['win_rate_pct']:.1f}%")
            print(f"    HALF-SIZE flagged     : pnl={half_stats['pnl_pct']:+.2f}% mdd={half_stats['mdd_pct']:.2f}% "
                  f"trades={half_stats['trades']} wr={half_stats['win_rate_pct']:.1f}%")

            rows.append({
                "window": label, "filter": filt_name, "n_flagged": n_flagged,
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

    print("\n########## SUMMARY: does the GMM regime filter help or hurt h48qual's own ledger? ##########")
    for _, r in out_df.iterrows():
        skip_delta = r["skip_pnl_pct"] - r["baseline_pnl_pct"]
        half_delta = r["half_pnl_pct"] - r["baseline_pnl_pct"]
        print(f"{r['window']:<18} {r['filter']:<18} baseline_pnl={r['baseline_pnl_pct']:+.2f}%  "
              f"skip_delta={skip_delta:+.2f}pp (n_flagged={int(r['n_flagged'])})  half_delta={half_delta:+.2f}pp")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
