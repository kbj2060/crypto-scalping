#!/usr/bin/env python3
"""Combined-signal follow-up to the two BTC unsupervised models trained earlier today:
GMM volatility regime (data/ensemble/unsupervised/btc/gmm_volatility_btc.pkl, standalone
r~+0.31-0.33 vs forward vol, stable, no sign-flip) and Isolation Forest anomaly
(data/ensemble/unsupervised/btc/isolation_forest_btc.pkl, standalone r~+0.18 on its own
held-out split). Both signals are individually real but each flagged too few h48qual trades
(0-4 out of 16-30) to judge as a binary entry filter. This script asks three questions using
the SAME cached causal score series and SAME h48qual ledger as the prior standalone/entry-filter
scripts (does not retrain either model, does not recompute scores):

(a) Does combining (AND / OR consensus) change the flagged-trade count to something usable, or
    does requiring both signals to agree just shrink the flagged set further? Uses the SAME
    per-window top-quintile/top-decile threshold convention as the individual entry-filter
    scripts (scripts/research_btc_{gmm_volatility,isolation_forest}_h48qual_entryfilter_20260802.py).

(b) First: raw correlation between the two signals themselves (gmm_cluster_rank vs if_score),
    to know whether they carry independent information at all. Then: does an averaged z-score
    of the two signals correlate better/more stably with forward realized vol than either alone,
    on the SAME held-out-split methodology (pooled / 2024 / 2025 / 2026_train_insample /
    2026_val_heldout) used in both standalone scripts? z-score stats computed on the pooled
    2024-2026 series (matches the non-causal diagnostic level of the prior standalone scripts --
    this is a correlation/stability check, not a live filter).

(c) Different framing: among the h48qual trades that ACTUALLY happened (46 trades total across
    VAL+OOS, no flagging/thresholding), does the anomaly/vol-regime score AT ENTRY correlate with
    that trade's own outcome (trade_return, continuous) or its adverse excursion
    (mae_price_move, an MDD-contribution proxy)? This uses the full trade sample so it isn't
    data-starved the way the binary-flag tests are.

Causality: score-at-entry is looked up via merge_asof(direction="backward") from the cached
score series, exactly as in the individual entry-filter scripts -- never a future bar.

DIAGNOSTIC per CLAUDE.md Fresh-Forward Rule: vectorized correlation/ledger replay, not a
bar-by-bar live walk-forward. Does not touch trading_bot.py or any live wiring. Does not retrain
either model -- both artifacts are loaded read-only via their cached score series.
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

GMM_SCORE_CSV = ROOT / "tmp/research_20260802/btc_gmm_volatility_signal_check/gmm_score_series_full.csv"
IF_SCORE_CSV = ROOT / "tmp/research_20260802/btc_isolation_forest_signal_check/if_score_series_full.csv"
LEDGER_DIR = ROOT / "tmp/causal_regen_20260516/btc_final_scale_map_20260708"
DATA_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026.csv"
OUT_DIR = ROOT / "tmp/research_20260802/btc_gmm_isoforest_combined_signal"

FORWARD_HORIZONS = {"h6_30m": 6, "h12_1h": 12, "h48_4h": 48, "h288_1d": 288}
FILTER_QUANTILES = {"top_quintile": 0.80, "top_decile": 0.90}

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


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 30:
        return float("nan"), float("nan"), n
    r = float(np.corrcoef(x, y)[0, 1])
    if abs(r) >= 1.0 or n <= 2:
        t = float("nan")
    else:
        t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return r, float(t), n


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gmm = pd.read_csv(GMM_SCORE_CSV, parse_dates=["timestamp"])
    isf = pd.read_csv(IF_SCORE_CSV, parse_dates=["timestamp"])
    assert len(gmm) == len(isf), "score series row-count mismatch -- expected identical timestamp grid"
    merged = gmm.merge(isf[["timestamp", "if_score"]], on="timestamp", how="inner", validate="one_to_one")
    assert len(merged) == len(gmm), "merge dropped rows -- timestamp grids not identical"
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    print(f"Merged score series: {len(merged)} rows, {merged['timestamp'].min()} .. {merged['timestamp'].max()}")

    # ---- (b1) raw correlation between the two signals themselves ----
    groups = {
        "pooled_all": merged,
        "2024": merged[merged["year"] == 2024],
        "2025": merged[merged["year"] == 2025],
        "2026_train_insample": merged[merged["split_2026"] == "2026_train_insample"],
        "2026_val_heldout": merged[merged["split_2026"] == "2026_val_heldout"],
    }
    print("\n########## (b1) Raw signal-vs-signal correlation: gmm_cluster_rank vs if_score ##########")
    xsig_rows = []
    for gname, gdf in groups.items():
        r, t, n = pearson(gdf["gmm_cluster_rank"].to_numpy(dtype=float), gdf["if_score"].to_numpy(dtype=float))
        xsig_rows.append({"group": gname, "pearson_r": r, "t_stat": t, "n": n})
        print(f"  {gname:<22} r={r:+.4f} t={t:+.2f} n={n}")
    xsig_df = pd.DataFrame(xsig_rows)
    xsig_df.to_csv(OUT_DIR / "signal_vs_signal_correlation.csv", index=False)

    # ---- (b2) combined z-scored score vs forward vol, same held-out discipline ----
    gmm_mean, gmm_std = merged["gmm_cluster_rank"].mean(), merged["gmm_cluster_rank"].std()
    if_mean, if_std = merged["if_score"].mean(), merged["if_score"].std()
    merged["z_gmm"] = (merged["gmm_cluster_rank"] - gmm_mean) / gmm_std
    merged["z_if"] = (merged["if_score"] - if_mean) / if_std
    merged["z_combined"] = (merged["z_gmm"] + merged["z_if"]) / 2.0

    df_full = pd.read_csv(DATA_PATH, usecols=["timestamp", "close"], parse_dates=["timestamp"]).sort_values("timestamp")
    assert len(df_full) == len(merged), "features file row count mismatch vs cached score series"
    close = df_full["close"].to_numpy(np.float64)
    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(merged)
    future = log_ret[1:]
    for label, h in FORWARD_HORIZONS.items():
        fwd_vol = np.full(n, np.nan)
        if len(future) >= h:
            windows = sliding_window_view(future, h)
            fwd_vol[: windows.shape[0]] = windows.std(axis=1) * np.sqrt(h)
        merged[f"fwd_vol_{label}"] = fwd_vol

    print("\n########## (b2) z_gmm / z_if / z_combined vs forward vol, per group ##########")
    combo_rows = []
    for gname, gdf in groups.items():
        gdf = merged.loc[gdf.index]
        for label in FORWARD_HORIZONS:
            target = gdf[f"fwd_vol_{label}"].to_numpy()
            for score_name in ("z_gmm", "z_if", "z_combined"):
                r, t, n_ = pearson(gdf[score_name].to_numpy(dtype=float), target)
                combo_rows.append({"group": gname, "horizon": label, "score": score_name,
                                    "pearson_r": r, "t_stat": t, "n": n_})
    combo_df = pd.DataFrame(combo_rows)
    combo_df.to_csv(OUT_DIR / "combined_score_vs_fwdvol_correlation.csv", index=False)
    for label in FORWARD_HORIZONS:
        print(f"\n--- horizon {label} ---")
        sub = combo_df[combo_df["horizon"] == label]
        for gname in groups:
            gsub = sub[sub["group"] == gname]
            vals = {row["score"]: (row["pearson_r"], row["t_stat"]) for _, row in gsub.iterrows()}
            print(f"  {gname:<22} " + "  ".join(f"{s}=r{v[0]:+.4f}/t{v[1]:+.2f}" for s, v in vals.items()))

    # sign-stability check on combined score, h12_1h (mirrors both individual standalone scripts)
    h12_sub = combo_df[(combo_df["horizon"] == "h12_1h") & (combo_df["score"] == "z_combined")]
    signs = {row["group"]: np.sign(row["pearson_r"]) for _, row in h12_sub.iterrows()}
    heldout_sign = signs.get("2026_val_heldout", float("nan"))
    other_signs = [v for k, v in signs.items() if k != "2026_val_heldout" and not np.isnan(v)]
    combined_sign_flip = bool(other_signs) and any(heldout_sign != s for s in other_signs)
    print(f"\nCombined z_combined sign-stability (h12_1h): {signs}  SIGN_FLIP_DETECTED={combined_sign_flip}")

    # ---- (a) flagged-count under AND / OR consensus, per h48qual window ----
    print("\n\n########## (a) Flagged-trade count: individual vs AND vs OR consensus ##########")
    flag_rows = []
    ledger_rows_for_c = []
    for wlabel, start, end, ledger_path in WINDOWS:
        trades = load_ledger(ledger_path, start, end)
        ref_pnl, ref_mdd, ref_trades = REFERENCE[wlabel]
        base = sequential_equity(trades["trade_return"].tolist())
        assert base["trades"] == ref_trades, f"{wlabel}: trade count mismatch"
        assert abs(base["pnl_pct"] - ref_pnl) < 0.5, f"{wlabel}: pnl mismatch"

        win_scores = merged[(merged["timestamp"] >= start) & (merged["timestamp"] <= end)]
        trades = pd.merge_asof(
            trades.sort_values("entry_timestamp"),
            win_scores[["timestamp", "gmm_cluster_rank", "if_score", "z_gmm", "z_if", "z_combined"]]
            .rename(columns={"timestamp": "entry_timestamp"}),
            on="entry_timestamp", direction="backward",
        )
        n_missing = trades["z_combined"].isna().sum()
        if n_missing:
            print(f"  WARNING: {n_missing} trades in {wlabel} had no matched score (dropped)")
        trades = trades.dropna(subset=["z_combined"]).reset_index(drop=True)
        trades["window"] = wlabel
        ledger_rows_for_c.append(trades)

        for filt_name, q in FILTER_QUANTILES.items():
            gmm_thr = float(win_scores["gmm_cluster_rank"].quantile(q))
            if_thr = float(win_scores["if_score"].quantile(q))
            gmm_flag = trades["gmm_cluster_rank"].to_numpy() > gmm_thr
            if_flag = trades["if_score"].to_numpy() > if_thr
            and_flag = gmm_flag & if_flag
            or_flag = gmm_flag | if_flag
            row = {
                "window": wlabel, "filter": filt_name, "n_total_trades": len(trades),
                "n_flagged_gmm_only": int(gmm_flag.sum()),
                "n_flagged_if_only": int(if_flag.sum()),
                "n_flagged_AND": int(and_flag.sum()),
                "n_flagged_OR": int(or_flag.sum()),
            }
            flag_rows.append(row)
            print(f"  [{wlabel} {filt_name}] total={len(trades)}  gmm_alone={row['n_flagged_gmm_only']}  "
                  f"if_alone={row['n_flagged_if_only']}  AND={row['n_flagged_AND']}  OR={row['n_flagged_OR']}")

    flag_df = pd.DataFrame(flag_rows)
    flag_df.to_csv(OUT_DIR / "flagged_count_consensus.csv", index=False)

    # ---- (c) score-at-entry vs trade OUTCOME (continuous), full 46-trade sample, no flagging ----
    print("\n\n########## (c) score-at-entry vs trade outcome, ALL trades (no threshold/flag) ##########")
    all_trades = pd.concat(ledger_rows_for_c, ignore_index=True)
    print(f"Total trades with matched score: {len(all_trades)}")

    outcome_cols = ["trade_return", "mae_price_move"]
    score_cols = ["gmm_cluster_rank", "if_score", "z_combined"]
    outcome_rows = []
    for scope_name, scope_df in [("pooled_VAL+OOS", all_trades)] + [
        (w, all_trades[all_trades["window"] == w]) for w, *_ in WINDOWS
    ]:
        for score_col in score_cols:
            for outcome_col in outcome_cols:
                r, t, n_ = pearson(scope_df[score_col].to_numpy(dtype=float), scope_df[outcome_col].to_numpy(dtype=float))
                outcome_rows.append({"scope": scope_name, "score": score_col, "outcome": outcome_col,
                                      "pearson_r": r, "t_stat": t, "n": n_})
                print(f"  [{scope_name:<16}] {score_col:<18} vs {outcome_col:<16} r={r:+.4f} t={t:+.2f} n={n_}")
    outcome_df = pd.DataFrame(outcome_rows)
    outcome_df.to_csv(OUT_DIR / "score_vs_trade_outcome.csv", index=False)
    all_trades.to_csv(OUT_DIR / "all_trades_with_scores.csv", index=False)

    # ---- summary ----
    summary = {
        "signal_vs_signal_pooled_r": float(xsig_df.loc[xsig_df["group"] == "pooled_all", "pearson_r"].iloc[0]),
        "signal_vs_signal_heldout_r": float(xsig_df.loc[xsig_df["group"] == "2026_val_heldout", "pearson_r"].iloc[0]),
        "combined_sign_flip_detected_h12_1h": bool(combined_sign_flip),
        "combined_signs_h12_1h": {k: (None if np.isnan(v) else float(v)) for k, v in signs.items()},
        "n_trades_total_with_matched_score": int(len(all_trades)),
    }
    with open(OUT_DIR / "combined_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote outputs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
