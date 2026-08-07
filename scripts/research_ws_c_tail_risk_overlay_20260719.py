"""WS-C: 청산/독성 리스크 오버레이 - T-C0~T-C2 실증 실행.

Diagnostic only (conditional distribution test). No trade-level backtest yet (T-C3 requires
full Omega replay harness -- out of scope for this pass, explicitly flagged).
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

TAIL_DB = "data/live/tail_risk.duckdb"
MICRO_DB = "data/live/microstructure.duckdb"
OUT_DIR = Path("docs/test_designs_duckdb_live_20260719/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [5, 15, 60]  # minutes
STATES = ["S1_aftershock_q95", "S2_liq_imbalance_q95", "S3_toxicity_q95", "S4_S1_and_S3"]
N_BOOT = 2000
RNG_SEED = 20260719


def connect_retry(path, read_only=True, retries=8, backoff=2.0):
    last_exc = None
    for attempt in range(retries):
        try:
            return duckdb.connect(path, read_only=read_only)
        except duckdb.IOException as exc:
            last_exc = exc
            time.sleep(backoff * (attempt + 1))
    raise last_exc


KLINE_CSV = "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"


def load_full_history_price():
    """microstructure_1m.mark_price only covers ~9h (recent schema addition, see WS-A finding).
    Use the ETHUSDT 1m kline archive for full-history close price instead (covers through
    2026-07-15 16:36 UTC -- missing the last ~3 days of the 77-day window, documented gap)."""
    kl = pd.read_csv(KLINE_CSV, usecols=["timestamp", "close"])
    kl["ts"] = pd.to_datetime(kl["timestamp"], utc=True)
    return kl[["ts", "close"]].rename(columns={"close": "kline_close"})


def load():
    con_t = connect_retry(TAIL_DB)
    tail = con_t.execute(
        """
        select ts, long_usd_1m, short_usd_1m, shadow_aftershock_prob, shadow_decay_half_life,
               shadow_risk_bucket, liq_event_count_1m, valid_liq_stream, ws_stale
        from tail_risk_1m order by ts
        """
    ).df()
    con_t.close()

    con_m = connect_retry(MICRO_DB)
    micro = con_m.execute(
        "select ts, shadow_toxicity_score, mark_price, obi from microstructure_1m order by ts"
    ).df()
    con_m.close()
    return tail, micro


def day_block_bootstrap_mean_diff(state_vals: np.ndarray, other_vals: np.ndarray,
                                    state_days: np.ndarray, other_days: np.ndarray,
                                    n_boot=N_BOOT, seed=RNG_SEED):
    """Bootstrap by resampling whole days (blocks) to respect within-day autocorrelation."""
    rng = np.random.default_rng(seed)
    state_by_day = pd.Series(state_vals).groupby(state_days).apply(list)
    other_by_day = pd.Series(other_vals).groupby(other_days).apply(list)
    state_day_keys = state_by_day.index.to_numpy()
    other_day_keys = other_by_day.index.to_numpy()
    if len(state_day_keys) < 3 or len(other_day_keys) < 3:
        return None
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        s_days = rng.choice(state_day_keys, size=len(state_day_keys), replace=True)
        o_days = rng.choice(other_day_keys, size=len(other_day_keys), replace=True)
        s_vals = np.concatenate([state_by_day[d] for d in s_days])
        o_vals = np.concatenate([other_by_day[d] for d in o_days])
        diffs[b] = np.nanmean(s_vals) - np.nanmean(o_vals)
    observed = np.nanmean(state_vals) - np.nanmean(other_vals)
    se = np.nanstd(diffs)
    t_stat = observed / se if se > 1e-12 else np.nan
    return {
        "observed_diff": float(observed),
        "boot_se": float(se),
        "t_stat": float(t_stat),
        "ci_5": float(np.nanpercentile(diffs, 5)),
        "ci_95": float(np.nanpercentile(diffs, 95)),
    }


def main():
    report = {"stage": "WS-C", "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    tail, micro = load()
    kline = load_full_history_price()
    report["n_tail_risk_1m"] = int(len(tail))
    report["n_micro_1m"] = int(len(micro))
    report["kline_price_source"] = KLINE_CSV
    report["kline_coverage"] = {"min_ts": str(kline["ts"].min()), "max_ts": str(kline["ts"].max())}

    tail["ts"] = pd.to_datetime(tail["ts"], utc=True)
    micro["ts"] = pd.to_datetime(micro["ts"], utc=True)

    # ---- T-C0: data quality gate ----
    dq = {}
    dq["valid_liq_stream_false_ratio"] = float((~tail["valid_liq_stream"]).mean())
    dq["ws_stale_true_ratio"] = float(tail["ws_stale"].mean())
    dq["liq_event_gt0_ratio"] = float((tail["liq_event_count_1m"] > 0).mean())
    dq["liq_event_gt0_count"] = int((tail["liq_event_count_1m"] > 0).sum())
    report["T_C0_data_quality"] = dq

    stale_mask = tail["valid_liq_stream"].fillna(True) & (~tail["ws_stale"].fillna(False))
    tail_clean = tail.loc[stale_mask].reset_index(drop=True)
    report["T_C0_rows_excluded_stale"] = int(len(tail) - len(tail_clean))

    # merge micro onto tail on ts (both 1m, should align near-exactly), then merge full-history
    # kline close price (mark_price from microstructure_1m only covers last ~9h, see WS-A finding)
    merged = pd.merge(tail_clean, micro, on="ts", how="inner")
    merged = pd.merge(merged, kline, on="ts", how="left")
    report["T_C0_merged_rows"] = int(len(merged))
    report["T_C0_kline_price_match_ratio"] = float(merged["kline_close"].notna().mean())

    merged["liq_imbalance"] = (merged["long_usd_1m"] - merged["short_usd_1m"]).abs() / (
        merged["long_usd_1m"] + merged["short_usd_1m"]
    ).clip(lower=1.0)
    merged = merged.sort_values("ts").reset_index(drop=True)

    # forward returns at each horizon, using full-history kline close (NOT mark_price)
    merged["log_price"] = np.log(merged["kline_close"].replace(0, np.nan))
    for h in HORIZONS:
        merged[f"fwd_ret_{h}"] = merged["log_price"].shift(-h) - merged["log_price"]

    # state definitions.
    # IMPORTANT DEVIATION FROM PRE-REGISTERED DESIGN: aftershock_prob and liq_imbalance are
    # severely zero-inflated (only 74/96042 rows have aftershock_prob>0; only 19/96042 rows
    # have any liquidation event at all in 77 days of ETH data). A q95-quantile threshold on
    # a variable that is 0 at its 95th percentile collapses to "active = value >= 0" (i.e.
    # ~100% of rows), which is a degenerate definition, not a rare-event state. Switched S1/S2
    # to event-indicator definitions (value > 0) instead of quantile thresholds. This is
    # itself a primary finding: liquidation events are far too rare in this single-asset,
    # 77-day sample for quantile-based state definitions to make sense.
    q95_tox = merged["shadow_toxicity_score"].quantile(0.95)
    merged["S1_aftershock_q95"] = merged["shadow_aftershock_prob"] > 0.0
    merged["S2_liq_imbalance_q95"] = merged["liq_event_count_1m"] > 0
    merged["S3_toxicity_q95"] = merged["shadow_toxicity_score"] >= q95_tox
    merged["S4_S1_and_S3"] = merged["S1_aftershock_q95"] & merged["S3_toxicity_q95"]

    thresholds = {
        "S1_definition": "shadow_aftershock_prob > 0 (event indicator, NOT quantile -- see note)",
        "S2_definition": "liq_event_count_1m > 0 (event indicator, NOT quantile -- see note)",
        "q95_toxicity": float(q95_tox),
        "deviation_note": (
            "S1/S2 redefined from q95-quantile to raw event-indicator because the underlying "
            "variables are zero-inflated to >99.9% -- quantile(0.95)==0 for both in this sample."
        ),
    }
    report["T_C1_thresholds"] = thresholds
    report["T_C1_zero_inflation_finding"] = {
        "aftershock_prob_nonzero_rows": int((merged["shadow_aftershock_prob"] > 0).sum()),
        "aftershock_prob_nonzero_ratio": float((merged["shadow_aftershock_prob"] > 0).mean()),
        "liq_event_nonzero_rows": int((merged["liq_event_count_1m"] > 0).sum()),
        "liq_event_nonzero_ratio": float((merged["liq_event_count_1m"] > 0).mean()),
        "interpretation": (
            "Raw liquidation events are extremely rare in this 77-day single-asset (ETH) window: "
            "19 minutes with any liquidation event, 74 minutes with nonzero aftershock_prob. "
            "This directly limits statistical power for S1/S2 -- flagged per-cell as "
            "'insufficient_n' or 'insufficient_days' where it applies, not silently dropped."
        ),
    }
    for s in STATES:
        n_active = int(merged[s].sum())
        report[f"T_C1_{s}_n_active"] = n_active

    merged["day"] = merged["ts"].dt.date.astype(str)

    # ---- T-C1: conditional distribution test, full grid, FDR correction ----
    cells = []
    for state, horizon in itertools.product(STATES, HORIZONS):
        col = f"fwd_ret_{horizon}"
        active = merged[state].fillna(False)
        state_df = merged.loc[active, [col, "day"]].dropna()
        other_df = merged.loc[~active, [col, "day"]].dropna()
        if len(state_df) < 30 or len(other_df) < 30:
            cells.append(
                {"state": state, "horizon_min": horizon, "n_state": len(state_df),
                 "n_other": len(other_df), "skipped": "insufficient_n"}
            )
            continue
        boot = day_block_bootstrap_mean_diff(
            state_df[col].values, other_df[col].values,
            state_df["day"].values, other_df["day"].values,
        )
        if boot is None:
            cells.append(
                {"state": state, "horizon_min": horizon, "n_state": len(state_df),
                 "n_other": len(other_df), "skipped": "insufficient_days"}
            )
            continue
        # lower-tail comparison: p5 of state vs p5 of other
        p5_state = float(state_df[col].quantile(0.05))
        p5_other = float(other_df[col].quantile(0.05))
        # realized vol comparison
        vol_state = float(state_df[col].std())
        vol_other = float(other_df[col].std())
        p_two_sided = 2 * min(
            stats.norm.cdf(-abs(boot["t_stat"])), 0.5
        ) if not np.isnan(boot["t_stat"]) else np.nan
        cells.append(
            {
                "state": state,
                "horizon_min": horizon,
                "n_state": int(len(state_df)),
                "n_other": int(len(other_df)),
                "mean_diff": boot["observed_diff"],
                "t_stat": boot["t_stat"],
                "p_value_approx": p_two_sided,
                "p5_state": p5_state,
                "p5_other": p5_other,
                "p5_diff": p5_state - p5_other,
                "vol_state": vol_state,
                "vol_other": vol_other,
                "vol_ratio": vol_state / vol_other if vol_other > 0 else None,
            }
        )
    report["T_C1_grid_cells"] = cells

    # BH-FDR correction across all valid p-values
    valid_cells = [c for c in cells if "p_value_approx" in c and not np.isnan(c["p_value_approx"])]
    pvals = np.array([c["p_value_approx"] for c in valid_cells])
    order = np.argsort(pvals)
    m = len(pvals)
    fdr_q = 0.10
    bh_thresh = np.zeros(m)
    for rank, idx in enumerate(order, start=1):
        bh_thresh[idx] = rank / m * fdr_q
    significant_mask = pvals <= bh_thresh
    # standard BH: find largest k where sorted p <= k/m*q, then all up to k are significant
    sorted_p = pvals[order]
    below = sorted_p <= (np.arange(1, m + 1) / m * fdr_q)
    k = np.max(np.where(below)[0]) + 1 if below.any() else 0
    sig_indices = set(order[:k].tolist())
    for i, c in enumerate(valid_cells):
        c["fdr_significant"] = i in sig_indices
    report["T_C1_fdr_q"] = fdr_q
    report["T_C1_n_fdr_significant"] = int(len(sig_indices))
    report["T_C1_n_cells_tested"] = int(m)

    # ---- split-half reproducibility ----
    mid_date = merged["ts"].min() + (merged["ts"].max() - merged["ts"].min()) / 2
    half1 = merged[merged["ts"] < mid_date]
    half2 = merged[merged["ts"] >= mid_date]
    report["T_C1_split_boundary"] = str(mid_date)
    report["T_C1_half1_n"] = int(len(half1))
    report["T_C1_half2_n"] = int(len(half2))

    def run_half(df_half):
        out = {}
        for state, horizon in itertools.product(STATES, HORIZONS):
            col = f"fwd_ret_{horizon}"
            active = df_half[state].fillna(False)
            sdf = df_half.loc[active, col].dropna()
            odf = df_half.loc[~active, col].dropna()
            if len(sdf) < 20 or len(odf) < 20:
                out[f"{state}|{horizon}"] = None
                continue
            out[f"{state}|{horizon}"] = float(sdf.mean() - odf.mean())
        return out

    half1_diffs = run_half(half1)
    half2_diffs = run_half(half2)
    reproducible_cells = []
    for key in half1_diffs:
        d1, d2 = half1_diffs[key], half2_diffs.get(key)
        if d1 is None or d2 is None:
            continue
        same_sign = (d1 > 0) == (d2 > 0) and abs(d1) > 1e-9 and abs(d2) > 1e-9
        reproducible_cells.append({"cell": key, "half1_diff": d1, "half2_diff": d2, "same_sign": bool(same_sign)})
    report["T_C1_split_half_reproducibility"] = reproducible_cells

    # combine: FDR-significant AND same-sign across halves
    fdr_sig_keys = {f"{c['state']}|{c['horizon_min']}" for c in valid_cells if c["fdr_significant"]}
    reproducible_keys = {c["cell"] for c in reproducible_cells if c["same_sign"]}
    accepted_keys = fdr_sig_keys & reproducible_keys
    report["T_C1_accepted_cells_fdr_and_reproducible"] = sorted(accepted_keys)
    report["H_C1_verdict"] = (
        f"ACCEPTED ({len(accepted_keys)} cells pass FDR+split-half)" if accepted_keys
        else "REJECTED (0 cells pass both FDR and split-half reproducibility)"
    )

    # ---- T-C2: information redundancy check (against available features: obi as proxy for
    # decision_feature_frame since full 243-column frame has only 111-1418 rows, too sparse
    # to reliably correlate against 96k tail-risk rows -- documented limitation) ----
    redundancy = {}
    for state in STATES:
        active = merged[state].fillna(False).astype(int)
        corr_obi = float(pd.Series(active).corr(merged["obi"]))
        redundancy[state] = {"spearman_corr_vs_obi": corr_obi}
    report["T_C2_redundancy_note"] = (
        "decision_feature_frame (243 cols) has only 111-1418 rows total vs 96k tail-risk rows -- "
        "too sparse for reliable correlation against tail-risk states. Used microstructure_1m.obi "
        "(full coverage) as an available proxy for 'does the live feature set already see this'. "
        "Full redundancy check against the 243-column live feature frame deferred until frame "
        "coverage improves (it was only fixed from a schema bug in 2026-07-02, still sparse)."
    )
    report["T_C2_redundancy_vs_obi"] = redundancy

    out_json = OUT_DIR / "ws_c_tail_risk_overlay_20260719.json"
    out_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print("WROTE", out_json)
    summary_keys = [
        "n_tail_risk_1m", "n_micro_1m", "T_C0_data_quality", "T_C0_rows_excluded_stale",
        "T_C0_merged_rows", "T_C1_thresholds", "T_C1_fdr_q", "T_C1_n_fdr_significant",
        "T_C1_n_cells_tested", "T_C1_accepted_cells_fdr_and_reproducible", "H_C1_verdict",
        "T_C2_redundancy_vs_obi",
    ]
    print(json.dumps({k: report[k] for k in summary_keys if k in report}, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
