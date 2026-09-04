#!/usr/bin/env python3
"""Phase 0 data audit for the liquidation-map v2 (OI cohort + direction split) design --
docs/experiments/eth_liquidation_map_v2_oi_cohort_direction_design_20260825.md section 8.

Answers, before any modeling:
  A. Metrics archive integrity: coverage span, 5m-grid gap rate, dupes, NaN/zero rates.
  B. Join offset: the +5min end-label correction is verified empirically via the
     sum_open_interest_value ~= sum_open_interest * close residual (same method
     live_oi_delta_signal_20260824.py's docstring documents) -- sweep small offsets, report argmin.
     Guards against a repeat of the 08-23 "1-bucket future reference" defect class.
  C. 1h resample/join convention: kline bar labeled T (open-time) pairs with the OI snapshot at
     T+1h (metrics end-label), i.e. the OI known at that bar's close. Report join coverage.
  D. |dOI| <= volume identity (contracts can't change hands faster than they trade): violation
     rate at 5m and 1h -- violations flag data glitches the v2 compute must clamp.
  E. Cohort-survival effective depth: v2a's survival math (dOI+ entries, pro-rata decay on OI
     drops) run over the full joined history; at sample points, the mass-weighted age quantiles
     of surviving cohorts. Directly answers "how much OI history does the live provider need"
     (REST backfill gives ~21d; is that enough?).
  F. Taker-share source equivalence: backtest will take taker share from the price CSVs'
     taker_buy_base; live takes it from the same field in REST klines; the metrics archive's
     sum_taker_long_short_vol_ratio is a third, independent encoding of the same quantity.
     Verify CSV-derived vs metrics-ratio-derived hourly shares agree on their overlap.
  G. Backtest-vs-live OI seam: archive sum_open_interest vs a fresh REST openInterestHist
     (period=1h, ~21d retention) on their overlap -- same values at same UTC instants proves the
     two sources are interchangeable (tz + semantics).

Read-only against all inputs; writes one JSON report. Server duckdb checks (oi_lsratio.duckdb tz/
values) are done separately over SSH -- local dev has no duckdb module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base

ROOT = Path(__file__).resolve().parents[1]
METRICS_CSV = ROOT / "data" / "TOTAL_ETHUSDT_metrics_2024_2026.csv"
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_v2_phase0_data_audit_20260825.json"

REST_21D_HOURS = 21 * 24          # openInterestHist real retention (~500 points at 1h, measured)
KLINE_FETCH_DEPTH_HOURS = 999     # server's current 1000-kline fetch depth (minus forming bar)
DEPTH_SAMPLE_EVERY_H = 7 * 24     # one effective-depth sample per week
DEPTH_WARMUP_H = 90 * 24          # let cohorts accumulate before the first sample


def load_metrics() -> tuple[pd.DataFrame, dict]:
    m = pd.read_csv(METRICS_CSV)
    m["create_time"] = pd.to_datetime(m["create_time"], utc=True)  # archive is UTC, end-labeled
    m = m.sort_values("create_time").reset_index(drop=True)
    # Exchange-side dropouts are published as literal 0.0 OI (75 rows found on first run,
    # clustered 2024-07-09..15 + isolated days) -- physically impossible for ETH OI, and a 0
    # poisons the cohort survival math (ratio 0 -> log -inf) and fakes |dOI|>volume violations.
    # Rule carried into v2 compute: OI <= 0 is missing -> forward-fill. Flagged, never silent.
    clean = {}
    for c in ("sum_open_interest", "sum_open_interest_value"):
        bad = m[c] <= 0
        clean[f"{c}_zeros_ffilled"] = int(bad.sum())
        m.loc[bad, c] = np.nan
        m[c] = m[c].ffill()
    r = m["sum_open_interest"] / m["sum_open_interest"].shift(1)
    clean["ratio_outliers_after_clean"] = int(((r < 0.5) | (r > 2.0)).sum())
    return m, clean


def audit_metrics(m: pd.DataFrame, clean: dict) -> dict:
    span = (m["create_time"].iloc[0], m["create_time"].iloc[-1])
    expected = int((span[1] - span[0]) / pd.Timedelta(minutes=5)) + 1
    dupes = int(m["create_time"].duplicated().sum())
    off_grid = int((m["create_time"].dt.minute % 5 != 0).sum() + (m["create_time"].dt.second != 0).sum())
    cols = ["sum_open_interest", "sum_open_interest_value", "count_long_short_ratio",
            "sum_taker_long_short_vol_ratio"]
    quality = {c: {"nan_pct": round(float(m[c].isna().mean() * 100), 4),
                   "nonpos_pct": round(float((m[c] <= 0).mean() * 100), 4)} for c in cols}
    return {
        "rows": int(len(m)), "start": str(span[0]), "end": str(span[1]),
        "grid_gap_pct": round((1 - len(m) / expected) * 100, 4),
        "dupes": dupes, "off_grid_rows": off_grid, "column_quality": quality,
        "zero_oi_cleaning": clean,
    }


def offset_residual_sweep(m: pd.DataFrame, px5: pd.DataFrame) -> dict:
    """Implied price = OI_value / OI. If create_time is a corrected end-label, the snapshot at E
    matches the close of the 5m price bar labeled E-5min (bar-start labels) -> argmin at k=-1."""
    implied = (m["sum_open_interest_value"] / m["sum_open_interest"]).to_numpy()
    close_by_ts = px5.set_index("timestamp")["close"]
    out = {}
    for k in (-2, -1, 0, 1, 2):
        target = m["create_time"] + pd.Timedelta(minutes=5 * k)
        matched = close_by_ts.reindex(target).to_numpy()
        ok = np.isfinite(matched) & np.isfinite(implied)
        resid = np.abs(implied[ok] - matched[ok]) / matched[ok]
        out[f"price_label=create_time{'+' if k >= 0 else ''}{5*k}min"] = {
            "n": int(ok.sum()), "median_resid_pct": round(float(np.median(resid) * 100), 4)}
    return out


def hourly_join(m: pd.DataFrame, px1h: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Kline bar T (open-label, [T,T+1h)) <- OI/LS/taker-ratio snapshot at end-label T+1h."""
    snap = m.set_index("create_time")[
        ["sum_open_interest", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]]
    j = px1h.copy()
    end_label = j["timestamp"] + pd.Timedelta(hours=1)
    for c in snap.columns:
        j[c] = snap[c].reindex(end_label).to_numpy()
    in_span = (j["timestamp"] >= m["create_time"].iloc[0]) & (end_label <= m["create_time"].iloc[-1])
    miss_in_span = float(j.loc[in_span, "sum_open_interest"].isna().mean() * 100)
    n_miss = int(j.loc[in_span, "sum_open_interest"].isna().sum())
    j = j[in_span].reset_index(drop=True)
    j["oi_ffilled"] = j["sum_open_interest"].isna()
    j["sum_open_interest"] = j["sum_open_interest"].ffill()
    j = j[j["sum_open_interest"].notna()].reset_index(drop=True)
    stats = {"bars_in_overlap": int(in_span.sum()), "oi_missing_pct": round(miss_in_span, 4),
             "oi_missing_bars_ffilled": n_miss,
             "join_start": str(j["timestamp"].iloc[0]), "join_end": str(j["timestamp"].iloc[-1])}
    return j, stats


def doi_vs_volume(j: pd.DataFrame, m: pd.DataFrame, px5: pd.DataFrame) -> dict:
    doi = j["sum_open_interest"].diff()
    viol_1h = (doi.abs() > j["volume"]) & doi.notna()
    # 5m: metrics snapshot diffs vs the 5m bar volume ending at that snapshot (price label E-5min)
    m5 = m.set_index("create_time")["sum_open_interest"]
    d5 = m5.diff()
    vol5 = px5.set_index("timestamp")["volume"].reindex(m5.index - pd.Timedelta(minutes=5)).to_numpy()
    ok5 = np.isfinite(vol5) & d5.notna().to_numpy()
    viol5 = np.abs(d5.to_numpy()[ok5]) > vol5[ok5]
    worst = (doi.abs() / j["volume"].replace(0, np.nan)).nlargest(3)
    return {
        "violation_pct_1h": round(float(viol_1h.mean() * 100), 4),
        "violation_pct_5m": round(float(viol5.mean() * 100), 4),
        "worst_1h_ratio_abs_doi_over_vol": [round(float(x), 2) for x in worst.tolist()],
    }


def effective_depth(j: pd.DataFrame) -> dict:
    """v2a survival math only (no liquidation-price binning): dOI+ births a cohort, dOI- decays
    all cohorts pro-rata. mass_j(i) = dOI+_j * exp(S_i - S_j), S = cumsum(log(min(1, OI_i/OI_i-1)))
    -- S non-increasing so the exponent is <= 0: underflow-safe. Invariant: total surviving mass
    + still-decaying unattributed initial OI == current OI (exact by construction)."""
    oi = j["sum_open_interest"].to_numpy(dtype="float64")
    ratio = oi[1:] / oi[:-1]
    births = np.concatenate([[0.0], np.maximum(oi[1:] - oi[:-1], 0.0)])
    log_decay = np.concatenate([[0.0], np.minimum(np.log(ratio), 0.0)])
    S = np.cumsum(log_decay)
    n = len(oi)
    samples, invariant_errs = [], []
    for i in range(DEPTH_WARMUP_H, n, DEPTH_SAMPLE_EVERY_H):
        mass = births[: i + 1] * np.exp(S[i] - S[: i + 1])
        total = float(mass.sum())
        invariant_errs.append(abs(total + oi[0] * np.exp(S[i]) - oi[i]) / oi[i])
        ages_h = (i - np.arange(i + 1)).astype("float64")
        order = np.argsort(ages_h)
        cum = np.cumsum(mass[order]) / total
        q = {p: float(ages_h[order][np.searchsorted(cum, p)]) for p in (0.5, 0.9, 0.95)}
        samples.append({
            "ts": str(j["timestamp"].iloc[i]),
            "age_p50_d": q[0.5] / 24, "age_p90_d": q[0.9] / 24, "age_p95_d": q[0.95] / 24,
            "mass_pct_older_21d": float(mass[ages_h > REST_21D_HOURS].sum() / total * 100),
            "mass_pct_older_999h": float(mass[ages_h > KLINE_FETCH_DEPTH_HOURS].sum() / total * 100),
            "unattributed_initial_pct": float(oi[0] * np.exp(S[i]) / oi[i] * 100),
        })
    def agg(key):
        v = [s[key] for s in samples]
        return {"median": round(float(np.median(v)), 2), "p90": round(float(np.quantile(v, 0.9)), 2),
                "max": round(float(np.max(v)), 2)}
    recent = [s for s in samples if s["ts"] >= "2026-01-01"]
    return {
        "n_sample_points": len(samples),
        "invariant_max_rel_err": float(np.max(invariant_errs)),
        "age_p50_d": agg("age_p50_d"), "age_p90_d": agg("age_p90_d"), "age_p95_d": agg("age_p95_d"),
        "mass_pct_older_21d": agg("mass_pct_older_21d"),
        "mass_pct_older_999h": agg("mass_pct_older_999h"),
        "unattributed_initial_pct": agg("unattributed_initial_pct"),
        "recent_2026_samples": {
            "n": len(recent),
            "age_p90_d_median": round(float(np.median([s["age_p90_d"] for s in recent])), 2) if recent else None,
            "mass_pct_older_21d_median": round(float(np.median([s["mass_pct_older_21d"] for s in recent])), 2) if recent else None,
        },
        "last_sample": samples[-1] if samples else None,
    }


def taker_share_equivalence(m: pd.DataFrame, px5: pd.DataFrame) -> dict:
    """CSV path: share_1h = sum(taker_buy_base) / sum(volume). Metrics path: per-5m share =
    r/(1+r) from sum_taker_long_short_vol_ratio (buy/sell), volume-weighted to 1h using the CSV's
    5m volumes. Same quantity, independent encodings -- agreement validates using the CSV/klines
    field in backtest+live and ignoring the metrics column."""
    p = px5[["timestamp", "volume", "taker_buy_base"]].copy()
    r = m.set_index("create_time")["sum_taker_long_short_vol_ratio"]
    share5 = (r / (1.0 + r)).reindex(p["timestamp"] + pd.Timedelta(minutes=5)).to_numpy()
    p["metrics_buy"] = share5 * p["volume"].to_numpy()
    p = p[np.isfinite(p["metrics_buy"])]
    g = p.set_index("timestamp").resample("1h").sum(min_count=10)
    g = g.dropna()
    csv_share = g["taker_buy_base"] / g["volume"]
    met_share = g["metrics_buy"] / g["volume"]
    diff = (csv_share - met_share).abs()
    return {"n_hours": int(len(g)), "median_abs_diff": round(float(diff.median()), 5),
            "p99_abs_diff": round(float(diff.quantile(0.99)), 5),
            "csv_share_median": round(float(csv_share.median()), 4)}


def rest_oi_seam(m: pd.DataFrame) -> dict:
    resp = requests.get(
        "https://fapi.binance.com/futures/data/openInterestHist",
        params={"symbol": "ETHUSDT", "period": "1h", "limit": 500}, timeout=15)
    resp.raise_for_status()
    rows = resp.json()
    rest = pd.DataFrame(rows)
    rest["ts"] = pd.to_datetime(rest["timestamp"].astype("int64"), unit="ms", utc=True)
    rest["oi"] = rest["sumOpenInterest"].astype("float64")
    arc = m.set_index("create_time")["sum_open_interest"]
    out = {"rest_rows": int(len(rest)), "rest_start": str(rest["ts"].iloc[0]),
           "rest_end": str(rest["ts"].iloc[-1])}
    for name, off in [("0", pd.Timedelta(0)), ("-5min", pd.Timedelta(minutes=-5)),
                      ("+5min", pd.Timedelta(minutes=5)), ("-1h", pd.Timedelta(hours=-1))]:
        matched = arc.reindex(rest["ts"] + off).to_numpy()
        ok = np.isfinite(matched)
        if ok.sum() < 10:
            out[f"offset_{name}"] = {"n": int(ok.sum())}
            continue
        rel = np.abs(rest["oi"].to_numpy()[ok] - matched[ok]) / matched[ok]
        out[f"offset_{name}"] = {"n": int(ok.sum()),
                                 "median_rel_diff_pct": round(float(np.median(rel) * 100), 5)}
    return out


def main() -> None:
    report: dict = {}
    m, clean = load_metrics()
    report["A_metrics_audit"] = audit_metrics(m, clean)
    print("A. metrics audit:", json.dumps(report["A_metrics_audit"], indent=2))

    px5 = base._load_local_5m()
    # keep taker_buy_base alongside base's standard columns for section F
    raw_archive = pd.read_csv(base.ARCHIVE_CSV, usecols=["open_time", "taker_buy_base"])
    raw_archive["timestamp"] = pd.to_datetime(raw_archive["open_time"], unit="ms", utc=True)
    raw_main = pd.read_csv(base.PRICE_CSV, usecols=["timestamp", "taker_buy_base"], parse_dates=["timestamp"])
    raw_main["timestamp"] = raw_main["timestamp"].dt.tz_localize("UTC")
    taker = pd.concat([raw_archive[["timestamp", "taker_buy_base"]], raw_main[["timestamp", "taker_buy_base"]]],
                      ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    px5 = px5.merge(taker, on="timestamp", how="left")

    report["B_offset_residual"] = offset_residual_sweep(m, px5)
    print("B. offset residual sweep:", json.dumps(report["B_offset_residual"], indent=2))

    px1h = base.load_hourly()
    j, join_stats = hourly_join(m, px1h)
    report["C_hourly_join"] = join_stats
    print("C. hourly join:", json.dumps(join_stats, indent=2))

    report["D_doi_vs_volume"] = doi_vs_volume(j, m, px5)
    print("D. |dOI| vs volume:", json.dumps(report["D_doi_vs_volume"], indent=2))

    report["E_effective_depth"] = effective_depth(j)
    print("E. cohort effective depth:", json.dumps(report["E_effective_depth"], indent=2))

    report["F_taker_equivalence"] = taker_share_equivalence(m, px5)
    print("F. taker share equivalence:", json.dumps(report["F_taker_equivalence"], indent=2))

    try:
        report["G_rest_oi_seam"] = rest_oi_seam(m)
    except requests.RequestException as e:
        report["G_rest_oi_seam"] = {"error": str(e)}
    print("G. archive vs REST OI seam:", json.dumps(report["G_rest_oi_seam"], indent=2))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {OUT_JSON}")


if __name__ == "__main__":
    main()
