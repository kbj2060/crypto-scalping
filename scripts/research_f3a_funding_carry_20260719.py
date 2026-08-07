"""F3-A: 펀딩 캐리 팩터 - 탐색 구간 스크리닝 (사전 등록 격자, kill-gate).

docs/factor_execution_test_design_20260719.md 기반.
전략: 8h 펀딩 확정 시점 f_i를 관측 -> |f_i|>=theta면 -sign(f_i) 방향(펀딩 수취 방향)
진입 -> holding 기간(8h 또는 24h) 보유 -> 실현 가격수익률 + 실현 펀딩 수취(그 기간 동안
확정되는 후속 펀딩 합) - 비용.

Fresh-Forward 규칙 준수: theta/holding 선택은 탐색 구간(<=2025-08-31)에서만.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = Path("docs/test_designs_duckdb_live_20260719/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FUNDING_DIR = "data/research/funding_extracted"
KLINE_5M = {
    "ETHUSDT": "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
    "BTCUSDT": "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv",
    "SOLUSDT": "binance_data/klines/SOLUSDT/SOLUSDT-5m-api.csv",
}
ASSETS = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]

EXPLORATION_END = pd.Timestamp("2025-08-31", tz="UTC")
VAL_END = pd.Timestamp("2025-12-31", tz="UTC")
OOS_END = pd.Timestamp("2026-03-31", tz="UTC")

COST1_ROUNDTRIP_BPS = 10.0  # WS-A verified one-side taker 5bps x2 (open+close)
COST3_ROUNDTRIP_BPS = 30.0  # project 3x stress convention


def load_funding(asset: str) -> pd.DataFrame:
    import glob
    files = sorted(glob.glob(f"{FUNDING_DIR}/{asset}/*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["calc_time"])
    df["ts"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df[["ts", "last_funding_rate"]]


def load_kline(asset: str) -> pd.DataFrame:
    df = pd.read_csv(KLINE_5M[asset], usecols=["timestamp", "close"])
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["ts", "close"]].sort_values("ts").reset_index(drop=True)


def build_asset_dataset(asset: str) -> pd.DataFrame:
    funding = load_funding(asset)
    kline = load_kline(asset)
    kline_ts = kline["ts"].values
    kline_close = kline["close"].values

    def price_at_or_after(t):
        idx = np.searchsorted(kline_ts, t, side="left")
        if idx >= len(kline_ts):
            return np.nan
        return kline_close[idx]

    funding["price"] = [price_at_or_after(t) for t in funding["ts"].values]
    funding = funding.dropna(subset=["price"]).reset_index(drop=True)
    funding["asset"] = asset
    return funding


def run_grid_for_asset(df: pd.DataFrame, thetas: dict, holdings=(8, 24)):
    """df: rows are funding prints (8h spaced) with ts, last_funding_rate, price.
    Returns list of trade-level records for each (theta_key, holding) combo."""
    n = len(df)
    results = {}
    for theta_key, theta_val in thetas.items():
        for holding in holdings:
            step = holding // 8  # number of funding intervals held
            records = []
            i = 0
            while i + step < n:
                f_i = df["last_funding_rate"].iloc[i]
                if abs(f_i) >= theta_val:
                    direction = -1.0 if f_i > 0 else 1.0
                    entry_ts = df["ts"].iloc[i]
                    entry_px = df["price"].iloc[i]
                    exit_ts = df["ts"].iloc[i + step]
                    exit_px = df["price"].iloc[i + step]
                    funding_sum = df["last_funding_rate"].iloc[i + 1: i + step + 1].sum()
                    price_ret = np.log(exit_px / entry_px)
                    funding_component = direction * funding_sum
                    price_component = direction * price_ret
                    records.append({
                        "entry_ts": entry_ts, "exit_ts": exit_ts, "direction": direction,
                        "funding_component": funding_component, "price_component": price_component,
                        "gross_return": funding_component + price_component,
                    })
                    i += step  # non-overlapping: next decision only after this trade closes
                else:
                    i += 1
            results[(theta_key, holding)] = pd.DataFrame(records)
    return results


def day_block_bootstrap_tstat(returns: pd.Series, days: pd.Series, n_boot=3000, seed=20260719):
    rng = np.random.default_rng(seed)
    by_day = pd.Series(returns.values).groupby(days.values).apply(list)
    day_keys = by_day.index.to_numpy()
    if len(day_keys) < 5:
        return None
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sampled_days = rng.choice(day_keys, size=len(day_keys), replace=True)
        vals = np.concatenate([by_day[d] for d in sampled_days])
        boot_means[b] = np.mean(vals)
    observed = float(np.mean(returns))
    se = float(np.std(boot_means))
    t = observed / se if se > 1e-12 else None
    return {"observed_mean": observed, "boot_se": se, "t_stat": t}


def main():
    report = {"stage": "F3-A", "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}

    print("Loading funding + kline data per asset...")
    datasets = {a: build_asset_dataset(a) for a in ASSETS}
    for a, df in datasets.items():
        report.setdefault("data_coverage", {})[a] = {
            "n_funding_prints": int(len(df)),
            "min_ts": str(df["ts"].min()), "max_ts": str(df["ts"].max()),
        }

    # exploration-only threshold selection (per asset, using |funding| quantiles in exploration window)
    thetas_per_asset = {}
    for a, df in datasets.items():
        expl = df[df["ts"] < EXPLORATION_END]
        absf = expl["last_funding_rate"].abs()
        thetas_per_asset[a] = {
            "theta_0": 0.0,
            "theta_q50": float(absf.quantile(0.50)),
            "theta_q75": float(absf.quantile(0.75)),
            "theta_q90": float(absf.quantile(0.90)),
        }
    report["thetas_per_asset_exploration"] = thetas_per_asset

    print("Running exploration-window grid per asset...")
    exploration_summary = []
    for a, df in datasets.items():
        expl_df = df[df["ts"] < EXPLORATION_END].reset_index(drop=True)
        grid_results = run_grid_for_asset(expl_df, thetas_per_asset[a])
        for (theta_key, holding), trades in grid_results.items():
            if len(trades) < 20:
                exploration_summary.append({
                    "asset": a, "theta_key": theta_key, "holding_h": holding,
                    "n_trades": len(trades), "skipped": "insufficient_n",
                })
                continue
            trades["day"] = trades["entry_ts"].dt.date.astype(str)
            n_days = trades["day"].nunique()
            gross_mean = float(trades["gross_return"].mean())
            funding_mean = float(trades["funding_component"].mean())
            price_mean = float(trades["price_component"].mean())
            net_cost1_mean = gross_mean - COST1_ROUNDTRIP_BPS / 1e4
            net_cost3_mean = gross_mean - COST3_ROUNDTRIP_BPS / 1e4
            boot = day_block_bootstrap_tstat(
                trades["gross_return"] - COST1_ROUNDTRIP_BPS / 1e4, trades["day"]
            ) if n_days >= 5 else None
            exploration_summary.append({
                "asset": a, "theta_key": theta_key, "theta_val": thetas_per_asset[a][theta_key],
                "holding_h": holding, "n_trades": int(len(trades)), "n_days": int(n_days),
                "gross_mean_return": gross_mean,
                "funding_component_mean": funding_mean,
                "price_component_mean": price_mean,
                "net_mean_return_cost1": net_cost1_mean,
                "net_mean_return_cost3": net_cost3_mean,
                "annualized_net_cost1_pct": net_cost1_mean * (365 * 24 / holding) * 100.0,
                "bootstrap_cost1": boot,
            })
    report["exploration_grid"] = exploration_summary

    # kill-gate: any variant with positive net_cost1 mean AND t>3 (bootstrapped) in exploration?
    passing = [
        r for r in exploration_summary
        if "skipped" not in r
        and r["net_mean_return_cost1"] > 0
        and r["bootstrap_cost1"] is not None
        and r["bootstrap_cost1"]["t_stat"] is not None
        and r["bootstrap_cost1"]["t_stat"] > 3
    ]
    report["n_variants_tested"] = len([r for r in exploration_summary if "skipped" not in r])
    report["n_variants_passing_exploration"] = len(passing)
    report["passing_variants"] = passing
    report["F3A_verdict"] = (
        f"PROCEED to val -- {len(passing)} variant(s) pass exploration kill-gate (net_cost1>0, t>3)"
        if passing else
        "KILLED at exploration -- 0 variants show net-positive, t>3 carry edge under cost1"
    )

    out_json = OUT_DIR / "f3a_funding_carry_exploration_20260719.json"
    out_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print("WROTE", out_json)
    print(json.dumps({k: v for k, v in report.items() if k not in ("exploration_grid",)}, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
