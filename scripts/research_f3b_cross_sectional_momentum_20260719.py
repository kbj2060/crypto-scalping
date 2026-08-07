"""F3-B: 크로스섹션 모멘텀 - 탐색 구간 스크리닝.

3자산(ETH/BTC/SOL) k일 수익률로 순위 -> 1등 롱 / 3등 숏 (달러 중립), 일 1회 리밸런스.
k in {7,14,30} (탐색 구간에서만 선택). 유니버스 3개뿐이라는 한계 명시.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("docs/test_designs_duckdb_live_20260719/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KLINE_5M = {
    "ETHUSDT": "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
    "BTCUSDT": "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv",
    "SOLUSDT": "binance_data/klines/SOLUSDT/SOLUSDT-5m-api.csv",
}
ASSETS = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]
EXPLORATION_END = pd.Timestamp("2025-08-31", tz="UTC")
COST1_ROUNDTRIP_BPS = 10.0
COST3_ROUNDTRIP_BPS = 30.0
K_GRID = [7, 14, 30]


def load_daily_close(asset: str) -> pd.Series:
    df = pd.read_csv(KLINE_5M[asset], usecols=["timestamp", "close"])
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    daily = df.set_index("ts")["close"].resample("1D").last().dropna()
    return daily


def day_block_bootstrap_tstat(returns: np.ndarray, n_boot=3000, seed=20260719):
    if len(returns) < 30:
        return None
    rng = np.random.default_rng(seed)
    boot_means = np.array([rng.choice(returns, size=len(returns), replace=True).mean() for _ in range(n_boot)])
    observed = float(np.mean(returns))
    se = float(np.std(boot_means))
    return {"observed_mean": observed, "boot_se": se, "t_stat": observed / se if se > 1e-12 else None}


def main():
    report = {"stage": "F3-B", "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    report["limitation_note"] = (
        "Universe = 3 assets only -- statistically thin cross-section, results are directional "
        "signal only, not a robust cross-sectional finding. Kill-gate applies: if no signal here, "
        "do not expand universe to go fishing (per design doc anti-pattern)."
    )

    closes = {a: load_daily_close(a) for a in ASSETS}
    common_idx = closes["ETHUSDT"].index
    for a in ASSETS[1:]:
        common_idx = common_idx.intersection(closes[a].index)
    price_df = pd.DataFrame({a: closes[a].reindex(common_idx) for a in ASSETS}).dropna()
    report["data_coverage"] = {
        "n_days": int(len(price_df)),
        "min_date": str(price_df.index.min()), "max_date": str(price_df.index.max()),
    }

    fwd_ret = np.log(price_df.shift(-1) / price_df)  # next-day forward return per asset

    results = []
    for k in K_GRID:
        past_ret = np.log(price_df / price_df.shift(k))
        ranks = past_ret.rank(axis=1, method="first")
        long_asset = ranks.idxmax(axis=1)
        short_asset = ranks.idxmin(axis=1)

        daily_pnl = []
        dates = []
        for i in range(len(price_df) - 1):
            date = price_df.index[i]
            if pd.isna(past_ret.iloc[i]).any():
                continue
            la, sa = long_asset.iloc[i], short_asset.iloc[i]
            if la == sa:
                continue
            r_long = fwd_ret[la].iloc[i]
            r_short = fwd_ret[sa].iloc[i]
            if pd.isna(r_long) or pd.isna(r_short):
                continue
            gross = 0.5 * r_long - 0.5 * r_short  # dollar-neutral, half notional each leg
            daily_pnl.append(gross)
            dates.append(date)

        pnl_arr = np.array(daily_pnl)
        if len(pnl_arr) < 30:
            results.append({"k": k, "skipped": "insufficient_n", "n": len(pnl_arr)})
            continue

        expl_mask = np.array([d < EXPLORATION_END for d in dates])
        expl_pnl = pnl_arr[expl_mask]
        gross_mean = float(np.mean(expl_pnl))
        # cost: 2 legs, each leg rebalanced daily = up to 2 round trips/day worst case;
        # conservative: charge full round-trip cost on both legs each day
        cost_per_day = 2 * COST1_ROUNDTRIP_BPS / 1e4
        net_cost1 = gross_mean - cost_per_day
        net_cost3 = gross_mean - 2 * COST3_ROUNDTRIP_BPS / 1e4
        boot = day_block_bootstrap_tstat(expl_pnl - cost_per_day)

        results.append({
            "k": k, "n_days_total": int(len(pnl_arr)), "n_days_exploration": int(len(expl_pnl)),
            "gross_mean_daily_return": gross_mean,
            "net_mean_daily_return_cost1": net_cost1,
            "net_mean_daily_return_cost3": net_cost3,
            "annualized_net_cost1_pct": net_cost1 * 365 * 100.0,
            "bootstrap_cost1": boot,
        })

    report["exploration_grid"] = results
    passing = [
        r for r in results
        if "skipped" not in r and r["net_mean_daily_return_cost1"] > 0
        and r["bootstrap_cost1"] is not None and r["bootstrap_cost1"]["t_stat"] is not None
        and r["bootstrap_cost1"]["t_stat"] > 3
    ]
    report["n_variants_passing_exploration"] = len(passing)
    report["passing_variants"] = passing
    report["F3B_verdict"] = (
        f"PROCEED to val -- {len(passing)} variant(s) pass" if passing else
        "KILLED at exploration -- 0/3 k-values show net-positive, t>3 cross-sectional edge under cost1"
    )

    out_json = OUT_DIR / "f3b_cross_sectional_momentum_20260719.json"
    out_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print("WROTE", out_json)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
