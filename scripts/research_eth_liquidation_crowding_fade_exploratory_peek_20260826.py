"""ETH §13 crowding-conditional liquidation-fade -- EXPLORATORY PEEK, NON-DECISIONAL.

Design doc: docs/experiments/eth_candidate_liquidation_feed_features_cheap_gate_20260817.md §13.
The pre-registered decisive gate is still >=2026-09-15 (tail_risk_1m valid 8 weeks) and MUST be
re-run fresh then with its own dated script -- this run's output is NOT eligible for any
promotion/kill decision, matching this repo's existing exploratory-scan convention (design doc
§10/§11; scripts/research_eth_liquidation_feed_features_cheap_gate_20260817.py's own
MODE="exploratory_scan_non_decisional").

User asked (2026-08-26) whether §13 could be peeked at with current data despite the 09-15 gate.
Data-maturity check done first (see memory eth_liquidation_crowding_conditional_fade_arm_
preregistration_20260823 2026-08-26 addendum):
  - tail_risk_1m valid epoch: 39.1/56 days needed -> event leg is real but thin, growing daily.
  - oi_lsratio.duckdb poller alone: only ~4.7/15 days -- BUT §13 itself specifies merging with
    data/TOTAL_ETHUSDT_metrics_2024_2026.csv for the OI/position-ratio legs, which covers
    2024-01..2026-08-22 at 5m resolution and dovetails almost exactly where the live poller
    starts (08-22) -- so those two legs are NOT actually data-starved once merged as designed.
  - funding: data/TOTAL_ETHUSDT_fundingRate_2025_2026.csv (correct ETHUSDT symbol, fixed 08-24)
    only covers through 2026-07-31 -- this IS the real bottleneck for the full 3-leg crowding
    condition; it bounds the usable event window to 2026-07-18..07-31 (~13 days) if required.

quant_ai conda env (duckdb, sklearn). All DB connections read_only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
TAIL_RISK_DB = ROOT / "data" / "live" / "tail_risk.duckdb"
OI_DB = ROOT / "data" / "live" / "oi_lsratio.duckdb"
OI_ARCHIVE_CSV = ROOT / "data" / "TOTAL_ETHUSDT_metrics_2024_2026.csv"
FUNDING_CSV = ROOT / "data" / "TOTAL_ETHUSDT_fundingRate_2025_2026.csv"
OUT_DIR = ROOT / "tmp" / "causal_regen_20260516" / "eth_liquidation_crowding_fade_exploratory_peek_20260826"

MODE = "exploratory_peek_non_decisional"
VALID_EPOCH_START = pd.Timestamp("2026-07-18 15:03:00", tz="UTC")
WINDOW_START, WINDOW_END = "2026-07-18", "2026-08-26"
HORIZON = 3  # matches design doc §12 item2 / §13 primary horizon (3 x 5m = 15min)
TRAIL = 2880  # 2 days of minutes -- liq_net_z_12 formula unchanged from §12 script
MIN_DAYS_PER_DIRECTION = 15  # pre-registered threshold
BOOT_SEED, BOOT_N = 20260826, 2000
BENCH = ["lag1_ret", "ret_12", "abs_ret_12", "taker_imbalance"]


def fetch_klines_5m() -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/klines"
    cur = int(pd.Timestamp(WINDOW_START, tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp(WINDOW_END, tz="UTC").timestamp() * 1000)
    rows = []
    while cur < end:
        r = requests.get(url, params=dict(symbol="ETHUSDT", interval="5m", startTime=cur,
                                          endTime=end, limit=1500), timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + 300_000
        time.sleep(0.15)
    k = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume",
                                    "close_time", "quote_volume", "trades", "taker_buy_base",
                                    "taker_buy_quote", "ignore"])
    k = k.astype({c: float for c in ["close", "quote_volume", "taker_buy_quote"]})
    k["open_time"] = pd.to_datetime(k.open_time, unit="ms", utc=True)
    return k.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def load_liq() -> pd.DataFrame:
    con = duckdb.connect(str(TAIL_RISK_DB), read_only=True)
    df = con.execute("""
        select ts, long_usd_1m, short_usd_1m
        from tail_risk_1m
        where valid_liq_stream = true and ws_stale = false
        order by ts""").df()
    con.close()
    df["ts"] = pd.to_datetime(df.ts, utc=True)
    return df.drop_duplicates("ts", keep="last")


def build_liq_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.set_index("ts").reindex(pd.date_range(df.ts.min(), df.ts.max(), freq="1min", tz="UTC"))
    present = g.long_usd_1m.notna()

    def roll_sum(x, w):
        s = x.rolling(w, min_periods=1).sum()
        cov = present.rolling(w, min_periods=1).mean()
        return s.where(cov >= 0.8)

    long_, short_ = g.long_usd_1m.fillna(0), g.short_usd_1m.fillna(0)
    total = long_ + short_
    out = pd.DataFrame(index=g.index)
    l12, s12 = roll_sum(long_, 12), roll_sum(short_, 12)
    trail_mean = roll_sum(total, TRAIL) / TRAIL
    eps = trail_mean * 12 * 0.01 + 1.0
    out["liq_net_z_12"] = (l12 - s12) / (trail_mean * 12 + eps)
    out.loc[out.index < VALID_EPOCH_START + pd.Timedelta(minutes=TRAIL), "liq_net_z_12"] = np.nan
    return out


def load_oi_position() -> pd.DataFrame:
    """Archive (2024-01..2026-08-22, 5m) + live poller (2026-08-22-> , 5m), merged per §13's own
    'OI는 oi_lsratio.duckdb+아카이브 병합' instruction. sum_toptrader_long_short_ratio (archive)
    and top_pos_ls_ratio (live poller) are the same underlying topLongShortPositionRatio metric,
    just renamed between the two collectors."""
    arc = pd.read_csv(OI_ARCHIVE_CSV, usecols=["create_time", "sum_open_interest_value",
                                                "sum_toptrader_long_short_ratio"])
    arc["ts"] = pd.to_datetime(arc.create_time, utc=True)
    arc = arc.rename(columns={"sum_toptrader_long_short_ratio": "top_pos_ls_ratio"})[
        ["ts", "sum_open_interest_value", "top_pos_ls_ratio"]]

    con = duckdb.connect(str(OI_DB), read_only=True)
    live = con.execute("""select ts, sum_open_interest_value, top_pos_ls_ratio
                           from oi_lsratio_5m order by ts""").df()
    con.close()
    live["ts"] = pd.to_datetime(live.ts, utc=True)

    merged = pd.concat([arc[arc.ts < live.ts.min()], live]).drop_duplicates("ts").sort_values("ts")
    return merged.set_index("ts")


def load_funding() -> pd.Series:
    f = pd.read_csv(FUNDING_CSV)
    f["ts"] = pd.to_datetime(f.calc_time, utc=True)
    return f.set_index("ts").sort_index()["last_funding_rate"]


def causal_expanding_quantile(s: pd.Series, q: float, min_periods: int) -> pd.Series:
    """Quantile threshold at t computed from all values strictly BEFORE t -- no lookahead,
    matches §13's '인과적, 버스트 이전 시점만 사용'."""
    return s.shift(1).expanding(min_periods=min_periods).quantile(q)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] fetching klines...", flush=True)
    k = fetch_klines_5m()
    print(f"  {len(k)} 5m bars {k.open_time.min()}..{k.open_time.max()}", flush=True)

    print("[2/4] loading liquidation feed...", flush=True)
    liq_1m = build_liq_features(load_liq())

    print("[3/4] loading OI/position (archive+live merge) and funding...", flush=True)
    oi_pos = load_oi_position()
    funding = load_funding()
    print(f"  oi_pos {oi_pos.index.min()}..{oi_pos.index.max()} ({len(oi_pos)} rows)", flush=True)
    print(f"  funding {funding.index.min()}..{funding.index.max()} ({len(funding)} rows)", flush=True)

    k = k.set_index("open_time")
    k["lag1_ret"] = np.log(k.close / k.close.shift(1))
    k["ret_12"] = np.log(k.close / k.close.shift(12))
    k["abs_ret_12"] = k.ret_12.abs()
    k["taker_imbalance"] = (2 * k.taker_buy_quote - k.quote_volume) / k.quote_volume.replace(0, np.nan)
    k[f"fwd_ret_{HORIZON}"] = np.log(k.close.shift(-HORIZON) / k.close)

    bar_close = k.index + pd.Timedelta("5min")
    liq_at_close = liq_1m.reindex((bar_close - pd.Timedelta("1min")).floor("1min")).set_index(k.index)
    oi_at_close = oi_pos.reindex(k.index, method="ffill")
    funding_at_close = funding.reindex(k.index, method="ffill")
    # funding.csv ends 2026-07-31 -- plain ffill would silently extrapolate its last known sign
    # 26 more days to 08-26 with no real data behind it. Mask anything past the true last
    # observation to NaN so those bars are honestly excluded from the funding leg, not guessed.
    funding_at_close[k.index > funding.index.max()] = np.nan

    df = pd.concat([k[["close", "lag1_ret", "ret_12", "abs_ret_12", "taker_imbalance",
                        f"fwd_ret_{HORIZON}"]], liq_at_close, oi_at_close], axis=1)
    df["funding_rate"] = funding_at_close
    print(f"  merged grid: {len(df)} bars", flush=True)

    print("[4/4] causal expanding thresholds + B2 gate...", flush=True)
    df["oi_hi_thresh"] = causal_expanding_quantile(df["sum_open_interest_value"], 2 / 3, 2000)
    df["pos_hi_thresh"] = causal_expanding_quantile(df["top_pos_ls_ratio"], 2 / 3, 2000)
    df["pos_lo_thresh"] = causal_expanding_quantile(df["top_pos_ls_ratio"], 1 / 3, 2000)
    df["liq_hi_thresh"] = causal_expanding_quantile(df["liq_net_z_12"], 0.9, 864)
    df["liq_lo_thresh"] = causal_expanding_quantile(df["liq_net_z_12"], 0.1, 864)

    oi_high = df["sum_open_interest_value"] > df["oi_hi_thresh"]
    pos_crowded_long = df["top_pos_ls_ratio"] > df["pos_hi_thresh"]
    pos_crowded_short = df["top_pos_ls_ratio"] < df["pos_lo_thresh"]
    funding_pos = df["funding_rate"] > 0
    funding_neg = df["funding_rate"] < 0

    burst_bottom = df["liq_net_z_12"] >= df["liq_hi_thresh"]   # long-liq-dominant burst
    burst_top = df["liq_net_z_12"] <= df["liq_lo_thresh"]      # short-liq-dominant burst
    crowd_bottom = oi_high & pos_crowded_long & funding_pos
    crowd_top = oi_high & pos_crowded_short & funding_neg
    fire_bottom = (burst_bottom & crowd_bottom).fillna(False)
    fire_top = (burst_top & crowd_top).fillna(False)

    df["crowding_match"] = np.select(
        [df["liq_net_z_12"] > 0, df["liq_net_z_12"] < 0],
        [crowd_bottom.fillna(False), crowd_top.fillna(False)], default=False)

    n_days_bottom = int(df.index[fire_bottom].normalize().nunique())
    n_days_top = int(df.index[fire_top].normalize().nunique())
    funding_bound_bars = int((df["funding_rate"].notna()).sum())

    res = {
        "design_doc": "docs/experiments/eth_candidate_liquidation_feed_features_cheap_gate_20260817.md#13",
        "mode": MODE, "NOT_ELIGIBLE_FOR_PROMOTION_OR_KILL_DECISION": True,
        "decisive_gate_still_at": "2026-09-15",
        "funding_data_coverage": [str(funding.index.min()), str(funding.index.max())],
        "n_bars_with_funding_available": funding_bound_bars,
        "fire_bottom_n_bars": int(fire_bottom.sum()), "fire_bottom_n_days": n_days_bottom,
        "fire_top_n_bars": int(fire_top.sum()), "fire_top_n_days": n_days_top,
        "min_days_threshold": MIN_DAYS_PER_DIRECTION,
        "bottom_meets_threshold": n_days_bottom >= MIN_DAYS_PER_DIRECTION,
        "top_meets_threshold": n_days_top >= MIN_DAYS_PER_DIRECTION,
    }

    valid = df.dropna(subset=[f"fwd_ret_{HORIZON}", "liq_net_z_12"] + BENCH).sort_index()
    res["n_total_valid_bars"] = int(len(valid))

    if len(valid) > 200:
        mid = valid.index[len(valid) // 2]
        tr, te = valid[valid.index < mid].copy(), valid[valid.index >= mid].copy()
        for part in (tr, te):
            part["interaction"] = part["liq_net_z_12"] * part["crowding_match"].astype(float)
        use_bench, use_liq, use_full = BENCH, BENCH + ["liq_net_z_12"], BENCH + ["liq_net_z_12", "interaction"]
        mu, sd = tr[use_full].mean(), tr[use_full].std().replace(0, 1)

        def fit_pred(cols):
            m = Ridge(alpha=1.0).fit((tr[cols] - mu[cols]) / sd[cols], tr[f"fwd_ret_{HORIZON}"])
            return m.predict((te[cols] - mu[cols]) / sd[cols])

        te["p_bench"], te["p_liq"], te["p_full"] = fit_pred(use_bench), fit_pred(use_liq), fit_pred(use_full)
        rho_bench = float(spearmanr(te.p_bench, te[f"fwd_ret_{HORIZON}"]).statistic)
        rho_liq = float(spearmanr(te.p_liq, te[f"fwd_ret_{HORIZON}"]).statistic)
        rho_full = float(spearmanr(te.p_full, te[f"fwd_ret_{HORIZON}"]).statistic)

        te["day"] = te.index.normalize()
        days = te.day.unique()
        rng = np.random.RandomState(BOOT_SEED)
        groups = {d: te[te.day == d] for d in days}
        deltas = []
        for _ in range(BOOT_N):
            bs = pd.concat([groups[d] for d in rng.choice(days, size=len(days), replace=True)])
            deltas.append(float(spearmanr(bs.p_full, bs[f"fwd_ret_{HORIZON}"]).statistic)
                          - float(spearmanr(bs.p_liq, bs[f"fwd_ret_{HORIZON}"]).statistic))
        lo, hi = np.percentile(deltas, [2.5, 97.5])

        res["gate_B2_interaction"] = dict(
            n_train=int(len(tr)), n_eval=int(len(te)), n_eval_days=int(len(days)),
            rho_bench=rho_bench, rho_liq=rho_liq, rho_full=rho_full,
            delta_rho_full_vs_liq=rho_full - rho_liq,
            boot_ci95=[float(lo), float(hi)], passed_ci_excludes_0=bool(lo > 0 or hi < 0))
    else:
        res["gate_B2_interaction"] = "skipped -- fewer than 200 valid bars total"

    (OUT_DIR / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    print(f"\nwrote {OUT_DIR/'summary.json'}")


if __name__ == "__main__":
    main()
