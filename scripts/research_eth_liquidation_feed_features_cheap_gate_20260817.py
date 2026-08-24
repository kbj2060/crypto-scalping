"""ETH liquidation-feed cheap gate (pre-registered design:
docs/experiments/eth_candidate_liquidation_feed_features_cheap_gate_20260817.md).

quant_ai env (duckdb). DB read_only. Gates: P (contamination/quality), B1
(liq_net_z_12 h=1 existence), B2 (ridge increment over lagged-return/vol
benchmarks, day-block bootstrap), C-lite (deciles + event lift incl. vol lift).
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
DB = ROOT / "data" / "live" / "tail_risk.duckdb"
OUT_DIR = ROOT / "tmp" / "causal_regen_20260516" / "eth_liquidation_feed_features_20260817"

# EXPLORATORY SCAN ONLY (non-decisional): valid liquidation data begins
# 2026-07-18 15:03 UTC (forceOrder endpoint defect before that -- see design doc
# section 9/10). The decisive pre-registered gate is deferred to >=8 weeks of
# valid data (~2026-09-15+).
MODE = "exploratory_scan_non_decisional"
WINDOW_START, WINDOW_END = "2026-07-18", "2026-08-12"
SPLITS = {"dev": ("2026-07-20", "2026-08-03 23:59:59"),
          "confirm": ("2026-08-04", "2026-08-11 23:59:59")}
HORIZONS = [1, 3, 12]
BOOT_SEED, BOOT_N = 20260817, 2000
DETECT_IC = 0.041

FEATS = ["liq_long_12", "liq_short_12", "liq_net_z_12", "liq_total_z_48",
         "liq_event_rate_z_48", "large_long_recent", "large_short_recent",
         "mins_since_large", "liq_asym_48", "aftershock_prob"]
BENCH = ["lag1_ret", "ret_12", "abs_ret_12", "taker_imbalance"]
TRAIL = 2880  # 2 days of minutes


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
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute("""
        select ts, long_usd_1m, short_usd_1m, liq_event_count_1m, shadow_aftershock_prob
        from tail_risk_1m
        where valid_liq_stream = true and ws_stale = false
        order by ts""").df()
    con.close()
    df["ts"] = pd.to_datetime(df.ts, utc=True)
    n_dup = int(df.ts.duplicated().sum())
    if n_dup:
        print(f"[warn] {n_dup} duplicate 1m buckets in tail_risk_1m -> keeping last", flush=True)
    return df.drop_duplicates("ts", keep="last")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # reindex onto the full 1m grid so rolling windows see real gaps as NaN
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
    out["liq_long_12"] = np.log1p(l12)
    out["liq_short_12"] = np.log1p(s12)
    trail_mean = roll_sum(total, TRAIL) / TRAIL
    eps = trail_mean * 12 * 0.01 + 1.0
    out["liq_net_z_12"] = (l12 - s12) / (trail_mean * 12 + eps)
    t48 = roll_sum(total, 48)
    mu48 = t48.rolling(TRAIL, min_periods=TRAIL // 2).mean()
    sd48 = t48.rolling(TRAIL, min_periods=TRAIL // 2).std()
    out["liq_total_z_48"] = (t48 - mu48) / sd48
    ev48 = roll_sum(g.liq_event_count_1m.fillna(0), 48)
    out["liq_event_rate_z_48"] = (ev48 - ev48.rolling(TRAIL, min_periods=TRAIL // 2).mean()) \
        / ev48.rolling(TRAIL, min_periods=TRAIL // 2).std()
    p99_l = long_.rolling(TRAIL, min_periods=TRAIL // 2).quantile(0.99).shift(1)
    p99_s = short_.rolling(TRAIL, min_periods=TRAIL // 2).quantile(0.99).shift(1)
    big_l = (long_ > p99_l) & (long_ > 0)
    big_s = (short_ > p99_s) & (short_ > 0)
    out["large_long_recent"] = big_l.rolling(12, min_periods=1).max().astype(float)
    out["large_short_recent"] = big_s.rolling(12, min_periods=1).max().astype(float)
    big_any = (big_l | big_s)
    idx = np.arange(len(g))
    last_big = pd.Series(np.where(big_any, idx, np.nan), index=g.index).ffill()
    out["mins_since_large"] = np.log1p(np.minimum(idx - last_big, 288))
    l48, s48 = roll_sum(long_, 48), roll_sum(short_, 48)
    denom = l48 + s48
    out["liq_asym_48"] = ((l48 - s48) / denom).where(denom > 0)
    out["aftershock_prob"] = g.shadow_aftershock_prob
    out["present"] = present
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    k = fetch_klines_5m()
    liq = build_features(load_liq())

    k = k.set_index("open_time")
    k["lag1_ret"] = np.log(k.close / k.close.shift(1))
    k["ret_12"] = np.log(k.close / k.close.shift(12))
    k["abs_ret_12"] = k.ret_12.abs()
    k["taker_imbalance"] = (2 * k.taker_buy_quote - k.quote_volume) / k.quote_volume.replace(0, np.nan)
    for h in HORIZONS:
        k[f"fwd_ret_{h}"] = np.log(k.close.shift(-h) / k.close)

    # liquidation state at each 5m bar close: last FULLY CLOSED 1m bucket
    # (the bucket starting exactly at bar close still accumulates -> lookahead)
    bar_close = k.index + pd.Timedelta("5min")
    feat_at_close = liq.reindex((bar_close - pd.Timedelta("1min")).floor("1min")).set_index(k.index)
    df = pd.concat([k[["close", "lag1_ret", "ret_12", "abs_ret_12", "taker_imbalance"]
                      + [f"fwd_ret_{h}" for h in HORIZONS]], feat_at_close[FEATS]], axis=1)
    df["open_time"] = df.index
    df = df.dropna(subset=["fwd_ret_1"])

    res = {"design_doc": "docs/experiments/eth_candidate_liquidation_feed_features_cheap_gate_20260817.md",
           "mode": MODE, "valid_data_epoch_start": "2026-07-18T15:03:00Z",
           "n_grid": int(len(df)), "splits": SPLITS,
           "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
           "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}

    res["gate_P_contamination"] = {
        f: float(spearmanr(df[f], df["close"], nan_policy="omit").statistic) for f in FEATS}

    def in_split(name):
        lo, hi = SPLITS[name]
        return df[(df.open_time >= pd.Timestamp(lo, tz="UTC")) & (df.open_time <= pd.Timestamp(hi, tz="UTC"))]

    ic = {}
    for sp in SPLITS:
        d = in_split(sp)
        ic[sp] = {"n": int(len(d))}
        for h in HORIZONS:
            ic[sp][f"h{h}"] = {}
            for f in FEATS + BENCH:
                pair = d[[f, f"fwd_ret_{h}"]].dropna()
                ic[sp][f"h{h}"][f] = dict(ic=float(spearmanr(pair[f], pair[f"fwd_ret_{h}"]).statistic)
                                          if len(pair) > 50 else float("nan"), n=int(len(pair)))
    res["ic_matrix"] = ic

    b1d, b1c = ic["dev"]["h1"]["liq_net_z_12"]["ic"], ic["confirm"]["h1"]["liq_net_z_12"]["ic"]
    res["gate_B1"] = dict(dev_ic=b1d, confirm_ic=b1c,
                          sign_agree=bool(np.sign(b1d) == np.sign(b1c)),
                          confirm_abs_ge_detect=bool(abs(b1c) >= DETECT_IC),
                          passed=bool(np.sign(b1d) == np.sign(b1c) and abs(b1c) >= DETECT_IC))

    dev, conf = in_split("dev"), in_split("confirm")
    use_f = BENCH + FEATS
    tr = dev.dropna(subset=use_f + ["fwd_ret_1"])
    te = conf.dropna(subset=use_f + ["fwd_ret_1"]).copy()
    mu, sd = tr[use_f].mean(), tr[use_f].std().replace(0, 1)

    def fit_pred(cols):
        m = Ridge(alpha=1.0).fit((tr[cols] - mu[cols]) / sd[cols], tr.fwd_ret_1)
        return m.predict((te[cols] - mu[cols]) / sd[cols])

    te["p_b"], te["p_f"] = fit_pred(BENCH), fit_pred(use_f)
    rho_b = float(spearmanr(te.p_b, te.fwd_ret_1).statistic)
    rho_f = float(spearmanr(te.p_f, te.fwd_ret_1).statistic)
    te["day"] = te.open_time.dt.date
    days = te.day.unique()
    rng = np.random.RandomState(BOOT_SEED)
    groups = {d: te[te.day == d] for d in days}
    deltas = []
    for _ in range(BOOT_N):
        bs = pd.concat([groups[d] for d in rng.choice(days, size=len(days), replace=True)])
        deltas.append(float(spearmanr(bs.p_f, bs.fwd_ret_1).statistic)
                      - float(spearmanr(bs.p_b, bs.fwd_ret_1).statistic))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    res["gate_B2"] = dict(n_train=int(len(tr)), n_confirm=int(len(te)),
                          rho_bench=rho_b, rho_full=rho_f, delta_rho=rho_f - rho_b,
                          boot_ci95=[float(lo), float(hi)], passed=bool(lo > 0))

    q_hi, q_lo = te.p_f.quantile(.9), te.p_f.quantile(.1)
    top, bot = te[te.p_f >= q_hi].fwd_ret_1, te[te.p_f <= q_lo].fwd_ret_1
    # event lift: bars with a large event in the last 15min vs matched random bars
    ev_mask = (df.large_long_recent > 0) | (df.large_short_recent > 0)
    ev = df[ev_mask]
    rng2 = np.random.RandomState(BOOT_SEED + 1)
    pool = df[~ev_mask].dropna(subset=["fwd_ret_1"])
    ctrl = pool.sample(min(len(ev) * 3, len(pool)), random_state=rng2)
    res["C_lite"] = dict(
        top_decile_bp=float(top.mean() * 1e4), bottom_decile_bp=float(bot.mean() * 1e4),
        longshort_gross_bp_per_trade=float((top.mean() - bot.mean()) / 2 * 1e4), roundtrip_cost_bp=11.0,
        event_n=int(len(ev)),
        event_fwd_bp=float(ev.fwd_ret_1.mean() * 1e4), ctrl_fwd_bp=float(ctrl.fwd_ret_1.mean() * 1e4),
        event_absfwd_bp=float(ev.fwd_ret_1.abs().mean() * 1e4),
        ctrl_absfwd_bp=float(ctrl.fwd_ret_1.abs().mean() * 1e4),
        vol_lift=float(ev.fwd_ret_1.abs().mean() / ctrl.fwd_ret_1.abs().mean()))

    (OUT_DIR / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k2: res[k2] for k2 in ["gate_B1", "gate_B2", "C_lite"]}, indent=1))
    print(f"\n[{MODE}] IC h1 (scanA | scanB):")
    for f in FEATS + BENCH:
        print(f"  {f:22s} {ic['dev']['h1'][f]['ic']:+.4f} | "
              f"{ic['confirm']['h1'][f]['ic']:+.4f}  (n_scanB={ic['confirm']['h1'][f]['n']})")
    print(f"\ncontamination>|0.5|: {[f for f, v in res['gate_P_contamination'].items() if abs(v) > 0.5]}")
    print(f"wrote {OUT_DIR/'summary.json'}")


if __name__ == "__main__":
    main()
