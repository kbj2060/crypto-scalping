"""ETH L2 summary-column cheap gate (pre-registered design:
docs/experiments/eth_candidate_l2_summary_features_cheap_gate_20260817.md).

Runs in the quant_ai env (needs duckdb). DB opened read_only.
Gates: P (contamination + coverage bias audit), B1 (imbalance_5 h=1 IC exists),
B2 (ridge increment over free benchmarks, day-block bootstrap CI), C-lite (decile bp).
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
DB = ROOT / "data" / "live" / "microstructure.duckdb"
OUT_DIR = ROOT / "tmp" / "causal_regen_20260516" / "eth_l2_summary_features_20260817"

WINDOW_START, WINDOW_END = "2026-05-13", "2026-08-12"
SPLITS = {"dev": ("2026-05-14", "2026-06-22 23:59:59"),
          "mid": ("2026-06-26", "2026-07-13 23:59:59"),
          "confirm": ("2026-08-02", "2026-08-11 23:59:59")}
HORIZONS = [1, 3, 12]
BOOT_SEED, BOOT_N = 20260817, 2000
DETECT_IC = 0.039  # 1.96/sqrt(~2500)

FEATS = ["imbalance_1", "imbalance_5", "imbalance_10", "imbalance_20", "imb_slope",
         "microprice_edge_bps", "spread_bps", "spread_bps_z", "log_depth20_z",
         "ofi_proxy_5", "d_imbalance_5", "d_log_depth20"]
BENCH = ["lag1_ret", "taker_imbalance", "obi_1m"]


def fetch_klines_5m() -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/klines"
    start = int(pd.Timestamp(WINDOW_START, tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp(WINDOW_END, tz="UTC").timestamp() * 1000)
    rows = []
    cur = start
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
    k = k.astype({c: float for c in ["open", "high", "low", "close", "quote_volume", "taker_buy_quote"]})
    k["open_time"] = pd.to_datetime(k.open_time, unit="ms", utc=True)
    k = k.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return k


def load_snapshots() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    s = con.execute("""
        select recorded_at_kst, mid, spread_bps, microprice_edge_bps,
               imbalance_1, imbalance_5, imbalance_10, imbalance_20,
               bid_notional_5, ask_notional_5, bid_notional_20, ask_notional_20
        from orderbook_decision_snapshots order by recorded_at_kst""").df()
    obi = con.execute("""
        select ts, obi from microstructure_1m
        where data_stale = false and obi is not null order by ts""").df()
    con.close()
    s["ts"] = pd.to_datetime(s.recorded_at_kst, utc=True)
    obi["ts"] = pd.to_datetime(obi.ts, utc=True)
    return s, obi


def build_features(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["imb_slope"] = s.imbalance_1 - s.imbalance_20
    s["spread_bps_z"] = (s.spread_bps - s.spread_bps.rolling(288, min_periods=100).mean()) \
        / s.spread_bps.rolling(288, min_periods=100).std()
    depth20 = s.bid_notional_20 + s.ask_notional_20
    ld = np.log(depth20.replace(0, np.nan))
    s["log_depth20_z"] = (ld - ld.rolling(288, min_periods=100).mean()) / ld.rolling(288, min_periods=100).std()
    dt = s.ts.diff().dt.total_seconds()
    ok = dt <= 310
    depth5 = s.bid_notional_5 + s.ask_notional_5
    s["ofi_proxy_5"] = ((s.bid_notional_5.diff() - s.ask_notional_5.diff())
                        / depth5.rolling(288, min_periods=100).mean()).where(ok)
    s["d_imbalance_5"] = s.imbalance_5.diff().where(ok)
    s["d_log_depth20"] = ld.diff().where(ok)
    return s


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    k = fetch_klines_5m()
    s, obi = load_snapshots()
    s = build_features(s)
    # slot assignment: snapshot belongs to the 5m bar containing it (floor)
    s["open_time"] = s.ts.dt.floor("5min")
    s = s.drop_duplicates("open_time", keep="last")
    k = k.set_index("open_time")
    k["lag1_ret"] = np.log(k.close / k.close.shift(1))
    k["taker_imbalance"] = (2 * k.taker_buy_quote - k.quote_volume) / k.quote_volume.replace(0, np.nan)
    for h in HORIZONS:
        k[f"fwd_ret_{h}"] = np.log(k.close.shift(-h) / k.close)
    df = s.merge(k[["close", "lag1_ret", "taker_imbalance"] + [f"fwd_ret_{h}" for h in HORIZONS]],
                 left_on="open_time", right_index=True, how="inner")
    # obi as-of join to bar close time (open_time + 5min), age <= 90s
    df = pd.merge_asof(df.sort_values("open_time"), obi.rename(columns={"obi": "obi_1m"}),
                       left_on="open_time", right_on="ts", direction="backward",
                       tolerance=pd.Timedelta("90s"), suffixes=("", "_obi"))
    res = {"design_doc": "docs/experiments/eth_candidate_l2_summary_features_cheap_gate_20260817.md",
           "n_snapshots": int(len(s)), "n_joined": int(len(df)),
           "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
           "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
           "splits": SPLITS}

    # Gate P(a): contamination vs mid (full window)
    res["gate_P_contamination"] = {
        f: float(spearmanr(df[f], df["mid"], nan_policy="omit").statistic) for f in FEATS}
    # Gate P(b): coverage bias — covered vs uncovered slots' |fwd_ret_1|
    grid = k.loc[(k.index >= pd.Timestamp(WINDOW_START, tz="UTC"))
                 & (k.index < pd.Timestamp(WINDOW_END, tz="UTC"))].copy()
    covered = grid.index.isin(df.open_time)
    a, b = grid.fwd_ret_1[covered].abs(), grid.fwd_ret_1[~covered].abs()
    res["gate_P_coverage_bias"] = dict(
        covered_slots=int(covered.sum()), uncovered_slots=int((~covered).sum()),
        covered_absret_bp_mean=float(a.mean() * 1e4), uncovered_absret_bp_mean=float(b.mean() * 1e4),
        covered_absret_bp_p90=float(a.quantile(.9) * 1e4), uncovered_absret_bp_p90=float(b.quantile(.9) * 1e4))

    # IC matrix per split/horizon/feature
    def in_split(name):
        lo, hi = SPLITS[name]
        return df[(df.open_time >= pd.Timestamp(lo, tz="UTC")) & (df.open_time <= pd.Timestamp(hi, tz="UTC"))]
    ic = {}
    for sp in SPLITS:
        d = in_split(sp)
        ic[sp] = {"n": int(len(d))}
        for h in HORIZONS:
            ic[sp][f"h{h}"] = {f: float(spearmanr(d[f], d[f"fwd_ret_{h}"], nan_policy="omit").statistic)
                               for f in FEATS + BENCH}
    res["ic_matrix"] = ic

    # Gate B1
    b1_dev, b1_conf = ic["dev"]["h1"]["imbalance_5"], ic["confirm"]["h1"]["imbalance_5"]
    res["gate_B1"] = dict(dev_ic=b1_dev, confirm_ic=b1_conf,
                          sign_agree=bool(np.sign(b1_dev) == np.sign(b1_conf)),
                          confirm_abs_ge_detect=bool(abs(b1_conf) >= DETECT_IC),
                          passed=bool(np.sign(b1_dev) == np.sign(b1_conf) and abs(b1_conf) >= DETECT_IC))

    # Gate B2: ridge increment, DEV-fit, CONFIRM-eval, day-block bootstrap of delta-rho
    dev, conf = in_split("dev"), in_split("confirm")
    use_b, use_f = BENCH, BENCH + FEATS
    tr = dev.dropna(subset=use_f + ["fwd_ret_1"])
    te = conf.dropna(subset=use_f + ["fwd_ret_1"]).copy()
    mu, sd = tr[use_f].mean(), tr[use_f].std().replace(0, 1)
    def rho(cols, frame):
        m = Ridge(alpha=1.0).fit(((tr[cols] - mu[cols]) / sd[cols]), tr.fwd_ret_1)
        p = m.predict(((frame[cols] - mu[cols]) / sd[cols]))
        return p, float(spearmanr(p, frame.fwd_ret_1).statistic)
    p_b, rho_b = rho(use_b, te)
    p_f, rho_f = rho(use_f, te)
    te["p_b"], te["p_f"] = p_b, p_f
    te["day"] = te.open_time.dt.date
    days = te.day.unique()
    rng = np.random.RandomState(BOOT_SEED)
    deltas = []
    for _ in range(BOOT_N):
        pick = rng.choice(days, size=len(days), replace=True)
        bs = pd.concat([te[te.day == d] for d in pick])
        deltas.append(float(spearmanr(bs.p_f, bs.fwd_ret_1).statistic)
                      - float(spearmanr(bs.p_b, bs.fwd_ret_1).statistic))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    res["gate_B2"] = dict(n_train=int(len(tr)), n_confirm=int(len(te)),
                          rho_bench=rho_b, rho_full=rho_f, delta_rho=rho_f - rho_b,
                          boot_ci95=[float(lo), float(hi)], passed=bool(lo > 0))

    # C-lite: decile long-short gross bp on CONFIRM (full-model score)
    q_hi, q_lo = te.p_f.quantile(.9), te.p_f.quantile(.1)
    top, bot = te[te.p_f >= q_hi].fwd_ret_1, te[te.p_f <= q_lo].fwd_ret_1
    res["C_lite"] = dict(top_decile_bp=float(top.mean() * 1e4), bottom_decile_bp=float(bot.mean() * 1e4),
                         longshort_gross_bp_per_trade=float((top.mean() - bot.mean()) / 2 * 1e4),
                         roundtrip_cost_bp=11.0)

    (OUT_DIR / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k2: res[k2] for k2 in ["gate_P_coverage_bias", "gate_B1", "gate_B2", "C_lite"]}, indent=1))
    print("\nIC h1 (dev | confirm):")
    for f in FEATS + BENCH:
        print(f"  {f:22s} {ic['dev']['h1'][f]:+.4f} | {ic['confirm']['h1'][f]:+.4f}")
    print(f"\ncontamination>|0.5|: {[f for f, v in res['gate_P_contamination'].items() if abs(v) > 0.5]}")
    print(f"wrote {OUT_DIR/'summary.json'}")


if __name__ == "__main__":
    main()
