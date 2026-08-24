"""ETH information-time sampling A/B cheap gate (pre-registered design:
docs/experiments/eth_candidate_infotime_sampling_ab_20260817.md).

Builds calendar-5m / dollar / volume / tick bars from the SAME 1m stream,
computes an identical 19-column causal feature contract + fixed-horizon label
on each arm's own clock, trains HGB with 5 pre-registered random seeds, and
evaluates skill (bacc / OvR AUC / IC) plus a minimal cost-aware strategy in
calendar time against max(always_long, always_short, flat).

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC_1M = ROOT / "data" / "training_features_1m.csv"
OUT_DIR = ROOT / "tmp" / "causal_regen_20260516" / "eth_infotime_sampling_ab_20260817"

WARMUP_START = "2024-12-01"
TRAIN_START, TRAIN_END = "2025-01-01", "2025-08-31 23:59:59"
VAL_START, VAL_END = "2025-09-01", "2025-12-31 23:59:59"
OOS_START, OOS_END = "2026-01-01", "2026-03-31 23:59:59"

SEEDS = [1491474210, 163789868, 1345858477, 922652315, 1247871276]
HORIZON = 12          # label/hold horizon in bars (arm's own clock)
DEADBAND = 0.0010     # 10bp label deadband
COST_SIDE = 0.00055   # taker 4.5bp + slippage 1bp per side (cost1)
TAU_GRID = [0.40, 0.45, 0.50, 0.55, 0.60]
SUBSAMPLE = 0.90

HGB_KW = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=200, l2_regularization=1.0, early_stopping=False)


def load_1m() -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
    df = pd.read_csv(SRC_1M, usecols=cols, parse_dates=["timestamp"])
    df = df[(df.timestamp >= WARMUP_START) & (df.timestamp <= OOS_END)]
    return df.sort_values("timestamp").reset_index(drop=True)


AGG = dict(open=("open", "first"), high=("high", "max"), low=("low", "min"),
           close=("close", "last"), volume=("volume", "sum"),
           quote_volume=("quote_volume", "sum"), trades=("trades", "sum"),
           taker_buy_quote=("taker_buy_quote", "sum"),
           timestamp=("timestamp", "last"), n_1m=("timestamp", "size"))


def build_calendar_5m(df: pd.DataFrame) -> pd.DataFrame:
    g = df.timestamp.dt.floor("5min")
    out = df.groupby(g, sort=True).agg(**AGG).reset_index(drop=True)
    return out


def build_threshold_bars(df: pd.DataFrame, field: str, theta: float) -> pd.DataFrame:
    c = df[field].to_numpy(dtype=np.float64)
    gid = ((np.cumsum(c) - c) // theta).astype(np.int64)
    out = df.groupby(gid, sort=True).agg(**AGG).reset_index(drop=True)
    return out


def add_features_labels(b: pd.DataFrame) -> pd.DataFrame:
    b = b.copy()
    o, h, l, c = b.open, b.high, b.low, b.close
    qv = b.quote_volume.replace(0, np.nan)
    lr = np.log(c / c.shift(1))
    b["log_return"] = lr
    b["ret_12"] = np.log(c / c.shift(12))
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    b["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    b["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    bw = (4 * sd) / mid
    b["bb_width_z_288"] = (bw - bw.rolling(288).mean()) / bw.rolling(288).std()
    hl2 = np.log(h / l) ** 2
    b["parkinson_vol_20"] = np.sqrt(hl2.rolling(20).mean() / (4 * np.log(2)))
    gk = 0.5 * hl2 - (2 * np.log(2) - 1) * (np.log(c / o) ** 2)
    b["garman_klass_vol_20"] = np.sqrt(gk.clip(lower=0).rolling(20).mean())
    rv = lr.rolling(20).std()
    b["vol_z_288"] = (rv - rv.rolling(288).mean()) / rv.rolling(288).std()
    ti = (2 * b.taker_buy_quote - b.quote_volume) / qv
    b["taker_imbalance"] = ti
    b["taker_imbalance_z_48"] = (ti - ti.rolling(48).mean()) / ti.rolling(48).std()
    signed = (2 * b.taker_buy_quote - b.quote_volume)
    cvd = signed.rolling(48).sum()
    b["cvd_48"] = (cvd - cvd.rolling(288).mean()) / cvd.rolling(288).std()
    tr_ = b.trades
    b["trade_intensity_z_288"] = (tr_ - tr_.rolling(288).mean()) / tr_.rolling(288).std()
    b["mean_reversion_z_20"] = (c - mid) / sd.replace(0, np.nan)
    ema48 = c.ewm(span=48, adjust=False).mean()
    b["ema_slope_48"] = ema48.pct_change()
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    rng = h.rolling(14).max() - l.rolling(14).min()
    b["chop_index_14"] = 100 * np.log10(tr.rolling(14).sum() / rng.replace(0, np.nan)) / np.log10(14)
    body_hi, body_lo = np.maximum(o, c), np.minimum(o, c)
    b["wick_ratio"] = ((h - body_hi) + (body_lo - l)) / (h - l).replace(0, np.nan)
    hh = b.timestamp.dt.hour + b.timestamp.dt.minute / 60
    b["hour_sin"], b["hour_cos"] = np.sin(2 * np.pi * hh / 24), np.cos(2 * np.pi * hh / 24)
    am = lr.abs() / qv
    b["amihud_z_288"] = (am - am.rolling(288).mean()) / am.rolling(288).std()
    b["fwd_ret_12"] = np.log(c.shift(-HORIZON) / c)
    b["label"] = np.where(b.fwd_ret_12 > DEADBAND, 1, np.where(b.fwd_ret_12 < -DEADBAND, -1, 0))
    return b


FEATS = ["log_return", "ret_12", "rsi_14", "macd_hist", "bb_width_z_288",
         "parkinson_vol_20", "garman_klass_vol_20", "vol_z_288", "taker_imbalance",
         "taker_imbalance_z_48", "cvd_48", "trade_intensity_z_288", "mean_reversion_z_20",
         "ema_slope_48", "chop_index_14", "wick_ratio", "hour_sin", "hour_cos", "amihud_z_288"]


def split_masks(b: pd.DataFrame):
    ts = b.timestamp
    train = (ts >= TRAIN_START) & (ts <= TRAIN_END)
    val = (ts >= VAL_START) & (ts <= VAL_END)
    oos = (ts >= OOS_START) & (ts <= OOS_END)
    # purge label overlap at split tails
    for m in (train, val):
        idx = np.where(m)[0]
        if len(idx) > HORIZON:
            m.iloc[idx[-HORIZON:]] = False
    return train, val, oos


def simulate(b: pd.DataFrame, p_long: np.ndarray, p_short: np.ndarray, mask: pd.Series,
             tau: float) -> dict:
    idx = np.where(mask.to_numpy())[0]
    o = b.open.to_numpy()
    n = len(b)
    pnl_events, trade_rets = [], []
    pos, entry_i, exit_i = 0, -1, -1
    for i in idx:
        pl, ps = p_long[i], p_short[i]
        sig = 0
        if max(pl, ps) > tau:
            sig = 1 if pl >= ps else -1
        if pos != 0 and (i >= exit_i or (sig != 0 and sig != pos)):
            j = min(i + 1, n - 1)
            r = pos * np.log(o[j] / o[entry_i]) - 2 * COST_SIDE
            pnl_events.append((j, r)); trade_rets.append(r)
            pos = 0
        if pos == 0 and sig != 0 and i + 1 + HORIZON < n:
            pos, entry_i, exit_i = sig, i + 1, i + 1 + HORIZON
    if pos != 0:
        r = pos * np.log(o[min(exit_i, n - 1)] / o[entry_i]) - 2 * COST_SIDE
        pnl_events.append((min(exit_i, n - 1), r)); trade_rets.append(r)
    eq = np.zeros(0)
    if pnl_events:
        s = pd.Series([r for _, r in pnl_events]).cumsum()
        eq = s.to_numpy()
    pnl = float(eq[-1]) if len(eq) else 0.0
    mdd = float((pd.Series(eq).cummax() - pd.Series(eq)).max()) if len(eq) else 0.0
    days = max((b.timestamp[mask].iloc[-1] - b.timestamp[mask].iloc[0]).days, 1) if mask.any() else 1
    gross_bp = float(np.mean([r + 2 * COST_SIDE for r in trade_rets]) * 1e4) if trade_rets else 0.0
    return dict(pnl=pnl, mdd=mdd, trades=len(trade_rets), trades_per_day=len(trade_rets) / days,
                gross_edge_bp_per_trade=gross_bp)


def benchmarks(b: pd.DataFrame, mask: pd.Series) -> dict:
    c = b.close[mask]
    if not len(c):
        return dict(always_long=0.0, always_short=0.0, flat=0.0, best=0.0)
    ret = float(np.log(c.iloc[-1] / c.iloc[0]))
    al, ash = ret - 2 * COST_SIDE, -ret - 2 * COST_SIDE
    return dict(always_long=al, always_short=ash, flat=0.0, best=max(al, ash, 0.0))


def run_arm(name: str, bars: pd.DataFrame) -> dict:
    b = add_features_labels(bars)
    train_m, val_m, oos_m = split_masks(b)
    feat_ok = b[FEATS].notna().all(axis=1) & np.isfinite(b[FEATS]).all(axis=1)
    lab_ok = b.fwd_ret_12.notna()
    stats = {}
    for sp, m in [("train", train_m), ("val", val_m), ("oos", oos_m)]:
        r = b.log_return[m & feat_ok]
        stats[sp] = dict(n_bars=int(m.sum()), kurtosis=float(kurtosis(r.dropna())),
                         qv_cv=float(b.quote_volume[m].std() / b.quote_volume[m].mean()),
                         med_bar_min=float(b.timestamp[m].diff().dt.total_seconds().median() / 60 if m.sum() > 1 else np.nan))
    Xtr_m = train_m & feat_ok & lab_ok
    seeds_out = []
    p_long_acc = np.zeros(len(b)); p_short_acc = np.zeros(len(b))
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        tr_idx = np.where(Xtr_m)[0]
        sub = rng.choice(tr_idx, size=int(len(tr_idx) * SUBSAMPLE), replace=False)
        clf = HistGradientBoostingClassifier(random_state=seed, **HGB_KW)
        clf.fit(b[FEATS].iloc[sub], b.label.iloc[sub])
        proba = clf.predict_proba(b[FEATS].where(feat_ok).fillna(0.0))
        cls = list(clf.classes_)
        pl = proba[:, cls.index(1)] if 1 in cls else np.zeros(len(b))
        ps = proba[:, cls.index(-1)] if -1 in cls else np.zeros(len(b))
        p_long_acc += pl / len(SEEDS); p_short_acc += ps / len(SEEDS)
        srow = dict(seed=seed)
        for sp, m in [("val", val_m), ("oos", oos_m)]:
            mm = (m & feat_ok & lab_ok).to_numpy()
            y = b.label.to_numpy()[mm]
            edge = (pl - ps)[mm]
            pred = np.where(edge > 0, 1, -1)
            srow[f"{sp}_bacc_nonneutral"] = float(np.nanmean([
                (pred[y == 1] == 1).mean() if (y == 1).any() else np.nan,
                (pred[y == -1] == -1).mean() if (y == -1).any() else np.nan]))
            srow[f"{sp}_ic"] = float(spearmanr(edge, b.fwd_ret_12.to_numpy()[mm]).statistic)
            try:
                srow[f"{sp}_auc_long"] = float(roc_auc_score((y == 1).astype(int), pl[mm]))
                srow[f"{sp}_auc_short"] = float(roc_auc_score((y == -1).astype(int), ps[mm]))
            except ValueError:
                srow[f"{sp}_auc_long"] = srow[f"{sp}_auc_short"] = float("nan")
        seeds_out.append(srow)
    # seed-mean probabilities drive one shared economic sim per arm (tau on VAL only)
    best_tau, best_val = None, -1e9
    for tau in TAU_GRID:
        r = simulate(b, p_long_acc, p_short_acc, val_m & feat_ok, tau)
        if r["pnl"] > best_val:
            best_val, best_tau = r["pnl"], tau
    econ = {"tau_selected_on_val": best_tau,
            "val": simulate(b, p_long_acc, p_short_acc, val_m & feat_ok, best_tau),
            "oos": simulate(b, p_long_acc, p_short_acc, oos_m & feat_ok, best_tau),
            "bench_val": benchmarks(b, val_m), "bench_oos": benchmarks(b, oos_m)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    b.loc[train_m | val_m | oos_m, ["timestamp", "open", "high", "low", "close",
                                    "quote_volume", "n_1m"]].to_csv(OUT_DIR / f"bars_{name}.csv", index=False)
    return dict(arm=name, dist_stats=stats, seeds=seeds_out, econ=econ)


def main():
    df = load_1m()
    tr1m = df[(df.timestamp >= TRAIN_START) & (df.timestamp <= TRAIN_END)]
    n5 = len(tr1m.timestamp.dt.floor("5min").unique())
    theta_d = float(tr1m.quote_volume.sum() / n5)
    theta_v = float(tr1m.volume.sum() / n5)
    theta_t = float(tr1m.trades.sum() / n5)
    arms = {
        "cal5m": build_calendar_5m(df),
        "dollar_1x": build_threshold_bars(df, "quote_volume", theta_d),
        "volume_1x": build_threshold_bars(df, "volume", theta_v),
        "tick_1x": build_threshold_bars(df, "trades", theta_t),
        "dollar_05x": build_threshold_bars(df, "quote_volume", theta_d * 0.5),
        "dollar_2x": build_threshold_bars(df, "quote_volume", theta_d * 2.0),
    }
    results = {"design_doc": "docs/experiments/eth_candidate_infotime_sampling_ab_20260817.md",
               "thetas": dict(dollar=theta_d, volume=theta_v, tick=theta_t),
               "seeds": SEEDS, "horizon_bars": HORIZON, "deadband": DEADBAND,
               "cost_per_side": COST_SIDE,
               "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
               "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
               "arms": {}}
    for name, bars in arms.items():
        print(f"[arm] {name}: {len(bars)} bars", flush=True)
        results["arms"][name] = run_arm(name, bars)
    out = OUT_DIR / "summary.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}", flush=True)
    # console digest
    for name, r in results["arms"].items():
        sv = pd.DataFrame(r["seeds"])
        e = r["econ"]
        print(f"{name:10s} | OOS IC {sv.oos_ic.mean():+.4f}±{sv.oos_ic.std():.4f} "
              f"| OOS bacc {sv.oos_bacc_nonneutral.mean():.4f} "
              f"| tau {e['tau_selected_on_val']} VAL pnl {e['val']['pnl']:+.4f} (bench {e['bench_val']['best']:+.4f}) "
              f"| OOS pnl {e['oos']['pnl']:+.4f} (bench {e['bench_oos']['best']:+.4f}) trades {e['oos']['trades']}", flush=True)


if __name__ == "__main__":
    main()
