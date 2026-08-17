"""ETH multi-timescale direction-existence screen (cheap gate, not a promotion run).

Resamples the same 5m source to {5m,15m,30m,1h,2h,4h} and applies an identical
methodology at every scale: model-free stats (AC/VR/sign persistence), a
feature-IC screen with circular-shift permutation nulls, a 5-random-seed
LightGBM next-bar direction probe, and a cost-aware benchmark backtest vs
max(always_long, always_short).

Pre-registered design: docs/experiments/eth_candidate_direction_timescale_resample_screen_20260817.md
fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/experiments/eth_direction_timescale_resample_screen_20260817_results.json"

TIMESCALES = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "2h": "2h", "4h": "4h"}
SPLITS = {
    "TRAIN": ("2024-01-01", "2025-08-31 23:59:59"),
    "VAL": ("2025-09-01", "2025-12-31 23:59:59"),
    "OOS": ("2026-01-01", "2026-02-28 23:59:59"),
}
SEEDS = [int(s) for s in np.random.default_rng(20260817).integers(1, 999_999, 5)]
EMBARGO = 24          # bars dropped after each split boundary on the later side
N_PERM = 200          # circular-shift permutations for IC null
N_BOOT = 500          # bootstrap resamples for AUC z
COST_ROUNDTRIP = 0.0010
TAUS = [0.00, 0.03, 0.05]

LAGS = [1, 2, 3, 6, 12, 24]


def load_5m() -> pd.DataFrame:
    frames = []
    for year in (2024, 2025, 2026):
        df = pd.read_csv(ROOT / f"data/rl_training_{year}_unified.csv",
                         usecols=["timestamp", "open", "high", "low", "close",
                                  "volume", "trades", "taker_buy_base"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    return df


def resample(df5: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "5min":
        return df5.copy()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "trades": "sum", "taker_buy_base": "sum"}
    out = df5.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["close"])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c = df["close"]
    logc = np.log(c)
    for k in LAGS:
        f[f"ret_{k}"] = logc.diff(k)
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = 100 - 100 / (1 + up / down.replace(0, np.nan))
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    f["bb_width"] = 4 * sd20 / ma20
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - c.shift()).abs(),
                    (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct14"] = tr.rolling(14).mean() / c
    r1 = logc.diff()
    vol20 = r1.rolling(20).std()
    f["vol_z"] = (vol20 - vol20.rolling(100).mean()) / vol20.rolling(100).std()
    lv = np.log1p(df["volume"])
    f["volume_z"] = (lv - lv.rolling(100).mean()) / lv.rolling(100).std()
    f["taker_ratio"] = (df["taker_buy_base"] / df["volume"].replace(0, np.nan)) - 0.5
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    f["close_pos"] = (c - df["low"]) / rng - 0.5
    f["y_ret"] = logc.diff().shift(-1)
    return f


def split_masks(index: pd.DatetimeIndex) -> dict:
    masks = {}
    prev_end = None
    for name, (start, end) in SPLITS.items():
        m = (index >= start) & (index <= end)
        if prev_end is not None:
            # embargo: drop the first EMBARGO bars of the later split
            idx = np.flatnonzero(m)
            m = m.copy()
            m[idx[:EMBARGO]] = False
        masks[name] = m
        prev_end = end
    return masks


def model_free_stats(ret: np.ndarray) -> dict:
    r = ret[~np.isnan(ret)]
    n = len(r)
    out = {"n": int(n)}
    rc = r - r.mean()
    denom = (rc ** 2).sum()
    acs = {}
    for lag in (1, 2, 3, 6, 12):
        ac = float((rc[lag:] * rc[:-lag]).sum() / denom)
        acs[str(lag)] = round(ac, 5)
    out["ac"] = acs
    out["ac1_z"] = round(acs["1"] * np.sqrt(n), 2)
    vr = {}
    for q in (2, 4, 8):
        # Lo-MacKinlay heteroskedasticity-robust VR z
        mu = r.mean()
        var1 = ((r - mu) ** 2).sum() / (n - 1)
        rq = np.convolve(r, np.ones(q), mode="valid")
        varq = ((rq - q * mu) ** 2).sum() / (len(rq) - 1)
        vr_q = varq / (q * var1)
        delta = np.zeros(q - 1)
        for j in range(1, q):
            num = (((r[j:] - mu) ** 2) * ((r[:-j] - mu) ** 2)).sum()
            delta[j - 1] = num / (((r - mu) ** 2).sum() ** 2 / n)
        theta = float((np.array([2 * (q - j) / q for j in range(1, q)]) ** 2 * delta).sum())
        z = (vr_q - 1) / np.sqrt(theta / n) if theta > 0 else np.nan
        vr[str(q)] = {"vr": round(float(vr_q), 4), "z": round(float(z), 2)}
    out["variance_ratio"] = vr
    s = np.sign(r)
    s = s[s != 0]
    same = (s[1:] == s[:-1]).sum()
    bt = binomtest(int(same), len(s) - 1, 0.5)
    out["sign_persistence"] = {"p_same": round(same / (len(s) - 1), 4),
                               "pvalue": round(bt.pvalue, 5)}
    return out


def ic_screen(feat: pd.DataFrame, feature_cols: list, masks: dict, rng: np.random.Generator) -> dict:
    results = {}
    ranks_cache = {}
    for split, m in masks.items():
        sub = feat.loc[m, feature_cols + ["y_ret"]].dropna()
        if len(sub) < 200:
            continue
        y = rankdata(sub["y_ret"].to_numpy())
        y = (y - y.mean()) / y.std()
        F = np.column_stack([rankdata(sub[c].to_numpy()) for c in feature_cols])
        F = (F - F.mean(axis=0)) / F.std(axis=0)
        n = len(y)
        obs = F.T @ y / n  # spearman IC vector
        shifts = rng.integers(50, n - 50, size=N_PERM)
        Y = np.column_stack([np.roll(y, int(s)) for s in shifts])
        null = F.T @ Y / n  # (k, N_PERM)
        z = (obs - null.mean(axis=1)) / null.std(axis=1)
        ranks_cache[split] = dict(zip(feature_cols, zip(obs.round(4), z.round(2))))
    for c in feature_cols:
        per = {s: {"ic": float(ranks_cache[s][c][0]), "z": float(ranks_cache[s][c][1])}
               for s in ranks_cache}
        signs = {np.sign(per[s]["ic"]) for s in per}
        per["pass"] = bool(len(per) == 3 and abs(per["TRAIN"]["z"]) >= 2 and len(signs) == 1)
        results[c] = per
    return results


def gbm_probe(feat: pd.DataFrame, feature_cols: list, masks: dict) -> dict:
    data = feat[feature_cols + ["y_ret"]].copy()
    data["y_up"] = (data["y_ret"] > 0).astype(int)
    data.loc[data["y_ret"] == 0, "y_up"] = np.nan
    valid = data[feature_cols + ["y_up"]].notna().all(axis=1) & data["y_ret"].notna()
    tr = masks["TRAIN"] & valid.to_numpy()
    # purge: drop last bar of TRAIN (its label overlaps the boundary)
    tr_idx = np.flatnonzero(tr)
    tr[tr_idx[-1:]] = False
    X_tr = data.loc[tr, feature_cols]
    y_tr = data.loc[tr, "y_up"].astype(int)
    out = {"n_train": int(tr.sum())}
    probs = {}
    aucs = {s: [] for s in ("TRAIN", "VAL", "OOS")}
    for seed in SEEDS:
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=300, learning_rate=0.05,
            num_leaves=31, min_child_samples=50, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, random_state=seed, verbose=-1)
        model.fit(X_tr, y_tr)
        for split in aucs:
            m = masks[split] & valid.to_numpy()
            if m.sum() < 100:
                continue
            p = model.predict_proba(data.loc[m, feature_cols])[:, 1]
            aucs[split].append(roc_auc_score(data.loc[m, "y_up"].astype(int), p))
            probs.setdefault(split, []).append(p)
    rng_b = np.random.default_rng(7)
    for split in list(aucs):
        if not aucs[split]:
            continue
        m = masks[split] & valid.to_numpy()
        y = data.loc[m, "y_up"].astype(int).to_numpy()
        p_ens = np.mean(probs[split], axis=0)
        auc_ens = roc_auc_score(y, p_ens)
        boots = []
        n = len(y)
        for _ in range(N_BOOT):
            idx = rng_b.integers(0, n, n)
            if y[idx].min() == y[idx].max():
                continue
            boots.append(roc_auc_score(y[idx], p_ens[idx]))
        se = np.std(boots)
        out[split] = {
            "auc_per_seed": [round(a, 4) for a in aucs[split]],
            "auc_median": round(float(np.median(aucs[split])), 4),
            "auc_ensemble": round(float(auc_ens), 4),
            "auc_boot_z": round(float((auc_ens - 0.5) / se), 2) if se > 0 else None,
            "n": int(m.sum()),
        }
    out["_probs"] = probs
    out["_valid"] = valid.to_numpy()
    return out


def backtest(feat: pd.DataFrame, masks: dict, probe: dict) -> dict:
    out = {}
    valid = probe["_valid"]
    chosen_tau = None
    for split in ("VAL", "OOS"):
        m = masks[split] & valid
        if split not in probe["_probs"]:
            continue
        p = np.mean(probe["_probs"][split], axis=0)
        y = feat.loc[m, "y_ret"].to_numpy()
        res = {}
        for tau in TAUS:
            pos = np.where(p > 0.5 + tau, 1, np.where(p < 0.5 - tau, -1, 0))
            gross = pos * y
            turns = np.abs(np.diff(pos, prepend=0))
            net = gross - turns * COST_ROUNDTRIP / 2  # half cost per unit position change
            res[str(tau)] = {"net_bp": round(float(net.sum()) * 1e4, 1),
                             "gross_bp": round(float(np.nansum(gross)) * 1e4, 1),
                             "n_flips": int((turns > 0).sum())}
        al = float(np.nansum(y)) * 1e4
        as_ = -al
        bench = max(al, as_) - COST_ROUNDTRIP * 1e4
        res["max_always_net_bp"] = round(bench, 1)
        res["abs_ret_median_bp"] = round(float(np.nanmedian(np.abs(y))) * 1e4, 2)
        if split == "VAL":
            chosen_tau = max(TAUS, key=lambda t: res[str(t)]["net_bp"])
            res["chosen_tau"] = chosen_tau
        else:
            res["chosen_tau_from_val"] = chosen_tau
            res["beats_benchmark"] = bool(res[str(chosen_tau)]["net_bp"] > bench)
        out[split] = res
    return out


def main():
    df5 = load_5m()
    print(f"5m rows={len(df5)} {df5.index[0]} .. {df5.index[-1]}", flush=True)
    rng = np.random.default_rng(20260817)
    report = {"seeds": SEEDS, "splits": SPLITS, "embargo_bars": EMBARGO,
              "cost_roundtrip": COST_ROUNDTRIP,
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "timescales": {}}
    for name, rule in TIMESCALES.items():
        df = resample(df5, rule)
        feat = build_features(df)
        feature_cols = [c for c in feat.columns if c != "y_ret"]
        masks = split_masks(feat.index)
        ts = {"n_bars": {s: int(m.sum()) for s, m in masks.items()}}
        ts["model_free"] = {s: model_free_stats(feat.loc[m, "y_ret"].to_numpy())
                            for s, m in masks.items()}
        ts["ic"] = ic_screen(feat, feature_cols, masks, rng)
        probe = gbm_probe(feat, feature_cols, masks)
        ts["gbm"] = {k: v for k, v in probe.items() if not k.startswith("_")}
        ts["backtest"] = backtest(feat, masks, probe)
        report["timescales"][name] = ts
        print(f"[{name}] bars={ts['n_bars']} "
              f"AUC med VAL={ts['gbm'].get('VAL', {}).get('auc_median')} "
              f"OOS={ts['gbm'].get('OOS', {}).get('auc_median')}", flush=True)
    OUT.write_text(json.dumps(report, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
