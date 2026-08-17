"""Extension of eth_direction_timescale_resample_screen_20260817: same multi-timescale
existence screen, but over the FULL unified-dataset feature set (~159 engineered/model-output
columns from data/rl_training_{2024,2025,2026}_unified.csv) instead of the original 15
hand-built OHLCV-derived features.

Columns present only from 2026-01-01 (no TRAIN coverage) are dropped. Columns whose TRAIN
Spearman correlation with close price exceeds 0.5 in absolute value are dropped as
price-trend-contaminated (per repo policy: feedback_raw_feature_price_trend_contamination).

Non-OHLCV feature columns are resampled to coarser bars via last-observed-value at bar close
(the value a live system would actually have at decision time) rather than re-aggregating,
since most of these are already engineered indicators computed at 5m cadence.

Pre-registered design (extended from):
docs/experiments/eth_candidate_direction_timescale_resample_screen_20260817.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/experiments/eth_direction_timescale_resample_screen_fullfeat_20260817_results.json"

TIMESCALES = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "2h": "2h", "4h": "4h"}
SPLITS = {
    "TRAIN": ("2024-01-01", "2025-08-31 23:59:59"),
    "VAL": ("2025-09-01", "2025-12-31 23:59:59"),
    "OOS": ("2026-01-01", "2026-02-28 23:59:59"),
}
SEEDS = [int(s) for s in np.random.default_rng(20260817).integers(1, 999_999, 5)]
EMBARGO = 24
N_PERM = 200
N_BOOT = 500
COST_ROUNDTRIP = 0.0010
TAUS = [0.00, 0.03, 0.05]

OHLCV_COLS = {"open", "high", "low", "close", "volume", "trades",
              "taker_buy_base", "taker_buy_quote", "quote_volume"}
CONTAM_THRESHOLD = 0.5


def load_all() -> pd.DataFrame:
    frames = []
    for year in (2024, 2025, 2026):
        frames.append(pd.read_csv(ROOT / f"data/rl_training_{year}_unified.csv"))
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    return df


def select_feature_cols(df: pd.DataFrame) -> tuple[list, dict]:
    train_mask = (df.index >= SPLITS["TRAIN"][0]) & (df.index <= SPLITS["TRAIN"][1])
    val_mask = (df.index >= SPLITS["VAL"][0]) & (df.index <= SPLITS["VAL"][1])
    oos_mask = (df.index >= SPLITS["OOS"][0]) & (df.index <= SPLITS["OOS"][1])
    numeric = [c for c in df.columns if c not in OHLCV_COLS and pd.api.types.is_numeric_dtype(df[c])
               and "target" not in c.lower()]
    dropped = {"no_train_coverage": [], "no_val_oos_coverage": [], "price_contaminated": []}
    kept = []
    close_train = df.loc[train_mask, "close"]
    for c in numeric:
        s_train = df.loc[train_mask, c]
        if s_train.notna().mean() < 0.95:
            dropped["no_train_coverage"].append(c)
            continue
        valid = s_train.notna() & close_train.notna()
        if valid.sum() < 1000:
            dropped["no_train_coverage"].append(c)
            continue
        val_cov = df.loc[val_mask, c].notna().mean()
        oos_cov = df.loc[oos_mask, c].notna().mean()
        if val_cov < 0.95 or oos_cov < 0.95:
            dropped["no_val_oos_coverage"].append([c, round(float(val_cov), 3), round(float(oos_cov), 3)])
            continue
        rho = pd.Series(rankdata(s_train[valid])).corr(pd.Series(rankdata(close_train[valid])))
        if abs(rho) > CONTAM_THRESHOLD:
            dropped["price_contaminated"].append([c, round(float(rho), 3)])
            continue
        kept.append(c)
    return kept, dropped


def resample(df: pd.DataFrame, rule: str, feature_cols: list) -> pd.DataFrame:
    if rule == "5min":
        out = df[["open", "high", "low", "close"] + feature_cols].copy()
        return out
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    for c in feature_cols:
        agg[c] = "last"
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["close"])


def split_masks(index: pd.DatetimeIndex) -> dict:
    masks = {}
    prev_end = None
    for name, (start, end) in SPLITS.items():
        m = (index >= start) & (index <= end)
        if prev_end is not None:
            idx = np.flatnonzero(m)
            m = m.copy()
            m[idx[:EMBARGO]] = False
        masks[name] = m
        prev_end = end
    return masks


def build_target(df: pd.DataFrame) -> pd.Series:
    return np.log(df["close"]).diff().shift(-1)


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
        Fstd = F.std(axis=0)
        Fstd[Fstd == 0] = np.nan
        F = (F - F.mean(axis=0)) / Fstd
        n = len(y)
        obs = np.nansum(F.T * y, axis=1) / n
        shifts = rng.integers(50, n - 50, size=N_PERM)
        Y = np.column_stack([np.roll(y, int(s)) for s in shifts])
        null = F.T @ Y / n
        z = (obs - np.nanmean(null, axis=1)) / np.nanstd(null, axis=1)
        ranks_cache[split] = dict(zip(feature_cols, zip(np.nan_to_num(obs).round(4),
                                                          np.nan_to_num(z).round(2))))
    for c in feature_cols:
        if not all(c in ranks_cache.get(s, {}) for s in ("TRAIN", "VAL", "OOS")):
            continue
        per = {s: {"ic": float(ranks_cache[s][c][0]), "z": float(ranks_cache[s][c][1])}
               for s in ("TRAIN", "VAL", "OOS")}
        signs = {np.sign(per[s]["ic"]) for s in per}
        per["pass"] = bool(abs(per["TRAIN"]["z"]) >= 2 and len(signs) == 1)
        results[c] = per
    return results


def gbm_probe(feat: pd.DataFrame, feature_cols: list, masks: dict) -> dict:
    data = feat[feature_cols + ["y_ret"]].copy()
    data["y_up"] = (data["y_ret"] > 0).astype(int)
    data.loc[data["y_ret"] == 0, "y_up"] = np.nan
    valid = data[feature_cols + ["y_up"]].notna().all(axis=1) & data["y_ret"].notna()
    tr = masks["TRAIN"] & valid.to_numpy()
    tr_idx = np.flatnonzero(tr)
    tr[tr_idx[-1:]] = False
    X_tr = data.loc[tr, feature_cols]
    y_tr = data.loc[tr, "y_up"].astype(int)
    out = {"n_train": int(tr.sum()), "n_features": len(feature_cols)}
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


def top_feature_importance(feat: pd.DataFrame, feature_cols: list, masks: dict, seed: int) -> list:
    data = feat[feature_cols + ["y_ret"]].copy()
    data["y_up"] = (data["y_ret"] > 0).astype(int)
    data.loc[data["y_ret"] == 0, "y_up"] = np.nan
    valid = data[feature_cols + ["y_up"]].notna().all(axis=1) & data["y_ret"].notna()
    tr = masks["TRAIN"] & valid.to_numpy()
    tr_idx = np.flatnonzero(tr)
    tr[tr_idx[-1:]] = False
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=0.05,
        num_leaves=31, min_child_samples=50, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1, random_state=seed, verbose=-1)
    model.fit(data.loc[tr, feature_cols], data.loc[tr, "y_up"].astype(int))
    imp = sorted(zip(feature_cols, model.feature_importances_.tolist()),
                 key=lambda x: -x[1])[:15]
    return [[c, int(v)] for c, v in imp]


def backtest(feat: pd.DataFrame, masks: dict, probe: dict) -> dict:
    out = {}
    valid = probe["_valid"]
    chosen_tau = None
    for split in ("VAL", "OOS"):
        if split not in probe["_probs"]:
            continue
        m = masks[split] & valid
        p = np.mean(probe["_probs"][split], axis=0)
        y = feat.loc[m, "y_ret"].to_numpy()
        res = {}
        for tau in TAUS:
            pos = np.where(p > 0.5 + tau, 1, np.where(p < 0.5 - tau, -1, 0))
            gross = pos * y
            turns = np.abs(np.diff(pos, prepend=0))
            net = gross - turns * COST_ROUNDTRIP / 2
            res[str(tau)] = {"net_bp": round(float(np.nansum(net)) * 1e4, 1),
                             "gross_bp": round(float(np.nansum(gross)) * 1e4, 1),
                             "n_flips": int((turns > 0).sum())}
        al = float(np.nansum(y)) * 1e4
        bench = max(al, -al) - COST_ROUNDTRIP * 1e4
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
    df = load_all()
    print(f"rows={len(df)} cols={df.shape[1]}", flush=True)
    feature_cols, dropped = select_feature_cols(df)
    print(f"kept {len(feature_cols)} features; dropped no_train_coverage="
          f"{len(dropped['no_train_coverage'])} price_contaminated={len(dropped['price_contaminated'])}",
          flush=True)
    print("price_contaminated:", dropped["price_contaminated"], flush=True)
    rng = np.random.default_rng(20260817)
    report = {"seeds": SEEDS, "splits": SPLITS, "embargo_bars": EMBARGO,
              "cost_roundtrip": COST_ROUNDTRIP, "n_candidate_cols": len(feature_cols),
              "dropped": dropped,
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "timescales": {}}
    for name, rule in TIMESCALES.items():
        rdf = resample(df, rule, feature_cols)
        feat = rdf[feature_cols].copy()
        feat["y_ret"] = build_target(rdf)
        masks = split_masks(feat.index)
        ts = {"n_bars": {s: int(m.sum()) for s, m in masks.items()}}
        ic = ic_screen(feat, feature_cols, masks, rng)
        passing = [c for c in ic if ic[c]["pass"]]
        ts["ic_n_pass"] = len(passing)
        ts["ic_pass_features"] = passing[:40]
        probe = gbm_probe(feat, feature_cols, masks)
        ts["gbm"] = {k: v for k, v in probe.items() if not k.startswith("_")}
        ts["backtest"] = backtest(feat, masks, probe)
        ts["top_importance_seed0"] = top_feature_importance(feat, feature_cols, masks, SEEDS[0])
        report["timescales"][name] = ts
        print(f"[{name}] bars={ts['n_bars']} ic_pass={ts['ic_n_pass']}/{len(feature_cols)} "
              f"AUC VAL={ts['gbm'].get('VAL', {}).get('auc_median')} "
              f"OOS={ts['gbm'].get('OOS', {}).get('auc_median')} "
              f"OOSnet={ts['backtest'].get('OOS', {}).get(str(ts['backtest'].get('OOS', {}).get('chosen_tau_from_val')), {}).get('net_bp')} "
              f"bench={ts['backtest'].get('OOS', {}).get('max_always_net_bp')}", flush=True)
    OUT.write_text(json.dumps(report, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
