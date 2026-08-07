"""WS-B 견고성 체크: 시간순 분할 지점을 바꿔가며 AUC가 안정적인지 확인.
(원본 T-B1의 60/20/20 분할이 우연히 좋은 test 구간을 뽑은 건 아닌지 검증)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
except Exception:
    lgb = None

MICRO_DB = "data/live/microstructure.duckdb"
KLINE_CSV = "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
OUT_DIR = Path("docs/test_designs_duckdb_live_20260719/results")


def connect_retry(path, read_only=True, retries=8, backoff=2.0):
    last_exc = None
    for attempt in range(retries):
        try:
            return duckdb.connect(path, read_only=read_only)
        except duckdb.IOException as exc:
            last_exc = exc
            time.sleep(backoff * (attempt + 1))
    raise last_exc


def build_dataset():
    con = connect_retry(MICRO_DB)
    snaps = con.execute(
        """select recorded_at_kst, best_bid, best_ask, mid, spread_bps, microprice_edge_bps,
        imbalance_1, imbalance_5, imbalance_10, imbalance_20, bid_qty_1, ask_qty_1
        from orderbook_decision_snapshots order by recorded_at_kst"""
    ).df()
    micro = con.execute(
        "select ts, obi, taker_buy_ratio, shadow_queue_collapse, shadow_absorption_score, "
        "shadow_toxicity_score from microstructure_1m order by ts"
    ).df()
    con.close()
    snaps["recorded_at_kst"] = pd.to_datetime(snaps["recorded_at_kst"], utc=True)
    micro["ts"] = pd.to_datetime(micro["ts"], utc=True)

    micro_sorted = micro.sort_values("ts").reset_index(drop=True)
    idx_feat = np.searchsorted(micro_sorted["ts"].values, snaps["recorded_at_kst"].values, side="left") - 1
    valid_feat_mask = idx_feat >= 0
    idx_feat_clipped = np.clip(idx_feat, 0, len(micro_sorted) - 1)
    for col in ["obi", "taker_buy_ratio", "shadow_queue_collapse", "shadow_absorption_score", "shadow_toxicity_score"]:
        snaps[col] = micro_sorted[col].iloc[idx_feat_clipped].reset_index(drop=True)
    snaps.loc[~valid_feat_mask, ["obi", "taker_buy_ratio", "shadow_queue_collapse",
                                  "shadow_absorption_score", "shadow_toxicity_score"]] = np.nan

    kl = pd.read_csv(KLINE_CSV, usecols=["timestamp", "low", "high", "close"])
    kl["ts"] = pd.to_datetime(kl["timestamp"], utc=True)
    kl = kl.sort_values("ts").reset_index(drop=True)
    kline_ts = kl["ts"].values
    idx_label = np.searchsorted(kline_ts, snaps["recorded_at_kst"].values, side="right")
    valid_label_mask = idx_label < len(kl)
    idx_label_clipped = np.clip(idx_label, 0, len(kl) - 1)
    snaps["label_bar_low"] = kl["low"].iloc[idx_label_clipped].values
    snaps["label_bar_high"] = kl["high"].iloc[idx_label_clipped].values
    snaps.loc[~valid_label_mask, ["label_bar_low", "label_bar_high"]] = np.nan

    snaps["filled_60s_buy"] = snaps["label_bar_low"] < snaps["best_bid"]
    snaps["filled_60s_sell"] = snaps["label_bar_high"] > snaps["best_ask"]

    valid_rows = snaps["label_bar_low"].notna() & snaps["best_bid"].notna() & (snaps["mid"] > 0)
    ds = snaps.loc[valid_rows].sort_values("recorded_at_kst").reset_index(drop=True)
    return ds


def main():
    report = {"stage": "WS-B-robustness", "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    ds = build_dataset()
    n = len(ds)
    report["n_valid_rows"] = int(n)

    features = ["spread_bps", "microprice_edge_bps", "imbalance_1", "imbalance_5",
                "imbalance_10", "imbalance_20", "obi", "taker_buy_ratio",
                "shadow_queue_collapse", "shadow_absorption_score", "shadow_toxicity_score"]

    purge = 288
    # 5 different train/test split points (walk-forward style, expanding train window)
    split_fracs = [(0.5, 0.65), (0.55, 0.70), (0.6, 0.75), (0.65, 0.80), (0.70, 0.85)]
    results = {"buy": [], "sell": []}
    for train_frac, test_start_frac in split_fracs:
        train_end = int(n * train_frac)
        test_start = int(n * test_start_frac)
        test_end = min(n, test_start + int(n * 0.15))
        train = ds.iloc[: max(train_end - purge, 0)]
        test = ds.iloc[test_start:test_end]
        for side in ["buy", "sell"]:
            target = f"filled_60s_{side}"
            cols = features + [target]
            tr = train[cols].dropna()
            te = test[cols].dropna()
            if len(tr) < 200 or len(te) < 50 or lgb is None:
                results[side].append({"train_frac": train_frac, "test_frac": test_start_frac,
                                       "skipped": "insufficient_n_or_no_lgb", "n_train": len(tr), "n_test": len(te)})
                continue
            Xtr, ytr = tr[features].values, tr[target].astype(int).values
            Xte, yte = te[features].values, te[target].astype(int).values
            if len(set(ytr)) < 2 or len(set(yte)) < 2:
                results[side].append({"train_frac": train_frac, "test_frac": test_start_frac,
                                       "skipped": "single_class", "n_train": len(tr), "n_test": len(te)})
                continue
            gbm = lgb.LGBMClassifier(max_depth=4, n_estimators=200, learning_rate=0.05,
                                      min_child_samples=30, verbose=-1)
            gbm.fit(Xtr, ytr)
            p_te = gbm.predict_proba(Xte)[:, 1]
            auc = float(roc_auc_score(yte, p_te))
            results[side].append({
                "train_frac": train_frac, "test_frac": test_start_frac,
                "n_train": len(tr), "n_test": len(te),
                "test_start_date": str(test["recorded_at_kst"].min()),
                "test_end_date": str(test["recorded_at_kst"].max()),
                "auc": auc,
            })

    report["split_results"] = results
    for side in ["buy", "sell"]:
        aucs = [r["auc"] for r in results[side] if "auc" in r]
        if aucs:
            report[f"{side}_auc_mean"] = float(np.mean(aucs))
            report[f"{side}_auc_std"] = float(np.std(aucs))
            report[f"{side}_auc_min"] = float(np.min(aucs))
            report[f"{side}_auc_max"] = float(np.max(aucs))
            report[f"{side}_n_splits_above_055"] = int(sum(a >= 0.55 for a in aucs))
            report[f"{side}_n_splits_total"] = len(aucs)

    out_json = OUT_DIR / "ws_b_robustness_check_20260719.json"
    out_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print("WROTE", out_json)
    print(json.dumps({k: v for k, v in report.items() if k != "split_results"}, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
