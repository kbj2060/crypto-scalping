"""BTC 1h Layer A cheap check: can the model's predicted probability of an imminent swing
transition actually elevate ahead of real pivots? Same causal 1h+DVOL feature set/split as the
other candidates this session.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_1h_full_2024_2026.csv"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_1h_pivot_transition_labels_20260805.parquet"
DVOL_PATH = ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv"

DROP_RAW = {"timestamp", "open", "high", "low", "close", "close_btc", "volume_btc", "quote_volume_btc"}
VAL_START, OOS_START, OOS_END = "2025-09-01", "2026-01-01", "2026-04-01"


def build_dvol_features() -> pd.DataFrame:
    df = pd.read_csv(DVOL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["available_at"] = df["timestamp"] + pd.Timedelta(hours=1)
    df = df[["available_at", "close"]].rename(columns={"available_at": "timestamp", "close": "dvol_btc"}).sort_values("timestamp")
    df["dvol_btc_roc_24h"] = df["dvol_btc"].pct_change(24)
    df["dvol_btc_roc_168h"] = df["dvol_btc"].pct_change(168)
    df["dvol_btc_pctrank_720h"] = df["dvol_btc"].rolling(720, min_periods=180).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
    return df


def main() -> int:
    panel = pd.read_csv(PANEL_PATH)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    labels = pd.read_parquet(LABEL_PATH, columns=["timestamp", "transition_soon", "is_pivot", "pivot_type"])
    dvol = build_dvol_features()

    df = panel.merge(labels, on="timestamp", how="inner")
    df = pd.merge_asof(df.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    df = df.dropna(subset=["transition_soon"]).reset_index(drop=True)

    feature_cols = [c for c in panel.columns if c not in DROP_RAW] + [
        "dvol_btc", "dvol_btc_roc_24h", "dvol_btc_roc_168h", "dvol_btc_pctrank_720h",
    ]
    X = df[feature_cols]
    y = df["transition_soon"].astype(int)

    train_mask = df["timestamp"] < VAL_START
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)
    oos_mask = (df["timestamp"] >= OOS_START) & (df["timestamp"] < OOS_END)
    print(f"train={train_mask.sum()} val={val_mask.sum()} oos={oos_mask.sum()}  base_rate(train)={y[train_mask].mean():.4f}")

    clf = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, min_child_samples=50,
                          class_weight="balanced", verbosity=-1)
    clf.fit(X[train_mask], y[train_mask])
    df["prob"] = clf.predict_proba(X)[:, 1]

    for name, mask in [("VAL", val_mask), ("OOS", oos_mask)]:
        yt, p = y[mask], df.loc[mask, "prob"]
        auc = roc_auc_score(yt, p)
        ap = average_precision_score(yt, p)
        base_rate = yt.mean()
        print(f"\n=== {name} (n={mask.sum()}, base_rate={base_rate:.4f}) ===")
        print(f"AUC={auc:.4f}  AP={ap:.4f} (baseline AP=base_rate={base_rate:.4f})")
        # top-decile precision: if we only "trust" the top 10% highest predicted probability bars,
        # what fraction actually are transition_soon=1?
        thresh = p.quantile(0.90)
        top_decile = yt[p >= thresh]
        print(f"top-decile (p>={thresh:.3f}) precision: {top_decile.mean():.4f}  n={len(top_decile)}")

    imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\ntop 15 features:")
    print(imp.head(15).to_string())

    df.to_parquet(ROOT / "tmp/btc_1h_volregime_20260805/pivot_transition_pred_full.parquet", index=False)
    print("\nwrote predictions parquet for plotting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
