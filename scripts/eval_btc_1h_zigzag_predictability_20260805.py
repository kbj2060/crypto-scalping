"""BTC 1h zigzag label -- cheap predictability check + 1-month OOS visual (per user request:
check continuity/turn-following before any backtest). Same causal feature set and split as the
other 1h candidates this session (native-1h microstructure + DVOL, no causalfix_final reuse).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_1h_full_2024_2026.csv"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_1h_zigzag_labels_20260805.parquet"
DVOL_PATH = ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv"

DROP_RAW = {"timestamp", "open", "high", "low", "close", "close_btc", "volume_btc", "quote_volume_btc"}
VAL_START, OOS_START, OOS_END = "2025-09-01", "2026-01-01", "2026-04-01"
ACTION_MAP = {"CASH": 0, "LONG": 1, "SHORT": 2}


def build_dvol_features() -> pd.DataFrame:
    df = pd.read_csv(DVOL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["available_at"] = df["timestamp"] + pd.Timedelta(hours=1)
    df = df[["available_at", "close"]].rename(columns={"available_at": "timestamp", "close": "dvol_btc"}).sort_values("timestamp")
    df["dvol_btc_roc_24h"] = df["dvol_btc"].pct_change(24)
    df["dvol_btc_roc_168h"] = df["dvol_btc"].pct_change(168)
    df["dvol_btc_pctrank_720h"] = df["dvol_btc"].rolling(720, min_periods=180).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
    return df


def load_frame():
    panel = pd.read_csv(PANEL_PATH)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    labels = pd.read_parquet(LABEL_PATH, columns=["timestamp", "zigzag_action", "zigzag_action_name"])
    dvol = build_dvol_features()

    df = panel.merge(labels, on="timestamp", how="inner")
    df = pd.merge_asof(df.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    df = df.dropna(subset=["zigzag_action"]).reset_index(drop=True)

    feature_cols = [c for c in panel.columns if c not in DROP_RAW] + [
        "dvol_btc", "dvol_btc_roc_24h", "dvol_btc_roc_168h", "dvol_btc_pctrank_720h",
    ]
    return df, feature_cols


def main() -> int:
    df, feature_cols = load_frame()
    X = df[feature_cols]
    y = df["zigzag_action"].astype(int)

    train_mask = df["timestamp"] < VAL_START
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)
    oos_mask = (df["timestamp"] >= OOS_START) & (df["timestamp"] < OOS_END)
    print(f"train={train_mask.sum()} val={val_mask.sum()} oos={oos_mask.sum()}")

    clf = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, min_child_samples=50, verbosity=-1)
    clf.fit(X[train_mask], y[train_mask])
    df["pred"] = clf.predict(X)

    maj_baseline = y[train_mask].value_counts(normalize=True).max()
    for name, mask in [("VAL", val_mask), ("OOS", oos_mask)]:
        yt, yp = y[mask], df.loc[mask, "pred"]
        acc = (yt == yp).mean()
        f1m = f1_score(yt, yp, average="macro")
        print(f"\n=== {name} (n={mask.sum()}) ===")
        print(f"majority baseline={maj_baseline:.4f}  acc={acc:.4f}  macro-F1={f1m:.4f}")
        print(classification_report(yt, yp, target_names=["CASH", "LONG", "SHORT"], digits=3))
        print(confusion_matrix(yt, yp, labels=[0, 1, 2]))

    imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\ntop 15 features:")
    print(imp.head(15).to_string())

    df.to_parquet(ROOT / "tmp/btc_1h_volregime_20260805/zigzag_pred_full.parquet", index=False)
    print("\nwrote predictions parquet for plotting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
