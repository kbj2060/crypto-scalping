"""BTC 1h new-architecture, Step 2 (cheap falsification): can DVOL + native-1h microstructure
features predict the vol-regime label (build_btc_1h_volregime_labels_20260805.py) at all?

Follows the project's established "cheap falsification first" convention (same pattern as the
2026-08-04 DVOL/on-chain axis checks): fit one classifier on a proper causal train/val/oos split,
look at whether it beats a majority-class baseline BEFORE building anything heavier.

Split (Fresh-Forward convention): train < 2025-09-01, VAL 2025-09-01..2025-12-31,
OOS 2026-01-01..2026-03-31.

NOTE: close_btc/volume_btc/quote_volume_btc in btc_features_1h_full_2024_2026.csv are leftover
mislabeled columns (values match ETH's price range, not BTC's) -- excluded from features here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_1h_full_2024_2026.csv"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_1h_volregime_labels_20260805.parquet"
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

    labels = pd.read_parquet(LABEL_PATH, columns=["timestamp", "label_3class", "trailing_vol_24h"])
    dvol = build_dvol_features()

    df = panel.merge(labels, on="timestamp", how="inner")
    df = pd.merge_asof(df.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    df = df.dropna(subset=["label_3class"]).reset_index(drop=True)

    # ablation: drop trailing_vol_24h and other near-tautological trailing-vol proxies to test
    # whether the signal survives without the "vol mean-reverts" shortcut
    VOL_PROXY_DROP = {
        "trailing_vol_24h", "volatility_z", "garman_klass_vol", "realized_vol_ratio",
        "rogers_satchell_vol", "parkinson_vol", "bb_width", "bb_width_z", "bb_width_pct_rank_288",
        "atr_pct_rank_288", "compression_score", "garch_vol_z",
    }
    feature_cols = [c for c in panel.columns if c not in DROP_RAW and c not in VOL_PROXY_DROP] + [
        "dvol_btc", "dvol_btc_roc_24h", "dvol_btc_roc_168h", "dvol_btc_pctrank_720h",
    ]
    X = df[feature_cols]
    y = df["label_3class"].astype(int)

    train_mask = df["timestamp"] < VAL_START
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)
    oos_mask = (df["timestamp"] >= OOS_START) & (df["timestamp"] < OOS_END)

    print(f"train={train_mask.sum()} val={val_mask.sum()} oos={oos_mask.sum()}")

    clf = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, min_child_samples=50, verbosity=-1)
    clf.fit(X[train_mask], y[train_mask])

    for name, mask in [("VAL", val_mask), ("OOS", oos_mask)]:
        yt, yp = y[mask], clf.predict(X[mask])
        maj_baseline = y[train_mask].value_counts(normalize=True).max()
        acc = (yt == yp).mean()
        f1m = f1_score(yt, yp, average="macro")
        print(f"\n=== {name} (n={mask.sum()}) ===")
        print(f"majority-class baseline acc={maj_baseline:.4f}  |  model acc={acc:.4f}  macro-F1={f1m:.4f}")
        print(classification_report(yt, yp, digits=3))
        print("confusion matrix (rows=true[-1,0,1], cols=pred[-1,0,1]):")
        print(confusion_matrix(yt, yp, labels=[-1, 0, 1]))

    imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\ntop 20 feature importances:")
    print(imp.head(20).to_string())
    dvol_rank = [i for i, c in enumerate(imp.index) if c.startswith("dvol")]
    print(f"\nDVOL feature ranks (0=top): {dvol_rank} out of {len(imp)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
