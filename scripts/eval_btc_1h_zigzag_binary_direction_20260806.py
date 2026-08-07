"""BTC 1h Layer B v3: simplify to BINARY direction (LONG vs SHORT), dropping CASH entirely.

Rationale: the 3-class confusion matrix (eval_btc_1h_zigzag_predictability_20260805.py) showed
CASH was by far the worst-predicted class (precision 0.39-0.46, recall 0.23-0.36) and the
CASH<->LONG/SHORT boundary is exactly what Layer A (the transition detector, OOS AUC 0.70) is
already responsible for. Removing CASH lets Layer B focus purely on "given we're in an active
wave, which way is it" -- a simpler, hopefully more learnable decision boundary. Trained/evaluated
ONLY on bars where the oracle zigzag_action is LONG or SHORT (matches how it will actually be used
downstream: only consulted when Layer A already says a transition is happening).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_1h_full_2024_2026.csv"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_1h_zigzag_labels_20260805.parquet"
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
    labels = pd.read_parquet(LABEL_PATH, columns=["timestamp", "zigzag_action"])
    labels = labels[labels["zigzag_action"].isin([1, 2])]  # drop CASH entirely
    dvol = build_dvol_features()

    df = panel.merge(labels, on="timestamp", how="inner")
    df = pd.merge_asof(df.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    df = df.dropna(subset=["zigzag_action"]).reset_index(drop=True)

    feature_cols = [c for c in panel.columns if c not in DROP_RAW] + [
        "dvol_btc", "dvol_btc_roc_24h", "dvol_btc_roc_168h", "dvol_btc_pctrank_720h",
    ]
    X = df[feature_cols]
    y = (df["zigzag_action"] == 1).astype(int)  # 1=LONG, 0=SHORT

    train_mask = df["timestamp"] < VAL_START
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)
    oos_mask = (df["timestamp"] >= OOS_START) & (df["timestamp"] < OOS_END)
    print(f"train={train_mask.sum()} val={val_mask.sum()} oos={oos_mask.sum()}  "
          f"LONG-rate(train)={y[train_mask].mean():.4f}")

    clf = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, min_child_samples=50, verbosity=-1)
    clf.fit(X[train_mask], y[train_mask])
    df["pred_prob_long"] = clf.predict_proba(X)[:, 1]
    df["pred_binary"] = (df["pred_prob_long"] >= 0.5).astype(int)

    for name, mask in [("VAL", val_mask), ("OOS", oos_mask)]:
        yt, yp, p = y[mask], df.loc[mask, "pred_binary"], df.loc[mask, "pred_prob_long"]
        acc = accuracy_score(yt, yp)
        auc = roc_auc_score(yt, p)
        maj_baseline = max(yt.mean(), 1 - yt.mean())
        print(f"\n=== {name} (n={mask.sum()}, LONG-rate={yt.mean():.4f}) ===")
        print(f"majority baseline={maj_baseline:.4f}  acc={acc:.4f}  AUC={auc:.4f}")
        print(classification_report(yt, yp, target_names=["SHORT", "LONG"], digits=3))
        # confidence-filtered accuracy: does restricting to high |prob-0.5| bars improve accuracy?
        conf = (p - 0.5).abs()
        for q in (0.5, 0.7, 0.9):
            thresh = conf.quantile(q)
            sub = conf >= thresh
            print(f"  top-{(1-q):.0%} by confidence (n={sub.sum()}): acc={accuracy_score(yt[sub], yp[sub]):.4f}")

    imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\ntop 15 features:")
    print(imp.head(15).to_string())

    df.to_parquet(ROOT / "tmp/btc_1h_volregime_20260805/zigzag_binary_pred_full.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
