"""BTC 5m Layer B v6: quality-WEIGHTED training (backlog item #1) -- use the oracle barrier-
simulated net_ret_sim as a per-sample LightGBM training weight, not a separate filter model and
not a hard active-bar restriction. CASH bars (no net_ret_sim, quality is only defined where the
raw zigzag signal is active) get the baseline weight of 1.0, unaffected. Active bars get
upweighted if the trade would have been profitable, downweighted if not -- the model is pushed to
fit the LONG/SHORT boundary harder on bars where getting it right actually mattered economically,
instead of treating every active bar as equally informative.

Different failure mode from the earlier "quality meta-label" attempt: that trained a SEPARATE
classifier conditioned on the (noisy) PREDICTED action and collapsed to AUC 0.50. This reuses the
ORACLE quality signal only as a loss weight during Layer B's own training -- no second model, no
serve-time dependency on a quality classifier that has to generalize on its own.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
ZIGZAG_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet"
QUALITY_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_quality_oracle_20260806.parquet"
DVOL_PATH = ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv"

DROP_RAW = {"timestamp", "open", "high", "low", "close", "close_btc", "volume_btc", "quote_volume_btc"}
VAL_START, OOS_START, OOS_END = "2025-09-01", "2026-01-01", "2026-04-01"
WEIGHT_SCALE = 40.0  # net_ret_sim is O(1e-2); scale so weight swings are meaningful but bounded
WEIGHT_MIN, WEIGHT_MAX = 0.2, 3.0


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
    panel = pd.read_parquet(PANEL_PATH)
    labels = pd.read_parquet(ZIGZAG_PATH, columns=["timestamp", "zigzag_action"])
    qual = pd.read_parquet(QUALITY_PATH, columns=["timestamp", "net_ret_sim"])
    dvol = build_dvol_features()

    df = panel.merge(labels, on="timestamp", how="inner").merge(qual, on="timestamp", how="inner")
    df = pd.merge_asof(df.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    df = df.dropna(subset=["zigzag_action"]).reset_index(drop=True)

    feature_cols = [c for c in panel.columns if c not in DROP_RAW] + [
        "dvol_btc", "dvol_btc_roc_24h", "dvol_btc_roc_168h", "dvol_btc_pctrank_720h",
    ]
    X = df[feature_cols]
    y = df["zigzag_action"].astype(int)

    sample_weight = 1.0 + df["net_ret_sim"].fillna(0.0).to_numpy() * WEIGHT_SCALE
    sample_weight = np.clip(sample_weight, WEIGHT_MIN, WEIGHT_MAX)
    print(f"weight stats: mean={sample_weight.mean():.3f} min={sample_weight.min():.3f} max={sample_weight.max():.3f}")
    print(f"active-bar weight mean: {sample_weight[df['net_ret_sim'].notna()].mean():.3f}  "
          f"CASH-bar weight (should be 1.0): {sample_weight[df['net_ret_sim'].isna()].mean():.3f}")

    train_mask = (df["timestamp"] < VAL_START).to_numpy()
    val_mask = ((df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)).to_numpy()
    oos_mask = ((df["timestamp"] >= OOS_START) & (df["timestamp"] < OOS_END)).to_numpy()
    print(f"train={train_mask.sum()} val={val_mask.sum()} oos={oos_mask.sum()}")

    clf = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, min_child_samples=100, verbosity=-1)
    clf.fit(X[train_mask], y[train_mask], sample_weight=sample_weight[train_mask])
    df["pred"] = clf.predict(X)

    maj_baseline = y[train_mask].value_counts(normalize=True).max()
    for name, mask in [("VAL", val_mask), ("OOS", oos_mask)]:
        yt, yp = y[mask], df.loc[mask, "pred"]
        acc = (yt == yp).mean()
        f1m = f1_score(yt, yp, average="macro")
        print(f"\n=== {name} (n={mask.sum()}) ===")
        print(f"majority baseline={maj_baseline:.4f}  acc={acc:.4f}  macro-F1={f1m:.4f}")
        print(classification_report(yt, yp, target_names=["CASH", "LONG", "SHORT"], digits=3))

    df[["timestamp", "pred"]].to_parquet(ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerB_qualityweighted_pred.parquet", index=False)
    print("\nwrote predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
