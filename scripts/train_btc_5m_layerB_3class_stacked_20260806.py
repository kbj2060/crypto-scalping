"""BTC 5m Layer B v5: LightGBM cascade/stacking -- Layer A's predicted probability is added as an
input FEATURE to Layer B's LightGBM (approximates a "shared backbone" for tree ensembles, which
can't share internal representations across tasks the way a neural net can; this is the standard
GBDT stacking equivalent). Layer A probability on the TRAIN split is 5-fold OUT-OF-FOLD to avoid
leaking Layer A's in-sample overfit into Layer B's training; VAL/OOS use Layer A's already-fit
full-train model (tmp/btc_1h_volregime_20260805/btc5m_layerA_pred.parquet).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
ZIGZAG_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet"
PIVOT_PATH = ROOT / "data/splits/year_oos/btc_5m_pivot_transition_labels_20260806.parquet"
LAYERA_FULL_PRED_PATH = ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerA_pred.parquet"
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
    panel = pd.read_parquet(PANEL_PATH)
    labels = pd.read_parquet(ZIGZAG_PATH, columns=["timestamp", "zigzag_action"])
    dvol = build_dvol_features()

    df = panel.merge(labels, on="timestamp", how="inner")
    df = pd.merge_asof(df.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    df = df.dropna(subset=["zigzag_action"]).reset_index(drop=True)

    base_feature_cols = [c for c in panel.columns if c not in DROP_RAW] + [
        "dvol_btc", "dvol_btc_roc_24h", "dvol_btc_roc_168h", "dvol_btc_pctrank_720h",
    ]

    train_mask = (df["timestamp"] < VAL_START).to_numpy()
    val_mask = ((df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)).to_numpy()
    oos_mask = ((df["timestamp"] >= OOS_START) & (df["timestamp"] < OOS_END)).to_numpy()

    # --- step 1: Layer A out-of-fold probability on TRAIN, direct predict on VAL/OOS ---
    piv = pd.read_parquet(PIVOT_PATH, columns=["timestamp", "transition_soon"])
    dfA = df.merge(piv, on="timestamp", how="left")  # left join: keep Layer B's row set
    yA_train = dfA.loc[train_mask, "transition_soon"].to_numpy()
    XA_train = dfA.loc[train_mask, base_feature_cols].to_numpy()
    train_idx = np.flatnonzero(train_mask)

    oof_probA = np.full(train_mask.sum(), np.nan)
    kf = KFold(n_splits=5, shuffle=True, random_state=20260806)
    valid_rows = np.isfinite(yA_train)
    for fold_i, (fit_i, hold_i) in enumerate(kf.split(np.flatnonzero(valid_rows))):
        rows = np.flatnonzero(valid_rows)
        fit_rows, hold_rows = rows[fit_i], rows[hold_i]
        clf = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, min_child_samples=100,
                              class_weight="balanced", verbosity=-1)
        clf.fit(XA_train[fit_rows], yA_train[fit_rows])
        oof_probA[hold_rows] = clf.predict_proba(XA_train[hold_rows])[:, 1]
        print(f"Layer A OOF fold {fold_i} done")
    oof_probA = np.nan_to_num(oof_probA, nan=np.nanmean(oof_probA))

    layerA_full = pd.read_parquet(LAYERA_FULL_PRED_PATH)
    df = df.merge(layerA_full, on="timestamp", how="left")
    df["layerA_prob_stack"] = np.nan
    df.loc[train_mask, "layerA_prob_stack"] = oof_probA
    df.loc[~train_mask, "layerA_prob_stack"] = df.loc[~train_mask, "probA"]

    # --- step 2: Layer B with Layer A's (OOF-safe) probability as an extra feature ---
    feature_cols = base_feature_cols + ["layerA_prob_stack"]
    X = df[feature_cols]
    y = df["zigzag_action"].astype(int)

    print(f"train={train_mask.sum()} val={val_mask.sum()} oos={oos_mask.sum()}")
    clf = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, min_child_samples=100, verbosity=-1)
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

    imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    rank = list(imp.index).index("layerA_prob_stack")
    print(f"\nlayerA_prob_stack feature importance rank: {rank} / {len(imp)} (importance={imp['layerA_prob_stack']})")
    print(imp.head(10).to_string())

    df[["timestamp", "pred"]].to_parquet(ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerB_stacked_pred.parquet", index=False)
    print("\nwrote predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
