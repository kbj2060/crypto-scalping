"""BTC 5m retry: cheap predictability check for both Layer A (transition detector) and Layer B
(binary LONG/SHORT direction on active-wave bars only), same feature-set/split convention as the
1h attempts, using causalfix_final (5m) + DVOL causal features.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
ZIGZAG_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet"
PIVOT_PATH = ROOT / "data/splits/year_oos/btc_5m_pivot_transition_labels_20260806.parquet"
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
    dvol = build_dvol_features()
    panel = pd.merge_asof(panel.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    feature_cols = [c for c in panel.columns if c not in DROP_RAW]

    # ===== Layer A: transition detector =====
    print("=" * 20, "LAYER A: transition detector", "=" * 20)
    piv = pd.read_parquet(PIVOT_PATH, columns=["timestamp", "transition_soon"])
    dfA = panel.merge(piv, on="timestamp", how="inner").dropna(subset=["transition_soon"]).reset_index(drop=True)
    Xa = dfA[feature_cols]
    ya = dfA["transition_soon"].astype(int)
    tr = dfA["timestamp"] < VAL_START
    val = (dfA["timestamp"] >= VAL_START) & (dfA["timestamp"] < OOS_START)
    oos = (dfA["timestamp"] >= OOS_START) & (dfA["timestamp"] < OOS_END)
    print(f"train={tr.sum()} val={val.sum()} oos={oos.sum()} base_rate(train)={ya[tr].mean():.4f}")

    clfA = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, min_child_samples=100,
                           class_weight="balanced", verbosity=-1)
    clfA.fit(Xa[tr], ya[tr])
    probA = clfA.predict_proba(Xa)[:, 1]
    dfA["probA"] = probA
    for name, mask in [("VAL", val), ("OOS", oos)]:
        yt, p = ya[mask], dfA.loc[mask, "probA"]
        auc = roc_auc_score(yt, p)
        ap = average_precision_score(yt, p)
        base_rate = yt.mean()
        thresh = p.quantile(0.90)
        prec = yt[p >= thresh].mean()
        print(f"{name}: AUC={auc:.4f} AP={ap:.4f} (base={base_rate:.4f}) top-decile precision={prec:.4f}")

    dfA[["timestamp", "probA"]].to_parquet(ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerA_pred.parquet", index=False)

    # ===== Layer B: binary direction on active-wave bars, scored on ALL bars =====
    print("\n" + "=" * 20, "LAYER B: binary direction", "=" * 20)
    zz = pd.read_parquet(ZIGZAG_PATH, columns=["timestamp", "zigzag_action"])
    active = zz[zz["zigzag_action"].isin([1, 2])]
    dfB_train = panel.merge(active, on="timestamp", how="inner")
    trB = dfB_train["timestamp"] < VAL_START
    XB_train = dfB_train.loc[trB, feature_cols]
    yB_train = (dfB_train.loc[trB, "zigzag_action"] == 1).astype(int)
    print(f"Layer B train rows (active only)={len(XB_train)}  LONG-rate={yB_train.mean():.4f}")

    clfB = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, min_child_samples=100, verbosity=-1)
    clfB.fit(XB_train, yB_train)

    # score on ALL bars (not just active) -- matches real deployment
    panel["pred_prob_long"] = clfB.predict_proba(panel[feature_cols])[:, 1]
    dfB_eval = panel.merge(zz, on="timestamp", how="inner")
    dfB_eval = dfB_eval[dfB_eval["zigzag_action"].isin([1, 2])].reset_index(drop=True)  # eval accuracy only on oracle-active bars (standalone check)
    yB = (dfB_eval["zigzag_action"] == 1).astype(int)
    predB = (dfB_eval["pred_prob_long"] >= 0.5).astype(int)
    for name, start, end in [("VAL", VAL_START, OOS_START), ("OOS", OOS_START, OOS_END)]:
        mask = (dfB_eval["timestamp"] >= start) & (dfB_eval["timestamp"] < end)
        yt, yp, p = yB[mask], predB[mask], dfB_eval.loc[mask, "pred_prob_long"]
        acc = accuracy_score(yt, yp)
        auc = roc_auc_score(yt, p)
        maj = max(yt.mean(), 1 - yt.mean())
        print(f"{name} (active-only, n={mask.sum()}): baseline={maj:.4f} acc={acc:.4f} AUC={auc:.4f}")

    panel[["timestamp", "pred_prob_long"]].to_parquet(ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerB_pred_allbars.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
