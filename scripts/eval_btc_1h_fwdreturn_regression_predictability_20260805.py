"""BTC 1h new-architecture, Step 1+cheap-check (v2, redesigned per user feedback): the vol-regime
label (direction-agnostic) forced direction onto an unrelated, edge-less breakout entry and failed
end-to-end. New label: raw forward N-hour log return, direction AND magnitude unified in one
regression target -- the model's own prediction IS the entry signal (sign=direction,
magnitude=confidence/sizing), no separate entry-logic layer needed.

NOTE: raw-return regression on causalfix_final/5m was already closed (multiple 2026-08-04 lines).
This is a DIFFERENT feature set (native-1h microstructure + DVOL + on-chain-ready, no
causalfix_final reuse) and a different horizon regime -- not a re-run of the closed line, but
carries real prior odds of failure given that history. Cheap falsification first.

Split (Fresh-Forward convention): train < 2025-09-01, VAL 2025-09-01..2025-12-31,
OOS 2026-01-01..2026-03-31.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_1h_full_2024_2026.csv"
DVOL_PATH = ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv"

DROP_RAW = {"timestamp", "open", "high", "low", "close", "close_btc", "volume_btc", "quote_volume_btc"}
VAL_START, OOS_START, OOS_END = "2025-09-01", "2026-01-01", "2026-04-01"
HORIZON_H = 12


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
    panel = panel.sort_values("timestamp").reset_index(drop=True)

    panel["fwd_ret_12h"] = np.log(panel["close"].shift(-HORIZON_H) / panel["close"])

    dvol = build_dvol_features()
    df = pd.merge_asof(panel, dvol, on="timestamp", direction="backward")
    df = df.dropna(subset=["fwd_ret_12h"]).reset_index(drop=True)

    feature_cols = [c for c in panel.columns if c not in DROP_RAW and c != "fwd_ret_12h"] + [
        "dvol_btc", "dvol_btc_roc_24h", "dvol_btc_roc_168h", "dvol_btc_pctrank_720h",
    ]
    X = df[feature_cols]
    y = df["fwd_ret_12h"]

    train_mask = df["timestamp"] < VAL_START
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)
    oos_mask = (df["timestamp"] >= OOS_START) & (df["timestamp"] < OOS_END)
    print(f"train={train_mask.sum()} val={val_mask.sum()} oos={oos_mask.sum()}")

    reg = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05, min_child_samples=50, verbosity=-1)
    reg.fit(X[train_mask], y[train_mask])

    for name, mask in [("VAL", val_mask), ("OOS", oos_mask)]:
        yt, yp = y[mask], reg.predict(X[mask])
        rho, pval = spearmanr(yt, yp)
        sign_acc = (np.sign(yt) == np.sign(yp)).mean()
        # quintile analysis: does top predicted quintile actually realize higher return than bottom?
        q = pd.qcut(pd.Series(yp, index=yt.index), 5, labels=False, duplicates="drop")
        q_means = yt.groupby(q).mean()
        print(f"\n=== {name} (n={mask.sum()}) ===")
        print(f"Spearman rho={rho:.4f} (p={pval:.2e})  sign-match-rate={sign_acc:.4f} (baseline 0.5)")
        print("realized return by predicted quintile (0=lowest pred .. 4=highest pred):")
        print((q_means * 100).round(4).to_string())

    imp = pd.Series(reg.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\ntop 20 feature importances:")
    print(imp.head(20).to_string())
    dvol_rank = [i for i, c in enumerate(imp.index) if c.startswith("dvol")]
    print(f"\nDVOL feature ranks (0=top): {dvol_rank} out of {len(imp)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
