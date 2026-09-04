#!/usr/bin/env python3
"""K sweep for the 2 Homer candidates, HORIZON/GAP now fixed from research_eth_kalman_demarker_
gridscreen_20260831.py's result (demarker_extreme: H=8/GAP=12; kalman_deviation_meanrev: H=12/
GAP=12 -- both plain touch-only, v2 exclude-middle dropped after the grid screen showed it doesn't
beat plain at either signal's optimum). Same GBM-proxy/min(VAL,OOS) methodology, sequential
calibration order (HORIZON/GAP first, then K -- docs/homer/README.md 5.5/liquidity_sweep_topdown's
own precedent, NOT a 50/50-balance-only pick -- liquidity_sweep found raising K past its naive
50/50 point kept improving AUC monotonically up to K=4.0, so this sweeps a wide range rather than
stopping at whatever K gives a balanced split).

⚠️K_GRID upper bound: this repo hit a real bug once (smt_divergence, docs/homer/README.md 5.6) where
a capped K search silently returned the boundary value instead of the true optimum. This K_GRID
goes up to 6.0 and the result explicitly checks whether the winner sits at the top edge -- if so,
extend before treating it as final (not done automatically here, flagged for a human to notice).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from research_eth_candidate_pool_raw_lift_check_20260831 import (  # noqa: E402
    kalman_level_and_velocity,
    rolling_zscore,
)
from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker  # noqa: E402
from research_eth_kalman_demarker_gridscreen_20260831 import (  # noqa: E402
    START,
    build_fires,
    load_klines,
)
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

K_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
GBM_SEED = 20260831

SIGNAL_CONFIG = {
    "demarker_extreme": {"horizon": 8, "gap": 12},
    "kalman_deviation_meanrev": {"horizon": 12, "gap": 12},
}


def log(msg: str) -> None:
    print(f"[kalman_demarker_ksweep] {msg}", flush=True)


def sweep_signal(name: str, klines: pd.DataFrame, ind: pd.DataFrame, trigger_top: pd.Series,
                  trigger_bottom: pd.Series, extremeness: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    cfg = SIGNAL_CONFIG[name]
    horizon, gap = cfg["horizon"], cfg["gap"]
    results = []
    for K in K_GRID:
        fires = build_fires(klines, ind, trigger_top, trigger_bottom, extremeness, feature_cols, horizon, gap, K)
        fires = fires.dropna(subset=feature_cols + ["hit_plain"]).reset_index(drop=True)
        y = fires["hit_plain"].to_numpy().astype(int)
        ts = fires["timestamp"]
        train_mask = (ts < VAL_START).to_numpy()
        val_mask = ((ts >= VAL_START) & (ts < OOS_START)).to_numpy()
        oos_mask = ((ts >= OOS_START) & (ts < HOLDOUT_START)).to_numpy()

        clf = HistGradientBoostingClassifier(random_state=GBM_SEED)
        clf.fit(fires.loc[train_mask, feature_cols], y[train_mask])
        val_auc = roc_auc_score(y[val_mask], clf.predict_proba(fires.loc[val_mask, feature_cols])[:, 1])
        oos_auc = roc_auc_score(y[oos_mask], clf.predict_proba(fires.loc[oos_mask, feature_cols])[:, 1])
        row = {
            "signal": name, "horizon": horizon, "gap": gap, "K": K,
            "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()), "n_oos": int(oos_mask.sum()),
            "hit_rate": round(float(y.mean()), 4),
            "val_auc": round(float(val_auc), 4), "oos_auc": round(float(oos_auc), 4),
            "val_oos_gap": round(abs(val_auc - oos_auc), 4), "min_val_oos": round(min(val_auc, oos_auc), 4),
        }
        results.append(row)
        log(f"  {name} K={K:.2f}: n={len(fires):>5d} hit_rate={row['hit_rate']:.3f} "
            f"VAL={val_auc:.4f} OOS={oos_auc:.4f} min={row['min_val_oos']:.4f}")
    return pd.DataFrame(results)


def main() -> int:
    log("loading klines + Tier0 indicator frame...")
    klines = load_klines()
    ind = build_indicator_frame(klines)

    dem = compute_demarker(klines["high"], klines["low"])
    ind_dem = ind.copy()
    ind_dem["dem"] = dem.to_numpy()
    r_dem = sweep_signal("demarker_extreme", klines, ind_dem, dem >= 0.90, dem <= 0.10,
                         dem.fillna(0.5).to_numpy(), FEATURE_COLUMNS + ["dem"])

    levels, _ = kalman_level_and_velocity(klines["close"].to_numpy())
    kalman_dev = pd.Series((klines["close"].to_numpy() - levels) / levels, index=klines.index)
    kalman_dev_z = rolling_zscore(kalman_dev)
    ind_kal = ind.copy()
    ind_kal["kalman_dev_z"] = kalman_dev_z.to_numpy()
    r_kal = sweep_signal("kalman_deviation_meanrev", klines, ind_kal, kalman_dev_z >= 2.0, kalman_dev_z <= -2.0,
                         kalman_dev_z.fillna(0.0).to_numpy(), FEATURE_COLUMNS + ["kalman_dev_z"])

    all_results = pd.concat([r_dem, r_kal], ignore_index=True)
    out_dir = ROOT / "tmp/eth_kalman_demarker_gridscreen_20260831"
    all_results.to_csv(out_dir / "ksweep_results.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["K", "val_auc", "oos_auc", "min_val_oos", "val_oos_gap", "hit_rate", "n_train", "n_val", "n_oos"]
    for name in all_results["signal"].unique():
        sub = all_results[all_results["signal"] == name].sort_values("min_val_oos", ascending=False)
        best_K = sub.iloc[0]["K"]
        at_edge = best_K in (K_GRID[0], K_GRID[-1])
        log(f"\n=== {name}: full K sweep (sorted by min(VAL,OOS)) ===")
        print(sub[cols].to_string(index=False))
        if at_edge:
            log(f"  ⚠️ WARNING: best K={best_K} sits at the grid EDGE -- extend K_GRID before finalizing")

    log(f"\nsaved -> {out_dir / 'ksweep_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
