"""ETH trend-DL pre-probe: does multivariate combination of the 9 microstructure-panel
features materially exceed the best univariate IC at h=12/48?

Pre-registered design (locked before touching combined data):
docs/experiments/eth_candidate_trend_dl_multivariate_probe_20260823.md

DL escalation is justified ONLY if VAL IC(combined) >= 1.5x VAL IC(best univariate)
AND circular-shift |z| >= 3. Fixed hyperparameters, no tuning. TRAIN fit -> VAL eval.
OOS (2026-08-17+) untouched.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.research_eth_microstructure_panel_1h4h_direction_screen_20260823 import (  # noqa: E402
    ALL_FEATURES,
    build_frame,
    canonical_col,
    circular_shift_z,
)

SPLITS = {"TRAIN": ("2026-05-03", "2026-07-31"), "VAL": ("2026-08-01", "2026-08-16")}
HORIZONS = [12, 48]
# Best univariate VAL ICs from the screen (pre-registered bars: 1.5x these)
BEST_UNIVARIATE_VAL_IC = {12: 0.0713, 48: 0.0459}
LGB_SEEDS = [914275, 382041, 550918, 127463, 698230]  # genuinely random, not fixed-increment


def main() -> None:
    frame = build_frame()
    cols = [canonical_col(f) for f in ALL_FEATURES]

    train = frame[(frame["timestamp"] >= SPLITS["TRAIN"][0]) & (frame["timestamp"] <= SPLITS["TRAIN"][1])]
    val = frame[(frame["timestamp"] >= SPLITS["VAL"][0]) & (frame["timestamp"] <= SPLITS["VAL"][1])]

    for h in HORIZONS:
        tr = train.dropna(subset=cols + [f"fwd_{h}"])
        va = val.dropna(subset=cols + [f"fwd_{h}"])
        X_tr, y_tr = tr[cols].to_numpy(), tr[f"fwd_{h}"].to_numpy()
        X_va, y_va = va[cols].to_numpy(), va[f"fwd_{h}"].to_numpy()
        mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
        sd[sd == 0] = 1.0

        print(f"=== h={h} ({h * 5}min) | train n={len(tr)} val n={len(va)} ===")

        ridge = Ridge(alpha=1.0).fit((X_tr - mu) / sd, y_tr)
        pred_r = ridge.predict((X_va - mu) / sd)
        ic_r, z_r = circular_shift_z(pred_r, y_va)
        print(f"  ridge:    VAL IC={ic_r:+.4f}  shift-z={z_r:+.2f}")

        preds = np.zeros(len(va))
        for seed in LGB_SEEDS:
            model = lgb.LGBMRegressor(
                num_leaves=15, n_estimators=200, learning_rate=0.05,
                min_child_samples=100, random_state=seed, verbosity=-1,
            ).fit(X_tr, y_tr)
            preds += model.predict(X_va)
        preds /= len(LGB_SEEDS)
        ic_g, z_g = circular_shift_z(preds, y_va)
        print(f"  lgbm(5s): VAL IC={ic_g:+.4f}  shift-z={z_g:+.2f}")

        best_uni = BEST_UNIVARIATE_VAL_IC[h]
        bar = 1.5 * best_uni
        best_combined = max(ic_r, ic_g)
        best_z = z_r if ic_r >= ic_g else z_g
        passed = best_combined >= bar and abs(best_z) >= 3
        print(f"  bar: combined IC >= {bar:.4f} (=1.5x univariate {best_uni:.4f}) AND |z|>=3")
        print(f"  best combined IC={best_combined:+.4f} (ratio {best_combined / best_uni:+.2f}x) -> "
              f"{'PASS: DL escalation justified' if passed else 'FAIL: DL not justified'}\n")


if __name__ == "__main__":
    main()
