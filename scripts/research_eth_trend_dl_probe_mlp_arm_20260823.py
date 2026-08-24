"""Tiny-MLP contingency arm of the trend-DL multivariate probe (user-requested).

The probe doc pre-scoped this arm: "사용자가 그래도 DL을 원하면 소형 MLP(<10k 파라미터,
강한 정칙화) 정도가 상한" -- executed now on the user's explicit push, with the recorded
expectation (from the ridge>LGBM capacity gradient) that it should NOT beat ridge.
Same data/splits as the probe; VAL was already consumed, so this is a bounded contingency
completion, not a new search. Two sizes only, no tuning, N=5 random seeds each.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.research_eth_microstructure_panel_1h4h_direction_screen_20260823 import (  # noqa: E402
    ALL_FEATURES,
    build_frame,
    canonical_col,
    circular_shift_z,
)

warnings.filterwarnings("ignore")

SEEDS = [402193, 87456, 731905, 264871, 549038]
SIZES = {"mlp_16x8": (16, 8), "mlp_64x32": (64, 32)}
RIDGE_VAL_IC = {12: 0.0698, 48: 0.0747}  # the bar to beat, from the probe


def main() -> None:
    frame = build_frame()
    cols = [canonical_col(f) for f in ALL_FEATURES]
    train = frame[(frame["timestamp"] >= "2026-05-03") & (frame["timestamp"] <= "2026-07-31")]
    val = frame[(frame["timestamp"] >= "2026-08-01") & (frame["timestamp"] <= "2026-08-16")]

    for h in [12, 48]:
        tr = train.dropna(subset=cols + [f"fwd_{h}"])
        va = val.dropna(subset=cols + [f"fwd_{h}"])
        X_tr, y_tr = tr[cols].to_numpy(), tr[f"fwd_{h}"].to_numpy()
        X_va, y_va = va[cols].to_numpy(), va[f"fwd_{h}"].to_numpy()
        mu, sd = X_tr.mean(0), X_tr.std(0)
        sd[sd == 0] = 1.0
        Xs_tr, Xs_va = (X_tr - mu) / sd, (X_va - mu) / sd

        print(f"=== h={h} | ridge bar: VAL IC {RIDGE_VAL_IC[h]:+.4f} ===")
        for name, size in SIZES.items():
            per_seed = []
            preds = np.zeros(len(va))
            for seed in SEEDS:
                m = MLPRegressor(hidden_layer_sizes=size, alpha=1e-2, max_iter=500,
                                 random_state=seed, early_stopping=False).fit(Xs_tr, y_tr)
                p = m.predict(Xs_va)
                preds += p
                per_seed.append(circular_shift_z(p, y_va)[0])
            preds /= len(SEEDS)
            ic, z = circular_shift_z(preds, y_va)
            beat = ic > RIDGE_VAL_IC[h]
            print(f"  {name}: 5-seed-avg VAL IC={ic:+.4f} shift-z={z:+.2f} "
                  f"(per-seed IC {min(per_seed):+.4f}~{max(per_seed):+.4f}) "
                  f"-> {'BEATS ridge' if beat else 'does NOT beat ridge'}")
        print()


if __name__ == "__main__":
    main()
