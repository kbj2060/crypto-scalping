"""
Layer 5 sizing model: LightGBM quantile regression (q10/q50/q90 of net
return) per side, trained on the same CUSUM-gated event set as Layer 3
(build_trailing_targets), reusing the trailing-exit realized net return as
target -- but now predicting the full conditional distribution instead of
a point estimate, so margin_fraction can react to predicted UNCERTAINTY
(q90-q10 width) as well as predicted center (q50), not just point
conviction. Mirrors this project's existing m7 quantile_forest /
m7_q10/q90/qwidth pattern.

margin_fraction = clip(BASE + K_CENTER*q50 - K_WIDTH*(q90-q10), MIN, MAX)
notional = margin_fraction * LEVERAGE (fixed at 3x per the futures sizing
contract's canonical example)
"""
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from build_omega1_2_triple_barrier_labels_btc_20260708 import _atr_price_move  # noqa: E402
from compare_btc_label_schemes_20260803 import cusum_events  # noqa: E402
import train_btc_exit_stopping_rl_20260803 as M  # noqa: E402
from build_btc_cusum_trailing_final_model_20260803 import build_trailing_targets, EXCLUDE_COLS, VAL_START, OOS_START, OOS_END  # noqa: E402

MODEL_DIR = ROOT / "data/ensemble/supervised"
LEVERAGE = 3.0
BASE_FRAC, K_CENTER, K_WIDTH, MIN_FRAC, MAX_FRAC = 0.15, 15.0, 8.0, 0.10, 0.50


def main():
    frame = pd.read_parquet(M.FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    n = len(frame)
    atr = _atr_price_move(frame)
    events = cusum_events(frame, atr, mult=2.0)
    events = events[events < n - 48 - 2]
    targets = build_trailing_targets(frame, events)
    feat_cols = [c for c in frame.columns if c not in EXCLUDE_COLS]
    event_feats = frame.loc[targets["i"], ["timestamp"] + feat_cols].reset_index(drop=True)
    data = pd.concat([targets.drop(columns=["timestamp"]).reset_index(drop=True), event_feats], axis=1)
    train = data[data["timestamp"] < VAL_START]

    models = {}
    for side, target in [("long", "long_net"), ("short", "short_net")]:
        for q, tag in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=300, num_leaves=31,
                                   learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                                   random_state=0, verbosity=-1)
            m.fit(train[feat_cols], train[target])
            models[f"{side}_{tag}"] = m
        print(f"trained {side} quantile models", flush=True)

    for name, m in models.items():
        with open(MODEL_DIR / f"btc_layer5_sizing_{name}.pkl", "wb") as f:
            pickle.dump(m, f)
    print(f"saved 6 quantile models to {MODEL_DIR}")

    # --- evaluate on Fresh-Forward walk-forward trades ---
    t = pd.read_csv(ROOT / "tmp/btc_fresh_forward_walkforward_trades_20260803.csv", parse_dates=["entry_ts", "exit_ts"])
    ts_to_idx = pd.Series(frame.index.values, index=frame["timestamp"])
    event_idx = ts_to_idx.reindex(t["entry_ts"] - pd.Timedelta(minutes=5)).to_numpy()
    valid = ~np.isnan(event_idx)
    t = t[valid].copy()
    event_idx = event_idx[valid].astype(int)
    X = frame.loc[event_idx, feat_cols]

    q10 = np.where(t["side"].to_numpy() == 1, models["long_q10"].predict(X), models["short_q10"].predict(X))
    q50 = np.where(t["side"].to_numpy() == 1, models["long_q50"].predict(X), models["short_q50"].predict(X))
    q90 = np.where(t["side"].to_numpy() == 1, models["long_q90"].predict(X), models["short_q90"].predict(X))
    width = q90 - q10
    t["q10"], t["q50"], t["q90"], t["width"] = q10, q50, q90, width
    t["margin_fraction"] = np.clip(BASE_FRAC + K_CENTER * q50 - K_WIDTH * width, MIN_FRAC, MAX_FRAC)
    t["notional"] = t["margin_fraction"] * LEVERAGE
    t["account_pnl"] = t["net"] * t["notional"]
    t["period"] = np.where(t["entry_ts"] < OOS_START, "VAL", "OOS")

    def mdd(df):
        eq = 1.0 + df.sort_values("exit_ts")["account_pnl"].cumsum()
        peak = eq.cummax()
        return ((eq - peak) / peak).min()

    print("\n=== Layer 5 quantile-based sizing (vs prior conviction-only sizing) ===")
    for label, df in [("VAL", t[t.period == "VAL"]), ("OOS", t[t.period == "OOS"])]:
        print(f"{label}: n={len(df)} avg_margin_frac={df.margin_fraction.mean():.3f} avg_notional={df.notional.mean():.2f}x "
              f"account_pnl_mean={100*df.account_pnl.mean():.3f}% sum={100*df.account_pnl.sum():.2f}% "
              f"MDD={100*mdd(df):.2f}% Calmar={df.account_pnl.sum()/abs(mdd(df)):.2f}")

    t.to_csv(ROOT / "tmp/btc_layer5_quantile_sizing_trades_20260803.csv", index=False)
    print(f"\nwrote tmp/btc_layer5_quantile_sizing_trades_20260803.csv")


if __name__ == "__main__":
    main()
