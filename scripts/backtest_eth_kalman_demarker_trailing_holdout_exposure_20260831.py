#!/usr/bin/env python3
"""SINGLE final HOLDOUT (2026-04-01+) exposure for the demarker_extreme / kalman_deviation_meanrev
trailing-stop cost-gates -- final configs picked from backtest_eth_kalman_demarker_trailing_
gridsearch_20260831.py (96/96 combos passed VAL+OOS for BOTH signals) + the optimistic-ordering
crosscheck (chat, both within ~1bp of the standard engine): demarker_extreme SL=2.0/ARM=1.5/
Trail=0.1xATR, kalman_deviation_meanrev SL=4.0/ARM=1.5/Trail=0.1xATR. Companion to research_eth_
kalman_demarker_metalabel_holdout_20260831.py's classification HOLDOUT touch, same day, same
config -- together these are the ONE holdout exposure for these 2 models; do not re-run with a
different SL/ARM/Trail after seeing this result.

Runs locally (CPU only, no GPU needed).
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

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from research_eth_candidate_pool_raw_lift_check_20260831 import (  # noqa: E402
    kalman_level_and_velocity,
    rolling_zscore,
)
from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker  # noqa: E402
from research_eth_kalman_demarker_gridscreen_20260831 import build_fires, load_klines  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.001
HOLDOUT_START = pd.Timestamp("2026-04-01")

SIGNAL_CONFIG = {
    "demarker_extreme": {"horizon": 8, "gap": 12, "K": 0.70, "sl": 2.0, "arm": 1.5, "trail": 0.1},
    "kalman_deviation_meanrev": {"horizon": 12, "gap": 12, "K": 2.5, "sl": 4.0, "arm": 1.5, "trail": 0.1},
}


def log(msg: str) -> None:
    print(f"[kalman_demarker_trailing_holdout] {msg}", flush=True)


def holdout_signal(name: str, klines: pd.DataFrame, ind: pd.DataFrame, trigger_top: pd.Series,
                    trigger_bottom: pd.Series, extremeness: np.ndarray, feature_cols: list[str]) -> None:
    cfg = SIGNAL_CONFIG[name]
    horizon = cfg["horizon"]
    fires = build_fires(klines, ind, trigger_top, trigger_bottom, extremeness, feature_cols,
                        horizon, cfg["gap"], cfg["K"])
    fires = fires.sort_values("pos").reset_index(drop=True)

    open_px = klines["open"].to_numpy()
    high = klines["high"].to_numpy()
    low = klines["low"].to_numpy()
    close = klines["close"].to_numpy()
    ts_full = klines["timestamp"]
    atr_pct = ind["atr_pct"].to_numpy()

    decision_indices = fires["pos"].to_numpy(dtype=np.int64)
    scores = np.where(fires["side"].to_numpy() == "bottom", 1.0, -1.0)
    atr = atr_pct[decision_indices]

    end = ts_full.max() + pd.Timedelta(minutes=5)
    eligible = purged_decision_mask(ts_full, start=HOLDOUT_START, end=end, horizon_bars=horizon)
    eligible_set = set(np.flatnonzero(eligible).tolist())
    mask = np.array([d in eligible_set for d in decision_indices])
    log(f"\n=== {name} (H={horizon}, GAP={cfg['gap']}, K={cfg['K']}) "
        f"SL={cfg['sl']}/ARM={cfg['arm']}/Trail={cfg['trail']} -- SINGLE HOLDOUT EXPOSURE ===")
    log(f"  eligible HOLDOUT decisions: {int(mask.sum())} (of {len(fires)} total cluster-anchored candidates)")

    sl_moves = cfg["sl"] * atr
    arm_moves = cfg["arm"] * atr
    trail_moves = cfg["trail"] * atr
    tp_placeholder = np.full(len(fires), 999.0)

    result = simulate_single_position(
        timestamps=ts_full, open_px=open_px, high=high, low=low, close=close,
        decision_indices=decision_indices[mask], scores=scores[mask],
        tp_moves=tp_placeholder[mask], sl_moves=sl_moves[mask],
        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=horizon,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
        arm_moves=arm_moves[mask], trail_moves=trail_moves[mask],
    )
    ledger = result.ledger
    n_trades = int(len(ledger))
    avg_bp = float(ledger["trade_return"].mean() * 10000) if n_trades else float("nan")
    win_rate = float((ledger["price_move"] > 0).mean()) if n_trades else float("nan")
    total_return = float((1.0 + ledger["trade_return"]).prod() - 1.0) if n_trades else float("nan")
    log(f"  HOLDOUT (SINGLE EXPOSURE, now consumed): n_trades={n_trades} avg={avg_bp:+.2f}bp "
        f"win_rate={win_rate:.1%} total_account_return={total_return:+.2%} "
        f"skipped_while_open={result.skipped_while_open}")

    out_dir = ROOT / "tmp/eth_kalman_demarker_holdout_20260831"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(out_dir / f"{name}_holdout_ledger.csv", index=False)


def main() -> int:
    log("loading klines + Tier0 indicator frame...")
    klines = load_klines()
    ind = build_indicator_frame(klines)

    dem = compute_demarker(klines["high"], klines["low"])
    ind_dem = ind.copy()
    ind_dem["dem"] = dem.to_numpy()
    holdout_signal("demarker_extreme", klines, ind_dem, dem >= 0.90, dem <= 0.10,
                   dem.fillna(0.5).to_numpy(), FEATURE_COLUMNS + ["dem"])

    levels, _ = kalman_level_and_velocity(klines["close"].to_numpy())
    kalman_dev = pd.Series((klines["close"].to_numpy() - levels) / levels, index=klines.index)
    kalman_dev_z = rolling_zscore(kalman_dev)
    ind_kal = ind.copy()
    ind_kal["kalman_dev_z"] = kalman_dev_z.to_numpy()
    holdout_signal("kalman_deviation_meanrev", klines, ind_kal, kalman_dev_z >= 2.0, kalman_dev_z <= -2.0,
                   kalman_dev_z.fillna(0.0).to_numpy(), FEATURE_COLUMNS + ["kalman_dev_z"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
