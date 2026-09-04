#!/usr/bin/env python3
"""demarker_extreme / kalman_deviation_meanrev(둘 다 배포된 트레일링스톱 경제성게이트)에
방향-뒤집기 대조군 소급 적용.

backtest_eth_kalman_demarker_trailing_holdout_exposure_20260831.py와 동일하게 fires를 원본
지표 계산에서 재구성(build_fires 재사용, 재구현 아님) -- 유일한 차이는 VAL/OOS로 나눠서 계산하고
scores를 뒤집은 대조군을 추가하는 것.
"""
from __future__ import annotations

import json
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

OUT_PATH = ROOT / "data/research/kalman_demarker_costgate_flip_audit_20260901/report.json"
MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.001
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SIGNAL_CONFIG = {
    "demarker_extreme": {"horizon": 8, "gap": 12, "K": 0.70, "sl": 2.0, "arm": 1.5, "trail": 0.1},
    "kalman_deviation_meanrev": {"horizon": 12, "gap": 12, "K": 2.5, "sl": 4.0, "arm": 1.5, "trail": 0.1},
}


def run_one(klines, ind, trigger_top, trigger_bottom, extremeness, feature_cols, cfg, scores_sign) -> dict:
    fires = build_fires(klines, ind, trigger_top, trigger_bottom, extremeness, feature_cols,
                         cfg["horizon"], cfg["gap"], cfg["K"])
    fires = fires.loc[fires["timestamp"] < HOLDOUT_START].sort_values("pos").reset_index(drop=True)

    open_px, high, low, close = (klines[c].to_numpy() for c in ("open", "high", "low", "close"))
    ts_full = klines["timestamp"]
    atr_pct = ind["atr_pct"].to_numpy()

    decision_indices = fires["pos"].to_numpy(dtype=np.int64)
    scores = np.where(fires["side"].to_numpy() == "bottom", 1.0, -1.0) * scores_sign
    atr = atr_pct[decision_indices]
    sl_moves, arm_moves, trail_moves = cfg["sl"] * atr, cfg["arm"] * atr, cfg["trail"] * atr
    tp_placeholder = np.full(len(fires), 999.0)

    out = {}
    for wname, (start, end) in {"val": (VAL_START, OOS_START), "oos": (OOS_START, HOLDOUT_START)}.items():
        eligible = purged_decision_mask(ts_full, start=start, end=end, horizon_bars=cfg["horizon"])
        eligible_set = set(np.flatnonzero(eligible).tolist())
        mask = np.array([d in eligible_set for d in decision_indices])
        result = simulate_single_position(
            timestamps=ts_full, open_px=open_px, high=high, low=low, close=close,
            decision_indices=decision_indices[mask], scores=scores[mask],
            tp_moves=tp_placeholder[mask], sl_moves=sl_moves[mask],
            upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=cfg["horizon"],
            margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
            arm_moves=arm_moves[mask], trail_moves=trail_moves[mask],
        )
        ledger = result.ledger
        n_trades = int(len(ledger))
        if n_trades:
            win_rate = float((ledger["price_move"] > 0).mean())
            avg_trade_bp = float(ledger["trade_return"].mean() * 10000)
        else:
            win_rate = avg_trade_bp = float("nan")
        out[wname] = {"n_candidates": int(mask.sum()), "n_trades": n_trades, "win_rate": win_rate, "avg_trade_bp": avg_trade_bp}
    return out


def main() -> int:
    print("loading klines + Tier0 indicator frame...", flush=True)
    klines = load_klines()
    ind = build_indicator_frame(klines)

    dem = compute_demarker(klines["high"], klines["low"])
    ind_dem = ind.copy()
    ind_dem["dem"] = dem.to_numpy()

    levels, _ = kalman_level_and_velocity(klines["close"].to_numpy())
    kalman_dev = pd.Series((klines["close"].to_numpy() - levels) / levels, index=klines.index)
    kalman_dev_z = rolling_zscore(kalman_dev)
    ind_kal = ind.copy()
    ind_kal["kalman_dev_z"] = kalman_dev_z.to_numpy()

    args_by_signal = {
        "demarker_extreme": (ind_dem, dem >= 0.90, dem <= 0.10, dem.fillna(0.5).to_numpy(), FEATURE_COLUMNS + ["dem"]),
        "kalman_deviation_meanrev": (ind_kal, kalman_dev_z >= 2.0, kalman_dev_z <= -2.0, kalman_dev_z.fillna(0.0).to_numpy(), FEATURE_COLUMNS + ["kalman_dev_z"]),
    }

    report = {}
    for name, cfg in SIGNAL_CONFIG.items():
        ind_x, trig_top, trig_bot, extremeness, feat_cols = args_by_signal[name]
        print(f"\n=== {name} (SL={cfg['sl']} ARM={cfg['arm']} Trail={cfg['trail']}, H={cfg['horizon']}) ===")
        real = run_one(klines, ind_x, trig_top, trig_bot, extremeness, feat_cols, cfg, 1.0)
        flip = run_one(klines, ind_x, trig_top, trig_bot, extremeness, feat_cols, cfg, -1.0)
        for w in ("val", "oos"):
            r, f = real[w], flip[w]
            print(f"  {w}: real n={r['n_trades']} avg={r['avg_trade_bp']:+.2f}bp win={r['win_rate']:.1%} | "
                  f"flip n={f['n_trades']} avg={f['avg_trade_bp']:+.2f}bp win={f['win_rate']:.1%}")
        genuine = (real["val"]["avg_trade_bp"] > flip["val"]["avg_trade_bp"] and
                   real["oos"]["avg_trade_bp"] > flip["oos"]["avg_trade_bp"] and
                   real["val"]["avg_trade_bp"] > 0 and real["oos"]["avg_trade_bp"] > 0)
        print(f"  => {'GENUINE' if genuine else 'ARTIFACT-SUSPECT'}")
        report[name] = {"real": real, "flipped": flip, "genuine": genuine}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
