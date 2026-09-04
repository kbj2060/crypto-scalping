#!/usr/bin/env python3
"""taker_delta_z_climax / short_term_return_z(둘 다 이미 통과·배포된 트레일링스톱 경제성게이트)에
방향-뒤집기 대조군을 소급 적용.

`backtest_taker_and_str_z_trailing_stop_standard_engine_20260830.py`(VAL/OOS 검증 스크립트)와
동일 엔진(core.causal_futures_backtest.simulate_single_position) 재사용 -- 유일한 추가는 scores를
전부 뒤집은(-scores) 대조군 실행. 라벨/피쳐/config 전부 원본 그대로, HOLDOUT은 여전히 제외
(single-touch policy 미변경).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402

KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_PATH = ROOT / "data/research/taker_str_z_costgate_flip_audit_20260901/report.json"
MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.001

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SIGNALS = {
    "taker_delta_z_climax": {
        "fires_csv": ROOT / "data/labels/eth_5m_taker_delta_climax_metalabel_20260829/eth_5m_taker_delta_climax_metalabel_features.csv",
        "horizon_bars": 24, "sl_init": 2.0, "arm": 1.5, "trail": 0.2,
    },
    "short_term_return_z": {
        "fires_csv": ROOT / "data/labels/eth_5m_short_term_return_z_metalabel_20260829/eth_5m_short_term_return_z_metalabel_features.csv",
        "horizon_bars": 12, "sl_init": 2.0, "arm": 1.0, "trail": 0.2,
    },
}


def run_one(klines: pd.DataFrame, cfg: dict, scores_sign: float) -> dict:
    fires = pd.read_csv(cfg["fires_csv"], parse_dates=["timestamp"])
    fires = fires.loc[fires["timestamp"] < HOLDOUT_START].sort_values("pos").reset_index(drop=True)
    decision_indices = fires["pos"].to_numpy(dtype=np.int64)
    is_long = (fires["side"] == "bottom").to_numpy()
    scores = np.where(is_long, 1.0, -1.0) * scores_sign
    sl_moves = cfg["sl_init"] * fires["atr_pct"].to_numpy()
    arm_moves = cfg["arm"] * fires["atr_pct"].to_numpy()
    trail_moves = cfg["trail"] * fires["atr_pct"].to_numpy()
    tp_moves = np.zeros(len(fires))

    ts = klines["timestamp"]
    open_px, high, low, close = (klines[c].to_numpy() for c in ("open", "high", "low", "close"))

    out = {}
    for wname, (start, end) in {"val": (VAL_START, OOS_START), "oos": (OOS_START, HOLDOUT_START)}.items():
        eligible = purged_decision_mask(ts, start=start, end=end, horizon_bars=cfg["horizon_bars"])
        eligible_set = set(np.flatnonzero(eligible).tolist())
        mask = np.array([d in eligible_set for d in decision_indices])
        sub_idx, sub_scores = decision_indices[mask], scores[mask]
        sub_sl, sub_arm, sub_trail, sub_tp = sl_moves[mask], arm_moves[mask], trail_moves[mask], tp_moves[mask]
        result = simulate_single_position(
            timestamps=ts, open_px=open_px, high=high, low=low, close=close,
            decision_indices=sub_idx, scores=sub_scores, tp_moves=sub_tp, sl_moves=sub_sl,
            upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=cfg["horizon_bars"],
            margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
            arm_moves=sub_arm, trail_moves=sub_trail,
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
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    report = {}
    for name, cfg in SIGNALS.items():
        print(f"\n=== {name} (SL={cfg['sl_init']} ARM={cfg['arm']} Trail={cfg['trail']}) ===")
        real = run_one(klines, cfg, 1.0)
        flip = run_one(klines, cfg, -1.0)
        for w in ("val", "oos"):
            r, f = real[w], flip[w]
            print(f"  {w}: real n_trades={r['n_trades']} avg={r['avg_trade_bp']:+.2f}bp win={r['win_rate']:.1%} | "
                  f"flip n_trades={f['n_trades']} avg={f['avg_trade_bp']:+.2f}bp win={f['win_rate']:.1%}")
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
