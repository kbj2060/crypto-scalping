#!/usr/bin/env python3
"""fib_extension_exhaustion / liquidity_sweep_topdown / orthogonal_combo / smt_divergence(전부 이미
배포된 트레일링스톱 경제성게이트)에 방향-뒤집기 대조군을 소급 적용.

각 신호의 backtest_eth_<name>_trailing_holdout_exposure_20260831.py에서 그대로 가져온 FIRES_CSV/
HORIZON_BARS/SL/ARM/TRAIL(실제 채택·배포된 설정)만 사용 -- HOLDOUT은 여전히 건드리지 않고 VAL+OOS만
감사(giveback 감사와 동일 원칙). fib_extension_exhaustion과 orthogonal_combo는 원래 세션 자체가
"ARM=0.5 exit-structure artifact"(고승률의 원인)를 이미 의심해 random-entry baseline으로 확인했으나,
그건 "트리거 타이밍이 무작위보다 나은가"를 묻는 별개의 검정이고, 이번 방향뒤집기는 "같은 타이밍에서
방향이 맞았는가"를 묻는 좀 더 직접적인 검정이라 상호보완적 -- 특히 이 둘을 주의 깊게 본다.
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
OUT_PATH = ROOT / "data/research/evidence_signals_costgate_flip_audit_20260901/report.json"
MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.001
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SIGNALS = {
    "fib_extension_exhaustion": {
        "fires_csv": ROOT / "data/labels/eth_5m_fib_extension_exhaustion_metalabel_20260831/eth_5m_fib_extension_exhaustion_metalabel_FINAL_features.csv",
        "horizon_bars": 20, "sl": 3.5, "arm": 0.5, "trail": 0.1,
    },
    "liquidity_sweep_topdown": {
        "fires_csv": ROOT / "data/labels/eth_5m_liquidity_sweep_topdown_metalabel_20260830/eth_5m_liquidity_sweep_topdown_metalabel_features_H30_GAP12_K4.0.csv",
        "horizon_bars": 30, "sl": 4.0, "arm": 2.0, "trail": 0.1,
    },
    "orthogonal_combo": {
        "fires_csv": ROOT / "data/labels/eth_5m_orthogonal_combo_metalabel_20260830/eth_5m_orthogonal_combo_metalabel_features_H24_GAP12_ALLFIRES.csv",
        "horizon_bars": 24, "sl": 4.0, "arm": 0.5, "trail": 0.1,
    },
    "smt_divergence": {
        "fires_csv": ROOT / "data/labels/eth_5m_smt_divergence_metalabel_20260831/eth_5m_smt_divergence_metalabel_features.csv",
        "horizon_bars": 72, "sl": 4.0, "arm": 2.0, "trail": 0.1,
    },
}


def run_one(klines: pd.DataFrame, cfg: dict, scores_sign: float) -> dict:
    fires = pd.read_csv(cfg["fires_csv"], parse_dates=["timestamp"])
    fires = fires.loc[fires["timestamp"] < HOLDOUT_START].sort_values("pos").reset_index(drop=True)
    decision_indices = fires["pos"].to_numpy(dtype=np.int64)
    is_long = (fires["side"] == "bottom").to_numpy()
    scores = np.where(is_long, 1.0, -1.0) * scores_sign
    atr = fires["atr_pct"].to_numpy()
    sl_moves, arm_moves, trail_moves = cfg["sl"] * atr, cfg["arm"] * atr, cfg["trail"] * atr
    tp_placeholder = np.full(len(fires), 999.0)

    ts = klines["timestamp"]
    open_px, high, low, close = (klines[c].to_numpy() for c in ("open", "high", "low", "close"))

    out = {}
    for wname, (start, end) in {"val": (VAL_START, OOS_START), "oos": (OOS_START, HOLDOUT_START)}.items():
        eligible = purged_decision_mask(ts, start=start, end=end, horizon_bars=cfg["horizon_bars"])
        eligible_set = set(np.flatnonzero(eligible).tolist())
        mask = np.array([d in eligible_set for d in decision_indices])
        result = simulate_single_position(
            timestamps=ts, open_px=open_px, high=high, low=low, close=close,
            decision_indices=decision_indices[mask], scores=scores[mask],
            tp_moves=tp_placeholder[mask], sl_moves=sl_moves[mask],
            upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=cfg["horizon_bars"],
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
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    report = {}
    for name, cfg in SIGNALS.items():
        print(f"\n=== {name} (SL={cfg['sl']} ARM={cfg['arm']} Trail={cfg['trail']}, H={cfg['horizon_bars']}) ===")
        real = run_one(klines, cfg, 1.0)
        flip = run_one(klines, cfg, -1.0)
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
