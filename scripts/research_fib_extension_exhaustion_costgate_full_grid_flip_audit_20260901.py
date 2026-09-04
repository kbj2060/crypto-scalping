#!/usr/bin/env python3
"""fib_extension_exhaustion 96셀 그리드 전체(VAL+OOS만, HOLDOUT 미터치)를 재실행하고, VAL+OOS
동시양수인 모든 조합에 방향뒤집기 대조군을 적용 -- 원래 채택된 SL=3.5/ARM=0.5/Trail=0.1(9/96 통과
전부 ARM=0.5) 외에 3구간(VAL/OOS/향후 재검증) 전부 견고한 다른 ARM대 조합이 있는지 확인하는
순수 진단. HOLDOUT은 여전히 안 건드림 -- 새 파라미터 채택/재배포 목적이 아니라 v2 재설계 시
참고자료용.

backtest_eth_fib_extension_exhaustion_trailing_gridsearch_20260831.py와 완전히 동일한 그리드/엔진
재사용(재구현 아님).
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

FIRES_CSV = ROOT / "data/labels/eth_5m_fib_extension_exhaustion_metalabel_20260831/eth_5m_fib_extension_exhaustion_metalabel_FINAL_features.csv"
KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_PATH = ROOT / "data/research/fib_extension_exhaustion_costgate_full_grid_flip_audit_20260901/report.json"
MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.001
HORIZON_BARS = 20
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ARM_GRID = [0.5, 1.0, 1.5, 2.0]
TRAIL_GRID = [0.1, 0.2, 0.3, 0.5]


def split_metrics(klines, decision_indices, scores, atr, sl, arm, trail, mask) -> dict:
    ts = klines["timestamp"]
    open_px, high, low, close = (klines[c].to_numpy() for c in ("open", "high", "low", "close"))
    tp_placeholder = np.full(mask.sum(), 999.0)
    result = simulate_single_position(
        timestamps=ts, open_px=open_px, high=high, low=low, close=close,
        decision_indices=decision_indices[mask], scores=scores[mask],
        tp_moves=tp_placeholder, sl_moves=(sl * atr)[mask],
        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
        arm_moves=(arm * atr)[mask], trail_moves=(trail * atr)[mask],
    )
    ledger = result.ledger
    n = int(len(ledger))
    avg_bp = float(ledger["trade_return"].mean() * 10000) if n else float("nan")
    win = float((ledger["price_move"] > 0).mean()) if n else float("nan")
    return {"n": n, "avg_bp": avg_bp, "win_rate": win}


def main() -> int:
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    fires = pd.read_csv(FIRES_CSV, parse_dates=["timestamp"])
    fires = fires.loc[fires["timestamp"] < HOLDOUT_START].sort_values("pos").reset_index(drop=True)
    decision_indices = fires["pos"].to_numpy(dtype=np.int64)
    is_long = (fires["side"] == "bottom").to_numpy()
    atr = fires["atr_pct"].to_numpy()

    ts = klines["timestamp"]
    eligible_val = purged_decision_mask(ts, start=VAL_START, end=OOS_START, horizon_bars=HORIZON_BARS)
    eligible_oos = purged_decision_mask(ts, start=OOS_START, end=HOLDOUT_START, horizon_bars=HORIZON_BARS)
    val_set, oos_set = set(np.flatnonzero(eligible_val).tolist()), set(np.flatnonzero(eligible_oos).tolist())
    val_mask = np.array([d in val_set for d in decision_indices])
    oos_mask = np.array([d in oos_set for d in decision_indices])
    print(f"VAL candidates={val_mask.sum()} OOS candidates={oos_mask.sum()}", flush=True)

    scores_real = np.where(is_long, 1.0, -1.0)
    scores_flip = -scores_real

    print("\n=== 96-cell grid (real direction, VAL+OOS) ===", flush=True)
    passing = []
    for sl in SL_GRID:
        for arm in ARM_GRID:
            for trail in TRAIL_GRID:
                v = split_metrics(klines, decision_indices, scores_real, atr, sl, arm, trail, val_mask)
                o = split_metrics(klines, decision_indices, scores_real, atr, sl, arm, trail, oos_mask)
                if v["n"] > 0 and o["n"] > 0 and v["avg_bp"] > 0 and o["avg_bp"] > 0:
                    passing.append({"sl": sl, "arm": arm, "trail": trail, "val": v, "oos": o})
    print(f"{len(passing)}/96 combos VAL+OOS both positive", flush=True)

    print("\n=== direction-flip control on passing combos ===", flush=True)
    results = []
    for p in passing:
        sl, arm, trail = p["sl"], p["arm"], p["trail"]
        fv = split_metrics(klines, decision_indices, scores_flip, atr, sl, arm, trail, val_mask)
        fo = split_metrics(klines, decision_indices, scores_flip, atr, sl, arm, trail, oos_mask)
        genuine = (p["val"]["avg_bp"] > fv["avg_bp"] and p["oos"]["avg_bp"] > fo["avg_bp"])
        tag = "GENUINE" if genuine else "artifact"
        print(f"  SL={sl:.2f} ARM={arm:.2f} Trail={trail:.2f} | "
              f"real(val={p['val']['avg_bp']:+.2f} oos={p['oos']['avg_bp']:+.2f}) "
              f"flip(val={fv['avg_bp']:+.2f} oos={fo['avg_bp']:+.2f}) [{tag}]"
              f"{' <-- non-ARM0.5' if arm != 0.5 else ''}")
        results.append({"sl": sl, "arm": arm, "trail": trail, "real": p, "flipped": {"val": fv, "oos": fo}, "genuine": genuine})

    genuine_results = [r for r in results if r["genuine"]]
    non_arm05_genuine = [r for r in genuine_results if r["arm"] != 0.5]
    print(f"\n=== {len(genuine_results)}/{len(results)} GENUINE, of which {len(non_arm05_genuine)} are non-ARM=0.5 ===")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"n_passing_96": len(passing), "flip_results": results,
                                     "n_genuine": len(genuine_results), "n_non_arm05_genuine": len(non_arm05_genuine)},
                                    ensure_ascii=False, indent=2, default=str))
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
