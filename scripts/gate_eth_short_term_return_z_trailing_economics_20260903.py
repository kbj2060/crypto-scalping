#!/usr/bin/env python3
"""ETH `short_term_return_z` **트레일링스톱 경제성게이트** -- 마지막 미검증 배포신호.

## 왜 이걸 돌리는가

ETH 증거신호 8종 중 `short_term_return_z`만 경제성 게이트를 한 번도 통과하지 않았다
(호메로스 README 표에 "미검증"으로 남아 있다). 대시보드에서는 이미 확률 칩으로 서빙 중이므로
"이 칩을 매매 근거로 봐도 되는가"에 답이 없는 상태다.

## 설계 (`gate_btc_evidence_signals_trailing_economics_20260902.py` 그대로)

  · 96셀 그리드: SL[1.5~4.0] x ARM[0.5~2.0] x Trail[0.1~0.5]
  · 표준 왕복비용 **10bp** (수수료 우대 가정 금지)
  · margin 0.30 / leverage 3.0 -> notional 0.90 (Futures Risk Sizing Contract)
  · 보유한도 = 라벨 H = **12봉** (ETH 확정값: touch / 12 / 1.75)
  · `purged_decision_mask` + `simulate_single_position`

⭐**방향뒤집기 대조군을 96셀 전량에 적용한다.**
⭐**ARM<1.0은 노이즈 수확 아티팩트**로 따로 집계한다.

⚠️**타임스탬프 매핑 강제.** 저장된 `pos`는 빌더 프레임의 행 인덱스이므로 klines 인덱스로
그대로 쓰면 안 된다(BTC에서 108봉 어긋난 사고). searchsorted + 완전일치 검증.

⚠️**HOLDOUT 미터치.** VAL+OOS만 본다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402

KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
FIRES_CSV = (ROOT / "data/labels/eth_5m_short_term_return_z_metalabel_20260829"
             / "eth_5m_short_term_return_z_metalabel_features.csv")
OUT = ROOT / "data/research/eth_short_term_return_z_costgate_20260903/report.json"

MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.001
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
HORIZON_BARS = 12                    # ETH 확정 라벨 H

SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ARM_GRID = [0.5, 1.0, 1.5, 2.0]
TRAIL_GRID = [0.1, 0.2, 0.3, 0.5]


def log(m): print(f"[eth-strz] {m}", flush=True)


def run_grid(klines, fires, horizon_bars):
    ts = klines["timestamp"]
    o, h, l, c = (klines[x].to_numpy() for x in ("open", "high", "low", "close"))

    kl_ts = ts.to_numpy()
    f_ts = pd.to_datetime(fires["timestamp"]).to_numpy()
    dec = np.searchsorted(kl_ts, f_ts)
    inb = dec < len(kl_ts)
    if not inb.all():
        fires, f_ts, dec = fires.loc[inb].reset_index(drop=True), f_ts[inb], dec[inb]
    bad = int((kl_ts[dec] != f_ts).sum())
    if bad:
        raise ValueError(f"fires 타임스탬프가 klines에 없다: {bad}/{len(dec)}건")

    is_long = (fires["side"].astype(str) == "bottom").to_numpy()
    atr = fires["atr_pct"].to_numpy(dtype=float)
    if not np.all(np.diff(dec) >= 0):
        order = np.argsort(dec, kind="stable")
        dec, is_long, atr = dec[order], is_long[order], atr[order]

    masks = {}
    for wname, (s, e) in {"val": (VAL_START, OOS_START), "oos": (OOS_START, HOLDOUT_START)}.items():
        el = set(np.flatnonzero(purged_decision_mask(
            ts, start=s, end=e, horizon_bars=horizon_bars)).tolist())
        masks[wname] = np.array([d in el for d in dec])

    tp = np.full(len(dec), 999.0)
    cells = []
    for sl in SL_GRID:
        for arm in ARM_GRID:
            for trail in TRAIL_GRID:
                row = {"sl": sl, "arm": arm, "trail": trail}
                for wname, m in masks.items():
                    for tag, sgn in (("fwd", 1.0), ("flip", -1.0)):
                        sc = np.where(is_long, 1.0, -1.0) * sgn
                        r = simulate_single_position(
                            timestamps=ts, open_px=o, high=h, low=l, close=c,
                            decision_indices=dec[m], scores=sc[m], tp_moves=tp[m],
                            sl_moves=(sl*atr)[m], upper_threshold=1.0, lower_threshold=-1.0,
                            horizon_bars=horizon_bars, margin_fraction=MARGIN_FRACTION,
                            leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
                            arm_moves=(arm*atr)[m], trail_moves=(trail*atr)[m])
                        led = r.ledger
                        row[f"{wname}_{tag}_bp"] = (float(led["trade_return"].mean()*1e4)
                                                    if len(led) else float("nan"))
                        if tag == "fwd":
                            row[f"{wname}_n"] = int(len(led))
                            row[f"{wname}_wr"] = (float((led["price_move"] > 0).mean())
                                                  if len(led) else float("nan"))
                cells.append(row)
    return cells, {k: int(v.sum()) for k, v in masks.items()}


def main() -> int:
    t0 = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    kl = pd.read_csv(KLINES)
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True).dt.tz_localize(None)
    kl = kl.sort_values("timestamp").reset_index(drop=True)
    log(f"ETH 5m klines {len(kl):,}행")

    fires = pd.read_csv(FIRES_CSV, usecols=["timestamp", "side", "atr_pct"])
    fires["timestamp"] = pd.to_datetime(fires["timestamp"], utc=True).dt.tz_localize(None)
    fires = fires.loc[fires["timestamp"] < HOLDOUT_START].reset_index(drop=True)
    log(f"fires {len(fires):,}건 (HOLDOUT 이전) H={HORIZON_BARS}")
    log(f"그리드 {len(SL_GRID)*len(ARM_GRID)*len(TRAIL_GRID)}셀 x 정방향/뒤집기 x VAL/OOS")
    log("⚠️HOLDOUT 미터치")

    cells, ns = run_grid(kl, fires, HORIZON_BARS)
    log(f"후보 VAL {ns['val']} / OOS {ns['oos']}")

    passing = [c for c in cells if c["val_fwd_bp"] > 0 and c["oos_fwd_bp"] > 0]
    genuine = [c for c in passing
               if c["val_fwd_bp"] > c["val_flip_bp"] and c["oos_fwd_bp"] > c["oos_flip_bp"]]
    gen_arm1 = [c for c in genuine if c["arm"] >= 1.0]
    log(f"VAL+OOS 동시양수 {len(passing)}/96")
    log(f"방향뒤집기 통과(진짜) {len(genuine)}/96  그중 ARM>=1.0 **{len(gen_arm1)}**")
    best = None
    if gen_arm1:
        best = max(gen_arm1, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"]))
        log(f"⭐최선(ARM>=1.0): SL={best['sl']} ARM={best['arm']} Trail={best['trail']}")
        log(f"   VAL {best['val_fwd_bp']:+.2f}bp (뒤 {best['val_flip_bp']:+.2f}) "
            f"n={best['val_n']} 승률 {best['val_wr']*100:.1f}%")
        log(f"   OOS {best['oos_fwd_bp']:+.2f}bp (뒤 {best['oos_flip_bp']:+.2f}) "
            f"n={best['oos_n']} 승률 {best['oos_wr']*100:.1f}%")
    else:
        b = max(cells, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"]))
        log(f"❌ARM>=1.0 진짜 조합 없음. 격자 최선: SL={b['sl']} ARM={b['arm']} "
            f"Trail={b['trail']}  VAL {b['val_fwd_bp']:+.2f} / OOS {b['oos_fwd_bp']:+.2f}bp")

    rep = {"asset": "ETHUSDT", "signal": "short_term_return_z",
           "cost_bp": 10.0, "margin_fraction": MARGIN_FRACTION, "leverage": LEVERAGE,
           "horizon_bars": HORIZON_BARS, "n_fires": int(len(fires)), "n_candidates": ns,
           "grid": {"sl": SL_GRID, "arm": ARM_GRID, "trail": TRAIL_GRID},
           "holdout_touched": False,
           "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
           "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
           "n_passing_96": len(passing), "n_genuine": len(genuine),
           "n_genuine_arm_ge_1": len(gen_arm1), "best_arm_ge_1": best,
           "genuine_arm_ge_1": gen_arm1, "cells": cells,
           "runtime_sec": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
