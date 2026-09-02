#!/usr/bin/env python3
"""XRP V자반등 경제라벨 이식의 **Phase 0 킬게이트** -- 무작위 기준선 부호.

## 왜 이 한 숫자인가

BTC 경제라벨 이식은 4회 전부 실패했고, 그 원인이 마지막에 한 문장으로 정리됐다
(`btc_v_rebound_econ_label_closed_no_direction_skill_20260902`):

> ETH는 **무작위 기준선이 양수**(+2.6bp)라 약한 선택력으로도 통했다.
> BTC는 −1.04bp라 진짜 예측력이 필요한데 그게 없었다.
> 1시간봉으로 비용/ATR을 62%→15.1%로 낮춰 비용 장벽을 실제로 제거해도 AUC는 0.4932로 동일.

⇒ **자산이 이식 가능한지는 전체 파이프라인을 짓기 전에 이 한 숫자로 대부분 결정된다.**
무작위 진입을 같은 트레일링 브래킷에 태웠을 때 표준비용(왕복 10bp) 후 평균이 양수인가.

## 설계

세 자산(ETH/BTC/XRP)에 **동일한** 절차를 적용해 비교 가능하게 만든다.

  · 무작위 진입 B회 x 96셀 그리드(SL/ARM/Trail) -- 신호 없음, 순수 exit 구조 효과
  · 롱/숏 50:50, ATR은 그 봉의 실제 ATR(14)
  · 표준 왕복비용 10bp, margin 0.30 / leverage 3.0
  · 부수 지표: **비용/ATR 비율**(중앙 ATR 대비 10bp) -- BTC 62% / ETH 43%가 기록된 값

판정:
  · 최선 셀 무작위 기준선이 **양수** → ETH형. 약한 선택력으로도 통할 여지가 있다 → 이식 검토 가치.
  · **음수** → BTC형. 진짜 예측력이 필요하고, 같은 Tier0 피쳐셋으로는 BTC에서 실패했다 → 이식 보류.

⚠️HOLDOUT 미터치(VAL+OOS 구간만). 라이브 코드 변경 없음.
⚠️이 게이트는 **필요조건**이지 충분조건이 아니다. 통과해도 방향 예측력은 따로 봐야 한다.
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

ASSETS = {
    "ETH": ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
    "BTC": ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv",
    "XRP": ROOT / "binance_data/klines/XRPUSDT/XRPUSDT-5m-api.csv",
}
OUT = ROOT / "data/research/v_rebound_random_baseline_killgate_20260903.json"

MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.001
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ARM_GRID = [0.5, 1.0, 1.5, 2.0]
TRAIL_GRID = [0.1, 0.2, 0.3, 0.5]
HORIZON_BARS = 12          # V자반등 계열이 쓰는 대표 보유한도
N_ENTRIES, B_REPS, SEED = 3000, 8, 20260903


def log(m): print(f"[killgate] {m}", flush=True)


def load(path):
    kl = pd.read_csv(path)
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True).dt.tz_localize(None)
    kl = kl.sort_values("timestamp").reset_index(drop=True)
    h, l, c = kl["high"].to_numpy(), kl["low"].to_numpy(), kl["close"].to_numpy()
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    kl["atr_pct"] = pd.Series(tr).rolling(14).mean().to_numpy() / c
    return kl


def run(kl, rng):
    ts = kl["timestamp"]
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    atr_all = kl["atr_pct"].to_numpy()

    windows = {}
    for wname, (s, e) in {"val": (VAL_START, OOS_START), "oos": (OOS_START, HOLDOUT_START)}.items():
        windows[wname] = np.flatnonzero(
            purged_decision_mask(ts, start=s, end=e, horizon_bars=HORIZON_BARS)
            & np.isfinite(atr_all))

    acc = {}
    for _ in range(B_REPS):
        picks, sides = {}, {}
        for wname, pool in windows.items():
            n = min(N_ENTRIES, len(pool))
            picks[wname] = np.sort(rng.choice(pool, size=n, replace=False))
            sd = np.where(np.arange(n) < n // 2, 1.0, -1.0)
            rng.shuffle(sd)
            sides[wname] = sd
        for sl in SL_GRID:
            for arm in ARM_GRID:
                for trail in TRAIL_GRID:
                    key = (sl, arm, trail)
                    for wname in windows:
                        dec, sc = picks[wname], sides[wname]
                        a = atr_all[dec]
                        r = simulate_single_position(
                            timestamps=ts, open_px=o, high=h, low=l, close=c,
                            decision_indices=dec, scores=sc,
                            tp_moves=np.full(len(dec), 999.0), sl_moves=sl * a,
                            upper_threshold=1.0, lower_threshold=-1.0,
                            horizon_bars=HORIZON_BARS, margin_fraction=MARGIN_FRACTION,
                            leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
                            arm_moves=arm * a, trail_moves=trail * a)
                        led = r.ledger
                        v = float(led["trade_return"].mean() * 1e4) if len(led) else np.nan
                        acc.setdefault((key, wname), []).append(v)
    return acc


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    out = {"holdout_touched": False, "horizon_bars": HORIZON_BARS,
           "n_entries": N_ENTRIES, "B_reps": B_REPS, "seed": SEED,
           "cost_bp": 10.0, "assets": {}}

    for asset, path in ASSETS.items():
        kl = load(path)
        med_atr = float(np.nanmedian(kl["atr_pct"])) * 1e4
        cost_ratio = 10.0 / med_atr if med_atr > 0 else float("nan")
        log("")
        log(f"################ {asset} ################")
        log(f"  klines {len(kl):,}행 | ATR 중앙값 {med_atr:.1f}bp | "
            f"**비용/ATR {cost_ratio*100:.1f}%**")

        acc = run(kl, rng)
        cells = []
        keys = sorted({k for (k, w) in acc})
        for k in keys:
            v = float(np.nanmean(acc[(k, "val")]))
            o_ = float(np.nanmean(acc[(k, "oos")]))
            cells.append({"sl": k[0], "arm": k[1], "trail": k[2], "val_bp": v, "oos_bp": o_,
                          "min_bp": min(v, o_)})
        best = max(cells, key=lambda c: c["min_bp"])
        pos = [c for c in cells if c["val_bp"] > 0 and c["oos_bp"] > 0]
        log(f"  96셀 중 VAL·OOS 동시양수 **{len(pos)}**")
        log(f"  ⭐최선 셀 SL={best['sl']} ARM={best['arm']} Trail={best['trail']}: "
            f"VAL {best['val_bp']:+.2f} / OOS {best['oos_bp']:+.2f}bp")
        arr = np.array([c["min_bp"] for c in cells])
        log(f"  격자 전체 min(VAL,OOS) 분포: 중앙값 {np.median(arr):+.2f} / "
            f"최대 {arr.max():+.2f} / 최소 {arr.min():+.2f}bp")
        verdict = "ETH형(양수)" if best["min_bp"] > 0 else "BTC형(음수)"
        log(f"  ⇒ 무작위 기준선 판정: **{verdict}**")
        out["assets"][asset] = {"median_atr_bp": med_atr, "cost_over_atr": cost_ratio,
                                "n_cells_both_positive": len(pos), "best_cell": best,
                                "grid_median_min_bp": float(np.median(arr)),
                                "grid_max_min_bp": float(arr.max()),
                                "verdict": verdict, "cells": cells}

    log("")
    log("=== 종합 (무작위 진입만으로 비용을 넘는가) ===")
    for a, v in out["assets"].items():
        b = v["best_cell"]
        log(f"  {a}  비용/ATR {v['cost_over_atr']*100:5.1f}%  동시양수 {v['n_cells_both_positive']:>2}/96  "
            f"최선 {b['val_bp']:+6.2f}/{b['oos_bp']:+6.2f}bp  → {v['verdict']}")
    out["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({out['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
