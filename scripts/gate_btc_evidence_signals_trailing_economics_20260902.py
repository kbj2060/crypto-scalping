#!/usr/bin/env python3
"""BTC 증거신호 7종 **트레일링스톱 경제성게이트** -- ETH와 동일 설계.

## 왜 이걸 돌리는가

BTC 증거신호 7종은 분류(AUC)만 끝났고 **경제성 검증이 전혀 없다**(관련 스크립트 0건).
ETH는 5종이 이 게이트를 통과해서 배포됐다. 같은 잣대를 BTC에 대야 "매매 근거로 쓸 수
있는가"를 답할 수 있다.

⚠️사전 기대는 낮다 -- BTC 5분봉은 비용/ATR이 62%(ETH 43%)로 훨씬 불리하고,
V자반등 경제라벨은 이 비용 구조 때문에 4/4 실패했다(`btc_evidence_signal_and_shadow_20260902.md`).

## 설계 (ETH `research_*_costgate_full_grid_flip_audit_20260901.py` 그대로)

  · 96셀 그리드: SL[1.5~4.0] x ARM[0.5~2.0] x Trail[0.1~0.5]
  · 표준 왕복비용 **10bp** (수수료 우대 가정 금지)
  · margin 0.30 / leverage 3.0 -> notional 0.90 (Futures Risk Sizing Contract)
  · 신호 자체 H를 보유한도로 사용 (라벨이 "H봉 내 움직임"을 주장하므로 그 정의와 일치)
  · `purged_decision_mask`로 구간 경계 누수 제거, `simulate_single_position`으로 회계 통일

⭐**방향뒤집기 대조군을 96셀 전량에 적용한다.** 통과한 셀만 검사하면 오판한다
(2026-09-01 fib_extension_exhaustion에서 통과 9개 중 진짜 0개였던 사례).
⭐**ARM<1.0은 노이즈 수확 아티팩트**로 따로 집계한다 -- 무장이 너무 빨라 트레일이 노이즈를
줍는 구조적 효과이지 신호의 공로가 아니다.

판정: VAL>0 AND OOS>0 AND (정방향>뒤집기) 를 VAL/OOS 양 구간에서 만족 = "진짜".

⚠️**HOLDOUT 미터치.** VAL+OOS만 본다. 통과 신호가 나오면 그때 1회성으로 노출한다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402

_CTX = importlib.util.spec_from_file_location(
    "btcctx", ROOT / "scripts/build_btc_evidence_signal_frozen_contexts_20260902.py")
_ctxmod = importlib.util.module_from_spec(_CTX)
_CTX.loader.exec_module(_ctxmod)
SIGNALS, GRID_CHOSEN, CAND_CSV = _ctxmod.SIGNALS, _ctxmod.GRID_CHOSEN, _ctxmod.CAND_CSV
load_mod = _ctxmod.load_mod

KLINES = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
OUT = ROOT / "data/research/btc_evidence_signals_costgate_20260902/report.json"

MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.001
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ARM_GRID = [0.5, 1.0, 1.5, 2.0]
TRAIL_GRID = [0.1, 0.2, 0.3, 0.5]

# 신호별 보유한도 = 그 신호의 라벨 H (BTC 전용 그리드스크린 확정값)
HORIZON = {"demarker_extreme": 8, "kalman_deviation_meanrev": 10, "liquidity_sweep": 20,
           "short_term_return_z": 6, "taker_delta_climax": 6,
           "fib_extension_exhaustion": 10, "orthogonal_combo": 8}


def log(m): print(f"[btc-gate] {m}", flush=True)


def build_fires(name, rel, builder, prep, kind):
    """동결 컨텍스트 빌더(`build_btc_evidence_signal_frozen_contexts_20260902.py` 104~139행)의
    호출 규약을 **그대로** 복제한다 -- 첫 prep은 무인자, 이후는 프레임을 넘긴다.
    ⚠️tz를 붙이지 않는다: demarker/taker가 naive `START`와 비교한다."""
    mod = load_mod(rel)
    f = None
    for pname in prep:
        fnp = getattr(mod, pname, None)
        if fnp is None:
            continue
        f = fnp() if f is None else fnp(f)
    if "timestamp" in f.columns:
        f["timestamp"] = pd.to_datetime(f["timestamp"])
    fn = getattr(mod, builder)
    if kind == "demarker":
        g = GRID_CHOSEN["demarker"]
        out = fn(f, g["horizon"], g["k"], mod.CLUSTER_GAP)
    elif kind == "kalman":
        g = GRID_CHOSEN["kalman"]
        f["kalman_dev_z"] = mod.compute_kalman_dev_z(f["close"].to_numpy())
        bt = (f["kalman_dev_z"] <= -2.0).fillna(False).to_numpy()
        tt = (f["kalman_dev_z"] >= 2.0).fillna(False).to_numpy()
        out = fn(f, bt, tt, g["horizon"], g["k"], mod.CLUSTER_GAP)
    else:
        out = fn(f)
    fires = out[0] if isinstance(out, tuple) else out
    return fires, f


def run_grid(klines, fires, horizon_bars):
    ts = klines["timestamp"]
    o, h, l, c = (klines[x].to_numpy() for x in ("open", "high", "low", "close"))

    # 컬럼 정규화 -- 스크립트마다 이름이 다르다
    if "pos" in fires.columns:
        dec = fires["pos"].to_numpy(dtype=np.int64)
    elif "bar_idx" in fires.columns:
        dec = fires["bar_idx"].to_numpy(dtype=np.int64)
    else:
        raise KeyError(f"pos/bar_idx 없음: {list(fires.columns)[:12]}")
    if "side" in fires.columns:
        is_long = (fires["side"].astype(str) == "bottom").to_numpy()
    elif "is_bottom" in fires.columns:
        is_long = (fires["is_bottom"].to_numpy() == 1)
    else:
        raise KeyError(f"side/is_bottom 없음: {list(fires.columns)[:12]}")
    atr = fires["atr_pct"].to_numpy(dtype=float)
    if not np.all(np.diff(dec) >= 0):                  # simulate_single_position 요구사항
        order = np.argsort(dec, kind="stable")
        dec, is_long, atr = dec[order], is_long[order], atr[order]

    masks = {}
    for wname, (s, e) in {"val": (VAL_START, OOS_START), "oos": (OOS_START, HOLDOUT_START)}.items():
        el = set(np.flatnonzero(purged_decision_mask(
            ts, start=s, end=e, horizon_bars=horizon_bars)).tolist())
        masks[wname] = np.array([d in el for d in dec])

    tp = np.full(len(fires), 999.0)
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
    log(f"BTC 5m klines {len(kl):,}행")
    log(f"그리드 {len(SL_GRID)*len(ARM_GRID)*len(TRAIL_GRID)}셀 x 정방향/뒤집기 x VAL/OOS")
    log("⚠️HOLDOUT 미터치")

    rep = {"asset": "BTCUSDT", "cost_bp": 10.0, "margin_fraction": MARGIN_FRACTION,
           "leverage": LEVERAGE, "grid": {"sl": SL_GRID, "arm": ARM_GRID, "trail": TRAIL_GRID},
           "holdout_touched": False, "signals": {}}

    for name, rel, builder, prep, kind in SIGNALS:
        log("")
        log(f"=== {name} (H={HORIZON[name]}) ===")
        try:
            fires, _ = build_fires(name, rel, builder, prep, kind)
        except Exception as e:                                     # noqa: BLE001
            log(f"  ⚠️fires 빌드 실패: {type(e).__name__}: {e}")
            rep["signals"][name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        fires["timestamp"] = pd.to_datetime(fires["timestamp"])
        if fires["timestamp"].dt.tz is not None:
            fires["timestamp"] = fires["timestamp"].dt.tz_localize(None)
        fires = fires.loc[fires["timestamp"] < HOLDOUT_START].reset_index(drop=True)
        log(f"  fires {len(fires):,}건 (HOLDOUT 이전)")

        cells, ns = run_grid(kl, fires, HORIZON[name])
        log(f"  후보 VAL {ns['val']} / OOS {ns['oos']}")

        passing = [c for c in cells if c["val_fwd_bp"] > 0 and c["oos_fwd_bp"] > 0]
        genuine = [c for c in passing
                   if c["val_fwd_bp"] > c["val_flip_bp"] and c["oos_fwd_bp"] > c["oos_flip_bp"]]
        gen_arm1 = [c for c in genuine if c["arm"] >= 1.0]
        log(f"  VAL+OOS 동시양수 {len(passing)}/96")
        log(f"  방향뒤집기 통과(진짜) {len(genuine)}/96  그중 ARM>=1.0 **{len(gen_arm1)}**")
        if gen_arm1:
            best = max(gen_arm1, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"]))
            log(f"  ⭐최선(ARM>=1.0): SL={best['sl']} ARM={best['arm']} Trail={best['trail']}")
            log(f"     VAL {best['val_fwd_bp']:+.2f}bp (뒤 {best['val_flip_bp']:+.2f}) n={best['val_n']} 승률 {best['val_wr']*100:.1f}%")
            log(f"     OOS {best['oos_fwd_bp']:+.2f}bp (뒤 {best['oos_flip_bp']:+.2f}) n={best['oos_n']} 승률 {best['oos_wr']*100:.1f}%")
        else:
            b = max(cells, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"]))
            log(f"  ❌ARM>=1.0 진짜 조합 없음. 격자 최선: SL={b['sl']} ARM={b['arm']} "
                f"Trail={b['trail']}  VAL {b['val_fwd_bp']:+.2f} / OOS {b['oos_fwd_bp']:+.2f}bp")
        rep["signals"][name] = {
            "horizon_bars": HORIZON[name], "n_fires": int(len(fires)),
            "n_candidates": ns, "n_passing_96": len(passing),
            "n_genuine": len(genuine), "n_genuine_arm_ge_1": len(gen_arm1),
            "genuine_arm_ge_1": gen_arm1, "cells": cells}

    ok = [k for k, v in rep["signals"].items() if v.get("n_genuine_arm_ge_1", 0) > 0]
    log("")
    log("=== 종합 ===")
    for k, v in rep["signals"].items():
        if "error" in v:
            log(f"  {k:<26} ⚠️{v['error'][:40]}")
        else:
            log(f"  {k:<26} 동시양수 {v['n_passing_96']:>2}/96  진짜 {v['n_genuine']:>2}  "
                f"ARM>=1.0 진짜 **{v['n_genuine_arm_ge_1']}**")
    log(f"  ⇒ 통과 신호: {ok if ok else '없음'}")
    rep["passed_signals"] = ok
    rep["runtime_sec"] = round(time.time()-t0, 1)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
