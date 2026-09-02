#!/usr/bin/env python3
"""XRP 증거신호 5종 **트레일링스톱 경제성게이트** -- ETH/BTC와 동일 설계.

## 왜 이걸 돌리는가

XRP 증거신호 5종은 분류(AUC)만 끝났고 **경제성 검증이 전혀 없다**. 대시보드에 뜨는 확률은
분류 확률일 뿐, 왕복 10bp 후에 돈이 되는지는 한 번도 측정하지 않았다.
ETH는 6종, BTC는 3종이 이 게이트를 통과했다. 같은 잣대를 XRP에 대야 "매매 근거로 쓸 수
있는가"를 답할 수 있다.

## 설계 (`gate_btc_evidence_signals_trailing_economics_20260902.py` 그대로)

  · 96셀 그리드: SL[1.5~4.0] x ARM[0.5~2.0] x Trail[0.1~0.5]
  · 표준 왕복비용 **10bp** (수수료 우대 가정 금지)
  · margin 0.30 / leverage 3.0 -> notional 0.90 (Futures Risk Sizing Contract)
  · 신호 자체 H를 보유한도로 사용 (라벨이 "H봉 내 움직임"을 주장하므로 그 정의와 일치)
  · `purged_decision_mask`로 구간 경계 누수 제거, `simulate_single_position`으로 회계 통일

⭐**방향뒤집기 대조군을 96셀 전량에 적용한다.** 통과한 셀만 검사하면 오판한다.
⭐**ARM<1.0은 노이즈 수확 아티팩트**로 따로 집계한다.

⚠️**타임스탬프 매핑 강제.** BTC에서 저장된 `pos`를 klines 인덱스로 그대로 써서 108봉(9시간)
어긋난 사고가 났다. 여기서는 `searchsorted` + 완전일치 검증을 넣는다.

⚠️**자산 오염 가드.** XRP 동결컨텍스트 빌드에서 `prep[0]`이 BTC CSV를 읽은 사고가 있었다
(str_z/taker/orthogonal은 BTC 모듈을 재사용한다). 로더를 호출하지 않고 XRP CSV를 직접 읽으며,
행수가 XRP 기대치와 다르면 죽는다.

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
    "xrpctx", ROOT / "scripts/build_xrp_evidence_signal_frozen_contexts_20260903.py")
_ctxmod = importlib.util.module_from_spec(_CTX)
_CTX.loader.exec_module(_ctxmod)
SIGNALS = _ctxmod.SIGNALS
GRID_CHOSEN = _ctxmod.GRID_CHOSEN
CAND_CSV = _ctxmod.CAND_CSV
EXPECTED_ROWS = _ctxmod.EXPECTED_ROWS
TZ_AWARE = _ctxmod.TZ_AWARE
load_mod = _ctxmod.load_mod

KLINES = ROOT / "binance_data/klines/XRPUSDT/XRPUSDT-5m-api.csv"
OUT = ROOT / "data/research/xrp_evidence_signals_costgate_20260903/report.json"

MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.001
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ARM_GRID = [0.5, 1.0, 1.5, 2.0]
TRAIL_GRID = [0.1, 0.2, 0.3, 0.5]

# 신호별 보유한도 = 그 신호의 라벨 H (XRP 전용 그리드스크린 확정값)
# ⚠️giveback 계열(taker)은 해상에 2xH가 필요하지만 경제적 주장은 H봉 내 움직임이므로 H를 쓴다
# (BTC liquidity_sweep H=20/해상40에서와 동일한 처리).
HORIZON = {"demarker_extreme": 2, "kalman_deviation_meanrev": 5,
           "short_term_return_z": 12, "taker_delta_climax": 9,
           "orthogonal_combo": 8}


def log(m): print(f"[xrp-gate] {m}", flush=True)


def build_fires(name, rel, builder, prep, kind):
    """동결 컨텍스트 빌더(`build_xrp_evidence_signal_frozen_contexts_20260903.py`)의 호출
    규약을 **그대로** 복제한다.

    ⚠️`prep[0]`(load_tier0/load_frame)은 절대 호출하지 않는다 -- 그 모듈 자신의 TIER0_PATH를
    읽어 BTC 데이터가 들어온다. XRP 후보 CSV를 직접 읽고 변환 prep만 태운다."""
    mod = load_mod(rel)
    tz_aware = TZ_AWARE.get(name, False)
    f = pd.read_csv(CAND_CSV)
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
    if not tz_aware:
        f["timestamp"] = f["timestamp"].dt.tz_localize(None)
    for pname in prep[1:]:
        fnp = getattr(mod, pname, None)
        if fnp is not None:
            f = fnp(f)
    if "timestamp" in f.columns:
        f["timestamp"] = pd.to_datetime(f["timestamp"])
    # ⭐자산 오염 가드 -- 다른 자산 CSV를 읽으면 여기서 죽는다
    if abs(len(f) - EXPECTED_ROWS) > 200:
        raise RuntimeError(f"{name}: 행수 {len(f):,} != XRP 기대치 {EXPECTED_ROWS:,} "
                           f"-- 다른 자산 데이터를 읽었을 가능성")

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

    # ⚠️저장된 `pos`는 **빌더가 받은 프레임**의 행 인덱스다. klines 인덱스로 그대로 쓰면 안 된다.
    # (BTC에서 오프셋 108봉=9시간으로 모든 진입이 어긋난 사고. XRP는 오프셋 0이지만 그래도 강제.)
    kl_ts = ts.to_numpy()
    f_ts = pd.to_datetime(fires["timestamp"]).to_numpy()
    dec = np.searchsorted(kl_ts, f_ts)
    inb = dec < len(kl_ts)
    if not inb.all():
        fires, f_ts, dec = fires.loc[inb].reset_index(drop=True), f_ts[inb], dec[inb]
    bad = int((kl_ts[dec] != f_ts).sum())
    if bad:
        raise ValueError(f"fires 타임스탬프가 klines에 없다: {bad}/{len(dec)}건")
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
    log(f"XRP 5m klines {len(kl):,}행 ({kl.timestamp.min()} ~ {kl.timestamp.max()})")
    log(f"그리드 {len(SL_GRID)*len(ARM_GRID)*len(TRAIL_GRID)}셀 x 정방향/뒤집기 x VAL/OOS")
    log("⚠️HOLDOUT 미터치")

    rep = {"asset": "XRPUSDT", "cost_bp": 10.0, "margin_fraction": MARGIN_FRACTION,
           "leverage": LEVERAGE, "grid": {"sl": SL_GRID, "arm": ARM_GRID, "trail": TRAIL_GRID},
           "holdout_touched": False,
           "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
           "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
           "signals": {}}

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
            log(f"     VAL {best['val_fwd_bp']:+.2f}bp (뒤 {best['val_flip_bp']:+.2f}) "
                f"n={best['val_n']} 승률 {best['val_wr']*100:.1f}%")
            log(f"     OOS {best['oos_fwd_bp']:+.2f}bp (뒤 {best['oos_flip_bp']:+.2f}) "
                f"n={best['oos_n']} 승률 {best['oos_wr']*100:.1f}%")
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
            log(f"  {k:<26} ⚠️{v['error'][:60]}")
        else:
            log(f"  {k:<26} 동시양수 {v['n_passing_96']:>2}/96  진짜 {v['n_genuine']:>2}  "
                f"ARM>=1.0 진짜 **{v['n_genuine_arm_ge_1']}**")
    log(f"  ⇒ 통과 신호: {ok if ok else '없음'}")
    rep["passed_signals"] = ok
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
