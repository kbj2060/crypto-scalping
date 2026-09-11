#!/usr/bin/env python3
"""사용자 제안 검정: **"신호 봉의 저가/고가에 다음 봉에 산다"** 로 기준가 문제를 풀 수 있는가.

2026-09-10 판정([[feedback_label_reference_price_must_be_reachable_20260910]])은 "저가/고가는
못 사는 값"이었다. 사용자 제안은 그걸 **지정가 주문**으로 우회한다 -- 봉 t 의 저가에 지정가를
걸어 두고 봉 t+1 에 체결되기를 기다린다.

⭐측정 대상은 두 가지다.
  (1) **체결률**: 봉 t+1 이 정말 low[t] 까지 내려오는가.
  (2) **체결 조건부 라벨률**: 체결된 표본의 라벨이 안 체결된 표본과 같은가.
      -- 체결이 '신호가 틀렸을 때만' 일어나면 표본 자체가 역선택이라 재학습이 무의미하다.

라벨은 **배포본 정의를 그대로** 옮긴다(새로 발명하지 않는다):
  극점  build_eth_extreme_detector_week_20260909.py:94-96  W=12
        바닥: min(low[t+1 .. t+12]) >= low[t]
  V자   research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py::realized_outcome
        fast_mult = (max close[t+1..t+6] - extreme) / atr[t-1] >= 1.5
        giveback  = (peak - end) / (peak - extreme) <= 0.20,  peak = max high[t+1..t+12]

대조군으로 **종가 기준**(항상 체결 가능)을 같이 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KL = Path("/home/kbj20/crypto-scalping/binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv")
W_EXT, FAST, FULL, ATR_MULT, T_SUSTAIN = 12, 6, 12, 1.5, 0.20


def fwd_min(x: np.ndarray, k: int) -> np.ndarray:
    """[t+1 .. t+k] 의 최소. 뒤집어 rolling 하면 파이썬 루프 없이 같은 값이 나온다."""
    r = pd.Series(x[::-1]).rolling(k, min_periods=k).min().to_numpy()[::-1]
    out = np.full(len(x), np.nan)
    out[:-k] = r[1:len(x) - k + 1]
    return out


def fwd_max(x: np.ndarray, k: int) -> np.ndarray:
    r = pd.Series(x[::-1]).rolling(k, min_periods=k).max().to_numpy()[::-1]
    out = np.full(len(x), np.nan)
    out[:-k] = r[1:len(x) - k + 1]
    return out


def rate(mask: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    m = mask & np.isfinite(y)
    return int(m.sum()), (float(y[m].mean()) if m.sum() else float("nan"))


def main() -> int:
    d = pd.read_csv(KL, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    d = d.dropna().sort_values("timestamp").reset_index(drop=True)
    hi, lo, cl = (d[c].to_numpy(float) for c in ("high", "low", "close"))
    n = len(d)
    print(f"5분봉 {n:,} · {d.timestamp.iloc[0]:%Y-%m-%d} ~ {d.timestamp.iloc[-1]:%Y-%m-%d}")

    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - np.roll(cl, 1)), np.abs(lo - np.roll(cl, 1))))
    tr[0] = hi[0] - lo[0]
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().shift(1).to_numpy()   # 사건 **직전** ATR

    fmin12, fmax12 = fwd_min(lo, W_EXT), fwd_max(hi, W_EXT)
    nxt_lo, nxt_hi = np.roll(lo, -1), np.roll(hi, -1)
    nxt_lo[-1] = nxt_hi[-1] = np.nan

    # ── 극점(바닥) ─────────────────────────────────────────────────────────────
    y_ext_low = (fmin12 >= lo).astype(float); y_ext_low[~np.isfinite(fmin12)] = np.nan
    y_ext_cls = (fmin12 >= cl).astype(float); y_ext_cls[~np.isfinite(fmin12)] = np.nan
    fill = nxt_lo <= lo                       # 사용자 제안: low[t] 지정가, t+1 에 체결
    ok = np.isfinite(y_ext_low) & np.isfinite(nxt_lo)

    print("\n══ 극점 탐지기(바닥, W=12) — 사용자 제안 「low[t] 지정가, t+1 체결」 ══")
    nf, rf = rate(ok & fill, y_ext_low)
    nu, ru = rate(ok & ~fill, y_ext_low)
    print(f"  체결률 {ok.sum() and fill[ok].mean():.4f}  (체결 {nf:,} / 미체결 {nu:,})")
    print(f"  🔴체결된 표본의 라벨률  {rf:.6f}   ({int(rf*nf)}건)")
    print(f"    미체결 표본의 라벨률  {ru:.4f}")
    print(f"  ⭐항등식 확인: 체결 ⇒ low[t+1] <= low[t] ⇒ min(low[t+1..t+12]) <= low[t] ⇒ 라벨=0")
    print(f"    (동점 tie 만 예외: {int(rf*nf)}건)")

    print("\n══ 대조군: 종가 기준(항상 체결 가능) ══")
    nc, rc = rate(np.isfinite(y_ext_cls), y_ext_cls)
    print(f"  기저 라벨률 {rc:.4f}  (n={nc:,})   ← 저가 기준 기저 {rate(np.isfinite(y_ext_low), y_ext_low)[1]:.4f}")

    # ── V자반등(바닥쪽) ─────────────────────────────────────────────────────────
    fmax_cl6 = fwd_max(cl, FAST)
    end = np.roll(cl, -FULL); end[-FULL:] = np.nan
    for tag, anchor in (("저가 기준(배포본)", lo), ("종가 기준(대조군)", cl)):
        fast_mult = (fmax_cl6 - anchor) / atr
        denom = fmax12 - anchor
        with np.errstate(invalid="ignore", divide="ignore"):
            give = np.where(np.abs(denom) < 1e-12, np.nan, (fmax12 - end) / denom)
        y = ((fast_mult >= ATR_MULT) & np.isfinite(give) & (give <= T_SUSTAIN)).astype(float)
        y[~(np.isfinite(fast_mult) & np.isfinite(fmax12) & np.isfinite(end))] = np.nan
        if anchor is lo:
            okv = np.isfinite(y) & np.isfinite(nxt_lo)
            nf, rf = rate(okv & fill, y); nu, ru = rate(okv & ~fill, y)
            print(f"\n══ V자반등 {tag} — 같은 제안 적용 ══")
            print(f"  체결률 {fill[okv].mean():.4f}  (체결 {nf:,} / 미체결 {nu:,})")
            print(f"  🔴체결된 표본 라벨률 {rf:.4f}   미체결 {ru:.4f}   비 {rf/ru if ru else float('nan'):.3f}x")
        else:
            nc2, rc2 = rate(np.isfinite(y), y)
            print(f"\n══ V자반등 {tag} ══\n  기저 라벨률 {rc2:.4f} (n={nc2:,})")

    # ── 2차 질문(사용자): "매매는 논외, '이 봉이 극점이고 반등락이 온다'는 신호로만 쓴다면?" ──
    # 그러면 체결 가능성 논점은 **빠진다**. 대신 남는 질문은 하나다:
    # 배포 라벨은 "안 깨질 저점"만 요구하고 **반등 크기를 전혀 요구하지 않는다**.
    # 그래서 라벨=1 이 실제로 얼마나 되튀는지를 잰다. 되튀지 않으면 "반등락이 온다"는 읽기가
    # 라벨에 근거가 없다. ⚠️저가 기준 되튐은 라벨과 기계적으로 얽히므로(라벨이 전방 저가를
    # low[t] 위로 묶는다) **종가 기준 되튐**을 같이 본다 -- 사람이 화면 보고 반응하는 시점이다.
    print("\n\n══ '극점 = 반등락 온다' 가 라벨에 있는가 (바닥쪽) ══")
    reb_lo = (fmax12 - lo) / lo * 100.0
    reb_cl = (fmax12 - cl) / cl * 100.0
    okr = np.isfinite(y_ext_low) & np.isfinite(reb_lo) & np.isfinite(reb_cl)
    lab1, lab0 = okr & (y_ext_low == 1), okr & (y_ext_low == 0)
    print(f"  {'':22s} {'중앙':>8} {'상위25%':>8} {'>=0.2%':>8} {'>=0.5%':>8} {'>=1.0%':>8}")
    for nm, m, x in (("라벨=1 · 저가기준", lab1, reb_lo), ("라벨=0 · 저가기준", lab0, reb_lo),
                     ("라벨=1 · **종가기준**", lab1, reb_cl), ("라벨=0 · 종가기준", lab0, reb_cl),
                     ("전체(무조건) · 종가", okr, reb_cl)):
        v = x[m]
        print(f"  {nm:22s} {np.median(v):7.3f}% {np.percentile(v,75):7.3f}% "
              f"{(v>=0.2).mean():7.1%} {(v>=0.5).mean():7.1%} {(v>=1.0).mean():7.1%}")
    # 변동성 통제: 되튐을 그 시점 ATR 로 나눠 같은 국면끼리 비교한다.
    atr_pct = atr / np.maximum(cl, 1e-9) * 100.0
    okn = okr & np.isfinite(atr_pct) & (atr_pct > 0)
    for nm, m in (("라벨=1", okn & (y_ext_low == 1)), ("라벨=0", okn & (y_ext_low == 0))):
        v = (reb_cl / atr_pct)[m]
        print(f"  ATR 배수(종가기준) {nm}: 중앙 {np.median(v):.2f}배 · 상위25% {np.percentile(v,75):.2f}배")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
