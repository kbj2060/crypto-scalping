#!/usr/bin/env python3
"""증거신호 칩 + **부분 증거금(20%) · 물타기** 규약 시뮬레이션 (2026-09-05, 사용자 실사용 파라미터).

사용자: *"그래서 돈을 100%다 넣지 않아. 20%씩 넣고 너무 떨어지면 물타기를 하고 있어."*

앞 두 측정과의 차이:
  · `..._displayed_convention_pnl_...`  : 칩 표시 익절가(k×ATR)까지 보유 -> gross ≈ 0, 손실 = 수수료
  · `..._tight_tp_high_leverage_...`    : 고정 TP/SL · 30배 · **전액 증거금** -> 90셀 전부 음수
  · 이 스크립트                          : **20%씩 분할 투입 + 물타기**, 경로 의존 시뮬레이션

⭐**왜 새로 재야 하나.** 사이징은 기대값의 *부호*를 못 바꾸지만(선형 스케일), **물타기는 다르다** --
경로 의존적으로 진입가를 낮춰 승률을 더 올리고 왼쪽 꼬리를 두껍게 만든다. 가격이 그 구간에서
평균회귀한다면 물타기는 실제로 양의 기대값을 가질 수 있다. 그러니 단정하지 말고 측정한다.

레버리지·증거금 계약 (CLAUDE.md Futures Risk Sizing Contract):
    notional = margin_fraction × leverage        (트랜치당 0.20 × 30 = equity의 6.0배)
    PnL(계좌) = price_move × notional
    트랜치 청산: 평균 진입가 대비 역행이 margin/notional = 1/leverage = 3.33%에 도달할 때
      -- 물타기로 트랜치를 더해도 배율이 같으면 이 거리는 **평균 진입가 기준으로 리셋**된다.
      (증거금 M, 명목 N일 때 청산 거리 = M/N; 같은 배율로 더하면 M/N 불변)
    수수료: 각 트랜치의 명목에 왕복 10bp -> 물타기는 수수료를 트랜치 수만큼 곱한다.

규약:
  · 모집단·진입은 앞 스크립트와 동일(칩 라이브 결정 모집단, open[i+1])
  · 익절 = **평균 진입가** ± tp_bp (가격) 도달 -> 전량 청산
  · 물타기 = 평균 진입가 대비 역행이 add_bp 도달하면 트랜치 1개 추가(최대 max_adds회)
  · 청산 = 평균 진입가 대비 역행 3.33% 도달 -> 투입 증거금 **전액** 손실
  · 만기 = hold봉에서 종가 청산
  · 봉내 순서는 **비관**: 같은 봉에서 익절과 (물타기/청산)이 겹치면 불리한 쪽 먼저

같은 측면 무작위 귀무 동반. 일군집(하루) 부트스트랩 CI. HOLDOUT(≥2026-04-01) 미접촉.
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

TRAIN_END, VAL_END, OOS_END = (pd.Timestamp(x) for x in ("2025-09-01", "2026-01-01", "2026-04-01"))
COST_BP, LEVERAGE, TRANCHE = 10.0, 30.0, 0.20        # 트랜치당 증거금 = equity의 20%
LIQ_BP = 1e4 / LEVERAGE                               # 333.3bp 역행 -> 투입 증거금 전액 손실
B_BOOT, SEED = 600, 20260905
OUT = ROOT / "data/research/eth_chip_partial_margin_averaging_down_20260905"

TP_GRID = [16.7, 33.3]                                # 가격 bp
ADD_GRID = [100.0, 150.0, 200.0]                      # 역행 1.0% / 1.5% / 2.0%에서 물타기
MAX_ADDS = [0, 2, 4]                                  # 0 = 물타기 없음(대조), 4 = 총 5트랜치 = 100%
HOLD = 72                                             # 6시간


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def log(m: str) -> None:
    print(f"[avgdown] {m}", flush=True)


_META = _load("meta_live3", "scripts/live_evidence_signal_metalabel_20260829.py")
CHIP = {n: int(c["horizon_bars"]) for n, c in _META.METALABEL_SIGNALS.items()}


def first_fire_positions(fired: np.ndarray, horizon: int) -> np.ndarray:
    keep, last = [], -10**9
    for i in np.flatnonzero(fired):
        if i - last > horizon:
            keep.append(i)
        last = i
    return np.asarray(keep, dtype=int)


def simulate(o, h, l, c, pos, sgn, tp_bp, add_bp, max_adds, hold, n):
    """경로 의존 물타기 시뮬레이션. 반환 (account_pct, outcome, n_tranches).

    account_pct = 계좌 대비 % (트랜치당 증거금 TRANCHE, 배율 LEVERAGE, 수수료는 트랜치별 명목에).
    outcome: 'tp' | 'liq' | 'timeout'
    """
    if pos + 1 >= n:
        return np.nan, "", 0
    entry = o[pos + 1]
    avg, k = entry, 1                                   # 평균 진입가, 트랜치 수
    end = min(pos + 1 + hold, n - 1)
    for j in range(pos + 1, end + 1):
        hi, lo = h[j], l[j]
        adverse = (avg - lo) / avg if sgn > 0 else (hi - avg) / avg     # 최대 역행폭(비율)
        favor = (hi - avg) / avg if sgn > 0 else (avg - lo) / avg
        # ── 비관 순서: 불리한 사건(청산 -> 물타기)을 먼저 판정한다 ──
        if adverse * 1e4 >= LIQ_BP:
            return -TRANCHE * k * 100.0, "liq", k       # 투입 증거금 전액 손실
        while k <= max_adds and adverse * 1e4 >= add_bp:
            add_px = avg * (1 - sgn * add_bp / 1e4)     # 물타기 체결가(트리거 레벨에서 체결 가정)
            avg = (avg * k + add_px) / (k + 1)
            k += 1
            adverse = (avg - lo) / avg if sgn > 0 else (hi - avg) / avg
            if adverse * 1e4 >= LIQ_BP:
                return -TRANCHE * k * 100.0, "liq", k
        if favor * 1e4 >= tp_bp:
            gross = tp_bp
            return ((gross * k - COST_BP * k) / 1e4 * TRANCHE * LEVERAGE * 100.0), "tp", k
    move = sgn * (c[end] / avg - 1) * 1e4
    return ((move * k - COST_BP * k) / 1e4 * TRANCHE * LEVERAGE * 100.0), "timeout", k


def day_ci(vals, days, rng, b=B_BOOT):
    u = np.unique(days)
    if len(u) < 2:
        return np.nan, np.nan
    by = {d: vals[days == d] for d in u}
    m = np.empty(b)
    for i in range(b):
        m[i] = np.concatenate([by[d] for d in rng.choice(u, size=len(u), replace=True)]).mean()
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    log("신호 프레임 재구성...")
    _s1 = _load("s1_avg", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
    _s1.VAL_END = OOS_END
    sig, _f, _e = _s1.build_sig()
    ts = pd.to_datetime(sig["timestamp"]).dt.tz_localize(None)
    m = (ts < OOS_END).to_numpy()
    sig, ts = sig.loc[m].reset_index(drop=True), ts.loc[m].reset_index(drop=True)
    o, h, l, c = (sig[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    n = len(sig)
    day = ts.dt.floor("D").to_numpy()
    split = np.where(ts < TRAIN_END, "TRAIN", np.where(ts < VAL_END, "VAL", "OOS"))
    log(f"  {n:,}봉 · 트랜치 {TRANCHE:.0%} × {LEVERAGE:.0f}배 = 명목 {TRANCHE*LEVERAGE:.1f}×equity "
        f"· 트랜치 청산 역행 {LIQ_BP:.1f}bp")

    P, S = [], []
    for name, hz in CHIP.items():
        for side in ("bottom", "top"):
            col = f"{side}_{name}"
            if col not in sig.columns:
                continue
            for pos in first_fire_positions(sig[col].fillna(False).to_numpy(bool), hz):
                P.append(int(pos)); S.append(1.0 if side == "bottom" else -1.0)
    P, S = np.asarray(P), np.asarray(S)
    ok = P + 1 + HOLD < n
    P, S = P[ok], S[ok]
    log(f"  칩 발동 {len(P):,}건")

    long_frac = float((S > 0).mean())
    pool = np.flatnonzero(np.arange(n) + 1 + HOLD < n)
    Pn = rng.choice(pool, size=min(20000, len(pool)), replace=True)
    Sn = np.where(rng.random(len(Pn)) < long_frac, 1.0, -1.0)

    report = {"tranche_fraction": TRANCHE, "leverage": LEVERAGE, "notional_x_equity": TRANCHE * LEVERAGE,
              "liq_bp": round(LIQ_BP, 1), "cost_bp": COST_BP, "hold_bars": HOLD,
              "intrabar": "pessimistic (liq/add before tp)", "holdout_touched": False, "cells": []}

    log(f"\n=== 칩 발동 + 20% 트랜치 · 30배 · 물타기 (계좌 % · 비관 봉내순서) ===")
    log(f"{'TP':>6s} {'물타기':>7s} {'추가':>4s} {'창':6s} {'n':>6s} {'승률':>6s} {'계좌%/건':>9s} "
        f"{'일CI':>18s} {'전손률':>7s} {'평균트랜치':>9s} {'무작위':>8s}")
    for tp in TP_GRID:
        for max_add in MAX_ADDS:
            for add in (ADD_GRID if max_add else [ADD_GRID[0]]):
                res = np.array([simulate(o, h, l, c, p, s, tp, add, max_add, HOLD, n)[:3]
                                for p, s in zip(P, S)], dtype=object)
                v = res[:, 0].astype(float); oc = res[:, 1].astype(str); kk = res[:, 2].astype(int)
                nres = np.array([simulate(o, h, l, c, p, s, tp, add, max_add, HOLD, n)[:1]
                                 for p, s in zip(Pn, Sn)], dtype=object)
                nv = nres[:, 0].astype(float)
                nspl = split[Pn]
                for sp in ("TRAIN", "VAL", "OOS"):
                    msk = split[P] == sp
                    if msk.sum() < 30:
                        continue
                    vv, dd = v[msk], day[P][msk]
                    lo, hi = day_ci(vv, dd, rng)
                    nm = nv[nspl == sp]
                    log(f"{tp:6.1f} {add:7.0f} {max_add:4d} {sp:6s} {int(msk.sum()):6,d} "
                        f"{(oc[msk]=='tp').mean():6.1%} {vv.mean():+9.3f} [{lo:+8.3f},{hi:+8.3f}] "
                        f"{(oc[msk]=='liq').mean():7.2%} {kk[msk].mean():9.2f} "
                        f"{(nm.mean() if len(nm) else float('nan')):+8.3f}")
                    report["cells"].append({
                        "tp_bp": tp, "add_bp": add, "max_adds": max_add, "split": sp,
                        "n": int(msk.sum()), "win_rate": round(float((oc[msk] == "tp").mean()), 4),
                        "account_pct": round(float(vv.mean()), 4),
                        "ci_account_pct": [round(lo, 4), round(hi, 4)],
                        "ruin_rate": round(float((oc[msk] == "liq").mean()), 4),
                        "mean_tranches": round(float(kk[msk].mean()), 3),
                        "null_account_pct": (round(float(nm.mean()), 4) if len(nm) else None)})
            log("")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    log(f"산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
