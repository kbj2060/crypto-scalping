#!/usr/bin/env python3
"""⚠️전수조사 ①: **체결 봉 자체의 유리한 폭을 청산에 크레딧하는가** (2026-09-03).

의심: `trail_out(side, e, a, h[f:f+H], l[f:f+H], c[f:f+H])`가 **체결 봉 f부터** 평가한다.
bottom 신호의 매수 지정가는 직전 종가보다 3 ATR 아래이므로, 체결 봉의 **고가**는 대개
지정가보다 약 3 ATR 위다. 그러면 k=0에서 곧바로

    peak = hi[0] ≈ e*(1+3a) → (peak-e)/e = 3a ≥ ARM*a → 즉시 무장
    stop = peak*(1-TRAIL*a) ≈ e*(1+2.9a)   ← 진입가보다 2.9 ATR 위

가 되어 **진입 순간 2.9 ATR 이익이 잠긴다.** ATR 0.28%면 2.9*0.28%*0.9 ≈ 73bp로,
관측된 평균(+68bp)과 거의 일치한다.

⚠️봉 안의 순서를 우리는 모른다. 그리고 **가격이 내려와서 체결된 것이므로 고가가 저가보다
먼저 왔을 가능성이 오히려 높다** -- 그 고가 시점엔 포지션이 없었다. 크레딧하면 미래참조다.

이 스크립트가 재는 것 (모델·선택과 무관하게 **라벨 자체**를 본다):
  L0 현행     -- 체결 봉 f부터 (h[f:], l[f:], c[f:])
  L1 정직     -- 체결 봉 다음 f+1부터. 체결 봉의 어떤 폭도 크레딧하지 않는다
  L2 부분정직 -- f부터 보되 **k=0의 유리한 갱신만 금지**(불리한 스톱 판정은 유지) = 비관 브래킷
그리고 "k=0에서 즉시 무장되는 비율"을 직접 센다.

⚠️v1(HGB)과 v2(TabPFN)가 **같은 y로 학습·평가**됐으므로, 여기서 격차가 크면 진입 모델 계보
전체가 영향을 받는다.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research_eth_regime_gated_costgate_ensemble_20260902 import KLINES_PATH  # noqa: E402
from research_eth_entry_direction_oracle_ceiling_20260903 import (  # noqa: E402
    SL, ARM, TRAIL, NOTIONAL, COST)

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
PERF = ROOT / "tmp/eth_entry_v2_performance_20260903"
OUT = ROOT / "tmp/eth_entry_intrabar_audit_20260903"


def log(m): print(f"[audit1] {m}", flush=True)


def trail(side, e, a, hi, lo, cl, skip_first_favorable=False):
    """현행 `trail_out`과 동일하되, skip_first_favorable이면 k=0의 유리한 갱신을 막는다."""
    if side > 0:
        stop = e * (1 - SL * a); peak = e; armed = False
        for k in range(len(cl)):
            if lo[k] <= stop:
                return stop / e - 1.0, k, armed
            if not (skip_first_favorable and k == 0):
                if hi[k] > peak:
                    peak = hi[k]
                    if not armed and (peak - e) / e >= ARM * a:
                        armed = True
                    if armed:
                        stop = max(stop, peak * (1 - TRAIL * a))
        return cl[-1] / e - 1.0, len(cl) - 1, armed
    stop = e * (1 + SL * a); peak = e; armed = False
    for k in range(len(cl)):
        if hi[k] >= stop:
            return 1.0 - stop / e, k, armed
        if not (skip_first_favorable and k == 0):
            if lo[k] < peak:
                peak = lo[k]
                if not armed and (e - peak) / e >= ARM * a:
                    armed = True
                if armed:
                    stop = min(stop, peak * (1 + TRAIL * a))
    return 1.0 - cl[-1] / e, len(cl) - 1, armed


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    kl = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values(
        "timestamp").reset_index(drop=True)
    h, l, c = (kl[x].to_numpy(float) for x in ("high", "low", "close"))
    n = len(kl)
    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    sel = ((D.depth == 3.0) & (D.btf <= 6)).to_numpy()
    W = D[sel].reset_index(drop=True)
    hz = (W.ei - W.fi).to_numpy().astype(int)
    log(f"후보 팔 {len(W):,} (depth3.0/wait6)")

    # ⭐체결 봉의 고가가 진입가 대비 몇 ATR 위인가 (bottom 매수 기준)
    fi = W.fi.to_numpy().astype(int); e = W.lim.to_numpy(float)
    a = W.atr_pct.to_numpy(float); sd = W.sd.to_numpy(int)
    fav0 = np.where(sd > 0, (h[fi] - e) / e, (e - l[fi]) / e) / a      # ATR 단위
    log(f"⭐체결 봉의 유리한 폭 (ATR): 중앙 {np.median(fav0):.2f} · "
        f"평균 {fav0.mean():.2f} · ARM({ARM}) 이상 비율 **{(fav0 >= ARM).mean()*100:.1f}%**")

    rows = {}
    for tag, start, skip in (("L0 현행(체결봉 포함)", 0, False),
                             ("L1 정직(f+1부터)", 1, False),
                             ("L2 비관(f, 유리갱신 금지)", 0, True)):
        ys, arm0 = [], []
        for i in range(len(W)):
            f0 = fi[i] + start
            hh = hz[i] - start
            if hh <= 0 or f0 + hh > n:
                ys.append(np.nan); arm0.append(False); continue
            mv, k, armed = trail(sd[i], e[i], a[i], h[f0:f0 + hh], l[f0:f0 + hh],
                                 c[f0:f0 + hh], skip_first_favorable=skip)
            ys.append(mv * NOTIONAL - COST * NOTIONAL)
            arm0.append(k == 0)
        rows[tag] = np.array(ys, float)
        log(f"  {tag:26s} 완료 (첫 봉에서 청산 {np.mean(arm0)*100:.1f}%)")

    W["y_L0"], W["y_L1"], W["y_L2"] = rows["L0 현행(체결봉 포함)"], \
        rows["L1 정직(f+1부터)"], rows["L2 비관(f, 유리갱신 금지)"]
    # 원본 y와 L0 재현 일치 확인
    d = np.nanmax(np.abs(W["y_L0"] - W["y"]))
    log(f"\n⭐L0가 원본 y를 재현: max|Δ| = {d:.3e} {'✅' if d < 1e-12 else '❌ 재현 실패'}")

    print(f"\n=== 전체 후보 팔 {len(W):,}개 (bp, 비용 후) ===")
    print(f"{'라벨':26s}{'평균':>9s}{'중앙':>9s}{'승률':>8s}{'PF':>8s}")
    for tag in rows:
        b = W[{"L0 현행(체결봉 포함)": "y_L0", "L1 정직(f+1부터)": "y_L1",
               "L2 비관(f, 유리갱신 금지)": "y_L2"}[tag]].to_numpy() * 1e4
        b = b[np.isfinite(b)]
        w_ = b > 0
        pf = b[w_].sum() / -b[~w_].sum() if (~w_).any() else np.inf
        print(f"{tag:26s}{b.mean():+9.2f}{np.median(b):+9.2f}{w_.mean()*100:7.1f}%{pf:8.2f}")

    # v2가 실제로 고른 트레이드에서의 격차
    tp = PERF / "trades.csv"
    if tp.exists():
        T = pd.read_csv(tp, parse_dates=["timestamp"])
        key = W.set_index(["timestamp", "signal", "arm"])
        print(f"\n=== v2가 고른 트레이드 (창별, bp) ===")
        print(f"{'창':10s}{'n':>5s}{'L0 현행':>10s}{'L1 정직':>10s}{'격차':>9s}"
              f"{'L0 승률':>9s}{'L1 승률':>9s}")
        for wn in ("VAL", "OOS", "HOLDOUT"):
            t = T[T.split == wn]
            m = key.reindex(list(zip(t.timestamp, t.signal, t.arm)))
            b0 = m["y_L0"].to_numpy() * 1e4; b1 = m["y_L1"].to_numpy() * 1e4
            ok = np.isfinite(b0) & np.isfinite(b1)
            b0, b1 = b0[ok], b1[ok]
            print(f"{wn:10s}{len(b0):5d}{b0.mean():+10.2f}{b1.mean():+10.2f}"
                  f"{b1.mean()-b0.mean():+9.2f}{(b0>0).mean()*100:8.1f}%{(b1>0).mean()*100:8.1f}%")
    W[["timestamp", "signal", "side", "arm", "sd", "split", "atr_pct", "fi", "ei",
       "y", "y_L0", "y_L1", "y_L2"]].to_csv(OUT / "labels.csv", index=False)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
