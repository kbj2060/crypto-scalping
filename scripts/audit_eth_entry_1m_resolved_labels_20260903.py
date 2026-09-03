#!/usr/bin/env python3
"""전수조사 ②: **1분봉으로 체결 봉 안의 순서를 해상한 라벨** (2026-09-03).

`audit_eth_entry_intrabar_fill_bar_credit_20260903.py`가 L0(체결봉 전부 크레딧)와
L1(체결봉 전부 배제)의 격차를 −29~−81bp로 재고, 전체 후보 PF가 3.66 → 1.07로 무너지는 것을
보였다. 그러나 **둘 다 극단**이다 -- 체결 후 그 5분봉에 남은 시간의 움직임은 정당히 우리 것이다.

여기서는 1분봉으로 정확히 가른다:
  체결 5분봉 f를 1분봉 5개로 쪼개고, **지정가에 처음 닿은 1분봉**을 체결 시점으로 잡는다.
  그 1분봉의 나머지 + 이후 1분봉들에서 (high, low)를 모아 **체결 봉의 진짜 사후 폭**을 만든다.
  이후 봉(f+1~)은 5분봉 그대로 둔다 -- 청산 로직 자체는 5분봉 설계이므로 바꾸지 않는다.
  ⚠️이렇게 하면 바뀌는 건 **체결 봉의 (high, low) 한 쌍**뿐이고, 그것이 정확히 쟁점이다.

산출 라벨:
  L0 현행 · L1 정직(f+1) · **L3 1분해상**(체결 시점 이후만) · 그리고 체결 1분봉의 위치 분포
"""
from __future__ import annotations

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
M1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
OUT = ROOT / "tmp/eth_entry_1m_resolved_20260903"


def log(m): print(f"[audit2] {m}", flush=True)


def trail(side, e, a, hi, lo, cl):
    if side > 0:
        stop = e * (1 - SL * a); peak = e; armed = False
        for k in range(len(cl)):
            if lo[k] <= stop: return stop / e - 1.0
            if hi[k] > peak:
                peak = hi[k]
                if not armed and (peak - e) / e >= ARM * a: armed = True
                if armed: stop = max(stop, peak * (1 - TRAIL * a))
        return cl[-1] / e - 1.0
    stop = e * (1 + SL * a); peak = e; armed = False
    for k in range(len(cl)):
        if hi[k] >= stop: return 1.0 - stop / e
        if lo[k] < peak:
            peak = lo[k]
            if not armed and (e - peak) / e >= ARM * a: armed = True
            if armed: stop = min(stop, peak * (1 + TRAIL * a))
    return 1.0 - cl[-1] / e


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    kl = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values(
        "timestamp").reset_index(drop=True)
    h5, l5, c5 = (kl[x].to_numpy(float) for x in ("high", "low", "close"))
    ts5 = pd.DatetimeIndex(kl["timestamp"]); n5 = len(kl)

    m1 = pd.read_csv(M1, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    m1 = m1.drop_duplicates("timestamp")
    log(f"5분봉 {n5:,} · 1분봉 {len(m1):,} ({m1.timestamp.min()} ~ {m1.timestamp.max()})")
    # 1분봉을 5분봉 시작시각에 매핑
    m1["b5"] = m1["timestamp"].dt.floor("5min")
    g = {k: v for k, v in zip(m1["b5"].to_numpy(), np.arange(len(m1)))}   # 미사용(참고)
    m1h, m1l = m1["high"].to_numpy(float), m1["low"].to_numpy(float)
    # 5분봉 시작시각 -> 1분봉 시작 인덱스
    first = m1.groupby("b5", sort=True).apply(lambda d: d.index[0])
    cnt = m1.groupby("b5", sort=True).size()
    idx0 = {k: int(v) for k, v in first.items()}
    idxn = {k: int(v) for k, v in cnt.items()}

    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    ALL = "--all" in sys.argv          # 전체 깊이/대기 (원 설계의 학습 모집단)
    W = D.reset_index(drop=True) if ALL else \
        D[((D.depth == 3.0) & (D.btf <= 6)).to_numpy()].reset_index(drop=True)
    fi = W.fi.to_numpy().astype(int); ei = W.ei.to_numpy().astype(int)
    e = W.lim.to_numpy(float); a = W.atr_pct.to_numpy(float); sd = W.sd.to_numpy(int)
    log(f"후보 팔 {len(W):,}")

    y0, y1, y3 = [], [], []
    pos_in_bar, no1m = [], 0
    for i in range(len(W)):
        f, hz = int(fi[i]), int(ei[i] - fi[i])
        # L0 / L1
        y0.append(trail(sd[i], e[i], a[i], h5[f:f+hz], l5[f:f+hz], c5[f:f+hz]) * NOTIONAL - COST*NOTIONAL)
        if hz > 1:
            y1.append(trail(sd[i], e[i], a[i], h5[f+1:f+hz], l5[f+1:f+hz], c5[f+1:f+hz]) * NOTIONAL - COST*NOTIONAL)
        else:
            y1.append(np.nan)
        # L3: 1분봉으로 체결 시점 특정
        bt = ts5[f]
        if bt not in idx0:
            y3.append(np.nan); pos_in_bar.append(np.nan); no1m += 1; continue
        s0, nn = idx0[bt], idxn[bt]
        sub_h, sub_l = m1h[s0:s0+nn], m1l[s0:s0+nn]
        hit = np.flatnonzero(sub_l <= e[i]) if sd[i] > 0 else np.flatnonzero(sub_h >= e[i])
        if not len(hit):
            y3.append(np.nan); pos_in_bar.append(np.nan); continue
        k0 = int(hit[0])
        pos_in_bar.append(k0)
        # 체결 1분봉 이후(그 분 포함하되 체결 이후만 알 수 없으므로 **그 다음 분부터**)
        post_h = sub_h[k0+1:]; post_l = sub_l[k0+1:]
        if len(post_h):
            fh, fl = float(post_h.max()), float(post_l.min())
        else:
            fh = fl = float(e[i])            # 봉 마지막 분에 체결 -> 사후 폭 없음
        H = np.concatenate([[fh], h5[f+1:f+hz]])
        L = np.concatenate([[fl], l5[f+1:f+hz]])
        C = np.concatenate([[c5[f]], c5[f+1:f+hz]])
        y3.append(trail(sd[i], e[i], a[i], H, L, C) * NOTIONAL - COST*NOTIONAL)

    W["y_L0"], W["y_L1"], W["y_L3"] = y0, y1, y3
    W["fill_minute"] = pos_in_bar
    pib = np.array([p for p in pos_in_bar if np.isfinite(p)])
    log(f"1분봉 미커버 {no1m:,}건 · 체결 분 위치 분포(0~4): "
        + " ".join(f"{k}:{int((pib==k).sum())}" for k in range(5)))
    log(f"⭐체결이 봉의 마지막 2분(3,4)에서 일어난 비율 {(pib>=3).mean()*100:.1f}% "
        f"-- 그만큼 사후 폭이 거의 없다")

    print(f"\n=== 전체 후보 팔 (bp, 비용 후) ===")
    print(f"{'라벨':24s}{'n':>7s}{'평균':>9s}{'중앙':>9s}{'승률':>8s}{'PF':>8s}")
    for tag, col in (("L0 현행(체결봉 전부)", "y_L0"), ("L3 ⭐1분해상", "y_L3"),
                     ("L1 정직(f+1)", "y_L1")):
        b = W[col].to_numpy() * 1e4; b = b[np.isfinite(b)]
        w_ = b > 0
        pf = b[w_].sum() / -b[~w_].sum() if (~w_).any() else np.inf
        print(f"{tag:24s}{len(b):7d}{b.mean():+9.2f}{np.median(b):+9.2f}{w_.mean()*100:7.1f}%{pf:8.2f}")

    tp = PERF / "trades.csv"
    if tp.exists() and not ALL:
        T = pd.read_csv(tp, parse_dates=["timestamp"])
        key = W.set_index(["timestamp", "signal", "arm"])
        print(f"\n=== v2가 고른 트레이드 (창별, bp) ===")
        print(f"{'창':10s}{'n':>5s}{'L0':>10s}{'L3 ⭐':>10s}{'L1':>10s}{'L3 승률':>10s}")
        for wn in ("VAL", "OOS", "HOLDOUT"):
            t = T[T.split == wn]
            m = key.reindex(list(zip(t.timestamp, t.signal, t.arm)))
            b0, b3, b1 = (m[c].to_numpy()*1e4 for c in ("y_L0", "y_L3", "y_L1"))
            ok = np.isfinite(b0) & np.isfinite(b3)
            print(f"{wn:10s}{int(ok.sum()):5d}{np.nanmean(b0[ok]):+10.2f}"
                  f"{np.nanmean(b3[ok]):+10.2f}{np.nanmean(b1[ok]):+10.2f}"
                  f"{(b3[ok]>0).mean()*100:9.1f}%")
    out_name = "labels_1m_all.csv" if ALL else "labels_1m.csv"
    W[["timestamp","signal","side","arm","sd","depth","btf","split","atr_pct","fi","ei",
       "fill_minute","y","y_L0","y_L1","y_L3"]].to_csv(OUT/out_name, index=False)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
