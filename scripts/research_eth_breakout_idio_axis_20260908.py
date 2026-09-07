#!/usr/bin/env python3
"""⭐**고유 움직임 축** -- 모델 없는 상태 서술 (2026-09-08).

v4 절제가 가리킨 결론: 69피쳐 중 **BTC 동조 3개가 거의 전부**다.
  · `BTC동조만`(5피쳐): VAL 53.34[51.05]✅ OOS 53.79[51.42]✅ HOLD 52.02[50.07]
  · `기준−BTC동조`(66피쳐): VAL 50.92 OOS 53.06 HOLD 51.01 -- 세 창 전부 ❌ (무너진다)
  · 테이프 53피쳐·OBI 6피쳐는 넣으면 **떨어진다**(전체 0.5199 < 기준 0.5349)

⇒ 가설: **ETH 혼자 움직였으면 되돌림, 시장이 같이 움직였으면 지속.**
   앞서 찾은 '속도' 축과도 이어진다 -- 빠른 고유 움직임 = 임팩트성 = 되돌림.

여기서는 모델을 쓰지 않고 **분위 표**로 낸다(사용자 요청: 예측이 아니라 상태 서술).
`mv_idio_atr` = (ETH 진행폭 − BTC 진행폭)/ATR, 발현 방향 정렬. 전부 `s1-1` 까지의 정보다.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
ALLW = ("TRAIN", "VAL", "OOS", "HOLDOUT_SPENT")
SEED = 20260908
BOOT = 4000
NULLB = 500


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8 or len(v) < 30: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    d = pd.read_parquet(MY / "dataset_v4.parquet"); d["timestamp"] = pd.to_datetime(d["timestamp"])
    y = d["y"].to_numpy(int); sp = d["split"].to_numpy()
    day = d["timestamp"].dt.floor("D").to_numpy()
    print("=" * 100)
    print("1) 축별 5분위 돌파율 (분위 경계는 ⚠️TRAIN 에서만 결정)")
    print("=" * 100)
    for f, nm in (("v2_mv_idio_atr", "고유 움직임 (ETH−BTC)/ATR"),
                  ("v2_mv_btc_ret_atr", "BTC 동반 수익/ATR"),
                  ("v2_mv_same_sign", "BTC 같은 방향(0/1)"),
                  ("f_speed", "발현 속도"),
                  ("atr_at_anchor", "ATR 수준")):
        if f not in d.columns: continue
        v = d[f].to_numpy(float)
        if len(np.unique(v[np.isfinite(v)])) < 5:
            T = pd.DataFrame({w: pd.DataFrame({"b": v[sp == w], "y": y[sp == w]})
                              .groupby("b")["y"].mean() for w in ALLW})
        else:
            qs = np.nanquantile(v[sp == "TRAIN"], np.linspace(0, 1, 6))[1:-1]
            b = np.searchsorted(qs, v)
            T = pd.DataFrame({w: pd.DataFrame({"b": b[sp == w], "y": y[sp == w]})
                              .groupby("b")["y"].mean() for w in ALLW})
        same = ((T - 0.5) > 0).all(1) | ((T - 0.5) < 0).all(1)
        print(f"\n   {nm}")
        print("      Q  " + " ".join(f"{int(i)+1:>7}" for i in T.index))
        for w in ALLW:
            print(f"      {w[:5]:>5} " + " ".join(f"{T.loc[i, w]:.3f}  " for i in T.index))
        print(f"      네창 부호일치 {int(same.sum())}/{len(T)} · 일치분위 {[int(i)+1 for i in T.index[same]]}")

    print("\n" + "=" * 100)
    print("2) ⭐고유 움직임 상위 구간의 상태 서술 (임계값 TRAIN 결정 · 커버리지 · 무작위 대조군)")
    print("=" * 100)
    v = d["v2_mv_idio_atr"].to_numpy(float)
    for pct in (60, 70, 80, 90):
        thr = np.nanpercentile(v[sp == "TRAIN"], pct)
        m0 = v >= thr
        line = f"   고유≥p{pct} | "
        for w in ALLW:
            m = m0 & (sp == w) & np.isfinite(v)
            if m.sum() < 80: line += f"{w[:4]} -- | "; continue
            r = y[m].mean(); lo, hi = day_ci(y[m].astype(float), day[m], rng)
            idx = np.flatnonzero((sp == w) & np.isfinite(v))
            nl = np.array([y[rng.choice(idx, m.sum(), replace=False)].mean() for _ in range(NULLB)])
            p = max((nl <= r).mean(), 1 / NULLB)
            line += (f"{w[:4]} 되돌림 {1-r:.4f}[{1-hi:.4f},{1-lo:.4f}] "
                     f"커버 {m.sum()/max((sp==w).sum(),1):.0%} p={p:.3f}{'✅' if p<0.05 else '❌'} | ")
        print(line, flush=True)

    print("\n" + "=" * 100)
    print("3) ⭐고유 × 속도 결합 (둘 다 상위 = 빠른 고유 움직임)")
    print("=" * 100)
    s = d["f_speed"].to_numpy(float)
    ok = np.isfinite(v) & np.isfinite(s)
    tv = np.nanpercentile(v[(sp == "TRAIN") & ok], 70); tsq = np.nanpercentile(s[(sp == "TRAIN") & ok], 70)
    for nm, m0 in (("고유↑ & 속도↑", ok & (v >= tv) & (s >= tsq)),
                   ("고유↑ & 속도↓", ok & (v >= tv) & (s < tsq)),
                   ("고유↓ & 속도↑", ok & (v < tv) & (s >= tsq)),
                   ("고유↓ & 속도↓", ok & (v < tv) & (s < tsq))):
        line = f"   {nm:>14} | "
        for w in ALLW:
            m = m0 & (sp == w)
            if m.sum() < 60: line += f"{w[:4]} -- | "; continue
            r = y[m].mean(); lo, hi = day_ci(y[m].astype(float), day[m], rng)
            line += f"{w[:4]} 되돌림 {1-r:.4f}[{1-hi:.4f},{1-lo:.4f}] n{m.sum():>5} | "
        print(line, flush=True)
    print(json.dumps({"done": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
