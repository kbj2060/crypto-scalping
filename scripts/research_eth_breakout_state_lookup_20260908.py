#!/usr/bin/env python3
"""⭐**상태 조회(state lookup)** -- "지금은 되돌림 구간"을 현재 봉만으로 말할 수 있는가 (2026-09-08).

사용자: *"지금은 역사적으로 봤을 때 되돌리지 않는 구간이라는 걸 미래 봉이 아닌 현재 봉을 분석하는거지"*

## 앞 단계에서 걸러진 것
- 라벨 **자기상관은 없다**(후행 돌파율 상관 −0.012~+0.025, 규칙 정확도 세 창 일관 실패).
  ⇒ *"최근에 되돌림이 많았으니 지금도"* 는 성립하지 않는다.
- 레짐 버킷 5축 중 4축은 네 창 부호가 흩어진다.
- ⭐**남은 하나: ATR 수준.** 최상위 4분위 돌파율이 TRAIN/VAL/OOS/HOLDOUT
  **0.468 / 0.480 / 0.454 / 0.440** -- 네 창 전부 50% 미만이고 값도 안정적이다.

## 이 스크립트
ATR(및 결합 축)을 십분위까지 쪼개 **네 창 전부에서 안정적인 상태 서술**이 되는지 본다.
판정 기준을 예측 정확도가 아니라 **상태 진술의 재현성**으로 둔다:
  (1) 네 창 모두 같은 방향(50% 기준) (2) 일군집 CI 가 50% 를 배제 (3) 커버리지 명시
  (4) 십분위에 대해 단조 (5) 슬라이딩 임계값에서 부호가 안 뒤집힘
⚠️커버리지를 반드시 같이 낸다 -- 상위 10% 만 맞으면 하루 몇 건인지가 실제 유용성이다.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp/eth_breakout_reversal_20260908/dataset_v2.parquet"
ALLW = ("TRAIN", "VAL", "OOS", "HOLDOUT_SPENT")
SEED = 20260908
BOOT = 3000


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8 or len(v) < 20: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    A = pd.read_parquet(SRC); A["timestamp"] = pd.to_datetime(A["timestamp"])
    print("=" * 112)
    print("1) ATR 십분위별 돌파율 (앵커·트리거별, 창 = TRAIN/VAL/OOS/HOLDOUT)")
    print("=" * 112)
    for anch in ("first_fire", "any2/Wc3"):
        for TM in (0.75, 1.0):
            d = A[(A.anchor == anch) & (A.T_mult == TM)].reset_index(drop=True)
            if len(d) < 6000: continue
            v = d["atr_at_anchor"].to_numpy(float); y = d["y"].to_numpy(int); sp = d["split"].to_numpy()
            # ⚠️분위 경계는 TRAIN 에서만 정한다 (표본외 분위를 쓰면 그 자체가 미래참조)
            qs = np.nanquantile(v[sp == "TRAIN"], np.linspace(0, 1, 11))[1:-1]
            b = np.searchsorted(qs, v)
            T = pd.DataFrame({w: pd.DataFrame({"b": b[sp == w], "y": y[sp == w]})
                              .groupby("b")["y"].mean() for w in ALLW})
            same = ((T - 0.5) > 0).all(1) | ((T - 0.5) < 0).all(1)
            print(f"\n■ {anch} T={TM} n={len(d):,}")
            print("   D " + " ".join(f"{i+1:>6}" for i in range(len(T))))
            for w in ALLW:
                print(f"   {w[:5]:>5} " + " ".join(f"{T.loc[i, w]:.3f}" if i in T.index else "  --  "
                                                   for i in range(len(T))))
            print(f"   네창부호일치 {int(same.sum())}/{len(T)} · 일치 십분위 {[int(i)+1 for i in T.index[same]]}")

    print("\n" + "=" * 112)
    print("2) ⭐상위 ATR 구간의 상태 서술 -- 임계값 슬라이딩 · 커버리지 · CI")
    print("=" * 112)
    rows = []
    for anch in ("first_fire", "any2/Wc3"):
        for TM in (0.75, 1.0):
            d = A[(A.anchor == anch) & (A.T_mult == TM)].reset_index(drop=True)
            if len(d) < 6000: continue
            v = d["atr_at_anchor"].to_numpy(float); y = d["y"].to_numpy(int)
            sp = d["split"].to_numpy(); day = pd.to_datetime(d["timestamp"]).dt.floor("D").to_numpy()
            for pct in (60, 70, 80, 90):
                thr = np.nanpercentile(v[sp == "TRAIN"], pct)     # ⚠️TRAIN 에서만 임계값 결정
                m0 = v >= thr
                line = f"   {anch:>11} T={TM} ATR≥p{pct} | "
                rec = dict(anchor=anch, T=TM, pct=pct)
                for w in ALLW:
                    m = m0 & (sp == w)
                    if m.sum() < 100: line += f"{w[:4]} -- | "; continue
                    rate = y[m].mean()
                    lo, hi = day_ci(y[m].astype(float), day[m], rng)
                    cov = m.sum() / max((sp == w).sum(), 1)
                    line += (f"{w[:4]} 돌파 {rate:.4f}[{lo:.4f},{hi:.4f}] "
                             f"되돌림 {1-rate:.4f} 커버 {cov:.1%} n{m.sum():>4} | ")
                    rec[f"{w}_rate"] = float(rate); rec[f"{w}_hi"] = hi
                    rec[f"{w}_cov"] = float(cov); rec[f"{w}_n"] = int(m.sum())
                print(line, flush=True)
                rows.append(rec)
    R = pd.DataFrame(rows)
    R["pass4"] = [all(R.loc[i, f"{w}_hi"] < 0.5 for w in ALLW) for i in R.index]
    print("\n" + "=" * 112)
    print(f"⭐네 창 모두 **돌파율 CI 상한 < 50%** (= 되돌림 구간이라고 말할 수 있음): "
          f"{int(R.pass4.sum())}/{len(R)}")
    if R.pass4.any():
        print(R[R.pass4][["anchor", "T", "pct"] + [f"{w}_rate" for w in ALLW]
                         + [f"{w}_cov" for w in ALLW]].round(4).to_string(index=False))
    R["m"] = R[[f"{w}_rate" for w in ALLW]].max(1)
    print("\n=== 네 창 최대 돌파율이 가장 낮은 셀 상위 6 (= 가장 강한 되돌림 구간) ===")
    print(R.sort_values("m").head(6)[["anchor", "T", "pct"] + [f"{w}_rate" for w in ALLW]
                                     + [f"{w}_hi" for w in ALLW] + [f"{w}_cov" for w in ALLW]]
          .round(4).to_string(index=False))
    R.to_csv(ROOT / "tmp/eth_breakout_reversal_20260908/state_lookup.csv", index=False)
    print(json.dumps({"cells": len(R), "pass4": int(R.pass4.sum())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
