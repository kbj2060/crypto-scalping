#!/usr/bin/env python3
"""⭐ATR 상태 서술의 **견고성 감사** (2026-09-08).

발견: `first_fire T=1.0` 에서 **ATR≥p80 이면 되돌림률 55.6/54.3/55.4/63.4%**(네 창), 임계값에
대해 단조(p60 51.9 → p90 58.0). 커버 20~28%. 그러나 네 창 CI 상한<50% 는 0/16(VAL 0.5011,
OOS 0.5014 로 0.001 차이).

이 저장소는 이런 경계선 결과를 여러 번 다음 날 폐기했다. 폐기 사유가 될 만한 것을 전부 먼저 친다.
1. **날짜 클러스터링** -- 고ATR 사건은 소수의 고변동성 날에 몰린다. 독립 일수와 상위일 비중.
2. **상위 5%/1% 날 제거** 후에도 유지되는가.
3. **동일 커버리지 무작위 대조군** -- 같은 비율의 무작위 부분집합은 50% 인가(B=500).
4. **라벨 정의 견고성** -- 배리어 P 를 바꿔도(0.5% 고정 외) 같은 방향인가.
5. **속도 축과의 중복** -- 이미 알려진 "발현이 빠를수록 되돌림"의 다른 표현일 뿐인가
   (ATR 통제 후 속도, 속도 통제 후 ATR).
6. **연도별 안정성**.
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
BOOT = 4000
NULLB = 500


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8 or len(v) < 20: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    A = pd.read_parquet(SRC); A["timestamp"] = pd.to_datetime(A["timestamp"])
    d = A[(A.anchor == "first_fire") & (A.T_mult == 1.0)].reset_index(drop=True)
    v = d["atr_at_anchor"].to_numpy(float); y = d["y"].to_numpy(int)
    sp = d["split"].to_numpy(); ts = pd.to_datetime(d["timestamp"])
    day = ts.dt.floor("D").to_numpy()
    thr = np.nanpercentile(v[sp == "TRAIN"], 80)
    m0 = v >= thr
    print(f"임계 ATR p80(TRAIN 기준) = {thr:.5f} · 전체 {len(d):,} · 해당 {m0.sum():,}\n", flush=True)

    print("=" * 104)
    print("1·2) 날짜 클러스터링과 상위일 제거")
    print("=" * 104)
    for w in ALLW:
        m = m0 & (sp == w)
        if m.sum() < 80: print(f"   {w:>14}: n={m.sum()} -- 표본 부족"); continue
        dd = day[m]; u, cnt = np.unique(dd, return_counts=True)
        top5 = np.sort(cnt)[::-1][:max(1, len(u) // 20)].sum() / m.sum()
        # 상위 5% 날 제거
        keep_days = set(u[np.argsort(cnt)[::-1][max(1, len(u) // 20):]])
        k5 = np.array([x in keep_days for x in dd])
        r_all = y[m].mean(); r5 = y[m][k5].mean() if k5.sum() > 50 else np.nan
        print(f"   {w:>14}: n={m.sum():>5} 독립일 {len(u):>4} · 하루평균 {m.sum()/len(u):>4.1f}건 · "
              f"상위5%일 비중 {top5:>5.1%} · 돌파율 {r_all:.4f} → 상위5%일 제거 {r5:.4f}")

    print("\n" + "=" * 104)
    print("3) 동일 커버리지 무작위 대조군 (B=500) -- 같은 비율의 무작위 부분집합")
    print("=" * 104)
    for w in ALLW:
        m = m0 & (sp == w); mw = sp == w
        if m.sum() < 80: continue
        obs = y[m].mean()
        idx = np.flatnonzero(mw); k = m.sum()
        nl = np.array([y[rng.choice(idx, k, replace=False)].mean() for _ in range(NULLB)])
        p = max((nl <= obs).mean(), 1 / NULLB)
        print(f"   {w:>14}: 관측 돌파율 {obs:.4f} · 무작위 평균 {nl.mean():.4f} sd {nl.std():.4f} · "
              f"p(무작위≤관측)={p:.4f} {'✅' if p < 0.05 else '❌'}")

    print("\n" + "=" * 104)
    print("4) 라벨 견고성 -- 다른 배리어 P (v3 데이터셋, OBS=5)")
    print("=" * 104)
    v3 = ROOT / "tmp/eth_breakout_reversal_20260908/dataset_v3.parquet"
    if v3.exists():
        B3 = pd.read_parquet(v3)
        e = B3[(B3["anchor"] == "first_fire") & (B3["OBS"] == 5)].reset_index(drop=True)
        v3v = e["atr_at_anchor"].to_numpy(float); s3 = e["split"].to_numpy()
        t3 = np.nanpercentile(v3v[s3 == "TRAIN"], 80); mm = v3v >= t3
        for Pl in (25, 35, 50):
            yy = e[f"y_p{Pl}"].to_numpy(int)
            line = f"   P={Pl/100:.2f}% | "
            for w in ALLW:
                m = mm & (s3 == w)
                if m.sum() < 80: line += f"{w[:4]} -- | "; continue
                line += f"{w[:4]} 돌파 {yy[m].mean():.4f} (n{m.sum()}) | "
            print(line)
    print("   ⚠️v3 는 OBS=5 라 기준가가 현재가로 이동한 설정(부록 AN) -- 방향 확인용 참고치")

    print("\n" + "=" * 104)
    print("5) 속도 축과의 중복 -- 서로를 통제하면 남는가")
    print("=" * 104)
    sp_ = d["f_speed"].to_numpy(float) if "f_speed" in d.columns else None
    if sp_ is None:
        sp_ = d["T_atr"].to_numpy(float) / np.maximum(d["trig_min"].to_numpy(float) + 1, 1)
    ok = np.isfinite(v) & np.isfinite(sp_)
    sq = pd.qcut(pd.Series(sp_[ok]), 4, labels=False, duplicates="drop").to_numpy()
    aq = pd.qcut(pd.Series(v[ok]), 4, labels=False, duplicates="drop").to_numpy()
    G = pd.DataFrame({"a": aq, "s": sq, "y": y[ok]}).groupby(["a", "s"])["y"].agg(["mean", "count"])
    piv = G["mean"].unstack()
    print("   행=ATR 4분위, 열=속도 4분위, 값=돌파율")
    print(piv.round(3).to_string())
    print(f"   ATR 최상위행 평균 {piv.iloc[-1].mean():.3f} · 속도 최상위열 평균 {piv.iloc[:, -1].mean():.3f}")
    print(f"   ⇒ ATR 통제 후 속도 효과 {piv.iloc[:, -1].mean() - piv.iloc[:, 0].mean():+.3f} · "
          f"속도 통제 후 ATR 효과 {piv.iloc[-1].mean() - piv.iloc[0].mean():+.3f}")

    print("\n" + "=" * 104)
    print("6) 연도별 안정성 (ATR≥p80)")
    print("=" * 104)
    yr = pd.DataFrame({"y": y[m0], "yr": ts[m0].dt.year, "q": ts[m0].dt.to_period("Q").astype(str)})
    print("   연도: " + " · ".join(f"{k} {g['y'].mean():.3f}(n{len(g)})"
                                  for k, g in yr.groupby("yr")))
    qq = yr.groupby("q")["y"].agg(["mean", "count"])
    print(f"   분기 {len(qq)}개 중 50% 미만 {int((qq['mean']<0.5).sum())}개 · "
          + " ".join(f"{i}:{r['mean']:.3f}" for i, r in qq.iterrows()))
    print(json.dumps({"thr": float(thr)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
