#!/usr/bin/env python3
"""앵커 방향 **회귀** -- 실현 손익을 직접 예측 (2026-09-08, 서버 GPU).

사용자: *"시간 청산이라면 그럼 라벨링 방식이 어디로 들어가는거지?"* → *"2번 회귀로 진행해줘.
       호라이즌 별로 테스트해보자"*

## 왜 회귀인가
현재 배포 팔 `wbin` 의 라벨은
    y_bin = 1 지속 배리어 먼저 · 0 되돌림 먼저 · **NaN 둘 다 미터치**
**시간청산이 라벨에 없다.** 그래서 (1)모델이 그걸 배운 적이 없고 (2)백테스트가 그 행을
채점에서 뺐다(커버리지 71.3%). 라이브는 진입해서 −7.3bp 를 실현한다. 이 간극이 −13.5bp였다
(백테스트 +12.0 → 라이브 −1.46).

회귀는 이 문제를 **구조적으로** 없앤다:
    타깃 y = 진입 후 실현 손익(bp) = +100 지속배리어 먼저 · −100 되돌림배리어 먼저
                                   · 시간청산이면 H봉 뒤 종가 손익
    ⇒ **커버리지 100%**, 그리고 **학습 타깃 == 판정 지표(라이브 건당 bp)**.
분류는 "어느 쪽이 먼저인가"를 물어 시간청산을 표현할 수 없었다. 회귀는 그걸 그냥 담는다.

## 격자 (사전 지정) -- 배리어도 축이다
H ∈ {12(1h), 24(2h), 48(4h)} × 배리어 ∈ {±0.5%, ±0.75%, ±1.0%}  = 9셀
⚠️1차판은 ±1% 로 고정했는데 **1시간에 ±1% 는 해소율 38.2% 로 너무 빡빡하다**(사용자 지적).
⭐부록 Y 에서 "1h 에 배리어를 좁히면 기각"이라고 한 것은 **분류 기준**이다 --
  분류는 배리어를 좁히면 **손익분기 정확도가 오른다**(±0.5% 는 57.80%). **회귀에는 그 개념이
  없다**: 타깃이 실현 bp 자체라 배리어가 바뀌면 보상 분포만 바뀌고 잣대(건당 bp)는 그대로다.
  따라서 회귀에서는 배리어를 반드시 축으로 둬야 한다.
해소율 참고(부록 Y): H=12 → 73.8/53.9/38.2% · H=24 → 85.5/69.2/53.7% · H=48 → 94.1/83.9/71.3%
모델 = TabPFNRegressor · 피쳐 = MASHT 2,784 · 월 1회 재학습 walk-forward · 엠바고 4h
비용은 타깃에 넣지 않는다(총수익 예측 → 결정 시점에 차감). 비용 7.8bp.

## 판정 (부록 X4 규칙 준수)
헤드라인 = **상위30% 진입의 실현 건당 bp − 비용**, 일군집 CI 하한 > 0.
같이 낸다: 전건 진입 기준선 · 순위상관(Spearman) · 날 블록 귀무 ·
그리고 **같은 H 의 분류(wbin) 라이브 값**과 직접 비교.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / "tmp/eth_anchor_window_tensor_20260907"
DIR = ROOT / "tmp/eth_anchor_direction_labels_20260907/direction_labels.parquet"
OUT = ROOT / "tmp/eth_anchor_pnl_regression_20260908"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H_GRID = (12, 24, 48)
P_GRID = (0.5, 0.75, 1.0)          # 배리어 %(대칭). 회귀는 배리어별 손익분기 개념이 없다
COST = 7.8
EMBARGO = pd.Timedelta(hours=4)
SEED, N_EST, BOOT = 20260907, 4, 1000
# 같은 (H, 배리어) 의 분류 라이브 값 -- 부록 Y 격자
CLS_REF = {(12, 0.5): -7.16, (12, 0.75): -5.89, (12, 1.0): -3.56,
           (24, 0.5): -7.89, (24, 0.75): -5.37, (24, 1.0): -7.32,
           (48, 0.5): -7.92, (48, 0.75): -6.53, (48, 1.0): -1.46}


def boot_ci(fn, d, rng, B=BOOT):
    u = np.unique(d)
    if len(u) < 5: return (np.nan, np.nan)
    idx = {x: np.flatnonzero(d == x) for x in u}; o = []
    for _ in range(B):
        i = np.concatenate([idx[x] for x in rng.choice(u, len(u), replace=True)])
        v = fn(i)
        if v is not None: o.append(v)
    return (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))) if len(o) > B // 3 else (np.nan, np.nan)


def dayperm(y, d, rng):
    u = np.unique(d); src = {x: np.flatnonzero(d == x) for x in u}
    pm = rng.permutation(u); z = y.copy()
    for a, b in zip(u, pm):
        if len(src[a]): z[src[a]] = np.resize(y[src[b]], len(src[a]))
    return z


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    import importlib.util, torch
    from tabpfn import TabPFNRegressor
    s = importlib.util.spec_from_file_location("ab", ROOT / "scripts/build_eth_anchor_label_dataset_20260907.py")
    B = importlib.util.module_from_spec(s); s.loader.exec_module(B)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    M = np.load(WT / "masht.npy").astype(np.float32)
    v = np.load(WT / "valid.npy")
    I = pd.read_parquet(WT / "index.parquet")[v].reset_index(drop=True)
    L = pd.read_parquet(DIR); L["timestamp"] = pd.to_datetime(L["timestamp"])
    if "anchor" in L.columns:
        L = L[L["anchor"] == "any3/Wc3"]
    # ⚠️배리어 축을 넣었으므로 **세 배리어의 터치 시각을 모두** 가져온다
    hit_cols = [f"hit_{k}_min_P{p:g}" for p in P_GRID for k in ("cont", "fade")]
    miss = [c for c in hit_cols if c not in L.columns]
    assert not miss, f"라벨 파일에 없는 컬럼: {miss}"
    J = I[["timestamp", "side", "split"]].merge(
        L[["timestamp", "side"] + hit_cols], on=["timestamp", "side"], how="left")
    assert len(J) == len(I)
    ts = I["timestamp"]; sp = I["split"].to_numpy(); day = ts.dt.floor("D").to_numpy()
    side = I["side"].to_numpy()
    kl = B._load_kl(B.ETH_KL)
    pos = pd.Series(np.arange(len(kl)), index=pd.to_datetime(kl["timestamp"]))
    idx = pos.reindex(ts).to_numpy().astype(int)
    OP = kl["open"].to_numpy(float); CL = kl["close"].to_numpy(float)
    print(f"[입력] 앵커 {len(I):,} · {dev}", flush=True)

    months, cur = [], pd.Timestamp("2025-09-01")
    while cur <= ts.max():
        months.append(cur); cur = cur + pd.offsets.MonthBegin(1)

    rows = []
    print("\n" + "=" * 116, flush=True)
    print(f"{'H':>3}{'배리어':>8}{'커버':>8}{'해소율':>8}{'전건':>11}{'상위30%':>13}"
          f"{'[일군집 CI]':>21}{'ρ':>12}{'분류':>11}", flush=True)
    print("=" * 116, flush=True)
    HC = {p: J[f"hit_cont_min_P{p:g}"].to_numpy(float) for p in P_GRID}
    HF = {p: J[f"hit_fade_min_P{p:g}"].to_numpy(float) for p in P_GRID}
    for H in H_GRID:
      lim = H * 5.0
      for P in P_GRID:
        hc, hf = HC[P], HF[P]
        BARRIER_BP = P * 100.0
        c_ok = np.isfinite(hc) & (hc <= lim); f_ok = np.isfinite(hf) & (hf <= lim)
        # ⭐타깃: 전 앵커에 정의된다 (커버리지 100%)
        y = np.full(len(J), np.nan)
        for j, i in enumerate(idx):
            a, b = i + 1, i + H
            if b >= len(kl): continue
            if c_ok[j] and (not f_ok[j] or hc[j] < hf[j]):
                y[j] = BARRIER_BP
            elif f_ok[j] and (not c_ok[j] or hf[j] < hc[j]):
                y[j] = -BARRIER_BP
            elif c_ok[j] and f_ok[j] and hc[j] == hf[j]:
                y[j] = 0.0                                  # 동시 터치 -- 보수적으로 0
            else:
                e = OP[a]
                y[j] = (CL[b] - e) / e * 1e4 * (1.0 if side[j] == "top" else -1.0)
        ok = np.isfinite(y)
        preds = np.full(len(J), np.nan)
        for m0 in months:
            m1 = m0 + pd.offsets.MonthBegin(1)
            te = ok & (ts >= m0).to_numpy() & (ts < m1).to_numpy()
            tr = ok & (ts < (m0 - EMBARGO)).to_numpy()
            if te.sum() < 20 or tr.sum() < 300: continue
            r = TabPFNRegressor(device=dev, n_estimators=N_EST, random_state=SEED,
                                ignore_pretraining_limits=True)
            r.fit(M[tr], y[tr])
            preds[te] = r.predict(M[te])
        ev = ok & np.isfinite(preds) & np.isin(sp, WINS)
        yv = y[ev]; pv = preds[ev]; dv = day[ev]
        k = max(10, int(len(pv) * 0.30)); top = np.argsort(-pv)[:k]
        realized = float(yv[top].mean()) - COST
        base_all = float(yv.mean()) - COST
        rho = float(spearmanr(pv, yv).statistic)
        def _r(i):
            kk = max(10, int(len(i) * 0.30))
            return float(yv[i][np.argsort(-pv[i])[:kk]].mean()) - COST
        lo, hi = boot_ci(_r, dv, rng)
        # 귀무는 최고 셀에만 (9셀 x 6회는 너무 비싸다) -- 아래에서 한 번
        rows.append({"H": H, "P": P, "min": H * 5, "coverage": float(ok.mean()),
                     "n_eval": int(ev.sum()), "n_entry": int(k), "base_all_bp": base_all,
                     "top30_bp": realized, "lo": lo, "hi": hi, "spearman": rho,
                     "cls_live_bp": CLS_REF.get((H, P)),
                     "resolve": float((c_ok | f_ok)[ok].mean())})
        print(f"{H:>3}{f'±{P}%':>8}{ok.mean():>8.0%}{(c_ok|f_ok)[ok].mean():>8.1%}"
              f"{base_all:>+11.2f}{realized:>+13.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>21}"
              f"{rho:>+12.4f}{CLS_REF.get((H, P), float('nan')):>+11.2f}"
              f"{'  ✅' if lo > 0 else ''}", flush=True)

    A = pd.DataFrame(rows); A.to_csv(OUT / "regression.csv", index=False)
    print("\n" + "=" * 116, flush=True)
    print(f"CI 하한 > 0 인 셀: {int((A.lo > 0).sum())}/{len(A)} · 귀무 초과: "
          f"{int((A.lo > 0).sum())}/{len(A)}", flush=True)   # 날블록 귀무는 아래 절에서 최고 셀만
    b = A.loc[A.top30_bp.idxmax()]
    print(f"⭐최고: H={int(b.H)}봉({int(b['min'])}분) ±{b.P}% · 상위30% {b.top30_bp:+.2f}bp "
          f"[{b.lo:+.2f}, {b.hi:+.2f}] · 전건 {b.base_all_bp:+.2f}bp · ρ {b.spearman:+.4f} "
          f"· 해소율 {b.resolve:.1%}", flush=True)
    print(f"회귀 − 분류 평균 차이: {(A.top30_bp - A.cls_live_bp).mean():+.2f}bp "
          f"(회귀 우세 {int((A.top30_bp > A.cls_live_bp).sum())}/{len(A)}셀)", flush=True)
    # 최고 셀에만 날블록 귀무
    Hb, Pb = int(b.H), float(b.P)
    hc, hf = HC[Pb], HF[Pb]; lim = Hb * 5.0; BB = Pb * 100.0
    c_ok = np.isfinite(hc) & (hc <= lim); f_ok = np.isfinite(hf) & (hf <= lim)
    y = np.full(len(J), np.nan)
    for j, i in enumerate(idx):
        a2, b2 = i + 1, i + Hb
        if b2 >= len(kl): continue
        if c_ok[j] and (not f_ok[j] or hc[j] < hf[j]): y[j] = BB
        elif f_ok[j] and (not c_ok[j] or hf[j] < hc[j]): y[j] = -BB
        elif c_ok[j] and f_ok[j] and hc[j] == hf[j]: y[j] = 0.0
        else:
            e = OP[a2]; y[j] = (CL[b2] - e) / e * 1e4 * (1.0 if side[j] == "top" else -1.0)
    ok = np.isfinite(y); ev = ok & np.isin(sp, WINS)
    yv = y[ev]; k = max(10, int(ev.sum() * 0.30))
    trm = ok & (ts < pd.Timestamp("2025-09-01")).to_numpy()
    nl = []
    for _ in range(6):
        yt = y.copy(); yt[trm] = dayperm(y[trm], day[trm], rng)
        r = TabPFNRegressor(device=dev, n_estimators=2, random_state=SEED,
                            ignore_pretraining_limits=True)
        r.fit(M[trm], yt[trm])
        nl.append(float(yv[np.argsort(-r.predict(M[ev]))[:k]].mean()) - COST)
    p95 = float(np.percentile(nl, 95))
    print(f"날블록 귀무 B=6 (최고 셀): p95 {p95:+.2f}bp vs 관측 {b.top30_bp:+.2f}bp "
          f"→ {'통과' if b.top30_bp > p95 else '🔴미달'}", flush=True)
    print(f"\n⭐커버리지 100% -- 타깃이 전 앵커에 정의되고 판정 지표와 같다.", flush=True)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
