#!/usr/bin/env python3
"""극점 탐지기 v2 -- λ 스윕 + **배포판(하드게이트 포함) 정면 비교** (2026-09-10).

1차 실험에서 C(역추세 오답 표적 가중)가 강 등급에서 역추세 비중을 0.289 -> 0.114 로 줄이면서
정밀도를 유지했다. 그런데 **진짜 기준선은 게이트를 안 단 A 가 아니라 배포판 v1 = A + 하드게이트**다.
게이트는 강추세 구간의 역추세 콜을 전부 죽여 역추세 비중을 0 에 가깝게 만드는 대신
하루 4.39건을 버린다. 그래서 비교는 **같은 건/일에서 정밀도**로 해야 공정하다
(호메로스 "건수 맞춰야 답이 바뀐다" -- 명목 커버리지로 비교하면 불공정).

  · 커버리지-정밀도 곡선: 건/일 을 맞춰 놓고 정밀도·역추세비중·순bp 를 비교
  · λ 스윕: 역추세 오답 가중 배수
  · 측정창을 다시 반으로 갈라 두 구간 모두 확인 (한 창짜리 우위는 버린다)
  · 무작위 선택 귀무 (같은 건수, B=400) -- 정밀도 우위가 표본 크기 효과가 아닌지
"""
from __future__ import annotations
import argparse, glob, json, os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "tmp/eth_signal_map_20260909"
OUT = ROOT / "tmp/eth_extreme_v2_20260910"
SEEDS = [20260909, 771233, 305610, 517758, 961476]
H_EVAL, COST = 48, 10.0
TQ_HI, TQ_LO = 0.8, 0.2
RATES = (1.5, 2.5, 4.0, 7.0, 11.0)      # 건/일 격자 (v1 강 1.46 · 강+중 2.93 · 전체 7.21)
RNG = np.random.default_rng(20260910)


def load_px():
    best, bn = None, 0
    for f in glob.glob(str(EX / "klcache/ETHUSDT_5m_*.parquet")):
        d = pd.read_parquet(f)
        if len(d) > bn: best, bn = d, len(d)
    return best.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", default="1,2,4,8,16")
    ap.add_argument("--cap", type=float, default=4.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from sklearn.ensemble import HistGradientBoostingClassifier

    A = pd.read_parquet(EX / "extreme_frame.parquet")
    fm = json.load(open(EX / "extreme_frame_meta.json"))
    feats, VAL0 = fm["feats"], pd.Timestamp(fm["val0"])
    A = A[A["_y"] >= 0].sort_values("_ts").reset_index(drop=True)
    px = load_px()
    ts_all = px["timestamp"].to_numpy()
    op = px["open"].to_numpy(float); hi = px["high"].to_numpy(float)
    lo = px["low"].to_numpy(float); cl = px["close"].to_numpy(float); n = len(px)
    tr_ = np.maximum(hi[1:]-lo[1:], np.maximum(abs(hi[1:]-cl[:-1]), abs(lo[1:]-cl[:-1])))
    atr = np.full(n, np.nan); atr[1:] = pd.Series(tr_).ewm(alpha=1/14, adjust=False).mean().to_numpy()
    pos = pd.Series(np.arange(n), index=pd.DatetimeIndex(ts_all))
    A["_j"] = pos.reindex(pd.DatetimeIndex(A["_ts"])).to_numpy()
    A = A.dropna(subset=["_j"]).copy(); A["_j"] = A["_j"].astype(int)
    A = A[A["_j"] + H_EVAL + 1 < n].reset_index(drop=True)
    j = A["_j"].to_numpy(); lg = A["_long"].to_numpy(bool)
    fmin = pd.Series(lo[::-1]).rolling(H_EVAL, min_periods=H_EVAL).min().to_numpy()[::-1]
    fmax = pd.Series(hi[::-1]).rolling(H_EVAL, min_periods=H_EVAL).max().to_numpy()[::-1]
    entry = op[j+1]
    mae = np.where(lg, (fmin[j+1]-entry)/entry, (entry-fmax[j+1])/entry)
    A["_mae_atr"] = np.abs(mae)/(atr[j]/cl[j])
    A["_ret_bp"] = np.where(lg, (cl[j+H_EVAL]-entry)/entry, (entry-cl[j+H_EVAL])/entry)*1e4

    y = A["_y"].to_numpy(int); tq = A["_tq"].to_numpy(float); ts = A["_ts"]
    tr = (ts < VAL0).to_numpy(); oos = ~tr
    counter = (lg & (tq <= TQ_LO)) | (~lg & (tq >= TQ_HI))
    gate_ok = ~counter                      # 배포판 v1 하드게이트: 역추세 콜을 통째로 억제
    X = np.nan_to_num(A[feats].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ret_bp = A["_ret_bp"].to_numpy(); mcost = np.nan_to_num(A["_mae_atr"].to_numpy(), nan=1.0)

    oi = np.flatnonzero(oos); half = oi[len(oi)//2]
    ar = np.arange(len(A))
    cutm = oos & (ar <= half); evm = oos & (ar > half)
    days = (ts[evm].max()-ts[evm].min()).total_seconds()/86400
    q1 = ts[evm].min() + (ts[evm].max()-ts[evm].min())/2
    ev1 = evm & (ts <= q1).to_numpy(); ev2 = evm & (ts > q1).to_numpy()
    d1 = (ts[ev1].max()-ts[ev1].min()).total_seconds()/86400
    d2 = (ts[ev2].max()-ts[ev2].min()).total_seconds()/86400
    print(f"컷 {ts[cutm].min():%m-%d} ~ {ts[cutm].max():%m-%d} · 측정 {ts[evm].min():%m-%d} ~ "
          f"{ts[evm].max():%m-%d} ({days:.0f}일, 반기 {d1:.0f}/{d2:.0f}일)")
    print(f"측정창 기저 극점률 {y[evm].mean():.3f} · 역추세 비중 {counter[evm].mean():.3f}\n")

    def fit(w):
        P = np.zeros(len(A)); ps = []
        for sd in SEEDS:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                               min_samples_leaf=40, l2_regularization=1.0,
                                               random_state=sd)
            m.fit(X[tr], y[tr], sample_weight=w[tr]); p = m.predict_proba(X)[:, 1]
            ps.append(p); P += p/len(SEEDS)
        return P, ps

    def norm(w):
        w = np.clip(w, 0.05, a.cap).astype(float)
        for c in (0, 1):
            m = tr & (y == c)
            if m.sum(): w[m] /= w[m].mean()
        return w

    arms = {}
    P0, ps0 = fit(np.ones(len(A))); arms["A 무게없음"] = (P0, ps0, None)
    arms["v1 배포판(게이트)"] = (P0, ps0, gate_ok)
    for lam in [float(x) for x in a.lams.split(",")]:
        w = np.ones(len(A)); w[(y == 0) & counter] = 1.0 + lam
        P, ps = fit(norm(w)); arms[f"C λ={lam:g}"] = (P, ps, None)
    lam_b = 4.0
    w = np.ones(len(A)); w[y == 0] = mcost[y == 0]; w[(y == 0) & counter] *= (1.0+lam_b)
    P, ps = fit(norm(w)); arms[f"D 빗나감×역추세 λ={lam_b:g}"] = (P, ps, None)

    def pick(P, mask, rate, window, wdays):
        """건/일 을 맞춰 상위 k 개 -- 컷은 컷구간에서 잡고 측정창에 적용(순환 방지)."""
        pool_c = cutm & (mask if mask is not None else True)
        k = int(round(rate * ((ts[cutm].max()-ts[cutm].min()).total_seconds()/86400)))
        v = P[pool_c]
        if len(v) <= k: cut = -np.inf
        else: cut = np.sort(v)[::-1][k-1]
        sel = window & (P >= cut) & (mask if mask is not None else True)
        return sel

    rows = []
    for name, (P, ps, mask) in arms.items():
        for rate in RATES:
            sel = pick(P, mask, rate, evm, days)
            if sel.sum() < 25: continue
            s1, s2 = pick(P, mask, rate, ev1, d1), pick(P, mask, rate, ev2, d2)
            per_seed = []
            for p in ps:
                q = pick(p, mask, rate, evm, days)
                per_seed.append(y[q].mean() if q.sum() >= 15 else np.nan)
            # 무작위 귀무: 같은 모집단(게이트 적용 후)에서 같은 건수
            pool = np.flatnonzero(evm & (mask if mask is not None else True))
            nulls = [y[RNG.choice(pool, size=int(sel.sum()), replace=False)].mean() for _ in range(400)]
            rows.append(dict(arm=name, rate=rate, n=int(sel.sum()), per_day=sel.sum()/days,
                             prec=float(y[sel].mean()), null=float(np.mean(nulls)),
                             p_val=float(np.mean(np.array(nulls) >= y[sel].mean())),
                             counter=float(counter[sel].mean()),
                             net_bp=float(ret_bp[sel].mean()-COST),
                             prec1=float(y[s1].mean()) if s1.sum() >= 15 else np.nan,
                             prec2=float(y[s2].mean()) if s2.sum() >= 15 else np.nan,
                             prec_sd=float(np.nanstd(per_seed))))
        print(f"  {name} 완료", flush=True)

    d = pd.DataFrame(rows); d.to_csv(OUT / "sweep.csv", index=False)
    print("\n" + "=" * 118)
    print("커버리지-정밀도 곡선 (건/일 을 맞춘 정면 비교)")
    print("=" * 118)
    print(f"{'팔':<22}{'목표건/일':>9}{'실건/일':>8}{'정밀도':>8}{'귀무':>7}{'p':>7}"
          f"{'±시드':>7}{'전반':>7}{'후반':>7}{'역추세':>8}{'순bp':>8}")
    for r in d.itertuples():
        print(f"{r.arm:<22}{r.rate:>9.1f}{r.per_day:>8.2f}{r.prec:>8.3f}{r.null:>7.3f}{r.p_val:>7.3f}"
              f"{r.prec_sd:>7.3f}{r.prec1:>7.3f}{r.prec2:>7.3f}{r.counter:>8.3f}{r.net_bp:>8.2f}")

    print("\n" + "=" * 118)
    print("판정 -- 배포판 v1(게이트) 대비, 같은 건/일")
    print("=" * 118)
    b = d[d.arm == "v1 배포판(게이트)"].set_index("rate")
    best = None
    for name in arms:
        if name.startswith("v1 ") or name.startswith("A "): continue
        g = d[d.arm == name].set_index("rate"); wins = 0; tot = 0
        for rate in RATES:
            if rate not in g.index or rate not in b.index: continue
            dp = g.loc[rate, "prec"] - b.loc[rate, "prec"]; tot += 1; wins += dp > 0
        cov = d[(d.arm == name)]
        print(f"  {name:<22} 정밀도 승 {wins}/{tot} 커버리지  ·  "
              f"두 반기 모두 귀무 초과 {sum((cov.prec1>cov.null)&(cov.prec2>cov.null))}/{len(cov)}")
        if best is None or wins > best[1]: best = (name, wins)
    print(f"\n저장 {OUT/'sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
