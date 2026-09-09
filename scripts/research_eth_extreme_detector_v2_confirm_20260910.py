#!/usr/bin/env python3
"""극점 탐지기 v2 확인 배터리 (2026-09-10).

스윕에서 C(역추세 오답 가중)가 배포판 v1(하드게이트) 을 5/5 커버리지 정밀도 우위로 이겼다.
⚠️λ=4/8/16 이 **완전히 같은 숫자**였다 -- 가중치 상한 clip(·,0.05,4.0) 에서 포화됐기 때문이다.
   즉 실효 파라미터는 λ 가 아니라 **상한**이다. 여기서 상한을 스윕해 포화점을 확인한다.

추가 팔:
  E 연속 추세페널티  0.2/0.8 계단 대신 |tq-0.5| 에 연속 비례 (문턱 임의성 제거)
  C+게이트           학습 가중과 하드게이트를 같이 쓰면 더 나은가

확인 항목 (이 저장소 표준 대조군):
  · 순bp 를 **두 반기 따로** + 방향뒤집기 대조군 + 무작위 방향 귀무와 함께
  · 시드 견고성 (5 시드 개별 값)
  · 역추세 콜의 정밀도를 따로 -- v2 가 내는 소수의 역추세 콜이 실제로 맞는가
    (게이트는 이 질문에 답할 수 없다. 전부 죽이니까.)
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
RATES = (1.5, 2.5, 4.0)
RNG = np.random.default_rng(20260910)


def load_px():
    best, bn = None, 0
    for f in glob.glob(str(EX / "klcache/ETHUSDT_5m_*.parquet")):
        d = pd.read_parquet(f)
        if len(d) > bn: best, bn = d, len(d)
    return best.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--caps", default="2,3,4,6,10")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from sklearn.ensemble import HistGradientBoostingClassifier

    A = pd.read_parquet(EX / "extreme_frame.parquet")
    fm = json.load(open(EX / "extreme_frame_meta.json"))
    feats, VAL0 = fm["feats"], pd.Timestamp(fm["val0"])
    A = A[A["_y"] >= 0].sort_values("_ts").reset_index(drop=True)
    px = load_px()
    op = px["open"].to_numpy(float); hi = px["high"].to_numpy(float)
    lo = px["low"].to_numpy(float); cl = px["close"].to_numpy(float); n = len(px)
    pos = pd.Series(np.arange(n), index=pd.DatetimeIndex(px["timestamp"].to_numpy()))
    A["_j"] = pos.reindex(pd.DatetimeIndex(A["_ts"])).to_numpy()
    A = A.dropna(subset=["_j"]).copy(); A["_j"] = A["_j"].astype(int)
    A = A[A["_j"] + H_EVAL + 1 < n].reset_index(drop=True)
    j = A["_j"].to_numpy(); lg = A["_long"].to_numpy(bool)
    entry = op[j + 1]
    A["_ret_bp"] = np.where(lg, (cl[j+H_EVAL]-entry)/entry, (entry-cl[j+H_EVAL])/entry) * 1e4

    y = A["_y"].to_numpy(int); tq = A["_tq"].to_numpy(float); ts = A["_ts"]
    tr = (ts < VAL0).to_numpy(); oos = ~tr
    counter = (lg & (tq <= TQ_LO)) | (~lg & (tq >= TQ_HI))
    gate_ok = ~counter
    # 연속 추세 페널티: 방향과 어긋난 정도에 비례 (bottom 은 tq 가 낮을수록, top 은 높을수록 불리)
    mis = np.where(lg, np.clip(0.5 - tq, 0, None), np.clip(tq - 0.5, 0, None)) * 2.0   # 0..1
    X = np.nan_to_num(A[feats].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ret_bp = A["_ret_bp"].to_numpy()

    oi = np.flatnonzero(oos); half = oi[len(oi)//2]; ar = np.arange(len(A))
    cutm = oos & (ar <= half); evm = oos & (ar > half)
    cut_days = (ts[cutm].max()-ts[cutm].min()).total_seconds()/86400
    days = (ts[evm].max()-ts[evm].min()).total_seconds()/86400
    q1 = ts[evm].min() + (ts[evm].max()-ts[evm].min())/2
    ev1 = evm & (ts <= q1).to_numpy(); ev2 = evm & (ts > q1).to_numpy()
    print(f"측정 {ts[evm].min():%m-%d}~{ts[evm].max():%m-%d} ({days:.0f}일) · 기저 {y[evm].mean():.3f} "
          f"· 모집단 역추세 {counter[evm].mean():.3f}\n")

    def fit(w):
        P = np.zeros(len(A)); ps = []
        for sd in SEEDS:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                               min_samples_leaf=40, l2_regularization=1.0,
                                               random_state=sd)
            m.fit(X[tr], y[tr], sample_weight=w[tr]); p = m.predict_proba(X)[:, 1]
            ps.append(p); P += p/len(SEEDS)
        return P, ps

    def norm(w, cap):
        w = np.clip(w, 0.05, cap).astype(float)
        for c in (0, 1):
            m = tr & (y == c)
            if m.sum(): w[m] /= w[m].mean()
        return w

    arms = {}
    P0, ps0 = fit(np.ones(len(A)))
    arms["A 무게없음"] = (P0, ps0, None)
    arms["v1 배포판(게이트)"] = (P0, ps0, gate_ok)
    for cap in [float(x) for x in a.caps.split(",")]:
        w = np.ones(len(A)); w[(y == 0) & counter] = 1e9      # 상한이 실효 파라미터
        P, ps = fit(norm(w, cap)); arms[f"C 상한={cap:g}"] = (P, ps, None)
    w = np.ones(len(A)); w[y == 0] = 1.0 + 3.0*mis[y == 0]
    PE, psE = fit(norm(w, 4.0)); arms["E 연속페널티"] = (PE, psE, None)
    Pc, psc = arms["C 상한=4"][0], arms["C 상한=4"][1]
    arms["C 상한=4 +게이트"] = (Pc, psc, gate_ok)

    def pick(P, mask, rate, window):
        pool_c = cutm if mask is None else (cutm & mask)
        k = int(round(rate * cut_days)); v = P[pool_c]
        cut = -np.inf if len(v) <= k else np.sort(v)[::-1][k-1]
        return window & (P >= cut) & (True if mask is None else mask)

    rows = []
    for name, (P, ps, mask) in arms.items():
        for rate in RATES:
            sel = pick(P, mask, rate, evm)
            if sel.sum() < 25: continue
            s1, s2 = pick(P, mask, rate, ev1), pick(P, mask, rate, ev2)
            seeds_p = [y[pick(p, mask, rate, evm)].mean() for p in ps]
            # 방향뒤집기 대조군 + 무작위 방향 귀무
            flip = float(-ret_bp[sel].mean() - COST)
            rnd = float(np.mean([np.mean(np.where(RNG.random(int(sel.sum())) < .5, 1, -1)
                                         * ret_bp[sel]) - COST for _ in range(400)]))
            cm = sel & counter
            rows.append(dict(arm=name, rate=rate, n=int(sel.sum()), per_day=sel.sum()/days,
                             prec=float(y[sel].mean()), counter=float(counter[sel].mean()),
                             cnt_n=int(cm.sum()), cnt_prec=float(y[cm].mean()) if cm.sum() >= 10 else np.nan,
                             net=float(ret_bp[sel].mean()-COST),
                             net1=float(ret_bp[s1].mean()-COST) if s1.sum() >= 15 else np.nan,
                             net2=float(ret_bp[s2].mean()-COST) if s2.sum() >= 15 else np.nan,
                             flip=flip, rnd=rnd,
                             seed_lo=float(np.min(seeds_p)), seed_hi=float(np.max(seeds_p))))
        print(f"  {name} 완료", flush=True)

    d = pd.DataFrame(rows); d.to_csv(OUT / "confirm.csv", index=False)
    print("\n" + "=" * 122)
    print("확인 배터리 -- 순bp 는 두 반기 따로, 방향뒤집기/무작위 대조군과 함께")
    print("=" * 122)
    print(f"{'팔':<18}{'건/일':>6}{'정밀도':>8}{'시드범위':>14}{'역추세':>7}{'역추세건':>7}{'역추세정밀':>9}"
          f"{'순bp':>8}{'전반':>8}{'후반':>8}{'뒤집기':>8}{'무작위':>8}")
    for r in d.itertuples():
        cp = f"{r.cnt_prec:.3f}" if r.cnt_prec == r.cnt_prec else "  -  "
        print(f"{r.arm:<18}{r.per_day:>6.2f}{r.prec:>8.3f}{'['+f'{r.seed_lo:.3f},{r.seed_hi:.3f}'+']':>14}"
              f"{r.counter:>7.3f}{r.cnt_n:>7d}{cp:>9}{r.net:>8.2f}{r.net1:>8.2f}{r.net2:>8.2f}"
              f"{r.flip:>8.2f}{r.rnd:>8.2f}")
    print(f"\n저장 {OUT/'confirm.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
