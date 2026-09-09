#!/usr/bin/env python3
"""극점 탐지기 v2 견고성 -- **순환성 점검** (2026-09-10).

C 팔은 `tq`(ret144 의 2016봉 롤링분위)로 역추세 오답을 가중한다. 그런데 판정 지표인
"역추세 콜 비중"도 **같은 tq** 로 잰다 -- 학습에 쓴 정의로 평가하면 순환이다.
그래서 **다른 추세 정의**로 다시 잰다. 정의가 바뀌어도 역추세 비중이 줄고 정밀도가
유지되면 진짜다.

  T1 ret144/2016봉 분위 (학습에 쓴 것 -- 기준)
  T2 ret288 (24시간) 의 2016봉 분위      -- 창 길이를 바꾼다
  T3 EMA96 기울기 부호·크기의 분위        -- 통계량 자체를 바꾼다
  T4 ret144 의 **전체 표본** 분위          -- 롤링이 아니라 고정 분위

추가: 라벨 창 W=12 는 v1 이 고른 값이다. W=24 로 재라벨해도 같은 결론인가.
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
H_EVAL, COST, CAP = 48, 10.0, 4.0
TQ_HI, TQ_LO = 0.8, 0.2
RATES = (1.5, 2.5, 4.0)


def load_px():
    best, bn = None, 0
    for f in glob.glob(str(EX / "klcache/ETHUSDT_5m_*.parquet")):
        d = pd.read_parquet(f)
        if len(d) > bn: best, bn = d, len(d)
    return best.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--w", type=int, default=12)
    a = ap.parse_args()
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
    A = A[A["_j"] + max(H_EVAL, a.w) + 1 < n].reset_index(drop=True)
    j = A["_j"].to_numpy(); lg = A["_long"].to_numpy(bool)
    entry = op[j+1]
    A["_ret_bp"] = np.where(lg, (cl[j+H_EVAL]-entry)/entry, (entry-cl[j+H_EVAL])/entry)*1e4

    # --- 네 가지 추세 정의 (전부 인과적: 봉 i 까지) --------------------------------
    S = pd.Series(cl)
    r144 = (S/S.shift(144)-1.0); r288 = (S/S.shift(288)-1.0)
    ema = S.ewm(span=96, adjust=False).mean(); slope = (ema/ema.shift(24)-1.0)
    T = {}
    T["T1 ret144·롤링2016"] = r144.rolling(2016, min_periods=500).rank(pct=True).to_numpy()[j]
    T["T2 ret288·롤링2016"] = r288.rolling(2016, min_periods=500).rank(pct=True).to_numpy()[j]
    T["T3 EMA96기울기"]    = slope.rolling(2016, min_periods=500).rank(pct=True).to_numpy()[j]
    T["T4 ret144·전체분위"]  = r144.rank(pct=True).to_numpy()[j]

    y = A["_y"].to_numpy(int); ts = A["_ts"]
    tr = (ts < VAL0).to_numpy(); oos = ~tr
    X = np.nan_to_num(A[feats].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ret_bp = A["_ret_bp"].to_numpy()
    # ⚠️학습 가중은 스윕/배포 후보와 **완전히 같은** 정의를 쓴다 -- 프레임의 _tq
    #   (= v1 게이트가 쓰는 ATR 정규화 ret144 의 7일 롤링 분위). T1~T4 는 **평가 전용**이다.
    tq0 = A["_tq"].to_numpy(float)
    counter1 = (lg & (tq0 <= TQ_LO)) | (~lg & (tq0 >= TQ_HI))
    T = {"T0 _tq(학습에 쓴 정의)": tq0, **T}

    oi = np.flatnonzero(oos); half = oi[len(oi)//2]; ar = np.arange(len(A))
    cutm = oos & (ar <= half); evm = oos & (ar > half)
    cut_days = (ts[cutm].max()-ts[cutm].min()).total_seconds()/86400
    days = (ts[evm].max()-ts[evm].min()).total_seconds()/86400

    def fit(w):
        P = np.zeros(len(A))
        for sd in SEEDS:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                               min_samples_leaf=40, l2_regularization=1.0,
                                               random_state=sd)
            m.fit(X[tr], y[tr], sample_weight=w[tr]); P += m.predict_proba(X)[:, 1]/len(SEEDS)
        return P

    def norm(w):
        w = np.clip(w, 0.05, CAP).astype(float)
        for c in (0, 1):
            m = tr & (y == c)
            if m.sum(): w[m] /= w[m].mean()
        return w

    wC = np.ones(len(A)); wC[(y == 0) & counter1] = 1e9
    PA, PC = fit(np.ones(len(A))), fit(norm(wC))
    gate_ok = ~counter1

    def pick(P, mask, rate):
        pool_c = cutm if mask is None else (cutm & mask)
        k = int(round(rate*cut_days)); v = P[pool_c]
        cut = -np.inf if len(v) <= k else np.sort(v)[::-1][k-1]
        return evm & (P >= cut) & (True if mask is None else mask)

    print(f"라벨창 W={a.w} (프레임 라벨 그대로) · 측정 {ts[evm].min():%m-%d}~{ts[evm].max():%m-%d} "
          f"({days:.0f}일) · 기저 {y[evm].mean():.3f}\n")
    print("=" * 108)
    print("추세 정의를 바꿔서 역추세 비중을 다시 잰다 -- 학습에 쓴 정의(T1)로만 좋아지면 순환이다")
    print("=" * 108)
    print(f"{'추세정의':<20}{'건/일':>6}{'A 무게없음':>22}{'v1 게이트':>18}{'C 상한=4':>20}")
    print(f"{'':<20}{'':>6}{'역추세비중  정밀도':>22}{'역추세비중':>18}{'역추세비중  정밀도':>20}")
    rows = []
    for tname, tvals in T.items():
        tvv = np.nan_to_num(tvals, nan=0.5)
        cc = (lg & (tvv <= TQ_LO)) | (~lg & (tvv >= TQ_HI))
        for rate in RATES:
            sA, sV, sC = pick(PA, None, rate), pick(PA, gate_ok, rate), pick(PC, None, rate)
            rows.append(dict(trend=tname, rate=rate,
                             A_cnt=float(cc[sA].mean()), A_prec=float(y[sA].mean()),
                             V_cnt=float(cc[sV].mean()), V_prec=float(y[sV].mean()),
                             C_cnt=float(cc[sC].mean()), C_prec=float(y[sC].mean()),
                             C_net=float(ret_bp[sC].mean()-COST), V_net=float(ret_bp[sV].mean()-COST)))
            r = rows[-1]
            print(f"{tname if rate==RATES[0] else '':<20}{rate:>6.1f}"
                  f"{r['A_cnt']:>12.3f}{r['A_prec']:>10.3f}"
                  f"{r['V_cnt']:>14.3f}    "
                  f"{r['C_cnt']:>12.3f}{r['C_prec']:>10.3f}")
    d = pd.DataFrame(rows); d.to_csv(OUT / "robust_trend.csv", index=False)
    print("\n요약 -- C 가 A 대비 역추세 비중을 줄인 비율 (추세 정의별 평균)")
    for tname in T:
        g = d[d.trend == tname]
        print(f"  {tname:<20} A {g.A_cnt.mean():.3f} -> C {g.C_cnt.mean():.3f} "
              f"({(1-g.C_cnt.mean()/max(g.A_cnt.mean(),1e-9))*100:+.0f}%)   "
              f"정밀도 {g.A_prec.mean():.3f} -> {g.C_prec.mean():.3f}")
    print(f"\n저장 {OUT/'robust_trend.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
