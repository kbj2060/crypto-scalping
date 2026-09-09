#!/usr/bin/env python3
"""극점 탐지기 v2 -- **혼합 점수** 탐색 (2026-09-10).

관측된 긴장: 손실가중(v2)은 **강 등급**에서 이기지만 중/약에서 라벨 정밀도가 떨어진다.
기전은 명확하다 -- 라벨은 "극점인가"인데 가중치는 "비싼가"를 섞는다. 강추세 역방향 바닥은
*국소 극점으로는 맞을 수 있어도*(한 시간 튀고) 경제적으로는 최악이다. 그래서 그걸 눌러내면
라벨 정밀도는 내려가고 순bp 는 올라간다.

해법 후보: 두 점수를 섞어 랭킹을 만든다.
    score(α) = (1-α)·rank(P_극점) + α·rank(P_손실가중)
α=0 이 v1, α=1 이 v2. 중간이 둘 다 살리는지 본다. **분위 랭크로 섞는다** -- 두 점수의
스케일/캘리브레이션이 다르므로 확률을 직접 곱하면 한쪽이 지배한다.

판정(사용자 지정): 같은 등급·같은 건/일에서 정밀도 ↑ 이고 역추세 비중 ↓.
순bp 는 보조. 등급은 배타 구간(강>중>약) -- 라이브 grade_of 와 같은 정의.
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
TIER_Q = (("강", 0.05), ("중", 0.10), ("약", 0.25))
TQ_HI, TQ_LO, CAP_W = 0.8, 0.2, 4.0
H_EVAL, COST = 48, 10.0


def load_px():
    best, bn = None, 0
    for f in glob.glob(str(EX / "klcache/ETHUSDT_5m_*.parquet")):
        d = pd.read_parquet(f)
        if len(d) > bn: best, bn = d, len(d)
    return best.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from sklearn.ensemble import HistGradientBoostingClassifier

    A = pd.read_parquet(EX / "extreme_frame.parquet")
    fm = json.load(open(EX / "extreme_frame_meta.json"))
    feats, VAL0 = fm["feats"], pd.Timestamp(fm["val0"])
    A = A[A["_y"] >= 0].sort_values("_ts").reset_index(drop=True)
    px = load_px()
    op = px["open"].to_numpy(float); cl = px["close"].to_numpy(float); n = len(px)
    pos = pd.Series(np.arange(n), index=pd.DatetimeIndex(px["timestamp"].to_numpy()))
    A["_j"] = pos.reindex(pd.DatetimeIndex(A["_ts"])).to_numpy()
    A = A.dropna(subset=["_j"]).copy(); A["_j"] = A["_j"].astype(int)
    A = A[A["_j"] + H_EVAL + 1 < n].reset_index(drop=True)
    j = A["_j"].to_numpy(); lg = A["_long"].to_numpy(bool); entry = op[j+1]
    ret_bp = np.where(lg, (cl[j+H_EVAL]-entry)/entry, (entry-cl[j+H_EVAL])/entry)*1e4

    y = A["_y"].to_numpy(int); ts = A["_ts"]; tq = A["_tq"].to_numpy(float)
    counter = (lg & (tq <= TQ_LO)) | (~lg & (tq >= TQ_HI))
    tr = (ts < VAL0).to_numpy(); oos = ~tr
    X = np.nan_to_num(A[feats].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def fit(w):
        P = np.zeros(len(A))
        for sd in SEEDS:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                               min_samples_leaf=40, l2_regularization=1.0,
                                               random_state=sd)
            m.fit(X[tr], y[tr], sample_weight=w[tr]); P += m.predict_proba(X)[:, 1]/len(SEEDS)
        return P

    w2 = np.ones(len(A)); w2[(y == 0) & counter] = CAP_W
    for c in (0, 1):
        m = tr & (y == c); w2[m] /= w2[m].mean()
    P1, P2 = fit(np.ones(len(A))), fit(w2)

    o = np.flatnonzero(oos); k = len(o)//2
    cut_m = np.zeros(len(A), bool); cut_m[o[:k]] = True
    ev_m = np.zeros(len(A), bool); ev_m[o[k:]] = True
    days = (ts[ev_m].max()-ts[ev_m].min()).total_seconds()/86400
    q1 = ts[ev_m].min() + (ts[ev_m].max()-ts[ev_m].min())/2
    e1 = ev_m & (ts <= q1).to_numpy(); e2 = ev_m & (ts > q1).to_numpy()
    # 분위 랭크는 **컷창에서** 적합해 평가창에 적용한다(평가창 정보로 랭크를 만들면 순환)
    def rk(P):
        ref = np.sort(P[cut_m])
        return np.searchsorted(ref, P) / max(len(ref), 1)
    r1, r2 = rk(P1), rk(P2)

    print(f"평가 {ts[ev_m].min():%m-%d}~{ts[ev_m].max():%m-%d} ({days:.0f}일) · 기저 {y[ev_m].mean():.3f} "
          f"· 모집단 역추세 {counter[ev_m].mean():.3f}\n")
    print("=" * 112)
    print(f"{'α':<6}{'등급':<4}{'컷':>8}{'정밀도':>9}{'전반':>8}{'후반':>8}{'건/일':>8}"
          f"{'역추세':>8}{'역추세정밀':>10}{'순bp':>9}")
    print("=" * 112)
    rows = []
    for al in [float(x) for x in a.alphas.split(",")]:
        S = (1-al)*r1 + al*r2
        cuts = {g: float(np.quantile(S[cut_m], 1-q)) for g, q in TIER_Q}
        prev = None
        for g, _ in TIER_Q:
            band = (S >= cuts[g]) & ((S < cuts[prev]) if prev else True)
            sel = ev_m & band
            cs = sel & counter
            r = dict(alpha=al, tier=g, cut=cuts[g], prec=float(y[sel].mean()),
                     prec1=float(y[e1 & band].mean()) if (e1 & band).sum() >= 15 else np.nan,
                     prec2=float(y[e2 & band].mean()) if (e2 & band).sum() >= 15 else np.nan,
                     per_day=sel.sum()/days, counter=float(counter[sel].mean()),
                     cnt_prec=float(y[cs].mean()) if cs.sum() >= 10 else np.nan,
                     net=float(ret_bp[sel].mean()-COST), n=int(sel.sum()))
            rows.append(r); prev = g
            cp = f"{r['cnt_prec']:.3f}" if r['cnt_prec'] == r['cnt_prec'] else "  -  "
            print(f"{al:<6.2f}{g:<4}{r['cut']:>8.4f}{r['prec']:>9.4f}{r['prec1']:>8.3f}{r['prec2']:>8.3f}"
                  f"{r['per_day']:>8.2f}{r['counter']:>8.3f}{cp:>10}{r['net']:>9.2f}")
        print("-" * 112)
    d = pd.DataFrame(rows); d.to_csv(OUT / "blend.csv", index=False)

    print("\n판정 -- α=0(v1) 대비 세 등급 전부 정밀도↑ 이고 역추세↓ 인 α")
    b = d[d.alpha == 0.0].set_index("tier")
    for al in sorted(d.alpha.unique()):
        if al == 0.0: continue
        g = d[d.alpha == al].set_index("tier")
        ok = [(g.loc[t, "prec"] > b.loc[t, "prec"] and g.loc[t, "counter"] < b.loc[t, "counter"])
              for t, _ in TIER_Q]
        dp = " ".join(f"{t}{g.loc[t,'prec']-b.loc[t,'prec']:+.3f}" for t, _ in TIER_Q)
        dc = " ".join(f"{t}{g.loc[t,'counter']-b.loc[t,'counter']:+.3f}" for t, _ in TIER_Q)
        print(f"  α={al:<5.2f} 통과 {sum(ok)}/3   정밀도 {dp}   역추세 {dc}")
    print(f"\n저장 {OUT/'blend.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
