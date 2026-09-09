#!/usr/bin/env python3
"""극점 탐지기 v2 -- **손실가중 학습**으로 추세 하드게이트를 대체한다 (2026-09-10).

## 문제 (2026-09-09 사용자 지적 + 실측)
v1 은 강한 상승에서 천장 콜을 계속 뽑는다: 정밀도 51.6% 인데 **적중 +10.8 / 빗나감 −60.7bp**.
모집단 역추세 비중 71.0% -> 등급부여 87.3% 로 오히려 **늘린다**. 원인은 문서에 이미 규명돼 있다 --
**라벨이 손익을 벌하지 않으니 모델이 비싼 자리를 피할 이유가 없다.** 09-09 는 추세 피쳐를 더해
재학습해 봤으나 안 고쳐졌고(역추세 비중 30.8->30.2%), 결국 ret144 7일분위 하드게이트로 막았다 --
살리는 2.93건/일보다 **억제하는 4.39건/일이 더 많다**.

## 이 스크립트가 하는 일
피쳐를 더하는 대신 **목적함수를 바꾼다**. 라벨(_y = 극점이 W=12봉 버티는가)과 모집단·피쳐·분할은
v1 그대로 두고, 학습 표본 가중치에만 "이 자리를 잘못 부르면 얼마나 비싼가"를 넣는다.

  A. 기준선(v1)   가중치 없음
  B. 빗나감 가중   오답(_y=0)에 **ATR 정규화 MAE**(H=48 최대역행폭) 비례 가중
  C. 역추세 표적   오답 중 **강추세 역방향**인 것만 (1+LAM) 배
  D. B+C

⭐가중치를 **ATR 로 정규화**하는 게 핵심이다. 원시 bp 로 주면 "고변동 구간이 중요하다"가 되고,
   그 축은 09-09 에 이미 죽었다(강도만 타깃 정밀도 76.9% / 경제성 −20.68bp).
⭐클래스별로 평균 가중치를 1 로 맞춘다. 안 그러면 재가중과 클래스 재균형이 뒤섞인다.

## 판정 (사용자 지정, 2026-09-10)
**건수를 맞춘 뒤** (1) 정밀도가 v1 보다 높고 (2) 역추세 콜 비중이 낮으면 승격.
순bp/AUC 는 보조 지표로만 본다.
⚠️순환 방지: 등급 컷은 표본외 **전반부**에서 잡고 정밀도는 **후반부**에서 잰다(v1 규약).
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


def load_px() -> pd.DataFrame:
    """프레임을 만든 klcache 중 가장 긴 ETH 5분봉. _i 대신 _ts 로 조인한다(파일 선택 모호성 회피)."""
    best, bn = None, 0
    for f in glob.glob(str(EX / "klcache/ETHUSDT_5m_*.parquet")):
        d = pd.read_parquet(f)
        if len(d) > bn: best, bn = d, len(d)
    return best.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=2.0, help="C 팔 역추세 오답 추가배수")
    ap.add_argument("--cap", type=float, default=4.0, help="가중치 상한(꼬리 한 건이 학습을 지배하지 않도록)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    A = pd.read_parquet(EX / "extreme_frame.parquet")
    fm = json.load(open(EX / "extreme_frame_meta.json"))
    feats, VAL0 = fm["feats"], pd.Timestamp(fm["val0"])
    A = A[A["_y"] >= 0].sort_values("_ts").reset_index(drop=True)

    # --- 미래 창에서 빗나감 비용(ATR 정규화 MAE) 계산 ---------------------------------
    px = load_px()
    ts_all = px["timestamp"].to_numpy()
    op = px["open"].to_numpy(float); hi = px["high"].to_numpy(float)
    lo = px["low"].to_numpy(float); cl = px["close"].to_numpy(float)
    n = len(px)
    tr_ = np.maximum(hi[1:] - lo[1:], np.maximum(abs(hi[1:] - cl[:-1]), abs(lo[1:] - cl[:-1])))
    atr = np.full(n, np.nan); atr[1:] = pd.Series(tr_).ewm(alpha=1/14, adjust=False).mean().to_numpy()
    pos = pd.Series(np.arange(n), index=pd.DatetimeIndex(ts_all))
    A["_j"] = pos.reindex(pd.DatetimeIndex(A["_ts"])).to_numpy()
    A = A.dropna(subset=["_j"]).copy(); A["_j"] = A["_j"].astype(int)
    A = A[A["_j"] + H_EVAL + 1 < n].reset_index(drop=True)
    j = A["_j"].to_numpy(); lg = A["_long"].to_numpy(bool)

    # 롤링 극값 (뒤집어서 rolling -> [t+1, t+H])
    fmin = pd.Series(lo[::-1]).rolling(H_EVAL, min_periods=H_EVAL).min().to_numpy()[::-1]
    fmax = pd.Series(hi[::-1]).rolling(H_EVAL, min_periods=H_EVAL).max().to_numpy()[::-1]
    entry = op[j + 1]
    mae = np.where(lg, (fmin[j + 1] - entry) / entry, (entry - fmax[j + 1]) / entry)   # <=0
    ret = np.where(lg, (cl[j + H_EVAL] - entry) / entry, (entry - cl[j + H_EVAL]) / entry)
    A["_mae_atr"] = np.abs(mae) / (atr[j] / cl[j])       # ATR 정규화 -- 고변동 편향 제거
    A["_ret_bp"] = ret * 1e4

    y = A["_y"].to_numpy(int); tq = A["_tq"].to_numpy(float)
    ts = A["_ts"]; tr = (ts < VAL0).to_numpy(); oos = ~tr
    counter = (lg & (tq <= TQ_LO)) | (~lg & (tq >= TQ_HI))     # 역추세 콜
    X = np.nan_to_num(A[feats].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    print(f"학습 {tr.sum():,} · 표본외 {oos.sum():,} · 피쳐 {len(feats)} · 기저 {y[oos].mean():.4f}")
    print(f"빗나감 MAE/ATR 중앙 (오답) {np.median(A._mae_atr[y==0]):.2f} · (정답) {np.median(A._mae_atr[y==1]):.2f}")
    print(f"모집단 역추세 비중 {counter.mean():.3f} · 표본외 {counter[oos].mean():.3f}\n")

    def norm(w, mask):
        """클래스별 평균 1 로 정규화 -- 재가중과 클래스 재균형을 섞지 않는다."""
        w = np.clip(w, 0.05, a.cap).astype(float)
        for c in (0, 1):
            m = mask & (y == c)
            if m.sum(): w[m] /= w[m].mean()
        return w

    mcost = A["_mae_atr"].to_numpy()
    mcost = np.where(np.isfinite(mcost), mcost, np.nanmedian(mcost))
    arms = {}
    base = np.ones(len(A))
    arms["A 기준선(v1)"] = base
    wB = base.copy(); wB[y == 0] = mcost[y == 0]
    arms["B 빗나감가중"] = wB
    wC = base.copy(); wC[(y == 0) & counter] = 1.0 + a.lam
    arms["C 역추세표적"] = wC
    wD = wB.copy(); wD[(y == 0) & counter] *= (1.0 + a.lam)
    arms["D B+C"] = wD

    # 표본외를 반으로: 앞=컷 잡기, 뒤=정밀도 측정 (순환 방지)
    oi = np.flatnonzero(oos); half = oi[len(oi) // 2]
    cutm = oos & (np.arange(len(A)) <= half); evm = oos & (np.arange(len(A)) > half)
    print(f"컷 구간 {ts[cutm].min()} ~ {ts[cutm].max()} ({cutm.sum():,})")
    print(f"측정 구간 {ts[evm].min()} ~ {ts[evm].max()} ({evm.sum():,})\n")
    days = (ts[evm].max() - ts[evm].min()).total_seconds() / 86400

    rows = []
    for name, w0 in arms.items():
        w = norm(w0.copy(), tr)
        P = np.zeros(len(A)); ps = []
        for sd in SEEDS:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=None,
                                               min_samples_leaf=40, l2_regularization=1.0,
                                               random_state=sd)
            m.fit(X[tr], y[tr], sample_weight=w[tr])
            p = m.predict_proba(X)[:, 1]; ps.append(p); P += p / len(SEEDS)
        auc = roc_auc_score(y[oos], P[oos])
        for tier, q in (("강", 0.05), ("중", 0.10), ("약", 0.25)):
            cut = np.quantile(P[cutm], 1 - q)
            sel = evm & (P >= cut)
            if sel.sum() < 20: continue
            per_seed = []
            for p in ps:
                c2 = np.quantile(p[cutm], 1 - q); s2 = evm & (p >= c2)
                per_seed.append((y[s2].mean(), counter[s2].mean()) if s2.sum() >= 10 else (np.nan, np.nan))
            rows.append(dict(arm=name, tier=tier, n=int(sel.sum()), per_day=sel.sum()/days,
                             prec=float(y[sel].mean()), counter=float(counter[sel].mean()),
                             net_bp=float(A._ret_bp[sel].mean() - COST), auc=float(auc),
                             prec_sd=float(np.nanstd([v[0] for v in per_seed])),
                             counter_sd=float(np.nanstd([v[1] for v in per_seed]))))
        print(f"  {name:<16} AUC(표본외) {auc:.4f}", flush=True)

    d = pd.DataFrame(rows); d.to_csv(OUT / "arms.csv", index=False)
    print("\n" + "=" * 104)
    print(f"등급별 비교 -- 컷은 표본외 전반, 측정은 후반({days:.0f}일). 기저 극점률 {y[evm].mean():.3f}")
    print("=" * 104)
    print(f"{'팔':<16}{'등급':<5}{'건수':>6}{'건/일':>7}{'정밀도':>9}{'±시드':>7}{'역추세비중':>11}{'±시드':>7}{'순bp':>9}")
    for r in d.itertuples():
        print(f"{r.arm:<16}{r.tier:<5}{r.n:>6}{r.per_day:>7.2f}{r.prec:>9.3f}{r.prec_sd:>7.3f}"
              f"{r.counter:>11.3f}{r.counter_sd:>7.3f}{r.net_bp:>9.2f}")

    print("\n" + "=" * 104)
    print("판정 (기준선 A 대비, 같은 등급·같은 커버리지)")
    print("=" * 104)
    b = d[d.arm == "A 기준선(v1)"].set_index("tier")
    for name in arms:
        if name.startswith("A "): continue
        g = d[d.arm == name].set_index("tier")
        ok = []
        for t in ("강", "중", "약"):
            if t not in g.index or t not in b.index: continue
            dp, dc = g.loc[t, "prec"] - b.loc[t, "prec"], g.loc[t, "counter"] - b.loc[t, "counter"]
            ok.append(dp > 0 and dc < 0)
            print(f"  {name:<16}{t}  정밀도 {dp:+.3f}   역추세비중 {dc:+.3f}   "
                  f"{'통과' if (dp>0 and dc<0) else '미달'}")
        print(f"  -> {name}: {sum(ok)}/{len(ok)} 등급 통과\n")
    print(f"저장 {OUT/'arms.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
