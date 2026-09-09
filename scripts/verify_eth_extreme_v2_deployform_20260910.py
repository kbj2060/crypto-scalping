#!/usr/bin/env python3
"""극점 탐지기 v2 -- **배포 형태 그대로** 최종 검증 (2026-09-10).

지금까지의 비교는 HGB-vs-HGB 로 **가중치 효과만** 분리한 통제 실험이었다. 실제 배포는
부모(p1)가 **TabPFN**이고 그 옆에 손실가중 HGB 헤드(p2)를 붙이는 형태다. 그래서 여기서는
서버에서 뽑은 배포 아티팩트의 p1 실점수로 규칙을 다시 잰다.

    배포판   강 = p1 >= 강컷 AND 게이트 통과
    v2      강 = p1 >= 강컷 AND p2 >= 강컷   (게이트 면제)
            중/약은 두 안 모두 오늘의 게이트를 그대로 받는다

⚠️게이트는 **측면 무관**이다 -- gated_of(tq,long) = (tq>=.8)|(tq<=.2). 강추세 구간이면
   순추세 콜까지 죽인다. 라이브 코드 정의를 그대로 옮긴다.
"""
from __future__ import annotations
import json, os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import glob
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "tmp/eth_signal_map_20260909"
CW = ROOT / "data/live/eth_extreme_detector_costw_artifact"
TIER_Q = (("강", 0.05), ("중", 0.10), ("약", 0.25))
TQ_HI, TQ_LO, H, COST = 0.8, 0.2, 48, 10.0


def main() -> int:
    import joblib
    A = pd.read_parquet(EX / "extreme_frame.parquet")
    fm = json.load(open(EX / "extreme_frame_meta.json"))
    feats, VAL0 = fm["feats"], pd.Timestamp(fm["val0"])
    A = A[A["_y"] >= 0].sort_values("_ts").reset_index(drop=True)

    s = pd.read_csv(EX / "tabpfn_p1_scores_20260910.csv", parse_dates=["_ts"])
    A = A.merge(s, on="_ts", how="left")
    assert A["p1"].notna().all(), "p1 조인 실패"
    X = np.nan_to_num(A[feats].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    cw = joblib.load(CW / "model.joblib")
    p2 = np.mean([m.predict_proba(X)[:, 1] for m in cw], axis=0)
    p1 = A["p1"].to_numpy()

    best, bn = None, 0
    for f in glob.glob(str(EX / "klcache/ETHUSDT_5m_*.parquet")):
        d = pd.read_parquet(f)
        if len(d) > bn: best, bn = d, len(d)
    px = best.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    op = px["open"].to_numpy(float); cl = px["close"].to_numpy(float); n = len(px)
    pos = pd.Series(np.arange(n), index=pd.DatetimeIndex(px["timestamp"].to_numpy()))
    A["_j"] = pos.reindex(pd.DatetimeIndex(A["_ts"])).to_numpy()
    keep = A["_j"].notna().to_numpy()
    A = A[keep].reset_index(drop=True); p1, p2 = p1[keep], p2[keep]
    A["_j"] = A["_j"].astype(int)
    keep = (A["_j"] + H + 1 < n).to_numpy()
    A = A[keep].reset_index(drop=True); p1, p2 = p1[keep], p2[keep]
    j = A["_j"].to_numpy(); lg = A["_long"].to_numpy(bool); e = op[j+1]
    ret = np.where(lg, (cl[j+H]-e)/e, (e-cl[j+H])/e) * 1e4

    y = A["_y"].to_numpy(int); ts = A["_ts"]; tq = A["_tq"].to_numpy(float)
    cnt = (lg & (tq <= TQ_LO)) | (~lg & (tq >= TQ_HI))
    gate = (tq >= TQ_HI) | (tq <= TQ_LO)             # 라이브 gated_of -- 측면 무관
    oos = (ts >= VAL0).to_numpy(); o = np.flatnonzero(oos); k = len(o)//2
    cm = np.zeros(len(A), bool); cm[o[:k]] = True
    ev = np.zeros(len(A), bool); ev[o[k:]] = True
    days = (ts[ev].max()-ts[ev].min()).total_seconds()/86400
    q = ts[ev].min() + (ts[ev].max()-ts[ev].min())/2
    e1 = ev & (ts <= q).to_numpy(); e2 = ev & (ts > q).to_numpy()
    c1 = {g: float(np.quantile(p1[cm], 1-qq)) for g, qq in TIER_Q}
    c2 = {g: float(np.quantile(p2[cm], 1-qq)) for g, qq in TIER_Q}
    print(f"부모 p1 = 배포 TabPFN · 사이드카 p2 = 손실가중 HGB")
    print(f"평가 {ts[ev].min():%m-%d}~{ts[ev].max():%m-%d} ({days:.0f}일) · 기저 {y[ev].mean():.3f}")
    print(f"p1 강컷 {c1['강']:.4f} (배포 meta 0.5200) · p2 강컷 {c2['강']:.4f}\n")

    def grades(mode: str):
        """mode: base | dual_mid(강등->중) | dual_low(강등->약) | dual_drop(강등->미표시)"""
        g = np.full(len(A), "-", object)
        for gg, _ in reversed(TIER_Q): g[p1 >= c1[gg]] = gg
        if mode == "base":
            g[gate] = "-"; return g
        dem = (g == "강") & (p2 < c2["강"])
        g[dem] = {"dual_mid": "중", "dual_low": "약", "dual_drop": "-",
                  "dual_low_gated": "약"}[mode]
        if mode == "dual_low_gated":
            g[gate] = "-"                                 # 전 등급 게이트 유지(보수안)
        else:
            g[gate & np.isin(g, ["중", "약"])] = "-"       # 강만 게이트 면제
        return g

    print(f"{'안':<10}{'등급':<4}{'정밀도':>9}{'전반':>8}{'후반':>8}{'건/일':>8}{'역추세':>8}{'순bp':>9}{'n':>6}")
    res = {}
    for name, mode in (("배포판", "base"), ("v2 강등->중", "dual_mid"),
                       ("v2 강등->약", "dual_low"), ("v2 강등제거", "dual_drop"),
                       ("v2 보수(전게이트)", "dual_low_gated")):
        G = grades(mode)
        for gg, _ in TIER_Q:
            sel = ev & (G == gg)
            if sel.sum() < 10: continue
            res[(name, gg)] = (float(y[sel].mean()), float(cnt[sel].mean()), sel.sum()/days)
            print(f"{name:<10}{gg:<4}{y[sel].mean():>9.4f}{y[e1&(G==gg)].mean():>8.3f}"
                  f"{y[e2&(G==gg)].mean():>8.3f}{sel.sum()/days:>8.2f}{cnt[sel].mean():>8.3f}"
                  f"{ret[sel].mean()-COST:>9.2f}{int(sel.sum()):>6d}")
        print("-"*64)
    print("판정 (사용자 지정: 같은 등급에서 정밀도↑ · 역추세 비중↓)")
    for name in ("v2 강등->중", "v2 강등->약", "v2 강등제거", "v2 보수(전게이트)"):
        ok = 0; tot = 0
        line = []
        for gg, _ in TIER_Q:
            if (name, gg) not in res or ("배포판", gg) not in res: continue
            a, b = res[("배포판", gg)], res[(name, gg)]
            pss = b[0]-a[0] >= 0 and b[1]-a[1] <= 1e-9; ok += pss; tot += 1
            line.append(f"{gg} 정밀도{b[0]-a[0]:+.3f}/역추세{b[1]-a[1]:+.3f}")
        print(f"  {name:<12} {ok}/{tot}   " + "  ".join(line))

    # ⭐건수 매칭 -- 강 등급의 이득이 커버리지 때문인지 배제한다
    print("\n건수 매칭 강 등급 (배포판의 상위 k 개 vs v2 의 상위 k 개, k = 적은 쪽)")
    Gb, Gv = grades("base"), grades("dual_low_gated")
    sb, sv = ev & (Gb == "강"), ev & (Gv == "강")
    kk = int(min(sb.sum(), sv.sum()))
    def topk(mask, score):
        idx = np.flatnonzero(mask); keep = idx[np.argsort(score[idx])[::-1][:kk]]
        m = np.zeros(len(A), bool); m[keep] = True; return m
    tb, tv = topk(sb, p1), topk(sv, np.minimum(p1, p2))
    print(f"  k={kk}  배포판 {y[tb].mean():.4f}  vs  v2 {y[tv].mean():.4f}  "
          f"차이 {y[tv].mean()-y[tb].mean():+.4f}")
    # 겹침
    both = tb & tv
    onlyb, onlyv = tb & ~tv, tv & ~tb
    print(f"  두 안이 공통으로 고른 건수 {int(both.sum())}/{kk} ({both.sum()/kk*100:.0f}%)")
    print(f"  공통     n={int(both.sum()):3d}  정밀도 {y[both].mean():.4f}")
    print(f"  배포판만 n={int(onlyb.sum()):3d}  정밀도 {y[onlyb].mean():.4f}   <- v2 가 버린 것")
    print(f"  v2만     n={int(onlyv.sum()):3d}  정밀도 {y[onlyv].mean():.4f}   <- v2 가 새로 고른 것")
    from scipy import stats as st
    a_, b_ = int(y[onlyb].sum()), int(y[onlyv].sum())
    print(f"  불일치쌍 정확 이항검정(배포판 {a_} vs v2 {b_} 적중): "
          f"p = {st.binomtest(b_, a_+b_, 0.5).pvalue:.4f}")
    print(f"  두 반기 (건수매칭): 배포판 {y[tb&e1].mean():.3f}/{y[tb&e2].mean():.3f}  "
          f"v2 {y[tv&e1].mean():.3f}/{y[tv&e2].mean():.3f}")
    # 강 등급의 역추세 콜이 실제로 좋은가 (게이트 면제의 근거)
    Gv2 = grades("dual_low"); sv2 = ev & (Gv2 == "강") & cnt
    if sv2.sum() >= 5:
        print(f"\n강 등급 안의 역추세 콜: n={int(sv2.sum())} 정밀도 {y[sv2].mean():.3f} "
              f"(강 전체 {y[ev&(Gv2=='강')].mean():.3f}) 순bp {ret[sv2].mean()-COST:+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
