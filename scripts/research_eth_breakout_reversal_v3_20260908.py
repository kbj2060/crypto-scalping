#!/usr/bin/env python3
"""돌파/되돌림 **v3 평가 -- 체결(CVD/흡수)·미결제약정·호가** (2026-09-08).

부록 AM 의 규칙을 그대로 따른다:
  ⭐**단변량 분위 효과를 모델보다 먼저 본다.** 모델 정확도 − 최고 단변량 효과가 크게 벌어지면
    상호작용이 아니라 누수를 먼저 의심한다(1분 미래참조 사고).

프로토콜은 `research_eth_breakout_reversal_final_20260908.py` 와 동일하다
(월단위 walk-forward · 엠바고 4h · HGB · 일군집 부트스트랩 CI · 라벨셔플 대조군).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp/eth_breakout_reversal_20260908/dataset_v3.parquet"
OUT = ROOT / "tmp/eth_breakout_reversal_20260908"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
EMB = pd.Timedelta(hours=4)
SEED = 20260908
BOOT = 3000
CELLS = [("first_fire", 0.75), ("first_fire", 1.0), ("any2/Wc3", 0.75)]


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def wf(X, y, ts, months, uniq, shuffle=False, rng=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    pred = np.full(len(y), np.nan)
    for i, mo in enumerate(uniq):
        if i < 6: continue
        te = (months == mo).to_numpy(); cut = ts[te].min() - EMB; tr = (ts < cut).to_numpy()
        if tr.sum() < 2000 or te.sum() < 30: continue
        yy = y.copy()
        if shuffle: yy[tr] = rng.permutation(yy[tr])
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.15, random_state=SEED)
        c.fit(X[tr], yy[tr]); pred[te] = c.predict_proba(X[te])[:, 1]
    return pred


BASE_EXTRA = ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]


def base_cols(d):
    c = [x for x in d.columns if x.startswith(("f_", "sig_", "v2_")) and not x.startswith("x_")]
    return c + BASE_EXTRA


def main() -> int:
    rng = np.random.default_rng(SEED)
    from sklearn.metrics import roc_auc_score
    A = pd.read_parquet(SRC); A["timestamp"] = pd.to_datetime(A["timestamp"])
    V3C = [c for c in A.columns if c.startswith("v3c_")]
    V3O = [c for c in A.columns if c.startswith("v3o_")]
    V3B = [c for c in A.columns if c.startswith("v3b_")]

    print("=" * 112)
    print("0) 커버리지 (X4 규칙: 어떤 모집단인지 먼저 밝힌다)")
    print("=" * 112)
    for anch, TM in CELLS:
        d = A[(A.anchor == anch) & (A.T_mult == TM)]
        e = d[d.split.isin(WINS)]
        print(f"   {anch:>11} T={TM:<5} 표본외 n={len(e):>6} | 경로보유 {e.v3c_path_known.mean():.3f} "
              f"| 호가가용 {e.v3b_ok.mean():.3f} | OI가용 {e.v3o_doi3.notna().mean():.3f} "
              f"| 돌파율(기저) {e.y.mean():.4f}")

    print("\n" + "=" * 112)
    print("1) ⭐단변량 분위별 실제 돌파율 -- 모델보다 먼저 (first_fire T=0.75, 표본외 3창)")
    print("=" * 112)
    d0 = A[(A.anchor == "first_fire") & (A.T_mult == 0.75)]
    e0 = d0[d0.split.isin(WINS)]
    NAMES = {
        "v3c_burn": "⭐흡수계수 CVD/이동폭 (↑=많이 태움)", "v3c_cvd_frac": "CVD 비중(방향정렬)",
        "v3c_cvd_r": "CVD 강도(기준선비)", "v3c_flow_r": "거래대금 강도", "v3c_align": "CVD 방향일치 분비율",
        "v3c_kyle_r": "카일람다(가격영향/거래)", "v3c_impact_r": "거래량당 가격영향",
        "v3c_xl_frac": "초대형 체결 순방향", "v3c_lg_imb": "대형 체결 불균형", "v3c_cvd_div": "CVD 다이버전스(후−전)",
        "v3c_last_imb": "트리거 직전분 테이커불균형", "v3c_switch": "매수매도 전환율", "v3c_avgsz_r": "평균체결크기비",
        "v3o_doi1": "ΔOI 5분", "v3o_doi3": "ΔOI 15분", "v3o_doi6": "ΔOI 30분", "v3o_doi12": "ΔOI 1시간",
        "v3o_doi_acc": "ΔOI 가속(15분−1시간)",
        "v3b_wall1": "앞쪽 벽 비중 1%", "v3b_wall02": "앞쪽 벽 비중 0.2%", "v3b_ahead_r": "앞쪽 깊이/기준선",
        "v3b_ahead_drop": "⭐앞쪽 벽 취소율(스푸핑)", "v3b_behind_r": "뒤쪽 깊이/기준선",
        "v3b_tot_r": "전체 깊이/기준선", "v3b_near_conc": "근접 집중도",
    }
    uni = []
    for f, nm in NAMES.items():
        v = e0[f].to_numpy(float)
        if np.isfinite(v).sum() < 2000: continue
        q = pd.qcut(pd.Series(v), 5, labels=False, duplicates="drop")
        g = pd.DataFrame({"q": q, "y": e0["y"].to_numpy()}).groupby("q")["y"].agg(["mean", "count"])
        if len(g) < 5: continue
        d51 = g["mean"].iloc[-1] - g["mean"].iloc[0]
        s = " ".join(f"Q{int(i)+1} {r['mean']:.3f}" for i, r in g.iterrows())
        uni.append((abs(d51), f, nm, d51, s, int(g["count"].sum())))
    uni.sort(reverse=True)
    for a, f, nm, d51, s, n in uni:
        flag = "⭐" if a >= 0.03 else "  "
        print(f" {flag}{nm:>32}: {s}  (Q5−Q1 {d51:+.3f}, n={n:,})")
    best_uni = uni[0] if uni else None
    print(f"\n   최고 단변량 |Q5−Q1| = {best_uni[0]:.3f} ({best_uni[2]})  "
          f"-> 모델 없는 규칙 정확도 상한 ≈ {0.5 + best_uni[0]/4:.4f}")

    print("\n" + "=" * 112)
    print("2) 절제: 기저(v1+v2) vs +CVD / +OI / +호가 / +전체")
    print("=" * 112)
    res = []
    for anch, TM in CELLS:
        d = A[(A.anchor == anch) & (A.T_mult == TM)].sort_values("timestamp").reset_index(drop=True)
        if len(d) < 4000: continue
        BC = base_cols(d)
        SETS = {"기저(v1+v2)": BC, "+CVD/흡수": BC + V3C, "+OI": BC + V3O,
                "+호가": BC + V3B, "+전체v3": BC + V3C + V3O + V3B}
        y = d["y"].to_numpy(int); ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy()
        sp = d["split"].to_numpy(); months = ts.dt.to_period("M"); uniq = sorted(months.unique())
        print(f"\n   --- {anch} T={TM} (n={len(d):,}) ---")
        for nm, cols in SETS.items():
            X = d[cols].to_numpy(np.float32)
            pred = wf(X, y, ts, months, uniq)
            ok = np.isfinite(pred)
            line = f"   {nm:>12}({len(cols):>3}) | "
            row = dict(anchor=anch, T=TM, fset=nm, ncols=len(cols))
            for w in WINS:
                m = ok & (sp == w)
                acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
                lo, hi = day_ci(acc, day[m], rng)
                line += f"{w[:4]} {acc.mean():.4f}[{lo:.4f},{hi:.4f}] A{roc_auc_score(y[m], pred[m]):.3f} | "
                row |= {f"{w}_acc": float(acc.mean()), f"{w}_lo": lo, f"{w}_hi": hi,
                        f"{w}_auc": float(roc_auc_score(y[m], pred[m])), f"{w}_n": int(m.sum())}
            print(line, flush=True)
            res.append(row)
            if anch == "first_fire" and TM == 0.75:
                np.save(OUT / f"predv3_{nm.replace('/','_')}.npy", pred)
    R = pd.DataFrame(res); R.to_csv(OUT / "v3_ablation.csv", index=False)

    print("\n" + "=" * 112)
    print("3) 라벨 셔플 대조군 (최고 구성)")
    print("=" * 112)
    d = A[(A.anchor == "first_fire") & (A.T_mult == 0.75)].sort_values("timestamp").reset_index(drop=True)
    BC = base_cols(d); cols = BC + V3C + V3O + V3B
    y = d["y"].to_numpy(int); ts = d["timestamp"]; sp = d["split"].to_numpy()
    months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    X = d[cols].to_numpy(np.float32)
    for k in range(2):
        pc = wf(X, y, ts, months, uniq, shuffle=True, rng=rng)
        mm = np.isfinite(pc)
        s = " ".join(f"{w[:4]} {(((pc[mm&(sp==w)]>0.5).astype(int))==y[mm&(sp==w)]).mean():.4f}" for w in WINS)
        print(f"   셔플 #{k+1}: {s}")
    print(json.dumps({"cells": len(res)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
