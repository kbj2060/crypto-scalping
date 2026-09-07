#!/usr/bin/env python3
"""돌파/되돌림 **최종 구성 + 해석** (2026-09-08).

절제 실험 결과 최고 구성은 **v1봉 + 발현경로 + 레벨맥락 (53피쳐)** 이고 횡단면 21피쳐는 노이즈였다.
순열 중요도 상위 5개가 전부 **발현 경로** 피쳐다:
  직진성 +0.054 · 최대1분봉 +0.049 · 발현속도 +0.033 · 진행방향비율 +0.029 · 역행폭 +0.025
⇒ *"그 움직임이 어떻게 만들어졌는가"* 가 돌파/되돌림을 가른다.

이 스크립트: (1) 최종 구성 재확인 + 셔플 대조군 (2) **효과의 방향**을 사람이 읽을 수 있게
직진성/속도 분위별 실제 돌파율을 낸다 (3) 모델 없는 2피쳐 규칙이 얼마나 되는지.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp/eth_breakout_reversal_20260908/dataset_v2.parquet"
OUT = ROOT / "tmp/eth_breakout_reversal_20260908"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
EMB = pd.Timedelta(hours=4)
SEED = 20260908
BOOT = 3000


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


def main() -> int:
    rng = np.random.default_rng(SEED)
    from sklearn.metrics import roc_auc_score
    A = pd.read_parquet(SRC); A["timestamp"] = pd.to_datetime(A["timestamp"])
    print("=" * 108)
    print("1) 최종 구성 (v1봉 + 발현경로 + 레벨맥락) · 앵커·트리거별")
    print("=" * 108)
    res = []
    for anch in ("first_fire", "any2/Wc3"):
        for TM in (0.5, 0.75, 1.0):
            d = A[(A.anchor == anch) & (A.T_mult == TM)].sort_values("timestamp").reset_index(drop=True)
            if len(d) < 4000: continue
            cols = [c for c in d.columns if c.startswith(("f_", "sig_")) or
                    c.startswith("v2_")] + ["dir_up", "trig_min", "T_atr", "atr_at_anchor",
                                            "n_signals", "side_bottom"]
            cols = [c for c in cols if not c.startswith("x_")]
            X = d[cols].to_numpy(np.float32); y = d["y"].to_numpy(int)
            ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy(); sp = d["split"].to_numpy()
            months = ts.dt.to_period("M"); uniq = sorted(months.unique())
            pred = wf(X, y, ts, months, uniq)
            predc = wf(X, y, ts, months, uniq, shuffle=True, rng=rng)
            ok = np.isfinite(pred)
            line = f"{anch:>11} T={TM:<5}({len(cols)}피쳐) | "
            for w in WINS:
                m = ok & (sp == w)
                acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
                lo, hi = day_ci(acc, day[m], rng)
                mc = np.isfinite(predc) & (sp == w)
                ca = ((predc[mc] > 0.5).astype(int) == y[mc]).mean()
                line += (f"{w[:4]} {acc.mean():.4f}[{lo:.4f},{hi:.4f}] A{roc_auc_score(y[m], pred[m]):.3f} "
                         f"기저{max(y[m].mean(),1-y[m].mean()):.3f} 셔플{ca:.3f} | ")
                res.append(dict(anchor=anch, T=TM, win=w, n=int(m.sum()), acc=float(acc.mean()),
                                lo=lo, hi=hi, auc=float(roc_auc_score(y[m], pred[m])),
                                base=float(max(y[m].mean(), 1 - y[m].mean())), shuf=float(ca)))
            print(line, flush=True)
            if anch == "first_fire" and TM == 0.75:
                np.save(OUT / "pred_final.npy", pred); d.to_parquet(OUT / "final_events.parquet")
    pd.DataFrame(res).to_csv(OUT / "final_accuracy.csv", index=False)

    print("\n" + "=" * 108)
    print("2) ⭐효과의 방향 -- 발현 경로 분위별 실제 돌파율 (first_fire · T=0.75 · 표본외 3창)")
    print("=" * 108)
    d = pd.read_parquet(OUT / "final_events.parquet")
    e = d[d.split.isin(WINS)]
    for f, nm in (("v2_mv_straight", "직진성(1=일직선)"), ("f_speed", "발현속도(ATR/분)"),
                  ("v2_mv_mae_atr", "발현중 역행폭(ATR)"), ("v2_mv_updown", "진행방향 1분봉 비율"),
                  ("v2_mv_maxbar_atr", "최대 1분봉(ATR)")):
        v = e[f].to_numpy(float)
        q = pd.qcut(pd.Series(v), 5, labels=False, duplicates="drop")
        g = pd.DataFrame({"q": q, "y": e["y"].to_numpy()}).groupby("q")["y"].agg(["mean", "count"])
        s = " ".join(f"Q{int(i)+1} {r['mean']:.3f}" for i, r in g.iterrows())
        print(f"   {nm:>22} 돌파율: {s}   (Q5−Q1 {g['mean'].iloc[-1]-g['mean'].iloc[0]:+.3f})")

    print("\n" + "=" * 108)
    print("3) 모델 없는 2피쳐 규칙 (직진성 × 역행폭 중앙값 분할, 표본외 3창)")
    print("=" * 108)
    st = e["v2_mv_straight"].to_numpy(float); mae = e["v2_mv_mae_atr"].to_numpy(float)
    ms, mm = np.nanmedian(st), np.nanmedian(mae)
    for a, an in ((st >= ms, "직진↑"), (st < ms, "직진↓")):
        for b, bn in ((mae < mm, "역행↓"), (mae >= mm, "역행↑")):
            m = a & b
            if m.sum() < 100: continue
            br = e["y"].to_numpy()[m].mean()
            call = "돌파" if br > 0.5 else "되돌림"
            print(f"   {an}·{bn}: n={m.sum():>5} 돌파율 {br:.4f} -> **{call}** 예측시 정확도 "
                  f"{max(br,1-br):.4f}")
    print(json.dumps({"cells": len(res)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
