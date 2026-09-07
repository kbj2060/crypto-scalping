#!/usr/bin/env python3
"""돌파/되돌림 v3 모델 -- 관찰창 × 완화라벨 × OI (2026-09-08).

⚠️**OBS=0 은 무효**: 트리거는 분 `s1` **안에서** 발생하는데 기준가는 `cl1[s1-1]`(그 이전 종가)라,
트리거를 만든 움직임 자체가 배리어를 때린다. 돌파율이 0.73/0.68/0.63 으로 튀는 것이 그 증거다
(OBS>=5 는 0.49~0.51). 트리거를 **알게 되는 시점은 분 s1 종료 후**이므로 OBS>=1 만 유효하다.

판정: 세 창(VAL/OOS/HOLDOUT) 정확도 CI 하한 > 기저. 셔플 대조군·경계 트립와이어 동반.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp/eth_breakout_reversal_20260908/dataset_v3.parquet"
OUT = ROOT / "tmp/eth_breakout_reversal_20260908"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
OBS_OK = (5, 10, 15, 30)
P_LAB = (25, 35, 50)
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
        te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
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
    FE = [c for c in A.columns if c.startswith(("f_", "g_", "sig_"))] + \
         ["dir_up", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
    print(f"데이터 {A.shape} · 피쳐 {len(FE)}\n", flush=True)
    print("=" * 120)
    rows = []
    for anch in ("first_fire", "any2/Wc3"):
        for OBS in OBS_OK:
            d = A[(A.anchor == anch) & (A.OBS == OBS)].sort_values("timestamp").reset_index(drop=True)
            if len(d) < 4000: continue
            ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy(); sp = d["split"].to_numpy()
            months = ts.dt.to_period("M"); uniq = sorted(months.unique())
            X = d[FE].to_numpy(np.float32)
            for Pl in P_LAB:
                y = d[f"y_p{Pl}"].to_numpy(int)
                pred = wf(X, y, ts, months, uniq)
                line = f"{anch:>11} OBS={OBS:>2}분 P={Pl/100:.2f}% | "
                rec = dict(anchor=anch, OBS=OBS, P=Pl)
                for w in WINS:
                    m = np.isfinite(pred) & (sp == w)
                    if m.sum() < 100: continue
                    acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
                    lo, hi = day_ci(acc, day[m], rng)
                    base = max(y[m].mean(), 1 - y[m].mean())
                    line += (f"{w[:4]} {acc.mean():.4f}[{lo:.4f}] A{roc_auc_score(y[m], pred[m]):.3f} "
                             f"기저{base:.3f} | ")
                    rec[f"{w}_acc"] = float(acc.mean()); rec[f"{w}_lo"] = lo
                    rec[f"{w}_base"] = float(base)
                    rec[f"{w}_auc"] = float(roc_auc_score(y[m], pred[m]))
                print(line, flush=True)
                rows.append(rec)
    R = pd.DataFrame(rows); R.to_csv(OUT / "v3_accuracy.csv", index=False)
    R["pass3"] = [all(R.loc[i, f"{w}_lo"] > R.loc[i, f"{w}_base"] for w in WINS) for i in R.index]
    print("\n" + "=" * 120)
    print(f"⭐세 창 모두 CI 하한 > 기저: {int(R.pass3.sum())}/{len(R)}")
    if R.pass3.any():
        print(R[R.pass3][["anchor", "OBS", "P"] + [f"{w}_acc" for w in WINS]
                         + [f"{w}_lo" for w in WINS]].round(4).to_string(index=False))
    R["m"] = R[[f"{w}_acc" for w in WINS]].min(1) - R[[f"{w}_base" for w in WINS]].max(1)
    print("\n=== 세 창 최소 초과정확도 상위 8 ===")
    print(R.sort_values("m", ascending=False).head(8)
          [["anchor", "OBS", "P"] + [f"{w}_acc" for w in WINS] + [f"{w}_auc" for w in WINS] + ["m"]]
          .round(4).to_string(index=False))
    # 최고 셀 셔플 대조군
    b = R.sort_values("m", ascending=False).iloc[0]
    d = A[(A.anchor == b.anchor) & (A.OBS == b.OBS)].sort_values("timestamp").reset_index(drop=True)
    y = d[f"y_p{int(b.P)}"].to_numpy(int); ts = d["timestamp"]; sp = d["split"].to_numpy()
    months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    pc = wf(d[FE].to_numpy(np.float32), y, ts, months, uniq, True, np.random.default_rng(1))
    sh = {w: float(((pc[np.isfinite(pc) & (sp == w)] > 0.5).astype(int)
                    == y[np.isfinite(pc) & (sp == w)]).mean()) for w in WINS}
    print(f"\n최고 셀 셔플 대조군 {b.anchor} OBS={int(b.OBS)} P={int(b.P)}: "
          + " ".join(f"{w[:4]} {v:.4f}" for w, v in sh.items()))
    print(json.dumps({"cells": len(R), "pass3": int(R.pass3.sum())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
