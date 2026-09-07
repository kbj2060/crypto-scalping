#!/usr/bin/env python3
"""지연관측 팔 평가 -- **돌파 직후 OI·호가 반응**이 되돌림을 알려주는가 (2026-09-08).

s1 시점 팔에서 OI(ΔOI Q5−Q1 +0.005~0.009)와 호가는 사실상 평평했다. 그러나 사용자가 말한
메커니즘은 **돌파 이후의 반응**이다(OI 급감 = 청산연료 소진, 반대 벽 형성 = 유동성 사냥).
이 스크립트는 의사결정을 D분 늦추고 그 반응을 실제로 관측한 뒤 정확도를 잰다.

프로토콜은 s1 팔과 동일(월 walk-forward · 엠바고 4h · HGB · 일군집 CI · 셔플 대조군).
⭐단변량을 모델보다 먼저 본다.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "tmp/eth_breakout_reversal_20260908/dataset_v3.parquet"
DL = ROOT / "tmp/eth_breakout_reversal_20260908/dataset_delayed.parquet"
OUT = ROOT / "tmp/eth_breakout_reversal_20260908"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
EMB = pd.Timedelta(hours=4)
SEED = 20260908
BOOT = 3000
KEY = ["bar_idx", "anchor", "side_bottom", "T_mult"]
BASE_EXTRA = ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
NAMES = {"v4_doi_post": "⭐돌파직후 ΔOI (↓=청산연료 소진)", "v4_behind_build": "⭐반대 벽 형성(유동성 사냥)",
         "v4_ahead_chg": "앞쪽 깊이 변화", "v4_wall_post": "앞쪽 벽 비중(사후)",
         "v4_cvd_post": "돌파직후 CVD 강도", "v4_flow_post": "돌파직후 거래대금",
         "v4_kyle_post": "돌파직후 카일람다", "v4_impact_post": "돌파직후 가격영향",
         "v4_ret_post": "돌파후 추가진행(ATR)", "v4_mae_post": "돌파후 역행폭(ATR)",
         "v4_move_tot": "앵커대비 총이동(ATR)"}


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
    A = pd.read_parquet(V3); A["timestamp"] = pd.to_datetime(A["timestamp"])
    L = pd.read_parquet(DL)
    fcols = [c for c in A.columns if c.startswith(("f_", "sig_", "v2_", "v3c_", "v3o_", "v3b_"))
             and not c.startswith("x_")] + BASE_EXTRA
    fcols = [c for c in dict.fromkeys(fcols) if c not in KEY]
    A2 = A[KEY + fcols].drop_duplicates(KEY)
    res = []
    for Dm in (5, 10):
        d = L[(L.delay == Dm) & (L.valid_d == 1) & (L.anchor == "first_fire") & (L.T_mult == 0.75)]
        d = d.merge(A2, on=KEY, how="left", validate="one_to_one")
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        d = d.sort_values("timestamp").reset_index(drop=True)
        V4 = [c for c in d.columns if c.startswith("v4_")]
        e = d[d.split.isin(WINS)]
        print("=" * 112)
        print(f"D={Dm}분 지연 · first_fire T=0.75 | 표본외 n={len(e):,} 돌파율(기저) {e.y_d.mean():.4f} "
              f"| OI관측 {e.v4_doi_post.notna().mean():.3f} 호가관측 {e.v4_bk_ok.mean():.3f}")
        print("=" * 112)
        print(f"  ⭐단변량 분위별 실제 돌파율 (표본외 3창) -- 모델보다 먼저")
        uni = []
        for f, nm in NAMES.items():
            v = e[f].to_numpy(float)
            if np.isfinite(v).sum() < 1500: continue
            q = pd.qcut(pd.Series(v), 5, labels=False, duplicates="drop")
            g = pd.DataFrame({"q": q, "y": e["y_d"].to_numpy()}).groupby("q")["y"].agg(["mean", "count"])
            if len(g) < 5: continue
            d51 = g["mean"].iloc[-1] - g["mean"].iloc[0]
            nq = int(g["count"].mean()); se = float(np.sqrt(2 * 0.25 / nq))
            uni.append((abs(d51) / max(se, 1e-9), f, nm, d51, se,
                        " ".join(f"Q{int(i)+1} {r['mean']:.3f}" for i, r in g.iterrows()),
                        int(g["count"].sum())))
        uni.sort(reverse=True)
        for t, f, nm, d51, se, s, n in uni:
            flag = "⭐" if t >= 2 else "  "
            print(f"  {flag}{nm:>28}: {s}  (Q5−Q1 {d51:+.3f} ±{2*se:.3f}[2SE], t={t:.1f}, n={n:,})")
        y = d["y_d"].to_numpy(int); ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy()
        sp = d["split"].to_numpy(); months = ts.dt.to_period("M"); uniq = sorted(months.unique())
        SETS = {"기저(s1시점)": fcols, "+사후 전체": fcols + V4,
                "+사후 OI만": fcols + ["v4_doi_post", "v4_doi_bars"],
                "+사후 호가만": fcols + ["v4_ahead_chg", "v4_behind_build", "v4_wall_post", "v4_bk_ok"],
                "사후만(단독)": V4 + BASE_EXTRA}
        print(f"\n  절제:")
        for nm, cols in SETS.items():
            cols = [c for c in cols if c in d.columns]
            X = d[cols].to_numpy(np.float32)
            pred = wf(X, y, ts, months, uniq)
            ok = np.isfinite(pred)
            line = f"  {nm:>13}({len(cols):>3}) | "
            row = dict(delay=Dm, fset=nm, ncols=len(cols))
            for w in WINS:
                m = ok & (sp == w)
                acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
                lo, hi = day_ci(acc, day[m], rng)
                line += f"{w[:4]} {acc.mean():.4f}[{lo:.4f},{hi:.4f}] A{roc_auc_score(y[m], pred[m]):.3f} | "
                row |= {f"{w}_acc": float(acc.mean()), f"{w}_lo": lo, f"{w}_hi": hi}
            print(line, flush=True)
            res.append(row)
        if Dm == 5:
            X = d[[c for c in fcols + V4 if c in d.columns]].to_numpy(np.float32)
            for k in range(2):
                pc = wf(X, y, ts, months, uniq, shuffle=True, rng=rng)
                mm = np.isfinite(pc)
                s = " ".join(f"{w[:4]} {(((pc[mm&(sp==w)]>0.5).astype(int))==y[mm&(sp==w)]).mean():.4f}"
                             for w in WINS)
                print(f"  셔플 대조군 #{k+1}: {s}")
    pd.DataFrame(res).to_csv(OUT / "delayed_ablation.csv", index=False)
    print(json.dumps({"cells": len(res)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
