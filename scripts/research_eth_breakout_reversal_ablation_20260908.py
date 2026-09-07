#!/usr/bin/env python3
"""돌파/되돌림 정확도 -- **무엇이 올렸는가** 피쳐군 절제 + 중요도 (2026-09-08).

v1(발현 직전 봉만, 59피쳐) 50~54% -> v2(+발현 경로·레벨맥락·교차자산, 77피쳐) **53~57%**.
어느 피쳐군이 올렸는지 절제 실험으로 분해하고, 순열 중요도로 개별 피쳐를 본다.
같은 워크포워드(월 재학습·엠바고 4h)·같은 창·일군집 CI.
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
ANCH, TM = "first_fire", 0.75
SEED = 20260908
BOOT = 3000


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def wf(X, y, ts, months, uniq, seed=SEED):
    from sklearn.ensemble import HistGradientBoostingClassifier
    pred = np.full(len(y), np.nan)
    for i, mo in enumerate(uniq):
        if i < 6: continue
        te = (months == mo).to_numpy()
        cut = ts[te].min() - EMB
        tr = (ts < cut).to_numpy()
        if tr.sum() < 2000 or te.sum() < 30: continue
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.15, random_state=seed)
        c.fit(X[tr], y[tr]); pred[te] = c.predict_proba(X[te])[:, 1]
    return pred


def main() -> int:
    rng = np.random.default_rng(SEED)
    from sklearn.metrics import roc_auc_score
    A = pd.read_parquet(SRC); A["timestamp"] = pd.to_datetime(A["timestamp"])
    d = A[(A.anchor == ANCH) & (A.T_mult == TM)].sort_values("timestamp").reset_index(drop=True)
    y = d["y"].to_numpy(int); ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy()
    sp = d["split"].to_numpy(); months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    ALL = [c for c in d.columns if c.startswith(("f_", "x_", "sig_", "v2_"))] + \
          ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
    G = {
        "v1_봉": [c for c in ALL if c.startswith(("f_", "sig_")) or c in
                 ("dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom")],
        "x_횡단면": [c for c in ALL if c.startswith("x_")],
        "v2_경로": [c for c in ALL if c.startswith("v2_mv_") and "btc" not in c and "idio" not in c
                  and "same_sign" not in c],
        "v2_레벨": [c for c in ALL if c.startswith(("v2_brk", "v2_pos"))],
        "v2_교차": [c for c in ALL if c.startswith("v2_mv_") and ("btc" in c or "idio" in c
                                                               or "same_sign" in c)],
    }
    print(f"이벤트 {len(d):,} · 피쳐군 " + " ".join(f"{k}({len(v)})" for k, v in G.items()), flush=True)

    combos = [("v1 만", ["v1_봉"]),
              ("v1+횡단면", ["v1_봉", "x_횡단면"]),
              ("v1+경로", ["v1_봉", "v2_경로"]),
              ("v1+레벨", ["v1_봉", "v2_레벨"]),
              ("v1+교차", ["v1_봉", "v2_교차"]),
              ("v1+경로+레벨", ["v1_봉", "v2_경로", "v2_레벨"]),
              ("전체", list(G)),
              ("경로+레벨+교차만", ["v2_경로", "v2_레벨", "v2_교차"])]
    print("\n" + "=" * 104)
    print(f"{'구성':>18}{'피쳐':>5} | " + " | ".join(f"{w[:8]:>25}" for w in WINS))
    print("=" * 104)
    best = None
    for name, ks in combos:
        cols = sorted({c for k in ks for c in G[k]})
        X = d[cols].to_numpy(np.float32)
        pred = wf(X, y, ts, months, uniq)
        ok = np.isfinite(pred)
        line = f"{name:>18}{len(cols):>5} | "
        for w in WINS:
            m = ok & (sp == w)
            acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
            lo, hi = day_ci(acc, day[m], rng)
            auc = roc_auc_score(y[m], pred[m])
            line += f"{acc.mean():.4f}[{lo:.4f},{hi:.4f}] A{auc:.3f} | "
        print(line, flush=True)
        if name == "전체": best = (cols, pred)

    print("\n=== 순열 중요도 (전체 구성, 표본외 3창 합산 정확도 하락폭) ===", flush=True)
    cols, pred = best
    X = d[cols].to_numpy(np.float32)
    m = np.isfinite(pred) & np.isin(sp, WINS)
    base = ((pred[m] > 0.5).astype(int) == y[m]).mean()
    from sklearn.ensemble import HistGradientBoostingClassifier
    # 마지막 폴드 모델 하나로 순열 중요도 (근사)
    te = (months == uniq[-2]).to_numpy(); cut = ts[te].min() - EMB; tr = (ts < cut).to_numpy()
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.15, random_state=SEED).fit(X[tr], y[tr])
    ev = np.isin(sp, WINS)
    p0 = clf.predict_proba(X[ev])[:, 1]
    a0 = ((p0 > 0.5).astype(int) == y[ev]).mean()
    imp = []
    Xe = X[ev].copy()
    for j, c in enumerate(cols):
        sv = Xe[:, j].copy()
        Xe[:, j] = rng.permutation(sv)
        a = ((clf.predict_proba(Xe)[:, 1] > 0.5).astype(int) == y[ev]).mean()
        Xe[:, j] = sv
        imp.append((c, a0 - a))
    imp.sort(key=lambda t: -t[1])
    for c, v in imp[:18]: print(f"   {c:>24} {v:+.4f}")
    print(f"\n워크포워드 전체 정확도(3창 합산) {base:.4f}")
    print(json.dumps({"n": len(d)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
