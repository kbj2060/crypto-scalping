#!/usr/bin/env python3
"""OI/자금흐름 피쳐의 순수 기여 -- **수정판 v2 위에서** (2026-09-08).

v3(관찰창 OBS>=5)는 24셀 전부 우연이었다. 이유가 분명하다: 기준가를 `close[s2-1]` 로 바꾸면
질문이 *"지금 가격에서 어느 쪽이 먼저인가"* 가 되어 **부록 Z·AA 가 이미 닫은 대칭 방향 문제**로
돌아간다. 트리거 직후의 즉시성이 곧 신호였고, 기다리면 그게 사라진다.

그래서 OI 의 기여는 **+2pp 가 나왔던 수정판 v2 구성**(결정=트리거 시점, 기준=트리거 레벨) 위에서
재본다. v2 데이터셋은 `x_m_oi_z / x_m_oi_d12 / x_m_oi_d48` 을 갖고 있는데 최종 구성에서 `x_` 를
통째로 뺐으므로 **OI 는 한 번도 단독 검정된 적이 없다**.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp/eth_breakout_reversal_20260908/dataset_v2.parquet"
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


def wf(X, y, ts, months, uniq):
    from sklearn.ensemble import HistGradientBoostingClassifier
    pred = np.full(len(y), np.nan)
    for i, mo in enumerate(uniq):
        if i < 6: continue
        te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
        if tr.sum() < 2000 or te.sum() < 30: continue
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.15, random_state=SEED)
        c.fit(X[tr], y[tr]); pred[te] = c.predict_proba(X[te])[:, 1]
    return pred


def main() -> int:
    rng = np.random.default_rng(SEED)
    from sklearn.metrics import roc_auc_score
    A = pd.read_parquet(SRC); A["timestamp"] = pd.to_datetime(A["timestamp"])
    print("=" * 108)
    for anch, TM in (("first_fire", 0.75), ("first_fire", 1.0)):
        d = A[(A.anchor == anch) & (A.T_mult == TM)].sort_values("timestamp").reset_index(drop=True)
        y = d["y"].to_numpy(int); ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy()
        sp = d["split"].to_numpy(); months = ts.dt.to_period("M"); uniq = sorted(months.unique())
        base = [c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] + \
               ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
        oi = [c for c in d.columns if c.startswith("x_m_oi")]
        flow = [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))]
        print(f"\n■ {anch} T={TM} · n={len(d):,} · OI {len(oi)}개 {oi} · 자금흐름 {len(flow)}개")
        for nm, cols in (("기준(v2 수정판)", base), ("+OI", base + oi),
                         ("+자금흐름", base + flow), ("+OI+자금흐름", base + oi + flow)):
            pred = wf(d[cols].to_numpy(np.float32), y, ts, months, uniq)
            line = f"   {nm:>16}({len(cols):>3}) | "
            for w in WINS:
                m = np.isfinite(pred) & (sp == w)
                acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
                lo, hi = day_ci(acc, day[m], rng)
                b = max(y[m].mean(), 1 - y[m].mean())
                line += (f"{w[:4]} {acc.mean():.4f}[{lo:.4f}] A{roc_auc_score(y[m], pred[m]):.3f} "
                         f"기저{b:.3f} | ")
            print(line, flush=True)
    print(json.dumps({"done": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
