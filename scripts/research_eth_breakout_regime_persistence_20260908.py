#!/usr/bin/env python3
"""**사후 분석(regime persistence)** -- 되돌림률 자체가 시간적으로 지속되는가 (2026-09-08).

사용자: *"예측을 하지 말고 사후 분석을 하는건 어때? 지금은 역사적으로 봤을 때 되돌리지 않는
구간이라는 걸 미래 봉이 아닌 현재 봉을 분석하는거지"*

## 먼저 정직하게: 무엇이 같고 무엇이 새로운가
- **같은 것**: "이 상태에서 역사적으로 무슨 일이 있었나"는 지도학습 모델이 이미 계산한다.
  워크포워드 HGB 가 하는 게 정확히 그 조회다. 같은 피쳐로 조회 방식만 바꾸면 52~54% 에서 안 움직인다.
- ⭐**새로운 것**: *"지금은 구간이다"* 는 **사건 하나가 아니라 시간에 걸쳐 지속되는 상태**를 말한다.
  그러면 질문이 바뀐다 -- **최근에 되돌림이 많았으면 다음도 되돌림인가?**
  이건 사건별 피쳐가 아니라 **라벨 자체의 자기상관**이고, 한 번도 검정한 적이 없다.

## 🔴인과성 (부록 AM·계약 준수)
"최근 되돌림률"은 **그 시점에 이미 확정된** 라벨만 써야 한다. 사건의 라벨은 최대 H=48봉(4h) 뒤에
확정되므로, 시각 t 의 후행률은 **t − 4h 이전에 발동한 사건만** 쓴다(보수적: 조기 해소도 4h 로 간주).
이 지연을 빼먹으면 그 자체가 미래참조다.

## 검정
A. 라벨 자기상관 -- 후행 K건/기간의 돌파율이 현재 라벨과 상관이 있는가
B. 규칙 -- "후행률이 가리키는 쪽" 을 그대로 따르면 정확도가 얼마인가 (모델 없음)
C. 모델 -- 후행률 피쳐를 넣으면 기존 52~54% 가 올라가는가
D. 레짐 버킷 -- 변동성·세션 버킷별 역사적 돌파율이 창을 넘어 안정적인가(= 표시 가능한가)
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
LAG = pd.Timedelta(hours=4)          # H=48봉 -- 라벨 확정 지연 (보수적)
K_GRID = (20, 50, 100, 200, 500)
D_GRID = (1, 3, 7, 30)               # 후행 기간(일)
EMB = pd.Timedelta(hours=4)
SEED = 20260908
BOOT = 3000


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def trailing(ts_ns, y, K=None, days=None):
    """⭐t 시점에 **이미 확정된**(t-4h 이전 발동) 사건만으로 후행 돌파율. 없으면 NaN."""
    n = len(y)
    cut = ts_ns - np.int64(LAG.value)
    j = np.searchsorted(ts_ns, cut, side="right")          # j = 사용 가능한 사건 수
    cs = np.concatenate([[0.0], np.cumsum(y.astype(float))])
    out = np.full(n, np.nan); cnt = np.zeros(n)
    if K is not None:
        lo = np.maximum(j - K, 0)
        cnt = (j - lo).astype(float)
        out = np.where(cnt >= max(K // 4, 10), (cs[j] - cs[lo]) / np.maximum(cnt, 1), np.nan)
    else:
        start = ts_ns - np.int64(pd.Timedelta(days=days).value) - np.int64(LAG.value)
        lo = np.searchsorted(ts_ns, start, side="left")
        lo = np.minimum(lo, j)
        cnt = (j - lo).astype(float)
        out = np.where(cnt >= 10, (cs[j] - cs[lo]) / np.maximum(cnt, 1), np.nan)
    return out, cnt


def main() -> int:
    rng = np.random.default_rng(SEED)
    A = pd.read_parquet(SRC); A["timestamp"] = pd.to_datetime(A["timestamp"])
    d = A[(A.anchor == "first_fire") & (A.T_mult == 1.0)].sort_values("timestamp").reset_index(drop=True)
    y = d["y"].to_numpy(int); ts = d["timestamp"]; ts_ns = ts.astype("int64").to_numpy()
    day = ts.dt.floor("D").to_numpy(); sp = d["split"].to_numpy()
    ev = np.isin(sp, WINS)
    print(f"사건 {len(d):,} · 표본외 {ev.sum():,} · 전체 돌파율 {y.mean():.4f}\n", flush=True)

    print("=" * 104)
    print("A·B) 후행 돌파율의 지속성과 그것만 따르는 규칙 (표본외 3창)")
    print("=" * 104)
    print(f"{'후행창':>12} {'유효%':>7} {'상관':>8} {'평균|편차|':>10} | " +
          " | ".join(f"{w[:8]:>22}" for w in WINS))
    feats = {}
    for nm, kw in ([(f"최근 {k}건", dict(K=k)) for k in K_GRID] +
                   [(f"최근 {dd}일", dict(days=dd)) for dd in D_GRID]):
        tr, cnt = trailing(ts_ns, y, **kw)
        feats[nm] = tr
        m0 = ev & np.isfinite(tr)
        if m0.sum() < 300: continue
        corr = np.corrcoef(tr[m0], y[m0])[0, 1]
        dev = np.abs(tr[m0] - 0.5).mean()
        line = f"{nm:>12} {m0.mean():>6.1%} {corr:>+8.4f} {dev:>10.4f} | "
        for w in WINS:
            m = m0 & (sp == w)
            if m.sum() < 100: line += f"{'--':>22} | "; continue
            call = (tr[m] > 0.5).astype(int)          # 후행률이 가리키는 쪽
            acc = (call == y[m]).astype(float)
            lo, hi = day_ci(acc, day[m], rng)
            line += f"{acc.mean():.4f}[{lo:.4f},{hi:.4f}] | "
        print(line, flush=True)

    print("\n" + "=" * 104)
    print("C) 모델에 후행률 피쳐를 넣으면 (기준 = 부록 AN 의 v2수정판+자금흐름)")
    print("=" * 104)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    base = [c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] + \
           [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))] + \
           ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
    for k, v in feats.items(): d[f"tr_{k}"] = v
    trc = [c for c in d.columns if c.startswith("tr_")]
    months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    for nm, cols in (("기준", base), ("+후행률", base + trc), ("후행률만", trc)):
        X = d[cols].to_numpy(np.float32)
        pred = np.full(len(d), np.nan)
        for i, mo in enumerate(uniq):
            if i < 6: continue
            te = (months == mo).to_numpy(); tr_ = (ts < ts[te].min() - EMB).to_numpy()
            if tr_.sum() < 2000 or te.sum() < 30: continue
            c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=1.0, early_stopping=True,
                                               validation_fraction=0.15, random_state=SEED)
            c.fit(X[tr_], y[tr_]); pred[te] = c.predict_proba(X[te])[:, 1]
        line = f"   {nm:>10}({len(cols):>3}) | "
        for w in WINS:
            m = np.isfinite(pred) & (sp == w)
            acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
            lo, hi = day_ci(acc, day[m], rng)
            b = max(y[m].mean(), 1 - y[m].mean())
            line += (f"{w[:4]} {acc.mean():.4f}[{lo:.4f}] A{roc_auc_score(y[m], pred[m]):.3f} "
                     f"기저{b:.3f} | ")
        print(line, flush=True)

    print("\n" + "=" * 104)
    print("D) 레짐 버킷별 역사적 돌파율 -- 창을 넘어 안정적인가(표시 가능성)")
    print("=" * 104)
    for f, nm, q in (("f_rv_ratio", "단기/장기 변동성비", 4), ("f_atr_pct", "ATR 수준", 4),
                     ("f_hour", "시간대(UTC)", 4), ("f_taker_z", "테이커 불균형 z", 4),
                     ("x_m_retail_z", "개미 롱숏비 z", 4)):
        if f not in d.columns: continue
        v = d[f].to_numpy(float)
        bins = pd.qcut(pd.Series(v), q, labels=False, duplicates="drop")
        tab = {}
        for w in ("TRAIN",) + WINS:
            m = (sp == w) & np.isfinite(v)
            g = pd.DataFrame({"b": bins[m], "y": y[m]}).groupby("b")["y"].mean()
            tab[w] = g
        T = pd.DataFrame(tab)
        rng_ = T.max(1) - T.min(1)
        sgn_ok = ((T.sub(0.5) > 0).all(1) | (T.sub(0.5) < 0).all(1))
        print(f"   {nm:>18}: " + " | ".join(
            f"Q{int(i)+1} " + "/".join(f"{T.loc[i, w]:.3f}" for w in ("TRAIN",) + WINS)
            for i in T.index) + f"   네창 부호일치 {int(sgn_ok.sum())}/{len(T)}")
    print(json.dumps({"n": len(d)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
