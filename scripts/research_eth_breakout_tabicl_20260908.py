#!/usr/bin/env python3
"""돌파/되돌림 -- **TabICLv2** (2026-09-08, 로컬 CPU).

사용자: *"tabicl은 어떤지 테스트해줘"*

## 규격과 제약 (`research_eth_anchor_direction_tabicl_20260907.py` 관행 승계)
- TabICLv2 는 **표본 300~48K · 컬럼 2~100** 구간에 사전학습됐다. 우리 TRAIN 11,954행은 범위 안.
- ⚠️**컬럼 상한 100** -- `기준+테이프(122)` 는 초과라 그대로 못 쓴다.
  ⇒ `기준(69)` + 테이프 상위 31개(**TRAIN 전용** |Spearman| 순위)로 100개 세트를 만든다.
  선택에 VAL/OOS/HOLDOUT 을 절대 쓰지 않는다(선택 미래참조 방지).
- `fit(X, y)` 에 sample_weight 없음 -- 이 과제는 가중을 안 쓰므로 무관.
- CPU 실측 41초/fit(69피쳐·학습8000). GPU 는 TabPFN 작업이 점유 중이라 CPU 로 돈다.

## 비교 대상 (같은 워크포워드·엠바고·창)
HGB / TabPFN 결과는 `tmp/eth_breakout_atr_state_20260908_s1/` 에 있다. 기저 50.4/50.8/50.2.
⚠️저장소 기록: *"이 문제의 귀무는 0.5 가 아니다 -- 일군집 구조 때문에 0.52~0.54"*
   ⇒ 같은 모델 라벨셔플 귀무를 최고 구성에 병기한다.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
EMB = pd.Timedelta(hours=4)
SEED = 20260908
N_EST = 2
NULL_B = 5
BOOT = 3000
BTC = ["v2_mv_btc_ret_atr", "v2_mv_idio_atr", "v2_mv_same_sign", "dir_up", "T_atr"]


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def wf(X, y, ts, months, uniq, shuffle=False, rng=None, tag=""):
    from tabicl import TabICLClassifier
    pred = np.full(len(y), np.nan)
    t0 = time.time()
    for i, mo in enumerate(uniq):
        if i < 5: continue
        te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
        if tr.sum() < 1500 or te.sum() < 30: continue
        yy = y.copy()
        if shuffle: yy[np.flatnonzero(tr)] = rng.permutation(yy[np.flatnonzero(tr)])
        c = TabICLClassifier(device="cpu", n_estimators=N_EST, random_state=SEED, verbose=False)
        c.fit(X[tr], yy[tr]); pred[te] = c.predict_proba(X[te])[:, 1]
    print(f"      {tag} {time.time()-t0:.0f}초", flush=True)
    return pred


def report(tag, pred, y, sp, day, rng):
    from sklearn.metrics import roc_auc_score
    line = f"{tag:>26} | "; out = {}
    for w in WINS:
        m = np.isfinite(pred) & (sp == w)
        if m.sum() < 100: line += f"{w[:4]} -- | "; continue
        acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
        lo, hi = day_ci(acc, day[m], rng)
        b = max(y[m].mean(), 1 - y[m].mean())
        line += (f"{w[:4]} {acc.mean():.4f}[{lo:.4f}] A{roc_auc_score(y[m], pred[m]):.3f} "
                 f"기저{b:.3f} | ")
        out[w] = float(acc.mean())
    print(line, flush=True)
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    from scipy.stats import spearmanr
    d = pd.read_parquet(MY / "dataset_v4.parquet"); d["timestamp"] = pd.to_datetime(d["timestamp"])
    tape = [c for c in d.columns if c.startswith(("t_", "mv_"))]
    d = d[d[tape].notna().any(axis=1)].sort_values("timestamp").reset_index(drop=True)
    base = [c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] + \
           [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))] + \
           ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
    y = d["y"].to_numpy(int); ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy()
    sp = d["split"].to_numpy(); months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    # ⚠️TRAIN 전용 테이프 피쳐 선택 (컬럼 상한 100)
    tr0 = sp == "TRAIN"
    sc = []
    for c in tape:
        v = d[c].to_numpy(float)[tr0]; yy = y[tr0]; m = np.isfinite(v)
        if m.sum() < 500 or len(np.unique(v[m])) < 3: continue
        r = spearmanr(v[m], yy[m]).statistic
        if np.isfinite(r): sc.append((abs(r), c))
    sc.sort(reverse=True)
    top = [c for _, c in sc[:100 - len(base)]]
    print(f"행 {len(d):,} · 기준 {len(base)} · 테이프 {len(tape)} → TRAIN전용 상위 {len(top)} 선택")
    print(f"   선택된 테이프 상위8: {[c for _, c in sc[:8]]}\n" + "=" * 116, flush=True)
    SETS = {"BTC동조만(5)": BTC, "기준(69)": base, f"기준+테이프top{len(base)+len(top)}": base + top}
    accs = {}
    for nm, cols in SETS.items():
        X = np.nan_to_num(d[cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        accs[nm] = report(f"TabICL {nm}", wf(X, y, ts, months, uniq, tag=nm), y, sp, day, rng)
    best = max(accs, key=lambda k: min(accs[k].values()))
    print(f"\n⭐최고 구성: {best}", flush=True)
    print(f"라벨 셔플 귀무 (같은 모델, B={NULL_B}) -- 이 문제의 귀무는 0.5 가 아니다", flush=True)
    X = np.nan_to_num(d[SETS[best]].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    nl = {w: [] for w in WINS}
    for b in range(NULL_B):
        p = wf(X, y, ts, months, uniq, True, np.random.default_rng(500 + b), tag=f"셔플{b+1}")
        for w in WINS:
            m = np.isfinite(p) & (sp == w)
            nl[w].append(float(((p[m] > 0.5).astype(int) == y[m]).mean()))
    print()
    for w in WINS:
        a = np.array(nl[w]); obs = accs[best][w]
        print(f"   {w:>14} 관측 {obs:.4f} · 셔플 {np.round(a,4).tolist()} 평균 {a.mean():.4f} "
              f"max {a.max():.4f} {'✅' if obs > a.max() else '❌'}")
    json.dump({"acc": accs, "null": nl, "best": best, "tape_top": top},
              open(MY / "tabicl_result.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"best": best}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
