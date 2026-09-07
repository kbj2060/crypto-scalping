#!/usr/bin/env python3
"""돌파/되돌림 -- **TabPFN** 업그레이드 (2026-09-08, 서버 GPU 전용).

사용자: *"증거신호에서 규칙을 딥러닝으로 업그레이드한 것처럼 이것도 그렇게 진행해줘.
tabicl이나 tabpfn 으로 진행해줘"* (서버에 tabicl 미설치 → TabPFN 8.5.0)

## HGB 기준선 (v4, first_fire T=1.0, 18,157건, 기저 50.4/50.8/50.2)
| 구성 | VAL | OOS | HOLDOUT |
|---|---|---|---|
| `BTC동조만`(5) | 0.5334[0.5105] | 0.5379[0.5142] | 0.5202[0.5007] |
| `기준`(69) | 0.5233[0.5026] | 0.5464[0.5206] | 0.5382[0.5174] |
| `기준+테이프`(122) | 0.5022 | 0.5476 | 0.5518 |
⭐HGB 는 122피쳐에서 VAL 이 무너졌다(학습 ~12k행 과적합). **TabPFN 은 데이터셋별 학습이 없는
in-context 추론이라 그 실패모드에 다른 귀납 편향**을 준다 -- 테이프가 살아날 수 있는 유일한 경로다.

## ⚠️이 문제의 귀무는 0.5 가 아니다
`research_eth_anchor_direction_tabpfn_20260907.py` 기록: *"순진한 0.5 기준은 이 문제에서 틀렸다
-- 일군집 구조 때문에 귀무가 0.52~0.54"*. 그래서 **같은 모델로 라벨 셔플 귀무를 병기**한다.
(v4 HGB 셔플은 0.497~0.509 로 깨끗했으나 모델이 바뀌면 다시 재야 한다.)

프로토콜: 월 1회 재학습 expanding walk-forward · 엠바고 4h · 시드 5개 평균 · 일군집 CI ·
경계 계약 준수(피쳐는 전부 트리거 분 `s1-1` 까지, `.claude/CLAUDE.md`).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
SRC = MY / "dataset_v4.parquet"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
EMB = pd.Timedelta(hours=4)
SEEDS = [11, 23, 47, 71, 97]
SUB = 18000
NULL_B = 10
BOOT = 3000
DEVICE = "cuda"
BTC = ["v2_mv_btc_ret_atr", "v2_mv_idio_atr", "v2_mv_same_sign", "dir_up", "T_atr"]


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def fit_pred(Xtr, ytr, Xte, seed):
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(device=DEVICE, random_state=seed, ignore_pretraining_limits=True)
    clf.fit(Xtr, ytr.astype(int))
    return clf.predict_proba(Xte)[:, 1]


def wf(X, y, ts, months, uniq, seeds, shuffle=False, rng=None):
    pred = np.full(len(y), np.nan)
    for i, mo in enumerate(uniq):
        if i < 5: continue
        te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
        if tr.sum() < 1500 or te.sum() < 30: continue
        itr = np.flatnonzero(tr)
        yy = y.copy()
        if shuffle: yy[itr] = rng.permutation(yy[itr])
        ps = []
        for s in seeds:
            sel = itr if len(itr) <= SUB else np.random.default_rng(s).choice(itr, SUB, replace=False)
            ps.append(fit_pred(X[sel], yy[sel], X[te], s))
        pred[te] = np.mean(ps, axis=0)
        print(f"      {mo} 학습 {tr.sum():,} 평가 {te.sum():,}", flush=True)
    return pred


def report(tag, pred, y, sp, day, rng):
    from sklearn.metrics import roc_auc_score
    line = f"{tag:>22} | "; out = {}
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
    rng = np.random.default_rng(20260908)
    d = pd.read_parquet(SRC); d["timestamp"] = pd.to_datetime(d["timestamp"])
    tape = [c for c in d.columns if c.startswith(("t_", "mv_"))]
    have = d[tape].notna().any(axis=1).to_numpy()
    d = d[have].sort_values("timestamp").reset_index(drop=True)
    base = [c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] + \
           [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))] + \
           ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
    y = d["y"].to_numpy(int); ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy()
    sp = d["split"].to_numpy(); months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    print(f"행 {len(d):,} · 시드 {SEEDS} · 창별 "
          + " ".join(f"{w} {(sp==w).sum():,}" for w in ("TRAIN",) + WINS) + "\n" + "=" * 112, flush=True)
    SETS = {"BTC동조만(5)": BTC, "기준(69)": base, "기준+테이프(122)": base + tape}
    accs = {}
    for nm, cols in SETS.items():
        print(f"\n■ {nm}", flush=True)
        X = np.nan_to_num(d[cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        accs[nm] = report(f"TabPFN {nm}", wf(X, y, ts, months, uniq, SEEDS), y, sp, day, rng)
    best = max(accs, key=lambda k: min(accs[k].values()))
    print(f"\n⭐최고 구성: {best}\n" + "=" * 112, flush=True)
    print(f"라벨 셔플 귀무 (같은 모델·시드1개·B={NULL_B}) -- 이 문제의 귀무는 0.5 가 아니다", flush=True)
    X = np.nan_to_num(d[SETS[best]].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    nl = {w: [] for w in WINS}
    for b in range(NULL_B):
        p = wf(X, y, ts, months, uniq, SEEDS[:1], True, np.random.default_rng(1000 + b))
        for w in WINS:
            m = np.isfinite(p) & (sp == w)
            nl[w].append(float(((p[m] > 0.5).astype(int) == y[m]).mean()))
        print(f"   셔플 {b+1}/{NULL_B}", flush=True)
    print()
    for w in WINS:
        a = np.array(nl[w]); obs = accs[best][w]
        pc = (a >= obs).mean()
        print(f"   {w:>14} 관측 {obs:.4f} · 셔플 평균 {a.mean():.4f} p95 {np.percentile(a,95):.4f} "
              f"· p={max(pc,1/NULL_B):.3f} {'✅' if obs > np.percentile(a,95) else '❌'}")
    json.dump({"acc": accs, "null": {w: nl[w] for w in WINS}, "best": best},
              open(MY / "tabpfn_result.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"best": best}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
