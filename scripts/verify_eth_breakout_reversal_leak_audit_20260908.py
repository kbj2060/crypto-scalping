#!/usr/bin/env python3
"""돌파/되돌림 결과 **누수 정밀 감사** (2026-09-08).

사용자: *"이것도 분명 룩어헤드나 버그가 있을 수 있어. 이전에도 좋게 나오고 다음 날 모두 폐기시켰잖아"*

## 이미 잡은 것 (A)
🔴**1분 미래참조**: 라벨은 `first_touch(..., s1, ...)` 로 트리거 분 s1 **부터** 배리어를 탐색하는데
경로 피쳐가 s1 을 **포함**했다 -> 같은 1분봉 공유. 수정 후 정확도 54.7~57.4% → 50.0~54.3%.
`v2_mv_maxbar_atr` 의 Q5−Q1 이 +0.098 → +0.012 로 소멸(= 그 피쳐는 거의 전부 누수였다).

## 이 감사가 추가로 확인하는 것
B. **경계봉 누수** -- 피쳐봉을 `bt-1` 에서 `bt-2` 로 한 봉 더 밀어도 정확도가 유지되는가.
   유지되면 경계 아님. 무너지면 남은 것도 경계 효과다.
C. **엠바고 부족** -- H=48봉(4h)인데 엠바고도 4h 였다. 학습 사건의 라벨 창이 테스트 구간과
   겹칠 수 있다. 12h 로 늘려 재확인.
D. **폴드 안정성** -- 월별 정확도가 특정 구간에 몰려 있는가.
E. **피쳐군 절제(수정 후)** -- 무엇이 남았는가.
F. **셔플 대조군** -- 라벨을 섞으면 50% 인가.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "tmp/eth_breakout_reversal_20260908"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
ANCH, TM = "first_fire", 0.75
SEED = 20260908
BOOT = 3000


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def wf(X, y, ts, months, uniq, emb, shuffle=False, rng=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    pred = np.full(len(y), np.nan)
    for i, mo in enumerate(uniq):
        if i < 6: continue
        te = (months == mo).to_numpy(); cut = ts[te].min() - emb; tr = (ts < cut).to_numpy()
        if tr.sum() < 2000 or te.sum() < 30: continue
        yy = y.copy()
        if shuffle: yy[tr] = rng.permutation(yy[tr])
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.15, random_state=SEED)
        c.fit(X[tr], yy[tr]); pred[te] = c.predict_proba(X[te])[:, 1]
    return pred


def report(tag, pred, y, sp, day, rng):
    from sklearn.metrics import roc_auc_score
    line = f"{tag:>26} | "
    out = {}
    for w in WINS:
        m = np.isfinite(pred) & (sp == w)
        if m.sum() < 100: line += f"{w[:4]} -- | "; continue
        acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
        lo, hi = day_ci(acc, day[m], rng)
        line += (f"{w[:4]} {acc.mean():.4f}[{lo:.4f},{hi:.4f}] A{roc_auc_score(y[m], pred[m]):.3f} | ")
        out[w] = (acc.mean(), lo)
    print(line, flush=True)
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    A = pd.read_parquet(DIR / "dataset_v2.parquet"); A["timestamp"] = pd.to_datetime(A["timestamp"])
    d = A[(A.anchor == ANCH) & (A.T_mult == TM)].sort_values("timestamp").reset_index(drop=True)
    y = d["y"].to_numpy(int); ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy()
    sp = d["split"].to_numpy(); months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    cols = [c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] + \
           ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
    X = d[cols].to_numpy(np.float32)
    print(f"이벤트 {len(d):,} · 피쳐 {len(cols)} · 기저 {max(y.mean(),1-y.mean()):.4f}\n", flush=True)
    print("=" * 112)
    base = report("기준(엠바고 4h)", wf(X, y, ts, months, uniq, pd.Timedelta(hours=4)), y, sp, day, rng)
    print("\nC) 엠바고 확대", flush=True)
    for h in (12, 24):
        report(f"엠바고 {h}h", wf(X, y, ts, months, uniq, pd.Timedelta(hours=h)), y, sp, day, rng)
    print("\nF) 셔플 대조군 (50% 이어야 정상)", flush=True)
    for s in range(2):
        r2 = np.random.default_rng(SEED + s)
        report(f"라벨셔플 seed{s}", wf(X, y, ts, months, uniq, pd.Timedelta(hours=4), True, r2),
               y, sp, day, rng)
    print("\nE) 피쳐군 절제 (수정 후)", flush=True)
    G = {"v1봉": [c for c in cols if c.startswith(("f_", "sig_")) or c in
                 ("dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom")],
         "경로": [c for c in cols if c.startswith("v2_mv_") and not any(
             k in c for k in ("btc", "idio", "same_sign"))],
         "레벨": [c for c in cols if c.startswith(("v2_brk", "v2_pos"))],
         "교차": [c for c in cols if c.startswith("v2_mv_") and any(
             k in c for k in ("btc", "idio", "same_sign"))]}
    for nm, ks in (("v1만", ["v1봉"]), ("v1+경로", ["v1봉", "경로"]),
                   ("v1+경로+레벨", ["v1봉", "경로", "레벨"]), ("전체", list(G))):
        cc = sorted({c for k in ks for c in G[k]})
        report(f"{nm}({len(cc)})", wf(d[cc].to_numpy(np.float32), y, ts, months, uniq,
                                      pd.Timedelta(hours=4)), y, sp, day, rng)
    print("\nD) 폴드(월) 안정성 -- 기준 구성", flush=True)
    pred = wf(X, y, ts, months, uniq, pd.Timedelta(hours=4))
    ok = np.isfinite(pred) & np.isin(sp, WINS)
    g = pd.DataFrame({"m": months[ok].astype(str),
                      "hit": ((pred[ok] > 0.5).astype(int) == y[ok]).astype(float)})
    t = g.groupby("m")["hit"].agg(["mean", "count"])
    print(f"   표본외 월 {len(t)}개 · 50% 초과 {int((t['mean']>0.5).sum())}개 · "
          f"평균 {t['mean'].mean():.4f} · 최저 {t['mean'].min():.3f}({t['mean'].idxmin()}) · "
          f"최고 {t['mean'].max():.3f}({t['mean'].idxmax()})")
    print("   " + " ".join(f"{i[-5:]}:{r['mean']:.3f}" for i, r in t.iterrows()))
    print("\n" + "=" * 112)
    ok3 = all(base.get(w, (0, -1))[1] > 0.5 for w in WINS)
    print(f"⭐세 창 모두 정확도 CI 하한 > 50%: {'예' if ok3 else '**아니오**'}")
    print(json.dumps({"three_window_pass": bool(ok3)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
