#!/usr/bin/env python3
"""방향 축 **확장창 walk-forward + 월 1회 재학습** (2026-09-07).

사용자: *"구럼 학습할 때 모든 라벨 데이터를 가지고 진행하는거야? 아니면 실제 데이터
24년 1월부터 5분봉 하나하나 지나가면서 진행하는거야?"* → *"제안해준대로 진행해줘"*

## 왜 도는가
지금까지 방향 축 17가지 기각은 전부 **고정분할**에서 나왔다: TRAIN(~2025-08)으로 한 번
학습한 모델이 VAL(2025-09~12) · OOS(2026-01~03) · 3rd(2026-04~07) 를 본다. 3rd 창으로
갈수록 성적이 나빠지는 패턴(예: three/perm20 VAL 0.609 → OOS 0.566 → 3rd 0.511)이
**두 가지 중 무엇인지 이 스크립트가 가른다**:
  (a) 모델이 낡아서 — 2025-08 이전 시장으로 배운 게 2026 시장에 안 맞는다
  (b) 정보가 없어서 — 애초에 방향은 예측 불가

월 1회 재학습으로 (a)를 제거한다. 개선되면 (a), 그대로면 (b)로 확정.

## 규약 (Fresh-Forward)
- 각 월 M 의 예측은 **M 시작 이전에 확정된 라벨만** 학습에 쓴다.
- 라벨 지평 H=48봉(4시간)이 M 안으로 새지 않도록 **엠바고 4시간**: 학습행 조건은
  `timestamp < M_start - 4h`. (라벨이 M_start 를 넘어 해소되는 행을 제외)
- 피쳐는 이미 인과적으로 만들어진 프레임(features154)을 그대로 쓴다.
- 예측 시점에 그 월의 어떤 행도 학습에 들어가지 않는다.

## 무엇을 도는가 (다중검정 통제)
피쳐셋을 27셀 다시 훑지 **않는다**. 고정분할에서 팔별 min3 가 최고였던 **1셀씩만**
사전 지정해 그 셀만 walk-forward 로 재측정한다:
    hard/dirtop20 (min3 0.5337) · three/perm20 (0.5105) · wbin/perm20 (0.4818)
여기에 파이프라인 정상성 확인용 **P1 크기축**(all150) 을 같은 walk-forward 로 함께 돈다.
크기축은 고정분할에서 세 창 전부 통과했으므로, walk-forward 에서도 통과해야
"월간 재학습 구현 자체는 멀쩡하다"가 확인된다.

## 판정
- 방향: 세 창 모두 일군집 CI 하한 > 0.5 이고 날 블록 셔플 귀무 p95 초과인 셀이 있는가
- 대조: 같은 셀의 고정분할 수치 대비 창별 Δ (개선/동일/악화)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tabicl import TabICLClassifier

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import research_eth_anchor_direction_tabicl_20260907 as T  # noqa: E402

OUT = ROOT / "tmp/eth_anchor_walkforward_20260907"
N_EST = 4
EMBARGO = pd.Timedelta(hours=4)          # 라벨 지평 H=48봉 = 4시간
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")

# 사전 지정 셀 — 고정분할(tmp/eth_anchor_tabicl_deep_20260907/direction_3windows.csv)의 팔별 min3 최고
CELLS = [("hard", "dirtop20"), ("three", "perm20"), ("wbin", "perm20")]

# 고정분할 대조값 (같은 셀, 같은 n_estimators) — 위 CSV 에서 그대로 옮김
FIXED = {
    ("hard", "dirtop20"):  {"VAL": 0.5479, "OOS": 0.5538, "HOLDOUT_SPENT": 0.5337},
    ("three", "perm20"):   {"VAL": 0.6092, "OOS": 0.5663, "HOLDOUT_SPENT": 0.5105},
    ("wbin", "perm20"):    {"VAL": 0.5442, "OOS": 0.5501, "HOLDOUT_SPENT": 0.4818},
}


def month_starts(ts: pd.Series) -> list[pd.Timestamp]:
    """평가 대상 월 경계 (VAL 시작 ~ 마지막 데이터)."""
    lo = pd.Timestamp("2025-09-01")
    hi = ts.max()
    out, cur = [], lo
    while cur <= hi:
        out.append(cur)
        cur = cur + pd.offsets.MonthBegin(1)
    return out


def walk_forward(D, X, fcols, y, mask, multi, months, tag="", shuffle_rng=None):
    """월 1회 재학습. 각 월의 예측을 모아 반환.

    shuffle_rng 를 주면 **그 월의 학습 라벨만** 날 블록 셔플한다 (재학습 귀무).
    평가 라벨은 건드리지 않는다.
    """
    ts = D["timestamp"]
    day = D["timestamp"].dt.floor("D").to_numpy()
    Xf = np.nan_to_num(X[:, fcols].astype(np.float32))
    preds = np.full(len(D), np.nan)
    n_fit = []
    for i, m0 in enumerate(months):
        m1 = m0 + pd.offsets.MonthBegin(1)
        te = mask & (ts >= m0).to_numpy() & (ts < m1).to_numpy()
        if te.sum() < 20:
            n_fit.append((str(m0.date()), 0, int(te.sum())))
            continue
        tr = mask & (ts < (m0 - EMBARGO)).to_numpy()
        if tr.sum() < 300 or len(np.unique(y[tr])) < 2:
            n_fit.append((str(m0.date()), int(tr.sum()), int(te.sum())))
            continue
        yt = y
        if shuffle_rng is not None:                 # 날 블록 셔플 (학습 라벨만)
            yt = y.copy()
            uq = np.unique(day[tr]); perm = shuffle_rng.permutation(uq)
            src = {d: np.flatnonzero(tr & (day == d)) for d in uq}
            for d, d2 in zip(uq, perm):
                if len(src[d]):
                    yt[src[d]] = np.resize(y[src[d2]], len(src[d]))
            if len(np.unique(yt[tr])) < 2:
                continue
        t0 = time.time()
        n_est = 2 if shuffle_rng is not None else N_EST
        clf = TabICLClassifier(device="cpu", n_estimators=n_est, random_state=T.SEED, verbose=False)
        clf.fit(Xf[tr], yt[tr])
        p = clf.predict_proba(Xf[te])
        if multi:
            den = p[:, 0] + p[:, 2]
            preds[te] = np.where(den > 0, p[:, 2] / np.maximum(den, 1e-12), 0.5)
        else:
            preds[te] = p[:, 1]
        n_fit.append((str(m0.date()), int(tr.sum()), int(te.sum())))
        if shuffle_rng is None:
            print(f"      {tag}{m0.date()} 학습 {tr.sum():>5} → 예측 {te.sum():>4}  ({time.time()-t0:.0f}s)", flush=True)
    return preds, n_fit


def eval_windows(D, preds, y, mask, multi, rng):
    """창별 AUC + 일군집 CI + 월별 곡선."""
    sp = D["split"].to_numpy()
    day = D["timestamp"].dt.floor("D").to_numpy()
    mon = D["timestamp"].dt.to_period("M").astype(str).to_numpy()
    ok = mask & np.isfinite(preds)
    if multi:
        ok = ok & (y != 1)
    yy_all = (y == 2).astype(int) if multi else y
    res, curve = {}, []
    for w in WINS:
        s = ok & (sp == w)
        if s.sum() < 30 or len(np.unique(yy_all[s])) < 2:
            continue
        a = float(roc_auc_score(yy_all[s], preds[s]))
        lo, hi = T.day_auc_ci(yy_all[s], preds[s], day[s], rng)
        res[w] = {"n": int(s.sum()), "auc": a, "lo": lo, "hi": hi}
    for mm in sorted(np.unique(mon[ok])):
        s = ok & (mon == mm)
        if s.sum() < 20 or len(np.unique(yy_all[s])) < 2:
            continue
        curve.append({"month": mm, "n": int(s.sum()), "auc": float(roc_auc_score(yy_all[s], preds[s]))})
    return res, curve, ok, yy_all


def dayblock_null(y, p, days, rng, B=200):
    """예측 고정 · 라벨만 날 블록 셔플 (일내 셔플 금지 -- 라벨이 날 군집).

    저장소 관용구(`run_cell` 의 `np.resize` 블록 교환)와 같은 방식.
    이건 '이 순위가 우연을 넘는가'만 본다. 재학습까지 포함한 강한 귀무는
    `walk_forward(..., shuffle_rng=...)` 로 따로 돈다 (CI 통과 셀에만).
    """
    uniq = np.unique(days)
    src = {d: np.flatnonzero(days == d) for d in uniq}
    out = []
    for _ in range(B):
        perm = rng.permutation(uniq)
        yb = y.copy()
        for d, d2 in zip(uniq, perm):
            if len(src[d]):
                yb[src[d]] = np.resize(y[src[d2]], len(src[d]))
        if len(np.unique(yb)) < 2:
            continue
        out.append(roc_auc_score(yb, p))
    if not out:
        return {}
    out = np.array(out)
    return {"mean": float(out.mean()), "p95": float(np.percentile(out, 95)), "max": float(out.max())}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260907)
    D = pd.read_parquet(T.SRCD / "features154.parquet")
    meta = json.loads((T.SRCD / "meta.json").read_text())
    cols = [c for c in meta["feature_cols"] if c in D.columns]
    X = D[cols].to_numpy(np.float64); ci = {c: i for i, c in enumerate(cols)}
    sets = T.build_feature_sets(D, cols)
    months = month_starts(D["timestamp"])
    print(f"[입력] {D.shape} · split {D.split.value_counts().to_dict()}", flush=True)
    print(f"[월] {len(months)}개 {months[0].date()} ~ {months[-1].date()} · 엠바고 {EMBARGO}", flush=True)

    rows, curves, nulls = [], {}, {}

    # ---------------- 방향 3셀
    print("\n=== 방향: 월 1회 재학습 walk-forward ===", flush=True)
    for arm, sname in CELLS:
        mask, y, multi = T.arm_spec(D, arm)
        fc = [ci[c] for c in sets[sname]]
        print(f"\n  [{arm}/{sname}] 피쳐 {len(fc)}개 · 유효 {mask.sum()}", flush=True)
        preds, nfit = walk_forward(D, X, fc, y, mask, multi, months)
        res, curve, ok, yy = eval_windows(D, preds, y, mask, multi, rng)
        day = D["timestamp"].dt.floor("D").to_numpy()
        nl = {}
        for w in WINS:
            s = ok & (D["split"].to_numpy() == w)
            if s.sum() >= 30:
                nl[w] = dayblock_null(yy[s], preds[s], day[s], rng)
        rec = {"axis": "direction", "arm": arm, "featset": sname, "n_feat": len(fc)}
        for w in WINS:
            r = res.get(w, {})
            rec[f"{w}_n"] = r.get("n", 0); rec[f"{w}_auc"] = r.get("auc", np.nan)
            rec[f"{w}_lo"] = r.get("lo", np.nan); rec[f"{w}_hi"] = r.get("hi", np.nan)
            rec[f"{w}_null_p95"] = nl.get(w, {}).get("p95", np.nan)
            rec[f"{w}_fixed"] = FIXED[(arm, sname)][w]
            rec[f"{w}_delta"] = rec[f"{w}_auc"] - rec[f"{w}_fixed"]
        rec["min3"] = np.nanmin([rec[f"{w}_auc"] for w in WINS])
        rec["ci3"] = bool(all(rec[f"{w}_lo"] > 0.5 for w in WINS if np.isfinite(rec[f"{w}_lo"])))
        rec["null3"] = bool(all(rec[f"{w}_auc"] > rec[f"{w}_null_p95"] for w in WINS
                                if np.isfinite(rec[f"{w}_null_p95"])))
        rows.append(rec); curves[f"{arm}/{sname}"] = curve; nulls[f"{arm}/{sname}"] = nl
        np.save(OUT / f"preds_{arm}_{sname}.npy", preds)
        for w in WINS:
            r = res.get(w, {})
            print(f"      {w:<14} AUC {r.get('auc', float('nan')):.4f} "
                  f"[{r.get('lo', float('nan')):.3f}] · 고정 {rec[f'{w}_fixed']:.4f} "
                  f"· Δ {rec[f'{w}_delta']:+.4f} · 귀무p95 {rec[f'{w}_null_p95']:.4f}", flush=True)

    # ---------------- P1 크기축 (파이프라인 정상성)
    print("\n=== P1 크기축: 같은 walk-forward (구현 정상성 확인) ===", flush=True)
    rp = D["range_pct"].to_numpy()
    thr = float(np.nanmedian(rp[D["split"].to_numpy() == "TRAIN"]))
    y_sz = (rp > thr).astype(int); m_sz = np.isfinite(rp)
    fc = [ci[c] for c in sets["all150"]]
    preds, _ = walk_forward(D, X, fc, y_sz, m_sz, False, months, tag="크기 ")
    res, curve, ok, yy = eval_windows(D, preds, y_sz, m_sz, False, rng)
    day = D["timestamp"].dt.floor("D").to_numpy()
    # 같은 walk-forward 창에서 학습 없는 raw atr_pct 대조
    atr = D["atr_pct"].to_numpy()
    rec = {"axis": "size", "arm": f"range_pct>{thr:.4f}", "featset": "all150", "n_feat": len(fc)}
    for w in WINS:
        r = res.get(w, {}); rec[f"{w}_n"] = r.get("n", 0)
        rec[f"{w}_auc"] = r.get("auc", np.nan); rec[f"{w}_lo"] = r.get("lo", np.nan)
        s = ok & (D["split"].to_numpy() == w)
        rec[f"{w}_atr"] = float(roc_auc_score(yy[s], atr[s])) if s.sum() >= 30 else np.nan
        rec[f"{w}_null_p95"] = dayblock_null(yy[s], preds[s], day[s], rng).get("p95", np.nan) if s.sum() >= 30 else np.nan
        print(f"      {w:<14} AUC {rec[f'{w}_auc']:.4f} [{rec[f'{w}_lo']:.3f}] "
              f"· atr단독 {rec[f'{w}_atr']:.4f} · 귀무p95 {rec[f'{w}_null_p95']:.4f}", flush=True)
    rec["min3"] = np.nanmin([rec[f"{w}_auc"] for w in WINS])
    rec["ci3"] = bool(all(rec[f"{w}_lo"] > 0.5 for w in WINS if np.isfinite(rec[f"{w}_lo"])))
    rows.append(rec); curves["size/all150"] = curve

    # ---------------- 표
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "walkforward.csv", index=False)
    (OUT / "curves.json").write_text(json.dumps(curves, indent=1, ensure_ascii=False))
    (OUT / "nulls.json").write_text(json.dumps(nulls, indent=1, ensure_ascii=False))

    print("\n" + "=" * 118, flush=True)
    print(f"{'축':<10}{'팔':<8}{'피쳐셋':<10}"
          + "".join(f"{w+' WF':>12}{'고정':>9}{'Δ':>9}" for w in WINS), flush=True)
    for _, r in A.iterrows():
        line = f"{r['axis']:<10}{str(r['arm'])[:7]:<8}{r['featset']:<10}"
        for w in WINS:
            fx = r.get(f"{w}_fixed", np.nan); dl = r.get(f"{w}_delta", np.nan)
            line += f"{r[f'{w}_auc']:>12.4f}" + (f"{fx:>9.4f}{dl:>+9.4f}" if np.isfinite(fx) else " " * 18)
        print(line, flush=True)

    dirn = A[A.axis == "direction"]
    passed = dirn[dirn.ci3 & dirn.null3]
    print(f"\n방향 세 창 CI+귀무 통과: {len(passed)}/{len(dirn)} · min3 최고 {dirn.min3.max():.4f}", flush=True)
    print(f"고정분할 대비 창별 Δ 평균: "
          + " · ".join(f"{w} {dirn[f'{w}_delta'].mean():+.4f}" for w in WINS), flush=True)
    sz = A[A.axis == "size"].iloc[0]
    print(f"P1 크기축 walk-forward 세 창 CI 통과: {'✅ 예' if sz.ci3 else '❌ 아니오'} "
          f"(구현 정상성) · min3 {sz.min3:.4f}", flush=True)

    print("\n=== 월별 AUC 곡선 (낡음 진단) ===", flush=True)
    for k, c in curves.items():
        if c:
            print(f"  {k:<20}" + " ".join(f"{x['month'][2:]}:{x['auc']:.2f}" for x in c), flush=True)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
