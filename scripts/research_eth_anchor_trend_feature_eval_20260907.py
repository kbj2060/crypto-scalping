#!/usr/bin/env python3
"""추세 피쳐(DeMarker 연속값 + 레짐 공식)를 방향 축에 얹어 검정 (2026-09-07).

사용자: *"DeMarker 지표와 레짐을 피쳐로 넣는건 어때? 이게 추세를 읽기 좋아"*
       *"복합 오실레이터 신호에서도 뽑아낼 수 있는 순수 지표가 있나?"*

두 블록을 함께 검정한다 (둘 다 방향 축에 처음 들어가는 정보다).
  **추세 16** `build_eth_anchor_trend_features_20260907.py`
      DeMarker(14/28/56) 연속값 + S12_K3 레짐 라벨 공식(er12/er24/net24/slope12/확정레짐).
      인과성 재구성 PASS(오차 0.00e+00) · 기존 DC 추세 피쳐·atr_pct 와 |r|>0.5 상관 **없음**.
      ⚠️배포 레짐 **모델**은 안 쓴다 -- train_range 2024-01-01~2026-06-30 이 전 구간을 덮어 누수.
  **오실레이터 코어 16** `build_eth_anchor_oscillator_cores_20260907.py`
      8종 증거신호의 순수 연속 코어: p_fast/p_slow(0~1 백분위), delta_z, ret3_z,
      kalman_dev_z, funding_z, dem, fib 확장배수, 스윕 깊이/되찾기.
      라이브 `compute_signals()` 직접 호출 · **파리티 PASS**(8종 12갈래 재구성 불일치 0) ·
      인과성 PASS(오차 0.00e+00).
지금까지 8종은 **이진 발동**으로만 쓰였고 그 발동은 앵커 정의에 이미 흡수돼 있다.
연속값이 방향 축에 들어가는 것은 이번이 처음이다.

## 다중검정 통제 (사전 지정)
피쳐셋 3개 × 3팔 = 9셀만 돈다. 피쳐셋 격자를 다시 훑지 않는다.
  `new32`       추세16 + 오실레이터16 (이 축 자체에 신호가 있는가)
  `dirtop20`    대조 -- 기존 최고 소형셋 (부록 R 의 min3 최고)
  `top20+new32` 기존 최고 + 새 축 (증분이 있는가)
판정은 세 창(VAL/OOS/HOLDOUT_SPENT) 모두 일군집 CI 하한 > 0.5 ∧ 날 블록 귀무 p95 초과.
증분 판정은 `dirtop20` 대비 **같은 날 표집 짝비교 CI** 로 본다 (원시 AUC 차이 금지).
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
import research_eth_anchor_tabicl_deep_20260907 as DP      # noqa: E402

TRD = ROOT / "tmp/eth_anchor_trend_features_20260907"
OSC = ROOT / "tmp/eth_anchor_osc_cores_20260907"
OUT = ROOT / "tmp/eth_anchor_trend_eval_20260907"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
N_EST = 4


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260907)
    D = pd.read_parquet(T.SRCD / "features154.parquet")
    A = pd.read_parquet(TRD / "trend_features.parquet")
    O = pd.read_parquet(OSC / "osc_cores.parquet")
    tm = json.loads((TRD / "meta.json").read_text())
    om = json.loads((OSC / "meta.json").read_text())
    assert (A["timestamp"].to_numpy() == D["timestamp"].to_numpy()).all(), "추세 행 정렬 불일치"
    assert (O["timestamp"].to_numpy() == D["timestamp"].to_numpy()).all(), "오실 행 정렬 불일치"
    assert tm["causality"]["PASS"], "🔴추세 피쳐 인과성 FAIL -- 진행 금지"
    assert om["causality"]["PASS"], "🔴오실 코어 인과성 FAIL -- 진행 금지"
    assert om["parity"]["ALL_MATCH"], "🔴오실 코어 파리티 FAIL -- 진행 금지"

    meta = json.loads((T.SRCD / "meta.json").read_text())
    base_cols = [c for c in meta["feature_cols"] if c in D.columns]
    trend_cols = tm["feature_cols"]
    osc_cols = [c for c in om["feature_cols"] if c != "dem_al"]   # dem_al 은 추세블록 dem14_al 과 중복
    new_cols = trend_cols + osc_cols
    sets_base = T.build_feature_sets(D, base_cols)

    # 결합 프레임: 기존 150 + 추세 16 + 오실 15
    X = np.concatenate([D[base_cols].to_numpy(np.float64),
                        A[trend_cols].to_numpy(np.float64),
                        O[osc_cols].to_numpy(np.float64)], axis=1)
    allc = base_cols + trend_cols + osc_cols
    ci = {c: i for i, c in enumerate(allc)}
    SETS = {
        "new32": new_cols,
        "dirtop20": sets_base["dirtop20"],
        "top20+new32": list(sets_base["dirtop20"]) + new_cols,
    }
    print(f"[입력] 앵커 {len(D):,} · 기존 {len(base_cols)} + 추세 {len(trend_cols)} "
          f"+ 오실 {len(osc_cols)} · split {D.split.value_counts().to_dict()}", flush=True)
    for k, v in SETS.items():
        print(f"   {k:<14} {len(v)}개", flush=True)

    rows, keep = [], {}
    print("\n=== 3팔 × 3피쳐셋 × 세 창 ===", flush=True)
    for arm in ("hard", "three", "wbin"):
        for sname, feats in SETS.items():
            fc = [ci[c] for c in feats]
            t0 = time.time()
            r = DP.run3(D, X, arm, fc, rng, n_est=N_EST)
            rec = {"arm": arm, "featset": sname, "n_feat": len(fc), "sec": round(time.time() - t0, 1)}
            for w in WINS:
                rec[f"{w}_auc"] = r.get(f"{w}_auc", np.nan)
                rec[f"{w}_lo"] = r.get(f"{w}_lo", np.nan)
                rec[f"{w}_n"] = r.get(f"{w}_n", 0)
            rec["min3"] = np.nanmin([rec[f"{w}_auc"] for w in WINS])
            rec["ci3"] = bool(all(rec[f"{w}_lo"] > 0.5 for w in WINS if np.isfinite(rec[f"{w}_lo"])))
            rows.append(rec); keep[(arm, sname)] = r
            print(f"   {arm:<7}{sname:<14} " + " · ".join(
                f"{w[:3]} {rec[f'{w}_auc']:.4f}[{rec[f'{w}_lo']:.3f}]" for w in WINS)
                + f" · min3 {rec['min3']:.4f}", flush=True)

    # ---------------- 증분: dirtop20 대비 같은 날 표집 짝비교
    print("\n=== 증분 (dirtop20 대비, 같은 날 표집 짝비교 CI) ===", flush=True)
    inc = []
    for arm in ("hard", "three", "wbin"):
        base = keep[(arm, "dirtop20")]
        for sname in ("new32", "top20+new32"):
            cand = keep[(arm, sname)]
            rec = {"arm": arm, "featset": sname}
            for w in WINS:
                if f"{w}_pred" not in cand or f"{w}_pred" not in base:
                    continue
                y = cand[f"{w}_y"]; d = cand[f"{w}_day"]
                if len(y) != len(base[f"{w}_y"]) or not (y == base[f"{w}_y"]).all():
                    rec[f"{w}_d"] = np.nan; continue
                dl = cand[f"{w}_auc"] - base[f"{w}_auc"]
                lo, hi = DP.diff_ci(y, cand[f"{w}_pred"], base[f"{w}_pred"], d, rng)
                rec[f"{w}_d"] = dl; rec[f"{w}_dlo"] = lo; rec[f"{w}_dhi"] = hi
            inc.append(rec)
            print(f"   {arm:<7}{sname:<14} " + " · ".join(
                f"{w[:3]} Δ{rec.get(f'{w}_d', np.nan):+.4f}"
                f"[{rec.get(f'{w}_dlo', np.nan):+.3f},{rec.get(f'{w}_dhi', np.nan):+.3f}]"
                for w in WINS), flush=True)

    # ---------------- 귀무: CI 통과 셀에만 (없으면 min3 최고 셀에)
    A_ = pd.DataFrame(rows)
    cand = A_[A_.ci3] if A_.ci3.any() else A_.nlargest(1, "min3")
    print(f"\n=== 날 블록 셔플 귀무 B=15 ({len(cand)}셀) ===", flush=True)
    nulls = []
    for _, r in cand.iterrows():
        fc = [ci[c] for c in SETS[r.featset]]
        a = {w: [] for w in WINS}
        for _ in range(15):
            rr = _shuffled_run(D, X, r.arm, fc, rng)
            for w in WINS:
                if f"{w}_auc" in rr:
                    a[w].append(rr[f"{w}_auc"])
        rec = {"arm": r.arm, "featset": r.featset}
        for w in WINS:
            p95 = float(np.percentile(a[w], 95)) if a[w] else np.nan
            rec[f"{w}_null_p95"] = p95
            rec[f"{w}_p"] = float(np.mean(np.array(a[w]) >= r[f"{w}_auc"])) if a[w] else np.nan
        rec["PASS"] = bool(r.ci3 and all(r[f"{w}_auc"] > rec[f"{w}_null_p95"] for w in WINS
                                         if np.isfinite(rec[f"{w}_null_p95"])))
        nulls.append(rec)
        print(f"   {r.arm:<7}{r.featset:<14} " + " · ".join(
            f"{w[:3]} p95 {rec[f'{w}_null_p95']:.4f}(p={rec[f'{w}_p']:.3f})" for w in WINS)
            + f" → {'✅PASS' if rec['PASS'] else '❌'}", flush=True)

    A_.to_csv(OUT / "trend_eval.csv", index=False)
    pd.DataFrame(inc).to_csv(OUT / "increments.csv", index=False)
    pd.DataFrame(nulls).to_csv(OUT / "nulls.csv", index=False)
    print("\n" + "=" * 100, flush=True)
    print(f"세 창 CI 통과 {int(A_.ci3.sum())}/{len(A_)} · min3 최고 {A_.min3.max():.4f} "
          f"({A_.loc[A_.min3.idxmax(), 'arm']}/{A_.loc[A_.min3.idxmax(), 'featset']})", flush=True)
    print(f"최종 귀무까지 통과: {sum(n['PASS'] for n in nulls)}셀", flush=True)
    print(f"\n저장: {OUT}", flush=True)
    return 0


def _shuffled_run(D, X, arm, fc, rng):
    """학습 라벨만 날 블록 셔플 후 세 창 평가 (deep.run3 에 shuffle 옵션이 없어 여기서 구현)."""
    m, y, multi = T.arm_spec(D, arm)
    sp = D["split"].to_numpy(); day = D["timestamp"].dt.floor("D").to_numpy()
    tr = m & (sp == "TRAIN")
    yt = y.copy()
    uq = np.unique(day[tr]); perm = rng.permutation(uq)
    src = {d: np.flatnonzero(tr & (day == d)) for d in uq}
    for d, d2 in zip(uq, perm):
        if len(src[d]):
            yt[src[d]] = np.resize(y[src[d2]], len(src[d]))
    Xf = np.nan_to_num(X[:, fc].astype(np.float32))
    clf = TabICLClassifier(device="cpu", n_estimators=2, random_state=T.SEED, verbose=False)
    clf.fit(Xf[tr], yt[tr])
    out = {}
    for w in WINS:
        te = m & (sp == w)
        if te.sum() < 30:
            continue
        p = clf.predict_proba(Xf[te])
        if multi:
            cl = y[te] != 1
            den = p[cl][:, 0] + p[cl][:, 2]
            pr = np.where(den > 0, p[cl][:, 2] / np.maximum(den, 1e-12), 0.5)
            yy = (y[te][cl] == 2).astype(int)
        else:
            pr = p[:, 1]; yy = y[te]
        if len(np.unique(yy)) > 1:
            out[f"{w}_auc"] = float(roc_auc_score(yy, pr))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
