#!/usr/bin/env python3
"""지연(접근 경로) 블록 검정 -- **시퀀스 모델을 지을 값어치가 있나** (2026-09-07).

사용자: *"그럼 학습이 너무 어려울 것 같은데 시퀀스로 먹일 순 없나?"*

## 이게 시퀀스 모델의 싼 반증인 이유
1D-CNN/GRU 의 첫 층은 결국 **같은 지연들의 학습된 결합**이다. 지연을 명시적으로
줬는데도 `t` 단독을 못 이기면, 그 결합을 스스로 배워야 하는 시퀀스 모델이
3,237 TRAIN 표본에서 이길 가능성은 사실상 없다. 새 아키텍처 없이 지금 파이프라인으로
도는 테스트이고, **여기서 아무것도 안 나오면 2단계(진짜 시퀀스 모델)를 짓지 않는다.**

빌더: `build_eth_anchor_lag_features_20260907.py`
  TRAIN-only 순위 상위 8피쳐 × 시점 t,t-1,t-3,t-6,t-12,t-24
  `lag_level` 48열(원값) · `lag_delta` 40열(f(t)-f(t-k), 모양을 직접 준다)

## 피쳐셋 (사전 지정 4개 × 3팔 = 12셀)
  `t0_only`     같은 8피쳐를 t 에서만 (대조 -- 이게 기준선이다)
  `lag_level`   48열
  `lag_delta`   8열(t) + 40열(차분) = 48열
  `lag_all`     88열 (TabICL 2~100 컬럼 범위 안)

## 판정
핵심은 절대 AUC 가 아니라 **`t0_only` 대비 같은 날 표집 짝비교 증분**이다.
세 창 중 두 창 이상 CI 하한 > 0 인 지연셋이 하나라도 있어야 2단계로 간다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import research_eth_anchor_direction_tabicl_20260907 as T   # noqa: E402
import research_eth_anchor_tabicl_deep_20260907 as DP       # noqa: E402

LAG = ROOT / "tmp/eth_anchor_lag_features_20260907"
OUT = ROOT / "tmp/eth_anchor_lag_eval_20260907"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
N_EST = 4


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260907)
    D = pd.read_parquet(T.SRCD / "features154.parquet")
    L = pd.read_parquet(LAG / "lag_features.parquet")
    lm = json.loads((LAG / "meta.json").read_text())
    assert (L["timestamp"].to_numpy() == D["timestamp"].to_numpy()).all(), "행 정렬 불일치"

    base = lm["base_feats"]
    t0 = [f"{f}__t0" for f in base]
    lvl, dlt = lm["level_cols"], lm["delta_cols"]
    cols = lvl + dlt
    X = L[cols].to_numpy(np.float64)
    ci = {c: i for i, c in enumerate(cols)}
    SETS = {"t0_only": t0, "lag_level": lvl, "lag_delta": t0 + dlt, "lag_all": lvl + dlt}
    print(f"[입력] 앵커 {len(D):,} · 기반 {len(base)}피쳐 · 지연 {lm['lags']}", flush=True)
    for k, v in SETS.items():
        print(f"   {k:<12} {len(v):>3}열", flush=True)

    rows, keep = [], {}
    print("\n=== 3팔 × 4피쳐셋 × 세 창 ===", flush=True)
    for arm in ("hard", "three", "wbin"):
        for sname, feats in SETS.items():
            fc = [ci[c] for c in feats]
            t = time.time()
            r = DP.run3(D, X, arm, fc, rng, n_est=N_EST)
            rec = {"arm": arm, "featset": sname, "n_feat": len(fc), "sec": round(time.time() - t, 1)}
            for w in WINS:
                rec[f"{w}_auc"] = r.get(f"{w}_auc", np.nan)
                rec[f"{w}_lo"] = r.get(f"{w}_lo", np.nan)
            rec["min3"] = np.nanmin([rec[f"{w}_auc"] for w in WINS])
            rec["ci3"] = bool(all(rec[f"{w}_lo"] > 0.5 for w in WINS if np.isfinite(rec[f"{w}_lo"])))
            rows.append(rec); keep[(arm, sname)] = r
            print(f"   {arm:<7}{sname:<12} " + " · ".join(
                f"{w[:3]} {rec[f'{w}_auc']:.4f}[{rec[f'{w}_lo']:.3f}]" for w in WINS)
                + f" · min3 {rec['min3']:.4f}", flush=True)

    print("\n=== ⭐경로 기여: 지연셋 - t0_only (같은 날 표집 짝비교 CI) ===", flush=True)
    inc, best = [], 0
    for arm in ("hard", "three", "wbin"):
        b = keep[(arm, "t0_only")]
        for sname in ("lag_level", "lag_delta", "lag_all"):
            c = keep[(arm, sname)]
            rec = {"arm": arm, "featset": sname}; n_gt0 = 0
            for w in WINS:
                if f"{w}_pred" not in c or f"{w}_pred" not in b:
                    continue
                y = c[f"{w}_y"]
                assert (y == b[f"{w}_y"]).all(), "평가 라벨 불일치"
                lo, hi = DP.diff_ci(y, c[f"{w}_pred"], b[f"{w}_pred"], c[f"{w}_day"], rng)
                rec[f"{w}_d"] = c[f"{w}_auc"] - b[f"{w}_auc"]
                rec[f"{w}_lo"], rec[f"{w}_hi"] = lo, hi
                if np.isfinite(lo) and lo > 0:
                    n_gt0 += 1
            rec["n_win_gt0"] = n_gt0; best = max(best, n_gt0)
            inc.append(rec)
            print(f"   {arm:<7}{sname:<12} " + " · ".join(
                f"{w[:3]} Δ{rec.get(f'{w}_d', np.nan):+.4f}"
                f"[{rec.get(f'{w}_lo', np.nan):+.3f},{rec.get(f'{w}_hi', np.nan):+.3f}]"
                for w in WINS) + f"  창>0: {n_gt0}/3", flush=True)

    A = pd.DataFrame(rows); I = pd.DataFrame(inc)
    A.to_csv(OUT / "lag_eval.csv", index=False)
    I.to_csv(OUT / "path_increment.csv", index=False)
    go = bool(best >= 2)
    (OUT / "verdict.json").write_text(json.dumps(
        {"build_sequence_model": go, "best_windows_ci_gt0": int(best),
         "rule": "세 창 중 두 창 이상 짝비교 CI 하한>0 인 지연셋이 있어야 2단계(시퀀스 모델)로 간다",
         "sets": {k: len(v) for k, v in SETS.items()}}, indent=1, ensure_ascii=False))
    print("\n" + "=" * 96, flush=True)
    print(f"세 창 CI 통과 {int(A.ci3.sum())}/{len(A)} · min3 최고 {A.min3.max():.4f} "
          f"({A.loc[A.min3.idxmax(),'arm']}/{A.loc[A.min3.idxmax(),'featset']})", flush=True)
    print(f"경로 증분 최고: 두 창 이상 CI>0 인 셋 {int((I.n_win_gt0>=2).sum())}개 "
          f"(최대 {best}/3 창)", flush=True)
    print(f"⇒ 2단계 시퀀스 모델: {'✅ 지을 값어치 있음' if go else '❌ 짓지 않는다'}", flush=True)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
