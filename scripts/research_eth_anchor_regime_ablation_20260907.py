#!/usr/bin/env python3
"""레짐 블록 절제 -- **넣을 값어치가 있나** (2026-09-07).

사용자: *"레짐 피쳐가 효과가 없으면 빼줘. 괜히 넣었다가 버그 생겨서 다 다시 하게 되면 큰일이야"*

## 왜 절제인가
`research_eth_anchor_trend_feature_eval_20260907.py` 는 추세16 + 오실15 를 **한 덩어리**
(`new32`)로 본다. 그 결과만으로는 레짐 9개가 기여했는지 짐이었는지 알 수 없다.
여기서는 블록을 갈라 **레짐을 뺐을 때 손해가 나는지**만 본다.

  `regime9`   er12, er24, net24_al, slope12_al, regime_al, regime_raw_al,
              regime_agree, regime_against, bars_in_regime
  `dem7`      dem14/28/56_al, dem14/28/56_dist, dem14_slope6_al
  `osc15`     8종 증거신호의 순수 연속 코어
  `no_regime` dem7 + osc15 (= new32 에서 레짐만 뺀 것)
  `new32`     전부 (대조)

## 판정 규칙 (사전 지정)
**빼는 쪽이 기본값이다.** 레짐을 남기려면 `new32` 가 `no_regime` 을
**세 창 중 최소 두 창에서 짝비교 CI 하한 > 0** 으로 이겨야 한다.
그 외에는 전부 제거한다 -- 효과가 0 이거나 애매하면 코드 표면적을 줄이는 쪽이 옳다.
(사용자 우려: 쓸모없는 피쳐가 나중에 버그의 진입점이 된다)

증분은 **같은 날 표집 짝비교 CI** 로만 본다. 원시 AUC 차이로 판단하지 않는다.
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

TRD = ROOT / "tmp/eth_anchor_trend_features_20260907"
OSC = ROOT / "tmp/eth_anchor_osc_cores_20260907"
OUT = ROOT / "tmp/eth_anchor_regime_ablation_20260907"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
N_EST = 4

REGIME9 = ["er12", "er24", "net24_al", "slope12_al", "regime_al", "regime_raw_al",
           "regime_agree", "regime_against", "bars_in_regime"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260907)
    D = pd.read_parquet(T.SRCD / "features154.parquet")
    A = pd.read_parquet(TRD / "trend_features.parquet")
    O = pd.read_parquet(OSC / "osc_cores.parquet")
    tm = json.loads((TRD / "meta.json").read_text())
    om = json.loads((OSC / "meta.json").read_text())
    assert tm["causality"]["PASS"] and om["causality"]["PASS"] and om["parity"]["ALL_MATCH"]

    trend_cols = tm["feature_cols"]
    osc_cols = [c for c in om["feature_cols"] if c != "dem_al"]
    dem7 = [c for c in trend_cols if c not in REGIME9]
    assert sorted(dem7 + REGIME9) == sorted(trend_cols), "추세 블록 분해 불일치"

    meta = json.loads((T.SRCD / "meta.json").read_text())
    base_cols = [c for c in meta["feature_cols"] if c in D.columns]
    X = np.concatenate([D[base_cols].to_numpy(np.float64),
                        A[trend_cols].to_numpy(np.float64),
                        O[osc_cols].to_numpy(np.float64)], axis=1)
    ci = {c: i for i, c in enumerate(base_cols + trend_cols + osc_cols)}

    SETS = {
        "regime9": REGIME9,
        "dem7": dem7,
        "osc15": osc_cols,
        "no_regime": dem7 + osc_cols,
        "new32": trend_cols + osc_cols,
    }
    print(f"[입력] 앵커 {len(D):,}", flush=True)
    for k, v in SETS.items():
        print(f"   {k:<12} {len(v):>3}개", flush=True)

    rows, keep = [], {}
    print("\n=== 3팔 × 5피쳐셋 × 세 창 ===", flush=True)
    for arm in ("hard", "three", "wbin"):
        for sname, feats in SETS.items():
            fc = [ci[c] for c in feats]
            t0 = time.time()
            r = DP.run3(D, X, arm, fc, rng, n_est=N_EST)
            rec = {"arm": arm, "featset": sname, "n_feat": len(fc), "sec": round(time.time() - t0, 1)}
            for w in WINS:
                rec[f"{w}_auc"] = r.get(f"{w}_auc", np.nan)
                rec[f"{w}_lo"] = r.get(f"{w}_lo", np.nan)
            rec["min3"] = np.nanmin([rec[f"{w}_auc"] for w in WINS])
            rec["ci3"] = bool(all(rec[f"{w}_lo"] > 0.5 for w in WINS if np.isfinite(rec[f"{w}_lo"])))
            rows.append(rec); keep[(arm, sname)] = r
            print(f"   {arm:<7}{sname:<12} " + " · ".join(
                f"{w[:3]} {rec[f'{w}_auc']:.4f}[{rec[f'{w}_lo']:.3f}]" for w in WINS)
                + f" · min3 {rec['min3']:.4f}", flush=True)

    # ---------------- 핵심: 레짐을 넣어서 이기는가
    print("\n=== ⭐레짐 기여: new32 - no_regime (같은 날 표집 짝비교 CI) ===", flush=True)
    inc, wins_gt0 = [], 0
    for arm in ("hard", "three", "wbin"):
        b, c = keep[(arm, "no_regime")], keep[(arm, "new32")]
        rec = {"arm": arm}
        for w in WINS:
            if f"{w}_pred" not in c or f"{w}_pred" not in b:
                continue
            y = c[f"{w}_y"]
            assert (y == b[f"{w}_y"]).all(), "평가 대상 라벨 불일치"
            dl = c[f"{w}_auc"] - b[f"{w}_auc"]
            lo, hi = DP.diff_ci(y, c[f"{w}_pred"], b[f"{w}_pred"], c[f"{w}_day"], rng)
            rec[f"{w}_d"], rec[f"{w}_lo"], rec[f"{w}_hi"] = dl, lo, hi
            if np.isfinite(lo) and lo > 0:
                wins_gt0 += 1
        inc.append(rec)
        print(f"   {arm:<7}" + " · ".join(
            f"{w[:3]} Δ{rec.get(f'{w}_d', np.nan):+.4f}"
            f"[{rec.get(f'{w}_lo', np.nan):+.3f},{rec.get(f'{w}_hi', np.nan):+.3f}]"
            for w in WINS), flush=True)

    A_ = pd.DataFrame(rows)
    A_.to_csv(OUT / "ablation.csv", index=False)
    pd.DataFrame(inc).to_csv(OUT / "regime_increment.csv", index=False)

    print("\n" + "=" * 96, flush=True)
    # 팔별로 "세 창 중 두 창 이상 CI 하한 > 0" 인가
    keepit = []
    for rec in inc:
        n = sum(1 for w in WINS if np.isfinite(rec.get(f"{w}_lo", np.nan)) and rec[f"{w}_lo"] > 0)
        keepit.append((rec["arm"], n))
        print(f"   {rec['arm']:<7} 짝비교 CI 하한>0 인 창 {n}/3", flush=True)
    verdict = any(n >= 2 for _, n in keepit)
    print(f"\n판정 규칙: 어느 한 팔이라도 두 창 이상 CI 하한>0 이어야 레짐을 남긴다")
    print(f"⇒ 레짐 블록: {'✅ 유지' if verdict else '❌ 제거'}", flush=True)
    print(f"   (regime9 단독 min3 최고 {A_[A_.featset=='regime9'].min3.max():.4f} · "
          f"세 창 CI 통과 {int(A_[A_.featset=='regime9'].ci3.sum())}/3팔)", flush=True)
    (OUT / "verdict.json").write_text(json.dumps(
        {"keep_regime": bool(verdict), "per_arm_windows_ci_gt0": dict(keepit),
         "rule": "어느 한 팔이라도 세 창 중 두 창 이상 짝비교 CI 하한>0 이면 유지, 아니면 제거",
         "regime9_cols": REGIME9}, indent=1, ensure_ascii=False))
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
