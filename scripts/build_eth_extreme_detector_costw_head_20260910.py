#!/usr/bin/env python3
"""극점 탐지기 **손실가중 헤드(p2) 사이드카** 빌더 (2026-09-10, 사용자 지시).

배포된 극점 아티팩트(p1, TabPFN)는 **건드리지 않는다**. 그 옆에 작은 HGB 헤드를 하나 더 두고,
라이브가 `강` 등급을 줄 때만 **이중조건**으로 쓴다:

    강 = p1 >= 강컷  AND  p2 >= 강컷      (둘 다 동의할 때만)
    아니면 중으로 강등 -> 중/약은 오늘의 하드게이트를 그대로 받는다

p2 는 같은 모집단·피쳐·라벨·분할에 **학습 표본 가중치만** 바꿔 학습한다:
강추세 역방향 자리의 **오답**(_y=0)에 4배(상한). 그러면 "빗나가면 비싼 자리"를 모델이 피한다.

## 실측 근거 (표본외 2026-06-20~09-08, 81일)
· 시드 짝비교 10개(무작위 추출): 강 등급 정밀도 **+2.61pp**(t=4.13, 10/10 양수),
  **건수를 맞추면 +1.76pp**(t=2.63, 7/10) -- 이득의 1/3 은 커버리지, 2/3 은 실력.
· 역추세 콜 비중 0.289 -> 0.070 (게이트 없이). 추세 정의를 4가지로 바꿔도 -40~-76%.
· ⭐살아남는 역추세 콜의 정밀도가 **더 높다**(0.769~0.800 vs 전체 0.641) --
  하드게이트는 바로 그 좋은 콜을 지우고 있었다.
· 순bp(H=48·비용10bp) 강: 배포판 +12.18 -> 이중조건 **+12.99**, 두 반기 0.688/0.672.

⚠️p2 를 등급 전체에 쓰면 안 된다 -- 중/약 라벨 정밀도가 4~6pp 떨어진다(혼합 α 스윕 7값 전부).
   라벨은 "극점인가"인데 가중치는 "비싼가"를 섞기 때문이다. **강 승격에만** 쓴다.
"""
from __future__ import annotations
import argparse, json, os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "tmp/eth_signal_map_20260909"
SEEDS = [20260909, 771233, 305610, 517758, 961476]
TIER_Q = (("강", 0.05), ("중", 0.10), ("약", 0.25))
TQ_HI, TQ_LO, CAP_W = 0.8, 0.2, 4.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/live/eth_extreme_detector_costw_artifact"))
    a = ap.parse_args()
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)

    A = pd.read_parquet(EX / "extreme_frame.parquet")
    fm = json.load(open(EX / "extreme_frame_meta.json"))
    feats, VAL0, W = fm["feats"], pd.Timestamp(fm["val0"]), fm["w"]
    A = A[A["_y"] >= 0].sort_values("_ts").reset_index(drop=True)
    X = np.nan_to_num(A[feats].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = A["_y"].to_numpy(int); ts = A["_ts"]
    lg = A["_long"].to_numpy(bool); tq = A["_tq"].to_numpy(float)
    counter = (lg & (tq <= TQ_LO)) | (~lg & (tq >= TQ_HI))
    tr = (ts < VAL0).to_numpy(); oos = ~tr

    w = np.ones(len(A)); w[(y == 0) & counter] = CAP_W
    for c in (0, 1):                       # 클래스별 평균 1 -- 재가중과 클래스 재균형 분리
        m = tr & (y == c); w[m] /= w[m].mean()
    print(f"학습 {tr.sum():,} · 표본외 {oos.sum():,} · 피쳐 {len(feats)}")
    print(f"가중 대상: 강추세 역방향 오답 {int(((y==0)&counter&tr).sum()):,}건 "
          f"({((y==0)&counter&tr).sum()/tr.sum()*100:.1f}%) x{CAP_W:g}\n", flush=True)

    models, P = [], np.zeros(len(A))
    for sd in SEEDS:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                           min_samples_leaf=40, l2_regularization=1.0,
                                           random_state=sd)
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        P += m.predict_proba(X)[:, 1] / len(SEEDS); models.append(m)
        print(f"   시드 {sd}", flush=True)

    o = np.flatnonzero(oos); k = len(o) // 2
    cut_m = np.zeros(len(A), bool); cut_m[o[:k]] = True
    ev_m = np.zeros(len(A), bool); ev_m[o[k:]] = True
    cuts = {g: float(np.quantile(P[cut_m], 1 - q)) for g, q in TIER_Q}
    auc = float(roc_auc_score(y[oos], P[oos]))
    print(f"\n표본외 AUC {auc:.4f} (p1 보다 낮은 게 정상 -- 다른 목적함수다)")
    print(f"강 컷 {cuts['강']:.4f} · 컷창 {ts[cut_m].min()} ~ {ts[cut_m].max()}")

    joblib.dump(models, OUT / "model.joblib")
    meta = {
        "rule_id": f"eth_extreme_costw_head_w{W}_20260910",
        "role": "강 승격 이중조건 전용 사이드카 -- 등급 자체는 p1(부모 아티팩트)이 매긴다",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "hgb", "features": feats, "seeds": SEEDS, "label_window_bars": W,
        "n_train": int(tr.sum()), "auc_oos": round(auc, 4), "cuts": cuts,
        "cost_weight": {"target": "counter_trend_negatives", "cap": CAP_W,
                        "tq_hi": TQ_HI, "tq_lo": TQ_LO,
                        "trend": "ret144/atr 의 2016봉 롤링 분위(_tq) -- 부모 게이트와 같은 통계"},
        "cut_window": [str(ts[cut_m].min()), str(ts[cut_m].max())],
        "eval_window": [str(ts[ev_m].min()), str(ts[ev_m].max())],
        "evidence": {"paired_seeds": 10, "delta_precision_raw": 0.0261, "t_raw": 4.13,
                     "delta_precision_count_matched": 0.0176, "t_count_matched": 2.63,
                     "counter_share": [0.289, 0.070], "counter_call_precision": [0.353, 0.769]},
        "note": ("이 헤드만으로 등급을 매기면 안 된다 -- 중/약 라벨 정밀도가 4~6pp 떨어진다. "
                 "강 승격 이중조건에만 쓴다. 상세: "
                 "docs/experiments/eth_extreme_detector_costweight_v2_20260910.md"),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"\n저장 {OUT}  ({sum(f.stat().st_size for f in OUT.iterdir())/1e6:.1f}MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
