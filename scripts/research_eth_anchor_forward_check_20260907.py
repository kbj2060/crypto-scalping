#!/usr/bin/env python3
"""전방 확인 -- 안 쓴 31일에 Top-1 을 그대로 걸어본다 (2026-09-07).

사용자: *"전방 확인해줘"*

## 무엇을 보나
Top-1 = **TabPFN/MASHT · wbin · masht2784** (워크포워드 상위30% 진입 정확도
59.88% [53.94%, 64.56%], 귀무 p=0.017, 건당 +12.0bp).
그 모델을 **2026-06-30 이후 안 쓴 31일**(앵커 156, y_bin 유효 84)에 그대로 건다.

## 규약 -- 이 창은 어떤 선택에도 쓰이지 않았다
- 학습: 앵커 timestamp < 2026-06-30 − 엠바고 4시간 (라벨 지평 H=48봉이 새지 않게)
- Rocket 변환도 **학습 구간에서만 fit** 하고 전방은 transform 만 한다
  (앞선 측정은 전 구간 fit_transform 이었다 -- 랜덤커널이라 라벨 누수는 아니지만
   전방 확인에서는 배포와 같은 형태로 맞춘다)
- 전방 앵커의 라벨은 예측을 만든 뒤에만 본다

## ⚠️표본이 작다
상위30% 진입 ≈ 25건. n=25 의 정확도 CI 반폭은 ±19pp 수준이다.
**확증이 아니라 방향 확인**이다. 통과해도 "모순되지 않았다"까지고, 실패하면 나쁜 신호다.

## ⚠️기저가 다르다
전방 구간 지속률 **57.14%** vs 학습 구간 52.72%. 기저가 오르면 아무 모델이나
정확도가 오른다. 그래서 **기저 대비 리프트**와 **무작위 진입 귀무**를 같이 낸다.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / "tmp/eth_anchor_window_tensor_20260907"
FWD = ROOT / "tmp/eth_anchor_forward_window_20260907"
OUT = ROOT / "tmp/eth_anchor_forward_check_20260907"
CUT = pd.Timestamp("2026-06-30")
EMBARGO = pd.Timedelta(hours=4)
SEED = 20260907
BAR, COST = 100.0, 7.8
BE = (BAR + COST) / (2 * BAR)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    import torch
    from tabpfn import TabPFNClassifier
    from aeon.transformations.collection.convolution_based import MultiRocket, HydraTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr = np.load(WT / "X.npy"); vtr = np.load(WT / "valid.npy")
    Itr = pd.read_parquet(WT / "index.parquet")
    Xf = np.load(FWD / "X.npy"); vf = np.load(FWD / "valid.npy")
    If = pd.read_parquet(FWD / "index.parquet")
    print(f"[입력] 학습창 {Xtr.shape} 유효 {vtr.sum()} · 전방창 {Xf.shape} 유효 {vf.sum()} · {dev}", flush=True)

    # 학습 대상: 컷 - 엠바고 이전 + y_bin 유효
    ok_tr = vtr & np.isfinite(Itr["y_bin"].to_numpy()) & (Itr["timestamp"] < CUT - EMBARGO).to_numpy()
    ok_f = vf & np.isfinite(If["y_bin"].to_numpy())
    print(f"[분할] 학습 {ok_tr.sum()} (~{Itr.timestamp[ok_tr].max()}) · "
          f"전방 {ok_f.sum()} ({If.timestamp[ok_f].min()} ~ {If.timestamp[ok_f].max()})", flush=True)

    t0 = time.time()
    mr = MultiRocket(n_kernels=252, random_state=SEED, n_jobs=10)
    hy = HydraTransformer(n_kernels=8, n_groups=16, random_state=SEED, n_jobs=10)
    A = Xtr[ok_tr].astype(np.float32)
    mr.fit(A); hy.fit(A)
    def tf(Z):
        return np.concatenate([np.asarray(mr.transform(Z.astype(np.float32))),
                               np.asarray(hy.transform(Z.astype(np.float32)))], axis=1).astype(np.float32)
    Ftr = tf(Xtr[ok_tr]); Ffw = tf(Xf[ok_f])
    print(f"[변환] 학습 fit → 학습 {Ftr.shape} · 전방 {Ffw.shape}  ({time.time()-t0:.0f}s)", flush=True)

    ytr = Itr["y_bin"].to_numpy()[ok_tr].astype(int)
    yf = If["y_bin"].to_numpy()[ok_f].astype(int)
    clf = TabPFNClassifier(device=dev, n_estimators=4, random_state=SEED,
                           ignore_pretraining_limits=True, memory_saving_mode=True)
    clf.fit(Ftr, ytr)
    p = clf.predict_proba(Ffw)[:, 1]

    base = float(yf.mean())
    k = max(5, int(len(p) * 0.30))
    top = np.argsort(-p)[:k]
    acc = float(yf[top].mean())
    auc = float(roc_auc_score(yf, p)) if len(np.unique(yf)) > 1 else np.nan
    ev = acc * BAR - (1 - acc) * BAR - COST

    # 무작위 진입 귀무 -- 같은 건수를 무작위로 뽑으면?
    nl = [float(yf[rng.choice(len(yf), k, replace=False)].mean()) for _ in range(4000)]
    p_rand = float(np.mean(np.array(nl) >= acc))
    # 부트스트랩 CI (일 군집)
    day = If["timestamp"][ok_f].dt.floor("D").to_numpy()
    u = np.unique(day); idx = {d: np.flatnonzero(day == d) for d in u}
    bs = []
    for _ in range(2000):
        ii = np.concatenate([idx[d] for d in rng.choice(u, len(u), replace=True)])
        kk = max(5, int(len(ii) * 0.30))
        bs.append(float(yf[ii][np.argsort(-p[ii])[:kk]].mean()))
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    print("\n" + "=" * 92, flush=True)
    print("전방 확인 결과 (2026-06-30 ~ 2026-07-31, 어떤 선택에도 안 쓰인 창)", flush=True)
    print("=" * 92, flush=True)
    print(f"  전방 앵커 {len(yf)} · 기저(항상 지속) {base:.2%} · 진입 {k}건", flush=True)
    print(f"  전방 AUC                 : {auc:.4f}", flush=True)
    print(f"  상위30% 진입 정확도       : {acc:.2%}  [{lo:.2%}, {hi:.2%}]", flush=True)
    print(f"  기저 대비 리프트          : {acc - base:+.2%}", flush=True)
    print(f"  무작위 진입 귀무          : p={p_rand:.3f}  (귀무 p95 {np.percentile(nl,95):.2%})", flush=True)
    print(f"  건당 기대                : {ev:+.1f}bp  (손익분기 {BE:.2%})", flush=True)
    print(f"\n  ⇒ 워크포워드 측정치 59.88% [53.94%, 64.56%] 와 "
          f"{'일관' if lo <= 0.5988 <= hi or abs(acc-0.5988) < 0.10 else '불일치'}", flush=True)
    print(f"  ⚠️n={k} 이라 CI 반폭 {(hi-lo)/2:.1%} -- 확증 아님, 방향 확인만", flush=True)

    (OUT / "result.json").write_text(json.dumps(
        {"n_forward": int(len(yf)), "base_rate": base, "n_entries": int(k),
         "forward_auc": auc, "top30_acc": acc, "ci": [lo, hi], "lift_vs_base": acc - base,
         "p_random_entry": p_rand, "ev_bp": ev, "breakeven": BE,
         "walkforward_ref": 0.5988, "window": "2026-06-30..2026-07-31",
         "note": "Rocket 변환을 학습구간에서만 fit -- 배포 형태"}, indent=1, ensure_ascii=False))
    np.save(OUT / "forward_preds.npy", p)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
