#!/usr/bin/env python3
"""앵커 **접근 경로** 피쳐 -- 지연(lag) 블록 (2026-09-07).

사용자: *"지금 현재 학습 과정이 라벨 데이터의 입력피쳐가 앵커지점 딱 1개 5분봉 입력
       피쳐들로 학습하는건가?"* → *"그럼 학습이 너무 어려울 것 같은데 시퀀스로 먹일 순 없나?"*

## 현재 무엇이 빠져 있나
학습 행렬은 `(4755 앵커, 150 피쳐)` 로 **앵커당 1행**이다. 과거가 없는 건 아니다 --
150개가 전부 롤링 요약이라 1~288봉(최대 24시간) 창이 박혀 있다. 하지만 그건
**미리 정해둔 요약통계**다. 모델은 *접근 경로의 모양*("한 번에 내리꽂았나 세 번
나눠 내렸나", "6봉 전 실패한 반등이 있었나")을 볼 수단이 없다.

## 왜 시퀀스 모델보다 이걸 먼저 하나
1D-CNN 첫 층은 결국 **같은 지연들의 학습된 선형결합**이다. 지연이 `t` 단독 대비
아무것도 못 더하면 시퀀스 모델도 거의 확실히 못 더한다. 이건 새 아키텍처 없이
지금 파이프라인으로 도는 **싼 반증**이다. 표본도 제약이다(TRAIN 앵커 3,237개).

## 설계 (사전 지정)
피쳐 8개 × 시점 6개 = 48열. TabICL 사전학습 범위(2~100 컬럼) 안에 든다.
  피쳐: `dirtop20_train.json` 상위 8 -- **TRAIN 에서만 매긴 순위**라 누수 없다
  시점: t, t-1, t-3, t-6, t-12, t-24  (t 는 앵커 봉 자신)
두 형태를 낸다:
  `lag_level`  각 시점의 원값 (48열)
  `lag_delta`  f(t) - f(t-k) 차분 5개 × 8 = 40열 (모양을 직접 준다)
차분을 따로 내는 이유: 트리/ICL 은 두 컬럼의 차를 스스로 만들기 어렵다.

## 인과성
`t-k` 는 전부 과거 봉이므로 정의상 인과적이다. 그래도 봉 인덱스 정렬을 단언한다:
앵커 timestamp 가 봉단위 프레임의 `bar_idx` 와 정확히 일치해야 하고,
`t-k` 행의 timestamp 가 `앵커 timestamp - 5분*k` 여야 한다 (결측봉이 있으면 NaN).
⚠️이 저장소 규약: **행 τ 는 봉 τ 자신의 종가를 담는다**(게이트 L3 조인시점).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F154 = ROOT / "tmp/ilias_eth_154feature_dataset_20260821/ilias_eth_154feature_2024_2026H1_combined.csv"
LAB = ROOT / "tmp/eth_anchor_features154_20260907/features154.parquet"
RANK = ROOT / "tmp/eth_anchor_direction_feature_ranking_20260907/dirtop20_train.json"
OUT = ROOT / "tmp/eth_anchor_lag_features_20260907"

N_FEAT = 8
LAGS = (0, 1, 3, 6, 12, 24)       # 봉 단위. 0 = 앵커 봉 자신
BAR = pd.Timedelta("5min")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(LAB)
    rk = json.loads(RANK.read_text())
    feats = (rk if isinstance(rk, list) else list(rk))[:N_FEAT]
    print(f"[1/5] 앵커 {len(D):,} · 지연 대상 {N_FEAT}피쳐 (TRAIN-only 순위)", flush=True)
    for f in feats:
        print(f"      {f}", flush=True)

    F = pd.read_csv(F154, usecols=["timestamp", *feats])
    F["timestamp"] = pd.to_datetime(F["timestamp"], utc=True).dt.tz_localize(None)
    F = F.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    print(f"[2/5] 봉단위 프레임 {len(F):,} · {F.timestamp.min()} ~ {F.timestamp.max()}", flush=True)

    pos = pd.Series(np.arange(len(F)), index=F["timestamp"])
    idx = pos.reindex(D["timestamp"]).to_numpy()
    n_miss = int(np.isnan(idx).sum())
    assert n_miss == 0, f"앵커 {n_miss}개의 timestamp 가 봉 프레임에 없음"
    idx = idx.astype(int)

    V = F[feats].to_numpy(np.float64)
    ts = F["timestamp"].to_numpy()
    A = pd.DataFrame({"timestamp": D["timestamp"].to_numpy()})
    bad_align = 0
    for k in LAGS:
        j = idx - k
        ok = j >= 0
        # 정렬 단언: t-k 행의 timestamp 가 정확히 앵커 - 5분*k 인가 (결측봉이면 NaN 처리)
        want = D["timestamp"].to_numpy() - np.timedelta64(int(k * 5), "m")
        got = np.where(ok, ts[np.clip(j, 0, len(F) - 1)], np.datetime64("NaT"))
        mismatch = ok & (got != want)
        bad_align += int(mismatch.sum())
        for c, f in enumerate(feats):
            v = np.where(ok, V[np.clip(j, 0, len(F) - 1), c], np.nan)
            A[f"{f}__t{k}"] = np.where(mismatch, np.nan, v)
    print(f"[3/5] 지연 정렬: 타임스탬프 불일치(결측봉) {bad_align}/{len(D)*len(LAGS)} 셀 → NaN 처리", flush=True)

    # 차분 -- 모양을 직접 준다
    for k in LAGS[1:]:
        for f in feats:
            A[f"{f}__d{k}"] = A[f"{f}__t0"] - A[f"{f}__t{k}"]

    level_cols = [c for c in A.columns if "__t" in c]
    delta_cols = [c for c in A.columns if "__d" in c]
    A["split"] = D["split"].to_numpy()

    A.to_parquet(OUT / "lag_features.parquet", index=False)
    (OUT / "meta.json").write_text(json.dumps(
        {"base_feats": feats, "lags": list(LAGS),
         "level_cols": level_cols, "delta_cols": delta_cols,
         "n_level": len(level_cols), "n_delta": len(delta_cols),
         "misaligned_cells": bad_align,
         "note": "t-k 는 정의상 과거 봉 -- 인과적. 결측봉은 timestamp 대조로 잡아 NaN."},
        indent=1, ensure_ascii=False))

    print(f"[4/5] 레벨 {len(level_cols)}열 · 차분 {len(delta_cols)}열", flush=True)
    miss = A[level_cols + delta_cols].isna().mean()
    print(f"      결측률 중앙 {miss.median():.2%} · 최대 {miss.max():.2%} ({miss.idxmax()})", flush=True)
    print(f"\n[5/5] 저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
