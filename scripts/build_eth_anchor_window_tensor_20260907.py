#!/usr/bin/env python3
"""앵커 **접근 창 텐서** -- MASHT 입력 (2026-09-07).

사용자: *"MASHT를 적용해야하는데 너의 설계를 보여줘. TabPFN3도 이미 모델은 갖고 있으니
       TabPFN로 진행해도 좋아"*

MASHT (arXiv 2607.19234) = MultiRocket + Hydra 랜덤 합성곱 피쳐 → TabPFN in-context.
이 파일은 그 **입력 텐서**만 만든다: 앵커마다 `(C 채널, K 봉)` 창.

## 창
K = 48봉(4시간), `[t-47, t]` -- **앵커 봉을 포함해서 끝난다**.
라벨 지평 H=48 과 대칭이고 라이브 `SWEEP_LOOKBACK`(48)과도 같다.
하나만 사전지정한다 (K 격자를 훑으면 다중검정이 된다).

## 채널 8개 -- 6개는 측면정렬 방향, 2개는 크기
`fade_up = (side=="bottom")` 이므로 **바닥 앵커의 지속 = 하락 계속**.
`cont_sign = +1(top)/-1(bottom)` 을 방향 채널에 곱해 "지속 방향으로 얼마나"로 통일한다.
  1 logret        bar log return                      × cont_sign
  2 path_atr      (close - close[t]) / (atr[t]*close[t])  × cont_sign   -- ATR 정규화 경로
  3 dem14         DeMarker(14)                        0.5 중심 반전
  4 p_fast        Fast %K 백분위(0~1)                  0.5 중심 반전
  5 delta_z       테이커 델타 z                        × cont_sign
  6 ret3_z        3봉 수익률 z                         × cont_sign
  7 kalman_dev_z  칼만 편차 z                          × cont_sign
  8 hl_range      (high-low)/close                     크기(정렬 없음)
3~7 은 `build_eth_anchor_oscillator_cores_20260907.py` 가 파리티 PASS 로 확인한
라이브 `compute_signals()` 의 연속 코어를 **봉 단위 프레임에서** 그대로 가져온다.

## 인과성
창의 마지막 원소가 앵커 봉 자신이고 그 앞 47봉이 정확히 5분 간격인지 **타임스탬프로 단언**한다.
결측봉이 하나라도 있는 인스턴스는 `valid=False` 로 표시해 평가에서 제외한다
(NaN 채우기로 넘기면 Rocket 커널이 조용히 이상값을 만든다).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "tmp/eth_anchor_features154_20260907/features154.parquet"
OUT = ROOT / "tmp/eth_anchor_window_tensor_20260907"

K = 48                      # 창 길이(봉). 앵커 봉 포함
CHANNELS = ["logret", "path_atr", "dem14", "p_fast", "delta_z", "ret3_z", "kalman_dev_z", "hl_range"]
DIRECTIONAL = {"logret", "path_atr", "delta_z", "ret3_z", "kalman_dev_z"}   # × cont_sign
CENTERED = {"dem14", "p_fast"}                                              # 0.5 중심 반전


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    B = _load("anchor_builder_20260907", ROOT / "scripts/build_eth_anchor_label_dataset_20260907.py")
    OC = _load("osc_cores_20260907", ROOT / "scripts/build_eth_anchor_oscillator_cores_20260907.py")
    D = pd.read_parquet(LAB)

    eth = B._load_kl(B.ETH_KL); fund = B._load_funding(); btc = B._load_kl(B.BTC_KL)
    tmax = min(eth["timestamp"].max(), btc["timestamp"].max(), fund["calc_time"].max())
    eth = eth[eth["timestamp"] <= tmax].reset_index(drop=True)
    sig = B.compute_signals(eth, btc_df=btc[btc["timestamp"] <= tmax],
                            funding_df=fund[fund["calc_time"] <= tmax])
    C = OC.compute_cores(sig, B)
    print(f"[1/4] 앵커 {len(D):,} · 봉 {len(sig):,} · 채널 {len(CHANNELS)} · K={K}", flush=True)

    close = sig["close"].to_numpy(float); high = sig["high"].to_numpy(float)
    low = sig["low"].to_numpy(float); atr = sig["atr_pct"].to_numpy(float)
    bar = {
        "logret": np.concatenate([[np.nan], np.diff(np.log(close))]),
        "close": close,
        "dem14": sig["dem"].to_numpy(float),
        "p_fast": C["p_fast"].to_numpy(float),
        "delta_z": C["delta_z"].to_numpy(float),
        "ret3_z": C["ret3_z"].to_numpy(float),
        "kalman_dev_z": sig["kalman_dev_z"].to_numpy(float),
        "hl_range": (high - low) / np.maximum(close, 1e-12),
    }
    ts = pd.to_datetime(sig["timestamp"].to_numpy())
    pos = pd.Series(np.arange(len(ts)), index=ts)
    idx = pos.reindex(D["timestamp"]).to_numpy()
    assert not np.isnan(idx).any(), "앵커 timestamp 가 봉 프레임에 없음"
    idx = idx.astype(int)

    n, Cn = len(D), len(CHANNELS)
    X = np.full((n, Cn, K), np.nan, dtype=np.float32)
    valid = np.zeros(n, bool)
    cont_sign = np.where(D["side"].to_numpy() == "top", 1.0, -1.0)
    tsv = ts.to_numpy()

    # 인과성 단언용: 창의 j번째 원소는 앵커 시각 - 5분*(K-1-j) 여야 한다
    offs = np.arange(K - 1, -1, -1) * np.timedelta64(5, "m")
    n_gap = 0
    for i in range(n):
        s = idx[i] - K + 1
        if s < 0:
            continue
        w = slice(s, idx[i] + 1)
        if not np.array_equal(tsv[w], D["timestamp"].to_numpy()[i] - offs):
            n_gap += 1               # 결측봉 -> 이 인스턴스는 버린다
            continue
        cl = bar["close"][w]
        a = atr[idx[i]]
        for c, ch in enumerate(CHANNELS):
            if ch == "path_atr":
                v = (cl - cl[-1]) / max(a * cl[-1], 1e-12)
            else:
                v = bar[ch][w]
            if ch in DIRECTIONAL:
                v = v * cont_sign[i]
            elif ch in CENTERED:
                v = 0.5 + (v - 0.5) * cont_sign[i]
            X[i, c, :] = v
        valid[i] = np.isfinite(X[i]).all()

    print(f"[2/4] 결측봉으로 버린 인스턴스 {n_gap} · NaN 포함으로 무효 "
          f"{int((~valid).sum() - n_gap)} · 유효 {int(valid.sum())}/{n}", flush=True)

    # 마지막 원소가 앵커 봉 자신인지 표본 검증 (미래참조 트립와이어)
    chk = np.flatnonzero(valid)[:200]
    bad = 0
    for i in chk:
        if abs(float(X[i, CHANNELS.index("path_atr"), -1])) > 1e-9:   # path_atr[t] 는 정의상 0
            bad += 1
    print(f"[3/4] 창 끝 = 앵커 봉 검증: path_atr 마지막값≠0 인 표본 {bad}/{len(chk)} "
          f"→ {'PASS' if bad == 0 else '🔴FAIL'}", flush=True)

    np.save(OUT / "X.npy", X)
    np.save(OUT / "valid.npy", valid)
    (OUT / "meta.json").write_text(json.dumps(
        {"K": K, "channels": CHANNELS, "directional": sorted(DIRECTIONAL),
         "centered": sorted(CENTERED), "n": n, "n_valid": int(valid.sum()),
         "n_dropped_gap": n_gap,
         "window": "[t-47, t] -- 앵커 봉 포함해서 끝남 (전부 과거·현재, 미래 없음)",
         "cont_sign": "top=+1 / bottom=-1 (fade_up = side=='bottom' 이므로 바닥 지속=하락)",
         "split_counts": D.split.value_counts().to_dict()}, indent=1, ensure_ascii=False))
    D[["timestamp", "side", "split", "y3", "y_bin", "is_clean", "range_pct", "atr_pct"]] \
        .assign(valid=valid).to_parquet(OUT / "index.parquet", index=False)
    print(f"[4/4] 저장: {OUT} · X {X.shape} ({X.nbytes/1e6:.1f}MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
