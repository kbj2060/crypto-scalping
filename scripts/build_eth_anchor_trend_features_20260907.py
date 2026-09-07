#!/usr/bin/env python3
"""앵커 추세 피쳐: **DeMarker 지표 연속값 + 레짐 라벨 공식** (2026-09-07).

사용자: *"DeMarker 지표와 레짐을 피쳐로 넣는건 어때? 이게 추세를 읽기 좋아"*
       *"DeMarker 증거신호 말고 순수 DeMarker 지표 자체를 말한거야"*

## 왜 이게 빈틈인가
`demarker_extreme` 과 `kalman_deviation_meanrev` 는 **앵커를 이루는 8종 안에** 있다.
즉 지금까지 쓴 건 "DeMarker 가 0.90 이상/0.10 이하를 쳤다"는 **이진 발동**뿐이고,
그건 앵커 정의에 이미 흡수돼 있어 피쳐로서 새 정보가 없다.
**오실레이터 연속값**(얼마나 극단인지, 추세 대비 어느 쪽인지)은 한 번도 넣은 적이 없다.
레짐도 마찬가지 -- features154 의 `regime_*` 3개는 DC154 자체 파생이고,
배포 레짐 분류기(S12_K3)의 출력은 피쳐 프레임에 없다.

## 🔴배포 레짐 분류기를 그대로 쓰면 안 되는 이유
`tmp/eth_regime_s12k3_20260902/model.joblib` 의 `train_range` 는
**2024-01-01 ~ 2026-06-30** 으로 이 데이터셋 전 구간(~2026-06-29)을 덮는다.
그 모델의 `predict_proba` 를 VAL/OOS/3rd 에 얹으면 학습 구간을 다시 채점하는 꼴이다.

대신 **레짐 라벨 공식 자체를 직접 계산한다**. 아티팩트 `label_spec` 전문:
    er_12 = |c - c[-12]| / sum|diff|(12);  er_24 = 같은 식 over 24
    net_24 = c - c[-24];  slope_12 = EMA(c,12).pct_change()
    trend = (er_12 >= T1) | (er_24 >= T2)
    bull = trend & net_24>0 & slope_12>0;  bear = 대칭;  chop = 나머지
    K=3 연속봉 확정
    T1 = 0.28746928746924355 · T2 = 0.22529595661203153  (TRAIN 에서만 보정)
전부 과거만 본다 -- 학습 없음, 누수 0. 3클래스로 뭉개기 전의 **연속값**(er/net/slope)도
함께 낸다. 오히려 이쪽이 「추세인가 횡보인가」를 더 잘 담는다.

## 측면 정렬 (필수)
라벨이 측면 정렬이다: `fade_up = (side=="bottom")` 이므로
**바닥 앵커의 지속 = 하락 계속**, 천장 앵커의 지속 = 상승 계속.
따라서 `cont_sign = +1(top) / -1(bottom)` 을 곱해 "지속 방향을 얼마나 가리키는가"로 바꾼다.
방향 없는 크기 지표(er_12/er_24, |dem-0.5|)는 그대로 둔다.

## 인과성 검증
각 피쳐를 무작위 앵커 40개에서 **그 봉까지 잘라 재계산**해 전체 계산과 1e-9 이내로
같은지 확인한다 (게이트 L1 재구성 정신). 다르면 그 피쳐는 미래를 본 것이다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ETH_KL = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
LAB = ROOT / "tmp/eth_anchor_features154_20260907/features154.parquet"
OUT = ROOT / "tmp/eth_anchor_trend_features_20260907"

DEM_N = 14                       # 배포값 (live_evidence_signal_dashboard_20260823.py)
DEM_EXTRA = (28, 56)             # 더 긴 추세 창
T1_ER12 = 0.28746928746924355    # S12_K3 label_spec, TRAIN 보정
T2_ER24 = 0.22529595661203153
DEBOUNCE_K = 3


def _load_kl(p: Path) -> pd.DataFrame:
    kl = pd.read_csv(p)
    tc = next(c for c in kl.columns if kl[c].dtype == object and "time" in c.lower()) \
        if "timestamp" not in kl.columns else "timestamp"
    kl[tc] = pd.to_datetime(kl[tc], utc=True).dt.tz_localize(None)
    kl = kl.rename(columns={tc: "timestamp"}).sort_values("timestamp").reset_index(drop=True)
    return kl


def demarker(high: pd.Series, low: pd.Series, n: int) -> pd.Series:
    """정본 -- research_eth_demarker_evidence_signal_lift_check_20260831.py::compute_demarker 와 동일식."""
    up_move = high.diff()
    down_move = low.shift(1) - low
    de_max = up_move.clip(lower=0.0).fillna(0.0)
    de_min = down_move.clip(lower=0.0).fillna(0.0)
    sma_max = de_max.rolling(n, min_periods=n).mean()
    sma_min = de_min.rolling(n, min_periods=n).mean()
    return sma_max / (sma_max + sma_min).replace(0.0, np.nan)


def efficiency_ratio(c: pd.Series, w: int) -> pd.Series:
    """er_w = |c - c[-w]| / sum|diff| over w  -- 1 이면 완전 추세, 0 이면 완전 횡보."""
    num = (c - c.shift(w)).abs()
    den = c.diff().abs().rolling(w, min_periods=w).sum()
    return num / den.replace(0.0, np.nan)


def regime_states(c: pd.Series) -> dict[str, np.ndarray]:
    """S12_K3 라벨 공식 -- 전부 과거만 본다."""
    er12 = efficiency_ratio(c, 12)
    er24 = efficiency_ratio(c, 24)
    net24 = c - c.shift(24)
    slope12 = c.ewm(span=12, adjust=False).mean().pct_change()
    trend = (er12 >= T1_ER12) | (er24 >= T2_ER24)
    raw = np.where(trend & (net24 > 0) & (slope12 > 0), 1,
                   np.where(trend & (net24 < 0) & (slope12 < 0), -1, 0)).astype(float)
    raw[~np.isfinite(er12.to_numpy()) | ~np.isfinite(er24.to_numpy())] = np.nan
    # K=3 연속 확정 (과거 K봉이 같은 상태일 때만 전환) -- 순수 후진 상태기계
    conf = np.full(len(c), np.nan)
    cur = 0.0
    for i in range(len(c)):
        if not np.isfinite(raw[i]):
            conf[i] = np.nan
            continue
        if i >= DEBOUNCE_K - 1 and np.all(raw[i - DEBOUNCE_K + 1:i + 1] == raw[i]):
            cur = raw[i]
        conf[i] = cur
    # 확정 레짐이 유지된 봉 수 / 마지막 전환 이후 경과
    bars_in = np.zeros(len(c)); run = 0
    for i in range(len(c)):
        run = run + 1 if i > 0 and conf[i] == conf[i - 1] else 1
        bars_in[i] = run
    return {"er12": er12.to_numpy(), "er24": er24.to_numpy(), "net24_pct": (net24 / c).to_numpy(),
            "slope12": slope12.to_numpy(), "regime_raw": raw, "regime_conf": conf,
            "bars_in_regime": bars_in}


def build_bar_frame(kl: pd.DataFrame) -> pd.DataFrame:
    h, l, c = kl["high"].astype(float), kl["low"].astype(float), kl["close"].astype(float)
    out = {"timestamp": kl["timestamp"]}
    for n in (DEM_N, *DEM_EXTRA):
        out[f"dem{n}"] = demarker(h, l, n).to_numpy()
    out[f"dem{DEM_N}_slope6"] = pd.Series(out[f"dem{DEM_N}"]).diff(6).to_numpy()
    out.update(regime_states(c))
    return pd.DataFrame(out)


def align(B: pd.DataFrame, cont_sign: np.ndarray) -> pd.DataFrame:
    """측면 정렬 -- 방향성 피쳐에만 cont_sign 을 곱한다."""
    A = pd.DataFrame(index=B.index)
    for n in (DEM_N, *DEM_EXTRA):                       # 0.5 중심 대칭 반전
        A[f"dem{n}_al"] = 0.5 + (B[f"dem{n}"] - 0.5) * cont_sign
        A[f"dem{n}_dist"] = (B[f"dem{n}"] - 0.5).abs()  # 방향 없는 극단도
    A[f"dem{DEM_N}_slope6_al"] = B[f"dem{DEM_N}_slope6"] * cont_sign
    A["er12"] = B["er12"]                                # 추세강도(부호 없음)
    A["er24"] = B["er24"]
    A["net24_al"] = B["net24_pct"] * cont_sign
    A["slope12_al"] = B["slope12"] * cont_sign
    A["regime_al"] = B["regime_conf"] * cont_sign        # +1 추세가 지속방향, -1 반대, 0 횡보
    A["regime_raw_al"] = B["regime_raw"] * cont_sign
    A["regime_agree"] = (B["regime_conf"] * cont_sign > 0).astype(float)
    A["regime_against"] = (B["regime_conf"] * cont_sign < 0).astype(float)
    A["bars_in_regime"] = B["bars_in_regime"]
    return A


def causality_check(kl: pd.DataFrame, bar_idx: np.ndarray, rng, n_probe=40) -> dict:
    """무작위 앵커에서 **그 봉까지 잘라** 재계산 -- 전체 계산과 같아야 한다."""
    full = build_bar_frame(kl)
    cols = [c for c in full.columns if c != "timestamp"]
    probe = rng.choice(bar_idx[bar_idx > 3000], size=min(n_probe, (bar_idx > 3000).sum()), replace=False)
    bad, worst = [], 0.0
    for i in sorted(probe):
        cut = build_bar_frame(kl.iloc[: i + 1].reset_index(drop=True))
        for c in cols:
            a, b = full[c].to_numpy()[i], cut[c].to_numpy()[-1]
            if not np.isfinite(a) and not np.isfinite(b):
                continue
            d = abs(float(a) - float(b)) if np.isfinite(a) and np.isfinite(b) else np.inf
            worst = max(worst, d if np.isfinite(d) else 1e9)
            if d > 1e-9:
                bad.append((c, int(i), float(a), float(b)))
    return {"n_probe": len(probe), "n_bad": len(bad), "worst_abs_diff": float(worst),
            "bad_sample": bad[:8], "PASS": len(bad) == 0}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260907)
    D = pd.read_parquet(LAB)
    kl = _load_kl(ETH_KL)
    print(f"[1/5] 앵커 {len(D):,} · klines {len(kl):,} {kl.timestamp.min()} ~ {kl.timestamp.max()}", flush=True)

    B = build_bar_frame(kl)
    print(f"[2/5] 봉 피쳐 {B.shape[1]-1}개 계산", flush=True)

    # 앵커 봉에 조인 -- ⚠️행 tau 는 봉 tau 자신의 종가를 담는다(게이트 L3 조인시점 규약)
    M = D[["timestamp"]].merge(B, on="timestamp", how="left")
    assert len(M) == len(D), "조인 후 행수 변화"
    miss = M.drop(columns=["timestamp"]).isna().all(axis=1).mean()
    print(f"[3/5] 조인 결측(전 컬럼 NaN) {miss:.3%}", flush=True)

    idx = kl.set_index("timestamp").index.get_indexer(D["timestamp"])
    assert (idx >= 0).all(), "앵커 timestamp 가 klines 에 없음"
    chk = causality_check(kl, idx, rng)
    print(f"[4/5] 인과성 재구성: 탐침 {chk['n_probe']} · 불일치 {chk['n_bad']} "
          f"· 최대오차 {chk['worst_abs_diff']:.2e} → {'PASS' if chk['PASS'] else '🔴FAIL'}", flush=True)
    if not chk["PASS"]:
        print(f"    불일치 표본: {chk['bad_sample']}", flush=True)

    cont_sign = np.where(D["side"].to_numpy() == "top", 1.0, -1.0)
    A = align(M.drop(columns=["timestamp"]), cont_sign)
    A.insert(0, "timestamp", D["timestamp"].to_numpy())
    A["split"] = D["split"].to_numpy()

    # 기존 DC 추세 피쳐와의 중복도
    red = {}
    for c in ("regime_trending", "regime_persistence", "cvp_regime", "sig_trend_health", "atr_pct"):
        if c in D.columns:
            red[c] = {k: round(float(pd.Series(A[k]).corr(D[c])), 3)
                      for k in A.columns if k not in ("timestamp", "split")}
    fcols = [c for c in A.columns if c not in ("timestamp", "split")]
    A.to_parquet(OUT / "trend_features.parquet", index=False)
    (OUT / "meta.json").write_text(json.dumps(
        {"feature_cols": fcols, "n": len(A), "causality": chk,
         "redundancy_vs_dc": red, "dem_n": DEM_N, "dem_extra": list(DEM_EXTRA),
         "regime_spec": {"T1_er12": T1_ER12, "T2_er24": T2_ER24, "debounce_k": DEBOUNCE_K,
                         "source": "tmp/eth_regime_s12k3_20260902/model.joblib label_spec",
                         "model_used": False,
                         "why": "배포 모델 train_range 2024-01-01~2026-06-30 이 전 구간을 덮어 누수"}},
        indent=1, ensure_ascii=False))

    print(f"[5/5] 피쳐 {len(fcols)}개 · 결측률")
    for c in fcols:
        print(f"    {c:<22} 결측 {A[c].isna().mean():>6.2%} · 고유 {A[c].nunique():>5} "
              f"· 범위 [{A[c].min():>8.4f}, {A[c].max():>8.4f}]", flush=True)
    print("\n[중복도] 기존 DC 추세 피쳐와의 상관 (|r|>0.5 만)")
    for c, d in red.items():
        hi = {k: v for k, v in d.items() if abs(v) > 0.5}
        print(f"    {c:<20} {hi if hi else '없음'}", flush=True)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
