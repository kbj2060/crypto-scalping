#!/usr/bin/env python3
"""종합 청산 모델 **공유 피쳐 모듈** (2026-09-10) -- 학습 빌더와 라이브 워커가 같은 함수를 부른다.

대시보드가 보여주는 것 중 과거 재구성이 가능한 입력을 봉 단위 피쳐로 만든다:
  ev_*   증거신호 8종×양측 발동 나이·최근 12봉 표수·순표(compute_signals 그대로, funding 없음 = 극점 워커와 동일)
  ctx_*  증거신호 모듈의 맥락열 10개(p_fast..kalman_dev_z, 극점 탐지기 BASE_COLS)
  ext_*  극점 탐지기 v1 p1 · v2 손실가중 p2 · 게이트 · 나이(측면별, 12봉 안 최근 발동)
  vf_*   24시간 변동성 전망(HAR-RV + DVOL 아티팩트의 clf p · reg rv_fwd_pred · vrp)
  bo_*   돌파/되돌림 봉 피쳐(ret·btc_ret·rng_z·vol_z·taker_z·rv48/288·eth_btc_sp)
  x_m_*  포지셔닝 메트릭 4종 z/변화(bookDepth 아님 -- futures/data 4종)
  cal_*  시각·요일
재구성 불가라 제외: 청산맵·청산 버스트(서버 수집기 전용), V자반등·칩 메타라벨(TabPFN 미설치),
레짐 GBM(FeatureEngineer 136피쳐 -- 09-04 arm C 에서 유해, 라이브 파리티 위험).

⚠️파리티 원칙: 식은 이 파일 한 벌이다. 학습 빌더·라이브 워커는 `bar_features()` 를 그대로 부른다.
   DVOL 은 시간봉 종가가 **다음 시간 시작에** 알려진다고 본다(available = timestamp+1h).
   메트릭은 API timestamp(=아카이브 create_time−5분)가 봉 open_time 과 같은 행을 쓴다(라이브 러너 관례).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
import live_eth_extreme_detector_20260909 as X  # noqa: E402
import live_eth_breakout_features_20260908 as BO  # noqa: E402
import live_eth_vol_forecast_20260910 as VF  # noqa: E402

SIGNALS = X.B.SIGNALS
AGE_CAP = 48
EXT_WINDOW = 12
ANN = np.sqrt(288 * 365) * 100
POS_COLS = ["pos_side", "pos_hold", "pos_left", "pos_u", "pos_mfe", "pos_mae",
            "pos_u_atr", "pos_mfe_atr", "pos_mae_atr", "pos_giveback_atr"]


def _age(fired: np.ndarray) -> np.ndarray:
    """마지막 발동 이후 봉수(발동 봉=0). 없으면 NaN, AGE_CAP 초과는 NaN(=먼 과거는 정보 없음)."""
    idx = np.where(fired, np.arange(len(fired)), -1)
    last = np.maximum.accumulate(idx)
    age = np.where(last >= 0, np.arange(len(fired)) - last, np.nan).astype(float)
    return np.where(age > AGE_CAP, np.nan, age)


def bar_features(kl: pd.DataFrame, btc: pd.DataFrame, met: dict | None, met_ts, dv: pd.DataFrame | None,
                 ext_art: dict | None, costw_art: dict | None, vol_art: dict | None) -> pd.DataFrame:
    """kl/btc: timestamp(naive UTC open_time), open/high/low/close/volume/taker_buy_base. 봉 단위 피쳐 프레임."""
    kl = kl.reset_index(drop=True)
    sig = compute_signals(kl, btc_df=btc, funding_df=None)
    n = len(kl)
    F = pd.DataFrame({"ts": kl["timestamp"].to_numpy()})
    # --- 증거신호 ---
    any_b = np.zeros(n, bool); any_t = np.zeros(n, bool)
    for s in SIGNALS:
        for side, acc in (("bottom", any_b), ("top", any_t)):
            f = sig[f"{side}_{s}"].fillna(False).to_numpy(bool)
            F[f"ev_{side}_{s}_age"] = _age(f)
            acc |= f
    F["ev_bottom_n12"] = pd.Series(any_b.astype(float)).rolling(12, min_periods=1).sum().to_numpy()
    F["ev_top_n12"] = pd.Series(any_t.astype(float)).rolling(12, min_periods=1).sum().to_numpy()
    F["ev_net12"] = F["ev_bottom_n12"] - F["ev_top_n12"]
    for c in X.BASE_COLS:
        F[f"ctx_{c}"] = sig[c].to_numpy(float)
    F["atr"] = sig["atr_pct"].to_numpy(float)          # 포지션 스케일용 ATR14(비율)
    # --- 극점 탐지기 ---
    for side in ("bottom", "top"):
        for c in ("p", "p2", "gated", "age"):
            F[f"ext_{side}_{c}"] = np.nan
    if ext_art is not None:
        A = X.build_rows(sig, btc)
        if not A.empty:
            P = np.mean([m.predict_proba(A[X.FEATS])[:, 1] for m in ext_art["models"]], axis=0)
            P2 = (np.mean([m.predict_proba(A[X.FEATS])[:, 1] for m in costw_art["models"]], axis=0)
                  if costw_art is not None else np.full(len(A), np.nan))
            G = X.gated_of(A._tq.to_numpy(), A._long.to_numpy()).astype(float)
            for long, side in ((True, "bottom"), (False, "top")):
                m = (A._long == long).to_numpy()
                # 같은 봉 양측 발동은 없다(측면별 행) -- 봉 인덱스에 직접 놓고 12봉 앞으로 끌어온다
                col_p = np.full(n, np.nan); col_p2 = np.full(n, np.nan); col_g = np.full(n, np.nan)
                ii = A._i.to_numpy()[m]
                col_p[ii] = P[m]; col_p2[ii] = P2[m]; col_g[ii] = G[m]
                fired = np.zeros(n, bool); fired[ii] = True
                age = _age(fired)
                keep = age <= EXT_WINDOW
                for name, col in (("p", col_p), ("p2", col_p2), ("gated", col_g)):
                    ff = pd.Series(col).ffill(limit=EXT_WINDOW).to_numpy()
                    F[f"ext_{side}_{name}"] = np.where(keep, ff, np.nan)
                F[f"ext_{side}_age"] = np.where(keep, age, np.nan)
    # --- 24시간 변동성 전망 ---
    for c in ("p", "rv_fwd", "dvol", "vrp", "rv24"):
        F[f"vf_{c}"] = np.nan
    if dv is not None and len(dv) and vol_art is not None:
        r5 = np.log(kl["close"]).diff()
        rv = {h: (r5.rolling(h * 12, min_periods=h * 12).std() * ANN).to_numpy() for h in (1, 24, 168)}
        d = dv.sort_values("timestamp").copy()
        d["avail"] = d["timestamp"] + pd.Timedelta(hours=1)
        close_ts = pd.DataFrame({"t": kl["timestamp"] + pd.Timedelta(minutes=5)})
        j = pd.merge_asof(close_ts, d[["avail", "dvol"]], left_on="t", right_on="avail", direction="backward")
        dvol = j["dvol"].to_numpy(float)
        Xv = np.column_stack([np.log(np.clip(rv[1], 1e-6, None)), np.log(np.clip(rv[24], 1e-6, None)),
                              np.log(np.clip(rv[168], 1e-6, None)), np.log(np.clip(dvol, 1e-6, None)),
                              dvol - rv[24]])
        ok = np.isfinite(Xv).all(axis=1)
        M = vol_art["m"]
        p = np.full(n, np.nan); rf = np.full(n, np.nan)
        if ok.any():
            Z = (Xv[ok] - M["mu"]) / M["sd"]
            p[ok] = M["clf"].predict_proba(Z)[:, 1]
            rf[ok] = np.exp(M["reg"].predict(Xv[ok]))
        F["vf_p"], F["vf_rv_fwd"], F["vf_dvol"], F["vf_vrp"], F["vf_rv24"] = p, rf, dvol, dvol - rv[24], rv[24]
    # --- 돌파/되돌림 봉 피쳐 + 메트릭 ---
    bt5 = btc.set_index("timestamp")["close"].reindex(pd.DatetimeIndex(kl["timestamp"])).ffill().to_numpy(float)
    Fb, _, _ = BO.bar_features(kl["close"].to_numpy(float), kl["high"].to_numpy(float), kl["low"].to_numpy(float),
                               kl["volume"].to_numpy(float), kl["taker_buy_base"].to_numpy(float), bt5)
    for k, v in Fb.items():
        F[f"bo_{k}"] = v
    if met is not None:
        for k, v in BO.metric_features(met, met_ts, kl["timestamp"]).items():
            F[f"x_{k}"] = v                                  # 학습 빌더/아티팩트 이름 x_m_* 에 맞춘다
    else:
        for nm in BO.METRICS:
            F[f"x_m_{nm}_z"] = np.nan; F[f"x_m_{nm}_d12"] = np.nan; F[f"x_m_{nm}_d48"] = np.nan
    ts = pd.to_datetime(F["ts"])
    F["cal_hour"] = ts.dt.hour.to_numpy(); F["cal_weekday"] = ts.dt.weekday.to_numpy()
    return F


def feature_cols(F: pd.DataFrame) -> list[str]:
    return [c for c in F.columns if c not in ("ts", "atr")]


def position_features(side: np.ndarray, hold: np.ndarray, cap: int, u: np.ndarray, mfe: np.ndarray,
                      mae: np.ndarray, atr: np.ndarray) -> dict[str, np.ndarray]:
    """가격 변동은 원시 비율(레버·명목 곱하지 않음 -- 파리티 계약). ATR 은 ATR14 비율."""
    a = np.maximum(atr, 1e-9)
    return {"pos_side": side.astype(float), "pos_hold": hold.astype(float), "pos_left": (cap - hold).astype(float),
            "pos_u": u, "pos_mfe": mfe, "pos_mae": mae, "pos_u_atr": u / a, "pos_mfe_atr": mfe / a,
            "pos_mae_atr": mae / a, "pos_giveback_atr": (mfe - u) / a}


def load_artifacts() -> tuple[dict | None, dict | None, dict | None]:
    return X.load_artifact(), X.load_costw(), VF.load_artifact()
