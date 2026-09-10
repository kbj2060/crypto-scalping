#!/usr/bin/env python3
"""**모든 이벤트 트리거**를 중기 타임프레임으로 옮겨 같은 처방으로 비교 (2026-09-10).

사용자: *"증거신호 및 이벤트 트리거 모두 진행해줘."*
선행: 극점 탐지기 하나만 계열·용량 튜닝 + 확장 데이터 + TabPFN 을 받았다
(`research_eth_midterm_model_capacity_20260910.py`). 나머지 트리거에 같은 처방을 준다.

## 설계 — 피쳐 기반을 고정하고 라벨만 바꾼다
각 트리거의 **배포 파이프라인을 그대로 이식하지 않는다**. V자반등은 Tier0 자체 피쳐 + 동결 TabPFN 컨텍스트,
레짐은 FeatureEngineer 136피쳐를 쓰는데, 네 개를 각자 이식하면 타임프레임마다 정의가 갈라져 비교가 불가능해진다.
대신 **봉 수 기반이라 타임프레임 이식이 안전한 공통 피쳐 기반**(`live_eth_extreme_detector_20260909.FEATS`,
39개: 증거신호 맥락 10 + ATR분위 + ret/pos/dist 12 + BTC 3 + 시각 2 + 발동 플래그 8 + 개수 + 측면)을 쓰고,
**라벨(= 그 트리거가 던지는 질문)만** 바꾼다. 따라서 이 표는 "배포판 성능 재현"이 아니라
**"이 질문이 중기에서도 답할 수 있는가"** 의 타임프레임 간 비교다.

## 트리거 5종 (라벨은 전부 봉 수 기반 -- 그래서 이식된다)
  extreme      i+1..i+12 에서 봉 i 의 저점/고점을 깨지 않는가            (배포 정의 그대로, 발동 봉만)
  breakout_rev 앵커 이후 12봉 내 ±0.8×ATR 중 발현 방향에 먼저 닿는가     (앵커 = 직전 3봉 무발동 첫 발동)
  v_rebound    6봉 내 1.5×ATR 급반전 AND 12봉 창 끝 종가 반납 ≤20%       (배포 라벨 구조 재현, 매 봉 양측)
  regime_trend 12봉 뒤에 추세 상태인가(er 임계 = TRAIN 70분위, 3봉 확인) (매 봉, 측면 없음)
  vol_expand   다음 H봉 실현변동성 ≥ 1.3 × 직전 H봉                      (매 봉, H=12, DVOL·HAR 추가 피쳐)

각 트리거 × {5m,15m,1h,4h} × {현행 2.7년, 확장 4.8년} × {HGB격자·로지스틱·포레스트} + TabPFN(별도 스크립트).
분할·선택 규약은 선행 실험과 동일: TRAIN ≤2025-05-31 / 내부검증 ~08-31 / TEST 2025-09-01~.
출력 tmp/eth_all_triggers_midterm_20260910/report.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_exit_synth_dataset_20260910 as BD  # noqa: E402
import research_eth_midterm_model_capacity_20260910 as C  # noqa: E402
import research_eth_signals_midterm_timeframe_20260910 as M  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
import live_eth_extreme_detector_20260909 as X  # noqa: E402

OUT = ROOT / "tmp/eth_all_triggers_midterm_20260910"
W = 12                      # 라벨 창(봉) -- 모든 트리거 공통 축
MAX_ROWS = 120_000          # 매 봉 트리거의 5분봉 폭주 방지(무작위 부분표집, 그 사실을 기록)
ANN = np.sqrt(288 * 365) * 100


def log(m):
    print(f"[all {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- 공통 피쳐 기반
def base_frame(sig: pd.DataFrame, btc: pd.DataFrame):
    """봉 단위 공통 피쳐(X.FEATS 순서) — 측면별로 발동 플래그만 갈아끼운다."""
    S, tq = X._feature_frame(sig, btc)
    fires = {s: {"bottom": sig[f"bottom_{s}"].fillna(False).to_numpy(bool),
                 "top": sig[f"top_{s}"].fillna(False).to_numpy(bool)} for s in X.B.SIGNALS}
    return S, tq, fires


def side_matrix(S: pd.DataFrame, fires: dict, side: str) -> np.ndarray:
    D = S.copy()
    cnt = np.zeros(len(S))
    for s in X.B.SIGNALS:
        f = fires[s][side].astype(float)
        D[f"f_{s}"] = f
        cnt += f
    D["n_signals"] = cnt
    D["is_bottom"] = 1.0 if side == "bottom" else 0.0
    return D[X.FEATS].to_numpy(np.float32)


# ---------------------------------------------------------------- 트리거별 라벨
def lab_extreme(sig, fires, side):
    hi = sig.high.to_numpy(float); lo = sig.low.to_numpy(float); n = len(sig)
    anyf = np.zeros(n, bool)
    for s in X.B.SIGNALS:
        anyf |= fires[s][side]
    idx = np.flatnonzero(anyf); idx = idx[(idx >= 900) & (idx + W < n)]
    if side == "bottom":
        y = np.array([lo[i + 1:i + 1 + W].min() >= lo[i] for i in idx], int)
    else:
        y = np.array([hi[i + 1:i + 1 + W].max() <= hi[i] for i in idx], int)
    return idx, y


def lab_breakout(sig, fires, side):
    """앵커(직전 3봉 무발동 첫 발동) 이후 12봉 내 ±0.8ATR 선착. 발현 방향으로 닿으면 돌파=1."""
    c = sig.close.to_numpy(float); hi = sig.high.to_numpy(float); lo = sig.low.to_numpy(float)
    atr = sig.atr_pct.to_numpy(float); n = len(sig)
    anyf = np.zeros(n, bool)
    for s in X.B.SIGNALS:
        anyf |= fires[s][side]
    prev3 = pd.Series(anyf).rolling(3, min_periods=1).max().shift(1).fillna(0).to_numpy(bool)
    idx = np.flatnonzero(anyf & ~prev3); idx = idx[(idx >= 900) & (idx + W < n)]
    dir_up = (side == "top")            # 천장 발동 = 상승 발현
    keep, y = [], []
    for i in idx:
        b = 0.8 * max(atr[i], 1e-6)
        up = (hi[i + 1:i + 1 + W] >= c[i] * (1 + b)); dn = (lo[i + 1:i + 1 + W] <= c[i] * (1 - b))
        t_up = int(np.argmax(up)) if up.any() else 99
        t_dn = int(np.argmax(dn)) if dn.any() else 99
        if t_up == 99 and t_dn == 99:
            continue
        keep.append(i); y.append(int((t_up < t_dn) == dir_up))
    return np.array(keep, int), np.array(y, int)


def lab_v_rebound(sig, fires, side):
    """배포 라벨의 **구조 재현**: 6봉(30분) 내 최대 유리이동 ≥1.5×ATR AND 12봉(60분) 창 끝 종가 기준 되돌림 ≤20%.

    ⚠️바이트 단위 재현이 아니다. 배포 라벨 파일(`eth_5m_v_rebound_multitrigger_labels.csv`)과 대조한 결과
      fast_move 상관 0.85(장중 고저)~0.90(종가), 5분봉 양성률 **0.159 vs 배포 ~0.14** 로 맞췄다.
      되돌림은 **창 끝 종가** 기준이라야 분포가 맞는다 — 장중 저점으로 재면 양성률이 0.002 로 붕괴한다
      (정점 이후 장중 되돌림은 거의 항상 20%를 넘는다). 모든 타임프레임에 같은 공식을 쓰므로 TF 간 비교는 유효하다.
    """
    c = sig.close.to_numpy(float); hi = sig.high.to_numpy(float); lo = sig.low.to_numpy(float)
    atr = (sig.atr_pct.to_numpy(float) * c)
    n = len(sig); sgn = 1.0 if side == "bottom" else -1.0
    fav = hi if side == "bottom" else lo
    idx = np.arange(900, n - W - 1)
    y = np.zeros(len(idx), int)
    for k, i in enumerate(idx):
        seg = sgn * (fav[i + 1:i + 7] - c[i])
        mfe = seg.max()
        if mfe < 1.5 * max(atr[i], 1e-9):
            continue
        peak = fav[i + 1 + int(np.argmax(seg))]
        move = sgn * (peak - c[i])
        give = sgn * (peak - c[i + W])                       # 창 끝 종가까지의 반납
        y[k] = int(move > 0 and give <= 0.20 * move)
    return idx, y


def lab_regime(sig, fires, side, t1=None, t2=None):
    """S12_K3 계열 라벨을 12봉 **앞으로** 예측: er/net/slope + 3봉 확인. 임계는 TRAIN 70분위."""
    c = pd.Series(sig.close.to_numpy(float))
    d = c.diff().abs()
    er12 = (c - c.shift(12)).abs() / d.rolling(12).sum().replace(0, np.nan)
    er24 = (c - c.shift(24)).abs() / d.rolling(24).sum().replace(0, np.nan)
    trend = ((er12 >= t1) | (er24 >= t2)).astype(float)
    conf = trend.rolling(3).min().fillna(0).to_numpy()            # K=3 연속 확인
    n = len(sig); idx = np.arange(900, n - W)
    return idx, conf[idx + W].astype(int), (er12, er24)


def lab_vol(sig, fires, side):
    r = np.log(sig.close.to_numpy(float))
    lr = pd.Series(np.diff(r, prepend=r[0]))
    back = lr.rolling(W).std().to_numpy()
    fwd = lr.shift(-W).rolling(W).std().shift(-(W - 1)).to_numpy()
    n = len(sig); idx = np.arange(900, n - 2 * W)
    ok = np.isfinite(back[idx]) & np.isfinite(fwd[idx]) & (back[idx] > 0)
    idx = idx[ok]
    return idx, (fwd[idx] / back[idx] >= 1.3).astype(int)


TRIGGERS = {"extreme": (lab_extreme, True), "breakout_rev": (lab_breakout, True),
            "v_rebound": (lab_v_rebound, True), "regime_trend": (lab_regime, False),
            "vol_expand": (lab_vol, False)}


def vol_extra(sig: pd.DataFrame, dv: pd.DataFrame | None, mult: int) -> np.ndarray:
    """vol_expand 전용 추가 피쳐: HAR-RV 3종 + DVOL + VRP(배포 신호의 정의 입력)."""
    c = sig.close.to_numpy(float); n = len(c)
    lr = pd.Series(np.diff(np.log(c), prepend=np.log(c[0])))
    bars_per_h = max(1, 12 // mult)
    cols = [np.log(np.clip(lr.rolling(max(2, bars_per_h * h)).std().to_numpy() * ANN, 1e-6, None))
            for h in (1, 24, 168)]
    if dv is not None and len(dv):
        d = dv.sort_values("timestamp").copy(); d["avail"] = d["timestamp"] + pd.Timedelta(hours=1)
        j = pd.merge_asof(pd.DataFrame({"t": pd.to_datetime(sig["timestamp"])}), d[["avail", "dvol"]],
                          left_on="t", right_on="avail", direction="backward")
        dvol = j["dvol"].to_numpy(float)
    else:
        dvol = np.full(n, np.nan)
    rv24 = np.exp(cols[1])
    return np.column_stack(cols + [np.log(np.clip(dvol, 1e-6, None)), dvol - rv24]).astype(np.float32)


def build_arm(eth, btc, dv, mult: int, trig: str):
    """(X, y, ts, cols) — 트리거의 라벨 + 공통 피쳐(+vol 전용 추가)."""
    e, b = M.resample(eth, mult), M.resample(btc, mult)
    sig = compute_signals(e, btc_df=b, funding_df=None)
    S, tq, fires = base_frame(sig, b)
    fn, two_sided = TRIGGERS[trig]
    extra = vol_extra(sig, dv, mult) if trig == "vol_expand" else None
    kw = {}
    if trig == "regime_trend":                       # 임계는 TRAIN(2025-05-31 이전)에서만 잡는다
        ts_all = pd.to_datetime(sig["timestamp"])
        _, _, (er12, er24) = lab_regime(sig, fires, "bottom", t1=1.0, t2=1.0)
        m = (ts_all <= C.TRAIN_END).to_numpy()
        kw = {"t1": float(np.nanquantile(er12[m], 0.70)), "t2": float(np.nanquantile(er24[m], 0.70))}
    Xs, ys, tss = [], [], []
    for side in (("bottom", "top") if two_sided else ("bottom",)):
        out = fn(sig, fires, side, **kw) if kw else fn(sig, fires, side)
        idx, y = out[0], out[1]
        if len(idx) == 0:
            continue
        Xm = side_matrix(S, fires, side)[idx]
        if extra is not None:
            Xm = np.hstack([Xm, extra[idx]])
        Xs.append(Xm); ys.append(y); tss.append(pd.to_datetime(sig["timestamp"]).to_numpy()[idx])
    if not Xs:
        return None
    Xa = np.vstack(Xs); ya = np.concatenate(ys); tsa = np.concatenate(tss)
    ok = np.isfinite(Xa).all(axis=1)
    Xa, ya, tsa = Xa[ok], ya[ok], tsa[ok]
    o = np.argsort(tsa)
    cols = X.FEATS + (["l_rv1", "l_rv24", "l_rv168", "l_dvol", "vrp"] if extra is not None else [])
    return Xa[o], ya[o], pd.to_datetime(tsa[o]), cols


def evaluate(Xa, ya, ts, cols, tag) -> dict:
    A = pd.DataFrame(Xa, columns=cols)
    A["_ts"] = ts; A["_y"] = ya
    if len(A) > MAX_ROWS:                                    # 5분봉 매 봉 트리거 폭주 방지
        A = A.iloc[np.sort(np.random.default_rng(7).choice(len(A), MAX_ROWS, replace=False))].reset_index(drop=True)
    saved = X.FEATS
    try:
        X.FEATS = cols                                       # evaluate_tf 가 X.FEATS 를 본다
        return {**C.evaluate_tf(A, tag), "subsampled": bool(len(ya) > MAX_ROWS), "n_rows_total": int(len(ya))}
    finally:
        X.FEATS = saved


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    eth_x, btc_x = C.load_klines_extended()
    eth_c, btc_c = BD.load_klines("eth", "ETHUSDT"), BD.load_klines("btc", "BTCUSDT")
    dv = BD.load_dvol()
    rep = {"w_bars": W, "max_rows": MAX_ROWS, "train_end": C.TRAIN_END, "val_end": C.VAL_END,
           "feature_basis": "live_eth_extreme_detector_20260909.FEATS (+HAR/DVOL for vol_expand)",
           "note": "배포 파이프라인 재현이 아니라 «이 질문이 중기에서도 답되는가»의 TF 간 비교", "arms": {}}
    for trig in TRIGGERS:
        for tf, mult in C.TFS.items():
            for mode, (e, b) in (("current", (eth_c, btc_c)), ("extended", (eth_x, btc_x))):
                key = f"{trig}|{tf}|{mode}"
                try:
                    built = build_arm(e, b, dv, mult, trig)
                    if built is None:
                        rep["arms"][key] = {"error": "no_rows"}; continue
                    rep["arms"][key] = evaluate(*built, key)
                except Exception as ex:  # noqa: BLE001
                    rep["arms"][key] = {"error": f"{type(ex).__name__}: {ex}"}
                    log(f"⚠️{key}: {type(ex).__name__}: {ex}")
                (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log("=" * 110)
    for k, v in rep["arms"].items():
        if "families" in v:
            fams = " · ".join(f"{f} {d['test_auc']:.3f}" for f, d in v["families"].items())
            log(f"{k:30s} n_tr={v['n_train']:7,d} 기저={v['base_rate_test']:.3f} 최고={max(d['test_auc'] for d in v['families'].values()):.3f} | {fams}")
        else:
            log(f"{k:30s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
