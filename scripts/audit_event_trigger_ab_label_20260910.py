#!/usr/bin/env python3
"""**이벤트 트리거 A/B 라벨 감사** -- 라벨이 체결 불가 지점을 기준으로 채점하는가 (2026-09-10).

2026-09-10 앵커 돌파/되돌림이 이 검사로 철회됐다(AUC 0.78 -> 0.58 -> 0.50).
그 축은 "라벨 탐색이 트리거 봉 **안**에서 시작한다"였는데, 나머지 트리거는 형태가 다르다:
탐색창은 `i+1` 부터라 봉을 공유하지 않지만 **기준가가 트리거 봉의 저가/고가**다.
저가·고가는 장중 극점이라 **그 가격에 들어갈 수 없다** -- 실제 진입은 그 봉 종가다.

  A라벨(현행) : 기준가 = low[i] (바닥쪽) / high[i] (천장쪽)   <- 체결 불가
  B라벨(체결가능): 기준가 = close[i]                          <- 결정 시점에 실제 살 수 있는 값

같은 행·같은 피쳐·같은 모델로 A/A · A/B · B/B 세 조합의 AUC 를 낸다.
B/B 가 0.5 로 내려앉으면 그 트리거의 측정된 실력은 기준가 특혜였다는 뜻이다.

⚠️이 스크립트는 **배포 성능 재현이 아니다**. 배포 모델(V자=TabPFN·Tier0, 극점=TabPFN)과
  피쳐·모델이 다르다. 여기서 고정한 것은 행·피쳐·모델이고 **바뀌는 것은 라벨 기준가뿐**이라
  A/B 격차만 읽는다. 절대 AUC 를 배포 수치와 비교하지 않는다.

대상: 극점 탐지기 · V자반등. (변동성 확장·레짐은 진입가 개념이 없어 이 축이 없다 -- README 참조)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
# 워크트리에서 돌 수 있게 -- 데이터(binance_data/tmp)는 메인 체크아웃에만 있다.
DATA = ROOT if (ROOT / "binance_data").exists() else Path(subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
    capture_output=True, text=True).stdout.strip()).parent

FRAME = DATA / "tmp/eth_signal_map_20260909/extreme_frame.parquet"
KL5 = DATA / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
W_EXTREME = 12                                   # build_eth_extreme_detector_week_20260909.py
FAST, FULL, ATR_MULT, T_SUSTAIN = 6, 12, 1.5, 0.20   # realized_outcome (V자반등 라벨 원본)
ATR_N = 14                                       # add_causal_columns
SEEDS = [20260910, 7, 131, 977, 20250401]
WINDOWS = {"VAL 25-09~12": ("2025-09-01", "2026-01-01"),
           "OOS 26-01~03": ("2026-01-01", "2026-04-01"),
           "최근 26-04~": ("2026-04-01", "2100-01-01")}
TRAIN_END = "2025-09-01"


def load() -> tuple[pd.DataFrame, list[str], np.ndarray, dict[str, np.ndarray]]:
    A = pd.read_parquet(FRAME)
    feats = [c for c in A.columns if not c.startswith("_")]
    kl = (pd.read_csv(KL5, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
          .sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    pos = pd.Series(np.arange(len(kl)), index=kl["timestamp"])
    A = A[A["_ts"].isin(pos.index)].reset_index(drop=True)
    idx = pos.loc[A["_ts"]].to_numpy()
    tr = pd.concat([kl.high - kl.low, (kl.high - kl.close.shift(1)).abs(),
                    (kl.low - kl.close.shift(1)).abs()], axis=1).max(axis=1)
    bars = {"hi": kl.high.to_numpy(float), "lo": kl.low.to_numpy(float),
            "cl": kl.close.to_numpy(float),
            "atr": tr.rolling(ATR_N, min_periods=ATR_N).mean().to_numpy(float)}
    return A, feats, idx, bars


def lab_extreme(idx, bars, bottom, ref):
    """«이 봉이 국소 극점인가» -- 이후 W봉이 기준가를 깨지 않았는가. A: 저가/고가, B: 종가."""
    hi, lo, n = bars["hi"], bars["lo"], len(bars["cl"])
    y = np.full(len(idx), -1)
    ok = (idx + W_EXTREME < n) & (idx > 0)
    for k in np.flatnonzero(ok):
        i = idx[k]
        if bottom[k]:
            y[k] = int(lo[i + 1:i + 1 + W_EXTREME].min() >= ref[k])
        else:
            y[k] = int(hi[i + 1:i + 1 + W_EXTREME].max() <= ref[k])
    return y


def lab_v_rebound(idx, bars, bottom, ref):
    """realized_outcome 그대로 -- 기준가만 인자로 뺐다(A=장중극점 / B=종가)."""
    hi, lo, cl, atr, n = bars["hi"], bars["lo"], bars["cl"], bars["atr"], len(bars["cl"])
    y = np.full(len(idx), -1)
    ok = (idx + FULL < n) & (idx > 0)
    for k in np.flatnonzero(ok):
        i = idx[k]; a = atr[i - 1]
        if not np.isfinite(a) or a <= 0:
            continue
        fast, full = slice(i + 1, i + FAST + 1), slice(i + 1, i + FULL + 1)
        if bottom[k]:
            fast_move = cl[fast].max() - ref[k]; peak = hi[full].max()
            denom = peak - ref[k]; give = (peak - cl[i + FULL]) / denom if abs(denom) > 1e-12 else np.nan
        else:
            fast_move = ref[k] - cl[fast].min(); peak = lo[full].min()
            denom = ref[k] - peak; give = (cl[i + FULL] - peak) / denom if abs(denom) > 1e-12 else np.nan
        y[k] = int(fast_move / a >= ATR_MULT and np.isfinite(give) and give <= T_SUSTAIN)
    return y


def fit_eval(X, y_fit, y_eval, ts, tag):
    """y_fit 으로 학습하고 y_eval 로 채점한다. A학습/B평가(전이) 조합을 재기 위해 분리."""
    tr = (ts < TRAIN_END).to_numpy() & (y_fit >= 0)
    p = np.mean([HistGradientBoostingClassifier(random_state=s, max_iter=300)
                 .fit(X[tr], y_fit[tr]).predict_proba(X)[:, 1] for s in SEEDS], axis=0)
    out = []
    for name, (a, b) in WINDOWS.items():
        m = ((ts >= a) & (ts < b)).to_numpy() & (y_eval >= 0)
        yy = y_eval[m]
        auc = roc_auc_score(yy, p[m]) if len(set(yy)) > 1 else float("nan")
        out.append((tag, name, int(m.sum()), int(yy.sum()), float(yy.mean()), float(auc)))
    return out


def main() -> int:
    A, feats, idx, bars = load()
    X = np.nan_to_num(A[feats].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    ts = A["_ts"]
    bottom = A["is_bottom"].to_numpy() > 0.5
    refA = np.where(bottom, bars["lo"][idx], bars["hi"][idx])   # 장중 극점 -- 체결 불가
    refB = bars["cl"][idx]                                      # 종가 -- 실제 진입가

    yA = lab_extreme(idx, bars, bottom, refA)
    same = (yA >= 0) & (A["_y"].to_numpy() >= 0)
    par = float((yA[same] == A["_y"].to_numpy()[same]).mean())
    print(f"극점 A라벨 재현 파리티: {par:.6f}  (n={same.sum():,})")
    if par < 0.999:
        print("🔴 배포 라벨을 재현하지 못했다 -- 아래 숫자는 무효다.", file=sys.stderr)
        return 1

    # ⭐"한 틱도 안 내려간다"는 B라벨은 양성률 1.2% 로 퇴화한다 -- 실제 손절 여유를 준 팔을 같이 낸다.
    #   bottom 은 cl - t*ATR, top 은 cl + t*ATR 을 기준가로 쓰면 lab_extreme 이 그대로 재사용된다.
    atr_at = bars["atr"][idx - 1]
    arms = {"극점": [("B 종가", refB)] + [(f"B 종가-{t}ATR", refB + np.where(bottom, -t, t) * atr_at)
                                        for t in (0.5, 1.0)],
            "V자반등": [("B 종가", refB)]}

    rows = []
    for trig, fn in (("극점", lab_extreme), ("V자반등", lab_v_rebound)):
        ya = fn(idx, bars, bottom, refA)
        rows += [(trig, *r) for r in fit_eval(X, ya, ya, ts, "A학습·A평가")]
        for bname, bref in arms[trig]:
            yb = fn(idx, bars, bottom, bref)
            ok = (ya >= 0) & (yb >= 0)
            print(f"[{trig}] {bname}: 유효 {ok.sum():,}행 · A양성률 {ya[ok].mean():.3f} · "
                  f"B양성률 {yb[ok].mean():.3f} · 라벨 일치율 {(ya[ok] == yb[ok]).mean():.3f}")
            rows += [(trig, *r) for r in fit_eval(X, ya, yb, ts, f"A학습·{bname}평가")]
            rows += [(trig, *r) for r in fit_eval(X, yb, yb, ts, f"{bname}학습·평가")]

    print(f"\n{'트리거':<8}{'조합':<20}{'창':<14}{'n':>7}{'양성':>6}{'기저':>8}{'AUC':>8}")
    print("-" * 72)
    for trig, tag, win, n, npos, base, auc in rows:
        print(f"{trig:<8}{tag:<20}{win:<14}{n:>7,}{npos:>6,}{base:>8.3f}{auc:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
