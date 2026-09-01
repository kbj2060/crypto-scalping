#!/usr/bin/env python3
"""154피쳐 엔지니어링 세트 룩어헤드/오염 감사 -- V자반등 피쳐 확장에 쓰기 전 관문.

## 왜

2026-09-01 미포착 사건 정량분석: V자반등 사건의 38%가 9트리거 미포착인데, 라벨 품질은 포착
사건과 같으면서 **모델이 거의 못 잡는다**(포착률 27.9% vs 8.5%). 결과는 진짜 V자반등인데
**사전 피쳐(Tier0 23개)로는 안 보이는 사건**이다. 라벨 타이트화는 지렛대가 아님이 확인됐고
(조일수록 미포착 비중이 38%->58%로 오히려 증가), 8트리거 피쳐화는 구조적으로 불가능하다
(미포착 사건에서 트리거는 정의상 전부 0). 남은 축은 **피쳐**다.

대체 데이터원(청산/L2/OI/GEX)은 전부 2026년에 수집을 시작해 TRAIN(2024-01~2025-09)과 겹치지
않는다. 유일하게 전 구간을 덮는 확장 후보가 이 154피쳐 세트(2024-01-01~2026-06-30)다.

**그런데 그냥 쓸 수 없다**: 데이터 지도에 이 세트의 "오염 25컬럼 패치" 이력이 있고, Tier0
23피쳐와 달리 룩어헤드 감사를 통과한 적이 없다.

## 무엇을 검사하나

  A. 사용불가 컬럼: NaN 과다, 상수, 무한대
  B. **단일피쳐 방향 AUC** -- 피쳐 하나만으로 다음 봉 방향을 맞히는 정도.
     실제 피쳐는 0.50~0.53이 정상이다. **0.55를 넘으면 누출을 강하게 의심**한다
     (10만행 이상에서 단일 피쳐가 그 정도를 내는 건 정상적으로 불가능).
     같은 통계를 **과거 방향**(직전 봉)에 대해서도 재서 대조군으로 쓴다 -- 인과적 피쳐는
     과거와의 관계가 미래와의 관계보다 강하거나 비슷해야 한다.
  C. **교차상관 피크 시차**: 피쳐와 종가의 상관이 최대가 되는 시차. 인과적 피쳐는 피크가
     0 이하(과거)여야 한다. **피크가 양의 시차면 미래 정보로 계산됐을 가능성**이 있다.
  D. **다른 모델의 출력 컬럼**: `regime3_current_sensitive_wide24_*` 3종은 별도 모델 산출물이라
     ①인과적(walk-forward)으로 만들어졌는지 알 수 없고 ②V자반등 모델이 그걸 먹으면 순환성이
     생긴다. 이름으로 식별해 **무조건 제외 권고**.

⚠️**한계를 명시한다**: B/C는 **총체적(gross) 누출 탐지기**이지 인과성 증명이 아니다. 통과했다고
"인과적임이 증명됨"이 아니라 "명백한 누출은 안 보임"이다. 진짜 증명은 각 피쳐의 생성 코드를
읽는 것인데 154개 전부는 이 라운드 범위 밖이다. 그래서 **통과한 것 중에서도 실제로 쓸 것만
추려 생성 코드를 확인**하는 게 다음 단계다.

⚠️ HOLDOUT 구간(2026-04-01 이후)은 잘라내고 감사한다. 라이브 코드 변경 없음.

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/audit_eth_154feature_lookahead_20260901.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F154 = ROOT / "tmp/ilias_eth_154feature_dataset_20260821/ilias_eth_154feature_2024_2026H1_combined.csv"
ETH_CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OOS_END = pd.Timestamp("2026-04-01")     # HOLDOUT 보호

AUC_SUSPECT = 0.55        # 단일피쳐 방향 AUC 이 이상이면 누출 의심
AUC_HARD = 0.60           # 이 이상이면 사실상 확정
NAN_MAX = 0.30
MODEL_OUTPUT_PAT = re.compile(r"regime3_current|_bull_prob|_bear_prob|_confidence", re.I)
XCORR_LAGS = range(-12, 13)

OUT_JSON = ROOT / "data/research/eth_154feature_audit_20260901/report.json"


def log(msg: str) -> None:
    print(f"[audit154] {msg}", flush=True)


def rank_auc(x: np.ndarray, y: np.ndarray) -> float:
    """이진 라벨 y에 대해 점수 x의 AUC. 결측 제외, 순위 기반(정렬만, 학습 없음)."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 1000 or len(np.unique(y)) < 2:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    n1 = float((y == 1).sum()); n0 = float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    t0 = time.time()
    log(f"loading {F154.name} ...")
    df = pd.read_csv(F154)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.loc[df["timestamp"] < OOS_END].reset_index(drop=True)
    log(f"  {len(df):,}행 x {len(df.columns)}컬럼  ({df['timestamp'].min()} ~ {df['timestamp'].max()})")

    kl = pd.read_csv(ETH_CSV, usecols=["timestamp", "close"])
    kl["timestamp"] = pd.to_datetime(kl["timestamp"])
    if kl["timestamp"].dt.tz is not None:
        kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    m = df.merge(kl, on="timestamp", how="inner", suffixes=("", "_kl"))
    log(f"  klines 병합 후 {len(m):,}행 (매칭률 {len(m)/len(df)*100:.1f}%)")
    if len(m) < 0.9 * len(df):
        log("  ⚠️타임스탬프 매칭률이 낮다 -- 정렬 규약이 다를 수 있음")

    close = m["close"].to_numpy(dtype=float)
    fwd1 = np.full(len(close), np.nan); fwd1[:-1] = close[1:] / close[:-1] - 1
    bwd1 = np.full(len(close), np.nan); bwd1[1:] = close[1:] / close[:-1] - 1
    y_fwd = np.where(np.isfinite(fwd1), (fwd1 > 0).astype(float), np.nan)
    y_bwd = np.where(np.isfinite(bwd1), (bwd1 > 0).astype(float), np.nan)

    feats = [c for c in m.columns if c not in ("timestamp", "close")]
    log(f"  검사 대상 {len(feats)}개 피쳐")
    log("")

    rows = []
    for c in feats:
        x = pd.to_numeric(m[c], errors="coerce").to_numpy(dtype=float)
        nan_rate = float(np.mean(~np.isfinite(x)))
        rec = {"feature": c, "nan_rate": round(nan_rate, 4),
               "is_model_output": bool(MODEL_OUTPUT_PAT.search(c))}
        finite = x[np.isfinite(x)]
        rec["is_constant"] = bool(len(finite) == 0 or np.nanstd(finite) == 0)
        if rec["is_constant"] or nan_rate > NAN_MAX:
            rec["verdict"] = "unusable"
            rows.append(rec); continue

        a_f = rank_auc(x, y_fwd)
        a_b = rank_auc(x, y_bwd)
        rec["auc_next_bar"] = round(a_f, 4) if a_f == a_f else None
        rec["auc_prev_bar"] = round(a_b, 4) if a_b == a_b else None
        rec["fwd_minus_bwd"] = round(abs(a_f - 0.5) - abs(a_b - 0.5), 4) if (a_f == a_f and a_b == a_b) else None

        # 교차상관 피크 시차 (표본 상한을 둬 속도 확보)
        idx = np.isfinite(x)
        xs = pd.Series(np.where(idx, x, np.nan))
        best_lag, best_abs = None, -1.0
        for lag in XCORR_LAGS:
            cc = xs.corr(pd.Series(close).shift(-lag))
            if cc == cc and abs(cc) > best_abs:
                best_abs, best_lag = abs(cc), lag
        rec["xcorr_peak_lag"] = best_lag
        rec["xcorr_peak_abs"] = round(best_abs, 4) if best_abs >= 0 else None

        dev = abs(a_f - 0.5) if a_f == a_f else 0.0
        if rec["is_model_output"]:
            rec["verdict"] = "exclude_model_output"
        elif dev >= (AUC_HARD - 0.5):
            rec["verdict"] = "leak_likely"
        elif dev >= (AUC_SUSPECT - 0.5):
            rec["verdict"] = "suspect"
        elif best_lag is not None and best_lag > 0 and best_abs > 0.30:
            rec["verdict"] = "suspect_future_peak"
        else:
            rec["verdict"] = "pass"
        rows.append(rec)

    res = pd.DataFrame(rows)
    order = ["leak_likely", "suspect", "suspect_future_peak", "exclude_model_output", "unusable", "pass"]
    log("=== 판정 요약 ===")
    for v in order:
        sub = res.loc[res["verdict"] == v]
        if len(sub):
            log(f"  {v:22s} {len(sub):>3d}개")

    for v in ("leak_likely", "suspect", "suspect_future_peak", "exclude_model_output", "unusable"):
        sub = res.loc[res["verdict"] == v]
        if not len(sub):
            continue
        log("")
        log(f"--- {v} ({len(sub)}개) ---")
        for _, r in sub.sort_values("auc_next_bar", ascending=False, na_position="last").iterrows():
            log(f"  {r['feature']:<46s} AUC(다음봉) {r['auc_next_bar']}  "
                f"AUC(직전봉) {r['auc_prev_bar']}  피크시차 {r['xcorr_peak_lag']}  "
                f"NaN {r['nan_rate']}")

    ok = res.loc[res["verdict"] == "pass"]
    log("")
    log(f"=== 통과 {len(ok)}개 -- 단일피쳐 방향 AUC 상위 12 (참고: 0.50~0.53이 정상) ===")
    for _, r in ok.reindex(ok["auc_next_bar"].sub(0.5).abs().sort_values(ascending=False).index).head(12).iterrows():
        log(f"  {r['feature']:<46s} AUC(다음봉) {r['auc_next_bar']}  AUC(직전봉) {r['auc_prev_bar']}  "
            f"피크시차 {r['xcorr_peak_lag']}")

    report = {
        "signal": "eth_154feature_lookahead_audit", "asset": "ETHUSDT",
        "scope": {"source": str(F154.relative_to(ROOT)), "rows": int(len(m)),
                  "period": [str(m["timestamp"].min()), str(m["timestamp"].max())],
                  "holdout_touched": False, "live_code_changed": False,
                  "limitation": ("B(단일피쳐 방향 AUC)/C(교차상관 피크)는 **총체적 누출 탐지기**이지 "
                                 "인과성 증명이 아니다. 통과 = '명백한 누출 없음'이지 '인과적 증명됨'이 "
                                 "아니다. 실제로 채택할 피쳐는 생성 코드를 별도 확인해야 한다.")},
        "thresholds": {"auc_suspect": AUC_SUSPECT, "auc_hard": AUC_HARD, "nan_max": NAN_MAX},
        "counts": {v: int((res["verdict"] == v).sum()) for v in order},
        "features": rows, "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
