#!/usr/bin/env python3
"""ETH V자반등 **경제라벨** 자동매매 신호 -- 라이브 스코어러 (배선 전, 판단 산출 전용).

## 이게 무엇인가

2026-09-02에 VAL/OOS/HOLDOUT을 전부 통과한 후보의 라이브 서빙 코드다. 기존 배포 신호
(`live_eth_sweep_v_rebound_signal_20260829.py`, 대시보드 표시 전용)와 **별개 모델**이다 --
라벨 정의부터 다르다.

    기존 배포: 라벨 = "1.5×ATR 튀고 giveback<=0.20" (**모양**)  -> 매매 엣지 미입증
    이 후보  : 라벨 = "이 진입이 비용 후 이익인가" (**결과**)   -> HOLDOUT 통과

근거 전문: `docs/homer/v_rebound_open_issues_20260901.md` 20절.

## ⚠️ 자동 실행하지 않는다

이 모듈은 **판단만 산출**한다. `trading_bot.py`에 배선돼 있지 않고, 주문도 내지 않는다.
실제 자동매매 활성화는 사용자의 명시적 결정이 필요한 별도 단계다.

## 검증 요약 (전부 방향뒤집기 대조군 통과)

| 구간 | n | 기대값 | 누적 | 뒤집기 | 최대DD |
|---|---|---|---|---|---|
| VAL(선정) | 1,851 | +3.63bp | +6,714bp | — | −4,279bp |
| OOS(1회)  | 1,383 | +7.98bp | +11,031bp | +0.32bp | −5,032bp |
| **HOLDOUT(1회)** | **1,987** | **+6.09bp** | **+12,108bp** | **−4.15bp** | −8,949bp |

실행 강건성(OOS): 진입 1~2봉 지연 +8.93/+7.56bp · 비용 15bp +3.05bp(손익분기 ~18bp) ·
슬리피지 왕복 6bp +2.05bp · 동시보유 1/3/5 전부 양수.

⚠️**레짐 적응은 확인됐으나 완벽하지 않다**: HOLDOUT 롱 비중 64.8%(하락장이던 VAL/OOS는 3~11%)로
방향을 바꾸지만, 2026-06(시장 −21.95%)에 72.6% 롱으로 들어가 그 달 −3.25bp였다.
**월 단위로 크게 틀릴 수 있다.**

## 서빙 규격 (VAL에서 선정, OOS·HOLDOUT 각 1회 검증 -- 재조정 금지)

    임계값   p >= 0.8158  (5시드 앙상블 평균 확률)
    진입     다음 봉 시가 (지연 1~2봉까지 강건함이 실측됨)
    손절     entry -/+ 5.0 x ATR
    무장     +1.5 x ATR 도달 시 트레일 개시
    트레일   0.1 x ATR (한 방향으로만 조임)
    한도     동시보유 5  (⚠️한도 3이 OOS에서 낙폭 39% 낮고 건당 기대값은 더 높았으나,
             VAL이 고른 값은 5다. 사이징 재조정은 별도 결정 사항.)

Usage:
    from live_eth_v_rebound_econ_autotrade_signal_20260902 import compute_signal
    out = compute_signal()      # {"warmed_up", "error", "calls": [...], "bars_scored", ...}

CLI (스모크 테스트):
    python scripts/live_eth_v_rebound_econ_autotrade_signal_20260902.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ⚠️피쳐 계산은 **배포판 함수를 그대로 재사용**한다. 여기서 다시 구현하면 학습/추론 패리티가
# 조용히 깨진다(CLAUDE.md Position-Feature Train/Inference Parity Contract와 같은 취지).
_LIVE = importlib.import_module("live_eth_sweep_v_rebound_signal_20260829")
_fetch_klines = _LIVE._fetch_klines
_build_features = _LIVE._build_features
_every_bar_rows = _LIVE._every_bar_rows
_compute_signals = _LIVE.compute_signals
FEATURES = _LIVE.FEATURES               # Tier0 22 + rsi = 23
SYMBOL, BTC_SYMBOL = _LIVE.SYMBOL, "BTCUSDT"

CTX_CSV = ROOT / "data/labels/eth_5m_v_rebound_econ_label_20260902/tabpfn_train_context_frozen_econ_5seed_20260902.csv"

PROBA_THRESHOLD = 0.8158        # VAL 선정 (상위 5%)
BRACKET = {"sl_atr": 5.0, "arm_atr": 1.5, "trail_atr": 0.1}
MAX_CONCURRENT = 5
SCORE_TAIL_BARS = 3             # 최근 몇 봉을 채점할지(현재 봉 + 여유)

_MODELS: list[Any] | None = None


def _load_models() -> list[Any]:
    """동결 컨텍스트(시드별)로 5개 TabPFN을 적합해 캐시한다."""
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    from tabpfn import TabPFNClassifier
    df = pd.read_csv(CTX_CSV)
    models = []
    for sd, g in df.groupby("seed"):
        clf = TabPFNClassifier(device="cuda", random_state=int(sd),
                               ignore_pretraining_limits=True)
        clf.fit(g[FEATURES], g["label"].to_numpy())
        models.append(clf)
    _MODELS = models
    return models


def compute_signal() -> dict[str, Any]:
    """최근 봉들을 채점해 임계값을 넘은 진입 후보를 반환. 주문은 내지 않는다."""
    empty = {"warmed_up": False, "error": None, "calls": [], "bars_scored": 0,
             "threshold": PROBA_THRESHOLD, "bracket": BRACKET,
             "max_concurrent": MAX_CONCURRENT}
    try:
        if not CTX_CSV.exists():
            return {**empty, "error": "frozen_context_missing"}
        kl = _fetch_klines(SYMBOL)
        btc = _fetch_klines(BTC_SYMBOL)
        if kl is None or len(kl) < 900:
            return {**empty, "error": "klines_unavailable"}
        frame = _build_features(kl)
        sig = _compute_signals(kl, btc_df=btc, funding_df=None)
        cand = _every_bar_rows(frame, sig, SCORE_TAIL_BARS)
        cand = cand.dropna(subset=[c for c in FEATURES if c in cand.columns])

        # _every_bar_rows는 방향-상대 피쳐만 만든다 -- 나머지 Tier0는 frame에서 가져온다.
        need = [c for c in FEATURES if c not in cand.columns]
        for c in need:
            cand[c] = frame[c].to_numpy()[cand["pos"].to_numpy()]
        cand = cand.dropna(subset=FEATURES)
        if cand.empty:
            return {**empty, "error": "no_scoreable_bars"}

        models = _load_models()
        P = np.vstack([m.predict_proba(cand[FEATURES])[:, 1] for m in models])
        cand["proba"] = P.mean(axis=0)
        cand["proba_std"] = P.std(axis=0)

        atr = frame["atr"].to_numpy()
        close = frame["close"].to_numpy()
        calls = []
        for _, r in cand.loc[cand["proba"] >= PROBA_THRESHOLD].iterrows():
            i = int(r["pos"])
            a = float(atr[i])
            if not np.isfinite(a) or a <= 0:
                continue
            side = "long" if int(r["is_downside"]) == 1 else "short"
            sgn = 1.0 if side == "long" else -1.0
            ref = float(close[i])       # 참고가(실제 진입은 다음 봉 시가)
            calls.append({
                "timestamp_utc": str(r["timestamp"]), "side": side,
                "proba": round(float(r["proba"]), 4),
                "proba_seed_std": round(float(r["proba_std"]), 4),
                "atr": a, "ref_close": ref,
                "stop_loss_price": ref - sgn * BRACKET["sl_atr"] * a,
                "arm_at_price": ref + sgn * BRACKET["arm_atr"] * a,
                "trail_atr_distance": BRACKET["trail_atr"] * a,
                "triggers_display_only": r.get("triggers", ""),
            })
        return {**empty, "warmed_up": True, "calls": calls, "bars_scored": int(len(cand)),
                "latest_bar_utc": str(cand["timestamp"].max()),
                "n_models": len(models)}
    except Exception as e:                       # noqa: BLE001
        return {**empty, "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    t0 = time.time()
    out = compute_signal()
    out["cycle_sec"] = round(time.time() - t0, 2)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
