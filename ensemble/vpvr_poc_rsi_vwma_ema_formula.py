"""
VPVR/POC + RSI + VWMA100 + EMA 룰기반 매매 공식
================================================================================
5개 피처만으로 구성된 독립 룰기반 진입/청산 공식.

  1. EMA(20/50/200)            — 추세 방향 + 매크로 필터
  2. VWMA(100)                 — 거래량가중 이동평균 대비 이격(동적 지지/저항)
  3. VPVR/POC (cvp_* 재사용)   — POC 대비 거리 + 거래량 불균형 기반 평균회귀
  4. RSI(14, 기존 'rsi' 컬럼)  — 모멘텀 확인 + 과열/과매도 게이트

M7/DSAC 앙상블과 독립적으로 동작하며, 기존 `core/cvp.py`(VPVR/POC)와
`features/engineering.py`(RSI)가 생성한 컬럼을 그대로 재사용한다.
EMA/VWMA는 이 모듈에서 원시 OHLCV로부터 새로 계산한다.

사용:
    from ensemble.vpvr_poc_rsi_vwma_ema_formula import FormulaConfig, FormulaEngine
    df = FormulaEngine(FormulaConfig()).compute(df)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# FormulaEngine.compute()가 요구하는 필수 입력 컬럼.
# rsi / cvp_poc_dist / cvp_volume_imbalance / cvp_regime 은 기존 파이프라인
# (features/engineering.py, core/cvp.py) 산출물을 그대로 재사용한다.
REQUIRED_INPUT_COLS = [
    "open", "high", "low", "close", "volume",
    "rsi", "cvp_poc_dist", "cvp_volume_imbalance", "cvp_regime",
]

# 이 공식이 새로 계산해서 df에 추가하는 컬럼들.
FORMULA_OUTPUT_COLS = [
    "ema_20", "ema_50", "ema_200", "vwma_100",
    "ema_comp", "vwma_comp", "poc_comp", "rsi_comp",
    "composite_score", "macro_filter",
    "rsi_long_ok", "rsi_short_ok", "position_signal",
]


@dataclass
class FormulaConfig:
    # ── 이동평균 기간 (요청대로 고정값이 기본이나 실험을 위해 설정 가능) ──
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    vwma_window: int = 100

    # ── 컴포넌트 가중치 ──
    w_ema: float = 1.0
    w_vwma: float = 1.0
    w_poc: float = 1.0
    w_rsi: float = 0.6

    # ── 컴포넌트 민감도(스케일) ──
    k_ema: float = 8.0
    k_vwma: float = 8.0
    k_poc1: float = 2.0   # POC 거리 가중치
    k_poc2: float = 2.0   # 거래량 불균형 가중치
    w_regime_gate: float = 0.4  # cvp_regime(추세강도)로 POC 평균회귀 신호를 얼마나 감쇠시킬지

    # ── 진입/청산 임계값 ──
    theta_entry: float = 0.35
    theta_exit: float = 0.12
    theta_full: float = 0.75  # 이 값 이상이면 최대 사이즈

    # ── RSI 과열/과매도 게이트 ──
    rsi_hot: float = 75.0
    rsi_cold: float = 25.0

    # ── 포지션 사이징 ──
    base_size: float = 1.0

    # ── 리스크 관리(TP/SL/트레일링/최대 보유/쿨다운) ──
    tp_pct: float = 0.012
    sl_pct: float = 0.006
    trail_pct: float = 0.0   # 0이면 트레일링 비활성
    max_hold_bars: int = 96  # 5분봉 기준 8시간, 0이면 비활성
    cooldown_bars: int = 2

    def __post_init__(self) -> None:
        if self.theta_exit >= self.theta_entry:
            self.theta_exit = 0.5 * self.theta_entry


class FormulaEngine:
    """VPVR/POC + RSI + VWMA100 + EMA composite score 계산기."""

    def __init__(self, config: FormulaConfig | None = None):
        self.cfg = config or FormulaConfig()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
        if missing:
            raise KeyError(
                f"FormulaEngine.compute requires columns {missing} "
                f"(rsi/cvp_* must come from features/engineering.py + core/cvp.py)"
            )

        cfg = self.cfg
        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce").clip(lower=0)

        # ── 1. EMA(20/50/200) ──
        ema_fast = close.ewm(span=cfg.ema_fast, adjust=False).mean()
        ema_mid = close.ewm(span=cfg.ema_mid, adjust=False).mean()
        ema_slow = close.ewm(span=cfg.ema_slow, adjust=False).mean()
        df["ema_20"] = ema_fast
        df["ema_50"] = ema_mid
        df["ema_200"] = ema_slow

        ema_comp = np.tanh(cfg.k_ema * (ema_fast - ema_mid) / ema_mid.replace(0, np.nan))
        df["ema_comp"] = ema_comp.fillna(0.0)
        # 매크로 필터: ema_50 vs ema_200 (+1 상승추세 / -1 하락추세 / 0 동률)
        df["macro_filter"] = np.sign(ema_mid - ema_slow).fillna(0.0)

        # ── 2. VWMA(100) ──
        pv_sum = (close * volume).rolling(cfg.vwma_window, min_periods=1).sum()
        v_sum = volume.rolling(cfg.vwma_window, min_periods=1).sum().replace(0, np.nan)
        vwma = (pv_sum / v_sum).ffill()
        df["vwma_100"] = vwma

        vwma_dist = (close - vwma) / vwma.replace(0, np.nan)
        df["vwma_comp"] = np.tanh(cfg.k_vwma * vwma_dist).fillna(0.0)

        # ── 3. VPVR/POC (기존 cvp_* 재사용) ──
        poc_dist = pd.to_numeric(df["cvp_poc_dist"], errors="coerce").fillna(0.0)
        vol_imbalance = pd.to_numeric(df["cvp_volume_imbalance"], errors="coerce").fillna(0.0)
        regime = pd.to_numeric(df["cvp_regime"], errors="coerce").fillna(0.0).clip(-1, 1)

        poc_raw = cfg.k_poc1 * (-poc_dist) + cfg.k_poc2 * vol_imbalance
        regime_scale = (1.0 - cfg.w_regime_gate * regime.abs()).clip(lower=0.0)
        df["poc_comp"] = (np.tanh(poc_raw) * regime_scale).fillna(0.0)

        # ── 4. RSI(14, 기존 'rsi' 컬럼) ──
        rsi = pd.to_numeric(df["rsi"], errors="coerce").fillna(50.0)
        df["rsi_comp"] = (rsi - 50.0) / 50.0
        df["rsi_long_ok"] = rsi < cfg.rsi_hot
        df["rsi_short_ok"] = rsi > cfg.rsi_cold

        # ── 합성 스코어 ──
        composite = (
            cfg.w_ema * df["ema_comp"]
            + cfg.w_vwma * df["vwma_comp"]
            + cfg.w_poc * df["poc_comp"]
            + cfg.w_rsi * df["rsi_comp"]
        )
        df["composite_score"] = composite.clip(-1.0, 1.0)

        # 참고용 이산 신호 (실제 진입/청산 상태머신은 백테스트 루프에서 처리 —
        # 히스테리시스/최소보유/TP·SL은 시퀀셜 로직이 필요하기 때문)
        long_ready = (
            (df["composite_score"] > cfg.theta_entry)
            & (df["macro_filter"] >= 0)
            & df["rsi_long_ok"]
        )
        short_ready = (
            (df["composite_score"] < -cfg.theta_entry)
            & (df["macro_filter"] <= 0)
            & df["rsi_short_ok"]
        )
        df["position_signal"] = np.where(long_ready, 1, np.where(short_ready, -1, 0))

        return df
