"""
XGBTrendBrain — LightGBM 3-class 추세 분류기 (Brain B)
================================================================================
Brain B를 Transformer에서 트리 기반으로 교체:
  - 타겟 : Triple-Barrier 레이블 (DOWN=0 / FLAT=1 / UP=2)
  - 피처 : 학습 시 선택된 피처 컬럼 (ULTIMATE_FEATURE_COLS 기반 auto_select)
  - 추론 : 0.5ms (CPU, GPU 불필요)
  - 저장 : data/trend_xgb/trend_xgb.pkl

인터페이스:
    XGBTrendBrain.load(path)        → XGBTrendBrain 인스턴스
    brain.predict_from_df(df)       → TrendSignal
    (TFTSignalModel, TrendContextBrain 과 동일 인터페이스)
"""

import os
import sys
import logging
import pickle
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 경로 설정 ──────────────────────────────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_ENSEMBLE_DIR = os.path.dirname(_THIS_DIR)
_ROOT_DIR    = os.path.dirname(_ENSEMBLE_DIR)
for _p in [_ROOT_DIR, _ENSEMBLE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ────────────────────────────────────────────────────────────────
# 공유 데이터클래스 / Triple-Barrier 유틸
# ────────────────────────────────────────────────────────────────
@dataclass
class TrendSignal:
    trend_dir : int
    strength  : float
    rev_prob  : float
    probs     : Tuple[float, float, float]

    @property
    def is_up(self)   -> bool: return self.trend_dir == 2
    @property
    def is_down(self) -> bool: return self.trend_dir == 0
    @property
    def is_flat(self) -> bool: return self.trend_dir == 1

    def to_arbiter_dict(self) -> dict:
        return {'trend_dir': self.trend_dir, 'strength': self.strength,
                'rev_prob': self.rev_prob, 'probs': list(self.probs)}


def compute_atr(highs: np.ndarray, lows: np.ndarray,
                closes: np.ndarray, window: int = 14) -> np.ndarray:
    """Average True Range 계산."""
    T  = len(closes)
    tr = np.zeros(T)
    for i in range(1, T):
        tr[i] = max(
            highs[i]  - lows[i],
            abs(highs[i]  - closes[i-1]),
            abs(lows[i]   - closes[i-1]),
        )
    atr = pd.Series(tr).rolling(window, min_periods=1).mean().values
    return atr


def make_triple_barrier_label(
    closes   : np.ndarray,
    atr      : np.ndarray,
    t        : int,
    highs    : np.ndarray = None,
    lows     : np.ndarray = None,
    atr_mult : float = 1.5,
    max_hold : int   = 9,
) -> Tuple[int, float, float]:
    """Triple-Barrier 레이블링 (de Prado 2018 기반).

    highs/lows를 사용해 봉 내 장벽 터치를 정확히 판정한다.
    highs/lows 미제공 시 closes로 대체 (정밀도 저하).

    Returns:
        label   : 0=DOWN / 1=FLAT / 2=UP / -1=동시 터치(제외)
        str_lbl : 강도 [0, 1]
        rev_lbl : 반전 여부 [0, 1]
    """
    T_total   = len(closes)
    cur_close = float(closes[t - 1])
    cur_atr   = float(atr[t - 1])

    _highs = highs if highs is not None else closes
    _lows  = lows  if lows  is not None else closes

    barrier_size = atr_mult * cur_atr
    upper = cur_close + barrier_size
    lower = cur_close - barrier_size

    hit_up = max_hold
    hit_dn = max_hold

    for k in range(1, max_hold + 1):
        if t + k - 1 >= T_total:
            break
        bar_high = float(_highs[t + k - 1])
        bar_low  = float(_lows[t + k - 1])
        if bar_high >= upper and hit_up == max_hold:
            hit_up = k
        if bar_low <= lower and hit_dn == max_hold:
            hit_dn = k
        if hit_up < max_hold and hit_dn < max_hold:
            break

    if hit_up < hit_dn:
        label = 2   # UP
    elif hit_dn < hit_up:
        label = 0   # DOWN
    elif hit_up == hit_dn == max_hold:
        label = 1   # FLAT (어느 장벽도 미터치)
    else:
        return -1, 0.0, 0.0  # 같은 봉에서 양쪽 동시 터치 → 방향 불명확, 제외

    if label != 1:
        hit_time = min(hit_up, hit_dn)
        str_lbl  = float(np.clip(1.0 - (hit_time - 1) / max_hold, 0.0, 1.0))
    else:
        last_price = float(closes[min(t + max_hold - 1, T_total - 1)])
        str_lbl = float(np.tanh(abs(last_price / cur_close - 1) * 20.0))

    past_ret = float(closes[t - 1] / max(float(closes[max(0, t - 6)]), 1e-8) - 1)
    rev_lbl  = 1.0 if (
        (past_ret > 0 and label == 0) or
        (past_ret < 0 and label == 2)
    ) else 0.0

    return label, str_lbl, rev_lbl


class XGBTrendBrain:
    """
    LightGBM 기반 Brain B.

    학습 시 선택된 feature_cols 목록을 pickle 내부에 보존,
    추론 시 누락 컬럼은 0으로 채워서 robust하게 동작한다.
    """

    def __init__(self):
        self.model       = None          # fitted LGBMClassifier
        self.feature_cols: List[str] = []

    # ────────────────────────────────────────────────────────────
    # 추론
    # ────────────────────────────────────────────────────────────
    def predict_from_df(self,
                        df: pd.DataFrame,
                        timestamp_col: str = 'timestamp',
                        min_candles: int   = 1) -> Optional[TrendSignal]:
        """입력 DataFrame의 마지막 행 → TrendSignal.

        Args:
            df: 5m 캔들 DataFrame (feature 컬럼 포함 or OHLCV만 있어도 동작)
            timestamp_col: 타임스탬프 컬럼명 (정렬용)
            min_candles: 최소 행 수 (기본 1)

        Returns:
            TrendSignal or None (데이터 부족 / 모델 미로드)
        """
        if self.model is None or not self.feature_cols:
            return None
        if len(df) < min_candles:
            return None

        df_w = df.copy()
        if timestamp_col in df_w.columns:
            df_w[timestamp_col] = pd.to_datetime(df_w[timestamp_col])
            df_w = df_w.set_index(timestamp_col).sort_index()

        # pred×conf → signal_* 인라인 결합 (라이브 데이터에 pred_* 있을 때)
        _PRED_CONF_MAP = {
            'pred_timesfm': 'conf_timesfm', 'pred_chronos': 'conf_chronos',
            'pred_ttm':     'conf_ttm',     'pred_patchtst': 'conf_patchtst',
            'pred_tide':    'conf_tide',    'pred_mdjd':    'conf_mdjd',
            'pred_ridge':   'conf_ridge',
        }
        for p_col, c_col in _PRED_CONF_MAP.items():
            sig_col = p_col.replace('pred_', 'signal_')
            if sig_col not in df_w.columns:
                if p_col in df_w.columns and c_col in df_w.columns:
                    df_w[sig_col] = df_w[p_col] * df_w[c_col]
                elif p_col in df_w.columns:
                    df_w[sig_col] = df_w[p_col]

        # 추세 구조 피처가 없으면 인라인 생성
        _trend_feats = ['ret_12', 'ret_24', 'ret_48', 'hh_count_24', 'hl_count_24', 'trend_accel']
        if any(f in self.feature_cols and f not in df_w.columns for f in _trend_feats):
            c = df_w['close']
            h = df_w['high'] if 'high' in df_w.columns else c
            l = df_w['low']  if 'low'  in df_w.columns else c
            if 'ret_12' not in df_w.columns:
                df_w['ret_12'] = np.tanh(c.pct_change(12) * 10)
            if 'ret_24' not in df_w.columns:
                df_w['ret_24'] = np.tanh(c.pct_change(24) * 10)
            if 'ret_48' not in df_w.columns:
                df_w['ret_48'] = np.tanh(c.pct_change(48) * 10)
            if 'hh_count_24' not in df_w.columns:
                df_w['hh_count_24'] = (h > h.shift(1)).astype(float).rolling(24, min_periods=1).sum() / 24.0
            if 'hl_count_24' not in df_w.columns:
                df_w['hl_count_24'] = (l > l.shift(1)).astype(float).rolling(24, min_periods=1).sum() / 24.0
            if 'trend_accel' not in df_w.columns:
                df_w['trend_accel'] = np.tanh((c.pct_change(12) - c.pct_change(48) / 4) * 20)

        # 누락 피처 → NaN (LightGBM native missing value 처리)
        # 0 채움은 "RSI=0 극단값"처럼 오해될 수 있으므로 NaN이 안전
        for col in self.feature_cols:
            if col not in df_w.columns:
                df_w[col] = np.nan

        # 가장 최근 행 1개만 사용
        last_row = df_w[self.feature_cols].iloc[[-1]].astype(np.float32)
        last_row = last_row.replace([np.inf, -np.inf], np.nan)  # inf도 NaN으로

        probs = self.model.predict_proba(last_row)[0]   # [P(DOWN), P(FLAT), P(UP)]
        return self._to_trend_signal(probs)

    def _to_trend_signal(self, probs: np.ndarray) -> TrendSignal:
        """(3,) 확률 배열 → TrendSignal."""
        trend_dir = int(np.argmax(probs))

        # 강도: 최대 확률 클래스의 랜덤 기준선(1/3) 초과분 → [0, 1] 클립
        # probs[dir]=0.5 → strength=0.25, 0.7 → 0.55, 0.9 → 0.85
        strength = float(np.clip((probs[trend_dir] - 1.0 / 3.0) * 1.5, 0.0, 1.0))

        # 반전 확률: 예측 방향의 반대편 확률
        if trend_dir == 2:       # UP → DOWN 확률이 반전 위험
            rev_prob = float(probs[0])
        elif trend_dir == 0:     # DOWN → UP 확률이 반전 위험
            rev_prob = float(probs[2])
        else:                    # FLAT
            rev_prob = 0.5

        return TrendSignal(
            trend_dir = trend_dir,
            strength  = strength,
            rev_prob  = rev_prob,
            probs     = tuple(float(p) for p in probs),
        )

    # ────────────────────────────────────────────────────────────
    # 저장 / 로드
    # ────────────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'feature_cols': self.feature_cols}, f)
        logger.info(f"✅ XGBTrendBrain 저장: {path} ({len(self.feature_cols)}개 피처)")

    @classmethod
    def load(cls, path: str) -> 'XGBTrendBrain':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        instance = cls()
        instance.model        = data['model']
        instance.feature_cols = data['feature_cols']
        logger.info(f"✅ XGBTrendBrain 로드: {path} ({len(instance.feature_cols)}개 피처)")
        return instance
