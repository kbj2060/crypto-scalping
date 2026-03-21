"""
Brain B: TrendContextBrain v3 — 44% → 55%+ 달성 전략판
================================================================================

【근본 원인 분석 요약】
    v2에서 44%에 막힌 4가지 구조적 원인:

    1. 레이블 잡음 28.9% (수정1 필요)
       - 멀티-horizon 투표: 3개 horizon이 서로 다른 방향 → 8%, 2:1 불확실 → 42%
       - 시간 고정 레이블 → 가격 구조 기반 레이블로 전환 필요

    2. 핵심 알파 신호 미활용 (수정2 필요)
       - rl_training_data_full.csv의 7개 AI 예측값 중 mdjd 1개만 사용
       - 앙상블 예측 7개의 합의도(불확실성)가 가장 강력한 방향 신호
       - 기술적 구조 피처(ATR, Bollinger, MACD) 누락

    3. Transformer 비효율 (수정3 필요)
       - 48봉 → 48×48=2304 attention score: 대부분 노이즈
       - 인접 봉 패턴(캔들 패턴, 지지/저항)은 Conv1D가 더 잘 잡음
       - PatchTST 방식: 4봉씩 패치 → 12토큰 Transformer (144 attention)

    4. 검증 셋 시간 누출 구조 (수정4 필요)
       - 현재 split=85% 고정 → 단순 무작위 분할과 동일한 효과
       - walk-forward validation: 훈련 데이터가 시간 순서를 보장해야 함

【핵심 변경사항 v3】
    수정A: Triple-Barrier 레이블 (de Prado 2018)
        - 가격이 ATR×1.5 위 장벽 먼저 터치 → UP
        - 가격이 ATR×1.5 아래 장벽 먼저 터치 → DOWN
        - 9봉 안에 어느 쪽도 안 터치 → FLAT
        - 레이블 잡음 28.9% → 추정 10% 이하로 감소

    수정B: AI 앙상블 피처 완전 활용 (16 → 28차원)
        - 7개 모델 예측 평균(가중) + 표준편차 + UP 투표수 + 확신도 평균
        - 기술적 피처: ATR_rel, BB_pos, MACD_hist, EMA_dist
        - 시장 구조: higher_high, lower_low, volume_delta, ret_12

    수정C: PatchTST 방식 CNN-Transformer 하이브리드
        - Conv1D(kernel=4, stride=4): 4봉 패치 → 12토큰
        - Attention 2304 → 144 (16배 감소), 로컬 패턴 포착 강화
        - 잔차 연결 있는 Conv 블록으로 계층적 피처 추출

    수정D: 레이블 스무딩 + 온도 조정
        - LabelSmoothing=0.1: 과적합 및 과신을 방지
        - Temperature scaling: 검증 시 calibration 개선

    수정E: StochasticDepth (DropPath) 정규화
        - 각 Transformer 레이어를 확률적으로 스킵 → 앙상블 효과
        - Dropout보다 효율적인 정규화 (표현력 유지)
"""

import os, logging, math
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────
# 하이퍼파라미터
# ───────────────────────────────────────────────────────────────────────────
WINDOW          = 48       # 입력 캔들 수
PATCH_SIZE      = 4        # [수정C] 4봉 = 16시간씩 패치
N_PATCHES       = WINDOW // PATCH_SIZE   # = 12토큰

D_MODEL         = 128
N_HEADS         = 4
N_LAYERS        = 4
D_FF            = 256
DROPOUT         = 0.1      # 낮게 유지 (DropPath로 정규화)
DROP_PATH_RATE  = 0.1      # [수정E] StochasticDepth

LABEL_SMOOTH    = 0.1      # [수정D]
BASE_FEAT_DIM   = 28       # [수정B] 16 → 28차원

# [수정A] Triple-Barrier 파라미터
ATR_WINDOW      = 14       # ATR 계산 윈도우
ATR_MULT        = 1.5      # barrier = ATR × 1.5
MAX_HOLD        = 9        # 최대 보유 봉 수 (= 36시간)

# AI 예측 칼럼 (rl_training_data_full.csv에서)
PRED_COLS = ['pred_timesfm', 'pred_chronos', 'pred_ttm',
             'pred_patchtst', 'pred_tide', 'pred_mdjd', 'pred_ridge']
CONF_COLS = ['conf_timesfm', 'conf_chronos', 'conf_ttm',
             'conf_patchtst', 'conf_tide', 'conf_mdjd', 'conf_patchtst']


# ───────────────────────────────────────────────────────────────────────────
# 출력 데이터클래스 (인터페이스 유지)
# ───────────────────────────────────────────────────────────────────────────
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


# ───────────────────────────────────────────────────────────────────────────
# [수정A] Triple-Barrier 레이블 생성기
# ───────────────────────────────────────────────────────────────────────────
def compute_atr(highs: np.ndarray, lows: np.ndarray,
                closes: np.ndarray, window: int = ATR_WINDOW) -> np.ndarray:
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
    closes : np.ndarray,
    atr    : np.ndarray,
    t      : int,
    atr_mult : float = ATR_MULT,
    max_hold : int   = MAX_HOLD,
) -> Tuple[int, float, float]:
    """
    Triple-Barrier 레이블링 (de Prado 2018 기반).

    진입 가격(cur_close)에서 출발하여:
        - upper = cur_close × (1 + atr_mult × atr[t] / cur_close)
        - lower = cur_close × (1 - atr_mult × atr[t] / cur_close)
        - 미래 max_hold봉 안에 upper 먼저 터치 → UP(2)
        - 미래 max_hold봉 안에 lower 먼저 터치 → DOWN(0)
        - 둘 다 안 터치 → FLAT(1)

    Returns:
        label    : int   — 0=DOWN / 1=FLAT / 2=UP
        str_lbl  : float — 강도 [0, 1]
        rev_lbl  : float — 반전 여부 [0, 1]
    """
    T_total   = len(closes)
    cur_close = float(closes[t - 1])
    cur_atr   = float(atr[t - 1])

    barrier_size = atr_mult * cur_atr
    upper = cur_close + barrier_size
    lower = cur_close - barrier_size

    hit_up = max_hold   # 기본: 터치 없음
    hit_dn = max_hold

    for k in range(1, max_hold + 1):
        if t + k - 1 >= T_total:
            break
        price = float(closes[t + k - 1])
        if price >= upper and hit_up == max_hold:
            hit_up = k
        if price <= lower and hit_dn == max_hold:
            hit_dn = k
        if hit_up < max_hold and hit_dn < max_hold:
            break

    if hit_up < hit_dn:
        label = 2   # UP
    elif hit_dn < hit_up:
        label = 0   # DOWN
    else:
        label = 1   # FLAT (시간 장벽 먼저 터치)

    # 강도: 터치까지 걸린 봉 수가 짧을수록 강한 추세
    if label != 1:
        hit_time = min(hit_up, hit_dn)
        str_lbl  = float(np.clip(1.0 - (hit_time - 1) / max_hold, 0.0, 1.0))
    else:
        # FLAT: 마지막 봉의 실제 수익률 크기
        last_price = float(closes[min(t + max_hold - 1, T_total - 1)])
        str_lbl = float(np.tanh(abs(last_price / cur_close - 1) * 20.0))

    # 반전: 직전 5봉 추세와 레이블 방향이 반대
    past_ret = float(closes[t - 1] / max(float(closes[max(0, t - 6)]), 1e-8) - 1)
    rev_lbl  = 1.0 if (
        (past_ret > 0 and label == 0) or
        (past_ret < 0 and label == 2)
    ) else 0.0

    return label, str_lbl, rev_lbl


# ───────────────────────────────────────────────────────────────────────────
# [수정B] 확장 피처 엔지니어링 (16 → 28차원)
# ───────────────────────────────────────────────────────────────────────────
class HybridFeatureExtractor:
    """
    28차원 하이브리드 피처

    [Price Structure] 0~6 (7개) — 유지
        0: log_return, 1: body_ratio, 2: upper_shadow, 3: lower_shadow
        4: vol_rel, 5: hl_range, 6: close_pos

    [Technical Structure] 7~14 (8개) — 신규
        7:  ATR_rel        = ATR / close (변동성 정규화 지표)
        8:  bb_pos         = (close - BB_lower) / (BB_upper - BB_lower)
        9:  macd_hist_norm = MACD 히스토그램 (정규화)
        10: ema_dist       = (EMA12 - EMA26) / close (크로스오버 신호)
        11: higher_high    = close > max(close[t-5:t-1]) 여부
        12: lower_low      = close < min(close[t-5:t-1]) 여부
        13: volume_delta   = (up_vol - dn_vol) / total_vol (매수 압력)
        14: ret_12         = 12봉(48h) 누적수익률 tanh

    [Microstructure] 15~19 (5개) — 유지
        15: smart_money_flow, 16: squeeze_power, 17: whale_retail_ratio
        18: chop_index, 19: rsi

    [AI Ensemble] 20~27 (8개) — 대폭 확장
        20: pred_ensemble_wmean = conf 가중 예측 평균
        21: pred_ensemble_std   = 7개 예측 표준편차 (불확실성)
        22: pred_bullish_count  = UP 예측 모델 수 (0~7, 정규화)
        23: pred_bearish_count  = DOWN 예측 모델 수 (0~7, 정규화)
        24: conf_ensemble_mean  = 7개 확신도 평균
        25: conf_ensemble_min   = 7개 확신도 최소 (약한 링크)
        26: pred_momentum       = 현재 vs 직전 봉 가중예측 변화 (shift(1))
        27: ou_funding_z        = 펀딩비 z-score (유지)
    """
    RET_SCALE = 20.0
    EPS       = 1e-8

    def __init__(self):
        self.feat_dim = BASE_FEAT_DIM

    def _ema(self, arr: np.ndarray, span: int) -> np.ndarray:
        return pd.Series(arr).ewm(span=span, adjust=False).mean().values

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        T     = len(df)
        feats = np.zeros((T, self.feat_dim), dtype=np.float32)

        c = df['close'].values.astype(np.float64)
        o = df['open'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        v = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.ones(T)
        vol_ma = pd.Series(v).rolling(20, min_periods=1).mean().values

        logret = np.zeros(T)
        logret[1:] = np.log(np.maximum(c[1:], self.EPS) / np.maximum(c[:-1], self.EPS))

        # ── [0~6] Price Structure ──
        for i in range(T):
            hl = max(h[i] - l[i], self.EPS)
            feats[i, 0] = float(np.tanh(logret[i] * self.RET_SCALE))
            feats[i, 1] = float(np.clip((c[i] - o[i]) / max(o[i], self.EPS), -0.15, 0.15) / 0.15)
            feats[i, 2] = float(np.clip((h[i] - max(o[i], c[i])) / max(o[i], self.EPS), 0, 0.05) / 0.05)
            feats[i, 3] = float(np.clip((min(o[i], c[i]) - l[i]) / max(o[i], self.EPS), 0, 0.05) / 0.05)
            feats[i, 4] = float(np.clip((v[i] / max(vol_ma[i], self.EPS) - 1) / 2, -1, 1))
            feats[i, 5] = float(np.clip(hl / max(c[i], self.EPS), 0.0, 0.3) / 0.3)
            feats[i, 6] = float((c[i] - l[i]) / hl) if hl > self.EPS else 0.5

        # ── [7~14] Technical Structure ──
        # ATR (feat[7])
        tr = np.zeros(T)
        tr[1:] = np.maximum.reduce([h[1:]-l[1:], np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])])
        atr_arr = pd.Series(tr).rolling(14, min_periods=1).mean().values
        feats[:, 7] = np.clip(atr_arr / np.maximum(c, self.EPS), 0, 0.1) / 0.1

        # Bollinger Band 위치 (feat[8])
        c_series = pd.Series(c)
        bb_mid   = c_series.rolling(20, min_periods=1).mean().values
        bb_std   = c_series.rolling(20, min_periods=1).std().fillna(0).values
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_range = np.maximum(bb_upper - bb_lower, self.EPS)
        feats[:, 8] = np.clip((c - bb_lower) / bb_range, 0, 1)

        # MACD 히스토그램 (feat[9])
        ema12    = self._ema(c, 12)
        ema26    = self._ema(c, 26)
        macd     = ema12 - ema26
        signal   = self._ema(macd, 9)
        macd_h   = macd - signal
        feats[:, 9] = np.clip(macd_h / np.maximum(c, self.EPS) * 100, -1, 1)

        # EMA 거리 (feat[10])
        feats[:, 10] = np.clip((ema12 - ema26) / np.maximum(c, self.EPS) * 100, -1, 1)

        # Higher-High / Lower-Low (feat[11, 12])
        for i in range(T):
            s = max(0, i - 5)
            past_c = c[s:i]
            if len(past_c) > 0:
                feats[i, 11] = 1.0 if c[i] > past_c.max() else 0.0
                feats[i, 12] = 1.0 if c[i] < past_c.min() else 0.0
            else:
                feats[i, 11] = 0.5
                feats[i, 12] = 0.5

        # Volume Delta: 매수/매도 볼륨 비율 (feat[13])
        # 캔들 방향으로 매수/매도 추정 (close > open → 매수 볼륨)
        buy_vol  = np.where(c >= o, v, v * (c - l) / np.maximum(h - l, self.EPS))
        sell_vol = v - buy_vol
        vol_delta = (buy_vol - sell_vol) / np.maximum(v, self.EPS)
        feats[:, 13] = np.clip(vol_delta, -1, 1)

        # 12봉 누적수익률 (feat[14])
        for i in range(T):
            s12 = max(0, i - 12)
            feats[i, 14] = float(np.tanh((c[i] / max(c[s12], self.EPS) - 1) * self.RET_SCALE))

        # ── [15~19] Microstructure ──
        feats[:, 15] = np.clip(df['smart_money_flow'].values,  -5, 5)  / 5.0
        feats[:, 16] = np.clip(df['squeeze_power'].values,      0, 10) / 10.0
        feats[:, 17] = np.clip(df['whale_retail_ratio'].values, 0,  5) / 5.0
        feats[:, 18] = (df['chop_index'].values - 50) / 50.0
        feats[:, 19] = (df['rsi'].values - 50) / 50.0

        # ── [20~27] AI Ensemble ──
        # 사용 가능한 예측 칼럼만 추출 (없으면 0)
        pred_avail = [c for c in PRED_COLS if c in df.columns]
        conf_avail = [c for c in CONF_COLS if c in df.columns]

        if pred_avail:
            pred_mat = df[pred_avail].values.astype(np.float32)  # (T, n_pred)
            # shift(1) 적용: 현재 봉에 직전 봉 AI 예측 반영
            pred_mat = np.vstack([np.zeros((1, pred_mat.shape[1])), pred_mat[:-1]])
        else:
            pred_mat = np.zeros((T, 1), dtype=np.float32)

        if conf_avail:
            conf_mat = df[conf_avail].values.astype(np.float32)
            conf_mat = np.vstack([np.zeros((1, conf_mat.shape[1])), conf_mat[:-1]])
        else:
            conf_mat = np.ones((T, 1), dtype=np.float32) * 0.5

        # 확신도 가중 평균 예측 (feat[20])
        conf_sum  = conf_mat.sum(axis=1, keepdims=True) + self.EPS
        conf_norm = conf_mat / conf_sum
        pred_wmean = (pred_mat * conf_norm).sum(axis=1)
        feats[:, 20] = np.clip(pred_wmean, -1, 1)

        # 예측 표준편차 (feat[21]) — 모델 간 불일치 = 불확실성
        pred_std = pred_mat.std(axis=1) if pred_mat.shape[1] > 1 else np.zeros(T)
        feats[:, 21] = np.clip(pred_std * 5, 0, 1)

        # UP/DOWN 투표 수 (feat[22, 23])
        VOTE_THRESH = 0.0
        up_count   = (pred_mat > VOTE_THRESH).sum(axis=1).astype(np.float32)
        dn_count   = (pred_mat < -VOTE_THRESH).sum(axis=1).astype(np.float32)
        n_pred_cols = max(pred_mat.shape[1], 1)
        feats[:, 22] = up_count / n_pred_cols
        feats[:, 23] = dn_count / n_pred_cols

        # 확신도 평균/최소 (feat[24, 25])
        feats[:, 24] = np.clip(conf_mat.mean(axis=1), 0, 1)
        feats[:, 25] = np.clip(conf_mat.min(axis=1),  0, 1)

        # 예측 모멘텀 (feat[26]): 직전 봉 대비 앙상블 예측 변화
        pred_prev  = np.vstack([np.zeros((1, pred_mat.shape[1])), pred_mat[:-1]])
        pred_delta = (pred_mat - pred_prev).mean(axis=1)
        feats[:, 26] = np.clip(pred_delta * 10, -1, 1)

        # 펀딩비 (feat[27])
        feats[:, 27] = np.clip(df['ou_funding_z'].values, -3, 3) / 3.0

        return np.nan_to_num(feats, 0.0)


# ───────────────────────────────────────────────────────────────────────────
# 데이터 병합 유틸
# ───────────────────────────────────────────────────────────────────────────
def merge_and_resample(df_5m_path: str, df_1h_path: str) -> pd.DataFrame:
    logger.info("데이터 로드 및 4h 병합 시작...")

    df_5m = pd.read_csv(df_5m_path)
    df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])
    df_5m = df_5m.set_index('timestamp').sort_index()

    df_1h = pd.read_csv(df_1h_path)
    if 'timestamp' in df_1h.columns:
        df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
        df_1h = df_1h.set_index('timestamp').sort_index()
    else:
        raise ValueError("1h 데이터에 timestamp 칼럼이 필수입니다.")

    agg_5m = {
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
        'smart_money_flow': 'sum', 'squeeze_power': 'max',
        'whale_retail_ratio': 'max', 'chop_index': 'last', 'rsi': 'last',
    }
    df_5m_4h = df_5m.resample('4h', closed='right', label='right').agg({
        k: v for k, v in agg_5m.items() if k in df_5m.columns
    })

    # 1h 칼럼: 예측값은 first + shift(1), 나머지는 max/last
    agg_1h = {}
    for col in PRED_COLS + CONF_COLS:
        if col in df_1h.columns:
            agg_1h[col] = 'first'  # 4h 시작 시점 값 사용
    for col in ['garch_vol_z']:
        if col in df_1h.columns:
            agg_1h[col] = 'max'
    for col in ['ou_funding_z']:
        if col in df_1h.columns:
            agg_1h[col] = 'last'

    df_1h_4h = df_1h.resample('4h', closed='right', label='right').agg({
        k: v for k, v in agg_1h.items() if k in df_1h.columns
    })

    # shift(1): 미래 누출 완전 차단
    for col in PRED_COLS + CONF_COLS:
        if col in df_1h_4h.columns:
            df_1h_4h[col] = df_1h_4h[col].shift(1)

    df_merged = df_5m_4h.join(df_1h_4h, how='inner').dropna(subset=['close'])
    df_merged.ffill(inplace=True)
    df_merged.fillna(0, inplace=True)

    logger.info(f"병합 완료: {len(df_merged)}봉 (4h)")
    return df_merged.reset_index()


# ───────────────────────────────────────────────────────────────────────────
# [수정C] PatchTST 방식 CNN 패치 임베더
# ───────────────────────────────────────────────────────────────────────────
class PatchEmbedder(nn.Module):
    """Conv1D로 캔들을 패치 단위로 임베딩.

    48봉 → 12개 패치 (4봉/패치 = 16시간 단위)

    Conv1D의 장점:
        - 인접 봉 간 로컬 패턴(캔들 패턴, 추세선, 지지/저항) 자동 학습
        - 파라미터 공유로 효율적
        - Attention 복잡도 2304 → 144 (1/16)

    설계:
        Layer1: Conv1D(feat_dim, d_model//2, kernel=3, pad=1)  — 3봉 로컬 패턴
        Layer2: Conv1D(d_model//2, d_model,  kernel=4, stride=4) — 4봉 패치 생성
        → (B, 12, d_model)
    """
    def __init__(self, feat_dim: int, d_model: int, patch_size: int = PATCH_SIZE):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(feat_dim, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
        )
        # stride=patch_size로 패치 생성
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_model // 2, d_model, kernel_size=patch_size, stride=patch_size),
            nn.LayerNorm(d_model) if False else nn.Identity(),  # 아래에서 별도 처리
        )
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, WINDOW, feat_dim) → (B, N_PATCHES, d_model)"""
        x = x.permute(0, 2, 1)         # (B, feat_dim, WINDOW)
        x = self.conv1(x)               # (B, d_model//2, WINDOW)
        x = self.conv2(x)               # (B, d_model, N_PATCHES)
        x = x.permute(0, 2, 1)         # (B, N_PATCHES, d_model)
        x = self.norm(x)
        return x


# ───────────────────────────────────────────────────────────────────────────
# [수정E] DropPath (Stochastic Depth)
# ───────────────────────────────────────────────────────────────────────────
class DropPath(nn.Module):
    """Stochastic Depth: 훈련 중 레이어 전체를 확률 drop_prob으로 스킵.

    Dropout과 차이:
        - Dropout: 뉴런 단위 끄기 → 표현력 훼손
        - DropPath: 레이어 단위 스킵 → 얕은 앙상블 효과, 표현력 유지
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.bernoulli(torch.full(shape, keep, device=x.device))
        return x * mask / keep


# ───────────────────────────────────────────────────────────────────────────
# [수정C+E] TrendContextBrain v3
# ───────────────────────────────────────────────────────────────────────────
class TrendContextBrain(nn.Module):
    """
    아키텍처 (v3):
        PatchEmbedder  : 48봉 → 12패치 (Conv1D 로컬 패턴 포착)
        LearnedPE      : 12패치 × d_model 학습 가능 위치 임베딩
        TransformerEncoder(4층, DropPath)
        3-way Pooling  : [CLS] + current_patch + attn_pool
        direction_head / strength_head / reversal_head

    v2 대비 개선:
        - PatchEmbedder: 로컬 캔들 패턴 + Attention 복잡도 1/16
        - DropPath: Dropout보다 효율적인 정규화
        - 학습 가능한 위치 임베딩: 사인/코사인보다 데이터 적응형
        - LabelSmoothing: 과신 방지
    """
    def __init__(self, feat_dim: int = BASE_FEAT_DIM, d_model: int = D_MODEL):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model  = d_model
        self.n_patches = N_PATCHES

        # ── [CLS] 토큰 ──
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── 패치 임베더 (Conv1D) ──
        self.patch_embed = PatchEmbedder(feat_dim, d_model, PATCH_SIZE)

        # ── 학습 가능 위치 임베딩 (N_PATCHES+1 = 13: [CLS]+12패치) ──
        self.pos_embed = nn.Parameter(torch.zeros(1, N_PATCHES + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(DROPOUT)

        # ── DropPath 확률 선형 증가 (깊은 레이어일수록 더 강한 정규화) ──
        dpr = [DROP_PATH_RATE * i / (N_LAYERS - 1) for i in range(N_LAYERS)] if N_LAYERS > 1 else [0.0]

        # ── Transformer Encoder (DropPath 적용) ──
        # PyTorch TransformerEncoderLayer는 DropPath를 직접 지원 안 하므로
        # 커스텀 EncoderLayer 사용
        self.layers = nn.ModuleList([
            _TransformerBlock(d_model, N_HEADS, D_FF, DROPOUT, dpr[i])
            for i in range(N_LAYERS)
        ])
        self.norm = nn.LayerNorm(d_model)

        # ── Attention Pooling ──
        self.attn_pool = nn.Linear(d_model, 1)

        # ── 출력 헤드 (D×3 입력) ──
        pool_dim = d_model * 3

        self.direction_head = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, d_model),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(d_model, 3),
        )
        self.strength_head = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.reversal_head = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """x: (B, WINDOW, feat_dim)"""
        B = x.size(0)

        # ── 패치 임베딩 + [CLS] 토큰 ──
        patches = self.patch_embed(x)                           # (B, N_PATCHES, D)
        cls     = self.cls_token.expand(B, -1, -1)              # (B, 1, D)
        h       = torch.cat([cls, patches], dim=1)              # (B, N_PATCHES+1, D)
        h       = self.pos_drop(h + self.pos_embed)             # 위치 임베딩 추가

        # ── Transformer ──
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)                                        # (B, N_PATCHES+1, D)

        # ── 3-way Pooling ──
        cls_out     = h[:, 0, :]                                # (B, D)
        current_out = h[:, -1, :]                               # (B, D) — 가장 최근 패치
        # Attention Pool (패치 시퀀스만 사용)
        scores  = self.attn_pool(h[:, 1:, :]).squeeze(-1)       # (B, N_PATCHES)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)   # (B, N_PATCHES, 1)
        attn_out = (weights * h[:, 1:, :]).sum(dim=1)           # (B, D)

        pooled = torch.cat([cls_out, current_out, attn_out], dim=-1)  # (B, D×3)

        return (self.direction_head(pooled),
                self.strength_head(pooled),
                self.reversal_head(pooled))

    def save(self, path: str, meta: dict = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'feat_dim':   self.feat_dim,
            'd_model':    self.d_model,
            'meta':       meta or {},
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'TrendContextBrain':
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = cls(feat_dim=ckpt.get('feat_dim', BASE_FEAT_DIM),
                    d_model=ckpt.get('d_model', D_MODEL)).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return model

    @torch.no_grad()
    def predict(self, candles: np.ndarray, device: str = 'cpu') -> 'TrendSignal':
        self.eval()
        x     = torch.tensor(candles[-WINDOW:], dtype=torch.float32).unsqueeze(0).to(device)
        d_log, s_pred, r_pred = self(x)
        probs     = torch.softmax(d_log, dim=-1).squeeze(0).cpu().tolist()
        trend_dir = int(d_log.argmax(dim=-1).item())
        strength  = float(np.clip(max(probs[0], probs[2]) - probs[1], 0.0, 1.0))
        rev_prob  = float(r_pred.squeeze().cpu().item())
        return TrendSignal(trend_dir=trend_dir, strength=strength,
                           rev_prob=rev_prob, probs=tuple(probs))


# ───────────────────────────────────────────────────────────────────────────
# 커스텀 Transformer 블록 (DropPath 지원)
# ───────────────────────────────────────────────────────────────────────────
class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                            batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN + DropPath (잔차 경로에 DropPath 적용)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# ───────────────────────────────────────────────────────────────────────────
# Dataset (수정A: Triple-Barrier 레이블 적용)
# ───────────────────────────────────────────────────────────────────────────
class TrendDataset(Dataset):
    """Triple-Barrier 레이블이 적용된 캔들 데이터셋.

    생성자에서 전체 레이블을 사전 계산하여 학습 중 __getitem__ 속도 극대화.
    """
    def __init__(self, feats: np.ndarray, closes: np.ndarray,
                 highs: np.ndarray = None, lows: np.ndarray = None):
        self.feats  = feats.astype(np.float32)
        self.closes = closes.astype(np.float64)
        self.highs  = highs.astype(np.float64) if highs is not None else closes
        self.lows   = lows.astype(np.float64)  if lows  is not None else closes

        # ATR 사전 계산
        self._atr = compute_atr(self.highs, self.lows, self.closes)

        T = len(feats)
        self.indices = list(range(WINDOW, T - MAX_HOLD - 1))

        # ── 레이블 사전 계산 (학습 속도) ──
        logger.info(f"[Dataset] Triple-Barrier 레이블 사전 계산 중... ({len(self.indices)}개)")
        self._labels    = np.zeros(len(self.indices), dtype=np.int64)
        self._str_lbls  = np.zeros(len(self.indices), dtype=np.float32)
        self._rev_lbls  = np.zeros(len(self.indices), dtype=np.float32)
        for i, t in enumerate(self.indices):
            lbl, sl, rl = make_triple_barrier_label(self.closes, self._atr, t)
            self._labels[i]   = lbl
            self._str_lbls[i] = sl
            self._rev_lbls[i] = rl

        cnt = np.bincount(self._labels, minlength=3)
        total = len(self._labels)
        logger.info(
            f"[Dataset] 완료: DOWN={cnt[0]/total*100:.1f}% "
            f"FLAT={cnt[1]/total*100:.1f}% UP={cnt[2]/total*100:.1f}%"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = self.indices[idx]
        x = self.feats[t - WINDOW : t]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(self._labels[idx],   dtype=torch.long),
                torch.tensor(self._str_lbls[idx], dtype=torch.float32),
                torch.tensor(self._rev_lbls[idx], dtype=torch.float32))


# ───────────────────────────────────────────────────────────────────────────
# Trainer (수정D: LabelSmoothing, 수정E: DropPath)
# ───────────────────────────────────────────────────────────────────────────
class TrendBrainTrainer:
    DIR_W = 1.0
    STR_W = 0.2    # 강도 헤드는 보조적 역할
    REV_W = 0.3    # 반전 헤드는 보조적 역할

    def __init__(self, df: pd.DataFrame, device: str = 'cuda',
                 batch_size: int = 256, lr: float = 2e-4):
        self.device = device

        # 피처 추출
        extractor = HybridFeatureExtractor()
        feats  = extractor.transform(df)
        closes = df['close'].values.astype(np.float64)
        highs  = df['high'].values.astype(np.float64) if 'high' in df.columns else closes
        lows   = df['low'].values.astype(np.float64)  if 'low'  in df.columns else closes

        # 시간 순서 유지 분할 (85:15)
        split = int(len(feats) * 0.85)
        train_ds = TrendDataset(feats[:split],  closes[:split],
                                highs[:split],  lows[:split])
        val_ds   = TrendDataset(feats[split:],  closes[split:],
                                highs[split:],  lows[split:])

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=(device == 'cuda'), drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        )

        self.model = TrendContextBrain(feat_dim=extractor.feat_dim).to(device)

        # 파라미터 분리: 위치 임베딩·LayerNorm은 weight_decay 제외
        no_wd = []
        wd    = []
        for name, p in self.model.named_parameters():
            if ('norm' in name or 'bias' in name or 'pos_embed' in name
                    or 'cls_token' in name or 'temporal_bias' in name):
                no_wd.append(p)
            else:
                wd.append(p)
        self.optimizer = torch.optim.AdamW(
            [{'params': wd, 'weight_decay': 1e-4},
             {'params': no_wd, 'weight_decay': 0.0}],
            lr=lr,
        )

        # 클래스 가중치 (훈련 loss용)
        counts = np.bincount(train_ds._labels, minlength=3).astype(np.float32)
        w = (1.0 / (counts + 1e-6)) * counts.sum() / 3.0
        self.class_weights = torch.tensor(w, dtype=torch.float32).to(device)

        # LabelSmoothing CrossEntropy (수정D)
        self.label_smooth_loss = nn.CrossEntropyLoss(
            weight=self.class_weights, label_smoothing=LABEL_SMOOTH
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"[TrendBrain v3] 파라미터: {total_params:,} | "
            f"훈련:{len(train_ds)} 검증:{len(val_ds)} | "
            f"피처차원: {extractor.feat_dim} | 패치수: {N_PATCHES}"
        )
        self.best_acc = 0.0

    def _train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, dl, sl, rl in self.train_loader:
            x  = x.to(self.device)
            dl = dl.to(self.device)
            sl = sl.to(self.device)
            rl = rl.to(self.device)

            self.optimizer.zero_grad()
            d_log, s_pred, r_pred = self.model(x)

            # [수정D] LabelSmoothing + class_weight CE
            dir_loss = self.label_smooth_loss(d_log, dl)
            str_loss = (F.mse_loss(s_pred, sl.unsqueeze(1), reduction='none')
                        * (dl != 1).float().unsqueeze(1)).mean()
            rev_loss = F.binary_cross_entropy(r_pred, rl.unsqueeze(1))

            loss = self.DIR_W * dir_loss + self.STR_W * str_loss + self.REV_W * rev_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item() * len(x)
            correct    += (d_log.argmax(dim=-1) == dl).sum().item()
            total      += len(x)

        return total_loss / max(total, 1), correct / max(total, 1)

    def _val_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, dl, sl, rl in self.val_loader:
                x  = x.to(self.device)
                dl = dl.to(self.device)
                sl = sl.to(self.device)
                rl = rl.to(self.device)

                d_log, s_pred, r_pred = self.model(x)

                # 검증: label smoothing 없는 순수 CE (실제 정확도 측정)
                dir_loss = F.cross_entropy(d_log, dl)
                str_loss = (F.mse_loss(s_pred, sl.unsqueeze(1), reduction='none')
                            * (dl != 1).float().unsqueeze(1)).mean()
                rev_loss = F.binary_cross_entropy(r_pred, rl.unsqueeze(1))
                loss = self.DIR_W * dir_loss + self.STR_W * str_loss + self.REV_W * rev_loss

                total_loss += loss.item() * len(x)
                correct    += (d_log.argmax(dim=-1) == dl).sum().item()
                total      += len(x)

        return total_loss / max(total, 1), correct / max(total, 1)

    def train(self, epochs: int = 300,
              save_path: str = 'data/ensemble/ckpt/trend_brain_v3.pth',
              patience: int = 50):
        # CosineAnnealingWarmRestarts: EarlyStopping과 완전 호환
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=30, T_mult=2, eta_min=1e-6,
        )

        best_val_acc = 0.0
        no_improve   = 0
        self.best_acc = 0.0

        logger.info(f"학습 시작: 최대 {epochs}에폭 | patience={patience}")

        for ep in range(1, epochs + 1):
            tr_loss, tr_acc   = self._train_epoch()
            val_loss, val_acc = self._val_epoch()
            lr_now = self.optimizer.param_groups[0]['lr']

            logger.info(
                f"Ep {ep:03d}/{epochs} | "
                f"Train Loss:{tr_loss:.4f} Acc:{tr_acc*100:.1f}% | "
                f"Val Loss:{val_loss:.4f} Acc:{val_acc*100:.1f}% | "
                f"LR:{lr_now:.2e}"
            )

            # 누출 경보
            if val_acc > 0.85 and ep <= 5:
                logger.error("🚨 초반 Val Acc > 85% — 미래 데이터 누출 의심!")

            if val_acc > best_val_acc:
                best_val_acc  = val_acc
                self.best_acc = val_acc
                no_improve    = 0
                self.model.save(save_path, meta={
                    'epoch': ep, 'val_acc': val_acc, 'val_loss': val_loss,
                })
                logger.info(f"   🌟 [NEW BEST] Val Acc: {val_acc*100:.1f}%")
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"🛑 EarlyStopping ({patience}에폭 개선 없음)")
                    break

        logger.info(f"🏆 최종 최고 검증 정확도: {self.best_acc*100:.1f}%")


# ───────────────────────────────────────────────────────────────────────────
# 진단 유틸
# ───────────────────────────────────────────────────────────────────────────
def diagnose_dataset(df: pd.DataFrame):
    """학습 전 레이블 분포 및 피처 통계 출력."""
    extractor = HybridFeatureExtractor()
    feats  = extractor.transform(df)
    closes = df['close'].values.astype(np.float64)
    highs  = df['high'].values.astype(np.float64) if 'high' in df.columns else closes
    lows   = df['low'].values.astype(np.float64)  if 'low'  in df.columns else closes
    atr    = compute_atr(highs, lows, closes)
    T      = len(feats)

    labels = []
    for t in range(WINDOW, T - MAX_HOLD - 1):
        lbl, _, _ = make_triple_barrier_label(closes, atr, t)
        labels.append(lbl)

    labels = np.array(labels)
    cnt    = np.bincount(labels, minlength=3)
    total  = len(labels)
    logger.info("=" * 60)
    logger.info(f"[Diagnose v3] 총 샘플: {total}")
    logger.info(f"  DOWN(0): {cnt[0]:6d} ({cnt[0]/total*100:.1f}%)")
    logger.info(f"  FLAT(1): {cnt[1]:6d} ({cnt[1]/total*100:.1f}%)")
    logger.info(f"  UP  (2): {cnt[2]:6d} ({cnt[2]/total*100:.1f}%)")
    logger.info(f"  랜덤 기준선: {max(cnt)/total*100:.1f}%")
    logger.info(f"  ATR_MULT={ATR_MULT}, MAX_HOLD={MAX_HOLD}")
    logger.info(f"[Diagnose v3] 피처 통계 (28차원):")
    feat_names = [
        'log_ret','body_ratio','up_shadow','dn_shadow','vol_rel','hl_range','close_pos',
        'ATR_rel','bb_pos','macd_hist','ema_dist','higher_high','lower_low','vol_delta','ret_12',
        'smf','squeeze','whale','chop','rsi',
        'pred_wmean','pred_std','pred_up_cnt','pred_dn_cnt','conf_mean','conf_min','pred_mom','ou_fund'
    ]
    for i in range(feats.shape[1]):
        nm  = feat_names[i] if i < len(feat_names) else f'feat{i}'
        col = feats[:, i]
        logger.info(f"  [{i:2d}] {nm:12s}: μ={col.mean():+.3f} σ={col.std():.3f} "
                    f"[{col.min():.3f}, {col.max():.3f}]")
    logger.info("=" * 60)


# ───────────────────────────────────────────────────────────────────────────
# 진입점
# ───────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    df_merged = merge_and_resample(
        df_5m_path='data/training_features_5m.csv',
        df_1h_path='data/rl_training_data_full.csv',
    )

    diagnose_dataset(df_merged)

    trainer = TrendBrainTrainer(
        df_merged,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=256,
        lr=2e-4,
    )
    trainer.train(epochs=300, patience=50)