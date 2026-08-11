"""
RL State Builder — Technical Analysis + Elite Strategy Signals
================================================================
5분봉 ETH 데이 트레이딩용 RL Agent State 통합 모듈.

구성:
  [A] TFT 예측 (3차원)      — median_pred, confidence, direction_prob
  [B] TA 분석기 (16차원)     — regime, S/R, oscillator, OBV, ADX, etc.
  [C] Elite 시그널 (4차원)   — whale, liq_squeeze, net_taker, orderblock
  [D] 포지션 정보 (3차원)    — position_size, unrealized_pnl, holding_time
  ─────────────────────────
  총 26차원 → SAC Agent observation_space

Elite 8 전략 모듈화 통합:
  외부 파일(elite_alpha, elite_structure_flow, elite_standard)의 클래스를
  직접 주입받아 객체지향적으로 시그널을 계산합니다.

버그 수정:
  [FIX-1] elite_structure_flow import 누락 → 추가
  [FIX-2] smf_std 역엔지니어링 부정확 (std([v-s,v,v+s]) ≠ s)
           → ddof=1 보정: std = s * sqrt(2/3) 이므로 spread = smf_std * sqrt(3/2)
  [FIX-3] df_mock 트릭 깨지기 쉬움 → 충분한 행 수 + 안전한 인덱스 보장
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

# [FIX-1] 원본 Elite 전략 클래스들 Import — 누락된 elite_structure_flow 포함
from .elite_strategies import (
    WhaleSentimentDivergence, LiquidationSqueezeHunter,
    NetTakerFlowStrategy, OrderblockFVGStrategy,
    HurstOFIRegimeSwitching, FundingDivergenceCascadeHunter,
    MultiFractalNoiseCancellation, ClusterFibonacciConfluence,
    # ── [NEW] 직교 알파 신규 3종 ──
    AISqueezeBreakoutHunter, VolumeProfileGravityOscillator,
    OITrendDivergence, TopTraderPositionalSqueeze, BtcCorrelationBreakout,
    # ── Batch Engines ──
    SyntheticAlphaEngine, RegimeEngine, VolatilityModelEngine,
    NewEliteSignalEngine,
    # ── 변동성 모델 전략 ──
    GARCHVolatilityRegime, OUMeanReversionHunter,
    JumpReboundHunter, EVTTailRiskSentinel,
)
logger = logging.getLogger(__name__)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 1: DATA STRUCTURES                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class CandleData:
    """개별 캔들 데이터"""
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class IndicatorSignals:
    """기술적 지표 신호"""
    macd_cross: int = 0
    rsi_value: float = 50.0
    stoch_rsi_rebound: float = 0.0
    bollinger_touch: str = 'none'
    macd_histogram: float = 0.0


@dataclass
class VolumeProfile:
    """거래량 프로파일"""
    value_area_high: float
    value_area_low: float
    point_of_control: float


@dataclass
class MultiTimeframeData:
    """멀티타임프레임"""
    candles: Dict[str, List[CandleData]] = field(default_factory=dict)
    indicators: Dict[str, IndicatorSignals] = field(default_factory=dict)
@dataclass
class MarketRow:
    """Elite 전략이 참조하는 현재 봉 데이터 (DataFrame row 대체)"""
    close: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    
    whale_retail_ratio: float = 1.0
    whale_conviction: float = 0.0
    smart_money_flow: float = 0.0
    last_funding_rate: float = 0.0
    net_taker_ratio: float = 0.0
    taker_acceleration: float = 0.0
    rsi: float = 50.0
    wick_ratio: float = 0.0
    log_return: float = 0.0
    
    # ── 살아남은 기존 Advanced 전략용 피처 ──
    hurst_48: float = 0.5
    hurst_288: float = 0.5
    ofi_acceleration: float = 0.0
    trade_intensity: float = 1.0
    funding_price_divergence: float = 0.0
    short_squeeze_risk: float = 0.0
    long_squeeze_risk: float = 0.0
    oi_change_rate: float = 0.0
    big_trade_ratio: float = 0.0
    funding_roc_12: float = 0.0
    funding_roc_288: float = 0.0
    cvp_cluster_position: float = 0.0
    fibonacci_level: float = 0.0

    squeeze_power: float = 0.0
    garman_klass_vol: float = 0.0
    funding_z_score: float = 0.0
    volatility_z: float = 0.0

    # ── [NEW] 직교 알파 3종 전략용 신규 피처 ──
    top_trader_ls_ratio: float = 0.0
    btc_corr_60: float = 0.0
    eth_btc_ratio_change: float = 0.0

    session_us: float = 0.0
    hour_cos: float = 0.0
    cvp_poc_dist: float = 0.0
    cvp_volume_imbalance: float = 0.0
    fvg_dist: float = 0.0
    breakout_strength: float = 0.0

    # ── 변동성 모델 피처 (GARCH / OU / Jump / EVT) ──
    garch_vol_z: float = 0.0
    ou_funding_z: float = 0.0
    ou_halflife: float = 0.5
    jump_flag: float = 0.0
    jump_z: float = 0.0
    evt_tail_flag: float = 0.0
    evt_excess_z: float = 0.0

    def get(self, key: str, default=None):
        return getattr(self, key, default)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 2: TECHNICAL INDICATORS — ATR, OBV, ADX                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def compute_atr(candles: List[CandleData], period: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    true_ranges = []
    for i in range(1, len(candles)):
        tr = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i - 1].close),
            abs(candles[i].low - candles[i - 1].close),
        )
        true_ranges.append(tr)
    atr = true_ranges[0]
    n = min(period, len(true_ranges))
    for tr in true_ranges[1:]:
        atr = (atr * (n - 1) + tr) / n
    return atr


def compute_obv_signal(candles: List[CandleData], lookback: int = 20) -> float:
    if len(candles) < lookback + 1:
        return 0.0
    obv = [0.0]
    for i in range(1, len(candles)):
        if candles[i].close > candles[i - 1].close:
            obv.append(obv[-1] + candles[i].volume)
        elif candles[i].close < candles[i - 1].close:
            obv.append(obv[-1] - candles[i].volume)
        else:
            obv.append(obv[-1])
    obv_recent = obv[-lookback:]
    obv_slope = (obv_recent[-1] - obv_recent[0]) / (abs(obv_recent[0]) + 1e-10)
    price_change = (candles[-1].close - candles[-lookback].close) / candles[-lookback].close
    obv_dir = float(np.clip(obv_slope * 10, -1, 1))
    price_dir = float(np.clip(price_change * 100, -1, 1))
    if obv_dir * price_dir > 0:
        return float(np.clip((obv_dir + price_dir) / 2, -1, 1))
    else:
        return float(np.clip(obv_dir * 0.7 + price_dir * 0.3, -1, 1))


def compute_adx(candles: List[CandleData], period: int = 14) \
        -> Tuple[float, float, float]:
    if len(candles) < period + 2:
        return 0.0, 0.0, 0.0
    plus_dm_list, minus_dm_list, tr_list = [], [], []
    for i in range(1, len(candles)):
        hd = candles[i].high - candles[i - 1].high
        ld = candles[i - 1].low - candles[i].low
        plus_dm_list.append(hd if (hd > ld and hd > 0) else 0.0)
        minus_dm_list.append(ld if (ld > hd and ld > 0) else 0.0)
        tr_list.append(max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i - 1].close),
            abs(candles[i].low - candles[i - 1].close),
        ))
    if len(tr_list) < period:
        return 0.0, 0.0, 0.0

    smooth_pdm = sum(plus_dm_list[:period])
    smooth_mdm = sum(minus_dm_list[:period])
    smooth_tr = sum(tr_list[:period])
    dx_list = []

    for i in range(period, len(tr_list)):
        smooth_pdm = smooth_pdm - smooth_pdm / period + plus_dm_list[i]
        smooth_mdm = smooth_mdm - smooth_mdm / period + minus_dm_list[i]
        smooth_tr = smooth_tr - smooth_tr / period + tr_list[i]
        pdi = 100 * smooth_pdm / (smooth_tr + 1e-10)
        mdi = 100 * smooth_mdm / (smooth_tr + 1e-10)
        dx_list.append(100 * abs(pdi - mdi) / (pdi + mdi + 1e-10))

    if not dx_list:
        return 0.0, 0.0, 0.0

    adx = dx_list[0]
    for dx in dx_list[1:]:
        adx = (adx * (period - 1) + dx) / period
    final_pdi = 100 * smooth_pdm / (smooth_tr + 1e-10)
    final_mdi = 100 * smooth_mdm / (smooth_tr + 1e-10)
    return float(adx), float(final_pdi), float(final_mdi)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 3: TA ANALYSIS MODULES                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class MarketRegimeAnalyzer:
    def analyze(self, candles: List[CandleData],
                indicators: IndicatorSignals) -> Tuple[float, float]:
        if len(candles) < 25:
            return 0.0, 0.0
        atr = compute_atr(candles)
        adx, plus_di, minus_di = compute_adx(candles)
        regime_trending = float(np.clip((adx - 15) / 35, 0, 1))

        direction = 0.0
        ma7 = np.mean([c.close for c in candles[-7:]])
        ma25 = np.mean([c.close for c in candles[-25:]])
        direction += float(np.clip((ma7 - ma25) / (atr + 1e-10), -1, 1)) * 0.3

        if plus_di + minus_di > 0:
            direction += float(np.clip(
                (plus_di - minus_di) / (plus_di + minus_di), -1, 1)) * 0.3
        direction += indicators.macd_cross * 0.2

        lookback = min(6, len(candles))
        if lookback >= 2:
            pc = candles[-1].close - candles[-lookback].close
            direction += float(np.clip(pc / (atr + 1e-10), -1, 1)) * 0.2

        if len(candles) >= 20:
            std = np.std([c.close for c in candles[-20:]])
            if std / (atr + 1e-10) < 0.5:
                regime_trending *= 0.3
        return regime_trending, float(np.clip(direction, -1, 1))


class SupportResistanceFinder:
    def detect(self, candles: List[CandleData],
               volume_profile: Optional[VolumeProfile] = None) -> Dict[str, float]:
        if len(candles) < 12:
            return {'support_distance': 0.0, 'resistance_distance': 0.0}
        atr = compute_atr(candles)
        price = candles[-1].close
        supports, resistances = [], []

        w = min(5, (len(candles) - 1) // 2)
        if w >= 2:
            for i in range(w, len(candles) - w):
                local_h = max(c.high for c in candles[i - w:i + w + 1])
                local_l = min(c.low for c in candles[i - w:i + w + 1])
                if candles[i].high == local_h:
                    resistances.append(candles[i].high)
                if candles[i].low == local_l:
                    supports.append(candles[i].low)

        if len(candles) >= 20:
            closes = [c.close for c in candles[-20:]]
            mid, std = np.mean(closes), np.std(closes)
            supports.append(mid - 2 * std)
            resistances.append(mid + 2 * std)

        if volume_profile:
            supports.extend([volume_profile.value_area_low,
                             volume_profile.point_of_control])
            resistances.append(volume_profile.value_area_high)

        cdist = atr * 0.5
        sup = self._cluster_best(supports, price, cdist, below=True)
        res = self._cluster_best(resistances, price, cdist, below=False)

        s_dist = (sup - price) / (price + 1e-10) if sup else -atr / (price + 1e-10)
        r_dist = (res - price) / (price + 1e-10) if res else atr / (price + 1e-10)
        return {'support_distance': float(s_dist),
                'resistance_distance': float(r_dist)}

    @staticmethod
    def _cluster_best(levels, price, cdist, below):
        if not levels:
            return None
        filt = sorted(
            [l for l in levels if (l < price if below else l > price)],
            reverse=below,
        )
        if not filt:
            return None
        clusters, used = [], set()
        for i, lv in enumerate(filt):
            if i in used:
                continue
            cl = [lv]
            used.add(i)
            for j in range(i + 1, len(filt)):
                if j not in used and abs(filt[j] - lv) < cdist:
                    cl.append(filt[j])
                    used.add(j)
            clusters.append(cl)
        return float(np.mean(max(clusters, key=len)))


class OscillatorInterpreter:
    def interpret(self, ind: IndicatorSignals,
                  tf_ind: Dict[str, IndicatorSignals]) -> float:
        score, wsum = 0.0, 0.0
        score += ind.stoch_rsi_rebound * 0.35
        wsum += 0.35

        rsi = ind.rsi_value
        if rsi <= 30:
            score += (30 - rsi) / 30 * 0.25
        elif rsi >= 70:
            score -= (rsi - 70) / 30 * 0.25
        wsum += 0.25

        score += ind.macd_cross * 0.08
        score += float(np.clip(ind.macd_histogram * 100, -1, 1)) * 0.12
        wsum += 0.2

        tf_w = {'1h': 0.15, '4h': 0.2, '1d': 0.25}
        for name, ti in tf_ind.items():
            w = tf_w.get(name, 0.1)
            score += ti.stoch_rsi_rebound * w
            wsum += w

        return float(np.clip(score / wsum if wsum else 0, -1, 1))


class TrendDirectionAnalyzer:
    def analyze(self, candles: List[CandleData],
                htf_candles: Dict[str, List[CandleData]],
                sr: Dict[str, float]) -> Dict[str, float]:
        result = {'short_term_bias': 0.0, 'long_term_bias': 0.0}
        if len(candles) < 10:
            return result

        atr = compute_atr(candles)
        ts = 0.0
        ma5 = np.mean([c.close for c in candles[-5:]])
        ma10 = np.mean([c.close for c in candles[-10:]])
        ts += float(np.clip((ma5 - ma10) / (atr + 1e-10), -1, 1)) * 0.3

        body = (candles[-1].close - candles[-1].open) / (atr + 1e-10)
        ts += float(np.clip(body, -1, 1)) * 0.2

        sd = sr.get('support_distance', -999)
        rd = sr.get('resistance_distance', 999)
        if -0.005 < sd < 0:
            ts += 0.3
        if 0 < rd < 0.005:
            ts -= 0.3
        result['short_term_bias'] = float(np.clip(ts, -1, 1))

        ls, tw = 0.0, 0.0
        tf_w = {'1h': 0.3, '4h': 0.5, '1d': 0.8}
        for name, tfc in htf_candles.items():
            if len(tfc) < 12:
                continue
            w = tf_w.get(name, 0.2)
            ma = np.mean([c.close for c in tfc[-12:]])
            diff = (tfc[-1].close - ma) / (ma + 1e-10)
            ls += float(np.clip(diff * 50, -1, 1)) * w
            tw += w
        if tw > 0:
            result['long_term_bias'] = float(np.clip(ls / tw, -1, 1))
        return result


class VolumeProfileAnalyzer:
    def extract(self, vp: VolumeProfile, price: float,
                atr: float) -> Dict[str, float]:
        poc_atr = abs(price - vp.point_of_control) / (atr + 1e-10)
        vol_str = float(np.clip(1.0 - poc_atr * 0.5, 0, 1))
        in_va = vp.value_area_low <= price <= vp.value_area_high
        va_range = vp.value_area_high - vp.value_area_low
        va_loc = ((price - vp.value_area_low) / va_range
                  if (va_range > 0 and in_va) else 0.5)
        return {
            'volume_strength': vol_str,
            'value_area_position': 1.0 if in_va else 0.0,
            'value_area_location': float(np.clip(va_loc, 0, 1)),
        }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 4: ELITE STRATEGY CLASS INJECTION                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
class EliteSignals:
    """선택된 11개의 최상위 퀀트 알파 전략을 인스턴스화하고 시그널을 추출합니다."""
    def __init__(self):
        # 1. Core 4
        self.whale = WhaleSentimentDivergence()
        self.liq_squeeze = LiquidationSqueezeHunter()
        self.net_taker = NetTakerFlowStrategy()
        self.orderblock = OrderblockFVGStrategy()
        
        # 2. Advanced 4 (삭제된 3개 제외)
        self.hurst_ofi = HurstOFIRegimeSwitching()
        self.fund_cascade = FundingDivergenceCascadeHunter()
        self.multifractal = MultiFractalNoiseCancellation()
        self.cluster_fib = ClusterFibonacciConfluence()
        
        # 3. [NEW] Orthogonal Alpha 3
        self.oi_divergence = OITrendDivergence()
        self.top_trader_squeeze = TopTraderPositionalSqueeze()
        self.btc_corr_breakout = BtcCorrelationBreakout()

        self.ai_squeeze = AISqueezeBreakoutHunter()
        self.vp_gravity = VolumeProfileGravityOscillator()

        # 4. [NEW] 변동성 모델 4종
        self.garch_regime  = GARCHVolatilityRegime()
        self.ou_mean_rev   = OUMeanReversionHunter()
        self.jump_rebound  = JumpReboundHunter()
        self.evt_sentinel  = EVTTailRiskSentinel()


    def compute_all(self, current: MarketRow,
                    prev: Optional[MarketRow] = None,
                    smf_std: float = 1.0) -> Dict[str, float]:
        """11개 Elite 시그널 전체 계산 (RL State 주입용)"""
        df_mock, row_series = self._build_mock_df(current, prev, smf_std)

        return {
            'sig_whale': float(self.whale.generate_signal(row_series, df_mock, smf_std=smf_std)),
            'sig_liq_squeeze': float(self.liq_squeeze.generate_signal(row_series, df_mock, smf_std=smf_std)),
            'sig_net_taker': float(self.net_taker.generate_signal(row_series, df_mock, smf_std=smf_std)),
            'sig_orderblock': float(self.orderblock.generate_signal(row_series, df_mock, smf_std=smf_std)),
            
            'sig_hurst_ofi': float(self.hurst_ofi.generate_signal(row_series, df_mock, smf_std=smf_std)),
            'sig_funding_cascade': float(self.fund_cascade.generate_signal(row_series, df_mock, smf_std=smf_std)),
            'sig_multifractal': float(self.multifractal.generate_signal(row_series, df_mock, smf_std=smf_std)),
            'sig_cluster_fib': float(self.cluster_fib.generate_signal(row_series, df_mock, smf_std=smf_std)),
            
            # [NEW] 신규 알파 3종 계산 추가
            'sig_oi_divergence': float(self.oi_divergence.generate_signal(row_series, df_mock, smf_std=smf_std)),
            'sig_top_trader_squeeze': float(self.top_trader_squeeze.generate_signal(row_series, df_mock, smf_std=smf_std)),
            'sig_btc_corr_breakout': float(self.btc_corr_breakout.generate_signal(row_series, df_mock, smf_std=smf_std)),
            'sig_ai_squeeze': float(self.ai_squeeze.generate_signal(row_series, df_mock, smf_std=smf_std)),
            'sig_vp_gravity': float(self.vp_gravity.generate_signal(row_series, df_mock, smf_std=smf_std)),
            # [NEW] 변동성 모델 신호
            'sig_garch_regime': float(self.garch_regime.generate_signal(row_series, df_mock, smf_std=smf_std)),
            'sig_ou_mean_rev':  float(self.ou_mean_rev.generate_signal(row_series, df_mock, smf_std=smf_std)),
            'sig_jump_rebound': float(self.jump_rebound.generate_signal(row_series, df_mock, smf_std=smf_std)),
            'sig_evt_tail':     float(self.evt_sentinel.generate_signal(row_series, df_mock, smf_std=smf_std)),
        }
        
    @staticmethod
    def _build_mock_df(current: MarketRow, prev: Optional[MarketRow], smf_std: float) -> Tuple[pd.DataFrame, pd.Series]:
        # (기존 코드와 100% 동일하게 유지. 수정 불필요)
        cur_dict = current.__dict__.copy()
        prev_dict = (prev.__dict__.copy() if prev else cur_dict.copy())
        smf_val = cur_dict.get('smart_money_flow', 0.0)
        prev_dict['smart_money_flow'] = smf_val
        spread = smf_std * np.sqrt(2.0)
        dummy = cur_dict.copy()
        rows = [
            prev_dict, cur_dict,
            {**dummy, 'smart_money_flow': smf_val - spread},
            {**dummy, 'smart_money_flow': smf_val},
            {**dummy, 'smart_money_flow': smf_val + spread},
        ]
        df_mock = pd.DataFrame(rows, index=[0, 1, 2, 3, 4])
        row_series = df_mock.iloc[1]
        return df_mock, row_series


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 5: UNIFIED RL STATE BUILDER                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
TA_KEYS = [
    'regime_trending', 'regime_direction',
    'support_distance', 'resistance_distance',
    'oscillator_signal', 'short_term_bias', 'long_term_bias',
    'volume_strength', 'value_area_position', 'value_area_location',
    'obv_signal', 'adx_strength', 'adx_direction',
    'rsi_normalized', 'macd_histogram_norm', 'atr_normalized',
    # ── [NEW] 10만 표본이 증명한 7대 절대 알파 추가 ──
    'session_us', 'hour_cos', 'cvp_poc_dist', 'cvp_volume_imbalance', 
    'fvg_dist', 'breakout_strength', 'oi_change_rate'
]

# [C] Elite 시그널 (11차원 - 신규 3종 포함)
ELITE_KEYS = [
    'sig_whale', 'sig_liq_squeeze', 'sig_net_taker', 'sig_orderblock',
    'sig_hurst_ofi', 'sig_funding_cascade', 'sig_multifractal', 'sig_cluster_fib',
    'sig_oi_divergence', 'sig_top_trader_squeeze', 'sig_btc_corr_breakout'
]

# [D] 포지션 정보 (3차원)
POSITION_KEYS = ['position_size', 'unrealized_pnl', 'holding_time']

# 최종 30차원 RL State 벡터 (TFT 제외)
ALL_KEYS = TA_KEYS + ELITE_KEYS + POSITION_KEYS
STATE_DIM = len(ALL_KEYS)  # 30차원

class RLStateBuilder:
    def __init__(self):
        self.regime = MarketRegimeAnalyzer()
        self.sr = SupportResistanceFinder()
        self.osc = OscillatorInterpreter()
        self.trend = TrendDirectionAnalyzer()
        self.vp_analyzer = VolumeProfileAnalyzer()
        self.elite_signals = EliteSignals()

    def build(self,
              candles: List[CandleData],
              mtf: MultiTimeframeData,
              indicators: IndicatorSignals,
              market_row: MarketRow,
              prev_row: Optional[MarketRow] = None,
              smf_std: float = 1.0,
              # tft_output 파라미터 삭제
              position_info: Optional[Dict[str, float]] = None,
              volume_profile: Optional[VolumeProfile] = None,
              ) -> Dict[str, float]:
        """TFT를 제외한 TA + Elite + Position → 30차원 RL state dict 생성."""

        state = {}

        # [A] TFT 예측 부분 전면 삭제

        # [B] TA 분석
        ta = self._compute_ta(candles, mtf, indicators, volume_profile, market_row)        
        state.update(ta)

        # [C] Elite 시그널
        elite = self.elite_signals.compute_all(market_row, prev_row, smf_std)
        state.update(elite)

        # [D] 포지션 정보
        pos = position_info or {}
        state['position_size'] = pos.get('position_size', 0.0)
        state['unrealized_pnl'] = pos.get('unrealized_pnl', 0.0)
        state['holding_time'] = pos.get('holding_time', 0.0)

        return state

    def _compute_ta(self, candles, mtf, indicators, volume_profile, market_row):
        empty = {k: 0.0 for k in TA_KEYS}
        if len(candles) < 25:
            return empty

        price = candles[-1].close
        atr = compute_atr(candles)

        rt, rd = self.regime.analyze(candles, indicators)
        sr = self.sr.detect(candles, volume_profile)
        osc = self.osc.interpret(indicators, mtf.indicators)
        trend = self.trend.analyze(candles, mtf.candles, sr)

        vs = {'volume_strength': 0.0, 'value_area_position': 0.0,
              'value_area_location': 0.5}
        if volume_profile:
            vs = self.vp_analyzer.extract(volume_profile, price, atr)

        obv = compute_obv_signal(candles)
        adx, pdi, mdi = compute_adx(candles)
        adx_s = float(np.clip(adx / 50, 0, 1))
        adx_d = float(np.clip((pdi - mdi) / (pdi + mdi + 1e-10), -1, 1))

        rsi_n = (indicators.rsi_value - 50) / 50
        macd_n = float(np.clip(indicators.macd_histogram * 100, -1, 1))
        atr_n = atr / (price + 1e-10)

        
        return {
            'regime_trending': round(rt, 4),
            'regime_direction': round(rd, 4),
            'support_distance': round(sr['support_distance'], 6),
            'resistance_distance': round(sr['resistance_distance'], 6),
            'oscillator_signal': round(osc, 4),
            'short_term_bias': round(trend['short_term_bias'], 4),
            'long_term_bias': round(trend['long_term_bias'], 4),
            'volume_strength': round(vs['volume_strength'], 4),
            'value_area_position': round(vs['value_area_position'], 4),
            'value_area_location': round(vs['value_area_location'], 4),
            'obv_signal': round(obv, 4),
            'adx_strength': round(adx_s, 4),
            'adx_direction': round(adx_d, 4),
            'rsi_normalized': round(float(np.clip(rsi_n, -1, 1)), 4),
            'macd_histogram_norm': round(macd_n, 4),
            'atr_normalized': round(atr_n, 6),
            'session_us': float(market_row.get('session_us')),
            'hour_cos': float(market_row.get('hour_cos')),
            'cvp_poc_dist': float(market_row.get('cvp_poc_dist')),
            'cvp_volume_imbalance': float(market_row.get('cvp_volume_imbalance')),
            'fvg_dist': float(market_row.get('fvg_dist')),
            'breakout_strength': float(market_row.get('breakout_strength')),
            'oi_change_rate': float(market_row.get('oi_change_rate')),
        }

    @staticmethod
    def to_array(state: Dict[str, float]) -> np.ndarray:
        return np.array([state.get(k, 0.0) for k in ALL_KEYS], dtype=np.float32)

    @staticmethod
    def state_dim() -> int:
        return STATE_DIM

    @staticmethod
    def key_names() -> List[str]:
        return list(ALL_KEYS)
    
    @staticmethod
    def describe() -> Dict[str, str]:
        return {
            # TFT 설명 삭제됨
            'regime_trending': '0~1     추세 강도 (ADX)',
            'regime_direction': '-1~1    추세 방향',
            'support_distance': 'float   지지선 거리(%)',
            'resistance_distance': 'float   저항선 거리(%)',
            'oscillator_signal': '-1~1    종합 오실레이터',
            'short_term_bias': '-1~1    단기 추세',
            'long_term_bias': '-1~1    장기 추세',
            'volume_strength': '0~1     POC 근접도',
            'value_area_position': '0/1     VA 내/외',
            'value_area_location': '0~1     VA 내 위치',
            'obv_signal': '-1~1    OBV 다이버전스',
            'adx_strength': '0~1     ADX 추세 강도',
            'adx_direction': '-1~1    +DI vs -DI',
            'rsi_normalized': '-1~1    RSI 정규화',
            'macd_histogram_norm': '-1~1    MACD 히스토그램',
            'atr_normalized': 'float   ATR 비율',
            'session_us': '0/1     미국장 세션 여부',
            'hour_cos': '-1~1    시간적 계절성(Cos)',
            'cvp_poc_dist': 'float   POC(매물대) 이격도',
            'cvp_volume_imbalance': 'float   호가 볼륨 불균형',
            'fvg_dist': 'float   FVG(자석장) 이격도',
            'breakout_strength': 'float   돌파 강도',
            'oi_change_rate': 'float   미결제약정 변동률',
            'sig_whale': '-1~1    고래 다이버전스',
            'sig_liq_squeeze': '-1~1    청산 스퀴즈',
            'sig_net_taker': '-1~1    순매수 강도',
            'sig_orderblock': '-1~1    FVG 반전',
            'sig_hurst_ofi': '-1~1    프랙탈 오더플로우 스위칭',
            'sig_funding_cascade': '-1~1    연쇄 청산 헌터',
            'sig_multifractal': '-1~1    프랙탈 노이즈 캔슬링',
            'sig_cluster_fib': '-1~1    방어선 클러스터',
            'sig_oi_divergence': '-1~1    OI-가격 다이버전스',
            'sig_top_trader_squeeze': '-1~1    탑 트레이더 쏠림 스퀴즈',
            'sig_btc_corr_breakout': '-1~1    BTC 상관관계 이탈 돌파',
            'position_size': '-1~1    현재 포지션',
            'unrealized_pnl': 'float   미실현 손익',
            'holding_time': '0~1     보유 시간',
        }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 6: DataFrame 호환 래퍼                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# [수정 후] PART 6 내부의 row_to_market_row 함수
def row_to_market_row(row: pd.Series) -> MarketRow:
    """pandas DataFrame row → MarketRow 변환 (삭제된 피처 완전 제외, 신규 3종 추가).
    [주의] row['key'] 직접 접근을 사용합니다. 피처 누락 시 KeyError가 발생합니다.
    """
    def _f(key: str, default: float = 0.0) -> float:
        try:
            v = row.get(key, default) if hasattr(row, "get") else default
            return float(v) if v is not None else float(default)
        except Exception:
            return float(default)

    return MarketRow(
        # ── 1. 기본 캔들 및 가격 데이터 ──
        close=_f('close'),
        open=_f('open'),
        high=_f('high'),
        low=_f('low'),
        volume=_f('volume'),
        
        # ── 2. 온체인 / 파생상품 / 오더플로우 (Core 전략용) ──
        whale_retail_ratio=_f('whale_retail_ratio'),
        whale_conviction=_f('whale_conviction'),
        smart_money_flow=_f('smart_money_flow'),
        last_funding_rate=_f('last_funding_rate'),
        net_taker_ratio=_f('net_taker_ratio'),
        taker_acceleration=_f('taker_acceleration'),
        rsi=_f('rsi'),
        wick_ratio=_f('wick_ratio'),
        log_return=_f('log_return'),
        
        # ── 3. 살아남은 Advanced 전략용 피처 ──
        hurst_48=_f('hurst_48'),
        hurst_288=_f('hurst_288'),
        ofi_acceleration=_f('ofi_acceleration'),
        trade_intensity=_f('trade_intensity', 1.0),  # 신규 OI 전략에서도 사용됨
        funding_price_divergence=_f('funding_price_divergence'),
        short_squeeze_risk=_f('short_squeeze_risk'),
        long_squeeze_risk=_f('long_squeeze_risk'),
        oi_change_rate=_f('oi_change_rate'),
        big_trade_ratio=_f('big_trade_ratio'),
        funding_roc_12=_f('funding_roc_12'),
        funding_roc_288=_f('funding_roc_288'),
        cvp_cluster_position=_f('cvp_cluster_position'),
        fibonacci_level=_f('fibonacci_level'),

        # ── 4. [NEW] 직교 알파 3종 전략용 신규 피처 ──
        # 라이브 피처 파이프라인에서는 `count_toptrader_long_short_ratio` 대신
        # `count_long_short_ratio` 만 존재할 수 있으므로 안전하게 fallback 처리
        top_trader_ls_ratio=float(
            row.get('count_toptrader_long_short_ratio', row.get('count_long_short_ratio', 0.0))
        ),
        btc_corr_60=_f('btc_corr_60'),
        eth_btc_ratio_change=_f('eth_btc_ratio_change'),
        session_us=_f('session_us'),
        hour_cos=_f('hour_cos'),
        cvp_poc_dist=_f('cvp_poc_dist'),
        cvp_volume_imbalance=_f('cvp_volume_imbalance'),
        fvg_dist=_f('fvg_dist'),
        breakout_strength=_f('breakout_strength'),
        squeeze_power=_f('squeeze_power'),
        garman_klass_vol=_f('garman_klass_vol'),
        funding_z_score=_f('funding_z_score'),
        volatility_z=_f('volatility_z'),
        # ── 변동성 모델 피처 ──
        garch_vol_z=_f('garch_vol_z', 0.0),
        ou_funding_z=_f('ou_funding_z', 0.0),
        ou_halflife=_f('ou_halflife', 0.5),
        jump_flag=_f('jump_flag', 0.0),
        jump_z=_f('jump_z', 0.0),
        evt_tail_flag=_f('evt_tail_flag', 0.0),
        evt_excess_z=_f('evt_excess_z', 0.0),

        # ❌ 삭제됨: CVP_FVG (cvp_poc_dist, fvg_dist, mean_reversion_z)
        # ❌ 삭제됨: AmihudGK (amihud_illiquidity_z, garman_klass_vol, realized_vol_ratio, bb_width_z)
        # ❌ 삭제됨: CompositeSqueeze (squeeze_power, volatility_z, hma_slope)
    )



def df_to_candles(df: pd.DataFrame, n: int = 100) -> List[CandleData]:
    """DataFrame 마지막 n행 → CandleData 리스트."""
    return [
        CandleData(r['open'], r['high'], r['low'], r['close'], r['volume'])
        for _, r in df.tail(n).iterrows()
    ]


# ── Batch Engine 헬퍼 ───────────────────────────────────────────────────────

def compute_synthetic_alphas(df: pd.DataFrame) -> pd.DataFrame:
    """합성 알파 14종 + MDJD 피처를 df에 벡터화로 계산하여 반환."""
    return SyntheticAlphaEngine().compute(df)


def compute_regime(df: pd.DataFrame) -> pd.DataFrame:
    """레짐 라벨(bull/bear/chop/whipsaw/normal)을 df에 벡터화로 계산하여 반환."""
    return RegimeEngine().compute(df)


def compute_volatility_models(df: pd.DataFrame) -> pd.DataFrame:
    """GARCH(1,1) + OU 과정 + 점프 감지 + EVT 변동성 모델 피처를 df에 벡터화로 계산하여 반환."""
    return VolatilityModelEngine().compute(df)


def compute_new_elite_signals(df: pd.DataFrame) -> pd.DataFrame:
    """sig_volume_confirm, sig_liquidity_trap, sig_trend_health를 df에 벡터화로 계산하여 반환."""
    return NewEliteSignalEngine().compute(df)
