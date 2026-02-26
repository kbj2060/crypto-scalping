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
from elite_alpha import WhaleSentimentDivergence, LiquidationSqueezeHunter
from elite_structure_flow import NetTakerFlowStrategy, OrderblockFVGStrategy
from elite_standard import BTCEthCorrelation, VolatilitySqueeze, VWAPDeviation, HMAMomentum

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
    # 온체인 / 파생상품
    whale_retail_ratio: float = 1.0
    whale_conviction: float = 0.0
    smart_money_flow: float = 0.0
    last_funding_rate: float = 0.0
    # 오더플로우
    net_taker_ratio: float = 0.0
    taker_acceleration: float = 0.0
    # 캔들 패턴
    rsi: float = 50.0
    wick_ratio: float = 0.0
    log_return: float = 0.0
    # elite_standard 호환 필드
    btc_corr_60: float = 0.0
    bb_width_z: float = 0.0
    vwap_dist: float = 0.0
    hma_slope: float = 0.0

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
    """
    원본 전략 파일의 클래스를 직접 인스턴스화하여 호출.

    [FIX-2] smf_std 보정:
      pandas std(ddof=1)에서 [v-s, v, v+s]의 std = s * sqrt(2/3)
      → 원하는 smf_std가 나오려면 spread = smf_std * sqrt(3/2) 로 보정

    [FIX-3] df_mock 안정성:
      - prev_row와 cur_row를 명시적으로 인덱스 0, 1에 배치
      - row_series.name = 1 → df.index.get_loc(1) = 1 → prev = df.iloc[0] ✓
      - smf 보정용 더미 행은 인덱스 2~4에 추가 (실제 조회에 영향 없음)
    """

    def __init__(self):
        # 핵심 4개 (RL state에 사용)
        self.whale_strat = WhaleSentimentDivergence()
        self.liq_squeeze_strat = LiquidationSqueezeHunter()
        self.net_taker_strat = NetTakerFlowStrategy()
        self.orderblock_strat = OrderblockFVGStrategy()

        # 호환성/미래 확장용 (현재 RL state에 미사용)
        self.btc_eth_strat = BTCEthCorrelation()
        self.vol_squeeze_strat = VolatilitySqueeze()
        self.vwap_dev_strat = VWAPDeviation()
        self.hma_mom_strat = HMAMomentum()

    def compute_all(self, current: MarketRow,
                    prev: Optional[MarketRow] = None,
                    smf_std: float = 1.0) -> Dict[str, float]:
        """4개 핵심 Elite 시그널 계산."""

        df_mock, row_series = self._build_mock_df(current, prev, smf_std)

        return {
            'sig_whale': float(self.whale_strat.generate_signal(
                row_series, df_mock)),
            'sig_liq_squeeze': float(self.liq_squeeze_strat.generate_signal(
                row_series, df_mock)),
            'sig_net_taker': float(self.net_taker_strat.generate_signal(
                row_series, df_mock)),
            'sig_orderblock': float(self.orderblock_strat.generate_signal(
                row_series, df_mock)),
        }

    def compute_standard(self, current: MarketRow,
                         prev: Optional[MarketRow] = None,
                         smf_std: float = 1.0) -> Dict[str, float]:
        """Standard 4개 (독립 사용 시)."""
        df_mock, row_series = self._build_mock_df(current, prev, smf_std)

        return {
            'sig_btc_eth_corr': float(self.btc_eth_strat.generate_signal(
                row_series, df_mock)),
            'sig_vol_squeeze': float(self.vol_squeeze_strat.generate_signal(
                row_series, df_mock)),
            'sig_vwap_dev': float(self.vwap_dev_strat.generate_signal(
                row_series, df_mock)),
            'sig_hma_momentum': float(self.hma_mom_strat.generate_signal(
                row_series, df_mock)),
        }

    def compute_all_8(self, current: MarketRow,
                      prev: Optional[MarketRow] = None,
                      smf_std: float = 1.0) -> Dict[str, float]:
        """전체 8개 (독립 사용 시)."""
        result = self.compute_all(current, prev, smf_std)
        result.update(self.compute_standard(current, prev, smf_std))
        return result

    @staticmethod
    def _build_mock_df(current: MarketRow,
                       prev: Optional[MarketRow],
                       smf_std: float) -> Tuple[pd.DataFrame, pd.Series]:
        """
        [FIX-2 + FIX-3] 원본 전략 호환 mock DataFrame 생성.

        구조 (5행):
          index 0: prev_row 데이터                    ← whale_strat 이전 봉 조회용
          index 1: current_row 데이터                  ← row_series (분석 대상)
          index 2: smart_money_flow = smf_val - k     ← smf_std 보정용
          index 3: smart_money_flow = smf_val         ← smf_std 보정용
          index 4: smart_money_flow = smf_val + k     ← smf_std 보정용

        smf_std 보정 수학:
          5개 값 [v, v, v-k, v, v+k]의 pandas std(ddof=1):
            mean = v, 편차 = [0, 0, -k, 0, k]
            var = (0 + 0 + k² + 0 + k²) / (5-1) = 2k²/4 = k²/2
            std = k / sqrt(2)
          → 원하는 std가 target이면: k = target * sqrt(2)

          핵심: prev_row의 smf 값도 smf_val로 맞춰야 정확.
          prev_row의 다른 컬럼(close 등)은 원본 유지 (whale_strat이 참조).
        """
        cur_dict = current.__dict__.copy()
        prev_dict = (prev.__dict__.copy() if prev else cur_dict.copy())

        smf_val = cur_dict.get('smart_money_flow', 0.0)

        # [FIX-2] prev_row의 smf도 smf_val로 통일 → std 정확도 보장
        # (prev의 close 등 다른 컬럼은 원본 유지)
        prev_dict['smart_money_flow'] = smf_val

        # k = target_std * sqrt(2) → std([v,v,v-k,v,v+k], ddof=1) = target_std
        spread = smf_std * np.sqrt(2.0)

        # [FIX-3] 안정적인 5행 DataFrame
        dummy = cur_dict.copy()
        rows = [
            prev_dict,                                          # index 0: prev
            cur_dict,                                           # index 1: current ★
            {**dummy, 'smart_money_flow': smf_val - spread},    # index 2: 더미
            {**dummy, 'smart_money_flow': smf_val},             # index 3: 더미
            {**dummy, 'smart_money_flow': smf_val + spread},    # index 4: 더미
        ]
        df_mock = pd.DataFrame(rows, index=[0, 1, 2, 3, 4])

        # row_series: index=1 → whale_strat에서
        # df.index.get_loc(row.name) = 1 → prev = df.iloc[0] ✓
        row_series = df_mock.iloc[1]

        return df_mock, row_series


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 5: UNIFIED RL STATE BUILDER                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── State 키 정의 (순서 고정) ──
TFT_KEYS = ['tft_median_pred', 'tft_confidence', 'tft_direction_prob']

TA_KEYS = [
    'regime_trending', 'regime_direction',
    'support_distance', 'resistance_distance',
    'oscillator_signal', 'short_term_bias', 'long_term_bias',
    'volume_strength', 'value_area_position', 'value_area_location',
    'obv_signal', 'adx_strength', 'adx_direction',
    'rsi_normalized', 'macd_histogram_norm', 'atr_normalized',
]

ELITE_KEYS = ['sig_whale', 'sig_liq_squeeze', 'sig_net_taker', 'sig_orderblock']
POSITION_KEYS = ['position_size', 'unrealized_pnl', 'holding_time']

ALL_KEYS = TFT_KEYS + TA_KEYS + ELITE_KEYS + POSITION_KEYS
STATE_DIM = len(ALL_KEYS)  # 26


class RLStateBuilder:
    """
    TFT + TA + Elite + Position → 26차원 RL State 벡터.

    사용법:
        builder = RLStateBuilder()
        state = builder.build(candles, mtf, indicators, market_row, ...)
        array = builder.to_array(state)  # shape (26,)
    """

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
              tft_output: Optional[Dict[str, float]] = None,
              position_info: Optional[Dict[str, float]] = None,
              volume_profile: Optional[VolumeProfile] = None,
              ) -> Dict[str, float]:
        """전체 RL state dict 생성."""

        state = {}

        # [A] TFT 예측
        tft = tft_output or {}
        state['tft_median_pred'] = tft.get('median_pred', 0.0)
        state['tft_confidence'] = tft.get('confidence', 0.0)
        state['tft_direction_prob'] = tft.get('direction_prob', 0.5)

        # [B] TA 분석
        ta = self._compute_ta(candles, mtf, indicators, volume_profile)
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

    def _compute_ta(self, candles, mtf, indicators, volume_profile):
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
            'tft_median_pred': 'float   TFT 중앙값 예측',
            'tft_confidence': '0~1     TFT 신뢰도',
            'tft_direction_prob': '0~1     TFT 방향 확률',
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
            'sig_whale': '-1~1    고래 다이버전스',
            'sig_liq_squeeze': '-1~1    청산 스퀴즈',
            'sig_net_taker': '-1~1    순매수 강도',
            'sig_orderblock': '-1~1    FVG 반전',
            'position_size': '-1~1    현재 포지션',
            'unrealized_pnl': 'float   미실현 손익',
            'holding_time': '0~1     보유 시간',
        }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 6: DataFrame 호환 래퍼                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def row_to_market_row(row: pd.Series) -> MarketRow:
    """pandas DataFrame row → MarketRow 변환."""
    return MarketRow(
        close=row.get('close', 0), open=row.get('open', 0),
        high=row.get('high', 0), low=row.get('low', 0),
        volume=row.get('volume', 0),
        whale_retail_ratio=row.get('whale_retail_ratio', 1.0),
        whale_conviction=row.get('whale_conviction', 0),
        smart_money_flow=row.get('smart_money_flow', 0),
        last_funding_rate=row.get('last_funding_rate', 0),
        net_taker_ratio=row.get('net_taker_ratio', 0),
        taker_acceleration=row.get('taker_acceleration', 0),
        rsi=row.get('rsi', 50), wick_ratio=row.get('wick_ratio', 0),
        log_return=row.get('log_return', 0),
        btc_corr_60=row.get('btc_corr_60', 0),
        bb_width_z=row.get('bb_width_z', 0),
        vwap_dist=row.get('vwap_dist', 0),
        hma_slope=row.get('hma_slope', 0),
    )


def df_to_candles(df: pd.DataFrame, n: int = 100) -> List[CandleData]:
    """DataFrame 마지막 n행 → CandleData 리스트."""
    return [
        CandleData(r['open'], r['high'], r['low'], r['close'], r['volume'])
        for _, r in df.tail(n).iterrows()
    ]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 7: VERIFICATION TEST                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    # ── 가상 캔들 ──
    price = 2065.0
    candles = []
    for _ in range(100):
        ch = np.random.randn() * 3
        o = price + ch
        h = o + abs(np.random.randn()) * 5
        l = o - abs(np.random.randn()) * 5
        c = o + np.random.randn() * 2
        v = 100000 + np.random.randint(-20000, 50000)
        candles.append(CandleData(o, h, l, c, v))
        price = c
    candles[-1] = CandleData(2068, 2072, 2064, 2068.89, 150000)

    # ── 멀티타임프레임 ──
    mtf = MultiTimeframeData(
        candles={
            '1h': [CandleData(2065, 2075, 2055, 2070, 5e6) for _ in range(24)],
            '4h': [CandleData(2050, 2085, 2040, 2060, 15e6) for _ in range(12)],
        },
        indicators={
            '1h': IndicatorSignals(macd_cross=0, rsi_value=55,
                                   stoch_rsi_rebound=0.3, macd_histogram=0.001),
            '4h': IndicatorSignals(macd_cross=1, rsi_value=48,
                                   stoch_rsi_rebound=-0.2, macd_histogram=0.002),
        },
    )
    indicators = IndicatorSignals(
        macd_cross=1, rsi_value=35, stoch_rsi_rebound=0.85,
        bollinger_touch='lower', macd_histogram=0.003,
    )

    # ── Elite용 MarketRow ──
    cur_row = MarketRow(
        close=2068.89, open=2068, high=2072, low=2064, volume=150000,
        whale_retail_ratio=1.58, whale_conviction=0.3,
        smart_money_flow=2.5, last_funding_rate=-0.0002,
        net_taker_ratio=0.15, taker_acceleration=0.05,
        rsi=35, wick_ratio=0.6,
    )
    prev_row = MarketRow(close=2070, open=2069, high=2073, low=2065, volume=120000)

    # ── 빌드 ──
    builder = RLStateBuilder()
    vp = VolumeProfile(2075, 2060, 2065)
    state = builder.build(
        candles=candles, mtf=mtf, indicators=indicators,
        market_row=cur_row, prev_row=prev_row, smf_std=1.2,
        tft_output={'median_pred': 0.003, 'confidence': 0.72, 'direction_prob': 0.65},
        position_info={'position_size': 0.0, 'unrealized_pnl': 0.0, 'holding_time': 0.0},
        volume_profile=vp,
    )
    state_array = builder.to_array(state)

    # ── [FIX-2 검증] smf_std 정확도 테스트 ──
    target_std = 1.2
    _, row_s = EliteSignals._build_mock_df(cur_row, prev_row, target_std)
    df_test, _ = EliteSignals._build_mock_df(cur_row, prev_row, target_std)
    actual_std = df_test['smart_money_flow'].std()
    std_error = abs(actual_std - target_std) / target_std * 100

    # ── 출력 ──
    print("\n" + "=" * 70)
    print("📊 RL Agent Unified State Vector — 26 dimensions")
    print("=" * 70)

    desc = builder.describe()
    sections = [
        ("🔮 [A] TFT Prediction", TFT_KEYS),
        ("📈 [B] Technical Analysis", TA_KEYS),
        ("🐋 [C] Elite Signals", ELITE_KEYS),
        ("💼 [D] Position Info", POSITION_KEYS),
    ]
    idx = 0
    for title, keys in sections:
        print(f"\n  {title}\n  {'─' * 60}")
        for k in keys:
            v = state.get(k, 0.0)
            bar = "█" * int(abs(v) * 20)
            sign = "+" if v >= 0 else "-"
            print(f"  [{idx:2d}] {k:25s} {v:+.4f}  {sign}{bar}")
            idx += 1

    print(f"\n  {'─' * 60}")
    print(f"  Total dimensions: {len(state_array)}")
    print(f"  State array: {state_array}")
    print(f"  SAC observation_space = Box(-inf, inf, shape=({STATE_DIM},))")

    # FIX-2 검증 결과
    print(f"\n  {'─' * 60}")
    print(f"  [FIX-2 검증] smf_std 정확도:")
    print(f"    목표 std: {target_std:.4f}")
    print(f"    실제 std: {actual_std:.4f}")
    print(f"    오차:     {std_error:.1f}%")
    print(f"    {'✅ PASS' if std_error < 15 else '❌ FAIL'} (허용 15%)")