"""
Unified Elite Quant Strategies (11 Core Alpha Signals) - High Sensitivity Tuned
================================================================
RL Agent의 상태(State)로 들어가는 모든 퀀트 알파 시그널을 단일 파일로 통합합니다.
[민감도 대폭 상향 + **kwargs 파라미터 호환 버전]
"""
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

# ═══════════════════════════════════════════════════════════════════════════
#  [1] Original Core 4 Strategies (Alpha & Structure Flow)
# ═══════════════════════════════════════════════════════════════════════════

class WhaleSentimentDivergence(BaseStrategy):
    def __init__(self):
        super().__init__("WhaleSentiment")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        if df is None or len(df) < 2: return 0.0
        idx_pos = df.index.get_loc(row.name)
        if idx_pos == 0: return 0.0
        prev_row = df.iloc[idx_pos - 1]

        ratio      = float(row['whale_retail_ratio'])
        conviction = float(row['whale_conviction'])
        cur_close  = float(row['close'])
        prev_close = float(prev_row['close'])

        price_dir      = 1.0 if cur_close > prev_close else -1.0 if cur_close < prev_close else 0.0
        whale_strength = (ratio - 1.48) * 5.0
        whale_dir      = whale_strength * (1.0 + abs(conviction))

        if price_dir * whale_dir < 0:
            return float(np.clip(whale_dir, -1.0, 1.0))
        else:
            return float(np.clip(whale_dir * 0.3, -1.0, 1.0))


class LiquidationSqueezeHunter(BaseStrategy):
    def __init__(self):
        super().__init__("LiqSqueeze")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        smf     = float(row['smart_money_flow'])
        funding = float(row['last_funding_rate'])

        # Builder가 직접 주입해준 진짜 글로벌 표준편차를 사용!
        smf_std = kwargs.get('smf_std', 1.0)

        oi_strength = smf / smf_std
        # [민감도 향상] 1.0 -> 0.7 로 허들 완화
        if oi_strength < 0.7: return 0.0

        funding_strength = abs(funding) / 0.0003
        # [민감도 향상] 0.3 -> 0.15 로 쏠림 허들 완화
        if funding_strength < 0.15: return 0.0

        squeeze_signal = -np.sign(funding) * min(oi_strength, 2.0) * min(funding_strength, 1.5)
        return float(np.clip(squeeze_signal / 2.0, -1.0, 1.0))  # 3.0 -> 2.0 스케일링 강화


class NetTakerFlowStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("NetTakerFlow")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        ntr = float(row['net_taker_ratio'])
        acc = float(row['taker_acceleration'])

        if np.sign(ntr) == np.sign(acc):
            signal = (ntr * 1.5) + acc
        else:
            signal = ntr * 0.5
        return float(np.clip(signal * 2.0, -1.0, 1.0))


class OrderblockFVGStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("OrderblockFVG")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        wick    = float(row['wick_ratio'])
        log_ret = float(row['log_return'])

        # [민감도 향상] 수익률 0.002 -> 0.001 (0.1% 변동), 꼬리 0.5 -> 0.4 (40%) 로 대폭 완화
        if abs(log_ret) > 0.001 and abs(wick) > 0.4:
            signal = -np.sign(log_ret) * (abs(wick) * 1.5)  # 강도 증폭
            return float(np.clip(signal, -1.0, 1.0))
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  [2] Advanced 7 Strategies (Microstructure, Regime, Math)
# ═══════════════════════════════════════════════════════════════════════════

class HurstOFIRegimeSwitching(BaseStrategy):
    def __init__(self):
        super().__init__("HurstOFI_RegimeSwitching")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        try:
            hurst = float(row['hurst_288'] if 'hurst_288' in row.index else row['hurst_48'])
            ofi_accel  = float(row['ofi_acceleration'])
            taker_accel = float(row['taker_acceleration'])
            net_taker  = float(row['net_taker_ratio'])

            flow_momentum = ofi_accel + (taker_accel * 1.5) + net_taker

            # [초민감도] 0.53 -> 0.51 완화
            if hurst > 0.51:
                signal = np.sign(flow_momentum) * min(abs(flow_momentum) * 3.0, 1.0)
            elif hurst < 0.49:
                if flow_momentum > 0.5:   signal = -0.8
                elif flow_momentum < -0.5: signal = 0.8
                else:                      signal = 0.0
            else:
                # [초민감도] 평상시(랜덤워크)에도 오더플로우를 100% 반영하여 0.0 방지
                signal = flow_momentum * 1.0 
            return float(np.clip(signal, -1.0, 1.0))
        except (AttributeError, TypeError, ValueError):
            return 0.0



class FundingDivergenceCascadeHunter(BaseStrategy):
    def __init__(self):
        super().__init__("FundingDiv_Cascade")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        try:
            div        = float(row['funding_price_divergence'])
            short_risk = float(row['short_squeeze_risk'])
            long_risk  = float(row['long_squeeze_risk'])
            oi_change  = float(row['oi_change_rate'])
            big_trade  = float(row['big_trade_ratio'])

            # [초민감도] 다이버전스(0.6->0.3), 리스크(0.5->0.3), 고래 거래(0.3->0.1) 대폭 완화
            if div > 0.3 and short_risk > 0.3:
                if oi_change <= 0.0 and big_trade > 0.1:
                    return min(div * 2.0, 1.0)
            elif div < -0.3 and long_risk > 0.3:
                if oi_change <= 0.0 and big_trade < -0.1:
                    return max(div * 2.0, -1.0)
            return 0.0
        except (AttributeError, TypeError, ValueError):
            return 0.0


class MultiFractalNoiseCancellation(BaseStrategy):
    def __init__(self):
        super().__init__("MultiFractal_NoiseCancel")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        try:
            roc_12  = float(row['funding_roc_12'])
            roc_288 = float(row['funding_roc_288'])

            # [초민감도] 장기 추세 0.3->0.1, 단기 노이즈 0.5->0.2로 허들 파괴
            if abs(roc_288) > 0.1:
                if np.sign(roc_12) != np.sign(roc_288) and abs(roc_12) > 0.2:
                    signal = np.sign(roc_288) * min(abs(roc_288) * 4.0, 1.0)
                    return float(np.clip(signal, -1.0, 1.0))
            return 0.0
        except (AttributeError, TypeError, ValueError):
            return 0.0


class ClusterFibonacciConfluence(BaseStrategy):
    def __init__(self):
        super().__init__("ClusterFib_Confluence")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        try:
            cvp_pos = float(row['cvp_cluster_position'])
            fib     = float(row['fibonacci_level'])
            wick    = float(row['wick_ratio'])
            log_ret = float(row['log_return'])

            # [초민감도] 오차 범위 0.03 -> 0.08 (8% 오차 허용), 꼬리 길이 0.4 -> 0.2로 대폭 완화
            if abs(cvp_pos - fib) < 0.08 and cvp_pos != 0.0:
                if abs(wick) > 0.2: 
                    signal = -np.sign(log_ret) * (abs(wick) * 2.5)
                    return float(np.clip(signal, -1.0, 1.0))
            return 0.0
        except (AttributeError, TypeError, ValueError):
            return 0.0

class OITrendDivergence(BaseStrategy):
    """
    미결제약정(OI)과 가격의 다이버전스 감지기 (5분봉 초민감도 패치)
    """
    def __init__(self):
        super().__init__("OI_TrendDivergence")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        try:
            oi_change  = float(row['oi_change_rate'])
            log_ret    = float(row['log_return'])
            trade_int  = float(row['trade_intensity'])

            # [수정] 5분봉 기준 0.2% 변동만 있어도 유의미한 수급으로 판단
            if abs(oi_change) > 0.002:
                # 1. 가격 하락 + OI 증가 (강한 숏 진입 누적 -> 반등 시 숏 스퀴즈 위협)
                if log_ret < -0.0005 and oi_change > 0:
                    base_signal = 0.5 * (oi_change * 100.0) # 스케일링 증폭
                    return float(np.clip(base_signal * trade_int, 0.0, 1.0))
                
                # 2. 가격 상승 + OI 증가 (강한 롱 진입 누적 -> 하락 시 롱 스퀴즈 위협)
                elif log_ret > 0.0005 and oi_change > 0:
                    base_signal = -0.5 * (oi_change * 100.0)
                    return float(np.clip(base_signal * trade_int, -1.0, 0.0))
                    
                else:
                    return np.sign(log_ret) * 0.2
            return 0.0
        except (AttributeError, TypeError, ValueError, KeyError):
            return 0.0


class TopTraderPositionalSqueeze(BaseStrategy):
    """
    탑 트레이더 포지션 쏠림 기반 스퀴즈 헌터 (정적 임계치 패치)
    """
    def __init__(self):
        super().__init__("TopTrader_PosSqueeze")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        try:
            ls_ratio = float(row.get('top_trader_ls_ratio', 1.0))
            log_ret = float(row['log_return'])
            
            # [수정] df 길이에 의존하지 않고, 코인 시장의 보편적 극단값(Static Threshold) 적용
            # 롱/숏 비율이 1.5배 이상이면 극단적 롱 쏠림, 0.7 이하이면 극단적 숏 쏠림으로 간주
            q_high = 1.5
            q_low = 0.7

            # 롱 극단 쏠림 -> 하락 시 연쇄 청산(숏 시그널)
            if ls_ratio > q_high and log_ret < 0:
                signal = -0.8 * (ls_ratio / q_high)
                return float(np.clip(signal, -1.0, -0.2))
                
            # 숏 극단 쏠림 -> 상승 시 연쇄 청산(롱 시그널)
            elif ls_ratio < q_low and log_ret > 0:
                signal = 0.8 * (q_low / max(ls_ratio, 0.1))
                return float(np.clip(signal, 0.2, 1.0))
                
            return 0.0
        except (AttributeError, TypeError, ValueError):
            return 0.0


class BtcCorrelationBreakout(BaseStrategy):
    """
    BTC 상관관계 이탈 기반 고유 모멘텀 캐쳐 (정적 임계치 패치)
    """
    def __init__(self):
        super().__init__("BTC_CorrelationBreakout")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        try:
            corr = float(row.get('btc_corr_60', 0.0))
            ratio_change = float(row.get('eth_btc_ratio_change', 0.0))
            accel = float(row['taker_acceleration'])

            # [수정] df.tail() 대신 정적 임계치 사용. 상관관계가 0.3 이하로 깨질 때 디커플링으로 판단
            if corr < 0.3:
                # ETH/BTC 비율 변화와 체결 가속도 방향이 일치할 때
                if np.sign(ratio_change) == np.sign(accel) and accel != 0:
                    signal = np.sign(ratio_change) * (1.0 - corr) * 2.0
                    return float(np.clip(signal, -1.0, 1.0))
            return 0.0
        except (AttributeError, TypeError, ValueError):
            return 0.0

class AISqueezeBreakoutHunter(BaseStrategy):
    """
    [AI 검증 1순위 알파] 변동성 스퀴즈 돌파 전략 (비선형 폭발 캡처)
    - AI 피처 중요도 1위~4위 지표(squeeze_power, garman_klass_vol, funding_z_score, volatility_z)를 결합.
    - 얌전한 오더플로우가 놓치는 플래시 크래시/펌프(Flash Crash/Pump) 타점을 저격합니다.
    """
    def __init__(self):
        super().__init__("AI_Squeeze_Breakout")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        try:
            # 필수 피처 추출 (안전한 .get 활용, 없을 경우 0.0 처리하여 에러 방지)
            sqz_power = float(row.get('squeeze_power', 0.0))
            fund_z = float(row.get('funding_z_score', 0.0))
            vol_z = float(row.get('volatility_z', 0.0))
            gk_vol = max(float(row.get('garman_klass_vol', 0.001)), 1e-5) # 0 나누기 방지
            log_ret = float(row.get('log_return', 0.0))

            # 1. 변동성이 극도로 응축(Squeeze)되었다가 폭발(Breakout, vol_z > 1.5)하는 순간 포착
            if sqz_power > 1.2 and vol_z > 1.5:
                
                # 2. 펀딩비 Z-Score를 통한 스퀴즈 방향성 결정 (역발상 타격)
                # 펀딩비가 극단적 음수(숏 쏠림)일 때 상승 변동성이 터지면 -> 숏 스퀴즈(Long 진입)
                if fund_z < -1.5 and log_ret > 0:
                    base_signal = (sqz_power * vol_z) / 5.0
                    return float(np.clip(base_signal, 0.2, 1.0))
                
                # 펀딩비가 극단적 양수(롱 쏠림)일 때 하락 변동성이 터지면 -> 롱 스퀴즈(Short 진입)
                elif fund_z > 1.5 and log_ret < 0:
                    base_signal = -(sqz_power * vol_z) / 5.0
                    return float(np.clip(base_signal, -1.0, -0.2))
                
                # 펀딩비가 중립적일 때는 순수 변동성 돌파 방향을 추종
                else:
                    signal = np.sign(log_ret) * (vol_z / 4.0)
                    return float(np.clip(signal, -1.0, 1.0))
            
            return 0.0
        except (AttributeError, TypeError, ValueError, KeyError):
            return 0.0


class VolumeProfileGravityOscillator(BaseStrategy):
    """
    [초극단 1% 백테스트 생존 알파] 극단적 매물대 중력 회귀 오실레이터
    - 추세가 없는 횡보장(Hurst < 0.5)에서 가격이 최다 거래 매물대(POC)를 벗어났을 때의 고무줄 탄성(Mean-Reversion)을 계산합니다.
    """
    def __init__(self):
        super().__init__("VP_Gravity_Oscillator")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        try:
            poc_dist = float(row.get('cvp_poc_dist', 0.0))
            hurst = float(row.get('hurst_48', 0.5))
            
            # 1. 국면 필터링: 추세장(Hurst > 0.5)에서는 회귀 전략이 박살 나므로 철저히 차단(Lock)
            if hurst < 0.48:
                # 2. 이격도 필터링: POC(매물대)로부터 충분히 멀어졌을 때만 작동
                # (스케일은 데이터에 따라 다르나, 통상 정규화된 값이 0.01 이상일 때 유의미)
                if abs(poc_dist) > 0.01:
                    
                    # 3. 횡보 성향이 강할수록(Hurst가 낮을수록) 회귀 강도 증폭
                    reversion_strength = (0.5 - hurst) * 2.0  # 최대 1.0
                    
                    # 4. 방향성: 가격이 POC 위에 있으면(양수) 숏(-), 아래에 있으면(음수) 롱(+)
                    signal = -np.sign(poc_dist) * min(abs(poc_dist) * 50.0, 1.0) * reversion_strength
                    
                    return float(np.clip(signal, -1.0, 1.0))
            
            return 0.0
        except (AttributeError, TypeError, ValueError, KeyError):
            return 0.0