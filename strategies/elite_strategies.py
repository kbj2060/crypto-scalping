"""
Unified Elite Quant Strategies (11 Core Alpha Signals) - High Sensitivity Tuned
================================================================
RL Agent의 상태(State)로 들어가는 모든 퀀트 알파 시그널을 단일 파일로 통합합니다.
[민감도 대폭 상향 + **kwargs 파라미터 호환 버전]
"""
import pandas as pd
import numpy as np
from scipy.stats import norm as _scipy_norm
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


# ═══════════════════════════════════════════════════════════════════════════
#  [Batch Engines] DataFrame 전체 벡터화 계산 엔진
# ═══════════════════════════════════════════════════════════════════════════

class SyntheticAlphaEngine:
    """배치 합성 알파 피처 계산 엔진 (DataFrame 전체 벡터화)

    전제: df에 smf_std 컬럼이 이미 있어야 함 (호출 전 계산 필요).
    """
    COLS = [
        'ofti', 'kel', 'mta_funding', 'svps',
        'pred_mdjd', 'conf_mdjd',
        'cada', 'mshd', 'fvci', 'wpad', 'fdlv',
        'vsdi', 'vebr', 'tlad', 'mtmb', 'fcsz',
    ]
    _ROLL = 288

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        _ROLL = self._ROLL

        # OFTI
        _ofti_raw = (
            df.get('smart_money_flow', pd.Series(0., index=df.index))
            * df.get('whale_conviction', pd.Series(0., index=df.index))
            * (df.get('amihud_illiquidity_z', pd.Series(0., index=df.index)).abs() + 1.0)
        )
        df['ofti'] = np.tanh(_ofti_raw * 3.0).fillna(0)

        # KEL
        _kel_raw = (
            df.get('oi_change_rate', pd.Series(0., index=df.index))
            / (df.get('garman_klass_vol', pd.Series(1e-6, index=df.index)) + 1e-6)
            * np.sign(df.get('funding_pressure', pd.Series(0., index=df.index)))
        )
        _kel_mean = _kel_raw.rolling(_ROLL, min_periods=1).mean()
        _kel_std  = _kel_raw.rolling(_ROLL, min_periods=1).std().replace(0, 1e-8)
        df['kel'] = np.tanh((_kel_raw - _kel_mean) / _kel_std * 0.5).fillna(0)

        # MTA
        _weighted_roc = (
            0.5 * df.get('funding_roc_12',  pd.Series(0., index=df.index))
            + 0.3 * df.get('funding_roc_48',  pd.Series(0., index=df.index))
            + 0.2 * df.get('funding_roc_288', pd.Series(0., index=df.index))
        )
        _funding_abs = df.get('funding_abs', pd.Series(1e-5, index=df.index)).clip(lower=1e-5)
        _sq          = df.get('squeeze_power', pd.Series(0., index=df.index))
        _sq_z        = (_sq - _sq.rolling(_ROLL, min_periods=1).mean()) \
                       / (_sq.rolling(_ROLL, min_periods=1).std().replace(0, 1e-8))
        df['mta_funding'] = ((_weighted_roc / _funding_abs) * np.tanh(_sq_z)).clip(-3, 3) / 3
        df['mta_funding'] = df['mta_funding'].fillna(0)

        # SVPS
        df['svps'] = np.tanh(
            2.0
            * df.get('cvp_poc_dist',         pd.Series(0., index=df.index))
            * df.get('cvp_volume_imbalance',  pd.Series(0., index=df.index))
            * np.exp(-df.get('cvp_vah_val_width', pd.Series(0., index=df.index)).clip(0, 5))
        ).fillna(0)

        # MDJD
        _sqz_mean  = df.get('squeeze_power', pd.Series(0., index=df.index)).rolling(288, min_periods=1).mean()
        _sqz_std   = df.get('squeeze_power', pd.Series(0., index=df.index)).rolling(288, min_periods=1).std().replace(0, 1e-8)
        _squeeze_z = (df.get('squeeze_power', pd.Series(0., index=df.index)) - _sqz_mean) / _sqz_std
        _trend_4h  = df.get('mtf_trend_4h', pd.Series(0., index=df.index))
        _trend_4h_z = _trend_4h / (_trend_4h.rolling(288, min_periods=1).std() + 1e-8)
        _D = (0.005 * df.get('smart_money_flow', pd.Series(0., index=df.index))
              * (1 + np.tanh(df.get('whale_conviction', pd.Series(0., index=df.index))))
              + 0.002 * _trend_4h)
        _I = (0.003 * df.get('net_taker_ratio', pd.Series(0., index=df.index))
              * np.exp(np.tanh(df.get('taker_acceleration', pd.Series(0., index=df.index))))
              * (df.get('amihud_illiquidity_z', pd.Series(0., index=df.index)).clip(lower=0) + 1.0))
        _J = (0.01 * np.tanh(_squeeze_z)
              * np.tanh(df.get('funding_pressure', pd.Series(0., index=df.index)))
              * (df.get('breakout_strength', pd.Series(0., index=df.index)) > 0.4).astype(float))
        _trend_dampener = 1.0 - np.tanh(_trend_4h_z.abs())
        _G = (-0.005 * df.get('cvp_poc_dist', pd.Series(0., index=df.index))
              * np.exp(-df.get('cvp_volume_imbalance', pd.Series(0., index=df.index)).clip(-5, 5))
              * _trend_dampener)
        _R_hat = _D + _I + _J + _G
        df['pred_mdjd'] = np.sign(_R_hat).clip(-1, 1).fillna(0)
        df['conf_mdjd'] = np.tanh(_R_hat.abs() * 100).fillna(0)

        # CADA
        df['cada'] = np.tanh(
            df.get('eth_btc_ratio_change', pd.Series(0., index=df.index))
            * np.exp(-df.get('btc_corr_60', pd.Series(0., index=df.index)).clip(-1, 1))
            * df.get('smart_money_flow', pd.Series(0., index=df.index))
        ).fillna(0)

        # MSHD
        df['mshd'] = (
            -np.sign(df.get('log_return', pd.Series(0., index=df.index)))
            * df.get('wick_ratio', pd.Series(0., index=df.index)).clip(0, 5)
            * np.tanh(df.get('big_trade_ratio', pd.Series(0., index=df.index)))
            * np.exp(-df.get('trade_intensity', pd.Series(0., index=df.index)).clip(0, 5))
        ).fillna(0)

        # FVCI
        df['fvci'] = (
            (1.0 - df.get('chop_index', pd.Series(50., index=df.index)).clip(0, 100) / 100.0)
            * np.tanh(df.get('volatility_z', pd.Series(0., index=df.index)))
            * np.sign(df.get('hurst_change', pd.Series(0., index=df.index)))
        ).fillna(0)

        # WPAD
        df['wpad'] = np.tanh(
            3.0
            * df.get('whale_retail_ratio', pd.Series(0., index=df.index))
            * (df.get('smart_money_flow', pd.Series(0., index=df.index))
               - df.get('net_taker_ratio', pd.Series(0., index=df.index)))
        ).fillna(0)

        # FDLV
        df['fdlv'] = (
            np.sign(df.get('fvg_dist', pd.Series(0., index=df.index)))
            * np.tanh(
                200.0
                * df.get('fvg_dist', pd.Series(0., index=df.index)).abs()
                * df.get('taker_acceleration', pd.Series(0., index=df.index))
                * np.exp(-df.get('cvp_vah_val_width', pd.Series(0., index=df.index)).clip(0, 5))
            )
        ).fillna(0)

        # VSDI
        df['vsdi'] = (
            np.tanh(
                (df.get('garman_klass_vol', pd.Series(0., index=df.index))
                 - df.get('parkinson_vol', pd.Series(0., index=df.index)))
                / (df.get('rogers_satchell_vol', pd.Series(1e-8, index=df.index)) + 1e-8)
            )
            * df.get('regime_break', pd.Series(0., index=df.index))
        ).fillna(0)

        # VEBR
        df['vebr'] = -np.tanh(
            100.0
            * df.get('vwap_dist', pd.Series(0., index=df.index))
            / (df.get('bb_width_z', pd.Series(0., index=df.index)) + 3.0)
            * df.get('mean_reversion_z', pd.Series(0., index=df.index))
        ).fillna(0)

        # TLAD
        df['tlad'] = (
            np.sign(df.get('log_return', pd.Series(0., index=df.index)))
            * (1.0 - np.exp(-df.get('amihud_illiquidity_z', pd.Series(0., index=df.index)).clip(lower=0)))
            * df.get('is_hour_open', pd.Series(0., index=df.index))
        ).fillna(0)

        # MTMB
        df['mtmb'] = np.tanh(
            1000.0
            * (df.get('mtf_trend_1h', pd.Series(0., index=df.index))
               - df.get('mtf_trend_4h', pd.Series(0., index=df.index)))
            * df.get('trade_intensity', pd.Series(0., index=df.index))
        ).fillna(0)

        # FCSZ
        _sqz    = df.get('squeeze_power', pd.Series(0., index=df.index))
        _sqz_z  = (_sqz - _sqz.rolling(_ROLL, min_periods=1).mean()) \
                   / (_sqz.rolling(_ROLL, min_periods=1).std().replace(0, 1e-8))
        _vol_z  = df.get('volatility_z', pd.Series(0., index=df.index))
        _foc_arg = (
            (df.get('funding_roc_12', pd.Series(0., index=df.index))
             + 0.5 * df.get('funding_roc_48', pd.Series(0., index=df.index)))
            / (_vol_z.abs() + 1e-8)
        )
        df['fcsz'] = (
            pd.Series(_scipy_norm.cdf(_foc_arg.values), index=df.index) * np.tanh(_sqz_z)
        ).fillna(0)

        return df


class RegimeEngine:
    """레짐 라벨 벡터화 계산 (bull/bear/chop/whipsaw/normal)

    전제: df에 mtf_trend_1h 컬럼이 이미 있어야 함 (호출 전 계산 필요).
    """
    COLS = ['regime_bull', 'regime_bear', 'regime_chop', 'regime_whipsaw', 'regime_normal']

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        diff_abs_sum = df['close'].diff().abs().rolling(24).sum()
        net_change   = df['close'] - df['close'].shift(24)
        er           = (net_change.abs() / diff_abs_sum).fillna(0)

        raw_vol      = df['close'].pct_change().rolling(24).std().fillna(0)
        # bfill 제거: 롤링 초반 NaN을 미래 변동성으로 채우는 룩어헤드 편향 방지
        vol_mean_24h = raw_vol.rolling(288).mean().ffill().fillna(0)
        vol_std_24h  = raw_vol.rolling(288).std().ffill().fillna(0) + 1e-8
        vol_z        = (raw_vol - vol_mean_24h) / vol_std_24h
        mtf_1h_trend = df['mtf_trend_1h'].fillna(0.0)

        for col in self.COLS:
            df[col] = 0.0

        bull_idx    = (er >= 0.20) & (net_change > 0) & (mtf_1h_trend > 0)
        bear_idx    = (er >= 0.20) & (net_change < 0) & (mtf_1h_trend < 0)
        chop_idx    = ~(bull_idx | bear_idx) & (vol_z < -0.5)
        whipsaw_idx = ~(bull_idx | bear_idx) & (vol_z >  0.5)

        df.loc[bull_idx,    'regime_bull']    = 1.0
        df.loc[bear_idx,    'regime_bear']    = 1.0
        df.loc[chop_idx,    'regime_chop']    = 1.0
        df.loc[whipsaw_idx, 'regime_whipsaw'] = 1.0
        df.loc[~(chop_idx | whipsaw_idx | bull_idx | bear_idx), 'regime_normal'] = 1.0

        return df


# ═══════════════════════════════════════════════════════════════════════════
#  [3] 변동성 모델 전략 클래스 (GARCH / OU / Jump Diffusion / EVT)
# ═══════════════════════════════════════════════════════════════════════════

class GARCHVolatilityRegime(BaseStrategy):
    """GARCH(1,1) 조건부 변동성 레짐 기반 모멘텀/평균회귀 스위칭

    - 고변동성 레짐 (garch_vol_z > 1.5): 오더플로우 모멘텀 추종
    - 저변동성 레짐 (garch_vol_z < -0.5): 단기 수익률 평균회귀
    """
    def __init__(self):
        super().__init__("GARCH_VolRegime")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        try:
            garch_z    = float(row.get('garch_vol_z', 0.0))
            net_taker  = float(row.get('net_taker_ratio', 0.0))
            log_ret    = float(row.get('log_return', 0.0))

            if garch_z > 1.5:
                # 고변동성: 오더플로우 방향으로 모멘텀
                return float(np.clip(np.sign(net_taker) * min(garch_z / 3.0, 1.0), -1, 1))
            elif garch_z < -0.5:
                # 저변동성: 수익률 평균회귀
                return float(np.clip(-np.sign(log_ret) * 0.4, -1, 1))
            return 0.0
        except (AttributeError, TypeError, ValueError, KeyError):
            return 0.0


class OUMeanReversionHunter(BaseStrategy):
    """OU(Ornstein-Uhlenbeck) 과정 기반 펀딩비 평균회귀 신호

    - 펀딩비가 OU 평균에서 |z| > 1.5σ 이상 이탈 시 반전 베팅
    - 평균회귀 속도(반감기)가 짧을수록 신호 강도 증폭
    """
    def __init__(self):
        super().__init__("OU_MeanReversion")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        try:
            ou_z       = float(row.get('ou_funding_z', 0.0))
            ou_halflife = float(row.get('ou_halflife', 0.5))

            if abs(ou_z) > 1.5:
                # 반감기가 짧을수록(빠른 회귀) 신호 강도 증폭 (0.5~1.5배)
                speed_factor = 1.0 + max(0.5 - ou_halflife, 0.0)
                return float(np.clip(
                    -np.sign(ou_z) * min(abs(ou_z) / 3.0, 1.0) * speed_factor,
                    -1, 1,
                ))
            return 0.0
        except (AttributeError, TypeError, ValueError, KeyError):
            return 0.0


class JumpReboundHunter(BaseStrategy):
    """점프 확산 모델 기반 청산 캐스케이드/점프 후 반전 신호

    - 바이파워 분산 비율로 점프 감지 (jump_flag)
    - 점프 크기(jump_z) + GARCH vol 급등 동반 시 반전 베팅
    - 크립토 청산 캐스케이드의 오버슈팅 회수 노림
    """
    def __init__(self):
        super().__init__("Jump_Rebound")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        try:
            jump_flag = float(row.get('jump_flag', 0.0))
            jump_z    = float(row.get('jump_z', 0.0))
            garch_z   = float(row.get('garch_vol_z', 0.0))

            if jump_flag > 0.5 and abs(jump_z) > 2.0 and garch_z > 0.5:
                rebound_strength = min(abs(jump_z) / 5.0, 0.8)
                return float(np.clip(-np.sign(jump_z) * rebound_strength, -1, 1))
            return 0.0
        except (AttributeError, TypeError, ValueError, KeyError):
            return 0.0


class EVTTailRiskSentinel(BaseStrategy):
    """GPD(Generalized Pareto Distribution) 기반 극단값 이론 꼬리 리스크 감지

    - 99th percentile 임계값 초과 시 극단 사건으로 분류
    - EVT에서 꼬리 사건 직후 오버슈팅 반전 확률이 높음을 이용
    """
    def __init__(self):
        super().__init__("EVT_TailRisk")

    def generate_signal(self, row, df=None, **kwargs) -> float:
        try:
            tail_flag  = float(row.get('evt_tail_flag', 0.0))
            evt_excess = float(row.get('evt_excess_z', 0.0))
            log_ret    = float(row.get('log_return', 0.0))

            if tail_flag > 0.5 and evt_excess > 0.5:
                reversal_strength = min(evt_excess / 5.0, 0.7)
                return float(np.clip(-np.sign(log_ret) * reversal_strength, -1, 1))
            return 0.0
        except (AttributeError, TypeError, ValueError, KeyError):
            return 0.0


class VolatilityModelEngine:
    """GARCH(1,1) + OU 과정 + 점프 감지 + EVT 배치 피처 계산 엔진

    생성 컬럼:
      garch_vol      : GARCH(1,1) 조건부 변동성 (σ_t)
      garch_vol_z    : 롤링 288봉 기준 정규화 (-3~3)
      ou_funding_z   : OU 조정 펀딩비 z-점수 (-3~3)
      ou_halflife    : OU 평균회귀 반감기 (정규화 0~1, 288봉 기준)
      jump_flag      : MAD 4σ 기반 점프 여부 (0/1, ~1-2% 빈도)
      jump_z         : 로버스트 z-점수 (부호 포함, -10~10)
      evt_tail_flag  : EVT 97th percentile 초과 여부 (0/1, ~3% 빈도)
      evt_excess_z   : 임계값 초과 크기 정규화 (0~5)
    """
    COLS = [
        'garch_vol', 'garch_vol_z',
        'ou_funding_z', 'ou_halflife',
        'jump_flag', 'jump_z',
        'evt_tail_flag', 'evt_excess_z',
    ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        self._compute_garch(df)
        self._compute_ou(df)
        self._compute_jump(df)
        self._compute_evt(df)
        return df

    def _compute_garch(self, df: pd.DataFrame) -> None:
        """GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
        파라미터: α=0.10, β=0.85 (크립토 표준값, α+β=0.95 < 1)
        """
        _rets = df.get('log_return', pd.Series(0., index=df.index)).values.astype(np.float64)
        N = len(_rets)
        _eps2 = _rets ** 2
        _init_var = float(np.nanmean(_eps2[:min(288, N)])) or 1e-8
        _alpha, _beta = 0.10, 0.85
        _omega = _init_var * (1.0 - _alpha - _beta)

        _sigma2 = np.empty(N, dtype=np.float64)
        _sigma2[0] = _init_var
        for _t in range(1, N):
            _sigma2[_t] = _omega + _alpha * _eps2[_t - 1] + _beta * _sigma2[_t - 1]

        _garch_s = pd.Series(np.sqrt(_sigma2), index=df.index)
        _mean = _garch_s.rolling(288, min_periods=1).mean()
        _std  = _garch_s.rolling(288, min_periods=1).std().replace(0, 1e-8)
        df['garch_vol']   = _garch_s.astype(np.float32)
        df['garch_vol_z'] = ((_garch_s - _mean) / _std).clip(-3, 3).fillna(0).astype(np.float32)

    def _compute_ou(self, df: pd.DataFrame) -> None:
        """OU z-점수: funding rate 레벨의 롤링 z-점수
        OU 반감기: funding_roc_12 (연속 신호) 의 AR(1) 자기상관계수에서 추정
          halflife = -ln(2) / ln(ρ₁)  (ρ₁ = lag-1 자기상관)

        [수정 이유] last_funding_rate는 8시간 주기 계단 함수 → 5분봉 Δf≈0 →
          OLS-based beta≈0 → theta≈0 → halflife 항상 1.0 상수 문제 해결
          funding_roc_12는 연속 신호로 의미 있는 AR(1) 추정 가능
        """
        _f        = df.get('last_funding_rate', pd.Series(0., index=df.index))
        _mu_ou    = _f.rolling(288, min_periods=1).mean().fillna(0)
        _sigma_ou = _f.rolling(288, min_periods=1).std().replace(0, 1e-8)
        df['ou_funding_z'] = ((_f - _mu_ou) / _sigma_ou).clip(-3, 3).fillna(0).astype(np.float32)

        # 반감기: funding_roc_12의 AR(1) 자기상관 기반 추정 (5일 창)
        _roc     = df.get('funding_roc_12', _f)
        _roc_lag = _roc.shift(1)
        _ROLL_AR = 1440  # 5일

        # lag-1 자기상관 ρ₁ = Cov(X_t, X_{t-1}) / Var(X_{t-1})
        _m      = _roc.rolling(_ROLL_AR, min_periods=60).mean()
        _m_lag  = _roc_lag.rolling(_ROLL_AR, min_periods=60).mean()
        _cov_ar = ((_roc - _m) * (_roc_lag - _m_lag)).rolling(_ROLL_AR, min_periods=60).mean()
        _var_ar = _roc_lag.rolling(_ROLL_AR, min_periods=60).var().replace(0, 1e-12)
        _rho1   = (_cov_ar / _var_ar).clip(0.001, 0.9999)

        # halflife = -ln(2)/ln(ρ₁), 1~1440봉 범위 → 0~1 정규화
        _halflife = (-np.log(2) / np.log(_rho1 + 1e-8)).clip(1, _ROLL_AR)
        df['ou_halflife'] = (_halflife / _ROLL_AR).fillna(0.5).astype(np.float32)

    def _compute_jump(self, df: pd.DataFrame) -> None:
        """MAD(Median Absolute Deviation) 기반 로버스트 점프 감지
        robust_σ = MAD * 1.4826  (정규분포에서 σ와 동등)
        jump_flag: |r| > median + 4 * robust_σ  (4σ 이상 이탈)
        jump_z   : 부호 포함 로버스트 z-점수 (-10~10)

        [수정 이유] 바이파워분산은 n=12 창에서 8.6σ 이상 점프만 감지 →
          5분봉 ETH에서 사실상 항상 0. MAD 기반 4σ 기준은 ~1-2% 빈도로
          의미 있는 청산 캐스케이드/플래시 크래시를 포착함
        """
        _r      = df.get('log_return', pd.Series(0., index=df.index))
        _r_abs  = _r.abs()
        _ROLL_J = 288  # 1일 창 (5분봉 기준)

        _median      = _r_abs.rolling(_ROLL_J, min_periods=20).median()
        _mad         = (_r_abs - _median).abs().rolling(_ROLL_J, min_periods=20).median()
        _robust_sig  = (_mad * 1.4826).replace(0, 1e-8)

        df['jump_flag'] = (_r_abs > _median + 4.0 * _robust_sig).astype(np.float32)
        df['jump_z']    = (_r / (_robust_sig + 1e-8)).clip(-10, 10).fillna(0).astype(np.float32)

    def _compute_evt(self, df: pd.DataFrame) -> None:
        """EVT(Peaks Over Threshold): 롤링 97th percentile 임계값 초과 감지
        임계값: 576봉(2일) 롤링 97th percentile (최소 50봉 필요)
        evt_tail_flag: ~3% 빈도로 발생 (99th→97th 완화)
        evt_excess_z : 임계값 초과 크기 정규화 (0~5)

        [수정 이유] 99th percentile + 1440봉 창은 너무 희소하여 사실상 0.
          97th percentile + 576봉 창으로 완화 → 실질적인 신호 밀도 확보
        """
        _r_abs  = df.get('log_return', pd.Series(0., index=df.index)).abs()
        _ROLL_E = 576  # 2일 창

        _thresh = (
            _r_abs.rolling(_ROLL_E, min_periods=50).quantile(0.97)
            .ffill()
            .fillna(_r_abs.expanding(min_periods=10).quantile(0.97).fillna(0))
        )
        # 임계값이 0이 되지 않도록 최솟값 보장
        _thresh = _thresh.clip(lower=_r_abs.rolling(_ROLL_E, min_periods=10).mean().fillna(0) * 0.5 + 1e-8)

        df['evt_tail_flag'] = (_r_abs > _thresh).astype(np.float32)
        df['evt_excess_z']  = ((_r_abs - _thresh) / (_thresh + 1e-8)).clip(0, 5).fillna(0).astype(np.float32)