"""
Multi-Timeframe (MTF) Processor
Look-ahead Bias(미래 참조)를 완벽하게 차단한 상위 프레임 데이터 병합기
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MTFProcessor:
    def __init__(self, df_3m):
        """
        Args:
            df_3m: 3분봉 원본 데이터 (index는 datetime이어야 함)
        """
        self.df_3m = df_3m.copy()
        if not isinstance(self.df_3m.index, pd.DatetimeIndex):
            raise ValueError("데이터프레임의 인덱스는 DatetimeIndex여야 합니다.")

    def add_mtf_features(self):
        """15분봉, 1시간봉 지표를 안전하게 병합"""
        logger.info("⏳ MTF 피처 생성 중... (Look-ahead Bias 방지 적용)")
        
        # 1. 15분봉 데이터 생성 및 지표 계산
        df_15m = self._resample_data('15min')
        df_15m = self._calculate_higher_indicators(df_15m, suffix='_15m')
        
        # [핵심] 미래 참조 방지: 15분봉 지표를 한 칸 밑으로 내림 (Shift)
        # 10:00~10:15 캔들의 지표는 10:15분에 확정되므로, 10:15분 이후 3분봉부터 보여야 함
        df_15m_shifted = df_15m.shift(1)
        
        # 2. 1시간봉 데이터 생성 및 지표 계산
        df_1h = self._resample_data('1h')
        df_1h = self._calculate_higher_indicators(df_1h, suffix='_1h')
        
        # [핵심] 미래 참조 방지: 1시간봉 지표를 한 칸 밑으로 내림
        df_1h_shifted = df_1h.shift(1)
        
        # 3. 3분봉에 병합 (ffill: 직전의 유효한 상위 봉 값을 계속 가져옴)
        # merge_asof 등을 쓸 수도 있지만, reindex + ffill이 가장 직관적이고 안전함
        
        # 15분봉 병합
        merged_15m = df_15m_shifted.reindex(self.df_3m.index, method='ffill')
        
        # 1시간봉 병합
        merged_1h = df_1h_shifted.reindex(self.df_3m.index, method='ffill')
        
        # 원본에 추가
        result_df = self.df_3m.copy()
        
        # 15분봉 피처 추가
        for col in merged_15m.columns:
            result_df[col] = merged_15m[col]
            
        # 1시간봉 피처 추가
        for col in merged_1h.columns:
            result_df[col] = merged_1h[col]
            
        # NaN 처리 (앞부분 데이터 부족)
        result_df = result_df.fillna(0)
        
        logger.info(f"✅ MTF 피처 추가 완료: 15분봉 {len(merged_15m.columns)}개, 1시간봉 {len(merged_1h.columns)}개")
        
        return result_df

    def _resample_data(self, interval):
        """OHLCV 리샘플링"""
        logic = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        return self.df_3m.resample(interval).agg(logic)

    def _calculate_higher_indicators(self, df, suffix):
        """[수정] 초기 NaN을 중립값으로 채움"""
        # 1. RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # [수정] RSI NaN -> 50 (중립)
        rsi = rsi.fillna(50.0)
        
        # 2. Trend
        ema_long = df['close'].ewm(span=20, adjust=False).mean()
        trend = np.where(df['close'] > ema_long, 1.0, -1.0)
        # EMA 계산 전 초기값은 Trend 0으로 가정
        trend[:20] = 0.0 
        
        result = pd.DataFrame(index=df.index)
        result[f'rsi{suffix}'] = rsi
        result[f'trend{suffix}'] = trend
        
        return result
