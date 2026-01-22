"""
Advanced Feature Engineering Module
RL 모델을 위한 최적의 25개 피처 생성 (가격, 거래량, 패턴, 매크로)
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, eth_data, btc_data=None):
        """
        Args:
            eth_data: ETH/USDT DataFrame (Close, Volume, Taker_Buy 등 포함)
            btc_data: BTC/USDT DataFrame (선택적, 매크로 피처용)
        """
        self.df = eth_data.copy()
        self.btc_df = btc_data.copy() if btc_data is not None else None
        
        # 인덱스 정렬
        if self.btc_df is not None:
            common_index = self.df.index.intersection(self.btc_df.index)
            self.df = self.df.loc[common_index]
            self.btc_df = self.btc_df.loc[common_index]

    def generate_features(self):
        """모든 피처 생성 및 병합"""
        try:
            logger.info("🚀 고급 피처 엔지니어링 시작...")
            
            # 1. 가격 & 변동성 피처 (9개)
            self._add_price_volatility_features()
            
            # 2. 거래량 & 오더플로우 피처 (6개)
            self._add_volume_flow_features()
            
            # 3. 패턴 & 유동성 피처 (5개)
            self._add_pattern_liquidity_features()
            
            # 4. 시장 상관관계 피처 (5개)
            if self.btc_df is not None:
                self._add_market_correlation_features()
            else:
                logger.warning("⚠️ BTC 데이터가 없어 상관관계 피처를 0으로 채웁니다.")
                self._fill_missing_market_features()

            # NaN 처리 (초기 데이터)
            self.df = self.df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            logger.info(f"✅ 피처 생성 완료: 총 {len(self.df.columns)}개 컬럼")
            return self.df
            
        except Exception as e:
            logger.error(f"피처 생성 중 오류: {e}", exc_info=True)
            return None

    def _add_price_volatility_features(self):
        """A. 가격 변동성 & 모멘텀 (9개)"""
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        
        # 1. Log Return (1봉 수익률) 🔥
        self.df['log_return'] = np.log(close / close.shift(1)).fillna(0)
        
        # 2. Rolling Return (6봉, 약 15~20분 추세)
        self.df['roll_return_6'] = close.pct_change(6).fillna(0)
        
        # 3. ATR Ratio (변동성 확장 비율) 🔥
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        atr = tr.rolling(14).mean()
        self.df['atr_ratio'] = (atr / close).fillna(0)
        
        # 4. BB Width (스퀴즈 감지) 🔥 & 5. BB Position
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        self.df['bb_width'] = ((upper - lower) / ma20).fillna(0)
        self.df['bb_pos'] = ((close - lower) / (upper - lower + 1e-8)).fillna(0.5)
        
        # 6. RSI (14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        # 7. MACD Hist (추세 반전)
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        self.df['macd_hist'] = macd - signal
        
        # 8. HMA Ratio (추세 괴리율)
        # HMA 간단 구현 (WMA 기반)
        def wma(series, period):
            weights = np.arange(1, period + 1)
            return series.rolling(period).apply(
                lambda x: np.dot(x.values, weights) / weights.sum() if len(x) == period else np.nan, 
                raw=False
            )
        
        try:
            half_length = int(14 / 2)
            sqrt_length = int(np.sqrt(14))
            wmaf = wma(close, half_length)
            wmas = wma(close, 14)
            hma = wma(2 * wmaf - wmas, sqrt_length)
            self.df['hma_ratio'] = ((close - hma) / close).fillna(0)
        except Exception as e:
            logger.warning(f"HMA 계산 실패: {e}, 0으로 채웁니다.")
            self.df['hma_ratio'] = 0.0
        
        # 9. CCI (고빈도 매매용)
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(14).mean()
        mad_tp = (tp - sma_tp).abs().rolling(14).mean()
        self.df['cci'] = (tp - sma_tp) / (0.015 * mad_tp + 1e-8)

    def _add_volume_flow_features(self):
        """B. 거래량 & 오더플로우 (6개)"""
        vol = self.df['volume']
        close = self.df['close']
        
        # 10. Relative Volume (RVol) 🔥
        self.df['rvol'] = vol / (vol.rolling(20).mean() + 1e-8)
        
        # 11. Taker Buy Ratio (공격적 매수세)
        taker_buy = self.df.get('taker_buy_base_asset_volume', self.df.get('taker_buy_base', vol * 0.5))
        self.df['taker_ratio'] = taker_buy / (vol + 1e-8)
        
        # 12. CVD Change (순매수 거래량 변화율) 🔥
        taker_sell = vol - taker_buy
        delta = taker_buy - taker_sell
        cvd = delta.cumsum()
        self.df['cvd_change'] = cvd.diff(3).fillna(0)  # 최근 3봉간 수급 변화
        
        # 13. MFI (자금 흐름)
        tp = (self.df['high'] + self.df['low'] + close) / 3
        rmf = tp * vol
        pmf = np.where(tp > tp.shift(1), rmf, 0)
        nmf = np.where(tp < tp.shift(1), rmf, 0)
        mfr = pd.Series(pmf).rolling(14).sum() / (pd.Series(nmf).rolling(14).sum() + 1e-8)
        self.df['mfi'] = 100 - 100 / (1 + mfr)
        
        # 14. CMF (매집/분산)
        mf_mult = ((close - self.df['low']) - (self.df['high'] - close)) / (self.df['high'] - self.df['low'] + 1e-8)
        self.df['cmf'] = (mf_mult * vol).rolling(20).sum() / (vol.rolling(20).sum() + 1e-8)
        
        # 15. VWAP Distance
        cum_vol = vol.cumsum()
        cum_pv = (tp * vol).cumsum()
        vwap = cum_pv / (cum_vol + 1e-8)
        self.df['vwap_dist'] = (close - vwap) / vwap

    def _add_pattern_liquidity_features(self):
        """C. 패턴 & 유동성 스윕 (5개)"""
        open_p = self.df['open']
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        body_top = np.maximum(open_p, close)
        body_bottom = np.minimum(open_p, close)
        range_len = high - low + 1e-8
        
        # 16. High Wick Ratio (윗꼬리) 🔥
        self.df['wick_upper'] = (high - body_top) / range_len
        
        # 17. Low Wick Ratio (아랫꼬리)
        self.df['wick_lower'] = (body_bottom - low) / range_len
        
        # 18. Range Position (최근 60봉 박스권 위치)
        roll_high = high.rolling(60).max()
        roll_low = low.rolling(60).min()
        self.df['range_pos'] = (close - roll_low) / (roll_high - roll_low + 1e-8)
        
        # 19. Swing Break Flag (스윙 하이/로우 돌파 여부)
        # 직전 10봉 고점 돌파시 1, 저점 이탈시 -1
        prev_high_10 = high.shift(1).rolling(10).max()
        prev_low_10 = low.shift(1).rolling(10).min()
        self.df['swing_break'] = np.where(close > prev_high_10, 1.0, 
                                          np.where(close < prev_low_10, -1.0, 0.0))
        
        # 20. Chop Index (추세 vs 횡보)
        atr_sum = range_len.rolling(14).sum()
        h_l_diff = high.rolling(14).max() - low.rolling(14).min()
        self.df['chop'] = 100 * np.log10(atr_sum / (h_l_diff + 1e-8)) / np.log10(14)

    def _add_market_correlation_features(self):
        """D. 시장 상관관계 (5개)"""
        btc_close = self.btc_df['close']
        eth_close = self.df['close']
        
        # 21. BTC Log Return
        self.df['btc_return'] = np.log(btc_close / btc_close.shift(1)).fillna(0)
        
        # 22. BTC RSI
        delta = btc_close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        self.df['btc_rsi'] = 100 - (100 / (1 + rs))
        
        # 23. BTC-ETH Correlation (30봉) 🔥
        self.df['btc_corr'] = eth_close.rolling(30).corr(btc_close).fillna(0)
        
        # 24. BTC Volatility (ATR Ratio)
        btc_tr = np.maximum(self.btc_df['high'] - self.btc_df['low'], 
                            np.abs(self.btc_df['high'] - btc_close.shift(1)))
        self.df['btc_vol'] = (btc_tr.rolling(14).mean() / btc_close).fillna(0)
        
        # 25. ETH/BTC Ratio Change
        ratio = eth_close / btc_close
        self.df['eth_btc_ratio'] = ratio.pct_change(6).fillna(0)

    def _fill_missing_market_features(self):
        """BTC 데이터가 없을 때 상관관계 피처를 0으로 채움"""
        for col in ['btc_return', 'btc_rsi', 'btc_corr', 'btc_vol', 'eth_btc_ratio']:
            self.df[col] = 0.0
