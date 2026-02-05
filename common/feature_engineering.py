import pandas as pd
import numpy as np
import pandas_ta as ta

# 외부에서 참조할 피처 목록 상수 정의
ULTIMATE_FEATURE_COLS = [
    # Group A: Smart Money & Sentiment (Alpha)
    'whale_retail_ratio', 'whale_conviction', 'smart_money_flow', 'funding_pressure', 'squeeze_power',
    # Group B: Order Flow
    'net_taker_ratio', 'taker_acceleration', 'trade_intensity',
    # Group C: Technical
    'log_return', 'volatility_z', 'rsi', 'macd_hist', 'bb_width', 'bb_width_z', 'vwap_dist', 'hma_slope', 'wick_ratio',
    # Group D: Market Structure
    'btc_corr_60', 'eth_btc_ratio_change', 'fvg_dist', 'chop_index'
]

class FeatureEngineer:
    def __init__(self):
        self.windows = {
            'short': 5,      # 15 min
            'medium': 20,    # 1 hour
            'long': 480,     # 24 hours
            'volatility': 20
        }

    def process(self, eth_df, btc_df):
        """
        메인 파이프라인: 데이터 병합 및 피처 생성
        """
        # 1. 복사본 생성 (원본 보존)
        eth = eth_df.copy()
        btc = btc_df.copy()

        # 2. 데이터 병합 (Timestamp 기준)
        df = self._merge_data(eth, btc)
        
        # 3. 피처 생성 그룹별 실행
        df = self._create_alpha_features(df)    # Group A
        df = self._create_order_flow(df)        # Group B
        df = self._create_technical(df)         # Group C
        df = self._create_market_structure(df)  # Group D
        
        # 4. 결측치 및 인피니티 처리
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        return df

    def _merge_data(self, eth, btc):
        eth['timestamp'] = pd.to_datetime(eth['timestamp'])
        btc['timestamp'] = pd.to_datetime(btc['timestamp'])
        
        # BTC 데이터 접미사 추가
        btc_renamed = btc[['timestamp', 'close', 'volume', 'quote_volume']].rename(
            columns={'close': 'close_btc', 'volume': 'volume_btc', 'quote_volume': 'quote_volume_btc'}
        )
        
        # Nearest 병합
        merged = pd.merge_asof(
            eth.sort_values('timestamp'), 
            btc_renamed.sort_values('timestamp'), 
            on='timestamp', 
            direction='nearest'
        )
        return merged

    def _create_alpha_features(self, df):
        # 1. Whale Retail Ratio (고래/개미 비율)
        df['whale_retail_ratio'] = df['sum_toptrader_long_short_ratio'] / df['count_long_short_ratio'].replace(0, 1)
        
        # 2. Whale Conviction (고래 확신 - 변화율)
        df['whale_conviction'] = df['sum_toptrader_long_short_ratio'].pct_change()
        
        # 3. Smart Money Flow (OI 가치 변화)
        df['smart_money_flow'] = df['sum_open_interest_value'].diff()
        
        # 4. Funding Pressure (펀딩비 누적 압력 - 24시간 Rolling Sum)
        df['funding_pressure'] = df['last_funding_rate'].rolling(window=self.windows['long']).sum()
        
        # 5. Squeeze Power
        df['squeeze_power'] = df['sum_open_interest_value'] * df['last_funding_rate']
        
        return df

    def _create_order_flow(self, df):
        # 1. Net Taker Ratio
        taker_sell_quote = df['quote_volume'] - df['taker_buy_quote']
        net_flow = df['taker_buy_quote'] - taker_sell_quote
        df['net_taker_ratio'] = net_flow / df['quote_volume'].replace(0, 1)
        
        # 2. Taker Acceleration
        short_ma = df['net_taker_ratio'].rolling(window=2).mean()
        long_ma = df['net_taker_ratio'].rolling(window=20).mean()
        df['taker_acceleration'] = short_ma - long_ma
        
        # 3. Trade Intensity
        df['trade_intensity'] = df['trades'] / df['volume'].replace(0, 1)
        
        return df

    def _create_technical(self, df):
        # 1. Log Return
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. Volatility Z-Score
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        atr_mean = atr.rolling(window=self.windows['long']).mean()
        atr_std = atr.rolling(window=self.windows['long']).std()
        df['volatility_z'] = (atr - atr_mean) / atr_std
        
        # 3. RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # 4. MACD Hist (안전한 컬럼 찾기)
        macd = ta.macd(df['close'])
        # 컬럼명 중 'MACDh'가 포함된 컬럼 자동 찾기
        hist_col = [c for c in macd.columns if 'MACDh' in c][0]
        df['macd_hist'] = macd[hist_col]

        # 5. BB Width (안전한 컬럼 찾기)
        bb = ta.bbands(df['close'], length=20, std=2)
        # BBU(상단), BBL(하단), BBM(중심)으로 시작하는 컬럼 자동 매핑
        upper_col = [c for c in bb.columns if c.startswith('BBU')][0]
        lower_col = [c for c in bb.columns if c.startswith('BBL')][0]
        mid_col = [c for c in bb.columns if c.startswith('BBM')][0]
        
        # (상단 - 하단) / 중심
        df['bb_width'] = (bb[upper_col] - bb[lower_col]) / (bb[mid_col] + 1e-8)
        # BB Width Z-Score (상대적 수치: 절대값에 휘둘리지 않도록)
        bbw_mean = df['bb_width'].rolling(window=100, min_periods=1).mean()
        bbw_std = df['bb_width'].rolling(window=100, min_periods=1).std().replace(0, 1e-8)
        df['bb_width_z'] = (df['bb_width'] - bbw_mean) / bbw_std
        
        # 6. VWAP Dist (Rolling Mean Anchor)
        vwap_approx = df['close'].rolling(window=self.windows['long']).mean()
        df['vwap_dist'] = (df['close'] - vwap_approx) / vwap_approx
        
        # 7. HMA Slope
        hma = ta.hma(df['close'], length=20)
        df['hma_slope'] = hma.diff()
        
        # 8. Wick Ratio
        body_size = np.abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        df['wick_ratio'] = np.where(total_range == 0, 0, (total_range - body_size) / total_range)
        
        return df

    def _create_market_structure(self, df):
        # 1. BTC Correlation
        df['btc_corr_60'] = df['close'].rolling(window=20).corr(df['close_btc'])
        
        # 2. ETH/BTC Ratio Change
        eth_btc_ratio = df['close'] / df['close_btc']
        df['eth_btc_ratio_change'] = eth_btc_ratio.pct_change()
        
        # 3. FVG Dist (Placeholder)
        df['fvg_dist'] = 0.0 
        
        # 4. Chop Index
        df['chop_index'] = ta.chop(df['high'], df['low'], df['close'], length=14)
        
        return df