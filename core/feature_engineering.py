import pandas as pd
import numpy as np
import pandas_ta as ta

# ──────────────────────────────────────────────────────────────
# 외부에서 참조할 피처 목록 상수 정의
# ──────────────────────────────────────────────────────────────
ULTIMATE_FEATURE_COLS = [
    # Group A: Smart Money & Sentiment (Alpha)
    'whale_retail_ratio', 'whale_conviction', 'smart_money_flow',
    'funding_pressure', 'squeeze_power', 'oi_change_rate',

    # Group B: Order Flow
    'net_taker_ratio', 'taker_acceleration', 'trade_intensity',
    'big_trade_ratio',  # volume_imbalance 제거 (net_taker_ratio와 중복)

    # Group C: Technical
    'volatility_z', 'rsi', 'macd_hist',
    'bb_width', 'bb_width_z', 'vwap_dist', 'hma_slope', 'wick_ratio',
    'garman_klass_vol', 'realized_vol_ratio',

    # Group D: Market Structure
    'btc_corr_60', 'eth_btc_ratio_change', 'fvg_dist', 'chop_index',

    # Group E: Temporal (NEW)
    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
    'session_asia', 'session_europe', 'session_us',
    'is_hour_open',

    # Group F: Strategy Meta (TFT 전용 — 개별 시그널은 RL Agent에서 사용)
    'strategy_consensus', 'strategy_conviction', 'strategy_conflict',
    'momentum_regime', 'reversion_regime',
]


class FeatureEngineer:
    """
    5분봉 ETH 데이 트레이딩용 피처 엔지니어링 파이프라인.

    변경 이력:
    - v2: FVG 실제 구현, VWAP 정확한 계산, temporal 피처 추가,
          변동성 레짐 피처 추가, 결측치 처리 개선
    """

    def __init__(self, candle_minutes: int = 5):
        self.candle_minutes = candle_minutes
        self.windows = {
            'short': 5,       # 25 min
            'medium': 20,     # 100 min (~1.7h)
            'long': 288,      # 288 × 5min = 24h (수정: 480은 3min봉 기준이었음)
            'volatility': 20,
            'corr': 60,       # 60 × 5min = 5h
        }

    # ================================================================
    # PUBLIC
    # ================================================================
    def process(self, eth_df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
        """
        메인 파이프라인: 데이터 병합 및 피처 생성

        Returns:
            피처가 추가된 DataFrame (결측치 처리 완료)
        """
        eth = eth_df.copy()
        btc = btc_df.copy()

        df = self._merge_data(eth, btc)

        df = self._create_alpha_features(df)       # Group A
        df = self._create_order_flow(df)            # Group B
        df = self._create_technical(df)             # Group C
        df = self._create_market_structure(df)      # Group D
        df = self._create_temporal_features(df)     # Group E (NEW)
        df = self._create_strategy_meta(df)         # Group F (NEW)
        
        df['target_cumret_6'] = (df['close'].shift(-6) / df['close'] - 1)

        df = self._handle_missing(df)

        return df

    # ================================================================
    # DATA MERGE
    # ================================================================
    def _merge_data(self, eth: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
        eth['timestamp'] = pd.to_datetime(eth['timestamp'])
        btc['timestamp'] = pd.to_datetime(btc['timestamp'])

        # BTC 컬럼명 정규화: 이미 rename된 경우와 아닌 경우 모두 처리
        btc_cols_needed = {'close': 'close_btc', 'volume': 'volume_btc', 'quote_volume': 'quote_volume_btc'}
        
        # 이미 rename된 컬럼이 있는지 확인
        already_renamed = 'close_btc' in btc.columns and 'close' not in btc.columns
        
        if already_renamed:
            btc_renamed = btc[['timestamp', 'close_btc', 'volume_btc', 'quote_volume_btc']].copy()
        else:
            btc_renamed = btc[['timestamp', 'close', 'volume', 'quote_volume']].rename(
                columns=btc_cols_needed
            )

        merged = pd.merge_asof(
            eth.sort_values('timestamp'),
            btc_renamed.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
        )
        return merged

    
    # ================================================================
    # GROUP A: Smart Money & Sentiment
    # ================================================================
    def _create_alpha_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Whale / Retail Ratio
        df['whale_retail_ratio'] = (
            df['sum_toptrader_long_short_ratio']
            / df['count_long_short_ratio'].replace(0, np.nan)
        )

        # 2. Whale Conviction — diff가 pct_change보다 안전 (0 근처 극단값 방지)
        df['whale_conviction'] = df['sum_toptrader_long_short_ratio'].diff()

        # 3. Smart Money Flow (OI 가치 변화)
        df['smart_money_flow'] = df['sum_open_interest_value'].diff()

        # 4. Funding Pressure (24h 누적)
        df['funding_pressure'] = (
            df['last_funding_rate']
            .rolling(window=self.windows['long'], min_periods=1)
            .sum()
        )

        # 5. Squeeze Power
        df['squeeze_power'] = df['sum_open_interest_value'] * df['last_funding_rate']

        # 6. OI Change Rate (NEW) — OI 변화율, 급변 감지
        df['oi_change_rate'] = df['sum_open_interest_value'].pct_change().clip(-1, 1)

        return df

    # ================================================================
    # GROUP B: Order Flow
    # ================================================================
    def _create_order_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        quote_vol = df['quote_volume'].replace(0, np.nan)
        taker_buy = df['taker_buy_quote']
        taker_sell = df['quote_volume'] - taker_buy

        # 1. Net Taker Ratio
        net_flow = taker_buy - taker_sell
        df['net_taker_ratio'] = net_flow / quote_vol

        # 2. Taker Acceleration (단기 MA - 장기 MA)
        short_ma = df['net_taker_ratio'].rolling(window=2, min_periods=1).mean()
        long_ma = df['net_taker_ratio'].rolling(window=20, min_periods=1).mean()
        df['taker_acceleration'] = short_ma - long_ma

        # 3. Trade Intensity (건당 거래량의 역수 → 소매 거래 활발도)
        df['trade_intensity'] = df['trades'] / df['volume'].replace(0, np.nan)

        # 4. Big Trade Ratio (NEW) — 평균 거래 크기 (고래 활동 proxy)
        # volume_imbalance 제거: net_taker_ratio와 수식이 동일 (분모만 다름)
        avg_trade_size = df['quote_volume'] / df['trades'].replace(0, np.nan)
        avg_trade_rolling = avg_trade_size.rolling(window=self.windows['long'], min_periods=1).mean()
        avg_trade_std = avg_trade_size.rolling(window=self.windows['long'], min_periods=1).std().replace(0, 1e-8)
        df['big_trade_ratio'] = (avg_trade_size - avg_trade_rolling) / avg_trade_std

        return df

    # ================================================================
    # GROUP C: Technical
    # ================================================================
    def _create_technical(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['close']
        high = df['high']
        low = df['low']
        opn = df['open']

        # 1. Log Return
        df['log_return'] = np.log(close / close.shift(1))

        # 2. Volatility Z-Score (ATR 기반)
        atr = ta.atr(high, low, close, length=14)
        win = self.windows['long']
        atr_mean = atr.rolling(window=win, min_periods=1).mean()
        atr_std = atr.rolling(window=win, min_periods=1).std().replace(0, 1e-8)
        df['volatility_z'] = (atr - atr_mean) / atr_std

        # 3. RSI
        df['rsi'] = ta.rsi(close, length=14)

        # 4. MACD Histogram
        macd = ta.macd(close)
        hist_col = [c for c in macd.columns if 'MACDh' in c][0]
        df['macd_hist'] = macd[hist_col]

        # 5. Bollinger Band Width + Z-Score
        bb = ta.bbands(close, length=20, std=2)
        upper_col = [c for c in bb.columns if c.startswith('BBU')][0]
        lower_col = [c for c in bb.columns if c.startswith('BBL')][0]
        mid_col = [c for c in bb.columns if c.startswith('BBM')][0]

        df['bb_width'] = (bb[upper_col] - bb[lower_col]) / (bb[mid_col] + 1e-8)
        bbw_mean = df['bb_width'].rolling(window=100, min_periods=1).mean()
        bbw_std = df['bb_width'].rolling(window=100, min_periods=1).std().replace(0, 1e-8)
        df['bb_width_z'] = (df['bb_width'] - bbw_mean) / bbw_std

        # 6. VWAP Distance — 정확한 VWAP 계산 (FIX)
        df['vwap_dist'] = self._calc_vwap_dist(df)

        # 7. HMA Slope
        hma = ta.hma(close, length=20)
        df['hma_slope'] = hma.diff()

        # 8. Wick Ratio
        body_size = np.abs(close - opn)
        total_range = high - low
        df['wick_ratio'] = np.where(
            total_range == 0, 0, (total_range - body_size) / total_range
        )

        # 9. Garman-Klass Volatility (NEW) — 고빈도에서 close-to-close보다 정확
        df['garman_klass_vol'] = self._garman_klass(high, low, opn, close)

        # 10. Realized Volatility Ratio (NEW) — 단기/장기 변동성 비율
        rv_short = df['log_return'].rolling(window=12, min_periods=1).std()    # 1시간
        rv_long = df['log_return'].rolling(window=self.windows['long'], min_periods=1).std()
        df['realized_vol_ratio'] = rv_short / rv_long.replace(0, 1e-8)

        return df

    def _calc_vwap_dist(self, df: pd.DataFrame) -> pd.Series:
        """
        정확한 VWAP 계산: cumsum(typical_price * volume) / cumsum(volume)
        세션 단위(24h) rolling으로 계산
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        tp_vol = typical_price * df['volume']

        win = self.windows['long']
        cum_tp_vol = tp_vol.rolling(window=win, min_periods=1).sum()
        cum_vol = df['volume'].rolling(window=win, min_periods=1).sum()

        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
        return (df['close'] - vwap) / (vwap + 1e-8)

    @staticmethod
    def _garman_klass(high: pd.Series, low: pd.Series,
                      opn: pd.Series, close: pd.Series,
                      window: int = 20) -> pd.Series:
        """
        Garman-Klass volatility estimator.
        고빈도 데이터에서 일반 close-to-close 변동성보다 5배 효율적.
        """
        # 방어: high < low 또는 close/open이 0인 이상 데이터 처리
        h = high.clip(lower=low)           # high가 low보다 작을 수 없음
        o = opn.replace(0, np.nan)
        c = close.replace(0, np.nan)
        l = low.replace(0, np.nan)

        log_hl = (np.log(h / l)) ** 2
        log_co = (np.log(c / o)) ** 2

        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return (gk.rolling(window=window, min_periods=1).mean().clip(lower=0) ** 0.5)

    # ================================================================
    # GROUP D: Market Structure
    # ================================================================
    def _create_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['close']
        close_btc = df['close_btc']

        # 1. BTC Correlation — 윈도우를 변수명과 일치시킴 (FIX)
        df['btc_corr_60'] = close.rolling(window=self.windows['corr']).corr(close_btc)

        # 2. ETH/BTC Ratio Change
        eth_btc_ratio = close / close_btc.replace(0, np.nan)
        df['eth_btc_ratio_change'] = eth_btc_ratio.pct_change()

        # 3. FVG Distance — 실제 구현 (FIX)
        df['fvg_dist'] = self._calc_fvg_dist(df)

        # 4. Choppiness Index
        df['chop_index'] = ta.chop(df['high'], df['low'], close, length=14)

        return df

    @staticmethod
    def _calc_fvg_dist(df: pd.DataFrame) -> pd.Series:
        """
        Fair Value Gap (FVG) 탐지 및 현재 가격과의 거리 계산.

        Bullish FVG: candle[i-2].high < candle[i].low (갭 상승)
        Bearish FVG: candle[i-2].low > candle[i].high (갭 하락)

        가장 최근 FVG까지의 정규화 거리를 반환.

        TODO: 데이터가 20만 행 이상일 경우 numba @njit 데코레이터 적용 권장
            (현재 10만 행 기준 약 30초~1분 소요)
            from numba import njit 후 내부 루프를 별도 함수로 분리
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        n = len(df)
        fvg_dist = np.zeros(n, dtype=np.float64)

        # 최근 FVG 추적 (최대 50개 캔들 lookback)
        lookback = 50

        for i in range(2, n):
            nearest_gap_dist = 0.0
            min_abs_dist = np.inf

            for j in range(i, max(i - lookback, 1), -1):
                # Bullish FVG: 2캔들 전 high < 현재 캔들 low
                if high[j - 2] < low[j]:
                    gap_mid = (high[j - 2] + low[j]) / 2
                    dist = (close[i] - gap_mid) / (close[i] + 1e-8)
                    if abs(dist) < min_abs_dist:
                        min_abs_dist = abs(dist)
                        nearest_gap_dist = dist

                # Bearish FVG: 2캔들 전 low > 현재 캔들 high
                if low[j - 2] > high[j]:
                    gap_mid = (low[j - 2] + high[j]) / 2
                    dist = (close[i] - gap_mid) / (close[i] + 1e-8)
                    if abs(dist) < min_abs_dist:
                        min_abs_dist = abs(dist)
                        nearest_gap_dist = dist

            fvg_dist[i] = nearest_gap_dist

        return pd.Series(fvg_dist, index=df.index)

    # ================================================================
    # GROUP E: Temporal Features (NEW)
    # ================================================================
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시간 기반 피처.
        - 순환 인코딩 (sin/cos)으로 23시→0시 연속성 보장
        - 거래 세션 구분 (아시아/유럽/미국)
        - 정각 효과 (기관 알고리즘 활동 집중 시간)
        """
        ts = df['timestamp']
        hour = ts.dt.hour
        minute = ts.dt.minute

        # 1. 순환 인코딩 — hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        # 2. 순환 인코딩 — minute
        df['minute_sin'] = np.sin(2 * np.pi * minute / 60)
        df['minute_cos'] = np.cos(2 * np.pi * minute / 60)

        # 3. 거래 세션 (UTC 기준)
        #    아시아: 00:00–08:00 / 유럽: 08:00–16:00 / 미국: 16:00–24:00
        df['session_asia'] = ((hour >= 0) & (hour < 8)).astype(np.float32)
        df['session_europe'] = ((hour >= 8) & (hour < 16)).astype(np.float32)
        df['session_us'] = ((hour >= 16) & (hour < 24)).astype(np.float32)

        # 4. 정각 효과 (매 시 정각 ±5분)
        df['is_hour_open'] = (minute < 5).astype(np.float32)

        return df

    # ================================================================
    # GROUP F: Strategy Meta Features (NEW)
    # ================================================================
    def _create_strategy_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        8개 전략 시그널의 메타 피처 5개 생성 (TFT 입력용).

        개별 시그널(0/1/-1)은 원본 피처와 중복되므로 TFT에 직접 넣지 않고,
        전략 간 합의/충돌/레짐 정보만 추출하여 TFT가 학습하기 어려운
        비선형 메타 정보를 제공한다.

        생성 피처:
            - strategy_consensus:  전체 합의 방향 (-1~+1)
            - strategy_conviction: 합의 강도 (0~1)
            - strategy_conflict:   롱/숏 동시 발생 정도 (0~1, 혼조장 감지)
            - momentum_regime:     추세추종 전략 최근 적중률 (추세장 감지)
            - reversion_regime:    평균회귀 전략 최근 적중률 (횡보장 감지)

        개별 시그널 8개는 RL Agent의 State로 별도 전달.
        """
        # ── 8개 전략 시그널 생성 ──
        signals = self._generate_all_signals(df)

        num_strategies = signals.shape[1]

        # 1. Consensus: 전체 합의 방향 (-1 ~ +1)
        df['strategy_consensus'] = signals.sum(axis=1) / num_strategies

        # 2. Conviction: 시그널 발생 비율 (0 ~ 1)
        df['strategy_conviction'] = signals.abs().sum(axis=1) / num_strategies

        # 3. Conflict: 롱/숏 동시 발생 → 혼조장 (0 ~ 1)
        long_count = (signals > 0).sum(axis=1)
        short_count = (signals < 0).sum(axis=1)
        df['strategy_conflict'] = (2 * long_count * short_count) / (num_strategies ** 2)

        # 4~5. 레짐 감지: 전략 적중률 기반
        actual_dir = np.sign(df['close'].shift(-1) - df['close'])

        # 추세추종 전략 적중률 → momentum_regime
        momentum_strats = ['HMAMomentum', 'NetTakerFlow', 'BTCEthCorr']
        momentum_cols = [c for c in momentum_strats if c in signals.columns]
        if momentum_cols:
            momentum_hits = signals[momentum_cols].eq(actual_dir, axis=0).astype(float)
            df['momentum_regime'] = momentum_hits.mean(axis=1).rolling(
                window=50, min_periods=1
            ).mean()
        else:
            df['momentum_regime'] = 0.5

        # 평균회귀 전략 적중률 → reversion_regime
        reversion_strats = ['VWAPDeviation', 'VolSqueeze', 'OrderblockFVG']
        reversion_cols = [c for c in reversion_strats if c in signals.columns]
        if reversion_cols:
            reversion_hits = signals[reversion_cols].eq(actual_dir, axis=0).astype(float)
            df['reversion_regime'] = reversion_hits.mean(axis=1).rolling(
                window=50, min_periods=1
            ).mean()
        else:
            df['reversion_regime'] = 0.5

        # 마지막 행은 shift(-1) 때문에 NaN → 직전 값으로 채움
        df['momentum_regime'] = df['momentum_regime'].ffill()
        df['reversion_regime'] = df['reversion_regime'].ffill()

        return df

    def _generate_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        8개 Elite 전략의 시그널을 일괄 생성.
        
        벡터화 구현: .apply(axis=1) 대신 조건 벡터 연산으로 처리하여 성능 확보.
        (10만 행 기준 .apply는 ~2분, 벡터화는 ~1초)

        Returns:
            DataFrame with columns = strategy names, values = {-1, 0, 1}
        """
        signals = pd.DataFrame(index=df.index)

        # ── 1. WhaleSentiment ──
        prev_close = df['close'].shift(1)
        price_drop = df['close'] < prev_close
        price_rise = df['close'] > prev_close
        wr = df.get('whale_retail_ratio', pd.Series(1.0, index=df.index))
        wc = df.get('whale_conviction', pd.Series(0.0, index=df.index))

        long_ws = price_drop & (wr > 1.55) & (wc > 0)
        short_ws = price_rise & (wr < 1.40) & (wc < 0)
        signals['WhaleSentiment'] = np.where(long_ws, 1, np.where(short_ws, -1, 0))

        # ── 2. LiqSqueeze ──
        smf = df.get('smart_money_flow', pd.Series(0.0, index=df.index))
        smf_std = smf.expanding(min_periods=1).std().replace(0, 1.0)
        fr = df.get('last_funding_rate', pd.Series(0.0, index=df.index))
        oi_surge = smf > smf_std
        high_funding = fr.abs() > 0.0001

        signals['LiqSqueeze'] = np.where(
            oi_surge & high_funding & (fr < 0), 1,
            np.where(oi_surge & high_funding & (fr > 0), -1, 0)
        )

        # ── 3. BTCEthCorr ──
        btc_corr = df.get('btc_corr_60', pd.Series(0.0, index=df.index))
        lr = df.get('log_return', pd.Series(0.0, index=df.index))
        corr_high = btc_corr > 0.8

        signals['BTCEthCorr'] = np.where(
            corr_high & (lr > 0), 1,
            np.where(corr_high & (lr < 0), -1, 0)
        )

        # ── 4. VolSqueeze ──
        bbwz = df.get('bb_width_z', pd.Series(0.0, index=df.index))
        squeeze = bbwz < -1.5
        bullish_candle = df['close'] > df['open']
        bearish_candle = df['close'] < df['open']

        signals['VolSqueeze'] = np.where(
            squeeze & bullish_candle, 1,
            np.where(squeeze & bearish_candle, -1, 0)
        )

        # ── 5. VWAPDeviation ──
        vd = df.get('vwap_dist', pd.Series(0.0, index=df.index))
        signals['VWAPDeviation'] = np.where(
            vd < -0.03, 1, np.where(vd > 0.03, -1, 0)
        )

        # ── 6. HMAMomentum ──
        hs = df.get('hma_slope', pd.Series(0.0, index=df.index))
        signals['HMAMomentum'] = np.where(hs > 0, 1, np.where(hs < 0, -1, 0))

        # ── 7. OrderblockFVG ──
        rsi = df.get('rsi', pd.Series(50.0, index=df.index))
        wick = df.get('wick_ratio', pd.Series(0.0, index=df.index))

        signals['OrderblockFVG'] = np.where(
            (rsi < 30) & (wick > 0.5), 1,
            np.where((rsi > 70) & (wick > 0.5), -1, 0)
        )

        # ── 8. NetTakerFlow ──
        ntr = df.get('net_taker_ratio', pd.Series(0.0, index=df.index))
        ta_acc = df.get('taker_acceleration', pd.Series(0.0, index=df.index))

        signals['NetTakerFlow'] = np.where(
            (ntr > 0.1) & (ta_acc > 0), 1,
            np.where((ntr < -0.1) & (ta_acc < 0), -1, 0)
        )

        return signals

    # ================================================================
    # MISSING VALUE HANDLING (IMPROVED)
    # ================================================================
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        결측치 처리 전략 (개선):
        - inf → NaN 변환
        - diff 기반 피처 (첫 행이 NaN): fillna(0) 사용
        - 롤링 윈도우 피처: forward fill 후 backward fill
        - 그래도 남는 NaN → dropna (최소한의 행만 제거)
        """
        df = df.replace([np.inf, -np.inf], np.nan)

        # diff 기반 피처는 0으로 채우기 (첫 행의 변화량은 0으로 간주)
        diff_features = ['whale_conviction', 'smart_money_flow', 'log_return', 
                        'hma_slope', 'eth_btc_ratio_change', 'oi_change_rate',
                        'strategy_consensus', 'strategy_conviction', 'strategy_conflict',
                        'momentum_regime', 'reversion_regime']
        for col in diff_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 나머지 피처는 forward/backward fill
        feature_cols = [c for c in ULTIMATE_FEATURE_COLS if c in df.columns]
        other_features = [c for c in feature_cols if c not in diff_features]
        if other_features:
            df[other_features] = df[other_features].ffill().bfill()

        # 그래도 NaN이 남으면 해당 행만 제거
        df = df.dropna(subset=feature_cols)

        return df