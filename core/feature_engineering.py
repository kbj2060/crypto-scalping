import pandas as pd
import numpy as np
from .cvp import add_cvp_features

# [수정사항 3] regime_mean_reverting 중복 제거에 따른 컬럼 리스트 수정
# [신규 추가] rogers_satchell_vol, parkinson_vol, amihud_illiquidity_z 추가
ULTIMATE_FEATURE_COLS = [
    'whale_retail_ratio', 'whale_conviction', 'smart_money_flow',
    'funding_pressure', 'squeeze_power', 'oi_change_rate',
    'net_taker_ratio', 'taker_acceleration', 'trade_intensity',
    'big_trade_ratio',
    'volatility_z', 'rsi', 'macd_hist',
    'bb_width', 'bb_width_z', 'vwap_dist', 'hma_slope', 'wick_ratio',
    'garman_klass_vol', 'realized_vol_ratio',
    'rogers_satchell_vol', 'parkinson_vol', 'amihud_illiquidity_z', 
    'btc_corr_60', 'eth_btc_ratio_change', 'fvg_dist', 'chop_index',
    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
    'session_asia', 'session_europe', 'session_us',
    'is_hour_open',
    'regime_break',
    'turtle_signal', 'dual_momentum', 'mean_reversion_z',
    'breakout_strength', 'volume_profile_signal', 'fibonacci_level',
    'funding_roc_12', 'funding_roc_48', 'funding_roc_288',
    'funding_z_score', 'funding_abs',
    'long_squeeze_risk', 'short_squeeze_risk',
    'funding_price_divergence',
    'hurst_12', 'hurst_48', 'hurst_288',
    'regime_trending', 'hurst_change', 
    'cvp_poc_dist', 'cvp_vah_val_width', 'cvp_cluster_position',
    'cvp_volume_imbalance', 'cvp_regime',
    'ofi_acceleration',
    'kalman_velocity', 'return_autocorr', 'realized_skewness',
    'ofti', 'kel', 'mta_funding', 'svps',
    'pred_mdjd', 'conf_mdjd',
    # SyntheticAlphaEngine 확장 출력
    'cada', 'mshd', 'fvci', 'wpad', 'fdlv', 'vsdi', 'vebr', 'tlad', 'mtmb', 'fcsz',
    # VolatilityModelEngine 출력
    'garch_vol', 'garch_vol_z', 'ou_funding_z', 'ou_halflife',
    'jump_flag', 'jump_z', 'evt_tail_flag', 'evt_excess_z',
    # NewEliteSignalEngine 출력
    'sig_volume_confirm', 'sig_liquidity_trap', 'sig_trend_health',
]

EXCLUDE_FEATURE_COLS: list = [
    'timestamp', 'close_time', 'ignore',
    'open', 'high', 'low', 'close', 'close_btc',
    'volume', 'quote_volume', 'trades',
]

MUST_INCLUDE_FEATURES = [
    'rsi', 'mtf_trend_1h', 'bb_width_z', 'taker_acceleration',
    'smart_money_flow', 'trade_intensity', 'hma_slope',
    'btc_corr_60', 'mtf_trend_4h', 'bb_width', 'rogers_satchell_vol', 'parkinson_vol', 'amihud_illiquidity_z', 'garman_klass_vol',
]


class FeatureEngineer:
    def __init__(self, candle_minutes: int = 5):
        self.candle_minutes = candle_minutes
        self.windows = {
            'short': 5,
            'medium': 20,
            'long': 288,
            'volatility': 20,
            'corr': 60,
        }

    def process(self, eth_df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
        eth = eth_df.copy()
        btc = btc_df.copy()

        df = self._merge_data(eth, btc)

        df = self._create_alpha_features(df)
        df = self._create_order_flow(df)
        df = self._create_technical(df)
        df = self._create_advanced_volatility(df) # [신규 추가] 고급 변동성 지표 통합
        df = self._create_market_structure(df)
        df = self._create_temporal_features(df)
        df = self._add_regime_break(df)
        df = add_cvp_features(df, lookback=200, n_clusters=4)

        quant = QuantSignalFeatures(df)
        df = quant.add_all_signals()
        
        funding_features = FundingRateMomentum(df)
        df = funding_features.add_all_features()
        
        hurst_features = HurstExponentFeatures(df)
        df = hurst_features.add_all_features()
        
        # [수정사항 9] ofi_acceleration: 극단적 노이즈를 제어하기 위한 EWM 평활화 후 3-lag diff
        ntr_smooth = df['net_taker_ratio'].ewm(span=5).mean()
        df['ofi_acceleration'] = ntr_smooth.diff(3).fillna(0)

        df = self._create_predictive_stats(df)
        df = self._create_synthetic_alpha(df)
        df = self.add_mdjd_features(df)

        # ── 엘리트 퀀트 엔진 (합성 알파 + 변동성 모델 + 신규 Elite 시그널) ──
        from strategies.elite_strategies import (
            SyntheticAlphaEngine, VolatilityModelEngine, NewEliteSignalEngine,
        )
        SyntheticAlphaEngine().compute(df)    # cada, mshd, fvci, wpad, fdlv, vsdi, vebr, tlad, mtmb, fcsz
        VolatilityModelEngine().compute(df)   # garch_vol_z, ou_funding_z, jump_flag, jump_z, evt_tail_flag, evt_excess_z
        NewEliteSignalEngine().compute(df)    # sig_volume_confirm, sig_liquidity_trap, sig_trend_health

        df = self._handle_missing(df)
        return df

    def _merge_data(self, eth: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
        # dtype 통일 (pandas 버전에 따라 us/ns 불일치 방지)
        eth['timestamp'] = pd.to_datetime(eth['timestamp']).astype('datetime64[us]')
        btc['timestamp'] = pd.to_datetime(btc['timestamp']).astype('datetime64[us]')

        btc_cols_needed = {'close': 'close_btc', 'volume': 'volume_btc', 'quote_volume': 'quote_volume_btc'}
        already_renamed = 'close_btc' in btc.columns and 'close' not in btc.columns
        
        if already_renamed:
            btc_renamed = btc[['timestamp', 'close_btc', 'volume_btc', 'quote_volume_btc']].copy()
        else:
            btc_renamed = btc[['timestamp', 'close', 'volume', 'quote_volume']].rename(columns=btc_cols_needed)

        merged = pd.merge_asof(
            eth.sort_values('timestamp'),
            btc_renamed.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
        )
        return merged

    def _create_alpha_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['whale_retail_ratio'] = (
            df['sum_toptrader_long_short_ratio']
            / df['count_long_short_ratio'].replace(0, np.nan)
        )
        df['whale_conviction'] = df['sum_toptrader_long_short_ratio'].diff()

        # [수정사항 1] smart_money_flow: 비정상성 해소를 위해 절대값 diff 대신 pct_change 사용
        df['smart_money_flow'] = df['sum_open_interest_value'].pct_change().clip(-1, 1).fillna(0)

        df['funding_pressure'] = (
            df['last_funding_rate']
            .rolling(window=self.windows['long'], min_periods=1)
            .sum()
        )
        df['squeeze_power'] = df['sum_open_interest_value'] * df['last_funding_rate']
        df['oi_change_rate'] = df['sum_open_interest_value'].pct_change().clip(-1, 1).fillna(0)

        return df

    def _create_order_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        quote_vol = df['quote_volume'].replace(0, np.nan)
        taker_buy = df['taker_buy_quote']
        taker_sell = df['quote_volume'] - taker_buy

        net_flow = taker_buy - taker_sell
        df['net_taker_ratio'] = net_flow / quote_vol

        short_ma = df['net_taker_ratio'].rolling(window=2, min_periods=1).mean()
        long_ma = df['net_taker_ratio'].rolling(window=20, min_periods=1).mean()
        df['taker_acceleration'] = short_ma - long_ma

        df['trade_intensity'] = df['trades'] / df['volume'].replace(0, np.nan)

        avg_trade_size = df['quote_volume'] / df['trades'].replace(0, np.nan)
        avg_trade_rolling = avg_trade_size.rolling(window=self.windows['long'], min_periods=1).mean()
        avg_trade_std = avg_trade_size.rolling(window=self.windows['long'], min_periods=1).std().replace(0, 1e-8)
        df['big_trade_ratio'] = (avg_trade_size - avg_trade_rolling) / avg_trade_std

        return df

    def _create_technical(self, df: pd.DataFrame) -> pd.DataFrame:
        """외부 TA 라이브러리(pandas-ta)를 제거하고 순수 벡터화 연산으로 대체"""
        close = df['close']
        high = df['high']
        low = df['low']
        opn = df['open']

        df['log_return'] = np.log(close / close.shift(1))

        # 1. ATR (Average True Range)
        atr = self._calc_atr(high, low, close, length=14)
        win = self.windows['long']
        atr_mean = atr.rolling(window=win, min_periods=1).mean()
        atr_std = atr.rolling(window=win, min_periods=1).std().replace(0, 1e-8)
        df['volatility_z'] = (atr - atr_mean) / atr_std

        # 2. RSI (Relative Strength Index)
        df['rsi'] = self._calc_rsi(close, length=14)

        # 3. MACD Histogram (12, 26, 9)
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df['macd_hist'] = macd_line - signal_line

        # 4. Bollinger Bands Width (20, 2)
        bb_mid = close.rolling(window=20, min_periods=1).mean()
        bb_std = close.rolling(window=20, min_periods=1).std(ddof=0)
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        df['bb_width'] = (bb_upper - bb_lower) / (bb_mid + 1e-8)
        bbw_mean = df['bb_width'].rolling(window=100, min_periods=1).mean()
        bbw_std = df['bb_width'].rolling(window=100, min_periods=1).std().replace(0, 1e-8)
        df['bb_width_z'] = (df['bb_width'] - bbw_mean) / bbw_std

        df['vwap_dist'] = self._calc_vwap_dist(df)

        # 5. HMA Slope (Hull Moving Average)
        hma = self._calc_hma(close, n=20)
        df['hma_slope'] = hma.diff() / (close + 1e-8)

        # 6. Wick Ratio
        body_size = np.abs(close - opn)
        total_range = high - low
        df['wick_ratio'] = np.where(
            total_range == 0, 0, (total_range - body_size) / total_range
        )

        df['garman_klass_vol'] = self._garman_klass(high, low, opn, close)

        rv_short = df['log_return'].rolling(window=12, min_periods=1).std()
        rv_long = df['log_return'].rolling(window=self.windows['long'], min_periods=1).std()
        df['realized_vol_ratio'] = rv_short / rv_long.replace(0, 1e-8)
        ema_1h = close.ewm(span=12, adjust=False).mean()
        ema_4h = close.ewm(span=48, adjust=False).mean()
        
        df['mtf_trend_1h'] = ema_1h.pct_change().fillna(0)
        df['mtf_trend_4h'] = ema_4h.pct_change().fillna(0)

        return df

    def _create_advanced_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """[신규 추가] GK Volatility를 보완하는 고급 변동성 및 유동성 지표"""
        o = df['open'].replace(0, np.nan)
        h = df['high'].replace(0, np.nan)
        l = df['low'].replace(0, np.nan)
        c = df['close'].replace(0, np.nan)
        v = df['volume'].replace(0, np.nan)
        
        window = 20
        
        # 1. Rogers-Satchell Volatility (추세장 특화)
        # 공식: ln(H/O)*ln(H/C) + ln(L/O)*ln(L/C)
        rs_raw = (np.log(h/o) * np.log(h/c)) + (np.log(l/o) * np.log(l/c))
        df['rogers_satchell_vol'] = np.sqrt(rs_raw.rolling(window=window, min_periods=1).mean().clip(lower=0))
        
        # 2. Parkinson Volatility (꼬리/휩쏘 특화)
        # 공식: (ln(H/L))^2 / (4 * ln(2))
        parkinson_raw = (np.log(h/l)) ** 2 / (4 * np.log(2))
        df['parkinson_vol'] = np.sqrt(parkinson_raw.rolling(window=window, min_periods=1).mean().clip(lower=0))
        
        # 3. Amihud Illiquidity (유동성 공백 파악)
        # 공식: |Return| / Volume
        ret_abs = np.abs(c.pct_change())
        amihud_raw = ret_abs / v
        
        # Z-score로 정규화하여 스케일 안정화
        amihud_mean = amihud_raw.rolling(window=288, min_periods=1).mean()
        amihud_std = amihud_raw.rolling(window=288, min_periods=1).std().replace(0, 1e-8)
        df['amihud_illiquidity_z'] = (amihud_raw - amihud_mean) / amihud_std
        
        # 결측치 보정
        df['amihud_illiquidity_z'] = df['amihud_illiquidity_z'].fillna(0)
        
        return df

    def _calc_vwap_dist(self, df: pd.DataFrame) -> pd.Series:
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
        h = high.clip(lower=low)
        o = opn.replace(0, np.nan)
        c = close.replace(0, np.nan)
        l = low.replace(0, np.nan)

        log_hl = (np.log(h / l)) ** 2
        log_co = (np.log(c / o)) ** 2

        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return (gk.rolling(window=window, min_periods=1).mean().clip(lower=0) ** 0.5)

    def _create_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['close']
        close_btc = df['close_btc']

        eth_ret = close.pct_change()
        btc_ret = close_btc.pct_change()
        df['btc_corr_60'] = eth_ret.rolling(window=self.windows['corr']).corr(btc_ret).fillna(0)

        eth_btc_ratio = close / close_btc.replace(0, np.nan)
        df['eth_btc_ratio_change'] = eth_btc_ratio.pct_change()

        df['fvg_dist'] = self._calc_fvg_dist(df)
        
        # 7. Choppiness Index (CHOP) - 커스텀 계산 적용
        df['chop_index'] = self._calc_chop(df['high'], df['low'], close, length=14)

        return df

    def _calc_rma(self, x: pd.Series, n: int) -> pd.Series:
        """Wilder's Smoothing (RMA) 계산 헬퍼 함수"""
        return x.ewm(alpha=1/n, adjust=False).mean()

    def _calc_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
        """True Range 기반 ATR 계산"""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return self._calc_rma(tr, length)

    def _calc_rsi(self, close: pd.Series, length: int) -> pd.Series:
        """벡터화된 RSI 연산 (pandas-ta의 기본 Wilder's Smoothing 방식과 100% 동일)"""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = self._calc_rma(gain, length)
        avg_loss = self._calc_rma(loss, length)
        
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def _calc_wma(self, s: pd.Series, period: int) -> pd.Series:
        """HMA를 위한 가중이동평균(WMA) 계산"""
        weights = np.arange(1, period + 1)
        return s.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    def _calc_hma(self, close: pd.Series, n: int) -> pd.Series:
        """선형대수 벡터 기반의 Hull Moving Average (HMA) 계산"""
        half_length = int(n / 2)
        sqrt_length = int(np.sqrt(n))
        
        wma_half = self._calc_wma(close, half_length)
        wma_full = self._calc_wma(close, n)
        raw_hma = 2 * wma_half - wma_full
        
        return self._calc_wma(raw_hma, sqrt_length)

    def _calc_chop(self, high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
        """Choppiness Index (CHOP) - ZeroDivision 완벽 방어형"""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr_sum = tr.rolling(window=length).sum()
        high_max = high.rolling(window=length).max()
        low_min = low.rolling(window=length).min()
        
        # 💡 [핵심 수정] atr_sum(분자)에도 1e-8을 더해 log10(0) 에러를 원천 차단합니다.
        chop = 100 * np.log10((atr_sum + 1e-8) / (high_max - low_min + 1e-8)) / np.log10(length)
        return chop

    @staticmethod
    def _calc_fvg_dist(df: pd.DataFrame) -> pd.Series:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        n = len(df)
        fvg_dist = np.zeros(n, dtype=np.float64)

        lookback = 50

        for i in range(2, n):
            nearest_gap_dist = 0.0
            min_abs_dist = np.inf

            for j in range(i, max(i - lookback, 1), -1):
                if high[j - 2] < low[j]:
                    gap_mid = (high[j - 2] + low[j]) / 2
                    dist = (close[i] - gap_mid) / (close[i] + 1e-8)
                    if abs(dist) < min_abs_dist:
                        min_abs_dist = abs(dist)
                        nearest_gap_dist = dist

                if low[j - 2] > high[j]:
                    gap_mid = (low[j - 2] + high[j]) / 2
                    dist = (close[i] - gap_mid) / (close[i] + 1e-8)
                    if abs(dist) < min_abs_dist:
                        min_abs_dist = abs(dist)
                        nearest_gap_dist = dist

            fvg_dist[i] = nearest_gap_dist

        return pd.Series(fvg_dist, index=df.index)

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        import pandas_market_calendars as mcal

        ts = df['timestamp']
        hour = ts.dt.hour
        minute = ts.dt.minute

        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['minute_sin'] = np.sin(2 * np.pi * minute / 60)
        df['minute_cos'] = np.cos(2 * np.pi * minute / 60)

        ts_utc = ts.dt.tz_localize('UTC') if ts.dt.tz is None else ts.dt.tz_convert('UTC')
        start_date = ts_utc.min().date()
        end_date = ts_utc.max().date()
        
        try:
            # 아시아 세션 (JPX)
            tse = mcal.get_calendar('JPX')
            df['session_asia'] = ts_utc.isin(mcal.date_range(tse.schedule(start_date=start_date, end_date=end_date), frequency='1min')).astype(np.float32)
            
            # 유럽 세션 (LSE)
            lse = mcal.get_calendar('LSE')
            df['session_europe'] = ts_utc.isin(mcal.date_range(lse.schedule(start_date=start_date, end_date=end_date), frequency='1min')).astype(np.float32)
            
            # 미국 세션 (NYSE)
            nyse = mcal.get_calendar('NYSE')
            df['session_us'] = ts_utc.isin(mcal.date_range(nyse.schedule(start_date=start_date, end_date=end_date), frequency='1min')).astype(np.float32)
        except Exception as e:
            logger.warning(f"Market calendars 갱신 실패. 정적 로직으로 대체 ({e})")
            df['session_asia']   = ((hour >= 0) & (hour < 8)).astype(np.float32)
            df['session_europe'] = ((hour >= 8) & (hour < 16)).astype(np.float32)
            df['session_us'] = ((hour >= 14.5) & (hour < 21)).astype(np.float32)

        df['is_hour_open'] = (minute < 5).astype(np.float32)

        return df

    def _create_synthetic_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """합성 알파 피처 4종:
        OFTI  — 오더플로우 독성 지수
        KEL   — 유동성 운동 에너지
        MTA   — 다중-시간 펀딩비 가속도
        SVPS  — 공간적 볼륨 프로파일 왜곡
        """
        ROLL = 288  # 24h (5분봉 기준)

        # ── OFTI: Order Flow Toxicity Index ──────────────────────────────────
        # smart_money_flow × whale_conviction × amihud_illiquidity 세기
        # amihud는 abs()로 안정화 (Z-스코어는 음수 가능), +1 로 최소 스케일 보장
        ofti_raw = (
            df['smart_money_flow']
            * df['whale_conviction']
            * (df['amihud_illiquidity_z'].abs() + 1.0)
        )
        df['ofti'] = np.tanh(ofti_raw * 3.0).fillna(0)

        # ── KEL: Kinetic Energy of Liquidity ─────────────────────────────────
        # oi_change_rate / garman_klass_vol 비율로 "억눌린 에너지" 측정
        # funding_pressure 부호로 방향 결정 → Z-스코어 후 tanh 바운딩
        kel_raw = (
            df['oi_change_rate']
            / (df['garman_klass_vol'] + 1e-6)
            * np.sign(df['funding_pressure'])
        )
        kel_mean = kel_raw.rolling(ROLL, min_periods=1).mean()
        kel_std  = kel_raw.rolling(ROLL, min_periods=1).std().replace(0, 1e-8)
        df['kel'] = np.tanh((kel_raw - kel_mean) / kel_std * 0.5).fillna(0)

        # ── MTA: Multi-Timeframe Funding Acceleration ─────────────────────────
        # 단기 가중 ROC 합산, funding_abs로 정규화, squeeze_power Z-스코어로 타이밍 필터
        weighted_roc = (
            0.5 * df['funding_roc_12']
            + 0.3 * df['funding_roc_48']
            + 0.2 * df['funding_roc_288']
        )
        # funding_abs는 0.0001 스케일 → max(1e-5, ...) 로 실질 정규화
        mta_normalized = weighted_roc / df['funding_abs'].clip(lower=1e-5)

        sq_mean = df['squeeze_power'].rolling(ROLL, min_periods=1).mean()
        sq_std  = df['squeeze_power'].rolling(ROLL, min_periods=1).std().replace(0, 1e-8)
        squeeze_z = (df['squeeze_power'] - sq_mean) / sq_std

        df['mta_funding'] = (mta_normalized * np.tanh(squeeze_z)).clip(-3, 3) / 3
        df['mta_funding'] = df['mta_funding'].fillna(0)

        # ── SVPS: Spatial Volume Profile Skew ────────────────────────────────
        # POC 거리 × 거래량 불균형 × exp(-매물대 두께)
        # exp 폭주 방지: vah_val_width를 [0, 5] 클리핑
        df['svps'] = np.tanh(
            2.0
            * df['cvp_poc_dist']
            * df['cvp_volume_imbalance']
            * np.exp(-df['cvp_vah_val_width'].clip(0, 5))
        ).fillna(0)

        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan)

        diff_features = [
            'whale_conviction', 'smart_money_flow', 'log_return',
            'hma_slope', 'eth_btc_ratio_change', 'oi_change_rate',
            'turtle_signal', 'dual_momentum', 'mean_reversion_z',
            'breakout_strength', 'volume_profile_signal', 'fibonacci_level',
            'funding_roc_12', 'funding_roc_48', 'funding_roc_288',
            'hurst_change', 'ofi_acceleration',
            'cvp_poc_dist', 'cvp_vah_val_width', 'cvp_cluster_position',
            'cvp_volume_imbalance', 'cvp_regime',
            'amihud_illiquidity_z',
            'kalman_velocity', 'return_autocorr', 'realized_skewness',
        ]
        for col in diff_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        if 'regime_break' in df.columns:
            df['regime_break'] = df['regime_break'].fillna(0)

        feature_cols = [c for c in ULTIMATE_FEATURE_COLS if c in df.columns]
        other_features = [c for c in feature_cols if c not in diff_features and c != 'regime_break']
        if other_features:
            # bfill 제거: 롤링 윈도우 초반 NaN을 미래 데이터로 채우는 룩어헤드 편향 방지
            # ffill로 과거 전파 후, 여전히 NaN인 초반 구간(워밍업)은 0으로 채움
            df[other_features] = df[other_features].ffill().fillna(0)

        df = df.dropna(subset=feature_cols)

        return df

    def _create_predictive_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """순수 수학/통계 기반 예측 피처 3종"""
        close = df['close']
        df['kalman_velocity']  = self._kalman_trend_velocity(close)
        df['return_autocorr']  = self._return_autocorrelation(close)
        df['realized_skewness'] = self._realized_skewness(close)
        return df

    def _kalman_trend_velocity(self, close: pd.Series,
                                obs_noise: float = 1e-3,
                                proc_noise: float = 1e-5) -> pd.Series:
        """칼만 필터 추세 속도 (Kalman Trend Velocity)

        상태 벡터: [가격 레벨, 가격 속도]
        측정값: close price

        단순 MA보다 lag이 적고 노이즈에 강한 상태공간 추정.
        velocity > 0 → 상승 추세 강도, velocity < 0 → 하락 추세 강도.
        가격 대비 정규화하여 cross-asset 비교 가능.
        """
        vals = close.values.astype(np.float64)
        n    = len(vals)

        # 상태 전이 행렬: level(t+1) = level(t) + velocity(t)
        F = np.array([[1., 1.], [0., 1.]])
        H = np.array([[1., 0.]])
        Q = np.eye(2) * proc_noise   # 프로세스 노이즈
        R = np.array([[obs_noise]])  # 측정 노이즈

        x = np.array([vals[0], 0.0])
        P = np.eye(2)
        velocities = np.empty(n)

        for i in range(n):
            # Predict
            x = F @ x
            P = F @ P @ F.T + Q
            # Update
            S   = (H @ P @ H.T + R)[0, 0]
            K   = (P @ H.T).flatten() / S
            inn = vals[i] - (H @ x)[0]   # innovation
            x   = x + K * inn
            P   = (np.eye(2) - np.outer(K, H)) @ P
            velocities[i] = x[1]

        # 가격 대비 정규화: 상대 속도 (return/bar 단위)
        rel_velocity = velocities / (vals + 1e-8)
        return pd.Series(np.clip(rel_velocity, -0.05, 0.05), index=close.index).fillna(0)

    def _return_autocorrelation(self, close: pd.Series,
                                 window: int = 48,
                                 lag: int = 1) -> pd.Series:
        """롤링 수익률 자기상관계수 (Return Autocorrelation)

        lag=1 피어슨 상관: r(t) ↔ r(t-1)
        - 양수(+): 모멘텀 구간 → 추세 지속 신호
        - 음수(-): 평균회귀 구간 → 반전 신호
        - 0 근방: 랜덤워크 → 비예측 구간

        Hurst 지수 대비 장점: 방향성(모멘텀 vs 회귀) 명시적 제공.
        """
        returns = close.pct_change().fillna(0)

        def _autocorr(x):
            if len(x) < lag + 4:
                return 0.0
            r_t  = x[lag:]
            r_tm = x[:-lag]
            denom = r_t.std() * r_tm.std()
            if denom < 1e-10:
                return 0.0
            return np.corrcoef(r_t, r_tm)[0, 1]

        return (
            returns
            .rolling(window, min_periods=window // 2)
            .apply(_autocorr, raw=True)
            .fillna(0)
        )

    def _realized_skewness(self, close: pd.Series, window: int = 96) -> pd.Series:
        """실현 비대칭도 (Realized Skewness)

        3차 표준화 모멘트: E[(r - μ)³] / σ³
        - 음수(왼꼬리): 극단 하락 잦음 → 숏 리스크 프리미엄 존재
        - 양수(오른꼬리): 극단 상승 잦음 → 롱 기대 수익 프리미엄
        - 0 근방: 대칭 분포

        기존 Parkinson/GK 변동성이 잡지 못하는 방향성 있는 꼬리 위험 포착.
        [-3, 3] clip으로 이상치 제어.
        """
        returns = close.pct_change().fillna(0)

        def _skew(x):
            if len(x) < 8:
                return 0.0
            mu  = x.mean()
            sig = x.std()
            if sig < 1e-10:
                return 0.0
            return ((x - mu) ** 3).mean() / (sig ** 3 + 1e-10)

        result = (
            returns
            .rolling(window, min_periods=window // 2)
            .apply(_skew, raw=True)
        )
        return result.clip(-3, 3).fillna(0)

    def _add_regime_break(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'volatility_z' not in df.columns:
            df['regime_break'] = 0
            return df

        vol = df['volatility_z']
        window = 20
        vol_std = vol.rolling(window).std()
        threshold = vol_std.quantile(0.95)
        df['regime_break'] = (vol_std > threshold).astype(np.float32)
        return df

    def augment_training_data(self, df: pd.DataFrame, noise_level: float = 0.01) -> pd.DataFrame:
        augmented = df.copy()

        exclude_cols = ['session_asia', 'session_europe', 'session_us',
                        'is_hour_open', 'regime_break']
        feature_cols = [c for c in ULTIMATE_FEATURE_COLS
                        if c in df.columns and c not in exclude_cols]

        for col in feature_cols:
            std = df[col].std()
            if std == 0: continue
            noise = np.random.normal(0, std * noise_level, len(df))
            augmented[col] = df[col] + noise

        return augmented

    @staticmethod
    def add_mdjd_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Microstructure-Driven Jump-Diffusion (MDJD) Feature Generator
        """
        # 1. 안전한 Z-스코어 정규화 (Squeeze Power)
        sqz_mean = df['squeeze_power'].rolling(window=288, min_periods=1).mean()
        sqz_std  = df['squeeze_power'].rolling(window=288, min_periods=1).std()
        squeeze_z = (df['squeeze_power'] - sqz_mean) / (sqz_std + 1e-8)

        # 파라미터 사전 정의 (스케일을 맞춘 휴리스틱 가중치)
        W1, W2 = 0.005, 0.002
        BETA    = 0.003
        GAMMA   = 0.01
        RHO     = 0.005
        DELTA   = 0.4

        # 2. 컴포넌트별 계산
        # D_t: Smart Money Drift
        D = W1 * df['smart_money_flow'] * (1 + np.tanh(df['whale_conviction'])) + \
            W2 * df['mtf_trend_4h']

        # I_t: Order-book Imbalance Shock
        I = BETA * df['net_taker_ratio'] * np.exp(np.tanh(df['taker_acceleration'])) * \
            (df['amihud_illiquidity_z'].clip(lower=0) + 1.0)

        # J_t: Liquidity Squeeze Jump
        J = GAMMA * np.tanh(squeeze_z) * np.tanh(df['funding_pressure']) * \
            (df['breakout_strength'] > DELTA).astype(float)

        # G_t: Volume Profile Gravity
        # [수정] mtf_trend_4h는 pct_change 스케일(~0.0001)이라 tanh(x) ≈ x → 상수 1.0에 수렴
        # Z-스코어로 정규화해야 dampener가 실제로 작동함
        trend_4h_z = df['mtf_trend_4h'] / (df['mtf_trend_4h'].rolling(288, min_periods=1).std() + 1e-8)
        trend_dampener = 1.0 - np.tanh(trend_4h_z.abs())
        G = -RHO * df['cvp_poc_dist'] * np.exp(-df['cvp_volume_imbalance'].clip(-5, 5)) * trend_dampener

        # 3. MDJD 앙상블 신호 생성
        R_hat = D + I + J + G

        df['pred_mdjd'] = np.sign(R_hat).clip(-1, 1)
        df['conf_mdjd'] = np.tanh(np.abs(R_hat) * 100)

        return df

class QuantSignalFeatures:
    """유명 퀀트 알고리즘의 신호를 피처로 변환"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.close = df['close']
        self.high = df['high']
        self.low = df['low']
        self.volume = df['volume']
    
    def add_all_signals(self) -> pd.DataFrame:
        self.df['turtle_signal'] = self._turtle_trading()
        self.df['dual_momentum'] = self._dual_momentum()
        self.df['mean_reversion_z'] = self._mean_reversion()
        self.df['breakout_strength'] = self._breakout()
        self.df['volume_profile_signal'] = self._volume_profile()
        self.df['fibonacci_level'] = self._fibonacci()
        return self.df
    
    def _turtle_trading(self) -> pd.Series:
        entry_high = self.close.rolling(288).max()
        exit_low = self.close.rolling(144).min()
        
        signal = np.where(
            self.close > entry_high.shift(1), 1.0,
            np.where(self.close < exit_low.shift(1), -1.0, 0.0)
        )
        return pd.Series(signal, index=self.df.index).fillna(0).ewm(span=5).mean()
    
    def _dual_momentum(self) -> pd.Series:
        abs_momentum = (self.close / self.close.shift(2016) - 1).fillna(0)
        
        if 'close_btc' in self.df.columns:
            btc_momentum = (self.df['close_btc'] / self.df['close_btc'].shift(2016) - 1).fillna(0)
            rel_momentum = abs_momentum - btc_momentum
        else:
            rel_momentum = 0
        
        signal = np.where(
            (abs_momentum > 0) & (rel_momentum > 0), 1.0,
            np.where((abs_momentum < 0) & (rel_momentum < 0), -1.0, 0.0)
        )
        return pd.Series(signal, index=self.df.index).fillna(0)
    
    def _mean_reversion(self) -> pd.Series:
        window = 288
        ma = self.close.rolling(window).mean()
        std = self.close.rolling(window).std()
        z_score = (self.close - ma) / (std + 1e-8)
        
        signal = -np.tanh(z_score / 2)
        return pd.Series(signal, index=self.df.index).fillna(0)
    
    def _breakout(self) -> pd.Series:
        window = 144
        box_high = self.high.rolling(window).max()
        box_low = self.low.rolling(window).min()
        box_range = box_high - box_low
        
        box_center = (box_high + box_low) / 2
        strength = (self.close - box_center) / (box_range + 1e-8)
        return pd.Series(np.clip(strength, -1, 1), index=self.df.index).fillna(0)
    
    def _volume_profile(self) -> pd.Series:
        window = 288
        vwap = (self.close * self.volume).rolling(window).sum() / (self.volume.rolling(window).sum() + 1e-8)
        deviation = (self.close - vwap) / (vwap + 1e-8)
        volume_surge = self.volume / (self.volume.rolling(window).mean() + 1e-8)
        
        signal = -np.tanh(deviation * volume_surge)
        return pd.Series(signal, index=self.df.index).fillna(0)
    
    def _fibonacci(self) -> pd.Series:
        """[수정사항 10] 피보나치 레벨의 연속성 보장을 위해 0~1 정규화된 위치값으로 반환"""
        window = 288
        swing_high = self.high.rolling(window).max()
        swing_low = self.low.rolling(window).min()
        swing_range = swing_high - swing_low
        
        # 바닥(0)에서 천장(1)까지 현재 가격이 피보나치 레인지 내 어디에 있는지 연속적으로 매핑
        position = (self.close.values - swing_low.values) / (swing_range.values + 1e-8)
        return pd.Series(position, index=self.df.index).clip(0, 1).fillna(0.5)


class FundingRateMomentum:
    """펀딩비 기반 모멘텀 - 롱/숏 스퀴즈 포착"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        if 'last_funding_rate' not in df.columns:
            self.funding_rate = pd.Series(0, index=df.index)
        else:
            self.funding_rate = df['last_funding_rate']
    
    def add_all_features(self):
        self.df['funding_roc_12'] = self._calculate_roc(12)
        self.df['funding_roc_48'] = self._calculate_roc(48)
        self.df['funding_roc_288'] = self._calculate_roc(288)
        self.df['funding_z_score'] = self._calculate_zscore(288)
        self.df['funding_abs'] = np.abs(self.funding_rate)
        self.df['long_squeeze_risk'] = self._long_squeeze_score()
        self.df['short_squeeze_risk'] = self._short_squeeze_score()
        self.df['funding_price_divergence'] = self._divergence()
        return self.df
    
    def _calculate_roc(self, window):
        """[수정사항 6] ROC 분모 0에 의한 폭발 방지 및 강한 하한선 적용"""
        shifted = self.funding_rate.shift(window)
        roc = (self.funding_rate - shifted) / (shifted.abs().clip(lower=1e-4) + 1e-8)
        return roc.clip(-10, 10).fillna(0)
    
    def _calculate_zscore(self, window):
        mean = self.funding_rate.rolling(window, min_periods=1).mean()
        std = self.funding_rate.rolling(window, min_periods=1).std()
        z = (self.funding_rate - mean) / (std + 1e-8)
        return z.fillna(0)
    
    def _long_squeeze_score(self):
        """[수정사항 7] 롱/숏 스퀴즈 매직 넘버 대칭 통일 (0.0002)"""
        funding_extreme = np.clip(self.funding_rate / 0.0002, 0, 1)
        funding_surge = np.clip(self.df.get('funding_roc_12', 0) / 3, 0, 1)
        if 'oi_change_rate' in self.df.columns:
            oi_buildup = np.clip(self.df['oi_change_rate'] * 10, 0, 1)
        else:
            oi_buildup = 0
        score = 0.5 * funding_extreme + 0.3 * funding_surge + 0.2 * oi_buildup
        return score
    
    def _short_squeeze_score(self):
        """[수정사항 7] 롱/숏 스퀴즈 매직 넘버 대칭 통일 (0.0002)"""
        funding_extreme = np.clip(-self.funding_rate / 0.0002, 0, 1)
        funding_plunge = np.clip(-self.df.get('funding_roc_12', 0) / 3, 0, 1)
        if 'oi_change_rate' in self.df.columns:
            oi_buildup = np.clip(self.df['oi_change_rate'] * 10, 0, 1)
        else:
            oi_buildup = 0
        score = 0.5 * funding_extreme + 0.3 * funding_plunge + 0.2 * oi_buildup
        return score
    
    def _divergence(self):
        """[수정사항 5] Z-Score 기반의 방향성 보존 발산 스코어로 수정 (정보 손실 차단)"""
        price_change = self.df['close'].pct_change(12)
        funding_change = self.funding_rate.diff(12)
        
        price_z = price_change / (price_change.rolling(48).std() + 1e-8)
        funding_z = funding_change / (funding_change.rolling(48).std() + 1e-8)
        
        # 양수=가격상승+펀딩하락(숏스퀴즈 진행), 음수=가격하락+펀딩상승(롱스퀴즈 진행)
        divergence = (price_z - funding_z).clip(-3, 3)
        return pd.Series(divergence, index=self.df.index).fillna(0)


class HurstExponentFeatures:
    """[수정사항 4] 연산 속도 O(N) 최적화 된 벡터화 허스트 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.close = df['close'].values
    
    def add_all_features(self):
        self.df['hurst_12'] = self._rolling_hurst_fast(12)
        self.df['hurst_48'] = self._rolling_hurst_fast(48)
        self.df['hurst_288'] = self._rolling_hurst_fast(288)
        
        self.df['regime_trending'] = (self.df['hurst_48'] > 0.5).astype(float)
        
        self.df['hurst_change'] = self.df['hurst_48'].diff(12).fillna(0)
        
        return self.df

    def _rolling_hurst_fast(self, window):
        """벡터화된 R/S Hurst 근사 — O(N) 속도"""
        returns = pd.Series(self.close).pct_change().fillna(0)
        
        def rs_hurst(x):
            if len(x) < 10:
                return 0.5
            mean_r = x.mean()
            deviate = np.cumsum(x - mean_r)
            R = deviate.max() - deviate.min()
            S = x.std()
            if S < 1e-10:
                return 0.5
            return np.log(R / S + 1e-10) / np.log(len(x))
        
        return returns.rolling(window, min_periods=window//2).apply(rs_hurst, raw=True).fillna(0.5)

    