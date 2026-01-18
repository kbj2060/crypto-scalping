"""
메인 트레이딩 봇
"""
import logging
import time
import sys
import os
from datetime import datetime, timedelta
import config
from core import DataCollector, RiskManager, BinanceClient
from core.indicators import Indicators
from strategies import (
    LiquiditySweepStrategy,
    BTCEthCorrelationStrategy,
    CVDDeltaStrategy,
    VolatilitySqueezeStrategy,
    FundingRateStrategy,
    OrderblockFVGStrategy,
    LiquidationSpikeStrategy,
    # 횡보장 Top 5 Mean-Reversion 전략
    BollingerMeanReversionStrategy,
    VWAPDeviationStrategy,
    RangeTopBottomStrategy,
    StochRSIMeanReversionStrategy,
    CVDFakePressureStrategy
)

# 로깅 설정
# logs 디렉토리가 없으면 생성
os.makedirs('logs', exist_ok=True)

# Windows에서 UTF-8 인코딩 설정 (이모지 출력을 위해)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python 3.6 이하에서는 reconfigure가 없음
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        self.data_collector = DataCollector()
        self.risk_manager = RiskManager()
        self.client = BinanceClient()
        
        # 전략 초기화 (폭발장/횡보장 분리)
        self.breakout_strategies = []
        self.range_strategies = []
        
        # 폭발장 전략
        if config.STRATEGIES['liquidity_sweep']:
            self.breakout_strategies.append(LiquiditySweepStrategy())
        if config.STRATEGIES['btc_eth_correlation']:
            self.breakout_strategies.append(BTCEthCorrelationStrategy())
        if config.STRATEGIES['cvd_delta']:
            self.breakout_strategies.append(CVDDeltaStrategy())
        if config.STRATEGIES['volatility_squeeze']:
            self.breakout_strategies.append(VolatilitySqueezeStrategy())
        # 펀딩비 전략: 선물 거래에서만 활성화
        if config.STRATEGIES['funding_rate'] and self.client.use_futures:
            self.breakout_strategies.append(FundingRateStrategy())
        if config.STRATEGIES['orderblock_fvg']:
            self.breakout_strategies.append(OrderblockFVGStrategy())
        # 청산 스파이크 전략: 선물 거래에서만 활성화
        if config.STRATEGIES.get('liquidation_spike', False) and self.client.use_futures:
            self.breakout_strategies.append(LiquidationSpikeStrategy())
        
        # 횡보장 전략 (Mean-Reversion 7개)
        if config.STRATEGIES.get('bollinger_mean_reversion', False):
            self.range_strategies.append(BollingerMeanReversionStrategy())
            logger.info("✓ 볼린저 밴드 평균 회귀 전략 활성화")
        if config.STRATEGIES.get('vwap_deviation', False):
            self.range_strategies.append(VWAPDeviationStrategy())
            logger.info("✓ VWAP 편차 평균 회귀 전략 활성화")
        if config.STRATEGIES.get('range_top_bottom', False):
            self.range_strategies.append(RangeTopBottomStrategy())
            logger.info("✓ Range Top/Bottom 반전 전략 활성화")
        if config.STRATEGIES.get('stoch_rsi_mean_reversion', False):
            self.range_strategies.append(StochRSIMeanReversionStrategy())
            logger.info("✓ Stoch RSI 평균 회귀 전략 활성화")
        if config.STRATEGIES.get('cvd_fake_pressure', False):
            self.range_strategies.append(CVDFakePressureStrategy())
            logger.info("✓ CVD Fake Pressure 전략 활성화")
        
        # 전체 전략 리스트 (하위 호환성)
        self.strategies = self.breakout_strategies + self.range_strategies
        
        # 시장 모드 상태
        self.current_market_mode = None  # 'TREND', 'RANGE', 'NEUTRAL'
        
        # 핵심 돌파 전략 개수 계산 (Liquidity Sweep, Funding Rate 제외)
        core_breakout_count = sum(1 for s in self.breakout_strategies 
                                 if s.name not in ['Liquidity Sweep', 'Funding Rate'])
        logger.info(f"트레이딩 봇 초기화 완료 - 활성 전략: {len(self.strategies)}개")
        logger.info(f"   추세장(돌파): 핵심 {core_breakout_count}개 + 보조 필터, 횡보장: {len(self.range_strategies)}개")
    
    def update_data(self):
        """데이터 업데이트"""
        return self.data_collector.update_data()
    
    def detect_market_mode(self):
        """시장 상태 판단 (Trend / Range / Neutral)"""
        try:
            eth_data = self.data_collector.get_candles('ETH', count=50)
            if eth_data is None or len(eth_data) < 30:
                return 'NEUTRAL'
            
            # 1. BBW 계산
            bb_bands = Indicators.calculate_bollinger_bands(eth_data, period=20, std_dev=2.0)
            if bb_bands is None:
                return 'NEUTRAL'
            bbw = Indicators.calculate_bbw(bb_bands)
            if bbw is None:
                return 'NEUTRAL'
            latest_bbw = float(bbw.iloc[-1])
            
            # 2. ADX 계산
            adx = Indicators.calculate_adx(eth_data, period=14)
            if adx is None:
                return 'NEUTRAL'
            latest_adx = float(adx.iloc[-1])
            
            # 3. ATR 증가율 계산
            atr = Indicators.calculate_atr(eth_data, period=14)
            if atr is None:
                return 'NEUTRAL'
            if len(atr) < 20:
                return 'NEUTRAL'
            latest_atr = float(atr.iloc[-1])
            atr_ma = float(Indicators.calculate_sma(atr, period=20).iloc[-1])
            atr_increase_pct = ((latest_atr - atr_ma) / atr_ma) * 100 if atr_ma > 0 else 0
            
            # 시장 상태 지표 로깅
            logger.info(f"📊 시장 상태 지표 - BBW: {latest_bbw:.4f}, ADX: {latest_adx:.2f}, ATR 증가율: {atr_increase_pct:.2f}%")
            
            # TREND Mode 우선 판단 (추세장이 더 명확할 때 우선)
            # BBW > 0.008 (0.8%) OR ADX > 25 OR ATR 증가율 > 20%
            if latest_bbw > 0.008 or latest_adx > 25 or atr_increase_pct > 20:
                logger.info(f"→ TREND 모드 판단: BBW={latest_bbw:.4f} > 0.008 또는 ADX={latest_adx:.2f} > 25 또는 ATR 증가율={atr_increase_pct:.2f}% > 20%")
                return 'TREND'
            
            # RANGE Mode 판단 (TREND가 아닐 때)
            # BBW < 0.006 (0.6%) AND ADX < 20
            if latest_bbw < 0.006 and latest_adx < 20:
                logger.info(f"→ RANGE 모드 판단: BBW={latest_bbw:.4f} < 0.006 AND ADX={latest_adx:.2f} < 20")
                return 'RANGE'
            
            # Neutral Mode (중간 구간)
            # BBW: 0.006~0.008 사이 OR ADX: 20~25 사이
            logger.info(f"→ NEUTRAL 모드: 중간 구간 (BBW: {latest_bbw:.4f}, ADX: {latest_adx:.2f}, ATR 증가율: {atr_increase_pct:.2f}%)")
            return 'NEUTRAL'
            
        except Exception as e:
            logger.error(f"시장 상태 판단 실패: {e}")
            return 'NEUTRAL'
    
    def analyze_strategies(self):
        """시장 상태에 따라 적절한 전략만 분석"""
        # 1. 시장 상태 판단
        market_mode = self.detect_market_mode()
        self.current_market_mode = market_mode
        
        logger.info("=" * 60)
        logger.info("📊 전략 분석 시작 (3분봉 데이터 기준)")
        logger.info("=" * 60)
        
        # 데이터 상태 확인
        eth_data_len = len(self.data_collector.eth_data) if self.data_collector.eth_data is not None else 0
        btc_data_len = len(self.data_collector.btc_data) if self.data_collector.btc_data is not None else 0
        logger.info(f"📦 데이터 상태 - ETH: {eth_data_len}개 캔들, BTC: {btc_data_len}개 캔들")
        
        # 시장 상태 표시
        mode_emoji = "🔥" if market_mode == 'TREND' else "📊" if market_mode == 'RANGE' else "⚪"
        logger.info(f"{mode_emoji} 현재 시장 상태: {market_mode}")
        
        all_signals = []
        
        # 2. 시장 모드에 따라 전략 분석
        if market_mode == 'TREND':
            all_signals = self._analyze_trend_mode()
        elif market_mode == 'RANGE':
            all_signals = self._analyze_range_mode()
        else:
            logger.info("⚪ Neutral Mode: 거래 금지 (명확한 추세/횡보가 아님)")
            return []
        
        # 3. 전체 요약
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"📈 신호 요약 - {market_mode} 모드: {len(all_signals)}개 신호 발견")
        logger.info("=" * 60)
        
        return all_signals
    
    def _analyze_trend_mode(self):
        """추세장(돌파장) 전략 분석 - 핵심 돌파 전략 3개 + 보조 필터"""
        signals = []
        
        # 핵심 돌파 전략 목록
        core_strategies = ['Volatility Squeeze', 'Orderblock FVG', 'CVD Delta']
        filter_strategies = ['BTC/ETH Correlation', 'Liquidation Spike']
        excluded_strategies = ['Liquidity Sweep', 'Funding Rate']
        
        logger.info("")
        logger.info("🔥 추세장 모드 (Trend Mode) - 돌파 전략 최적화")
        logger.info("   필수 핵심: Volatility Squeeze + Orderblock FVG + CVD Delta")
        logger.info("   보조 필터: BTC/ETH Correlation (환경), Liquidation Spike (보너스)")
        logger.info("   제외 전략: Liquidity Sweep, Funding Rate")
        logger.info("-" * 60)
        
        for strategy in self.breakout_strategies:
            try:
                # 제외된 전략은 신호만 수집하고 로그는 생략
                is_excluded = strategy.name in excluded_strategies
                
                signal = strategy.analyze(self.data_collector)
                if signal:
                    score = signal['confidence']
                    signal_type = signal['signal']
                    entry_price = signal.get('entry_price', 0)
                    
                    if self.risk_manager.validate_signal(signal):
                        signals.append(signal)
                        if not is_excluded:
                            # 핵심 전략 표시
                            if strategy.name in core_strategies:
                                logger.info(f"⭐ {strategy.name:25s} | {signal_type:5s} | Score: {score:.2%} | 진입가: ${entry_price:.2f} [핵심]")
                            elif strategy.name in filter_strategies:
                                logger.info(f"🔍 {strategy.name:25s} | {signal_type:5s} | Score: {score:.2%} | 진입가: ${entry_price:.2f} [보조]")
                            else:
                                logger.info(f"✅ {strategy.name:25s} | {signal_type:5s} | Score: {score:.2%} | 진입가: ${entry_price:.2f}")
                    else:
                        if not is_excluded:
                            logger.info(f"⚠️  {strategy.name:25s} | {signal_type:5s} | Score: {score:.2%} | 검증 실패")
                else:
                    if not is_excluded:
                        logger.info(f"⚪ {strategy.name:25s} | 신호 없음 | Score: 0.00%")
            except Exception as e:
                logger.error(f"❌ {strategy.name:25s} | 분석 오류: {e}", exc_info=True)
        
        return signals
    
    def _analyze_range_mode(self):
        """횡보장 전략 분석 - 단일 신호로도 충분"""
        signals = []
        
        logger.info("")
        logger.info("📊 횡보장 모드 (Range Mode) - 횡보장 전략 5개 분석")
        logger.info("-" * 60)
        
        for strategy in self.range_strategies:
            try:
                signal = strategy.analyze(self.data_collector)
                if signal:
                    score = signal['confidence']
                    signal_type = signal['signal']
                    entry_price = signal.get('entry_price', 0)
                    
                    if self.risk_manager.validate_signal(signal):
                        signals.append(signal)
                        logger.info(f"✅ {strategy.name:25s} | {signal_type:5s} | Score: {score:.2%} | 진입가: ${entry_price:.2f}")
                    else:
                        logger.info(f"⚠️  {strategy.name:25s} | {signal_type:5s} | Score: {score:.2%} | 검증 실패")
                else:
                    logger.info(f"⚪ {strategy.name:25s} | 신호 없음 | Score: 0.00%")
            except Exception as e:
                logger.error(f"❌ {strategy.name:25s} | 분석 오류: {e}", exc_info=True)
        
        return signals
    
    def combine_signals(self, signals):
        """시장 모드에 따라 다른 진입 규칙 적용"""
        if not signals:
            return None
        
        # 현재 시장 모드에 따라 다른 로직 적용
        if self.current_market_mode == 'TREND':
            return self._combine_trend_signals(signals)
        elif self.current_market_mode == 'RANGE':
            return self._combine_range_signals(signals)
        else:
            logger.info("⚪ Neutral Mode: 거래 금지")
            return None
    
    def _combine_trend_signals(self, signals):
        """추세장 진입 규칙: 핵심 돌파 전략 3중주 2개 이상 필수 + 환경 필터"""
        if not signals:
            return None
        
        # 전략별 신호 추출
        btc_signal = self._get_signal_by_strategy(signals, 'BTC/ETH Correlation')
        cvd_signal = self._get_signal_by_strategy(signals, 'CVD Delta')
        squeeze_signal = self._get_signal_by_strategy(signals, 'Volatility Squeeze')
        fvg_signal = self._get_signal_by_strategy(signals, 'Orderblock FVG')
        liquidation_signal = self._get_signal_by_strategy(signals, 'Liquidation Spike')
        
        # Liquidation Spike 발생 시 보너스 점수로 활용 (즉시 진입은 하지 않음)
        liquidation_bonus = 0.1 if liquidation_signal else 0.0
        
        # LONG 진입 조합 체크
        # 핵심 3중주 (2개 이상 필수): Volatility Squeeze + Orderblock FVG + CVD Delta
        # 환경 확인 (보조): BTC/ETH Correlation
        squeeze_long = bool(squeeze_signal and squeeze_signal.get('signal') == 'LONG')
        fvg_long = bool(fvg_signal and fvg_signal.get('signal') == 'LONG')
        cvd_long = bool(cvd_signal and cvd_signal.get('signal') == 'LONG')
        btc_long = bool(btc_signal and btc_signal.get('signal') == 'LONG')
        liquidation_long = bool(liquidation_signal and liquidation_signal.get('signal') == 'SHORT')  # 숏 청산 → 롱 보너스
        
        # 핵심 돌파 전략 3중주 중 2개 이상 필요
        core_signals_count = sum([squeeze_long, fvg_long, cvd_long])
        if core_signals_count >= 2:
            # 활성 핵심 전략 수집
            active_strategies = []
            confidence_sum = 0
            confidence_count = 0
            entry_prices = []
            
            if squeeze_long:
                active_strategies.append('Volatility Squeeze')
                confidence_sum += squeeze_signal.get('confidence', 0)
                confidence_count += 1
                entry_prices.append(squeeze_signal.get('entry_price', 0))
            
            if fvg_long:
                active_strategies.append('Orderblock FVG')
                confidence_sum += fvg_signal.get('confidence', 0)
                confidence_count += 1
                entry_prices.append(fvg_signal.get('entry_price', 0))
            
            if cvd_long:
                active_strategies.append('CVD Delta')
                confidence_sum += cvd_signal.get('confidence', 0)
                confidence_count += 1
                entry_prices.append(cvd_signal.get('entry_price', 0))
            
            # 환경 확인 (보조): BTC/ETH Correlation
            if btc_long:
                active_strategies.append('BTC/ETH Correlation (보조)')
                confidence_sum += btc_signal.get('confidence', 0)
                confidence_count += 1
                entry_prices.append(btc_signal.get('entry_price', 0))
                logger.info("✅ 환경 확인 통과: BTC도 같은 방향(우상향)")
            else:
                logger.info("⚠️  환경 확인 미통과: BTC 방향 불일치 (보조 필터)")
            
            # 보너스: Liquidation Spike (돌파의 기폭제)
            if liquidation_long:
                active_strategies.append('Liquidation Spike (보너스)')
                logger.info("✅ 보너스: 청산 스파이크로 돌파 가속화")
            
            # 평균 신뢰도 계산
            avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0
            if liquidation_long:
                avg_confidence = min(avg_confidence + liquidation_bonus, 1.0)  # 최대 1.0으로 제한
            
            # 진입가 계산
            avg_entry = sum(entry_prices) / len(entry_prices) if entry_prices else 0
            
            # 손절가 계산
            stop_loss_signals = []
            if squeeze_long and squeeze_signal.get('stop_loss'):
                stop_loss_signals.append(squeeze_signal.get('stop_loss'))
            if fvg_long and fvg_signal.get('stop_loss'):
                stop_loss_signals.append(fvg_signal.get('stop_loss'))
            if cvd_long and cvd_signal.get('stop_loss'):
                stop_loss_signals.append(cvd_signal.get('stop_loss'))
            stop_loss = max(stop_loss_signals) if stop_loss_signals else None
            
            logger.info(f"🎯 추세장 롱 진입: 핵심 3중주 {core_signals_count}/3개 충족 (2개 이상 필수)")
            logger.info(f"   활성 전략: {', '.join(active_strategies)}")
            logger.info(f"   신뢰도: {avg_confidence:.2%}")
            return {
                'signal': 'LONG',
                'entry_price': avg_entry,
                'stop_loss': stop_loss,
                'confidence': avg_confidence,
                'strategy': 'Trend Mode - Breakout Confluence',
                'strategies': active_strategies
            }
        
        # SHORT 진입 조합 체크
        # 핵심 3중주 (2개 이상 필수): Volatility Squeeze + Orderblock FVG + CVD Delta
        # 환경 확인 (보조): BTC/ETH Correlation
        squeeze_short = bool(squeeze_signal and squeeze_signal.get('signal') == 'SHORT')
        fvg_short = bool(fvg_signal and fvg_signal.get('signal') == 'SHORT')
        cvd_short = bool(cvd_signal and cvd_signal.get('signal') == 'SHORT')
        btc_short = bool(btc_signal and btc_signal.get('signal') == 'SHORT')
        liquidation_short = bool(liquidation_signal and liquidation_signal.get('signal') == 'LONG')  # 롱 청산 → 숏 보너스
        
        # 핵심 돌파 전략 3중주 중 2개 이상 필요
        core_signals_count = sum([squeeze_short, fvg_short, cvd_short])
        if core_signals_count >= 2:
            # 활성 핵심 전략 수집
            active_strategies = []
            confidence_sum = 0
            confidence_count = 0
            entry_prices = []
            
            if squeeze_short:
                active_strategies.append('Volatility Squeeze')
                confidence_sum += squeeze_signal.get('confidence', 0)
                confidence_count += 1
                entry_prices.append(squeeze_signal.get('entry_price', 0))
            
            if fvg_short:
                active_strategies.append('Orderblock FVG')
                confidence_sum += fvg_signal.get('confidence', 0)
                confidence_count += 1
                entry_prices.append(fvg_signal.get('entry_price', 0))
            
            if cvd_short:
                active_strategies.append('CVD Delta')
                confidence_sum += cvd_signal.get('confidence', 0)
                confidence_count += 1
                entry_prices.append(cvd_signal.get('entry_price', 0))
            
            # 환경 확인 (보조): BTC/ETH Correlation
            if btc_short:
                active_strategies.append('BTC/ETH Correlation (보조)')
                confidence_sum += btc_signal.get('confidence', 0)
                confidence_count += 1
                entry_prices.append(btc_signal.get('entry_price', 0))
                logger.info("✅ 환경 확인 통과: BTC도 같은 방향(우하향)")
            else:
                logger.info("⚠️  환경 확인 미통과: BTC 방향 불일치 (보조 필터)")
            
            # 보너스: Liquidation Spike (돌파의 기폭제)
            if liquidation_short:
                active_strategies.append('Liquidation Spike (보너스)')
                logger.info("✅ 보너스: 청산 스파이크로 돌파 가속화")
            
            # 평균 신뢰도 계산
            avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0
            if liquidation_short:
                avg_confidence = min(avg_confidence + liquidation_bonus, 1.0)  # 최대 1.0으로 제한
            
            # 진입가 계산
            avg_entry = sum(entry_prices) / len(entry_prices) if entry_prices else 0
            
            # 손절가 계산
            stop_loss_signals = []
            if squeeze_short and squeeze_signal.get('stop_loss'):
                stop_loss_signals.append(squeeze_signal.get('stop_loss'))
            if fvg_short and fvg_signal.get('stop_loss'):
                stop_loss_signals.append(fvg_signal.get('stop_loss'))
            if cvd_short and cvd_signal.get('stop_loss'):
                stop_loss_signals.append(cvd_signal.get('stop_loss'))
            stop_loss = max(stop_loss_signals) if stop_loss_signals else None
            
            logger.info(f"🎯 추세장 숏 진입: 핵심 3중주 {core_signals_count}/3개 충족 (2개 이상 필수)")
            logger.info(f"   활성 전략: {', '.join(active_strategies)}")
            logger.info(f"   신뢰도: {avg_confidence:.2%}")
            return {
                'signal': 'SHORT',
                'entry_price': avg_entry,
                'stop_loss': stop_loss,
                'confidence': avg_confidence,
                'strategy': 'Trend Mode - Breakout Confluence',
                'strategies': active_strategies
            }
        
        # 진입 조건 미충족 시 상세 로그
        long_count = sum([squeeze_long, fvg_long, cvd_long])
        short_count = sum([squeeze_short, fvg_short, cvd_short])
        
        if long_count < 2 and short_count < 2:
            missing_strategies = []
            if not squeeze_long and not squeeze_short:
                missing_strategies.append('Volatility Squeeze')
            if not fvg_long and not fvg_short:
                missing_strategies.append('Orderblock FVG')
            if not cvd_long and not cvd_short:
                missing_strategies.append('CVD Delta')
            
            logger.info(f"⚠️  추세장 진입 조건 미충족: 핵심 3중주 2개 이상 필요 (현재: LONG {long_count}/3, SHORT {short_count}/3)")
            if missing_strategies:
                logger.info(f"   부족한 전략: {', '.join(missing_strategies)}")
        
        return None
    
    def _combine_range_signals(self, signals):
        """횡보장 진입 규칙: 박스권 경계 확인 필수 + 반전 신호 강화 택 1"""
        if not signals:
            return None
        
        # 전략별 신호 추출
        stoch_signal = self._get_signal_by_strategy(signals, 'Stoch RSI Mean Reversion')
        bollinger_signal = self._get_signal_by_strategy(signals, 'Bollinger Mean Reversion')
        cvd_fake_signal = self._get_signal_by_strategy(signals, 'CVD Fake Pressure')
        range_signal = self._get_signal_by_strategy(signals, 'Range Top/Bottom')
        vwap_signal = self._get_signal_by_strategy(signals, 'VWAP Deviation')
        
        # LONG 진입 조합 체크
        # 박스권 하단 확인 (필수): Bollinger + Range Top/Bottom
        # 반전 신호 강화 (택 1): Stoch RSI, CVD Fake Pressure, VWAP Deviation
        bollinger_long = bool(bollinger_signal and bollinger_signal.get('signal') == 'LONG')
        range_long = bool(range_signal and range_signal.get('signal') == 'LONG')
        stoch_long = bool(stoch_signal and stoch_signal.get('signal') == 'LONG')
        cvd_fake_long = bool(cvd_fake_signal and cvd_fake_signal.get('signal') == 'LONG')
        vwap_long = bool(vwap_signal and vwap_signal.get('signal') == 'LONG')
        
        # 박스권 하단 확인 (필수)
        box_bottom_confirmed = bollinger_long and range_long
        
        # 반전 신호 강화 (택 1)
        reversal_signal = stoch_long or cvd_fake_long or vwap_long
        
        # 진입 조건: 박스권 하단 확인 (필수) + 반전 신호 강화 (택 1)
        if box_bottom_confirmed and reversal_signal:
            # 활성 전략 수집
            active_strategies = ['Bollinger Mean Reversion', 'Range Top/Bottom']
            confidence_sum = bollinger_signal.get('confidence', 0) + range_signal.get('confidence', 0)
            confidence_count = 2
            entry_prices = [bollinger_signal.get('entry_price', 0), range_signal.get('entry_price', 0)]
            
            # 반전 신호 강화 (택 1)
            if stoch_long:
                active_strategies.append('Stoch RSI Mean Reversion')
                confidence_sum += stoch_signal.get('confidence', 0)
                confidence_count += 1
                entry_prices.append(stoch_signal.get('entry_price', 0))
                logger.info("✅ 반전 신호: Stoch RSI 골든크로스 시도")
            elif cvd_fake_long:
                active_strategies.append('CVD Fake Pressure')
                confidence_sum += cvd_fake_signal.get('confidence', 0)
                confidence_count += 1
                entry_prices.append(cvd_fake_signal.get('entry_price', 0))
                logger.info("✅ 반전 신호: CVD 급감으로 매도 물량 흡수")
            elif vwap_long:
                active_strategies.append('VWAP Deviation')
                confidence_sum += vwap_signal.get('confidence', 0)
                confidence_count += 1
                entry_prices.append(vwap_signal.get('entry_price', 0))
                logger.info("✅ 반전 신호: VWAP 대비 과도한 하락")
            
            avg_confidence = confidence_sum / confidence_count
            avg_entry = sum(entry_prices) / len(entry_prices)
            stop_loss = max([s.get('stop_loss', 0) for s in [bollinger_signal, range_signal] 
                           if s and s.get('stop_loss')], default=None)
            
            logger.info("🎯 횡보장 롱 진입: 박스권 하단 확인 (필수) + 반전 신호 강화")
            logger.info(f"   활성 전략: {', '.join(active_strategies)}")
            logger.info(f"   신뢰도: {avg_confidence:.2%}")
            return {
                'signal': 'LONG',
                'entry_price': avg_entry,
                'stop_loss': stop_loss,
                'confidence': avg_confidence,
                'strategy': 'Range Mode - Long Mean Reversion',
                'strategies': active_strategies
            }
        
        # SHORT 진입 조합 체크
        # 박스권 상단 확인 (필수): Bollinger + Range Top/Bottom
        # 반전 신호 강화 (택 1): Stoch RSI, CVD Fake Pressure, VWAP Deviation
        bollinger_short = bool(bollinger_signal and bollinger_signal.get('signal') == 'SHORT')
        range_short = bool(range_signal and range_signal.get('signal') == 'SHORT')
        stoch_short = bool(stoch_signal and stoch_signal.get('signal') == 'SHORT')
        cvd_fake_short = bool(cvd_fake_signal and cvd_fake_signal.get('signal') == 'SHORT')
        vwap_short = bool(vwap_signal and vwap_signal.get('signal') == 'SHORT')
        
        # 박스권 상단 확인 (필수)
        box_top_confirmed = bollinger_short and range_short
        
        # 반전 신호 강화 (택 1)
        reversal_signal_short = stoch_short or cvd_fake_short or vwap_short
        
        # 진입 조건: 박스권 상단 확인 (필수) + 반전 신호 강화 (택 1)
        if box_top_confirmed and reversal_signal_short:
            # 활성 전략 수집
            active_strategies = ['Bollinger Mean Reversion', 'Range Top/Bottom']
            confidence_sum = bollinger_signal.get('confidence', 0) + range_signal.get('confidence', 0)
            confidence_count = 2
            entry_prices = [bollinger_signal.get('entry_price', 0), range_signal.get('entry_price', 0)]
            
            # 반전 신호 강화 (택 1)
            if stoch_short:
                active_strategies.append('Stoch RSI Mean Reversion')
                confidence_sum += stoch_signal.get('confidence', 0)
                confidence_count += 1
                entry_prices.append(stoch_signal.get('entry_price', 0))
                logger.info("✅ 반전 신호: Stoch RSI 데드크로스 시도")
            elif cvd_fake_short:
                active_strategies.append('CVD Fake Pressure')
                confidence_sum += cvd_fake_signal.get('confidence', 0)
                confidence_count += 1
                entry_prices.append(cvd_fake_signal.get('entry_price', 0))
                logger.info("✅ 반전 신호: CVD 급증으로 매수 물량 흡수")
            elif vwap_short:
                active_strategies.append('VWAP Deviation')
                confidence_sum += vwap_signal.get('confidence', 0)
                confidence_count += 1
                entry_prices.append(vwap_signal.get('entry_price', 0))
                logger.info("✅ 반전 신호: VWAP 대비 과도한 상승")
            
            avg_confidence = confidence_sum / confidence_count
            avg_entry = sum(entry_prices) / len(entry_prices)
            stop_loss = max([s.get('stop_loss', 0) for s in [bollinger_signal, range_signal] 
                           if s and s.get('stop_loss')], default=None)
            
            logger.info("🎯 횡보장 숏 진입: 박스권 상단 확인 (필수) + 반전 신호 강화")
            logger.info(f"   활성 전략: {', '.join(active_strategies)}")
            logger.info(f"   신뢰도: {avg_confidence:.2%}")
            return {
                'signal': 'SHORT',
                'entry_price': avg_entry,
                'stop_loss': stop_loss,
                'confidence': avg_confidence,
                'strategy': 'Range Mode - Short Mean Reversion',
                'strategies': active_strategies
            }
        
        # 진입 조건 미충족 시 상세 로그
        if not box_bottom_confirmed and not box_top_confirmed:
            missing = []
            if not bollinger_long and not bollinger_short:
                missing.append('Bollinger Mean Reversion')
            if not range_long and not range_short:
                missing.append('Range Top/Bottom')
            logger.info(f"⚠️  횡보장 진입 조건 미충족: 박스권 경계 확인 필수 (부족: {', '.join(missing)})")
        elif box_bottom_confirmed and not reversal_signal:
            logger.info("⚠️  횡보장 롱 진입 조건 미충족: 반전 신호 강화 필요 (Stoch RSI, CVD Fake Pressure, VWAP Deviation 중 택 1)")
        elif box_top_confirmed and not reversal_signal_short:
            logger.info("⚠️  횡보장 숏 진입 조건 미충족: 반전 신호 강화 필요 (Stoch RSI, CVD Fake Pressure, VWAP Deviation 중 택 1)")
        
        return None
        
        # STEP 2: 필수 조합 체크
        # CVD 방향성 확인 (양전환/음전환)
        cvd_bullish = False
        cvd_bearish = False
        if cvd_signal:
            if cvd_signal['signal'] == 'LONG':
                cvd_bullish = True
            elif cvd_signal['signal'] == 'SHORT':
                cvd_bearish = True
        
        # 롱 필수 조합 체크
        long_required_combination = False
        if long_score >= 2:
            # (A) 저점 스윕 + CVD 양전환
            if sweep_signal and sweep_signal['signal'] == 'LONG' and cvd_bullish:
                long_required_combination = True
                logger.info("✅ 롱 필수 조합 (A): 저점 스윕 + CVD 양전환")
            
            # (B) FVG/OB 리테스트 + CVD 양전환
            elif fvg_signal and fvg_signal['signal'] == 'LONG' and cvd_bullish:
                long_required_combination = True
                logger.info("✅ 롱 필수 조합 (B): FVG/OB 리테스트 + CVD 양전환")
            
            # (C) 청산 스파이크 + 스윕
            elif liquidation_signal and sweep_signal:
                if (liquidation_signal['signal'] == 'LONG' and 
                    sweep_signal['signal'] == 'LONG'):
                    long_required_combination = True
                    logger.info("✅ 롱 필수 조합 (C): 청산 스파이크 + 저점 스윕")
        
        # 숏 필수 조합 체크
        short_required_combination = False
        if short_score >= 2:
            # (A) 고점 스윕 + CVD 음전환
            if sweep_signal and sweep_signal['signal'] == 'SHORT' and cvd_bearish:
                short_required_combination = True
                logger.info("✅ 숏 필수 조합 (A): 고점 스윕 + CVD 음전환")
            
            # (B) OB 리테스트 + CVD 음전환
            elif fvg_signal and fvg_signal['signal'] == 'SHORT' and cvd_bearish:
                short_required_combination = True
                logger.info("✅ 숏 필수 조합 (B): OB 리테스트 + CVD 음전환")
            
            # (C) 청산 스파이크 + 고점 스윕
            elif liquidation_signal and sweep_signal:
                if (liquidation_signal['signal'] == 'SHORT' and 
                    sweep_signal['signal'] == 'SHORT'):
                    short_required_combination = True
                    logger.info("✅ 숏 필수 조합 (C): 청산 스파이크 + 고점 스윕")
        
        # STEP 3: 조건 충족 시 진입
        if long_score >= 2 and long_required_combination:
            avg_confidence = sum(s['confidence'] for s in long_signals) / len(long_signals)
            avg_entry = sum(s['entry_price'] for s in long_signals) / len(long_signals)
            stop_loss = max([s.get('stop_loss', 0) for s in long_signals if s.get('stop_loss')], default=None)
            
            logger.info(f"🎯 하이브리드 롱 진입: 점수 {long_score}/{total_strategies}점 + 필수 조합 충족")
            logger.info(f"   활성 전략: {', '.join([s['strategy'] for s in long_signals])}")
            return {
                'signal': 'LONG',
                'entry_price': avg_entry,
                'stop_loss': stop_loss,
                'confidence': avg_confidence,
                'strategy_count': long_score,
                'strategies': [s['strategy'] for s in long_signals],
                'combination_rank': 1  # 하이브리드 시스템
            }
        
        elif short_score >= 2 and short_required_combination:
            avg_confidence = sum(s['confidence'] for s in short_signals) / len(short_signals)
            avg_entry = sum(s['entry_price'] for s in short_signals) / len(short_signals)
            stop_loss = min([s.get('stop_loss', float('inf')) for s in short_signals if s.get('stop_loss')], default=None)
            if stop_loss == float('inf'):
                stop_loss = None
            
            logger.info(f"🎯 하이브리드 숏 진입: 점수 {short_score}/{total_strategies}점 + 필수 조합 충족")
            logger.info(f"   활성 전략: {', '.join([s['strategy'] for s in short_signals])}")
            return {
                'signal': 'SHORT',
                'entry_price': avg_entry,
                'stop_loss': stop_loss,
                'confidence': avg_confidence,
                'strategy_count': short_score,
                'strategies': [s['strategy'] for s in short_signals],
                'combination_rank': 1  # 하이브리드 시스템
            }
        
        # 필수 조합 미충족
        if long_score >= 2:
            logger.info(f"⚠️  롱 점수 {long_score}/{total_strategies}점 충족했으나 필수 조합 미충족")
            logger.info(f"   활성 전략: {', '.join([s['strategy'] for s in long_signals])}")
            logger.info(f"   필요한 조합: (A) 저점 스윕 + CVD 양전환, (B) FVG/OB + CVD 양전환, (C) 청산 스파이크 + 저점 스윕")
        if short_score >= 2:
            logger.info(f"⚠️  숏 점수 {short_score}/{total_strategies}점 충족했으나 필수 조합 미충족")
            logger.info(f"   활성 전략: {', '.join([s['strategy'] for s in short_signals])}")
            logger.info(f"   필요한 조합: (A) 고점 스윕 + CVD 음전환, (B) OB + CVD 음전환, (C) 청산 스파이크 + 고점 스윕")
        
        return None
    
    def _get_signal_by_strategy(self, signals, strategy_name):
        """특정 전략의 신호 반환"""
        for s in signals:
            if s['strategy'] == strategy_name:
                return s
        return None
    
    def _get_signal_by_strategy(self, signals, strategy_name):
        """특정 전략의 신호 반환"""
        for s in signals:
            if s['strategy'] == strategy_name:
                return s
        return None
    
    def execute_trade(self, final_signal):
        """거래 실행"""
        try:
            use_spot = not self.client.use_futures
            side = 'BUY' if final_signal['signal'] == 'LONG' else 'SELL'
            
            # 스팟 거래에서 SHORT는 보유 자산 매도만 가능
            if use_spot and side == 'SELL':
                # 스팟 매도: 보유 자산 확인
                current_position = self.client.get_position(config.ETH_SYMBOL)
                if current_position is None or current_position['size'] == 0:
                    logger.warning("매도할 자산이 없습니다 (스팟 거래)")
                    return False
                
                # 보유 자산 전체 매도
                position_size = current_position['size']
                logger.info(f"거래 실행: {side} {position_size} {config.ETH_SYMBOL} (보유 자산 매도)")
                order = self.client.place_order(
                    symbol=config.ETH_SYMBOL,
                    side=side,
                    quantity=position_size,
                    order_type='MARKET'
                )
            else:
                # 선물 거래 또는 스팟 매수
                # 현재 포지션 확인
                current_position = self.client.get_position(config.ETH_SYMBOL)
                
                if current_position is not None:
                    # 기존 포지션이 있으면 청산
                    logger.info("기존 포지션 청산 중...")
                    self.client.close_position(config.ETH_SYMBOL)
                    time.sleep(1)
                
                # 포지션 크기 계산
                entry_price = final_signal['entry_price']
                stop_loss = final_signal.get('stop_loss')
                
                if use_spot and side == 'BUY':
                    # 스팟 매수: USDT 금액 계산
                    position_size = self.risk_manager.calculate_position_size(
                        entry_price, 
                        stop_loss,
                        use_spot=True
                    )
                    if position_size is None or position_size < 1:  # 최소 1 USDT
                        logger.warning("포지션 크기가 너무 작음")
                        return False
                    
                    logger.info(f"거래 실행: {side} {position_size} USDT worth of {config.ETH_SYMBOL} @ {entry_price}")
                    order = self.client.place_order(
                        symbol=config.ETH_SYMBOL,
                        side=side,
                        quantity=position_size,  # USDT 금액
                        order_type='MARKET',
                        quote_quantity=position_size
                    )
                else:
                    # 선물 거래: 코인 수량 계산
                    position_size = self.risk_manager.calculate_position_size(
                        entry_price, 
                        stop_loss,
                        use_spot=False
                    )
                    
                    if position_size is None or position_size < 0.001:
                        logger.warning("포지션 크기가 너무 작음")
                        return False
                    
                    logger.info(f"거래 실행: {side} {position_size} {config.ETH_SYMBOL} @ {entry_price}")
                    order = self.client.place_order(
                        symbol=config.ETH_SYMBOL,
                        side=side,
                        quantity=position_size,
                        order_type='MARKET'
                    )
            
            if order:
                logger.info(f"주문 성공: {order}")
                return True
            else:
                logger.error("주문 실패")
                return False
                
        except Exception as e:
            logger.error(f"거래 실행 실패: {e}")
            return False
    
    def _wait_for_next_candle(self):
        """다음 캔들까지 카운트다운하며 대기 (같은 줄에서 업데이트)"""
        # 현재 시간
        now = datetime.now()
        
        # 다음 3분 단위 시간 계산 (0분, 3분, 6분, 9분...)
        current_minute = now.minute
        next_minute = ((current_minute // 3) + 1) * 3
        
        if next_minute >= 60:
            # 다음 시간으로 넘어감
            next_candle_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            next_candle_time = now.replace(minute=next_minute, second=0, microsecond=0)
        
        # 남은 시간 계산
        remaining = (next_candle_time - now).total_seconds()
        
        # 카운트다운 표시 (같은 줄에서 업데이트)
        while remaining > 0:
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            print(f"\r⏰ 다음 캔들까지: {mins:02d}:{secs:02d} 남음", end='', flush=True)
            time.sleep(1)
            remaining -= 1
        
        print("\r" + " " * 50 + "\r", end='', flush=True)  # 줄 지우기
        logger.info("🕐 새 캔들 시작!")
    
    def monitor_positions(self):
        """포지션 모니터링 및 손절/익절"""
        try:
            # 스팟 거래에서는 자산 조회 권한이 없을 수 있으므로 예외 처리
            position = self.client.get_position(config.ETH_SYMBOL)
            if position is None:
                return
            
            current_price = self.client.get_ticker(config.ETH_SYMBOL)
            if current_price is None:
                return
            
            entry_price = position['entry_price']
            size = position['size']
            
            # 스팟 거래에서는 size가 양수만 가능 (SHORT 없음)
            if not self.client.use_futures:
                if size <= 0:
                    return
                side = 'LONG'
            else:
                side = 'LONG' if size > 0 else 'SHORT'
            
            # 손절 확인 (기본 0.2%)
            stop_loss_price = entry_price * (1 - config.STOP_LOSS_PERCENT / 100) if side == 'LONG' else entry_price * (1 + config.STOP_LOSS_PERCENT / 100)
            
            if self.risk_manager.should_stop_loss(entry_price, current_price, stop_loss_price, side):
                logger.info(f"손절 실행: {side} 포지션")
                self.client.close_position(config.ETH_SYMBOL)
                return
            
            # 익절 확인
            if self.risk_manager.should_take_profit(entry_price, current_price, side):
                logger.info(f"익절 고려: {side} 포지션, 수익률 계산 중...")
                # 익절은 더 보수적으로 설정 가능
            
        except Exception as e:
            # 스팟 거래에서 자산 조회 실패는 정상일 수 있음 (권한 없음)
            if not self.client.use_futures:
                # 디버그 레벨로만 로깅하여 경고 메시지 감소
                logger.debug(f"포지션 모니터링 스킵 (스팟 거래, 계정 조회 권한 없음)")
            else:
                logger.error(f"포지션 모니터링 실패: {e}")
    
    def run(self):
        """봇 실행"""
        logger.info("트레이딩 봇 시작")
        
        # 초기 데이터 로드
        if not self.update_data():
            logger.error("초기 데이터 로드 실패")
            return
        
        iteration = 0
        
        while True:
            try:
                iteration += 1
                logger.info(f"=== 반복 {iteration} ===")
                
                # 데이터 업데이트
                logger.info("📥 최신 3분봉 데이터 수집 중...")
                if not self.update_data():
                    logger.warning("데이터 업데이트 실패, 재시도 중...")
                    time.sleep(5)
                    continue
                
                # 현재 가격 확인
                current_eth_price = self.client.get_ticker(config.ETH_SYMBOL)
                current_btc_price = self.client.get_ticker(config.BTC_SYMBOL)
                if current_eth_price and current_btc_price:
                    logger.info(f"💰 현재 가격 - ETH: ${current_eth_price:.2f} | BTC: ${current_btc_price:.2f}")
                
                # 포지션 모니터링
                logger.info("👀 포지션 모니터링 중...")
                self.monitor_positions()
                
                # 전략 분석
                signals = self.analyze_strategies()
                
                if signals:
                    logger.info("🔍 신호 조합 분석 중...")
                    # 신호 결합
                    final_signal = self.combine_signals(signals)
                    
                    if final_signal:
                        rank = final_signal.get('combination_rank', 'N/A')
                        logger.info("")
                        logger.info("🎯" + "=" * 58)
                        logger.info(f"✅ 최종 거래 결정: {final_signal['signal']}")
                        logger.info(f"   진입가: ${final_signal['entry_price']:.2f}")
                        logger.info(f"   신뢰도: {final_signal['confidence']:.2%}")
                        logger.info(f"   조합 순위: {rank}위")
                        strategies_list = final_signal.get('strategies', [final_signal.get('strategy', 'Unknown')])
                        logger.info(f"   사용 전략: {', '.join(strategies_list)}")
                        if final_signal.get('stop_loss'):
                            logger.info(f"   손절가: ${final_signal['stop_loss']:.2f}")
                        logger.info("=" * 60)
                        logger.info("")
                        
                        # 거래 실행 (분석 모드에서는 비활성화)
                        if config.ENABLE_TRADING:
                            logger.info("💼 거래 실행 중...")
                            self.execute_trade(final_signal)
                        else:
                            logger.info("📊 분석 모드: 거래 실행 비활성화 (ENABLE_TRADING=False)")
                            logger.info("   신호만 분석하고 실제 거래는 수행하지 않습니다.")
                    else:
                        logger.info("⚠️  신호 조합 실패: 조건을 만족하는 조합이 없습니다")
                else:
                    logger.info("⚪ 거래 신호 없음: 다음 캔들 대기 중...")
                
                # 다음 캔들까지 카운트다운하며 대기
                self._wait_for_next_candle()
                
            except KeyboardInterrupt:
                logger.info("봇 종료 요청")
                break
            except Exception as e:
                logger.error(f"봇 실행 중 오류: {e}")
                time.sleep(10)


if __name__ == '__main__':
    bot = TradingBot()
    bot.run()
