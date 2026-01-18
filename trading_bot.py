"""
메인 트레이딩 봇
"""
import logging
import time
import sys
from datetime import datetime, timedelta
import config
from data_collector import DataCollector
from risk_manager import RiskManager
from binance_client import BinanceClient
from strategies import (
    LiquiditySweepStrategy,
    BTCEthCorrelationStrategy,
    CVDDeltaStrategy,
    VolatilitySqueezeStrategy,
    FundingRateStrategy,
    OrderblockFVGStrategy,
    LiquidationSpikeStrategy
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        self.data_collector = DataCollector()
        self.risk_manager = RiskManager()
        self.client = BinanceClient()
        
        # 전략 초기화
        self.strategies = []
        if config.STRATEGIES['liquidity_sweep']:
            self.strategies.append(LiquiditySweepStrategy())
        if config.STRATEGIES['btc_eth_correlation']:
            self.strategies.append(BTCEthCorrelationStrategy())
        if config.STRATEGIES['cvd_delta']:
            self.strategies.append(CVDDeltaStrategy())
        if config.STRATEGIES['volatility_squeeze']:
            self.strategies.append(VolatilitySqueezeStrategy())
        if config.STRATEGIES['funding_rate']:
            self.strategies.append(FundingRateStrategy())
        if config.STRATEGIES['orderblock_fvg']:
            self.strategies.append(OrderblockFVGStrategy())
        if config.STRATEGIES.get('liquidation_spike', False):
            self.strategies.append(LiquidationSpikeStrategy())
        
        logger.info(f"트레이딩 봇 초기화 완료 - 활성 전략: {len(self.strategies)}개")
    
    def update_data(self):
        """데이터 업데이트"""
        return self.data_collector.update_data()
    
    def analyze_strategies(self):
        """모든 전략 분석"""
        signals = []
        
        logger.info("=" * 60)
        logger.info("📊 전략 분석 시작 (3분봉 데이터 기준)")
        logger.info("=" * 60)
        
        # 데이터 상태 확인
        eth_data_len = len(self.data_collector.eth_data) if self.data_collector.eth_data is not None else 0
        btc_data_len = len(self.data_collector.btc_data) if self.data_collector.btc_data is not None else 0
        logger.info(f"📦 데이터 상태 - ETH: {eth_data_len}개 캔들, BTC: {btc_data_len}개 캔들")
        
        for strategy in self.strategies:
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
        
        logger.info("=" * 60)
        logger.info(f"📈 총 {len(signals)}개의 유효한 신호 발견")
        logger.info("=" * 60)
        
        return signals
    
    def combine_signals(self, signals):
        """하이브리드 진입 규칙: 점수 기반 + 필수 조합 체크"""
        if not signals:
            return None
        
        # 전략별 신호 추출
        btc_signal = self._get_signal_by_strategy(signals, 'BTC/ETH Correlation')
        cvd_signal = self._get_signal_by_strategy(signals, 'CVD Delta')
        sweep_signal = self._get_signal_by_strategy(signals, 'Liquidity Sweep')
        squeeze_signal = self._get_signal_by_strategy(signals, 'Volatility Squeeze')
        fvg_signal = self._get_signal_by_strategy(signals, 'Orderblock FVG')
        funding_signal = self._get_signal_by_strategy(signals, 'Funding Rate')
        liquidation_signal = self._get_signal_by_strategy(signals, 'Liquidation Spike')
        
        # STEP 1: 점수 계산 (7개 전략 중 3점 이상이면 진입 고려)
        # 7개 전략: Liquidity Sweep, BTC/ETH Correlation, CVD Delta, Volatility Squeeze, 
        #           Funding Rate, Orderblock FVG, Liquidation Spike
        long_signals = [s for s in signals if s['signal'] == 'LONG']
        short_signals = [s for s in signals if s['signal'] == 'SHORT']
        
        long_score = len(long_signals)
        short_score = len(short_signals)
        total_strategies = 7  # 총 7개 전략
        
        # 점수 부족 시 진입 불가 (7개 중 최소 2개 필요 - 개발 초기 모니터링용)
        if long_score < 2 and short_score < 2:
            logger.info(f"⚠️  점수 부족: LONG {long_score}/{total_strategies}점, SHORT {short_score}/{total_strategies}점 (최소 2점 필요)")
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
                        logger.info(f"   사용 전략: {', '.join(final_signal['strategies'])}")
                        if final_signal.get('stop_loss'):
                            logger.info(f"   손절가: ${final_signal['stop_loss']:.2f}")
                        logger.info("=" * 60)
                        logger.info("")
                        
                        # 거래 실행
                        logger.info("💼 거래 실행 중...")
                        self.execute_trade(final_signal)
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
