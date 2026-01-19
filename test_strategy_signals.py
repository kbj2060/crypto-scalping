"""
전략 신호 발생 테스트 스크립트
1년치 데이터를 사용하여 각 전략이 얼마나 신호를 발생시키는지 분석합니다.
"""
import os
import sys
import pandas as pd
import logging
from datetime import datetime
from collections import defaultdict

# 상위 폴더를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core import DataCollector, BinanceClient
from strategies import (
    BTCEthCorrelationStrategy,
    VolatilitySqueezeStrategy,
    OrderblockFVGStrategy,
    HMAMomentumStrategy,
    BollingerMeanReversionStrategy,
    VWAPDeviationStrategy,
    RangeTopBottomStrategy,
    StochRSIMeanReversionStrategy,
    MFIMomentumStrategy,
    CMFDivergenceStrategy
)

# 로깅 설정
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_strategy_signals.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class StrategySignalTester:
    """전략 신호 발생 테스트 클래스"""
    
    def __init__(self):
        # 저장된 데이터 사용
        self.data_collector = DataCollector(use_saved_data=True)
        self.client = BinanceClient()
        
        # 전략 초기화
        self.breakout_strategies = []
        self.range_strategies = []
        
        # 폭발장 전략
        if config.STRATEGIES.get('btc_eth_correlation', False):
            self.breakout_strategies.append(BTCEthCorrelationStrategy())
        if config.STRATEGIES.get('volatility_squeeze', False):
            self.breakout_strategies.append(VolatilitySqueezeStrategy())
        if config.STRATEGIES.get('orderblock_fvg', False):
            self.breakout_strategies.append(OrderblockFVGStrategy())
        if config.STRATEGIES.get('hma_momentum', False):
            self.breakout_strategies.append(HMAMomentumStrategy())
        if config.STRATEGIES.get('mfi_momentum', False):
            self.breakout_strategies.append(MFIMomentumStrategy())
        
        # 횡보장 전략
        if config.STRATEGIES.get('bollinger_mean_reversion', False):
            self.range_strategies.append(BollingerMeanReversionStrategy())
        if config.STRATEGIES.get('vwap_deviation', False):
            self.range_strategies.append(VWAPDeviationStrategy())
        if config.STRATEGIES.get('range_top_bottom', False):
            self.range_strategies.append(RangeTopBottomStrategy())
        if config.STRATEGIES.get('stoch_rsi_mean_reversion', False):
            self.range_strategies.append(StochRSIMeanReversionStrategy())
        if config.STRATEGIES.get('cmf_divergence', False):
            self.range_strategies.append(CMFDivergenceStrategy())
        
        self.strategies = self.breakout_strategies + self.range_strategies
        
        if len(self.strategies) == 0:
            raise ValueError("활성화된 전략이 없습니다. config.py에서 전략을 활성화하세요.")
        
        logger.info(f"전략 초기화 완료: {len(self.strategies)}개 전략")
        logger.info(f"  - 폭발장 전략: {len(self.breakout_strategies)}개")
        logger.info(f"  - 횡보장 전략: {len(self.range_strategies)}개")
        
        # 통계 저장
        self.signal_stats = defaultdict(lambda: {
            'total': 0,
            'long': 0,
            'short': 0,
            'signals_by_candle': defaultdict(int)
        })
        
    def test_strategies(self, start_index=40, end_index=None, step=1):
        """
        전략 신호 발생 테스트
        
        Args:
            start_index: 시작 인덱스 (lookback 고려)
            end_index: 종료 인덱스 (None이면 전체)
            step: 스텝 크기 (1이면 모든 캔들, 10이면 10개마다)
        """
        logger.info("=" * 80)
        logger.info("📊 전략 신호 발생 테스트 시작")
        logger.info("=" * 80)
        
        # 데이터 확인
        if self.data_collector.eth_data is None or len(self.data_collector.eth_data) == 0:
            logger.error("저장된 데이터가 없습니다. model/collect_training_data.py를 먼저 실행하세요.")
            return
        
        total_candles = len(self.data_collector.eth_data)
        if end_index is None:
            end_index = total_candles
        
        logger.info(f"테스트 범위: 인덱스 {start_index} ~ {end_index} (총 {end_index - start_index}개 캔들)")
        logger.info(f"스텝 크기: {step} (총 {(end_index - start_index) // step}개 테스트)")
        logger.info("")
        
        # 인덱스 초기화
        self.data_collector.current_index = start_index
        
        test_count = 0
        processed_count = 0
        
        for idx in range(start_index, end_index, step):
            try:
                # 현재 인덱스 설정
                self.data_collector.current_index = idx
                
                # 각 전략 테스트
                for strategy in self.strategies:
                    try:
                        result = strategy.analyze(self.data_collector)
                        
                        if result and 'signal' in result:
                            signal_type = result['signal']
                            strategy_name = strategy.name
                            
                            # 통계 업데이트
                            self.signal_stats[strategy_name]['total'] += 1
                            if signal_type == 'LONG':
                                self.signal_stats[strategy_name]['long'] += 1
                            elif signal_type == 'SHORT':
                                self.signal_stats[strategy_name]['short'] += 1
                            
                            # 캔들별 신호 카운트
                            self.signal_stats[strategy_name]['signals_by_candle'][idx] += 1
                            
                    except Exception as e:
                        logger.debug(f"전략 {strategy.name} 분석 실패 (인덱스 {idx}): {e}")
                        continue
                
                processed_count += 1
                
                # 진행 상황 출력 (1000개마다)
                if processed_count % 1000 == 0:
                    logger.info(f"진행 중... {processed_count}개 캔들 처리 완료")
                
            except Exception as e:
                logger.error(f"인덱스 {idx} 처리 중 오류: {e}")
                continue
        
        logger.info("")
        logger.info(f"✅ 테스트 완료: {processed_count}개 캔들 처리")
        logger.info("")
        
    def print_statistics(self):
        """통계 출력"""
        logger.info("=" * 80)
        logger.info("📈 전략별 신호 발생 통계")
        logger.info("=" * 80)
        logger.info("")
        
        # 전체 통계
        total_signals = sum(stats['total'] for stats in self.signal_stats.values())
        total_long = sum(stats['long'] for stats in self.signal_stats.values())
        total_short = sum(stats['short'] for stats in self.signal_stats.values())
        
        logger.info(f"전체 신호 발생: {total_signals}개")
        logger.info(f"  - 롱 신호: {total_long}개 ({total_long/total_signals*100:.2f}%)" if total_signals > 0 else "  - 롱 신호: 0개")
        logger.info(f"  - 숏 신호: {total_short}개 ({total_short/total_signals*100:.2f}%)" if total_signals > 0 else "  - 숏 신호: 0개")
        logger.info("")
        
        # 전략별 통계
        logger.info("전략별 상세 통계:")
        logger.info("-" * 80)
        logger.info(f"{'전략명':<30} {'총 신호':<10} {'롱':<10} {'숏':<10} {'롱%':<10} {'숏%':<10}")
        logger.info("-" * 80)
        
        # 모든 전략에 대해 통계 수집 (신호가 없는 전략도 포함)
        all_strategy_stats = []
        for strategy in self.strategies:
            strategy_name = strategy.name
            stats = self.signal_stats.get(strategy_name, {
                'total': 0,
                'long': 0,
                'short': 0
            })
            all_strategy_stats.append((strategy_name, stats))
        
        # 신호 수로 정렬
        sorted_strategies = sorted(
            all_strategy_stats,
            key=lambda x: x[1]['total'],
            reverse=True
        )
        
        for strategy_name, stats in sorted_strategies:
            total = stats['total']
            long_count = stats['long']
            short_count = stats['short']
            
            long_pct = (long_count / total * 100) if total > 0 else 0
            short_pct = (short_count / total * 100) if total > 0 else 0
            
            logger.info(
                f"{strategy_name:<30} {total:<10} {long_count:<10} {short_count:<10} "
                f"{long_pct:<10.2f} {short_pct:<10.2f}"
            )
        
        logger.info("-" * 80)
        logger.info("")
        
        # 신호 밀도 분석
        logger.info("신호 밀도 분석:")
        logger.info("-" * 80)
        
        if self.data_collector.eth_data is not None:
            total_candles = len(self.data_collector.eth_data)
            logger.info(f"전체 캔들 수: {total_candles}개")
            
            for strategy_name, stats in sorted_strategies:
                total = stats['total']
                # 신호가 발생한 고유 캔들 수
                signals_by_candle = self.signal_stats.get(strategy_name, {}).get('signals_by_candle', {})
                unique_candles = len(signals_by_candle)
                
                if total > 0:
                    signal_density = (unique_candles / total_candles * 100) if total_candles > 0 else 0
                    avg_signals_per_candle = total / unique_candles if unique_candles > 0 else 0
                    
                    logger.info(
                        f"{strategy_name:<30}: "
                        f"신호 발생 캔들 {unique_candles}개 ({signal_density:.2f}%), "
                        f"캔들당 평균 {avg_signals_per_candle:.2f}개 신호"
                    )
                else:
                    logger.info(
                        f"{strategy_name:<30}: "
                        f"신호 발생 캔들 0개 (0.00%), "
                        f"캔들당 평균 0.00개 신호"
                    )
        
        logger.info("")
        
        # 균형도 분석
        logger.info("신호 균형도 분석:")
        logger.info("-" * 80)
        
        for strategy_name, stats in sorted_strategies:
            total = stats['total']
            long_count = stats['long']
            short_count = stats['short']
            
            if total > 0:
                balance_ratio = min(long_count, short_count) / max(long_count, short_count) if max(long_count, short_count) > 0 else 0
                balance_status = "균형" if balance_ratio > 0.7 else "불균형" if balance_ratio > 0.3 else "매우 불균형"
                
                logger.info(
                    f"{strategy_name:<30}: "
                    f"균형 비율 {balance_ratio:.2f} ({balance_status})"
                )
            else:
                logger.info(
                    f"{strategy_name:<30}: "
                    f"신호 없음 (분석 불가)"
                )
        
        logger.info("")
        logger.info("=" * 80)


def main():
    """메인 함수"""
    try:
        tester = StrategySignalTester()
        
        # 1주일치 데이터 테스트
        # 3분 캔들 기준: 1일 = 480개, 1주일 = 3360개
        start_index = 40
        candles_per_day = 480  # 3분 캔들: 24시간 * 60분 / 3
        candles_per_week = candles_per_day * 7  # 1주일
        end_index = start_index + candles_per_week
        
        logger.info("1주일치 데이터 테스트 시작")
        logger.info(f"테스트 범위: 인덱스 {start_index} ~ {end_index} (총 {candles_per_week}개 캔들, 약 1주일)")
        logger.info("")
        
        # 1주일치 테스트
        tester.test_strategies(start_index=start_index, end_index=end_index, step=1)
        
        # 통계 출력
        tester.print_statistics()
        
        logger.info("✅ 모든 테스트 완료!")
        
    except KeyboardInterrupt:
        logger.info("테스트 중단")
    except Exception as e:
        logger.error(f"테스트 실패: {e}", exc_info=True)


if __name__ == '__main__':
    main()
