"""
전략 신호 사전 계산 스크립트
Elite 8 전략의 모든 신호를 미리 계산하여 CSV 파일로 저장합니다.
이를 통해 PPO 및 TD3 학습 시 신호 계산 시간을 절약할 수 있습니다.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

# 상위 폴더를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import (
    WhaleSentimentDivergence, LiquidationSqueezeHunter,
    OrderblockFVGStrategy, NetTakerFlowStrategy,
    BTCEthCorrelation, VolatilitySqueeze, VWAPDeviation, HMAMomentum,
)

# 로깅 설정
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/generate_signals.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_strategy_signals(input_file='data/integrated_eth_3m_data.csv', 
                             output_file='data/cached_strategies.csv'):
    """
    Elite 8 전략 신호를 사전 계산하여 CSV로 저장
    
    Args:
        input_file: 입력 데이터 파일 (통합된 ETH 데이터)
        output_file: 출력 신호 파일
    
    Returns:
        bool: 성공 여부
    """
    logger.info("=" * 60)
    logger.info("🎯 전략 신호 사전 계산 시작 (Elite 8)")
    logger.info("=" * 60)
    
    # 1. 데이터 로드
    if not os.path.exists(input_file):
        logger.error(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        logger.error("먼저 utils/future_csv_merger.py를 실행하여 통합 데이터를 생성하세요.")
        return False
    
    logger.info(f"📥 데이터 로드 중: {input_file}")
    df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
    logger.info(f"✅ 데이터 로드 완료: {len(df):,}개 캔들, {len(df.columns)}개 컬럼")
    logger.info(f"   기간: {df.index.min()} ~ {df.index.max()}")
    
    # 2. Elite 8 전략 초기화
    strategies = [
        WhaleSentimentDivergence(),     # strategy_0
        LiquidationSqueezeHunter(),     # strategy_1
        OrderblockFVGStrategy(),        # strategy_2
        NetTakerFlowStrategy(),         # strategy_3
        BTCEthCorrelation(),            # strategy_4
        VolatilitySqueeze(),            # strategy_5
        VWAPDeviation(),                # strategy_6
        HMAMomentum(),                  # strategy_7
    ]
    
    logger.info(f"🧠 전략 초기화 완료: {len(strategies)}개 (Elite 8)")
    for i, strat in enumerate(strategies):
        logger.info(f"   [{i}] {strat.name}")
    
    # 3. 신호 계산
    logger.info("")
    logger.info("⚙️  전략 신호 계산 중...")
    
    signals_dict = {}
    
    for i, strategy in enumerate(strategies):
        logger.info(f"   계산 중: strategy_{i} ({strategy.name})")
        signals = []
        
        # 각 행에 대해 신호 계산 (진행 표시줄 포함)
        for idx, row in tqdm(df.iterrows(), 
                            total=len(df), 
                            desc=f"Strategy {i}",
                            leave=False,
                            ncols=80):
            try:
                signal = strategy.generate_signal(row, df)
                signals.append(signal)
            except Exception as e:
                # 오류 발생 시 중립 신호(0) 사용
                logger.debug(f"신호 계산 실패 (idx={idx}): {e}")
                signals.append(0)
        
        # 결과 저장
        signals_dict[f'strategy_{i}'] = signals
        
        # 통계 정보
        signals_array = np.array(signals)
        long_count = np.sum(signals_array == 1)
        short_count = np.sum(signals_array == -1)
        neutral_count = np.sum(signals_array == 0)
        
        logger.info(f"   ✅ strategy_{i} 완료: Long={long_count:,} ({long_count/len(signals)*100:.1f}%), "
                   f"Short={short_count:,} ({short_count/len(signals)*100:.1f}%), "
                   f"Neutral={neutral_count:,} ({neutral_count/len(signals)*100:.1f}%)")
    
    # 4. DataFrame 생성
    logger.info("")
    logger.info("📊 DataFrame 생성 중...")
    signals_df = pd.DataFrame(signals_dict, index=df.index)
    
    # 5. CSV 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    signals_df.to_csv(output_file)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✅ 전략 신호 저장 완료: {output_file}")
    logger.info(f"   Shape: {signals_df.shape}")
    logger.info(f"   Columns: {signals_df.columns.tolist()}")
    logger.info(f"   메모리 크기: {signals_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    logger.info("=" * 60)
    logger.info("")
    logger.info("이제 다음 명령으로 학습을 시작할 수 있습니다:")
    logger.info("  - PPO: python macroHFT/train_ppo.py")
    logger.info("  - TD3: python TD3/train_td3.py")
    logger.info("")
    logger.info("전략 신호는 자동으로 로드됩니다 (빠른 학습 시작!)")
    logger.info("=" * 60)
    
    return True


if __name__ == '__main__':
    try:
        success = generate_strategy_signals()
        if not success:
            logger.error("전략 신호 생성 실패")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n작업 중단됨")
        sys.exit(0)
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        sys.exit(1)
