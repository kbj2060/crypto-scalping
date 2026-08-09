"""
학습 데이터 준비 파이프라인 (통합 스크립트)
1. 통합 데이터 확인/로드
2. 피처 엔지니어링
3. 전략 신호 사전 계산
4. 학습용 CSV 파일 저장
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# 상위 폴더를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.engineering import FeatureEngineer
from features.schema import build_active_feature_keep, prune_to_active_feature_keep
from strategies import (
    WhaleSentimentDivergence, LiquidationSqueezeHunter,
    OrderblockFVGStrategy, NetTakerFlowStrategy
)

# 로깅 설정
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/prepare_training_data.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def prepare_training_data():
    """
    학습 데이터 준비 전체 파이프라인
    """
    logger.info("=" * 80)
    logger.info("🚀 학습 데이터 준비 파이프라인 시작")
    logger.info("=" * 80)
    
    # 1. 통합 데이터 로드
    logger.info("")
    logger.info("📥 Step 1: 통합 데이터 로드")
    logger.info("-" * 80)
    
    integrated_file = 'data/integrated_eth_3m_data.csv'
    if not os.path.exists(integrated_file):
        logger.error(f"❌ 통합 데이터 파일이 없습니다: {integrated_file}")
        logger.error("먼저 utils/future_csv_merger.py를 실행하여 통합 데이터를 생성하세요.")
        return False
    
    df = pd.read_csv(integrated_file, index_col='timestamp', parse_dates=True)
    logger.info(f"✅ 데이터 로드 완료: {len(df):,}개 캔들, {len(df.columns)}개 컬럼")
    logger.info(f"   기간: {df.index.min()} ~ {df.index.max()}")
    
    # 2. 피처 엔지니어링 (필요한 경우)
    logger.info("")
    logger.info("🔧 Step 2: 피처 엔지니어링 확인")
    logger.info("-" * 80)
    
    active_keep = build_active_feature_keep(include_entry_price=False)
    # 필수 피처가 있는지 확인 (실사용 스키마 기준)
    missing_features = [col for col in active_keep if col not in df.columns and col != "timestamp"]
    
    if missing_features:
        logger.info(f"⚠️  누락된 피처 발견: {len(missing_features)}개")
        logger.info("   피처 엔지니어링 실행 중...")
        
        # BTC 데이터 로드 (피처 엔지니어링에 필요)
        btc_file = 'data/btc_3m_1year.csv'
        if not os.path.exists(btc_file):
            logger.warning(f"⚠️  BTC 데이터 없음: {btc_file}")
            logger.warning("BTC 관련 피처는 기본값으로 채워집니다.")
            btc_df = None
        else:
            btc_df = pd.read_csv(btc_file, index_col='timestamp', parse_dates=True)
            logger.info(f"✅ BTC 데이터 로드: {len(btc_df):,}개 캔들")
        
        # 피처 엔지니어링 실행
        engineer = FeatureEngineer(keep_only_active=True, include_entry_price=False)
        
        # ETH 데이터에 timestamp 컬럼 추가 (process 메서드가 필요로 함)
        df_for_fe = df.reset_index()
        
        if btc_df is not None:
            btc_for_fe = btc_df.reset_index()
            try:
                df = engineer.process(df_for_fe, btc_for_fe)
                logger.info(f"✅ 피처 엔지니어링 완료: {len(df.columns)}개 컬럼")
            except Exception as e:
                logger.error(f"❌ 피처 엔지니어링 실패: {e}")
                logger.info("   기존 데이터로 계속 진행합니다...")
        else:
            logger.warning("BTC 데이터 없이 진행 (일부 피처 누락 가능)")
    else:
        logger.info(f"✅ 모든 필수 피처 존재(실사용 스키마): {len(active_keep)}개")

    df = prune_to_active_feature_keep(
        df if isinstance(df, pd.DataFrame) else pd.DataFrame(df),
        include_entry_price=False,
        extra_keep=["timestamp"],
    )
    
    # 3. 전략 신호 계산
    logger.info("")
    logger.info("🎯 Step 3: 전략 신호 계산 (Elite 8)")
    logger.info("-" * 80)
    
    strategies = [
        WhaleSentimentDivergence(),     # strategy_0
        LiquidationSqueezeHunter(),     # strategy_1
        OrderblockFVGStrategy(),        # strategy_2
        NetTakerFlowStrategy(),         # strategy_3
    ]
    
    logger.info(f"🧠 전략 초기화: {len(strategies)}개")
    
    signals_dict = {}
    for i, strategy in enumerate(strategies):
        logger.info(f"   계산 중: strategy_{i} ({strategy.name})...")
        signals = []
        
        # tqdm으로 진행률 표시
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Strategy {i}", ncols=100, leave=False):
            try:
                signal = strategy.generate_signal(row, df)
                signals.append(signal)
            except Exception:
                signals.append(0)
        
        signals_dict[f'strategy_{i}'] = signals
        
        # 통계
        signals_array = np.array(signals)
        long_pct = np.sum(signals_array == 1) / len(signals) * 100
        short_pct = np.sum(signals_array == -1) / len(signals) * 100
        logger.info(f"      ✅ Long: {long_pct:.1f}% | Short: {short_pct:.1f}%")
    
    signals_df = pd.DataFrame(signals_dict, index=df.index)
    
    # 4. 파일 저장
    logger.info("")
    logger.info("💾 Step 4: 파일 저장")
    logger.info("-" * 80)
    
    os.makedirs('data', exist_ok=True)
    
    # 4-1. 통합 피처 저장 (원본 컬럼 + 신호 컬럼)
    training_features_file = 'data/training_features.csv'
    df.to_csv(training_features_file)
    logger.info(f"✅ 학습 피처 저장: {training_features_file}")
    logger.info(f"   Shape: {df.shape}")
    
    # 4-2. 전략 신호만 별도 저장
    cached_strategies_file = 'data/cached_strategies.csv'
    signals_df.to_csv(cached_strategies_file)
    logger.info(f"✅ 전략 신호 저장: {cached_strategies_file}")
    logger.info(f"   Shape: {signals_df.shape}")
    
    # 5. 완료 메시지
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ 학습 데이터 준비 완료!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("생성된 파일:")
    logger.info(f"  1️⃣  {training_features_file}")
    logger.info(f"      - Shape: {df.shape}")
    logger.info(f"      - 컬럼: {len(df.columns)}개")
    logger.info("")
    logger.info(f"  2️⃣  {cached_strategies_file}")
    logger.info(f"      - Shape: {signals_df.shape}")
    logger.info(f"      - 전략: {len(strategies)}개 (Elite 8)")
    logger.info("")
    logger.info("이제 학습을 시작할 수 있습니다:")
    logger.info("  - PPO:  python macroHFT/train_ppo.py")
    logger.info("  - TD3:  python TD3/train_td3.py")
    logger.info("")
    logger.info("💡 전략 신호가 자동으로 로드되어 학습이 빠르게 시작됩니다!")
    logger.info("=" * 80)
    
    return True


if __name__ == '__main__':
    try:
        success = prepare_training_data()
        if not success:
            logger.error("데이터 준비 실패")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n작업 중단됨")
        sys.exit(0)
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        sys.exit(1)
