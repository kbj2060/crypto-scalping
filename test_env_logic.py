"""
환경 로직 검증 스크립트
AI를 끄고, 우리가 만든 가짜 봇이 돈을 벌 수 있는지 확인합니다.
Reward가 0이나 음수가 나오면 trading_env.py의 보상 계산식이 고장 난 것입니다.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from core.data_collector import DataCollector
from model.trading_env import TradingEnvironment

def test_env_logic():
    """환경 로직 검증"""
    print("=" * 60)
    print("🧪 환경 로직 검증 테스트 시작")
    print("=" * 60)
    
    # 1. 환경 설정
    collector = DataCollector(use_saved_data=True)
    if not collector.load_saved_data():
        print("❌ 데이터 로드 실패")
        return False
    
    # 전략은 필요 없음 (보상 계산만 테스트)
    env = TradingEnvironment(collector, strategies=[])
    
    # 2. 강제로 '상승장' 구간 찾기 (데이터를 미리 까봄)
    # 예: 가격이 10봉 연속 오르는 구간을 찾아서 테스트
    # get_candles는 최소 20개 캔들이 필요하므로 시작 인덱스는 최소 20 이상이어야 함
    prices = collector.eth_data['close'].values
    start_idx = -1
    min_start_idx = 20  # get_candles가 20개를 반환하려면 current_index >= 20 필요
    
    # 1% 이상 오르는 구간 찾기 (최소 20 이상부터 시작)
    for i in range(min_start_idx, len(prices) - 20):
        if i + 10 < len(prices):
            if prices[i + 10] > prices[i] * 1.01:  # 1% 이상 오르는 구간 발견
                start_idx = i
                break
    
    if start_idx == -1:
        print("❌ 상승 구간을 찾지 못했습니다. 데이터가 너무 짧거나 횡보장입니다.")
        return False
    
    print(f"🧪 테스트 구간: 인덱스 {start_idx} (진입) -> {start_idx + 10} (청산)")
    print(f"   진입 가격: ${prices[start_idx]:.2f}")
    print(f"   청산 가격: ${prices[start_idx + 10]:.2f}")
    print(f"   예상 수익률: {(prices[start_idx + 10] - prices[start_idx]) / prices[start_idx] * 100:.2f}%")
    print()
    
    # 3. 가짜 트레이딩 실행
    collector.current_index = start_idx
    env.prev_pnl = 0
    total_reward = 0
    entry_price = prices[start_idx]
    
    # Step 1: LONG 진입
    obs = env.get_observation(position_info=[0, 0, 0])
    if obs is None:
        print("❌ 관측 생성 실패")
        return False
    
    # action 1 (LONG) - 진입 시점 보상은 보통 0
    reward = env.calculate_reward(0, False, 0, 0)
    total_reward += reward
    print(f"Step 1 (LONG 진입): Reward {reward:.4f}")
    
    # Step 2: 10스텝 HOLD (가격 상승 중)
    prev_pnl = 0.0
    for step in range(9):
        collector.current_index += 1
        if collector.current_index >= len(prices):
            break
        
        # 가격이 올랐으므로 PnL 상승 -> 보상 양수여야 함
        current_price = prices[collector.current_index]
        pnl = (current_price - entry_price) / entry_price
        pnl_change = pnl - prev_pnl  # 이전 스텝 대비 변화량
        
        reward = env.calculate_reward(pnl, False, step + 1, pnl_change)
        total_reward += reward
        print(f"Step {step + 2} (HOLD): PnL {pnl:.4f} ({pnl*100:.2f}%), PnL Change {pnl_change:.4f}, Reward {reward:.4f}")
        prev_pnl = pnl
    
    # Step 3: 청산
    collector.current_index += 1
    if collector.current_index < len(prices):
        final_price = prices[collector.current_index]
        final_pnl = (final_price - entry_price) / entry_price
        final_pnl_change = final_pnl - prev_pnl
        
        reward = env.calculate_reward(final_pnl, True, 10, final_pnl_change)
        total_reward += reward
        print(f"Step Final (청산): PnL {final_pnl:.4f} ({final_pnl*100:.2f}%), Reward {reward:.4f}")
    
    print()
    print("=" * 60)
    print(f"💰 총 보상 합계: {total_reward:.4f}")
    print("=" * 60)
    
    if total_reward > 0:
        print("✅ 환경 로직 정상: 돈을 벌면 보상을 줍니다.")
        return True
    else:
        print("❌ 환경 로직 오류: 수익이 났는데 보상이 0 이하입니다.")
        print("   trading_env.py의 calculate_reward 함수를 확인하세요.")
        return False


if __name__ == '__main__':
    try:
        success = test_env_logic()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
