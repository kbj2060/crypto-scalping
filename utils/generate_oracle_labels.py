"""
Phase 0: Oracle Label Generator (Risk-Aware Version)
- "관망(Flat)"을 가르치기 위해 포지션 보유에 대한 페널티(Risk Penalty)를 도입
- 잦은 매매를 막기 위해 진입 장벽(Entry Cost)을 높임
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import config

def calculate_oracle_labels(df, fee=0.0005, leverage=1.0, 
                          risk_penalty=0.0002):
    """
    미래 가격을 보고 수수료를 고려한 최적의 포지션을 역산(Backtracking)
    
    States:
        -1: Short
         0: Flat (관망)
         1: Long
    
    Args:
        df: DataFrame with 'close' column
        fee: 실제 거래 수수료 (0.05%)
        leverage: 레버리지 (1.0 고정)
        risk_penalty: 포지션 유지 비용 (캔들당 0.02%). 
                     이 값이 높을수록 횡보장에서 Flat을 선호하게 됨.
    
    Returns:
        oracle_actions: np.array of optimal positions (-1, 0, 1)
    """
    prices = df['close'].values
    n = len(prices)
    
    # DP Tables: [Time, Current_Position] -> Max_Future_Value
    # Position index mapping: 0 -> Short(-1), 1 -> Flat(0), 2 -> Long(1)
    dp = np.zeros((n, 3))
    best_action = np.zeros((n, 3), dtype=int)  # 다음 스텝의 포지션 저장
    
    # 마지막 시점 초기화 (가치 0 - 모든 포지션 청산)
    dp[-1] = 0
    
    # Backward Induction
    print(f"⏳ Oracle Labeling (Risk Penalty: {risk_penalty*100:.3f}% per step)...")
    for t in tqdm(range(n - 2, -1, -1), desc="DP Backward"):
        current_price = prices[t]
        next_price = prices[t+1]
        
        # 수익률
        ret = (next_price - current_price) / current_price
        
        for curr_pos_idx in range(3):  # 0(Short), 1(Flat), 2(Long)
            curr_pos_val = curr_pos_idx - 1  # -1, 0, 1
            
            # 가능한 다음 포지션 탐색
            max_val = -float('inf')
            best_next_pos = 1  # Default Flat
            
            for next_pos_idx in range(3):  # 0(Short), 1(Flat), 2(Long)
                next_pos_val = next_pos_idx - 1
                
                # 1. 거래 비용 (실제 수수료)
                # 진입/스위칭 시에는 Oracle이 좀 더 보수적으로 판단하도록 수수료를 1.5배로 인식시킴 (Slippage 고려)
                trade_size = abs(next_pos_val - curr_pos_val)
                effective_fee = fee * 1.5 if trade_size > 0 else 0.0
                cost = trade_size * effective_fee * leverage
                
                # 2. 리스크 페널티 (Holding Cost)
                # 포지션을 잡고 있으면(Flat이 아니면) 매 틱마다 페널티 부과
                # "확실한 수익이 없으면 쉬어라"는 압박
                holding_cost = risk_penalty if next_pos_val != 0 else 0.0
                
                # 3. PnL 계산
                step_pnl = (ret * next_pos_val * leverage) - cost - holding_cost
                
                # 4. 미래 가치 (Bellman Equation)
                total_val = step_pnl + dp[t+1, next_pos_idx]
                
                if total_val > max_val:
                    max_val = total_val
                    best_next_pos = next_pos_idx
            
            dp[t, curr_pos_idx] = max_val
            best_action[t, curr_pos_idx] = best_next_pos

    # Forward Pass로 최적 경로(Trajectory) 추출
    oracle_actions = np.zeros(n, dtype=np.int8)
    curr_pos_idx = 1  # Start with Flat
    
    print("⏳ Extracting Optimal Path...")
    for t in tqdm(range(n), desc="Forward Path"):
        oracle_actions[t] = curr_pos_idx - 1  # Index to -1, 0, 1
        if t < n - 1:
            curr_pos_idx = best_action[t, curr_pos_idx]
        
    return oracle_actions

def main():
    # 1. 데이터 로드
    data_path = 'data/training_features.csv'
    if not os.path.exists(data_path):
        print(f"⚠️ {data_path} not found. Running prepare_training_data.py first...")
        import subprocess
        result = subprocess.run([sys.executable, 'utils/prepare_training_data.py'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to generate training features:\n{result.stderr}")
            return
        
    print(f"📂 Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True, date_format='%Y-%m-%d %H:%M:%S')
    print(f"✅ Loaded {len(df):,} rows")
    
    # 2. Oracle 계산
    # [설정] 리스크 페널티 0.02% (3분당)
    # 횡보장(수익률 < 0.02%)에서는 포지션을 청산하고 Flat으로 가게 만듦
    leverage = 1.0  # 레버리지 1.0 고정 (방향성만 학습)
    fee = 0.0005    # Binance Taker 수수료
    risk_penalty = 0.0002  # 포지션 보유 비용 (0.02% per step)
    
    print(f"\n🧮 Oracle Configuration:")
    print(f"  Fee: {fee*100:.3f}% (x1.5 for entry = {fee*1.5*100:.3f}%)")
    print(f"  Leverage: {leverage}")
    print(f"  Risk Penalty: {risk_penalty*100:.3f}% per step (Holding Cost)")
    
    oracle_actions = calculate_oracle_labels(df, fee=fee, leverage=leverage, risk_penalty=risk_penalty)
    
    # 3. 결과 저장
    df['oracle_action'] = oracle_actions
    
    save_path = 'data/training_features_with_oracle.csv'
    df.to_csv(save_path)
    print(f"\n✅ Oracle labels saved to {save_path}")
    
    # 4. 통계 출력
    print("\n📊 [Risk-Aware Oracle Statistics]")
    action_counts = pd.Series(oracle_actions).value_counts().sort_index()
    action_names = {-1: 'Short', 0: 'Flat', 1: 'Long'}
    
    total = len(oracle_actions)
    for action, count in action_counts.items():
        pct = (count / total) * 100
        print(f"  {action_names.get(action, action):>5}: {count:>6} ({pct:>5.2f}%)")
    
    # 5. 거래 횟수 확인
    trades = np.sum(np.abs(np.diff(oracle_actions)))
    print(f"\n📉 Total Trades: {trades:,} (Reduced Frequency)")
    
    # 6. 예상 수익 계산 (백테스트)
    print("\n💰 [Oracle Backtest]")
    current_pos = 0
    total_pnl = 0.0
    
    for i in range(len(df) - 1):
        next_pos = oracle_actions[i]
        price_return = (df['close'].iloc[i+1] - df['close'].iloc[i]) / df['close'].iloc[i]
        
        # 포지션 변경 비용
        trade_size = abs(next_pos - current_pos)
        cost = trade_size * fee * leverage
        
        # [수정] holding_cost 제거 - risk_penalty는 DP 유도용이지 실제 비용이 아님
        # 실제 거래에서는 포지션 유지 비용 없음
        pnl = (price_return * next_pos * leverage) - cost
        total_pnl += pnl
        
        current_pos = next_pos
    
    print(f"  Total PnL: {total_pnl*100:.2f}%")
    print(f"  Avg PnL per Trade: {(total_pnl/trades)*100 if trades > 0 else 0:.4f}%")
    print(f"\n🎯 Expected: Flat ratio 20-40%, Trades reduced by 70-80%")

if __name__ == "__main__":
    main()
