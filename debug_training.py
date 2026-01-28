"""
긴급 진단 스크립트 (debug_training.py)
AI의 뇌 속을 직접 들여다보는 코드입니다.
"""
import torch
import numpy as np
import pandas as pd
from model.train_ppo import PPOTrainer
from model import config

def debug_agent():
    print("🚨 [긴급 진단] PPO 에이전트 상태 점검 시작...\n")
    
    # 1. 트레이너 초기화 (데이터 로드 포함)
    trainer = PPOTrainer(enable_visualization=False)
    env = trainer.env
    agent = trainer.agent
    
    # 2. 데이터 상태 점검 (눈 검사)
    print("-" * 50)
    print("1️⃣ 입력 데이터(State) 점검")
    
    # 임의의 시점에서 관측값 가져오기
    idx = 1000
    env.collector.current_index = idx
    # 가상의 포지션 정보
    pos_info = [0.0, 0.0, 0.0] 
    
    state = env.get_observation(position_info=pos_info, current_index=idx)
    
    if state is None:
        print("❌ 오류: State가 None입니다. 데이터 로드 실패.")
        return

    obs_seq, obs_info = state
    
    print(f"   - Sequence Shape: {obs_seq.shape}")
    print(f"   - Info Shape: {obs_info.shape}")
    print(f"   - Seq Mean: {obs_seq.mean().item():.4f} | Max: {obs_seq.max().item():.4f} | Min: {obs_seq.min().item():.4f}")
    
    if torch.isnan(obs_seq).any():
        print("❌ 치명적 오류: 입력 데이터에 NaN(결측치)이 있습니다!")
    elif obs_seq.abs().sum() == 0:
        print("❌ 치명적 오류: 입력 데이터가 전부 0입니다! (Scaler 고장)")
    else:
        print("✅ 입력 데이터 정상 (값 분포 확인됨)")

    # 3. 신경망 출력 점검 (뇌 검사)
    print("\n" + "-" * 50)
    print("2️⃣ 신경망 출력(Action Probability) 점검")
    
    agent.model.eval()
    with torch.no_grad():
        # 배치를 하나 만들어서 넣어봄
        obs_seq = obs_seq.to(agent.device)
        obs_info = obs_info.to(agent.device)
        
        # LSTM 상태 초기화
        agent.reset_episode_states()
        
        # return_states=True로 호출 (3개 반환값)
        probs, value, states = agent.model(obs_seq, obs_info, states=None, return_states=True)
        
    probs_np = probs.cpu().numpy()[0]
    value_np = value.cpu().numpy()[0]
    entropy = -np.sum(probs_np * np.log(probs_np + 1e-8))
    
    print(f"   - Action Probabilities: {probs_np}")
    # value_np가 스칼라 배열인 경우 처리
    if isinstance(value_np, np.ndarray):
        value_scalar = value_np.item() if value_np.size == 1 else float(value_np[0])
    else:
        value_scalar = float(value_np)
    print(f"   - State Value (V): {value_scalar:.4f}")
    print(f"   - Entropy: {entropy:.4f}")
    
    if np.any(np.isnan(probs_np)):
        print("❌ 치명적 오류: 신경망 출력이 NaN입니다. (가중치 폭발)")
    elif np.max(probs_np) > 0.99:
        print("⚠️ 경고: 초기 상태인데 확신이 너무 강합니다. (Policy Collapse 의심)")
        print("   -> Entropy Coefficient를 높이거나 초기화를 다시 해야 합니다.")
    elif np.allclose(probs_np, 1.0/len(probs_np), atol=0.01):
        print("✅ 신경망 출력 정상 (초기 탐색 가능 상태)")
    else:
        print("✅ 신경망 출력 양호 (적절한 확률 분포)")

    # 4. 리워드 스케일 점검
    print("\n" + "-" * 50)
    print("3️⃣ 리워드 함수 점검 (가상 시뮬레이션)")
    
    # 3-Action 구조: 0=Neutral, 1=Long, 2=Short
    # 1% 수익 상황 가정 (Long 포지션 청산)
    r_profit = env.calculate_reward(
        step_pnl=0.01, 
        realized_pnl=0.01, 
        trade_done=True, 
        action=0,  # Neutral (청산)
        prev_position='LONG', 
        current_position=None
    )
    # -1% 손실 상황 가정 (Long 포지션 청산)
    r_loss = env.calculate_reward(
        step_pnl=-0.01, 
        realized_pnl=-0.01, 
        trade_done=True, 
        action=0,  # Neutral (청산)
        prev_position='LONG', 
        current_position=None
    )
    
    print(f"   - 1% 익절 시 리워드: {r_profit:.4f}")
    print(f"   - 1% 손절 시 리워드: {r_loss:.4f}")
    print(f"   - 리워드 비율 (익절/손절): {abs(r_profit/r_loss) if r_loss != 0 else 'N/A':.2f}")
    
    if abs(r_profit) < 0.1:
        print("⚠️ 경고: 리워드가 너무 작습니다. 학습이 느릴 수 있습니다.")
    elif abs(r_profit) > 100:
        print("⚠️ 경고: 리워드가 너무 큽니다. 학습이 불안정할 수 있습니다.")
    else:
        print("✅ 리워드 스케일 양호")
    
    # 5. 추가 진단: 홀딩 보상 확인
    print("\n" + "-" * 50)
    print("4️⃣ 홀딩 보상 점검")
    
    # 포지션 유지 중 작은 수익
    r_hold_profit = env.calculate_reward(
        step_pnl=0.001,  # 0.1% 수익
        realized_pnl=0.0,
        trade_done=False,
        action=1,  # Long 유지
        prev_position='LONG',
        current_position='LONG'
    )
    
    # 포지션 유지 중 작은 손실
    r_hold_loss = env.calculate_reward(
        step_pnl=-0.001,  # 0.1% 손실
        realized_pnl=0.0,
        trade_done=False,
        action=1,  # Long 유지
        prev_position='LONG',
        current_position='LONG'
    )
    
    print(f"   - 홀딩 중 0.1% 수익 시 리워드: {r_hold_profit:.4f}")
    print(f"   - 홀딩 중 0.1% 손실 시 리워드: {r_hold_loss:.4f}")
    
    if r_hold_profit > 0 and r_hold_loss < 0:
        print("✅ 홀딩 보상 정상 (수익 시 양수, 손실 시 음수)")
    else:
        print("⚠️ 경고: 홀딩 보상 로직 확인 필요")

    # 6. 데이터 분할 확인
    print("\n" + "-" * 50)
    print("5️⃣ 데이터 분할 확인")
    
    total_len = len(env.collector.eth_data)
    train_end = int(total_len * config.TRAIN_SPLIT)
    val_end = int(total_len * (config.TRAIN_SPLIT + config.VAL_SPLIT))
    test_start = val_end
    
    print(f"   - 전체 데이터: {total_len}개")
    print(f"   - Train Set: 0 ~ {train_end} ({train_end/total_len*100:.1f}%)")
    print(f"   - Val Set: {train_end} ~ {val_end} ({(val_end-train_end)/total_len*100:.1f}%)")
    print(f"   - Test Set: {val_end} ~ {total_len} ({(total_len-val_end)/total_len*100:.1f}%)")
    
    # 전략 신호가 Train Set 이후에 계산되었는지 확인
    if 'strategy_0' in env.collector.eth_data.columns:
        test_strategy_sum = env.collector.eth_data['strategy_0'].iloc[test_start:].abs().sum()
        if test_strategy_sum == 0:
            print("✅ 데이터 누수 차단 확인: Test Set의 전략 신호는 0입니다.")
        else:
            print(f"⚠️ 경고: Test Set에 전략 신호가 있습니다! (합계: {test_strategy_sum:.2f})")
            print("   -> 전략 계산이 Train Set만 수행되었는지 확인 필요")
    
    print("\n" + "=" * 50)
    print("✅ 진단 완료!")
    print("=" * 50)

if __name__ == "__main__":
    debug_agent()
