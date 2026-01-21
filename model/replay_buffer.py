"""
Experience Replay Buffer
DDQN 학습을 위한 경험 저장소
"""
from collections import deque
import random
import numpy as np
import torch


class ReplayBuffer:
    """Experience Replay Buffer: (state, action, reward, next_state, done) 튜플 저장"""
    
    def __init__(self, capacity=50000):
        """
        Args:
            capacity: 버퍼 최대 크기 (오래된 기억은 자동으로 삭제)
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        경험 저장
        
        Args:
            state: (obs_seq, obs_info) 튜플 또는 단일 텐서
            action: 행동 인덱스 (0, 1, 2)
            reward: 보상값
            next_state: 다음 상태 (state와 동일한 형태)
            done: 에피소드 종료 여부 (bool)
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        배치 샘플링
        
        Args:
            batch_size: 샘플링할 개수
        Returns:
            states: 배치 상태
            actions: 배치 행동
            rewards: 배치 보상
            next_states: 배치 다음 상태
            dones: 배치 종료 플래그
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        # 튜플 형태의 state 처리
        states_seq = []
        states_info = []
        actions = []
        rewards = []
        next_states_seq = []
        next_states_info = []
        dones = []
        
        for state, action, reward, next_state, done in batch:
            if isinstance(state, tuple):
                # Late Fusion 구조: (obs_seq, obs_info)
                states_seq.append(state[0])
                states_info.append(state[1])
                
                if next_state is not None:
                    next_states_seq.append(next_state[0])
                    next_states_info.append(next_state[1])
                else:
                    # next_state가 None인 경우 (에피소드 종료)
                    next_states_seq.append(state[0])  # 더미 값
                    next_states_info.append(state[1])  # 더미 값
            else:
                # 단일 텐서인 경우 (하위 호환성)
                states_seq.append(state)
                states_info.append(torch.zeros(1, 13))  # 더미 정보
                
                if next_state is not None:
                    next_states_seq.append(next_state)
                    next_states_info.append(torch.zeros(1, 13))
                else:
                    next_states_seq.append(state)
                    next_states_info.append(torch.zeros(1, 13))
            
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        # 텐서로 변환
        states_seq = torch.cat(states_seq, dim=0)
        states_info = torch.cat(states_info, dim=0)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states_seq = torch.cat(next_states_seq, dim=0)
        next_states_info = torch.cat(next_states_info, dim=0)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        return (states_seq, states_info), actions, rewards, (next_states_seq, next_states_info), dones
    
    def __len__(self):
        """버퍼 크기"""
        return len(self.buffer)
