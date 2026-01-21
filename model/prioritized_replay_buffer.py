"""
Prioritized Experience Replay (PER)
중요한 경험(높은 TD-Error)을 우선적으로 학습하여 효율성을 극대화함
"""
import numpy as np
import torch
import random
from collections import namedtuple

class SumTree:
    """PER을 위한 고속 합계 트리 자료구조"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6):
        """
        Args:
            capacity: 버퍼 최대 크기
            alpha: 우선순위 반영 비율 (0=랜덤, 1=완전우선)
        """
        self.tree = SumTree(capacity)
        self.alpha = alpha  # 우선순위 반영 비율 (0=랜덤, 1=완전우선)
        self.epsilon = 1e-5 # 0으로 나누기 방지 및 최소 우선순위 보장
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        """새로운 경험 저장 (최대 우선순위로 저장하여 한 번은 꼭 학습하게 함)"""
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0
        
        experience = (state, action, reward, next_state, done)
        self.tree.add(max_priority, experience)

    def sample(self, batch_size, beta=0.4):
        """
        우선순위에 따라 샘플링 및 가중치(Weights) 반환
        beta: 편향 보정 계수 (학습 후반부로 갈수록 1에 가까워져야 함)
        """
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            
            # 간혹 트리 오류로 None이 반환될 경우 대비
            if data is None: 
                # 데이터가 없으면 랜덤하게 하나 뽑음 (안전장치)
                idx = random.randint(self.tree.capacity - 1, self.tree.capacity + self.tree.n_entries - 2)
                data = self.tree.data[idx - self.tree.capacity + 1]
                p = self.tree.tree[idx]
            
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max() # 정규화

        # 데이터 파싱 (기존 ReplayBuffer와 호환)
        states_seq, states_info = [], []
        actions, rewards = [], []
        next_states_seq, next_states_info = [], []
        dones = []

        for state, action, reward, next_state, done in batch:
            # (데이터 파싱 로직은 기존과 동일)
            if isinstance(state, tuple):
                # Late Fusion 구조: (obs_seq, obs_info)
                states_seq.append(state[0])
                states_info.append(state[1])
                
                if next_state is not None:
                    next_states_seq.append(next_state[0])
                    next_states_info.append(next_state[1])
                else:
                    # next_state가 None인 경우 (에피소드 종료)
                    next_states_seq.append(state[0])
                    next_states_info.append(state[1])
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

        # 텐서 변환
        states_seq = torch.cat(states_seq, dim=0)
        states_info = torch.cat(states_info, dim=0)
        next_states_seq = torch.cat(next_states_seq, dim=0)
        next_states_info = torch.cat(next_states_info, dim=0)
        
        states = (states_seq, states_info)
        next_states = (next_states_seq, next_states_info)
        
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        weights = torch.tensor(is_weight, dtype=torch.float32)

        return states, actions, rewards, next_states, dones, weights, idxs

    def update_priorities(self, idxs, errors):
        """학습 후 TD-Error를 기반으로 우선순위 업데이트"""
        for idx, error in zip(idxs, errors):
            p = (error + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            
    def __len__(self):
        return self.tree.n_entries
