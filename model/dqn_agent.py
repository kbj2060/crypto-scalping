"""
Double DQN 에이전트 (Rainbow DQN 업그레이드)
[Final Fix] N-step Buffer Flush Logic + PER Beta Cumulative Step
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from .dqn_model import DuelingGRU
from .replay_buffer import ReplayBuffer
from .prioritized_replay_buffer import PrioritizedReplayBuffer
import logging

logger = logging.getLogger(__name__)

class DDQNAgent:
    """Double DQN 에이전트 + N-step Learning"""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, action_dim=3,
                 lr=0.0001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, buffer_size=50000, batch_size=64,
                 target_update=1000, device='cpu', use_per=False, n_step=3):
        
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        # [수정 1] PER Beta 스케줄링을 위한 '누적' 학습 스텝 카운터 (리셋 안 됨)
        self.total_train_steps = 0
        
        # Multi-step Learning 설정
        self.n_step = n_step
        self.n_step_buffer = deque()  # maxlen 제거 (수동 관리)
        
        # Epsilon-Greedy 파라미터
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 네트워크 2개 운영
        self.policy_net = DuelingGRU(input_dim, hidden_dim, num_layers, action_dim).to(device)
        self.target_net = DuelingGRU(input_dim, hidden_dim, num_layers, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Experience Replay Buffer
        self.use_per = use_per
        if use_per:
            self.memory = PrioritizedReplayBuffer(capacity=buffer_size, alpha=0.6)
            logger.info(f"✅ Rainbow 기능 활성화: PER + {n_step}-step Learning")
        else:
            self.memory = ReplayBuffer(capacity=buffer_size)
            logger.info(f"✅ N-step Learning 활성화 ({n_step}-step)")
    
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            if isinstance(state, tuple):
                obs_seq, obs_info = state
                obs_seq = obs_seq.to(self.device)
                obs_info = obs_info.to(self.device)
                q_values = self.policy_net(obs_seq)
            else:
                q_values = self.policy_net(state.to(self.device))
            
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """[수정 2] 데이터 유실 없는 N-step 저장 로직 (Flush 적용)"""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # 버퍼가 찰 때까지 대기 (단, done이면 바로 처리 시작)
        if len(self.n_step_buffer) < self.n_step:
            if not done:
                return
        
        # 버퍼 비우기 루프 (N-step 처리가 끝날 때까지)
        while self.n_step_buffer:
            # 버퍼에 남은 데이터가 1개 이상이면 처리
            if len(self.n_step_buffer) < self.n_step and not done:
                break # done이 아니면 N개 찰 때까지 대기
            
            # N-step 보상 계산
            # 실제 남은 길이만큼만 계산 (마지막엔 1-step, 2-step 등 줄어듦)
            current_n = min(self.n_step, len(self.n_step_buffer))
            reward_n = 0
            for i in range(current_n):
                reward_n += self.n_step_buffer[i][2] * (self.gamma ** i)
            
            state_t, action_t, _, _, _ = self.n_step_buffer[0]
            _, _, _, next_state_tn, done_tn = self.n_step_buffer[current_n - 1]
            
            # 메모리에 저장
            self.memory.push(state_t, action_t, reward_n, next_state_tn, done_tn)
            
            # 처리한 첫 번째 요소 제거
            self.n_step_buffer.popleft()
            
            # done이 아니면 하나만 처리하고 루프 종료 (슬라이딩 윈도우)
            if not done:
                break
                
        # 에피소드 종료 시 버퍼 완전 초기화 (혹시 남아있을 경우)
        if done:
            self.n_step_buffer.clear()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        
        self.total_train_steps += 1  # [수정 3] 누적 스텝 증가
        
        # PER 샘플링
        if self.use_per:
            # [수정 4] update_counter 대신 누적된 total_train_steps 사용
            # 10만 스텝에 걸쳐 0.4 -> 1.0 도달
            beta = min(1.0, 0.4 + 0.6 * (self.total_train_steps / 100000))
            states, actions, rewards, next_states, dones, weights, idxs = self.memory.sample(self.batch_size, beta)
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = None
            idxs = None
        
        # (이하 학습 로직 동일)
        states_seq, states_info = states
        states_seq = states_seq.to(self.device)
        states_info = states_info.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states_seq, next_states_info = next_states
        next_states_seq = next_states_seq.to(self.device)
        next_states_info = next_states_info.to(self.device)
        dones = dones.to(self.device)
        
        current_q_values = self.policy_net(states_seq)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values_policy = self.policy_net(next_states_seq)
            next_actions = next_q_values_policy.argmax(dim=1)
            next_q_values_target = self.target_net(next_states_seq)
            next_q = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            gamma_n = self.gamma ** self.n_step
            target_q = rewards + (gamma_n * next_q * (~dones).float())
        
        if self.use_per:
            loss_element_wise = F.smooth_l1_loss(current_q, target_q, reduction='none')
            loss = (loss_element_wise * weights).mean()
        else:
            loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        if self.use_per and idxs is not None:
            td_errors = loss_element_wise.detach().cpu().numpy()
            self.memory.update_priorities(idxs, td_errors)
        
        self.update_counter += 1
        if self.update_counter >= self.target_update:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.update_counter = 0
            logger.debug(f"타겟 네트워크 업데이트 완료")
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

    def save_model(self, filepath):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,
            'total_train_steps': self.total_train_steps # [수정] 저장 항목 추가
        }, filepath)
        logger.info(f"모델 저장 완료: {filepath}")
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.update_counter = checkpoint.get('update_counter', 0)
        self.total_train_steps = checkpoint.get('total_train_steps', 0) # [수정] 로드 항목 추가
        logger.info(f"모델 로드 완료: {filepath}")

    def reset_noise(self):
        if hasattr(self.policy_net, 'reset_noise'):
            self.policy_net.reset_noise()
        if hasattr(self.target_net, 'reset_noise'):
            self.target_net.reset_noise()