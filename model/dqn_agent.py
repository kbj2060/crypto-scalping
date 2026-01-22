"""
Double DQN 에이전트 (Rainbow DQN 업그레이드)
+ Multi-step Learning (N-step) 적용됨
+ Prioritized Experience Replay (PER)
+ Dueling Network
+ NoisyNet
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque  # [추가] N-step 버퍼용
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
                 target_update=1000, device='cpu', use_per=False, n_step=3):  # [설정] n_step 기본값 3
        """
        Args:
            n_step: Multi-step Learning의 스텝 수 (기본 3)
        """
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        # Multi-step Learning 설정
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)  # [추가] N-step 임시 저장소
        
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
        """행동 선택"""
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
        """
        [수정됨] N-step Learning을 위한 경험 저장 로직
        바로 저장하지 않고, n_step_buffer에 모았다가 N개가 차면 보상을 합쳐서 저장
        """
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_step:
            if done:
                self.n_step_buffer.clear()
            return
        
        # N-step 보상 계산 (Discounted Reward Sum)
        # R = r0 + gamma*r1 + gamma^2*r2 + ...
        reward_n = 0
        for i in range(self.n_step):
            reward_n += self.n_step_buffer[i][2] * (self.gamma ** i)
        
        # 저장할 데이터 추출
        # 상태(State): N스텝 전의 상태 (현재 버퍼의 맨 앞)
        # 행동(Action): N스텝 전의 행동
        # 다음 상태(Next State): 현재 시점의 상태 (버퍼의 마지막 next_state)
        # 종료 여부(Done): 현재 시점의 종료 여부
        state_t, action_t, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_state_tn, done_tn = self.n_step_buffer[-1]
        
        # 메모리에 저장
        self.memory.push(state_t, action_t, reward_n, next_state_tn, done_tn)
        
        # 에피소드 종료 시 버퍼 비움
        if done:
            self.n_step_buffer.clear()

    def train_step(self):
        """학습 로직 (N-step 적용)"""
        if len(self.memory) < self.batch_size:
            return None
        
        # PER 샘플링
        if self.use_per:
            beta = min(1.0, 0.4 + 0.6 * (self.update_counter / 20000))
            states, actions, rewards, next_states, dones, weights, idxs = self.memory.sample(self.batch_size, beta)
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = None
            idxs = None
        
        # 데이터 디바이스 이동
        states_seq, states_info = states
        states_seq = states_seq.to(self.device)
        states_info = states_info.to(self.device)
        
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        
        next_states_seq, next_states_info = next_states
        next_states_seq = next_states_seq.to(self.device)
        next_states_info = next_states_info.to(self.device)
        
        dones = dones.to(self.device)
        
        # Current Q
        current_q_values = self.policy_net(states_seq)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q (Double DQN)
        with torch.no_grad():
            next_q_values_policy = self.policy_net(next_states_seq)
            next_actions = next_q_values_policy.argmax(dim=1)
            
            next_q_values_target = self.target_net(next_states_seq)
            next_q = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # [수정됨] N-step Target Calculation
            # Y = Reward_n + (γ ^ n) * Q_target * (1 - Done)
            gamma_n = self.gamma ** self.n_step
            target_q = rewards + (gamma_n * next_q * (~dones).float())
        
        # Loss Calculation
        if self.use_per:
            loss_element_wise = F.smooth_l1_loss(current_q, target_q, reduction='none')
            loss = (loss_element_wise * weights).mean()
        else:
            loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # PER Update
        if self.use_per and idxs is not None:
            td_errors = loss_element_wise.detach().cpu().numpy()
            self.memory.update_priorities(idxs, td_errors)
        
        # Target Update
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
            'update_counter': self.update_counter
        }, filepath)
        logger.info(f"모델 저장 완료: {filepath}")
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.update_counter = checkpoint.get('update_counter', 0)
        logger.info(f"모델 로드 완료: {filepath}")
    
    def reset_noise(self):
        if hasattr(self.policy_net, 'reset_noise'):
            self.policy_net.reset_noise()
        if hasattr(self.target_net, 'reset_noise'):
            self.target_net.reset_noise()
