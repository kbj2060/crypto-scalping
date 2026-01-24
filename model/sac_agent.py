"""
SAC (Soft Actor-Critic) Agent
Replay Buffer와 Alpha(엔트로피 계수) 자동 튜닝 기능 포함
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import sys
import os

# 상위 폴더를 경로에 추가 (config 모듈 접근용)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from .sac_network import SACActor, SACCritic
import logging

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Experience Replay Buffer for SAC"""
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """배치 샘플링"""
        batch = random.sample(self.buffer, batch_size)
        
        # State Unpacking (seq, info)
        obs_seq, obs_info = zip(*[b[0] for b in batch])
        actions = torch.FloatTensor(np.array([b[1] for b in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1).to(self.device)
        
        # next_state가 None일 수 있으므로 처리
        next_states = []
        for b in batch:
            if b[3] is None:
                # next_state가 None이면 현재 state 사용
                next_states.append(b[0])
            else:
                next_states.append(b[3])
        
        next_obs_seq, next_obs_info = zip(*next_states)
        dones = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1).to(self.device)

        # Tensor 변환
        obs_seq = torch.cat(obs_seq, dim=0).to(self.device)
        obs_info = torch.cat(obs_info, dim=0).to(self.device)
        next_obs_seq = torch.cat(next_obs_seq, dim=0).to(self.device)
        next_obs_info = torch.cat(next_obs_info, dim=0).to(self.device)

        return (obs_seq, obs_info), actions, rewards, (next_obs_seq, next_obs_info), dones

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    """Soft Actor-Critic Agent"""
    def __init__(self, state_dim, action_dim, info_dim=13, hidden_dim=None, device='cpu'):
        """
        Args:
            state_dim: 시계열 피처 차원 (29)
            action_dim: 행동 차원 (1: 연속형 매수/매도 강도)
            info_dim: 포지션 정보 차원 (전략 점수 + 포지션 정보)
            hidden_dim: Hidden dimension (None이면 config에서 가져옴)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.gamma = config.SAC_GAMMA
        self.tau = config.SAC_TAU
        self.alpha = config.SAC_ALPHA  # 초기 엔트로피 계수
        
        # Hidden Dim도 Config 사용 (None이면 config에서 가져오기)
        if hidden_dim is None:
            hidden_dim = config.NETWORK_HIDDEN_DIM
        
        # Networks
        self.actor = SACActor(state_dim, action_dim, info_dim, hidden_dim).to(device)
        self.critic = SACCritic(state_dim, action_dim, info_dim, hidden_dim).to(device)
        self.critic_target = SACCritic(state_dim, action_dim, info_dim, hidden_dim).to(device)
        
        # Target Network 초기화
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers (config에서 학습률 가져오기)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.SAC_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.SAC_LEARNING_RATE)
        
        # Automatic Entropy Tuning
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.SAC_LEARNING_RATE)
        
        # Replay Buffer (config에서 크기 가져오기)
        self.memory = ReplayBuffer(capacity=config.SAC_REPLAY_BUFFER_SIZE, device=device)
        
        logger.info(f"✅ SAC Agent 초기화 완료 (State: {state_dim}, Action: {action_dim}, Info: {info_dim})")

    def select_action(self, state, evaluate=False):
        """
        행동 선택
        
        Args:
            state: (obs_seq, obs_info) 튜플
            evaluate: True면 평균 행동 반환, False면 샘플링
        Returns:
            action: (action_dim,) numpy array
        """
        obs_seq, obs_info = state
        obs_seq = obs_seq.to(self.device)
        obs_info = obs_info.to(self.device)
        
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(obs_seq, obs_info)
            else:
                action, _, _ = self.actor.sample(obs_seq, obs_info)
        
        return action.cpu().numpy()[0]  # 1D array

    def update(self, batch_size=None):
        """
        SAC 업데이트 (Actor, Critic, Alpha)
        
        Args:
            batch_size: 배치 크기 (None이면 config에서 가져옴)
        Returns:
            critic_loss: Critic 손실
            actor_loss: Actor 손실
            alpha: 현재 엔트로피 계수
        """
        # Config의 Batch Size 사용 (인자가 없으면)
        if batch_size is None:
            batch_size = config.SAC_BATCH_SIZE
        
        if len(self.memory) < batch_size:
            return 0, 0, 0

        # Sample Batch
        state, action, reward, next_state, done = self.memory.sample(batch_size)
        obs_seq, obs_info = state
        next_obs_seq, next_obs_info = next_state

        # Target Q 계산
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs_seq, next_obs_info)
            q1_next, q2_next = self.critic_target(next_obs_seq, next_action, next_obs_info)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.gamma * min_q_next

        # Critic Update
        q1, q2 = self.critic(obs_seq, action, obs_info)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Actor Update
        action_new, log_prob, _ = self.actor.sample(obs_seq, obs_info)
        q1_new, q2_new = self.critic(obs_seq, action_new, obs_info)
        min_q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Alpha Update (Automatic Entropy Tuning)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        # Soft Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item(), self.alpha

    def save_model(self, path):
        """모델 저장"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha': self.alpha
        }, path)
        logger.info(f"💾 SAC 모델 저장 완료: {path}")

    def load_model(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = checkpoint.get('alpha', 0.2)
        logger.info(f"✅ SAC 모델 로드 완료: {path}")
