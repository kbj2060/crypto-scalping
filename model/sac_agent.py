"""
Continuous SAC Agent
- Uses Gaussian Policy (Reparameterization Trick)
- Automatic Entropy Tuning with target_entropy = -action_dim
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import random
from collections import deque
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from .sac_network import SACActor, SACCritic

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Experience Replay Buffer for Continuous SAC"""
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        """action: Continuous array"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """배치 샘플링"""
        batch = random.sample(self.buffer, batch_size)
        
        # State Unpacking (seq, info)
        obs_seq, obs_info = zip(*[b[0] for b in batch])
        # Action is Continuous FloatTensor
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
    """Soft Actor-Critic Agent (Continuous)"""
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
        # Continuous Action Dim (보통 1 또는 3)
        self.action_dim = action_dim 
        
        self.gamma = config.SAC_GAMMA
        self.tau = config.SAC_TAU
        self.alpha = config.SAC_ALPHA
        
        # Hidden Dim도 Config 사용 (None이면 config에서 가져오기)
        if hidden_dim is None:
            hidden_dim = config.NETWORK_HIDDEN_DIM
        
        # Networks (Continuous SACActor 사용)
        self.actor = SACActor(state_dim, action_dim, info_dim, hidden_dim).to(device)
        self.critic = SACCritic(state_dim, action_dim, info_dim, hidden_dim).to(device)
        self.critic_target = SACCritic(state_dim, action_dim, info_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers (config에서 학습률 가져오기)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.SAC_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.SAC_LEARNING_RATE)
        
        # Auto Entropy Tuning (Continuous: target = -action_dim)
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.SAC_LEARNING_RATE)
        
        # Replay Buffer (config에서 크기 가져오기)
        self.memory = ReplayBuffer(capacity=config.SAC_REPLAY_BUFFER_SIZE, device=device)
        
        # 스케줄러 초기화 (setup_schedulers에서 설정됨)
        self.actor_scheduler = None
        self.critic_scheduler = None
        self.alpha_scheduler = None
        
        # [NEW] 상태 관리를 위한 변수
        self.actor_state = None

        logger.info(f"✅ Continuous SAC Agent Initialized. Action Dim: {action_dim}")

    def setup_schedulers(self, total_steps, warmup_ratio=0.05):
        """
        Warmup + Linear Decay 스케줄러 설정
        
        Args:
            total_steps: 전체 학습 스텝 수
            warmup_ratio: Warmup 구간 비율 (기본값 0.05 = 5%)
        """
        warmup_steps = int(total_steps * warmup_ratio)
        
        def lr_lambda(step):
            # 1. Warmup 구간: 0 -> 1로 선형 증가
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            
            # 2. Linear Decay 구간: 1 -> 0으로 선형 감소
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

        self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=lr_lambda)
        self.critic_scheduler = LambdaLR(self.critic_optimizer, lr_lambda=lr_lambda)
        self.alpha_scheduler = LambdaLR(self.alpha_optimizer, lr_lambda=lr_lambda)
        
        logger.info(f"📈 스케줄러 설정 완료: 총 {total_steps} 스텝, Warmup {warmup_steps} 스텝 ({warmup_ratio*100:.1f}%)")
    
    def step_schedulers(self):
        """매 업데이트마다 호출하여 LR 조절"""
        if self.actor_scheduler:
            self.actor_scheduler.step()
        if self.critic_scheduler:
            self.critic_scheduler.step()
        if self.alpha_scheduler:
            self.alpha_scheduler.step()

    def reset_episode_states(self):
        """에피소드 시작 시 상태 초기화"""
        self.actor_state = None

    def select_action(self, state, evaluate=False):
        """
        Stateful Action Selection
        LSTM 상태를 유지하며 행동 결정
        
        Args:
            state: (obs_seq, obs_info) 튜플
            evaluate: True면 평균 행동 반환, False면 샘플링
        Returns:
            action: (numpy array) Continuous value [-1, 1]
        """
        # 튜플 입력 처리
        if isinstance(state, (tuple, list)):
            if len(state) == 3:
                obs_seq, obs_info, _ = state
            else:
                obs_seq, obs_info = state
        else:
            obs_seq, obs_info = state

        obs_seq = obs_seq.to(self.device)
        obs_info = obs_info.to(self.device)
        
        with torch.no_grad():
            if evaluate:
                # 평가 시에는 상태 갱신 없이 mean action 사용
                mu, _, next_states = self.actor(obs_seq, obs_info, self.actor_state)
                action = torch.tanh(mu)
                # 평가 시에도 상태는 갱신 (연속성 유지)
                self.actor_state = next_states
            else:
                # [FIX] 상태를 입력으로 넣고, 다음 상태를 받아와 저장
                action, _, _, next_states = self.actor.sample(obs_seq, obs_info, self.actor_state)
                self.actor_state = next_states  # 상태 갱신 (기억 유지)
        
        return action.cpu().numpy()[0]  # 1D array

    def update(self, batch_size=None):
        """
        Continuous SAC 업데이트 (Actor, Critic, Alpha)
        
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

        # ----------------------------
        # 1. Critic Update
        # ----------------------------
        with torch.no_grad():
            # 학습 시에는 랜덤 배치이므로 상태(states)를 None으로 하여 초기화된 상태에서 시작
            # (Replay Buffer에 Hidden State를 저장하지 않는 방식)
            next_action, next_log_prob, _, _ = self.actor.sample(next_obs_seq, next_obs_info, states=None)
            q1_next, q2_next, _ = self.critic_target(next_obs_seq, next_action, next_obs_info, states=None)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.gamma * min_q_next

        # Current Q
        q1, q2, _ = self.critic(obs_seq, action, obs_info, states=None)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ----------------------------
        # 2. Actor Update
        # ----------------------------
        # Current Action sampling (Reparameterization)
        action_new, log_prob, _, _ = self.actor.sample(obs_seq, obs_info, states=None)
        q1_new, q2_new, _ = self.critic(obs_seq, action_new, obs_info, states=None)
        min_q_new = torch.min(q1_new, q2_new)
        
        # Maximize (min_q - alpha * log_prob)
        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ----------------------------
        # 3. Alpha Update
        # ----------------------------
        # Target Entropy = -Action Dim
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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = checkpoint.get('alpha', 0.2)
        logger.info(f"✅ SAC 모델 로드 완료: {path}")
