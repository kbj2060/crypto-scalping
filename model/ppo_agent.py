"""
PPO Agent (Optimized for xLSTM & Tuple State)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from .xlstm_network import xLSTMActorCritic

class PPOAgent:
    def __init__(self, state_dim, action_dim, info_dim=13, hidden_dim=None, device='cpu'):
        self.device = device
        
        # Hyperparameters
        self.gamma = config.PPO_GAMMA
        self.lmbda = config.PPO_LAMBDA
        self.eps_clip = config.PPO_EPS_CLIP
        self.k_epochs = config.PPO_K_EPOCHS
        self.entropy_coef = config.PPO_ENTROPY_COEF
        
        # Network parameters
        hidden_dim = hidden_dim if hidden_dim is not None else config.NETWORK_HIDDEN_DIM
        
        # Model
        self.model = xLSTMActorCritic(
            state_dim, action_dim, info_dim, hidden_dim,
            num_layers=config.NETWORK_NUM_LAYERS,
            dropout=config.NETWORK_DROPOUT,
            use_checkpointing=config.NETWORK_USE_CHECKPOINTING
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.PPO_LEARNING_RATE)
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max',
            factor=config.PPO_SCHEDULER_FACTOR,
            patience=config.PPO_SCHEDULER_PATIENCE,
            min_lr=config.PPO_SCHEDULER_MIN_LR
        )
        
        self.memory = []
        self.current_states = None  # LSTM 상태 유지

    def reset_episode_states(self):
        """에피소드 시작 시 상태 초기화"""
        self.current_states = None

    def put_data(self, transition):
        """트랜지션 저장"""
        self.memory.append(transition)

    def select_action(self, state):
        """
        state: (obs_seq, obs_info) 튜플
        Returns: (action, log_prob)
        """
        # 튜플 언패킹
        obs_seq, obs_info = state
        
        obs_seq = obs_seq.to(self.device)
        obs_info = obs_info.to(self.device)
        
        # 추론 (No Gradients)
        with torch.no_grad():
            # LSTM 상태 유지
            probs, _, self.current_states = self.model(
                obs_seq, obs_info, 
                states=self.current_states,
                return_states=True
            )
        
        dist = Categorical(probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action).item()

    def train_net(self, episode=1):
        """PPO 네트워크 학습"""
        if not self.memory:
            return 0.0
        
        # 데이터 분리
        s_list, a_list, r_list, next_s_list, prob_list, done_list = [], [], [], [], [], []
        
        for data in self.memory:
            s, a, r, next_s, prob, done = data
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            next_s_list.append(next_s)
            prob_list.append([prob])
            done_list.append([0 if done else 1])

        # 배치 텐서 변환
        s_seq_batch = torch.cat([item[0] for item in s_list], dim=0).to(self.device)
        s_info_batch = torch.cat([item[1] for item in s_list], dim=0).to(self.device)
        
        ns_seq_batch = torch.cat([item[0] for item in next_s_list], dim=0).to(self.device)
        ns_info_batch = torch.cat([item[1] for item in next_s_list], dim=0).to(self.device)

        a_batch = torch.tensor(a_list, dtype=torch.long).to(self.device)
        r_batch = torch.tensor(r_list, dtype=torch.float).to(self.device)
        done_batch = torch.tensor(done_list, dtype=torch.float).to(self.device)
        prob_a_batch = torch.tensor(prob_list, dtype=torch.float).to(self.device)

        # GAE 계산을 위한 Value 계산
        with torch.no_grad():
            _, v_s = self.model(s_seq_batch, s_info_batch, states=None, return_states=False)
            _, v_next = self.model(ns_seq_batch, ns_info_batch, states=None, return_states=False)
        
        # TD Target
        td_target = r_batch + self.gamma * v_next * done_batch
        delta = td_target - v_s
        
        # GAE (Generalized Advantage Estimation)
        advantages = []
        gae = 0
        for step in reversed(range(len(r_batch))):
            if done_batch[step] == 0:
                delta_step = delta[step]
                gae = delta_step + self.gamma * self.lmbda * gae
            else:
                gae = delta[step]
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)
        returns = advantages + v_s.squeeze()
        
        # Advantage 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0.0
        
        for _ in range(self.k_epochs):
            # 현재 정책의 확률과 가치 계산
            pi_probs, v_s = self.model(s_seq_batch, s_info_batch, states=None, return_states=False)
            
            # Action Probabilities
            dist = Categorical(pi_probs)
            pi_a = dist.log_prob(a_batch.squeeze()).unsqueeze(1)
            ratio = torch.exp(pi_a - prob_a_batch)

            # PPO Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.smooth_l1_loss(v_s.squeeze(), returns.detach())
            entropy_loss = dist.entropy().mean()
            
            # 엔트로피 감소
            current_entropy_coef = max(
                config.PPO_ENTROPY_MIN,
                self.entropy_coef * (config.PPO_ENTROPY_DECAY ** episode)
            )
            
            loss = actor_loss + 1.0 * critic_loss - current_entropy_coef * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()

        self.memory = []  # 메모리 초기화
        return total_loss / self.k_epochs

    def save_model(self, path):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load_model(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def step_scheduler(self, metric):
        """학습률 스케줄러 업데이트"""
        self.scheduler.step(metric)
