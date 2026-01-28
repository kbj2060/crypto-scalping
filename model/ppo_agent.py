"""
PPO Agent (Clean Ver.)
- Reverted Gradient Clipping to standard (10.0)
- Kept: Scheduler, EMA Target, Temp Sched
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from torch.distributions import Categorical
import sys
import os
import copy  # [New] For Target Network

from . import config
from .xlstm_network import XLSTMNetwork

class PPOAgent:
    def __init__(self, state_dim, action_dim, info_dim=13, hidden_dim=None, device='cpu'):
        self.device = device
        self.action_dim = action_dim
        
        hidden_dim = hidden_dim if hidden_dim is not None else config.NETWORK_HIDDEN_DIM
        
        # [1] Main Model
        self.model = XLSTMNetwork(
            input_dim=state_dim, 
            action_dim=action_dim,
            info_dim=info_dim,
            hidden_dim=hidden_dim,
            num_layers=config.NETWORK_NUM_LAYERS
        ).to(device)
        
        # [New] Target Model for Critic EMA (안정성 강화)
        self.model_target = copy.deepcopy(self.model)
        self.model_target.eval() # 학습 모드 끔
        self.target_tau = 0.995  # Soft Update Rate
        
        # [New] Temperature Scheduling Parameters (탐색 조절)
        self.temperature = 1.0  # [수정] 1.5 -> 1.0 (차분한 탐색)
        self.min_temp = 0.6
        self.temp_decay = 0.999

        # Optimizer & Scheduler
        lr = config.PPO_LEARNING_RATE
        if lr > 0.001: 
            lr = 0.0003
            import logging
            logging.warning(f"⚠️ 학습률 조정: {config.PPO_LEARNING_RATE} → {lr}")
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=100,
            T_mult=2,
            eta_min=1e-6
        )
        
        self.current_states = None
        self.gamma = config.PPO_GAMMA
        self.lmbda = config.PPO_LAMBDA
        self.eps_clip = config.PPO_EPS_CLIP
        self.k_epochs = config.PPO_K_EPOCHS
        self.entropy_coef = config.PPO_ENTROPY_COEF
        
        self.data = []
        
    def reset_episode_states(self):
        self.current_states = None
        
    def load_model(self, path):
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                else:
                    model_state = checkpoint
                
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in model_state.items() 
                                 if k in model_dict and v.size() == model_dict[k].size()}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict, strict=False)
                
                # [New] Target model도 로드된 가중치로 동기화
                self.model_target.load_state_dict(model_dict, strict=False)
                
                if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
                    try: self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except: pass
                
                print(f"✅ 모델 로드 성공: {path}")
            except Exception as e:
                print(f"⚠️ 모델 로드 실패: {e}")

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def put_data(self, transition):
        self.data.append(transition)
        
    def select_action(self, state, action_mask=None):
        obs_seq, obs_info = state
        
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.FloatTensor(obs_seq).to(self.device)
        else: obs_seq = obs_seq.to(self.device)
            
        if not isinstance(obs_info, torch.Tensor):
            obs_info = torch.FloatTensor(obs_info).unsqueeze(0).to(self.device)
        else: obs_info = obs_info.to(self.device)
        
        with torch.no_grad():
            # [New] Apply Temperature during sampling
            probs, value, self.current_states = self.model(
                obs_seq, obs_info, self.current_states, temperature=self.temperature
            )
            
            if action_mask is not None:
                mask = torch.FloatTensor(action_mask).to(self.device)
                probs = probs * mask
                if probs.sum() == 0: 
                    probs = torch.ones_like(probs) / len(probs)
                else:
                    probs = probs / probs.sum()
                
            dist = Categorical(probs)
            action = dist.sample()
            
        return action.item(), dist.log_prob(action).item(), value.item()

    def train_net(self, episode=1):
        if not self.data: return 0.0
        
        # 1. Prepare Data
        s_seq_lst, s_info_lst, a_lst, r_lst, next_s_seq_lst, next_s_info_lst, prob_a_lst, done_lst, old_v_lst = [], [], [], [], [], [], [], [], []
        
        for transition in self.data:
            if len(transition) == 7:
                s, a, r, next_s, prob_a, done, val = transition
            else:
                s, a, r, next_s, prob_a, done = transition
                val = 0.0
            
            s_seq_lst.append(s[0]); s_info_lst.append(s[1])
            a_lst.append([a]); r_lst.append([r])
            next_s_seq_lst.append(next_s[0]); next_s_info_lst.append(next_s[1])
            prob_a_lst.append([prob_a]); done_lst.append([0 if done else 1])
            old_v_lst.append([val])

        def to_tensor(data, dtype=torch.float):
            if isinstance(data[0], torch.Tensor):
                return torch.cat(data, dim=0).to(self.device)
            return torch.tensor(np.array(data), dtype=dtype).to(self.device)

        s_seq = to_tensor(s_seq_lst); s_info = to_tensor(s_info_lst)
        next_s_seq = to_tensor(next_s_seq_lst); next_s_info = to_tensor(next_s_info_lst)
        a = torch.tensor(a_lst, dtype=torch.long).to(self.device)
        r = torch.tensor(r_lst, dtype=torch.float).to(self.device)
        done_mask = torch.tensor(done_lst, dtype=torch.float).to(self.device)
        prob_a = torch.tensor(prob_a_lst, dtype=torch.float).to(self.device)
        old_v = torch.tensor(old_v_lst, dtype=torch.float).to(self.device)
        
        self.data = []

        # 2. GAE Calculation with Target Critic (Stabilization)
        with torch.no_grad():
            # Current V(s) from Main Model
            _, v, _ = self.model(s_seq, s_info, states=None, temperature=self.temperature)
            
            # [New] Target V(s') from Target Model (EMA)
            # Critic 안정성을 위해 Target Network 사용
            _, next_v, _ = self.model_target(next_s_seq, next_s_info, states=None, temperature=self.temperature)
            
            # TD Error using Target Value
            td_target = r + self.gamma * next_v * done_mask
            delta = td_target - v
            
            advantage_lst = []
            gae = 0.0
            for t in reversed(range(len(delta))):
                gae = delta[t] + self.gamma * self.lmbda * done_mask[t] * gae
                advantage_lst.insert(0, gae)
                
            advantage = torch.stack(advantage_lst)
            target_v = advantage + v 

        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        # 3. PPO Update
        total_loss = 0.0
        for epoch in range(self.k_epochs):
            # Pass current temperature for accurate log_prob
            curr_probs, curr_v, _ = self.model(s_seq, s_info, states=None, temperature=self.temperature)
            dist = Categorical(curr_probs)
            curr_log_prob = dist.log_prob(a.squeeze()).unsqueeze(1)
            
            ratio = torch.exp(curr_log_prob - prob_a)
            
            with torch.no_grad():
                log_ratio = curr_log_prob - prob_a
                approx_kl = (torch.exp(log_ratio) - 1) - log_ratio
                if approx_kl.mean() > config.PPO_KL_TARGET:
                    break
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            
            curr_entropy_coef = max(config.PPO_ENTROPY_MIN, self.entropy_coef * (config.PPO_ENTROPY_DECAY ** episode))
            
            actor_loss = -torch.min(surr1, surr2).mean()
            
            if config.PPO_USE_VALUE_CLIP:
                v_clipped = old_v + torch.clamp(curr_v - old_v, -config.PPO_VALUE_CLIP_EPS, config.PPO_VALUE_CLIP_EPS)
                v_loss_1 = (curr_v - target_v)**2
                v_loss_2 = (v_clipped - target_v)**2
                critic_loss = 0.5 * torch.max(v_loss_1, v_loss_2).mean()
            else:
                critic_loss = 0.5 * F.mse_loss(curr_v, target_v)
            
            entropy_loss = curr_entropy_coef * dist.entropy().mean()
            loss = actor_loss + critic_loss - entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # [Fix] Reverted to standard clipping (10.0)
            # m_t가 내부적으로 폭발을 막아주므로 강제 억제 불필요
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            self.optimizer.step()
            total_loss += loss.item()

        # [New] Update Temperature (Decay)
        self.temperature = max(self.temperature * self.temp_decay, self.min_temp)

        # [New] Update Target Critic (EMA)
        # Main Model의 파라미터를 Target Model로 부드럽게 복사
        for param, target_param in zip(self.model.parameters(), self.model_target.parameters()):
            target_param.data.copy_(self.target_tau * target_param.data + (1.0 - self.target_tau) * param.data)

        self.scheduler.step()
        return total_loss / self.k_epochs
