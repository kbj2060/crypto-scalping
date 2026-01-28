"""
PPO Agent (Hierarchical RL Version)
- High-Level Agent (Entry): 진입 담당 (Wait, Long, Short)
- Low-Level Agent (Exit): 청산 담당 (Hold, Exit)
- 두 개의 독립된 두뇌가 협력하여 매매를 수행
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
import copy

from . import config
from .xlstm_network import XLSTMNetwork

# ==============================================================================
# 1. Core PPO Agent (실제 학습 로직을 가진 내부 작업자)
# ==============================================================================
class CorePPOAgent:
    def __init__(self, state_dim, action_dim, info_dim=3, hidden_dim=None, device='cpu', name="core"):
        self.device = device
        self.name = name
        self.action_dim = action_dim
        
        hidden_dim = hidden_dim if hidden_dim is not None else config.NETWORK_HIDDEN_DIM
        
        self.model = XLSTMNetwork(
            input_dim=state_dim, 
            action_dim=action_dim,
            info_dim=info_dim,
            hidden_dim=hidden_dim,
            num_layers=config.NETWORK_NUM_LAYERS
        ).to(device)
        
        self.model_target = copy.deepcopy(self.model)
        self.model_target.eval()
        self.target_tau = 0.995 
        
        self.temperature = 1.0
        self.min_temp = 0.6
        self.temp_decay = 0.999

        lr = config.PPO_LEARNING_RATE
        if lr > 0.001:
            lr = 0.0003
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2, eta_min=1e-6
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
        
    def select_action(self, state, action_mask=None):
        obs_seq, obs_info = state
        
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.FloatTensor(obs_seq).to(self.device)
        else:
            obs_seq = obs_seq.to(self.device)
            
        if not isinstance(obs_info, torch.Tensor):
            obs_info = torch.FloatTensor(obs_info).unsqueeze(0).to(self.device)
        else:
            obs_info = obs_info.to(self.device)
        
        with torch.no_grad():
            probs, value, self.current_states = self.model(
                obs_seq, obs_info, self.current_states, temperature=self.temperature
            )
            
            # Action Masking (필요 시)
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

    def put_data(self, transition):
        self.data.append(transition)

    def train_net(self, episode=1):
        if not self.data:
            return 0.0
        
        # 데이터 분리 및 텐서 변환
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
        
        self.data = []  # Clear buffer

        # GAE Calculation
        with torch.no_grad():
            _, v, _ = self.model(s_seq, s_info, states=None, temperature=self.temperature)
            _, next_v, _ = self.model_target(next_s_seq, next_s_info, states=None, temperature=self.temperature)
            
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

        # PPO Update Loop
        total_loss = 0.0
        for epoch in range(self.k_epochs):
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
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            total_loss += loss.item()

        self.temperature = max(self.temperature * self.temp_decay, self.min_temp)

        for param, target_param in zip(self.model.parameters(), self.model_target.parameters()):
            target_param.data.copy_(self.target_tau * target_param.data + (1.0 - self.target_tau) * param.data)

        self.scheduler.step()
        return total_loss / self.k_epochs

# ==============================================================================
# 2. Hierarchical PPO Agent (지휘관 클래스)
# ==============================================================================
class PPOAgent:
    def __init__(self, state_dim, action_dim=4, info_dim=3, hidden_dim=None, device='cpu'):
        self.device = device
        
        # [1] Entry Agent: 진입 전문 (3 Actions: WAIT=0, LONG=1, SHORT=2)
        # info_dim=3 (position info)가 들어오지만, 진입 시점엔 모두 0임.
        self.entry_agent = CorePPOAgent(
            state_dim=state_dim, action_dim=3, info_dim=info_dim, 
            hidden_dim=hidden_dim, device=device, name="entry"
        )
        
        # [2] Exit Agent: 청산 전문 (2 Actions: HOLD=0, EXIT=1)
        # 이미 포지션이 있는 상태에서 호출됨.
        self.exit_agent = CorePPOAgent(
            state_dim=state_dim, action_dim=2, info_dim=info_dim, 
            hidden_dim=hidden_dim, device=device, name="exit"
        )
        
    def reset_episode_states(self):
        self.entry_agent.reset_episode_states()
        self.exit_agent.reset_episode_states()
        
    def load_model(self, path):
        # 경로 분리 (예: model_best.pth -> model_best_entry.pth, model_best_exit.pth)
        base, ext = os.path.splitext(path)
        entry_path = f"{base}_entry{ext}"
        exit_path = f"{base}_exit{ext}"
        
        if os.path.exists(entry_path):
            checkpoint = torch.load(entry_path, map_location=self.device)
            self.entry_agent.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.entry_agent.model_target.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"✅ Entry 모델 로드 성공: {entry_path}")
            
        if os.path.exists(exit_path):
            checkpoint = torch.load(exit_path, map_location=self.device)
            self.exit_agent.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.exit_agent.model_target.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"✅ Exit 모델 로드 성공: {exit_path}")

    def save_model(self, path):
        base, ext = os.path.splitext(path)
        
        # Entry Agent 저장
        torch.save({
            'model_state_dict': self.entry_agent.model.state_dict(),
            'optimizer_state_dict': self.entry_agent.optimizer.state_dict(),
        }, f"{base}_entry{ext}")
        
        # Exit Agent 저장
        torch.save({
            'model_state_dict': self.exit_agent.model.state_dict(),
            'optimizer_state_dict': self.exit_agent.optimizer.state_dict(),
        }, f"{base}_exit{ext}")

    def select_action(self, state, action_mask=None):
        """
        상태(포지션 유무)에 따라 적절한 에이전트를 호출하고,
        Global Action(0~3)으로 변환하여 반환
        """
        obs_seq, obs_info = state
        
        # obs_info의 첫 번째 값(Position Flag) 확인
        # 1.0=Long, -1.0=Short, 0.0=None
        if isinstance(obs_info, torch.Tensor):
            pos_flag = obs_info[0, 0].item() if obs_info.dim() > 1 else obs_info[0].item()
        else:
            pos_flag = obs_info[0]

        is_position_open = (abs(pos_flag) > 0.1)  # 0이 아니면 포지션 있음
        
        if not is_position_open:
            # [Case A] 무포지션 -> Entry Agent 호출
            # Local Actions: 0(Wait), 1(Long), 2(Short)
            action, log_prob, val = self.entry_agent.select_action(state)
            
            # Global Mapping: 동일함 (0->0, 1->1, 2->2)
            return action, log_prob, val
            
        else:
            # [Case B] 포지션 보유 중 -> Exit Agent 호출
            # Local Actions: 0(Hold), 1(Exit)
            action, log_prob, val = self.exit_agent.select_action(state)
            
            # Global Mapping
            # Local 0 (Hold) -> Global 0 (Wait/Hold)
            # Local 1 (Exit) -> Global 3 (Exit)
            if action == 1:
                global_action = 3
            else:
                global_action = 0
                
            return global_action, log_prob, val

    def put_data(self, transition):
        """
        Transition의 상태를 보고 어느 에이전트의 버퍼에 넣을지 결정
        Transition: (s, a, r, next_s, prob_a, done, val)
        """
        s = transition[0]
        obs_info = s[1]
        
        # 텐서/배열 처리
        if isinstance(obs_info, torch.Tensor):
            pos_flag = obs_info[0, 0].item() if obs_info.dim() > 1 else obs_info[0].item()
        else:
            pos_flag = obs_info[0]
            
        is_position_open = (abs(pos_flag) > 0.1)
        
        # Action Remapping for Training
        # 학습할 때는 Local Action으로 변환해서 저장해야 함
        s, global_a, r, next_s, prob_a, done, val = transition
        
        if not is_position_open:
            # Entry Agent 데이터
            # Global Action 0, 1, 2는 Local Action과 동일
            if global_a <= 2:
                self.entry_agent.put_data((s, global_a, r, next_s, prob_a, done, val))
        else:
            # Exit Agent 데이터
            # Global Action 0(Hold) -> Local 0
            # Global 3(Exit) -> Local 1
            if global_a == 3:
                local_a = 1
            elif global_a == 0:
                local_a = 0
            else:
                # 포지션 있는데 진입 액션(1,2)이 들어오면? (이론상 안됨)
                return 

            self.exit_agent.put_data((s, local_a, r, next_s, prob_a, done, val))

    def train_net(self, episode=1):
        # 두 에이전트 모두 학습
        loss_entry = self.entry_agent.train_net(episode)
        loss_exit = self.exit_agent.train_net(episode)
        
        return (loss_entry + loss_exit) / 2.0
