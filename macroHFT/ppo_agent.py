import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import os
from common import config
from .xlstm_network import XLSTMNetwork

class Router(nn.Module):
    def __init__(self, input_dim, num_experts=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        if x.dim() == 3: 
            x = x[:, -1, :] 
        return self.net(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim=3, info_dim=11, hidden_dim=None, device='cpu'):
        self.device = device
        self.action_dim = action_dim 
        
        hidden_dim = hidden_dim if hidden_dim is not None else config.NETWORK_HIDDEN_DIM
        dropout = getattr(config, 'NETWORK_DROPOUT', 0.1)

        # Experts (Trend, Volatility, Sideways)
        self.experts = nn.ModuleList([
            XLSTMNetwork(state_dim, action_dim, info_dim, hidden_dim, config.NETWORK_NUM_LAYERS, dropout),
            XLSTMNetwork(state_dim, action_dim, info_dim, hidden_dim, config.NETWORK_NUM_LAYERS, dropout),
            XLSTMNetwork(state_dim, action_dim, info_dim, hidden_dim, config.NETWORK_NUM_LAYERS, dropout)
        ]).to(device)
        self.expert_names = ['trend', 'volatility', 'sideways']

        self.router = Router(state_dim, num_experts=3).to(device)
        
        self.lr = config.PPO_LEARNING_RATE
        # Optimizer
        self.opt_experts = [optim.Adam(exp.parameters(), lr=self.lr * 0.5, eps=1e-5) for exp in self.experts]
        self.opt_router = optim.Adam(self.router.parameters(), lr=self.lr, eps=1e-5)
        
        self.gamma = config.PPO_GAMMA
        self.lmbda = config.PPO_LAMBDA
        self.eps_clip = config.PPO_EPS_CLIP
        self.k_epochs = config.PPO_K_EPOCHS
        self.entropy_coef = config.PPO_ENTROPY_COEF
        
        self.data = []
        self.current_states = [None] * 3

    def reset_episode_states(self):
        self.current_states = [None] * 3

    def save_model(self, path):
        torch.save({
            'experts': [exp.state_dict() for exp in self.experts],
            'router': self.router.state_dict(),
            'opt_experts': [opt.state_dict() for opt in self.opt_experts],
            'opt_router': self.opt_router.state_dict(),
            'hidden_dim': getattr(config, 'NETWORK_HIDDEN_DIM', None),
        }, path)

    def load_model(self, path):
        if not os.path.exists(path):
            print(f"⚠️ 모델 파일 없음: {path}")
            return
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if 'experts' in checkpoint:
                for i, state in enumerate(checkpoint['experts']):
                    self.experts[i].load_state_dict(state, strict=False)
                if 'router' in checkpoint:
                    self.router.load_state_dict(checkpoint['router'], strict=False)
                print(f"✅ MacroHFT 로드 완료: {path}")
            elif 'model_state_dict' in checkpoint:
                print(f"⚠️ 구버전(단일) 모델 감지. Trend Expert에만 로드합니다.")
                self.experts[0].load_state_dict(checkpoint['model_state_dict'], strict=False)
        except Exception as e:
            print(f"❌ 모델 로드 에러: {e}")

    def select_action(self, state, action_mask=None, mode='router', expert_idx=0, deterministic=False):
        obs_seq, obs_info = state
        
        # 텐서 변환
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)
            obs_info = torch.as_tensor(obs_info, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            obs_seq = obs_seq.to(self.device)
            obs_info = obs_info.to(self.device)

        with torch.no_grad():
            if mode == 'expert':
                net = self.experts[expert_idx]
                logits, value, _, _, next_state, _ = net(
                    obs_seq, obs_info, states=self.current_states[expert_idx]
                )
                self.current_states[expert_idx] = next_state
            else: 
                logits_list = []
                for i, net in enumerate(self.experts):
                    l, _, _, _, ns, _ = net(obs_seq, obs_info, states=self.current_states[i])
                    logits_list.append(l)
                    self.current_states[i] = ns
                
                weights = self.router(obs_seq)
                stacked_logits = torch.stack(logits_list, dim=1)
                logits = torch.sum(weights.unsqueeze(-1) * stacked_logits, dim=1)
                value = 0.0

            if action_mask is not None:
                mask_tensor = torch.as_tensor(action_mask, device=self.device)
                logits = logits + (mask_tensor - 1) * 1e10

            dist = Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            
            if isinstance(value, torch.Tensor):
                value = value.item()

        return action.item(), dist.log_prob(action).item(), value

    def put_data(self, transition):
        self.data.append(transition)

    def train_net(self, episode=1, mode='router', expert_idx=0):
        if not self.data: return {}

        batch_data = list(zip(*self.data))
        
        # State Seq
        s_seq_np = np.array([x[0].cpu().numpy() if isinstance(x[0], torch.Tensor) else x[0] for x in batch_data[0]])
        s_seq = torch.tensor(s_seq_np, dtype=torch.float32, device=self.device).squeeze(1)
        
        # State Info
        s_info_np = np.array([x[1].cpu().numpy() if isinstance(x[1], torch.Tensor) else x[1] for x in batch_data[0]])
        s_info = torch.tensor(s_info_np, dtype=torch.float32, device=self.device).squeeze(1)
        
        a = torch.tensor(batch_data[1], dtype=torch.long, device=self.device)
        r = torch.tensor(batch_data[2], dtype=torch.float32, device=self.device)
        prob_a = torch.tensor(batch_data[4], dtype=torch.float32, device=self.device)
        done_mask = torch.tensor([0.0 if x else 1.0 for x in batch_data[5]], dtype=torch.float32, device=self.device)
        val = torch.tensor([x.item() if torch.is_tensor(x) else float(x) for x in batch_data[6]], dtype=torch.float32, device=self.device)
        
        # Volatility Label & Action Masks
        vol_label = torch.tensor([x if isinstance(x, float) else 0.0 for x in batch_data[7]], dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array(batch_data[8]), dtype=torch.float32, device=self.device)

        self.data = []

        # GAE Calculation
        with torch.no_grad():
            next_val = torch.roll(val, -1)
            next_val[-1] = 0.0
            deltas = r + self.gamma * next_val * done_mask - val
            deltas[-1] = r[-1] - val[-1]
            advantage = torch.zeros_like(r).to(self.device)
            running_adv = 0.0
            for t in reversed(range(len(r))):
                running_adv = deltas[t] + self.gamma * self.lmbda * running_adv * done_mask[t]
                advantage[t] = running_adv
            target_val = advantage + val
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        if mode == 'expert':
            optimizer = self.opt_experts[expert_idx]
            network = self.experts[expert_idx]
            target_params = network.parameters()
        else:
            optimizer = self.opt_router
            target_params = self.router.parameters()

        base_entropy = self.entropy_coef
        avg_vol = torch.mean(vol_label).item()
        dynamic_entropy_coef = base_entropy * (1.0 + 0.5 * avg_vol)

        avg_loss = 0.0
        
        for _ in range(self.k_epochs):
            optimizer.zero_grad() # 일반 zero_grad 사용

            if mode == 'expert':
                logits, curr_val, _, _, _, _ = network(s_seq, s_info)
            else:
                l_list = []
                # 라우터 모드에서는 전문가들 고정
                with torch.no_grad():
                    for exp in self.experts:
                        l, _, _, _, _, _ = exp(s_seq, s_info)
                        l_list.append(l)
                weights = self.router(s_seq)
                logits = torch.sum(weights.unsqueeze(-1) * torch.stack(l_list, dim=1), dim=1)
                curr_val = torch.zeros_like(val)

            logits = logits + (masks - 1) * 1e10
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(a)
            ratio = torch.exp(log_prob - prob_a)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = 0.0
            if mode == 'expert':
                critic_loss = 0.5 * nn.MSELoss()(curr_val.squeeze(), target_val)

            entropy_loss = -dynamic_entropy_coef * dist.entropy().mean()
            loss = actor_loss + critic_loss + entropy_loss

            # 일반 역전파
            loss.backward()
            nn.utils.clip_grad_norm_(target_params, 0.5)
            optimizer.step()
            
            avg_loss += loss.item()

        return {'Loss': avg_loss / self.k_epochs}