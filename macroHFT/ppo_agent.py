import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import random
import os
from common import config
from .macrohft_network import TrendExpert, VolatilityExpert, SidewaysExpert

# [Router] Gating Network (DDQN Structure)
class MoERouter(nn.Module):
    def __init__(self, input_dim, num_experts=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(), # Swish
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, num_experts) # Q-Values for Experts
        )
        
    def forward(self, x):
        if x.dim() == 3: x = x[:, -1, :] # Use last state
        return self.net(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim=3, info_dim=11, hidden_dim=None, device='cpu'):
        self.device = device
        self.action_dim = action_dim
        
        # === Mixture of Experts (MoE) ===
        # 각 전문가가 이제 'Transformer' 두뇌를 가짐
        self.experts = nn.ModuleList([
            TrendExpert(state_dim, action_dim, info_dim),      # 0: Trend (Deep Transformer)
            VolatilityExpert(state_dim, action_dim, info_dim), # 1: Volatility (Fast Transformer)
            SidewaysExpert(state_dim, action_dim, info_dim)    # 2: Sideways (Balanced)
        ]).to(device)
        self.expert_names = ['trend', 'volatility', 'sideways']

        # === Router (Manager) ===
        self.router = MoERouter(state_dim, num_experts=3).to(device)
        self.router_target = MoERouter(state_dim, num_experts=3).to(device)
        self.router_target.load_state_dict(self.router.state_dict())
        
        # Hyperparams
        self.lr = getattr(config, 'PPO_LEARNING_RATE', 1e-4)
        self.gamma = getattr(config, 'PPO_GAMMA', 0.99)
        self.lmbda = getattr(config, 'PPO_LAMBDA', 0.95)
        self.eps_clip = getattr(config, 'PPO_EPS_CLIP', 0.2)
        self.k_epochs = getattr(config, 'PPO_K_EPOCHS', 10)
        self.entropy_coef = getattr(config, 'PPO_ENTROPY_COEF', 0.01)
        
        # Optimizers (Different Learning Rates for Stability)
        self.opt_experts = [
            optim.AdamW(self.experts[0].parameters(), lr=self.lr * 0.5), # Deep net needs low LR
            optim.AdamW(self.experts[1].parameters(), lr=self.lr * 1.0),
            optim.AdamW(self.experts[2].parameters(), lr=self.lr * 0.8)
        ]
        self.opt_router = optim.AdamW(self.router.parameters(), lr=self.lr)
        self.router_loss_fn = nn.HuberLoss() # Robust Regression
        
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        
        self.data = []
        self.current_states = [None] * 3
        
        if device == 'cuda':
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None
            
        # [Compiler] Linux 환경이면 컴파일 (속도 2배)
        if os.name != 'nt' and hasattr(torch, 'compile'):
            try:
                import logging
                logger = logging.getLogger(__name__)
                logger.info("⚡ Compiling Transformers...")
                for i in range(3): 
                    self.experts[i] = torch.compile(self.experts[i])
                self.router = torch.compile(self.router)
                logger.info("✅ Compilation Done!")
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"⚠️ Compilation failed (ignored): {e}")

    def reset_episode_states(self):
        """에피소드 시작 시 internal states 리셋"""
        self.current_states = [None] * 3

    def save_model(self, path):
        """모델 저장 (experts + router + optimizers)"""
        torch.save({
            'experts': [exp.state_dict() for exp in self.experts],
            'router': self.router.state_dict(),
            'router_target': self.router_target.state_dict(),
            'opt_experts': [opt.state_dict() for opt in self.opt_experts],
            'opt_router': self.opt_router.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load_model(self, path):
        """모델 로드"""
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
                if 'router_target' in checkpoint:
                    self.router_target.load_state_dict(checkpoint['router_target'], strict=False)
                if 'epsilon' in checkpoint:
                    self.epsilon = checkpoint['epsilon']
                print(f"✅ MacroHFT MoE 로드 완료: {path}")
            elif 'model_state_dict' in checkpoint:
                print(f"⚠️ 구버전(단일) 모델 감지. Trend Expert에만 로드합니다.")
                self.experts[0].load_state_dict(checkpoint['model_state_dict'], strict=False)
        except Exception as e:
            print(f"❌ 모델 로드 에러: {e}")

    def select_action(self, state, action_mask=None, mode='router', expert_idx=0, deterministic=False):
        obs_seq, obs_info = state
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)
            obs_info = torch.as_tensor(obs_info, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            obs_seq = obs_seq.to(self.device)
            obs_info = obs_info.to(self.device)

        with torch.no_grad():
            # 1. Router가 상황 파악 후 전문가 호출
            if mode == 'router':
                q_values = self.router(obs_seq)
                if not deterministic and random.random() < self.epsilon:
                    selected_expert = random.randint(0, 2)
                else:
                    selected_expert = q_values.argmax(dim=-1).item()
            else:
                selected_expert = expert_idx

            # 2. 선택된 Transformer가 시세 예측
            net = self.experts[selected_expert]
            logits, value, _, _, _, _ = net(obs_seq, obs_info)

            # Masking
            if action_mask is not None:
                mask_tensor = torch.as_tensor(action_mask, device=self.device)
                logits = logits + (mask_tensor - 1) * 1e9

            dist = Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
            
            if isinstance(value, torch.Tensor): value = value.item()

        return action.item(), dist.log_prob(action).item(), value, selected_expert

    def put_data(self, transition):
        self.data.append(transition)

    def train_net(self, episode=1, mode='router', expert_idx=0, teacher_lambda=0.0):
        if not self.data: return {}
        
        batch_data = list(zip(*self.data))
        
        # Data Preparation
        s_seq = torch.tensor(np.array([x[0].cpu().numpy() if isinstance(x[0], torch.Tensor) else x[0] for x in batch_data[0]]), dtype=torch.float32, device=self.device).squeeze(1)
        s_info = torch.tensor(np.array([x[1].cpu().numpy() if isinstance(x[1], torch.Tensor) else x[1] for x in batch_data[0]]), dtype=torch.float32, device=self.device).squeeze(1)
        a = torch.tensor(batch_data[1], dtype=torch.long, device=self.device)
        r = torch.tensor(batch_data[2], dtype=torch.float32, device=self.device)
        prob_a = torch.tensor(batch_data[4], dtype=torch.float32, device=self.device)
        done_mask = torch.tensor([0.0 if x else 1.0 for x in batch_data[5]], dtype=torch.float32, device=self.device)
        val = torch.tensor([x.item() if torch.is_tensor(x) else float(x) for x in batch_data[6]], dtype=torch.float32, device=self.device)
        oracle_actions = torch.tensor(batch_data[9], dtype=torch.long, device=self.device)
        router_choices = torch.tensor(batch_data[10], dtype=torch.long, device=self.device)

        self.data = []

        # === 1. Router Update (DDQN) ===
        router_loss_val = 0.0
        if mode == 'router':
            with torch.no_grad():
                # Next state approximation (Using current for simplicity)
                expected_q = r 
            
            curr_q = self.router(s_seq)
            curr_q_selected = curr_q.gather(1, router_choices.unsqueeze(1)).squeeze()
            router_loss = self.router_loss_fn(curr_q_selected, expected_q)
            
            self.opt_router.zero_grad()
            if self.scaler:
                self.scaler.scale(router_loss).backward()
                self.scaler.step(self.opt_router)
                self.scaler.update()
            else:
                router_loss.backward()
                self.opt_router.step()
            
            router_loss_val = router_loss.item()
            
            if episode % 10 == 0:
                self.router_target.load_state_dict(self.router.state_dict())
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # === 2. Expert Update (PPO + BC) ===
        # GAE
        with torch.no_grad():
            next_val = torch.roll(val, -1)
            if done_mask[-1] == 1.0:
                next_val[-1] = val[-1]
            else:
                next_val[-1] = 0.0
                
            deltas = r + self.gamma * next_val * done_mask - val
            deltas[-1] = r[-1] + self.gamma * next_val[-1] * done_mask[-1] - val[-1]
            
            advantage = torch.zeros_like(r).to(self.device)
            running_adv = 0.0
            for t in reversed(range(len(r))):
                running_adv = deltas[t] + self.gamma * self.lmbda * running_adv * done_mask[t]
                advantage[t] = running_adv
            target_val = advantage + val
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        avg_loss = 0.0
        avg_bc_loss = 0.0

        for _ in range(self.k_epochs):
            # 선택된 전문가별로 마스킹하여 학습 (효율성)
            for k in range(3):
                # 해당 전문가가 선택된 샘플만 추출
                mask = (router_choices == k)
                if mask.sum() == 0: continue
                
                optimizer = self.opt_experts[k]
                network = self.experts[k]
                
                optimizer.zero_grad()
                
                # Sub-batch slicing
                b_s_seq = s_seq[mask]
                b_s_info = s_info[mask]
                b_a = a[mask]
                b_prob_a = prob_a[mask]
                b_adv = advantage[mask]
                b_target_val = target_val[mask]
                b_oracle = oracle_actions[mask]
                
                if self.scaler:
                    with torch.amp.autocast(self.device):
                        logits, curr_val, _, _, _, _ = network(b_s_seq, b_s_info)
                        dist = Categorical(logits=logits)
                        
                        # PPO Loss
                        log_prob = dist.log_prob(b_a)
                        ratio = torch.exp(log_prob - b_prob_a)
                        surr1 = ratio * b_adv
                        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = 0.5 * nn.MSELoss()(curr_val.squeeze(), b_target_val)
                        entropy_loss = -self.entropy_coef * dist.entropy().mean()
                        
                        # BC Loss (Teacher Forcing)
                        bc_loss = nn.CrossEntropyLoss()(logits, b_oracle)
                        
                        total_loss = actor_loss + critic_loss + entropy_loss + (teacher_lambda * bc_loss)
                    
                    self.scaler.scale(total_loss).backward()
                    nn.utils.clip_grad_norm_(network.parameters(), 0.5)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # CPU Fallback
                    logits, curr_val, _, _, _, _ = network(b_s_seq, b_s_info)
                    dist = Categorical(logits=logits)
                    log_prob = dist.log_prob(b_a)
                    ratio = torch.exp(log_prob - b_prob_a)
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = 0.5 * nn.MSELoss()(curr_val.squeeze(), b_target_val)
                    entropy_loss = -self.entropy_coef * dist.entropy().mean()
                    bc_loss = nn.CrossEntropyLoss()(logits, b_oracle)
                    total_loss = actor_loss + critic_loss + entropy_loss + (teacher_lambda * bc_loss)
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), 0.5)
                    optimizer.step()

                avg_loss += total_loss.item()
                avg_bc_loss += bc_loss.item()

        return {
            'Loss': avg_loss / (self.k_epochs * 3 + 1e-9),
            'Router_Loss': router_loss_val,
            'BC_Loss': avg_bc_loss / (self.k_epochs * 3 + 1e-9)
        }