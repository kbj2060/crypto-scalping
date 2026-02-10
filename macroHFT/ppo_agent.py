import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import random
import os
from common import config
from .macrohft_network import TrendExpert, VolatilityExpert, SidewaysExpert

# [Router] DDQN Router
class DDQNRouter(nn.Module):
    def __init__(self, input_dim, num_experts=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )
        
    def forward(self, x):
        if x.dim() == 3: x = x[:, -1, :] 
        return self.net(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim=3, info_dim=11, hidden_dim=None, device='cpu'):
        self.device = device
        
        # Experts (Transformer Based)
        self.experts = nn.ModuleList([
            TrendExpert(state_dim, action_dim, info_dim),
            VolatilityExpert(state_dim, action_dim, info_dim),
            SidewaysExpert(state_dim, action_dim, info_dim)
        ]).to(device)
        self.expert_names = ['trend', 'volatility', 'sideways']

        # Router (DDQN)
        self.router = DDQNRouter(state_dim, num_experts=3).to(device)
        self.router_target = DDQNRouter(state_dim, num_experts=3).to(device)
        self.router_target.load_state_dict(self.router.state_dict())
        
        self.lr = config.PPO_LEARNING_RATE
        self.gamma = config.PPO_GAMMA
        self.eps_clip = config.PPO_EPS_CLIP
        self.k_epochs = config.PPO_K_EPOCHS
        self.entropy_coef = config.PPO_ENTROPY_COEF
        
        # Optimizers
        self.opt_experts = [optim.AdamW(exp.parameters(), lr=self.lr) for exp in self.experts]
        self.opt_router = optim.AdamW(self.router.parameters(), lr=self.lr)
        self.router_loss_fn = nn.MSELoss()
        
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        
        self.data = []
        self.current_states = [None] * 3
        
        if device == 'cuda':
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None

        # Compiler
        if os.name != 'nt' and hasattr(torch, 'compile'):
             for i in range(3): self.experts[i] = torch.compile(self.experts[i])
             self.router = torch.compile(self.router)

    def select_action(self, state, action_mask=None, mode='router', expert_idx=0, deterministic=False):
        obs_seq, obs_info = state
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)
            obs_info = torch.as_tensor(obs_info, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            obs_seq = obs_seq.to(self.device)
            obs_info = obs_info.to(self.device)

        with torch.no_grad():
            if mode == 'router':
                q_values = self.router(obs_seq)
                if not deterministic and random.random() < self.epsilon:
                    selected_expert = random.randint(0, 2)
                else:
                    selected_expert = q_values.argmax(dim=-1).item()
            else:
                selected_expert = expert_idx

            net = self.experts[selected_expert]
            logits, value, _, _, next_state, _ = net(obs_seq, obs_info, states=self.current_states[selected_expert])
            self.current_states[selected_expert] = next_state

            if action_mask is not None:
                mask_tensor = torch.as_tensor(action_mask, device=self.device)
                logits = logits + (mask_tensor - 1) * 1e9

            dist = Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
            
            if isinstance(value, torch.Tensor): value = value.item()

        return action.item(), dist.log_prob(action).item(), value, selected_expert

    def put_data(self, transition):
        self.data.append(transition)

    def train_net(self, episode=1, mode='router', expert_idx=0):
        if not self.data: return {}
        
        batch_data = list(zip(*self.data))
        
        s_seq = torch.tensor(np.array([x[0].cpu().numpy() if isinstance(x[0], torch.Tensor) else x[0] for x in batch_data[0]]), dtype=torch.float32, device=self.device).squeeze(1)
        s_info = torch.tensor(np.array([x[1].cpu().numpy() if isinstance(x[1], torch.Tensor) else x[1] for x in batch_data[0]]), dtype=torch.float32, device=self.device).squeeze(1)
        a = torch.tensor(batch_data[1], dtype=torch.long, device=self.device)
        r = torch.tensor(batch_data[2], dtype=torch.float32, device=self.device)
        prob_a = torch.tensor(batch_data[4], dtype=torch.float32, device=self.device)
        done_mask = torch.tensor([0.0 if x else 1.0 for x in batch_data[5]], dtype=torch.float32, device=self.device)
        val = torch.tensor([x.item() if torch.is_tensor(x) else float(x) for x in batch_data[6]], dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array(batch_data[8]), dtype=torch.float32, device=self.device)
        
        # [Corrected] Selected Expert Index for Router Training
        router_choices = torch.tensor(batch_data[9], dtype=torch.long, device=self.device)

        self.data = []

        # --- Train Router (DDQN) ---
        router_loss_val = 0.0
        if mode == 'router':
            with torch.no_grad():
                expected_q = r # Simplified Target: Reward Only (Monte Carlo Style for stability)
            
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

        # --- Train Experts (PPO) ---
        with torch.no_grad():
            next_val = torch.roll(val, -1); next_val[-1] = 0.0
            deltas = r + self.gamma * next_val * done_mask - val
            advantage = torch.zeros_like(r).to(self.device)
            running_adv = 0.0
            for t in reversed(range(len(r))):
                running_adv = deltas[t] + self.gamma * config.PPO_LAMBDA * running_adv * done_mask[t]
                advantage[t] = running_adv
            target_val = advantage + val
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        avg_loss = 0.0

        for _ in range(self.k_epochs):
            # No Oracle/BC Loss anymore
            if mode == 'expert':
                optimizer = self.opt_experts[expert_idx]
                network = self.experts[expert_idx]
            else:
                # In router mode, train all experts based on usage
                # For simplicity in snippet, we skip or implement multi-expert train
                continue

            optimizer.zero_grad()
            
            if self.scaler:
                with torch.amp.autocast(self.device):
                    logits, curr_val, _, _, _, _ = network(s_seq, s_info)
                    logits = logits + (masks - 1) * 1e9
                    dist = Categorical(logits=logits)
                    
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - prob_a)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = 0.5 * nn.MSELoss()(curr_val.squeeze(), target_val)
                    entropy_loss = -self.entropy_coef * dist.entropy().mean()
                    
                    total_loss = actor_loss + critic_loss + entropy_loss
                
                self.scaler.scale(total_loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                logits, curr_val, _, _, _, _ = network(s_seq, s_info)
                # ... (CPU Logic Same)
                # ...
                pass

            avg_loss += total_loss.item()

        return {'Loss': avg_loss / self.k_epochs, 'Router_Loss': router_loss_val}
    
    # save/load methods (omitted for brevity, keep existing)
    def save_model(self, path):
        torch.save({
            'experts': [exp.state_dict() for exp in self.experts],
            'router': self.router.state_dict(),
            'opt_experts': [opt.state_dict() for opt in self.opt_experts],
            'opt_router': self.opt_router.state_dict(),
        }, path)

    def load_model(self, path):
        if not os.path.exists(path): return
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if 'experts' in checkpoint:
                for i, state in enumerate(checkpoint['experts']):
                    self.experts[i].load_state_dict(state, strict=False)
                if 'router' in checkpoint:
                    self.router.load_state_dict(checkpoint['router'], strict=False)
        except Exception as e:
            print(f"❌ Load Error: {e}")