import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import os

from . import config
from .xlstm_network import XLSTMNetwork

class PPOAgent:
    def __init__(self, state_dim, action_dim=3, info_dim=15, hidden_dim=None, device='cpu'):
        self.device = device
        self.action_dim = action_dim 

        hidden_dim = hidden_dim if hidden_dim is not None else config.NETWORK_HIDDEN_DIM
        dropout = getattr(config, 'NETWORK_DROPOUT', 0.1)

        self.model = XLSTMNetwork(
            input_dim=state_dim,
            action_dim=action_dim,
            info_dim=info_dim,
            hidden_dim=hidden_dim,
            num_layers=config.NETWORK_NUM_LAYERS,
            dropout=dropout,
        ).to(device)

        self.lr = config.PPO_LEARNING_RATE
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)
        
        self.lr_warmup_episodes = getattr(config, 'PPO_LR_WARMUP_EPISODES', 0)
        self.temp_warmup_episodes = getattr(config, 'PPO_TEMP_WARMUP_EPISODES', 0)
        self.temperature = getattr(config, 'PPO_TEMP_INIT', 1.0)
        self.min_temp = getattr(config, 'PPO_TEMP_MIN', 0.5)
        self.temp_decay = getattr(config, 'PPO_TEMP_DECAY', 0.999)
        self.gamma = config.PPO_GAMMA
        self.lmbda = config.PPO_LAMBDA
        self.eps_clip = config.PPO_EPS_CLIP
        self.k_epochs = config.PPO_K_EPOCHS
        self.entropy_coef = config.PPO_ENTROPY_COEF
        self.data = []
        self.current_states = None

    def reset_episode_states(self):
        self.current_states = None

    def load_model(self, path):
        if not os.path.exists(path):
            print(f"⚠️ 모델 파일 없음: {path}")
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
        print(f"✅ 모델 로드 성공: {path}")

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hidden_dim': getattr(config, 'NETWORK_HIDDEN_DIM', None),
            'num_layers': getattr(config, 'NETWORK_NUM_LAYERS', None),
        }, path)

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
            # [Fix] Unpack 6 values (ignore gate_mean here)
            logits, value, cvar, _, self.current_states, _ = self.model(
                obs_seq, obs_info, states=self.current_states, temperature=None
            )

            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                logits = logits + (mask_tensor - 1) * 1e10

            current_temp = max(self.temperature, 1e-8)
            adjusted_logits = logits / current_temp

            dist = Categorical(logits=adjusted_logits)
            action = dist.sample()
            
            if action_mask is not None and action_mask[action.item()] == 0:
                allowed_indices = torch.where(mask_tensor == 1)[0]
                if len(allowed_indices) > 0:
                    allowed_logits = adjusted_logits[0, allowed_indices]
                    best_allowed_idx = torch.argmax(allowed_logits)
                    action = allowed_indices[best_allowed_idx].unsqueeze(0)

        return action.item(), dist.log_prob(action).item(), value.item()

    def put_data(self, transition):
        self.data.append(transition)

    def quantile_loss(self, preds, targets, quantile=0.05):
        errors = targets - preds
        loss = torch.max((quantile - 1) * errors, quantile * errors)
        return loss.mean()

    def train_net(self, episode=1):
        if not self.data:
            return {}

        if self.lr_warmup_episodes > 0 and episode <= self.lr_warmup_episodes:
            warmup_lr = self.lr * (episode / self.lr_warmup_episodes)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr

        # Data Unpacking
        s_seq_lst, s_info_lst, a_lst, r_lst, next_s_seq_lst, next_s_info_lst = [], [], [], [], [], []
        prob_a_lst, done_lst, old_v_lst, aux_target_lst, mask_lst = [], [], [], [], []

        for transition in self.data:
            if len(transition) == 9:
                s, a, r, next_s, prob_a, done, val, aux_target, mask = transition
            elif len(transition) == 8:
                s, a, r, next_s, prob_a, done, val, aux_target = transition
                mask = [1.0, 1.0, 1.0]
            else:
                continue

            s_seq_lst.append(s[0])
            s_info_lst.append(s[1])
            a_lst.append([a])
            r_lst.append([r])
            next_s_seq_lst.append(next_s[0])
            next_s_info_lst.append(next_s[1])
            prob_a_lst.append([prob_a])
            done_lst.append([0 if done else 1])
            old_v_lst.append([val])
            aux_target_lst.append([aux_target])
            mask_lst.append(mask)

        def to_tensor(data, dtype=torch.float):
            if isinstance(data[0], torch.Tensor):
                return torch.cat(data, dim=0).to(self.device)
            return torch.tensor(np.array(data), dtype=dtype).to(self.device)

        s_seq = to_tensor(s_seq_lst)
        s_info = to_tensor(s_info_lst)
        next_s_seq = to_tensor(next_s_seq_lst)
        next_s_info = to_tensor(next_s_info_lst)
        a = torch.tensor(a_lst, dtype=torch.long).to(self.device)
        r = torch.tensor(r_lst, dtype=torch.float).to(self.device)
        done_mask = torch.tensor(done_lst, dtype=torch.float).to(self.device)
        prob_a = torch.tensor(prob_a_lst, dtype=torch.float).to(self.device)
        old_v = torch.tensor(old_v_lst, dtype=torch.float).to(self.device)
        masks = torch.tensor(np.array(mask_lst), dtype=torch.float).to(self.device)

        self.data = []

        # GAE Calculation
        with torch.no_grad():
            # [Fix] Unpack 6 values
            _, v, _, _, _, _ = self.model(s_seq, s_info, states=None)
            _, next_v, _, _, _, _ = self.model(next_s_seq, next_s_info, states=None)

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

        # Log variables
        n_epochs_run = 0
        avg_total_loss = 0.0
        avg_actor_loss = 0.0
        avg_critic_loss = 0.0
        avg_risk_loss = 0.0
        avg_entropy_loss = 0.0
        avg_ortho_loss = 0.0
        avg_gate_mean = 0.0

        current_temp = max(self.temperature, 1e-8)

        for epoch in range(self.k_epochs):
            # [Fix] Unpack 6 values, receive gate_mean
            curr_logits, curr_v, curr_cvar, curr_aux, _, gate_mean = self.model(s_seq, s_info, states=None)

            # Masking & Temperature
            curr_logits = curr_logits + (masks - 1) * 1e10
            curr_logits = curr_logits / current_temp

            dist = Categorical(logits=curr_logits)
            curr_log_prob = dist.log_prob(a.squeeze()).unsqueeze(1)
            ratio = torch.exp(curr_log_prob - prob_a)

            with torch.no_grad():
                log_ratio = curr_log_prob - prob_a
                approx_kl = (torch.exp(log_ratio) - 1) - log_ratio
                
                # [Step 2 Fix] Allow at least 1 epoch to run (epoch > 0 condition)
                if approx_kl.mean() > config.PPO_KL_TARGET and epoch > 0:
                    break

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            
            # Dynamic Entropy
            base_entropy_coef = max(config.PPO_ENTROPY_MIN, self.entropy_coef * (0.999 ** episode))
            with torch.no_grad():
                pred_error = torch.sqrt(((curr_v - target_v)**2).mean())
                dynamic_scale = 1.0 + torch.clamp(pred_error, 0.0, 1.0)
            
            curr_entropy_coef = base_entropy_coef * dynamic_scale
            entropy_loss = curr_entropy_coef * dist.entropy().mean()
            
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic Loss
            if config.PPO_USE_VALUE_CLIP:
                v_clipped = old_v + torch.clamp(curr_v - old_v, -config.PPO_VALUE_CLIP_EPS, config.PPO_VALUE_CLIP_EPS)
                v_loss_1 = (curr_v - target_v) ** 2
                v_loss_2 = (v_clipped - target_v) ** 2
                critic_loss = 0.5 * torch.max(v_loss_1, v_loss_2).mean()
            else:
                critic_loss = 0.5 * F.mse_loss(curr_v, target_v)

            # Risk & Ortho Loss
            risk_loss = self.quantile_loss(curr_cvar, target_v, quantile=0.05) * 0.5
            
            strat_layer = self.model.strategy_processor.out_proj[0].weight
            wwt = torch.mm(strat_layer, strat_layer.t())
            identity = torch.eye(wwt.size(0)).to(self.device)
            ortho_loss = torch.norm(wwt - identity) * 0.01

            loss = actor_loss + critic_loss + risk_loss - entropy_loss + ortho_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

            # [Step 2 Fix] Increment run count
            n_epochs_run += 1
            
            # Log Accumulation
            avg_total_loss += loss.item()
            avg_actor_loss += actor_loss.item()
            avg_critic_loss += critic_loss.item()
            avg_risk_loss += risk_loss.item()
            avg_entropy_loss += entropy_loss.item()
            avg_ortho_loss += ortho_loss.item()
            
            # [Step 3 Fix] Accumulate gate_mean from return value
            if gate_mean is not None:
                avg_gate_mean += gate_mean

        if episode > self.temp_warmup_episodes:
            self.temperature = max(self.temperature * self.temp_decay, self.min_temp)

        # [Step 1 Fix] Remove trailing spaces in keys
        denom = max(1, n_epochs_run)
        return {
            "Loss/Total": avg_total_loss / denom,
            "Loss/Actor": avg_actor_loss / denom,
            "Loss/Critic": avg_critic_loss / denom,
            "Loss/Risk_CVaR": avg_risk_loss / denom,
            "Loss/Entropy": avg_entropy_loss / denom,
            "Loss/Ortho": avg_ortho_loss / denom,
            "Info/Gate_Mean": avg_gate_mean / denom,
        }