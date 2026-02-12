"""
PPO Agent for MacroHFT - Simplified, No Meta-Lambda
=====================================================
- Distributional Critic (Quantile Huber)
- PPO Router (Hard Selection + Value Network + GAE)
- No Meta-Lambda, No Auxiliary Losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import os
import glob
from common import config
from .macrohft_network import TrendExpert, VolatilityExpert, SidewaysExpert

# ----------------------------------------------------------------------
# D-PPO Loss Function (Quantile Huber)
# ----------------------------------------------------------------------
def quantile_huber_loss(quantiles, target, tau_hat):
    u = target - quantiles
    abs_u = u.abs()
    huber = torch.where(abs_u <= 1.0, 0.5 * u.pow(2), abs_u - 0.5)
    loss = (torch.abs(tau_hat - (u < 0).float()) * huber).mean()
    return loss

# ----------------------------------------------------------------------
# PPO Router (Policy Network + Value Head)
# ----------------------------------------------------------------------
class PPORouter(nn.Module):
    def __init__(self, input_dim, num_experts=3, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_dim // 2, num_experts)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        feat = self.shared(x)
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value

# ----------------------------------------------------------------------
# PPOAgent (메인) - Meta-Lambda 제거
# ----------------------------------------------------------------------
class PPOAgent:
    def __init__(self, state_dim, action_dim=3, info_dim=11, hidden_dim=None, device='cpu'):
        self.device = device

        # 1. Experts (d_model=128 통일, Transformer Only)
        self.experts = nn.ModuleList([
            TrendExpert(state_dim, action_dim, info_dim).to(device),
            VolatilityExpert(state_dim, action_dim, info_dim).to(device),
            SidewaysExpert(state_dim, action_dim, info_dim).to(device)
        ])
        self.expert_names = ['trend', 'volatility', 'sideways']

        # 2. PPO Router (하이퍼파라미터 보수적)
        self.router = PPORouter(state_dim, num_experts=3).to(device)
        self.router_eps_clip = getattr(config, 'ROUTER_EPS_CLIP', 0.1)
        self.router_entropy_coef = getattr(config, 'ROUTER_ENTROPY_COEF', 0.01)
        self.router_lr = getattr(config, 'ROUTER_LR', 1e-5)
        self.router_gamma = getattr(config, 'ROUTER_GAMMA', 0.99)
        self.router_lambda = getattr(config, 'PPO_LAMBDA', 0.95)

        # 3. 전문가 PPO 하이퍼파라미터
        self.lr = getattr(config, 'PPO_LEARNING_RATE', 1e-5)
        # 모든 전문가 Gamma 통일 (0.99)
        self.gammas = getattr(config, 'EXPERT_GAMMAS', {0:0.99, 1:0.99, 2:0.99})
        self.eps_clip = getattr(config, 'PPO_EPS_CLIP', 0.1)
        self.k_epochs = getattr(config, 'PPO_K_EPOCHS', 3)
        self.entropy_coef = getattr(config, 'PPO_ENTROPY_COEF', 0.01)

        # 4. Learning Rate Decay & Adaptive Entropy
        self.lr_decay = getattr(config, 'LR_DECAY', 0.999)
        self.base_entropy_coef = self.entropy_coef
        self.base_router_entropy_coef = self.router_entropy_coef

        # 5. Optimizers (SAM OFF, AMP OFF 기본)
        self.use_sam = getattr(config, 'USE_SAM', False)
        self.scaler = None
        if device == 'cuda' and getattr(config, 'USE_AMP', False):
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None

        # 전문가 옵티마이저 (AdamW)
        self.opt_experts = []
        for exp in self.experts:
            self.opt_experts.append(optim.AdamW(exp.parameters(), lr=self.lr))

        # 라우터 옵티마이저
        self.opt_router = optim.AdamW(self.router.parameters(), lr=self.router_lr)

        # 6. Exploration (ε-greedy 사용 안 함)
        self.epsilon = 0.0
        self.epsilon_min = 0.0
        self.epsilon_decay = 1.0

        # 7. Buffer & States
        self.data = []
        self.router_data = []
        self.current_states = [None] * 3

        # 8. Torch Compile (선택)
        use_compile = getattr(config, 'USE_TORCH_COMPILE', False)
        if use_compile and os.name != 'nt' and hasattr(torch, 'compile'):
            try:
                print("🚀 Applying torch.compile to models...")
                for i in range(3):
                    self.experts[i] = torch.compile(self.experts[i])
                self.router = torch.compile(self.router)
            except Exception as e:
                print(f"⚠️ torch.compile failed: {e}")

    # ------------------------------------------------------------------
    # 기본 메서드
    # ------------------------------------------------------------------
    def reset_episode_states(self):
        self.current_states = [None] * 3

    def _strip_prefix(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "").replace("module.", "")
            new_state_dict[new_key] = v
        return new_state_dict

    def save_model(self, path):
        tmp_path = path + ".tmp"
        try:
            torch.save({
                'experts': [exp.state_dict() for exp in self.experts],
                'router': self.router.state_dict(),
                'opt_experts': [opt.state_dict() for opt in self.opt_experts],
                'opt_router': self.opt_router.state_dict(),
                'epsilon': self.epsilon,
            }, tmp_path)
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp_path, path)
        except Exception as e:
            print(f"❌ Save Error: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def load_model(self, path):
        if not os.path.exists(path):
            return
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if 'experts' in checkpoint:
                for i, state in enumerate(checkpoint['experts']):
                    self.experts[i].load_state_dict(self._strip_prefix(state), strict=False)
                if 'router' in checkpoint:
                    self.router.load_state_dict(self._strip_prefix(checkpoint['router']), strict=False)
                if 'epsilon' in checkpoint:
                    self.epsilon = checkpoint['epsilon']
        except Exception as e:
            print(f"❌ Load Error: {e}")

    def load_dream_team(self, base_dir):
        """Dream Team Ensemble (라우터 + 각 전문가 최고 모델 로드)"""
        print(f"🧬 Assembling Dream Team from: {base_dir}")
        files = {
            'router': 'best_router.pth',
            0: 'best_trend.pth',
            1: 'best_volatility.pth',
            2: 'best_sideways.pth'
        }

        router_path = self._find_file(base_dir, files['router'])
        if router_path:
            try:
                ckpt = torch.load(router_path, map_location=self.device)
                if 'router' in ckpt:
                    self.router.load_state_dict(self._strip_prefix(ckpt['router']))
                if 'opt_router' in ckpt:
                    self.opt_router.load_state_dict(ckpt['opt_router'])
                if 'epsilon' in ckpt:
                    self.epsilon = ckpt['epsilon']
                print(f"   ✅ Router System loaded from {os.path.basename(router_path)}")
            except Exception as e:
                print(f"   ❌ Router Load Failed: {e}")

        expert_names = ['Trend', 'Volatility', 'Sideways']
        for idx in range(3):
            fname = files[idx]
            fpath = self._find_file(base_dir, fname)
            fallback = False
            if not fpath and router_path:
                fpath = router_path
                fallback = True
            if fpath:
                try:
                    ckpt = torch.load(fpath, map_location=self.device)
                    if 'experts' in ckpt and len(ckpt['experts']) > idx:
                        self.experts[idx].load_state_dict(self._strip_prefix(ckpt['experts'][idx]))
                    if 'opt_experts' in ckpt and len(ckpt['opt_experts']) > idx:
                        self.opt_experts[idx].load_state_dict(ckpt['opt_experts'][idx])
                    source = "Router Fallback" if fallback else os.path.basename(fpath)
                    print(f"   ✅ {expert_names[idx]} Expert loaded from {source}")
                except Exception as e:
                    print(f"   ❌ {expert_names[idx]} Load Failed: {e}")

    def _find_file(self, directory, suffix):
        exact = os.path.join(directory, suffix)
        if os.path.exists(exact):
            return exact
        candidates = glob.glob(os.path.join(directory, f"*{suffix}"))
        if candidates:
            return max(candidates, key=os.path.getctime)
        return None

    # ------------------------------------------------------------------
    # 행동 선택 (변경 없음)
    # ------------------------------------------------------------------
    def select_action(self, state, action_mask=None, mode='router', expert_idx=0, deterministic=False):
        obs_seq, obs_info = state
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            obs_info = torch.as_tensor(obs_info, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            obs_seq = obs_seq.to(self.device)
            obs_info = obs_info.to(self.device)

        with torch.no_grad():
            if mode == 'router':
                router_logits, router_value = self.router(obs_seq)
                router_dist = Categorical(logits=router_logits)
                if deterministic:
                    selected_expert = router_logits.argmax(dim=-1).item()
                else:
                    selected_expert = router_dist.sample().item()
                router_log_prob = router_dist.log_prob(torch.tensor([selected_expert], device=self.device)).item()
            else:
                selected_expert = expert_idx
                router_log_prob = None
                router_value = None

            net = self.experts[selected_expert]
            out = net(obs_seq, obs_info, states=self.current_states[selected_expert])
            logits, value_mean = out[0], out[1]

            if action_mask is not None:
                mask_tensor = torch.as_tensor(action_mask, device=self.device)
                logits = logits + (mask_tensor - 1) * 1e9

            dist = Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action).item()
            value = value_mean.item() if isinstance(value_mean, torch.Tensor) else value_mean

        return action.item(), log_prob, value, selected_expert, router_log_prob, router_value

    # ------------------------------------------------------------------
    # 전이 저장
    # ------------------------------------------------------------------
    def put_data(self, transition):
        self.data.append(transition)
        if transition[10] is not None:
            self.router_data.append((
                transition[0],   # state
                transition[9],   # selected_expert
                transition[10],  # router_log_prob
                transition[2],   # reward
                transition[5],   # done
                transition[11],  # router_value
            ))

    # ------------------------------------------------------------------
    # 통합 학습 (전문가 PPO + 라우터 PPO)
    # ------------------------------------------------------------------
    def train_net(self, episode=1, mode='router', expert_idx=0):
        if not self.data:
            return {}

        batch_data = list(zip(*self.data))
        self.data = []

        # ----- 텐서 변환 -----
        s_seq = torch.tensor(
            np.array([x[0].cpu().numpy() if isinstance(x[0], torch.Tensor) else x[0] for x in batch_data[0]]),
            dtype=torch.float32, device=self.device).squeeze(1)
        s_info = torch.tensor(
            np.array([x[1].cpu().numpy() if isinstance(x[1], torch.Tensor) else x[1] for x in batch_data[0]]),
            dtype=torch.float32, device=self.device).squeeze(1)
        a = torch.tensor(batch_data[1], dtype=torch.long, device=self.device)
        r = torch.tensor(batch_data[2], dtype=torch.float32, device=self.device)
        prob_a = torch.tensor(batch_data[4], dtype=torch.float32, device=self.device)
        done_mask = torch.tensor([0.0 if x else 1.0 for x in batch_data[5]], dtype=torch.float32, device=self.device)
        val = torch.tensor([x.item() if torch.is_tensor(x) else float(x) for x in batch_data[6]], dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array(batch_data[8]), dtype=torch.float32, device=self.device)
        router_choices = torch.tensor(batch_data[9], dtype=torch.long, device=self.device)

        # ----- Adaptive Entropy (리워드 폭락 시 탐험 강화) -----
        episode_return = r.sum().item()
        if episode_return < -100:
            self.entropy_coef = min(0.05, self.base_entropy_coef * 1.5)
            self.router_entropy_coef = min(0.05, self.base_router_entropy_coef * 1.5)
            self.base_entropy_coef = self.entropy_coef
            self.base_router_entropy_coef = self.router_entropy_coef
        else:
            self.entropy_coef = max(0.005, self.base_entropy_coef * 0.995)
            self.router_entropy_coef = max(0.005, self.base_router_entropy_coef * 0.995)
            self.base_entropy_coef = self.entropy_coef
            self.base_router_entropy_coef = self.router_entropy_coef

        # ------------------------------------------------------------------
        # [1] 라우터 PPO 학습 (GAE, Value Clipping)
        # ------------------------------------------------------------------
        router_loss_val = 0.0
        if mode == 'router' and len(self.router_data) > 0:
            router_batch = list(zip(*self.router_data))
            self.router_data = []

            router_states = torch.tensor(
                np.array([x[0].cpu().numpy() if isinstance(x[0], torch.Tensor) else x[0] for x in router_batch[0]]),
                dtype=torch.float32, device=self.device).squeeze(1)
            router_actions = torch.tensor(router_batch[1], dtype=torch.long, device=self.device)
            router_old_logprobs = torch.tensor(router_batch[2], dtype=torch.float32, device=self.device)
            router_rewards = torch.tensor(router_batch[3], dtype=torch.float32, device=self.device)
            router_dones = torch.tensor([0.0 if x else 1.0 for x in router_batch[4]], dtype=torch.float32, device=self.device)
            router_old_values = torch.tensor(router_batch[5], dtype=torch.float32, device=self.device)

            with torch.no_grad():
                _, router_values = self.router(router_states)
                router_values = router_values.squeeze(-1)
                advantages = torch.zeros_like(router_rewards)
                last_gae_lam = 0.0
                for t in reversed(range(len(router_rewards))):
                    if t == len(router_rewards) - 1:
                        next_value = 0.0
                    else:
                        next_value = router_values[t + 1]
                    delta = router_rewards[t] + self.router_gamma * next_value * router_dones[t] - router_values[t]
                    advantages[t] = delta + self.router_gamma * self.router_lambda * last_gae_lam * router_dones[t]
                    last_gae_lam = advantages[t]
                returns = advantages + router_values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            self.opt_router.zero_grad()
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    logits, new_values = self.router(router_states)
                    dist = Categorical(logits=logits)
                    new_logprobs = dist.log_prob(router_actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_logprobs - router_old_logprobs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.router_eps_clip, 1 + self.router_eps_clip) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_pred_clipped = router_old_values + torch.clamp(
                        new_values - router_old_values, -self.router_eps_clip, self.router_eps_clip
                    )
                    value_loss1 = F.mse_loss(new_values, returns)
                    value_loss2 = F.mse_loss(value_pred_clipped, returns)
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                    entropy_loss = -self.router_entropy_coef * entropy
                    router_total_loss = policy_loss + value_loss + entropy_loss

                self.scaler.scale(router_total_loss).backward()
                self.scaler.unscale_(self.opt_router)
                torch.nn.utils.clip_grad_norm_(self.router.parameters(), 0.5)
                self.scaler.step(self.opt_router)
                self.scaler.update()
            else:
                logits, new_values = self.router(router_states)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(router_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logprobs - router_old_logprobs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.router_eps_clip, 1 + self.router_eps_clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * F.mse_loss(new_values.squeeze(), returns)
                entropy_loss = -self.router_entropy_coef * entropy
                router_total_loss = policy_loss + value_loss + entropy_loss

                router_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.router.parameters(), 0.5)
                self.opt_router.step()

            router_loss_val = router_total_loss.item()

        # ------------------------------------------------------------------
        # [2] 전문가 PPO 학습 (D-PPO, Outlier Rejection)
        # ------------------------------------------------------------------
        avg_loss = 0.0
        N_Q = getattr(config, 'NUM_QUANTILES', 32)
        tau_hat = torch.linspace(0.5 / N_Q, 1 - 0.5 / N_Q, N_Q, device=self.device)

        for _ in range(self.k_epochs):
            target_experts = [expert_idx] if mode == 'expert' else [0, 1, 2]

            for k in target_experts:
                mask = (router_choices == k)
                if mask.sum() < 4:
                    continue

                expert_gamma = self.gammas[k]

                with torch.no_grad():
                    next_val = torch.roll(val, -1)
                    next_val[-1] = 0.0
                    deltas = r + expert_gamma * next_val * done_mask - val
                    advantage = torch.zeros_like(r).to(self.device)
                    running_adv = 0.0
                    lambda_gae = getattr(config, 'PPO_LAMBDA', 0.95)
                    for t in reversed(range(len(r))):
                        running_adv = deltas[t] + expert_gamma * lambda_gae * running_adv * done_mask[t]
                        advantage[t] = running_adv
                    target_val = advantage + val
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

                b_s_seq = s_seq[mask]
                b_s_info = s_info[mask]
                b_a = a[mask]
                b_prob_a = prob_a[mask]
                b_adv = advantage[mask]
                b_target_val = target_val[mask]
                b_masks = masks[mask]

                optimizer = self.opt_experts[k]
                network = self.experts[k]

                def compute_loss():
                    out = network(b_s_seq, b_s_info)
                    for o in out:
                        if torch.is_tensor(o) and (torch.isnan(o).any() or torch.isinf(o).any()):
                            return None

                    logits, curr_val_mean = out[0], out[1]
                    quantiles = out[5]

                    logits = logits + (b_masks - 1) * 1e9
                    dist = Categorical(logits=logits)
                    log_prob = dist.log_prob(b_a)
                    ratio = torch.exp(log_prob - b_prob_a)

                    # Outlier Rejection
                    if (ratio > 3.0).any() or (ratio < 0.33).any():
                        return None

                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                    actor_loss = -torch.min(surr1, surr2).mean()

                    target_expanded = b_target_val.unsqueeze(1)
                    critic_loss = quantile_huber_loss(quantiles, target_expanded, tau_hat)

                    entropy_loss = -self.entropy_coef * dist.entropy().mean()

                    total_loss = actor_loss + 0.5 * critic_loss + entropy_loss
                    return total_loss

                # ----- AdamW Step -----
                optimizer.zero_grad()
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        loss = compute_loss()
                    if loss is None or torch.isnan(loss) or torch.isinf(loss):
                        continue
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss = compute_loss()
                    if loss is None or torch.isnan(loss) or torch.isinf(loss):
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
                    optimizer.step()

                avg_loss += loss.item()

        # ----- Learning Rate Decay -----
        for opt in self.opt_experts + [self.opt_router]:
            for param_group in opt.param_groups:
                param_group['lr'] = max(1e-6, param_group['lr'] * self.lr_decay)

        return {'Loss': avg_loss / self.k_epochs, 'Router_Loss': router_loss_val}