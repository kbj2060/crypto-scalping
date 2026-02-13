"""
PPO Agent for MacroHFT v5.0 - DISCRETE LEVERAGE
=================================================
- 이산 행동 공간 (15개: 방향 3 × 레버리지 5)
- CVaR 분위수 회귀 유지
- ICM 통합 유지 (보상은 train_ppo에서 더함)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
import os
from common import config
from .macrohft_network import TrendExpert, VolatilityExpert, SidewaysExpert

# ----------------------------------------------------------------------
# Quantile Huber Loss (기존)
# ----------------------------------------------------------------------
def quantile_huber_loss(quantiles, target, tau_hat):
    u = target - quantiles
    abs_u = u.abs()
    huber = torch.where(abs_u <= 1.0, 0.5 * u.pow(2), abs_u - 0.5)
    loss = (torch.abs(tau_hat - (u < 0).float()) * huber).mean()
    return loss

# ----------------------------------------------------------------------
# CVaR Loss
# ----------------------------------------------------------------------
def cvar_loss(quantiles, target, tau_hat, alpha=0.05):
    u = target - quantiles
    abs_u = u.abs()
    huber = torch.where(abs_u <= 1.0, 0.5 * u.pow(2), abs_u - 0.5)
    weights = torch.abs(tau_hat - (u < 0).float())
    cvar_mask = (tau_hat <= alpha).float()
    loss = (weights * cvar_mask * huber).mean() / (alpha + 1e-8)
    return loss

# ----------------------------------------------------------------------
# PPORouter (변경 없음)
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
# PPOAgent (메인) - Discrete Action
# ----------------------------------------------------------------------
class PPOAgent:
    def __init__(self, state_dim, action_dim, info_dim=11, device='cpu'):
        self.device = device
        self.action_dim = action_dim  # 15

        # 1. Experts (이산 행동 버전)
        self.experts = nn.ModuleList([
            TrendExpert(state_dim, action_dim, info_dim).to(device),
            VolatilityExpert(state_dim, action_dim, info_dim).to(device),
            SidewaysExpert(state_dim, action_dim, info_dim).to(device)
        ])
        self.expert_names = ['trend', 'volatility', 'sideways']

        # 2. Router
        self.router = PPORouter(state_dim, num_experts=3).to(device)
        self.router_eps_clip = getattr(config, 'ROUTER_EPS_CLIP', 0.2)
        self.router_entropy_coef = getattr(config, 'ROUTER_ENTROPY_COEF', 0.05)
        self.router_lr = getattr(config, 'ROUTER_LR', 3e-5)
        self.router_gamma = getattr(config, 'ROUTER_GAMMA', 0.99)
        self.router_lambda = getattr(config, 'PPO_LAMBDA', 0.95)
        self.router_exp3_eta = getattr(config, 'ROUTER_EXP3_ETA', 0.1)

        # 3. PPO 하이퍼파라미터
        self.lr = getattr(config, 'PPO_LEARNING_RATE', 3e-5)
        self.gammas = getattr(config, 'EXPERT_GAMMAS', {0:0.995, 1:0.99, 2:0.90})
        self.eps_clip = getattr(config, 'PPO_EPS_CLIP', 0.15)
        self.k_epochs = getattr(config, 'PPO_K_EPOCHS', 5)
        self.entropy_coef = getattr(config, 'PPO_ENTROPY_COEF', 0.1)   # 탐험 강화
        self.lr_decay = getattr(config, 'ENTROPY_DECAY', 0.999)

        # 4. Sharpe Ratio & Orthogonal 계수
        self.sharpe_coef = getattr(config, 'SHARPE_COEF', 0.001)
        self.ortho_coef = getattr(config, 'ORTHO_COEF', 0.01)

        # 5. CVaR 계수
        self.cvar_coef = getattr(config, 'CVAR_COEF', 0.5)
        self.cvar_alpha = getattr(config, 'CVAR_ALPHA', 0.05)

        # 6. Optimizers
        self.opt_experts = [optim.AdamW(exp.parameters(), lr=self.lr) for exp in self.experts]
        self.opt_router = optim.AdamW(self.router.parameters(), lr=self.router_lr)

        # 7. Mixed Precision
        self.use_amp = getattr(config, 'USE_AMP', False) and device == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # 8. EXP3.P 가중치
        self.router_weights = torch.ones(3, device=device) * 1.0

        # 9. Buffer
        self.data = []
        self.router_data = []

        # 10. Compile
        if getattr(config, 'USE_TORCH_COMPILE', False) and os.name != 'nt' and hasattr(torch, 'compile'):
            try:
                for i in range(3):
                    self.experts[i] = torch.compile(self.experts[i])
                self.router = torch.compile(self.router)
            except:
                pass

    # ------------------------------------------------------------------
    # 에피소드 상태 초기화 (호환성)
    # ------------------------------------------------------------------
    def reset_episode_states(self):
        pass

    # ------------------------------------------------------------------
    # 행동 선택 (이산)
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
            # ---------- 라우터 결정 ----------
            if mode == 'router':
                router_logits, router_value = self.router(obs_seq)
                probs = F.softmax(router_logits, dim=-1).squeeze(0)
                if deterministic:
                    selected_expert = router_logits.argmax(dim=-1).item()
                else:
                    dist = Categorical(probs=probs)
                    selected_expert = dist.sample().item()
                router_log_prob = torch.log(probs[selected_expert] + 1e-10).item()
            else:
                selected_expert = expert_idx
                router_log_prob = None
                router_value = None

            # ---------- 전문가 행동 ----------
            net = self.experts[selected_expert]
            out = net(obs_seq, obs_info)
            action_logits, value_mean, _, _ = out  # 이산 행동 logits

            # 액션 마스킹
            if action_mask is not None:
                mask_tensor = torch.as_tensor(action_mask, device=self.device)
                # action_mask는 15차원이어야 함 (15개 행동 각각 허용/금지)
                action_logits = action_logits + (mask_tensor - 1) * 1e9

            action_dist = Categorical(logits=action_logits)
            if deterministic:
                action_idx = action_logits.argmax(dim=-1).item()
            else:
                action_idx = action_dist.sample().item()
            log_prob = action_dist.log_prob(torch.tensor(action_idx, device=self.device)).item()
            value = value_mean.item()

            # 🔥 행동 인덱스 → (direction, leverage) 디코딩
            # action_idx: 0~14,  direction = action_idx % 3, leverage_idx = action_idx // 3
            direction = action_idx % 3
            leverage_idx = action_idx // 3
            scale = config.LEVERAGE_CANDIDATES[leverage_idx] / config.MAX_LEVERAGE  # 0.05, 0.25, 0.5, 0.75, 1.0
            action = (direction, scale)  # 기존 코드 호환성 유지

        return action, log_prob, value, selected_expert, router_log_prob, router_value

    # ------------------------------------------------------------------
    # 전이 저장 (action = (direction, scale) 튜플)
    # ------------------------------------------------------------------
    def put_data(self, transition):
        self.data.append(transition)
        if transition[10] is not None:
            obs_seq, obs_info = transition[0]
            self.router_data.append((
                obs_seq,                # 0: (1, T, D)
                obs_info,              # 1: (1, info_dim)
                transition[9],          # 2: selected_expert
                transition[10],         # 3: router_log_prob
                transition[2],         # 4: reward
                transition[5],         # 5: done
                transition[11],        # 6: router_value
            ))

    # ------------------------------------------------------------------
    # 전문가 학습 (배치 처리, GAE 정확)
    # ------------------------------------------------------------------
    def train_net(self, episode=1, mode='router', expert_idx=0):
        if not self.data:
            return {}

        batch_data = list(zip(*self.data))
        self.data = []

        # ----- [1] 전문가 학습용 텐서 변환 (squeeze(1) 필수) -----
        s_seq = torch.tensor(
            np.array([x[0].cpu().numpy() if isinstance(x[0], torch.Tensor) else x[0] for x in batch_data[0]]),
            dtype=torch.float32, device=self.device).squeeze(1)   # (B, 1, T, D) -> (B, T, D)

        s_info = torch.tensor(
            np.array([x[1].cpu().numpy() if isinstance(x[1], torch.Tensor) else x[1] for x in batch_data[0]]),
            dtype=torch.float32, device=self.device).squeeze(1)   # (B, 1, info_dim) -> (B, info_dim)

        # 🔥 행동: (direction, scale) → action_idx 복원
        a_dir = torch.tensor([x[0] for x in batch_data[1]], dtype=torch.long, device=self.device)
        a_scale = torch.tensor([x[1] for x in batch_data[1]], dtype=torch.float32, device=self.device)

        leverage_values = torch.tensor(config.LEVERAGE_CANDIDATES, device=self.device)
        target_leverage = a_scale * config.MAX_LEVERAGE
        leverage_idx = torch.argmin(torch.abs(target_leverage.unsqueeze(1) - leverage_values.unsqueeze(0)), dim=1)
        action_idx = a_dir + leverage_idx * 3  # 0~14

        r = torch.tensor(batch_data[2], dtype=torch.float32, device=self.device)
        prob_a = torch.tensor(batch_data[4], dtype=torch.float32, device=self.device)
        done_mask = torch.tensor([0.0 if x else 1.0 for x in batch_data[5]], dtype=torch.float32, device=self.device)
        val = torch.tensor([x.item() if torch.is_tensor(x) else float(x) for x in batch_data[6]], dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array(batch_data[8]), dtype=torch.float32, device=self.device)  # (B, 15)
        router_choices = torch.tensor(batch_data[9], dtype=torch.long, device=self.device)

        # 엔트로피 계수 감쇠
        self.entropy_coef = max(0.005, self.entropy_coef * config.ENTROPY_DECAY)

        # ========== [2] 라우터 학습 (EXP3.P + PPO) ==========
        router_loss_val = 0.0
        if mode == 'router' and len(self.router_data) > 0:
            router_batch = list(zip(*self.router_data))
            self.router_data = []

            router_states = torch.tensor(
                np.array([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in router_batch[0]]),
                dtype=torch.float32, device=self.device).squeeze(1)
            router_info = torch.tensor(
                np.array([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in router_batch[1]]),
                dtype=torch.float32, device=self.device).squeeze(1)
            router_actions = torch.tensor(router_batch[2], dtype=torch.long, device=self.device)
            router_old_logprobs = torch.tensor(router_batch[3], dtype=torch.float32, device=self.device)
            router_rewards = torch.tensor(router_batch[4], dtype=torch.float32, device=self.device)
            router_dones = torch.tensor([0.0 if x else 1.0 for x in router_batch[5]], dtype=torch.float32, device=self.device)
            router_old_values = torch.tensor(router_batch[6], dtype=torch.float32, device=self.device)

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

            with torch.no_grad():
                for i in range(len(router_rewards)):
                    exp3_reward = router_rewards[i].item()
                    prob = F.softmax(self.router(router_states[i:i+1])[0], dim=-1).squeeze(0)
                    estimated_reward = exp3_reward / (prob[router_actions[i]].item() + 1e-10)
                    self.router_weights[router_actions[i]] *= np.exp(self.router_exp3_eta * estimated_reward)
                    self.router_weights = self.router_weights / self.router_weights.sum() * 3.0

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
                # 🔥 Non-AMP 버전: router_total_loss 명시적 정의
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

                # 🔥 여기서 router_total_loss를 계산하고 저장
                router_total_loss = policy_loss + value_loss + entropy_loss

                router_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.router.parameters(), 0.5)
                self.opt_router.step()

            router_loss_val = router_total_loss.item()   # 이제 항상 정의됨

        # ========== [3] 전문가 PPO 학습 (이산 행동) ==========
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

                # ----- GAE (전문가) -----
                with torch.no_grad():
                    advantages = torch.zeros_like(r)
                    last_gae_lam = 0.0
                    for t in reversed(range(len(r))):
                        if t == len(r) - 1:
                            next_value = 0.0
                        else:
                            next_value = val[t + 1]
                        delta = r[t] + expert_gamma * next_value * done_mask[t] - val[t]
                        advantages[t] = delta + expert_gamma * config.PPO_LAMBDA * last_gae_lam * done_mask[t]
                        last_gae_lam = advantages[t]
                    target_val = advantages + val
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

                b_s_seq = s_seq[mask]
                b_s_info = s_info[mask]
                b_action_idx = action_idx[mask]
                b_prob_a = prob_a[mask]
                b_adv = advantages[mask]
                b_target_val = target_val[mask]
                b_masks = masks[mask]  # (B, 15)
                b_rewards = r[mask]

                optimizer = self.opt_experts[k]
                network = self.experts[k]

                def compute_loss():
                    out = network(b_s_seq, b_s_info)
                    action_logits, value, quantiles, proj_context = out

                    # 액션 마스킹 적용
                    action_logits = action_logits + (b_masks - 1) * 1e9
                    action_dist = Categorical(logits=action_logits)
                    log_prob = action_dist.log_prob(b_action_idx)
                    entropy = action_dist.entropy().mean()

                    ratio = torch.exp(log_prob - b_prob_a)

                    if (ratio > 3.0).any() or (ratio < 0.33).any():
                        return None

                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                    actor_loss = -torch.min(surr1, surr2).mean()

                    target_expanded = b_target_val.unsqueeze(1)
                    quantile_loss = quantile_huber_loss(quantiles, target_expanded, tau_hat)
                    cvar_loss_val = cvar_loss(quantiles, target_expanded, tau_hat, alpha=self.cvar_alpha)
                    critic_loss = quantile_loss + self.cvar_coef * cvar_loss_val

                    entropy_loss = -self.entropy_coef * entropy
                    total_loss = actor_loss + 0.5 * critic_loss + entropy_loss

                    # Sharpe Ratio Loss
                    if len(b_rewards) > 1:
                        ret_mean = b_rewards.mean()
                        ret_std = b_rewards.std() + 1e-6
                        sharpe = ret_mean / ret_std
                        sharpe = torch.tanh(sharpe)
                        sharpe_loss = 1.0 - sharpe
                        total_loss += self.sharpe_coef * sharpe_loss

                    # Orthogonal Regularization
                    if self.ortho_coef > 0 and b_s_seq.shape[0] >= 8:
                        ortho_loss = 0.0
                        other_contexts = []
                        with torch.no_grad():
                            for other_idx in range(3):
                                if other_idx != k:
                                    o_out = self.experts[other_idx](b_s_seq, b_s_info)
                                    other_contexts.append(o_out[-1].detach())
                        if other_contexts:
                            norm_k = F.normalize(proj_context, p=2, dim=1, eps=1e-4)
                            if not torch.isnan(norm_k).any():
                                for o_ctx in other_contexts:
                                    norm_o = F.normalize(o_ctx, p=2, dim=1, eps=1e-4)
                                    sim = (norm_k * norm_o).sum(dim=1).abs().mean()
                                    ortho_loss += sim
                        total_loss += self.ortho_coef * ortho_loss

                    return total_loss

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

        # LR Decay
        for opt in self.opt_experts + [self.opt_router]:
            for param_group in opt.param_groups:
                param_group['lr'] = max(1e-6, param_group['lr'] * 0.999)

        return {'Loss': avg_loss / max(1, self.k_epochs), 'Router_Loss': router_loss_val}

    # ------------------------------------------------------------------
    # 모델 저장/로드
    # ------------------------------------------------------------------
    def save_model(self, path):
        torch.save({
            'experts': [exp.state_dict() for exp in self.experts],
            'router': self.router.state_dict(),
            'opt_experts': [opt.state_dict() for opt in self.opt_experts],
            'opt_router': self.opt_router.state_dict(),
        }, path)

    def load_model(self, path):
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            for i, state in enumerate(ckpt['experts']):
                self.experts[i].load_state_dict(self._strip_prefix(state), strict=False)
            self.router.load_state_dict(self._strip_prefix(ckpt['router']), strict=False)
            for i, opt_state in enumerate(ckpt['opt_experts']):
                self.opt_experts[i].load_state_dict(opt_state)
            self.opt_router.load_state_dict(ckpt['opt_router'])

    def _strip_prefix(self, sd):
        return {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in sd.items()}