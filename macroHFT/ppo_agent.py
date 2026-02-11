"""
PPO Agent for MacroHFT v8 SOTA
================================
1. Distributional RL (D-PPO): Quantile Huber Loss
2. Meta-Lambda: 전문가별 손실 회피 계수 자동 튜닝
3. Reward Distribution Learning: 보상 분위수 예측 Loss
4. Dream Team Ensemble, Atomic Saving, Compile 지원
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import random
import os
import glob
from common import config
from .macrohft_network import TrendExpert, VolatilityExpert, SidewaysExpert
from .macrohft_reward import LambdaMetaLearner  # [제안 1] 메타 λ 학습기

# ----------------------------------------------------------------------
# D-PPO Loss Function (Quantile Huber)
# ----------------------------------------------------------------------
def quantile_huber_loss(quantiles, target, tau_hat):
    """
    quantiles: (B, N) - 예측된 분위수
    target: (B, 1)   - TD Target
    tau_hat: (N,)    - 분위수 중간점
    """
    u = target - quantiles
    abs_u = u.abs()
    huber = torch.where(abs_u <= 1.0, 0.5 * u.pow(2), abs_u - 0.5)
    loss = (torch.abs(tau_hat - (u < 0).float()) * huber).mean()
    return loss

# ----------------------------------------------------------------------
# DDQN Router (기존 유지)
# ----------------------------------------------------------------------
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
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(x)

# ----------------------------------------------------------------------
# PPOAgent (메인)
# ----------------------------------------------------------------------
class PPOAgent:
    def __init__(self, state_dim, action_dim=3, info_dim=11, hidden_dim=None, device='cpu'):
        self.device = device

        # ------------------------------------------------------------------
        # 1. Experts (Transformer + Mamba + FiLM)
        # ------------------------------------------------------------------
        self.experts = nn.ModuleList([
            TrendExpert(state_dim, action_dim, info_dim).to(device),
            VolatilityExpert(state_dim, action_dim, info_dim).to(device),
            SidewaysExpert(state_dim, action_dim, info_dim).to(device)
        ])
        self.expert_names = ['trend', 'volatility', 'sideways']

        # ------------------------------------------------------------------
        # 2. 메타 λ 러너 (전문가별 손실 회피 계수)
        # ------------------------------------------------------------------
        self.lambda_learner = LambdaMetaLearner(num_experts=3, init_lambda=2.25).to(device)
        self.meta_lr = getattr(config, 'META_LAMBDA_LR', 1e-4)
        self.meta_optimizer = optim.Adam(self.lambda_learner.parameters(), lr=self.meta_lr)

        # 에피소드 PnL 저장 (메타 러닝용)
        self.episode_pnl_history = []  # (expert_idx, episode_pnl)

        # ------------------------------------------------------------------
        # 3. Router (DDQN)
        # ------------------------------------------------------------------
        self.router = DDQNRouter(state_dim, num_experts=3).to(device)
        self.router_target = DDQNRouter(state_dim, num_experts=3).to(device)
        self.router_target.load_state_dict(self.router.state_dict())

        # ------------------------------------------------------------------
        # 4. Hyperparameters
        # ------------------------------------------------------------------
        self.lr = getattr(config, 'PPO_LEARNING_RATE', 1e-4)
        default_gammas = {0: 0.995, 1: 0.99, 2: 0.90}
        self.gammas = getattr(config, 'EXPERT_GAMMAS', default_gammas)

        self.eps_clip = getattr(config, 'PPO_EPS_CLIP', 0.2)
        self.k_epochs = getattr(config, 'PPO_K_EPOCHS', 10)
        self.entropy_coef = getattr(config, 'PPO_ENTROPY_COEF', 0.01)

        # ------------------------------------------------------------------
        # 5. Optimizers
        # ------------------------------------------------------------------
        self.opt_experts = [optim.AdamW(exp.parameters(), lr=self.lr) for exp in self.experts]
        self.opt_router = optim.AdamW(self.router.parameters(), lr=self.lr)
        self.router_loss_fn = nn.MSELoss()

        # ------------------------------------------------------------------
        # 6. Exploration (Epsilon Greedy)
        # ------------------------------------------------------------------
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

        # ------------------------------------------------------------------
        # 7. Buffer & States
        # ------------------------------------------------------------------
        self.data = []
        self.current_states = [None] * 3

        # ------------------------------------------------------------------
        # 8. AMP Scaler (Mixed Precision)
        # ------------------------------------------------------------------
        if device == 'cuda' and getattr(config, 'USE_AMP', False):
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None

        # ------------------------------------------------------------------
        # 9. Torch Compile (Optional)
        # ------------------------------------------------------------------
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
    # 에피소드 시작 시 Transformer 상태 초기화
    # ------------------------------------------------------------------
    def reset_episode_states(self):
        self.current_states = [None] * 3

    # ------------------------------------------------------------------
    # torch.compile Prefix 제거 (모델 로딩 호환성)
    # ------------------------------------------------------------------
    def _strip_prefix(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "").replace("module.", "")
            new_state_dict[new_key] = v
        return new_state_dict

    # ------------------------------------------------------------------
    # 원자적 저장 (Atomic Save)
    # ------------------------------------------------------------------
    def save_model(self, path):
        tmp_path = path + ".tmp"
        try:
            torch.save({
                'experts': [exp.state_dict() for exp in self.experts],
                'router': self.router.state_dict(),
                'router_target': self.router_target.state_dict(),
                'opt_experts': [opt.state_dict() for opt in self.opt_experts],
                'opt_router': self.opt_router.state_dict(),
                'epsilon': self.epsilon,
                'lambda_learner': self.lambda_learner.state_dict(),  # 메타 λ 저장
            }, tmp_path)
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp_path, path)
        except Exception as e:
            print(f"❌ Save Error: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ------------------------------------------------------------------
    # 모델 로딩 (컴파일 호환)
    # ------------------------------------------------------------------
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
                if 'router_target' in checkpoint:
                    self.router_target.load_state_dict(self._strip_prefix(checkpoint['router_target']), strict=False)
                if 'epsilon' in checkpoint:
                    self.epsilon = checkpoint['epsilon']
                if 'lambda_learner' in checkpoint:
                    self.lambda_learner.load_state_dict(self._strip_prefix(checkpoint['lambda_learner']))
        except Exception as e:
            print(f"❌ Load Error: {e}")

    # ------------------------------------------------------------------
    # Dream Team Ensemble (최고 모델 합체)
    # ------------------------------------------------------------------
    def load_dream_team(self, base_dir):
        print(f"🧬 Assembling Dream Team from: {base_dir}")
        files = {
            'router': 'best_router.pth',
            0: 'best_trend.pth',
            1: 'best_volatility.pth',
            2: 'best_sideways.pth'
        }

        # Router Load
        router_path = self._find_file(base_dir, files['router'])
        if router_path:
            try:
                ckpt = torch.load(router_path, map_location=self.device)
                if 'router' in ckpt:
                    self.router.load_state_dict(self._strip_prefix(ckpt['router']))
                    self.router_target.load_state_dict(self._strip_prefix(ckpt.get('router_target', ckpt['router'])))
                if 'opt_router' in ckpt:
                    self.opt_router.load_state_dict(ckpt['opt_router'])
                if 'epsilon' in ckpt:
                    self.epsilon = ckpt['epsilon']
                if 'lambda_learner' in ckpt:
                    self.lambda_learner.load_state_dict(self._strip_prefix(ckpt['lambda_learner']))
                print(f"   ✅ Router System loaded from {os.path.basename(router_path)}")
            except Exception as e:
                print(f"   ❌ Router Load Failed: {e}")

        # Experts Load
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
    # 행동 선택 (Action Selection)
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
            # Router 모드: DDQN으로 전문가 선택
            if mode == 'router':
                q_values = self.router(obs_seq)
                if not deterministic and random.random() < self.epsilon:
                    selected_expert = random.randint(0, 2)
                else:
                    selected_expert = q_values.argmax(dim=-1).item()
            else:
                selected_expert = expert_idx

            # 선택된 전문가로 forward
            net = self.experts[selected_expert]
            out = net(obs_seq, obs_info, states=self.current_states[selected_expert])
            logits, value_mean = out[0], out[1]  # value는 quantile mean

            # Action Masking
            if action_mask is not None:
                mask_tensor = torch.as_tensor(action_mask, device=self.device)
                logits = logits + (mask_tensor - 1) * 1e9

            dist = Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
            value = value_mean.item() if isinstance(value_mean, torch.Tensor) else value_mean

        return action.item(), dist.log_prob(action).item(), value, selected_expert

    # ------------------------------------------------------------------
    # 전이 저장
    # ------------------------------------------------------------------
    def put_data(self, transition):
        self.data.append(transition)

    # ------------------------------------------------------------------
    # [제안 1] 메타 λ 업데이트
    # ------------------------------------------------------------------
    def update_meta_lambdas(self, episode_pnl_list):
        """
        전문가별 평균 수익률을 바탕으로 λ(log_lambdas)를 업데이트합니다.
        간단 휴리스틱: 수익률이 높으면 λ 감소 (리스크 허용), 낮으면 λ 증가 (손실 회피 강화)
        """
        if len(episode_pnl_list) < 1:
            return
        exp_pnl = {0: [], 1: [], 2: []}
        for idx, pnl in episode_pnl_list:
            exp_pnl[idx].append(pnl)
        for idx in range(3):
            if len(exp_pnl[idx]) == 0:
                continue
            avg_pnl = np.mean(exp_pnl[idx])
            # 목표 λ: 수익률 1% → λ=2.0, 2% → λ=1.8, -1% → λ=2.5
            target_lambda = 2.25 - 0.2 * avg_pnl  # avg_pnl은 소수점 단위 (0.01 = 1%)
            target_lambda = np.clip(target_lambda, 1.5, 3.0)
            current_log = self.lambda_learner.log_lambdas.data[idx].item()
            new_log = np.log(target_lambda)
            # EMA 스타일 업데이트
            self.lambda_learner.log_lambdas.data[idx] += 0.05 * (new_log - current_log)

    # ------------------------------------------------------------------
    # [제안 2] 보상 분포 loss 포함된 전문가 학습 (D-PPO + Reward Distribution)
    # ------------------------------------------------------------------
    def train_net(self, episode=1, mode='router', expert_idx=0):
        if not self.data:
            return {}

        batch_data = list(zip(*self.data))

        # ----- 텐서 변환 (모두 self.device로) -----
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

        self.data = []  # 버퍼 초기화

        # ----- Router 학습 (DDQN) -----
        router_loss_val = 0.0
        if mode == 'router':
            with torch.no_grad():
                returns = torch.zeros_like(r)
                running_return = 0.0
                router_gamma = 0.99
                for t in reversed(range(len(r))):
                    running_return = r[t] + router_gamma * running_return * done_mask[t]
                    returns[t] = running_return
                expected_q = returns

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
                clean_state = self._strip_prefix(self.router.state_dict())
                self.router_target.load_state_dict(clean_state)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # ----- Expert 학습 (D-PPO + 보상 분포) -----
        avg_loss = 0.0
        N_Q = getattr(config, 'NUM_QUANTILES', 32)
        tau_hat = torch.linspace(0.5 / N_Q, 1 - 0.5 / N_Q, N_Q, device=self.device)  # ✅ device 명시

        for _ in range(self.k_epochs):
            if mode == 'expert':
                target_experts = [expert_idx]
            else:
                target_experts = [0, 1, 2]

            for k in target_experts:
                mask = (router_choices == k)
                if mask.sum() == 0:
                    continue

                expert_gamma = self.gammas[k]

                # ----- Advantage & Target Value (GAE) -----
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
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                b_s_seq = s_seq[mask]
                b_s_info = s_info[mask]
                b_a = a[mask]
                b_prob_a = prob_a[mask]
                b_adv = advantage[mask]
                b_target_val = target_val[mask]
                b_masks = masks[mask]

                optimizer = self.opt_experts[k]
                network = self.experts[k]
                optimizer.zero_grad()

                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        out = network(b_s_seq, b_s_info)
                        logits, curr_val_mean = out[0], out[1]
                        quantiles = out[5]               # 가치 분위수 (32)
                        reward_quantiles = out[6] if len(out) > 6 else None  # 보상 분위수

                        logits = logits + (b_masks - 1) * 1e9
                        dist = Categorical(logits=logits)
                        log_prob = dist.log_prob(b_a)
                        ratio = torch.exp(log_prob - b_prob_a)

                        # PPO Clipping
                        surr1 = ratio * b_adv
                        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                        actor_loss = -torch.min(surr1, surr2).mean()

                        # D-PPO Critic Loss (Quantile Huber)
                        target_expanded = b_target_val.unsqueeze(1)
                        critic_loss = quantile_huber_loss(quantiles, target_expanded, tau_hat)

                        # Entropy Bonus
                        entropy_loss = -self.entropy_coef * dist.entropy().mean()

                        total_loss = actor_loss + 0.5 * critic_loss + entropy_loss

                        # [제안 2] 보상 분포 학습 Loss (선택적 가중치)
                        if reward_quantiles is not None:
                            r_batch = r[mask]
                            target_r = r_batch.unsqueeze(1).expand(-1, N_Q)
                            reward_loss = quantile_huber_loss(reward_quantiles, target_r, tau_hat)
                            total_loss += 0.1 * reward_loss  # 가중치 0.1

                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # CPU Fallback
                    out = network(b_s_seq, b_s_info)
                    logits, curr_val_mean = out[0], out[1]
                    quantiles = out[5]
                    reward_quantiles = out[6] if len(out) > 6 else None

                    logits = logits + (b_masks - 1) * 1e9
                    dist = Categorical(logits=logits)
                    log_prob = dist.log_prob(b_a)
                    ratio = torch.exp(log_prob - b_prob_a)
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                    actor_loss = -torch.min(surr1, surr2).mean()

                    target_expanded = b_target_val.unsqueeze(1)
                    critic_loss = quantile_huber_loss(quantiles, target_expanded, tau_hat)

                    entropy_loss = -self.entropy_coef * dist.entropy().mean()

                    total_loss = actor_loss + 0.5 * critic_loss + entropy_loss

                    if reward_quantiles is not None:
                        r_batch = r[mask]
                        target_r = r_batch.unsqueeze(1).expand(-1, N_Q)
                        reward_loss = quantile_huber_loss(reward_quantiles, target_r, tau_hat)
                        total_loss += 0.1 * reward_loss

                    total_loss.backward()
                    optimizer.step()

                avg_loss += total_loss.item()

        return {'Loss': avg_loss / self.k_epochs, 'Router_Loss': router_loss_val}