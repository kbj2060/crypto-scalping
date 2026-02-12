"""
PPO Agent for MacroHFT v8 SOTA - PPO Router Version
====================================================
1. Distributional RL (D-PPO): Quantile Huber Loss
2. Meta-Lambda: 전문가별 손실 회피 계수 자동 튜닝
3. Reward Distribution Learning: 보상 분위수 예측 Loss
4. PPO Router: 전문가 선택을 PPO로 학습 (DDQN 대체)
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
from .macrohft_reward import LambdaMetaLearner

# ----------------------------------------------------------------------
# SAM (Sharpness-Aware Minimization) Optimizer - AMP 호환 버전
# ----------------------------------------------------------------------
class SAM(torch.optim.Optimizer):
    """SAM optimizer with optional scaler support for AMP compatibility."""
    def __init__(self, params, base_optimizer, scaler=None, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.scaler = scaler
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        if self.scaler is not None:
            self.scaler.step(self.base_optimizer)
        else:
            self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"] if p.grad is not None
            ]), p=2
        )
        return norm

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
# [변경] PPO Router (Policy Network) - DDQNRouter 대체
# ----------------------------------------------------------------------
class PPORouter(nn.Module):
    """PPO Router: state -> expert selection logits"""
    def __init__(self, input_dim, num_experts=3, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
    def forward(self, x):
        # x: (B, T, D) or (B, D)
        if x.dim() == 3:
            x = x[:, -1, :]   # last token pooling
        return self.net(x)    # (B, num_experts)

# ----------------------------------------------------------------------
# PPOAgent (메인)
# ----------------------------------------------------------------------
class PPOAgent:
    def __init__(self, state_dim, action_dim=3, info_dim=11, hidden_dim=None, device='cpu'):
        self.device = device

        # ------------------------------------------------------------------
        # 1. Experts (변경 없음)
        # ------------------------------------------------------------------
        self.experts = nn.ModuleList([
            TrendExpert(state_dim, action_dim, info_dim).to(device),
            VolatilityExpert(state_dim, action_dim, info_dim).to(device),
            SidewaysExpert(state_dim, action_dim, info_dim).to(device)
        ])
        self.expert_names = ['trend', 'volatility', 'sideways']

        # ------------------------------------------------------------------
        # 2. 메타 λ 러너 (변경 없음)
        # ------------------------------------------------------------------
        self.lambda_learner = LambdaMetaLearner(num_experts=3, init_lambda=2.25).to(device)
        self.meta_lr = getattr(config, 'META_LAMBDA_LR', 1e-4)
        self.meta_optimizer = optim.Adam(self.lambda_learner.parameters(), lr=self.meta_lr)
        self.episode_pnl_history = []

        # ------------------------------------------------------------------
        # 3. [변경] PPO Router (DDQN Router 대체)
        # ------------------------------------------------------------------
        self.router = PPORouter(state_dim, num_experts=3).to(device)
        # PPO 하이퍼파라미터 (라우터 전용)
        self.router_eps_clip = getattr(config, 'ROUTER_EPS_CLIP', 0.2)
        self.router_entropy_coef = getattr(config, 'ROUTER_ENTROPY_COEF', 0.01)
        self.router_lr = getattr(config, 'ROUTER_LR', 3e-5)  # 전문가보다 낮게
        self.router_gamma = getattr(config, 'ROUTER_GAMMA', 0.99)

        # ------------------------------------------------------------------
        # 4. Hyperparameters (전문가용)
        # ------------------------------------------------------------------
        self.lr = getattr(config, 'PPO_LEARNING_RATE', 1e-4)
        default_gammas = {0: 0.995, 1: 0.99, 2: 0.90}
        self.gammas = getattr(config, 'EXPERT_GAMMAS', default_gammas)
        self.eps_clip = getattr(config, 'PPO_EPS_CLIP', 0.2)
        self.k_epochs = getattr(config, 'PPO_K_EPOCHS', 10)
        self.entropy_coef = getattr(config, 'PPO_ENTROPY_COEF', 0.01)

        # ------------------------------------------------------------------
        # 5. Optimizers (전문가: SAM or AdamW, 라우터: AdamW)
        # ------------------------------------------------------------------
        self.use_sam = getattr(config, 'USE_SAM', True)
        # AMP 설정
        self.scaler = None
        if device == 'cuda' and getattr(config, 'USE_AMP', False):
            if self.use_sam:
                print("⚠️ SAM enabled – disabling AMP to avoid conflict")
                self.scaler = None
            else:
                self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None

        # 전문가 옵티마이저
        self.opt_experts = []
        for exp in self.experts:
            if self.use_sam:
                self.opt_experts.append(
                    SAM(exp.parameters(), optim.AdamW, scaler=self.scaler,
                        lr=self.lr, rho=0.02)
                )
            else:
                self.opt_experts.append(optim.AdamW(exp.parameters(), lr=self.lr))

        # [변경] 라우터 옵티마이저 (PPO, AdamW)
        self.opt_router = optim.AdamW(self.router.parameters(), lr=self.router_lr)

        # ------------------------------------------------------------------
        # 6. Exploration (Epsilon greedy - PPO Router에서는 사용 안 함, 대신 entropy 탐험)
        #    단, 초기 전문가 선택 탐험을 위해 유지 (select_action에서 사용)
        # ------------------------------------------------------------------
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

        # ------------------------------------------------------------------
        # 7. Buffer & States
        # ------------------------------------------------------------------
        self.data = []          # 전문가 PPO 데이터 (기존)
        self.router_data = []   # 라우터 PPO 데이터 (별도 저장)
        self.current_states = [None] * 3

        # ------------------------------------------------------------------
        # 8. Torch Compile (Optional)
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
    # 에피소드 시작 시 상태 초기화
    # ------------------------------------------------------------------
    def reset_episode_states(self):
        self.current_states = [None] * 3

    # ------------------------------------------------------------------
    # torch.compile Prefix 제거
    # ------------------------------------------------------------------
    def _strip_prefix(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "").replace("module.", "")
            new_state_dict[new_key] = v
        return new_state_dict

    # ------------------------------------------------------------------
    # 원자적 저장 (전문가 + 라우터)
    # ------------------------------------------------------------------
    def save_model(self, path):
        tmp_path = path + ".tmp"
        try:
            torch.save({
                'experts': [exp.state_dict() for exp in self.experts],
                'router': self.router.state_dict(),
                'opt_experts': [opt.state_dict() for opt in self.opt_experts],
                'opt_router': self.opt_router.state_dict(),
                'epsilon': self.epsilon,
                'lambda_learner': self.lambda_learner.state_dict(),
            }, tmp_path)
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp_path, path)
        except Exception as e:
            print(f"❌ Save Error: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ------------------------------------------------------------------
    # 모델 로딩
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
                if 'epsilon' in checkpoint:
                    self.epsilon = checkpoint['epsilon']
                if 'lambda_learner' in checkpoint:
                    self.lambda_learner.load_state_dict(self._strip_prefix(checkpoint['lambda_learner']))
        except Exception as e:
            print(f"❌ Load Error: {e}")

    # ------------------------------------------------------------------
    # Dream Team Ensemble (기존과 동일)
    # ------------------------------------------------------------------
    def load_dream_team(self, base_dir):
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
                if 'lambda_learner' in ckpt:
                    self.lambda_learner.load_state_dict(self._strip_prefix(ckpt['lambda_learner']))
                print(f"   ✅ Router System loaded from {os.path.basename(router_path)}")
            except Exception as e:
                print(f"   ❌ Router Load Failed: {e}")
        # 2. Experts Load (각각의 파일에서 해당 인덱스 Expert만 추출)
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
    # [변경] 행동 선택 (PPO Router 통합)
    # ------------------------------------------------------------------
    def select_action(self, state, action_mask=None, mode='router', expert_idx=0, deterministic=False):
        """
        Returns:
            action: 전문가의 행동 (0,1,2)
            log_prob: 전문가 행동의 log probability
            value: 선택된 전문가의 value_mean
            selected_expert: 선택된 전문가 인덱스
            router_log_prob: [mode='router'일 때] 라우터의 log probability (PPO 학습용)
        """
        obs_seq, obs_info = state
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            obs_info = torch.as_tensor(obs_info, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            obs_seq = obs_seq.to(self.device)
            obs_info = obs_info.to(self.device)

        with torch.no_grad():
            # ---------- 전문가 선택 ----------
            if mode == 'router':
                # PPO Router forward
                router_logits = self.router(obs_seq)   # (1, 3)
                router_dist = Categorical(logits=router_logits)
                
                if deterministic:
                    selected_expert = router_logits.argmax(dim=-1).item()
                    router_log_prob = router_dist.log_prob(torch.tensor([selected_expert], device=self.device)).item()
                else:
                    # epsilon greedy? -> PPO는 entropy로 탐험, epsilon은 선택사항
                    if random.random() < self.epsilon:
                        selected_expert = random.randint(0, 2)
                        # 선택된 전문가에 대한 log prob 재계산 (epsilon 정책과 별개, PPO는 실제 샘플링된 action의 log prob 필요)
                        # 간단히: epsilon일 때도 라우터의 log prob을 저장 (on-policy 유지)
                        router_log_prob = router_dist.log_prob(torch.tensor([selected_expert], device=self.device)).item()
                    else:
                        selected_expert = router_dist.sample().item()
                        router_log_prob = router_dist.log_prob(torch.tensor([selected_expert], device=self.device)).item()
            else:
                # expert 모드: 고정 전문가
                selected_expert = expert_idx
                router_log_prob = None   # 사용 안 함

            # ---------- 선택된 전문가로 행동 결정 ----------
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

        return action.item(), log_prob, value, selected_expert, router_log_prob

    # ------------------------------------------------------------------
    # 전이 저장 (전문가 + 라우터 데이터 분리 저장)
    # ------------------------------------------------------------------
    def put_data(self, transition):
        """
        transition = (
            state,               # 0: obs_seq, obs_info
            action,              # 1: 전문가 행동
            reward,              # 2: 즉각 보상
            next_state,          # 3: next obs
            prob,               # 4: 전문가 행동 log_prob
            done,               # 5: 종료 여부
            val,                # 6: 전문가 value
            volatility_label,   # 7: (디버깅)
            action_mask,        # 8: action mask
            selected_expert,    # 9: 선택된 전문가
            router_log_prob     # 10: [추가] 라우터 log_prob (router 모드일 때만)
        )
        """
        self.data.append(transition)   # 전문가 PPO용
        if transition[10] is not None:
            # 라우터 데이터 별도 저장 (selected_expert, router_log_prob, reward, done, state 등)
            self.router_data.append((
                transition[0],          # state
                transition[9],          # selected_expert
                transition[10],         # router_log_prob
                transition[2],         # reward
                transition[5],         # done
                transition[6],         # expert value (baseline으로 사용)
            ))

    # ------------------------------------------------------------------
    # 메타 λ 업데이트 (변경 없음)
    # ------------------------------------------------------------------
    def update_meta_lambdas(self, episode_pnl_list):
        if len(episode_pnl_list) < 1:
            return
        exp_pnl = {0: [], 1: [], 2: []}
        for idx, pnl in episode_pnl_list:
            exp_pnl[idx].append(pnl)
        for idx in range(3):
            if len(exp_pnl[idx]) == 0:
                continue
            avg_pnl = np.mean(exp_pnl[idx])
            target_lambda = 2.25 - 0.2 * avg_pnl   # avg_pnl은 소수점 단위 (0.01 = 1%)
            target_lambda = np.clip(target_lambda, 1.5, 3.0)
            current_log = self.lambda_learner.log_lambdas.data[idx].item()
            new_log = np.log(target_lambda)
            self.lambda_learner.log_lambdas.data[idx] += 0.05 * (new_log - current_log)

    # ------------------------------------------------------------------
    # [변경] 통합 학습 (전문가 PPO + 라우터 PPO)
    # ------------------------------------------------------------------
    def train_net(self, episode=1, mode='router', expert_idx=0):
        if not self.data:
            return {}

        batch_data = list(zip(*self.data))
        self.data = []  # 전문가 버퍼 초기화

        # ----- 텐서 변환 (전문가 학습용) -----
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
        # router_log_prob는 batch_data[10] -> 라우터 학습에서 사용

        # ------------------------------------------------------------------
        # [1] 라우터 PPO 학습 (mode == 'router'일 때만)
        # ------------------------------------------------------------------
        router_loss_val = 0.0
        if mode == 'router' and len(self.router_data) > 0:
            router_batch = list(zip(*self.router_data))
            self.router_data = []  # 버퍼 초기화

            # 라우터 데이터 텐서화
            router_states = torch.tensor(
                np.array([x[0].cpu().numpy() if isinstance(x[0], torch.Tensor) else x[0] for x in router_batch[0]]),
                dtype=torch.float32, device=self.device).squeeze(1)
            router_actions = torch.tensor(router_batch[1], dtype=torch.long, device=self.device)
            router_old_logprobs = torch.tensor(router_batch[2], dtype=torch.float32, device=self.device)
            router_rewards = torch.tensor(router_batch[3], dtype=torch.float32, device=self.device)
            router_dones = torch.tensor([0.0 if x else 1.0 for x in router_batch[4]], dtype=torch.float32, device=self.device)
            router_baselines = torch.tensor(router_batch[5], dtype=torch.float32, device=self.device)  # expert value_mean

            # ----- GAE for Router -----
            with torch.no_grad():
                # 라우터의 가치 네트워크가 없으므로, expert value를 baseline으로 사용
                returns = torch.zeros_like(router_rewards)
                running_return = 0.0
                for t in reversed(range(len(router_rewards))):
                    running_return = router_rewards[t] + self.router_gamma * running_return * router_dones[t]
                    returns[t] = running_return
                advantage = returns - returns.mean()   # ✅ 간단하고 강건한 방법
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # 정규화

            # ----- PPO Update for Router (single epoch, on-policy) -----
            # 라우터는 1 epoch만 업데이트 (on-policy)
            logits = self.router(router_states)
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(router_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logprobs - router_old_logprobs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.router_eps_clip, 1 + self.router_eps_clip) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -self.router_entropy_coef * entropy

            router_total_loss = policy_loss + entropy_loss

            # 옵티마이저 스텝
            self.opt_router.zero_grad()
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    logits = self.router(router_states)
                    dist = Categorical(logits=logits)
                    new_logprobs = dist.log_prob(router_actions)
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(new_logprobs - router_old_logprobs)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.router_eps_clip, 1 + self.router_eps_clip) * advantage
                    policy_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = -self.router_entropy_coef * entropy
                    router_total_loss = policy_loss + entropy_loss
                self.scaler.scale(router_total_loss).backward()
                self.scaler.unscale_(self.opt_router)
                torch.nn.utils.clip_grad_norm_(self.router.parameters(), 1.0)
                self.scaler.step(self.opt_router)
                self.scaler.update()
            else:
                router_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.router.parameters(), 1.0)
                self.opt_router.step()

            router_loss_val = router_total_loss.item()

            # Epsilon decay (exploration)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # ------------------------------------------------------------------
        # [2] 전문가 PPO 학습 (기존 코드 유지, SAM/AMP 호환)
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

                # ----- Advantage & Target Value -----
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

                # Batch Slicing
                b_s_seq = s_seq[mask]
                b_s_info = s_info[mask]
                b_a = a[mask]
                b_prob_a = prob_a[mask]
                b_adv = advantage[mask]
                b_target_val = target_val[mask]
                b_masks = masks[mask]
                b_rewards = r[mask]

                optimizer = self.opt_experts[k]
                network = self.experts[k]

                # ----------------------------------------------------------
                # Loss 계산 함수 (Closure)
                # ----------------------------------------------------------
                def compute_loss():
                    out = network(b_s_seq, b_s_info)
                    for o in out:
                        if torch.is_tensor(o) and (torch.isnan(o).any() or torch.isinf(o).any()):
                            return None

                    logits, curr_val_mean = out[0], out[1]
                    quantiles = out[5]
                    reward_quantiles = out[6]
                    context_k = out[7]

                    logits = logits + (b_masks - 1) * 1e9
                    dist = Categorical(logits=logits)
                    log_prob = dist.log_prob(b_a)
                    ratio = torch.exp(log_prob - b_prob_a)

                    # 1. PPO Actor Loss
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # 2. D-PPO Critic Loss
                    target_expanded = b_target_val.unsqueeze(1)
                    critic_loss = quantile_huber_loss(quantiles, target_expanded, tau_hat)

                    # 3. Entropy Bonus
                    entropy_loss = -self.entropy_coef * dist.entropy().mean()

                    total_loss = actor_loss + 0.5 * critic_loss + entropy_loss

                    # 4. Orthogonal Regularization (선택)
                    if mode == 'router' and b_s_seq.shape[0] >= 8:
                        ortho_loss = 0.0
                        other_contexts = []
                        with torch.no_grad():
                            for other_idx in range(3):
                                if other_idx != k:
                                    o_out = self.experts[other_idx](b_s_seq, b_s_info)
                                    other_contexts.append(o_out[7].detach())
                        if other_contexts:
                            norm_k = F.normalize(context_k, p=2, dim=1, eps=1e-4)
                            if not torch.isnan(norm_k).any():
                                for o_ctx in other_contexts:
                                    norm_o = F.normalize(o_ctx, p=2, dim=1, eps=1e-4)
                                    sim = (norm_k * norm_o).sum(dim=1).clamp(-0.999, 0.999)
                                    ortho_loss += (1 - sim.abs()).mean()
                        total_loss += 0.005 * ortho_loss

                    # 5. Sharpe Ratio Loss (선택)
                    if len(b_rewards) >= 4:
                        r_mean = b_rewards.mean()
                        r_std = b_rewards.std() + 1e-2
                        sharpe = r_mean / r_std
                        sharpe = torch.tanh(sharpe)
                        sharpe_loss = 1.0 - sharpe
                        total_loss += 0.005 * sharpe_loss

                    # 6. Reward Distribution Learning
                    if reward_quantiles is not None:
                        target_r = b_rewards.unsqueeze(1).expand(-1, N_Q)
                        reward_loss = quantile_huber_loss(reward_quantiles, target_r, tau_hat)
                        total_loss += 0.05 * reward_loss

                    return total_loss

                # ----------------------------------------------------------
                # SAM Step 1 (or AdamW)
                # ----------------------------------------------------------
                optimizer.zero_grad()
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        loss_1 = compute_loss()
                    if loss_1 is None or torch.isnan(loss_1) or torch.isinf(loss_1):
                        continue
                    self.scaler.scale(loss_1).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                    if self.use_sam:
                        optimizer.first_step(zero_grad=True)
                    else:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        continue  # SAM이 아니면 step2 없음
                else:
                    loss_1 = compute_loss()
                    if loss_1 is None or torch.isnan(loss_1) or torch.isinf(loss_1):
                        continue
                    loss_1.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                    if self.use_sam:
                        optimizer.first_step(zero_grad=True)
                    else:
                        optimizer.step()
                        continue

                # ----------------------------------------------------------
                # SAM Step 2
                # ----------------------------------------------------------
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        loss_2 = compute_loss()
                    if loss_2 is None or torch.isnan(loss_2) or torch.isinf(loss_2):
                        continue
                    self.scaler.scale(loss_2).backward()
                    # SAM Step 2에서는 unscale_() 생략
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                    optimizer.second_step(zero_grad=True)
                    self.scaler.update()
                else:
                    loss_2 = compute_loss()
                    if loss_2 is None or torch.isnan(loss_2) or torch.isinf(loss_2):
                        continue
                    loss_2.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                    optimizer.second_step(zero_grad=True)

                avg_loss += loss_1.item()

        return {'Loss': avg_loss / self.k_epochs, 'Router_Loss': router_loss_val}