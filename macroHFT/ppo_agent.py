import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import random
import os
import glob
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
        
        # [수정] Expert별 Gamma 설정 (Multi-Horizon)
        default_gammas = {0: 0.995, 1: 0.99, 2: 0.90}
        self.gammas = getattr(config, 'EXPERT_GAMMAS', default_gammas)
        
        self.gamma = config.PPO_GAMMA  # Router용 기본 Gamma
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
        # [Fix] 트랜스포머 상태 저장을 위한 변수 초기화
        self.current_states = [None] * 3
        
        if device == 'cuda':
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None

        # Compiler
        if os.name != 'nt' and hasattr(torch, 'compile'):
             try:
                 for i in range(3): self.experts[i] = torch.compile(self.experts[i])
                 self.router = torch.compile(self.router)
             except: pass

    # [Fix] 누락되었던 메서드 추가!
    def reset_episode_states(self):
        """에피소드 시작 시 Transformer의 Hidden State 초기화"""
        self.current_states = [None] * 3

    def save_model(self, path):
        torch.save({
            'experts': [exp.state_dict() for exp in self.experts],
            'router': self.router.state_dict(),
            'router_target': self.router_target.state_dict(),
            'opt_experts': [opt.state_dict() for opt in self.opt_experts],
            'opt_router': self.opt_router.state_dict(),
            'epsilon': self.epsilon,
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
                if 'router_target' in checkpoint:
                    self.router_target.load_state_dict(checkpoint['router_target'], strict=False)
                if 'epsilon' in checkpoint:
                    self.epsilon = checkpoint['epsilon']
        except Exception as e:
            print(f"❌ Load Error: {e}")

    # [New] 드림팀 합체 로딩 기능
    def load_dream_team(self, base_dir):
        """
        각 분야 최고(Best) 모델 파일들에서 해당 파트만 추출하여 로드합니다.
        - Router & Epsilon <- best_router.pth
        - Trend Expert <- best_trend.pth
        - Volatility Expert <- best_volatility.pth
        - Sideways Expert <- best_sideways.pth
        """
        print(f"🧬 Assembling Dream Team from: {base_dir}")
        
        # 파일 매핑
        files = {
            'router': 'best_router.pth',
            0: 'best_trend.pth',       # Trend Expert Index
            1: 'best_volatility.pth',  # Volatility Expert Index
            2: 'best_sideways.pth'     # Sideways Expert Index
        }
        
        # 1. Router & Global State 로드
        router_path = self._find_file(base_dir, files['router'])
        if router_path:
            try:
                ckpt = torch.load(router_path, map_location=self.device)
                
                # 라우터 네트워크 로드
                if 'router' in ckpt:
                    self.router.load_state_dict(ckpt['router'])
                    self.router_target.load_state_dict(ckpt.get('router_target', ckpt['router']))
                
                # 라우터 옵티마이저 로드 (학습 이어하기 위해 필수)
                if 'opt_router' in ckpt:
                    self.opt_router.load_state_dict(ckpt['opt_router'])
                    
                # 엡실론(탐험율) 로드
                if 'epsilon' in ckpt:
                    self.epsilon = ckpt['epsilon']
                    
                print(f"   ✅ Router System loaded from {os.path.basename(router_path)}")
            except Exception as e:
                print(f"   ❌ Router Load Failed: {e}")
        else:
            print("   ⚠️ Best Router file not found. Keeping initialization.")

        # 2. Experts 로드 (각각의 파일에서 해당 인덱스 Expert만 추출)
        expert_names = ['Trend', 'Volatility', 'Sideways']
        
        for idx in range(3):
            fname = files[idx]
            fpath = self._find_file(base_dir, fname)
            
            # 파일이 없으면 Router 파일에서라도 가져오기 (Fallback)
            if not fpath and router_path:
                fpath = router_path
                fallback = True
            else:
                fallback = False
            
            if fpath:
                try:
                    ckpt = torch.load(fpath, map_location=self.device)
                    
                    # Expert 가중치
                    if 'experts' in ckpt and len(ckpt['experts']) > idx:
                        self.experts[idx].load_state_dict(ckpt['experts'][idx])
                    
                    # Expert 옵티마이저 (중요: 모멘텀 유지를 위해)
                    if 'opt_experts' in ckpt and len(ckpt['opt_experts']) > idx:
                        self.opt_experts[idx].load_state_dict(ckpt['opt_experts'][idx])
                        
                    source = "Router Fallback" if fallback else os.path.basename(fpath)
                    print(f"   ✅ {expert_names[idx]} Expert & Optimizer loaded from {source}")
                    
                except Exception as e:
                    print(f"   ❌ {expert_names[idx]} Load Failed: {e}")

    def _find_file(self, directory, suffix):
        """파일 찾기 헬퍼 (정확한 이름 -> 패턴 매칭 순)"""
        exact = os.path.join(directory, suffix)
        if os.path.exists(exact): return exact
        
        candidates = glob.glob(os.path.join(directory, f"*{suffix}"))
        if candidates: return max(candidates, key=os.path.getctime)
        return None

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
        
        # Selected Expert Index
        router_choices = torch.tensor(batch_data[9], dtype=torch.long, device=self.device)

        self.data = []

        # --- Train Router (DDQN) ---
        router_loss_val = 0.0
        if mode == 'router':
            with torch.no_grad():
                # [Fix] 1-step Reward가 아닌 Discounted Return 사용 (장기적 관점)
                returns = torch.zeros_like(r)
                running_return = 0.0
                for t in reversed(range(len(r))):
                    running_return = r[t] + self.gamma * running_return * done_mask[t]
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
                self.router_target.load_state_dict(self.router.state_dict())
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # --- Train Experts (PPO) ---
        avg_loss = 0.0

        for _ in range(self.k_epochs):
            # [Improvement] Router 모드여도 사용된 Expert들은 학습시킴
            # 현재 배치에서 Expert 0, 1, 2가 각각 어디서 쓰였는지 마스킹
            target_experts = [expert_idx] if mode == 'expert' else [0, 1, 2]
            
            for k in target_experts:
                mask = (router_choices == k)
                if mask.sum() == 0: continue
                
                # [수정] Expert별 Gamma 사용
                expert_gamma = self.gammas[k]
                
                # Advantage 계산 (해당 Expert의 Gamma 사용)
                with torch.no_grad():
                    next_val = torch.roll(val, -1)
                    next_val[-1] = 0.0 # Last state has no next value
                    deltas = r + expert_gamma * next_val * done_mask - val
                    advantage = torch.zeros_like(r).to(self.device)
                    running_adv = 0.0
                    for t in reversed(range(len(r))):
                        running_adv = deltas[t] + expert_gamma * config.PPO_LAMBDA * running_adv * done_mask[t]
                        advantage[t] = running_adv
                    target_val = advantage + val
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                
                # Expert별 Sub-batch
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
                    with torch.amp.autocast(self.device):
                        logits, curr_val, _, _, _, _ = network(b_s_seq, b_s_info)
                        logits = logits + (b_masks - 1) * 1e9
                        dist = Categorical(logits=logits)
                        
                        log_prob = dist.log_prob(b_a)
                        ratio = torch.exp(log_prob - b_prob_a)
                        surr1 = ratio * b_adv
                        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = 0.5 * nn.MSELoss()(curr_val.squeeze(), b_target_val)
                        entropy_loss = -self.entropy_coef * dist.entropy().mean()
                        
                        total_loss = actor_loss + critic_loss + entropy_loss
                    
                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # CPU Fallback
                    logits, curr_val, _, _, _, _ = network(b_s_seq, b_s_info)
                    logits = logits + (b_masks - 1) * 1e9
                    dist = Categorical(logits=logits)
                    log_prob = dist.log_prob(b_a)
                    ratio = torch.exp(log_prob - b_prob_a)
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = 0.5 * nn.MSELoss()(curr_val.squeeze(), b_target_val)
                    entropy_loss = -self.entropy_coef * dist.entropy().mean()
                    total_loss = actor_loss + critic_loss + entropy_loss
                    total_loss.backward()
                    optimizer.step()

                avg_loss += total_loss.item()

        return {'Loss': avg_loss / self.k_epochs, 'Router_Loss': router_loss_val}