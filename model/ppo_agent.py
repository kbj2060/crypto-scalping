import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import os
from . import config
from .xlstm_network import XLSTMNetwork

class Router(nn.Module):
    """
    시장 상황(State)을 보고 3명의 전문가 중 누구의 의견을 들을지 결정하는 관리자
    """
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
            nn.Softmax(dim=-1) # [w1, w2, w3] 합은 1
        )
        
    def forward(self, x):
        # x: [Batch, Seq, Feature] -> 마지막 시점의 Feature만 사용하여 판단
        # 입력이 (Batch, 1, Feature) 등으로 들어올 경우 처리
        if x.dim() == 3: 
            x = x[:, -1, :] 
        return self.net(x)

class PPOAgent:
    """
    MacroHFT 구조를 지원하는 통합 PPO 에이전트
    - Experts: 3개의 XLSTMNetwork (Trend, Volatility, Sideways)
    - Router: 1개의 MLP (가중치 분배)
    """
    def __init__(self, state_dim, action_dim=3, info_dim=15, hidden_dim=None, device='cpu'):
        self.device = device
        self.action_dim = action_dim 
        
        hidden_dim = hidden_dim if hidden_dim is not None else config.NETWORK_HIDDEN_DIM
        dropout = getattr(config, 'NETWORK_DROPOUT', 0.1)

        # 1. 전문가(Sub-Agents) 3명 생성 (Trend, Volatility, Sideways)
        self.experts = nn.ModuleList([
            XLSTMNetwork(state_dim, action_dim, info_dim, hidden_dim, config.NETWORK_NUM_LAYERS, dropout),
            XLSTMNetwork(state_dim, action_dim, info_dim, hidden_dim, config.NETWORK_NUM_LAYERS, dropout),
            XLSTMNetwork(state_dim, action_dim, info_dim, hidden_dim, config.NETWORK_NUM_LAYERS, dropout)
        ]).to(device)
        # [에러 해결 핵심] expert_names 속성 정의
        self.expert_names = ['trend', 'volatility', 'sideways']

        # 2. 관리자(Router) 생성
        self.router = Router(state_dim, num_experts=3).to(device)

        # 3. 옵티마이저 분리
        self.lr = config.PPO_LEARNING_RATE
        
        # 전문가용 옵티마이저 3개
        self.opt_experts = [
            optim.Adam(exp.parameters(), lr=self.lr, eps=1e-5) 
            for exp in self.experts
        ]
        # 라우터용 옵티마이저 1개
        self.opt_router = optim.Adam(self.router.parameters(), lr=self.lr, eps=1e-5)
        
        # PPO 파라미터
        self.gamma = config.PPO_GAMMA
        self.lmbda = config.PPO_LAMBDA
        self.eps_clip = config.PPO_EPS_CLIP
        self.k_epochs = config.PPO_K_EPOCHS
        self.entropy_coef = config.PPO_ENTROPY_COEF
        
        self.data = []
        self.current_states = [None] * 3 # 전문가별 LSTM 은닉 상태 관리

    def reset_episode_states(self):
        """에피소드 시작 시 LSTM 상태 초기화"""
        self.current_states = [None] * 3

    def save_model(self, path):
        """전문가 3명 + 라우터 1명 모두 저장"""
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
            
            # MacroHFT 구조인지 확인
            if 'experts' in checkpoint:
                for i, state in enumerate(checkpoint['experts']):
                    self.experts[i].load_state_dict(state, strict=False)
                if 'router' in checkpoint:
                    self.router.load_state_dict(checkpoint['router'], strict=False)
                print(f"✅ MacroHFT 로드 완료: {path}")
            
            # 구버전 호환 (Trend 전문가에게만 로드)
            elif 'model_state_dict' in checkpoint:
                print(f"⚠️ 구버전(단일) 모델 감지. Trend Expert에만 로드합니다.")
                self.experts[0].load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                print(f"❌ 알 수 없는 모델 형식: {path}")
        except Exception as e:
            print(f"❌ 모델 로드 중 에러 발생: {e}")

    def select_action(self, state, action_mask=None, mode='router', expert_idx=0):
        obs_seq, obs_info = state
        
        # 텐서 변환 및 Device 이동
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.FloatTensor(obs_seq).to(self.device)
            obs_info = torch.FloatTensor(obs_info).unsqueeze(0).to(self.device)
        else:
            obs_seq = obs_seq.to(self.device)
            obs_info = obs_info.to(self.device)

        with torch.no_grad():
            if mode == 'expert':
                # [Phase 1] 특정 전문가 독단적 수행
                net = self.experts[expert_idx]
                # forward returns: logits, val_mean, val_cvar, aux_val, next_states, gate_mean
                logits, value, _, _, next_state, _ = net(
                    obs_seq, obs_info, states=self.current_states[expert_idx]
                )
                self.current_states[expert_idx] = next_state
                weights = None
                
            else: # mode == 'router'
                # [Phase 2] 라우터 기반 앙상블
                logits_list = []
                for i, net in enumerate(self.experts):
                    l, _, _, _, ns, _ = net(obs_seq, obs_info, states=self.current_states[i])
                    logits_list.append(l)
                    self.current_states[i] = ns
                
                # 라우터 비중 결정
                weights = self.router(obs_seq) # [1, 3]
                
                # 가중 합
                stacked_logits = torch.stack(logits_list, dim=1) # [1, 3, Action]
                weighted_logits = torch.sum(weights.unsqueeze(-1) * stacked_logits, dim=1)
                logits = weighted_logits
                value = 0.0 # 단순화

            # 액션 마스킹
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                logits = logits + (mask_tensor - 1) * 1e10

            dist = Categorical(logits=logits)
            action = dist.sample()
            
            # (Router 모드일 때만 weights 기록)
            if weights is None: weights_val = 0.0
            else: weights_val = weights.cpu().numpy()

        return action.item(), dist.log_prob(action).item(), value

    def put_data(self, transition):
        self.data.append(transition)

    def train_net(self, episode=1, mode='router', expert_idx=0):
        if not self.data: return {}

        # [에러 해결 핵심] GRU 차원 문제(ValueError) 해결을 위한 squeeze(1) 적용
        # (Batch, 1, Seq, Feat) -> (Batch, Seq, Feat)
        s_seq_np = np.array([x[0][0].cpu().numpy() if isinstance(x[0][0], torch.Tensor) else x[0][0] for x in self.data])
        s_seq = torch.tensor(s_seq_np, dtype=torch.float).squeeze(1).to(self.device)
        
        s_info_np = np.array([x[0][1].cpu().numpy() if isinstance(x[0][1], torch.Tensor) else x[0][1] for x in self.data])
        s_info = torch.tensor(s_info_np, dtype=torch.float).squeeze(1).to(self.device)

        a = torch.tensor([x[1] for x in self.data], dtype=torch.long).to(self.device)
        r = torch.tensor([x[2] for x in self.data], dtype=torch.float).to(self.device)
        prob_a = torch.tensor([x[4] for x in self.data], dtype=torch.float).to(self.device)
        masks = torch.tensor(np.array([x[8] for x in self.data]), dtype=torch.float).to(self.device)
        
        self.data = [] # 버퍼 비우기

        # 모드에 따라 업데이트할 네트워크 선택
        if mode == 'expert':
            optimizer = self.opt_experts[expert_idx]
            network = self.experts[expert_idx]
        else:
            optimizer = self.opt_router
        
        avg_loss = 0.0
        
        # PPO Update Loop
        for _ in range(self.k_epochs):
            if mode == 'expert':
                logits, _, _, _, _, _ = network(s_seq, s_info)
            else:
                # Router 모드
                l_list = []
                with torch.no_grad():
                    for exp in self.experts:
                        l, _, _, _, _, _ = exp(s_seq, s_info)
                        l_list.append(l)
                weights = self.router(s_seq)
                stacked = torch.stack(l_list, dim=1)
                logits = torch.sum(weights.unsqueeze(-1) * stacked, dim=1)

            # Masking
            logits = logits + (masks - 1) * 1e10
            
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(a)
            ratio = torch.exp(log_prob - prob_a)
            
            # Advantage (Reward Normalization)
            advantage = (r - r.mean()) / (r.std() + 1e-8)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            
            loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * dist.entropy().mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.router.parameters() if mode == 'router' else network.parameters(), 
                0.5
            )
            optimizer.step()
            
            avg_loss += loss.item()

        return {'Loss/Total': avg_loss / self.k_epochs}