"""
MetaController (Level 2) - 시장 레짐 판별 + 리스크 예산 할당
- K=5 스텝(15분) 주기로 의사결정
- 출력: direction (Long/Short/Flat) + confidence (risk_budget)
- 기존 MacroHFT 네트워크 구조 재활용
- TacticalAgent에게 goal을 내려보냄
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from collections import deque
import logging

from core import config
from common.fusion_transformer import QuantTransformerBackbone, StrategyInteractionLayer, CrossAttentionFusion

logger = logging.getLogger(__name__)


class MetaNetwork(nn.Module):
    """
    MetaController 전용 네트워크
    - Tactical Mode (RoPE + Causal Mask) 사용
    - 출력: direction logits (3) + value (1) + regime embedding (16)
    - regime embedding은 TacticalAgent에게 전달되는 latent goal
    """
    REGIME_DIM = 16  # Regime latent embedding 차원
    
    def __init__(self, input_dim, info_dim=11, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        
        # 1. Backbone (Tactical Mode - causal masking)
        self.backbone = QuantTransformerBackbone(
            state_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=num_layers,
            dropout=dropout,
            mode='tactical'
        )
        
        # 2. Strategy Fusion
        self.strategy_processor = StrategyInteractionLayer()
        self.fusion = CrossAttentionFusion(hidden_dim, query_dim=67)  # Strat(64) + Pos(3)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        
        # 3. Heads
        # Direction: Flat(0), Long(1), Short(2)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Regime Embedding Head (TacticalAgent에게 전달할 latent goal)
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, self.REGIME_DIM),
            nn.Tanh()  # bounded output
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x, info):
        """
        Args:
            x: [B, T, F] state sequence
            info: [B, 11] pos_val(1) + strategies(8) + pos_meta(2)
        Returns:
            direction_logits: [B, 3]
            value: [B, 1]
            regime_embedding: [B, REGIME_DIM]
        """
        context, seq_encodings, _ = self.backbone(x)
        if info.dim() == 3:
            info = info.squeeze(1)
        
        pos_val = info[:, 0:1]
        strategies = info[:, 1:9]
        pos_meta = info[:, 9:11]
        
        pos_info = torch.cat([pos_val, pos_meta], dim=1)  # (B, 3)
        strat_features = self.strategy_processor(strategies)  # (B, 64)
        query_vec = torch.cat([strat_features, pos_info], dim=1)  # (B, 67)
        
        fused = self.fusion(seq_encodings, query_vec)
        gate = self.gate(fused)
        final_repr = fused * gate
        
        direction_logits = self.direction_head(final_repr)
        value = self.value_head(final_repr)
        regime_embedding = self.regime_head(final_repr)
        
        return direction_logits, value, regime_embedding


class MetaController:
    """
    Level 2 Agent: 시장 레짐 판별 + 방향 + 리스크 예산
    - PPO 기반 학습
    - K 스텝마다 한 번 행동 (temporal abstraction)
    """
    
    def __init__(self, state_dim, info_dim=11, hidden_dim=256, device='cpu',
                 decision_interval=5):
        """
        Args:
            state_dim: Feature 차원 수
            info_dim: Info vector 차원 (11 for Elite 8)
            hidden_dim: 네트워크 히든 차원
            device: 'cuda' or 'cpu'
            decision_interval: K = 몇 스텝마다 의사결정 (5 = 15분)
        """
        self.device = device
        self.decision_interval = decision_interval
        self.info_dim = info_dim
        
        # Network
        self.network = MetaNetwork(
            input_dim=state_dim,
            info_dim=info_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=0.1
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.PPO_LEARNING_RATE * 0.5,  # Meta는 더 느리게 학습
            eps=1e-5
        )
        
        # PPO Hyperparameters
        self.gamma = 0.995   # Meta는 더 먼 미래를 봄
        self.lmbda = 0.97
        self.eps_clip = 0.15  # 더 보수적 클리핑
        self.k_epochs = 5
        self.entropy_coef = 0.03  # 적극적 탐험 (레짐이 다양해야 함)
        
        # [AMP] Mixed Precision Training
        if device == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
        
        # Experience buffer (한 에피소드 분량)
        self.data = []
        
        # Current decision state
        self.current_direction = 0       # 0=Flat, 1=Long, 2=Short
        self.current_confidence = 0.0    # Softmax probability of chosen action
        self.current_regime_embedding = None  # Latent goal for TacticalAgent
        self.steps_since_decision = 0
        
        # Compile if possible
        import os
        if os.name != 'nt' and hasattr(torch, 'compile') and device == 'cuda':
            try:
                self.network = torch.compile(self.network)
                logger.info("✅ MetaController 네트워크 컴파일 완료")
            except Exception as e:
                logger.warning(f"⚠️ 컴파일 실패: {e}")
    
    def should_decide(self) -> bool:
        """현재 스텝에서 의사결정을 해야 하는지 확인"""
        return self.steps_since_decision >= self.decision_interval
    
    def select_action(self, state, deterministic=False):
        """
        시장 레짐 판별 및 방향 결정
        
        Args:
            state: (obs_seq, obs_info) tuple
            deterministic: True면 greedy action
        
        Returns:
            goal: dict with 'direction', 'confidence', 'risk_budget', 'regime_embedding'
        """
        obs_seq, obs_info = state
        
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)
            obs_info = torch.as_tensor(obs_info, dtype=torch.float32, device=self.device)
        else:
            obs_seq = obs_seq.to(self.device)
            obs_info = obs_info.to(self.device)
        
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(0)
        if obs_info.dim() == 1:
            obs_info = obs_info.unsqueeze(0)
        
        with torch.no_grad():
            logits, value, regime_emb = self.network(obs_seq, obs_info)
            
            dist = Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            
            # Confidence = softmax probability of chosen action
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.gather(1, action.unsqueeze(1)).squeeze(1)
            
            log_prob = dist.log_prob(action)
        
        self.current_direction = action.item()
        self.current_confidence = confidence.item()
        self.current_regime_embedding = regime_emb.squeeze(0).detach()
        self.steps_since_decision = 0
        
        # Risk Budget: confidence를 비선형 변환 (높은 확신 → 높은 리스크 허용)
        # sigmoid(2*(conf-0.5)) → [0.27, 0.73] 범위로 매핑
        risk_budget = float(torch.sigmoid(torch.tensor(2.0 * (self.current_confidence - 0.5))).item())
        
        goal = {
            'direction': self.current_direction,           # 0=Flat, 1=Long, 2=Short
            'confidence': self.current_confidence,         # 0.0 ~ 1.0
            'risk_budget': risk_budget,                    # 0.27 ~ 0.73
            'regime_embedding': self.current_regime_embedding,  # (REGIME_DIM,) tensor
            'log_prob': log_prob.item(),
            'value': value.item(),
        }
        
        return goal
    
    def step(self):
        """매 환경 스텝마다 호출 (내부 카운터 증가)"""
        self.steps_since_decision += 1
    
    def get_goal_for_tactical(self) -> dict:
        """
        TacticalAgent에게 전달할 goal 정보
        (select_action 이후 호출)
        """
        if self.current_regime_embedding is None:
            # 아직 결정 안 됨 → 중립 goal
            return {
                'direction': 0,
                'confidence': 0.0,
                'risk_budget': 0.3,
                'regime_embedding': torch.zeros(MetaNetwork.REGIME_DIM, device=self.device),
            }
        
        return {
            'direction': self.current_direction,
            'confidence': self.current_confidence,
            'risk_budget': float(torch.sigmoid(torch.tensor(2.0 * (self.current_confidence - 0.5))).item()),
            'regime_embedding': self.current_regime_embedding,
        }
    
    def put_data(self, transition):
        """경험 저장"""
        self.data.append(transition)
    
    def train_net(self):
        """PPO 학습 (에피소드 끝에 호출)"""
        if not self.data:
            return {}
        
        batch = list(zip(*self.data))
        
        # State unpacking
        s_seq_list = [x[0].cpu().numpy() if isinstance(x[0], torch.Tensor) else x[0] 
                      for x in batch[0]]
        s_info_list = [x[1].cpu().numpy() if isinstance(x[1], torch.Tensor) else x[1]
                       for x in batch[0]]
        
        s_seq = torch.tensor(np.array(s_seq_list), dtype=torch.float32, device=self.device).squeeze(1)
        s_info = torch.tensor(np.array(s_info_list), dtype=torch.float32, device=self.device).squeeze(1)
        
        # Ensure info is 2D with correct dim
        if s_info.dim() == 1:
            s_info = s_info.unsqueeze(0)
        if s_info.shape[-1] > self.info_dim:
            s_info = s_info[..., :self.info_dim]
        
        a = torch.tensor(batch[1], dtype=torch.long, device=self.device)
        r = torch.tensor(batch[2], dtype=torch.float32, device=self.device)
        old_log_prob = torch.tensor(batch[3], dtype=torch.float32, device=self.device)
        done_mask = torch.tensor([0.0 if x else 1.0 for x in batch[4]], 
                                 dtype=torch.float32, device=self.device)
        old_val = torch.tensor([x if isinstance(x, float) else float(x) for x in batch[5]],
                               dtype=torch.float32, device=self.device)
        
        self.data = []
        
        # GAE
        with torch.no_grad():
            next_val = torch.roll(old_val, -1)
            if done_mask[-1] == 1.0:
                next_val[-1] = old_val[-1]
            else:
                next_val[-1] = 0.0
            
            deltas = r + self.gamma * next_val * done_mask - old_val
            advantage = torch.zeros_like(r)
            running_adv = 0.0
            for t in reversed(range(len(r))):
                running_adv = deltas[t] + self.gamma * self.lmbda * running_adv * done_mask[t]
                advantage[t] = running_adv
            target_val = advantage + old_val
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        # PPO Update
        avg_loss = 0.0
        for _ in range(self.k_epochs):
            # [AMP] Mixed Precision Training
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    logits, curr_val, _ = self.network(s_seq, s_info)
                    dist = Categorical(logits=logits)
                    log_prob = dist.log_prob(a)
                    
                    ratio = torch.exp(log_prob - old_log_prob)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = 0.5 * nn.MSELoss()(curr_val.squeeze(), target_val)
                    entropy_loss = -self.entropy_coef * dist.entropy().mean()
                    
                    loss = actor_loss + critic_loss + entropy_loss
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, curr_val, _ = self.network(s_seq, s_info)
                dist = Categorical(logits=logits)
                log_prob = dist.log_prob(a)
                
                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * nn.MSELoss()(curr_val.squeeze(), target_val)
                entropy_loss = -self.entropy_coef * dist.entropy().mean()
                
                loss = actor_loss + critic_loss + entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
            
            avg_loss += loss.item()
        
        return {
            'meta_loss': avg_loss / self.k_epochs,
            'meta_entropy': dist.entropy().mean().item(),
        }
    
    def reset(self):
        """에피소드 시작 시 리셋"""
        self.current_direction = 0
        self.current_confidence = 0.0
        self.current_regime_embedding = None
        self.steps_since_decision = self.decision_interval  # 첫 스텝에서 바로 결정
        self.data = []
    
    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        import os
        if not os.path.exists(path):
            logger.warning(f"MetaController 모델 없음: {path}")
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'], strict=False)
        if 'optimizer' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                pass
        logger.info(f"✅ MetaController 로드: {path}")
