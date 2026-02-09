"""
TacticalAgent (Level 1) - Goal-Conditioned TD3
- MetaController의 goal(방향, 리스크 예산, regime embedding)을 받아 실행
- Kelly Criterion 기반 포지션 사이징
- Intrinsic Reward: goal 방향 일치도 보너스
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import logging

from common import config
from common.fusion_transformer import QuantTransformerBackbone, StrategyInteractionLayer, CrossAttentionFusion
from .meta_controller import MetaNetwork

logger = logging.getLogger(__name__)


# ==============================================================================
# Goal-Conditioned Actor & Critic Networks
# ==============================================================================

class GoalConditionedActor(nn.Module):
    """
    TD3 Actor + Goal Conditioning
    - 기존 StrategicActor에 MetaController의 regime_embedding을 추가
    - info_dim = 12 (기존) + REGIME_DIM (16) + direction(1) + risk_budget(1) = 30
    """
    GOAL_DIM = MetaNetwork.REGIME_DIM + 2  # regime_emb(16) + direction(1) + risk_budget(1) = 18
    
    def __init__(self, input_dim, action_dim=1, base_info_dim=12, hidden_dim=256, 
                 num_layers=3, dropout=0.1):
        super().__init__()
        
        self.base_info_dim = base_info_dim
        self.goal_dim = self.GOAL_DIM
        total_info_dim = base_info_dim + self.goal_dim  # 12 + 18 = 30
        
        seq_len = getattr(config, 'LOOKBACK', 60)
        
        # 1. Backbone (Strategic Mode)
        self.backbone = QuantTransformerBackbone(
            state_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=num_layers,
            seq_len=seq_len,
            dropout=dropout,
            mode='strategic'
        )
        
        # 2. Strategy + Goal Fusion
        self.strategy_processor = StrategyInteractionLayer()
        
        # Goal Encoder: regime_embedding + scalars → compressed goal
        self.goal_encoder = nn.Sequential(
            nn.Linear(self.goal_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
        )
        
        # query_dim = Strat(64) + Pos(3) + Vol(1) + GoalEncoded(32) = 100
        self.fusion = CrossAttentionFusion(hidden_dim, query_dim=100)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        
        # 3. Actor Head (Continuous Action)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh()
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
    
    def forward(self, x, info, goal):
        """
        Args:
            x: [B, T, F] state sequence
            info: [B, 12] base info (pos_val + strategies + pos_meta + volatility)
            goal: [B, GOAL_DIM] from MetaController
        Returns:
            action: [B, 1]
            states: None
            risk_gate: scalar
        """
        context, seq_encodings, _ = self.backbone(x)
        
        if info.dim() == 3: info = info.squeeze(1)
        if goal.dim() == 3: goal = goal.squeeze(1)
        
        # Parse info (기존 TD3와 동일)
        pos_val = info[:, 0:1]
        strategies = info[:, 1:9]
        pos_meta = info[:, 9:11]
        volatility = info[:, 11:12]
        
        pos_info = torch.cat([pos_val, pos_meta, volatility], dim=1)  # (B, 4)
        strat_features = self.strategy_processor(strategies)           # (B, 64)
        
        # Goal encoding
        goal_encoded = self.goal_encoder(goal)  # (B, 32)
        
        # Combined query: strat(64) + pos(4) + goal(32) = 100
        query_vec = torch.cat([strat_features, pos_info, goal_encoded], dim=1)
        
        fused = self.fusion(seq_encodings, query_vec)
        gate = self.gate(fused)
        action = self.actor(fused * gate)
        
        return action, None, gate.mean().item()


class GoalConditionedCritic(nn.Module):
    """TD3 Twin Critic + Goal Conditioning"""
    
    GOAL_DIM = MetaNetwork.REGIME_DIM + 2
    
    def __init__(self, input_dim, action_dim=1, base_info_dim=12, hidden_dim=256,
                 num_layers=3, dropout=0.1):
        super().__init__()
        
        seq_len = getattr(config, 'LOOKBACK', 60)
        
        self.backbone = QuantTransformerBackbone(
            state_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=num_layers,
            seq_len=seq_len,
            dropout=dropout,
            mode='strategic'
        )
        self.strategy_processor = StrategyInteractionLayer()
        self.goal_encoder = nn.Sequential(
            nn.Linear(self.GOAL_DIM, 64), nn.GELU(),
            nn.Linear(64, 32), nn.LayerNorm(32),
        )
        self.fusion = CrossAttentionFusion(hidden_dim, query_dim=100)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        
        # Twin Q-Networks (State + Action)
        self.q1_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1)
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
    
    def forward(self, x, info, goal, action):
        context, seq_encodings, _ = self.backbone(x)
        if info.dim() == 3: info = info.squeeze(1)
        if goal.dim() == 3: goal = goal.squeeze(1)
        
        pos_val = info[:, 0:1]
        strategies = info[:, 1:9]
        pos_meta = info[:, 9:11]
        volatility = info[:, 11:12]
        
        pos_info = torch.cat([pos_val, pos_meta, volatility], dim=1)
        strat_features = self.strategy_processor(strategies)
        goal_encoded = self.goal_encoder(goal)
        query_vec = torch.cat([strat_features, pos_info, goal_encoded], dim=1)
        
        fused = self.fusion(seq_encodings, query_vec)
        gate = self.gate(fused)
        state_repr = fused * gate
        
        q_input = torch.cat([state_repr, action], dim=1)
        return self.q1_net(q_input), self.q2_net(q_input)


# ==============================================================================
# Goal-Conditioned Replay Buffer
# ==============================================================================

class GoalConditionedReplayBuffer:
    """Goal을 포함하는 Replay Buffer"""
    
    def __init__(self, state_dim, info_dim, goal_dim, action_dim, max_size=100000, device='cpu'):
        self.device = device
        self.ptr = 0
        self.size = 0
        self.max_size = max_size
        
        lookback = getattr(config, 'LOOKBACK', 60)
        
        self.state_seq = np.zeros((max_size, lookback, state_dim), dtype=np.float32)
        self.state_info = np.zeros((max_size, info_dim), dtype=np.float32)
        self.goal = np.zeros((max_size, goal_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        
        self.next_state_seq = np.zeros((max_size, lookback, state_dim), dtype=np.float32)
        self.next_state_info = np.zeros((max_size, info_dim), dtype=np.float32)
        self.next_goal = np.zeros((max_size, goal_dim), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)
    
    def add(self, state, goal, action, reward, next_state, next_goal, done):
        idx = self.ptr
        
        seq, info = state
        seq_np = seq.cpu().numpy() if isinstance(seq, torch.Tensor) else np.asarray(seq, dtype=np.float32)
        if seq_np.ndim == 3: seq_np = seq_np.squeeze(0)
        self.state_seq[idx] = seq_np
        
        info_np = info.cpu().numpy() if isinstance(info, torch.Tensor) else np.asarray(info, dtype=np.float32)
        if info_np.ndim == 2: info_np = info_np.squeeze(0)
        self.state_info[idx] = info_np.flatten()[:self.state_info.shape[1]]
        
        goal_np = goal.cpu().numpy() if isinstance(goal, torch.Tensor) else np.asarray(goal, dtype=np.float32)
        self.goal[idx] = goal_np.flatten()[:self.goal.shape[1]]
        
        self.action[idx] = np.atleast_1d(np.asarray(action, dtype=np.float32)).reshape(-1)[:self.action.shape[1]]
        self.reward[idx] = reward
        
        nseq, ninfo = next_state
        nseq_np = nseq.cpu().numpy() if isinstance(nseq, torch.Tensor) else np.asarray(nseq, dtype=np.float32)
        if nseq_np.ndim == 3: nseq_np = nseq_np.squeeze(0)
        self.next_state_seq[idx] = nseq_np
        
        ninfo_np = ninfo.cpu().numpy() if isinstance(ninfo, torch.Tensor) else np.asarray(ninfo, dtype=np.float32)
        if ninfo_np.ndim == 2: ninfo_np = ninfo_np.squeeze(0)
        self.next_state_info[idx] = ninfo_np.flatten()[:self.next_state_info.shape[1]]
        
        ng_np = next_goal.cpu().numpy() if isinstance(next_goal, torch.Tensor) else np.asarray(next_goal, dtype=np.float32)
        self.next_goal[idx] = ng_np.flatten()[:self.next_goal.shape[1]]
        
        self.not_done[idx] = 1.0 - float(done)
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state_seq[ind]).to(self.device),
            torch.FloatTensor(self.state_info[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state_seq[ind]).to(self.device),
            torch.FloatTensor(self.next_state_info[ind]).to(self.device),
            torch.FloatTensor(self.next_goal[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )


# ==============================================================================
# Goal-Conditioned TD3 Agent
# ==============================================================================

class GoalConditionedTD3Agent:
    """
    Level 1 Agent: Goal-Conditioned TD3
    - MetaController의 goal을 조건으로 받아 실행
    - Kelly Criterion으로 포지션 사이징 제한
    """
    
    def __init__(self, state_dim, action_dim=1, info_dim=12, device='cuda'):
        self.device = device
        self.info_dim = info_dim
        self.goal_dim = GoalConditionedActor.GOAL_DIM  # 18
        
        # TD3 Hyperparameters
        self.gamma = config.TD3_GAMMA
        self.tau = config.TD3_TAU
        self.policy_noise = config.TD3_POLICY_NOISE
        self.noise_clip = config.TD3_NOISE_CLIP
        self.policy_freq = config.TD3_POLICY_FREQ
        self.total_it = 0
        
        # [AMP] Mixed Precision Training
        if device == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
        
        # Networks
        self.actor = GoalConditionedActor(state_dim, action_dim, info_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.TD3_LEARNING_RATE)
        
        self.critic = GoalConditionedCritic(state_dim, action_dim, info_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.TD3_LEARNING_RATE)
        
        # Goal-Conditioned Replay Buffer
        self.replay_buffer = GoalConditionedReplayBuffer(
            state_dim, info_dim, self.goal_dim, action_dim,
            max_size=config.TD3_BUFFER_SIZE, device=device
        )
    
    def _goal_to_tensor(self, goal_dict):
        """
        goal dict → flat tensor [direction_onehot(1), risk_budget(1), regime_embedding(16)]
        """
        direction = goal_dict.get('direction', 0)
        risk_budget = goal_dict.get('risk_budget', 0.3)
        regime_emb = goal_dict.get('regime_embedding', None)
        
        # direction을 signed value로 변환: 0=0.0, 1=1.0(Long), 2=-1.0(Short)
        dir_val = 0.0 if direction == 0 else (1.0 if direction == 1 else -1.0)
        
        scalars = torch.tensor([dir_val, risk_budget], dtype=torch.float32, device=self.device)
        
        if regime_emb is None:
            regime_emb = torch.zeros(MetaNetwork.REGIME_DIM, device=self.device)
        elif not isinstance(regime_emb, torch.Tensor):
            regime_emb = torch.tensor(regime_emb, dtype=torch.float32, device=self.device)
        
        regime_emb = regime_emb.to(self.device)
        
        return torch.cat([scalars, regime_emb])  # (18,)
    
    def select_action(self, state, goal_dict, noise=0.1):
        """
        Goal-conditioned action selection
        
        Args:
            state: (obs_seq, obs_info) tuple
            goal_dict: MetaController's goal
            noise: exploration noise std
        
        Returns:
            action: np.array, shape (1,)
            gate_mean: float
        """
        s0, s1 = state
        state_seq = s0.cpu().numpy() if isinstance(s0, torch.Tensor) else np.asarray(s0)
        if state_seq.ndim == 3: state_seq = state_seq.squeeze(0)
        state_info = s1.cpu().numpy() if isinstance(s1, torch.Tensor) else np.asarray(s1)
        if state_info.ndim == 2: state_info = state_info.squeeze(0)
        
        obs_seq = torch.FloatTensor(state_seq).to(self.device).unsqueeze(0)
        obs_info = torch.FloatTensor(state_info).to(self.device).unsqueeze(0)
        goal_tensor = self._goal_to_tensor(goal_dict).unsqueeze(0)
        
        self.actor.eval()
        with torch.no_grad():
            action, _, gate_mean = self.actor(obs_seq, obs_info, goal_tensor)
        self.actor.train()
        
        action = action.cpu().numpy().flatten()
        
        if noise > 0:
            # Goal 방향에 맞는 biased exploration
            direction = goal_dict.get('direction', 0)
            bias = 0.0
            if direction == 1:  # Long bias
                bias = 0.05  # 살짝 양수 방향으로 편향
            elif direction == 2:  # Short bias
                bias = -0.05
            
            action = action + np.random.normal(bias, noise, size=action.shape)
        
        return np.clip(action, -1, 1), gate_mean
    
    def train(self, batch_size=256):
        if self.replay_buffer.size < batch_size:
            return None
        
        self.total_it += 1
        
        (s_seq, s_info, goal, action, 
         ns_seq, ns_info, next_goal, reward, not_done) = self.replay_buffer.sample(batch_size)
        
        # Target Q (no autocast needed for inference)
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action, _, _ = self.actor_target(ns_seq, ns_info, next_goal)
            next_action = (next_action + noise).clamp(-1, 1)
            
            target_Q1, target_Q2 = self.critic_target(ns_seq, ns_info, next_goal, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done * self.gamma * target_Q)
        
        # [AMP] Critic Update with Mixed Precision
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                current_Q1, current_Q2 = self.critic(s_seq, s_info, goal, action)
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.scaler.step(self.critic_optimizer)
            self.scaler.update()
        else:
            current_Q1, current_Q2 = self.critic(s_seq, s_info, goal, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()
        
        # [AMP] Actor Update with Mixed Precision
        actor_loss_val = 0.0
        if self.total_it % self.policy_freq == 0:
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    pi, _, _ = self.actor(s_seq, s_info, goal)
                    q1, _ = self.critic(s_seq, s_info, goal, pi)
                    
                    # Goal alignment bonus: action 방향이 goal direction과 일치하면 보너스
                    goal_direction = goal[:, 0:1]  # -1, 0, 1
                    alignment = (pi * goal_direction).mean()  # 양수면 aligned
                    
                    actor_loss = -q1.mean() - 0.1 * alignment  # Q 최대화 + goal 정렬
                actor_loss_val = actor_loss.item()
                
                self.actor_optimizer.zero_grad()
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(self.actor_optimizer)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
            else:
                pi, _, _ = self.actor(s_seq, s_info, goal)
                q1, _ = self.critic(s_seq, s_info, goal, pi)
                
                goal_direction = goal[:, 0:1]
                alignment = (pi * goal_direction).mean()
                
                actor_loss = -q1.mean() - 0.1 * alignment
                actor_loss_val = actor_loss.item()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()
            
            # Soft update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        metrics = {
            'tactical_critic_loss': critic_loss.item(),
            'tactical_actor_loss': actor_loss_val,
            'tactical_q1_mean': current_Q1.mean().item(),
            'tactical_target_q_mean': target_Q.mean().item(),
        }
        
        if self.total_it % 1000 == 0:
            logger.info(
                "[Tactical] Step %d | Q1: %.3f | CriticLoss: %.3f",
                self.total_it, metrics['tactical_q1_mean'], metrics['tactical_critic_loss']
            )
        
        return metrics
    
    def save(self, path):
        base = path.replace('.pth', '')
        torch.save(self.actor.state_dict(), base + '_tactical_actor.pth')
        torch.save(self.critic.state_dict(), base + '_tactical_critic.pth')
    
    def load(self, path):
        base = path.replace('.pth', '')
        actor_path = base + '_tactical_actor.pth'
        critic_path = base + '_tactical_critic.pth'
        import os
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device), strict=False)
            self.actor_target = copy.deepcopy(self.actor)
            logger.info(f"✅ TacticalAgent Actor 로드: {actor_path}")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device), strict=False)
            self.critic_target = copy.deepcopy(self.critic)
            logger.info(f"✅ TacticalAgent Critic 로드: {critic_path}")
