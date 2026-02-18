"""
TD3 Agent with Conservative Q-Learning (CQL) & Dimension Fix
- [Added] CQL Loss to prevent Q-value overestimation for OOD actions.
"""
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import logging

from core import config
from .td3_network import PositionAwareActor, TD3Critic

logger = logging.getLogger(__name__)


class ReplayBuffer:
    def __init__(self, state_dim, info_dim, action_dim, max_size=100000, device='cpu'):
        self.device = device
        self.ptr = 0
        self.size = 0
        self.max_size = max_size
        # [수정] Elite 8 + Volatility = 12
        info_dim = 12
        lookback = getattr(config, 'LOOKBACK', 60)

        self.state_seq = np.zeros((max_size, lookback, state_dim), dtype=np.float32)
        self.state_info = np.zeros((max_size, info_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)

        self.next_state_seq = np.zeros((max_size, lookback, state_dim), dtype=np.float32)
        self.next_state_info = np.zeros((max_size, info_dim), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)
        
        # [Teacher-Guided] Oracle Action 저장
        self.oracle_action = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done, oracle_action=None):
        idx = self.ptr

        seq = state[0]
        info = state[1]
        seq_np = seq.cpu().numpy() if isinstance(seq, torch.Tensor) else np.asarray(seq, dtype=np.float32)
        if seq_np.ndim == 3:
            seq_np = seq_np.squeeze(0)
        self.state_seq[idx] = seq_np
        info_np = info.cpu().numpy() if isinstance(info, torch.Tensor) else np.asarray(info, dtype=np.float32)
        if info_np.ndim == 2:
            info_np = info_np.squeeze(0)
        self.state_info[idx] = info_np.flatten()[: self.state_info.shape[1]]

        action = np.atleast_1d(np.asarray(action, dtype=np.float32))
        self.action[idx] = action.reshape(-1)[: self.action.shape[1]]

        self.reward[idx] = reward

        nseq = next_state[0]
        ninfo = next_state[1]
        nseq_np = nseq.cpu().numpy() if isinstance(nseq, torch.Tensor) else np.asarray(nseq, dtype=np.float32)
        if nseq_np.ndim == 3:
            nseq_np = nseq_np.squeeze(0)
        self.next_state_seq[idx] = nseq_np
        ninfo_np = ninfo.cpu().numpy() if isinstance(ninfo, torch.Tensor) else np.asarray(ninfo, dtype=np.float32)
        if ninfo_np.ndim == 2:
            ninfo_np = ninfo_np.squeeze(0)
        self.next_state_info[idx] = ninfo_np.flatten()[: self.next_state_info.shape[1]]

        self.not_done[idx] = 1.0 - float(done)
        
        # [Teacher-Guided] Oracle Action 저장
        if oracle_action is not None:
            self.oracle_action[idx] = float(oracle_action)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state_seq[ind]).to(self.device),
            torch.FloatTensor(self.state_info[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state_seq[ind]).to(self.device),
            torch.FloatTensor(self.next_state_info[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            # [Teacher-Guided] Oracle Action 반환
            torch.FloatTensor(self.oracle_action[ind]).to(self.device),
        )


class TD3Agent:
    def __init__(self, state_dim, action_dim=1, info_dim=12, device='cuda'):
        self.device = device

        self.gamma = config.TD3_GAMMA
        self.tau = config.TD3_TAU
        self.policy_noise = config.TD3_POLICY_NOISE
        self.noise_clip = config.TD3_NOISE_CLIP
        self.policy_freq = config.TD3_POLICY_FREQ
        self.total_it = 0

        self.actor = PositionAwareActor(state_dim, action_dim, info_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.TD3_LEARNING_RATE)

        self.critic = TD3Critic(state_dim, action_dim, info_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.TD3_LEARNING_RATE)

        self.replay_buffer = ReplayBuffer(
            state_dim, info_dim, action_dim,
            max_size=config.TD3_BUFFER_SIZE, device=device
        )

        self.position_cooldown = 0
        self.min_hold_steps = 5
        self.last_position = None
        self.uncertainty_history = torch.zeros(1000, device=self.device)
        self.uh_ptr = 0
        self.uh_full = False

        # [Added] CQL Hyperparameters
        self.cql_alpha = 0.5  # CQL Loss 가중치
        self.num_random = 4   # CQL 샘플링 개수

    def select_action(self, state, noise=0.0, current_position=None):
        """Returns (action, gate_mean, risk) for TensorBoard tracking."""
        s0 = state[0]
        s1 = state[1]
        state_seq = s0.cpu().numpy() if isinstance(s0, torch.Tensor) else np.asarray(s0)
        if state_seq.ndim == 3:
            state_seq = state_seq.squeeze(0)
        state_info = s1.cpu().numpy() if isinstance(s1, torch.Tensor) else np.asarray(s1)
        if state_info.ndim == 2:
            state_info = state_info.squeeze(0)

        obs_seq = torch.FloatTensor(state_seq).to(self.device).unsqueeze(0)
        obs_info = torch.FloatTensor(state_info).to(self.device).unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            action, _, gate_mean = self.actor(obs_seq, obs_info)
        self.actor.train()

        action = action.cpu().numpy().flatten()

        risk_val = 0.0
        if noise > 0:
            risk_val = self._estimate_uncertainty(obs_seq, obs_info, action)
            adaptive_noise = noise * (1.0 - risk_val)
            action = action + np.random.normal(0, adaptive_noise, size=action.shape)

        return np.clip(action, -1, 1), gate_mean, risk_val

    def _estimate_uncertainty(self, obs_seq, obs_info, action_np):
        """분위수(Quantile) 기반 정규화 불확실성 추정"""
        with torch.no_grad():
            action_t = torch.FloatTensor(action_np).to(self.device).unsqueeze(0)
            q1, q2 = self.critic(obs_seq, obs_info, action_t)
            abs_diff = torch.abs(q1 - q2)

            self.uncertainty_history[self.uh_ptr] = abs_diff.item()
            self.uh_ptr = (self.uh_ptr + 1) % 1000
            if self.uh_ptr == 0:
                self.uh_full = True

            valid_len = 1000 if self.uh_full else self.uh_ptr + 1
            history = self.uncertainty_history[:valid_len]
            q10 = torch.quantile(history, 0.1)
            q90 = torch.quantile(history, 0.9)
            normalized = (abs_diff - q10) / (q90 - q10 + 1e-6)
            return torch.clamp(normalized, 0.05, 0.95).item()

    def train(self, batch_size=256, teacher_lambda=0.0):
        if self.replay_buffer.size < batch_size:
            return None

        self.total_it += 1
        s_seq, s_info, action, ns_seq, ns_info, reward, not_done, oracle_action = self.replay_buffer.sample(batch_size)

        # -----------------------------------------------------------------
        # 1. Target Q 계산 (TD3 기존 로직)
        # -----------------------------------------------------------------
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action, _, _ = self.actor_target(ns_seq, ns_info)
            next_action = (next_action + noise).clamp(-1, 1)

            target_Q1, target_Q2 = self.critic_target(ns_seq, ns_info, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done * self.gamma * target_Q)

        current_Q1, current_Q2 = self.critic(s_seq, s_info, action)
        
        # 기본 MSE Loss
        critic_loss_mse = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # -----------------------------------------------------------------
        # 2. CQL (Conservative Q-Learning) Loss 추가
        # [긴급 처방 1] CQL 비활성화 (야수 모드 ON)
        # - 오프라인 데이터가 아닌 온라인 학습에서는 CQL이 과도한 보수성 유발
        # - Q=0.06에서 정체된 원인: CQL이 Q값을 억제
        # - 해결: cql_alpha = 0으로 설정하여 일반 TD3로 전환
        # -----------------------------------------------------------------
        # [DISABLED] CQL Loss 계산 (보수성 제거)
        cql_loss = torch.tensor(0.0, device=self.device)  # CQL 완전 비활성화
        
        # 기존 CQL 로직 (주석 처리)
        # (1) Random Actions 샘플링 (-1 ~ 1)
        # batch_size * num_random 개의 랜덤 액션 생성
        # [Batch, Num_Random, Action_Dim]
        # random_actions = torch.FloatTensor(batch_size, self.num_random, action.shape[1]).uniform_(-1, 1).to(self.device)
        
        # Critic 입력을 위해 차원 확장 및 병합
        # s_seq: [Batch, Seq, Feat] -> [Batch, 1, Seq, Feat] -> [Batch, Num_Random, Seq, Feat]
        # s_seq_exp = s_seq.unsqueeze(1).expand(-1, self.num_random, -1, -1).reshape(-1, s_seq.size(1), s_seq.size(2))
        # s_info_exp = s_info.unsqueeze(1).expand(-1, self.num_random, -1).reshape(-1, s_info.size(1))
        # random_actions_flat = random_actions.reshape(-1, action.shape[1])

        # Random Action에 대한 Q-값 계산
        # q1_rand, q2_rand = self.critic(s_seq_exp, s_info_exp, random_actions_flat)
        
        # 다시 [Batch, Num_Random, 1] 형태로 복구
        # q1_rand = q1_rand.view(batch_size, self.num_random, 1)
        # q2_rand = q2_rand.view(batch_size, self.num_random, 1)

        # (2) CQL Loss: log(sum(exp(Q_random))) - Q_current
        # 무작위 행동의 Q값 총합(LogSumExp)이 현재 정책의 Q값보다 작아지도록(최소화) 함
        # cql1_loss = torch.logsumexp(q1_rand, dim=1).mean() - current_Q1.mean()
        # cql2_loss = torch.logsumexp(q2_rand, dim=1).mean() - current_Q2.mean()
        
        # cql_loss = (cql1_loss + cql2_loss) * 0.5 * self.cql_alpha

        # 최종 Critic Loss (CQL 제외)
        critic_loss = critic_loss_mse + cql_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -----------------------------------------------------------------
        # 3. Actor Update
        # -----------------------------------------------------------------
        # [수정] 변수 초기화 (Actor 업데이트 안 할 때를 대비)
        actor_loss_val = 0.0
        l_rl_val = 0.0
        l_teacher_val = 0.0
        
        if self.total_it % self.policy_freq == 0:
            pi, _, _ = self.actor(s_seq, s_info)
            q1, _ = self.critic(s_seq, s_info, pi)
            
            # RL Loss: Q-값 최대화
            l_rl = -q1.mean()
            
            # [Teacher-Guided] BC Loss: Oracle Action과 유사하게
            l_teacher = F.mse_loss(pi, oracle_action)
            
            # Total Actor Loss
            actor_loss = l_rl + (teacher_lambda * l_teacher)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # [수정] 텐서 값을 스칼라로 변환하여 저장
            actor_loss_val = actor_loss.item()
            l_rl_val = l_rl.item()
            l_teacher_val = l_teacher.item()

        with torch.no_grad():
            td_error_abs = (target_Q - current_Q1).abs().mean().item()
        metrics = {
            'critic_loss': critic_loss.item(),
            'cql_loss': cql_loss.item(),
            'actor_loss': actor_loss_val,
            'l_rl': l_rl_val,           # [Teacher-Guided] 초기화된 값 사용
            'l_teacher': l_teacher_val,  # [Teacher-Guided] 초기화된 값 사용
            'q1_mean': current_Q1.mean().item(),
            'q2_mean': current_Q2.mean().item(),
            'target_q_mean': target_Q.mean().item(),
            'td_error_abs': td_error_abs,
        }
        if self.total_it % 1000 == 0:
            logger.info(
                "[Critic Debug] Step %d | Q1: %.3f, CQL Loss: %.4f | TD Error: %.3f",
                self.total_it, metrics['q1_mean'], metrics['cql_loss'], metrics['td_error_abs'],
            )
        return metrics

    def save(self, filename):
        base = filename.replace('.pth', '')
        torch.save(self.actor.state_dict(), base + '_actor.pth')
        torch.save(self.critic.state_dict(), base + '_critic.pth')

    def load(self, filename):
        base = filename.replace('.pth', '')
        self.actor.load_state_dict(torch.load(base + '_actor.pth', map_location=self.device))
        self.critic.load_state_dict(torch.load(base + '_critic.pth', map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)