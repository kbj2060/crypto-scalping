"""
PPO 에이전트 (Final Ver)
ReduceLROnPlateau + 엔트로피 탐험 + 안정적 파라미터
NoisyNet 제거 -> 엔트로피로 탐험 조절
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import os

# 상위 폴더를 경로에 추가 (config 모듈 접근용)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from .xlstm_network import xLSTMActorCritic


class PPOAgent:
    """PPO 알고리즘 기반 강화학습 에이전트"""
    def __init__(self, state_dim, action_dim, hidden_dim=None, device='cpu', info_dim=13):
        self.device = device
        
        # 하이퍼파라미터 (config에서 가져오기)
        self.gamma = config.PPO_GAMMA
        self.lmbda = config.PPO_LAMBDA
        self.eps_clip = config.PPO_EPS_CLIP
        self.k_epochs = config.PPO_K_EPOCHS
        
        # [중요] NoisyNet 대신 엔트로피로 탐험 조절
        self.entropy_coef = config.PPO_ENTROPY_COEF
        
        # [개선 1] 에피소드 내 상태 유지를 위한 변수
        self.current_states = None
        
        # 네트워크 파라미터 (config에서 가져오기)
        hidden_dim = hidden_dim if hidden_dim is not None else config.NETWORK_HIDDEN_DIM
        
        # 모델 생성 (Dropout 등 파라미터 추가)
        self.model = xLSTMActorCritic(
            state_dim, action_dim, info_dim=info_dim, hidden_dim=hidden_dim,
            num_layers=config.NETWORK_NUM_LAYERS,
            dropout=config.NETWORK_DROPOUT,
            use_checkpointing=config.NETWORK_USE_CHECKPOINTING
        ).to(device)
        
        # 학습률 (config에서 가져오기)
        # xLSTM의 안정적인 학습을 위해 더 세밀하게 조정
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.PPO_LEARNING_RATE)
        
        # 스케줄러: 보상이 정체되면 학습률을 줄여서 미세 조정
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max',
            factor=config.PPO_SCHEDULER_FACTOR,
            patience=config.PPO_SCHEDULER_PATIENCE,
            min_lr=config.PPO_SCHEDULER_MIN_LR
        )
        
        self.memory = []  # (state, action, log_prob, reward, is_terminal)

    def reset_episode_states(self):
        """에피소드 시작 시 상태 초기화"""
        self.current_states = None

    def select_action(self, state):
        """
        Stateful Action Selection
        LSTM의 은닉 상태(Hidden State)를 유지하며 행동 결정
        
        Args:
            state: (obs_seq, obs_info, action_mask) 튜플 또는 (obs_seq, obs_info) 튜플
                   - obs_seq: (1, seq_len, state_dim) 시계열 데이터
                   - obs_info: (1, info_dim) 포지션 정보
                   - action_mask: (3,) 마스크 텐서 [HOLD, LONG, SHORT] (1=가능, 0=불가능)
        Returns:
            action: 행동 인덱스 (0: Hold, 1: Long, 2: Short)
            log_prob: 로그 확률
        """
        with torch.no_grad():
            # 상태 언패킹 (Mask가 있는지 확인)
            if len(state) == 3:
                obs_seq, obs_info, action_mask = state
            else:
                obs_seq, obs_info = state
                action_mask = torch.ones(3)  # 마스킹 없음

            # [개선 2] 상태 전달 및 업데이트 (return_states=True)
            # 이전 스텝의 self.current_states를 넣어주고,
            # 업데이트된 상태를 다시 self.current_states에 저장
            probs, _, self.current_states = self.model(
                obs_seq.to(self.device),
                info=obs_info.to(self.device),
                states=self.current_states,
                return_states=True
            )
            
            # Action Masking 적용
            action_mask = action_mask.to(self.device)
            masked_probs = probs * action_mask
            
            # 확률 합이 0이 되는 것을 방지 (모두 0이면 에러 남 -> 작은 값 더하기)
            if masked_probs.sum() == 0:
                masked_probs = torch.ones_like(masked_probs) * action_mask  # 가능한 행동 중에서 균등
                
            # 재정규화 (Softmax 효과)
            masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
            
            dist = Categorical(masked_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()

    def store_transition(self, state, action, log_prob, reward, is_terminal):
        """트랜지션 저장 (Mask 제외하고 저장)"""
        # state는 (obs_seq, obs_info, mask) 튜플이므로 앞 2개만 저장
        if len(state) == 3:
            state_to_save = (state[0], state[1])
        else:
            state_to_save = state
        self.memory.append((state_to_save, action, log_prob, reward, is_terminal))

    def compute_gae(self, rewards, values, is_terminals, next_value=0):
        """
        GAE (Generalized Advantage Estimation) 계산
        
        Args:
            rewards: 보상 리스트
            values: 가치 함수 값 리스트
            is_terminals: 종료 플래그 리스트
            next_value: 다음 상태의 가치 (마지막 상태용)
        Returns:
            advantages: 어드밴티지 텐서
            returns: 반환값 텐서
        """
        advantages = []
        gae = 0
        next_value = next_value
        
        for step in reversed(range(len(rewards))):
            if is_terminals[step]:
                delta = rewards[step] - values[step]
                gae = delta
            else:
                delta = rewards[step] + self.gamma * next_value - values[step]
                gae = delta + self.gamma * self.lmbda * gae
            
            advantages.insert(0, gae)
            next_value = values[step]
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return torch.tensor(advantages, dtype=torch.float).unsqueeze(1), \
               torch.tensor(returns, dtype=torch.float).unsqueeze(1)

    def update(self, next_state=None, episode=1):
        """PPO 업데이트 수행
        
        Args:
            next_state: 다음 상태 텐서 (부트스트랩 가치 계산용, None이면 0 사용)
            episode: 현재 에피소드 번호 (엔트로피 스케줄러용)
        Returns:
            평균 loss 값
        """
        if len(self.memory) == 0:
            return 0.0
        
        # 1. 메모리에서 데이터 추출 (Late Fusion 구조: (obs_seq, obs_info) 튜플)
        states_seq = torch.cat([m[0][0] for m in self.memory], dim=0).to(self.device)
        states_info = torch.cat([m[0][1] for m in self.memory], dim=0).to(self.device)
        
        actions = torch.tensor([m[1] for m in self.memory], dtype=torch.long).unsqueeze(1).to(self.device)
        old_log_probs = torch.tensor([m[2] for m in self.memory], dtype=torch.float).unsqueeze(1).to(self.device)
        rewards = [m[3] for m in self.memory]
        is_terminals = [m[4] for m in self.memory]

        # 2. 부트스트랩 가치(next_value) 결정
        next_value = 0
        if next_state is not None:
            with torch.no_grad():
                # 다음 상태의 Value를 예측하여 미래 보상의 기댓값으로 사용
                obs_seq, obs_info = next_state
                _, next_v = self.model(obs_seq.to(self.device), info=obs_info.to(self.device))
                next_value = next_v.item()

        # 3. 현재 상태들의 가치 계산
        with torch.no_grad():
            _, values = self.model(states_seq, info=states_info)
            values = values.squeeze().cpu().numpy().tolist()
        
        # 4. GAE 계산
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            if is_terminals[step]:
                delta = rewards[step] - values[step]
                gae = delta
            else:
                nv = values[step + 1] if step + 1 < len(values) else next_value
                delta = rewards[step] + self.gamma * nv - values[step]
                gae = delta + self.gamma * self.lmbda * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float).to(self.device)
        
        # Advantage 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 업데이트 (k_epochs 반복)
        total_loss = 0
        for _ in range(self.k_epochs):
            probs, curr_values = self.model(states_seq, info=states_info)
            dist = Categorical(probs)
            curr_log_probs = dist.log_prob(actions.squeeze()).unsqueeze(1)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(curr_log_probs - old_log_probs)
            
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.unsqueeze(1)
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(curr_values.squeeze(), returns)
            
            # 엔트로피 보너스 (탐험 유도 -> 수렴)
            # 엔트로피 감소 속도 (config에서 가져오기)
            current_entropy_coef = max(
                config.PPO_ENTROPY_MIN,
                self.entropy_coef * (config.PPO_ENTROPY_DECAY ** episode)
            )
            entropy_bonus = current_entropy_coef * entropy
            
            # [수정] Critic Loss 가중치를 0.5 -> 1.0으로 상향 (가치 판단 안정화)
            loss = actor_loss + 1.0 * critic_loss - entropy_bonus
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()

        self.memory = []
        return total_loss / self.k_epochs

    def save_model(self, filepath):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def step_scheduler(self, metric):
        """외부에서 호출하여 학습률 조절"""
        self.scheduler.step(metric)

    def load_model(self, filepath):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # reset_noise 메서드는 이제 필요 없으므로 빈 함수로 둠 (호출부 호환성)
    def reset_noise(self):
        pass