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
from .xlstm_network import xLSTMActorCritic


class PPOAgent:
    """PPO 알고리즘 기반 강화학습 에이전트"""
    def __init__(self, state_dim, action_dim, hidden_dim=128, device='cpu', info_dim=13):
        self.device = device
        
        # 하이퍼파라미터
        self.gamma = 0.99
        self.lmbda = 0.95
        self.eps_clip = 0.2
        self.k_epochs = 10
        
        # [중요] NoisyNet 대신 엔트로피로 탐험 조절
        # 0.05(초반 탐험) -> 학습 진행 시 0.01(수렴)로 자동 감소
        self.entropy_coef = 0.05
        
        # 모델 생성 (Noisy 옵션 제거)
        self.model = xLSTMActorCritic(state_dim, action_dim, info_dim=info_dim, hidden_dim=hidden_dim).to(device)
        
        # 학습률 0.0001 (LayerNorm과 궁합이 좋음)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        
        # 스케줄러: 보상이 정체되면 학습률을 줄여서 미세 조정
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=200, min_lr=1e-6
        )
        
        self.memory = []  # (state, action, log_prob, reward, is_terminal)

    def select_action(self, state):
        """
        상태에 따른 행동 선택
        
        Args:
            state: (obs_seq, obs_info) 튜플
                   - obs_seq: (1, seq_len, state_dim) 시계열 데이터
                   - obs_info: (1, info_dim) 포지션 정보
        Returns:
            action: 행동 인덱스 (0: Hold, 1: Long, 2: Short)
            log_prob: 로그 확률
        """
        with torch.no_grad():
            obs_seq, obs_info = state
            probs, _ = self.model(obs_seq.to(self.device), info=obs_info.to(self.device))
            
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()

    def store_transition(self, state, action, log_prob, reward, is_terminal):
        """트랜지션 저장"""
        self.memory.append((state, action, log_prob, reward, is_terminal))

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
            current_entropy_coef = max(0.01, self.entropy_coef * (0.995 ** episode))
            entropy_bonus = current_entropy_coef * entropy
            
            loss = actor_loss + 0.5 * critic_loss - entropy_bonus
            
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