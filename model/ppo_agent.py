"""
PPO (Proximal Policy Optimization) 에이전트
클리핑과 GAE(Generalized Advantage Estimation) 구현
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from .xlstm_network import xLSTMActorCritic


class PPOAgent:
    """PPO 알고리즘 기반 강화학습 에이전트"""
    def __init__(self, state_dim, action_dim, hidden_dim=128, device='cpu'):
        self.device = device
        self.gamma = 0.999  # 할인율 (인내심 강화: 미래 보상에 더 높은 가치 부여)
        self.lmbda = 0.95  # GAE 파라미터
        self.eps_clip = 0.2  # PPO 클리핑 범위
        self.k_epochs = 10  # 업데이트 반복 횟수
        
        # 신경망 모델
        self.model = xLSTMActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        self.memory = []  # (state, action, log_prob, reward, is_terminal)

    def select_action(self, state):
        """
        상태에 따른 행동 선택
        
        Args:
            state: (1, seq_len, state_dim) 텐서
        Returns:
            action: 행동 인덱스 (0: Hold, 1: Long, 2: Short)
            log_prob: 로그 확률
        """
        with torch.no_grad():
            probs, _ = self.model(state)
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
        """
        if len(self.memory) == 0:
            return
        
        # 1. 메모리에서 데이터 추출
        states = torch.cat([m[0] for m in self.memory], dim=0)
        actions = torch.tensor([m[1] for m in self.memory], dtype=torch.long).unsqueeze(1).to(self.device)
        old_log_probs = torch.tensor([m[2] for m in self.memory], dtype=torch.float).unsqueeze(1).to(self.device)
        rewards = [m[3] for m in self.memory]
        is_terminals = [m[4] for m in self.memory]

        # 2. 부트스트랩 가치(next_value) 결정 [핵심 추가]
        next_value = 0
        if next_state is not None:
            with torch.no_grad():
                # 다음 상태의 Value를 예측하여 미래 보상의 기댓값으로 사용
                _, next_v = self.model(next_state.to(self.device))
                next_value = next_v.item()

        # 3. 현재 상태들의 가치 계산
        with torch.no_grad():
            _, values = self.model(states)
            values = values.squeeze().cpu().numpy().tolist()
        
        # 4. GAE 계산 시 추출한 next_value 전달
        advantages, returns = self.compute_gae(rewards, values, is_terminals, next_value=next_value)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # PPO 업데이트 (k_epochs 반복)
        for _ in range(self.k_epochs):
            probs, values = self.model(states)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions.squeeze()).unsqueeze(1)
            entropy = dist.entropy().mean()
            
            # PPO Ratio & Clipped Objective
            ratio = torch.exp(log_probs - old_log_probs)
            advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            surr1 = ratio * advantages_normalized
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_normalized
            
            # [핵심] 엔트로피 스케줄러 적용
            # 초기 0.05에서 시작하여 서서히 0.01로 수렴 (인내심 강화)
            # 에피소드가 진행될수록 탐험을 줄이고 수렴을 강화
            entropy_coef = max(0.01, 0.05 * (0.998 ** episode))
            entropy_bonus = entropy_coef * entropy
            
            # Loss Function 계산
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values, returns)
            
            # Softmax의 조기 수렴을 방지하기 위해 가변적인 entropy_bonus 사용
            loss = actor_loss + 0.5 * critic_loss - entropy_bonus
            
            # 역전파 및 최적화
            self.optimizer.zero_grad()
            loss.backward()
            # xLSTM 안정화를 위한 그래디언트 클리핑
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
        
        # 메모리 초기화
        self.memory = []

    def save_model(self, filepath):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load_model(self, filepath):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
