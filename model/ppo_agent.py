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
        self.gamma = 0.99  # 할인율
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

    def update(self):
        """PPO 업데이트 수행"""
        if len(self.memory) == 0:
            return
        
        # 메모리에서 데이터 추출
        states = torch.cat([m[0] for m in self.memory], dim=0)
        actions = torch.tensor([m[1] for m in self.memory], dtype=torch.long).unsqueeze(1).to(self.device)
        old_log_probs = torch.tensor([m[2] for m in self.memory], dtype=torch.float).unsqueeze(1).to(self.device)
        rewards = [m[3] for m in self.memory]
        is_terminals = [m[4] for m in self.memory]

        # 현재 가치 함수 계산
        with torch.no_grad():
            _, values = self.model(states)
            values = values.squeeze().cpu().numpy().tolist()
        
        # GAE 계산
        advantages, returns = self.compute_gae(rewards, values, is_terminals)
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
            
            # Loss Function
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values, returns)
            entropy_bonus = 0.01 * entropy
            
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
