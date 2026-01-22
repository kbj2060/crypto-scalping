"""
Double DQN 에이전트
일반 DQN의 Q값 과대평가 문제를 보정하여 더 정밀한 매매를 가능하게 함
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from .dqn_model import DuelingGRU
from .replay_buffer import ReplayBuffer
from .prioritized_replay_buffer import PrioritizedReplayBuffer
import logging

logger = logging.getLogger(__name__)


class DDQNAgent:
    """Double DQN 에이전트"""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, action_dim=3,
                 lr=0.0001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, buffer_size=50000, batch_size=64,
                 target_update=1000, device='cpu', use_per=False):
        """
        Args:
            input_dim: 입력 피처 개수
            hidden_dim: GRU hidden dimension
            num_layers: GRU 레이어 수
            action_dim: 행동 개수
            lr: 학습률
            gamma: 할인율
            epsilon_start: 초기 탐험 확률
            epsilon_end: 최소 탐험 확률
            epsilon_decay: 탐험 감소 비율
            buffer_size: 리플레이 버퍼 크기
            batch_size: 배치 크기
            target_update: 타겟 네트워크 업데이트 주기
            device: 디바이스 ('cpu' or 'cuda')
        """
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        # Epsilon-Greedy 파라미터
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 네트워크 2개 운영
        # Policy Network (Main): 실시간으로 행동을 결정하고 학습되는 네트워크
        self.policy_net = DuelingGRU(input_dim, hidden_dim, num_layers, action_dim).to(device)
        
        # Target Network: 학습의 정답지(Target)를 만드는 네트워크
        self.target_net = DuelingGRU(input_dim, hidden_dim, num_layers, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target Network는 학습하지 않음
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Experience Replay Buffer (PER 또는 일반)
        self.use_per = use_per
        if use_per:
            self.memory = PrioritizedReplayBuffer(capacity=buffer_size, alpha=0.6)
            logger.info("✅ Prioritized Experience Replay (PER) 활성화")
        else:
            self.memory = ReplayBuffer(capacity=buffer_size)
    
    def act(self, state, training=True):
        """
        Epsilon-Greedy 행동 선택
        
        Args:
            state: (obs_seq, obs_info) 튜플 또는 단일 텐서
            training: 학습 모드 여부 (False면 탐험 안 함)
        Returns:
            action: 행동 인덱스 (0, 1, 2)
        """
        # 탐험: 무작위 행동
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        # 활용: 정책 네트워크가 예측한 최적 행동
        with torch.no_grad():
            if isinstance(state, tuple):
                obs_seq, obs_info = state
                obs_seq = obs_seq.to(self.device)
                obs_info = obs_info.to(self.device)
                # Dueling GRU는 시계열만 사용 (info는 나중에 통합 가능)
                q_values = self.policy_net(obs_seq)
            else:
                q_values = self.policy_net(state.to(self.device))
            
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Double DQN 학습 로직 (PER 지원)
        
        Returns:
            loss: 손실값 (학습하지 않았으면 None)
        """
        # 버퍼에 충분한 샘플이 없으면 학습하지 않음
        if len(self.memory) < self.batch_size:
            return None
        
        # [변경] PER 샘플링 (weights, idxs 추가됨)
        if self.use_per:
            # beta 값은 학습이 진행될수록 0.4 -> 1.0으로 증가시키는 것이 정석
            beta = min(1.0, 0.4 + 0.6 * (self.update_counter / 20000))
            states, actions, rewards, next_states, dones, weights, idxs = self.memory.sample(self.batch_size, beta)
            weights = weights.to(self.device)  # 중요!
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = None
            idxs = None
        
        # 디바이스로 이동
        states_seq, states_info = states
        states_seq = states_seq.to(self.device)
        states_info = states_info.to(self.device)
        
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        
        next_states_seq, next_states_info = next_states
        next_states_seq = next_states_seq.to(self.device)
        next_states_info = next_states_info.to(self.device)
        
        dones = dones.to(self.device)
        
        # Current Q: Policy Network가 예측한 현재 상태의 Q값
        current_q_values = self.policy_net(states_seq)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q (Double DQN 방식)
        with torch.no_grad():
            # 선택: 다음 상태에서 가장 좋은 행동을 Policy Network가 고름
            next_q_values_policy = self.policy_net(next_states_seq)
            next_actions = next_q_values_policy.argmax(dim=1)
            
            # 평가: 그 행동의 가치는 Target Network가 계산
            next_q_values_target = self.target_net(next_states_seq)
            next_q = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # 최종 타겟: Y = Reward + γ * Q_target * (1 - Done)
            target_q = rewards + (self.gamma * next_q * (~dones).float())
        
        # [변경] Loss 계산: PER 가중치 적용
        if self.use_per:
            # Element-wise Loss 필요 (reduction='none')
            loss_element_wise = F.smooth_l1_loss(current_q, target_q, reduction='none')
            # PER 가중치 적용 (중요한 샘플은 Loss를 더 작게, 덜 중요한건 크게 보정하여 학습 편향 방지)
            loss = (loss_element_wise * weights).mean()
        else:
            # 일반 Loss
            loss = F.smooth_l1_loss(current_q, target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # 그래디언트 클리핑 (안정성)
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # [신규] PER 우선순위 업데이트 (TD-Error = 실제 오차)
        if self.use_per and idxs is not None:
            # 오차가 클수록(많이 틀릴수록) 우선순위를 높임
            td_errors = loss_element_wise.detach().cpu().numpy()
            self.memory.update_priorities(idxs, td_errors)
        
        # 타겟 네트워크 업데이트 (Hard Update: 1000 스텝마다)
        self.update_counter += 1
        if self.update_counter >= self.target_update:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.update_counter = 0
            logger.debug(f"타겟 네트워크 업데이트 완료")
        
        # Epsilon 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save_model(self, filepath):
        """모델 저장"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter
        }, filepath)
        logger.info(f"모델 저장 완료: {filepath}")
    
    def load_model(self, filepath):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.update_counter = checkpoint.get('update_counter', 0)
        logger.info(f"모델 로드 완료: {filepath}")
    
    def reset_noise(self):
        """NoisyNet의 노이즈 파라미터를 재설정 (탐험 다양성 확보)"""
        # policy_net에 reset_noise 메서드가 있을 때만 실행 (NoisyNet 사용 시)
        if hasattr(self.policy_net, 'reset_noise'):
            self.policy_net.reset_noise()
        
        # target_net도 같이 리셋해주면 학습 안정성에 도움됨
        if hasattr(self.target_net, 'reset_noise'):
            self.target_net.reset_noise()