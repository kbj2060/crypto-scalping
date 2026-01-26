"""
PPO Agent (Orthogonal Initialization + 가중치 초기화 강화)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from .xlstm_network import XLSTMNetwork

class PPOAgent:
    def __init__(self, state_dim, action_dim, info_dim=13, hidden_dim=None, device='cpu'):
        self.device = device
        self.action_dim = action_dim
        
        # Network parameters
        hidden_dim = hidden_dim if hidden_dim is not None else config.NETWORK_HIDDEN_DIM
        
        # [핵심] 네트워크 초기화 (XLSTMNetwork 사용)
        # [핵심 수정] info_dim 전달!
        self.model = XLSTMNetwork(
            input_dim=state_dim, 
            action_dim=action_dim,
            info_dim=info_dim,  # <-- 이 부분이 누락되어 있었습니다
            hidden_dim=hidden_dim,
            num_layers=config.NETWORK_NUM_LAYERS
        ).to(device)
        
        # 학습률 안전장치 (너무 크면 안됨)
        lr = config.PPO_LEARNING_RATE
        if lr > 0.001: 
            lr = 0.0003  # 강제 조정
            import logging
            logging.warning(f"⚠️ 학습률이 너무 큽니다. {config.PPO_LEARNING_RATE} → {lr}로 조정합니다.")
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        
        self.current_states = None
        self.gamma = config.PPO_GAMMA
        self.lmbda = config.PPO_LAMBDA  # PPO_GAE_LAMBDA 대신 PPO_LAMBDA 사용
        self.eps_clip = config.PPO_EPS_CLIP
        self.k_epochs = config.PPO_K_EPOCHS
        self.entropy_coef = config.PPO_ENTROPY_COEF
        
        self.data = []
        
    def reset_episode_states(self):
        """에피소드 시작 시 상태 초기화"""
        self.current_states = None
        
    def load_model(self, path):
        """모델 로드 (안전한 로딩)"""
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device)
                
                # 체크포인트 형식 확인
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                else:
                    model_state = checkpoint
                
                # 모델 구조가 다를 경우를 대비한 안전 로딩
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in model_state.items() 
                                 if k in model_dict and v.size() == model_dict[k].size()}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict, strict=False)
                
                # Optimizer도 로드 (있는 경우)
                if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except:
                        pass
                
                print(f"✅ 모델 로드 성공 (일치하는 레이어만): {path}")
            except Exception as e:
                print(f"⚠️ 모델 로드 실패: {e}")
        else:
            print(f"⚠️ 모델 파일 없음: {path}")

    def save_model(self, path):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def put_data(self, transition):
        """트랜지션 저장"""
        self.data.append(transition)
        
    def select_action(self, state, action_mask=None):
        """
        행동 선택
        state: (obs_seq, obs_info) 튜플
        action_mask: [1, 1, 1] 형태의 리스트 or 텐서 (1=가능, 0=불가능)
        Returns: (action, log_prob)
        """
        obs_seq, obs_info = state
        
        # 텐서 변환 (이미 텐서인 경우도 처리)
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.FloatTensor(obs_seq).to(self.device)
        else:
            obs_seq = obs_seq.to(self.device)
            
        if not isinstance(obs_info, torch.Tensor):
            obs_info = torch.FloatTensor(obs_info).unsqueeze(0).to(self.device)
        else:
            obs_info = obs_info.to(self.device)
        
        with torch.no_grad():
            probs, _, self.current_states = self.model(obs_seq, obs_info, self.current_states)
            
            # Action Masking (만약 필요하다면)
            if action_mask is not None:
                mask = torch.FloatTensor(action_mask).to(self.device)
                probs = probs * mask
                if probs.sum() == 0: 
                    probs = torch.ones_like(probs) / len(probs)
                else:
                    probs = probs / probs.sum()
                
            dist = Categorical(probs)
            action = dist.sample()
            
        return action.item(), dist.log_prob(action).item()

    def train_net(self, episode=1):
        """PPO 네트워크 학습"""
        if not self.data: 
            return 0.0
        
        # 데이터 배치 만들기
        s_seq_lst, s_info_lst, a_lst, r_lst, next_s_seq_lst, next_s_info_lst, prob_a_lst, done_lst = [], [], [], [], [], [], [], []
        
        for transition in self.data:
            s, a, r, next_s, prob_a, done = transition
            
            # State는 (obs_seq, obs_info) 튜플
            s_seq_lst.append(s[0])
            s_info_lst.append(s[1])
            a_lst.append([a])
            r_lst.append([r])
            next_s_seq_lst.append(next_s[0])
            next_s_info_lst.append(next_s[1])
            prob_a_lst.append([prob_a])
            done_lst.append([0 if done else 1])  # done=True면 mask=0

        # Tensor 변환
        # obs_seq는 이미 배치 차원이 있을 수 있으므로 확인
        if isinstance(s_seq_lst[0], torch.Tensor):
            s_seq = torch.cat(s_seq_lst, dim=0).to(self.device)
            next_s_seq = torch.cat(next_s_seq_lst, dim=0).to(self.device)
        else:
            s_seq = torch.tensor(np.concatenate(s_seq_lst), dtype=torch.float).to(self.device)
            next_s_seq = torch.tensor(np.concatenate(next_s_seq_lst), dtype=torch.float).to(self.device)
            
        if isinstance(s_info_lst[0], torch.Tensor):
            s_info = torch.cat(s_info_lst, dim=0).to(self.device)
            next_s_info = torch.cat(next_s_info_lst, dim=0).to(self.device)
        else:
            s_info = torch.tensor(np.stack(s_info_lst), dtype=torch.float).to(self.device)
            next_s_info = torch.tensor(np.stack(next_s_info_lst), dtype=torch.float).to(self.device)
            
        a = torch.tensor(a_lst, dtype=torch.long).to(self.device)
        r = torch.tensor(r_lst, dtype=torch.float).to(self.device)
        done_mask = torch.tensor(done_lst, dtype=torch.float).to(self.device)
        prob_a = torch.tensor(prob_a_lst, dtype=torch.float).to(self.device)

        self.data = []  # 버퍼 비우기

        # Advantage Calculation
        with torch.no_grad():
            # LSTM State 처리가 복잡하므로 배치 단위에선 Stateless로 근사
            _, v, _ = self.model(s_seq, s_info, states=None)
            _, next_v, _ = self.model(next_s_seq, next_s_info, states=None)
            
            td_target = r + self.gamma * next_v * done_mask
            delta = td_target - v
        
        # GAE (Generalized Advantage Estimation) - 간단 버전
        # 실제 구현에선 Loop를 돌며 GAE를 계산해야 정확함
        # 여기서는 즉시 보상 기반으로 단순화하여 학습 안정성 확보
        advantage = delta 
        
        # Advantage 정규화
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        # PPO Update Loop
        total_loss = 0.0
        for i in range(self.k_epochs):
            curr_probs, curr_v, _ = self.model(s_seq, s_info, states=None)
            dist = Categorical(curr_probs)
            curr_log_prob = dist.log_prob(a.squeeze()).unsqueeze(1)
            
            ratio = torch.exp(curr_log_prob - prob_a)
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            
            # 엔트로피 감소
            current_entropy_coef = max(
                config.PPO_ENTROPY_MIN,
                self.entropy_coef * (config.PPO_ENTROPY_DECAY ** episode)
            )
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * F.mse_loss(curr_v, td_target)
            entropy_loss = current_entropy_coef * dist.entropy().mean()
            
            loss = actor_loss + critic_loss - entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # Gradient Clipping
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / self.k_epochs
