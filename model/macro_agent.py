"""
MacroHFT: 계층형 에이전트 (Router + 3명의 전문가)
- Router가 시장 상황에 따라 Trend/Volatility/Sideways 전문가 비중을 결정
- 전문가는 동결(freeze), Router만 학습
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import os

from . import config
from .xlstm_network import XLSTMNetwork


class RouterNetwork(nn.Module):
    """시장 상황(State)을 보고 어떤 전문가에 비중을 줄지 결정하는 관리자"""

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
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(x)


class MacroHFTAgent(nn.Module):
    """3명의 전문가(Sub-Agents) + Router 조합"""

    def __init__(self, state_dim, action_dim, info_dim, device="cpu"):
        super().__init__()
        self.device = device
        self.action_dim = action_dim

        # 1. 전문가 로드 (동결)
        self.expert_trend = self._load_expert("data/agent_trend_best.pth", state_dim, action_dim, info_dim)
        self.expert_vol = self._load_expert("data/agent_volatility_best.pth", state_dim, action_dim, info_dim)
        self.expert_side = self._load_expert("data/agent_sideways_best.pth", state_dim, action_dim, info_dim)
        self.experts = [self.expert_trend, self.expert_vol, self.expert_side]

        for expert in self.experts:
            for p in expert.parameters():
                p.requires_grad = False
            expert.eval()

        # 2. Router (학습 대상)
        self.router = RouterNetwork(input_dim=state_dim, num_experts=3).to(device)
        self.optimizer = torch.optim.Adam(self.router.parameters(), lr=1e-4)

    def _load_expert(self, path, s_dim, a_dim, i_dim):
        model = XLSTMNetwork(
            input_dim=s_dim,
            action_dim=a_dim,
            info_dim=i_dim,
            hidden_dim=config.NETWORK_HIDDEN_DIM,
            num_layers=config.NETWORK_NUM_LAYERS,
            dropout=0.0,
        ).to(self.device)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded Expert: {path}")
        else:
            print(f"⚠️ Expert not found: {path} (Random Init)")
        return model

    def forward(self, x, info, states=None):
        """
        1. Router가 전문가별 가중치 계산
        2. 각 전문가가 Logits 출력
        3. 가중 평균 Logits 반환
        """
        weights = self.router(x)

        expert_logits = []
        with torch.no_grad():
            for expert in self.experts:
                out = expert(x, info, states=states)
                logits = out[0]
                expert_logits.append(logits)
        stacked = torch.stack(expert_logits, dim=1)
        final_logits = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return final_logits, weights

    def select_action(self, state, action_mask=None):
        obs_seq, obs_info = state
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.FloatTensor(obs_seq).to(self.device)
        if not isinstance(obs_info, torch.Tensor):
            obs_info = torch.FloatTensor(obs_info).unsqueeze(0).to(self.device)

        logits, router_weights = self.forward(obs_seq, obs_info)

        if action_mask is not None:
            mask_tensor = torch.FloatTensor(action_mask).to(self.device)
            logits = logits + (mask_tensor - 1) * 1e10

        dist = Categorical(logits=logits)
        action = dist.sample()
        if action_mask is not None and action_mask[action.item()] == 0:
            allowed = torch.where(mask_tensor == 1)[0]
            if len(allowed) > 0:
                best_idx = logits[0, allowed].argmax()
                action = allowed[best_idx].unsqueeze(0)
                dist = Categorical(logits=logits)

        value_placeholder = 0.0
        return action.item(), dist.log_prob(action).item(), value_placeholder
