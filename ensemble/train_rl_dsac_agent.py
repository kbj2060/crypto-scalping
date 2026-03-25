"""
DSAC 코인 트레이딩 에이전트 (Distributional Soft Actor-Critic)
================================================================
기존 SAC 파이프라인을 유지하면서 Critic을 분포형(quantile)으로 교체한다.

핵심 차이:
  - SAC: Q(s,a) 평균 스칼라 추정
  - DSAC: Q(s,a)의 분위수 분포를 추정하고, Actor는 CVaR 기반으로 업데이트

구현 요약:
  1) Actor: 기존 GaussianActor 재사용 (연속 action, -1~+1)
  2) Critic: DistributionalTwinCritic (각각 N개 quantile 출력)
  3) Critic loss: quantile Huber regression
  4) Actor loss: mean-Q 대신 하위 분위 평균(CVaR) 사용
"""

import copy
import gc
import logging
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR, _SCRIPT_DIR, os.path.join(_ROOT_DIR, "ensemble"), os.path.join(_ROOT_DIR, "strategies")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from ensemble.train_rl_agent import (  # noqa: E402
    OnlineHMMDetector,
    MultiTimeframeFeatures,
    STACKED_STATE_DIM,
)
from ensemble.train_rl_sac_agent import (  # noqa: E402
    SACTradingEnv,
    GaussianActor,
    FeatureExtractor,
    ReplayBuffer,
    SACRouter as _BaseSACRouter,
)


def _quantile_huber_loss(
    pred_q: torch.Tensor,
    target_q: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """IQN 계열 quantile Huber loss.

    Args:
        pred_q:   [B, N]
        target_q: [B, N]
        taus:     [N] in (0,1)
    """
    td = target_q.unsqueeze(1) - pred_q.unsqueeze(2)  # [B, N, N]
    abs_td = td.abs()
    huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa))
    tau = taus.view(1, -1, 1)
    weight = (tau - (td.detach() < 0).float()).abs()
    return (weight * huber / kappa).mean()


class DistributionalTwinCritic(nn.Module):
    """각 Critic이 N개 quantile을 출력하는 Twin Critic."""

    def __init__(self, state_dim=STACKED_STATE_DIM, hidden_dim=256, n_quantiles=32):
        super().__init__()
        self.n_quantiles = int(n_quantiles)

        self.feat1 = FeatureExtractor(state_dim, hidden_dim)
        self.q1 = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_quantiles),
        )

        self.feat2 = FeatureExtractor(state_dim, hidden_dim)
        self.q2 = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_quantiles),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = self.feat1(state)
        f2 = self.feat2(state)
        x1 = torch.cat([f1, action], dim=1)
        x2 = torch.cat([f2, action], dim=1)
        return self.q1(x1), self.q2(x2)  # [B, N], [B, N]


class DSACAgent:
    """Distributional Soft Actor-Critic (risk-averse via CVaR)."""

    def __init__(
        self,
        state_dim=STACKED_STATE_DIM,
        hidden_dim=256,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        n_quantiles=32,
        cvar_frac=0.25,
        device="cuda",
    ):
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.n_quantiles = int(n_quantiles)
        self.cvar_frac = float(cvar_frac)

        self.actor = GaussianActor(state_dim, hidden_dim).to(device)
        self.critic = DistributionalTwinCritic(state_dim, hidden_dim, self.n_quantiles).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_target.eval()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.target_entropy = -1.0  # action_dim=1
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.taus = torch.linspace(
            0.5 / self.n_quantiles,
            1.0 - 0.5 / self.n_quantiles,
            self.n_quantiles,
            device=device,
            dtype=torch.float32,
        )

        self.memory = ReplayBuffer(capacity=500000)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    def act(self, state: np.ndarray, deterministic: bool = False) -> float:
        state_ts = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(state_ts)
            else:
                action, _ = self.actor.sample(state_ts)
        return float(action.cpu().item())

    def _target_quantiles(self, ns: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(ns)  # [B,1], [B,1]
            tq1, tq2 = self.critic_target(ns, next_action)      # [B,N], [B,N]

            # 보수적으로 mean-Q가 낮은 헤드의 전체 분포를 사용
            tq1_m = tq1.mean(dim=1, keepdim=True)
            tq2_m = tq2.mean(dim=1, keepdim=True)
            chosen_tq = torch.where(tq1_m <= tq2_m, tq1, tq2)

            entropy_term = self.log_alpha.exp().detach() * next_log_prob  # [B,1]
            target_q = r + self.gamma * (1.0 - d) * (chosen_tq - entropy_term)
            return target_q

    def _cvar_min(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        k = max(1, int(self.n_quantiles * self.cvar_frac))
        q1_s, _ = torch.sort(q1, dim=1)
        q2_s, _ = torch.sort(q2, dim=1)
        c1 = q1_s[:, :k].mean(dim=1, keepdim=True)
        c2 = q2_s[:, :k].mean(dim=1, keepdim=True)
        return torch.min(c1, c2)

    def update(self, batch_size=256) -> dict:
        if len(self.memory) < batch_size:
            return {}

        s, a, r, ns, d = self.memory.sample(batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        target_q = self._target_quantiles(ns, r, d)  # [B,N]

        q1, q2 = self.critic(s, a)  # [B,N], [B,N]
        critic_loss = _quantile_huber_loss(q1, target_q, self.taus) + _quantile_huber_loss(q2, target_q, self.taus)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        new_action, log_prob = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, new_action)
        q_cvar = self._cvar_min(q1_new, q2_new)  # [B,1]
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * log_prob - q_cvar).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.log_alpha.exp().item()),
            "mean_q": float(torch.min(q1_new.mean(dim=1), q2_new.mean(dim=1)).mean().item()),
            "cvar_q": float(q_cvar.mean().item()),
        }


class DSACRouter(_BaseSACRouter):
    """라이브 추론 호환 라우터 (기존 SACRouter 인터페이스 유지)."""

    def decide(self, features, pos):
        action_int, leverage, info = super().decide(features, pos)
        info = dict(info or {})
        info["agent"] = "DSAC"
        return action_int, leverage, info


# 호환용 alias (trading_bot에서 SACRouter 이름으로 import 가능)
SACRouter = DSACRouter


def train():
    csv_path = "data/rl_training_data_full.csv"
    if not os.path.exists(csv_path):
        logger.error("데이터가 없습니다. --mode generate_csv 실행 요망")
        return

    df = pd.read_csv(csv_path)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val = df.iloc[split_idx:].reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    logger.info("[HMM] OnlineHMMDetector 초기 학습 시작...")
    hmm_detector = OnlineHMMDetector()
    hmm_detector.fit(df_train, n_iter=30)
    logger.info("[HMM] 초기 학습 완료.")

    logger.info("[MTF] 멀티타임프레임 피처 선계산 중...")
    mtf_train = MultiTimeframeFeatures(df_train["close"].values.astype(np.float32))
    mtf_val = MultiTimeframeFeatures(df_val["close"].values.astype(np.float32))
    logger.info("[MTF] 선계산 완료.")

    train_hmm = copy.deepcopy(hmm_detector)
    env = SACTradingEnv(df_train, phase="train", hmm_detector=train_hmm, mtf_features=mtf_train)
    agent = DSACAgent(STACKED_STATE_DIM, hidden_dim=256, n_quantiles=32, cvar_frac=0.25, device=device)

    nep = 1000
    batch = 256
    update_freq = 4
    min_buffer = 4096
    warmup_steps = 10000
    global_step = 0

    best_val_score = -float("inf")
    best_val_pnl = -float("inf")

    os.makedirs("data/ensemble/ckpt", exist_ok=True)
    ckpt_path = "data/ensemble/ckpt/dsac_checkpoint.pth"
    best_path = "data/ensemble/ckpt/best_dsac_agents.pth"

    start_ep = 1
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            agent.actor.load_state_dict(ckpt["actor"])
            agent.critic.load_state_dict(ckpt["critic"])
            agent.critic_target.load_state_dict(ckpt["critic_target"])
            agent.log_alpha.data.copy_(ckpt["log_alpha"])
            agent.actor_optimizer.load_state_dict(ckpt["actor_opt"])
            agent.critic_optimizer.load_state_dict(ckpt["critic_opt"])
            agent.alpha_optimizer.load_state_dict(ckpt["alpha_opt"])
            global_step = int(ckpt.get("global_step", 0))
            best_val_pnl = float(ckpt.get("best_val_pnl", -float("inf")))
            best_val_score = float(ckpt.get("best_val_score", -float("inf")))
            start_ep = int(ckpt.get("epoch", 0)) + 1
            logger.info(f"♻️ [복원] ep={start_ep-1} | global_step={global_step} | best_pnl={best_val_pnl:.2f}%")

            if len(agent.memory) < min_buffer:
                refill_steps = max(warmup_steps, min_buffer)
                logger.info(f"    [WARMUP 재실행] 버퍼 비어있음 → {refill_steps} 스텝 랜덤 탐험으로 리필")
                warmup_env = SACTradingEnv(
                    df_train,
                    phase="train",
                    hmm_detector=copy.deepcopy(hmm_detector),
                    mtf_features=mtf_train,
                )
                ws = warmup_env.reset()
                for _ in range(refill_steps):
                    wa = np.random.uniform(-1.0, 1.0)
                    wns, wr, wd, _ = warmup_env.step(wa)
                    agent.memory.push(ws, wa, wr, wns, wd)
                    ws = wns
                    if wd:
                        ws = warmup_env.reset()
                logger.info(f"    [WARMUP 완료] 버퍼: {len(agent.memory)}")
        except Exception as e:
            logger.warning(f"⚠️ 체크포인트 복원 실패: {e}")

    def _save_checkpoint(ep: int):
        torch.save(
            {
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "critic_target": agent.critic_target.state_dict(),
                "log_alpha": agent.log_alpha.data,
                "actor_opt": agent.actor_optimizer.state_dict(),
                "critic_opt": agent.critic_optimizer.state_dict(),
                "alpha_opt": agent.alpha_optimizer.state_dict(),
                "global_step": global_step,
                "best_val_pnl": best_val_pnl,
                "best_val_score": best_val_score,
                "epoch": ep,
            },
            ckpt_path,
        )

    ep = start_ep
    try:
        for ep in range(start_ep, nep + 1):
            state = env.reset()
            ep_reward = 0.0
            done = False
            last_stats = {}

            while not done:
                global_step += 1

                if global_step < warmup_steps:
                    action = np.random.uniform(-1.0, 1.0)
                else:
                    action = agent.act(state, deterministic=False)

                next_state, reward, done, _ = env.step(action)
                agent.memory.push(state, action, reward, next_state, done)
                ep_reward += reward
                state = next_state

                if global_step % update_freq == 0 and len(agent.memory) >= min_buffer:
                    last_stats = agent.update(batch)

            pnl = (env.balance / env.initial_balance - 1.0) * 100.0
            _cvar = float(last_stats.get("cvar_q", 0.0))
            logger.info(
                f"Ep {ep:04d} | PnL:{pnl:6.1f}% Tr:{env.total_trades:4d} "
                f"WR:{env.win_rate * 100:4.0f}% Rew:{ep_reward:7.3f} | "
                f"buf:{len(agent.memory):6d} | α:{agent.alpha:.4f} | CVaR_Q:{_cvar:+.4f}"
            )

            if ep % 10 == 0:
                val_hmm = copy.deepcopy(hmm_detector)
                val_env = SACTradingEnv(df_val, phase="val", hmm_detector=val_hmm, mtf_features=mtf_val)

                val_state = val_env.reset()
                val_done = False
                agent.actor.eval()
                while not val_done:
                    with torch.no_grad():
                        val_action = agent.act(val_state, deterministic=True)
                    val_state, _, val_done, _ = val_env.step(val_action)
                agent.actor.train()

                val_pnl = (val_env.balance / val_env.initial_balance - 1.0) * 100.0
                val_wr = val_env.win_rate
                if val_env.total_trades == 0:
                    val_trade_score = -5.0
                elif val_pnl > 0:
                    val_trade_score = min(val_env.total_trades / 30.0, 1.0) * 5.0
                else:
                    val_trade_score = -min(val_env.total_trades / 30.0, 1.0) * 10.0
                val_score = val_pnl * 5.0 + val_wr * 20.0 + val_trade_score

                logger.info(
                    f"    [VAL] PnL:{val_pnl:6.2f}% | Tr:{val_env.total_trades:4d} | "
                    f"WR:{val_wr*100:.0f}% | Score:{val_score:.2f}"
                )

                if val_score > best_val_score:
                    best_val_score, best_val_pnl = val_score, val_pnl
                    torch.save(
                        {
                            "actor": agent.actor.state_dict(),
                            "critic": agent.critic.state_dict(),
                            "best_pnl": best_val_pnl,
                            "best_score": best_val_score,
                            "epoch": ep,
                            "meta": {"algo": "DSAC", "n_quantiles": agent.n_quantiles, "cvar_frac": agent.cvar_frac},
                        },
                        best_path,
                    )
                    logger.info(f"    🎉 [NEW BEST] 저장 완료 (PnL:{best_val_pnl:.2f}%)")

                if ep % 50 == 0:
                    hmm_detector.update_online(n_iter=5)
                    train_hmm.A = hmm_detector.A.copy()
                    train_hmm.mu = hmm_detector.mu.copy()
                    train_hmm.sigma = hmm_detector.sigma.copy()
                    train_hmm.pi = hmm_detector.pi.copy()
                    train_hmm._obs_mean = hmm_detector._obs_mean.copy()
                    train_hmm._obs_std = hmm_detector._obs_std.copy()
                    logger.info("    [HMM] 온라인 업데이트 완료")

                _save_checkpoint(ep)

            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logger.info("⚠️ 학습 중단.")
        _save_checkpoint(ep)


if __name__ == "__main__":
    train()
