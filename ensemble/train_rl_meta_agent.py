#!/usr/bin/env python3
"""
메타-RL 오케스트레이터 에이전트 학습.

롱/숏/프라이머리 DSAC 스페셜리스트의 추론 결과값(logit, std, action)을
입력 피처로 삼아, 최종 진입·청산 결정을 학습한다.

상태 벡터 (META_STATE_DIM = 28):
  [0-1]   primary_raw_n, primary_std_n
  [2-6]   long_logit_n, long_raw_n, long_std_n, short_logit_n, short_raw_n
  [7-8]   short_std_n, std_diff_n
  [9-12]  direction_n, agreement_n, conviction_n, ambiguity_n
  [13-15] regime_net, regime_chop_n, regime_entropy
  [16-20] m7_net_dir, m7_conf, m7_quality_n, m7_anomaly, m7_vol_rank
  [21-24] pos_sign, hold_norm, unrealized_n, leverage
  [25-27] garch_z, jump_z, smart_flow_n

사용법:
  python ensemble/train_rl_meta_agent.py \
      --csv data/splits/year_oos/rl_meta_2026.csv \
      --episodes 300 --device cpu
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("meta_agent")

# ─────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────
META_STATE_DIM   = 28
_POS_THRESH      = 0.18   # |action| > thresh → LONG/SHORT
_CLOSE_THRESH    = 0.08   # |action| < thresh (보유 중) → 청산
_MAX_HOLD        = 288    # 강제 청산 최대 보유 (캔들)
_FORCE_STOP      = -0.025 # 강제 청산 손실
LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0
_CKPT_PATH = str(_ROOT / "data" / "ensemble" / "ckpt" / "dsac_meta_checkpoint.pth")
_BEST_PATH = str(_ROOT / "data" / "ensemble" / "ckpt" / "best_dsac_meta_agents.pth")


# ─────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────
def _sf(v, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _nt(x: float, scale: float) -> float:
    """tanh 정규화."""
    return float(np.tanh(x / max(scale, 1e-8)))


def _resolve_device(s: str) -> str:
    if s == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return s


# ─────────────────────────────────────────────────────────────────
# 신경망
# ─────────────────────────────────────────────────────────────────
class _FeatExtractor(nn.Module):
    def __init__(self, state_dim: int = META_STATE_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),    nn.LayerNorm(hidden), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetaGaussianActor(nn.Module):
    """28D state → tanh action ∈ [-1, 1]  (양수=LONG, 음수=SHORT)."""

    def __init__(self, state_dim: int = META_STATE_DIM, hidden: int = 256):
        super().__init__()
        self.feat      = _FeatExtractor(state_dim, hidden)
        self.mu_head   = nn.Linear(hidden, 1)
        self.lsd_head  = nn.Linear(hidden, 1)

    def forward_logits(self, state: torch.Tensor):
        f = self.feat(state)
        mu  = self.mu_head(f)
        lsd = self.lsd_head(f).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return f, mu, lsd

    def sample(self, state: torch.Tensor):
        _, mu, lsd = self.forward_logits(state)
        std = lsd.exp()
        dist = Normal(mu, std)
        x = dist.rsample()
        act = torch.tanh(x)
        lp  = dist.log_prob(x) - torch.log(1.0 - act.pow(2) + 1e-6)
        return act, lp.sum(-1, keepdim=True)

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        _, mu, _ = self.forward_logits(state)
        return torch.tanh(mu)


class _TwinCritic(nn.Module):
    def __init__(self, state_dim: int = META_STATE_DIM, hidden: int = 256, n_q: int = 32):
        super().__init__()
        self.n_q = n_q
        self.f1 = _FeatExtractor(state_dim, hidden)
        self.q1 = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, n_q))
        self.f2 = _FeatExtractor(state_dim, hidden)
        self.q2 = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, n_q))

    def forward(self, state, action):
        sa1 = torch.cat([self.f1(state), action], dim=-1)
        sa2 = torch.cat([self.f2(state), action], dim=-1)
        return self.q1(sa1), self.q2(sa2)


# ─────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────
class _ReplayBuffer:
    def __init__(self, capacity: int = 300_000):
        cap = capacity
        self._s  = np.zeros((cap, META_STATE_DIM), dtype=np.float32)
        self._a  = np.zeros((cap,), dtype=np.float32)
        self._r  = np.zeros((cap,), dtype=np.float32)
        self._ns = np.zeros((cap, META_STATE_DIM), dtype=np.float32)
        self._d  = np.zeros((cap,), dtype=np.float32)
        self._ptr = self._size = 0
        self._cap = cap

    def push(self, s, a, r, ns, d):
        i = self._ptr
        self._s[i]  = s
        self._a[i]  = a
        self._r[i]  = r
        self._ns[i] = ns
        self._d[i]  = d
        self._ptr  = (i + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, n: int):
        idx = np.random.randint(0, self._size, n)
        return (self._s[idx], self._a[idx], self._r[idx],
                self._ns[idx], self._d[idx])

    def __len__(self) -> int:
        return self._size


# ─────────────────────────────────────────────────────────────────
# 상태 벡터 구성
# ─────────────────────────────────────────────────────────────────
def _build_state(row: Any, pos_sign: float, hold: int, unr: float, lev: float) -> np.ndarray:
    """CSV 한 행 + 포지션 상태 → 28D 상태 벡터."""
    # Specialist outputs
    p_raw   = _nt(_sf(row.meta_primary_raw), 1.0)
    p_std   = _nt(max(_sf(row.meta_primary_std, 1.0), 0.01), 2.0)
    l_logit = _nt(_sf(row.meta_long_logit), 3.0)
    l_raw   = float(np.clip(_sf(row.meta_long_raw), 0.0, 1.0)) * 2.0 - 1.0  # [0,1]→[-1,1]
    l_std   = _nt(max(_sf(row.meta_long_std, 1.0), 0.01), 2.0)
    s_logit = _nt(_sf(row.meta_short_logit), 3.0)
    s_raw   = float(np.clip(_sf(row.meta_short_raw), 0.0, 1.0)) * 2.0 - 1.0
    s_std   = _nt(max(_sf(row.meta_short_std, 1.0), 0.01), 2.0)
    std_diff= _nt(_sf(row.meta_long_std, 1.0) - _sf(row.meta_short_std, 1.0), 1.0)

    # Derived signals
    l_s_raw = _sf(row.meta_long_std, 1.0)
    r_s_raw = _sf(row.meta_short_std, 1.0)
    avg_std = max(0.5 * (l_s_raw + r_s_raw), 1e-6)
    dir_raw = _sf(row.meta_long_logit) - _sf(row.meta_short_logit)
    direction  = _nt(dir_raw, 3.0)
    agreement  = _nt(abs(dir_raw) / avg_std, 2.0)
    confidence = 1.0 / (1.0 + avg_std)
    conviction = _nt(abs(dir_raw) * confidence, 2.0)
    ambiguity  = _nt(min(_sf(row.meta_long_logit), _sf(row.meta_short_logit)), 3.0)

    # Regime
    bull    = float(np.clip(_sf(getattr(row, "regime_bull",    0.0)), 0.0, 1.0))
    bear    = float(np.clip(_sf(getattr(row, "regime_bear",    0.0)), 0.0, 1.0))
    chop    = float(np.clip(_sf(getattr(row, "regime_chop",    0.0)), 0.0, 1.0))
    whipsaw = float(np.clip(_sf(getattr(row, "regime_whipsaw", 0.0)), 0.0, 1.0))
    normal  = float(np.clip(_sf(getattr(row, "regime_normal",  0.0)), 0.0, 1.0))
    regime_net  = float(np.tanh(bull - bear))
    regime_chop = float(np.tanh(chop + whipsaw))
    probs = np.array([bull, bear, chop, whipsaw, normal], dtype=float) + 1e-8
    probs = probs / probs.sum()
    regime_entropy = float(-np.sum(probs * np.log(probs)) / np.log(5.0))

    # M7 context
    m7_up   = float(np.clip(_sf(getattr(row, "m7_trend_xgb_up", 1/3)), 0.0, 1.0))
    m7_dn   = float(np.clip(_sf(getattr(row, "m7_trend_xgb_dn", 1/3)), 0.0, 1.0))
    m7_conf = float(np.clip(_sf(getattr(row, "m7_confidence",   0.0)), 0.0, 1.0))
    m7_qual = _nt(_sf(getattr(row, "m7_quality_pred", 0.0)), 0.003)
    iso_a   = _sf(getattr(row, "m7_iso_anom", 0.0)) >= 0.5
    vae_a   = _sf(getattr(row, "m7_vae_anom", 0.0)) >= 0.5
    m7_anom = float(np.clip(np.tanh(float(iso_a) * 0.8 + float(vae_a) * 0.8), 0.0, 1.0))
    m7_vr   = float(np.clip(_sf(getattr(row, "m7_gmm_vol_rank", 0.5)), 0.0, 1.0))

    # Position
    hold_n = float(np.clip(np.log1p(max(hold, 0)) / np.log1p(_MAX_HOLD), 0.0, 1.0))
    unr_n  = _nt(unr, 0.02)

    # Risk
    gz = float(np.tanh(_sf(getattr(row, "garch_vol_z",     0.0)) / 3.0))
    jz = float(np.tanh(_sf(getattr(row, "jump_z",          0.0)) / 3.0))
    sf = _nt(_sf(getattr(row, "smart_money_flow", 0.0)), 0.05)

    state = np.array([
        p_raw,    p_std,
        l_logit,  l_raw,  l_std,
        s_logit,  s_raw,  s_std,  std_diff,
        direction, agreement, conviction, ambiguity,
        regime_net, regime_chop, regime_entropy,
        float(np.tanh(m7_up - m7_dn)), m7_conf, m7_qual, m7_anom, m7_vr,
        pos_sign, hold_n, unr_n, float(np.clip(lev, 0.0, 1.0)),
        gz, jz, sf,
    ], dtype=np.float32)
    assert len(state) == META_STATE_DIM, f"state dim mismatch: {len(state)}"
    return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)


# ─────────────────────────────────────────────────────────────────
# 환경
# ─────────────────────────────────────────────────────────────────
class MetaAgentEnv:
    FEE  = 0.0005
    SLIP = 0.0002
    MAX_EPISODE_STEPS = 4096

    def __init__(self, df: "pd.DataFrame", phase: str = "train"):
        import pandas as pd  # noqa
        self.df    = df.reset_index(drop=True)
        self.phase = phase
        self._rows = list(self.df.itertuples(index=False))
        self._close = df["close"].values.astype(np.float64)
        self._open  = (df["open"].values  if "open"  in df.columns else self._close).astype(np.float64)
        self.reset()

    def reset(self, start: int | None = None) -> np.ndarray:
        n = len(self._rows)
        if self.phase == "train":
            max_s = max(0, n - self.MAX_EPISODE_STEPS - 2)
            self._start = start if start is not None else random.randint(0, max_s)
        else:
            self._start = 0
        self._step = self._start
        self._end  = min(self._start + self.MAX_EPISODE_STEPS, n - 2) if self.phase == "train" else n - 2

        self.balance  = 1.0
        self.pos: str | None = None
        self.entry_px  = 0.0
        self.hold      = 0
        self.lev       = 0.0
        self.unr       = 0.0
        self.peak_bal  = 1.0
        self.total_trades = self.wins = 0
        self._just_closed = self._force_closed = False
        self._last_pnl = 0.0
        return _build_state(self._rows[self._step], 0.0, 0, 0.0, 0.0)

    @property
    def _pos_sign(self) -> float:
        return 1.0 if self.pos == "LONG" else (-1.0 if self.pos == "SHORT" else 0.0)

    def _unrealized(self, price: float) -> float:
        if self.pos is None or self.entry_px <= 0.0:
            return 0.0
        if self.pos == "LONG":
            raw = (price - self.entry_px) / self.entry_px
        else:
            raw = (self.entry_px - price) / self.entry_px
        return raw * self.lev

    def _close_position(self, fill_px: float) -> float:
        """포지션 청산 → realized PnL 반환."""
        if self.pos is None:
            return 0.0
        if self.pos == "LONG":
            raw = (fill_px * (1.0 - self.SLIP) - self.entry_px) / self.entry_px
        else:
            raw = (self.entry_px - fill_px * (1.0 + self.SLIP)) / self.entry_px
        realized = raw * self.lev - self.FEE * self.lev
        self.balance *= (1.0 + realized)
        self.peak_bal = max(self.peak_bal, self.balance)
        self._last_pnl = realized
        self.total_trades += 1
        if realized > 0:
            self.wins += 1
        self.pos = self.entry_px = None
        self.lev = self.unr = self.hold = 0
        return realized

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        row        = self._rows[self._step]
        cur_px     = float(self._close[self._step])
        next_px    = float(self._close[self._step + 1])
        fill_px_in = float(self._open[min(self._step + 1, len(self._open) - 1)])

        prev_port = self.balance * (1.0 + max(self.unr, -1.0)) if self.pos else self.balance
        self._just_closed = self._force_closed = False

        # 강제 청산 조건
        self.unr = self._unrealized(cur_px)
        if self.pos is not None and (self.unr <= _FORCE_STOP or self.hold >= _MAX_HOLD):
            self._close_position(cur_px * (1 - self.SLIP) if self.pos == "LONG" else cur_px * (1 + self.SLIP))
            self._just_closed = self._force_closed = True

        is_entering = False
        if self.pos is None:
            if action > _POS_THRESH:
                self.pos = "LONG"
                self.lev = float(np.clip(abs(action), 0.05, 0.35))
                self.entry_px = fill_px_in * (1.0 + self.SLIP)
                self.balance -= self.balance * self.FEE * self.lev
                self.hold = 0
                is_entering = True
            elif action < -_POS_THRESH:
                self.pos = "SHORT"
                self.lev = float(np.clip(abs(action), 0.05, 0.35))
                self.entry_px = fill_px_in * (1.0 - self.SLIP)
                self.balance -= self.balance * self.FEE * self.lev
                self.hold = 0
                is_entering = True
        else:
            # 청산 신호
            if self.pos == "LONG"  and action < _CLOSE_THRESH:
                self._close_position(fill_px_in)
                self._just_closed = True
            elif self.pos == "SHORT" and action > -_CLOSE_THRESH:
                self._close_position(fill_px_in)
                self._just_closed = True
            else:
                self.hold += 1

        # 미실현 PnL 업데이트
        self.unr = self._unrealized(next_px) if self.pos else 0.0
        curr_port = self.balance * (1.0 + max(self.unr, -1.0)) if self.pos else self.balance

        # ── 보상 ─────────────────────────────────────────────────
        step_delta = (curr_port - prev_port) / max(prev_port, 1e-8) * 50.0
        r1 = float(np.tanh(step_delta))

        r2 = 0.0
        if self.pos and self.unr < -0.005:
            dd_excess = abs(self.unr) - 0.005
            r2 = -0.03 * float(np.clip(dd_excess / 0.020, 0.0, 3.0) ** 2)

        r3 = 0.0
        if self._just_closed:
            if self._force_closed:
                r3 = -0.30
            elif self._last_pnl > 0:
                r3 = 0.15 * min(self._last_pnl / 0.01, 1.0)
            else:
                r3 = -0.08

        r6 = -0.01 * self.lev if is_entering else 0.0

        r7 = 0.0
        if self.pos and self.unr < -0.004 and self.hold > 24:
            r7 = -0.010 * float(np.clip(abs(self.unr) / 0.02, 0.0, 1.0))

        # 스페셜리스트 반대 방향 진입 페널티
        r_pen = 0.0
        if is_entering:
            l_raw = _sf(getattr(row, "meta_long_raw",  0.5))
            s_raw = _sf(getattr(row, "meta_short_raw", 0.5))
            if self.pos == "LONG"  and s_raw > 0.65 and l_raw < 0.35:
                r_pen = -0.005
            elif self.pos == "SHORT" and l_raw > 0.65 and s_raw < 0.35:
                r_pen = -0.005

        raw_r = r1 + r2 + r3 + r6 + r7 + r_pen
        reward = float(np.tanh(raw_r))

        self._step += 1
        done = self._step >= self._end

        # 에피소드 종료 강제 청산
        if done and self.pos:
            ep_px = float(self._close[min(self._step, len(self._close) - 1)])
            realized = self._close_position(ep_px)
            terminal_r = float(np.tanh(realized * 50.0))
            if realized > 0:
                terminal_r += 0.15 * min(realized / 0.01, 1.0)
            else:
                terminal_r -= 0.05
            reward = float(np.tanh(raw_r + terminal_r))

        next_state = _build_state(
            self._rows[min(self._step, len(self._rows) - 1)],
            self._pos_sign, self.hold, self.unr, self.lev,
        )
        info = {
            "pnl_pct": (self.balance - 1.0) * 100.0,
            "wr": self.wins / max(1, self.total_trades),
            "trades": self.total_trades,
        }
        return next_state, reward, done, info


# ─────────────────────────────────────────────────────────────────
# 에이전트
# ─────────────────────────────────────────────────────────────────
def _qhuber(pred_q, target_q, taus, kappa=1.0):
    td   = target_q.unsqueeze(1) - pred_q.unsqueeze(2)
    ahd  = td.abs()
    huber = torch.where(ahd <= kappa, 0.5 * td.pow(2), kappa * (ahd - 0.5 * kappa))
    tau  = taus.view(1, -1, 1)
    w    = (tau - (td.detach() < 0).float()).abs()
    return (w * huber / kappa).mean()


class MetaDSACAgent:
    def __init__(
        self, state_dim=META_STATE_DIM, hidden=256,
        n_q=32, cvar_frac=0.40, gamma=0.99, tau=0.005,
        lr=3e-4, alpha_init=0.05, alpha_min=5e-3,
        device="cpu",
    ):
        self.device   = device
        self.gamma    = gamma
        self.tau      = tau
        self.n_q      = n_q
        self.cvar_k   = max(1, int(n_q * cvar_frac))
        self.alpha_min= alpha_min
        self.target_entropy = -0.5  # heuristic for tanh action

        self.actor   = MetaGaussianActor(state_dim, hidden).to(device)
        self.critic  = _TwinCritic(state_dim, hidden, n_q).to(device)
        self.ctarget = copy.deepcopy(self.critic).to(device)
        self.ctarget.eval()

        self.opt_a = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha = torch.tensor([np.log(alpha_init)], dtype=torch.float32,
                                      device=device, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
        self.taus = torch.linspace(0.5/n_q, 1.0 - 0.5/n_q, n_q, device=device)

        self.memory = _ReplayBuffer(capacity=300_000)
        self._updates = 0

    @property
    def alpha(self) -> float:
        return float(torch.clamp(self.log_alpha.exp(), min=self.alpha_min).item())

    def act(self, state: np.ndarray, deterministic: bool = False) -> float:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a = self.actor.deterministic(s) if deterministic else self.actor.sample(s)[0]
        return float(a.cpu().item())

    def update(self, batch: int = 256) -> dict:
        if len(self.memory) < batch:
            return {}
        s, a, r, ns, d = self.memory.sample(batch)
        S  = torch.FloatTensor(s).to(self.device)
        A  = torch.FloatTensor(a).unsqueeze(1).to(self.device)
        R  = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        NS = torch.FloatTensor(ns).to(self.device)
        D  = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        with torch.no_grad():
            na, nlp = self.actor.sample(NS)
            tq1, tq2 = self.ctarget(NS, na)
            alpha = torch.clamp(self.log_alpha.exp().detach(), min=self.alpha_min)
            min_tq = torch.minimum(tq1, tq2)
            target = R + self.gamma * (1.0 - D) * (min_tq - alpha * nlp)

        q1, q2 = self.critic(S, A)
        closs = _qhuber(q1, target, self.taus) + _qhuber(q2, target, self.taus)
        self.opt_c.zero_grad()
        closs.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_c.step()

        na2, lp2 = self.actor.sample(S)
        q1n, q2n = self.critic(S, na2)
        q_cvar = self._cvar(q1n, q2n)
        alpha2 = torch.clamp(self.log_alpha.exp().detach(), min=self.alpha_min)
        aloss = (alpha2 * lp2 - q_cvar).mean()
        self.opt_a.zero_grad()
        aloss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_a.step()

        alpha3 = torch.clamp(self.log_alpha.exp(), min=self.alpha_min)
        entloss = -(alpha3 * (lp2 + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad()
        entloss.backward()
        self.opt_alpha.step()
        with torch.no_grad():
            self.log_alpha.data.clamp_(min=float(np.log(self.alpha_min)))

        for tp, p in zip(self.ctarget.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        self._updates += 1
        return {"closs": float(closs.item()), "aloss": float(aloss.item()), "alpha": self.alpha}

    def _cvar(self, q1, q2):
        k = self.cvar_k
        q1s, _ = torch.sort(q1, dim=1); q2s, _ = torch.sort(q2, dim=1)
        c1 = q1s[:, :k].mean(1, keepdim=True)
        c2 = q2s[:, :k].mean(1, keepdim=True)
        return torch.minimum(c1, c2)


# ─────────────────────────────────────────────────────────────────
# 검증
# ─────────────────────────────────────────────────────────────────
def _validate(agent: MetaDSACAgent, df_val: "pd.DataFrame", device: str) -> dict:
    env = MetaAgentEnv(df_val, phase="val")
    state = env.reset()
    ep_r = 0.0
    while True:
        a = agent.act(state, deterministic=True)
        state, r, done, info = env.step(a)
        ep_r += r
        if done:
            break
    return {
        "pnl_pct": info["pnl_pct"],
        "wr":      info["wr"],
        "trades":  info["trades"],
        "ep_r":    ep_r,
    }


# ─────────────────────────────────────────────────────────────────
# 학습
# ─────────────────────────────────────────────────────────────────
def train(
    csv_path: str = "data/splits/year_oos/rl_meta_2026.csv",
    episodes: int = 300,
    train_ratio: float = 0.80,
    batch: int = 256,
    update_freq: int = 4,
    min_buffer: int = 2048,
    warmup: int = 4000,
    val_interval: int = 10,
    early_stop: int = 20,
    fresh_start: bool = False,
    device: str = "cpu",
    lr: float = 3e-4,
    alpha_init: float = 0.05,
) -> None:
    import pandas as pd

    if not os.path.exists(csv_path):
        log.error("CSV 없음: %s  →  먼저 generate_specialist_inference.py 를 실행하세요.", csv_path)
        return

    df = pd.read_csv(csv_path)
    # meta_ 컬럼 확인
    missing = [c for c in ("meta_primary_raw", "meta_long_logit", "meta_short_logit") if c not in df.columns]
    if missing:
        log.error("필수 meta 컬럼 없음: %s  →  generate_specialist_inference.py 를 먼저 실행하세요.", missing)
        return

    log.info("CSV 로드: %s  rows=%d  cols=%d", csv_path, len(df), len(df.columns))
    split = int(len(df) * train_ratio)
    df_tr, df_vl = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
    log.info("Train=%d  Val=%d", len(df_tr), len(df_vl))

    dev = _resolve_device(device)
    log.info("Device=%s", dev)

    agent = MetaDSACAgent(device=dev, lr=lr, alpha_init=alpha_init)
    env   = MetaAgentEnv(df_tr, phase="train")

    start_ep = 1
    best_pnl = -1e9
    bad_count = 0
    global_step = 0

    os.makedirs("data/ensemble/ckpt", exist_ok=True)
    if (not fresh_start) and os.path.exists(_CKPT_PATH):
        try:
            ck = torch.load(_CKPT_PATH, map_location=dev, weights_only=False)
            agent.actor.load_state_dict(ck["actor"])
            agent.critic.load_state_dict(ck["critic"])
            agent.ctarget.load_state_dict(ck["ctarget"])
            agent.log_alpha.data.copy_(ck["log_alpha"])
            agent.opt_a.load_state_dict(ck["opt_a"])
            agent.opt_c.load_state_dict(ck["opt_c"])
            agent.opt_alpha.load_state_dict(ck["opt_alpha"])
            global_step = int(ck.get("global_step", 0))
            best_pnl    = float(ck.get("best_pnl", -1e9))
            bad_count   = int(ck.get("bad_count", 0))
            start_ep    = int(ck.get("epoch", 0)) + 1
            log.info("♻️ 복원: ep=%d  global_step=%d  best_pnl=%.2f%%", start_ep - 1, global_step, best_pnl)
        except Exception as e:
            log.warning("체크포인트 로드 실패, 새로 시작: %s", e)

    for ep in range(start_ep, episodes + 1):
        state = env.reset()
        ep_r = ep_trades = 0
        done = False

        while not done:
            # 워밍업 중 랜덤 액션
            if global_step < warmup:
                a = float(np.random.uniform(-1.0, 1.0))
            else:
                a = agent.act(state)
            next_s, r, done, info = env.step(a)
            agent.memory.push(state, a, r, next_s, float(done))
            state = next_s
            ep_r += r
            global_step += 1

            if global_step >= min_buffer and global_step % update_freq == 0:
                agent.update(batch)

        ep_trades = info["trades"]
        log.info(
            "ep=%d/%d  pnl=%.2f%%  wr=%.0f%%  trades=%d  ep_r=%.2f  buf=%d",
            ep, episodes, info["pnl_pct"], info["wr"] * 100, ep_trades, ep_r, len(agent.memory),
        )

        if ep % val_interval == 0:
            vm = _validate(agent, df_vl, dev)
            log.info(
                "  [VAL] pnl=%.2f%%  wr=%.0f%%  trades=%d",
                vm["pnl_pct"], vm["wr"] * 100, vm["trades"],
            )
            if vm["pnl_pct"] > best_pnl:
                best_pnl = vm["pnl_pct"]
                bad_count = 0
                torch.save({
                    "actor":    agent.actor.state_dict(),
                    "state_dim": META_STATE_DIM,
                    "version": "meta_v1",
                }, _BEST_PATH)
                log.info("  ✅ best 저장 (pnl=%.2f%%)", best_pnl)
            else:
                bad_count += 1
                if bad_count >= early_stop:
                    log.info("  조기 종료 (bad_count=%d)", bad_count)
                    break

        # 체크포인트 (매 에피소드)
        torch.save({
            "actor":      agent.actor.state_dict(),
            "critic":     agent.critic.state_dict(),
            "ctarget":    agent.ctarget.state_dict(),
            "log_alpha":  agent.log_alpha.data,
            "opt_a":      agent.opt_a.state_dict(),
            "opt_c":      agent.opt_c.state_dict(),
            "opt_alpha":  agent.opt_alpha.state_dict(),
            "global_step": global_step,
            "best_pnl":   best_pnl,
            "bad_count":  bad_count,
            "epoch":      ep,
            "state_dim":  META_STATE_DIM,
            "version":    "meta_v1",
        }, _CKPT_PATH)

    log.info("학습 완료. best pnl=%.2f%%  모델: %s", best_pnl, _BEST_PATH)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="메타-RL 에이전트 학습")
    ap.add_argument("--csv",          default="data/splits/year_oos/rl_meta_2026.csv")
    ap.add_argument("--episodes",     type=int,   default=300)
    ap.add_argument("--train-ratio",  type=float, default=0.80)
    ap.add_argument("--batch",        type=int,   default=256)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--alpha-init",   type=float, default=0.05)
    ap.add_argument("--val-interval", type=int,   default=10)
    ap.add_argument("--early-stop",   type=int,   default=20)
    ap.add_argument("--device",       default="auto")
    ap.add_argument("--fresh-start",  action="store_true")
    args = ap.parse_args()

    train(
        csv_path=args.csv,
        episodes=args.episodes,
        train_ratio=args.train_ratio,
        batch=args.batch,
        lr=args.lr,
        alpha_init=args.alpha_init,
        val_interval=args.val_interval,
        early_stop=args.early_stop,
        device=args.device,
        fresh_start=args.fresh_start,
    )


if __name__ == "__main__":
    main()
