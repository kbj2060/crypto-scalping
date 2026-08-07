#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import prepare_features  # noqa: E402
from scripts.train_eval_alpha6_1_catboost_parent_baseline_20260521 import (  # noqa: E402
    DEFAULT_LABEL_DIR,
    DEFAULT_RAW_2025,
    DEFAULT_RAW_2026,
    DEFAULT_SPEC_DIR,
    CatSpec,
    _balanced_weights,
    _binary_proba,
    _build_projection,
    _cat_specs,
    _compose_policy,
    _fit_cat,
    _grid,
    _read_spec,
    _sanitize_feature_cols,
    _score_eval,
)
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days, _fill_price, _json_default, _read  # noqa: E402


MODEL_ID = "alpha6_1_dsac_risk_allocator_20260524"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha6_1_dsac_risk_allocator_20260524"


@dataclass(frozen=True)
class RiskTemplate:
    name: str
    notional: float
    leverage: float
    tp_atr_mult: float
    sl_atr_mult: float
    max_hold: int


RISK_TEMPLATES = [
    RiskTemplate("n010_l1_tp15_sl10_h24", 0.10, 1.0, 1.5, 1.0, 24),
    RiskTemplate("n015_l1_tp20_sl12_h48", 0.15, 1.0, 2.0, 1.2, 48),
    RiskTemplate("n025_l1_tp20_sl12_h48", 0.25, 1.0, 2.0, 1.2, 48),
    RiskTemplate("n025_l2_tp20_sl12_h48", 0.25, 2.0, 2.0, 1.2, 48),
    RiskTemplate("n035_l2_tp25_sl15_h72", 0.35, 2.0, 2.5, 1.5, 72),
    RiskTemplate("n015_l3_tp20_sl12_h48", 0.15, 3.0, 2.0, 1.2, 48),
    RiskTemplate("n020_l3_tp25_sl15_h72", 0.20, 3.0, 2.5, 1.5, 72),
    RiskTemplate("n025_l3_tp30_sl18_h96", 0.25, 3.0, 3.0, 1.8, 96),
]


def _exposure(tpl: RiskTemplate) -> float:
    return float(tpl.notional) * float(tpl.leverage)


class Replay:
    def __init__(self, capacity: int = 200_000) -> None:
        self.capacity = int(capacity)
        self.buf: list[tuple[np.ndarray, int, float, np.ndarray, float]] = []
        self.pos = 0

    def add(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, done: bool) -> None:
        item = (s.copy(), int(a), float(r), ns.copy(), float(done))
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buf, int(n))
        s, a, r, ns, d = zip(*batch)
        rewards = np.asarray(r, dtype=np.float32)
        scale = float(max(rewards.std(), 0.03))
        return (
            torch.tensor(np.asarray(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor((rewards - float(rewards.mean())) / scale, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.asarray(ns), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buf)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, action_dim))
        self.q2 = nn.Sequential(nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, action_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(x), self.q2(x)


class DiscreteSAC:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str,
        *,
        lr: float = 3e-4,
        gamma: float = 0.0,
        tau: float = 0.02,
        alpha_init: float = 0.08,
        alpha_min: float = 0.005,
        alpha_max: float = 0.25,
    ) -> None:
        self.device = torch.device(device)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.target = copy.deepcopy(self.critic).to(self.device)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1e-4)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=1e-4)
        self.log_alpha = torch.tensor([math.log(alpha_init)], dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.target_entropy = 0.50 * math.log(action_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)

    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        x = torch.tensor(state[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs = torch.softmax(self.actor(x), dim=-1)[0]
        if deterministic:
            return int(torch.argmax(probs).item())
        return int(torch.distributions.Categorical(probs=probs).sample().item())

    def update(self, replay: Replay, batch_size: int) -> dict[str, float]:
        if len(replay) < int(batch_size):
            return {}
        s, a, r, ns, d = replay.sample(batch_size)
        s, a, r, ns, d = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device), d.to(self.device)
        alpha = self.log_alpha.exp().clamp(self.alpha_min, self.alpha_max)
        with torch.no_grad():
            npb = torch.softmax(self.actor(ns), dim=-1)
            nlp = torch.log(npb + 1e-8)
            tq1, tq2 = self.target(ns)
            nv = (npb * (torch.minimum(tq1, tq2) - alpha * nlp)).sum(dim=1, keepdim=True)
            target = r + self.gamma * (1.0 - d) * nv
        q1, q2 = self.critic(s)
        q1a = q1.gather(1, a.unsqueeze(1))
        q2a = q2.gather(1, a.unsqueeze(1))
        critic_loss = F.mse_loss(q1a, target) + F.mse_loss(q2a, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        probs = torch.softmax(self.actor(s), dim=-1)
        logp = torch.log(probs + 1e-8)
        pq1, pq2 = self.critic(s)
        actor_loss = (probs * (alpha.detach() * logp - torch.minimum(pq1, pq2))).sum(dim=1).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        entropy = -(probs * logp).sum(dim=1, keepdim=True)
        alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(math.log(self.alpha_min), math.log(self.alpha_max))
            for tp, p in zip(self.target.parameters(), self.critic.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(alpha.item()),
            "entropy": float(entropy.mean().item()),
        }


def _x(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return prepare_features(frame, side_hint=0, close=_close(frame), feature_cols=cols)


def _cat_spec_by_name(name: str) -> CatSpec:
    specs = {s.name: s for s in _cat_specs()}
    if name not in specs:
        raise ValueError(f"unknown cat spec {name}; choices={sorted(specs)}")
    return specs[name]


def _policy_features(frame: pd.DataFrame, p_entry: np.ndarray, p_long: np.ndarray, actions: np.ndarray, feature_frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    p_entry = np.asarray(p_entry, dtype=np.float64)
    p_long = np.asarray(p_long, dtype=np.float64)
    p_short = 1.0 - p_long
    margin = np.abs(p_long - p_short)
    atr = pd.to_numeric(frame.get("atr14_pct", 0.003), errors="coerce").fillna(0.003).to_numpy(dtype=np.float64)
    cols: list[np.ndarray] = [p_entry, p_long, p_short, margin, atr, (actions == 1).astype(float), (actions == 2).astype(float)]
    names = ["p_entry", "p_long", "p_short", "side_margin", "atr14_pct", "is_long_signal", "is_short_signal"]
    extra_names = [
        c
        for c in feature_frame.columns
        if any(tok in c.lower() for tok in ("regime", "funding", "oi_", "rsi", "atr", "whipsaw", "instability", "vol", "trend"))
    ][:80]
    for col in extra_names:
        vals = pd.to_numeric(feature_frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy(dtype=np.float64)
        cols.append(vals)
        names.append(col)
    return np.vstack(cols).T.astype(np.float32), names


def _tp_sl_from_template(frame: pd.DataFrame, template: RiskTemplate) -> tuple[np.ndarray, np.ndarray]:
    atr = pd.to_numeric(frame.get("atr14_pct", 0.003), errors="coerce").fillna(0.003).to_numpy(dtype=np.float64)
    fallback_tp = pd.to_numeric(frame.get("label_tp_pct", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    fallback_sl = pd.to_numeric(frame.get("label_sl_pct", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    tp = np.clip(np.maximum(atr * float(template.tp_atr_mult), fallback_tp * 0.5), 5e-4, 0.05)
    sl = np.clip(np.maximum(atr * float(template.sl_atr_mult), fallback_sl * 0.5), 5e-4, 0.05)
    return tp, sl


def _path_reward(frame: pd.DataFrame, idx: int, side_cls: int, tpl: RiskTemplate, *, fee: float, slip: float) -> float:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    tp, sl = _tp_sl_from_template(frame, tpl)
    side = 1 if int(side_cls) == 1 else -1
    exposure = _exposure(tpl)
    entry_i = min(int(idx) + 1, len(frame) - 1)
    entry = max(float(close[entry_i]), 1e-12)
    end = min(entry_i + int(tpl.max_hold), len(frame) - 1)
    raw = 0.0
    mae = 0.0
    for j in range(entry_i + 1, end + 1):
        if side > 0:
            fav = float(high[j] / entry - 1.0)
            adv = float(low[j] / entry - 1.0)
        else:
            fav = float(entry / max(low[j], 1e-12) - 1.0)
            adv = float(entry / max(high[j], 1e-12) - 1.0)
        mae = max(mae, max(0.0, -adv * exposure))
        if adv <= -float(sl[idx]):
            raw = -float(sl[idx])
            break
        if fav >= float(tp[idx]):
            raw = float(tp[idx])
            break
    else:
        px = max(float(close[end]), 1e-12)
        raw = (px - entry) / entry if side > 0 else (entry - px) / entry
    pnl = raw * exposure - 2.0 * (float(fee) + float(slip)) * exposure
    leverage_penalty = 0.00003 * max(0.0, float(tpl.leverage) - 1.0) * float(tpl.notional)
    return float(pnl - 0.60 * mae - 0.00005 * (float(tpl.max_hold) / 24.0) - 0.00020 * exposure - leverage_penalty)


class RiskAllocatorEnv:
    def __init__(
        self,
        frame: pd.DataFrame,
        states: np.ndarray,
        actions: np.ndarray,
        candidate_idx: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        fee: float,
        slip: float,
    ) -> None:
        self.frame = frame
        self.states = states.astype(np.float32)
        self.actions = actions.astype(np.int64)
        self.candidate_idx = candidate_idx.astype(np.int64)
        self.mean = mean.astype(np.float32)
        self.std = np.where(std <= 1e-6, 1.0, std).astype(np.float32)
        self.fee = float(fee)
        self.slip = float(slip)
        self.ptr = 0

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    def reset(self) -> np.ndarray:
        self.ptr = 0
        return self._state()

    def _state(self) -> np.ndarray:
        idx = int(self.candidate_idx[min(self.ptr, len(self.candidate_idx) - 1)])
        return ((self.states[idx] - self.mean) / self.std).astype(np.float32)

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        idx = int(self.candidate_idx[self.ptr])
        reward = _path_reward(self.frame, idx, int(self.actions[idx]), RISK_TEMPLATES[int(action_id)], fee=self.fee, slip=self.slip)
        self.ptr += 1
        done = self.ptr >= len(self.candidate_idx)
        return (self._state() if not done else np.zeros(self.state_dim, dtype=np.float32)), reward, done, {"idx": idx}


def _train_dsac(env: RiskAllocatorEnv, *, episodes: int, warmup: int, batch_size: int, device: str, seed: int) -> tuple[DiscreteSAC, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = DiscreteSAC(env.state_dim, len(RISK_TEMPLATES), device)
    replay = Replay()
    step = 0
    last: dict[str, Any] = {}
    for ep in range(int(episodes)):
        s = env.reset()
        rewards: list[float] = []
        while True:
            a = int(np.random.randint(len(RISK_TEMPLATES))) if step < int(warmup) else agent.act(s, deterministic=False)
            ns, r, done, _ = env.step(a)
            replay.add(s, a, r, ns, done)
            rewards.append(float(r))
            s = ns
            step += 1
            if step >= int(warmup):
                last = agent.update(replay, int(batch_size)) or last
            if done:
                break
        print(f"[dsac-risk] episode={ep+1}/{episodes} reward_mean={np.mean(rewards):.6f} reward_sum={np.sum(rewards):.6f} update={last}", flush=True)
    return agent, last


def _template_ids(agent: DiscreteSAC, states: np.ndarray, candidate_idx: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = np.zeros(len(states), dtype=np.int64)
    for idx in candidate_idx:
        state = ((states[int(idx)] - mean) / np.where(std <= 1e-6, 1.0, std)).astype(np.float32)
        out[int(idx)] = agent.act(state, deterministic=True)
    return out


def _variable_risk_backtest(
    frame: pd.DataFrame,
    actions: np.ndarray,
    template_ids: np.ndarray,
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    tp_by_tpl, sl_by_tpl = zip(*[_tp_sl_from_template(frame, t) for t in RISK_TEMPLATES])
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_i = -1
    entry_equity = 1.0
    tpl = RISK_TEMPLATES[0]
    tpl_id = 0
    trades = wins = long_entries = short_entries = 0
    exits: dict[str, int] = {}
    tpl_counts: dict[str, int] = {}

    def equity(i: int) -> float:
        if side == 0:
            return cash
        raw = (close[i] - entry) / max(entry, 1e-12) if side > 0 else (entry - close[i]) / max(entry, 1e-12)
        return cash * (1.0 + raw * _exposure(tpl))

    def exit_pos(i: int, reason: str, fill_px: float | None = None) -> None:
        nonlocal cash, side, entry, entry_i, trades, wins
        if fill_px is None:
            fill_px = _fill_price(frame, min(i + 1, len(frame) - 1), side, float(slip), entry=False)
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        exposure = _exposure(tpl)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * float(fee) * exposure
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        side = 0
        entry = 0.0
        entry_i = -1

    for i in range(len(frame) - 2):
        desired = int(actions[i])
        if side != 0:
            hold = i - entry_i
            tp = float(tp_by_tpl[tpl_id][entry_i])
            sl = float(sl_by_tpl[tpl_id][entry_i])
            if side > 0:
                tp_hit = high[i] >= entry * (1.0 + tp)
                sl_hit = low[i] <= entry * (1.0 - sl)
                if tp_hit and sl_hit:
                    exit_pos(i, "ambiguous_bar_sl_first", entry * (1.0 - sl) * (1.0 - float(slip)))
                elif tp_hit:
                    exit_pos(i, "tp", entry * (1.0 + tp) * (1.0 - float(slip)))
                elif sl_hit:
                    exit_pos(i, "sl", entry * (1.0 - sl) * (1.0 - float(slip)))
            else:
                tp_hit = low[i] <= entry * (1.0 - tp)
                sl_hit = high[i] >= entry * (1.0 + sl)
                if tp_hit and sl_hit:
                    exit_pos(i, "ambiguous_bar_sl_first", entry * (1.0 + sl) * (1.0 + float(slip)))
                elif tp_hit:
                    exit_pos(i, "tp", entry * (1.0 - tp) * (1.0 + float(slip)))
                elif sl_hit:
                    exit_pos(i, "sl", entry * (1.0 + sl) * (1.0 + float(slip)))
            if side != 0 and hold >= int(tpl.max_hold):
                exit_pos(i, "max_hold")
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side == 0 and desired != 0:
            tpl_id = int(template_ids[i])
            tpl = RISK_TEMPLATES[tpl_id]
            side = 1 if desired == 1 else -1
            entry_i = int(i)
            entry = _fill_price(frame, min(i + 1, len(frame) - 1), side, float(slip), entry=True)
            entry_equity = cash
            cash -= cash * float(fee) * _exposure(tpl)
            long_entries += int(side > 0)
            short_entries += int(side < 0)
            tpl_counts[tpl.name] = tpl_counts.get(tpl.name, 0) + 1
    if side != 0:
        exit_pos(len(frame) - 2, "end")
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "calmar": float(((cash - 1.0) * 100.0) / max(abs(mdd * 100.0), 1e-12)),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(frame)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exits": exits,
        "template_counts": tpl_counts,
    }


def _eval_variable_costs(frame: pd.DataFrame, actions: np.ndarray, template_ids: np.ndarray, *, fee: float, slip: float) -> dict[str, Any]:
    return {f"cost{m}": _variable_risk_backtest(frame, actions, template_ids, fee=fee * m, slip=slip * m) for m in (1, 2, 3)}


def _fit_precision_gate(
    states: np.ndarray,
    actions: np.ndarray,
    success_labels: np.ndarray,
    *,
    seed: int,
) -> tuple[lgb.LGBMClassifier, dict[str, Any]]:
    candidate_idx = np.flatnonzero(actions != 0)
    if len(candidate_idx) < 200:
        raise RuntimeError(f"too few candidates for precision gate: {len(candidate_idx)}")
    y = success_labels[candidate_idx].astype(np.int64)
    if int(y.sum()) < 10 or int((1 - y).sum()) < 10:
        raise RuntimeError(f"precision gate target is degenerate: positives={int(y.sum())} negatives={int((1 - y).sum())}")
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=160,
        learning_rate=0.035,
        max_depth=2,
        num_leaves=4,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=4.0,
        reg_lambda=10.0,
        class_weight="balanced",
        random_state=int(seed),
        verbosity=-1,
    )
    model.fit(states[candidate_idx], y)
    prob = model.predict_proba(states[candidate_idx])[:, 1]
    metrics = {
        "candidates": int(len(candidate_idx)),
        "label_precision": float(y.mean()),
        "prob_mean": float(np.mean(prob)),
        "prob_p50": float(np.quantile(prob, 0.50)),
        "prob_p90": float(np.quantile(prob, 0.90)),
    }
    return model, metrics


def _fixed_policy_pnl_label(
    frame: pd.DataFrame,
    actions: np.ndarray,
    tp: np.ndarray,
    sl: np.ndarray,
    *,
    fee: float,
    slip: float,
    exposure: float,
    max_hold: int,
) -> np.ndarray:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    out = np.zeros(len(frame), dtype=np.int64)
    for idx in np.flatnonzero(actions != 0):
        entry_i = min(int(idx) + 1, len(frame) - 1)
        entry = max(float(close[entry_i]), 1e-12)
        side = 1 if int(actions[idx]) == 1 else -1
        end = min(entry_i + int(max_hold), len(frame) - 1)
        raw = 0.0
        for j in range(entry_i + 1, end + 1):
            if side > 0:
                fav = float(high[j] / entry - 1.0)
                adv = float(low[j] / entry - 1.0)
            else:
                fav = float(entry / max(low[j], 1e-12) - 1.0)
                adv = float(entry / max(high[j], 1e-12) - 1.0)
            if adv <= -float(sl[idx]):
                raw = -float(sl[idx])
                break
            if fav >= float(tp[idx]):
                raw = float(tp[idx])
                break
        else:
            px = max(float(close[end]), 1e-12)
            raw = (px - entry) / entry if side > 0 else (entry - px) / entry
        pnl = raw * float(exposure) - 2.0 * (float(fee) + float(slip)) * float(exposure)
        out[int(idx)] = int(pnl > 0.0)
    return out


def _gate_prob(model: lgb.LGBMClassifier, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    out = np.zeros(len(actions), dtype=np.float64)
    idx = np.flatnonzero(actions != 0)
    if len(idx) > 0:
        out[idx] = model.predict_proba(states[idx])[:, 1]
    return out


def _filter_actions(actions: np.ndarray, gate_prob: np.ndarray, threshold: float) -> np.ndarray:
    out = actions.copy()
    out[(out != 0) & (gate_prob < float(threshold))] = 0
    return out


def _select_precision_threshold(
    frame: pd.DataFrame,
    actions: np.ndarray,
    labels: np.ndarray,
    tp: np.ndarray,
    sl: np.ndarray,
    gate_prob: np.ndarray,
    thresholds: list[float],
    *,
    fee: float,
    slip: float,
    exposure: float,
    max_hold: int,
    min_trades: int,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        filtered = _filter_actions(actions, gate_prob, float(threshold))
        ev = _score_eval(
            frame,
            filtered,
            labels,
            tp,
            sl,
            fee=float(fee),
            slip=float(slip),
            exposure=float(exposure),
            max_hold=int(max_hold),
        )
        cand = {
            "threshold": float(threshold),
            "kept_signals": int(np.sum(filtered != 0)),
            "dropped_signals": int(np.sum((actions != 0) & (filtered == 0))),
            "validation": ev,
        }
        trades = int(cand["validation"].get("backtest", {}).get("cost1", {}).get("trades", 0))
        score = float(cand["validation"]["score"])
        if trades < int(min_trades):
            score -= 1000.0
        cand["_selection_score"] = float(score)
        if best is None or float(cand["_selection_score"]) > float(best["_selection_score"]):
            best = cand
    assert best is not None
    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attach DSAC notional/leverage/TP/SL allocator to Alpha6.1 fixed entry/direction parent.")
    p.add_argument("--variant", default="current_tail111")
    p.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    p.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    p.add_argument("--train-file", default="alpha5_24_entry_rebalanced_train.parquet")
    p.add_argument("--val-file", default="alpha5_24_entry_rebalanced_val.parquet")
    p.add_argument("--oos-file", default="alpha5_24_entry_rebalanced_oos.parquet")
    p.add_argument("--raw-2025-csv", type=Path, default=DEFAULT_RAW_2025)
    p.add_argument("--raw-2026-csv", type=Path, default=DEFAULT_RAW_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--entry-spec", default="regularized")
    p.add_argument("--direction-spec", default="regularized")
    p.add_argument("--task-type", default="CPU")
    p.add_argument("--devices", default="0")
    p.add_argument("--entry-thresholds", default="0.50,0.55,0.60,0.65")
    p.add_argument("--side-thresholds", default="0.55,0.60,0.65")
    p.add_argument("--margin-thresholds", default="0.03,0.05,0.08")
    p.add_argument("--tp-atr-mults", default="1.5,2.0,2.5")
    p.add_argument("--sl-atr-mults", default="1.0,1.2,1.5")
    p.add_argument("--guardrails", default="none,block_whipsaw")
    p.add_argument("--baseline-max-hold-bars", type=int, default=96)
    p.add_argument("--baseline-unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-train-candidates", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=62124)
    p.add_argument("--precision-gate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--precision-target", choices=["label_action", "path_pnl"], default="path_pnl")
    p.add_argument("--precision-thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    p.add_argument("--precision-min-trades", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit = _verify_state24_sticky090_inputs(_read(args.raw_2025_csv), _read(args.raw_2026_csv), args.manifest, args.clean4_report)
    train_df = pd.read_parquet(args.label_dir / str(args.train_file))
    val_df = pd.read_parquet(args.label_dir / str(args.val_file))
    oos_df = pd.read_parquet(args.label_dir / str(args.oos_file))
    spec = _read_spec(args.spec_dir, str(args.variant))
    feature_cols, leak_audit = _sanitize_feature_cols(train_df, list(spec.get("features", [])))
    x_train_all, (x_val_all, x_oos_all), projection_meta, projection = _build_projection(
        train_df,
        [val_df, oos_df],
        feature_cols,
        enable_pca=bool(spec.get("extra_pca_enable", False)),
        pca_components=int(spec.get("extra_pca_components", 0) or 0),
    )

    entry_mask = pd.to_numeric(train_df["entry_train_keep"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    dir_mask = pd.to_numeric(train_df["direction_train_keep"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    entry_spec = _cat_spec_by_name(str(args.entry_spec))
    direction_spec = _cat_spec_by_name(str(args.direction_spec))
    print(f"[fit] variant={args.variant} entry={entry_spec.name} direction={direction_spec.name} features={x_train_all.shape[1]}", flush=True)
    entry_model = _fit_cat(
        x_train_all.loc[entry_mask].reset_index(drop=True),
        pd.to_numeric(train_df.loc[entry_mask, "entry_label"], errors="coerce").fillna(0).to_numpy(np.int64),
        np.clip(pd.to_numeric(train_df.loc[entry_mask, "entry_sample_weight"], errors="coerce").fillna(0).to_numpy(np.float64), 1e-4, None)
        * _balanced_weights(pd.to_numeric(train_df.loc[entry_mask, "entry_label"], errors="coerce").fillna(0).to_numpy(np.int64)),
        entry_spec,
        int(args.seed + 11),
        task_type=str(args.task_type),
        devices=str(args.devices),
    )
    direction_model = _fit_cat(
        x_train_all.loc[dir_mask].reset_index(drop=True),
        (pd.to_numeric(train_df.loc[dir_mask, "direction_label"], errors="coerce").fillna(0).to_numpy(np.int64) == 1).astype(np.int64),
        np.clip(pd.to_numeric(train_df.loc[dir_mask, "direction_sample_weight"], errors="coerce").fillna(0).to_numpy(np.float64), 1e-4, None)
        * _balanced_weights((pd.to_numeric(train_df.loc[dir_mask, "direction_label"], errors="coerce").fillna(0).to_numpy(np.int64) == 1).astype(np.int64)),
        direction_spec,
        int(args.seed + 29),
        task_type=str(args.task_type),
        devices=str(args.devices),
    )
    p_entry_train = _binary_proba(entry_model, x_train_all)
    p_long_train = _binary_proba(direction_model, x_train_all)
    p_entry_val = _binary_proba(entry_model, x_val_all)
    p_long_val = _binary_proba(direction_model, x_val_all)
    p_entry_oos = _binary_proba(entry_model, x_oos_all)
    p_long_oos = _binary_proba(direction_model, x_oos_all)
    y_train = pd.to_numeric(train_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    y_val = pd.to_numeric(val_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    y_oos = pd.to_numeric(oos_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)

    best: dict[str, Any] | None = None
    for entry_th in _grid(args.entry_thresholds):
        for side_th in _grid(args.side_thresholds):
            for margin_th in _grid(args.margin_thresholds):
                for tp_mult in _grid(args.tp_atr_mults):
                    for sl_mult in _grid(args.sl_atr_mults):
                        for guardrail in [x.strip() for x in str(args.guardrails).split(",") if x.strip()]:
                            val_actions, val_tp, val_sl, _ = _compose_policy(
                                val_df,
                                p_entry_val,
                                p_long_val,
                                entry_threshold=entry_th,
                                side_threshold=side_th,
                                margin_threshold=margin_th,
                                tp_atr_mult=tp_mult,
                                sl_atr_mult=sl_mult,
                                guardrail=guardrail,
                            )
                            val_eval = _score_eval(
                                val_df,
                                val_actions,
                                y_val,
                                val_tp,
                                val_sl,
                                fee=float(args.fee),
                                slip=float(args.slip),
                                exposure=float(args.baseline_unit_exposure),
                                max_hold=int(args.baseline_max_hold_bars),
                            )
                            cand = {
                                "entry_threshold": float(entry_th),
                                "side_threshold": float(side_th),
                                "margin_threshold": float(margin_th),
                                "tp_atr_mult": float(tp_mult),
                                "sl_atr_mult": float(sl_mult),
                                "guardrail": guardrail,
                                "validation": val_eval,
                            }
                            if best is None or float(cand["validation"]["score"]) > float(best["validation"]["score"]):
                                best = cand
    assert best is not None
    print(f"[baseline-best] {json.dumps(best, ensure_ascii=False, default=_json_default)[:1200]}", flush=True)

    train_actions, train_tp, train_sl, _ = _compose_policy(
        train_df,
        p_entry_train,
        p_long_train,
        entry_threshold=float(best["entry_threshold"]),
        side_threshold=float(best["side_threshold"]),
        margin_threshold=float(best["margin_threshold"]),
        tp_atr_mult=float(best["tp_atr_mult"]),
        sl_atr_mult=float(best["sl_atr_mult"]),
        guardrail=str(best["guardrail"]),
    )
    val_actions, val_tp, val_sl, _ = _compose_policy(
        val_df,
        p_entry_val,
        p_long_val,
        entry_threshold=float(best["entry_threshold"]),
        side_threshold=float(best["side_threshold"]),
        margin_threshold=float(best["margin_threshold"]),
        tp_atr_mult=float(best["tp_atr_mult"]),
        sl_atr_mult=float(best["sl_atr_mult"]),
        guardrail=str(best["guardrail"]),
    )
    oos_actions, oos_tp, oos_sl, _ = _compose_policy(
        oos_df,
        p_entry_oos,
        p_long_oos,
        entry_threshold=float(best["entry_threshold"]),
        side_threshold=float(best["side_threshold"]),
        margin_threshold=float(best["margin_threshold"]),
        tp_atr_mult=float(best["tp_atr_mult"]),
        sl_atr_mult=float(best["sl_atr_mult"]),
        guardrail=str(best["guardrail"]),
    )
    baseline_oos = _score_eval(
        oos_df,
        oos_actions,
        y_oos,
        oos_tp,
        oos_sl,
        fee=float(args.fee),
        slip=float(args.slip),
        exposure=float(args.baseline_unit_exposure),
        max_hold=int(args.baseline_max_hold_bars),
    )
    baseline_before_precision_gate = {"validation": best["validation"], "oos": baseline_oos}

    gate_model: lgb.LGBMClassifier | None = None
    precision_gate_summary: dict[str, Any] = {"enabled": bool(args.precision_gate)}
    if bool(args.precision_gate):
        train_state_for_gate, gate_state_names = _policy_features(train_df, p_entry_train, p_long_train, train_actions, x_train_all)
        val_state_for_gate, _ = _policy_features(val_df, p_entry_val, p_long_val, val_actions, x_val_all)
        oos_state_for_gate, _ = _policy_features(oos_df, p_entry_oos, p_long_oos, oos_actions, x_oos_all)
        gate_labels = (train_actions.astype(np.int64) == y_train.astype(np.int64)).astype(np.int64)
        gate_target = "candidate action equals label_action"
        if str(args.precision_target) == "path_pnl":
            gate_labels = _fixed_policy_pnl_label(
                train_df,
                train_actions,
                train_tp,
                train_sl,
                fee=float(args.fee),
                slip=float(args.slip),
                exposure=float(args.baseline_unit_exposure),
                max_hold=int(args.baseline_max_hold_bars),
            )
            gate_target = "train-only candidate fixed-policy net path PnL > 0"
        gate_model, gate_train_metrics = _fit_precision_gate(train_state_for_gate, train_actions, gate_labels, seed=int(args.seed + 707))
        train_gate_prob = _gate_prob(gate_model, train_state_for_gate, train_actions)
        val_gate_prob = _gate_prob(gate_model, val_state_for_gate, val_actions)
        oos_gate_prob = _gate_prob(gate_model, oos_state_for_gate, oos_actions)
        gate_best = _select_precision_threshold(
            val_df,
            val_actions,
            y_val,
            val_tp,
            val_sl,
            val_gate_prob,
            _grid(args.precision_thresholds),
            fee=float(args.fee),
            slip=float(args.slip),
            exposure=float(args.baseline_unit_exposure),
            max_hold=int(args.baseline_max_hold_bars),
            min_trades=int(args.precision_min_trades),
        )
        gate_th = float(gate_best["threshold"])
        train_actions = _filter_actions(train_actions, train_gate_prob, gate_th)
        val_actions = _filter_actions(val_actions, val_gate_prob, gate_th)
        oos_actions = _filter_actions(oos_actions, oos_gate_prob, gate_th)
        best = copy.deepcopy(best)
        best["precision_gate"] = {k: v for k, v in gate_best.items() if k != "_selection_score"}
        best["validation"] = _score_eval(
            val_df,
            val_actions,
            y_val,
            val_tp,
            val_sl,
            fee=float(args.fee),
            slip=float(args.slip),
            exposure=float(args.baseline_unit_exposure),
            max_hold=int(args.baseline_max_hold_bars),
        )
        baseline_oos = _score_eval(
            oos_df,
            oos_actions,
            y_oos,
            oos_tp,
            oos_sl,
            fee=float(args.fee),
            slip=float(args.slip),
            exposure=float(args.baseline_unit_exposure),
            max_hold=int(args.baseline_max_hold_bars),
        )
        precision_gate_summary = {
            "enabled": True,
            "model": "LightGBM shallow binary correct-signal gate",
            "target": gate_target,
            "target_mode": str(args.precision_target),
            "threshold": gate_th,
            "train_metrics": gate_train_metrics,
            "state_names": gate_state_names,
            "baseline_before_gate": baseline_before_precision_gate,
            "selected_on_validation": {k: v for k, v in gate_best.items() if k != "_selection_score"},
            "kept_signals": {
                "train": int(np.sum(train_actions != 0)),
                "validation": int(np.sum(val_actions != 0)),
                "oos": int(np.sum(oos_actions != 0)),
            },
        }
        print(f"[precision-gate] {json.dumps(precision_gate_summary, ensure_ascii=False, default=_json_default)[:1600]}", flush=True)

    train_state, state_names = _policy_features(train_df, p_entry_train, p_long_train, train_actions, x_train_all)
    val_state, _ = _policy_features(val_df, p_entry_val, p_long_val, val_actions, x_val_all)
    oos_state, _ = _policy_features(oos_df, p_entry_oos, p_long_oos, oos_actions, x_oos_all)
    train_candidates = np.flatnonzero(train_actions != 0)
    train_candidates = train_candidates[train_candidates < len(train_df) - 100]
    if int(args.max_train_candidates) > 0 and len(train_candidates) > int(args.max_train_candidates):
        train_candidates = train_candidates[-int(args.max_train_candidates) :]
    if len(train_candidates) < 100:
        raise RuntimeError(f"too few train candidates for DSAC risk allocator: {len(train_candidates)}")
    mean = train_state[train_candidates].mean(axis=0)
    std = train_state[train_candidates].std(axis=0)
    env = RiskAllocatorEnv(train_df, train_state, train_actions, train_candidates, mean, std, fee=float(args.fee), slip=float(args.slip))
    agent, last_update = _train_dsac(env, episodes=int(args.episodes), warmup=int(args.warmup), batch_size=int(args.batch_size), device=str(args.device), seed=int(args.seed))

    val_candidates = np.flatnonzero(val_actions != 0)
    oos_candidates = np.flatnonzero(oos_actions != 0)
    val_tpl = _template_ids(agent, val_state, val_candidates, mean, std)
    oos_tpl = _template_ids(agent, oos_state, oos_candidates, mean, std)
    dsac_val = _eval_variable_costs(val_df, val_actions, val_tpl, fee=float(args.fee), slip=float(args.slip))
    dsac_oos = _eval_variable_costs(oos_df, oos_actions, oos_tpl, fee=float(args.fee), slip=float(args.slip))

    model_prefix = f"{args.variant}_{entry_spec.name}_{direction_spec.name}"
    joblib.dump(
        {
            "entry_model": entry_model,
            "direction_model": direction_model,
            "projection": projection,
            "feature_cols": feature_cols,
            "projection_meta": projection_meta,
            "baseline_best": best,
            "state_names": state_names,
            "risk_templates": [t.__dict__ for t in RISK_TEMPLATES],
            "mean": mean,
            "std": std,
            "precision_gate_model": gate_model,
            "precision_gate": precision_gate_summary,
        },
        args.out_dir / f"{model_prefix}_alpha6_1_parent_risk_input.joblib",
    )
    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "config": {
                "state_dim": int(env.state_dim),
                "risk_templates": [t.__dict__ for t in RISK_TEMPLATES],
                "mean": mean,
                "std": std,
            },
        },
        args.out_dir / f"{model_prefix}_dsac_risk_allocator.pt",
    )
    summary = {
        "model_id": MODEL_ID,
        "variant": str(args.variant),
        "entry_spec": entry_spec.__dict__,
        "direction_spec": direction_spec.__dict__,
        "baseline_best": best,
        "baseline_oos": baseline_oos,
        "precision_gate": precision_gate_summary,
        "dsac_validation": dsac_val,
        "dsac_oos": dsac_oos,
        "last_update": last_update,
        "train_candidates": int(len(train_candidates)),
        "val_candidates": int(len(val_candidates)),
        "oos_candidates": int(len(oos_candidates)),
        "risk_templates": [t.__dict__ for t in RISK_TEMPLATES],
        "audit": {
            "preprocess_inputs": audit,
            "leak_audit": leak_audit,
            "entry_direction_fixed": True,
            "precision_gate_train_only": bool(args.precision_gate),
            "dsac_controls": "notional,leverage,tp_atr_mult,sl_atr_mult,max_hold only",
            "selection_contract": "baseline entry/direction thresholds selected on validation; DSAC trained on train candidates and evaluated on validation/OOS without changing entry side.",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
