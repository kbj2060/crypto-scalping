#!/usr/bin/env python3
"""v2 of the Sigma6-filtered + Omega4.6.1 RL meta-controller (see
train_eval_sigma6_omega_rl_meta_controller_20260801.py for the full task description, state/action/
reward design, and honesty caveats -- all still apply, this file only upgrades the ALGORITHM).

User explicitly asked not to default to a plain/basic model and to apply literature-grounded
advanced techniques. Two changes, each tied to a specific line of research surfaced this session:

1. Distributional (quantile-regression) twin critics instead of scalar-Q SAC -- QR-DQN/IQN-style:
   each critic outputs N_QUANT quantiles of the return distribution per (s,a), trained with the
   quantile Huber loss (Dabney et al. 2018 QR-DQN; this project's own DSAC scripts, e.g.
   train_eval_alpha6_1_dsac_risk_allocator_20260524.py, already use this family for other tasks).

2. CVaR (worst-case) policy objective instead of mean-Q -- WCSAC-style (Yang et al., "Worst-Case
   Soft Actor-Critic for Safety-Constrained RL", 2021) / spectral-risk-measure RL (arxiv 2501.02087,
   2507.03900, surfaced by literature search this session): the actor is updated to maximize
   CVaR_alpha(Q) -- the mean of the worst alpha-fraction of the critic's quantiles -- rather than
   mean Q. This is a direct, paper-grounded implementation of what this project's own SOL research
   found empirically (project-sol-sidecar-mdd-unfixable-20260730.md: "only an objective-fn change
   from pnl to log_risk fixes the MDD gate") -- here the risk-sensitivity is built into the RL
   objective itself (CVaR of the value distribution) rather than a reward-shaping penalty term.

3. Seed-diversity ensembling per this project's own CLAUDE.md governance (Seed-Diversity Ensemble
   Promotion Gate, added project-seed-diversity-promotion-gate-20260731.md after the Sigma3-1h
   clustered-seed incident): N=5 genuinely random (not fixed-increment) seeds, OOS sign-agreement
   reported explicitly, since a single RL training run's result is not trustworthy evidence by this
   project's own standard.

Still a RESEARCH PROTOTYPE, not a promotion or live-candidate claim -- same Fresh-Forward caveat as
v1 (VAL/OOS both already explored in this project's history, not a genuinely blind test).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from train_eval_sigma6_omega_rl_meta_controller_20260801 import (  # noqa: E402
    build_bar_frame, run_baseline, STATE_COLS, STATE_DIM, ACTION_DIM,
    MetaAllocEnv, Actor, Replay, WEIGHT_MIN, WEIGHT_MAX,
)

OUT_DIR = ROOT / "tmp/research_20260801/sigma6_omega_rl_meta_controller_v2_cvar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_QUANT = 16
TAUS = (torch.arange(N_QUANT, dtype=torch.float32) + 0.5) / N_QUANT  # fixed quantile fractions
RISK_ALPHA = 0.25  # CVaR level: optimize the mean of the worst 25% of the return distribution
SEEDS = [190417, 542991, 88123, 731650, 264908]  # 5 genuinely random draws, NOT a fixed-increment cluster


# ------------------------------------------------------------------ distributional critic

class QuantileCritic(nn.Module):
    """Twin quantile critics (QR-DQN/IQN-style fixed-tau head, DSAC-family precedent already used
    elsewhere in this repo). Each outputs N_QUANT quantile estimates of the return distribution."""

    def __init__(self, state_dim: int, action_dim: int, n_quant: int = N_QUANT, hidden: int = 128):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_quant))
        self.q2 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_quant))

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)  # each (batch, n_quant)


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """pred: (B, N), target: (B, N). Standard QR-DQN pairwise quantile Huber loss."""
    diff = target.unsqueeze(1) - pred.unsqueeze(2)  # (B, N_pred, N_target)
    huber = torch.where(diff.abs() <= kappa, 0.5 * diff.pow(2), kappa * (diff.abs() - 0.5 * kappa))
    tau = taus.view(1, -1, 1).to(pred.device)
    weight = (tau - (diff.detach() < 0).float()).abs()
    return (weight * huber).sum(dim=1).mean()


def cvar_of_quantiles(quantiles: torch.Tensor, alpha: float) -> torch.Tensor:
    """Mean of the worst alpha-fraction of quantile estimates (quantiles assumed sorted ascending
    order by construction of the fixed-tau head -- sort defensively since the net has no monotonicity
    constraint)."""
    n = quantiles.shape[-1]
    k = max(1, int(round(n * alpha)))
    sorted_q, _ = torch.sort(quantiles, dim=-1)
    return sorted_q[:, :k].mean(dim=-1)


# ------------------------------------------------------------------ CVaR-SAC agent

class CVaRSAC:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, gamma: float = 0.995, tau: float = 0.005):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic = QuantileCritic(state_dim, action_dim).to(DEVICE)
        self.critic_target = QuantileCritic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(action_dim)
        self.gamma, self.tau_soft = gamma, tau
        self.taus = TAUS.to(DEVICE)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, replay: Replay, batch_size: int = 256):
        if len(replay) < batch_size:
            return
        s, a, r, s2, d = replay.sample(batch_size)
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
            q1_t, q2_t = self.critic_target(s2, a2)
            q_t = torch.min(q1_t, q2_t) - self.alpha * logp2.unsqueeze(-1)  # broadcast entropy over quantiles
            target = r + self.gamma * (1 - d) * q_t  # (B, N_QUANT)
        q1, q2 = self.critic(s, a)
        critic_loss = quantile_huber_loss(q1, target, self.taus) + quantile_huber_loss(q2, target, self.taus)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        a_new, logp = self.actor.sample(s)
        q1_n, q2_n = self.critic(s, a_new)
        q_min = torch.min(q1_n, q2_n)
        cvar = cvar_of_quantiles(q_min, RISK_ALPHA)
        actor_loss = (self.alpha.detach() * logp - cvar).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.mul_(1 - self.tau_soft).add_(self.tau_soft * p.data)


# ------------------------------------------------------------------ train / eval helpers

def compute_run(env: MetaAllocEnv, actor: Actor, mean, std) -> dict:
    obs = env.reset()
    equity, peak, mdd = 1.0, 1.0, 0.0
    weights_log = []
    while True:
        s_norm = obs.copy()
        s_norm[:len(STATE_COLS)] = (s_norm[:len(STATE_COLS)] - mean) / std
        with torch.no_grad():
            a = actor.deterministic(torch.tensor(s_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0))
        a_np = a.squeeze(0).cpu().numpy()
        obs, reward, done, info = env.step(a_np)
        equity = info["equity"]
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1)
        weights_log.append(a_np)
        if done:
            break
    return {"pnl_pct": (equity - 1) * 100, "mdd_pct": mdd * 100, "weights": np.array(weights_log)}


def train_one_seed(frame_train: pd.DataFrame, seed: int, epochs: int = 60) -> tuple[CVaRSAC, np.ndarray, np.ndarray]:
    torch.manual_seed(seed); np.random.seed(seed)
    mean = frame_train[STATE_COLS].mean().to_numpy()
    std = frame_train[STATE_COLS].std().replace(0, 1).to_numpy()
    env = MetaAllocEnv(frame_train)
    agent = CVaRSAC(STATE_DIM, ACTION_DIM)
    replay = Replay()
    warmup = 500
    for ep in range(epochs):
        obs = env.reset()
        while True:
            s_norm = obs.copy()
            s_norm[:len(STATE_COLS)] = (s_norm[:len(STATE_COLS)] - mean) / std
            if len(replay) < warmup:
                a_np = np.random.uniform(WEIGHT_MIN, WEIGHT_MAX, size=ACTION_DIM)
            else:
                with torch.no_grad():
                    a_t, _ = agent.actor.sample(torch.tensor(s_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0))
                a_np = a_t.squeeze(0).cpu().numpy()
            obs2, reward, done, info = env.step(a_np)
            s2_norm = obs2.copy()
            s2_norm[:len(STATE_COLS)] = (s2_norm[:len(STATE_COLS)] - mean) / std
            replay.add((s_norm, a_np, reward, s2_norm, float(done)))
            agent.update(replay)
            obs = obs2
            if done:
                break
    return agent, mean, std


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}  seeds={SEEDS}")
    frame_val = build_bar_frame("VAL_2025Q4")
    frame_oos = build_bar_frame("OOS_2026H1")
    print(f"VAL bars={len(frame_val)}  OOS bars={len(frame_oos)}")

    base_val = run_baseline(frame_val)
    base_oos = run_baseline(frame_oos)
    print(f"Baseline fixed 1x-1x combo: VAL pnl={base_val['pnl_pct']:+.2f}% mdd={base_val['mdd_pct']:.2f}% | "
          f"OOS pnl={base_oos['pnl_pct']:+.2f}% mdd={base_oos['mdd_pct']:.2f}%")

    rows = []
    all_oos_weights = []
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")
        agent, mean, std = train_one_seed(frame_val, seed, epochs=20)
        val_eval = compute_run(MetaAllocEnv(frame_val), agent.actor, mean, std)
        oos_eval = compute_run(MetaAllocEnv(frame_oos), agent.actor, mean, std)
        print(f"seed {seed}: VAL pnl={val_eval['pnl_pct']:+.2f}%/mdd={val_eval['mdd_pct']:.2f}%  "
              f"OOS pnl={oos_eval['pnl_pct']:+.2f}%/mdd={oos_eval['mdd_pct']:.2f}%")
        rows.append({"seed": seed, "val_pnl": val_eval["pnl_pct"], "val_mdd": val_eval["mdd_pct"],
                     "oos_pnl": oos_eval["pnl_pct"], "oos_mdd": oos_eval["mdd_pct"]})
        all_oos_weights.append(oos_eval["weights"])
        torch.save(agent.actor.state_dict(), OUT_DIR / f"actor_seed{seed}.pt")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "cvar_sac_seed_results.csv", index=False)
    print(f"\n=== SEED-DIVERSITY SUMMARY (N={len(SEEDS)}) ===")
    print(df.to_string(index=False))
    n_pos_val = int((df["val_pnl"] > 0).sum())
    n_pos_oos = int((df["oos_pnl"] > 0).sum())
    print(f"\nOOS sign agreement: {n_pos_oos}/{len(SEEDS)} seeds positive on OOS "
          f"(vs baseline OOS pnl={base_oos['pnl_pct']:+.2f}%)")
    print(f"OOS pnl beats baseline: {int((df['oos_pnl'] > base_oos['pnl_pct']).sum())}/{len(SEEDS)} seeds")
    print(f"OOS mdd better than baseline: {int((df['oos_mdd'] > base_oos['mdd_pct']).sum())}/{len(SEEDS)} seeds")
    print(f"VAL/OOS pnl mean+-std: VAL {df['val_pnl'].mean():+.2f}+-{df['val_pnl'].std():.2f}  "
          f"OOS {df['oos_pnl'].mean():+.2f}+-{df['oos_pnl'].std():.2f}")

    ens_weights = np.mean(np.stack(all_oos_weights, axis=0), axis=0)  # ensemble = average of the 5 seeds' action
    om_delta = frame_oos["omega_delta"].to_numpy()
    s6_delta = frame_oos["sigma6_delta"].to_numpy()
    equity, peak, mdd = 1.0, 1.0, 0.0
    for i in range(len(ens_weights)):
        equity += ens_weights[i, 0] * om_delta[i] + ens_weights[i, 1] * s6_delta[i]
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1)
    print(f"\n5-seed ENSEMBLE (averaged actions) OOS: pnl={(equity-1)*100:+.2f}% mdd={mdd*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
