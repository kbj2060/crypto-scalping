"""Stage 3 (reinforcement learning) of docs/experiments/sol_dl_rl_architecture_survey_20260807.json.

Self-contained discrete SAC (twin-Q, auto-entropy) position-target agent on the SOL panel.
Deliberately NOT the ETH-tuned SACTradingEnv (ensemble/rl_continuous_common.py): that env carries
years of ETH-specific reward shaping (kelly bonuses, idle penalties, plateau/adverse-hold
penalties, HMM overlays) that would confound a first-principles SOL test. Here the reward is the
un-shaped mark-to-market account return under the survey's exact cost model, so the agent is
scored on precisely what the other candidates are scored on.

- State:   126 standardized panel features (train stats, same exclusions as the LGBM control)
           + [position(-1/0/1), unrealized pnl on notional, bars-in-position/288]
- Action:  target position in {flat, long, short}; side flip = close + reopen (2 half-turn costs)
- Reward:  position * bar close-to-close return * notional - 5bps * |position change| * notional,
           notional = margin_fraction(0.30) * leverage(3) = 0.90 per the sizing contract
- Splits/costs identical to every other candidate; N=5 contract seeds; VAL-only selection;
  greedy deterministic policy replayed bar-by-bar for evaluation (fresh-forward, no ledger input).

Usage:
  python scripts/train_eval_sol_discrete_sac_rl_20260807.py --stage val
  python scripts/train_eval_sol_discrete_sac_rl_20260807.py --stage oos
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    RAW_LEVEL_COLS, PANEL_PATH, HORIZON_BARS,
    TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

OUT_DIR = ROOT / "tmp/sol_dl_rl_survey_20260807/rl_discrete_sac"
SEEDS = [903174, 42517, 6688211, 15093, 771442]
LGBM_CONTROL_VAL_PNL = -6.903

MARGIN_FRACTION, LEVERAGE = 0.30, 3.0
NOTIONAL = MARGIN_FRACTION * LEVERAGE
HALF_TURN_COST = 0.0005  # 10bps roundtrip on price move, split per side

EPISODE_LEN = 2048
TOTAL_ENV_STEPS = 200_000
WARMUP_STEPS = 5_000
BUFFER_CAP = 300_000
BATCH = 512
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
TARGET_ENTROPY_FRAC = 0.6  # target entropy = frac * log(n_actions)
ACTIONS = np.array([0.0, 1.0, -1.0])  # flat, long, short


def build_data():
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    x = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    ts = panel["timestamp"]
    train_mask = (ts <= TRAIN_END).to_numpy()
    tr_idx = np.flatnonzero(train_mask)
    train_mask[tr_idx[-HORIZON_BARS:]] = False  # symmetry with supervised purge
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()
    mean = np.nanmean(x[train_mask], axis=0)
    std = np.nanstd(x[train_mask], axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_std = np.clip(np.nan_to_num((x - mean) / std, nan=0.0), -10.0, 10.0).astype(np.float32)
    close = panel["close"].to_numpy(dtype=np.float64)
    bar_ret = np.zeros(len(close))
    bar_ret[1:] = close[1:] / close[:-1] - 1.0  # return realized while holding INTO bar t
    return panel, x_std, bar_ret, train_mask, val_mask, oos_mask, feat_cols


def make_state(x_row: np.ndarray, pos: float, upnl: float, bars_held: int) -> np.ndarray:
    return np.concatenate([x_row, np.array([pos, upnl * 10.0, bars_held / 288.0], dtype=np.float32)])


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, cap: int, state_dim: int):
        self.cap = cap
        self.s = np.zeros((cap, state_dim), dtype=np.float32)
        self.a = np.zeros(cap, dtype=np.int64)
        self.r = np.zeros(cap, dtype=np.float32)
        self.s2 = np.zeros((cap, state_dim), dtype=np.float32)
        self.d = np.zeros(cap, dtype=np.float32)
        self.n = 0
        self.ptr = 0

    def push(self, s, a, r, s2, d):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s2[self.ptr] = s2
        self.d[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch, device):
        idx = np.random.randint(0, self.n, size=batch)
        to = lambda arr: torch.from_numpy(arr[idx]).to(device)
        return to(self.s), torch.from_numpy(self.a[idx]).to(device), to(self.r), to(self.s2), to(self.d)


def step_env(pos: float, action_idx: int, bar_ret_next: float):
    """Transition from bar t (deciding) to bar t+1. Reward = holding new position over bar t+1's
    close-to-close return minus cost of the position change, all on account notional."""
    new_pos = ACTIONS[action_idx]
    turn = abs(new_pos - pos)
    reward = float(new_pos * bar_ret_next * NOTIONAL - HALF_TURN_COST * turn * NOTIONAL)
    return new_pos, reward


def train_one_seed(seed: int, x_std, bar_ret, train_rows, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    state_dim = x_std.shape[1] + 3
    n_act = 3
    policy = MLP(state_dim, n_act).to(device)
    q1 = MLP(state_dim, n_act).to(device)
    q2 = MLP(state_dim, n_act).to(device)
    q1_t = MLP(state_dim, n_act).to(device)
    q2_t = MLP(state_dim, n_act).to(device)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())
    opt_pi = torch.optim.Adam(policy.parameters(), lr=LR)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=LR)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    opt_a = torch.optim.Adam([log_alpha], lr=LR)
    target_entropy = TARGET_ENTROPY_FRAC * float(np.log(n_act))

    buf = ReplayBuffer(BUFFER_CAP, state_dim)
    # episodes must fit inside the train region contiguously
    lo, hi = int(train_rows[0]), int(train_rows[-1])
    steps = 0
    while steps < TOTAL_ENV_STEPS:
        start = int(rng.integers(lo, hi - EPISODE_LEN - 1))
        pos, bars_held, upnl = 0.0, 0, 0.0
        s = make_state(x_std[start], pos, upnl, bars_held)
        for t in range(start, start + EPISODE_LEN):
            if steps < WARMUP_STEPS:
                a = int(rng.integers(0, n_act))
            else:
                with torch.no_grad():
                    logits = policy(torch.from_numpy(s).unsqueeze(0).to(device))
                    a = int(torch.distributions.Categorical(logits=logits).sample())
            new_pos, r = step_env(pos, a, bar_ret[t + 1])
            if new_pos != pos:
                bars_held = 0
                upnl = 0.0
            else:
                bars_held += 1
            upnl += float(new_pos * bar_ret[t + 1] * NOTIONAL)
            pos = new_pos
            s2 = make_state(x_std[t + 1], pos, upnl, bars_held)
            done = 1.0 if t == start + EPISODE_LEN - 1 else 0.0
            buf.push(s, a, r, s2, done)
            s = s2
            steps += 1

            if steps >= WARMUP_STEPS and buf.n >= BATCH:
                bs, ba, br, bs2, bd = buf.sample(BATCH, device)
                alpha = log_alpha.exp().detach()
                with torch.no_grad():
                    logits2 = policy(bs2)
                    logp2 = F.log_softmax(logits2, dim=-1)
                    p2 = logp2.exp()
                    qt = torch.min(q1_t(bs2), q2_t(bs2))
                    v2 = (p2 * (qt - alpha * logp2)).sum(dim=-1)
                    target = br + GAMMA * (1.0 - bd) * v2
                q1_pred = q1(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                q2_pred = q2(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                loss_q = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)
                opt_q.zero_grad()
                loss_q.backward()
                opt_q.step()

                logits = policy(bs)
                logp = F.log_softmax(logits, dim=-1)
                p = logp.exp()
                with torch.no_grad():
                    qmin = torch.min(q1(bs), q2(bs))
                loss_pi = (p * (log_alpha.exp().detach() * logp - qmin)).sum(dim=-1).mean()
                opt_pi.zero_grad()
                loss_pi.backward()
                opt_pi.step()

                entropy = -(p * logp).sum(dim=-1).detach()
                loss_a = (log_alpha.exp() * (entropy - target_entropy)).mean()
                opt_a.zero_grad()
                loss_a.backward()
                opt_a.step()

                with torch.no_grad():
                    for tp, sp in zip(q1_t.parameters(), q1.parameters()):
                        tp.mul_(1 - TAU).add_(TAU * sp)
                    for tp, sp in zip(q2_t.parameters(), q2.parameters()):
                        tp.mul_(1 - TAU).add_(TAU * sp)
        print(f"[sac seed={seed}] steps={steps} alpha={float(log_alpha.exp()):.4f}", flush=True)
    return policy


@torch.no_grad()
def greedy_replay(policy, x_std, bar_ret, rows, device):
    """Deterministic bar-by-bar replay of the greedy policy over a contiguous split."""
    policy.eval()
    pos, bars_held, upnl = 0.0, 0, 0.0
    equity = [1.0]
    n_entries = 0
    pos_log = np.zeros(len(rows))
    for k, t in enumerate(rows[:-1]):
        s = make_state(x_std[t], pos, upnl, bars_held)
        logits = policy(torch.from_numpy(s).unsqueeze(0).to(device))
        a = int(logits.argmax(dim=-1))
        new_pos, r = step_env(pos, a, bar_ret[t + 1])
        if new_pos != pos:
            if new_pos != 0.0:
                n_entries += 1
            bars_held, upnl = 0, 0.0
        else:
            bars_held += 1
        upnl += float(new_pos * bar_ret[t + 1] * NOTIONAL)
        pos = new_pos
        pos_log[k] = pos
        equity.append(equity[-1] * (1.0 + r))
    equity = np.array(equity)
    running_max = np.maximum.accumulate(equity)
    return {
        "pnl_pct": float((equity[-1] - 1.0) * 100.0),
        "mdd_pct": float(((equity - running_max) / running_max).min() * 100.0),
        "n_entries": int(n_entries),
        "frac_bars_long": float((pos_log > 0).mean()),
        "frac_bars_short": float((pos_log < 0).mean()),
        "frac_bars_flat": float((pos_log == 0).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "oos"], required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    panel, x_std, bar_ret, train_mask, val_mask, oos_mask, feat_cols = build_data()
    train_rows = np.flatnonzero(train_mask)
    val_rows = np.flatnonzero(val_mask)
    oos_rows = np.flatnonzero(oos_mask)
    state_dim = x_std.shape[1] + 3

    if args.stage == "val":
        per_seed = []
        for seed in SEEDS:
            policy = train_one_seed(seed, x_std, bar_ret, train_rows, device)
            torch.save(policy.state_dict(), OUT_DIR / f"policy_seed{seed}.pt")
            r = greedy_replay(policy, x_std, bar_ret, val_rows, device)
            per_seed.append({"seed": seed, **r})
            print(json.dumps(per_seed[-1]), flush=True)
        pnls = [r["pnl_pct"] for r in per_seed]
        entries = [r["n_entries"] for r in per_seed]
        seed_mean = float(np.mean(pnls))
        earns_oos = bool(seed_mean > 0 and seed_mean > LGBM_CONTROL_VAL_PNL and np.mean(entries) >= 15)
        out = {
            "stage": "val", "algo": "discrete_sac_unshaped", "seeds": SEEDS,
            "seed_mean_pnl_pct": seed_mean, "n_pos_seeds": int(sum(p > 0 for p in pnls)),
            "per_seed": per_seed, "earns_oos_read": earns_oos, "lgbm_control_val_pnl": LGBM_CONTROL_VAL_PNL,
        }
        (OUT_DIR / "val_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({k: out[k] for k in ("seed_mean_pnl_pct", "n_pos_seeds", "earns_oos_read")}, indent=2))
    else:
        prior = json.loads((OUT_DIR / "val_results.json").read_text())
        if not prior.get("earns_oos_read"):
            print(json.dumps({"algo": "discrete_sac_unshaped", "oos": "REFUSED -- candidate did not pass VAL gate"}))
            return 1
        per_seed = []
        for seed in SEEDS:
            policy = MLP(state_dim, 3).to(device)
            policy.load_state_dict(torch.load(OUT_DIR / f"policy_seed{seed}.pt", map_location=device))
            r = greedy_replay(policy, x_std, bar_ret, oos_rows, device)
            per_seed.append({"seed": seed, **r})
        pnls = [r["pnl_pct"] for r in per_seed]
        out = {
            "stage": "oos", "seed_mean_pnl_pct": float(np.mean(pnls)),
            "n_pos_seeds": int(sum(p > 0 for p in pnls)), "per_seed": per_seed,
        }
        (OUT_DIR / "oos_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
