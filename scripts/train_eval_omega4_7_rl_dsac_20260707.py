"""Omega4.7-RL (omega4_7_rl_dsac_20260707): reinforcement-learning successor experiment to
Omega4.6.1.

Design brief (user request 2026-07-07): keep ONLY Omega4.6.1's feature processing -- the zig075
bundle's 102-column base feature contract and its exact numeric coercion (`parent._base_input`
semantics) -- and replace the entire model stack (TabM 3-head parent, risk sidecar, static TP/SL
barrier, greedy router) with a from-scratch RL agent.

Architecture (researched choice): Discrete Soft Actor-Critic (discrete SAC, Christodoulou 2019
style) -- twin Q-networks + categorical policy + automatic entropy temperature. Chosen over
alternatives because (a) it is the discrete-action analogue of the DSAC lineage this repo already
uses elsewhere (ensemble/train_rl_dsac_*.py), (b) max-entropy RL is the standard defense against
the degenerate always-one-action policies that plain DQN collapses to on financial tapes, and
(c) the action space here is naturally tiny ({CASH, LONG, SHORT}) so a categorical policy is exact.

Environment: bar-by-bar 5m tape. State = z-scored 102 base features (TRAIN-window statistics only)
+ position context (pos, unrealized_move, hold_bars/288). Action directly sets the target position
in {0, +1, -1} at fixed notional 1.0 (no sizing model -- deliberately minimal v1; sizing was
Candidate 4 territory and is out of scope). Reward = new_pos * next-bar close-to-close return -
(fee+slip) * |position change| (same fee/slip source as every Omega4.6.1 replay), scaled x100 for
optimizer conditioning.

Fresh-Forward-aware protocol: agent trains ONLY on TRAIN 2025-01-01..09-30 (strictly pre-VAL).
During training, the greedy (argmax) policy is periodically evaluated on VAL 2025-10-01..12-31;
the best-VAL checkpoint per seed is kept, the best seed is selected on VAL, and that single frozen
agent is scored ONCE on OOS 2026-01-01..06-30. Other seeds' OOS numbers are reported for
transparency only. Stored-tape based -> DIAGNOSTIC research score, not a live-promotion claim.
trading_bot.py / trading_bot_modules/* are NOT touched.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import replay_omega4_6_1_greedy_val_20260706 as valmod  # noqa: E402

MODEL_ID = "omega4_7_rl_dsac_20260707"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = "2025-01-01", "2025-09-30 23:59:59"
VAL_START, VAL_END = "2025-10-01", "2025-12-31 23:59:59"
OOS_START, OOS_END = "2026-01-01", "2026-06-30"

SEEDS = [0, 1, 2]
TOTAL_STEPS = 150_000
EPISODE_LEN = 576          # 2 days of 5m bars per training episode
EVAL_EVERY = 10_000
BATCH_SIZE = 256
BUFFER_SIZE = 200_000
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
HIDDEN = 256
REWARD_SCALE = 100.0
TARGET_ENTROPY_FRAC = 0.6  # target entropy = frac * log(n_actions)
STRIDE = 1                 # decision interval in 5m bars (12 = hourly decisions)
ACTIONS = np.array([0, 1, -1], dtype=np.int64)  # index -> target position
N_ACTIONS = 3
POS_FEATS = 3              # pos, unrealized_move, hold/288

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_2025_frame(start: str, end: str) -> pd.DataFrame:
    frame = pd.read_csv(valmod.BASE_2025, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    overlay = pd.read_csv(valmod.WIDE24_2025, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)].reset_index(drop=True)
    return frame


def base_features(frame: pd.DataFrame, base_cols: list[str]) -> np.ndarray:
    """Omega4.6.1's exact base-feature processing (parent._base_input without the POS_COLS
    padding, which the RL state replaces with its own position context)."""
    x = frame.reindex(columns=base_cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.to_numpy(dtype=np.float32)


class Tape:
    """One evaluation/training window: z-scored features + per-bar simple returns + cost."""

    def __init__(self, frame: pd.DataFrame, base_cols: list[str], mu: np.ndarray, sd: np.ndarray,
                 fee: float, slip: float, cost_mult: float):
        raw = base_features(frame, base_cols)
        close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
        if STRIDE > 1:
            # hourly-decision variant: decisions only every STRIDE bars; per-step return is the
            # compounded close-to-close return across the whole decision block (still causal --
            # features at decision bar i, reward realized over (i, i+STRIDE]).
            idx = np.arange(0, len(close), STRIDE)
            raw = raw[idx]
            close = close[idx]
            frame = frame.iloc[idx].reset_index(drop=True)
        self.x = ((raw - mu) / sd).astype(np.float32)
        self.ret = np.zeros(len(close), dtype=np.float64)
        self.ret[:-1] = close[1:] / close[:-1] - 1.0  # ret[i] = decision i -> i+1 return
        self.cost = (float(fee) + float(slip)) * float(cost_mult)
        self.n = len(close)
        self.timestamps = frame["timestamp"].reset_index(drop=True)


class QNet(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
                                 nn.Linear(HIDDEN, N_ACTIONS))

    def forward(self, s):
        return self.net(s)


class PolicyNet(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
                                 nn.Linear(HIDDEN, N_ACTIONS))

    def forward(self, s):
        return F.log_softmax(self.net(s), dim=-1)


def make_state(tape: Tape, i: int, pos: int, entry_price_move: float, hold: int) -> np.ndarray:
    return np.concatenate([tape.x[i], np.array([float(pos), float(entry_price_move), float(hold) / 288.0], dtype=np.float32)])


class ReplayBuffer:
    def __init__(self, n_state: int):
        self.s = np.zeros((BUFFER_SIZE, n_state), dtype=np.float32)
        self.a = np.zeros(BUFFER_SIZE, dtype=np.int64)
        self.r = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.s2 = np.zeros((BUFFER_SIZE, n_state), dtype=np.float32)
        self.done = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, s, a, r, s2, done):
        self.s[self.idx], self.a[self.idx], self.r[self.idx], self.s2[self.idx], self.done[self.idx] = s, a, r, s2, done
        self.idx = (self.idx + 1) % BUFFER_SIZE
        self.full = self.full or self.idx == 0

    def sample(self, rng: np.random.Generator):
        hi = BUFFER_SIZE if self.full else self.idx
        j = rng.integers(0, hi, size=BATCH_SIZE)
        to = lambda arr: torch.as_tensor(arr[j], device=DEVICE)
        return to(self.s), to(self.a), to(self.r), to(self.s2), to(self.done)


@torch.no_grad()
def evaluate_policy(policy: PolicyNet, tape: Tape) -> dict:
    """Greedy (argmax) rollout over the full window with the same cost model; returns
    pnl/mdd/trades/wr computed from completed position segments."""
    policy.eval()
    pos = 0
    entry_move_ref = 0.0
    hold = 0
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    seg_start_eq = 1.0
    trades: list[float] = []
    # batch the whole tape's features through the net per unique position context would be wrong
    # (state depends on running position), so step sequentially but keep tensors small.
    for i in range(tape.n - 1):
        s = torch.as_tensor(make_state(tape, i, pos, entry_move_ref, hold), device=DEVICE).unsqueeze(0)
        a = int(policy(s).argmax(dim=-1).item())
        new_pos = int(ACTIONS[a])
        if new_pos != pos:
            equity *= 1.0 - tape.cost * abs(new_pos - pos)
            if pos != 0:
                trades.append(equity / seg_start_eq - 1.0)
            if new_pos != 0:
                seg_start_eq = equity
            entry_move_ref, hold = 0.0, 0
        else:
            hold += 1
        if new_pos != 0:
            equity *= 1.0 + new_pos * tape.ret[i]
            entry_move_ref += new_pos * tape.ret[i]
        pos = new_pos
        peak = max(peak, equity)
        mdd = min(mdd, equity / max(peak, 1e-12) - 1.0)
    if pos != 0:
        trades.append(equity / seg_start_eq - 1.0)
    tr = np.asarray(trades)
    policy.train()
    return {"pnl": float((equity - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(len(tr)),
            "wr": float((tr > 0).mean()) if len(tr) else 0.0}


def train_one_seed(seed: int, train_tape: Tape, val_tape: Tape, n_state: int) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    policy = PolicyNet(n_state).to(DEVICE)
    q1, q2 = QNet(n_state).to(DEVICE), QNet(n_state).to(DEVICE)
    q1_t, q2_t = QNet(n_state).to(DEVICE), QNet(n_state).to(DEVICE)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())
    opt_pi = torch.optim.Adam(policy.parameters(), lr=LR)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=LR)
    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    opt_a = torch.optim.Adam([log_alpha], lr=LR)
    target_entropy = TARGET_ENTROPY_FRAC * float(np.log(N_ACTIONS))

    buffer = ReplayBuffer(n_state)
    best_val = {"pnl": -1e18}
    best_state = None

    step = 0
    t0 = time.time()
    while step < TOTAL_STEPS:
        ep_start = int(rng.integers(0, max(train_tape.n - EPISODE_LEN - 2, 1)))
        pos, entry_move_ref, hold = 0, 0.0, 0
        s = make_state(train_tape, ep_start, pos, entry_move_ref, hold)
        for k in range(EPISODE_LEN):
            i = ep_start + k
            with torch.no_grad():
                logp = policy(torch.as_tensor(s, device=DEVICE).unsqueeze(0))
                a = int(torch.distributions.Categorical(logits=logp).sample().item())
            new_pos = int(ACTIONS[a])
            cost = train_tape.cost * abs(new_pos - pos)
            r = (new_pos * train_tape.ret[i] - cost) * REWARD_SCALE
            if new_pos != pos:
                entry_move_ref, hold = 0.0, 0
            else:
                hold += 1
            if new_pos != 0:
                entry_move_ref += new_pos * train_tape.ret[i]
            pos = new_pos
            done = 1.0 if k == EPISODE_LEN - 1 else 0.0
            s2 = make_state(train_tape, i + 1, pos, entry_move_ref, hold)
            buffer.add(s, a, r, s2, done)
            s = s2
            step += 1

            if (buffer.full or buffer.idx > 5_000) and step % 1 == 0:
                bs, ba, br, bs2, bdone = buffer.sample(rng)
                alpha = log_alpha.exp().detach()
                with torch.no_grad():
                    logp2 = policy(bs2)
                    p2 = logp2.exp()
                    qmin2 = torch.min(q1_t(bs2), q2_t(bs2))
                    v2 = (p2 * (qmin2 - alpha * logp2)).sum(dim=-1)
                    target = br + GAMMA * (1.0 - bdone) * v2
                q1v = q1(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                q2v = q2(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                loss_q = F.mse_loss(q1v, target) + F.mse_loss(q2v, target)
                opt_q.zero_grad(set_to_none=True)
                loss_q.backward()
                opt_q.step()

                logp_all = policy(bs)
                p_all = logp_all.exp()
                with torch.no_grad():
                    qmin = torch.min(q1(bs), q2(bs))
                loss_pi = (p_all * (log_alpha.exp().detach() * logp_all - qmin)).sum(dim=-1).mean()
                opt_pi.zero_grad(set_to_none=True)
                loss_pi.backward()
                opt_pi.step()

                entropy = -(p_all.detach() * logp_all.detach()).sum(dim=-1).mean()
                loss_a = (log_alpha.exp() * (entropy - target_entropy).detach()).mean()
                opt_a.zero_grad(set_to_none=True)
                loss_a.backward()
                opt_a.step()

                with torch.no_grad():
                    for tp, sp in zip(q1_t.parameters(), q1.parameters()):
                        tp.mul_(1 - TAU).add_(TAU * sp)
                    for tp, sp in zip(q2_t.parameters(), q2.parameters()):
                        tp.mul_(1 - TAU).add_(TAU * sp)

            if step % EVAL_EVERY == 0:
                m = evaluate_policy(policy, val_tape)
                el = time.time() - t0
                print(f"  seed={seed} step={step} ({el:,.0f}s) VAL pnl={m['pnl']:+7.2f}% mdd={m['mdd']:+6.2f}% "
                      f"n={m['trades']} wr={m['wr']:.3f} alpha={float(log_alpha.exp()):.4f}", flush=True)
                if m["pnl"] > best_val["pnl"]:
                    best_val = m
                    best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            if step >= TOTAL_STEPS:
                break

    if best_state is None:
        best_val = evaluate_policy(policy, val_tape)
        best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    return best_val, best_state


def main() -> int:
    cfg = retest.COMPONENTS["zig075"]
    bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
    base_cols = list(bundle["base_cols"])
    print(f"Omega4.7-RL: {len(base_cols)} base features from zig075 bundle (Omega4.6.1 feature contract)", flush=True)
    fee, slip = omega._load_fee_slip()

    train_frame = load_2025_frame(TRAIN_START, TRAIN_END)
    val_frame = load_2025_frame(VAL_START, VAL_END)
    oos_frame = retest.load_frame_current(OOS_START, OOS_END)
    print(f"frames: train={len(train_frame)} val={len(val_frame)} oos={len(oos_frame)} device={DEVICE}", flush=True)

    train_raw = base_features(train_frame, base_cols)
    mu = train_raw.mean(axis=0)
    sd = train_raw.std(axis=0)
    sd[sd < 1e-9] = 1.0

    train_tape = Tape(train_frame, base_cols, mu, sd, fee, slip, retest.COST_MULT)
    val_tape = Tape(val_frame, base_cols, mu, sd, fee, slip, retest.COST_MULT)
    oos_tape = Tape(oos_frame, base_cols, mu, sd, fee, slip, retest.COST_MULT)
    n_state = train_tape.x.shape[1] + POS_FEATS

    seed_results = {}
    seed_states = {}
    for seed in SEEDS:
        print(f"\n=== training seed {seed} ===", flush=True)
        best_val, best_state = train_one_seed(seed, train_tape, val_tape, n_state)
        seed_results[seed] = {"val": best_val}
        seed_states[seed] = best_state
        print(f"seed {seed} best-VAL: {best_val}", flush=True)

    # VAL selects the seed; freeze it
    sel_seed = max(SEEDS, key=lambda s: seed_results[s]["val"]["pnl"])
    print(f"\nVAL-selected seed: {sel_seed} ({seed_results[sel_seed]['val']})", flush=True)

    policy = PolicyNet(n_state).to(DEVICE)
    policy.load_state_dict(seed_states[sel_seed])
    frozen_oos = evaluate_policy(policy, oos_tape)
    seed_results[sel_seed]["oos_oneshot"] = frozen_oos
    print(f"OOS ONE-SHOT (frozen seed {sel_seed}): {frozen_oos}", flush=True)

    # transparency: other seeds' OOS (NOT selection-relevant)
    for seed in SEEDS:
        if seed == sel_seed:
            continue
        policy.load_state_dict(seed_states[seed])
        seed_results[seed]["oos_transparency_only"] = evaluate_policy(policy, oos_tape)
        print(f"OOS transparency seed {seed}: {seed_results[seed]['oos_transparency_only']}", flush=True)

    torch.save({"policy_state": seed_states[sel_seed], "base_cols": base_cols, "mu": mu, "sd": sd,
                "n_state": n_state, "config": {"hidden": HIDDEN, "actions": ACTIONS.tolist(), "model_id": MODEL_ID}},
               OUT_DIR / "omega4_7_rl_policy_bundle.pt")

    result = {
        "model_id": MODEL_ID,
        "architecture": "discrete SAC (twin Q + categorical policy + auto entropy), 102 Omega4.6.1 base features + position context",
        "protocol": {"train": [TRAIN_START, TRAIN_END], "val_select": [VAL_START, VAL_END], "oos_oneshot": [OOS_START, OOS_END]},
        "baseline_reference_oos_gate": {"pnl": 145.34, "mdd": -10.13, "trades": 24, "wr": 0.542,
                                         "note": "Omega4.6.1 frozen live baseline (different sizing model -- RL runs fixed notional 1.0, so compare shape/sign, not magnitude, and rely on wr/mdd/trade-count)"},
        "seeds": {str(k): v for k, v in seed_results.items()},
        "selected_seed": sel_seed,
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
    }
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUT_DIR / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
