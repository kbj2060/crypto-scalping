"""BTC-specific DSAC (Distributional Soft Actor-Critic) agent, literature-informed variant
(train_eval_btc_dsac_advanced_20260802).

Context: this project's only prior RL-for-trading attempt, Omega4.7-RL
(scripts/train_eval_omega4_7_rl_dsac_20260707.py), was a discrete SAC agent on ETH 5m bars and
failed catastrophically OOS (-88..-99%) despite plausible-looking training/VAL curves -- a
signature of overestimation bias / distribution-shift extrapolation error rather than "bad
architecture". No BTC-specific DSAC agent has ever been trained in this repo (BTC only appears as
an exogenous feature inside the ETH-focused "unified" DSAC setup, see
data/ensemble/ckpt/dsac_unified_train_config.json). This script trains a genuinely BTC-specific
agent using continuous-control DSAC upgrades from 2020-2024 literature chosen specifically to
target the failure mode above (see module docstring literature notes below), not a generic
architecture swap.

Literature-informed upgrades (2-4 concrete, implementable techniques):

1. Distributional quantile critics with a REDQ-style ensemble (Chen et al. 2021, "Randomized
   Ensemble Double Q-Learning") + TQC-style truncation (Kuznetsov et al. 2020, "Controlling
   Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics"):
   N_CRITICS=5 independent quantile critics (N_QUANTILES atoms each, pinball/quantile-Huber loss,
   matching this repo's existing IQN-style DSAC convention elsewhere in scripts/*dsac*), with the
   Bellman target computed from a random subset of M_TARGET=2 critics (REDQ in-target
   minimization) and the policy-gradient Q estimate computed by dropping the top
   TRUNCATE_QUANTILES atoms per critic before averaging (TQC truncation). Both mechanisms directly
   attack Q-overestimation, which compounds every bar in this project's default single twin-critic
   DSAC and is a textbook cause of "trains fine, degenerates OOS."

2. DSAC-T-style stabilization (Duan et al. 2023/2024, "Distributional Soft Actor-Critic with
   Three Refinements", arXiv:2310.05858): expected-value (mean-of-quantiles) substitution for the
   actor's Q target (instead of a single sampled/median atom) and twin-critic-of-distributions
   variance smoothing via the REDQ ensemble above -- reduces critic-gradient variance from return
   randomness, which matters here because 5m BTC bar returns are extremely noisy relative to
   trading costs.

3. CQL-style conservative regularization for the offline/limited-interaction regime (Kumar et al.
   2020, "Conservative Q-Learning"; applied here because training only ever replays a FIXED
   historical tape -- there is no live simulator, so this is effectively offline RL even though
   the code samples "online" from a replay buffer). Adds a logsumexp-over-random-actions minus
   buffer-action Q penalty (weight CQL_ALPHA) to the critic loss, penalizing Q-value extrapolation
   on actions away from what the fixed historical tape actually supports. This is the direct,
   literature-grounded counter to "confident training curves, then the OOS regime differs and the
   policy's Q-values were extrapolating on unsupported actions."

4. Cost-aware reward shaping per recent RL-for-trading literature (transaction-cost-aware
   objectives, e.g. cost-efficient RL execution work, 2022-2025): reward is realized next-bar
   return minus this repo's standard fee+slip cost model (same fee/slip source as
   Omega4.6.1/Omega4.7, see _load_fee_slip below) MINUS an explicit turnover penalty on
   |action-change| beyond the realized trading cost, to bias the policy away from the
   high-turnover overfitting mode that a noisy distributional critic can otherwise exploit
   in-sample.

Fresh-Forward-aware protocol (per CLAUDE.md): TRAIN 2024-01-01..2025-08-31 (strictly pre-VAL),
VAL 2025-09-01..2025-12-31 selects the best checkpoint per seed AND the best seed, OOS
2026-01-01..2026-03-31 is scored ONCE per seed for transparency (not cherry-picked). This is a
stored-tape replay -> DIAGNOSTIC research score per CLAUDE.md Fresh-Forward rule, NOT a live
promotion claim regardless of outcome. trading_bot.py / trading_bot_modules/* are not touched.
Checkpoints go to data/ensemble/ckpt/btc_dsac_advanced_20260802/ (isolated, does not overwrite any
existing *dsac* live artifact).
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

MODEL_ID = "btc_dsac_advanced_20260802"
CKPT_DIR = ROOT / f"data/ensemble/ckpt/{MODEL_ID}"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / f"tmp/research_20260802/btc_dsac_advanced"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026.csv"

TRAIN_START, TRAIN_END = "2024-01-01", "2025-08-31 23:59:59"
VAL_START, VAL_END = "2025-09-01", "2025-12-31 23:59:59"
OOS_START, OOS_END = "2026-01-01", "2026-03-31 23:59:59"

# fee/slip: same convention as Omega4.6.1/Omega4.7 (scripts/train_eval_omega1_2_tabm_diffusion_risk_20260603.py)
FEE_RATE = 0.0005
SLIP_RATE = 0.0002
COST_MULT = 1.0

# Non-stationary / absolute-scale columns excluded from the feature set (raw price & volume
# levels drift across years; the ~130 remaining columns are engineered ratios/z-scores/signals).
EXCLUDE_COLS = {
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value", "close_btc", "volume_btc",
    "quote_volume_btc",
}

SEEDS = [11, 47, 269]
TOTAL_STEPS = 20_000
EPISODE_LEN = 576          # 2 days of 5m bars
EVAL_EVERY = 4_000
BATCH_SIZE = 256
BUFFER_SIZE = 250_000
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
HIDDEN = 256
REWARD_SCALE = 100.0
TURNOVER_PENALTY = 0.05   # extra shaping beyond realized fee/slip cost, per literature note (4)
POS_FEATS = 3

# --- distributional / ensemble critic config (upgrades 1+2) ---
N_CRITICS = 5              # REDQ-style ensemble size
M_TARGET = 2                # REDQ in-target-minimization subset size
N_QUANTILES = 25            # atoms per critic (matches this repo's IQN-style DSAC convention)
TRUNCATE_QUANTILES = 4       # TQC: drop top-k atoms per critic before averaging for policy grad
QUANTILE_TAUS = (torch.arange(N_QUANTILES, dtype=torch.float32) + 0.5) / N_QUANTILES

# --- CQL-style conservative penalty (upgrade 3) ---
CQL_ALPHA = 0.3
CQL_N_RANDOM = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUANTILE_TAUS = QUANTILE_TAUS.to(DEVICE)


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return frame


def feature_cols(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in EXCLUDE_COLS]


def base_features(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = frame.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.to_numpy(dtype=np.float32)


class Tape:
    def __init__(self, frame: pd.DataFrame, cols: list[str], mu: np.ndarray, sd: np.ndarray):
        raw = base_features(frame, cols)
        close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
        self.x = ((raw - mu) / sd).astype(np.float32)
        self.ret = np.zeros(len(close), dtype=np.float64)
        self.ret[:-1] = close[1:] / close[:-1] - 1.0
        self.cost = (FEE_RATE + SLIP_RATE) * COST_MULT
        self.n = len(close)
        self.timestamps = frame["timestamp"].reset_index(drop=True)


def make_state(tape: Tape, i: int, pos: float, entry_move: float, hold: int) -> np.ndarray:
    return np.concatenate([tape.x[i], np.array([pos, entry_move, hold / 288.0], dtype=np.float32)])


class QuantileCritic(nn.Module):
    """One quantile-critic head: outputs N_QUANTILES atoms for Q(s,a)."""

    def __init__(self, n_state: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state + 1, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, N_QUANTILES),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class GaussianPolicy(nn.Module):
    """Continuous tanh-squashed Gaussian policy; action = target position in [-1, 1]."""

    def __init__(self, n_state: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_state, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN), nn.SiLU())
        self.mu = nn.Linear(HIDDEN, 1)
        self.log_std = nn.Linear(HIDDEN, 1)

    def forward(self, s, deterministic=False):
        h = self.body(s)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), -5.0, 2.0)
        std = log_std.exp()
        if deterministic:
            a = torch.tanh(mu)
            return a, None
        dist = torch.distributions.Normal(mu, std)
        pre = dist.rsample()
        a = torch.tanh(pre)
        logp = dist.log_prob(pre) - torch.log(1.0 - a.pow(2) + 1e-6)
        return a, logp.sum(dim=-1, keepdim=True)


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """pred: (B, N), target: (B, N') -> pairwise quantile Huber loss (TQC/IQN style)."""
    diff = target.unsqueeze(1) - pred.unsqueeze(2)  # (B, N, N')
    huber = torch.where(diff.abs() <= 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)
    taus = QUANTILE_TAUS.view(1, -1, 1)
    weight = torch.abs(taus - (diff.detach() < 0).float())
    return (weight * huber).sum(dim=1).mean(dim=1).mean()


class ReplayBuffer:
    def __init__(self, n_state: int):
        self.s = np.zeros((BUFFER_SIZE, n_state), dtype=np.float32)
        self.a = np.zeros((BUFFER_SIZE, 1), dtype=np.float32)
        self.r = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.s2 = np.zeros((BUFFER_SIZE, n_state), dtype=np.float32)
        self.done = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, s, a, r, s2, done):
        self.s[self.idx] = s
        self.a[self.idx, 0] = a
        self.r[self.idx] = r
        self.s2[self.idx] = s2
        self.done[self.idx] = done
        self.idx = (self.idx + 1) % BUFFER_SIZE
        self.full = self.full or self.idx == 0

    def sample(self, rng: np.random.Generator):
        hi = BUFFER_SIZE if self.full else self.idx
        j = rng.integers(0, hi, size=BATCH_SIZE)
        to = lambda arr: torch.as_tensor(arr[j], device=DEVICE)
        return to(self.s), to(self.a), to(self.r), to(self.s2), to(self.done)


@torch.no_grad()
def evaluate_policy(policy: GaussianPolicy, tape: Tape) -> dict:
    policy.eval()
    pos = 0.0
    entry_move_ref = 0.0
    hold = 0
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    seg_start_eq = 1.0
    trades: list[float] = []
    for i in range(tape.n - 1):
        s = torch.as_tensor(make_state(tape, i, pos, entry_move_ref, hold), device=DEVICE).unsqueeze(0)
        a, _ = policy(s, deterministic=True)
        new_pos = float(a.item())
        if abs(new_pos) < 0.15:
            new_pos = 0.0
        if abs(new_pos - pos) > 1e-6:
            equity *= 1.0 - tape.cost * abs(new_pos - pos)
            if pos != 0.0:
                trades.append(equity / seg_start_eq - 1.0)
            if new_pos != 0.0:
                seg_start_eq = equity
            entry_move_ref, hold = 0.0, 0
        else:
            hold += 1
        if new_pos != 0.0:
            equity *= 1.0 + new_pos * tape.ret[i]
            entry_move_ref += new_pos * tape.ret[i]
        pos = new_pos
        peak = max(peak, equity)
        mdd = min(mdd, equity / max(peak, 1e-12) - 1.0)
    if pos != 0.0:
        trades.append(equity / seg_start_eq - 1.0)
    tr = np.asarray(trades)
    policy.train()
    return {"pnl": float((equity - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(len(tr)),
            "wr": float((tr > 0).mean()) if len(tr) else 0.0}


def train_one_seed(seed: int, train_tape: Tape, val_tape: Tape, n_state: int) -> tuple[dict, dict, list]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    policy = GaussianPolicy(n_state).to(DEVICE)
    critics = [QuantileCritic(n_state).to(DEVICE) for _ in range(N_CRITICS)]
    targets = [QuantileCritic(n_state).to(DEVICE) for _ in range(N_CRITICS)]
    for c, t in zip(critics, targets):
        t.load_state_dict(c.state_dict())
    opt_pi = torch.optim.Adam(policy.parameters(), lr=LR)
    opt_q = torch.optim.Adam([p for c in critics for p in c.parameters()], lr=LR)
    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    opt_a = torch.optim.Adam([log_alpha], lr=LR)
    target_entropy = -1.0  # standard SAC default for 1-D continuous action

    buffer = ReplayBuffer(n_state)
    best_val = {"pnl": -1e18}
    best_state = None
    history = []

    step = 0
    t0 = time.time()
    while step < TOTAL_STEPS:
        ep_start = int(rng.integers(0, max(train_tape.n - EPISODE_LEN - 2, 1)))
        pos, entry_move_ref, hold = 0.0, 0.0, 0
        s = make_state(train_tape, ep_start, pos, entry_move_ref, hold)
        for k in range(EPISODE_LEN):
            i = ep_start + k
            with torch.no_grad():
                a_t, _ = policy(torch.as_tensor(s, device=DEVICE).unsqueeze(0))
                a = float(a_t.item())
            new_pos = a if abs(a) >= 0.15 else 0.0
            turnover = abs(new_pos - pos)
            cost = train_tape.cost * turnover
            shaping = TURNOVER_PENALTY * max(turnover - 0.02, 0.0) * train_tape.cost
            r = (new_pos * train_tape.ret[i] - cost - shaping) * REWARD_SCALE
            if abs(new_pos - pos) > 1e-6:
                entry_move_ref, hold = 0.0, 0
            else:
                hold += 1
            if new_pos != 0.0:
                entry_move_ref += new_pos * train_tape.ret[i]
            pos = new_pos
            done = 1.0 if k == EPISODE_LEN - 1 else 0.0
            s2 = make_state(train_tape, i + 1, pos, entry_move_ref, hold)
            buffer.add(s, a, r, s2, done)
            s = s2
            step += 1

            if buffer.full or buffer.idx > 5_000:
                bs, ba, br, bs2, bdone = buffer.sample(rng)
                alpha = log_alpha.exp().detach()

                # ---- critic update: REDQ in-target-min + CQL conservative penalty ----
                with torch.no_grad():
                    a2, logp2 = policy(bs2)
                    subset = rng.choice(N_CRITICS, size=M_TARGET, replace=False)
                    tgt_atoms = torch.stack([targets[j](bs2, a2) for j in subset], dim=0)  # (M, B, N)
                    tgt_min = tgt_atoms.min(dim=0).values  # elementwise-min across critics, REDQ-style
                    tgt_min = tgt_min - alpha * logp2
                    target_atoms = br.unsqueeze(1) + GAMMA * (1.0 - bdone).unsqueeze(1) * tgt_min

                q_loss = 0.0
                cql_loss = 0.0
                for c in critics:
                    pred = c(bs, ba)
                    q_loss = q_loss + quantile_huber_loss(pred, target_atoms)
                    # CQL-style conservative penalty: push down Q on random/OOD actions
                    # relative to the buffer (behavior) action, offline-regularizing extrapolation.
                    rand_a = (torch.rand(bs.shape[0], CQL_N_RANDOM, device=DEVICE) * 2.0 - 1.0)
                    bs_rep = bs.unsqueeze(1).expand(-1, CQL_N_RANDOM, -1).reshape(-1, bs.shape[1])
                    rand_q = c(bs_rep, rand_a.reshape(-1, 1)).mean(dim=-1).reshape(bs.shape[0], CQL_N_RANDOM)
                    logsumexp_rand = torch.logsumexp(rand_q, dim=-1)
                    buf_q = pred.mean(dim=-1)
                    cql_loss = cql_loss + (logsumexp_rand - buf_q).mean()
                q_loss = q_loss + CQL_ALPHA * cql_loss

                opt_q.zero_grad(set_to_none=True)
                q_loss.backward()
                opt_q.step()

                # ---- actor update: DSAC-T style expected-value (mean-of-quantiles) target,
                #      TQC-style truncation (drop top-k atoms per critic) before averaging ----
                a_pi, logp_pi = policy(bs)
                q_means = []
                for c in critics:
                    atoms = c(bs, a_pi)
                    atoms_sorted, _ = torch.sort(atoms, dim=-1)
                    kept = atoms_sorted[:, : N_QUANTILES - TRUNCATE_QUANTILES]
                    q_means.append(kept.mean(dim=-1))
                q_mean = torch.stack(q_means, dim=0).mean(dim=0)
                loss_pi = (alpha * logp_pi.squeeze(-1) - q_mean).mean()
                opt_pi.zero_grad(set_to_none=True)
                loss_pi.backward()
                opt_pi.step()

                loss_a = -(log_alpha * (logp_pi.detach() + target_entropy)).mean()
                opt_a.zero_grad(set_to_none=True)
                loss_a.backward()
                opt_a.step()

                with torch.no_grad():
                    for c, t in zip(critics, targets):
                        for tp, sp in zip(t.parameters(), c.parameters()):
                            tp.mul_(1 - TAU).add_(TAU * sp)

            if step % EVAL_EVERY == 0:
                m = evaluate_policy(policy, val_tape)
                el = time.time() - t0
                print(f"  seed={seed} step={step} ({el:,.0f}s) VAL pnl={m['pnl']:+7.2f}% mdd={m['mdd']:+6.2f}% "
                      f"n={m['trades']} wr={m['wr']:.3f} alpha={float(log_alpha.exp().item()):.4f}", flush=True)
                history.append({"step": step, **m})
                if m["pnl"] > best_val["pnl"]:
                    best_val = m
                    best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            if step >= TOTAL_STEPS:
                break

    if best_state is None:
        best_val = evaluate_policy(policy, val_tape)
        best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    return best_val, best_state, history


def main() -> int:
    frame = load_frame()
    cols = feature_cols(frame)
    print(f"BTC DSAC-advanced: {len(cols)} engineered features, device={DEVICE}", flush=True)

    train_frame = frame[(frame["timestamp"] >= TRAIN_START) & (frame["timestamp"] <= TRAIN_END)].reset_index(drop=True)
    val_frame = frame[(frame["timestamp"] >= VAL_START) & (frame["timestamp"] <= VAL_END)].reset_index(drop=True)
    oos_frame = frame[(frame["timestamp"] >= OOS_START) & (frame["timestamp"] <= OOS_END)].reset_index(drop=True)
    print(f"frames: train={len(train_frame)} val={len(val_frame)} oos={len(oos_frame)}", flush=True)

    train_raw = base_features(train_frame, cols)
    mu = train_raw.mean(axis=0)
    sd = train_raw.std(axis=0)
    sd[sd < 1e-9] = 1.0

    train_tape = Tape(train_frame, cols, mu, sd)
    val_tape = Tape(val_frame, cols, mu, sd)
    oos_tape = Tape(oos_frame, cols, mu, sd)
    n_state = train_tape.x.shape[1] + POS_FEATS

    seed_results = {}
    seed_states = {}
    seed_history = {}
    for seed in SEEDS:
        print(f"\n=== training seed {seed} ===", flush=True)
        best_val, best_state, history = train_one_seed(seed, train_tape, val_tape, n_state)
        seed_results[seed] = {"val": best_val}
        seed_states[seed] = best_state
        seed_history[seed] = history
        print(f"seed {seed} best-VAL: {best_val}", flush=True)

    sel_seed = max(SEEDS, key=lambda s: seed_results[s]["val"]["pnl"])
    print(f"\nVAL-selected seed: {sel_seed} ({seed_results[sel_seed]['val']})", flush=True)

    policy = GaussianPolicy(n_state).to(DEVICE)
    policy.load_state_dict(seed_states[sel_seed])
    frozen_oos = evaluate_policy(policy, oos_tape)
    seed_results[sel_seed]["oos_oneshot"] = frozen_oos
    print(f"OOS ONE-SHOT (frozen seed {sel_seed}): {frozen_oos}", flush=True)

    for seed in SEEDS:
        if seed == sel_seed:
            continue
        policy.load_state_dict(seed_states[seed])
        seed_results[seed]["oos_transparency_only"] = evaluate_policy(policy, oos_tape)
        print(f"OOS transparency seed {seed}: {seed_results[seed]['oos_transparency_only']}", flush=True)

    torch.save({"policy_state": seed_states[sel_seed], "feature_cols": cols, "mu": mu, "sd": sd,
                "n_state": n_state, "config": {"hidden": HIDDEN, "model_id": MODEL_ID}},
               CKPT_DIR / "btc_dsac_advanced_policy_bundle.pt")

    result = {
        "model_id": MODEL_ID,
        "architecture": "continuous DSAC: REDQ-ensemble(5) quantile critics (25 atoms) + TQC "
                         "truncation(top-4 dropped) + CQL conservative penalty + DSAC-T "
                         "expected-value actor target, tanh-Gaussian policy over 1-D target "
                         "position + position context, BTC 130 engineered 5m features",
        "literature_upgrades": [
            "REDQ (Chen et al. 2021) in-target-min ensemble critics",
            "TQC (Kuznetsov et al. 2020) quantile truncation for overestimation control",
            "CQL (Kumar et al. 2020) conservative penalty for offline/fixed-tape regime",
            "DSAC-T (Duan et al. 2023, arXiv:2310.05858) expected-value actor target",
        ],
        "protocol": {"train": [TRAIN_START, TRAIN_END], "val_select": [VAL_START, VAL_END], "oos_oneshot": [OOS_START, OOS_END]},
        "cost_model": {"fee": FEE_RATE, "slip": SLIP_RATE, "cost_mult": COST_MULT},
        "baseline_reference_prior_rl": {"model_id": "omega4_7_rl_dsac_20260707", "asset": "ETH",
                                         "oos_pnl_pct_range": [-99, -88],
                                         "note": "prior RL attempt in this repo, discrete SAC, catastrophic OOS -- comparison point for whether literature upgrades change the qualitative outcome"},
        "seeds": {str(k): v for k, v in seed_results.items()},
        "seed_val_history": {str(k): v for k, v in seed_history.items()},
        "selected_seed": sel_seed,
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "promotion_status": "research/dev score only, NOT a live promotion claim (single investigation, per CLAUDE.md Fresh-Forward rule)",
    }
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUT_DIR / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
