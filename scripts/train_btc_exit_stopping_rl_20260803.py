"""
Layer 4 -- exit-timing policy as an entropy-regularized OPTIMAL STOPPING
problem (not a full trading MDP), reusing this project's own advanced
DSAC building blocks (DistributionalTwinCritic = TQC/DSAC-T quantile
critics, CQL conservative regularization, dynamic entropy, ReDo dormant-
neuron resets -- all already implemented in
ensemble/train_rl_dsac_unified_2025.py::DSACAgent) applied to a MUCH
smaller, better-posed sub-problem than the full-trading RL that failed
for BTC before (see project memory project-btc-dsac-advanced-rl-closed-20260802
-- entropy-temperature collapse when direction+sizing+exit are learned
jointly). Here, Layer 1-3 have ALREADY decided entry/direction/conviction;
this policy ONLY adjusts the trailing-stop tightness bar-by-bar while a
position is open.

Action (1-dim, tanh in [-1,1] -> mapped to [0.5, 2.0]): multiplier on the
existing ATR-adaptive trailing threshold (base formula unchanged, this
policy only tightens/loosens it -- keeps a safety floor, avoids an
unconstrained black-box exit).

State (12-dim): unrealized pnl%, bars_held (normalized), running MFE%,
running MAE%, entry conviction score (Layer 3's predicted net return for
the taken side), mtf1h_ts_t_value, mtf1h_ts_action, 3x regime3 probs
(bull/bear/chop), 3x short-memory of the last 3 bars' log-returns
(EarnHFT/MacroHFT-style light recurrent-ish context without a full RNN).

Reward: potential-based shaping (Ng et al. 1999 -- provably policy-
invariant) with potential = unrealized pnl%, i.e. r_t = pnl%_t - pnl%_{t-1}
each bar (dense feedback, avoids the sparse-terminal-reward failure mode),
plus a small per-bar cost-of-waiting penalty and the realized net cost at
termination.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from build_omega1_2_triple_barrier_labels_btc_20260708 import _atr_price_move  # noqa: E402
from compare_btc_label_schemes_20260803 import cusum_events  # noqa: E402
from build_btc_cusum_trailing_final_model_20260803 import HARD_SL_MULT, HARD_SL_MIN, VAL_START, OOS_START, OOS_END  # noqa: E402
from ensemble.train_rl_dsac_unified_2025 import DSACAgent  # noqa: E402

FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_final95_2024_2026.parquet"
MODEL_DIR = ROOT / "data/ensemble/supervised"
OUT_DIR = ROOT / "data/ensemble/ckpt"
COST_STD = (0.0005 + 0.0002) * 2.0
ENTRY_THRESHOLD = 0.006
MAX_HOLD_BARS = 288
STATE_DIM = 13  # pnl, bars_norm, mfe, mae, conviction, ts_t/5, ts_action, 3xregime3, 3x short-memory returns
TRAIL_MIN_PCT, TRAIL_MAX_PCT, TRAIL_BASE_MULT = 0.010, 0.018, 1.0
WAIT_COST = 0.00005  # small per-bar cost-of-waiting shaping term


def trail_threshold(atr_pct_i: float, action_mult: float) -> float:
    base = np.clip(max(TRAIL_MIN_PCT, atr_pct_i * TRAIL_BASE_MULT), TRAIL_MIN_PCT, TRAIL_MAX_PCT)
    return float(np.clip(base * action_mult, TRAIL_MIN_PCT * 0.5, TRAIL_MAX_PCT * 2.0))


class ExitEpisode:
    """One confirmed trade's forward bar sequence -- state/step logic for the
    exit-timing optimal-stopping policy."""

    def __init__(self, frame, i, side, conviction, close, high, low, open_px, atr):
        self.side = side  # 1 long, 2 short
        self.entry_i = i + 1
        self.entry_price = float(open_px[self.entry_i])
        self.conviction = float(conviction)
        self.close, self.high, self.low, self.atr = close, high, low, atr
        self.mtf1h_t = float(frame["mtf1h_ts_t_value"].iat[i])
        self.mtf1h_action = float(frame["mtf1h_ts_action"].iat[i])
        self.regime = [float(frame["regime3_current_sensitive_wide24_bull_prob"].iat[i]),
                        float(frame["regime3_current_sensitive_wide24_bear_prob"].iat[i]),
                        float(frame["regime3_current_sensitive_wide24_confidence"].iat[i])]
        self.n = len(close)
        self.sl_move = max(HARD_SL_MIN, HARD_SL_MULT * float(atr[i]))
        self.reset()

    def reset(self):
        self.bar = 0
        self.extreme = self.entry_price
        self.prev_pnl = 0.0
        self.done = False
        return self._state()

    def _pnl(self, k):
        price = self.close[min(self.entry_i + k, self.n - 1)]
        return (price / self.entry_price - 1.0) if self.side == 1 else (1.0 - price / self.entry_price)

    def _state(self):
        k = self.bar
        idx = min(self.entry_i + k, self.n - 1)
        pnl = self._pnl(k)
        window = self.close[max(self.entry_i, idx - 3):idx + 1]
        rets = np.diff(np.log(np.clip(window, 1e-9, None))) if len(window) > 1 else np.zeros(1)
        rets = np.pad(rets, (max(0, 3 - len(rets)), 0))[-3:]
        mfe = max(0.0, pnl if self.bar == 0 else max(self._pnl(j) for j in range(k + 1)))
        mae = min(0.0, pnl if self.bar == 0 else min(self._pnl(j) for j in range(k + 1)))
        return np.array([
            pnl, k / MAX_HOLD_BARS, mfe, mae, self.conviction,
            self.mtf1h_t / 5.0, self.mtf1h_action,
            *self.regime, *rets,
        ], dtype=np.float32)

    def step(self, action_raw: float):
        mult = 0.5 + (float(np.clip(action_raw, -1.0, 1.0)) + 1.0) * 0.75  # [-1,1] -> [0.5, 2.0]
        self.bar += 1
        idx = min(self.entry_i + self.bar, self.n - 1)
        hi, lo, close_px = self.high[idx], self.low[idx], self.close[idx]

        sl_hit = (lo <= self.entry_price * (1 - self.sl_move)) if self.side == 1 else (hi >= self.entry_price * (1 + self.sl_move))
        if self.side == 1:
            self.extreme = max(self.extreme, hi)
        else:
            self.extreme = min(self.extreme, lo)
        trail = trail_threshold(float(self.atr[idx]), mult)
        trail_hit = (close_px <= self.extreme * (1 - trail)) if self.side == 1 else (close_px >= self.extreme * (1 + trail))

        terminal = sl_hit or trail_hit or self.bar >= MAX_HOLD_BARS or idx >= self.n - 1
        pnl = self._pnl(self.bar)
        reward = (pnl - self.prev_pnl) - WAIT_COST
        self.prev_pnl = pnl
        if terminal:
            net = pnl - COST_STD
            reward += net  # terminal bonus/penalty on realized net outcome
            self.done = True
        return self._state(), float(reward), terminal, {"net": pnl - COST_STD if terminal else None}


def build_confirmed_trades(frame, models, feat_cols, start, end, close, high, low, open_px, atr):
    """Single-position-at-a-time blocking uses the REAL fixed-trailing (mult=1.0)
    exit bar, not a flat MAX_HOLD_BARS block -- matches the entry set the fixed
    baseline evaluates on, so RL-vs-baseline comparisons are apples-to-apples."""
    from build_btc_cusum_trendscan_zigzag_hybrid_20260803 import simulate_trade
    n = len(frame)
    events = cusum_events(frame, atr, mult=2.0)
    events = events[events < n - MAX_HOLD_BARS - 2]
    ts = frame["timestamp"]
    mask = (ts.iloc[events].to_numpy() >= np.datetime64(start)) & (ts.iloc[events].to_numpy() < np.datetime64(end))
    events = events[mask]
    X = frame.loc[events, feat_cols]
    pl = models["long"].predict(X)
    ps = models["short"].predict(X)
    trades = []
    last_exit = -1
    for k, i in enumerate(events):
        entry_i = i + 1
        if entry_i <= last_exit:
            continue
        side, conv = (1, pl[k]) if (pl[k] >= ENTRY_THRESHOLD and pl[k] >= ps[k]) else \
                     ((2, ps[k]) if ps[k] >= ENTRY_THRESHOLD else (0, 0.0))
        if side == 0:
            continue
        sl_move = max(HARD_SL_MIN, HARD_SL_MULT * float(atr[i]))
        _, _, bars = simulate_trade(side, float(open_px[entry_i]), atr, high, low, close, entry_i, sl_move)
        trades.append((int(i), side, float(conv)))
        last_exit = min(entry_i + bars, n - 1)
    return trades


def run_episode(ep: ExitEpisode, agent: DSACAgent, deterministic: bool, train: bool):
    s = ep.reset()
    total_reward = 0.0
    while not ep.done:
        a = agent.act(s, deterministic=deterministic)
        a_scalar = float(np.asarray(a).reshape(-1)[0])
        ns, r, done, info = ep.step(a_scalar)
        if train:
            agent.memory.push(s, np.array([a_scalar], dtype=np.float32), r, ns, float(done))
        s = ns
        total_reward += r
    return total_reward, info.get("net")


def main():
    frame = pd.read_parquet(FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    with open(MODEL_DIR / "btc_cusum_trailing_final_long.pkl", "rb") as f:
        long_model = pickle.load(f)
    with open(MODEL_DIR / "btc_cusum_trailing_final_short.pkl", "rb") as f:
        short_model = pickle.load(f)
    models = {"long": long_model, "short": short_model}
    feat_cols = long_model.feature_name_

    close = frame["close"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    open_px = frame["open"].to_numpy(dtype=np.float64)
    atr = _atr_price_move(frame)

    args = (close, high, low, open_px, atr)
    train_trades = build_confirmed_trades(frame, models, feat_cols, frame["timestamp"].iloc[0], VAL_START, *args)
    val_trades = build_confirmed_trades(frame, models, feat_cols, VAL_START, OOS_START, *args)
    oos_trades = build_confirmed_trades(frame, models, feat_cols, OOS_START, OOS_END, *args)
    print(f"confirmed trades: train={len(train_trades)} val={len(val_trades)} oos={len(oos_trades)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LR = 1e-4  # lowered from 3e-4 -- prior run peaked at epoch 5 then degraded (primacy bias signature)
    agent = DSACAgent(state_dim=STATE_DIM, hidden_dim=128, n_quantiles=25,
                       lr_actor=LR, lr_critic=LR,
                       cql_reg=True, cql_alpha=0.12, dynamic_entropy=True,
                       # entropy target bounds overridden from the shared defaults
                       # (-0.80/-0.45, tuned for the old multi-dim trading action space)
                       # to a range appropriate for our 1-dim stopping action -- v4's log
                       # showed alpha decaying too fast/too early (0.0278->0.0077 by epoch
                       # 12), exactly tracking policy_entropy's collapse. Standard SAC
                       # practice targets entropy ~= -action_dim = -1 for continuous
                       # actions; -0.45 was pulling toward MORE determinism than that.
                       entropy_min=-1.30, entropy_max=-0.70,
                       # more conservative twin-critic pessimism (was 0.65 default) --
                       # v4's mean_q rose smoothly 0.035->0.33 before crashing, a classic
                       # overestimation-then-correction signature; leaning harder on the
                       # more pessimistic of q1/q2 should blunt that climb.
                       pessimism_min_weight=0.75,
                       # ReDo (dormant-neuron rejuvenation) + primacy soft-reset: this
                       # project's own DSACAgent already implements the exact mitigations
                       # the primacy-bias/plasticity-loss literature recommends
                       # (Nikishin et al. 2022 "The Primacy Bias in Deep RL"; Sokar et al.
                       # 2023 "The Dormant Neuron Phenomenon") -- were OFF last run.
                       redo_enable=True, redo_interval=300, redo_ratio=0.10,
                       primacy_soft_reset=True, primacy_window=60, primacy_reset_cooldown=90,
                       device=device)
    print(f"DSACAgent on {device}, lr={LR}, cql_alpha=0.12, entropy=[-1.30,-0.70], "
          f"pessimism_min_weight=0.75, redo_enable=True, primacy_soft_reset=True", flush=True)

    # --- additional overfitting-mitigation (2025 offline-RL regularization literature),
    # applied externally so the shared ensemble/train_rl_dsac_unified_2025.py module
    # (used by other live-adjacent training paths) is not modified:
    # AdamW + weight decay + dropout on the shared feature trunks via forward hooks
    # (module.training tracks the parent's train()/eval() mode automatically), plus a
    # cosine LR decay schedule (standard recent practice, smooths the late-training
    # instability that a constant LR causes once primacy bias sets in).
    import torch.nn.functional as F
    WEIGHT_DECAY = 1e-3
    agent.actor_optimizer = torch.optim.AdamW(agent.actor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    agent.critic_optimizer = torch.optim.AdamW(agent.critic.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    N_EPOCHS = 80
    actor_sched = torch.optim.lr_scheduler.CosineAnnealingLR(agent.actor_optimizer, T_max=N_EPOCHS)
    critic_sched = torch.optim.lr_scheduler.CosineAnnealingLR(agent.critic_optimizer, T_max=N_EPOCHS)

    DROPOUT_P = 0.15

    def _dropout_hook(module, inp, out):
        return F.dropout(out, p=DROPOUT_P, training=module.training)

    agent.actor.feat.register_forward_hook(_dropout_hook)
    agent.critic.feat1.register_forward_hook(_dropout_hook)
    agent.critic.feat2.register_forward_hook(_dropout_hook)
    print(f"regularization: AdamW weight_decay={WEIGHT_DECAY}, dropout p={DROPOUT_P}, cosine LR decay over {N_EPOCHS} epochs", flush=True)

    def evaluate(trades, label, verbose=True):
        nets = []
        for i, side, conv in trades:
            ep = ExitEpisode(frame, i, side, conv, close, high, low, open_px, atr)
            _, net = run_episode(ep, agent, deterministic=True, train=False)
            nets.append(net)
        nets = np.array(nets)
        if verbose:
            print(f"{label}: n={len(nets)} win%={100*(nets>0).mean():.1f} mean_net={100*nets.mean():.3f}% sum_net={100*nets.sum():.2f}%")
        return nets

    import time
    rng = np.random.default_rng(0)
    UPDATES_PER_EPISODE = 1
    CHECK_EVERY = 2  # finer-grained than last run's every-5 -- don't miss an early peak
    PATIENCE_CHECKS = 10  # 20 epochs of patience at CHECK_EVERY=2
    best_val_mean = -np.inf
    best_state = None
    best_epoch = -1
    checks_since_best = 0
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        order = rng.permutation(len(train_trades))
        ep_rewards = []
        update_infos = []
        for idx in order:
            i, side, conv = train_trades[idx]
            ep = ExitEpisode(frame, i, side, conv, close, high, low, open_px, atr)
            total_r, _ = run_episode(ep, agent, deterministic=False, train=True)
            ep_rewards.append(total_r)
            if len(agent.memory) >= 256:
                for _ in range(UPDATES_PER_EPISODE):
                    info = agent.update(batch_size=256)
                    if info:
                        update_infos.append(info)
        actor_sched.step()
        critic_sched.step()

        elapsed = time.time() - t0
        cur_lr = agent.actor_optimizer.param_groups[0]["lr"]
        if update_infos:
            avg = {k: float(np.mean([d[k] for d in update_infos if k in d])) for k in update_infos[0]}
            loss_str = (f"critic_loss={avg.get('critic_loss', float('nan')):.4f} "
                        f"actor_loss={avg.get('actor_loss', float('nan')):.4f} "
                        f"alpha={avg.get('alpha', float('nan')):.4f} "
                        f"mean_q={avg.get('mean_q', float('nan')):.4f} "
                        f"policy_entropy={avg.get('policy_entropy', float('nan')):.4f} "
                        f"cql_pen={avg.get('cql_pen', float('nan')):.4f} "
                        f"redo_count={avg.get('redo_count', 0):.1f}")
        else:
            loss_str = "(no updates yet, buffer filling)"
        print(f"[{elapsed:6.1f}s] epoch {epoch+1}/{N_EPOCHS} lr={cur_lr:.2e} "
              f"train_reward={np.mean(ep_rewards):.4f} n_updates={len(update_infos)} | {loss_str}", flush=True)

        # Entropy-collapse early stop: v4's log showed policy_entropy was a LEADING
        # indicator (started dropping ~epoch 9) well before VAL confirmed degradation
        # (VAL patience=10 checks didn't trigger until epoch 22). React directly to
        # entropy instead of waiting for VAL to lag behind the collapse.
        ENTROPY_FLOOR = -0.50
        cur_entropy = avg.get("policy_entropy", None) if update_infos else None
        if cur_entropy is not None and cur_entropy < ENTROPY_FLOOR:
            print(f"entropy-collapse early stop at epoch {epoch+1}: policy_entropy={cur_entropy:.4f} < {ENTROPY_FLOOR}", flush=True)
            val_nets = evaluate(val_trades, f"  [epoch {epoch+1}] VAL", verbose=False)
            val_mean = float(val_nets.mean())
            if val_mean > best_val_mean:
                best_val_mean = val_mean
                best_epoch = epoch + 1
                best_state = {"actor": {k: v.clone() for k, v in agent.actor.state_dict().items()},
                               "critic": {k: v.clone() for k, v in agent.critic.state_dict().items()}}
            break

        if (epoch + 1) % CHECK_EVERY == 0 or epoch == N_EPOCHS - 1:
            val_nets = evaluate(val_trades, f"  [epoch {epoch+1}] VAL", verbose=False)
            val_mean = float(val_nets.mean())
            print(f"  >>> VAL_mean_net={100*val_mean:.3f}% (best so far: {100*best_val_mean:.3f}% @ epoch {best_epoch}, "
                  f"checks_since_best={checks_since_best}/{PATIENCE_CHECKS})", flush=True)
            if val_mean > best_val_mean:
                best_val_mean = val_mean
                best_epoch = epoch + 1
                best_state = {"actor": {k: v.clone() for k, v in agent.actor.state_dict().items()},
                               "critic": {k: v.clone() for k, v in agent.critic.state_dict().items()}}
                checks_since_best = 0
            else:
                checks_since_best += 1
                if checks_since_best >= PATIENCE_CHECKS:
                    print(f"early stopping at epoch {epoch+1}: no VAL improvement for {PATIENCE_CHECKS} checks", flush=True)
                    break

    print(f"\nbest VAL mean_net during training: {100*best_val_mean:.3f}% @ epoch {best_epoch} -- restoring that checkpoint for final eval", flush=True)
    agent.actor.load_state_dict(best_state["actor"])
    agent.critic.load_state_dict(best_state["critic"])

    print("\n=== Layer 4 RL exit policy (deterministic eval, best-VAL checkpoint) ===")
    evaluate(val_trades, "VAL")
    evaluate(oos_trades, "OOS")

    torch.save(agent.actor.state_dict(), OUT_DIR / "btc_exit_stopping_rl_actor_20260803.pth")
    torch.save(agent.critic.state_dict(), OUT_DIR / "btc_exit_stopping_rl_critic_20260803.pth")
    print(f"saved actor/critic to {OUT_DIR}")


if __name__ == "__main__":
    main()
