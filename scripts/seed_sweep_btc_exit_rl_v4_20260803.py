"""
Reproduce the "v4" Layer 4 exit-RL config (lr=1e-4, cql_alpha=0.05, default
entropy bounds, default pessimism, ReDo+primacy-reset on, weight_decay=1e-3,
dropout=0.15, cosine LR over 80 epochs, VAL-based early stop patience=10)
across 5 different random seeds, to check whether v4's apparent VAL+OOS
improvement over the fixed-trailing baseline (VAL +0.578->+0.703%, OOS
+0.670->+0.730%) reproduces, or was a lucky single-seed early-checkpoint
artifact (all 3 prior runs picked their best checkpoint at epoch ~2-5,
before any real training had happened).

Per this project's own seed-diversity promotion policy (CLAUDE.md): a
single-seed "improvement" is not adoptable evidence; N>=5 genuinely
different seeds must agree in sign/direction.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import train_btc_exit_stopping_rl_20260803 as M  # noqa: E402

SEEDS = [270705, 190412, 830177, 550923, 4021]  # genuinely distinct, not fixed-increment (per seed-diversity gate)
OUT_DIR = ROOT / "data/ensemble/ckpt"
N_EPOCHS = 80
CHECK_EVERY = 2
PATIENCE_CHECKS = 10


def run_one_seed(seed, frame, train_trades, val_trades, oos_trades, close, high, low, open_px, atr):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = M.DSACAgent(state_dim=M.STATE_DIM, hidden_dim=128, n_quantiles=25,
                         lr_actor=1e-4, lr_critic=1e-4,
                         cql_reg=True, cql_alpha=0.05, dynamic_entropy=True,
                         redo_enable=True, redo_interval=300, redo_ratio=0.10,
                         primacy_soft_reset=True, primacy_window=60, primacy_reset_cooldown=90,
                         device=device)
    agent.actor_optimizer = torch.optim.AdamW(agent.actor.parameters(), lr=1e-4, weight_decay=1e-3)
    agent.critic_optimizer = torch.optim.AdamW(agent.critic.parameters(), lr=1e-4, weight_decay=1e-3)

    def _dropout_hook(module, inp, out):
        return F.dropout(out, p=0.15, training=module.training)

    agent.actor.feat.register_forward_hook(_dropout_hook)
    agent.critic.feat1.register_forward_hook(_dropout_hook)
    agent.critic.feat2.register_forward_hook(_dropout_hook)

    actor_sched = torch.optim.lr_scheduler.CosineAnnealingLR(agent.actor_optimizer, T_max=N_EPOCHS)
    critic_sched = torch.optim.lr_scheduler.CosineAnnealingLR(agent.critic_optimizer, T_max=N_EPOCHS)

    def evaluate(trades):
        nets = []
        for i, side, conv in trades:
            ep = M.ExitEpisode(frame, i, side, conv, close, high, low, open_px, atr)
            _, net = M.run_episode(ep, agent, deterministic=True, train=False)
            nets.append(net)
        return np.array(nets)

    best_val_mean, best_epoch, best_state = -np.inf, -1, None
    checks_since_best = 0
    for epoch in range(N_EPOCHS):
        order = rng.permutation(len(train_trades))
        for idx in order:
            i, side, conv = train_trades[idx]
            ep = M.ExitEpisode(frame, i, side, conv, close, high, low, open_px, atr)
            M.run_episode(ep, agent, deterministic=False, train=True)
            if len(agent.memory) >= 256:
                agent.update(batch_size=256)
        actor_sched.step()
        critic_sched.step()
        if (epoch + 1) % CHECK_EVERY == 0 or epoch == N_EPOCHS - 1:
            val_mean = float(evaluate(val_trades).mean())
            if val_mean > best_val_mean:
                best_val_mean, best_epoch = val_mean, epoch + 1
                best_state = {"actor": {k: v.clone() for k, v in agent.actor.state_dict().items()}}
                checks_since_best = 0
            else:
                checks_since_best += 1
                if checks_since_best >= PATIENCE_CHECKS:
                    break
    agent.actor.load_state_dict(best_state["actor"])
    val_nets = evaluate(val_trades)
    oos_nets = evaluate(oos_trades)
    return {
        "seed": seed, "best_epoch": best_epoch,
        "val_mean_net": float(val_nets.mean()), "val_win": float(100 * (val_nets > 0).mean()),
        "oos_mean_net": float(oos_nets.mean()), "oos_win": float(100 * (oos_nets > 0).mean()),
    }


def main():
    frame = pd.read_parquet(M.FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    with open(M.MODEL_DIR / "btc_cusum_trailing_final_long.pkl", "rb") as f:
        long_model = pickle.load(f)
    with open(M.MODEL_DIR / "btc_cusum_trailing_final_short.pkl", "rb") as f:
        short_model = pickle.load(f)
    models = {"long": long_model, "short": short_model}
    feat_cols = long_model.feature_name_

    close = frame["close"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    open_px = frame["open"].to_numpy(dtype=np.float64)
    atr = M._atr_price_move(frame)
    args = (close, high, low, open_px, atr)
    train_trades = M.build_confirmed_trades(frame, models, feat_cols, frame["timestamp"].iloc[0], M.VAL_START, *args)
    val_trades = M.build_confirmed_trades(frame, models, feat_cols, M.VAL_START, M.OOS_START, *args)
    oos_trades = M.build_confirmed_trades(frame, models, feat_cols, M.OOS_START, M.OOS_END, *args)
    print(f"confirmed trades: train={len(train_trades)} val={len(val_trades)} oos={len(oos_trades)}", flush=True)

    results = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        r = run_one_seed(seed, frame, train_trades, val_trades, oos_trades, close, high, low, open_px, atr)
        print(f"seed={seed} best_epoch={r['best_epoch']} VAL={r['val_mean_net']*100:.3f}%(win{r['val_win']:.1f}%) "
              f"OOS={r['oos_mean_net']*100:.3f}%(win{r['oos_win']:.1f}%)", flush=True)
        results.append(r)

    df = pd.DataFrame(results)
    print("\n=== SUMMARY across 5 seeds ===")
    print(df.to_string(index=False))
    print(f"\nbaseline (fixed trailing): VAL=0.578% OOS=0.670%")
    print(f"VAL beats baseline: {(df['val_mean_net']*100 > 0.578).sum()}/5")
    print(f"OOS beats baseline: {(df['oos_mean_net']*100 > 0.670).sum()}/5")
    df.to_csv(ROOT / "tmp/btc_exit_rl_v4_seed_sweep_20260803.csv", index=False)


if __name__ == "__main__":
    main()
