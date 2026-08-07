"""Retrain the Layer 4 exit-RL policy with seed=270705 (best of the 5-seed
reproduction sweep, OOS +0.731%) and save the actor checkpoint to disk --
the seed-sweep script only kept best_state in memory, never persisted it."""
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
from seed_sweep_btc_exit_rl_v4_20260803 import run_one_seed  # noqa: E402

SEED = 270705
OUT_DIR = ROOT / "data/ensemble/ckpt"


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

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)
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

    N_EPOCHS = 80
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
        if (epoch + 1) % 2 == 0 or epoch == N_EPOCHS - 1:
            val_mean = float(evaluate(val_trades).mean())
            print(f"epoch {epoch+1}: VAL_mean_net={100*val_mean:.3f}% (best={100*best_val_mean:.3f}%@{best_epoch})", flush=True)
            if val_mean > best_val_mean:
                best_val_mean, best_epoch = val_mean, epoch + 1
                best_state = {"actor": {k: v.clone() for k, v in agent.actor.state_dict().items()},
                               "critic": {k: v.clone() for k, v in agent.critic.state_dict().items()}}
                checks_since_best = 0
            else:
                checks_since_best += 1
                if checks_since_best >= 10:
                    print(f"early stop at epoch {epoch+1}", flush=True)
                    break

    agent.actor.load_state_dict(best_state["actor"])
    agent.critic.load_state_dict(best_state["critic"])
    val_nets = evaluate(val_trades)
    oos_nets = evaluate(oos_trades)
    print(f"FINAL (seed={SEED}, best_epoch={best_epoch}): "
          f"VAL n={len(val_nets)} win%={100*(val_nets>0).mean():.1f} mean={100*val_nets.mean():.3f}% | "
          f"OOS n={len(oos_nets)} win%={100*(oos_nets>0).mean():.1f} mean={100*oos_nets.mean():.3f}%", flush=True)

    torch.save(agent.actor.state_dict(), OUT_DIR / "btc_exit_stopping_rl_actor_seed270705_20260803.pth")
    torch.save(agent.critic.state_dict(), OUT_DIR / "btc_exit_stopping_rl_critic_seed270705_20260803.pth")
    print(f"saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
