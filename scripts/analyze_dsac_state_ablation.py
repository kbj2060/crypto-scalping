#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ensemble.train_rl_dsac_agent import DSACCompactTradingEnv, DSAC_STATE_DIM, GaussianActor  # noqa: E402


STATE_NAMES = [
    "m7_prob_up_scaled",
    "m7_prob_dn_scaled",
    "m7_trend_entropy_scaled",
    "m7_quality_norm",
    "m7_hold_norm",
    "m7_q_mid_norm",
    "m7_q_uncertainty_norm",
    "m7_q_skew",
    "m7_gmm_cluster_norm",
    "m7_gmm_conf",
    "m7_gmm_vol_rank",
    "m7_anomaly_score",
    "m7_tp_offset_norm",
    "m7_sl_offset_norm",
    "mtf_trend_1h_norm",
    "mtf_trend_4h_norm",
    "spread_norm",
    "rogers_satchell_vol_norm",
    "micro_vol5_norm",
    "amihud_norm",
    "smart_money_flow_norm",
    "taker_acceleration_norm",
    "current_position",
    "unrealized_pnl_norm",
    "time_in_trade_norm",
    "hold_vs_expected",
    "margin_usage",
    "drawdown_norm",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DSAC 29D compact-state ablation and saliency analysis")
    p.add_argument("--csv", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sample-step", type=int, default=20)
    return p.parse_args()


def load_actor(path: str, device: str) -> GaussianActor:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor = GaussianActor(state_dim=int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM)).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def rollout(df: pd.DataFrame, actor: GaussianActor, device: str, mask_dim: int | None = None) -> dict:
    env = DSACCompactTradingEnv(df=df, phase="eval")
    state = env.reset()
    done = False
    actions = []
    states = []
    pbar = tqdm(total=max(len(df) - 1, 1), desc=("baseline" if mask_dim is None else f"mask-{mask_dim:02d}"), leave=False)
    while not done:
        s = np.asarray(state, dtype=np.float32).copy()
        if mask_dim is not None:
            s[mask_dim] = 0.0
        states.append(s.copy())
        with torch.no_grad():
            a = actor.deterministic(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0))
        action = float(a.detach().cpu().item())
        actions.append(action)
        state, _, done, _ = env.step(action)
        pbar.update(1)
    pbar.close()
    arr = np.asarray(getattr(env, "equity_curve", []) or [1.0], dtype=np.float64)
    if arr.size <= 1:
        pnl_pct = float((getattr(env, "balance", 1.0) / max(getattr(env, "initial_balance", 1.0), 1e-12) - 1.0) * 100.0)
        mdd_pct = 0.0
    else:
        pnl_pct = float((arr[-1] / max(arr[0], 1e-12) - 1.0) * 100.0)
        peak = np.maximum.accumulate(arr)
        mdd_pct = float(np.min(arr / np.maximum(peak, 1e-12) - 1.0) * 100.0)
    trades = int(getattr(env, "trade_count", 0))
    wins = int(getattr(env, "win_count", 0))
    return {
        "pnl_pct": pnl_pct,
        "mdd_pct": mdd_pct,
        "trades": trades,
        "wr_pct": float(wins / trades * 100.0) if trades else 0.0,
        "action_mean": float(np.mean(actions)) if actions else 0.0,
        "action_std": float(np.std(actions)) if actions else 0.0,
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
    }


def saliency(actor: GaussianActor, states: np.ndarray, device: str, sample_step: int) -> list[dict]:
    if states.size == 0:
        return []
    sampled = states[:: max(1, int(sample_step))]
    x = torch.tensor(sampled, dtype=torch.float32, device=device, requires_grad=True)
    y = actor.deterministic(x).sum()
    y.backward()
    grad = x.grad.detach().abs().cpu().numpy()
    val = np.abs(sampled)
    grad_mean = grad.mean(axis=0)
    grad_x_val = (grad * val).mean(axis=0)
    rows = []
    for i, name in enumerate(STATE_NAMES):
        rows.append(
            {
                "idx": i,
                "feature": name,
                "grad_abs_mean": float(grad_mean[i]),
                "grad_x_abs_value_mean": float(grad_x_val[i]),
                "state_abs_mean": float(val[:, i].mean()),
            }
        )
    return sorted(rows, key=lambda r: r["grad_x_abs_value_mean"], reverse=True)


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.csv)
    if args.limit > 0:
        df = df.iloc[: args.limit].copy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor = load_actor(args.ckpt, device)

    base = rollout(df, actor, device, None)
    base_states = base.pop("states")
    base.pop("actions", None)

    ablations = []
    for i, name in enumerate(STATE_NAMES):
        r = rollout(df, actor, device, i)
        r.pop("states", None)
        r.pop("actions", None)
        r["idx"] = i
        r["feature"] = name
        r["delta_pnl_pct"] = float(r["pnl_pct"] - base["pnl_pct"])
        r["delta_mdd_pct"] = float(r["mdd_pct"] - base["mdd_pct"])
        ablations.append(r)

    sal = saliency(actor, base_states, device, args.sample_step)
    sal_by_feature = {r["feature"]: r for r in sal}
    for r in ablations:
        r.update({k: v for k, v in sal_by_feature.get(r["feature"], {}).items() if k not in {"idx", "feature"}})

    out = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "csv": args.csv,
        "ckpt": args.ckpt,
        "rows": int(len(df)),
        "state_dim": int(DSAC_STATE_DIM),
        "state_names": STATE_NAMES,
        "baseline": base,
        "ablations_by_delta_pnl_desc": sorted(ablations, key=lambda r: r["delta_pnl_pct"], reverse=True),
        "ablations_by_delta_pnl_asc": sorted(ablations, key=lambda r: r["delta_pnl_pct"]),
        "saliency_desc": sal,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: out[k] for k in ["csv", "ckpt", "rows", "baseline"]}, ensure_ascii=False, indent=2))
    print("top_mask_improves")
    for r in out["ablations_by_delta_pnl_desc"][:10]:
        print(r["idx"], r["feature"], "delta", round(r["delta_pnl_pct"], 4), "pnl", round(r["pnl_pct"], 4))
    print("top_saliency")
    for r in out["saliency_desc"][:10]:
        print(r["idx"], r["feature"], "g*v", round(r["grad_x_abs_value_mean"], 6))
    print(f"saved={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
