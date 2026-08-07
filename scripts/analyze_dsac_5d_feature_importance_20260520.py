#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/llewyn/crypto-scalping")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_env(args: argparse.Namespace) -> None:
    os.environ["DSAC_ALPHA5_V2_STATE_ENABLE"] = "1"
    os.environ["DSAC_V2_MULTI_ACTION_ENABLE"] = "1"
    os.environ["DSAC_ALL_FEATURES_ENABLE"] = "1" if bool(args.all_features_enable) else "0"
    os.environ["DSAC_EXTRA_PCA_ENABLE"] = "1" if bool(args.extra_pca_enable) else "0"
    os.environ["DSAC_EXTRA_PCA_COMPONENTS"] = str(int(args.extra_pca_components))
    os.environ["DSAC_RECURRENT_ENABLE"] = "0"
    os.environ["DSAC_ATTN_STACK_ENABLE"] = "1"
    os.environ["DSAC_STACK_N"] = "2"
    os.environ.setdefault("DSAC_V2_TP_MIN", "0.0025")
    os.environ.setdefault("DSAC_V2_TP_MAX", "0.0220")
    os.environ.setdefault("DSAC_V2_SL_MIN", "0.0015")
    os.environ.setdefault("DSAC_V2_SL_MAX", "0.0160")


BASE_29 = [
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
    "entry_distance_norm",
    "drawdown_norm",
]
BASE_32 = [
    *BASE_29,
    "liq_distance_norm",
    "margin_buffer_norm",
    "funding_phase_norm",
]


def _group(name: str) -> str:
    if name.startswith("m7_"):
        return "m7"
    if name.startswith("a5dir_"):
        return "catboost_router"
    if name in {
        "whale_retail_ratio_norm",
        "funding_pressure_norm",
        "funding_abs_norm",
        "net_taker_ratio_norm",
        "execution_quality",
    }:
        return "catboost_major"
    if name.startswith("market_state_"):
        return "market_state"
    if name.startswith("pca_"):
        return "pca_all_features"
    if name in {"liq_distance_norm", "margin_buffer_norm", "funding_phase_norm"}:
        return "futures_state"
    if name in {"current_position", "unrealized_pnl_norm", "time_in_trade_norm", "hold_vs_expected", "entry_distance_norm", "drawdown_norm", "current_leverage_norm", "current_exposure_norm"}:
        return "position_state"
    if name in {"spread_norm", "rogers_satchell_vol_norm", "micro_vol5_norm", "amihud_norm", "smart_money_flow_norm", "taker_acceleration_norm"}:
        return "microstructure"
    if name.startswith("mtf_"):
        return "mtf"
    return "other"


def _base_feature_names(dsac: Any, pca_meta: dict[str, Any] | None, base_dim: int) -> list[str]:
    market = [c.replace("market_state_2024_unsup_v5_", "market_state_") for c in dsac.DSAC_MARKET_STATE_COLS]
    alpha5_extra = [
        "current_leverage_norm",
        "current_exposure_norm",
        "a5dir_available",
        "a5dir_none_prob",
        "a5dir_long_prob",
        "a5dir_short_prob",
        "a5dir_prob_max",
        "a5dir_edge",
        "a5dir_margin",
        "a5dir_whipsaw_prob",
        "whale_retail_ratio_norm",
        "funding_pressure_norm",
        "funding_abs_norm",
        "net_taker_ratio_norm",
        "execution_quality",
    ]
    if pca_meta and pca_meta.get("enabled"):
        pca_n = int(pca_meta.get("output_dim") or pca_meta.get("components") or 0)
    else:
        pca_n = int(getattr(dsac, "DSAC_ALL_FEATURES_OUT_DIM", 0))
    tail_len = len(market) + len(alpha5_extra) + pca_n
    compact_len = int(base_dim) - tail_len
    if compact_len == len(BASE_32):
        compact = BASE_32
    elif compact_len == len(BASE_29):
        compact = BASE_29
    else:
        compact = [f"compact_{i:02d}" for i in range(max(compact_len, 0))]
    return compact + market + alpha5_extra + [f"pca_{i:02d}" for i in range(pca_n)]


def _pca_top_loadings(pca_meta: dict[str, Any] | None, top_k: int = 8) -> dict[str, list[dict[str, float | str]]]:
    if not pca_meta or not pca_meta.get("enabled"):
        return {}
    cols = list(pca_meta.get("input_cols") or [])
    comps_raw = pca_meta.get("components")
    comps = np.asarray([] if comps_raw is None else comps_raw, dtype=np.float64)
    if comps.ndim != 2 or not cols:
        return {}
    out: dict[str, list[dict[str, float | str]]] = {}
    for i in range(min(comps.shape[0], 64)):
        row = comps[i]
        idxs = np.argsort(np.abs(row))[::-1][:top_k]
        out[f"pca_{i:02d}"] = [{"feature": str(cols[j]), "loading": float(row[j])} for j in idxs if j < len(cols)]
    return out


def _load_frame(csv_path: Path, limit: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if limit > 0:
        df = df.iloc[:limit].copy()
    return df.reset_index(drop=True)


def _collect_states(env: Any, actor: Any, device: str, max_steps: int, sample_every: int) -> tuple[np.ndarray, np.ndarray]:
    state = env.reset()
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    done = False
    step = 0
    while not done and step < max_steps:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        with torch.no_grad():
            a = actor.deterministic(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0))
        av = a.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if step % max(1, sample_every) == 0:
            states.append(s.copy())
            actions.append(av.copy())
        state, _, done, _info = env.step(av)
        step += 1
    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def _score_tensors(actor: Any, critic: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    action = actor.deterministic(x)
    q1, q2 = critic(x, action)
    q = 0.5 * (q1.mean(dim=1, keepdim=True) + q2.mean(dim=1, keepdim=True))
    return action, q


def _aggregate_importance(values: np.ndarray, base_dim: int, names: list[str], prefix: str) -> pd.DataFrame:
    rows = []
    stacks = max(1, values.shape[0] // base_dim)
    for i, name in enumerate(names):
        idxs = [s * base_dim + i for s in range(stacks)]
        v = float(np.nansum(values[idxs]))
        rows.append({"feature": name, "group": _group(name), prefix: v})
    return pd.DataFrame(rows)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    _configure_env(args)
    from ensemble import train_rl_dsac_agent as dsac

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    pca_meta = ckpt.get("extra_feature_pca_meta")
    base_dim = int(ckpt.get("base_state_dim") or dsac.DSAC_STATE_DIM)
    state_dim = int(ckpt.get("state_dim") or dsac.DSAC_MODEL_STATE_DIM)
    action_dim = int(ckpt.get("action_dim") or dsac.DSAC_ACTION_DIM)
    base_names = _base_feature_names(dsac, pca_meta if isinstance(pca_meta, dict) else None, base_dim)
    if len(base_names) != base_dim:
        raise RuntimeError(f"feature name count {len(base_names)} != base_dim {base_dim}")

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    actor = dsac.GaussianActor(state_dim=state_dim, action_dim=action_dim).to(device)
    critic = dsac.DistributionalTwinCritic(state_dim=state_dim, action_dim=action_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    actor.eval()
    critic.eval()

    df = _load_frame(Path(args.csv), args.limit)
    split_idx = int(len(df) * float(args.train_ratio))
    eval_df = df.iloc[split_idx:].reset_index(drop=True)
    if args.eval_limit > 0:
        eval_df = eval_df.iloc[: args.eval_limit].copy()
    env = dsac.DSACCompactTradingEnv(
        eval_df,
        phase="val",
        side_mode="both",
        event_entry_filter_enable=True,
        event_prob_prefix=args.event_prefix,
        event_min_prob=0.02,
        event_min_edge=0.0,
        event_prob_gap=0.0,
        event_debounce_bars=1,
        event_fallback_bars=2,
        event_fallback_min_abs_action=0.01,
        event_fallback_quality_min=-0.05,
        event_fallback_prob_floor=0.01,
        extra_feature_pca_meta=pca_meta if isinstance(pca_meta, dict) else None,
    )
    states_np, actions_np = _collect_states(env, actor, device, args.max_steps, args.sample_every)
    if states_np.size == 0:
        raise RuntimeError("no states collected")
    if len(states_np) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        take = np.sort(rng.choice(len(states_np), size=args.max_samples, replace=False))
        states_np = states_np[take]
        actions_np = actions_np[take]

    x = torch.tensor(states_np, dtype=torch.float32, device=device, requires_grad=True)
    action, q = _score_tensors(actor, critic, x)
    action_abs_sum = action.abs().sum()
    q_sum = q.sum()
    actor_grad = torch.autograd.grad(action_abs_sum, x, retain_graph=True)[0].detach().abs().cpu().numpy()
    critic_grad = torch.autograd.grad(q_sum, x, retain_graph=False)[0].detach().abs().cpu().numpy()
    state_abs = np.abs(states_np)
    actor_sal = (actor_grad * state_abs).mean(axis=0)
    critic_sal = (critic_grad * state_abs).mean(axis=0)

    with torch.no_grad():
        base_action, base_q = _score_tensors(actor, critic, torch.tensor(states_np, dtype=torch.float32, device=device))
        base_action_np = base_action.detach().cpu().numpy()
        base_q_np = base_q.detach().cpu().numpy()

    perm_action_delta = np.zeros((state_dim,), dtype=np.float64)
    perm_q_delta = np.zeros((state_dim,), dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    for j in range(state_dim):
        perm = states_np.copy()
        perm[:, j] = perm[rng.permutation(len(perm)), j]
        with torch.no_grad():
            pa, pq = _score_tensors(actor, critic, torch.tensor(perm, dtype=torch.float32, device=device))
        pa_np = pa.detach().cpu().numpy()
        pq_np = pq.detach().cpu().numpy()
        perm_action_delta[j] = float(np.mean(np.abs(pa_np - base_action_np)))
        perm_q_delta[j] = float(abs(np.mean(pq_np - base_q_np)))

    frames = [
        _aggregate_importance(actor_sal, base_dim, base_names, "actor_grad_x_value"),
        _aggregate_importance(critic_sal, base_dim, base_names, "critic_grad_x_value"),
        _aggregate_importance(perm_action_delta, base_dim, base_names, "perm_action_delta"),
        _aggregate_importance(perm_q_delta, base_dim, base_names, "perm_q_delta"),
    ]
    out_df = frames[0]
    for f in frames[1:]:
        out_df = out_df.merge(f, on=["feature", "group"], how="outer")
    metric_cols = ["actor_grad_x_value", "critic_grad_x_value", "perm_action_delta", "perm_q_delta"]
    for col in metric_cols:
        mx = float(out_df[col].max()) if len(out_df) else 0.0
        out_df[f"{col}_norm"] = out_df[col] / mx if mx > 0 else 0.0
    out_df["combined_importance"] = (
        0.35 * out_df["actor_grad_x_value_norm"]
        + 0.35 * out_df["critic_grad_x_value_norm"]
        + 0.20 * out_df["perm_action_delta_norm"]
        + 0.10 * out_df["perm_q_delta_norm"]
    )
    out_df = out_df.sort_values("combined_importance", ascending=False).reset_index(drop=True)

    group_df = (
        out_df.groupby("group", as_index=False)
        .agg(
            feature_count=("feature", "count"),
            combined_importance_sum=("combined_importance", "sum"),
            combined_importance_mean=("combined_importance", "mean"),
            actor_saliency_sum=("actor_grad_x_value", "sum"),
            critic_saliency_sum=("critic_grad_x_value", "sum"),
            perm_action_delta_sum=("perm_action_delta", "sum"),
        )
        .sort_values("combined_importance_sum", ascending=False)
    )

    action_cols = ["signed_exposure", "exit_pressure", "leverage", "tp", "sl"][: base_action_np.shape[1]]
    action_stats = {
        name: {
            "mean": float(base_action_np[:, i].mean()),
            "std": float(base_action_np[:, i].std()),
            "p05": float(np.percentile(base_action_np[:, i], 5)),
            "p50": float(np.percentile(base_action_np[:, i], 50)),
            "p95": float(np.percentile(base_action_np[:, i], 95)),
        }
        for i, name in enumerate(action_cols)
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_dir / "feature_importance.csv", index=False)
    group_df.to_csv(args.out_dir / "group_importance.csv", index=False)
    pca_loadings = _pca_top_loadings(pca_meta if isinstance(pca_meta, dict) else None)
    (args.out_dir / "pca_top_loadings.json").write_text(json.dumps(pca_loadings, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "csv": str(args.csv),
        "ckpt": str(args.ckpt),
        "eval_rows": int(len(eval_df)),
        "sampled_states": int(len(states_np)),
        "base_dim": int(base_dim),
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "checkpoint": {k: ckpt.get(k) for k in ["state_schema", "best_score", "episode", "action_dim", "state_dim", "base_state_dim"]},
        "action_stats": action_stats,
        "top_features": out_df.head(args.top_k).to_dict(orient="records"),
        "group_importance": group_df.to_dict(orient="records"),
        "pca_top_loadings": {k: pca_loadings[k] for k in list(pca_loadings)[: min(10, len(pca_loadings))]},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze 5D alpha5 DSAC feature importance via actor/critic saliency and permutation sensitivity.")
    p.add_argument("--csv", default=str(ROOT / "tmp/causal_regen_20260516/alpha5_direction_router_rl_20260519/rl_training_2025_direction_router.csv"))
    p.add_argument("--ckpt", default=str(ROOT / "tmp/causal_regen_20260516/alpha5_dsac_single_router5_5d_exit_v1/best.pth"))
    p.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/dsac_5d_feature_importance_20260520")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=12000)
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--sample-every", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=1200)
    p.add_argument("--event-prefix", default="a5dir")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--all-features-enable", type=int, choices=[0, 1], default=1)
    p.add_argument("--extra-pca-enable", type=int, choices=[0, 1], default=1)
    p.add_argument("--extra-pca-components", type=int, default=32)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = analyze(args)
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "sampled_states": summary["sampled_states"],
        "top_features": summary["top_features"][:10],
        "group_importance": summary["group_importance"][:8],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
