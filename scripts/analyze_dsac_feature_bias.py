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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.train_rl_dsac_agent import (  # noqa: E402
    _POS_THRESH,
    DSAC_STATE_DIM,
    DSACCompactTradingEnv,
    GaussianActor,
)


STATE_NAMES = [
    "up",
    "dn",
    "fl",
    "trend_gap",
    "trend_entropy",
    "quality_norm",
    "hold_norm",
    "q_mid_norm",
    "q_uncertainty_norm",
    "q_skew",
    "gmm_cluster_norm",
    "gmm_conf",
    "vol_rank",
    "anomaly_score",
    "entry_long_norm",
    "entry_short_norm",
    "tp_offset_norm",
    "sl_offset_norm",
    "spread_norm",
    "spread_z_norm",
    "micro5_norm",
    "micro10_norm",
    "ret1_norm",
    "ret3_norm",
    "current_position",
    "unrealized_norm",
    "time_in_trade_norm",
    "hold_vs_expected",
    "margin_usage",
    "drawdown_norm",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze DSAC raw-action bias vs features")
    p.add_argument("--csv-path", required=True)
    p.add_argument("--ckpt-path", default="data/ensemble/ckpt/best_dsac_agents.pth")
    p.add_argument("--output-path", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--top-k", type=int, default=15)
    return p.parse_args()


def _load_actor(ckpt_path: str, device: str) -> GaussianActor:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor = GaussianActor(state_dim=int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM)).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def _corr_pairs(df: pd.DataFrame, target_col: str, top_k: int) -> tuple[list[dict], list[dict]]:
    corrs = []
    tgt = pd.to_numeric(df[target_col], errors="coerce")
    for col in df.columns:
        if col == target_col:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        mask = np.isfinite(tgt.to_numpy(dtype=np.float64)) & np.isfinite(s.to_numpy(dtype=np.float64))
        if int(mask.sum()) < 50:
            continue
        if float(np.nanstd(s.to_numpy(dtype=np.float64)[mask])) < 1e-10:
            continue
        corr = float(np.corrcoef(tgt.to_numpy(dtype=np.float64)[mask], s.to_numpy(dtype=np.float64)[mask])[0, 1])
        if np.isfinite(corr):
            corrs.append({"feature": col, "corr": corr})
    pos = sorted(corrs, key=lambda x: x["corr"], reverse=True)[:top_k]
    neg = sorted(corrs, key=lambda x: x["corr"])[:top_k]
    return pos, neg


def analyze(csv_path: str, ckpt_path: str, limit: int = 0, top_k: int = 15) -> dict:
    df = pd.read_csv(csv_path)
    if limit > 0:
        df = df.iloc[:limit].copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor = _load_actor(ckpt_path, device)
    env = DSACCompactTradingEnv(df=df, phase="eval")

    state = env.reset()
    done = False

    flat_rows = []
    flat_states = []
    flat_actions = []

    while not done:
        idx = int(env.current_step)
        if env.pos is None:
            state_vec = np.asarray(state, dtype=np.float32)
            state_ts = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor.deterministic(state_ts)
            raw_action = float(action.cpu().item())
            flat_rows.append(df.iloc[idx].copy())
            flat_states.append(state_vec.copy())
            flat_actions.append(raw_action)
            state, _, done, _ = env.step(raw_action)
        else:
            state_ts = torch.tensor(np.asarray(state, dtype=np.float32), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor.deterministic(state_ts)
            raw_action = float(action.cpu().item())
            state, _, done, _ = env.step(raw_action)

    flat_df = pd.DataFrame(flat_rows).reset_index(drop=True)
    state_df = pd.DataFrame(np.asarray(flat_states, dtype=np.float32), columns=STATE_NAMES)
    flat_df["raw_action"] = np.asarray(flat_actions, dtype=np.float64)
    merged = pd.concat([flat_df.reset_index(drop=True), state_df.reset_index(drop=True)], axis=1)

    long_mask = merged["raw_action"] > float(_POS_THRESH)
    short_mask = merged["raw_action"] < -float(_POS_THRESH)
    hold_mask = ~(long_mask | short_mask)

    state_pos, state_neg = _corr_pairs(pd.concat([state_df, merged[["raw_action"]]], axis=1), "raw_action", top_k)

    numeric_cols = []
    for c in merged.columns:
        if c in {"timestamp", "raw_action"} or c in STATE_NAMES:
            continue
        if pd.api.types.is_numeric_dtype(merged[c]):
            numeric_cols.append(c)
    raw_df = merged[numeric_cols + ["raw_action"]].copy()
    raw_pos, raw_neg = _corr_pairs(raw_df, "raw_action", top_k)

    long_short_shift = []
    if int(long_mask.sum()) >= 50 and int(short_mask.sum()) >= 50:
        for col in numeric_cols:
            s = pd.to_numeric(merged[col], errors="coerce")
            l = s[long_mask]
            sh = s[short_mask]
            if l.notna().sum() < 50 or sh.notna().sum() < 50:
                continue
            l_mean = float(l.mean())
            s_mean = float(sh.mean())
            diff = l_mean - s_mean
            if not np.isfinite(diff):
                continue
            long_short_shift.append(
                {
                    "feature": col,
                    "long_mean": l_mean,
                    "short_mean": s_mean,
                    "diff": diff,
                }
            )
    long_short_shift = sorted(long_short_shift, key=lambda x: abs(x["diff"]), reverse=True)[:top_k]

    result = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "csv_path": csv_path,
        "ckpt_path": ckpt_path,
        "rows": int(len(df)),
        "flat_rows": int(len(merged)),
        "state_dim": int(DSAC_STATE_DIM),
        "flat_action_distribution": {
            "mean": float(merged["raw_action"].mean()),
            "median": float(merged["raw_action"].median()),
            "long_ratio": float(long_mask.mean()),
            "short_ratio": float(short_mask.mean()),
            "hold_ratio": float(hold_mask.mean()),
            "long_short_ratio": float(long_mask.sum() / max(int(short_mask.sum()), 1)),
        },
        "top_state_positive_corr": state_pos,
        "top_state_negative_corr": state_neg,
        "top_raw_feature_positive_corr": raw_pos,
        "top_raw_feature_negative_corr": raw_neg,
        "top_long_short_feature_shifts": long_short_shift,
    }
    return result


def main() -> int:
    args = parse_args()
    result = analyze(args.csv_path, args.ckpt_path, args.limit, args.top_k)
    output_path = args.output_path.strip()
    if not output_path:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/ensemble/metrics/dsac_feature_bias_{stamp}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
