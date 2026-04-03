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
    DSAC_STATE_DIM,
    DSACCompactTradingEnv,
    GaussianActor,
)
from ensemble.train_rl_dsac_long_agent import (  # noqa: E402
    _POS_THRESH as LONG_POS_THRESH,
    _CLOSE_THRESH as LONG_CLOSE_THRESH,
    STATE_DIM as LONG_STATE_DIM,
    LongSpecialistEnv,
    SigmoidActor as LongSigmoidActor,
)
from ensemble.train_rl_dsac_short_agent import (  # noqa: E402
    _POS_THRESH as SHORT_POS_THRESH,
    _CLOSE_THRESH as SHORT_CLOSE_THRESH,
    STATE_DIM as SHORT_STATE_DIM,
    ShortSpecialistEnv,
    SigmoidActor as ShortSigmoidActor,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze specialist DSAC reward/action diagnostics on a CSV")
    p.add_argument("--csv-path", required=True)
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--output-path", default="")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def _load_ckpt(ckpt_path: str, device: str) -> tuple[torch.nn.Module, dict, str, int]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    side_mode = str(meta.get("side") or meta.get("side_mode") or "").strip().lower()
    algo = str(meta.get("algo", "")).strip().upper()
    if not side_mode:
        if "LONG" in algo:
            side_mode = "long"
        elif "SHORT" in algo:
            side_mode = "short"
        elif "long" in os.path.basename(ckpt_path).lower():
            side_mode = "long"
        elif "short" in os.path.basename(ckpt_path).lower():
            side_mode = "short"
        else:
            side_mode = "both"
    if side_mode == "long":
        state_dim = int(ckpt.get("state_dim", LONG_STATE_DIM) or LONG_STATE_DIM)
        actor = LongSigmoidActor(state_dim=state_dim).to(device)
    elif side_mode == "short":
        state_dim = int(ckpt.get("state_dim", SHORT_STATE_DIM) or SHORT_STATE_DIM)
        actor = ShortSigmoidActor(state_dim=state_dim).to(device)
    else:
        state_dim = int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM)
        actor = GaussianActor(state_dim=state_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, meta, side_mode, state_dim


def _map_action(raw_action: float, side_mode: str) -> float:
    raw_action = float(raw_action)
    if side_mode == "long":
        return float(np.clip(raw_action, 0.0, 1.0))
    if side_mode == "short":
        return float(np.clip(raw_action, 0.0, 1.0))
    return raw_action


def analyze(csv_path: str, ckpt_path: str, limit: int = 0) -> dict:
    df = pd.read_csv(csv_path)
    if limit > 0:
        df = df.iloc[:limit].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor, meta, side_mode, state_dim = _load_ckpt(ckpt_path, device)
    reward_beta = meta.get("reward_beta", None)
    specialist_pos_thresh = meta.get("specialist_pos_thresh", None)
    specialist_close_thresh = meta.get("specialist_close_thresh", None)
    if side_mode == "long":
        env = LongSpecialistEnv(df=df, phase="eval")
    elif side_mode == "short":
        env = ShortSpecialistEnv(df=df, phase="eval")
    else:
        env = DSACCompactTradingEnv(
            df=df,
            phase="eval",
            side_mode=side_mode,
            reward_beta=reward_beta,
            specialist_pos_thresh=specialist_pos_thresh,
            specialist_close_thresh=specialist_close_thresh,
        )

    pos_thresh = float(getattr(
        env,
        "specialist_pos_thresh",
        LONG_POS_THRESH if side_mode == "long" else (SHORT_POS_THRESH if side_mode == "short" else 0.15),
    ))
    close_thresh = float(getattr(
        env,
        "specialist_close_thresh",
        LONG_CLOSE_THRESH if side_mode == "long" else (SHORT_CLOSE_THRESH if side_mode == "short" else 0.05),
    ))

    state = env.reset()
    done = False

    reward_keys = ["r1_pnl", "r2_drawdown", "r3_quality", "r4_time_decay", "r5_idle", "r6_trade_cost", "raw_reward"]
    reward_all = {k: [] for k in reward_keys}
    reward_enter = {k: [] for k in reward_keys}
    reward_flat = {k: [] for k in reward_keys}
    raw_actions: list[float] = []
    mapped_actions: list[float] = []
    flat_open_hits = 0
    flat_close_hits = 0
    flat_steps = 0
    long_entries = 0
    short_entries = 0

    while not done:
        prev_pos = env.pos
        state_ts = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            raw = float(actor.deterministic(state_ts).cpu().item())
        mapped = _map_action(raw, side_mode)
        raw_actions.append(raw)
        mapped_actions.append(mapped)

        if prev_pos is None:
            flat_steps += 1
            if mapped > pos_thresh or mapped < -pos_thresh:
                flat_open_hits += 1
            if abs(mapped) < close_thresh:
                flat_close_hits += 1

        next_state, _, done, info = env.step(mapped)
        for k in reward_keys:
            val = _safe_float(info.get(k, 0.0), 0.0)
            reward_all[k].append(val)
            if prev_pos is None:
                reward_flat[k].append(val)
            if info.get("entered_long") or info.get("entered_short"):
                reward_enter[k].append(val)
        if info.get("entered_long"):
            long_entries += 1
        if info.get("entered_short"):
            short_entries += 1
        state = next_state

    def _summ(v: list[float]) -> dict[str, float]:
        arr = np.asarray(v, dtype=np.float64)
        if arr.size == 0:
            return {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }

    feature_summary = {}
    for col in ["m7_trend_xgb_up", "m7_trend_xgb_dn", "m7_q50", "m7_quality_pred"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
            if vals.size:
                feature_summary[col] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "p10": float(np.percentile(vals, 10)),
                    "p90": float(np.percentile(vals, 90)),
                }

    reward_beta_value = (
        float(_safe_float(reward_beta, 1.0))
        if side_mode in ("long", "short")
        else float(_safe_float(reward_beta, env._reward_beta()))
    )
    reward_mode_name = (
        f"specialist_{side_mode}"
        if side_mode in ("long", "short")
        else str(env._reward_mode_name())
    )

    result = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "csv_path": csv_path,
        "ckpt_path": ckpt_path,
        "rows": int(len(df)),
        "side_mode": side_mode,
        "state_dim": int(state_dim),
        "reward_beta": reward_beta_value,
        "specialist_pos_thresh": float(pos_thresh),
        "specialist_close_thresh": float(close_thresh),
        "reward_mode": reward_mode_name,
        "final_balance": float(env.balance),
        "total_return_pct": float((env.balance / env.initial_balance - 1.0) * 100.0),
        "total_trades": int(env.total_trades),
        "win_rate": float(env.win_rate),
        "entries": {
            "long": int(long_entries),
            "short": int(short_entries),
        },
        "action_stats": {
            "raw": _summ(raw_actions),
            "mapped": _summ(mapped_actions),
            "flat_open_hit_rate": float(flat_open_hits / max(flat_steps, 1)),
            "flat_close_hit_rate": float(flat_close_hits / max(flat_steps, 1)),
            "flat_steps": int(flat_steps),
        },
        "reward_components": {
            "all_steps": {k: _summ(v) for k, v in reward_all.items()},
            "flat_steps": {k: _summ(v) for k, v in reward_flat.items()},
            "entry_steps": {k: _summ(v) for k, v in reward_enter.items()},
        },
        "feature_summary": feature_summary,
    }
    return result


def main() -> int:
    args = parse_args()
    result = analyze(args.csv_path, args.ckpt_path, args.limit)

    output_path = args.output_path.strip()
    if not output_path:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/ensemble/metrics/dsac_specialist_diag_{stamp}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
