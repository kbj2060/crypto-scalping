#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
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
    _CLOSE_THRESH,
    _POS_THRESH,
    DSAC_STATE_DIM,
    DSACCompactTradingEnv,
    GaussianActor,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze DSAC long/short directional bias on a CSV")
    p.add_argument("--csv-path", required=True)
    p.add_argument("--ckpt-path", default="data/ensemble/ckpt/best_dsac_agents.pth")
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


def _load_actor(ckpt_path: str, device: str) -> GaussianActor:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor = GaussianActor(state_dim=int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM)).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def _flat_decision_bucket(action_val: float) -> str:
    if action_val > _POS_THRESH:
        return "flat_to_long"
    if action_val < -_POS_THRESH:
        return "flat_to_short"
    return "flat_hold"


def _regime_label(row: pd.Series) -> str:
    reg_cols = [c for c in row.index if c.startswith("regime_")]
    if not reg_cols:
        return "UNKNOWN"
    vals = {c: _safe_float(row.get(c, 0.0), 0.0) for c in reg_cols}
    best = max(vals.items(), key=lambda kv: kv[1])[0]
    return best.replace("regime_", "").upper()


def _record_nested(counter: dict[str, dict[str, float]], outer: str, inner: str, value: float = 1.0) -> None:
    counter.setdefault(outer, {})
    counter[outer][inner] = float(counter[outer].get(inner, 0.0) + value)


def _finalize_nested(counter: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for outer, inner_map in counter.items():
        total = float(sum(inner_map.values()))
        out[outer] = {k: float(v) for k, v in inner_map.items()}
        if total > 0:
            for k, v in inner_map.items():
                out[outer][f"{k}_ratio"] = float(v / total)
    return out


def analyze(csv_path: str, ckpt_path: str, limit: int = 0) -> dict:
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

    flat_decisions: dict[str, float] = defaultdict(float)
    position_time: dict[str, float] = defaultdict(float)
    month_flat_decisions: dict[str, dict[str, float]] = {}
    regime_flat_decisions: dict[str, dict[str, float]] = {}
    side_trade_pnls: dict[str, list[float]] = {"LONG": [], "SHORT": []}
    side_hold_bars: dict[str, list[int]] = {"LONG": [], "SHORT": []}
    side_entries: dict[str, int] = {"LONG": 0, "SHORT": 0}
    raw_actions: list[float] = []
    current_trade_bars = 0

    while not done:
        idx = int(env.current_step)
        row = df.iloc[idx]
        ts = row["timestamp"] if "timestamp" in df.columns else pd.NaT
        month = str(pd.Timestamp(ts).to_period("M")) if pd.notna(ts) else "UNKNOWN"
        regime = _regime_label(row)

        prev_pos = env.pos
        current_trade_bars = current_trade_bars + 1 if prev_pos is not None else 0
        position_time[prev_pos or "FLAT"] += 1.0

        state_ts = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = actor.deterministic(state_ts)
        action_val = float(action.cpu().item())
        raw_actions.append(action_val)

        if prev_pos is None:
            bucket = _flat_decision_bucket(action_val)
            flat_decisions[bucket] += 1.0
            _record_nested(month_flat_decisions, month, bucket, 1.0)
            _record_nested(regime_flat_decisions, regime, bucket, 1.0)

        next_state, _, done, _ = env.step(action_val)

        if prev_pos is None and env.pos is not None:
            side_entries[env.pos] += 1
            current_trade_bars = 0
        if prev_pos is not None and env.pos is None:
            realized = float(env._last_realized_pnl)
            side_trade_pnls[prev_pos].append(realized)
            side_hold_bars[prev_pos].append(int(current_trade_bars))
            current_trade_bars = 0

        state = next_state

    total_flat = float(sum(flat_decisions.values()))
    total_steps = float(sum(position_time.values()))
    result = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "csv_path": csv_path,
        "ckpt_path": ckpt_path,
        "rows": int(len(df)),
        "state_dim": int(DSAC_STATE_DIM),
        "thresholds": {
            "open": float(_POS_THRESH),
            "close": float(_CLOSE_THRESH),
        },
        "final_balance": float(env.balance),
        "total_return_pct": float((env.balance / env.initial_balance - 1.0) * 100.0),
        "total_trades": int(env.total_trades),
        "win_rate": float(env.win_rate),
        "flat_decisions": {
            "flat_to_long": int(flat_decisions.get("flat_to_long", 0.0)),
            "flat_to_short": int(flat_decisions.get("flat_to_short", 0.0)),
            "flat_hold": int(flat_decisions.get("flat_hold", 0.0)),
            "flat_to_long_ratio": float(flat_decisions.get("flat_to_long", 0.0) / total_flat) if total_flat else 0.0,
            "flat_to_short_ratio": float(flat_decisions.get("flat_to_short", 0.0) / total_flat) if total_flat else 0.0,
            "long_short_entry_ratio": float(flat_decisions.get("flat_to_long", 0.0) / max(flat_decisions.get("flat_to_short", 0.0), 1.0)),
        },
        "position_time_ratio": {
            "LONG": float(position_time.get("LONG", 0.0) / total_steps) if total_steps else 0.0,
            "SHORT": float(position_time.get("SHORT", 0.0) / total_steps) if total_steps else 0.0,
            "FLAT": float(position_time.get("FLAT", 0.0) / total_steps) if total_steps else 0.0,
        },
        "raw_action_stats": {
            "mean": float(np.mean(raw_actions)) if raw_actions else 0.0,
            "median": float(np.median(raw_actions)) if raw_actions else 0.0,
            "p10": float(np.percentile(raw_actions, 10)) if raw_actions else 0.0,
            "p90": float(np.percentile(raw_actions, 90)) if raw_actions else 0.0,
        },
        "by_side": {},
        "monthly_flat_decisions": _finalize_nested(month_flat_decisions),
        "regime_flat_decisions": _finalize_nested(regime_flat_decisions),
    }

    for side in ("LONG", "SHORT"):
        pnls = np.asarray(side_trade_pnls[side], dtype=np.float64)
        holds = np.asarray(side_hold_bars[side], dtype=np.float64)
        result["by_side"][side] = {
            "entries": int(side_entries.get(side, 0)),
            "closed_trades": int(pnls.size),
            "mean_pnl_pct": float(np.mean(pnls) * 100.0) if pnls.size else 0.0,
            "median_pnl_pct": float(np.median(pnls) * 100.0) if pnls.size else 0.0,
            "win_rate": float(np.mean(pnls > 0.0)) if pnls.size else 0.0,
            "avg_hold_bars": float(np.mean(holds)) if holds.size else 0.0,
            "gross_pnl_pct_sum": float(np.sum(pnls) * 100.0) if pnls.size else 0.0,
        }
    return result


def main() -> int:
    args = parse_args()
    result = analyze(args.csv_path, args.ckpt_path, args.limit)

    output_path = args.output_path.strip()
    if not output_path:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/ensemble/metrics/dsac_direction_bias_{stamp}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
