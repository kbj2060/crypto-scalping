#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from alpha5_direction_router_score_rl_csv_20260519 import (
    DEFAULT_OUT_DIR as DEFAULT_ROUTER_OUT_DIR,
    score_router_frame,
)
from ensemble.train_rl_dsac_agent import train


ROOT = Path("/home/llewyn/crypto-scalping")
DEFAULT_RL_2025 = ROOT / "data/rl_training_2025_unified.csv"
DEFAULT_RL_2026 = ROOT / "data/rl_training_2026_unified.csv"
DEFAULT_SPECIALIST_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_dsac_direction_specialists_20260519"

SIDE_PROFILES = {
    "long": {
        "side_mode_override": "long",
        "specialist_pos_thresh": 0.17,
        "specialist_close_thresh": 0.055,
        "min_val_trades_for_best": 20,
        "target_val_trades": 40,
        "event_min_prob": 0.58,
        "event_min_edge": 0.10,
        "event_prob_gap": 0.08,
        "event_debounce_bars": 2,
        "event_fallback_bars": 4,
        "event_fallback_min_abs_action": 0.24,
        "event_fallback_quality_min": 0.40,
        "event_fallback_prob_floor": 0.34,
    },
    "short": {
        "side_mode_override": "short",
        "specialist_pos_thresh": 0.19,
        "specialist_close_thresh": 0.065,
        "min_val_trades_for_best": 22,
        "target_val_trades": 44,
        "event_min_prob": 0.60,
        "event_min_edge": 0.12,
        "event_prob_gap": 0.10,
        "event_debounce_bars": 3,
        "event_fallback_bars": 5,
        "event_fallback_min_abs_action": 0.26,
        "event_fallback_quality_min": 0.42,
        "event_fallback_prob_floor": 0.36,
    },
}


def _ensure_router_csv(input_csv: Path, output_csv: Path, prefix: str) -> dict:
    if output_csv.exists():
        summary_path = output_csv.with_suffix(".router_summary.json")
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
    return score_router_frame(input_csv=input_csv, output_csv=output_csv, prefix=prefix)


def _specialist_paths(base_dir: Path, side: str) -> dict[str, Path]:
    d = base_dir / side
    d.mkdir(parents=True, exist_ok=True)
    return {
        "dir": d,
        "config_json_path": d / "train_config.json",
        "checkpoint_path": d / "checkpoint.pth",
        "best_path": d / "best.pth",
        "summary_path": d / "summary.json",
        "log_path": d / "run.log",
    }


def run_specialist(
    side: str,
    csv_path: Path,
    out_dir: Path,
    episodes: int,
    fresh_start: bool,
    device: str,
    prefix: str,
    smoke: bool = False,
) -> dict:
    if side not in SIDE_PROFILES:
        raise ValueError(f"invalid side: {side}")
    profile = dict(SIDE_PROFILES[side])
    paths = _specialist_paths(out_dir, side)
    batch_size = 64
    update_freq = 2
    min_buffer = 128 if smoke else 1024
    warmup_steps = 128 if smoke else 2048
    val_interval = 2 if smoke else 5
    early_stop = 4 if smoke else 12
    run_started_at = datetime.now().isoformat()
    train(
        csv_path=str(csv_path),
        train_ratio=0.8,
        episodes=int(episodes),
        fresh_start=bool(fresh_start),
        val_interval=int(val_interval),
        early_stop_patience=int(early_stop),
        min_val_trades_for_best=int(profile["min_val_trades_for_best"]),
        target_val_trades=int(profile["target_val_trades"]),
        val_trade_shortfall_penalty=18.0,
        val_trade_bonus_cap=6.0,
        val_side_bias_penalty=0.0,
        config_json_path=str(paths["config_json_path"]),
        checkpoint_path=str(paths["checkpoint_path"]),
        best_path=str(paths["best_path"]),
        side_mode_override=str(profile["side_mode_override"]),
        specialist_pos_thresh=float(profile["specialist_pos_thresh"]),
        specialist_close_thresh=float(profile["specialist_close_thresh"]),
        batch_size=int(batch_size),
        update_freq=int(update_freq),
        min_buffer=int(min_buffer),
        warmup_steps=int(warmup_steps),
        device=str(device),
        skip_focus_segment_filter=True,
        terminate_on_regime_change=False,
        event_entry_filter_enable=True,
        event_prob_prefix=str(prefix),
        event_min_prob=float(profile["event_min_prob"]),
        event_min_edge=float(profile["event_min_edge"]),
        event_prob_gap=float(profile["event_prob_gap"]),
        event_debounce_bars=int(profile["event_debounce_bars"]),
        event_fallback_bars=int(profile["event_fallback_bars"]),
        event_fallback_min_abs_action=float(profile["event_fallback_min_abs_action"]),
        event_fallback_quality_min=float(profile["event_fallback_quality_min"]),
        event_fallback_prob_floor=float(profile["event_fallback_prob_floor"]),
    )
    summary = {
        "side": side,
        "csv_path": str(csv_path),
        "episodes": int(episodes),
        "smoke": bool(smoke),
        "device": str(device),
        "run_started_at": run_started_at,
        "run_finished_at": datetime.now().isoformat(),
        "paths": {k: str(v) for k, v in paths.items()},
        "profile": profile,
    }
    paths["summary_path"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train alpha5 direction-routed DSAC long/short specialists.")
    p.add_argument("--rl-2025", default=str(DEFAULT_RL_2025))
    p.add_argument("--rl-2026", default=str(DEFAULT_RL_2026))
    p.add_argument("--router-dir", default=str(DEFAULT_ROUTER_OUT_DIR))
    p.add_argument("--out-dir", default=str(DEFAULT_SPECIALIST_DIR))
    p.add_argument("--prefix", default="a5dir")
    p.add_argument("--sides", default="long,short")
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--fresh-start", action="store_true")
    p.add_argument("--startup-check-only", action="store_true")
    p.add_argument("--skip-score", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    router_dir = Path(args.router_dir)
    router_dir.mkdir(parents=True, exist_ok=True)
    scored_2025 = router_dir / "rl_training_2025_direction_router.csv"
    scored_2026 = router_dir / "rl_training_2026_direction_router.csv"
    summaries = {}
    if not args.skip_score:
        summaries["score_2025"] = _ensure_router_csv(Path(args.rl_2025), scored_2025, args.prefix)
        summaries["score_2026"] = _ensure_router_csv(Path(args.rl_2026), scored_2026, args.prefix)
    if args.startup_check_only:
        payload = {
            "status": "startup_check_ok",
            "router_csv_2025": str(scored_2025),
            "router_csv_2026": str(scored_2026),
            "score_summaries": summaries,
            "sides": [x.strip() for x in str(args.sides).split(",") if x.strip()],
            "out_dir": str(args.out_dir),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": datetime.now().isoformat(),
        "router_csv_2025": str(scored_2025),
        "router_csv_2026": str(scored_2026),
        "prefix": str(args.prefix),
        "episodes": int(args.episodes),
        "device": str(args.device),
        "smoke": bool(args.smoke),
        "summaries": summaries,
        "specialists": {},
    }
    for side in [x.strip() for x in str(args.sides).split(",") if x.strip()]:
        manifest["specialists"][side] = run_specialist(
            side=side,
            csv_path=scored_2025,
            out_dir=out_dir,
            episodes=2 if args.smoke else int(args.episodes),
            fresh_start=bool(args.fresh_start),
            device=str(args.device),
            prefix=str(args.prefix),
            smoke=bool(args.smoke),
        )
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["finished_at"] = datetime.now().isoformat()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
