#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import sys

def _set_v2_env() -> None:
    os.environ["DSAC_ALPHA5_V2_STATE_ENABLE"] = "1"
    os.environ["DSAC_V2_MULTI_ACTION_ENABLE"] = "1"
    os.environ["DSAC_V2_TP_MIN"] = "0.0025"
    os.environ["DSAC_V2_TP_MAX"] = "0.0220"
    os.environ["DSAC_V2_SL_MIN"] = "0.0015"
    os.environ["DSAC_V2_SL_MAX"] = "0.0160"
    os.environ["DSAC_V2_EXIT_CLOSE"] = "0.72"
    os.environ["DSAC_V2_REDUCE_EXIT"] = "0.50"
    os.environ["DSAC_V2_TARGET_EXPOSURE_FLOOR"] = "0.08"
    os.environ["DSAC_V2_RESIZE_REL_TOL"] = "0.06"
    os.environ["RL_DD_PENALTY_COEFF"] = "0.020"
    os.environ["RL_KELLY_ALIGN_BONUS"] = "0.120"
    os.environ["RL_KELLY_CHOP_LOSS_PENALTY"] = "1.20"
    os.environ["DSAC_DD_PENALTY_COEFF"] = "0.020"
    os.environ["DSAC_KELLY_ALIGN_BONUS"] = "0.120"
    os.environ["DSAC_KELLY_CHOP_LOSS_PENALTY"] = "1.20"
    os.environ["DSAC_ADVERSE_HOLD_ENABLE"] = "1"


_set_v2_env()

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from alpha5_direction_router_score_rl_csv_20260519 import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_ROUTER_OUT_DIR,
    score_router_frame,
)
from ensemble.train_rl_dsac_agent import train  # noqa: E402


ROOT = Path("/home/llewyn/crypto-scalping")
DEFAULT_RL_2025 = ROOT / "data/rl_training_2025_unified.csv"
DEFAULT_RL_2026 = ROOT / "data/rl_training_2026_unified.csv"
DEFAULT_SPECIALIST_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_dsac_direction_specialists_v2_20260519"

SIDE_PROFILES = {
    "long": {
        "side_mode_override": "long",
        "specialist_pos_thresh": 0.12,
        "specialist_close_thresh": 0.030,
        "min_val_trades_for_best": 120,
        "target_val_trades": 365,
        "target_val_trades_per_day_low": 5.0,
        "target_val_trades_per_day_high": 10.0,
        "event_min_prob": 0.18,
        "event_min_edge": 0.010,
        "event_prob_gap": 0.000,
        "event_debounce_bars": 1,
        "event_fallback_bars": 2,
        "event_fallback_min_abs_action": 0.06,
        "event_fallback_quality_min": -0.02,
        "event_fallback_prob_floor": 0.10,
        "anti_flat_lambda": 0.02,
        "anti_flat_min_abs": 0.05,
        "val_trade_shortfall_penalty": 10.0,
        "val_trade_bonus_cap": 8.0,
        "event_eval_prob_quantile": 20.0,
        "density_curriculum_enable": True,
        "density_ramp_episodes": 60,
        "density_start_min_val_trades_for_best": 16,
        "density_start_specialist_pos_thresh": 0.03,
        "density_start_event_min_prob": 0.04,
        "density_start_event_min_edge": 0.000,
        "density_start_event_fallback_min_abs_action": 0.015,
        "density_start_event_fallback_quality_min": -0.05,
        "density_start_event_fallback_prob_floor": 0.02,
        "density_start_event_eval_prob_quantile": 5.0,
        "density_start_anti_flat_lambda": 0.08,
    },
    "short": {
        "side_mode_override": "short",
        "specialist_pos_thresh": 0.12,
        "specialist_close_thresh": 0.032,
        "min_val_trades_for_best": 120,
        "target_val_trades": 365,
        "target_val_trades_per_day_low": 5.0,
        "target_val_trades_per_day_high": 10.0,
        "event_min_prob": 0.18,
        "event_min_edge": 0.010,
        "event_prob_gap": 0.000,
        "event_debounce_bars": 1,
        "event_fallback_bars": 2,
        "event_fallback_min_abs_action": 0.06,
        "event_fallback_quality_min": -0.02,
        "event_fallback_prob_floor": 0.10,
        "anti_flat_lambda": 0.02,
        "anti_flat_min_abs": 0.05,
        "val_trade_shortfall_penalty": 10.0,
        "val_trade_bonus_cap": 8.0,
        "event_eval_prob_quantile": 20.0,
        "density_curriculum_enable": True,
        "density_ramp_episodes": 60,
        "density_start_min_val_trades_for_best": 16,
        "density_start_specialist_pos_thresh": 0.03,
        "density_start_event_min_prob": 0.04,
        "density_start_event_min_edge": 0.000,
        "density_start_event_fallback_min_abs_action": 0.015,
        "density_start_event_fallback_quality_min": -0.05,
        "density_start_event_fallback_prob_floor": 0.02,
        "density_start_event_eval_prob_quantile": 5.0,
        "density_start_anti_flat_lambda": 0.08,
    },
}


def _ensure_router_csv(input_csv: Path, output_csv: Path, prefix: str) -> dict:
    summary_path = output_csv.with_suffix(".router_summary.json")
    if output_csv.exists() and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            src_ok = str(summary.get("input_csv", "")) == str(input_csv)
            out_ok = str(summary.get("output_csv", "")) == str(output_csv)
            prefix_ok = str(summary.get("prefix", "")) == str(prefix)
            fresh_enough = output_csv.stat().st_mtime >= input_csv.stat().st_mtime
            if src_ok and out_ok and prefix_ok and fresh_enough:
                return summary
        except Exception:
            pass
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
    profile = dict(SIDE_PROFILES[side])
    paths = _specialist_paths(out_dir, side)
    train(
        csv_path=str(csv_path),
        train_ratio=0.8,
        episodes=2 if smoke else int(episodes),
        fresh_start=bool(fresh_start),
        val_interval=2 if smoke else 5,
        early_stop_patience=4 if smoke else 12,
        min_val_trades_for_best=int(profile["min_val_trades_for_best"]),
        target_val_trades=int(profile["target_val_trades"]),
        target_val_trades_per_day_low=float(profile["target_val_trades_per_day_low"]),
        target_val_trades_per_day_high=float(profile["target_val_trades_per_day_high"]),
        val_trade_shortfall_penalty=float(profile["val_trade_shortfall_penalty"]),
        val_trade_bonus_cap=float(profile["val_trade_bonus_cap"]),
        config_json_path=str(paths["config_json_path"]),
        checkpoint_path=str(paths["checkpoint_path"]),
        best_path=str(paths["best_path"]),
        side_mode_override=str(profile["side_mode_override"]),
        specialist_pos_thresh=float(profile["specialist_pos_thresh"]),
        specialist_close_thresh=float(profile["specialist_close_thresh"]),
        batch_size=64,
        update_freq=2,
        min_buffer=128 if smoke else 1024,
        warmup_steps=128 if smoke else 2048,
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
        event_eval_prob_quantile=float(profile["event_eval_prob_quantile"]),
        terminal_reward_scale=0.60,
        terminal_quality_win=0.10,
        terminal_quality_loss=0.02,
        anti_flat_lambda=float(profile["anti_flat_lambda"]),
        anti_flat_min_abs=float(profile["anti_flat_min_abs"]),
        direction_reg_lambda=0.0,
        side_balance_lambda=0.0,
        action_phase1_episodes=20 if not smoke else 1,
        action_phase2_episodes=45 if not smoke else 2,
        cvar_warmup_updates=30000 if not smoke else 0,
        pessimism_warmup_updates=20000 if not smoke else 0,
        density_curriculum_enable=bool(profile["density_curriculum_enable"]),
        density_ramp_episodes=int(profile["density_ramp_episodes"]),
        density_start_min_val_trades_for_best=int(profile["density_start_min_val_trades_for_best"]),
        density_start_specialist_pos_thresh=float(profile["density_start_specialist_pos_thresh"]),
        density_start_event_min_prob=float(profile["density_start_event_min_prob"]),
        density_start_event_min_edge=float(profile["density_start_event_min_edge"]),
        density_start_event_fallback_min_abs_action=float(profile["density_start_event_fallback_min_abs_action"]),
        density_start_event_fallback_quality_min=float(profile["density_start_event_fallback_quality_min"]),
        density_start_event_fallback_prob_floor=float(profile["density_start_event_fallback_prob_floor"]),
        density_start_event_eval_prob_quantile=float(profile["density_start_event_eval_prob_quantile"]),
        density_start_anti_flat_lambda=float(profile["density_start_anti_flat_lambda"]),
    )
    summary = {
        "side": side,
        "csv_path": str(csv_path),
        "episodes": int(episodes),
        "smoke": bool(smoke),
        "device": str(device),
        "run_finished_at": datetime.now().isoformat(),
        "paths": {k: str(v) for k, v in paths.items()},
        "profile": profile,
    }
    paths["summary_path"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train alpha5 v2 direction-routed DSAC long/short specialists.")
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
    _set_v2_env()
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
        print(
            json.dumps(
                {
                    "status": "startup_check_ok",
                    "v2_multi_action": True,
                    "router_csv_2025": str(scored_2025),
                    "router_csv_2026": str(scored_2026),
                    "sides": [x.strip() for x in str(args.sides).split(",") if x.strip()],
                    "out_dir": str(args.out_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": datetime.now().isoformat(),
        "v2_multi_action": True,
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
            episodes=int(args.episodes),
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
