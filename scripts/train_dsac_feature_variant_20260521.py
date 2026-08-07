#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

ROOT = Path("/home/llewyn/crypto-scalping")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT_CSV = ROOT / "tmp/causal_regen_20260516/dsac_feature_inventory_regime_fixed_20260521/rl_training_2025_direction_router_feature_inventory_base_with_family_pca.csv"
DEFAULT_SPEC_DIR = ROOT / "tmp/causal_regen_20260516/dsac_feature_variant_specs_regime_fixed_20260521"
DEFAULT_OUT_ROOT = ROOT / "tmp/causal_regen_20260516/dsac_feature_screen_regime_fixed_20260521"
VAL_CUTOFF = pd.Timestamp("2025-10-01")


PROFILE = {
    "side_mode_override": "both",
    "specialist_pos_thresh": 0.10,
    "specialist_close_thresh": 0.030,
    "min_val_trades_for_best": 24,
    "target_val_trades": 365,
    "target_val_trades_per_day_low": 5.0,
    "target_val_trades_per_day_high": 10.0,
    "cvar_frac": 0.70,
    "event_min_prob": 0.06,
    "event_min_edge": 0.002,
    "event_prob_gap": 0.000,
    "event_debounce_bars": 1,
    "event_fallback_bars": 2,
    "event_fallback_min_abs_action": 0.030,
    "event_fallback_quality_min": -0.03,
    "event_fallback_prob_floor": 0.04,
    "anti_flat_lambda": 0.04,
    "anti_flat_min_abs": 0.05,
    "val_trade_shortfall_penalty": 2.0,
    "val_trade_bonus_cap": 8.0,
    "val_side_bias_penalty": 0.0,
    "event_eval_prob_quantile": 0.0,
    "density_curriculum_enable": True,
    "density_ramp_episodes": 160,
    "density_start_min_val_trades_for_best": 8,
    "density_start_specialist_pos_thresh": 0.02,
    "density_start_event_min_prob": 0.02,
    "density_start_event_min_edge": 0.000,
    "density_start_event_fallback_min_abs_action": 0.010,
    "density_start_event_fallback_quality_min": -0.05,
    "density_start_event_fallback_prob_floor": 0.01,
    "density_start_event_eval_prob_quantile": 0.0,
    "density_start_anti_flat_lambda": 0.06,
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _set_common_env() -> None:
    os.environ["DSAC_ALPHA5_V2_STATE_ENABLE"] = "1"
    os.environ["DSAC_V2_MULTI_ACTION_ENABLE"] = "1"
    os.environ["DSAC_RECURRENT_ENABLE"] = "0"
    os.environ["DSAC_ATTN_STACK_ENABLE"] = "1"
    os.environ["DSAC_STACK_N"] = "2"
    os.environ["DSAC_V2_TP_MIN"] = "0.0015"
    os.environ["DSAC_V2_TP_MAX"] = "0.0120"
    os.environ["DSAC_V2_SL_MIN"] = "0.0008"
    os.environ["DSAC_V2_SL_MAX"] = "0.0100"
    os.environ["DSAC_V2_NOTIONAL_BUCKETS"] = "0.15,0.25,0.40,0.60,0.85,1.20"
    os.environ["DSAC_V2_AUTO_LEVERAGE_BUCKETS"] = "1.0,1.5,2.0,3.0,4.0,5.0"
    os.environ["DSAC_V2_TP_BUCKETS"] = "0.0015,0.0025,0.0040,0.0060,0.0090,0.0120"
    os.environ["DSAC_V2_SL_BUCKETS"] = "0.0008,0.0012,0.0018,0.0025,0.0040,0.0070,0.0100"
    os.environ["DSAC_V2_INTENT_ENTRY_TH"] = "0.25"
    os.environ["DSAC_V2_INTENT_EXIT_TH"] = "0.08"
    os.environ["DSAC_V2_INTENT_REVERSE_TH"] = "0.45"
    os.environ["DSAC_V2_EXIT_CLOSE"] = "0.72"
    os.environ["DSAC_V2_REDUCE_EXIT"] = "0.50"
    os.environ["DSAC_V2_EXIT_SOFT_ENABLE"] = "1"
    os.environ["DSAC_V2_EXIT_SOFT_MAX_REDUCE"] = "0.80"
    os.environ["DSAC_V2_CONTINUOUS_RISK_ENABLE"] = "0"
    os.environ["DSAC_V2_TARGET_EXPOSURE_FLOOR"] = "0.08"
    os.environ["DSAC_V2_RESIZE_REL_TOL"] = "0.06"
    os.environ["DSAC_FAST_VAL_MODE"] = "1"
    # Alpha5 barrier backtests do not have the DSAC env's extra force-close rail.
    # Keep training/validation execution closer to the backtest contract.
    os.environ["RL_FORCE_CLOSE_ENABLE"] = "0"
    os.environ["DSAC_ALPHA5_BARRIER_VAL_ENABLE"] = "1"
    os.environ["DSAC_ALPHA5_BARRIER_MAX_HOLD_BARS"] = "96"
    os.environ["DSAC_ALPHA5_LABEL_DIR"] = str(
        ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_high_quality_training_data_20260518"
    )
    os.environ["RL_DD_PENALTY_COEFF"] = "0.020"
    os.environ["RL_KELLY_ALIGN_BONUS"] = "0.120"
    os.environ["RL_KELLY_CHOP_LOSS_PENALTY"] = "1.20"
    os.environ["DSAC_DD_PENALTY_COEFF"] = "0.020"
    os.environ["DSAC_KELLY_ALIGN_BONUS"] = "0.120"
    os.environ["DSAC_KELLY_CHOP_LOSS_PENALTY"] = "1.20"
    os.environ["DSAC_ADVERSE_HOLD_ENABLE"] = "0"


def _set_torch_perf() -> None:
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a single DSAC feature-screen variant on clean 2025 router rows.")
    p.add_argument("--variant", required=True, help="Variant name or spec JSON path")
    p.add_argument("--spec-dir", default=str(DEFAULT_SPEC_DIR))
    p.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--fresh-start", action="store_true")
    p.add_argument(
        "--alpha5-netwr-compat",
        action="store_true",
        help="Use the cleaned input CSV but revert DSAC state/hyperparameters to the Alpha5 netwr compact-state profile.",
    )
    return p.parse_args()


def _resolve_spec_path(variant: str, spec_dir: Path) -> Path:
    p = Path(variant)
    if p.exists():
        return p
    cand = spec_dir / f"{variant}.json"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"variant spec not found: {variant}")


def _resolve_train_ratio(csv_path: Path) -> float:
    df = pd.read_csv(csv_path, usecols=["timestamp"])
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    train_rows = int((ts < VAL_CUTOFF).sum())
    total_rows = int(ts.notna().sum())
    if train_rows <= 0 or total_rows <= 0 or train_rows >= total_rows:
        raise RuntimeError(f"invalid train/val split for {csv_path}: train_rows={train_rows} total_rows={total_rows}")
    return float(train_rows / total_rows)


def main() -> None:
    args = parse_args()
    spec_path = _resolve_spec_path(str(args.variant), Path(args.spec_dir))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_root) / spec["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_common_env()
    _set_torch_perf()
    feature_cols = list(spec.get("features", []))
    if args.alpha5_netwr_compat:
        os.environ["DSAC_ALL_FEATURES_ENABLE"] = "0"
        os.environ.pop("DSAC_ALL_FEATURE_LIST_JSON", None)
        os.environ["DSAC_EXTRA_PCA_ENABLE"] = "0"
        os.environ["DSAC_EXTRA_PCA_COMPONENTS"] = "0"
    elif feature_cols:
        os.environ["DSAC_ALL_FEATURES_ENABLE"] = "1"
        os.environ["DSAC_ALL_FEATURE_LIST_JSON"] = str(spec_path)
        os.environ["DSAC_EXTRA_PCA_ENABLE"] = "1" if bool(spec.get("extra_pca_enable", False)) else "0"
        os.environ["DSAC_EXTRA_PCA_COMPONENTS"] = str(int(spec.get("extra_pca_components", 0) or 0 or 32))
    else:
        os.environ["DSAC_ALL_FEATURES_ENABLE"] = "0"
        os.environ.pop("DSAC_ALL_FEATURE_LIST_JSON", None)
        os.environ["DSAC_EXTRA_PCA_ENABLE"] = "0"
        os.environ["DSAC_EXTRA_PCA_COMPONENTS"] = "0"

    from ensemble.train_rl_dsac_agent import train  # import after env setup

    train_ratio = 0.8 if args.alpha5_netwr_compat else _resolve_train_ratio(input_csv)
    manifest = {
        "variant": spec["name"],
        "spec_path": str(spec_path),
        "input_csv": str(input_csv),
        "out_dir": str(out_dir),
        "train_ratio": train_ratio,
        "episodes": int(args.episodes),
        "alpha5_netwr_compat": bool(args.alpha5_netwr_compat),
        "generated_at": datetime.now().isoformat(),
        "spec": spec,
    }
    (out_dir / "variant_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    train(
        csv_path=str(input_csv),
        train_ratio=float(train_ratio),
        episodes=int(args.episodes),
        fresh_start=bool(args.fresh_start),
        lr_actor=9e-4 if args.alpha5_netwr_compat else _env_float("DSAC_LR_ACTOR", 1e-4),
        lr_critic=9e-4 if args.alpha5_netwr_compat else _env_float("DSAC_LR_CRITIC", 1e-4),
        lr_alpha=3e-4 if args.alpha5_netwr_compat else _env_float("DSAC_LR_ALPHA", 1e-4),
        lr_min=3e-5 if args.alpha5_netwr_compat else 1e-5,
        val_interval=5 if args.alpha5_netwr_compat else 10,
        early_stop_patience=24,
        min_val_trades_for_best=int(PROFILE["min_val_trades_for_best"]),
        target_val_trades=int(PROFILE["target_val_trades"]),
        target_val_trades_per_day_low=float(PROFILE["target_val_trades_per_day_low"]),
        target_val_trades_per_day_high=float(PROFILE["target_val_trades_per_day_high"]),
        val_trade_shortfall_penalty=float(PROFILE["val_trade_shortfall_penalty"]),
        val_trade_bonus_cap=float(PROFILE["val_trade_bonus_cap"]),
        val_side_bias_penalty=float(PROFILE["val_side_bias_penalty"]),
        cvar_frac=float(PROFILE["cvar_frac"]),
        config_json_path=str(out_dir / "train_config.json"),
        checkpoint_path=str(out_dir / "checkpoint.pth"),
        best_path=str(out_dir / "best.pth"),
        side_mode_override=str(PROFILE["side_mode_override"]),
        specialist_pos_thresh=float(PROFILE["specialist_pos_thresh"]),
        specialist_close_thresh=float(PROFILE["specialist_close_thresh"]),
        batch_size=64 if args.alpha5_netwr_compat else 128,
        update_freq=2,
        min_buffer=1024 if args.alpha5_netwr_compat else 2048,
        warmup_steps=2048,
        device=str(args.device),
        skip_focus_segment_filter=True,
        terminate_on_regime_change=False,
        event_entry_filter_enable=True,
        event_prob_prefix="a5dir",
        event_min_prob=float(PROFILE["event_min_prob"]),
        event_min_edge=float(PROFILE["event_min_edge"]),
        event_prob_gap=float(PROFILE["event_prob_gap"]),
        event_debounce_bars=int(PROFILE["event_debounce_bars"]),
        event_fallback_bars=int(PROFILE["event_fallback_bars"]),
        event_fallback_min_abs_action=float(PROFILE["event_fallback_min_abs_action"]),
        event_fallback_quality_min=float(PROFILE["event_fallback_quality_min"]),
        event_fallback_prob_floor=float(PROFILE["event_fallback_prob_floor"]),
        event_eval_prob_quantile=float(PROFILE["event_eval_prob_quantile"]),
        terminal_reward_scale=0.60,
        terminal_quality_win=0.10,
        terminal_quality_loss=0.02,
        anti_flat_lambda=float(PROFILE["anti_flat_lambda"]),
        anti_flat_min_abs=float(PROFILE["anti_flat_min_abs"]),
        direction_reg_lambda=0.0,
        side_balance_lambda=0.0,
        action_phase1_episodes=10,
        action_phase2_episodes=35,
        cvar_warmup_updates=30000,
        pessimism_warmup_updates=20000,
        density_curriculum_enable=bool(PROFILE["density_curriculum_enable"]),
        density_ramp_episodes=int(PROFILE["density_ramp_episodes"]),
        density_start_min_val_trades_for_best=int(PROFILE["density_start_min_val_trades_for_best"]),
        density_start_specialist_pos_thresh=float(PROFILE["density_start_specialist_pos_thresh"]),
        density_start_event_min_prob=float(PROFILE["density_start_event_min_prob"]),
        density_start_event_min_edge=float(PROFILE["density_start_event_min_edge"]),
        density_start_event_fallback_min_abs_action=float(PROFILE["density_start_event_fallback_min_abs_action"]),
        density_start_event_fallback_quality_min=float(PROFILE["density_start_event_fallback_quality_min"]),
        density_start_event_fallback_prob_floor=float(PROFILE["density_start_event_fallback_prob_floor"]),
        density_start_event_eval_prob_quantile=float(PROFILE["density_start_event_eval_prob_quantile"]),
        density_start_anti_flat_lambda=float(PROFILE["density_start_anti_flat_lambda"]),
    )


if __name__ == "__main__":
    main()
