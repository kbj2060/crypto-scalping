#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd


def _set_v2_env() -> None:
    os.environ["DSAC_ALPHA5_V2_STATE_ENABLE"] = "1"
    os.environ["DSAC_V2_MULTI_ACTION_ENABLE"] = "1"
    os.environ.setdefault("DSAC_ALL_FEATURES_ENABLE", "1")
    os.environ.setdefault("DSAC_EXTRA_PCA_ENABLE", "1")
    os.environ.setdefault("DSAC_EXTRA_PCA_COMPONENTS", "32")
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
    os.environ.setdefault("DSAC_V2_CONTINUOUS_RISK_ENABLE", "0")
    os.environ["DSAC_V2_TARGET_EXPOSURE_FLOOR"] = "0.08"
    os.environ["DSAC_V2_RESIZE_REL_TOL"] = "0.06"
    os.environ["RL_DD_PENALTY_COEFF"] = "0.020"
    os.environ["RL_KELLY_ALIGN_BONUS"] = "0.120"
    os.environ["RL_KELLY_CHOP_LOSS_PENALTY"] = "1.20"
    os.environ["DSAC_DD_PENALTY_COEFF"] = "0.020"
    os.environ["DSAC_KELLY_ALIGN_BONUS"] = "0.120"
    os.environ["DSAC_KELLY_CHOP_LOSS_PENALTY"] = "1.20"
    os.environ["DSAC_ADVERSE_HOLD_ENABLE"] = "0"


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
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_dsac_single_router5_density_20260520"
DEFAULT_MARKET_STATE_2025 = ROOT / "data/ensemble/retrained_v22_market_state_v5_20260511/market_state_v5_2025.csv"
DEFAULT_MARKET_STATE_2026 = ROOT / "data/ensemble/retrained_v22_market_state_v5_20260511/market_state_v5_2026.csv"
MARKET_STATE_COLS = [
    "market_state_2024_unsup_v5_factor_vol",
    "market_state_2024_unsup_v5_risk_off_prob",
    "market_state_2024_unsup_v5_prob_2",
    "market_state_2024_unsup_v5_prob_3",
    "market_state_2024_unsup_v5_factor_liquidity_stress",
    "market_state_2024_unsup_v5_confidence",
    "market_state_2024_unsup_v5_trend_bias",
]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


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

def _norm_path_list(paths: list[str] | None) -> list[str]:
    if not paths:
        return []
    return [str(Path(p)) for p in paths]


def _ensure_router_csv(
    input_csv: Path,
    output_csv: Path,
    prefix: str,
    router_model: str | None = None,
    router_meta: str | None = None,
    aux_parquet: list[str] | None = None,
) -> dict:
    summary_path = output_csv.with_suffix(".router_summary.json")
    if output_csv.exists() and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            src_ok = str(summary.get("input_csv", "")) == str(input_csv)
            out_ok = str(summary.get("output_csv", "")) == str(output_csv)
            prefix_ok = str(summary.get("prefix", "")) == str(prefix)
            model_ok = str(summary.get("router_model", "")) == str(router_model or summary.get("router_model", ""))
            meta_ok = str(summary.get("router_meta", "")) == str(router_meta or summary.get("router_meta", ""))
            aux_ok = _norm_path_list(summary.get("aux_paths")) == _norm_path_list(aux_parquet) if aux_parquet else True
            fresh_enough = output_csv.stat().st_mtime >= input_csv.stat().st_mtime
            if src_ok and out_ok and prefix_ok and model_ok and meta_ok and aux_ok and fresh_enough:
                return summary
        except Exception:
            pass
    return score_router_frame(
        input_csv=input_csv,
        output_csv=output_csv,
        prefix=prefix,
        router_model_path=(Path(router_model) if router_model else None),
        router_meta_path=(Path(router_meta) if router_meta else None),
        aux_paths=([Path(x) for x in aux_parquet] if aux_parquet else None),
    )


def _ensure_market_state_csv(input_csv: Path, output_csv: Path, market_state_csv: Path) -> dict:
    summary_path = output_csv.with_suffix(".market_state_summary.json")
    if output_csv.exists() and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            src_ok = str(summary.get("input_csv", "")) == str(input_csv)
            ms_ok = str(summary.get("market_state_csv", "")) == str(market_state_csv)
            fresh_enough = output_csv.stat().st_mtime >= max(input_csv.stat().st_mtime, market_state_csv.stat().st_mtime)
            if src_ok and ms_ok and fresh_enough:
                return summary
        except Exception:
            pass
    if not input_csv.exists():
        raise FileNotFoundError(f"router csv missing: {input_csv}")
    if not market_state_csv.exists():
        raise FileNotFoundError(f"market-state csv missing: {market_state_csv}")

    df = pd.read_csv(input_csv)
    if "timestamp" not in df.columns:
        raise ValueError(f"timestamp column missing in {input_csv}")
    missing = [c for c in MARKET_STATE_COLS if c not in df.columns]
    if missing:
        use_cols = ["timestamp", *MARKET_STATE_COLS]
        ms_cols = pd.read_csv(market_state_csv, nrows=0).columns.tolist()
        use_cols = [c for c in use_cols if c in ms_cols]
        if "timestamp" not in use_cols:
            raise ValueError(f"timestamp column missing in {market_state_csv}")
        ms = pd.read_csv(market_state_csv, usecols=use_cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        ms["timestamp"] = pd.to_datetime(ms["timestamp"], errors="coerce")
        ms = ms.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        df = df.merge(ms, on="timestamp", how="left", suffixes=("", "__market_state"))
        for col in MARKET_STATE_COLS:
            aux_col = f"{col}__market_state"
            if col in df.columns and aux_col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].where(df[col].notna(), pd.to_numeric(df[aux_col], errors="coerce"))
                df = df.drop(columns=[aux_col])
            elif aux_col in df.columns:
                df = df.rename(columns={aux_col: col})
            elif col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    present = [c for c in MARKET_STATE_COLS if c in df.columns]
    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "market_state_csv": str(market_state_csv),
        "rows": int(len(df)),
        "market_state_cols": present,
        "market_state_col_count": int(len(present)),
        "missing_after_merge": [c for c in MARKET_STATE_COLS if c not in df.columns],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _paths(base_dir: Path) -> dict[str, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": base_dir,
        "config_json_path": base_dir / "train_config.json",
        "checkpoint_path": base_dir / "checkpoint.pth",
        "best_path": base_dir / "best.pth",
        "summary_path": base_dir / "summary.json",
        "manifest_path": base_dir / "manifest.json",
    }


def run_single(
    csv_path: Path,
    out_dir: Path,
    episodes: int,
    fresh_start: bool,
    device: str,
    prefix: str,
    market_state_summary: dict | None = None,
    smoke: bool = False,
) -> dict:
    paths = _paths(out_dir)
    train(
        csv_path=str(csv_path),
        train_ratio=0.8,
        episodes=2 if smoke else int(episodes),
        fresh_start=bool(fresh_start),
        lr_actor=_env_float("DSAC_LR_ACTOR", 3e-4),
        lr_critic=_env_float("DSAC_LR_CRITIC", 3e-4),
        lr_alpha=_env_float("DSAC_LR_ALPHA", 3e-4),
        val_interval=2 if smoke else 5,
        early_stop_patience=4 if smoke else 24,
        min_val_trades_for_best=int(PROFILE["min_val_trades_for_best"]),
        target_val_trades=int(PROFILE["target_val_trades"]),
        target_val_trades_per_day_low=float(PROFILE["target_val_trades_per_day_low"]),
        target_val_trades_per_day_high=float(PROFILE["target_val_trades_per_day_high"]),
        val_trade_shortfall_penalty=float(PROFILE["val_trade_shortfall_penalty"]),
        val_trade_bonus_cap=float(PROFILE["val_trade_bonus_cap"]),
        val_side_bias_penalty=float(PROFILE["val_side_bias_penalty"]),
        cvar_frac=float(PROFILE["cvar_frac"]),
        config_json_path=str(paths["config_json_path"]),
        checkpoint_path=str(paths["checkpoint_path"]),
        best_path=str(paths["best_path"]),
        side_mode_override=str(PROFILE["side_mode_override"]),
        specialist_pos_thresh=float(PROFILE["specialist_pos_thresh"]),
        specialist_close_thresh=float(PROFILE["specialist_close_thresh"]),
        batch_size=64,
        update_freq=2,
        min_buffer=128 if smoke else 1024,
        warmup_steps=128 if smoke else 2048,
        device=str(device),
        skip_focus_segment_filter=True,
        terminate_on_regime_change=False,
        event_entry_filter_enable=True,
        event_prob_prefix=str(prefix),
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
        action_phase1_episodes=10 if not smoke else 1,
        action_phase2_episodes=35 if not smoke else 2,
        cvar_warmup_updates=30000 if not smoke else 0,
        pessimism_warmup_updates=20000 if not smoke else 0,
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
    summary = {
        "csv_path": str(csv_path),
        "episodes": int(episodes),
        "smoke": bool(smoke),
        "device": str(device),
        "run_finished_at": datetime.now().isoformat(),
        "paths": {k: str(v) for k, v in paths.items()},
        "profile": PROFILE,
        "market_state_summary": market_state_summary or {},
    }
    paths["summary_path"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["manifest_path"].write_text(
        json.dumps(
            {
                "started_at": datetime.now().isoformat(),
                "v2_multi_action": True,
                "mode": "single_dsac_router5",
                "profile": PROFILE,
                "csv_path": str(csv_path),
                "summary_path": str(paths["summary_path"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train alpha5 single-agent DSAC with router5 routed CSV.")
    p.add_argument("--rl-2025", default=str(DEFAULT_RL_2025))
    p.add_argument("--rl-2026", default=str(DEFAULT_RL_2026))
    p.add_argument("--router-dir", default=str(DEFAULT_ROUTER_OUT_DIR))
    p.add_argument("--router-model", default=None)
    p.add_argument("--router-meta", default=None)
    p.add_argument("--router-aux-parquet", action="append", default=None)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--prefix", default="a5dir")
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
        summaries["score_2025"] = _ensure_router_csv(
            Path(args.rl_2025),
            scored_2025,
            args.prefix,
            router_model=args.router_model,
            router_meta=args.router_meta,
            aux_parquet=args.router_aux_parquet,
        )
        summaries["score_2026"] = _ensure_router_csv(
            Path(args.rl_2026),
            scored_2026,
            args.prefix,
            router_model=args.router_model,
            router_meta=args.router_meta,
            aux_parquet=args.router_aux_parquet,
        )
    if args.startup_check_only:
        print(
            json.dumps(
                {
                    "status": "startup_check_ok",
                    "v2_multi_action": True,
                    "mode": "single_dsac_router5",
                    "router_csv_2025": str(scored_2025),
                    "router_csv_2026": str(scored_2026),
                    "market_state_2025": str(DEFAULT_MARKET_STATE_2025),
                    "market_state_2026": str(DEFAULT_MARKET_STATE_2026),
                    "out_dir": str(args.out_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv_2025 = out_dir / "rl_training_2025_direction_router_market_state.csv"
    summaries["market_state_2025"] = _ensure_market_state_csv(scored_2025, train_csv_2025, DEFAULT_MARKET_STATE_2025)
    run_single(
        csv_path=train_csv_2025,
        out_dir=out_dir,
        episodes=int(args.episodes),
        fresh_start=bool(args.fresh_start),
        device=str(args.device),
        prefix=str(args.prefix),
        market_state_summary=summaries.get("market_state_2025"),
        smoke=bool(args.smoke),
    )


if __name__ == "__main__":
    main()
