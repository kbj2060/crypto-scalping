#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

def _set_v2_env() -> None:
    os.environ["DSAC_ALPHA5_V2_STATE_ENABLE"] = "1"
    os.environ["DSAC_V2_MULTI_ACTION_ENABLE"] = "1"


_set_v2_env()

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from scripts.eval_best_dsac_agent_2026 import (  # noqa: E402
    _load_training_hmm,
    _resolve_runtime_device,
    _build_agent_from_ckpt,
)
from ensemble.train_rl_dsac_agent import (  # noqa: E402
    DSACCompactTradingEnv,
    MultiTimeframeFeatures,
    apply_dsac_specialist_soft_gate,
)


ROOT = Path("/home/llewyn/crypto-scalping")
DEFAULT_ROUTED_CSV = ROOT / "tmp/causal_regen_20260516/alpha5_direction_router_rl_20260519/rl_training_2026_direction_router.csv"
DEFAULT_MANIFEST = ROOT / "tmp/causal_regen_20260516/alpha5_dsac_direction_specialists_v2_20260519/manifest.json"

def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _profile(manifest: dict[str, Any], side: str) -> dict[str, Any]:
    return dict(manifest["specialists"][side]["profile"])


def _paths(manifest: dict[str, Any], side: str) -> dict[str, str]:
    return dict(manifest["specialists"][side]["paths"])


def _resolve_ckpt(paths: dict[str, str], allow_checkpoint_fallback: bool) -> str:
    best_path = str(paths.get("best_path", "") or "")
    ckpt_path = str(paths.get("checkpoint_path", "") or "")
    if best_path and Path(best_path).exists():
        return best_path
    if allow_checkpoint_fallback and ckpt_path and Path(ckpt_path).exists():
        return ckpt_path
    raise FileNotFoundError(f"No usable checkpoint found. best={best_path} checkpoint={ckpt_path}")


def _route_side(df: pd.DataFrame, idx: int, prefix: str) -> str:
    row = df.iloc[idx]
    avail = float(pd.to_numeric(row.get(f"{prefix}_available", 0.0), errors="coerce") or 0.0)
    if avail < 0.5:
        return "none"
    npv = float(pd.to_numeric(row.get(f"{prefix}_none_prob", 0.0), errors="coerce") or 0.0)
    lp = float(pd.to_numeric(row.get(f"{prefix}_long_prob", 0.0), errors="coerce") or 0.0)
    sp = float(pd.to_numeric(row.get(f"{prefix}_short_prob", 0.0), errors="coerce") or 0.0)
    if npv >= lp and npv >= sp:
        return "none"
    return "long" if lp >= sp else "short"


def _build_env(eval_df: pd.DataFrame, hmm_detector, pos_thresh: float, close_thresh: float) -> DSACCompactTradingEnv:
    mtf = MultiTimeframeFeatures(eval_df["close"].values.astype("float32"))
    return DSACCompactTradingEnv(
        eval_df.reset_index(drop=True),
        phase="val",
        hmm_detector=hmm_detector,
        mtf_features=mtf,
        side_mode="both",
        specialist_pos_thresh=float(pos_thresh),
        specialist_close_thresh=float(close_thresh),
        dd_penalty_coeff=float(os.getenv("DSAC_DD_PENALTY_COEFF", "0.03")),
        kelly_align_bonus=float(os.getenv("DSAC_KELLY_ALIGN_BONUS", "0.0")),
        kelly_chop_loss_penalty=float(os.getenv("DSAC_KELLY_CHOP_LOSS_PENALTY", "1.30")),
        adverse_hold_enable=str(os.getenv("DSAC_ADVERSE_HOLD_ENABLE", "0")).strip().lower() in {"1", "true", "on", "yes"},
        terminal_reward_scale=0.0,
        terminal_quality_win=0.0,
        terminal_quality_loss=0.0,
        event_entry_filter_enable=False,
    )


def _apply_side_profile(env: DSACCompactTradingEnv, profile: dict[str, Any], routed_side: str | None) -> None:
    env.side_mode = str(profile.get("side_mode_override", routed_side or env.side_mode or "both"))
    env.pos_thresh = float(profile.get("specialist_pos_thresh", env.pos_thresh))
    env.close_thresh = float(profile.get("specialist_close_thresh", env.close_thresh))


def _eval_routed(
    eval_df: pd.DataFrame,
    prefix: str,
    long_agent,
    long_epoch: int,
    long_profile: dict[str, Any],
    short_agent,
    short_epoch: int,
    short_profile: dict[str, Any],
    hmm_detector,
) -> dict[str, Any]:
    env = _build_env(
        eval_df,
        hmm_detector,
        pos_thresh=float(long_profile.get("specialist_pos_thresh", 0.13)),
        close_thresh=float(long_profile.get("specialist_close_thresh", 0.04)),
    )
    st = env.reset()
    done = False
    peak_eq = float(env.initial_balance)
    mdd_pct = 0.0
    le = se = 0
    route_counts = {"none": 0, "long": 0, "short": 0}
    while not done:
        prev_pos = env.pos
        if prev_pos == "LONG":
            _apply_side_profile(env, long_profile, "long")
            with torch.no_grad():
                raw = long_agent.act(st, deterministic=True)
            action = apply_dsac_specialist_soft_gate(
                raw, st, env.regime_bucket(), gate_scale=1.0, regime_ctx=env.regime_context(), side_mode="long"
            )
            routed_side = "long"
        elif prev_pos == "SHORT":
            _apply_side_profile(env, short_profile, "short")
            with torch.no_grad():
                raw = short_agent.act(st, deterministic=True)
            action = apply_dsac_specialist_soft_gate(
                raw, st, env.regime_bucket(), gate_scale=1.0, regime_ctx=env.regime_context(), side_mode="short"
            )
            routed_side = "short"
        else:
            routed_side = _route_side(eval_df, int(env.current_step), prefix)
            if routed_side == "long":
                _apply_side_profile(env, long_profile, "long")
                with torch.no_grad():
                    raw = long_agent.act(st, deterministic=True)
                action = apply_dsac_specialist_soft_gate(
                    raw, st, env.regime_bucket(), gate_scale=1.0, regime_ctx=env.regime_context(), side_mode="long"
                )
            elif routed_side == "short":
                _apply_side_profile(env, short_profile, "short")
                with torch.no_grad():
                    raw = short_agent.act(st, deterministic=True)
                action = apply_dsac_specialist_soft_gate(
                    raw, st, env.regime_bucket(), gate_scale=1.0, regime_ctx=env.regime_context(), side_mode="short"
                )
            else:
                env.side_mode = "both"
                action = 0.0
        route_counts[routed_side] = route_counts.get(routed_side, 0) + 1
        st, _, done, info = env.step(action)
        if prev_pos is None and env.pos == "LONG":
            le += 1
        elif prev_pos is None and env.pos == "SHORT":
            se += 1
        cur_eq = env.balance * (1.0 + env.unrealized_pnl if env.pos is not None else 1.0)
        peak_eq = max(peak_eq, cur_eq)
        mdd_pct = min(mdd_pct, (cur_eq / max(peak_eq, 1e-8) - 1.0) * 100.0)

    return {
        "pnl": float((env.balance / env.initial_balance - 1.0) * 100.0),
        "wr": float(env.win_rate),
        "mdd": float(mdd_pct),
        "tr": int(env.total_trades),
        "long_entries": int(le),
        "short_entries": int(se),
        "route_counts": route_counts,
        "long_epoch": int(long_epoch),
        "short_epoch": int(short_epoch),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate alpha5 v2 direction-router + long/short DSAC specialists.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--csv-path", default=str(DEFAULT_ROUTED_CSV))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--prefix", default="a5dir")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--allow-checkpoint-fallback", action="store_true")
    args = ap.parse_args()

    manifest = _load_manifest(Path(args.manifest))
    device = _resolve_runtime_device(args.device)
    eval_df = pd.read_csv(args.csv_path)
    long_paths = _paths(manifest, "long")
    short_paths = _paths(manifest, "short")
    long_ckpt_path = _resolve_ckpt(long_paths, bool(args.allow_checkpoint_fallback))
    short_ckpt_path = _resolve_ckpt(short_paths, bool(args.allow_checkpoint_fallback))
    with open(long_paths["config_json_path"], "r", encoding="utf-8") as f:
        long_cfg = json.load(f)
    with open(short_paths["config_json_path"], "r", encoding="utf-8") as f:
        short_cfg = json.load(f)
    long_profile = _profile(manifest, "long")
    short_profile = _profile(manifest, "short")
    hmm_detector = _load_training_hmm(long_cfg)
    long_agent, long_ckpt = _build_agent_from_ckpt(long_ckpt_path, long_cfg, device)
    short_agent, short_ckpt = _build_agent_from_ckpt(short_ckpt_path, short_cfg, device)
    result = _eval_routed(
        eval_df=eval_df,
        prefix=str(args.prefix),
        long_agent=long_agent,
        long_epoch=int(long_ckpt.get("epoch", long_cfg.get("episodes", 0))),
        long_profile=long_profile,
        short_agent=short_agent,
        short_epoch=int(short_ckpt.get("epoch", short_cfg.get("episodes", 0))),
        short_profile=short_profile,
        hmm_detector=hmm_detector,
    )
    payload = {
        "csv_path": args.csv_path,
        "manifest": args.manifest,
        "device": device,
        "long_ckpt": long_ckpt_path,
        "short_ckpt": short_ckpt_path,
        "result": result,
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
