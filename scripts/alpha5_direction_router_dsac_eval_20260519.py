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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from scripts.eval_best_dsac_agent_2026 import (  # noqa: E402
    _load_rl_frame,
    _load_training_hmm,
    _resolve_runtime_device,
    _soft_gate_scale,
    _build_agent_from_ckpt,
)
from ensemble.train_rl_dsac_agent import (  # noqa: E402
    DSACCompactTradingEnv,
    MultiTimeframeFeatures,
    apply_dsac_specialist_soft_gate,
)


ROOT = Path("/home/llewyn/crypto-scalping")
DEFAULT_ROUTED_CSV = ROOT / "tmp/causal_regen_20260516/alpha5_direction_router_rl_20260519/rl_training_2026_direction_router.csv"
DEFAULT_MANIFEST = ROOT / "tmp/causal_regen_20260516/alpha5_dsac_direction_specialists_full_20260519/manifest.json"


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
    lp = float(pd.to_numeric(row.get(f"{prefix}_long_prob", 0.0), errors="coerce") or 0.0)
    sp = float(pd.to_numeric(row.get(f"{prefix}_short_prob", 0.0), errors="coerce") or 0.0)
    return "long" if lp >= sp else "short"


def _build_env(eval_df: pd.DataFrame, hmm_detector, pos_thresh: float, close_thresh: float) -> DSACCompactTradingEnv:
    mtf = MultiTimeframeFeatures(eval_df["close"].values.astype(np.float32))
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


def _prepare_action(side: str, raw_action: float, env: DSACCompactTradingEnv, side_profile: dict[str, Any]) -> float:
    a = float(np.clip(raw_action, -1.0, 1.0))
    pos_thresh = float(side_profile.get("specialist_pos_thresh", 0.18))
    if env.pos is None:
        if side == "long":
            a = max(a, 0.0)
            return a if a > pos_thresh else 0.0
        if side == "short":
            a = min(a, 0.0)
            return a if a < -pos_thresh else 0.0
        return 0.0
    if env.pos == "LONG":
        return float(np.clip(a, -1.0, 1.0))
    if env.pos == "SHORT":
        return float(np.clip(a, -1.0, 1.0))
    return 0.0


def _eval_routed(
    eval_df: pd.DataFrame,
    prefix: str,
    long_agent,
    long_cfg: dict[str, Any],
    long_epoch: int,
    short_agent,
    short_cfg: dict[str, Any],
    short_epoch: int,
    hmm_detector,
    long_profile: dict[str, Any],
    short_profile: dict[str, Any],
) -> dict[str, Any]:
    if len(eval_df) < 32:
        return {"pnl": 0.0, "wr": 0.0, "mdd": 0.0, "tr": 0, "long_entries": 0, "short_entries": 0}

    shared_pos_thresh = min(float(long_profile.get("specialist_pos_thresh", 0.17)), float(short_profile.get("specialist_pos_thresh", 0.19)))
    shared_close_thresh = min(float(long_profile.get("specialist_close_thresh", 0.055)), float(short_profile.get("specialist_close_thresh", 0.065)))
    env = _build_env(eval_df, hmm_detector, pos_thresh=shared_pos_thresh, close_thresh=shared_close_thresh)
    st = env.reset()
    done = False
    peak_eq = float(env.initial_balance)
    mdd_pct = 0.0
    le = se = fcl = fcs = hs_l = hs_s = hn_l = hn_s = 0
    route_counts = {"none": 0, "long": 0, "short": 0}

    long_gate = max(0.50, _soft_gate_scale(long_epoch, int(long_cfg.get("soft_gate_warmup_epochs", 20)), int(long_cfg.get("soft_gate_ramp_epochs", 80))))
    short_gate = max(0.50, _soft_gate_scale(short_epoch, int(short_cfg.get("soft_gate_warmup_epochs", 20)), int(short_cfg.get("soft_gate_ramp_epochs", 80))))

    while not done:
        prev_pos = env.pos
        decision_step = int(env.current_step)
        if prev_pos == "LONG":
            with torch.no_grad():
                raw = long_agent.act(st, deterministic=True)
            gated = apply_dsac_specialist_soft_gate(raw, st, env.regime_bucket(), gate_scale=long_gate, regime_ctx=env.regime_context(decision_step), focus_regime="long")
            action = _prepare_action("long", gated, env, long_profile)
            routed_side = "long"
        elif prev_pos == "SHORT":
            with torch.no_grad():
                raw = short_agent.act(st, deterministic=True)
            gated = apply_dsac_specialist_soft_gate(raw, st, env.regime_bucket(), gate_scale=short_gate, regime_ctx=env.regime_context(decision_step), focus_regime="short")
            action = _prepare_action("short", gated, env, short_profile)
            routed_side = "short"
        else:
            routed_side = _route_side(eval_df, decision_step, prefix)
            if routed_side == "long":
                with torch.no_grad():
                    raw = long_agent.act(st, deterministic=True)
                gated = apply_dsac_specialist_soft_gate(raw, st, env.regime_bucket(), gate_scale=long_gate, regime_ctx=env.regime_context(decision_step), focus_regime="long")
                action = _prepare_action("long", gated, env, long_profile)
            elif routed_side == "short":
                with torch.no_grad():
                    raw = short_agent.act(st, deterministic=True)
                gated = apply_dsac_specialist_soft_gate(raw, st, env.regime_bucket(), gate_scale=short_gate, regime_ctx=env.regime_context(decision_step), focus_regime="short")
                action = _prepare_action("short", gated, env, short_profile)
            else:
                action = 0.0
        route_counts[routed_side] = route_counts.get(routed_side, 0) + 1
        st, _, done, info = env.step(action)
        if prev_pos is None and env.pos == "LONG":
            le += 1
        elif prev_pos is None and env.pos == "SHORT":
            se += 1
        if bool(info.get("force_closed", False)):
            closed_side = str(info.get("closed_side", "") or "")
            if closed_side == "LONG":
                fcl += 1
            elif closed_side == "SHORT":
                fcs += 1
        ch = int(info.get("closed_hold_count", 0) or 0)
        cs = str(info.get("closed_side", "") or "")
        if ch > 0 and cs == "LONG":
            hs_l += ch
            hn_l += 1
        elif ch > 0 and cs == "SHORT":
            hs_s += ch
            hn_s += 1
        cur_eq = env.balance * (1.0 + env.unrealized_pnl if env.pos is not None else 1.0)
        peak_eq = max(peak_eq, cur_eq)
        mdd_pct = min(mdd_pct, (cur_eq / max(peak_eq, 1e-8) - 1.0) * 100.0)

    pnl = (env.balance / env.initial_balance - 1.0) * 100.0
    return {
        "pnl": float(pnl),
        "wr": float(env.win_rate),
        "mdd": float(mdd_pct),
        "tr": int(env.total_trades),
        "long_entries": int(le),
        "short_entries": int(se),
        "fcl": int(fcl),
        "fcs": int(fcs),
        "avg_hold_long": float(hs_l / max(hn_l, 1)) if hn_l > 0 else 0.0,
        "avg_hold_short": float(hs_s / max(hn_s, 1)) if hn_s > 0 else 0.0,
        "route_counts": route_counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate alpha5 direction-router + long/short DSAC specialists on routed CSV.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--csv-path", default=str(DEFAULT_ROUTED_CSV))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--prefix", default="a5dir")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--allow-checkpoint-fallback", action="store_true")
    args = ap.parse_args()

    manifest = _load_manifest(Path(args.manifest))
    device = _resolve_runtime_device(args.device)
    eval_df = _load_rl_frame(args.csv_path)

    long_paths = _paths(manifest, "long")
    short_paths = _paths(manifest, "short")
    long_ckpt_path = _resolve_ckpt(long_paths, bool(args.allow_checkpoint_fallback))
    short_ckpt_path = _resolve_ckpt(short_paths, bool(args.allow_checkpoint_fallback))

    with open(long_paths["config_json_path"], "r", encoding="utf-8") as f:
        long_cfg = json.load(f)
    with open(short_paths["config_json_path"], "r", encoding="utf-8") as f:
        short_cfg = json.load(f)

    hmm_detector = _load_training_hmm(long_cfg)
    long_agent, long_ckpt = _build_agent_from_ckpt(long_ckpt_path, long_cfg, device)
    short_agent, short_ckpt = _build_agent_from_ckpt(short_ckpt_path, short_cfg, device)
    long_epoch = int(long_ckpt.get("epoch", long_cfg.get("episodes", 0)))
    short_epoch = int(short_ckpt.get("epoch", short_cfg.get("episodes", 0)))

    overall = _eval_routed(
        eval_df=eval_df,
        prefix=str(args.prefix),
        long_agent=long_agent,
        long_cfg=long_cfg,
        long_epoch=long_epoch,
        short_agent=short_agent,
        short_cfg=short_cfg,
        short_epoch=short_epoch,
        hmm_detector=hmm_detector,
        long_profile=_profile(manifest, "long"),
        short_profile=_profile(manifest, "short"),
    )

    payload = {
        "csv_path": args.csv_path,
        "manifest": args.manifest,
        "device": device,
        "long_ckpt": long_ckpt_path,
        "short_ckpt": short_ckpt_path,
        "long_epoch": long_epoch,
        "short_epoch": short_epoch,
        "rows": int(len(eval_df)),
        "overall": overall,
        "evaluated_at": datetime.now().isoformat(),
    }
    out_json = args.out_json or os.path.join(
        _ROOT_DIR,
        "tmp/causal_regen_20260516/alpha5_dsac_direction_specialists_full_20260519",
        f"routed_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(
        "[ROUTED EVAL] PnL:{pnl:.2f}% | Tr:{tr} | WR:{wr:.0f}% | MDD:{mdd:.2f}% | L:{le} S:{se} | AvgHoldL:{ahl:.1f} AvgHoldS:{ahs:.1f}".format(
            pnl=float(overall["pnl"]),
            tr=int(overall["tr"]),
            wr=float(overall["wr"]) * 100.0,
            mdd=float(overall["mdd"]),
            le=int(overall["long_entries"]),
            se=int(overall["short_entries"]),
            ahl=float(overall["avg_hold_long"]),
            ahs=float(overall["avg_hold_short"]),
        )
    )
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
