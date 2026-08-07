from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "ensemble")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.train_rl_dsac_agent import train


ALL_REGIMES = ["bull", "bear", "chop", "whipsaw", "normal"]
DEFAULT_CSV = "/home/llewyn/crypto-scalping/data/ensemble/event_driven/trade_candidates_v1_oof.csv"
DEFAULT_CKPT_DIR = "/home/llewyn/crypto-scalping/data/ensemble/ckpt/regime_dsac_event_masked_v7_full5"

REGIME_PROFILES = {
    "bull": {
        "side_mode_override": "long",
        "specialist_pos_thresh": 0.17,
        "specialist_close_thresh": 0.055,
        "min_val_trades_for_best": 20,
        "target_val_trades": 40,
        "val_side_bias_penalty": 0.0,
        "event_min_prob": 0.52,
        "event_min_edge": 0.006,
        "event_prob_gap": 0.004,
        "event_debounce_bars": 2,
        "event_fallback_bars": 4,
        "event_fallback_min_abs_action": 0.24,
        "event_fallback_quality_min": 0.42,
        "event_fallback_prob_floor": 0.36,
    },
    "bear": {
        "side_mode_override": "short",
        "specialist_pos_thresh": 0.19,
        "specialist_close_thresh": 0.065,
        "min_val_trades_for_best": 25,
        "target_val_trades": 50,
        "val_side_bias_penalty": 0.0,
        "event_min_prob": 0.54,
        "event_min_edge": 0.008,
        "event_prob_gap": 0.006,
        "event_debounce_bars": 3,
        "event_fallback_bars": 6,
        "event_fallback_min_abs_action": 0.28,
        "event_fallback_quality_min": 0.45,
        "event_fallback_prob_floor": 0.38,
    },
    "chop": {
        "side_mode_override": "long",
        "specialist_pos_thresh": 0.22,
        "specialist_close_thresh": 0.09,
        "specialist_force_close_th": -0.012,
        "specialist_rev_exit_thresh": 0.045,
        "specialist_idle_penalty": 0.0,
        "specialist_soft_gate_scale": 1.0,
        "min_val_trades_for_best": 4,
        "target_val_trades": 8,
        "val_side_bias_penalty": 0.0,
        "event_min_prob": 0.58,
        "event_min_edge": 0.012,
        "event_prob_gap": 0.015,
        "event_debounce_bars": 4,
        "event_fallback_bars": 2,
        "event_fallback_min_abs_action": 0.26,
        "event_fallback_quality_min": 0.42,
        "event_fallback_prob_floor": 0.36,
    },
    "whipsaw": {
        "side_mode_override": "both",
        "specialist_pos_thresh": 0.24,
        "specialist_close_thresh": 0.14,
        "specialist_force_close_th": -0.010,
        "specialist_rev_exit_thresh": 0.030,
        "specialist_idle_penalty": 0.0,
        "specialist_soft_gate_scale": 0.35,
        "min_val_trades_for_best": 8,
        "target_val_trades": 12,
        "val_side_bias_penalty": 12.0,
        "event_min_prob": 0.56,
        "event_min_edge": 0.010,
        "event_prob_gap": 0.012,
        "event_debounce_bars": 3,
        "event_fallback_bars": 1,
        "event_fallback_min_abs_action": 0.24,
        "event_fallback_quality_min": 0.38,
        "event_fallback_prob_floor": 0.34,
    },
    "normal": {
        "side_mode_override": "long",
        "specialist_pos_thresh": 0.12,
        "specialist_close_thresh": 0.040,
        "min_val_trades_for_best": 15,
        "target_val_trades": 25,
        "val_side_bias_penalty": 0.0,
        "event_min_prob": 0.52,
        "event_min_edge": 0.006,
        "event_prob_gap": 0.004,
        "event_debounce_bars": 1,
        "event_fallback_bars": 2,
        "event_fallback_min_abs_action": 0.20,
        "event_fallback_quality_min": 0.34,
        "event_fallback_prob_floor": 0.32,
    },
}

PROFILE_DEFAULTS = {
    "specialist_force_close_th": None,
    "specialist_rev_exit_thresh": None,
    "specialist_idle_penalty": None,
    "specialist_soft_gate_scale": 0.0,
    "target_val_trades": None,
    "event_fallback_bars": 0,
    "event_fallback_min_abs_action": 0.0,
    "event_fallback_quality_min": 0.0,
    "event_fallback_prob_floor": 0.0,
}


def _parse_regimes(raw: str) -> list[str]:
    vals = [x.strip().lower() for x in str(raw or "").split(",") if x.strip()]
    if not vals:
        return list(ALL_REGIMES)
    out = []
    for regime in vals:
        if regime not in ALL_REGIMES:
            raise ValueError(f"invalid regime: {regime}")
        if regime not in out:
            out.append(regime)
    return out


def _expert_paths(base_dir: str, regime: str) -> dict[str, str]:
    regime_dir = os.path.join(base_dir, regime)
    os.makedirs(regime_dir, exist_ok=True)
    return {
        "dir": regime_dir,
        "config_json_path": os.path.join(regime_dir, "train_config.json"),
        "checkpoint_path": os.path.join(regime_dir, "checkpoint.pth"),
        "best_path": os.path.join(regime_dir, "best.pth"),
        "summary_path": os.path.join(regime_dir, "summary.json"),
    }


def _dump_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_masked_experts(
    csv_path: str,
    regimes: list[str],
    train_ratio: float,
    episodes_per_regime: int,
    fresh_start: bool,
    base_ckpt_dir: str,
    hmm_cache_path: str,
    device: str,
) -> None:
    manifest = {
        "started_at": datetime.now().isoformat(),
        "csv_path": csv_path,
        "train_ratio": float(train_ratio),
        "episodes_per_regime": int(episodes_per_regime),
        "regimes": regimes,
        "experts": {},
    }
    os.makedirs(base_ckpt_dir, exist_ok=True)
    manifest_path = os.path.join(base_ckpt_dir, "manifest.json")
    _dump_json(manifest_path, manifest)

    for regime in regimes:
        profile = dict(PROFILE_DEFAULTS)
        profile.update(REGIME_PROFILES[regime])
        paths = _expert_paths(base_ckpt_dir, regime)
        manifest["experts"][regime] = {"profile": profile, "status": "running"}
        _dump_json(manifest_path, manifest)

        try:
            train(
                csv_path=csv_path,
                train_ratio=train_ratio,
                episodes=episodes_per_regime,
                fresh_start=fresh_start,
                val_interval=10,
                early_stop_patience=12,
                min_val_trades_for_best=int(profile["min_val_trades_for_best"]),
                target_val_trades=(
                    int(profile["target_val_trades"])
                    if profile["target_val_trades"] is not None
                    else int(profile["min_val_trades_for_best"])
                ),
                val_trade_shortfall_penalty=18.0,
                val_trade_bonus_cap=7.0,
                val_side_bias_penalty=float(profile["val_side_bias_penalty"]),
                hmm_cache_path=hmm_cache_path,
                config_json_path=paths["config_json_path"],
                checkpoint_path=paths["checkpoint_path"],
                best_path=paths["best_path"],
                focus_regime=regime,
                terminate_on_regime_change=True,
                side_mode_override=profile["side_mode_override"],
                specialist_pos_thresh=profile["specialist_pos_thresh"],
                specialist_close_thresh=profile["specialist_close_thresh"],
                specialist_idle_penalty=profile["specialist_idle_penalty"],
                specialist_force_close_th=profile["specialist_force_close_th"],
                specialist_rev_exit_thresh=profile["specialist_rev_exit_thresh"],
                specialist_soft_gate_scale=float(profile["specialist_soft_gate_scale"]),
                batch_size=64,
                update_freq=2,
                min_buffer=128,
                warmup_steps=128,
                device=device,
                skip_focus_segment_filter=False,
                event_entry_filter_enable=True,
                event_prob_prefix="evt_oof",
                event_min_prob=float(profile["event_min_prob"]),
                event_min_edge=float(profile["event_min_edge"]),
                event_prob_gap=float(profile["event_prob_gap"]),
                event_debounce_bars=int(profile["event_debounce_bars"]),
                event_fallback_bars=int(profile["event_fallback_bars"]),
                event_fallback_min_abs_action=float(profile["event_fallback_min_abs_action"]),
                event_fallback_quality_min=float(profile["event_fallback_quality_min"]),
                event_fallback_prob_floor=float(profile["event_fallback_prob_floor"]),
            )
        except Exception as exc:
            manifest["experts"][regime].update({
                "status": "failed",
                "failed_at": datetime.now().isoformat(),
                "error": str(exc),
            })
            _dump_json(paths["summary_path"], manifest["experts"][regime])
            _dump_json(manifest_path, manifest)
            raise

        best_exists = os.path.exists(paths["best_path"])
        ckpt_exists = os.path.exists(paths["checkpoint_path"])
        manifest["experts"][regime].update({
            "status": "completed" if best_exists else "completed_no_best",
            "completed_at": datetime.now().isoformat(),
            "best_exists": bool(best_exists),
            "best_path": paths["best_path"] if best_exists else None,
            "checkpoint_path": paths["checkpoint_path"] if ckpt_exists else None,
            "admission": "candidate" if best_exists else "disabled",
        })
        if not best_exists:
            manifest["experts"][regime]["admission_reason"] = (
                "No best.pth was produced; validation never met the minimum trade/score criteria."
            )
        _dump_json(paths["summary_path"], manifest["experts"][regime])
        _dump_json(manifest_path, manifest)

    manifest["finished_at"] = datetime.now().isoformat()
    missing_best = [
        regime for regime in regimes
        if not bool(manifest["experts"].get(regime, {}).get("best_exists", False))
    ]
    if missing_best:
        policy = {
            "version": 1,
            "mode": "auto_missing_best_guard_v1",
            "disabled_regimes": list(missing_best),
            "shadow_regimes": [],
            "reason": (
                "One or more requested regime experts did not produce best.pth; "
                "route them to cash instead of forcing evaluator/runtime load failures."
            ),
        }
        manifest["router_policy"] = dict(policy)
        _dump_json(os.path.join(base_ckpt_dir, "router_policy.json"), policy)
    _dump_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train full-timeline event-masked DSAC regime experts")
    p.add_argument("--csv-path", default=DEFAULT_CSV)
    p.add_argument("--regimes", default=",".join(ALL_REGIMES))
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--episodes-per-regime", type=int, default=180)
    p.add_argument("--fresh-start", action="store_true")
    p.add_argument("--base-ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--hmm-cache-path", default=os.path.join(DEFAULT_CKPT_DIR, "hmm_init_cache_dsac_event_masked.npz"))
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--startup-check-only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.startup_check_only:
        print("startup check ok: train_rl_dsac_event_masked_experts")
        raise SystemExit(0)
    run_masked_experts(
        csv_path=args.csv_path,
        regimes=_parse_regimes(args.regimes),
        train_ratio=args.train_ratio,
        episodes_per_regime=args.episodes_per_regime,
        fresh_start=args.fresh_start,
        base_ckpt_dir=args.base_ckpt_dir,
        hmm_cache_path=args.hmm_cache_path,
        device=args.device,
    )
