#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.train_rl_agent import REGIME_COLS  # noqa: E402
from ensemble.train_rl_dsac_agent import DSACCompactTradingEnv, MultiTimeframeFeatures  # noqa: E402
from scripts.eval_best_dsac_agent_2026 import (  # noqa: E402
    _apply_normal_soft_gate_exact,
    _build_agent_from_ckpt,
    _load_rl_frame,
    _load_training_hmm,
    _resolve_runtime_device,
    _soft_gate_scale,
)


DEFAULT_REPORT = ROOT / "data/ensemble/reports/eval_best_dsac_agent_2026_exact_20260511_redteam_audit.json"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _eval_policy_cost(eval_df: pd.DataFrame, agent: Any, hmm_detector: Any, train_cfg: dict[str, Any], ckpt_epoch: int, *, fee_mult: float = 1.0, slip_mult: float = 1.0) -> dict[str, Any]:
    if len(eval_df) < 32:
        return {"pnl": 0.0, "wr": 0.0, "mdd": 0.0, "tr": 0}
    dd_coeff = float(os.getenv("DSAC_DD_PENALTY_COEFF", "0.03"))
    kelly_align = float(os.getenv("DSAC_KELLY_ALIGN_BONUS", "0.0"))
    chop_loss = float(os.getenv("DSAC_KELLY_CHOP_LOSS_PENALTY", "1.30"))
    adverse_hold = _env_flag("DSAC_ADVERSE_HOLD_ENABLE", False)
    env = DSACCompactTradingEnv(
        eval_df.reset_index(drop=True),
        phase="val",
        hmm_detector=copy.deepcopy(hmm_detector),
        mtf_features=MultiTimeframeFeatures(eval_df["close"].values.astype(np.float32)),
        specialist_pos_thresh=float(train_cfg.get("specialist_pos_thresh", 0.12)),
        specialist_close_thresh=float(train_cfg.get("specialist_close_thresh", 0.03)),
        dd_penalty_coeff=dd_coeff,
        kelly_align_bonus=kelly_align,
        kelly_chop_loss_penalty=chop_loss,
        adverse_hold_enable=adverse_hold,
        terminal_reward_scale=0.0,
        terminal_quality_win=0.0,
        terminal_quality_loss=0.0,
        fee=0.0005 * float(fee_mult),
        slip=0.0002 * float(slip_mult),
    )
    st = env.reset()
    done = False
    peak_eq = float(env.initial_balance)
    mdd_pct = 0.0
    long_entries = short_entries = force_closed_long = force_closed_short = 0
    hold_sum_long = hold_sum_short = hold_n_long = hold_n_short = 0
    gate_scale = max(0.50, _soft_gate_scale(ckpt_epoch, int(train_cfg.get("soft_gate_warmup_epochs", 20)), int(train_cfg.get("soft_gate_ramp_epochs", 80))))
    while not done:
        prev_pos = env.pos
        with torch.no_grad():
            action = agent.act(st, deterministic=True)
        action = _apply_normal_soft_gate_exact(action, st, env.regime_bucket(), gate_scale=gate_scale)
        st, _reward, done, info = env.step(action)
        if prev_pos is None and env.pos == "LONG":
            long_entries += 1
        elif prev_pos is None and env.pos == "SHORT":
            short_entries += 1
        if bool(info.get("force_closed", False)):
            closed_side = str(info.get("closed_side", "") or "")
            if closed_side == "LONG":
                force_closed_long += 1
            elif closed_side == "SHORT":
                force_closed_short += 1
        hold = int(info.get("closed_hold_count", 0) or 0)
        closed_side = str(info.get("closed_side", "") or "")
        if hold > 0 and closed_side == "LONG":
            hold_sum_long += hold
            hold_n_long += 1
        elif hold > 0 and closed_side == "SHORT":
            hold_sum_short += hold
            hold_n_short += 1
        cur_eq = env.balance * (1.0 + env.unrealized_pnl if env.pos is not None else 1.0)
        peak_eq = max(peak_eq, cur_eq)
        mdd_pct = min(mdd_pct, (cur_eq / max(peak_eq, 1e-8) - 1.0) * 100.0)
    pnl = (env.balance / env.initial_balance - 1.0) * 100.0
    entries = max(long_entries + short_entries, 1)
    return {
        "pnl": float(pnl),
        "wr": float(env.win_rate),
        "mdd": float(mdd_pct),
        "tr": int(env.total_trades),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "side_balance": float(min(long_entries, short_entries) / entries),
        "force_closed_long": int(force_closed_long),
        "force_closed_short": int(force_closed_short),
        "avg_hold_long": float(hold_sum_long / max(hold_n_long, 1)) if hold_n_long else 0.0,
        "avg_hold_short": float(hold_sum_short / max(hold_n_short, 1)) if hold_n_short else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Red-team audit for exact standalone DSAC 2026 evaluation.")
    p.add_argument("--csv-path", default="data/rl_training_2026_unified.csv")
    p.add_argument("--ckpt-path", default="data/ensemble/ckpt/best_dsac_agents.pth")
    p.add_argument("--config-path", default="data/ensemble/ckpt/dsac_train_config_latest.json")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-json", type=Path, default=DEFAULT_REPORT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    train_cfg = json.loads(Path(args.config_path).read_text(encoding="utf-8"))
    device = _resolve_runtime_device(args.device)
    eval_df = _load_rl_frame(args.csv_path)
    hmm_detector = _load_training_hmm(train_cfg)
    agent, ckpt = _build_agent_from_ckpt(args.ckpt_path, train_cfg, device)
    ckpt_epoch = int(ckpt.get("epoch", train_cfg.get("episodes", 0)))
    base = _eval_policy_cost(eval_df, agent, hmm_detector, train_cfg, ckpt_epoch, fee_mult=1.0, slip_mult=1.0)
    cost2 = _eval_policy_cost(eval_df, agent, hmm_detector, train_cfg, ckpt_epoch, fee_mult=2.0, slip_mult=2.0)
    cost3 = _eval_policy_cost(eval_df, agent, hmm_detector, train_cfg, ckpt_epoch, fee_mult=3.0, slip_mult=3.0)
    zero_legacy = eval_df.copy()
    legacy_present = [c for c in REGIME_COLS if c in zero_legacy.columns]
    for col in legacy_present:
        zero_legacy[col] = 0.0
    legacy_zero = _eval_policy_cost(zero_legacy, agent, hmm_detector, train_cfg, ckpt_epoch, fee_mult=1.0, slip_mult=1.0)
    ms_cols = list(train_cfg.get("market_state_cols") or [])
    zero_market_state = eval_df.copy()
    ms_present = [c for c in ms_cols if c in zero_market_state.columns]
    for col in ms_present:
        zero_market_state[col] = 0.0
    market_state_zero = _eval_policy_cost(zero_market_state, agent, hmm_detector, train_cfg, ckpt_epoch, fee_mult=1.0, slip_mult=1.0)

    cfg_csv = str(train_cfg.get("csv_path", ""))
    blocking: list[str] = []
    warnings: list[str] = []
    if "2026" in cfg_csv:
        blocking.append("dsac_training_config_mentions_2026")
    if "2025" not in cfg_csv:
        warnings.append("dsac_training_config_not_explicit_2025")
    if legacy_present:
        warnings.append("policy_uses_legacy_regime_columns:" + ",".join(legacy_present))
    if base["pnl"] - legacy_zero["pnl"] > 50.0:
        blocking.append("legacy_regime_ablation_large_pnl_drop")
    if base["pnl"] - market_state_zero["pnl"] > 100.0:
        warnings.append("market_state_ablation_large_pnl_drop")
    if cost3["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    if abs(float(base["mdd"])) > 15.0:
        warnings.append("base_mdd_above_15pct")

    payload = {
        "status": "pass" if not blocking else "fail",
        "verdict": "candidate_promotable" if not blocking and base["pnl"] >= 100.0 and abs(float(base["mdd"])) <= 15.0 and cost3["pnl"] > 0.0 else "do_not_promote",
        "blocking": blocking,
        "warnings": warnings,
        "csv_path": args.csv_path,
        "ckpt_path": args.ckpt_path,
        "config_path": args.config_path,
        "device": device,
        "ckpt_epoch": ckpt_epoch,
        "train_config": {
            "csv_path": cfg_csv,
            "saved_at": train_cfg.get("saved_at"),
            "train_ratio": train_cfg.get("train_ratio"),
            "market_state_cols": ms_cols,
            "terminate_on_regime_change": train_cfg.get("terminate_on_regime_change"),
        },
        "checks": {
            "base_cost1": base,
            "base_cost2": cost2,
            "base_cost3": cost3,
            "legacy_regime_zero_ablation": legacy_zero,
            "market_state_zero_ablation": market_state_zero,
            "legacy_regime_cols_present": legacy_present,
            "market_state_cols_present": ms_present,
            "eval_range": [str(eval_df["timestamp"].iloc[0]), str(eval_df["timestamp"].iloc[-1])],
            "eval_rows": int(len(eval_df)),
        },
        "notes": [
            "This audit does not tune on 2026; it reproduces a pre-existing checkpoint and stress-tests it.",
            "Legacy regime columns are treated as a red-team risk because previous project bugs involved regime-derived artifacts.",
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"audit": str(args.out_json), "status": payload["status"], "verdict": payload["verdict"], "checks": payload["checks"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
