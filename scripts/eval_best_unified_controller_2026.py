#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from features.schema import prune_to_feature_keep
from ensemble.train_rl_dsac_unified_controller import (
    AI_FEATS,
    DSAC_ACTION_DIM,
    DSAC_STATE_DIM,
    DSAC_STATE_SCHEMA,
    REGIME_COLS,
    _CLOSE_THRESH,
    _POS_THRESH,
    _hmm_cache_key,
    _load_hmm_cache,
    _resolve_runtime_device,
    DSACAgent,
    DSACCompactTradingEnv,
    MultiTimeframeFeatures,
    OnlineHMMDetector,
)


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}




def _load_rl_frame(csv_path: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
    before_cols = len(df.columns)
    df = prune_to_feature_keep(df, include_entry_price=False, extra_keep=["timestamp", "ai_ready"] + AI_FEATS)
    if len(df.columns) != before_cols:
        print(f"[DATA] feature prune: {before_cols} -> {len(df.columns)} cols")
    if "ai_ready" in df.columns:
        ready = pd.to_numeric(df["ai_ready"], errors="coerce").fillna(0.0)
        df = df.loc[ready >= 0.5].reset_index(drop=True)
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start)].reset_index(drop=True)
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end)].reset_index(drop=True)
    return df


def _load_training_hmm(train_cfg: dict[str, Any]) -> OnlineHMMDetector:
    csv_path = str(train_cfg.get("csv_path", "data/rl_training_2025_unified.csv"))
    train_ratio = float(train_cfg.get("train_ratio", 0.8))
    hmm_cache_path = str(train_cfg.get("hmm_cache_path", "data/ensemble/ckpt/hmm_init_cache_dsac_unified_controller.npz"))
    hmm_fit_iter = 30
    hmm_key = _hmm_cache_key(csv_path=csv_path, train_ratio=train_ratio, n_iter=hmm_fit_iter)
    hmm_detector = None if bool(train_cfg.get("hmm_force_refit", False)) else _load_hmm_cache(hmm_cache_path, hmm_key)
    if hmm_detector is not None:
        print(f"[HMM] cache loaded: {hmm_cache_path}")
        return hmm_detector

    print("[HMM] cache miss; refitting from training CSV to mirror validation setup...")
    train_df_full = _load_rl_frame(csv_path)
    split_idx = int(len(train_df_full) * train_ratio)
    df_train = train_df_full.iloc[:split_idx].reset_index(drop=True)
    hmm_detector = OnlineHMMDetector()
    hmm_detector.fit(df_train, n_iter=hmm_fit_iter)
    return hmm_detector


def _apply_normal_soft_gate_exact(
    action: np.ndarray | float,
    state_vec: np.ndarray,
    regime: str,
    gate_scale: float,
    enabled: bool,
) -> np.ndarray | float:
    a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    scalar = a.size <= 1
    if a.size == 0:
        a = np.zeros(DSAC_ACTION_DIM, dtype=np.float32)
    if not enabled:
        return float(a[0]) if scalar else a

    gs = float(np.clip(gate_scale, 0.0, 1.0))
    if gs <= 1e-9:
        return float(a[0]) if scalar else a

    def _out(v: np.ndarray):
        return float(v[0]) if scalar else v.astype(np.float32)

    def _mix(mult: float):
        out = a.copy()
        m = 1.0 - gs * (1.0 - float(mult))
        out[0] *= m
        if out.size > 1:
            out[1:] = -1.0 + (out[1:] + 1.0) * m
        return _out(out)

    try:
        trend_entropy = float(state_vec[3])
        q_uncertainty = float(state_vec[7])
    except Exception:
        return _out(a)

    if regime == "whipsaw":
        if trend_entropy > 0.24:
            return _mix(0.80)
        return _out(a)
    if regime == "chop":
        if trend_entropy > 0.28 and q_uncertainty > 0.35:
            return _mix(0.40)
        return _mix(0.60)
    if regime != "normal":
        return _out(a)
    if trend_entropy > 0.30 and q_uncertainty > 0.40:
        return _mix(0.45)
    if trend_entropy > 0.24 and q_uncertainty > 0.32:
        return _mix(0.65)
    return _out(a)


def _soft_gate_scale(ep_now: int, warmup_epochs: int, ramp_epochs: int) -> float:
    warm = int(max(0, warmup_epochs))
    ramp = int(max(1, ramp_epochs))
    if ep_now <= warm:
        return 0.0
    if ep_now >= warm + ramp:
        return 1.0
    return float((ep_now - warm) / ramp)


def _build_agent_from_ckpt(
    ckpt_path: str,
    train_cfg: dict[str, Any],
    device: str,
) -> tuple[DSACAgent, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_state_dim = int(ckpt.get("state_dim", -1))
    ckpt_action_dim = int(ckpt.get("action_dim", -1))
    ckpt_state_schema = str(ckpt.get("state_schema", ""))
    if ckpt_state_dim != int(DSAC_STATE_DIM):
        raise ValueError(f"checkpoint state_dim={ckpt_state_dim} != current state_dim={DSAC_STATE_DIM}")
    if ckpt_action_dim != int(DSAC_ACTION_DIM):
        raise ValueError(f"checkpoint action_dim={ckpt_action_dim} != current action_dim={DSAC_ACTION_DIM}")
    if ckpt_state_schema != DSAC_STATE_SCHEMA:
        raise ValueError(
            f"checkpoint state_schema={ckpt_state_schema or 'missing'} != current state_schema={DSAC_STATE_SCHEMA}"
        )

    agent = DSACAgent(
        DSAC_STATE_DIM,
        hidden_dim=256,
        gamma=float(train_cfg.get("gamma", 0.99)),
        n_quantiles=32,
        cvar_frac=float(train_cfg.get("cvar_frac", 0.40)),
        pessimism_min_weight=float(train_cfg.get("pessimism_min_weight", 0.65)),
        adaptive_pessimism=bool(train_cfg.get("adaptive_pessimism", False)),
        pessimism_disagree_scale=float(train_cfg.get("pessimism_disagree_scale", 0.15)),
        pessimism_weight_min=float(train_cfg.get("pessimism_weight_min", 0.55)),
        pessimism_weight_max=float(train_cfg.get("pessimism_weight_max", 0.75)),
        dynamic_entropy=bool(train_cfg.get("dynamic_entropy", True)),
        entropy_min=float(train_cfg.get("entropy_min", -0.80)),
        entropy_max=float(train_cfg.get("entropy_max", -0.45)),
        entropy_std_low=float(train_cfg.get("entropy_std_low", 0.18)),
        entropy_std_high=float(train_cfg.get("entropy_std_high", 0.35)),
        entropy_step=float(train_cfg.get("entropy_step", 0.05)),
        critic_var_weight=bool(train_cfg.get("critic_var_weight", False)),
        critic_var_scale=float(train_cfg.get("critic_var_scale", 1.0)),
        critic_var_w_min=float(train_cfg.get("critic_var_w_min", 0.25)),
        primacy_soft_reset=bool(train_cfg.get("primacy_soft_reset", False)),
        primacy_window=int(train_cfg.get("primacy_window", 80)),
        primacy_imbalance_th=float(train_cfg.get("primacy_imbalance_th", 0.60)),
        primacy_entropy_low=float(train_cfg.get("primacy_entropy_low", 0.45)),
        primacy_reset_cooldown=int(train_cfg.get("primacy_reset_cooldown", 120)),
        direction_reg_lambda=float(train_cfg.get("direction_reg_lambda", 0.08)),
        side_balance_lambda=float(train_cfg.get("side_balance_lambda", 0.12)),
        cql_reg=bool(train_cfg.get("cql_reg", False)),
        cql_alpha=float(train_cfg.get("cql_alpha", 0.02)),
        redo_enable=bool(train_cfg.get("redo_enable", False)),
        redo_interval=int(train_cfg.get("redo_interval", 500)),
        redo_tau=float(train_cfg.get("redo_tau", 0.005)),
        redo_ratio=float(train_cfg.get("redo_ratio", 0.10)),
        alpha_min=float(train_cfg.get("alpha_min", 0.005)),
        alpha_init=float(train_cfg.get("alpha_init", 0.03)),
        anti_flat_lambda=float(train_cfg.get("anti_flat_lambda", 0.08)),
        anti_flat_min_abs=float(train_cfg.get("anti_flat_min_abs", 0.18)),
        anti_flat_anneal_updates=int(train_cfg.get("anti_flat_anneal_updates", 120000)),
        device=device,
    )
    agent.actor.load_state_dict(ckpt["actor"])
    agent.actor.eval()
    return agent, ckpt


def _eval_policy_exact(
    eval_df: pd.DataFrame,
    agent: DSACAgent,
    hmm_detector: OnlineHMMDetector,
    train_cfg: dict[str, Any],
    ckpt_epoch: int,
) -> dict[str, float]:
    if len(eval_df) < 32:
        return {
            "pnl": 0.0,
            "wr": 0.0,
            "mdd": 0.0,
            "tr": 0,
            "long_entries": 0,
            "short_entries": 0,
            "fcl": 0,
            "fcs": 0,
            "avg_hold_long": 0.0,
            "avg_hold_short": 0.0,
            "side_balance": 0.0,
            "side_pen": 0.0,
            "score": -5.0,
        }

    dd_coeff = float(os.getenv("DSAC_DD_PENALTY_COEFF", "0.03"))
    kelly_align = float(os.getenv("DSAC_KELLY_ALIGN_BONUS", "0.0"))
    chop_loss = float(os.getenv("DSAC_KELLY_CHOP_LOSS_PENALTY", "1.30"))
    adverse_hold = _env_flag("DSAC_ADVERSE_HOLD_ENABLE", False)
    eval_mtf = MultiTimeframeFeatures(eval_df["close"].values.astype(np.float32))
    env = DSACCompactTradingEnv(
        eval_df.reset_index(drop=True),
        phase="val",
        hmm_detector=copy.deepcopy(hmm_detector),
        mtf_features=eval_mtf,
        specialist_pos_thresh=float(train_cfg.get("specialist_pos_thresh", _POS_THRESH)),
        specialist_close_thresh=float(train_cfg.get("specialist_close_thresh", _CLOSE_THRESH)),
        dd_penalty_coeff=dd_coeff,
        kelly_align_bonus=kelly_align,
        kelly_chop_loss_penalty=chop_loss,
        adverse_hold_enable=adverse_hold,
        terminal_reward_scale=0.0,
        terminal_quality_win=0.0,
        terminal_quality_loss=0.0,
    )
    st = env.reset()
    done = False
    peak_eq = float(env.initial_balance)
    mdd_pct = 0.0
    le = 0
    se = 0
    fcl = 0
    fcs = 0
    hs_l = 0
    hs_s = 0
    hn_l = 0
    hn_s = 0
    gate_scale = max(
        0.50,
        _soft_gate_scale(
            ckpt_epoch,
            int(train_cfg.get("soft_gate_warmup_epochs", 20)),
            int(train_cfg.get("soft_gate_ramp_epochs", 80)),
        ),
    )
    soft_gate_enabled = bool(train_cfg.get("soft_gate_enabled", False))

    while not done:
        prev_pos = env.pos
        with torch.no_grad():
            action = agent.act(st, deterministic=True)
        action = _apply_normal_soft_gate_exact(
            action,
            st,
            env.regime_bucket(),
            gate_scale=gate_scale,
            enabled=soft_gate_enabled,
        )
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
    wr = env.win_rate
    if env.total_trades == 0:
        trade_score = -5.0
    elif pnl > 0:
        trade_score = min(env.total_trades / 30.0, 1.0) * 5.0
    else:
        trade_score = -min(env.total_trades / 30.0, 1.0) * 10.0
    side_total_entries = int(le + se)
    side_balance = float(min(le, se) / side_total_entries) if side_total_entries > 0 else 0.0
    side_imbalance = float(abs(le - se) / max(side_total_entries, 1))
    side_bias_pen = float(train_cfg.get("val_side_bias_penalty", 80.0)) * side_imbalance
    score = pnl * 3.0 + wr * 60.0 + trade_score + mdd_pct * 2.0 - side_bias_pen
    return {
        "pnl": float(pnl),
        "wr": float(wr),
        "mdd": float(mdd_pct),
        "tr": int(env.total_trades),
        "long_entries": int(le),
        "short_entries": int(se),
        "fcl": int(fcl),
        "fcs": int(fcs),
        "avg_hold_long": float(hs_l / max(hn_l, 1)) if hn_l > 0 else 0.0,
        "avg_hold_short": float(hs_s / max(hn_s, 1)) if hn_s > 0 else 0.0,
        "side_balance": float(side_balance),
        "side_pen": float(side_bias_pen),
        "score": float(score),
    }


def _regime_breakdown(
    eval_df: pd.DataFrame,
    agent: DSACAgent,
    hmm_detector: OnlineHMMDetector,
    train_cfg: dict[str, Any],
    ckpt_epoch: int,
) -> tuple[list[tuple[str, int, float, float, int]], float]:
    reg_cols = [c for c in REGIME_COLS if c in eval_df.columns]
    if not reg_cols:
        return [], 0.0
    reg_idx = np.argmax(eval_df[reg_cols].to_numpy(dtype=np.float64), axis=1)
    reg_weight = {"normal": 0.40, "bear": 0.15, "bull": 0.15, "chop": 0.15, "whipsaw": 0.15}
    w_sum = 0.0
    w_score = 0.0
    reg_logs: list[tuple[str, int, float, float, int]] = []
    for i, rc in enumerate(reg_cols):
        rname = rc.replace("regime_", "")
        sub = eval_df.iloc[reg_idx == i].copy()
        if len(sub) < 64:
            continue
        rs = _eval_policy_exact(sub, agent, hmm_detector, train_cfg, ckpt_epoch)
        w = float(reg_weight.get(rname, 0.0))
        w_score += w * float(rs["score"])
        w_sum += w
        reg_logs.append((rname, len(sub), float(rs["score"]), float(rs["pnl"]), int(rs["tr"])))
    regime_score = float(w_score / w_sum) if w_sum > 0 else 0.0
    return reg_logs, regime_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default="data/rl_training_2026_unified.csv")
    ap.add_argument("--ckpt-path", default="data/ensemble/ckpt/best_dsac_unified_controller.pth")
    ap.add_argument("--config-path", default="data/ensemble/ckpt/dsac_unified_controller_train_config.json")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        train_cfg = json.load(f)

    device = _resolve_runtime_device(args.device)
    eval_df = _load_rl_frame(args.csv_path, args.start, args.end)
    hmm_detector = _load_training_hmm(train_cfg)
    agent, ckpt = _build_agent_from_ckpt(args.ckpt_path, train_cfg, device)
    ckpt_epoch = int(ckpt.get("epoch", train_cfg.get("episodes", 0)))

    overall = _eval_policy_exact(eval_df, agent, hmm_detector, train_cfg, ckpt_epoch)
    reg_logs, regime_score = _regime_breakdown(eval_df, agent, hmm_detector, train_cfg, ckpt_epoch)
    val_score = 0.5 * float(overall["score"]) + 0.5 * float(regime_score or overall["score"])

    payload = {
        "csv_path": args.csv_path,
        "ckpt_path": args.ckpt_path,
        "config_path": args.config_path,
        "device": device,
        "ckpt_epoch": ckpt_epoch,
        "rows": int(len(eval_df)),
        "start": str(eval_df["timestamp"].iloc[0]) if len(eval_df) and "timestamp" in eval_df.columns else None,
        "end": str(eval_df["timestamp"].iloc[-1]) if len(eval_df) and "timestamp" in eval_df.columns else None,
        "overall": overall,
        "regime_logs": [
            {"regime": rn, "rows": n, "score": sc, "pnl": pnl, "trades": tr}
            for rn, n, sc, pnl, tr in reg_logs
        ],
        "val_score": float(val_score),
    }

    out_json = args.out_json or os.path.join(
        _ROOT_DIR,
        "data/ensemble/reports",
        f"eval_best_unified_controller_2026_exact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(
        "[VAL] PnL:{pnl:.2f}% | Tr:{tr} | WR:{wr:.0f}% | MDD:{mdd:.2f}% | "
        "L:{le} S:{se} | SideBal:{sb:.3f} | SidePen:{sp:.2f} | "
        "FCL:{fcl} FCS:{fcs} | AvgHoldL:{ahl:.1f} AvgHoldS:{ahs:.1f} | Score:{score:.2f}".format(
            pnl=float(overall["pnl"]),
            tr=int(overall["tr"]),
            wr=float(overall["wr"]) * 100.0,
            mdd=float(overall["mdd"]),
            le=int(overall["long_entries"]),
            se=int(overall["short_entries"]),
            sb=float(overall["side_balance"]),
            sp=float(overall["side_pen"]),
            fcl=int(overall["fcl"]),
            fcs=int(overall["fcs"]),
            ahl=float(overall["avg_hold_long"]),
            ahs=float(overall["avg_hold_short"]),
            score=float(val_score),
        )
    )
    if reg_logs:
        reg_msg = " | ".join(
            [f"{rn}:n={n} score={sc:.1f} pnl={pnl:.1f}% tr={tr}" for rn, n, sc, pnl, tr in reg_logs]
        )
        print(f"[VAL REGIME] {reg_msg}")
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
