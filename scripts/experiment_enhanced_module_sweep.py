#!/usr/bin/env python3
"""Sweep ENH_ENABLE_* flags and evaluate DSAC+Enhanced engine performance."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from ensemble.train_rl_dsac_agent import DSACRouter, GaussianActor, DSAC_STATE_DIM
from trading_bot import DSACTrendRouter
from enhanced_trading_engine import EnhancedTradingEngine, PartialExitLevel

MODULE_FLAGS = [
    ("micro", "ENH_ENABLE_MICRO"),
    ("momentum", "ENH_ENABLE_MOMENTUM"),
    ("partial", "ENH_ENABLE_PARTIAL"),
    ("cross_asset", "ENH_ENABLE_CROSS_ASSET"),
    ("bayes", "ENH_ENABLE_BAYES"),
]

BASE_ENV = {
    "ENH_ENABLE_MICRO": False,
    "ENH_ENABLE_VOL_REGIME": True,
    "ENH_ENABLE_CVAR_KELLY": True,
    "ENH_ENABLE_ATR_STOP": True,
    "ENH_ENABLE_MOMENTUM": False,
    "ENH_ENABLE_FUNDING": True,
    "ENH_ENABLE_PARTIAL": False,
    "ENH_ENABLE_CROSS_ASSET": False,
    "ENH_ENABLE_BAYES": False,
    "ENH_ENABLE_SESSION": True,
    "ENH_RUNTIME_ENABLE": True,
}

@dataclass
class RunMetrics:
    split: str
    case: str
    pnl_pct: float
    trades: int
    win_rate: float
    mdd_pct: float
    sharpe: float


def load_actor(path: str, device="cpu") -> GaussianActor:
    ckpt = pd.read_pickle(path) if path.endswith(".pkl") else None
    if ckpt is None:
        import torch
        data = torch.load(path, map_location=device)
        actor = GaussianActor(state_dim=int(data.get("state_dim", DSAC_STATE_DIM)))
        actor.load_state_dict(data["actor"])
    else:
        actor = GaussianActor(state_dim=int(ckpt.get("state_dim", DSAC_STATE_DIM)))
        actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def compute_session_flags(ts: pd.Timestamp) -> Dict[str, float]:
    hour = int(pd.Timestamp(ts).hour)
    return {
        "session_asia": 1.0 if 0 <= hour < 9 else 0.0,
        "session_europe": 1.0 if 6 <= hour < 15 else 0.0,
        "session_us": 1.0 if 11 <= hour < 20 else 0.0,
    }


def compute_diag_multiplier(row: pd.Series) -> Tuple[float, Dict[str, float]]:
    qwidth = max(float(row.get("m7_qwidth", 0.005) or 0.005), 1e-6)
    bayes_bias = float(row.get("m7_prob_up", 0.0)) - float(row.get("m7_prob_dn", 0.0))
    smart_flow = float(row.get("smart_money_flow", 0.0))
    mtf1 = float(row.get("mtf_trend_1h", 0.0))
    mtf4 = float(row.get("mtf_trend_4h", 0.0))
    q_mult = np.clip(1.0 + (0.015 - qwidth) / 0.05, 0.85, 1.2)
    smart_mult = np.clip(1.0 + smart_flow * 0.05, 0.8, 1.25)
    mtf_mult = 1.12 if mtf1 * mtf4 > 0 else 0.88
    bayes_mult = np.clip(1.0 + bayes_bias * 0.15, 0.8, 1.25)
    total = float(np.clip(q_mult * smart_mult * mtf_mult * bayes_mult, 0.6, 1.4))
    diag = {
        "bayes_z": bayes_bias,
        "qwidth": qwidth,
        "smart_flow": smart_flow,
        "mtf_align": float(mtf_mult),
    }
    return total, diag


def set_env(vars_to_set: Dict[str, bool | str]) -> Dict[str, str]:
    backup = {k: os.environ.get(k) for k in vars_to_set}
    for k, v in vars_to_set.items():
        if isinstance(v, bool):
            os.environ[k] = "1" if v else "0"
        else:
            os.environ[k] = str(v)
    return backup


def restore_env(backup: Dict[str, str]) -> None:
    for k, v in backup.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def simulate(df: pd.DataFrame, router: DSACRouter, engine: EnhancedTradingEngine) -> RunMetrics:
    diag_enabled = getattr(engine, "diag_mode", False)
    n = len(df)
    trades = 0
    wins = 0
    balance = 1.0
    eq = [1.0]
    pos = None
    entry = 0.0
    lev = 0.0
    slack = 0.0002
    slip_fee = 0.0005
    meta_router = DSACTrendRouter()
    for i in range(n - 1):
        row = df.iloc[i]
        stats = row.drop(labels=["timestamp"]).to_dict()
        stats = {k: float(v or 0.0) for k, v in stats.items()}
        context = {
            "type": pos,
            "entry_price": entry,
            "unrealized": 0.0,
            "mdd": 0.0,
            "hold_norm": float(min(meta_router.hold_count / 96.0, 1.0)),
        }
        action_cont, lev_raw, info = router.decide(stats, context)
        dsac_action = int(action_cont)
        dsac_kelly = float(np.clip(lev_raw, 0.0, 1.0))
        mult = 1.0
        if diag_enabled:
            mult, diag_vals = compute_diag_multiplier(row)
        dsac_kelly *= mult
        processed_df = df.iloc[max(0, i - 120): i + 1].reset_index(drop=True)
        session_flags = compute_session_flags(row["timestamp"])
        enhanced = engine.process(
            dsac_action=dsac_action,
            dsac_kelly=dsac_kelly,
            dsac_info=info,
            processed_df=processed_df,
            eth_buffer=processed_df,
            btc_buffer=processed_df,
            meta_router=meta_router,
            regime={},
            trend_signal=None,
            session_flags=session_flags,
        )
        action = int(enhanced.get("action", dsac_action))
        kelly = float(np.clip(enhanced.get("kelly", dsac_kelly), 0.0, 1.0))
        current_price = float(df.iloc[i + 1]["open"])
        if pos is None:
            if action == 1 and kelly > 0.0:
                pos = "LONG"
                entry = current_price * (1.0 + slip_fee)
                lev = kelly
                balance -= balance * slip_fee * lev
                meta_router._update_pos(action, current_price, leverage=kelly)
            elif action == 2 and kelly > 0.0:
                pos = "SHORT"
                entry = current_price * (1.0 - slip_fee)
                lev = kelly
                balance -= balance * slip_fee * lev
                meta_router._update_pos(action, current_price, leverage=kelly)
        else:
            close_price = current_price
            close_trade = False
            pnl = 0.0
            if action == 0 or (pos == "LONG" and action == 2) or (pos == "SHORT" and action == 1):
                close_price = current_price
                close_trade = True
                if pos == "LONG":
                    pnl = ((close_price * (1.0 - slip_fee) - entry) / entry) * lev
                else:
                    pnl = ((entry - close_price * (1.0 + slip_fee)) / entry) * lev
            if close_trade:
                balance *= 1.0 + pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                pos = None
                entry = 0.0
                lev = 0.0
                meta_router._update_pos(0, current_price)
                eq.append(balance)
            else:
                eq.append(balance)
                meta_router._update_pos(action, current_price, leverage=kelly)
        if pos is None:
            eq.append(balance)
    eq_arr = np.array(eq, dtype=float)
    run_max = np.maximum.accumulate(eq_arr)
    dd = eq_arr / np.maximum(run_max, 1e-12) - 1.0
    pnl_pct = (eq_arr[-1] - 1.0) * 100.0
    mdd = float(np.min(dd)) * 100.0
    rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-12)
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(365 * 24 * 12)) if len(rets) >= 3 and np.std(rets) > 0 else 0.0
    wr = float(wins) / trades if trades else 0.0
    return RunMetrics(
        split="",
        case="",
        pnl_pct=pnl_pct,
        trades=trades,
        win_rate=wr * 100.0,
        mdd_pct=mdd,
        sharpe=sharpe,
    )


def run_case(
    split: str,
    df: pd.DataFrame,
    actor_path: str,
    case_name: str,
    flag_overrides: Dict[str, bool],
    env_overrides: Dict[str, str] | None = None,
) -> RunMetrics:
    flags = BASE_ENV.copy()
    flags.update(flag_overrides)
    if env_overrides:
        flags.update(env_overrides)
    backup = set_env(flags)
    try:
        actor = load_actor(actor_path)
        router = DSACRouter(actor, device="cpu")
        engine = EnhancedTradingEngine()
        if env_overrides and "PARTIAL_LEVELS" in env_overrides:
            levels = json.loads(env_overrides["PARTIAL_LEVELS"])
            engine.partial_exit.levels = [
                PartialExitLevel(pnl_threshold=l[0], exit_fraction=l[1])
                for l in levels
            ]
            env_overrides = env_overrides.copy()
            env_overrides.pop("PARTIAL_LEVELS")
        if env_overrides and "KELLY_DIAG_MODE" in env_overrides:
            engine.diag_mode = str(env_overrides["KELLY_DIAG_MODE"]).lower() in ("1","true","on","yes")
            env_overrides = env_overrides.copy()
            env_overrides.pop("KELLY_DIAG_MODE")
        metrics = simulate(df, router, engine)
        metrics.split = split
        metrics.case = case_name
        return metrics
    finally:
        restore_env(backup)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-csv", default="data/splits/year_oos/rl_base_2024.csv")
    parser.add_argument("--ckpt", default="data/ensemble/ckpt/best_dsac_agents.pth")
    parser.add_argument("--split", choices=["2024Q4", "2025H1", "2025H2"], default="2024Q4")
    parser.add_argument("--output", default="data/ensemble/metrics/enhanced_module_sweep.json")
    parser.add_argument("--limit", type=int, default=2000,
                        help="Row limit per split to keep runtimes manageable")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.rl_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    masks = {
        "2024Q4": (df["timestamp"] >= "2024-10-01") & (df["timestamp"] <= "2024-12-31"),
        "2025H1": (df["timestamp"] >= "2025-01-01") & (df["timestamp"] <= "2025-06-30"),
        "2025H2": (df["timestamp"] >= "2025-07-01") & (df["timestamp"] <= "2025-12-31"),
    }
    mask = masks[args.split]
    subset = df[mask].copy()
    if args.limit is not None:
        subset = subset.head(args.limit).copy()
    if subset.empty:
        print(f"[warning] no rows for split {args.split}")
        return
    results = []
    cases = [("baseline", {}, None)]
    for name, flag in MODULE_FLAGS:
        cases.append((name, {flag: True}, None))
    # momentum relaxation grid
    momentum_configs = []
    for confirm in (2, 3):
        for window in (12, 18):
            for flip in (5, 7):
                label = f"momentum_relaxed_c{confirm}_w{window}_f{flip}"
                env_override = {
                    "MOM_CONFIRM_BARS": str(confirm),
                    "MOM_CHOP_WINDOW": str(window),
                    "MOM_CHOP_FLIP_MAX": str(flip),
                }
                momentum_configs.append((label, {"ENH_ENABLE_MOMENTUM": True}, env_override))
    cases.extend(momentum_configs)
    cases.append((
        "kelly_diag",
        {"ENH_ENABLE_MOMENTUM": True},
        {"KELLY_DIAG_MODE": "1"},
    ))
    bayes_configs = [
        (
            "bayes_lenient_soft",
            {"ENH_ENABLE_BAYES": True},
            {"BAYES_OFFSET_BASE": "0.03", "BAYES_ADAPT_RATE": "0.02"},
        ),
        (
            "bayes_lenient_agg",
            {"ENH_ENABLE_BAYES": True},
            {"BAYES_OFFSET_BASE": "0.02", "BAYES_ADAPT_RATE": "0.01", "BAYES_OFFSET_SCALE": "0.5"},
        ),
    ]
    cases.extend(bayes_configs)
    partial_configs = []
    for label_suffix in ("partial_modest","partial_aggressive"):
        levels = [[0.004,0.30],[0.010,0.30],[0.015,0.40]] if label_suffix=="partial_modest" else [[0.003,0.35],[0.008,0.35],[0.012,0.40]]
        partial_configs.append((
            f"momentum_{label_suffix}",
            {"ENH_ENABLE_MOMENTUM": True, "ENH_ENABLE_PARTIAL": True},
            {"PARTIAL_LEVELS": json.dumps(levels)}
        ))
    cases.extend(partial_configs)
    cases.append(("all", {flag: True for _, flag in MODULE_FLAGS}, None))
    for case_name, overrides, extra_env in cases:
        metrics = run_case(args.split, subset, args.ckpt, case_name, overrides, extra_env)
        results.append(metrics)
        print(f"[{args.split}] case={case_name} pnl={metrics.pnl_pct:.2f}% trades={metrics.trades} wr={metrics.win_rate:.1f}% mdd={metrics.mdd_pct:.1f}% sharpe={metrics.sharpe:.2f}")
    payload = [metrics.__dict__ for metrics in results]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
