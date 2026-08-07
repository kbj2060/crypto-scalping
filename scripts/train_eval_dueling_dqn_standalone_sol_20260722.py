#!/usr/bin/env python3
"""Standalone Dueling DQN (HOLD/LONG/SHORT) trained directly on SOL raw 5m features.

Design (see task brief 2026-07-22):
- No Alpha1-4 HGB-parent chain, no NeuralForecast features, no `--base-parent`/`--runner-model`.
  State = SOL's own raw engineered 5m technical features (data/splits/year_oos/sol_features_*.csv),
  excluding raw price/volume level columns (non-stationary) and excluding Regime3/CryptoMamba
  derived columns entirely -> because we do NOT consume Regime3/CryptoMamba columns, the
  2024-leakage boundary noted in the task brief (those sidecars were fit on 2024 data) does not
  apply here, so 2024 raw feature data is safely usable for training.
- Reward is a single-step contextual-bandit-style regression target, not multi-bar TD bootstrap:
  for each bar i, forward realized return over a fixed horizon H=12 bars (1h) is computed once,
  and turned into a reward for each of the 3 actions:
    r_hold = 0
    r_long  =  fwd_ret[i] - round_trip_cost
    r_short = -fwd_ret[i] - round_trip_cost
  round_trip_cost = 2 * (fee + slip) at "cost1" (fee=0.0004, slip=0.00015), matching this repo's
  standard cost1/cost2/cost3 convention used elsewhere (train at cost1, stress-eval at cost1-3).
  Because the reward already integrates the full horizon's PnL, gamma (next-state bootstrap) is
  optional extra signal, not required; it is swept in the VAL grid (0.0 = pure bandit regression,
  vs small values that also bootstrap off next-bar's best Q).
- Network/PER: reuses ensemble.dueling_dqn_parent.DuelingQNetwork / DuelingDQNConfig unmodified
  (per guardrail), with a PER-like |TD-error|-prioritized replay sampler adapted from
  scripts/train_eval_alpha5_3_hmm_dqn_router_parent_20260517.py's `_train_dqn_td_priority_only`.
- Execution / backtest: fully causal bar-by-bar walk. At each bar the frozen policy's argmax
  action is compared to current position; on a change, exit (if in position) and/or enter (if
  desired != 0) using NEXT bar's open with slip, matching this repo's `_fill_price` convention.
  No stored ledgers, no future rows, no lookahead.

Fresh-forward discipline: fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.dueling_dqn_parent import DuelingDQNConfig, DuelingQNetwork, make_action_model  # noqa: E402

MODEL_ID = "dueling_dqn_standalone_sol_20260722"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/dueling_dqn_standalone_sol_20260722"
DATA_FILES = [
    ROOT / "data/splits/year_oos/sol_features_2024.csv",
    ROOT / "data/splits/year_oos/sol_features_2025.csv",
    ROOT / "data/splits/year_oos/sol_features_2026.csv",
]

# Raw price/volume level columns excluded from state: non-stationary, and close/open are used
# directly for reward + execution rather than as normalized model inputs.
EXCLUDE_COLS = {
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value", "close_btc", "volume_btc",
    "quote_volume_btc",
}

ACTION_HOLD, ACTION_LONG, ACTION_SHORT = 0, 1, 2
HORIZON_BARS = 12  # 1h @ 5m bars
FEE = 0.0004
SLIP = 0.00015
ROUND_TRIP_COST = 2.0 * (FEE + SLIP)

TRAIN_END = pd.Timestamp("2025-09-01")
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")
FRESH_START = pd.Timestamp("2026-04-01")


def _seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_data() -> pd.DataFrame:
    frames = []
    for p in DATA_FILES:
        df = pd.read_csv(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    regime3_like = [c for c in cols if "regime3" in c.lower() or "cryptomamba" in c.lower() or "hmm" in c.lower()]
    if regime3_like:
        raise ValueError(f"unexpected regime3/cryptomamba columns present in raw feature file: {regime3_like}")
    return cols


def _close(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().to_numpy(dtype=np.float64)


def _open(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df["open"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().to_numpy(dtype=np.float64)


def _fill_price(open_arr: np.ndarray, idx: int, side: int, slip: float, *, entry: bool) -> float:
    px = float(open_arr[int(np.clip(idx, 0, len(open_arr) - 1))])
    if side > 0:
        return px * (1.0 + slip if entry else 1.0 - slip)
    return px * (1.0 - slip if entry else 1.0 + slip)


def _days(df: pd.DataFrame) -> float:
    return max((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400.0, 1e-8)


def _prep_matrix(x: np.ndarray, med: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.where(np.isfinite(x), x, np.nan)
    arr = np.where(np.isfinite(arr), arr, med)
    return ((arr - mean) / std).astype(np.float32)


def _fit_norm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.where(np.isfinite(x), x, np.nan)
    med = np.nanmedian(arr, axis=0).astype(np.float32)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    arr = np.where(np.isfinite(arr), arr, med)
    mean = arr.mean(axis=0).astype(np.float32)
    std = arr.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return med, mean, std


def _build_reward_table(close: np.ndarray, horizon: int, round_trip_cost: float) -> np.ndarray:
    n = len(close)
    fwd_ret = np.full(n, np.nan, dtype=np.float64)
    valid = np.arange(0, max(0, n - horizon))
    fwd_ret[valid] = close[valid + horizon] / np.maximum(close[valid], 1e-12) - 1.0
    r = np.zeros((n, 3), dtype=np.float32)
    r[:, ACTION_HOLD] = 0.0
    r[:, ACTION_LONG] = (fwd_ret - round_trip_cost).astype(np.float32)
    r[:, ACTION_SHORT] = (-fwd_ret - round_trip_cost).astype(np.float32)
    return r


def _train_dqn(
    x: np.ndarray,
    rewards: np.ndarray,
    *,
    cfg: DuelingDQNConfig,
    steps: int,
    batch_size: int,
    gamma: float,
    lr: float,
    seed: int,
    device: torch.device,
    log_every: int,
) -> tuple[DuelingQNetwork, dict[str, Any]]:
    """PER-like TD-error-prioritized training. Each ORIGINAL row is tripled into 3 replay
    tuples (one per action), sharing the same next-state (row i+1) since state does not carry
    position context. Reward already integrates the full horizon PnL for that action; gamma
    controls how much extra next-bar bootstrap signal is blended in (0.0 = pure regression)."""
    _seed(seed)
    n = int(len(x))
    next_idx = np.minimum(np.arange(n) + 1, n - 1)
    done = np.zeros(n, dtype=np.float32)
    done[-1] = 1.0

    actions_rep = np.tile(np.array([ACTION_HOLD, ACTION_LONG, ACTION_SHORT], dtype=np.int64), n)
    row_rep = np.repeat(np.arange(n, dtype=np.int64), 3)
    rewards_rep = rewards[row_rep, actions_rep].astype(np.float32)
    next_rep = next_idx[row_rep]
    done_rep = done[row_rep]

    reward_std = float(np.std(rewards_rep)) if float(np.std(rewards_rep)) > 1e-8 else 1.0
    r_norm = (rewards_rep / reward_std).astype(np.float32)

    model = DuelingQNetwork(cfg).to(device)
    target = DuelingQNetwork(cfg).to(device)
    target.load_state_dict(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)

    m = len(actions_rep)
    priorities = np.abs(r_norm).astype(np.float32) + 1e-6
    losses: list[float] = []
    td_means: list[float] = []
    for step in range(1, int(steps) + 1):
        prob = priorities ** 0.65
        prob = prob / np.maximum(prob.sum(), 1e-12)
        idx = np.random.choice(m, size=min(int(batch_size), m), replace=True, p=prob)
        xb = torch.from_numpy(x[row_rep[idx]]).to(device)
        nb = torch.from_numpy(x[next_rep[idx]]).to(device)
        ab = torch.from_numpy(actions_rep[idx]).to(device)
        rb = torch.from_numpy(r_norm[idx]).to(device)
        db = torch.from_numpy(done_rep[idx]).to(device)

        q = model(xb)
        qa = q.gather(1, ab.view(-1, 1)).squeeze(1)
        if float(gamma) > 0.0:
            with torch.no_grad():
                next_action = torch.argmax(model(nb), dim=1, keepdim=True)
                next_q = target(nb).gather(1, next_action).squeeze(1)
                td_target = rb + float(gamma) * (1.0 - db) * next_q
        else:
            td_target = rb
        td = td_target - qa
        loss = F.smooth_l1_loss(qa, td_target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        priorities[idx] = 0.90 * priorities[idx] + 0.10 * (np.abs(td.detach().cpu().numpy()) + 1e-6)
        if step % 100 == 0:
            target.load_state_dict(model.state_dict())
        if step % 50 == 0 or step == 1:
            losses.append(float(loss.detach().cpu()))
            td_means.append(float(torch.mean(torch.abs(td)).detach().cpu()))
        if int(log_every) > 0 and (step == 1 or step % int(log_every) == 0 or step == int(steps)):
            print(json.dumps({"stage": "dqn_train_progress", "step": int(step), "steps": int(steps),
                               "loss": float(loss.detach().cpu())}, ensure_ascii=False), flush=True)
    target.load_state_dict(model.state_dict())
    meta = {"reward_std": reward_std, "loss_tail": losses[-5:], "td_abs_tail": td_means[-5:],
            "steps": int(steps), "batch_size": int(batch_size), "gamma": float(gamma)}
    return model, meta


def _policy_actions(model: DuelingQNetwork, x: np.ndarray, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            outs.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(outs).astype(np.int64)


def _backtest(df: pd.DataFrame, actions: np.ndarray, *, fee: float, slip: float) -> dict[str, Any]:
    close = _close(df)
    open_arr = _open(df)
    n = len(df)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    action_counts = {"hold": 0, "long": 0, "short": 0}
    exposure = 1.0

    def mark(i: int) -> float:
        if pos == 0:
            return cash
        px = float(close[int(np.clip(i, 0, n - 1))])
        raw = (px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px) / max(entry_price, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, side: int) -> None:
        nonlocal pos, entry_price, entry_equity, cash, long_entries, short_entries
        fill_i = min(i + 1, n - 1)
        pos = int(side)
        entry_price = _fill_price(open_arr, fill_i, pos, slip, entry=True)
        entry_equity = cash
        cash -= cash * fee * exposure
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)

    def exit_position(i: int) -> None:
        nonlocal pos, cash, trades, wins
        fill_i = min(i + 1, n - 1)
        exit_px = _fill_price(open_arr, fill_i, pos, slip, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * fee * exposure
        trades += 1
        wins += int(cash > entry_equity)
        pos = 0

    for i in range(0, n - 2):
        eq = mark(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        desired = {ACTION_HOLD: 0, ACTION_LONG: 1, ACTION_SHORT: -1}[int(actions[i])]
        if desired > 0:
            action_counts["long"] += 1
        elif desired < 0:
            action_counts["short"] += 1
        else:
            action_counts["hold"] += 1
        if pos != 0 and desired != pos:
            exit_position(i)
        if pos == 0 and desired != 0:
            enter(i, desired)

    if pos != 0:
        exit_position(n - 2)
        eq = cash
    else:
        eq = mark(n - 1)
    peak = max(peak, eq)
    mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
    return {
        "pnl_pct": float((cash - 1.0) * 100.0),
        "mdd_pct": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "action_counts": action_counts,
    }


def _metrics_costs(df: pd.DataFrame, actions: np.ndarray) -> dict[str, Any]:
    return {f"cost{m}": _backtest(df, actions, fee=FEE * m, slip=SLIP * m) for m in (1, 2, 3)}


def _score(metrics: dict[str, Any]) -> float:
    c1 = metrics["cost1"]
    return float(c1["pnl_pct"]) - 0.5 * abs(float(c1["mdd_pct"]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260722)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = _load_data()
    feature_cols = _feature_cols(df)

    train_df = df[df["timestamp"] < TRAIN_END].reset_index(drop=True)
    val_df = df[(df["timestamp"] >= VAL_START) & (df["timestamp"] < VAL_END)].reset_index(drop=True)
    oos_df = df[(df["timestamp"] >= OOS_START) & (df["timestamp"] < OOS_END)].reset_index(drop=True)
    fresh_df = df[df["timestamp"] >= FRESH_START].reset_index(drop=True)

    print(json.dumps({
        "stage": "splits",
        "train": [str(train_df["timestamp"].iloc[0]), str(train_df["timestamp"].iloc[-1]), len(train_df)],
        "val": [str(val_df["timestamp"].iloc[0]), str(val_df["timestamp"].iloc[-1]), len(val_df)],
        "oos": [str(oos_df["timestamp"].iloc[0]), str(oos_df["timestamp"].iloc[-1]), len(oos_df)],
        "fresh": [str(fresh_df["timestamp"].iloc[0]), str(fresh_df["timestamp"].iloc[-1]), len(fresh_df)],
        "feature_count": len(feature_cols),
    }, ensure_ascii=False), flush=True)

    train_close = _close(train_df)
    train_x_raw = train_df[feature_cols].to_numpy(dtype=np.float32)
    med, mean, std = _fit_norm(train_x_raw)
    train_x = _prep_matrix(train_x_raw, med, mean, std)

    train_rewards = _build_reward_table(train_close, HORIZON_BARS, ROUND_TRIP_COST)
    valid_mask = np.isfinite(train_rewards).all(axis=1)
    train_x_use = train_x[valid_mask]
    train_rewards_use = train_rewards[valid_mask]

    def normed(frame: pd.DataFrame) -> np.ndarray:
        raw = frame[feature_cols].to_numpy(dtype=np.float32)
        return _prep_matrix(raw, med, mean, std)

    val_x = normed(val_df)
    oos_x = normed(oos_df)
    fresh_x = normed(fresh_df)

    grid = [
        {"hidden_dim": 128, "lr": 3e-4, "gamma": 0.0},
        {"hidden_dim": 128, "lr": 7e-4, "gamma": 0.1},
        {"hidden_dim": 256, "lr": 3e-4, "gamma": 0.0},
        {"hidden_dim": 256, "lr": 7e-4, "gamma": 0.0},
        {"hidden_dim": 256, "lr": 7e-4, "gamma": 0.1},
        {"hidden_dim": 384, "lr": 5e-4, "gamma": 0.0},
    ]
    grid_rows = []
    trained = {}
    for gi, gcfg in enumerate(grid):
        cfg = DuelingDQNConfig(input_dim=len(feature_cols), hidden_dim=int(gcfg["hidden_dim"]),
                                action_dim=3, dropout=0.05, temperature=0.18)
        model, train_meta = _train_dqn(
            train_x_use, train_rewards_use, cfg=cfg, steps=int(args.steps), batch_size=int(args.batch_size),
            gamma=float(gcfg["gamma"]), lr=float(gcfg["lr"]), seed=int(args.seed) + gi, device=device,
            log_every=int(args.log_every),
        )
        val_actions = _policy_actions(model, val_x, device)
        val_metrics = _metrics_costs(val_df, val_actions)
        score = _score(val_metrics)
        name = f"h{gcfg['hidden_dim']}_lr{gcfg['lr']:.0e}_g{gcfg['gamma']}"
        grid_rows.append({"candidate": name, **gcfg, "score": score,
                           "val_cost1_pnl": val_metrics["cost1"]["pnl_pct"],
                           "val_cost1_mdd": val_metrics["cost1"]["mdd_pct"],
                           "val_cost1_trades": val_metrics["cost1"]["trades"],
                           "val_cost3_pnl": val_metrics["cost3"]["pnl_pct"]})
        trained[name] = (model, cfg, gcfg, val_metrics, train_meta)
        print(json.dumps({"stage": "grid_result", "name": name, "score": score,
                           "val_cost1": val_metrics["cost1"]}, ensure_ascii=False), flush=True)

    best_name = max(grid_rows, key=lambda r: r["score"])["candidate"]
    best_model, best_cfg, best_gcfg, best_val_metrics, best_train_meta = trained[best_name]

    oos_actions = _policy_actions(best_model, oos_x, device)
    oos_metrics = _metrics_costs(oos_df, oos_actions)
    fresh_actions = _policy_actions(best_model, fresh_x, device)
    fresh_metrics = _metrics_costs(fresh_df, fresh_actions)

    report = {
        "model_id": MODEL_ID,
        "design": (
            "Standalone Dueling DQN (HOLD/LONG/SHORT) trained directly on SOL raw 5m technical "
            "features, no Alpha1-4 HGB-parent chain, no NeuralForecast, no Regime3/CryptoMamba "
            "columns. Reward = fixed-horizon (12 bars/1h) forward realized return minus round-trip "
            "cost1 (fee=0.0004, slip=0.00015 per side), single-step contextual-bandit-style "
            "regression target (gamma swept 0.0 vs small bootstrap in VAL grid). PER-like "
            "|TD-error|-prioritized replay, network reused unmodified from "
            "ensemble/dueling_dqn_parent.py."
        ),
        "leakage_boundary_handling": (
            "No Regime3/CryptoMamba columns used at all (verified via feature-name scan raising if "
            "any regime3/cryptomamba/hmm-named column were present) -> the 2024 HMM-fit-on-2024 "
            "leakage boundary noted in the task brief does not apply; 2024 raw price/technical "
            "feature data used safely in training."
        ),
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "reward": {"horizon_bars": HORIZON_BARS, "fee": FEE, "slip": SLIP, "round_trip_cost": ROUND_TRIP_COST},
        "split": {
            "train": [str(train_df["timestamp"].iloc[0]), str(train_df["timestamp"].iloc[-1]), int(len(train_df))],
            "val": [str(val_df["timestamp"].iloc[0]), str(val_df["timestamp"].iloc[-1]), int(len(val_df))],
            "oos": [str(oos_df["timestamp"].iloc[0]), str(oos_df["timestamp"].iloc[-1]), int(len(oos_df))],
            "fresh": [str(fresh_df["timestamp"].iloc[0]), str(fresh_df["timestamp"].iloc[-1]), int(len(fresh_df))],
        },
        "grid": grid_rows,
        "best_config": {"name": best_name, **best_gcfg},
        "best_train_meta": best_train_meta,
        "val_metrics": best_val_metrics,
        "oos_metrics": oos_metrics,
        "fresh_metrics": fresh_metrics,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "device": str(device),
    }
    (args.out_dir / "dueling_dqn_standalone_sol_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    pd.DataFrame(grid_rows).sort_values("score", ascending=False).to_csv(
        args.out_dir / "dueling_dqn_standalone_sol_grid.csv", index=False
    )

    action_model = make_action_model(
        best_model, config=best_cfg, medians=med, mean=mean, std=std, feature_cols=feature_cols
    )
    import joblib
    joblib.dump(
        {"model_id": MODEL_ID, "action_model": action_model, "config": best_gcfg, "feature_cols": feature_cols},
        args.out_dir / "dueling_dqn_standalone_sol_parent.pkl",
    )

    print(json.dumps({
        "stage": "final",
        "best_config": report["best_config"],
        "val_cost1": best_val_metrics["cost1"],
        "oos_cost1": oos_metrics["cost1"],
        "oos_cost3": oos_metrics["cost3"],
        "fresh_cost1": fresh_metrics["cost1"],
        "fresh_cost3": fresh_metrics["cost3"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
