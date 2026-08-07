#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_eval_state_option_moe_2026 import (  # noqa: E402
    DEFAULT_EVAL_CSV,
    DEFAULT_TRAIN_CSV,
    FEATURE_COLS,
    OPTION_INPUT_COLS,
    OptionSpec,
    _audit_decisions,
    _candidate_matrix,
    _compact,
    _day_codes,
    _feature_frame,
    _fill_price,
    _fit_critics,
    _fit_state_tokenizer,
    _json_default,
    _option_labels,
    _predict_option_cube,
    _prices,
    _range,
    _read,
    _sample_rows,
    _sha256,
    _split_train_validation,
    _write_csv,
)

MODEL_ID = "state_option_moe_2026_v2"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/state_option_moe_2026_v2"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/state_option_moe_2026_v2.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/state_option_moe_2026_v2_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/state_option_moe_2026_v2_ledger.csv"
DEFAULT_DOC = ROOT / "docs/experiments/state_option_moe_2026_v2.md"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/state_option_moe_2026_v2_contract.md"

CLEAN_BASE_REFERENCE = {
    "pnl": 177.3298088749005,
    "mdd": -17.759665,
    "trades": 363,
    "trades_per_day": 6.1875,
    "avg_leverage": 1.0,
    "cost_2x_pnl": 92.254878,
    "cost_3x_pnl": -7.969395,
}

V1_REFERENCE = {
    "pnl": 362.6561012937483,
    "mdd": -10.073050921470772,
    "trades_per_day": 5.254237288135593,
    "avg_leverage": 2.1132258064516125,
    "cost_3x_pnl": -26.855081188282725,
}

LEAK_PRONE_FEATURES = {
    "evt_candidate_side",
    "evt_candidate_label",
    "evt_side_margin",
}


@dataclass(frozen=True)
class SelectorConfigV2:
    name: str
    lambda_cvar: float
    lambda_cost: float
    lambda_mae: float
    lambda_turnover: float
    upside_weight: float
    prob_large_loss_block: float
    min_utility: float
    max_daily_trades: int
    loss_cooldown_bars: int
    hard_stop: float
    trail_trigger: float
    trail_giveback: float
    daily_loss_limit: float
    daily_dd_limit: float
    global_dd_cut: float
    global_dd_mult: float
    notional_cap: float
    notional_scale: float


def _days(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or df.empty:
        return max(float(len(df)) / 288.0, 1.0)
    start = pd.Timestamp(df["timestamp"].iloc[0]).normalize()
    end = pd.Timestamp(df["timestamp"].iloc[-1]).normalize()
    return max(float((end - start).days + 1), 1.0)


def _catalog_v2() -> list[OptionSpec]:
    rows: list[OptionSpec] = []
    for side_name, side in (("long", 1), ("short", -1)):
        for notional in (1.0, 1.5, 2.0, 2.4, 2.8, 3.2, 3.6, 4.2):
            for hold in (6, 12, 18, 24, 36, 48):
                rows.append(OptionSpec(f"{side_name}_n{notional:.1f}_h{hold}", side, float(notional), int(hold)))
    return rows


def _selector_grid_v2() -> list[SelectorConfigV2]:
    risk_profiles = [
        {"hard_stop": -0.025, "trail_trigger": 0.016, "trail_giveback": 0.008, "daily_loss_limit": 0.022, "daily_dd_limit": 0.024, "global_dd_cut": 0.10, "global_dd_mult": 0.45},
        {"hard_stop": -0.030, "trail_trigger": 0.020, "trail_giveback": 0.010, "daily_loss_limit": 0.026, "daily_dd_limit": 0.028, "global_dd_cut": 0.12, "global_dd_mult": 0.55},
        {"hard_stop": -0.035, "trail_trigger": 0.025, "trail_giveback": 0.014, "daily_loss_limit": 0.030, "daily_dd_limit": 0.032, "global_dd_cut": 0.14, "global_dd_mult": 0.65},
    ]
    rows: list[SelectorConfigV2] = []
    for cvar in (0.0, 0.25, 0.50):
        for cost in (0.0, 0.15):
            for mae in (0.0, 0.20):
                for turnover in (0.0, 0.10):
                    for upside in (0.00, 0.15, 0.30):
                        for prob in (0.80, 1.01):
                            for min_u in (-0.004, 0.000, 0.001):
                                for max_trades in (12, 18):
                                    for cap, scale in ((3.2, 1.00), (3.6, 1.00), (4.2, 1.08)):
                                        for rp_i, rp in enumerate(risk_profiles):
                                            name = (
                                                f"v2_cv{cvar:.2f}_co{cost:.2f}_ma{mae:.2f}_to{turnover:.2f}_"
                                                f"up{upside:.2f}_p{prob:.2f}_u{min_u:.3f}_mt{max_trades}_"
                                                f"cap{cap:.1f}_sc{scale:.2f}_rp{rp_i}"
                                            )
                                            rows.append(
                                                SelectorConfigV2(
                                                    name=name,
                                                    lambda_cvar=cvar,
                                                    lambda_cost=cost,
                                                    lambda_mae=mae,
                                                    lambda_turnover=turnover,
                                                    upside_weight=upside,
                                                    prob_large_loss_block=prob,
                                                    min_utility=min_u,
                                                    max_daily_trades=max_trades,
                                                    loss_cooldown_bars=18,
                                                    notional_cap=cap,
                                                    notional_scale=scale,
                                                    **rp,
                                                )
                                            )
    return rows


def _select_options_v2(pred: dict[str, np.ndarray], options: list[OptionSpec], cfg: SelectorConfigV2, *, fee: float) -> dict[str, np.ndarray]:
    q05 = pred["q05"].astype(np.float64)
    q50 = pred["q50"].astype(np.float64)
    q95 = pred["q95"].astype(np.float64)
    c3 = pred["cost3"].astype(np.float64)
    mae = pred["mae"].astype(np.float64)
    prob = pred["prob_large_loss"].astype(np.float64)
    notionals = np.asarray([o.notional for o in options], dtype=np.float64)[None, :]
    turnover = 2.0 * float(fee) * notionals
    upside = np.maximum(q95 - q50, 0.0)
    utility = (
        q50
        + float(cfg.upside_weight) * upside
        - float(cfg.lambda_cvar) * np.maximum(-q05, 0.0)
        - float(cfg.lambda_cost) * np.maximum(q50 - c3, 0.0)
        - float(cfg.lambda_mae) * mae
        - float(cfg.lambda_turnover) * turnover
    )
    utility = np.where(prob > float(cfg.prob_large_loss_block), -1e9, utility)
    best_idx = np.argmax(utility, axis=1).astype(np.int16)
    best_u = utility[np.arange(len(best_idx)), best_idx]
    side = np.asarray([options[int(j)].side for j in best_idx], dtype=np.int8)
    raw_notional = np.asarray([options[int(j)].notional for j in best_idx], dtype=np.float64)
    notional = np.minimum(raw_notional * float(cfg.notional_scale), float(cfg.notional_cap))
    hold = np.asarray([options[int(j)].hold_bars for j in best_idx], dtype=np.int16)
    option_id = np.asarray([options[int(j)].option_id for j in best_idx], dtype=object)
    cash = best_u < float(cfg.min_utility)
    side[cash] = 0
    notional[cash] = 0.0
    hold[cash] = 0
    option_id[cash] = "CASH"
    return {
        "side": side,
        "notional": notional,
        "hold": hold,
        "option_idx": best_idx,
        "option_id": option_id,
        "utility": best_u.astype(np.float64),
    }


def backtest_selected_options_v2(
    df: pd.DataFrame,
    selected: dict[str, np.ndarray],
    cfg: SelectorConfigV2,
    *,
    fee: float,
    slip: float,
    emit_ledger: bool = False,
) -> dict[str, Any]:
    close, fill = _prices(df)
    day_codes = _day_codes(df)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_idx = 0
    hold_bars = 0
    notional = 0.0
    leverage = 1.0
    option_id = ""
    utility = 0.0
    peak_unreal = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    blocks: dict[str, int] = {}
    ledger: list[dict[str, Any]] = []
    loss_cooldown_left = 0
    day_key: int | None = None
    daily_start_cash = 1.0
    daily_peak_eq = 1.0
    daily_trades = 0

    def block(reason: str) -> None:
        blocks[reason] = blocks.get(reason, 0) + 1

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        if pos > 0:
            raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12)
        else:
            raw = (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    def close_position(i: int, reason: str) -> None:
        nonlocal cash, pos, entry_price, notional, leverage, hold_bars, trades, wins
        nonlocal loss_cooldown_left, daily_trades, peak_unreal
        exit_idx = min(i + 1, len(df) - 1)
        exit_price = _fill_price(fill, exit_idx, pos, entry=False, slip=slip)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        pnl_frac = raw * notional - 2.0 * float(fee) * notional
        cash = before * max(0.0, 1.0 + pnl_frac)
        trades += 1
        daily_trades += 1
        is_win = pnl_frac > 0.0
        wins += int(is_win)
        if not is_win:
            loss_cooldown_left = max(loss_cooldown_left, int(cfg.loss_cooldown_bars))
        exits[reason] = exits.get(reason, 0) + 1
        if emit_ledger:
            ledger.append({
                "entry_idx": int(entry_idx),
                "exit_idx": int(i),
                "side": "LONG" if pos > 0 else "SHORT",
                "option_id": str(option_id),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "notional_exposure": float(notional),
                "leverage": float(leverage),
                "position_fraction": float(min(notional / max(leverage, 1e-12), 1.0)),
                "hold_bars": int(i - entry_idx),
                "pnl_frac": float(pnl_frac),
                "pnl_pct": float(pnl_frac * 100.0),
                "equity_before": float(before),
                "equity_after": float(cash),
                "exit_reason": str(reason),
                "utility": float(utility),
            })
        pos = 0
        entry_price = 0.0
        notional = 0.0
        leverage = 1.0
        hold_bars = 0
        peak_unreal = 0.0

    for i in range(0, len(df) - 2):
        key = int(day_codes[i])
        eq, unreal = mark(i)
        if key != day_key:
            day_key = key
            daily_start_cash = max(eq, 1e-12)
            daily_peak_eq = max(eq, 1e-12)
            daily_trades = 0
        peak = max(peak, eq)
        daily_peak_eq = max(daily_peak_eq, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        account_dd = max(0.0, 1.0 - eq / max(peak, 1e-12))
        daily_dd = max(0.0, 1.0 - eq / max(daily_peak_eq, 1e-12))
        daily_realized = cash / max(daily_start_cash, 1e-12) - 1.0

        if pos != 0:
            peak_unreal = max(peak_unreal, unreal)
            age = i - entry_idx
            if hold_bars > 0 and age >= hold_bars:
                close_position(i, "option_time_exit")
                continue
            if unreal <= float(cfg.hard_stop):
                close_position(i, "hard_stop")
                continue
            if peak_unreal >= float(cfg.trail_trigger) and unreal <= peak_unreal - float(cfg.trail_giveback):
                close_position(i, "trailing_giveback")
                continue
            continue

        if loss_cooldown_left > 0:
            loss_cooldown_left -= 1
            block("loss_cooldown")
            continue
        if daily_trades >= int(cfg.max_daily_trades):
            block("daily_trade_budget")
            continue
        if daily_realized <= -abs(float(cfg.daily_loss_limit)):
            block("daily_loss_lock")
            continue
        if daily_dd >= abs(float(cfg.daily_dd_limit)):
            block("daily_dd_lock")
            continue

        side = int(selected["side"][i])
        if side == 0:
            block("cash")
            continue
        n = float(selected["notional"][i])
        if account_dd >= float(cfg.global_dd_cut):
            n *= float(cfg.global_dd_mult)
        n = float(np.clip(n, 0.0, float(cfg.notional_cap)))
        if n <= 1e-12:
            block("zero_notional")
            continue
        pos = side
        entry_idx = i
        entry_price = _fill_price(fill, min(i + 1, len(df) - 1), pos, entry=True, slip=slip)
        notional = n
        leverage = max(1.0, min(float(n), float(cfg.notional_cap)))
        hold_bars = int(selected["hold"][i])
        option_id = str(selected["option_id"][i])
        utility = float(selected["utility"][i])
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage

    if pos != 0:
        close_position(len(df) - 2, "end_of_data")
    final_eq = cash
    return {
        "pnl": float((final_eq - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "trades_per_day": float(trades / _days(df)),
        "wr": float(wins / max(trades, 1)),
        "avg_notional": float(notional_sum / max(long_entries + short_entries, 1)),
        "avg_leverage": float(leverage_sum / max(long_entries + short_entries, 1)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "entry_blocks": blocks,
        "exits": exits,
        "ledger": ledger if emit_ledger else [],
    }


def _score_row(val_1x: dict[str, Any], val_3x: dict[str, Any], cfg: SelectorConfigV2) -> float:
    pnl = float(val_1x["pnl"])
    mdd = float(val_1x["mdd"])
    tpd = float(val_1x["trades_per_day"])
    cost3 = float(val_3x["pnl"])
    lev = float(val_1x["avg_leverage"])
    score = pnl
    score += 4.0 * mdd
    score += 10.0 * min(tpd, 8.0)
    score += 0.05 * cost3
    score -= 35.0 * max(0.0, 5.4 - tpd)
    score -= 25.0 * max(0.0, lev - 3.2)
    if pnl < CLEAN_BASE_REFERENCE["pnl"]:
        score -= 120.0
    if mdd < -22.0:
        score -= 140.0
    if cost3 < -65.0:
        score -= 60.0
    if cfg.notional_cap > 3.6 and mdd < -14.0:
        score -= 45.0
    return float(score)


def _experiment_doc(report: dict[str, Any]) -> str:
    cand = report["oos"]["candidate"]
    c2 = report["oos"]["cost_2x"]
    c3 = report["oos"]["cost_3x"]
    gate = report["promotion_gate"]
    return f"""# State Option MoE 2026 V2

Status: `{gate['decision']}`

## Summary

`{MODEL_ID}` expands the SOMoE option catalog, adds upside-aware utility, and validation-selects execution risk profiles without changing existing v1 artifacts.

## OOS Results

| Metric | Value |
|---|---:|
| PnL 1x | `{cand['pnl']:.6f}%` |
| MDD 1x | `{cand['mdd']:.6f}%` |
| Trades/day | `{cand['trades_per_day']:.6f}` |
| Avg leverage | `{cand['avg_leverage']:.6f}` |
| PnL 2x | `{c2['pnl']:.6f}%` |
| PnL 3x | `{c3['pnl']:.6f}%` |

## Selected Config

`{report['selected_config']['name']}`

## Gate

- Clean-base PnL gate: `{gate['clean_base_pnl_gate']}`
- Clean-base MDD gate: `{gate['clean_base_mdd_gate']}`
- V1 PnL lift gate: `{gate['v1_pnl_lift_gate']}`
- Trades/day gate: `{gate['trades_per_day_gate']}`
- Cost 3x not worse than V1: `{gate['cost_3x_not_worse_than_v1']}`
- Invariant audit: `{gate['invariant_audit_passed']}`

## Artifacts

- Report: `{report['artifacts']['report']}`
- Grid: `{report['artifacts']['grid']}`
- Ledger: `{report['artifacts']['ledger']}`
- Model dir: `{report['artifacts']['model_dir']}`
"""


def _contract_doc() -> str:
    return f"""# State Option MoE 2026 V2 Contract

Status: `experimental_challenger`

## Purpose

Lift alpha from the audited clean base and SOMoE v1 without reusing the rejected MuZero/AZ rank-1 path as the promotion baseline.

## Data

- Train: `{DEFAULT_TRAIN_CSV}`
- Validation split: rows before `2025-11-01` train; `2025-11-01` through `2025-12-31` validation.
- OOS: `{DEFAULT_EVAL_CSV}`
- Feature source: current SOMoE feature subset from `scripts/train_eval_state_option_moe_2026.py::FEATURE_COLS`.

## Layer IO

| Layer | Inputs | Outputs |
|---|---|---|
| State tokenizer | causal feature matrix | `state_token`, `state_distance` |
| Option catalog | side, notional, hold candidates | expanded LONG/SHORT option ids |
| Distributional critics | feature matrix + state + option params | q05/q50/q95, cost3, MAE, large-loss probability |
| Upside/risk selector | critic outputs + validation-selected config | side, notional, hold, utility |
| Execution risk profile | selected option stream | hard stop, trailing lock, daily loss/DD locks, global DD scaling |
| Accounting | fills with fee/slippage stress | PnL, MDD, trades/day, ledger |

## Promotion Reference

- Clean base PnL: `{CLEAN_BASE_REFERENCE['pnl']:.6f}%`
- Clean base MDD: `{CLEAN_BASE_REFERENCE['mdd']:.6f}%`
- SOMoE v1 PnL: `{V1_REFERENCE['pnl']:.6f}%`

Leak-prone event label columns are dropped by default: `{', '.join(sorted(LEAK_PRONE_FEATURES))}`.

V2 is allowed to be aggressive, but it must report cost 2x/3x and invariant audit separately.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_full = _read(Path(args.train_csv))
    eval_df = _read(Path(args.eval_csv))
    train_df, val_df = _split_train_validation(train_full, args.split_date)
    available_feature_cols = [c for c in FEATURE_COLS if c in train_full.columns]
    if bool(args.keep_leak_prone_features):
        feature_cols = available_feature_cols
        dropped_features: list[str] = []
    else:
        feature_cols = [c for c in available_feature_cols if c not in LEAK_PRONE_FEATURES]
        dropped_features = [c for c in available_feature_cols if c in LEAK_PRONE_FEATURES]
    train_x = _feature_frame(train_df, feature_cols)
    val_x = _feature_frame(val_df, feature_cols)
    eval_x = _feature_frame(eval_df, feature_cols)
    options = _catalog_v2()

    tokenizer, token_data, token_meta = _fit_state_tokenizer(
        train_x,
        val_x,
        eval_x,
        n_tokens=int(args.state_tokens),
        seed=int(args.seed),
    )
    train_labels = _option_labels(train_df, options, fee=float(args.fee), slip=float(args.slip))
    train_rows = _sample_rows(len(train_df), stride=int(args.train_row_stride), seed=int(args.seed), max_rows=int(args.max_base_train_rows))
    cand_x, cand_y, _rows, _opts = _candidate_matrix(
        train_x,
        token_data["train_token"],
        token_data["train_distance"],
        train_labels,
        options,
        train_rows,
    )
    if len(cand_x) > int(args.max_candidate_rows):
        rng = np.random.default_rng(int(args.seed))
        keep = np.sort(rng.choice(np.arange(len(cand_x)), size=int(args.max_candidate_rows), replace=False))
        cand_x = cand_x.iloc[keep].reset_index(drop=True)
        cand_y = {k: v[keep] for k, v in cand_y.items()}

    critics = _fit_critics(cand_x, cand_y, seed=int(args.seed), max_iter=int(args.max_iter))
    val_pred = _predict_option_cube(critics, val_x, token_data["val_token"], token_data["val_distance"], options)
    eval_pred = _predict_option_cube(critics, eval_x, token_data["eval_token"], token_data["eval_distance"], options)

    grid_rows: list[dict[str, Any]] = []
    selected_cfg: SelectorConfigV2 | None = None
    selected_score = -1e18
    selected_val: dict[str, Any] | None = None
    selector_grid = _selector_grid_v2()
    if int(args.max_grid_configs) > 0 and len(selector_grid) > int(args.max_grid_configs):
        rng = np.random.default_rng(int(args.seed))
        keep = np.sort(rng.choice(np.arange(len(selector_grid)), size=int(args.max_grid_configs), replace=False))
        selector_grid = [selector_grid[int(i)] for i in keep]
    for cfg in selector_grid:
        sel = _select_options_v2(val_pred, options, cfg, fee=float(args.fee))
        val_1x = backtest_selected_options_v2(val_df, sel, cfg, fee=float(args.fee), slip=float(args.slip))
        val_3x = backtest_selected_options_v2(val_df, sel, cfg, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
        score = _score_row(val_1x, val_3x, cfg)
        row = {
            **asdict(cfg),
            "val_pnl": val_1x["pnl"],
            "val_mdd": val_1x["mdd"],
            "val_trades": val_1x["trades"],
            "val_trades_per_day": val_1x["trades_per_day"],
            "val_avg_leverage": val_1x["avg_leverage"],
            "val_cost3_pnl": val_3x["pnl"],
            "val_cost3_mdd": val_3x["mdd"],
            "selection_score": score,
        }
        grid_rows.append(row)
        if score > selected_score:
            selected_score = score
            selected_cfg = cfg
            selected_val = {"candidate": _compact(val_1x), "cost_3x": _compact(val_3x), "score": float(score)}

    assert selected_cfg is not None
    eval_sel = _select_options_v2(eval_pred, options, selected_cfg, fee=float(args.fee))
    oos_1x = backtest_selected_options_v2(eval_df, eval_sel, selected_cfg, fee=float(args.fee), slip=float(args.slip), emit_ledger=True)
    oos_2x = backtest_selected_options_v2(eval_df, eval_sel, selected_cfg, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
    oos_3x = backtest_selected_options_v2(eval_df, eval_sel, selected_cfg, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tokenizer, model_dir / "state_encoder.pkl")
    joblib.dump(critics, model_dir / "option_critics.pkl")
    selector_payload = {
        "model_id": MODEL_ID,
        "selected_config": asdict(selected_cfg),
        "feature_cols": feature_cols,
        "option_input_cols": OPTION_INPUT_COLS,
        "options": [asdict(o) for o in options],
        "token_meta": token_meta,
    }
    joblib.dump(selector_payload, model_dir / "option_selector.pkl")

    _write_csv(Path(args.grid), grid_rows)
    _write_csv(Path(args.ledger), list(oos_1x.get("ledger", [])))
    oos_1x = {k: v for k, v in oos_1x.items() if k != "ledger"}
    audit = _audit_decisions(oos_1x)
    promotion = {
        "clean_base_pnl_gate": bool(float(oos_1x["pnl"]) >= CLEAN_BASE_REFERENCE["pnl"]),
        "clean_base_mdd_gate": bool(float(oos_1x["mdd"]) >= CLEAN_BASE_REFERENCE["mdd"]),
        "v1_pnl_lift_gate": bool(float(oos_1x["pnl"]) >= V1_REFERENCE["pnl"]),
        "v1_mdd_lift_gate": bool(float(oos_1x["mdd"]) >= V1_REFERENCE["mdd"]),
        "trades_per_day_gate": bool(float(oos_1x["trades_per_day"]) >= CLEAN_BASE_REFERENCE["trades_per_day"]),
        "avg_leverage_gate": bool(1.0 <= float(oos_1x["avg_leverage"]) <= 3.2),
        "cost_2x_survival": bool(float(oos_2x["pnl"]) > 0.0),
        "cost_3x_not_worse_than_v1": bool(float(oos_3x["pnl"]) >= V1_REFERENCE["cost_3x_pnl"]),
        "cost_3x_not_worse_than_clean_base": bool(float(oos_3x["pnl"]) >= CLEAN_BASE_REFERENCE["cost_3x_pnl"]),
        "invariant_audit_passed": bool(audit["passed"]),
    }
    promotion["alpha_iterate_gate"] = bool(
        promotion["clean_base_pnl_gate"]
        and promotion["clean_base_mdd_gate"]
        and promotion["v1_pnl_lift_gate"]
        and promotion["invariant_audit_passed"]
    )
    promotion["decision"] = "promote" if all(
        promotion[k]
        for k in (
            "clean_base_pnl_gate",
            "clean_base_mdd_gate",
            "v1_pnl_lift_gate",
            "trades_per_day_gate",
            "avg_leverage_gate",
            "cost_2x_survival",
            "cost_3x_not_worse_than_clean_base",
            "invariant_audit_passed",
        )
    ) else ("iterate" if promotion["alpha_iterate_gate"] else "reject")

    report = {
        "model_id": MODEL_ID,
        "contract": str(DEFAULT_CONTRACT),
        "data": {
            "train_csv": str(Path(args.train_csv)),
            "eval_csv": str(Path(args.eval_csv)),
            "train_range": _range(train_df),
            "validation_range": _range(val_df),
            "oos_range": _range(eval_df),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "oos_rows": int(len(eval_df)),
            "feature_cols": feature_cols,
            "dropped_leak_prone_feature_cols": dropped_features,
            "leak_prone_feature_policy": "kept_for_diagnostic_only" if bool(args.keep_leak_prone_features) else "dropped_by_default",
        },
        "training": {
            "state_tokens": int(args.state_tokens),
            "candidate_train_rows": int(len(cand_x)),
            "max_iter": int(args.max_iter),
            "seed": int(args.seed),
            "option_count": int(len(options)),
            "selector_grid_configs": int(len(selector_grid)),
        },
        "selected_config": asdict(selected_cfg),
        "validation": selected_val,
        "oos": {
            "candidate": _compact(oos_1x),
            "cost_2x": _compact(oos_2x),
            "cost_3x": _compact(oos_3x),
        },
        "clean_base_reference": CLEAN_BASE_REFERENCE,
        "v1_reference": V1_REFERENCE,
        "audit": audit,
        "promotion_gate": promotion,
        "artifacts": {
            "model_dir": str(model_dir),
            "state_encoder": str(model_dir / "state_encoder.pkl"),
            "option_critics": str(model_dir / "option_critics.pkl"),
            "option_selector": str(model_dir / "option_selector.pkl"),
            "report": str(Path(args.report)),
            "grid": str(Path(args.grid)),
            "ledger": str(Path(args.ledger)),
            "doc": str(Path(args.doc)),
            "contract": str(DEFAULT_CONTRACT),
        },
        "artifact_sha256": {
            "state_encoder": _sha256(model_dir / "state_encoder.pkl"),
            "option_critics": _sha256(model_dir / "option_critics.pkl"),
            "option_selector": _sha256(model_dir / "option_selector.pkl"),
        },
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    Path(args.doc).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc).write_text(_experiment_doc(report), encoding="utf-8")
    DEFAULT_CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONTRACT.write_text(_contract_doc(), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate State Option MoE 2026 V2.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--split-date", default="2025-11-01")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    p.add_argument("--state-tokens", type=int, default=96)
    p.add_argument("--train-row-stride", type=int, default=3)
    p.add_argument("--max-base-train-rows", type=int, default=26000)
    p.add_argument("--max-candidate-rows", type=int, default=650000)
    p.add_argument("--max-grid-configs", type=int, default=2800)
    p.add_argument("--max-iter", type=int, default=90)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-leak-prone-features", action="store_true")
    return p.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({
        "model_id": report["model_id"],
        "selected_config": report["selected_config"]["name"],
        "oos": report["oos"],
        "promotion_gate": report["promotion_gate"],
        "report": report["artifacts"]["report"],
    }, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
