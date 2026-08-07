#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_deep_exit_oracle_20260514 as deep_exit  # noqa: E402
from scripts import eval_alpha3_exit_front_run_layer_20260514 as front_run  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner  # noqa: E402


MODEL_ID = "alpha3_rescue_exit_governor_v2_20260515"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_rescue_exit_governor_v2_20260515_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_rescue_exit_governor_v2_20260515_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_rescue_exit_governor_v2_20260515_grid.csv"
CONTRACT_OUT = ROOT / "docs/model_contracts/alpha3_rescue_exit_governor_v2_20260515_contract.md"


@dataclass(frozen=True)
class RescueExitConfig:
    name: str
    min_hold: int
    sl_progress: float
    adverse_q_margin: float
    min_mfe: float
    giveback_frac: float
    time_frac: float
    exit_arm: str
    maker_fee_mult: float = 0.20


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _entry_cfg() -> alpha3.ImmediateLimitConfig:
    return alpha3.ImmediateLimitConfig(
        "alpha3_corrected_selected_touch0_skip_entry",
        "next_open",
        0.0,
        0.0,
        0.0,
        0.20,
        entry_miss="skip",
        exit_miss="market_fallback",
    )


def _exit_cfg(name: str) -> alpha3.ImmediateLimitConfig:
    if name == "exit4_pen0":
        return alpha3.ImmediateLimitConfig("exit4_pen0", "next_open", 0.0, 4.0, 0.0, 0.20, entry_miss="skip", exit_miss="market_fallback")
    if name == "exit2_pen0":
        return alpha3.ImmediateLimitConfig("exit2_pen0", "next_open", 0.0, 2.0, 0.0, 0.20, entry_miss="skip", exit_miss="market_fallback")
    return alpha3.ImmediateLimitConfig("exit0_pen0", "next_open", 0.0, 0.0, 0.0, 0.20, entry_miss="skip", exit_miss="market_fallback")


def _configs() -> list[RescueExitConfig]:
    rows: list[RescueExitConfig] = []
    for min_hold in (1, 2, 3, 5):
        for sl_progress in (0.35, 0.50, 0.65, 0.80):
            for adverse in (0.0, 0.001, 0.0025, 0.005):
                for time_frac in (0.55, 0.70, 0.85):
                    rows.append(
                        RescueExitConfig(
                            name=f"loss_rescue_h{min_hold}_sl{sl_progress:.2f}_q{adverse:.4f}_t{time_frac:.2f}_exit0",
                            min_hold=min_hold,
                            sl_progress=sl_progress,
                            adverse_q_margin=adverse,
                            min_mfe=99.0,
                            giveback_frac=99.0,
                            time_frac=time_frac,
                            exit_arm="exit0_pen0",
                        )
                    )
    for min_hold in (2, 3, 5):
        for min_mfe in (0.015, 0.025, 0.040):
            for giveback in (0.35, 0.50, 0.65):
                for adverse in (-0.002, 0.0, 0.002):
                    rows.append(
                        RescueExitConfig(
                            name=f"giveback_h{min_hold}_mfe{min_mfe:.3f}_gb{giveback:.2f}_q{adverse:.3f}_exit0",
                            min_hold=min_hold,
                            sl_progress=99.0,
                            adverse_q_margin=adverse,
                            min_mfe=min_mfe,
                            giveback_frac=giveback,
                            time_frac=99.0,
                            exit_arm="exit0_pen0",
                        )
                    )
    rows.append(RescueExitConfig("disabled_baseline", 999, 99.0, 99.0, 99.0, 99.0, 99.0, "exit0_pen0"))
    return rows


def _rescue_reason(
    cfg: RescueExitConfig,
    deep_q: np.ndarray,
    idx: int,
    *,
    pos: int,
    hold: int,
    unreal: float,
    mfe: float,
    effective_sl: float,
    max_hold: int,
) -> str:
    if hold < int(cfg.min_hold):
        return ""
    ql, qs = float(deep_q[int(idx), 0]), float(deep_q[int(idx), 1])
    q_same = ql if pos > 0 else qs
    q_opp = qs if pos > 0 else ql
    adverse = q_opp - q_same
    sl_abs = max(abs(float(effective_sl)), 1e-6)
    if unreal < 0.0 and abs(unreal) >= sl_abs * float(cfg.sl_progress) and adverse >= float(cfg.adverse_q_margin):
        return "rescue_loss_adverse_q"
    if mfe >= float(cfg.min_mfe) and (mfe - unreal) >= max(0.001, mfe * float(cfg.giveback_frac)) and adverse >= float(cfg.adverse_q_margin):
        return "rescue_profit_giveback"
    if max_hold > 0 and hold >= int(max_hold * float(cfg.time_frac)) and unreal <= 0.0 and adverse >= float(cfg.adverse_q_margin):
        return "rescue_time_decay"
    return ""


def backtest_rescue_exit(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    deep_q: np.ndarray,
    decisions: pd.DataFrame,
    overlay: v31.OverlayConfig,
    cfg: RescueExitConfig,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> dict[str, Any]:
    close = _close(df)
    fee_base = float(fee) * float(cost_mult)
    slip_base = float(slip) * float(cost_mult)
    entry_cfg = _entry_cfg()
    rescue_exit_cfg = _exit_cfg(cfg.exit_arm)
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    owner = ""
    entry_price = entry_equity = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cooldown = next_cooldown = deep_cooldown = 0
    add_done = False
    mfe = mae = 0.0
    entry_edge = 0.0
    entry_vol_anchor = 0.0
    trades = wins = long_entries = short_entries = deep_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    runner_actions: dict[str, int] = {}
    route_counts: dict[str, int] = {}

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip_base) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_base)) / max(entry_price, 1e-12)
        unrealized = raw * notional
        return cash * (1.0 + unrealized), unrealized

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        dd_abs = max(0.0, 1.0 - eq / max(peak, 1e-12))
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

        if pos != 0:
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold = i - entry_idx
            effective_tp, effective_sl = deep_exit._effective_deep_exits(owner, overlay, take_profit, stop_loss, entry_edge, entry_vol_anchor, hold, mfe)
            reason = _rescue_reason(cfg, deep_q, i, pos=pos, hold=hold, unreal=unreal, mfe=mfe, effective_sl=effective_sl, max_hold=max_hold)
            selected_exit_cfg = rescue_exit_cfg if reason else entry_cfg
            if not reason:
                if effective_tp > 0.0 and unreal >= effective_tp:
                    reason = f"{owner}_take_profit"
                elif effective_sl > 0.0 and unreal <= -abs(effective_sl):
                    reason = f"{owner}_stop_loss"
                elif max_hold > 0 and hold >= max_hold:
                    reason = f"{owner}_max_hold"

            if owner == "v21_2" and not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
                state = {
                    "parent_notional": parent_notional,
                    "notional": notional,
                    "bars_since_entry": hold,
                    "unrealized": unreal,
                    "mfe": mfe,
                    "mae": mae,
                    "drawdown_abs": dd_abs,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "max_hold": max_hold,
                }
                x_runner = _feature_frame(df, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x_runner)
                if p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40:
                    delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional))
                    filled, add_px, add_fee, _, route = alpha3._try_immediate_limit(df, i, pos, entry_cfg, entry=True, fee=fee_base, slip=slip_base)
                    if filled and delta > 0.0:
                        new_notional = notional + delta
                        entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                        before = cash
                        cash -= before * add_fee * delta
                        notional = new_notional
                        entry_vol_anchor = max(entry_vol_anchor, v31._vol_anchor(df.iloc[i]) * notional)
                        runner_actions["v21_add_on"] = runner_actions.get("v21_add_on", 0) + 1
                        route_counts[route] = route_counts.get(route, 0) + 1
                    else:
                        runner_actions["v21_add_on_limit_miss"] = runner_actions.get("v21_add_on_limit_miss", 0) + 1
                        route_counts[route] = route_counts.get(route, 0) + 1
                else:
                    runner_actions["v21_reject"] = runner_actions.get("v21_reject", 0) + 1
                add_done = True

            if reason:
                filled, exit_px, exit_fee, _, route = alpha3._try_immediate_limit(df, i, pos, selected_exit_cfg, entry=False, fee=fee_base, slip=slip_base)
                if not filled:
                    runner_actions["exit_limit_miss_hold"] = runner_actions.get("exit_limit_miss_hold", 0) + 1
                    continue
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * exit_fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[f"{owner}_{reason}" if reason.startswith("rescue_") else reason] = exits.get(f"{owner}_{reason}" if reason.startswith("rescue_") else reason, 0) + 1
                route_counts[route] = route_counts.get(route, 0) + 1
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(overlay.cooldown))
                add_done = False
                continue

        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if deep_cooldown > 0:
            deep_cooldown -= 1

        dec = decisions.iloc[i]
        if int(dec.action) != ACTION_CASH and int(dec.side) != 0:
            filled, px, entry_fee, _, route = alpha3._try_immediate_limit(df, i, int(dec.side), entry_cfg, entry=True, fee=fee_base, slip=slip_base)
            if not filled:
                runner_actions["parent_entry_limit_miss"] = runner_actions.get("parent_entry_limit_miss", 0) + 1
                route_counts[route] = route_counts.get(route, 0) + 1
                continue
            pos = int(dec.side)
            owner = "v21_2"
            entry_price = px
            entry_equity = cash
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional)
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            next_cooldown = int(dec.cooldown_bars)
            entry_edge = 0.0
            entry_vol_anchor = v31._vol_anchor(df.iloc[i]) * notional
            cash -= cash * entry_fee * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += float(dec.leverage)
            mfe = mae = 0.0
            add_done = False
            runner_actions["v21_entry"] = runner_actions.get("v21_entry", 0) + 1
            route_counts[route] = route_counts.get(route, 0) + 1
            continue

        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= overlay.edge_th and margin >= overlay.margin_th:
                filled, px, entry_fee, _, route = alpha3._try_immediate_limit(df, i, side, entry_cfg, entry=True, fee=fee_base, slip=slip_base)
                if not filled:
                    runner_actions["deep_entry_limit_miss"] = runner_actions.get("deep_entry_limit_miss", 0) + 1
                    route_counts[route] = route_counts.get(route, 0) + 1
                    continue
                pos = side
                owner = "deep_alpha"
                entry_price = px
                entry_equity = cash
                entry_idx = i
                parent_notional = notional = float(overlay.notional)
                take_profit = float(overlay.base_tp)
                stop_loss = float(overlay.base_sl)
                max_hold = int(overlay.base_hold)
                next_cooldown = int(overlay.cooldown)
                entry_edge = edge
                entry_vol_anchor = v31._vol_anchor(df.iloc[i]) * notional
                cash -= cash * entry_fee * notional
                long_entries += int(pos > 0)
                short_entries += int(pos < 0)
                deep_entries += 1
                notional_sum += notional
                leverage_sum += max(notional, 1.0)
                mfe = mae = 0.0
                add_done = True
                runner_actions["deep_entry"] = runner_actions.get("deep_entry", 0) + 1
                route_counts[route] = route_counts.get(route, 0) + 1

    if pos != 0:
        exit_px = _fill_price(df, len(df) - 1, pos, slip_base, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee_base * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
        route_counts["forced_end_market"] = route_counts.get("forced_end_market", 0) + 1

    n = max(long_entries + short_entries, 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)),
        "deep_entries": int(deep_entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n),
        "avg_leverage": float(leverage_sum / n),
        "exits": exits,
        "runner_actions": runner_actions,
        "route_counts": route_counts,
    }


def _metrics_rescue(df: pd.DataFrame, stack: dict[str, Any], q: np.ndarray, decisions: pd.DataFrame, cfg: RescueExitConfig) -> dict[str, Any]:
    return {
        f"cost{mult}": backtest_rescue_exit(
            df,
            stack["parent"],
            stack["jackpot_model"],
            stack["add_cfg"],
            q,
            decisions,
            stack["overlay"],
            cfg,
            fee=stack["fee"],
            slip=stack["slip"],
            cost_mult=float(mult),
        )
        for mult in (1, 2, 3)
    }


def _write_contract(cfg: RescueExitConfig) -> None:
    CONTRACT_OUT.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_OUT.write_text(
        f"""# Alpha3 Rescue Exit Governor v2 Contract

## Change From v1

v1 learned early-exit selector was rejected because it underperformed the corrected Alpha3 baseline and did not reduce SL/max-hold exits. v2 is rescue-only: it preserves the existing Alpha3 TP/SL/max-hold lifecycle and only inserts a reduce-only close when a live-available adverse state is detected.

## Selected Runtime

```json
{json.dumps(asdict(cfg), indent=2, ensure_ascii=False)}
```

## Safety

- Entry stack remains frozen Alpha3 corrected `touch0 skip-entry`.
- No entry, flip, add, or increase action is allowed.
- TP/SL/max-hold remain fallback rails.
- Selection uses 2025Q4 only; 2026 is fixed OOS.
""",
        encoding="utf-8",
    )


def main() -> int:
    print(f"[{MODEL_ID}] loading fixed Alpha3 stack", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = front_run._load_fixed_stack()
    train_all = _read(v31.DEFAULT_TRAIN)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    eval_df = _read(v31.DEFAULT_EVAL)

    print(f"[{MODEL_ID}] rebuilding Alpha3 decisions and frozen V27 q", flush=True)
    val_dec, val_q = front_run._decisions_and_q(val_df, stack)
    eval_dec, eval_q = front_run._decisions_and_q(eval_df, stack)

    entry_cfg = _entry_cfg()
    baseline_cfg = RescueExitConfig("disabled_baseline", 999, 99.0, 99.0, 99.0, 99.0, 99.0, "exit0_pen0")
    print(f"[{MODEL_ID}] selecting rescue-only exit governor on 2025Q4", flush=True)
    rows: list[dict[str, Any]] = []
    best: tuple[float, RescueExitConfig, dict[str, Any]] | None = None
    for cfg in _configs():
        metrics = _metrics_rescue(val_df, stack, val_q, val_dec, cfg)
        score = _score(metrics)
        rows.append(
            {
                **asdict(cfg),
                "selection_score": score,
                "val_cost1_pnl": metrics["cost1"]["pnl"],
                "val_cost1_mdd": metrics["cost1"]["mdd"],
                "val_cost1_trades": metrics["cost1"]["trades"],
                "val_cost2_pnl": metrics["cost2"]["pnl"],
                "val_cost3_pnl": metrics["cost3"]["pnl"],
                "val_cost1_exits": json.dumps(metrics["cost1"].get("exits", {}), sort_keys=True),
            }
        )
        if best is None or score > best[0]:
            best = (score, cfg, metrics)
            print(
                f"[{MODEL_ID}] new best {cfg.name} val c1={metrics['cost1']['pnl']:.2f} "
                f"mdd={metrics['cost1']['mdd']:.2f} c2={metrics['cost2']['pnl']:.2f} c3={metrics['cost3']['pnl']:.2f}",
                flush=True,
            )
    assert best is not None
    selected = best[1]
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)

    print(f"[{MODEL_ID}] fixed 2026 OOS", flush=True)
    taker = alpha2._metrics(eval_df, stack["parent"], stack["jackpot_model"], stack["add_cfg"], eval_q, eval_dec, l2._variants()[0], fee=stack["fee"], slip=stack["slip"])
    old_l2 = alpha2._metrics(eval_df, stack["parent"], stack["jackpot_model"], stack["add_cfg"], eval_q, eval_dec, stack["selected_l2_variant"], fee=stack["fee"], slip=stack["slip"])
    corrected_baseline = alpha3._metrics_signal_limit(
        eval_df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        eval_q,
        eval_dec,
        stack["overlay"],
        entry_cfg,
        fee=stack["fee"],
        slip=stack["slip"],
    )
    rescue_baseline_path = _metrics_rescue(eval_df, stack, eval_q, eval_dec, baseline_cfg)
    rescue = _metrics_rescue(eval_df, stack, eval_q, eval_dec, selected)
    experiments = [
        {"name": "alpha2_1_next_open_taker_control", "metrics": taker, "score": _score(taker)},
        {"name": "alpha2_1_old_l2_replay_fee20_control", "metrics": old_l2, "score": _score(old_l2)},
        {"name": "alpha3_corrected_touch0_skip_entry_baseline", "config": asdict(entry_cfg), "metrics": corrected_baseline, "score": _score(corrected_baseline)},
        {"name": "alpha3_rescue_path_disabled_control", "policy": asdict(baseline_cfg), "metrics": rescue_baseline_path, "score": _score(rescue_baseline_path)},
        {"name": f"alpha3_rescue_exit_governor::{selected.name}", "policy": asdict(selected), "metrics": rescue, "score": _score(rescue)},
    ]
    for exp in experiments:
        m = exp["metrics"]
        print(
            f"[{MODEL_ID}] {exp['name']} c1={m['cost1']['pnl']:.2f} mdd={m['cost1']['mdd']:.2f} "
            f"trades={m['cost1']['trades']} c2={m['cost2']['pnl']:.2f} c3={m['cost3']['pnl']:.2f}",
            flush=True,
        )

    _write_contract(selected)
    base_exits = corrected_baseline["cost1"].get("exits", {})
    rescue_exits = rescue["cost1"].get("exits", {})
    base_score = _score(corrected_baseline)
    rescue_score = _score(rescue)
    audit = {
        "model_id": MODEL_ID,
        "status": "promote_shadow_candidate" if rescue_score > base_score else "reject_do_not_promote",
        "selection_uses_2026": False,
        "selected_config": asdict(selected),
        "causality": [
            "Selection: 2025-10-01..2025-12-31 only.",
            "2026 is fixed OOS after config selection.",
            "Rescue state uses current open-position state and frozen V27 q only.",
        ],
        "exit_attribution_cost1": {
            "baseline": base_exits,
            "rescue": rescue_exits,
            "baseline_stop_loss_plus_max_hold": int(sum(v for k, v in base_exits.items() if "stop_loss" in k or "max_hold" in k)),
            "rescue_stop_loss_plus_max_hold": int(sum(v for k, v in rescue_exits.items() if "stop_loss" in k or "max_hold" in k)),
        },
        "blocking": [] if rescue_score > base_score else ["rescue_exit_underperforms_corrected_alpha3_baseline_on_2026_oos"],
        "warnings": [
            "5m OHLC high/low touch proxy is still not real queue simulation.",
            "Rescue close is full reduce-only close; no partial close/TWAP yet.",
        ],
    }
    report = {
        "model_id": MODEL_ID,
        "design": "v2 rescue-only exit layer. It keeps Alpha3 baseline lifecycle and only closes early on adverse loss-progress, profit-giveback, or time-decay states.",
        "selected_config": asdict(selected),
        "validation_best_score": float(best[0]),
        "validation_best_metrics": best[2],
        "experiments": experiments,
        "audit": audit,
        "artifacts": {
            "grid": str(GRID_OUT.relative_to(ROOT)),
            "audit": str(AUDIT_OUT.relative_to(ROOT)),
            "contract": str(CONTRACT_OUT.relative_to(ROOT)),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT), "contract": str(CONTRACT_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
