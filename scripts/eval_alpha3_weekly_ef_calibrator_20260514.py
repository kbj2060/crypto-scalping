#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3_exec  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner  # noqa: E402


MODEL_ID = "alpha3_weekly_ef_calibrator_20260514"
TEACHER_MODEL = ROOT / "data/ensemble/supervised/alpha1_l2_teacher_deep_parent_20260514/teacher_deep_parent_l2_replay.pt"
ALPHA2_AUDIT = ROOT / "data/ensemble/reports/alpha2_teacher_l2_runtime_sweep_20260514_audit.json"
ALPHA3_AUDIT = ROOT / "data/ensemble/reports/alpha2_1_signal_immediate_limit_20260514_audit.json"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_weekly_ef_calibrator_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_weekly_ef_calibrator_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_weekly_ef_calibrator_20260514_grid.csv"
SCHEDULE_OUT = ROOT / "data/ensemble/reports/alpha3_weekly_ef_calibrator_20260514_schedule.csv"


@dataclass(frozen=True)
class WeeklyCandidate:
    runtime: alpha2.Alpha2Runtime
    limit_cfg: alpha3_exec.ImmediateLimitConfig

    @property
    def name(self) -> str:
        return f"{self.runtime.name}::{self.limit_cfg.name}"


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 8:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.30 * c3["pnl"] - 0.35 * abs(c1["mdd"]))


def _base_runtime() -> alpha2.Alpha2Runtime:
    audit = json.loads(ALPHA2_AUDIT.read_text(encoding="utf-8"))
    rt = dict(audit.get("selected_runtime", {}) or {})
    return alpha2.Alpha2Runtime(
        name=str(rt.get("name", "noflip_c0.56_parent_scale1.10")),
        confidence=float(rt.get("confidence", 0.56)),
        parent_notional_scale=float(rt.get("parent_notional_scale", 1.10)),
        max_notional=float(rt.get("max_notional", 2.75)),
    )


def _base_limit_cfg() -> alpha3_exec.ImmediateLimitConfig:
    audit = json.loads(ALPHA3_AUDIT.read_text(encoding="utf-8"))
    cfg = dict(audit.get("selected_config", {}) or {})
    return alpha3_exec.ImmediateLimitConfig(
        name=str(cfg.get("name", "next_open_limit_offset2_entry_fallback_fee20")),
        anchor=str(cfg.get("anchor", "next_open")),
        entry_offset_bps=float(cfg.get("entry_offset_bps", 2.0)),
        exit_offset_bps=float(cfg.get("exit_offset_bps", 2.0)),
        penetration_bps=float(cfg.get("penetration_bps", 0.5)),
        maker_fee_mult=float(cfg.get("maker_fee_mult", 0.20)),
        entry_miss=str(cfg.get("entry_miss", "market_fallback")),
        exit_miss=str(cfg.get("exit_miss", "market_fallback")),
    )


def _runtime_candidates(base: alpha2.Alpha2Runtime) -> list[alpha2.Alpha2Runtime]:
    rows: list[alpha2.Alpha2Runtime] = []
    for conf in sorted({0.50, 0.56, 0.62, float(base.confidence)}):
        for scale in sorted({0.95, 1.10, 1.25, float(base.parent_notional_scale)}):
            rows.append(
                alpha2.Alpha2Runtime(
                    name=f"weekly_c{conf:.2f}_parent_scale{scale:.2f}",
                    confidence=float(conf),
                    parent_notional_scale=float(scale),
                    max_notional=float(base.max_notional),
                )
            )
    return rows


def _limit_candidates(base: alpha3_exec.ImmediateLimitConfig) -> list[alpha3_exec.ImmediateLimitConfig]:
    candidates = [
        base,
        alpha3_exec.ImmediateLimitConfig("weekly_next_open_touch0_fallback_fee20", "next_open", 0.0, 0.0, 0.0, 0.20, entry_miss="market_fallback", exit_miss="market_fallback"),
        alpha3_exec.ImmediateLimitConfig("weekly_next_open_offset1_fallback_fee20", "next_open", 1.0, 1.0, 0.5, 0.20, entry_miss="market_fallback", exit_miss="market_fallback"),
        alpha3_exec.ImmediateLimitConfig("weekly_next_open_offset2_fallback_fee35", "next_open", 2.0, 2.0, 0.5, 0.35, entry_miss="market_fallback", exit_miss="market_fallback"),
    ]
    dedup: dict[str, alpha3_exec.ImmediateLimitConfig] = {}
    for cfg in candidates:
        key = json.dumps(asdict(cfg), sort_keys=True)
        dedup[key] = cfg
    return list(dedup.values())


def _candidate_grid(base_rt: alpha2.Alpha2Runtime, base_cfg: alpha3_exec.ImmediateLimitConfig) -> list[WeeklyCandidate]:
    candidates = [WeeklyCandidate(rt, cfg) for rt in _runtime_candidates(base_rt) for cfg in _limit_candidates(base_cfg)]
    candidates.append(WeeklyCandidate(base_rt, base_cfg))
    dedup: dict[str, WeeklyCandidate] = {}
    for candidate in candidates:
        key = json.dumps({"runtime": asdict(candidate.runtime), "limit_cfg": asdict(candidate.limit_cfg)}, sort_keys=True)
        dedup[key] = candidate
    return list(dedup.values())


def _build_decisions_by_runtime(
    base_dec: pd.DataFrame,
    pred: dict[str, np.ndarray],
    buckets: tuple[float, ...],
    runtimes: list[alpha2.Alpha2Runtime],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for rt in runtimes:
        out[rt.name] = alpha2._decisions(base_dec, pred, buckets, rt)
    return out


def _slice_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    ts = pd.to_datetime(df["timestamp"])
    return ((ts >= start) & (ts < end)).to_numpy()


def _select_candidate(
    df: pd.DataFrame,
    parent: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    q: np.ndarray,
    decisions_by_runtime: dict[str, pd.DataFrame],
    overlay: v31.OverlayConfig,
    candidates: list[WeeklyCandidate],
    *,
    fee: float,
    slip: float,
    week_name: str,
    rows: list[dict[str, Any]],
) -> WeeklyCandidate:
    best = candidates[0]
    best_score = -1e18
    for candidate in candidates:
        dec = decisions_by_runtime[candidate.runtime.name]
        metrics = alpha3_exec._metrics_signal_limit(
            df,
            parent,
            jackpot_model,
            add_cfg,
            q,
            dec,
            overlay,
            candidate.limit_cfg,
            fee=fee,
            slip=slip,
        )
        score = _score(metrics["cost1"], metrics["cost2"], metrics["cost3"])
        rows.append(
            {
                "week": week_name,
                "candidate": candidate.name,
                "confidence": candidate.runtime.confidence,
                "parent_notional_scale": candidate.runtime.parent_notional_scale,
                "limit_cfg": candidate.limit_cfg.name,
                "selection_score": score,
                "sel_cost1_pnl": metrics["cost1"]["pnl"],
                "sel_cost1_mdd": metrics["cost1"]["mdd"],
                "sel_cost1_trades": metrics["cost1"]["trades"],
                "sel_cost2_pnl": metrics["cost2"]["pnl"],
                "sel_cost3_pnl": metrics["cost3"]["pnl"],
            }
        )
        if score > best_score:
            best_score = score
            best = candidate
    return best


def _weekly_bounds(eval_df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ts = pd.to_datetime(eval_df["timestamp"])
    start = ts.min().normalize()
    end = ts.max().normalize() + pd.Timedelta(days=1)
    bounds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=7), end)
        bounds.append((cur, nxt))
        cur = nxt
    return bounds


def _make_weekly_schedule(
    *,
    val: pd.DataFrame,
    val_q: np.ndarray,
    val_decisions_by_runtime: dict[str, pd.DataFrame],
    eval_df: pd.DataFrame,
    eval_q: np.ndarray,
    eval_decisions_by_runtime: dict[str, pd.DataFrame],
    parent: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    overlay: v31.OverlayConfig,
    candidates: list[WeeklyCandidate],
    base_candidate: WeeklyCandidate,
    fee: float,
    slip: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grid_rows: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    bounds = _weekly_bounds(eval_df)
    for idx, (week_start, week_end) in enumerate(bounds):
        if idx == 0:
            calibration_df = val
            calibration_q = val_q
            calibration_decisions = val_decisions_by_runtime
            calibration_source = "2025Q4_validation"
            selected = _select_candidate(
                calibration_df,
                parent,
                jackpot_model,
                add_cfg,
                calibration_q,
                calibration_decisions,
                overlay,
                candidates,
                fee=fee,
                slip=slip,
                week_name=f"{week_start.date()}",
                rows=grid_rows,
            )
        else:
            cal_start = week_start - pd.Timedelta(days=28)
            cal_end = week_start
            mask = _slice_window(eval_df, cal_start, cal_end)
            if int(mask.sum()) < 300:
                selected = base_candidate
                calibration_source = "fallback_base_not_enough_prior_rows"
            else:
                loc = np.flatnonzero(mask)
                calibration_df = eval_df.iloc[loc].reset_index(drop=True)
                calibration_q = eval_q[loc]
                calibration_decisions = {
                    name: frame.iloc[loc].reset_index(drop=True)
                    for name, frame in eval_decisions_by_runtime.items()
                }
                calibration_source = f"trailing_28d_{cal_start.date()}..{cal_end.date()}"
                selected = _select_candidate(
                    calibration_df,
                    parent,
                    jackpot_model,
                    add_cfg,
                    calibration_q,
                    calibration_decisions,
                    overlay,
                    candidates,
                    fee=fee,
                    slip=slip,
                    week_name=f"{week_start.date()}",
                    rows=grid_rows,
                )
        schedule.append(
            {
                "week_start": week_start,
                "week_end": week_end,
                "calibration_source": calibration_source,
                "candidate": selected.name,
                "runtime": selected.runtime,
                "limit_cfg": selected.limit_cfg,
            }
        )
        print(
            f"[{MODEL_ID}] week {week_start.date()} selected {selected.name} source={calibration_source}",
            flush=True,
        )
    return schedule, grid_rows


def _schedule_index(eval_df: pd.DataFrame, schedule: list[dict[str, Any]]) -> list[int]:
    ts = pd.to_datetime(eval_df["timestamp"])
    out = [0 for _ in range(len(eval_df))]
    for si, row in enumerate(schedule):
        mask = (ts >= row["week_start"]) & (ts < row["week_end"])
        for i in np.flatnonzero(mask.to_numpy()):
            out[int(i)] = int(si)
    return out


def backtest_weekly_dynamic(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    deep_q: np.ndarray,
    decisions_by_runtime: dict[str, pd.DataFrame],
    overlay: v31.OverlayConfig,
    schedule: list[dict[str, Any]],
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> dict[str, Any]:
    close = _close(df)
    fee_base = float(fee) * float(cost_mult)
    slip_base = float(slip) * float(cost_mult)
    schedule_idx = _schedule_index(df, schedule)
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
    actions: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    candidate_usage: dict[str, int] = {}

    def current_candidate(i: int) -> WeeklyCandidate:
        row = schedule[schedule_idx[int(np.clip(i, 0, len(schedule_idx) - 1))]]
        return WeeklyCandidate(row["runtime"], row["limit_cfg"])

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip_base) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_base)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        candidate = current_candidate(i)
        limit_cfg = candidate.limit_cfg
        candidate_usage[candidate.name] = candidate_usage.get(candidate.name, 0) + 1
        eq, unreal = mark(i)
        peak = max(peak, eq)
        dd_abs = max(0.0, 1.0 - eq / max(peak, 1e-12))
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

        if pos != 0:
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold = i - entry_idx
            reason = ""
            effective_tp = take_profit
            effective_sl = stop_loss
            if owner == "deep_alpha":
                if overlay.tp_util_mult > 0.0:
                    util_gain = 1.0 + overlay.tp_util_mult * max(entry_edge - overlay.edge_th, 0.0) / max(0.02, overlay.edge_th)
                    effective_tp = v31._clip(overlay.base_tp * util_gain, overlay.base_tp * 0.8, overlay.tp_cap)
                if overlay.sl_vol_mult > 0.0:
                    effective_sl = v31._clip(entry_vol_anchor * overlay.sl_vol_mult, overlay.base_sl * 0.6, overlay.sl_cap)
                if mfe > 0.0 and overlay.trail_gap_mult > 0.0:
                    trail_gap = entry_vol_anchor * overlay.trail_gap_mult
                    if overlay.hold_decay_start < 999 and hold >= overlay.hold_decay_start:
                        trail_gap = max(entry_vol_anchor * 0.35, trail_gap - overlay.hold_decay_rate * (hold - overlay.hold_decay_start) * entry_vol_anchor)
                    trail_stop = max(-effective_sl, mfe - trail_gap)
                    effective_sl = min(effective_sl, max(0.001, trail_stop))
            if effective_tp > 0.0 and unreal >= effective_tp:
                reason = f"{owner}_take_profit"
            elif effective_sl > 0.0 and unreal <= -abs(effective_sl):
                reason = f"{owner}_stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = f"{owner}_max_hold"

            if owner == "v21_2" and not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
                state = {"parent_notional": parent_notional, "notional": notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": dd_abs, "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
                x = _feature_frame(df, bundle, decisions_by_runtime[candidate.runtime.name], i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                if p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40:
                    delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional))
                    filled, add_px, add_fee, _, route = alpha3_exec._try_immediate_limit(df, i, pos, limit_cfg, entry=True, fee=fee_base, slip=slip_base)
                    if filled and delta > 0.0:
                        new_notional = notional + delta
                        entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                        before = cash
                        cash -= before * add_fee * delta
                        notional = new_notional
                        actions["v21_add_on"] = actions.get("v21_add_on", 0) + 1
                    else:
                        actions["v21_add_on_limit_miss"] = actions.get("v21_add_on_limit_miss", 0) + 1
                    route_counts[route] = route_counts.get(route, 0) + 1
                else:
                    actions["v21_reject"] = actions.get("v21_reject", 0) + 1
                add_done = True

            if reason:
                filled, exit_px, exit_fee, _, route = alpha3_exec._try_immediate_limit(df, i, pos, limit_cfg, entry=False, fee=fee_base, slip=slip_base)
                if not filled:
                    actions["exit_limit_miss_hold"] = actions.get("exit_limit_miss_hold", 0) + 1
                    continue
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * exit_fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
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

        dec = decisions_by_runtime[candidate.runtime.name].iloc[i]
        if int(dec.action) != ACTION_CASH and int(dec.side) != 0:
            filled, px, entry_fee, _, route = alpha3_exec._try_immediate_limit(df, i, int(dec.side), limit_cfg, entry=True, fee=fee_base, slip=slip_base)
            if not filled:
                actions["parent_entry_limit_miss"] = actions.get("parent_entry_limit_miss", 0) + 1
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
            cash -= cash * entry_fee * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += float(dec.leverage)
            mfe = mae = 0.0
            add_done = False
            actions["v21_entry"] = actions.get("v21_entry", 0) + 1
            route_counts[route] = route_counts.get(route, 0) + 1
            continue

        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= overlay.edge_th and margin >= overlay.margin_th:
                filled, px, entry_fee, _, route = alpha3_exec._try_immediate_limit(df, i, side, limit_cfg, entry=True, fee=fee_base, slip=slip_base)
                if not filled:
                    actions["deep_entry_limit_miss"] = actions.get("deep_entry_limit_miss", 0) + 1
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
                actions["deep_entry"] = actions.get("deep_entry", 0) + 1
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
        "runner_actions": actions,
        "route_counts": route_counts,
        "candidate_usage": candidate_usage,
    }


def _run_weekly_all(df, parent, jackpot_model, add_cfg, q, decisions_by_runtime, overlay, schedule, *, fee, slip) -> dict[str, Any]:
    return {
        f"cost{mult}": backtest_weekly_dynamic(
            df,
            parent,
            jackpot_model,
            add_cfg,
            q,
            decisions_by_runtime,
            overlay,
            schedule,
            fee=fee,
            slip=slip,
            cost_mult=float(mult),
        )
        for mult in (1, 2, 3)
    }


def main() -> int:
    print(f"[{MODEL_ID}] loading frozen Alpha3 stack", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    base_rt = _base_runtime()
    base_cfg = _base_limit_cfg()
    base_candidate = WeeklyCandidate(base_rt, base_cfg)
    candidates = _candidate_grid(base_rt, base_cfg)
    runtimes = list({c.runtime.name: c.runtime for c in [base_candidate, *candidates]}.values())

    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    teacher_payload = torch.load(TEACHER_MODEL, map_location="cpu", weights_only=False)
    teacher_model = alpha2._load_teacher_model(teacher_payload)
    feature_cols = list(teacher_payload["feature_cols"])
    norm = dict(dict(teacher_payload["train_meta"])["norm"])
    buckets = tuple(float(x) for x in teacher_payload["buckets"])
    fee = float(dict(parent["config"])["fee"])
    slip = float(dict(parent["config"])["slip"])
    selected_variant = next(v for v in l2._variants() if v.name == "alpha1_l2_conservative_fee20")
    overlay = selected_variant.overlay

    train_all = _read(v31.DEFAULT_TRAIN)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    eval_df = _read(v31.DEFAULT_EVAL)

    print(f"[{MODEL_ID}] predicting parent, teacher, and V27 utilities", flush=True)
    val_base_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_base_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    val_features = prepare_features(val, side_hint=0, close=_close(val), feature_cols=feature_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    val_pred = teacher._predict_deep(teacher_model, val_features, feature_cols, norm)
    eval_pred = teacher._predict_deep(teacher_model, eval_features, feature_cols, norm)
    val_decisions_by_runtime = _build_decisions_by_runtime(val_base_dec, val_pred, buckets, runtimes)
    eval_decisions_by_runtime = _build_decisions_by_runtime(eval_base_dec, eval_pred, buckets, runtimes)
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    print(f"[{MODEL_ID}] selecting weekly EF schedule without current-week leakage", flush=True)
    schedule, grid_rows = _make_weekly_schedule(
        val=val,
        val_q=val_q,
        val_decisions_by_runtime=val_decisions_by_runtime,
        eval_df=eval_df,
        eval_q=eval_q,
        eval_decisions_by_runtime=eval_decisions_by_runtime,
        parent=parent,
        jackpot_model=jackpot_model,
        add_cfg=add_cfg,
        overlay=overlay,
        candidates=candidates,
        base_candidate=base_candidate,
        fee=fee,
        slip=slip,
    )
    pd.DataFrame(grid_rows).sort_values(["week", "selection_score"], ascending=[True, False]).to_csv(GRID_OUT, index=False)
    schedule_rows = [
        {
            "week_start": str(row["week_start"]),
            "week_end": str(row["week_end"]),
            "calibration_source": row["calibration_source"],
            "candidate": row["candidate"],
            "confidence": row["runtime"].confidence,
            "parent_notional_scale": row["runtime"].parent_notional_scale,
            "limit_cfg": row["limit_cfg"].name,
            "anchor": row["limit_cfg"].anchor,
            "entry_offset_bps": row["limit_cfg"].entry_offset_bps,
            "exit_offset_bps": row["limit_cfg"].exit_offset_bps,
            "maker_fee_mult": row["limit_cfg"].maker_fee_mult,
            "entry_miss": row["limit_cfg"].entry_miss,
            "exit_miss": row["limit_cfg"].exit_miss,
        }
        for row in schedule
    ]
    pd.DataFrame(schedule_rows).to_csv(SCHEDULE_OUT, index=False)

    print(f"[{MODEL_ID}] fixed Alpha3 baseline OOS", flush=True)
    base_dec = eval_decisions_by_runtime[base_rt.name]
    baseline = alpha3_exec._metrics_signal_limit(eval_df, parent, jackpot_model, add_cfg, eval_q, base_dec, overlay, base_cfg, fee=fee, slip=slip)

    print(f"[{MODEL_ID}] weekly EF dynamic OOS", flush=True)
    weekly = _run_weekly_all(eval_df, parent, jackpot_model, add_cfg, eval_q, eval_decisions_by_runtime, overlay, schedule, fee=fee, slip=slip)
    print(
        f"[{MODEL_ID}] baseline c1={baseline['cost1']['pnl']:.2f} mdd={baseline['cost1']['mdd']:.2f} "
        f"c2={baseline['cost2']['pnl']:.2f} c3={baseline['cost3']['pnl']:.2f}",
        flush=True,
    )
    print(
        f"[{MODEL_ID}] weekly c1={weekly['cost1']['pnl']:.2f} mdd={weekly['cost1']['mdd']:.2f} "
        f"c2={weekly['cost2']['pnl']:.2f} c3={weekly['cost3']['pnl']:.2f}",
        flush=True,
    )

    warnings = [
        "weekly_calibrator_selects_only_from_prior_window_or_2025q4_for_first_week",
        "signal_limit_fill_uses_5m_high_low_touch_proxy_not_queue_fill",
        "real_l2_queue_and_partial_fill_require_forward_shadow_validation",
    ]
    if weekly["cost1"]["pnl"] <= baseline["cost1"]["pnl"]:
        warnings.append("weekly_ef_did_not_improve_cost1_pnl")
    if weekly["cost1"]["mdd"] < baseline["cost1"]["mdd"]:
        warnings.append("weekly_ef_worsened_cost1_mdd")
    audit = {
        "status": "pass",
        "verdict": "promote_candidate" if weekly["cost1"]["pnl"] > baseline["cost1"]["pnl"] and weekly["cost1"]["mdd"] >= baseline["cost1"]["mdd"] else "iterate",
        "blocking": [],
        "warnings": warnings,
        "selection_uses_current_week": False,
        "selection_uses_2026_current_or_future": False,
        "calibration_policy": "first_week_2025Q4_validation_then_trailing_28d_prior_only",
        "frozen_layers": {
            "hgb_parent": str(v31.DEFAULT_PARENT),
            "teacher_deep_overlay": str(TEACHER_MODEL),
            "v27_deep_scout": str(v31.DEFAULT_V27),
            "v21_2_jackpot_runner": str(v31.DEFAULT_JACKPOT),
            "v31_exit_overlay": "fixed alpha1_l2_conservative_fee20 overlay",
        },
        "base_runtime": asdict(base_rt),
        "base_limit_config": asdict(base_cfg),
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha3 + weekly EF runtime calibrator. Frozen model weights are unchanged. EF selects teacher confidence, parent notional scale, and post-only limit/fallback execution parameters once per week using only prior data, then replays 2026 OOS continuously.",
        "experiments": [
            {"name": "alpha3_baseline_fixed_runtime", "runtime": asdict(base_rt), "limit_cfg": asdict(base_cfg), "metrics": baseline, "score": _score(baseline["cost1"], baseline["cost2"], baseline["cost3"])},
            {"name": "alpha3_weekly_ef_calibrator", "metrics": weekly, "score": _score(weekly["cost1"], weekly["cost2"], weekly["cost3"])},
        ],
        "weekly_schedule": schedule_rows,
        "audit": audit,
        "artifacts": {"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT), "schedule": str(SCHEDULE_OUT)},
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "verdict": audit["verdict"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
