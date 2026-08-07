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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_eval_clean_base_deep_gated_gross_v2 as dgg  # noqa: E402
from scripts import train_eval_clean_base_deep_gated_gross_v2_safe_cap_buckets as safe  # noqa: E402
from scripts.experiment_cost_firewall_learned_cap_buckets_2026 import (  # noqa: E402
    _bucket,
    _cap_objective,
    _compact as _cap_compact,
    _cost_pass,
    _learn_cap_map,
    _planned_notional,
    _raw,
    _safe_float,
    _thresholds,
    replay_bucket_map,
)
from scripts.experiment_cost_firewall_notional_cap_sweep_2026 import (  # noqa: E402
    CapCandidate,
    replay as replay_static_cap,
)


MODEL_ID = "safe_cap_shadow_moe_v1"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/safe_cap_shadow_moe_v1"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/safe_cap_shadow_moe_v1_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/safe_cap_shadow_moe_v1_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/safe_cap_shadow_moe_v1_ledger.csv"
DEFAULT_BASE_LEDGER = ROOT / "data/ensemble/reports/safe_cap_shadow_moe_v1_base_safe_cap_ledger.csv"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/safe_cap_shadow_moe_v1_audit.json"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/safe_cap_shadow_moe_v1_contract.md"


@dataclass(frozen=True)
class ShadowMoEConfig:
    name: str
    reduce_threshold: float
    veto_threshold: float
    boost_threshold: float
    reduce_mult: float
    boost_mult: float
    support_margin: float
    cost_buffer_extra: float = 0.0
    max_notional: float = 5.0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def _clip01(v: float) -> float:
    return float(np.clip(float(v), 0.0, 1.0))


def _col(row: pd.Series, name: str, default: float = 0.0) -> float:
    return _safe_float(row.get(name, default), default)


def _shadow_scores(feature_row: pd.Series, side: int) -> tuple[float, float, dict[str, float]]:
    s = 1.0 if int(side) > 0 else -1.0
    ai_support = s * _col(feature_row, "ai_dir_edge")
    m7_prob_support = 0.5 * s * (
        (_col(feature_row, "m7_quant_up") - _col(feature_row, "m7_quant_dn"))
        + (_col(feature_row, "m7_trend_xgb_up") - _col(feature_row, "m7_trend_xgb_dn"))
    )
    m7_return_support = 0.5 * s * (
        np.tanh(_col(feature_row, "m7_q50") * 160.0)
        + np.tanh(_col(feature_row, "m7_expected_ret") * 180.0)
    )
    flow_support = np.tanh(
        s
        * (
            _col(feature_row, "ai_flow_pressure")
            + 3.0 * _col(feature_row, "ai_flow_slope")
            + 3.0 * _col(feature_row, "dlinear_smf_slope")
        )
        - 0.25 * abs(_col(feature_row, "ai_flow_exhaustion"))
        - 0.20 * _col(feature_row, "ai_flow_flip_prob")
    )
    confidence = 0.5 * _clip01(_col(feature_row, "conf_patchtst")) + 0.5 * _clip01(_col(feature_row, "patchtst_regime_sim"))
    reward_score = _clip01(np.log1p(max(_col(feature_row, "ai_reward_risk"), 0.0)) / np.log1p(8.0))
    support = (
        0.42 * ai_support
        + 0.18 * m7_prob_support
        + 0.14 * m7_return_support
        + 0.12 * flow_support
        + 0.08 * confidence
        + 0.06 * reward_score
    )

    directional_disagreement = (
        0.45 * max(0.0, -ai_support)
        + 0.20 * max(0.0, -m7_prob_support)
        + 0.15 * max(0.0, -m7_return_support)
        + 0.10 * max(0.0, -flow_support)
    )
    entropy_risk = _clip01((_col(feature_row, "ai_dir_entropy") - 0.72) / 0.24)
    adverse_risk = _clip01(_col(feature_row, "ai_adverse_risk") / 0.018)
    flip_risk = _clip01((_col(feature_row, "ai_flow_flip_prob") - 0.42) / 0.36)
    exhaust_risk = _clip01((abs(_col(feature_row, "ai_flow_exhaustion")) - 0.22) / 0.50)
    regime_risk = max(_clip01(_col(feature_row, "regime_chop")), _clip01(_col(feature_row, "regime_whipsaw")))
    liquidity_risk = _clip01((abs(_col(feature_row, "liquidity_vacuum")) - 0.62) / 0.38)
    funding_risk = _clip01(abs(_col(feature_row, "funding_pressure")) / 0.055)
    tail_risk = _clip01(_col(feature_row, "evt_tail_flag"))
    conflict = (
        directional_disagreement
        + 0.15 * entropy_risk
        + 0.14 * adverse_risk
        + 0.11 * flip_risk
        + 0.09 * exhaust_risk
        + 0.13 * regime_risk
        + 0.08 * liquidity_risk
        + 0.05 * funding_risk
        + 0.15 * tail_risk
    )
    parts = {
        "ai_support": float(ai_support),
        "m7_prob_support": float(m7_prob_support),
        "m7_return_support": float(m7_return_support),
        "flow_support": float(flow_support),
        "confidence": float(confidence),
        "reward_score": float(reward_score),
        "directional_disagreement": float(directional_disagreement),
        "entropy_risk": float(entropy_risk),
        "adverse_risk": float(adverse_risk),
        "flip_risk": float(flip_risk),
        "regime_risk": float(regime_risk),
        "tail_risk": float(tail_risk),
    }
    return float(support), float(conflict), parts


def _attach_shadow_context(ledger: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    out = ledger.copy()
    support: list[float] = []
    conflict: list[float] = []
    components: list[str] = []
    for _, row in out.iterrows():
        idx = int(row.get("entry_idx", -1))
        if idx < 0 or idx >= len(source_df):
            support.append(0.0)
            conflict.append(1.0)
            components.append("{}")
            continue
        sup, con, parts = _shadow_scores(source_df.iloc[idx], int(row.get("core_side", 0)))
        support.append(sup)
        conflict.append(con)
        components.append(json.dumps(parts, sort_keys=True))
    out["shadow_support_score"] = support
    out["shadow_conflict_score"] = conflict
    out["shadow_components"] = components
    return out


def _fill_price(fill_px: np.ndarray, idx: int, side: int, slip: float, *, entry: bool) -> float:
    px = float(fill_px[int(np.clip(idx, 0, len(fill_px) - 1))])
    if entry:
        return px * (1.0 + slip) if side > 0 else px * (1.0 - slip)
    return px * (1.0 - slip) if side > 0 else px * (1.0 + slip)


def _shadow_decision(base_n: float, support: float, conflict: float, cfg: ShadowMoEConfig) -> tuple[float, str, float]:
    if base_n <= 1e-12:
        return 0.0, "base_blocked", 0.0
    if cfg.name == "shadow_noop":
        return base_n, "shadow_noop", 1.0
    if conflict >= cfg.veto_threshold:
        return 0.0, "shadow_moe_veto", 0.0
    if conflict >= cfg.reduce_threshold and support < conflict + cfg.support_margin:
        return base_n * cfg.reduce_mult, "shadow_moe_reduce", cfg.reduce_mult
    if support >= cfg.boost_threshold and support >= conflict + cfg.support_margin:
        boosted = min(cfg.max_notional, base_n * cfg.boost_mult)
        mult = boosted / max(base_n, 1e-12)
        return boosted, "shadow_moe_boost", float(mult)
    return base_n, "shadow_moe_keep", 1.0


def replay_shadow_moe(
    df: pd.DataFrame,
    *,
    prices: tuple[np.ndarray, np.ndarray],
    safe_candidate: dict[str, Any],
    cfg: ShadowMoEConfig,
    fee: float,
    slip: float,
    cost_mult: float = 1.0,
    exchange_leverage_cap: float = 5.0,
    maintenance_margin: float = 0.006,
    liquidation_fee: float = 0.002,
    ledger_out: Path | None = None,
) -> dict[str, Any]:
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    trades = 0
    blocked = 0
    boosted = 0
    reduced = 0
    vetoed = 0
    liquidations = 0
    ruin_events = 0
    notional_sum = 0.0
    max_notional = 0.0
    max_margin_fraction = 0.0
    min_liq_buffer_pct = float("inf")
    reason_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    close, fill_px = prices
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    scheme = str(safe_candidate["scheme"])
    thresholds = safe_candidate["thresholds"]
    cap_map = {str(k): float(v) for k, v in dict(safe_candidate["cap_map"]).items()}
    fallback_cap = float(safe_candidate["fallback_cap"])
    cost_buffer = float(safe_candidate["cost_buffer"]) + float(cfg.cost_buffer_extra)
    gate_mode = str(safe_candidate["gate_notional_mode"])

    for _, row in df.iterrows():
        old_n = _safe_float(row.get("effective_core_notional", 0.0), 0.0)
        side = int(np.sign(int(row["core_side"])))
        before = cash
        bucket = _bucket(row, scheme, thresholds)
        learned_cap = float(cap_map.get(bucket, fallback_cap))
        base_record = {
            "trade_id": int(row["trade_id"]),
            "entry_idx": int(row["entry_idx"]),
            "core_exit_idx": int(row.get("core_exit_idx", row["entry_idx"])),
            "timestamp": row["timestamp"],
            "core_side": int(row["core_side"]),
            "action": str(row.get("action", "")),
            "regime": str(row.get("regime", "UNKNOWN")),
            "bucket": bucket,
            "learned_cap": learned_cap,
            "original_notional": old_n,
            "shadow_support_score": _safe_float(row.get("shadow_support_score", 0.0), 0.0),
            "shadow_conflict_score": _safe_float(row.get("shadow_conflict_score", 1.0), 1.0),
            "shadow_components": str(row.get("shadow_components", "{}")),
        }
        if old_n <= 1e-12 or cash <= 0.0:
            blocked += 1
            rows.append(
                {
                    **base_record,
                    "base_experiment_notional": 0.0,
                    "experiment_notional": 0.0,
                    "shadow_action": "source_zero_or_account_ruin",
                    "shadow_multiplier": 0.0,
                    "candidate_reasons": "source_zero_or_account_ruin",
                    "cash_before": before,
                    "cash_after": cash,
                    "blocked": True,
                    "base_blocked": True,
                    "liquidated": False,
                    "margin_fraction": 0.0,
                }
            )
            continue
        base_n = _planned_notional(old_n, learned_cap)
        base_passed, base_meta = _cost_pass(row, base_n, learned_cap, cost_buffer, gate_mode, fee_eff, slip_eff)
        if not base_passed:
            blocked += 1
            reason_counts["base_cost_gate_block"] = reason_counts.get("base_cost_gate_block", 0) + 1
            rows.append(
                {
                    **base_record,
                    "base_experiment_notional": 0.0,
                    "experiment_notional": 0.0,
                    "shadow_action": "base_cost_gate_block",
                    "shadow_multiplier": 0.0,
                    "candidate_reasons": "base_cost_gate_block",
                    **base_meta,
                    "cash_before": before,
                    "cash_after": cash,
                    "blocked": True,
                    "base_blocked": True,
                    "liquidated": False,
                    "margin_fraction": 0.0,
                }
            )
            continue

        n, action, mult = _shadow_decision(
            base_n,
            _safe_float(row.get("shadow_support_score", 0.0), 0.0),
            _safe_float(row.get("shadow_conflict_score", 1.0), 1.0),
            cfg,
        )
        n = float(np.clip(n, 0.0, min(float(cfg.max_notional), float(exchange_leverage_cap))))
        if n <= 1e-12:
            blocked += 1
            vetoed += int(action == "shadow_moe_veto")
            reason_counts[action] = reason_counts.get(action, 0) + 1
            rows.append(
                {
                    **base_record,
                    "base_experiment_notional": base_n,
                    "experiment_notional": 0.0,
                    "shadow_action": action,
                    "shadow_multiplier": mult,
                    "candidate_reasons": action,
                    **base_meta,
                    "cash_before": before,
                    "cash_after": cash,
                    "blocked": True,
                    "base_blocked": False,
                    "liquidated": False,
                    "margin_fraction": 0.0,
                }
            )
            continue
        passed, meta = _cost_pass(row, n, learned_cap, cost_buffer, gate_mode, fee_eff, slip_eff)
        if not passed:
            blocked += 1
            reason_counts["shadow_cost_gate_block"] = reason_counts.get("shadow_cost_gate_block", 0) + 1
            rows.append(
                {
                    **base_record,
                    "base_experiment_notional": base_n,
                    "experiment_notional": 0.0,
                    "shadow_action": "shadow_cost_gate_block",
                    "shadow_multiplier": 0.0,
                    "candidate_reasons": "shadow_cost_gate_block",
                    **meta,
                    "cash_before": before,
                    "cash_after": cash,
                    "blocked": True,
                    "base_blocked": False,
                    "liquidated": False,
                    "margin_fraction": 0.0,
                }
            )
            continue

        reasons = [action]
        if n > old_n + 1e-12:
            boosted += 1
            reasons.append("learned_cap_boost")
        if n > base_n + 1e-12:
            reasons.append("shadow_notional_boost")
        if n < base_n - 1e-12:
            reduced += 1
            reasons.append("shadow_notional_reduce")
        margin_fraction = float(n / max(exchange_leverage_cap, 1e-12))
        max_margin_fraction = max(max_margin_fraction, margin_fraction)
        entry_idx = int(row["entry_idx"])
        exit_idx = int(row.get("core_exit_idx", row.get("effective_exit_idx", entry_idx)))
        entry_px = _fill_price(fill_px, min(entry_idx + 1, len(fill_px) - 1), side, slip_eff, entry=True)
        entry_fee = cash * fee_eff * n
        cash -= entry_fee
        trade_min_liq_buffer = float("inf")
        liquidated = False
        liquidation_fee_cash = 0.0
        for j in range(entry_idx, exit_idx + 1):
            px = float(close[int(np.clip(j, 0, len(close) - 1))])
            mark = px * (1.0 - slip_eff) if side > 0 else px * (1.0 + slip_eff)
            unrealized = _raw(side, entry_px, mark) * n
            eq = cash * (1.0 + unrealized)
            liq_floor = before * n * maintenance_margin
            if liq_floor > 0.0:
                trade_min_liq_buffer = min(trade_min_liq_buffer, (eq - liq_floor) / liq_floor * 100.0)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
            if eq <= liq_floor or eq <= 0.0:
                liquidated = True
                liquidations += 1
                liquidation_fee_cash = before * n * liquidation_fee
                cash = max(0.0, eq - liquidation_fee_cash)
                if cash <= 0.0:
                    ruin_events += 1
                break
        min_liq_buffer_pct = min(min_liq_buffer_pct, trade_min_liq_buffer)
        if liquidated:
            after = cash
            pnl_frac = after / max(before, 1e-12) - 1.0
            wins += int(pnl_frac > 0.0)
            trades += 1
            notional_sum += n
            max_notional = max(max_notional, n)
            rows.append(
                {
                    **base_record,
                    "base_experiment_notional": base_n,
                    "experiment_notional": n,
                    "shadow_action": action,
                    "shadow_multiplier": mult,
                    "candidate_reasons": "|".join(reasons),
                    **meta,
                    "entry_fee_cash": entry_fee,
                    "exit_fee_cash": 0.0,
                    "liquidation_fee_cash": liquidation_fee_cash,
                    "trade_pnl_pct": pnl_frac * 100.0,
                    "cash_before": before,
                    "cash_after": after,
                    "blocked": False,
                    "base_blocked": False,
                    "liquidated": True,
                    "margin_fraction": margin_fraction,
                    "min_liq_buffer_pct": trade_min_liq_buffer,
                }
            )
            continue
        exit_px = _fill_price(fill_px, min(exit_idx + 1, len(fill_px) - 1), side, slip_eff, entry=False)
        realized = _raw(side, entry_px, exit_px) * n
        cash *= 1.0 + realized
        exit_fee = cash * fee_eff * n
        cash -= exit_fee
        after = cash
        if cash <= 0.0:
            ruin_events += 1
        pnl_frac = after / max(before, 1e-12) - 1.0
        wins += int(pnl_frac > 0.0)
        trades += 1
        notional_sum += n
        max_notional = max(max_notional, n)
        rows.append(
            {
                **base_record,
                "base_experiment_notional": base_n,
                "experiment_notional": n,
                "shadow_action": action,
                "shadow_multiplier": mult,
                "candidate_reasons": "|".join(reasons),
                **meta,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "realized_raw": realized / max(n, 1e-12),
                "entry_fee_cash": entry_fee,
                "exit_fee_cash": exit_fee,
                "liquidation_fee_cash": liquidation_fee_cash,
                "trade_pnl_pct": pnl_frac * 100.0,
                "cash_before": before,
                "cash_after": after,
                "blocked": False,
                "base_blocked": False,
                "liquidated": False,
                "margin_fraction": margin_fraction,
                "min_liq_buffer_pct": trade_min_liq_buffer,
            }
        )

    if ledger_out is not None:
        ledger_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(ledger_out, index=False)
    days = max((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400.0, 1e-9) if len(df) else 1.0
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "blocked": int(blocked),
        "boosted": int(boosted),
        "reduced": int(reduced),
        "vetoed": int(vetoed),
        "liquidations": int(liquidations),
        "ruin_events": int(ruin_events),
        "wr": float(wins / trades) if trades else 0.0,
        "trades_per_day": float(trades / days),
        "avg_notional": float(notional_sum / trades) if trades else 0.0,
        "max_notional": float(max_notional),
        "max_margin_fraction": float(max_margin_fraction),
        "min_liq_buffer_pct": float(min_liq_buffer_pct if np.isfinite(min_liq_buffer_pct) else np.nan),
        "final_cash": float(cash),
        "reason_counts": reason_counts,
        "cost_mult": float(cost_mult),
    }


def _shadow_configs() -> list[ShadowMoEConfig]:
    out = [
        ShadowMoEConfig(
            "shadow_noop",
            reduce_threshold=99.0,
            veto_threshold=99.0,
            boost_threshold=99.0,
            reduce_mult=1.0,
            boost_mult=1.0,
            support_margin=0.0,
        )
    ]
    for reduce_th in (0.42, 0.52, 0.62, 0.72):
        for veto_th in (0.82, 0.90, 1.01):
            for reduce_mult in (0.45, 0.60, 0.75):
                for boost_th in (0.28, 0.38, 0.48, 0.58):
                    for boost_mult in (1.00, 1.08, 1.15, 1.25):
                        for margin in (0.04, 0.10):
                            name = (
                                f"shadow_r{reduce_th:.2f}_v{veto_th:.2f}_rm{reduce_mult:.2f}_"
                                f"b{boost_th:.2f}_bm{boost_mult:.2f}_m{margin:.2f}"
                            ).replace(".", "p")
                            out.append(
                                ShadowMoEConfig(
                                    name=name,
                                    reduce_threshold=reduce_th,
                                    veto_threshold=veto_th,
                                    boost_threshold=boost_th,
                                    reduce_mult=reduce_mult,
                                    boost_mult=boost_mult,
                                    support_margin=margin,
                                )
                            )
    return out


def _overlay_score(metrics: dict[str, Any], baseline: dict[str, Any]) -> float:
    val = metrics["validation_cost1"]
    c2 = metrics["validation_cost2"]
    c3 = metrics["validation_cost3"]
    base = baseline["validation_cost1"]
    score = float(val["pnl"]) + 0.10 * float(c2["pnl"]) + 0.06 * float(c3["pnl"])
    score += 4.0 * max(0.0, float(val["mdd"]) - float(base["mdd"]))
    score -= 6.0 * max(0.0, abs(float(val["mdd"])) - 25.0)
    score -= 70.0 * max(0.0, 0.85 * float(base["trades"]) - float(val["trades"]))
    score -= 120.0 * int(val.get("liquidations", 0) or 0)
    score -= 250.0 * int(val.get("ruin_events", 0) or 0)
    score -= 80.0 * max(0.0, -float(c2["pnl"]))
    score -= 50.0 * max(0.0, -float(c3["pnl"]))
    return float(score)


def _overlay_blockers(metrics: dict[str, Any], baseline: dict[str, Any], args: argparse.Namespace) -> list[str]:
    val = metrics["validation_cost1"]
    c2 = metrics["validation_cost2"]
    c3 = metrics["validation_cost3"]
    base = baseline["validation_cost1"]
    blockers: list[str] = []
    if int(val.get("liquidations", 0) or 0) > 0:
        blockers.append("validation_liquidation")
    if int(val.get("ruin_events", 0) or 0) > 0:
        blockers.append("validation_ruin")
    if float(val.get("max_margin_fraction", 0.0)) > 1.0 + 1e-12:
        blockers.append("validation_margin_fraction_gt_1")
    if float(c2.get("pnl", -1e9)) <= 0.0:
        blockers.append("validation_cost2_not_survived")
    if float(c3.get("pnl", -1e9)) <= 0.0:
        blockers.append("validation_cost3_not_survived")
    if float(val.get("pnl", -1e9)) < float(base.get("pnl", -1e9)) * float(args.min_validation_pnl_ratio):
        blockers.append("validation_pnl_below_safe_cap_floor")
    if float(val.get("trades", 0.0)) < float(base.get("trades", 0.0)) * float(args.min_validation_trade_ratio):
        blockers.append("validation_trade_count_too_low")
    if float(val.get("mdd", 0.0)) < float(base.get("mdd", 0.0)) - float(args.max_validation_mdd_worsening):
        blockers.append("validation_mdd_worse_than_allowed")
    return blockers


def _select_safe_cap(
    args: argparse.Namespace,
    cap_train: pd.DataFrame,
    cap_train_prices: tuple[np.ndarray, np.ndarray],
    validation: pd.DataFrame,
    validation_prices: tuple[np.ndarray, np.ndarray],
    oos: pd.DataFrame,
    oos_prices: tuple[np.ndarray, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    cap_choices = [float(x) for x in str(args.cap_choices).split(",") if str(x).strip()]
    thresholds = _thresholds(cap_train)
    baseline_cand = CapCandidate("static_cost_firewall_0p0035_cap3p6", 0.0035, 1.0, 3.6, "base")
    baseline = {
        "candidate": {
            "name": baseline_cand.name,
            "scheme": "static",
            "fallback_cap": 3.6,
            "cost_buffer": 0.0035,
            "gate_notional_mode": "base",
        },
        "validation_cost1": replay_static_cap(validation, baseline_cand, prices=validation_prices, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
        "validation_cost2": replay_static_cap(validation, baseline_cand, prices=validation_prices, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
        "validation_cost3": replay_static_cap(validation, baseline_cand, prices=validation_prices, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
        "oos_cost1": replay_static_cap(oos, baseline_cand, prices=oos_prices, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
        "oos_cost2": replay_static_cap(oos, baseline_cand, prices=oos_prices, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
        "oos_cost3": replay_static_cap(oos, baseline_cand, prices=oos_prices, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
    }
    detailed: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for cfg in safe._config_grid(args):
        cap_map, fallback_cap_raw, learn_diag = _learn_cap_map(
            cap_train,
            prices=cap_train_prices,
            cfg=cfg,
            thresholds=thresholds,
            cap_choices=cap_choices,
            fee=float(args.fee),
            slip=float(args.slip),
            exchange_leverage_cap=float(args.exchange_leverage_cap),
        )
        fallback_cap = float(min(float(fallback_cap_raw), float(args.fallback_cap_max)))
        metrics = {
            "validation_cost1": replay_bucket_map(validation, prices=validation_prices, scheme=cfg.scheme, thresholds=thresholds, cap_map=cap_map, fallback_cap=fallback_cap, cost_buffer=cfg.cost_buffer, gate_mode=cfg.gate_notional_mode, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "validation_cost2": replay_bucket_map(validation, prices=validation_prices, scheme=cfg.scheme, thresholds=thresholds, cap_map=cap_map, fallback_cap=fallback_cap, cost_buffer=cfg.cost_buffer, gate_mode=cfg.gate_notional_mode, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "validation_cost3": replay_bucket_map(validation, prices=validation_prices, scheme=cfg.scheme, thresholds=thresholds, cap_map=cap_map, fallback_cap=fallback_cap, cost_buffer=cfg.cost_buffer, gate_mode=cfg.gate_notional_mode, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "oos_cost1": replay_bucket_map(oos, prices=oos_prices, scheme=cfg.scheme, thresholds=thresholds, cap_map=cap_map, fallback_cap=fallback_cap, cost_buffer=cfg.cost_buffer, gate_mode=cfg.gate_notional_mode, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "oos_cost2": replay_bucket_map(oos, prices=oos_prices, scheme=cfg.scheme, thresholds=thresholds, cap_map=cap_map, fallback_cap=fallback_cap, cost_buffer=cfg.cost_buffer, gate_mode=cfg.gate_notional_mode, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "oos_cost3": replay_bucket_map(oos, prices=oos_prices, scheme=cfg.scheme, thresholds=thresholds, cap_map=cap_map, fallback_cap=fallback_cap, cost_buffer=cfg.cost_buffer, gate_mode=cfg.gate_notional_mode, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
        }
        score = safe._score(metrics)
        candidate = {
            **asdict(cfg),
            "cap_map": cap_map,
            "fallback_cap_raw": fallback_cap_raw,
            "fallback_cap": fallback_cap,
            "thresholds": thresholds,
            "learn_diagnostics": learn_diag,
        }
        detailed.append({"candidate": candidate, "score": score, **metrics})
        row: dict[str, Any] = {
            "safe_cap_name": cfg.name,
            "scheme": cfg.scheme,
            "min_bucket_trades": cfg.min_bucket_trades,
            "cost_buffer": cfg.cost_buffer,
            "gate_notional_mode": cfg.gate_notional_mode,
            "fallback_cap": fallback_cap,
            "learned_buckets": len(cap_map),
            "safe_cap_score": score,
        }
        for prefix, data in (
            ("safe_val", metrics["validation_cost1"]),
            ("safe_val_cost2", metrics["validation_cost2"]),
            ("safe_val_cost3", metrics["validation_cost3"]),
            ("safe_oos", metrics["oos_cost1"]),
            ("safe_oos_cost2", metrics["oos_cost2"]),
            ("safe_oos_cost3", metrics["oos_cost3"]),
        ):
            for key in ("pnl", "mdd", "trades", "blocked", "boosted", "liquidations", "ruin_events", "avg_notional", "max_notional", "max_margin_fraction"):
                row[f"{prefix}_{key}"] = data.get(key)
        rows.append(row)
    grid = pd.DataFrame(rows)
    baseline_row = pd.Series({"val_pnl": baseline["validation_cost1"]["pnl"], "val_mdd": baseline["validation_cost1"]["mdd"]})
    blocker_frame = grid.rename(
        columns={
            "safe_val_pnl": "val_pnl",
            "safe_val_mdd": "val_mdd",
            "safe_val_liquidations": "val_liquidations",
            "safe_val_ruin_events": "val_ruin_events",
            "safe_val_max_margin_fraction": "val_max_margin_fraction",
            "safe_val_cost2_pnl": "val_cost2_pnl",
            "safe_val_cost3_pnl": "val_cost3_pnl",
        }
    )
    blockers = blocker_frame.apply(lambda r: safe._selection_blockers(r, baseline_row, args), axis=1)
    grid["safe_selection_eligible"] = blockers.apply(lambda xs: len(xs) == 0)
    grid["safe_selection_blockers"] = blockers.apply(lambda xs: "|".join(xs))
    grid = grid.sort_values("safe_cap_score", ascending=False).reset_index(drop=True)
    eligible = grid[grid["safe_selection_eligible"]].sort_values("safe_cap_score", ascending=False).reset_index(drop=True)
    selected_name = str(eligible.iloc[0]["safe_cap_name"]) if not eligible.empty else baseline_cand.name
    selected = next((d for d in detailed if d["candidate"]["name"] == selected_name), baseline)
    return selected, baseline, grid


def _write_contract(report: dict[str, Any]) -> str:
    sel = report["selected"]
    c1 = sel["oos_cost1"]
    c3 = sel["oos_cost3"]
    return f"""# Safe Cap Shadow Alpha Agreement MoE V1

Status: `{report['verdict']}`

## Architecture

```mermaid
flowchart TD
    A["5m OHLCV + AI Feature Combo"] --> B["Clean Base Deep Gated Gross V2"]
    B --> C["Safe Learned Cap Buckets"]
    A --> D["Shadow Alpha Agreement MoE"]
    D --> E["support / conflict score"]
    C --> F["base safe notional"]
    E --> G["veto / reduce / keep / boost"]
    F --> G
    G --> H["Accounting Replay"]
    H --> I["PnL / MDD / Fees / Slippage Ledger"]
```

## Data Splits

- Parent train: `{report['data']['parent_train_range']}`
- Safe-cap bucket train: `{report['data']['cap_train_range']}`
- Shadow-MoE validation selection: `{report['data']['validation_range']}`
- OOS report-only: `{report['data']['oos_range']}`

## Selected Overlay

- Parent DGG config: `{report['selected_parent_config']['name']}`
- Safe cap: `{report['selected_safe_cap']['candidate']['name']}`
- Shadow config: `{sel['candidate']['name']}`

## OOS Result

- PnL: `{c1['pnl']:.6f}%`
- MDD: `{c1['mdd']:.6f}%`
- Trades: `{c1['trades']}`
- Average notional: `{c1['avg_notional']:.6f}`
- 3x cost PnL: `{c3['pnl']:.6f}%`

## Invariants

- Shadow layer never changes side or exit index.
- Shadow layer never creates a trade blocked by the safe-cap parent.
- Cap map is learned before validation.
- Shadow config is selected on 2025 validation only.
- 2026 OOS is report-only.
- Fees and slippage are charged on final notional.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate Shadow Alpha Agreement MoE on top of safe learned cap.")
    p.add_argument("--policy", type=Path, default=dgg.v1.base.DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=dgg.v1.base.DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=dgg.v1.base.DEFAULT_AUDIT)
    p.add_argument("--lifecycle-report", type=Path, default=dgg.v1.base.DEFAULT_LIFECYCLE_REPORT)
    p.add_argument("--lifecycle-model", type=Path, default=dgg.v1.base.DEFAULT_LIFECYCLE_MODEL)
    p.add_argument("--train-csv", type=Path, default=dgg.v1.base.DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=dgg.v1.base.DEFAULT_EVAL_CSV)
    p.add_argument("--parent-train-end", default="2025-10-01")
    p.add_argument("--cap-train-end", default="2025-11-01")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--deep-epochs", type=int, default=12)
    p.add_argument("--deep-batch-size", type=int, default=128)
    p.add_argument("--exchange-leverage-cap", type=float, default=5.0)
    p.add_argument("--cap-choices", default="3.6,4.0,4.5,5.0")
    p.add_argument("--fallback-cap-max", type=float, default=3.6)
    p.add_argument("--min-bucket-trades-floor", type=int, default=10)
    p.add_argument("--min-cost-buffer", type=float, default=0.0035)
    p.add_argument("--max-validation-mdd-worsening", type=float, default=8.0)
    p.add_argument("--min-validation-pnl-ratio", type=float, default=0.94)
    p.add_argument("--min-validation-trade-ratio", type=float, default=0.82)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid-csv-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--ledger-csv-out", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--base-ledger-csv-out", type=Path, default=DEFAULT_BASE_LEDGER)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--contract-out", type=Path, default=DEFAULT_CONTRACT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cap_choices = [float(x) for x in str(args.cap_choices).split(",") if str(x).strip()]
    if max(cap_choices) > float(args.exchange_leverage_cap) + 1e-12:
        raise SystemExit("cap choices exceed exchange leverage cap")

    train_full = dgg.v1.base._read(args.train_csv)
    parent_train = safe._split_by_date(train_full, None, args.parent_train_end)
    cap_train_raw = safe._split_by_date(train_full, args.parent_train_end, args.cap_train_end)
    validation_raw = safe._split_by_date(train_full, args.cap_train_end, None)
    oos_df = dgg.v1.base._read(args.eval_csv)
    if parent_train.empty or cap_train_raw.empty or validation_raw.empty or oos_df.empty:
        raise SystemExit("empty chronological split")

    bundle = safe._build_parent_model(args, parent_train)
    selected_parent, parent_grid, parent_val = safe._select_dgg_config(args, bundle, validation_raw)

    cap_train, cap_train_prices, cap_train_parent = safe._dgg_ledger_for(args, bundle, selected_parent, cap_train_raw)
    validation, validation_prices, validation_parent = safe._dgg_ledger_for(args, bundle, selected_parent, validation_raw)
    oos, oos_prices, oos_parent = safe._dgg_ledger_for(args, bundle, selected_parent, oos_df)
    validation = _attach_shadow_context(validation, validation_raw)
    oos = _attach_shadow_context(oos, oos_df)

    selected_safe, safe_static_baseline, safe_grid = _select_safe_cap(
        args,
        cap_train,
        cap_train_prices,
        validation,
        validation_prices,
        oos,
        oos_prices,
    )
    if str(selected_safe["candidate"].get("scheme")) == "static":
        raise SystemExit("safe cap parent selection fell back to static; refusing to layer shadow MoE")

    safe_candidate = selected_safe["candidate"]
    safe_base_validation = {
        "validation_cost1": selected_safe["validation_cost1"],
        "validation_cost2": selected_safe["validation_cost2"],
        "validation_cost3": selected_safe["validation_cost3"],
    }
    args.base_ledger_csv_out.parent.mkdir(parents=True, exist_ok=True)
    replay_bucket_map(
        oos,
        prices=oos_prices,
        scheme=safe_candidate["scheme"],
        thresholds=safe_candidate["thresholds"],
        cap_map=safe_candidate["cap_map"],
        fallback_cap=safe_candidate["fallback_cap"],
        cost_buffer=safe_candidate["cost_buffer"],
        gate_mode=safe_candidate["gate_notional_mode"],
        fee=args.fee,
        slip=args.slip,
        cost_mult=1.0,
        exchange_leverage_cap=args.exchange_leverage_cap,
        ledger_out=args.base_ledger_csv_out,
    )

    detailed: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for cfg in _shadow_configs():
        metrics = {
            "validation_cost1": replay_shadow_moe(validation, prices=validation_prices, safe_candidate=safe_candidate, cfg=cfg, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "validation_cost2": replay_shadow_moe(validation, prices=validation_prices, safe_candidate=safe_candidate, cfg=cfg, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "validation_cost3": replay_shadow_moe(validation, prices=validation_prices, safe_candidate=safe_candidate, cfg=cfg, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
        }
        score = _overlay_score(metrics, safe_base_validation)
        blockers = _overlay_blockers(metrics, safe_base_validation, args)
        candidate = asdict(cfg)
        detailed.append({"candidate": candidate, "score": score, "selection_blockers": blockers, **metrics})
        row: dict[str, Any] = {
            **candidate,
            "score": score,
            "selection_eligible": len(blockers) == 0,
            "selection_blockers": "|".join(blockers),
        }
        for prefix, data in (
            ("val", metrics["validation_cost1"]),
            ("val_cost2", metrics["validation_cost2"]),
            ("val_cost3", metrics["validation_cost3"]),
        ):
            for key in ("pnl", "mdd", "trades", "blocked", "boosted", "reduced", "vetoed", "liquidations", "ruin_events", "avg_notional", "max_notional", "max_margin_fraction"):
                row[f"{prefix}_{key}"] = data.get(key)
        rows.append(row)

    grid = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    eligible = grid[grid["selection_eligible"]].sort_values("score", ascending=False).reset_index(drop=True)
    selected_name = str(eligible.iloc[0]["name"]) if not eligible.empty else "shadow_noop"
    selected = next(d for d in detailed if d["candidate"]["name"] == selected_name)
    selected_cfg = ShadowMoEConfig(**selected["candidate"])
    selected["oos_cost1"] = replay_shadow_moe(
        oos,
        prices=oos_prices,
        safe_candidate=safe_candidate,
        cfg=selected_cfg,
        fee=args.fee,
        slip=args.slip,
        cost_mult=1.0,
        exchange_leverage_cap=args.exchange_leverage_cap,
        ledger_out=args.ledger_csv_out,
    )
    selected["oos_cost2"] = replay_shadow_moe(
        oos,
        prices=oos_prices,
        safe_candidate=safe_candidate,
        cfg=selected_cfg,
        fee=args.fee,
        slip=args.slip,
        cost_mult=2.0,
        exchange_leverage_cap=args.exchange_leverage_cap,
    )
    selected["oos_cost3"] = replay_shadow_moe(
        oos,
        prices=oos_prices,
        safe_candidate=safe_candidate,
        cfg=selected_cfg,
        fee=args.fee,
        slip=args.slip,
        cost_mult=3.0,
        exchange_leverage_cap=args.exchange_leverage_cap,
    )

    selected_ledger = pd.read_csv(args.ledger_csv_out)
    base_ledger = pd.read_csv(args.base_ledger_csv_out)
    blocking: list[str] = []
    if len(selected_ledger) != len(oos):
        blocking.append("selected OOS ledger row count mismatch")
    if len(base_ledger) != len(oos):
        blocking.append("base OOS ledger row count mismatch")
    if len(selected_ledger) == len(base_ledger):
        created = (
            (pd.to_numeric(base_ledger.get("experiment_notional", 0.0), errors="coerce").fillna(0.0) <= 1e-12)
            & (pd.to_numeric(selected_ledger.get("experiment_notional", 0.0), errors="coerce").fillna(0.0) > 1e-12)
        )
        if bool(created.any()):
            blocking.append("shadow layer created trades blocked by safe cap")
    if int(selected["oos_cost1"].get("liquidations", 0) or 0) > 0:
        blocking.append("selected OOS liquidation")
    if int(selected["oos_cost1"].get("ruin_events", 0) or 0) > 0:
        blocking.append("selected OOS account ruin")
    if float(selected["oos_cost1"].get("max_margin_fraction", 0.0)) > 1.0 + 1e-12:
        blocking.append("selected OOS margin fraction above 1")
    if float(selected["oos_cost2"].get("pnl", 0.0)) <= 0.0:
        blocking.append("selected OOS cost2 failed")
    if float(selected["oos_cost3"].get("pnl", 0.0)) <= 0.0:
        blocking.append("selected OOS cost3 failed")
    required_cols = {"cash_before", "cash_after", "entry_fee_cash", "exit_fee_cash", "trade_pnl_pct", "experiment_notional", "base_experiment_notional"}
    missing_cols = sorted(required_cols - set(selected_ledger.columns))
    if missing_cols:
        blocking.append(f"selected ledger missing accounting columns: {missing_cols}")

    audit = {
        "model_id": MODEL_ID,
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "invariants": {
            "parent_trained_before_cap_train": True,
            "cap_map_learned_before_validation": True,
            "shadow_selection_uses_2025_validation_only": True,
            "oos_is_2026_report_only": True,
            "shadow_never_flips_side": True,
            "shadow_never_changes_exit": True,
            "shadow_does_not_create_safe_cap_blocked_entries": not any("created trades" in b for b in blocking),
            "fees_and_slippage_on_final_notional": True,
            "max_margin_fraction_lte_1": float(selected["oos_cost1"].get("max_margin_fraction", 0.0)) <= 1.0 + 1e-12,
            "no_liquidations": int(selected["oos_cost1"].get("liquidations", 0) or 0) == 0,
            "no_ruin": int(selected["oos_cost1"].get("ruin_events", 0) or 0) == 0,
            "cost2_survives": float(selected["oos_cost2"].get("pnl", 0.0)) > 0.0,
            "cost3_survives": float(selected["oos_cost3"].get("pnl", 0.0)) > 0.0,
        },
    }

    args.model_dir.mkdir(parents=True, exist_ok=True)
    torch_out = args.model_dir / "gru_state_encoder_ensemble.pt"
    model_out = args.model_dir / "safe_cap_shadow_moe_v1.pkl"
    policy_out = args.model_dir / "shadow_moe_policy.json"
    torch.save({"models": [m.state_dict() for m in bundle["deep_model"].models], "meta": bundle["deep_meta"], "sequence_features": bundle["seq_features"]}, torch_out)
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "sequence_features": bundle["seq_features"],
            "sequence_scaler": bundle["seq_scaler"],
            "state_model": bundle["state_model"],
            "head_model": bundle["head_model"],
            "head_meta": bundle["head_meta"],
            "deep_meta": bundle["deep_meta"],
            "selected_parent_config": asdict(selected_parent),
            "selected_safe_cap_candidate": safe_candidate,
            "selected_shadow_config": selected["candidate"],
            "torch_model": str(torch_out),
        },
        model_out,
    )
    policy_out.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "selected_safe_cap_candidate": safe_candidate,
                "selected_shadow_config": selected["candidate"],
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.grid_csv_out, index=False)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    verdict = "promote_candidate" if audit["status"] == "pass" and selected_name != "shadow_noop" else "reject_or_noop"
    report = {
        "model_id": MODEL_ID,
        "verdict": verdict,
        "selected_parent_config": asdict(selected_parent),
        "parent_validation": parent_val,
        "selected_safe_cap": {
            **selected_safe,
            "compact": {
                "validation_cost1": _cap_compact(selected_safe["validation_cost1"]),
                "oos_cost1": _cap_compact(selected_safe["oos_cost1"]),
                "oos_cost2": _cap_compact(selected_safe["oos_cost2"]),
                "oos_cost3": _cap_compact(selected_safe["oos_cost3"]),
            },
        },
        "safe_static_baseline": {
            **safe_static_baseline,
            "compact": {
                "validation_cost1": _cap_compact(safe_static_baseline["validation_cost1"]),
                "oos_cost1": _cap_compact(safe_static_baseline["oos_cost1"]),
                "oos_cost2": _cap_compact(safe_static_baseline["oos_cost2"]),
                "oos_cost3": _cap_compact(safe_static_baseline["oos_cost3"]),
            },
        },
        "selected": {
            **selected,
            "compact": {
                "validation_cost1": _cap_compact(selected["validation_cost1"]),
                "oos_cost1": _cap_compact(selected["oos_cost1"]),
                "oos_cost2": _cap_compact(selected["oos_cost2"]),
                "oos_cost3": _cap_compact(selected["oos_cost3"]),
            },
        },
        "audit_path": str(args.audit_out),
        "audit": audit,
        "data": {
            "parent_train_range": dgg.v1.base._range(parent_train),
            "cap_train_range": dgg.v1.base._range(cap_train_raw),
            "validation_range": dgg.v1.base._range(validation_raw),
            "oos_range": dgg.v1.base._range(oos_df),
            "parent_train_rows": int(len(parent_train)),
            "cap_train_parent_trades": int(len(cap_train)),
            "validation_parent_trades": int(len(validation)),
            "oos_parent_trades": int(len(oos)),
        },
        "artifacts": {
            "model": str(model_out),
            "torch_model": str(torch_out),
            "policy_json": str(policy_out),
            "report": str(args.report_out),
            "grid_csv": str(args.grid_csv_out),
            "ledger_csv": str(args.ledger_csv_out),
            "base_ledger_csv": str(args.base_ledger_csv_out),
            "audit": str(args.audit_out),
            "contract": str(args.contract_out),
        },
        "parent_grid_top10": sorted(parent_grid, key=lambda r: r["parent_selection_score"], reverse=True)[:10],
        "safe_cap_grid_top10": safe_grid.head(10).to_dict(orient="records"),
        "shadow_grid_top15": grid.head(15).to_dict(orient="records"),
        "shadow_grid_top_eligible": eligible.head(15).to_dict(orient="records"),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.contract_out.parent.mkdir(parents=True, exist_ok=True)
    args.contract_out.write_text(_write_contract(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "selected_safe_cap": selected_safe["candidate"]["name"],
                "selected_shadow": selected_name,
                "audit": audit["status"],
                "safe_cap_oos_cost1": report["selected_safe_cap"]["compact"]["oos_cost1"],
                "shadow_oos_cost1": report["selected"]["compact"]["oos_cost1"],
                "shadow_oos_cost2": report["selected"]["compact"]["oos_cost2"],
                "shadow_oos_cost3": report["selected"]["compact"]["oos_cost3"],
                "report": str(args.report_out),
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
