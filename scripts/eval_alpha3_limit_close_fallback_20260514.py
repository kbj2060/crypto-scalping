#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as base  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha3_limit_close_fallback_20260514"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_limit_close_fallback_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_limit_close_fallback_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_limit_close_fallback_20260514_grid.csv"


def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(col, default))
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _close_fallback_price(df: pd.DataFrame, idx: int, side: int, slip: float, *, entry: bool) -> float:
    px = float(pd.to_numeric(df["close"], errors="coerce").ffill().iloc[int(np.clip(idx, 0, len(df) - 1))])
    if side > 0:
        return px * (1.0 + slip if entry else 1.0 - slip)
    return px * (1.0 - slip if entry else 1.0 + slip)


def _try_immediate_limit_close_fallback(
    df: pd.DataFrame,
    signal_i: int,
    side: int,
    cfg: base.ImmediateLimitConfig,
    *,
    entry: bool,
    fee: float,
    slip: float,
) -> tuple[bool, float, float, float, str]:
    fill_i = min(int(signal_i) + 1, len(df) - 1)
    offset = cfg.entry_offset_bps if entry else cfg.exit_offset_bps
    limit_px = base._limit_price(df, signal_i, side, entry=entry, offset_bps=offset, anchor=cfg.anchor)
    if limit_px > 0.0 and base._limit_touched(df, fill_i, limit_px, side, entry=entry, penetration_bps=cfg.penetration_bps):
        return True, float(limit_px), float(fee * cfg.maker_fee_mult), 0.0, "signal_immediate_maker_limit"
    if entry:
        if cfg.entry_miss == "market_fallback":
            return (
                True,
                float(_close_fallback_price(df, fill_i, side, slip, entry=True)),
                float(fee),
                float(slip),
                "entry_market_fallback_after_limit_miss_close",
            )
        return False, 0.0, 0.0, 0.0, "signal_immediate_limit_miss"
    if cfg.exit_miss != "market_fallback":
        return False, 0.0, 0.0, 0.0, "signal_immediate_limit_miss"
    return (
        True,
        float(_close_fallback_price(df, fill_i, side, slip, entry=False)),
        float(fee),
        float(slip),
        "exit_market_fallback_after_limit_miss_close",
    )


def _metrics_signal_limit_close(df, parent, jackpot_model, add_cfg, q, decisions, overlay, cfg, *, fee, slip) -> dict[str, Any]:
    original = base._try_immediate_limit
    base._try_immediate_limit = _try_immediate_limit_close_fallback
    try:
        return base._metrics_signal_limit(df, parent, jackpot_model, add_cfg, q, decisions, overlay, cfg, fee=fee, slip=slip)
    finally:
        base._try_immediate_limit = original


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def main() -> int:
    print(f"[{MODEL_ID}] loading frozen Alpha3 stack", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    alpha2_audit = json.loads(base.ALPHA2_AUDIT.read_text(encoding="utf-8"))
    runtime = dict(alpha2_audit.get("selected_runtime", {}) or {})
    rt = alpha2.Alpha2Runtime(
        name=str(runtime.get("name", "noflip_c0.56_parent_scale1.10")),
        confidence=float(runtime.get("confidence", 0.56)),
        parent_notional_scale=float(runtime.get("parent_notional_scale", 1.10)),
        max_notional=float(runtime.get("max_notional", 2.75)),
    )
    selected_variant = next(v for v in l2._variants() if v.name == "alpha1_l2_conservative_fee20")
    overlay = selected_variant.overlay
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    _, v27_model = v31._load_v27(v31.DEFAULT_V27)
    v27_payload = torch.load(v31.DEFAULT_V27, map_location="cpu", weights_only=False)
    teacher_payload = torch.load(base.TEACHER_MODEL, map_location="cpu", weights_only=False)
    teacher_model = alpha2._load_teacher_model(teacher_payload)
    feature_cols = list(teacher_payload["feature_cols"])
    norm = dict(dict(teacher_payload["train_meta"])["norm"])
    buckets = tuple(float(x) for x in teacher_payload["buckets"])
    fee = float(dict(parent["config"])["fee"])
    slip = float(dict(parent["config"])["slip"])

    train_all = _read(v31.DEFAULT_TRAIN)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    eval_df = _read(v31.DEFAULT_EVAL)
    print(f"[{MODEL_ID}] rebuilding decisions and V27 q", flush=True)
    val_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    val_features = prepare_features(val, side_hint=0, close=_close(val), feature_cols=feature_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    val_pred = teacher._predict_deep(teacher_model, val_features, feature_cols, norm)
    eval_pred = teacher._predict_deep(teacher_model, eval_features, feature_cols, norm)
    val_dec = alpha2._decisions(val_dec, val_pred, buckets, rt)
    eval_dec = alpha2._decisions(eval_dec, eval_pred, buckets, rt)
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    rows: list[dict[str, Any]] = []
    best_cfg: base.ImmediateLimitConfig | None = None
    best_entry_fallback_cfg: base.ImmediateLimitConfig | None = None
    best_score = -1e18
    best_entry_fallback_score = -1e18
    print(f"[{MODEL_ID}] selecting close-fallback execution config on 2025Q4", flush=True)
    for cfg in base._configs():
        m = _metrics_signal_limit_close(val, parent, jackpot_model, add_cfg, val_q, val_dec, overlay, cfg, fee=fee, slip=slip)
        score = _score(m)
        rows.append(
            {
                **asdict(cfg),
                "selection_score": score,
                "val_cost1_pnl": m["cost1"]["pnl"],
                "val_cost1_mdd": m["cost1"]["mdd"],
                "val_cost1_trades": m["cost1"]["trades"],
                "val_cost2_pnl": m["cost2"]["pnl"],
                "val_cost3_pnl": m["cost3"]["pnl"],
            }
        )
        print(
            f"[{MODEL_ID}] {cfg.name} val c1={m['cost1']['pnl']:.2f} mdd={m['cost1']['mdd']:.2f} "
            f"c2={m['cost2']['pnl']:.2f} c3={m['cost3']['pnl']:.2f}",
            flush=True,
        )
        if score > best_score:
            best_score = float(score)
            best_cfg = cfg
        if str(cfg.entry_miss) == "market_fallback" and score > best_entry_fallback_score:
            best_entry_fallback_score = float(score)
            best_entry_fallback_cfg = cfg
    assert best_cfg is not None
    assert best_entry_fallback_cfg is not None
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)

    alpha3_current_cfg = base.ImmediateLimitConfig(
        "next_open_limit_offset2_entry_fallback_fee20",
        "next_open",
        2.0,
        2.0,
        0.5,
        0.20,
        entry_miss="market_fallback",
        exit_miss="market_fallback",
    )
    signal_close_cfg = base.ImmediateLimitConfig(
        "signal_close_limit_offset2_entry_fallback_fee20",
        "signal_close",
        2.0,
        2.0,
        0.5,
        0.20,
        entry_miss="market_fallback",
        exit_miss="market_fallback",
    )
    print(f"[{MODEL_ID}] fixed 2026 OOS", flush=True)
    experiments = []
    for name, cfg in [
        ("alpha3_current_config_close_fallback", alpha3_current_cfg),
        ("alpha3_signal_close_config_close_fallback", signal_close_cfg),
        (f"alpha3_best_entry_fallback_close::{best_entry_fallback_cfg.name}", best_entry_fallback_cfg),
        (f"alpha3_selected_close_fallback::{best_cfg.name}", best_cfg),
    ]:
        metrics = _metrics_signal_limit_close(eval_df, parent, jackpot_model, add_cfg, eval_q, eval_dec, overlay, cfg, fee=fee, slip=slip)
        experiments.append({"name": name, "config": asdict(cfg), "metrics": metrics, "score": _score(metrics)})
        print(
            f"[{MODEL_ID}] {name} c1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} "
            f"c2={metrics['cost2']['pnl']:.2f} c3={metrics['cost3']['pnl']:.2f}",
            flush=True,
        )

    audit = {
        "status": "pass",
        "verdict": "causal_ohlcv_close_fallback_retest",
        "blocking": [],
        "warnings": [
            "limit_fill_uses_next_bar_5m_high_low_touch_proxy_not_orderbook_queue",
            "fallback_uses_same_next_bar_close_after_touch_window_to_preserve_ohlcv_time_order",
            "live_two_second_fallback_requires_forward_shadow_validation",
        ],
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "selected_config": asdict(best_cfg),
        "selected_entry_fallback_config": asdict(best_entry_fallback_cfg),
        "fallback_contract": "signal i -> maker touch check on i+1 high/low -> if miss, market fallback at i+1 close +/- slippage",
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Frozen Alpha3 decision stack retested with OHLCV-causal close fallback. Limit touch is checked on the next bar high/low; missed maker orders fall back at that same next bar close with taker fee/slippage instead of next bar open.",
        "experiments": experiments,
        "selection_grid": str(GRID_OUT),
        "audit": audit,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT), "selected": best_cfg.name}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
