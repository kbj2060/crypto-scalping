#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    build_training_set,
    predict_policy_frame,
    prepare_features,
)
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.eval_hf_v13_v40_6_full_v31_stack_retrain import (  # noqa: E402
    DEFAULT_EVAL,
    DEFAULT_PARENT,
    DEFAULT_PARENT_REPORT,
    DEFAULT_TRAIN,
    _build_v40_6_frames,
    _load_bundle,
    _projection_targets,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _audit_contract,
    _close,
    _days,
    _feature_cols,
    _fill_price,
    _json_default,
    _read,
)
from scripts.train_eval_hf_v13_multitrack_foundation_parent_v40 import _parent_cfg  # noqa: E402


MODEL_ID = "hf_v13_v40_6_nohold_deep_scout_v42_20260512"
DEFAULT_V27 = ROOT / "data/ensemble/supervised/hf_v13_deep_alpha_candidate_expansion_v27_20260511/v27_deep_alpha_candidate_expansion.pt"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v40_6_nohold_deep_scout_v42_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v40_6_nohold_deep_scout_v42_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v40_6_nohold_deep_scout_v42_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v40_6_nohold_deep_scout_v42_20260512_grid.csv"


def _scout_grid() -> list[v31.OverlayConfig]:
    rows: list[v31.OverlayConfig] = []
    for cfg in v31._grid():
        rows.append(
            replace(
                cfg,
                name=f"{cfg.name}_nohold_cd",
                cooldown=0,
                base_hold=0,
            )
        )
    rows.extend(
        [
            v31.OverlayConfig(
                "v42_precision_nohold_cd",
                0.014,
                0.006,
                0.8,
                0,
                0.034,
                0.016,
                0,
                1.0,
                2.4,
                0.8,
                0.65,
                18,
                0.030,
                0.065,
                0.032,
            ),
            v31.OverlayConfig(
                "v42_balanced_nohold_cd",
                0.010,
                0.004,
                1.0,
                0,
                0.040,
                0.018,
                0,
                1.5,
                2.5,
                1.0,
                0.50,
                18,
                0.025,
                0.075,
                0.036,
            ),
        ]
    )
    for edge in (0.018, 0.022, 0.026, 0.030, 0.035, 0.040):
        for notional in (0.25, 0.40, 0.60):
            rows.append(
                v31.OverlayConfig(
                    f"v42_ultra_precision_e{edge:.3f}_n{notional:.2f}_nohold_cd",
                    edge,
                    max(0.006, edge * 0.45),
                    notional,
                    0,
                    0.030,
                    0.014,
                    0,
                    0.8,
                    2.2,
                    0.7,
                    0.70,
                    12,
                    0.035,
                    0.055,
                    0.028,
                )
            )
    return rows


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(
        c1["pnl"]
        + 0.30 * c2["pnl"]
        + 0.20 * c3["pnl"]
        - 0.30 * abs(c1["mdd"])
        + 0.12 * min(float(c1.get("deep_entries", 0)), 80.0)
    )


def backtest_nohold_deep(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    deep_q: np.ndarray,
    cfg: v31.OverlayConfig | None,
    *,
    fee: float,
    slip: float,
    cost_mult: float = 1.0,
    decisions: pd.DataFrame | None = None,
    enable_deep: bool = True,
    record: bool = False,
) -> dict[str, Any]:
    close = _close(df)
    if decisions is None:
        decisions = predict_policy_frame(bundle, df, close=close)
    fee_eff = fee * cost_mult
    slip_eff = slip * cost_mult
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    owner = ""
    entry_price = entry_equity = 0.0
    entry_idx = 0
    notional = 0.0
    leverage = 1.0
    take_profit = stop_loss = 0.0
    entry_edge = 0.0
    entry_margin = 0.0
    entry_vol_anchor = 0.0
    mfe = mae = 0.0
    trades = wins = long_entries = short_entries = deep_entries = parent_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    actions: dict[str, int] = {"cash": 0, "parent_entry": 0, "deep_entry": 0}
    records: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (
            (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12)
            if pos > 0
            else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
        )
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

        if pos != 0:
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold = i - entry_idx
            reason = ""
            effective_tp = take_profit
            effective_sl = stop_loss

            if owner == "deep_scout" and cfg is not None:
                if cfg.tp_util_mult > 0.0:
                    util_gain = 1.0 + cfg.tp_util_mult * max(entry_edge - cfg.edge_th, 0.0) / max(0.02, cfg.edge_th)
                    effective_tp = float(np.clip(cfg.base_tp * util_gain, cfg.base_tp * 0.8, cfg.tp_cap))
                if cfg.sl_vol_mult > 0.0:
                    effective_sl = float(np.clip(entry_vol_anchor * cfg.sl_vol_mult, cfg.base_sl * 0.6, cfg.sl_cap))
                if mfe > 0.0 and cfg.trail_gap_mult > 0.0:
                    trail_gap = entry_vol_anchor * cfg.trail_gap_mult
                    if cfg.hold_decay_start < 999 and hold >= cfg.hold_decay_start:
                        decay_bars = hold - cfg.hold_decay_start
                        trail_gap = max(entry_vol_anchor * 0.35, trail_gap - cfg.hold_decay_rate * decay_bars * entry_vol_anchor)
                    trail_stop = max(-effective_sl, mfe - trail_gap)
                    effective_sl = min(effective_sl, max(0.001, trail_stop))

            if effective_tp > 0.0 and unreal >= effective_tp:
                reason = f"{owner}_take_profit"
            elif effective_sl > 0.0 and unreal <= -abs(effective_sl):
                reason = f"{owner}_stop_loss"

            if reason:
                fill_i = min(i + 1, len(df) - 1)
                exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
                raw = (
                    (exit_px - entry_price) / max(entry_price, 1e-12)
                    if pos > 0
                    else (entry_price - exit_px) / max(entry_price, 1e-12)
                )
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee_eff * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update(
                        {
                            "exit_signal_timestamp": str(df["timestamp"].iloc[i]),
                            "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]),
                            "exit_reason": reason,
                            "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0),
                            "final_notional_exposure": float(notional),
                            "mfe_pct": float(mfe * 100.0),
                            "mae_pct": float(mae * 100.0),
                            "effective_tp": float(effective_tp),
                            "effective_sl": float(effective_sl),
                            "fee_exit_pct": float(fee_eff * notional * 100.0),
                            "cash_after": float(cash),
                        }
                    )
                    records.append(out)
                pos = 0
                owner = ""
                open_record = None
                mfe = mae = 0.0
                continue

        if pos != 0:
            continue

        dec = decisions.iloc[i]
        if int(dec.action) != ACTION_CASH and int(dec.side) != 0:
            fill_i = min(i + 1, len(df) - 1)
            pos = int(dec.side)
            owner = "v40_6_parent"
            entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
            entry_equity = cash
            entry_idx = i
            notional = float(dec.notional_exposure)
            leverage = float(dec.leverage)
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            cash -= cash * fee_eff * notional
            parent_entries += 1
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += leverage
            actions["parent_entry"] += 1
            if record:
                open_record = {
                    "entry_signal_timestamp": str(df["timestamp"].iloc[i]),
                    "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]),
                    "owner": owner,
                    "side": "LONG" if pos > 0 else "SHORT",
                    "entry_price": float(entry_price),
                    "notional_exposure": float(notional),
                    "leverage": float(leverage),
                    "take_profit": float(take_profit),
                    "stop_loss": float(stop_loss),
                    "raw_max_hold_bars": int(dec.max_hold_bars),
                    "raw_cooldown_bars": int(dec.cooldown_bars),
                    "effective_max_hold_bars": 0,
                    "effective_cooldown_bars": 0,
                    "quality_score": float(dec.quality_score),
                    "confidence": float(dec.confidence),
                    "fee_entry_pct": float(fee_eff * notional * 100.0),
                }
            continue

        actions["cash"] += 1
        if enable_deep and cfg is not None and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= cfg.edge_th and margin >= cfg.margin_th:
                fill_i = min(i + 1, len(df) - 1)
                pos = side
                owner = "deep_scout"
                entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                entry_equity = cash
                entry_idx = i
                notional = float(cfg.notional)
                leverage = max(float(cfg.notional), 1.0)
                take_profit = float(cfg.base_tp)
                stop_loss = float(cfg.base_sl)
                entry_edge = edge
                entry_margin = margin
                entry_vol_anchor = v31._vol_anchor(df.iloc[i]) * notional
                cash -= cash * fee_eff * notional
                deep_entries += 1
                long_entries += int(pos > 0)
                short_entries += int(pos < 0)
                notional_sum += notional
                leverage_sum += leverage
                actions["deep_entry"] += 1
                if record:
                    open_record = {
                        "entry_signal_timestamp": str(df["timestamp"].iloc[i]),
                        "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]),
                        "owner": owner,
                        "side": "LONG" if pos > 0 else "SHORT",
                        "entry_price": float(entry_price),
                        "notional_exposure": float(notional),
                        "leverage": float(leverage),
                        "deep_q_long": ql,
                        "deep_q_short": qs,
                        "deep_edge": float(edge),
                        "deep_margin": float(margin),
                        "deep_vol_anchor": float(entry_vol_anchor),
                        "take_profit": float(take_profit),
                        "stop_loss": float(stop_loss),
                        "effective_max_hold_bars": 0,
                        "effective_cooldown_bars": 0,
                        "fee_entry_pct": float(fee_eff * notional * 100.0),
                    }

    if pos != 0:
        fill_i = len(df) - 1
        exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
        raw = (
            (exit_px - entry_price) / max(entry_price, 1e-12)
            if pos > 0
            else (entry_price - exit_px) / max(entry_price, 1e-12)
        )
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
        if record and open_record is not None:
            out = dict(open_record)
            out.update(
                {
                    "exit_signal_timestamp": str(df["timestamp"].iloc[fill_i]),
                    "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]),
                    "exit_reason": "forced_end",
                    "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0),
                    "final_notional_exposure": float(notional),
                    "mfe_pct": float(mfe * 100.0),
                    "mae_pct": float(mae * 100.0),
                    "fee_exit_pct": float(fee_eff * notional * 100.0),
                    "cash_after": float(cash),
                }
            )
            records.append(out)

    n = max(long_entries + short_entries, 1)
    out = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)),
        "deep_entries": int(deep_entries),
        "parent_entries": int(parent_entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n),
        "avg_leverage": float(leverage_sum / n),
        "exits": exits,
        "actions": actions,
    }
    if record:
        out["trade_records"] = records
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attach frozen V27 Deep Scout to v40.6 no-max-hold/no-cooldown parent.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--parent-report", type=Path, default=DEFAULT_PARENT_REPORT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--train-stride", type=int, default=48)
    p.add_argument("--embed-batch", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[{MODEL_ID}] loading data and v40.6 parent", flush=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    parent_bundle = _load_bundle(args.parent_model)
    with args.parent_report.open(encoding="utf-8") as f:
        parent_report = json.load(f)

    feature_cols = _feature_cols(train_all, eval_df)
    cfg_parent = _parent_cfg()
    print(f"[{MODEL_ID}] rebuilding v40.6 target-aware encoded frames", flush=True)
    x_train, y, training_meta = build_training_set(
        train_df,
        cfg=cfg_parent,
        stride_bars=int(args.train_stride),
        batch_size=512,
        feature_cols=feature_cols,
    )
    train_idx_sample = np.arange(
        0,
        max(0, len(train_df) - cfg_parent.max_train_horizon_bars - 1),
        max(1, int(args.train_stride)),
        dtype=np.int64,
    )
    if len(train_idx_sample) != len(x_train):
        raise RuntimeError(f"train sample mismatch: {len(train_idx_sample)} vs {len(x_train)}")
    proj_targets = _projection_targets(y)
    train_feat = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    val_feat = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_feat = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    _, val_full, eval_full, encoding_meta = _build_v40_6_frames(
        args=args,
        parent_report=parent_report,
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        train_feat=train_feat,
        val_feat=val_feat,
        eval_feat=eval_feat,
        train_idx_sample=train_idx_sample,
        proj_targets=proj_targets,
    )

    print(f"[{MODEL_ID}] predicting parent and frozen V27 scout utilities", flush=True)
    base = dict(parent_bundle.get("config", {}))
    fee = float(base.get("fee", cfg_parent.fee))
    slip = float(base.get("slip", cfg_parent.slip))
    val_dec = predict_policy_frame(parent_bundle, val_full, close=_close(val_full))
    eval_dec = predict_policy_frame(parent_bundle, eval_full, close=_close(eval_full))
    v27_payload, v27_model = v31._load_v27(args.v27_model)
    val_q = v31._predict_all(v27_model, val_full, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_full, v27_payload["seq_cols"], v27_payload["norm"])

    print(f"[{MODEL_ID}] selecting Deep Scout sleeve on 2025 Q4", flush=True)
    baseline_val = {
        f"cost{m}": backtest_nohold_deep(
            val_full,
            parent_bundle,
            val_q,
            None,
            fee=fee,
            slip=slip,
            cost_mult=float(m),
            decisions=val_dec,
            enable_deep=False,
        )
        for m in (1, 2, 3)
    }
    baseline_selection_score = _score(baseline_val["cost1"], baseline_val["cost2"], baseline_val["cost3"])
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for scout_cfg in _scout_grid():
        v1 = backtest_nohold_deep(val_full, parent_bundle, val_q, scout_cfg, fee=fee, slip=slip, cost_mult=1.0, decisions=val_dec)
        v2 = backtest_nohold_deep(val_full, parent_bundle, val_q, scout_cfg, fee=fee, slip=slip, cost_mult=2.0, decisions=val_dec)
        v3 = backtest_nohold_deep(val_full, parent_bundle, val_q, scout_cfg, fee=fee, slip=slip, cost_mult=3.0, decisions=val_dec)
        row = {
            "config": asdict(scout_cfg),
            "validation_cost1": v1,
            "validation_cost2": v2,
            "validation_cost3": v3,
            "selection_score": _score(v1, v2, v3),
        }
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    assert best is not None
    selected = v31.OverlayConfig(**best["config"])
    selected_beats_baseline = (
        float(best["selection_score"]) > float(baseline_selection_score)
        and float(best["validation_cost1"]["pnl"]) > float(baseline_val["cost1"]["pnl"])
        and float(best["validation_cost2"]["pnl"]) > 0.0
        and float(best["validation_cost3"]["pnl"]) > 0.0
    )

    print(f"[{MODEL_ID}] evaluating fixed 2026 OOS", flush=True)
    baseline_oos: dict[str, Any] = {}
    candidate_metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        baseline_oos[f"cost{mult}"] = backtest_nohold_deep(
            eval_full,
            parent_bundle,
            eval_q,
            None,
            fee=fee,
            slip=slip,
            cost_mult=float(mult),
            decisions=eval_dec,
            enable_deep=False,
            record=(mult == 1),
        )
        if mult == 1:
            ledger = pd.DataFrame(baseline_oos[f"cost{mult}"].pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_baseline_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["baseline_cost1"] = str(lp)
        r = backtest_nohold_deep(
            eval_full,
            parent_bundle,
            eval_q,
            selected,
            fee=fee,
            slip=slip,
            cost_mult=float(mult),
            decisions=eval_dec,
            enable_deep=True,
            record=(mult == 1),
        )
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_deep_cost1_ledger.csv")
            ledger.to_csv(lp, index=False)
            ledgers["deep_cost1"] = str(lp)
        candidate_metrics[f"cost{mult}"] = r
    oos_candidate_beats_baseline = (
        float(candidate_metrics["cost1"]["pnl"]) > float(baseline_oos["cost1"]["pnl"])
        and float(candidate_metrics["cost2"]["pnl"]) > 0.0
        and float(candidate_metrics["cost3"]["pnl"]) > 0.0
    )
    promoted_variant = "deep_scout" if selected_beats_baseline and oos_candidate_beats_baseline else "baseline_no_deep"
    metrics = candidate_metrics if promoted_variant == "deep_scout" else baseline_oos

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "v40_6_nohold_deep_scout_v42_manifest.json"
    manifest = {
        "model_id": MODEL_ID,
        "parent_model": str(args.parent_model),
        "parent_contract": "v40_6_no_maxhold_no_cooldown",
        "v27_model": str(args.v27_model),
        "selected_scout_config": asdict(selected),
        "deep_scout_only_when_parent_cash": True,
        "parent_effective_max_hold_bars": 0,
        "parent_effective_cooldown_bars": 0,
        "deep_effective_max_hold_bars": 0,
        "deep_effective_cooldown_bars": 0,
        "promoted_variant": promoted_variant,
        "selected_beats_validation_baseline": bool(selected_beats_baseline),
        "candidate_beats_oos_baseline": bool(oos_candidate_beats_baseline),
        "metrics": metrics,
        "candidate_metrics": candidate_metrics,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame(
        [
            {
                **{f"cfg_{k}": v for k, v in r["config"].items()},
                "score": r["selection_score"],
                "val_pnl": r["validation_cost1"]["pnl"],
                "val_mdd": r["validation_cost1"]["mdd"],
                "val_trades": r["validation_cost1"]["trades"],
                "val_deep_entries": r["validation_cost1"].get("deep_entries", 0),
                "val_c2_pnl": r["validation_cost2"]["pnl"],
                "val_c3_pnl": r["validation_cost3"]["pnl"],
            }
            for r in rows
        ]
    ).sort_values("score", ascending=False).to_csv(args.grid_out, index=False)

    feature_audit_cols = [c for c in list(parent_bundle.get("feature_cols") or []) if not c.startswith("macro_factor_") and not c.startswith("micro_factor_")]
    feature_audit = _audit_contract(train_all, eval_df, feature_audit_cols)
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit.get("status") != "pass":
        blocking.extend(feature_audit.get("blocking", []))
    warnings.extend(feature_audit.get("warnings", []))
    if candidate_metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if candidate_metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    if candidate_metrics["cost1"]["pnl"] <= baseline_oos["cost1"]["pnl"]:
        warnings.append("deep_scout_did_not_beat_nohold_baseline_cost1")
    if not oos_candidate_beats_baseline:
        warnings.append("deep_scout_rejected_by_oos_baseline_gate")
    if not selected_beats_baseline:
        warnings.append("deep_scout_rejected_by_validation_baseline_gate")
    if any("max_hold" in k for k in candidate_metrics["cost1"].get("exits", {})):
        blocking.append("effective_max_hold_exit_detected")
    if any("cooldown" in k for k in candidate_metrics["cost1"].get("actions", {})):
        blocking.append("effective_cooldown_action_detected")

    audit = {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after scout selection",
        "parent_contract": "v40_6_no_maxhold_no_cooldown",
        "parent_max_hold_disabled": True,
        "parent_cooldown_disabled": True,
        "deep_scout_max_hold_disabled": True,
        "deep_scout_cooldown_disabled": True,
        "deep_scout_only_when_parent_cash": True,
        "v27_entry_frozen": True,
        "feature_audit": feature_audit,
        "verdict": "promote" if selected_beats_baseline and oos_candidate_beats_baseline and not blocking else "reject",
        "baseline_selection_score": float(baseline_selection_score),
        "deep_selection_score": float(best["selection_score"]),
        "candidate_beats_oos_baseline": bool(oos_candidate_beats_baseline),
        "promoted_variant": promoted_variant,
        "selected_config": asdict(selected),
        "baseline_oos": baseline_oos,
        "candidate_metrics": candidate_metrics,
        "metrics": metrics,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Attach frozen V27 Deep Scout as a residual CASH-only sleeve to the v40.6 target-aware fully learned governor. The main no-max-hold/no-cooldown execution contract is preserved for both parent and scout positions; exits are TP/SL/trailing only, with no time-based max-hold and no model cooldown.",
        "parent_model": str(args.parent_model),
        "parent_report": str(args.parent_report),
        "v27_model": str(args.v27_model),
        "encoding_meta": encoding_meta,
        "training_meta": training_meta,
        "baseline_validation": baseline_val,
        "baseline_selection_score": float(baseline_selection_score),
        "selection_result": best,
        "selected_config": asdict(selected),
        "selected_beats_baseline": bool(selected_beats_baseline),
        "candidate_beats_oos_baseline": bool(oos_candidate_beats_baseline),
        "promoted_variant": promoted_variant,
        "baseline_oos": baseline_oos,
        "candidate_metrics": candidate_metrics,
        "metrics": metrics,
        "audit": audit,
        "artifacts": {
            "manifest": str(manifest_path),
            "report": str(args.report_out),
            "audit": str(args.audit_out),
            "grid": str(args.grid_out),
            "ledgers": ledgers,
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "audit": str(args.audit_out),
                "grid": str(args.grid_out),
                "manifest": str(manifest_path),
                "selected_config": asdict(selected),
                "selected_beats_baseline": bool(selected_beats_baseline),
                "candidate_beats_oos_baseline": bool(oos_candidate_beats_baseline),
                "promoted_variant": promoted_variant,
                "baseline_oos": baseline_oos,
                "candidate_metrics": candidate_metrics,
                "metrics": metrics,
                "verdict": audit["verdict"],
                "audit_status": audit["status"],
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
