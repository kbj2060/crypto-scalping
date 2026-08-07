#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.certified_teacher_regime_moe import CLEAN_PREFIX, append_clean_regime, fit_clean_regime_predictor, load_csv, merge_teacher_sources  # noqa: E402
from ensemble.certified_teacher_trend_convex_sleeve import (  # noqa: E402
    CONTRACTS,
    MODEL_ID,
    Position,
    append_event_scores,
    build_event_candidates,
    feature_cols,
    fit_scorer,
    model_cols,
    predict_scorer,
    replay_convex,
    save_bundle,
    small_grid,
)
from pipeline.certified_feature_audit import audit_ai_contracts, audit_frame_contract  # noqa: E402
from pipeline.teacher_meta_side_features import append_side_teacher_features  # noqa: E402


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--state-2024", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--base-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--ai-2025", type=Path, default=ROOT / "data/tmp/unified_build_ckpt/03_after_ai.csv")
    p.add_argument("--m7-2025", type=Path, default=ROOT / "data/splits/year_oos/rl_training_2025_m7.csv")
    p.add_argument("--base-2026", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--ai-2026", type=Path, default=ROOT / "data/tmp/unified_build_ckpt_2026/03_after_ai.csv")
    p.add_argument("--m7-2026", type=Path, default=ROOT / "data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/ensemble/supervised/certified_teacher_trend_convex_sleeve_v5")
    p.add_argument("--report-out", type=Path, default=ROOT / "data/ensemble/reports/certified_teacher_trend_convex_sleeve_v5_summary.json")
    p.add_argument("--audit-out", type=Path, default=ROOT / "data/ensemble/reports/certified_teacher_trend_convex_sleeve_v5_audit.json")
    p.add_argument("--contract-out", type=Path, default=ROOT / "docs/model_contracts/certified_teacher_trend_convex_sleeve_v5_contract.md")
    p.add_argument("--max-features", type=int, default=112)
    p.add_argument("--label-stride", type=int, default=24)
    p.add_argument("--max-train-rows", type=int, default=100000)
    p.add_argument("--seed-score", type=float, default=0.42)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    return p.parse_args()


def _build(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    regime = fit_clean_regime_predictor(load_csv(args.state_2024))
    y2025 = append_event_scores(append_side_teacher_features(append_clean_regime(merge_teacher_sources(load_csv(args.base_2025), load_csv(args.ai_2025), load_csv(args.m7_2025)), regime)))
    y2026 = append_event_scores(append_side_teacher_features(append_clean_regime(merge_teacher_sources(load_csv(args.base_2026), load_csv(args.ai_2026), load_csv(args.m7_2026)), regime)))
    return y2025, y2026, regime


def _raw(side: int, entry: float, price: float) -> float:
    if entry <= 0.0 or price <= 0.0:
        return 0.0
    return float(side) * (float(price) / float(entry) - 1.0)


def _row(frame: pd.DataFrame, pos: Position, exit_idx: int, exit_price: float, realized: float, gross: float, cost: float, equity: float, reason: str, trade_id: int) -> dict[str, Any]:
    return {
        "trade_id": int(trade_id),
        "timestamp": str(frame.iloc[pos.signal_idx]["timestamp"]),
        "entry_time": str(frame.iloc[pos.entry_idx]["timestamp"]),
        "exit_time": str(frame.iloc[exit_idx]["timestamp"]),
        "entry_idx": int(pos.entry_idx),
        "exit_idx": int(exit_idx),
        "side": "LONG" if pos.side > 0 else "SHORT",
        "contract_family": pos.family,
        "action": "trade",
        "sleeve": MODEL_ID,
        "entry_price": float(pos.entry_price),
        "exit_price": float(exit_price),
        "notional": float(pos.notional),
        "leverage": float(pos.leverage),
        "margin_fraction": float(pos.notional / max(pos.leverage, 1e-12)),
        "expected_net_pct": float(pos.expected_pct),
        "pred_adverse_pct": float(pos.adverse_pct),
        "event_score": float(pos.event_score),
        "realized_raw": float(realized),
        "exit_fee_cash": float(cost),
        "trade_pnl_pct": float((gross - cost) * 100.0),
        "cash_after": float(equity),
        "blocked": False,
        "stop_reason": reason,
    }


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "ledger"}


def _overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
    ta = pd.to_datetime(a["timestamp"], errors="coerce").dropna().astype("int64")
    tb = pd.to_datetime(b["timestamp"], errors="coerce").dropna().astype("int64")
    return int(len(set(ta.tolist()) & set(tb.tolist())))


def _predict(frame: pd.DataFrame, cols: list[str], long_model: dict[str, Any], short_model: dict[str, Any], *, fee: float, slip: float, seed_score: float) -> pd.DataFrame:
    long = predict_scorer(long_model, build_event_candidates(frame, cols, 1, fee=fee, slip=slip, label=False, min_seed_score=seed_score))
    short = predict_scorer(short_model, build_event_candidates(frame, cols, -1, fee=fee, slip=slip, label=False, min_seed_score=seed_score))
    if long.empty and short.empty:
        return pd.DataFrame()
    long["side"] = 1
    short["side"] = -1
    return pd.concat([long, short], ignore_index=True)


def backtest(frame: pd.DataFrame, pred: pd.DataFrame, cfg: Any, *, fee: float, slip: float) -> dict[str, Any]:
    by_idx = {int(k): g for k, g in pred.groupby("_idx", sort=False)} if not pred.empty else {}
    equity = 1.0
    peak = 1.0
    min_eq = 1.0
    pos: Position | None = None
    ledger: list[dict[str, Any]] = []
    blocks: dict[str, int] = {}
    day_key = ""
    day_entries = 0
    last_entry = -100000
    for i in range(0, len(frame) - 1):
        ts = pd.Timestamp(frame.iloc[i]["timestamp"])
        key = ts.date().isoformat()
        if key != day_key:
            day_key, day_entries = key, 0
        next_open = float(frame.iloc[i + 1]["open"])
        if pos is not None:
            contract = CONTRACTS[pos.family]
            raw = _raw(pos.side, pos.entry_price, float(frame.iloc[i]["close"]))
            pos.peak_raw = max(pos.peak_raw, raw)
            mark = equity * max(0.0, 1.0 + pos.notional * raw)
            peak = max(peak, mark)
            min_eq = min(min_eq, mark)
            reason = ""
            if raw <= -float(contract["stop_loss"]):
                reason = "stop_loss"
            elif raw >= float(contract["take_profit"]):
                reason = "loose_take_profit"
            elif pos.peak_raw >= float(contract["trail_activate"]) and raw <= pos.peak_raw - float(contract["trailing_stop"]):
                reason = "convex_trailing_stop"
            elif i - pos.entry_idx >= int(contract["max_hold_bars"]):
                reason = "max_hold"
            if reason:
                realized = _raw(pos.side, pos.entry_price, next_open)
                cost = pos.notional * (fee + slip)
                gross = pos.notional * realized
                equity *= max(0.0, 1.0 + gross - cost)
                peak = max(peak, equity)
                min_eq = min(min_eq, equity)
                ledger.append(_row(frame, pos, i + 1, next_open, realized, gross, cost, equity, reason, len(ledger)))
                pos = None
                continue
        if pos is not None:
            continue
        if i - last_entry < cfg.min_gap_bars:
            blocks["min_gap"] = blocks.get("min_gap", 0) + 1
            continue
        if day_entries >= cfg.top_k_per_day:
            blocks["daily_budget"] = blocks.get("daily_budget", 0) + 1
            continue
        if i not in by_idx:
            continue
        choices = by_idx[i]
        choices = choices[
            (pd.to_numeric(choices["cand_event_score"], errors="coerce") >= cfg.min_event_score)
            & (pd.to_numeric(choices["pred_edge_pct"], errors="coerce") >= cfg.min_pred_edge_pct)
            & (pd.to_numeric(choices["pred_adverse_pct"], errors="coerce") <= cfg.max_pred_adverse_pct)
        ]
        if choices.empty:
            blocks["model_gate"] = blocks.get("model_gate", 0) + 1
            continue
        best = choices.sort_values("rank_score", ascending=False).iloc[0]
        edge = float(best["pred_edge_pct"])
        adverse = float(best["pred_adverse_pct"])
        event_score = float(best["cand_event_score"])
        side = int(best["side"])
        edge_scale = np.clip((edge - cfg.min_pred_edge_pct + 0.30) / 1.20, 0.10, 1.20)
        adv_scale = np.clip(1.0 - adverse / max(cfg.max_pred_adverse_pct, 1e-12), 0.20, 1.0)
        event_scale = np.clip((event_score - cfg.min_event_score + 0.25) / 0.55, 0.20, 1.15)
        notional = float(np.clip(cfg.min_notional + (cfg.max_notional - cfg.min_notional) * edge_scale * adv_scale * event_scale, cfg.min_notional, cfg.max_notional))
        equity *= max(0.0, 1.0 - notional * (fee + slip))
        min_eq = min(min_eq, equity)
        pos = Position(side, str(best["cand_family"]), i, i + 1, next_open, notional, cfg.leverage, edge, adverse, event_score)
        day_entries += 1
        last_entry = i
    if pos is not None:
        i = len(frame) - 1
        px = float(frame.iloc[i]["close"])
        realized = _raw(pos.side, pos.entry_price, px)
        cost = pos.notional * (fee + slip)
        gross = pos.notional * realized
        equity *= max(0.0, 1.0 + gross - cost)
        min_eq = min(min_eq, equity)
        ledger.append(_row(frame, pos, i, px, realized, gross, cost, equity, "end", len(ledger)))
    ledger.append({"trade_id": -1, "timestamp": str(frame.iloc[-1]["timestamp"]), "action": "coverage_end", "side": "COVERAGE", "cash_after": float(equity), "stop_reason": "coverage_end"})
    trades = [r for r in ledger if r.get("action") == "trade"]
    days = max((pd.Timestamp(frame.iloc[-1]["timestamp"]) - pd.Timestamp(frame.iloc[0]["timestamp"])).total_seconds() / 86400.0, 1e-12)
    wins = sum(float(r["trade_pnl_pct"]) > 0.0 for r in trades)
    return {
        "pnl": float((equity - 1.0) * 100.0),
        "mdd": float((min_eq / max(peak, 1e-12) - 1.0) * 100.0),
        "trades": int(len(trades)),
        "trades_per_day": float(len(trades) / days),
        "wr": float(wins / len(trades)) if trades else 0.0,
        "avg_notional": float(np.mean([float(r["notional"]) for r in trades])) if trades else 0.0,
        "max_margin_fraction": float(np.max([float(r["margin_fraction"]) for r in trades])) if trades else 0.0,
        "final_equity": float(equity),
        "coverage_start": str(frame.iloc[0]["timestamp"]),
        "coverage_end": str(frame.iloc[-1]["timestamp"]),
        "block_reason_counts": blocks,
        "ledger": ledger,
    }


def _score(r: dict[str, Any], r2: dict[str, Any]) -> float:
    tpd = float(r["trades_per_day"])
    if tpd < 0.6 or int(r["trades"]) < 20:
        return -1e9 + float(r["pnl"]) - 1000.0 * max(0.0, 0.6 - tpd)
    return float(float(r["pnl"]) - 1.0 * abs(float(r["mdd"])) - 0.45 * max(0.0, float(r["pnl"]) - float(r2["pnl"])) + 0.25 * min(tpd, 4.0))


def _write_contract(path: Path, report: dict[str, Any], audit: dict[str, Any]) -> None:
    c1 = report["metrics"]["cost1"]
    lines = [
        "# Certified Teacher Trend Convex Sleeve V5",
        "",
        f"- Model ID: `{MODEL_ID}`",
        "- Candidate universe: rare high-conviction trend events, not every bar.",
        "- Inputs: certified AI meta, M7 meta, clean_regime_2024_unsup_v4, and causal market/microstructure features.",
        "- Execution: next-bar open entry, convex trend contracts with loose TP and trailing exits.",
        "- Selection: 2025 selection only. 2026 is fixed OOS and not used for config selection.",
        f"- Audit: `{audit['status']}`",
        f"- Blocking: `{audit['blocking']}`",
        "",
        "## Selected Config",
        f"- Config: `{report['selected_config']}`",
        "",
        "## OOS Cost1",
        f"- PnL: `{c1['pnl']}`",
        f"- MDD: `{c1['mdd']}`",
        f"- Trades/day: `{c1['trades_per_day']}`",
        f"- Avg notional: `{c1['avg_notional']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ai_audit = audit_ai_contracts()
    y2025, y2026, regime = _build(args)
    y2025.to_csv(args.out_dir / "features_2025.csv", index=False)
    y2026.to_csv(args.out_dir / "features_2026.csv", index=False)
    fit = y2025[y2025["timestamp"] < pd.Timestamp("2025-09-01")].copy()
    selection = y2025[(y2025["timestamp"] >= pd.Timestamp("2025-09-01")) & (y2025["timestamp"] < pd.Timestamp("2025-11-01"))].copy()
    holdout = y2025[y2025["timestamp"] >= pd.Timestamp("2025-11-01")].copy()
    cols = feature_cols([fit, selection, holdout, y2026], CLEAN_PREFIX)
    priority = [c for c in cols if c.startswith(("event_", "teacher_", CLEAN_PREFIX, "m7_", "ai_", "patchtst_", "tide_", "timesnet_", "dlinear_")) or c in {"pred_patchtst", "conf_patchtst", "breakout_strength", "mtf_trend_1h", "mtf_trend_4h"}]
    cols = (priority + [c for c in cols if c not in priority])[: args.max_features]
    fa = audit_frame_contract(y2025, feature_cols=cols, clean_prefix=CLEAN_PREFIX)
    if fa["status"] != "pass":
        raise ValueError(json.dumps(fa, ensure_ascii=False))
    print(f"[{MODEL_ID}] event labels", flush=True)
    long_train = build_event_candidates(fit, cols, 1, fee=args.fee, slip=args.slip, label=True, min_seed_score=args.seed_score, row_stride=args.label_stride)
    short_train = build_event_candidates(fit, cols, -1, fee=args.fee, slip=args.slip, label=True, min_seed_score=args.seed_score, row_stride=args.label_stride)
    if long_train.empty or short_train.empty:
        raise RuntimeError("empty event training candidates")
    long_model = fit_scorer(long_train, model_cols(long_train), seed=5101, max_rows=args.max_train_rows)
    short_model = fit_scorer(short_train, model_cols(short_train), seed=6101, max_rows=args.max_train_rows)
    sel_pred = _predict(selection, cols, long_model, short_model, fee=args.fee, slip=args.slip, seed_score=args.seed_score)
    hold_pred = _predict(holdout, cols, long_model, short_model, fee=args.fee, slip=args.slip, seed_score=args.seed_score)
    oos_pred = _predict(y2026, cols, long_model, short_model, fee=args.fee, slip=args.slip, seed_score=args.seed_score)
    rows = []
    best = None
    best_score = -1e18
    best_sel = None
    for idx, cfg in enumerate(small_grid(), start=1):
        print(f"[{MODEL_ID}] grid {idx}/{len(small_grid())}", flush=True)
        r = backtest(selection, sel_pred, cfg, fee=args.fee, slip=args.slip)
        r2 = backtest(selection, sel_pred, cfg, fee=args.fee * 2, slip=args.slip * 2)
        s = _score(r, r2)
        rows.append({"score": s, **asdict(cfg), **{f"selection_{k}": v for k, v in _compact(r).items()}, "selection_cost2_pnl": r2["pnl"]})
        if s > best_score:
            best, best_score, best_sel = cfg, s, r
    if best is None:
        raise RuntimeError("no selected config")
    hold = backtest(holdout, hold_pred, best, fee=args.fee, slip=args.slip)
    metrics = {}
    ledgers = {}
    for mult in (1, 2, 3):
        r = backtest(y2026, oos_pred, best, fee=args.fee * mult, slip=args.slip * mult)
        k = f"cost{mult}"
        metrics[k] = _compact(r)
        lp = args.report_out.with_name(args.report_out.stem + f"_{k}_ledger.csv")
        pd.DataFrame(r["ledger"]).to_csv(lp, index=False)
        ledgers[k] = str(lp)
    model_path = args.out_dir / "model.pkl"
    save_bundle(model_path, {"model_id": MODEL_ID, "regime": regime, "long_model": long_model, "short_model": short_model, "feature_cols": cols, "selected_config": asdict(best), "seed_score": args.seed_score})
    grid_path = args.report_out.with_name(args.report_out.stem + "_selection_grid.csv")
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(grid_path, index=False)
    report = {
        "model_id": MODEL_ID,
        "data": {
            "fit_range": [str(fit["timestamp"].iloc[0]), str(fit["timestamp"].iloc[-1])],
            "selection_range": [str(selection["timestamp"].iloc[0]), str(selection["timestamp"].iloc[-1])],
            "holdout_range": [str(holdout["timestamp"].iloc[0]), str(holdout["timestamp"].iloc[-1])],
            "oos_range": [str(y2026["timestamp"].iloc[0]), str(y2026["timestamp"].iloc[-1])],
        },
        "event_candidate_counts": {"fit_long": len(long_train), "fit_short": len(short_train), "selection": len(sel_pred), "holdout": len(hold_pred), "oos": len(oos_pred)},
        "feature_count": len(cols),
        "selected_config": asdict(best),
        "selection_score": best_score,
        "selection_result": _compact(best_sel),
        "holdout_result": _compact(hold),
        "metrics": metrics,
        "artifacts": {"model": str(model_path), "selection_grid": str(grid_path), "ledgers": ledgers},
        "data_audit": {"ai_contracts": ai_audit, "feature_contract": fa, "train_eval_overlap": _overlap(fit, y2026) + _overlap(selection, y2026) + _overlap(holdout, y2026)},
    }
    blocking = []
    if ai_audit["status"] != "pass":
        blocking += ai_audit["blocking"]
    if fa["status"] != "pass":
        blocking += fa["blocking"]
    if report["data_audit"]["train_eval_overlap"] != 0:
        blocking.append("train_eval_timestamp_overlap")
    audit = {
        "model_id": MODEL_ID,
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": ["M7 embedded provenance caveat remains"],
        "invariants": {
            "rare_event_universe": True,
            "legacy_regime_v2_quarantined": True,
            "2026_fixed_oos_no_selection": True,
            "cost_1x_2x_3x_reported": True,
            "next_bar_open_execution": True,
        },
    }
    report["audit"] = audit
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    _write_contract(args.contract_out, report, audit)
    print(json.dumps({"status": audit["status"], "metrics": metrics, "report": str(args.report_out), "audit": str(args.audit_out)}, indent=2, ensure_ascii=False, default=_json_default))
    return 0 if audit["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
