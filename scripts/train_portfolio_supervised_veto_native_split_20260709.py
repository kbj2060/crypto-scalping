#!/usr/bin/env python3
"""Native 2-action supervised portfolio veto gate.

This is a smaller follow-up to the 3-asset ranker:
- keep the existing rule top-candidate selection unchanged;
- learn only TAKE_TOP vs SKIP_TOP from 2024 native counterfactual outcomes;
- select a conservative threshold on 2025-01..08 monthly calibration stability;
- evaluate 2025-09..12 final validation and 2026 OOS once.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

import train_portfolio_supervised_ranker_native_split_20260709 as split

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "portfolio_supervised_veto_native_split_20260709"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
DOC_PATH = ROOT / f"docs/model_contracts/{MODEL_ID}.md"
AUDIT_PATH = ROOT / f"docs/audits/{MODEL_ID}_redteam.md"
ASSETS = ("eth", "sol", "btc")

FEATURE_COLS = split.FEATURE_COLS + [
    "rule_rank_score",
    "tp_sl_ratio",
    "tp_minus_sl",
]


def _json_default(obj: Any) -> Any:
    return split._json_default(obj)


def _top_candidate(world: dict[str, Any], ts: pd.Timestamp) -> split.native.Candidate | None:
    candidates = [split.native._candidate_for_asset(world, asset, ts) for asset in ASSETS]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda c: (split.native.ASSET_SCORES[c.asset], c.notional), reverse=True)
    return candidates[0]


def _features(world: dict[str, Any], c: split.native.Candidate, ts: pd.Timestamp) -> dict[str, float]:
    row = split._features(world, c, ts)
    row["rule_rank_score"] = float(split.native.ASSET_SCORES[c.asset] + 0.01 * c.notional)
    row["tp_sl_ratio"] = float(c.take_profit / max(abs(c.stop_loss), 1.0e-8))
    row["tp_minus_sl"] = float(c.take_profit - abs(c.stop_loss))
    return row


def _utility(closed: dict[str, Any]) -> float:
    ret = float(closed["trade_return"])
    mae = float(closed.get("mae_price_move", 0.0) or 0.0)
    mfe = float(closed.get("mfe_price_move", 0.0) or 0.0)
    hold_bars = max(int(closed["exit_i"]) - int(closed["entry_i"]), 0)
    stop_penalty = 0.08 if str(closed.get("reason")) == "stop_loss" else 0.0
    adverse_penalty = 0.55 * max(0.0, -mae - 0.012)
    no_followthrough_penalty = 0.03 if ret < 0.0 and mfe < 0.006 else 0.0
    hold_penalty = 0.00002 * hold_bars
    return float(ret - adverse_penalty - stop_penalty - no_followthrough_penalty - hold_penalty)


def _build_dataset(world: dict[str, Any], device: torch.device) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    position: split.native.Position | None = None
    cash = 1.0
    candidate_count = 0
    for ts in world["timestamps"]:
        if position is not None:
            position, cash, _closed, _mark = split.native._try_close(world, position, ts, cash, device)
            continue
        c = _top_candidate(world, ts)
        if c is None:
            continue
        if candidate_count % 50 == 0:
            print(f"stage=build_veto_dataset idx={candidate_count}", flush=True)
        candidate_count += 1
        closed = split._simulate_candidate(world, c, device)
        rows.append(
            {
                "timestamp": ts,
                "asset": c.asset,
                "component": c.component,
                "label": _utility(closed),
                "trade_return": float(closed["trade_return"]),
                "mae_price_move": float(closed.get("mae_price_move", 0.0) or 0.0),
                "mfe_price_move": float(closed.get("mfe_price_move", 0.0) or 0.0),
                "hold_bars": int(closed["exit_i"]) - int(closed["entry_i"]),
                "reason": str(closed.get("reason")),
                **_features(world, c, ts),
            }
        )
        position, cash = split.native._open_position(world, c, cash)
    return pd.DataFrame(rows)


def _train_model(train_df: pd.DataFrame) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=80,
        learning_rate=0.04,
        num_leaves=5,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.4,
        reg_lambda=4.0,
        random_state=60709,
        verbose=-1,
    )
    model.fit(train_df[FEATURE_COLS], train_df["label"])
    return model


def _slice_world(world: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out: dict[str, Any] = {}
    for asset in ASSETS:
        aw = world[asset]
        frame = aw["frame"]
        idx = np.flatnonzero((frame["timestamp"] >= start_ts) & (frame["timestamp"] <= end_ts))
        sliced = frame.iloc[idx].reset_index(drop=True)
        out[asset] = {
            "frame": sliced,
            "components": {name: split._slice_component(comp, idx) for name, comp in aw["components"].items()},
            "fee_slip": aw["fee_slip"],
            "arrays": split.native._arrays(sliced),
            "ts_to_i": {pd.Timestamp(ts): i for i, ts in enumerate(sliced["timestamp"])},
        }
    common = set(out["eth"]["ts_to_i"]).intersection(out["sol"]["ts_to_i"]).intersection(out["btc"]["ts_to_i"])
    out["timestamps"] = sorted(common)
    return out


def _replay_veto(world: dict[str, Any], model: lgb.LGBMRegressor, *, threshold: float, device: torch.device) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    position: split.native.Position | None = None
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for ts in world["timestamps"]:
        if position is not None:
            position, cash, closed, mark_equity = split.native._try_close(world, position, ts, cash, device)
            peak = max(peak, mark_equity)
            mdd = min(mdd, mark_equity / max(peak, 1e-12) - 1.0)
            if closed is not None:
                rows.append(closed)
                peak = max(peak, cash)
                mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
            continue
        c = _top_candidate(world, ts)
        if c is None:
            continue
        feat = pd.DataFrame([_features(world, c, ts)])
        score = float(model.predict(feat[FEATURE_COLS])[0])
        take = score >= float(threshold)
        decisions.append({"timestamp": ts, "top_asset": c.asset, "top_component": c.component, "score": score, "threshold": float(threshold), "action": "take" if take else "skip"})
        if not take:
            continue
        position, cash = split.native._open_position(world, c, cash)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    if position is not None:
        cash, closed = split.native._force_close(world, position, cash)
        rows.append(closed)
    ledger = pd.DataFrame(rows)
    metrics = split.native._compound_metrics(ledger)
    metrics["mark_to_market_mdd"] = float(mdd * 100.0)
    metrics["decisions"] = int(len(decisions))
    metrics["skip_decisions"] = int(sum(d["action"] == "skip" for d in decisions))
    metrics["final_cash"] = float(cash)
    return metrics, ledger, pd.DataFrame(decisions)


def _calibration_score(month_rows: list[dict[str, Any]]) -> float:
    pnls = np.asarray([r["metrics"]["pnl"] for r in month_rows], dtype=np.float64)
    mdds = np.asarray([r["metrics"]["mdd"] for r in month_rows], dtype=np.float64)
    trades = np.asarray([r["metrics"]["trades"] for r in month_rows], dtype=np.float64)
    if np.any(trades < 2):
        return -np.inf
    return float(np.median(pnls) - 0.45 * abs(np.min(mdds)) - 0.35 * np.std(pnls))


def _write_doc(report: dict[str, Any]) -> None:
    lines = [
        "# Portfolio Supervised Veto Native Split - 2026-07-09",
        "",
        "2-action LightGBM veto gate. The rule top candidate is unchanged; the model only chooses TAKE_TOP or SKIP_TOP.",
        "",
        f"Selected threshold: `{report['selected_threshold']}`",
        "",
        "| split | PnL | MDD | MTM MDD | trades | WR | decisions | skips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("train_2024", "calibration_2025_01_08", "final_validation_2025_09_12", "oos_2026"):
        m = report["results"][name]
        lines.append(f"| {name} | {m['pnl']:.2f}% | {m['mdd']:.2f}% | {m['mark_to_market_mdd']:.2f}% | {m['trades']} | {m['wr']:.2%} | {m['decisions']} | {m['skip_decisions']} |")
    lines.extend(["", f"Promotion verdict: `promotion_pass={str(report['promotion_pass']).lower()}`.", ""])
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def _write_audit(report: dict[str, Any]) -> None:
    final = report["results"]["final_validation_2025_09_12"]
    oos = report["results"]["oos_2026"]
    lines = [
        "# Portfolio Supervised Veto Native Split Redteam - 2026-07-09",
        "",
        f"`promotion_pass={str(report['promotion_pass']).lower()}`.",
        "",
        f"- Final validation: PnL `{final['pnl']:.2f}%`, MDD `{final['mdd']:.2f}%`, trades `{final['trades']}`.",
        f"- OOS: PnL `{oos['pnl']:.2f}%`, MDD `{oos['mdd']:.2f}%`, trades `{oos['trades']}`.",
        "- P0 leakage finding: none found. Replay is native bar-by-bar and does not use saved trade ledgers or saved exit timestamps as inputs.",
        "- P1 caveat: 2024 parent predictions are generated by scoring frozen bundles on 2024 features because existing train_predictions artifacts cover 2025 Jan-Sep, not 2024.",
        "- P2 caveat: SOL/BTC 2024 use the timestamp-only 2024 regime3_current wide24 overlay, with missing rows dropped and reported in diagnostics.",
        "",
    ]
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = split.eth_retest.DEVICE

    print("stage=build_world train_2024", flush=True)
    train_world = split._build_world("train_2024", "2024-01-01", "2024-12-31 23:59:59", device)
    print("stage=build_world calibration_2025", flush=True)
    calibration_world = split._build_world("validation_2025", "2025-01-01", "2025-08-31 23:59:59", device)
    print("stage=build_world final_validation_2025", flush=True)
    final_world = split._build_world("validation_2025", "2025-09-01", "2025-12-31 23:59:59", device)
    print("stage=build_world oos_2026", flush=True)
    oos_world = split._build_world("oos", "2026-01-01", "2026-06-30 23:59:59", device)

    print("stage=build_train_dataset", flush=True)
    train_df = _build_dataset(train_world, device)
    if len(train_df) < 20:
        raise RuntimeError(f"training dataset too thin: rows={len(train_df)}")
    train_df.to_csv(OUT_DIR / "train_2024_veto_training_set.csv", index=False)
    print(f"stage=train_model rows={len(train_df)}", flush=True)
    model = _train_model(train_df)
    with open(OUT_DIR / "veto_lgbm.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS}, f)

    threshold_grid = [-0.12, -0.08, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02]
    month_specs = [
        ("2025_01", "2025-01-01", "2025-01-31 23:59:59"),
        ("2025_02", "2025-02-01", "2025-02-28 23:59:59"),
        ("2025_03", "2025-03-01", "2025-03-31 23:59:59"),
        ("2025_04", "2025-04-01", "2025-04-30 23:59:59"),
        ("2025_05", "2025-05-01", "2025-05-31 23:59:59"),
        ("2025_06", "2025-06-01", "2025-06-30 23:59:59"),
        ("2025_07", "2025-07-01", "2025-07-31 23:59:59"),
        ("2025_08", "2025-08-01", "2025-08-31 23:59:59"),
    ]
    month_worlds = {name: _slice_world(calibration_world, start, end) for name, start, end in month_specs}
    grid_rows: list[dict[str, Any]] = []
    best_threshold = threshold_grid[0]
    best_score = -np.inf
    for th in threshold_grid:
        month_rows: list[dict[str, Any]] = []
        for name, mw in month_worlds.items():
            metrics, _ledger, _decisions = _replay_veto(mw, model, threshold=float(th), device=device)
            month_rows.append({"month": name, "metrics": metrics})
        score = _calibration_score(month_rows)
        eligible = bool(np.isfinite(score))
        grid_rows.append({"threshold": float(th), "eligible": eligible, "score": float(score), "months": month_rows})
        if eligible and score > best_score:
            best_score = float(score)
            best_threshold = float(th)
    pd.DataFrame(grid_rows).to_json(OUT_DIR / "monthly_calibration_threshold_grid.jsonl", orient="records", lines=True, force_ascii=False)

    results: dict[str, Any] = {}
    for name, world in (
        ("train_2024", train_world),
        ("calibration_2025_01_08", calibration_world),
        ("final_validation_2025_09_12", final_world),
        ("oos_2026", oos_world),
    ):
        print(f"stage=replay name={name} threshold={best_threshold}", flush=True)
        metrics, ledger, decisions = _replay_veto(world, model, threshold=best_threshold, device=device)
        results[name] = metrics
        ledger.to_csv(OUT_DIR / f"{name}_ledger.csv", index=False)
        decisions.to_csv(OUT_DIR / f"{name}_decisions.csv", index=False)

    final = results["final_validation_2025_09_12"]
    promotion_pass = bool(final["pnl"] > 0.0 and final["mdd"] >= -25.0 and final["trades"] >= 12)
    report = {
        "method": "portfolio_supervised_veto_native_lgbm_time_split",
        "model_id": MODEL_ID,
        "action_space": {"0": "SKIP_TOP", "1": "TAKE_TOP"},
        "split_contract": {
            "train": "2024-01-01..2024-12-31",
            "calibration_aux_validation": "2025-01-01..2025-08-31 monthly threshold stability",
            "final_validation": "2025-09-01..2025-12-31",
            "oos": "2026-01-01..2026-06-30",
        },
        "training_rows": int(len(train_df)),
        "feature_cols": FEATURE_COLS,
        "selected_threshold": float(best_threshold),
        "threshold_grid": grid_rows,
        "results": results,
        "diagnostics": split.DIAGNOSTICS,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "promotion_pass": promotion_pass,
        "promotion_grade": False,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_doc(report)
    _write_audit(report)
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "doc": str(DOC_PATH), "audit": str(AUDIT_PATH), "selected_threshold": best_threshold, "results": results, "promotion_pass": promotion_pass}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
