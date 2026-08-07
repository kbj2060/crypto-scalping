#!/usr/bin/env python3
"""Validation-only parameter search for the causal online portfolio bandit gate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import replay_portfolio_online_bandit_gate_native_20260709 as bandit
import replay_portfolio_rl_gate_2action_native_20260708 as native

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/causal_regen_20260516/portfolio_online_bandit_gate_param_search_20260709"
DOC_PATH = ROOT / "docs/model_contracts/portfolio_online_bandit_gate_param_search_20260709.md"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


class TunedBandit(bandit.OnlineBandit):
    def __init__(self, *, min_samples: int, l2: float, skip_margin: float, tail_penalty_coef: float, notional_penalty_coef: float) -> None:
        super().__init__(min_samples=min_samples, l2=l2, skip_margin=skip_margin)
        self.tail_penalty_coef = float(tail_penalty_coef)
        self.notional_penalty_coef = float(notional_penalty_coef)

    def update(self, feat: pd.DataFrame, trade_return: float, notional: float) -> None:
        tail_penalty = max(0.0, -float(trade_return) - 0.03)
        reward = float(trade_return) - self.tail_penalty_coef * tail_penalty - self.notional_penalty_coef * float(notional)
        self.xs.append(bandit._x_take(feat)[0])
        self.ys.append(float(reward))
        if len(self.ys) >= self.min_samples:
            self.weights = bandit._fit_ridge(np.vstack(self.xs), np.asarray(self.ys, dtype=np.float64), self.l2)


def _replay_tuned(world: dict[str, Any], *, config: dict[str, Any], device: Any) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    learner = TunedBandit(**config)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    position: native.Position | None = None
    position_feat: pd.DataFrame | None = None
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    lag_returns = {"eth": [], "sol": [], "btc": []}
    for ts in world["timestamps"]:
        if position is not None:
            position, cash, closed, mark_equity = native._try_close(world, position, ts, cash, device)
            peak = max(peak, mark_equity)
            mdd = min(mdd, mark_equity / max(peak, 1e-12) - 1.0)
            if closed is not None:
                rows.append(closed)
                lag_returns[closed["asset"]].append(float(closed["trade_return"]))
                if position_feat is not None:
                    learner.update(position_feat, float(closed["trade_return"]), float(closed["notional"]))
                position_feat = None
                peak = max(peak, cash)
                mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
            continue
        top = bandit._select_top(world, ts)
        if top is None:
            continue
        feat = bandit._features(top, ts, world, lag_returns)
        take, pred = learner.take(feat)
        decisions.append({"timestamp": ts, "asset": top.asset, "action": "TAKE_TOP" if take else "SKIP", "pred_reward": float(pred), "closed_samples_before_decision": int(len(learner.ys))})
        if not take:
            continue
        position, cash = native._open_position(world, top, cash)
        position_feat = feat
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    if position is not None:
        cash, closed = native._force_close(world, position, cash)
        rows.append(closed)
    ledger = pd.DataFrame(rows)
    metrics = native._compound_metrics(ledger)
    metrics["mark_to_market_mdd"] = float(mdd * 100.0)
    metrics["decisions"] = int(len(decisions))
    metrics["skips"] = int(sum(d["action"] == "SKIP" for d in decisions))
    metrics["skip_rate"] = float(metrics["skips"] / max(metrics["decisions"], 1))
    metrics["closed_samples_final"] = int(len(learner.ys))
    return metrics, ledger, pd.DataFrame(decisions)


def _score(metrics: dict[str, Any]) -> float:
    # Validation-only objective: improve return, avoid high drawdown, and avoid
    # degenerate all-skip behavior.
    return float(metrics["pnl"]) - 0.35 * abs(float(metrics["mdd"])) - 20.0 * max(0.0, float(metrics["skip_rate"]) - 0.60)


def _write_doc(report: dict[str, Any]) -> None:
    sel = report["selected"]
    lines = [
        "# Portfolio Online Bandit Gate Param Search - 2026-07-09",
        "",
        "Validation-only search. OOS was evaluated once after config selection.",
        "",
        f"Selected config: `{sel['config']}`",
        "",
        "| split | PnL | MDD | MTM MDD | trades | WR | decisions | skips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "oos_extended", "oos_frozen_q1_2026"):
        m = report["results"][split]
        lines.append(f"| {split} | {m['pnl']:.2f}% | {m['mdd']:.2f}% | {m['mark_to_market_mdd']:.2f}% | {m['trades']} | {m['wr']:.2%} | {m.get('decisions', 0)} | {m.get('skips', 0)} |")
    lines.append("")
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = native.eth_retest.DEVICE
    print("stage=build_validation_world", flush=True)
    val_world = native._build_world("validation", device)
    grid: list[dict[str, Any]] = []
    for skip_margin in (-0.20, -0.15, -0.12, -0.10, -0.08, -0.05, -0.03, -0.015):
        grid.append(
            {
                "min_samples": 8,
                "l2": 50.0,
                "skip_margin": float(skip_margin),
                "tail_penalty_coef": 0.0,
                "notional_penalty_coef": 0.0,
            }
        )
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score = -np.inf
    for idx, cfg in enumerate(grid):
        if idx % 1 == 0:
            print(f"stage=validation_grid idx={idx}/{len(grid)}", flush=True)
        metrics, _ledger, _decisions = _replay_tuned(val_world, config=cfg, device=device)
        eligible = metrics["trades"] >= 20 and metrics["skip_rate"] <= 0.70 and metrics["mdd"] >= -25.0
        score = _score(metrics) if eligible else -np.inf
        row = {"idx": idx, "config": cfg, "validation": metrics, "eligible": bool(eligible), "score": float(score)}
        rows.append(row)
        if eligible and score > best_score:
            best = row
            best_score = float(score)
    if best is None:
        raise RuntimeError("no eligible validation config")
    pd.DataFrame(rows).to_json(OUT_DIR / "validation_grid.jsonl", orient="records", lines=True, force_ascii=False)

    print(f"stage=selected idx={best['idx']} config={best['config']}", flush=True)
    selected_cfg = best["config"]
    val_metrics, val_ledger, val_decisions = _replay_tuned(val_world, config=selected_cfg, device=device)
    print("stage=build_oos_world", flush=True)
    oos_world = native._build_world("oos", device)
    oos_metrics, oos_ledger, oos_decisions = _replay_tuned(oos_world, config=selected_cfg, device=device)
    q1 = oos_ledger.loc[pd.to_datetime(oos_ledger["entry_timestamp"]) < pd.Timestamp("2026-04-01")].reset_index(drop=True) if not oos_ledger.empty else oos_ledger
    q1_metrics = native._compound_metrics(q1)
    q1_metrics["mark_to_market_mdd"] = q1_metrics["mdd"]
    q1_metrics["decisions"] = int((oos_decisions["timestamp"] < pd.Timestamp("2026-04-01")).sum()) if not oos_decisions.empty else 0
    q1_metrics["skips"] = int(((oos_decisions["timestamp"] < pd.Timestamp("2026-04-01")) & (oos_decisions["action"] == "SKIP")).sum()) if not oos_decisions.empty else 0
    val_ledger.to_csv(OUT_DIR / "validation_ledger.csv", index=False)
    val_decisions.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    oos_ledger.to_csv(OUT_DIR / "oos_extended_ledger.csv", index=False)
    oos_decisions.to_csv(OUT_DIR / "oos_extended_decisions.csv", index=False)
    q1.to_csv(OUT_DIR / "oos_frozen_q1_2026_ledger.csv", index=False)
    report = {
        "method": "portfolio_online_bandit_gate_native_param_search",
        "selection_data": "validation_only",
        "oos_usage": "reported_once_after_config_selection",
        "selected": best,
        "results": {"validation": val_metrics, "oos_extended": oos_metrics, "oos_frozen_q1_2026": q1_metrics},
        "grid_count": len(rows),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "uses_only_past_closed_trades_for_learning": True,
        "promotion_grade": False,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_doc(report)
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "doc": str(DOC_PATH), "selected": best, "results": report["results"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
