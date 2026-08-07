#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dt_lifecycle_iql_cql_cvar.candidate import (  # noqa: E402
    CandidateConfig,
    EmpiricalDTLifecyclePolicy,
    backtest_candidate,
)
from scripts.run_lifecycle_manager_grid import DEFAULT_EVAL_CSV, DEFAULT_POLICY, DEFAULT_TRAIN_CSV  # noqa: E402


DEFAULT_ZERO_REPORT = ROOT / "data/ensemble/reports/zero_style_remaining_layers_walkforward_2026.json"
DEFAULT_REPORT = ROOT / "docs/experiments/dt_iql_cql_cvar_vs_zero_style_smoke.json"


def _read(path: Path, *, start: str | None, end: str | None, limit_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise KeyError(f"{path} missing timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end)]
    if limit_rows is not None and int(limit_rows) > 0:
        df = df.head(int(limit_rows))
    return df.reset_index(drop=True)


def _range(df: pd.DataFrame) -> dict[str, Any]:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    return {"rows": int(len(df)), "start": str(ts.min()), "end": str(ts.max())}


def _compact(bt: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "pnl",
        "mdd",
        "trades",
        "wr",
        "trades_per_day",
        "long_entries",
        "short_entries",
        "avg_notional",
        "avg_leverage",
        "gate_blocks",
        "lifecycle_exits",
        "scale_downs",
        "scale_ups",
    )
    return {k: bt.get(k) for k in keys if k in bt}


def _zero_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path)}
    obj = json.loads(path.read_text(encoding="utf-8"))
    decision = obj.get("decision", {})
    latest = obj.get("stage_reports", [])[-1] if obj.get("stage_reports") else {}
    eval_result = latest.get("eval_result", {}).get("eval", {})
    stress = {}
    for k, rows in obj.get("cost_stress", {}).items():
        if rows:
            stress[k] = rows[0].get("eval", {})
    return {
        "available": True,
        "path": str(path),
        "type": obj.get("type"),
        "decision": decision,
        "eval": _compact(eval_result),
        "cost_stress": {k: _compact(v) for k, v in stress.items()},
        "audit": obj.get("audit", {}),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare shadow DT lifecycle + IQL/CQL/CVaR candidate against existing MuZero/AZ report artifacts."
    )
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--entry-source", choices=["model", "heuristic"], default="model")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--zero-report", type=Path, default=DEFAULT_ZERO_REPORT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--smoke-rows", type=int, default=None)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--leverage-cap", type=float, default=3.60)
    p.add_argument("--max-notional", type=float, default=3.60)
    p.add_argument("--horizon-bars", type=int, default=144)
    p.add_argument("--window-bars", type=int, default=48)
    p.add_argument("--lb-quantile", type=float, default=0.20)
    p.add_argument("--min-support", type=int, default=20)
    p.add_argument("--min-lower-bound", type=float, default=-0.012)
    p.add_argument("--min-cvar", type=float, default=-0.020)
    p.add_argument("--iql-floor", type=float, default=-0.004)
    p.add_argument("--cql-penalty", type=float, default=0.010)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    entry_bundle = {"model_type": "heuristic_csv_policy_v0"} if args.entry_source == "heuristic" else joblib.load(args.policy)
    train_df = _read(args.train_csv, start=None, end=None)
    eval_df = _read(args.eval_csv, start=args.start, end=args.end, limit_rows=args.smoke_rows)
    cfg = CandidateConfig(
        horizon_bars=int(args.horizon_bars),
        window_bars=int(args.window_bars),
        lb_quantile=float(args.lb_quantile),
        min_support=int(args.min_support),
        min_lower_bound=float(args.min_lower_bound),
        min_cvar=float(args.min_cvar),
        iql_floor=float(args.iql_floor),
        cql_penalty=float(args.cql_penalty),
        max_notional=float(args.max_notional),
        leverage_cap=float(args.leverage_cap),
    )
    candidate = EmpiricalDTLifecyclePolicy(cfg)
    fit_meta = candidate.fit(train_df, entry_bundle, fee=float(args.fee), slip=float(args.slip))
    candidate_1x = backtest_candidate(eval_df, entry_bundle, candidate, fee=float(args.fee), slip=float(args.slip))
    stress = {}
    for mult in (1.0, 2.0, 3.0):
        stress[f"cost_{mult:g}x"] = _compact(
            backtest_candidate(eval_df, entry_bundle, candidate, fee=float(args.fee) * mult, slip=float(args.slip) * mult)
        )
    zero = _zero_artifact(args.zero_report)
    report = {
        "type": "dt_lifecycle_iql_cql_cvar_vs_zero_style",
        "note": "Candidate is a shadow surrogate: trajectory window interface + empirical IQL/CQL lower-bound gate + CVaR critic + allocator stub. Existing MuZero/AZ result is read from its report artifact, not retrained.",
        "inputs": {
            "policy": str(args.policy),
            "entry_source": str(args.entry_source),
            "train_csv": str(args.train_csv),
            "eval_csv": str(args.eval_csv),
            "zero_report": str(args.zero_report),
            "fee": float(args.fee),
            "slip": float(args.slip),
            "leverage_cap": float(args.leverage_cap),
            "max_notional": float(args.max_notional),
            "start": args.start,
            "end": args.end,
            "smoke_rows": args.smoke_rows,
        },
        "audit": {
            "train_range": _range(train_df),
            "eval_range": _range(eval_df),
            "candidate_fit": fit_meta,
            "zero_artifact_period_note": "Exact only when start/end/smoke_rows match the artifact's full eval period.",
        },
        "candidate_config": cfg.asdict(),
        "candidate": {"eval": _compact(candidate_1x), "cost_stress": stress},
        "zero_style_muzero_az_artifact": zero,
        "delta_vs_zero_artifact": {
            "pnl": None if not zero.get("eval") else float(candidate_1x.get("pnl", 0.0) - float(zero["eval"].get("pnl", 0.0) or 0.0)),
            "mdd": None if not zero.get("eval") else float(candidate_1x.get("mdd", 0.0) - float(zero["eval"].get("mdd", 0.0) or 0.0)),
            "trades": None if not zero.get("eval") else int(candidate_1x.get("trades", 0) - int(zero["eval"].get("trades", 0) or 0)),
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "candidate": report["candidate"]["eval"], "zero": zero.get("eval", {})}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
