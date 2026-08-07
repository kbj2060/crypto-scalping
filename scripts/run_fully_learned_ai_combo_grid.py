#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMBO_DIR = ROOT / "tmp/ai_feature_combo_grid"
DEFAULT_BASE_TRAIN = ROOT / "data/ensemble/event_driven/trade_candidates_v1_oof_regime_v3.csv"
DEFAULT_BASE_EVAL = ROOT / "data/ensemble/event_driven/trade_candidates_2026_oofdet_router_regime_v3_manifest_policy.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/reports/fully_learned_ai_combo_grid"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/fully_learned_ai_combo_grid"
DEFAULT_SUMMARY = ROOT / "data/ensemble/reports/fully_learned_ai_combo_grid_2026.json"


AI_GROUPS = ("patchtst", "tide", "timesnet", "dlinear")


def _combo_name(groups: tuple[str, ...]) -> str:
    return "__".join(groups) if groups else "none"


def _combo_paths(combo_dir: Path, groups: tuple[str, ...], base_train: Path, base_eval: Path) -> tuple[Path, Path]:
    if not groups:
        return base_train, base_eval
    name = _combo_name(groups)
    return combo_dir / f"trade_candidates_2025_{name}.csv", combo_dir / f"trade_candidates_2026_{name}.csv"


def _result_from_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    eval_result = dict(report.get("eval", {}) or {})
    validation = dict(report.get("validation", {}) or {})
    train = dict(report.get("train", {}) or {})
    return {
        "report": str(path),
        "eval": {k: eval_result.get(k) for k in (
            "pnl",
            "mdd",
            "trades",
            "wr",
            "trades_per_day",
            "long_entries",
            "short_entries",
            "avg_notional",
            "avg_leverage",
            "avg_take_profit",
            "avg_stop_loss",
            "avg_max_hold_bars",
        )},
        "validation": {k: validation.get(k) for k in ("pnl", "mdd", "trades", "wr", "trades_per_day")},
        "train": {k: train.get(k) for k in ("pnl", "mdd", "trades", "wr", "trades_per_day")},
        "config": report.get("config", {}),
    }


def _all_combos() -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = [()]
    for r in range(1, len(AI_GROUPS) + 1):
        combos.extend(tuple(c) for c in itertools.combinations(AI_GROUPS, r))
    return combos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run fully learned governor AI feature combo grid without Polymarket features.")
    p.add_argument("--combo-dir", type=Path, default=DEFAULT_COMBO_DIR)
    p.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    p.add_argument("--base-eval", type=Path, default=DEFAULT_BASE_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "type": "fully_learned_ai_combo_grid_2026",
        "note": "Polymarket excluded. Every combo retrains action, notional, leverage, TP, SL, max-hold, cooldown, and quality heads.",
        "base_train": str(args.base_train),
        "base_eval": str(args.base_eval),
        "combo_dir": str(args.combo_dir),
        "combos": {},
    }
    if args.resume and args.summary_out.exists():
        summary = json.loads(args.summary_out.read_text(encoding="utf-8"))
        summary.setdefault("combos", {})

    for groups in _all_combos():
        name = _combo_name(groups)
        report_out = args.out_dir / f"{name}_2026.json"
        model_out = args.model_dir / f"{name}.pkl"
        train_csv, eval_csv = _combo_paths(args.combo_dir, groups, args.base_train, args.base_eval)
        if not train_csv.exists() or not eval_csv.exists():
            raise FileNotFoundError(f"{name}: missing train/eval csv: {train_csv}, {eval_csv}")
        if args.resume and report_out.exists():
            summary["combos"][name] = {
                "groups": list(groups),
                "train_csv": str(train_csv),
                "eval_csv": str(eval_csv),
                "model": str(model_out),
                **_result_from_report(report_out),
            }
            print(f"[SKIP] {name}", flush=True)
            continue

        print(f"[RUN] {name}", flush=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts/train_eval_fully_learned_governor.py"),
            "--train-csv",
            str(train_csv),
            "--eval-csv",
            str(eval_csv),
            "--stride-bars",
            "12",
            "--batch-size",
            "512",
            "--max-train-horizon-bars",
            "288",
            "--adverse-penalty",
            "2.45",
            "--size-penalty",
            "0.180",
            "--hold-penalty",
            "0.042",
            "--turnover-bonus",
            "0.0012",
            "--cash-score",
            "0.020",
            "--notional-buckets",
            "0.20,0.32,0.50,0.75,1.05,1.45,2.00,2.70,3.60",
            "--take-profit-buckets",
            "0.007,0.011,0.018,0.030,0.050,0.090,0.180,0.450,0.900",
            "--stop-loss-buckets",
            "0.004,0.006,0.009,0.014,0.022,0.035,0.055",
            "--max-hold-buckets",
            "6,12,24,48,96,192,288",
            "--cooldown-buckets",
            "0,1,3,6,12,24,48",
            "--model-out",
            str(model_out),
            "--report-out",
            str(report_out),
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        summary["combos"][name] = {
            "groups": list(groups),
            "train_csv": str(train_csv),
            "eval_csv": str(eval_csv),
            "model": str(model_out),
            **_result_from_report(report_out),
        }
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    ranked = sorted(
        summary["combos"].items(),
        key=lambda kv: float(((kv[1].get("eval") or {}).get("pnl") or -1e18)),
        reverse=True,
    )
    summary["ranked_by_eval_pnl"] = [
        {
            "name": name,
            "pnl": row.get("eval", {}).get("pnl"),
            "mdd": row.get("eval", {}).get("mdd"),
            "trades": row.get("eval", {}).get("trades"),
            "validation_pnl": row.get("validation", {}).get("pnl"),
            "validation_mdd": row.get("validation", {}).get("mdd"),
        }
        for name, row in ranked
    ]
    args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": str(args.summary_out), "top5": summary["ranked_by_eval_pnl"][:5]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
