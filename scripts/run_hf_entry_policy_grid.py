#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/hf_entry_grid"
DEFAULT_REPORT_DIR = ROOT / "data/ensemble/reports/hf_entry_grid"
DEFAULT_SUMMARY = ROOT / "data/ensemble/reports/hf_entry_grid_patchtst_tide_dlinear_2026.json"


def _result(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {
        "report": str(path),
        "model": d.get("model_out"),
        "label_distribution": d.get("label_distribution", {}),
        "train": {k: d.get("train", {}).get(k) for k in ("pnl", "mdd", "trades", "trades_per_day", "wr")},
        "validation": {k: d.get("validation", {}).get(k) for k in ("pnl", "mdd", "trades", "trades_per_day", "wr")},
        "eval": {k: d.get("eval", {}).get(k) for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "long_entries", "short_entries", "avg_notional", "avg_leverage")},
        "config": d.get("config", {}),
    }


def _configs() -> list[dict[str, Any]]:
    return [
        {
            "name": "hf_v1_short_h96",
            "stride": 3,
            "horizon": 96,
            "adverse": 1.15,
            "size": 0.095,
            "hold": 0.010,
            "turnover": 0.010,
            "cash": -0.006,
            "notional": "0.10,0.16,0.24,0.34,0.48,0.68,0.95,1.30,1.80,2.40",
            "tp": "0.0035,0.005,0.007,0.010,0.014,0.020,0.030,0.045,0.070,0.110",
            "sl": "0.003,0.0045,0.006,0.0085,0.012,0.018,0.026",
            "hold_b": "3,6,9,12,18,24,36,48,72,96",
            "cool": "0,0,1,2,3,6",
        },
        {
            "name": "hf_v2_more_turnover_h72",
            "stride": 2,
            "horizon": 72,
            "adverse": 0.95,
            "size": 0.070,
            "hold": 0.006,
            "turnover": 0.018,
            "cash": -0.012,
            "notional": "0.08,0.12,0.18,0.26,0.38,0.55,0.80,1.15,1.60,2.20",
            "tp": "0.0028,0.004,0.0055,0.0075,0.010,0.014,0.020,0.030,0.045,0.070",
            "sl": "0.0025,0.0035,0.005,0.007,0.010,0.014,0.020",
            "hold_b": "3,5,8,12,18,24,36,48,72",
            "cool": "0,0,0,1,2,3",
        },
        {
            "name": "hf_v3_scalp_h48",
            "stride": 2,
            "horizon": 48,
            "adverse": 0.70,
            "size": 0.050,
            "hold": 0.004,
            "turnover": 0.030,
            "cash": -0.020,
            "notional": "0.05,0.08,0.12,0.18,0.26,0.38,0.55,0.80,1.15,1.60",
            "tp": "0.002,0.003,0.004,0.0055,0.0075,0.010,0.014,0.020,0.030",
            "sl": "0.002,0.003,0.0045,0.006,0.0085,0.012,0.018",
            "hold_b": "2,3,5,8,12,18,24,36,48",
            "cool": "0,0,0,0,1,2",
        },
        {
            "name": "hf_v4_balanced_h144",
            "stride": 3,
            "horizon": 144,
            "adverse": 1.55,
            "size": 0.110,
            "hold": 0.014,
            "turnover": 0.008,
            "cash": -0.004,
            "notional": "0.10,0.16,0.24,0.34,0.48,0.68,0.95,1.30,1.80,2.40,3.00",
            "tp": "0.004,0.006,0.009,0.013,0.020,0.030,0.050,0.080,0.130,0.220",
            "sl": "0.003,0.0045,0.0065,0.009,0.013,0.020,0.030",
            "hold_b": "3,6,12,18,24,36,48,72,96,144",
            "cool": "0,0,1,2,3,6,12",
        },
    ]


def _audit(train: Path, eval_path: Path) -> dict[str, Any]:
    import pandas as pd

    train_df = pd.read_csv(train, usecols=["timestamp"])
    eval_df = pd.read_csv(eval_path, usecols=["timestamp"])
    train_ts = pd.to_datetime(train_df["timestamp"], errors="coerce").dropna()
    eval_ts = pd.to_datetime(eval_df["timestamp"], errors="coerce").dropna()
    overlap = set(train_ts.astype("int64").tolist()) & set(eval_ts.astype("int64").tolist())
    return {
        "train_rows": int(len(train_ts)),
        "eval_rows": int(len(eval_ts)),
        "train_range": [str(train_ts.min()), str(train_ts.max())],
        "eval_range": [str(eval_ts.min()), str(eval_ts.max())],
        "timestamp_overlap_rows": int(len(overlap)),
        "train_duplicate_timestamps": int(train_ts.duplicated().sum()),
        "eval_duplicate_timestamps": int(eval_ts.duplicated().sum()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HF fully learned entry policy grid.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--resume", action="store_true", default=False)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "type": "hf_entry_grid_patchtst_tide_dlinear_2026",
        "goal": "Increase round-trip frequency toward 5+ trades/day while preserving fee/slippage accounting.",
        "audit": _audit(args.train_csv, args.eval_csv),
        "results": {},
    }
    for cfg in _configs():
        name = str(cfg["name"])
        model = args.model_dir / f"{name}.pkl"
        report = args.report_dir / f"{name}_2026.json"
        if args.resume and report.exists():
            summary["results"][name] = _result(report)
            continue
        cmd = [
            sys.executable,
            str(ROOT / "scripts/train_eval_fully_learned_governor.py"),
            "--train-csv",
            str(args.train_csv),
            "--eval-csv",
            str(args.eval_csv),
            "--stride-bars",
            str(cfg["stride"]),
            "--batch-size",
            "512",
            "--max-train-horizon-bars",
            str(cfg["horizon"]),
            "--adverse-penalty",
            str(cfg["adverse"]),
            "--size-penalty",
            str(cfg["size"]),
            "--hold-penalty",
            str(cfg["hold"]),
            "--turnover-bonus",
            str(cfg["turnover"]),
            "--cash-score",
            str(cfg["cash"]),
            "--notional-buckets",
            str(cfg["notional"]),
            "--take-profit-buckets",
            str(cfg["tp"]),
            "--stop-loss-buckets",
            str(cfg["sl"]),
            "--max-hold-buckets",
            str(cfg["hold_b"]),
            "--cooldown-buckets",
            str(cfg["cool"]),
            "--model-out",
            str(model),
            "--report-out",
            str(report),
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        summary["results"][name] = _result(report)
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    ranked_pnl = sorted(summary["results"].items(), key=lambda kv: float(kv[1]["eval"].get("pnl") or -1e18), reverse=True)
    ranked_freq = sorted(summary["results"].items(), key=lambda kv: float(kv[1]["eval"].get("trades_per_day") or -1e18), reverse=True)
    ranked_goal = sorted(
        summary["results"].items(),
        key=lambda kv: (
            float(kv[1]["eval"].get("pnl") or -1e18)
            if float(kv[1]["eval"].get("trades_per_day") or 0.0) >= 5.0
            else -1e18
        ),
        reverse=True,
    )
    summary["ranked_by_pnl"] = [{"name": n, **r["eval"]} for n, r in ranked_pnl]
    summary["ranked_by_frequency"] = [{"name": n, **r["eval"]} for n, r in ranked_freq]
    summary["ranked_goal_5_trades_per_day"] = [
        {"name": n, **r["eval"]} for n, r in ranked_goal if float(r["eval"].get("trades_per_day") or 0.0) >= 5.0
    ]
    args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": str(args.summary_out), "top_pnl": summary["ranked_by_pnl"][:4], "top_freq": summary["ranked_by_frequency"][:4], "goal": summary["ranked_goal_5_trades_per_day"][:4]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
