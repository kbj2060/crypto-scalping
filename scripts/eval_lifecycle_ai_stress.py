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

from scripts.train_eval_fully_learned_governor import backtest_policy  # noqa: E402

try:
    from scripts.run_lifecycle_manager_grid import backtest_lifecycle  # noqa: E402
except ModuleNotFoundError:
    backtest_lifecycle = None


DEFAULT_POLICY = ROOT / "data/ensemble/supervised/fully_learned_ai_combo_grid/patchtst__tide__dlinear.pkl"
DEFAULT_LIFECYCLE = ROOT / "data/ensemble/supervised/lifecycle_manager/patchtst_tide_dlinear_lifecycle_exit.pkl"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/lifecycle_ai_failure_stress_patchtst_tide_dlinear_2026.json"

AI_GROUPS = {
    "patchtst": [
        "pred_patchtst",
        "conf_patchtst",
        "ai_dir_edge",
        "ai_dir_p_up",
        "ai_dir_p_down",
        "ai_dir_p_flat",
        "ai_dir_entropy",
        "patchtst_median",
        "patchtst_regime_sim",
        "patchtst_pred",
        "patchtst_confidence",
    ],
    "tide": [
        "ai_adverse_risk",
        "ai_reward_risk",
        "ai_vol_regime_pct",
        "tide_vol_raw",
        "tide_vol_zscore",
    ],
    "dlinear": [
        "ai_flow_pressure",
        "ai_flow_exhaustion",
        "ai_flow_flip_prob",
        "ai_flow_slope",
        "dlinear_smf_ema",
        "dlinear_smf_slope",
    ],
}

LIFECYCLE_CONFIGS = {
    "early_exit_0.84": {
        "exit_threshold": 0.84,
        "scale_threshold": None,
        "scale_multiplier": 1.0,
        "scale_up_threshold": None,
        "scale_up_multiplier": 1.0,
    },
    "best_eval_scaledown": {
        "exit_threshold": 0.85,
        "scale_threshold": 0.65,
        "scale_multiplier": 0.50,
        "scale_up_threshold": None,
        "scale_up_multiplier": 1.0,
    },
    "best_validation_scaleup": {
        "exit_threshold": 0.85,
        "scale_threshold": None,
        "scale_multiplier": 1.0,
        "scale_up_threshold": 0.10,
        "scale_up_multiplier": 1.25,
    },
    "highest_return_regime_conf_scaleup": {
        "exit_threshold": 0.85,
        "scale_threshold": None,
        "scale_multiplier": 1.0,
        "scale_up_threshold": 0.10,
        "scale_up_multiplier": 1.25,
        "scale_up_min_regime_confidence": 0.55,
    },
    "mdd_safe_hybrid_guarded": {
        "exit_threshold": 0.85,
        "scale_threshold": 0.65,
        "scale_multiplier": 0.50,
        "scale_up_threshold": 0.10,
        "scale_up_multiplier": 1.20,
        "scale_up_max_account_dd": 0.06,
        "scale_up_max_position_drawdown": 0.020,
        "scale_up_min_unrealized": 0.004,
        "scale_up_min_regime_confidence": 0.55,
        "scale_up_block_regime_disagree": True,
    },
    "highest_return_ext_hybrid": {
        "exit_threshold": 0.85,
        "scale_threshold": 0.65,
        "scale_multiplier": 0.50,
        "scale_up_threshold": 0.10,
        "scale_up_multiplier": 1.35,
        "scale_up_max_account_dd": 0.08,
        "scale_up_max_position_drawdown": 0.015,
        "scale_up_min_unrealized": 0.003,
        "scale_up_min_regime_confidence": 0.55,
        "scale_up_block_regime_disagree": True,
    },
    "mdd_safe_ext_hybrid": {
        "exit_threshold": 0.85,
        "scale_threshold": 0.65,
        "scale_multiplier": 0.50,
        "scale_up_threshold": 0.10,
        "scale_up_multiplier": 1.15,
        "scale_up_max_account_dd": 0.08,
        "scale_up_max_position_drawdown": 0.025,
        "scale_up_min_unrealized": 0.003,
        "scale_up_min_regime_confidence": 0.55,
        "scale_up_block_regime_disagree": True,
    },
}


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _cols_for(groups: list[str]) -> list[str]:
    cols: list[str] = []
    for group in groups:
        cols.extend(AI_GROUPS.get(group, []))
    return list(dict.fromkeys(cols))


def _stress_frame(base: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = base.copy()
    if name == "normal":
        return out, {"mode": name, "changed_cols": []}
    if name == "all_ai_zero":
        cols = _cols_for(list(AI_GROUPS))
        for col in cols:
            if col in out.columns:
                out[col] = 0.0
        return out, {"mode": name, "changed_cols": [c for c in cols if c in out.columns]}
    if name.endswith("_zero"):
        groups = name.removesuffix("_zero").split("__")
        cols = _cols_for(groups)
        for col in cols:
            if col in out.columns:
                out[col] = 0.0
        return out, {"mode": name, "changed_cols": [c for c in cols if c in out.columns]}
    if name.endswith("_stale_1d"):
        groups = name.removesuffix("_stale_1d").split("__")
        cols = _cols_for(groups)
        for col in cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").shift(288).ffill().fillna(0.0)
        return out, {"mode": name, "changed_cols": [c for c in cols if c in out.columns], "shift_bars": 288}
    raise ValueError(f"unknown stress mode: {name}")


def _compact(bt: dict[str, Any]) -> dict[str, Any]:
    return {
        k: bt.get(k)
        for k in (
            "pnl",
            "mdd",
            "trades",
            "wr",
            "trades_per_day",
            "long_entries",
            "short_entries",
            "avg_notional",
            "avg_leverage",
            "lifecycle_exits",
            "scale_downs",
            "scale_ups",
        )
        if k in bt
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stress test baseline/lifecycle governor under AI feature failures.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--lifecycle", type=Path, default=DEFAULT_LIFECYCLE)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--min-age", type=int, default=3)
    p.add_argument("--max-notional", type=float, default=3.60)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    policy = joblib.load(args.policy)
    lifecycle = joblib.load(args.lifecycle)
    exit_model = lifecycle["model"]
    base = _read(args.eval_csv)
    modes = [
        "normal",
        "patchtst_zero",
        "tide_zero",
        "dlinear_zero",
        "patchtst__tide_zero",
        "patchtst__dlinear_zero",
        "tide__dlinear_zero",
        "all_ai_zero",
        "patchtst_stale_1d",
        "tide_stale_1d",
        "dlinear_stale_1d",
    ]
    results: dict[str, Any] = {}
    for mode in modes:
        df, meta = _stress_frame(base, mode)
        entry = backtest_policy(df, policy, fee=float(args.fee), slip=float(args.slip))
        row: dict[str, Any] = {"stress": meta, "entry_only": _compact(entry), "lifecycle": {}}
        for name, cfg in LIFECYCLE_CONFIGS.items():
            bt = backtest_lifecycle(
                df,
                policy,
                exit_model,
                fee=float(args.fee),
                slip=float(args.slip),
                min_age=int(args.min_age),
                max_notional=float(args.max_notional),
                **cfg,
            )
            row["lifecycle"][name] = _compact(bt)
        results[mode] = row
    report = {
        "type": "lifecycle_ai_failure_stress_patchtst_tide_dlinear_2026",
        "policy": str(args.policy),
        "lifecycle": str(args.lifecycle),
        "eval_csv": str(args.eval_csv),
        "results": results,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "normal": results["normal"],
                "all_ai_zero": results["all_ai_zero"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
