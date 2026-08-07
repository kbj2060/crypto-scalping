#!/usr/bin/env python3
"""Revive the 'clean base policy + clean exit governor' thread from the 2026-05-06
MuZero/AZ clean-scope reaudit (docs/experiments/clean_scope_muzero_az_reaudit_2026.md),
retrained/re-selected on the project's canonical fresh-forward split
(CLAUDE.md: VAL 2025-09-01..12-31, OOS 2026-01-01..03-31 -- OOS truncated to 2026-02-28
here since the frozen feature CSV does not extend past that date; documented as a
known limitation, not silently ignored).

Does NOT touch the MuZero/AZ entry-planner/overlay layers (confirmed OOS-collapsing on
2026-05-06, checkpoints since deleted) -- only the underlying policy+exit governor pair
that showed real but cost3-negative alpha.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    FullyLearnedGovernorConfig,
    build_training_set,
    train_policy,
)
from scripts.eval_hf_risk_overlay_grid import _read  # noqa: E402
from scripts.train_eval_hf_no_limit_exit_governor import (  # noqa: E402
    _base_frame,
    _compact,
    backtest_no_limit_exit,
    collect_exit_samples,
    train_exit_model,
)

DEFAULT_TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_SELECTION = ROOT / "data/ensemble/reports/hf_no_limit_exit_final_selection_2026.json"
DEFAULT_SOURCE_POLICY = ROOT / "data/ensemble/supervised/hf_entry_grid/hf_v4_balanced_h144.pkl"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_revival_canonical_20260731"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_revival_canonical_20260731.json"

VAL_START = "2025-09-01"
OOS_START = "2026-01-01"


def _load_selected(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    sel = obj.get("selected_balanced") or {}
    return dict(sel["entry_config"]), dict(sel["risk_config"]), dict(sel["exit_config"])


def _policy_config_from_bundle(path: Path) -> FullyLearnedGovernorConfig:
    if not path.exists():
        return FullyLearnedGovernorConfig()
    bundle = joblib.load(path)
    cfg = dict(bundle.get("config", {}) or {})
    allowed = set(FullyLearnedGovernorConfig.__dataclass_fields__.keys())
    return FullyLearnedGovernorConfig(**{k: v for k, v in cfg.items() if k in allowed})


def _range(df: pd.DataFrame) -> list[str]:
    if "timestamp" not in df.columns or df.empty:
        return ["", ""]
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    return [str(ts.min()), str(ts.max())] if not ts.empty else ["", ""]


def _score(bt: dict[str, Any], *, mdd_weight: float = 3.0) -> float:
    tpd = float(bt.get("trades_per_day", 0.0) or 0.0)
    sparse_penalty = 60.0 * max(0.0, 4.0 - tpd)
    return float(bt.get("pnl", 0.0) or 0.0) + float(mdd_weight) * float(bt.get("mdd", 0.0) or 0.0) - sparse_penalty


def _train_clean_policy(train_df: pd.DataFrame, *, cfg: FullyLearnedGovernorConfig, stride_bars: int, batch_size: int, seed: int, model_out: Path) -> dict[str, Any]:
    x, y, meta = build_training_set(train_df, cfg=cfg, stride_bars=int(stride_bars), batch_size=int(batch_size))
    bundle = train_policy(x, y, cfg=cfg, random_state=int(seed))
    bundle["clean_scope"] = {"train_range": _range(train_df), "fit_rows": int(len(train_df)), "seed": int(seed)}
    bundle["training_meta"] = meta
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_out)
    return bundle


def _train_clean_exit_model(train_df: pd.DataFrame, policy: dict[str, Any], entry_cfg: dict[str, Any], *, fee: float, slip: float, max_samples: int, seed: int, model_out: Path) -> Any:
    x, y, meta = collect_exit_samples(
        train_df, policy, entry_config=entry_cfg, fee=float(fee), slip=float(slip),
        entry_stride=36, min_age=3, max_age=144, age_stride=24, future_horizon=72,
        exit_edge=0.0015, adverse_gap=0.012, max_samples=int(max_samples), seed=int(seed),
    )
    model = train_exit_model(x, y, seed=int(seed))
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "sample_meta": meta, "entry_config": dict(entry_cfg)}, model_out)
    return model


def _select_controls_on_validation(val_df: pd.DataFrame, policy: dict[str, Any], exit_model: Any, entry_cfg: dict[str, Any], base_risk_cfg: dict[str, Any], *, fee: float, slip: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pre = _base_frame(val_df, policy, entry_cfg)
    for max_daily in (12, 16):
        for daily_loss in (0.025, 0.04):
            for daily_dd in (0.025, 0.035):
                for cooldown in (12, 24):
                    risk_cfg = dict(base_risk_cfg)
                    risk_cfg.update({"max_daily_trades": int(max_daily), "daily_loss_limit": float(daily_loss), "daily_dd_limit": float(daily_dd), "loss_cooldown_bars": int(cooldown)})
                    for th in (0.45, 0.55, 0.65):
                        for age in (3, 6, 12):
                            exit_cfg = {"exit_threshold": float(th), "min_exit_age": int(age)}
                            bt = backtest_no_limit_exit(val_df, policy, exit_model, entry_config=entry_cfg, risk_config=risk_cfg, exit_threshold=float(th), min_exit_age=int(age), fee=float(fee), slip=float(slip), precomputed=pre)
                            rows.append({"name": f"exit{th:.2f}_age{age}_max{max_daily}_dd{daily_dd}_loss{daily_loss}_cd{cooldown}", "entry_config": dict(entry_cfg), "risk_config": risk_cfg, "exit_config": exit_cfg, "eval": _compact(bt), "score": _score(bt)})
    ranked = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
    return ranked[0], ranked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    ap.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    ap.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    ap.add_argument("--source-policy", type=Path, default=DEFAULT_SOURCE_POLICY)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--policy-stride", type=int, default=3)
    ap.add_argument("--policy-batch-size", type=int, default=512)
    ap.add_argument("--samples", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_2025 = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    ts = pd.to_datetime(all_2025["timestamp"], errors="coerce")
    val_start = pd.Timestamp(VAL_START)
    train_df = all_2025.loc[ts < val_start].reset_index(drop=True)
    val_df = all_2025.loc[ts >= val_start].reset_index(drop=True)
    if train_df.empty or val_df.empty or eval_df.empty:
        raise ValueError("empty train/validation/eval split")

    print(f"[split] train={_range(train_df)} rows={len(train_df)}")
    print(f"[split] val(canonical)={_range(val_df)} rows={len(val_df)}")
    print(f"[split] eval/OOS(frozen csv, truncated vs canonical 03-31)={_range(eval_df)} rows={len(eval_df)}")

    entry_cfg0, risk_cfg0, _ = _load_selected(args.selection_report)
    cfg = _policy_config_from_bundle(args.source_policy)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    policy = _train_clean_policy(train_df, cfg=cfg, stride_bars=int(args.policy_stride), batch_size=int(args.policy_batch_size), seed=int(args.seed), model_out=args.model_dir / "hf_v4_clean_train_canonical.pkl")
    exit_model = _train_clean_exit_model(train_df, policy, entry_cfg0, fee=float(args.fee), slip=float(args.slip), max_samples=int(args.samples), seed=int(args.seed) + 1, model_out=args.model_dir / "hf_no_limit_exit_clean_train_canonical.pkl")

    selected, control_rows = _select_controls_on_validation(val_df, policy, exit_model, entry_cfg0, risk_cfg0, fee=float(args.fee), slip=float(args.slip))
    entry_cfg = dict(selected["entry_config"])
    risk_cfg = dict(selected["risk_config"])
    exit_cfg = dict(selected["exit_config"])
    print(f"[selected on canonical VAL] {selected['name']} -> {selected['eval']}")

    eval_pre = _base_frame(eval_df, policy, entry_cfg)
    cost_results: dict[str, Any] = {}
    for mult, label in ((1.0, "cost1"), (2.0, "cost2"), (3.0, "cost3")):
        bt = backtest_no_limit_exit(eval_df, policy, exit_model, entry_config=entry_cfg, risk_config=risk_cfg, exit_threshold=float(exit_cfg["exit_threshold"]), min_exit_age=int(exit_cfg["min_exit_age"]), fee=float(args.fee) * mult, slip=float(args.slip) * mult, precomputed=eval_pre)
        cost_results[label] = _compact(bt)
        print(f"[OOS {label}] pnl={bt.get('pnl'):.4f}% mdd={bt.get('mdd'):.4f}% trades={bt.get('trades')}")

    val_bt = backtest_no_limit_exit(val_df, policy, exit_model, entry_config=entry_cfg, risk_config=risk_cfg, exit_threshold=float(exit_cfg["exit_threshold"]), min_exit_age=int(exit_cfg["min_exit_age"]), fee=float(args.fee), slip=float(args.slip))

    report = {
        "purpose": "revive clean_scope_muzero_az_2026 base policy+exit governor on canonical fresh-forward split",
        "known_limitation": "OOS truncated to 2026-02-28 (frozen feature csv), not the full canonical 2026-03-31",
        "split": {"train": _range(train_df), "val_canonical": _range(val_df), "oos_truncated": _range(eval_df)},
        "entry_config": entry_cfg,
        "risk_config": risk_cfg,
        "exit_config": exit_cfg,
        "val_canonical_eval": _compact(val_bt),
        "oos_cost_stress": cost_results,
        "prior_reference_2026-05-06": {"oos_1x": "+177.33%/mdd-17.76%/363tr", "oos_3x": "-7.97%"},
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[report] wrote {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
