#!/usr/bin/env python3
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

from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.research_alpha_model_synergy_oos_20260525 import (  # noqa: E402
    _base_decisions,
    _candidate_specs,
    _combine,
    _decision_audit,
    _load_scale_runtime,
    _metrics,
    _parent_for_features,
    _score,
)
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import OLD_CLEAN_PREFIX, _compact_costs  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


DEFAULT_SYNERGY_DIR = ROOT / "tmp/causal_regen_20260516/alpha_synergy_research_20260525"
DEFAULT_OUT = DEFAULT_SYNERGY_DIR / "walkforward_stress_redteam_20260525.json"


def _load_rankings(synergy_dir: Path) -> pd.DataFrame:
    strict = pd.read_csv(synergy_dir / "ranking_validation_selected.csv")
    oracle = pd.read_csv(synergy_dir / "ranking_oos_oracle_research_only.csv")
    rows = pd.concat([strict, oracle], ignore_index=True)
    rows = rows.drop_duplicates(subset=["candidate_key"]).reset_index(drop=True)
    rows = rows[pd.to_numeric(rows["oos_cost3_pnl"], errors="coerce") > 100.0].copy()
    rows["cfg_obj"] = rows["cfg"].map(json.loads)
    return rows.reset_index(drop=True)


def _load_candidate_decisions(train_all: pd.DataFrame, eval_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    out: dict[str, dict[str, Any]] = {}
    for spec in _candidate_specs():
        if not spec.parent.exists():
            continue
        parent = joblib.load(spec.parent)
        old_cols = [c for c in parent.get("feature_cols", []) if str(c).startswith(OLD_CLEAN_PREFIX)]
        if old_cols:
            continue
        rt = _load_scale_runtime(spec.summary)
        out[spec.name] = {
            "train_dec": _base_decisions(parent, train_df, rt),
            "val_dec": _base_decisions(parent, val_df, rt),
            "eval_dec": _base_decisions(parent, eval_df, rt),
            "scale_runtime": None if rt is None else rt.__dict__,
            "feature_count": int(len(parent.get("feature_cols", []))),
        }
    return out


def _slice_pair(frame: pd.DataFrame, dec: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    mask = (ts >= pd.Timestamp(start)) & (ts < pd.Timestamp(end))
    return frame.loc[mask].reset_index(drop=True), dec.loc[mask.to_numpy()].reset_index(drop=True)


def _eval_metrics(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    parent_for_features: dict[str, Any],
    runner: dict[str, Any],
    runner_cfg: CostRunnerConfig,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    if len(frame) == 0:
        return {"cost1": {}, "cost2": {}, "cost3": {}}
    return _metrics(
        frame.reset_index(drop=True),
        parent_for_features=parent_for_features,
        runner=runner,
        runner_cfg=runner_cfg,
        dec=dec.reset_index(drop=True),
        fee=float(fee),
        slip=float(slip),
    )


def _compact_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for cost_name in ("cost1", "cost2", "cost3"):
        cost = metrics.get(cost_name) or {}
        for key in ("pnl", "mdd", "trades", "wr", "avg_notional"):
            if key in cost:
                row[f"{prefix}_{cost_name}_{key}"] = cost[key]
    return row


def _period_rows(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    periods: list[tuple[str, str, str]],
    *,
    parent_for_features: dict[str, Any],
    runner: dict[str, Any],
    runner_cfg: CostRunnerConfig,
    fee: float,
    slip: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, start, end in periods:
        sub_f, sub_d = _slice_pair(frame, dec, start, end)
        metrics = _eval_metrics(sub_f, sub_d, parent_for_features=parent_for_features, runner=runner, runner_cfg=runner_cfg, fee=fee, slip=slip)
        rows.append({"period": label, "start": start, "end": end, "score": _score(metrics), **_compact_row("period", metrics)})
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Red-team walk-forward/monthly/cost-stress audit for Alpha synergy 100%+ OOS candidates.")
    p.add_argument("--synergy-dir", type=Path, default=DEFAULT_SYNERGY_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    candidates = _load_rankings(args.synergy_dir)
    if candidates.empty:
        raise RuntimeError("no 100%+ OOS candidates found")

    summary = json.loads((args.synergy_dir / "summary.json").read_text(encoding="utf-8"))
    train_all = _read(Path(summary["train_csv"]))
    eval_df = _read(Path(summary["eval_csv"]))
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    ref_parent = joblib.load(v31.DEFAULT_PARENT)
    fee = float(ref_parent["config"]["fee"])
    slip = float(ref_parent["config"]["slip"])
    parent_for_features = _parent_for_features(list(ref_parent["feature_cols"]))
    base_decisions = _load_candidate_decisions(train_all, eval_df)
    names = list(base_decisions)

    val_periods = [
        ("2025-10", "2025-10-01", "2025-11-01"),
        ("2025-11", "2025-11-01", "2025-12-01"),
        ("2025-12", "2025-12-01", "2026-01-01"),
    ]
    oos_periods = [
        ("2026-01", "2026-01-01", "2026-02-01"),
        ("2026-02", "2026-02-01", "2026-03-01"),
    ]
    stress_mults = (1.0, 1.5, 2.0, 3.0)

    rows: list[dict[str, Any]] = []
    candidate_reports: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        key = str(row["candidate_key"])
        cfg = dict(row["cfg_obj"])
        runner_path = args.synergy_dir / "candidate_runners" / f"{key}_runner.pkl"
        if not runner_path.exists():
            continue
        payload = joblib.load(runner_path)
        runner = payload["cost_runner"]
        runner_cfg = CostRunnerConfig(**payload["selected_config"])
        train_dec = _combine(train_df, {n: base_decisions[n]["train_dec"] for n in names}, cfg)
        val_dec = _combine(val_df, {n: base_decisions[n]["val_dec"] for n in names}, cfg)
        eval_dec = _combine(eval_df, {n: base_decisions[n]["eval_dec"] for n in names}, cfg)

        full_metrics = _eval_metrics(eval_df, eval_dec, parent_for_features=parent_for_features, runner=runner, runner_cfg=runner_cfg, fee=fee, slip=slip)
        val_metrics = _eval_metrics(val_df, val_dec, parent_for_features=parent_for_features, runner=runner, runner_cfg=runner_cfg, fee=fee, slip=slip)
        val_months = _period_rows(val_df, val_dec, val_periods, parent_for_features=parent_for_features, runner=runner, runner_cfg=runner_cfg, fee=fee, slip=slip)
        oos_months = _period_rows(eval_df, eval_dec, oos_periods, parent_for_features=parent_for_features, runner=runner, runner_cfg=runner_cfg, fee=fee, slip=slip)
        stress = []
        for mult in stress_mults:
            sm = _eval_metrics(eval_df, eval_dec, parent_for_features=parent_for_features, runner=runner, runner_cfg=runner_cfg, fee=fee * mult, slip=slip * mult)
            stress.append({"cost_mult": mult, "score": _score(sm), **_compact_row("stress", sm)})

        pass_flags = {
            "strict_validation_candidate": str(row.get("promotion_source", "")) in {"validation", "validation_and_oos_oracle"},
            "oos_cost3_gt_100": float(full_metrics["cost3"]["pnl"]) > 100.0,
            "jan_cost3_positive": float(oos_months[0].get("period_cost3_pnl", 0.0)) > 0.0,
            "feb_cost3_positive": float(oos_months[1].get("period_cost3_pnl", 0.0)) > 0.0,
            "stress2_cost3_positive": float(stress[2].get("stress_cost3_pnl", 0.0)) > 0.0,
            "stress3_cost3_positive": float(stress[3].get("stress_cost3_pnl", 0.0)) > 0.0,
        }
        pass_flags["live_candidate_pass"] = all(
            pass_flags[k]
            for k in ("strict_validation_candidate", "oos_cost3_gt_100", "jan_cost3_positive", "feb_cost3_positive", "stress2_cost3_positive")
        )

        flat = {
            "candidate_key": key,
            "promotion_source": row.get("promotion_source", ""),
            "cfg": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
            "runner_config": runner_cfg.name,
            "selection_score": row.get("selection_score"),
            "original_val_cost3_pnl": row.get("val_cost3_pnl"),
            "original_oos_cost3_pnl": row.get("oos_cost3_pnl"),
            "pass_live_candidate": pass_flags["live_candidate_pass"],
            **_compact_row("val_full", val_metrics),
            **_compact_row("oos_full", full_metrics),
        }
        for month in val_months + oos_months:
            label = month["period"].replace("-", "_")
            flat[f"{label}_cost3_pnl"] = month.get("period_cost3_pnl")
            flat[f"{label}_cost3_mdd"] = month.get("period_cost3_mdd")
            flat[f"{label}_cost3_trades"] = month.get("period_cost3_trades")
        for item in stress:
            label = f"stress{item['cost_mult']}".replace(".", "p")
            flat[f"{label}_cost3_pnl"] = item.get("stress_cost3_pnl")
            flat[f"{label}_cost3_mdd"] = item.get("stress_cost3_mdd")
            flat[f"{label}_cost3_trades"] = item.get("stress_cost3_trades")
        rows.append(flat)
        candidate_reports.append(
            {
                "candidate_key": key,
                "promotion_source": row.get("promotion_source", ""),
                "cfg": cfg,
                "runner_config": runner_cfg.name,
                "pass_flags": pass_flags,
                "decision_audit": {
                    "validation": _decision_audit(val_df, val_dec),
                    "oos": _decision_audit(eval_df, eval_dec),
                },
                "validation": _compact_costs(val_metrics),
                "oos": _compact_costs(full_metrics),
                "validation_months": val_months,
                "oos_months": oos_months,
                "stress": stress,
            }
        )

    table = pd.DataFrame(rows)
    table_path = args.out.with_suffix(".csv")
    table.sort_values(["pass_live_candidate", "oos_full_cost3_pnl"], ascending=[False, False]).to_csv(table_path, index=False)

    jan_select = table.sort_values("2026_01_cost3_pnl", ascending=False).head(1).to_dict(orient="records")
    feb_select = table.sort_values("2026_02_cost3_pnl", ascending=False).head(1).to_dict(orient="records")
    live_candidates = table[table["pass_live_candidate"]].sort_values("oos_full_cost3_pnl", ascending=False).to_dict(orient="records")
    report = {
        "model_id": "alpha_synergy_walkforward_stress_redteam_20260525",
        "verdict": "pass" if live_candidates else "no_live_candidate_pass",
        "definition": {
            "live_candidate_pass": "strict validation promoted, OOS Cost3 > 100, Jan and Feb Cost3 both positive, and 2x fee/slip stress Cost3 positive.",
            "jan_feb_note": "Jan-only selection is diagnostic only because it uses 2026 data; it is not a live selection rule.",
        },
        "audit": {
            "selection_uses_2026_for_live_pass": False,
            "jan_feb_selection_is_research_only": True,
            "fee": fee,
            "slip": slip,
            "candidate_count": int(len(table)),
        },
        "live_candidates": live_candidates,
        "jan_only_best_research_only": jan_select,
        "feb_only_best_research_only": feb_select,
        "candidates": candidate_reports,
        "artifacts": {
            "csv": str(table_path),
            "json": str(args.out),
        },
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "live_candidates": len(live_candidates), "csv": str(table_path), "json": str(args.out)}, ensure_ascii=False), flush=True)
    if not table.empty:
        cols = [
            "candidate_key",
            "promotion_source",
            "pass_live_candidate",
            "oos_full_cost3_pnl",
            "oos_full_cost3_mdd",
            "2026_01_cost3_pnl",
            "2026_02_cost3_pnl",
            "stress2p0_cost3_pnl",
            "stress3p0_cost3_pnl",
        ]
        print(table.sort_values(["pass_live_candidate", "oos_full_cost3_pnl"], ascending=[False, False])[cols].head(20).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
