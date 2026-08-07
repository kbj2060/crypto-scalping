#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.microstructure_wnc_sleeve import train_microstructure_classifier
from ensemble.trend_bull_bear_sleeve import TARGET_REGIMES, train_trend_classifier


BASE_TRAIN_CSV = ROOT / "data/ensemble/event_driven/trade_candidates_v1_oof_regime_v3.csv"
BASE_EVAL_CSV = ROOT / "data/ensemble/event_driven/trade_candidates_2026_oofdet_router_regime_v3_manifest_policy.csv"
AI_TRAIN_CSV = ROOT / "data/tmp/unified_build_ckpt/03_after_ai.csv"
AI_EVAL_CSV = ROOT / "data/tmp/unified_build_ckpt_2026/03_after_ai.csv"
CURRENT_MICRO_MODEL = ROOT / "data/ensemble/supervised/microstructure_wnc_sleeve_realistic_5x.pkl"
CURRENT_TREND_MODEL = ROOT / "data/ensemble/supervised/trend_bull_bear_sleeve_v1_c68_g34_notional5_leverage5.pkl"
DEFAULT_OUT_DIR = ROOT / "tmp/ai_feature_ablation"
DEFAULT_SUMMARY = ROOT / "data/ensemble/reports/governor_ai_feature_ablation_2026.json"


AI_GROUPS: dict[str, list[str]] = {
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
    ],
    "tide": [
        "ai_adverse_risk",
        "ai_reward_risk",
        "ai_vol_regime_pct",
        "tide_vol_raw",
        "tide_vol_zscore",
    ],
    "timesnet": [
        "ai_anchor_revert_prob",
        "ai_anchor_overheat",
        "ai_anchor_trend_escape_prob",
        "timesnet_cycle_sin",
        "timesnet_cycle_cos",
        "timesnet_cycle_delta",
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
ALL_AI_MODEL_COLUMNS = sorted({c for cols in AI_GROUPS.values() for c in cols})


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise KeyError(f"{path} missing timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _build_variant_csv(base_path: Path, ai_path: Path, variant: str, out_path: Path) -> dict[str, Any]:
    base = _read_csv(base_path)
    ai = _read_csv(ai_path)
    selected_cols = list(AI_GROUPS[variant])
    missing = [c for c in selected_cols if c not in ai.columns]
    if missing:
        raise ValueError(f"{variant} missing AI cols in {ai_path}: {missing}")

    drop_cols = [c for c in ALL_AI_MODEL_COLUMNS if c in base.columns]
    base = base.drop(columns=drop_cols, errors="ignore")
    merge_cols = ["timestamp"] + selected_cols
    merged = base.merge(ai[merge_cols], on="timestamp", how="left", validate="one_to_one")
    for col in selected_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return {
        "path": str(out_path),
        "rows": int(len(merged)),
        "cols": int(len(merged.columns)),
        "selected_ai_cols": selected_cols,
        "dropped_base_ai_cols": drop_cols,
    }


def _current_config(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    return dict(bundle.get("config", {}))


def _train_micro(train_df: pd.DataFrame, model_out: Path, variant: str, config: dict[str, Any]) -> dict[str, Any]:
    model, feature_cols, train_summary = train_microstructure_classifier(
        train_df,
        feature_mode="full",
        model_mode="ensemble",
    )
    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "classes": [int(c) for c in list(model.classes_)],
        "config": config,
        "train_summary": train_summary,
        "target_regimes": ["whipsaw", "normal", "chop"],
        "model_type": "microstructure_wnc_sleeve_v2",
        "profile": f"ai_ablation_{variant}",
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_out)
    return {
        "model_out": str(model_out),
        "feature_count": int(len(feature_cols)),
        "ai_feature_cols": [c for c in feature_cols if c in ALL_AI_MODEL_COLUMNS],
        "train_summary": train_summary,
    }


def _train_trend(train_df: pd.DataFrame, model_out: Path, variant: str, config: dict[str, Any]) -> dict[str, Any]:
    model, feature_cols, train_summary = train_trend_classifier(train_df, feature_mode="full")
    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "classes": [int(c) for c in list(model.classes_)],
        "config": config,
        "train_summary": train_summary,
        "target_regimes": list(TARGET_REGIMES),
        "model_type": "trend_bull_bear_sleeve_v1",
        "feature_mode": "full",
        "profile": f"ai_ablation_{variant}",
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_out)
    return {
        "model_out": str(model_out),
        "feature_count": int(len(feature_cols)),
        "ai_feature_cols": [c for c in feature_cols if c in ALL_AI_MODEL_COLUMNS],
        "train_summary": train_summary,
    }


def _run_governor(eval_csv: Path, micro_model: Path, trend_model: Path, report_out: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts/eval_governor_microstructure_wnc_oos_2026.py"),
        "--eval-csv",
        str(eval_csv),
        "--micro-model",
        str(micro_model),
        "--trend-model",
        str(trend_model),
        "--report-out",
        str(report_out),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    report = json.loads(report_out.read_text(encoding="utf-8"))
    result = dict(report.get("result", {}))
    return {
        "report_out": str(report_out),
        "pnl_pct": result.get("pnl"),
        "mdd_pct": result.get("mdd"),
        "trades": result.get("trades"),
        "win_rate_pct": (float(result["wr"]) * 100.0 if result.get("wr") is not None else None),
        "trades_per_day": result.get("trades_per_day"),
        "sniper_entries": result.get("sniper_entries"),
        "trend_entries": result.get("trend_entries"),
        "micro_entries": result.get("micro_entries"),
        "trend_regime_entries": result.get("trend_regime_entries"),
        "micro_regime_entries": result.get("micro_regime_entries"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one-AI-model-at-a-time governor ablation and backtest on 2026.")
    p.add_argument("--variants", default="patchtst,tide,timesnet,dlinear")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--base-train-csv", type=Path, default=BASE_TRAIN_CSV)
    p.add_argument("--base-eval-csv", type=Path, default=BASE_EVAL_CSV)
    p.add_argument("--ai-train-csv", type=Path, default=AI_TRAIN_CSV)
    p.add_argument("--ai-eval-csv", type=Path, default=AI_EVAL_CSV)
    p.add_argument("--current-micro-model", type=Path, default=CURRENT_MICRO_MODEL)
    p.add_argument("--current-trend-model", type=Path, default=CURRENT_TREND_MODEL)
    p.add_argument("--skip-csv-build", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    variants = [v.strip().lower() for v in str(args.variants).split(",") if v.strip()]
    unknown = [v for v in variants if v not in AI_GROUPS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; choices={sorted(AI_GROUPS)}")

    micro_config = _current_config(args.current_micro_model)
    trend_config = _current_config(args.current_trend_model)
    summary: dict[str, Any] = {
        "type": "governor_ai_feature_ablation_2026",
        "base_train_csv": str(args.base_train_csv),
        "base_eval_csv": str(args.base_eval_csv),
        "ai_train_csv": str(args.ai_train_csv),
        "ai_eval_csv": str(args.ai_eval_csv),
        "variants": {},
    }
    for variant in variants:
        print(f"[VARIANT] {variant}", flush=True)
        train_csv = args.out_dir / f"trade_candidates_2025_{variant}_only.csv"
        eval_csv = args.out_dir / f"trade_candidates_2026_{variant}_only.csv"
        if args.skip_csv_build:
            train_info = {"path": str(train_csv)}
            eval_info = {"path": str(eval_csv)}
        else:
            train_info = _build_variant_csv(args.base_train_csv, args.ai_train_csv, variant, train_csv)
            eval_info = _build_variant_csv(args.base_eval_csv, args.ai_eval_csv, variant, eval_csv)

        train_df = pd.read_csv(train_csv)
        micro_model = args.out_dir / f"microstructure_wnc_sleeve_realistic_5x_{variant}_only.pkl"
        trend_model = args.out_dir / f"trend_bull_bear_sleeve_5x_{variant}_only.pkl"
        micro_info = _train_micro(train_df, micro_model, variant, micro_config)
        trend_info = _train_trend(train_df, trend_model, variant, trend_config)
        report_out = args.out_dir / f"governor_2026_{variant}_only.json"
        governor_info = _run_governor(eval_csv, micro_model, trend_model, report_out)
        summary["variants"][variant] = {
            "train_csv": train_info,
            "eval_csv": eval_info,
            "micro": micro_info,
            "trend": trend_info,
            "governor": governor_info,
        }
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"variant": variant, **governor_info}, ensure_ascii=False), flush=True)

    ranked = sorted(
        (
            {"variant": k, **v["governor"]}
            for k, v in summary["variants"].items()
        ),
        key=lambda x: float(x.get("pnl_pct") or -1e30),
        reverse=True,
    )
    summary["ranking_by_pnl"] = ranked
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_out": str(args.summary_out), "ranking_by_pnl": ranked}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
