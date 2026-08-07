#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.microstructure_wnc_sleeve import train_microstructure_classifier
from ensemble.trend_bull_bear_sleeve import TARGET_REGIMES, train_trend_classifier
from scripts.run_governor_ai_model_ablation import (
    AI_GROUPS,
    ALL_AI_MODEL_COLUMNS,
    BASE_EVAL_CSV,
    BASE_TRAIN_CSV,
    AI_EVAL_CSV,
    AI_TRAIN_CSV,
    CURRENT_MICRO_MODEL,
    CURRENT_TREND_MODEL,
)


DEFAULT_OUT_DIR = ROOT / "tmp/ai_feature_combo_grid"
DEFAULT_SUMMARY = ROOT / "data/ensemble/reports/governor_ai_feature_combo_grid_2026.json"


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise KeyError(f"{path} missing timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _combo_name(groups: tuple[str, ...]) -> str:
    return "__".join(groups) if groups else "none"


def _combo_cols(groups: tuple[str, ...]) -> list[str]:
    cols: list[str] = []
    for group in groups:
        cols.extend(AI_GROUPS[group])
    return list(dict.fromkeys(cols))


def _build_combo_csv(base_path: Path, ai_path: Path, groups: tuple[str, ...], out_path: Path) -> dict[str, Any]:
    base = _read_csv(base_path)
    ai = _read_csv(ai_path)
    selected_cols = _combo_cols(groups)
    missing = [c for c in selected_cols if c not in ai.columns]
    if missing:
        raise ValueError(f"{groups} missing AI cols in {ai_path}: {missing}")
    drop_cols = [c for c in ALL_AI_MODEL_COLUMNS if c in base.columns]
    merged = base.drop(columns=drop_cols, errors="ignore")
    source_nan_rates: dict[str, float] = {}
    if selected_cols:
        merged = merged.merge(ai[["timestamp"] + selected_cols], on="timestamp", how="left", validate="one_to_one", indicator="__ai_merge")
        missing_merge_rate = float((merged["__ai_merge"] == "left_only").mean())
        merged.drop(columns=["__ai_merge"], inplace=True)
        if missing_merge_rate > 0.0:
            raise ValueError(f"{groups} AI timestamp merge missed {missing_merge_rate:.6f} of rows in {ai_path}")
        for col in selected_cols:
            source_nan_rates[col] = float(pd.to_numeric(merged[col], errors="coerce").isna().mean())
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return {
        "path": str(out_path),
        "rows": int(len(merged)),
        "cols": int(len(merged.columns)),
        "groups": list(groups),
        "selected_ai_cols": selected_cols,
        "dropped_base_ai_cols": drop_cols,
        "source_nan_rates": source_nan_rates,
    }


def _load_config(path: Path) -> dict[str, Any]:
    return dict(joblib.load(path).get("config", {}))


def _train_micro(train_df: pd.DataFrame, model_out: Path, groups: tuple[str, ...], config: dict[str, Any]) -> dict[str, Any]:
    model, feature_cols, train_summary = train_microstructure_classifier(train_df, feature_mode="full", model_mode="ensemble")
    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "classes": [int(c) for c in list(model.classes_)],
        "config": config,
        "train_summary": train_summary,
        "target_regimes": ["whipsaw", "normal", "chop"],
        "model_type": "microstructure_wnc_sleeve_v2",
        "profile": f"ai_combo_{_combo_name(groups)}",
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_out)
    return {
        "model_out": str(model_out),
        "feature_count": int(len(feature_cols)),
        "ai_feature_cols": [c for c in feature_cols if c in ALL_AI_MODEL_COLUMNS],
        "train_summary": train_summary,
    }


def _train_trend(train_df: pd.DataFrame, model_out: Path, groups: tuple[str, ...], config: dict[str, Any]) -> dict[str, Any]:
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
        "profile": f"ai_combo_{_combo_name(groups)}",
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_out)
    return {
        "model_out": str(model_out),
        "feature_count": int(len(feature_cols)),
        "ai_feature_cols": [c for c in feature_cols if c in ALL_AI_MODEL_COLUMNS],
        "train_summary": train_summary,
    }


def _run_governor(eval_csv: Path, micro_model: Path, trend_model: Path, report_out: Path, *, disable_sniper: bool) -> dict[str, Any]:
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
    if disable_sniper:
        cmd.append("--disable-sniper")
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    result = json.loads(report_out.read_text(encoding="utf-8")).get("result", {})
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


def _all_nonempty_combos() -> list[tuple[str, ...]]:
    groups = list(AI_GROUPS.keys())
    out: list[tuple[str, ...]] = []
    for r in range(1, len(groups) + 1):
        out.extend(tuple(c) for c in itertools.combinations(groups, r))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate all AI feature subset combos for trend+micro governor sleeves.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--base-train-csv", type=Path, default=BASE_TRAIN_CSV)
    p.add_argument("--base-eval-csv", type=Path, default=BASE_EVAL_CSV)
    p.add_argument("--ai-train-csv", type=Path, default=AI_TRAIN_CSV)
    p.add_argument("--ai-eval-csv", type=Path, default=AI_EVAL_CSV)
    p.add_argument("--current-micro-model", type=Path, default=CURRENT_MICRO_MODEL)
    p.add_argument("--current-trend-model", type=Path, default=CURRENT_TREND_MODEL)
    p.add_argument("--disable-sniper", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    combos = _all_nonempty_combos()
    micro_config = _load_config(args.current_micro_model)
    trend_config = _load_config(args.current_trend_model)
    summary: dict[str, Any]
    if args.resume and args.summary_out.exists():
        summary = json.loads(args.summary_out.read_text(encoding="utf-8"))
        summary.setdefault("combos", {})
    else:
        summary = {
            "type": "governor_ai_feature_combo_grid_2026",
            "sniper_enabled": not bool(args.disable_sniper),
            "note": "Legacy sniper checkpoint has no direct PatchTST/TiDE/TimesNet/DLinear state features; combos retrain trend+micro sleeves and evaluate full governor.",
            "base_train_csv": str(args.base_train_csv),
            "base_eval_csv": str(args.base_eval_csv),
            "ai_train_csv": str(args.ai_train_csv),
            "ai_eval_csv": str(args.ai_eval_csv),
            "combos": {},
        }
    for groups in combos:
        name = _combo_name(groups)
        if args.resume and name in summary["combos"] and "governor" in summary["combos"][name]:
            print(f"[SKIP] {name}", flush=True)
            continue
        print(f"[COMBO] {name}", flush=True)
        train_csv = args.out_dir / f"trade_candidates_2025_{name}.csv"
        eval_csv = args.out_dir / f"trade_candidates_2026_{name}.csv"
        train_info = _build_combo_csv(args.base_train_csv, args.ai_train_csv, groups, train_csv)
        eval_info = _build_combo_csv(args.base_eval_csv, args.ai_eval_csv, groups, eval_csv)
        train_df = pd.read_csv(train_csv)
        micro_model = args.out_dir / f"micro_{name}.pkl"
        trend_model = args.out_dir / f"trend_{name}.pkl"
        micro_info = _train_micro(train_df, micro_model, groups, micro_config)
        trend_info = _train_trend(train_df, trend_model, groups, trend_config)
        report_out = args.out_dir / f"governor_2026_{name}{'_no_sniper' if args.disable_sniper else ''}.json"
        governor = _run_governor(eval_csv, micro_model, trend_model, report_out, disable_sniper=bool(args.disable_sniper))
        summary["combos"][name] = {
            "groups": list(groups),
            "train_csv": train_info,
            "eval_csv": eval_info,
            "micro": micro_info,
            "trend": trend_info,
            "governor": governor,
        }
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"combo": name, **governor}, ensure_ascii=False), flush=True)
    ranking = sorted(
        ({"combo": k, **v["governor"]} for k, v in summary["combos"].items()),
        key=lambda x: float(x.get("pnl_pct") or -1e30),
        reverse=True,
    )
    summary["ranking_by_pnl"] = ranking
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_out": str(args.summary_out), "ranking_by_pnl": ranking}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
