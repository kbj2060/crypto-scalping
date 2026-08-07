#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "alpha5_a5dir_2024_train_2025_score_20260521"

DEFAULT_FEATURES_2024 = ROOT / "data/splits/year_oos/training_features_2024.csv"
DEFAULT_RL_BASE_2024 = ROOT / "data/splits/year_oos/rl_base_2024.csv"
DEFAULT_RL_UNIFIED_2024 = ROOT / "data/rl_training_2024_unified.csv"
DEFAULT_RL_UNIFIED_2025 = ROOT / "data/rl_training_2025_unified.csv"
DEFAULT_UNIFIED_2024_CKPT = ROOT / "data/tmp/unified_build_ckpt_2024"

DEFAULT_CLEAN4_2024 = ROOT / "data/ensemble/supervised/clean_regime4_state24_sticky090_v2_20260517/training_features_2024_clean_regime4_state24_sticky090_v2.csv"
DEFAULT_PRED4_2024 = ROOT / "data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2024_regime4_pred_tft_vsn_selected.csv"
DEFAULT_CLEAN4_2025 = ROOT / "data/ensemble/supervised/clean_regime4_state24_sticky090_v2_20260517/training_features_2025_clean_regime4_state24_sticky090_v2.csv"
DEFAULT_PRED4_2025 = ROOT / "data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2025_regime4_pred_tft_vsn_selected.csv"
DEFAULT_CLEAN4_REPORT = ROOT / "data/ensemble/reports/clean_regime4_state24_sticky090_v2_20260517_report.json"
DEFAULT_ROUTER_FEATURE_LIST_JSON = ROOT / "tmp/causal_regen_20260516/alpha5_router5_full_candidate_search_20260521/rank_pruned_stable_top48_feature_list.json"

DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260521"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a5dir router artifacts from 2024 training data and score 2025 RL rows."
    )
    p.add_argument("--features-2024", type=Path, default=DEFAULT_FEATURES_2024)
    p.add_argument("--rl-base-2024", type=Path, default=DEFAULT_RL_BASE_2024)
    p.add_argument("--rl-2024-unified", type=Path, default=DEFAULT_RL_UNIFIED_2024)
    p.add_argument("--rl-2025-unified", type=Path, default=DEFAULT_RL_UNIFIED_2025)
    p.add_argument("--unified-2024-ckpt", type=Path, default=DEFAULT_UNIFIED_2024_CKPT)
    p.add_argument("--clean4-2024", type=Path, default=DEFAULT_CLEAN4_2024)
    p.add_argument("--pred4-2024", type=Path, default=DEFAULT_PRED4_2024)
    p.add_argument("--clean4-2025", type=Path, default=DEFAULT_CLEAN4_2025)
    p.add_argument("--pred4-2025", type=Path, default=DEFAULT_PRED4_2025)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--prefix", default="a5dir")
    p.add_argument("--train-end", default="2024-10-01")
    p.add_argument("--val-start", default="2024-10-01")
    p.add_argument("--val-end", default="2025-01-01")
    p.add_argument("--oos-start", default="2025-01-01")
    p.add_argument("--devices", default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight-router3", type=float, default=0.8)
    p.add_argument("--weight-router4", type=float, default=0.2)
    p.add_argument("--router-feature-list-json", type=Path, default=DEFAULT_ROUTER_FEATURE_LIST_JSON)
    p.add_argument("--force", action="store_true")
    p.add_argument("--startup-check-only", action="store_true")
    return p.parse_args()


def _run_py(script_rel: str, *args: str) -> None:
    script_path = ROOT / script_rel
    cmd = [sys.executable, str(script_path), *args]
    print(json.dumps({"stage": "run", "script": script_rel, "cmd": cmd}, ensure_ascii=False), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _exists_all(paths: list[Path]) -> bool:
    return all(Path(p).exists() for p in paths)


def _step_paths(base: Path) -> dict[str, Path]:
    return {
        "tp_sl_dir": base / "01_alpha4_tp_sl_action_score_2024_to_2025",
        "fixed_dir": base / "02_fixed_regime4_state24_sticky090_tp18_sl10_preprocess_2024_to_2025",
        "alpha5_27_dir": base / "03_alpha5_27_label_factory_2024_to_2025",
        "alpha5_28_dir": base / "04_alpha5_28_label_factory_split_ambiguous_2024_to_2025",
        "alpha5_29_dir": base / "05_alpha5_29_hier_label_factory_2024_to_2025",
        "alpha5_30_dir": base / "06_alpha5_30_direction_learnable_2024_to_2025",
        "router_dir": base / "07_alpha5_router_v5_2024_to_2025",
        "score_dir": base / "08_alpha5_direction_router_rl_2024_to_2025",
        "manifest": base / "build_manifest.json",
    }


def main() -> int:
    args = parse_args()
    for src in (
        args.features_2024,
        args.rl_base_2024,
        args.rl_2025_unified,
        args.clean4_2024,
        args.pred4_2024,
        args.clean4_2025,
        args.pred4_2025,
        args.clean4_report,
        args.router_feature_list_json,
    ):
        if not src.exists():
            raise FileNotFoundError(src)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    p = _step_paths(args.out_dir)
    for key, path in p.items():
        if key.endswith("_dir"):
            path.mkdir(parents=True, exist_ok=True)

    tp_sl_train = p["tp_sl_dir"] / args.rl_2024_unified.name
    tp_sl_eval = p["tp_sl_dir"] / args.rl_2025_unified.name
    tp_sl_audit = p["tp_sl_dir"] / "tp_sl_path_edge_feature_audit.json"

    fixed_current = p["fixed_dir"] / "trade_candidates_2024_regime4_state24_sticky090_tp18_sl10_fixed.csv"
    fixed_next = p["fixed_dir"] / "trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
    fixed_manifest = p["fixed_dir"] / "fixed_regime4_state24_sticky090_tp18_sl10_preprocess_manifest.json"

    a27_train = p["alpha5_27_dir"] / "alpha5_27_label_factory_train.parquet"
    a27_val = p["alpha5_27_dir"] / "alpha5_27_label_factory_val.parquet"
    a27_oos = p["alpha5_27_dir"] / "alpha5_27_label_factory_oos.parquet"

    a28_train = p["alpha5_28_dir"] / "alpha5_28_label_factory_train.parquet"
    a28_val = p["alpha5_28_dir"] / "alpha5_28_label_factory_val.parquet"
    a28_oos = p["alpha5_28_dir"] / "alpha5_28_label_factory_oos.parquet"

    a29_train = p["alpha5_29_dir"] / "alpha5_29_hier_label_factory_train.parquet"
    a29_val = p["alpha5_29_dir"] / "alpha5_29_hier_label_factory_val.parquet"
    a29_oos = p["alpha5_29_dir"] / "alpha5_29_hier_label_factory_oos.parquet"

    a30_train = p["alpha5_30_dir"] / "alpha5_30_direction_learnable_train.parquet"
    a30_val = p["alpha5_30_dir"] / "alpha5_30_direction_learnable_val.parquet"
    a30_oos = p["alpha5_30_dir"] / "alpha5_30_direction_learnable_oos.parquet"

    router_model = p["router_dir"] / "router3_catboost_gpu.cbm"
    router_meta = p["router_dir"] / "router_ensemble_meta.joblib"
    router_summary = p["router_dir"] / "router5_summary.json"

    score_2025 = p["score_dir"] / "rl_training_2025_direction_router.csv"
    score_2025_summary = score_2025.with_suffix(".router_summary.json")

    aux_parquets = [a29_train, a29_val, a29_oos, a30_train, a30_val, a30_oos]

    startup = {
        "model_id": MODEL_ID,
        "python": sys.executable,
        "rl_2024_unified": str(args.rl_2024_unified),
        "rl_2025_unified": str(args.rl_2025_unified),
        "fixed_current": str(fixed_current),
        "fixed_next": str(fixed_next),
        "router_model": str(router_model),
        "router_meta": str(router_meta),
        "score_2025": str(score_2025),
        "out_dir": str(args.out_dir),
        "router_feature_list_json": str(args.router_feature_list_json),
    }
    if args.startup_check_only:
        print(json.dumps(startup, ensure_ascii=False, indent=2), flush=True)
        return 0

    if args.force or not args.rl_2024_unified.exists():
        _run_py(
            "pipeline/build_unified_rl_dataset.py",
            "--features-path",
            str(args.features_2024),
            "--rl-path",
            str(args.rl_base_2024),
            "--output-path",
            str(args.rl_2024_unified),
            "--checkpoint-dir",
            str(args.unified_2024_ckpt),
        )

    if args.force or not _exists_all([tp_sl_train, tp_sl_eval, tp_sl_audit]):
        _run_py(
            "scripts/build_alpha4_tp_sl_path_edge_feature_20260517.py",
            "--train-csv",
            str(args.rl_2024_unified),
            "--eval-csv",
            str(args.rl_2025_unified),
            "--out-dir",
            str(p["tp_sl_dir"]),
        )

    if args.force or not _exists_all([fixed_current, fixed_next, fixed_manifest]):
        _run_py(
            "scripts/build_fixed_regime4_tp_sl_preprocess_20260517.py",
            "--train-csv",
            str(tp_sl_train),
            "--eval-csv",
            str(tp_sl_eval),
            "--clean4-2025",
            str(args.clean4_2024),
            "--pred4-2025",
            str(args.pred4_2024),
            "--clean4-2026",
            str(args.clean4_2025),
            "--pred4-2026",
            str(args.pred4_2025),
            "--out-dir",
            str(p["fixed_dir"]),
            "--train-out",
            str(fixed_current),
            "--eval-out",
            str(fixed_next),
            "--manifest-out",
            str(fixed_manifest),
        )

    if args.force or not _exists_all([a27_train, a27_val, a27_oos]):
        _run_py(
            "scripts/build_alpha5_27_label_factory_20260519.py",
            "--train-2025-csv",
            str(fixed_current),
            "--oos-2026-csv",
            str(fixed_next),
            "--manifest",
            str(fixed_manifest),
            "--clean4-report",
            str(args.clean4_report),
            "--out-dir",
            str(p["alpha5_27_dir"]),
            "--train-end",
            str(args.train_end),
            "--val-start",
            str(args.val_start),
            "--val-end",
            str(args.val_end),
            "--oos-start",
            str(args.oos_start),
        )

    if args.force or not _exists_all([a28_train, a28_val, a28_oos]):
        _run_py(
            "scripts/build_alpha5_28_label_factory_split_ambiguous_20260519.py",
            "--in-dir",
            str(p["alpha5_27_dir"]),
            "--out-dir",
            str(p["alpha5_28_dir"]),
        )

    if args.force or not _exists_all([a29_train, a29_val, a29_oos]):
        _run_py(
            "scripts/build_alpha5_29_hier_label_factory_20260519.py",
            "--base-dir",
            str(p["alpha5_27_dir"]),
            "--split-dir",
            str(p["alpha5_28_dir"]),
            "--out-dir",
            str(p["alpha5_29_dir"]),
        )

    if args.force or not _exists_all([a30_train, a30_val, a30_oos]):
        _run_py(
            "scripts/build_alpha5_30_direction_learnable_20260519.py",
            "--in-dir",
            str(p["alpha5_29_dir"]),
            "--out-dir",
            str(p["alpha5_30_dir"]),
        )

    if args.force or not _exists_all([router_model, router_meta, router_summary]):
        _run_py(
            "scripts/alpha5_router_v5_train_20260520.py",
            "--data-dir",
            str(p["alpha5_29_dir"]),
            "--out-dir",
            str(p["router_dir"]),
            "--devices",
            str(args.devices),
            "--seed",
            str(args.seed),
            "--weight-router3",
            str(args.weight_router3),
            "--weight-router4",
            str(args.weight_router4),
            "--raw-2025-csv",
            str(fixed_current),
            "--raw-2026-csv",
            str(fixed_next),
            "--manifest",
            str(fixed_manifest),
            "--clean4-report",
            str(args.clean4_report),
            "--feature-list-json",
            str(args.router_feature_list_json),
        )

    if args.force or not _exists_all([score_2025, score_2025_summary]):
        score_args = [
            "--input-csv",
            str(args.rl_2025_unified),
            "--output-csv",
            str(score_2025),
            "--prefix",
            str(args.prefix),
            "--router-model",
            str(router_model),
            "--router-meta",
            str(router_meta),
        ]
        for aux in aux_parquets:
            score_args.extend(["--aux-parquet", str(aux)])
        _run_py("scripts/alpha5_direction_router_score_rl_csv_20260519.py", *score_args)

    dsac_cmd = [
        sys.executable,
        str(ROOT / "scripts/alpha5_dsac_single_router5_density_20260520.py"),
        "--rl-2025",
        str(args.rl_2025_unified),
        "--router-dir",
        str(p["score_dir"]),
        "--router-model",
        str(router_model),
        "--router-meta",
        str(router_meta),
    ]
    for aux in aux_parquets:
        dsac_cmd.extend(["--router-aux-parquet", str(aux)])

    manifest = {
        "model_id": MODEL_ID,
        "status": "ok",
        "selection_contract": {
            "train_window": f"{args.train_end} split inside current-year frame",
            "validation_window": f"{args.val_start}..{args.val_end}",
            "oos_window": f"{args.oos_start}+",
            "router_fit": "2024-only",
            "router_score_target": "2025 RL unified CSV",
        },
        "inputs": {
            "features_2024": str(args.features_2024),
            "rl_base_2024": str(args.rl_base_2024),
            "rl_2024_unified": str(args.rl_2024_unified),
            "rl_2025_unified": str(args.rl_2025_unified),
            "clean4_2024": str(args.clean4_2024),
            "pred4_2024": str(args.pred4_2024),
            "clean4_2025": str(args.clean4_2025),
            "pred4_2025": str(args.pred4_2025),
            "clean4_report": str(args.clean4_report),
        },
        "artifacts": {
            "tp_sl_dir": str(p["tp_sl_dir"]),
            "fixed_current": str(fixed_current),
            "fixed_next": str(fixed_next),
            "fixed_manifest": str(fixed_manifest),
            "alpha5_27_dir": str(p["alpha5_27_dir"]),
            "alpha5_28_dir": str(p["alpha5_28_dir"]),
            "alpha5_29_dir": str(p["alpha5_29_dir"]),
            "alpha5_30_dir": str(p["alpha5_30_dir"]),
            "router_dir": str(p["router_dir"]),
            "router_model": str(router_model),
            "router_meta": str(router_meta),
            "router_summary": str(router_summary),
            "router_feature_list_json": str(args.router_feature_list_json),
            "score_dir": str(p["score_dir"]),
            "score_2025": str(score_2025),
            "score_2025_summary": str(score_2025_summary),
            "aux_parquets": [str(x) for x in aux_parquets],
        },
        "suggested_dsac_command": dsac_cmd,
    }
    p["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(p["manifest"]), "score_2025": str(score_2025)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
