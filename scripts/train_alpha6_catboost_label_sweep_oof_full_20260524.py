#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha6_catboost_5head_policy_20260522 import (  # noqa: E402
    DEFAULT_FEATURE_CSV,
    DEFAULT_LABEL_DIR,
    DEFAULT_SPEC_DIR,
    _label_frame,
    _numeric_matrix,
    _read_feature_frame,
    _read_spec,
)
from scripts.alpha6_catboost_entry_quality_exit_policy_20260522 import (  # noqa: E402
    CONTEXT_COLS,
    EQEConfig,
    _apply_label_preset,
    _build_entry_labels,
    _build_exit_dataset,
    _estimate_expected_return_by_bucket,
    _fit_entry_models,
    _fit_exit_model,
    _predict_entry,
)


TRAIN_SCRIPT = ROOT / "scripts/alpha6_catboost_entry_quality_exit_policy_20260522.py"

DEFAULT_CANDIDATES = (
    "current_quality:bucket5",
    "density_balanced:bucket5",
    "scalp_short_horizon:horizon_reg",
    "perturbation_robust:bucket5",
    "adverse_conformal:bucket5",
    "sam_conformal:bucket5",
    "high_precision_robust:bucket5",
    "turnover_balanced_robust:bucket5",
    "short_horizon_robust:horizon_reg",
    "regime_conditional:bucket5",
    "pullback_entry:horizon_reg",
    "ts2vec_ood_proxy:bucket5",
    "diffusion_stress_proxy:bucket5",
    "psr_path_quality:bucket5",
    "ts2vec_ood:bucket5",
    "cost_beta_neutral:bucket5",
    "mamba_regime_filter:bucket5",
    "timegrad_mc:bucket5",
    "timellm_uncertainty:bucket5",
)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)


def _parse_candidates(raw: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        parts = token.split(":")
        preset = parts[0]
        mode = parts[1] if len(parts) > 1 and parts[1] else "bucket5"
        fixed = int(parts[2]) if len(parts) > 2 and parts[2] else 0
        name = preset if mode == "bucket5" else f"{preset}_{mode}"
        if mode == "fixed":
            name = f"{preset}_fixed{fixed}"
        out.append({"name": _safe_name(name), "preset": preset, "mode": mode, "fixed": fixed})
    return out


def _namespace(args: argparse.Namespace, *, task_type: str, seed: int, cfg: EQEConfig, mode: str, fixed: int) -> argparse.Namespace:
    fixed_horizon = int(fixed)
    if mode == "fixed" and fixed_horizon <= 0:
        fixed_horizon = int(cfg.score_horizons[-1])
    return argparse.Namespace(
        iterations=int(args.oof_iterations),
        learning_rate=float(args.learning_rate),
        depth=int(args.depth),
        l2_leaf_reg=float(args.l2_leaf_reg),
        exit_iterations=int(args.oof_exit_iterations),
        exit_learning_rate=float(args.exit_learning_rate),
        exit_depth=int(args.exit_depth),
        task_type=str(task_type),
        seed=int(seed),
        verbose=0,
        target_head_mode=str(mode),
        fixed_target_horizon=int(fixed_horizon),
        max_target_horizon=int(cfg.max_train_horizon_bars),
        cash_action_weight=float(args.cash_action_weight),
    )


def _train_oof_fold(
    train: pd.DataFrame,
    spec_features: list[str],
    fit_pos: np.ndarray,
    fold_pos: np.ndarray,
    *,
    args: argparse.Namespace,
    preset: str,
    mode: str,
    fixed: int,
    seed: int,
    task_type: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    cfg = _apply_label_preset(replace(EQEConfig(), fixed_notional=float(args.fixed_notional)), preset)
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    pipe.fit(_numeric_matrix(train.iloc[fit_pos], spec_features))
    x_all = pipe.transform(_numeric_matrix(train, spec_features))
    valid, y, label_meta = _build_entry_labels(
        train,
        cfg,
        stride_bars=int(args.stride_bars),
        batch_size=int(args.batch_size),
        adaptive_sampling=False,
        label_preset=preset,
        session_topk=int(args.session_topk),
    )
    fit_set = set(int(v) for v in fit_pos)
    keep = np.asarray([int(v) in fit_set for v in valid], dtype=bool)
    valid_fit = valid[keep]
    y_fit = {k: v[keep] if len(v) == len(valid) else v for k, v in y.items()}
    ns = _namespace(args, task_type=task_type, seed=seed, cfg=cfg, mode=mode, fixed=fixed)
    entry_models = _fit_entry_models(x_all[valid_fit], y_fit, ns)
    train_dec = _predict_entry(entry_models, x_all, cfg)
    expected = _estimate_expected_return_by_bucket(train, valid_fit, y_fit, cfg)
    x_exit, y_exit, w_exit, exit_meta = _build_exit_dataset(
        train,
        x_all,
        valid_fit,
        y_fit,
        train_dec,
        cfg,
        max_samples=int(args.oof_exit_max_trades),
        step=int(args.oof_exit_step),
        cost_mult=3.0,
        weight_scale=float(args.exit_weight_scale),
        target_head_mode=str(mode),
        expected_return_by_bucket=expected,
    )
    exit_model = _fit_exit_model(x_exit, y_exit, w_exit, ns)
    fold_dec = _predict_entry(entry_models, x_all[fold_pos], cfg)
    fold_dec.insert(0, "row_pos", fold_pos.astype(np.int64))
    fold_dec.insert(1, "timestamp", train.iloc[fold_pos]["timestamp"].to_numpy())
    meta = {
        "label_meta": label_meta,
        "entry_label_distribution": entry_models["label_distribution"],
        "exit_meta": exit_meta,
        "fit_rows": int(len(fit_pos)),
        "fold_rows": int(len(fold_pos)),
        "exit_samples": int(len(y_exit)),
        "exit_close_rate": float(np.mean(y_exit)) if len(y_exit) else 0.0,
        "task_type": str(task_type),
    }
    _ = exit_model
    return fold_dec, meta


def _run_final_with_fallback(candidate: dict[str, Any], args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    base_cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--variant",
        str(args.variant),
        "--out-dir",
        str(out_dir),
        "--label-preset",
        str(candidate["preset"]),
        "--target-head-mode",
        str(candidate["mode"]),
        "--iterations",
        str(args.final_iterations),
        "--learning-rate",
        str(args.learning_rate),
        "--depth",
        str(args.depth),
        "--l2-leaf-reg",
        str(args.l2_leaf_reg),
        "--exit-iterations",
        str(args.final_exit_iterations),
        "--exit-learning-rate",
        str(args.exit_learning_rate),
        "--exit-depth",
        str(args.exit_depth),
        "--entry-thresholds",
        str(args.entry_thresholds),
        "--exit-max-trades",
        str(args.exit_max_trades),
        "--exit-step",
        str(args.exit_step),
        "--eval-costs",
        str(args.eval_costs),
        "--exit-threshold-grid",
        str(args.exit_threshold_grid),
        "--fixed-notional",
        str(args.fixed_notional),
        "--verbose",
        "0",
    ]
    fixed = int(candidate.get("fixed") or 0)
    if fixed > 0:
        base_cmd.extend(["--fixed-target-horizon", str(fixed)])
    if str(candidate["preset"]) == "pullback_entry":
        base_cmd.extend(["--entry-pullback-atr", "0.30"])
    if args.representation_feature_file is not None:
        base_cmd.extend(["--representation-feature-file", str(args.representation_feature_file)])
    if str(args.extra_final_args).strip():
        base_cmd.extend(str(args.extra_final_args).strip().split())

    attempts = ["GPU", "CPU"] if args.task_type.upper() == "GPU" else ["CPU"]
    last_rc = 0
    for task_type in attempts:
        cmd = [*base_cmd, "--task-type", task_type]
        log = out_dir / f"final_{task_type.lower()}.log"
        with log.open("w") as fh:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        last_rc = int(proc.returncode)
        if proc.returncode == 0:
            return {"status": "ok", "task_type": task_type, "log": str(log)}
    return {"status": f"failed:{last_rc}", "task_type": attempts[-1], "log": str(out_dir / f"final_{attempts[-1].lower()}.log")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Alpha6 CatBoost label presets with purged 2-fold OOF and final full-train bundles.")
    ap.add_argument("--variant", default="current_tail111")
    ap.add_argument("--out-root", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha6_label_oof_full_20260524")
    ap.add_argument("--representation-feature-file", type=Path, default=None)
    ap.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--purge-bars", type=int, default=96)
    ap.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stride-bars", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--session-topk", type=int, default=2)
    ap.add_argument("--fixed-notional", type=float, default=0.25)
    ap.add_argument("--learning-rate", type=float, default=0.045)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--l2-leaf-reg", type=float, default=7.0)
    ap.add_argument("--exit-learning-rate", type=float, default=0.035)
    ap.add_argument("--exit-depth", type=int, default=5)
    ap.add_argument("--cash-action-weight", type=float, default=0.35)
    ap.add_argument("--exit-weight-scale", type=float, default=80.0)
    ap.add_argument("--oof-iterations", type=int, default=260)
    ap.add_argument("--oof-exit-iterations", type=int, default=120)
    ap.add_argument("--oof-exit-max-trades", type=int, default=2500)
    ap.add_argument("--oof-exit-step", type=int, default=6)
    ap.add_argument("--final-iterations", type=int, default=650)
    ap.add_argument("--final-exit-iterations", type=int, default=500)
    ap.add_argument("--entry-thresholds", type=int, default=50)
    ap.add_argument("--exit-max-trades", type=int, default=9000)
    ap.add_argument("--exit-step", type=int, default=2)
    ap.add_argument("--eval-costs", default="1,2,3")
    ap.add_argument("--exit-threshold-grid", default="0.35,0.45,0.55,0.70")
    ap.add_argument("--extra-final-args", default="")
    ap.add_argument("--oof-only", action="store_true")
    ap.add_argument("--final-only", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    candidates = _parse_candidates(args.candidates)
    spec = _read_spec(DEFAULT_SPEC_DIR, args.variant)
    feat, _, _ = _read_feature_frame(DEFAULT_FEATURE_CSV, list(spec["features"]), CONTEXT_COLS)
    if args.representation_feature_file is not None:
        rep_path = Path(args.representation_feature_file)
        if not rep_path.exists():
            raise FileNotFoundError(rep_path)
        rep = pd.read_parquet(rep_path) if rep_path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(rep_path)
        rep_cols = ["timestamp", *[c for c in rep.columns if str(c).startswith("rep_")]]
        feat = feat.merge(rep[rep_cols], on="timestamp", how="left")
    frame = feat.merge(_label_frame(DEFAULT_LABEL_DIR), on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    train = frame[frame["dataset_split"].astype(str).str.lower().eq("train")].copy().reset_index(drop=True)
    spec_features = [c for c in spec["features"] if c in train.columns]
    train_pos = np.arange(len(train), dtype=np.int64)
    fold_parts = np.array_split(train_pos, int(args.folds))
    manifest: list[dict[str, Any]] = []

    for expert_idx, candidate in enumerate(candidates):
        out_dir = args.out_root / str(candidate["name"])
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[alpha6-oof-full] candidate={candidate['name']} preset={candidate['preset']} mode={candidate['mode']}", flush=True)
        record: dict[str, Any] = {"candidate": candidate, "out_dir": str(out_dir), "oof": [], "final": None}
        if not args.final_only:
            fold_frames: list[pd.DataFrame] = []
            for fold_id, fold_pos in enumerate(fold_parts, start=1):
                lo, hi = int(fold_pos.min()), int(fold_pos.max())
                purge_lo = max(0, lo - int(args.purge_bars))
                purge_hi = min(len(train) - 1, hi + int(args.purge_bars))
                fit_pos = train_pos[(train_pos < purge_lo) | (train_pos > purge_hi)]
                task_attempts = ["GPU", "CPU"] if args.task_type.upper() == "GPU" else ["CPU"]
                fold_meta: dict[str, Any] | None = None
                fold_dec: pd.DataFrame | None = None
                for task_type in task_attempts:
                    try:
                        fold_dec, fold_meta = _train_oof_fold(
                            train,
                            spec_features,
                            fit_pos,
                            fold_pos,
                            args=args,
                            preset=str(candidate["preset"]),
                            mode=str(candidate["mode"]),
                            fixed=int(candidate.get("fixed") or 0),
                            seed=int(args.seed) + 1000 * expert_idx + fold_id,
                            task_type=task_type,
                        )
                        break
                    except Exception as exc:
                        fold_meta = {"status": "failed", "task_type": task_type, "error": repr(exc)}
                        if task_type == task_attempts[-1]:
                            break
                if fold_dec is None or fold_meta is None or fold_meta.get("status") == "failed":
                    record["oof"].append({"fold": int(fold_id), **(fold_meta or {"status": "failed"})})
                    print(
                        f"[alpha6-oof-full] {candidate['name']} fold={fold_id}/{args.folds} failed={record['oof'][-1].get('error')}",
                        flush=True,
                    )
                    if not args.keep_going:
                        raise RuntimeError(f"OOF fold failed for {candidate['name']}: {record['oof'][-1]}")
                    break
                fold_dec.to_csv(out_dir / f"oof_fold{fold_id}_predictions.csv", index=False)
                fold_frames.append(fold_dec)
                fold_meta.update({"fold": int(fold_id), "purge_lo": int(purge_lo), "purge_hi": int(purge_hi)})
                record["oof"].append(fold_meta)
                print(
                    f"[alpha6-oof-full] {candidate['name']} fold={fold_id}/{args.folds} fit_rows={len(fit_pos)} pred_rows={len(fold_pos)} task={fold_meta.get('task_type')}",
                    flush=True,
                )
            if fold_frames:
                oof = pd.concat(fold_frames, ignore_index=True).sort_values("row_pos")
                oof.to_csv(out_dir / "oof_train_predictions.csv", index=False)
            (out_dir / "oof_meta.json").write_text(json.dumps(record["oof"], ensure_ascii=False, indent=2, default=str))
        if not args.oof_only:
            record["final"] = _run_final_with_fallback(candidate, args, out_dir)
            print(f"[alpha6-oof-full] {candidate['name']} final={record['final']}", flush=True)
        manifest.append(record)
        (args.out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        final_record = record.get("final") or {}
        if final_record.get("status", "ok").startswith("failed") and not args.keep_going:
            raise RuntimeError(f"final failed for {candidate['name']}: {record['final']}")


if __name__ == "__main__":
    main()
