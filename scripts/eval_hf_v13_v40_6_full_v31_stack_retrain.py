#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chronos import Chronos2Pipeline  # noqa: E402
from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    build_training_set,
    predict_policy_frame,
    prepare_features,
)
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import train_eval_hf_v13_jackpot_runner_v21_2 as v21  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _audit_contract,
    _close,
    _feature_cols,
    _json_default,
    _read,
)
from scripts.train_eval_hf_v13_multitrack_foundation_parent_v40 import (  # noqa: E402
    CHRONOS_MODEL,
    KAIROS_MODEL,
    MACRO_COLS,
    MACRO_LEN,
    MICRO_LEN,
    _embedding_cache_path,
    _extract_macro_embeddings,
    _extract_micro_embeddings,
    _parent_cfg,
)
from tsfm.model.kairos import AutoModel as KairosAutoModel  # noqa: E402


MODEL_ID = "hf_v13_v40_6_full_v31_stack_retrain_20260512"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_PARENT = ROOT / "data/ensemble/supervised/hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512/target_aware_full_bundle.pkl"
DEFAULT_PARENT_REPORT = ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512_summary.json"
DEFAULT_BASELINE_REPORT = ROOT / "data/ensemble/reports/hf_v13_v31_parent_swap_v40_20260512_summary.json"
DEFAULT_V27 = ROOT / "data/ensemble/supervised/hf_v13_deep_alpha_candidate_expansion_v27_20260511/v27_deep_alpha_candidate_expansion.pt"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v40_6_full_v31_stack_retrain_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v40_6_full_v31_stack_retrain_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v40_6_full_v31_stack_retrain_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v40_6_full_v31_stack_retrain_20260512_grid.csv"
V40_6_CACHE_CONSUMER = "hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512"


def _load_bundle(path: Path) -> dict[str, Any]:
    try:
        obj = joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path} did not contain a dict bundle")
    return obj


def _projection_targets(y: dict[str, np.ndarray]) -> np.ndarray:
    action = np.asarray(y["action"], dtype=np.int64)
    side = np.where(action == ACTION_LONG, 1.0, np.where(action == ACTION_SHORT, -1.0, 0.0)).astype(np.float32)
    quality = np.asarray(y["quality"], dtype=np.float32)
    signed_quality = side * np.clip(np.abs(quality), 0.0, None)
    trade_flag = (action != ACTION_CASH).astype(np.float32)
    return np.column_stack([side, signed_quality, trade_flag]).astype(np.float32)


def _fit_pls(train_x: np.ndarray, train_y: np.ndarray, apply_x: np.ndarray, n_components: int) -> tuple[np.ndarray, PLSRegression]:
    model = PLSRegression(n_components=int(n_components), scale=True)
    model.fit(train_x, train_y)
    scores = model.transform(apply_x)
    if scores.ndim == 1:
        scores = scores[:, None]
    return scores.astype(np.float32), model


def _add_factor_cols(base: pd.DataFrame, prefix: str, values: np.ndarray) -> pd.DataFrame:
    out = base.reset_index(drop=True).copy()
    for j in range(values.shape[1]):
        out[f"{prefix}_{j:03d}"] = values[:, j].astype(np.float32)
    return out


def _cache_path(
    emb_dir: Path,
    *,
    prefix: str,
    model_name: str,
    frame: pd.DataFrame,
    indices: np.ndarray,
    cols: list[str],
    window_len: int,
    train_csv: str,
    eval_csv: str,
    split: str,
    stride: int | None = None,
) -> Path:
    csv_name = eval_csv if split == "eval" else train_csv
    stride_part = f"stride={stride}|" if stride is not None else ""
    return _embedding_cache_path(
        emb_dir,
        prefix=prefix,
        model_name=model_name,
        frame=frame,
        indices=indices,
        cols=cols,
        window_len=window_len,
        extra_tag=f"csv={csv_name}|{stride_part}split={split}|consumer={V40_6_CACHE_CONSUMER}",
    )


def _need_models(paths: dict[str, Path]) -> bool:
    return not all(p.exists() for p in paths.values())


def _load_or_extract(
    *,
    paths: dict[str, Path],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    eval_feat: pd.DataFrame,
    train_idx_sample: np.ndarray,
    train_idx_full: np.ndarray,
    val_idx: np.ndarray,
    eval_idx: np.ndarray,
    micro_cols: list[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    if _need_models(paths):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{MODEL_ID}] loading Chronos/Kairos on {device} for full-train factors", flush=True)
        chronos = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL, device_map=device)
        kairos = KairosAutoModel.from_pretrained(KAIROS_MODEL, trust_remote_code=True).to(device).eval()
    else:
        chronos = None
        kairos = None

    def macro(key: str, df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
        if paths[key].exists():
            return np.load(paths[key])
        assert chronos is not None
        return _extract_macro_embeddings(chronos, df, idx, cache_path=paths[key], batch_size=batch_size)

    def micro(key: str, df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
        if paths[key].exists():
            return np.load(paths[key])
        assert kairos is not None
        return _extract_micro_embeddings(kairos, df, idx, micro_cols, cache_path=paths[key], batch_size=batch_size)

    return {
        "train_sample_macro": macro("train_sample_macro", train_df, train_idx_sample),
        "train_full_macro": macro("train_full_macro", train_df, train_idx_full),
        "val_macro": macro("val_macro", val_df, val_idx),
        "eval_macro": macro("eval_macro", eval_df, eval_idx),
        "train_sample_micro": micro("train_sample_micro", train_feat, train_idx_sample),
        "train_full_micro": micro("train_full_micro", train_feat, train_idx_full),
        "val_micro": micro("val_micro", val_feat, val_idx),
        "eval_micro": micro("eval_micro", eval_feat, eval_idx),
    }


def _build_v40_6_frames(
    *,
    args: argparse.Namespace,
    parent_report: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    eval_feat: pd.DataFrame,
    train_idx_sample: np.ndarray,
    proj_targets: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    comp = parent_report["comparison"]
    spec = dict(comp["selected_projection_spec"])
    macro_dim = int(spec["macro_dim"])
    micro_dim = int(spec["micro_dim"])
    macro_cols = list(parent_report.get("macro_cols") or MACRO_COLS)
    micro_cols = list(parent_report.get("micro_cols") or [])
    train_idx_full = np.arange(len(train_df), dtype=np.int64)
    val_idx = np.arange(len(val_df), dtype=np.int64)
    eval_idx = np.arange(len(eval_df), dtype=np.int64)
    emb_dir = ROOT / "data/ensemble/supervised/hf_v13_multitrack_foundation_parent_v40_20260512" / "embeddings"
    paths = {
        "train_sample_macro": _cache_path(
            emb_dir,
            prefix="tree_train_macro",
            model_name=CHRONOS_MODEL,
            frame=train_df,
            indices=train_idx_sample,
            cols=[c for c in macro_cols if c in train_df.columns],
            window_len=MACRO_LEN,
            train_csv=args.train_csv.name,
            eval_csv=args.eval_csv.name,
            split="train",
            stride=int(args.train_stride),
        ),
        "train_full_macro": _cache_path(
            emb_dir,
            prefix="full_train_macro",
            model_name=CHRONOS_MODEL,
            frame=train_df,
            indices=train_idx_full,
            cols=[c for c in macro_cols if c in train_df.columns],
            window_len=MACRO_LEN,
            train_csv=args.train_csv.name,
            eval_csv=args.eval_csv.name,
            split="train_full",
        ),
        "val_macro": _cache_path(
            emb_dir,
            prefix="val_macro",
            model_name=CHRONOS_MODEL,
            frame=val_df,
            indices=val_idx,
            cols=[c for c in macro_cols if c in val_df.columns],
            window_len=MACRO_LEN,
            train_csv=args.train_csv.name,
            eval_csv=args.eval_csv.name,
            split="val",
        ),
        "eval_macro": _cache_path(
            emb_dir,
            prefix="eval_macro",
            model_name=CHRONOS_MODEL,
            frame=eval_df,
            indices=eval_idx,
            cols=[c for c in macro_cols if c in eval_df.columns],
            window_len=MACRO_LEN,
            train_csv=args.train_csv.name,
            eval_csv=args.eval_csv.name,
            split="eval",
        ),
        "train_sample_micro": _cache_path(
            emb_dir,
            prefix="tree_train_micro",
            model_name=KAIROS_MODEL,
            frame=train_feat,
            indices=train_idx_sample,
            cols=micro_cols,
            window_len=MICRO_LEN,
            train_csv=args.train_csv.name,
            eval_csv=args.eval_csv.name,
            split="train",
            stride=int(args.train_stride),
        ),
        "train_full_micro": _cache_path(
            emb_dir,
            prefix="full_train_micro",
            model_name=KAIROS_MODEL,
            frame=train_feat,
            indices=train_idx_full,
            cols=micro_cols,
            window_len=MICRO_LEN,
            train_csv=args.train_csv.name,
            eval_csv=args.eval_csv.name,
            split="train_full",
        ),
        "val_micro": _cache_path(
            emb_dir,
            prefix="val_micro",
            model_name=KAIROS_MODEL,
            frame=val_feat,
            indices=val_idx,
            cols=micro_cols,
            window_len=MICRO_LEN,
            train_csv=args.train_csv.name,
            eval_csv=args.eval_csv.name,
            split="val",
        ),
        "eval_micro": _cache_path(
            emb_dir,
            prefix="eval_micro",
            model_name=KAIROS_MODEL,
            frame=eval_feat,
            indices=eval_idx,
            cols=micro_cols,
            window_len=MICRO_LEN,
            train_csv=args.train_csv.name,
            eval_csv=args.eval_csv.name,
            split="eval",
        ),
    }
    emb = _load_or_extract(
        paths=paths,
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        train_feat=train_feat,
        val_feat=val_feat,
        eval_feat=eval_feat,
        train_idx_sample=train_idx_sample,
        train_idx_full=train_idx_full,
        val_idx=val_idx,
        eval_idx=eval_idx,
        micro_cols=micro_cols,
        batch_size=int(args.embed_batch),
    )
    train_sample_macro_f, macro_pls = _fit_pls(emb["train_sample_macro"], proj_targets, emb["train_sample_macro"], macro_dim)
    train_full_macro_f = macro_pls.transform(emb["train_full_macro"]).astype(np.float32)
    val_macro_f = macro_pls.transform(emb["val_macro"]).astype(np.float32)
    eval_macro_f = macro_pls.transform(emb["eval_macro"]).astype(np.float32)
    train_sample_micro_f, micro_pls = _fit_pls(emb["train_sample_micro"], proj_targets, emb["train_sample_micro"], micro_dim)
    train_full_micro_f = micro_pls.transform(emb["train_full_micro"]).astype(np.float32)
    val_micro_f = micro_pls.transform(emb["val_micro"]).astype(np.float32)
    eval_micro_f = micro_pls.transform(emb["eval_micro"]).astype(np.float32)
    train_full = _add_factor_cols(train_df, "macro_factor", train_full_macro_f)
    train_full = _add_factor_cols(train_full, "micro_factor", train_full_micro_f)
    val_full = _add_factor_cols(val_df, "macro_factor", val_macro_f)
    val_full = _add_factor_cols(val_full, "micro_factor", val_micro_f)
    eval_full = _add_factor_cols(eval_df, "macro_factor", eval_macro_f)
    eval_full = _add_factor_cols(eval_full, "micro_factor", eval_micro_f)
    meta = {
        "projection_spec": spec,
        "macro_cols": macro_cols,
        "micro_cols": micro_cols,
        "cache_consumer": V40_6_CACHE_CONSUMER,
        "embedding_cache_paths": {k: str(v) for k, v in paths.items()},
        "train_sample_factor_shapes": {"macro": list(train_sample_macro_f.shape), "micro": list(train_sample_micro_f.shape)},
        "train_full_factor_shapes": {"macro": list(train_full_macro_f.shape), "micro": list(train_full_micro_f.shape)},
    }
    return train_full, val_full, eval_full, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain V21.2 runner and V31 overlay around v40.6 encoded parent.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--parent-report", type=Path, default=DEFAULT_PARENT_REPORT)
    p.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--train-stride", type=int, default=48)
    p.add_argument("--embed-batch", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[{MODEL_ID}] loading data and v40.6 parent", flush=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    parent_bundle = _load_bundle(args.parent_model)
    with args.parent_report.open(encoding="utf-8") as f:
        parent_report = json.load(f)
    feature_cols = _feature_cols(train_all, eval_df)
    cfg = _parent_cfg()
    print(f"[{MODEL_ID}] building train labels for PLS projection", flush=True)
    x_train, y, training_meta = build_training_set(train_df, cfg=cfg, stride_bars=int(args.train_stride), batch_size=512, feature_cols=feature_cols)
    train_idx_sample = np.arange(0, max(0, len(train_df) - cfg.max_train_horizon_bars - 1), max(1, int(args.train_stride)), dtype=np.int64)
    if len(train_idx_sample) != len(x_train):
        raise RuntimeError(f"train sample mismatch: {len(train_idx_sample)} vs {len(x_train)}")
    proj_targets = _projection_targets(y)
    print(f"[{MODEL_ID}] preparing base features and encoded frames", flush=True)
    train_feat = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    val_feat = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_feat = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    train_full, val_full, eval_full, encoding_meta = _build_v40_6_frames(
        args=args,
        parent_report=parent_report,
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        train_feat=train_feat,
        val_feat=val_feat,
        eval_feat=eval_feat,
        train_idx_sample=train_idx_sample,
        proj_targets=proj_targets,
    )
    print(f"[{MODEL_ID}] training V21.2 runner on v40.6 parent decisions", flush=True)
    base = dict(parent_bundle.get("config", {}))
    fee = float(base.get("fee", cfg.fee))
    slip = float(base.get("slip", cfg.slip))
    runner_model = v21._fit_cost_runner(train_full, parent_bundle, fee=fee, slip=slip)
    val_dec = predict_policy_frame(parent_bundle, val_full, close=_close(val_full))
    eval_dec = predict_policy_frame(parent_bundle, eval_full, close=_close(eval_full))
    print(f"[{MODEL_ID}] predicting frozen V27 utilities", flush=True)
    v27_payload, v27_model = v31._load_v27(args.v27_model)
    val_q = v31._predict_all(v27_model, val_full, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_full, v27_payload["seq_cols"], v27_payload["norm"])

    print(f"[{MODEL_ID}] joint-selecting V21.2 config and V31 overlay on 2025 Q4", flush=True)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for add_cfg in v21._grid():
        for overlay in v31._grid():
            v1 = v31.backtest(val_full, parent_bundle, runner_model, add_cfg, val_q, overlay, fee=fee, slip=slip, cost_mult=1.0, decisions=val_dec)
            v2 = v31.backtest(val_full, parent_bundle, runner_model, add_cfg, val_q, overlay, fee=fee, slip=slip, cost_mult=2.0, decisions=val_dec)
            v3 = v31.backtest(val_full, parent_bundle, runner_model, add_cfg, val_q, overlay, fee=fee, slip=slip, cost_mult=3.0, decisions=val_dec)
            score = v31._score(v1, v2, v3)
            row = {
                "runner_config": asdict(add_cfg),
                "overlay_config": asdict(overlay),
                "runner": add_cfg.name,
                "overlay": overlay.name,
                "selection_score": score,
                "validation_cost1": v1,
                "validation_cost2": v2,
                "validation_cost3": v3,
            }
            rows.append(row)
            if best is None or score > best["selection_score"]:
                best = row
    assert best is not None
    selected_add = v21.CostRunnerConfig(**best["runner_config"])
    selected_overlay = v31.OverlayConfig(**best["overlay_config"])
    print(f"[{MODEL_ID}] evaluating 2026 OOS", flush=True)
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = v31.backtest(
            eval_full,
            parent_bundle,
            runner_model,
            selected_add,
            eval_q,
            selected_overlay,
            fee=fee,
            slip=slip,
            cost_mult=float(mult),
            decisions=eval_dec,
            record=(mult == 1),
        )
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runner_path = args.out_dir / "v40_6_retrained_v21_2_runner.pkl"
    joblib.dump({"model_id": MODEL_ID, "base_parent": str(args.parent_model), "cost_runner": runner_model, "selected_config": asdict(selected_add)}, runner_path)
    manifest_path = args.out_dir / "v40_6_full_v31_stack_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "parent_model": str(args.parent_model),
                "v27_model": str(args.v27_model),
                "runner_model": str(runner_path),
                "selected_runner_config": asdict(selected_add),
                "selected_overlay": asdict(selected_overlay),
                "metrics": metrics,
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "runner": r["runner"],
                "overlay": r["overlay"],
                "selection_score": r["selection_score"],
                "val_pnl": r["validation_cost1"]["pnl"],
                "val_mdd": r["validation_cost1"]["mdd"],
                "val_trades": r["validation_cost1"]["trades"],
                "val_deep_entries": r["validation_cost1"].get("deep_entries", 0),
                "val_c2_pnl": r["validation_cost2"]["pnl"],
                "val_c3_pnl": r["validation_cost3"]["pnl"],
                "val_adds": r["validation_cost1"].get("runner_actions", {}).get("v21_add_on", 0),
                "val_rejects": r["validation_cost1"].get("runner_actions", {}).get("v21_reject", 0),
            }
            for r in rows
        ]
    ).sort_values("selection_score", ascending=False).to_csv(args.grid_out, index=False)

    feature_audit_cols = [c for c in list(parent_bundle.get("feature_cols") or []) if not c.startswith("macro_factor_") and not c.startswith("micro_factor_")]
    feature_audit = _audit_contract(train_all, eval_df, feature_audit_cols)
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit.get("status") != "pass":
        blocking.extend(feature_audit.get("blocking", []))
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    audit = {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31 joint runner/overlay selection",
        "oos_window": "2026 fixed OOS only after selection",
        "parent": "v40_6_hgb_pls_encoded_parent",
        "v21_2_runner_retrained": True,
        "v27_entry_frozen": True,
        "v31_overlay_reselected": True,
        "feature_audit": feature_audit,
        "runner_meta": {k: v for k, v in runner_model.items() if k not in {"regressor", "q10_regressor", "q90_regressor", "classifier", "jackpot_classifier", "bad_classifier", "cost3_classifier", "feature_cols"}},
    }
    baseline = {}
    if args.baseline_report.exists():
        with args.baseline_report.open(encoding="utf-8") as f:
            baseline = json.load(f).get("summary", [])
    report = {
        "model_id": MODEL_ID,
        "design": "Full-stack v40.6/V31 compatibility test: regenerate v40.6 encoded factors, retrain V21.2 cost-stressed jackpot runner on v40.6 parent decisions, jointly select runner config and V31 overlay on 2025 Q4, then evaluate fixed 2026 OOS. V27 is kept frozen for this first full-stack retrain.",
        "parent_model": str(args.parent_model),
        "parent_report": str(args.parent_report),
        "encoding_meta": encoding_meta,
        "training_meta": training_meta,
        "runner_model": str(runner_path),
        "selected_runner_config": asdict(selected_add),
        "selected_overlay": asdict(selected_overlay),
        "selection_result": best,
        "metrics": metrics,
        "baseline_parent_swap_summary": baseline,
        "audit": audit,
        "artifacts": {
            "manifest": str(manifest_path),
            "runner_model": str(runner_path),
            "report": str(args.report_out),
            "audit": str(args.audit_out),
            "grid": str(args.grid_out),
            "ledgers": ledgers,
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "audit": str(args.audit_out),
                "grid": str(args.grid_out),
                "runner_model": str(runner_path),
                "selected_runner": asdict(selected_add),
                "selected_overlay": asdict(selected_overlay),
                "metrics": metrics,
                "audit_status": audit["status"],
            },
            ensure_ascii=False,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
