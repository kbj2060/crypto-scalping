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
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _audit_contract,
    _close,
    _feature_cols,
    _json_default,
    _read,
)
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402
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


MODEL_ID = "hf_v13_v31_parent_swap_v40_20260512"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_PARENT = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl"
DEFAULT_JACKPOT = ROOT / "data/ensemble/supervised/hf_v13_jackpot_runner_v21_2_20260511/v21_2_jackpot_runner.pkl"
DEFAULT_V27 = ROOT / "data/ensemble/supervised/hf_v13_deep_alpha_candidate_expansion_v27_20260511/v27_deep_alpha_candidate_expansion.pt"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v31_parent_swap_v40_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v31_parent_swap_v40_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v31_parent_swap_v40_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v31_parent_swap_v40_20260512_grid.csv"


PARENT_VARIANTS: dict[str, dict[str, Any]] = {
    "v31_original_parent": {
        "kind": "raw",
        "bundle": DEFAULT_PARENT,
        "report": ROOT / "data/ensemble/reports/hf_v13_frozen_v27_rule_exit_overlay_v31_20260511_summary.json",
    },
    "v31_v40_6_hgb_pls_parent": {
        "kind": "encoded",
        "bundle": ROOT / "data/ensemble/supervised/hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512/target_aware_full_bundle.pkl",
        "report": ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512_summary.json",
        "cache_consumer": "hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512",
        "nan_missing": False,
    },
    "v31_v40_9_lgbm_low_mdd_parent": {
        "kind": "encoded",
        "bundle": ROOT / "data/ensemble/supervised/hf_v13_tree_vs_foundation_target_aware_lgbm_native_quant_v40_9_20260512/target_aware_full_bundle.pkl",
        "report": ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_target_aware_lgbm_native_quant_v40_9_20260512_summary.json",
        "cache_consumer": "hf_v13_tree_vs_foundation_target_aware_lgbm_native_quant_v40_9_20260512",
        "nan_missing": True,
    },
    "v31_v40_10_lgbm_tradefloor_parent": {
        "kind": "encoded",
        "bundle": ROOT / "data/ensemble/supervised/hf_v13_tree_vs_foundation_target_aware_lgbm_native_quant_tradefloor_v40_10_20260512/target_aware_full_bundle.pkl",
        "report": ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_target_aware_lgbm_native_quant_tradefloor_v40_10_20260512_summary.json",
        "cache_consumer": "hf_v13_tree_vs_foundation_target_aware_lgbm_native_quant_v40_9_20260512",
        "nan_missing": True,
    },
}


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


def _source_missing_cols(source: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    return [
        c
        for c in feature_cols
        if c not in source.columns
        and c != "side_hint"
        and not c.startswith("mom_")
        and not c.startswith("abs_mom_")
    ]


def _preserve_source_missing_as_nan(prepared: pd.DataFrame, source: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = prepared.copy()
    for col in _source_missing_cols(source, feature_cols):
        if col in out.columns:
            out[col] = np.nan
    return out


def _frame_with_prepared_features(raw: pd.DataFrame, prepared: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = raw.reset_index(drop=True).copy()
    prepared_reset = prepared.reset_index(drop=True)
    for col in feature_cols:
        if col in prepared_reset.columns:
            out[col] = prepared_reset[col].to_numpy()
    return out


def _cache_paths(
    *,
    emb_dir: Path,
    train_csv_name: str,
    eval_csv_name: str,
    train_stride: int,
    consumer: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    eval_feat: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    eval_idx: np.ndarray,
    macro_cols: list[str],
    micro_cols: list[str],
) -> dict[str, Path]:
    return {
        "train_macro": _embedding_cache_path(
            emb_dir,
            prefix="tree_train_macro",
            model_name=CHRONOS_MODEL,
            frame=train_df,
            indices=train_idx,
            cols=[c for c in macro_cols if c in train_df.columns],
            window_len=MACRO_LEN,
            extra_tag=f"csv={train_csv_name}|stride={train_stride}|split=train|consumer={consumer}",
        ),
        "val_macro": _embedding_cache_path(
            emb_dir,
            prefix="val_macro",
            model_name=CHRONOS_MODEL,
            frame=val_df,
            indices=val_idx,
            cols=[c for c in macro_cols if c in val_df.columns],
            window_len=MACRO_LEN,
            extra_tag=f"csv={train_csv_name}|split=val|consumer={consumer}",
        ),
        "eval_macro": _embedding_cache_path(
            emb_dir,
            prefix="eval_macro",
            model_name=CHRONOS_MODEL,
            frame=eval_df,
            indices=eval_idx,
            cols=[c for c in macro_cols if c in eval_df.columns],
            window_len=MACRO_LEN,
            extra_tag=f"csv={eval_csv_name}|split=eval|consumer={consumer}",
        ),
        "train_micro": _embedding_cache_path(
            emb_dir,
            prefix="tree_train_micro",
            model_name=KAIROS_MODEL,
            frame=train_feat,
            indices=train_idx,
            cols=micro_cols,
            window_len=MICRO_LEN,
            extra_tag=f"csv={train_csv_name}|stride={train_stride}|split=train|consumer={consumer}",
        ),
        "val_micro": _embedding_cache_path(
            emb_dir,
            prefix="val_micro",
            model_name=KAIROS_MODEL,
            frame=val_feat,
            indices=val_idx,
            cols=micro_cols,
            window_len=MICRO_LEN,
            extra_tag=f"csv={train_csv_name}|split=val|consumer={consumer}",
        ),
        "eval_micro": _embedding_cache_path(
            emb_dir,
            prefix="eval_micro",
            model_name=KAIROS_MODEL,
            frame=eval_feat,
            indices=eval_idx,
            cols=micro_cols,
            window_len=MICRO_LEN,
            extra_tag=f"csv={eval_csv_name}|split=eval|consumer={consumer}",
        ),
    }


def _embedding_models_needed(paths: dict[str, Path]) -> bool:
    return not all(p.exists() for p in paths.values())


def _load_or_extract_embeddings(
    paths: dict[str, Path],
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    eval_feat: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    eval_idx: np.ndarray,
    micro_cols: list[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    chronos = None
    kairos = None
    if _embedding_models_needed(paths):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{MODEL_ID}] loading Chronos/Kairos on {device} for missing embedding caches", flush=True)
        chronos = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL, device_map=device)
        kairos = KairosAutoModel.from_pretrained(KAIROS_MODEL, trust_remote_code=True).to(device).eval()
    if chronos is None:
        return {k: np.load(v) for k, v in paths.items()}
    return {
        "train_macro": _extract_macro_embeddings(chronos, train_df, train_idx, cache_path=paths["train_macro"], batch_size=batch_size),
        "val_macro": _extract_macro_embeddings(chronos, val_df, val_idx, cache_path=paths["val_macro"], batch_size=batch_size),
        "eval_macro": _extract_macro_embeddings(chronos, eval_df, eval_idx, cache_path=paths["eval_macro"], batch_size=batch_size),
        "train_micro": _extract_micro_embeddings(kairos, train_feat, train_idx, micro_cols, cache_path=paths["train_micro"], batch_size=batch_size),
        "val_micro": _extract_micro_embeddings(kairos, val_feat, val_idx, micro_cols, cache_path=paths["val_micro"], batch_size=batch_size),
        "eval_micro": _extract_micro_embeddings(kairos, eval_feat, eval_idx, micro_cols, cache_path=paths["eval_micro"], batch_size=batch_size),
    }


def _build_encoded_frames(
    *,
    variant_name: str,
    variant: dict[str, Any],
    bundle: dict[str, Any],
    report: dict[str, Any],
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_feat_zero: pd.DataFrame,
    val_feat_zero: pd.DataFrame,
    eval_feat_zero: pd.DataFrame,
    train_feat_nan: pd.DataFrame,
    val_feat_nan: pd.DataFrame,
    eval_feat_nan: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    eval_idx: np.ndarray,
    proj_targets: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    comp = report["comparison"]
    spec = dict(comp["selected_projection_spec"])
    macro_dim = int(spec["macro_dim"])
    micro_dim = int(spec["micro_dim"])
    micro_cols = list(report.get("micro_cols") or [])
    macro_cols = list(report.get("macro_cols") or MACRO_COLS)
    nan_missing = bool(variant.get("nan_missing", False))
    train_feat = train_feat_nan if nan_missing else train_feat_zero
    val_feat = val_feat_nan if nan_missing else val_feat_zero
    eval_feat = eval_feat_nan if nan_missing else eval_feat_zero
    paths = _cache_paths(
        emb_dir=ROOT / "data/ensemble/supervised/hf_v13_multitrack_foundation_parent_v40_20260512" / "embeddings",
        train_csv_name=args.train_csv.name,
        eval_csv_name=args.eval_csv.name,
        train_stride=int(args.train_stride),
        consumer=str(variant["cache_consumer"]),
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        train_feat=train_feat,
        val_feat=val_feat,
        eval_feat=eval_feat,
        train_idx=train_idx,
        val_idx=val_idx,
        eval_idx=eval_idx,
        macro_cols=macro_cols,
        micro_cols=micro_cols,
    )
    emb = _load_or_extract_embeddings(
        paths,
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        train_feat=train_feat,
        val_feat=val_feat,
        eval_feat=eval_feat,
        train_idx=train_idx,
        val_idx=val_idx,
        eval_idx=eval_idx,
        micro_cols=micro_cols,
        batch_size=int(args.embed_batch),
    )
    train_macro_f, macro_pls = _fit_pls(emb["train_macro"], proj_targets, emb["train_macro"], macro_dim)
    val_macro_f = macro_pls.transform(emb["val_macro"]).astype(np.float32)
    eval_macro_f = macro_pls.transform(emb["eval_macro"]).astype(np.float32)
    train_micro_f, micro_pls = _fit_pls(emb["train_micro"], proj_targets, emb["train_micro"], micro_dim)
    val_micro_f = micro_pls.transform(emb["val_micro"]).astype(np.float32)
    eval_micro_f = micro_pls.transform(emb["eval_micro"]).astype(np.float32)

    feature_cols = list(bundle.get("feature_cols") or [])
    raw_cols = [c for c in feature_cols if not c.startswith("macro_factor_") and not c.startswith("micro_factor_")]
    if nan_missing:
        val_full = _frame_with_prepared_features(val_df, val_feat, raw_cols)
        eval_full = _frame_with_prepared_features(eval_df, eval_feat, raw_cols)
    else:
        val_full = val_df.reset_index(drop=True).copy()
        eval_full = eval_df.reset_index(drop=True).copy()
    val_full = _add_factor_cols(val_full, "macro_factor", val_macro_f)
    val_full = _add_factor_cols(val_full, "micro_factor", val_micro_f)
    eval_full = _add_factor_cols(eval_full, "macro_factor", eval_macro_f)
    eval_full = _add_factor_cols(eval_full, "micro_factor", eval_micro_f)
    meta = {
        "variant": variant_name,
        "projection_spec": spec,
        "macro_cols": macro_cols,
        "micro_cols": micro_cols,
        "nan_missing": nan_missing,
        "embedding_cache_consumer": str(variant["cache_consumer"]),
        "embedding_cache_paths": {k: str(v) for k, v in paths.items()},
        "factor_train_shapes": {"macro": list(train_macro_f.shape), "micro": list(train_micro_f.shape)},
    }
    return val_full, eval_full, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate V31 with parent-policy swaps, including v40 encoded parents.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
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
    print(f"[{MODEL_ID}] loading shared artifacts", flush=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    cfg = _parent_cfg()
    original_parent = _load_bundle(args.parent_model)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(args.v27_model)

    print(f"[{MODEL_ID}] building labels and prepared feature frames", flush=True)
    x_train_base, y, training_meta = build_training_set(train_df, cfg=cfg, stride_bars=int(args.train_stride), batch_size=512, feature_cols=feature_cols)
    train_idx = np.arange(0, max(0, len(train_df) - cfg.max_train_horizon_bars - 1), max(1, int(args.train_stride)), dtype=np.int64)
    if len(train_idx) != len(x_train_base):
        raise RuntimeError(f"train_idx/x mismatch: {len(train_idx)} vs {len(x_train_base)}")
    val_idx = np.arange(len(val_df), dtype=np.int64)
    eval_idx = np.arange(len(eval_df), dtype=np.int64)
    proj_targets = _projection_targets(y)
    train_feat_zero = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    val_feat_zero = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_feat_zero = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    train_feat_nan = _preserve_source_missing_as_nan(train_feat_zero, train_df, feature_cols)
    val_feat_nan = _preserve_source_missing_as_nan(val_feat_zero, val_df, feature_cols)
    eval_feat_nan = _preserve_source_missing_as_nan(eval_feat_zero, eval_df, feature_cols)

    print(f"[{MODEL_ID}] predicting frozen V27 utilities", flush=True)
    val_q = v31._predict_all(v27_model, val_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    ledgers: dict[str, str] = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for variant_name, variant in PARENT_VARIANTS.items():
        print(f"[{MODEL_ID}] evaluating {variant_name}", flush=True)
        parent_bundle = _load_bundle(Path(variant["bundle"]))
        with Path(variant["report"]).open(encoding="utf-8") as f:
            parent_report = json.load(f)
        if variant["kind"] == "raw":
            val_frame = val_df.reset_index(drop=True).copy()
            eval_frame = eval_df.reset_index(drop=True).copy()
            parent_meta = {"variant": variant_name, "kind": "raw"}
        else:
            val_frame, eval_frame, parent_meta = _build_encoded_frames(
                variant_name=variant_name,
                variant=variant,
                bundle=parent_bundle,
                report=parent_report,
                args=args,
                train_df=train_df,
                val_df=val_df,
                eval_df=eval_df,
                train_feat_zero=train_feat_zero,
                val_feat_zero=val_feat_zero,
                eval_feat_zero=eval_feat_zero,
                train_feat_nan=train_feat_nan,
                val_feat_nan=val_feat_nan,
                eval_feat_nan=eval_feat_nan,
                train_idx=train_idx,
                val_idx=val_idx,
                eval_idx=eval_idx,
                proj_targets=proj_targets,
            )
        val_dec = predict_policy_frame(parent_bundle, val_frame, close=_close(val_frame))
        eval_dec = predict_policy_frame(parent_bundle, eval_frame, close=_close(eval_frame))
        base = dict(parent_bundle.get("config", {}))
        fee = float(base.get("fee", cfg.fee))
        slip = float(base.get("slip", cfg.slip))

        best: dict[str, Any] | None = None
        grid_rows: list[dict[str, Any]] = []
        for overlay in v31._grid():
            v1 = v31.backtest(val_frame, original_parent, jackpot_model, add_cfg, val_q, overlay, fee=fee, slip=slip, cost_mult=1.0, decisions=val_dec)
            v2 = v31.backtest(val_frame, original_parent, jackpot_model, add_cfg, val_q, overlay, fee=fee, slip=slip, cost_mult=2.0, decisions=val_dec)
            v3 = v31.backtest(val_frame, original_parent, jackpot_model, add_cfg, val_q, overlay, fee=fee, slip=slip, cost_mult=3.0, decisions=val_dec)
            score = v31._score(v1, v2, v3)
            row = {
                "variant": variant_name,
                "overlay": overlay.name,
                "selection_score": score,
                "validation_cost1": v1,
                "validation_cost2": v2,
                "validation_cost3": v3,
                "overlay_config": asdict(overlay),
            }
            grid_rows.append(row)
            rows.append(
                {
                    "variant": variant_name,
                    "overlay": overlay.name,
                    "selection_score": score,
                    "val_pnl": v1["pnl"],
                    "val_mdd": v1["mdd"],
                    "val_trades": v1["trades"],
                    "val_deep_entries": v1.get("deep_entries", 0),
                    "val_c2_pnl": v2["pnl"],
                    "val_c3_pnl": v3["pnl"],
                }
            )
            if best is None or score > best["selection_score"]:
                best = row
        assert best is not None
        selected = v31.OverlayConfig(**best["overlay_config"])
        metrics: dict[str, Any] = {}
        for mult in (1, 2, 3):
            r = v31.backtest(
                eval_frame,
                original_parent,
                jackpot_model,
                add_cfg,
                eval_q,
                selected,
                fee=fee,
                slip=slip,
                cost_mult=float(mult),
                decisions=eval_dec,
                record=(mult == 1),
            )
            if mult == 1:
                ledger = pd.DataFrame(r.pop("trade_records", []))
                ledger_path = args.report_out.with_name(f"{args.report_out.stem}_{variant_name}_cost1_ledger.csv")
                ledger.to_csv(ledger_path, index=False)
                ledgers[variant_name] = str(ledger_path)
            metrics[f"cost{mult}"] = r
        feature_audit_cols = [c for c in list(parent_bundle.get("feature_cols") or []) if not c.startswith("macro_factor_") and not c.startswith("micro_factor_")]
        feature_audit = _audit_contract(train_all, eval_df, feature_audit_cols)
        results[variant_name] = {
            "parent_model": str(variant["bundle"]),
            "parent_report": str(variant["report"]),
            "parent_meta": parent_meta,
            "selected_overlay": asdict(selected),
            "selection_result": best,
            "metrics": metrics,
            "feature_audit": feature_audit,
            "runner_feature_schema": "original_v31_parent_schema_preserved_for_v21_2",
        }
        manifest = {
            "model_id": MODEL_ID,
            "variant": variant_name,
            "parent_model": str(variant["bundle"]),
            "parent_report": str(variant["report"]),
            "v27_model": str(args.v27_model),
            "jackpot_model": str(args.jackpot_model),
            "selected_overlay": asdict(selected),
            "metrics": metrics,
        }
        (args.out_dir / f"{variant_name}_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )

    summary_rows = []
    for name, item in results.items():
        m = item["metrics"]
        summary_rows.append(
            {
                "variant": name,
                "overlay": item["selected_overlay"]["name"],
                "cost1_pnl": m["cost1"]["pnl"],
                "cost1_mdd": m["cost1"]["mdd"],
                "cost1_trades": m["cost1"]["trades"],
                "cost1_trades_day": m["cost1"]["trades_per_day"],
                "cost1_deep_entries": m["cost1"].get("deep_entries", 0),
                "cost2_pnl": m["cost2"]["pnl"],
                "cost2_mdd": m["cost2"]["mdd"],
                "cost3_pnl": m["cost3"]["pnl"],
                "cost3_mdd": m["cost3"]["mdd"],
            }
        )
    grid_df = pd.DataFrame(rows).sort_values(["variant", "selection_score"], ascending=[True, False])
    grid_df.to_csv(args.grid_out, index=False)
    summary_df = pd.DataFrame(summary_rows).sort_values("cost1_pnl", ascending=False)
    best_variant = summary_df.iloc[0].to_dict() if len(summary_df) else {}
    blocking = []
    warnings = []
    for name, item in results.items():
        fa = item["feature_audit"]
        if fa.get("status") != "pass":
            blocking.extend([f"{name}:{x}" for x in fa.get("blocking", [])])
        warnings.extend([f"{name}:{x}" for x in fa.get("warnings", [])])
        if item["metrics"]["cost2"]["pnl"] <= 0:
            warnings.append(f"{name}:cost2_not_survived")
        if item["metrics"]["cost3"]["pnl"] <= 0:
            warnings.append(f"{name}:cost3_not_survived")
    audit = {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31 per parent variant",
        "oos_window": "2026 fixed OOS only after overlay selection",
        "v27_entry_frozen": True,
        "v21_2_runner_model_preserved": True,
        "runner_feature_schema": "original parent raw schema used for V21.2 feature frame",
        "encoded_parents_regenerate_factors": True,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "V31 parent-swap A/B. Each parent produces its own decision frame. Frozen V27, V31 exit overlay grid, and V21.2 jackpot runner are preserved. Encoded v40 parents regenerate Chronos/Kairos PLS factors from train-only projections before decision generation.",
        "training_meta": training_meta,
        "variants": results,
        "summary": summary_rows,
        "best_by_cost1": best_variant,
        "audit": audit,
        "artifacts": {
            "out_dir": str(args.out_dir),
            "report": str(args.report_out),
            "audit": str(args.audit_out),
            "grid": str(args.grid_out),
            "ledgers": ledgers,
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "summary": summary_rows, "best_by_cost1": best_variant, "audit_status": audit["status"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
