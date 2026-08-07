#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.alpha6_catboost_entry_quality_exit_policy_20260522 import (  # noqa: E402
    EQEConfig,
    _predict_entry,
)
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import _json_default  # noqa: E402
from scripts.eval_omega1_regime3_expertdq_risk_replay_20260602 import ACTIVE_TEMPLATE  # noqa: E402
from scripts.train_eval_omega1_expertdq_dsac_risk4_allocator_20260602 import (  # noqa: E402
    ACTION_CASH,
    ACTION_DIM,
    ACTIVE_SCALES,
    NOTIONAL_BUCKETS,
    LEVERAGE_BUCKETS,
    TP_BUCKETS,
    SL_BUCKETS,
    VALID_ACTION_IDS,
    _active,
    _action_count_names,
    _apply_norm,
    _build_dataset,
    _compose_decisions,
    _fit_norm,
    _metrics_row,
    _numeric_feature_cols,
    _policy_actions,
    _seed_everything,
    _train_dsac,
)
from scripts.train_eval_omega1_expertdq_dsac_risk_allocator_20260602 import (  # noqa: E402
    _load_variant_frames,
)


MODEL_ID = "alpha6_parent_dsac_risk4_allocator_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
FORBIDDEN_EXACT_FEATURES = {
    "tp_sl_action_score",
    "m7_hdb_label",
    "m7_target_hold",
}
FORBIDDEN_PREFIXES = (
    "clean_regime4_",
    "regime4_pred_",
    "regime3_pred_",
)


def _forbidden_feature_hits(cols: list[str]) -> list[str]:
    hits: list[str] = []
    for col in cols:
        name = str(col)
        if name in FORBIDDEN_EXACT_FEATURES or any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            hits.append(name)
    return hits


def _assert_no_forbidden_features(cols: list[str], *, where: str) -> None:
    hits = _forbidden_feature_hits(cols)
    if hits:
        raise RuntimeError(f"forbidden feature contract violation in {where}: {hits[:80]}")


def _cfg_from_bundle(bundle: dict[str, Any]) -> EQEConfig:
    raw = dict(bundle.get("config") or {})
    allowed = {f.name for f in fields(EQEConfig)}
    return EQEConfig(**{k: v for k, v in raw.items() if k in allowed})


def _predict_alpha6_full(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    cols = list(bundle["feature_cols"])
    _assert_no_forbidden_features(cols, where="alpha6_bundle.feature_cols")
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Alpha6 feature contract mismatch. Missing columns: {missing[:30]}")
    x_raw = frame.loc[:, cols].copy()
    for col in cols:
        x_raw[col] = pd.to_numeric(x_raw[col], errors="coerce")
    x_raw = x_raw.replace([np.inf, -np.inf], np.nan)
    x = bundle["pipeline"].transform(x_raw)
    return _predict_entry(bundle["entry_models"], x, _cfg_from_bundle(bundle))


def _load_alpha6_oof(path: Path, train_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    oof = pd.read_csv(path)
    required = {"timestamp", "action", "quality_score", "confidence", "target_bucket", "target_horizon", "notional"}
    missing = sorted(required - set(oof.columns))
    if missing:
        raise RuntimeError(f"Alpha6 OOF prediction missing columns: {missing}")
    oof["timestamp"] = pd.to_datetime(oof["timestamp"], errors="raise")
    if oof["timestamp"].duplicated().any():
        dup = oof.loc[oof["timestamp"].duplicated(), "timestamp"].head(5).astype(str).tolist()
        raise RuntimeError(f"Alpha6 OOF duplicate timestamps: {dup}")
    train_ts = pd.to_datetime(train_df["timestamp"], errors="raise")
    aligned = oof.set_index("timestamp").reindex(train_ts)
    covered = ~aligned["action"].isna().to_numpy()
    missing = train_ts[~covered].astype(str).tolist()
    return aligned.loc[covered].reset_index(drop=True), covered, missing


def _alpha6_decisions(pred: pd.DataFrame) -> pd.DataFrame:
    action = pd.to_numeric(pred["action"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    side = np.where(action == 1, 1, np.where(action == 2, -1, 0)).astype(np.int64)
    active = (action != ACTION_CASH) & (side != 0)
    notional = float(ACTIVE_TEMPLATE["notional"])
    leverage = float(ACTIVE_TEMPLATE["leverage"])
    out = pd.DataFrame(
        {
            "action": action,
            "side": side,
            "notional_exposure": np.where(active, notional, 0.0),
            "leverage": np.where(active, leverage, 1.0),
            "position_fraction": np.where(active, notional / max(leverage, 1e-8), 0.0),
            "take_profit": np.where(active, float(ACTIVE_TEMPLATE["take_profit"]), 0.0),
            "stop_loss": np.where(active, float(ACTIVE_TEMPLATE["stop_loss"]), 0.0),
            "max_hold_bars": np.zeros(len(pred), dtype=np.int64),
            "cooldown_bars": np.zeros(len(pred), dtype=np.int64),
            "quality_score": pd.to_numeric(pred["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
            "confidence": pd.to_numeric(pred["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
            "target_bucket": pd.to_numeric(pred["target_bucket"], errors="coerce").fillna(0).astype(np.int64).to_numpy(),
            "target_horizon": pd.to_numeric(pred["target_horizon"], errors="coerce").fillna(0).astype(np.int64).to_numpy(),
        }
    )
    return out


def _alpha6_state(pred: pd.DataFrame, dec: pd.DataFrame) -> pd.DataFrame:
    action = pd.to_numeric(pred["action"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    side = np.where(action == 1, 1.0, np.where(action == 2, -1.0, 0.0))
    out = pd.DataFrame(index=pred.index)
    out["alpha6_action"] = action.astype(np.float64)
    out["alpha6_side"] = side.astype(np.float64)
    out["alpha6_is_long"] = (action == 1).astype(np.float64)
    out["alpha6_is_short"] = (action == 2).astype(np.float64)
    out["alpha6_quality_score"] = pd.to_numeric(pred["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["alpha6_confidence"] = pd.to_numeric(pred["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["alpha6_target_bucket"] = pd.to_numeric(pred["target_bucket"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["alpha6_target_horizon"] = pd.to_numeric(pred["target_horizon"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["alpha6_side_x_quality"] = out["alpha6_side"] * out["alpha6_quality_score"]
    out["alpha6_side_x_confidence"] = out["alpha6_side"] * out["alpha6_confidence"]
    out["alpha6_active"] = _active(dec).astype(np.float64)
    return out


def _decision_state(dec: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=dec.index)
    for col in (
        "action",
        "side",
        "notional_exposure",
        "leverage",
        "position_fraction",
        "take_profit",
        "stop_loss",
        "quality_score",
        "confidence",
        "target_bucket",
        "target_horizon",
    ):
        if col not in dec.columns:
            raise RuntimeError(f"decision state missing: {col}")
        out[f"decision_{col}"] = pd.to_numeric(dec[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["decision_side_x_quality"] = out["decision_side"] * out["decision_quality_score"]
    out["decision_side_x_confidence"] = out["decision_side"] * out["decision_confidence"]
    out["decision_rr"] = out["decision_take_profit"] / np.maximum(np.abs(out["decision_stop_loss"]), 1e-8)
    return out


def _build_state_frame(frame: pd.DataFrame, pred: pd.DataFrame, dec: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    _assert_no_forbidden_features(feature_cols, where="alpha6_dsac_raw_feature_cols")
    x_base = frame.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    parts = [
        x_base.reset_index(drop=True),
        _alpha6_state(pred.reset_index(drop=True), dec.reset_index(drop=True)).reset_index(drop=True),
        _decision_state(dec.reset_index(drop=True)).reset_index(drop=True),
    ]
    out = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if out.columns.duplicated().any():
        dup = out.columns[out.columns.duplicated()].tolist()
        raise RuntimeError(f"duplicate Alpha6 DSAC state columns: {dup[:20]}")
    _assert_no_forbidden_features(list(out.columns), where="alpha6_dsac_state_columns")
    return out


def _usage(actions: np.ndarray, active: np.ndarray) -> dict[str, int]:
    counts: dict[int, int] = {}
    for a in actions[np.asarray(active, dtype=bool)]:
        counts[int(a)] = counts.get(int(a), 0) + 1
    return _action_count_names(counts, limit=20)


def _summary_counts(dec: pd.DataFrame, name: str) -> dict[str, Any]:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(int)
    return {
        f"{name}_active_rows": int(_active(dec).sum()),
        f"{name}_action_counts": {str(k): int(v) for k, v in action.value_counts().sort_index().to_dict().items()},
        f"{name}_side_counts": {str(k): int(v) for k, v in side.value_counts().sort_index().to_dict().items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="soft_floor_0p00")
    ap.add_argument("--alpha6-bundle", type=Path, required=True)
    ap.add_argument("--alpha6-oof", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--bc-coef", type=float, default=0.0)
    ap.add_argument("--n-quantiles", type=int, default=32)
    ap.add_argument("--cvar-frac", type=float, default=0.25)
    ap.add_argument("--pessimism-weight", type=float, default=0.80)
    ap.add_argument("--target-entropy", type=float, default=2.0)
    ap.add_argument("--oracle-risk-penalty", type=float, default=0.0)
    ap.add_argument("--samples-per-row", type=int, default=96)
    ap.add_argument("--max-active-rows", type=int, default=0)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    _seed_everything(260602)
    out_dir = OUT_DIR / str(args.variant) / args.alpha6_bundle.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    bundle = joblib.load(args.alpha6_bundle)
    _assert_no_forbidden_features(list(bundle.get("feature_cols", [])), where="alpha6_bundle.feature_cols")
    train_df, val_df, oos_df, _train_src, _val_src, _oos_src, overlay = _load_variant_frames(str(args.variant))
    train_pred, train_covered, train_missing = _load_alpha6_oof(args.alpha6_oof, train_df)
    if not bool(train_covered.any()):
        raise RuntimeError("Alpha6 OOF covers zero train rows")
    train_df = train_df.loc[train_covered].copy().reset_index(drop=True)
    val_pred = _predict_alpha6_full(bundle, val_df)
    oos_pred = _predict_alpha6_full(bundle, oos_df)
    train_dec = _alpha6_decisions(train_pred)
    val_dec = _alpha6_decisions(val_pred)
    oos_dec = _alpha6_decisions(oos_pred)

    feature_cols = [c for c in _numeric_feature_cols(train_df) if c not in FORBIDDEN_EXACT_FEATURES and not any(str(c).startswith(p) for p in FORBIDDEN_PREFIXES)]
    _assert_no_forbidden_features(feature_cols, where="selected_numeric_feature_cols")
    s_train = _build_state_frame(train_df, train_pred, train_dec, feature_cols)
    s_val = _build_state_frame(val_df, val_pred, val_dec, feature_cols)
    s_oos = _build_state_frame(oos_df, oos_pred, oos_dec, feature_cols)
    norm = _fit_norm(s_train)
    x_train = _apply_norm(s_train, norm)
    x_val = _apply_norm(s_val, norm)
    x_oos = _apply_norm(s_oos, norm)

    parent_cfg = joblib.load(v31.DEFAULT_PARENT)["config"]
    fee = float(parent_cfg["fee"])
    slip = float(parent_cfg["slip"])
    dataset, data_diag = _build_dataset(
        train_df,
        x_train,
        train_dec,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        samples_per_row=int(args.samples_per_row),
        max_active_rows=int(args.max_active_rows),
        oracle_risk_penalty=float(args.oracle_risk_penalty),
    )
    print(
        json.dumps(
            {
                "stage": "train_start",
                "model_id": MODEL_ID,
                "variant": args.variant,
                "device": str(device),
                "state_dim": int(x_train.shape[1]),
                "action_dim": int(ACTION_DIM),
                "valid_action_dim": int(len(VALID_ACTION_IDS)),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "oos_rows": int(len(oos_df)),
                "data_diag": data_diag,
                "alpha6_oof_dropped_train_rows": int(len(train_missing)),
                "alpha6_oof_first_dropped_timestamps": train_missing[:20],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    actor, train_diag = _train_dsac(
        dataset,
        state_dim=int(x_train.shape[1]),
        device=device,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        bc_coef=float(args.bc_coef),
        n_quantiles=int(args.n_quantiles),
        cvar_frac=float(args.cvar_frac),
        pessimism_weight=float(args.pessimism_weight),
        target_entropy=float(args.target_entropy),
    )

    a_train = _policy_actions(actor, x_train, device=device)
    a_val = _policy_actions(actor, x_val, device=device)
    a_oos = _policy_actions(actor, x_oos, device=device)
    dsac_train = _compose_decisions(train_dec, a_train)
    dsac_val = _compose_decisions(val_dec, a_val)
    dsac_oos = _compose_decisions(oos_dec, a_oos)

    rows = [
        _metrics_row("val", "alpha6_parent_fixed_template", val_df, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        _metrics_row("oos", "alpha6_parent_fixed_template", oos_df, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        _metrics_row("val", "alpha6_parent_dsac_risk4_allocator", val_df, dsac_val, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        _metrics_row("oos", "alpha6_parent_dsac_risk4_allocator", oos_df, dsac_oos, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
    ]
    grid = pd.DataFrame(rows)
    grid_path = out_dir / "grid.csv"
    grid.to_csv(grid_path, index=False)

    model_path = out_dir / "alpha6_parent_dsac_risk4_allocator.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "variant": str(args.variant),
            "alpha6_bundle": str(args.alpha6_bundle),
            "alpha6_oof": str(args.alpha6_oof),
            "state_dim": int(x_train.shape[1]),
            "action_dim": int(ACTION_DIM),
            "valid_action_dim": int(len(VALID_ACTION_IDS)),
            "valid_action_ids": VALID_ACTION_IDS.tolist(),
            "n_quantiles": int(args.n_quantiles),
            "cvar_frac": float(args.cvar_frac),
            "pessimism_weight": float(args.pessimism_weight),
            "target_entropy": float(args.target_entropy),
            "oracle_risk_penalty": float(args.oracle_risk_penalty),
            "state_columns": list(norm["columns"]),
            "state_normalizer": norm,
            "buckets": {"notional": NOTIONAL_BUCKETS, "leverage": LEVERAGE_BUCKETS, "tp": TP_BUCKETS, "sl": SL_BUCKETS},
            "actor_state_dict": actor.state_dict(),
        },
        model_path,
    )

    fixed_oos = grid[(grid["split"] == "oos") & (grid["variant"] == "alpha6_parent_fixed_template")].iloc[0].to_dict()
    dsac_oos = grid[(grid["split"] == "oos") & (grid["variant"] == "alpha6_parent_dsac_risk4_allocator")].iloc[0].to_dict()
    summary = {
        "model_id": MODEL_ID,
        "variant": str(args.variant),
        "design": "Alpha6 parent action/quality is the decision layer. Train split uses purged OOF Alpha6 predictions; VAL/OOS use the full-train Alpha6 bundle. DSAC owns only veto/notional/leverage/TP/SL risk allocation; max-hold and cooldown are absent.",
        "selection_basis": "2025Q4 validation fast replay; 2026 OOS report-only.",
        "selection_uses_2026": False,
        "legacy_compat_alias": False,
        "alpha6_bundle": str(args.alpha6_bundle),
        "alpha6_oof": str(args.alpha6_oof),
        "alpha6_feature_cols": list(bundle["feature_cols"]),
        "base_template": ACTIVE_TEMPLATE,
        "expert_scales_reference": ACTIVE_SCALES,
        "training": {
            "device": str(device),
            "state_dim": int(x_train.shape[1]),
            "action_dim": int(ACTION_DIM),
            "valid_action_dim": int(len(VALID_ACTION_IDS)),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "bc_coef": float(args.bc_coef),
            "n_quantiles": int(args.n_quantiles),
            "cvar_frac": float(args.cvar_frac),
            "pessimism_weight": float(args.pessimism_weight),
            "target_entropy": float(args.target_entropy),
            "oracle_risk_penalty": float(args.oracle_risk_penalty),
            "samples_per_row": int(args.samples_per_row),
            "cost_mult": float(args.cost_mult),
            "reward_label": "complete_trade_net_pnl_after_entry_exit_fee_slippage; no max-hold, no cooldown",
            "data_diag": data_diag,
            "train_diag": train_diag,
            "action_usage": {"train": _usage(a_train, _active(train_dec)), "val": _usage(a_val, _active(val_dec)), "oos": _usage(a_oos, _active(oos_dec))},
            "base_decision_counts": {
                **_summary_counts(train_dec, "train_alpha6"),
                **_summary_counts(val_dec, "val_alpha6"),
                **_summary_counts(oos_dec, "oos_alpha6"),
            },
        },
        "fast_replay": {"fixed_oos_cost3": fixed_oos, "dsac_oos_cost3": dsac_oos, "delta_pnl": float(dsac_oos["pnl"]) - float(fixed_oos["pnl"])},
        "overlay": overlay,
        "artifacts": {"summary": str(out_dir / "summary.json"), "grid": str(grid_path), "model": str(model_path)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(out_dir / "summary.json"),
                "fast_oos_fixed_cost3": fixed_oos,
                "fast_oos_dsac_cost3": dsac_oos,
                "fast_delta_oos_cost3_pnl": summary["fast_replay"]["delta_pnl"],
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
