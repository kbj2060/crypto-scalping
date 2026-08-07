#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import predict_policy_frame  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_ft_transformer_mtl_parent_v2_20260515 as ft_v2  # noqa: E402
from scripts import eval_alpha4_new_features_full_retrain_20260517 as a4  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.alpha6_catboost_entry_quality_exit_policy_20260522 import _target_horizon_bucket  # noqa: E402
from scripts.analyze_alpha6_sleeve_complementarity_20260523 import Expert, _parse_exit_threshold  # noqa: E402
from scripts.train_alpha6_dsac_ensemble_router_20260523 import MODEL_SPECS  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import _no_deep_overlay, _q0  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


PRIMARY = 0
COVERAGE = 1
CONFIRMERS = (2, 3)
RISKS = (4, 5)
OLD_REGIME_PREFIXES = (
    "clean_regime_2024_unsup_v4_",
    "regime4_pred_",
)
ALPHA61_STICKY_PREFIX = "clean_regime4_state24_sticky090_v2_"
FIXED_STICKY_PREFIX = "clean_regime4_2024_unsup_v1_"


def _class_prob(model: Any, x: np.ndarray, cls: int) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)
    if cls not in classes:
        return np.zeros(len(x), dtype=np.float64)
    return np.asarray(proba[:, int(np.flatnonzero(classes == cls)[0])], dtype=np.float64)


def _predict_alpha6_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    cols = list(bundle["feature_cols"])
    local = frame.copy()
    for col in cols:
        if col in local.columns or not col.startswith(ALPHA61_STICKY_PREFIX):
            continue
        alias = FIXED_STICKY_PREFIX + col[len(ALPHA61_STICKY_PREFIX) :]
        if alias in local.columns:
            local[col] = local[alias]
    missing = [c for c in cols if c not in local.columns]
    for col in missing:
        local[col] = np.nan
    x_raw = local[cols].replace([np.inf, -np.inf], np.nan)
    x = bundle["pipeline"].transform(x_raw)
    models = bundle["entry_models"]
    action_model = models["action_model"]
    cash_p = _class_prob(action_model, x, 0)
    long_p = _class_prob(action_model, x, 1)
    short_p = _class_prob(action_model, x, 2)
    proba = np.vstack([cash_p, long_p, short_p]).T
    action = np.argmax(proba, axis=1).astype(np.int64)
    quality = np.asarray(models["quality_model"].predict(x), dtype=np.float64)
    target_head_mode = str(models.get("target_head_mode", "bucket5")).strip().lower()
    if target_head_mode == "horizon_reg":
        horizon_model = models.get("target_horizon_model") or models.get("target_model")
        max_horizon = int(models.get("max_target_horizon") or 96)
        pred_horizon = np.expm1(np.asarray(horizon_model.predict(x), dtype=np.float64))
        target_horizon = np.clip(np.rint(pred_horizon), 2, max(2, max_horizon)).astype(np.int64)
        target_horizon = np.where(action == 0, 0, target_horizon)
        target_bucket = np.where(action == 0, 0, _target_horizon_bucket(target_horizon)).astype(np.int64)
    else:
        bucket_model = models.get("target_bucket_model") or models.get("target_model")
        bucket_proba = bucket_model.predict_proba(x)
        bucket_classes = np.asarray(bucket_model.classes_, dtype=int)
        target_bucket = bucket_classes[np.argmax(bucket_proba, axis=1)].astype(np.int64)
        horizon_map = {0: 6, 1: 12, 2: 24, 3: 48, 4: 96}
        target_horizon = np.asarray([horizon_map.get(int(b), 96) for b in target_bucket], dtype=np.int64)
        target_bucket = np.where(action == 0, 0, target_bucket)
        target_horizon = np.where(action == 0, 0, target_horizon)
    return (
        pd.DataFrame(
            {
                "action": action,
                "quality_score": quality,
                "confidence": np.max(proba, axis=1),
                "target_bucket": target_bucket,
                "target_horizon": target_horizon,
                "notional": np.full(len(frame), float(bundle.get("config", {}).get("fixed_notional", 0.25))),
            }
        ),
        x,
        missing,
    )


def _load_alpha6_experts(frame: pd.DataFrame) -> tuple[list[Expert], dict[str, Any]]:
    experts: list[Expert] = []
    missing_by_expert: dict[str, list[str]] = {}
    for name, prefix in MODEL_SPECS:
        bundle = joblib.load(f"{prefix}_bundle.joblib")
        summary = json.loads(Path(f"{prefix}_summary.json").read_text())
        dec, x, missing = _predict_alpha6_bundle(bundle, frame)
        missing_by_expert[name] = missing
        best = summary["best"]
        experts.append(
            Expert(
                name=name,
                prefix=Path(prefix),
                bundle=bundle,
                summary=summary,
                dec=dec,
                x=np.asarray(x, dtype=np.float64),
                entry_threshold=float(best["entry_threshold"]),
                exit_threshold=_parse_exit_threshold(best.get("exit_threshold", 0.55)),
            )
        )
    return experts, {"missing_by_expert": missing_by_expert}


def _desired(e: Expert, i: int) -> int:
    row = e.dec.iloc[i]
    return int(row.action) if float(row.quality_score) >= float(e.entry_threshold) else 0


def _alpha61_meta(experts: list[Expert]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(len(experts[0].dec)):
        desired = [_desired(e, i) for e in experts]
        side = 0
        if desired[PRIMARY]:
            side = desired[PRIMARY]
        elif desired[COVERAGE]:
            side = desired[COVERAGE]
        same = sum(1 for d in desired if d == side and side != 0)
        opp = sum(1 for d in desired if side != 0 and d not in (0, side))
        rows.append(
            {
                "alpha61_side": 1 if side == 1 else (-1 if side == 2 else 0),
                "alpha61_agreement": same,
                "alpha61_opp": opp,
                "alpha61_risk_opp": sum(1 for idx in RISKS if side != 0 and desired[idx] not in (0, side)),
                "alpha61_high_precision_same": bool(side != 0 and desired[2] == side),
                "alpha61_all_confirm_same": bool(side != 0 and all(desired[idx] == side for idx in (2, 3))),
            }
        )
    return pd.DataFrame(rows)


def _parent_for_features(parent: dict[str, Any]) -> dict[str, Any]:
    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    out = copy.deepcopy(parent_ref)
    out["feature_cols"] = list(parent["feature_cols"])
    return out


def _load_runner(path: Path) -> tuple[dict[str, Any], CostRunnerConfig]:
    payload = joblib.load(path)
    return payload["cost_runner"], CostRunnerConfig(**payload["selected_config"])


def _metrics(df: pd.DataFrame, parent: dict[str, Any], runner: dict[str, Any], cfg: CostRunnerConfig, dec: pd.DataFrame, fee: float, slip: float) -> dict[str, Any]:
    return a4._metrics(df, _parent_for_features(parent), runner, cfg, _q0(df), dec, _no_deep_overlay(), ft_v2.ft_v1._limit_cfg(), fee=fee, slip=slip)


def _scale_notional(dec: pd.DataFrame, scale: np.ndarray) -> pd.DataFrame:
    out = dec.copy()
    scale = np.asarray(scale, dtype=np.float64)
    out["notional_exposure"] = np.clip(pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy() * scale, 0.0, 2.75)
    lev = np.maximum(pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0).to_numpy(), 1e-9)
    out["position_fraction"] = np.clip(out["notional_exposure"].to_numpy(dtype=np.float64) / lev, 0.0, 1.0)
    out.loc[out["notional_exposure"] <= 0, ["action", "side", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[out["notional_exposure"] <= 0, "leverage"] = 1.0
    return out


def _fusion_decisions(base_dec: pd.DataFrame, meta: pd.DataFrame, mode: str) -> pd.DataFrame:
    dec = base_dec.copy().reset_index(drop=True)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    a61_side = meta["alpha61_side"].to_numpy(dtype=np.int64)
    agree = (side != 0) & (a61_side == side)
    opp = (side != 0) & (a61_side == -side)
    risk_opp = meta["alpha61_risk_opp"].to_numpy(dtype=np.int64)
    agreement = meta["alpha61_agreement"].to_numpy(dtype=np.int64)
    highp = meta["alpha61_high_precision_same"].to_numpy(dtype=bool)

    if mode == "alpha43":
        return dec
    if mode == "alpha43_alpha61_agree_only":
        return _scale_notional(dec, np.where(agree, 1.0, 0.0))
    if mode == "alpha43_alpha61_veto_opp":
        return _scale_notional(dec, np.where(opp | (risk_opp >= 2), 0.0, 1.0))
    if mode == "alpha43_alpha61_soft_scale":
        scale = np.ones(len(dec), dtype=np.float64)
        scale[opp | (risk_opp >= 2)] = 0.0
        scale[(side != 0) & ~agree] = 0.50
        scale[agree & highp & (agreement >= 3)] = 1.10
        return _scale_notional(dec, scale)
    if mode == "alpha43_alpha61_sniper_scale":
        scale = np.full(len(dec), 0.35, dtype=np.float64)
        scale[side == 0] = 0.0
        scale[opp | (risk_opp >= 1)] = 0.0
        scale[agree & highp & (agreement >= 3)] = 1.0
        scale[agree & highp & (agreement >= 5)] = 1.25
        return _scale_notional(dec, scale)
    raise ValueError(f"unknown mode {mode}")


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        cost: {k: metrics[cost][k] for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional", "avg_leverage", "exits")}
        for cost in ("cost1", "cost2", "cost3")
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-csv", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/trade_candidates_2026_patchtst__tide__dlinear.csv")
    ap.add_argument("--parent", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha4_3_legacy_regime_block_ablation_alpha43basis_20260517/no_legacy/parent.pkl")
    ap.add_argument("--runner", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha4_3_legacy_regime_block_ablation_alpha43basis_20260517/no_legacy/runners/no_legacy__parent_direct_raw_no_teacher_runner.pkl")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha6_2_alpha43_no_legacy_fusion_20260524")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_df = _read(args.eval_csv)
    parent = joblib.load(args.parent)
    runner, runner_cfg = _load_runner(args.runner)
    fee = float(parent["config"]["fee"])
    slip = float(parent["config"]["slip"])

    forbidden = [c for c in parent["feature_cols"] if c.startswith(OLD_REGIME_PREFIXES)]
    if forbidden:
        raise ValueError(f"parent contains regime features: {forbidden[:20]}")

    base_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    experts, alpha61_audit = _load_alpha6_experts(eval_df)
    alpha61 = _alpha61_meta(experts)

    rows = []
    outputs: dict[str, Any] = {}
    modes = [
        "alpha43",
        "alpha43_alpha61_agree_only",
        "alpha43_alpha61_veto_opp",
        "alpha43_alpha61_soft_scale",
        "alpha43_alpha61_sniper_scale",
    ]
    for mode in modes:
        dec = _fusion_decisions(base_dec, alpha61, mode)
        metrics = _metrics(eval_df, parent, runner, runner_cfg, dec, fee, slip)
        compact = _compact(metrics)
        outputs[mode] = compact
        rows.append(
            {
                "mode": mode,
                "cost3_pnl": compact["cost3"]["pnl"],
                "cost3_mdd": compact["cost3"]["mdd"],
                "cost3_trades": compact["cost3"]["trades"],
                "cost3_wr": compact["cost3"]["wr"],
                "cost3_avg_notional": compact["cost3"]["avg_notional"],
                "cost2_pnl": compact["cost2"]["pnl"],
                "cost1_pnl": compact["cost1"]["pnl"],
            }
        )
        dec.assign(timestamp=eval_df["timestamp"].to_numpy()).to_csv(args.out_dir / f"{mode}_decisions.csv", index=False)
        print(rows[-1], flush=True)

    rank = pd.DataFrame(rows).sort_values("cost3_pnl", ascending=False).reset_index(drop=True)
    rank.to_csv(args.out_dir / "ranking.csv", index=False)
    alpha61.to_csv(args.out_dir / "alpha61_meta.csv", index=False)
    summary = {
        "model_id": "alpha6_2_alpha43_no_legacy_fusion_20260524",
        "design": "Alpha4.3 no-legacy parent-direct OOS on 2026, fused with Alpha6.1 scoring-stack experts as agreement/risk filters. Alpha4 parent has zero regime input columns.",
        "eval_csv": str(args.eval_csv),
        "parent": str(args.parent),
        "runner": str(args.runner),
        "runner_config": asdict(runner_cfg),
        "fee": fee,
        "slip": slip,
        "audit": {
            "alpha43_parent_forbidden_regime_feature_count": len(forbidden),
            "alpha43_parent_sticky_feature_count": int(sum(c.startswith(FIXED_STICKY_PREFIX) or c.startswith(ALPHA61_STICKY_PREFIX) for c in parent["feature_cols"])),
            "alpha43_feature_count": len(parent["feature_cols"]),
            "alpha43_feature_cols": list(parent["feature_cols"]),
            "alpha61_missing_columns": alpha61_audit["missing_by_expert"],
            "selection_uses_2026": False,
            "note": "Alpha6.1 missing columns are inserted as NaN and handled by each bundle's training-time imputer; this is a neutralized inference overlay, not Alpha6.1 retraining on 2026.",
        },
        "results": outputs,
        "ranking": rank.to_dict(orient="records"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    print(f"[out] {args.out_dir}", flush=True)
    print(rank.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
