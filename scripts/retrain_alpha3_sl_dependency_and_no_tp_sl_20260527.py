#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    FullyLearnedGovernorConfig,
    build_training_set,
    predict_policy_frame,
    prepare_features,
    train_policy,
)
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_limit_close_fallback_20260514 as alpha3_close  # noqa: E402
from scripts import eval_alpha3_regime4_state24_v2_full_retrain_20260526 as alpha3_full  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts import train_eval_hf_v13_jackpot_runner_v21_2 as v21  # noqa: E402
from scripts.eval_alpha3_ft_v2_retrained_downstream_20260515 import _fit_cost_runner_with_decisions  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402


MODEL_ID = "alpha3_sl_dependency_and_no_tp_sl_retrain_20260527"
REPORT_OUT = ROOT / f"data/ensemble/reports/{MODEL_ID}_summary.json"
GRID_OUT = ROOT / f"data/ensemble/reports/{MODEL_ID}_grid.csv"
OUT_DIR = ROOT / f"data/ensemble/supervised/{MODEL_ID}"
BASE_REPORT = ROOT / "data/ensemble/reports/alpha3_regime4_state24_v2_full_retrain_20260526_summary.json"


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _safe_seq_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in v27._seq_cols(df) if not c.startswith(alpha3_full.CLEAN_PREFIX) and not c.startswith(alpha3_full.SOURCE_STATE24_PREFIX)]
    bad = [
        c
        for c in cols
        if any(tok in c.lower() for tok in ("target", "label", "future", "cash_after", "pnl_after", "regime_v2", "hdb", "hmm"))
    ]
    if bad:
        raise RuntimeError(f"forbidden seq columns selected: {bad}")
    return cols[:80]


def _metrics(
    df: pd.DataFrame,
    parent: dict[str, Any],
    runner: dict[str, Any],
    add_cfg: v21.CostRunnerConfig,
    q: np.ndarray,
    decisions: pd.DataFrame,
    overlay: v31.OverlayConfig,
    limit_cfg: Any,
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    return alpha3_close._metrics_signal_limit_close(
        df,
        parent,
        runner,
        add_cfg,
        q,
        decisions,
        overlay,
        limit_cfg,
        fee=fee,
        slip=slip,
    )


def _merge_state24(base: pd.DataFrame, side_path: Path) -> pd.DataFrame:
    side = alpha3_full._rename_state24_sidecar(_read(side_path))
    merged, _ = alpha3_full._merge_state24(base, side)
    return merged


def _feature_cols(original_parent: dict[str, Any], state24_cols: list[str]) -> list[str]:
    raw_cols = list(original_parent["feature_cols"])
    kept = [c for c in raw_cols if not c.startswith(alpha3_full.CLEAN_PREFIX)]
    kept = [c for c in kept if not c.startswith(alpha3_full.SOURCE_STATE24_PREFIX)]
    kept.extend(state24_cols)
    if "side_hint" not in kept:
        kept.insert(0, "side_hint")
    return list(dict.fromkeys(kept))


def _variants(base_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sl = tuple(float(x) for x in base_cfg["stop_loss_buckets"])
    tp = tuple(float(x) for x in base_cfg["take_profit_buckets"])
    return [
        {
            "name": "sl_dependency_reduced",
            "cfg_update": {
                "stop_loss_buckets": (
                    min(0.090, sl[0] * 1.45),
                    min(0.090, sl[1] * 1.45),
                    min(0.090, sl[2] * 1.35),
                    min(0.090, sl[3] * 1.30),
                    min(0.090, sl[4] * 1.20),
                    min(0.090, sl[5] * 1.15),
                    min(0.090, sl[6] * 1.10),
                ),
                "take_profit_buckets": tp,
            },
        },
        {
            "name": "no_tp_sl_bucket_fixed",
            "cfg_update": {
                "take_profit_buckets": (0.040,),
                "stop_loss_buckets": (0.022,),
            },
        },
    ]


def _stoploss_ratio(cost3: dict[str, Any]) -> float:
    trades = max(1, int(cost3["trades"]))
    exits = dict(cost3.get("exits", {}))
    sl_hits = int(exits.get("deep_alpha_stop_loss", 0)) + int(exits.get("v21_2_stop_loss", 0))
    return float(sl_hits / trades)


def _train_variant(
    *,
    variant_name: str,
    cfg_obj: FullyLearnedGovernorConfig,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    fee: float,
    slip: float,
    seed: int,
    teacher_epochs: int,
    deep_epochs: int,
) -> dict[str, Any]:
    x_parent, y_parent, _ = build_training_set(
        train_df,
        cfg=cfg_obj,
        stride_bars=6,
        batch_size=512,
        feature_cols=feature_cols,
    )
    parent_bundle = train_policy(
        x_parent.reindex(columns=feature_cols),
        y_parent,
        cfg=cfg_obj,
        random_state=int(seed),
        feature_cols=feature_cols,
    )
    train_parent_dec = predict_policy_frame(parent_bundle, train_df, close=_close(train_df))
    val_parent_dec = predict_policy_frame(parent_bundle, val_df, close=_close(val_df))
    eval_parent_dec = predict_policy_frame(parent_bundle, eval_df, close=_close(eval_df))

    train_features = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    train_seq = teacher._seq_tensor(train_features, np.arange(len(train_df), dtype=np.int64), feature_cols)
    buckets = tuple(float(x) for x in cfg_obj.notional_buckets)
    teacher_model, teacher_meta = teacher._train_teacher_model(
        train_seq,
        train_parent_dec["action"].astype(int).to_numpy(dtype=np.int64),
        pd.to_numeric(train_parent_dec["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        teacher._bucket_labels(train_parent_dec, buckets),
        n_buckets=len(buckets),
        epochs=int(teacher_epochs),
    )
    val_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    val_teacher_pred = teacher._predict_deep(teacher_model, val_features, feature_cols, teacher_meta["norm"])
    eval_teacher_pred = teacher._predict_deep(teacher_model, eval_features, feature_cols, teacher_meta["norm"])

    seq_cols = _safe_seq_cols(train_df)
    deep_train = v27._build_train_set(train_df, seq_cols, fee=fee, slip=slip, stride=3)
    deep_norm = v27._normalizer(deep_train["seq"])
    deep_model = v27._train_model(deep_train, deep_norm, epochs=int(deep_epochs))
    val_q = v27._predict_all(deep_model, val_df, seq_cols, deep_norm)
    eval_q = v27._predict_all(deep_model, eval_df, seq_cols, deep_norm)

    best_rt: alpha2.Alpha2Runtime | None = None
    best_rt_score = -1e18
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    noop_cfg = next(c for c in v21._grid() if c.name == "v21_2_parent_noop")
    overlay_ref = next(v.overlay for v in l2._variants() if v.name == "alpha1_l2_conservative_fee20")
    limit_cfg = alpha3_full._canonical_limit_cfg()
    for rt in alpha2._runtimes():
        val_decisions = alpha2._decisions(val_parent_dec, val_teacher_pred, buckets, rt)
        metrics = _metrics(val_df, parent_bundle, jackpot_model, noop_cfg, val_q, val_decisions, overlay_ref, limit_cfg, fee=fee, slip=slip)
        score = _score(metrics)
        if score > best_rt_score:
            best_rt_score = score
            best_rt = rt
    if best_rt is None:
        raise RuntimeError("teacher runtime selection failed")

    train_decisions = alpha2._decisions(train_parent_dec, teacher._predict_deep(teacher_model, train_features, feature_cols, teacher_meta["norm"]), buckets, best_rt)
    val_decisions = alpha2._decisions(val_parent_dec, val_teacher_pred, buckets, best_rt)
    eval_decisions = alpha2._decisions(eval_parent_dec, eval_teacher_pred, buckets, best_rt)

    runner = _fit_cost_runner_with_decisions(train_df, parent_bundle, train_decisions, fee=fee, slip=slip)
    best_row: dict[str, Any] | None = None
    for add_cfg in v21._grid():
        for overlay in v31._grid():
            metrics = _metrics(val_df, parent_bundle, runner, add_cfg, val_q, val_decisions, overlay, limit_cfg, fee=fee, slip=slip)
            row = {
                "runner_config": asdict(add_cfg),
                "overlay_config": asdict(overlay),
                "score": _score(metrics),
            }
            if best_row is None or float(row["score"]) > float(best_row["score"]):
                best_row = row
    if best_row is None:
        raise RuntimeError("runner/overlay selection failed")

    selected_runner = v21.CostRunnerConfig(**best_row["runner_config"])
    selected_overlay = v31.OverlayConfig(**best_row["overlay_config"])
    oos_metrics = _metrics(eval_df, parent_bundle, runner, selected_runner, eval_q, eval_decisions, selected_overlay, limit_cfg, fee=fee, slip=slip)
    return {
        "variant": variant_name,
        "cfg": asdict(cfg_obj),
        "selected_teacher_runtime": asdict(best_rt),
        "selected_runner": asdict(selected_runner),
        "selected_overlay": asdict(selected_overlay),
        "metrics": oos_metrics,
        "score": _score(oos_metrics),
        "stoploss_ratio_cost3": _stoploss_ratio(oos_metrics["cost3"]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Retrain Alpha3 variants for stop-loss dependency reduction and no TP/SL bucket ablation.")
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--teacher-epochs", type=int, default=18)
    p.add_argument("--deep-epochs", type=int, default=18)
    args = p.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    base_parent = joblib.load(v31.DEFAULT_PARENT)
    base_cfg = dict(base_parent["config"])
    fee = float(base_cfg["fee"])
    slip = float(base_cfg["slip"])

    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train_all = _merge_state24(train_all, alpha3_full.SIDE_CLEAN4_2025)
    eval_df = _merge_state24(eval_df, alpha3_full.SIDE_CLEAN4_2026)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    state24_cols = sorted(c for c in train_all.columns if c.startswith(alpha3_full.TARGET_STATE24_PREFIX))
    feature_cols = _feature_cols(base_parent, state24_cols)

    variants = _variants(base_cfg)
    rows: list[dict[str, Any]] = []
    for v in variants:
        cfg_dict = dict(base_cfg)
        cfg_dict.update(dict(v["cfg_update"]))
        cfg_obj = FullyLearnedGovernorConfig(**cfg_dict)
        result = _train_variant(
            variant_name=str(v["name"]),
            cfg_obj=cfg_obj,
            feature_cols=feature_cols,
            train_df=train_df,
            val_df=val_df,
            eval_df=eval_df,
            fee=fee,
            slip=slip,
            seed=int(args.seed),
            teacher_epochs=int(args.teacher_epochs),
            deep_epochs=int(args.deep_epochs),
        )
        c3 = result["metrics"]["cost3"]
        rows.append(
            {
                "variant": result["variant"],
                "oos_cost3_pnl": float(c3["pnl"]),
                "oos_cost3_mdd": float(c3["mdd"]),
                "oos_cost3_wr": float(c3["wr"]),
                "oos_cost3_trades": int(c3["trades"]),
                "oos_cost3_v21_sl": int(c3["exits"].get("v21_2_stop_loss", 0)),
                "oos_cost3_deep_sl": int(c3["exits"].get("deep_alpha_stop_loss", 0)),
                "stoploss_ratio_cost3": float(result["stoploss_ratio_cost3"]),
                "score": float(result["score"]),
                "tp_buckets": json.dumps(result["cfg"]["take_profit_buckets"]),
                "sl_buckets": json.dumps(result["cfg"]["stop_loss_buckets"]),
            }
        )

    grid = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    grid.to_csv(GRID_OUT, index=False)

    baseline_report = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    baseline_metrics = baseline_report.get("alpha_sub_version_metrics", baseline_report.get("candidate_metrics"))
    baseline_cost3 = dict(baseline_metrics["cost3"])
    baseline_sl_ratio = _stoploss_ratio(baseline_cost3)

    best_variant = str(grid.iloc[0]["variant"])
    best_row = rows[[r["variant"] for r in rows].index(best_variant)]
    report = {
        "model_id": MODEL_ID,
        "design": "Two retrained variants: (1) stop-loss dependency reduced by widened SL buckets, (2) no TP/SL bucket by fixed single TP/SL values.",
        "protocol": {
            "selection_window": "2025-10-01..2025-12-31",
            "oos_window": "2026 fixed OOS",
            "selection_uses_2026": False,
        },
        "baseline_cost3": baseline_cost3,
        "baseline_stoploss_ratio_cost3": baseline_sl_ratio,
        "variants": rows,
        "selected_variant": best_row,
        "grid": str(GRID_OUT),
        "delta_vs_baseline_cost3": {
            "pnl": float(best_row["oos_cost3_pnl"] - baseline_cost3["pnl"]),
            "mdd": float(best_row["oos_cost3_mdd"] - baseline_cost3["mdd"]),
            "wr": float(best_row["oos_cost3_wr"] - baseline_cost3["wr"]),
            "trades": int(best_row["oos_cost3_trades"] - baseline_cost3["trades"]),
            "stoploss_ratio": float(best_row["stoploss_ratio_cost3"] - baseline_sl_ratio),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "grid": str(GRID_OUT), "selected_variant": best_variant}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
