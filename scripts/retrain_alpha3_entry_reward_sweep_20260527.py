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
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_regime4_state24_v2_full_retrain_20260526 as alpha3_full  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts import train_eval_hf_v13_jackpot_runner_v21_2 as v21  # noqa: E402
from scripts.backtest_alpha3_exit_guard_persistence_20260527 import (  # noqa: E402
    ExitGuardConfig,
    _default_limit_cfg,
    _metrics_guard,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402


MODEL_ID = "alpha3_entry_reward_sweep_20260527"
BASE_REPORT = ROOT / "data/ensemble/reports/alpha3_regime4_state24_v2_full_retrain_20260526_summary.json"
OUT_DIR = ROOT / f"data/ensemble/supervised/{MODEL_ID}"
GRID_OUT = ROOT / f"data/ensemble/reports/{MODEL_ID}_grid.csv"
REPORT_OUT = ROOT / f"data/ensemble/reports/{MODEL_ID}_summary.json"


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _sl_ratio(cost3: dict[str, Any]) -> float:
    exits = dict(cost3.get("exits", {}))
    sl_hits = sum(v for k, v in exits.items() if "stop_loss" in str(k))
    return float(sl_hits / max(int(cost3.get("trades", 0)), 1))


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


def _load_stack() -> tuple[dict[str, Any], Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    rep = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    exp = dict(rep["experiments"][-1])
    parent = joblib.load(exp["artifacts"]["parent"])
    runner_payload = joblib.load(exp["artifacts"]["runner"])
    runner = runner_payload["cost_runner"]
    add_cfg = v21.CostRunnerConfig(**dict(exp["selected_runner_config"]))
    overlay = v31.OverlayConfig(**dict(exp["selected_overlay"]))
    deep_payload = torch.load(exp["artifacts"]["deep_scout"], map_location="cpu", weights_only=False)
    deep_model = v27.DeepAlphaTCN(len(deep_payload["seq_cols"]))
    deep_model.load_state_dict(deep_payload["state_dict"])
    deep_model = deep_model.cpu().eval()
    teacher_payload = torch.load(exp["artifacts"]["teacher"], map_location="cpu", weights_only=False)
    teacher_cols = list(teacher_payload["feature_cols"])
    return parent, runner, add_cfg, overlay, deep_model, deep_payload, teacher_cols, rep, exp, teacher_payload, rep.get("alpha_sub_version_metrics", rep.get("candidate_metrics"))


def _variant_cfgs(base_cfg: dict[str, Any]) -> list[tuple[str, FullyLearnedGovernorConfig]]:
    out: list[tuple[str, FullyLearnedGovernorConfig]] = []
    out.append(("baseline_cfg", FullyLearnedGovernorConfig(**base_cfg)))

    v1 = dict(base_cfg)
    v1["turnover_bonus"] = 0.0002
    v1["hold_penalty"] = 0.0015
    v1["cash_score"] = 0.0002
    out.append(("entry_sparse_hold", FullyLearnedGovernorConfig(**v1)))

    v2 = dict(base_cfg)
    v2["turnover_bonus"] = 0.0
    v2["hold_penalty"] = 0.0008
    v2["cash_score"] = 0.00045
    out.append(("entry_sparse_cash", FullyLearnedGovernorConfig(**v2)))

    v3 = dict(base_cfg)
    v3["turnover_bonus"] = -0.0003
    v3["hold_penalty"] = 0.0005
    v3["cash_score"] = 0.0007
    v3["adverse_penalty"] = 0.78
    out.append(("entry_sparse_anti_churn", FullyLearnedGovernorConfig(**v3)))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Parent/teacher entry reward sweep with guard evaluation.")
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--teacher-epochs", type=int, default=14)
    args = p.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    parent_base, runner, add_cfg, overlay, deep_model, deep_payload, teacher_cols, base_report, base_exp, teacher_payload_base, baseline_metrics = _load_stack()
    fee = float(parent_base["config"]["fee"])
    slip = float(parent_base["config"]["slip"])
    guard_cfg = ExitGuardConfig(
        name="guard_soft3_hard1p45",
        hard_sl_mult=1.45,
        soft_sl_mult=1.0,
        early_bars=18,
        early_sl_mult=1.35,
        soft_min_hold=3,
        soft_persist_bars=3,
        regime_bad_th=0.50,
        flow_bad_th=0.02,
        giveback_trigger=0.72,
        giveback_min_mfe=0.014,
        giveback_min_hold=3,
        entry_quality_min=-999.0,
        entry_conf_min=0.0,
        same_side_entry_gap=0,
        cooldown_after_hard_stop=0,
        cooldown_after_soft_stop=0,
        cooldown_after_giveback=0,
    )
    limit_cfg = _default_limit_cfg()

    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train_all = _merge_state24(train_all, alpha3_full.SIDE_CLEAN4_2025)
    eval_df = _merge_state24(eval_df, alpha3_full.SIDE_CLEAN4_2026)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)

    state24_cols = sorted(c for c in train_all.columns if c.startswith(alpha3_full.TARGET_STATE24_PREFIX))
    feature_cols = _feature_cols(parent_base, state24_cols)

    val_q = v27._predict_all(deep_model, val_df, deep_payload["seq_cols"], deep_payload["norm"])
    eval_q = v27._predict_all(deep_model, eval_df, deep_payload["seq_cols"], deep_payload["norm"])

    rows: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    for name, cfg in _variant_cfgs(dict(parent_base["config"])):
        x_parent, y_parent, _ = build_training_set(
            train_df,
            cfg=cfg,
            stride_bars=6,
            batch_size=512,
            feature_cols=feature_cols,
        )
        parent_bundle = train_policy(
            x_parent.reindex(columns=feature_cols),
            y_parent,
            cfg=cfg,
            random_state=int(args.seed),
            feature_cols=feature_cols,
        )
        train_parent = predict_policy_frame(parent_bundle, train_df, close=_close(train_df))
        val_parent = predict_policy_frame(parent_bundle, val_df, close=_close(val_df))
        eval_parent = predict_policy_frame(parent_bundle, eval_df, close=_close(eval_df))

        train_features = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
        train_seq = alpha2.teacher._seq_tensor(train_features, np.arange(len(train_df), dtype=np.int64), feature_cols)
        buckets = tuple(float(x) for x in cfg.notional_buckets)
        teacher_model, teacher_meta = alpha2.teacher._train_teacher_model(
            train_seq,
            train_parent["action"].astype(int).to_numpy(dtype=np.int64),
            pd.to_numeric(train_parent["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            alpha2.teacher._bucket_labels(train_parent, buckets),
            n_buckets=len(buckets),
            epochs=int(args.teacher_epochs),
        )
        val_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
        eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
        val_teacher = alpha2.teacher._predict_deep(teacher_model, val_features, feature_cols, teacher_meta["norm"])
        eval_teacher = alpha2.teacher._predict_deep(teacher_model, eval_features, feature_cols, teacher_meta["norm"])

        best_rt = None
        best_val_score = -1e18
        best_val_metrics = None
        for rt in alpha2._runtimes():
            val_dec = alpha2._decisions(val_parent, val_teacher, buckets, rt)
            val_metrics = _metrics_guard(val_df, parent_bundle, runner, add_cfg, val_q, val_dec, overlay, limit_cfg, guard_cfg, fee=fee, slip=slip)
            sc = _score(val_metrics)
            if sc > best_val_score:
                best_val_score = sc
                best_rt = rt
                best_val_metrics = val_metrics
        if best_rt is None or best_val_metrics is None:
            raise RuntimeError(f"runtime selection failed for {name}")

        eval_dec = alpha2._decisions(eval_parent, eval_teacher, buckets, best_rt)
        eval_metrics = _metrics_guard(eval_df, parent_bundle, runner, add_cfg, eval_q, eval_dec, overlay, limit_cfg, guard_cfg, fee=fee, slip=slip)
        c3 = eval_metrics["cost3"]
        row = {
            "variant": name,
            "val_score": float(best_val_score),
            "val_cost3_pnl": float(best_val_metrics["cost3"]["pnl"]),
            "oos_score": float(_score(eval_metrics)),
            "oos_cost3_pnl": float(c3["pnl"]),
            "oos_cost3_mdd": float(c3["mdd"]),
            "oos_cost3_wr": float(c3["wr"]),
            "oos_cost3_trades": int(c3["trades"]),
            "oos_sl_ratio": float(_sl_ratio(c3)),
            "turnover_bonus": float(cfg.turnover_bonus),
            "hold_penalty": float(cfg.hold_penalty),
            "cash_score": float(cfg.cash_score),
            "adverse_penalty": float(cfg.adverse_penalty),
        }
        rows.append(row)
        detail.append(
            {
                "variant": name,
                "cfg": asdict(cfg),
                "selected_runtime": asdict(best_rt),
                "metrics": eval_metrics,
            }
        )

    grid = pd.DataFrame(rows).sort_values("val_score", ascending=False).reset_index(drop=True)
    grid.to_csv(GRID_OUT, index=False)
    best_name = str(grid.iloc[0]["variant"])
    best_row = next(r for r in rows if r["variant"] == best_name)

    baseline_c3 = baseline_metrics["cost3"]
    delta = {
        "pnl": float(best_row["oos_cost3_pnl"] - float(baseline_c3["pnl"])),
        "mdd": float(best_row["oos_cost3_mdd"] - float(baseline_c3["mdd"])),
        "wr": float(best_row["oos_cost3_wr"] - float(baseline_c3["wr"])),
        "trades": int(best_row["oos_cost3_trades"] - int(baseline_c3["trades"])),
        "sl_ratio": float(best_row["oos_sl_ratio"] - _sl_ratio(baseline_c3)),
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Retrain parent/teacher with entry reward-shape sweeps (turnover_bonus/hold_penalty/cash_score) while keeping deep/runner/execution fixed; evaluate with guard_soft3_hard1p45.",
        "protocol": {
            "selection_uses_2026": False,
            "selection_window": "2025-10-01..2025-12-31",
            "oos_window": "2026 fixed OOS",
        },
        "base_model": str(BASE_REPORT),
        "baseline_cost3": baseline_c3,
        "guard_config": asdict(guard_cfg),
        "grid": str(GRID_OUT),
        "rows": rows,
        "selected_variant": best_name,
        "selected_row": best_row,
        "delta_vs_baseline": delta,
        "detail": detail,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "grid": str(GRID_OUT), "selected_variant": best_name}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
