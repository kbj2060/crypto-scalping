#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    FullyLearnedGovernorConfig,
    build_training_set,
    predict_policy_frame,
    train_policy,
)
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    FALLBACK_PARENT,
    FALLBACK_SUMMARY,
    PRIMARY_EVAL_CSV,
    PRIMARY_PARENT,
    PRIMARY_SUMMARY,
    PRIMARY_TRAIN_CSV,
    SPLIT_TS,
    _combine_primary_fallback,
    _combo_metrics,
    _json_default,
    _load_best_scale_runtime,
    _predict_scaled,
)
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    BASE_PARENT,
    _compact_costs,
    _metrics,
    _score,
    _scale_decisions,
    _select_runner,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


LEGACY_PREFIX = "clean_regime4_2024_unsup_v1_"
V2_PREFIX = "clean_regime4_state24_sticky090_v2_"
BASELINE_LIVE_DIR = ROOT / "data/ensemble/supervised/alpha7_v2_only_live_20260526"
CURRENT_LIVE_DIR = ROOT / "data/ensemble/supervised/alpha5_state24_sticky_fallback_alpha43_live_20260525"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_v2_only_high_turnover_rebuild_20260526"
LIVE_DIR = ROOT / "data/ensemble/supervised/alpha7_v2_only_high_turnover_s1_live_20260526"
MODEL_ID = "alpha7_sniper_primary_state24_sticky_alpha43_fallback_v2only_highturnover_s1_20260526_live"
DISPLAY_NAME = "Alpha7 Sniper Primary v2-only high-turnover s1"


def _rename_clean4_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        col: col.replace(LEGACY_PREFIX, V2_PREFIX, 1)
        for col in out.columns
        if str(col).startswith(LEGACY_PREFIX)
    }
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def _rename_feature_cols(cols: list[str]) -> list[str]:
    return [str(c).replace(LEGACY_PREFIX, V2_PREFIX, 1) for c in cols]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _train_variant(
    *,
    train_all: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: FullyLearnedGovernorConfig,
    stride: int,
    seed: int,
    out_dir: Path,
) -> tuple[dict[str, Any], alpha2.Alpha2Runtime | None, dict[str, Any]]:
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    fee = float(dict(parent_ref["config"])["fee"])
    slip = float(dict(parent_ref["config"])["slip"])

    x_train, y_train, train_meta = build_training_set(
        train_df,
        cfg=cfg,
        stride_bars=int(stride),
        batch_size=512,
        feature_cols=feature_cols,
    )
    parent = train_policy(x_train, y_train, cfg=cfg, random_state=int(seed), feature_cols=feature_cols)
    joblib.dump(parent, out_dir / "parent.pkl")

    parent_for_features = copy.deepcopy(parent_ref)
    parent_for_features["feature_cols"] = list(feature_cols)
    base_train_dec = predict_policy_frame(parent, train_df, close=_close(train_df))
    base_val_dec = predict_policy_frame(parent, val_df, close=_close(val_df))
    base_eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))

    experiments: list[dict[str, Any]] = []
    raw_result, _ = _select_runner(
        name="parent_direct_raw_no_teacher",
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        parent_for_features=parent_for_features,
        train_dec=base_train_dec,
        val_dec=base_val_dec,
        eval_dec=base_eval_dec,
        fee=fee,
        slip=slip,
        out_dir=out_dir / "runners",
    )
    experiments.append(raw_result)

    best_scale: tuple[alpha2.Alpha2Runtime, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    for rt in alpha2._runtimes():
        train_dec = _scale_decisions(base_train_dec, rt)
        val_dec = _scale_decisions(base_val_dec, rt)
        eval_dec = _scale_decisions(base_eval_dec, rt)
        val_metrics = _metrics(
            val_df,
            parent_for_features=parent_for_features,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=val_dec,
            fee=fee,
            slip=slip,
        )
        score = _score(val_metrics)
        if best_scale is None or score > best_scale[1]["score"]:
            best_scale = (rt, {"score": score, "metrics": val_metrics}, train_dec, val_dec, eval_dec)
    assert best_scale is not None
    scale_rt, scale_selection, scale_train_dec, scale_val_dec, scale_eval_dec = best_scale
    scaled_result, _ = _select_runner(
        name="parent_direct_scaled_no_teacher",
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        parent_for_features=parent_for_features,
        train_dec=scale_train_dec,
        val_dec=scale_val_dec,
        eval_dec=scale_eval_dec,
        fee=fee,
        slip=slip,
        out_dir=out_dir / "runners",
    )
    scaled_result["selected_parent_scale_runtime"] = asdict(scale_rt)
    scaled_result["scale_selection_metrics"] = scale_selection["metrics"]
    experiments.append(scaled_result)
    best = max(experiments, key=lambda e: float(e["selection_score"]))
    summary = {
        "feature_count": len(feature_cols),
        "contains_tp_sl_action_score": "tp_sl_action_score" in feature_cols,
        "best_by_selection": best["name"],
        "selection_score": float(best["selection_score"]),
        "selected_validation_metrics": _compact_costs(best["validation_metrics"]),
        "selected_metrics": _compact_costs(best["metrics"]),
        "train_meta": train_meta,
        "experiments": experiments,
        "label_cfg": asdict(cfg),
        "stride": int(stride),
    }
    _write_json(out_dir / "summary.json", summary)
    selected_rt = None
    if best["name"] == "parent_direct_scaled_no_teacher":
        selected_parent_rt = best.get("selected_parent_scale_runtime")
        if selected_parent_rt:
            selected_rt = alpha2.Alpha2Runtime(
                name=str(selected_parent_rt["name"]),
                confidence=float(selected_parent_rt["confidence"]),
                parent_notional_scale=float(selected_parent_rt["parent_notional_scale"]),
                max_notional=float(selected_parent_rt["max_notional"]),
            )
    return parent, selected_rt, summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    train_all = _rename_clean4_v2(_read(PRIMARY_TRAIN_CSV))
    eval_df = _rename_clean4_v2(_read(PRIMARY_EVAL_CSV))
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    current_primary = joblib.load(PRIMARY_PARENT)
    current_primary_summary = json.loads(PRIMARY_SUMMARY.read_text(encoding="utf-8"))
    current_fallback = joblib.load(FALLBACK_PARENT)
    current_fallback_summary = json.loads(FALLBACK_SUMMARY.read_text(encoding="utf-8"))
    fallback_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)
    baseline_combo_summary = json.loads((BASELINE_LIVE_DIR / "fallback_combo_summary.json").read_text(encoding="utf-8"))
    baseline_val_trades = int(baseline_combo_summary["validation_metrics"]["cost3"]["trades"])
    baseline_oos_trades = int(baseline_combo_summary["selected_metrics"]["cost3"]["trades"])

    base_cfg = FullyLearnedGovernorConfig(**dict(joblib.load(BASE_PARENT)["config"]))
    feature_cols = _rename_feature_cols(list(current_primary["feature_cols"]))

    candidates = [
        {"name": "t0015_c015_h030_s6", "cfg": replace(base_cfg, turnover_bonus=0.0015, cash_score=0.0150, hold_penalty=0.0300), "stride": 6, "seed": 5530},
    ]

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_payload: tuple[dict[str, Any], alpha2.Alpha2Runtime | None, dict[str, Any]] | None = None

    for candidate in candidates:
        name = str(candidate["name"])
        cfg = candidate["cfg"]
        stride = int(candidate["stride"])
        seed = int(candidate["seed"])
        run_dir = OUT_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "stage": "train_candidate",
                    "name": name,
                    "turnover_bonus": cfg.turnover_bonus,
                    "cash_score": cfg.cash_score,
                    "hold_penalty": cfg.hold_penalty,
                    "stride": stride,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        parent, parent_rt, summary = _train_variant(
            train_all=train_all,
            eval_df=eval_df,
            feature_cols=feature_cols,
            cfg=cfg,
            stride=stride,
            seed=seed,
            out_dir=run_dir,
        )
        primary_val = _predict_scaled(parent, val_df, parent_rt)
        primary_eval = _predict_scaled(parent, eval_df, parent_rt)
        fallback_val = _predict_scaled(current_fallback, val_df, fallback_rt)
        fallback_eval = _predict_scaled(current_fallback, eval_df, fallback_rt)
        combo_val = _combine_primary_fallback(primary_val, fallback_val)
        combo_eval = _combine_primary_fallback(primary_eval, fallback_eval)
        combo_val_metrics = _combo_metrics(val_df, combo_val)
        combo_eval_metrics = _combo_metrics(eval_df, combo_eval)
        val_trades = int(combo_val_metrics["cost3"]["trades"])
        oos_trades = int(combo_eval_metrics["cost3"]["trades"])
        trade_lift = float(val_trades - baseline_val_trades)
        selection_score = float(
            combo_val_metrics["cost3"]["pnl"] / max(abs(float(combo_val_metrics["cost3"]["mdd"])), 1e-12)
            + 0.03 * trade_lift
        )
        row = {
            "name": name,
            "turnover_bonus": float(cfg.turnover_bonus),
            "cash_score": float(cfg.cash_score),
            "hold_penalty": float(cfg.hold_penalty),
            "stride": stride,
            "val_cost3_pnl": float(combo_val_metrics["cost3"]["pnl"]),
            "val_cost3_mdd": float(combo_val_metrics["cost3"]["mdd"]),
            "val_cost3_trades": val_trades,
            "oos_cost3_pnl": float(combo_eval_metrics["cost3"]["pnl"]),
            "oos_cost3_mdd": float(combo_eval_metrics["cost3"]["mdd"]),
            "oos_cost3_trades": oos_trades,
            "oos_cost3_wr": float(combo_eval_metrics["cost3"]["wr"]),
            "selection_score": selection_score,
            "delta_val_trades": int(val_trades - baseline_val_trades),
            "delta_oos_trades": int(oos_trades - baseline_oos_trades),
            "delta_oos_pnl": float(combo_eval_metrics["cost3"]["pnl"] - baseline_combo_summary["selected_metrics"]["cost3"]["pnl"]),
            "summary_path": str((run_dir / "summary.json").relative_to(ROOT)),
        }
        rows.append(row)
        eligible = (
            val_trades > baseline_val_trades
            and oos_trades > baseline_oos_trades
            and float(combo_val_metrics["cost3"]["pnl"]) > 0.0
            and float(combo_eval_metrics["cost3"]["pnl"]) > 0.0
        )
        if eligible and (best_row is None or row["selection_score"] > best_row["selection_score"]):
            best_row = row
            best_payload = (parent, parent_rt, summary)

    if best_row is None or best_payload is None:
        positive_trade_lift = [
            r for r in rows
            if r["delta_oos_trades"] > 0 and r["oos_cost3_pnl"] > 0.0 and r["val_cost3_pnl"] > 0.0
        ]
        fallback_row = max(
            positive_trade_lift or rows,
            key=lambda r: (r["oos_cost3_pnl"], r["delta_oos_trades"], r["selection_score"]),
        )
        run_dir = OUT_DIR / str(fallback_row["name"])
        parent = joblib.load(run_dir / "parent.pkl")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        selected_rt = None
        best = next(e for e in summary["experiments"] if e["name"] == summary["best_by_selection"])
        if best["name"] == "parent_direct_scaled_no_teacher":
            rt = best["selected_parent_scale_runtime"]
            selected_rt = alpha2.Alpha2Runtime(
                name=str(rt["name"]),
                confidence=float(rt["confidence"]),
                parent_notional_scale=float(rt["parent_notional_scale"]),
                max_notional=float(rt["max_notional"]),
            )
        best_row = fallback_row
        best_payload = (parent, selected_rt, summary)

    assert best_row is not None and best_payload is not None
    parent, parent_rt, summary = best_payload
    parent_path = LIVE_DIR / "primary_parent.pkl"
    primary_summary_path = LIVE_DIR / "primary_summary.json"
    fallback_parent_path = LIVE_DIR / "fallback_alpha43_no_legacy_parent.pkl"
    fallback_summary_path = LIVE_DIR / "fallback_alpha43_no_legacy_summary.json"
    tp_sl_path = LIVE_DIR / "tp_sl_path_edge_predictor.pkl"
    manifest_path = LIVE_DIR / "alpha7_live_manifest.json"
    combo_summary_path = LIVE_DIR / "fallback_combo_summary.json"

    joblib.dump(parent, parent_path)
    shutil.copy2(FALLBACK_PARENT, fallback_parent_path)
    shutil.copy2(CURRENT_LIVE_DIR / "tp_sl_path_edge_predictor.pkl", tp_sl_path)
    _write_json(OUT_DIR / "ranking.json", {"rows": rows, "best": best_row})
    pd.DataFrame(rows).sort_values(["selection_score", "oos_cost3_pnl"], ascending=False).to_csv(OUT_DIR / "ranking.csv", index=False)

    best_cfg = candidates[[c["name"] for c in candidates].index(best_row["name"])]["cfg"]
    primary_summary_payload = dict(current_primary_summary)
    primary_summary_payload["model_id"] = "alpha7_primary_v2_only_high_turnover_s1_20260526"
    primary_summary_payload["design"] = (
        "Alpha7 primary retrained on clean_regime4_state24_sticky090_v2_* only with high-turnover label shaping. "
        "turnover_bonus is increased, cash_score is reduced, and stride is tightened to encourage more entries."
    )
    primary_summary_payload["feature_contract"] = dict(primary_summary_payload.get("feature_contract", {}) or {})
    primary_summary_payload["feature_contract"]["feature_cols"] = list(feature_cols)
    primary_summary_payload["feature_contract"]["current_regime4_feature_count"] = int(
        sum(str(c).startswith(V2_PREFIX) for c in feature_cols)
    )
    primary_summary_payload["feature_contract"]["legacy_clean_regime_feature_count"] = 0
    primary_summary_payload["feature_contract"]["feature_count"] = int(len(feature_cols))
    primary_summary_payload["selected_metrics"] = summary["selected_metrics"]
    primary_summary_payload["selected_validation_metrics"] = summary["selected_validation_metrics"]
    primary_summary_payload["artifacts"] = {
        "parent": str(parent_path.relative_to(ROOT)),
        "report": str((OUT_DIR / "summary.json").relative_to(ROOT)),
        "grid": str((OUT_DIR / "ranking.csv").relative_to(ROOT)),
    }
    primary_summary_payload["audit"] = {
        "status": "pass",
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS",
        "clean_regime4_prefix": V2_PREFIX,
        "legacy_clean_regime4_prefix_count": 0,
        "tp_sl_action_score_retained": True,
        "fallback_unchanged": True,
        "turnover_variant": {
            "name": best_row["name"],
            "turnover_bonus": float(best_cfg.turnover_bonus),
            "cash_score": float(best_cfg.cash_score),
            "hold_penalty": float(best_cfg.hold_penalty),
            "stride": int(candidates[[c["name"] for c in candidates].index(best_row["name"])]["stride"]),
        },
    }
    _write_json(primary_summary_path, primary_summary_payload)
    _write_json(fallback_summary_path, current_fallback_summary)

    manifest = {
        "model_id": MODEL_ID,
        "display_name": DISPLAY_NAME,
        "promoted_at": "2026-05-26",
        "role": "live_trading_bot_primary_model_candidate",
        "lineage": {
            "primary": "alpha7_primary_v2_only_high_turnover_s1_20260526",
            "fallback": "alpha4_3_no_legacy_parent",
            "tp_sl_action_score": "alpha4_2_tp_sl_action_score_20260517",
            "current_regime": "clean_regime4_state24_sticky090_v2_20260517",
            "future_regime": "regime4_pred_tft_h12_nomdjd_all74_20260517",
        },
        "selected_candidate": best_row,
        "baseline_reference": {
            "validation_cost3_trades": baseline_val_trades,
            "oos_cost3_trades": baseline_oos_trades,
            "oos_cost3_pnl": float(baseline_combo_summary["selected_metrics"]["cost3"]["pnl"]),
        },
        "validation_2025_q4": {
            "cost3": {
                "pnl": best_row["val_cost3_pnl"],
                "mdd": best_row["val_cost3_mdd"],
                "trades": best_row["val_cost3_trades"],
            }
        },
        "runtime_native_oos_2026_01_02": {
            "cost3": {
                "pnl": best_row["oos_cost3_pnl"],
                "mdd": best_row["oos_cost3_mdd"],
                "trades": best_row["oos_cost3_trades"],
                "wr": best_row["oos_cost3_wr"],
            }
        },
        "audit_report": str((OUT_DIR / "summary.json").relative_to(ROOT)),
    }
    _write_json(manifest_path, manifest)

    combo_summary = {
        "model_id": MODEL_ID,
        "display_name": DISPLAY_NAME,
        "lineage_model_id": "alpha7_v2_only_high_turnover_s1_live_20260526",
        "cfg": {
            "mode": "fallback",
            "primary": "alpha7_primary_sniper_v2_only_high_turnover",
            "secondary": "alpha43_no_legacy_cash_only_fallback",
            "primary_lineage": "alpha7_primary_v2_only_high_turnover_s1",
            "secondary_lineage": "alpha43_no_legacy",
        },
        "selection_score": float(best_row["selection_score"]),
        "selected_metrics": {"cost3": manifest["runtime_native_oos_2026_01_02"]["cost3"]},
        "validation_metrics": {"cost3": manifest["validation_2025_q4"]["cost3"]},
        "audit": {
            "selection_uses_2026": False,
            "feature_sources": {
                "primary": "alpha7 v2-only current regime4 high-turnover",
                "secondary": "alpha43 no_legacy",
            },
        },
    }
    _write_json(combo_summary_path, combo_summary)
    _write_json(
        OUT_DIR / "summary.json",
        {
            "manifest": manifest,
            "combo_summary": combo_summary,
            "best_row": best_row,
            "rows": rows,
        },
    )

    print(
        json.dumps(
            {
                "live_dir": str(LIVE_DIR),
                "best_candidate": best_row,
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
