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
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    OLD_CLEAN_PREFIX,
    _compact_costs,
    _metrics,
    _score,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


DEFAULT_META_DIR = ROOT / "tmp/causal_regen_20260516/alpha4_3_sticky_alpha61_oof_meta_20260525"
DEFAULT_MODEL_DIR = ROOT / "tmp/causal_regen_20260516/alpha4_3_sticky_regime_retrain_20260524/sticky_current"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/sticky_current_alpha61_external_overlay_20260525"
STICKY_PREFIX = "clean_regime4_2024_unsup_v1_"


def _load_runner(path: Path) -> tuple[dict[str, Any], CostRunnerConfig]:
    payload = joblib.load(path)
    return payload["cost_runner"], CostRunnerConfig(**payload["selected_config"])


def _load_scale_runtime(summary_path: Path) -> alpha2.Alpha2Runtime | None:
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    best_name = summary.get("best_by_selection")
    for exp in summary.get("experiments", []):
        if exp.get("name") != best_name:
            continue
        rt = exp.get("selected_parent_scale_runtime")
        if not rt:
            return None
        return alpha2.Alpha2Runtime(
            name=str(rt["name"]),
            confidence=float(rt["confidence"]),
            parent_notional_scale=float(rt["parent_notional_scale"]),
            max_notional=float(rt["max_notional"]),
        )
    return None


def _parent_for_features(parent: dict[str, Any]) -> dict[str, Any]:
    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    out = copy.deepcopy(parent_ref)
    out["feature_cols"] = list(parent["feature_cols"])
    return out


def _base_decisions(parent: dict[str, Any], frame: pd.DataFrame, rt: alpha2.Alpha2Runtime | None) -> pd.DataFrame:
    dec = predict_policy_frame(parent, frame, close=_close(frame))
    if rt is not None:
        dec = alpha2._scale_parent_notional(dec, rt)
    return dec.reset_index(drop=True)


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in frame.columns:
        return np.full(len(frame), float(default), dtype=np.float64)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).to_numpy(dtype=np.float64)


def _alpha61_state(frame: pd.DataFrame, *, consensus_margin: float, edge_margin: float) -> dict[str, np.ndarray]:
    long_edge = _num(frame, "a61_long_edge_sum")
    short_edge = _num(frame, "a61_short_edge_sum")
    long_cons = _num(frame, "a61_consensus_long")
    short_cons = _num(frame, "a61_consensus_short")
    cons_delta = long_cons - short_cons
    edge_delta = long_edge - short_edge
    side = np.zeros(len(frame), dtype=np.int64)
    side[(cons_delta >= consensus_margin) | (edge_delta >= edge_margin)] = 1
    side[(cons_delta <= -consensus_margin) | (edge_delta <= -edge_margin)] = -1
    active = _num(frame, "a61_active_model_count") > 0
    side[~active] = 0
    return {
        "side": side,
        "risk": _num(frame, "a61_risk_opposition"),
        "quality_top": _num(frame, "a61_quality_top"),
        "active_count": _num(frame, "a61_active_model_count"),
        "entropy": _num(frame, "a61_disagreement_entropy"),
        "cons_abs": np.abs(cons_delta),
        "edge_abs": np.abs(edge_delta),
    }


def _scale_notional(dec: pd.DataFrame, scale: np.ndarray) -> pd.DataFrame:
    out = dec.copy()
    scale = np.asarray(scale, dtype=np.float64)
    active = (pd.to_numeric(out["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64) != 0) & (
        pd.to_numeric(out["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64) != 0
    )
    notional = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    leverage = pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    scaled = np.clip(notional * scale, 0.0, 2.75)
    out.loc[active, "notional_exposure"] = scaled[active]
    out.loc[active, "position_fraction"] = scaled[active] / np.maximum(leverage[active], 1e-12)
    zero = active & (scaled <= 1e-12)
    out.loc[zero, ["action", "side", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[zero, "leverage"] = 1.0
    return out


def _overlay_decisions(base_dec: pd.DataFrame, frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if cfg["mode"] == "base":
        return base_dec.copy()
    st = _alpha61_state(frame, consensus_margin=float(cfg["consensus_margin"]), edge_margin=float(cfg["edge_margin"]))
    side = pd.to_numeric(base_dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    active = side != 0
    same = active & (st["side"] == side)
    opp = active & (st["side"] == -side)
    unsupported = active & (st["side"] == 0)
    risk = active & (st["risk"] >= float(cfg["risk_threshold"]))
    high_disagreement = active & (st["entropy"] >= float(cfg["entropy_threshold"]))
    strong_same = same & (st["active_count"] >= float(cfg["min_active"])) & (st["quality_top"] >= float(cfg["quality_threshold"]))

    scale = np.ones(len(base_dec), dtype=np.float64)
    if cfg["mode"] == "risk_veto":
        scale[risk] = 0.0
    elif cfg["mode"] == "opp_veto":
        scale[opp] = 0.0
    elif cfg["mode"] == "risk_or_opp_veto":
        scale[risk | opp] = 0.0
    elif cfg["mode"] == "soft_router":
        scale[unsupported] = float(cfg["unsupported_scale"])
        scale[opp] = float(cfg["opp_scale"])
        scale[risk & ~same] = 0.0
        scale[high_disagreement & ~same] = np.minimum(scale[high_disagreement & ~same], float(cfg["disagreement_scale"]))
        scale[strong_same] = np.maximum(scale[strong_same], float(cfg["same_boost"]))
    elif cfg["mode"] == "sniper_veto":
        scale[:] = float(cfg["base_scale"])
        scale[~active] = 0.0
        scale[unsupported | opp | risk | high_disagreement] = 0.0
        scale[strong_same] = float(cfg["same_boost"])
    else:
        raise ValueError(f"unknown overlay mode: {cfg['mode']}")
    return _scale_notional(base_dec, scale)


def _candidate_grid(val_frame: pd.DataFrame) -> list[dict[str, Any]]:
    q80 = float(np.nanquantile(_num(val_frame, "a61_quality_top"), 0.80))
    q90 = float(np.nanquantile(_num(val_frame, "a61_quality_top"), 0.90))
    rows: list[dict[str, Any]] = [
        {
            "mode": "base",
            "consensus_margin": 1.0,
            "edge_margin": 1.0,
            "risk_threshold": 2.0,
            "entropy_threshold": 2.0,
            "min_active": 0.0,
            "quality_threshold": 0.0,
            "unsupported_scale": 1.0,
            "opp_scale": 1.0,
            "disagreement_scale": 1.0,
            "same_boost": 1.0,
            "base_scale": 1.0,
        }
    ]
    for mode in ("risk_veto", "opp_veto", "risk_or_opp_veto"):
        for cm in (0.001, 1 / 3):
            for em in (0.0001, 0.0008):
                rows.append(
                    {
                        "mode": mode,
                        "consensus_margin": cm,
                        "edge_margin": em,
                        "risk_threshold": 1.0,
                        "entropy_threshold": 2.0,
                        "min_active": 0.0,
                        "quality_threshold": 0.0,
                        "unsupported_scale": 1.0,
                        "opp_scale": 1.0,
                        "disagreement_scale": 1.0,
                        "same_boost": 1.0,
                        "base_scale": 1.0,
                    }
                )
    for cm in (0.001, 1 / 3):
        for em in (0.0001, 0.0008):
            for unsupported in (0.65, 1.0):
                for opp_scale in (0.0, 0.35):
                    for boost in (1.0, 1.20):
                        rows.append(
                            {
                                "mode": "soft_router",
                                "consensus_margin": cm,
                                "edge_margin": em,
                                "risk_threshold": 1.0,
                                "entropy_threshold": 0.92,
                                "min_active": 4.0,
                                "quality_threshold": q80,
                                "unsupported_scale": unsupported,
                                "opp_scale": opp_scale,
                                "disagreement_scale": 0.65,
                                "same_boost": boost,
                                "base_scale": 1.0,
                            }
                        )
    for cm in (0.001, 1 / 6):
        for em in (0.0001, 0.0004):
            for base_scale in (0.20, 0.35):
                rows.append(
                    {
                        "mode": "sniper_veto",
                        "consensus_margin": cm,
                        "edge_margin": em,
                        "risk_threshold": 1.0,
                        "entropy_threshold": 0.92,
                        "min_active": 5.0,
                        "quality_threshold": q90,
                        "unsupported_scale": 0.0,
                        "opp_scale": 0.0,
                        "disagreement_scale": 0.0,
                        "same_boost": 1.20,
                        "base_scale": base_scale,
                    }
                )
    return rows


def _select_robust_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = next(row for row in rows if row["mode"] == "base")
    min_trades = 0.90 * float(base["val_cost3_trades"])
    max_trades = 1.25 * float(base["val_cost3_trades"])
    min_pnl = float(base["val_cost3_pnl"]) + 5.0
    ranked = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    for row in ranked:
        if row["mode"] == "base":
            continue
        if float(row["val_cost3_pnl"]) < min_pnl:
            continue
        if not (min_trades <= float(row["val_cost3_trades"]) <= max_trades):
            continue
        if float(row["val_cost3_mdd"]) < float(base["val_cost3_mdd"]):
            continue
        return row
    return base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select an external Alpha6.1 OOF post-filter/router/veto overlay for sticky_current.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_META_DIR / "trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv")
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_META_DIR / "trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv")
    p.add_argument("--parent", type=Path, default=DEFAULT_MODEL_DIR / "parent.pkl")
    p.add_argument("--runner", type=Path, default=DEFAULT_MODEL_DIR / "runners/sticky_current__parent_direct_scaled_no_teacher_runner.pkl")
    p.add_argument("--variant-summary", type=Path, default=DEFAULT_MODEL_DIR / "sticky_current_summary.json")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    parent = joblib.load(args.parent)
    runner, runner_cfg = _load_runner(args.runner)
    scale_rt = _load_scale_runtime(args.variant_summary)
    fee = float(parent["config"]["fee"])
    slip = float(parent["config"]["slip"])

    forbidden = [c for c in parent["feature_cols"] if c.startswith(OLD_CLEAN_PREFIX)]
    if forbidden:
        raise ValueError(f"sticky_current parent still contains old regime proxy columns: {forbidden[:20]}")

    parent_for_features = _parent_for_features(parent)
    base_val = _base_decisions(parent, val_df, scale_rt)
    base_eval = _base_decisions(parent, eval_df, scale_rt)
    candidates = _candidate_grid(val_df)
    rows: list[dict[str, Any]] = []
    for idx, cfg in enumerate(candidates):
        val_dec = _overlay_decisions(base_val, val_df, cfg)
        val_metrics = _metrics(val_df, parent_for_features=parent_for_features, runner=runner, runner_cfg=runner_cfg, dec=val_dec, fee=fee, slip=slip)
        score = _score(val_metrics)
        row = {
            "candidate_id": idx,
            **cfg,
            "score": float(score),
            "val_cost1_pnl": val_metrics["cost1"]["pnl"],
            "val_cost1_mdd": val_metrics["cost1"]["mdd"],
            "val_cost1_trades": val_metrics["cost1"]["trades"],
            "val_cost2_pnl": val_metrics["cost2"]["pnl"],
            "val_cost3_pnl": val_metrics["cost3"]["pnl"],
            "val_cost3_mdd": val_metrics["cost3"]["mdd"],
            "val_cost3_trades": val_metrics["cost3"]["trades"],
        }
        rows.append(row)

    raw_best_row = max(rows, key=lambda row: float(row["score"]))
    selected_row = _select_robust_candidate(rows)
    best = {
        "candidate_id": int(selected_row["candidate_id"]),
        "cfg": candidates[int(selected_row["candidate_id"])],
        "score": float(selected_row["score"]),
    }
    val_dec = _overlay_decisions(base_val, val_df, best["cfg"])
    best["validation_metrics"] = _metrics(val_df, parent_for_features=parent_for_features, runner=runner, runner_cfg=runner_cfg, dec=val_dec, fee=fee, slip=slip)
    eval_dec = _overlay_decisions(base_eval, eval_df, best["cfg"])
    eval_metrics = _metrics(eval_df, parent_for_features=parent_for_features, runner=runner, runner_cfg=runner_cfg, dec=eval_dec, fee=fee, slip=slip)
    base_eval_metrics = _metrics(eval_df, parent_for_features=parent_for_features, runner=runner, runner_cfg=runner_cfg, dec=base_eval, fee=fee, slip=slip)

    ranking = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    ranking.to_csv(args.out_dir / "overlay_validation_ranking.csv", index=False)
    eval_dec.assign(timestamp=eval_df["timestamp"].to_numpy()).to_csv(args.out_dir / "selected_overlay_decisions_2026.csv", index=False)
    base_eval.assign(timestamp=eval_df["timestamp"].to_numpy()).to_csv(args.out_dir / "base_sticky_current_decisions_2026.csv", index=False)
    summary = {
        "model_id": "sticky_current_alpha61_external_overlay_20260525",
        "design": "sticky_current parent/runner is preserved; Alpha6.1 OOF meta signals are used only as an external post-filter/router/veto selected on 2025Q4 and fixed for 2026.",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "parent": str(args.parent),
        "runner": str(args.runner),
        "runner_config": asdict(runner_cfg),
        "scale_runtime": None if scale_rt is None else asdict(scale_rt),
        "fee": fee,
        "slip": slip,
        "split": {
            "overlay_selection": "2025-10-01..2025-12-31 using Alpha6.1 OOF meta columns",
            "oos": "2026 fixed OOS",
        },
        "audit": {
            "selection_uses_2026": False,
            "alpha61_train_meta_source": "OOF walk-forward columns from build_alpha61_oof_meta_for_sticky_current_20260525.py",
            "old_regime_proxy_feature_count": len(forbidden),
            "sticky_feature_count": int(sum(c.startswith(STICKY_PREFIX) for c in parent["feature_cols"])),
            "candidate_count": len(candidates),
        },
        "base_2026_metrics": _compact_costs(base_eval_metrics),
        "selected_candidate": best["cfg"],
        "selected_candidate_id": best["candidate_id"],
        "raw_best_validation_candidate": raw_best_row,
        "selection_guard": {
            "policy": "Prefer baseline unless an overlay improves 2025Q4 Cost3 by at least 5 points, does not worsen Cost3 MDD, and keeps Cost3 trades within 90%-125% of baseline.",
            "reason": "External Alpha6.1 overlay is a post-filter; trade-count collapse on validation is treated as an overfit risk.",
        },
        "selected_validation_score": best["score"],
        "selected_validation_metrics": _compact_costs(best["validation_metrics"]),
        "selected_2026_metrics": _compact_costs(eval_metrics),
        "validation_top10": ranking.head(10).to_dict(orient="records"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"selected": best["cfg"], "base": summary["base_2026_metrics"], "overlay": summary["selected_2026_metrics"]}, ensure_ascii=False, default=_json_default), flush=True)
    print(f"[out] {args.out_dir}", flush=True)
    print(ranking.head(10).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
