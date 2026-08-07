#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=PerformanceWarning)

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_ft_transformer_mtl_parent_v2_20260515 as ft_v2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.eval_alpha3_ft_v2_retrained_downstream_20260515 import _fit_cost_runner_with_decisions  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    OLD_CLEAN_PREFIX,
    _compact_costs,
    _metrics,
    _score,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _grid as _runner_grid  # noqa: E402


DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha4_3_sticky_tp_sl_action_score_20260524"
DEFAULT_TRAIN = DATA_DIR / "trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_EVAL = DATA_DIR / "trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/alpha_synergy_research_20260525"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    parent: Path
    summary: Path
    family: str


def _candidate_specs() -> list[CandidateSpec]:
    base = ROOT / "tmp/causal_regen_20260516"
    return [
        CandidateSpec(
            "alpha43_no_legacy",
            base / "alpha4_3_legacy_regime_block_ablation_alpha43basis_20260517/no_legacy/parent.pkl",
            base / "alpha4_3_legacy_regime_block_ablation_alpha43basis_20260517/no_legacy/no_legacy_summary.json",
            "alpha4.3_no_regime",
        ),
        CandidateSpec(
            "alpha43_sticky_current",
            base / "alpha4_3_sticky_regime_retrain_20260524/sticky_current/parent.pkl",
            base / "alpha4_3_sticky_regime_retrain_20260524/sticky_current/sticky_current_summary.json",
            "alpha4.3_sticky_regime",
        ),
        CandidateSpec(
            "alpha5_regime4_tp_sl",
            base / "alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517/parent.pkl",
            base / "alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517/alpha5_regime4_tp18_sl10_no_teacher_no_deep_summary.json",
            "alpha5_regime4",
        ),
        CandidateSpec(
            "alpha5_1_interactions",
            base / "alpha5_1_regime4_interactions_no_teacher_no_deep_20260517/parent.pkl",
            base / "alpha5_1_regime4_interactions_no_teacher_no_deep_20260517/alpha5_1_regime4_interactions_no_teacher_no_deep_summary.json",
            "alpha5_regime4_interactions",
        ),
        CandidateSpec(
            "alpha5_2_factor_bridge",
            base / "alpha5_2_regime4_factor_bridge_no_teacher_no_deep_20260517/parent.pkl",
            base / "alpha5_2_regime4_factor_bridge_no_teacher_no_deep_20260517/alpha5_2_regime4_factor_bridge_no_teacher_no_deep_summary.json",
            "alpha5_regime4_factor_bridge",
        ),
    ]


def _load_scale_runtime(summary_path: Path) -> alpha2.Alpha2Runtime | None:
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_name = summary.get("best_by_selection")
    for exp in summary.get("experiments", []):
        if best_name is not None and exp.get("name") != best_name:
            continue
        rt = exp.get("selected_parent_scale_runtime")
        if not rt:
            continue
        return alpha2.Alpha2Runtime(
            name=str(rt["name"]),
            confidence=float(rt["confidence"]),
            parent_notional_scale=float(rt["parent_notional_scale"]),
            max_notional=float(rt["max_notional"]),
        )
    return None


def _parent_for_features(feature_cols: list[str] | None = None) -> dict[str, Any]:
    ref = joblib.load(v31.DEFAULT_PARENT)
    out = copy.deepcopy(ref)
    if feature_cols is not None:
        out["feature_cols"] = list(feature_cols)
    return out


def _q0(df: pd.DataFrame) -> np.ndarray:
    return np.zeros((len(df), 2), dtype=np.float32)


def _base_decisions(parent: dict[str, Any], df: pd.DataFrame, rt: alpha2.Alpha2Runtime | None) -> pd.DataFrame:
    dec = predict_policy_frame(parent, df, close=_close(df)).reset_index(drop=True)
    if rt is not None:
        dec = alpha2._scale_parent_notional(dec, rt).reset_index(drop=True)
    return dec


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return (action != ACTION_CASH) & (side != 0)


def _quality(dec: pd.DataFrame) -> np.ndarray:
    if "quality_score" not in dec.columns:
        return np.zeros(len(dec), dtype=np.float64)
    return pd.to_numeric(dec["quality_score"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)


def _side(dec: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)


def _notional(dec: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(dec["notional_exposure"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)


def _zero_like(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy()
    cols = ["action", "side", "position_fraction", "notional_exposure", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]
    for col in cols:
        if col in out.columns:
            out[col] = 0
    if "leverage" in out.columns:
        out["leverage"] = 1.0
    return out


def _copy_rows(target: pd.DataFrame, source: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    out = target.copy()
    for col in source.columns:
        out.loc[mask, col] = source.loc[mask, col].to_numpy()
    return out


def _scale_active_notional(dec: pd.DataFrame, scale: np.ndarray) -> pd.DataFrame:
    out = dec.copy()
    active = _active(out)
    lev = pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    ntl = _notional(out)
    scaled = np.clip(ntl * scale, 0.0, 2.75)
    out.loc[active, "notional_exposure"] = scaled[active]
    out.loc[active, "position_fraction"] = scaled[active] / np.maximum(lev[active], 1e-12)
    kill = active & (scaled <= 1e-12)
    if np.any(kill):
        for col in ("action", "side", "position_fraction", "notional_exposure", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"):
            if col in out.columns:
                out.loc[kill, col] = 0
        if "leverage" in out.columns:
            out.loc[kill, "leverage"] = 1.0
    return out


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).to_numpy(dtype=np.float64)


def _regime_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    trend = _num(frame, "clean_regime4_2024_unsup_v1_trend_prob")
    micro = _num(frame, "clean_regime4_2024_unsup_v1_micro_prob")
    instability = _num(frame, "clean_regime4_2024_unsup_v1_instability_prob")
    confidence = _num(frame, "clean_regime4_2024_unsup_v1_confidence")
    return {
        "trend": (trend >= np.maximum(micro, instability)) & (confidence >= 0.25),
        "micro": (micro > trend) & (micro >= instability),
        "unstable": instability > np.maximum(trend, micro),
    }


def _combine(frame: pd.DataFrame, decisions: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.DataFrame:
    names = list(decisions)
    primary = decisions[str(cfg["primary"])].reset_index(drop=True)
    out = primary.copy()
    mode = str(cfg["mode"])
    if mode == "single":
        return out
    if mode == "fallback":
        secondary = decisions[str(cfg["secondary"])].reset_index(drop=True)
        mask = ~_active(out) & _active(secondary)
        return _copy_rows(out, secondary, mask)
    if mode == "confidence_mux":
        best_name = np.full(len(frame), str(cfg["primary"]), dtype=object)
        best_q = np.where(_active(out), _quality(out), -1e9)
        for name in names:
            dec = decisions[name].reset_index(drop=True)
            q = np.where(_active(dec), _quality(dec), -1e9) + float(cfg.get(f"bias__{name}", 0.0))
            take = q > best_q
            best_q[take] = q[take]
            best_name[take] = name
        out = _zero_like(primary)
        for name in names:
            mask = best_name == name
            out = _copy_rows(out, decisions[name].reset_index(drop=True), mask)
        return out
    if mode == "same_side_blend":
        primary_side = _side(primary)
        primary_active = _active(primary)
        scale = np.ones(len(frame), dtype=np.float64)
        out = primary.copy()
        for name in names:
            if name == cfg["primary"]:
                continue
            dec = decisions[name].reset_index(drop=True)
            side = _side(dec)
            same = primary_active & _active(dec) & (side == primary_side)
            opp = primary_active & _active(dec) & (side == -primary_side)
            vacant = ~primary_active & _active(dec)
            scale[same] += float(cfg["same_add_scale"])
            scale[opp] *= float(cfg["conflict_scale"])
            replace = vacant | (primary_active & (np.abs(_quality(dec)) > np.abs(_quality(out)) + float(cfg["quality_margin"])))
            out = _copy_rows(out, dec, replace)
            primary_active = _active(out)
            primary_side = _side(out)
        return _scale_active_notional(out, scale)
    if mode == "regime_mux":
        masks = _regime_masks(frame)
        out = decisions[str(cfg["micro"])].reset_index(drop=True).copy()
        out = _copy_rows(out, decisions[str(cfg["trend"])].reset_index(drop=True), masks["trend"])
        out = _copy_rows(out, decisions[str(cfg["unstable"])].reset_index(drop=True), masks["unstable"])
        fallback = decisions[str(cfg["fallback"])].reset_index(drop=True)
        out = _copy_rows(out, fallback, ~_active(out) & _active(fallback))
        return out
    raise ValueError(f"unknown combo mode: {mode}")


def _combo_grid(names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{"mode": "single", "primary": name} for name in names]
    for primary in names:
        for secondary in names:
            if primary != secondary:
                rows.append({"mode": "fallback", "primary": primary, "secondary": secondary})
    for primary in names:
        for same_add in (0.15, 0.30, 0.50):
            for conflict in (0.0, 0.35, 0.65):
                rows.append({"mode": "same_side_blend", "primary": primary, "same_add_scale": same_add, "conflict_scale": conflict, "quality_margin": 0.01})
    rows.append({"mode": "confidence_mux", "primary": names[0]})
    for trend in names:
        for micro in names:
            for unstable in names:
                if len({trend, micro, unstable}) < 2:
                    continue
                rows.append({"mode": "regime_mux", "trend": trend, "micro": micro, "unstable": unstable, "fallback": names[0], "primary": trend})
    return rows


def _select_runner(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    parent_for_features: dict[str, Any],
    train_dec: pd.DataFrame,
    val_dec: pd.DataFrame,
    eval_dec: pd.DataFrame,
    fee: float,
    slip: float,
    out_dir: Path,
    name: str,
) -> dict[str, Any]:
    runner = _fit_cost_runner_with_decisions(train_df, parent_for_features, train_dec, fee=fee, slip=slip)
    best_cfg: CostRunnerConfig | None = None
    best_val: dict[str, Any] | None = None
    best_score = -1e18
    rows: list[dict[str, Any]] = []
    for cfg in _runner_grid():
        val_metrics = _metrics(val_df, parent_for_features=parent_for_features, runner=runner, runner_cfg=cfg, dec=val_dec, fee=fee, slip=slip)
        score = _score(val_metrics)
        rows.append({"runner_config": cfg.name, "score": score, "val_cost3_pnl": val_metrics["cost3"]["pnl"], "val_cost3_mdd": val_metrics["cost3"]["mdd"], "val_cost3_trades": val_metrics["cost3"]["trades"]})
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_val = val_metrics
    assert best_cfg is not None and best_val is not None
    eval_metrics = _metrics(eval_df, parent_for_features=parent_for_features, runner=runner, runner_cfg=best_cfg, dec=eval_dec, fee=fee, slip=slip)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"cost_runner": runner, "selected_config": asdict(best_cfg), "combo": name}, out_dir / f"{name}_runner.pkl")
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(out_dir / f"{name}_runner_grid.csv", index=False)
    return {
        "runner_config": asdict(best_cfg),
        "selection_score": float(best_score),
        "validation_metrics": best_val,
        "metrics": eval_metrics,
        "runner_snapshot_count": int(runner.get("snapshot_count", 0)),
    }


def _decision_audit(frame: pd.DataFrame, dec: pd.DataFrame) -> dict[str, Any]:
    active = _active(dec)
    side = _side(dec)
    return {
        "rows": int(len(dec)),
        "timestamp_rows": int(len(frame)),
        "active_rows": int(active.sum()),
        "long_rows": int((active & (side > 0)).sum()),
        "short_rows": int((active & (side < 0)).sum()),
        "avg_active_notional": float(np.mean(_notional(dec)[active])) if np.any(active) else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Research Alpha model decision-stream synergies with 2025Q4 selection and fixed 2026 OOS.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--promote-top", type=int, default=8)
    p.add_argument("--max-combos", type=int, default=0, help="Debug limit. 0 means run the full grid.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    ref_parent = joblib.load(v31.DEFAULT_PARENT)
    fee = float(ref_parent["config"]["fee"])
    slip = float(ref_parent["config"]["slip"])
    parent_for_features = _parent_for_features(list(ref_parent["feature_cols"]))

    candidates: dict[str, dict[str, Any]] = {}
    for spec in _candidate_specs():
        if not spec.parent.exists():
            print(f"[skip] missing parent {spec.name}: {spec.parent}", flush=True)
            continue
        parent = joblib.load(spec.parent)
        old_cols = [c for c in parent.get("feature_cols", []) if str(c).startswith(OLD_CLEAN_PREFIX)]
        if old_cols:
            print(f"[skip] old-regime feature leak {spec.name}: {old_cols[:5]}", flush=True)
            continue
        rt = _load_scale_runtime(spec.summary)
        print(f"[candidate] {spec.name} family={spec.family} scale={None if rt is None else rt.name}", flush=True)
        candidates[spec.name] = {
            "spec": spec,
            "parent": parent,
            "scale_runtime": None if rt is None else asdict(rt),
            "train_dec": _base_decisions(parent, train_df, rt),
            "val_dec": _base_decisions(parent, val_df, rt),
            "eval_dec": _base_decisions(parent, eval_df, rt),
            "feature_count": int(len(parent.get("feature_cols", []))),
        }
    if len(candidates) < 2:
        raise RuntimeError("need at least two leak-free alpha candidates")

    names = list(candidates)
    screen_rows: list[dict[str, Any]] = []
    combo_cfgs: dict[str, dict[str, Any]] = {}
    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    combo_grid = _combo_grid(names)
    if int(args.max_combos) > 0:
        combo_grid = combo_grid[: int(args.max_combos)]
    for idx, cfg in enumerate(combo_grid):
        key = f"c{idx:04d}_{cfg['mode']}"
        combo_cfgs[key] = dict(cfg)
        val_dec = _combine(val_df, {n: candidates[n]["val_dec"] for n in names}, cfg)
        eval_dec = _combine(eval_df, {n: candidates[n]["eval_dec"] for n in names}, cfg)
        val_metrics = _metrics(
            val_df,
            parent_for_features=parent_for_features,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=val_dec,
            fee=fee,
            slip=slip,
        )
        eval_metrics = _metrics(
            eval_df,
            parent_for_features=parent_for_features,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=eval_dec,
            fee=fee,
            slip=slip,
        )
        row = {
            "candidate_key": key,
            "cfg": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
            "screen_score": _score(val_metrics),
            "val_cost1_pnl": val_metrics["cost1"]["pnl"],
            "val_cost1_mdd": val_metrics["cost1"]["mdd"],
            "val_cost1_trades": val_metrics["cost1"]["trades"],
            "val_cost3_pnl": val_metrics["cost3"]["pnl"],
            "val_cost3_mdd": val_metrics["cost3"]["mdd"],
            "val_cost3_trades": val_metrics["cost3"]["trades"],
            "oos_screen_score": _score(eval_metrics),
            "oos_cost1_pnl": eval_metrics["cost1"]["pnl"],
            "oos_cost1_mdd": eval_metrics["cost1"]["mdd"],
            "oos_cost1_trades": eval_metrics["cost1"]["trades"],
            "oos_cost3_pnl": eval_metrics["cost3"]["pnl"],
            "oos_cost3_mdd": eval_metrics["cost3"]["mdd"],
            "oos_cost3_trades": eval_metrics["cost3"]["trades"],
            "runner_config": noop_cfg.name,
            "runner_snapshot_count": 0,
        }
        screen_rows.append(row)
        print(
            f"[screen] {key} val_c3={row['val_cost3_pnl']:.2f} oos_c3={row['oos_cost3_pnl']:.2f} "
            f"trades={row['oos_cost3_trades']}",
            flush=True,
        )

    screen = pd.DataFrame(screen_rows)
    screen.sort_values("screen_score", ascending=False).to_csv(args.out_dir / "screen_ranking_validation.csv", index=False)
    screen.sort_values("oos_cost3_pnl", ascending=False).to_csv(args.out_dir / "screen_ranking_oos_oracle_research_only.csv", index=False)

    validation_promote_keys = list(screen.sort_values("screen_score", ascending=False).head(int(args.promote_top))["candidate_key"])
    oracle_promote_keys = list(screen.sort_values("oos_cost3_pnl", ascending=False).head(max(2, int(args.promote_top) // 2))["candidate_key"])
    promote_keys = list(dict.fromkeys(validation_promote_keys + oracle_promote_keys))
    promoted_rows: list[dict[str, Any]] = []
    promoted_decisions: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for key in promote_keys:
        cfg = combo_cfgs[key]
        train_dec = _combine(train_df, {n: candidates[n]["train_dec"] for n in names}, cfg)
        val_dec = _combine(val_df, {n: candidates[n]["val_dec"] for n in names}, cfg)
        eval_dec = _combine(eval_df, {n: candidates[n]["eval_dec"] for n in names}, cfg)
        promoted_decisions[key] = (train_dec, val_dec, eval_dec)
        runner_payload = _select_runner(
            train_df,
            val_df,
            eval_df,
            parent_for_features=parent_for_features,
            train_dec=train_dec,
            val_dec=val_dec,
            eval_dec=eval_dec,
            fee=fee,
            slip=slip,
            out_dir=args.out_dir / "candidate_runners",
            name=key,
        )
        promoted_rows.append(
            {
                "candidate_key": key,
                "promotion_source": (
                    "validation_and_oos_oracle"
                    if key in validation_promote_keys and key in oracle_promote_keys
                    else "validation"
                    if key in validation_promote_keys
                    else "oos_oracle_research_only"
                ),
                "cfg": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
                "selection_score": runner_payload["selection_score"],
                "val_cost1_pnl": runner_payload["validation_metrics"]["cost1"]["pnl"],
                "val_cost1_mdd": runner_payload["validation_metrics"]["cost1"]["mdd"],
                "val_cost1_trades": runner_payload["validation_metrics"]["cost1"]["trades"],
                "val_cost3_pnl": runner_payload["validation_metrics"]["cost3"]["pnl"],
                "val_cost3_mdd": runner_payload["validation_metrics"]["cost3"]["mdd"],
                "val_cost3_trades": runner_payload["validation_metrics"]["cost3"]["trades"],
                "oos_score": _score(runner_payload["metrics"]),
                "oos_cost1_pnl": runner_payload["metrics"]["cost1"]["pnl"],
                "oos_cost1_mdd": runner_payload["metrics"]["cost1"]["mdd"],
                "oos_cost1_trades": runner_payload["metrics"]["cost1"]["trades"],
                "oos_cost3_pnl": runner_payload["metrics"]["cost3"]["pnl"],
                "oos_cost3_mdd": runner_payload["metrics"]["cost3"]["mdd"],
                "oos_cost3_trades": runner_payload["metrics"]["cost3"]["trades"],
                "runner_config": runner_payload["runner_config"]["name"],
                "runner_snapshot_count": runner_payload["runner_snapshot_count"],
            }
        )
        print(
            f"[promote] {key} val_c3={promoted_rows[-1]['val_cost3_pnl']:.2f} "
            f"oos_c3={promoted_rows[-1]['oos_cost3_pnl']:.2f}",
            flush=True,
        )

    ranking = pd.DataFrame(promoted_rows)
    strict_validation = ranking[ranking["promotion_source"].isin(("validation", "validation_and_oos_oracle"))].copy()
    strict_validation.sort_values("selection_score", ascending=False).to_csv(args.out_dir / "ranking_validation_selected.csv", index=False)
    ranking.sort_values("selection_score", ascending=False).to_csv(args.out_dir / "ranking_promoted_mixed_for_research.csv", index=False)
    ranking.sort_values("oos_cost3_pnl", ascending=False).to_csv(args.out_dir / "ranking_oos_oracle_research_only.csv", index=False)
    selected = strict_validation.sort_values("selection_score", ascending=False).head(int(args.promote_top))
    best_key = str(selected.iloc[0]["candidate_key"])
    train_dec, val_dec, eval_dec = promoted_decisions[best_key]
    train_dec.assign(timestamp=train_df["timestamp"].to_numpy()).to_csv(args.out_dir / "selected_train_decisions.csv", index=False)
    val_dec.assign(timestamp=val_df["timestamp"].to_numpy()).to_csv(args.out_dir / "selected_validation_decisions.csv", index=False)
    eval_dec.assign(timestamp=eval_df["timestamp"].to_numpy()).to_csv(args.out_dir / "selected_oos_decisions.csv", index=False)

    summary = {
        "model_id": "alpha_synergy_research_20260525",
        "design": "Leak-free Alpha parent decision streams are combined with simple current-state routers. Selection is 2025Q4 only; 2026 is fixed OOS. OOS oracle ranking is reported separately as research-only.",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "split": {
            "train_runner": [str(train_df["timestamp"].iloc[0]), str(train_df["timestamp"].iloc[-1])],
            "selection": [str(val_df["timestamp"].iloc[0]), str(val_df["timestamp"].iloc[-1])],
            "oos": [str(eval_df["timestamp"].iloc[0]), str(eval_df["timestamp"].iloc[-1])],
        },
        "audit": {
            "selection_uses_2026": False,
            "old_clean_regime_feature_candidates_skipped": True,
            "fee": fee,
            "slip": slip,
            "candidate_count": len(candidates),
            "screen_combo_count": int(len(screen)),
            "promoted_combo_count": int(len(ranking)),
            "strict_validation_promoted_combo_count": int(len(strict_validation)),
            "oos_oracle_promoted_combo_count": int(sum(ranking["promotion_source"].eq("oos_oracle_research_only"))),
        },
        "candidates": {
            name: {
                "family": data["spec"].family,
                "parent": str(data["spec"].parent),
                "summary": str(data["spec"].summary),
                "feature_count": data["feature_count"],
                "scale_runtime": data["scale_runtime"],
                "eval_decision_audit": _decision_audit(eval_df, data["eval_dec"]),
            }
            for name, data in candidates.items()
        },
        "best_by_validation": selected.to_dict(orient="records"),
        "best_oos_oracle_research_only": ranking.sort_values("oos_cost3_pnl", ascending=False).head(10).to_dict(orient="records"),
        "selected_decision_audit": {
            "train": _decision_audit(train_df, train_dec),
            "validation": _decision_audit(val_df, val_dec),
            "oos": _decision_audit(eval_df, eval_dec),
        },
        "selected_costs": _compact_costs(
            _metrics(
                eval_df,
                parent_for_features=parent_for_features,
                runner=joblib.load(args.out_dir / "candidate_runners" / f"{best_key}_runner.pkl")["cost_runner"],
                runner_cfg=CostRunnerConfig(**joblib.load(args.out_dir / "candidate_runners" / f"{best_key}_runner.pkl")["selected_config"]),
                dec=eval_dec,
                fee=fee,
                slip=slip,
            )
        ),
        "artifacts": {
            "validation_ranking": str(args.out_dir / "ranking_validation_selected.csv"),
            "promoted_mixed_ranking": str(args.out_dir / "ranking_promoted_mixed_for_research.csv"),
            "oos_oracle_ranking": str(args.out_dir / "ranking_oos_oracle_research_only.csv"),
            "selected_oos_decisions": str(args.out_dir / "selected_oos_decisions.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"best_by_validation": summary["best_by_validation"][0], "selected_costs": summary["selected_costs"]}, ensure_ascii=False, default=_json_default), flush=True)
    print(f"[out] {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
