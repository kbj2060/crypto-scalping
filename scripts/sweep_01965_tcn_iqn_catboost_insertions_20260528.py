#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 import (  # noqa: E402
    _active,
    _apply_decision_mods,
    _decision_sources,
    _load_stack,
    _score,
    _sl_ratio,
)
from scripts.precision_retest_01965_alpha7_combo_20260527 import CANDIDATE, _cfg_from_results  # noqa: E402
from scripts.test_01965_tcn_iqn_catboost_fallback_20260528 import (  # noqa: E402
    DECONTAM_DIR,
    _apply_scaler,
    _build_replacement_decisions,
    _cat_probs,
    _combine_primary_with_replacement,
    _eligible_primary_cash,
    _eval_final_dec,
    _feature_matrix,
    _fit_scaler,
    _fit_side_catboost,
    _iqn_scores,
    _load_train_val_eval,
    _patch_decontam_sources,
    _simulate_action_targets,
    _success_label_for_side,
    _train_tcn_iqn,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "01965_tcn_iqn_catboost_insertion_sweep_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
ALPHA7_PRIMARY = DECONTAM_DIR / "primary_parent.pkl"


def _side_action(side: int) -> int:
    return 1 if int(side) > 0 else 2


def _side_prob(side: int, i: int, long_p: np.ndarray, short_p: np.ndarray) -> float:
    return float(long_p[i] if int(side) > 0 else short_p[i])


def _side_score(side: int, i: int, scores: np.ndarray) -> tuple[float, float, int]:
    action = _side_action(side)
    score = float(scores[i, action])
    cash = float(scores[i, 0])
    return score, score - cash, action


def _empty_like(template: pd.DataFrame) -> pd.DataFrame:
    out = template.copy().reset_index(drop=True)
    out.loc[:, ["action", "side"]] = 0
    for col in ["notional_exposure", "position_fraction", "quality_score", "confidence"]:
        if col in out.columns:
            out[col] = 0.0
    return out


def _copy_row(out: pd.DataFrame, template: pd.DataFrame, i: int) -> None:
    for col in template.columns:
        if col in out.columns:
            out.at[i, col] = template.at[i, col]


def _fallback_gate_decisions(
    *,
    mode: str,
    primary: pd.DataFrame,
    baseline: pd.DataFrame,
    scores: np.ndarray,
    long_p: np.ndarray,
    short_p: np.ndarray,
    allowed: np.ndarray,
    seq_len: int,
    cvar_min: float,
    edge_min: float,
    cat_min: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = _empty_like(primary)
    primary_cash = (~_active(primary)).to_numpy(dtype=bool)
    baseline_active = _active(baseline).to_numpy(dtype=bool)
    allowed = np.asarray(allowed, dtype=bool)
    counts = {"kept": 0, "blocked": 0, "outside_scope": 0, "seq_warmup": 0, "long": 0, "short": 0, "scaled_half": 0, "scaled_up": 0}
    for i in range(len(out)):
        if not (primary_cash[i] and baseline_active[i] and allowed[i]):
            counts["outside_scope"] += 1
            continue
        if i < int(seq_len) - 1:
            counts["seq_warmup"] += 1
            continue
        side = int(baseline.at[i, "side"])
        score, edge, action = _side_score(side, i, scores)
        best_action = int(np.argmax(scores[i]))
        p_success = _side_prob(side, i, long_p, short_p)
        same_side = best_action == action
        pass_risk = score >= float(cvar_min) and edge >= float(edge_min) and p_success >= float(cat_min)

        keep = False
        scale = 1.0
        if mode == "fallback_veto":
            keep = pass_risk
        elif mode == "fallback_side_confirm":
            keep = pass_risk and same_side
        elif mode == "fallback_notional_scaler":
            if pass_risk and same_side:
                keep = True
                scale = 1.15
                counts["scaled_up"] += 1
            elif p_success >= float(cat_min) and score >= float(cvar_min) - 0.015:
                keep = True
                scale = 0.55
                counts["scaled_half"] += 1
            else:
                keep = False
        else:
            raise RuntimeError(f"unknown fallback gate mode: {mode}")

        if not keep:
            counts["blocked"] += 1
            continue
        _copy_row(out, baseline, i)
        if mode == "fallback_notional_scaler":
            n = float(pd.to_numeric(pd.Series([out.at[i, "notional_exposure"]]), errors="coerce").fillna(0.0).iloc[0])
            out.at[i, "notional_exposure"] = float(np.clip(n * scale, 0.0, 2.0))
            if "leverage" in out.columns and "position_fraction" in out.columns:
                lev = float(pd.to_numeric(pd.Series([out.at[i, "leverage"]]), errors="coerce").fillna(1.0).iloc[0])
                out.at[i, "position_fraction"] = float(np.clip(float(out.at[i, "notional_exposure"]) / max(lev, 1e-12), 0.0, 1.0))
        out.at[i, "quality_score"] = float(edge)
        out.at[i, "confidence"] = float(p_success)
        counts["kept"] += 1
        counts["long" if side > 0 else "short"] += 1
    return out, counts


def _primary_weak_veto_decisions(
    *,
    baseline: pd.DataFrame,
    primary: pd.DataFrame,
    scores: np.ndarray,
    long_p: np.ndarray,
    short_p: np.ndarray,
    seq_len: int,
    cvar_min: float,
    edge_min: float,
    cat_min: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = baseline.copy().reset_index(drop=True)
    primary_active = _active(primary).to_numpy(dtype=bool)
    q = pd.to_numeric(primary["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    conf = pd.to_numeric(primary["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    weak = primary_active & ((q <= 0.0020) | (conf <= 0.56))
    counts = {"checked": 0, "blocked": 0, "kept": 0, "seq_warmup": 0}
    for i in np.flatnonzero(weak):
        if i < int(seq_len) - 1:
            counts["seq_warmup"] += 1
            continue
        side = int(primary.at[i, "side"])
        score, edge, _action = _side_score(side, int(i), scores)
        p_success = _side_prob(side, int(i), long_p, short_p)
        counts["checked"] += 1
        if not (score >= float(cvar_min) and edge >= float(edge_min) and p_success >= float(cat_min)):
            out.loc[int(i), ["action", "side"]] = 0
            counts["blocked"] += 1
        else:
            counts["kept"] += 1
    return out, counts


def _deep_veto_q(
    *,
    q: np.ndarray,
    scores: np.ndarray,
    long_p: np.ndarray,
    short_p: np.ndarray,
    seq_len: int,
    cvar_min: float,
    edge_min: float,
    cat_min: float,
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.array(q, copy=True)
    counts = {"long_blocked": 0, "short_blocked": 0, "seq_warmup": 0}
    for i in range(len(out)):
        if i < int(seq_len) - 1:
            counts["seq_warmup"] += 1
            out[i, :] = -1e9
            continue
        long_score, long_edge, _ = _side_score(1, i, scores)
        short_score, short_edge, _ = _side_score(-1, i, scores)
        if not (long_score >= float(cvar_min) and long_edge >= float(edge_min) and float(long_p[i]) >= float(cat_min)):
            out[i, 0] = -1e9
            counts["long_blocked"] += 1
        if not (short_score >= float(cvar_min) and short_edge >= float(edge_min) and float(short_p[i]) >= float(cat_min)):
            out[i, 1] = -1e9
            counts["short_blocked"] += 1
    return out, counts


def _grid(scores: np.ndarray, active_mask: np.ndarray, profile: str) -> tuple[list[float], list[float], list[float]]:
    active_scores = scores[np.asarray(active_mask, dtype=bool)]
    if len(active_scores) == 0:
        active_scores = scores
    edge_basis = active_scores[:, 1:3].max(axis=1) - active_scores[:, 0]
    best_basis = active_scores[:, 1:3].max(axis=1)
    if profile == "full":
        qs = (0.55, 0.70, 0.80, 0.90, 0.95, 0.98)
        edge = sorted({-0.02, 0.0, 0.002, 0.005, 0.010, *[float(np.quantile(edge_basis, q)) for q in qs]})
        cvar = sorted({-0.05, -0.02, -0.005, 0.0, 0.005, 0.010, *[float(np.quantile(best_basis, q)) for q in qs]})
        cat = [0.45, 0.50, 0.55, 0.58, 0.62, 0.66, 0.70]
    else:
        qs = (0.70, 0.85, 0.95)
        edge = sorted({-0.01, 0.0, 0.005, *[float(np.quantile(edge_basis, q)) for q in qs]})
        cvar = sorted({-0.03, -0.005, 0.0, *[float(np.quantile(best_basis, q)) for q in qs]})
        cat = [0.48, 0.52, 0.58, 0.64]
    return cvar, edge, cat


def _eval_costs(df: pd.DataFrame, q: np.ndarray, dec: pd.DataFrame, stack: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    return {f"cost{c}": _eval_final_dec(df, q, dec, stack, cfg, cost_mult=c) for c in (1, 2, 3)}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sweep TCN-IQN-CatBoost insertion points on 01965.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=20260528)
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--tau-samples", type=int, default=32)
    ap.add_argument("--risk-tau", type=float, default=0.25)
    ap.add_argument("--notional", type=float, default=2.0)
    ap.add_argument("--take-profit", type=float, default=0.060)
    ap.add_argument("--stop-loss", type=float, default=0.045)
    ap.add_argument("--max-hold", type=int, default=96)
    ap.add_argument("--margin-limit", type=float, default=0.12)
    ap.add_argument("--catboost-task-type", choices=["CPU", "GPU"], default="CPU")
    ap.add_argument("--grid-profile", choices=["standard", "full"], default="standard")
    ap.add_argument("--min-val-affected-rows", type=int, default=8)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_results()
    if cfg.get("source") != "alpha7_combo_primary_fallback":
        raise RuntimeError(f"01965 source contract changed: {cfg.get('source')}")

    _patch_decontam_sources()
    stack = _load_stack()
    train_df, val_df, eval_df = _load_train_val_eval()
    print(json.dumps({"stage": "frames_loaded", "train": len(train_df), "val": len(val_df), "oos": len(eval_df)}), flush=True)
    sources_train = _decision_sources(train_df, train_df, stack["parent"])
    sources = _decision_sources(val_df, eval_df, stack["parent"])
    print(json.dumps({"stage": "decision_sources_ready"}), flush=True)

    primary_train = _apply_decision_mods(sources_train["alpha7_primary"][0], cfg)
    primary_val = _apply_decision_mods(sources["alpha7_primary"][0], cfg)
    primary_eval = _apply_decision_mods(sources["alpha7_primary"][1], cfg)
    baseline_val = _apply_decision_mods(sources["alpha7_combo_primary_fallback"][0], cfg)
    baseline_eval = _apply_decision_mods(sources["alpha7_combo_primary_fallback"][1], cfg)
    allowed_val = ((~_active(primary_val)) & _active(baseline_val)).to_numpy(dtype=bool)
    allowed_eval = ((~_active(primary_eval)) & _active(baseline_eval)).to_numpy(dtype=bool)

    feature_cols = list(joblib.load(ALPHA7_PRIMARY)["feature_cols"])
    x_train_df = _feature_matrix(train_df, feature_cols, name="train")
    x_val_df = _feature_matrix(val_df, feature_cols, name="val")
    x_eval_df = _feature_matrix(eval_df, feature_cols, name="eval")
    train_indices = np.flatnonzero(_eligible_primary_cash(primary_train, int(args.seq_len), int(args.max_hold))).astype(np.int64)
    if len(train_indices) < 1000:
        raise RuntimeError(f"too few primary-CASH train rows: {len(train_indices)}")
    scaler = _fit_scaler(x_train_df.iloc[train_indices].reset_index(drop=True))
    x_train = _apply_scaler(x_train_df, scaler)
    x_val = _apply_scaler(x_val_df, scaler)
    x_eval = _apply_scaler(x_eval_df, scaler)
    y_train = _simulate_action_targets(
        train_df,
        train_indices,
        notional=float(args.notional),
        tp=float(args.take_profit),
        sl=float(args.stop_loss),
        max_hold=int(args.max_hold),
        fee=float(stack["fee"]),
        slip=float(stack["slip"]),
        cost_mult=3.0,
        margin_limit=float(args.margin_limit),
        dd_lambda=4.0,
        liquidation_penalty=0.75,
        entry_hurdle=0.0,
        theta_penalty=0.0005,
    )
    model, train_diag = _train_tcn_iqn(
        x_train,
        train_indices,
        y_train,
        seq_len=int(args.seq_len),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        tau_samples=int(args.tau_samples),
        seed=int(args.seed),
    )
    print(json.dumps({"stage": "tcn_iqn_trained", "last_loss": train_diag["losses"][-1], "redo": train_diag.get("redo_reset_neurons")}), flush=True)

    long_labels = _success_label_for_side(train_df, train_indices, side=1, notional=float(args.notional), tp=float(args.take_profit), sl=float(args.stop_loss), max_hold=int(args.max_hold), margin_limit=float(args.margin_limit))
    short_labels = _success_label_for_side(train_df, train_indices, side=-1, notional=float(args.notional), tp=float(args.take_profit), sl=float(args.stop_loss), max_hold=int(args.max_hold), margin_limit=float(args.margin_limit))
    cat_long = _fit_side_catboost(x_train_df, train_indices, long_labels, seed=int(args.seed) + 11, task_type=str(args.catboost_task_type))
    cat_short = _fit_side_catboost(x_train_df, train_indices, short_labels, seed=int(args.seed) + 29, task_type=str(args.catboost_task_type))
    print(json.dumps({"stage": "catboost_trained", "long_success_rate": float(np.mean(long_labels)), "short_success_rate": float(np.mean(short_labels))}), flush=True)

    val_scores = _iqn_scores(model, x_val, seq_len=int(args.seq_len), risk_tau=float(args.risk_tau), num_tau=32, batch_size=2048)
    eval_scores = _iqn_scores(model, x_eval, seq_len=int(args.seq_len), risk_tau=float(args.risk_tau), num_tau=32, batch_size=2048)
    val_long_p, val_short_p = _cat_probs(cat_long, x_val_df), _cat_probs(cat_short, x_val_df)
    eval_long_p, eval_short_p = _cat_probs(cat_long, x_eval_df), _cat_probs(cat_short, x_eval_df)
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    baseline_costs_val = _eval_costs(val_df, val_q, baseline_val, stack, cfg)
    baseline_costs_eval = _eval_costs(eval_df, eval_q, baseline_eval, stack, cfg)
    print(json.dumps({"stage": "baseline_ready", "oos_cost3_pnl": baseline_costs_eval["cost3"]["pnl"]}), flush=True)

    cvar_grid, edge_grid, cat_grid = _grid(val_scores, allowed_val, str(args.grid_profile))
    modes = [
        "strict_replacement",
        "fallback_veto",
        "fallback_side_confirm",
        "fallback_notional_scaler",
        "deep_scout_veto",
        "primary_weak_veto",
    ]
    print(json.dumps({"stage": "grid_ready", "profile": args.grid_profile, "modes": modes, "per_mode_variants": len(cvar_grid) * len(edge_grid) * len(cat_grid)}), flush=True)

    rows: list[dict[str, Any]] = []
    best_by_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        best: dict[str, Any] | None = None
        for cvar_min in cvar_grid:
            for edge_min in edge_grid:
                for cat_min in cat_grid:
                    counts: dict[str, int]
                    if mode == "strict_replacement":
                        fb, counts = _build_replacement_decisions(
                            primary_val,
                            baseline_val,
                            val_scores,
                            val_long_p,
                            val_short_p,
                            allowed_mask=allowed_val,
                            seq_len=int(args.seq_len),
                            cvar_min=float(cvar_min),
                            edge_min=float(edge_min),
                            cat_min=float(cat_min),
                            notional=float(args.notional),
                            leverage=float(args.notional),
                            tp=float(args.take_profit),
                            sl=float(args.stop_loss),
                            max_hold=int(args.max_hold),
                            cooldown=2,
                            risk_source="existing_fallback",
                        )
                        dec = _combine_primary_with_replacement(primary_val, fb)
                        q_use = val_q
                        affected = int(counts["long"] + counts["short"])
                    elif mode in {"fallback_veto", "fallback_side_confirm", "fallback_notional_scaler"}:
                        fb, counts = _fallback_gate_decisions(
                            mode=mode,
                            primary=primary_val,
                            baseline=baseline_val,
                            scores=val_scores,
                            long_p=val_long_p,
                            short_p=val_short_p,
                            allowed=allowed_val,
                            seq_len=int(args.seq_len),
                            cvar_min=float(cvar_min),
                            edge_min=float(edge_min),
                            cat_min=float(cat_min),
                        )
                        dec = _combine_primary_with_replacement(primary_val, fb)
                        q_use = val_q
                        affected = int(counts.get("kept", 0) + counts.get("blocked", 0))
                    elif mode == "deep_scout_veto":
                        q_use, counts = _deep_veto_q(q=val_q, scores=val_scores, long_p=val_long_p, short_p=val_short_p, seq_len=int(args.seq_len), cvar_min=float(cvar_min), edge_min=float(edge_min), cat_min=float(cat_min))
                        dec = baseline_val
                        affected = int(counts["long_blocked"] + counts["short_blocked"])
                    elif mode == "primary_weak_veto":
                        dec, counts = _primary_weak_veto_decisions(baseline=baseline_val, primary=primary_val, scores=val_scores, long_p=val_long_p, short_p=val_short_p, seq_len=int(args.seq_len), cvar_min=float(cvar_min), edge_min=float(edge_min), cat_min=float(cat_min))
                        q_use = val_q
                        affected = int(counts["checked"])
                    else:
                        raise RuntimeError(mode)
                    c3 = _eval_final_dec(val_df, q_use, dec, stack, cfg, cost_mult=3)
                    selection = _score(c3) if affected >= int(args.min_val_affected_rows) else -1e9 + float(c3["pnl"])
                    row = {
                        "mode": mode,
                        "cvar_min": float(cvar_min),
                        "edge_min": float(edge_min),
                        "cat_min": float(cat_min),
                        "selection_score": float(selection),
                        "val_cost3_pnl": float(c3["pnl"]),
                        "val_cost3_mdd": float(c3["mdd"]),
                        "val_cost3_wr": float(c3["wr"]),
                        "val_cost3_trades": int(c3["trades"]),
                        "val_sl_ratio": float(_sl_ratio(c3)),
                        "affected": int(affected),
                        "counts": json.dumps(counts, sort_keys=True),
                    }
                    rows.append(row)
                    if best is None or float(row["selection_score"]) > float(best["selection_score"]):
                        best = row
        if best is None:
            raise RuntimeError(f"no rows for mode: {mode}")
        best_by_mode[mode] = best
        print(json.dumps({"stage": "mode_selected", "mode": mode, "best": best}, ensure_ascii=False), flush=True)

    final_rows: list[dict[str, Any]] = []
    final_payload: dict[str, Any] = {}
    for mode, best in best_by_mode.items():
        cvar_min, edge_min, cat_min = float(best["cvar_min"]), float(best["edge_min"]), float(best["cat_min"])
        if mode == "strict_replacement":
            val_fb, val_counts = _build_replacement_decisions(primary_val, baseline_val, val_scores, val_long_p, val_short_p, allowed_mask=allowed_val, seq_len=int(args.seq_len), cvar_min=cvar_min, edge_min=edge_min, cat_min=cat_min, notional=float(args.notional), leverage=float(args.notional), tp=float(args.take_profit), sl=float(args.stop_loss), max_hold=int(args.max_hold), cooldown=2, risk_source="existing_fallback")
            eval_fb, eval_counts = _build_replacement_decisions(primary_eval, baseline_eval, eval_scores, eval_long_p, eval_short_p, allowed_mask=allowed_eval, seq_len=int(args.seq_len), cvar_min=cvar_min, edge_min=edge_min, cat_min=cat_min, notional=float(args.notional), leverage=float(args.notional), tp=float(args.take_profit), sl=float(args.stop_loss), max_hold=int(args.max_hold), cooldown=2, risk_source="existing_fallback")
            val_dec = _combine_primary_with_replacement(primary_val, val_fb)
            eval_dec = _combine_primary_with_replacement(primary_eval, eval_fb)
            val_q_use, eval_q_use = val_q, eval_q
        elif mode in {"fallback_veto", "fallback_side_confirm", "fallback_notional_scaler"}:
            val_fb, val_counts = _fallback_gate_decisions(mode=mode, primary=primary_val, baseline=baseline_val, scores=val_scores, long_p=val_long_p, short_p=val_short_p, allowed=allowed_val, seq_len=int(args.seq_len), cvar_min=cvar_min, edge_min=edge_min, cat_min=cat_min)
            eval_fb, eval_counts = _fallback_gate_decisions(mode=mode, primary=primary_eval, baseline=baseline_eval, scores=eval_scores, long_p=eval_long_p, short_p=eval_short_p, allowed=allowed_eval, seq_len=int(args.seq_len), cvar_min=cvar_min, edge_min=edge_min, cat_min=cat_min)
            val_dec = _combine_primary_with_replacement(primary_val, val_fb)
            eval_dec = _combine_primary_with_replacement(primary_eval, eval_fb)
            val_q_use, eval_q_use = val_q, eval_q
        elif mode == "deep_scout_veto":
            val_q_use, val_counts = _deep_veto_q(q=val_q, scores=val_scores, long_p=val_long_p, short_p=val_short_p, seq_len=int(args.seq_len), cvar_min=cvar_min, edge_min=edge_min, cat_min=cat_min)
            eval_q_use, eval_counts = _deep_veto_q(q=eval_q, scores=eval_scores, long_p=eval_long_p, short_p=eval_short_p, seq_len=int(args.seq_len), cvar_min=cvar_min, edge_min=edge_min, cat_min=cat_min)
            val_dec, eval_dec = baseline_val, baseline_eval
        else:
            val_dec, val_counts = _primary_weak_veto_decisions(baseline=baseline_val, primary=primary_val, scores=val_scores, long_p=val_long_p, short_p=val_short_p, seq_len=int(args.seq_len), cvar_min=cvar_min, edge_min=edge_min, cat_min=cat_min)
            eval_dec, eval_counts = _primary_weak_veto_decisions(baseline=baseline_eval, primary=primary_eval, scores=eval_scores, long_p=eval_long_p, short_p=eval_short_p, seq_len=int(args.seq_len), cvar_min=cvar_min, edge_min=edge_min, cat_min=cat_min)
            val_q_use, eval_q_use = val_q, eval_q
        val_costs = _eval_costs(val_df, val_q_use, val_dec, stack, cfg)
        eval_costs = _eval_costs(eval_df, eval_q_use, eval_dec, stack, cfg)
        out = {
            "mode": mode,
            "cvar_min": cvar_min,
            "edge_min": edge_min,
            "cat_min": cat_min,
            "val_cost3_pnl": float(val_costs["cost3"]["pnl"]),
            "val_cost3_mdd": float(val_costs["cost3"]["mdd"]),
            "val_cost3_wr": float(val_costs["cost3"]["wr"]),
            "val_cost3_trades": int(val_costs["cost3"]["trades"]),
            "oos_cost3_pnl": float(eval_costs["cost3"]["pnl"]),
            "oos_cost3_mdd": float(eval_costs["cost3"]["mdd"]),
            "oos_cost3_wr": float(eval_costs["cost3"]["wr"]),
            "oos_cost3_trades": int(eval_costs["cost3"]["trades"]),
            "delta_oos_cost3_pnl": float(eval_costs["cost3"]["pnl"]) - float(baseline_costs_eval["cost3"]["pnl"]),
            "val_counts": val_counts,
            "oos_counts": eval_counts,
        }
        final_rows.append(out)
        final_payload[mode] = {"selected": best, "final": out, "val_metrics": val_costs, "oos_metrics": eval_costs}
        print(json.dumps({"stage": "mode_oos_done", "mode": mode, "oos_cost3_pnl": out["oos_cost3_pnl"], "delta": out["delta_oos_cost3_pnl"]}, ensure_ascii=False), flush=True)

    model_path = args.out_dir / "tcn_iqn_fallback_shared.pt"
    torch.save({"model_id": MODEL_ID, "state_dict": model.state_dict(), "feature_cols": feature_cols, "scaler": scaler, "train_diag": train_diag}, model_path)
    cat_long_path = args.out_dir / "catboost_long_success.cbm"
    cat_short_path = args.out_dir / "catboost_short_success.cbm"
    cat_long.save_model(cat_long_path)
    cat_short.save_model(cat_short_path)
    pd.DataFrame(rows).to_csv(args.out_dir / "validation_grid.csv", index=False)
    pd.DataFrame(final_rows).sort_values("oos_cost3_pnl", ascending=False).to_csv(args.out_dir / "final_modes.csv", index=False)
    summary = {
        "model_id": MODEL_ID,
        "candidate": CANDIDATE,
        "design": "Shared full-trained TCN-IQN-CatBoost tested across multiple 01965 insertion points. Selection uses 2025Q4 validation only; 2026 OOS is evaluated once per mode.",
        "baseline_01965": {"val": baseline_costs_val, "oos": baseline_costs_eval},
        "training": {
            "epochs": int(args.epochs),
            "train_rows": int(len(train_df)),
            "primary_cash_train_rows": int(len(train_indices)),
            "target_reward_mean": y_train.mean(axis=0).tolist(),
            "long_success_rate": float(np.mean(long_labels)),
            "short_success_rate": float(np.mean(short_labels)),
            "train_diag": train_diag,
        },
        "modes": final_payload,
        "artifacts": {
            "tcn_iqn": str(model_path),
            "catboost_long": str(cat_long_path),
            "catboost_short": str(cat_short_path),
            "validation_grid": str(args.out_dir / "validation_grid.csv"),
            "final_modes": str(args.out_dir / "final_modes.csv"),
        },
        "audit": {
            "selection_uses_2026": False,
            "selection_window": "2025-10-01..2025-12-31",
            "oos_window": "2026 fixed OOS",
            "live_path_modified": False,
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.out_dir / "summary.json"), "final_modes": final_rows}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
