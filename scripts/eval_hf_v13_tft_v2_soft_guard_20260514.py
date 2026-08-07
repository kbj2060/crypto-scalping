#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, ACTION_SHORT, FullyLearnedGovernorConfig, predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_hf_v13_deep_tabular_parent_mdd_20260514 as tft_v2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import eval_hf_v13_v31_frozen_parent_layer_ablation_v45 as v45  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "hf_v13_tft_v2_soft_guard_20260514"
ARTIFACT = ROOT / "data/ensemble/supervised/hf_v13_deep_tabular_parent_mdd_v2_20260514/tft_lite.pt"
OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_tft_v2_soft_guard_20260514"
REPORT_OUT = ROOT / "data/ensemble/reports/hf_v13_tft_v2_soft_guard_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/hf_v13_tft_v2_soft_guard_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/hf_v13_tft_v2_soft_guard_20260514_grid.csv"


@dataclass(frozen=True)
class GuardConfig:
    name: str
    mode: str
    base_conf: float = 0.30
    base_q: float = -0.010
    base_unc: float = 0.090
    min_scale: float = 0.0
    weak_scale: float = 0.45
    max_scale: float = 1.20
    dynamic_unc: bool = False
    cost_penalty: bool = False
    volatility_penalty: float = 0.0
    liquidity_penalty: float = 0.0
    hierarchical: bool = False
    macro_th: float = 0.0
    mdd_runtime: bool = False


def _safe_array(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=np.float64)
    arr = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default)).to_numpy(dtype=np.float64)
    return arr


def _risk_pressure(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    vol = (
        0.30 * np.clip(np.abs(_safe_array(df, "volatility_z", 0.0)) / 3.0, 0.0, 1.5)
        + 0.25 * np.clip(np.maximum(_safe_array(df, "realized_vol_ratio", 1.0) - 1.0, 0.0), 0.0, 1.5)
        + 0.20 * np.clip(np.abs(_safe_array(df, "bb_width_z", 0.0)) / 3.0, 0.0, 1.5)
        + 0.15 * np.clip(_safe_array(df, "clean_regime_2024_unsup_v4_transition_risk", 0.0), 0.0, 1.0)
        + 0.10 * np.clip(_safe_array(df, "clean_regime_2024_unsup_v4_whipsaw_prob", 0.0), 0.0, 1.0)
    )
    liq = (
        0.35 * np.clip(np.abs(_safe_array(df, "amihud_illiquidity_z", 0.0)) / 3.0, 0.0, 1.5)
        + 0.25 * np.clip(_safe_array(df, "liquidity_vacuum", 0.0), 0.0, 1.0)
        + 0.20 * np.clip(np.maximum(-_safe_array(df, "execution_quality", 0.0), 0.0), 0.0, 1.0)
        + 0.20 * np.clip(_safe_array(df, "clean_regime_2024_unsup_v4_factor_liquidity", 0.0), 0.0, 1.0)
    )
    return np.clip(vol, 0.0, 1.5), np.clip(liq, 0.0, 1.5)


def _macro_alignment(df: pd.DataFrame, side: np.ndarray) -> np.ndarray:
    direction = np.asarray(side, dtype=np.float64)
    score = (
        0.28 * _safe_array(df, "mtf_trend_1h", 0.0)
        + 0.22 * _safe_array(df, "mtf_trend_4h", 0.0)
        + 0.18 * _safe_array(df, "mom_1d", 0.0)
        + 0.14 * _safe_array(df, "m7_expected_ret", 0.0)
        + 0.10 * _safe_array(df, "ai_dir_edge", 0.0)
        + 0.08 * _safe_array(df, "clean_regime_2024_unsup_v4_factor_trend", 0.0)
    )
    return direction * score


def _outputs_to_arrays(outputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proba = np.asarray(outputs["action"], dtype=np.float64)
    action = np.argmax(proba, axis=1).astype(np.int64)
    side = np.where(action == ACTION_LONG, 1, np.where(action == ACTION_SHORT, -1, 0)).astype(np.int64)
    conf = np.max(proba, axis=1)
    quality = np.asarray(outputs["quality"], dtype=np.float64)
    unc = np.asarray(outputs.get("action_uncertainty", np.zeros_like(conf)), dtype=np.float64)
    return action, side, conf, quality, unc


def _guard_decisions(df: pd.DataFrame, teacher: pd.DataFrame, outputs: dict[str, np.ndarray], cfg: GuardConfig) -> pd.DataFrame:
    out = teacher.copy()
    t_action, t_side, conf, quality, unc = _outputs_to_arrays(outputs)
    teacher_action = out["action"].astype(int).to_numpy()
    teacher_side = out["side"].astype(int).to_numpy()
    active = (teacher_action != ACTION_CASH) & (teacher_side != 0)
    agree = active & (t_action == teacher_action) & (t_side == teacher_side)
    vol, liq = _risk_pressure(df)

    conf_req = np.full(len(df), float(cfg.base_conf), dtype=np.float64)
    q_req = np.full(len(df), float(cfg.base_q), dtype=np.float64)
    unc_cap = np.full(len(df), float(cfg.base_unc), dtype=np.float64)
    if cfg.dynamic_unc:
        unc_cap = np.clip(cfg.base_unc * (1.25 - 0.55 * vol), 0.045, 0.125)
        conf_req = conf_req + cfg.volatility_penalty * vol
        q_req = q_req + 0.010 * vol
    if cfg.cost_penalty:
        conf_req = conf_req + cfg.liquidity_penalty * liq
        q_req = q_req + 0.015 * liq
        unc_cap = np.clip(unc_cap * (1.0 - 0.35 * liq), 0.035, 0.125)

    pass_gate = agree & (conf >= conf_req) & (quality >= q_req) & (unc <= unc_cap)
    strong = pass_gate & (conf >= conf_req + 0.16) & (quality >= q_req + 0.015) & (unc <= np.minimum(0.045, unc_cap * 0.70))
    marginal = agree & ~pass_gate & (conf >= conf_req - 0.06) & (quality >= q_req - 0.010) & (unc <= unc_cap * 1.15)
    macro = _macro_alignment(df, teacher_side)
    if cfg.hierarchical:
        macro_pass = macro >= cfg.macro_th
        macro_soft = macro >= cfg.macro_th - 0.006
        marginal = (pass_gate & ~macro_pass & macro_soft) | marginal
        pass_gate = pass_gate & macro_pass
        strong = strong & macro_pass

    if cfg.mode == "hard_veto":
        scale = np.where(pass_gate, 1.0, 0.0)
    else:
        denom = np.maximum(unc_cap - 0.030, 1e-6)
        unc_score = np.clip((unc_cap - unc) / denom, 0.0, 1.0)
        conf_score = np.clip((conf - conf_req) / 0.25, 0.0, 1.0)
        base_scale = cfg.weak_scale + (1.0 - cfg.weak_scale) * (0.55 * unc_score + 0.45 * conf_score)
        scale = np.where(pass_gate, base_scale, np.where(marginal, cfg.weak_scale, 0.0))
        scale = np.where(strong, cfg.max_scale, scale)
        scale = np.clip(scale, cfg.min_scale, cfg.max_scale)

    survival_unc_cap = np.clip(unc_cap * 0.55, 0.025, 0.070)
    survival_pass = agree & (conf >= conf_req + 0.12) & (quality >= q_req + 0.012) & (unc <= survival_unc_cap)
    if cfg.hierarchical:
        survival_pass = survival_pass & (macro >= cfg.macro_th + 0.003)
    survival_scale = np.where(survival_pass, np.minimum(scale, 0.55), 0.0)

    out.loc[active, "notional_exposure"] = out.loc[active, "notional_exposure"].to_numpy(dtype=np.float64) * scale[active]
    out.loc[active, "position_fraction"] = out.loc[active, "position_fraction"].to_numpy(dtype=np.float64) * scale[active]
    zero = active & (scale <= 1e-8)
    out.loc[zero, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[zero, "leverage"] = 1.0
    out.loc[:, "tft_v2_action"] = t_action
    out.loc[:, "tft_v2_confidence"] = conf
    out.loc[:, "tft_v2_quality"] = quality
    out.loc[:, "tft_v2_uncertainty"] = unc
    out.loc[:, "tft_guard_scale"] = scale
    out.loc[:, "tft_guard_mdd_survival_scale"] = survival_scale
    out.loc[:, "tft_guard_unc_cap"] = unc_cap
    out.loc[:, "tft_guard_conf_req"] = conf_req
    out.loc[:, "tft_guard_macro_alignment"] = macro
    return out


def _grid() -> list[GuardConfig]:
    rows = [
        GuardConfig("v2_hard_veto_reference", "hard_veto", 0.30, -0.010, 0.090),
    ]
    for base_conf in (0.28, 0.30, 0.34, 0.38):
        for weak in (0.35, 0.50, 0.65):
            rows.append(GuardConfig(f"soft_static_c{base_conf:.2f}_w{weak:.2f}", "soft", base_conf, -0.010, 0.090, weak_scale=weak, max_scale=1.20))
            rows.append(GuardConfig(f"soft_dyn_c{base_conf:.2f}_w{weak:.2f}", "soft", base_conf, -0.010, 0.090, weak_scale=weak, max_scale=1.20, dynamic_unc=True, volatility_penalty=0.05))
            rows.append(GuardConfig(f"soft_dyn_cost_c{base_conf:.2f}_w{weak:.2f}", "soft", base_conf, -0.005, 0.090, weak_scale=weak, max_scale=1.15, dynamic_unc=True, cost_penalty=True, volatility_penalty=0.06, liquidity_penalty=0.08))
    for macro_th in (-0.004, 0.000, 0.004):
        rows.append(GuardConfig(f"hier_soft_macro{macro_th:+.3f}", "soft", 0.30, -0.010, 0.090, weak_scale=0.50, max_scale=1.15, dynamic_unc=True, cost_penalty=True, volatility_penalty=0.04, liquidity_penalty=0.05, hierarchical=True, macro_th=macro_th))
        rows.append(GuardConfig(f"hier_mdd_macro{macro_th:+.3f}", "soft", 0.30, -0.010, 0.090, weak_scale=0.50, max_scale=1.10, dynamic_unc=True, cost_penalty=True, volatility_penalty=0.04, liquidity_penalty=0.05, hierarchical=True, macro_th=macro_th, mdd_runtime=True))
    return rows


def _metrics(df: pd.DataFrame, q: np.ndarray, decisions: pd.DataFrame, parent: dict[str, Any], jackpot_model: dict[str, Any], add_cfg: CostRunnerConfig, variant: v45.LayerVariant, base_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        f"cost{m}": v45.backtest_variant(df, parent, jackpot_model, add_cfg, q, variant, fee=float(base_cfg["fee"]), slip=float(base_cfg["slip"]), cost_mult=float(m), decisions=decisions)
        for m in (1, 2, 3)
    }


def _score(metrics: dict[str, Any]) -> float:
    c1, c2, c3 = metrics["cost1"], metrics["cost2"], metrics["cost3"]
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(c1["pnl"] + 0.40 * c2["pnl"] + 0.18 * c3["pnl"] - 4.0 * abs(c1["mdd"]))


def main() -> int:
    p = argparse.ArgumentParser(description="Apply report ideas to TFT-lite v2 guard: soft-veto, dynamic uncertainty, cost-aware scaling.")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--batch-size", type=int, default=768)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    np.random.seed(20260514)
    torch.manual_seed(20260514)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[{MODEL_ID}] loading alpha1 and TFT v2 artifacts", flush=True)
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    cfg = FullyLearnedGovernorConfig(**dict(parent["config"]))
    feature_cols = list(parent.get("feature_cols") or [])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    audit_base = _audit_contract(train_all, eval_df, feature_cols)

    payload = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
    model = tft_v2.TFTLiteParent(len(feature_cols), cfg)
    model.load_state_dict(payload["state_dict"])
    norm = payload["normalizer"]

    def outputs_for(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        pre = prepare_features(df, side_hint=0, close=_close(df), feature_cols=feature_cols)
        x = tft_v2._normalise_apply(pre, norm)
        seq = tft_v2._sequence_array(x, np.arange(len(df), dtype=np.int64))
        return tft_v2._predict_outputs(model, x, seq, device, int(args.batch_size), mc_passes=8)

    print(f"[{MODEL_ID}] predicting TFT v2 val/eval outputs", flush=True)
    val_outputs = outputs_for(val_df, 20260515)
    eval_outputs = outputs_for(eval_df, 20260516)
    val_teacher = predict_policy_frame(parent, val_df, close=_close(val_df))
    eval_teacher = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    val_q = v31._predict_all(v27_model, val_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    variant = v45.LayerVariant("alpha1_tft_v2_soft_guard", "tft_v2_soft_guard", tft_v2._overlay_alpha1())
    base_cfg = dict(parent["config"])

    grid = _grid()
    if args.quick:
        grid = [g for g in grid if g.name == "v2_hard_veto_reference" or ("soft_dyn_cost" in g.name and g.base_conf in (0.30, 0.34) and g.weak_scale in (0.50, 0.65)) or ("soft_static" in g.name and g.base_conf in (0.30,) and g.weak_scale in (0.50,)) or g.name.startswith("hier_")]
    rows: list[dict[str, Any]] = []
    selected: GuardConfig | None = None
    best_score = -1e18
    for gc in grid:
        dec = _guard_decisions(val_df, val_teacher, val_outputs, gc)
        run_variant = replace(variant, mdd_entry_guard=True, mdd_parent_scale_col="tft_guard_mdd_survival_scale") if gc.mdd_runtime else variant
        vm = _metrics(val_df, val_q, dec, parent, jackpot_model, add_cfg, run_variant, base_cfg)
        score = _score(vm)
        row = {**asdict(gc), "score": score, "val_pnl": vm["cost1"]["pnl"], "val_mdd": vm["cost1"]["mdd"], "val_trades": vm["cost1"]["trades"], "val_cost2_pnl": vm["cost2"]["pnl"], "val_cost3_pnl": vm["cost3"]["pnl"]}
        rows.append(row)
        if score > best_score:
            best_score = score
            selected = gc
            print(f"[{MODEL_ID}] new val best {gc.name} score={score:.2f} pnl={row['val_pnl']:.2f} mdd={row['val_mdd']:.2f} c2={row['val_cost2_pnl']:.2f} c3={row['val_cost3_pnl']:.2f}", flush=True)
    assert selected is not None

    experiments: list[dict[str, Any]] = []
    baseline_metrics = _metrics(eval_df, eval_q, eval_teacher, parent, jackpot_model, add_cfg, variant, base_cfg)
    experiments.append({"name": "alpha1_hgb_parent_baseline", "metrics": baseline_metrics, "score": _score(baseline_metrics)})
    for gc in [g for g in grid if g.name == "v2_hard_veto_reference"] + [selected]:
        dec = _guard_decisions(eval_df, eval_teacher, eval_outputs, gc)
        run_variant = replace(variant, mdd_entry_guard=True, mdd_parent_scale_col="tft_guard_mdd_survival_scale") if gc.mdd_runtime else variant
        metrics = _metrics(eval_df, eval_q, dec, parent, jackpot_model, add_cfg, run_variant, base_cfg)
        experiments.append({"name": gc.name, "config": asdict(gc), "metrics": metrics, "score": _score(metrics)})
        print(f"[{MODEL_ID}] OOS {gc.name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)
    best = max(experiments, key=lambda e: e["score"])
    blocking = list(audit_base.get("blocking", []))
    warnings = list(audit_base.get("warnings", []))
    if best["name"] == "alpha1_hgb_parent_baseline":
        warnings.append("soft_guard_did_not_beat_alpha1_score")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best["name"] != "alpha1_hgb_parent_baseline" and best["metrics"]["cost1"]["mdd"] > baseline_metrics["cost1"]["mdd"] and best["metrics"]["cost1"]["pnl"] > 0 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS after selection",
        "design": "TFT-lite v2 guard logic ablation: hard veto, confidence-weighted soft notional scaling, volatility-adjusted uncertainty cap, cost-aware liquidity penalty, hierarchical macro guard, and runtime MDD survival scaling.",
        "base_audit": audit_base,
    }
    report = {
        "model_id": MODEL_ID,
        "selected": best,
        "selected_guard_from_validation": asdict(selected),
        "experiments": experiments,
        "grid_path": str(GRID_OUT),
        "audit_path": str(AUDIT_OUT),
        "artifact": str(ARTIFACT),
        "audit": audit,
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] selected={best['name']} report={REPORT_OUT}", flush=True)
    print(f"[{MODEL_ID}] audit={AUDIT_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
