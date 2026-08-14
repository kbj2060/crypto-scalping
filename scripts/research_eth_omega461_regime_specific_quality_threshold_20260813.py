#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey2 priority #1 (docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md
"점검 결과" item 1). Tests regime-conditional `quality_threshold` -- flagged as unresolved issue 4 in
Odyssey(1) since 2026-08-11 and never attempted. The live gate is a single global threshold
(h48qual=0.50, zig075=0.75) applied regardless of which regime3 expert (bull/bear/chop) routed the
bar; the architecture design doc explicitly left this as an open axis.

No retraining: `dir_action`/`quality_for_action`/`router_expert` are threshold-independent raw
columns already saved in every `*_predictions_qXXX.csv` (see
train_omega1_regime3_routed_expert_direction_quality_20260602._prediction_output) -- this script
loads ONE such file per component/split, recomputes `final_action` itself for an arbitrary
per-regime threshold map, and feeds the result through the exact same downstream pipeline
(ATR TP/SL, risk-sidecar sizing, exit simulation) as every other 08-13 experiment via
research_eth_omega461_exit_sweep_20260721.prep_component/replay_exit_variant and
replay_omega4_6_1_greedy_router_20260706.greedy_replay. Does NOT touch trading_bot_modules/
omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env, or any live deployed threshold.

Design (avoids combinatorial explosion / multiple-comparisons blowup): for each component,
independently sweep ONE regime's threshold at a time (holding the other two regimes at the live
global baseline) over a coarse grid, on VAL only. Pick each regime's own VAL-best value, then
combine into a single joint regime-threshold map and evaluate it jointly (component + portfolio)
on VAL. If the joint map beats the flat-global baseline on VAL (pnl AND mdd, no_gate AND
with_gate), open OOS once (single touch) on that one joint map only.

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.
VAL = 2025-10-01..2025-12-31, OOS = 2026-01-01..2026-03-31 (base_sweep windows).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_sweep_20260721 as base_sweep  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as helpers  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as router  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_regime_specific_quality_threshold_20260813"
REGIMES = ["bull", "bear", "chop"]
THRESHOLD_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
GLOBAL_BASELINE = {"h48qual": 0.50, "zig075": 0.75}
BASELINE_EXIT_THRESHOLD = 0.95


def log(msg: str) -> None:
    print(f"[regime_thr] {msg}", flush=True)


def load_raw(name: str, cfg: dict, split: str, pred_csv: Path) -> pd.DataFrame:
    src = pd.read_csv(pred_csv)
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    for c in src.columns:
        if str(src[c].dtype).lower().startswith("str"):
            src[c] = src[c].astype(object)
    return src


def build_final_action(src: pd.DataFrame, prefix: str, regime_thresholds: dict[str, float]) -> np.ndarray:
    """Recompute final_action from the raw threshold-independent columns using a per-regime
    threshold map. Mirrors train_omega1_regime3_routed_expert_direction_quality_20260602._prediction_output's
    gating rule exactly (final_action = dir_action unless dir_action!=CASH and quality_for_action < threshold),
    just with `threshold` varying by `router_expert` instead of being a single scalar."""
    dir_action = pd.to_numeric(src[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    quality_for_action = pd.to_numeric(src[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    router_expert = src[f"{prefix}router_expert"].astype(str).to_numpy()
    threshold_per_bar = np.array([regime_thresholds[r] for r in router_expert], dtype=np.float64)
    final_action = dir_action.copy()
    final_action[(dir_action != 0) & (quality_for_action < threshold_per_bar)] = 0
    return final_action


def evaluate_component(name: str, cfg: dict, frame: pd.DataFrame, src: pd.DataFrame, prefix: str,
                        regime_thresholds: dict[str, float], *, oof: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (component_metrics, prepped_dict_for_router)."""
    src2 = src.copy()
    src2[f"{prefix}final_action"] = build_final_action(src, prefix, regime_thresholds)

    keep_ts = set(src2["timestamp"])
    frame2 = frame[frame["timestamp"].isin(keep_ts)].reset_index(drop=True)
    src2 = src2[src2["timestamp"].isin(set(frame2["timestamp"]))].reset_index(drop=True)
    if len(src2) != len(frame2) or not src2["timestamp"].equals(frame2["timestamp"]):
        raise RuntimeError(f"{name}: prediction/frame timestamp mismatch")

    x = base_sweep.parent._base_input(frame2, base_sweep.torch.load(cfg["bundle"], map_location="cpu", weights_only=False)["base_cols"])
    dec_base = base_sweep.parent._to_decisions(src2, oof=oof)
    dec, _ = base_sweep.atr_eval._apply_atr_safety_sltp(
        dec_base, frame2, atr_window=cfg["atr_window"], tp_mult=cfg["tp_mult"], sl_mult=cfg["sl_mult"],
        min_tp=cfg["min_tp"], min_sl=cfg["min_sl"], max_tp=cfg["max_tp"], max_sl=cfg["max_sl"],
    )
    atr_pct = base_sweep.atr_eval._atr_pct(frame2, cfg["atr_window"])
    fee, slip = base_sweep.omega._load_fee_slip()
    bundle = base_sweep.torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
    loaded = base_sweep.parent._load_payloads(bundle["models"], device=base_sweep.DEVICE)

    import pickle
    with open(cfg["sidecar_pkl"], "rb") as f:
        pkl = pickle.load(f)
    features = base_sweep.rs._risk_feature_frame(frame2, src2, dec, bundle["base_cols"], atr_pct=atr_pct, feature_mode=pkl["risk_feature_mode"])
    x_all, _ = base_sweep.rs._feature_matrix(features, pkl["feature_columns"])
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    score = base_sweep.rs._predict_side_split_models(pkl["model"], x_all, side_all) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all), dtype=np.float64)
    mapping = pkl["selected_mapping"]
    margin = base_sweep.rs._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in base_sweep.rs.MARGIN_CFG_KEYS})
    leverage = None
    if pkl["dynamic_leverage"]:
        leverage = base_sweep.rs._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in base_sweep.rs.LEVERAGE_CFG_KEYS})

    p = dict(frame=frame2, x=x, dec=dec, loaded=loaded, margin=margin, leverage=leverage,
             fee=fee, slip=slip, notional_scaled_sltp=pkl["notional_scaled_sltp"])
    m, _ledger = base_sweep.replay_exit_variant(
        p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
        exit_threshold=BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=base_sweep.COST_MULT,
        notional_scaled_sltp=p["notional_scaled_sltp"], device=base_sweep.DEVICE,
    )
    return {k: v for k, v in m.items() if k != "exit_reasons"}, p


def portfolio_eval(frame: pd.DataFrame, prepped: dict[str, dict[str, Any]]) -> tuple[dict, dict]:
    router_components = {name: helpers._as_router_component(p, exit_threshold=BASELINE_EXIT_THRESHOLD) for name, p in prepped.items()}
    fee0, slip0 = prepped["h48qual"]["fee"], prepped["h48qual"]["slip"]
    _, ledger = router.greedy_replay(frame, router_components, fee=fee0, slip=slip0, cost_mult=base_sweep.COST_MULT, device=base_sweep.DEVICE)
    return helpers._ledger_stats(ledger, frame), helpers._duration_gated(ledger, frame, router.DURATION_THRESHOLD)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_frame = base_sweep.load_frame(base_sweep.VAL_START, base_sweep.VAL_END, base_csv=base_sweep.BASE_2025, wide24_csv=base_sweep.WIDE24_2025)
    prefix = base_sweep.omega._tabm_prefix(True)
    log(f"prefix={prefix!r}")

    raw = {}
    for name, cfg in base_sweep.COMPONENTS.items():
        pred_csv = base_sweep.EXT_PRED_DIR / name / f"validation_predictions_{cfg['q_tag']}.csv"
        raw[name] = load_raw(name, cfg, "VAL", pred_csv)
        counts = raw[name][f"{prefix}router_expert"].value_counts().to_dict()
        log(f"{name} VAL regime bar counts: {counts}")

    # --- G0: baseline (flat global threshold, same for all 3 regimes) must reproduce known numbers.
    baseline_map = {name: {r: GLOBAL_BASELINE[name] for r in REGIMES} for name in base_sweep.COMPONENTS}
    prepped_baseline = {}
    baseline_component_rows = {}
    for name, cfg in base_sweep.COMPONENTS.items():
        m, p = evaluate_component(name, cfg, val_frame, raw[name], prefix, baseline_map[name], oof=True)
        baseline_component_rows[name] = m
        prepped_baseline[name] = p
        log(f"G0 baseline component={name} pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% trades={m['trades']}")
    no_gate_base, with_gate_base = portfolio_eval(val_frame, prepped_baseline)
    log(f"G0 baseline PORTFOLIO no_gate={no_gate_base} with_gate={with_gate_base}")
    g0_ok = abs(no_gate_base["pnl"] - 36.82) < 0.5 and abs(no_gate_base["mdd"] - (-24.34)) < 0.5 and no_gate_base["trades"] == 29
    log(f"G0 self-consistency: {'PASS' if g0_ok else 'FAIL'}")
    if not g0_ok:
        (OUT_DIR / "report.json").write_text(json.dumps({"g0_ok": False}, indent=2))
        return 1

    # --- Stage 1: univariate per-regime sweep (component level only, cheap), holding other 2 regimes
    # at global baseline. One component at a time.
    univariate_rows = []
    best_per_regime: dict[str, dict[str, float]] = {}
    for name, cfg in base_sweep.COMPONENTS.items():
        best_per_regime[name] = {}
        for regime in REGIMES:
            best_pnl, best_q = None, GLOBAL_BASELINE[name]
            for q in THRESHOLD_GRID:
                tmap = dict(baseline_map[name])
                tmap[regime] = q
                m, _p = evaluate_component(name, cfg, val_frame, raw[name], prefix, tmap, oof=True)
                univariate_rows.append({"component": name, "regime": regime, "threshold": q, **m})
                if best_pnl is None or m["pnl"] > best_pnl:
                    best_pnl, best_q = m["pnl"], q
            best_per_regime[name][regime] = best_q
            log(f"{name} regime={regime}: VAL-best threshold={best_q} (component pnl={best_pnl:.2f}%)")

    pd.DataFrame(univariate_rows).to_csv(OUT_DIR / "univariate_sweep_val.csv", index=False)

    # --- Stage 2: combine each component's per-regime VAL-best values into one joint map, evaluate
    # jointly (component + portfolio) on VAL.
    joint_map = best_per_regime
    log(f"Joint regime-threshold map (VAL-selected): {joint_map}")
    prepped_joint = {}
    joint_component_rows = {}
    for name, cfg in base_sweep.COMPONENTS.items():
        m, p = evaluate_component(name, cfg, val_frame, raw[name], prefix, joint_map[name], oof=True)
        joint_component_rows[name] = m
        prepped_joint[name] = p
        log(f"JOINT component={name} pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% trades={m['trades']}")
    no_gate_joint, with_gate_joint = portfolio_eval(val_frame, prepped_joint)
    log(f"JOINT PORTFOLIO no_gate={no_gate_joint} with_gate={with_gate_joint}")

    beats = (no_gate_joint["pnl"] >= no_gate_base["pnl"] and no_gate_joint["mdd"] >= no_gate_base["mdd"] and
             with_gate_joint["pnl"] >= with_gate_base["pnl"] and with_gate_joint["mdd"] >= with_gate_base["mdd"])
    log(f"Joint map {'BEATS' if beats else 'does NOT beat'} flat-global baseline on VAL (pnl+mdd, no_gate+with_gate).")

    result: dict[str, Any] = {
        "g0_ok": True, "baseline_no_gate": no_gate_base, "baseline_with_gate": with_gate_base,
        "joint_regime_threshold_map": joint_map, "joint_no_gate": no_gate_joint, "joint_with_gate": with_gate_joint,
        "joint_beats_baseline_val": bool(beats), "oos_run": False,
    }

    if not beats:
        log("VAL bar not cleared -- OOS NOT opened. Negative pilot result for regime-specific quality_threshold.")
        (OUT_DIR / "report.json").write_text(json.dumps(result, indent=2, default=str))
        return 0

    # --- Single-touch OOS on the joint map only (vs flat-global baseline).
    oos_frame = base_sweep.load_frame(base_sweep.OOS_START, base_sweep.OOS_END, base_csv=base_sweep.BASE_2026, wide24_csv=base_sweep.WIDE24_2026)
    prefix_oos = base_sweep.omega._tabm_prefix(False)
    raw_oos = {}
    for name, cfg in base_sweep.COMPONENTS.items():
        pred_csv = base_sweep.EXT_PRED_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        raw_oos[name] = load_raw(name, cfg, "OOS", pred_csv)

    prepped_oos_base, prepped_oos_joint = {}, {}
    for name, cfg in base_sweep.COMPONENTS.items():
        _, p_base = evaluate_component(name, cfg, oos_frame, raw_oos[name], prefix_oos, baseline_map[name], oof=False)
        _, p_joint = evaluate_component(name, cfg, oos_frame, raw_oos[name], prefix_oos, joint_map[name], oof=False)
        prepped_oos_base[name], prepped_oos_joint[name] = p_base, p_joint
    oos_no_gate_base, oos_with_gate_base = portfolio_eval(oos_frame, prepped_oos_base)
    oos_no_gate_joint, oos_with_gate_joint = portfolio_eval(oos_frame, prepped_oos_joint)
    oos_survives = (oos_no_gate_joint["pnl"] >= oos_no_gate_base["pnl"] and oos_no_gate_joint["mdd"] >= oos_no_gate_base["mdd"])
    log(f"OOS baseline no_gate={oos_no_gate_base} | OOS joint no_gate={oos_no_gate_joint} -> {'SURVIVES' if oos_survives else 'REVERSES'}")

    result.update({
        "oos_run": True, "oos_baseline_no_gate": oos_no_gate_base, "oos_joint_no_gate": oos_no_gate_joint,
        "oos_baseline_with_gate": oos_with_gate_base, "oos_joint_with_gate": oos_with_gate_joint,
        "oos_survives": bool(oos_survives),
    })
    (OUT_DIR / "report.json").write_text(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
