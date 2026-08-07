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

from scripts.eval_hf_entry_overlay_grid import _audit  # noqa: E402
from scripts.train_eval_alphazero_style_governor_2026 import DEFAULT_MODEL_OUT as DEFAULT_AZ_MODEL  # noqa: E402
from scripts.train_eval_dsac_replacement_heads_2026 import (  # noqa: E402
    DEFAULT_EVAL_CSV,
    DEFAULT_EXIT_BUNDLE,
    DEFAULT_POLICY,
    DEFAULT_SELECTION,
    DEFAULT_TRAIN_CSV,
    _load_selected,
    _read,
)
from scripts.train_eval_hf_no_limit_exit_governor import MODEL_COLS, _base_frame, collect_exit_samples  # noqa: E402
from scripts.train_eval_muzero_style_exit_governor_2026 import DEFAULT_MODEL_OUT as DEFAULT_MZ_EXIT_MODEL  # noqa: E402
from scripts.train_eval_muzero_style_governor_2026 import DEFAULT_MODEL_OUT as DEFAULT_MZ_ENTRY_MODEL  # noqa: E402
from scripts.train_eval_zero_style_remaining_layers_2026 import (  # noqa: E402
    AZExitArbiter,
    REGIME_COLS,
    _eval_az_scale_variants,
    _eval_mz_scale_variants,
    _fit_az_scale,
    _load_mz_exit,
    _load_mz_risk,
    _load_pv,
    _proba_1,
    _select_stage,
)
from scripts.train_eval_zero_style_risk_overlay_2026 import (  # noqa: E402
    DEFAULT_AZ_RISK_OUT,
    RISK_ACTIONS,
    _apply_scale,
    _mz_entry_decisions,
    _risk_targets,
    _run_bt,
    _state_frame,
    _train_mz_risk,
)
from scripts.train_eval_alphazero_style_governor_2026 import PolicyValueNet, PVBundle, _predict_pv, _train_pv  # noqa: E402


DEFAULT_REPORT = ROOT / "data/ensemble/reports/zero_style_remaining_layers_walkforward_2026.json"
DEFAULT_STAGE_DIR = ROOT / "data/ensemble/supervised/zero_style/remaining_layers_walkforward"


def _range(df: pd.DataFrame) -> list[str]:
    if "timestamp" not in df.columns or df.empty:
        return ["", ""]
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if ts.empty:
        return ["", ""]
    return [str(ts.min()), str(ts.max())]


def _overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
    if "timestamp" not in a.columns or "timestamp" not in b.columns:
        return 0
    ta = pd.to_datetime(a["timestamp"], errors="coerce").dropna().astype("int64")
    tb = pd.to_datetime(b["timestamp"], errors="coerce").dropna().astype("int64")
    return int(len(set(ta.tolist()) & set(tb.tolist())))


def _apply_az_risk(dec: pd.DataFrame, state: pd.DataFrame, bundle: PVBundle, device: str) -> pd.DataFrame:
    x = state.reindex(columns=bundle.feature_cols).to_numpy(dtype=np.float32)
    probs, values = _predict_pv(bundle, x, device)
    idx = np.argmax(probs, axis=1)
    idx = np.where(values < -0.15, 3, idx)
    return _apply_scale(dec, idx)


def _reconstruct_scale_decision(name: str, dec: pd.DataFrame, state: pd.DataFrame, az_bundle: PVBundle, mz_bundle: Any, device: str) -> pd.DataFrame:
    if "_az_" in name:
        x = state.reindex(columns=az_bundle.feature_cols).to_numpy(dtype=np.float32)
        probs, values = _predict_pv(az_bundle, x, device)
        parts = name.split("_")
        cf = float(parts[-2].replace("cf", ""))
        vf = float(parts[-1].replace("vf", ""))
        idx = np.argmax(probs, axis=1)
        idx = np.where((probs.max(axis=1) < cf) | (values < vf), 3, idx)
        return _apply_scale(dec, idx)
    parts = name.split("_")
    gamma = float(parts[3].replace("g", ""))
    prior = float(parts[4].replace("p", ""))
    depth = int(parts[5].replace("d", ""))
    sf = float(parts[6].replace("sf", ""))
    from scripts.train_eval_zero_style_risk_overlay_2026 import _predict_mz_risk

    scores, _, _ = _predict_mz_risk(mz_bundle, state.reindex(columns=mz_bundle.feature_cols).to_numpy(dtype=np.float32), device=device, gamma=gamma, prior_weight=prior, depth=depth)
    idx = np.where(scores.max(axis=1) < sf, 3, np.argmax(scores, axis=1))
    return _apply_scale(dec, idx)


def _fit_exit_arbiter(train_df: pd.DataFrame, policy: dict[str, Any], entry_cfg: dict[str, Any], base_exit: Any, az_exit: Any, mz_exit: Any, args: argparse.Namespace, device: str) -> tuple[AZExitArbiter, dict[str, Any], dict[str, Any]]:
    x_exit, y_exit, exit_meta = collect_exit_samples(
        train_df,
        policy,
        entry_config=entry_cfg,
        fee=args.fee,
        slip=args.slip,
        entry_stride=24,
        min_age=3,
        max_age=288,
        age_stride=12,
        future_horizon=args.horizon,
        exit_edge=0.0015,
        adverse_gap=0.012,
        max_samples=args.samples,
        seed=args.seed + 30,
    )
    x_arr = x_exit.reindex(columns=MODEL_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    meta_probs = np.column_stack([_proba_1(base_exit, x_arr), _proba_1(az_exit, x_arr), _proba_1(mz_exit, x_arr)]).astype(np.float32)
    pi = np.zeros((len(y_exit), 2), dtype=np.float32)
    pi[:, 1] = y_exit.astype(np.float32) * 0.85 + 0.075
    pi[:, 0] = 1.0 - pi[:, 1]
    value = np.where(y_exit > 0, 0.35, 0.10).astype(np.float32)
    net, mean, std, meta = _train_pv(
        np.concatenate([x_arr, meta_probs], axis=1),
        pi,
        value,
        n_actions=2,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed + 40,
    )
    bundle = PVBundle(net, mean, std, list(MODEL_COLS) + ["p_base_exit", "p_az_exit", "p_mz_exit"], ("hold", "exit"))
    return AZExitArbiter(bundle, base_exit, az_exit, mz_exit, device), exit_meta, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward validation for remaining Zero-style layers.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-bundle", type=Path, default=DEFAULT_EXIT_BUNDLE)
    p.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--mz-entry-model", type=Path, default=DEFAULT_MZ_ENTRY_MODEL)
    p.add_argument("--az-model", type=Path, default=DEFAULT_AZ_MODEL)
    p.add_argument("--az-risk-model", type=Path, default=DEFAULT_AZ_RISK_OUT)
    p.add_argument("--mz-exit-model", type=Path, default=DEFAULT_MZ_EXIT_MODEL)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    p.add_argument("--split-date", type=str, default="2025-11-01")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1.2e-3)
    p.add_argument("--samples", type=int, default=70000)
    p.add_argument("--horizon", type=int, default=144)
    p.add_argument("--dynamics-step", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.010)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    policy = joblib.load(args.policy)
    exit_bundle = joblib.load(args.exit_bundle)
    base_exit = exit_bundle["model"] if isinstance(exit_bundle, dict) and "model" in exit_bundle else exit_bundle
    entry_cfg, risk_cfg, exit_cfg = _load_selected(args.selection_report)
    from scripts.train_eval_muzero_style_governor_2026 import _load_az_exit

    az_exit = _load_az_exit(args.az_model, device)
    mz_exit = _load_mz_exit(args.mz_exit_model, device)
    az_risk = _load_pv(args.az_risk_model, 6, RISK_ACTIONS, device)

    all_2025 = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    ts = pd.to_datetime(all_2025["timestamp"], errors="coerce")
    train_df = all_2025.loc[ts < pd.Timestamp(args.split_date)].reset_index(drop=True)
    val_df = all_2025.loc[ts >= pd.Timestamp(args.split_date)].reset_index(drop=True)

    # Build fixed stage1 streams for train/validation/eval.
    from scripts.train_eval_zero_style_risk_overlay_2026 import _load_mz_entry

    mz_entry = _load_mz_entry(args.mz_entry_model, device)

    train_feat0, train_dec0, _, _, train_scores, train_probs, train_vals = _mz_entry_decisions(train_df, policy, entry_cfg, mz_entry, device=device)
    train_state0 = _state_frame(train_feat0, train_dec0, train_scores, train_probs, train_vals)
    train_dec = _apply_az_risk(train_dec0, train_state0, az_risk, device)

    val_feat, val_dec0, val_close, val_fill, val_scores, val_probs, val_vals = _mz_entry_decisions(val_df, policy, entry_cfg, mz_entry, device=device)
    val_state0 = _state_frame(val_feat, val_dec0, val_scores, val_probs, val_vals)
    val_dec = _apply_az_risk(val_dec0, val_state0, az_risk, device)

    eval_feat, eval_dec0, eval_close, eval_fill, eval_scores, eval_probs, eval_vals = _mz_entry_decisions(eval_df, policy, entry_cfg, mz_entry, device=device)
    eval_state0 = _state_frame(eval_feat, eval_dec0, eval_scores, eval_probs, eval_vals)
    eval_dec = _apply_az_risk(eval_dec0, eval_state0, az_risk, device)

    exit_model: Any = az_exit
    exit_cfg_current = {"exit_threshold": 0.45, "min_exit_age": exit_cfg["min_exit_age"]}
    val_current = _run_bt("stage1_current", val_df, policy, exit_model, entry_cfg, risk_cfg, exit_cfg_current, (val_feat, val_dec, val_close, val_fill), fee=args.fee, slip=args.slip)
    eval_current = _run_bt("stage1_current", eval_df, policy, exit_model, entry_cfg, risk_cfg, exit_cfg_current, (eval_feat, eval_dec, eval_close, eval_fill), fee=args.fee, slip=args.slip, monthly=True)
    stage_reports: list[dict[str, Any]] = [{"stage": 1, "validation_selected": val_current, "eval_result": eval_current, "improved": True}]

    # Stage 2: fit on train, select on validation, apply once to eval.
    train_state2 = _state_frame(train_feat0, train_dec, train_scores, train_probs, train_vals)
    x2, x2n, pi2, v2, r2, meta2 = _risk_targets(train_df, train_state2, train_dec, horizon=args.horizon, dynamics_step=args.dynamics_step, fee=args.fee, slip=args.slip, temperature=args.temperature, max_samples=args.samples, seed=args.seed)
    az2, az2_meta = _fit_az_scale(x2, pi2, v2, args, device, args.seed + 10)
    az2.feature_cols = list(train_state2.columns)
    mz2_net, mz2_mean, mz2_std, mz2_meta = _train_mz_risk(x2, x2n, pi2, v2, r2, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device, seed=args.seed + 20)
    from scripts.train_eval_zero_style_risk_overlay_2026 import MZRiskBundle

    mz2 = MZRiskBundle(mz2_net, mz2_mean, mz2_std, list(train_state2.columns), RISK_ACTIONS)
    val_state2 = _state_frame(val_feat, val_dec, val_scores, val_probs, val_vals)
    val_stage2 = _eval_az_scale_variants("stage2_sleeve", val_df, policy, az_exit, entry_cfg, risk_cfg, exit_cfg, val_feat, val_dec, val_close, val_fill, val_state2, az2, device, fee=args.fee, slip=args.slip)
    val_stage2 += _eval_mz_scale_variants("stage2_sleeve", val_df, policy, az_exit, entry_cfg, risk_cfg, exit_cfg, val_feat, val_dec, val_close, val_fill, val_state2, mz2, device, fee=args.fee, slip=args.slip)
    val_best2 = _select_stage(val_current, val_stage2)
    if val_best2["name"] != val_current["name"]:
        val_dec = _reconstruct_scale_decision(val_best2["name"], val_dec, val_state2, az2, mz2, device)
        eval_state2 = _state_frame(eval_feat, eval_dec, eval_scores, eval_probs, eval_vals)
        eval_dec = _reconstruct_scale_decision(val_best2["name"], eval_dec, eval_state2, az2, mz2, device)
        train_dec = _reconstruct_scale_decision(val_best2["name"], train_dec, train_state2, az2, mz2, device)
        val_current = val_best2
        eval_current = _run_bt("stage2_selected_eval", eval_df, policy, exit_model, entry_cfg, risk_cfg, exit_cfg_current, (eval_feat, eval_dec, eval_close, eval_fill), fee=args.fee, slip=args.slip, monthly=True)
    stage_reports.append({"stage": 2, "validation_best": max(val_stage2, key=lambda r: r["eval"]["pnl"]), "validation_selected": val_current, "eval_result": eval_current, "improved": val_best2["name"] == val_current["name"] and val_best2["name"] != "stage1_current", "label_meta": meta2, "train_meta": {"az": az2_meta, "mz": mz2_meta}})

    # Stage 3: fit exit arbiter on train, select on validation.
    exit_arb, exit_meta, exit_train_meta = _fit_exit_arbiter(train_df, policy, entry_cfg, base_exit, az_exit, mz_exit, args, device)
    val_stage3 = [
        _run_bt("stage3_keep_current_exit", val_df, policy, exit_model, entry_cfg, risk_cfg, exit_cfg_current, (val_feat, val_dec, val_close, val_fill), fee=args.fee, slip=args.slip),
        _run_bt("stage3_az_exit_arb_th0.45", val_df, policy, exit_arb, entry_cfg, risk_cfg, {"exit_threshold": 0.45, "min_exit_age": exit_cfg["min_exit_age"]}, (val_feat, val_dec, val_close, val_fill), fee=args.fee, slip=args.slip),
        _run_bt("stage3_az_exit_arb_th0.55", val_df, policy, exit_arb, entry_cfg, risk_cfg, {"exit_threshold": 0.55, "min_exit_age": exit_cfg["min_exit_age"]}, (val_feat, val_dec, val_close, val_fill), fee=args.fee, slip=args.slip),
        _run_bt("stage3_mz_exit_defensive_th0.54", val_df, policy, mz_exit, entry_cfg, risk_cfg, {"exit_threshold": 0.54, "min_exit_age": exit_cfg["min_exit_age"]}, (val_feat, val_dec, val_close, val_fill), fee=args.fee, slip=args.slip),
    ]
    val_best3 = _select_stage(val_current, val_stage3)
    if val_best3["name"].startswith("stage3_az_exit_arb"):
        exit_model = exit_arb
        th = float(val_best3["name"].rsplit("th", 1)[1])
        exit_cfg_current = {"exit_threshold": th, "min_exit_age": exit_cfg["min_exit_age"]}
        val_current = val_best3
        eval_current = _run_bt("stage3_selected_eval", eval_df, policy, exit_model, entry_cfg, risk_cfg, exit_cfg_current, (eval_feat, eval_dec, eval_close, eval_fill), fee=args.fee, slip=args.slip, monthly=True)
    elif val_best3["name"].startswith("stage3_mz_exit"):
        exit_model = mz_exit
        exit_cfg_current = {"exit_threshold": 0.54, "min_exit_age": exit_cfg["min_exit_age"]}
        val_current = val_best3
        eval_current = _run_bt("stage3_selected_eval", eval_df, policy, exit_model, entry_cfg, risk_cfg, exit_cfg_current, (eval_feat, eval_dec, eval_close, eval_fill), fee=args.fee, slip=args.slip, monthly=True)
    stage_reports.append({"stage": 3, "validation_best": max(val_stage3, key=lambda r: r["eval"]["pnl"]), "validation_selected": val_current, "eval_result": eval_current, "improved": val_best3["name"] == val_current["name"] and val_best3["name"] != "stage3_keep_current_exit", "label_meta": exit_meta, "train_meta": exit_train_meta})

    # Stage 4: fit regime overlay on train, select on validation.
    train_state4_full = _state_frame(train_feat0, train_dec, train_scores, train_probs, train_vals)
    val_state4_full = _state_frame(val_feat, val_dec, val_scores, val_probs, val_vals)
    eval_state4_full = _state_frame(eval_feat, eval_dec, eval_scores, eval_probs, eval_vals)
    cols4 = [c for c in REGIME_COLS if c in train_state4_full.columns]
    train_state4 = train_state4_full.reindex(columns=cols4).fillna(0.0)
    val_state4 = val_state4_full.reindex(columns=cols4).fillna(0.0)
    eval_state4 = eval_state4_full.reindex(columns=cols4).fillna(0.0)
    x4, x4n, pi4, v4, r4, meta4 = _risk_targets(train_df, train_state4, train_dec, horizon=args.horizon, dynamics_step=args.dynamics_step, fee=args.fee, slip=args.slip, temperature=args.temperature, max_samples=args.samples, seed=args.seed + 50)
    az4, az4_meta = _fit_az_scale(x4, pi4, v4, args, device, args.seed + 60)
    az4.feature_cols = cols4
    mz4_net, mz4_mean, mz4_std, mz4_meta = _train_mz_risk(x4, x4n, pi4, v4, r4, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device, seed=args.seed + 70)
    mz4 = MZRiskBundle(mz4_net, mz4_mean, mz4_std, cols4, RISK_ACTIONS)
    val_stage4 = _eval_az_scale_variants("stage4_regime", val_df, policy, exit_model, entry_cfg, risk_cfg, exit_cfg_current, val_feat, val_dec, val_close, val_fill, val_state4, az4, device, fee=args.fee, slip=args.slip)
    val_stage4 += _eval_mz_scale_variants("stage4_regime", val_df, policy, exit_model, entry_cfg, risk_cfg, exit_cfg_current, val_feat, val_dec, val_close, val_fill, val_state4, mz4, device, fee=args.fee, slip=args.slip)
    val_best4 = _select_stage(val_current, val_stage4)
    if val_best4["name"] != val_current["name"]:
        val_dec = _reconstruct_scale_decision(val_best4["name"], val_dec, val_state4, az4, mz4, device)
        eval_dec = _reconstruct_scale_decision(val_best4["name"], eval_dec, eval_state4, az4, mz4, device)
        val_current = val_best4
        eval_current = _run_bt("stage4_selected_eval", eval_df, policy, exit_model, entry_cfg, risk_cfg, exit_cfg_current, (eval_feat, eval_dec, eval_close, eval_fill), fee=args.fee, slip=args.slip, monthly=True)
    stage_reports.append({"stage": 4, "validation_best": max(val_stage4, key=lambda r: r["eval"]["pnl"]), "validation_selected": val_current, "eval_result": eval_current, "improved": val_best4["name"] == val_current["name"] and val_best4["name"] != stage_reports[-1]["validation_selected"]["name"], "label_meta": meta4, "train_meta": {"az": az4_meta, "mz": mz4_meta}})

    cost_stress: dict[str, list[dict[str, Any]]] = {}
    for mult in (1.0, 2.0, 3.0):
        cost_stress[f"cost_{mult:g}x"] = [
            _run_bt("walkforward_selected_eval_replay", eval_df, policy, exit_model, entry_cfg, risk_cfg, exit_cfg_current, (eval_feat, eval_dec, eval_close, eval_fill), fee=args.fee * mult, slip=args.slip * mult),
        ]

    args.stage_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"type": "wf_stage2_sleeve_az", "state_dict": az2.net.state_dict(), "mean": az2.mean, "std": az2.std, "feature_cols": az2.feature_cols, "actions": list(RISK_ACTIONS)}, args.stage_dir / "wf_stage2_sleeve_az.pt")
    torch.save({"type": "wf_stage2_sleeve_mz", "state_dict": mz2.net.state_dict(), "mean": mz2.mean, "std": mz2.std, "feature_cols": mz2.feature_cols, "actions": list(RISK_ACTIONS)}, args.stage_dir / "wf_stage2_sleeve_mz.pt")
    torch.save({"type": "wf_stage4_regime_az", "state_dict": az4.net.state_dict(), "mean": az4.mean, "std": az4.std, "feature_cols": az4.feature_cols, "actions": list(RISK_ACTIONS)}, args.stage_dir / "wf_stage4_regime_az.pt")
    torch.save({"type": "wf_stage4_regime_mz", "state_dict": mz4.net.state_dict(), "mean": mz4.mean, "std": mz4.std, "feature_cols": mz4.feature_cols, "actions": list(RISK_ACTIONS)}, args.stage_dir / "wf_stage4_regime_mz.pt")

    report = {
        "type": "zero_style_remaining_layers_walkforward_2026",
        "note": "Layers are trained on 2025 pre-validation, selected on late-2025 validation, then applied once to 2026.",
        "audit": {
            "source_audit": _audit(args.train_csv, args.eval_csv, policy),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "eval_rows": int(len(eval_df)),
            "train_range": _range(train_df),
            "validation_range": _range(val_df),
            "eval_range": _range(eval_df),
            "train_validation_overlap": _overlap(train_df, val_df),
            "train_eval_overlap": _overlap(train_df, eval_df),
            "validation_eval_overlap": _overlap(val_df, eval_df),
        },
        "stage_reports": stage_reports,
        "cost_stress": cost_stress,
        "stage_dir": str(args.stage_dir),
        "decision": {
            "final_validation_name": val_current["name"],
            "final_validation_pnl": val_current["eval"]["pnl"],
            "final_eval_pnl": eval_current["eval"]["pnl"],
            "final_eval_mdd": eval_current["eval"]["mdd"],
            "final_eval_trades": eval_current["eval"]["trades"],
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "decision": report["decision"], "stages": stage_reports}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
