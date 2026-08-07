#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_mamba_sac_3head_feature_coordinator_20260604 as feat_coord  # noqa: E402
import train_eval_omega1_2_mamba_sac_lifecycle_controller_20260604 as lifecycle  # noqa: E402
import train_eval_omega1_2_supervised_risk_selector_20260604 as sup_risk  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_risk_selector_20260605 as tabm_risk  # noqa: E402


MODEL_ID = "omega1_2_lifecycle_with_tabm_risk_selector_20260605"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DEFAULT_RISK_SELECTOR = ROOT / "tmp/causal_regen_20260516/omega1_2_tabm_risk_selector_20260605_full_e50_s6k_seed260605/tabm_risk_selector.pt"


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _load_risk_selector(path: Path, *, device: torch.device) -> tuple[tabm_risk.RiskTabM, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    n_features = int(payload["n_features"])
    n_classes = int(payload["n_classes"])
    model = tabm_risk.RiskTabM(n_features, n_classes)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload["scaler"]


def _final_action(src: pd.DataFrame, *, oof: bool) -> np.ndarray:
    prefix = "omega1_regime3_expertdq_oof_" if oof else "omega1_regime3_expertdq_"
    action = pd.to_numeric(src[f"{prefix}final_action"], errors="raise").to_numpy(dtype=np.int64)
    if not set(np.unique(action)).issubset({omega.ACTION_CASH, omega.ACTION_LONG, omega.ACTION_SHORT}):
        raise RuntimeError(f"unexpected final_action values: {sorted(np.unique(action).tolist())}")
    return action


def _risk_decision_from_selector(
    base_x: pd.DataFrame,
    src: pd.DataFrame,
    *,
    oof: bool,
    model: tabm_risk.RiskTabM,
    scaler: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    action = _final_action(src, oof=oof)
    candidate = action != omega.ACTION_CASH
    risk_class = np.zeros(len(action), dtype=np.int64)
    if bool(candidate.any()):
        x_risk = sup_risk._risk_features(base_x, src, oof=oof)
        risk_class[candidate] = tabm_risk._predict(
            model,
            x_risk.loc[candidate].reset_index(drop=True),
            scaler,
            device=device,
            batch_size=int(batch_size),
        )
    return sup_risk._risk_decision(src, oof=oof, action=action, risk_class=risk_class)


def _prepare_frames_with_tabm_risk(
    *,
    threehead_dir: Path,
    risk_selector: Path,
    quality_threshold: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    base_frames = feat_coord._prepare_frames(threehead_dir, quality_threshold=float(quality_threshold), device=device)
    bundle = feat_coord._load_3head_payloads(threehead_dir)
    risk_model, risk_scaler = _load_risk_selector(risk_selector, device=device)

    train_x, train_src = feat_coord._predict_3head_frame(base_frames["train_df"], bundle, quality_threshold=float(quality_threshold), device=device, oof=True)
    val_x, val_src = feat_coord._predict_3head_frame(base_frames["val_df"], bundle, quality_threshold=float(quality_threshold), device=device, oof=True)
    oos_x, oos_src = feat_coord._predict_3head_frame(base_frames["oos_df"], bundle, quality_threshold=float(quality_threshold), device=device, oof=False)

    train_dec = _risk_decision_from_selector(train_x, train_src, oof=True, model=risk_model, scaler=risk_scaler, device=device, batch_size=int(batch_size))
    val_dec = _risk_decision_from_selector(val_x, val_src, oof=True, model=risk_model, scaler=risk_scaler, device=device, batch_size=int(batch_size))
    oos_dec = _risk_decision_from_selector(oos_x, oos_src, oof=False, model=risk_model, scaler=risk_scaler, device=device, batch_size=int(batch_size))

    feature_cols = omega._numeric_feature_cols(
        pd.concat([base_frames["train_df"], base_frames["val_df"]], axis=0, ignore_index=True),
        base_frames["oos_df"],
    )
    s_train = omega._build_state_frame(base_frames["train_df"], train_src, train_dec, oof=True, feature_cols=feature_cols)
    s_val = omega._build_state_frame(base_frames["val_df"], val_src, val_dec, oof=True, feature_cols=feature_cols)
    s_oos = omega._build_state_frame(base_frames["oos_df"], oos_src, oos_dec, oof=False, feature_cols=feature_cols)
    for state, src, prefix in (
        (s_train, train_src, "omega1_regime3_expertdq_oof"),
        (s_val, val_src, "omega1_regime3_expertdq_oof"),
        (s_oos, oos_src, "omega1_regime3_expertdq"),
    ):
        state["threehead_exit_p_hold_feature_only"] = pd.to_numeric(src[f"{prefix}_exit_p_hold_feature_only"], errors="raise").to_numpy(dtype=np.float64)
        state["threehead_exit_p_exit_feature_only"] = pd.to_numeric(src[f"{prefix}_exit_p_exit_feature_only"], errors="raise").to_numpy(dtype=np.float64)
        state["threehead_exit_edge_feature_only"] = pd.to_numeric(src[f"{prefix}_exit_edge_feature_only"], errors="raise").to_numpy(dtype=np.float64)

    out = dict(base_frames)
    out.update(
        {
            "train_dec": train_dec,
            "val_dec": val_dec,
            "oos_dec": oos_dec,
            "s_train": s_train,
            "s_val": s_val,
            "s_oos": s_oos,
            "risk_selector": str(risk_selector),
        }
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threehead-dir", type=Path, default=feat_coord.DEFAULT_3HEAD_DIR)
    ap.add_argument("--risk-selector", type=Path, default=DEFAULT_RISK_SELECTOR)
    ap.add_argument("--quality-threshold", type=float, default=0.75)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-train-entries", type=int, default=600)
    ap.add_argument("--samples-per-entry", type=int, default=6)
    ap.add_argument("--train-max-sim-bars", type=int, default=96)
    ap.add_argument("--min-action-edge", type=float, default=0.002)
    ap.add_argument("--disable-resize", action="store_true")
    ap.add_argument("--disable-reverse", action="store_true")
    ap.add_argument("--position-only-training", action="store_true")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--risk-batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--class-balance-actor", action="store_true")
    ap.add_argument("--select-mode", choices=["actor", "q", "actor_q"], default="actor")
    ap.add_argument("--force-parent-entry", action="store_true")
    ap.add_argument("--force-entry-mult", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=260622)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    lifecycle._seed_everything(int(args.seed))
    device = lifecycle._device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = _prepare_frames_with_tabm_risk(
        threehead_dir=Path(args.threehead_dir),
        risk_selector=Path(args.risk_selector),
        quality_threshold=float(args.quality_threshold),
        device=device,
        batch_size=int(args.risk_batch_size),
    )
    fee, slip = omega._load_fee_slip()
    state_cols = [c for c in lifecycle._base_state(frames["s_train"]).columns if c != "timestamp"]
    bad = [c for c in state_cols if "clean_regime4" in c or "regime4_pred" in c or "tp_sl_action_score" in c or c.startswith("teacher_")]
    if bad:
        raise RuntimeError(f"forbidden lifecycle state columns passed audit: {bad[:20]}")
    norm = lifecycle._fit_norm(lifecycle._base_state(frames["s_train"])[state_cols])
    data, data_diag = lifecycle._build_dataset(
        frames,
        seq_len=int(args.seq_len),
        max_entries=int(args.max_train_entries),
        samples_per_entry=int(args.samples_per_entry),
        seed=int(args.seed),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_sim_bars=int(args.train_max_sim_bars),
        min_action_edge=float(args.min_action_edge),
        disable_resize=bool(args.disable_resize),
        disable_reverse=bool(args.disable_reverse),
        position_only_training=bool(args.position_only_training),
        norm=norm,
    )
    print(json.dumps({"stage": "hybrid_lifecycle_train_start", "device": str(device), "seq_shape": list(data.seq.shape), "data_diag": data_diag}, ensure_ascii=False), flush=True)
    model, train_diag = lifecycle._train(
        data,
        device=device,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        class_balance_actor=bool(args.class_balance_actor),
    )
    val = lifecycle._replay(
        frames,
        "val",
        model,
        norm,
        seq_len=int(args.seq_len),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
        disable_resize=bool(args.disable_resize),
        disable_reverse=bool(args.disable_reverse),
        select_mode=str(args.select_mode),
        force_parent_entry=bool(args.force_parent_entry),
        force_entry_mult=float(args.force_entry_mult),
    )
    oos = lifecycle._replay(
        frames,
        "oos",
        model,
        norm,
        seq_len=int(args.seq_len),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
        disable_resize=bool(args.disable_resize),
        disable_reverse=bool(args.disable_reverse),
        select_mode=str(args.select_mode),
        force_parent_entry=bool(args.force_parent_entry),
        force_entry_mult=float(args.force_entry_mult),
    )
    report = {
        "model_id": MODEL_ID,
        "design": "Exit Head feature-only + Mamba lifecycle baseline with only Parent fixed risk fields replaced by TabM finite risk-template selector. Parent final_action and quality threshold are preserved.",
        "threehead_dir": str(args.threehead_dir),
        "risk_selector": str(args.risk_selector),
        "risk_templates": sup_risk.RISK_TEMPLATES,
        "quality_threshold": float(args.quality_threshold),
        "state_columns": state_cols,
        "training": {
            "seq_len": int(args.seq_len),
            "max_train_entries": int(args.max_train_entries),
            "samples_per_entry": int(args.samples_per_entry),
            "train_max_sim_bars": int(args.train_max_sim_bars),
            "min_action_edge": float(args.min_action_edge),
            "disable_resize": bool(args.disable_resize),
            "disable_reverse": bool(args.disable_reverse),
            "class_balance_actor": bool(args.class_balance_actor),
            "select_mode": str(args.select_mode),
            "position_only_training": bool(args.position_only_training),
            "force_parent_entry": bool(args.force_parent_entry),
            "force_entry_mult": float(args.force_entry_mult),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "data_diag": data_diag,
            "train_diag": train_diag,
        },
        "cost_accounting": {"fee": fee, "slip": slip, "cost_mult": float(args.cost_mult), "delta_notional_resize_fee": True, "partial_exit_fee": True},
        "results": {"validation": val, "oos": oos},
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json"), "model": str(out_dir / "lifecycle_controller.pt")},
    }
    torch.save(
        {"model_state_dict": model.state_dict(), "normalizer": norm, "seq_len": int(args.seq_len), "state_columns": state_cols, "action_names": lifecycle.ACTION_NAMES},
        out_dir / "lifecycle_controller.pt",
    )
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
