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
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_stop_loss_hazard_veto_20260604 as ledger_hazard  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_candidate_hazard_veto_20260604"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _candidate_labels(frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> pd.DataFrame:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = np.asarray(omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    rows: list[dict[str, Any]] = []
    for i in np.flatnonzero(active):
        if i >= len(frame) - 2:
            continue
        row = dec.iloc[int(i)]
        side = int(row["side"])
        if side == 0:
            continue
        filled, entry_px, _entry_fee, _ = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        notional = float(row["notional_exposure"])
        take_profit = float(row["take_profit"])
        stop_loss = float(row["stop_loss"])
        event = "timeout"
        mfe = 0.0
        mae = 0.0
        event_i = len(frame) - 1
        for j in range(int(i) + 1, len(frame) - 1):
            px = float(arrays["close"][j])
            raw = (px * (1.0 - slip_eff) - entry_px) / max(entry_px, 1e-12) if side > 0 else (entry_px - px * (1.0 + slip_eff)) / max(entry_px, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            if unreal >= take_profit:
                event = "take_profit"
                event_i = int(j)
                break
            if unreal <= -abs(stop_loss):
                event = "stop_loss"
                event_i = int(j)
                break
        rows.append(
            {
                "signal_i": int(i),
                "timestamp": str(frame["timestamp"].iloc[int(i)]),
                "side": "LONG" if side > 0 else "SHORT",
                "event": event,
                "event_i": int(event_i),
                "bars_to_event": int(event_i - int(i)),
                "mfe_pct": float(mfe * 100.0),
                "mae_pct": float(mae * 100.0),
                "stop_loss_label": int(event == "stop_loss"),
            }
        )
    return pd.DataFrame(rows)


def _fit_model(x: pd.DataFrame, y: np.ndarray, *, seed: int) -> CatBoostClassifier:
    if len(np.unique(y)) < 2:
        raise RuntimeError("candidate hazard labels need both classes")
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=500,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=30.0,
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        auto_class_weights="Balanced",
    )
    model.fit(x, y)
    return model


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--slip", type=float, default=0.00015)
    ap.add_argument("--seed", type=int, default=260604)
    ap.add_argument("--out-suffix", default="candidate_hazard_20260604")
    args = ap.parse_args()

    out_dir = OUT_DIR.with_name(f"{OUT_DIR.name}_{args.out_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _device(args.device)
    frames = tabm._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    bundle = torch.load(ledger_hazard.BASE_DIR / "true_3head_tabm_bundle.pt", map_location=device, weights_only=False)
    train_x, train_pred = ledger_hazard._predict_frame(frames["train_raw"], bundle, oof=True, device=device)
    val_x = tabm._base_input(frames["val_raw"], list(bundle["base_cols"]))
    oos_x = tabm._base_input(frames["oos_raw"], list(bundle["base_cols"]))
    val_pred = ledger_hazard._read_predictions(ledger_hazard.BASE_DIR / "validation_predictions_2025_true3head.csv", frames["val_raw"])
    oos_pred = ledger_hazard._read_predictions(ledger_hazard.BASE_DIR / "oos_predictions_2026_true3head.csv", frames["oos_raw"])

    train_dec = ledger_hazard._decisions(train_pred, oof=True)
    val_dec = ledger_hazard._decisions(val_pred, oof=True)
    oos_dec = ledger_hazard._decisions(oos_pred, oof=False)

    train_labels = _candidate_labels(frames["train_raw"], train_dec, fee=fee, slip=slip, cost_mult=args.cost_mult)
    val_labels = _candidate_labels(frames["val_raw"], val_dec, fee=fee, slip=slip, cost_mult=args.cost_mult)
    oos_labels = _candidate_labels(frames["oos_raw"], oos_dec, fee=fee, slip=slip, cost_mult=args.cost_mult)

    train_feat_all = ledger_hazard._hazard_features(train_x, train_pred, oof=True)
    val_feat_all = ledger_hazard._hazard_features(val_x, val_pred, oof=True)
    oos_feat_all = ledger_hazard._hazard_features(oos_x, oos_pred, oof=False)

    train_idx = train_labels["signal_i"].to_numpy(dtype=np.int64)
    val_idx = val_labels["signal_i"].to_numpy(dtype=np.int64)
    oos_idx = oos_labels["signal_i"].to_numpy(dtype=np.int64)
    train_feat = train_feat_all.iloc[train_idx].reset_index(drop=True)
    val_feat = val_feat_all.iloc[val_idx].reset_index(drop=True)
    oos_feat = oos_feat_all.iloc[oos_idx].reset_index(drop=True)
    y_train = train_labels["stop_loss_label"].to_numpy(dtype=np.int64)
    y_val = val_labels["stop_loss_label"].to_numpy(dtype=np.int64)
    y_oos = oos_labels["stop_loss_label"].to_numpy(dtype=np.int64)

    model = _fit_model(train_feat, y_train, seed=args.seed)
    train_prob_entries = model.predict_proba(train_feat)[:, 1]
    val_prob_entries = model.predict_proba(val_feat)[:, 1]
    oos_prob_entries = model.predict_proba(oos_feat)[:, 1]
    train_labels["stop_loss_prob"] = train_prob_entries
    val_labels["stop_loss_prob"] = val_prob_entries
    oos_labels["stop_loss_prob"] = oos_prob_entries

    train_prob_all = model.predict_proba(train_feat_all)[:, 1]
    val_prob_all = model.predict_proba(val_feat_all)[:, 1]
    oos_prob_all = model.predict_proba(oos_feat_all)[:, 1]

    rows: list[dict[str, Any]] = []
    base_val = ledger_hazard._metrics(frames["val_raw"], val_dec, fee=fee, slip=slip, cost_mult=args.cost_mult)
    base_oos = ledger_hazard._metrics(frames["oos_raw"], oos_dec, fee=fee, slip=slip, cost_mult=args.cost_mult)
    rows.append(
        {
            "threshold": 2.0,
            "variant": "baseline_no_candidate_hazard",
            "val_pnl": base_val["pnl"],
            "val_mdd": base_val["mdd"],
            "val_wr": base_val["wr"],
            "val_trades": base_val["trades"],
            "oos_pnl": base_oos["pnl"],
            "oos_mdd": base_oos["mdd"],
            "oos_wr": base_oos["wr"],
            "oos_trades": base_oos["trades"],
            "val_exit_reasons": base_val.get("exit_reasons", {}),
            "oos_exit_reasons": base_oos.get("exit_reasons", {}),
        }
    )
    for thr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        val_h = ledger_hazard._decisions(val_pred, oof=True, hazard_prob=val_prob_all, hazard_threshold=thr)
        oos_h = ledger_hazard._decisions(oos_pred, oof=False, hazard_prob=oos_prob_all, hazard_threshold=thr)
        vm = ledger_hazard._metrics(frames["val_raw"], val_h, fee=fee, slip=slip, cost_mult=args.cost_mult)
        om = ledger_hazard._metrics(frames["oos_raw"], oos_h, fee=fee, slip=slip, cost_mult=args.cost_mult)
        rows.append(
            {
                "threshold": float(thr),
                "variant": "candidate_stop_loss_hazard_veto",
                "val_pnl": vm["pnl"],
                "val_mdd": vm["mdd"],
                "val_wr": vm["wr"],
                "val_trades": vm["trades"],
                "oos_pnl": om["pnl"],
                "oos_mdd": om["mdd"],
                "oos_wr": om["wr"],
                "oos_trades": om["trades"],
                "val_exit_reasons": vm.get("exit_reasons", {}),
                "oos_exit_reasons": om.get("exit_reasons", {}),
            }
        )

    ranking = pd.DataFrame(rows).sort_values(["val_pnl", "val_mdd"], ascending=[False, False]).reset_index(drop=True)
    ranking.to_csv(out_dir / "candidate_hazard_veto_grid.csv", index=False)
    train_labels.to_csv(out_dir / "train_candidate_labels_with_hazard.csv", index=False)
    val_labels.to_csv(out_dir / "validation_candidate_labels_with_hazard.csv", index=False)
    oos_labels.to_csv(out_dir / "oos_candidate_labels_with_hazard.csv", index=False)
    model.save_model(out_dir / "candidate_stop_loss_hazard_veto.cbm")
    report = {
        "design": "Candidate-level stop-loss hazard veto. Labels every active practical candidate by whether practical TP/SL path hits stop_loss before take_profit. Threshold is selected on validation only.",
        "baseline": {"validation": base_val, "oos": base_oos},
        "hazard_auc": {
            "train_candidates": _safe_auc(y_train, train_prob_entries),
            "validation_candidates": _safe_auc(y_val, val_prob_entries),
            "oos_candidates": _safe_auc(y_oos, oos_prob_entries),
        },
        "candidate_counts": {
            "train": int(len(train_labels)),
            "validation": int(len(val_labels)),
            "oos": int(len(oos_labels)),
            "train_stop_loss_rate": float(np.mean(y_train)),
            "validation_stop_loss_rate": float(np.mean(y_val)),
            "oos_stop_loss_rate": float(np.mean(y_oos)),
        },
        "ranking": rows,
        "artifacts": {
            "ranking": str(out_dir / "candidate_hazard_veto_grid.csv"),
            "train_candidates": str(out_dir / "train_candidate_labels_with_hazard.csv"),
            "validation_candidates": str(out_dir / "validation_candidate_labels_with_hazard.csv"),
            "oos_candidates": str(out_dir / "oos_candidate_labels_with_hazard.csv"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(ranking.to_string(index=False))
    print(json.dumps({"report": str(out_dir / "report.json"), "hazard_auc": report["hazard_auc"], "candidate_counts": report["candidate_counts"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
