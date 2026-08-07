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

import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega1_2_stop_loss_hazard_veto_20260604"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASE_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"
PRACTICAL = {
    "thr_bull": 0.72,
    "thr_bear": 0.64,
    "thr_chop": 0.65,
    "scale_bull": 0.65,
    "scale_bear": 0.90,
    "scale_chop": 0.90,
    "tp": 0.026,
    "sl": 0.012,
}
PRED_FEATURES = [
    "router_confidence",
    "router_margin",
    "dir_p_cash",
    "dir_p_long",
    "dir_p_short",
    "dir_confidence",
    "dir_side_edge",
    "dir_trade_prob",
    "dir_action",
    "quality_p_cash",
    "quality_p_long",
    "quality_p_short",
    "quality_for_action",
]


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _prefix(oof: bool) -> str:
    return "omega1_regime3_expertdq_oof_" if oof else "omega1_regime3_expertdq_"


def _read_predictions(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if not pred["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError(f"timestamp contract mismatch: {path}")
    return pred


@torch.no_grad()
def _predict_frame(frame: pd.DataFrame, bundle: dict[str, Any], *, oof: bool, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cols = list(bundle["base_cols"])
    x = tabm._base_input(frame, base_cols)
    preds = {expert: tabm._predict_payload(bundle["models"][expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = tabm._routed(preds, route, "direction", 3)
    quality = tabm._routed(preds, route, "quality", 3)
    src = tabm._prediction_output(frame, direction, quality, threshold=0.50, prefix=_prefix(oof).rstrip("_"))
    return x, src


def _decisions(pred: pd.DataFrame, *, oof: bool, hazard_prob: np.ndarray | None = None, hazard_threshold: float = 2.0) -> pd.DataFrame:
    prefix = _prefix(oof)
    action = pd.to_numeric(pred[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    quality = pd.to_numeric(pred[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    expert = pred[f"{prefix}router_expert"].astype(str).replace({"chop_expert": "chop"}).to_numpy()
    thresholds = {"bull": PRACTICAL["thr_bull"], "bear": PRACTICAL["thr_bear"], "chop": PRACTICAL["thr_chop"]}
    scales = {"bull": PRACTICAL["scale_bull"], "bear": PRACTICAL["scale_bear"], "chop": PRACTICAL["scale_chop"]}
    thr = np.array([float(thresholds[str(x)]) for x in expert], dtype=np.float64)
    scale = np.array([float(scales[str(x)]) for x in expert], dtype=np.float64)
    passed = quality >= thr
    if hazard_prob is not None:
        if len(hazard_prob) != len(pred):
            raise RuntimeError("hazard probability length mismatch")
        passed &= np.asarray(hazard_prob, dtype=np.float64) < float(hazard_threshold)
    final = np.where(passed, action, omega.ACTION_CASH).astype(np.int64)
    active = final != omega.ACTION_CASH
    side = np.where(final == omega.ACTION_LONG, 1, np.where(final == omega.ACTION_SHORT, -1, 0)).astype(np.int64)
    notional = np.where(active, 0.45 * scale, 0.0)
    return pd.DataFrame(
        {
            "action": final,
            "side": side,
            "notional_exposure": notional,
            "leverage": np.where(active, 2.0, 1.0),
            "position_fraction": notional,
            "take_profit": np.where(active, PRACTICAL["tp"], 0.0),
            "stop_loss": np.where(active, PRACTICAL["sl"], 0.0),
            "max_hold_bars": 0,
            "cooldown_bars": 0,
            "quality_score": quality,
            "confidence": pd.to_numeric(pred[f"{prefix}dir_confidence"], errors="raise").to_numpy(dtype=np.float64),
            "router_expert": np.where(expert == "chop", "chop_expert", expert),
        }
    )


def _hazard_features(base_x: pd.DataFrame, pred: pd.DataFrame, *, oof: bool, action_side: np.ndarray | None = None) -> pd.DataFrame:
    prefix = _prefix(oof)
    out = base_x.copy().reset_index(drop=True)
    for col in PRED_FEATURES:
        out[f"pred_{col}"] = pd.to_numeric(pred[f"{prefix}{col}"], errors="raise").to_numpy(dtype=np.float32)
    expert = pred[f"{prefix}router_expert"].astype(str).replace({"chop": "chop_expert"})
    for name in ("bull", "bear", "chop_expert"):
        out[f"router_is_{name}"] = expert.eq(name).astype(np.float32).to_numpy()
    if action_side is None:
        action = pd.to_numeric(pred[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
        action_side = np.where(action == omega.ACTION_LONG, 1.0, np.where(action == omega.ACTION_SHORT, -1.0, 0.0))
    out["candidate_side"] = np.asarray(action_side, dtype=np.float32)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _metrics(frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> dict[str, Any]:
    return omega._metrics(frame, dec, fee=fee, slip=slip, cost_mult=cost_mult)


def _trade_ledger(frame: pd.DataFrame, dec: pd.DataFrame, pred: pd.DataFrame, *, oof: bool, fee: float, slip: float, cost_mult: float) -> pd.DataFrame:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    rows: list[dict[str, Any]] = []
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_signal_i = 0
    entry_i = 0
    notional = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    cash = 1.0
    mfe = 0.0
    mae = 0.0
    prefix = _prefix(oof)
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            reason = ""
            if unreal >= take_profit:
                reason = "take_profit"
            elif unreal <= -abs(stop_loss):
                reason = "stop_loss"
            if reason:
                filled, exit_px, exit_fee, _ = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                rows.append(
                    {
                        "entry_signal_i": int(entry_signal_i),
                        "entry_i": int(entry_i),
                        "exit_i": int(i),
                        "entry_time": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                        "exit_time": str(frame["timestamp"].iloc[int(i)]),
                        "side": "LONG" if pos > 0 else "SHORT",
                        "exit_reason": reason,
                        "net_trade_return_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0),
                        "mfe_pct": float(mfe * 100.0),
                        "mae_pct": float(mae * 100.0),
                        "quality": float(pred[f"{prefix}quality_for_action"].iloc[int(entry_signal_i)]),
                        "dir_conf": float(pred[f"{prefix}dir_confidence"].iloc[int(entry_signal_i)]),
                        "router_conf": float(pred[f"{prefix}router_confidence"].iloc[int(entry_signal_i)]),
                        "router_margin": float(pred[f"{prefix}router_margin"].iloc[int(entry_signal_i)]),
                        "dir_side_edge": float(pred[f"{prefix}dir_side_edge"].iloc[int(entry_signal_i)]),
                        "router_expert": str(pred[f"{prefix}router_expert"].iloc[int(entry_signal_i)]),
                    }
                )
                pos = 0
                continue
        if pos != 0 or not bool(active[i]):
            continue
        row = dec.iloc[int(i)]
        side = int(row["side"])
        if side == 0:
            continue
        filled, px, entry_fee, _ = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        pos = side
        entry_signal_i = int(i)
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_price = float(px)
        entry_equity = cash
        notional = float(row["notional_exposure"])
        take_profit = float(row["take_profit"])
        stop_loss = float(row["stop_loss"])
        cash -= cash * entry_fee * notional
        mfe = 0.0
        mae = 0.0
    if pos != 0:
        exit_px = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        rows.append({"entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(len(frame) - 1), "entry_time": str(frame["timestamp"].iloc[int(entry_signal_i)]), "exit_time": str(frame["timestamp"].iloc[-1]), "side": "LONG" if pos > 0 else "SHORT", "exit_reason": "forced_end", "net_trade_return_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "mfe_pct": float(mfe * 100.0), "mae_pct": float(mae * 100.0)})
    return pd.DataFrame(rows)


def _fit_hazard(x: pd.DataFrame, y: np.ndarray, *, seed: int) -> CatBoostClassifier:
    if len(np.unique(y)) < 2:
        raise RuntimeError("hazard labels need both classes")
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        depth=3,
        learning_rate=0.035,
        l2_leaf_reg=20.0,
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
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
    ap.add_argument("--seed", type=int, default=260604)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    device = _device(args.device)
    out_dir = OUT_DIR if not args.out_suffix.strip() else OUT_DIR.parent / f"{MODEL_ID}_{args.out_suffix.strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = tabm._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    bundle = torch.load(BASE_DIR / "true_3head_tabm_bundle.pt", map_location=device, weights_only=False)
    train_x, train_pred = _predict_frame(frames["train_raw"], bundle, oof=True, device=device)
    val_x = tabm._base_input(frames["val_raw"], list(bundle["base_cols"]))
    oos_x = tabm._base_input(frames["oos_raw"], list(bundle["base_cols"]))
    val_pred = _read_predictions(BASE_DIR / "validation_predictions_2025_true3head.csv", frames["val_raw"])
    oos_pred = _read_predictions(BASE_DIR / "oos_predictions_2026_true3head.csv", frames["oos_raw"])

    train_dec = _decisions(train_pred, oof=True)
    val_dec = _decisions(val_pred, oof=True)
    oos_dec = _decisions(oos_pred, oof=False)
    train_ledger = _trade_ledger(frames["train_raw"], train_dec, train_pred, oof=True, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    val_ledger = _trade_ledger(frames["val_raw"], val_dec, val_pred, oof=True, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    oos_ledger = _trade_ledger(frames["oos_raw"], oos_dec, oos_pred, oof=False, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    train_ledger.to_csv(out_dir / "train_trade_ledger.csv", index=False)
    val_ledger.to_csv(out_dir / "validation_trade_ledger.csv", index=False)
    oos_ledger.to_csv(out_dir / "oos_trade_ledger.csv", index=False)

    train_hazard_x_all = _hazard_features(train_x, train_pred, oof=True)
    val_hazard_x_all = _hazard_features(val_x, val_pred, oof=True)
    oos_hazard_x_all = _hazard_features(oos_x, oos_pred, oof=False)
    train_entry_idx = pd.to_numeric(train_ledger["entry_signal_i"], errors="raise").to_numpy(dtype=np.int64)
    val_entry_idx = pd.to_numeric(val_ledger["entry_signal_i"], errors="raise").to_numpy(dtype=np.int64) if len(val_ledger) else np.array([], dtype=np.int64)
    oos_entry_idx = pd.to_numeric(oos_ledger["entry_signal_i"], errors="raise").to_numpy(dtype=np.int64) if len(oos_ledger) else np.array([], dtype=np.int64)
    y_train = train_ledger["exit_reason"].astype(str).eq("stop_loss").astype(np.int64).to_numpy()
    y_val = val_ledger["exit_reason"].astype(str).eq("stop_loss").astype(np.int64).to_numpy() if len(val_ledger) else np.array([], dtype=np.int64)
    y_oos = oos_ledger["exit_reason"].astype(str).eq("stop_loss").astype(np.int64).to_numpy() if len(oos_ledger) else np.array([], dtype=np.int64)
    model = _fit_hazard(train_hazard_x_all.iloc[train_entry_idx].reset_index(drop=True), y_train, seed=int(args.seed))
    model.save_model(str(out_dir / "stop_loss_hazard_veto.cbm"))
    p_train_entries = model.predict_proba(train_hazard_x_all.iloc[train_entry_idx].reset_index(drop=True))[:, 1]
    p_val_entries = model.predict_proba(val_hazard_x_all.iloc[val_entry_idx].reset_index(drop=True))[:, 1] if len(val_entry_idx) else np.array([])
    p_oos_entries = model.predict_proba(oos_hazard_x_all.iloc[oos_entry_idx].reset_index(drop=True))[:, 1] if len(oos_entry_idx) else np.array([])

    val_all_prob = model.predict_proba(val_hazard_x_all)[:, 1]
    oos_all_prob = model.predict_proba(oos_hazard_x_all)[:, 1]
    rows: list[dict[str, Any]] = []
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    base_val = _metrics(frames["val_raw"], val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    base_oos = _metrics(frames["oos_raw"], oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    rows.append({"threshold": 2.0, "variant": "baseline_no_hazard_veto", "val_pnl": base_val["pnl"], "val_mdd": base_val["mdd"], "val_wr": base_val["wr"], "val_trades": base_val["trades"], "oos_pnl": base_oos["pnl"], "oos_mdd": base_oos["mdd"], "oos_wr": base_oos["wr"], "oos_trades": base_oos["trades"], "val_exit_reasons": base_val.get("exit_reasons", {}), "oos_exit_reasons": base_oos.get("exit_reasons", {})})
    for thr in thresholds:
        vd = _decisions(val_pred, oof=True, hazard_prob=val_all_prob, hazard_threshold=float(thr))
        od = _decisions(oos_pred, oof=False, hazard_prob=oos_all_prob, hazard_threshold=float(thr))
        val = _metrics(frames["val_raw"], vd, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        oos = _metrics(frames["oos_raw"], od, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        rows.append({"threshold": float(thr), "variant": "stop_loss_hazard_veto", "val_pnl": val["pnl"], "val_mdd": val["mdd"], "val_wr": val["wr"], "val_trades": val["trades"], "oos_pnl": oos["pnl"], "oos_mdd": oos["mdd"], "oos_wr": oos["wr"], "oos_trades": oos["trades"], "val_exit_reasons": val.get("exit_reasons", {}), "oos_exit_reasons": oos.get("exit_reasons", {})})
    ranking = pd.DataFrame(rows).sort_values(["val_pnl", "val_wr"], ascending=False)
    ranking.to_csv(out_dir / "hazard_veto_grid.csv", index=False)
    train_ledger.assign(stop_loss_prob=p_train_entries).to_csv(out_dir / "train_trade_ledger_with_hazard.csv", index=False)
    val_ledger.assign(stop_loss_prob=p_val_entries).to_csv(out_dir / "validation_trade_ledger_with_hazard.csv", index=False)
    oos_ledger.assign(stop_loss_prob=p_oos_entries).to_csv(out_dir / "oos_trade_ledger_with_hazard.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Stop-loss-only hazard veto for Omega1.2 practical sniper baseline. Direction/Quality and risk template are fixed; model learns whether an accepted entry becomes stop_loss.",
        "ledger_counts": {
            "train": int(len(train_ledger)),
            "validation": int(len(val_ledger)),
            "oos": int(len(oos_ledger)),
            "train_stop_loss_rate": float(y_train.mean()) if len(y_train) else 0.0,
            "validation_stop_loss_rate": float(y_val.mean()) if len(y_val) else 0.0,
            "oos_stop_loss_rate": float(y_oos.mean()) if len(y_oos) else 0.0,
        },
        "hazard_auc": {
            "train_entries": _safe_auc(y_train, p_train_entries),
            "validation_entries": _safe_auc(y_val, p_val_entries),
            "oos_entries": _safe_auc(y_oos, p_oos_entries),
        },
        "ranking": rows,
        "artifacts": {
            "out_dir": str(out_dir),
            "ranking": str(out_dir / "hazard_veto_grid.csv"),
            "model": str(out_dir / "stop_loss_hazard_veto.cbm"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(ranking.to_string(index=False))
    print(json.dumps({"report": str(out_dir / "report.json"), "ranking": str(out_dir / "hazard_veto_grid.csv"), "hazard_auc": report["hazard_auc"], "ledger_counts": report["ledger_counts"]}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
