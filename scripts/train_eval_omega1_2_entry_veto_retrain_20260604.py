#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as th  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega1_2_entry_veto_retrain_20260604"
BASE_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

THR_MAP = {"bull": 0.72, "bear": 0.64, "chop": 0.65}
SCALE_MAP = {"bull": 0.65, "bear": 0.90, "chop": 0.90}
BASE_SCALES = {"bull": 0.75, "bear": 0.90, "chop_expert": 0.90, "chop": 0.90}
TP = 0.026
SL = 0.014


@dataclass(frozen=True)
class VetoCfg:
    hidden: int = 160
    dropout: float = 0.08
    batch_size: int = 2048
    lr: float = 1.5e-3
    weight_decay: float = 2.0e-4
    patience: int = 8


CFG = VetoCfg()


class VetoMLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(n_features), CFG.hidden),
            nn.LayerNorm(CFG.hidden),
            nn.SiLU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.hidden, CFG.hidden),
            nn.LayerNorm(CFG.hidden),
            nn.SiLU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _fit_scaler(x: pd.DataFrame) -> dict[str, Any]:
    arr = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"columns": list(x.columns), "mean": mean, "std": std}


def _apply_scaler(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    cols = list(scaler["columns"])
    if list(x.columns) != cols:
        raise RuntimeError("entry veto feature contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = (arr - scaler["mean"]) / scaler["std"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite entry veto features")
    return out.astype(np.float32)


def _set_thresholds(src: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = src.copy()
    q = pd.to_numeric(out[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    action = pd.to_numeric(out[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    expert = out[f"{prefix}router_expert"].astype(str).to_numpy()
    thr = np.asarray([THR_MAP.get(x, THR_MAP["chop"]) for x in expert], dtype=np.float64)
    out[f"{prefix}quality_threshold"] = thr
    out[f"{prefix}final_action"] = np.where(q >= thr, action, omega.ACTION_CASH).astype(np.int64)
    return out


def _scale_dec(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy()
    active = omega._active(out)
    for expert, scale in SCALE_MAP.items():
        key = "chop_expert" if expert == "chop" else expert
        mask = active & out["router_expert"].astype(str).eq(key)
        ratio = float(scale) / float(BASE_SCALES[key])
        out.loc[mask, "notional_exposure"] = pd.to_numeric(out.loc[mask, "notional_exposure"], errors="raise") * ratio
        out.loc[mask, "position_fraction"] = pd.to_numeric(out.loc[mask, "position_fraction"], errors="raise") * ratio
    active = omega._active(out)
    out.loc[active, "take_profit"] = TP
    out.loc[active, "stop_loss"] = SL
    out.loc[active, "max_hold_bars"] = 0
    out.loc[active, "cooldown_bars"] = 0
    return out


def _build_dec(src: pd.DataFrame, prefix: str, *, oof: bool) -> pd.DataFrame:
    return _scale_dec(omega._to_fixed_decisions(_set_thresholds(src, prefix), oof=oof))


def _predict_3head(frames: dict[str, Any], frame: pd.DataFrame, models: dict[str, dict[str, Any]], *, device: torch.device, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = th._base_input(frame, list(frames["feature_cols"]))
    preds = {expert: th._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = th._routed(preds, route, "direction", 3)
    quality = th._routed(preds, route, "quality", 3)
    src = th._prediction_output(frame, direction, quality, threshold=0.50, prefix=prefix.rstrip("_"))
    return x, src


def _veto_features(base_x: pd.DataFrame, src: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [
        f"{prefix}router_confidence",
        f"{prefix}router_margin",
        f"{prefix}dir_p_cash",
        f"{prefix}dir_p_long",
        f"{prefix}dir_p_short",
        f"{prefix}dir_confidence",
        f"{prefix}dir_side_edge",
        f"{prefix}dir_trade_prob",
        f"{prefix}quality_p_cash",
        f"{prefix}quality_p_long",
        f"{prefix}quality_p_short",
        f"{prefix}quality_for_action",
    ]
    missing = [c for c in cols if c not in src.columns]
    if missing:
        raise RuntimeError(f"missing veto source columns: {missing}")
    signal = src[cols].reset_index(drop=True).copy()
    signal.columns = [f"veto_signal_{c.removeprefix(prefix)}" for c in signal.columns]
    out = pd.concat([base_x.reset_index(drop=True), signal], axis=1)
    if out.columns.duplicated().any():
        raise RuntimeError("duplicate veto feature columns")
    return out.astype(np.float32)


def _build_labels(frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> np.ndarray:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    labels = np.zeros(len(dec), dtype=np.int64)
    for idx in np.flatnonzero(omega._active(dec) & (np.arange(len(dec)) < len(dec) - 3)):
        _score, meta = omega._simulate_trade(frame, arrays, int(idx), dec.iloc[int(idx)], fee=fee, slip=slip, cost_mult=cost_mult)
        # 1 means veto this entry. Penalize losing/SL trades, but keep winners and neutral misses.
        labels[int(idx)] = int(float(meta.get("net", 0.0)) <= 0.0 or str(meta.get("exit_reason", "")) == "stop_loss")
    return labels


def _train_one(x: pd.DataFrame, y: np.ndarray, w: np.ndarray, *, device: torch.device, seed: int) -> tuple[VetoMLP, dict[str, Any], dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    scaler = _fit_scaler(x)
    x_np = _apply_scaler(x, scaler)
    y_np = np.asarray(y, dtype=np.int64)
    w_np = np.asarray(w, dtype=np.float32)
    n = len(y_np)
    split = max(int(n * 0.85), min(n - 1, 256))
    tr = np.arange(split)
    va = np.arange(split, n)
    model = VetoMLP(x_np.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    ds = TensorDataset(torch.from_numpy(x_np[tr]), torch.from_numpy(y_np[tr]), torch.from_numpy(w_np[tr]))
    dl = DataLoader(ds, batch_size=CFG.batch_size, shuffle=True, drop_last=False)
    best_state = None
    best = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(80):
        last_epoch = epoch + 1
        model.train()
        for xb, yb, wb in dl:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            loss_i = torch.nn.functional.cross_entropy(model(xb), yb, reduction="none")
            loss = (loss_i * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_np[va]).to(device)
            vy = torch.from_numpy(y_np[va]).to(device)
            vw = torch.from_numpy(w_np[va]).to(device)
            vloss_i = torch.nn.functional.cross_entropy(model(vx), vy, reduction="none")
            vloss = float(((vloss_i * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
        if vloss + 1e-6 < best:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= CFG.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    diag = {"best_validation_loss": float(best), "epochs_ran": int(last_epoch), "positive_rate": float(np.mean(y_np))}
    return model.cpu(), scaler, diag


@torch.no_grad()
def _predict_veto(model: VetoMLP, scaler: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> np.ndarray:
    model = model.to(device)
    model.eval()
    x_np = _apply_scaler(x, scaler)
    out = []
    for start in range(0, len(x_np), 8192):
        prob = torch.softmax(model(torch.from_numpy(x_np[start : start + 8192]).to(device)), dim=1)[:, 1]
        out.append(prob.detach().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def _apply_veto(dec: pd.DataFrame, src: pd.DataFrame, prefix: str, probs: np.ndarray, threshold: float) -> pd.DataFrame:
    out = dec.copy()
    active = omega._active(out)
    veto = active & (probs >= float(threshold))
    out.loc[veto, "action"] = omega.ACTION_CASH
    out.loc[veto, "side"] = 0
    out.loc[veto, "notional_exposure"] = 0.0
    out.loc[veto, "position_fraction"] = 0.0
    out["entry_veto_prob"] = probs
    out["entry_veto_threshold"] = float(threshold)
    out["entry_vetoed"] = veto.astype(np.int64)
    return out


def _monthly_min(frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> float:
    ts = pd.to_datetime(frame["timestamp"], errors="raise")
    vals = []
    for _per, idxs in ts.dt.to_period("M").groupby(ts.dt.to_period("M")).groups.items():
        idx = np.asarray(list(idxs), dtype=np.int64)
        vals.append(omega._metrics(frame.iloc[idx].reset_index(drop=True), dec.iloc[idx].reset_index(drop=True), fee=fee, slip=slip, cost_mult=cost_mult)["pnl"])
    return float(min(vals)) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=260604)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    device = _device(args.device)
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{args.out_suffix.strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = th._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    bundle = torch.load(BASE_DIR / "true_3head_tabm_bundle.pt", map_location="cpu", weights_only=False)
    models = bundle["models"]
    train_x, train_src = _predict_3head(frames, frames["train_raw"], models, device=device, prefix="omega1_regime3_expertdq_oof_")
    val_x, val_src = _predict_3head(frames, frames["val_raw"], models, device=device, prefix="omega1_regime3_expertdq_oof_")
    oos_x, oos_src_oof = _predict_3head(frames, frames["oos_raw"], models, device=device, prefix="omega1_regime3_expertdq_oof_")
    oos_src = oos_src_oof.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in oos_src_oof.columns})
    train_dec = _build_dec(train_src, "omega1_regime3_expertdq_oof_", oof=True)
    val_dec = _build_dec(val_src, "omega1_regime3_expertdq_oof_", oof=True)
    oos_dec = _build_dec(oos_src, "omega1_regime3_expertdq_", oof=False)
    train_labels = _build_labels(frames["train_raw"], train_dec, fee=fee, slip=slip, cost_mult=3.0)
    train_feat = _veto_features(train_x, train_src, "omega1_regime3_expertdq_oof_")
    val_feat = _veto_features(val_x, val_src, "omega1_regime3_expertdq_oof_")
    oos_feat = _veto_features(oos_x, oos_src, "omega1_regime3_expertdq_")
    payloads: dict[str, Any] = {}
    val_probs = np.zeros(len(val_dec), dtype=np.float64)
    oos_probs = np.zeros(len(oos_dec), dtype=np.float64)
    train_expert = train_src["omega1_regime3_expertdq_oof_router_expert"].astype(str).to_numpy()
    val_expert = val_src["omega1_regime3_expertdq_oof_router_expert"].astype(str).to_numpy()
    oos_expert = oos_src["omega1_regime3_expertdq_router_expert"].astype(str).to_numpy()
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        mask = train_expert == expert
        if int(mask.sum()) < 100:
            raise RuntimeError(f"too few veto training rows for {expert}")
        y = train_labels[mask]
        w = compute_sample_weight(class_weight="balanced", y=y).astype(np.float32)
        model, scaler, diag = _train_one(train_feat.loc[mask].reset_index(drop=True), y, w, device=device, seed=int(args.seed) + idx)
        payloads[expert] = {"state_dict": model.state_dict(), "scaler": scaler, "diag": diag, "n_features": train_feat.shape[1], "columns": list(train_feat.columns)}
        vm = val_expert == expert
        om = oos_expert == expert
        val_probs[vm] = _predict_veto(model, scaler, val_feat.loc[vm].reset_index(drop=True), device=device)
        oos_probs[om] = _predict_veto(model, scaler, oos_feat.loc[om].reset_index(drop=True), device=device)
    rows: list[dict[str, Any]] = []
    base_val = omega._metrics(frames["val_raw"], val_dec, fee=fee, slip=slip, cost_mult=3.0)
    base_oos = omega._metrics(frames["oos_raw"], oos_dec, fee=fee, slip=slip, cost_mult=3.0)
    rows.append({"threshold": -1.0, "variant": "baseline", "val_pnl": base_val["pnl"], "val_mdd": base_val["mdd"], "val_wr": base_val["wr"], "val_trades": base_val["trades"], "val_min_month_pnl": _monthly_min(frames["val_raw"], val_dec, fee=fee, slip=slip, cost_mult=3.0), "oos_pnl": base_oos["pnl"], "oos_mdd": base_oos["mdd"], "oos_wr": base_oos["wr"], "oos_trades": base_oos["trades"], "oos_min_month_pnl": _monthly_min(frames["oos_raw"], oos_dec, fee=fee, slip=slip, cost_mult=3.0)})
    for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        vd = _apply_veto(val_dec, val_src, "omega1_regime3_expertdq_oof_", val_probs, thr)
        od = _apply_veto(oos_dec, oos_src, "omega1_regime3_expertdq_", oos_probs, thr)
        val = omega._metrics(frames["val_raw"], vd, fee=fee, slip=slip, cost_mult=3.0)
        oos = omega._metrics(frames["oos_raw"], od, fee=fee, slip=slip, cost_mult=3.0)
        rows.append({"threshold": thr, "variant": f"entry_veto_retrained_{thr:.2f}", "val_pnl": val["pnl"], "val_mdd": val["mdd"], "val_wr": val["wr"], "val_trades": val["trades"], "val_min_month_pnl": _monthly_min(frames["val_raw"], vd, fee=fee, slip=slip, cost_mult=3.0), "oos_pnl": oos["pnl"], "oos_mdd": oos["mdd"], "oos_wr": oos["wr"], "oos_trades": oos["trades"], "oos_min_month_pnl": _monthly_min(frames["oos_raw"], od, fee=fee, slip=slip, cost_mult=3.0), "val_vetoed": int(vd["entry_vetoed"].sum()), "oos_vetoed": int(od["entry_vetoed"].sum())})
    ranking = pd.DataFrame(rows).sort_values(["val_pnl", "val_min_month_pnl"], ascending=False)
    ranking.to_csv(out_dir / "ranking.csv", index=False)
    torch.save({"models": payloads, "config": CFG.__dict__, "base_model_dir": str(BASE_DIR)}, out_dir / "entry_veto_models.pt")
    report = {"model_id": MODEL_ID, "design": "Retrained per-expert entry-veto MLP on top of frozen Omega1.2 True 3-head TabM D/Q outputs. Veto labels are generated from Cost3 fixed-template trade simulation; no OOS labels are used for training.", "threshold_map": THR_MAP, "scale_map": SCALE_MAP, "tp": TP, "sl": SL, "summaries": {k: v["diag"] for k, v in payloads.items()}, "ranking": rows, "artifacts": {"out_dir": str(out_dir), "ranking": str(out_dir / "ranking.csv")}}
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(ranking.to_string(index=False))
    print(json.dumps({"report": str(out_dir / "report.json"), "ranking": str(out_dir / "ranking.csv")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
