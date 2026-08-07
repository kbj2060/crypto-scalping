#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_eval_omega4_5head_margin_leverage_20260622 as risk5  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega4_frozen_risk_heads_margin_leverage_20260622"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASELINE_BUNDLE = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_3head_parent72_loose_entry_quality_20260620_smoke_loose_entry_loose_quality_terminal_giveback_exit_e2_train15k_exit15k_q070"
    / "true_3head_tabm_bundle.pt"
)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class FrozenEncoderRiskHeads(nn.Module):
    def __init__(self, baseline_payload: dict[str, Any]) -> None:
        super().__init__()
        self.base = parent.ThreeHeadTabM(int(baseline_payload["n_features"]), cfg=parent.CFG)
        self.base.load_state_dict(baseline_payload["state_dict"])
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.margin_fraction_head = nn.Linear(int(parent.CFG.hidden), len(risk5.MARGIN_BUCKETS))
        self.leverage_head = nn.Linear(int(parent.CFG.hidden), len(risk5.LEVERAGE_BUCKETS))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.base.eval()
        with torch.no_grad():
            h = self.base.encode(x)
        return {
            "margin_fraction": self.margin_fraction_head(h),
            "leverage": self.leverage_head(h),
        }


def _ce_tabm(logits: torch.Tensor, target: torch.Tensor, classes: int) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, int(classes)),
        target[:, None].expand(-1, int(parent.CFG.k)).reshape(-1),
        reduction="none",
    ).reshape(-1, int(parent.CFG.k)).mean(dim=1)


def _fit_risk_heads(
    baseline_payload: dict[str, Any],
    x: pd.DataFrame,
    y_margin: np.ndarray,
    y_leverage: np.ndarray,
    risk_active: np.ndarray,
    route_frame: pd.DataFrame,
    *,
    expert_idx: int,
    seed: int,
    epochs: int,
    device: torch.device,
    model_path: Path,
) -> dict[str, Any]:
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    x_np = parent._standardize_apply(x, baseline_payload["scaler"])
    y_margin_np = np.asarray(y_margin, dtype=np.int64)
    y_leverage_np = np.asarray(y_leverage, dtype=np.int64)
    route_w = parent._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    margin_w = risk5._risk_weights(y_margin_np, risk_active, route_w)
    leverage_w = risk5._risk_weights(y_leverage_np, risk_active, route_w)
    if min(float(margin_w.sum()), float(leverage_w.sum())) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid risk-only sample weights")
    n = len(y_margin_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    ds = TensorDataset(
        torch.from_numpy(x_np[train_idx]),
        torch.from_numpy(y_margin_np[train_idx]),
        torch.from_numpy(y_leverage_np[train_idx]),
        torch.from_numpy(margin_w[train_idx]),
        torch.from_numpy(leverage_w[train_idx]),
    )
    loader = DataLoader(ds, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
    model = FrozenEncoderRiskHeads(baseline_payload).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(parent.CFG.lr), weight_decay=float(parent.CFG.weight_decay))
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        model.base.eval()
        for xb, ym, yl, wm, wl in loader:
            xb, ym, yl, wm, wl = xb.to(device), ym.to(device), yl.to(device), wm.to(device), wl.to(device)
            out = model(xb)
            loss_margin = (_ce_tabm(out["margin_fraction"], ym, len(risk5.MARGIN_BUCKETS)) * wm).sum() / torch.clamp(wm.sum(), min=1.0)
            loss_leverage = (_ce_tabm(out["leverage"], yl, len(risk5.LEVERAGE_BUCKETS)) * wl).sum() / torch.clamp(wl.sum(), min=1.0)
            loss = loss_margin + loss_leverage
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_np[val_idx]).to(device)
            vm = torch.from_numpy(y_margin_np[val_idx]).to(device)
            vl = torch.from_numpy(y_leverage_np[val_idx]).to(device)
            vwm = torch.from_numpy(margin_w[val_idx]).to(device)
            vwl = torch.from_numpy(leverage_w[val_idx]).to(device)
            vo = model(vx)
            vloss = (
                (_ce_tabm(vo["margin_fraction"], vm, len(risk5.MARGIN_BUCKETS)) * vwm).sum() / torch.clamp(vwm.sum(), min=1.0)
                + (_ce_tabm(vo["leverage"], vl, len(risk5.LEVERAGE_BUCKETS)) * vwl).sum() / torch.clamp(vwl.sum(), min=1.0)
            )
            val_loss = float(vloss.detach().cpu())
        if val_loss + 1.0e-6 < best_loss:
            best_loss = val_loss
            best_state = {
                "margin_fraction_head.weight": model.margin_fraction_head.weight.detach().cpu().clone(),
                "margin_fraction_head.bias": model.margin_fraction_head.bias.detach().cpu().clone(),
                "leverage_head.weight": model.leverage_head.weight.detach().cpu().clone(),
                "leverage_head.bias": model.leverage_head.bias.detach().cpu().clone(),
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(parent.CFG.patience):
                break
    if best_state is not None:
        model.margin_fraction_head.weight.data.copy_(best_state["margin_fraction_head.weight"].to(device))
        model.margin_fraction_head.bias.data.copy_(best_state["margin_fraction_head.bias"].to(device))
        model.leverage_head.weight.data.copy_(best_state["leverage_head.weight"].to(device))
        model.leverage_head.bias.data.copy_(best_state["leverage_head.bias"].to(device))
    payload = {
        "model_id": MODEL_ID,
        "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "baseline_n_features": int(baseline_payload["n_features"]),
        "risk_state_dict": {
            "margin_fraction_head.weight": model.margin_fraction_head.weight.detach().cpu(),
            "margin_fraction_head.bias": model.margin_fraction_head.bias.detach().cpu(),
            "leverage_head.weight": model.leverage_head.weight.detach().cpu(),
            "leverage_head.bias": model.leverage_head.bias.detach().cpu(),
        },
        "best_validation_loss": float(best_loss),
        "epochs_ran": int(last_epoch),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, model_path)
    return payload


@torch.no_grad()
def _predict_risk_payload(
    baseline_payload: dict[str, Any],
    risk_payload: dict[str, Any],
    x: pd.DataFrame,
    *,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model = FrozenEncoderRiskHeads(baseline_payload).to(device)
    model.margin_fraction_head.weight.data.copy_(risk_payload["risk_state_dict"]["margin_fraction_head.weight"].to(device))
    model.margin_fraction_head.bias.data.copy_(risk_payload["risk_state_dict"]["margin_fraction_head.bias"].to(device))
    model.leverage_head.weight.data.copy_(risk_payload["risk_state_dict"]["leverage_head.weight"].to(device))
    model.leverage_head.bias.data.copy_(risk_payload["risk_state_dict"]["leverage_head.bias"].to(device))
    model.eval()
    x_np = parent._standardize_apply(x, baseline_payload["scaler"])
    chunks: dict[str, list[np.ndarray]] = {"margin_fraction": [], "leverage": []}
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
        out = model(xb)
        chunks["margin_fraction"].append(torch.softmax(out["margin_fraction"], dim=-1).mean(dim=1).detach().cpu().numpy())
        chunks["leverage"].append(torch.softmax(out["leverage"], dim=-1).mean(dim=1).detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in chunks.items()}


def _apply_margin_cap(dec: pd.DataFrame, cap: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = omega._active(out)
    if not bool(active.any()):
        return out
    margin = pd.to_numeric(out.loc[active, "margin_fraction"], errors="raise").to_numpy(dtype=np.float64)
    leverage = pd.to_numeric(out.loc[active, "leverage"], errors="raise").to_numpy(dtype=np.float64)
    capped_margin = np.minimum(margin, float(cap))
    notional = capped_margin * leverage
    tp_price_move = float(omega.BASE_TEMPLATE["take_profit"])
    sl_price_move = float(omega.BASE_TEMPLATE["stop_loss"])
    out.loc[active, "margin_fraction"] = capped_margin
    out.loc[active, "notional_exposure"] = notional
    out.loc[active, "position_fraction"] = capped_margin
    out.loc[active, "take_profit"] = tp_price_move * notional
    out.loc[active, "stop_loss"] = sl_price_move * notional
    return out


def _load_baseline_bundle(bundle_path: Path, *, device: torch.device) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payload = torch.load(bundle_path, map_location=device, weights_only=False)
    return payload["models"], list(payload["base_cols"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-bundle", type=Path, default=BASELINE_BUNDLE)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-train-rows", type=int, default=15000)
    ap.add_argument("--quality-threshold", type=float, default=0.70)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260622)
    ap.add_argument("--out-suffix", default="e8_train15k_q070")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = parent._device(str(args.device))
    out_dir = OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("stage=load_baseline_bundle", flush=True)
    baseline_models, base_cols = _load_baseline_bundle(Path(args.baseline_bundle), device=device)
    print("stage=prepare_frames", flush=True)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=omega4.LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()
    train_raw = frames["train_raw"]
    if int(args.max_train_rows) > 0:
        train_fit_frame = train_raw.iloc[: int(args.max_train_rows)].reset_index(drop=True)
    else:
        train_fit_frame = train_raw.reset_index(drop=True)
    x_train = parent._base_input(train_fit_frame, base_cols)
    print(f"stage=build_risk_labels rows={len(train_fit_frame)}", flush=True)
    y_margin, y_leverage, risk_active, risk_label_diag = risk5._build_margin_leverage_labels(train_fit_frame, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    risk_models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        print(f"stage=train_risk_head expert={expert}", flush=True)
        payload = _fit_risk_heads(
            baseline_models[expert],
            x_train,
            y_margin,
            y_leverage,
            risk_active,
            train_fit_frame,
            expert_idx=idx,
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=device,
            model_path=out_dir / "models" / f"{expert}_frozen_margin_leverage_heads.pt",
        )
        risk_models[expert] = payload
        summaries[expert] = {"best_validation_loss": payload["best_validation_loss"], "epochs_ran": payload["epochs_ran"]}

    def predict(frame: pd.DataFrame, *, oof: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        x = parent._base_input(frame, base_cols)
        baseline_preds = {expert: parent._predict_payload(baseline_models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        risk_preds = {expert: _predict_risk_payload(baseline_models[expert], risk_models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = parent._routed(baseline_preds, route, "direction", 3)
        quality = parent._routed(baseline_preds, route, "quality", 3)
        margin = parent._routed(risk_preds, route, "margin_fraction", len(risk5.MARGIN_BUCKETS))
        leverage = parent._routed(risk_preds, route, "leverage", len(risk5.LEVERAGE_BUCKETS))
        prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
        src = parent._prediction_output(frame, direction, quality, threshold=float(args.quality_threshold), prefix=prefix)
        fixed_dec = parent._to_decisions(src, oof=oof)
        learned_dec = risk5._apply_learned_risk(fixed_dec, margin, leverage)
        capped_dec = _apply_margin_cap(learned_dec, cap=0.45)
        return src, fixed_dec, learned_dec, capped_dec

    print("stage=predict_validation", flush=True)
    val_src, val_fixed_dec, val_learned_dec, val_capped_dec = predict(frames["val_raw"], oof=True)
    print("stage=predict_oos", flush=True)
    oos_src, oos_fixed_dec, oos_learned_dec, oos_capped_dec = predict(frames["oos_raw"], oof=False)
    val_src.to_csv(out_dir / "validation_predictions_baseline_entry_q070.csv", index=False)
    oos_src.to_csv(out_dir / "oos_predictions_baseline_entry_q070.csv", index=False)
    val_learned_dec.to_csv(out_dir / "validation_decisions_frozen_risk_heads_q070.csv", index=False)
    oos_learned_dec.to_csv(out_dir / "oos_decisions_frozen_risk_heads_q070.csv", index=False)
    val_capped_dec.to_csv(out_dir / "validation_decisions_frozen_risk_heads_margin_cap045_q070.csv", index=False)
    oos_capped_dec.to_csv(out_dir / "oos_decisions_frozen_risk_heads_margin_cap045_q070.csv", index=False)

    results = {
        "frozen_baseline_fixed_risk": {
            "validation": omega._metrics(frames["val_raw"], val_fixed_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
            "oos": omega._metrics(frames["oos_raw"], oos_fixed_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        },
        "frozen_risk_heads_raw": {
            "validation": omega._metrics(frames["val_raw"], val_learned_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
            "oos": omega._metrics(frames["oos_raw"], oos_learned_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        },
        "frozen_risk_heads_margin_cap045": {
            "validation": omega._metrics(frames["val_raw"], val_capped_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
            "oos": omega._metrics(frames["oos_raw"], oos_capped_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        },
    }
    report = {
        "model_id": MODEL_ID,
        "baseline_bundle": str(args.baseline_bundle),
        "design": "Omega4 baseline bundle is frozen. Direction, quality, exit heads and encoder weights are unchanged. Only separate margin_fraction and leverage heads are trained on frozen expert encodings. Runtime remains no-exit.",
        "risk_contract": {
            "margin_fraction_buckets": [float(x) for x in risk5.MARGIN_BUCKETS.tolist()],
            "leverage_buckets": [float(x) for x in risk5.LEVERAGE_BUCKETS.tolist()],
            "notional": "margin_fraction * leverage",
            "tp_sl": "BASE_TEMPLATE take_profit/stop_loss are price-move targets; account thresholds are price_move * notional",
        },
        "risk_label_diag": risk_label_diag,
        "summaries": summaries,
        "results": results,
        "risk_prediction_distribution": {
            "raw_validation": risk5._risk_distribution(val_learned_dec),
            "raw_oos": risk5._risk_distribution(oos_learned_dec),
            "cap045_validation": risk5._risk_distribution(val_capped_dec),
            "cap045_oos": risk5._risk_distribution(oos_capped_dec),
        },
        "artifacts": {
            "out_dir": str(out_dir),
            "report": str(out_dir / "report.json"),
            "bundle": str(out_dir / "frozen_risk_heads_margin_leverage_bundle.pt"),
            "oos_decisions": str(out_dir / "oos_decisions_frozen_risk_heads_q070.csv"),
        },
    }
    torch.save(
        {
            "model_id": MODEL_ID,
            "baseline_bundle": str(args.baseline_bundle),
            "risk_models": risk_models,
            "base_cols": base_cols,
            "margin_fraction_buckets": risk5.MARGIN_BUCKETS,
            "leverage_buckets": risk5.LEVERAGE_BUCKETS,
            "model_class": "FrozenEncoderRiskHeads",
        },
        out_dir / "frozen_risk_heads_margin_leverage_bundle.pt",
    )
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": results, "risk_distribution": report["risk_prediction_distribution"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
