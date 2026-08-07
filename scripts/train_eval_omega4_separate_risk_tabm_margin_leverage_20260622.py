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


MODEL_ID = "omega4_separate_risk_tabm_margin_leverage_20260622"
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


class RiskOnlyTabM(nn.Module):
    def __init__(self, n_features: int, *, cfg: parent.ThreeHeadConfig = parent.CFG) -> None:
        super().__init__()
        self.k = int(cfg.k)
        self.n_features = int(n_features)
        self.input_scale = nn.Parameter(torch.randn(self.k, self.n_features) * 0.03 + 1.0)
        self.input_bias = nn.Parameter(torch.zeros(self.k, self.n_features))
        self.in_proj = nn.Linear(self.n_features, int(cfg.hidden))
        self.blocks = nn.ModuleList(nn.Linear(int(cfg.hidden), int(cfg.hidden)) for _ in range(max(0, int(cfg.layers) - 1)))
        self.expert_scale = nn.ParameterList(
            nn.Parameter(torch.randn(self.k, int(cfg.hidden)) * 0.03 + 1.0) for _ in range(max(0, int(cfg.layers) - 1))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(int(cfg.hidden)) for _ in range(max(0, int(cfg.layers))))
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.margin_fraction_head = nn.Linear(int(cfg.hidden), len(risk5.MARGIN_BUCKETS))
        self.leverage_head = nn.Linear(int(cfg.hidden), len(risk5.LEVERAGE_BUCKETS))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xk = x.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.in_proj(xk)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            h = layer(h * self.expert_scale[idx].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            h = h + residual
        return h

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(x)
        return {
            "margin_fraction": self.margin_fraction_head(h),
            "leverage": self.leverage_head(h),
        }


def _standardize_fit(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    out = ((arr - mean) / std).astype(np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite separate risk TabM training matrix")
    return out, {"mean": mean, "std": std, "columns": list(x.columns)}


def _standardize_apply(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    cols = list(scaler["columns"])
    if list(x.columns) != cols:
        raise RuntimeError("separate risk TabM feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = ((arr - scaler["mean"]) / scaler["std"]).astype(np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite separate risk TabM inference matrix")
    return out


def _numeric_input(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _load_baseline_bundle(bundle_path: Path, *, device: torch.device) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payload = torch.load(bundle_path, map_location=device, weights_only=False)
    return payload["models"], list(payload["base_cols"])


def _baseline_outputs(
    frame: pd.DataFrame,
    baseline_models: dict[str, dict[str, Any]],
    base_cols: list[str],
    *,
    quality_threshold: float,
    oof: bool,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    x_parent = parent._base_input(frame, base_cols)
    preds = {expert: parent._predict_payload(baseline_models[expert], x_parent, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = parent._routed(preds, route, "direction", 3)
    quality = parent._routed(preds, route, "quality", 3)
    prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
    src = parent._prediction_output(frame, direction, quality, threshold=float(quality_threshold), prefix=prefix)
    dec = parent._to_decisions(src, oof=oof)
    return src, dec, {"direction": direction, "quality": quality, "route": route}


def _parent_feature_frame(parent_arrays: dict[str, np.ndarray], dec: pd.DataFrame) -> pd.DataFrame:
    direction = parent_arrays["direction"]
    quality = parent_arrays["quality"]
    route = parent_arrays["route"].astype(np.int64)
    dir_action = np.argmax(direction, axis=1).astype(np.int64)
    q_for_dir = quality[np.arange(len(quality)), dir_action]
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.float64)
    out = pd.DataFrame(
        {
            "parent_dir_p_cash": direction[:, 0],
            "parent_dir_p_long": direction[:, 1],
            "parent_dir_p_short": direction[:, 2],
            "parent_dir_confidence": np.max(direction, axis=1),
            "parent_dir_trade_prob": direction[:, 1] + direction[:, 2],
            "parent_dir_side_edge": direction[:, 1] - direction[:, 2],
            "parent_quality_p_cash": quality[:, 0],
            "parent_quality_p_long": quality[:, 1],
            "parent_quality_p_short": quality[:, 2],
            "parent_quality_for_direction": q_for_dir,
            "parent_final_side": side,
            "parent_route_id": route.astype(np.float64),
        }
    )
    return out.astype(np.float32)


def _route_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    route = hard._route_id(frame)
    probs = parent._route_probs(frame)
    top2 = np.sort(probs, axis=1)[:, -2:]
    out = pd.DataFrame(
        {
            "risk_route_bull_prob": probs[:, 0],
            "risk_route_bear_prob": probs[:, 1],
            "risk_route_chop_prob": probs[:, 2],
            "risk_route_bull_onehot": (route == 0).astype(np.float64),
            "risk_route_bear_onehot": (route == 1).astype(np.float64),
            "risk_route_chop_onehot": (route == 2).astype(np.float64),
            "risk_route_confidence": probs.max(axis=1),
            "risk_route_margin": top2[:, 1] - top2[:, 0],
        }
    )
    return out.astype(np.float32)


def _make_variant_input(
    frame: pd.DataFrame,
    market_cols: list[str],
    parent_arrays: dict[str, np.ndarray],
    dec: pd.DataFrame,
    *,
    variant: str,
) -> pd.DataFrame:
    x = _numeric_input(frame, market_cols)
    if variant in {"B2_parent", "B3_parent_regime"}:
        x = pd.concat([x.reset_index(drop=True), _parent_feature_frame(parent_arrays, dec).reset_index(drop=True)], axis=1)
    if variant == "B3_parent_regime":
        x = pd.concat([x.reset_index(drop=True), _route_feature_frame(frame).reset_index(drop=True)], axis=1)
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _ce_tabm(logits: torch.Tensor, target: torch.Tensor, classes: int) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, int(classes)),
        target[:, None].expand(-1, int(parent.CFG.k)).reshape(-1),
        reduction="none",
    ).reshape(-1, int(parent.CFG.k)).mean(dim=1)


def _fit_expert(
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
    x_np, scaler = _standardize_fit(x)
    y_margin_np = np.asarray(y_margin, dtype=np.int64)
    y_leverage_np = np.asarray(y_leverage, dtype=np.int64)
    route_w = parent._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    margin_w = risk5._risk_weights(y_margin_np, risk_active, route_w)
    leverage_w = risk5._risk_weights(y_leverage_np, risk_active, route_w)
    if min(float(margin_w.sum()), float(leverage_w.sum())) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid separate risk sample weights")
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
    model = RiskOnlyTabM(x_np.shape[1], cfg=parent.CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(parent.CFG.lr), weight_decay=float(parent.CFG.weight_decay))
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        for xb, ym, yl, wm, wl in loader:
            xb, ym, yl, wm, wl = xb.to(device), ym.to(device), yl.to(device), wm.to(device), wl.to(device)
            out = model(xb)
            loss_margin = (_ce_tabm(out["margin_fraction"], ym, len(risk5.MARGIN_BUCKETS)) * wm).sum() / torch.clamp(wm.sum(), min=1.0)
            loss_leverage = (_ce_tabm(out["leverage"], yl, len(risk5.LEVERAGE_BUCKETS)) * wl).sum() / torch.clamp(wl.sum(), min=1.0)
            loss = loss_margin + loss_leverage
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(parent.CFG.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "model_id": MODEL_ID,
        "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "n_features": int(x_np.shape[1]),
        "best_validation_loss": float(best_loss),
        "epochs_ran": int(last_epoch),
        "input_columns": list(x.columns),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, model_path)
    return payload


@torch.no_grad()
def _predict_payload(payload: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> dict[str, np.ndarray]:
    model = RiskOnlyTabM(int(payload["n_features"]), cfg=parent.CFG).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = _standardize_apply(x, payload["scaler"])
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-bundle", type=Path, default=BASELINE_BUNDLE)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--quality-threshold", type=float, default=0.70)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260622)
    ap.add_argument("--out-suffix", default="e8_fulltrain_q070")
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
    train_fit_frame = train_raw.iloc[: int(args.max_train_rows)].reset_index(drop=True) if int(args.max_train_rows) > 0 else train_raw.reset_index(drop=True)
    print("stage=baseline_outputs_train", flush=True)
    train_src, train_fixed_dec, train_parent = _baseline_outputs(
        train_fit_frame,
        baseline_models,
        base_cols,
        quality_threshold=float(args.quality_threshold),
        oof=True,
        device=device,
    )
    del train_src
    print(f"stage=build_risk_labels rows={len(train_fit_frame)}", flush=True)
    y_margin, y_leverage, risk_active, risk_label_diag = risk5._build_margin_leverage_labels(train_fit_frame, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    print("stage=baseline_outputs_validation", flush=True)
    val_src, val_fixed_dec, val_parent = _baseline_outputs(
        frames["val_raw"],
        baseline_models,
        base_cols,
        quality_threshold=float(args.quality_threshold),
        oof=True,
        device=device,
    )
    print("stage=baseline_outputs_oos", flush=True)
    oos_src, oos_fixed_dec, oos_parent = _baseline_outputs(
        frames["oos_raw"],
        baseline_models,
        base_cols,
        quality_threshold=float(args.quality_threshold),
        oof=False,
        device=device,
    )
    val_src.to_csv(out_dir / "validation_predictions_baseline_entry_q070.csv", index=False)
    oos_src.to_csv(out_dir / "oos_predictions_baseline_entry_q070.csv", index=False)

    variants = {
        "B1_market": "market features only",
        "B2_parent": "market features + frozen parent probabilities and selected side",
        "B3_parent_regime": "market features + frozen parent probabilities + explicit route features",
    }
    results: dict[str, Any] = {
        "baseline_fixed_risk": {
            "validation": omega._metrics(frames["val_raw"], val_fixed_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
            "oos": omega._metrics(frames["oos_raw"], oos_fixed_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        }
    }
    risk_distributions: dict[str, Any] = {}
    variant_summaries: dict[str, Any] = {}
    bundle_models: dict[str, dict[str, dict[str, Any]]] = {}
    for variant, description in variants.items():
        print(f"stage=variant_start variant={variant}", flush=True)
        x_train = _make_variant_input(train_fit_frame, base_cols, train_parent, train_fixed_dec, variant=variant)
        x_val = _make_variant_input(frames["val_raw"], base_cols, val_parent, val_fixed_dec, variant=variant)
        x_oos = _make_variant_input(frames["oos_raw"], base_cols, oos_parent, oos_fixed_dec, variant=variant)
        models: dict[str, dict[str, Any]] = {}
        summaries: dict[str, Any] = {}
        for idx, expert in enumerate(hard.EXPERT_NAMES):
            print(f"stage=train_variant_expert variant={variant} expert={expert}", flush=True)
            payload = _fit_expert(
                x_train,
                y_margin,
                y_leverage,
                risk_active,
                train_fit_frame,
                expert_idx=idx,
                seed=int(args.seed) + 1000 * len(bundle_models),
                epochs=int(args.epochs),
                device=device,
                model_path=out_dir / "models" / variant / f"{expert}_separate_risk_tabm.pt",
            )
            models[expert] = payload
            summaries[expert] = {"best_validation_loss": payload["best_validation_loss"], "epochs_ran": payload["epochs_ran"]}
        bundle_models[variant] = models
        variant_summaries[variant] = {"description": description, "input_feature_count": int(x_train.shape[1]), "experts": summaries}

        def routed_risk(x: pd.DataFrame, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            preds = {expert: _predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
            route = hard._route_id(frame)
            margin = parent._routed(preds, route, "margin_fraction", len(risk5.MARGIN_BUCKETS))
            leverage = parent._routed(preds, route, "leverage", len(risk5.LEVERAGE_BUCKETS))
            return margin, leverage

        val_margin, val_leverage = routed_risk(x_val, frames["val_raw"])
        oos_margin, oos_leverage = routed_risk(x_oos, frames["oos_raw"])
        val_dec = risk5._apply_learned_risk(val_fixed_dec, val_margin, val_leverage)
        oos_dec = risk5._apply_learned_risk(oos_fixed_dec, oos_margin, oos_leverage)
        val_cap = _apply_margin_cap(val_dec, cap=0.45)
        oos_cap = _apply_margin_cap(oos_dec, cap=0.45)
        val_dec.to_csv(out_dir / f"validation_decisions_{variant}_raw_q070.csv", index=False)
        oos_dec.to_csv(out_dir / f"oos_decisions_{variant}_raw_q070.csv", index=False)
        val_cap.to_csv(out_dir / f"validation_decisions_{variant}_cap045_q070.csv", index=False)
        oos_cap.to_csv(out_dir / f"oos_decisions_{variant}_cap045_q070.csv", index=False)
        results[f"{variant}_raw"] = {
            "validation": omega._metrics(frames["val_raw"], val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
            "oos": omega._metrics(frames["oos_raw"], oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        }
        results[f"{variant}_cap045"] = {
            "validation": omega._metrics(frames["val_raw"], val_cap, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
            "oos": omega._metrics(frames["oos_raw"], oos_cap, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        }
        risk_distributions[f"{variant}_raw_validation"] = risk5._risk_distribution(val_dec)
        risk_distributions[f"{variant}_raw_oos"] = risk5._risk_distribution(oos_dec)
        risk_distributions[f"{variant}_cap045_validation"] = risk5._risk_distribution(val_cap)
        risk_distributions[f"{variant}_cap045_oos"] = risk5._risk_distribution(oos_cap)

    ranking = []
    for name, metrics in results.items():
        if name == "baseline_fixed_risk":
            continue
        ranking.append(
            {
                "variant": name,
                "oos_pnl": float(metrics["oos"]["pnl"]),
                "oos_mdd": float(metrics["oos"]["mdd"]),
                "oos_trades": int(metrics["oos"]["trades"]),
                "validation_pnl": float(metrics["validation"]["pnl"]),
                "validation_mdd": float(metrics["validation"]["mdd"]),
            }
        )
    ranking.sort(key=lambda r: (float(r["oos_pnl"]), float(r["validation_pnl"])), reverse=True)
    report = {
        "model_id": MODEL_ID,
        "baseline_bundle": str(args.baseline_bundle),
        "design": "Frozen Omega4 entry/exit parent plus separate regime-routed RiskOnlyTabM models. B1/B2/B3 compare risk-model input contracts; parent action is never changed.",
        "variants": variants,
        "risk_contract": {
            "margin_fraction_buckets": [float(x) for x in risk5.MARGIN_BUCKETS.tolist()],
            "leverage_buckets": [float(x) for x in risk5.LEVERAGE_BUCKETS.tolist()],
            "notional": "margin_fraction * leverage",
            "tp_sl": "BASE_TEMPLATE take_profit/stop_loss are price-move targets; account thresholds are price_move * notional",
        },
        "risk_label_diag": risk_label_diag,
        "summaries": variant_summaries,
        "results": results,
        "ranking_by_oos_pnl": ranking,
        "risk_prediction_distribution": risk_distributions,
        "artifacts": {
            "out_dir": str(out_dir),
            "report": str(out_dir / "report.json"),
            "bundle": str(out_dir / "separate_risk_tabm_bundle.pt"),
        },
    }
    torch.save(
        {
            "model_id": MODEL_ID,
            "baseline_bundle": str(args.baseline_bundle),
            "base_cols": base_cols,
            "variant_models": bundle_models,
            "variant_summaries": variant_summaries,
            "margin_fraction_buckets": risk5.MARGIN_BUCKETS,
            "leverage_buckets": risk5.LEVERAGE_BUCKETS,
            "model_class": "RiskOnlyTabM",
        },
        out_dir / "separate_risk_tabm_bundle.pt",
    )
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "ranking": ranking, "baseline": results["baseline_fixed_risk"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
