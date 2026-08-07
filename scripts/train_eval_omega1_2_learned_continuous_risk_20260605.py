#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_mamba_sac_3head_feature_coordinator_20260604 as feat_coord  # noqa: E402
import train_eval_omega1_2_mamba_sac_lifecycle_controller_20260604 as lifecycle  # noqa: E402
import train_eval_omega1_2_supervised_risk_selector_20260604 as sup_risk  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_learned_continuous_risk_20260605"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

RISK_COLS = ["take_profit", "stop_loss", "leverage", "notional_exposure"]
BOUNDS = {
    "take_profit": (0.010, 0.050),
    "stop_loss": (0.006, 0.030),
    "leverage": (1.0, 4.0),
    "notional_exposure": (0.18, 0.80),
}
BASE_RISK = np.asarray([0.026, 0.014, 2.0, 0.45], dtype=np.float32)


@dataclass
class RiskDataset:
    x: np.ndarray
    y: np.ndarray
    score: np.ndarray
    weight: np.ndarray


class ContinuousRiskNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, dropout: float = 0.08) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.risk_head = nn.Sequential(nn.Linear(hidden, 128), nn.SiLU(), nn.Linear(128, len(RISK_COLS)))
        self.score_head = nn.Sequential(nn.Linear(hidden, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.risk_head(h), self.score_head(h).squeeze(-1)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _risk_to_unit(risk: np.ndarray) -> np.ndarray:
    out = np.zeros_like(risk, dtype=np.float32)
    for j, col in enumerate(RISK_COLS):
        lo, hi = BOUNDS[col]
        out[:, j] = (risk[:, j] - lo) / max(hi - lo, 1e-12)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _unit_to_risk(unit: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(unit, dtype=np.float32), 0.0, 1.0)
    out = np.zeros_like(arr, dtype=np.float32)
    for j, col in enumerate(RISK_COLS):
        lo, hi = BOUNDS[col]
        out[:, j] = lo + arr[:, j] * (hi - lo)
    out[:, 1] = np.maximum(out[:, 1], 0.006)
    out[:, 3] = np.maximum(out[:, 3], 0.0)
    return out.astype(np.float32)


def _fit_norm(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    med = np.nanmedian(arr, axis=0).astype(np.float32)
    q25 = np.nanpercentile(arr, 25, axis=0).astype(np.float32)
    q75 = np.nanpercentile(arr, 75, axis=0).astype(np.float32)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    out = (arr - med) / scale
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite risk training matrix")
    return np.tanh(out / 3.0).astype(np.float32), {"columns": list(x.columns), "median": med, "scale": scale}


def _apply_norm(x: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    cols = list(norm["columns"])
    if list(x.columns) != cols:
        raise RuntimeError("learned risk feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = (arr - norm["median"]) / norm["scale"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite risk inference matrix")
    return np.tanh(out / 3.0).astype(np.float32)


def _final_action(src: pd.DataFrame, *, oof: bool) -> np.ndarray:
    prefix = "omega1_regime3_expertdq_oof_" if oof else "omega1_regime3_expertdq_"
    action = pd.to_numeric(src[f"{prefix}final_action"], errors="raise").to_numpy(dtype=np.int64)
    if not set(np.unique(action)).issubset({omega.ACTION_CASH, omega.ACTION_LONG, omega.ACTION_SHORT}):
        raise RuntimeError(f"unexpected final_action values: {sorted(np.unique(action).tolist())}")
    return action


def _side_from_action(action: np.ndarray) -> np.ndarray:
    side = np.zeros(len(action), dtype=np.int64)
    side[action == omega.ACTION_LONG] = 1
    side[action == omega.ACTION_SHORT] = -1
    return side


def _single_dec_row(action: int, side: int, risk: np.ndarray) -> pd.Series:
    return pd.Series(
        {
            "action": int(action),
            "side": int(side),
            "quality_score": 1.0,
            "confidence": 1.0,
            "notional_exposure": float(risk[3]),
            "leverage": float(risk[2]),
            "max_hold_bars": 72,
            "take_profit": float(risk[0]),
            "stop_loss": float(risk[1]),
        }
    )


def _candidate_risks(rng: np.random.Generator, n: int, *, mode: str) -> np.ndarray:
    risks: list[np.ndarray] = [BASE_RISK.copy()]
    lo = np.asarray([BOUNDS[c][0] for c in RISK_COLS], dtype=np.float32)
    hi = np.asarray([BOUNDS[c][1] for c in RISK_COLS], dtype=np.float32)
    if mode in {"direct", "ensemble"}:
        risks.extend(rng.uniform(lo, hi, size=(max(int(n) - 1, 0), len(RISK_COLS))).astype(np.float32))
    elif mode == "delta_anchor":
        span = hi - lo
        for _ in range(max(int(n) - 1, 0)):
            delta = rng.normal(0.0, [0.20, 0.20, 0.18, 0.22], size=len(RISK_COLS)).astype(np.float32) * span
            risks.append(np.clip(BASE_RISK + delta, lo, hi).astype(np.float32))
    else:
        raise RuntimeError(f"unknown risk label mode: {mode}")
    return np.asarray(risks[: max(int(n), 1)], dtype=np.float32)


def _build_risk_dataset(
    frame: pd.DataFrame,
    src: pd.DataFrame,
    x_risk: pd.DataFrame,
    *,
    oof: bool,
    seed: int,
    candidates_per_row: int,
    max_rows: int,
    label_mode: str,
    fee: float,
    slip: float,
    cost_mult: float,
) -> tuple[RiskDataset, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    action = _final_action(src, oof=oof)
    side = _side_from_action(action)
    active_idx = np.flatnonzero(action != omega.ACTION_CASH)
    if int(max_rows) > 0 and len(active_idx) > int(max_rows):
        keep = np.linspace(0, len(active_idx) - 1, int(max_rows)).round().astype(np.int64)
        active_idx = active_idx[keep]
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    y: list[np.ndarray] = []
    scores: list[float] = []
    weights: list[float] = []
    reasons: dict[str, int] = {}
    best_raw: list[float] = []
    for idx in active_idx:
        risks = _candidate_risks(rng, int(candidates_per_row), mode=str(label_mode))
        row_scores = []
        row_nets = []
        row_reasons = []
        for risk in risks:
            score, meta = omega._simulate_trade(
                frame,
                arrays,
                int(idx),
                _single_dec_row(int(action[int(idx)]), int(side[int(idx)]), risk),
                fee=fee,
                slip=slip,
                cost_mult=cost_mult,
            )
            row_scores.append(float(score))
            row_nets.append(float(meta.get("net", score)))
            row_reasons.append(str(meta.get("exit_reason", "unknown")))
        best_i = int(np.argmax(row_scores))
        y.append(risks[best_i])
        scores.append(float(row_scores[best_i]))
        best_raw.append(float(row_nets[best_i]))
        reasons[row_reasons[best_i]] = reasons.get(row_reasons[best_i], 0) + 1
        scale = max(float(np.std(row_scores)), 1e-4)
        weights.append(float(np.exp(np.clip((float(row_scores[best_i]) - float(np.median(row_scores))) / scale, -4.0, 4.0))))
    if len(y) < 200:
        raise RuntimeError(f"not enough learned risk rows: {len(y)}")
    x_sel = x_risk.iloc[active_idx].reset_index(drop=True)
    x_np, norm = _fit_norm(x_sel)
    y_np = _risk_to_unit(np.asarray(y, dtype=np.float32))
    return (
        RiskDataset(x_np, y_np, np.asarray(scores, dtype=np.float32), np.asarray(weights, dtype=np.float32)),
        {
            "rows": int(len(y)),
            "candidates_per_row": int(candidates_per_row),
            "label_mode": str(label_mode),
            "score_mean": float(np.mean(scores)),
            "score_p50": float(np.percentile(scores, 50)),
            "score_p90": float(np.percentile(scores, 90)),
            "net_mean": float(np.mean(best_raw)),
            "best_exit_reasons": reasons,
            "normalizer": norm,
        },
    )


def _train_risk_net(
    data: RiskDataset,
    *,
    device: torch.device,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    hidden: int,
) -> tuple[ContinuousRiskNet, dict[str, Any]]:
    _seed_everything(seed)
    model = ContinuousRiskNet(data.x.shape[1], hidden=int(hidden)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=2e-5)
    ds = TensorDataset(torch.from_numpy(data.x), torch.from_numpy(data.y), torch.from_numpy(data.score), torch.from_numpy(data.weight))
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=True)
    it = iter(dl)
    last: dict[str, Any] = {}
    score_scale = max(float(np.std(data.score)), 1e-4)
    score_center = float(np.median(data.score))
    for step in range(1, int(steps) + 1):
        try:
            xb, yb, sb, wb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb, sb, wb = next(it)
        xb = xb.to(device)
        yb = yb.to(device)
        sb = ((sb.to(device) - score_center) / score_scale).clamp(-5.0, 5.0)
        wb = wb.to(device).clamp(0.25, 20.0)
        pred_raw, pred_score = model(xb)
        pred = torch.sigmoid(pred_raw)
        mse = ((pred - yb) ** 2).mean(dim=1)
        risk_loss = (mse * wb).sum() / torch.clamp(wb.sum(), min=1.0)
        score_loss = torch.nn.functional.smooth_l1_loss(pred_score, sb)
        loss = risk_loss + 0.25 * score_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        if step % 250 == 0 or step == int(steps):
            last = {"step": int(step), "risk_loss": float(risk_loss.detach().cpu()), "score_loss": float(score_loss.detach().cpu())}
            print(json.dumps({"stage": "learned_risk_train", **last}, ensure_ascii=False), flush=True)
    return model.cpu(), last


def _train_sklearn_risk(data: RiskDataset, *, kind: str, seed: int) -> tuple[Any, dict[str, Any]]:
    if kind == "hgb":
        model: Any = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.035,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                min_samples_leaf=40,
                random_state=int(seed),
            )
        )
    elif kind == "extratrees":
        model = ExtraTreesRegressor(
            n_estimators=180,
            max_depth=6,
            min_samples_leaf=35,
            random_state=int(seed),
            n_jobs=-1,
        )
    elif kind == "extratrees_ensemble":
        models = []
        rng = np.random.default_rng(int(seed))
        n = len(data.x)
        for k in range(3):
            idx = rng.choice(np.arange(n), size=n, replace=True)
            m = ExtraTreesRegressor(
                n_estimators=160,
                max_depth=6,
                min_samples_leaf=35,
                random_state=int(seed) + k + 1,
                n_jobs=-1,
            )
            m.fit(data.x[idx], data.y[idx], sample_weight=data.weight[idx])
            models.append(m)
        pred = np.mean([m.predict(data.x) for m in models], axis=0)
        mse = float(np.mean((np.clip(pred, 0.0, 1.0) - data.y) ** 2))
        diag = {"kind": kind, "train_mse": mse, "members": len(models)}
        print(json.dumps({"stage": "sklearn_risk_train", **diag}, ensure_ascii=False), flush=True)
        return models, diag
    else:
        raise RuntimeError(f"unknown sklearn risk kind: {kind}")
    model.fit(data.x, data.y, sample_weight=data.weight)
    pred = np.clip(model.predict(data.x), 0.0, 1.0)
    mse = float(np.mean((pred - data.y) ** 2))
    diag = {"kind": kind, "train_mse": mse}
    print(json.dumps({"stage": "sklearn_risk_train", **diag}, ensure_ascii=False), flush=True)
    return model, diag


@torch.no_grad()
def _predict_risk(model: ContinuousRiskNet, x: pd.DataFrame, norm: dict[str, Any], *, device: torch.device, batch_size: int, mode: str) -> np.ndarray:
    model = model.to(device)
    model.eval()
    arr = _apply_norm(x, norm)
    outs: list[np.ndarray] = []
    for start in range(0, len(arr), int(batch_size)):
        xb = torch.tensor(arr[start : start + int(batch_size)], dtype=torch.float32, device=device)
        raw, _score = model(xb)
        unit = torch.sigmoid(raw).cpu().numpy().astype(np.float32)
        if str(mode) == "anchor_delta":
            anchor = _risk_to_unit(np.repeat(BASE_RISK[None, :], len(unit), axis=0))
            unit = np.clip(anchor + (unit - 0.5) * 0.55, 0.0, 1.0)
        outs.append(unit)
    unit_all = np.concatenate(outs) if outs else np.zeros((0, len(RISK_COLS)), dtype=np.float32)
    return _unit_to_risk(unit_all)


def _predict_sklearn_risk(
    model: Any,
    x: pd.DataFrame,
    norm: dict[str, Any],
    *,
    mode: str,
    base_blend: float,
    uncertainty_gate: float,
) -> np.ndarray:
    arr = _apply_norm(x, norm)
    if isinstance(model, list):
        member_preds = np.stack([m.predict(arr) for m in model], axis=0).astype(np.float32)
        unit = np.mean(member_preds, axis=0).astype(np.float32)
        uncertainty = np.mean(np.std(member_preds, axis=0), axis=1).astype(np.float32)
    elif hasattr(model, "estimators_") and not isinstance(model, MultiOutputRegressor):
        tree_preds = np.stack([est.predict(arr) for est in model.estimators_], axis=0).astype(np.float32)
        unit = np.mean(tree_preds, axis=0).astype(np.float32)
        uncertainty = np.mean(np.std(tree_preds, axis=0), axis=1).astype(np.float32)
    else:
        unit = model.predict(arr).astype(np.float32)
        uncertainty = np.zeros(len(unit), dtype=np.float32)
    unit = np.clip(unit, 0.0, 1.0)
    if str(mode) == "anchor_delta":
        anchor = _risk_to_unit(np.repeat(BASE_RISK[None, :], len(unit), axis=0))
        unit = np.clip(anchor + (unit - 0.5) * 0.55, 0.0, 1.0)
    risk = _unit_to_risk(unit)
    if len(risk):
        blend = np.full(len(risk), float(np.clip(base_blend, 0.0, 1.0)), dtype=np.float32)
        if float(uncertainty_gate) > 0.0:
            dynamic = np.clip(uncertainty / float(uncertainty_gate), 0.0, 1.0)
            blend = np.maximum(blend, dynamic.astype(np.float32))
        risk = (1.0 - blend[:, None]) * risk + blend[:, None] * BASE_RISK[None, :]
    return risk


def _decision_from_learned_risk(
    base_x: pd.DataFrame,
    src: pd.DataFrame,
    *,
    oof: bool,
    model: Any,
    norm: dict[str, Any],
    device: torch.device,
    batch_size: int,
    mode: str,
    base_blend: float,
    uncertainty_gate: float,
    model_kind: str,
) -> pd.DataFrame:
    action = _final_action(src, oof=oof)
    side = _side_from_action(action)
    active = action != omega.ACTION_CASH
    risk = np.zeros((len(action), len(RISK_COLS)), dtype=np.float32)
    risk[:, 1] = 0.0
    risk[:, 2] = 1.0
    if bool(active.any()):
        x_risk = sup_risk._risk_features(base_x, src, oof=oof)
        if model_kind == "mlp":
            risk[active] = _predict_risk(model, x_risk.loc[active].reset_index(drop=True), norm, device=device, batch_size=int(batch_size), mode=mode)
            if float(base_blend) > 0.0:
                blend = float(np.clip(base_blend, 0.0, 1.0))
                risk[active] = (1.0 - blend) * risk[active] + blend * BASE_RISK[None, :]
        else:
            risk[active] = _predict_sklearn_risk(model, x_risk.loc[active].reset_index(drop=True), norm, mode=mode, base_blend=float(base_blend), uncertainty_gate=float(uncertainty_gate))
    prefix = "omega1_regime3_expertdq_oof_" if oof else "omega1_regime3_expertdq_"
    dec = pd.DataFrame(
        {
            "timestamp": src["timestamp"].to_numpy(),
            "action": action,
            "side": side,
            "quality_score": pd.to_numeric(src[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64),
            "confidence": pd.to_numeric(src[f"{prefix}dir_confidence"], errors="raise").to_numpy(dtype=np.float64),
            "notional_exposure": np.where(active, risk[:, 3], 0.0),
            "leverage": np.where(active, risk[:, 2], 1.0),
            "max_hold_bars": 0,
            "take_profit": np.where(active, risk[:, 0], 0.0),
            "stop_loss": np.where(active, risk[:, 1], 0.0),
        }
    )
    return dec


def _prepare_frames_with_learned_risk(
    *,
    threehead_dir: Path,
    quality_threshold: float,
    device: torch.device,
    label_mode: str,
    pred_mode: str,
    risk_rows: int,
    candidates_per_row: int,
    risk_steps: int,
    risk_lr: float,
    risk_batch_size: int,
    risk_hidden: int,
    risk_base_blend: float,
    risk_uncertainty_gate: float,
    risk_model_kind: str,
    seed: int,
    cost_mult: float,
) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    base_frames = feat_coord._prepare_frames(threehead_dir, quality_threshold=float(quality_threshold), device=device)
    fee, slip = omega._load_fee_slip()
    bundle = feat_coord._load_3head_payloads(threehead_dir)
    train_x, train_src = feat_coord._predict_3head_frame(base_frames["train_df"], bundle, quality_threshold=float(quality_threshold), device=device, oof=True)
    val_x, val_src = feat_coord._predict_3head_frame(base_frames["val_df"], bundle, quality_threshold=float(quality_threshold), device=device, oof=True)
    oos_x, oos_src = feat_coord._predict_3head_frame(base_frames["oos_df"], bundle, quality_threshold=float(quality_threshold), device=device, oof=False)
    train_x_risk = sup_risk._risk_features(train_x, train_src, oof=True)
    risk_data, risk_diag = _build_risk_dataset(
        base_frames["train_df"],
        train_src,
        train_x_risk,
        oof=True,
        seed=int(seed),
        candidates_per_row=int(candidates_per_row),
        max_rows=int(risk_rows),
        label_mode=str(label_mode),
        fee=fee,
        slip=slip,
        cost_mult=float(cost_mult),
    )
    norm = risk_diag.pop("normalizer")
    if str(risk_model_kind) == "mlp":
        risk_model, train_diag = _train_risk_net(
            risk_data,
            device=device,
            seed=int(seed),
            steps=int(risk_steps),
            batch_size=int(risk_batch_size),
            lr=float(risk_lr),
            hidden=int(risk_hidden),
        )
    else:
        risk_model, train_diag = _train_sklearn_risk(risk_data, kind=str(risk_model_kind), seed=int(seed))
    train_dec = _decision_from_learned_risk(train_x, train_src, oof=True, model=risk_model, norm=norm, device=device, batch_size=int(risk_batch_size), mode=str(pred_mode), base_blend=float(risk_base_blend), uncertainty_gate=float(risk_uncertainty_gate), model_kind=str(risk_model_kind))
    val_dec = _decision_from_learned_risk(val_x, val_src, oof=True, model=risk_model, norm=norm, device=device, batch_size=int(risk_batch_size), mode=str(pred_mode), base_blend=float(risk_base_blend), uncertainty_gate=float(risk_uncertainty_gate), model_kind=str(risk_model_kind))
    oos_dec = _decision_from_learned_risk(oos_x, oos_src, oof=False, model=risk_model, norm=norm, device=device, batch_size=int(risk_batch_size), mode=str(pred_mode), base_blend=float(risk_base_blend), uncertainty_gate=float(risk_uncertainty_gate), model_kind=str(risk_model_kind))
    feature_cols = omega._numeric_feature_cols(pd.concat([base_frames["train_df"], base_frames["val_df"]], axis=0, ignore_index=True), base_frames["oos_df"])
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
    out.update({"train_dec": train_dec, "val_dec": val_dec, "oos_dec": oos_dec, "s_train": s_train, "s_val": s_val, "s_oos": s_oos})
    return out, risk_model, norm, {"risk_data_diag": risk_diag, "risk_train_diag": train_diag}


def _risk_summary(dec: pd.DataFrame) -> dict[str, float]:
    active = pd.to_numeric(dec["action"], errors="raise").to_numpy(dtype=np.int64) != omega.ACTION_CASH
    out: dict[str, float] = {"active": int(active.sum())}
    if bool(active.any()):
        for col in RISK_COLS:
            vals = pd.to_numeric(dec.loc[active, col], errors="raise").to_numpy(dtype=np.float64)
            out[f"{col}_mean"] = float(np.mean(vals))
            out[f"{col}_p10"] = float(np.percentile(vals, 10))
            out[f"{col}_p90"] = float(np.percentile(vals, 90))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threehead-dir", type=Path, default=feat_coord.DEFAULT_3HEAD_DIR)
    ap.add_argument("--quality-threshold", type=float, default=0.75)
    ap.add_argument("--risk-label-mode", choices=["direct", "delta_anchor", "ensemble"], default="direct")
    ap.add_argument("--risk-pred-mode", choices=["direct", "anchor_delta"], default="direct")
    ap.add_argument("--risk-model-kind", choices=["mlp", "hgb", "extratrees", "extratrees_ensemble"], default="mlp")
    ap.add_argument("--risk-rows", type=int, default=5000)
    ap.add_argument("--risk-candidates-per-row", type=int, default=32)
    ap.add_argument("--risk-steps", type=int, default=800)
    ap.add_argument("--risk-hidden", type=int, default=256)
    ap.add_argument("--risk-batch-size", type=int, default=512)
    ap.add_argument("--risk-lr", type=float, default=2e-4)
    ap.add_argument("--risk-base-blend", type=float, default=0.0)
    ap.add_argument("--risk-uncertainty-gate", type=float, default=0.0)
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
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--class-balance-actor", action="store_true")
    ap.add_argument("--select-mode", choices=["actor_q", "q_only"], default="actor_q")
    ap.add_argument("--force-parent-entry", action="store_true")
    ap.add_argument("--force-entry-mult", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=260605)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = _device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, risk_model, risk_norm, risk_info = _prepare_frames_with_learned_risk(
        threehead_dir=Path(args.threehead_dir),
        quality_threshold=float(args.quality_threshold),
        device=device,
        label_mode=str(args.risk_label_mode),
        pred_mode=str(args.risk_pred_mode),
        risk_rows=int(args.risk_rows),
        candidates_per_row=int(args.risk_candidates_per_row),
        risk_steps=int(args.risk_steps),
        risk_lr=float(args.risk_lr),
        risk_batch_size=int(args.risk_batch_size),
        risk_hidden=int(args.risk_hidden),
        risk_base_blend=float(args.risk_base_blend),
        risk_uncertainty_gate=float(args.risk_uncertainty_gate),
        risk_model_kind=str(args.risk_model_kind),
        seed=int(args.seed),
        cost_mult=float(args.cost_mult),
    )
    fee, slip = omega._load_fee_slip()
    state_cols = [c for c in lifecycle._base_state(frames["s_train"]).columns if c != "timestamp"]
    bad = [c for c in state_cols if "clean_regime4" in c or "regime4_pred" in c or "tp_sl_action_score" in c or str(c).startswith("teacher_")]
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
    print(json.dumps({"stage": "learned_risk_lifecycle_train_start", "device": str(device), "seq_shape": list(data.seq.shape), "data_diag": data_diag}, ensure_ascii=False), flush=True)
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
    if str(args.risk_model_kind) == "mlp":
        torch.save(
            {"model_state_dict": risk_model.state_dict(), "normalizer": risk_norm, "state_columns": list(risk_norm["columns"]), "risk_cols": RISK_COLS, "bounds": BOUNDS},
            out_dir / "continuous_risk_model.pt",
        )
        risk_artifact = str(out_dir / "continuous_risk_model.pt")
    else:
        with (out_dir / "continuous_risk_model.pkl").open("wb") as f:
            pickle.dump({"model": risk_model, "normalizer": risk_norm, "state_columns": list(risk_norm["columns"]), "risk_cols": RISK_COLS, "bounds": BOUNDS}, f)
        risk_artifact = str(out_dir / "continuous_risk_model.pkl")
    torch.save(
        {"model_state_dict": model.state_dict(), "normalizer": norm, "seq_len": int(args.seq_len), "state_columns": state_cols, "action_names": lifecycle.ACTION_NAMES},
        out_dir / "lifecycle_controller.pt",
    )
    report = {
        "model_id": MODEL_ID,
        "design": "Exit Head feature-only + Mamba lifecycle baseline with fixed notional/leverage/TP/SL replaced by learned continuous risk heads. Parent action/quality are preserved; risk fields are continuous predictions, not finite templates.",
        "threehead_dir": str(args.threehead_dir),
        "quality_threshold": float(args.quality_threshold),
        "risk_model": {
            "type": "ContinuousRiskNet",
            "model_kind": str(args.risk_model_kind),
            "risk_cols": RISK_COLS,
            "bounds": BOUNDS,
            "label_mode": str(args.risk_label_mode),
            "pred_mode": str(args.risk_pred_mode),
            "rows": int(args.risk_rows),
            "candidates_per_row": int(args.risk_candidates_per_row),
            "steps": int(args.risk_steps),
            "hidden": int(args.risk_hidden),
            "base_blend": float(args.risk_base_blend),
            "uncertainty_gate": float(args.risk_uncertainty_gate),
            **risk_info,
        },
        "risk_summary": {split: _risk_summary(frames[f"{split}_dec"]) for split in ("train", "val", "oos")},
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
        "cost_accounting": {
            "fee": fee,
            "slip": slip,
            "cost_mult": float(args.cost_mult),
            "delta_notional_resize_fee": True,
            "partial_exit_fee": True,
            "note": "Lifecycle accounting uses notional_exposure as effective account exposure; leverage is preserved as model/state output but is not separately multiplied by lifecycle replay PnL.",
        },
        "results": {"validation": val, "oos": oos},
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json"), "risk_model": risk_artifact, "model": str(out_dir / "lifecycle_controller.pt")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
