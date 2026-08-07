#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train_omega1_direction_head_direction_only_20260602 as base
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard
import train_omega1_regime3_routed_expert_direction_quality_20260602 as cat_dq


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_regime3_routed_expert_direction_quality_tabm_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_regime3_routed_expert_direction_quality_tabm_20260602"

QUALITY_THRESHOLDS = cat_dq.QUALITY_THRESHOLDS
SPECS = [("hard", 0.0), ("soft", 0.00), ("soft", 0.05), ("soft", 0.10), ("soft", 0.20)]


@dataclass(frozen=True)
class TabMConfig:
    k: int = 8
    hidden: int = 192
    layers: int = 3
    dropout: float = 0.08
    batch_size: int = 2048
    lr: float = 2.0e-3
    weight_decay: float = 2.0e-4
    epochs_oof_direction: int = 34
    epochs_final_direction: int = 52
    epochs_quality: int = 42
    patience: int = 8


CFG = TabMConfig()


def _json_default(obj: Any) -> Any:
    return base._json_default(obj)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabMClassifier(nn.Module):
    """Small TabM-style tabular classifier with BatchEnsemble mini-experts."""

    def __init__(self, n_features: int, n_classes: int = 3, *, cfg: TabMConfig = CFG) -> None:
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
        self.out = nn.Linear(int(cfg.hidden), int(n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xk = x.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.in_proj(xk)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            h = layer(h * self.expert_scale[idx].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            h = h + residual
        return self.out(h)


def _standardize_fit(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    out = (arr - mean) / std
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized training matrix")
    return out.astype(np.float32), {"mean": mean, "std": std, "columns": list(x.columns)}


def _standardize_apply(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    cols = list(scaler["columns"])
    if list(x.columns) != cols:
        raise RuntimeError("TabM feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = (arr - scaler["mean"]) / scaler["std"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized inference matrix")
    return out.astype(np.float32)


def _fit_tabm(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None,
    seed: int,
    epochs: int,
    model_path: Path,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    x_np, scaler = _standardize_fit(x)
    y_np = np.asarray(y, dtype=np.int64)
    weights = compute_sample_weight(class_weight="balanced", y=y_np).astype(np.float32)
    if sample_weight is not None:
        weights *= np.asarray(sample_weight, dtype=np.float32)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise RuntimeError("invalid TabM sample weights")

    n = len(y_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    if split >= n:
        split = n
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    dev = _device()
    model = TabMClassifier(x_np.shape[1], 3, cfg=CFG).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))

    ds = TensorDataset(
        torch.from_numpy(x_np[train_idx]),
        torch.from_numpy(y_np[train_idx]),
        torch.from_numpy(weights[train_idx]),
    )
    loader = DataLoader(ds, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    for epoch in range(int(epochs)):
        model.train()
        for xb, yb, wb in loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            wb = wb.to(dev, non_blocking=True)
            logits = model(xb)
            loss_k = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 3),
                yb[:, None].expand(-1, int(CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(CFG.k))
            loss = (loss_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        if len(val_idx):
            model.eval()
            with torch.no_grad():
                vx = torch.from_numpy(x_np[val_idx]).to(dev)
                vy = torch.from_numpy(y_np[val_idx]).to(dev)
                vw = torch.from_numpy(weights[val_idx]).to(dev)
                vlogits = model(vx)
                vloss_k = torch.nn.functional.cross_entropy(
                    vlogits.reshape(-1, 3),
                    vy[:, None].expand(-1, int(CFG.k)).reshape(-1),
                    reduction="none",
                ).reshape(-1, int(CFG.k))
                vloss = float(((vloss_k.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
        else:
            vloss = float(loss.detach().cpu())
        if vloss + 1.0e-6 < best_loss:
            best_loss = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(CFG.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "model_id": MODEL_ID,
        "config": CFG.__dict__,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "n_features": int(x_np.shape[1]),
        "n_classes": 3,
        "best_validation_loss": float(best_loss),
        "epochs_ran": int(epoch + 1),
    }
    torch.save(payload, model_path)
    return payload


@torch.no_grad()
def _predict_tabm(payload: dict[str, Any], x: pd.DataFrame) -> np.ndarray:
    dev = _device()
    model = TabMClassifier(int(payload["n_features"]), int(payload["n_classes"]), cfg=CFG).to(dev)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = _standardize_apply(x, payload["scaler"])
    out: list[np.ndarray] = []
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(dev)
        probs = torch.softmax(model(xb), dim=-1).mean(dim=1)
        out.append(probs.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def _route_probs(frame: pd.DataFrame) -> np.ndarray:
    values = frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite Regime3 route probabilities")
    return values


def _quality_base_features(x: pd.DataFrame, frame: pd.DataFrame, dir_proba: np.ndarray) -> pd.DataFrame:
    return cat_dq._quality_base_features(x, frame, dir_proba)


def _fit_head_models(
    x: pd.DataFrame,
    y: np.ndarray,
    frame: pd.DataFrame,
    *,
    mode: str,
    floor: float,
    seed: int,
    epochs: int,
    model_dir: Path,
    suffix: str,
) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    route = hard._route_id(frame)
    probs = _route_probs(frame)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        if mode == "hard":
            mask = route == idx
            if int(mask.sum()) < 1000:
                raise RuntimeError(f"{expert}: too few hard-routed rows: {int(mask.sum())}")
            x_fit = x.loc[mask].reset_index(drop=True)
            y_fit = y[mask]
            sample_weight = None
            effective_rows = int(mask.sum())
            weight_sum = None
        elif mode == "soft":
            x_fit = x.reset_index(drop=True)
            y_fit = y
            sample_weight = float(floor) + probs[:, idx]
            effective_rows = int(len(y))
            weight_sum = float(np.asarray(sample_weight, dtype=np.float64).sum())
        else:
            raise ValueError(f"unknown mode: {mode}")
        classes = sorted(np.unique(y_fit).astype(int).tolist())
        if classes != [0, 1, 2]:
            raise RuntimeError(f"{mode}/{expert}: missing zigzag_action classes: {classes}")
        path = model_dir / f"{expert}_{suffix}.pt"
        payload = _fit_tabm(x_fit, y_fit, sample_weight=sample_weight, seed=seed + idx, epochs=epochs, model_path=path)
        models[expert] = payload
        summaries[expert] = {
            "rows": effective_rows,
            "weight_sum": weight_sum,
            "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y_fit, minlength=3))},
            "model": str(path),
            "epochs_ran": int(payload["epochs_ran"]),
            "best_validation_loss": float(payload["best_validation_loss"]),
        }
    return {"models": models, "summaries": summaries}


def _predict_expert_models(models: dict[str, dict[str, Any]], x: pd.DataFrame) -> dict[str, np.ndarray]:
    return {expert: _predict_tabm(model, x) for expert, model in models.items()}


def _routed_proba(expert_proba: dict[str, np.ndarray], route: np.ndarray) -> np.ndarray:
    return cat_dq._routed_proba(expert_proba, route)


def _oof_direction(train: pd.DataFrame, *, mode: str, floor: float, variant_dir: Path) -> dict[str, Any]:
    n = len(train)
    starts = [int(n * 0.35), int(n * 0.50), int(n * 0.65), int(n * 0.80)]
    ends = [int(n * 0.50), int(n * 0.65), int(n * 0.80), n]
    y = train["zigzag_action"].to_numpy(dtype=np.int64)
    direction_proba = np.full((n, 3), np.nan, dtype=np.float64)
    covered = np.zeros(n, dtype=bool)
    x_parts: list[pd.DataFrame] = []
    frame_parts: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    pca_reports: list[dict[str, Any]] = []
    for fold, (start, end) in enumerate(zip(starts, ends), start=1):
        fit_frame = train.iloc[:start].reset_index(drop=True)
        pred_frame = train.iloc[start:end].reset_index(drop=True)
        transformer = hard.volpca.VolPca(6).fit(fit_frame)
        x_fit = hard._features_with_transform(fit_frame, transformer)
        x_pred = hard._features_with_transform(pred_frame, transformer)
        bundle = _fit_head_models(
            x_fit,
            y[:start],
            fit_frame,
            mode=mode,
            floor=floor,
            seed=20260602 + fold * 100,
            epochs=CFG.epochs_oof_direction,
            model_dir=variant_dir / "oof_direction" / f"fold_{fold}",
            suffix="direction_head_tabm",
        )
        expert_pred = _predict_expert_models(bundle["models"], x_pred)
        routed = _routed_proba(expert_pred, hard._route_id(pred_frame))
        direction_proba[start:end] = routed
        covered[start:end] = True
        x_parts.append(x_pred)
        frame_parts.append(pred_frame)
        fold_reports.append(
            {
                "fold": fold,
                "train_rows": int(start),
                "predict_start": int(start),
                "predict_end": int(end),
                "direction_expert_summaries": bundle["summaries"],
                "metrics": base._metrics(y[start:end], routed),
            }
        )
        pca_reports.append({"fold": fold, "explained_variance": transformer.explained_variance})
    return {
        "direction_proba": direction_proba,
        "covered": covered,
        "x_oof": pd.concat(x_parts, ignore_index=True),
        "frame_oof": pd.concat(frame_parts, ignore_index=True),
        "folds": fold_reports,
        "pca_folds": pca_reports,
    }


def _train_final_direction(train: pd.DataFrame, oos: pd.DataFrame, *, mode: str, floor: float, variant_dir: Path) -> dict[str, Any]:
    y_train = train["zigzag_action"].to_numpy(dtype=np.int64)
    transformer = hard.volpca.VolPca(6).fit(train)
    x_train = hard._features_with_transform(train, transformer)
    x_oos = hard._features_with_transform(oos, transformer)
    bundle = _fit_head_models(
        x_train,
        y_train,
        train,
        mode=mode,
        floor=floor,
        seed=20260602,
        epochs=CFG.epochs_final_direction,
        model_dir=variant_dir / "final_direction",
        suffix="direction_head_tabm",
    )
    expert_pred = _predict_expert_models(bundle["models"], x_oos)
    routed = _routed_proba(expert_pred, hard._route_id(oos))
    return {
        "transformer": transformer,
        "x_train": x_train,
        "x_oos": x_oos,
        "direction_summaries": bundle["summaries"],
        "oos_direction_proba": routed,
        "oos_expert_direction_proba": expert_pred,
    }


def _train_quality_from_oof(oof: dict[str, Any], train_y: np.ndarray, *, mode: str, floor: float, variant_dir: Path) -> dict[str, Any]:
    covered = oof["covered"]
    frame_oof = oof["frame_oof"].reset_index(drop=True)
    direction_oof = oof["direction_proba"][covered]
    xq = _quality_base_features(oof["x_oof"], frame_oof, direction_oof)
    yq = train_y[covered]
    bundle = _fit_head_models(
        xq,
        yq,
        frame_oof,
        mode=mode,
        floor=floor,
        seed=20260603,
        epochs=CFG.epochs_quality,
        model_dir=variant_dir / "quality",
        suffix="quality_head_tabm",
    )
    expert_quality = _predict_expert_models(bundle["models"], xq)
    quality_oof = _routed_proba(expert_quality, hard._route_id(frame_oof))
    return {
        "x_quality_oof": xq,
        "quality_summaries": bundle["summaries"],
        "quality_oof_proba": quality_oof,
        "quality_models": bundle["models"],
    }


def _evaluate_variant(train: pd.DataFrame, oos: pd.DataFrame, *, mode: str, floor: float) -> dict[str, Any]:
    variant = f"{mode}_floor_{floor:.2f}".replace(".", "p")
    variant_dir = OUT_DIR / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    y_train = train["zigzag_action"].to_numpy(dtype=np.int64)
    y_oos = oos["zigzag_action"].to_numpy(dtype=np.int64)
    oof = _oof_direction(train, mode=mode, floor=floor, variant_dir=variant_dir)
    direction_oof = oof["direction_proba"][oof["covered"]]
    direction_oof_metrics = base._metrics(y_train[oof["covered"]], direction_oof)
    quality = _train_quality_from_oof(oof, y_train, mode=mode, floor=floor, variant_dir=variant_dir)
    selected_threshold, threshold_rows = cat_dq._select_threshold(y_train[oof["covered"]], direction_oof, quality["quality_oof_proba"])
    filtered_oof = cat_dq._apply_quality_filter(direction_oof, quality["quality_oof_proba"], selected_threshold)
    filtered_oof_metrics = base._metrics(y_train[oof["covered"]], filtered_oof)

    final_direction = _train_final_direction(train, oos, mode=mode, floor=floor, variant_dir=variant_dir)
    xq_oos = _quality_base_features(final_direction["x_oos"], oos, final_direction["oos_direction_proba"])
    expert_quality_oos = _predict_expert_models(quality["quality_models"], xq_oos)
    quality_oos = _routed_proba(expert_quality_oos, hard._route_id(oos))
    direction_oos_metrics = base._metrics(y_oos, final_direction["oos_direction_proba"])
    filtered_oos = cat_dq._apply_quality_filter(final_direction["oos_direction_proba"], quality_oos, selected_threshold)
    filtered_oos_metrics = base._metrics(y_oos, filtered_oos)

    oof_out = cat_dq._prediction_output(oof["frame_oof"], direction_oof, quality["quality_oof_proba"], threshold=selected_threshold, prefix="omega1_regime3_expertdq_oof")
    oos_out = cat_dq._prediction_output(oos, final_direction["oos_direction_proba"], quality_oos, threshold=selected_threshold, prefix="omega1_regime3_expertdq")
    oof_path = variant_dir / f"training_features_2025_{variant}_omega1_regime3_expertdq_oof_20260602.csv"
    oos_path = variant_dir / f"training_features_2026_rebuilt_{variant}_omega1_regime3_expertdq_20260602.csv"
    oof_out.to_csv(oof_path, index=False)
    oos_out.to_csv(oos_path, index=False)
    contract_path = variant_dir / f"{variant}_omega1_regime3_expertdq_tabm_contract.joblib"
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "variant": variant,
            "mode": mode,
            "floor": float(floor),
            "label_source_direction": "zigzag_action",
            "label_source_quality": "zigzag_action",
            "route_cols": hard.ROUTE_COLS,
            "route_extra_cols": hard.ROUTE_EXTRA_COLS,
            "expert_names": hard.EXPERT_NAMES,
            "base_cols": hard.volpca.BASE_COLS,
            "volatility_cols": hard.volpca.VOL_COLS,
            "direction_feature_cols": list(final_direction["x_train"].columns),
            "quality_feature_cols": list(xq_oos.columns),
            "selected_quality_threshold": float(selected_threshold),
            "pca_transformer": final_direction["transformer"],
            "tabm_config": CFG.__dict__,
            "direction_model_paths": {k: v["model"] for k, v in final_direction["direction_summaries"].items()},
            "quality_model_paths": {k: v["model"] for k, v in quality["quality_summaries"].items()},
        },
        contract_path,
    )
    delta = {
        "delta_direction_oos_bacc": float(direction_oos_metrics["balanced_accuracy"] - hard.BASELINE_VOLPCA06["oos_bacc"]),
        "delta_filtered_oos_bacc": float(filtered_oos_metrics["balanced_accuracy"] - hard.BASELINE_VOLPCA06["oos_bacc"]),
        "delta_direction_oos_auc": None if direction_oos_metrics["ovr_auc"] is None else float(direction_oos_metrics["ovr_auc"] - hard.BASELINE_VOLPCA06["oos_auc"]),
        "delta_filtered_oos_auc": None if filtered_oos_metrics["ovr_auc"] is None else float(filtered_oos_metrics["ovr_auc"] - hard.BASELINE_VOLPCA06["oos_auc"]),
        "delta_direction_oos_proxy_wr": None if direction_oos_metrics["proxy_wr"] is None else float(direction_oos_metrics["proxy_wr"] - hard.BASELINE_VOLPCA06["oos_proxy_wr"]),
        "delta_filtered_oos_proxy_wr": None if filtered_oos_metrics["proxy_wr"] is None else float(filtered_oos_metrics["proxy_wr"] - hard.BASELINE_VOLPCA06["oos_proxy_wr"]),
        "delta_filtered_oos_proxy_trades": int(filtered_oos_metrics["proxy_trades"] - hard.BASELINE_VOLPCA06["oos_proxy_trades"]),
    }
    return {
        "variant": variant,
        "mode": mode,
        "floor": float(floor),
        "selected_quality_threshold": float(selected_threshold),
        "direction_oof_metrics": direction_oof_metrics,
        "filtered_oof_metrics": filtered_oof_metrics,
        "direction_oos_metrics": direction_oos_metrics,
        "filtered_oos_metrics": filtered_oos_metrics,
        "delta_vs_global_volatility_pca06": delta,
        "threshold_grid": threshold_rows,
        "direction_folds": oof["folds"],
        "pca_folds": oof["pca_folds"],
        "final_pca_explained_variance": final_direction["transformer"].explained_variance,
        "direction_summaries": final_direction["direction_summaries"],
        "quality_summaries": quality["quality_summaries"],
        "artifacts": {
            "variant_dir": str(variant_dir),
            "oof_2025": str(oof_path),
            "oos_2026": str(oos_path),
            "contract": str(contract_path),
        },
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = hard._build_frame(2025)
    oos = hard._build_frame(2026)
    required = [*hard.volpca.BASE_COLS, *hard.volpca.VOL_COLS, *hard.ROUTE_COLS, *hard.ROUTE_EXTRA_COLS]
    hard._assert_finite(train, required, "train")
    hard._assert_finite(oos, required, "oos")
    variants: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for mode, floor in SPECS:
        payload = _evaluate_variant(train, oos, mode=mode, floor=float(floor))
        variants[payload["variant"]] = payload
        rows.append(
            {
                "variant": payload["variant"],
                "mode": payload["mode"],
                "floor": payload["floor"],
                "selected_quality_threshold": payload["selected_quality_threshold"],
                "direction_oof_bacc": payload["direction_oof_metrics"]["balanced_accuracy"],
                "direction_oof_auc": payload["direction_oof_metrics"]["ovr_auc"],
                "direction_oof_proxy_wr": payload["direction_oof_metrics"]["proxy_wr"],
                "filtered_oof_bacc": payload["filtered_oof_metrics"]["balanced_accuracy"],
                "filtered_oof_auc": payload["filtered_oof_metrics"]["ovr_auc"],
                "filtered_oof_proxy_wr": payload["filtered_oof_metrics"]["proxy_wr"],
                "filtered_oof_proxy_trades": payload["filtered_oof_metrics"]["proxy_trades"],
                "direction_oos_bacc": payload["direction_oos_metrics"]["balanced_accuracy"],
                "direction_oos_auc": payload["direction_oos_metrics"]["ovr_auc"],
                "direction_oos_proxy_wr": payload["direction_oos_metrics"]["proxy_wr"],
                "direction_oos_proxy_trades": payload["direction_oos_metrics"]["proxy_trades"],
                "filtered_oos_bacc": payload["filtered_oos_metrics"]["balanced_accuracy"],
                "filtered_oos_auc": payload["filtered_oos_metrics"]["ovr_auc"],
                "filtered_oos_proxy_wr": payload["filtered_oos_metrics"]["proxy_wr"],
                "filtered_oos_proxy_trades": payload["filtered_oos_metrics"]["proxy_trades"],
                **payload["delta_vs_global_volatility_pca06"],
            }
        )
    rows.sort(key=lambda r: (float(r["filtered_oos_bacc"]), float(r["filtered_oos_proxy_wr"] or 0.0)), reverse=True)
    report = {
        "model_id": MODEL_ID,
        "design": "Same Regime3 router and same Direction/Quality input builder as the CatBoost expert-DQ baseline, but each expert Direction Head and Quality Head is replaced by a TabM-style BatchEnsemble PyTorch tabular classifier. Replay/risk is unchanged.",
        "baseline": hard.BASELINE_VOLPCA06,
        "tabm_config": CFG.__dict__,
        "ranking": rows,
        "selected_by_filtered_oos_bacc": rows[0]["variant"],
        "variants": variants,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    pd.DataFrame(rows).to_csv(OUT_DIR / "ranking.csv", index=False)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "ranking": rows}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
