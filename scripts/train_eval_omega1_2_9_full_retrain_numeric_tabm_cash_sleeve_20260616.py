#!/usr/bin/env python3
from __future__ import annotations

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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_8_full_retrain_numeric_cash_sleeve_20260616 as hgb_exp  # noqa: E402


MODEL_ID = "omega1_2_9_full_retrain_numeric_tabm_cash_sleeve_20260616"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


class TabMRegressor(nn.Module):
    def __init__(self, n_features: int, *, k: int = 4, hidden: int = 96, layers: int = 3, dropout: float = 0.08) -> None:
        super().__init__()
        self.k = int(k)
        self.n_features = int(n_features)
        self.input_scale = nn.Parameter(torch.randn(self.k, self.n_features) * 0.03 + 1.0)
        self.input_bias = nn.Parameter(torch.zeros(self.k, self.n_features))
        self.in_proj = nn.Linear(self.n_features, int(hidden))
        self.blocks = nn.ModuleList(nn.Linear(int(hidden), int(hidden)) for _ in range(max(0, int(layers) - 1)))
        self.expert_scale = nn.ParameterList(
            nn.Parameter(torch.randn(self.k, int(hidden)) * 0.03 + 1.0) for _ in range(max(0, int(layers) - 1))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(int(hidden)) for _ in range(max(0, int(layers))))
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(int(hidden), 2)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)).mean(dim=1)


def _standardize_fit(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    out = (arr - mean) / std
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized TabM matrix")
    return out.astype(np.float32), {"mean": mean, "std": std, "columns": list(x.columns)}


def _standardize_apply(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    if list(x.columns) != list(scaler["columns"]):
        raise RuntimeError("TabM feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = (arr - scaler["mean"]) / scaler["std"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized TabM inference matrix")
    return out.astype(np.float32)


def _train_tabm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_pred: pd.DataFrame,
    *,
    seed: int,
    epochs: int = 36,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _seed(seed)
    x_all, scaler = _standardize_fit(x_train)
    x_pred_np = _standardize_apply(x_pred, scaler)
    y = np.asarray(y_train, dtype=np.float32)
    n = len(y)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabMRegressor(x_all.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=2.0e-4)
    loss_fn = nn.SmoothL1Loss()
    ds = TensorDataset(torch.from_numpy(x_all[train_idx]), torch.from_numpy(y[train_idx]))
    dl = DataLoader(ds, batch_size=2048, shuffle=True, drop_last=False)
    x_val_t = torch.from_numpy(x_all[val_idx]).to(device) if len(val_idx) else None
    y_val_t = torch.from_numpy(y[val_idx]).to(device) if len(val_idx) else None
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t), y_val_t).detach().cpu()) if x_val_t is not None and y_val_t is not None and len(val_idx) else 0.0
        if val_loss < best_loss - 1.0e-6:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 6:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_train = model(torch.from_numpy(x_all).to(device)).detach().cpu().numpy().astype(np.float64)
        pred = model(torch.from_numpy(x_pred_np).to(device)).detach().cpu().numpy().astype(np.float64)
    return pred_train, pred, {"epochs": int(last_epoch), "best_val_loss": float(best_loss), "device": str(device)}


def _fit_predict_lower_bound_tabm(
    x_val: pd.DataFrame,
    x_oos: pd.DataFrame,
    labels: pd.DataFrame,
    long_col: str,
    short_col: str,
    *,
    seed: int,
    cal_q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    idx = labels["i"].to_numpy(dtype=np.int64)
    y = np.column_stack(
        [
            labels[long_col].to_numpy(dtype=np.float32),
            labels[short_col].to_numpy(dtype=np.float32),
        ]
    )
    val_long = np.zeros(len(x_val), dtype=np.float64)
    val_short = np.zeros(len(x_val), dtype=np.float64)
    folds_meta: list[dict[str, Any]] = []
    for fold_id, (tr_pos, va_pos) in enumerate(hgb_exp._chron_folds(idx)):
        tr = tr_pos
        va = va_pos
        pred_train, pred_va, fit_diag = _train_tabm(
            x_val.iloc[tr],
            y[np.searchsorted(idx, tr)],
            x_val.iloc[va],
            seed=seed + fold_id * 10,
            epochs=32,
        )
        target_train = y[np.searchsorted(idx, tr)]
        ql = float(np.quantile(np.abs(target_train[:, 0] - pred_train[:, 0]), cal_q))
        qs = float(np.quantile(np.abs(target_train[:, 1] - pred_train[:, 1]), cal_q))
        val_long[va] = pred_va[:, 0] - ql
        val_short[va] = pred_va[:, 1] - qs
        folds_meta.append({"fold": int(fold_id), "train_rows": int(len(tr)), "val_rows": int(len(va)), "long_abs_resid_q": ql, "short_abs_resid_q": qs, "fit": fit_diag})

    pred_train, pred_oos, final_diag = _train_tabm(x_val.iloc[idx], y, x_oos, seed=seed + 101, epochs=40)
    ql = float(np.quantile(np.abs(y[:, 0] - pred_train[:, 0]), cal_q))
    qs = float(np.quantile(np.abs(y[:, 1] - pred_train[:, 1]), cal_q))
    oos_long = pred_oos[:, 0] - ql
    oos_short = pred_oos[:, 1] - qs
    diag = {
        "model": "TabMRegressor",
        "target_cols": [long_col, short_col],
        "cal_q": float(cal_q),
        "folds": folds_meta,
        "final_long_abs_resid_q": ql,
        "final_short_abs_resid_q": qs,
        "final_fit": final_diag,
    }
    return val_long, val_short, oos_long, oos_short, diag


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "build_payloads", "model_id": MODEL_ID}, ensure_ascii=True), flush=True)
    val_payload, oos_payload, meta = hgb_exp._build_payloads()
    x_val = val_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_oos = oos_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fee = float(meta["fee"])
    slip = float(meta["slip"])
    base_val = hgb_exp.omega._metrics(val_payload["frame"], val_payload["dec"], fee=fee, slip=slip, cost_mult=3.0)
    base_oos = hgb_exp.omega._metrics(oos_payload["frame"], oos_payload["dec"], fee=fee, slip=slip, cost_mult=3.0)
    base_val_sleeve = {**base_val, "primary_entries": base_val["long_entries"] + base_val["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}
    base_oos_sleeve = {**base_oos, "primary_entries": base_oos["long_entries"] + base_oos["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}

    path_labels, path_diag = hgb_exp._path_label_table(val_payload, hgb_exp.RISK)
    ev_labels, ev_diag = hgb_exp._utility_from_path_labels(path_labels, hgb_exp.RISK, {"stop_penalty": 0.0, "mae_penalty": 0.0, "time_penalty": 0.0})
    diagnostics: dict[str, Any] = {
        "mode": "full_retrain_parent_numeric_tabm_cash_sleeve",
        "baseline_model_id": hgb_exp.BASELINE_ID,
        "parent_artifact": meta["parent_dir"],
        "risk": hgb_exp.asdict(hgb_exp.RISK),
        "feature_count": int(x_val.shape[1]),
        "features": list(x_val.columns),
        "baseline": {"validation": base_val_sleeve, "oos": base_oos_sleeve},
        "path_labels": path_diag,
        "ev_labels": ev_diag,
    }

    utility_preds: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    # Keep the promising numeric utility families from the HGB run; TabM is much slower than HGB.
    utility_cfg_ids = (1, 2)
    for cfg_id in utility_cfg_ids:
        cfg = hgb_exp.UTILITY_CFGS[cfg_id]
        print(json.dumps({"stage": "fit_utility_tabm", "cfg_id": int(cfg_id), "config": cfg}, ensure_ascii=True), flush=True)
        labels, diag = hgb_exp._utility_from_path_labels(path_labels, hgb_exp.RISK, cfg)
        vl, vs, ol, os, fit_diag = _fit_predict_lower_bound_tabm(x_val, x_oos, labels, "long_utility", "short_utility", seed=291000 + cfg_id * 100, cal_q=0.50)
        utility_preds[cfg_id] = (vl, vs, ol, os)
        diagnostics[f"utility_cfg_{cfg_id}"] = {"config": cfg, "labels": diag, "fit": fit_diag}

    rows: list[dict[str, Any]] = [
        {
            "candidate": "full_retrain_primary_only",
            "family": "baseline",
            "utility_cfg_id": None,
            "cal_q": None,
            "ev_min": None,
            "utility_min": None,
            "margin_min": None,
            **hgb_exp.sleeve._metric_row("val", base_val_sleeve),
            **hgb_exp.sleeve._metric_row("oos", base_oos_sleeve),
            "val_delta_pnl": 0.0,
            "oos_delta_pnl": 0.0,
        }
    ]

    for cal_q in (0.50, 0.65):
        print(json.dumps({"stage": "fit_ev_tabm", "cal_q": float(cal_q)}, ensure_ascii=True), flush=True)
        ev_vl, ev_vs, ev_ol, ev_os, ev_fit_diag = _fit_predict_lower_bound_tabm(x_val, x_oos, ev_labels, "long_net", "short_net", seed=290000, cal_q=cal_q)
        diagnostics[f"ev_lower_bound_cal_q{cal_q:.2f}"] = ev_fit_diag
        for ev_min in (0.001, 0.002, 0.004):
            val_ev_a, val_ev_c = hgb_exp._actions_from_scores(ev_vl, ev_vs, ev_min)
            oos_ev_a, oos_ev_c = hgb_exp._actions_from_scores(ev_ol, ev_os, ev_min)
            val_m = hgb_exp.sleeve._metrics_with_fallback(val_payload["frame"], val_payload["dec"], hgb_exp.RISK, val_ev_a, val_ev_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
            oos_m = hgb_exp.sleeve._metrics_with_fallback(oos_payload["frame"], oos_payload["dec"], hgb_exp.RISK, oos_ev_a, oos_ev_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
            ev_name = f"full_retrain_tabm_ev_cal{cal_q:.2f}_ev{ev_min:.3f}"
            rows.append(hgb_exp._metric_row(ev_name, "tabm_ev_lower_bound_only", None, cal_q, ev_min, None, None, val_m, oos_m, base_val_sleeve, base_oos_sleeve))
            for cfg_id, (uvl, uvs, uol, uos) in utility_preds.items():
                for utility_min in (0.0, 0.001, 0.002):
                    for margin_min in (0.0, 0.001):
                        val_a, val_c, val_filter = hgb_exp._apply_agreement(val_ev_a, val_ev_c, uvl, uvs, utility_min=utility_min, margin_min=margin_min)
                        oos_a, oos_c, oos_filter = hgb_exp._apply_agreement(oos_ev_a, oos_ev_c, uol, uos, utility_min=utility_min, margin_min=margin_min)
                        cand = f"full_retrain_tabm_ev_cal{cal_q:.2f}_ev{ev_min:.3f}_numcfg{cfg_id}_u{utility_min:.3f}_m{margin_min:.3f}"
                        diagnostics[f"{cand}_filter"] = {"validation": val_filter, "oos": oos_filter}
                        val_m = hgb_exp.sleeve._metrics_with_fallback(val_payload["frame"], val_payload["dec"], hgb_exp.RISK, val_a, val_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
                        oos_m = hgb_exp.sleeve._metrics_with_fallback(oos_payload["frame"], oos_payload["dec"], hgb_exp.RISK, oos_a, oos_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
                        rows.append(hgb_exp._metric_row(cand, "tabm_numeric_agreement_veto", cfg_id, cal_q, ev_min, utility_min, margin_min, val_m, oos_m, base_val_sleeve, base_oos_sleeve))

    ranking = pd.DataFrame(rows)
    ranking["selection_score_val_only"] = (
        ranking["val_delta_pnl"].fillna(0.0)
        + 0.12 * ranking["val_fallback_entries"].fillna(0.0)
        + 8.0 * ranking["val_wr"].fillna(0.0)
        + 0.20 * ranking["val_mdd"].fillna(0.0)
    )
    ranking = ranking.sort_values(["selection_score_val_only", "val_delta_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "full_retrain_numeric_tabm_cash_sleeve_ranking.csv", index=False)
    hybrid = ranking[ranking["family"].eq("tabm_numeric_agreement_veto")].copy()
    selected = hybrid.iloc[0].to_dict() if len(hybrid) else ranking.iloc[0].to_dict()
    best_oos = hybrid.sort_values(["oos_pnl", "oos_delta_pnl"], ascending=False).iloc[0].to_dict() if len(hybrid) else ranking.sort_values(["oos_pnl", "oos_delta_pnl"], ascending=False).iloc[0].to_dict()
    best_controls = ranking[~ranking["family"].eq("tabm_numeric_agreement_veto")].head(5).to_dict(orient="records")

    blockers: list[str] = []
    if len(hybrid) == 0:
        blockers.append("no TabM numeric hybrid candidates produced")
    bad = [c for c in x_val.columns if c == "tp_sl_action_score" or c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_")]
    if bad:
        blockers.append(f"forbidden feature columns: {bad[:20]}")

    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_full_retrain_tabm_numeric_eval" if not blockers else "redteam_fail",
        "method": "Full-retrained 3-head parent artifact is preserved. Cash sleeve EV lower-bound and numeric utility agreement/veto regressors are TabM-style BatchEnsemble PyTorch regressors.",
        "selection_policy": "hybrid_validation_only_no_oos_selection; TabM EV-only rows are controls, OOS is diagnostic",
        "diagnostics": diagnostics,
        "baseline": {"validation": base_val_sleeve, "oos": base_oos_sleeve},
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "best_tabm_ev_only_controls": best_controls,
        "top20_hybrid": hybrid.head(20).to_dict(orient="records"),
        "top20_all_including_controls": ranking.head(20).to_dict(orient="records"),
        "redteam_pass": not blockers,
        "redteam_blockers": blockers,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "full_retrain_numeric_tabm_cash_sleeve_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "status": report["status"], "selected": selected, "best_oos_diagnostic": best_oos}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
