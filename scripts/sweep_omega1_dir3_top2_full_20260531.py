#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from build_omega1_dir3_finpaper_features_20260531 import (
    SEQ_COLS,
    VsnLstm,
    _add_label,
    _read_base,
    _score_torch,
    _seq_matrix,
)
from build_omega1_dir3_remaining_features_20260531 import (
    PATCH_COLS,
    _patch_features,
    _score_model,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = ROOT / "data/splits/year_oos"
DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
DEFAULT_OUT_ROOT = ROOT / "data/ensemble/supervised"
DEFAULT_REPORT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_dir3_top2_full_sweep_20260531"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _metrics(scored: pd.DataFrame, labels: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    df = scored.merge(labels[["timestamp", "zigzag_action"]], on="timestamp", how="inner", validate="one_to_one")
    y = df["zigzag_action"].astype(int).to_numpy()
    proba = df[cols].to_numpy(float)
    pred = proba.argmax(axis=1)
    trade = pred != 0
    return {
        "rows": int(len(df)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "ovr_auc": float(roc_auc_score(y, proba, multi_class="ovr", labels=[0, 1, 2])),
        "pred_counts": {str(i): int(v) for i, v in enumerate(np.bincount(pred, minlength=3))},
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y, minlength=3))},
        "proxy_trades": int(trade.sum()),
        "proxy_long_trades": int((pred == 1).sum()),
        "proxy_short_trades": int((pred == 2).sum()),
        "proxy_trade_rate": float(trade.mean()),
        "proxy_wr": float((pred[trade] == y[trade]).mean()) if trade.any() else None,
    }


def _selection_score(m: dict[str, Any]) -> float:
    return float(m["balanced_accuracy"]) + 0.15 * float(m["ovr_auc"]) + 0.05 * float(m["proxy_wr"] or 0.0)


def _fit_hgb_variant(train_x: pd.DataFrame, train_y: np.ndarray, *, seed: int, max_iter: int, lr: float, leaf_nodes: int, l2: float) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "hgb",
                HistGradientBoostingClassifier(
                    max_iter=int(max_iter),
                    learning_rate=float(lr),
                    max_leaf_nodes=int(leaf_nodes),
                    l2_regularization=float(l2),
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=35,
                    random_state=int(seed),
                ),
            ),
        ]
    )
    model.fit(train_x, train_y, hgb__sample_weight=compute_sample_weight(class_weight="balanced", y=train_y))
    return model


def _write_feature_set(out_root: Path, name: str, scored_2025: pd.DataFrame, scored_2026: pd.DataFrame) -> dict[str, str]:
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    p25 = out_dir / f"training_features_2025_{name}.csv"
    p26 = out_dir / f"training_features_2026_rebuilt_{name}.csv"
    scored_2025.to_csv(p25, index=False)
    scored_2026.to_csv(p26, index=False)
    return {"features_2025": str(p25), "features_2026": str(p26)}


def _append_vsn_proba(timestamps: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": timestamps.to_numpy()})
    out["dir3_vsnlstm_h6_fl_prob"] = proba[:, 0]
    out["dir3_vsnlstm_h6_up_prob"] = proba[:, 1]
    out["dir3_vsnlstm_h6_dn_prob"] = proba[:, 2]
    out["dir3_vsnlstm_h6_confidence"] = proba.max(axis=1)
    out["dir3_vsnlstm_h6_side_edge"] = proba[:, 1] - proba[:, 2]
    out["dir3_vsnlstm_h6_trade_prob"] = proba[:, 1] + proba[:, 2]
    return out


def _predict_vsn(model: torch.nn.Module, frame: pd.DataFrame, cols: list[str], scaler: Any, batch_size: int) -> pd.DataFrame:
    seq, _ = _seq_matrix(frame, cols, scaler)
    device = next(model.parameters()).device
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(seq), int(batch_size)):
            xb = torch.tensor(seq[start : start + int(batch_size)], dtype=torch.float32, device=device)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return _append_vsn_proba(frame["timestamp"].iloc[71:], np.vstack(probs))


def _train_vsn_with_early_stop(
    train: pd.DataFrame,
    cols: list[str],
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    val_start: str,
) -> tuple[torch.nn.Module, Any, list[dict[str, Any]]]:
    seq, scaler = _seq_matrix(train, cols, None)
    y = train["zigzag_action"].astype(int).to_numpy()[71:]
    ts = pd.to_datetime(train["timestamp"].iloc[71:]).reset_index(drop=True)
    val_mask = ts >= pd.Timestamp(val_start)
    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("empty VSN train/validation split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = VsnLstm(len(cols)).to(device)
    counts = np.bincount(y[train_idx], minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device), label_smoothing=0.02)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-3)
    x_t = torch.tensor(seq, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    rng = np.random.default_rng(int(seed))
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    bad = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        losses: list[float] = []
        order = rng.permutation(train_idx)
        for start in range(0, len(order), int(batch_size)):
            idx = order[start : start + int(batch_size)]
            xb = x_t[idx].to(device, non_blocking=True)
            yb = y_t[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(val_idx), int(batch_size * 2)):
                idx = val_idx[start : start + int(batch_size * 2)]
                probs.append(torch.softmax(model(x_t[idx].to(device, non_blocking=True)), dim=1).cpu().numpy())
        proba = np.vstack(probs)
        pred = proba.argmax(axis=1)
        trade = pred != 0
        val_metrics = {
            "balanced_accuracy": float(balanced_accuracy_score(y[val_idx], pred)),
            "macro_f1": float(f1_score(y[val_idx], pred, average="macro")),
            "ovr_auc": float(roc_auc_score(y[val_idx], proba, multi_class="ovr", labels=[0, 1, 2])),
            "proxy_trades": int(trade.sum()),
            "proxy_wr": float((pred[trade] == y[val_idx][trade]).mean()) if trade.any() else None,
        }
        row = {"epoch": int(epoch), "loss": float(np.mean(losses)), "val": val_metrics}
        history.append(row)
        print(json.dumps({"vsn_seed": int(seed), **row}, ensure_ascii=False), flush=True)
        score = _selection_score({**val_metrics, "rows": len(val_idx)})
        if score > best_score + 1e-4:
            best_score = score
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= int(patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, history


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seeds", default="20260531,20260532,20260533")
    parser.add_argument("--vsn-max-epochs", type=int, default=12)
    parser.add_argument("--vsn-patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--val-start", default="2024-10-01")
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    frames = {year: _add_label(_read_base(args.split_dir, year), args.label_dir, year) for year in [2024, 2025, 2026]}
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]

    patch_cols = [c for c in PATCH_COLS if c in frames[2024].columns]
    patch_x = {year: _patch_features(frames[year], patch_cols) for year in [2024, 2025, 2026]}
    patch_train = patch_x[2024].merge(frames[2024][["timestamp", "zigzag_action"]], on="timestamp", how="inner", validate="one_to_one")
    patch_y = patch_train["zigzag_action"].astype(int).to_numpy()
    patch_train_x = patch_train.drop(columns=["timestamp", "zigzag_action"])
    patch_grid = [
        {"max_iter": 260, "lr": 0.035, "leaf_nodes": 31, "l2": 0.08},
        {"max_iter": 380, "lr": 0.025, "leaf_nodes": 31, "l2": 0.10},
        {"max_iter": 320, "lr": 0.030, "leaf_nodes": 47, "l2": 0.12},
    ]
    patch_runs: list[dict[str, Any]] = []
    best_patch: dict[str, Any] | None = None
    for seed in seeds:
        for cfg in patch_grid:
            model = _fit_hgb_variant(patch_train_x, patch_y, seed=seed, **cfg)
            scored_2025 = _score_model("patch", model, patch_x[2025])
            scored_2026 = _score_model("patch", model, patch_x[2026])
            m25 = _metrics(scored_2025, frames[2025], ["dir3_patch_h6_fl_prob", "dir3_patch_h6_up_prob", "dir3_patch_h6_dn_prob"])
            m26 = _metrics(scored_2026, frames[2026], ["dir3_patch_h6_fl_prob", "dir3_patch_h6_up_prob", "dir3_patch_h6_dn_prob"])
            run = {"seed": seed, "config": cfg, "metrics_2025": m25, "metrics_2026": m26, "selection_score": _selection_score(m25)}
            patch_runs.append(run)
            print(json.dumps({"patch_run": run}, ensure_ascii=False, default=_json_default), flush=True)
            if best_patch is None or run["selection_score"] > best_patch["selection_score"]:
                best_patch = {**run, "model": model, "scored_2025": scored_2025, "scored_2026": scored_2026}
    assert best_patch is not None
    patch_name = "omega1_dir3_patch_full_20260531"
    patch_paths = _write_feature_set(args.out_root, patch_name, best_patch["scored_2025"], best_patch["scored_2026"])
    patch_dir = args.out_root / patch_name
    joblib.dump({"model": best_patch["model"], "source_cols": patch_cols, "run": {k: v for k, v in best_patch.items() if k not in {"model", "scored_2025", "scored_2026"}}}, patch_dir / "patch_full_hgb.joblib")

    vsn_cols = [c for c in SEQ_COLS if c in frames[2024].columns]
    vsn_runs: list[dict[str, Any]] = []
    best_vsn: dict[str, Any] | None = None
    for seed in seeds:
        model, scaler, history = _train_vsn_with_early_stop(
            frames[2024],
            vsn_cols,
            seed=seed,
            max_epochs=args.vsn_max_epochs,
            patience=args.vsn_patience,
            batch_size=args.batch_size,
            lr=6e-4,
            val_start=args.val_start,
        )
        scored_2025 = _predict_vsn(model, frames[2025], vsn_cols, scaler, args.batch_size)
        scored_2026 = _predict_vsn(model, frames[2026], vsn_cols, scaler, args.batch_size)
        m25 = _metrics(scored_2025, frames[2025], ["dir3_vsnlstm_h6_fl_prob", "dir3_vsnlstm_h6_up_prob", "dir3_vsnlstm_h6_dn_prob"])
        m26 = _metrics(scored_2026, frames[2026], ["dir3_vsnlstm_h6_fl_prob", "dir3_vsnlstm_h6_up_prob", "dir3_vsnlstm_h6_dn_prob"])
        run = {"seed": seed, "metrics_2025": m25, "metrics_2026": m26, "selection_score": _selection_score(m25), "history": history}
        vsn_runs.append(run)
        print(json.dumps({"vsn_run": {k: v for k, v in run.items() if k != "history"}}, ensure_ascii=False, default=_json_default), flush=True)
        if best_vsn is None or run["selection_score"] > best_vsn["selection_score"]:
            best_vsn = {**run, "model": model, "scaler": scaler, "scored_2025": scored_2025, "scored_2026": scored_2026}
    assert best_vsn is not None
    vsn_name = "omega1_dir3_vsnlstm_full_20260531"
    vsn_paths = _write_feature_set(args.out_root, vsn_name, best_vsn["scored_2025"], best_vsn["scored_2026"])
    vsn_dir = args.out_root / vsn_name
    torch.save({"model_state": best_vsn["model"].state_dict(), "cols": vsn_cols, "run": {k: v for k, v in best_vsn.items() if k not in {"model", "scaler", "scored_2025", "scored_2026"}}}, vsn_dir / "vsnlstm_full.pt")
    joblib.dump({"scaler": best_vsn["scaler"], "cols": vsn_cols}, vsn_dir / "vsnlstm_scaler.joblib")

    report = {
        "model_id": "omega1_dir3_top2_full_sweep_20260531",
        "train_year": 2024,
        "selection_year": 2025,
        "oos_year": 2026,
        "label_source": "zigzag_action",
        "patch": {
            "feature_paths": patch_paths,
            "runs": patch_runs,
            "best": {k: v for k, v in best_patch.items() if k not in {"model", "scored_2025", "scored_2026"}},
        },
        "vsnlstm": {
            "feature_paths": vsn_paths,
            "runs": vsn_runs,
            "best": {k: v for k, v in best_vsn.items() if k not in {"model", "scaler", "scored_2025", "scored_2026"}},
        },
    }
    out_path = args.report_dir / "top2_full_sweep_summary.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(out_path), "patch_best": report["patch"]["best"], "vsnlstm_best": report["vsnlstm"]["best"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
