#!/usr/bin/env python3
"""Permutation feature importance for the Omega4.6.1 3-head TabM components (h48qual, zig075)
on the fixed Fresh-Forward VAL window (2025-09-01..2025-12-31, per CLAUDE.md).

READ-ONLY diagnostic. Loads the frozen `true_3head_tabm_bundle.pt` bundles for h48qual and
zig075 (torch.load, no training), builds the VAL feature dataframe using the SAME data path
already used elsewhere in this project for VAL scoring (`data/splits/year_oos/training_features_2025.csv`
+ regime3-current wide24 overlay, see scripts/replay_omega4_6_1_greedy_val_20260706.py::load_val_frame),
and computes per-feature permutation importance against each of the two heads that are actually
trained with a supervision signal in this architecture (`direction` and `quality`; both are
trained against the same `zigzag_action` 3-class label per
scripts/train_eval_omega1_2_tabm_3head_20260603.py::_fit_expert_3head).

Method: for each of the 102 base_cols, shuffle that column's values across the VAL frame (fixed
seed, single permutation per feature -- no repeats, for runtime reasons; documented as a
limitation), re-run inference through the SAME (already-computed) per-row expert routing
(Regime3-current bull/bear/chop argmax, held fixed -- routing is an external mechanism, not a
model input we are attributing here), and measure the drop in macro one-vs-rest ROC-AUC of the
head's softmax output vs the true zigzag_action label, relative to the unpermuted baseline AUC.
importance = baseline_auc - permuted_auc (higher = more important; near-zero/negative = removal
candidate).

Does NOT modify/retrain/promote anything. Does not touch data/ensemble/ckpt/ or any live-wired
file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

BASE_2025 = ROOT / "data/splits/year_oos/training_features_2025.csv"
WIDE24_2025 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2025_regime3_current_sensitive_hmm_wide24.csv"
LABELS_2025 = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531/zigzag_action_labels_2025.csv"

VAL_START, VAL_END = "2025-09-01", "2025-12-31 23:59:59"

COMPONENTS = {
    "h48qual": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/true_3head_tabm_bundle.pt",
    "zig075": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt",
}

OUT_JSON = ROOT / "data/ensemble/reports/omega4_6_1_feature_importance_permutation_20260724.json"
OUT_CSV = ROOT / "data/ensemble/reports/omega4_6_1_feature_importance_permutation_20260724.csv"

SEED = 20260724
DEVICE = torch.device("cpu")


def load_val_frame() -> pd.DataFrame:
    frame = pd.read_csv(BASE_2025, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    overlay = pd.read_csv(WIDE24_2025, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    frame = frame[(frame["timestamp"] >= VAL_START) & (frame["timestamp"] <= VAL_END)].reset_index(drop=True)
    return frame


def load_labels() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_2025, usecols=["timestamp", "zigzag_action"], low_memory=False)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    return labels


@torch.no_grad()
def _forward_expert(model: parent.ThreeHeadTabM, x_std: np.ndarray) -> dict[str, np.ndarray]:
    chunks = {"direction": [], "quality": []}
    for start in range(0, len(x_std), 8192):
        xb = torch.from_numpy(x_std[start : start + 8192]).to(DEVICE)
        out = model(xb)
        chunks["direction"].append(torch.softmax(out["direction"], dim=-1).mean(dim=1).cpu().numpy())
        chunks["quality"].append(torch.softmax(out["quality"], dim=-1).mean(dim=1).cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in chunks.items()}


def _routed_predict(
    x_raw: np.ndarray,
    columns: list[str],
    route: np.ndarray,
    loaded_models: dict[str, tuple[parent.ThreeHeadTabM, dict[str, Any]]],
) -> dict[str, np.ndarray]:
    """x_raw: (n, len(columns)) raw (pre-standardization) float32 matrix, columns aligned to
    each expert's scaler['columns'] order (base_cols + POS_COLS, identical for bull/bear/chop
    within one bundle by construction -- verified at load time)."""
    n = x_raw.shape[0]
    direction = np.zeros((n, 3), dtype=np.float64)
    quality = np.zeros((n, 3), dtype=np.float64)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        mask = route == idx
        if not bool(mask.any()):
            continue
        model, scaler = loaded_models[expert]
        if list(scaler["columns"]) != columns:
            raise RuntimeError(f"{expert}: scaler column order mismatch with supplied raw matrix")
        mean = scaler["mean"]
        std = scaler["std"]
        x_std = ((x_raw[mask] - mean) / std).astype(np.float32)
        out = _forward_expert(model, x_std)
        direction[mask] = out["direction"]
        quality[mask] = out["quality"]
    return {"direction": direction, "quality": quality}


def _auc_ovr(y_true: np.ndarray, proba: np.ndarray) -> float:
    present = sorted(np.unique(y_true).tolist())
    if len(present) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    except ValueError:
        return float("nan")


def run_component(name: str, bundle_path: Path, frame: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    print(f"[{name}] loading bundle {bundle_path}", flush=True)
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    base_cols: list[str] = list(bundle["base_cols"])
    pos_cols: list[str] = list(bundle["pos_cols"])
    models = bundle["models"]
    loaded_models = parent._load_payloads(models, device=DEVICE)

    x_df = parent._base_input(frame, base_cols)  # columns: base_cols (reindexed) + pos_cols (=0)
    columns = list(x_df.columns)
    if columns != base_cols + pos_cols:
        raise RuntimeError(f"{name}: unexpected _base_input column order")
    x_raw = x_df.to_numpy(dtype=np.float32)
    route = hard._route_id(frame)

    baseline = _routed_predict(x_raw, columns, route, loaded_models)
    base_auc_dir = _auc_ovr(y, baseline["direction"])
    base_auc_qual = _auc_ovr(y, baseline["quality"])
    print(f"[{name}] baseline AUC direction={base_auc_dir:.4f} quality={base_auc_qual:.4f} n={len(y)}", flush=True)

    rng = np.random.default_rng(SEED)
    rows = []
    for j, col in enumerate(base_cols):
        perm = rng.permutation(len(x_raw))
        x_perm = x_raw.copy()
        x_perm[:, j] = x_raw[perm, j]
        pred = _routed_predict(x_perm, columns, route, loaded_models)
        auc_dir = _auc_ovr(y, pred["direction"])
        auc_qual = _auc_ovr(y, pred["quality"])
        rows.append(
            {
                "component": name,
                "feature": col,
                "baseline_auc_direction": base_auc_dir,
                "permuted_auc_direction": auc_dir,
                "importance_direction": base_auc_dir - auc_dir,
                "baseline_auc_quality": base_auc_qual,
                "permuted_auc_quality": auc_qual,
                "importance_quality": base_auc_qual - auc_qual,
            }
        )
        if (j + 1) % 20 == 0 or (j + 1) == len(base_cols):
            print(f"[{name}] permuted {j + 1}/{len(base_cols)}", flush=True)

    return {
        "component": name,
        "bundle": str(bundle_path),
        "n_rows": int(len(y)),
        "baseline_auc_direction": base_auc_dir,
        "baseline_auc_quality": base_auc_qual,
        "rows": rows,
    }


def main() -> int:
    print("Loading VAL frame + labels...", flush=True)
    frame = load_val_frame()
    labels = load_labels()
    merged = frame[["timestamp"]].merge(labels, on="timestamp", how="inner")
    frame = frame[frame["timestamp"].isin(merged["timestamp"])].reset_index(drop=True)
    y = pd.to_numeric(merged.set_index("timestamp").loc[frame["timestamp"], "zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    print(f"VAL frame rows={len(frame)} range={frame['timestamp'].iloc[0]}..{frame['timestamp'].iloc[-1]} label_dist={np.bincount(y).tolist()}", flush=True)

    results: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for name, bundle_path in COMPONENTS.items():
        res = run_component(name, bundle_path, frame, y)
        results[name] = {k: v for k, v in res.items() if k != "rows"}
        all_rows.extend(res["rows"])

    report = {
        "method": "permutation_importance",
        "metric": "roc_auc_score(multi_class='ovr', average='macro') vs zigzag_action (0=cash,1=long,2=short) label; importance = baseline_auc - permuted_auc",
        "val_window": {"start": VAL_START, "end": VAL_END, "rule": "CLAUDE.md Fresh-Forward Validation Rule fixed VAL split"},
        "n_rows": int(len(y)),
        "seed": SEED,
        "permutations_per_feature": 1,
        "routing_held_fixed": True,
        "components": results,
        "data_sources": {
            "base_frame": str(BASE_2025),
            "regime3_wide24_overlay": str(WIDE24_2025),
            "labels": str(LABELS_2025),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_JSON}", flush=True)
    print(f"Wrote {OUT_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
