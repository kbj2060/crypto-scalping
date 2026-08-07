#!/usr/bin/env python3
"""Build a decision tape identical in schema to
tmp/causal_regen_20260516/omega6_decision_tape_20260704/tape.parquet, except the `primary_*`
columns come from a 5-SEED ENSEMBLE (average softmax across independently-trained primary
bundles) instead of a single seed.

Rationale: user asked to retrain L2 to improve win rate. A full architecture/feature/label
change is a much larger undertaking; ensembling across the 5 primary seeds already trained in
this session (original seed 260703 + seed710/711/712/713, all on the identical train-only split
SPLIT_TS=2025-10-01, see scripts/train_eval_omega6_tabm_3head_20260703.py) is a legitimate,
quick, no-lookahead variance-reduction step that can plausibly improve precision (win rate) by
averaging out per-seed noise -- standard bagging/ensembling, not a new architecture.

Direction/quality prediction only depends on the CURRENT row's features (see
Omega6LiveAdapter._predict_parent -- it reads frame.iloc[-1] only), so this script batches all
rows through each seed's per-expert model in one forward pass instead of the slow per-bar window
loop the original tape precompute used -- purely a speed optimization, same causal inputs.

Route confidence/margin/expert assignment come from the Regime3 columns only (seed-independent),
so those are copied unchanged from the existing tape. fallback_* columns are also copied
unchanged (no multi-seed fallback bundles exist).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import backtest_omega6_synthesis_fresh_forward_20260703 as bt  # noqa: E402
import train_eval_omega6_tabm_3head_20260703 as omega6_tabm  # noqa: E402
from trading_bot_modules.omega6_live import EXPERTS, ROUTE_COLS, FORBIDDEN_FEATURE_PREFIXES  # noqa: E402

BASE_TAPE_PATH = ROOT / "tmp/causal_regen_20260516/omega6_decision_tape_20260704/tape.parquet"
OUT_PATH = ROOT / "tmp/causal_regen_20260516/omega6_ensemble5_decision_tape_20260704/tape.parquet"

SEED_BUNDLES = [
    ROOT / "tmp/causal_regen_20260516/omega6_true_3head_tabm_20260703_primary/true_3head_tabm_bundle.pt",
    ROOT / "tmp/causal_regen_20260516/omega6_true_3head_tabm_20260703_seed710/true_3head_tabm_bundle.pt",
    ROOT / "tmp/causal_regen_20260516/omega6_true_3head_tabm_20260703_seed711/true_3head_tabm_bundle.pt",
    ROOT / "tmp/causal_regen_20260516/omega6_true_3head_tabm_20260703_seed712/true_3head_tabm_bundle.pt",
    ROOT / "tmp/causal_regen_20260516/omega6_true_3head_tabm_20260703_seed713/true_3head_tabm_bundle.pt",
]


def _load_bundle(path: Path, device: torch.device) -> dict[str, tuple[torch.nn.Module, dict, list[str], list[str]]]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    pos_cols = list(bundle["pos_cols"])
    experts = {}
    for expert, payload_raw in dict(bundle["models"]).items():
        payload = dict(payload_raw)
        input_cols = list(payload["input_columns"])
        bad = sorted(c for c in input_cols if any(str(c).startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES))
        if bad:
            raise RuntimeError(f"{path} {expert} contains forbidden feature prefixes: {bad}")
        cfg = omega6_tabm.ThreeHeadConfig(**dict(payload["config"]))
        model = omega6_tabm.ThreeHeadTabM(int(payload["n_features"]), cfg=cfg).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        experts[str(expert)] = (model, dict(payload["scaler"]), input_cols, pos_cols)
    missing = sorted(set(EXPERTS) - set(experts))
    if missing:
        raise RuntimeError(f"{path} missing experts: {missing}")
    return experts


def _batch_input(frame: pd.DataFrame, input_cols: list[str], pos_cols: list[str]) -> np.ndarray:
    data = {}
    for col in input_cols:
        if col in pos_cols:
            data[col] = np.zeros(len(frame), dtype=np.float32)
            continue
        if col not in frame.columns:
            raise RuntimeError(f"missing input feature: {col}")
        data[col] = pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=np.float32)
    arr = np.column_stack([data[c] for c in input_cols]).astype(np.float32)
    if not np.isfinite(arr).all():
        raise RuntimeError("non-finite input features")
    return arr


@torch.no_grad()
def _bundle_predict(frame: pd.DataFrame, experts: dict, expert_idx: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    n = len(frame)
    dir_out = np.zeros((n, 3), dtype=np.float64)
    qual_out = np.zeros((n, 3), dtype=np.float64)
    for e_i, expert_name in enumerate(EXPERTS):
        mask = expert_idx == e_i
        if not mask.any():
            continue
        model, scaler, input_cols, pos_cols = experts[expert_name]
        sub_frame = frame.loc[mask]
        arr = _batch_input(sub_frame, input_cols, pos_cols)
        cols = list(scaler["columns"])
        if list(input_cols) != cols:
            raise RuntimeError("feature column contract mismatch vs scaler")
        z = (arr - scaler["mean"]) / scaler["std"]
        if not np.isfinite(z).all():
            raise RuntimeError("standardized features non-finite")
        out = model(torch.from_numpy(z.astype(np.float32)).to(device))
        direction = torch.softmax(out["direction"], dim=-1).mean(dim=1).detach().cpu().numpy().astype(np.float64)
        quality = torch.softmax(out["quality"], dim=-1).mean(dim=1).detach().cpu().numpy().astype(np.float64)
        idx = np.where(mask)[0]
        dir_out[idx] = direction
        qual_out[idx] = quality
    return dir_out, qual_out


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_tape = pd.read_parquet(BASE_TAPE_PATH)
    base_tape["timestamp"] = pd.to_datetime(base_tape["timestamp"])
    base_tape = base_tape.sort_values("i").reset_index(drop=True)

    frame = bt._load_combined_frame()
    idx_min, idx_max = int(base_tape["i"].min()), int(base_tape["i"].max())
    frame_slice = frame.iloc[idx_min : idx_max + 1].reset_index(drop=True)
    if len(frame_slice) != len(base_tape):
        raise RuntimeError(f"frame slice length {len(frame_slice)} != base tape length {len(base_tape)}")

    route_probs = frame_slice[ROUTE_COLS].to_numpy(dtype=np.float64)
    row_sum = route_probs.sum(axis=1, keepdims=True)
    if not np.isfinite(route_probs).all() or (row_sum <= 0.0).any():
        raise RuntimeError("invalid Regime3 route probabilities")
    route_probs = route_probs / row_sum
    expert_idx = route_probs.argmax(axis=1)
    route_confidence = route_probs[np.arange(len(route_probs)), expert_idx]
    sorted_p = np.sort(route_probs, axis=1)
    route_margin = sorted_p[:, -1] - sorted_p[:, -2]

    n = len(frame_slice)
    dir_sum = np.zeros((n, 3), dtype=np.float64)
    qual_sum = np.zeros((n, 3), dtype=np.float64)
    for seed_i, bundle_path in enumerate(SEED_BUNDLES):
        print(f"seed {seed_i + 1}/{len(SEED_BUNDLES)}: {bundle_path.parent.name}", flush=True)
        experts = _load_bundle(bundle_path, device)
        dir_p, qual_p = _bundle_predict(frame_slice, experts, expert_idx, device)
        dir_sum += dir_p
        qual_sum += qual_p
        del experts

    dir_avg = dir_sum / len(SEED_BUNDLES)
    qual_avg = qual_sum / len(SEED_BUNDLES)

    out = base_tape.copy()
    out["primary_route_confidence"] = route_confidence
    out["primary_route_margin"] = route_margin
    out["primary_expert"] = [EXPERTS[i] for i in expert_idx]
    out["primary_dir_p_cash"] = dir_avg[:, 0]
    out["primary_dir_p_long"] = dir_avg[:, 1]
    out["primary_dir_p_short"] = dir_avg[:, 2]
    out["primary_quality_p_cash"] = qual_avg[:, 0]
    out["primary_quality_p_long"] = qual_avg[:, 1]
    out["primary_quality_p_short"] = qual_avg[:, 2]
    dir_action = dir_avg.argmax(axis=1)
    qual_for_action = np.where(dir_action > 0, qual_avg[np.arange(n), dir_action], qual_avg[:, 0])
    default_threshold = 0.45
    final_action = np.where((dir_action != 0) & (qual_for_action >= default_threshold), dir_action, 0)
    out["primary_action"] = final_action
    out["primary_side"] = np.where(final_action == 1, 1, np.where(final_action == 2, -1, 0))
    out["primary_quality_score"] = np.where(final_action != 0, qual_for_action, 0.0)
    out["primary_confidence"] = dir_avg.max(axis=1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"wrote {len(out)} rows to {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
