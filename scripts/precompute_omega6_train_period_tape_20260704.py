#!/usr/bin/env python3
"""Priority-2 support: build a decision tape (same schema as omega6_decision_tape_20260704)
for the L2 TRAIN period (2025-01-02..2025-09-30, i.e. before SPLIT_TS) using the ORIGINAL
zigzag-swing-label primary/fallback bundles (the frozen v2 winner's own L2, NOT the failed
barrier-matched retrain). The existing decision tape only covers CONTEXT_BARS before
VAL_START through OOS_END, so it doesn't cover this earlier train-only window needed to fit
a NEW L4 risk sizing sidecar against the frozen v2 winner's own entry set (persistence=3,
threshold=0.58, ATR 15x/5x barriers, cooldown=12) instead of the v1 fixed-barrier baseline the
existing sidecar (tmp/causal_regen_20260516/omega6_risk_sidecar_20260703/risk_sidecar.pkl) was
trained on.

Batched vectorized inference (same technique as the other precompute_*_tape scripts).
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

OUT_PATH = ROOT / "tmp/causal_regen_20260516/omega6_train_period_decision_tape_20260704/tape.parquet"
PRIMARY_BUNDLE = ROOT / "tmp/causal_regen_20260516/omega6_true_3head_tabm_20260703_primary/true_3head_tabm_bundle.pt"
FALLBACK_BUNDLE = ROOT / "tmp/causal_regen_20260516/omega6_true_3head_tabm_20260703_fallback/true_3head_tabm_bundle.pt"
DEFAULT_THRESHOLD = 0.45
TRAIN_START = pd.Timestamp("2025-01-02")
TRAIN_END = pd.Timestamp("2025-09-30 23:59:59")
ATR_WINDOW = 192


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


def _atr_pct(frame: pd.DataFrame, window: int) -> np.ndarray:
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).rolling(window=max(int(window), 1), min_periods=1).mean().to_numpy(dtype=np.float64)
    return atr / np.maximum(close, 1e-12)


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
    frame = bt._load_combined_frame()
    start_idx = int(frame["timestamp"].searchsorted(TRAIN_START, side="left"))
    end_idx = int(frame["timestamp"].searchsorted(TRAIN_END, side="right"))
    frame_slice = frame.iloc[start_idx:end_idx].reset_index(drop=True)
    print(f"train period rows: {len(frame_slice)} ({frame_slice['timestamp'].min()}..{frame_slice['timestamp'].max()})", flush=True)

    route_probs = frame_slice[ROUTE_COLS].to_numpy(dtype=np.float64)
    row_sum = route_probs.sum(axis=1, keepdims=True)
    if not np.isfinite(route_probs).all() or (row_sum <= 0.0).any():
        raise RuntimeError("invalid Regime3 route probabilities")
    route_probs = route_probs / row_sum
    expert_idx = route_probs.argmax(axis=1)
    route_confidence = route_probs[np.arange(len(route_probs)), expert_idx]
    sorted_p = np.sort(route_probs, axis=1)
    route_margin = sorted_p[:, -1] - sorted_p[:, -2]
    expert_names = [EXPERTS[i] for i in expert_idx]

    out = frame_slice[["timestamp", "open", "high", "low", "close", "jump_flag", "evt_tail_flag", "jump_z"]].copy()
    out["atr_pct"] = _atr_pct(frame_slice, ATR_WINDOW)
    out["i"] = np.arange(start_idx, end_idx)

    for prefix, bundle_path in (("primary", PRIMARY_BUNDLE), ("fallback", FALLBACK_BUNDLE)):
        print(f"predicting {prefix} from {bundle_path.parent.name}", flush=True)
        experts = _load_bundle(bundle_path, device)
        dir_p, qual_p = _bundle_predict(frame_slice, experts, expert_idx, device)
        del experts
        dir_action = dir_p.argmax(axis=1)
        n = len(frame_slice)
        qual_for_action = np.where(dir_action > 0, qual_p[np.arange(n), dir_action], qual_p[:, 0])
        final_action = np.where((dir_action != 0) & (qual_for_action >= DEFAULT_THRESHOLD), dir_action, 0)
        side = np.where(final_action == 1, 1, np.where(final_action == 2, -1, 0))
        out[f"{prefix}_action"] = final_action
        out[f"{prefix}_side"] = side
        out[f"{prefix}_expert"] = expert_names
        out[f"{prefix}_route_confidence"] = route_confidence
        out[f"{prefix}_route_margin"] = route_margin
        out[f"{prefix}_dir_p_cash"] = dir_p[:, 0]
        out[f"{prefix}_dir_p_long"] = dir_p[:, 1]
        out[f"{prefix}_dir_p_short"] = dir_p[:, 2]
        out[f"{prefix}_quality_p_cash"] = qual_p[:, 0]
        out[f"{prefix}_quality_p_long"] = qual_p[:, 1]
        out[f"{prefix}_quality_p_short"] = qual_p[:, 2]
        out[f"{prefix}_quality_score"] = np.where(final_action != 0, qual_for_action, 0.0)
        out[f"{prefix}_confidence"] = dir_p.max(axis=1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"wrote {len(out)} rows to {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
