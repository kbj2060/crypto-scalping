#!/usr/bin/env python3
"""Build a decision tape (v2-replay-compatible schema) for the m7-free Omega6 L2 retrain
(scripts/train_eval_omega6_nom7_tabm_3head_20260704.py), covering context + validation
(2025-10-01..12-31) + the full extended fresh window (2026-01-01..06-30).

Frame = 2025 alpha7 candidates (train, full year, unchanged -- has all needed columns as a
superset) concatenated with the NEW extended 2026 frame
(tmp/causal_regen_20260516/extended_eval_frame_nom7_20260704/frame.parquet, built without
SevenModelEnsemble and without the 6 other drift-confirmed columns). The m7-free bundles' 123
required non-position input columns are confirmed present in both halves.

2026-03-02 onward is a genuinely untouched window: no architecture search, no threshold sweep,
no prior model has ever scored it. It is scored exactly once, after gates pass on validation.
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

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega6_nom7_tabm_3head_20260704 as nom7_tabm  # noqa: E402
from trading_bot_modules.omega6_live import EXPERTS, ROUTE_COLS, FORBIDDEN_FEATURE_PREFIXES  # noqa: E402

OUT_PATH = ROOT / "tmp/causal_regen_20260516/omega6_nom7_extended_decision_tape_20260704/tape.parquet"
PRIMARY_BUNDLE = ROOT / "tmp/causal_regen_20260516/omega6_nom7_3head_tabm_20260704_primary/true_3head_tabm_bundle.pt"
FALLBACK_BUNDLE = ROOT / "tmp/causal_regen_20260516/omega6_nom7_3head_tabm_20260704_fallback/true_3head_tabm_bundle.pt"
EXTENDED_2026 = ROOT / "tmp/causal_regen_20260516/extended_eval_frame_nom7_20260704/frame.parquet"
DEFAULT_THRESHOLD = 0.45
ATR_WINDOW = 192
CONTEXT_START = pd.Timestamp("2025-09-20")  # ample context before VAL_START for persistence warm-up


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
        cfg = nom7_tabm.ThreeHeadConfig(**dict(payload["config"]))
        model = nom7_tabm.ThreeHeadTabM(int(payload["n_features"]), cfg=cfg).to(device)
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
    atr = pd.Series(tr).rolling(window=window, min_periods=1).mean().to_numpy(dtype=np.float64)
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

    train_2025, _eval_old, _overlay = omega._load_omega_frames()
    train_2025["timestamp"] = pd.to_datetime(train_2025["timestamp"])
    ext_2026 = pd.read_parquet(EXTENDED_2026)
    ext_2026["timestamp"] = pd.to_datetime(ext_2026["timestamp"])

    train_2025 = train_2025[train_2025["timestamp"] >= CONTEXT_START].reset_index(drop=True)
    combined = pd.concat([train_2025, ext_2026], ignore_index=True, sort=False)
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    primary_experts_tmp = _load_bundle(PRIMARY_BUNDLE, torch.device("cpu"))
    needed_cols = sorted({c for _m, _s, cols, pos in primary_experts_tmp.values() for c in cols if c not in pos})
    del primary_experts_tmp
    needed_present = [c for c in needed_cols if c in combined.columns]
    # Drop the handful of warmup-NaN rows at the very start of the 2026 extension (regime3
    # cmamba/stability sidecars need a seq_len-1 lookback INSIDE the 2026 file before their
    # first valid output; these fall at 2026-01-01 00:00..04:55, well before both the
    # validation window and the untouched 2026-03-02+ fresh window, so dropping them affects
    # neither -- not filled/imputed, just excluded).
    before = len(combined)
    combined = combined.dropna(subset=needed_present, how="any").reset_index(drop=True)
    dropped = before - len(combined)
    print(f"dropped {dropped} rows with NaN in required L2 input columns (warmup edge)", flush=True)
    print(f"combined frame: {len(combined)} rows ({combined['timestamp'].min()}..{combined['timestamp'].max()})", flush=True)

    route_probs = combined[ROUTE_COLS].to_numpy(dtype=np.float64)
    row_sum = route_probs.sum(axis=1, keepdims=True)
    if not np.isfinite(route_probs).all() or (row_sum <= 0.0).any():
        raise RuntimeError("invalid Regime3 route probabilities")
    route_probs = route_probs / row_sum
    expert_idx = route_probs.argmax(axis=1)
    route_confidence = route_probs[np.arange(len(route_probs)), expert_idx]
    sorted_p = np.sort(route_probs, axis=1)
    route_margin = sorted_p[:, -1] - sorted_p[:, -2]
    expert_names = [EXPERTS[i] for i in expert_idx]

    out = combined[["timestamp", "open", "high", "low", "close"]].copy()
    for c in ("jump_flag", "evt_tail_flag", "jump_z"):
        out[c] = pd.to_numeric(combined.get(c, 0.0), errors="coerce").fillna(0.0)
    out["atr_pct"] = _atr_pct(combined, ATR_WINDOW)
    out["i"] = np.arange(len(combined))

    for prefix, bundle_path in (("primary", PRIMARY_BUNDLE), ("fallback", FALLBACK_BUNDLE)):
        print(f"predicting {prefix} from {bundle_path.parent.name}", flush=True)
        experts = _load_bundle(bundle_path, device)
        dir_p, qual_p = _bundle_predict(combined, experts, expert_idx, device)
        del experts
        dir_action = dir_p.argmax(axis=1)
        n = len(combined)
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
    print(f"primary_side nonzero pct: {(out['primary_side'] != 0).mean():.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
