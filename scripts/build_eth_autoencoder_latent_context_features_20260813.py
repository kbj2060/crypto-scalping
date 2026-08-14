#!/usr/bin/env python3
"""Odyssey2 priority #3 (docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md).
Grafts the Odyssey(1) autoencoder-latent idea (verify_eth_h48qual_autoencoder_latent_mi_r2_gate_20260812.py
-- 139-column raw pool denoising-autoencoder, latent_dim=16) from its original target (direction/
quality classification, where it numerically beat FINAL12 but still lost to always-short) onto the
risk-sizing GBM via train_eval_omega4_2_risk_sidecar_20260622.py's `--risk-context-feature-dir`
extension point (same mechanism as the ensemble-epistemic script). Odyssey(1)'s own reading of that
result: latent features moved DIRECTION further than FINAL12 but couldn't clear the always-short
bar -- i.e. "some signal, not enough to beat a strong regime beta as a direction filter." Testing it
as a continuous SIZING input (not a hard gate) is a genuinely different question.

Architecture/training recipe copied verbatim from the original script (same POOL construction,
same Autoencoder class, same denoising/early-stopping regime) -- NOT imported as a module because
the original is a flat top-level script (importing it would re-run its own full analysis). Only
difference: the autoencoder is refit here on a TRAIN window causally aligned with the risk sidecar's
OWN _prepare_frames() output (not the original script's fixed 2024-06..2025-09 window) so the latent
vectors can be joined row-for-row onto train_eval_omega4_2_risk_sidecar_20260622's frames without a
row-count mismatch. Pure feature engineering + a small unsupervised autoencoder fit -- the risk
sidecar GBM itself is the only thing "really" retrained downstream.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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

import train_eval_omega4_2_risk_sidecar_20260622 as sidecar_script  # noqa: E402

omega, omega4 = sidecar_script.omega, sidecar_script.omega4

LATENT_DIM = 16
SEED = 260813

DENY_PREFIXES = ("clean_regime4_", "regime4_pred_", "regime3_pred_", "teacher_", "teacher_oof_", "a5dir_")
DENY_TOKENS = ("target", "future", "label", "pnl", "zigzag", "wave3", "tp_sl_action_score")
NON_FEATURE = {"timestamp", "open", "high", "low", "close", "open_btc", "high_btc", "low_btc", "close_btc"}
PRICE_LIKE = ["sum_open_interest_value"]
REPLACE = {
    "funding_pressure": ("funding_pressure_diff1", "diff1"),
    "last_funding_rate": ("last_funding_rate_dt288", "dt288"),
    "squeeze_power": ("squeeze_power_dt288", "dt288"),
    "long_squeeze_risk": ("long_squeeze_risk_dt288", "dt288"),
    "funding_abs": ("funding_abs_dt288", "dt288"),
    "whale_retail_ratio": ("whale_retail_ratio_dt288", "dt288"),
    "count_long_short_ratio": ("count_long_short_ratio_dt288", "dt288"),
    "sum_toptrader_long_short_ratio": ("sum_toptrader_long_short_ratio_dt288", "dt288"),
}

COMPONENT_CONFIG = {
    "h48qual": {
        "train_csv": ROOT / "tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/02_fixed_regime4_state24_sticky090_tp18_sl10_preprocess_2024_to_2025/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv",
        "eval_csv": ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv",
        "direction_label_dir": ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531",
        "quality_mode": "same_as_direction",
    },
    "zig075": {
        "train_csv": ROOT / "tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/02_fixed_regime4_state24_sticky090_tp18_sl10_preprocess_2024_to_2025/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv",
        "eval_csv": ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv",
        "direction_label_dir": ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531",
        "quality_mode": "same_as_direction",
    },
}


def log(msg: str) -> None:
    print(f"[ae_latent] {msg}", flush=True)


def is_candidate(col: str) -> bool:
    if col in NON_FEATURE or col in PRICE_LIKE or col in REPLACE:
        return False
    if any(col.startswith(p) for p in DENY_PREFIXES):
        return False
    if any(t in col for t in DENY_TOKENS):
        return False
    return True


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", choices=list(COMPONENT_CONFIG.keys()), required=True)
    args = ap.parse_args()
    cfg = COMPONENT_CONFIG[args.component]
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = ROOT / f"tmp/causal_regen_20260516/eth_{args.component}_autoencoder_latent_context_20260813"
    out_dir.mkdir(parents=True, exist_ok=True)

    omega.TRAIN_CSV = Path(cfg["train_csv"])
    omega.EVAL_CSV = Path(cfg["eval_csv"])
    log("stage=prepare_frames (sidecar's own frame construction, for row alignment)")
    frames = omega4._prepare_frames(
        disable_tp_sl=False, direction_label_dir=Path(cfg["direction_label_dir"]), quality_mode=str(cfg["quality_mode"]),
        quality_label_dir=None, quality_min_edge=0.0, quality_max_mae=0.0, quality_min_mfe_mae=0.0, quality_max_hold_bars=0,
    )
    train_frame, val_frame, oos_frame = frames["train_raw"], frames["val_raw"], frames["oos_raw"]
    log(f"  train={len(train_frame)} val={len(val_frame)} oos={len(oos_frame)}")

    log("stage=load_panel (data/splits/year_oos/eth_features_2024_2026_analysis.csv, committed, 2024-06..2026-08 coverage)")
    panel = pd.read_csv(ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv", low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    candidate_raw = [c for c in panel.columns if is_candidate(c) and pd.api.types.is_numeric_dtype(panel[c])]
    for raw, (derived, kind) in REPLACE.items():
        if raw not in panel.columns:
            continue
        src = pd.to_numeric(panel[raw], errors="coerce").astype(np.float64)
        if kind == "diff1":
            panel[derived] = src.diff(1).fillna(0.0)
        elif kind == "dt288":
            panel[derived] = (src - src.rolling(288, min_periods=96).mean()).fillna(0.0)
    POOL = sorted(set(candidate_raw) | {d for d, _ in REPLACE.values() if d in panel.columns})
    log(f"  pool size={len(POOL)}")

    def join_pool(frame: pd.DataFrame, split: str) -> pd.DataFrame:
        merged = frame[["timestamp"]].merge(panel[["timestamp", *POOL]], on="timestamp", how="left", validate="one_to_one")
        missing = merged[POOL].isna().any(axis=1).sum()
        if missing:
            raise RuntimeError(f"{split}: {missing} rows have no matching panel timestamp -- pool coverage gap")
        return merged

    train_pool = join_pool(train_frame, "train")
    val_pool = join_pool(val_frame, "validation")
    oos_pool = join_pool(oos_frame, "oos")

    X_train_raw = train_pool[POOL].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_val_raw = val_pool[POOL].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_oos_raw = oos_pool[POOL].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # TRAIN-internal 85/15 chronological split for early stopping (never touches val/oos), same
    # discipline as the original script.
    n_train = len(X_train_raw)
    split_point = int(n_train * 0.85)
    fit_idx, es_idx = np.arange(split_point), np.arange(split_point, n_train)
    log(f"  fit n={len(fit_idx)}  early-stop n={len(es_idx)}")

    mean, std = X_train_raw.iloc[fit_idx].mean(), X_train_raw.iloc[fit_idx].std().replace(0.0, 1.0)

    def standardize(x: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(((x - mean) / std).clip(-10, 10).to_numpy(), dtype=torch.float32)

    X_fit, X_es = standardize(X_train_raw.iloc[fit_idx]), standardize(X_train_raw.iloc[es_idx])
    X_train_t, X_val_t, X_oos_t = standardize(X_train_raw), standardize(X_val_raw), standardize(X_oos_raw)

    model = Autoencoder(len(POOL), LATENT_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_fit), batch_size=2048, shuffle=True, generator=torch.Generator().manual_seed(SEED))

    log(f"stage=train_autoencoder input_dim={len(POOL)} latent_dim={LATENT_DIM}")
    best_es_loss, best_state, patience, bad_epochs = float("inf"), None, 8, 0
    t0 = time.time()
    for epoch in range(200):
        model.train()
        for (batch,) in loader:
            noisy = batch + torch.randn_like(batch) * 0.05
            opt.zero_grad()
            recon, _ = model(noisy)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            es_loss = float(loss_fn(model(X_es)[0], X_es))
        if es_loss < best_es_loss - 1e-5:
            best_es_loss, best_state, bad_epochs = es_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            log(f"  early stop at epoch {epoch} (best_es_loss={best_es_loss:.4f})")
            break
    model.load_state_dict(best_state)
    log(f"  trained in {time.time()-t0:.0f}s, best_es_loss={best_es_loss:.4f}")

    model.eval()
    with torch.no_grad():
        _, z_train = model(X_train_t)
        _, z_val = model(X_val_t)
        _, z_oos = model(X_oos_t)
        mse_train = float(loss_fn(model(X_train_t)[0], X_train_t))
        mse_val = float(loss_fn(model(X_val_t)[0], X_val_t))
        mse_oos = float(loss_fn(model(X_oos_t)[0], X_oos_t))
    log(f"  recon MSE: train={mse_train:.4f} val={mse_val:.4f} oos={mse_oos:.4f} (val/oos >> train = generalization warning)")

    for split, frame, z in [("train", train_frame, z_train), ("validation", val_frame, z_val), ("oos", oos_frame, z_oos)]:
        z_np = z.numpy()
        cols = {f"trend_ctx_latent_{i}": z_np[:, i] for i in range(LATENT_DIM)}
        ctx = pd.DataFrame({"timestamp": frame["timestamp"].to_numpy(), **cols})
        out_path = out_dir / f"{split}_context_features.csv"
        ctx.to_csv(out_path, index=False)
        log(f"  wrote {out_path}")

    log(f"DONE component={args.component} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
