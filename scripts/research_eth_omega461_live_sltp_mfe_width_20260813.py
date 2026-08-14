#!/usr/bin/env python3
"""RESEARCH ONLY -- ETH live Omega4.6.1 TP/SL width redesign: replace the ATR-percent-floor-bound
static TP/SL formula in _apply_atr_safety_sltp with a per-trade width driven by a supervised MFE
(Maximum Favorable Excursion) quantile regression prediction, instead of a fixed ATR-derived
constant.

Motivation (full writeup: docs/experiments/eth_omega461_live_sltp_mfe_width_20260813.md). The live
_ComponentConfig defaults (trading_bot_modules/omega4_6_1_live.py:86-98 -- atr_window=192,
tp_mult=12, sl_mult=6, min_tp=0.075, min_sl=0.040) leave the "ATR-adaptive" term (atr_pct*tp_mult)
below the min_tp/min_sl floor ~95-98.5% of the time (ETH 5m 192-bar ATR% median ~0.26%), so
take_profit/stop_loss is effectively a FIXED 7.5%/4.0% width regardless of the trade. That fixed
width takes 366-925 bars (30-77+ hours) to resolve on average (tmp/research_20260721/
exit_threshold_sweep_VAL.csv), capping trade count inside any fixed backtest/live window. The
2026-07-28 sweep already closed the "pick a different FIXED floor constant" axis (component-level
win did not survive a portfolio-level recheck -- tmp/research_20260728/tpsl_floor_portfolio_check/
summary.json, confirmed_vs_*_baseline: false both ways). This script tries a DIFFERENT mechanism:
size TP/SL from a per-row LEARNED estimate of the direction-conditional 48-bar MFE (the one signal
that passed this session's Odyssey sub-project MI/R^2 gate for h48qual's quality_head redesign --
docs/experiments/eth_h48qual_mfe_quantile_quality_regression_20260812.md step 1: VAL R2=+0.08,
OOS R2=+0.14, spearman +0.28/+0.39 both p<0.001).

Isolation of the one changed variable: direction/quality/entry timing/max_tp/max_sl/margin/
leverage sizing are all held byte-identical to the live baseline. margin_fraction/leverage are
computed upstream from the ORIGINAL ATR-based `dec` (unmodified) exactly as
research_eth_omega461_exit_sweep_20260721.prep_component already does (train_eval_omega4_2_
risk_sidecar_20260622._risk_feature_frame consumes decision_take_profit/decision_stop_loss as
RISK-SIZING MODEL INPUTS -- swapping them before sizing would confound sizing with width, so sizing
is computed first from the baseline dec, then ONLY AFTER that the take_profit/stop_loss actually
used by the exit replay loop are overwritten with the MFE-predicted width on a COPY of dec).

Reuses, does NOT reimplement:
  - research_eth_omega461_exit_sweep_20260721.py (base_sweep below): load_frame, prep_component,
    replay_exit_variant.
  - replay_omega4_6_1_greedy_router_20260706.py (router below): greedy_replay -- the single-shared-
    position-slot h48qual>zig075 priority router, the ACTUAL mechanism the live adapter uses to
    combine both components (PRIORITY=("h48qual","zig075"), SCALE_MAP identical to
    trading_bot_modules/omega4_6_1_live.py). prepare_component() from that module is NOT reused
    as-is: it hardcodes oof=False (correct for its own prior OOS-only usage), which would read the
    wrong prefixed prediction columns for a VAL prediction CSV (verified directly --
    validation_predictions_q050.csv only has *_oof_* columns). Instead the extra "component" dict
    fields greedy_replay needs beyond prep_component's own output (base_np, exit_runtime, pos_idx,
    route) are derived with the SAME rs._prepare_exit_runtime / hard._route_id calls
    prepare_component itself uses internally (see _as_router_component).

Two feature panels for the MFE regressor, run side by side per an orchestrator correction
(2026-08-13): this script never used the abandoned "Odyssey final-boss v1/v2/v3" track's reduced
FINAL12+autoencoder-latent(16)=28 panel or any from-scratch replacement model -- from the start it
trained directly on the live h48qual/zig075 parent bundles' own 102-column base_cols (verified
byte-identical between both bundles), because that is the panel the live TabM parents themselves
consume causally, and because the "final boss" track built ENTIRELY NEW direction+quality models
(replacing the live bundles outright), a different scope than this script's TP/SL-width-only change
on top of the unmodified live bundles. To give the requested apples-to-apples control anyway, this
script ALSO builds a second panel: FINAL12's available columns (see FINAL12_UNAVAILABLE below for
the 2 that are not) + a 16-dim denoising-autoencoder latent trained on the 102-col base_cols pool
(same architecture as verify_eth_h48qual_autoencoder_latent_mi_r2_gate_20260812.py: 64->32->16,
Gaussian noise 0.05, ReLU+Dropout 0.1, MSE, Adam lr=1e-3, batch=2048, patience=8, TRAIN-tail-15%
early-stop holdout, TRAIN-fit standardization only). Both panels are trained/scored identically
otherwise (same TRAIN window, same target, same HistGradientBoostingRegressor recipe) and reported
side by side -- neither is discarded regardless of which wins.

MFE label source: tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619/
{train,validation,oos}_triple_barrier_labels.csv, tb_long_mfe_h48_conservative /
tb_short_mfe_h48_conservative columns (dense, every bar, produced by
build_omega1_2_triple_barrier_labels_20260619.py as a byproduct of its h48_conservative virtual
triple-barrier simulation -- NOT recomputed here, NOT a saved trade ledger). Models trained ONLY on
the TRAIN split (2025-01-01..2025-09-30, strictly before VAL start).

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.
VAL window = 2025-10-01..2025-12-31, matching research_eth_omega461_exit_sweep_20260721.py's own
VAL window exactly (one month short of the canonical CLAUDE.md 09-01 start because the frozen OOF
prediction CSVs only exist from 2025-10-01 onward -- documented caveat, not silently fixed).

OOS is NOT run anywhere in this script (orchestrator instruction -- VAL-first, report back).

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega4_6_1_greedy_router_20260706 as router  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as base_sweep  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260813/omega461_live_sltp_mfe_width"
TB_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619"
TB_COLS = ["timestamp", "tb_long_mfe_h48_conservative", "tb_short_mfe_h48_conservative"]
TRAIN_START, TRAIN_END = "2025-01-01", "2025-09-30"

# Safety floors -- NOT swept (the closed 2026-07-28 axis swept FIXED floor CONSTANTS as the PRIMARY
# width driver; here the floor is a rarely-binding backstop because the primary driver is now the
# per-row MFE prediction). Values match build_omega1_2_triple_barrier_labels_20260619.py's own
# h48_conservative BarrierConfig(min_tp=0.006, min_sl=0.004) -- a principled, pre-existing anchor
# rather than a newly invented constant.
FLOOR_TP = 0.006
FLOOR_SL = 0.004
TP_SCALE_GRID = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0]
MFE_MODEL_SEED = 260813
AE_SEED = 260813

# FINAL12 (train_eval_eth_h48qual_final_boss_v3_dual_component_20260813.FINAL12, same list used by
# h48orig/"final boss" v1-v3). Verified directly against research_eth_omega461_exit_sweep_20260721.
# load_frame()'s output frame: 8 present as-is, 2 derivable from present raw columns (diff1/dt288,
# same transform recipe as train_eval_omega4_3head_parent72_eth_h48qual_final12_h384_20260811.py's
# _add_derived_columns), 2 NOT derivable -- their raw prerequisites (m7_vae_error, sig_whale) do not
# exist anywhere in this frame's source CSVs (a different, h48orig-only feature-generation lineage).
# m7_vae_error is also exactly the feature family trading_bot_modules/omega4_6_1_live.py's own
# bundle-integrity check explicitly forbids (base_cols may not start with "m7_"/"ai_"/"patchtst"/
# "tide_"/"dlinear" -- _Component.__init__, contract-drift guard), so this is not merely missing
# data, it is a feature the live-compatible pipeline structurally excludes by design.
FINAL10 = [
    "cvp_regime", "ou_halflife", "realized_skewness", "mta_funding", "vwap_dist_24",
    "funding_roc_48", "breakout_strength", "regime3_current_sensitive_wide24_chop_prob",
    "funding_pressure_diff1", "sum_toptrader_long_short_ratio_dt288",
]
FINAL12_UNAVAILABLE = ["m7_vae_error_dt288", "sig_whale_dt288"]
LATENT_DIM = 16


def log(msg: str) -> None:
    print(msg, flush=True)


def _load_tb_labels(split: str) -> pd.DataFrame:
    path = TB_DIR / f"{split}_triple_barrier_labels.csv"
    df = pd.read_csv(path, usecols=TB_COLS, parse_dates=["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _mfe_regressor(seed: int) -> HistGradientBoostingRegressor:
    # Same "strong regularization" recipe that passed the Odyssey MI/R^2 gate for this exact MFE
    # target (docs/experiments/eth_h48qual_mfe_quantile_quality_regression_20260812.md step 1):
    # depth=2 + early stopping + l2=2.0, quantile loss at q=0.5 (median). Held IDENTICAL across both
    # feature panels so the comparison isolates the feature set, not the model recipe.
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=0.5, max_depth=2, learning_rate=0.03, l2_regularization=2.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, max_iter=1000,
        random_state=seed,
    )


# ---------------------------------------------------------------------------------------------
# Feature panel builders. Each returns a (timestamp + feature columns) DataFrame per split, built
# so that any downstream train/label inner-join only needs a timestamp merge, never a recompute --
# derived (rolling/diff) columns are always computed on the FULL, gap-free, originally-ordered
# split frame first, exactly as train_eval_omega4_3head_parent72_eth_h48qual_final12_h384_20260811.
# _add_derived_columns does, so joining against the (smaller) label set afterward cannot corrupt a
# rolling window with skipped rows.
# ---------------------------------------------------------------------------------------------

def base102_panel(base_cols: list[str], frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = base_sweep.parent._base_input(frame, base_cols)  # base_cols + 13 zero POS_COLS, same input
    # contract the live parent TabM itself consumes at entry time -- POS_COLS are always 0 here
    # (no open position at entry-time scoring) so they cannot influence a tree split; kept for
    # byte-identical reuse of the already-validated parent._base_input helper.
    feature_cols = list(x.columns)
    out = pd.concat([frame[["timestamp"]].reset_index(drop=True), x.reset_index(drop=True)], axis=1)
    return out, feature_cols


def _add_final10_derived(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["funding_pressure_diff1"] = out["funding_pressure"].astype(np.float64).diff(1).fillna(0.0)
    src = out["sum_toptrader_long_short_ratio"].astype(np.float64)
    out["sum_toptrader_long_short_ratio_dt288"] = (src - src.rolling(288, min_periods=96).mean()).fillna(0.0)
    return out


class _DenoisingAutoencoder(nn.Module):
    def __init__(self, n_in: int, latent: int = LATENT_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, n_in),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


def _fit_denoising_autoencoder(x_train_std: np.ndarray, *, seed: int, noise_std: float = 0.05,
                                patience: int = 8, max_epochs: int = 300, batch_size: int = 2048,
                                holdout_frac: float = 0.15) -> tuple[_DenoisingAutoencoder, dict[str, Any]]:
    torch.manual_seed(seed)
    n = len(x_train_std)
    n_holdout = max(int(n * holdout_frac), 1)
    fit_x = x_train_std[: n - n_holdout]
    hold_x = x_train_std[n - n_holdout:]
    model = _DenoisingAutoencoder(x_train_std.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    fit_t = torch.from_numpy(fit_x.astype(np.float32))
    hold_t = torch.from_numpy(hold_x.astype(np.float32))
    dl = DataLoader(TensorDataset(fit_t), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_loss, best_state, bad_epochs, epoch = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        for (batch,) in dl:
            noisy = batch + noise_std * torch.randn_like(batch)
            opt.zero_grad()
            recon, _ = model(noisy)
            loss = torch.mean((recon - batch) ** 2)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            recon, _ = model(hold_t)
            hold_loss = float(torch.mean((recon - hold_t) ** 2))
        if hold_loss < best_loss - 1.0e-6:
            best_loss, best_state, bad_epochs = hold_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_holdout_mse": best_loss, "epochs_run": epoch + 1, "n_train_fit": len(fit_x), "n_holdout": len(hold_x)}


def final10_latent16_panel(base_cols: list[str], train_frame: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[str], dict[str, Any]]:
    """Fits the autoencoder + standardizer on train_frame ONLY, then transforms every frame in
    `frames` (which must include train_frame itself under some key) with the frozen fit. Returns
    {key: (timestamp + 26 feature cols)} plus the feature column list and a fit diagnostic."""
    x_base_train = base_sweep.parent._base_input(train_frame, base_cols)[base_cols].to_numpy(dtype=np.float64)
    mean = x_base_train.mean(axis=0)
    std = x_base_train.std(axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    x_train_std = (x_base_train - mean) / std
    ae, ae_diag = _fit_denoising_autoencoder(x_train_std, seed=AE_SEED)
    latent_cols = [f"ae_latent_{i}" for i in range(LATENT_DIM)]

    out: dict[str, pd.DataFrame] = {}
    for key, frame in frames.items():
        x_base = base_sweep.parent._base_input(frame, base_cols)[base_cols].to_numpy(dtype=np.float64)
        x_std = (x_base - mean) / std
        with torch.no_grad():
            _, latent = ae(torch.from_numpy(x_std.astype(np.float32)))
        latent_np = latent.numpy().astype(np.float64)
        final10 = _add_final10_derived(frame)[FINAL10].reset_index(drop=True)
        combined = pd.concat([final10, pd.DataFrame(latent_np, columns=latent_cols)], axis=1)
        out[key] = pd.concat([frame[["timestamp"]].reset_index(drop=True), combined], axis=1)
    feature_cols = FINAL10 + latent_cols
    diag = {"autoencoder": ae_diag, "final10_cols": FINAL10, "final12_unavailable": FINAL12_UNAVAILABLE,
            "latent_cols": latent_cols, "ae_input_pool": "base_cols (102, live-bundle-compatible)"}
    return out, feature_cols, diag


# ---------------------------------------------------------------------------------------------
# Generic MFE-model train/score/width machinery -- identical regardless of which panel produced
# the (timestamp + feature_cols) frames passed in.
# ---------------------------------------------------------------------------------------------

def train_mfe_models(panel_train: pd.DataFrame, feature_cols: list[str], train_labels: pd.DataFrame, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    merged = panel_train.merge(train_labels, on="timestamp", how="inner").reset_index(drop=True)
    x = merged[feature_cols]
    y_long = merged["tb_long_mfe_h48_conservative"].to_numpy(dtype=np.float64)
    y_short = merged["tb_short_mfe_h48_conservative"].to_numpy(dtype=np.float64)
    model_long = _mfe_regressor(seed).fit(x, y_long)
    model_short = _mfe_regressor(seed + 1).fit(x, y_short)
    pl, ps = model_long.predict(x), model_short.predict(x)
    diag = {
        "n_features": len(feature_cols), "panel_rows": int(len(panel_train)), "label_rows": int(len(train_labels)), "merged_rows": int(len(merged)),
        "long_n_iter": int(getattr(model_long, "n_iter_", 0)), "short_n_iter": int(getattr(model_short, "n_iter_", 0)),
        "long_r2_train": float(r2_score(y_long, pl)), "short_r2_train": float(r2_score(y_short, ps)),
        "long_spearman_train": float(spearmanr(y_long, pl)[0]), "short_spearman_train": float(spearmanr(y_short, ps)[0]),
        "long_mfe_median_train": float(np.median(y_long)), "short_mfe_median_train": float(np.median(y_short)),
    }
    return {"long": model_long, "short": model_short}, diag


def val_sanity_gate(models: dict[str, Any], panel_val: pd.DataFrame, feature_cols: list[str], val_labels: pd.DataFrame) -> dict[str, Any]:
    merged = panel_val.merge(val_labels, on="timestamp", how="inner").reset_index(drop=True)
    x = merged[feature_cols]
    y_long = merged["tb_long_mfe_h48_conservative"].to_numpy(dtype=np.float64)
    y_short = merged["tb_short_mfe_h48_conservative"].to_numpy(dtype=np.float64)
    pl, ps = models["long"].predict(x), models["short"].predict(x)
    return {
        "val_rows": int(len(merged)),
        "long_r2_val": float(r2_score(y_long, pl)), "short_r2_val": float(r2_score(y_short, ps)),
        "long_spearman_val": float(spearmanr(y_long, pl)[0]), "short_spearman_val": float(spearmanr(y_short, ps)[0]),
    }


def predicted_width(models: dict[str, Any], x: pd.DataFrame, side: np.ndarray) -> np.ndarray:
    pred_long = np.clip(models["long"].predict(x), 0.0, None)
    pred_short = np.clip(models["short"].predict(x), 0.0, None)
    return np.where(side > 0, pred_long, np.where(side < 0, pred_short, 0.0))


def apply_mfe_width_sltp(dec: pd.DataFrame, width: np.ndarray, *, tp_scale: float, sl_ratio: float,
                          min_tp: float, min_sl: float, max_tp: float, max_sl: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mirrors eval_omega4_1_atr_safety_sltp_20260622._apply_atr_safety_sltp's active-row masking
    exactly, swapping the driver from atr_pct*tp_mult to width*tp_scale (width = per-row predicted
    MFE, raw price-move units -- same unit contract as the ATR-based formula it replaces)."""
    out = dec.copy().reset_index(drop=True)
    active = base_sweep.omega._active(out)
    tp_raw = np.asarray(width, dtype=np.float64) * float(tp_scale)
    sl_raw = tp_raw * float(sl_ratio)
    tp = np.clip(np.maximum(float(min_tp), tp_raw), 0.0, float(max_tp))
    sl = np.clip(np.maximum(float(min_sl), sl_raw), 0.0, float(max_sl))
    out.loc[active, "take_profit"] = tp[active]
    out.loc[active, "stop_loss"] = sl[active]
    out.loc[~active, ["take_profit", "stop_loss"]] = 0.0
    active_tp, active_sl = tp[active], sl[active]
    diag = {
        "tp_scale": float(tp_scale), "sl_ratio": float(sl_ratio), "min_tp": float(min_tp), "min_sl": float(min_sl),
        "max_tp": float(max_tp), "max_sl": float(max_sl), "active_rows": int(active.sum()),
        "tp_p50": float(np.quantile(active_tp, 0.5)) if len(active_tp) else 0.0,
        "tp_p90": float(np.quantile(active_tp, 0.9)) if len(active_tp) else 0.0,
        "sl_p50": float(np.quantile(active_sl, 0.5)) if len(active_sl) else 0.0,
        "tp_floor_bind_rate": float((active_tp <= min_tp + 1.0e-12).mean()) if len(active_tp) else 0.0,
        "tp_cap_bind_rate": float((active_tp >= max_tp - 1.0e-12).mean()) if len(active_tp) else 0.0,
    }
    return out, diag


def _ledger_stats(ledger: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    if len(ledger) == 0:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0, "avg_hold_bars": 0.0, "trades_per_day": 0.0}
    returns = ledger["trade_return"].to_numpy()
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1.0e-12) - 1.0
    hold = (ledger["exit_i"] - ledger["entry_i"]).clip(lower=0)
    return {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0), "trades": int(len(ledger)),
            "wr": float((returns > 0).mean()), "avg_hold_bars": float(hold.mean()),
            "trades_per_day": float(len(ledger) / base_sweep.rs._duration_days(frame))}


def _duration_gated(ledger: pd.DataFrame, frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    if len(ledger) == 0:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0, "skipped": 0, "avg_hold_bars": 0.0}
    active = ledger.copy()
    active["entry_timestamp_dt"] = pd.to_datetime(active["entry_timestamp"])
    market = frame[["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp_dt"})
    active = active.merge(market, on="entry_timestamp_dt", how="left")
    hit = active["ou_halflife"] <= threshold
    gated_returns = np.where(hit, 0.0, active["trade_return"])
    curve = np.concatenate([[1.0], np.cumprod(1.0 + gated_returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1.0e-12) - 1.0
    kept = active.loc[~hit]
    hold = (kept["exit_i"] - kept["entry_i"]).clip(lower=0)
    n_kept = int((~hit).sum())
    return {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0), "trades": n_kept,
            "wr": float((gated_returns[~hit] > 0).mean()) if n_kept else 0.0, "skipped": int(hit.sum()),
            "avg_hold_bars": float(hold.mean()) if n_kept else 0.0}


def _as_router_component(p: dict[str, Any], *, exit_threshold: float) -> dict[str, Any]:
    base_np, exit_runtime, pos_idx = base_sweep.rs._prepare_exit_runtime(p["x"], p["loaded"])
    route = base_sweep.hard._route_id(p["frame"])
    return {"dec": p["dec"], "margin": p["margin"], "leverage": p["leverage"], "base_np": base_np,
            "exit_runtime": exit_runtime, "pos_idx": pos_idx, "route": route, "exit_threshold": float(exit_threshold)}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("stage=load_frames")
    val_frame = base_sweep.load_frame(base_sweep.VAL_START, base_sweep.VAL_END, base_csv=base_sweep.BASE_2025, wide24_csv=base_sweep.WIDE24_2025)
    train_frame = base_sweep.load_frame(TRAIN_START, TRAIN_END, base_csv=base_sweep.BASE_2025, wide24_csv=base_sweep.WIDE24_2025)
    log(f"val_frame rows={len(val_frame)} range=[{val_frame['timestamp'].min()}, {val_frame['timestamp'].max()}]")
    log(f"train_frame rows={len(train_frame)} range=[{train_frame['timestamp'].min()}, {train_frame['timestamp'].max()}]")

    bundle_h48 = torch.load(base_sweep.COMPONENTS["h48qual"]["bundle"], map_location="cpu", weights_only=False)
    bundle_zig = torch.load(base_sweep.COMPONENTS["zig075"]["bundle"], map_location="cpu", weights_only=False)
    base_cols = list(bundle_h48["base_cols"])
    if list(bundle_zig["base_cols"]) != base_cols:
        raise RuntimeError("h48qual/zig075 base_cols differ -- cannot share one MFE model; would need per-component training")
    missing_from_frame = [c for c in base_cols if c not in val_frame.columns]
    log(f"shared base_cols n={len(base_cols)} (verified identical across h48qual/zig075 bundles); missing_from_val_frame={missing_from_frame}")
    if missing_from_frame:
        raise RuntimeError(f"base_cols missing from load_frame() output: {missing_from_frame}")
    missing_final10_prereqs = [c for c in ("cvp_regime", "ou_halflife", "realized_skewness", "mta_funding", "vwap_dist_24",
                                            "funding_roc_48", "breakout_strength", "regime3_current_sensitive_wide24_chop_prob",
                                            "funding_pressure", "sum_toptrader_long_short_ratio") if c not in val_frame.columns]
    log(f"FINAL10 prerequisite columns missing from val_frame: {missing_final10_prereqs} (should be empty)")
    if missing_final10_prereqs:
        raise RuntimeError(f"FINAL10 prerequisites missing: {missing_final10_prereqs}")

    train_labels = _load_tb_labels("train")
    val_labels = _load_tb_labels("validation")

    log("stage=prep_components (baseline ATR-floor dec/margin/leverage, computed ONCE, shared by both feature-set arms)")
    prepped: dict[str, dict[str, Any]] = {}
    baseline_rows: list[dict[str, Any]] = []
    for name, cfg in base_sweep.COMPONENTS.items():
        pred_csv = base_sweep.EXT_PRED_DIR / name / f"validation_predictions_{cfg['q_tag']}.csv"
        p = base_sweep.prep_component(name, cfg, val_frame, pred_csv, oof=True)
        prepped[name] = p
        m_base, _ = base_sweep.replay_exit_variant(
            p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
            exit_threshold=base_sweep.BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=base_sweep.COST_MULT,
            notional_scaled_sltp=p["notional_scaled_sltp"], device=base_sweep.DEVICE,
        )
        baseline_rows.append({"feature_set": "n/a", "component": name, "variant": "baseline_atr_floor", "tp_scale": None,
                               **{k: v for k, v in m_base.items() if k != "exit_reasons"}, "exit_reasons": json.dumps(m_base["exit_reasons"])})
        log(f"  component={name} baseline pnl={m_base['pnl']:.2f} trades={m_base['trades']} avg_hold_bars={m_base['avg_hold_bars']:.1f}")

    router_base = {name: _as_router_component(p, exit_threshold=base_sweep.BASELINE_EXIT_THRESHOLD) for name, p in prepped.items()}
    fee0, slip0 = prepped["h48qual"]["fee"], prepped["h48qual"]["slip"]
    _, ledger_base_combined = router.greedy_replay(val_frame, router_base, fee=fee0, slip=slip0, cost_mult=base_sweep.COST_MULT, device=base_sweep.DEVICE)
    combined_baseline = {"no_gate": _ledger_stats(ledger_base_combined, val_frame), "with_gate": _duration_gated(ledger_base_combined, val_frame, router.DURATION_THRESHOLD),
                          "source_component_counts": ledger_base_combined["source_component"].value_counts().to_dict() if len(ledger_base_combined) else {}}
    ledger_base_combined.to_csv(OUT_DIR / "priority_combined_ledger_baseline_VAL.csv", index=False)
    log(f"priority_combined baseline: {json.dumps(combined_baseline, default=base_sweep.omega._json_default)}")

    # ------------------------------------------------------------------------------------------
    # Two feature-set arms
    # ------------------------------------------------------------------------------------------
    panel_diags: dict[str, Any] = {}
    all_variant_rows: list[dict[str, Any]] = list(baseline_rows)
    combined_results: dict[str, Any] = {"baseline": combined_baseline}

    log("stage=build_panel_base102")
    panel_train_102, feat_cols_102 = base102_panel(base_cols, train_frame)
    panel_val_102, _ = base102_panel(base_cols, val_frame)

    log("stage=build_panel_final10_latent16 (fits autoencoder on TRAIN only)")
    panels_26, feat_cols_26, ae_diag = final10_latent16_panel(base_cols, train_frame, {"train": train_frame, "val": val_frame})
    panel_train_26, panel_val_26 = panels_26["train"], panels_26["val"]
    log(json.dumps(ae_diag["autoencoder"], indent=2))

    feature_sets = {
        "base102": (panel_train_102, panel_val_102, feat_cols_102, MFE_MODEL_SEED),
        "final10_latent16": (panel_train_26, panel_val_26, feat_cols_26, MFE_MODEL_SEED + 2000),
    }

    for fset_name, (panel_train, panel_val, feature_cols, seed) in feature_sets.items():
        log(f"stage=train_mfe_models feature_set={fset_name} n_features={len(feature_cols)}")
        models, train_diag = train_mfe_models(panel_train, feature_cols, train_labels, seed=seed)
        log(json.dumps(train_diag, indent=2))
        val_diag = val_sanity_gate(models, panel_val, feature_cols, val_labels)
        log(f"  val_sanity_gate: {json.dumps(val_diag)}")
        panel_diags[fset_name] = {"train": train_diag, "val_sanity_gate": val_diag}

        # per-component scoring input: panel_val row-aligned to val_frame == p["frame"] (verified
        # identical for both components -- prep_component's pred-CSV reconciliation is a no-op
        # subset here since val_frame's timestamps are already a subset of both pred CSVs').
        x_val_scoring = panel_val[feature_cols]

        for name, p in prepped.items():
            cfg = base_sweep.COMPONENTS[name]
            side = pd.to_numeric(p["dec"]["side"], errors="raise").to_numpy(dtype=np.int64)
            width = predicted_width(models, x_val_scoring, side)
            sl_ratio = float(cfg["sl_mult"]) / float(cfg["tp_mult"])
            for scale in TP_SCALE_GRID:
                dec_mfe, wdiag = apply_mfe_width_sltp(p["dec"], width, tp_scale=scale, sl_ratio=sl_ratio,
                                                       min_tp=FLOOR_TP, min_sl=FLOOR_SL, max_tp=cfg["max_tp"], max_sl=cfg["max_sl"])
                m, _ = base_sweep.replay_exit_variant(
                    p["frame"], p["x"], dec_mfe, p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
                    exit_threshold=base_sweep.BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=base_sweep.COST_MULT,
                    notional_scaled_sltp=p["notional_scaled_sltp"], device=base_sweep.DEVICE,
                )
                all_variant_rows.append({"feature_set": fset_name, "component": name, "variant": f"mfe_width_scale{scale:g}", "tp_scale": scale,
                                          **{k: v for k, v in m.items() if k != "exit_reasons"}, "exit_reasons": json.dumps(m["exit_reasons"]),
                                          **{f"width_{k}": v for k, v in wdiag.items()}})
            log(f"  component={name} feature_set={fset_name} done")

        for scale in TP_SCALE_GRID:
            comps = {}
            for name, p in prepped.items():
                cfg = base_sweep.COMPONENTS[name]
                side = pd.to_numeric(p["dec"]["side"], errors="raise").to_numpy(dtype=np.int64)
                width = predicted_width(models, x_val_scoring, side)
                sl_ratio = float(cfg["sl_mult"]) / float(cfg["tp_mult"])
                dec_mfe, _ = apply_mfe_width_sltp(p["dec"], width, tp_scale=scale, sl_ratio=sl_ratio,
                                                   min_tp=FLOOR_TP, min_sl=FLOOR_SL, max_tp=cfg["max_tp"], max_sl=cfg["max_sl"])
                comps[name] = {**router_base[name], "dec": dec_mfe}
            _, ledger_mfe = router.greedy_replay(val_frame, comps, fee=fee0, slip=slip0, cost_mult=base_sweep.COST_MULT, device=base_sweep.DEVICE)
            key = f"{fset_name}__mfe_width_scale{scale:g}"
            res = {"no_gate": _ledger_stats(ledger_mfe, val_frame), "with_gate": _duration_gated(ledger_mfe, val_frame, router.DURATION_THRESHOLD),
                   "source_component_counts": ledger_mfe["source_component"].value_counts().to_dict() if len(ledger_mfe) else {}}
            combined_results[key] = res
            ledger_mfe.to_csv(OUT_DIR / f"priority_combined_ledger_{key}_VAL.csv", index=False)
            log(f"  priority_combined {key}: {json.dumps(res, default=base_sweep.omega._json_default)}")

    component_df = pd.DataFrame(all_variant_rows)
    component_df.to_csv(OUT_DIR / "component_variants_VAL.csv", index=False)
    print_cols = ["feature_set", "component", "variant", "pnl", "mdd", "trades", "wr", "avg_hold_bars", "trades_per_day"]
    log(component_df[print_cols].to_string(index=False))

    report = {
        "model_id": "omega461_live_sltp_mfe_width_20260813",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "val_window": [base_sweep.VAL_START, base_sweep.VAL_END], "oos_run": False,
        "train_window_for_mfe_model": [TRAIN_START, TRAIN_END],
        "feature_sets": {"base102": {"n_features": len(feat_cols_102)}, "final10_latent16": {"n_features": len(feat_cols_26), **ae_diag}},
        "panel_diags": panel_diags,
        "floor_tp": FLOOR_TP, "floor_sl": FLOOR_SL, "tp_scale_grid": TP_SCALE_GRID,
        "component_variants_val": all_variant_rows, "priority_combined_val": combined_results,
    }
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=base_sweep.omega._json_default), encoding="utf-8"
    )
    log(f"stage=done report={OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
