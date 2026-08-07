"""BTC analogue of eval_eth_cmamba_entry_filter_20260721.py: test BTC's own (newly-built,
research-only) CryptoMamba future-regime model as an ENTRY-TIME-ONLY filter on top of the live v1
single-component replay. Skip an entry if CryptoMamba's current +6bar directional prediction
disagrees with the entry side; never re-checked during the hold.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_btc_20260708 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_btc_20260708 as sidecar  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_btc_20260708 as omega4  # noqa: E402

LEVERAGE_CAP, NOTIONAL_CAP = 5.0, 1.8

CMAMBA_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_btc_h6_nocurrent_20260721"
CMAMBA_VAL = CMAMBA_DIR / "btc_features_2025_regime3_cryptomamba_pred_btc_h6_nocurrent_20260721.csv"
CMAMBA_OOS = CMAMBA_DIR / "btc_features_2026_regime3_cryptomamba_pred_btc_h6_nocurrent_20260721.csv"

_PARENT_DIR = ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_20260708"
_SIDECAR_DIR = ROOT / "tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260708"


def _cmamba_dir_signal(frame: pd.DataFrame, cmamba_path: Path) -> np.ndarray:
    cm = pd.read_csv(cmamba_path, parse_dates=["timestamp"])
    cm = cm[["timestamp", "regime3_cmamba_h6_future_pred_id"]]
    merged = frame[["timestamp"]].merge(cm, on="timestamp", how="left", validate="one_to_one")
    pred_id = merged["regime3_cmamba_h6_future_pred_id"].to_numpy()
    sig = np.zeros(len(merged), dtype=np.int64)
    sig[pred_id == 0] = 1
    sig[pred_id == 1] = -1
    return sig


def _compound_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    cash = peak = 1.0
    mdd = 0.0
    wins = 0
    for ret in ledger["trade_return"].to_numpy(dtype=np.float64):
        cash *= 1.0 + float(ret)
        wins += int(ret > 0.0)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    return {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(len(ledger)), "wr": float(wins / len(ledger))}


def _scaled_margin_leverage(dec: pd.DataFrame, base_margin: np.ndarray, base_leverage: np.ndarray, *, long_scale: float, short_scale: float):
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    scale = np.where(side > 0, float(long_scale), np.where(side < 0, float(short_scale), 1.0))
    leverage = np.minimum(base_leverage * scale, LEVERAGE_CAP)
    notional = np.minimum(base_margin * leverage, NOTIONAL_CAP)
    with np.errstate(divide="ignore", invalid="ignore"):
        leverage = np.where(base_margin > 0.0, notional / np.maximum(base_margin, 1e-12), leverage)
    return base_margin, leverage


def main() -> int:
    device = parent._device("cpu")

    print("stage=load_bundle_and_sidecar", flush=True)
    bundle = torch.load(_PARENT_DIR / "true_3head_tabm_bundle.pt", map_location=device, weights_only=False)
    models: dict[str, Any] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    loaded = parent._load_payloads(models, device=device)
    with open(_SIDECAR_DIR / "risk_sidecar.pkl", "rb") as f:
        pkl = pickle.load(f)

    print("stage=prepare_frames", flush=True)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=omega4.LABEL_DIR,
        quality_mode="quality_label_action",
        quality_label_dir=ROOT / "tmp/causal_regen_20260516/btc_h48_conservative_padded_to_zigzag_timestamps_20260708",
        quality_min_edge=0.0, quality_max_mae=0.0, quality_min_mfe_mae=0.0, quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()

    pred_dir, tag = _PARENT_DIR, "q055"
    val_src = sidecar._load_precomputed_prediction(pred_dir, "validation", tag, frames["val_raw"])
    oos_src = sidecar._load_precomputed_prediction(pred_dir, "oos", tag, frames["oos_raw"])
    x_val = parent._base_input(frames["val_raw"], base_cols)
    x_oos = parent._base_input(frames["oos_raw"], base_cols)
    val_dec_base = parent._to_decisions(val_src, oof=True)
    oos_dec_base = parent._to_decisions(oos_src, oof=False)

    print("stage=apply_atr_contract", flush=True)
    atr_kwargs = dict(atr_window=192, tp_mult=12.0, sl_mult=6.0, min_tp=0.075, min_sl=0.040, max_tp=0.22, max_sl=0.12)
    val_dec, _ = atr_eval._apply_atr_safety_sltp(val_dec_base, frames["val_raw"], **atr_kwargs)
    oos_dec, _ = atr_eval._apply_atr_safety_sltp(oos_dec_base, frames["oos_raw"], **atr_kwargs)
    val_atr = atr_eval._atr_pct(frames["val_raw"], 192)
    oos_atr = atr_eval._atr_pct(frames["oos_raw"], 192)

    val_cmamba_sig = _cmamba_dir_signal(frames["val_raw"], CMAMBA_VAL)
    oos_cmamba_sig = _cmamba_dir_signal(frames["oos_raw"], CMAMBA_OOS)

    def _apply_entry_filter(dec: pd.DataFrame, sig: np.ndarray):
        d = dec.copy()
        side = pd.to_numeric(d["side"], errors="raise").to_numpy(dtype=np.int64)
        block = ((side > 0) & (sig == -1)) | ((side < 0) & (sig == 1))
        d.loc[block, "side"] = 0
        return d, int(block.sum())

    def _score_size_replay(dec: pd.DataFrame, frame: pd.DataFrame, src: pd.DataFrame, atr_pct: np.ndarray, x_base: pd.DataFrame):
        feats = sidecar._risk_feature_frame(frame, src, dec, base_cols, atr_pct=atr_pct, feature_mode=pkl["risk_feature_mode"])
        x_feat, _ = sidecar._feature_matrix(feats, pkl["feature_columns"])
        side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
        score = sidecar._predict_side_split_models(pkl["model"], x_feat, side_all) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_feat), dtype=np.float64)
        mapping = pkl["selected_mapping"]
        base_margin = sidecar._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
        base_leverage = sidecar._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS}) if pkl["dynamic_leverage"] else np.ones(len(dec))
        margin, leverage = _scaled_margin_leverage(dec, base_margin, base_leverage, long_scale=0.5, short_scale=2.5)
        _, ledger = sidecar._replay_with_risk(
            frame, x_base, dec, loaded, risk_margin_fraction=margin, risk_leverage=leverage,
            exit_threshold=0.95, fee=fee, slip=slip, cost_mult=3.0,
            notional_scaled_sltp=False, exit_sizing_input_mode="actual", device=device,
        )
        return ledger

    print("stage=baseline_replay", flush=True)
    val_ledger_base = _score_size_replay(val_dec, frames["val_raw"], val_src, val_atr, x_val)
    oos_ledger_base = _score_size_replay(oos_dec, frames["oos_raw"], oos_src, oos_atr, x_oos)

    print("stage=filtered_replay", flush=True)
    val_dec_f, val_blocked = _apply_entry_filter(val_dec, val_cmamba_sig)
    oos_dec_f, oos_blocked = _apply_entry_filter(oos_dec, oos_cmamba_sig)
    val_ledger_f = _score_size_replay(val_dec_f, frames["val_raw"], val_src, val_atr, x_val)
    oos_ledger_f = _score_size_replay(oos_dec_f, frames["oos_raw"], oos_src, oos_atr, x_oos)

    print(f"VAL baseline : {_compound_metrics(val_ledger_base)}", flush=True)
    print(f"VAL filtered : {_compound_metrics(val_ledger_f)}  blocked={val_blocked}", flush=True)
    print(f"OOS baseline : {_compound_metrics(oos_ledger_base)}", flush=True)
    print(f"OOS filtered : {_compound_metrics(oos_ledger_f)}  blocked={oos_blocked}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
