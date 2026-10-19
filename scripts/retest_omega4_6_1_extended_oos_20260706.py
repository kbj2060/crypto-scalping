#!/usr/bin/env python3
"""Faithful re-score of Omega4.6.1 (duration_ou_halflife_risk_gate) on the extended 2026-01-01..
06-30 OOS window, reusing the FROZEN artifacts (parent 3-head TabM bundles, risk_sidecar.pkl) with
NO retraining -- only inference/replay functions imported directly from the original training
scripts and called on new data. Also runs the SAME code on the ORIGINAL Jan-Feb frame first as a
self-check against the already-published numbers, to catch reimplementation bugs before trusting
the extended result.

Known limitation (documented in build_omega4_6_1_extended_parent_predictions_20260706.py):
ou_halflife/kel/evt_excess_z/btc_corr_60/dual_momentum differ from the original alpha6/7-lineage
feature file (features/elite.py formulas appear to have changed since 2026-05-29, git history too
sparse to recover the old version). ou_halflife specifically feeds the duration gate rule.
"""

from __future__ import annotations

import json
import pickle
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

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

WIDE24_2026 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"
BASE_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
ORIG_OOS_ALPHA6 = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
EXT_PRED_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"

COMPONENTS = {
    "h48qual": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/true_3head_tabm_bundle.pt",
        "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_h48qual_q050_precomputed_20260630/risk_sidecar.pkl",
        "q_tag": "q050",
        "quality_threshold": 0.50,
        "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
        "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
        "exit_threshold": 0.95,
    },
    "zig075": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt",
        "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_zig075_q075_precomputed_20260630/risk_sidecar.pkl",
        "q_tag": "q075",
        "quality_threshold": 0.75,
        "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
        "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
        "exit_threshold": 0.95,
    },
}
COST_MULT = 1.0
DEVICE = parent._device("cpu")


def load_frame_current(start: str, end: str) -> pd.DataFrame:
    frame = pd.read_csv(BASE_2026, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    overlay = pd.read_csv(WIDE24_2026, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)].reset_index(drop=True)
    return frame


def load_frame_orig_alpha6() -> pd.DataFrame:
    frame = pd.read_csv(ORIG_OOS_ALPHA6, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    overlay = pd.read_csv(WIDE24_2026, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    return frame


def score_component(frame: pd.DataFrame, oos_pred_csv: Path, cfg: dict, *, prefix: str, oof: bool = False) -> tuple[dict, pd.DataFrame]:
    bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
    base_cols = bundle["base_cols"]
    models = bundle["models"]

    oos_src_raw = pd.read_csv(oos_pred_csv)
    oos_src_raw["timestamp"] = pd.to_datetime(oos_src_raw["timestamp"])
    # Align frame to the prediction file's timestamps (inner join) -- the original training
    # pipeline's omega._align() trims a few rows for label warm-up/tail that the raw feature
    # frame doesn't reflect; predictions are the authoritative row set.
    keep_ts = set(oos_src_raw["timestamp"])
    frame = frame[frame["timestamp"].isin(keep_ts)].reset_index(drop=True)
    oos_src = oos_src_raw[oos_src_raw["timestamp"].isin(set(frame["timestamp"]))].reset_index(drop=True)
    if len(oos_src) != len(frame) or not oos_src["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError(f"{prefix}: prediction/frame timestamp mismatch after align ({len(oos_src)} vs {len(frame)})")

    x_oos = parent._base_input(frame, base_cols)
    oos_dec_base = parent._to_decisions(oos_src, oof=oof)
    oos_dec, atr_diag = atr_eval._apply_atr_safety_sltp(
        oos_dec_base, frame, atr_window=cfg["atr_window"], tp_mult=cfg["tp_mult"], sl_mult=cfg["sl_mult"],
        min_tp=cfg["min_tp"], min_sl=cfg["min_sl"], max_tp=cfg["max_tp"], max_sl=cfg["max_sl"],
    )
    oos_atr = atr_eval._atr_pct(frame, cfg["atr_window"])

    fee, slip = omega._load_fee_slip()
    loaded = parent._load_payloads(models, device=DEVICE)

    with open(cfg["sidecar_pkl"], "rb") as f:
        pkl = pickle.load(f)

    oos_features = sidecar._risk_feature_frame(frame, oos_src, oos_dec, base_cols, atr_pct=oos_atr, feature_mode=pkl["risk_feature_mode"])
    x_oos_all, _ = sidecar._feature_matrix(oos_features, pkl["feature_columns"])
    oos_side_all = pd.to_numeric(oos_dec["side"], errors="raise").to_numpy(dtype=np.int64)

    if pkl["side_split_model"]:
        oos_score = sidecar._predict_side_split_models(pkl["model"], x_oos_all, oos_side_all)
    else:
        oos_score = np.asarray(pkl["model"].predict(x_oos_all), dtype=np.float64)

    mapping = pkl["selected_mapping"]
    margin_kwargs = {k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS}
    oos_margin = sidecar._risk_margins(oos_dec, oos_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **margin_kwargs)
    oos_leverage = None
    if pkl["dynamic_leverage"]:
        lev_kwargs = {k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS}
        oos_leverage = sidecar._risk_leverage(oos_dec, oos_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **lev_kwargs)

    oos_sized_m, oos_sized_ledger = sidecar._replay_with_risk(
        frame, x_oos, oos_dec, loaded,
        risk_margin_fraction=oos_margin, risk_leverage=oos_leverage,
        exit_threshold=cfg["exit_threshold"], fee=fee, slip=slip, cost_mult=COST_MULT,
        notional_scaled_sltp=pkl["notional_scaled_sltp"], exit_sizing_input_mode=pkl["exit_sizing_input_mode"],
        device=DEVICE,
    )
    return oos_sized_m, oos_sized_ledger


def main() -> int:
    print("=== SELF-CHECK: original alpha6/7 Jan-Feb frame (should roughly match published numbers) ===", flush=True)
    orig_frame = load_frame_orig_alpha6()
    for name, cfg in COMPONENTS.items():
        orig_pred = ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_{'zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630' if name == 'h48qual' else 'current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629'}/oos_predictions_{cfg['q_tag']}.csv"
        m, ledger = score_component(orig_frame, orig_pred, cfg, prefix=f"{name}_selfcheck")
        print(f"{name} self-check (orig Jan-Feb): pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% trades={m['trades']} wr={m['wr']:.3f}", flush=True)

    print("\n=== EXTENDED: current-code Jan-Jun 2026 frame ===", flush=True)
    ext_frame = load_frame_current("2026-01-01", "2026-06-30")
    results = {}
    for name, cfg in COMPONENTS.items():
        pred_csv = EXT_PRED_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        m, ledger = score_component(ext_frame, pred_csv, cfg, prefix=name)
        results[name] = (m, ledger)
        print(f"{name} extended (Jan-Jun): pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% trades={m['trades']} wr={m['wr']:.3f}", flush=True)
        ledger.to_csv(EXT_PRED_DIR / f"{name}_extended_sized_ledger.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
