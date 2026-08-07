#!/usr/bin/env python3
"""Genuine fresh-forward, bar-by-bar, single-account greedy router replay of Omega4.6.1
(h48qual > zig075 priority), run on TWO model sets over the SAME VAL/OOS windows for an
apples-to-apples comparison:
  - "baseline102": the original live 102-base-feature h48qual/zig075 parent bundles + their
    original risk sidecars (unchanged, frozen artifacts).
  - "reduced80": the 20260724 research candidate trained on an 80-feature base_cols subset
    (102 minus 17 negative-importance features minus 5 features unavailable in the training
    data), with matching risk sidecars retrained on the SAME config except the
    validation-avg-notional exposure band, which had to be relaxed (see NOTIONAL_BAND_DEVIATION
    below) -- documented, not silent.

Windows (per CLAUDE.md's Fresh-Forward rule, with the project's own SPLIT_TS-driven VAL start):
  VAL: 2025-10-01 .. 2025-12-31
  OOS: 2026-01-01 .. 2026-03-31

Both model sets reuse the EXACT same greedy-router mechanics as the live-realistic OOS harness
(replay_omega4_6_1_greedy_router_20260706.py): same SCALE_MAP, LEVERAGE_CAP, NOTIONAL_CAP,
PRIORITY order, and VAL-selected DURATION_THRESHOLD constant -- nothing here is re-tuned per
model set. This is a comparison-only run: no promotion/artifact-integrity gate is invoked, and
nothing is wired into the live bot.

fresh_forward_bar_by_bar=True, trade_ledgers_used_as_input=False,
saved_parent_exit_timestamps_used=False, future_rows_used_for_entry=False -- causal walk-forward
over each window's own frame, no stored ledger reuse, no future-row joins.
"""
from __future__ import annotations

import importlib
import json
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
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_reduced80_20260724"
SCALE_MAP = {"h48qual_L": 0.38, "h48qual_S": 2.499, "zig075_L": 2.446, "zig075_S": 2.478}
LEVERAGE_CAP, NOTIONAL_CAP = 5.0, 1.8
PRIORITY = ("h48qual", "zig075")
DURATION_THRESHOLD = 0.005417  # frozen, VAL-reselected 2026-07-06 (replay_omega4_6_1_greedy_router_20260706.py)
VAL_START, VAL_END = "2025-10-01", "2025-12-31 23:59:59"
OOS_START, OOS_END = "2026-01-01", "2026-03-31 23:59:59"

NOTIONAL_BAND_DEVIATION = (
    "Baseline sidecars (h48qual q050, zig075 q075) were selected under "
    "--min-validation-avg-notional 0.45 --max-validation-avg-notional 0.95. Retraining the "
    "reduced80 sidecars with that exact band raised RuntimeError('no eligible risk mapping "
    "after validation average notional constraint') for BOTH components -- the 80-feature "
    "parents' risk-score distributions produced no grid candidate simultaneously inside that "
    "notional band. Per explicit user decision, the reduced80 sidecars were retrained with the "
    "band disabled (--min/--max-validation-avg-notional 0.0, i.e. no artificial notional-band "
    "constraint) while every other sidecar config was kept identical (HGB, side-split, dynamic "
    "leverage, selection_objective=log_risk, selection_scope=validation_only, log_risk tail/"
    "liquidation params). The resulting selected mappings landed at validation avg_notional "
    "~0.55 (h48qual) and ~0.55 (zig075) -- values that would themselves have been INSIDE the "
    "original [0.45, 0.95] band. The actual reason for the original band's failure is a "
    "different, unflagged interaction in the sidecar script: exposure_ok() ANDs the notional "
    "check with --require-dynamic-leverage-mapping, and when the notional-filtered eligible set "
    "is empty, disabling the notional band (0.0/0.0) makes the RuntimeError() branch itself "
    "unreachable, so the code silently falls through to a trades-floor-only eligible set that "
    "does NOT re-check the dynamic-leverage requirement either. Consequence: the h48qual "
    "reduced80 sidecar's selected mapping ended up with leverage_min==leverage_max==2.0 and "
    "long/short leverage scale both 1.0 -- i.e. FIXED, not dynamic, leverage -- unlike the "
    "original h48qual sidecar and unlike the reduced80 zig075 sidecar (which did land a real "
    "dynamic-leverage mapping, leverage_min=1.75/max=2.25). This is a second, more material "
    "deviation from baseline parity for h48qual specifically, disclosed here rather than hidden."
)

CONFIGS = {
    "baseline102": {
        "h48qual": {
            "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/true_3head_tabm_bundle.pt",
            "pred_dir": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630",
            "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_h48qual_q050_precomputed_20260630/risk_sidecar.pkl",
            "tag": "q050",
            "direction_label_dir": ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531",
            "quality_mode": "quality_label_action",
            "quality_label_dir": ROOT / "tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps",
        },
        "zig075": {
            "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt",
            "pred_dir": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629",
            "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_zig075_q075_precomputed_20260630/risk_sidecar.pkl",
            "tag": "q075",
            "direction_label_dir": ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531",
            "quality_mode": "same_as_direction",
            "quality_label_dir": None,
        },
    },
    "reduced80": {
        "h48qual": {
            "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_reduced80_h48qual_20260724/true_3head_tabm_bundle.pt",
            "pred_dir": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_reduced80_h48qual_20260724",
            "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_reduced80_h48qual_q050_20260724/risk_sidecar.pkl",
            "tag": "q050",
            "direction_label_dir": ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531",
            "quality_mode": "quality_label_action",
            "quality_label_dir": ROOT / "tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps",
        },
        "zig075": {
            "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_reduced80_zig075_20260724/true_3head_tabm_bundle.pt",
            "pred_dir": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_reduced80_zig075_20260724",
            "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_reduced80_zig075_q075_20260724/risk_sidecar.pkl",
            "tag": "q075",
            "direction_label_dir": ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531",
            "quality_mode": "same_as_direction",
            "quality_label_dir": None,
        },
    },
}

ATR_CFG = dict(atr_window=192, tp_mult=12.0, sl_mult=6.0, min_tp=0.075, min_sl=0.040, max_tp=0.22, max_sl=0.12)

# The original training pipeline's OOS eval_csv (alpha6/7 lineage) only extends to 2026-02-28, so
# it cannot cover the task's required OOS window through 2026-03-31. Per the project's own
# precedent (scripts/build_omega4_6_1_extended_parent_predictions_20260706.py,
# tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/), OOS scoring for BOTH model sets
# here instead uses the current-code rebuilt 2026 feature frame (training_features_2026_rebuilt.csv
# + regime3 overlay), re-inferring parent direction/quality predictions live via the frozen
# TabM bundles (base_cols read from each bundle -- correctly picks up 80 vs 102 automatically).
# Known caveat carried over from that precedent doc: ou_halflife/kel/evt_excess_z/btc_corr_60/
# dual_momentum differ somewhat from the legacy alpha6/7 feature vintage the models were
# ORIGINALLY trained+labeled against (ou_halflife corr=-0.03 on the Jan-Feb overlap) -- this
# applies identically to baseline102 and reduced80, so it does not bias the comparison between
# them, only the absolute numbers vs. any older report that used the legacy OOS file.
OOS_BASE_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
OOS_WIDE24_2026 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"


def _build_oos_frame() -> pd.DataFrame:
    frame = pd.read_csv(OOS_BASE_2026, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    overlay = pd.read_csv(OOS_WIDE24_2026, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    merged = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    if merged[cols].isna().any().any():
        raise RuntimeError("regime3 overlay has gaps after merge")
    return merged


def _infer_oos_prediction(frame_full: pd.DataFrame, cfg: dict, device: torch.device) -> pd.DataFrame:
    """Live causal re-inference of the frozen parent bundle's direction/quality heads on the
    extended current-code OOS frame -- no stored prediction file, no future rows, matches
    build_omega4_6_1_extended_parent_predictions_20260706.py's method exactly."""
    bundle = torch.load(Path(cfg["bundle"]), map_location=device, weights_only=False)
    base_cols, models = list(bundle["base_cols"]), bundle["models"]
    route = hard._route_id(frame_full)
    x = parent._base_input(frame_full, base_cols)
    preds = {expert: parent._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
    direction = parent._routed(preds, route, "direction", 3)
    quality = parent._routed(preds, route, "quality", 3)
    threshold = 0.50 if str(cfg["tag"]) == "q050" else 0.75
    oof = parent._prediction_output(frame_full, direction, quality, threshold=float(threshold), prefix="omega1_regime3_expertdq_oof")
    return oof.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in oof.columns})


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


def _load_component_frame(cfg: dict, device: torch.device, split_key: str) -> pd.DataFrame:
    """Load the exact frame the parent model's frames dict produces for this split, using the
    SAME omega4 loading path the training script itself used (guarantees byte-identical row set
    to whatever validation_predictions_*/oos_predictions_* were computed against)."""
    omega4 = importlib.import_module("train_eval_omega4_3head_parent72_loose_entry_quality_20260620")
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=Path(cfg["direction_label_dir"]),
        quality_mode=str(cfg["quality_mode"]),
        quality_label_dir=Path(cfg["quality_label_dir"]) if cfg["quality_label_dir"] is not None else None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    return frames[split_key]


def _prepare_component(frame_full: pd.DataFrame, cfg: dict, split: str, device: torch.device) -> dict[str, Any]:
    bundle = torch.load(Path(cfg["bundle"]), map_location=device, weights_only=False)
    base_cols, models = list(bundle["base_cols"]), bundle["models"]
    if split == "oos":
        # Extended current-code frame + live re-inference (see _infer_oos_prediction docstring).
        # Tolerant inner-join alignment (a handful of rows can differ due to label warm-up/tail
        # trimming upstream), matching retest_omega4_6_1_extended_oos_20260706.py's own method.
        pred = _infer_oos_prediction(frame_full, cfg, device)
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        keep_ts = set(pred["timestamp"])
        frame_full = frame_full[frame_full["timestamp"].isin(keep_ts)].reset_index(drop=True)
        pred = pred[pred["timestamp"].isin(set(frame_full["timestamp"]))].reset_index(drop=True)
        if len(pred) != len(frame_full) or not pred["timestamp"].equals(frame_full["timestamp"]):
            raise RuntimeError(f"{cfg['tag']}: oos prediction/frame timestamp mismatch after align")
    else:
        pred_full = pd.read_csv(Path(cfg["pred_dir"]) / f"{split}_predictions_{cfg['tag']}.csv")
        pred_full["timestamp"] = pd.to_datetime(pred_full["timestamp"])
        keep_ts = set(pred_full["timestamp"])
        frame_full = frame_full[frame_full["timestamp"].isin(keep_ts)].reset_index(drop=True)
        pred = pred_full[pred_full["timestamp"].isin(set(frame_full["timestamp"]))].reset_index(drop=True)
        if len(pred) != len(frame_full) or not pred["timestamp"].equals(frame_full["timestamp"]):
            raise RuntimeError(f"{cfg['tag']}: {split} prediction/frame timestamp mismatch after align")

    x = parent._base_input(frame_full, base_cols)
    dec_base = parent._to_decisions(pred, oof=(split != "oos"))
    dec, _ = atr_eval._apply_atr_safety_sltp(dec_base, frame_full, **ATR_CFG)
    atr = atr_eval._atr_pct(frame_full, ATR_CFG["atr_window"])
    loaded = parent._load_payloads(models, device=device)

    with open(Path(cfg["sidecar_pkl"]), "rb") as f:
        pkl = pickle.load(f)
    features = sidecar._risk_feature_frame(frame_full, pred, dec, base_cols, atr_pct=atr, feature_mode=pkl["risk_feature_mode"])
    x_all, _ = sidecar._feature_matrix(features, pkl["feature_columns"])
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    score = sidecar._predict_side_split_models(pkl["model"], x_all, side_all) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all), dtype=np.float64)
    mapping = pkl["selected_mapping"]
    margin = sidecar._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
    leverage = sidecar._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS}) if pkl["dynamic_leverage"] else np.ones(len(dec))
    base_np, exit_runtime, pos_idx = sidecar._prepare_exit_runtime(x, loaded)
    return {
        "frame": frame_full, "dec": dec, "margin": margin, "leverage": leverage, "base_np": base_np,
        "exit_runtime": exit_runtime, "pos_idx": pos_idx, "route": hard._route_id(frame_full), "exit_threshold": 0.95,
    }


def _slice_component(comp: dict, mask: np.ndarray) -> dict:
    idx = np.flatnonzero(mask)
    return {
        "frame": comp["frame"].iloc[idx].reset_index(drop=True),
        "dec": comp["dec"].iloc[idx].reset_index(drop=True),
        "margin": comp["margin"][idx],
        "leverage": comp["leverage"][idx],
        "base_np": comp["base_np"][idx],
        "exit_runtime": comp["exit_runtime"],
        "pos_idx": comp["pos_idx"],
        "route": comp["route"][idx],
        "exit_threshold": comp["exit_threshold"],
    }


@torch.no_grad()
def greedy_replay(frame: pd.DataFrame, components: dict[str, Any], *, fee: float, slip: float, cost_mult: float, device: torch.device) -> pd.DataFrame:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(frame)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash = 1.0
    pos = 0
    active_comp = None
    entry_price = entry_equity = 1.0
    entry_i = entry_signal_i = 0
    notional = leverage_v = margin_fraction = 0.0
    take_profit = stop_loss = 0.0
    mfe = mae = 0.0
    rows: list[dict] = []

    for i in range(0, n - 2):
        if pos != 0:
            comp = components[active_comp]
            move = (arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price
            mfe, mae = max(mfe, move), min(mae, move)
            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            else:
                hold = max(i - entry_i, 0)
                giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                prob = sidecar._predict_exit_prob_one(
                    comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                    pos_values=[float(pos), float(hold), float(move), float(mfe), float(mae),
                                float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move),
                                float(move + abs(stop_loss)), float(notional), float(leverage_v),
                                float(notional * leverage_v), float(take_profit), float(stop_loss)],
                    device=device,
                )
                if prob >= comp["exit_threshold"]:
                    reason = "exit_head"
            if reason:
                exit_px = arrays["close"][i] * (1 - slip_eff if pos > 0 else 1 + slip_eff)
                raw_exit = (exit_px - entry_price) / entry_price if pos > 0 else (entry_price - exit_px) / entry_price
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee_eff * notional
                trade_return = cash / max(entry_equity, 1e-12) - 1.0
                rows.append({"entry_signal_i": entry_signal_i, "entry_i": entry_i, "exit_i": i,
                             "entry_timestamp": str(frame["timestamp"].iloc[entry_signal_i]),
                             "exit_timestamp": str(frame["timestamp"].iloc[i]), "side": int(pos),
                             "source_component": active_comp, "reason": reason,
                             "win": int(cash > entry_equity), "trade_return": float(trade_return),
                             "notional": float(notional), "margin_fraction": float(margin_fraction),
                             "leverage": float(leverage_v)})
                pos, active_comp = 0, None
            continue

        for name in PRIORITY:
            comp = components[name]
            side = int(comp["dec"]["side"].iloc[i])
            active = omega._active(comp["dec"])
            active_i = bool(active.iloc[i]) if hasattr(active, "iloc") else bool(active[i])
            if side == 0 or not active_i:
                continue
            row_margin, row_leverage = float(comp["margin"][i]), float(comp["leverage"][i])
            if row_margin <= 0.0:
                continue
            scale = SCALE_MAP.get(f"{name}_{'L' if side > 0 else 'S'}", 1.0)
            row_leverage = min(row_leverage * scale, LEVERAGE_CAP)
            row_notional = min(row_margin * row_leverage, NOTIONAL_CAP)
            row_leverage = row_notional / max(row_margin, 1e-12)
            if row_notional <= 0.0:
                continue
            entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + slip_eff if side > 0 else 1 - slip_eff)
            pos, active_comp = side, name
            entry_price, entry_equity = float(entry_px), cash
            entry_i, entry_signal_i = min(i + 1, n - 1), i
            margin_fraction, leverage_v, notional = row_margin, row_leverage, row_notional
            take_profit = float(comp["dec"]["take_profit"].iloc[i])
            stop_loss = float(comp["dec"]["stop_loss"].iloc[i])
            cash -= cash * fee_eff * notional
            mfe = mae = 0.0
            break

    return pd.DataFrame(rows)


def _duration_gate(ledger: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {"no_gate": _compound_metrics(ledger), "with_gate": _compound_metrics(ledger)}
    market = frame[["timestamp", "ou_halflife"]]
    lg = ledger.copy()
    lg["entry_timestamp_dt"] = pd.to_datetime(lg["entry_timestamp"])
    lg = lg.merge(market.rename(columns={"timestamp": "entry_timestamp_dt"}), on="entry_timestamp_dt", how="left")
    hit = lg["ou_halflife"] <= DURATION_THRESHOLD
    gated_returns = np.where(hit, 0.0, lg["trade_return"])
    curve = np.concatenate([[1.0], np.cumprod(1.0 + gated_returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    n_active = int((~hit).sum())
    with_gate = {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0), "trades": n_active,
                 "wr": float((gated_returns[~hit] > 0).mean()) if n_active else 0.0, "skipped": int(hit.sum())}
    return {"no_gate": _compound_metrics(ledger), "with_gate": with_gate}


def run_model_set(name: str, device: torch.device, fee: float, slip: float) -> dict[str, Any]:
    cfgs = CONFIGS[name]
    result: dict[str, Any] = {}
    oos_frame_cache = None
    for split, frame_key, start, end in (("validation", "val_raw", VAL_START, VAL_END), ("oos", "oos_raw", OOS_START, OOS_END)):
        comp_fulls: dict[str, dict[str, Any]] = {}
        for cname, cfg in cfgs.items():
            if split == "oos":
                if oos_frame_cache is None:
                    oos_frame_cache = _build_oos_frame()
                frame_full = oos_frame_cache
            else:
                frame_full = _load_component_frame(cfg, device, frame_key)
            comp_fulls[cname] = _prepare_component(frame_full, cfg, split, device)

        # Intersect timestamps across all components (their per-component inner-join alignment
        # against each own prediction file can trim a handful of different rows), then restrict
        # to the target window. Every component is re-indexed onto this exact common timestamp
        # list so array positions line up 1:1 in greedy_replay.
        common_ts = None
        for cname, comp_full in comp_fulls.items():
            ts = pd.to_datetime(comp_full["frame"]["timestamp"])
            common_ts = set(ts) if common_ts is None else (common_ts & set(ts))
        common_ts = sorted(t for t in common_ts if pd.Timestamp(start) <= t <= pd.Timestamp(end))
        if not common_ts:
            raise RuntimeError(f"{name}/{split}: empty common timestamp window after alignment")
        common_index = pd.DatetimeIndex(common_ts)
        first_frame = next(iter(comp_fulls.values()))["frame"]
        frame_windowed = first_frame.set_index("timestamp").reindex(common_index).reset_index().rename(columns={"index": "timestamp"})

        components: dict[str, Any] = {}
        for cname, comp_full in comp_fulls.items():
            comp_frame = comp_full["frame"]
            positions = comp_frame.set_index("timestamp").index.get_indexer(common_index)
            if (positions < 0).any():
                raise RuntimeError(f"{name}/{split}/{cname}: common timestamp missing from component frame")
            mask_positions = np.zeros(len(comp_frame), dtype=bool)
            mask_positions[positions] = True
            sliced = _slice_component(comp_full, mask_positions)
            # _slice_component preserves comp_frame's own row order restricted to mask_positions;
            # since comp_frame is sorted ascending and common_index is also sorted ascending,
            # boolean masking already yields rows in common_index order.
            components[cname] = sliced
        ledger = greedy_replay(frame_windowed, components, fee=fee, slip=slip, cost_mult=1.0, device=device)
        gate = _duration_gate(ledger, frame_windowed)
        ledger.to_csv(OUT_DIR / f"{name}_{split}_ledger.csv", index=False)
        result[split] = {
            **gate,
            "rows": int(len(frame_windowed)),
            "date_range": [str(frame_windowed["timestamp"].iloc[0]) if len(frame_windowed) else None,
                           str(frame_windowed["timestamp"].iloc[-1]) if len(frame_windowed) else None],
            "source_counts": ledger["source_component"].value_counts().to_dict() if not ledger.empty else {},
        }
    return result


def main() -> int:
    device = parent._device("cpu")
    fee, slip = omega._load_fee_slip()
    report: dict[str, Any] = {
        "method": "eth_two_component_greedy_router_fresh_forward_bar_by_bar",
        "windows": {"validation": [VAL_START, VAL_END], "oos": [OOS_START, OOS_END]},
        "scale_map": SCALE_MAP, "leverage_cap": LEVERAGE_CAP, "notional_cap": NOTIONAL_CAP,
        "duration_threshold": DURATION_THRESHOLD, "priority": list(PRIORITY),
        "notional_band_deviation_reduced80": NOTIONAL_BAND_DEVIATION,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }
    for model_set in ("baseline102", "reduced80"):
        print(f"=== running {model_set} ===", flush=True)
        report[model_set] = run_model_set(model_set, device, fee, slip)
        print(json.dumps(report[model_set], indent=2, default=_json_default), flush=True)
    (OUT_DIR / "fresh_forward_comparison_report.json").write_text(json.dumps(report, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"\nWROTE {OUT_DIR / 'fresh_forward_comparison_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
