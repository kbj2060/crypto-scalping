#!/usr/bin/env python3
"""RESEARCH ONLY -- train a standalone reversal-risk classifier for the LIVE ETH Omega4.6.1
h48qual/zig075 components.

Motivation (see project memory project-eth-omega461-exit-logic-experiments-20260721.md, ROOT
CAUSE section): the live model's learned exit head is structurally inert at the live
EXIT_THRESHOLD=0.95 gate -- TP/SL always closes trades first, so no hand-tuned exit-timing rule
that only reshapes the exit-head's inputs/labels can ever change live behavior. This script
instead trains a SEPARATE classifier ("reduce is safer") whose job is to shrink position
notional directly via partial scale-out, bypassing the exit head entirely. It is only used by
the research eval harness in this repo (research_eth_omega461_reversal_risk_scaleout_eval_20260721.py);
it is NOT wired into trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py,
or .env.

Setup (SPLIT_TS, frame loading/alignment) forked from train_eval_omega1_2_tabm_exit_head_20260603.py
per the approved plan; components/frame-loading/replay conventions forked from
research_eth_omega461_exit_sweep_20260721.py ("sweep" below), which is itself the harness used by
research_eth_omega461_exit_ideas2_20260721.py ("ideas2" below).

Data windows:
  TRAIN (fit):        2025-01-01 .. 2025-09-30  (train_predictions_qXXX.csv, OOF, before SPLIT_TS)
  internal VAL (AUC):  2025-10-01 .. 2025-12-31  (validation_predictions_qXXX.csv, OOF) -- same
                        window as the eval script's VAL split; used here ONLY as a training-time
                        held-out AUC/PR check, not as a backtest.

Label (forward-looking at TRAIN time only, not a causality violation -- mirrors the existing
exit-head label convention of using the trade's own eventual outcome to shape training labels):
for each historical trade (replayed with the SAME baseline lifecycle as the live model: TP/SL/
exit_head@0.95, no trailing/reversal mechanism), once armed (mfe_so_far >= activate_frac * TP,
activate_frac=0.7 fixed at train time), label=1 ("reduce is safer") if the eventual giveback from
that bar to the trade's actual close is >= giveback_thr (giveback_thr=0.4 fixed at train time),
else 0. giveback_i = (move_i - final_move) / max(abs(move_i), eps), where move_i is the unrealized
price-move at bar i and final_move is the trade's actual terminal raw price-move (before fees).

Features: the pos_state columns matching omega4_6_1_live.py::_Component.exit_probability
(pos_side, pos_hold_bars, pos_unrealized, pos_mfe, pos_mae, pos_giveback, pos_dist_to_tp,
pos_dist_to_sl, pos_notional, pos_leverage, pos_exposure, pos_tp, pos_sl) + the full numeric
base_cols feature set already fed to the exit head (same base_x the exit head runtime consumes)
+ two proxy columns confirmed to exist in the prediction CSVs with "if entering fresh at this
bar" semantics (the parent model's own per-bar quality/direction heads):
  omega1_regime3_expertdq_oof_quality_for_action
  omega1_regime3_expertdq_oof_dir_p_long / _dir_p_short (side-appropriate: dir_p_long if the
    trade is long, dir_p_short if short)

Model: HistGradientBoostingClassifier (one scalar in-position task; the shared 3-head TabM trunk
is not warranted here). Escalates to a small MLP only if internal VAL AUC < 0.55 (none needed --
see report.json).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head_ref  # noqa: E402 (SPLIT_TS reference)

OUT_ROOT = ROOT / "tmp/causal_regen_20260516"
SPLIT_TS = exit_head_ref.SPLIT_TS  # pd.Timestamp("2025-10-01"), per plan
ACTIVATE_FRAC_TRAIN = 0.7
GIVEBACK_THR_TRAIN = 0.4
PROXY_QUALITY_COL = "omega1_regime3_expertdq_oof_quality_for_action"
PROXY_DIR_LONG_COL = "omega1_regime3_expertdq_oof_dir_p_long"
PROXY_DIR_SHORT_COL = "omega1_regime3_expertdq_oof_dir_p_short"
MIN_AUC_FOR_HGB = 0.55


def _load_proxy_columns(pred_csv: Path, keep_ts: set) -> pd.DataFrame:
    src = pd.read_csv(pred_csv, usecols=["timestamp", PROXY_QUALITY_COL, PROXY_DIR_LONG_COL, PROXY_DIR_SHORT_COL])
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    src = src[src["timestamp"].isin(keep_ts)].reset_index(drop=True)
    return src


@torch.no_grad()
def build_dataset(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple],
    proxy: pd.DataFrame,
    *,
    risk_margin_fraction: np.ndarray,
    risk_leverage: np.ndarray,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    device: torch.device,
    activate_frac: float,
    giveback_thr: float,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Causal bar-by-bar replay identical in lifecycle logic to
    sweep.replay_exit_variant(exit_threshold=0.95, no trailing) -- i.e. the exact baseline trade
    lifecycle the live model follows -- with per-bar (post-arm) feature/label recording added.
    fresh_forward_bar_by_bar=true; labels use only each trade's OWN eventual close (train-time
    label shaping only, matches the existing exit-head label convention), no other trade's
    future data and no saved ledger is used as an input to the simulation itself.
    """
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    exit_threshold = sweep.BASELINE_EXIT_THRESHOLD
    cash = 1.0
    pos = 0
    entry_price = 0.0
    entry_i = 0
    entry_signal_i = 0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    armed = False
    pending_rows: list[dict[str, float]] = []
    pending_moves: list[float] = []
    all_rows: list[dict[str, float]] = []
    all_labels: list[int] = []
    trades = 0
    armed_trades = 0
    route = hard._route_id(frame)
    from train_eval_omega1_2_tabm_diffusion_risk_20260603 import _try_execution as omega_try_execution, _fill_price as omega_fill_price
    import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit

    base_np, exit_runtime, pos_idx = rs._prepare_exit_runtime(base_x, loaded_models)
    proxy_quality = proxy[PROXY_QUALITY_COL].to_numpy(dtype=np.float64)
    proxy_dir_long = proxy[PROXY_DIR_LONG_COL].to_numpy(dtype=np.float64)
    proxy_dir_short = proxy[PROXY_DIR_SHORT_COL].to_numpy(dtype=np.float64)
    base_cols = list(base_x.columns)

    def flush_trade(final_move: float) -> None:
        nonlocal pending_rows, pending_moves
        for row, move_i in zip(pending_rows, pending_moves):
            giveback_i = (move_i - final_move) / max(abs(move_i), 1.0e-8)
            all_rows.append(row)
            all_labels.append(int(giveback_i >= float(giveback_thr)))
        pending_rows = []
        pending_moves = []

    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = price_exit._price_move(arrays, int(i), side=pos, entry_price=float(entry_price), slip_eff=slip_eff)
            mfe = max(mfe, move)
            mae = min(mae, move)
        else:
            move = 0.0

        if pos != 0:
            reason = ""
            hold = max(int(i) - int(entry_i), 0)
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            if not armed and take_profit > 0.0 and mfe >= float(activate_frac) * take_profit:
                armed = True
            if not reason and armed:
                giveback_now = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                base_row = {base_cols[j]: float(base_np[i, j]) for j in range(len(base_cols))}
                base_row.update({
                    "pos_side": float(pos), "pos_hold_bars": float(hold), "pos_unrealized": float(move),
                    "pos_mfe": float(mfe), "pos_mae": float(mae), "pos_giveback": float(np.clip(giveback_now, 0.0, 10.0)),
                    "pos_dist_to_tp": float(take_profit - move), "pos_dist_to_sl": float(move + abs(stop_loss)),
                    "pos_notional": float(notional), "pos_leverage": float(leverage), "pos_exposure": float(notional * leverage),
                    "pos_tp": float(take_profit), "pos_sl": float(stop_loss),
                    "proxy_quality_for_action": float(proxy_quality[i]),
                    "proxy_dir_p_side": float(proxy_dir_long[i] if pos > 0 else proxy_dir_short[i]),
                })
                pending_rows.append(base_row)
                pending_moves.append(float(move))
            if not reason:
                giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(route[i])]
                prob = rs._predict_exit_prob_one(
                    base_np, exit_runtime, pos_idx, row_i=int(i), expert=expert,
                    pos_values=[
                        float(pos), float(hold), float(move), float(mfe), float(mae),
                        float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move), float(move + abs(stop_loss)),
                        float(notional), float(leverage), float(notional * leverage), float(take_profit), float(stop_loss),
                    ],
                    device=device,
                )
                if prob >= float(exit_threshold):
                    reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _route = omega_try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                trades += 1
                if pending_rows:
                    armed_trades += 1
                    flush_trade(raw_exit)
                pos = 0
                armed = False
                continue
        if pos != 0 or not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, fee_paid, _route = omega_try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        row_leverage = float(risk_leverage[int(i)])
        row_margin = float(risk_margin_fraction[int(i)])
        row_notional = row_margin * row_leverage
        if row_notional <= 0.0:
            continue
        pos = side
        entry_price = float(px)
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_signal_i = int(i)
        leverage = row_leverage
        notional = row_notional
        base_tp = float(row.get("take_profit", 0.0) or 0.0)
        base_sl = float(row.get("stop_loss", 0.0) or 0.0)
        if bool(notional_scaled_sltp):
            take_profit = base_tp * row_notional
            stop_loss = base_sl * row_notional
        else:
            take_profit = base_tp
            stop_loss = base_sl
        mfe = 0.0
        mae = 0.0
        armed = False

    if pos != 0 and pending_rows:
        exit_px = omega_fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
        armed_trades += 1
        flush_trade(raw_exit)

    x = pd.DataFrame(all_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(all_labels, dtype=np.int64)
    diag = {
        "trades": int(trades), "armed_trades": int(armed_trades), "rows": int(len(y)),
        "positive_count": int(y.sum()) if len(y) else 0,
        "positive_rate": float(y.mean()) if len(y) else 0.0,
    }
    return x, y, diag


def prep_split(name: str, cfg: dict, frame: pd.DataFrame, pred_csv: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    p = sweep.prep_component(name, cfg, frame, pred_csv, oof=True)
    proxy = _load_proxy_columns(pred_csv, set(p["frame"]["timestamp"]))
    if len(proxy) != len(p["frame"]) or not proxy["timestamp"].equals(p["frame"]["timestamp"]):
        raise RuntimeError(f"{name}: proxy/frame timestamp mismatch ({len(proxy)} vs {len(p['frame'])})")
    return p, proxy


def train_component(name: str, cfg: dict) -> dict[str, Any]:
    full_2025 = sweep.load_frame("2025-01-01", "2025-12-31", base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    train_pred = sweep.EXT_PRED_DIR / name / f"train_predictions_{cfg['q_tag']}.csv"
    val_pred = sweep.EXT_PRED_DIR / name / f"validation_predictions_{cfg['q_tag']}.csv"

    train_frame = full_2025[full_2025["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_frame = full_2025[full_2025["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    print(f"stage=prep component={name} split=TRAIN", flush=True)
    p_train, proxy_train = prep_split(name, cfg, train_frame, train_pred)
    print(f"stage=prep component={name} split=VAL(internal)", flush=True)
    p_val, proxy_val = prep_split(name, cfg, val_frame, val_pred)

    print(f"stage=build_dataset component={name} split=TRAIN rows_frame={len(p_train['frame'])}", flush=True)
    x_train, y_train, diag_train = build_dataset(
        p_train["frame"], p_train["x"], p_train["dec"], p_train["loaded"], proxy_train,
        risk_margin_fraction=p_train["margin"], risk_leverage=p_train["leverage"],
        fee=p_train["fee"], slip=p_train["slip"], cost_mult=sweep.COST_MULT,
        notional_scaled_sltp=p_train["notional_scaled_sltp"], device=sweep.DEVICE,
        activate_frac=ACTIVATE_FRAC_TRAIN, giveback_thr=GIVEBACK_THR_TRAIN,
    )
    print(f"stage=build_dataset component={name} split=VAL(internal) rows_frame={len(p_val['frame'])}", flush=True)
    x_val, y_val, diag_val = build_dataset(
        p_val["frame"], p_val["x"], p_val["dec"], p_val["loaded"], proxy_val,
        risk_margin_fraction=p_val["margin"], risk_leverage=p_val["leverage"],
        fee=p_val["fee"], slip=p_val["slip"], cost_mult=sweep.COST_MULT,
        notional_scaled_sltp=p_val["notional_scaled_sltp"], device=sweep.DEVICE,
        activate_frac=ACTIVATE_FRAC_TRAIN, giveback_thr=GIVEBACK_THR_TRAIN,
    )
    print(f"stage=train_diag component={name} train={diag_train} val={diag_val}", flush=True)

    feature_cols = list(x_train.columns)
    x_val = x_val.reindex(columns=feature_cols, fill_value=0.0)

    model_kind = "hgb"
    clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300, random_state=260721)
    if len(np.unique(y_train)) < 2:
        raise RuntimeError(f"{name}: training labels are single-class, cannot fit a classifier (positive_rate={diag_train['positive_rate']})")
    clf.fit(x_train.to_numpy(dtype=np.float64), y_train)

    val_proba = clf.predict_proba(x_val.to_numpy(dtype=np.float64))[:, 1] if len(y_val) and len(np.unique(y_val)) > 1 else None
    val_auc = float(roc_auc_score(y_val, val_proba)) if val_proba is not None else float("nan")
    val_ap = float(average_precision_score(y_val, val_proba)) if val_proba is not None else float("nan")

    escalated = False
    if not np.isnan(val_auc) and val_auc < MIN_AUC_FOR_HGB:
        escalated = True
        from sklearn.neural_network import MLPClassifier
        model_kind = "mlp"
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1.0e-3, max_iter=500, random_state=260721, early_stopping=True)
        clf.fit(x_train.to_numpy(dtype=np.float64), y_train)
        val_proba = clf.predict_proba(x_val.to_numpy(dtype=np.float64))[:, 1] if len(y_val) and len(np.unique(y_val)) > 1 else None
        val_auc = float(roc_auc_score(y_val, val_proba)) if val_proba is not None else float("nan")
        val_ap = float(average_precision_score(y_val, val_proba)) if val_proba is not None else float("nan")

    out_dir = OUT_ROOT / f"eth_omega461_reversal_risk_scaleout_20260721_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump({"model": clf, "model_kind": model_kind, "feature_columns": feature_cols}, f)

    report = {
        "component": name, "model_kind": model_kind, "escalated_from_hgb": escalated,
        "activate_frac_train": ACTIVATE_FRAC_TRAIN, "giveback_thr_train": GIVEBACK_THR_TRAIN,
        "split_ts": str(SPLIT_TS), "train_window": ["2025-01-01", str(SPLIT_TS.date())],
        "internal_val_window": [str(SPLIT_TS.date()), "2025-12-31"],
        "train_diag": diag_train, "internal_val_diag": diag_val,
        "internal_val_auc": val_auc, "internal_val_ap": val_ap,
        "feature_count": len(feature_cols),
        "proxy_columns_used": [PROXY_QUALITY_COL, PROXY_DIR_LONG_COL, PROXY_DIR_SHORT_COL],
        "flag_low_val_positive_count": bool(diag_val["positive_count"] < 50),
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"stage=done component={name} val_auc={val_auc:.4f} val_ap={val_ap:.4f} "
          f"val_positive_count={diag_val['positive_count']} model_kind={model_kind}", flush=True)
    return report


def main() -> int:
    reports = {}
    for name, cfg in sweep.COMPONENTS.items():
        reports[name] = train_component(name, cfg)
    print(json.dumps(reports, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
