#!/usr/bin/env python3
"""RESEARCH ONLY -- train a standalone "DP reversal-risk" scale-out classifier for the LIVE ETH
Omega4.6.1 h48qual/zig075 components (round 10 of the exit-logic research thread).

Motivation: round 3 (train_eth_omega461_reversal_risk_scaleout_20260721.py) trained a similar
scale-out classifier on a giveback-ratio label ("did this trade eventually give back a lot of
its peak profit") and failed OOS confirmation. This round uses a DIFFERENT, better-grounded
label source: the DP-optimal per-bar entry action (CASH / ENTER_LONG / ENTER_SHORT) from the
same finite-state FLAT/LONG/SHORT x age backward-induction value function already used in
scripts/build_omega1_2_1_dp_trajectory_labels_20260620.py (lines ~108-156) and verbatim-ported
for a visualization sample in scripts/adhoc_dp_label_chart_sample_20260721.py. Per user
direction: used here as a REVERSAL signal for an ALREADY-OPEN position, not as an entry filter.

Label definition (chosen and stated per the task): label=1 ("reversal risk -- reduce") iff the
DP-optimal action at that bar is the OPPOSITE side to the open position (LONG position + DP says
ENTER_SHORT, or vice versa). CASH (DP says "no fresh edge right now") is treated as label=0, NOT
as reversal risk. Rationale: CASH only means the DP no longer sees a fresh-entry edge -- it does
not mean the DP considers the trend to have reversed against an existing position. Requiring the
DP to positively favor the OPPOSITE direction is the more specific, more defensible reversal
signal (matches the user's own framing: "the DP considers the trend has reversed"). This is a
DIFFERENT labeling philosophy than round 3 (post-hoc giveback outcome) -- it is a forward-looking
DP value-function judgement at the bar itself, not the trade's own eventual realized outcome.

DP action computation is legitimately forward-looking (full backward induction over the window)
-- same convention as every other DP-label script in this repo: it is used ONLY to build TRAINING
labels, never as an input to the causal bar-by-bar backtest replay itself (see the eval script,
which only ever consults the trained classifier's predict_proba on causally-available features).

Real transaction costs (not the original DP label script's placeholder 0.0001*3 x notional
0.025*2): fee_per_side = FEE_RATE + SLIP_RATE from train_eval_omega1_2_tabm_diffusion_risk_20260603
(omega._load_fee_slip(), the same real project fee/slip this whole research thread's harness
uses), applied in raw price-move units (NOTIONAL=1.0 for the DP recursion -- only the resulting
CASH/LONG/SHORT action is needed here, not any TP/SL/PnL bookkeeping, so absolute notional scale
does not matter for correctness of the ACTION decision beyond how the two tiny constants
hold_penalty/min_entry_edge -- inherited unchanged from the original script -- interact with it;
both are negligible relative to typical multi-bar price moves either way).

Data windows (identical to round 3):
  TRAIN (fit):        2025-01-01 .. 2025-09-30  (train_predictions_qXXX.csv, OOF, before SPLIT_TS)
  internal VAL (AUC):  2025-10-01 .. 2025-12-31  (validation_predictions_qXXX.csv, OOF) -- same
                        window as the eval script's VAL split; used here ONLY as a training-time
                        held-out AUC/PR check, not as a backtest.

Features: IDENTICAL schema to round 3 -- the pos_state columns matching
omega4_6_1_live.py::_Component.exit_probability (pos_side, pos_hold_bars, pos_unrealized,
pos_mfe, pos_mae, pos_giveback, pos_dist_to_tp, pos_dist_to_sl, pos_notional, pos_leverage,
pos_exposure, pos_tp, pos_sl) + the full numeric base_cols feature set already fed to the exit
head + the same 2 proxy columns (quality_for_action, side-appropriate dir_p_long/dir_p_short).

Model: HistGradientBoostingClassifier first; escalates to a small MLP only if internal VAL
AUC < 0.55 (same escalation rule as round 3).

MAX_AGE is a CLI-configurable horizon (bars, 5m each). Step 4 of the task plan retries this
script with MAX_AGE in {48, 192} if MAX_AGE=96 (8h, the first config tried) does not yield an OOS
winner in the eval script.
"""

from __future__ import annotations

import argparse
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
ACTIVATE_FRAC_TRAIN = 0.7  # when to start collecting labeled rows (mfe >= this * TP), matches round 3
PROXY_QUALITY_COL = "omega1_regime3_expertdq_oof_quality_for_action"
PROXY_DIR_LONG_COL = "omega1_regime3_expertdq_oof_dir_p_long"
PROXY_DIR_SHORT_COL = "omega1_regime3_expertdq_oof_dir_p_short"
MIN_AUC_FOR_HGB = 0.55

# DP recursion constants (fee_per_side is real project cost, loaded at runtime; the rest are
# inherited unchanged, tiny secondary terms from build_omega1_2_1_dp_trajectory_labels_20260620.py).
DP_HOLD_PENALTY = 0.000002
DP_MIN_ENTRY_EDGE = 0.00008
DP_NOTIONAL = 1.0  # raw price-move units; only the resulting action matters here, not PnL scale


def _load_proxy_columns(pred_csv: Path, keep_ts: set) -> pd.DataFrame:
    src = pd.read_csv(pred_csv, usecols=["timestamp", PROXY_QUALITY_COL, PROXY_DIR_LONG_COL, PROXY_DIR_SHORT_COL])
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    src = src[src["timestamp"].isin(keep_ts)].reset_index(drop=True)
    return src


def dp_reversal_action_array(
    frame: pd.DataFrame, *, max_age: int, fee_per_side: float,
    hold_penalty: float = DP_HOLD_PENALTY, min_entry_edge: float = DP_MIN_ENTRY_EDGE,
) -> tuple[np.ndarray, int]:
    """Ported (backward-induction recursion only, not the TP/SL/MFE/MAE bookkeeping) from
    build_omega1_2_1_dp_trajectory_labels_20260620.py lines 108-156 / the standalone copy in
    adhoc_dp_label_chart_sample_20260721.py lines 56-90. Returns (p_flat, boundary_start):
    p_flat[i] in {0=CASH, 1=ENTER_LONG, 2=ENTER_SHORT} is the DP-optimal FLAT-state action at bar
    i; rows >= boundary_start sit in the unresolved value-function boundary zone (same
    `n - max_age - 2` cutoff the source script uses) and must be excluded from training.
    """
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    n = len(close)
    next_ret = np.zeros(n, dtype=np.float64)
    next_ret[:-1] = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0

    v_flat = np.zeros(n + 1, dtype=np.float64)
    v_long = np.zeros((n + 1, max_age + 2), dtype=np.float64)
    v_short = np.zeros((n + 1, max_age + 2), dtype=np.float64)
    p_flat = np.zeros(n, dtype=np.int8)

    entry_cost = float(fee_per_side) * DP_NOTIONAL
    exit_cost = float(fee_per_side) * DP_NOTIONAL
    for i in range(n - 2, -1, -1):
        ret = float(next_ret[i]) * DP_NOTIONAL
        cash_v = v_flat[i + 1]
        enter_long = -entry_cost + ret - hold_penalty + v_long[i + 1, 1]
        enter_short = -entry_cost - ret - hold_penalty + v_short[i + 1, 1]
        vals = (cash_v, enter_long, enter_short)
        best = int(np.argmax(vals))
        if best != 0 and vals[best] - cash_v < min_entry_edge:
            best = 0
        p_flat[i] = best
        v_flat[i] = vals[best]
        for age in range(max_age, 0, -1):
            exit_v = -exit_cost + v_flat[i + 1]
            if age >= max_age:
                v_long[i, age] = exit_v
                v_short[i, age] = exit_v
                continue
            hold_long = ret - hold_penalty + v_long[i + 1, age + 1]
            hold_short = -ret - hold_penalty + v_short[i + 1, age + 1]
            v_long[i, age] = exit_v if exit_v >= hold_long else hold_long
            v_short[i, age] = exit_v if exit_v >= hold_short else hold_short

    boundary_start = max(n - max_age - 2, 0)
    return p_flat, boundary_start


@torch.no_grad()
def build_dataset(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple],
    proxy: pd.DataFrame,
    dp_action: np.ndarray,
    dp_boundary: int,
    *,
    risk_margin_fraction: np.ndarray,
    risk_leverage: np.ndarray,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    device: torch.device,
    activate_frac: float,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Causal bar-by-bar replay identical in lifecycle logic to the baseline (TP/SL/exit_head@0.95,
    no trailing/scale-out) -- i.e. the exact live trade lifecycle -- with per-bar (post-arm)
    feature/label recording added. fresh_forward_bar_by_bar=true for the SIMULATION; the DP
    action array is a forward-looking (whole-window backward induction) TRAINING LABEL source
    only, not consulted by the simulation's entry/exit decisions themselves. No saved ledger is
    used as an input.
    """
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    exit_threshold = sweep.BASELINE_EXIT_THRESHOLD
    pos = 0
    entry_price = 0.0
    entry_i = 0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    armed = False
    all_rows: list[dict[str, float]] = []
    all_labels: list[int] = []
    trades = 0
    armed_trades = 0
    dp_pos_opposite_count = 0
    route = hard._route_id(frame)
    from train_eval_omega1_2_tabm_diffusion_risk_20260603 import _try_execution as omega_try_execution, _fill_price as omega_fill_price
    import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit

    base_np, exit_runtime, pos_idx = rs._prepare_exit_runtime(base_x, loaded_models)
    proxy_quality = proxy[PROXY_QUALITY_COL].to_numpy(dtype=np.float64)
    proxy_dir_long = proxy[PROXY_DIR_LONG_COL].to_numpy(dtype=np.float64)
    proxy_dir_short = proxy[PROXY_DIR_SHORT_COL].to_numpy(dtype=np.float64)
    base_cols = list(base_x.columns)
    armed_this_trade = False

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
                armed_this_trade = True
            if not reason and armed and int(i) < dp_boundary:
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
                opposite_action = 2 if pos > 0 else 1
                label = int(int(dp_action[i]) == opposite_action)
                dp_pos_opposite_count += label
                all_rows.append(base_row)
                all_labels.append(label)
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
                trades += 1
                armed_trades += int(armed_this_trade)
                pos = 0
                armed = False
                armed_this_trade = False
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
        armed_this_trade = False

    if pos != 0:
        trades += 1
        armed_trades += int(armed_this_trade)

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


def train_component(name: str, cfg: dict, *, max_age: int, fee_per_side_real: float) -> dict[str, Any]:
    full_2025 = sweep.load_frame("2025-01-01", "2025-12-31", base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    train_pred = sweep.EXT_PRED_DIR / name / f"train_predictions_{cfg['q_tag']}.csv"
    val_pred = sweep.EXT_PRED_DIR / name / f"validation_predictions_{cfg['q_tag']}.csv"

    train_frame = full_2025[full_2025["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_frame = full_2025[full_2025["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    print(f"stage=prep component={name} max_age={max_age} split=TRAIN", flush=True)
    p_train, proxy_train = prep_split(name, cfg, train_frame, train_pred)
    print(f"stage=prep component={name} max_age={max_age} split=VAL(internal)", flush=True)
    p_val, proxy_val = prep_split(name, cfg, val_frame, val_pred)

    print(f"stage=dp_recursion component={name} max_age={max_age} split=TRAIN rows={len(p_train['frame'])}", flush=True)
    dp_action_train, dp_boundary_train = dp_reversal_action_array(p_train["frame"], max_age=max_age, fee_per_side=fee_per_side_real)
    print(f"stage=dp_recursion component={name} max_age={max_age} split=VAL(internal) rows={len(p_val['frame'])}", flush=True)
    dp_action_val, dp_boundary_val = dp_reversal_action_array(p_val["frame"], max_age=max_age, fee_per_side=fee_per_side_real)
    dp_diag = {
        "train": {
            "cash": int((dp_action_train[:dp_boundary_train] == 0).sum()),
            "long": int((dp_action_train[:dp_boundary_train] == 1).sum()),
            "short": int((dp_action_train[:dp_boundary_train] == 2).sum()),
            "boundary_start": int(dp_boundary_train), "n": int(len(dp_action_train)),
        },
        "val": {
            "cash": int((dp_action_val[:dp_boundary_val] == 0).sum()),
            "long": int((dp_action_val[:dp_boundary_val] == 1).sum()),
            "short": int((dp_action_val[:dp_boundary_val] == 2).sum()),
            "boundary_start": int(dp_boundary_val), "n": int(len(dp_action_val)),
        },
    }
    print(f"stage=dp_diag component={name} max_age={max_age} {dp_diag}", flush=True)

    print(f"stage=build_dataset component={name} max_age={max_age} split=TRAIN rows_frame={len(p_train['frame'])}", flush=True)
    x_train, y_train, diag_train = build_dataset(
        p_train["frame"], p_train["x"], p_train["dec"], p_train["loaded"], proxy_train,
        dp_action_train, dp_boundary_train,
        risk_margin_fraction=p_train["margin"], risk_leverage=p_train["leverage"],
        fee=p_train["fee"], slip=p_train["slip"], cost_mult=sweep.COST_MULT,
        notional_scaled_sltp=p_train["notional_scaled_sltp"], device=sweep.DEVICE,
        activate_frac=ACTIVATE_FRAC_TRAIN,
    )
    print(f"stage=build_dataset component={name} max_age={max_age} split=VAL(internal) rows_frame={len(p_val['frame'])}", flush=True)
    x_val, y_val, diag_val = build_dataset(
        p_val["frame"], p_val["x"], p_val["dec"], p_val["loaded"], proxy_val,
        dp_action_val, dp_boundary_val,
        risk_margin_fraction=p_val["margin"], risk_leverage=p_val["leverage"],
        fee=p_val["fee"], slip=p_val["slip"], cost_mult=sweep.COST_MULT,
        notional_scaled_sltp=p_val["notional_scaled_sltp"], device=sweep.DEVICE,
        activate_frac=ACTIVATE_FRAC_TRAIN,
    )
    print(f"stage=train_diag component={name} max_age={max_age} train={diag_train} val={diag_val}", flush=True)

    feature_cols = list(x_train.columns)
    x_val = x_val.reindex(columns=feature_cols, fill_value=0.0)

    model_kind = "hgb"
    if len(np.unique(y_train)) < 2:
        raise RuntimeError(f"{name}: training labels are single-class, cannot fit a classifier (positive_rate={diag_train['positive_rate']})")
    clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300, random_state=260721)
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

    out_dir = OUT_ROOT / f"eth_omega461_dp_reversal_scaleout_20260721_maxage{max_age}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump({"model": clf, "model_kind": model_kind, "feature_columns": feature_cols}, f)

    report = {
        "component": name, "model_kind": model_kind, "escalated_from_hgb": escalated,
        "max_age": int(max_age), "fee_per_side_real": float(fee_per_side_real),
        "label_definition": "opposite_side_only (DP action == opposite of open position side; CASH -> label 0)",
        "activate_frac_train": ACTIVATE_FRAC_TRAIN,
        "split_ts": str(SPLIT_TS), "train_window": ["2025-01-01", str(SPLIT_TS.date())],
        "internal_val_window": [str(SPLIT_TS.date()), "2025-12-31"],
        "dp_diag": dp_diag,
        "train_diag": diag_train, "internal_val_diag": diag_val,
        "internal_val_auc": val_auc, "internal_val_ap": val_ap,
        "feature_count": len(feature_cols),
        "proxy_columns_used": [PROXY_QUALITY_COL, PROXY_DIR_LONG_COL, PROXY_DIR_SHORT_COL],
        "flag_low_val_positive_count": bool(diag_val["positive_count"] < 50),
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"stage=done component={name} max_age={max_age} val_auc={val_auc:.4f} val_ap={val_ap:.4f} "
          f"val_positive_count={diag_val['positive_count']} model_kind={model_kind}", flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age", type=int, default=96)
    args = ap.parse_args()
    fee, slip = omega._load_fee_slip()
    fee_per_side_real = float(fee) + float(slip)
    print(f"stage=start max_age={args.max_age} fee_per_side_real={fee_per_side_real} (fee={fee}, slip={slip})", flush=True)
    reports = {}
    for name, cfg in sweep.COMPONENTS.items():
        reports[name] = train_component(name, cfg, max_age=int(args.max_age), fee_per_side_real=fee_per_side_real)
    print(json.dumps(reports, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
