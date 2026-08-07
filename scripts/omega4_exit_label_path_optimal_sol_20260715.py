"""SOL port of _build_exit_dataset_entry_label_path_optimal from
train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py (lines 606-787): the
dynamic-programming "suffix maximum" optimal-stopping exit label. Ported verbatim except that
`omega`/`exit_head` are bound to SOL's own risk-template/execution module
(train_eval_omega1_2_tabm_diffusion_risk_sol_20260707: BASE_TEMPLATE notional=0.45/leverage=2.0/
tp=0.026/sl=0.014) instead of ETH's, per the lesson in
project-sol-omega4-6-1-full-stack-20260707: never assume ETH-tuned constants transfer.

This is the RL/DP-as-refinement design (chosen over full DP-label replacement, which failed --
see build_sol_dp_trajectory_labels_20260715.py / _lowfreq variant): zigzag_action already
identifies WHERE a large-enough, learnable swing exists (segment boundaries). Within each
zigzag segment, the direction is fixed (taken from zigzag), and this function uses full
knowledge of the realized price path inside that segment to compute, for every in-position bar,
whether exiting now is truly reward-optimal versus the best achievable exit later in the same
segment (a Bellman suffix-maximum, i.e. the exact solution of a known-model MDP restricted to
the segment). The resulting exit_action label replaces the heuristic
entry_label_terminal_giveback rule as the exit-head's training target.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_sol_20260707 as omega  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402


def build_exit_dataset_entry_label_path_optimal(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    exit_edge_min: float,
    max_samples: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]]:
    required = {"timestamp", "zigzag_action", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"entry-label path-optimal exit dataset missing columns: {missing}")
    if len(frame) != len(state):
        raise RuntimeError("entry-label path-optimal exit frame/state length mismatch")
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    action = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    notional = float(omega.BASE_TEMPLATE["notional"])
    leverage = float(omega.BASE_TEMPLATE["leverage"])
    take_profit = float(omega.BASE_TEMPLATE["take_profit"])
    stop_loss = float(omega.BASE_TEMPLATE["stop_loss"])

    rows: list[dict[str, float]] = []
    labels: list[int] = []
    frame_rows: list[pd.Series] = []
    exit_edges: list[float] = []
    reason_counts: dict[str, int] = {}
    segment_count = 0
    skipped_segments = 0
    positive_count = 0
    segment_id = -1
    i = 0
    last_i = len(frame) - 2
    while i < last_i:
        side_action = int(action[i])
        if side_action not in (1, 2):
            i += 1
            continue
        start_i = i
        while i < last_i and int(action[i]) == side_action:
            i += 1
        end_i = min(i - 1, last_i)
        side = 1 if side_action == 1 else -1
        segment_id += 1
        filled, entry_price, entry_fee, _route = omega._try_execution(
            arrays,
            int(start_i),
            side,
            entry=True,
            fee_base=fee_eff,
            slip_base=slip_eff,
        )
        entry_i = min(int(start_i) + 1, len(frame) - 1)
        if not filled or end_i < entry_i:
            skipped_segments += 1
            continue
        cash_after_entry_fee = 1.0 - 1.0 * float(entry_fee) * notional
        path_idx = np.arange(entry_i, end_i + 1, dtype=np.int64)
        exit_net = np.zeros(len(path_idx), dtype=np.float64)
        exit_fill_i = np.zeros(len(path_idx), dtype=np.int64)
        for k, row_i in enumerate(path_idx):
            net, fill_i, _reason = exit_head._exit_fill_net(
                arrays,
                signal_i=int(row_i),
                side=side,
                entry_price=float(entry_price),
                cash_after_entry_fee=cash_after_entry_fee,
                notional=notional,
                fee_eff=fee_eff,
                slip_eff=slip_eff,
            )
            exit_net[k] = float(net)
            exit_fill_i[k] = int(fill_i)
        suffix_best_value = np.maximum.accumulate(exit_net[::-1])[::-1]
        suffix_best_pos_from_here = np.zeros(len(exit_net), dtype=np.int64)
        best_pos = len(exit_net) - 1
        best_value = exit_net[best_pos]
        for k in range(len(exit_net) - 1, -1, -1):
            if exit_net[k] >= best_value:
                best_value = exit_net[k]
                best_pos = k
            suffix_best_pos_from_here[k] = best_pos

        entry_state = state.iloc[int(start_i)]
        mfe = 0.0
        mae = 0.0
        for k, row_i in enumerate(path_idx):
            px = float(arrays["close"][int(row_i)])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            if k == len(path_idx) - 1:
                future_value = float("-inf")
                edge = 0.0
                label = 1
                reason = "segment_end_forced_exit"
                best_future_i = int(row_i)
                best_future_fill_i = int(exit_fill_i[k])
                best_future_net = float(exit_net[k])
            else:
                future_pos = int(suffix_best_pos_from_here[k + 1])
                best_future_net = float(suffix_best_value[k + 1])
                future_value = best_future_net
                edge = float(exit_net[k] - future_value)
                label = int(edge >= float(exit_edge_min))
                reason = "oracle_dp_exit_now" if label else "oracle_dp_hold"
                best_future_i = int(path_idx[future_pos])
                best_future_fill_i = int(exit_fill_i[future_pos])
            row = exit_head._position_feature_row(
                state,
                entry_state,
                row_i=int(row_i),
                side=side,
                entry_price=float(entry_price),
                entry_i=int(entry_i),
                notional=notional,
                leverage=leverage,
                take_profit=take_profit,
                stop_loss=stop_loss,
                mfe=mfe,
                mae=mae,
                unreal=unreal,
            )
            rows.append(row)
            labels.append(label)
            positive_count += int(label)
            exit_edges.append(float(edge))
            frow = frame.iloc[int(row_i)].copy()
            frow["exit_path_segment_id"] = int(segment_id)
            frow["exit_path_entry_signal_i"] = int(start_i)
            frow["exit_path_entry_i"] = int(entry_i)
            frow["exit_path_end_i"] = int(end_i)
            frow["exit_path_side"] = int(side)
            frow["exit_path_hold_bars"] = int(max(int(row_i) - int(entry_i), 0))
            frow["exit_path_entry_price"] = float(entry_price)
            frow["exit_path_now_net"] = float(exit_net[k])
            frow["exit_path_best_future_net"] = float(best_future_net)
            frow["exit_path_future_value"] = float(future_value) if np.isfinite(future_value) else float("nan")
            frow["exit_path_edge"] = float(edge)
            frow["exit_path_label"] = int(label)
            frow["exit_path_reason"] = reason
            frow["exit_path_best_future_i"] = int(best_future_i)
            frow["exit_path_best_future_fill_i"] = int(best_future_fill_i)
            frow["exit_path_mfe"] = float(mfe)
            frow["exit_path_mae"] = float(mae)
            frow["exit_path_unrealized"] = float(unreal)
            frame_rows.append(frow)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if max_samples > 0 and len(rows) >= int(max_samples):
                break
        segment_count += 1
        if max_samples > 0 and len(rows) >= int(max_samples):
            break
    if not rows:
        raise RuntimeError("empty entry-label path-optimal Exit Head dataset")
    x = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(labels, dtype=np.int64)
    f = pd.DataFrame(frame_rows).reset_index(drop=True)
    return x, y, f, {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "positive_count": int(positive_count),
        "negative_count": int(len(y) - positive_count),
        "exit_edge_mean": float(np.mean(exit_edges)) if exit_edges else 0.0,
        "exit_edge_p50": float(np.quantile(exit_edges, 0.50)) if exit_edges else 0.0,
        "exit_edge_p90": float(np.quantile(exit_edges, 0.90)) if exit_edges else 0.0,
        "exit_edge_p99": float(np.quantile(exit_edges, 0.99)) if exit_edges else 0.0,
        "continued_exit_reasons": reason_counts,
        "used_segments": int(segment_count),
        "skipped_segments": int(skipped_segments),
        "risk_template": {
            "notional": notional,
            "leverage": leverage,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
        },
        "label_mode": "entry_label_path_optimal_stopping_every_in_position_bar",
    }
