#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega4_1_exit_head_price_move_sltp_retrain_20260622"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASELINE_BUNDLE = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_3head_parent72_loose_entry_quality_20260620_smoke_loose_entry_loose_quality_terminal_giveback_exit_e2_train15k_exit15k_q070"
    / "true_3head_tabm_bundle.pt"
)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _price_move(
    arrays: dict[str, np.ndarray],
    row_i: int,
    *,
    side: int,
    entry_price: float,
    slip_eff: float,
) -> float:
    px = float(arrays["close"][int(row_i)])
    if int(side) > 0:
        return float((px * (1.0 - slip_eff) - float(entry_price)) / max(float(entry_price), 1.0e-12))
    return float((float(entry_price) - px * (1.0 + slip_eff)) / max(float(entry_price), 1.0e-12))


def _position_feature_row_price_move(
    state: pd.DataFrame,
    entry_state: pd.Series,
    *,
    row_i: int,
    side: int,
    entry_price: float,
    entry_i: int,
    notional: float,
    leverage: float,
    take_profit: float,
    stop_loss: float,
    mfe: float,
    mae: float,
    price_move: float,
) -> dict[str, float]:
    cur = state.iloc[int(row_i)]
    out: dict[str, float] = {f"cur_{c}": float(cur[c]) for c in state.columns}
    entry_cols = [c for c in state.columns if c.startswith("tabm_") or c.startswith("fixed_")]
    for col in entry_cols:
        out[f"entry_{col}"] = float(entry_state[col])
        out[f"drift_{col}"] = float(cur[col]) - float(entry_state[col])
    hold = max(int(row_i) - int(entry_i), 0)
    giveback = (float(mfe) - float(price_move)) / max(abs(float(mfe)), 1.0e-8) if float(mfe) > 0.0 else 0.0
    out.update(
        {
            "pos_side": float(side),
            "pos_hold_bars": float(hold),
            "pos_unrealized": float(price_move),
            "pos_mfe": float(mfe),
            "pos_mae": float(mae),
            "pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
            "pos_dist_to_tp": float(take_profit - price_move),
            "pos_dist_to_sl": float(price_move + abs(stop_loss)),
            "pos_notional": float(notional),
            "pos_leverage": float(leverage),
            "pos_exposure": float(notional * leverage),
            "pos_tp": float(take_profit),
            "pos_sl": float(stop_loss),
        }
    )
    return out


def _build_exit_dataset_price_move_terminal_giveback(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    max_samples: int,
    terminal_window: int,
    adverse_price_move: float,
    min_mfe_for_giveback: float,
    giveback_min: float,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]]:
    required = {"timestamp", "zigzag_action", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"price-move exit dataset missing columns: {missing}")
    if len(frame) != len(state):
        raise RuntimeError("price-move exit frame/state length mismatch")

    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    action = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    notional = float(omega.BASE_TEMPLATE["notional"])
    leverage = float(omega.BASE_TEMPLATE["leverage"])
    take_profit = float(omega.BASE_TEMPLATE["take_profit"])
    stop_loss = float(omega.BASE_TEMPLATE["stop_loss"])
    tw = max(int(terminal_window), 1)

    rows: list[dict[str, float]] = []
    labels: list[int] = []
    frame_rows: list[pd.Series] = []
    reason_counts: dict[str, int] = {}
    barrier_counts: dict[str, int] = {}
    used_segments = 0
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
        filled, entry_price, _entry_fee, _route = omega._try_execution(
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

        entry_state = state.iloc[int(start_i)]
        mfe = 0.0
        mae = 0.0
        segment_rows = 0
        for row_i in range(entry_i, end_i + 1):
            move = _price_move(arrays, int(row_i), side=side, entry_price=float(entry_price), slip_eff=slip_eff)
            if move >= take_profit:
                barrier_counts["take_profit"] = barrier_counts.get("take_profit", 0) + 1
                break
            if move <= -abs(stop_loss):
                barrier_counts["stop_loss"] = barrier_counts.get("stop_loss", 0) + 1
                break
            mfe = max(mfe, move)
            mae = min(mae, move)
            giveback = (mfe - move) / max(abs(mfe), 1.0e-8) if mfe > 0.0 else 0.0
            bars_to_segment_end = int(end_i) - int(row_i)
            terminal = bars_to_segment_end < tw
            adverse = move <= float(adverse_price_move)
            gave_back = mfe >= float(min_mfe_for_giveback) and giveback >= float(giveback_min) and move > 0.0
            if terminal:
                label = 1
                reason = "terminal_window_exit"
            elif adverse:
                label = 1
                reason = "adverse_price_move_exit"
            elif gave_back:
                label = 1
                reason = "mfe_giveback_exit"
            else:
                label = 0
                reason = "hold"
            row = _position_feature_row_price_move(
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
                price_move=move,
            )
            rows.append(row)
            labels.append(label)
            positive_count += int(label)
            segment_rows += 1
            frow = frame.iloc[int(row_i)].copy()
            frow["exit_path_segment_id"] = int(segment_id)
            frow["exit_path_entry_signal_i"] = int(start_i)
            frow["exit_path_entry_i"] = int(entry_i)
            frow["exit_path_end_i"] = int(end_i)
            frow["exit_path_side"] = int(side)
            frow["exit_path_hold_bars"] = int(max(int(row_i) - int(entry_i), 0))
            frow["exit_price_move_label"] = int(label)
            frow["exit_price_move_reason"] = reason
            frow["exit_path_mfe_price_move"] = float(mfe)
            frow["exit_path_mae_price_move"] = float(mae)
            frow["exit_path_price_move"] = float(move)
            frow["exit_path_giveback"] = float(giveback)
            frow["exit_path_bars_to_segment_end"] = int(bars_to_segment_end)
            frame_rows.append(frow)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if max_samples > 0 and len(rows) >= int(max_samples):
                break
        if segment_rows > 0:
            used_segments += 1
        else:
            skipped_segments += 1
        if max_samples > 0 and len(rows) >= int(max_samples):
            break

    if not rows:
        raise RuntimeError("empty price-move Exit Head dataset")
    x = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(labels, dtype=np.int64)
    frame_exit = pd.DataFrame(frame_rows).reset_index(drop=True)
    return x, y, frame_exit, {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "positive_count": int(positive_count),
        "negative_count": int(len(y) - positive_count),
        "continued_exit_reasons": reason_counts,
        "price_barrier_counts_before_exit_head": barrier_counts,
        "used_segments": int(used_segments),
        "skipped_segments": int(skipped_segments),
        "risk_template": {
            "notional": notional,
            "leverage": leverage,
            "take_profit_price_move": take_profit,
            "stop_loss_price_move": stop_loss,
        },
        "label_mode": "price_move_terminal_giveback_every_in_position_bar",
        "terminal_window": int(tw),
        "adverse_price_move": float(adverse_price_move),
        "min_mfe_for_giveback": float(min_mfe_for_giveback),
        "giveback_min": float(giveback_min),
    }


def _ce_tabm(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, 2),
        target[:, None].expand(-1, int(parent.CFG.k)).reshape(-1),
        reduction="none",
    ).reshape(-1, int(parent.CFG.k)).mean(dim=1)


def _fit_exit_head_only(
    baseline_payload: dict[str, Any],
    x_exit: pd.DataFrame,
    y_exit: np.ndarray,
    exit_route_frame: pd.DataFrame,
    *,
    expert_idx: int,
    seed: int,
    epochs: int,
    device: torch.device,
    model_path: Path,
) -> dict[str, Any]:
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    if list(x_exit.columns) != list(baseline_payload["scaler"]["columns"]):
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} feature column contract mismatch for exit-only retrain")
    x_np = parent._standardize_apply(x_exit, baseline_payload["scaler"])
    y_np = np.asarray(y_exit, dtype=np.int64)
    classes = sorted(np.unique(y_np).astype(int).tolist())
    if classes != [0, 1]:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} exit labels need both classes [0,1], got {classes}")
    route_w = parent._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    weights = compute_sample_weight(class_weight="balanced", y=y_np).astype(np.float32) * route_w
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid exit-only sample weights")

    n = len(y_np)
    split = max(int(n * 0.85), min(n - 1, 256))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    ds = TensorDataset(torch.from_numpy(x_np[train_idx]), torch.from_numpy(y_np[train_idx]), torch.from_numpy(weights[train_idx]))
    dl = DataLoader(ds, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
    model = parent.ThreeHeadTabM(int(baseline_payload["n_features"]), cfg=parent.CFG).to(device)
    model.load_state_dict(baseline_payload["state_dict"])
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.exit_head.parameters():
        param.requires_grad_(True)
    opt = torch.optim.AdamW(model.exit_head.parameters(), lr=float(parent.CFG.lr), weight_decay=float(parent.CFG.weight_decay))
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        for xb, yb, wb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            out = model(xb)
            loss = (_ce_tabm(out["exit"], yb) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.exit_head.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_np[val_idx]).to(device)
            vy = torch.from_numpy(y_np[val_idx]).to(device)
            vw = torch.from_numpy(weights[val_idx]).to(device)
            vo = model(vx)
            val_loss = float(((_ce_tabm(vo["exit"], vy) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
        if val_loss + 1.0e-6 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(parent.CFG.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        **baseline_payload,
        "model_id": MODEL_ID,
        "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "best_exit_validation_loss": float(best_loss),
        "exit_epochs_ran": int(last_epoch),
        "frozen_contract": "encoder_direction_quality_frozen_exit_head_only_retrained",
        "exit_feature_semantics": "position price_move fields are raw price moves; PnL still uses price_move * notional",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, model_path)
    return payload


def _duration_days(frame: pd.DataFrame) -> float:
    return max((pd.to_datetime(frame["timestamp"].iloc[-1]) - pd.to_datetime(frame["timestamp"].iloc[0])).total_seconds() / 86400.0, 1.0e-9)


@torch.no_grad()
def _metrics_shared_exit_price_move_sltp(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple[parent.ThreeHeadTabM, dict[str, Any]]],
    *,
    threshold: float,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
) -> dict[str, Any]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    reasons: dict[str, int] = {}
    route = hard._route_id(frame)
    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = _price_move(arrays, int(i), side=pos, entry_price=float(entry_price), slip_eff=slip_eff)
            unreal = move * notional
            mfe = max(mfe, move)
            mae = min(mae, move)
            eq = cash * (1.0 + unreal)
        else:
            move = 0.0
            unreal = 0.0
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)
        if pos != 0:
            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            else:
                xrow = base_x.iloc[[i]].copy().reset_index(drop=True)
                hold = max(int(i) - int(entry_i), 0)
                giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                vals = {
                    "pos_side": float(pos),
                    "pos_hold_bars": float(hold),
                    "pos_unrealized": float(move),
                    "pos_mfe": float(mfe),
                    "pos_mae": float(mae),
                    "pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
                    "pos_dist_to_tp": float(take_profit - move),
                    "pos_dist_to_sl": float(move + abs(stop_loss)),
                    "pos_notional": float(notional),
                    "pos_leverage": float(leverage),
                    "pos_exposure": float(notional * leverage),
                    "pos_tp": float(take_profit),
                    "pos_sl": float(stop_loss),
                }
                for col, val in vals.items():
                    xrow[col] = val
                expert = hard.EXPERT_NAMES[int(route[i])]
                model, scaler = loaded_models[expert]
                prob = float(parent._predict_loaded_exit(model, scaler, xrow, device=device)[0, 1])
                if prob >= float(threshold):
                    reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                reasons[reason] = reasons.get(reason, 0) + 1
                pos = 0
                continue
        if pos != 0:
            continue
        if not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        pos = side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        leverage = float(row.get("leverage", 1.0) or 1.0)
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
        cash -= cash * entry_fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        mfe = 0.0
        mae = 0.0
    if pos != 0:
        exit_px = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
    n_entries = max(long_entries + short_entries, 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "trades_per_day": float(trades / _duration_days(frame)),
        "avg_notional": float(notional_sum / n_entries),
        "avg_leverage": float(leverage_sum / n_entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
    }


def _load_payloads(payloads: dict[str, dict[str, Any]], *, device: torch.device) -> dict[str, tuple[parent.ThreeHeadTabM, dict[str, Any]]]:
    return parent._load_payloads(payloads, device=device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-bundle", type=Path, default=BASELINE_BUNDLE)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-exit-samples", type=int, default=0)
    ap.add_argument("--quality-threshold", type=float, default=0.70)
    ap.add_argument("--exit-threshold", type=float, default=0.70)
    ap.add_argument("--exit-terminal-window", type=int, default=3)
    ap.add_argument("--exit-adverse-price-move", type=float, default=-0.010)
    ap.add_argument("--exit-min-mfe-for-giveback", type=float, default=0.006)
    ap.add_argument("--exit-giveback-min", type=float, default=0.65)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260622)
    ap.add_argument("--out-suffix", default="e8_full_exit_q070")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    device = parent._device(str(args.device))
    out_dir = OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}" if str(args.out_suffix).strip() else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("stage=load_baseline", flush=True)
    bundle = torch.load(Path(args.baseline_bundle), map_location=device, weights_only=False)
    baseline_models: dict[str, dict[str, Any]] = bundle["models"]
    base_cols = list(bundle["base_cols"])

    print("stage=prepare_frames", flush=True)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=omega4.LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()

    print("stage=build_price_move_exit_labels", flush=True)
    x_exit_raw, y_exit, frame_exit, exit_diag = _build_exit_dataset_price_move_terminal_giveback(
        frames["train_df"],
        frames["s_train_label"],
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_samples=int(args.max_exit_samples),
        terminal_window=int(args.exit_terminal_window),
        adverse_price_move=float(args.exit_adverse_price_move),
        min_mfe_for_giveback=float(args.exit_min_mfe_for_giveback),
        giveback_min=float(args.exit_giveback_min),
    )
    x_exit = parent._exit_input_from_position_rows(x_exit_raw, base_cols)

    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        print(f"stage=train_exit_head_only expert={expert}", flush=True)
        payload = _fit_exit_head_only(
            baseline_models[expert],
            x_exit,
            y_exit,
            frame_exit,
            expert_idx=idx,
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=device,
            model_path=out_dir / "models" / f"{expert}_3head_tabm_exit_price_move.pt",
        )
        models[expert] = payload
        summaries[expert] = {
            "model": str(out_dir / "models" / f"{expert}_3head_tabm_exit_price_move.pt"),
            "exit_epochs_ran": int(payload["exit_epochs_ran"]),
            "best_exit_validation_loss": float(payload["best_exit_validation_loss"]),
        }

    def predict_decisions(frame: pd.DataFrame, *, oof: bool, use_models: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = parent._base_input(frame, base_cols)
        preds = {expert: parent._predict_payload(use_models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = parent._routed(preds, route, "direction", 3)
        quality = parent._routed(preds, route, "quality", 3)
        prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
        src = parent._prediction_output(frame, direction, quality, threshold=float(args.quality_threshold), prefix=prefix)
        return x, parent._to_decisions(src, oof=oof)

    print("stage=predict_decisions", flush=True)
    x_val, val_dec = predict_decisions(frames["val_raw"], oof=True, use_models=models)
    x_oos, oos_dec = predict_decisions(frames["oos_raw"], oof=False, use_models=models)
    val_dec.to_csv(out_dir / "validation_decisions_q070.csv", index=False)
    oos_dec.to_csv(out_dir / "oos_decisions_q070.csv", index=False)

    print("stage=evaluate_original_exit_price_move_contract", flush=True)
    original_loaded = _load_payloads(baseline_models, device=device)
    orig_val = _metrics_shared_exit_price_move_sltp(
        frames["val_raw"],
        x_val,
        val_dec,
        original_loaded,
        threshold=float(args.exit_threshold),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
    )
    orig_oos = _metrics_shared_exit_price_move_sltp(
        frames["oos_raw"],
        x_oos,
        oos_dec,
        original_loaded,
        threshold=float(args.exit_threshold),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
    )

    print("stage=evaluate_retrained_exit_price_move_contract", flush=True)
    loaded = _load_payloads(models, device=device)
    val_m = _metrics_shared_exit_price_move_sltp(
        frames["val_raw"],
        x_val,
        val_dec,
        loaded,
        threshold=float(args.exit_threshold),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
    )
    oos_m = _metrics_shared_exit_price_move_sltp(
        frames["oos_raw"],
        x_oos,
        oos_dec,
        loaded,
        threshold=float(args.exit_threshold),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
    )

    report = {
        "model_id": MODEL_ID,
        "baseline_bundle": str(args.baseline_bundle),
        "design": "Omega4.1 baseline encoder/direction/quality are frozen. Only exit_head is retrained with price-move SLTP position features.",
        "contract": {
            "sltp_hit": "raw price_move is compared directly to take_profit/stop_loss",
            "pnl_sizing": "realized price_move * notional",
            "notional": "kept only for sizing and position feature context",
            "max_hold_bars": int(omega.BASE_TEMPLATE["max_hold"]),
            "cooldown_bars": int(omega.BASE_TEMPLATE["cooldown"]),
            "quality_threshold": float(args.quality_threshold),
            "exit_threshold": float(args.exit_threshold),
        },
        "exit_label": exit_diag,
        "summaries": summaries,
        "reference_existing_baseline": {
            "exit070_validation": {"pnl": 3.2756516763214893, "mdd": -7.817488798061978, "trades": 149, "wr": 0.6711409395973155},
            "exit070_oos": {"pnl": 7.513325496582635, "mdd": -5.61401353413885, "trades": 100, "wr": 0.63},
            "note": "Existing reference uses the previous account-threshold SLTP replay contract.",
        },
        "results": {
            "original_exit_head_price_move_sltp": {"validation": orig_val, "oos": orig_oos},
            "retrained_exit_head_price_move_sltp": {"validation": val_m, "oos": oos_m},
        },
        "artifacts": {
            "out_dir": str(out_dir),
            "report": str(out_dir / "report.json"),
            "bundle": str(out_dir / "true_3head_tabm_bundle.pt"),
        },
    }
    torch.save(
        {"models": models, "base_cols": base_cols, "pos_cols": parent.POS_COLS, "config": parent.CFG.__dict__, "model_id": MODEL_ID},
        out_dir / "true_3head_tabm_bundle.pt",
    )
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"], "exit_label": exit_diag}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
