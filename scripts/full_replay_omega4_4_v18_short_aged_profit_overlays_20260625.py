#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as risk  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega4_4_v18_short_aged_profit_overlay_full_replay_20260625"
BASELINE_DIR = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624"
)
REPORT_PATH = BASELINE_DIR / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass(frozen=True)
class OverlaySpec:
    variant: str
    mode: str
    side: int
    cap_bars: int
    min_unreal: float
    giveback_frac: float = 0.0
    partial_fraction: float = 0.0


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


def _target_side(spec: OverlaySpec, side: int) -> bool:
    return int(spec.side) == 0 or int(spec.side) == int(side)


def _specs() -> list[OverlaySpec]:
    return [
        OverlaySpec("baseline_full_replay", "none", -1, 0, 0.0),
        OverlaySpec("short_trail_cap1152_u0.035_gb0.25", "trailing_lock", -1, 1152, 0.035, giveback_frac=0.25),
        OverlaySpec("short_partial_cap1152_u0.035_p0.25", "partial_deleverage", -1, 1152, 0.035, partial_fraction=0.25),
        OverlaySpec("short_partial_cap1152_u0.035_p0.50", "partial_deleverage", -1, 1152, 0.035, partial_fraction=0.50),
        OverlaySpec("short_trail_cap1152_u0.050_gb0.25", "trailing_lock", -1, 1152, 0.050, giveback_frac=0.25),
        OverlaySpec("short_partial_cap864_u0.050_p0.50", "partial_deleverage", -1, 864, 0.050, partial_fraction=0.50),
        OverlaySpec("short_hard_cap1760_u0.035", "hard_exit", -1, 1760, 0.035),
    ]


def _prepare_payload(report: dict[str, Any], device: torch.device) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]], dict[str, Any]]:
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    omega.TRAIN_CSV = Path(report["risk_model"]["train_csv"])
    omega.EVAL_CSV = Path(report["risk_model"]["eval_csv"])
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=Path(report["risk_model"]["direction_label_dir"]),
        quality_mode=str(report["risk_model"]["quality_mode"]),
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    bundle = torch.load(Path(report["baseline_bundle"]), map_location=device, weights_only=False)
    models: dict[str, dict[str, Any]] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    loaded = parent._load_payloads(models, device=device)
    sidecar = pickle.load((BASELINE_DIR / "risk_sidecar.pkl").open("rb"))
    selected_mapping = dict(sidecar["selected_mapping"])
    margin_cfg = {k: float(selected_mapping[k]) for k in risk.MARGIN_CFG_KEYS}
    leverage_cfg = {k: float(selected_mapping[k]) for k in risk.LEVERAGE_CFG_KEYS if k in selected_mapping}

    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for split, frame_key, oof in (("validation", "val_raw", True), ("oos", "oos_raw", False)):
        frame = frames[frame_key]
        x, src, dec_base = risk._predict_decisions(
            frame,
            oof=oof,
            models=models,
            base_cols=base_cols,
            quality_threshold=float(report["contract"]["quality_threshold"]),
            device=device,
        )
        dec, _diag = atr_eval._apply_atr_safety_sltp(
            dec_base,
            frame,
            atr_window=int(report["contract"]["atr_window"]),
            tp_mult=float(report["contract"]["take_profit_atr_multiple"]),
            sl_mult=float(report["contract"]["stop_loss_atr_multiple"]),
            min_tp=float(report["contract"]["floor_take_profit_price_move"]),
            min_sl=float(report["contract"]["floor_stop_loss_price_move"]),
            max_tp=float(report["contract"]["cap_take_profit_price_move"]),
            max_sl=float(report["contract"]["cap_stop_loss_price_move"]),
        )
        features = risk._risk_feature_frame(
            frame,
            src,
            dec,
            base_cols,
            atr_pct=atr_eval._atr_pct(frame, int(report["contract"]["atr_window"])),
            feature_mode=str(sidecar["risk_feature_mode"]),
        )
        x_risk, _ = risk._feature_matrix(features, list(sidecar["feature_columns"]))
        side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
        if bool(sidecar["side_split_model"]):
            score = risk._predict_side_split_models(sidecar["model"], x_risk, side)
        else:
            score = np.asarray(sidecar["model"].predict(x_risk), dtype=np.float64)
        margin = risk._risk_margins(
            dec,
            score,
            train_q50=float(sidecar["train_score_q50"]),
            train_iqr=float(sidecar["train_score_iqr"]),
            **margin_cfg,
        )
        leverage = risk._risk_leverage(
            dec,
            score,
            train_q50=float(sidecar["train_score_q50"]),
            train_iqr=float(sidecar["train_score_iqr"]),
            **leverage_cfg,
        )
        payload[split] = (frame, x, dec, margin, leverage)
    return payload, {"loaded": loaded}


@torch.no_grad()
def _replay_overlay(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple[parent.ThreeHeadTabM, dict[str, Any]]],
    margin: np.ndarray,
    leverage_arr: np.ndarray,
    spec: OverlaySpec,
    *,
    report: dict[str, Any],
    fee: float,
    slip: float,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * 3.0
    slip_eff = float(slip) * 3.0
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    entry_signal_i = 0
    notional = 0.0
    original_notional = 0.0
    leverage = 1.0
    margin_fraction = 0.0
    exit_input_notional = 0.0
    exit_input_leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    partial_done = False
    overlay_hits = 0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    margin_sum = 0.0
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    route = hard._route_id(frame)
    base_np, exit_runtime, pos_idx = risk._prepare_exit_runtime(base_x, loaded_models)
    exit_threshold = float(report["contract"]["exit_threshold"])

    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = price_exit._price_move(arrays, int(i), side=pos, entry_price=float(entry_price), slip_eff=slip_eff)
            unreal = move * notional
            mfe = max(mfe, move)
            mae = min(mae, move)
            eq = cash * (1.0 + unreal)
        else:
            move = 0.0
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)

        if pos != 0:
            reason = ""
            exit_prob = 0.0
            hold = max(int(i) - int(entry_i), 0)
            if _target_side(spec, pos) and int(spec.cap_bars) > 0 and hold >= int(spec.cap_bars) and move >= float(spec.min_unreal):
                if spec.mode == "hard_exit":
                    reason = "short_aged_profit_exit" if pos < 0 else "long_aged_profit_exit"
                    overlay_hits += 1
                elif spec.mode == "trailing_lock":
                    floor = float(mfe) * (1.0 - float(spec.giveback_frac))
                    if move <= floor:
                        reason = "short_trailing_profit_exit" if pos < 0 else "long_trailing_profit_exit"
                        overlay_hits += 1
                elif spec.mode == "partial_deleverage" and not partial_done and 0.0 < float(spec.partial_fraction) < 1.0:
                    filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                    if filled:
                        raw_partial = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                        close_notional = notional * float(spec.partial_fraction)
                        before = cash
                        cash = cash * (1.0 + raw_partial * close_notional)
                        cash -= before * exit_fee * close_notional
                        notional -= close_notional
                        exit_input_notional = notional
                        partial_done = True
                        overlay_hits += 1
                        reasons["short_partial_deleverage" if pos < 0 else "long_partial_deleverage"] = reasons.get("short_partial_deleverage" if pos < 0 else "long_partial_deleverage", 0) + 1
            if not reason:
                if take_profit > 0.0 and move >= take_profit:
                    reason = "take_profit"
                elif stop_loss > 0.0 and move <= -abs(stop_loss):
                    reason = "stop_loss"
                else:
                    giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                    expert = hard.EXPERT_NAMES[int(route[i])]
                    prob = risk._predict_exit_prob_one(
                        base_np,
                        exit_runtime,
                        pos_idx,
                        row_i=int(i),
                        expert=expert,
                        pos_values=[
                            float(pos),
                            float(hold),
                            float(move),
                            float(mfe),
                            float(mae),
                            float(np.clip(giveback, 0.0, 10.0)),
                            float(take_profit - move),
                            float(move + abs(stop_loss)),
                            float(exit_input_notional),
                            float(exit_input_leverage),
                            float(exit_input_notional * exit_input_leverage),
                            float(take_profit),
                            float(stop_loss),
                        ],
                        device=device,
                    )
                    exit_prob = float(prob)
                    if prob >= exit_threshold:
                        reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
                trades += 1
                win = int(cash > entry_equity)
                wins += win
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(
                    {
                        "entry_signal_i": int(entry_signal_i),
                        "entry_i": int(entry_i),
                        "exit_i": int(i),
                        "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                        "exit_timestamp": str(frame["timestamp"].iloc[int(i)]),
                        "side": int(pos),
                        "reason": reason,
                        "win": int(win),
                        "raw_exit_price_move": float(raw_exit),
                        "mfe_price_move": float(mfe),
                        "mae_price_move": float(mae),
                        "trade_return": float(trade_return),
                        "net_per_notional": float(trade_return / max(original_notional, 1.0e-12)),
                        "notional": float(original_notional),
                        "margin_fraction": float(margin_fraction),
                        "leverage": float(leverage),
                        "exit_input_notional": float(exit_input_notional),
                        "exit_input_leverage": float(exit_input_leverage),
                        "exit_input_exposure": float(exit_input_notional * exit_input_leverage),
                        "exit_prob": float(exit_prob),
                        "take_profit": float(take_profit),
                        "stop_loss": float(stop_loss),
                        "partial_done": bool(partial_done),
                    }
                )
                pos = 0
                partial_done = False
                continue
        if pos != 0 or not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, fee_paid, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        row_leverage = float(leverage_arr[int(i)])
        row_margin = float(margin[int(i)])
        row_notional = row_margin * row_leverage
        if row_notional <= 0.0:
            continue
        pos = side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_signal_i = int(i)
        leverage = row_leverage
        margin_fraction = row_margin
        notional = row_notional
        original_notional = row_notional
        exit_input_notional = row_notional
        exit_input_leverage = row_leverage
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
        cash -= cash * float(fee_paid) * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += original_notional
        leverage_sum += leverage
        margin_sum += margin_fraction
        mfe = 0.0
        mae = 0.0
        partial_done = False

    if pos != 0:
        exit_px = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
        trades += 1
        win = int(cash > entry_equity)
        wins += win
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(
            {
                "entry_signal_i": int(entry_signal_i),
                "entry_i": int(entry_i),
                "exit_i": int(len(frame) - 1),
                "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                "exit_timestamp": str(frame["timestamp"].iloc[-1]),
                "side": int(pos),
                "reason": "forced_end",
                "win": int(win),
                "raw_exit_price_move": float(raw_exit),
                "mfe_price_move": float(mfe),
                "mae_price_move": float(mae),
                "trade_return": float(trade_return),
                "net_per_notional": float(trade_return / max(original_notional, 1.0e-12)),
                "notional": float(original_notional),
                "margin_fraction": float(margin_fraction),
                "leverage": float(leverage),
                "exit_input_notional": float(exit_input_notional),
                "exit_input_leverage": float(exit_input_leverage),
                "exit_input_exposure": float(exit_input_notional * exit_input_leverage),
                "exit_prob": 0.0,
                "take_profit": float(take_profit),
                "stop_loss": float(stop_loss),
                "partial_done": bool(partial_done),
            }
        )

    n_entries = max(long_entries + short_entries, 1)
    metrics = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "trades_per_day": float(trades / risk._duration_days(frame)),
        "avg_notional": float(notional_sum / n_entries),
        "avg_margin_fraction": float(margin_sum / n_entries),
        "avg_leverage": float(leverage_sum / n_entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "overlay_hits": int(overlay_hits),
        "exit_reasons": reasons,
    }
    ledger = pd.DataFrame(rows)
    log_metrics, ledger = risk._ledger_metrics_with_margins(
        frame,
        ledger,
        None,
        **{k: float(v) for k, v in report["risk_model"]["log_risk_params"].items()},
    )
    for key in ("log_growth_sum", "tail_excess_sum", "liquidation_excess_sum", "log_risk_utility"):
        metrics[key] = log_metrics[key]
    return metrics, ledger


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    device = parent._device("cuda")
    payload, extra = _prepare_payload(report, device)
    fee, slip = omega._load_fee_slip()
    rows: list[dict[str, Any]] = []
    for spec in _specs():
        row: dict[str, Any] = {
            "variant": spec.variant,
            "mode": spec.mode,
            "side": spec.side,
            "cap_bars": spec.cap_bars,
            "min_unreal": spec.min_unreal,
            "giveback_frac": spec.giveback_frac,
            "partial_fraction": spec.partial_fraction,
        }
        for split, (frame, base_x, dec, margin, leverage) in payload.items():
            metrics, ledger = _replay_overlay(
                frame,
                base_x,
                dec,
                extra["loaded"],
                margin,
                leverage,
                spec,
                report=report,
                fee=fee,
                slip=slip,
                device=device,
            )
            for key, value in metrics.items():
                if key == "exit_reasons":
                    row[f"{split}_exit_reasons"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                else:
                    row[f"{split}_{key}"] = value
            ledger.to_csv(OUT_DIR / f"{split}_{spec.variant}_ledger.csv", index=False)
        rows.append(row)
        print(json.dumps({"variant": spec.variant, "validation_pnl": row["validation_pnl"], "oos_pnl": row["oos_pnl"]}, ensure_ascii=False), flush=True)

    df = pd.DataFrame(rows)
    base = df.loc[df["variant"].eq("baseline_full_replay")].iloc[0]
    for split in ("validation", "oos"):
        df[f"{split}_delta_pnl"] = df[f"{split}_pnl"] - float(base[f"{split}_pnl"])
        df[f"{split}_delta_mdd"] = df[f"{split}_mdd"] - float(base[f"{split}_mdd"])
        df[f"{split}_delta_log_risk"] = df[f"{split}_log_risk_utility"] - float(base[f"{split}_log_risk_utility"])
    df = df.sort_values(["validation_delta_log_risk", "validation_delta_pnl"], ascending=[False, False]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "full_replay_overlay_results.csv", index=False)
    report_out = {
        "model_id": MODEL_ID,
        "source_model": "omega4_4_v18_baseline_20260624",
        "source_report": str(REPORT_PATH),
        "baseline": base.to_dict(),
        "results": df.to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "grid": str(OUT_DIR / "full_replay_overlay_results.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report_out, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "grid": str(OUT_DIR / "full_replay_overlay_results.csv")}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
