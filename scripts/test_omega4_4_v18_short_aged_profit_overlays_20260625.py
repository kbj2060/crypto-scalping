#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as risk  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402


MODEL_ID = "omega4_4_v18_short_aged_profit_overlay_tests_20260625"
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
    lock_floor: float = 0.0
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


def _duration_days(frame: pd.DataFrame) -> float:
    return max((pd.to_datetime(frame["timestamp"].iloc[-1]) - pd.to_datetime(frame["timestamp"].iloc[0])).total_seconds() / 86400.0, 1.0e-9)


def _load_frames(report: dict[str, Any]) -> dict[str, pd.DataFrame]:
    omega.TRAIN_CSV = Path(report["risk_model"]["train_csv"])
    omega.EVAL_CSV = Path(report["risk_model"]["eval_csv"])
    return omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=Path(report["risk_model"]["direction_label_dir"]),
        quality_mode=str(report["risk_model"]["quality_mode"]),
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )


def _fill_exit(arrays: dict[str, np.ndarray], idx: int, side: int, fee_eff: float, slip_eff: float, *, forced: bool) -> tuple[float, float]:
    if forced:
        return omega._fill_price(arrays, idx, side, slip_eff, entry=False), fee_eff
    filled, px, fee_paid, _ = omega._try_execution(arrays, idx, side, entry=False, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return 0.0, 0.0
    return float(px), float(fee_paid)


def _should_target_side(spec: OverlaySpec, side: int) -> bool:
    return int(spec.side) == 0 or int(spec.side) == int(side)


def _trade_path_moves(
    arrays: dict[str, np.ndarray],
    *,
    side: int,
    entry_price: float,
    slip_eff: float,
    start_i: int,
    end_i: int,
) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    for i in range(int(start_i), int(end_i) + 1):
        hold = max(int(i) - int(start_i), 0)
        move = price_exit._price_move(arrays, int(i), side=int(side), entry_price=float(entry_price), slip_eff=float(slip_eff))
        out.append((int(i), int(hold), float(move)))
    return out


def _simulate_split(
    frame: pd.DataFrame,
    baseline_ledger: pd.DataFrame,
    spec: OverlaySpec,
    *,
    fee_eff: float,
    slip_eff: float,
    log_risk_params: dict[str, float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    entries = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    margin_sum = 0.0
    log_growth_sum = 0.0
    tail_excess_sum = 0.0
    liquidation_excess_sum = 0.0
    log_risk_utility_sum = 0.0
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    overlay_hits = 0

    tail_budget = float(log_risk_params.get("tail_budget", 0.02))
    tail_penalty = float(log_risk_params.get("tail_penalty", 0.5))
    liquidation_buffer = float(log_risk_params.get("liquidation_buffer", 0.12))
    liquidation_penalty = float(log_risk_params.get("liquidation_penalty", 0.25))

    for _, base_row in baseline_ledger.sort_values("entry_i").iterrows():
        side = int(base_row["side"])
        entry_signal_i = int(base_row["entry_signal_i"])
        entry_i = int(base_row["entry_i"])
        baseline_exit_i = int(base_row["exit_i"])
        notional = float(base_row["notional"])
        leverage = float(base_row["leverage"])
        margin_fraction = float(base_row["margin_fraction"])
        if notional <= 0.0:
            continue
        filled, entry_price, entry_fee, _ = omega._try_execution(
            arrays,
            entry_signal_i,
            side,
            entry=True,
            fee_base=fee_eff,
            slip_base=slip_eff,
        )
        if not filled:
            raise RuntimeError(f"entry fill mismatch at {entry_signal_i}")

        entries += 1
        long_entries += int(side > 0)
        short_entries += int(side < 0)
        notional_sum += notional
        leverage_sum += leverage
        margin_sum += margin_fraction
        entry_equity = cash
        cash -= cash * float(entry_fee) * notional
        remaining_notional = notional
        triggered = False
        partial_done = False
        mfe = 0.0
        mae = 0.0
        exit_i = baseline_exit_i
        exit_reason = str(base_row["reason"])
        forced = exit_reason == "forced_end"
        overlay_reason = ""
        overlay_exit_price = 0.0
        overlay_exit_fee = 0.0

        for i, hold, move in _trade_path_moves(
            arrays,
            side=side,
            entry_price=entry_price,
            slip_eff=slip_eff,
            start_i=entry_i,
            end_i=baseline_exit_i,
        ):
            mfe = max(mfe, move)
            mae = min(mae, move)
            eq = cash * (1.0 + move * remaining_notional)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)
            if not _should_target_side(spec, side):
                continue
            if int(spec.cap_bars) <= 0:
                continue
            if hold >= int(spec.cap_bars) and move >= float(spec.min_unreal):
                triggered = True
                if spec.mode == "hard_exit":
                    px, fee_paid = _fill_exit(arrays, i, side, fee_eff, slip_eff, forced=False)
                    if px > 0.0:
                        exit_i = i
                        overlay_reason = "short_aged_profit_exit" if side < 0 else "long_aged_profit_exit"
                        overlay_exit_price = px
                        overlay_exit_fee = fee_paid
                        break
                elif spec.mode == "floor_lock":
                    if move <= float(spec.lock_floor):
                        px, fee_paid = _fill_exit(arrays, i, side, fee_eff, slip_eff, forced=False)
                        if px > 0.0:
                            exit_i = i
                            overlay_reason = "short_profit_floor_exit" if side < 0 else "long_profit_floor_exit"
                            overlay_exit_price = px
                            overlay_exit_fee = fee_paid
                            break
                elif spec.mode == "trailing_lock":
                    floor = max(float(spec.lock_floor), float(mfe) * (1.0 - float(spec.giveback_frac)))
                    if move <= floor:
                        px, fee_paid = _fill_exit(arrays, i, side, fee_eff, slip_eff, forced=False)
                        if px > 0.0:
                            exit_i = i
                            overlay_reason = "short_trailing_profit_exit" if side < 0 else "long_trailing_profit_exit"
                            overlay_exit_price = px
                            overlay_exit_fee = fee_paid
                            break
                elif spec.mode == "partial_deleverage" and not partial_done and 0.0 < float(spec.partial_fraction) < 1.0:
                    px, fee_paid = _fill_exit(arrays, i, side, fee_eff, slip_eff, forced=False)
                    if px > 0.0:
                        raw_partial = (px - entry_price) / max(entry_price, 1.0e-12) if side > 0 else (entry_price - px) / max(entry_price, 1.0e-12)
                        close_notional = remaining_notional * float(spec.partial_fraction)
                        before = cash
                        cash = cash * (1.0 + raw_partial * close_notional)
                        cash -= before * fee_paid * close_notional
                        remaining_notional -= close_notional
                        partial_done = True
                        overlay_hits += 1
                        reasons["short_partial_deleverage" if side < 0 else "long_partial_deleverage"] = reasons.get("short_partial_deleverage" if side < 0 else "long_partial_deleverage", 0) + 1

        if overlay_reason:
            exit_reason = overlay_reason
            forced = False
            overlay_hits += 1
        if overlay_exit_price > 0.0:
            exit_price = overlay_exit_price
            exit_fee = overlay_exit_fee
        else:
            exit_price, exit_fee = _fill_exit(arrays, baseline_exit_i, side, fee_eff, slip_eff, forced=forced)
            if exit_price <= 0.0:
                raise RuntimeError(f"exit fill mismatch at {baseline_exit_i}")
        raw_exit = (exit_price - entry_price) / max(entry_price, 1.0e-12) if side > 0 else (entry_price - exit_price) / max(entry_price, 1.0e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * remaining_notional)
        cash -= before * exit_fee * remaining_notional
        trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
        account_return = cash / max(before, 1.0e-12) - 1.0
        wins += int(cash > entry_equity)
        reasons[exit_reason] = reasons.get(exit_reason, 0) + 1
        log_growth = float(np.log1p(max(account_return, -0.999999)))
        tail_excess = max(-float(mae) * float(notional) - tail_budget, 0.0)
        liquidation_excess = max(-float(mae) * float(leverage) - liquidation_buffer, 0.0)
        log_risk_utility = log_growth - tail_penalty * tail_excess - liquidation_penalty * liquidation_excess
        log_growth_sum += log_growth
        tail_excess_sum += tail_excess
        liquidation_excess_sum += liquidation_excess
        log_risk_utility_sum += log_risk_utility
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1.0e-12) - 1.0)
        rows.append(
            {
                "entry_signal_i": entry_signal_i,
                "entry_i": entry_i,
                "exit_i": int(exit_i),
                "entry_timestamp": str(frame["timestamp"].iloc[entry_signal_i]),
                "exit_timestamp": str(frame["timestamp"].iloc[int(exit_i)]),
                "side": side,
                "reason": exit_reason,
                "win": int(cash > entry_equity),
                "raw_exit_price_move": float(raw_exit),
                "mfe_price_move": float(mfe),
                "mae_price_move": float(mae),
                "trade_return": float(trade_return),
                "net_per_notional": float(trade_return / max(notional, 1.0e-12)),
                "notional": float(notional),
                "remaining_notional_at_final_exit": float(remaining_notional),
                "margin_fraction": float(margin_fraction),
                "leverage": float(leverage),
                "take_profit": float(base_row.get("take_profit", 0.0)),
                "stop_loss": float(base_row.get("stop_loss", 0.0)),
                "overlay_triggered": bool(triggered),
                "partial_done": bool(partial_done),
            }
        )

    n_entries = max(entries, 1)
    n_rows = max(len(rows), 1)
    return (
        {
            "pnl": float((cash - 1.0) * 100.0),
            "mdd": float(mdd * 100.0),
            "trades": int(len(rows)),
            "entries": int(entries),
            "wr": float(wins / n_rows),
            "trades_per_day": float(len(rows) / _duration_days(frame)),
            "avg_notional": float(notional_sum / n_entries),
            "avg_margin_fraction": float(margin_sum / n_entries),
            "avg_leverage": float(leverage_sum / n_entries),
            "log_growth_sum": float(log_growth_sum),
            "tail_excess_sum": float(tail_excess_sum),
            "liquidation_excess_sum": float(liquidation_excess_sum),
            "log_risk_utility": float(log_risk_utility_sum),
            "long_entries": int(long_entries),
            "short_entries": int(short_entries),
            "overlay_hits": int(overlay_hits),
            "exit_reasons": {str(k): int(v) for k, v in reasons.items()},
        },
        pd.DataFrame(rows),
    )


def _specs() -> list[OverlaySpec]:
    specs = [OverlaySpec("baseline_path_replay", "none", -1, 0, 0.0)]
    cap_bars_grid = (144, 288, 576, 864, 1152, 1760)
    min_unreal_grid = (0.015, 0.025, 0.035, 0.050)
    for cap_bars in cap_bars_grid:
        for min_unreal in min_unreal_grid:
            specs.append(OverlaySpec(f"short_hard_cap{cap_bars}_u{min_unreal:.3f}", "hard_exit", -1, cap_bars, min_unreal))
            for floor in (0.000, 0.010, 0.020):
                specs.append(OverlaySpec(f"short_floor_cap{cap_bars}_u{min_unreal:.3f}_f{floor:.3f}", "floor_lock", -1, cap_bars, min_unreal, lock_floor=floor))
            for giveback in (0.25, 0.40, 0.60):
                specs.append(OverlaySpec(f"short_trail_cap{cap_bars}_u{min_unreal:.3f}_gb{giveback:.2f}", "trailing_lock", -1, cap_bars, min_unreal, lock_floor=0.0, giveback_frac=giveback))
            for frac in (0.25, 0.50, 0.75):
                specs.append(OverlaySpec(f"short_partial_cap{cap_bars}_u{min_unreal:.3f}_p{frac:.2f}", "partial_deleverage", -1, cap_bars, min_unreal, partial_fraction=frac))
    # A small symmetric smoke set checks whether the concept is short-specific.
    for side_name, side in (("long", 1), ("both", 0)):
        for cap_bars in (288, 576, 1152):
            for min_unreal in (0.025, 0.035):
                specs.append(OverlaySpec(f"{side_name}_hard_cap{cap_bars}_u{min_unreal:.3f}", "hard_exit", side, cap_bars, min_unreal))
    return specs


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    frames = _load_frames(report)
    fee, slip = omega._load_fee_slip()
    fee_eff = float(fee) * 3.0
    slip_eff = float(slip) * 3.0
    log_risk_params = {k: float(v) for k, v in report["risk_model"]["log_risk_params"].items()}

    split_payload = {
        "validation": (frames["val_raw"], pd.read_csv(BASELINE_DIR / "validation_selected_risk_replayed_trade_ledger.csv")),
        "oos": (frames["oos_raw"], pd.read_csv(BASELINE_DIR / "oos_selected_risk_replayed_trade_ledger.csv")),
    }

    rows: list[dict[str, Any]] = []
    ledgers: dict[tuple[str, str], pd.DataFrame] = {}
    for spec in _specs():
        row: dict[str, Any] = {
            "variant": spec.variant,
            "mode": spec.mode,
            "side": spec.side,
            "cap_bars": spec.cap_bars,
            "min_unreal": spec.min_unreal,
            "lock_floor": spec.lock_floor,
            "giveback_frac": spec.giveback_frac,
            "partial_fraction": spec.partial_fraction,
        }
        for split, (frame, ledger) in split_payload.items():
            metrics, out_ledger = _simulate_split(
                frame,
                ledger,
                spec,
                fee_eff=fee_eff,
                slip_eff=slip_eff,
                log_risk_params=log_risk_params,
            )
            for key, value in metrics.items():
                if key == "exit_reasons":
                    row[f"{split}_exit_reasons"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                else:
                    row[f"{split}_{key}"] = value
            if spec.variant == "baseline_path_replay" or int(metrics["overlay_hits"]) > 0:
                ledgers[(split, spec.variant)] = out_ledger
        rows.append(row)

    ranking = pd.DataFrame(rows)
    baseline = ranking.loc[ranking["variant"].eq("baseline_path_replay")].iloc[0]
    for split in ("validation", "oos"):
        ranking[f"{split}_delta_pnl"] = ranking[f"{split}_pnl"] - float(baseline[f"{split}_pnl"])
        ranking[f"{split}_delta_mdd"] = ranking[f"{split}_mdd"] - float(baseline[f"{split}_mdd"])
        ranking[f"{split}_delta_log_risk"] = ranking[f"{split}_log_risk_utility"] - float(baseline[f"{split}_log_risk_utility"])
    ranking["validation_selection_ok"] = (
        (ranking["validation_overlay_hits"] > 0)
        & (ranking["validation_mdd"] >= float(baseline["validation_mdd"]) - 1.0)
        & (ranking["validation_pnl"] >= float(baseline["validation_pnl"]) - 1.0)
    )
    ranking["validation_score"] = (
        ranking["validation_log_risk_utility"]
        + 0.0025 * ranking["validation_pnl"]
        + 0.0010 * ranking["validation_mdd"]
        + 0.0005 * ranking["validation_overlay_hits"]
    )
    ranking = ranking.sort_values(
        ["validation_selection_ok", "validation_score", "validation_log_risk_utility", "validation_pnl"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "overlay_grid_results.csv", index=False)

    selected_pool = ranking[ranking["validation_selection_ok"]].copy()
    selected = selected_pool.iloc[0].to_dict() if len(selected_pool) else ranking.iloc[0].to_dict()
    top_variants = ["baseline_path_replay"] + [str(v) for v in ranking["variant"].head(12).tolist()]
    for (split, variant), ledger in ledgers.items():
        if variant in set(top_variants):
            ledger.to_csv(OUT_DIR / f"{split}_{variant}_ledger.csv", index=False)

    report_out = {
        "model_id": MODEL_ID,
        "source_model": "omega4_4_v18_baseline_20260624",
        "source_report": str(REPORT_PATH),
        "method": "Path-level replay overlay scan on Omega4.4 v18 full-replay ledger. Does not retrain parent or risk sidecar; does not insert extra re-entry opportunities after earlier exits.",
        "baseline_path_replay": baseline.to_dict(),
        "selection_policy": "validation-only overlay scan: overlay_hits > 0, validation_mdd no more than 1pp worse than baseline path replay, validation_pnl no more than 1pp below baseline; sorted by validation score/log-risk/PnL. OOS is readout only.",
        "selected_by_validation": selected,
        "top_by_validation": ranking.head(20).to_dict(orient="records"),
        "top_by_oos_readout": ranking.sort_values(["oos_pnl", "validation_pnl"], ascending=[False, False]).head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "grid": str(OUT_DIR / "overlay_grid_results.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report_out, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(OUT_DIR / "report.json"),
                "grid": str(OUT_DIR / "overlay_grid_results.csv"),
                "baseline": {
                    "validation_pnl": float(baseline["validation_pnl"]),
                    "validation_mdd": float(baseline["validation_mdd"]),
                    "oos_pnl": float(baseline["oos_pnl"]),
                    "oos_mdd": float(baseline["oos_mdd"]),
                },
                "selected_by_validation": {
                    "variant": selected["variant"],
                    "validation_pnl": selected["validation_pnl"],
                    "validation_mdd": selected["validation_mdd"],
                    "validation_overlay_hits": selected["validation_overlay_hits"],
                    "oos_pnl": selected["oos_pnl"],
                    "oos_mdd": selected["oos_mdd"],
                    "oos_overlay_hits": selected["oos_overlay_hits"],
                },
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
