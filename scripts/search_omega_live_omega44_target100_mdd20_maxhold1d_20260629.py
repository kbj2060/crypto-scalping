#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
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

import full_replay_omega4_4_v18_short_aged_profit_overlays_20260625 as v18  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as risk  # noqa: E402


MODEL_ID = "omega_live_omega44_target100_mdd20_maxhold1d_lev5_20260629"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LEDGER_DIR = OUT_DIR / "ledgers"

MAX_HOLD_BARS = 288
LEVERAGE_CAP = 5.0
TARGET_PNL = 100.0
TARGET_MDD = -20.0


@dataclass(frozen=True)
class ContractSpec:
    variant: str
    fixed_notional: float
    leverage: float
    tp_price_move: float
    sl_price_move: float
    long_scale: float = 1.0
    short_scale: float = 1.0
    side_filter: int = 0
    exit_threshold: float = 2.0
    dd_governor: bool = False
    stage: str = "stage1_noexit"


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


def _tag(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _stage1_specs() -> list[ContractSpec]:
    specs: list[ContractSpec] = []
    notional_grid = (0.90, 1.10, 1.30, 1.50, 1.75, 2.00, 2.25, 2.50, 2.80, 3.10)
    tp_sl_grid = (
        (0.016, 0.008),
        (0.020, 0.010),
        (0.026, 0.014),
        (0.030, 0.014),
        (0.035, 0.018),
        (0.040, 0.020),
        (0.052, 0.028),
    )
    side_scales = (
        (1.00, 1.00),
        (0.85, 1.15),
        (0.75, 1.35),
        (0.65, 1.55),
    )
    for notional in notional_grid:
        for tp, sl in tp_sl_grid:
            for long_scale, short_scale in side_scales:
                for side_filter in (0, -1):
                    for dd_governor in (False, True):
                        specs.append(
                            ContractSpec(
                                variant=(
                                    f"n{_tag(notional)}_tp{_tag(tp)}_sl{_tag(sl)}"
                                    f"_l{_tag(long_scale)}_s{_tag(short_scale)}"
                                    f"{'_shortonly' if side_filter < 0 else ''}"
                                    f"{'_ddgov' if dd_governor else ''}"
                                ),
                                fixed_notional=float(notional),
                                leverage=LEVERAGE_CAP,
                                tp_price_move=float(tp),
                                sl_price_move=float(sl),
                                long_scale=float(long_scale),
                                short_scale=float(short_scale),
                                side_filter=int(side_filter),
                                exit_threshold=2.0,
                                dd_governor=bool(dd_governor),
                            )
                        )
    return specs


def _stage2_specs(stage1: pd.DataFrame) -> list[ContractSpec]:
    eligible = stage1[
        (stage1["validation_mdd"] >= TARGET_MDD)
        & (stage1["oos_mdd"] >= TARGET_MDD)
        & (stage1["validation_max_hold_bars"] <= MAX_HOLD_BARS)
        & (stage1["oos_max_hold_bars"] <= MAX_HOLD_BARS)
        & (stage1["validation_max_leverage"] <= LEVERAGE_CAP)
        & (stage1["oos_max_leverage"] <= LEVERAGE_CAP)
    ].copy()
    if eligible.empty:
        eligible = stage1.copy()
    eligible["target_score"] = (
        np.minimum(eligible["validation_pnl"], eligible["oos_pnl"])
        - 3.0 * np.maximum(0.0, TARGET_MDD - eligible["validation_mdd"])
        - 3.0 * np.maximum(0.0, TARGET_MDD - eligible["oos_mdd"])
    )
    top = eligible.sort_values(["target_score", "oos_pnl", "validation_pnl"], ascending=False).head(12)
    specs: list[ContractSpec] = []
    for row in top.itertuples(index=False):
        for threshold in (0.65, 0.70, 0.80, 0.90):
            specs.append(
                ContractSpec(
                    variant=f"{row.variant}_exit{_tag(threshold)}",
                    fixed_notional=float(row.fixed_notional),
                    leverage=float(row.leverage),
                    tp_price_move=float(row.tp_price_move),
                    sl_price_move=float(row.sl_price_move),
                    long_scale=float(row.long_scale),
                    short_scale=float(row.short_scale),
                    side_filter=int(row.side_filter),
                    exit_threshold=float(threshold),
                    dd_governor=bool(row.dd_governor),
                    stage="stage2_exithead",
                )
            )
    return specs


def _report_for_exit(report: dict[str, Any], threshold: float) -> dict[str, Any]:
    out = copy.deepcopy(report)
    out["contract"]["exit_threshold"] = float(threshold)
    return out


def _notional_at_entry(spec: ContractSpec, side: int, dd: float) -> float:
    side_scale = spec.short_scale if int(side) < 0 else spec.long_scale
    notional = float(spec.fixed_notional) * float(side_scale)
    if spec.dd_governor:
        if dd <= -0.16:
            notional *= 0.35
        elif dd <= -0.12:
            notional *= 0.55
        elif dd <= -0.08:
            notional *= 0.75
    return float(np.clip(notional, 0.0, LEVERAGE_CAP))


def _apply_contract_decisions(dec: pd.DataFrame, spec: ContractSpec) -> pd.DataFrame:
    out = dec.copy()
    active = pd.to_numeric(out["side"], errors="raise").ne(0)
    out.loc[active, "take_profit"] = float(spec.tp_price_move)
    out.loc[active, "stop_loss"] = float(spec.sl_price_move)
    if spec.side_filter:
        out.loc[pd.to_numeric(out["side"], errors="raise").ne(int(spec.side_filter)), "side"] = 0
        if "action" in out.columns:
            out.loc[pd.to_numeric(out["side"], errors="raise").eq(0), "action"] = 0
    return out


@torch.no_grad()
def _replay_contract(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec_base: pd.DataFrame,
    loaded_models: dict[str, tuple[parent.ThreeHeadTabM, dict[str, Any]]],
    spec: ContractSpec,
    *,
    report: dict[str, Any],
    fee: float,
    slip: float,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    dec = _apply_contract_decisions(dec_base, spec)
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
    leverage = 1.0
    margin_fraction = 0.0
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
    margin_sum = 0.0
    max_hold_seen = 0
    max_leverage_seen = 0.0
    max_notional_seen = 0.0
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    use_exit_head = float(spec.exit_threshold) <= 1.0
    route = hard._route_id(frame) if use_exit_head else None
    if use_exit_head:
        base_np, exit_runtime, pos_idx = risk._prepare_exit_runtime(base_x, loaded_models)
    else:
        base_np, exit_runtime, pos_idx = None, None, None

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
        current_dd = eq / max(peak, 1.0e-12) - 1.0
        mdd = min(mdd, current_dd)

        if pos != 0:
            reason = ""
            exit_prob = 0.0
            hold = max(int(i) - int(entry_i), 0)
            max_hold_seen = max(max_hold_seen, hold)
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            elif hold >= MAX_HOLD_BARS:
                reason = "max_hold_1d"
            elif use_exit_head:
                giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(route[i])]  # type: ignore[index]
                prob = risk._predict_exit_prob_one(
                    base_np,  # type: ignore[arg-type]
                    exit_runtime,  # type: ignore[arg-type]
                    pos_idx,  # type: ignore[arg-type]
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
                        float(notional),
                        float(leverage),
                        float(notional * leverage),
                        float(take_profit),
                        float(stop_loss),
                    ],
                    device=device,
                )
                exit_prob = float(prob)
                if prob >= float(spec.exit_threshold):
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
                        "net_per_notional": float(trade_return / max(notional, 1.0e-12)),
                        "notional": float(notional),
                        "margin_fraction": float(margin_fraction),
                        "leverage": float(leverage),
                        "exit_prob": float(exit_prob),
                        "take_profit": float(take_profit),
                        "stop_loss": float(stop_loss),
                        "hold_bars": int(hold),
                        "cash_after": float(cash),
                    }
                )
                pos = 0
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
        entry_dd = cash / max(peak, 1.0e-12) - 1.0
        row_leverage = min(float(spec.leverage), LEVERAGE_CAP)
        row_notional = _notional_at_entry(spec, side, entry_dd)
        row_margin = row_notional / max(row_leverage, 1.0e-12)
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
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
        cash -= cash * float(fee_paid) * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        margin_sum += margin_fraction
        max_leverage_seen = max(max_leverage_seen, leverage)
        max_notional_seen = max(max_notional_seen, notional)
        mfe = 0.0
        mae = 0.0

    if pos != 0:
        exit_px = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        hold = max(len(frame) - 1 - int(entry_i), 0)
        max_hold_seen = max(max_hold_seen, hold)
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
                "net_per_notional": float(trade_return / max(notional, 1.0e-12)),
                "notional": float(notional),
                "margin_fraction": float(margin_fraction),
                "leverage": float(leverage),
                "exit_prob": 0.0,
                "take_profit": float(take_profit),
                "stop_loss": float(stop_loss),
                "hold_bars": int(hold),
                "cash_after": float(cash),
            }
        )

    ledger = pd.DataFrame(rows)
    if not ledger.empty:
        hold = pd.to_numeric(ledger["exit_i"], errors="raise") - pd.to_numeric(ledger["entry_i"], errors="raise")
        max_hold_seen = int(max(max_hold_seen, int(hold.max())))
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
        "max_leverage": float(max_leverage_seen),
        "max_notional": float(max_notional_seen),
        "max_hold_bars": int(max_hold_seen),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
    }
    return metrics, ledger


def _run_specs(
    specs: list[ContractSpec],
    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]],
    loaded: dict[str, tuple[parent.ThreeHeadTabM, dict[str, Any]]],
    report: dict[str, Any],
    fee: float,
    slip: float,
    device: torch.device,
    *,
    save_top_ledgers: bool,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    ledgers: dict[tuple[str, str], pd.DataFrame] = {}
    for idx, spec in enumerate(specs, start=1):
        row: dict[str, Any] = {
            "variant": spec.variant,
            "stage": spec.stage,
            "fixed_notional": spec.fixed_notional,
            "leverage": spec.leverage,
            "tp_price_move": spec.tp_price_move,
            "sl_price_move": spec.sl_price_move,
            "long_scale": spec.long_scale,
            "short_scale": spec.short_scale,
            "side_filter": spec.side_filter,
            "exit_threshold": spec.exit_threshold,
            "dd_governor": spec.dd_governor,
            "max_hold_contract_bars": MAX_HOLD_BARS,
            "leverage_cap": LEVERAGE_CAP,
        }
        for split, (frame, base_x, dec, _base_margin, _base_leverage) in payload.items():
            metrics, ledger = _replay_contract(
                frame,
                base_x,
                dec,
                loaded,
                spec,
                report=_report_for_exit(report, spec.exit_threshold),
                fee=fee,
                slip=slip,
                device=device,
            )
            for key, value in metrics.items():
                row[f"{split}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True) if key == "exit_reasons" else value
            if save_top_ledgers:
                ledgers[(split, spec.variant)] = ledger
        rows.append(row)
        if idx == 1 or idx % 50 == 0 or idx == len(specs):
            print(
                json.dumps(
                    {
                        "idx": idx,
                        "total": len(specs),
                        "stage": spec.stage,
                        "variant": spec.variant,
                        "validation_pnl": row["validation_pnl"],
                        "validation_mdd": row["validation_mdd"],
                        "oos_pnl": row["oos_pnl"],
                        "oos_mdd": row["oos_mdd"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    df = pd.DataFrame(rows)
    df["pass_target"] = (
        (df["validation_pnl"] >= TARGET_PNL)
        & (df["oos_pnl"] >= TARGET_PNL)
        & (df["validation_mdd"] >= TARGET_MDD)
        & (df["oos_mdd"] >= TARGET_MDD)
        & (df["validation_max_hold_bars"] <= MAX_HOLD_BARS)
        & (df["oos_max_hold_bars"] <= MAX_HOLD_BARS)
        & (df["validation_max_leverage"] <= LEVERAGE_CAP)
        & (df["oos_max_leverage"] <= LEVERAGE_CAP)
    )
    df["target_score"] = (
        np.minimum(df["validation_pnl"], df["oos_pnl"])
        - 4.0 * np.maximum(0.0, TARGET_MDD - df["validation_mdd"])
        - 4.0 * np.maximum(0.0, TARGET_MDD - df["oos_mdd"])
        - 0.02 * np.maximum(0.0, df["validation_max_hold_bars"] - MAX_HOLD_BARS)
        - 0.02 * np.maximum(0.0, df["oos_max_hold_bars"] - MAX_HOLD_BARS)
    )
    return df.sort_values(["pass_target", "target_score", "oos_pnl", "validation_pnl"], ascending=False).reset_index(drop=True), ledgers


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads(v18.REPORT_PATH.read_text(encoding="utf-8"))
    device = parent._device("cuda")
    payload, extra = v18._prepare_payload(report, device)
    fee, slip = v18.omega._load_fee_slip()

    stage1_specs = _stage1_specs()
    stage1, _ = _run_specs(stage1_specs, payload, extra["loaded"], report, fee, slip, device, save_top_ledgers=False)
    stage1.to_csv(OUT_DIR / "stage1_noexit_grid.csv", index=False)

    stage2_specs = _stage2_specs(stage1)
    stage2, stage2_ledgers = _run_specs(stage2_specs, payload, extra["loaded"], report, fee, slip, device, save_top_ledgers=True)
    stage2.to_csv(OUT_DIR / "stage2_exithead_grid.csv", index=False)

    combined = pd.concat([stage1.assign(search_stage="stage1"), stage2.assign(search_stage="stage2")], ignore_index=True)
    combined = combined.sort_values(["pass_target", "target_score", "oos_pnl", "validation_pnl"], ascending=False).reset_index(drop=True)
    combined.to_csv(OUT_DIR / "combined_grid.csv", index=False)
    passed = combined[combined["pass_target"]].copy()
    passed.to_csv(OUT_DIR / "target_pass.csv", index=False)

    ledger_variants = set(combined.head(10)["variant"].astype(str).tolist()) | set(passed.head(10)["variant"].astype(str).tolist())
    for (split, variant), ledger in stage2_ledgers.items():
        if variant in ledger_variants:
            ledger.to_csv(LEDGER_DIR / f"{split}_{variant}_ledger.csv", index=False)

    report_out = {
        "model_id": MODEL_ID,
        "source_models": {
            "live_model": "omega3_aggressive_compensated_scale200_cap090_20260618",
            "omega44_model": "omega4_4_v18_baseline_20260624",
        },
        "contract": {
            "validation_pnl_min": TARGET_PNL,
            "oos_pnl_min": TARGET_PNL,
            "mdd_floor_pct": TARGET_MDD,
            "max_hold_bars": MAX_HOLD_BARS,
            "max_hold_days": 1.0,
            "leverage_cap": LEVERAGE_CAP,
            "tp_sl_contract": "direct price-move barriers; no notional division in TP/SL lines",
        },
        "method": [
            "Use Omega4.4 v18 parent decisions and optional exit head.",
            "Borrow live omega3 aggressive exposure idea through fixed aggressive notional and side-specific short scaling.",
            "Force universal one-day max-hold exit on every open position.",
            "Keep leverage capped at 5x and compute notional = margin_fraction * leverage.",
        ],
        "stage1_count": int(len(stage1)),
        "stage2_count": int(len(stage2)),
        "pass_count": int(len(passed)),
        "top20": combined.head(20).to_dict(orient="records"),
        "passed": passed.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "stage1_grid": str(OUT_DIR / "stage1_noexit_grid.csv"),
            "stage2_grid": str(OUT_DIR / "stage2_exithead_grid.csv"),
            "combined_grid": str(OUT_DIR / "combined_grid.csv"),
            "target_pass": str(OUT_DIR / "target_pass.csv"),
            "ledgers": str(LEDGER_DIR),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report_out, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(OUT_DIR / "report.json"),
                "pass_count": int(len(passed)),
                "top_variant": str(combined.iloc[0]["variant"]) if len(combined) else "",
                "top_validation_pnl": float(combined.iloc[0]["validation_pnl"]) if len(combined) else 0.0,
                "top_oos_pnl": float(combined.iloc[0]["oos_pnl"]) if len(combined) else 0.0,
                "top_validation_mdd": float(combined.iloc[0]["validation_mdd"]) if len(combined) else 0.0,
                "top_oos_mdd": float(combined.iloc[0]["oos_mdd"]) if len(combined) else 0.0,
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
