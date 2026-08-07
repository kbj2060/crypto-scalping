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

import eval_omega1_2_true3head_overlays_20260604 as overlay  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as th  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit  # noqa: E402


MODEL_ID = "omega_live_parent_price_maxhold1d_lev5_target100_20260629"
BASE_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LEDGER_DIR = OUT_DIR / "ledgers"

MAX_HOLD_BARS = 288
LEVERAGE_CAP = 5.0
TARGET_PNL = 100.0
TARGET_MDD = -20.0


@dataclass(frozen=True)
class LiveSpec:
    variant: str
    q_mode: str
    fixed_notional: float
    leverage: float
    tp_price_move: float
    sl_price_move: float
    long_scale: float = 1.0
    short_scale: float = 1.0
    side_filter: int = 0
    dd_governor: bool = False


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


def _tag(value: float | str) -> str:
    if isinstance(value, str):
        return value.replace(".", "p")
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _align(frame: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    out = frame[["timestamp"]].merge(pred, on="timestamp", how="left", validate="one_to_one")
    if out.isna().any().any():
        bad = out.loc[out.isna().any(axis=1), "timestamp"].head(10).tolist()
        raise RuntimeError(f"prediction alignment produced NaN: {bad}")
    return out


def _build_split(frames: dict[str, pd.DataFrame], split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if split == "validation":
        frame = frames["val_raw"].reset_index(drop=True)
        pred = pd.read_csv(BASE_DIR / "validation_predictions_2025_true3head.csv", parse_dates=["timestamp"])
        prefix = "omega1_regime3_expertdq_oof_"
        src = _align(frame, pred)
    elif split == "oos":
        frame = frames["oos_raw"].reset_index(drop=True)
        pred = pd.read_csv(BASE_DIR / "oos_predictions_2026_true3head.csv", parse_dates=["timestamp"])
        prefix = "omega1_regime3_expertdq_"
        src = _align(frame, pred)
    else:
        raise RuntimeError(f"unknown split: {split}")
    dec = _build_dec(src, prefix, oof=(split == "validation"), q_mode="live")
    return frame, src, dec, prefix


def _thresholds(src: pd.DataFrame, prefix: str, q_mode: str) -> np.ndarray:
    if q_mode == "live":
        expert = src[f"{prefix}router_expert"].astype(str).replace({"chop_expert": "chop"}).to_numpy()
        return np.asarray([overlay.THR_MAP.get(str(x), overlay.THR_MAP["chop"]) for x in expert], dtype=np.float64)
    try:
        value = float(q_mode)
    except ValueError as exc:
        raise RuntimeError(f"unknown q_mode: {q_mode}") from exc
    return np.full(len(src), value, dtype=np.float64)


def _build_dec(src: pd.DataFrame, prefix: str, *, oof: bool, q_mode: str) -> pd.DataFrame:
    work = src.copy()
    q = pd.to_numeric(work[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    action = pd.to_numeric(work[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    thr = _thresholds(work, prefix, q_mode)
    work[f"{prefix}quality_threshold"] = thr
    work[f"{prefix}final_action"] = np.where(q >= thr, action, omega.ACTION_CASH).astype(np.int64)
    dec = omega._to_fixed_decisions(work, oof=oof)
    active = omega._active(dec)
    for expert, scale in overlay.SCALE_MAP.items():
        key = "chop_expert" if expert == "chop" else expert
        mask = active & dec["router_expert"].astype(str).eq(key)
        ratio = float(scale) / float(overlay.BASE_SCALES[key])
        dec.loc[mask, "notional_exposure"] = pd.to_numeric(dec.loc[mask, "notional_exposure"], errors="raise") * ratio
        dec.loc[mask, "position_fraction"] = pd.to_numeric(dec.loc[mask, "position_fraction"], errors="raise") * ratio
    active = omega._active(dec)
    dec.loc[active, "max_hold_bars"] = MAX_HOLD_BARS
    dec.loc[active, "cooldown_bars"] = 0
    return dec


def _specs() -> list[LiveSpec]:
    out: list[LiveSpec] = []
    for q_mode in ("live", "0.60", "0.64", "0.70"):
        for notional in (0.70, 0.90, 1.10, 1.30, 1.50, 1.75, 2.00, 2.30, 2.60):
            for tp, sl in ((0.016, 0.008), (0.020, 0.010), (0.026, 0.014), (0.035, 0.018), (0.052, 0.028)):
                for long_scale, short_scale in ((1.0, 1.0), (0.75, 1.35), (0.65, 1.55), (1.20, 0.90)):
                    for side_filter in (0, -1):
                        for dd_governor in (False, True):
                            out.append(
                                LiveSpec(
                                    variant=(
                                        f"q{_tag(q_mode)}_n{_tag(notional)}_tp{_tag(tp)}_sl{_tag(sl)}"
                                        f"_l{_tag(long_scale)}_s{_tag(short_scale)}"
                                        f"{'_shortonly' if side_filter < 0 else ''}"
                                        f"{'_ddgov' if dd_governor else ''}"
                                    ),
                                    q_mode=q_mode,
                                    fixed_notional=float(notional),
                                    leverage=LEVERAGE_CAP,
                                    tp_price_move=float(tp),
                                    sl_price_move=float(sl),
                                    long_scale=float(long_scale),
                                    short_scale=float(short_scale),
                                    side_filter=int(side_filter),
                                    dd_governor=bool(dd_governor),
                                )
                            )
    return out


def _notional(spec: LiveSpec, side: int, dd: float) -> float:
    n = float(spec.fixed_notional) * (float(spec.short_scale) if int(side) < 0 else float(spec.long_scale))
    if spec.dd_governor:
        if dd <= -0.16:
            n *= 0.35
        elif dd <= -0.12:
            n *= 0.55
        elif dd <= -0.08:
            n *= 0.75
    return float(np.clip(n, 0.0, LEVERAGE_CAP))


def _duration_days(frame: pd.DataFrame) -> float:
    ts = pd.to_datetime(frame["timestamp"], errors="raise")
    return max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0, 1.0e-9)


def _apply_spec_dec(dec: pd.DataFrame, spec: LiveSpec) -> pd.DataFrame:
    out = dec.copy()
    active = omega._active(out)
    out.loc[active, "take_profit"] = float(spec.tp_price_move)
    out.loc[active, "stop_loss"] = float(spec.sl_price_move)
    if spec.side_filter:
        bad = pd.to_numeric(out["side"], errors="raise").ne(int(spec.side_filter))
        out.loc[bad, "side"] = 0
        if "action" in out.columns:
            out.loc[bad, "action"] = 0
    return out


def _replay(frame: pd.DataFrame, dec_base: pd.DataFrame, spec: LiveSpec, *, fee: float, slip: float) -> tuple[dict[str, Any], pd.DataFrame]:
    dec = _apply_spec_dec(dec_base, spec)
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
    tp = 0.0
    sl = 0.0
    mfe = 0.0
    mae = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = margin_sum = leverage_sum = 0.0
    max_hold_seen = 0
    max_leverage_seen = 0.0
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = price_exit._price_move(arrays, int(i), side=pos, entry_price=entry_price, slip_eff=slip_eff)
            unreal = move * notional
            mfe = max(mfe, move)
            mae = min(mae, move)
            eq = cash * (1.0 + unreal)
        else:
            move = 0.0
            eq = cash
        peak = max(peak, eq)
        dd = eq / max(peak, 1.0e-12) - 1.0
        mdd = min(mdd, dd)
        if pos != 0:
            hold = max(int(i) - int(entry_i), 0)
            max_hold_seen = max(max_hold_seen, hold)
            reason = ""
            if tp > 0.0 and move >= tp:
                reason = "take_profit"
            elif sl > 0.0 and move <= -abs(sl):
                reason = "stop_loss"
            elif hold >= MAX_HOLD_BARS:
                reason = "max_hold_1d"
            if reason:
                filled, exit_px, exit_fee, route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                ret = cash / max(entry_equity, 1.0e-12) - 1.0
                trades += 1
                wins += int(cash > entry_equity)
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
                        "win": int(cash > entry_equity),
                        "raw_exit_price_move": float(raw_exit),
                        "mfe_price_move": float(mfe),
                        "mae_price_move": float(mae),
                        "trade_return": float(ret),
                        "notional": float(notional),
                        "margin_fraction": float(margin_fraction),
                        "leverage": float(leverage),
                        "take_profit": float(tp),
                        "stop_loss": float(sl),
                        "hold_bars": int(hold),
                        "cash_after": float(cash),
                        "exit_route": route,
                    }
                )
                pos = 0
                continue
        if pos != 0 or not bool(active[i]):
            continue
        row = dec.iloc[int(i)]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        entry_dd = cash / max(peak, 1.0e-12) - 1.0
        row_leverage = min(float(spec.leverage), LEVERAGE_CAP)
        row_notional = _notional(spec, side, entry_dd)
        if row_notional <= 0.0:
            continue
        pos = side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_signal_i = int(i)
        leverage = row_leverage
        notional = row_notional
        margin_fraction = row_notional / max(row_leverage, 1.0e-12)
        tp = float(row.get("take_profit", 0.0) or 0.0)
        sl = float(row.get("stop_loss", 0.0) or 0.0)
        cash -= cash * entry_fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        margin_sum += margin_fraction
        leverage_sum += leverage
        max_leverage_seen = max(max_leverage_seen, leverage)
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
        ret = cash / max(entry_equity, 1.0e-12) - 1.0
        trades += 1
        wins += int(cash > entry_equity)
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
                "win": int(cash > entry_equity),
                "raw_exit_price_move": float(raw_exit),
                "mfe_price_move": float(mfe),
                "mae_price_move": float(mae),
                "trade_return": float(ret),
                "notional": float(notional),
                "margin_fraction": float(margin_fraction),
                "leverage": float(leverage),
                "take_profit": float(tp),
                "stop_loss": float(sl),
                "hold_bars": int(hold),
                "cash_after": float(cash),
                "exit_route": "forced_end",
            }
        )
    ledger = pd.DataFrame(rows)
    if not ledger.empty:
        max_hold_seen = int(max(max_hold_seen, int((pd.to_numeric(ledger["exit_i"]) - pd.to_numeric(ledger["entry_i"])).max())))
    n_entries = max(long_entries + short_entries, 1)
    return (
        {
            "pnl": float((cash - 1.0) * 100.0),
            "mdd": float(mdd * 100.0),
            "trades": int(trades),
            "wr": float(wins / trades) if trades else 0.0,
            "trades_per_day": float(trades / _duration_days(frame)),
            "avg_notional": float(notional_sum / n_entries),
            "avg_margin_fraction": float(margin_sum / n_entries),
            "avg_leverage": float(leverage_sum / n_entries),
            "max_leverage": float(max_leverage_seen),
            "max_hold_bars": int(max_hold_seen),
            "long_entries": int(long_entries),
            "short_entries": int(short_entries),
            "exit_reasons": reasons,
        },
        ledger,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    frames = th._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    split_payload: dict[str, tuple[pd.DataFrame, pd.DataFrame, str]] = {}
    for split in ("validation", "oos"):
        frame, src, _dec, prefix = _build_split(frames, split)
        split_payload[split] = (frame, src, prefix)

    rows: list[dict[str, Any]] = []
    top_ledgers: dict[tuple[str, str], pd.DataFrame] = {}
    specs = _specs()
    dec_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for idx, spec in enumerate(specs, start=1):
        rec: dict[str, Any] = {
            "variant": spec.variant,
            "q_mode": spec.q_mode,
            "fixed_notional": spec.fixed_notional,
            "leverage": spec.leverage,
            "tp_price_move": spec.tp_price_move,
            "sl_price_move": spec.sl_price_move,
            "long_scale": spec.long_scale,
            "short_scale": spec.short_scale,
            "side_filter": spec.side_filter,
            "dd_governor": spec.dd_governor,
            "max_hold_contract_bars": MAX_HOLD_BARS,
            "leverage_cap": LEVERAGE_CAP,
        }
        ledgers: dict[str, pd.DataFrame] = {}
        for split, (frame, src, prefix) in split_payload.items():
            key = (split, spec.q_mode)
            if key not in dec_cache:
                dec_cache[key] = _build_dec(src, prefix, oof=(split == "validation"), q_mode=spec.q_mode)
            metrics, ledger = _replay(frame, dec_cache[key], spec, fee=fee, slip=slip)
            ledgers[split] = ledger
            for k, v in metrics.items():
                rec[f"{split}_{k}"] = json.dumps(v, ensure_ascii=False, sort_keys=True) if k == "exit_reasons" else v
        rec["pass_target"] = (
            rec["validation_pnl"] >= TARGET_PNL
            and rec["oos_pnl"] >= TARGET_PNL
            and rec["validation_mdd"] >= TARGET_MDD
            and rec["oos_mdd"] >= TARGET_MDD
            and rec["validation_max_hold_bars"] <= MAX_HOLD_BARS
            and rec["oos_max_hold_bars"] <= MAX_HOLD_BARS
            and rec["validation_max_leverage"] <= LEVERAGE_CAP
            and rec["oos_max_leverage"] <= LEVERAGE_CAP
        )
        rec["target_score"] = min(float(rec["validation_pnl"]), float(rec["oos_pnl"])) - 4.0 * max(0.0, TARGET_MDD - float(rec["validation_mdd"])) - 4.0 * max(0.0, TARGET_MDD - float(rec["oos_mdd"]))
        rows.append(rec)
        if idx == 1 or idx % 100 == 0 or idx == len(specs):
            print(json.dumps({"idx": idx, "total": len(specs), "variant": spec.variant, "validation_pnl": rec["validation_pnl"], "validation_mdd": rec["validation_mdd"], "oos_pnl": rec["oos_pnl"], "oos_mdd": rec["oos_mdd"]}, ensure_ascii=False), flush=True)

    grid = pd.DataFrame(rows).sort_values(["pass_target", "target_score", "oos_pnl", "validation_pnl"], ascending=False).reset_index(drop=True)
    grid.to_csv(OUT_DIR / "live_parent_price_maxhold1d_grid.csv", index=False)
    passed = grid[grid["pass_target"]].copy()
    passed.to_csv(OUT_DIR / "target_pass.csv", index=False)

    keep_variants = set(grid.head(10)["variant"].astype(str).tolist()) | set(passed.head(10)["variant"].astype(str).tolist())
    for _, row in grid.head(20).iterrows():
        spec = next(s for s in specs if s.variant == row["variant"])
        for split, (frame, src, prefix) in split_payload.items():
            dec = _build_dec(src, prefix, oof=(split == "validation"), q_mode=spec.q_mode)
            _metrics, ledger = _replay(frame, dec, spec, fee=fee, slip=slip)
            if spec.variant in keep_variants:
                ledger.to_csv(LEDGER_DIR / f"{split}_{spec.variant}_ledger.csv", index=False)
                top_ledgers[(split, spec.variant)] = ledger

    report = {
        "model_id": MODEL_ID,
        "source_model": "omega3_aggressive_compensated_scale200_cap090_20260618",
        "omega44_reference": "omega4_4_v18_baseline_20260624 stage1 maxhold result kept separately",
        "contract": {
            "validation_pnl_min": TARGET_PNL,
            "oos_pnl_min": TARGET_PNL,
            "mdd_floor_pct": TARGET_MDD,
            "max_hold_bars": MAX_HOLD_BARS,
            "leverage_cap": LEVERAGE_CAP,
            "tp_sl_contract": "direct price-move barriers, no notional division",
        },
        "pass_count": int(len(passed)),
        "top20": grid.head(20).to_dict(orient="records"),
        "passed": passed.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "grid": str(OUT_DIR / "live_parent_price_maxhold1d_grid.csv"),
            "target_pass": str(OUT_DIR / "target_pass.csv"),
            "ledgers": str(LEDGER_DIR),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "pass_count": int(len(passed)), "top": grid.head(5).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
