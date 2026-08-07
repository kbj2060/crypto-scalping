#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_stop_loss_hazard_veto_20260604 as base_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_mfe_trailing_exit_20260604"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _metrics_trailing(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    mfe_trigger: float,
    lock_profit: float,
) -> dict[str, Any]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = np.asarray(omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    equity_curve: list[float] = [cash]
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    notional = 0.0
    tp = 0.0
    sl = 0.0
    mfe = 0.0
    trades: list[float] = []
    reasons: dict[str, int] = {}
    long_entries = 0
    short_entries = 0

    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            reason = ""
            if unreal >= tp:
                reason = "take_profit"
            elif unreal <= -abs(sl):
                reason = "stop_loss"
            elif mfe >= float(mfe_trigger) and unreal <= float(lock_profit):
                reason = "mfe_trailing_exit"
            if reason:
                filled, exit_px, exit_fee, _ = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if filled:
                    raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                    before = cash
                    cash = cash * (1.0 + raw_exit * notional)
                    cash -= before * exit_fee * notional
                    trades.append(float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0))
                    reasons[reason] = reasons.get(reason, 0) + 1
                    pos = 0
                    equity_curve.append(cash)
                    continue
        if pos != 0 or not bool(active[i]):
            if pos == 0:
                equity_curve.append(cash)
            continue
        row = dec.iloc[int(i)]
        side = int(row["side"])
        if side == 0:
            equity_curve.append(cash)
            continue
        filled, px, entry_fee, _ = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            equity_curve.append(cash)
            continue
        pos = side
        if pos > 0:
            long_entries += 1
        else:
            short_entries += 1
        entry_price = float(px)
        entry_equity = cash
        notional = float(row["notional_exposure"])
        tp = float(row["take_profit"])
        sl = float(row["stop_loss"])
        cash -= cash * entry_fee * notional
        mfe = 0.0
        equity_curve.append(cash)

    if pos != 0:
        exit_px = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trades.append(float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0))
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        equity_curve.append(cash)

    eq = np.asarray(equity_curve, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(peak, 1e-12) - 1.0) * 100.0
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(dd.min()) if len(dd) else 0.0,
        "trades": int(len(trades)),
        "wr": float(np.mean(np.asarray(trades) > 0.0)) if trades else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--out-suffix", default="practical_mfe_trailing_20260604")
    args = ap.parse_args()

    out_dir = OUT_DIR.with_name(f"{OUT_DIR.name}_{args.out_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = tabm._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_pred = base_eval._read_predictions(base_eval.BASE_DIR / "validation_predictions_2025_true3head.csv", frames["val_raw"])
    oos_pred = base_eval._read_predictions(base_eval.BASE_DIR / "oos_predictions_2026_true3head.csv", frames["oos_raw"])
    val_dec = base_eval._decisions(val_pred, oof=True)
    oos_dec = base_eval._decisions(oos_pred, oof=False)
    base_val = base_eval._metrics(frames["val_raw"], val_dec, fee=fee, slip=slip, cost_mult=args.cost_mult)
    base_oos = base_eval._metrics(frames["oos_raw"], oos_dec, fee=fee, slip=slip, cost_mult=args.cost_mult)
    rows: list[dict[str, Any]] = [
        {
            "variant": "baseline_no_trailing",
            "mfe_trigger": 999.0,
            "lock_profit": 999.0,
            "val_pnl": base_val["pnl"],
            "val_mdd": base_val["mdd"],
            "val_wr": base_val["wr"],
            "val_trades": base_val["trades"],
            "oos_pnl": base_oos["pnl"],
            "oos_mdd": base_oos["mdd"],
            "oos_wr": base_oos["wr"],
            "oos_trades": base_oos["trades"],
            "val_exit_reasons": base_val.get("exit_reasons", {}),
            "oos_exit_reasons": base_oos.get("exit_reasons", {}),
        }
    ]
    for trigger in [0.006, 0.008, 0.010, 0.012, 0.016, 0.020]:
        for lock in [-0.002, 0.0, 0.002, 0.004, 0.006]:
            if lock >= trigger:
                continue
            val = _metrics_trailing(frames["val_raw"], val_dec, fee=fee, slip=slip, cost_mult=args.cost_mult, mfe_trigger=trigger, lock_profit=lock)
            oos = _metrics_trailing(frames["oos_raw"], oos_dec, fee=fee, slip=slip, cost_mult=args.cost_mult, mfe_trigger=trigger, lock_profit=lock)
            rows.append(
                {
                    "variant": "mfe_trailing_exit",
                    "mfe_trigger": float(trigger),
                    "lock_profit": float(lock),
                    "val_pnl": val["pnl"],
                    "val_mdd": val["mdd"],
                    "val_wr": val["wr"],
                    "val_trades": val["trades"],
                    "oos_pnl": oos["pnl"],
                    "oos_mdd": oos["mdd"],
                    "oos_wr": oos["wr"],
                    "oos_trades": oos["trades"],
                    "val_exit_reasons": val.get("exit_reasons", {}),
                    "oos_exit_reasons": oos.get("exit_reasons", {}),
                }
            )
    ranking = pd.DataFrame(rows).sort_values(["val_pnl", "val_mdd"], ascending=[False, False]).reset_index(drop=True)
    ranking.to_csv(out_dir / "mfe_trailing_grid.csv", index=False)
    report = {
        "design": "Fixed practical entries with an MFE-based trailing exit overlay. Params are selected on 2025 validation only; 2026 OOS is reported fixed.",
        "baseline": {"validation": base_val, "oos": base_oos},
        "ranking": rows,
        "artifacts": {"ranking": str(out_dir / "mfe_trailing_grid.csv")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(ranking.head(12).to_string(index=False))
    print(json.dumps({"report": str(out_dir / "report.json"), "ranking": str(out_dir / "mfe_trailing_grid.csv")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
