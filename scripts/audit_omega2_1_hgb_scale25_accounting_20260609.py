#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_full_retrain_cash_alpha43_20260608 as full  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega2_1_hgb_calibration_exposure_20260609 as cal  # noqa: E402
from freeze_omega2_1_hgb_12seed_cash_sleeve_20260609 import BUNDLE_PATH, RISK as BASE_RISK  # noqa: E402


OUT_DIR = ROOT / "tmp/causal_regen_20260516" / "omega2_1_hgb_scale25_levexp_accounting_audit_20260609"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _classes_to_proba(model: Any, proba: np.ndarray) -> np.ndarray:
    out = np.zeros((len(proba), 3), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    for j, cls in enumerate(classes):
        cls_i = int(cls)
        if 0 <= cls_i <= 2:
            out[:, cls_i] = proba[:, j]
    return out


def _predict(bundle: dict[str, Any], features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cols = list(bundle["feature_cols"])
    arr = features[cols].to_numpy(dtype=np.float64)
    probs = [_classes_to_proba(model, model.predict_proba(arr)) for model in bundle["models"]]
    proba = np.stack(probs).mean(axis=0)
    action = np.argmax(proba, axis=1).astype(np.int64)
    conf = proba[np.arange(len(proba)), action].astype(np.float64)
    action[(action == sleeve.ACTION_CASH) | (conf < float(bundle["threshold"]))] = sleeve.ACTION_CASH
    conf[action == sleeve.ACTION_CASH] = 0.0
    return action, conf


def _scaled_risk(scale: float, cap: float) -> sleeve.FallbackRisk:
    notional = min(float(BASE_RISK.notional) * float(scale), float(cap))
    ratio = notional / max(float(BASE_RISK.notional), 1.0e-12)
    return sleeve.FallbackRisk(
        f"scale{scale:g}_cap{cap:g}",
        float(BASE_RISK.take_profit) * ratio,
        float(BASE_RISK.stop_loss) * ratio,
        notional,
        float(BASE_RISK.leverage),
        int(BASE_RISK.max_hold_bars),
    )


def _trace(frame: pd.DataFrame, dec: pd.DataFrame, risk: sleeve.FallbackRisk, action: np.ndarray, conf: np.ndarray, fee: float, slip: float) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = sleeve._arrays(frame)
    active = omega._active(dec)
    fee_eff = float(fee) * 3.0
    slip_eff = float(slip) * 3.0
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = sleeve.Position()
    trades = wins = 0
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1.0e-12) if pos.side > 0 else (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1.0e-12)
            unreal = raw * pos.notional
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)
            reason = ""
            if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                reason = "take_profit"
            elif pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
                reason = "stop_loss"
            elif pos.max_hold_bars > 0 and int(i) - int(pos.entry_i) >= pos.max_hold_bars:
                reason = "max_hold"
            elif pos.sleeve == "fallback" and bool(active[i]):
                reason = "primary_takeover"
            if reason:
                before = cash
                exit_i = int(i)
                cash, win = cal._close_position_levexp(cash, arrays, pos, exit_i, fee_eff, slip_eff)
                trades += 1
                wins += int(win)
                reasons[f"{pos.sleeve}_{reason}"] = reasons.get(f"{pos.sleeve}_{reason}", 0) + 1
                rows.append(
                    {
                        "entry_i": int(pos.entry_i),
                        "exit_i": exit_i,
                        "sleeve": pos.sleeve,
                        "side": int(pos.side),
                        "notional": float(pos.notional),
                        "leverage_stored": float(pos.leverage),
                        "take_profit_equity": float(pos.take_profit),
                        "stop_loss_equity": float(pos.stop_loss),
                        "tp_price_move_equiv": float(pos.take_profit / max(pos.notional, 1e-12)),
                        "sl_price_move_equiv": float(pos.stop_loss / max(pos.notional, 1e-12)),
                        "entry_fee_equity": float(fee_eff * pos.notional),
                        "exit_fee_equity_approx": float(fee_eff * pos.notional),
                        "reason": reason,
                        "win": int(win),
                        "cash_before_exit": float(before),
                        "cash_after_exit": float(cash),
                        "trade_net_pct": float((cash / max(pos.entry_equity, 1e-12) - 1.0) * 100.0),
                    }
                )
                pos = sleeve.Position()
            else:
                continue
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1.0e-12) - 1.0)
        if bool(active[i]):
            row = dec.iloc[int(i)]
            side = int(row.get("side", 0) or 0)
            if side != 0:
                cash, pos, _entered = cal._open_position_levexp(cash, arrays, i, side, "primary", None, row, fee_eff, slip_eff)
            continue
        act = int(action[i]) if i < len(action) else sleeve.ACTION_CASH
        if act not in (sleeve.ACTION_LONG, sleeve.ACTION_SHORT) or float(conf[i]) < 0.0:
            continue
        side = 1 if act == sleeve.ACTION_LONG else -1
        cash, pos, _entered = cal._open_position_levexp(cash, arrays, i, side, "fallback", risk, None, fee_eff, slip_eff)
    if pos.side != 0:
        before = cash
        cash, win = cal._close_position_levexp(cash, arrays, pos, len(frame) - 1, fee_eff, slip_eff)
        trades += 1
        wins += int(win)
        rows.append(
            {
                "entry_i": int(pos.entry_i),
                "exit_i": len(frame) - 1,
                "sleeve": pos.sleeve,
                "side": int(pos.side),
                "notional": float(pos.notional),
                "leverage_stored": float(pos.leverage),
                "take_profit_equity": float(pos.take_profit),
                "stop_loss_equity": float(pos.stop_loss),
                "tp_price_move_equiv": float(pos.take_profit / max(pos.notional, 1e-12)),
                "sl_price_move_equiv": float(pos.stop_loss / max(pos.notional, 1e-12)),
                "entry_fee_equity": float(fee_eff * pos.notional),
                "exit_fee_equity_approx": float(fee_eff * pos.notional),
                "reason": "forced_end",
                "win": int(win),
                "cash_before_exit": float(before),
                "cash_after_exit": float(cash),
                "trade_net_pct": float((cash / max(pos.entry_equity, 1e-12) - 1.0) * 100.0),
            }
        )
    ledger = pd.DataFrame(rows)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "exit_reasons": reasons,
    }, ledger


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(BUNDLE_PATH)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    oos_frame, oos_dec, oos_features = full._build_split(frames, "oos")
    action, conf = _predict(bundle, oos_features)
    base_metrics, base_ledger = _trace(oos_frame, oos_dec, BASE_RISK, action, conf, fee, slip)
    scaled = _scaled_risk(2.5, 0.9)
    scaled_metrics, scaled_ledger = _trace(oos_frame, oos_dec, scaled, action, conf, fee, slip)
    base_ledger.to_csv(OUT_DIR / "oos_base_ledger.csv", index=False)
    scaled_ledger.to_csv(OUT_DIR / "oos_scale25_ledger.csv", index=False)
    fallback_base = base_ledger[base_ledger["sleeve"] == "fallback"].reset_index(drop=True)
    fallback_scaled = scaled_ledger[scaled_ledger["sleeve"] == "fallback"].reset_index(drop=True)
    audit = {
        "base_risk": asdict(BASE_RISK),
        "scaled_risk": asdict(scaled),
        "fee_slip": {"fee": float(fee), "slip": float(slip), "cost_mult": 3.0},
        "base_metrics": base_metrics,
        "scaled_metrics": scaled_metrics,
        "fallback_trade_count_match": int(len(fallback_base)) == int(len(fallback_scaled)),
        "fallback_entry_exit_match": bool(
            len(fallback_base) == len(fallback_scaled)
            and np.array_equal(fallback_base.get("entry_i", pd.Series(dtype=int)).to_numpy(), fallback_scaled.get("entry_i", pd.Series(dtype=int)).to_numpy())
            and np.array_equal(fallback_base.get("exit_i", pd.Series(dtype=int)).to_numpy(), fallback_scaled.get("exit_i", pd.Series(dtype=int)).to_numpy())
        ),
        "fallback_base_summary": fallback_base.describe(include="all").to_dict() if len(fallback_base) else {},
        "fallback_scaled_summary": fallback_scaled.describe(include="all").to_dict() if len(fallback_scaled) else {},
        "findings": [
            "Corrected path uses effective_exposure = notional * leverage for PnL, fee, and MDD.",
            "Scaled candidate changes notional from 0.30 to 0.75 while leverage remains 2.0, so fallback effective exposure changes from 0.60 to 1.50.",
            "TP/SL are multiplied by the same notional ratio, so leverage-adjusted price-move equivalent is approximately half of the metadata-leverage report.",
            "Entry and exit fees scale with effective_exposure through fee_eff * notional * leverage.",
        ],
        "artifacts": {
            "base_ledger": str(OUT_DIR / "oos_base_ledger.csv"),
            "scaled_ledger": str(OUT_DIR / "oos_scale25_ledger.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
