#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import plot_omega1_2_1_cash_alpha43_sleeve_trade_charts_20260608 as chart  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_label_family_20260606 as label_family  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as base  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_cash_alpha43_arch_feedback_20260608"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
RISK = chart.RISK
THRESHOLD = chart.THRESHOLD


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


def _metrics_policy(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    fallback_action: np.ndarray,
    fallback_conf: np.ndarray,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    takeover_policy: str,
) -> dict[str, Any]:
    arrays = sleeve._arrays(frame)
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = sleeve.Position()
    wins = trades = 0
    primary_entries = fallback_entries = long_entries = short_entries = 0
    primary_takeovers = 0
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
                primary_side = int(dec.iloc[int(i)].get("side", 0) or 0)
                should_takeover = False
                if takeover_policy == "any_primary":
                    should_takeover = True
                elif takeover_policy == "opposite_primary_only":
                    should_takeover = primary_side != 0 and primary_side != pos.side
                elif takeover_policy == "loss_or_opposite_primary":
                    should_takeover = unreal < 0.0 or (primary_side != 0 and primary_side != pos.side)
                elif takeover_policy == "profitable_primary_takeover_only":
                    should_takeover = unreal > 0.0
                elif takeover_policy == "no_primary_takeover":
                    should_takeover = False
                else:
                    raise RuntimeError(f"unknown takeover_policy: {takeover_policy}")
                if should_takeover:
                    reason = "primary_takeover"
                    primary_takeovers += 1
            if reason:
                cash, win = sleeve._close_position(cash, arrays, pos, i, fee_eff, slip_eff)
                trades += 1
                wins += int(win)
                reasons[f"{pos.sleeve}_{reason}"] = reasons.get(f"{pos.sleeve}_{reason}", 0) + 1
                pos = sleeve.Position()
            else:
                continue

        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1.0e-12) - 1.0)
        if bool(active[i]):
            row = dec.iloc[int(i)]
            side = int(row.get("side", 0) or 0)
            if side != 0:
                cash, pos, entered = sleeve._open_position(cash, arrays, i, side, "primary", None, row, fee_eff, slip_eff)
                if entered:
                    primary_entries += 1
                    long_entries += int(side > 0)
                    short_entries += int(side < 0)
            continue

        action = int(fallback_action[int(i)]) if int(i) < len(fallback_action) else sleeve.ACTION_CASH
        conf = float(fallback_conf[int(i)]) if int(i) < len(fallback_conf) else 0.0
        if action not in (sleeve.ACTION_LONG, sleeve.ACTION_SHORT) or conf < THRESHOLD:
            continue
        side = 1 if action == sleeve.ACTION_LONG else -1
        cash, pos, entered = sleeve._open_position(cash, arrays, i, side, "fallback", RISK, None, fee_eff, slip_eff)
        if entered:
            fallback_entries += 1
            long_entries += int(side > 0)
            short_entries += int(side < 0)

    if pos.side != 0:
        cash, win = sleeve._close_position(cash, arrays, pos, len(frame) - 1, fee_eff, slip_eff)
        trades += 1
        wins += int(win)
        reasons[f"{pos.sleeve}_forced_end"] = reasons.get(f"{pos.sleeve}_forced_end", 0) + 1

    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "primary_entries": int(primary_entries),
        "fallback_entries": int(fallback_entries),
        "primary_takeovers": int(primary_takeovers),
        "exit_reasons": reasons,
    }


def _metric_row(prefix: str, m: dict[str, Any]) -> dict[str, Any]:
    return sleeve._metric_row(prefix, m)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_dec, val_features = chart._build_split(frames, "validation")
    oos_frame, oos_dec, oos_features = chart._build_split(frames, "oos")
    val_cash = ~omega._active(val_dec)
    y_val, valid_mask, label_diag = label_family._label_family("tb_atr08_h48", val_frame, val_dec, val_cash, 2025)
    train_mask = val_cash & valid_mask
    val_action, val_conf, oof_diag = label_family._predict_oof("hgb", val_features, y_val, train_mask, seed=260608)
    oos_action, oos_conf, _fitted = label_family._fit_predict("hgb", val_features, y_val, train_mask, oos_features, seed=260608)
    rows: list[dict[str, Any]] = []
    for policy in (
        "any_primary",
        "opposite_primary_only",
        "loss_or_opposite_primary",
        "profitable_primary_takeover_only",
        "no_primary_takeover",
    ):
        val_m = _metrics_policy(val_frame, val_dec, val_action, val_conf, fee=fee, slip=slip, cost_mult=3.0, takeover_policy=policy)
        oos_m = _metrics_policy(oos_frame, oos_dec, oos_action, oos_conf, fee=fee, slip=slip, cost_mult=3.0, takeover_policy=policy)
        rows.append({"variant": f"takeover_{policy}", "takeover_policy": policy, **_metric_row("val", val_m), **_metric_row("oos", oos_m)})
    ranking = pd.DataFrame(rows)
    ranking["val_delta_pnl"] = ranking["val_pnl"] - sleeve.AGGRESSIVE_VAL["pnl"]
    ranking["oos_delta_pnl"] = ranking["oos_pnl"] - sleeve.AGGRESSIVE_OOS["pnl"]
    ranking["val_delta_mdd"] = ranking["val_mdd"] - sleeve.AGGRESSIVE_VAL["mdd"]
    ranking["oos_delta_mdd"] = ranking["oos_mdd"] - sleeve.AGGRESSIVE_OOS["mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "primary_takeover_policy_ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "candidate": chart.CANDIDATE,
        "test": "primary_takeover_policy_sweep",
        "label_diag": label_diag,
        "oof_diag": oof_diag,
        "ranking": ranking.to_dict(orient="records"),
        "artifacts": {"ranking": str(OUT_DIR / "primary_takeover_policy_ranking.csv"), "report": str(OUT_DIR / "report.json")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top": report["ranking"][:5]}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
