#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_label_family_20260606 as label_family  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as base  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_cash_alpha43_sleeve_chart_20260608"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
CANDIDATE = "alpha43_style_tb_atr08_h48_hgb_base_tp026_sl014_n030_h192_thr0.55"
RISK = sleeve.FallbackRisk("base_tp026_sl014_n030_h192", 0.026, 0.014, 0.30, 2.0, 192)
THRESHOLD = 0.55
FORBIDDEN_PREFIXES = ("clean_regime4_", "regime4_pred_", "teacher_")
FORBIDDEN_EXACT = {"tp_sl_action_score"}


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


def _audit_features(cols: list[str]) -> None:
    bad = [c for c in cols if c in FORBIDDEN_EXACT or c.startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise RuntimeError(f"forbidden chart feature columns: {bad}")


def _build_split(frames: dict[str, pd.DataFrame], split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame, src, dec0, prefix = base._build_split(frames, split)
    dec = sleeve._apply_aggressive(dec0)
    feat = sleeve._extra_features(base._feature_frame(frame, src, dec0, prefix), dec)
    _audit_features(list(feat.columns))
    return frame, dec, feat


def _open_position(
    cash: float,
    arrays: dict[str, np.ndarray],
    frame: pd.DataFrame,
    i: int,
    side: int,
    sleeve_name: str,
    row: pd.Series | None,
    fee_eff: float,
    slip_eff: float,
) -> tuple[float, sleeve.Position, dict[str, Any] | None]:
    risk = RISK if sleeve_name == "fallback" else None
    cash_after, pos, entered = sleeve._open_position(cash, arrays, i, side, sleeve_name, risk, row, fee_eff, slip_eff)
    if not entered:
        return cash, sleeve.Position(), None
    return (
        cash_after,
        pos,
        {
            "sleeve": sleeve_name,
            "side": int(side),
            "entry_signal_i": int(i),
            "entry_i": int(pos.entry_i),
            "entry_signal_ts": str(pd.to_datetime(frame["timestamp"].iloc[int(i)])),
            "entry_ts": str(pd.to_datetime(frame["timestamp"].iloc[int(pos.entry_i)])),
            "entry_price": float(pos.entry_price),
            "entry_equity": float(pos.entry_equity),
            "notional": float(pos.notional),
            "leverage": float(pos.leverage),
            "take_profit": float(pos.take_profit),
            "stop_loss": float(pos.stop_loss),
            "max_hold_bars": int(pos.max_hold_bars),
        },
    )


def _close_position(
    cash: float,
    arrays: dict[str, np.ndarray],
    frame: pd.DataFrame,
    pos: sleeve.Position,
    entry_record: dict[str, Any],
    i: int,
    reason: str,
    fee_eff: float,
    slip_eff: float,
) -> tuple[float, bool, dict[str, Any]]:
    cash_after, win = sleeve._close_position(cash, arrays, pos, i, fee_eff, slip_eff)
    exit_px = omega._fill_price(arrays, int(i), int(pos.side), slip_eff, entry=False)
    raw = (exit_px - pos.entry_price) / max(pos.entry_price, 1.0e-12) if pos.side > 0 else (pos.entry_price - exit_px) / max(pos.entry_price, 1.0e-12)
    row = dict(entry_record)
    row.update(
        {
            "exit_i": int(i),
            "exit_ts": str(pd.to_datetime(frame["timestamp"].iloc[int(i)])),
            "exit_price": float(exit_px),
            "exit_reason": str(reason),
            "raw_price_return_pct": float(raw * 100.0),
            "net_trade_return_pct": float((cash_after / max(float(pos.entry_equity), 1.0e-12) - 1.0) * 100.0),
            "cash_before_exit": float(cash),
            "cash_after": float(cash_after),
            "win": int(win),
            "hold_bars": int(i - pos.entry_i),
        }
    )
    return cash_after, win, row


def _replay_with_ledger(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    fallback_action: np.ndarray,
    fallback_conf: np.ndarray,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    arrays = sleeve._arrays(frame)
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = sleeve.Position()
    entry_record: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    wins = 0
    primary_entries = 0
    fallback_entries = 0
    long_entries = 0
    short_entries = 0
    primary_takeovers = 0
    reasons: dict[str, int] = {}

    for i in range(0, len(frame) - 2):
        mark_eq = cash
        if pos.side != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1.0e-12) if pos.side > 0 else (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1.0e-12)
            unreal = raw * pos.notional
            mark_eq = cash * (1.0 + unreal)
            peak = max(peak, mark_eq)
            mdd = min(mdd, mark_eq / max(peak, 1.0e-12) - 1.0)
            reason = ""
            if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                reason = "take_profit"
            elif pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
                reason = "stop_loss"
            elif pos.max_hold_bars > 0 and int(i) - int(pos.entry_i) >= pos.max_hold_bars:
                reason = "max_hold"
            elif pos.sleeve == "fallback" and bool(active[i]):
                reason = "primary_takeover"
                primary_takeovers += 1
            if reason:
                if entry_record is None:
                    raise RuntimeError("open position missing entry_record")
                cash, win, ledger_row = _close_position(cash, arrays, frame, pos, entry_record, i, reason, fee_eff, slip_eff)
                wins += int(win)
                rows.append(ledger_row)
                reasons[f"{pos.sleeve}_{reason}"] = reasons.get(f"{pos.sleeve}_{reason}", 0) + 1
                pos = sleeve.Position()
                entry_record = None
            else:
                equity_rows.append({"timestamp": frame["timestamp"].iloc[i], "equity": float(mark_eq), "cash": float(cash), "sleeve": pos.sleeve, "side": int(pos.side)})
                continue

        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1.0e-12) - 1.0)
        if bool(active[i]):
            row = dec.iloc[int(i)]
            side = int(row.get("side", 0) or 0)
            if side != 0:
                cash, pos, entry_record = _open_position(cash, arrays, frame, i, side, "primary", row, fee_eff, slip_eff)
                if entry_record is not None:
                    primary_entries += 1
                    long_entries += int(side > 0)
                    short_entries += int(side < 0)
            equity_rows.append({"timestamp": frame["timestamp"].iloc[i], "equity": float(cash), "cash": float(cash), "sleeve": pos.sleeve, "side": int(pos.side)})
            continue

        action = int(fallback_action[int(i)]) if int(i) < len(fallback_action) else sleeve.ACTION_CASH
        conf = float(fallback_conf[int(i)]) if int(i) < len(fallback_conf) else 0.0
        if action in (sleeve.ACTION_LONG, sleeve.ACTION_SHORT) and conf >= THRESHOLD:
            side = 1 if action == sleeve.ACTION_LONG else -1
            cash, pos, entry_record = _open_position(cash, arrays, frame, i, side, "fallback", None, fee_eff, slip_eff)
            if entry_record is not None:
                fallback_entries += 1
                long_entries += int(side > 0)
                short_entries += int(side < 0)
        equity_rows.append({"timestamp": frame["timestamp"].iloc[i], "equity": float(cash), "cash": float(cash), "sleeve": pos.sleeve, "side": int(pos.side)})

    if pos.side != 0:
        if entry_record is None:
            raise RuntimeError("open final position missing entry_record")
        cash, win, ledger_row = _close_position(cash, arrays, frame, pos, entry_record, len(frame) - 1, "forced_end", fee_eff, slip_eff)
        wins += int(win)
        rows.append(ledger_row)
        reasons[f"{pos.sleeve}_forced_end"] = reasons.get(f"{pos.sleeve}_forced_end", 0) + 1

    ledger = pd.DataFrame(rows)
    equity = pd.DataFrame(equity_rows)
    trades = int(len(ledger))
    metrics = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": trades,
        "wr": float(wins / trades) if trades else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "primary_entries": int(primary_entries),
        "fallback_entries": int(fallback_entries),
        "primary_takeovers": int(primary_takeovers),
        "exit_reasons": reasons,
    }
    return metrics, ledger, equity


def _plot_chart(split: str, frame: pd.DataFrame, ledger: pd.DataFrame, equity: pd.DataFrame, metrics: dict[str, Any], out: Path) -> None:
    ts = pd.to_datetime(frame["timestamp"], errors="raise")
    close = pd.to_numeric(frame["close"], errors="raise")
    fig, (ax_price, ax_eq) = plt.subplots(2, 1, figsize=(18, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax_price.plot(ts, close, color="#263238", linewidth=0.8, label="close")
    if not ledger.empty:
        for sleeve_name, marker, size in (("primary", "^", 70), ("fallback", "o", 46)):
            sub = ledger[ledger["sleeve"] == sleeve_name]
            if sub.empty:
                continue
            long = sub[sub["side"] > 0]
            short = sub[sub["side"] < 0]
            if not long.empty:
                ax_price.scatter(pd.to_datetime(long["entry_ts"]), long["entry_price"], marker=marker, s=size, color="#1b9e77", edgecolor="white", linewidth=0.5, label=f"{sleeve_name} long entry")
            if not short.empty:
                ax_price.scatter(pd.to_datetime(short["entry_ts"]), short["entry_price"], marker=marker, s=size, color="#d95f02", edgecolor="white", linewidth=0.5, label=f"{sleeve_name} short entry")
        exits = ledger.copy()
        exits["win_bool"] = pd.to_numeric(exits["net_trade_return_pct"], errors="raise") > 0.0
        wins = exits[exits["win_bool"]]
        losses = exits[~exits["win_bool"]]
        if not wins.empty:
            ax_price.scatter(pd.to_datetime(wins["exit_ts"]), wins["exit_price"], marker="x", s=46, color="#2e7d32", label="win exit")
        if not losses.empty:
            ax_price.scatter(pd.to_datetime(losses["exit_ts"]), losses["exit_price"], marker="x", s=46, color="#c62828", label="loss exit")
    eq_ts = pd.to_datetime(equity["timestamp"], errors="raise") if not equity.empty else ts
    eq = pd.to_numeric(equity["equity"], errors="coerce") if not equity.empty else pd.Series(np.ones(len(ts)))
    ax_eq.plot(eq_ts, (eq - 1.0) * 100.0, color="#1565c0", linewidth=1.0, label="equity pnl %")
    ax_price.set_title(
        f"{CANDIDATE} | {split} | PnL {metrics['pnl']:.2f}% MDD {metrics['mdd']:.2f}% WR {metrics['wr']:.2%} trades {metrics['trades']}"
    )
    ax_price.grid(alpha=0.18)
    ax_eq.grid(alpha=0.18)
    ax_price.legend(loc="upper left", fontsize=8, ncol=3)
    ax_eq.legend(loc="upper left", fontsize=8)
    ax_eq.set_ylabel("PnL %")
    ax_price.set_ylabel("price")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_dec, val_features = _build_split(frames, "validation")
    oos_frame, oos_dec, oos_features = _build_split(frames, "oos")
    val_cash = ~omega._active(val_dec)
    y_val, valid_mask, label_diag = label_family._label_family("tb_atr08_h48", val_frame, val_dec, val_cash, 2025)
    train_mask = val_cash & valid_mask
    val_action, val_conf, oof_diag = label_family._predict_oof("hgb", val_features, y_val, train_mask, seed=260608)
    oos_action, oos_conf, _fitted = label_family._fit_predict("hgb", val_features, y_val, train_mask, oos_features, seed=260608)

    val_metrics, val_ledger, val_equity = _replay_with_ledger(val_frame, val_dec, val_action, val_conf, fee=fee, slip=slip, cost_mult=3.0)
    oos_metrics, oos_ledger, oos_equity = _replay_with_ledger(oos_frame, oos_dec, oos_action, oos_conf, fee=fee, slip=slip, cost_mult=3.0)
    val_ledger.to_csv(OUT_DIR / "validation_trade_ledger.csv", index=False)
    oos_ledger.to_csv(OUT_DIR / "oos_trade_ledger.csv", index=False)
    val_equity.to_csv(OUT_DIR / "validation_equity.csv", index=False)
    oos_equity.to_csv(OUT_DIR / "oos_equity.csv", index=False)
    _plot_chart("validation", val_frame, val_ledger, val_equity, val_metrics, OUT_DIR / "validation_trade_chart.png")
    _plot_chart("oos", oos_frame, oos_ledger, oos_equity, oos_metrics, OUT_DIR / "oos_trade_chart.png")
    report = {
        "model_id": MODEL_ID,
        "candidate": CANDIDATE,
        "architecture": {
            "primary": "omega1_2_1_aggressive_compensated_scale200_cap090",
            "sleeve": "Alpha43-style HGB cash-only parent retrained on Omega-only features",
            "label": "triple_barrier atr_mult=0.8 max_hold=48 min_barrier=0.0035",
            "risk": asdict(RISK),
            "threshold": THRESHOLD,
            "feature_count": int(val_features.shape[1]),
            "forbidden_feature_audit": {"passed": True, "forbidden": []},
        },
        "label_diag": label_diag,
        "oof_diag": oof_diag,
        "validation": val_metrics,
        "oos": oos_metrics,
        "artifacts": {
            "validation_chart": str(OUT_DIR / "validation_trade_chart.png"),
            "oos_chart": str(OUT_DIR / "oos_trade_chart.png"),
            "validation_ledger": str(OUT_DIR / "validation_trade_ledger.csv"),
            "oos_ledger": str(OUT_DIR / "oos_trade_ledger.csv"),
            "validation_equity": str(OUT_DIR / "validation_equity.csv"),
            "oos_equity": str(OUT_DIR / "oos_equity.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "validation": val_metrics, "oos": oos_metrics, "charts": report["artifacts"]}, indent=2, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
