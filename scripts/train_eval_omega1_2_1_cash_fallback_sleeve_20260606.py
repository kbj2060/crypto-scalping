#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_exposure_selector_20260606 as base  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_cash_fallback_sleeve_20260606"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

AGGRESSIVE_VAL = {"pnl": 100.54272942091158, "mdd": -10.677652697162888, "wr": 0.6363636363636364, "trades": 33}
AGGRESSIVE_OOS = {"pnl": 72.76004148106665, "mdd": -8.108170708968387, "wr": 0.7222222222222222, "trades": 18}

ACTION_CASH = 0
ACTION_LONG = 1
ACTION_SHORT = 2


@dataclass(frozen=True)
class FallbackRisk:
    name: str
    take_profit: float
    stop_loss: float
    notional: float
    leverage: float
    max_hold_bars: int


RISKS = [
    FallbackRisk("micro_tp010_sl007_n030_h96", 0.010, 0.007, 0.30, 2.0, 96),
    FallbackRisk("base_tp026_sl014_n0405_h192", 0.026, 0.014, 0.405, 2.0, 192),
    FallbackRisk("mid_tp030_sl018_n055_h192", 0.030, 0.018, 0.55, 2.0, 192),
]


@dataclass
class Position:
    sleeve: str = ""
    side: int = 0
    entry_price: float = 0.0
    entry_i: int = 0
    entry_equity: float = 1.0
    notional: float = 0.0
    leverage: float = 1.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    max_hold_bars: int = 0


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


def _arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}


def _apply_aggressive(dec: pd.DataFrame) -> pd.DataFrame:
    active = np.flatnonzero(omega._active(dec))
    out = dec.copy().reset_index(drop=True)
    if len(active) == 0:
        return out
    base_notional = pd.to_numeric(out.loc[active, "notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    new_notional = np.minimum(base_notional * 2.0, 0.90)
    ratio = new_notional / np.maximum(base_notional, 1.0e-12)
    out.loc[active, "notional_exposure"] = new_notional
    out.loc[active, "position_fraction"] = new_notional
    out.loc[active, "take_profit"] = pd.to_numeric(out.loc[active, "take_profit"], errors="raise").to_numpy(dtype=np.float64) * ratio
    out.loc[active, "stop_loss"] = pd.to_numeric(out.loc[active, "stop_loss"], errors="raise").to_numpy(dtype=np.float64) * ratio
    out.loc[active, "max_hold_bars"] = 0
    out.loc[active, "cooldown_bars"] = 0
    return out


def _extra_features(features: pd.DataFrame, dec: pd.DataFrame) -> pd.DataFrame:
    out = features.copy().reset_index(drop=True)
    active = omega._active(dec)
    cash = (~active).astype(float)
    out["primary_is_cash"] = cash
    out["primary_active_roll_12"] = pd.Series(active.astype(float)).rolling(12, min_periods=1).mean().to_numpy(dtype=np.float64)
    out["primary_active_roll_48"] = pd.Series(active.astype(float)).rolling(48, min_periods=1).mean().to_numpy(dtype=np.float64)
    cash_streak = np.zeros(len(out), dtype=np.float64)
    cur = 0
    for i, is_cash in enumerate(~active):
        cur = cur + 1 if bool(is_cash) else 0
        cash_streak[i] = cur
    out["primary_cash_streak"] = np.tanh(cash_streak / 144.0)
    bad = [c for c in out.columns if c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_") or c == "tp_sl_action_score"]
    if bad:
        raise RuntimeError(f"forbidden cash fallback feature columns: {bad}")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _simulate_one(
    arrays: dict[str, np.ndarray],
    signal_i: int,
    side: int,
    risk: FallbackRisk,
    *,
    fee_eff: float,
    slip_eff: float,
) -> tuple[float, dict[str, Any]]:
    filled, entry_px, entry_fee, entry_route = omega._try_execution(arrays, int(signal_i), int(side), entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return -1.0e-6, {"active": 0, "reason": "entry_miss", "net": -1.0e-6}
    entry_i = min(int(signal_i) + 1, len(arrays["close"]) - 1)
    end_i = min(len(arrays["close"]) - 2, entry_i + int(risk.max_hold_bars))
    cash = 1.0 - float(entry_fee) * float(risk.notional)
    exit_px: float | None = None
    exit_fee = fee_eff
    reason = "max_hold"
    mfe = 0.0
    mae = 0.0
    for j in range(entry_i, end_i + 1):
        px = float(arrays["close"][j])
        raw = (px * (1.0 - slip_eff) - entry_px) / max(entry_px, 1.0e-12) if side > 0 else (entry_px - px * (1.0 + slip_eff)) / max(entry_px, 1.0e-12)
        unreal = raw * float(risk.notional)
        mfe = max(mfe, unreal)
        mae = min(mae, unreal)
        if unreal >= float(risk.take_profit):
            _ok, exit_px, exit_fee, _route = omega._try_execution(arrays, int(j), int(side), entry=False, fee_base=fee_eff, slip_base=slip_eff)
            reason = "take_profit"
            break
        if unreal <= -abs(float(risk.stop_loss)):
            _ok, exit_px, exit_fee, _route = omega._try_execution(arrays, int(j), int(side), entry=False, fee_base=fee_eff, slip_base=slip_eff)
            reason = "stop_loss"
            break
    if exit_px is None:
        exit_px = omega._fill_price(arrays, min(end_i + 1, len(arrays["close"]) - 1), int(side), slip_eff, entry=False)
    raw_exit = (exit_px - entry_px) / max(entry_px, 1.0e-12) if side > 0 else (entry_px - exit_px) / max(entry_px, 1.0e-12)
    before = cash
    cash = cash * (1.0 + raw_exit * float(risk.notional))
    cash -= before * float(exit_fee) * float(risk.notional)
    net = float(cash - 1.0)
    # Penalize fragile paths that win only after a deep adverse excursion.
    score = net - 0.20 * max(0.0, -mae - 0.012) * float(risk.notional) * float(risk.leverage)
    return score, {"active": 1, "reason": reason, "net": net, "mfe": float(mfe), "mae": float(mae), "entry_route": entry_route}


def _build_labels(frame: pd.DataFrame, dec: pd.DataFrame, risk: FallbackRisk, min_edge: float) -> tuple[np.ndarray, dict[str, Any]]:
    arrays = _arrays(frame)
    active = omega._active(dec)
    cash_idx = np.flatnonzero(~active & (np.arange(len(frame)) < len(frame) - int(risk.max_hold_bars) - 3))
    fee, slip = omega._load_fee_slip()
    fee_eff = float(fee) * 3.0
    slip_eff = float(slip) * 3.0
    y = np.zeros(len(frame), dtype=np.int64)
    best_net = np.zeros(len(frame), dtype=np.float64)
    reasons: dict[str, int] = {}
    for idx in cash_idx:
        long_score, long_meta = _simulate_one(arrays, int(idx), 1, risk, fee_eff=fee_eff, slip_eff=slip_eff)
        short_score, short_meta = _simulate_one(arrays, int(idx), -1, risk, fee_eff=fee_eff, slip_eff=slip_eff)
        if long_score > short_score:
            best_side, best_score, meta = ACTION_LONG, long_score, long_meta
        else:
            best_side, best_score, meta = ACTION_SHORT, short_score, short_meta
        best_net[int(idx)] = float(meta.get("net", best_score))
        reasons[str(meta.get("reason", "unknown"))] = reasons.get(str(meta.get("reason", "unknown")), 0) + 1
        if best_score > float(min_edge):
            y[int(idx)] = int(best_side)
    counts = {str(k): int(v) for k, v in pd.Series(y[cash_idx]).value_counts().sort_index().items()}
    return y, {"cash_rows": int(len(cash_idx)), "label_counts": counts, "best_net_mean": float(np.mean(best_net[cash_idx])) if len(cash_idx) else 0.0, "sim_reasons": reasons}


def _make_model(name: str, seed: int):
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=80, learning_rate=0.035, max_leaf_nodes=7, l2_regularization=2.0, random_state=int(seed))
    if name == "extra":
        return ExtraTreesClassifier(n_estimators=240, max_depth=5, min_samples_leaf=35, class_weight="balanced", random_state=int(seed), n_jobs=-1)
    raise RuntimeError(f"unknown model: {name}")


def _predict_oof(model_name: str, x: pd.DataFrame, y: np.ndarray, cash_mask: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    idx = np.flatnonzero(cash_mask)
    action = np.zeros(len(x), dtype=np.int64)
    conf = np.zeros(len(x), dtype=np.float64)
    folds = []
    n = len(idx)
    for train_frac, end_frac in ((0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.00)):
        train_end = int(n * train_frac)
        val_end = int(n * end_frac)
        if train_end < 100 or val_end <= train_end:
            continue
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        model = _make_model(model_name, seed + train_end)
        model.fit(x.iloc[train_idx].to_numpy(dtype=np.float64), y[train_idx])
        proba = model.predict_proba(x.iloc[val_idx].to_numpy(dtype=np.float64))
        classes = np.asarray(model.classes_, dtype=np.int64)
        best = np.argmax(proba, axis=1)
        action[val_idx] = classes[best]
        conf[val_idx] = proba[np.arange(len(val_idx)), best]
        folds.append({"train_rows": int(len(train_idx)), "val_rows": int(len(val_idx))})
    return action, conf, {"folds": folds, "oof_rows": int(np.count_nonzero(conf > 0.0))}


def _fit_predict(model_name: str, x_train: pd.DataFrame, y_train: np.ndarray, train_cash_mask: np.ndarray, x_eval: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(train_cash_mask)
    model = _make_model(model_name, seed)
    model.fit(x_train.iloc[idx].to_numpy(dtype=np.float64), y_train[idx])
    proba = model.predict_proba(x_eval.to_numpy(dtype=np.float64))
    classes = np.asarray(model.classes_, dtype=np.int64)
    best = np.argmax(proba, axis=1)
    return classes[best].astype(np.int64), proba[np.arange(len(x_eval)), best].astype(np.float64)


def _open_position(cash: float, arrays: dict[str, np.ndarray], i: int, side: int, sleeve: str, risk: FallbackRisk | None, row: pd.Series | None, fee_eff: float, slip_eff: float) -> tuple[float, Position, bool]:
    filled, entry_px, entry_fee, _route = omega._try_execution(arrays, int(i), int(side), entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return cash, Position(), False
    if sleeve == "primary":
        assert row is not None
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = abs(float(row.get("stop_loss", 0.0) or 0.0))
        leverage = float(row.get("leverage", 1.0) or 1.0)
        max_hold = int(row.get("max_hold_bars", 0) or 0)
    else:
        assert risk is not None
        notional = float(risk.notional)
        take_profit = float(risk.take_profit)
        stop_loss = abs(float(risk.stop_loss))
        leverage = float(risk.leverage)
        max_hold = int(risk.max_hold_bars)
    if notional <= 0.0:
        return cash, Position(), False
    entry_equity = cash
    cash -= cash * float(entry_fee) * notional
    return (
        cash,
        Position(sleeve=sleeve, side=int(side), entry_price=float(entry_px), entry_i=int(i), entry_equity=float(entry_equity), notional=notional, leverage=leverage, take_profit=take_profit, stop_loss=stop_loss, max_hold_bars=max_hold),
        True,
    )


def _close_position(cash: float, arrays: dict[str, np.ndarray], pos: Position, i: int, fee_eff: float, slip_eff: float) -> tuple[float, bool]:
    if pos.side == 0:
        return cash, False
    _ok, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), int(pos.side), entry=False, fee_base=fee_eff, slip_base=slip_eff)
    raw = (exit_px - pos.entry_price) / max(pos.entry_price, 1.0e-12) if pos.side > 0 else (pos.entry_price - exit_px) / max(pos.entry_price, 1.0e-12)
    before = cash
    cash = cash * (1.0 + raw * pos.notional)
    cash -= before * float(exit_fee) * pos.notional
    return cash, cash > pos.entry_equity


def _metrics_with_fallback(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    risk: FallbackRisk,
    fallback_action: np.ndarray,
    fallback_conf: np.ndarray,
    threshold: float,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> dict[str, Any]:
    arrays = _arrays(frame)
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = Position()
    trades = wins = 0
    primary_entries = fallback_entries = long_entries = short_entries = 0
    reasons: dict[str, int] = {}
    primary_takeovers = 0
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
                primary_takeovers += 1
            if reason:
                cash, win = _close_position(cash, arrays, pos, i, fee_eff, slip_eff)
                trades += 1
                wins += int(win)
                reasons[f"{pos.sleeve}_{reason}"] = reasons.get(f"{pos.sleeve}_{reason}", 0) + 1
                pos = Position()
            else:
                continue
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1.0e-12) - 1.0)
        if bool(active[i]):
            row = dec.iloc[int(i)]
            side = int(row.get("side", 0) or 0)
            if side != 0:
                cash, pos, entered = _open_position(cash, arrays, i, side, "primary", None, row, fee_eff, slip_eff)
                if entered:
                    primary_entries += 1
                    long_entries += int(side > 0)
                    short_entries += int(side < 0)
            continue
        action = int(fallback_action[int(i)]) if int(i) < len(fallback_action) else ACTION_CASH
        conf = float(fallback_conf[int(i)]) if int(i) < len(fallback_conf) else 0.0
        if action not in (ACTION_LONG, ACTION_SHORT) or conf < float(threshold):
            continue
        side = 1 if action == ACTION_LONG else -1
        cash, pos, entered = _open_position(cash, arrays, i, side, "fallback", risk, None, fee_eff, slip_eff)
        if entered:
            fallback_entries += 1
            long_entries += int(side > 0)
            short_entries += int(side < 0)
    if pos.side != 0:
        cash, win = _close_position(cash, arrays, pos, len(frame) - 1, fee_eff, slip_eff)
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


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_primary_entries": int(metrics["primary_entries"]),
        f"{prefix}_fallback_entries": int(metrics["fallback_entries"]),
        f"{prefix}_primary_takeovers": int(metrics["primary_takeovers"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "load_frames"}, ensure_ascii=False), flush=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_src, val_dec0, val_prefix = base._build_split(frames, "validation")
    oos_frame, oos_src, oos_dec0, oos_prefix = base._build_split(frames, "oos")
    val_dec = _apply_aggressive(val_dec0)
    oos_dec = _apply_aggressive(oos_dec0)
    print(json.dumps({"stage": "features"}, ensure_ascii=False), flush=True)
    val_features = _extra_features(base._feature_frame(val_frame, val_src, val_dec0, val_prefix), val_dec)
    oos_features = _extra_features(base._feature_frame(oos_frame, oos_src, oos_dec0, oos_prefix), oos_dec)
    val_cash = ~omega._active(val_dec)
    oos_cash = ~omega._active(oos_dec)
    baseline_val = omega._metrics(val_frame, val_dec, fee=fee, slip=slip, cost_mult=3.0)
    baseline_oos = omega._metrics(oos_frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0)
    rows: list[dict[str, Any]] = []
    rows.append({"model": "aggressive_primary_only", "risk": "none", "min_edge": 0.0, "threshold": 1.0, **_metric_row("val", {**baseline_val, "primary_entries": baseline_val["long_entries"] + baseline_val["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}), **_metric_row("oos", {**baseline_oos, "primary_entries": baseline_oos["long_entries"] + baseline_oos["short_entries"], "fallback_entries": 0, "primary_takeovers": 0})})
    diagnostics: dict[str, Any] = {"val_cash_rows": int(np.count_nonzero(val_cash)), "oos_cash_rows": int(np.count_nonzero(oos_cash)), "feature_count": int(val_features.shape[1]), "features": list(val_features.columns)}
    for risk in RISKS:
        for min_edge in (0.002, 0.004, 0.006):
            print(json.dumps({"stage": "labels", "risk": risk.name, "min_edge": min_edge}, ensure_ascii=False), flush=True)
            y_val, label_diag = _build_labels(val_frame, val_dec, risk, min_edge)
            diagnostics[f"{risk.name}_edge{min_edge}"] = label_diag
            if len(set(y_val[val_cash].tolist())) < 2:
                continue
            for model_name in ("hgb", "extra"):
                print(json.dumps({"stage": "fit_eval", "risk": risk.name, "min_edge": min_edge, "model": model_name}, ensure_ascii=False), flush=True)
                val_action, val_conf, oof_diag = _predict_oof(model_name, val_features, y_val, val_cash, seed=260606)
                oos_action, oos_conf = _fit_predict(model_name, val_features, y_val, val_cash, oos_features, seed=260606)
                diagnostics[f"{risk.name}_edge{min_edge}_{model_name}_oof"] = oof_diag
                for threshold in (0.45, 0.55, 0.65, 0.75, 0.85):
                    val_m = _metrics_with_fallback(val_frame, val_dec, risk, val_action, val_conf, threshold, fee=fee, slip=slip, cost_mult=3.0)
                    oos_m = _metrics_with_fallback(oos_frame, oos_dec, risk, oos_action, oos_conf, threshold, fee=fee, slip=slip, cost_mult=3.0)
                    row = {"model": model_name, "risk": risk.name, "min_edge": float(min_edge), "threshold": float(threshold)}
                    row.update(_metric_row("val", val_m))
                    row.update(_metric_row("oos", oos_m))
                    rows.append(row)
    ranking = pd.DataFrame(rows)
    ranking["val_delta_pnl"] = ranking["val_pnl"] - AGGRESSIVE_VAL["pnl"]
    ranking["oos_delta_pnl"] = ranking["oos_pnl"] - AGGRESSIVE_OOS["pnl"]
    ranking["val_delta_mdd"] = ranking["val_mdd"] - AGGRESSIVE_VAL["mdd"]
    ranking["oos_delta_mdd"] = ranking["oos_mdd"] - AGGRESSIVE_OOS["mdd"]
    ranking["score"] = ranking["oos_pnl"] + 0.75 * ranking["val_pnl"] + 0.35 * ranking["oos_mdd"] + 0.35 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "cash_fallback_sleeve_ranking.csv", index=False)
    promotable = ranking[
        (ranking["model"] != "aggressive_primary_only")
        & (ranking["oos_pnl"] > AGGRESSIVE_OOS["pnl"])
        & (ranking["val_pnl"] > AGGRESSIVE_VAL["pnl"])
        & (ranking["oos_mdd"] >= AGGRESSIVE_OOS["mdd"] * 1.35)
        & (ranking["val_mdd"] >= AGGRESSIVE_VAL["mdd"] * 1.35)
    ].copy()
    promotable.to_csv(OUT_DIR / "cash_fallback_sleeve_promotable.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "baseline": {"model_id": "omega1_2_1_aggressive_compensated_scale200_cap090", "validation": AGGRESSIVE_VAL, "oos": AGGRESSIVE_OOS},
        "method": "Primary aggressive_200_cap090 is preserved. Fallback sleeve is called only when primary is CASH and no position is open. If primary signal appears while fallback is open, fallback closes via primary_takeover and primary gets priority.",
        "diagnostics": diagnostics,
        "best": ranking.iloc[0].to_dict(),
        "promotable_count": int(len(promotable)),
        "top10": ranking.head(10).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "cash_fallback_sleeve_ranking.csv"),
            "promotable": str(OUT_DIR / "cash_fallback_sleeve_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "best": report["best"], "promotable_count": int(len(promotable))}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
