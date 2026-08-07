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

import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as exposure  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_aggressive_risk_layer_priority_20260607"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

ACTION_KEEP = 0
ACTION_VETO = 1
ACTION_SHRINK70 = 2
ACTION_SHRINK85 = 3
ACTION_BOOST105 = 4
ACTION_BOOST110 = 5
ACTION_BOOST111 = 6

ACTION_NAMES = {
    ACTION_KEEP: "keep",
    ACTION_VETO: "veto",
    ACTION_SHRINK70: "shrink70",
    ACTION_SHRINK85: "shrink85",
    ACTION_BOOST105: "boost105",
    ACTION_BOOST110: "boost110",
    ACTION_BOOST111: "boost111",
}
ACTION_SCALE = {
    ACTION_KEEP: 1.00,
    ACTION_VETO: 0.00,
    ACTION_SHRINK70: 0.70,
    ACTION_SHRINK85: 0.85,
    ACTION_BOOST105: 1.05,
    ACTION_BOOST110: 1.10,
    ACTION_BOOST111: 1.1111111111,
}


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


@dataclass(frozen=True)
class SplitData:
    frame: pd.DataFrame
    dec: pd.DataFrame
    feat: pd.DataFrame
    active_idx: np.ndarray


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_avg_notional": float(metrics.get("avg_notional", 0.0)),
        f"{prefix}_avg_leverage": float(metrics.get("avg_leverage", 0.0)),
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def _build_split(frames: dict[str, pd.DataFrame], split: str) -> SplitData:
    frame, src, dec, prefix = exposure._build_split(frames, split)
    dec = sleeve._apply_aggressive(dec)
    feat = exposure._feature_frame(frame, src, dec, prefix)
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.float64)
    long_edge = pd.to_numeric(feat["dir_p_long"], errors="raise").to_numpy(dtype=np.float64) - pd.to_numeric(feat["dir_p_short"], errors="raise").to_numpy(dtype=np.float64)
    feat["side_aligned_dir_edge"] = np.where(side > 0, long_edge, np.where(side < 0, -long_edge, 0.0))
    feat["side_quality_prob"] = np.where(
        side > 0,
        pd.to_numeric(feat["quality_p_long"], errors="raise").to_numpy(dtype=np.float64),
        np.where(side < 0, pd.to_numeric(feat["quality_p_short"], errors="raise").to_numpy(dtype=np.float64), pd.to_numeric(feat["quality_p_cash"], errors="raise").to_numpy(dtype=np.float64)),
    )
    feat["notional_tp"] = pd.to_numeric(dec["notional_exposure"], errors="raise").to_numpy(dtype=np.float64) * pd.to_numeric(dec["take_profit"], errors="raise").to_numpy(dtype=np.float64)
    feat["notional_sl"] = pd.to_numeric(dec["notional_exposure"], errors="raise").to_numpy(dtype=np.float64) * pd.to_numeric(dec["stop_loss"], errors="raise").to_numpy(dtype=np.float64)
    active_idx = np.flatnonzero(omega._active(dec))
    bad = [c for c in feat.columns if c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_") or c == "tp_sl_action_score"]
    if bad:
        raise RuntimeError(f"{split}: forbidden feature columns: {bad[:40]}")
    return SplitData(frame=frame, dec=dec, feat=feat, active_idx=active_idx)


def _vol_multiplier(feat: pd.DataFrame, idx: np.ndarray, ref_atr: float) -> np.ndarray:
    atr = pd.to_numeric(feat.iloc[idx]["atr14_pct"], errors="raise").to_numpy(dtype=np.float64)
    return np.clip(atr / max(float(ref_atr), 1.0e-12), 0.85, 1.15)


def _transform_dec(
    dec: pd.DataFrame,
    idx: np.ndarray,
    action_ids: np.ndarray,
    *,
    risk_mode: str,
    feat: pd.DataFrame,
    ref_atr: float,
    cap: float,
) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    idx = np.asarray(idx, dtype=np.int64)
    action_ids = np.asarray(action_ids, dtype=np.int64)
    if len(idx) != len(action_ids):
        raise RuntimeError("idx/action length mismatch")
    for action_id in sorted(set(int(x) for x in action_ids.tolist())):
        rows = idx[action_ids == action_id]
        if len(rows) == 0 or action_id == ACTION_KEEP:
            continue
        if action_id == ACTION_VETO:
            out.loc[rows, "action"] = 0
            out.loc[rows, "side"] = 0
            out.loc[rows, "position_fraction"] = 0.0
            out.loc[rows, "notional_exposure"] = 0.0
            continue
        scale = float(ACTION_SCALE[action_id])
        base = pd.to_numeric(out.loc[rows, "notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
        new = np.minimum(base * scale, float(cap))
        ratio = new / np.maximum(base, 1.0e-12)
        out.loc[rows, "notional_exposure"] = new
        out.loc[rows, "position_fraction"] = new
        if risk_mode == "exposure_only":
            continue
        tp = pd.to_numeric(out.loc[rows, "take_profit"], errors="raise").to_numpy(dtype=np.float64)
        sl = pd.to_numeric(out.loc[rows, "stop_loss"], errors="raise").to_numpy(dtype=np.float64)
        if risk_mode == "coupled":
            adj = ratio
        elif risk_mode == "coupled_atr":
            adj = ratio * _vol_multiplier(feat, rows, ref_atr)
        else:
            raise RuntimeError(f"unknown risk_mode: {risk_mode}")
        out.loc[rows, "take_profit"] = tp * adj
        out.loc[rows, "stop_loss"] = sl * adj
    return out


def _simulate_rewards(
    data: SplitData,
    *,
    risk_mode: str,
    fee: float,
    slip: float,
    ref_atr: float,
    cap: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    arrays = {c: pd.to_numeric(data.frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active_idx = data.active_idx
    actions = np.array(list(ACTION_NAMES.keys()), dtype=np.int64)
    rewards = np.zeros((len(active_idx), len(actions)), dtype=np.float64)
    reasons: dict[str, int] = {}
    for row_i, idx in enumerate(active_idx):
        for col_i, action_id in enumerate(actions):
            if action_id == ACTION_VETO:
                rewards[row_i, col_i] = 0.0
                continue
            dec_one = _transform_dec(data.dec, np.array([idx]), np.array([action_id]), risk_mode=risk_mode, feat=data.feat, ref_atr=ref_atr, cap=cap)
            score, meta = omega._simulate_trade(data.frame, arrays, int(idx), dec_one.iloc[int(idx)], fee=fee, slip=slip, cost_mult=3.0)
            # Prefer net PnL, but keep a small penalty for fragile adverse paths.
            net = float(meta.get("net", score))
            mae = float(meta.get("mae", 0.0))
            exposure = float(dec_one.iloc[int(idx)].get("notional_exposure", 0.0) or 0.0) * float(dec_one.iloc[int(idx)].get("leverage", 1.0) or 1.0)
            rewards[row_i, col_i] = net - 0.10 * max(0.0, -mae - 0.018) * exposure
            reasons[str(meta.get("exit_reason", "inactive"))] = reasons.get(str(meta.get("exit_reason", "inactive")), 0) + 1
    labels = actions[np.argmax(rewards, axis=1)]
    counts = {ACTION_NAMES[int(k)]: int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()}
    return rewards, {"label_counts": counts, "sim_reasons": reasons, "rows": int(len(active_idx)), "actions": [ACTION_NAMES[int(a)] for a in actions]}


def _make_model(name: str, seed: int):
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.035, max_leaf_nodes=7, l2_regularization=2.0, random_state=int(seed))
    if name == "extra":
        return ExtraTreesClassifier(n_estimators=300, max_depth=5, min_samples_leaf=30, class_weight="balanced", random_state=int(seed), n_jobs=-1)
    raise RuntimeError(f"unknown model: {name}")


def _predict(model: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=np.int64)
    best = np.argmax(proba, axis=1)
    return classes[best].astype(np.int64), proba[np.arange(len(x)), best].astype(np.float64)


def _oof_predict(model_name: str, x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pred = np.full(len(y), ACTION_KEEP, dtype=np.int64)
    conf = np.zeros(len(y), dtype=np.float64)
    folds: list[dict[str, int]] = []
    n = len(y)
    for train_frac, end_frac in ((0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.00)):
        train_end = int(n * train_frac)
        val_end = int(n * end_frac)
        if train_end < 100 or val_end <= train_end:
            continue
        model = _make_model(model_name, seed + train_end)
        model.fit(x[:train_end], y[:train_end])
        pred[train_end:val_end], conf[train_end:val_end] = _predict(model, x[train_end:val_end])
        folds.append({"train_rows": train_end, "val_rows": val_end - train_end})
    return pred, conf, {"folds": folds, "oof_rows": int(np.count_nonzero(conf > 0.0))}


def _apply_confidence(pred: np.ndarray, conf: np.ndarray, threshold: float) -> np.ndarray:
    out = pred.copy()
    out[conf < float(threshold)] = ACTION_KEEP
    return out


def _fit_full_predict(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    model = _make_model(model_name, seed)
    model.fit(x_train, y_train)
    return _predict(model, x_eval)


def _metrics_giveback(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    activation: float,
    giveback_frac: float,
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
    entry_idx = 0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    max_hold = 0
    cooldown = 0
    next_cooldown = 0
    mfe = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    reasons: dict[str, int] = {}
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1.0e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            eq = cash * (1.0 + unreal)
        else:
            unreal = 0.0
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)
        if pos != 0:
            hold = int(i) - int(entry_idx)
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            elif mfe >= float(activation) and mfe > 0.0 and (mfe - unreal) / max(mfe, 1.0e-12) >= float(giveback_frac):
                reason = "giveback_defense"
            elif max_hold > 0 and hold >= max_hold:
                reason = "max_hold"
            if reason:
                filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * float(exit_fee) * notional
                trades += 1
                wins += int(cash > entry_equity)
                reasons[reason] = reasons.get(reason, 0) + 1
                pos = 0
                cooldown = int(next_cooldown)
                next_cooldown = 0
                mfe = 0.0
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
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
        entry_equity = float(cash)
        entry_idx = int(i)
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        leverage = float(row.get("leverage", 1.0) or 1.0)
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
        max_hold = int(row.get("max_hold_bars", 0) or 0)
        next_cooldown = int(row.get("cooldown_bars", 0) or 0)
        mfe = 0.0
        cash -= cash * float(entry_fee) * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
    if pos != 0:
        fill_i = len(frame) - 1
        exit_px = omega._fill_price(arrays, fill_i, pos, slip_eff, entry=False)
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
        "trades_per_day": float(trades / max((pd.to_datetime(frame["timestamp"].iloc[-1]) - pd.to_datetime(frame["timestamp"].iloc[0])).total_seconds() / 86400.0, 1.0e-9)),
        "avg_notional": float(notional_sum / n_entries),
        "avg_leverage": float(leverage_sum / n_entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
    }


def _evaluate_decisions(
    val: SplitData,
    oos: SplitData,
    val_dec: pd.DataFrame,
    oos_dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    tag: str,
    row_extra: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    val_m = omega._metrics(val.frame, val_dec, fee=fee, slip=slip, cost_mult=3.0)
    oos_m = omega._metrics(oos.frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0)
    rows.append({"candidate": tag, "giveback_activation": None, "giveback_frac": None, **row_extra, **_metric_row("val", val_m), **_metric_row("oos", oos_m)})
    for activation in (0.012, 0.018, 0.026, 0.034):
        for frac in (0.45, 0.60, 0.75):
            val_g = _metrics_giveback(val.frame, val_dec, fee=fee, slip=slip, cost_mult=3.0, activation=activation, giveback_frac=frac)
            oos_g = _metrics_giveback(oos.frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0, activation=activation, giveback_frac=frac)
            rows.append({"candidate": f"{tag}_giveback_a{activation:.3f}_f{frac:.2f}", "giveback_activation": float(activation), "giveback_frac": float(frac), **row_extra, **_metric_row("val", val_g), **_metric_row("oos", oos_g)})
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val = _build_split(frames, "validation")
    oos = _build_split(frames, "oos")
    ref_atr = float(pd.to_numeric(val.feat.iloc[val.active_idx]["atr14_pct"], errors="raise").median())

    feature_cols = [
        "bar_range_pct",
        "body_pct",
        "atr14_pct",
        "ret_1",
        "ret_3",
        "ret_6",
        "ret_12",
        "ret_24",
        "ret_vol_12",
        "ret_vol_24",
        "ret_vol_48",
        "range_mean_12",
        "range_mean_24",
        "ema9_21_gap",
        "router_confidence",
        "router_margin",
        "dir_confidence",
        "dir_trade_prob",
        "quality_for_action",
        "side_aligned_dir_edge",
        "side_quality_prob",
        "side",
        "base_notional",
        "base_tp",
        "base_sl",
        "notional_tp",
        "notional_sl",
    ]
    missing = [c for c in feature_cols if c not in val.feat.columns or c not in oos.feat.columns]
    if missing:
        raise RuntimeError(f"missing risk layer features: {missing}")
    x_val = val.feat.iloc[val.active_idx][feature_cols].to_numpy(dtype=np.float64)
    x_oos = oos.feat.iloc[oos.active_idx][feature_cols].to_numpy(dtype=np.float64)

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "feature_cols": feature_cols,
        "val_active_rows": int(len(val.active_idx)),
        "oos_active_rows": int(len(oos.active_idx)),
        "ref_atr14_pct": ref_atr,
    }
    rows.extend(_evaluate_decisions(val, oos, val.dec, oos.dec, fee=fee, slip=slip, tag="baseline_aggressive", row_extra={"model_name": "none", "risk_mode": "baseline", "selector_threshold": None, "cap": None, "val_changed_rows": 0, "oos_changed_rows": 0}))

    for risk_mode in ("exposure_only", "coupled", "coupled_atr"):
        rewards, diag = _simulate_rewards(val, risk_mode=risk_mode, fee=fee, slip=slip, ref_atr=ref_atr, cap=0.90)
        actions = np.array(list(ACTION_NAMES.keys()), dtype=np.int64)
        y = actions[np.argmax(rewards, axis=1)]
        diagnostics[f"labels_{risk_mode}"] = diag
        for model_name in ("hgb", "extra"):
            pred_oof, conf_oof, oof_diag = _oof_predict(model_name, x_val, y, seed=260607)
            pred_oos, conf_oos = _fit_full_predict(model_name, x_val, y, x_oos, seed=260607)
            diagnostics[f"{risk_mode}_{model_name}"] = oof_diag
            for threshold in (0.0, 0.45, 0.55, 0.65):
                val_actions = _apply_confidence(pred_oof, conf_oof, threshold)
                oos_actions = _apply_confidence(pred_oos, conf_oos, threshold)
                val_dec = _transform_dec(val.dec, val.active_idx, val_actions, risk_mode=risk_mode, feat=val.feat, ref_atr=ref_atr, cap=0.90)
                oos_dec = _transform_dec(oos.dec, oos.active_idx, oos_actions, risk_mode=risk_mode, feat=oos.feat, ref_atr=ref_atr, cap=0.90)
                val_changed = int(np.count_nonzero(val_actions != ACTION_KEEP))
                oos_changed = int(np.count_nonzero(oos_actions != ACTION_KEEP))
                tag = f"learned_{model_name}_{risk_mode}_thr{threshold:.2f}"
                rows.extend(
                    _evaluate_decisions(
                        val,
                        oos,
                        val_dec,
                        oos_dec,
                        fee=fee,
                        slip=slip,
                        tag=tag,
                        row_extra={"model_name": model_name, "risk_mode": risk_mode, "selector_threshold": float(threshold), "cap": 0.90, "val_changed_rows": val_changed, "oos_changed_rows": oos_changed},
                    )
                )

    ranking = pd.DataFrame(rows)
    ranking["score"] = ranking["oos_pnl"] + 0.50 * ranking["val_pnl"] + 0.35 * ranking["oos_mdd"] + 0.25 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "risk_layer_priority_ranking.csv", index=False)
    strict = ranking[(ranking["val_pnl"] > 100.54272942091158) & (ranking["oos_pnl"] > 72.76004148106665) & (ranking["val_mdd"] >= -12.0) & (ranking["oos_mdd"] >= -9.0)].copy()
    strict = strict.sort_values(["oos_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    strict.to_csv(OUT_DIR / "risk_layer_priority_strict_candidates.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "baseline_id": "omega1_2_1_aggressive_compensated_scale200_cap090",
        "method": "Priority risk-layer sweep: OOF learned exposure selector, coupled TP/SL and ATR-coupled variants, plus giveback defense. Direction/quality/lifecycle parent is frozen.",
        "diagnostics": diagnostics,
        "top20": ranking.head(20).to_dict(orient="records"),
        "strict_top20": strict.head(20).to_dict(orient="records"),
        "artifacts": {
            "ranking": str(OUT_DIR / "risk_layer_priority_ranking.csv"),
            "strict": str(OUT_DIR / "risk_layer_priority_strict_candidates.csv"),
            "report": str(OUT_DIR / "risk_layer_priority_report.json"),
        },
    }
    (OUT_DIR / "risk_layer_priority_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": report["artifacts"]["report"], "top": report["top20"][:5], "strict_top": report["strict_top20"][:5]}, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
