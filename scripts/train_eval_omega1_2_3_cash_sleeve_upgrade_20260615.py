#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_2_tp_runner_cash_sleeve_20260615 as base  # noqa: E402


MODEL_ID = "omega1_2_3_cash_sleeve_upgrade_20260615"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASELINE_ID = base.BASELINE_ID


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


def _enhanced_features(payload: dict[str, Any]) -> pd.DataFrame:
    out = base._features(payload)
    frame = payload["frame"].reset_index(drop=True)
    dec = payload["dec"].reset_index(drop=True)
    active = base._active(dec)
    close = pd.to_numeric(frame["close"], errors="raise").reset_index(drop=True)
    high = pd.to_numeric(frame["high"], errors="raise").reset_index(drop=True)
    low = pd.to_numeric(frame["low"], errors="raise").reset_index(drop=True)
    ret1 = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rng = ((high - low) / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["cash_ret_sum_12"] = ret1.rolling(12, min_periods=1).sum().to_numpy(dtype=np.float64)
    out["cash_ret_sum_48"] = ret1.rolling(48, min_periods=1).sum().to_numpy(dtype=np.float64)
    out["cash_ret_vol_12"] = ret1.rolling(12, min_periods=2).std().fillna(0.0).to_numpy(dtype=np.float64)
    out["cash_ret_vol_48"] = ret1.rolling(48, min_periods=2).std().fillna(0.0).to_numpy(dtype=np.float64)
    out["cash_range_ratio_12_48"] = (
        rng.rolling(12, min_periods=1).mean() / rng.rolling(48, min_periods=1).mean().replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0).to_numpy(dtype=np.float64)

    probs = out[["tabm_dir_p_cash", "tabm_dir_p_long", "tabm_dir_p_short"]].clip(1e-9, 1.0)
    out["tabm_dir_entropy"] = (-(probs * np.log(probs)).sum(axis=1) / np.log(3.0)).to_numpy(dtype=np.float64)
    out["tabm_long_short_gap"] = (out["tabm_dir_p_long"] - out["tabm_dir_p_short"]).to_numpy(dtype=np.float64)
    out["tabm_abs_side_gap"] = np.abs(out["tabm_long_short_gap"]).to_numpy(dtype=np.float64)
    out["tabm_quality_side_gap"] = (out["tabm_quality_p_long"] - out["tabm_quality_p_short"]).to_numpy(dtype=np.float64)
    out["tabm_quality_abs_gap"] = np.abs(out["tabm_quality_side_gap"]).to_numpy(dtype=np.float64)

    since_exit = np.zeros(len(out), dtype=np.float64)
    last_active_len = np.zeros(len(out), dtype=np.float64)
    last_side = np.zeros(len(out), dtype=np.float64)
    cur_cash = 0
    cur_active = 0
    prev_active_len = 0
    prev_side = 0
    sides = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    for i, is_active in enumerate(active):
        if bool(is_active):
            cur_active += 1
            cur_cash = 0
            prev_side = int(sides[i])
        else:
            if i > 0 and bool(active[i - 1]):
                prev_active_len = cur_active
                cur_active = 0
            cur_cash += 1
        since_exit[i] = cur_cash
        last_active_len[i] = prev_active_len
        last_side[i] = prev_side
    out["time_since_primary_exit"] = np.tanh(since_exit / 144.0)
    out["last_primary_active_len"] = np.tanh(last_active_len / 288.0)
    out["last_primary_side"] = last_side

    bad = [c for c in out.columns if c in base.FORBIDDEN_FEATURE_EXACT or c.startswith(base.FORBIDDEN_FEATURE_PREFIXES)]
    if bad:
        raise RuntimeError(f"forbidden sleeve feature columns: {bad[:20]}")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _simulate_label_detail(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    active: np.ndarray,
    i: int,
    side: int,
    risk: base.SleeveRisk,
    fee_eff: float,
    slip_eff: float,
) -> dict[str, Any]:
    fill_i = min(int(i) + 1, len(frame) - 1)
    entry_px = base._exec_price(float(arrays["open"][fill_i]), int(side), slip_eff, entry=True)
    pos = base.Position(
        sleeve="label",
        side=int(side),
        entry_signal_i=int(i),
        entry_i=int(fill_i),
        entry_price=entry_px,
        entry_equity=1.0,
        notional=float(risk.notional),
        margin_notional=float(risk.notional),
        leverage=float(risk.leverage),
        take_profit=float(risk.take_profit),
        stop_loss=abs(float(risk.stop_loss)),
        floor_unreal=-abs(float(risk.stop_loss)),
        max_hold_bars=int(risk.max_hold_bars),
    )
    cash = 1.0 - float(fee_eff) * float(risk.notional)
    end_i = min(len(frame) - 2, fill_i + int(risk.max_hold_bars))
    target = 0.0
    mfe = 0.0
    mae = 0.0
    reason = "max_hold"
    takeover_bars = int(risk.max_hold_bars) + 1
    for j in range(fill_i, end_i + 1):
        best, worst = base._bar_best_worst(arrays, pos, j, slip_eff)
        close_unreal = base._close_unreal(arrays, pos, j, slip_eff)
        mfe = max(mfe, best, close_unreal)
        mae = min(mae, worst, close_unreal)
        target = close_unreal
        if bool(active[j]):
            target = close_unreal
            reason = "primary_takeover"
            takeover_bars = int(j) - int(fill_i)
            break
        if worst <= -abs(float(risk.stop_loss)):
            target = -abs(float(risk.stop_loss))
            reason = "stop_loss"
            break
        if best >= float(risk.take_profit):
            target = float(risk.take_profit)
            reason = "take_profit"
            break
    exit_px = base._exit_price_from_unreal(pos, target)
    cash, _net_pct = base._runtime_close(cash, pos, exit_px, fee_eff)
    return {
        "net": float(cash - 1.0),
        "stop": int(reason == "stop_loss"),
        "takeover": int(reason == "primary_takeover"),
        "mfe": float(mfe),
        "mae": float(mae),
        "bars_to_takeover": int(takeover_bars),
        "reason": reason,
    }


def _label_table(payload: dict[str, Any], risk: base.SleeveRisk, min_edge: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = payload["frame"].reset_index(drop=True)
    dec = payload["dec"].reset_index(drop=True)
    arrays = base.repair._arrays(frame)
    active = base._active(dec)
    cash_mask = ~active
    fee_eff = float(payload["fee"]) * 3.0
    slip_eff = float(payload["slip"]) * 3.0
    rows: list[dict[str, Any]] = []
    for i in np.flatnonzero(cash_mask):
        if i >= len(frame) - int(risk.max_hold_bars) - 3:
            continue
        long_d = _simulate_label_detail(frame, arrays, active, int(i), 1, risk, fee_eff, slip_eff)
        short_d = _simulate_label_detail(frame, arrays, active, int(i), -1, risk, fee_eff, slip_eff)
        best_action = base.ACTION_LONG if long_d["net"] >= short_d["net"] else base.ACTION_SHORT
        best = long_d if best_action == base.ACTION_LONG else short_d
        rows.append(
            {
                "i": int(i),
                "long_net": float(long_d["net"]),
                "short_net": float(short_d["net"]),
                "long_stop": int(long_d["stop"]),
                "short_stop": int(short_d["stop"]),
                "long_takeover": int(long_d["takeover"]),
                "short_takeover": int(short_d["takeover"]),
                "best_net": float(best["net"]),
                "best_action": int(best_action),
                "cls_label": int(best_action) if float(best["net"]) > float(min_edge) else base.ACTION_CASH,
                "opp_label": int(float(best["net"]) > float(min_edge)),
                "best_stop": int(best["stop"]),
                "best_takeover": int(best["takeover"]),
            }
        )
    labels = pd.DataFrame(rows)
    diag = {
        "valid_cash_rows": int(len(labels)),
        "class_counts": {str(k): int(v) for k, v in labels["cls_label"].value_counts().sort_index().items()} if len(labels) else {},
        "best_net_mean": float(labels["best_net"].mean()) if len(labels) else 0.0,
        "best_stop_rate": float(labels["best_stop"].mean()) if len(labels) else 0.0,
        "best_takeover_rate": float(labels["best_takeover"].mean()) if len(labels) else 0.0,
    }
    return labels, diag


def _model(kind: str, model_type: str, seed: int):
    if kind == "hgb" and model_type == "classifier":
        return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.035, max_leaf_nodes=9, l2_regularization=2.0, random_state=int(seed))
    if kind == "extra" and model_type == "classifier":
        return ExtraTreesClassifier(n_estimators=320, max_depth=7, min_samples_leaf=28, class_weight="balanced", random_state=int(seed), n_jobs=-1)
    if kind == "hgb" and model_type == "regressor":
        return HistGradientBoostingRegressor(max_iter=140, learning_rate=0.035, max_leaf_nodes=9, l2_regularization=2.0, random_state=int(seed))
    if kind == "extra" and model_type == "regressor":
        return ExtraTreesRegressor(n_estimators=320, max_depth=7, min_samples_leaf=28, random_state=int(seed), n_jobs=-1)
    raise RuntimeError(f"unknown model: {kind}/{model_type}")


def _chron_folds(idx: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = []
    n = len(idx)
    for train_frac, end_frac in ((0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.00)):
        train_end = int(n * train_frac)
        val_end = int(n * end_frac)
        if train_end >= 100 and val_end > train_end:
            folds.append((idx[:train_end], idx[train_end:val_end]))
    return folds


def _fit_predict_classifier(kind: str, x_val: pd.DataFrame, y: np.ndarray, idx: np.ndarray, x_oos: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    val_action = np.zeros(len(x_val), dtype=np.int64)
    val_conf = np.zeros(len(x_val), dtype=np.float64)
    folds_meta = []
    for fold_id, (tr, va) in enumerate(_chron_folds(idx)):
        if len(np.unique(y[tr])) < 2:
            folds_meta.append({"fold": int(fold_id), "skipped": "single_class"})
            continue
        m = _model(kind, "classifier", seed + fold_id)
        m.fit(x_val.iloc[tr].to_numpy(dtype=np.float64), y[tr])
        proba = m.predict_proba(x_val.iloc[va].to_numpy(dtype=np.float64))
        classes = np.asarray(m.classes_, dtype=np.int64)
        best = np.argmax(proba, axis=1)
        val_action[va] = classes[best]
        val_conf[va] = proba[np.arange(len(va)), best]
        folds_meta.append({"fold": int(fold_id), "train_rows": int(len(tr)), "val_rows": int(len(va)), "classes": classes.tolist()})
    m = _model(kind, "classifier", seed)
    m.fit(x_val.iloc[idx].to_numpy(dtype=np.float64), y[idx])
    proba = m.predict_proba(x_oos.to_numpy(dtype=np.float64))
    classes = np.asarray(m.classes_, dtype=np.int64)
    best = np.argmax(proba, axis=1)
    return val_action, val_conf, classes[best].astype(np.int64), proba[np.arange(len(x_oos)), best].astype(np.float64), {"folds": folds_meta}


def _fit_predict_binary(kind: str, x_val: pd.DataFrame, y: np.ndarray, idx: np.ndarray, x_oos: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    val_prob = np.zeros(len(x_val), dtype=np.float64)
    folds_meta = []
    for fold_id, (tr, va) in enumerate(_chron_folds(idx)):
        if len(np.unique(y[tr])) < 2:
            folds_meta.append({"fold": int(fold_id), "skipped": "single_class"})
            continue
        m = _model(kind, "classifier", seed + fold_id)
        m.fit(x_val.iloc[tr].to_numpy(dtype=np.float64), y[tr])
        proba = m.predict_proba(x_val.iloc[va].to_numpy(dtype=np.float64))
        classes = np.asarray(m.classes_, dtype=np.int64)
        one_col = np.flatnonzero(classes == 1)
        val_prob[va] = proba[:, int(one_col[0])] if len(one_col) else 0.0
        folds_meta.append({"fold": int(fold_id), "train_rows": int(len(tr)), "val_rows": int(len(va)), "classes": classes.tolist()})
    m = _model(kind, "classifier", seed)
    m.fit(x_val.iloc[idx].to_numpy(dtype=np.float64), y[idx])
    proba = m.predict_proba(x_oos.to_numpy(dtype=np.float64))
    classes = np.asarray(m.classes_, dtype=np.int64)
    one_col = np.flatnonzero(classes == 1)
    oos_prob = proba[:, int(one_col[0])] if len(one_col) else np.zeros(len(x_oos), dtype=np.float64)
    return val_prob, oos_prob.astype(np.float64), {"folds": folds_meta}


def _fit_predict_regressor(kind: str, x_val: pd.DataFrame, y: np.ndarray, idx: np.ndarray, x_oos: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    val_pred = np.zeros(len(x_val), dtype=np.float64)
    folds_meta = []
    for fold_id, (tr, va) in enumerate(_chron_folds(idx)):
        m = _model(kind, "regressor", seed + fold_id)
        m.fit(x_val.iloc[tr].to_numpy(dtype=np.float64), y[tr])
        val_pred[va] = m.predict(x_val.iloc[va].to_numpy(dtype=np.float64))
        folds_meta.append({"fold": int(fold_id), "train_rows": int(len(tr)), "val_rows": int(len(va))})
    m = _model(kind, "regressor", seed)
    m.fit(x_val.iloc[idx].to_numpy(dtype=np.float64), y[idx])
    return val_pred, m.predict(x_oos.to_numpy(dtype=np.float64)).astype(np.float64), {"folds": folds_meta}


def _fallback_only(ledger: pd.DataFrame) -> dict[str, Any]:
    fb = ledger[ledger["sleeve"] == "fallback"].copy() if len(ledger) else ledger.copy()
    rets = fb["net_trade_return_pct"].astype(float).to_numpy() if len(fb) else np.asarray([], dtype=np.float64)
    eq = [1.0]
    for ret in rets:
        eq.append(eq[-1] * (1.0 + float(ret) / 100.0))
    eq_arr = np.asarray(eq, dtype=np.float64)
    dd = (eq_arr / np.maximum(np.maximum.accumulate(eq_arr), 1e-12) - 1.0) * 100.0
    wins = rets[rets > 0.0]
    losses = rets[rets <= 0.0]
    return {
        "fallback_trades": int(len(rets)),
        "fallback_pnl": float((eq_arr[-1] - 1.0) * 100.0),
        "fallback_mdd": float(dd.min()) if len(dd) else 0.0,
        "fallback_wr": float(np.mean(rets > 0.0)) if len(rets) else 0.0,
        "fallback_avg_trade": float(np.mean(rets)) if len(rets) else 0.0,
        "fallback_profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 1e-12 else None,
        "fallback_stop_rate": float(np.mean(fb["exit_reason"].eq("fallback_stop_loss"))) if len(fb) else 0.0,
        "fallback_take_profit": int(fb["exit_reason"].eq("fallback_take_profit").sum()) if len(fb) else 0,
        "fallback_stop_loss": int(fb["exit_reason"].eq("fallback_stop_loss").sum()) if len(fb) else 0,
        "fallback_primary_takeover": int(fb["exit_reason"].eq("fallback_primary_takeover").sum()) if len(fb) else 0,
        "fallback_max_hold": int(fb["exit_reason"].eq("fallback_max_hold").sum()) if len(fb) else 0,
    }


def _row(prefix: str, combo: dict[str, Any], ledger: pd.DataFrame) -> dict[str, Any]:
    out = base._row(prefix, combo)
    fb = _fallback_only(ledger)
    out.update({f"{prefix}_{k}": v for k, v in fb.items()})
    return out


def _actions_from_ev(long_ev: np.ndarray, short_ev: np.ndarray, ev_min: float) -> tuple[np.ndarray, np.ndarray]:
    best_long = long_ev >= short_ev
    best_ev = np.where(best_long, long_ev, short_ev)
    action = np.where(best_ev > float(ev_min), np.where(best_long, base.ACTION_LONG, base.ACTION_SHORT), base.ACTION_CASH).astype(np.int64)
    conf = np.clip((best_ev - float(ev_min)) / 0.02, 0.0, 1.0).astype(np.float64)
    return action, conf


def _apply_veto(action: np.ndarray, conf: np.ndarray, stop_prob: np.ndarray, stop_max: float, opp_prob: np.ndarray | None = None, opp_min: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    out_action = action.copy()
    out_conf = conf.copy()
    keep = stop_prob <= float(stop_max)
    if opp_prob is not None:
        keep &= opp_prob >= float(opp_min)
        out_conf = np.minimum(out_conf, opp_prob)
    out_action[~keep] = base.ACTION_CASH
    out_conf[~keep] = 0.0
    return out_action, out_conf


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_report = json.loads(base.BASELINE_REPORT.read_text(encoding="utf-8"))
    selected_cfg = baseline_report["selected_config"]
    cfg = base.repair.RunnerConfig(
        int(selected_cfg["candidate_id"]),
        str(selected_cfg["mode"]),
        float(selected_cfg["quality_min"]),
        float(selected_cfg["extend_mult"]),
        float(selected_cfg["floor_frac"]),
        int(selected_cfg["max_extensions"]),
    )
    print(json.dumps({"stage": "load", "model_id": MODEL_ID, "baseline": BASELINE_ID, "cfg": asdict(cfg)}, ensure_ascii=False), flush=True)
    data = base.legacy_runner._build()
    x_val = _enhanced_features(data["validation"])
    x_oos = _enhanced_features(data["oos"])
    base_val, base_val_ledger = base._simulate_combo(data["validation"], cfg, None, None, None, 1.0)
    base_oos, base_oos_ledger = base._simulate_combo(data["oos"], cfg, None, None, None, 1.0)
    base_val_ledger.to_csv(OUT_DIR / "validation_baseline_replay_ledger.csv", index=False)
    base_oos_ledger.to_csv(OUT_DIR / "oos_baseline_replay_ledger.csv", index=False)

    rows: list[dict[str, Any]] = [
        {"candidate": "baseline_tp_runner_clean_repair", "family": "baseline", "risk": "none", "min_edge": 0.0, "gate": "none", **_row("val", base_val, base_val_ledger), **_row("oos", base_oos, base_oos_ledger)}
    ]
    diagnostics: dict[str, Any] = {
        "feature_count": int(x_val.shape[1]),
        "features": list(x_val.columns),
        "selected_tp_runner_config": asdict(cfg),
        "baseline_replay": {"validation": base_val, "oos": base_oos},
    }
    ledgers: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for risk in base.RISKS:
        for min_edge in (0.002, 0.004, 0.006):
            print(json.dumps({"stage": "labels", "risk": risk.name, "min_edge": min_edge}, ensure_ascii=False), flush=True)
            labels, label_diag = _label_table(data["validation"], risk, min_edge)
            diagnostics[f"{risk.name}_edge{min_edge}"] = label_diag
            if len(labels) < 500:
                continue
            idx = labels["i"].to_numpy(dtype=np.int64)
            y_cls = np.zeros(len(x_val), dtype=np.int64)
            y_cls[idx] = labels["cls_label"].to_numpy(dtype=np.int64)
            y_opp = np.zeros(len(x_val), dtype=np.int64)
            y_opp[idx] = labels["opp_label"].to_numpy(dtype=np.int64)
            y_stop = np.zeros(len(x_val), dtype=np.int64)
            y_stop[idx] = labels["best_stop"].to_numpy(dtype=np.int64)
            y_long = np.zeros(len(x_val), dtype=np.float64)
            y_short = np.zeros(len(x_val), dtype=np.float64)
            y_long[idx] = labels["long_net"].to_numpy(dtype=np.float64)
            y_short[idx] = labels["short_net"].to_numpy(dtype=np.float64)

            print(json.dumps({"stage": "fit_base_models", "risk": risk.name, "min_edge": min_edge}, ensure_ascii=False), flush=True)
            cls_preds = {}
            for kind in ("hgb", "extra"):
                if len(np.unique(y_cls[idx])) < 2:
                    continue
                cls_preds[kind] = _fit_predict_classifier(kind, x_val, y_cls, idx, x_oos, seed=260615)
            opp_preds = {}
            stop_preds = {}
            for kind in ("hgb", "extra"):
                if len(np.unique(y_opp[idx])) >= 2:
                    opp_preds[kind] = _fit_predict_binary(kind, x_val, y_opp, idx, x_oos, seed=261000)
                if len(np.unique(y_stop[idx])) >= 2:
                    stop_preds[kind] = _fit_predict_binary(kind, x_val, y_stop, idx, x_oos, seed=261500)
            ev_preds = {}
            for kind in ("hgb", "extra"):
                val_long, oos_long, long_diag = _fit_predict_regressor(kind, x_val, y_long, idx, x_oos, seed=262000)
                val_short, oos_short, short_diag = _fit_predict_regressor(kind, x_val, y_short, idx, x_oos, seed=262500)
                ev_preds[kind] = (val_long, val_short, oos_long, oos_short, {"long": long_diag, "short": short_diag})

            candidates: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
            for kind, pred in cls_preds.items():
                val_a, val_c, oos_a, oos_c, diag = pred
                diagnostics[f"{risk.name}_edge{min_edge}_cls_{kind}"] = diag
                for thr in (0.55, 0.65, 0.75, 0.85, 0.90):
                    candidates.append((f"cls_{kind}_thr{thr}", f"cls_{kind}", val_a, val_c, oos_a, oos_c, thr))

            if "hgb" in cls_preds and "extra" in cls_preds:
                hv_a, hv_c, ho_a, ho_c, _ = cls_preds["hgb"]
                ev_a, ev_c, eo_a, eo_c, _ = cls_preds["extra"]
                val_agree = (hv_a == ev_a) & np.isin(hv_a, [base.ACTION_LONG, base.ACTION_SHORT])
                oos_agree = (ho_a == eo_a) & np.isin(ho_a, [base.ACTION_LONG, base.ACTION_SHORT])
                val_a = np.where(val_agree, hv_a, base.ACTION_CASH).astype(np.int64)
                oos_a = np.where(oos_agree, ho_a, base.ACTION_CASH).astype(np.int64)
                val_c = np.where(val_agree, np.minimum(hv_c, ev_c), 0.0).astype(np.float64)
                oos_c = np.where(oos_agree, np.minimum(ho_c, eo_c), 0.0).astype(np.float64)
                for thr in (0.45, 0.55, 0.65, 0.75):
                    candidates.append((f"cls_agreement_thr{thr}", "cls_agreement", val_a, val_c, oos_a, oos_c, thr))

            for kind, pred in ev_preds.items():
                val_long, val_short, oos_long, oos_short, diag = pred
                diagnostics[f"{risk.name}_edge{min_edge}_ev_{kind}"] = diag
                for ev_min in (0.001, 0.002, 0.004, 0.006):
                    val_a, val_c = _actions_from_ev(val_long, val_short, ev_min)
                    oos_a, oos_c = _actions_from_ev(oos_long, oos_short, ev_min)
                    candidates.append((f"ev_{kind}_min{ev_min}", f"ev_{kind}", val_a, val_c, oos_a, oos_c, 0.0))
                    if kind in stop_preds:
                        val_stop, oos_stop, stop_diag = stop_preds[kind]
                        diagnostics[f"{risk.name}_edge{min_edge}_stop_{kind}"] = stop_diag
                        for stop_max in (0.35, 0.45, 0.55):
                            va2, vc2 = _apply_veto(val_a, val_c, val_stop, stop_max)
                            oa2, oc2 = _apply_veto(oos_a, oos_c, oos_stop, stop_max)
                            candidates.append((f"ev_{kind}_min{ev_min}_stop{stop_max}", f"ev_stop_{kind}", va2, vc2, oa2, oc2, 0.0))
                    if kind in opp_preds and kind in stop_preds:
                        val_opp, oos_opp, opp_diag = opp_preds[kind]
                        diagnostics[f"{risk.name}_edge{min_edge}_opp_{kind}"] = opp_diag
                        val_stop, oos_stop, _ = stop_preds[kind]
                        for opp_min in (0.45, 0.55, 0.65):
                            for stop_max in (0.45, 0.55):
                                va2, vc2 = _apply_veto(val_a, val_c, val_stop, stop_max, val_opp, opp_min)
                                oa2, oc2 = _apply_veto(oos_a, oos_c, oos_stop, stop_max, oos_opp, opp_min)
                                candidates.append((f"hurdle_ev_{kind}_min{ev_min}_opp{opp_min}_stop{stop_max}", f"hurdle_ev_{kind}", va2, vc2, oa2, oc2, 0.0))

            for name, family, val_a, val_c, oos_a, oos_c, threshold in candidates:
                val_m, val_ledger = base._simulate_combo(data["validation"], cfg, risk, val_a, val_c, threshold)
                oos_m, oos_ledger = base._simulate_combo(data["oos"], cfg, risk, oos_a, oos_c, threshold)
                key = f"{risk.name}_edge{min_edge}_{name}"
                row = {"candidate": key, "family": family, "risk": risk.name, "min_edge": float(min_edge), "gate": name}
                row.update(_row("val", val_m, val_ledger))
                row.update(_row("oos", oos_m, oos_ledger))
                rows.append(row)
                ledgers[key] = (val_ledger, oos_ledger)

    ranking = pd.DataFrame(rows)
    ranking["val_delta_pnl"] = ranking["val_pnl"] - float(base_val["pnl"])
    ranking["oos_delta_pnl"] = ranking["oos_pnl"] - float(base_oos["pnl"])
    ranking["fallback_score_val_only"] = (
        ranking["val_fallback_pnl"].fillna(0.0)
        + 1.5 * ranking["val_fallback_profit_factor"].fillna(0.0)
        - 0.60 * ranking["val_fallback_stop_rate"].fillna(0.0) * 100.0
        + 0.10 * ranking["val_fallback_trades"].fillna(0.0)
        + 0.20 * ranking["val_delta_pnl"].fillna(0.0)
    )
    ranking = ranking.sort_values(["fallback_score_val_only", "val_fallback_pnl", "val_delta_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "upgrade_validation_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    best_oos = ranking.sort_values(["oos_fallback_pnl", "oos_fallback_profit_factor", "oos_delta_pnl"], ascending=False).iloc[0].to_dict()
    if str(selected["candidate"]) in ledgers:
        val_ledger, oos_ledger = ledgers[str(selected["candidate"])]
        val_ledger.to_csv(OUT_DIR / "selected_validation_ledger.csv", index=False)
        oos_ledger.to_csv(OUT_DIR / "selected_oos_ledger.csv", index=False)
        val_ledger[val_ledger["sleeve"] == "fallback"].to_csv(OUT_DIR / "selected_validation_fallback_only_ledger.csv", index=False)
        oos_ledger[oos_ledger["sleeve"] == "fallback"].to_csv(OUT_DIR / "selected_oos_fallback_only_ledger.csv", index=False)
    if str(best_oos["candidate"]) in ledgers:
        _val_ledger, oos_ledger = ledgers[str(best_oos["candidate"])]
        oos_ledger.to_csv(OUT_DIR / "best_oos_diagnostic_ledger.csv", index=False)
        oos_ledger[oos_ledger["sleeve"] == "fallback"].to_csv(OUT_DIR / "best_oos_diagnostic_fallback_only_ledger.csv", index=False)

    feature_names = list(x_val.columns)
    forbidden_features = [c for c in feature_names if c in base.FORBIDDEN_FEATURE_EXACT or c.startswith(base.FORBIDDEN_FEATURE_PREFIXES)]
    redteam_blockers: list[str] = []
    if len(x_val) != len(data["validation"]["dec"]):
        redteam_blockers.append("validation feature/decision row count mismatch")
    if len(x_oos) != len(data["oos"]["dec"]):
        redteam_blockers.append("oos feature/decision row count mismatch")
    if feature_names != list(x_oos.columns):
        redteam_blockers.append("validation/oos feature columns mismatch")
    if forbidden_features:
        redteam_blockers.append(f"forbidden feature columns present: {forbidden_features[:20]}")
    if len(ranking) <= 1:
        redteam_blockers.append("no upgraded sleeve candidates were produced")
    if str(selected["candidate"]) == "baseline_tp_runner_clean_repair":
        redteam_blockers.append("validation selection returned baseline instead of upgraded sleeve candidate")

    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_shadow_candidate" if not redteam_blockers else "redteam_fail",
        "baseline_model_id": BASELINE_ID,
        "method": "Omega1.2.3 cash sleeve upgrade sweep: classifier, classifier agreement, EV regression, stop-risk veto, and opportunity hurdle candidates. Selection uses validation only; OOS is diagnostic.",
        "selection_policy": "validation_only_no_oos_selection",
        "redteam_policy": "PnL and OOS lift are diagnostics only. FAIL is limited to logical defects, data/feature contract violations, forbidden feature leakage, or failed candidate generation.",
        "diagnostics": diagnostics,
        "baseline": {"validation": base_val, "oos": base_oos},
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "top20": ranking.head(20).to_dict(orient="records"),
        "redteam_pass": not redteam_blockers,
        "redteam_blockers": redteam_blockers,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "upgrade_validation_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
            "selected_validation_ledger": str(OUT_DIR / "selected_validation_ledger.csv"),
            "selected_oos_ledger": str(OUT_DIR / "selected_oos_ledger.csv"),
            "selected_validation_fallback_only_ledger": str(OUT_DIR / "selected_validation_fallback_only_ledger.csv"),
            "selected_oos_fallback_only_ledger": str(OUT_DIR / "selected_oos_fallback_only_ledger.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "status": report["status"], "selected": selected, "best_oos_diagnostic": best_oos}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
