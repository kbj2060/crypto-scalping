#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402
import train_eval_omega1_2_1_tp_runner_meta_selector_20260610 as meta  # noqa: E402


MODEL_ID = "omega1_2_1_inposition_exit_model_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

LAMBDA_GIVEBACK = 0.30
LAMBDA_TIME = 0.03
EPS_R = 0.12
TEMPERATURE_R = 0.07
MIN_HOLD_BARS = 2
MAX_LABEL_FWD_BARS = 48


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


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _feature_row(frame: pd.DataFrame, state: pd.DataFrame, pos: base.Position, i: int, unreal: float) -> dict[str, float]:
    row = state.iloc[int(i)]
    close = pd.to_numeric(frame["close"], errors="raise")
    ret3 = float(close.pct_change(3).iloc[int(i)] if int(i) >= 3 else 0.0)
    ret6 = float(close.pct_change(6).iloc[int(i)] if int(i) >= 6 else 0.0)
    ret12 = float(close.pct_change(12).iloc[int(i)] if int(i) >= 12 else 0.0)
    side = float(pos.side)
    mfe = max(float(pos.mfe), float(unreal))
    mae = min(float(pos.mae), float(unreal))
    stop = max(abs(float(pos.stop_loss)), 1e-8)
    tp = max(float(pos.take_profit), 1e-8)
    giveback = max(0.0, mfe - float(unreal))
    giveback_ratio = giveback / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
    return {
        "side": side,
        "hold_bars": float(max(int(i) - int(pos.entry_i), 0)),
        "unreal": float(unreal),
        "current_pnl_R": float(unreal / stop),
        "mfe": float(mfe),
        "mae": float(mae),
        "mfe_R": float(mfe / stop),
        "mae_R": float(mae / stop),
        "giveback": float(giveback),
        "giveback_R": float(giveback / stop),
        "giveback_ratio": float(np.clip(giveback_ratio, 0.0, 10.0)),
        "dist_tp": float(tp - float(unreal)),
        "dist_sl": float(float(unreal) + stop),
        "tp_progress": float(float(unreal) / tp),
        "sl_progress": float(-float(unreal) / stop),
        "floor_unreal": float(pos.floor_unreal),
        "ret3_side": float(ret3 * side),
        "ret6_side": float(ret6 * side),
        "ret12_side": float(ret12 * side),
        "ret3_abs": float(abs(ret3)),
        "ret6_abs": float(abs(ret6)),
        "quality": float(row.get("tabm_quality_for_action", 0.0)),
        "router_confidence": float(row.get("tabm_router_confidence", 0.0)),
        "router_margin": float(row.get("tabm_router_margin", 0.0)),
        "dir_confidence": float(row.get("tabm_dir_confidence", 0.0)),
        "dir_side_edge": float(row.get("tabm_dir_side_edge", 0.0)),
        "dir_trade_prob": float(row.get("tabm_dir_trade_prob", 0.0)),
        "p_long_minus_short": float(row.get("tabm_dir_p_long", 0.0) - row.get("tabm_dir_p_short", 0.0)),
        "atr14_pct": float(row.get("atr14_pct", 0.0)),
        "bar_range_pct": float(row.get("bar_range_pct", 0.0)),
        "ema9_21_gap_side": float(row.get("ema9_21_gap", 0.0) * side),
        "tod_sin": float(row.get("tod_sin", 0.0)),
        "tod_cos": float(row.get("tod_cos", 0.0)),
    }


def _future_labels(arrays: dict[str, np.ndarray], pos: base.Position, i: int, *, slip_eff: float) -> dict[str, float]:
    stop = max(abs(float(pos.stop_loss)), 1e-8)
    last = min(len(arrays["close"]) - 2, int(i) + int(MAX_LABEL_FWD_BARS))
    if last <= int(i):
        return {"edge_hold_R": 0.0, "y_exit": 1.0, "y_exit_soft": 1.0, "future_worst_R": 0.0, "future_best_R": 0.0}

    idxs = np.arange(int(i), last + 1, dtype=np.int64)
    unreal = np.asarray([base._unreal(arrays, pos, int(j), slip_eff) for j in idxs], dtype=np.float64)
    r_exit = unreal / stop
    mfe0 = max(float(pos.mfe), float(unreal[0]))
    mfe_path = np.maximum.accumulate(np.maximum(unreal, mfe0))
    giveback = np.maximum(0.0, mfe_path - unreal) / stop
    age_norm = (idxs - int(pos.entry_i)) / max(float(MAX_LABEL_FWD_BARS), 1.0)
    utility = r_exit - float(LAMBDA_GIVEBACK) * giveback - float(LAMBDA_TIME) * age_norm
    u_now = float(utility[0])
    best_future_u = float(np.max(utility[1:])) if len(utility) > 1 else u_now
    edge_hold_R = best_future_u - u_now
    age_bars = max(int(i) - int(pos.entry_i), 0)
    y_exit = float((age_bars >= int(MIN_HOLD_BARS) and edge_hold_R <= float(EPS_R)) or int(i) >= last)
    y_exit_soft = float(_sigmoid((float(EPS_R) - edge_hold_R) / float(TEMPERATURE_R)))
    return {
        "edge_hold_R": float(edge_hold_R),
        "y_exit": y_exit,
        "y_exit_soft": y_exit_soft,
        "future_worst_R": float(np.min(r_exit)),
        "future_best_R": float(np.max(r_exit)),
    }


def _tp_runner_extend_allowed(bundle: dict[str, Any] | None, frame: pd.DataFrame, state: pd.DataFrame, pos: base.Position, i: int, unreal: float) -> bool:
    if not bundle:
        return False
    template = meta.RunnerTemplate(**bundle["template"])
    return meta._selector_allowed(
        bundle.get("model"),
        list(bundle.get("feature_cols", [])),
        frame,
        state,
        pos,
        int(i),
        float(unreal),
        template=template,
        proba_min=float(bundle.get("proba_min", 2.0)),
    )


def _collect_exit_dataset(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    tp_bundle: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    template = meta.RunnerTemplate(**tp_bundle["template"]) if tp_bundle else meta.TEMPLATES[0]
    cash = 1.0
    pos = base.Position()
    rows: list[dict[str, Any]] = []
    trade_id = 0
    extensions = 0
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            reason = ""
            if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                if tp_bundle and extensions < int(template.max_extensions) and _tp_runner_extend_allowed(tp_bundle, frame, state, pos, i, unreal):
                    extensions += 1
                    old_tp = float(pos.take_profit)
                    pos.floor_unreal = max(float(pos.floor_unreal), old_tp * float(template.floor_frac))
                    pos.take_profit = old_tp * float(template.extend_mult)
                else:
                    reason = "take_profit"
            elif pos.floor_unreal > -abs(pos.stop_loss) and unreal <= pos.floor_unreal:
                reason = "meta_runner_profit_lock_exit"
            elif pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
                reason = "stop_loss"

            if not reason:
                feat = _feature_row(frame, state, pos, i, unreal)
                lab = _future_labels(arrays, pos, i, slip_eff=slip_eff)
                rows.append(
                    {
                        "trade_id": int(trade_id),
                        "decision_i": int(i),
                        "entry_i": int(pos.entry_i),
                        "entry_signal_i": int(pos.entry_signal_i),
                        "decision_time": str(frame["timestamp"].iloc[int(i)]),
                        "entry_time": str(frame["timestamp"].iloc[int(pos.entry_signal_i)]),
                        **feat,
                        **lab,
                    }
                )

            if reason:
                cash, pos, _ = base._close_fraction(cash, arrays, pos, i, 1.0, fee_eff, slip_eff)
                extensions = 0
            continue

        if bool(active[i]):
            cash, pos, entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
            if entered:
                trade_id += 1
                extensions = 0

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("empty in-position exit dataset")
    diag = {
        "rows": int(len(df)),
        "trades": int(df["trade_id"].nunique()),
        "positive": int(df["y_exit"].sum()),
        "positive_rate": float(df["y_exit"].mean()),
        "soft_mean": float(df["y_exit_soft"].mean()),
        "edge_mean": float(df["edge_hold_R"].mean()),
        "edge_median": float(df["edge_hold_R"].median()),
    }
    return df, diag


def _train_models(df: pd.DataFrame, *, kind: str, seed: int) -> tuple[Any, Any, Any, list[str], dict[str, Any]]:
    drop = {
        "trade_id",
        "decision_i",
        "entry_i",
        "entry_signal_i",
        "decision_time",
        "entry_time",
        "edge_hold_R",
        "y_exit",
        "y_exit_soft",
        "future_worst_R",
        "future_best_R",
    }
    feature_cols = [c for c in df.columns if c not in drop]
    x = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["y_exit"].astype(int).to_numpy()
    edge = df["edge_hold_R"].to_numpy(dtype=np.float64)
    worst = df["future_worst_R"].to_numpy(dtype=np.float64)
    if len(np.unique(y)) < 2:
        raise RuntimeError("exit labels are single-class")
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=6, min_samples_leaf=12, l2_regularization=1.0, learning_rate=0.04, random_state=int(seed))
        edge_reg = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=6, min_samples_leaf=12, l2_regularization=1.0, learning_rate=0.04, random_state=int(seed) + 1)
        worst_reg = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=6, min_samples_leaf=12, l2_regularization=1.0, learning_rate=0.04, random_state=int(seed) + 2)
    elif kind == "et":
        clf = ExtraTreesClassifier(n_estimators=180, max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=int(seed))
        edge_reg = ExtraTreesRegressor(n_estimators=180, max_depth=4, min_samples_leaf=8, random_state=int(seed) + 1)
        worst_reg = ExtraTreesRegressor(n_estimators=180, max_depth=4, min_samples_leaf=8, random_state=int(seed) + 2)
    else:
        raise RuntimeError(f"unknown model kind: {kind}")
    clf.fit(x, y)
    edge_reg.fit(x, edge)
    worst_reg.fit(x, worst)
    diag = {"kind": kind, "seed": int(seed), "feature_cols": feature_cols, "rows": int(len(df)), "positive_rate": float(np.mean(y))}
    return clf, edge_reg, worst_reg, feature_cols, diag


def _predict_exit(clf: Any, edge_reg: Any, worst_reg: Any, feature_cols: list[str], feat: dict[str, float], *, p_min: float, edge_max: float, worst_min: float) -> bool:
    x = np.asarray([[float(feat[c]) for c in feature_cols]], dtype=np.float64)
    p = float(clf.predict_proba(x)[0, 1]) if hasattr(clf, "predict_proba") else float(clf.predict(x)[0])
    edge = float(edge_reg.predict(x)[0])
    worst = float(worst_reg.predict(x)[0])
    return bool(p >= float(p_min) and edge <= float(edge_max) and worst >= float(worst_min))


def _metrics(cash: float, equity_curve: list[float], trades: list[float], reasons: dict[str, int], model_exits: int, long_entries: int, short_entries: int) -> dict[str, Any]:
    eq = np.asarray(equity_curve if equity_curve else [1.0], dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(peak, 1e-12) - 1.0) * 100.0
    arr = np.asarray(trades, dtype=np.float64)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(dd.min()),
        "trades": int(len(trades)),
        "wr": float(np.mean(arr > 0.0)) if len(arr) else 0.0,
        "model_exits": int(model_exits),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": dict(reasons),
    }


def _simulate(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    tp_bundle: dict[str, Any] | None,
    clf: Any | None,
    edge_reg: Any | None,
    worst_reg: Any | None,
    feature_cols: list[str],
    p_min: float,
    edge_max: float,
    worst_min: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    template = meta.RunnerTemplate(**tp_bundle["template"]) if tp_bundle else meta.TEMPLATES[0]
    cash = 1.0
    equity_curve = [cash]
    trades: list[float] = []
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    pos = base.Position()
    extensions = 0
    model_exits = 0
    long_entries = short_entries = 0
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            equity_curve.append(cash * (1.0 + unreal))
            reason = ""
            if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                if tp_bundle and extensions < int(template.max_extensions) and _tp_runner_extend_allowed(tp_bundle, frame, state, pos, i, unreal):
                    extensions += 1
                    old_tp = float(pos.take_profit)
                    pos.floor_unreal = max(float(pos.floor_unreal), old_tp * float(template.floor_frac))
                    pos.take_profit = old_tp * float(template.extend_mult)
                else:
                    reason = "take_profit"
            elif pos.floor_unreal > -abs(pos.stop_loss) and unreal <= pos.floor_unreal:
                reason = "meta_runner_profit_lock_exit"
            elif pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
                reason = "stop_loss"
            elif clf is not None and max(int(i) - int(pos.entry_i), 0) >= int(MIN_HOLD_BARS):
                feat = _feature_row(frame, state, pos, i, unreal)
                if _predict_exit(clf, edge_reg, worst_reg, feature_cols, feat, p_min=p_min, edge_max=edge_max, worst_min=worst_min):
                    reason = "model_exit"
                    model_exits += 1
            if reason:
                close_pos = base.Position(**pos.__dict__)
                cash, pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(runner._ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason, extensions))
                extensions = 0
            continue
        equity_curve.append(cash)
        if bool(active[i]):
            side = int(dec.iloc[int(i)].get("side", 0) or 0)
            cash, pos, entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
            if entered:
                long_entries += int(side > 0)
                short_entries += int(side < 0)
                extensions = 0
    if pos.side != 0:
        close_pos = base.Position(**pos.__dict__)
        cash, pos, _ = base._close_fraction(cash, arrays, close_pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(runner._ledger_row(frame, arrays, close_pos, len(frame) - 1, cash, net_pct, "forced_end", extensions))
    return _metrics(cash, equity_curve, trades, reasons, model_exits, long_entries, short_entries), pd.DataFrame(rows)


def _row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_model_exits": int(metrics["model_exits"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = runner._build()
    print(json.dumps({"stage": "build_done", "sec": round(time.time() - t0, 3)}), flush=True)
    bundle_path = ROOT / "data/ensemble/supervised/omega1_2_1_tp_runner_meta_selector_20260610/tp_runner_meta_selector.joblib"
    tp_bundle = None
    if bundle_path.exists():
        import joblib

        tp_bundle = joblib.load(bundle_path)

    train_df, dataset_diag = _collect_exit_dataset(
        data["validation"]["frame"],
        data["validation"]["dec"],
        data["validation"]["state"],
        fee=float(data["validation"]["fee"]),
        slip=float(data["validation"]["slip"]),
        cost_mult=3.0,
        tp_bundle=tp_bundle,
    )
    train_df.to_csv(OUT_DIR / "validation_inposition_exit_dataset.csv", index=False)
    print(json.dumps({"stage": "dataset_done", **dataset_diag, "sec": round(time.time() - t0, 3)}), flush=True)

    configs: list[dict[str, Any]] = [
        {"variant": "baseline_no_runner", "tp_bundle": None, "clf": None, "edge_reg": None, "worst_reg": None, "feature_cols": [], "p_min": 2.0, "edge_max": -999.0, "worst_min": -999.0},
        {"variant": "tp_runner_only", "tp_bundle": tp_bundle, "clf": None, "edge_reg": None, "worst_reg": None, "feature_cols": [], "p_min": 2.0, "edge_max": -999.0, "worst_min": -999.0},
    ]
    model_diags: list[dict[str, Any]] = []
    for kind in ("hgb",):
        for seed in (260613,):
            clf, edge_reg, worst_reg, feature_cols, diag = _train_models(train_df, kind=kind, seed=seed)
            model_diags.append(diag)
            for p_min, edge_max, worst_min in (
                (0.75, 0.08, -0.90),
                (0.85, 0.03, -0.60),
            ):
                configs.append(
                    {
                        "variant": f"tp_runner_exit_{kind}_s{seed}_p{p_min:.2f}_e{edge_max:.2f}_w{worst_min:.2f}",
                        "tp_bundle": tp_bundle,
                        "clf": clf,
                        "edge_reg": edge_reg,
                        "worst_reg": worst_reg,
                        "feature_cols": feature_cols,
                        "p_min": float(p_min),
                        "edge_max": float(edge_max),
                        "worst_min": float(worst_min),
                    }
                )

    rows: list[dict[str, Any]] = []
    ledgers: dict[int, dict[str, pd.DataFrame]] = {}
    for idx, cfg in enumerate(configs):
        print(json.dumps({"stage": "simulate_start", "variant": cfg["variant"], "sec": round(time.time() - t0, 3)}), flush=True)
        row: dict[str, Any] = {
            "variant_id": int(idx),
            "variant": str(cfg["variant"]),
            "p_min": float(cfg["p_min"]),
            "edge_max": float(cfg["edge_max"]),
            "worst_min": float(cfg["worst_min"]),
        }
        ledgers[idx] = {}
        for split in ("validation", "oos"):
            m, ledger = _simulate(
                data[split]["frame"],
                data[split]["dec"],
                data[split]["state"],
                fee=float(data[split]["fee"]),
                slip=float(data[split]["slip"]),
                cost_mult=3.0,
                tp_bundle=cfg["tp_bundle"],
                clf=cfg["clf"],
                edge_reg=cfg["edge_reg"],
                worst_reg=cfg["worst_reg"],
                feature_cols=list(cfg["feature_cols"]),
                p_min=float(cfg["p_min"]),
                edge_max=float(cfg["edge_max"]),
                worst_min=float(cfg["worst_min"]),
            )
            row.update(_row(split, m))
            ledgers[idx][split] = ledger
        rows.append(row)

    ranking = pd.DataFrame(rows)
    base_oos = float(ranking.loc[ranking["variant"].eq("baseline_no_runner"), "oos_pnl"].iloc[0])
    base_val = float(ranking.loc[ranking["variant"].eq("baseline_no_runner"), "validation_pnl"].iloc[0])
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - base_oos
    ranking["delta_validation_pnl"] = ranking["validation_pnl"] - base_val
    ranking["score"] = ranking["oos_pnl"] + 0.45 * ranking["validation_pnl"] + 0.25 * ranking["oos_mdd"] + 0.20 * ranking["validation_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "oos_mdd"], ascending=[False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "inposition_exit_model_ranking.csv", index=False)
    for variant_id in sorted(set([0, 1] + [int(x) for x in ranking["variant_id"].head(5).tolist()])):
        for split, ledger in ledgers[variant_id].items():
            ledger.to_csv(OUT_DIR / f"{split}_variant{variant_id}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "method": "In-position candle-expanded exit model. Labels compare current exit utility against best future utility; runtime closes only when p_exit, hold-edge, and future-worst gates agree.",
        "label_params": {
            "lambda_giveback": LAMBDA_GIVEBACK,
            "lambda_time": LAMBDA_TIME,
            "eps_R": EPS_R,
            "temperature_R": TEMPERATURE_R,
            "min_hold_bars": MIN_HOLD_BARS,
            "max_label_fwd_bars": MAX_LABEL_FWD_BARS,
        },
        "dataset_diag": dataset_diag,
        "model_diags": model_diags,
        "top": ranking.head(30).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "dataset": str(OUT_DIR / "validation_inposition_exit_dataset.csv"),
            "ranking": str(OUT_DIR / "inposition_exit_model_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top10": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
