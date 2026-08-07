#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402


MODEL_ID = "omega1_2_1_autonomous_exit_no_sltp_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

MIN_HOLD_BARS = 2
TRAIN_FWD_BARS = 288
FEE_COST_MULT = 3.0
UTILITY_GIVEBACK = 0.35
UTILITY_TIME = 0.00008


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


def _disable_sltp(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = base.omega._active(out)
    out.loc[active, "take_profit"] = 0.0
    out.loc[active, "stop_loss"] = 0.0
    out.loc[active, "max_hold_bars"] = 0
    out.loc[active, "cooldown_bars"] = 0
    return out


def _build() -> dict[str, dict[str, Any]]:
    fee, slip = base.omega._load_fee_slip()
    splits = base._build_splits()
    out: dict[str, dict[str, Any]] = {}
    for split, payload in splits.items():
        dec = base._to_decisions(payload["src"], payload["prefix"], oof=payload["oof"], thresholds=base.HIGH_THRESHOLDS)
        dec = _disable_sltp(dec)
        state = base._state_base(payload["frame"], payload["src"], dec, payload["prefix"])
        out[split] = {
            "frame": payload["frame"].reset_index(drop=True),
            "dec": dec.reset_index(drop=True),
            "state": state.reset_index(drop=True),
            "fee": fee,
            "slip": slip,
        }
    return out


def _entry_pos(cash: float, arrays: dict[str, np.ndarray], dec: pd.DataFrame, i: int, fee_eff: float, slip_eff: float) -> tuple[float, base.Position, bool]:
    cash2, pos, entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
    if entered:
        pos.take_profit = 0.0
        pos.stop_loss = 0.0
        pos.floor_unreal = -1e9
    return cash2, pos, entered


def _feature_row(frame: pd.DataFrame, state: pd.DataFrame, pos: base.Position, i: int, unreal: float) -> dict[str, float]:
    row = state.iloc[int(i)]
    side = float(pos.side)
    hold = max(int(i) - int(pos.entry_i), 0)
    mfe = max(float(pos.mfe), float(unreal))
    mae = min(float(pos.mae), float(unreal))
    giveback = max(0.0, mfe - float(unreal))
    giveback_ratio = giveback / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
    ret = {}
    for lag in (1, 3, 6, 12, 24, 48):
        val = float(row.get(f"ret_{lag}", 0.0))
        ret[f"ret{lag}_side"] = val * side
        ret[f"ret{lag}_abs"] = abs(val)
    return {
        "side": side,
        "hold_bars": float(hold),
        "hold_log1p": float(np.log1p(hold)),
        "unreal": float(unreal),
        "unreal_per_bar": float(unreal / max(hold, 1)),
        "mfe": float(mfe),
        "mae": float(mae),
        "giveback": float(giveback),
        "giveback_ratio": float(np.clip(giveback_ratio, 0.0, 10.0)),
        "mfe_to_mae": float(mfe / max(abs(mae), 1e-8)),
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
        **ret,
    }


def _path_unreal(arrays: dict[str, np.ndarray], pos: base.Position, start: int, end: int, slip_eff: float) -> np.ndarray:
    return np.asarray([base._unreal(arrays, pos, j, slip_eff) for j in range(int(start), int(end) + 1)], dtype=np.float64)


def _labels_from_path(unreal_path: np.ndarray, *, global_i: int, entry_i: int) -> dict[str, float]:
    if len(unreal_path) <= 1:
        return {"y_exit": 1.0, "exit_soft": 1.0, "hold_edge": 0.0, "future_worst": float(unreal_path[0]) if len(unreal_path) else 0.0}
    idx = np.arange(len(unreal_path), dtype=np.float64)
    mfe = np.maximum.accumulate(unreal_path)
    giveback = np.maximum(0.0, mfe - unreal_path)
    utility = unreal_path - float(UTILITY_GIVEBACK) * giveback - float(UTILITY_TIME) * idx
    now_u = float(utility[0])
    best_future = float(np.max(utility[1:]))
    hold_edge = best_future - now_u
    hold_bars = max(int(global_i) - int(entry_i), 0)
    # Positive means current exit is near the best ex-ante path utility.
    y_exit = float(hold_bars >= int(MIN_HOLD_BARS) and hold_edge <= 0.0015)
    exit_soft = float(1.0 / (1.0 + np.exp(np.clip((hold_edge - 0.0015) / 0.0015, -50.0, 50.0))))
    return {
        "y_exit": y_exit,
        "exit_soft": exit_soft,
        "hold_edge": float(hold_edge),
        "future_worst": float(np.min(unreal_path)),
        "future_best": float(np.max(unreal_path)),
    }


def _collect_dataset(payload: dict[str, Any], *, cost_mult: float, train_fwd_bars: int, max_entries: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = payload["frame"]
    dec = payload["dec"]
    state = payload["state"]
    arrays = base._arrays(frame)
    active_idx = np.flatnonzero(base.omega._active(dec))
    fee_eff = float(payload["fee"]) * float(cost_mult)
    slip_eff = float(payload["slip"]) * float(cost_mult)
    rows: list[dict[str, Any]] = []
    entries = 0
    for entry_signal_i in active_idx:
        cash, pos, entered = _entry_pos(1.0, arrays, dec, int(entry_signal_i), fee_eff, slip_eff)
        if not entered:
            continue
        entries += 1
        last = min(len(frame) - 2, int(pos.entry_i) + int(train_fwd_bars))
        for i in range(int(pos.entry_i), last + 1):
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            path = _path_unreal(arrays, pos, i, last, slip_eff)
            rows.append(
                {
                    "entry_signal_i": int(entry_signal_i),
                    "entry_i": int(pos.entry_i),
                    "decision_i": int(i),
                    "entry_time": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                    "decision_time": str(frame["timestamp"].iloc[int(i)]),
                    **_feature_row(frame, state, pos, i, unreal),
                    **_labels_from_path(path, global_i=i, entry_i=int(pos.entry_i)),
                }
            )
        if max_entries is not None and entries >= int(max_entries):
            break
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("empty autonomous exit dataset")
    diag = {
        "rows": int(len(df)),
        "entries": int(entries),
        "positive": int(df["y_exit"].sum()),
        "positive_rate": float(df["y_exit"].mean()),
        "hold_edge_mean": float(df["hold_edge"].mean()),
        "hold_edge_median": float(df["hold_edge"].median()),
        "train_fwd_bars": int(train_fwd_bars),
        "max_entries": None if max_entries is None else int(max_entries),
    }
    return df, diag


def _train(df: pd.DataFrame, *, kind: str, seed: int) -> tuple[Any, Any, Any, list[str], dict[str, Any]]:
    drop = {
        "entry_signal_i",
        "entry_i",
        "decision_i",
        "entry_time",
        "decision_time",
        "y_exit",
        "exit_soft",
        "hold_edge",
        "future_worst",
        "future_best",
    }
    feature_cols = [c for c in df.columns if c not in drop]
    x = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["y_exit"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError("exit labels are single-class")
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=8, min_samples_leaf=18, l2_regularization=1.0, learning_rate=0.04, random_state=int(seed))
        edge = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=8, min_samples_leaf=18, l2_regularization=1.0, learning_rate=0.04, random_state=int(seed) + 1)
        worst = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=8, min_samples_leaf=18, l2_regularization=1.0, learning_rate=0.04, random_state=int(seed) + 2)
    elif kind == "et":
        clf = ExtraTreesClassifier(n_estimators=240, max_depth=5, min_samples_leaf=12, class_weight="balanced", random_state=int(seed))
        edge = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=6, min_samples_leaf=20, l2_regularization=1.0, learning_rate=0.04, random_state=int(seed) + 1)
        worst = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=6, min_samples_leaf=20, l2_regularization=1.0, learning_rate=0.04, random_state=int(seed) + 2)
    else:
        raise RuntimeError(f"unknown model kind: {kind}")
    clf.fit(x, y)
    edge.fit(x, df["hold_edge"].to_numpy(dtype=np.float64))
    worst.fit(x, df["future_worst"].to_numpy(dtype=np.float64))
    return clf, edge, worst, feature_cols, {"kind": kind, "seed": int(seed), "feature_cols": feature_cols, "positive_rate": float(np.mean(y))}


def _should_exit(clf: Any, edge: Any, worst: Any, feature_cols: list[str], feat: dict[str, float], *, p_min: float, edge_max: float, worst_min: float) -> bool:
    x = np.asarray([[float(feat[c]) for c in feature_cols]], dtype=np.float64)
    p = float(clf.predict_proba(x)[0, 1]) if hasattr(clf, "predict_proba") else float(clf.predict(x)[0])
    e = float(edge.predict(x)[0])
    w = float(worst.predict(x)[0])
    return bool(p >= float(p_min) and e <= float(edge_max) and w >= float(worst_min))


def _metrics(cash: float, equity: list[float], trades: list[float], reasons: dict[str, int], long_entries: int, short_entries: int) -> dict[str, Any]:
    eq = np.asarray(equity if equity else [1.0], dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(peak, 1e-12) - 1.0) * 100.0
    arr = np.asarray(trades, dtype=np.float64)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(dd.min()),
        "trades": int(len(trades)),
        "wr": float(np.mean(arr > 0.0)) if len(arr) else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": dict(reasons),
    }


def _ledger_row(frame: pd.DataFrame, arrays: dict[str, np.ndarray], pos: base.Position, exit_i: int, cash: float, net_pct: float, reason: str) -> dict[str, Any]:
    return {
        "side": "LONG" if pos.side > 0 else "SHORT",
        "entry_signal_i": int(pos.entry_signal_i),
        "entry_i": int(pos.entry_i),
        "exit_i": int(exit_i),
        "entry_time": str(frame["timestamp"].iloc[int(pos.entry_signal_i)]),
        "exit_time": str(frame["timestamp"].iloc[int(exit_i)]),
        "entry_price": float(pos.entry_price),
        "exit_price": float(arrays["close"][int(exit_i)]),
        "effective_exposure": float(pos.notional),
        "margin_notional": float(pos.margin_notional),
        "leverage": float(pos.leverage),
        "tp_equity_ret": 0.0,
        "sl_equity_ret": 0.0,
        "net_trade_return_pct": float(net_pct),
        "mfe_pct": float(pos.mfe * 100.0),
        "mae_pct": float(pos.mae * 100.0),
        "exit_reason": str(reason),
        "cash_after": float(cash),
    }


def _simulate(payload: dict[str, Any], *, clf: Any | None, edge: Any | None, worst: Any | None, feature_cols: list[str], p_min: float, edge_max: float, worst_min: float, cost_mult: float) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = payload["frame"]
    dec = payload["dec"]
    state = payload["state"]
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(payload["fee"]) * float(cost_mult)
    slip_eff = float(payload["slip"]) * float(cost_mult)
    cash = 1.0
    equity = [cash]
    trades: list[float] = []
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    pos = base.Position()
    long_entries = short_entries = 0
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            equity.append(cash * (1.0 + unreal))
            reason = ""
            if clf is not None and max(int(i) - int(pos.entry_i), 0) >= int(MIN_HOLD_BARS):
                feat = _feature_row(frame, state, pos, i, unreal)
                if _should_exit(clf, edge, worst, feature_cols, feat, p_min=p_min, edge_max=edge_max, worst_min=worst_min):
                    reason = "model_exit"
            if reason:
                close_pos = base.Position(**pos.__dict__)
                cash, pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(_ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason))
            continue
        equity.append(cash)
        if bool(active[i]):
            side = int(dec.iloc[int(i)].get("side", 0) or 0)
            cash, pos, entered = _entry_pos(cash, arrays, dec, i, fee_eff, slip_eff)
            if entered:
                long_entries += int(side > 0)
                short_entries += int(side < 0)
    if pos.side != 0:
        close_pos = base.Position(**pos.__dict__)
        cash, pos, _ = base._close_fraction(cash, arrays, close_pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(_ledger_row(frame, arrays, close_pos, len(frame) - 1, cash, net_pct, "forced_end"))
    return _metrics(cash, equity, trades, reasons, long_entries, short_entries), pd.DataFrame(rows)


def _row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train-entries", type=int, default=80)
    ap.add_argument("--train-fwd-bars", type=int, default=144)
    ap.add_argument("--smoke-grid", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(json.dumps({"stage": "build_start"}), flush=True)
    data = _build()
    print(json.dumps({"stage": "build_done", "sec": round(time.time() - t0, 3)}), flush=True)
    train_df, dataset_diag = _collect_dataset(
        data["validation"],
        cost_mult=FEE_COST_MULT,
        train_fwd_bars=int(args.train_fwd_bars),
        max_entries=int(args.max_train_entries) if int(args.max_train_entries) > 0 else None,
    )
    train_df.to_csv(OUT_DIR / "validation_autonomous_exit_dataset.csv", index=False)
    print(json.dumps({"stage": "dataset_done", **dataset_diag, "sec": round(time.time() - t0, 3)}), flush=True)

    configs: list[dict[str, Any]] = [
        {"variant": "no_sltp_no_exit_model", "clf": None, "edge": None, "worst": None, "feature_cols": [], "p_min": 2.0, "edge_max": -999.0, "worst_min": -999.0}
    ]
    model_diags: list[dict[str, Any]] = []
    model_specs = (("hgb", 260612),) if bool(args.smoke_grid) else (("hgb", 260612), ("et", 260613))
    p_grid = (0.65,) if bool(args.smoke_grid) else (0.55, 0.65, 0.75, 0.85)
    edge_grid = (0.0015, 0.0030) if bool(args.smoke_grid) else (0.0005, 0.0015, 0.0030, 0.0060)
    worst_grid = (-0.20,) if bool(args.smoke_grid) else (-0.45, -0.30, -0.20, -0.12)
    for kind, seed in model_specs:
        clf, edge, worst, feature_cols, diag = _train(train_df, kind=kind, seed=seed)
        model_diags.append(diag)
        for p_min in p_grid:
            for edge_max in edge_grid:
                for worst_min in worst_grid:
                    configs.append(
                        {
                            "variant": f"autonomous_exit_{kind}_p{p_min:.2f}_e{edge_max:.4f}_w{worst_min:.2f}",
                            "clf": clf,
                            "edge": edge,
                            "worst": worst,
                            "feature_cols": feature_cols,
                            "p_min": float(p_min),
                            "edge_max": float(edge_max),
                            "worst_min": float(worst_min),
                        }
                    )

    rows: list[dict[str, Any]] = []
    ledgers: dict[int, dict[str, pd.DataFrame]] = {}
    for idx, cfg in enumerate(configs):
        print(json.dumps({"stage": "simulate_start", "idx": int(idx), "variant": cfg["variant"], "sec": round(time.time() - t0, 3)}), flush=True)
        row = {"variant_id": int(idx), "variant": str(cfg["variant"]), "p_min": float(cfg["p_min"]), "edge_max": float(cfg["edge_max"]), "worst_min": float(cfg["worst_min"])}
        ledgers[idx] = {}
        for split in ("validation", "oos"):
            metrics, ledger = _simulate(
                data[split],
                clf=cfg["clf"],
                edge=cfg["edge"],
                worst=cfg["worst"],
                feature_cols=list(cfg["feature_cols"]),
                p_min=float(cfg["p_min"]),
                edge_max=float(cfg["edge_max"]),
                worst_min=float(cfg["worst_min"]),
                cost_mult=FEE_COST_MULT,
            )
            row.update(_row(split, metrics))
            ledgers[idx][split] = ledger
        print(json.dumps({"stage": "simulate_done", "idx": int(idx), "variant": cfg["variant"], "oos_pnl": row["oos_pnl"], "oos_trades": row["oos_trades"], "sec": round(time.time() - t0, 3)}), flush=True)
        rows.append(row)

    ranking = pd.DataFrame(rows)
    baseline = ranking[ranking["variant"].eq("no_sltp_no_exit_model")].iloc[0]
    ranking["score"] = ranking["oos_pnl"] + 0.40 * ranking["validation_pnl"] + 0.25 * ranking["oos_mdd"] + 0.15 * ranking["validation_mdd"]
    ranking["delta_vs_no_exit_oos_pnl"] = ranking["oos_pnl"] - float(baseline["oos_pnl"])
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "autonomous_exit_no_sltp_ranking.csv", index=False)
    keep_ids = sorted(set([int(baseline["variant_id"])] + [int(x) for x in ranking["variant_id"].head(10).tolist()]))
    for variant_id in keep_ids:
        for split, ledger in ledgers[variant_id].items():
            ledger.to_csv(OUT_DIR / f"{split}_variant{variant_id}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "purpose": "Remove fixed SL/TP buckets from the current Omega research baseline and let a validation-trained in-position exit layer own exits.",
        "accounting": "Cost3 fee/slippage multiplier 3.0; true leverage exposure inherited from parent; TP/SL thresholds are set to zero before simulation.",
        "dataset_diag": dataset_diag,
        "model_diags": model_diags,
        "baseline_no_exit": baseline.to_dict(),
        "top": ranking.head(30).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "dataset": str(OUT_DIR / "validation_autonomous_exit_dataset.csv"),
            "ranking": str(OUT_DIR / "autonomous_exit_no_sltp_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top10": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
