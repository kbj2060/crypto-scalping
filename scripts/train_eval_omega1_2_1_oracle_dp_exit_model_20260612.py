#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402
import train_eval_omega1_2_1_tp_runner_meta_selector_20260610 as meta  # noqa: E402


MODEL_ID = "omega1_2_1_oracle_dp_exit_model_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
TP_RUNNER_BUNDLE = ROOT / "data/ensemble/supervised/omega1_2_1_tp_runner_meta_selector_20260610/tp_runner_meta_selector.joblib"

LAMBDA_GIVEBACK = 0.30
LAMBDA_TIME = 0.025
ORACLE_TOL_BARS = 2
MAX_PATH_BARS = 720
EARLY_DENSE_BARS = 48


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


def _simulate_tp_runner_path(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    tp_bundle: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    template = meta.RunnerTemplate(**tp_bundle["template"]) if tp_bundle else meta.TEMPLATES[0]
    cash = 1.0
    pos = base.Position()
    paths: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    trade_id = 0
    extensions = 0
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            cur.append({"i": int(i), "unreal": float(unreal), "pos": base.Position(**pos.__dict__)})
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
            if reason:
                close_pos = base.Position(**pos.__dict__)
                cash, pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                paths.append({"trade_id": int(trade_id), "reason": reason, "bars": cur, "extensions": int(extensions)})
                cur = []
                extensions = 0
            continue
        if bool(active[i]):
            cash, pos, entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
            if entered:
                trade_id += 1
                cur = []
                extensions = 0
    return paths


def _sample_mask(unreal: np.ndarray, utility: np.ndarray, oracle_idx: int, pos0: base.Position) -> np.ndarray:
    n = len(unreal)
    idx = np.arange(n)
    mfe = np.maximum.accumulate(unreal)
    giveback = np.maximum(0.0, mfe - unreal) / max(abs(float(pos0.stop_loss)), 1e-8)
    tp_progress = unreal / max(float(pos0.take_profit), 1e-8)
    sl_progress = -unreal / max(abs(float(pos0.stop_loss)), 1e-8)
    mask = (
        (idx < EARLY_DENSE_BARS)
        | (idx % 12 == 0)
        | (np.abs(idx - int(oracle_idx)) <= 4)
        | (tp_progress >= 0.55)
        | (sl_progress >= 0.45)
        | (giveback >= 0.35)
    )
    if n > MAX_PATH_BARS:
        mask &= idx < MAX_PATH_BARS
        mask |= np.abs(idx - int(min(oracle_idx, MAX_PATH_BARS - 1))) <= 4
    return mask


def _build_oracle_dataset(frame: pd.DataFrame, state: pd.DataFrame, paths: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        bars = path["bars"]
        if len(bars) < 3:
            continue
        if len(bars) > MAX_PATH_BARS:
            bars = bars[:MAX_PATH_BARS]
        unreal = np.asarray([b["unreal"] for b in bars], dtype=np.float64)
        pos0 = bars[0]["pos"]
        stop = max(abs(float(pos0.stop_loss)), 1e-8)
        r = unreal / stop
        mfe_path = np.maximum.accumulate(unreal)
        giveback_R = np.maximum(0.0, mfe_path - unreal) / stop
        age_norm = np.arange(len(unreal), dtype=np.float64) / max(float(len(unreal) - 1), 1.0)
        utility = r - float(LAMBDA_GIVEBACK) * giveback_R - float(LAMBDA_TIME) * age_norm
        oracle_idx = int(np.argmax(utility))
        mask = _sample_mask(unreal, utility, oracle_idx, pos0)
        for local_idx, keep in enumerate(mask):
            if not bool(keep):
                continue
            item = bars[local_idx]
            pos = item["pos"]
            i = int(item["i"])
            fwd = unreal[local_idx:]
            fwd_R = fwd / stop
            forward_mfe_R = float(np.max(fwd_R) - r[local_idx])
            forward_mae_R = float(np.min(fwd_R) - r[local_idx])
            near_oracle = abs(int(local_idx) - oracle_idx) <= int(ORACLE_TOL_BARS)
            late_oracle = int(local_idx) > oracle_idx + int(ORACLE_TOL_BARS)
            label_exit = int(near_oracle or late_oracle)
            rows.append(
                {
                    "trade_id": int(path["trade_id"]),
                    "decision_i": i,
                    "entry_i": int(pos.entry_i),
                    "entry_signal_i": int(pos.entry_signal_i),
                    "decision_time": str(frame["timestamp"].iloc[i]),
                    "entry_time": str(frame["timestamp"].iloc[int(pos.entry_signal_i)]),
                    "oracle_local_i": int(oracle_idx),
                    "local_i": int(local_idx),
                    "remaining_path_bars": int(len(unreal) - local_idx - 1),
                    "label_exit": label_exit,
                    "label_exit_soft": float(1.0 / (1.0 + np.exp(np.clip((oracle_idx - local_idx) / 3.0, -50, 50)))),
                    "oracle_utility": float(utility[oracle_idx]),
                    "current_utility": float(utility[local_idx]),
                    "utility_gap": float(utility[oracle_idx] - utility[local_idx]),
                    "forward_mfe_R": forward_mfe_R,
                    "forward_mae_R": forward_mae_R,
                    **_feature_row(frame, state, pos, i, float(item["unreal"])),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("empty oracle DP exit dataset")
    diag = {
        "rows": int(len(df)),
        "trades": int(df["trade_id"].nunique()),
        "positive": int(df["label_exit"].sum()),
        "positive_rate": float(df["label_exit"].mean()),
        "soft_mean": float(df["label_exit_soft"].mean()),
        "utility_gap_mean": float(df["utility_gap"].mean()),
        "forward_mfe_R_median": float(df["forward_mfe_R"].median()),
        "forward_mae_R_median": float(df["forward_mae_R"].median()),
    }
    return df, diag


def _train(df: pd.DataFrame, *, seed: int) -> tuple[Any, Any, Any, list[str], dict[str, Any]]:
    drop = {
        "trade_id",
        "decision_i",
        "entry_i",
        "entry_signal_i",
        "decision_time",
        "entry_time",
        "oracle_local_i",
        "local_i",
        "remaining_path_bars",
        "label_exit",
        "label_exit_soft",
        "oracle_utility",
        "current_utility",
        "utility_gap",
        "forward_mfe_R",
        "forward_mae_R",
    }
    feature_cols = [c for c in df.columns if c not in drop]
    x = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["label_exit"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError("single-class oracle exit labels")
    clf = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=8, min_samples_leaf=10, l2_regularization=1.0, learning_rate=0.04, random_state=int(seed))
    mfe = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=8, min_samples_leaf=10, l2_regularization=1.0, learning_rate=0.04, random_state=int(seed) + 1)
    mae = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=8, min_samples_leaf=10, l2_regularization=1.0, learning_rate=0.04, random_state=int(seed) + 2)
    clf.fit(x, y)
    mfe.fit(x, df["forward_mfe_R"].to_numpy(dtype=np.float64))
    mae.fit(x, df["forward_mae_R"].to_numpy(dtype=np.float64))
    return clf, mfe, mae, feature_cols, {"seed": int(seed), "rows": int(len(df)), "positive_rate": float(np.mean(y)), "feature_cols": feature_cols}


def _predict_exit(clf: Any, mfe_reg: Any, mae_reg: Any, feature_cols: list[str], feat: dict[str, float], *, p_min: float, mfe_max: float, mae_min: float) -> bool:
    x = np.asarray([[float(feat[c]) for c in feature_cols]], dtype=np.float64)
    p = float(clf.predict_proba(x)[0, 1])
    f_mfe = float(mfe_reg.predict(x)[0])
    f_mae = float(mae_reg.predict(x)[0])
    return bool(p >= float(p_min) and f_mfe <= float(mfe_max) and f_mae >= float(mae_min))


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
    mfe_reg: Any | None,
    mae_reg: Any | None,
    feature_cols: list[str],
    p_min: float,
    mfe_max: float,
    mae_min: float,
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
            elif clf is not None and max(int(i) - int(pos.entry_i), 0) >= 2:
                feat = _feature_row(frame, state, pos, i, unreal)
                if _predict_exit(clf, mfe_reg, mae_reg, feature_cols, feat, p_min=p_min, mfe_max=mfe_max, mae_min=mae_min):
                    reason = "oracle_model_exit"
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
    tp_bundle = joblib.load(TP_RUNNER_BUNDLE) if TP_RUNNER_BUNDLE.exists() else None
    paths = _simulate_tp_runner_path(
        data["validation"]["frame"],
        data["validation"]["dec"],
        data["validation"]["state"],
        fee=float(data["validation"]["fee"]),
        slip=float(data["validation"]["slip"]),
        cost_mult=3.0,
        tp_bundle=tp_bundle,
    )
    train_df, dataset_diag = _build_oracle_dataset(data["validation"]["frame"], data["validation"]["state"], paths)
    train_df.to_csv(OUT_DIR / "validation_oracle_dp_exit_dataset.csv", index=False)
    print(json.dumps({"stage": "dataset_done", **dataset_diag, "sec": round(time.time() - t0, 3)}), flush=True)
    clf, mfe_reg, mae_reg, feature_cols, model_diag = _train(train_df, seed=260613)
    configs: list[dict[str, Any]] = [
        {"variant": "baseline_no_runner", "tp_bundle": None, "clf": None, "mfe_reg": None, "mae_reg": None, "feature_cols": [], "p_min": 2.0, "mfe_max": -999.0, "mae_min": -999.0},
        {"variant": "tp_runner_only", "tp_bundle": tp_bundle, "clf": None, "mfe_reg": None, "mae_reg": None, "feature_cols": [], "p_min": 2.0, "mfe_max": -999.0, "mae_min": -999.0},
    ]
    for p_min, mfe_max, mae_min in (
        (0.55, 0.20, -1.20),
        (0.65, 0.15, -0.90),
        (0.75, 0.10, -0.70),
        (0.85, 0.06, -0.50),
    ):
        configs.append(
            {
                "variant": f"tp_runner_oracle_exit_hgb_p{p_min:.2f}_mfe{mfe_max:.2f}_mae{mae_min:.2f}",
                "tp_bundle": tp_bundle,
                "clf": clf,
                "mfe_reg": mfe_reg,
                "mae_reg": mae_reg,
                "feature_cols": feature_cols,
                "p_min": float(p_min),
                "mfe_max": float(mfe_max),
                "mae_min": float(mae_min),
            }
        )

    rows: list[dict[str, Any]] = []
    ledgers: dict[int, dict[str, pd.DataFrame]] = {}
    for idx, cfg in enumerate(configs):
        print(json.dumps({"stage": "simulate_start", "variant": cfg["variant"], "sec": round(time.time() - t0, 3)}), flush=True)
        row: dict[str, Any] = {"variant_id": int(idx), "variant": str(cfg["variant"]), "p_min": float(cfg["p_min"]), "mfe_max": float(cfg["mfe_max"]), "mae_min": float(cfg["mae_min"])}
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
                mfe_reg=cfg["mfe_reg"],
                mae_reg=cfg["mae_reg"],
                feature_cols=list(cfg["feature_cols"]),
                p_min=float(cfg["p_min"]),
                mfe_max=float(cfg["mfe_max"]),
                mae_min=float(cfg["mae_min"]),
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
    ranking.to_csv(OUT_DIR / "oracle_dp_exit_model_ranking.csv", index=False)
    for variant_id in sorted(set([0, 1] + [int(x) for x in ranking["variant_id"].head(5).tolist()])):
        for split, ledger in ledgers[variant_id].items():
            ledger.to_csv(OUT_DIR / f"{split}_variant{variant_id}_ledger.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "method": "Oracle DP optimal exit classification plus forward MFE/MAE regression. Entry/TP runner path is frozen; labels are built from in-position rows sampled around early, barrier-near, giveback, and oracle regions.",
        "dataset_diag": dataset_diag,
        "model_diag": model_diag,
        "top": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "dataset": str(OUT_DIR / "validation_oracle_dp_exit_dataset.csv"),
            "ranking": str(OUT_DIR / "oracle_dp_exit_model_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top10": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
