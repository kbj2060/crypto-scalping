#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_omega3_full_distill_residual_regularized_loop17_20260619 as loop17  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as ex  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega3_lifecycle_exit_no_price_barrier_loop21_20260619"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
CURRENT = {
    "validation": {"pnl": 100.542729421, "mdd": -10.677653, "trades": 33, "wr": 0.636364},
    "oos": {"pnl": 72.760041481, "mdd": -8.108171, "trades": 18, "wr": 0.722222},
}
MAX_AGE_BARS = 192
LABEL_FWD_BARS = 48
MIN_HOLD_BARS = 2


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


def _learned_entry_decision(
    dec0: pd.DataFrame,
    x: pd.DataFrame,
    models: dict[str, HistGradientBoostingRegressor],
    *,
    notional_scale: float = 1.003,
    cap: float = 0.81,
    max_age: int = MAX_AGE_BARS,
    disable_price_barriers: bool = True,
) -> pd.DataFrame:
    out = loop17._base_distill(dec0, x, models, notional_scale=notional_scale, tp_scale=1.0, sl_scale=1.0, cap=cap)
    active = np.flatnonzero(omega._active(out))
    if len(active):
        if bool(disable_price_barriers):
            out.loc[active, "take_profit"] = 0.0
            out.loc[active, "stop_loss"] = 0.0
        out.loc[active, "max_hold_bars"] = int(max_age)
        out.loc[active, "cooldown_bars"] = 0
    return out


def _fit_risk_models(x_train: pd.DataFrame, train_teacher: pd.DataFrame, seed: int) -> dict[str, HistGradientBoostingRegressor]:
    active = np.flatnonzero(omega._active(train_teacher))
    return {
        "notional": loop17._fit_regressor(x_train, active, pd.to_numeric(train_teacher.loc[active, "notional_exposure"]).to_numpy(dtype=np.float64), seed + 1),
        "take_profit": loop17._fit_regressor(x_train, active, pd.to_numeric(train_teacher.loc[active, "take_profit"]).to_numpy(dtype=np.float64), seed + 2),
        "stop_loss": loop17._fit_regressor(x_train, active, pd.to_numeric(train_teacher.loc[active, "stop_loss"]).to_numpy(dtype=np.float64), seed + 3),
    }


def _market_feature_cache(frame: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(frame["close"], errors="raise")
    high = pd.to_numeric(frame["high"], errors="raise")
    low = pd.to_numeric(frame["low"], errors="raise")
    open_ = pd.to_numeric(frame["open"], errors="raise")
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    out = pd.DataFrame(index=frame.index)
    out["bar_range_pct"] = ((high - low) / close).replace([np.inf, -np.inf], np.nan)
    out["body_pct"] = ((close - open_) / close).replace([np.inf, -np.inf], np.nan)
    out["atr14_pct"] = (atr / close).replace([np.inf, -np.inf], np.nan)
    for lag in (1, 3, 6, 12, 24):
        out[f"ret_{lag}"] = close.pct_change(lag).replace([np.inf, -np.inf], np.nan)
    for win in (6, 12, 24):
        out[f"ret_vol_{win}"] = ret.rolling(win, min_periods=max(3, win // 3)).std()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _pos_features(
    frame: pd.DataFrame,
    market: pd.DataFrame,
    x_state: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    active: np.ndarray,
    pos: ex.Position,
    i: int,
    unreal: float,
) -> dict[str, float]:
    px = float(arrays["close"][int(i)])
    raw = (px - float(pos.entry_price)) / max(float(pos.entry_price), 1.0e-12) if int(pos.side) > 0 else (float(pos.entry_price) - px) / max(float(pos.entry_price), 1.0e-12)
    mfe = max(float(pos.mfe), float(unreal))
    mae = min(float(pos.mae), float(unreal))
    hold = max(int(i) - int(pos.entry_i), 0)
    giveback = max(0.0, mfe - float(unreal))
    row = {
        "side": float(pos.side),
        "notional": float(pos.notional),
        "raw_return": float(raw),
        "unreal": float(unreal),
        "mfe": float(mfe),
        "mae": float(mae),
        "giveback": float(giveback),
        "giveback_ratio": float(giveback / max(abs(mfe), 1.0e-8)) if mfe > 0.0 else 0.0,
        "hold_bars": float(hold),
        "hold_tanh_24": float(np.tanh(hold / 24.0)),
        "hold_tanh_96": float(np.tanh(hold / 96.0)),
        "age_frac": float(hold / max(int(MAX_AGE_BARS), 1)),
        "parent_active_now": 1.0 if bool(active[int(i)]) else 0.0,
        "entry_age_parent_active": 1.0 if bool(active[max(0, int(pos.entry_signal_i))]) else 0.0,
    }
    for col in market.columns:
        row[f"mkt_{col}"] = float(market.iloc[int(i)][col])
    for col in (
        "tabm_quality_for_action",
        "tabm_router_confidence",
        "tabm_router_margin",
        "tabm_dir_confidence",
        "tabm_dir_side_edge",
        "tabm_dir_trade_prob",
        "primary_active_roll_12",
        "primary_active_roll_48",
        "primary_cash_streak",
    ):
        if col in x_state.columns:
            row[f"state_{col}"] = float(x_state.iloc[int(i)][col])
    return row


def _future_edge(arrays: dict[str, np.ndarray], pos: ex.Position, i: int, *, slip_eff: float) -> dict[str, float]:
    cur = ex._unreal(arrays, pos, int(i), slip_eff)
    last = min(len(arrays["close"]) - 2, int(i) + LABEL_FWD_BARS, int(pos.entry_i) + int(MAX_AGE_BARS))
    if last <= int(i):
        return {"edge": 0.0, "future_worst": float(cur), "future_best": float(cur), "exit": 1.0}
    vals = np.asarray([ex._unreal(arrays, pos, j, slip_eff) for j in range(int(i), last + 1)], dtype=np.float64)
    ages = np.arange(int(i), last + 1, dtype=np.float64) - float(pos.entry_i)
    mfe_path = np.maximum.accumulate(np.maximum(vals, float(pos.mfe)))
    giveback = np.maximum(0.0, mfe_path - vals)
    utility = vals - 0.35 * giveback - 0.00003 * ages
    now = float(utility[0])
    best_future = float(np.max(utility[1:])) if len(utility) > 1 else now
    edge = best_future - now
    hold = max(int(i) - int(pos.entry_i), 0)
    # Exit when waiting has little expected utility or when future path exposes large extra downside.
    future_worst = float(np.min(vals))
    future_best = float(np.max(vals))
    exit_now = bool(hold >= MIN_HOLD_BARS and (edge <= 0.0015 or future_worst < float(cur) - 0.018))
    return {"edge": float(edge), "future_worst": future_worst, "future_best": future_best, "exit": float(exit_now)}


def _collect_lifecycle_dataset(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    x_state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    start: int,
    end: int,
    stride: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work_frame = frame.reset_index(drop=True)
    arrays = ex._arrays(work_frame)
    active = np.asarray(omega._active(dec), dtype=bool)
    market = _market_feature_cache(work_frame)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    pos = ex.Position()
    rows: list[dict[str, float]] = []
    trade_id = 0
    for i in range(max(0, int(start)), min(int(end), len(work_frame) - 2)):
        if pos.side != 0:
            unreal = ex._unreal(arrays, pos, int(i), slip_eff)
            pos.mfe = max(float(pos.mfe), float(unreal))
            pos.mae = min(float(pos.mae), float(unreal))
            hold = max(int(i) - int(pos.entry_i), 0)
            if hold % int(stride) == 0:
                rows.append({
                    "trade_id": float(trade_id),
                    "i": float(i),
                    **_pos_features(work_frame, market, x_state, arrays, active, pos, int(i), unreal),
                    **_future_edge(arrays, pos, int(i), slip_eff=slip_eff),
                })
            if pos.take_profit > 0.0 and unreal >= float(pos.take_profit):
                cash, pos, _ = ex._close_fraction(cash, arrays, pos, int(i), 1.0, fee_eff, slip_eff)
            elif pos.stop_loss > 0.0 and unreal <= -abs(float(pos.stop_loss)):
                cash, pos, _ = ex._close_fraction(cash, arrays, pos, int(i), 1.0, fee_eff, slip_eff)
            elif hold >= int(MAX_AGE_BARS):
                cash, pos, _ = ex._close_fraction(cash, arrays, pos, int(i), 1.0, fee_eff, slip_eff)
            continue
        if bool(active[i]):
            cash, pos, entered = ex._enter(cash, arrays, dec, int(i), fee_eff, slip_eff)
            if entered:
                trade_id += 1
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if df.empty:
        raise RuntimeError("empty lifecycle dataset")
    diag = {
        "rows": int(len(df)),
        "trades": int(df["trade_id"].nunique()),
        "exit_rate": float(df["exit"].mean()),
        "edge_mean": float(df["edge"].mean()),
        "future_worst_mean": float(df["future_worst"].mean()),
    }
    return df, diag


def _train_exit_model(df: pd.DataFrame, seed: int, *, linear: bool = False) -> tuple[Any, Any, list[str], dict[str, Any]]:
    drop = {"trade_id", "i", "edge", "future_worst", "future_best", "exit"}
    feature_cols = [c for c in df.columns if c not in drop]
    y = df["exit"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError("single-class lifecycle exit labels")
    x = df[feature_cols].to_numpy(dtype=np.float64)
    if linear:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.35, class_weight="balanced", max_iter=500, random_state=int(seed)),
        )
        edge = make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=int(seed) + 1))
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.035,
            max_leaf_nodes=9,
            min_samples_leaf=25,
            l2_regularization=1.5,
            random_state=int(seed),
        )
        edge = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.035,
            max_leaf_nodes=9,
            min_samples_leaf=25,
            l2_regularization=1.5,
            random_state=int(seed) + 1,
        )
    clf.fit(x, y)
    edge.fit(x, df["edge"].to_numpy(dtype=np.float64))
    diag = {"feature_count": int(len(feature_cols)), "rows": int(len(df)), "positive_rate": float(np.mean(y)), "linear": bool(linear)}
    return clf, edge, feature_cols, diag


def _simulate_lifecycle(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    x_state: pd.DataFrame,
    *,
    clf: HistGradientBoostingClassifier | None,
    edge_model: HistGradientBoostingRegressor | None,
    feature_cols: list[str],
    p_exit: float,
    edge_max: float,
    min_hold: int,
    fee: float,
    slip: float,
    cost_mult: float,
) -> dict[str, Any]:
    frame = frame.reset_index(drop=True)
    arrays = ex._arrays(frame)
    active = np.asarray(omega._active(dec), dtype=bool)
    market = _market_feature_cache(frame)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = ex.Position()
    trades = 0
    wins = 0
    reasons: dict[str, int] = {}
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    model_exits = 0
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = ex._unreal(arrays, pos, int(i), slip_eff)
            pos.mfe = max(float(pos.mfe), float(unreal))
            pos.mae = min(float(pos.mae), float(unreal))
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)
            hold = max(int(i) - int(pos.entry_i), 0)
            reason = ""
            if pos.take_profit > 0.0 and unreal >= float(pos.take_profit):
                reason = "take_profit"
            elif pos.stop_loss > 0.0 and unreal <= -abs(float(pos.stop_loss)):
                reason = "stop_loss"
            elif hold >= int(MAX_AGE_BARS):
                reason = "max_age"
            elif clf is not None and hold >= int(min_hold):
                feat = _pos_features(frame, market, x_state, arrays, active, pos, int(i), unreal)
                x = np.asarray([[float(feat.get(c, 0.0)) for c in feature_cols]], dtype=np.float64)
                proba = clf.predict_proba(x)[0]
                classes = list(clf.classes_)
                p = float(proba[classes.index(1)]) if 1 in classes else 0.0
                e = float(edge_model.predict(x)[0]) if edge_model is not None else 0.0
                if p >= float(p_exit) and e <= float(edge_max):
                    reason = "learned_exit"
            if reason:
                entry_equity = float(pos.entry_equity)
                cash, pos, _ = ex._close_fraction(cash, arrays, pos, int(i), 1.0, fee_eff, slip_eff)
                trades += 1
                wins += int(cash > entry_equity)
                reasons[reason] = reasons.get(reason, 0) + 1
                model_exits += int(reason == "learned_exit")
            continue
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1.0e-12) - 1.0)
        if bool(active[i]):
            side = int(dec.iloc[int(i)].get("side", 0) or 0)
            cash, pos, entered = ex._enter(cash, arrays, dec, int(i), fee_eff, slip_eff)
            if entered:
                long_entries += int(side > 0)
                short_entries += int(side < 0)
                notional_sum += float(pos.notional)
    if pos.side != 0:
        entry_equity = float(pos.entry_equity)
        cash, pos, _ = ex._close_fraction(cash, arrays, pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        trades += 1
        wins += int(cash > entry_equity)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
    entries = max(long_entries + short_entries, 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "avg_notional": float(notional_sum / entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "model_exits": int(model_exits),
        "exit_reasons": dict(reasons),
    }


def _pack(candidate: str, cfg: dict[str, Any], cal_m: dict[str, Any], val_m: dict[str, Any], oos_m: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, **cfg}
    for prefix, metrics in (("cal", cal_m), ("val", val_m), ("oos", oos_m)):
        row[f"{prefix}_pnl"] = float(metrics["pnl"])
        row[f"{prefix}_mdd"] = float(metrics["mdd"])
        row[f"{prefix}_wr"] = float(metrics["wr"])
        row[f"{prefix}_trades"] = int(metrics["trades"])
        row[f"{prefix}_avg_notional"] = float(metrics.get("avg_notional", 0.0))
        row[f"{prefix}_model_exits"] = int(metrics.get("model_exits", 0))
        row[f"{prefix}_reasons"] = dict(metrics.get("exit_reasons", {}))
    row["val_delta"] = float(row["val_pnl"] - CURRENT["validation"]["pnl"])
    row["oos_delta"] = float(row["oos_pnl"] - CURRENT["oos"]["pnl"])
    row["strict_pass"] = bool(
        row["val_pnl"] >= CURRENT["validation"]["pnl"]
        and row["oos_pnl"] >= CURRENT["oos"]["pnl"]
        and row["val_mdd"] >= CURRENT["validation"]["mdd"]
        and row["oos_mdd"] >= CURRENT["oos"]["mdd"]
        and row["val_trades"] == CURRENT["validation"]["trades"]
        and row["oos_trades"] == CURRENT["oos"]["trades"]
    )
    row["cal_score"] = float(row["cal_pnl"] + 2.0 * row["cal_mdd"] - 1.0 * max(0, row["cal_trades"] - cfg["cal_current_trades"]))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=260821)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--max-train-rows", type=int, default=30000)
    ap.add_argument("--keep-learned-barriers", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    train_frame, _train_src, train_dec0, train_teacher, x_train = loop17._split(frames, "train", device)
    val_frame, _val_src, val_dec0, val_teacher, x_val = loop17._split(frames, "validation", device)
    oos_frame, _oos_src, oos_dec0, oos_teacher, x_oos = loop17._split(frames, "oos", device)
    risk_models = _fit_risk_models(x_train, train_teacher, int(args.seed))
    train_dec = _learned_entry_decision(train_dec0, x_train, risk_models, disable_price_barriers=not bool(args.keep_learned_barriers))
    val_dec = _learned_entry_decision(val_dec0, x_val, risk_models, disable_price_barriers=not bool(args.keep_learned_barriers))
    oos_dec = _learned_entry_decision(oos_dec0, x_oos, risk_models, disable_price_barriers=not bool(args.keep_learned_barriers))
    cal_n = min(45000, len(train_frame) // 3)
    train_end = len(train_frame) - cal_n
    dataset_name = "lifecycle_exit_train_dataset_hybrid_barriers.csv" if bool(args.keep_learned_barriers) else "lifecycle_exit_train_dataset.csv"
    dataset_path = OUT_DIR / dataset_name
    if dataset_path.exists():
        train_df = pd.read_csv(dataset_path)
        dataset_diag = {
            "rows": int(len(train_df)),
            "trades": int(train_df["trade_id"].nunique()),
            "exit_rate": float(train_df["exit"].mean()),
            "edge_mean": float(train_df["edge"].mean()),
            "future_worst_mean": float(train_df["future_worst"].mean()),
            "reused": True,
        }
    else:
        train_df, dataset_diag = _collect_lifecycle_dataset(
            train_frame,
            train_dec,
            x_train,
            fee=fee,
            slip=slip,
            cost_mult=3.0,
            start=0,
            end=train_end,
            stride=int(args.stride),
        )
        train_df.to_csv(dataset_path, index=False)
    if int(args.max_train_rows) > 0 and len(train_df) > int(args.max_train_rows):
        train_df = train_df.sample(n=int(args.max_train_rows), random_state=int(args.seed)).sort_index().reset_index(drop=True)
        dataset_diag["sampled_rows_for_training"] = int(len(train_df))
    clf, edge_model, feature_cols, train_diag = _train_exit_model(train_df, int(args.seed), linear=bool(args.linear))
    cal_frame = train_frame.iloc[train_end:].reset_index(drop=True)
    cal_dec = train_dec.iloc[train_end:].reset_index(drop=True)
    cal_x = x_train.iloc[train_end:].reset_index(drop=True)
    cal_teacher = train_teacher.iloc[train_end:].reset_index(drop=True)
    cal_current = omega._metrics(cal_frame, cal_teacher, fee=fee, slip=slip, cost_mult=3.0)
    rows: list[dict[str, Any]] = []
    p_grid = (0.65, 0.75) if bool(args.fast) else (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
    edge_grid = (0.0015,) if bool(args.fast) else (0.0005, 0.001, 0.002, 0.004)
    hold_grid = (4, 12) if bool(args.fast) else (2, 4, 8, 12)
    for p_exit in p_grid:
        for edge_max in edge_grid:
            for min_hold in hold_grid:
                cfg = {
                    "p_exit": float(p_exit),
                    "edge_max": float(edge_max),
                    "min_hold": int(min_hold),
                    "max_age_bars": int(MAX_AGE_BARS),
                    "cal_current_trades": int(cal_current["trades"]),
                }
                cal_m = _simulate_lifecycle(cal_frame, cal_dec, cal_x, clf=clf, edge_model=edge_model, feature_cols=feature_cols, p_exit=p_exit, edge_max=edge_max, min_hold=min_hold, fee=fee, slip=slip, cost_mult=3.0)
                val_m = _simulate_lifecycle(val_frame, val_dec, x_val, clf=clf, edge_model=edge_model, feature_cols=feature_cols, p_exit=p_exit, edge_max=edge_max, min_hold=min_hold, fee=fee, slip=slip, cost_mult=3.0)
                oos_m = _simulate_lifecycle(oos_frame, oos_dec, x_oos, clf=clf, edge_model=edge_model, feature_cols=feature_cols, p_exit=p_exit, edge_max=edge_max, min_hold=min_hold, fee=fee, slip=slip, cost_mult=3.0)
                rows.append(_pack("lifecycle_exit_no_price_barrier", cfg, cal_m, val_m, oos_m))
    ranking = pd.DataFrame(rows).sort_values(["cal_score", "cal_pnl"], ascending=[False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    report = {
        "model_id": MODEL_ID,
        "design": "Entry and notional use learned full-risk distillation. In no-barrier mode take_profit/stop_loss are disabled; in hybrid mode learned TP/SL barriers are retained and a train-only lifecycle classifier can early-exit before them.",
        "selection": "selected by trailing train calibration score only; official validation/OOS are post-selection evaluation.",
        "current_baseline": CURRENT,
        "calibration_current_teacher": cal_current,
        "dataset_diag": dataset_diag,
        "train_diag": train_diag,
        "keep_learned_barriers": bool(args.keep_learned_barriers),
        "selected_by_train_calibration": selected,
        "best_validation_diagnostic": ranking.sort_values(["val_pnl", "val_mdd"], ascending=[False, False]).iloc[0].to_dict(),
        "strict_pass_count": int(ranking["strict_pass"].sum()),
        "top": ranking.head(30).to_dict(orient="records"),
        "artifacts": {"out": str(OUT_DIR), "dataset": str(OUT_DIR / "lifecycle_exit_train_dataset.csv"), "ranking": str(OUT_DIR / "ranking.csv")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected, "strict_pass_count": report["strict_pass_count"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
