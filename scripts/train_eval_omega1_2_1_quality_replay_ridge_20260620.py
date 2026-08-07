#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_1_quality_replay_ridge_20260620"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

RAW_2025 = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
RAW_2026 = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
PARENT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"
TRAIN_PARENT = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_parent_train_inference_20260620/train_predictions_2025_jan_sep_true3head_in_sample.csv"

SPLIT_TS = pd.Timestamp("2025-10-01")
ACTION_CASH = 0
ACTION_LONG = 1
ACTION_SHORT = 2
FEE_RATE = 0.0005
SLIP_RATE = 0.0002
COST_MULT = 3.0
BASE_NOTIONAL = 0.45
BASE_TP = 0.026
BASE_SL = 0.014
EXPERT_SCALE = {"bull": 0.75, "bear": 0.90, "chop": 0.90, "chop_expert": 0.90}
MAX_LABEL_HORIZON = 384
RIDGE_ALPHA = 25.0
MAX_FEATURES = 160


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(type(obj).__name__)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=["timestamp"], low_memory=False).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _parent_predictions(split: str) -> tuple[pd.DataFrame, str]:
    if split == "train":
        return _read(TRAIN_PARENT), "omega1_regime3_expertdq_train_"
    if split == "validation":
        return _read(PARENT_DIR / "validation_predictions_2025_true3head.csv"), "omega1_regime3_expertdq_oof_"
    if split == "oos":
        return _read(PARENT_DIR / "oos_predictions_2026_true3head.csv"), "omega1_regime3_expertdq_"
    raise RuntimeError(split)


def _align(raw: pd.DataFrame, parent: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [
        f"{prefix}router_expert",
        f"{prefix}router_confidence",
        f"{prefix}router_margin",
        f"{prefix}dir_p_cash",
        f"{prefix}dir_p_long",
        f"{prefix}dir_p_short",
        f"{prefix}dir_confidence",
        f"{prefix}dir_side_edge",
        f"{prefix}dir_trade_prob",
        f"{prefix}dir_action",
        f"{prefix}quality_for_action",
        f"{prefix}final_action",
    ]
    missing = [c for c in cols if c not in parent.columns]
    if missing:
        raise RuntimeError(f"parent predictions missing columns: {missing}")
    merged = raw.merge(parent[["timestamp", *cols]], on="timestamp", how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError("empty raw/parent alignment")
    out = merged.rename(
        columns={
            f"{prefix}router_expert": "parent_router_expert",
            f"{prefix}router_confidence": "parent_router_confidence",
            f"{prefix}router_margin": "parent_router_margin",
            f"{prefix}dir_p_cash": "parent_dir_p_cash",
            f"{prefix}dir_p_long": "parent_dir_p_long",
            f"{prefix}dir_p_short": "parent_dir_p_short",
            f"{prefix}dir_confidence": "parent_dir_confidence",
            f"{prefix}dir_side_edge": "parent_dir_side_edge",
            f"{prefix}dir_trade_prob": "parent_dir_trade_prob",
            f"{prefix}dir_action": "parent_dir_action",
            f"{prefix}quality_for_action": "parent_old_quality",
            f"{prefix}final_action": "parent_old_final_action",
        }
    )
    return out.reset_index(drop=True)


def _side(action: int) -> int:
    if int(action) == ACTION_LONG:
        return 1
    if int(action) == ACTION_SHORT:
        return -1
    return 0


def _notional(expert: str) -> float:
    return BASE_NOTIONAL * EXPERT_SCALE.get(str(expert).replace("chop_expert", "chop"), 0.90)


def _replay_quality_labels(df: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    n = len(df)
    fee = FEE_RATE * COST_MULT
    slip = SLIP_RATE * COST_MULT
    rows: list[dict[str, Any]] = []
    action = pd.to_numeric(df["parent_dir_action"], errors="coerce").fillna(0).astype(int).to_numpy()
    expert = df["parent_router_expert"].astype(str).to_numpy()
    for i in range(n):
        side = _side(int(action[i]))
        notional = _notional(str(expert[i]))
        if side == 0 or i + 2 >= n:
            rows.append({"quality_target_net_return": 0.0, "quality_target_return_over_mae": 0.0, "quality_label_hold_bars": 0, "quality_label_reason": "cash"})
            continue
        entry_i = min(i + 1, n - 1)
        entry = float(close[entry_i]) * (1.0 + slip if side > 0 else 1.0 - slip)
        end_i = min(n - 1, entry_i + MAX_LABEL_HORIZON)
        exit_i = end_i
        reason = "vertical"
        mfe = 0.0
        mae = 0.0
        exit_px = float(close[end_i])
        for j in range(entry_i, end_i + 1):
            px = float(close[j])
            raw = (px * (1.0 - slip) - entry) / max(entry, 1e-12) if side > 0 else (entry - px * (1.0 + slip)) / max(entry, 1e-12)
            pnl = raw * notional
            mfe = max(mfe, pnl)
            mae = min(mae, pnl)
            if pnl >= BASE_TP:
                exit_i = j
                exit_px = px
                reason = "take_profit"
                break
            if pnl <= -abs(BASE_SL):
                exit_i = j
                exit_px = px
                reason = "stop_loss"
                break
        raw_exit = (exit_px * (1.0 - slip) - entry) / max(entry, 1e-12) if side > 0 else (entry - exit_px * (1.0 + slip)) / max(entry, 1e-12)
        net = raw_exit * notional - 2.0 * fee * notional
        rows.append(
            {
                "quality_target_net_return": float(net),
                "quality_target_return_over_mae": float(net / max(abs(mae), 1e-6)),
                "quality_label_hold_bars": int(exit_i - entry_i),
                "quality_label_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _feature_cols(train: pd.DataFrame, *others: pd.DataFrame) -> list[str]:
    banned_tokens = ("target", "future", "label", "pnl", "zigzag", "wave3", "timestamp")
    banned_prefixes = ("clean_regime4_", "regime4_pred_", "teacher_", "teacher_oof_")
    banned_exact = {"tp_sl_action_score", "execution_quality", "parent_old_quality", "parent_old_final_action"}
    common = [c for c in train.columns if all(c in other.columns for other in others)]
    cols: list[str] = []
    for col in common:
        low = col.lower()
        if col in banned_exact:
            continue
        if any(tok in low for tok in banned_tokens):
            continue
        if any(col.startswith(prefix) for prefix in banned_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(train[col]) and all(pd.api.types.is_numeric_dtype(other[col]) for other in others):
            cols.append(col)
    if len(cols) > MAX_FEATURES:
        variances = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).var().sort_values(ascending=False)
        cols = list(variances.head(MAX_FEATURES).index)
    return cols


def _fit_ridge(train: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    active = pd.to_numeric(train["parent_dir_action"], errors="coerce").fillna(0).astype(int) != ACTION_CASH
    y_raw = pd.to_numeric(train.loc[active, "quality_target_net_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    q01, q99 = np.quantile(y_raw, [0.01, 0.99])
    y = np.clip(y_raw, q01, q99)
    x = train.loc[active, cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    xz = (x - mean) / std
    xz = np.column_stack([np.ones(len(xz)), xz])
    util_w = 1.0 + np.clip(np.abs(y), 0.0, 0.03) * 25.0
    xtw = xz.T * util_w
    reg = np.eye(xz.shape[1]) * RIDGE_ALPHA
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xtw @ xz + reg, xtw @ y)
    pred = xz @ beta
    corr = float(np.corrcoef(pred, y)[0, 1]) if len(y) > 3 and np.std(pred) > 0 and np.std(y) > 0 else 0.0
    return {"columns": cols, "mean": mean, "std": std, "beta": beta, "clip": [float(q01), float(q99)], "train_active_rows": int(active.sum()), "train_corr": corr}


def _predict_quality(df: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    x = df[model["columns"]].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    xz = (x - model["mean"]) / model["std"]
    xz = np.column_stack([np.ones(len(xz)), xz])
    return (xz @ model["beta"]).astype(np.float64)


def _backtest(df: pd.DataFrame, quality: np.ndarray, *, threshold: float, scale_notional: bool) -> dict[str, Any]:
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    action = pd.to_numeric(df["parent_dir_action"], errors="coerce").fillna(0).astype(int).to_numpy()
    expert = df["parent_router_expert"].astype(str).to_numpy()
    fee = FEE_RATE * COST_MULT
    slip = SLIP_RATE * COST_MULT
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    reasons: dict[str, int] = {}
    i = 0
    while i < len(df) - 2:
        eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        side = _side(int(action[i]))
        if side == 0 or float(quality[i]) < float(threshold):
            i += 1
            continue
        entry_i = i + 1
        entry = float(close[entry_i]) * (1.0 + slip if side > 0 else 1.0 - slip)
        notional = _notional(str(expert[i]))
        if scale_notional:
            rank_scale = float(np.clip((quality[i] - threshold) / max(0.02, abs(threshold) + 0.02), 0.30, 1.25))
            notional *= rank_scale
        cash -= cash * fee * notional
        entry_cash = cash
        exit_i = len(df) - 1
        exit_px = float(close[exit_i])
        reason = "forced_end"
        for j in range(entry_i, len(df) - 1):
            px = float(close[j])
            raw = (px * (1.0 - slip) - entry) / max(entry, 1e-12) if side > 0 else (entry - px * (1.0 + slip)) / max(entry, 1e-12)
            pnl = raw * notional
            peak = max(peak, cash * (1.0 + pnl))
            mdd = min(mdd, cash * (1.0 + pnl) / max(peak, 1e-12) - 1.0)
            if pnl >= BASE_TP:
                exit_i = j
                exit_px = px
                reason = "take_profit"
                break
            if pnl <= -abs(BASE_SL):
                exit_i = j
                exit_px = px
                reason = "stop_loss"
                break
        raw_exit = (exit_px * (1.0 - slip) - entry) / max(entry, 1e-12) if side > 0 else (entry - exit_px * (1.0 + slip)) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee * notional
        trades += 1
        wins += int(cash > entry_cash)
        long_entries += int(side > 0)
        short_entries += int(side < 0)
        reasons[reason] = reasons.get(reason, 0) + 1
        i = max(exit_i + 1, i + 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
    }


def _metric_row(name: str, val: dict[str, Any], oos: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": name,
        **cfg,
        "validation_pnl": val["pnl"],
        "validation_mdd": val["mdd"],
        "validation_trades": val["trades"],
        "validation_wr": val["wr"],
        "oos_pnl": oos["pnl"],
        "oos_mdd": oos["mdd"],
        "oos_trades": oos["trades"],
        "oos_wr": oos["wr"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read(RAW_2025)
    raw_2026 = _read(RAW_2026)
    train_parent, train_prefix = _parent_predictions("train")
    val_parent, val_prefix = _parent_predictions("validation")
    oos_parent, oos_prefix = _parent_predictions("oos")
    train = _align(raw_2025[raw_2025["timestamp"] < SPLIT_TS].reset_index(drop=True), train_parent, train_prefix)
    val = _align(raw_2025[raw_2025["timestamp"] >= SPLIT_TS].reset_index(drop=True), val_parent, val_prefix)
    oos = _align(raw_2026, oos_parent, oos_prefix)
    train_labels = _replay_quality_labels(train)
    val_labels = _replay_quality_labels(val)
    oos_labels = _replay_quality_labels(oos)
    train = pd.concat([train, train_labels], axis=1)
    val = pd.concat([val, val_labels], axis=1)
    oos = pd.concat([oos, oos_labels], axis=1)
    cols = _feature_cols(train, val, oos)
    model = _fit_ridge(train, cols)
    val_q = _predict_quality(val, model)
    oos_q = _predict_quality(oos, model)
    val["quality_replay_pred"] = val_q
    oos["quality_replay_pred"] = oos_q
    thresholds = [float(x) for x in np.quantile(val_q, [0.50, 0.60, 0.70, 0.80, 0.90])]
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for scale in (False, True):
        for thr in thresholds:
            cfg = {"threshold": float(thr), "scale_notional": bool(scale)}
            name = f"{'scaled' if scale else 'gate'}_q{int(round((val_q >= thr).mean() * 100)):02d}_thr{thr:.6f}".replace("-", "m").replace(".", "p")
            val_m = _backtest(val, val_q, threshold=thr, scale_notional=scale)
            oos_m = _backtest(oos, oos_q, threshold=thr, scale_notional=scale)
            reports[name] = {"validation": val_m, "oos": oos_m, "config": cfg}
            rows.append(_metric_row(name, val_m, oos_m, cfg))
    ranking = pd.DataFrame(rows).sort_values(["validation_pnl", "validation_mdd", "validation_wr"], ascending=[False, False, False]).reset_index(drop=True)
    train[["timestamp", "parent_dir_action", "parent_router_expert", "quality_target_net_return", "quality_target_return_over_mae", "quality_label_hold_bars", "quality_label_reason"]].to_csv(
        OUT_DIR / "train_quality_replay_labels.csv", index=False
    )
    val[["timestamp", "parent_dir_action", "parent_router_expert", "quality_target_net_return", "quality_replay_pred", "quality_label_hold_bars", "quality_label_reason"]].to_csv(
        OUT_DIR / "validation_quality_predictions.csv", index=False
    )
    oos[["timestamp", "parent_dir_action", "parent_router_expert", "quality_target_net_return", "quality_replay_pred", "quality_label_hold_bars", "quality_label_reason"]].to_csv(
        OUT_DIR / "oos_quality_predictions.csv", index=False
    )
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "purpose": "Step-1 quality-label replacement probe. Keep parent direction/actions fixed; replace quality semantics with barrier-replay cost-included net-return regression target.",
        "label_contract": {
            "side_source": "parent_dir_action from current 3-head parent predictions",
            "target": "quality_target_net_return",
            "replay": "fixed TP/SL close-path barrier replay with fee/slippage cost included",
            "tp_account_pnl": BASE_TP,
            "sl_account_pnl": BASE_SL,
            "max_label_horizon_bars": MAX_LABEL_HORIZON,
            "note": "This is a lightweight probe. Full parent retrain should use OOF direction-side labels for train labels.",
        },
        "ridge_model": {k: v for k, v in model.items() if k not in {"mean", "std", "beta"}},
        "label_audit": {
            "train_rows": int(len(train)),
            "train_active_rows": int((train["parent_dir_action"].astype(int) != ACTION_CASH).sum()),
            "train_target_mean": float(train["quality_target_net_return"].mean()),
            "train_target_positive_rate": float((train["quality_target_net_return"] > 0.0).mean()),
            "validation_target_mean": float(val["quality_target_net_return"].mean()),
            "validation_target_positive_rate": float((val["quality_target_net_return"] > 0.0).mean()),
            "oos_target_mean": float(oos["quality_target_net_return"].mean()),
            "oos_target_positive_rate": float((oos["quality_target_net_return"] > 0.0).mean()),
        },
        "ranking_by_validation_pnl": ranking.to_dict(orient="records"),
        "results": reports,
        "artifacts": {
            "out_dir": OUT_DIR,
            "ranking": OUT_DIR / "ranking.csv",
            "report": OUT_DIR / "report.json",
            "train_labels": OUT_DIR / "train_quality_replay_labels.csv",
            "validation_predictions": OUT_DIR / "validation_quality_predictions.csv",
            "oos_predictions": OUT_DIR / "oos_quality_predictions.csv",
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top5": ranking.head(5).to_dict(orient="records"), "label_audit": report["label_audit"]}, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
