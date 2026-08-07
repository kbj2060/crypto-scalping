#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WIN_SKLEARN_TARGET = Path("C:/Users/kbj20/AppData/Local/Temp/codex_pydeps_sklearn")
if WIN_SKLEARN_TARGET.exists() and str(WIN_SKLEARN_TARGET) not in sys.path:
    sys.path.insert(0, str(WIN_SKLEARN_TARGET))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_triple_barrier_hgb_smoke_20260619"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619"
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")

DENY_PREFIXES = ("clean_regime4_", "regime4_pred_", "regime3_pred_", "teacher_", "teacher_oof_", "a5dir_")
DENY_TOKENS = ("target", "future", "label", "pnl", "zigzag", "wave3", "tp_sl_action_score")
NON_FEATURE_COLS = {"timestamp"}

FEE_RATE = 0.0005
SLIP_RATE = 0.0002
COST_MULT = 3.0
NOTIONAL = 0.81
LEVERAGE = 2.0
TAKE_PROFIT = 0.052
STOP_LOSS = 0.028
MAX_HOLD_BARS = 192


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


def _read_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "timestamp" not in df.columns:
        raise RuntimeError(f"{path} missing timestamp")
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _forbidden_feature(col: str) -> bool:
    low = col.lower()
    return col.startswith(DENY_PREFIXES) or col.startswith("tb_") or col.startswith("entry_timestamp") or any(tok in low for tok in DENY_TOKENS)


def _feature_cols(train: pd.DataFrame, val: pd.DataFrame, oos: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in train.columns:
        if col in NON_FEATURE_COLS or col not in val.columns or col not in oos.columns:
            continue
        if _forbidden_feature(str(col)):
            continue
        if pd.api.types.is_numeric_dtype(train[col]) and pd.api.types.is_numeric_dtype(val[col]) and pd.api.types.is_numeric_dtype(oos[col]):
            cols.append(str(col))
    if len(cols) < 50:
        raise RuntimeError(f"too few smoke features: {len(cols)}")
    return cols


def _align_labels(frame: pd.DataFrame, split: str, label_col: str) -> pd.DataFrame:
    labels = pd.read_csv(LABEL_DIR / f"{split}_triple_barrier_labels.csv", parse_dates=["timestamp"], usecols=["timestamp", label_col])
    out = frame.merge(labels, on="timestamp", how="inner", validate="one_to_one")
    if out.empty:
        raise RuntimeError(f"{split}: empty frame/label intersection")
    return out.reset_index(drop=True)


def _matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.to_numpy(dtype=np.float32)


def _prediction_frame(frame: pd.DataFrame, pred: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    cls_idx = list(getattr(_prediction_frame, "classes_", [0, 1, 2]))
    out = pd.DataFrame({"timestamp": frame["timestamp"].to_numpy(), "pred_action": pred.astype(int)})
    for cls in (0, 1, 2):
        if cls in cls_idx:
            out[f"p_{cls}"] = proba[:, cls_idx.index(cls)]
        else:
            out[f"p_{cls}"] = 0.0
    out["confidence"] = out[["p_0", "p_1", "p_2"]].max(axis=1)
    return out


def _replay(frame: pd.DataFrame, pred: pd.DataFrame, threshold: float) -> dict[str, Any]:
    open_px = pd.to_numeric(frame["open"], errors="raise").to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    action = pred["pred_action"].to_numpy(dtype=np.int64)
    confidence = pred["confidence"].to_numpy(dtype=np.float64)
    fee = FEE_RATE * COST_MULT
    slip = SLIP_RATE * COST_MULT
    tp_move = TAKE_PROFIT / NOTIONAL
    sl_move = STOP_LOSS / NOTIONAL
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    reasons: dict[str, int] = {}
    i = 0
    while i < len(frame) - 2:
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
        side = int(action[i])
        if side == 0 or float(confidence[i]) < float(threshold):
            i += 1
            continue
        pos = 1 if side == 1 else -1
        entry_i = i + 1
        entry = float(open_px[entry_i]) * (1.0 + slip * pos)
        entry_equity = cash
        cash -= cash * fee * NOTIONAL
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        exit_i = min(len(frame) - 1, entry_i + MAX_HOLD_BARS)
        reason = "max_hold"
        exit_px = float(close[exit_i])
        for j in range(entry_i, exit_i + 1):
            if pos > 0:
                hit_sl = bool(low[j] <= entry * (1.0 - sl_move))
                hit_tp = bool(high[j] >= entry * (1.0 + tp_move))
                if hit_sl:
                    exit_px = entry * (1.0 - sl_move) * (1.0 - slip)
                    reason = "stop_loss"
                    exit_i = j
                    break
                if hit_tp:
                    exit_px = entry * (1.0 + tp_move) * (1.0 - slip)
                    reason = "take_profit"
                    exit_i = j
                    break
            else:
                hit_sl = bool(high[j] >= entry * (1.0 + sl_move))
                hit_tp = bool(low[j] <= entry * (1.0 - tp_move))
                if hit_sl:
                    exit_px = entry * (1.0 + sl_move) * (1.0 + slip)
                    reason = "stop_loss"
                    exit_i = j
                    break
                if hit_tp:
                    exit_px = entry * (1.0 - tp_move) * (1.0 + slip)
                    reason = "take_profit"
                    exit_i = j
                    break
        raw = (exit_px - entry) / max(entry, 1e-12) if pos > 0 else (entry - exit_px) / max(entry, 1e-12)
        before = cash
        cash *= 1.0 + raw * NOTIONAL
        cash -= before * fee * NOTIONAL
        trades += 1
        wins += int(cash > entry_equity)
        reasons[reason] = reasons.get(reason, 0) + 1
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
        i = max(exit_i + 1, i + 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "wr": float(wins / trades) if trades else 0.0,
        "trades": int(trades),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-config", default="h96_conservative")
    ap.add_argument("--thresholds", default="0.45,0.55,0.65,0.75,0.85")
    args = ap.parse_args()
    label_col = f"tb_action_{args.label_config}"
    out_dir = OUT_DIR / str(args.label_config)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_2025 = _read_frame(TRAIN_CSV)
    raw_2026 = _read_frame(EVAL_CSV)
    train = _align_labels(raw_2025.loc[pd.to_datetime(raw_2025["timestamp"]) < SPLIT_TS].reset_index(drop=True), "train", label_col)
    val = _align_labels(raw_2025.loc[pd.to_datetime(raw_2025["timestamp"]) >= SPLIT_TS].reset_index(drop=True), "validation", label_col)
    oos = _align_labels(raw_2026, "oos", label_col)
    cols = _feature_cols(train, val, oos)
    x_train = _matrix(train, cols)
    y_train = train[label_col].to_numpy(dtype=np.int64)
    x_val = _matrix(val, cols)
    y_val = val[label_col].to_numpy(dtype=np.int64)
    x_oos = _matrix(oos, cols)
    y_oos = oos[label_col].to_numpy(dtype=np.int64)

    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=260619,
    )
    model.fit(x_train, y_train)
    _prediction_frame.classes_ = list(model.classes_)
    val_pred = model.predict(x_val)
    oos_pred = model.predict(x_oos)
    val_proba = model.predict_proba(x_val)
    oos_proba = model.predict_proba(x_oos)
    val_pf = _prediction_frame(val, val_pred, val_proba)
    oos_pf = _prediction_frame(oos, oos_pred, oos_proba)
    val_pf.to_csv(out_dir / "validation_predictions.csv", index=False)
    oos_pf.to_csv(out_dir / "oos_predictions.csv", index=False)

    rows = []
    for threshold in [float(x) for x in str(args.thresholds).split(",") if x.strip()]:
        rows.append(
            {
                "candidate": f"hgb_{args.label_config}_thr{threshold:.2f}",
                "threshold": threshold,
                "val_bacc": float(balanced_accuracy_score(y_val, val_pred)),
                "oos_bacc": float(balanced_accuracy_score(y_oos, oos_pred)),
                **{f"val_{k}": v for k, v in _replay(val, val_pf, threshold).items()},
                **{f"oos_{k}": v for k, v in _replay(oos, oos_pf, threshold).items()},
            }
        )
    ranking = pd.DataFrame(rows).sort_values(["val_pnl", "oos_pnl"], ascending=[False, False]).reset_index(drop=True)
    ranking.to_csv(out_dir / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "status": "hgb_smoke_not_tabm",
        "label_config": str(args.label_config),
        "label_col": label_col,
        "feature_count": int(len(cols)),
        "forbidden_feature_audit": {"passed": True, "deny_prefixes": DENY_PREFIXES, "deny_tokens": DENY_TOKENS},
        "risk_contract": {
            "notional": NOTIONAL,
            "leverage": LEVERAGE,
            "take_profit_account": TAKE_PROFIT,
            "stop_loss_account": STOP_LOSS,
            "tp_price_move": TAKE_PROFIT / NOTIONAL,
            "sl_price_move": STOP_LOSS / NOTIONAL,
            "max_hold_bars": MAX_HOLD_BARS,
        },
        "class_counts": {
            "train": pd.Series(y_train).value_counts().sort_index().to_dict(),
            "validation": pd.Series(y_val).value_counts().sort_index().to_dict(),
            "oos": pd.Series(y_oos).value_counts().sort_index().to_dict(),
        },
        "best_by_validation": ranking.iloc[0].to_dict(),
        "ranking": ranking.to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(out_dir),
            "ranking": str(out_dir / "ranking.csv"),
            "validation_predictions": str(out_dir / "validation_predictions.csv"),
            "oos_predictions": str(out_dir / "oos_predictions.csv"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "best": report["best_by_validation"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
