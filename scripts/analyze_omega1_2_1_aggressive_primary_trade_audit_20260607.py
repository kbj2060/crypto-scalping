#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as exposure  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402


MODEL_ID = "omega1_2_1_aggressive_primary_trade_audit_20260607"
BASELINE_ID = "omega1_2_1_aggressive_compensated_scale200_cap090"
LEDGER_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_1_current_baseline_growth_20260606"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

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


def _audit_columns(cols: list[str], *, where: str) -> None:
    bad = [c for c in cols if c.startswith(FORBIDDEN_PREFIXES) or c in FORBIDDEN_EXACT]
    if bad:
        raise RuntimeError(f"{where}: forbidden feature columns detected: {bad[:40]}")


def _ledger_path(split: str) -> Path:
    path = LEDGER_DIR / f"{BASELINE_ID}_{split}_trade_ledger_20260606.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _context_frame(frame: pd.DataFrame, src: pd.DataFrame, dec: pd.DataFrame, prefix: str) -> pd.DataFrame:
    aggressive_dec = sleeve._apply_aggressive(dec)
    features = exposure._feature_frame(frame, src, aggressive_dec, prefix)
    _audit_columns(list(features.columns), where="feature_frame")

    out = pd.DataFrame(index=frame.index)
    out["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    out["close"] = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    out["high"] = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    out["low"] = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)

    for col in (
        "action",
        "side",
        "notional_exposure",
        "leverage",
        "take_profit",
        "stop_loss",
        "quality_score",
        "confidence",
        "router_expert",
    ):
        out[col] = aggressive_dec[col].to_numpy()

    for col in (
        "router_confidence",
        "router_margin",
        "dir_p_cash",
        "dir_p_long",
        "dir_p_short",
        "dir_confidence",
        "dir_side_edge",
        "dir_trade_prob",
        "quality_p_cash",
        "quality_p_long",
        "quality_p_short",
        "quality_for_action",
        "atr14_pct",
        "ret_1",
        "ret_3",
        "ret_6",
        "ret_12",
        "ret_24",
        "ret_vol_12",
        "ret_vol_24",
        "range_mean_12",
        "range_mean_24",
        "ema9_21_gap",
        "bar_range_pct",
        "body_pct",
    ):
        out[col] = features[col].to_numpy(dtype=np.float64)

    for col in ("dir_action", "final_action", "quality_threshold"):
        src_col = f"{prefix}{col}"
        if src_col in src.columns:
            out[col] = src[src_col].to_numpy()

    side = pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.float64)
    long_edge = pd.to_numeric(out["dir_p_long"], errors="raise").to_numpy(dtype=np.float64) - pd.to_numeric(out["dir_p_short"], errors="raise").to_numpy(dtype=np.float64)
    out["side_aligned_dir_edge"] = np.where(side > 0, long_edge, np.where(side < 0, -long_edge, 0.0))
    out["side_quality_prob"] = np.where(
        side > 0,
        pd.to_numeric(out["quality_p_long"], errors="raise").to_numpy(dtype=np.float64),
        np.where(side < 0, pd.to_numeric(out["quality_p_short"], errors="raise").to_numpy(dtype=np.float64), pd.to_numeric(out["quality_p_cash"], errors="raise").to_numpy(dtype=np.float64)),
    )

    _audit_columns(list(out.columns), where="context_frame")
    return out.reset_index(drop=True)


def _join_context(ledger: pd.DataFrame, ctx: pd.DataFrame, *, split: str) -> pd.DataFrame:
    required = {"entry_signal_i", "exit_i", "net_trade_return_pct", "mfe_pct", "mae_pct", "exit_reason", "side"}
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise RuntimeError(f"{split}: ledger missing required columns: {missing}")

    out = ledger.copy().reset_index(drop=True)
    entry_idx = pd.to_numeric(out["entry_signal_i"], errors="raise").to_numpy(dtype=np.int64)
    exit_idx = pd.to_numeric(out["exit_i"], errors="raise").to_numpy(dtype=np.int64)
    if np.any(entry_idx < 0) or np.any(entry_idx >= len(ctx)):
        raise RuntimeError(f"{split}: entry_signal_i out of range")
    if np.any(exit_idx < 0) or np.any(exit_idx >= len(ctx)):
        raise RuntimeError(f"{split}: exit_i out of range")

    entry_ctx = ctx.iloc[entry_idx].reset_index(drop=True).add_prefix("entry_")
    exit_ctx = ctx.iloc[exit_idx].reset_index(drop=True)[
        [
            "timestamp",
            "close",
            "router_confidence",
            "dir_confidence",
            "dir_trade_prob",
            "quality_for_action",
            "atr14_pct",
            "ret_6",
            "ret_24",
            "ret_vol_24",
        ]
    ].reset_index(drop=True).add_prefix("exit_")
    out = pd.concat([out, entry_ctx, exit_ctx], axis=1)

    pnl = pd.to_numeric(out["net_trade_return_pct"], errors="raise")
    mfe = pd.to_numeric(out["mfe_pct"], errors="raise")
    mae = pd.to_numeric(out["mae_pct"], errors="raise")
    tp = pd.to_numeric(out["tp_equity_ret"], errors="raise") * 100.0
    sl = pd.to_numeric(out["sl_equity_ret"], errors="raise") * 100.0
    out["is_win"] = (pnl > 0).astype(int)
    out["hold_bars"] = pd.to_numeric(out["exit_i"], errors="raise") - pd.to_numeric(out["entry_signal_i"], errors="raise")
    out["hold_hours"] = out["hold_bars"] * 5.0 / 60.0
    out["mfe_to_tp"] = mfe / np.maximum(tp, 1.0e-12)
    out["mae_to_sl"] = mae.abs() / np.maximum(sl, 1.0e-12)
    out["giveback_pct"] = mfe - pnl
    out["sl_after_positive_mfe"] = ((out["exit_reason"] == "stop_loss") & (mfe > 0.50)).astype(int)
    out["near_zero_mfe_sl"] = ((out["exit_reason"] == "stop_loss") & (mfe <= 0.25)).astype(int)
    out["deep_mae_win"] = ((out["exit_reason"] == "take_profit") & (mae <= -1.50)).astype(int)
    return out


def _group_summary(df: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, g in df.groupby(by, dropna=False):
        rows.append(
            {
                by: str(key),
                "trades": int(len(g)),
                "pnl_sum": float(g["net_trade_return_pct"].sum()),
                "pnl_mean": float(g["net_trade_return_pct"].mean()),
                "wr": float((g["net_trade_return_pct"] > 0).mean()),
                "mfe_mean": float(g["mfe_pct"].mean()),
                "mae_mean": float(g["mae_pct"].mean()),
                "hold_hours_mean": float(g["hold_hours"].mean()),
            }
        )
    return sorted(rows, key=lambda x: x["pnl_sum"], reverse=True)


def _numeric_mean_by_win(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        out[col] = {
            "all": float(s.mean()),
            "wins": float(s[df["is_win"] == 1].mean()) if int((df["is_win"] == 1).sum()) else None,
            "losses": float(s[df["is_win"] == 0].mean()) if int((df["is_win"] == 0).sum()) else None,
        }
    return out


def _split_report(split: str, df: pd.DataFrame) -> dict[str, Any]:
    pnl = pd.to_numeric(df["net_trade_return_pct"], errors="raise")
    losers = df.loc[df["is_win"] == 0].copy()
    sl = df.loc[df["exit_reason"] == "stop_loss"].copy()
    top_losses = df.sort_values("net_trade_return_pct").head(8)
    top_wins = df.sort_values("net_trade_return_pct", ascending=False).head(8)
    return {
        "split": split,
        "trades": int(len(df)),
        "pnl_sum": float(pnl.sum()),
        "pnl_mean": float(pnl.mean()),
        "win_rate": float((pnl > 0).mean()) if len(df) else 0.0,
        "long_short": {str(k): int(v) for k, v in df["side"].value_counts().sort_index().items()},
        "exit_reasons": {str(k): int(v) for k, v in df["exit_reason"].value_counts().sort_index().items()},
        "side_summary": _group_summary(df, "side"),
        "expert_summary": _group_summary(df, "entry_router_expert"),
        "exit_reason_summary": _group_summary(df, "exit_reason"),
        "signal_means_by_win": _numeric_mean_by_win(
            df,
            [
                "entry_quality_score",
                "entry_confidence",
                "entry_router_confidence",
                "entry_router_margin",
                "entry_dir_confidence",
                "entry_dir_trade_prob",
                "entry_quality_for_action",
                "entry_side_aligned_dir_edge",
                "entry_side_quality_prob",
                "entry_atr14_pct",
                "entry_ret_6",
                "entry_ret_24",
                "entry_ret_vol_24",
                "mfe_pct",
                "mae_pct",
                "mfe_to_tp",
                "mae_to_sl",
                "hold_hours",
            ],
        ),
        "loss_diagnostics": {
            "sl_count": int(len(sl)),
            "sl_after_positive_mfe_count": int(df["sl_after_positive_mfe"].sum()),
            "near_zero_mfe_sl_count": int(df["near_zero_mfe_sl"].sum()),
            "loser_mfe_mean": float(losers["mfe_pct"].mean()) if len(losers) else None,
            "loser_mae_mean": float(losers["mae_pct"].mean()) if len(losers) else None,
            "loser_hold_hours_mean": float(losers["hold_hours"].mean()) if len(losers) else None,
            "deep_mae_win_count": int(df["deep_mae_win"].sum()),
        },
        "top_losses": top_losses[
            [
                "trade_id",
                "side",
                "entry_time",
                "exit_time",
                "net_trade_return_pct",
                "mfe_pct",
                "mae_pct",
                "exit_reason",
                "entry_router_expert",
                "entry_quality_for_action",
                "entry_side_aligned_dir_edge",
                "entry_atr14_pct",
            ]
        ].to_dict(orient="records"),
        "top_wins": top_wins[
            [
                "trade_id",
                "side",
                "entry_time",
                "exit_time",
                "net_trade_return_pct",
                "mfe_pct",
                "mae_pct",
                "exit_reason",
                "entry_router_expert",
                "entry_quality_for_action",
                "entry_side_aligned_dir_edge",
                "entry_atr14_pct",
            ]
        ].to_dict(orient="records"),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)

    reports: dict[str, Any] = {
        "model_id": MODEL_ID,
        "baseline_id": BASELINE_ID,
        "ledger_dir": str(LEDGER_DIR),
        "outputs": {},
    }
    for split in ("validation", "oos"):
        frame, src, dec, prefix = exposure._build_split(frames, split)
        ctx = _context_frame(frame, src, dec, prefix)
        ledger = pd.read_csv(_ledger_path(split), parse_dates=["entry_time", "exit_time"])
        enriched = _join_context(ledger, ctx, split=split)
        out_csv = OUT_DIR / f"{split}_primary_trade_ledger_enriched.csv"
        enriched.to_csv(out_csv, index=False)
        reports["outputs"][split] = str(out_csv)
        reports[split] = _split_report(split, enriched)

    report_path = OUT_DIR / "primary_trade_audit_report.json"
    report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps(reports, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
