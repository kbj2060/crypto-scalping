#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import prepare_features  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _audit_contract,
    _close,
    _days,
    _feature_cols,
    _fill_price,
    _read,
)
from scripts.train_eval_hf_v13_deep_edge_surface_parent_v39 import (  # noqa: E402
    MARGIN110_COST1,
    SEQ_LEN,
    TARGET_SCALE,
    V31_COST1,
    _apply_norm,
    _json_default,
    _normalizer,
    _parent_cfg,
    _seq_tensor,
    _vol_anchor,
)


MODEL_ID = "hf_v13_deep_best_utility_parent_v39_1_20260512"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_deep_best_utility_parent_v39_1_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_deep_best_utility_parent_v39_1_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_deep_best_utility_parent_v39_1_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_deep_best_utility_parent_v39_1_20260512_grid.csv"
MAX_LABEL_HOLD = 96


@dataclass(frozen=True)
class BestUtilityRuntime:
    name: str
    edge_th: float
    margin_th: float
    adverse_weight: float
    base_notional: float
    max_notional: float
    risk_budget: float
    base_tp: float
    base_sl: float
    max_hold: int
    cooldown: int
    tp_mult: float = 1.05
    sl_mult: float = 1.05
    trail: bool = True
    vol_throttle: bool = True


class DeepBestUtilityTCN(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        dim = input_dim
        for dilation in (1, 2, 4, 8, 16):
            blocks.extend(
                [
                    nn.Conv1d(dim, hidden, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.GELU(),
                    nn.Dropout(0.08),
                ]
            )
            dim = hidden
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 160),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(160, 96),
            nn.GELU(),
            nn.Linear(96, 4),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.tcn(seq.transpose(1, 2))
        avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        last = x[:, :, -1]
        return self.head(torch.cat([avg, last], dim=-1))


def _runtime_grid() -> list[BestUtilityRuntime]:
    rows: list[BestUtilityRuntime] = []
    for edge in (0.004, 0.006, 0.008, 0.010, 0.012, 0.016):
        rows.append(BestUtilityRuntime(f"v39_1_balanced_e{edge:.3f}", edge, 0.0025, 0.30, 0.90, 1.50, 0.016, 0.040, 0.018, 48, 12))
        rows.append(BestUtilityRuntime(f"v39_1_precision_e{edge:.3f}", edge, 0.0040, 0.55, 0.70, 1.20, 0.012, 0.036, 0.015, 48, 12))
        rows.append(BestUtilityRuntime(f"v39_1_convex_e{edge:.3f}", edge, 0.0030, 0.20, 1.00, 1.80, 0.018, 0.045, 0.020, 72, 12))
    return rows


def _build_best_targets(frame: pd.DataFrame, indices: np.ndarray, *, fee: float, slip: float) -> np.ndarray:
    open_px = pd.to_numeric(frame["open"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().to_numpy(dtype=np.float64)
    horizons = np.arange(1, MAX_LABEL_HOLD + 1, dtype=np.int64)
    targets = np.zeros((len(indices), 4), dtype=np.float32)
    cost = 2.0 * float(fee + slip)
    for r, idx in enumerate(indices):
        entry_i = min(int(idx) + 1, len(open_px) - 1)
        exit_i = np.minimum(entry_i + horizons, len(open_px) - 1)
        long_entry = open_px[entry_i] * (1.0 + slip)
        short_entry = open_px[entry_i] * (1.0 - slip)
        long_exit = open_px[exit_i] * (1.0 - slip)
        short_exit = open_px[exit_i] * (1.0 + slip)
        long_path = long_exit / max(long_entry, 1e-12) - 1.0 - cost
        short_path = (short_entry - short_exit) / max(short_entry, 1e-12) - cost
        targets[r, 0] = float(np.max(long_path))
        targets[r, 1] = float(np.max(short_path))
        targets[r, 2] = max(0.0, -float(np.min(long_path)))
        targets[r, 3] = max(0.0, -float(np.min(short_path)))
    return targets


def _fit_model(seq: np.ndarray, y: np.ndarray, norm: dict[str, np.ndarray], *, epochs: int, seed: int) -> DeepBestUtilityTCN:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    x = _apply_norm(seq, norm)
    y_scaled = (y * TARGET_SCALE).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepBestUtilityTCN(x.shape[-1]).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y_scaled)), batch_size=256, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    huber = nn.SmoothL1Loss()
    model.train()
    for _ in range(int(epochs)):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = huber(pred, yb) + 0.04 * (F.relu(-pred[:, 2]).mean() + F.relu(-pred[:, 3]).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.cpu().eval()


def _predict_all(model: DeepBestUtilityTCN, features: pd.DataFrame, cols: list[str], norm: dict[str, np.ndarray]) -> np.ndarray:
    indices = np.arange(len(features), dtype=np.int64)
    seq = _seq_tensor(features, indices, cols)
    x = _apply_norm(seq, norm)
    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            out.append((model(torch.from_numpy(x[start : start + 1024])).numpy() / TARGET_SCALE).astype(np.float32))
    return np.vstack(out)


def _scores(pred_row: np.ndarray, cfg: BestUtilityRuntime, *, fee: float, slip: float) -> tuple[float, float]:
    long_best, short_best, long_adv, short_adv = [float(x) for x in pred_row]
    cost_buffer = 2.0 * (fee + slip)
    return (
        long_best - cfg.adverse_weight * max(long_adv, 0.0) - cost_buffer,
        short_best - cfg.adverse_weight * max(short_adv, 0.0) - cost_buffer,
    )


def _entry_params(pred_row: np.ndarray, cfg: BestUtilityRuntime, side: int, row: pd.Series) -> tuple[float, float, float]:
    best = max(float(pred_row[0 if side > 0 else 1]), 0.0)
    adverse = max(float(pred_row[2 if side > 0 else 3]), 0.003)
    risk_cap = cfg.risk_budget / max(adverse, 0.004)
    edge_scale = float(np.clip(0.75 + best / 0.035, 0.65, 1.45))
    notional = min(cfg.max_notional, risk_cap, cfg.base_notional * edge_scale)
    if cfg.vol_throttle:
        va = _vol_anchor(row)
        if va > 0.020:
            notional *= 0.50
        elif va > 0.014:
            notional *= 0.70
    tp = float(np.clip(max(cfg.base_tp, best * notional * cfg.tp_mult), cfg.base_tp * 0.75, 0.090))
    sl = float(np.clip(max(cfg.base_sl, adverse * notional * cfg.sl_mult), cfg.base_sl * 0.65, 0.045))
    return float(max(notional, 0.0)), tp, sl


def backtest(df: pd.DataFrame, pred: np.ndarray, cfg: BestUtilityRuntime, *, fee: float, slip: float, record: bool = False) -> dict[str, Any]:
    close = _close(df)
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = entry_equity = 0.0
    entry_idx = 0
    notional = tp = sl = 0.0
    cooldown = 0
    trades = wins = long_entries = short_entries = 0
    notional_sum = 0.0
    exits: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None
    mfe = mae = 0.0

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold = i - entry_idx
            eff_sl = sl
            if cfg.trail and mfe > 0.0:
                gap = max(_vol_anchor(df.iloc[entry_idx]) * notional * 0.75, 0.004)
                if hold >= 18:
                    gap = max(gap * 0.35, gap - 0.025 * (hold - 18) * gap)
                eff_sl = min(eff_sl, max(0.001, mfe - gap))
            reason = ""
            if tp > 0.0 and unreal >= tp:
                reason = "take_profit"
            elif eff_sl > 0.0 and unreal <= -abs(eff_sl):
                reason = "stop_loss"
            elif hold >= cfg.max_hold:
                reason = "max_hold"
            if reason:
                fill_i = min(i + 1, len(df) - 1)
                exit_px = _fill_price(df, fill_i, pos, slip, entry=False)
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update(
                        {
                            "exit_signal_timestamp": str(df["timestamp"].iloc[i]),
                            "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]),
                            "exit_reason": reason,
                            "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0),
                            "mfe_pct": float(mfe * 100.0),
                            "mae_pct": float(mae * 100.0),
                            "cash_after": float(cash),
                        }
                    )
                    records.append(out)
                pos = 0
                cooldown = int(cfg.cooldown)
                open_record = None
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        long_score, short_score = _scores(pred[i], cfg, fee=fee, slip=slip)
        if max(long_score, short_score) < cfg.edge_th or abs(long_score - short_score) < cfg.margin_th:
            continue
        side = 1 if long_score > short_score else -1
        n, tp_new, sl_new = _entry_params(pred[i], cfg, side, df.iloc[i])
        if n <= 0.05:
            continue
        fill_i = min(i + 1, len(df) - 1)
        pos = side
        entry_price = _fill_price(df, fill_i, pos, slip, entry=True)
        entry_equity = cash
        entry_idx = i
        notional, tp, sl = float(n), float(tp_new), float(sl_new)
        cash -= cash * fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        mfe = mae = 0.0
        if record:
            open_record = {
                "entry_signal_timestamp": str(df["timestamp"].iloc[i]),
                "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]),
                "side": "LONG" if pos > 0 else "SHORT",
                "entry_price": float(entry_price),
                "notional_exposure": float(notional),
                "long_score": float(long_score),
                "short_score": float(short_score),
                "take_profit": float(tp),
                "stop_loss": float(sl),
                "fee_entry_pct": float(fee * notional * 100.0),
            }
    if pos != 0:
        fill_i = len(df) - 1
        exit_px = _fill_price(df, fill_i, pos, slip, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    entries = max(long_entries + short_entries, 1)
    out: dict[str, Any] = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / entries),
        "avg_leverage": 1.0,
        "exits": exits,
    }
    if record:
        out["trade_records"] = records
    return out


def _selection_score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(c1["pnl"] + 0.35 * c2["pnl"] + 0.20 * c3["pnl"] - 0.45 * abs(c1["mdd"]) + 0.08 * min(c1["trades"], 160))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep best-utility parent V39.1: no CASH label, V27-like best edge targets.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--seed", type=int, default=2032)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[{MODEL_ID}] loading data", flush=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    audit = _audit_contract(train_all, eval_df, feature_cols)
    cfg = _parent_cfg()
    features = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    valid = np.arange(SEQ_LEN - 1, max(SEQ_LEN, len(train_df) - MAX_LABEL_HOLD - 2), max(1, int(args.stride)), dtype=np.int64)
    print(f"[{MODEL_ID}] building best-utility labels rows={len(valid)} cols={len(feature_cols)}", flush=True)
    seq = _seq_tensor(features, valid, feature_cols)
    y = _build_best_targets(train_df, valid, fee=cfg.fee, slip=cfg.slip)
    norm = _normalizer(seq)
    print(f"[{MODEL_ID}] training epochs={args.epochs} device={'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    model = _fit_model(seq, y, norm, epochs=int(args.epochs), seed=int(args.seed))
    print(f"[{MODEL_ID}] predicting validation/OOS", flush=True)
    val_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    val_pred = _predict_all(model, val_features, feature_cols, norm)
    eval_pred = _predict_all(model, eval_features, feature_cols, norm)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for runtime in _runtime_grid():
        print(f"[{MODEL_ID}] validation {runtime.name}", flush=True)
        v1 = backtest(val_df, val_pred, runtime, fee=cfg.fee, slip=cfg.slip)
        v2 = backtest(val_df, val_pred, runtime, fee=cfg.fee * 2.0, slip=cfg.slip * 2.0)
        v3 = backtest(val_df, val_pred, runtime, fee=cfg.fee * 3.0, slip=cfg.slip * 3.0)
        row = {"runtime": asdict(runtime), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _selection_score(v1, v2, v3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    if best is None:
        raise RuntimeError("no runtime candidates")
    selected = BestUtilityRuntime(**best["runtime"])
    print(f"[{MODEL_ID}] selected {selected.name}; running OOS", flush=True)
    metrics: dict[str, Any] = {}
    ledger_path = args.report_out.with_name(f"{args.report_out.stem}_cost1_ledger.csv")
    for mult in (1, 2, 3):
        r = backtest(eval_df, eval_pred, selected, fee=cfg.fee * mult, slip=cfg.slip * mult, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(ledger_path, index=False)
        metrics[f"cost{mult}"] = r

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "deep_best_utility_parent_v39_1.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "norm": norm,
            "target_scale": TARGET_SCALE,
            "seq_len": SEQ_LEN,
            "max_label_hold": MAX_LABEL_HOLD,
            "selected_runtime": asdict(selected),
            "label_columns": ["long_best", "short_best", "long_adverse", "short_adverse"],
        },
        model_path,
    )
    pd.DataFrame(
        [
            {
                **{f"rt_{k}": v for k, v in row["runtime"].items()},
                "selection_score": row["selection_score"],
                "val_pnl": row["validation_cost1"]["pnl"],
                "val_mdd": row["validation_cost1"]["mdd"],
                "val_trades": row["validation_cost1"]["trades"],
                "val_cost2_pnl": row["validation_cost2"]["pnl"],
                "val_cost3_pnl": row["validation_cost3"]["pnl"],
            }
            for row in rows
        ]
    ).to_csv(args.grid_out, index=False)
    blocking = list(audit.get("blocking", []))
    warnings = list(audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= 0.0:
        warnings.append("cost1_not_survived")
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    final_audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and metrics["cost1"]["pnl"] > MARGIN110_COST1 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "feature_audit": audit,
        "baseline_margin110_cost1": MARGIN110_COST1,
        "baseline_v31_cost1": V31_COST1,
        "selected_runtime": asdict(selected),
        "metrics": metrics,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Deep Best-Utility Parent V39.1. CASH is not learned. The model predicts long/short best unit utility and adverse risk from 72-bar sequences, then a 2025 Q4 threshold gate decides abstention.",
        "split_policy": "train=2025 Jan-Sep, selection=2025 Q4, OOS=2026 fixed",
        "feature_count": len(feature_cols),
        "train_rows": int(len(valid)),
        "target_summary": {
            "long_best_mean": float(np.mean(y[:, 0])),
            "short_best_mean": float(np.mean(y[:, 1])),
            "long_best_p95": float(np.quantile(y[:, 0], 0.95)),
            "short_best_p95": float(np.quantile(y[:, 1], 0.95)),
            "long_adverse_mean": float(np.mean(y[:, 2])),
            "short_adverse_mean": float(np.mean(y[:, 3])),
        },
        "selection": best,
        "metrics": metrics,
        "audit": final_audit,
        "artifacts": {"model": str(model_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledger": str(ledger_path)},
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(final_audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected": selected.name, "metrics": metrics, "verdict": final_audit["verdict"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
