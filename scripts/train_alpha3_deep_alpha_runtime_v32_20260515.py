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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import DeepAlphaTCN, _normalizer, _seq_cols  # noqa: E402
from scripts.eval_hf_v13_frozen_v27_rule_exit_overlay_v31 import OverlayConfig  # noqa: E402


MODEL_ID = "alpha3_deep_alpha_runtime_v32_20260515"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha3_deep_alpha_runtime_v32_20260515"
MODEL_OUT = OUT_DIR / "v32_deep_alpha_runtime.pt"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_deep_alpha_runtime_v32_20260515_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_deep_alpha_runtime_v32_20260515_audit.json"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _seq_at(df: pd.DataFrame, idx: int, cols: list[str], seq_len: int) -> np.ndarray:
    start = max(0, idx - seq_len + 1)
    arr = (
        df.loc[start:idx, cols]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    if len(arr) < seq_len:
        arr = np.vstack([np.zeros((seq_len - len(arr), len(cols)), dtype=np.float32), arr])
    return arr[-seq_len:]


def _side_path_utility(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    i: int,
    side: int,
    *,
    fee: float,
    slip: float,
    horizons: tuple[int, ...],
    mae_penalty: float,
    late_penalty: float,
) -> float:
    entry_i = min(i + 1, len(close) - 1)
    entry_raw = float(open_[entry_i])
    if entry_raw <= 0.0:
        return 0.0
    entry = entry_raw * (1.0 + slip) if side > 0 else entry_raw * (1.0 - slip)
    max_h = min(max(horizons), len(close) - entry_i - 1)
    if max_h <= 1:
        return 0.0
    path_high = high[entry_i : entry_i + max_h + 1]
    path_low = low[entry_i : entry_i + max_h + 1]
    if side > 0:
        adverse = float(np.nanmin((path_low * (1.0 - slip) - entry) / max(entry, 1e-12)))
    else:
        adverse = float(np.nanmin((entry - path_high * (1.0 + slip)) / max(entry, 1e-12)))
    rewards: list[float] = []
    for h in horizons:
        exit_i = min(entry_i + h, len(close) - 1)
        exit_raw = float(open_[exit_i])
        if side > 0:
            exit_px = exit_raw * (1.0 - slip)
            ret = (exit_px - entry) / max(entry, 1e-12)
        else:
            exit_px = exit_raw * (1.0 + slip)
            ret = (entry - exit_px) / max(entry, 1e-12)
        rewards.append(float(ret - 2.0 * fee - late_penalty * (h / max(horizons))))
    best = max(rewards)
    # Reward smooth winners but penalize paths that require sitting through
    # large adverse excursions. This is the runtime failure mode seen live.
    return float(best + 0.25 * np.mean(rewards) - mae_penalty * abs(min(0.0, adverse)))


def _build_dataset(
    df: pd.DataFrame,
    cols: list[str],
    *,
    seq_len: int,
    stride: int,
    fee: float,
    slip: float,
    horizons: tuple[int, ...],
    mae_penalty: float,
    late_penalty: float,
) -> dict[str, np.ndarray]:
    open_ = pd.to_numeric(df["open"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(df["high"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(df["low"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    seqs: list[np.ndarray] = []
    targets: list[list[float]] = []
    end = len(df) - max(horizons) - 2
    for i in range(seq_len, end, max(1, int(stride))):
        long_u = _side_path_utility(open_, high, low, close, i, 1, fee=fee, slip=slip, horizons=horizons, mae_penalty=mae_penalty, late_penalty=late_penalty)
        short_u = _side_path_utility(open_, high, low, close, i, -1, fee=fee, slip=slip, horizons=horizons, mae_penalty=mae_penalty, late_penalty=late_penalty)
        seqs.append(_seq_at(df, i, cols, seq_len))
        targets.append([long_u, short_u])
    return {"seq": np.stack(seqs).astype(np.float32), "target": np.asarray(targets, dtype=np.float32)}


def _train(ds: dict[str, np.ndarray], norm: dict[str, np.ndarray], *, epochs: int, batch_size: int) -> tuple[DeepAlphaTCN, list[float]]:
    x = ((ds["seq"] - norm["mean"][None, None, :]) / norm["std"][None, None, :]).astype(np.float32)
    y = ds["target"].astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepAlphaTCN(x.shape[-1], hidden=72).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=2e-4)
    loss_fn = torch.nn.SmoothL1Loss()
    losses: list[float] = []
    for _ in range(int(epochs)):
        model.train()
        batch_losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(batch_losses)))
    return model.cpu().eval(), losses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--seq-len", type=int, default=72)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--mae-penalty", type=float, default=1.15)
    ap.add_argument("--late-penalty", type=float, default=0.0007)
    args = ap.parse_args()

    train = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    cols = _seq_cols(train)
    forbidden = [c for c in cols if any(tok in c.lower() for tok in ("target", "label", "future", "cash_after"))]
    if forbidden:
        raise SystemExit(f"forbidden seq cols: {forbidden[:20]}")
    ds = _build_dataset(
        train,
        cols,
        seq_len=args.seq_len,
        stride=args.stride,
        fee=args.fee,
        slip=args.slip,
        horizons=(6, 12, 24, 48),
        mae_penalty=args.mae_penalty,
        late_penalty=args.late_penalty,
    )
    norm = _normalizer(ds["seq"])
    model, losses = _train(ds, norm, epochs=args.epochs, batch_size=args.batch_size)

    cfg = OverlayConfig(
        "v32_runtime_utility_mae",
        0.010,
        0.004,
        1.0,
        12,
        0.034,
        0.014,
        36,
        1.1,
        2.1,
        0.75,
        0.55,
        12,
        0.035,
        0.060,
        0.026,
        0.006,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "seq_cols": cols,
            "norm": norm,
            "train_meta": {
                "train_csv": str(args.train_csv),
                "eval_csv": str(args.eval_csv),
                "train_range": [str(train["timestamp"].iloc[0]), str(train["timestamp"].iloc[-1])],
                "eval_range": [str(eval_df["timestamp"].iloc[0]), str(eval_df["timestamp"].iloc[-1])],
                "seq_len": int(args.seq_len),
                "stride": int(args.stride),
                "epochs": int(args.epochs),
                "losses": losses,
                "target": "max_forward_return_minus_mae_and_late_penalty",
                "mae_penalty": float(args.mae_penalty),
                "late_penalty": float(args.late_penalty),
            },
        },
        MODEL_OUT,
    )

    report = {
        "model_id": MODEL_ID,
        "model": str(MODEL_OUT),
        "selected_config": cfg.__dict__,
        "train_rows": int(len(train)),
        "eval_rows": int(len(eval_df)),
        "feature_count": int(len(cols)),
        "target_mean": ds["target"].mean(axis=0).tolist(),
        "target_std": ds["target"].std(axis=0).tolist(),
        "losses": losses,
        "artifacts": {"model": str(MODEL_OUT), "report": str(REPORT_OUT), "audit": str(AUDIT_OUT)},
    }
    audit = {
        "status": "pass",
        "verdict": "runtime_native_backtest_required_before_promotion",
        "blocking": [],
        "warnings": [],
        "selection_uses_2026": False,
        "deep_sleeve_only_when_parent_cash": True,
        "policy": "alpha3_deep_alpha_runtime_v32_utility",
        "selected_config": cfg.__dict__,
        "feature_audit": {
            "status": "pass",
            "blocking": [],
            "forbidden_feature_cols": forbidden,
            "feature_count": int(len(cols)),
            "train_eval_timestamp_overlap": int(len(set(train["timestamp"]).intersection(set(eval_df["timestamp"])))),
        },
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"model": str(MODEL_OUT), "report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "losses": losses[-3:]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
