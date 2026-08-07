#!/usr/bin/env python3
"""Retrain the Omega6 L3 TCN sequence entry gate against Omega6's own L2 decision trace.

Fixes the Open Issue in docs/model_contracts/omega6_synthesis_v1_20260703_contract.md: the
originally-reused scripts/train_eval_omega462_live_native_sequence_entry_gate_20260703.py
artifact's feature_cols were the Omega4.6.2 dual-parent's own decision trace (h48qual_*/
zig075_*), incompatible with Omega6's differently-trained L2. This script builds a fresh
24-bar sequence dataset from Omega6's own primary/fallback decision trace, train-split only
(timestamp < SPLIT_TS = 2025-10-01), trains a SequenceEntryTCN (architecture reused verbatim
from trading_bot_modules/omega6_live.py::SequenceEntryTCN) to predict the counterfactual SHORT
trade's realized net_per_notional return, and selects a decision threshold via a held-out
train-only calibration slice.

Threshold convention (confirmed from
scripts/train_eval_omega462_live_native_sequence_entry_gate_20260703.py::select_threshold):
score >= threshold keeps HIGH-scoring (high predicted short return) bars.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega6_tabm_3head_20260703 as omega6_tabm  # noqa: E402
from trading_bot_modules.omega6_live import (  # noqa: E402
    L5_BASE_SL_PRICE_MOVE,
    L5_BASE_TP_PRICE_MOVE,
    L5_MAX_HOLD_BARS,
    Omega6LiveAdapter,
    SequenceEntryTCN,
)

MODEL_ID = "omega6_sequence_gate_20260703"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SPLIT_TS = omega6_tabm.SPLIT_TS  # 2025-10-01, identical boundary to the L2 trainer
CONTEXT_BARS = 260
LOOKBACK = 24
CALIB_FRACTION = 0.20  # last 20% of train-eligible bars reserved for train-only threshold calibration


def _build_tape_and_labels(
    frame: pd.DataFrame,
    adapter: Omega6LiveAdapter,
    arrays: dict[str, np.ndarray],
    *,
    fee: float,
    slip: float,
    start_idx: int,
    end_idx: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For every bar in [start_idx, end_idx) where BOTH primary and fallback are CASH (the
    exact situation where decide_latest would consult L3), record a tape row plus a
    counterfactual SHORT trade_return label (simulated with baseline L5 TP/SL/time-stop)."""
    tape_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    take_profit = float(L5_BASE_TP_PRICE_MOVE) * 2.0  # matches L4_BASELINE_LEVERAGE=2.0 barrier scale
    stop_loss = float(L5_BASE_SL_PRICE_MOVE) * 2.0
    for i in range(start_idx, end_idx):
        window = frame.iloc[max(0, i - CONTEXT_BARS + 1) : i + 1]
        primary_out = adapter._predict_parent(adapter.primary, window)
        if primary_out["side"] != 0:
            continue
        fallback_out = adapter._predict_parent(adapter.fallback, window)
        if fallback_out["side"] != 0:
            continue
        atr_pct = adapter._atr_pct(window, adapter.atr_window)
        ts = pd.Timestamp(frame.iloc[i]["timestamp"])
        tape_rows.append({"row": int(i), **adapter.l3_tape_row(primary_out, fallback_out, atr_pct, ts)})

        filled, entry_px, entry_fee, _r = omega._try_execution(arrays, i, -1, entry=True, fee_base=fee, slip_base=slip)
        if not filled:
            continue
        exit_i = min(i + L5_MAX_HOLD_BARS, end_idx - 1)
        reason = "time_stop"
        for j in range(i + 1, exit_i + 1):
            px = float(arrays["close"][j])
            raw = (entry_px - px * (1.0 + slip)) / max(entry_px, 1e-12)
            if raw >= take_profit:
                exit_i, reason = j, "take_profit"
                break
            if raw <= -abs(stop_loss):
                exit_i, reason = j, "stop_loss"
                break
        filled, exit_px, exit_fee, _r = omega._try_execution(arrays, exit_i, -1, entry=False, fee_base=fee, slip_base=slip)
        if not filled:
            continue
        raw_exit = (entry_px - exit_px) / max(entry_px, 1e-12)
        net_per_notional = float(raw_exit - entry_fee - exit_fee)
        label_rows.append({"row": int(i), "entry_i": int(i), "exit_i": int(exit_i), "reason": reason, "trade_return": net_per_notional})
    return pd.DataFrame(tape_rows), pd.DataFrame(label_rows)


def _build_sequences(tape: pd.DataFrame, labels: pd.DataFrame, feature_cols: list[str], lookback: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    values = tape[feature_cols].to_numpy(dtype=np.float32)
    row_to_pos = {int(r): p for p, r in enumerate(tape["row"].astype(int).to_numpy())}
    seqs: list[np.ndarray] = []
    targets: list[float] = []
    kept: list[int] = []
    for idx, label in labels.reset_index(drop=True).iterrows():
        pos = row_to_pos.get(int(label["entry_i"]))
        if pos is None or pos < lookback - 1:
            continue
        seqs.append(values[pos - lookback + 1 : pos + 1])
        targets.append(float(label["trade_return"]))
        kept.append(int(idx))
    if not seqs:
        raise RuntimeError("Omega6 sequence gate: no sequences built")
    kept_labels = labels.reset_index(drop=True).iloc[kept].reset_index(drop=True)
    return np.stack(seqs).astype(np.float32), np.asarray(targets, dtype=np.float32), kept_labels


def _select_threshold(calib_labels: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    candidates = sorted(set(float(x) for x in np.quantile(scores, [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9])))
    trade_return = calib_labels["trade_return"].astype(float).to_numpy()
    min_trades = max(10, int(np.floor(len(calib_labels) * 0.15)))
    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        keep = scores >= threshold
        picked = trade_return[keep]
        pnl = float(picked.sum()) if len(picked) else 0.0
        wr = float((picked > 0.0).mean()) if len(picked) else 0.0
        rows.append({"threshold": float(threshold), "trades": int(len(picked)), "pnl": pnl, "wr": wr, "eligible": int(len(picked) >= min_trades)})
    eligible = [r for r in rows if r["eligible"]] or rows
    selected = max(eligible, key=lambda r: (r["pnl"], r["wr"], r["trades"]))
    return {"selected": selected, "grid": rows, "min_trades": int(min_trades)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=260703)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    ap.add_argument(
        "--primary-bundle",
        default=str(ROOT / "tmp/causal_regen_20260516/omega6_true_3head_tabm_20260703_primary/true_3head_tabm_bundle.pt"),
    )
    ap.add_argument(
        "--fallback-bundle",
        default=str(ROOT / "tmp/causal_regen_20260516/omega6_true_3head_tabm_20260703_fallback/true_3head_tabm_bundle.pt"),
    )
    ap.add_argument(
        "--seed-tcn-gate",
        default=str(ROOT / "tmp/causal_regen_20260516/omega462_live_native_tcn_sequence_entry_gate_20260703/tcn_seq_gate_L24_flat.pt"),
        help="Only used to bootstrap Omega6LiveAdapter construction (fail-fast requires a valid path); not a training input.",
    )
    ap.add_argument(
        "--risk-sidecar",
        default=str(ROOT / "tmp/causal_regen_20260516/omega6_risk_sidecar_20260703/risk_sidecar.pkl"),
        help="Only used to bootstrap Omega6LiveAdapter construction; not a training input for L3.",
    )
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device_str = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    adapter = Omega6LiveAdapter(
        primary_bundle_path=args.primary_bundle,
        fallback_bundle_path=args.fallback_bundle,
        tcn_gate_path=args.seed_tcn_gate,
        risk_sidecar_path=args.risk_sidecar,
        device=device_str,
    )

    train, eval_df, _overlay = omega._load_omega_frames()
    combined = pd.concat([train, eval_df], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    fee, slip = omega._load_fee_slip()
    arrays = {c: pd.to_numeric(combined[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}

    train_start_idx = CONTEXT_BARS
    train_end_idx = int(combined["timestamp"].searchsorted(SPLIT_TS, side="left"))
    tape, labels = _build_tape_and_labels(combined, adapter, arrays, fee=fee, slip=slip, start_idx=train_start_idx, end_idx=train_end_idx)
    if labels.empty:
        raise RuntimeError("Omega6 sequence gate: no CASH+CASH candidate bars found in train split")

    feature_cols = [c for c in tape.columns if c != "row"]
    seqs, y, kept_labels = _build_sequences(tape, labels, feature_cols, LOOKBACK)

    split_at = max(int(len(seqs) * (1.0 - CALIB_FRACTION)), 1)
    train_seq, calib_seq = seqs[:split_at], seqs[split_at:]
    train_y = y[:split_at]
    calib_labels = kept_labels.iloc[split_at:].reset_index(drop=True)
    if len(calib_seq) < 10:
        raise RuntimeError(f"Omega6 sequence gate: too few calibration sequences ({len(calib_seq)})")

    mean = np.nanmean(train_seq, axis=(0, 1)).astype(np.float32)
    std = (np.nanstd(train_seq, axis=(0, 1)) + 1.0e-6).astype(np.float32)
    x_train = ((train_seq - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    x_calib = ((calib_seq - mean[None, None, :]) / std[None, None, :]).astype(np.float32)

    device = torch.device(device_str)
    model = SequenceEntryTCN(seq_dim=len(feature_cols)).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(train_y)), batch_size=int(args.batch_size), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1.0e-4)
    loss_fn = nn.SmoothL1Loss()
    losses: list[float] = []
    model.train()
    for _epoch in range(int(args.epochs)):
        epoch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
    model = model.cpu().eval()

    with torch.no_grad():
        calib_scores = model(torch.from_numpy(x_calib)).numpy().astype(np.float32)
    threshold_payload = _select_threshold(calib_labels, calib_scores)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "tcn_seq_gate_L24_omega6.pt"
    payload = {
        "model_id": MODEL_ID,
        "name": "tcn_seq_gate_L24_omega6",
        "lookback": int(LOOKBACK),
        "sample_mode": "cash_only_short_counterfactual",
        "feature_cols": feature_cols,
        "mean": mean,
        "std": std,
        "threshold": float(threshold_payload["selected"]["threshold"]),
        "threshold_payload": threshold_payload,
        "state_dict": model.state_dict(),
        "train_report": {
            "epochs": int(args.epochs),
            "losses": losses,
            "n_train_sequences": int(len(train_seq)),
            "n_calib_sequences": int(len(calib_seq)),
            "train_window": {"start": str(combined.iloc[train_start_idx]["timestamp"]), "end": str(combined.iloc[train_end_idx - 1]["timestamp"])},
            "lineage": "Trained from scratch against Omega6's own L2 (primary/fallback) decision trace, "
            "train-split only (timestamp < 2025-10-01), fixing the L3 incompatibility Open Issue in "
            "docs/model_contracts/omega6_synthesis_v1_20260703_contract.md.",
        },
    }
    torch.save(payload, out_path)
    report = {
        "model_id": MODEL_ID,
        "artifact": str(out_path),
        "n_candidate_bars": int(len(tape)),
        "n_labels": int(len(labels)),
        "n_sequences": int(len(seqs)),
        "threshold": threshold_payload,
        "train_window": payload["train_report"]["train_window"],
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
