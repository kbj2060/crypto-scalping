#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_omega462_hf_policy_bar_forward_val_oos_20260702 import (  # noqa: E402
    BARS_PER_HOUR,
    close_position,
    current_raw_move,
    json_default,
    load_frame,
    make_parent,
    required_columns,
    summarize_ledger,
    write_json,
)
from scripts.train_eval_omega462_live_native_entry_gate_20260702 import (  # noqa: E402
    DEFAULT_FEATURES_2025,
    DEFAULT_OOS_FEATURES,
    DEFAULT_TRAIN_FEATURES,
    close_if_needed,
    counterfactual_entry_label,
    feature_row,
    load_policy,
    safe_float,
    summarize_entries,
)
from tmp.causal_regen_20260516.extended_oos_20260702.run_omega5_additional_oos_replay import (  # noqa: E402
    ROUNDTRIP_COST_DEFAULT,
    atr_pct_series,
    parent_decision_at,
)
from trading_bot_modules.omega4_6_2_source_parent_live import EPS, Omega462SourceParentLiveAdapter  # noqa: E402


MODEL_ID = "omega462_live_native_tcn_sequence_entry_gate_20260703"
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/omega462_live_native_tcn_sequence_entry_gate_20260703"


@dataclass
class SequenceGateArtifact:
    name: str
    lookback: int
    sample_mode: str
    feature_cols: list[str]
    mean: np.ndarray
    std: np.ndarray
    threshold: float
    threshold_payload: dict[str, Any]
    model: "SequenceEntryTCN"
    train_report: dict[str, Any]
    path: str


class SequenceEntryTCN(nn.Module):
    def __init__(self, seq_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=8, dilation=8),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.net(seq.transpose(1, 2))
        pooled = x.mean(dim=-1)
        last = x[:, :, -1]
        return self.head(torch.cat([pooled, last], dim=1)).squeeze(-1)


def frame_with_runtime_features(
    *,
    feature_path: Path,
    start: str,
    end: str,
    parent_variant: str,
) -> tuple[Omega462SourceParentLiveAdapter, pd.DataFrame, np.ndarray, int, int, dict[str, Any]]:
    parent_for_contract = make_parent(parent_variant)
    raw_required = required_columns(parent_for_contract)
    del parent_for_contract

    parent = make_parent(parent_variant)
    frame, source_audit = load_frame(feature_path, start, end, raw_required)
    work = parent.regime3._append_current(frame.copy())
    atr = atr_pct_series(work)
    ts = work["timestamp"].to_numpy()
    start_i = int(np.flatnonzero(ts >= pd.Timestamp(start).to_datetime64())[0])
    end_idx = np.flatnonzero(ts < pd.Timestamp(end).to_datetime64())
    end_i = int(end_idx[-1]) if len(end_idx) else len(work) - 1
    return parent, work, atr, start_i, end_i, source_audit


def parse_csv_list(text: str, *, cast: type = str) -> list[Any]:
    out: list[Any] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    if not out:
        raise RuntimeError(f"empty csv list: {text!r}")
    return out


def feature_vector(feature: dict[str, float], feature_cols: list[str]) -> np.ndarray:
    return np.asarray([safe_float(feature.get(col, 0.0)) for col in feature_cols], dtype=np.float32)


def feature_columns_from_tape(tape: pd.DataFrame) -> list[str]:
    drop_cols = {"split", "row", "timestamp", "signal", "candidate", "in_position"}
    cols = [c for c in tape.columns if c not in drop_cols]
    if not cols:
        raise RuntimeError("sequence feature tape has no feature columns")
    tape[cols].apply(pd.to_numeric, errors="raise")
    return cols


def collect_train_tape_and_labels(
    *,
    split: str,
    feature_path: Path,
    start: str,
    end: str,
    parent_variant: str,
    policy: dict[str, float],
    out_dir: Path,
    save_tape: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    parent, work, atr, start_i, end_i, source_audit = frame_with_runtime_features(
        feature_path=feature_path,
        start=start,
        end=end,
        parent_variant=parent_variant,
    )
    high_arr = pd.to_numeric(work["high"], errors="raise").to_numpy(dtype=np.float64)
    low_arr = pd.to_numeric(work["low"], errors="raise").to_numpy(dtype=np.float64)
    close_arr = pd.to_numeric(work["close"], errors="raise").to_numpy(dtype=np.float64)
    ts_arr = work["timestamp"].to_numpy()

    tape_rows: list[dict[str, Any]] = []
    entry_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    position: dict[str, Any] | None = None
    overlay_loss_streak = 0

    for i in range(start_i, end_i + 1):
        row = work.iloc[i]
        now = pd.Timestamp(row["timestamp"])
        if (i - start_i) % 5000 == 0:
            print(
                json.dumps(
                    {
                        "split": split,
                        "done": int(i - start_i),
                        "total": int(end_i - start_i + 1),
                        "timestamp": str(now),
                        "closed": int(len(ledger)),
                        "position": None if position is None else int(position["side"]),
                        "labels": int(len(entry_rows)),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if position is not None:
            closed, _, _ = close_if_needed(work=work, position=position, i=i, policy=policy)
            if closed is not None:
                ledger.append(closed)
                parent.record_closed_trade(
                    exit_timestamp=closed["exit_timestamp"],
                    net_per_notional=float(closed["net_per_notional"]),
                )
                overlay_loss_streak = overlay_loss_streak + 1 if float(closed["trade_return"]) <= 0.0 else 0
                position = None
                continue

        parent_dec = parent_decision_at(parent, work.iloc[i : i + 1], float(atr[i]), now)
        parent_trace = dict(parent_dec.trace or {})
        in_position = position is not None
        signal = int(parent_dec.action) != 0 and int(parent_dec.side) != 0 and float(parent_dec.notional_exposure) > EPS
        candidate = not in_position and signal
        features = feature_row(
            work=work,
            atr=atr,
            i=i,
            parent_dec=parent_dec,
            parent_trace=parent_trace,
            overlay_loss_streak=overlay_loss_streak,
        )
        tape_rows.append(
            {
                **features,
                "split": split,
                "row": int(i),
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "signal": int(signal),
                "candidate": int(candidate),
                "in_position": int(in_position),
            }
        )

        if signal:
            label = counterfactual_entry_label(
                work=work,
                high_arr=high_arr,
                low_arr=low_arr,
                close_arr=close_arr,
                ts_arr=ts_arr,
                i=i,
                end_i=end_i,
                parent_dec=parent_dec,
                policy=policy,
            )
            if label is not None:
                entry_rows.append(
                    {
                        "split": split,
                        "entry_i": int(i),
                        "entry_timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "side": int(label["side"]),
                        "candidate_was_in_position": int(in_position),
                        "exit_i": int(label["exit_i"]),
                        "exit_timestamp": str(label["exit_timestamp"]),
                        "reason": str(label["reason"]),
                        "raw_exit_price_move": float(label["raw_exit_price_move"]),
                        "mfe_price_move": float(label["mfe_price_move"]),
                        "mae_price_move": float(label["mae_price_move"]),
                        "net_per_notional": float(label["net_per_notional"]),
                        "trade_return": float(label["trade_return"]),
                        "win": int(label["win"]),
                        "hold_hours": float(label["hold_hours"]),
                    }
                )

        decisions.append(
            {
                "split": split,
                "row": int(i),
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "action": int(parent_dec.action),
                "side": int(parent_dec.side),
                "notional": float(parent_dec.notional_exposure),
                "margin_fraction": float(parent_dec.position_fraction),
                "leverage": float(parent_dec.leverage),
                "quality_score": float(parent_dec.quality_score),
                "confidence": float(parent_dec.confidence),
                "router_expert": str(parent_dec.router_expert),
                "ignored_because_in_position": bool(in_position),
                "ledger_replay_used": bool(parent_trace.get("ledger_replay_used", True)),
                "source_parent_live_native_adapter": bool(parent_trace.get("source_parent_live_native_adapter", False)),
                "source_parent_policy_row": int(parent_trace.get("source_parent_policy_row", -999)),
                "fresh_forward_bar_by_bar": True,
                "future_rows_used_for_entry": False,
            }
        )

        if not candidate:
            continue

        leverage = float(parent_dec.leverage)
        notional = min(float(parent_dec.notional_exposure), float(policy["cap"]))
        if notional <= EPS:
            continue
        margin = notional / max(leverage, EPS)
        if abs(margin * leverage - notional) > 1.0e-8:
            raise RuntimeError("sequence train replay violates notional=margin_fraction*leverage")
        position = {
            "entry_i": int(i),
            "side": int(parent_dec.side),
            "entry_price": float(row["close"]),
            "notional": float(notional),
            "base_parent_notional": float(parent_dec.notional_exposure),
            "margin_fraction": float(margin),
            "leverage": float(leverage),
            "tp_price_move": float(policy["tp"]),
            "sl_price_move": float(policy["sl"]),
            "roundtrip_cost": float(ROUNDTRIP_COST_DEFAULT),
            "router_expert": str(parent_dec.router_expert),
            "parent_quality_score": float(parent_dec.quality_score),
            "parent_confidence": float(parent_dec.confidence),
            "overlay_loss_scale": 1.0,
        }

    tape_df = pd.DataFrame(tape_rows)
    labels_df = pd.DataFrame(entry_rows)
    decisions_df = pd.DataFrame(decisions)
    ledger_df = pd.DataFrame(ledger)
    decisions_path = out_dir / f"{split}_decisions.csv"
    ledger_path = out_dir / f"{split}_ledger.csv"
    labels_path = out_dir / f"{split}_counterfactual_entry_labels.csv"
    tape_path = out_dir / f"{split}_feature_tape.csv"
    decisions_df.to_csv(decisions_path, index=False)
    ledger_df.to_csv(ledger_path, index=False)
    labels_df.to_csv(labels_path, index=False)
    if save_tape:
        tape_df.to_csv(tape_path, index=False)

    metrics = summarize_ledger(ledger_df, decisions_df)
    return tape_df, labels_df, metrics, {
        "source": source_audit,
        "decisions": str(decisions_path),
        "ledger": str(ledger_path),
        "counterfactual_labels": str(labels_path),
        "feature_tape": str(tape_path) if save_tape else "",
        "entry_label_summary": summarize_entries(labels_df),
    }


def build_sequences(
    *,
    tape: pd.DataFrame,
    labels: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if labels.empty:
        raise RuntimeError("no sequence labels")
    values = tape[feature_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    row_to_pos = {int(row): pos for pos, row in enumerate(tape["row"].astype(int).to_numpy())}
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
        raise RuntimeError(f"no sequences after lookback={lookback}")
    kept_labels = labels.reset_index(drop=True).iloc[kept].reset_index(drop=True)
    return np.stack(seqs).astype(np.float32), np.asarray(targets, dtype=np.float32), kept_labels


def normalizer(seqs: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": np.nanmean(seqs, axis=(0, 1)).astype(np.float32),
        "std": (np.nanstd(seqs, axis=(0, 1)) + 1.0e-6).astype(np.float32),
    }


def apply_norm(seqs: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    return ((seqs - norm["mean"][None, None, :]) / norm["std"][None, None, :]).astype(np.float32)


def select_threshold(calib_labels: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    if calib_labels.empty:
        raise RuntimeError("cannot select sequence gate threshold from empty calibration labels")
    candidates = sorted(set(float(x) for x in np.quantile(scores, [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9])))
    rows: list[dict[str, Any]] = []
    min_trades = max(10, int(np.floor(len(calib_labels) * 0.25)))
    trade_return = calib_labels["trade_return"].astype(float).to_numpy()
    for threshold in candidates:
        keep = scores >= threshold
        picked_returns = trade_return[keep]
        pnl = float(picked_returns.sum()) if len(picked_returns) else 0.0
        wr = float((picked_returns > 0.0).mean()) if len(picked_returns) else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "trades": int(len(picked_returns)),
                "pnl": pnl,
                "pnl_pct": float(pnl * 100.0),
                "wr": wr,
                "avg_score": float(scores[keep].mean()) if keep.any() else 0.0,
                "eligible": int(len(picked_returns) >= min_trades),
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        eligible = rows
    selected = max(eligible, key=lambda row: (row["pnl"], row["wr"], row["trades"]))
    return {
        "selected": selected,
        "grid": rows,
        "selection_method": "train-only chronological calibration; threshold predicted counterfactual trade_return",
        "min_trades": int(min_trades),
    }


def train_tcn(
    *,
    train_seq: np.ndarray,
    train_y: np.ndarray,
    calib_seq: np.ndarray,
    calib_labels: pd.DataFrame,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[SequenceEntryTCN, dict[str, np.ndarray], dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    norm = normalizer(train_seq)
    x_train = apply_norm(train_seq, norm)
    y_train = train_y.astype(np.float32)
    model = SequenceEntryTCN(seq_dim=x_train.shape[-1]).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    loss_fn = nn.SmoothL1Loss()
    losses: list[float] = []
    model.train()
    for epoch in range(epochs):
        epoch_losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
    model = model.cpu().eval()
    calib_scores = predict_sequences(model, calib_seq, norm)
    threshold_payload = select_threshold(calib_labels, calib_scores)
    report = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "losses": losses,
        "calibration_score_summary": {
            "min": float(np.min(calib_scores)),
            "p25": float(np.quantile(calib_scores, 0.25)),
            "median": float(np.median(calib_scores)),
            "p75": float(np.quantile(calib_scores, 0.75)),
            "max": float(np.max(calib_scores)),
        },
        "threshold": threshold_payload,
    }
    return model, norm, report


def predict_sequences(model: SequenceEntryTCN, seqs: np.ndarray, norm: dict[str, np.ndarray], *, batch_size: int = 512) -> np.ndarray:
    x = apply_norm(seqs, norm)
    outs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            pred = model(torch.from_numpy(x[start : start + batch_size])).numpy()
            outs.append(pred.astype(np.float32))
    return np.concatenate(outs, axis=0)


def predict_one(artifact: SequenceGateArtifact, history: deque[np.ndarray]) -> float | None:
    if len(history) < artifact.lookback:
        return None
    seq = np.stack(list(history)[-artifact.lookback :]).astype(np.float32)
    x = ((seq - artifact.mean[None, :]) / artifact.std[None, :]).astype(np.float32)
    with torch.no_grad():
        return float(artifact.model(torch.from_numpy(x[None, :, :])).numpy()[0])


def save_artifact(artifact: SequenceGateArtifact, out_dir: Path) -> str:
    payload = {
        "model_id": MODEL_ID,
        "name": artifact.name,
        "lookback": int(artifact.lookback),
        "sample_mode": artifact.sample_mode,
        "feature_cols": artifact.feature_cols,
        "mean": artifact.mean,
        "std": artifact.std,
        "threshold": float(artifact.threshold),
        "threshold_payload": artifact.threshold_payload,
        "state_dict": artifact.model.state_dict(),
        "train_report": artifact.train_report,
    }
    path = out_dir / f"{artifact.name}.pt"
    torch.save(payload, path)
    return str(path)


def load_artifact(path: Path, *, threshold: float | None = None, name: str | None = None) -> SequenceGateArtifact:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    feature_cols = list(payload["feature_cols"])
    model = SequenceEntryTCN(seq_dim=len(feature_cols))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    loaded_threshold = float(payload["threshold"] if threshold is None else threshold)
    loaded_name = str(payload["name"] if name is None else name)
    return SequenceGateArtifact(
        name=loaded_name,
        lookback=int(payload["lookback"]),
        sample_mode=str(payload["sample_mode"]),
        feature_cols=feature_cols,
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
        threshold=loaded_threshold,
        threshold_payload=dict(payload["threshold_payload"]),
        model=model,
        train_report=dict(payload["train_report"]),
        path=str(path),
    )


def fit_sequence_gate_artifacts(
    *,
    tape: pd.DataFrame,
    labels: pd.DataFrame,
    feature_cols: list[str],
    gate_train_end: str,
    lookbacks: list[int],
    sample_modes: list[str],
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> list[SequenceGateArtifact]:
    labels = labels.copy()
    labels["entry_timestamp"] = pd.to_datetime(labels["entry_timestamp"], errors="raise")
    train_cut = pd.Timestamp(gate_train_end)
    artifacts: list[SequenceGateArtifact] = []
    for sample_mode in sample_modes:
        if sample_mode == "flat":
            mode_labels = labels[labels["candidate_was_in_position"].astype(int) == 0].reset_index(drop=True)
        elif sample_mode == "all":
            mode_labels = labels.reset_index(drop=True)
        else:
            raise RuntimeError(f"unknown sample_mode={sample_mode}")
        train_labels_raw = mode_labels[mode_labels["entry_timestamp"] < train_cut].reset_index(drop=True)
        calib_labels_raw = mode_labels[mode_labels["entry_timestamp"] >= train_cut].reset_index(drop=True)
        if train_labels_raw.empty or calib_labels_raw.empty:
            raise RuntimeError(
                f"empty sequence chronological split for {sample_mode}: train={len(train_labels_raw)} calib={len(calib_labels_raw)}"
            )
        for lookback in lookbacks:
            train_seq, train_y, train_labels = build_sequences(
                tape=tape,
                labels=train_labels_raw,
                feature_cols=feature_cols,
                lookback=lookback,
            )
            calib_seq, _, calib_labels = build_sequences(
                tape=tape,
                labels=calib_labels_raw,
                feature_cols=feature_cols,
                lookback=lookback,
            )
            model, norm, train_report = train_tcn(
                train_seq=train_seq,
                train_y=train_y,
                calib_seq=calib_seq,
                calib_labels=calib_labels,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                seed=seed + lookback + (1000 if sample_mode == "all" else 0),
                device=device,
            )
            threshold = float(train_report["threshold"]["selected"]["threshold"])
            name = f"tcn_seq_gate_L{lookback}_{sample_mode}"
            artifact = SequenceGateArtifact(
                name=name,
                lookback=int(lookback),
                sample_mode=sample_mode,
                feature_cols=feature_cols,
                mean=norm["mean"],
                std=norm["std"],
                threshold=threshold,
                threshold_payload=train_report["threshold"],
                model=model,
                train_report={
                    **train_report,
                    "train_summary": summarize_entries(train_labels),
                    "calibration_summary": summarize_entries(calib_labels),
                    "train_sequence_rows": int(len(train_seq)),
                    "calibration_sequence_rows": int(len(calib_seq)),
                },
                path="",
            )
            artifact.path = save_artifact(artifact, out_dir)
            artifacts.append(artifact)
            print(
                json.dumps(
                    {
                        "trained": name,
                        "lookback": int(lookback),
                        "sample_mode": sample_mode,
                        "train_sequences": int(len(train_seq)),
                        "calib_sequences": int(len(calib_seq)),
                        "threshold": threshold,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return artifacts


def simulate_with_sequence_gate(
    *,
    split: str,
    feature_path: Path,
    start: str,
    end: str,
    parent_variant: str,
    policy: dict[str, float],
    out_dir: Path,
    artifact: SequenceGateArtifact,
    allowed_sides: set[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent, work, atr, start_i, end_i, source_audit = frame_with_runtime_features(
        feature_path=feature_path,
        start=start,
        end=end,
        parent_variant=parent_variant,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    position: dict[str, Any] | None = None
    overlay_loss_streak = 0
    gate_counts: Counter[str] = Counter()
    history: deque[np.ndarray] = deque(maxlen=artifact.lookback)

    for i in range(start_i, end_i + 1):
        row = work.iloc[i]
        now = pd.Timestamp(row["timestamp"])
        if (i - start_i) % 5000 == 0:
            print(
                json.dumps(
                    {
                        "split": split,
                        "gate": artifact.name,
                        "done": int(i - start_i),
                        "total": int(end_i - start_i + 1),
                        "timestamp": str(now),
                        "closed": int(len(ledger)),
                        "position": None if position is None else int(position["side"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if position is not None:
            closed, _, _ = close_if_needed(work=work, position=position, i=i, policy=policy)
            if closed is not None:
                ledger.append(closed)
                parent.record_closed_trade(
                    exit_timestamp=closed["exit_timestamp"],
                    net_per_notional=float(closed["net_per_notional"]),
                )
                overlay_loss_streak = overlay_loss_streak + 1 if float(closed["trade_return"]) <= 0.0 else 0
                position = None
                continue

        parent_dec = parent_decision_at(parent, work.iloc[i : i + 1], float(atr[i]), now)
        parent_trace = dict(parent_dec.trace or {})
        in_position = position is not None
        signal = int(parent_dec.action) != 0 and int(parent_dec.side) != 0 and float(parent_dec.notional_exposure) > EPS
        candidate = not in_position and signal
        features = feature_row(
            work=work,
            atr=atr,
            i=i,
            parent_dec=parent_dec,
            parent_trace=parent_trace,
            overlay_loss_streak=overlay_loss_streak,
        )
        history.append(feature_vector(features, artifact.feature_cols))

        gate_score = np.nan
        gate_reason = ""
        if candidate:
            if allowed_sides is not None and int(parent_dec.side) not in allowed_sides:
                gate_reason = "sequence_entry_gate_side_veto"
                gate_counts[gate_reason] += 1
            else:
                score = predict_one(artifact, history)
                if score is None:
                    gate_reason = "sequence_entry_gate_warmup_veto"
                    gate_counts[gate_reason] += 1
                else:
                    gate_score = float(score)
                    if gate_score < artifact.threshold:
                        gate_reason = "sequence_entry_gate_veto"
                        gate_counts[gate_reason] += 1
                    else:
                        gate_counts["sequence_entry_gate_allow"] += 1

        decisions.append(
            {
                "split": split,
                "row": int(i),
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "action": int(parent_dec.action),
                "side": int(parent_dec.side),
                "notional": float(parent_dec.notional_exposure),
                "margin_fraction": float(parent_dec.position_fraction),
                "leverage": float(parent_dec.leverage),
                "quality_score": float(parent_dec.quality_score),
                "confidence": float(parent_dec.confidence),
                "router_expert": str(parent_dec.router_expert),
                "ignored_because_in_position": bool(in_position),
                "sequence_gate": artifact.name,
                "sequence_gate_score": gate_score,
                "sequence_gate_threshold": float(artifact.threshold),
                "sequence_gate_reason": gate_reason,
                "ledger_replay_used": bool(parent_trace.get("ledger_replay_used", True)),
                "source_parent_live_native_adapter": bool(parent_trace.get("source_parent_live_native_adapter", False)),
                "source_parent_policy_row": int(parent_trace.get("source_parent_policy_row", -999)),
                "fresh_forward_bar_by_bar": True,
                "future_rows_used_for_entry": False,
            }
        )
        if not candidate or gate_reason:
            continue

        leverage = float(parent_dec.leverage)
        notional = min(float(parent_dec.notional_exposure), float(policy["cap"]))
        if notional <= EPS:
            continue
        margin = notional / max(leverage, EPS)
        if abs(margin * leverage - notional) > 1.0e-8:
            raise RuntimeError("sequence gate replay violates notional=margin_fraction*leverage")
        position = {
            "entry_i": int(i),
            "side": int(parent_dec.side),
            "entry_price": float(row["close"]),
            "notional": float(notional),
            "base_parent_notional": float(parent_dec.notional_exposure),
            "margin_fraction": float(margin),
            "leverage": float(leverage),
            "tp_price_move": float(policy["tp"]),
            "sl_price_move": float(policy["sl"]),
            "roundtrip_cost": float(ROUNDTRIP_COST_DEFAULT),
            "router_expert": str(parent_dec.router_expert),
            "parent_quality_score": float(parent_dec.quality_score),
            "parent_confidence": float(parent_dec.confidence),
            "overlay_loss_scale": 1.0,
        }

    decisions_df = pd.DataFrame(decisions)
    ledger_df = pd.DataFrame(ledger)
    safe_name = f"{split}_{artifact.name}"
    decisions_path = out_dir / f"{safe_name}_decisions.csv"
    ledger_path = out_dir / f"{safe_name}_ledger.csv"
    decisions_df.to_csv(decisions_path, index=False)
    ledger_df.to_csv(ledger_path, index=False)
    metrics = summarize_ledger(ledger_df, decisions_df)
    metrics["sequence_gate_counts"] = dict(gate_counts)
    return metrics, {
        "source": source_audit,
        "decisions": str(decisions_path),
        "ledger": str(ledger_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(args.torch_threads))
    device = torch.device("cuda" if bool(args.cuda) and torch.cuda.is_available() else "cpu")
    policy = load_policy(args)
    lookbacks = parse_csv_list(args.lookbacks, cast=int)
    sample_modes = parse_csv_list(args.sample_modes, cast=str)

    train_tape, train_labels, train_metrics, train_split_artifacts = collect_train_tape_and_labels(
        split="train_live_native",
        feature_path=Path(args.train_features),
        start=args.train_start,
        end=args.train_end,
        parent_variant=args.parent_runtime_variant,
        policy=policy,
        out_dir=out_dir,
        save_tape=bool(args.save_tape),
    )
    feature_cols = feature_columns_from_tape(train_tape)
    (out_dir / "sequence_feature_cols.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    gate_artifacts = fit_sequence_gate_artifacts(
        tape=train_tape,
        labels=train_labels,
        feature_cols=feature_cols,
        gate_train_end=args.gate_train_end,
        lookbacks=lookbacks,
        sample_modes=sample_modes,
        out_dir=out_dir,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        device=device,
    )

    results: dict[str, Any] = {}
    split_artifacts: dict[str, Any] = {}
    integrity: dict[str, int] = {}
    for artifact in gate_artifacts:
        validation_metrics, validation_artifacts = simulate_with_sequence_gate(
            split="validation",
            feature_path=Path(args.features_2025),
            start=args.validation_start,
            end=args.validation_end,
            parent_variant=args.parent_runtime_variant,
            policy=policy,
            out_dir=out_dir,
            artifact=artifact,
        )
        oos_metrics, oos_artifacts = simulate_with_sequence_gate(
            split="oos",
            feature_path=Path(args.oos_features),
            start=args.oos_start,
            end=args.oos_end,
            parent_variant=args.parent_runtime_variant,
            policy=policy,
            out_dir=out_dir,
            artifact=artifact,
        )
        results[artifact.name] = {
            "lookback": int(artifact.lookback),
            "sample_mode": artifact.sample_mode,
            "threshold": float(artifact.threshold),
            "artifact": artifact.path,
            "train": artifact.train_report,
            "validation": validation_metrics,
            "oos": oos_metrics,
        }
        split_artifacts[artifact.name] = {
            "validation": validation_artifacts,
            "oos": oos_artifacts,
        }
        integrity[f"{artifact.name}_validation_ledger_replay_trace_count"] = int(validation_metrics["ledger_replay_trace_count"])
        integrity[f"{artifact.name}_validation_non_live_native_trace_count"] = int(validation_metrics["non_live_native_trace_count"])
        integrity[f"{artifact.name}_validation_non_minus_one_policy_row_count"] = int(validation_metrics["non_minus_one_policy_row_count"])
        integrity[f"{artifact.name}_oos_ledger_replay_trace_count"] = int(oos_metrics["ledger_replay_trace_count"])
        integrity[f"{artifact.name}_oos_non_live_native_trace_count"] = int(oos_metrics["non_live_native_trace_count"])
        integrity[f"{artifact.name}_oos_non_minus_one_policy_row_count"] = int(oos_metrics["non_minus_one_policy_row_count"])

    ranked = sorted(
        [
            {
                "name": name,
                "validation_compound_pnl_pct": float(payload["validation"]["compound_pnl_pct"]),
                "validation_compound_mdd_pct": float(payload["validation"]["compound_mdd_pct"]),
                "validation_trades": int(payload["validation"]["trades"]),
                "oos_compound_pnl_pct": float(payload["oos"]["compound_pnl_pct"]),
                "oos_compound_mdd_pct": float(payload["oos"]["compound_mdd_pct"]),
                "oos_trades": int(payload["oos"]["trades"]),
            }
            for name, payload in results.items()
        ],
        key=lambda row: (row["validation_compound_pnl_pct"], row["oos_compound_pnl_pct"]),
        reverse=True,
    )

    report = {
        "schema_version": "omega462.live_native_sequence_entry_gate.train_eval.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "parent_runtime_variant": str(args.parent_runtime_variant),
        "policy": policy,
        "device": str(device),
        "training_contract": {
            "train_tape_start": args.train_start,
            "train_tape_end_exclusive": args.train_end,
            "gate_train_end": args.gate_train_end,
            "validation_rows_used_for_training": False,
            "oos_rows_used_for_training": False,
            "trade_ledgers_used_as_model_input": False,
            "labels_use_future_prices_only_inside_train_split": True,
            "sequence_inputs_use_rows_up_to_current_t_only": True,
            "parent_signal_rows_only_for_gate_training": True,
        },
        "fresh_forward_definition": "fixed split, causal 5m bar-by-bar replay; sequence gate sees only current/past live-native feature rows buffered inside the split",
        "train_live_native": {
            "metrics": train_metrics,
            "artifacts": train_split_artifacts,
            "feature_count": int(len(feature_cols)),
            "label_summary": summarize_entries(train_labels),
        },
        "results": results,
        "ranked": ranked,
        "split_artifacts": split_artifacts,
        "integrity": integrity,
        "artifacts": {
            "out_dir": str(out_dir),
            "report": str(out_dir / "report.json"),
            "feature_cols": str(out_dir / "sequence_feature_cols.json"),
        },
    }
    write_json(out_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", default=str(DEFAULT_TRAIN_FEATURES))
    parser.add_argument("--features-2025", default=str(DEFAULT_FEATURES_2025))
    parser.add_argument("--oos-features", default=str(DEFAULT_OOS_FEATURES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--parent-runtime-variant", choices=["source_v5", "cap220_no_v5"], default="source_v5")
    parser.add_argument("--train-start", default="2024-01-01 00:00:00")
    parser.add_argument("--train-end", default="2025-09-01 00:00:00")
    parser.add_argument("--gate-train-end", default="2025-05-01 00:00:00")
    parser.add_argument("--validation-start", default="2025-09-01 00:00:00")
    parser.add_argument("--validation-end", default="2026-01-01 00:00:00")
    parser.add_argument("--oos-start", default="2026-01-01 00:00:00")
    parser.add_argument("--oos-end", default="2026-04-01 00:00:00")
    parser.add_argument("--tp", type=float, default=0.026)
    parser.add_argument("--sl", type=float, default=0.014)
    parser.add_argument("--cap", type=float, default=4.106)
    parser.add_argument("--max-hold-hours", type=float, default=90.0)
    parser.add_argument("--lookbacks", default="24,48,96")
    parser.add_argument("--sample-modes", default="flat,all")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8.0e-4)
    parser.add_argument("--seed", type=int, default=260703)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--save-tape", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"ranked": report["ranked"], "integrity": report["integrity"]}, ensure_ascii=False, indent=2, default=json_default), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
