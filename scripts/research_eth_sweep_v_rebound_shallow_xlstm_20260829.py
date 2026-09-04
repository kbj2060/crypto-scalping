#!/usr/bin/env python3
"""User's final requested comparison: a shallow xLSTM (Beck et al. 2024, "xLSTM: Extended Long
Short-Term Memory") against the established TabPFN result (VAL AUC 0.6423/OOS AUC 0.6566).

Architecture note: implements the sLSTM cell (scalar memory, exponential input-gate + sigmoid
forget-gate, log-space stabilizer to prevent overflow) directly in plain PyTorch rather than
installing the official `xlstm` pip package -- that package pulls in `mlstm_kernels` (compiled
CUDA kernels via `ninja`), which is unnecessary build/runtime risk on the shared live-bot server
for a small, "shallow" one-off test; the sLSTM formulation itself is a few dozen lines and is
implemented faithfully to the paper (see sLSTMCell docstring for the exact equations).

Input representation (REQUIRED architecture change vs the tabular Tier0+rsi models -- xLSTM is a
sequence model): a 24-bar (2h) trailing window of PER-BAR features ending at the sweep bar, not a
single flat feature vector. Reuses build_eth_5m_sweep_v_rebound_features_tier0_20260829.py::
build_indicator_frame (same causal indicator computation, unmodified) to get these per-bar
values for the FULL history, then slices a window per sweep event. Tier0's 2 event-specific
(not meaningfully defined per-bar-in-history) features -- sweep_penetration_atr, range_width_pct
-- are dropped from the per-timestep vector (they describe the sweep bar's relationship to the
level, not a generic bar state); is_downside is appended as a constant channel across the
sequence. That leaves 19 generic per-bar Tier0 indicators + rsi = 20 sequence features + 1
constant direction channel = 21 total per timestep.

Same TRAIN(<2025-09-01)/VAL(2025-09-01..12-31)/OOS(2026-01-01..03-31) split as every other model
in this lineage. Single-seed cheap-gate first (matches this project's TabM precedent: a clearly
negative cheap-gate that agrees with prior expectations doesn't need 5-seed confirmation).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TIER0_BUILDER = ROOT / "scripts/build_eth_5m_sweep_v_rebound_features_tier0_20260829.py"
LABEL_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"
RSI_SOURCES = [ROOT / f"data/splits/year_oos/training_features_{y}.csv" for y in ("2024", "2025", "2026_rebuilt")]

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
SEED = 20260829
SEQ_LEN = 24  # 2h trailing window

SEQ_FEATURES = [
    "atr", "atr_percentile_864", "hour_utc", "weekday", "delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile",
    "rsi",
]
N_FEATURES = len(SEQ_FEATURES) + 1  # + is_downside constant channel

HIDDEN = 32
N_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 256
LR = 2e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 60
PATIENCE = 10


def load_tier0_builder():
    spec = importlib.util.spec_from_file_location("tier0_builder_xlstm_20260829", TIER0_BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class sLSTMCell(nn.Module):
    """sLSTM cell, Beck et al. 2024 (arXiv:2405.04517) Sec 2.2 -- scalar memory, exponential
    input gate + sigmoid forget gate, log-space stabilizer state m_t to prevent exp() overflow:
      z_t = tanh(W_z x_t + R_z h_{t-1})                          cell input
      log(i_t) = W_i x_t + R_i h_{t-1}                            input gate, exponential (in log-space directly)
      log(f_t) = logsigmoid(W_f x_t + R_f h_{t-1})                forget gate, sigmoid (log-space)
      o_t = sigmoid(W_o x_t + R_o h_{t-1})                        output gate
      m_t = max(log(f_t) + m_{t-1}, log(i_t))                     stabilizer
      i'_t = exp(log(i_t) - m_t); f'_t = exp(log(f_t) + m_{t-1} - m_t)
      c_t = f'_t c_{t-1} + i'_t z_t;  n_t = f'_t n_{t-1} + i'_t    cell + normalizer state
      h_t = o_t * (c_t / max(n_t, eps))
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.R = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x_seq.shape
        device = x_seq.device
        h = torch.zeros(batch, self.hidden_size, device=device)
        c = torch.zeros(batch, self.hidden_size, device=device)
        n = torch.zeros(batch, self.hidden_size, device=device)
        m = torch.full((batch, self.hidden_size), -1e6, device=device)
        outputs = []
        for t in range(seq_len):
            gates = self.W(x_seq[:, t, :]) + self.R(h)
            z_tilde, i_tilde, f_tilde, o_tilde = gates.chunk(4, dim=-1)
            z = torch.tanh(z_tilde)
            o = torch.sigmoid(o_tilde)
            log_f = F.logsigmoid(f_tilde)
            m_new = torch.maximum(log_f + m, i_tilde)  # m here is still m_{t-1}
            i_prime = torch.exp(i_tilde - m_new)
            f_prime = torch.exp(log_f + m - m_new)     # m here is still m_{t-1} (correct)
            c = f_prime * c + i_prime * z
            n = f_prime * n + i_prime
            h = o * (c / torch.clamp(n, min=1e-6))
            m = m_new
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class ShallowXLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.cells = nn.ModuleList([sLSTMCell(hidden_size, hidden_size) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x_seq)
        for cell in self.cells:
            h = cell(h) + h  # residual, matches this project's TabM-style post-norm-residual convention
            h = self.dropout(h)
        h = self.norm(h[:, -1, :])  # final timestep's hidden state
        return self.head(h).squeeze(-1)


def build_sequences() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    tier0_mod = load_tier0_builder()
    sweep_impl = tier0_mod.load_sweep_impl()
    frame = tier0_mod.build_indicator_frame(sweep_impl)
    frame["atr"] = sweep_impl.add_causal_columns(sweep_impl.load_5m(tier0_mod.SOURCE))["atr"]
    frame["atr_percentile_864"] = frame["atr"].rolling(864, min_periods=864).rank(pct=True)
    frame["hour_utc"] = frame["timestamp"].dt.hour
    frame["weekday"] = frame["timestamp"].dt.weekday

    rsi_frames = []
    for p in RSI_SOURCES:
        f = pd.read_csv(p, usecols=["timestamp", "rsi"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        rsi_frames.append(f)
    rsi = pd.concat(rsi_frames, ignore_index=True).drop_duplicates("timestamp")
    frame = frame.merge(rsi, on="timestamp", how="left")

    labels = pd.read_csv(LABEL_CSV)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
    feat_matrix = frame[SEQ_FEATURES].to_numpy(dtype=np.float64)

    sequences, keep_mask = [], []
    for cidx in labels["candidate_index"].to_numpy():
        window = feat_matrix[cidx - SEQ_LEN + 1: cidx + 1]
        if len(window) < SEQ_LEN or not np.isfinite(window).all():
            keep_mask.append(False)
            sequences.append(None)
            continue
        keep_mask.append(True)
        sequences.append(window)

    keep_mask = np.array(keep_mask)
    labels = labels.loc[keep_mask].reset_index(drop=True)
    seq_array = np.stack([s for s, k in zip(sequences, keep_mask) if k])  # (N, SEQ_LEN, n_seq_features)

    is_down = (labels["side"] == "downside").to_numpy(dtype=np.float64)
    direction_channel = np.repeat(is_down[:, None, None], SEQ_LEN, axis=1)  # (N, SEQ_LEN, 1)
    x = np.concatenate([seq_array, direction_channel], axis=2)
    y = labels["label"].to_numpy(dtype=np.float64)
    return x, y, labels["timestamp"].to_numpy(), labels


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("building sequence dataset (this reuses build_indicator_frame, may take a minute)...")
    x, y, timestamps_raw, labels_df = build_sequences()
    timestamps = pd.to_datetime(timestamps_raw, utc=True)
    print(f"total usable sequences: {len(x)} (of {len(pd.read_csv(LABEL_CSV))} raw sweep events, "
          f"rest dropped for insufficient {SEQ_LEN}-bar history or NaN warmup)")

    window_end = timestamps + pd.Timedelta(minutes=30)
    train_mask = (timestamps < VAL_START) & (window_end < VAL_START)
    val_mask = (timestamps >= VAL_START) & (timestamps <= VAL_END) & (window_end < OOS_START)
    oos_mask = (timestamps >= OOS_START) & (timestamps <= OOS_END)
    print(f"train n={train_mask.sum()}  val n={val_mask.sum()}  oos n={oos_mask.sum()}")

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_oos, y_oos = x[oos_mask], y[oos_mask]

    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std[std < 1e-8] = 1.0

    def to_tensor(arr, labels_arr):
        arr = (arr - mean) / std
        return torch.tensor(arr, dtype=torch.float32, device=device), torch.tensor(labels_arr, dtype=torch.float32, device=device)

    xt_train, yt_train = to_tensor(x_train, y_train)
    xt_val, yt_val = to_tensor(x_val, y_val)
    xt_oos, yt_oos = to_tensor(x_oos, y_oos)

    model = ShallowXLSTM(N_FEATURES, HIDDEN, N_LAYERS, DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_auc, best_state, bad_epochs = -1.0, None, 0
    n_train = len(xt_train)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            opt.zero_grad()
            logits = model(xt_train[idx])
            loss = F.binary_cross_entropy_with_logits(logits, yt_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_logits = model(xt_val).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_logits)
        print(f"  epoch {epoch:3d}  train_loss={total_loss/n_train:.4f}  val_auc={val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc, best_state, bad_epochs = val_auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  early stop at epoch {epoch} (best val_auc={best_val_auc:.4f} at earlier epoch)")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_proba = torch.sigmoid(model(xt_val)).cpu().numpy()
        oos_proba = torch.sigmoid(model(xt_oos)).cpu().numpy()
    val_auc = roc_auc_score(y_val, val_proba)
    oos_auc = roc_auc_score(y_oos, oos_proba)
    val_acc = ((val_proba >= 0.5).astype(int) == y_val).mean()
    oos_acc = ((oos_proba >= 0.5).astype(int) == y_oos).mean()
    val_naive = max(y_val.mean(), 1 - y_val.mean())
    oos_naive = max(y_oos.mean(), 1 - y_oos.mean())

    print(f"\n=== SUMMARY: shallow xLSTM (sLSTM, {N_LAYERS} layers, hidden={HIDDEN}, seq_len={SEQ_LEN}) ===")
    print(f"  VAL  AUC {val_auc:.4f}   acc {val_acc:.4f} (naive {val_naive:.4f}, lift {val_acc-val_naive:+.4f})")
    print(f"  OOS  AUC {oos_auc:.4f}   acc {oos_acc:.4f} (naive {oos_naive:.4f}, lift {oos_acc-oos_naive:+.4f})")
    print(f"\n=== FOR COMPARISON ===")
    print(f"  TabPFN (Tier0+rsi, current SOTA):  VAL AUC 0.6423+/-0.0008   OOS AUC 0.6566+/-0.0002")
    print(f"  GBM (Tier0):                       VAL AUC 0.6222            OOS AUC 0.6425")
    print(f"  TabM (Tier0, downsized, REJECTED): VAL AUC 0.6108            OOS AUC 0.6232")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
