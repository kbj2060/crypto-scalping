#!/usr/bin/env python3
"""BTC v3 research candidate: Mamba in-trade exit-hazard sidecar.

Parallel candidate to Stage 1's fixed ATR stop/arm/trail/time-exit contract -- does not replace
it. Reuses the CBlock/CMBlock multi-scale architecture verbatim from CryptoMambaRegimePred
(scripts/train_regime3_cryptomamba_pred_20260531.py), only swapping its 3-class future-regime head
for a single-logit exit-hazard head and its label for a trade-specific one.

For every Stage 1 sparse event (docs/model_contracts/btc_v3_stage1_sparse_events_20260714.md),
re-simulates the exact same ATR stop/arm/trail/time-exit contract (imports
_simulate_event_outcome from build_btc_v3_sparse_event_dataset_20260714.py unmodified) to recover
each event's entry_fill_i/exit_fill_i on the 5-minute tape, then builds one training example per
in-trade 5-minute bar: input = trailing --seq-len 5m OHLC-derived causal features ending at that
bar, label = whether the event's REAL exit_reason fires within the next --horizon bars. This is a
valid supervised target (mirrors CryptoMamba's own "future regime id" target) because it is the
already-simulated future of THIS SPECIFIC event under the frozen exit contract, not new information
a live model would not eventually observe -- at inference time the model only ever sees bars <= t.

Enforces docs/model_contracts/btc_v3_holdout_policy_20260714.md: refuses to build past HOLDOUT_START.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from mamba_ssm import Mamba
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.retrain_clean_regime_hmm_20260517 import _json_default  # noqa: E402
import train_eval_btc_v2_regime_trendscan_20260714 as btc_v2  # noqa: E402
import build_btc_v3_sparse_event_dataset_20260714 as sparse_mod  # noqa: E402

MODEL_ID = "btc_v3_mamba_exit_hazard_20260715"
HOLDOUT_START = pd.Timestamp("2026-07-14 00:00:00")
OUT_DIR = ROOT / "tmp/causal_regen_20260516/btc_v3_hmm_mamba_candidate_20260715"
VAL_START = pd.Timestamp("2025-10-01")

SEQ_LEN = 60
HORIZON = 6
D_MODEL = 128
D_STATE = 32
N_CBLOCKS = 4
N_CMBLOCKS = 2
DROPOUT = 0.10
LR = 5e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
EPOCHS = 6
PATIENCE = 2
TRAIN_STRIDE = 6
MAX_IN_TRADE_BARS = 500
SEED = 20260715


def _in_trade_features(arrays: dict[str, np.ndarray], entry_price: float, entry_atr: float, side: int, entry_fill_i: int, n: int) -> np.ndarray:
    """Causal per-bar features derived only from the 5m OHLC tape and the event's own entry
    context (price/ATR at signal time) -- no new engineered dataset, matching Stage 2's finding
    that the existing BTC feature contract needs no rebuild."""
    close = arrays["close"]
    move = np.where(
        side > 0,
        (close * (1.0 - btc_v2.SLIP_RATE) - entry_price) / entry_price,
        (entry_price - close * (1.0 + btc_v2.SLIP_RATE)) / entry_price,
    )
    move_atr = move / max(entry_atr, 1e-6)
    peak_move_atr = np.maximum.accumulate(np.where(np.arange(n) >= entry_fill_i, move_atr, -np.inf))
    peak_move_atr = np.where(np.isfinite(peak_move_atr), peak_move_atr, 0.0)
    drawdown_from_peak_atr = peak_move_atr - move_atr
    log_close = np.log(np.clip(close, 1e-9, None))
    bar_return = np.diff(log_close, prepend=log_close[0])
    ret_series = pd.Series(bar_return)
    rvol_12 = ret_series.rolling(12, min_periods=1).std().fillna(0.0).to_numpy()
    hold_bars = np.clip(np.arange(n) - entry_fill_i, 0, None).astype(np.float64)
    hold_bars_frac = np.clip(hold_bars / btc_v2.MAX_HOLD_BARS, 0.0, 1.0)
    return np.stack([move_atr, peak_move_atr, drawdown_from_peak_atr, bar_return, rvol_12, hold_bars_frac], axis=1)


# NOTE: stop_dist/trail_dist (distance to the fixed ATR contract's own stop/trail levels) were
# deliberately excluded here -- they are near-direct linear re-encodings of the exit-hazard label
# itself (whether the fixed contract's exit triggers soon), which made the first candidate run's
# 0.91 val AUC untrustworthy. See docs/model_contracts/btc_v3_hmm_mamba_candidate_20260715.md.
FEATURE_NAMES = ["move_atr", "peak_move_atr", "drawdown_from_peak_atr", "bar_return", "rvol_12", "hold_bars_frac"]


def _collect_events(history_end: pd.Timestamp) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], np.ndarray]:
    """Reproduces Stage 1's event-detection + simulation exactly (same ts_action transitions, same
    _simulate_event_outcome), but keeps the per-event fill indices/entry context that Stage 1's
    output parquet does not persist."""
    hourly, feature_columns = btc_v2._read_hourly()
    hourly = hourly.loc[hourly["timestamp"] <= history_end].reset_index(drop=True)
    five_minute = btc_v2._read_five_minute()
    five_minute = five_minute.loc[five_minute["timestamp"] <= history_end].reset_index(drop=True)

    action = hourly["ts_action"].to_numpy()
    is_event = (action != 0) & (action != np.roll(action, 1))
    is_event[0] = bool(action[0] != 0)
    event_hours = hourly.loc[is_event].reset_index(drop=True)

    five_ts = five_minute["timestamp"].to_numpy()
    arrays = {c: pd.to_numeric(five_minute[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(five_ts)

    events: list[dict[str, Any]] = []
    for _, ev in event_hours.iterrows():
        entry_avail = ev["timestamp"] + pd.Timedelta(hours=1)
        idx = np.searchsorted(five_ts, np.datetime64(entry_avail), side="left")
        if idx >= n:
            continue
        side = 1 if int(ev["ts_action"]) == 1 else -1
        atr_local = np.full(n, float(ev["atr_pct"]), dtype=np.float64)
        outcome = sparse_mod._simulate_event_outcome(arrays, atr_local, int(idx), side)
        if not outcome.get("valid"):
            continue
        events.append({
            "event_hour_timestamp": ev["timestamp"],
            "side": side,
            "entry_price": outcome["entry_price"],
            "entry_atr": float(ev["atr_pct"]),
            "entry_fill_i": outcome["entry_fill_i"],
            "exit_fill_i": outcome["exit_fill_i"],
            "exit_reason": outcome["reason"],
        })
    return events, arrays, five_ts


class HazardDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, idx: np.ndarray, seq_len: int) -> None:
        self.x = x
        self.y = y
        self.idx = idx.astype(np.int64)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return int(len(self.idx))

    def __getitem__(self, item: int):
        end = int(self.idx[item])
        start = end - self.seq_len + 1
        seq = self.x[start : end + 1]
        return torch.from_numpy(seq), torch.tensor(float(self.y[end]), dtype=torch.float32)


class CMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class CBlock(nn.Module):
    def __init__(self, d_model: int, n_cmblocks: int, seq_len_in: int, seq_len_out: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([CMBlock(d_model, d_state, d_conv, expand) for _ in range(int(n_cmblocks))])
        self.seq_proj = nn.Linear(int(seq_len_in), int(seq_len_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.seq_proj(x.permute(0, 2, 1)).permute(0, 2, 1)


class CryptoMambaExitHazard(nn.Module):
    """Same CBlock/CMBlock multi-scale pyramid as CryptoMambaRegimePred; head replaced with a
    single-logit exit-hazard output instead of a 3-class regime softmax."""

    def __init__(self, n_features: int, seq_len: int, d_model: int, n_cblocks: int, n_cmblocks: int, d_state: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(n_features), int(d_model))
        seq_lens = [int(seq_len)]
        for _ in range(int(n_cblocks)):
            seq_lens.append(max(seq_lens[-1] * 3 // 4, 8))
        self.cblocks = nn.ModuleList(
            [CBlock(d_model, n_cmblocks, seq_lens[i], seq_lens[i + 1], d_state=d_state, d_conv=4, expand=2) for i in range(int(n_cblocks))]
        )
        self.merge = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(int(d_model) * int(n_cblocks), int(d_model)), nn.GELU(), nn.LayerNorm(int(d_model)))
        self.head = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(int(d_model), 64), nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        outs: list[torch.Tensor] = []
        for block in self.cblocks:
            z = block(z)
            outs.append(z[:, -1, :])
        return self.head(self.merge(torch.cat(outs, dim=-1))).squeeze(-1)


@torch.no_grad()
def _predict(model: nn.Module, x: np.ndarray, y: np.ndarray, idx: np.ndarray, seq_len: int, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(HazardDataset(x, y, idx, seq_len), batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    outs: list[np.ndarray] = []
    for xb, _ in loader:
        logits = model(xb.to(device, non_blocking=True))
        outs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def build(history_end: pd.Timestamp) -> dict[str, Any]:
    if history_end >= HOLDOUT_START:
        raise RuntimeError(
            f"history_end={history_end} >= HOLDOUT_START={HOLDOUT_START} -- refusing per "
            f"docs/model_contracts/btc_v3_holdout_policy_20260714.md"
        )
    print("stage=collect_events_and_resimulate_exit_contract", flush=True)
    events, arrays, five_ts = _collect_events(history_end)
    n = len(five_ts)
    print(f"stage=events_collected n_events={len(events)}", flush=True)

    print("stage=build_per_event_feature_sequences", flush=True)
    feat_full = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    covered = np.zeros(n, dtype=bool)
    label_full = np.zeros(n, dtype=np.float64)
    train_mask_full = np.zeros(n, dtype=bool)
    idx_pool: list[int] = []

    for ev in events:
        entry_fill_i = int(ev["entry_fill_i"])
        exit_fill_i = int(ev["exit_fill_i"])
        end_i = min(exit_fill_i, entry_fill_i + MAX_IN_TRADE_BARS)
        feats = _in_trade_features(arrays, ev["entry_price"], ev["entry_atr"], ev["side"], entry_fill_i, n)
        for t in range(max(entry_fill_i, SEQ_LEN - 1), end_i + 1, TRAIN_STRIDE):
            feat_full[t] = feats[t]
            covered[t] = True
            label_full[t] = 1.0 if (exit_fill_i - t) <= HORIZON else 0.0
            train_mask_full[t] = ev["event_hour_timestamp"] < VAL_START
            idx_pool.append(t)

    idx_pool_arr = np.unique(np.asarray(idx_pool, dtype=np.int64))
    print(f"stage=samples_built n_samples={len(idx_pool_arr)}", flush=True)
    if len(idx_pool_arr) < 200:
        raise RuntimeError(f"too few in-trade samples: {len(idx_pool_arr)}")

    scaler = StandardScaler()
    train_idx = idx_pool_arr[train_mask_full[idx_pool_arr]]
    val_idx = idx_pool_arr[~train_mask_full[idx_pool_arr]]
    scaler.fit(feat_full[train_idx])
    x = scaler.transform(feat_full).astype(np.float32)
    x = np.nan_to_num(x)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for CryptoMambaExitHazard training")
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = CryptoMambaExitHazard(len(FEATURE_NAMES), SEQ_LEN, D_MODEL, N_CBLOCKS, N_CMBLOCKS, D_STATE, DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pos_weight = torch.tensor([float((label_full[train_idx] == 0).sum() / max((label_full[train_idx] == 1).sum(), 1.0))], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(HazardDataset(x, label_full, train_idx, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    best_state = None
    best = float("inf")
    bad = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_proba = _predict(model, x, label_full, val_idx, SEQ_LEN, BATCH_SIZE * 2, device)
        val_loss = float(nn.functional.binary_cross_entropy(torch.from_numpy(val_proba).clamp(1e-6, 1 - 1e-6), torch.from_numpy(label_full[val_idx]).float()))
        val_auc = float(roc_auc_score(label_full[val_idx], val_proba)) if len(np.unique(label_full[val_idx])) > 1 else None
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_auc": val_auc}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best - 1e-4:
            best = val_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{MODEL_ID}.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "feature_names": FEATURE_NAMES,
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "d_model": D_MODEL,
            "d_state": D_STATE,
            "cblocks": N_CBLOCKS,
            "cmblocks": N_CMBLOCKS,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        model_path,
    )

    val_proba_final = _predict(model, x, label_full, val_idx, SEQ_LEN, BATCH_SIZE * 2, device)
    val_auc_final = float(roc_auc_score(label_full[val_idx], val_proba_final)) if len(np.unique(label_full[val_idx])) > 1 else None

    report = {
        "model_id": MODEL_ID,
        "status": "research_candidate_not_live",
        "supersedes": "none (parallel to stage1's fixed ATR exit contract)",
        "history_end": str(history_end),
        "holdout_start": str(HOLDOUT_START),
        "val_start": str(VAL_START),
        "architecture": {"type": "CryptoMamba C-Block Merge (single-logit exit hazard head)", "seq_len": SEQ_LEN, "horizon": HORIZON, "d_model": D_MODEL, "d_state": D_STATE, "cblocks": N_CBLOCKS, "cmblocks": N_CMBLOCKS},
        "feature_names": FEATURE_NAMES,
        "n_events": len(events),
        "n_samples": int(len(idx_pool_arr)),
        "n_train_samples": int(len(train_idx)),
        "n_val_samples": int(len(val_idx)),
        "train_stride": TRAIN_STRIDE,
        "max_in_trade_bars": MAX_IN_TRADE_BARS,
        "history": history,
        "validation": {"val_loss": best, "val_auc": val_auc_final, "val_positive_rate": float(label_full[val_idx].mean()) if len(val_idx) else None},
        "artifacts": {"model": str(model_path)},
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "notes": [
            "Label is this specific event's own already-simulated future exit under the frozen ATR contract, analogous to CryptoMambaRegimePred's future-regime-id target.",
            "Inputs are derived only from the 5m OHLC tape plus the event's own entry price/ATR context -- no new engineered feature dataset was built.",
        ],
    }
    report_path = out_dir / "btc_v3_mamba_exit_hazard_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default))
    print(f"stage=done model={model_path} report={report_path} val_auc={val_auc_final}", flush=True)
    return report


def main() -> int:
    history_end = pd.Timestamp("2026-07-12 23:59:59")
    build(history_end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
