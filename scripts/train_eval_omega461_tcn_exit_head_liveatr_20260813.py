#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey2 priority #5 (last item in the priority queue): replace h48qual's
exit_head (currently TabM, live-ATR relabel recipe -- scripts/research_eth_omega461_exit_head_
liveatr_relabel_20260813.py, see docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_
20260813.md -- the current confirmed Odyssey2 baseline) with a causal TCN (temporal convolutional
network) that sees a WINDOW of recent bars instead of a single-bar snapshot, on the IDENTICAL
dataset/label priority #4 (GBDT) used, to see whether sequence context around the exit decision
helps once the label recipe is held fixed. zig075 is not touched by this script.

=== Why this needs a genuinely different runtime shape than #4 (GBDT) ===
GBDT (research_eth_omega461_gbdt_exit_head_val_20260813.py) could duck-type straight into
train_eval_omega4_2_risk_sidecar_20260622._predict_exit_prob_one because that function only ever
hands the model a SINGLE feature row (`row = base_np[row_i]`). A TCN structurally needs a WINDOW of
history per decision. This training script therefore builds a windowed dataset (same rows/labels as
#4, each row additionally carries the WINDOW of the 102 base_cols market features ending at that
row's absolute bar index) and trains a TCNExitClassifier per regime expert. The companion VAL script
(research_eth_omega461_tcn_exit_head_val_20260813.py) is where the runtime-injection duck-typing
problem this creates is actually solved (windowed copies of _predict_exit_prob_one/replay_exit_
variant/greedy_replay) -- see that script's module docstring.

=== Dataset -- reused via IMPORT, not reimplemented ===
Calls train_eval_omega461_gbdt_exit_head_liveatr_20260813._build_dataset(max_candidates) UNCHANGED
-- the exact function #4 used (same seed=260813, same max_candidates=1500, same live-ATR barrier
recipe, same dataset-vs-report.json reference check). This guarantees byte-identical rows/labels to
#4's GBDT dataset; this script adds NOTHING to that function. omega4._prepare_frames is called a
second time (redundant, ~seconds, not the expensive candidate-simulation loop) only to recover
frames["train_df"] itself (the GBDT script does not return it), needed here to build the
bar-level market-feature matrix the windows are sliced from.

=== Windowing -- new logic, this script only ===
Each dataset row's absolute bar index (`row_i`) is recovered as
`exit_path_entry_i + exit_path_hold_bars` (both already columns of `frame_exit`, since the dataset
builder iterates `row_i` in `range(entry_i, barrier_end_i + 1)` over the exact same `frame` this
script also loads) and cross-checked against `frame_exit["timestamp"]`. A bar-level market-feature
matrix (`market_np`, base_cols only, 102-dim, one row per bar of frames["train_df"]) is built via
parent._base_input (the SAME function prep_component/greedy.prepare_component use at replay time),
standardized (mean/std fit over the full TRAIN split), and each dataset row's window is
`market_std[row_i-WINDOW+1 : row_i+1]` (left-zero-padded if that goes negative -- see
`_slice_window`, reused unchanged by the VAL script's windowed replay for train/inference
consistency). Position-state (pos_cols, 13-dim) is NOT part of the sequence (no historical
trajectory is available or meaningful for bars before the current one within a single window) --
it enters the model as a separate scalar branch, concatenated with the pooled TCN output before a
small MLP head (see TCNExitClassifier). This mirrors research_eth_omega461_gbdt_exit_head_val_
20260813.py's per-expert regime-routed structure (hard.EXPERT_NAMES = bull/bear/chop, one model
each) and the same balanced x soft-route-probability sample weighting as
train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622._fit_exit_head_only / #4's GBDT.

=== Architecture -- reused from Phase1's TCN research, adapted only where structurally required ===
CausalConv1d / TCNBlock copied unchanged (same dilated-causal-conv design) from
scripts/verify_eth_h48qual_tcn_sequence_model_20260812.py and scripts/tune_eth_h48qual_tcn_sequence_
model_hpsearch_20260812.py. Default hyperparameters are that tuning run's best-VAL-margin winner for
the closest-matching feature theme (`raw_lite`, tmp/eth_h48qual_tcn_hpsearch_multivariant_20260812/
winner_raw_lite.json): window=48, hidden=32, n_blocks=5, kernel_size=5, dropout=0.2998,
lr=0.002484, weight_decay=0.000890, batch_size=1024, optimizer=Adam (matching what those HP values
were tuned against). Phase1's TCNClassifier was a 3-class direction head with no position-state
input; TCNExitClassifier here is a NEW small extension (pos MLP branch + concat + head, not present
in Phase1) needed because exit_head is inherently a post-entry problem the original architecture
never had to solve -- reasonable but not separately hyperparameter-tuned (out of scope; documented,
not hidden).

=== CPU-only training-time budget -- explicit, not silently reduced ===
This dev box has no CUDA device (torch.cuda.is_available()==False, 12 CPU cores). A synthetic timing
probe (batch=1024, window=48, this exact architecture) measured ~4.3K rows/sec forward+backward on
CPU; window-slicing overhead measured separately at ~207K rows/sec (~2% of total). Full TRAIN split
is ~1.05M rows (85% of #4's 1,234,431-row dataset) -- one full epoch over ALL of it would take
~4 minutes of pure model compute alone. Per Phase1's own precedent (MAX_TRAIN_WINDOWS_PER_EPOCH, see
verify_eth_h48qual_tcn_sequence_model_20260812.py/tune_eth_h48qual_tcn_sequence_model_hpsearch_
20260812.py, both of which ALSO capped windows/epoch even on a GPU), each epoch here draws a random
subsample of --max-train-windows-per-epoch (default 80,000, no replacement) from the TRAIN split
rather than iterating the full 1.05M rows every epoch; sample weighting (not the subsample draw) is
what encodes "balanced x route probability" importance, matching TabM/GBDT's loss-weighting
convention exactly. Held-out early-stopping loss is checked against a FIXED (drawn once per expert)
random subsample of the held-out 15% split (default cap 30,000 rows) rather than the whole thing,
for the same compute-budget reason -- this does not touch the actual comparison (VAL-period trading
replay in the companion VAL script uses the full VAL frame, not this internal dev-loss subsample).
--max-candidates defaults to 1500 (the FULL #4/TabM-parity candidate count) because the dataset
BUILD itself (candidate-barrier simulation, shared with #4) is not the bottleneck (~500-660s, a
fixed one-time cost per #4's own report) -- only per-epoch TRAINING throughput needed budgeting.

fresh_forward_bar_by_bar=true (dataset build is the same causal forward barrier simulation #4 used,
unmodified, called via import). trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false. Training uses only the
pre-2025-10-01 TRAIN split (identical frames to the TabM/GBDT runs). Does NOT touch
trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env. Does NOT touch
zig075 in any way.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import research_eth_omega461_exit_head_h48cons_relabel_20260813 as h48cons  # noqa: E402
import research_eth_omega461_exit_head_liveatr_relabel_20260813 as liveatr  # noqa: E402
import train_eval_omega461_gbdt_exit_head_liveatr_20260813 as gbdt_train  # noqa: E402

MODEL_ID = "eth_omega461_tcn_exit_head_liveatr_20260813"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

# Tuned HP source: tmp/eth_h48qual_tcn_hpsearch_multivariant_20260812/winner_raw_lite.json
# (docs/experiments/eth_h48qual_tcn_hpsearch_multivariant_20260812.md), the closest-matching
# feature theme to this script's 102 base_cols market-feature sequence.
WINDOW_DEFAULT = 48
HIDDEN_DEFAULT = 32
N_BLOCKS_DEFAULT = 5
KERNEL_SIZE_DEFAULT = 5
DROPOUT_DEFAULT = 0.299802956322068
LR_DEFAULT = 0.002484431047341557
WEIGHT_DECAY_DEFAULT = 0.0008896182007864648
BATCH_SIZE_DEFAULT = 1024
# New (not part of the Phase1-tuned search -- that architecture had no position-state branch).
POS_HIDDEN_DEFAULT = 16
HEAD_HIDDEN_DEFAULT = 32

MAX_EPOCHS_DEFAULT = 25
PATIENCE_DEFAULT = 6
MAX_TRAIN_WINDOWS_PER_EPOCH_DEFAULT = 80_000
EVAL_SUBSAMPLE_CAP_DEFAULT = 30_000
GRAD_CLIP_NORM = 2.0


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


# ---------------------------------------------------------------------------
# Architecture -- CausalConv1d/TCNBlock copied unchanged from
# verify_eth_h48qual_tcn_sequence_model_20260812.py / tune_eth_h48qual_tcn_sequence_model_
# hpsearch_20260812.py (kernel_size/dropout parameterized per the hpsearch version).
# TCNExitClassifier is NEW: pos-state scalar branch (Phase1's direction TCN had none) concatenated
# with the pooled sequence branch before a small MLP head, per the coordinator's explicit design
# instruction (position-state is a scalar, not a sequence).
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=self.pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out[:, :, : -self.pad] if self.pad > 0 else out


class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        return self.norm(out + x)


class TCNExitClassifier(nn.Module):
    """seq: (B, in_ch, T) causal window of market features (T==window, left-zero-padded upstream).
    pos: (B, pos_dim) current-bar position state (scalar, not a sequence -- see module docstring).
    Output: (B, 2) raw logits [hold, exit], matching TabM/GBDT's 2-class exit_head convention."""

    def __init__(
        self, in_ch: int, pos_dim: int, *, hidden: int, n_blocks: int, kernel_size: int, dropout: float,
        pos_hidden: int, head_hidden: int,
    ) -> None:
        super().__init__()
        dilations = [2 ** i for i in range(int(n_blocks))]
        self.input_proj = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList([TCNBlock(hidden, d, kernel_size, dropout) for d in dilations])
        self.pos_proj = nn.Sequential(nn.Linear(pos_dim, pos_hidden), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden + pos_hidden, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 2)
        )

    def forward(self, seq: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(seq)
        for block in self.blocks:
            h = block(h)
        pooled = h[:, :, -1]  # causal: last timestep = current bar, matches Phase1's TCNClassifier
        p = self.pos_proj(pos)
        return self.head(torch.cat([pooled, p], dim=-1))


def _slice_window(arr: np.ndarray, row_i: int, window: int) -> np.ndarray:
    """Left-zero-pads when fewer than `window` bars of history exist before row_i (inclusive).
    Shared verbatim by the VAL script's windowed replay (imported, not reimplemented) so train-time
    and replay-time windowing are byte-identical. Phase1's training scripts instead SKIPPED any
    index without a full window (idx >= WINDOW - 1 filter) -- not reusable here because a replay
    loop must emit a hold/exit decision at every bar a position is open, it cannot skip a bar.
    Padding with zeros (the harness's pre-existing "no information" convention, matching how
    _base_input already zero-fills pos_cols for bars where no trade is open) is the causal,
    consistent choice: the model must learn to cope with partial context near a series' start,
    exactly as it will occasionally see in live replay too."""
    lo = int(row_i) - int(window) + 1
    src_lo = max(lo, 0)
    window_arr = arr[src_lo : int(row_i) + 1]
    if lo < 0:
        pad = np.zeros((-lo, arr.shape[1]), dtype=arr.dtype)
        window_arr = np.concatenate([pad, window_arr], axis=0)
    return window_arr


def _build_batch_windows(market_std: np.ndarray, row_idx: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros((len(row_idx), window, market_std.shape[1]), dtype=np.float32)
    for b, ri in enumerate(row_idx):
        w = _slice_window(market_std, int(ri), window)
        out[b, window - len(w) :, :] = w
    return out


def _fit_scaler(arr: np.ndarray) -> dict[str, np.ndarray]:
    mean = arr.mean(axis=0).astype(np.float32)
    std = arr.std(axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    return {"mean": mean, "std": std}


def _apply_scaler(arr: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    return np.clip((arr - scaler["mean"]) / scaler["std"], -10.0, 10.0).astype(np.float32)


def _weighted_ce_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    per_sample = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    return (per_sample * weight).sum() / torch.clamp(weight.sum(), min=1.0)


def _fit_expert(
    market_std: np.ndarray,
    pos_std: np.ndarray,
    row_idx: np.ndarray,
    y: np.ndarray,
    route_w: np.ndarray,
    *,
    seed: int,
    window: int,
    hp: dict[str, Any],
    max_epochs: int,
    patience: int,
    max_train_windows_per_epoch: int,
    eval_subsample_cap: int,
) -> tuple[TCNExitClassifier, dict[str, Any]]:
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))

    weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float32) * route_w.astype(np.float32)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise RuntimeError("invalid TCN sample weights")

    n = len(y)
    split = max(int(n * 0.85), min(n - 1, 256))
    train_idx = np.arange(split)
    val_idx_full = np.arange(split, n)
    val_idx = (
        rng.choice(val_idx_full, size=int(eval_subsample_cap), replace=False)
        if len(val_idx_full) > int(eval_subsample_cap)
        else val_idx_full
    )

    model = TCNExitClassifier(
        in_ch=market_std.shape[1], pos_dim=pos_std.shape[1], hidden=hp["hidden"], n_blocks=hp["n_blocks"],
        kernel_size=hp["kernel_size"], dropout=hp["dropout"], pos_hidden=hp["pos_hidden"], head_hidden=hp["head_hidden"],
    )
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    val_seq = torch.from_numpy(_build_batch_windows(market_std, row_idx[val_idx], window))
    val_pos = torch.from_numpy(pos_std[val_idx])
    val_y = torch.from_numpy(y[val_idx].astype(np.int64))
    val_w = torch.from_numpy(weights[val_idx])

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    t0 = time.time()
    for epoch in range(int(max_epochs)):
        last_epoch = epoch + 1
        model.train()
        epoch_idx = rng.choice(train_idx, size=min(int(max_train_windows_per_epoch), len(train_idx)), replace=False)
        for start in range(0, len(epoch_idx), int(hp["batch_size"])):
            batch_idx = epoch_idx[start : start + int(hp["batch_size"])]
            seq_np = _build_batch_windows(market_std, row_idx[batch_idx], window).transpose(0, 2, 1)  # (B, C, T)
            seq = torch.from_numpy(np.ascontiguousarray(seq_np))
            pos = torch.from_numpy(pos_std[batch_idx])
            yb = torch.from_numpy(y[batch_idx].astype(np.int64))
            wb = torch.from_numpy(weights[batch_idx])
            opt.zero_grad(set_to_none=True)
            logits = model(seq, pos)
            loss = _weighted_ce_loss(logits, yb, wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_seq.transpose(1, 2), val_pos)
            val_loss = float(_weighted_ce_loss(val_logits, val_y, val_w).detach().cpu())
        if val_loss + 1.0e-6 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(
            f"    epoch={last_epoch} train_windows={len(epoch_idx)} val_loss={val_loss:.4f} "
            f"best={best_loss:.4f} stale={stale} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
        if stale >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = torch.softmax(model(val_seq.transpose(1, 2), val_pos), dim=-1)[:, 1].detach().cpu().numpy()
    diag = {
        "train_rows": int(len(train_idx)),
        "val_rows_full": int(len(val_idx_full)),
        "val_rows_eval_subsample": int(len(val_idx)),
        "val_auc": float(roc_auc_score(y[val_idx], val_probs)) if len(np.unique(y[val_idx])) > 1 else None,
        "val_logloss": float(log_loss(y[val_idx], val_probs, labels=[0, 1])),
        "val_positive_rate": float(np.mean(y[val_idx])),
        "train_positive_rate": float(np.mean(y[train_idx])),
        "route_weight_sum": float(route_w.sum()),
        "epochs_ran": int(last_epoch),
        "best_val_loss": float(best_loss),
        "fit_elapsed_sec": time.time() - t0,
    }
    return model, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-candidates", type=int, default=gbdt_train.MAX_CANDIDATES_DEFAULT)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--skip-reference-check", action="store_true")
    ap.add_argument("--window", type=int, default=WINDOW_DEFAULT)
    ap.add_argument("--hidden", type=int, default=HIDDEN_DEFAULT)
    ap.add_argument("--n-blocks", type=int, default=N_BLOCKS_DEFAULT)
    ap.add_argument("--kernel-size", type=int, default=KERNEL_SIZE_DEFAULT)
    ap.add_argument("--dropout", type=float, default=DROPOUT_DEFAULT)
    ap.add_argument("--lr", type=float, default=LR_DEFAULT)
    ap.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--pos-hidden", type=int, default=POS_HIDDEN_DEFAULT)
    ap.add_argument("--head-hidden", type=int, default=HEAD_HIDDEN_DEFAULT)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS_DEFAULT)
    ap.add_argument("--patience", type=int, default=PATIENCE_DEFAULT)
    ap.add_argument("--max-train-windows-per-epoch", type=int, default=MAX_TRAIN_WINDOWS_PER_EPOCH_DEFAULT)
    ap.add_argument("--eval-subsample-cap", type=int, default=EVAL_SUBSAMPLE_CAP_DEFAULT)
    ap.add_argument("--seed", type=int, default=gbdt_train.SEED)
    args = ap.parse_args()

    print(f"stage=start window={args.window} hidden={args.hidden} n_blocks={args.n_blocks}", flush=True)
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    (out_dir / "h48qual").mkdir(parents=True, exist_ok=True)

    print("stage=build_dataset (import train_eval_omega461_gbdt_exit_head_liveatr_20260813._build_dataset, unchanged)", flush=True)
    x_exit_raw, y_exit, frame_exit, exit_diag = gbdt_train._build_dataset(int(args.max_candidates))

    reference_check: dict[str, Any] | None = None
    if int(args.max_candidates) == gbdt_train.MAX_CANDIDATES_DEFAULT and gbdt_train.REFERENCE_REPORT.exists() and not args.skip_reference_check:
        ref = json.loads(gbdt_train.REFERENCE_REPORT.read_text(encoding="utf-8"))["dataset"]
        reference_check = {
            "rows_match": int(exit_diag["rows"]) == int(ref["rows"]),
            "positive_count_match": int(exit_diag["positive_count"]) == int(ref["positive_count"]),
            "used_candidates_match": int(exit_diag["used_candidates"]) == int(ref["used_candidates"]),
            "rebuilt_rows": int(exit_diag["rows"]), "reference_rows": int(ref["rows"]),
        }
        print(f"stage=dataset_reference_check {reference_check}", flush=True)
        if not (reference_check["rows_match"] and reference_check["positive_count_match"]):
            print("WARNING: rebuilt dataset does NOT match the original full1500 TabM/GBDT run's report.json.", flush=True)

    print("stage=prepare_frames_for_market_matrix (redundant 2nd call, needed only for frames['train_df'])", flush=True)
    t0 = time.time()
    frames = omega4._prepare_frames(
        disable_tp_sl=False, direction_label_dir=liveatr.DIRECTION_LABEL_DIR,
        quality_mode="same_as_direction", quality_label_dir=None,
        quality_min_edge=0.0, quality_max_mae=0.0, quality_min_mfe_mae=0.0, quality_max_hold_bars=0,
    )
    print(f"  train_df rows={len(frames['train_df'])} elapsed={time.time() - t0:.1f}s", flush=True)

    baseline_bundle_path = h48cons.sweep.COMPONENTS["h48qual"]["bundle"]
    base_cols = list(torch.load(baseline_bundle_path, map_location="cpu", weights_only=False)["base_cols"])

    print("stage=recover_row_i_and_verify", flush=True)
    row_i = (
        pd.to_numeric(frame_exit["exit_path_entry_i"], errors="raise").to_numpy(dtype=np.int64)
        + pd.to_numeric(frame_exit["exit_path_hold_bars"], errors="raise").to_numpy(dtype=np.int64)
    )
    train_ts = pd.to_datetime(frames["train_df"]["timestamp"]).to_numpy()
    exit_ts = pd.to_datetime(frame_exit["timestamp"]).to_numpy()
    ts_match = bool(np.array_equal(train_ts[row_i], exit_ts))
    print(f"  row_i recovered from exit_path_entry_i + exit_path_hold_bars; timestamp cross-check match={ts_match}", flush=True)
    if not ts_match:
        raise RuntimeError("row_i recovery does not reproduce frame_exit timestamps -- aborting before training")

    market_np_full = parent._base_input(frames["train_df"], base_cols)[base_cols].to_numpy(dtype=np.float32)
    # Spot-check: market_np_full[row_i] must equal the already-computed cur_<col> values GBDT/TabM
    # trained on -- proves the windowed dataset is "same data, more history", not a different source.
    rng_check = np.random.default_rng(0)
    sample_rows = rng_check.choice(len(row_i), size=min(500, len(row_i)), replace=False)
    sample_cols = rng_check.choice(len(base_cols), size=min(20, len(base_cols)), replace=False)
    mismatches = 0
    for r in sample_rows:
        for c in sample_cols:
            cur_col = f"cur_{base_cols[c]}"
            if cur_col not in x_exit_raw.columns:
                continue
            expected = float(pd.to_numeric(x_exit_raw[cur_col].iloc[int(r)], errors="coerce"))
            actual = float(market_np_full[int(row_i[r]), int(c)])
            if not np.isclose(expected, actual, atol=1e-4, rtol=1e-4, equal_nan=True):
                mismatches += 1
    print(f"  cur_<col> spot-check: {len(sample_rows) * len(sample_cols)} cells compared, mismatches={mismatches}", flush=True)
    if mismatches > 0:
        raise RuntimeError(f"market_np/row_i does not reproduce x_exit_raw cur_ values ({mismatches} mismatches) -- aborting")

    market_scaler = _fit_scaler(market_np_full)
    market_std_full = _apply_scaler(market_np_full, market_scaler)

    pos_np = x_exit_raw[parent.POS_COLS].to_numpy(dtype=np.float32)
    pos_scaler = _fit_scaler(pos_np)
    pos_std = _apply_scaler(pos_np, pos_scaler)

    y = np.asarray(y_exit, dtype=np.int64)
    route_probs = parent._route_probs(frame_exit)  # (n, 3), bull/bear/chop per hard.ROUTE_COLS

    hp = {
        "hidden": int(args.hidden), "n_blocks": int(args.n_blocks), "kernel_size": int(args.kernel_size),
        "dropout": float(args.dropout), "lr": float(args.lr), "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size), "pos_hidden": int(args.pos_hidden), "head_hidden": int(args.head_hidden),
    }

    models: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        print(f"stage=fit_expert expert={expert}", flush=True)
        model, diag = _fit_expert(
            market_std_full, pos_std, row_i, y, route_probs[:, idx],
            seed=int(args.seed) + idx, window=int(args.window), hp=hp,
            max_epochs=int(args.max_epochs), patience=int(args.patience),
            max_train_windows_per_epoch=int(args.max_train_windows_per_epoch),
            eval_subsample_cap=int(args.eval_subsample_cap),
        )
        models[expert] = {k: v.cpu() for k, v in model.state_dict().items()}
        diagnostics[expert] = diag
        print(f"  {expert}: {diag}", flush=True)

    bundle = {
        "model_id": MODEL_ID,
        "framework": "torch_tcn",
        "base_cols": base_cols,
        "pos_cols": list(parent.POS_COLS),
        "window": int(args.window),
        "arch": {k: hp[k] for k in ("hidden", "n_blocks", "kernel_size", "dropout", "pos_hidden", "head_hidden")},
        "market_scaler": market_scaler,
        "pos_scaler": pos_scaler,
        "models": models,
    }
    bundle_path = out_dir / "h48qual" / "tcn_exit_bundle.pt"
    torch.save(bundle, bundle_path)
    print(f"bundle={bundle_path}", flush=True)

    report = {
        "model_id": MODEL_ID,
        "design": (
            "Same live-ATR-barrier candidate/label recipe and dataset as the TabM/GBDT baselines "
            "(imported unchanged from train_eval_omega461_gbdt_exit_head_liveatr_20260813._build_"
            "dataset, seed=260813, max_candidates=1500). Adds a WINDOW of the 102 base_cols market "
            "features ending at each row's recovered absolute bar index (row_i = exit_path_entry_i "
            "+ exit_path_hold_bars, cross-checked against frame_exit timestamps and cur_ values). "
            "Position-state (13-dim) enters as a separate scalar branch, concatenated with the "
            "pooled causal-TCN output before a small MLP head. Same per-expert soft route-"
            "probability x balanced sample weighting as _fit_exit_head_only / GBDT's _fit_expert."
        ),
        "hyperparameters": {**hp, "window": int(args.window)},
        "training_budget": {
            "max_epochs": int(args.max_epochs), "patience": int(args.patience),
            "max_train_windows_per_epoch": int(args.max_train_windows_per_epoch),
            "eval_subsample_cap": int(args.eval_subsample_cap),
            "note": (
                "CPU-only dev box, no CUDA. Per-epoch training subsample and held-out eval subsample "
                "are compute-budget devices (Phase1's own TCN scripts also capped windows/epoch, even "
                "on GPU) -- the FULL 1500-candidate/1,234,431-row dataset (100% TabM/GBDT parity) is "
                "used; only per-epoch throughput is bounded, not dataset scale. See module docstring."
            ),
        },
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "row_i_timestamp_crosscheck_match": ts_match,
        "cur_value_spotcheck_cells_compared": int(len(sample_rows) * len(sample_cols)),
        "cur_value_spotcheck_mismatches": int(mismatches),
        "dataset": exit_diag,
        "dataset_reference_check": reference_check,
        "expert_diagnostics": diagnostics,
        "bundle_path": str(bundle_path),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"report={out_dir / 'report.json'}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
