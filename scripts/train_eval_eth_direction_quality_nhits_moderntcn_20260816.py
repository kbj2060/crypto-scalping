"""ETH direction+quality (h48qual label contract) -- ModernTCN vs N-HiTS backbone-swap candidate.

Contract: docs/model_contracts/eth_candidate_nhits_moderntcn_direction_quality_contract_20260816.md
(read that first -- registry-overlap reasoning, full scope, and the disclosed faithfulness
compromises live there, this docstring only summarizes the parts needed to read the code).

Motivation (user instruction, 2026-08-16): an external 918-experiment benchmark
(arXiv:2603.16886) found ModernTCN best overall point-forecast RMSE and N-HiTS specifically best
on ETH/USDT among 9 architectures -- "don't reject these without trying." Both architectures are
genuinely new to this project (grep-confirmed 0 prior mentions for ModernTCN; N-HiTS has only a
dead, never-evaluated-for-direction/quality checkpoint from an abandoned 2026-04 NeuralForecast
ensemble pack, docs/data_ensemble_cleanup_candidates.md). Neither has a real prior attempt on THIS
project's classification task, which is why this retest does not re-litigate the closed
`eth_odyssey_dl_rl_architecture_axis_closed_20260816` line (see contract doc's registry-overlap
section for the full argument, including why ModernTCN's large-kernel depthwise conv + structural
reparam + multi-stage downsampling is architecturally distinct from the plain dilated-causal TCN
already falsified 0/75 OOS in that line).

Data/label/window convention: reused byte-for-byte in spirit from the two established sequence-
architecture reference scripts (read in full before writing this file, per instruction):
  - scripts/verify_eth_h48qual_tcn_sequence_model_20260812.py
  - scripts/tune_eth_h48qual_tcn_sequence_model_hpsearch_20260812.py
Panel: data/splits/year_oos/eth_features_2024_2026_analysis.csv. Feature source: the reference
scripts' "raw_lite" SEQ_COLS (8 lightly-processed causal columns) -- this is what the instruction
means by "reuse the TCN sequence script's exact feature source"; the HP-search script's other 4
feature-set variants were that script's OWN additional exploration, not the base contract, so they
are not reused here. WINDOW=96 (8h) kept as the reference scripts' default, also swept in the HP
stage in {48,96,192} exactly like the reference HP script, so the choice isn't just asserted.
Splits: TRAIN 2024-06-01..2025-09-30, VAL 2025-10-01..2025-12-31, OOS 2026-01-01..2026-02-28 --
this is the reference scripts' convention and deliberately NOT the CLAUDE.md canonical Fresh-
Forward window (VAL 2025-09-01..2025-12-31 / OOS 2026-01-01..2026-03-31): the h48_conservative
quality label (see below) is only built through 2026-02-28
(tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619/oos_triple_barrier_labels.csv
does not extend further), which is exactly why the reference scripts already ended OOS there.
Declared per the repo's boundary-change disclosure rule.

Label contract -- matches h48qual (NOT zig075's same_as_direction), per instruction, since quality
is meant to be a genuinely different signal from direction here, not a copy of it:
  - direction: zigzag_action, 3-class (CASH/LONG/SHORT), loaded exactly like both reference
    scripts (tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/
    label_contracts/zigzag_action_labels_20260531/zigzag_action_labels_{2024,2025,2026}.csv).
  - quality: h48_conservative (48-bar horizon, tp_mult=1.2/sl_mult=0.8/min_tp=0.006/min_sl=0.004
    ATR-relative triple barrier), the actual live-deployed h48qual recipe (not the session-local
    384bar redesign). Loaded from
    tmp/eth_h48_conservative_orig_padded_to_zigzag_timestamps_20260811/zigzag_action_labels_{2025,2026}.csv
    -- built by scripts/pad_eth_h48_conservative_orig_labels_to_zigzag_timestamps_20260811.py from
    the original tb_action_h48_conservative barrier column, padded onto zigzag_action's timestamp
    grid (missing filled as CASH=0). Column is literally named "zigzag_action" in those files (an
    artifact of the padding script's own naming, not a bug introduced here) -- renamed to
    `h48_conservative` on load in this script to avoid confusion with the direction label.

Scope decision, DISCLOSED not silent: only direction_head + quality_head are trained here (the
loss is `loss_dir + 0.80*loss_qual`, dropping the `+1.15*loss_exit` term of the live 3-head
design). exit_head fundamentally needs the episodic, position-state-conditioned dataset built by
`train_eval_omega1_2_tabm_exit_head_20260603.py`'s `_build_exit_dataset_independent`, which is a
different kind of learning problem (position lifecycle, not a bar-level snapshot/window) orthogonal
to the backbone-architecture question this experiment targets. This mirrors the established
convention: the TCN reference line itself split direction (this file's precedent,
`verify_eth_h48qual_tcn_sequence_model_20260812.py`) and exit_head
(`research_eth_omega461_tcn_exit_head_val_20260813.py`) into two SEPARATE scripts rather than one
fused design. Neither ModernTCN nor N-HiTS's source literature is a position-state exit predictor
either. PnL simulation below trades on direction_head's argmax only (same methodology as both
reference scripts); quality_head is evaluated on its own classification metrics (balanced_accuracy/
macro_f1 against h48_conservative) as a secondary, diagnostic-only result, not used to gate the
backtest -- replicating the live system's exact `quality_for_action` gate-derivation+threshold-
calibration machinery is out of scope for a backbone-swap experiment.

Comparison baseline for "(a) the live TabM baseline's OOS direction_balanced_accuracy" (per
instruction): reproducing the literal live regime-routed h48qual bundle (HMM router + 3 bull/bear/
chop expert sub-models + FINAL12 feature engineering) is a separate, heavy pipeline orthogonal to
this backbone question. Instead this script trains a same-conditions "TabM control" -- the
UNMODIFIED canonical `ThreeHeadTabM` (scripts/train_eval_omega1_2_tabm_3head_20260603.py) on the
IDENTICAL SEQ_COLS/window-last-bar/labels/split/N-seed protocol as ModernTCN/N-HiTS (window
collapsed to its last timestep, since TabM is a single-bar-snapshot architecture) -- giving a
genuine, self-computed, apples-to-apples backbone ablation. This is disclosed as a narrower
condition than the live 115-feature regime-routed production model, not a stand-in for it.

Modern DL training checklist (mid-task addition, `feedback_modern_dl_training_checklist` memory),
baked into every stage below, not bolted on after a plain-CE baseline:
  - purge/embargo gap: EMBARGO_BARS (=max(WINDOW,96)) dropped from both sides of the internal
    train/early-stop-val split boundary (see `_valid_indices`/`_split_with_embargo`).
  - EMA of model weights: a real Polyak/exponential shadow copy (decay=0.999, see `EMAWeights`),
    distinct from ELR's per-sample soft-target EMA below -- both implemented, kept separate.
  - sized LR warmup: linear ramp over the first 10% of total optimizer steps (see `_warmup_lr_lambda`).
  - label smoothing: eps=0.05, folded into both the plain-CE and GCE loss paths via a shared
    smoothed-target helper (`_smoothed_target`) so it applies uniformly regardless of which
    noise-robust-loss variant is active.
  - noise-robust loss options (GCE q=0.7 / ELR lambda=3.0 beta=0.7, Zhang&Sabuncu arXiv:1805.07836
    / Liu et al. arXiv:2007.00151) and latent-space mixup (alpha=1.0, Zhang et al. arXiv:1710.09412):
    exact hyperparameter values and mechanism ported from this repo's own prior TabM regularizer
    research (`scripts/research_eth_candidate_faithful_tabm_batchensemble_combo_regularizer_20260816.py`,
    `..._regularizer_isolation_20260816.py`) -- that research found GCE-alone beat plain CE
    slightly, ELR-alone and mixup-alone were BOTH WORSE than baseline individually, and naive
    all-3-combined was worse than any single technique, for TabM on the direction/zigzag_action
    label. There is no guarantee the same winner (GCE-alone) holds for a different backbone or for
    quality/h48_conservative, so `--stage isolation` reruns the identical isolation design
    (none/gce_only/elr_only/mixup_only, N>=5 seeds) independently per architecture here rather than
    assuming the TabM result transfers.

Staged pipeline (`--stage`), per the mid-task sequencing instruction -- architecture first, THEN
which regularizer helps, THEN hyperparameters on top of the winning regularizer, THEN the final
seed-diversity run, so the HP sweep and the regularizer choice are never varied at the same time:
  1. sanity     -- tiny local CPU smoke run, both architectures, 1 seed, few epochs/rows. Not a
                   result -- only checks the pipeline trains without shape/NaN/crash bugs.
  2. isolation  -- per architecture: {none, gce_only, elr_only, mixup_only} x N=5 genuinely random
                   seeds, fixed fixed_epochs (no early stopping, comparable curves), literature-
                   default architecture capacity. Picks the winning regularizer config by peak
                   embargoed-held-out direction balanced_accuracy.
  3. hpsearch   -- Optuna TPE, N_TRIALS per architecture, architecture-capacity + training HPs,
                   regularizer FIXED to stage 2's winner. Objective = embargoed held-out combined
                   loss (single seed, cheap screening, mirrors the TCN reference HP script). Top-K
                   candidates re-scored on VAL trade simulation (direction argmax) vs always_short;
                   adopt by margin (select-on-validation-only).
  4. final      -- N>=5 genuinely random seeds (random.SystemRandom, not fixed-increment, per the
                   Seed-Diversity Ensemble Promotion Gate / `_is_clustered_seed_list`) at stage 3's
                   best HP + stage 2's best regularizer. Reports direction+quality classification
                   metrics and PnL (cost1/2/3, VAL+OOS) vs always_long/always_short, plus the TabM
                   control (N=5 seeds, base hygiene only, no GCE/ELR/mixup search -- that axis is
                   already covered for TabM by the two research scripts cited above).
  5. all        -- runs 2->3->4 in sequence for both architectures (stage 1 is local-only, not
                   part of the server job).

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false,
future_rows_used_for_entry=false (direction/quality/PnL are all recomputed bar-by-bar causal
walk-forward from the trained model's own predictions on VAL/OOS, matching both reference scripts'
`omega._metrics` trade-simulation harness; no stored ledger or future-row join is used anywhere).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "tmp/eth_candidate_nhits_moderntcn_direction_quality_20260816"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL_PATH = ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv"
DIRECTION_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
QUALITY_LABEL_DIR = ROOT / "tmp/eth_h48_conservative_orig_padded_to_zigzag_timestamps_20260811"

TRAIN_START, TRAIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-09-30 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:59:59")

SEQ_COLS = ["log_return", "volatility_z", "rsi", "macd_hist", "bb_width_z", "wick_ratio", "net_taker_ratio", "cvd_12"]
DEFAULT_WINDOW = 96
EMBARGO_MIN = 96

N_SEEDS_ISOLATION = 5
N_SEEDS_FINAL = 5
ISOLATION_EPOCHS = 12
N_TRIALS_HPSEARCH = 25
MAX_EPOCHS_TRIAL = 10
PATIENCE_TRIAL = 3
MAX_EPOCHS_FINAL = 30
PATIENCE_FINAL = 6
TOP_K_CANDIDATES = 5
MAX_WINDOWS_PER_EPOCH = 40000

LABEL_SMOOTHING_EPS = 0.05
EMA_DECAY = 0.999
WARMUP_FRACTION = 0.10
GCE_Q = 0.7
ELR_LAMBDA = 3.0
ELR_BETA = 0.7
MIXUP_ALPHA = 1.0
GCE_EPS = 1.0e-7
ELR_EPS = 1.0e-4

REGULARIZER_VARIANTS = [
    {"name": "none", "use_gce": False, "use_elr": False, "use_mixup": False},
    {"name": "gce_only", "use_gce": True, "use_elr": False, "use_mixup": False},
    {"name": "elr_only", "use_gce": False, "use_elr": True, "use_mixup": False},
    {"name": "mixup_only", "use_gce": False, "use_elr": False, "use_mixup": True},
]


def log(msg: str) -> None:
    print(f"[nhits_moderntcn] {msg}", flush=True)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# =====================================================================================================
# 1. Data loading -- panel + direction (zigzag_action) + quality (h48_conservative) labels
# =====================================================================================================

def load_panel_and_labels() -> dict[str, Any]:
    log("panel + zigzag_action(direction) + h48_conservative(quality) 라벨 로딩...")
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    assert (panel["timestamp"].diff().dropna() == pd.Timedelta("5min")).all(), "5분봉 연속성 깨짐"
    missing = [c for c in SEQ_COLS if c not in panel.columns]
    assert not missing, f"SEQ_COLS 누락: {missing}"

    dir_labels = pd.concat([
        pd.read_csv(DIRECTION_LABEL_DIR / f"zigzag_action_labels_{y}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
        for y in (2024, 2025, 2026)
    ], ignore_index=True).drop_duplicates("timestamp", keep="last")
    dir_map = dir_labels.set_index("timestamp")["zigzag_action"]

    qual_labels = pd.concat([
        pd.read_csv(QUALITY_LABEL_DIR / f"zigzag_action_labels_{y}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"]).rename(columns={"zigzag_action": "h48_conservative"})
        for y in (2025, 2026)
    ], ignore_index=True).drop_duplicates("timestamp", keep="last")
    qual_map = qual_labels.set_index("timestamp")["h48_conservative"]

    raw = panel[SEQ_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y_dir_full = dir_map.reindex(panel["timestamp"]).to_numpy()
    y_qual_full = qual_map.reindex(panel["timestamp"]).to_numpy()
    log(f"  패널 {len(panel)}행. direction 결측={np.isnan(y_dir_full).sum()} quality 결측={np.isnan(y_qual_full).sum()}")
    return {"panel": panel, "raw": raw, "y_dir_full": y_dir_full, "y_qual_full": y_qual_full}


def _valid_indices(mask: np.ndarray, window: int, y_dir_full: np.ndarray, y_qual_full: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(mask)
    idx = idx[idx >= window - 1]
    idx = idx[~pd.isna(y_dir_full[idx]) & ~pd.isna(y_qual_full[idx])]
    return idx


def _split_with_embargo(train_idx_all: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Purge/embargo gap around the internal fit/early-stop-val split boundary (modern DL training
    checklist item). Labels look forward up to ~48 bars (h48_conservative) / a pivot-dependent
    horizon (zigzag_action); windows look only backward (causal). EMBARGO_BARS is dropped from
    both sides of the boundary so no fit-side label horizon crosses into the held-out side and no
    held-out window's backward reach is drawn from rows the fit side's label already consumed."""
    embargo = max(int(window), EMBARGO_MIN)
    split_point = int(len(train_idx_all) * 0.85)
    fit_idx = train_idx_all[: max(split_point - embargo, 0)]
    es_idx = train_idx_all[split_point + embargo:]
    return fit_idx, es_idx


class WindowDataset(Dataset):
    """Returns (C, WINDOW) causal window ending at idx, direction label, quality label, row_id."""

    def __init__(self, raw_std: np.ndarray, window: int, indices: np.ndarray, y_dir_full: np.ndarray, y_qual_full: np.ndarray):
        self.raw_std = raw_std
        self.window = int(window)
        self.indices = indices
        self.y_dir_full = y_dir_full
        self.y_qual_full = y_qual_full

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        window = self.raw_std[idx - self.window + 1: idx + 1]  # (WINDOW, C)
        y_dir = int(self.y_dir_full[idx])
        y_qual = int(self.y_qual_full[idx])
        return torch.from_numpy(window.T.copy()), y_dir, y_qual, i  # (C, WINDOW)


def _standardize_fit(raw: np.ndarray, fit_idx: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    fit_rows = np.concatenate([np.arange(i - window + 1, i + 1) for i in fit_idx[:5000]])
    mean = raw[fit_rows].mean(axis=0)
    std = raw[fit_rows].std(axis=0)
    std[std < 1e-6] = 1.0
    raw_std = np.clip((raw - mean) / std, -10, 10).astype(np.float32)
    return raw_std, np.stack([mean, std])


# =====================================================================================================
# 2. Backbones -- unified interface: encode(x: (B,C,WINDOW)) -> h: (B, K, hidden_dim)
#    K=1 for ModernTCN/N-HiTS (plain sequence encoders); K=cfg.k for the TabM control (BatchEnsemble).
#    This lets one loss/training loop (built for TabM's K-member ensemble) serve all three backbones
#    unmodified -- nn.Linear broadcasts over the K dim regardless of its size.
# =====================================================================================================

class TabMControlBackbone(nn.Module):
    """UNMODIFIED canonical ThreeHeadTabM encoder (scripts/train_eval_omega1_2_tabm_3head_20260603.py
    ThreeHeadTabM.encode, copied verbatim) applied to the window's LAST timestep only (TabM is a
    single-bar-snapshot architecture, not a sequence model) -- the same-conditions control referenced
    in the module docstring's baseline-comparison section."""

    def __init__(self, n_features: int, *, k: int = 8, hidden: int = 192, layers: int = 3, dropout: float = 0.08):
        super().__init__()
        self.k = int(k)
        self.hidden_dim = int(hidden)
        self.input_scale = nn.Parameter(torch.randn(self.k, n_features) * 0.03 + 1.0)
        self.input_bias = nn.Parameter(torch.zeros(self.k, n_features))
        self.in_proj = nn.Linear(n_features, hidden)
        self.blocks = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(max(0, layers - 1)))
        self.expert_scale = nn.ParameterList(nn.Parameter(torch.randn(self.k, hidden) * 0.03 + 1.0) for _ in range(max(0, layers - 1)))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(max(0, layers)))
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_last = x[:, :, -1]  # (B, C) -- last timestep of the window
        xk = x_last.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.in_proj(xk)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            h = layer(h * self.expert_scale[idx].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            h = h + residual
        return h  # (B, k, hidden)


# --- ModernTCN (Luo & Wang, ICLR 2024 Spotlight, arXiv/OpenReview id vpJMJerXHU) -------------------
# Ported from the official repo's classification variant (github.com/luodhhh/ModernTCN,
# ModernTCN-classification/models/ModernTCN.py, fetched 2026-08-16), preserving: patch-embedding
# stem, multi-stage downsampling, ReparamLargeKernelConv (large+small depthwise branches, BN-fused,
# genuinely mergeable via structural reparameterization -- `merge_kernel()` below is a direct port
# of the official `fuse_bn`/`get_equivalent_kernel_bias`/`PaddingTwoEdge1d` logic, not a stub), and
# the two-stage ConvFFN (per-variable then cross-variable channel mixing) block design.
# Two disclosed deltas from the official classification source (both verified by reading the code,
# not guessed):
#   1. The official ModernTCN.__init__ accepts `revin`/`affine`/`subtract_last` and builds
#      `self.revin_layer`, but `forward_feature()` never actually calls it -- RevIN is dead code in
#      their classification task despite being listed as a component. This implementation actually
#      wires RevIN up (toggleable, HP-searched both ways) since the task literature names RevIN as
#      an "optional" ModernTCN component and our data benefits from testing it either way.
#   2. `stem_ratio` and `dw_dims` are accepted constructor arguments in the official code that are
#      never referenced anywhere in ModernTCN/Stage/Block's bodies (verified dead parameters) --
#      dropped here rather than faithfully threading through an unused value.
# 2026-08-18 fidelity audit (docs/experiments -- see the regime-hard-split ModernTCN line) found two
# further UNDISCLOSED deviations, now fixed in place (this file, not a copy -- both are genuine bugs,
# not deliberate choices, so there is nothing to disclose-and-keep):
#   3. ConvFFN dropout/activation order was Linear->GELU->Dropout->Linear->Dropout; the official is
#      Linear->Dropout->GELU->Linear->Dropout (dropout on the raw linear output, not the activated
#      one). Fixed in ModernTCNBlock.forward below.
#   4. The official classification() applies a dropout (`class_dropout`) to the pooled feature right
#      before the final Linear head; this port skipped straight from GELU+flatten to the head. Added
#      to TwoHeadClassifier as `class_drop` (default 0.1, matching the backbone's own dropout rate --
#      the official's exact default wasn't recoverable from the fetched source, so this is a matched
#      value, not a confirmed one).
class _ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, groups, padding=None):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


def _fuse_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> tuple[torch.Tensor, torch.Tensor]:
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1)
    return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std


class ReparamLargeKernelConv(nn.Module):
    """Large-kernel depthwise conv with a parallel small-kernel branch (both BN'd) during training;
    `merge_kernel()` fuses both branches (+ their BN stats) into ONE plain conv for inference --
    structural reparameterization, ported faithfully from the official repo (not a disclosed
    simplification)."""

    def __init__(self, channels: int, large_size: int, small_size: int, groups: int):
        super().__init__()
        self.large_size = large_size
        self.small_size = small_size
        self.merged: nn.Conv1d | None = None
        self.large = _ConvBN(channels, channels, large_size, groups)
        self.small = _ConvBN(channels, channels, small_size, groups, padding=small_size // 2) if small_size else None

    def forward(self, x):
        if self.merged is not None:
            return self.merged(x)
        out = self.large(x)
        if self.small is not None:
            out = out + self.small(x)
        return out

    @torch.no_grad()
    def merge_kernel(self) -> None:
        if self.merged is not None:
            return
        eq_k, eq_b = _fuse_bn(self.large.conv, self.large.bn)
        if self.small is not None:
            small_k, small_b = _fuse_bn(self.small.conv, self.small.bn)
            eq_b = eq_b + small_b
            pad = (self.large_size - self.small_size) // 2
            small_k_padded = torch.zeros_like(eq_k)
            small_k_padded[:, :, pad: pad + self.small_size] = small_k
            eq_k = eq_k + small_k_padded
        merged = nn.Conv1d(self.large.conv.in_channels, self.large.conv.out_channels, self.large_size,
                            stride=1, padding=self.large_size // 2, groups=self.large.conv.groups, bias=True)
        merged.weight.data.copy_(eq_k)
        merged.bias.data.copy_(eq_b)
        self.merged = merged
        del self.large
        self.small = None


class ModernTCNBlock(nn.Module):
    def __init__(self, nvars: int, dmodel: int, ffn_ratio: int, large_size: int, small_size: int, dropout: float):
        super().__init__()
        dff = dmodel * ffn_ratio
        self.dw = ReparamLargeKernelConv(nvars * dmodel, large_size, small_size, groups=nvars * dmodel)
        self.norm = nn.BatchNorm1d(dmodel)
        self.nvars, self.dmodel = nvars, dmodel
        self.ffn1pw1 = nn.Conv1d(nvars * dmodel, nvars * dff, 1, groups=nvars)
        self.ffn1pw2 = nn.Conv1d(nvars * dff, nvars * dmodel, 1, groups=nvars)
        self.ffn2pw1 = nn.Conv1d(nvars * dmodel, nvars * dff, 1, groups=dmodel)
        self.ffn2pw2 = nn.Conv1d(nvars * dff, nvars * dmodel, 1, groups=dmodel)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, M, D, N)
        inp = x
        B, M, D, N = x.shape
        x = self.dw(x.reshape(B, M * D, N))
        x = self.norm(x.reshape(B * M, D, N)).reshape(B, M, D, N).reshape(B, M * D, N)
        x = self.act(self.drop(self.ffn1pw1(x)))  # dropout before activation, matches official order (fixed 2026-08-18)
        x = self.drop(self.ffn1pw2(x)).reshape(B, M, D, N)
        x = x.permute(0, 2, 1, 3).reshape(B, D * M, N)
        x = self.act(self.drop(self.ffn2pw1(x)))  # dropout before activation, matches official order (fixed 2026-08-18)
        x = self.drop(self.ffn2pw2(x)).reshape(B, D, M, N).permute(0, 2, 1, 3)
        return inp + x


class RevIN(nn.Module):
    """Per-instance reversible normalization (ts-kim/RevIN, as used -- but never actually called --
    by the official ModernTCN classification code). Only the normalize direction is used here
    (classification has no forecast to denormalize back)."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):  # x: (B, C, T)
        mean = x.mean(dim=-1, keepdim=True).detach()
        std = (x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
        x = (x - mean) / std
        if self.affine:
            x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x


class ModernTCNBackbone(nn.Module):
    def __init__(self, n_vars: int, window: int, *, dims: list[int], num_blocks: list[int],
                 large_size: list[int], small_size: list[int], ffn_ratio: int, downsample_ratio: int,
                 patch_size: int, patch_stride: int, dropout: float, use_revin: bool):
        super().__init__()
        self.nvars = n_vars
        self.patch_size, self.patch_stride, self.downsample_ratio = patch_size, patch_stride, downsample_ratio
        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(n_vars)
        self.stem = nn.Sequential(nn.Conv1d(1, dims[0], patch_size, stride=patch_stride), nn.BatchNorm1d(dims[0]))
        self.downsamplers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.downsamplers.append(nn.Sequential(nn.BatchNorm1d(dims[i]), nn.Conv1d(dims[i], dims[i + 1], downsample_ratio, stride=downsample_ratio)))
        self.stages = nn.ModuleList([
            nn.ModuleList([ModernTCNBlock(n_vars, dims[s], ffn_ratio, large_size[s], small_size[s], dropout) for _ in range(num_blocks[s])])
            for s in range(len(dims))
        ])
        with torch.no_grad():
            dummy = torch.zeros(2, n_vars, window)
            self.hidden_dim = int(self._forward_feature(dummy).reshape(2, -1).shape[1])

    def _forward_feature(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, M, L)
        if self.use_revin:
            x = self.revin(x)
        B, M, L = x.shape
        x = x.unsqueeze(-2)  # (B, M, 1, L)
        for i, stage in enumerate(self.stages):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad_len)], dim=-1)
                x = self.stem(x)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
                x = self.downsamplers[i - 1](x)
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            for block in stage:
                x = block(x)
        return x  # (B, M, D_last, N_last)

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, WINDOW)
        feat = self._forward_feature(x)
        h = torch.nn.functional.gelu(feat).reshape(feat.shape[0], -1)
        return h.unsqueeze(1)  # (B, 1, hidden_dim) -- K=1

    def structural_reparam(self) -> None:
        for m in self.modules():
            if isinstance(m, ReparamLargeKernelConv):
                m.merge_kernel()


# --- N-HiTS (Challu et al., AAAI 2023, arXiv:2201.12886) --------------------------------------------
# Block/basis mechanics (pooling -> MLP -> theta -> linear-interpolation basis expansion ->
# doubly-residual backcast/forecast accumulation) ported from Nixtla's neuralforecast reference
# implementation (neuralforecast/models/nhits.py, NHITSBlock/_IdentityBasis, fetched 2026-08-16 --
# the same library this repo already trusts for its abandoned NHITS_0.ckpt pack). DISCLOSED
# classification adaptation (the paper's own output layer is a point-forecast regression head,
# ours is a shared representation for 2 classification heads):
#   - Original: univariate insample_y (B,L) is pooled/MLP'd/basis-expanded per block into a
#     (backcast: B,L) + (forecast: B,H) pair; `forecast = forecast + block_forecast` accumulates
#     the horizon-H point forecast across the stack (doubly residual stacking); `residuals =
#     (residuals - backcast) * mask` feeds the next stack the still-unexplained part.
#   - Here: there is no univariate target and no future horizon to forecast, so `insample_y` is
#     generalized to the full multivariate window (B,C,WINDOW), pooled per-channel then flattened
#     (exactly mirroring the original's `hist_exog` pooling/flatten path). Backcast is generalized
#     to size C*WINDOW (reshaped back to (B,C,WINDOW) for the residual subtraction) so doubly-
#     residual stacking still operates over the WHOLE window, not a scalar. The "forecast" half of
#     `_IdentityBasis` (linear interpolation from n_theta coefficients up to a target length) is
#     reused UNMODIFIED as a mechanism, just retargeted from "H future timesteps" to a fixed-size
#     `repr_dim` latent representation; `forecast = forecast + block_forecast` becomes `repr_acc =
#     repr_acc + block_repr`, preserving the exact accumulation semantics for a latent vector
#     instead of a price path. The original's "Level with Naive1" initialization (repeat the last
#     insample value H times) has no valid analogue in representation space and is dropped --
#     `repr_acc` starts at zero. This preserves N-HiTS's defining mechanism (hierarchical multi-
#     rate pooling + per-stack basis-expansion + doubly-residual accumulation) while replacing only
#     the forecast-specific output semantics, which is exactly what the task instruction asked to
#     be documented rather than silently decided.
class _IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, out_size: int, interpolation_mode: str = "linear"):
        super().__init__()
        self.backcast_size = backcast_size
        self.out_size = out_size
        self.interpolation_mode = interpolation_mode

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, : self.backcast_size]
        knots = theta[:, self.backcast_size :].unsqueeze(1)  # (B,1,n_knots)
        out = torch.nn.functional.interpolate(knots, size=self.out_size, mode=self.interpolation_mode,
                                               align_corners=False if self.interpolation_mode != "nearest" else None)
        return backcast, out.squeeze(1)  # (B, backcast_size), (B, out_size)


class NHiTSBlock(nn.Module):
    def __init__(self, n_vars: int, window: int, repr_dim: int, pool_kernel: int, freq_downsample: int,
                 mlp_hidden: int, dropout: float):
        super().__init__()
        self.n_vars, self.window = n_vars, window
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel, ceil_mode=True)
        pooled_len = int(np.ceil(window / pool_kernel))
        mlp_in = n_vars * pooled_len
        backcast_size = n_vars * window
        n_knots = max(repr_dim // freq_downsample, 1)
        n_theta = backcast_size + n_knots
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_theta),
        )
        self.basis = _IdentityBasis(backcast_size, repr_dim)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # residual: (B,C,WINDOW)
        B = residual.shape[0]
        pooled = self.pool(residual.reshape(B * self.n_vars, 1, self.window)).reshape(B, -1)
        theta = self.mlp(pooled)
        backcast_flat, repr_out = self.basis(theta)
        backcast = backcast_flat.reshape(B, self.n_vars, self.window)
        return backcast, repr_out


class NHiTSBackbone(nn.Module):
    def __init__(self, n_vars: int, window: int, *, repr_dim: int, n_blocks: list[int],
                 pool_kernel_sizes: list[int], freq_downsample: list[int], mlp_hidden: int, dropout: float):
        super().__init__()
        self.n_vars, self.window, self.repr_dim = n_vars, window, repr_dim
        self.hidden_dim = repr_dim
        blocks = []
        for stage_i, nb in enumerate(n_blocks):
            for _ in range(nb):
                blocks.append(NHiTSBlock(n_vars, window, repr_dim, pool_kernel_sizes[stage_i], freq_downsample[stage_i], mlp_hidden, dropout))
        self.blocks = nn.ModuleList(blocks)

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, WINDOW)
        residual = x
        repr_acc = x.new_zeros(x.shape[0], self.repr_dim)
        for block in self.blocks:
            backcast, block_repr = block(residual)
            residual = residual - backcast
            repr_acc = repr_acc + block_repr
        return repr_acc.unsqueeze(1)  # (B, 1, repr_dim) -- K=1


# =====================================================================================================
# 3. Two-head classifier wrapper (direction + quality only, see module docstring scope decision)
# =====================================================================================================

class TwoHeadClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int, class_drop: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.class_dropout = nn.Dropout(class_drop)  # matches official class_dropout, added 2026-08-18 (was missing)
        self.direction_head = nn.Linear(hidden_dim, 3)
        self.quality_head = nn.Linear(hidden_dim, 3)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.class_dropout(self.encode(x))
        return {"direction": self.direction_head(h), "quality": self.quality_head(h)}


class EMAWeights:
    """Real Polyak/exponential-moving-average shadow of the model's weights (checklist item,
    distinct from ELR's per-sample soft-target EMA below). Shadow is used for eval/final inference;
    the raw (non-EMA) weights keep training via the optimizer as normal."""

    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


def _warmup_lr_lambda(total_steps: int, warmup_fraction: float = WARMUP_FRACTION):
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def fn(step: int) -> float:
        if step < warmup_steps:
            return 0.1 + 0.9 * (step / warmup_steps)
        return 1.0

    return fn


# --- checklist-integrated loss helpers (label smoothing folded into both plain-CE and GCE paths) ---

def _smoothed_target(target: torch.Tensor, n_classes: int, eps: float = LABEL_SMOOTHING_EPS) -> torch.Tensor:
    y = torch.full((target.shape[0], n_classes), eps / n_classes, device=target.device, dtype=torch.float32)
    y.scatter_(1, target.view(-1, 1), 1.0 - eps + eps / n_classes)
    return y


def _cls_loss(logits_k: torch.Tensor, target: torch.Tensor, *, use_gce: bool, n_classes: int) -> torch.Tensor:
    """(B,K,C), (B,) long -> (B,K). Label smoothing always applied; GCE (Zhang&Sabuncu, q=0.7) or
    smoothed plain CE depending on `use_gce`."""
    target_soft = _smoothed_target(target, n_classes)
    if use_gce:
        probs_k = torch.softmax(logits_k, dim=-1)
        py = (probs_k * target_soft.unsqueeze(1)).sum(-1).clamp(min=GCE_EPS)
        return (1.0 - py.pow(GCE_Q)) / GCE_Q
    logp_k = torch.log_softmax(logits_k, dim=-1)
    return -(target_soft.unsqueeze(1) * logp_k).sum(-1)


def _elr_term(probs_mean: torch.Tensor, target_ema: torch.Tensor) -> torch.Tensor:
    dot = (probs_mean * target_ema).sum(dim=-1).clamp(max=1.0 - ELR_EPS)
    return -torch.log(1.0 - dot)


# =====================================================================================================
# 4. omega trade-simulation harness (verbatim pattern from both TCN reference scripts)
# =====================================================================================================

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

_FEE, _SLIP = omega._load_fee_slip()
COST_MULTS = {"cost1": 1.0, "cost2": 2.0, "cost3": 3.0}
omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0


def build_dec(action: np.ndarray) -> pd.DataFrame:
    action = action.astype(np.int64)
    active = action != omega.ACTION_CASH
    side = np.where(action == omega.ACTION_LONG, 1, np.where(action == omega.ACTION_SHORT, -1, 0)).astype(np.int64)
    return pd.DataFrame({
        "action": action, "side": side,
        "notional_exposure": np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0),
        "leverage": np.where(active, float(omega.BASE_TEMPLATE["leverage"]), 1.0),
        "position_fraction": np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0),
        "take_profit": np.where(active, float(omega.BASE_TEMPLATE["take_profit"]), 0.0),
        "stop_loss": np.where(active, float(omega.BASE_TEMPLATE["stop_loss"]), 0.0),
        "max_hold_bars": np.where(active, int(omega.BASE_TEMPLATE["max_hold"]), 0).astype(np.int64),
        "cooldown_bars": np.where(active, int(omega.BASE_TEMPLATE["cooldown"]), 0).astype(np.int64),
    })


def forced_side(dec: pd.DataFrame, side_value: int) -> pd.DataFrame:
    out = dec.copy()
    active = omega._active(dec)
    out.loc[active, "side"] = side_value
    out.loc[active, "action"] = omega.ACTION_LONG if side_value > 0 else omega.ACTION_SHORT
    return out


def pnl_vs_benchmarks(panel: pd.DataFrame, idx: np.ndarray, direction_pred: np.ndarray) -> dict[str, Any]:
    ohlc = panel.iloc[idx][["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
    dec = build_dec(direction_pred)
    out: dict[str, Any] = {}
    for cost_name, cost_mult in COST_MULTS.items():
        m_model = omega._metrics(ohlc, dec, fee=_FEE, slip=_SLIP, cost_mult=cost_mult)
        m_short = omega._metrics(ohlc, forced_side(dec, -1), fee=_FEE, slip=_SLIP, cost_mult=cost_mult)
        m_long = omega._metrics(ohlc, forced_side(dec, 1), fee=_FEE, slip=_SLIP, cost_mult=cost_mult)
        out[cost_name] = {
            "model_pnl": m_model["pnl"], "model_trades": m_model["trades"], "model_wr": m_model["wr"],
            "always_short_pnl": m_short["pnl"], "always_long_pnl": m_long["pnl"],
            "beats_always_short": m_model["pnl"] > m_short["pnl"], "beats_always_long": m_model["pnl"] > m_long["pnl"],
        }
    return out


# =====================================================================================================
# 5. Backbone factory + architecture-specific default/HP-searchable configs
# =====================================================================================================

def build_backbone(arch: str, n_vars: int, window: int, params: dict[str, Any]) -> nn.Module:
    if arch == "moderntcn":
        n_stage = int(params.get("n_stage", 2))
        dim0 = int(params.get("dim0", 32))
        dims = [dim0 * (2 ** i) for i in range(n_stage)]
        large = int(params.get("large_size", 13))
        return ModernTCNBackbone(
            n_vars, window,
            dims=dims, num_blocks=[int(params.get("num_blocks", 1))] * n_stage,
            large_size=[large] * n_stage, small_size=[5] * n_stage,
            ffn_ratio=int(params.get("ffn_ratio", 2)), downsample_ratio=int(params.get("downsample_ratio", 2)),
            patch_size=int(params.get("patch_size", 1)), patch_stride=int(params.get("patch_stride", 1)),
            dropout=float(params.get("dropout", 0.1)), use_revin=bool(params.get("use_revin", True)),
        )
    if arch == "nhits":
        preset = params.get("pool_preset", [2, 2, 1])
        freq = params.get("freq_preset", [4, 2, 1])
        return NHiTSBackbone(
            n_vars, window, repr_dim=int(params.get("repr_dim", 64)),
            n_blocks=[int(params.get("blocks_per_stack", 1))] * len(preset),
            pool_kernel_sizes=list(preset), freq_downsample=list(freq),
            mlp_hidden=int(params.get("mlp_hidden", 128)), dropout=float(params.get("dropout", 0.1)),
        )
    raise ValueError(f"unknown arch: {arch}")


ARCH_DEFAULT_PARAMS = {
    "moderntcn": {"n_stage": 2, "dim0": 32, "large_size": 13, "num_blocks": 1, "ffn_ratio": 2,
                  "downsample_ratio": 2, "patch_size": 1, "patch_stride": 1, "dropout": 0.1, "use_revin": True},
    "nhits": {"pool_preset": [2, 2, 1], "freq_preset": [4, 2, 1], "repr_dim": 64, "blocks_per_stack": 1,
              "mlp_hidden": 128, "dropout": 0.1},
}
ARCH_DEFAULT_TRAIN = {"lr": 2.0e-3, "weight_decay": 2.0e-4, "batch_size": 512, "window": DEFAULT_WINDOW}


# =====================================================================================================
# 6. Training loop (checklist-integrated: embargo split, EMA weights, warmup, label smoothing,
#    optional GCE/ELR/mixup) -- shared by isolation/hpsearch/final stages.
# =====================================================================================================

def _fit_one(arch: str, arch_params: dict[str, Any], train_params: dict[str, Any], *, seed: int,
             epochs: int, patience: int | None, use_gce: bool, use_elr: bool, use_mixup: bool,
             data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    _seed_everything(seed)
    window = int(train_params.get("window", DEFAULT_WINDOW))
    train_idx_all = _valid_indices((data["panel"]["timestamp"] >= TRAIN_START) & (data["panel"]["timestamp"] <= TRAIN_END), window, data["y_dir_full"], data["y_qual_full"])
    fit_idx, es_idx = _split_with_embargo(train_idx_all, window)
    raw_std, _ = _standardize_fit(data["raw"], fit_idx, window)
    ds_fit = WindowDataset(raw_std, window, fit_idx, data["y_dir_full"], data["y_qual_full"])
    ds_es = WindowDataset(raw_std, window, es_idx, data["y_dir_full"], data["y_qual_full"])
    dl_es = DataLoader(ds_es, batch_size=1024, shuffle=False)

    backbone = build_backbone(arch, len(SEQ_COLS), window, arch_params).to(device)
    model = TwoHeadClassifier(backbone, backbone.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_params.get("lr", 2e-3)), weight_decay=float(train_params.get("weight_decay", 2e-4)))
    ema = EMAWeights(model)
    # single reusable shadow-weight eval model (avoid reconstructing -- and re-randomizing init,
    # then immediately overwriting -- a fresh backbone every epoch just to hold the EMA copy).
    eval_model = TwoHeadClassifier(build_backbone(arch, len(SEQ_COLS), window, arch_params), backbone.hidden_dim).to(device)

    y_dir_fit = data["y_dir_full"][fit_idx].astype(np.int64)
    y_qual_fit = data["y_qual_full"][fit_idx].astype(np.int64)
    dir_w_all = compute_sample_weight("balanced", y_dir_fit).astype(np.float32)
    qual_w_all = compute_sample_weight("balanced", y_qual_fit).astype(np.float32)
    fit_pos = {idx: i for i, idx in enumerate(fit_idx)}

    batch_size = int(train_params.get("batch_size", 512))
    max_windows = min(len(fit_idx), MAX_WINDOWS_PER_EPOCH)
    steps_per_epoch = max(1, max_windows // batch_size)
    total_steps = steps_per_epoch * int(epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _warmup_lr_lambda(total_steps))

    n_fit = len(fit_idx)
    ema_dir_target = torch.full((n_fit, 3), 1.0 / 3.0, dtype=torch.float32, device=device)
    ema_qual_target = torch.full((n_fit, 3), 1.0 / 3.0, dtype=torch.float32, device=device)

    rng = np.random.default_rng(seed)
    best_es_loss, best_bacc, stale, last_epoch = float("inf"), -1.0, 0, 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        epoch_local_idx = rng.choice(np.arange(n_fit), size=max_windows, replace=False)
        loader = DataLoader(torch.utils.data.Subset(ds_fit, epoch_local_idx), batch_size=batch_size, shuffle=True)
        for xb, yb_dir, yb_qual, ridb in loader:
            xb, yb_dir, yb_qual, ridb = xb.to(device), yb_dir.to(device), yb_qual.to(device), ridb.to(device)
            # ridb is WindowDataset.__getitem__'s own `i` argument, i.e. already the local fit-array
            # position (0..n_fit-1) -- Subset(ds_fit, epoch_local_idx)[j] calls ds_fit[epoch_local_idx[j]],
            # so ridb IS epoch_local_idx[j] already; indexing epoch_local_idx by it again would be wrong.
            row_id = ridb
            wb_dir = torch.from_numpy(dir_w_all[ridb.cpu().numpy()]).to(device)
            wb_qual = torch.from_numpy(qual_w_all[ridb.cpu().numpy()]).to(device)

            h = model.encode(xb)
            logits_dir_u = model.direction_head(h)
            logits_qual_u = model.quality_head(h)
            if use_elr:
                with torch.no_grad():
                    pd_mean = torch.softmax(logits_dir_u, dim=-1).mean(dim=1)
                    pq_mean = torch.softmax(logits_qual_u, dim=-1).mean(dim=1)
                    ema_dir_target[row_id] = ELR_BETA * ema_dir_target[row_id] + (1.0 - ELR_BETA) * pd_mean
                    ema_qual_target[row_id] = ELR_BETA * ema_qual_target[row_id] + (1.0 - ELR_BETA) * pq_mean

            bsz = xb.shape[0]
            if use_mixup:
                perm = torch.randperm(bsz, device=device)
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                h_use = lam * h + (1.0 - lam) * h[perm]
            else:
                perm = torch.arange(bsz, device=device)
                lam = 1.0
                h_use = h

            logits_dir_m = model.direction_head(h_use)
            logits_qual_m = model.quality_head(h_use)
            yd_perm, yq_perm = yb_dir[perm], yb_qual[perm]
            wd_perm, wq_perm = wb_dir[perm], wb_qual[perm]
            cl_dir = lam * _cls_loss(logits_dir_m, yb_dir, use_gce=use_gce, n_classes=3) + (1.0 - lam) * _cls_loss(logits_dir_m, yd_perm, use_gce=use_gce, n_classes=3)
            cl_qual = lam * _cls_loss(logits_qual_m, yb_qual, use_gce=use_gce, n_classes=3) + (1.0 - lam) * _cls_loss(logits_qual_m, yq_perm, use_gce=use_gce, n_classes=3)
            w_dir_mix = lam * wb_dir + (1.0 - lam) * wd_perm
            w_qual_mix = lam * wb_qual + (1.0 - lam) * wq_perm
            loss_dir = (cl_dir.mean(dim=1) * w_dir_mix).sum() / torch.clamp(w_dir_mix.sum(), min=1.0)
            loss_qual = (cl_qual.mean(dim=1) * w_qual_mix).sum() / torch.clamp(w_qual_mix.sum(), min=1.0)

            loss_elr = torch.zeros((), device=device)
            if use_elr:
                pd_m = torch.softmax(logits_dir_m, dim=-1).mean(dim=1)
                pq_m = torch.softmax(logits_qual_m, dim=-1).mean(dim=1)
                t_dir_mix = lam * ema_dir_target[row_id] + (1.0 - lam) * ema_dir_target[row_id[perm]]
                t_qual_mix = lam * ema_qual_target[row_id] + (1.0 - lam) * ema_qual_target[row_id[perm]]
                loss_elr = ELR_LAMBDA * (((_elr_term(pd_m, t_dir_mix)) * w_dir_mix).sum() / torch.clamp(w_dir_mix.sum(), min=1.0)
                                          + ((_elr_term(pq_m, t_qual_mix)) * w_qual_mix).sum() / torch.clamp(w_qual_mix.sum(), min=1.0))

            loss = loss_dir + 0.80 * loss_qual + loss_elr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            scheduler.step()
            ema.update(model)

        # eval with EMA shadow weights (reuse the one eval_model instance, just refresh its weights)
        ema.copy_to(eval_model)
        eval_model.eval()
        with torch.no_grad():
            es_losses, es_bacc_preds, es_bacc_true = [], [], []
            for xb, yb_dir, yb_qual, _ in dl_es:
                xb, yb_dir, yb_qual = xb.to(device), yb_dir.to(device), yb_qual.to(device)
                out = eval_model(xb)
                ld = torch.nn.functional.cross_entropy(out["direction"].reshape(-1, 3), yb_dir[:, None].expand(-1, out["direction"].shape[1]).reshape(-1), reduction="none").reshape(-1, out["direction"].shape[1]).mean(dim=1)
                lq = torch.nn.functional.cross_entropy(out["quality"].reshape(-1, 3), yb_qual[:, None].expand(-1, out["quality"].shape[1]).reshape(-1), reduction="none").reshape(-1, out["quality"].shape[1]).mean(dim=1)
                es_losses.append((ld + 0.80 * lq).sum().item())
                es_bacc_preds.append(torch.softmax(out["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy())
                es_bacc_true.append(yb_dir.cpu().numpy())
            es_loss = sum(es_losses) / max(1, len(es_idx))
            bacc = balanced_accuracy_score(np.concatenate(es_bacc_true), np.concatenate(es_bacc_preds)) if es_bacc_true else float("nan")
        if es_loss < best_es_loss - 1e-5:
            best_es_loss, stale = es_loss, 0
        else:
            stale += 1
        best_bacc = max(best_bacc, bacc)
        if patience is not None and stale >= patience:
            break

    ema.copy_to(eval_model)
    return {"model": eval_model, "scaler_raw_std": raw_std, "window": window, "es_loss": best_es_loss,
            "es_bacc_peak": best_bacc, "epochs_ran": last_epoch}


@torch.no_grad()
def _predict(model: nn.Module, raw_std: np.ndarray, window: int, idx: np.ndarray, y_dir_full: np.ndarray, y_qual_full: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    ds = WindowDataset(raw_std, window, idx, y_dir_full, y_qual_full)
    loader = DataLoader(ds, batch_size=1024, shuffle=False)
    dir_preds, qual_preds = [], []
    for xb, _, _, _ in loader:
        out = model(xb.to(device))
        dir_preds.append(torch.softmax(out["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy())
        qual_preds.append(torch.softmax(out["quality"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy())
    return {"direction": np.concatenate(dir_preds), "quality": np.concatenate(qual_preds)}


def classification_report(idx: np.ndarray, preds: dict[str, np.ndarray], y_dir_full: np.ndarray, y_qual_full: np.ndarray) -> dict[str, Any]:
    y_dir_true = y_dir_full[idx].astype(np.int64)
    y_qual_true = y_qual_full[idx].astype(np.int64)
    return {
        "direction_balanced_accuracy": float(balanced_accuracy_score(y_dir_true, preds["direction"])),
        "direction_macro_f1": float(f1_score(y_dir_true, preds["direction"], average="macro")),
        "quality_balanced_accuracy": float(balanced_accuracy_score(y_qual_true, preds["quality"])),
        "quality_macro_f1": float(f1_score(y_qual_true, preds["quality"], average="macro")),
    }


# =====================================================================================================
# 7. Stage: sanity (local, tiny)
# =====================================================================================================

def stage_sanity(device: torch.device) -> None:
    log("=== stage=sanity (local, tiny smoke run, both architectures) ===")
    data = load_panel_and_labels()
    # subsample panel timeline for speed: keep only a few months around TRAIN_END for this check
    small_mask = (data["panel"]["timestamp"] >= pd.Timestamp("2025-06-01")) & (data["panel"]["timestamp"] <= pd.Timestamp("2025-09-30"))
    small_idx = _valid_indices(small_mask.to_numpy(), DEFAULT_WINDOW, data["y_dir_full"], data["y_qual_full"])
    log(f"  스모크 표본 {len(small_idx)}행 (2025-06~09)")
    global TRAIN_START, TRAIN_END
    saved = (TRAIN_START, TRAIN_END)
    TRAIN_START, TRAIN_END = pd.Timestamp("2025-06-01"), pd.Timestamp("2025-09-30 23:59:59")
    try:
        for arch in ("moderntcn", "nhits"):
            log(f"  --- arch={arch} ---")
            t0 = time.time()
            result = _fit_one(arch, ARCH_DEFAULT_PARAMS[arch], {**ARCH_DEFAULT_TRAIN, "batch_size": 128}, seed=1,
                               epochs=2, patience=None, use_gce=True, use_elr=True, use_mixup=True, data=data, device=device)
            preds = _predict(result["model"], result["scaler_raw_std"], result["window"], small_idx[-200:], data["y_dir_full"], data["y_qual_full"], device)
            rep = classification_report(small_idx[-200:], preds, data["y_dir_full"], data["y_qual_full"])
            assert np.isfinite(list(rep.values())).all(), f"{arch}: non-finite metrics -- NaN in training"
            log(f"  {arch} OK ({time.time()-t0:.0f}s) es_loss={result['es_loss']:.4f} sample_report={rep}")
    finally:
        TRAIN_START, TRAIN_END = saved
    log("=== stage=sanity PASSED for both architectures ===")


# =====================================================================================================
# 8. Stage: isolation (per architecture: none/gce_only/elr_only/mixup_only, N=5 seeds)
# =====================================================================================================

def stage_isolation(arch: str, device: torch.device, data: dict[str, Any]) -> dict[str, Any]:
    log(f"=== stage=isolation arch={arch} epochs={ISOLATION_EPOCHS} n_seeds={N_SEEDS_ISOLATION} ===")
    seeds = random.SystemRandom().sample(range(1_000_000, 999_000_000), N_SEEDS_ISOLATION)
    log(f"  시드(무작위): {seeds}")
    results: dict[str, Any] = {"seeds": seeds, "variants": {}}
    for variant in REGULARIZER_VARIANTS:
        curves = []
        for seed in seeds:
            t0 = time.time()
            r = _fit_one(arch, ARCH_DEFAULT_PARAMS[arch], ARCH_DEFAULT_TRAIN, seed=seed, epochs=ISOLATION_EPOCHS,
                         patience=None, use_gce=variant["use_gce"], use_elr=variant["use_elr"], use_mixup=variant["use_mixup"],
                         data=data, device=device)
            curves.append({"seed": seed, "es_loss": r["es_loss"], "es_bacc_peak": r["es_bacc_peak"]})
            log(f"  [{variant['name']}] seed={seed} es_loss={r['es_loss']:.4f} es_bacc_peak={r['es_bacc_peak']:.4f} ({time.time()-t0:.0f}s)")
        mean_bacc = float(np.mean([c["es_bacc_peak"] for c in curves]))
        results["variants"][variant["name"]] = {"curves": curves, "mean_es_bacc_peak": mean_bacc}
        log(f"  variant={variant['name']} mean_es_bacc_peak={mean_bacc:.4f}")
    winner = max(results["variants"].items(), key=lambda kv: kv[1]["mean_es_bacc_peak"])
    results["winner"] = winner[0]
    log(f"  === isolation winner for {arch}: {winner[0]} (mean_es_bacc_peak={winner[1]['mean_es_bacc_peak']:.4f}) ===")
    (OUT_DIR / f"isolation_{arch}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return results


# =====================================================================================================
# 9. Stage: hpsearch (Optuna TPE, regularizer fixed to isolation winner)
# =====================================================================================================

def stage_hpsearch(arch: str, device: torch.device, data: dict[str, Any], winner_variant: dict[str, bool]) -> dict[str, Any]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log(f"=== stage=hpsearch arch={arch} n_trials={N_TRIALS_HPSEARCH} regularizer={winner_variant} ===")

    def objective(trial: "optuna.Trial") -> float:
        window = trial.suggest_categorical("window", [48, 96, 192])
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        if arch == "moderntcn":
            params = {
                "n_stage": trial.suggest_int("n_stage", 1, 3),
                "dim0": trial.suggest_categorical("dim0", [16, 32, 64]),
                "large_size": trial.suggest_categorical("large_size", [9, 13, 21]),
                "num_blocks": trial.suggest_int("num_blocks", 1, 2),
                "ffn_ratio": trial.suggest_categorical("ffn_ratio", [1, 2, 4]),
                "downsample_ratio": 2, "patch_size": 1, "patch_stride": 1,
                "dropout": dropout, "use_revin": trial.suggest_categorical("use_revin", [True, False]),
            }
        else:
            preset = trial.suggest_categorical("pool_preset_idx", [0, 1, 2])
            presets = [([2, 2, 1], [4, 2, 1]), ([4, 2, 1], [8, 4, 1]), ([8, 4, 1], [8, 4, 1])]
            pk, fd = presets[preset]
            params = {
                "pool_preset": pk, "freq_preset": fd,
                "repr_dim": trial.suggest_categorical("repr_dim", [32, 64, 128]),
                "blocks_per_stack": trial.suggest_int("blocks_per_stack", 1, 2),
                "mlp_hidden": trial.suggest_categorical("mlp_hidden", [64, 128, 256]),
                "dropout": dropout,
            }
        train_params = {"lr": lr, "weight_decay": weight_decay, "batch_size": batch_size, "window": window}
        try:
            r = _fit_one(arch, params, train_params, seed=0, epochs=MAX_EPOCHS_TRIAL, patience=PATIENCE_TRIAL,
                         use_gce=winner_variant["use_gce"], use_elr=winner_variant["use_elr"], use_mixup=winner_variant["use_mixup"],
                         data=data, device=device)
        except RuntimeError as exc:
            log(f"  trial pruned (RuntimeError: {exc})")
            raise optuna.TrialPruned()
        trial.set_user_attr("params", params)
        trial.set_user_attr("train_params", train_params)
        return r["es_loss"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=20260816))
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS_HPSEARCH, show_progress_bar=False)
    log(f"  Optuna {N_TRIALS_HPSEARCH} trials 완료 ({time.time()-t0:.0f}s) best_es_loss={study.best_value:.4f}")
    study.trials_dataframe().to_csv(OUT_DIR / f"optuna_trials_{arch}.csv", index=False)

    trials_sorted = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)
    top = trials_sorted[:TOP_K_CANDIDATES]
    log(f"  상위 {len(top)}개 후보 VAL 재평가 (direction argmax PnL, always_short 대비 margin 최대 채택)...")
    cand_rows = []
    for rank, trial in enumerate(top):
        params, train_params = trial.user_attrs["params"], trial.user_attrs["train_params"]
        r = _fit_one(arch, params, train_params, seed=0, epochs=MAX_EPOCHS_TRIAL, patience=PATIENCE_TRIAL,
                     use_gce=winner_variant["use_gce"], use_elr=winner_variant["use_elr"], use_mixup=winner_variant["use_mixup"],
                     data=data, device=device)
        val_mask = (data["panel"]["timestamp"] >= VAL_START) & (data["panel"]["timestamp"] <= VAL_END)
        val_idx = _valid_indices(val_mask.to_numpy(), r["window"], data["y_dir_full"], data["y_qual_full"])
        preds = _predict(r["model"], r["scaler_raw_std"], r["window"], val_idx, data["y_dir_full"], data["y_qual_full"], device)
        pnl = pnl_vs_benchmarks(data["panel"], val_idx, preds["direction"])
        margin = pnl["cost3"]["model_pnl"] - pnl["cost3"]["always_short_pnl"]
        cand_rows.append({"rank": rank, "trial_number": trial.number, "es_loss": trial.value, "params": params,
                           "train_params": train_params, "val_margin_cost3": margin, "val_pnl_cost3": pnl["cost3"]["model_pnl"]})
        log(f"    trial#{trial.number} es_loss={trial.value:.4f} VAL_margin(cost3)={margin:+.2f}")
    winner = cand_rows[int(np.argmax([r["val_margin_cost3"] for r in cand_rows]))]
    log(f"  === hpsearch winner for {arch}: trial#{winner['trial_number']} margin={winner['val_margin_cost3']:+.2f} ===")
    (OUT_DIR / f"hpsearch_{arch}.json").write_text(json.dumps({"candidates": cand_rows, "winner": winner}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return winner


# =====================================================================================================
# 10. Stage: final (N>=5 genuinely random seeds at best HP + best regularizer)
# =====================================================================================================

def stage_final(arch: str, device: torch.device, data: dict[str, Any], best_params: dict[str, Any],
                 best_train_params: dict[str, Any], winner_variant: dict[str, bool]) -> dict[str, Any]:
    log(f"=== stage=final arch={arch} n_seeds={N_SEEDS_FINAL} params={best_params} train={best_train_params} regularizer={winner_variant} ===")
    seeds = random.SystemRandom().sample(range(1_000_000, 999_000_000), N_SEEDS_FINAL)
    log(f"  시드(무작위): {seeds}")
    window = int(best_train_params.get("window", DEFAULT_WINDOW))
    val_mask = (data["panel"]["timestamp"] >= VAL_START) & (data["panel"]["timestamp"] <= VAL_END)
    oos_mask = (data["panel"]["timestamp"] >= OOS_START) & (data["panel"]["timestamp"] <= OOS_END)
    val_idx = _valid_indices(val_mask.to_numpy(), window, data["y_dir_full"], data["y_qual_full"])
    oos_idx = _valid_indices(oos_mask.to_numpy(), window, data["y_dir_full"], data["y_qual_full"])

    per_seed = []
    for seed in seeds:
        t0 = time.time()
        r = _fit_one(arch, best_params, best_train_params, seed=seed, epochs=MAX_EPOCHS_FINAL, patience=PATIENCE_FINAL,
                     use_gce=winner_variant["use_gce"], use_elr=winner_variant["use_elr"], use_mixup=winner_variant["use_mixup"],
                     data=data, device=device)
        row: dict[str, Any] = {"seed": seed, "epochs_ran": r["epochs_ran"], "es_loss": r["es_loss"]}
        for split_name, idx in (("VAL", val_idx), ("OOS", oos_idx)):
            preds = _predict(r["model"], r["scaler_raw_std"], r["window"], idx, data["y_dir_full"], data["y_qual_full"], device)
            row[f"{split_name}_classification"] = classification_report(idx, preds, data["y_dir_full"], data["y_qual_full"])
            row[f"{split_name}_pnl"] = pnl_vs_benchmarks(data["panel"], idx, preds["direction"])
        per_seed.append(row)
        log(f"  seed={seed} done ({time.time()-t0:.0f}s) VAL_dir_bacc={row['VAL_classification']['direction_balanced_accuracy']:.4f} "
            f"OOS_dir_bacc={row['OOS_classification']['direction_balanced_accuracy']:.4f}")

    summary: dict[str, Any] = {"arch": arch, "seeds": seeds, "params": best_params, "train_params": best_train_params,
                                "regularizer": winner_variant, "per_seed": per_seed}
    for split_name in ("VAL", "OOS"):
        bacc = [row[f"{split_name}_classification"]["direction_balanced_accuracy"] for row in per_seed]
        qbacc = [row[f"{split_name}_classification"]["quality_balanced_accuracy"] for row in per_seed]
        beats_cost3 = [row[f"{split_name}_pnl"]["cost3"]["beats_always_short"] for row in per_seed]
        summary[f"{split_name}_direction_balanced_accuracy_mean"] = float(np.mean(bacc))
        summary[f"{split_name}_direction_balanced_accuracy_std"] = float(np.std(bacc))
        summary[f"{split_name}_quality_balanced_accuracy_mean"] = float(np.mean(qbacc))
        summary[f"{split_name}_beats_always_short_cost3_count"] = int(sum(beats_cost3))
        log(f"  {split_name}: dir_bacc={summary[f'{split_name}_direction_balanced_accuracy_mean']:.4f}"
            f"±{summary[f'{split_name}_direction_balanced_accuracy_std']:.4f}  "
            f"qual_bacc={summary[f'{split_name}_quality_balanced_accuracy_mean']:.4f}  "
            f"beats_always_short(cost3)={summary[f'{split_name}_beats_always_short_cost3_count']}/{N_SEEDS_FINAL}")
    (OUT_DIR / f"final_{arch}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


# =====================================================================================================
# 11. TabM control (same-conditions baseline, N=5 seeds, base checklist hygiene only, no GCE/ELR/mixup
#     search -- that axis is already separately researched for TabM by the two scripts cited above)
# =====================================================================================================

def stage_tabm_control(device: torch.device, data: dict[str, Any]) -> dict[str, Any]:
    log("=== stage=tabm_control (same-conditions comparison baseline, N=5 seeds, no_gce/no_elr/no_mixup) ===")
    seeds = random.SystemRandom().sample(range(1_000_000, 999_000_000), N_SEEDS_FINAL)
    log(f"  시드(무작위): {seeds}")

    def build_tabm(n_vars, window, params):
        return TabMControlBackbone(n_vars, k=8, hidden=192, layers=3, dropout=0.08)

    global build_backbone
    orig_build = build_backbone

    def _patched(arch, n_vars, window, params):
        if arch == "tabm_control":
            return build_tabm(n_vars, window, params)
        return orig_build(arch, n_vars, window, params)
    build_backbone = _patched

    val_mask = (data["panel"]["timestamp"] >= VAL_START) & (data["panel"]["timestamp"] <= VAL_END)
    oos_mask = (data["panel"]["timestamp"] >= OOS_START) & (data["panel"]["timestamp"] <= OOS_END)
    val_idx = _valid_indices(val_mask.to_numpy(), DEFAULT_WINDOW, data["y_dir_full"], data["y_qual_full"])
    oos_idx = _valid_indices(oos_mask.to_numpy(), DEFAULT_WINDOW, data["y_dir_full"], data["y_qual_full"])

    per_seed = []
    try:
        for seed in seeds:
            t0 = time.time()
            r = _fit_one("tabm_control", {}, ARCH_DEFAULT_TRAIN, seed=seed, epochs=MAX_EPOCHS_FINAL, patience=PATIENCE_FINAL,
                         use_gce=False, use_elr=False, use_mixup=False, data=data, device=device)
            row: dict[str, Any] = {"seed": seed, "epochs_ran": r["epochs_ran"]}
            for split_name, idx in (("VAL", val_idx), ("OOS", oos_idx)):
                preds = _predict(r["model"], r["scaler_raw_std"], r["window"], idx, data["y_dir_full"], data["y_qual_full"], device)
                row[f"{split_name}_classification"] = classification_report(idx, preds, data["y_dir_full"], data["y_qual_full"])
                row[f"{split_name}_pnl"] = pnl_vs_benchmarks(data["panel"], idx, preds["direction"])
            per_seed.append(row)
            log(f"  seed={seed} done ({time.time()-t0:.0f}s) VAL_dir_bacc={row['VAL_classification']['direction_balanced_accuracy']:.4f} "
                f"OOS_dir_bacc={row['OOS_classification']['direction_balanced_accuracy']:.4f}")
    finally:
        build_backbone = orig_build

    summary: dict[str, Any] = {"arch": "tabm_control", "seeds": seeds, "per_seed": per_seed}
    for split_name in ("VAL", "OOS"):
        bacc = [row[f"{split_name}_classification"]["direction_balanced_accuracy"] for row in per_seed]
        summary[f"{split_name}_direction_balanced_accuracy_mean"] = float(np.mean(bacc))
        summary[f"{split_name}_direction_balanced_accuracy_std"] = float(np.std(bacc))
    (OUT_DIR / "final_tabm_control.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


# =====================================================================================================
# main
# =====================================================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sanity", "isolation", "hpsearch", "final", "all", "tabm_control"], required=True)
    ap.add_argument("--arch", choices=["moderntcn", "nhits"], default=None)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    device = _device(args.device)
    log(f"device={device}")

    if args.stage == "sanity":
        stage_sanity(device)
        return 0

    data = load_panel_and_labels()

    if args.stage == "tabm_control":
        stage_tabm_control(device, data)
        return 0

    archs = [args.arch] if args.arch else ["moderntcn", "nhits"]
    for arch in archs:
        if args.stage in ("isolation", "all"):
            iso = stage_isolation(arch, device, data)
        else:
            iso_path = OUT_DIR / f"isolation_{arch}.json"
            assert iso_path.exists(), f"isolation report missing for {arch}: run --stage isolation first"
            iso = json.loads(iso_path.read_text())
        winner_name = iso["winner"]
        winner_variant = next(v for v in REGULARIZER_VARIANTS if v["name"] == winner_name)

        if args.stage in ("hpsearch", "all"):
            hp_winner = stage_hpsearch(arch, device, data, winner_variant)
        else:
            hp_path = OUT_DIR / f"hpsearch_{arch}.json"
            assert hp_path.exists(), f"hpsearch report missing for {arch}: run --stage hpsearch first"
            hp_winner = json.loads(hp_path.read_text())["winner"]

        if args.stage in ("final", "all"):
            stage_final(arch, device, data, hp_winner["params"], hp_winner["train_params"], winner_variant)

    if args.stage == "all":
        stage_tabm_control(device, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
