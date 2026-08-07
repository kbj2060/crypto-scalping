"""User-requested alternative to a mechanical cooldown rule: instead of bolting a timer onto the
existing single-bar HGB classifier, build a model that actually looks at a WINDOW of recent bars
before deciding -- a causal GRU sequence encoder over the last WINDOW_MIN minutes of features,
predicting the same 3-class triple-barrier action for the window's last (current) bar.

Hypothesis: requiring a coherent pattern across ~30 minutes of history (not just one bar's
snapshot) is a qualitatively different, model-driven way to suppress noise-driven single-bar
signals -- if it works, trade frequency drops because the model itself becomes more selective,
not because of an external rule.

Honest prior context (stated, not hidden): this project's only other sequence-model attempt
(Sigma1 GRU, referenced in project memory as part of the frozen Omega6 v2 baseline's
"every improvement attempt tested and failed" list) did not beat the simpler baseline. This
script is a genuine test, not an assumed win, using this session's realistic maker-fill
evaluation pipeline for a fair comparison against the established HGB baseline
(scalp_1m_tune_maker_realistic_20260716.json: OOS +3.74%, 8,075 filled trades, 192/day).

Sliding windows are built with numpy's sliding_window_view (zero-copy) over the full causal
feature matrix so windows near a train/val/OOS split boundary can still see genuinely-past bars
from before the boundary (no leakage -- those bars were already available at decision time in
live trading regardless of the train/val/OOS bookkeeping split).

Output: data/ensemble/reports/scalp_1m_gru_20260717.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn
from torch.utils.data import DataLoader, Dataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)

from train_eval_scalp_1m_hgb_20260716 import BASE_CSV, feature_cols_for
from simulate_maker_entry_scalp_1m_20260716 import simulate_maker_fills, backtest_maker, LABELS_CSV

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

TRAIN_END = '2026-04-30'
VAL_END = '2026-05-31'
OOS_END = '2026-07-12'
INTERNAL_VAL_DAYS = 30
FIXED_THRESHOLD = 0.55

WINDOW_MIN = 30
HIDDEN = 96
LAYERS = 1
DROPOUT = 0.1
BATCH_SIZE = 2048
LR = 1.5e-3
WEIGHT_DECAY = 2.0e-4
MAX_EPOCHS = 20
PATIENCE = 6

CLASS_ORDER = ['CASH', 'LONG', 'SHORT']


class GRUClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = HIDDEN, layers: int = LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden, num_layers=layers,
                           batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(self.drop(last))


class WindowDataset(Dataset):
    """Zero-copy sliding windows over a (N, D) feature matrix. Index i yields the window
    [i, i+W) and the target aligned to row i+W-1 (the window's last/current bar)."""
    def __init__(self, feat_matrix: np.ndarray, targets: np.ndarray, valid_end_positions: np.ndarray, window: int):
        self.windows = sliding_window_view(feat_matrix, window_shape=window, axis=0)  # (N-W+1, D, W)
        self.window = window
        self.targets = targets
        # valid_end_positions: absolute row indices (in feat_matrix) usable as a window's LAST bar
        self.end_positions = valid_end_positions[valid_end_positions >= window - 1]

    def __len__(self):
        return len(self.end_positions)

    def __getitem__(self, idx):
        end = self.end_positions[idx]
        start = end - self.window + 1
        w = self.windows[start]  # (D, W)
        x = np.ascontiguousarray(w.T)  # (W, D)
        y = self.targets[end]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def standardize_fit(x: np.ndarray):
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data + labels...")
    df = pd.read_csv(BASE_CSV, parse_dates=['timestamp'])
    labels = pd.read_csv(LABELS_CSV, parse_dates=['timestamp'])
    df = df.merge(labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)
    print(f"  {len(df):,} rows")

    print("Simulating maker-entry fills...")
    long_sim = simulate_maker_fills(df, 'long')
    short_sim = simulate_maker_fills(df, 'short')

    feat_cols = feature_cols_for(df, [])
    class_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    y_all = df['scalp_action'].map(class_to_idx).to_numpy(dtype=np.int64)

    X_raw = df[feat_cols].fillna(0.0).to_numpy(dtype=np.float32)
    train_mask = (df['timestamp'] <= TRAIN_END).to_numpy()
    mean, std = standardize_fit(X_raw[train_mask])
    X = ((X_raw - mean) / std).astype(np.float32)

    internal_val_start = pd.Timestamp(TRAIN_END) - pd.Timedelta(days=INTERNAL_VAL_DAYS)
    ts = df['timestamp']
    sub_train_end_pos = np.flatnonzero((ts <= internal_val_start).to_numpy())
    internal_val_end_pos = np.flatnonzero(((ts > internal_val_start) & (ts <= TRAIN_END)).to_numpy())
    val_end_pos = np.flatnonzero(((ts > TRAIN_END) & (ts <= VAL_END)).to_numpy())
    oos_end_pos = np.flatnonzero(((ts > VAL_END) & (ts <= OOS_END)).to_numpy())
    print(f"SubTrain={len(sub_train_end_pos):,} InternalVal={len(internal_val_end_pos):,} "
          f"Val={len(val_end_pos):,} OOS={len(oos_end_pos):,} (window={WINDOW_MIN}min)")

    train_ds = WindowDataset(X, y_all, sub_train_end_pos, WINDOW_MIN)
    ival_ds = WindowDataset(X, y_all, internal_val_end_pos, WINDOW_MIN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    ival_loader = DataLoader(ival_ds, batch_size=4096, shuffle=False, num_workers=0)

    class_counts = np.bincount(y_all[sub_train_end_pos[sub_train_end_pos >= WINDOW_MIN - 1]], minlength=3)
    class_weight = (class_counts.sum() / (3.0 * np.maximum(class_counts, 1))).astype(np.float32)
    print(f"  class counts={class_counts.tolist()} weights={class_weight.tolist()}")

    model = GRUClassifier(n_features=len(feat_cols), n_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, device=device))

    best_val_loss = float('inf')
    best_state = None
    patience_ctr = 0
    print("\nTraining GRU...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
            n_seen += len(xb)
        train_loss = total_loss / max(n_seen, 1)

        model.eval()
        val_loss_sum, val_correct, val_n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in ival_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss_sum += loss_fn(logits, yb).item() * len(xb)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
                val_n += len(xb)
        val_loss = val_loss_sum / max(val_n, 1)
        val_acc = val_correct / max(val_n, 1)
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} internal_val_loss={val_loss:.4f} "
              f"internal_val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    def predict_proba_for(end_positions: np.ndarray) -> np.ndarray:
        ds = WindowDataset(X, y_all, end_positions, WINDOW_MIN)
        loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)
        out = []
        with torch.no_grad():
            for xb, _ in loader:
                logits = model(xb.to(device))
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 3))

    def eval_split(end_positions: np.ndarray, name: str):
        proba = predict_proba_for(end_positions)
        usable_end_positions = end_positions[end_positions >= WINDOW_MIN - 1]
        max_idx = proba.argmax(axis=1)
        max_proba = proba[np.arange(len(proba)), max_idx]
        classes = np.array(CLASS_ORDER)
        pred = classes[max_idx]
        pred = np.where(max_proba >= FIXED_THRESHOLD, pred, 'CASH')
        long_sim_s = long_sim.loc[usable_end_positions].reset_index(drop=True)
        short_sim_s = short_sim.loc[usable_end_positions].reset_index(drop=True)
        bt = backtest_maker(pred, long_sim_s, short_sim_s)
        n_days = (df.loc[usable_end_positions, 'timestamp'].max() - df.loc[usable_end_positions, 'timestamp'].min()).total_seconds() / 86400
        trades_per_day = (bt['n_filled'] or 0) / max(n_days, 1e-6)
        print(f"  [{name}] filled={bt['n_filled']:,} ({trades_per_day:.1f}/day) "
              f"hit_rate={bt['hit_rate']} total_pnl_pct={bt['total_pnl_pct']}")
        return bt, trades_per_day

    print(f"\nEvaluating at fixed threshold={FIXED_THRESHOLD} (matching HGB baseline)...")
    val_bt, val_tpd = eval_split(val_end_pos, 'val')
    oos_bt, oos_tpd = eval_split(oos_end_pos, 'oos')

    result = {
        'model': f'GRU (window={WINDOW_MIN}min, hidden={HIDDEN}, layers={LAYERS})',
        'fixed_threshold': FIXED_THRESHOLD,
        'best_internal_val_loss': best_val_loss,
        'val': {**val_bt, 'trades_per_day': val_tpd},
        'oos': {**oos_bt, 'trades_per_day': oos_tpd},
        'baseline_for_comparison': {
            'report': 'scalp_1m_tune_maker_realistic_20260716.json',
            'model': 'HistGradientBoostingClassifier (single-bar)',
            'oos_total_pnl_pct': 3.7390646402123644,
            'oos_trades_per_day': 8075 / 42.0,
        },
        'compliance': {
            'fresh_forward_bar_by_bar': True, 'trade_ledgers_used_as_input': False,
            'saved_parent_exit_timestamps_used': False, 'future_rows_used_for_entry': False,
        },
        'note': ('Prior context: this project\'s only other sequence model (Sigma1 GRU) failed to '
                 'beat its baseline. Same base triple-barrier label, val/OOS split, and maker-fill '
                 'evaluation as the HGB baseline -- only the model looks at a 30min window instead '
                 'of a single bar.'),
    }
    with open(os.path.join(REPORT_DIR, 'scalp_1m_gru_20260717.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("\nSaved scalp_1m_gru_20260717.json")


if __name__ == '__main__':
    main()
