"""Lever A-2: TabM primary model ablation. Replaces the HGB primary classifier with a simplified
single-head TabM (Gorishniy et al. 2024, arXiv 2410.24210 -- parameter-efficient ensemble: K
"weak learner" heads share one MLP backbone but each gets its own per-layer affine
scale/bias, approximating a K-model deep ensemble at a fraction of the cost). Architecture is a
minimal port of this repo's existing `ThreeHeadTabM`
(scripts/train_eval_omega1_2_tabm_3head_20260603.py:87-122) stripped down to a single 3-class
direction head -- the 3-head version's exit/quality heads and Regime3-routing dependencies
(pos_* columns, hard/cat_dq imports) aren't part of this ablation's scope.

Same base triple-barrier label (scalp_action) and same val/OOS split/evaluation (fixed
confidence threshold=0.55, realistic maker-fill simulation) as the HGB baseline
(scalp_1m_tune_maker_realistic_20260716.json, OOS +3.74%) -- only the model architecture changes,
isolating the model-vs-HGB comparison from the label/execution levers already tested separately.

Output: data/ensemble/reports/scalp_1m_tabm_20260716.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
INTERNAL_VAL_DAYS = 30  # tail slice of train used only for early stopping, not the real val/OOS
FIXED_THRESHOLD = 0.55

K = 8
HIDDEN = 192
LAYERS = 3
DROPOUT = 0.08
BATCH_SIZE = 4096
LR = 2.0e-3
WEIGHT_DECAY = 2.0e-4
MAX_EPOCHS = 25
PATIENCE = 6

CLASS_ORDER = ['CASH', 'LONG', 'SHORT']


class TabM(nn.Module):
    def __init__(self, n_features: int, n_classes: int, k: int = K, hidden: int = HIDDEN,
                 layers: int = LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.k = k
        self.input_scale = nn.Parameter(torch.randn(k, n_features) * 0.03 + 1.0)
        self.input_bias = nn.Parameter(torch.zeros(k, n_features))
        self.in_proj = nn.Linear(n_features, hidden)
        self.blocks = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(max(0, layers - 1)))
        self.expert_scale = nn.ParameterList(
            nn.Parameter(torch.randn(k, hidden) * 0.03 + 1.0) for _ in range(max(0, layers - 1))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xk = x.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.in_proj(xk)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            h = layer(h * self.expert_scale[idx].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            h = h + residual
        logits_k = self.head(h)  # (batch, k, n_classes)
        return logits_k.mean(dim=1)  # ensemble-average logits


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

    print("Simulating maker-entry fills (long + short)...")
    long_sim = simulate_maker_fills(df, 'long')
    short_sim = simulate_maker_fills(df, 'short')

    feat_cols = feature_cols_for(df, [])
    train_all = df[df['timestamp'] <= TRAIN_END]
    val = df[(df['timestamp'] > TRAIN_END) & (df['timestamp'] <= VAL_END)]
    oos = df[(df['timestamp'] > VAL_END) & (df['timestamp'] <= OOS_END)]

    internal_val_start = pd.Timestamp(TRAIN_END) - pd.Timedelta(days=INTERNAL_VAL_DAYS)
    sub_train = train_all[train_all['timestamp'] <= internal_val_start]
    internal_val = train_all[train_all['timestamp'] > internal_val_start]
    print(f"SubTrain={len(sub_train):,} InternalVal={len(internal_val):,} "
          f"Val={len(val):,} OOS={len(oos):,}")

    class_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    X_train_raw = sub_train[feat_cols].fillna(0.0).to_numpy(dtype=np.float32)
    mean, std = standardize_fit(X_train_raw)
    X_train = (X_train_raw - mean) / std
    y_train = sub_train['scalp_action'].map(class_to_idx).to_numpy(dtype=np.int64)

    X_ival = ((internal_val[feat_cols].fillna(0.0).to_numpy(dtype=np.float32) - mean) / std)
    y_ival = internal_val['scalp_action'].map(class_to_idx).to_numpy(dtype=np.int64)

    class_counts = np.bincount(y_train, minlength=3)
    class_weight = (len(y_train) / (3.0 * np.maximum(class_counts, 1))).astype(np.float32)
    print(f"  class counts={class_counts.tolist()} weights={class_weight.tolist()}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = TabM(n_features=len(feat_cols), n_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, device=device))

    X_ival_t = torch.from_numpy(X_ival).to(device)
    y_ival_t = torch.from_numpy(y_ival).to(device)

    best_val_loss = float('inf')
    best_state = None
    patience_ctr = 0
    print("\nTraining TabM...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        train_loss = total_loss / len(train_ds)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_ival_t)
            val_loss = loss_fn(val_logits, y_ival_t).item()
            val_acc = (val_logits.argmax(dim=1) == y_ival_t).float().mean().item()
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

    def predict_proba(split_df: pd.DataFrame) -> np.ndarray:
        X = ((split_df[feat_cols].fillna(0.0).to_numpy(dtype=np.float32) - mean) / std)
        with torch.no_grad():
            logits = model(torch.from_numpy(X).to(device))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict_with_threshold(split_df: pd.DataFrame, threshold: float):
        proba = predict_proba(split_df)
        max_idx = proba.argmax(axis=1)
        max_proba = proba[np.arange(len(proba)), max_idx]
        classes = np.array(CLASS_ORDER)
        pred = classes[max_idx]
        return np.where(max_proba >= threshold, pred, 'CASH')

    print(f"\nEvaluating at fixed threshold={FIXED_THRESHOLD} (matching HGB baseline)...")
    result = {}
    for name, split_df in [('val', val), ('oos', oos)]:
        pred = predict_with_threshold(split_df, FIXED_THRESHOLD)
        idx = split_df.index
        long_sim_s, short_sim_s = long_sim.loc[idx].reset_index(drop=True), short_sim.loc[idx].reset_index(drop=True)
        bt = backtest_maker(pred, long_sim_s, short_sim_s)
        print(f"  [{name}] signals={bt['n_signals']:,} filled={bt['n_filled']:,} "
              f"fill_rate={bt['fill_rate']} hit_rate={bt['hit_rate']} total_pnl_pct={bt['total_pnl_pct']}")
        result[name] = bt

    result['model'] = 'TabM (K=8, hidden=192, layers=3)'
    result['fixed_threshold'] = FIXED_THRESHOLD
    result['best_internal_val_loss'] = best_val_loss
    result['baseline_for_comparison'] = {
        'report': 'scalp_1m_tune_maker_realistic_20260716.json',
        'model': 'HistGradientBoostingClassifier',
        'oos_total_pnl_pct': 3.7390646402123644,
    }
    result['compliance'] = {
        'fresh_forward_bar_by_bar': True, 'trade_ledgers_used_as_input': False,
        'saved_parent_exit_timestamps_used': False, 'future_rows_used_for_entry': False,
    }
    result['note'] = ('Same base triple-barrier label, val/OOS split, and maker-fill evaluation as '
                       'the HGB baseline -- only the classifier architecture changes.')
    with open(os.path.join(REPORT_DIR, 'scalp_1m_tabm_20260716.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("\nSaved scalp_1m_tabm_20260716.json")


if __name__ == '__main__':
    main()
